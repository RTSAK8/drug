"""
基于 PrimeKG 的 HAN 三元组预测模型
使用 PrimeKG 原始关系建图，三元组作为预测目标
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import random
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ====================== 1. 数据加载 ======================
def load_features(filepath, id_col='Drug_ID'):
    df = pd.read_csv(filepath)
    features_dict = {}
    for _, row in df.iterrows():
        entity_id = str(row[id_col])
        if '.' in entity_id and entity_id.endswith('.0'):
            entity_id_int = entity_id[:-2]
            features_dict[entity_id] = row.drop(id_col).values.astype(np.float32)
            features_dict[entity_id_int] = row.drop(id_col).values.astype(np.float32)
        else:
            features_dict[entity_id] = row.drop(id_col).values.astype(np.float32)
    return features_dict


def prepare_data(train_dir, test_dir):
    print("加载数据...")
    drug_features = load_features(os.path.join(train_dir, 'drug_embeddings.csv'), 'Drug_ID')
    gene_features = load_features(os.path.join(train_dir, 'gene_embeddings.csv'), 'Gene_ID')
    disease_features = load_features(os.path.join(train_dir, 'disease_embeddings.csv'), 'Disease_ID')

    # 训练三元组
    train_triplets_df = pd.read_csv(os.path.join(train_dir, 'triplets.csv'))
    train_pos = []
    for _, row in train_triplets_df.iterrows():
        d, g, c = str(row['Drug_ID']), str(row['Gene_ID']), str(row['Disease_ID'])
        if d in drug_features and g in gene_features and c in disease_features:
            train_pos.append((d, g, c))

    if not train_pos:
        print("错误: 没有有效训练三元组！")
        return None

    # 测试三元组
    test_triplets_df = pd.read_csv(os.path.join(test_dir, 'drugbank_triplets.csv'))
    test_pos = []
    for _, row in test_triplets_df.iterrows():
        d, g, c = str(row['Drug_ID']), str(row['Gene_ID']), str(row['Disease_ID'])
        if d in drug_features and g in gene_features and c in disease_features:
            test_pos.append((d, g, c))

    if not test_pos:
        print("错误: 没有有效测试三元组！")
        return None

    all_drugs = list(drug_features.keys())
    all_genes = list(gene_features.keys())
    all_diseases = list(disease_features.keys())

    drug_to_idx = {drug: i for i, drug in enumerate(all_drugs)}
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    disease_to_idx = {dis: i for i, dis in enumerate(all_diseases)}

    return (train_pos, test_pos, drug_features, gene_features, disease_features,
            drug_to_idx, gene_to_idx, disease_to_idx,
            all_drugs, all_genes, all_diseases)


# ====================== 2. 负采样 ======================
def generate_negatives(pos_triplets, drug_pool, gene_pool, disease_pool,
                       neg_ratio=3, replace_dims=("drug", "gene", "disease"),
                       max_attempts=50, global_exclude=None):
    pos_set = set(pos_triplets)
    if global_exclude is None:
        global_exclude = set()
    local_exclude = set(pos_set)
    neg_samples = []
    for d, g, c in pos_triplets:
        for dim in replace_dims:
            pool = {"drug": drug_pool, "gene": gene_pool, "disease": disease_pool}[dim]
            if len(pool) <= 1:
                continue
            cnt = 0
            attempts = 0
            while cnt < neg_ratio and attempts < max_attempts * neg_ratio:
                attempts += 1
                new_id = random.choice(pool)
                if dim == "drug" and new_id == d: continue
                if dim == "gene" and new_id == g: continue
                if dim == "disease" and new_id == c: continue
                new_tri = (new_id, g, c) if dim == "drug" else \
                          (d, new_id, c) if dim == "gene" else (d, g, new_id)
                if new_tri not in local_exclude and new_tri not in global_exclude:
                    neg_samples.append(new_tri)
                    local_exclude.add(new_tri)
                    cnt += 1
    global_exclude.update(local_exclude)
    return neg_samples


# ====================== 3. 用 PrimeKG 关系建图 ======================
def build_primekg_graph(train_pos, val_pos, 
                        drug_features, gene_features, disease_features,
                        drug_to_idx, gene_to_idx, disease_to_idx,
                        ppi_file, dpi_file, ddi_file, device):
    num_drugs = len(drug_to_idx)
    num_genes = len(gene_to_idx)
    num_diseases = len(disease_to_idx)
    total_nodes = num_drugs + num_genes + num_diseases
    
    gene_offset = num_drugs
    disease_offset = num_drugs + num_genes
    
    # 收集训练集中的所有节点
    train_drugs = set([t[0] for t in train_pos])
    train_genes = set([t[1] for t in train_pos])
    train_diseases = set([t[2] for t in train_pos])
    
    # 构建边列表
    edge_list = []
    edge_types = []
    
    # 加载 PPI
    if os.path.exists(ppi_file):
        ppi_df = pd.read_csv(ppi_file)
        for _, row in ppi_df.iterrows():
            g1, g2 = str(row['Gene_ID_1']), str(row['Gene_ID_2'])
            if g1 in train_genes and g2 in train_genes:
                if g1 in gene_to_idx and g2 in gene_to_idx:
                    idx1 = gene_offset + gene_to_idx[g1]
                    idx2 = gene_offset + gene_to_idx[g2]
                    edge_list.extend([[idx1, idx2], [idx2, idx1]])
                    edge_types.extend([0, 0])
    
    # 加载 DPI
    if os.path.exists(dpi_file):
        dpi_df = pd.read_csv(dpi_file)
        for _, row in dpi_df.iterrows():
            d, g = str(row['Drug_ID']), str(row['Gene_ID'])
            if d in train_drugs and g in train_genes:
                if d in drug_to_idx and g in gene_to_idx:
                    idx_d = drug_to_idx[d]
                    idx_g = gene_offset + gene_to_idx[g]
                    edge_list.extend([[idx_d, idx_g], [idx_g, idx_d]])
                    edge_types.extend([1, 1])
    
    # 加载 DDI
    if os.path.exists(ddi_file):
        ddi_df = pd.read_csv(ddi_file)
        for _, row in ddi_df.iterrows():
            d1, d2 = str(row['Drug_ID_1']), str(row['Drug_ID_2'])
            if d1 in train_drugs and d2 in train_drugs:
                if d1 in drug_to_idx and d2 in drug_to_idx:
                    idx1 = drug_to_idx[d1]
                    idx2 = drug_to_idx[d2]
                    edge_list.extend([[idx1, idx2], [idx2, idx1]])
                    edge_types.extend([2, 2])
    
    # 创建特征矩阵
    drug_feat_dim = len(list(drug_features.values())[0])
    gene_feat_dim = len(list(gene_features.values())[0])
    disease_feat_dim = len(list(disease_features.values())[0])
    max_feat_dim = max(drug_feat_dim, gene_feat_dim, disease_feat_dim)
    
    x = torch.zeros((total_nodes, max_feat_dim), dtype=torch.float)
    
    for drug_id, idx in drug_to_idx.items():
        feat = torch.FloatTensor(drug_features[drug_id])
        x[idx, :len(feat)] = feat
    
    for gene_id, idx in gene_to_idx.items():
        feat = torch.FloatTensor(gene_features[gene_id])
        x[gene_offset + idx, :len(feat)] = feat
    
    for disease_id, idx in disease_to_idx.items():
        feat = torch.FloatTensor(disease_features[disease_id])
        x[disease_offset + idx, :len(feat)] = feat
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.zeros(0, dtype=torch.long)
    
    data = Data(
        x=x.to(device),
        edge_index=edge_index.to(device),
        edge_type=edge_type.to(device),
        num_nodes=total_nodes
    )

    return {
        'data': data,
        'num_drugs': num_drugs,
        'num_genes': num_genes,
        'num_diseases': num_diseases,
        'gene_offset': gene_offset,
        'disease_offset': disease_offset,
        'drug_to_idx': drug_to_idx,
        'gene_to_idx': gene_to_idx,
        'disease_to_idx': disease_to_idx
    }


# ====================== 4. 模型 ======================
class SimpleHAN(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=128, 
                 num_heads=4, dropout=0.3):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                            dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1,
                            dropout=dropout, concat=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.decoder = nn.Sequential(
            nn.Linear(3 * out_channels, hidden_channels),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, data, drug_idx=None, gene_idx=None, disease_idx=None):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.dropout(F.elu(x))
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        if drug_idx is not None:
            tri = torch.cat([x[drug_idx], x[gene_idx], x[disease_idx]], dim=1)
            return torch.sigmoid(self.decoder(tri)).squeeze()
        
        return x


# ====================== 5. 训练/评估 ======================
def train_epoch(model, graph_data, train_triplets, train_labels, optimizer, criterion):
    model.train()
    data = graph_data['data']
    
    drug_idx = []
    gene_idx = []
    dis_idx = []
    for d, g, c in train_triplets:
        drug_idx.append(graph_data['drug_to_idx'][d])
        gene_idx.append(graph_data['gene_offset'] + graph_data['gene_to_idx'][g])
        dis_idx.append(graph_data['disease_offset'] + graph_data['disease_to_idx'][c])
        
    drug_idx = torch.tensor(drug_idx, dtype=torch.long).to(device)
    gene_idx = torch.tensor(gene_idx, dtype=torch.long).to(device)
    dis_idx = torch.tensor(dis_idx, dtype=torch.long).to(device)
    labels = torch.tensor(train_labels, dtype=torch.float).to(device)

    optimizer.zero_grad()
    preds = model(data, drug_idx, gene_idx, dis_idx)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, graph_data, val_triplets, val_labels):
    model.eval()
    data = graph_data['data']
    
    drug_idx = []
    gene_idx = []
    dis_idx = []
    for d, g, c in val_triplets:
        drug_idx.append(graph_data['drug_to_idx'][d])
        gene_idx.append(graph_data['gene_offset'] + graph_data['gene_to_idx'][g])
        dis_idx.append(graph_data['disease_offset'] + graph_data['disease_to_idx'][c])
        
    drug_idx = torch.tensor(drug_idx, dtype=torch.long).to(device)
    gene_idx = torch.tensor(gene_idx, dtype=torch.long).to(device)
    dis_idx = torch.tensor(dis_idx, dtype=torch.long).to(device)

    with torch.no_grad():
        preds = model(data, drug_idx, gene_idx, dis_idx).cpu().numpy()
    
    auc_score = roc_auc_score(val_labels, preds)
    precision, recall, thresholds = precision_recall_curve(val_labels, preds)
    aupr = auc(recall, precision)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = f1_scores[:-1]
    
    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_threshold = float(thresholds[best_idx])
    else:
        best_f1 = 0.0
        best_threshold = 0.5
    
    return auc_score, aupr, best_f1, best_threshold, preds


def predict_scores(model, graph_data, triplets):
    data = graph_data['data']
    drug_idx = []
    gene_idx = []
    dis_idx = []
    for d, g, c in triplets:
        drug_idx.append(graph_data['drug_to_idx'][d])
        gene_idx.append(graph_data['gene_offset'] + graph_data['gene_to_idx'][g])
        dis_idx.append(graph_data['disease_offset'] + graph_data['disease_to_idx'][c])

    drug_idx = torch.tensor(drug_idx, dtype=torch.long).to(device)
    gene_idx = torch.tensor(gene_idx, dtype=torch.long).to(device)
    dis_idx = torch.tensor(dis_idx, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        return model(data, drug_idx, gene_idx, dis_idx).cpu().numpy()


def ranking_metrics_full_candidates(model, graph_data, pos_triplets, replace_dim='disease'):
    ranks = []

    drug_pool = sorted(set(graph_data['drug_to_idx'].keys()))
    gene_pool = sorted(set(graph_data['gene_to_idx'].keys()))
    disease_pool = sorted(set(graph_data['disease_to_idx'].keys()))

    for d, g, c in pos_triplets:
        if replace_dim == 'drug':
            candidates = drug_pool
            candidate_triplets = [(cand, g, c) for cand in candidates]
            target = d
        elif replace_dim == 'gene':
            candidates = gene_pool
            candidate_triplets = [(d, cand, c) for cand in candidates]
            target = g
        else:
            candidates = disease_pool
            candidate_triplets = [(d, g, cand) for cand in candidates]
            target = c

        if len(candidates) < 2 or target not in candidates:
            continue

        scores = predict_scores(model, graph_data, candidate_triplets)
        target_idx = candidates.index(target)
        sorted_idx = np.argsort(scores)[::-1]
        rank = int(np.where(sorted_idx == target_idx)[0][0]) + 1
        ranks.append(rank)

    if not ranks:
        return 0.0, {'hit@1': 0.0, 'hit@3': 0.0, 'hit@5': 0.0}

    ranks = np.array(ranks)
    mrr = np.mean(1 / ranks)
    hits = {f'hit@{k}': np.mean(ranks <= k) for k in [1, 3, 5]}
    return mrr, hits


# ====================== 6. 主实验 ======================
def run_experiment(train_dir='./dataset', test_dir='./test_dataset', 
                   epochs=100, lr=1e-3, replace_dims=("gene",), neg_ratio=3):
    
    # PrimeKG 文件路径
    ppi_file = os.path.join(train_dir, 'ppi.csv')
    dpi_file = os.path.join(train_dir, 'drug_protein.csv')
    ddi_file = os.path.join(train_dir, 'drug_drug.csv')
    
    data_result = prepare_data(train_dir, test_dir)
    if data_result is None:
        return
    (train_pos, test_pos, drug_features, gene_features, disease_features,
     drug_to_idx, gene_to_idx, disease_to_idx,
     drug_pool, gene_pool, disease_pool) = data_result

    print(f"训练三元组: {len(train_pos)}")
    print(f"测试三元组: {len(test_pos)}")

    train_drug_pool = sorted({d for d, _, _ in train_pos})
    train_gene_pool = sorted({g for _, g, _ in train_pos})
    train_disease_pool = sorted({c for _, _, c in train_pos})
    global_pos_set = set(train_pos) | set(test_pos)
    
    # 生成训练集负样本
    train_neg = generate_negatives(
        train_pos,
        train_drug_pool,
        train_gene_pool,
        train_disease_pool,
        neg_ratio,
        replace_dims,
        global_exclude=global_pos_set.copy(),
    )
    # 生成测试集负样本
    test_neg = generate_negatives(
        test_pos,
        train_drug_pool,
        train_gene_pool,
        train_disease_pool,
        neg_ratio,
        replace_dims,
        global_exclude=global_pos_set.copy(),
    )

    train_triplets = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    test_triplets = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    # 用 PrimeKG 关系建图（只用训练集）
    graph_data = build_primekg_graph(
        train_pos, test_pos,
        drug_features, gene_features, disease_features,
        drug_to_idx, gene_to_idx, disease_to_idx,
        ppi_file, dpi_file, ddi_file, device
    )

    # 初始化模型
    in_channels = graph_data['data'].x.shape[1]
    model = SimpleHAN(in_channels, hidden_channels=256, out_channels=128, 
                     num_heads=4, dropout=0.3).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    for epoch in range(epochs):
        loss = train_epoch(model, graph_data, train_triplets, train_labels,
                           optimizer, criterion)
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            # 只在训练集上监控训练进度
            train_auc, train_aupr, train_f1, train_threshold, _ = evaluate(model, graph_data, 
                                                                          train_triplets, train_labels)
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss:.4f} | "
                  f"Train AUC: {train_auc:.4f} | AUPR: {train_aupr:.4f}")
            scheduler.step(train_auc)

    # 保存最后一个epoch的模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./results/han_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    # 在测试集上进行最终评估
    test_auc, test_aupr, test_f1, test_threshold, test_pred = evaluate(model, graph_data, 
                                                                       test_triplets, test_labels)
    replace_dim = replace_dims[0] if isinstance(replace_dims, (tuple, list)) else replace_dims
    test_mrr, test_hits = ranking_metrics_full_candidates(model, graph_data, test_pos, replace_dim)
    
    result = {
        'timestamp': timestamp,
        'mode': replace_dim,
        'auc': test_auc,
        'aupr': test_aupr, 
        'f1_score': test_f1,
        'threshold': test_threshold,
        'mrr': test_mrr,
        **test_hits,
    }
    
    os.makedirs('./results', exist_ok=True)
    file_path = f'./results/han_independent_test.csv'
    df = pd.DataFrame([result])
    
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
        print(f'结果已追加到 {file_path}')
    else:
        df.to_csv(file_path, mode='w', index=False)
        print(f'结果已保存到 {file_path}')
    
    print(f"\n最终测试结果:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  AUPR: {test_aupr:.4f}")
    print(f"  F1-score: {test_f1:.4f} @ threshold {test_threshold:.4f}")
    print(f"  MRR: {test_mrr:.4f}")
    print(f"  hit@1: {test_hits['hit@1']:.4f}")
    print(f"  hit@3: {test_hits['hit@3']:.4f}")
    print(f"  hit@5: {test_hits['hit@5']:.4f}")
    
    return result


# ====================== 7. 运行 ======================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 在triplets.csv上训练，在drugbank_triplets.csv上测试
    run_experiment(
        train_dir='./dataset',  # 训练数据目录
        test_dir='./dataset',  # 测试数据目录
        epochs=100,
        lr=1e-3,
        replace_dims=("gene",),  # 替换维度
        neg_ratio=3
    )
    run_experiment(
        train_dir='./dataset',  # 训练数据目录
        test_dir='./dataset',  # 测试数据目录
        epochs=100,
        lr=1e-3,
        replace_dims=("disease",),  # 替换维度
        neg_ratio=3
    )