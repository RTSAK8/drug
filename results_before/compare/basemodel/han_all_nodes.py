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
from sklearn.model_selection import KFold
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


def prepare_data(data_dir):
    print("加载数据...")
    drug_features = load_features(os.path.join(data_dir, 'drug_embeddings.csv'), 'Drug_ID')
    gene_features = load_features(os.path.join(data_dir, 'gene_embeddings.csv'), 'Gene_ID')
    disease_features = load_features(os.path.join(data_dir, 'disease_embeddings.csv'), 'Disease_ID')

    triplets_df = pd.read_csv(os.path.join(data_dir, 'triplets.csv'))
    pos_triplets = []
    for _, row in triplets_df.iterrows():
        d, g, c = str(row['Drug_ID']), str(row['Gene_ID']), str(row['Disease_ID'])
        if d in drug_features and g in gene_features and c in disease_features:
            pos_triplets.append((d, g, c))

    if not pos_triplets:
        print("错误: 没有有效三元组！")
        return None

    all_drugs = list(drug_features.keys())
    all_genes = list(gene_features.keys())
    all_diseases = list(disease_features.keys())

    drug_to_idx = {drug: i for i, drug in enumerate(all_drugs)}
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    disease_to_idx = {dis: i for i, dis in enumerate(all_diseases)}

    return (pos_triplets, drug_features, gene_features, disease_features,
            drug_to_idx, gene_to_idx, disease_to_idx,
            all_drugs, all_genes, all_diseases)


# ====================== 2. 负采样 ======================
def generate_negatives(pos_triplets, drug_pool, gene_pool, disease_pool,
                       neg_ratio=3, replace_dims=("drug", "gene", "disease"),
                       max_attempts=50):
    pos_set = set(pos_triplets)
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
                if new_tri not in pos_set and new_tri not in neg_samples:
                    neg_samples.append(new_tri)
                    cnt += 1
    return neg_samples


# ====================== 3. 用 PrimeKG 关系建图 ======================
def build_primekg_graph(train_pos, val_pos, 
                        drug_features, gene_features, disease_features,
                        drug_to_idx, gene_to_idx, disease_to_idx,
                        ppi_file, dpi_file, ddi_file, protein_disease_file, device):
    """
    使用 PrimeKG 原始关系建图，并加入训练集三元组拆分的二元边（为了与超图模型公平对比）：
    - ppi.csv: protein-protein interactions (gene-gene) - type 0
    - drug_protein.csv: drug-protein interactions (drug-gene) - type 1
    - drug_drug.csv: drug-drug interactions - type 2
    - protein_disease.csv: protein-disease associations (gene-disease) - type 3
    - train_pos triplets: 拆分为 drug-gene (type 4), gene-disease (type 5), drug-disease (type 6)
    
    只包含训练集和验证集中出现的节点
    """
    num_drugs = len(drug_to_idx)
    num_genes = len(gene_to_idx)
    num_diseases = len(disease_to_idx)
    total_nodes = num_drugs + num_genes + num_diseases
    
    gene_offset = num_drugs
    disease_offset = num_drugs + num_genes
    
    # 收集训练和验证集中出现的所有节点ID（原始ID）
    train_drugs = set([t[0] for t in train_pos])
    train_genes = set([t[1] for t in train_pos])
    train_diseases = set([t[2] for t in train_pos])
    val_drugs = set([t[0] for t in val_pos])
    val_genes = set([t[1] for t in val_pos])
    val_diseases = set([t[2] for t in val_pos])
    
    all_drugs_in_fold = train_drugs | val_drugs
    all_genes_in_fold = train_genes | val_genes
    all_diseases_in_fold = train_diseases | val_diseases
    
    print(f"  本fold节点: {len(all_drugs_in_fold)} drugs, "
          f"{len(all_genes_in_fold)} genes, "
          f"{len(all_diseases_in_fold)} diseases")
    
    # 构建边列表
    edge_list = []
    edge_types = []
    
    # 3.1 加载 PPI (gene-gene)
    if os.path.exists(ppi_file):
        ppi_df = pd.read_csv(ppi_file)
        for _, row in ppi_df.iterrows():
            g1, g2 = str(row['Gene_ID_1']), str(row['Gene_ID_2'])
            if g1 in all_genes_in_fold and g2 in all_genes_in_fold:
                if g1 in gene_to_idx and g2 in gene_to_idx:
                    idx1 = gene_offset + gene_to_idx[g1]
                    idx2 = gene_offset + gene_to_idx[g2]
                    edge_list.extend([[idx1, idx2], [idx2, idx1]])
                    edge_types.extend([0, 0])
    
    # 3.2 加载 DPI (drug-gene)
    if os.path.exists(dpi_file):
        dpi_df = pd.read_csv(dpi_file)
        for _, row in dpi_df.iterrows():
            d, g = str(row['Drug_ID']), str(row['Gene_ID'])
            if d in all_drugs_in_fold and g in all_genes_in_fold:
                if d in drug_to_idx and g in gene_to_idx:
                    idx_d = drug_to_idx[d]
                    idx_g = gene_offset + gene_to_idx[g]
                    edge_list.extend([[idx_d, idx_g], [idx_g, idx_d]])
                    edge_types.extend([1, 1])
    
    # 3.3 加载 DDI (drug-drug)
    if os.path.exists(ddi_file):
        ddi_df = pd.read_csv(ddi_file)
        for _, row in ddi_df.iterrows():
            d1, d2 = str(row['Drug_ID_1']), str(row['Drug_ID_2'])
            if d1 in all_drugs_in_fold and d2 in all_drugs_in_fold:
                if d1 in drug_to_idx and d2 in drug_to_idx:
                    idx1 = drug_to_idx[d1]
                    idx2 = drug_to_idx[d2]
                    edge_list.extend([[idx1, idx2], [idx2, idx1]])
                    edge_types.extend([2, 2])
    
    # 3.4 加载 Protein-Disease (gene-disease)
    if os.path.exists(protein_disease_file):
        pd_df = pd.read_csv(protein_disease_file)
        for _, row in pd_df.iterrows():
            p, d = str(row['Gene_ID']), str(row['Disease_ID'])
            if p in all_genes_in_fold and d in all_diseases_in_fold:
                if p in gene_to_idx and d in disease_to_idx:
                    idx_p = gene_offset + gene_to_idx[p]
                    idx_d = disease_offset + disease_to_idx[d]
                    edge_list.extend([[idx_p, idx_d], [idx_d, idx_p]])
                    edge_types.extend([3, 3])
    
    # 3.5 【关键修改】加入训练集三元组拆分的二元边（与超图模型公平对比）
    # 超图模型将三元组作为3阶超边，这里将其拆分为3条二元边
    triplet_dg_edges = []  # drug-gene
    triplet_gc_edges = []  # gene-disease  
    triplet_dc_edges = []  # drug-disease
    
    for d, g, c in train_pos:
        if d in drug_to_idx and g in gene_to_idx and c in disease_to_idx:
            idx_d = drug_to_idx[d]
            idx_g = gene_offset + gene_to_idx[g]
            idx_c = disease_offset + disease_to_idx[c]
            
            # Drug-Gene 边（来自训练三元组）
            triplet_dg_edges.extend([[idx_d, idx_g], [idx_g, idx_d]])
            # Gene-Disease 边（来自训练三元组）
            triplet_gc_edges.extend([[idx_g, idx_c], [idx_c, idx_g]])
            # Drug-Disease 边（来自训练三元组，间接关系）
            triplet_dc_edges.extend([[idx_d, idx_c], [idx_c, idx_d]])
    
    # 添加三元组拆分的边（使用新的边类型 4,5,6）
    if triplet_dg_edges:
        edge_list.extend(triplet_dg_edges)
        edge_types.extend([4] * len(triplet_dg_edges))  # type 4: triplet drug-gene
        
    if triplet_gc_edges:
        edge_list.extend(triplet_gc_edges)
        edge_types.extend([5] * len(triplet_gc_edges))  # type 5: triplet gene-disease
        
    if triplet_dc_edges:
        edge_list.extend(triplet_dc_edges)
        edge_types.extend([6] * len(triplet_dc_edges))  # type 6: triplet drug-disease
    
    # 创建特征矩阵（全量节点）
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
    
    # 创建Data对象
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.zeros(0, dtype=torch.long)
    
    data = Data(
        x=x.to(device),
        edge_index=edge_index.to(device),
        edge_type=edge_type.to(device),
        num_nodes=total_nodes
    )
    
    # 统计信息（更新）
    n_ppi = sum(1 for t in edge_types if t == 0)
    n_dpi = sum(1 for t in edge_types if t == 1)
    n_ddi = sum(1 for t in edge_types if t == 2)
    n_pd = sum(1 for t in edge_types if t == 3)
    n_tri_dg = sum(1 for t in edge_types if t == 4)
    n_tri_gc = sum(1 for t in edge_types if t == 5)
    n_tri_dc = sum(1 for t in edge_types if t == 6)
    
    print(f"  PrimeKG边数: {edge_index.shape[1]} 条")
    print(f"    - PPI (gene-gene): {n_ppi} 条")
    print(f"    - DPI (drug-gene): {n_dpi} 条")
    print(f"    - DDI (drug-drug): {n_ddi} 条")
    print(f"    - Protein-Disease: {n_pd} 条")
    print(f"    - Triplet Drug-Gene: {n_tri_dg} 条 (来自训练集)")
    print(f"    - Triplet Gene-Disease: {n_tri_gc} 条 (来自训练集)")
    print(f"    - Triplet Drug-Disease: {n_tri_dc} 条 (来自训练集)")

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


# ====================== 6. 主实验 ======================
def run_experiment(data_dir='./dataset', n_splits=5, epochs=1000, lr=1e-3,
                   replace_dims=("gene",), neg_ratio=3):
    
    # PrimeKG 文件路径
    ppi_file = os.path.join(data_dir, 'ppi.csv')
    dpi_file = os.path.join(data_dir, 'drug_protein.csv')
    ddi_file = os.path.join(data_dir, 'drug_drug.csv')
    protein_disease_file = os.path.join(data_dir, 'protein_disease.csv')
    
    data_result = prepare_data(data_dir)
    if data_result is None:
        return
    (pos_triplets, drug_features, gene_features, disease_features,
     drug_to_idx, gene_to_idx, disease_to_idx,
     drug_pool, gene_pool, disease_pool) = data_result

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pos_array = np.array(pos_triplets)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(pos_array)):
        print(f"\nFold {fold+1}/{n_splits}")
        train_pos = [tuple(t) for t in pos_array[train_idx]]
        val_pos = [tuple(t) for t in pos_array[val_idx]]
        
        # 生成负样本
        train_neg = generate_negatives(train_pos, drug_pool, gene_pool, disease_pool,
                                      neg_ratio, replace_dims)
        val_neg = generate_negatives(val_pos, drug_pool, gene_pool, disease_pool,
                                    neg_ratio, replace_dims)

        train_triplets = train_pos + train_neg
        train_labels = [1] * len(train_pos) + [0] * len(train_neg)
        val_triplets = val_pos + val_neg
        val_labels = [1] * len(val_pos) + [0] * len(val_neg)

        # ========== 关键：用 PrimeKG 关系建图 ==========
        graph_data = build_primekg_graph(
            train_pos, val_pos,
            drug_features, gene_features, disease_features,
            drug_to_idx, gene_to_idx, disease_to_idx,
            ppi_file, dpi_file, ddi_file, protein_disease_file, device
        )
        
        print(f"  训练三元组: {len(train_triplets)} 个")
        print(f"  验证三元组: {len(val_triplets)} 个")

        # 初始化模型
        in_channels = graph_data['data'].x.shape[1]
        model = SimpleHAN(in_channels, hidden_channels=256, out_channels=128, 
                         num_heads=4, dropout=0.3).to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

        best_auc = 0
        for epoch in range(epochs):
            loss = train_epoch(model, graph_data, train_triplets, train_labels,
                               optimizer, criterion)
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                auc_score, aupr, f1_score, threshold, _ = evaluate(model, graph_data, 
                                                                   val_triplets, val_labels)
                print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss:.4f} | "
                      f"Val AUC: {auc_score:.4f} | AUPR: {aupr:.4f}")
                if auc_score > best_auc:
                    best_auc = auc_score
                    torch.save(model.state_dict(), f"best_fold{fold+1}.pth")
                scheduler.step(auc_score)

        # 最终评估
        model.load_state_dict(torch.load(f"best_fold{fold+1}.pth"))
        auc_score, aupr, f1_score, threshold, _ = evaluate(model, graph_data, 
                                                           val_triplets, val_labels)
        tm = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = replace_dims
        fold_results.append({'timestamp': tm, 'mode': mode, 'fold': fold + 1, 'auc': auc_score, 'aupr': aupr, 
                            'f1_score': f1_score, 'threshold': threshold})
        os.remove(f"best_fold{fold+1}.pth")
        torch.cuda.empty_cache()

    # 汇总并保存结果（修改部分）
    df = pd.DataFrame(fold_results)
    
    # 计算统计指标
    mean_auc = df['auc'].mean()
    std_auc = df['auc'].std()
    mean_aupr = df['aupr'].mean()
    std_aupr = df['aupr'].std()
    mean_f1 = df['f1_score'].mean()
    std_f1 = df['f1_score'].std()
    
    print(f"\n{'='*50}")
    print(f"平均 AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"平均 AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")
    print(f"平均 F1-score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # 添加统计汇总行到DataFrame
    summary_rows = [
        {'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"), 'mode': replace_dims, 'fold': 'MEAN', 
         'auc': mean_auc, 'aupr': mean_aupr, 'f1_score': mean_f1, 'threshold': ''},
        {'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"), 'mode': replace_dims, 'fold': 'STD', 
         'auc': std_auc, 'aupr': std_aupr, 'f1_score': std_f1, 'threshold': ''}
    ]
    
    # 合并原始结果和统计行
    df_with_summary = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    
    # 保存到CSV
    os.makedirs('./results', exist_ok=True)
    file_path = f'./results/han_all_nodes_summary_new.csv'
    if os.path.exists(file_path):
        df_with_summary.to_csv(file_path, mode='a', header=False, index=False)
        print(f'结果已追加到 {file_path}')
    else:
        df_with_summary.to_csv(file_path, mode='w', index=False)
        print(f'结果已保存到 {file_path}')
    
    return fold_results


# ====================== 7. 运行 ======================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    replace_dim = ['gene', 'disease']
    for dim in replace_dim:
        print(f'\n=== 以 {dim} 替换进行实验 ===')
        run_experiment(replace_dims=(dim,))  # 改为 (dim,) 加个逗号变成单元素元组