"""
基于 PrimeKG 的 RGCN 基线模型 - 与超图模型公平对比版本
包含 Protein-Disease 关系，将三元组分解为3条二元边
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
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
            feat = row.drop(id_col).values.astype(np.float32)
            features_dict[entity_id] = feat
            features_dict[entity_id_int] = feat
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

    print(f"实体统计: {len(all_drugs)} Drugs, {len(all_genes)} Genes, {len(all_diseases)} Diseases")
    print(f"正样本三元组: {len(pos_triplets)}")

    return (pos_triplets, drug_features, gene_features, disease_features,
            drug_to_idx, gene_to_idx, disease_to_idx,
            all_drugs, all_genes, all_diseases)


# ====================== 2. 负采样 ======================
def generate_negatives(pos_triplets, drug_pool, gene_pool, disease_pool,
                       neg_ratio=3, replace_dims=("drug", "gene", "disease"),
                       max_attempts=50):
    if isinstance(replace_dims, str):
        replace_dims = (replace_dims,)
    
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


# ====================== 3. 建图（公平对比版 + Protein-Disease）======================
def build_fair_comparison_graph(train_pos, val_pos, drug_features, gene_features, disease_features,
                                drug_to_idx, gene_to_idx, disease_to_idx,
                                ppi_file, dpi_file, ddi_file, protein_disease_file, device):
    """
    构建与HGNN公平对比的RGCN图：
    PrimeKG关系：
    - Type 0: PPI (gene-gene)
    - Type 1: DPI (drug-gene) 
    - Type 2: DDI (drug-drug)
    - Type 3: Protein-Disease (gene-disease)
    训练三元组分解（模拟超边）：
    - Type 4: Triplet-Drug-Gene
    - Type 5: Triplet-Drug-Disease
    - Type 6: Triplet-Gene-Disease
    """
    num_drugs = len(drug_to_idx)
    num_genes = len(gene_to_idx)
    num_diseases = len(disease_to_idx)
    total_nodes = num_drugs + num_genes + num_diseases
    
    gene_offset = num_drugs
    disease_offset = num_drugs + num_genes
    
    # 收集本fold的所有节点
    all_drugs = set([t[0] for t in train_pos]) | set([t[0] for t in val_pos])
    all_genes = set([t[1] for t in train_pos]) | set([t[1] for t in val_pos])
    all_diseases = set([t[2] for t in train_pos]) | set([t[2] for t in val_pos])
    
    edge_list = []
    edge_types = []
    
    # Type 0: PPI (gene-gene)
    ppi_edges = 0
    if os.path.exists(ppi_file):
        df = pd.read_csv(ppi_file)
        for _, row in df.iterrows():
            g1, g2 = str(row['Gene_ID_1']), str(row['Gene_ID_2'])
            if g1 in all_genes and g2 in all_genes and g1 in gene_to_idx and g2 in gene_to_idx:
                i1 = gene_offset + gene_to_idx[g1]
                i2 = gene_offset + gene_to_idx[g2]
                edge_list.extend([[i1, i2], [i2, i1]])
                edge_types.extend([0, 0])
                ppi_edges += 2
    
    # Type 1: DPI (drug-gene)
    dpi_edges = 0
    if os.path.exists(dpi_file):
        df = pd.read_csv(dpi_file)
        for _, row in df.iterrows():
            d, g = str(row['Drug_ID']), str(row['Gene_ID'])
            if d in all_drugs and g in all_genes and d in drug_to_idx and g in gene_to_idx:
                i_d = drug_to_idx[d]
                i_g = gene_offset + gene_to_idx[g]
                edge_list.extend([[i_d, i_g], [i_g, i_d]])
                edge_types.extend([1, 1])
                dpi_edges += 2
    
    # Type 2: DDI (drug-drug)
    ddi_edges = 0
    if os.path.exists(ddi_file):
        df = pd.read_csv(ddi_file)
        for _, row in df.iterrows():
            d1, d2 = str(row['Drug_ID_1']), str(row['Drug_ID_2'])
            if d1 in all_drugs and d2 in all_drugs and d1 in drug_to_idx and d2 in drug_to_idx:
                i1 = drug_to_idx[d1]
                i2 = drug_to_idx[d2]
                edge_list.extend([[i1, i2], [i2, i1]])
                edge_types.extend([2, 2])
                ddi_edges += 2
    
    # Type 3: Protein-Disease (gene-disease)
    pd_edges = 0
    if os.path.exists(protein_disease_file):
        df = pd.read_csv(protein_disease_file)
        for _, row in df.iterrows():
            p, d = str(row['Gene_ID']), str(row['Disease_ID'])
            if p in all_genes and d in all_diseases and p in gene_to_idx and d in disease_to_idx:
                i_p = gene_offset + gene_to_idx[p]
                i_d = disease_offset + disease_to_idx[d]
                edge_list.extend([[i_p, i_d], [i_d, i_p]])
                edge_types.extend([3, 3])
                pd_edges += 2
    
    # Type 4-6: 训练三元组分解为3条边（模拟超边）
    triplet_dg = 0  # Drug-Gene (Type 4)
    triplet_dc = 0  # Drug-Disease (Type 5)
    triplet_gc = 0  # Gene-Disease (Type 6)
    
    for d, g, c in train_pos:
        if d in drug_to_idx and g in gene_to_idx and c in disease_to_idx:
            i_d = drug_to_idx[d]
            i_g = gene_offset + gene_to_idx[g]
            i_c = disease_offset + disease_to_idx[c]
            
            edge_list.extend([[i_d, i_g], [i_g, i_d]])
            edge_types.extend([4, 4])
            triplet_dg += 2
            
            edge_list.extend([[i_d, i_c], [i_c, i_d]])
            edge_types.extend([5, 5])
            triplet_dc += 2
            
            edge_list.extend([[i_g, i_c], [i_c, i_g]])
            edge_types.extend([6, 6])
            triplet_gc += 2
    
    print(f"\n图构建统计:")
    print(f"  PrimeKG - PPI: {ppi_edges}, DPI: {dpi_edges}, DDI: {ddi_edges}, Protein-Disease: {pd_edges}")
    print(f"  Triplet - Drug-Gene: {triplet_dg}, Drug-Disease: {triplet_dc}, Gene-Disease: {triplet_gc}")
    print(f"  总边数: {len(edge_list)}")
    
    # 特征矩阵
    max_feat_dim = max(len(list(drug_features.values())[0]),
                      len(list(gene_features.values())[0]),
                      len(list(disease_features.values())[0]))
    
    x = torch.zeros((total_nodes, max_feat_dim), dtype=torch.float)
    
    for drug_id, idx in drug_to_idx.items():
        x[idx, :len(drug_features[drug_id])] = torch.FloatTensor(drug_features[drug_id])
    for gene_id, idx in gene_to_idx.items():
        x[gene_offset + idx, :len(gene_features[gene_id])] = torch.FloatTensor(gene_features[gene_id])
    for disease_id, idx in disease_to_idx.items():
        x[disease_offset + idx, :len(disease_features[disease_id])] = torch.FloatTensor(disease_features[disease_id])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.zeros(0, dtype=torch.long)
    
    return {
        'x': x.to(device),
        'edge_index': edge_index.to(device),
        'edge_type': edge_type.to(device),
        'num_drugs': num_drugs,
        'num_genes': num_genes,
        'num_diseases': num_diseases,
        'gene_offset': gene_offset,
        'disease_offset': disease_offset,
        'drug_to_idx': drug_to_idx,
        'gene_to_idx': gene_to_idx,
        'disease_to_idx': disease_to_idx,
        'num_relations': 7
    }


# ====================== 4. RGCN模型 ======================
class RGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations=7, num_bases=4, dropout=0.3):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases)
        
        self.dropout = dropout
        
        self.decoder = nn.Sequential(
            nn.Linear(3 * out_channels, hidden_channels),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_type,
                drug_indices=None, gene_indices=None, disease_indices=None):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_type)
        
        if drug_indices is not None:
            tri_embed = torch.cat([x[drug_indices], x[gene_indices], x[disease_indices]], dim=1)
            return torch.sigmoid(self.decoder(tri_embed)).squeeze()
        return x


# ====================== 5. 训练/评估 ======================
def train_epoch(model, graph_data, triplets, labels, optimizer, criterion):
    model.train()
    drug_idx, gene_idx, dis_idx = [], [], []
    for d, g, c in triplets:
        drug_idx.append(graph_data['drug_to_idx'][d])
        gene_idx.append(graph_data['gene_offset'] + graph_data['gene_to_idx'][g])
        dis_idx.append(graph_data['disease_offset'] + graph_data['disease_to_idx'][c])
    
    drug_idx = torch.tensor(drug_idx, dtype=torch.long).to(device)
    gene_idx = torch.tensor(gene_idx, dtype=torch.long).to(device)
    dis_idx = torch.tensor(dis_idx, dtype=torch.long).to(device)
    labels = torch.tensor(labels, dtype=torch.float).to(device)

    optimizer.zero_grad()
    preds = model(graph_data['x'], graph_data['edge_index'], graph_data['edge_type'],
                  drug_idx, gene_idx, dis_idx)
    loss = criterion(preds, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, graph_data, triplets, labels):
    model.eval()
    drug_idx, gene_idx, dis_idx = [], [], []
    for d, g, c in triplets:
        drug_idx.append(graph_data['drug_to_idx'][d])
        gene_idx.append(graph_data['gene_offset'] + graph_data['gene_to_idx'][g])
        dis_idx.append(graph_data['disease_offset'] + graph_data['disease_to_idx'][c])
    
    drug_idx = torch.tensor(drug_idx, dtype=torch.long).to(device)
    gene_idx = torch.tensor(gene_idx, dtype=torch.long).to(device)
    dis_idx = torch.tensor(dis_idx, dtype=torch.long).to(device)

    with torch.no_grad():
        preds = model(graph_data['x'], graph_data['edge_index'], graph_data['edge_type'],
                      drug_idx, gene_idx, dis_idx).cpu().numpy()
    
    auc_score = roc_auc_score(labels, preds)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
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
def run_experiment(data_dir='./dataset', n_splits=5, epochs=100, lr=1e-3,
                   replace_dims=("disease",), neg_ratio=3,
                   hidden_channels=256, out_channels=128):
    
    data_result = prepare_data(data_dir)
    if data_result is None:
        return
    (pos_triplets, drug_features, gene_features, disease_features,
     drug_to_idx, gene_to_idx, disease_to_idx,
     drug_pool, gene_pool, disease_pool) = data_result

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pos_array = np.array(pos_triplets)
    fold_results = []

    print(f"\n{'='*70}")
    print(f"RGCN Baseline with Protein-Disease (Fair Comparison)")
    print(f"Settings: {n_splits}-Fold, Epochs={epochs}, Hidden={hidden_channels}, Out={out_channels}")
    print(f"Relations: PPI(0), DPI(1), DDI(2), Protein-Disease(3), Triplet-DG(4), Triplet-DC(5), Triplet-GC(6)")
    print(f"{'='*70}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(pos_array)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        train_pos = [tuple(t) for t in pos_array[train_idx]]
        val_pos = [tuple(t) for t in pos_array[val_idx]]
        
        print(f"Train: {len(train_pos)} pos, Val: {len(val_pos)} pos")

        # 【关键修复】使用关键字参数调用，避免顺序错误
        train_neg = generate_negatives(
            train_pos, drug_pool, gene_pool, disease_pool, 
            replace_dims=replace_dims,  # 明确指定参数名
            neg_ratio=neg_ratio
        )
        val_neg = generate_negatives(
            val_pos, drug_pool, gene_pool, disease_pool, 
            replace_dims=replace_dims,  # 明确指定参数名
            neg_ratio=neg_ratio
        )

        train_triplets = train_pos + train_neg
        train_labels = [1] * len(train_pos) + [0] * len(train_neg)
        val_triplets = val_pos + val_neg
        val_labels = [1] * len(val_pos) + [0] * len(val_neg)

        # 文件路径（包含protein_disease）
        ppi_file = os.path.join(data_dir, 'ppi.csv')
        dpi_file = os.path.join(data_dir, 'drug_protein.csv')
        ddi_file = os.path.join(data_dir, 'drug_drug.csv')
        protein_disease_file = os.path.join(data_dir, 'protein_disease.csv')

        graph_data = build_fair_comparison_graph(
            train_pos, val_pos,
            drug_features, gene_features, disease_features,
            drug_to_idx, gene_to_idx, disease_to_idx,
            ppi_file, dpi_file, ddi_file, protein_disease_file, device
        )
        
        model = RGCNModel(
            in_channels=graph_data['x'].shape[1],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=7,
            num_bases=4,
            dropout=0.3
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

        best_auc = 0
        best_metrics = None
        
        for epoch in range(epochs):
            loss = train_epoch(model, graph_data, train_triplets, train_labels,
                               optimizer, criterion)
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                auc_score, aupr, f1_score, threshold, _ = evaluate(
                    model, graph_data, val_triplets, val_labels)
                
                print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | "
                      f"AUC: {auc_score:.4f} | AUPR: {aupr:.4f} | F1: {f1_score:.4f}")
                
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_metrics = (auc_score, aupr, f1_score, threshold)
                    torch.save(model.state_dict(), f"best_rgcn_fold{fold+1}.pth")
                
                scheduler.step(auc_score)

        model.load_state_dict(torch.load(f"best_rgcn_fold{fold+1}.pth"))
        final_auc, final_aupr, final_f1, final_thresh, _ = evaluate(
            model, graph_data, val_triplets, val_labels)
        
        tm = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_results.append({
            'timestamp': tm, 
            'model': 'RGCN',
            'mode': str(replace_dims), 
            'fold': fold + 1, 
            'auc': final_auc, 
            'aupr': final_aupr, 
            'f1_score': final_f1, 
            'threshold': final_thresh
        })
        
        print(f"Fold {fold+1} Best: AUC={final_auc:.4f}, AUPR={final_aupr:.4f}, F1={final_f1:.4f}")
        
        if os.path.exists(f"best_rgcn_fold{fold+1}.pth"):
            os.remove(f"best_rgcn_fold{fold+1}.pth")
        torch.cuda.empty_cache()

    # 汇总
    df = pd.DataFrame(fold_results)
    
    # 计算统计指标
    mean_auc = df['auc'].mean()
    std_auc = df['auc'].std()
    mean_aupr = df['aupr'].mean()
    std_aupr = df['aupr'].std()
    mean_f1 = df['f1_score'].mean()
    std_f1 = df['f1_score'].std()
    
    print(f"\n{'='*70}")
    print("RGCN Final Results")
    print(f"{'='*70}")
    print(f"AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")
    print(f"F1:   {mean_f1:.4f} ± {std_f1:.4f}")
    
    # 添加统计汇总行到DataFrame
    summary_rows = [
        {'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"), 'model': 'RGCN', 'mode': str(replace_dims), 'fold': 'MEAN', 
         'auc': mean_auc, 'aupr': mean_aupr, 'f1_score': mean_f1, 'threshold': ''},
        {'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"), 'model': 'RGCN', 'mode': str(replace_dims), 'fold': 'STD', 
         'auc': std_auc, 'aupr': std_aupr, 'f1_score': std_f1, 'threshold': ''}
    ]
    
    # 合并原始结果和统计行
    df_with_summary = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    
    # 保存到CSV
    os.makedirs('./results', exist_ok=True)
    file_path = f'./results/rgcn_summary_new.csv'
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 运行实验
    modes = ["gene", "disease"]
    for mode in modes:
        print(f'\n=== 以 {mode} 替换进行实验 ===')
        run_experiment(
            data_dir='./dataset',
            n_splits=5,
            epochs=1000,
            lr=1e-3,
            replace_dims=(mode,),  # 单元素元组
            neg_ratio=3,
            hidden_channels=256,
            out_channels=128
        )
