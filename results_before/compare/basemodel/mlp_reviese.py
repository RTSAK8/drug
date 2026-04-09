
import os, warnings, random, torch, numpy as np, pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
import torch.nn as nn
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42); random.seed(42)
os.makedirs('./results', exist_ok=True)

# ---------- 1. 负采样 ----------
def generate_negatives(pos_triplets, drug_pool, gene_pool, disease_pool,
                       neg_ratio=3, replace_dims=('drug','gene','disease'), max_attempts=50):
    pos_set = set(pos_triplets)
    neg_samples = []
    for d, g, c in pos_triplets:
        for dim in replace_dims:
            pool = {'drug': drug_pool, 'gene': gene_pool, 'disease': disease_pool}[dim]
            if len(pool) <= 1: continue
            cnt = 0
            attempts = 0
            while cnt < neg_ratio and attempts < max_attempts * neg_ratio:
                attempts += 1
                new_id = random.choice(pool)
                if dim == 'drug' and new_id == d: continue
                if dim == 'gene' and new_id == g: continue
                if dim == 'disease' and new_id == c: continue
                new_tri = (new_id, g, c) if dim == 'drug' else \
                          (d, new_id, c) if dim == 'gene' else (d, g, new_id)
                if new_tri not in pos_set and new_tri not in neg_samples:
                    neg_samples.append(new_tri)
                    cnt += 1
            if cnt < neg_ratio:
                print(f'[Warn] 仅生成 {cnt}/{neg_ratio} 个 {dim} 负样本')
    return neg_samples

# ---------- 2. 特征 & 三元组加载 ----------
def load_features(path, id_col):
    df = pd.read_csv(path)
    feat = {}
    for _, r in df.iterrows():
        k = str(r[id_col])
        vec = r.drop(id_col).values.astype(np.float32)
        feat[k] = vec
        if '.' in k and k.endswith('.0'):
            feat[k[:-2]] = vec
    return feat

def prepare_data(data_dir):
    print('Loading features ...')
    drug_feat = load_features(os.path.join(data_dir, 'drug_embeddings.csv'), 'Drug_ID')
    gene_feat = load_features(os.path.join(data_dir, 'gene_embeddings.csv'), 'Gene_ID')
    dis_feat  = load_features(os.path.join(data_dir, 'disease_embeddings.csv'), 'Disease_ID')
    triplets  = pd.read_csv(os.path.join(data_dir, 'triplets.csv'))
    pos = []
    for _, r in triplets.iterrows():
        d, g, c = str(r['Drug_ID']), str(r['Gene_ID']), str(r['Disease_ID'])
        if d in drug_feat and g in gene_feat and c in dis_feat:
            pos.append((d, g, c))
    if not pos:
        print('No valid triplets!'); return None
    print(f'Pos triplets: {len(pos)}')
    return pos, drug_feat, gene_feat, dis_feat

# ---------- 3. 模型 ----------
# class MLP(torch.nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(dim, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 1)
#         )
#     def forward(self, x):
#         return torch.sigmoid(self.net(x)).squeeze()
class MLP(torch.nn.Module):
    """简单的多层感知机模型"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze()
# ---------- 4. 训练 / 评估 ----------
def make_xy(triplets, labels, drug_feat, gene_feat, dis_feat):
    """返回 (N, max_dim) 的 float32 矩阵 和 (N,) 的 float32 标签"""
    # 1. 先计算统一维度
    d_dim = max(len(v) for v in drug_feat.values())
    g_dim = max(len(v) for v in gene_feat.values())
    c_dim = max(len(v) for v in dis_feat.values())
    total_dim = d_dim + g_dim + c_dim

    # 2. 分配矩阵
    X = np.zeros((len(triplets), total_dim), dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # 3. 填充并 zero-pad
    for i, (d, g, c) in enumerate(triplets):
        d_vec = drug_feat[d]
        g_vec = gene_feat[g]
        c_vec = dis_feat[c]
        # pad
        d_pad = np.pad(d_vec, (0, d_dim - len(d_vec)))
        g_pad = np.pad(g_vec, (0, g_dim - len(g_vec)))
        c_pad = np.pad(c_vec, (0, c_dim - len(c_vec)))
        X[i] = np.concatenate([d_pad, g_pad, c_pad])
    return torch.tensor(X).to(device), torch.tensor(y).to(device)

def train_epoch(model, X, y, optimizer, criterion, batch_size=256):
    model.train()
    idx = torch.randperm(X.size(0))
    L = 0
    for i in range(0, len(idx), batch_size):
        sl = idx[i:i+batch_size]
        pred = model(X[sl])
        loss = criterion(pred, y[sl])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        L += loss.item() * len(sl)
    return L / len(idx)



def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    
    y_true = y.cpu().numpy()
    
    # 计算AUC
    auc_ = roc_auc_score(y_true, pred)
    
    # 计算AUPR和最佳F1
    prec, rec, thresholds = precision_recall_curve(y_true, pred)
    aupr = auc(rec, prec)
    
    # 计算最佳阈值下的F1-score
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
    f1_scores = f1_scores[:-1]  # 移除最后一个值
    
    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_threshold = float(thresholds[best_idx])
    else:
        best_f1 = 0.0
        best_threshold = 0.5
    
    return auc_, aupr, best_f1, best_threshold, pred

# ---------- 5. 排名指标 ----------
def predict_scores(model, triplets, drug_feat, gene_feat, dis_feat):
    labels = [0] * len(triplets)
    X, _ = make_xy(triplets, labels, drug_feat, gene_feat, dis_feat)
    model.eval()
    with torch.no_grad():
        return model(X).cpu().numpy()


def ranking_metrics_full_candidates(model, pos_triplets, replace_dim,
                                   drug_pool, gene_pool, disease_pool,
                                   drug_feat, gene_feat, dis_feat):
    ranks = []

    unique_drugs = sorted(set(drug_pool))
    unique_genes = sorted(set(gene_pool))
    unique_diseases = sorted(set(disease_pool))

    for d, g, c in pos_triplets:
        if replace_dim == 'drug':
            candidates = unique_drugs
            candidate_triplets = [(cand, g, c) for cand in candidates]
            target = d
        elif replace_dim == 'gene':
            candidates = unique_genes
            candidate_triplets = [(d, cand, c) for cand in candidates]
            target = g
        else:
            candidates = unique_diseases
            candidate_triplets = [(d, g, cand) for cand in candidates]
            target = c

        if len(candidates) < 2 or target not in candidates:
            continue

        scores = predict_scores(model, candidate_triplets, drug_feat, gene_feat, dis_feat)
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

# ---------- 6. 主实验 ----------
def run(data_dir='./dataset', splits=5, epochs=100, lr=1e-3, neg_ratio=3, replace_dim='gene'):
    print('mode = ', replace_dim)
    data = prepare_data(data_dir)
    if data is None: return
    pos_triplets, drug_feat, gene_feat, dis_feat = data
    kf = KFold(splits, shuffle=True, random_state=42)
    res = []
    for fold, (tr, va) in enumerate(kf.split(pos_triplets)):
        print(f'\nFold {fold+1}/{splits}')
        # 正样本
        train_pos = [pos_triplets[i] for i in tr]
        val_pos   = [pos_triplets[i] for i in va]
        # 负样本
        train_neg = generate_negatives(train_pos,
                                       list(set([d for d,_,_ in train_pos])),
                                       list(set([g for _,g,_ in train_pos])),
                                       list(set([c for _,_,c in train_pos])),
                                       neg_ratio, [replace_dim])
        val_neg   = generate_negatives(val_pos,
                                       list(set([d for d,_,_ in val_pos])),
                                       list(set([g for _,g,_ in val_pos])),
                                       list(set([c for _,_,c in val_pos])),
                                       neg_ratio, [replace_dim])
        # 拼接
        train_tri = train_pos + train_neg
        train_lab = [1]*len(train_pos) + [0]*len(train_neg)
        val_tri   = val_pos + val_neg
        val_lab   = [1]*len(val_pos) + [0]*len(val_neg)
        # 转矩阵
        X_train, y_train = make_xy(train_tri, train_lab, drug_feat, gene_feat, dis_feat)
        X_val,   y_val   = make_xy(val_tri,   val_lab,   drug_feat, gene_feat, dis_feat)
        # 模型
        dim = X_train.shape[1]
        model = MLP(dim).to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
        best = 0
        for epoch in range(epochs):
            tr_loss = train_epoch(model, X_train, y_train, optimizer, criterion)
            if (epoch+1)%10 == 0 or epoch==epochs-1:
                auc_, aupr, f1_score, threshold, pred = evaluate(model, X_val, y_val)
                scheduler.step(auc_)
                if auc_ > best: best = auc_
                print(f'Epoch {epoch+1:03d}/{epochs} | loss {tr_loss:.4f} | AUC {auc_:.4f} | AUPR {aupr:.4f}')
                print(f' | Best F1-score: {f1_score:.4f} @ threshold {threshold:.4f}')
        # 最终指标
        auc_, aupr, f1_score, threshold, pred = evaluate(model, X_val, y_val)
        mrr, hits = ranking_metrics_full_candidates(
            model=model,
            pos_triplets=val_pos,
            replace_dim=replace_dim,
            drug_pool=list(drug_feat.keys()),
            gene_pool=list(gene_feat.keys()),
            disease_pool=list(dis_feat.keys()),
            drug_feat=drug_feat,
            gene_feat=gene_feat,
            dis_feat=dis_feat,
        )
        print(f'Fold {fold+1} 最终 | AUC {auc_:.4f} | AUPR {aupr:.4f} | MRR {mrr:.4f}')
        print(f'最佳 F1-score {f1_score:.4f} @ threshold {threshold:.4f}')
        tm = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = replace_dim
        res.append({'timestamp':tm, 'mode': mode, 'fold':fold+1, 'auc':auc_, 'aupr':aupr, 'f1_score':f1_score, 'threshold':threshold, 'mrr':mrr, **hits})
    # 汇总
    df = pd.DataFrame(res)
    print(f"\n平均 AUC: {df['auc'].mean():.4f}±{df['auc'].std():.4f}")
    print(f"平均 AUPR: {df['aupr'].mean():.4f}±{df['aupr'].std():.4f}")
    print(f"平均 F1-score: {df['f1_score'].mean():.4f}±{df['f1_score'].std():.4f}")
    print(f"平均 MRR: {df['mrr'].mean():.4f}±{df['mrr'].std():.4f}")
    # df.loc['mean'] = df.mean()
    # df.loc['std']  = df.std()
    file_path = f'./results/mlp_summary.csv'
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
        print(f'结果已追加到 {file_path}')
    else:
        df.to_csv(file_path, mode='w', index=False)
        print(f'结果已保存到 {file_path}')


# ---------- 7. 运行 ----------
if __name__ == '__main__':
    replace_dim = ['gene', 'disease']
    for dim in replace_dim:
        print(f'\n=== 以 {dim} 替换进行实验 ===')
        run(data_dir='./dataset', splits=5, epochs=1000, lr=1e-3, neg_ratio=3, replace_dim=dim)
