import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from config import Config
from data_manager import DataManager
from models import BioEncoder, HgnnEncoder, RelationalGraphEncoder, Decoder, HGNN
from train_utils import train_step, validate_step, build_hypergraph, build_binary_edges, apply_offsets, generate_negatives_for_fold, save_summary_results, metrics_graph_with_hits

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_triplets_data(triplets_df: pd.DataFrame, data_manager: DataManager) -> pd.DataFrame:
    """准备三元组数据"""
    if triplets_df is None or triplets_df.empty:
        return pd.DataFrame(columns=["Drug_ID", "Gene_ID", "Disease_ID", "Label"])
    required_cols = {"Drug_ID", "Gene_ID", "Disease_ID"}
    if not required_cols.issubset(set(triplets_df.columns)):
        print(f"警告: 三元组文件缺少必要列 {required_cols}，实际列: {list(triplets_df.columns)}")
        return pd.DataFrame(columns=["Drug_ID", "Gene_ID", "Disease_ID", "Label"])

    prepared_df = triplets_df.copy()

    # 映射ID
    prepared_df["Drug_ID"] = prepared_df["Drug_ID"].map(data_manager.drug_to_idx).fillna(-1).astype(int)
    prepared_df["Gene_ID"] = prepared_df["Gene_ID"].map(data_manager.gene_to_idx).fillna(-1).astype(int)
    prepared_df["Disease_ID"] = prepared_df["Disease_ID"].map(data_manager.disease_to_idx).fillna(-1).astype(int)
    prepared_df["Label"] = 1

    # 移除无效映射
    prepared_df = prepared_df[(prepared_df["Drug_ID"] != -1) & (prepared_df["Gene_ID"] != -1) & (prepared_df["Disease_ID"] != -1)]

    print(f"有效三元组数量: {len(prepared_df)}")
    return prepared_df


def run_experiment(config: Config):
    set_seed(config.random_state)

    data_manager = DataManager(config)

    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    data = data_manager.load_all_data()
    data_manager.create_index_mappings(data)

    print("\n处理实体特征...")
    drug_fea = data_manager.process_entity_features(data["drug"], "Drug_ID", data_manager.drug_to_idx)
    gene_fea = data_manager.process_entity_features(data["gene"], "Gene_ID", data_manager.gene_to_idx)
    disease_fea = data_manager.process_entity_features(data["disease"], "Disease_ID", data_manager.disease_to_idx)

    print("\n准备三元组数据...")
    train_triplets_df = prepare_triplets_data(data["triplets"], data_manager)
    test_triplets_df = prepare_triplets_data(data.get("drugbank_triplets", pd.DataFrame()), data_manager)

    if test_triplets_df.empty:
        print("警告: drugbank_triplets 为空，将无法进行独立测试评估")

    global_pos_set = set(tuple(x) for x in train_triplets_df[["Drug_ID", "Gene_ID", "Disease_ID"]].values)
    global_pos_set |= set(tuple(x) for x in test_triplets_df[["Drug_ID", "Gene_ID", "Disease_ID"]].values)

    print("\n准备二元边...")
    global_edges = data_manager.prepare_binary_edges(data)

    print(f"边统计:")
    for edge_type, edge_data in global_edges.items():
        if isinstance(edge_data, np.ndarray):
            print(f"  {edge_type}: {edge_data.shape[1]} edges")
        else:
            print(f"  {edge_type}: {len(edge_data)} hyperedges")

    modes = [["gene"], ["disease"]]

    for current_mode in modes:
        run_mode_experiment(current_mode=current_mode, train_triplets_df=train_triplets_df, test_triplets_df=test_triplets_df, global_pos_set=global_pos_set, global_edges=global_edges, data_manager=data_manager, drug_fea=drug_fea, gene_fea=gene_fea, disease_fea=disease_fea, config=config)


def run_mode_experiment(current_mode, train_triplets_df, test_triplets_df, global_pos_set, global_edges, data_manager, drug_fea, gene_fea, disease_fea, config):
    """运行单个模式的实验（triplets 全量训练 + drugbank 独立测试）"""
    print(f"\n{'='*60}")
    print(f"开始模式: {current_mode[0]} 留出")
    print(f"{'='*60}")

    for ablation_name, include_relations in config.graph_ablation_configs.items():
        print(f"\n消融实验: {ablation_name}")
        print(f"包含的关系: {include_relations if include_relations else '仅三元组'}")

        offset_edges = data_manager.get_offset_edges(global_edges, include_relations)

        if test_triplets_df.empty:
            print("跳过：drugbank_triplets 为空")
            continue

        train_df = train_triplets_df
        test_df = test_triplets_df

        train_drug_pool = train_df["Drug_ID"].unique().tolist()
        train_gene_pool = train_df["Gene_ID"].unique().tolist()
        train_disease_pool = train_df["Disease_ID"].unique().tolist()

        train_neg = generate_negatives_for_fold(
            pos_samples=train_df[["Drug_ID", "Gene_ID", "Disease_ID"]].values,
            drug_pool=train_drug_pool,
            gene_pool=train_gene_pool,
            disease_pool=train_disease_pool,
            global_exclude=global_pos_set.copy(),
            replace_dims=current_mode,
            neg_ratio=config.neg_ratio,
        )

        test_neg = generate_negatives_for_fold(
            pos_samples=test_df[["Drug_ID", "Gene_ID", "Disease_ID"]].values,
            drug_pool=train_drug_pool,
            gene_pool=train_gene_pool,
            disease_pool=train_disease_pool,
            global_exclude=global_pos_set.copy(),
            replace_dims=current_mode,
            neg_ratio=config.neg_ratio,
        )

        train_pos_offset = apply_offsets(
            train_df[["Drug_ID", "Gene_ID", "Disease_ID", "Label"]].values.copy(),
            data_manager.gene_offset,
            data_manager.disease_offset,
        )
        test_pos_offset = apply_offsets(
            test_df[["Drug_ID", "Gene_ID", "Disease_ID", "Label"]].values.copy(),
            data_manager.gene_offset,
            data_manager.disease_offset,
        )
        train_neg_offset = apply_offsets(train_neg.copy(), data_manager.gene_offset, data_manager.disease_offset)
        test_neg_offset = apply_offsets(test_neg.copy(), data_manager.gene_offset, data_manager.disease_offset)

        train_data = np.vstack([train_pos_offset, train_neg_offset])
        test_data = np.vstack([test_pos_offset, test_neg_offset])

        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

        # 注意：超图仅由训练集构建，避免测试集信息泄露
        # 超图：只包含三元组(+可选 gene_hyperedges)
        hypergraph_train_pos = build_hypergraph(
            data=train_pos_offset,
            gene_hyperedges=offset_edges["gene_hyperedges"],
        )
        hypergraph_train_neg = build_hypergraph(
            data=train_neg_offset,
            gene_hyperedges=offset_edges["gene_hyperedges"],
        )

        # 二元关系：以普通边形式编码，并显式携带 edge_type
        bin_edge_index, bin_edge_type = build_binary_edges(
            ppi_data=offset_edges["ppi"],
            dpi_data=offset_edges["dpi"],
            ddi_data=offset_edges["ddi"],
            protein_disease_data=offset_edges["protein_disease"],
            make_undirected=True,
        )

        test_metrics = train_fold_model(
            current_mode=current_mode,
            ablation_name=ablation_name,
            fold=0,
            train_data=train_data,
            val_data=test_data,
            train_pos=train_pos_offset,
            val_pos=test_pos_offset,
            hypergraph_pos=hypergraph_train_pos,
            hypergraph_neg=hypergraph_train_neg,
            bin_edge_index=bin_edge_index,
            bin_edge_type=bin_edge_type,
            drug_fea=drug_fea,
            gene_fea=gene_fea,
            disease_fea=disease_fea,
            config=config,
            use_checkpoint=False,
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_metrics = {f"{k}_mean": float(v) for k, v in test_metrics.items()}
        avg_metrics.update({f"{k}_std": 0.0 for k in test_metrics.keys()})

        print(f"\n{'='*60}")
        print(f"{ablation_name} 独立测试集(DrugBank)结果:")
        print(f"{'='*60}")
        print(f"AUC:  {avg_metrics['auc_mean']:.4f}")
        print(f"AUPR: {avg_metrics['aupr_mean']:.4f}")
        print(f"F1:   {avg_metrics['f1_score_mean']:.4f}")

        save_summary_results(
            mode=current_mode[0],
            ablation_name=ablation_name,
            avg_metrics=avg_metrics,
            all_fold_metrics=[test_metrics],
            config=vars(config),
        )


def train_fold_model(current_mode, ablation_name: str, fold: int, train_data: np.ndarray, val_data: np.ndarray, train_pos: np.ndarray, val_pos: np.ndarray, hypergraph_pos: torch.Tensor, hypergraph_neg: torch.Tensor, bin_edge_index: torch.Tensor, bin_edge_type: torch.Tensor, drug_fea: torch.Tensor, gene_fea: torch.Tensor, disease_fea: torch.Tensor, config: Config, use_checkpoint: bool = True):
    """训练模型，并在给定评估集上评估（use_checkpoint=False 时不做训练中验证/选最优）"""

    bio_encoder = BioEncoder(dim_drug=drug_fea.shape[1], dim_gene=gene_fea.shape[1], dim_disease=disease_fea.shape[1], encoder_output_dim=config.encoder_output_dim)

    graph_encoder = HgnnEncoder(
        in_channels=config.encoder_output_dim,
        dim_1=config.hidden_dim,
        use_residual=config.hgnn_use_residual,
        jk=config.hgnn_jk,
        norm=config.hgnn_norm,
        hyperedge_dropout=config.hgnn_hyperedge_dropout,
        dropout=config.hgnn_dropout,
    )

    binary_graph_encoder = RelationalGraphEncoder(
        in_channels=config.encoder_output_dim,
        dim_1=config.hidden_dim,
        num_relations=4,
        use_residual=getattr(config, "rgcn_use_residual", False),
        dropout=getattr(config, "rgcn_dropout", 0.4),
    )

    decoder_input_dim = (config.hidden_dim // 4) * 3
    decoder = Decoder(decoder_input_dim)

    model = HGNN(
        bio_encoder=bio_encoder,
        graph_encoder=graph_encoder,
        binary_graph_encoder=binary_graph_encoder,
        decoder=decoder,
        out_channels=config.hidden_dim // 4,
        replace_dims=current_mode,
        loss_type=config.loss_type,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.L2)

    best_mrr = 0
    model_path = f"best_model_{current_mode[0]}_{ablation_name}_fold{fold}.pth"

    for epoch in range(config.epochs):
        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            loss_func=criterion,
            drug_emb=drug_fea,
            gene_emb=gene_fea,
            disease_emb=disease_fea,
            hp_pos=hypergraph_pos,
            hp_neg=hypergraph_neg,
            bin_edge_index=bin_edge_index,
            bin_edge_type=bin_edge_type,
            train_data=train_data,
            train_pos=train_pos,
            epoch=epoch,
            alpha=config.alpha,
        )

        if use_checkpoint and (epoch % 10 == 0 or epoch == config.epochs - 1):
            auc, aupr, f1_score, threshold, mrr, hits = validate_step(
                model=model,
                drug_emb=drug_fea,
                gene_emb=gene_fea,
                disease_emb=disease_fea,
                hp_pos=hypergraph_pos,
                hp_neg=hypergraph_neg,
                bin_edge_index=bin_edge_index,
                bin_edge_type=bin_edge_type,
                val_data=val_data,
                val_pos=val_pos,
            )

            print(f"Epoch {epoch+1:03d} | Val: AUC={auc:.4f}, AUPR={aupr:.4f}, " f"F1={f1_score:.4f}, thresh={threshold:.4f}, ")

            if mrr > best_mrr:
                best_mrr = mrr
                torch.save(model.state_dict(), model_path)

            torch.cuda.empty_cache()

    if use_checkpoint and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    with torch.no_grad():
        val_data_tensor = torch.from_numpy(val_data).long().to(device)
        val_pos_tensor = torch.from_numpy(val_pos).long().to(device)

        pred, _, mrr, ranks = model(
            drug_fea,
            gene_fea,
            disease_fea,
            hypergraph_pos,
            hypergraph_neg,
            val_data_tensor[:, 0],
            val_data_tensor[:, 1],
            val_data_tensor[:, 2],
            val_pos_tensor[:, 0],
            val_pos_tensor[:, 1],
            val_pos_tensor[:, 2],
            bin_edge_index,
            bin_edge_type,
        )

        y_true = val_data[:, 3]
        y_pred = pred.cpu().numpy()

        auc, aupr, best_f1_score, threshold, hits = metrics_graph_with_hits(y_true, y_pred)

        final_metrics = {"auc": auc, "aupr": aupr, "f1_score": best_f1_score, "threshold": threshold}

        print(f"\n最终测试结果:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

    if use_checkpoint and os.path.exists(model_path):
        os.remove(model_path)

    return final_metrics


if __name__ == "__main__":
    config = Config()
    run_experiment(config)
