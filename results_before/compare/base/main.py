
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
import warnings
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
from models import BioEncoder, HgnnEncoder, Decoder, HGNN
from train_utils import (train_step, validate_step, build_hypergraph,
                        apply_offsets, generate_negatives_for_fold,
                        save_summary_results, metrics_graph_with_hits,
                        calculate_hits_ranking_based)

try:
    import wandb
except ImportError:
    wandb = None

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def _wandb_enabled(config: Config) -> bool:
    return bool(getattr(config, "use_wandb", False))


def _init_wandb_run(config: Config, data_manager: DataManager):
    if not _wandb_enabled(config):
        return None
    if wandb is None:
        warnings.warn("检测到已启用wandb，但当前环境未安装wandb，已跳过日志记录。")
        return None

    project = getattr(config, "wandb_project", "protein-hgnn")
    entity = getattr(config, "wandb_entity", None)
    name_prefix = str(getattr(config, "wandb_name_prefix", "base"))
    run_index = int(getattr(config, "wandb_run_index", 1))

    run = wandb.init(
        project=project,
        entity=entity,
        group="mode_comparison",
        name=f"{name_prefix}_{run_index}",
        config=vars(config),
        reinit=True,
    )

    for mode_key in ("gene", "disease"):
        run.define_metric(f"{mode_key}/train_epoch", hidden=True)
        run.define_metric(f"{mode_key}/train_total_loss", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_main_loss", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_cl_loss", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_mrr", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_hit@1", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_hit@3", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_hit@5", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/val_epoch", hidden=True)
        run.define_metric(f"{mode_key}/val/*", step_metric=f"{mode_key}/val_epoch")

    run.summary["n_ablations"] = int(len(config.graph_ablation_configs))
    run.summary["num_drugs"] = int(len(data_manager.drug_to_idx))
    run.summary["num_genes"] = int(len(data_manager.gene_to_idx))
    run.summary["num_diseases"] = int(len(data_manager.disease_to_idx))
    return run

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
    prepared_df = triplets_df.copy()
    
    # 映射ID
    prepared_df['Drug_ID'] = prepared_df['Drug_ID'].map(data_manager.drug_to_idx).fillna(-1).astype(int)
    prepared_df['Gene_ID'] = prepared_df['Gene_ID'].map(data_manager.gene_to_idx).fillna(-1).astype(int)
    prepared_df['Disease_ID'] = prepared_df['Disease_ID'].map(data_manager.disease_to_idx).fillna(-1).astype(int)
    prepared_df['Label'] = 1
    
    # 移除无效映射
    prepared_df = prepared_df[
        (prepared_df['Drug_ID'] != -1) & 
        (prepared_df['Gene_ID'] != -1) & 
        (prepared_df['Disease_ID'] != -1)
    ]
    
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
    drug_fea = data_manager.process_entity_features(data['drug'], 'Drug_ID', data_manager.drug_to_idx)
    gene_fea = data_manager.process_entity_features(data['gene'], 'Gene_ID', data_manager.gene_to_idx)
    disease_fea = data_manager.process_entity_features(data['disease'], 'Disease_ID', data_manager.disease_to_idx)
    
    print("\n准备三元组数据...")
    train_triplets_df = prepare_triplets_data(data['triplets'], data_manager)
    test_triplets_df = prepare_triplets_data(data.get('drugbank_triplets', pd.DataFrame()), data_manager)

    if bool(getattr(config, "use_drugbank_independent_test", True)) and test_triplets_df.empty:
        print("警告: drugbank_triplets 为空，将无法进行独立测试评估")

    train_pos_set = set(tuple(x) for x in train_triplets_df[['Drug_ID', 'Gene_ID', 'Disease_ID']].values)
    all_pos_set = set(train_pos_set)
    all_pos_set |= set(tuple(x) for x in test_triplets_df[['Drug_ID', 'Gene_ID', 'Disease_ID']].values)
    
    print("\n准备二元边...")
    global_edges = data_manager.prepare_binary_edges(data)
    
    print(f"边统计:")
    for edge_type, edge_data in global_edges.items():
        if isinstance(edge_data, np.ndarray):
            print(f"  {edge_type}: {edge_data.shape[1]} edges")
        else:
            print(f"  {edge_type}: {len(edge_data)} hyperedges")
    
    modes = [["gene"], ["disease"]]

    wandb_run = _init_wandb_run(config, data_manager)
    try:
        total_ablations = len(config.graph_ablation_configs)
        for mode_index, current_mode in enumerate(modes):
            run_mode_experiment(
                current_mode=current_mode,
                mode_index=mode_index,
                total_ablations=total_ablations,
                train_triplets_df=train_triplets_df,
                test_triplets_df=test_triplets_df,
                train_pos_set=train_pos_set,
                all_pos_set=all_pos_set,
                global_edges=global_edges,
                data_manager=data_manager,
                drug_fea=drug_fea,
                gene_fea=gene_fea,
                disease_fea=disease_fea,
                config=config,
                wandb_run=wandb_run,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

def run_mode_experiment(current_mode, mode_index, total_ablations, train_triplets_df, test_triplets_df,
                       train_pos_set, all_pos_set, global_edges,
                       data_manager, drug_fea,
                       gene_fea, disease_fea,
                       config, wandb_run=None):
    """运行单个模式的实验"""
    print(f"\n{'='*60}")
    print(f"开始模式: {current_mode[0]} 留出")
    print(f"{'='*60}")
    
    for ablation_index, (ablation_name, include_relations) in enumerate(config.graph_ablation_configs.items()):
        print(f"\n消融实验: {ablation_name}")
        print(f"包含的关系: {include_relations if include_relations else '仅三元组'}")

        if bool(getattr(config, "use_drugbank_independent_test", True)) and test_triplets_df.empty:
            print("跳过：drugbank_triplets 为空")
            continue

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.summary[f"{mode_key}/meta/n_folds"] = 1
        
        offset_edges = data_manager.get_offset_edges(global_edges, include_relations)
        
        all_fold_metrics = []

        print("\n--- Fold 1/1 (triplets训练 + drugbank独立测试) ---")

        train_df = train_triplets_df
        val_df = test_triplets_df

        train_drug_pool = train_df['Drug_ID'].unique().tolist()
        train_gene_pool = train_df['Gene_ID'].unique().tolist()
        train_disease_pool = train_df['Disease_ID'].unique().tolist()

        train_neg = generate_negatives_for_fold(
            pos_samples=train_df[['Drug_ID', 'Gene_ID', 'Disease_ID']].values,
            drug_pool=train_drug_pool,
            gene_pool=train_gene_pool,
            disease_pool=train_disease_pool,
            global_exclude=train_pos_set.copy(),
            replace_dims=current_mode,
            neg_ratio=config.neg_ratio
        )

        val_neg = generate_negatives_for_fold(
            pos_samples=val_df[['Drug_ID', 'Gene_ID', 'Disease_ID']].values,
            drug_pool=train_drug_pool,
            gene_pool=train_gene_pool,
            disease_pool=train_disease_pool,
            global_exclude=all_pos_set.copy(),
            replace_dims=current_mode,
            neg_ratio=config.neg_ratio
        )

        train_pos_offset = apply_offsets(
            train_df[['Drug_ID', 'Gene_ID', 'Disease_ID', 'Label']].values.copy(),
            data_manager.gene_offset, data_manager.disease_offset
        )
        val_pos_offset = apply_offsets(
            val_df[['Drug_ID', 'Gene_ID', 'Disease_ID', 'Label']].values.copy(),
            data_manager.gene_offset, data_manager.disease_offset
        )
        train_neg_offset = apply_offsets(
            train_neg.copy(),
            data_manager.gene_offset, data_manager.disease_offset
        )
        val_neg_offset = apply_offsets(
            val_neg.copy(),
            data_manager.gene_offset, data_manager.disease_offset
        )

        train_data = np.vstack([train_pos_offset, train_neg_offset])
        val_data = np.vstack([val_pos_offset, val_neg_offset])

        np.random.shuffle(train_data)
        np.random.shuffle(val_data)

        hypergraph_train_pos = build_hypergraph(
            data=train_pos_offset,
            ppi_data=offset_edges['ppi'],
            dpi_data=offset_edges['dpi'],
            ddi_data=offset_edges['ddi'],
            protein_disease_data=offset_edges['protein_disease'],
            gene_hyperedges=offset_edges['gene_hyperedges']
        )

        hypergraph_train_neg = build_hypergraph(
            data=train_neg_offset,
            ppi_data=offset_edges['ppi'],
            dpi_data=offset_edges['dpi'],
            ddi_data=offset_edges['ddi'],
            protein_disease_data=offset_edges['protein_disease'],
            gene_hyperedges=offset_edges['gene_hyperedges']
        )

        fold_metrics = train_fold_model(
            current_mode=current_mode,
            mode_index=mode_index,
            total_ablations=total_ablations,
            ablation_index=ablation_index,
            ablation_name=ablation_name,
            fold=0,
            total_folds=1,
            train_data=train_data,
            val_data=val_data,
            train_pos=train_pos_offset,
            val_pos=val_pos_offset,
            hypergraph_pos=hypergraph_train_pos,
            hypergraph_neg=hypergraph_train_neg,
            drug_fea=drug_fea,
            gene_fea=gene_fea,
            disease_fea=disease_fea,
            config=config,
            use_checkpoint=not bool(getattr(config, "use_drugbank_independent_test", True)),
            wandb_run=wandb_run,
        )

        all_fold_metrics.append(fold_metrics)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if all_fold_metrics:
            avg_metrics = {}
            for metric in all_fold_metrics[0].keys():
                values = [m[metric] for m in all_fold_metrics]
                avg_metrics[f'{metric}_mean'] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
            
            print(f"\n{'='*60}")
            print(f"{ablation_name} 独立测试集(DrugBank)结果:")
            print(f"{'='*60}")
            print(f"AUC:  {avg_metrics['auc_mean']:.4f}")
            print(f"AUPR: {avg_metrics['aupr_mean']:.4f}")
            print(f"F1:   {avg_metrics['f1_score_mean']:.4f}")
            if 'mrr_mean' in avg_metrics:
                print(f"MRR:  {avg_metrics['mrr_mean']:.4f}")
            if 'hit@1_mean' in avg_metrics:
                print(f"Hit@1:{avg_metrics['hit@1_mean']:.4f}")
            if 'hit@3_mean' in avg_metrics:
                print(f"Hit@3:{avg_metrics['hit@3_mean']:.4f}")
            if 'hit@5_mean' in avg_metrics:
                print(f"Hit@5:{avg_metrics['hit@5_mean']:.4f}")

            if wandb_run is not None:
                mode_key = current_mode[0]
                wandb_run.summary[f"{mode_key}/cv/auc_mean"] = float(avg_metrics.get('auc_mean', 0.0))
                wandb_run.summary[f"{mode_key}/cv/aupr_mean"] = float(avg_metrics.get('aupr_mean', 0.0))
                wandb_run.summary[f"{mode_key}/cv/f1_score_mean"] = float(avg_metrics.get('f1_score_mean', 0.0))
                wandb_run.summary[f"{mode_key}/cv/mrr_mean"] = float(avg_metrics.get('mrr_mean', 0.0))
                wandb_run.summary[f"{mode_key}/cv/hit@1_mean"] = float(avg_metrics.get('hit@1_mean', 0.0))
                wandb_run.summary[f"{mode_key}/cv/hit@3_mean"] = float(avg_metrics.get('hit@3_mean', 0.0))
                wandb_run.summary[f"{mode_key}/cv/hit@5_mean"] = float(avg_metrics.get('hit@5_mean', 0.0))
            
            save_summary_results(
                mode=current_mode[0],
                ablation_name=ablation_name,
                avg_metrics=avg_metrics,
                all_fold_metrics=all_fold_metrics,
                config=vars(config)
            )

def train_fold_model(current_mode, mode_index: int, total_ablations: int,
                    ablation_index: int, ablation_name: str, fold: int,
                    total_folds: int,
                    train_data: np.ndarray, val_data: np.ndarray,
                    train_pos: np.ndarray, val_pos: np.ndarray,
                    hypergraph_pos: torch.Tensor, hypergraph_neg: torch.Tensor,
                    drug_fea: torch.Tensor, gene_fea: torch.Tensor,
                    disease_fea: torch.Tensor, config: Config,
                    use_checkpoint: bool = True,
                    wandb_run=None):
    """训练单个fold的模型"""
    
    bio_encoder = BioEncoder(
        dim_drug=drug_fea.shape[1],
        dim_gene=gene_fea.shape[1],
        dim_disease=disease_fea.shape[1],
        encoder_output_dim=config.encoder_output_dim
    )
    
    graph_encoder = HgnnEncoder(
        in_channels=config.encoder_output_dim,
        dim_1=config.hidden_dim
    )
    
    decoder_input_dim = (config.hidden_dim // 4) * 3
    decoder = Decoder(decoder_input_dim)
    
    model = HGNN(
        bio_encoder=bio_encoder,
        graph_encoder=graph_encoder,
        decoder=decoder,
        out_channels=config.hidden_dim // 4,
        replace_dims=current_mode,
        loss_type=config.loss_type
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.L2
    )
    
    best_mrr = 0
    model_path = f"best_model_{current_mode[0]}_{ablation_name}_fold{fold}.pth"
    
    for epoch in range(config.epochs):
        global_step = (
            mode_index * (total_ablations * total_folds * config.epochs)
            + ablation_index * (total_folds * config.epochs)
            + fold * config.epochs
            + epoch
            + 1
        )

        train_metrics = train_step(
            model=model,
            optimizer=optimizer,
            loss_func=criterion,
            drug_emb=drug_fea,
            gene_emb=gene_fea,
            disease_emb=disease_fea,
            hp_pos=hypergraph_pos,
            hp_neg=hypergraph_neg,
            train_data=train_data,
            train_pos=train_pos,
            epoch=epoch,
            alpha=config.alpha
        )

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.log(
                {
                    f"{mode_key}/train_total_loss": float(train_metrics['loss']),
                    f"{mode_key}/train_main_loss": float(train_metrics['main_loss']),
                    f"{mode_key}/train_cl_loss": float(train_metrics['cl_loss']),
                    f"{mode_key}/train_mrr": float(train_metrics['mrr']),
                    f"{mode_key}/train_hit@1": float(train_metrics['hit@1']),
                    f"{mode_key}/train_hit@3": float(train_metrics['hit@3']),
                    f"{mode_key}/train_hit@5": float(train_metrics['hit@5']),
                    f"{mode_key}/train_epoch": int(epoch + 1),
                },
                step=global_step,
                commit=False,
            )
        
        if use_checkpoint and (epoch % 10 == 0 or epoch == config.epochs - 1):
            auc, aupr, f1_score, threshold, mrr, hits = validate_step(
                model=model,
                drug_emb=drug_fea,
                gene_emb=gene_fea,
                disease_emb=disease_fea,
                hp_pos=hypergraph_pos,
                hp_neg=hypergraph_neg,
                val_data=val_data,
                val_pos=val_pos
            )
            
            print(f"Epoch {epoch+1:03d} | Val: AUC={auc:.4f}, AUPR={aupr:.4f}, "
                  f"F1={f1_score:.4f}, thresh={threshold:.4f}, ")
            
            if mrr > best_mrr:
                best_mrr = mrr
                torch.save(model.state_dict(), model_path)

            if wandb_run is not None:
                mode_key = current_mode[0]
                wandb_run.log(
                    {
                        f"{mode_key}/val_epoch": int(epoch + 1),
                        f"{mode_key}/val/auc": float(auc),
                        f"{mode_key}/val/aupr": float(aupr),
                        f"{mode_key}/val/f1_score": float(f1_score),
                        f"{mode_key}/val/threshold": float(threshold),
                        f"{mode_key}/val/mrr": float(mrr),
                    },
                    step=global_step,
                )
            
            torch.cuda.empty_cache()
    
    if use_checkpoint and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    with torch.no_grad():
        val_data_tensor = torch.from_numpy(val_data).long().to(device)
        val_pos_tensor = torch.from_numpy(val_pos).long().to(device)
        
        pred, _, mrr, ranks = model(
            drug_fea, gene_fea, disease_fea,
            hypergraph_pos, hypergraph_neg,
            val_data_tensor[:, 0],
            val_data_tensor[:, 1],
            val_data_tensor[:, 2],
            val_pos_tensor[:, 0],
            val_pos_tensor[:, 1],
            val_pos_tensor[:, 2]
        )
        
        y_true = val_data[:, 3]
        y_pred = pred.cpu().numpy()
        
        auc, aupr, best_f1_score, threshold, _ = metrics_graph_with_hits(y_true, y_pred)
        rank_hits = calculate_hits_ranking_based(ranks, [1, 3, 5])
        
        final_metrics = {
            'auc': auc,
            'aupr': aupr,
            'f1_score': best_f1_score,
            'threshold': threshold,
            'mrr': float(mrr),
            'hit@1': float(rank_hits['hit@1']),
            'hit@3': float(rank_hits['hit@3']),
            'hit@5': float(rank_hits['hit@5']),
        }
        
        print(f"\n最终测试结果:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

    if wandb_run is not None:
        mode_key = current_mode[0]
        wandb_run.summary[f"{mode_key}/fold{fold+1}/auc"] = float(final_metrics.get('auc', 0.0))
        wandb_run.summary[f"{mode_key}/fold{fold+1}/aupr"] = float(final_metrics.get('aupr', 0.0))
        wandb_run.summary[f"{mode_key}/fold{fold+1}/f1_score"] = float(final_metrics.get('f1_score', 0.0))
        wandb_run.summary[f"{mode_key}/fold{fold+1}/threshold"] = float(final_metrics.get('threshold', 0.0))
        wandb_run.summary[f"{mode_key}/fold{fold+1}/mrr"] = float(final_metrics.get('mrr', 0.0))
        wandb_run.summary[f"{mode_key}/fold{fold+1}/hit@1"] = float(final_metrics.get('hit@1', 0.0))
        wandb_run.summary[f"{mode_key}/fold{fold+1}/hit@3"] = float(final_metrics.get('hit@3', 0.0))
        wandb_run.summary[f"{mode_key}/fold{fold+1}/hit@5"] = float(final_metrics.get('hit@5', 0.0))
    
    if use_checkpoint and os.path.exists(model_path):
        os.remove(model_path)
    
    return final_metrics

if __name__ == "__main__":
    config = Config()
    run_experiment(config)