import os

import numpy as np
import torch
import torch.nn as nn

from config import Config
from models import BioEncoder, HgnnEncoder, RelationalGraphEncoder, Decoder, HGNN
from utils.runtime import get_device
from utils.metrics import metrics_graph_with_hits, calculate_hits_ranking_based
from utils.training import train_step, validate_step


# 设置设备
device = get_device()


def train_fold_model(
    current_mode,
    mode_index: int,
    total_ablations: int,
    ablation_index: int,
    ablation_name: str,
    fold: int,
    train_data: np.ndarray,
    val_data: np.ndarray,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    hypergraph_pos: torch.Tensor,
    hypergraph_neg: torch.Tensor,
    bin_edge_index: torch.Tensor,
    bin_edge_type: torch.Tensor,
    drug_fea: torch.Tensor,
    gene_fea: torch.Tensor,
    disease_fea: torch.Tensor,
    config: Config,
    use_checkpoint: bool = True,
    wandb_run=None,
):
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

    pool_start_weights = getattr(config, "qccl_pool_start_weights", [1.0, 1.0, 1.0])
    pool_end_weights = getattr(config, "qccl_pool_end_weights", [1.0, 1.0, 1.0])
    start_mid_weight = float(pool_start_weights[1]) if isinstance(pool_start_weights, (list, tuple)) and len(pool_start_weights) > 1 else 1.0
    end_mid_weight = float(pool_end_weights[1]) if isinstance(pool_end_weights, (list, tuple)) and len(pool_end_weights) > 1 else 1.0
    start_late_weight = float(pool_start_weights[2]) if isinstance(pool_start_weights, (list, tuple)) and len(pool_start_weights) > 2 else 1.0
    end_late_weight = float(pool_end_weights[2]) if isinstance(pool_end_weights, (list, tuple)) and len(pool_end_weights) > 2 else 1.0

    qccl_config = {
        "temperature": float(getattr(config, "qccl_temperature", 0.3)),
        "gene_query_temperature": None if float(getattr(config, "qccl_gene_query_temperature", -1.0)) <= 0 else float(getattr(config, "qccl_gene_query_temperature", -1.0)),
        "drug_query_temperature": None if float(getattr(config, "qccl_drug_query_temperature", -1.0)) <= 0 else float(getattr(config, "qccl_drug_query_temperature", -1.0)),
        "disease_query_temperature": None if float(getattr(config, "qccl_disease_query_temperature", -1.0)) <= 0 else float(getattr(config, "qccl_disease_query_temperature", -1.0)),
        "norm_mode": str(getattr(config, "qccl_asymmetric_norm_mode", "symmetric")),
        "multi_positive_mode": "group" if bool(getattr(config, "qccl_multi_positive", False)) else "strict",
        "suppress_false_negatives": bool(getattr(config, "qccl_multi_positive", False)),
        "use_hardness_reweight": bool(getattr(config, "qccl_hardness_reweight", False)),
        "hardness_alpha": float(getattr(config, "qccl_hardness_alpha", 1.0)),
        "hardness_tau": float(getattr(config, "qccl_hardness_temperature", 1.0)),
        "use_query_hardness_reweight": bool(getattr(config, "qccl_query_hardness_reweight", False)),
        "query_hardness_alpha": float(getattr(config, "qccl_query_hardness_alpha", 1.0)),
        "query_hardness_tau": float(getattr(config, "qccl_query_hardness_temperature", 1.0)),
        "enable_three_pool_schedule": bool(getattr(config, "qccl_three_pool_schedule", False)),
        "pool_schedule_start": float(getattr(config, "qccl_pool_easy_quantile", 0.33)),
        "pool_schedule_end": float(getattr(config, "qccl_pool_hard_quantile", 0.66)),
        "pool_schedule_mid_weight": max(0.0, end_mid_weight - start_mid_weight),
        "pool_schedule_late_weight": max(0.0, end_late_weight - start_late_weight),
        "enable_memory_queue": bool(getattr(config, "qccl_memory_queue", False)),
        "memory_queue_size": int(getattr(config, "qccl_memory_queue_size", 0)),
        "memory_queue_warmup_steps": int(round(max(0.0, min(1.0, float(getattr(config, "qccl_memory_queue_warmup_progress", 0.0)))) * max(1, int(config.epochs)))),
        "memory_queue_negatives_per_query": int(getattr(config, "qccl_memory_queue_negatives_per_query", 0)),
        "memory_queue_unique_only": True,
        "cond_dim_ratio": float(getattr(config, "qccl_cond_dim_ratio", 0.5)),
        "num_layers": int(getattr(config, "qccl_num_layers", 2)),
        "query_dropout": float(getattr(config, "qccl_query_dropout", 0.3)),
        "film_layer_norm": bool(getattr(config, "qccl_film_layer_norm", False)),
    }

    model = HGNN(
        bio_encoder=bio_encoder,
        graph_encoder=graph_encoder,
        binary_graph_encoder=binary_graph_encoder,
        decoder=decoder,
        out_channels=config.hidden_dim // 4,
        replace_dims=current_mode,
        loss_type=config.loss_type,
        fusion_method=getattr(config, "fusion_method", "gate"),
        qccl_config=qccl_config,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.L2)

    best_mrr = 0
    model_path = f"best_model_{current_mode[0]}_{ablation_name}_fold{fold}.pth"

    for epoch in range(config.epochs):
        global_step = mode_index * (total_ablations * config.epochs) + ablation_index * config.epochs + epoch + 1
        epoch_progress = float(epoch) / float(max(config.epochs - 1, 1))

        train_metrics = train_step(
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
            epoch_progress=epoch_progress,
            alpha=config.alpha,
        )

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.log(
                {
                    f"{mode_key}/train_total_loss": float(train_metrics["loss"]),
                    f"{mode_key}/train_main_loss": float(train_metrics["main_loss"]),
                    f"{mode_key}/train_cl_loss": float(train_metrics["cl_loss"]),
                    f"{mode_key}/train_mrr": float(train_metrics["mrr"]),
                    f"{mode_key}/train_hit@1": float(train_metrics["hit@1"]),
                    f"{mode_key}/train_hit@3": float(train_metrics["hit@3"]),
                    f"{mode_key}/train_hit@5": float(train_metrics["hit@5"]),
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
                bin_edge_index=bin_edge_index,
                bin_edge_type=bin_edge_type,
                val_data=val_data,
                val_pos=val_pos,
                epoch_progress=epoch_progress,
            )

            print(f"Epoch {epoch+1:03d} | Val: AUC={auc:.4f}, AUPR={aupr:.4f}, " f"F1={f1_score:.4f}, thresh={threshold:.4f}, ")

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
            1.0,
        )

        y_true = val_data[:, 3]
        y_pred = pred.cpu().numpy()

        auc, aupr, best_f1_score, threshold, _ = metrics_graph_with_hits(y_true, y_pred)
        rank_hits = calculate_hits_ranking_based(ranks, [1, 3, 5])

        final_metrics = {
            "auc": auc,
            "aupr": aupr,
            "f1_score": best_f1_score,
            "threshold": threshold,
            "mrr": float(mrr),
            "hit@1": float(rank_hits["hit@1"]),
            "hit@3": float(rank_hits["hit@3"]),
            "hit@5": float(rank_hits["hit@5"]),
        }

        print(f"\n最终测试结果:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

    if use_checkpoint and os.path.exists(model_path):
        os.remove(model_path)

    return final_metrics
