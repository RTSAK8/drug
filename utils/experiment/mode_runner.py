import numpy as np
import torch

from train_utils import (
    build_hypergraph,
    build_binary_edges,
    apply_offsets,
    generate_negatives_for_fold,
    save_summary_results,
)

from .training_loop import train_fold_model


def run_mode_experiment(
    current_mode,
    mode_index,
    train_triplets_df,
    test_triplets_df,
    global_pos_set,
    global_edges,
    data_manager,
    drug_fea,
    gene_fea,
    disease_fea,
    config,
    wandb_run=None,
):
    """运行单个模式的实验（triplets 全量训练 + drugbank 独立测试）"""
    print(f"{'='*60}")
    print(f"开始模式: {current_mode[0]} 留出")
    print(f"{'='*60}")

    total_ablations = len(config.graph_ablation_configs)

    for ablation_index, (ablation_name, include_relations) in enumerate(config.graph_ablation_configs.items()):
        print(f"消融实验: {ablation_name}")
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

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.summary[f"{mode_key}/meta/train_pos_count"] = int(len(train_df))
            wandb_run.summary[f"{mode_key}/meta/test_pos_count"] = int(len(test_df))
            wandb_run.summary[f"{mode_key}/meta/train_neg_count"] = int(len(train_neg))
            wandb_run.summary[f"{mode_key}/meta/test_neg_count"] = int(len(test_neg))
            wandb_run.summary[f"{mode_key}/meta/train_drug_pool"] = int(len(train_drug_pool))
            wandb_run.summary[f"{mode_key}/meta/train_gene_pool"] = int(len(train_gene_pool))
            wandb_run.summary[f"{mode_key}/meta/train_disease_pool"] = int(len(train_disease_pool))

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

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.summary[f"{mode_key}/meta/hypergraph_train_pos_incidence"] = int(hypergraph_train_pos.shape[1])
            wandb_run.summary[f"{mode_key}/meta/hypergraph_train_neg_incidence"] = int(hypergraph_train_neg.shape[1])
            wandb_run.summary[f"{mode_key}/meta/binary_edge_count"] = int(bin_edge_index.shape[1])
            wandb_run.summary[f"{mode_key}/meta/binary_edge_type_count"] = int(bin_edge_type.shape[0])

        test_metrics = train_fold_model(
            current_mode=current_mode,
            mode_index=mode_index,
            total_ablations=total_ablations,
            ablation_index=ablation_index,
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
            wandb_run=wandb_run,
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_metrics = {f"{k}_mean": float(v) for k, v in test_metrics.items()}
        avg_metrics.update({f"{k}_std": 0.0 for k in test_metrics.keys()})

        print(f"{'='*60}")
        print(f"{ablation_name} 独立测试集(DrugBank)结果:")
        print(f"{'='*60}")
        print(f"AUC:  {avg_metrics['auc_mean']:.4f}")
        print(f"AUPR: {avg_metrics['aupr_mean']:.4f}")
        print(f"F1:   {avg_metrics['f1_score_mean']:.4f}")
        if "mrr_mean" in avg_metrics:
            print(f"MRR:  {avg_metrics['mrr_mean']:.4f}")
        if "hit@1_mean" in avg_metrics:
            print(f"Hit@1:{avg_metrics['hit@1_mean']:.4f}")
        if "hit@3_mean" in avg_metrics:
            print(f"Hit@3:{avg_metrics['hit@3_mean']:.4f}")
        if "hit@5_mean" in avg_metrics:
            print(f"Hit@5:{avg_metrics['hit@5_mean']:.4f}")

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.summary[f"{mode_key}/test/auc"] = float(test_metrics.get("auc", 0.0))
            wandb_run.summary[f"{mode_key}/test/aupr"] = float(test_metrics.get("aupr", 0.0))
            wandb_run.summary[f"{mode_key}/test/f1_score"] = float(test_metrics.get("f1_score", 0.0))
            wandb_run.summary[f"{mode_key}/test/threshold"] = float(test_metrics.get("threshold", 0.0))
            wandb_run.summary[f"{mode_key}/test/mrr"] = float(test_metrics.get("mrr", 0.0))
            wandb_run.summary[f"{mode_key}/test/hit@1"] = float(test_metrics.get("hit@1", 0.0))
            wandb_run.summary[f"{mode_key}/test/hit@3"] = float(test_metrics.get("hit@3", 0.0))
            wandb_run.summary[f"{mode_key}/test/hit@5"] = float(test_metrics.get("hit@5", 0.0))

        save_summary_results(
            mode=current_mode[0],
            ablation_name=ablation_name,
            avg_metrics=avg_metrics,
            all_fold_metrics=[test_metrics],
            config=vars(config),
        )
