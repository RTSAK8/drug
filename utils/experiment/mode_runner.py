import numpy as np
import torch
from sklearn.model_selection import KFold

from utils.hypergraph import build_hypergraph, build_binary_edges
from utils.preprocessing import apply_offsets, generate_negatives_for_fold
from utils.results import save_summary_results
from utils.runtime import set_global_seed

from .training_loop import train_fold_model


def _build_fold_data(
    current_mode,
    train_df,
    test_df,
    global_pos_set,
    offset_edges,
    data_manager,
    config,
):
    """构建单个 fold / 单次 train-test 实验所需的数据。"""
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

    # 注意：超图仅由训练集构建，避免测试 fold / 独立测试集信息泄露。
    hypergraph_train_pos = build_hypergraph(
        data=train_pos_offset,
        gene_hyperedges=offset_edges["gene_hyperedges"],
    )
    hypergraph_train_neg = build_hypergraph(
        data=train_neg_offset,
        gene_hyperedges=offset_edges["gene_hyperedges"],
    )

    # 二元关系来自外部关系网络，作为固定辅助图使用。
    bin_edge_index, bin_edge_type = build_binary_edges(
        ppi_data=offset_edges["ppi"],
        dpi_data=offset_edges["dpi"],
        ddi_data=offset_edges["ddi"],
        protein_disease_data=offset_edges["protein_disease"],
        make_undirected=True,
    )

    return {
        "train_drug_pool": train_drug_pool,
        "train_gene_pool": train_gene_pool,
        "train_disease_pool": train_disease_pool,
        "train_neg": train_neg,
        "test_neg": test_neg,
        "train_pos_offset": train_pos_offset,
        "test_pos_offset": test_pos_offset,
        "train_data": train_data,
        "test_data": test_data,
        "hypergraph_train_pos": hypergraph_train_pos,
        "hypergraph_train_neg": hypergraph_train_neg,
        "bin_edge_index": bin_edge_index,
        "bin_edge_type": bin_edge_type,
    }


def _metric_mean_std(all_fold_metrics):
    """计算多折结果的 mean/std。"""
    avg_metrics = {}
    if not all_fold_metrics:
        return avg_metrics

    metric_names = all_fold_metrics[0].keys()
    for metric_name in metric_names:
        values = np.array([float(m.get(metric_name, 0.0)) for m in all_fold_metrics], dtype=float)
        avg_metrics[f"{metric_name}_mean"] = float(values.mean())
        avg_metrics[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return avg_metrics


def _print_avg_metrics(avg_metrics):
    """按统一格式打印汇总指标。"""
    for name in ["auc", "aupr", "f1_score", "mrr", "hit@1", "hit@3", "hit@5"]:
        mean_key = f"{name}_mean"
        std_key = f"{name}_std"
        if mean_key in avg_metrics:
            print(f"{name}: {avg_metrics[mean_key]:.4f} ± {avg_metrics.get(std_key, 0.0):.4f}")


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
    """运行单个模式的实验（triplets 全量训练 + drugbank 独立测试）。"""
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

        fold_data = _build_fold_data(
            current_mode=current_mode,
            train_df=train_df,
            test_df=test_df,
            global_pos_set=global_pos_set,
            offset_edges=offset_edges,
            data_manager=data_manager,
            config=config,
        )

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.summary[f"{mode_key}.meta.train_pos_count"] = int(len(train_df))
            wandb_run.summary[f"{mode_key}.meta.test_pos_count"] = int(len(test_df))
            wandb_run.summary[f"{mode_key}.meta.train_neg_count"] = int(len(fold_data["train_neg"]))
            wandb_run.summary[f"{mode_key}.meta.test_neg_count"] = int(len(fold_data["test_neg"]))
            wandb_run.summary[f"{mode_key}.meta.train_drug_pool"] = int(len(fold_data["train_drug_pool"]))
            wandb_run.summary[f"{mode_key}.meta.train_gene_pool"] = int(len(fold_data["train_gene_pool"]))
            wandb_run.summary[f"{mode_key}.meta.train_disease_pool"] = int(len(fold_data["train_disease_pool"]))
            wandb_run.summary[f"{mode_key}.meta.hypergraph_train_pos_incidence"] = int(fold_data["hypergraph_train_pos"].shape[1])
            wandb_run.summary[f"{mode_key}.meta.hypergraph_train_neg_incidence"] = int(fold_data["hypergraph_train_neg"].shape[1])
            wandb_run.summary[f"{mode_key}.meta.boundary_edge_count"] = int(fold_data["bin_edge_index"].shape[1])
            wandb_run.summary[f"{mode_key}.meta.boundary_edge_type_count"] = int(fold_data["bin_edge_type"].shape[0])

        test_metrics = train_fold_model(
            current_mode=current_mode,
            mode_index=mode_index,
            total_ablations=total_ablations,
            ablation_index=ablation_index,
            ablation_name=ablation_name,
            fold=0,
            train_data=fold_data["train_data"],
            val_data=fold_data["test_data"],
            train_pos=fold_data["train_pos_offset"],
            val_pos=fold_data["test_pos_offset"],
            hypergraph_pos=fold_data["hypergraph_train_pos"],
            hypergraph_neg=fold_data["hypergraph_train_neg"],
            bin_edge_index=fold_data["bin_edge_index"],
            bin_edge_type=fold_data["bin_edge_type"],
            drug_fea=drug_fea,
            gene_fea=gene_fea,
            disease_fea=disease_fea,
            config=config,
            use_checkpoint=False,
            wandb_run=wandb_run,
            total_folds=1,
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_metrics = {f"{k}_mean": float(v) for k, v in test_metrics.items()}
        avg_metrics.update({f"{k}_std": 0.0 for k in test_metrics.keys()})

        print(f"{'='*60}")
        print(f"{ablation_name} 独立测试集(DrugBank)结果:")
        print(f"{'='*60}")
        _print_avg_metrics(avg_metrics)

        if wandb_run is not None:
            mode_key = current_mode[0]
            wandb_run.summary[f"{mode_key}.test.auc"] = float(test_metrics.get("auc", 0.0))
            wandb_run.summary[f"{mode_key}.test.aupr"] = float(test_metrics.get("aupr", 0.0))
            wandb_run.summary[f"{mode_key}.test.f1_score"] = float(test_metrics.get("f1_score", 0.0))
            wandb_run.summary[f"{mode_key}.test.threshold"] = float(test_metrics.get("threshold", 0.0))
            wandb_run.summary[f"{mode_key}.test.mrr"] = float(test_metrics.get("mrr", 0.0))
            wandb_run.summary[f"{mode_key}.test.hit@1"] = float(test_metrics.get("hit@1", 0.0))
            wandb_run.summary[f"{mode_key}.test.hit@3"] = float(test_metrics.get("hit@3", 0.0))
            wandb_run.summary[f"{mode_key}.test.hit@5"] = float(test_metrics.get("hit@5", 0.0))

        save_summary_results(
            mode=current_mode[0],
            ablation_name=ablation_name,
            avg_metrics=avg_metrics,
            all_fold_metrics=[test_metrics],
            config=vars(config),
        )


def run_mode_cross_validation(
    current_mode,
    mode_index,
    all_triplets_df,
    global_pos_set,
    global_edges,
    data_manager,
    drug_fea,
    gene_fea,
    disease_fea,
    config,
    wandb_run=None,
):
    """运行单个模式的 K 折交叉验证（仅对 triplets.csv 内部划分）。"""
    n_splits = int(getattr(config, "n_splits", 5))

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(all_triplets_df) < n_splits:
        raise ValueError(f"样本数 {len(all_triplets_df)} 小于 n_splits={n_splits}")

    print(f"{'='*60}")
    print(f"开始模式: {current_mode[0]} 留出 | {n_splits}-fold cross-validation")
    print(f"{'='*60}")

    total_ablations = len(config.graph_ablation_configs)
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(getattr(config, "random_state", 42)),
    )
    indices = np.arange(len(all_triplets_df))

    for ablation_index, (ablation_name, include_relations) in enumerate(config.graph_ablation_configs.items()):
        print(f"消融实验: {ablation_name}")
        print(f"包含的关系: {include_relations if include_relations else '仅三元组'}")

        offset_edges = data_manager.get_offset_edges(global_edges, include_relations)
        all_fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(indices)):
            fold_seed = int(getattr(config, "random_state", 42)) + fold
            set_global_seed(fold_seed)

            train_df = all_triplets_df.iloc[train_idx].reset_index(drop=True)
            test_df = all_triplets_df.iloc[test_idx].reset_index(drop=True)

            print(f"\nFold {fold + 1}/{n_splits}")
            print(f"train_pos={len(train_df)}, test_pos={len(test_df)}, seed={fold_seed}")

            fold_data = _build_fold_data(
                current_mode=current_mode,
                train_df=train_df,
                test_df=test_df,
                global_pos_set=global_pos_set,
                offset_edges=offset_edges,
                data_manager=data_manager,
                config=config,
            )

            if wandb_run is not None:
                mode_key = current_mode[0]
                wandb_run.summary[f"{mode_key}.cv.n_splits"] = int(n_splits)
                wandb_run.summary[f"{mode_key}.cv.fold{fold + 1}.train_pos_count"] = int(len(train_df))
                wandb_run.summary[f"{mode_key}.cv.fold{fold + 1}.test_pos_count"] = int(len(test_df))
                wandb_run.summary[f"{mode_key}.cv.fold{fold + 1}.train_neg_count"] = int(len(fold_data["train_neg"]))
                wandb_run.summary[f"{mode_key}.cv.fold{fold + 1}.test_neg_count"] = int(len(fold_data["test_neg"]))

            fold_metrics = train_fold_model(
                current_mode=current_mode,
                mode_index=mode_index,
                total_ablations=total_ablations,
                ablation_index=ablation_index,
                ablation_name=ablation_name,
                fold=fold,
                train_data=fold_data["train_data"],
                val_data=fold_data["test_data"],
                train_pos=fold_data["train_pos_offset"],
                val_pos=fold_data["test_pos_offset"],
                hypergraph_pos=fold_data["hypergraph_train_pos"],
                hypergraph_neg=fold_data["hypergraph_train_neg"],
                bin_edge_index=fold_data["bin_edge_index"],
                bin_edge_type=fold_data["bin_edge_type"],
                drug_fea=drug_fea,
                gene_fea=gene_fea,
                disease_fea=disease_fea,
                config=config,
                use_checkpoint=False,
                wandb_run=wandb_run,
                total_folds=n_splits,
            )
            all_fold_metrics.append(fold_metrics)

            if wandb_run is not None:
                mode_key = current_mode[0]
                for metric_name, metric_value in fold_metrics.items():
                    wandb_run.summary[f"{mode_key}.cv.fold{fold + 1}.{metric_name}"] = float(metric_value)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_metrics = _metric_mean_std(all_fold_metrics)

        print(f"\n{'='*60}")
        print(f"{ablation_name} {n_splits}-fold CV 汇总结果 | mode={current_mode[0]}")
        print(f"{'='*60}")
        _print_avg_metrics(avg_metrics)

        if wandb_run is not None:
            mode_key = current_mode[0]
            for metric_name, metric_value in avg_metrics.items():
                wandb_run.summary[f"{mode_key}.cv.{metric_name}"] = float(metric_value)

        save_summary_results(
            mode=current_mode[0],
            ablation_name=ablation_name,
            avg_metrics=avg_metrics,
            all_fold_metrics=all_fold_metrics,
            config=vars(config),
        )