import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import KFold

from utils.hypergraph import build_hypergraph, build_binary_edges
from utils.preprocessing import (
    apply_offsets,
    generate_negatives_for_fold,
    generate_negatives_with_query_disease,
)
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

    # 评估负样本额外记录其来源正查询的真实 Disease_ID。
    # missing-disease 分类统计必须按真实查询疾病分组，而不是按负候选疾病分组。
    test_neg, test_neg_query_disease_ids = generate_negatives_with_query_disease(
        pos_samples=test_df[["Drug_ID", "Gene_ID", "Disease_ID"]].values,
        drug_pool=train_drug_pool,
        gene_pool=train_gene_pool,
        disease_pool=train_disease_pool,
        global_exclude=global_pos_set.copy(),
        replace_dims=current_mode,
        neg_ratio=config.neg_ratio,
        max_attempts=int(getattr(config, "max_attempts", 50)),
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

    # 正样本的查询疾病就是其真实疾病；负样本使用负采样时保存的来源疾病。
    test_pos_query_disease_ids = test_df["Disease_ID"].to_numpy(dtype=np.int64)
    test_query_disease_ids = np.concatenate(
        [test_pos_query_disease_ids, test_neg_query_disease_ids]
    )
    if len(test_query_disease_ids) != len(test_data):
        raise RuntimeError(
            "test query-disease metadata is not aligned with test_data: "
            f"{len(test_query_disease_ids)} vs {len(test_data)}"
        )

    np.random.shuffle(train_data)
    test_perm = np.random.permutation(len(test_data))
    test_data = test_data[test_perm]
    test_query_disease_ids = test_query_disease_ids[test_perm]

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
        "test_query_disease_ids": test_query_disease_ids,
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
    for name in [
        "auc",
        "aupr",
        "f1_score",
        "mrr",
        "hit@1",
        "hit@3",
        "hit@5",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@50",
        "recall@100",
    ]:
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
    evaluation_label=None,
):
    """运行单个模式的一次 train-test 实验。"""
    print(f"{'='*60}")
    evaluation_label = evaluation_label or "DrugBank external test"
    print(f"开始模式: {current_mode[0]} 留出 | {evaluation_label}")
    print(f"{'='*60}")

    total_ablations = len(config.graph_ablation_configs)

    for ablation_index, (ablation_name, include_relations) in enumerate(config.graph_ablation_configs.items()):
        print(f"消融实验: {ablation_name}")
        print(f"包含的关系: {include_relations if include_relations else '仅三元组'}")

        offset_edges = data_manager.get_offset_edges(global_edges, include_relations)

        if test_triplets_df.empty:
            print(f"跳过：{evaluation_label} 测试集为空")
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
            val_query_disease_ids=fold_data["test_query_disease_ids"],
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
            disease_offset=data_manager.disease_offset,
            disease_idx_to_id={idx: raw_id for raw_id, idx in data_manager.disease_to_idx.items()},
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_metrics = {f"{k}_mean": float(v) for k, v in test_metrics.items()}
        avg_metrics.update({f"{k}_std": 0.0 for k in test_metrics.keys()})

        print(f"{'='*60}")
        print(f"{ablation_name} {evaluation_label} 结果:")
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
            wandb_run.summary[f"{mode_key}.test.recall@1"] = float(test_metrics.get("recall@1", 0.0))
            wandb_run.summary[f"{mode_key}.test.recall@5"] = float(test_metrics.get("recall@5", 0.0))
            wandb_run.summary[f"{mode_key}.test.recall@10"] = float(test_metrics.get("recall@10", 0.0))
            wandb_run.summary[f"{mode_key}.test.recall@50"] = float(test_metrics.get("recall@50", 0.0))
            wandb_run.summary[f"{mode_key}.test.recall@100"] = float(test_metrics.get("recall@100", 0.0))

        save_summary_results(
            mode=current_mode[0],
            ablation_name=ablation_name,
            avg_metrics=avg_metrics,
            all_fold_metrics=[test_metrics],
            config=vars(config),
            results_dir=getattr(config, "results_dir", "results"),
        )



def _normalise_raw_id(value):
    """将 CSV 中的数值 ID 规范化为无 .0 的字符串。"""
    if pd.isna(value):
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _load_radar_class_mapping(mapping_path, class_column):
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Disease mapping file not found: {path}")
    frame = pd.read_csv(path)
    if "Disease_ID" not in frame.columns:
        raise ValueError(f"Mapping file must contain Disease_ID: {path}")
    if class_column not in frame.columns:
        raise ValueError(
            f"Mapping file does not contain {class_column!r}; "
            f"available columns: {list(frame.columns)}"
        )
    result = {}
    for _, row in frame.iterrows():
        key = _normalise_raw_id(row["Disease_ID"])
        category = str(row[class_column]).strip() if pd.notna(row[class_column]) else "Others"
        if key:
            result[key] = category or "Others"
    return result


def build_category_stratified_holdout(all_triplets_df, data_manager, config):
    """从 triplets.csv 构造一次按雷达图类别分层的留出集。

    该拆分只用于类别雷达图，使稀有类别在一次评估中仍有至少一个真实查询。
    它不替代论文中的 DrugBank external test 结果。
    """
    if all_triplets_df.empty:
        raise ValueError("triplets.csv is empty")

    class_column = str(getattr(config, "radar_class_column", "Paper_Radar_Class_EN"))
    class_by_raw_id = _load_radar_class_mapping(
        getattr(config, "disease_mapping_file", "./resources/disease_icd11_mapping.csv"),
        class_column,
    )
    idx_to_raw = {int(idx): raw_id for raw_id, idx in data_manager.disease_to_idx.items()}

    work = all_triplets_df.copy().reset_index(drop=True)
    categories = []
    for local_id in work["Disease_ID"].tolist():
        raw_id = _normalise_raw_id(idx_to_raw.get(int(local_id), local_id))
        categories.append(class_by_raw_id.get(raw_id, "Others"))
    work["_radar_category"] = categories

    fraction = float(getattr(config, "radar_holdout_fraction", 0.20))
    if not 0.0 < fraction < 1.0:
        raise ValueError("radar_holdout_fraction must be between 0 and 1")
    min_per_class = max(1, int(getattr(config, "radar_min_test_per_class", 1)))
    seed = int(getattr(config, "random_state", 42))
    rng = np.random.default_rng(seed)

    preferred_order = list(getattr(config, "radar_class_order", []))
    observed = work["_radar_category"].dropna().astype(str).unique().tolist()
    category_order = [c for c in preferred_order if c in observed]
    category_order += sorted(c for c in observed if c not in category_order)

    test_indices = []
    split_rows = []
    for category in category_order:
        indices = work.index[work["_radar_category"] == category].to_numpy(dtype=np.int64)
        if len(indices) == 0:
            continue
        requested = max(min_per_class, int(np.ceil(len(indices) * fraction)))
        # 类别只有 1 个三元组时允许该三元组进入测试；其它类别至少保留 1 条训练样本。
        n_test = 1 if len(indices) == 1 else min(requested, len(indices) - 1)
        selected = rng.choice(indices, size=n_test, replace=False)
        test_indices.extend(int(x) for x in selected)
        split_rows.append((category, len(indices), n_test))

    test_indices = sorted(set(test_indices))
    if not test_indices:
        raise RuntimeError("No rows were selected for category-stratified holdout")

    test_df = work.loc[test_indices].drop(columns=["_radar_category"]).reset_index(drop=True)
    train_df = work.drop(index=test_indices).drop(columns=["_radar_category"]).reset_index(drop=True)

    print("疾病类别分层留出统计:")
    for category, total, test_count in split_rows:
        print(f"  {category:<16} total={total:4d}, test={test_count:3d}, train={total-test_count:4d}")
    print(f"  合计: train={len(train_df)}, test={len(test_df)}, seed={seed}")
    return train_df, test_df


def run_mode_category_stratified_holdout(
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
    """运行一次疾病类别分层留出，仅用于生成类别覆盖更完整的雷达图。"""
    train_df, test_df = build_category_stratified_holdout(
        all_triplets_df=all_triplets_df,
        data_manager=data_manager,
        config=config,
    )
    return run_mode_experiment(
        current_mode=current_mode,
        mode_index=mode_index,
        train_triplets_df=train_df,
        test_triplets_df=test_df,
        global_pos_set=global_pos_set,
        global_edges=global_edges,
        data_manager=data_manager,
        drug_fea=drug_fea,
        gene_fea=gene_fea,
        disease_fea=disease_fea,
        config=config,
        wandb_run=wandb_run,
        evaluation_label="category-stratified holdout",
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
                val_query_disease_ids=fold_data["test_query_disease_ids"],
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
                disease_offset=data_manager.disease_offset,
                disease_idx_to_id={idx: raw_id for raw_id, idx in data_manager.disease_to_idx.items()},
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
            results_dir=getattr(config, "results_dir", "results"),
        )