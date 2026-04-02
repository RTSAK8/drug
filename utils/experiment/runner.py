import pandas as pd
import numpy as np
import torch

from config import Config
from data_manager import DataManager

from .data_prep import set_seed, prepare_triplets_data
from .init_wandb import _init_wandb_run
from .mode_runner import run_mode_experiment


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def run_experiment(config: Config):
    set_seed(config.random_state)

    data_manager = DataManager(config)

    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    data = data_manager.load_all_data()
    data_manager.create_index_mappings(data)

    print("处理实体特征...")
    drug_fea = data_manager.process_entity_features(data["drug"], "Drug_ID", data_manager.drug_to_idx)
    gene_fea = data_manager.process_entity_features(data["gene"], "Gene_ID", data_manager.gene_to_idx)
    disease_fea = data_manager.process_entity_features(data["disease"], "Disease_ID", data_manager.disease_to_idx)

    print("准备三元组数据...")
    train_triplets_df = prepare_triplets_data(data["triplets"], data_manager)
    test_triplets_df = prepare_triplets_data(data.get("drugbank_triplets", pd.DataFrame()), data_manager)

    if test_triplets_df.empty:
        print("警告: drugbank_triplets 为空，将无法进行独立测试评估")

    global_pos_set = set(tuple(x) for x in train_triplets_df[["Drug_ID", "Gene_ID", "Disease_ID"]].values)
    global_pos_set |= set(tuple(x) for x in test_triplets_df[["Drug_ID", "Gene_ID", "Disease_ID"]].values)

    print("准备二元边...")
    global_edges = data_manager.prepare_binary_edges(data)

    print(f"边统计:")
    for edge_type, edge_data in global_edges.items():
        if isinstance(edge_data, np.ndarray):
            print(f"  {edge_type}: {edge_data.shape[1]} edges")
        else:
            print(f"  {edge_type}: {len(edge_data)} hyperedges")

    modes = [["gene"], ["disease"]]

    wandb_run = _init_wandb_run(
        config=config,
        data_manager=data_manager,
        ablation_count=len(config.graph_ablation_configs),
    )
    try:
        for mode_index, current_mode in enumerate(modes):
            run_mode_experiment(
                current_mode=current_mode,
                mode_index=mode_index,
                train_triplets_df=train_triplets_df,
                test_triplets_df=test_triplets_df,
                global_pos_set=global_pos_set,
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
