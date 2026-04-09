import pandas as pd

from data_manager import DataManager
from utils.runtime import set_global_seed


def set_seed(seed: int = 42):
    """设置随机种子"""
    set_global_seed(seed)


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
