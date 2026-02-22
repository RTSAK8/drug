"""
数据管理器 - 加载和处理药物-基因-疾病数据
"""

import numpy as np
import pandas as pd
import torch
import os
from typing import Dict, List, Tuple, Optional
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self.drug_to_idx = {}
        self.gene_to_idx = {}
        self.disease_to_idx = {}
        self.gene_offset = 0
        self.disease_offset = 0

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据文件"""
        data_files = {
            "drug": self.config.drug_file,
            "gene": self.config.gene_file,
            "disease": self.config.disease_file,
            "triplets": self.config.triplets_file,
            "drugbank_triplets": getattr(self.config, "drugbank_triplets_file", ""),
            "ppi": self.config.ppi_file,
            "dpi": self.config.dpi_file,
            "ddi": self.config.ddi_file,
            "protein_disease": self.config.protein_disease_file,
            "gene_hyperedges": self.config.gene_hyperedges_file,
        }

        data = {}
        for key, filepath in data_files.items():
            if os.path.exists(filepath):
                data[key] = pd.read_csv(filepath)
                print(f"加载 {key}: {data[key].shape}")
            else:
                print(f"警告: 文件不存在 {filepath}")
                data[key] = pd.DataFrame()

        return data

    def create_index_mappings(self, data: Dict[str, pd.DataFrame]):
        """创建实体到索引的映射"""
        # 药物映射
        if not data["drug"].empty:
            self.drug_to_idx = {d: idx for idx, d in enumerate(data["drug"]["Drug_ID"].unique())}

        # 基因映射
        if not data["gene"].empty:
            self.gene_to_idx = {g: idx for idx, g in enumerate(data["gene"]["Gene_ID"].unique())}

        # 疾病映射
        if not data["disease"].empty:
            self.disease_to_idx = {d: idx for idx, d in enumerate(data["disease"]["Disease_ID"].unique())}

        # 计算偏移量
        self.gene_offset = len(self.drug_to_idx)
        self.disease_offset = self.gene_offset + len(self.gene_to_idx)

        print(f"索引映射统计:")
        print(f"  药物数量: {len(self.drug_to_idx)}")
        print(f"  基因数量: {len(self.gene_to_idx)}")
        print(f"  疾病数量: {len(self.disease_to_idx)}")
        print(f"  基因偏移: {self.gene_offset}")
        print(f"  疾病偏移: {self.disease_offset}")

    def process_entity_features(self, df: pd.DataFrame, id_col: str, idx_mapping: Dict) -> torch.Tensor:
        """统一的实体特征处理函数"""
        if df.empty:
            num_entities = len(idx_mapping)
            num_features = 0
            return torch.zeros((num_entities, 128)).to(device)  # 默认特征维度

        processed = df.set_index(id_col).reset_index().drop_duplicates(subset=id_col, keep="first").set_index(id_col)
        feature_cols = [c for c in processed.columns if c != id_col]

        # 构建特征矩阵
        num_entities = len(idx_mapping)
        num_features = len(feature_cols)
        feature_matrix = np.zeros((num_entities, num_features))

        for entity, idx in idx_mapping.items():
            if entity in processed.index:
                entity_data = processed.loc[[entity]].head(1)
                feature_vector = entity_data[feature_cols].values.flatten()
                feature_matrix[idx] = feature_vector

        return torch.FloatTensor(feature_matrix).to(device)

    def prepare_binary_edges(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """准备所有二元关系边"""
        edges = {}

        # 药物-药物相互作用
        edges["ddi"] = self._prepare_edges(data["ddi"], "Drug_ID_1", "Drug_ID_2", self.drug_to_idx, self.drug_to_idx)

        # 蛋白质-蛋白质相互作用
        edges["ppi"] = self._prepare_edges(data["ppi"], "Gene_ID_1", "Gene_ID_2", self.gene_to_idx, self.gene_to_idx)

        # 药物-蛋白质相互作用
        edges["dpi"] = self._prepare_edges(data["dpi"], "Drug_ID", "Gene_ID", self.drug_to_idx, self.gene_to_idx)

        edges["protein_disease"] = self._prepare_edges(data.get("protein_disease", pd.DataFrame()), "Gene_ID", "Disease_ID", self.gene_to_idx, self.disease_to_idx)

        # 基因功能超边，暂时不用
        edges["gene_hyperedges"] = self._prepare_gene_hyperedges(data["gene_hyperedges"])

        return edges

    def _prepare_edges(self, df: pd.DataFrame, col1: str, col2: str, mapper1: Dict, mapper2: Dict) -> np.ndarray:
        """准备二元边"""
        if df.empty or col1 not in df.columns or col2 not in df.columns:
            return np.empty((2, 0), dtype=np.int64)

        # 向量化映射
        mapped_col1 = df[col1].map(mapper1).fillna(-1).astype(np.int64)
        mapped_col2 = df[col2].map(mapper2).fillna(-1).astype(np.int64)

        # 过滤无效映射
        valid_mask = (mapped_col1 != -1) & (mapped_col2 != -1)

        if not valid_mask.any():
            return np.empty((2, 0), dtype=np.int64)

        node1_indices = mapped_col1[valid_mask].values
        node2_indices = mapped_col2[valid_mask].values

        return np.vstack((node1_indices, node2_indices))

    def _prepare_gene_hyperedges(self, df: pd.DataFrame) -> List[List[int]]:
        """准备基因超边"""
        if df.empty:
            return []

        hyperedges = []
        for idx, row in df.iterrows():
            genes_in_edge = []
            for col in df.columns:
                gene = row[col]
                if pd.notna(gene) and gene != "" and gene in self.gene_to_idx:
                    genes_in_edge.append(self.gene_to_idx[gene])

            if len(genes_in_edge) >= 2:
                hyperedges.append(genes_in_edge)

        return hyperedges

    def offset_binary_edges(self, edges_array: np.ndarray, entity_type1_offset: int, entity_type2_offset: int) -> Optional[np.ndarray]:
        """对二元边进行偏移处理"""
        if edges_array is None or edges_array.shape[1] == 0:
            return None

        offsetted_edges = edges_array.copy()
        offsetted_edges[0, :] += entity_type1_offset
        offsetted_edges[1, :] += entity_type2_offset
        return offsetted_edges

    def offset_gene_hyperedges(self, hyperedges: List[List[int]]) -> List[List[int]]:
        """对基因超边进行偏移处理"""
        if not hyperedges:
            return []

        offset_hyperedges = []
        for hyperedge in hyperedges:
            offset_hyperedge = [gene_id + self.gene_offset for gene_id in hyperedge]
            offset_hyperedges.append(offset_hyperedge)

        return offset_hyperedges

    def get_offset_edges(self, global_edges: Dict[str, np.ndarray], include_relations: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """获取指定关系的偏移后的边"""
        offset_edges = {}

        # 根据include_relations选择性地包含边数据
        offset_edges["ddi"] = self.offset_binary_edges(global_edges["ddi"], 0, 0) if "ddi" in include_relations else None

        offset_edges["ppi"] = self.offset_binary_edges(global_edges["ppi"], self.gene_offset, self.gene_offset) if "ppi" in include_relations else None

        offset_edges["dpi"] = self.offset_binary_edges(global_edges["dpi"], 0, self.gene_offset) if "dpi" in include_relations else None

        offset_edges["protein_disease"] = self.offset_binary_edges(global_edges["protein_disease"], self.gene_offset, self.disease_offset) if "protein_disease" in include_relations else None

        offset_edges["gene_hyperedges"] = self.offset_gene_hyperedges(global_edges["gene_hyperedges"]) if "gene_hyperedges" in include_relations else None

        return offset_edges
