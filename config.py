"""
配置文件 - 用于药物-基因-疾病超图对比学习
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """实验配置类"""
    
    # 训练参数
    epochs: int = 1000
    learning_rate: float = 0.005
    alpha: float = 0.4  # 对比学习权重
    L2: float = 1e-4
    loss_type: List[str] = field(default_factory=lambda: ["qccl"])  # 使用QCCL

    # 数据参数
    data_dir: str = "./dataset"
    neg_ratio: int = 3
    max_attempts: int = 50

    # 交叉验证参数
    n_splits: int = 5
    random_state: int = 42

    # 模型参数
    encoder_output_dim: int = 128
    hidden_dim: int = 512

    # HGNN表达增强（方法5）
    # 残差连接通常能提升稳定性；默认关闭以保持与旧实验可比。
    hgnn_use_residual: bool = True
    # Jumping Knowledge: last | sum | mean | concat
    hgnn_jk: str = "last"
    # batch | layer
    hgnn_norm: str = "batch"
    # 训练时对超边incidence做dropout（0表示关闭）
    hgnn_hyperedge_dropout: float = 0.0
    hgnn_dropout: float = 0.4

    # 二元边RGCN使用残差连接和dropout
    rgcn_use_residual: bool = True
    rgcn_dropout: float = 0.4

    # 分支融合方式：gate(按比例加和) | concat(拼接后投影)
    fusion_method: str = "concat"

    # wandb配置
    use_wandb: bool = True
    wandb_project: str = "Drug Effect Prediction"
    wandb_entity: str = ""

    # 消融实验配置
    graph_ablation_configs: Dict[str, List[str]] = field(
        default_factory=lambda: {
            # 'OnlyTriplets': [],  # 只有三元组
            # 'WithPPI': ['ppi'],  # 包含PPI
            # 'WithDPI': ['dpi'],  # 包含DPI
            # 'WithDDI': ['ddi'],  # 包含DDI
            "WithoutGeneHyperedges": ["ppi", "dpi", "ddi", "protein_disease"],  # 不包含基因超边
            # 'AllRelations': ['ppi', 'dpi', 'ddi', 'gene_hyperedges'],  # 所有关系
        }
    )

    def __post_init__(self):
        # 核心数据文件
        self.drug_file = os.path.join(self.data_dir, "drug_embeddings.csv")
        self.gene_file = os.path.join(self.data_dir, "gene_embeddings.csv")
        self.disease_file = os.path.join(self.data_dir, "disease_embeddings.csv")
        self.triplets_file = os.path.join(self.data_dir, "triplets.csv")
        # 独立测试集（DrugBank）
        self.drugbank_triplets_file = os.path.join(self.data_dir, "drugbank_triplets.csv")

        # PrimeKG关系文件
        self.ppi_file = os.path.join(self.data_dir, "ppi.csv")
        self.dpi_file = os.path.join(self.data_dir, "drug_protein.csv")
        self.ddi_file = os.path.join(self.data_dir, "drug_drug.csv")
        self.protein_disease_file = os.path.join(self.data_dir, "protein_disease.csv")
        self.gene_hyperedges_file = os.path.join(self.data_dir, "pathway_gene.csv")
