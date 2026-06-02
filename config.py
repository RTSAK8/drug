"""
配置文件 - 用于药物-基因-疾病超图对比学习
"""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping

from omegaconf import DictConfig, OmegaConf


@dataclass
class Config:
    """实验配置类"""
    
    # 训练参数
    epochs: int = 1000
    learning_rate: float = 0.005
    alpha: float = 0.4  # 兼容原始 BCE + QCCL 权重
    bce_weight: float = 0.3
    qfal_weight: float = 0.5
    qccl_weight: float = 0.2
    L2: float = 1e-4
    loss_type: List[str] = field(default_factory=lambda: ["qccl"])  # 可选: qccl, qfal

    # 数据参数
    data_dir: str = "./dataset"
    neg_ratio: int = 3
    max_attempts: int = 50
    report_overlap_stats: bool = True
    drop_test_overlap_with_train: bool = False

    # 交叉验证参数
    n_splits: int = 5
    random_state: int = 42

    # 模型参数
    encoder_output_dim: int = 128
    hidden_dim: int = 512

    # HGNN表达增强（方法5）
    # 残差连接通常能提升稳定性
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

    # QCCL基础配置
    qccl_temperature: float = 0.3
    # <=0 时回退到 qccl_temperature
    qccl_gene_query_temperature: float = -1.0
    qccl_drug_query_temperature: float = -1.0
    qccl_disease_query_temperature: float = -1.0
    # symmetric | query_only | candidate_only | none
    qccl_asymmetric_norm_mode: str = "symmetric"

    # QCCL v2-lite: multi-positive 去假负 + hardness重加权
    qccl_multi_positive: bool = False
    qccl_hardness_reweight: bool = False
    qccl_hardness_alpha: float = 1.0
    qccl_hardness_temperature: float = 1.0
    qccl_query_hardness_reweight: bool = False
    qccl_query_hardness_alpha: float = 1.0
    qccl_query_hardness_temperature: float = 1.0

    # QCCL v2-standard: 三池负样本调度
    qccl_three_pool_schedule: bool = False
    qccl_pool_easy_quantile: float = 0.33
    qccl_pool_hard_quantile: float = 0.66
    qccl_pool_start_weights: List[float] = field(default_factory=lambda: [1.25, 1.0, 0.75])
    qccl_pool_end_weights: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])

    # QCCL v2-advanced: memory queue
    qccl_memory_queue: bool = False
    qccl_memory_queue_size: int = 0
    qccl_memory_queue_warmup_progress: float = 0.0
    qccl_memory_queue_negatives_per_query: int = 0

    # QFAL: Query-conditioned False-negative-aware Listwise Ranking Loss
    qfal_num_negatives: int = 64
    qfal_temperature: float = 0.2
    qfal_known_positive_masking: bool = True
    qfal_false_negative_weight: float = 0.05
    qfal_self_adversarial: bool = True
    qfal_adversarial_temperature: float = 1.0
    qfal_max_queries: int = 2048

    # QCCL模块结构（方便消融）
    qccl_cond_dim_ratio: float = 0.5
    qccl_num_layers: int = 2
    qccl_query_dropout: float = 0.3
    qccl_film_layer_norm: bool = False

    # wandb配置
    use_wandb: bool = True
    wandb_project: str = "Drug Effect Prediction"
    wandb_entity: str = ""
    wandb_name_prefix: str = "XXX"
    wandb_run_index: int = 1

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

    @classmethod
    def from_omegaconf(cls, cfg: Any = None) -> "Config":
        """将Hydra/OmegaConf配置安全转换为Config对象。"""
        if cfg is None:
            return cls()

        if isinstance(cfg, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        elif isinstance(cfg, Mapping):
            cfg_dict = dict(cfg)
        else:
            raise TypeError("cfg must be a DictConfig or mapping")

        if not isinstance(cfg_dict, dict):
            raise TypeError("resolved config must be a dictionary")

        cfg_dict.pop("hydra", None)

        valid_fields = {item.name for item in fields(cls)}
        unknown_keys = sorted(set(cfg_dict.keys()) - valid_fields)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

        init_kwargs = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        return cls(**init_kwargs)

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
