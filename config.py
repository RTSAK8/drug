"""Validated configuration for the paper model and controlled ablations."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class Config:
    experiment_name: str = "full"
    method_name: str = "paper"
    data_dir: str = "dataset"
    results_dir: str = "results"
    device: str = "auto"
    seeds: list[int] = field(default_factory=lambda: [42])
    deterministic: bool = True
    deterministic_strict: bool = False
    resume_from: str = ""  # Existing run directory, not a standalone checkpoint.
    eval_protocol: str = "independent_test"
    n_splits: int = 5
    validation_fraction: float = 0.15
    tasks: list[str] = field(default_factory=lambda: ["gene", "disease"])
    epochs: int = 1000
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    eval_every: int = 10
    checkpoint_every: int = 10
    early_stopping_patience: int = 20  # validation events, not epochs
    early_stopping_min_delta: float = 1e-5
    gradient_clip_norm: float = 5.0
    selection_metric: str = "mrr_filtered_query_macro"
    task_weights: dict[str, float] = field(default_factory=lambda: {"gene": 1.0, "disease": 1.0})
    scheduler: str = "none"
    scheduler_patience: int = 5  # Validation events.
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    bce_weight: float = 0.3
    qfal_weight: float = 0.5
    neg_ratio: int = 3
    negative_filter_scope: str = "train_only"
    negative_sampling_strategy: str = "legacy"
    negative_refresh_every: int = 0  # Optimizer updates; 0 keeps fixed BCE negatives.
    min_triplet_mapping_fraction: float = 0.0
    encoder_output_dim: int = 128
    hidden_dim: int = 512
    feature_dropout: float = 0.3
    decoder_dropout: float = 0.4
    decoder_normalization: str = "batch"
    decoder_interactions: bool = False
    feature_residual: bool = False
    final_layer_residual: bool = False
    directional_relations: bool = False
    use_hypergraph: bool = True
    use_relational_graph: bool = True
    hgnn_use_residual: bool = True
    rgcn_use_residual: bool = True
    hgnn_dropout: float = 0.4
    rgcn_dropout: float = 0.4
    hgnn_hyperedge_dropout: float = 0.0
    hypergraph_normalization: str = "symmetric"
    fusion_method: str = "concat"
    relations: list[str] = field(default_factory=lambda: ["ppi", "dpi", "ddi", "protein_disease"])
    feature_dims: dict[str, int] = field(default_factory=lambda: {"drug": 768, "gene": 1024, "disease": 768})
    qfal_num_negatives: int = 64
    qfal_temperature: float = 0.2
    qfal_known_positive_masking: bool = True
    qfal_false_negative_weight: float = 0.05
    qfal_self_adversarial: bool = True
    qfal_adversarial_temperature: float = 1.0
    qfal_max_queries: int = 256  # unique queries per task/step; 0 = all
    qfal_query_sampling: str = "random"
    qfal_positive_mode: str = "attenuate"
    ranking_query_batch_size: int = 16
    ranking_candidate_batch_size: int = 128
    prediction_batch_size: int = 8192
    ranking_tie_policy: str = "average"
    hit_k: list[int] = field(default_factory=lambda: [1, 3, 5])
    recall_k: list[int] = field(default_factory=lambda: [1, 5, 10])
    use_wandb: bool = False
    wandb_project: str = "drug-triplet-completion"
    wandb_entity: str = ""

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> "Config":
        values = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a mapping")
        values.pop("hydra", None)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown/retired config keys: {sorted(unknown)}; see README.md")
        return cls(**values)

    def __post_init__(self):
        for key in ("deterministic", "use_hypergraph", "use_relational_graph", "hgnn_use_residual",
                    "rgcn_use_residual", "qfal_known_positive_masking", "qfal_self_adversarial", "use_wandb",
                    "deterministic_strict", "decoder_interactions", "feature_residual",
                    "final_layer_residual", "directional_relations"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a YAML boolean, not a string or number")
        for key, allowed in {
            "eval_protocol": {"independent_test", "cross_validation"},
            "negative_filter_scope": {"train_only", "all_known"},
            "fusion_method": {"concat", "gate"},
            "hypergraph_normalization": {"symmetric", "random_walk"},
            "ranking_tie_policy": {"average", "optimistic", "pessimistic"},
            "decoder_normalization": {"batch", "layer", "none"},
            "scheduler": {"none", "plateau"},
            "qfal_query_sampling": {"random", "cycle"},
            "qfal_positive_mode": {"attenuate", "exclude", "multi_positive"},
            "negative_sampling_strategy": {"legacy", "partitioned"},
        }.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {sorted(allowed)}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or set(self.tasks) - {"gene", "disease"}:
            raise ValueError("tasks must contain unique gene/disease entries")
        if set(self.task_weights) != {"gene", "disease"} or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                for v in self.task_weights.values()):
            raise ValueError("task_weights must contain positive finite gene/disease weights")
        selection_metrics = {"auc", "aupr", "average_precision", "f1", "mrr_raw", "mrr_filtered",
                             "mrr_raw_query_macro", "mrr_filtered_query_macro"}
        selection_metrics |= {f"hit@{k}_{mode}" for k in self.hit_k for mode in ("raw", "filtered")}
        selection_metrics |= {f"recall@{k}_{mode}_query_macro" for k in self.recall_k for mode in ("raw", "filtered")}
        if self.selection_metric not in selection_metrics:
            raise ValueError("selection_metric must name a reported performance metric (higher is better)")
        if self.deterministic_strict and not self.deterministic:
            raise ValueError("deterministic_strict requires deterministic=true")
        if not 0 < self.scheduler_factor < 1 or not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("Require 0 < scheduler_factor < 1 and 0 < min_learning_rate <= learning_rate")
        if not 0 <= self.min_triplet_mapping_fraction <= 1:
            raise ValueError("min_triplet_mapping_fraction must be in [0, 1]")
        if not self.seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("seeds must contain integers in [0, 2**32)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate seeds are not independent repeats")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        for key in ("epochs", "eval_every", "early_stopping_patience", "neg_ratio", "encoder_output_dim",
                    "qfal_num_negatives", "ranking_query_batch_size", "ranking_candidate_batch_size", "prediction_batch_size",
                    "scheduler_patience", "checkpoint_every"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.n_splits) is not int or self.n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        if type(self.hidden_dim) is not int or self.hidden_dim < 8 or self.hidden_dim % 4:
            raise ValueError("hidden_dim must be >= 8 and divisible by 4 (decoder hidden widths must be nonzero)")
        for key in ("feature_dropout", "decoder_dropout", "hgnn_dropout", "rgcn_dropout", "hgnn_hyperedge_dropout"):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must be in [0, 1)")
        for key in ("learning_rate", "qfal_temperature", "qfal_adversarial_temperature"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("bce_weight", "qfal_weight", "weight_decay", "gradient_clip_norm", "early_stopping_min_delta"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.bce_weight + self.qfal_weight <= 0:
            raise ValueError("At least one training loss must be active")
        if type(self.qfal_max_queries) is not int or self.qfal_max_queries < 0:
            raise ValueError("qfal_max_queries must be a nonnegative integer")
        if type(self.negative_refresh_every) is not int or self.negative_refresh_every < 0:
            raise ValueError("negative_refresh_every must be a nonnegative integer")
        if self.negative_refresh_every and (
                self.negative_sampling_strategy != "partitioned" or self.negative_filter_scope != "train_only"):
            raise ValueError("Negative refresh requires partitioned sampling and train_only filtering")
        if self.qfal_positive_mode != "attenuate" and not self.qfal_known_positive_masking:
            raise ValueError("Disabling positive masking only applies to qfal_positive_mode=attenuate")
        if not 0 <= self.qfal_false_negative_weight < 1:
            raise ValueError("qfal_false_negative_weight must be in [0, 1)")
        if set(self.relations) - {"ppi", "dpi", "ddi", "protein_disease"} or len(set(self.relations)) != len(self.relations):
            raise ValueError("Unknown or duplicate auxiliary relations")
        if self.use_relational_graph and not self.relations:
            raise ValueError("A relational branch requires at least one relation")
        if set(self.feature_dims) != {"drug", "gene", "disease"} or any(type(v) is not int or v < 1 for v in self.feature_dims.values()):
            raise ValueError("feature_dims must specify positive drug/gene/disease dimensions")
        for values in (self.hit_k, self.recall_k):
            if not values or any(type(k) is not int or k < 1 for k in values) or len(set(values)) != len(values):
                raise ValueError("Metric cutoffs must be unique positive integers")
        if self.eval_protocol == "independent_test" and self.negative_filter_scope != "train_only":
            raise ValueError("Independent test requires train_only negative filtering")
        root = Path(__file__).resolve().parent
        for key in ("data_dir", "results_dir"):
            value = Path(getattr(self, key)).expanduser()
            setattr(self, key, str(value.resolve() if value.is_absolute() else (root / value).resolve()))
        if self.resume_from:
            value = Path(self.resume_from).expanduser()
            self.resume_from = str(value.resolve() if value.is_absolute() else (root / value).resolve())

    def to_dict(self) -> dict:
        return asdict(self)
