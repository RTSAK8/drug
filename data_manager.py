"""Strict CSV loading with stable string IDs, feature validation, and provenance."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Config
from utils.hypergraph import build_binary_edges
from utils.preprocessing import triplet_set

LOG = logging.getLogger(__name__)
TRIPLET_COLUMNS = ["Drug_ID", "Gene_ID", "Disease_ID"]


class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self.root = Path(config.data_dir)
        self.maps = {}
        self.offsets = {}
        self.counts = {}
        self.features = {}
        self.stats = {}
        self.rejected = {}
        self.source_files = []

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file is missing: {path}")
        self.source_files.append(path)
        return path

    @staticmethod
    def _normalize_ids(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        frame = frame.copy()
        for column in columns:
            frame[column] = frame[column].astype("string").str.strip()
        return frame

    def _load_features(self, kind: str, column: str):
        path = self._path(f"{kind}_embeddings.csv")
        frame = pd.read_csv(path, dtype={column: "string"}, keep_default_na=False)
        frame = self._normalize_ids(frame, [column])
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"Missing entity IDs in {path.name}")
        # Exact repeated rows may be removed; conflicting vectors must not be hidden.
        original_count = len(frame)
        frame = frame.drop_duplicates().reset_index(drop=True)
        if frame[column].duplicated().any():
            raise ValueError(f"Conflicting duplicate {column} vectors in {path.name}")
        feature_columns = [c for c in frame.columns if c != column]
        expected = self.config.feature_dims[kind]
        if len(feature_columns) != expected:
            raise ValueError(f"{path.name}: expected {expected} feature columns, found {len(feature_columns)}")
        values = frame[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"{path.name}: require >=2 entities and finite numeric vectors")
        self.maps[kind] = {str(entity): index for index, entity in enumerate(frame[column])}
        self.counts[kind] = len(frame)
        self.features[kind] = torch.from_numpy(values)
        self.stats[f"features_{kind}"] = {"rows_raw": original_count, "entities": len(frame), "dimension": expected}

    def _load_triplets(self, name: str) -> np.ndarray:
        frame = self._normalize_ids(pd.read_csv(self._path(name), dtype="string", keep_default_na=False), TRIPLET_COLUMNS)
        frame = frame[TRIPLET_COLUMNS]
        initial = len(frame)
        frame = frame.drop_duplicates().reset_index(drop=True)
        mapped = pd.DataFrame(index=frame.index)
        for kind, column in zip(("drug", "gene", "disease"), TRIPLET_COLUMNS):
            mapped[column] = frame[column].map(self.maps[kind]) + self.offsets[kind]
        valid = mapped.notna().all(axis=1)
        self.rejected[f"{Path(name).stem}_unmapped"] = frame.loc[~valid].copy()
        self.stats[name] = {"raw": initial, "duplicates_removed": initial-len(frame),
                            "unmapped_removed": int((~valid).sum()), "mapped": int(valid.sum())}
        if not valid.any():
            raise ValueError(f"No valid mapped triplets remain in {name}")
        return mapped.loc[valid].to_numpy(dtype=np.int64)

    def _load_relation(self, name: str, filename: str, kinds: tuple, columns: list) -> np.ndarray:
        path = self._path(filename)
        retained, raw, matched, self_links = [], 0, 0, 0
        # Large PPI/DDI files are filtered before concatenation.
        for chunk in pd.read_csv(path, usecols=columns, dtype="string", keep_default_na=False, chunksize=200_000):
            chunk = self._normalize_ids(chunk, columns)
            raw += len(chunk)
            a = chunk[columns[0]].map(self.maps[kinds[0]]) + self.offsets[kinds[0]]
            b = chunk[columns[1]].map(self.maps[kinds[1]]) + self.offsets[kinds[1]]
            valid = a.notna() & b.notna()
            pairs = np.column_stack([a[valid].to_numpy(dtype=np.int64), b[valid].to_numpy(dtype=np.int64)])
            matched += len(pairs)
            self_links += int((pairs[:, 0] == pairs[:, 1]).sum())
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            if len(pairs):
                # All four relations use explicitly documented undirected semantics.
                retained.append(np.unique(np.sort(pairs, axis=1), axis=0))
        edges = np.unique(np.concatenate(retained), axis=0) if retained else np.empty((0, 2), dtype=np.int64)
        self.stats[name] = {"raw": raw, "mapped": matched, "self_links_removed": self_links, "unique_undirected": len(edges)}
        if len(edges) == 0:
            raise ValueError(f"Selected relation {name} has no mapped non-self edges; fix IDs or explicitly remove it from relations")
        return edges

    def load(self):
        offset = 0
        for kind, column in zip(("drug", "gene", "disease"), TRIPLET_COLUMNS):
            self._load_features(kind, column)
            self.offsets[kind] = offset
            offset += self.counts[kind]
        self.candidates = {kind: np.arange(self.offsets[kind], self.offsets[kind]+self.counts[kind], dtype=np.int64)
                           for kind in ("gene", "disease")}
        self.positives = self._load_triplets("triplets.csv")
        self.external = np.empty((0, 3), dtype=np.int64)
        if self.config.eval_protocol == "independent_test":
            external = self._load_triplets("drugbank_triplets.csv")
            known = triplet_set(self.positives)
            overlap = np.asarray([tuple(row) in known for row in external], dtype=bool)
            self.rejected["drugbank_overlap"] = pd.DataFrame(external[overlap], columns=TRIPLET_COLUMNS)
            self.external = external[~overlap]
            self.stats["drugbank_triplets.csv"].update(overlap_removed=int(overlap.sum()), final_test_count=len(self.external))
            if len(self.external) == 0:
                raise ValueError("External test is empty after mapping/deduplication/overlap removal")
        definitions = {
            "ppi": ("ppi.csv", ("gene", "gene"), ["Gene_ID_1", "Gene_ID_2"]),
            "dpi": ("drug_protein.csv", ("drug", "gene"), ["Drug_ID", "Gene_ID"]),
            "ddi": ("drug_drug.csv", ("drug", "drug"), ["Drug_ID_1", "Drug_ID_2"]),
            "protein_disease": ("protein_disease.csv", ("gene", "disease"), ["Gene_ID", "Disease_ID"]),
        }
        relations = {}
        if self.config.use_relational_graph:
            for name in self.config.relations:
                relations[name] = self._load_relation(name, *definitions[name])
        self.edge_index, self.edge_type = build_binary_edges(relations)
        self.stats["typed_directed_edges"] = int(self.edge_type.numel())
        LOG.info("Loaded data: %s", self.stats)
        return self

    def mapping_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"entity_type": kind, "original_id": original, "global_id": local+self.offsets[kind]}
                             for kind, mapping in self.maps.items() for original, local in mapping.items()])
