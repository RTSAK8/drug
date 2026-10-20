"""Query-aware sampling. All IDs here are in the global node index space."""

from collections import defaultdict
import hashlib

import numpy as np


def triplet_set(triplets: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row[:3])) for row in triplets}


def query_key(row, task: str) -> tuple[int, int]:
    d, g, s = map(int, row[:3])
    if task == "gene":
        return d, s
    if task == "disease":
        return d, g
    raise ValueError(f"Unsupported task: {task}")


def answer_index(task: str) -> int:
    if task not in {"gene", "disease"}:
        raise ValueError(f"Unsupported task: {task}")
    return 1 if task == "gene" else 2


def make_triplet(key: tuple[int, int], answer: int, task: str) -> tuple[int, int, int]:
    return (key[0], answer, key[1]) if task == "gene" else (key[0], key[1], answer)


def answer_map(triplets, task: str) -> dict[tuple[int, int], set[int]]:
    grouped = defaultdict(set)
    index = answer_index(task)
    for row in triplets:
        grouped[query_key(row, task)].add(int(row[index]))
    return dict(grouped)


def negative_partition(triplet, seed: int) -> str:
    """Label/task-independent 80/10/10 reservation, stable across Python processes.

    The same triplet belongs to the same partition under both completion tasks.
    This reserves future BCE draws without consulting held-out labels or samples.
    """
    payload = f"{int(seed)}:{int(triplet[0])}:{int(triplet[1])}:{int(triplet[2])}".encode("ascii")
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 100
    return "train" if bucket < 80 else ("validation" if bucket < 90 else "test")


def sample_negatives(positives: np.ndarray, candidates: np.ndarray, exclude: set,
                     task: str, ratio: int, rng: np.random.Generator,
                     partition: str | None = None, partition_seed: int = 0) -> tuple[np.ndarray, dict]:
    """Sample distinct unlabeled answers without replacement; report pool exhaustion.

    A query with m positive targets requests m*ratio distinct negatives. A small
    available pool is capped explicitly rather than silently losing retries.
    """
    if partition not in {None, "train", "validation", "test"}:
        raise ValueError(f"Unknown candidate reservation: {partition}")
    blocked = answer_map(exclude | triplet_set(positives), task)
    groups = answer_map(positives, task)
    rows, exhausted = [], []
    requested = 0
    for key in sorted(groups):
        wanted = len(groups[key]) * ratio
        requested += wanted
        unavailable = blocked.get(key, set())
        available = np.asarray([x for x in candidates if int(x) not in unavailable and (
            partition is None or negative_partition(make_triplet(key, int(x), task), partition_seed) == partition
        )], dtype=np.int64)
        count = min(wanted, len(available))
        if count < wanted:
            exhausted.append({"query": list(key), "requested": wanted, "available": len(available)})
        if count:
            selected = rng.choice(available, size=count, replace=False)
            rows.extend(make_triplet(key, int(a), task) for a in selected)
    result = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return result, {"requested": requested, "actual": len(result), "shortfall": requested-len(result),
                    "exhausted_queries": exhausted, "partition": partition, "partition_seed": partition_seed}


def labelled_data(positives: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("Classification requires both positives and sampled negatives; inspect sampling_stats.json")
    return np.concatenate([
        np.column_stack([positives, np.ones(len(positives), dtype=np.int64)]),
        np.column_stack([negatives, np.zeros(len(negatives), dtype=np.int64)]),
    ]).astype(np.int64)
