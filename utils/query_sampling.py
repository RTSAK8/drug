"""Serializable query/answer coverage without adding optimizer steps per epoch."""

import copy

import numpy as np


class QuerySampler:
    def __init__(self, known_answers: dict, rng: np.random.Generator):
        if not known_answers or any(not answers for answers in known_answers.values()):
            raise ValueError("Query sampling requires nonempty queries and answer sets")
        self.keys = sorted(known_answers)
        self.answers = [sorted(map(int, known_answers[key])) for key in self.keys]
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.rng = rng
        self.order = []
        self.cursor = 0
        self.visits = [0] * len(self.keys)
        self.answer_orders = [[] for _ in self.keys]
        self.answer_cursors = [0] * len(self.keys)
        self.answer_visits = [[0] * len(answers) for answers in self.answers]

    def queries(self, limit: int, mode: str) -> list[tuple[int, int]]:
        size = min(limit or len(self.keys), len(self.keys))
        if mode == "random":
            chosen = self.rng.choice(len(self.keys), size, replace=False).tolist() if size < len(self.keys) else list(range(size))
        elif mode == "cycle":
            if self.cursor >= len(self.order):
                self.order = self.rng.permutation(len(self.keys)).tolist()
                self.cursor = 0
            # A short final batch finishes the cycle; never repeats a query within it.
            chosen = self.order[self.cursor:self.cursor + size]
            self.cursor += len(chosen)
        else:
            raise ValueError(f"Unknown query sampler: {mode}")
        for i in chosen:
            self.visits[i] += 1
        return [self.keys[i] for i in chosen]

    def target(self, key: tuple[int, int], mode: str) -> int:
        i = self.index[key]
        if mode == "cycle":
            if self.answer_cursors[i] >= len(self.answer_orders[i]):
                self.answer_orders[i] = self.rng.permutation(len(self.answers[i])).tolist()
                self.answer_cursors[i] = 0
            j = self.answer_orders[i][self.answer_cursors[i]]
            self.answer_cursors[i] += 1
        else:
            j = int(self.rng.integers(len(self.answers[i])))
        self.answer_visits[i][j] += 1
        return self.answers[i][j]

    def all_targets(self, key: tuple[int, int]) -> list[int]:
        i = self.index[key]
        self.answer_visits[i] = [value + 1 for value in self.answer_visits[i]]
        return self.answers[i]

    def diagnostics(self) -> dict:
        total_answers = sum(map(len, self.answers))
        return {
            "query_coverage": sum(v > 0 for v in self.visits) / len(self.keys),
            "answer_coverage": sum(v > 0 for row in self.answer_visits for v in row) / total_answers,
            "query_visits_min": min(self.visits),
            "query_visits_max": max(self.visits),
        }

    def state_dict(self) -> dict:
        return copy.deepcopy({
            "keys": self.keys, "answers": self.answers, "order": self.order, "cursor": self.cursor,
            "visits": self.visits, "answer_orders": self.answer_orders, "answer_cursors": self.answer_cursors,
            "answer_visits": self.answer_visits, "rng": self.rng.bit_generator.state,
        })

    def load_state_dict(self, state: dict) -> None:
        state = copy.deepcopy(state)
        if state["keys"] != self.keys or state["answers"] != self.answers:
            raise ValueError("Cannot resume query sampling with different training answers")
        for key in ("order", "cursor", "visits", "answer_orders", "answer_cursors", "answer_visits"):
            setattr(self, key, state[key])
        self.rng.bit_generator.state = state["rng"]
