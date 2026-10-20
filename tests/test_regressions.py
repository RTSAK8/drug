"""Optional server-side regression checks. Not executed during this delivery."""

import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from models.hgnn import HGNN
from models.encoders import HypergraphLayer
from models.losses import qfal_loss
from utils.hypergraph import build_hypergraph
from utils.metrics import rank_queries
from utils.preprocessing import sample_negatives, triplet_set
from utils.training import evaluate


class SamplingTests(unittest.TestCase):
    def test_small_pool_reports_shortfall_and_never_samples_positive(self):
        positives = np.array([[0, 1, 4]])
        negatives, stats = sample_negatives(positives, np.array([1, 2]), triplet_set(positives),
                                            "gene", 3, np.random.default_rng(1))
        self.assertEqual(negatives.tolist(), [[0, 2, 4]])
        self.assertEqual(stats["shortfall"], 2)

    def test_available_pool_fills_quota_without_duplicates(self):
        positives = np.array([[0, 1, 9], [0, 2, 9]])
        negatives, stats = sample_negatives(positives, np.arange(1, 9), triplet_set(positives),
                                            "gene", 2, np.random.default_rng(1))
        self.assertEqual(len(negatives), 4)
        self.assertEqual(len(triplet_set(negatives)), 4)
        self.assertEqual(stats["shortfall"], 0)
        self.assertFalse(triplet_set(negatives) & triplet_set(positives))


class OperatorTests(unittest.TestCase):
    def test_symmetric_hypergraph_matches_explicit_matrix(self):
        incidence = torch.tensor([[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]])
        h = torch.zeros(5, 2)
        h[incidence[0], incidence[1]] = 1
        degrees = h.sum(1)
        scale = degrees.clamp_min(1).rsqrt().masked_fill(degrees == 0, 0)
        a = torch.diag(scale) @ h @ torch.diag(1 / h.sum(0)) @ h.T @ torch.diag(scale)
        layer = HypergraphLayer(2, 2, "symmetric")
        with torch.no_grad():
            layer.linear.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        torch.testing.assert_close(layer(x, incidence), a @ x)

    def test_qfal_adversarial_temperature_uses_raw_logits(self):
        config = Config(qfal_num_negatives=2, qfal_max_queries=0, qfal_temperature=0.2,
                        qfal_adversarial_temperature=1.0)
        model = SimpleNamespace(score=lambda embeddings, rows: rows[:, 1].float() - 1)
        loss = qfal_loss(model, torch.zeros(5, 2), {(0, 4): {1}}, np.array([1, 2, 3]),
                         "gene", config, np.random.default_rng(1))
        negatives = torch.tensor([1.0, 2.0])
        weights = 2 * torch.softmax(negatives, dim=0)
        expected = torch.logsumexp(torch.cat([torch.zeros(1), negatives / 0.2 + weights.log()]), dim=0)
        torch.testing.assert_close(loss, expected)


class EvaluationTests(unittest.TestCase):
    def test_filtered_rank_and_multi_answer_recall_are_distinct(self):
        # g=1 and g=2 are held-out answers; g=3 is a known training answer.
        score_map = torch.tensor([0.0, 0.9, 0.8, 1.0, 0.0])
        model = SimpleNamespace(training=False, score=lambda embeddings, rows: score_map[rows[:, 1]])
        positives = np.array([[0, 1, 4], [0, 2, 4]])
        known = np.concatenate([positives, np.array([[0, 3, 4]])])
        metrics, ranks, _ = rank_queries(model, torch.zeros(5, 2), positives, known,
                                         np.array([1, 2, 3]), "gene", Config())
        self.assertEqual(ranks.raw_rank.tolist(), [2.0, 3.0])
        self.assertEqual(ranks.filtered_rank.tolist(), [1.0, 1.0])
        self.assertEqual(metrics["hit@1_filtered"], 1.0)
        self.assertEqual(metrics["recall@1_filtered_query_macro"], 0.5)

    def test_evaluate_is_repeatable_and_does_not_update_batchnorm(self):
        torch.manual_seed(7)
        config = Config(hidden_dim=16, encoder_output_dim=4, feature_dims={"drug": 6, "gene": 6, "disease": 6})
        model = HGNN(config, config.feature_dims)
        features = {kind: torch.randn(3, 6) for kind in ("drug", "gene", "disease")}
        positive = np.array([[0, 3, 6], [1, 4, 7]])
        incidence = build_hypergraph(positive, torch.device("cpu"))
        graph = (incidence, torch.tensor([[0, 3], [3, 0]]), torch.tensor([1, 1]))
        labelled = {
            "gene": np.array([[0,3,6,1], [1,4,7,1], [0,4,6,0], [1,5,7,0]]),
            "disease": np.array([[0,3,6,1], [1,4,7,1], [0,3,7,0], [1,4,8,0]]),
        }
        candidates = {"gene": np.array([3,4,5]), "disease": np.array([6,7,8])}
        before = copy.deepcopy(model.state_dict())
        first = evaluate(model, features, graph, labelled, positive, positive, candidates, config,
                         thresholds={"gene": 0.0, "disease": 0.0})
        second = evaluate(model, features, graph, labelled, positive, positive, candidates, config,
                          thresholds={"gene": 0.0, "disease": 0.0})
        self.assertFalse(model.training)
        self.assertEqual(first[0], second[0])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
