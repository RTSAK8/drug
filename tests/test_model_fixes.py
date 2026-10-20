"""Behavioral regressions for isolated nodes, coverage and unlabeled supervision."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from data_manager import DataManager
from models.encoders import GraphBranch, HypergraphLayer
from models.losses import bounded_cross_entropy, classification_loss, qfal_loss
from utils.experiment.data_prep import prepare_fold


class IsolatedNodeTests(unittest.TestCase):
    def test_singleton_fallback_and_feature_gradients_for_both_normalizations(self):
        for normalization in ("symmetric", "random_walk"):
            layer = HypergraphLayer(2, 2, normalization)
            with torch.no_grad():
                layer.linear.weight.copy_(torch.eye(2))
            x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], requires_grad=True)
            incidence = torch.tensor([[0, 1], [0, 0]])
            output = layer(x, incidence)
            torch.testing.assert_close(output[:2], torch.tensor([[2., 3.], [2., 3.]]))
            torch.testing.assert_close(output[2:], x[2:])
            output[2:].sum().backward()
            torch.testing.assert_close(x.grad[2:], torch.ones(2, 2))

    def test_empty_graph_behaves_like_per_node_projection(self):
        layer = HypergraphLayer(2, 3, "symmetric")
        x = torch.randn(4, 2, requires_grad=True)
        actual = layer(x, torch.empty(2, 0, dtype=torch.long))
        torch.testing.assert_close(actual, layer.linear(x) + layer.bias)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_three_layers_keep_isolated_features_even_without_residuals(self):
        for residual in (False, True):
            branch = GraphBranch(2, 8, 0., residual, "hypergraph").eval()
            with torch.no_grad():
                for conv in branch.convs:
                    conv.linear.weight.fill_(0.1)
                for projection in branch.projections:
                    projection.weight.fill_(0.1)
            x = torch.tensor([[1., 1.], [1., 1.], [2., 2.], [4., 4.]], requires_grad=True)
            output = branch(x, torch.tensor([[0, 1], [0, 0]]))
            self.assertFalse(torch.allclose(output[2], output[3]))
            output[2:].sum().backward()
            self.assertTrue(bool((x.grad[2:] > 0).all()))


class UnlabeledLossTests(unittest.TestCase):
    def test_zero_q_recovers_bce(self):
        logits = torch.tensor([-5., 0., 2., 10.], requires_grad=True)
        labels = torch.tensor([1., 0., 1., 0.])
        actual = classification_loss(logits, labels, 0.)
        expected = F.binary_cross_entropy_with_logits(logits, labels)
        torch.testing.assert_close(actual, expected)

    def test_positive_supervision_unchanged_and_conflicting_unlabeled_gradient_reduced(self):
        labels = torch.tensor([1., 0.])
        original = torch.tensor([-10., 10.], requires_grad=True)
        robust = original.detach().clone().requires_grad_()
        classification_loss(original, labels, 0.).backward()
        classification_loss(robust, labels, 0.2).backward()
        torch.testing.assert_close(robust.grad[0], original.grad[0])
        self.assertGreater(float(robust.grad[1]), 0.)
        self.assertLess(float(robust.grad[1]), float(original.grad[1]) * 0.15)

    def test_bounded_transform_is_finite_at_extremes_and_continuous_near_zero_q(self):
        values = torch.tensor([0., 1., 1000.], requires_grad=True)
        result = bounded_cross_entropy(values, 0.2)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool((result <= 5.).all()))
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(values.grad).all()))
        torch.testing.assert_close(bounded_cross_entropy(values.detach(), 1e-10), values.detach())

    def test_qfal_also_limits_conflicting_candidate_influence(self):
        def compute(q):
            scores = torch.tensor([0., 0., 10., 0.], requires_grad=True)
            model = SimpleNamespace(score=lambda embeddings, rows: scores[rows[:, 1]])
            cfg = Config(unlabeled_loss_q=q, qfal_self_adversarial=False, qfal_num_negatives=1)
            loss = qfal_loss(model, torch.zeros(4, 2), {(0, 3): {1}},
                             np.array([1, 2]), "gene", cfg, np.random.default_rng(7))
            loss.backward()
            return loss.detach(), scores.grad
        ordinary, original_gradient = compute(0.)
        robust, robust_gradient = compute(0.2)
        self.assertGreater(float(ordinary), 40.)
        self.assertLessEqual(float(robust), 5.)
        self.assertLess(float(robust_gradient[2]), float(original_gradient[2]) * 0.001)
        self.assertTrue(bool(torch.isfinite(robust_gradient).all()))

    def test_invalid_noise_parameters_rejected(self):
        for q in (-1., 1.1, float("nan"), float("inf"), True, "0.2"):
            with self.subTest(q=q), self.assertRaises(ValueError):
                Config(unlabeled_loss_q=q)

    def test_heldout_labels_do_not_change_train_or_validation_samples(self):
        config = Config(use_relational_graph=False)
        outer = np.array([[i, 30 + i, 60 + i] for i in range(12)])
        candidates = {"gene": np.arange(30, 60), "disease": np.arange(60, 90)}
        data = SimpleNamespace(candidates=candidates)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            first.mkdir()
            second.mkdir()
            _, original = prepare_fold(data, config, 42, 0, outer, np.array([[20, 50, 80]]), first)
            # Deliberately change the test set to a sampled training "negative".
            # Training may not consult this label to remove the collision.
            heldout = original["train"]["gene"][original["train"]["gene"][:, 3] == 0][:1, :3]
            _, changed = prepare_fold(data, config, 42, 0, outer, heldout, second)
            for split in ("train", "validation"):
                for task in config.tasks:
                    np.testing.assert_array_equal(original[split][task], changed[split][task])


class CoverageTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        for kind, column, ids in (
            ("drug", "Drug_ID", ["d0", "d1"]),
            ("gene", "Gene_ID", ["g0", "g1"]),
            ("disease", "Disease_ID", ["s0", "s1"]),
        ):
            pd.DataFrame({column: ids, "f0": [1., 2.], "f1": [2., 3.]}).to_csv(
                root / f"{kind}_embeddings.csv", index=False)
        columns = ["Drug_ID", "Gene_ID", "Disease_ID"]
        pd.DataFrame([["d0", "g0", "s0"]], columns=columns).to_csv(root / "triplets.csv", index=False)
        pd.DataFrame([["d1", "g1", "s1"], ["missing_drug", "g1", "s1"]], columns=columns).to_csv(
            root / "drugbank_triplets.csv", index=False)

    @staticmethod
    def configuration(root, **overrides):
        return Config(data_dir=str(root), feature_dims={k: 2 for k in ("drug", "gene", "disease")},
                      use_relational_graph=False, **overrides)

    def test_default_rejects_incomplete_external_vocabulary_with_actionable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, r"50.000%.*1/2.*missing_drug"):
                DataManager(self.configuration(root)).load()

    def test_explicit_subset_retains_coverage_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            data = DataManager(self.configuration(root, min_triplet_mapping_fraction=0.5)).load()
            self.assertEqual(len(data.external), 1)
            self.assertEqual(data.stats["drugbank_triplets.csv"]["mapping_fraction"], 0.5)
            self.assertEqual(data.stats["drugbank_triplets.csv"]["final_fraction_of_unique"], 0.5)
            self.assertEqual(data.stats["drugbank_triplets.csv"]["missing_entity_ids"]["Drug_ID"], ["missing_drug"])

    def test_complete_external_vocab_is_accepted_and_cv_does_not_read_external(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            external = root / "drugbank_triplets.csv"
            frame = pd.read_csv(external).iloc[:1]
            frame.to_csv(external, index=False)
            data = DataManager(self.configuration(root)).load()
            self.assertEqual(len(data.external), 1)
            external.write_text("not a valid triplet CSV", encoding="utf-8")
            data = DataManager(self.configuration(root, eval_protocol="cross_validation")).load()
            self.assertEqual(len(data.positives), 1)
            self.assertNotIn(external, data.source_files)


if __name__ == "__main__":
    unittest.main()
