"""Server-side optimization regressions. Authored but not run during delivery."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from models.encoders import GraphBranch
from models.losses import qfal_loss
from utils.hypergraph import build_binary_edges, build_hypergraph
from utils.preprocessing import negative_partition, sample_negatives, triplet_set
from utils.query_sampling import QuerySampler
import utils.experiment.training_loop as training_loop


class ConfigurationTests(unittest.TestCase):
    def test_hydra_methods_protocols_and_ablations_compose(self):
        from hydra import compose, initialize_config_dir
        root = Path(__file__).resolve().parents[1] / "conf"
        with initialize_config_dir(version_base="1.3", config_dir=str(root)):
            for method in sorted((root / "method").glob("*.yaml")):
                for protocol in ("independent_test", "cross_validation"):
                    for ablation in sorted((root / "ablation").glob("*.yaml")):
                        with self.subTest(method=method.stem, protocol=protocol, ablation=ablation.stem):
                            config = Config.from_omegaconf(compose(config_name="default", overrides=[
                                f"method={method.stem}", f"protocol={protocol}", f"ablation={ablation.stem}"]))
                            self.assertEqual(config.method_name, method.stem)

    def test_invalid_refresh_cannot_consult_heldout_labels(self):
        with self.assertRaises(ValueError):
            Config(negative_refresh_every=1)
        with self.assertRaises(ValueError):
            Config(eval_protocol="cross_validation", negative_refresh_every=1,
                   negative_sampling_strategy="partitioned", negative_filter_scope="all_known")


class RepresentationTests(unittest.TestCase):
    def test_initial_feature_path_preserves_isolated_nodes(self):
        branch = GraphBranch(2, 8, 0.0, True, "hypergraph", feature_residual=True).eval()
        with torch.no_grad():
            for parameter in branch.parameters():
                parameter.zero_()
            branch.initial_projection.weight.copy_(torch.eye(2))
        x = torch.tensor([[0., 0.], [0., 0.], [0., 0.], [1., 2.], [4., 5.]])
        incidence = torch.tensor([[0, 1, 2], [0, 0, 0]])
        output = branch(x, incidence)
        torch.testing.assert_close(output[3:], x[3:])

    def test_reverse_relations_share_only_symmetric_types(self):
        edges, types = build_binary_edges({"dpi": np.array([[0, 3]]), "ppi": np.array([[3, 4]])}, directional=True)
        actual = {(int(a), int(b)): int(t) for (a, b), t in zip(edges.T, types)}
        self.assertNotEqual(actual[(0, 3)], actual[(3, 0)])
        self.assertEqual(actual[(3, 4)], actual[(4, 3)])


class SamplingTests(unittest.TestCase):
    def test_cycle_covers_queries_and_round_trips_state(self):
        known = {(i, 99): {i + 10, i + 20} for i in range(5)}
        sampler = QuerySampler(known, np.random.default_rng(3))
        draws = []
        for _ in range(3):
            draws.extend(sampler.queries(2, "cycle"))
        self.assertEqual(len(draws), 5)
        self.assertEqual(set(draws), set(known))
        key = draws[0]
        self.assertNotEqual(sampler.target(key, "cycle"), sampler.target(key, "cycle"))
        restored = QuerySampler(known, np.random.default_rng(100))
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(sampler.queries(3, "cycle"), restored.queries(3, "cycle"))
        self.assertEqual(sampler.target(key, "cycle"), restored.target(key, "cycle"))

    def test_future_bce_negatives_stay_in_reserved_partition(self):
        positives = np.array([[0, 1, 999]])
        pools = {}
        for split in ("train", "validation", "test"):
            draws, _ = sample_negatives(positives, np.arange(1, 401), triplet_set(positives), "gene", 400,
                                        np.random.default_rng(8), partition=split, partition_seed=15)
            pools[split] = triplet_set(draws)
            self.assertTrue(pools[split])
            self.assertTrue(all(negative_partition(row, 15) == split for row in draws))
        self.assertFalse(pools["train"] & pools["validation"])
        self.assertFalse(pools["train"] & pools["test"])
        self.assertFalse(pools["validation"] & pools["test"])


class LossTests(unittest.TestCase):
    @staticmethod
    def fixed_sampler():
        return SimpleNamespace(rng=np.random.default_rng(1), queries=lambda limit, mode: [(0, 4)],
                               target=lambda key, mode: 1, all_targets=lambda key: [1, 2], diagnostics=lambda: {})

    def test_excluded_positive_cannot_change_negative_normalization(self):
        config = Config(qfal_positive_mode="exclude", qfal_num_negatives=2)
        values = torch.tensor([0., 0., 100., 0., 0.], requires_grad=True)
        model = SimpleNamespace(score=lambda embeddings, rows: values[rows[:, 1]])
        loss = qfal_loss(model, torch.zeros(5, 2), {(0, 4): {1, 2}}, np.array([1, 2, 3]),
                         "gene", config, np.random.default_rng(1), sampler=self.fixed_sampler())
        torch.testing.assert_close(loss, torch.log(torch.tensor(2.)))
        loss.backward()
        self.assertEqual(float(values.grad[2]), 0.)
        self.assertGreater(float(values.grad[3]), 0.)

    def test_all_attenuated_candidates_zero_has_finite_backward(self):
        config = Config(qfal_false_negative_weight=0., qfal_num_negatives=2)
        values = torch.tensor([0., 0., 100., 0., 0.], requires_grad=True)
        model = SimpleNamespace(score=lambda embeddings, rows: values[rows[:, 1]])
        loss = qfal_loss(model, torch.zeros(5, 2), {(0, 4): {1, 2, 3}}, np.array([1, 2, 3]),
                         "gene", config, np.random.default_rng(1), sampler=self.fixed_sampler())
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertTrue(bool(torch.isfinite(values.grad).all()))

    def test_multi_positive_averages_targets_without_competition(self):
        config = Config(qfal_positive_mode="multi_positive", qfal_self_adversarial=False)
        values = torch.tensor([0., 0., 1., 0., 0.], requires_grad=True)
        model = SimpleNamespace(score=lambda embeddings, rows: values[rows[:, 1]])
        loss = qfal_loss(model, torch.zeros(5, 2), {(0, 4): {1, 2}}, np.array([1, 2, 3]),
                         "gene", config, np.random.default_rng(1), sampler=self.fixed_sampler())
        expected = torch.nn.functional.softplus(-values[1:3] / config.qfal_temperature).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(bool((values.grad[1:3] < 0).all()))


class ResumeTests(unittest.TestCase):
    def test_interrupted_cpu_training_matches_uninterrupted_training(self):
        config = Config(device="cpu", epochs=3, eval_every=1, checkpoint_every=1, learning_rate=0.001,
                        hidden_dim=16, encoder_output_dim=4, feature_dims={k: 6 for k in ("drug", "gene", "disease")},
                        decoder_normalization="layer", feature_residual=True, qfal_query_sampling="cycle",
                        qfal_num_negatives=2, scheduler="plateau")
        features = {kind: torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10 + i
                    for i, kind in enumerate(("drug", "gene", "disease"))}
        positive = np.array([[0, 3, 6], [1, 4, 7]])
        validation = np.array([[2, 5, 8]])
        graph = (build_hypergraph(positive, torch.device("cpu")),
                 torch.tensor([[0, 3], [3, 0]]), torch.tensor([1, 1]))
        train = {"gene": np.array([[0,3,6,1], [1,4,7,1], [0,4,6,0], [1,5,7,0]]),
                 "disease": np.array([[0,3,6,1], [1,4,7,1], [0,3,7,0], [1,4,8,0]])}
        valid = {"gene": np.array([[2,5,8,1], [2,4,8,0]]), "disease": np.array([[2,5,8,1], [2,5,7,0]])}
        candidates = {"gene": np.array([3,4,5]), "disease": np.array([6,7,8])}
        with tempfile.TemporaryDirectory() as temporary:
            reference, interrupted = Path(temporary) / "reference", Path(temporary) / "interrupted"
            reference.mkdir()
            interrupted.mkdir()
            args = (features, graph, positive, validation, train, valid, candidates)
            expected_model, expected_thresholds = training_loop.fit_model(config, *args, reference, 42, 0)
            original_save = training_loop.atomic_torch_save

            def interrupt_after_commit(path, payload):
                original_save(path, payload)
                if path.name == "last_checkpoint.pt" and payload["epoch"] == 1:
                    raise RuntimeError("simulated process interruption")

            with patch.object(training_loop, "atomic_torch_save", side_effect=interrupt_after_commit):
                with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                    training_loop.fit_model(config, *args, interrupted, 42, 0)
            resumed = Config(**{**config.to_dict(), "resume_from": str(interrupted)})
            actual_model, actual_thresholds = training_loop.fit_model(resumed, *args, interrupted, 42, 0)
            self.assertEqual(expected_thresholds, actual_thresholds)
            for key, value in expected_model.state_dict().items():
                torch.testing.assert_close(value, actual_model.state_dict()[key], rtol=0, atol=0)
            expected_last = torch.load(reference / "last_checkpoint.pt", weights_only=True)
            actual_last = torch.load(interrupted / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(expected_last["history"], actual_last["history"])
            self.assertEqual(expected_last["samplers"], actual_last["samplers"])


if __name__ == "__main__":
    unittest.main()
