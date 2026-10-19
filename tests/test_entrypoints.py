"""Entrypoint regressions for server execution; not run during static delivery."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    def test_enhanced_configuration_does_not_load_training_dependencies(self):
        script = textwrap.dedent("""
            import importlib.abc
            from pathlib import Path
            import runpy
            import sys

            root = Path(sys.argv[1])
            sys.path.insert(0, str(root))

            class RejectTrainingImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'torch', 'torch_geometric'}:
                        raise RuntimeError('Config-only entry unexpectedly imported ' + fullname)
                    return None

            sys.meta_path.insert(0, RejectTrainingImports())
            sys.argv = [str(root / 'main.py'), 'method=enhanced', '--cfg', 'job']
            runpy.run_path(sys.argv[0], run_name='__main__')
        """)
        result = subprocess.run([sys.executable, "-c", script, str(ROOT)], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("method_name: enhanced", result.stdout)

    def test_public_exports_match_direct_implementations(self):
        from models import HGNN
        from models.hgnn import HGNN as direct_model
        from utils.experiment import run_experiment
        from utils.experiment.runner import run_experiment as direct_runner

        self.assertIs(HGNN, direct_model)
        self.assertIs(run_experiment, direct_runner)


if __name__ == "__main__":
    unittest.main()
