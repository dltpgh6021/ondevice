"""Keep project imports local when ROS supplies a package also named scripts."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase2ImportTests(unittest.TestCase):
    def test_external_scripts_package_does_not_shadow_project_entry_points(self):
        scratch = ROOT / "results" / "import-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            external = Path(directory) / "scripts"
            external.mkdir()
            (external / "__init__.py").write_text(
                "raise RuntimeError('external scripts package selected')\n",
                encoding="utf-8",
            )
            environment = dict(os.environ, PYTHONPATH=directory)
            code = (
                "from pathlib import Path; import scripts; "
                "from scripts.phase2 import model_family; "
                "from scripts.analyze_phase2 import analyze; "
                "assert Path(scripts.__file__).resolve() == "
                "Path('scripts/__init__.py').resolve()"
            )
            for arguments in (["-c", code], ["scripts/analyze_phase2.py", "--help"]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, *arguments], cwd=ROOT, env=environment,
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
