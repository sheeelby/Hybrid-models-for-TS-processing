from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_synth
import synth_eval


class EvalSynthCliTests(unittest.TestCase):
    def test_resolve_config_path_accepts_short_config_name(self) -> None:
        resolved = eval_synth.resolve_config_path(Path("synth_eval_complex_trend_modwt_safe.json"))
        self.assertEqual(
            resolved,
            ROOT / "configs" / "synth_eval_complex_trend_modwt_safe.json",
        )
        self.assertTrue(resolved.exists())

    def test_synth_eval_wrapper_reexports_main(self) -> None:
        self.assertIs(synth_eval.main, eval_synth.main)


if __name__ == "__main__":
    unittest.main()
