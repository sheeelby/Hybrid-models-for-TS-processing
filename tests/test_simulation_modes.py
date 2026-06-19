from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hybridts.pipelines._simulation import (
    clamp_simulation_mixed_alpha,
    normalize_simulation_mode,
    simulation_context_window,
)


class SimulationModeTests(unittest.TestCase):
    def test_normalize_simulation_mode_defaults_to_rollout(self) -> None:
        self.assertEqual(normalize_simulation_mode(None), "rollout")
        self.assertEqual(normalize_simulation_mode("unknown"), "rollout")
        self.assertEqual(normalize_simulation_mode("mixed"), "mixed")

    def test_clamp_simulation_mixed_alpha_bounds_values(self) -> None:
        self.assertAlmostEqual(clamp_simulation_mixed_alpha(-1.0), 0.0, places=9)
        self.assertAlmostEqual(clamp_simulation_mixed_alpha(2.0), 1.0, places=9)
        self.assertAlmostEqual(clamp_simulation_mixed_alpha(0.25), 0.25, places=9)

    def test_mixed_window_blends_truth_and_rollout_inside_train(self) -> None:
        history = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
        out = np.array([10.0, 19.0, 29.0, 39.0, 49.0, 59.0], dtype=float)

        fitted = simulation_context_window(history, out, 4, 3, mode="fitted")
        rollout = simulation_context_window(history, out, 4, 3, mode="rollout")
        mixed = simulation_context_window(history, out, 4, 3, mode="mixed", mixed_alpha=0.5)

        np.testing.assert_allclose(fitted, np.array([20.0, 30.0, 40.0]))
        np.testing.assert_allclose(rollout, np.array([19.0, 29.0, 39.0]))
        np.testing.assert_allclose(mixed, np.array([19.5, 29.5, 39.5]))

    def test_mixed_window_uses_available_truth_on_first_test_step(self) -> None:
        history = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
        out = np.array([10.0, 19.0, 29.0, 39.0, 49.0, 59.0], dtype=float)

        fitted = simulation_context_window(history, out, 5, 3, mode="fitted")
        rollout = simulation_context_window(history, out, 5, 3, mode="rollout")
        mixed = simulation_context_window(history, out, 5, 3, mode="mixed", mixed_alpha=0.5)

        np.testing.assert_allclose(fitted, np.array([29.0, 39.0, 49.0]))
        np.testing.assert_allclose(rollout, np.array([29.0, 39.0, 49.0]))
        np.testing.assert_allclose(mixed, np.array([29.5, 39.5, 49.5]))


if __name__ == "__main__":
    unittest.main()
