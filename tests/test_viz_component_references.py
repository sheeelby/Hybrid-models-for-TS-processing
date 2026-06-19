from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hybridts.viz import _resolve_reference_components_for_plot


class VizComponentReferenceTests(unittest.TestCase):
    def test_resolves_aggregated_low_frequency_reference_components(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        d1 = np.array([0.1, 0.2, 0.3])
        d2 = np.array([0.4, 0.5, 0.6])
        d3 = np.array([0.7, 0.8, 0.9])

        resolved = _resolve_reference_components_for_plot(
            component_forecasts={
                "VW + N-Beats + ETS": {
                    "LOWFREQ(A_J + D_1 + D_2)": np.array([0.0, 0.0, 0.0]),
                    "D_3": np.array([0.0, 0.0, 0.0]),
                }
            },
            reference_names=("A_J", "D_1", "D_2", "D_3"),
            reference_components=(a, d1, d2, d3),
        )

        self.assertEqual([name for name, _ in resolved], ["LOWFREQ(A_J + D_1 + D_2)", "D_3"])
        np.testing.assert_allclose(resolved[0][1], a + d1 + d2)
        np.testing.assert_allclose(resolved[1][1], d3)

    def test_falls_back_to_original_components_when_names_are_unknown(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        d1 = np.array([0.1, 0.2, 0.3])

        resolved = _resolve_reference_components_for_plot(
            component_forecasts={
                "model": {
                    "mystery_component": np.array([0.0, 0.0, 0.0]),
                }
            },
            reference_names=("A_J", "D_1"),
            reference_components=(a, d1),
        )

        self.assertEqual([name for name, _ in resolved], ["A_J", "D_1"])
        np.testing.assert_allclose(resolved[0][1], a)
        np.testing.assert_allclose(resolved[1][1], d1)


if __name__ == "__main__":
    unittest.main()
