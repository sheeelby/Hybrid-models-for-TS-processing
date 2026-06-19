from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hybridts.pipelines.synth_eval import PROFILES, PROFILE_GROUPS, SynthProfile, _resolve_profiles, _trend_series


class SynthProfileTrendTests(unittest.TestCase):
    def test_smooth_slope_transition_changes_late_slope(self) -> None:
        profile = SynthProfile(
            name="smooth_transition",
            trend_slope=0.01,
            season_period=None,
            season_amp=0.0,
            noise_std=0.0,
            trend_slope_transitions=((0.5, 0.03, 0.05),),
        )

        trend = _trend_series(length=240, profile=profile)
        diffs = np.diff(trend)

        self.assertTrue(profile.has_complex_trend)
        self.assertAlmostEqual(float(trend[0]), 0.0, places=7)
        self.assertLess(float(np.mean(diffs[20:60])), 0.02)
        self.assertGreater(float(np.mean(diffs[180:220])), 0.03)

    def test_gaussian_bumps_create_local_peak_and_dip(self) -> None:
        profile = SynthProfile(
            name="bumpy_trend",
            trend_slope=0.0,
            season_period=None,
            season_amp=0.0,
            noise_std=0.0,
            trend_bumps=((0.3, 2.5, 0.05), (0.72, -2.0, 0.06)),
        )

        trend = _trend_series(length=240, profile=profile)
        peak_idx = int(np.argmax(trend))
        dip_idx = int(np.argmin(trend))

        self.assertAlmostEqual(float(trend[0]), 0.0, places=7)
        self.assertGreater(float(np.max(trend)), 1.5)
        self.assertLess(float(np.min(trend)), -1.0)
        self.assertTrue(55 <= peak_idx <= 90)
        self.assertTrue(145 <= dip_idx <= 195)

    def test_complex_trend_group_uses_new_profiles(self) -> None:
        expected = (
            "complex_trend_sigmoid_saturation",
            "complex_trend_staircase_plateaus",
            "complex_trend_dip_rebound_seasonal",
        )

        self.assertEqual(PROFILE_GROUPS["complex_trend_examples"], expected)
        self.assertTrue(all(name in PROFILES for name in expected))
        resolved = _resolve_profiles(["complex_trend_examples"])
        self.assertEqual(tuple(profile.name for profile in resolved), expected)


if __name__ == "__main__":
    unittest.main()
