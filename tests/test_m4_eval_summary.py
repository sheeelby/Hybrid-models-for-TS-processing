from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hybridts.pipelines._model_specs import parse_model_spec
from hybridts.pipelines.m4_eval import (
    _build_family_comparison_df,
    _build_train_config,
    _merge_method_kwargs,
    _resolve_vw_kwargs,
)


class M4EvalSummaryTests(unittest.TestCase):
    def test_merge_method_kwargs_overrides_shared_values(self) -> None:
        merged = _merge_method_kwargs(
            {"detail_anchor": True, "aggregate_low_freq_components": 1},
            by_method={
                "modwt": {
                    "aggregate_low_freq_components": 3,
                    "detail_dampen": False,
                }
            },
            method="modwt",
        )

        self.assertEqual(
            merged,
            {
                "detail_anchor": True,
                "aggregate_low_freq_components": 3,
                "detail_dampen": False,
            },
        )

    def test_resolve_vw_kwargs_extracts_neural_component_count_override(self) -> None:
        neural_component_count, kwargs = _resolve_vw_kwargs(
            {"detail_anchor": True},
            by_method={
                "stl": {
                    "neural_component_count": 1,
                    "detail_dampen": False,
                }
            },
            method="stl",
            default_neural_component_count=2,
        )

        self.assertEqual(neural_component_count, 1)
        self.assertEqual(
            kwargs,
            {
                "detail_anchor": True,
                "detail_dampen": False,
            },
        )

    def test_build_train_config_applies_overrides(self) -> None:
        cfg = _build_train_config(
            lookback=16,
            horizon=8,
            epochs=3,
            train_kwargs={
                "batch_size": 16,
                "lr": 1e-3,
                "device": "cpu",
            },
        )

        self.assertEqual(cfg.lookback, 16)
        self.assertEqual(cfg.horizon, 8)
        self.assertEqual(cfg.epochs, 3)
        self.assertEqual(cfg.batch_size, 16)
        self.assertAlmostEqual(cfg.lr, 1e-3, places=9)
        self.assertAlmostEqual(cfg.weight_decay, 2e-4, places=9)
        self.assertAlmostEqual(cfg.clip or 0.0, 0.5, places=9)
        self.assertEqual(cfg.device, "cpu")

    def test_family_summary_prefers_best_modwt_stl_and_classical_models(self) -> None:
        modwt_label = parse_model_spec("vw_nbeats_ets").label
        stl_label = parse_model_spec("stl_nbeats_ets").label
        df = pd.DataFrame(
            [
                {
                    "category": "monthly",
                    "series_id": "S1",
                    f"{modwt_label.replace(' ', '_')}_sMAPE": 4.0,
                    f"{stl_label.replace(' ', '_')}_sMAPE": 6.0,
                    "ETS_sMAPE": 5.0,
                    "ARIMA_sMAPE": 7.0,
                },
                {
                    "category": "monthly",
                    "series_id": "S2",
                    f"{modwt_label.replace(' ', '_')}_sMAPE": 6.0,
                    f"{stl_label.replace(' ', '_')}_sMAPE": 8.0,
                    "ETS_sMAPE": 7.0,
                    "ARIMA_sMAPE": 9.0,
                },
            ]
        )

        summary_df = _build_family_comparison_df(
            df,
            hybrid_models=(
                parse_model_spec("vw_nbeats_ets"),
                parse_model_spec("stl_nbeats_ets"),
            ),
            metric_names=("sMAPE",),
        )

        self.assertFalse(summary_df.empty)
        row = summary_df[
            (summary_df["scope"] == "monthly")
            & (summary_df["metric"] == "sMAPE")
        ].iloc[0]

        self.assertEqual(row["modwt_best_model"], modwt_label)
        self.assertAlmostEqual(float(row["modwt_score"]), 5.0, places=6)
        self.assertEqual(row["stl_best_model"], stl_label)
        self.assertAlmostEqual(float(row["stl_score"]), 7.0, places=6)
        self.assertEqual(row["classical_best_model"], "ETS")
        self.assertAlmostEqual(float(row["classical_score"]), 6.0, places=6)
        self.assertAlmostEqual(float(row["modwt_minus_stl"]), -2.0, places=6)
        self.assertAlmostEqual(float(row["modwt_minus_classical"]), -1.0, places=6)
        self.assertAlmostEqual(float(row["stl_minus_classical"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
