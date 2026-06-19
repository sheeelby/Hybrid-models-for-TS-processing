from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hybridts.hybrids import VWHybridMixed
from hybridts.hybrids.decomposition import DecompositionSpec, decompose_series
from hybridts.pipelines._model_specs import parse_model_spec
from hybridts.training import TrainConfig


def _linear_base_model(cfg: TrainConfig) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(cfg.lookback, cfg.horizon),
    )


def _zero_base_model(cfg: TrainConfig) -> torch.nn.Module:
    layer = torch.nn.Linear(cfg.lookback, cfg.horizon)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        layer,
    )


class VWHybridMixedTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        torch.manual_seed(0)

    def test_stl_model_spec_uses_two_neural_components(self) -> None:
        self.assertEqual(parse_model_spec("stl_timesnet_ets").neural_component_count, 2)
        self.assertEqual(parse_model_spec("vw_timesnet_ets").neural_component_count, 1)

    def test_stl_mixed_fit_forecasts_two_neural_components(self) -> None:
        horizon = 6
        period = 12
        t = np.arange(120, dtype=float)
        y = 0.08 * t + 1.5 * np.sin(2.0 * np.pi * t / period) + 0.05 * np.cos(2.0 * np.pi * t / 6.0)

        dec = decompose_series(
            y,
            spec=DecompositionSpec(method="stl", seasonal_period=period),
            check=True,
        )
        cfg = TrainConfig(
            lookback=24,
            horizon=horizon,
            epochs=0,
            batch_size=8,
            device="cpu",
        )
        model = VWHybridMixed(
            aj_model_fn=_linear_base_model,
            neural_component_count=2,
            detail_method="ets",
            cfg=cfg,
            seasonal_period=period,
            component_names=dec.names,
            enable_output_blend=False,
        ).fit(y, components_override=list(dec.components))

        self.assertEqual(len(model.neural_components), 2)
        self.assertIs(model.aj_component, model.neural_components[0])

        forecast = model.forecast(y, components_override=list(dec.components))
        self.assertEqual(forecast.shape, (horizon,))

        component_forecast = model.forecast_components(y, components_override=list(dec.components))
        self.assertEqual(set(component_forecast.keys()), set(dec.names))
        for name in dec.names:
            self.assertEqual(component_forecast[name].shape, (horizon,))

    def test_stl_mixed_prefers_classical_fallback_for_bad_seasonal_neural_component(self) -> None:
        horizon = 6
        period = 12
        t = np.arange(144, dtype=float)
        y = 0.03 * t + 2.0 * np.sin(2.0 * np.pi * t / period)

        dec = decompose_series(
            y,
            spec=DecompositionSpec(method="stl", seasonal_period=period),
            check=True,
        )
        cfg = TrainConfig(
            lookback=24,
            horizon=horizon,
            epochs=0,
            batch_size=8,
            device="cpu",
        )
        model = VWHybridMixed(
            aj_model_fn=_zero_base_model,
            neural_component_count=2,
            detail_method="ets",
            cfg=cfg,
            seasonal_period=period,
            seasonal_periods=(period,),
            component_names=dec.names,
            enable_output_blend=False,
        ).fit(y, components_override=list(dec.components))

        self.assertGreaterEqual(len(model.neural_policies), 2)
        self.assertEqual(model.neural_policies[1].baseline_kind, "ets")
        self.assertEqual(model.neural_policies[1].baseline_period, period)

    def test_modwt_mixed_can_aggregate_low_frequency_components(self) -> None:
        horizon = 6
        t = np.arange(144, dtype=float)
        y = 0.04 * t + 1.8 * np.sin(2.0 * np.pi * t / 12.0) + 0.25 * np.cos(2.0 * np.pi * t / 3.0)

        dec = decompose_series(
            y,
            spec=DecompositionSpec(method="modwt", wavelet="db4", level=3, boundary="reflect"),
            check=True,
        )
        cfg = TrainConfig(
            lookback=24,
            horizon=horizon,
            epochs=0,
            batch_size=8,
            device="cpu",
        )
        model = VWHybridMixed(
            aj_model_fn=_linear_base_model,
            neural_component_count=1,
            detail_method="ets",
            cfg=cfg,
            wavelet="db4",
            level=3,
            boundary="reflect",
            component_names=dec.names,
            aggregate_low_freq_components=3,
            enable_output_blend=False,
        ).fit(y, components_override=list(dec.components))

        self.assertLess(len(model.component_names_), len(dec.names))
        self.assertTrue(model.component_names_[0].startswith("LOWFREQ("))

        forecast = model.forecast(y, components_override=list(dec.components))
        self.assertEqual(forecast.shape, (horizon,))

        component_forecast = model.forecast_components(y, components_override=list(dec.components))
        self.assertEqual(set(component_forecast.keys()), set(model.component_names_))
        for name in model.component_names_:
            self.assertEqual(component_forecast[name].shape, (horizon,))


if __name__ == "__main__":
    unittest.main()
