import numpy as np
import pandas as pd

from alphaweave.eval.harness import check_benchmark_factor_regression


def _frozen_benchmark():
    """A small, fixed dataset standing in for the real frozen CI benchmark:
    same signal/prices every run, so the resulting Sharpe never drifts on
    its own -- only a code change to engine.py can move it.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    prices = pd.DataFrame(
        {"close": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))},
        index=dates,
    )
    signal = pd.Series(1.0, index=dates)
    return signal, prices


def test_regression_gate_passes_against_own_baseline():
    signal, prices = _frozen_benchmark()
    from alphaweave.backtest.engine import run_backtest

    baseline_sharpe = run_backtest(signal, prices).sharpe

    result = check_benchmark_factor_regression(signal, prices, baseline_sharpe)

    assert result.passed
    assert result.current_sharpe == baseline_sharpe


def test_regression_gate_fails_when_sharpe_drops_beyond_tolerance():
    signal, prices = _frozen_benchmark()
    result = check_benchmark_factor_regression(
        signal, prices, baseline_sharpe=10.0, tolerance=0.15
    )
    assert not result.passed


def test_regression_gate_fails_on_nan_sharpe():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    flat_prices = pd.DataFrame({"close": [100.0] * 5}, index=dates)
    signal = pd.Series(0.0, index=dates)
    result = check_benchmark_factor_regression(signal, flat_prices, baseline_sharpe=0.0)
    assert not result.passed
