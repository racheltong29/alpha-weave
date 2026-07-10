import numpy as np
import pandas as pd

from alphaweave.backtest.engine import run_backtest, walk_forward_splits


def _synthetic_prices(n=252, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"close": prices}, index=dates)


def test_run_backtest_smoke():
    prices = _synthetic_prices()
    signal = pd.Series(1.0, index=prices.index)  # always long
    result = run_backtest(signal, prices)
    assert len(result.returns) > 0
    assert isinstance(result.sharpe, float)
    assert result.max_dd <= 0


def test_run_backtest_flat_signal_has_zero_turnover_cost():
    prices = _synthetic_prices()
    signal = pd.Series(0.0, index=prices.index)
    result = run_backtest(signal, prices)
    assert np.allclose(result.returns.to_numpy(), 0.0)


def test_walk_forward_splits_basic():
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    splits = walk_forward_splits(dates, train_years=3, test_years=1)
    assert len(splits) > 0
    for train, test in splits:
        assert train.max() < test.min()
