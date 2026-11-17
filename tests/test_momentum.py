import numpy as np
import pandas as pd

from alphaweave.factors.momentum import compute_momentum_signal


def _prices(n=300, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=dates, name="close"
    )


def test_momentum_signal_bounded():
    signal = compute_momentum_signal(_prices())
    valid = signal.dropna()
    assert (valid.abs() <= 1).all()


def test_momentum_signal_uses_only_past_data():
    close = _prices()
    signal = compute_momentum_signal(close, lookback=126, skip=21)
    # A signal at date T must not change if future prices change.
    perturbed = close.copy()
    cutoff = close.index[200]
    perturbed.loc[perturbed.index > cutoff] *= 2.0
    perturbed_signal = compute_momentum_signal(perturbed, lookback=126, skip=21)
    pd.testing.assert_series_equal(
        signal.loc[:cutoff].dropna(), perturbed_signal.loc[:cutoff].dropna()
    )


def test_momentum_signal_all_nan_for_short_series():
    close = _prices(n=50)
    signal = compute_momentum_signal(close, lookback=126, skip=21)
    assert signal.dropna().empty
