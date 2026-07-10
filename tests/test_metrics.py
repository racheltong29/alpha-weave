import numpy as np
import pytest

from alphaweave.backtest.metrics import benjamini_hochberg, max_drawdown, sharpe_ratio, t_stat


def test_sharpe_ratio_zero_vol_returns_nan():
    returns = np.array([0.01, 0.01, 0.01, 0.01])
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_known_value():
    # Constant excess return with known std should match hand-computed Sharpe.
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.001, scale=0.01, size=2520)
    result = sharpe_ratio(returns, periods_per_year=252)
    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)
    assert result == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_simple_case():
    # Prices go up 10%, down 20%, up 5% -> drawdown from peak of 1.10 to 0.88
    returns = np.array([0.10, -0.20, 0.05])
    dd = max_drawdown(returns)
    cumulative = np.cumprod(1 + returns)
    expected = (cumulative.min() / cumulative[: np.argmin(cumulative) + 1].max()) - 1
    assert dd == pytest.approx(expected, rel=1e-9)
    assert dd < 0


def test_max_drawdown_all_positive_is_zero():
    returns = np.array([0.01, 0.02, 0.01, 0.03])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-12)


def test_t_stat_significant_positive_mean():
    returns = np.full(100, 0.01) + np.random.default_rng(1).normal(0, 0.0001, 100)
    assert t_stat(returns) > 0


def test_benjamini_hochberg_all_significant():
    p_values = [0.001, 0.002, 0.003]
    result = benjamini_hochberg(p_values, fdr=0.05)
    assert all(result)


def test_benjamini_hochberg_mixed():
    p_values = [0.001, 0.5, 0.8, 0.9]
    result = benjamini_hochberg(p_values, fdr=0.05)
    assert result[0] is True
    assert result[2] is False
    assert result[3] is False


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg([]) == []
