"""Backtest performance metrics.

These are the ground truth every downstream factor-acceptance decision rests
on, so they are fully implemented and unit-tested against hand-computed
values rather than left as stubs.
"""

import numpy as np
from scipy import stats


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio of a periodic (e.g. daily) return series."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return float("nan")
    period_rf = risk_free_rate_annual / periods_per_year
    excess = returns - period_rf
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the cumulative return curve.

    Returned as a negative fraction, e.g. -0.23 for a 23% drawdown.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return float("nan")
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1.0
    return float(drawdowns.min())


def t_stat(returns: np.ndarray) -> float:
    """One-sample t-statistic testing whether mean return differs from zero."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return float("nan")
    result = stats.ttest_1samp(returns, popmean=0.0)
    return float(result.statistic)


def p_value(returns: np.ndarray) -> float:
    """Two-sided p-value for the same one-sample t-test as `t_stat`.

    Kept separate from `t_stat` (rather than returned as a tuple) so each
    metric can be reported independently, but both come from the same
    `ttest_1samp` call conceptually -- this is the input `benjamini_hochberg`
    expects when correcting across many tested hypotheses.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return float("nan")
    result = stats.ttest_1samp(returns, popmean=0.0)
    return float(result.pvalue)


def benjamini_hochberg(p_values: list[float], fdr: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction.

    Returns a boolean list (same order as input) indicating which hypotheses
    remain significant after correcting for the number of factors tested.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * fdr
    passed = ranked <= thresholds
    if not passed.any():
        return [False] * n
    max_rank = np.max(np.where(passed)[0])
    significant = np.zeros(n, dtype=bool)
    significant[order[: max_rank + 1]] = True
    return significant.tolist()
