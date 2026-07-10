"""Vectorized walk-forward backtest engine.

This is an interface-level scaffold: the signatures and control flow are
real, but the transaction-cost model and walk-forward splitting logic are
minimal placeholders to be filled in during the roadmap's weeks 3-4.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaweave.backtest.metrics import max_drawdown, sharpe_ratio, t_stat


@dataclass
class BacktestResult:
    returns: pd.Series
    sharpe: float
    max_dd: float
    t_stat: float
    metadata: dict = field(default_factory=dict)


def run_backtest(
    signal: pd.Series,
    prices: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a long/short backtest of `signal` against `prices`.

    `signal` is indexed by (date, ticker) and gives a position weight in
    [-1, 1] known only from data available as of `date` (point-in-time).
    `prices` must contain a 'close' column indexed the same way.

    NOTE: transaction-cost and position-sizing logic here is a placeholder.
    Real implementation must ensure `signal` at date T only used information
    available strictly before T's close to avoid lookahead bias.
    """
    aligned = prices["close"].pct_change().reindex(signal.index)
    gross_returns = signal.shift(1) * aligned
    turnover = signal.diff().abs()
    costs = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = (gross_returns - costs).dropna()

    returns_array = net_returns.to_numpy()
    return BacktestResult(
        returns=net_returns,
        sharpe=sharpe_ratio(returns_array, periods_per_year=periods_per_year),
        max_dd=max_drawdown(returns_array),
        t_stat=t_stat(returns_array),
        metadata={"transaction_cost_bps": transaction_cost_bps},
    )


def walk_forward_splits(
    dates: pd.DatetimeIndex, train_years: int = 3, test_years: int = 1
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield rolling (train, test) date-index splits for out-of-sample validation.

    Placeholder: splits by calendar year boundaries. Real implementation
    should be careful about partial final windows and gap periods between
    train/test to avoid leakage.
    """
    years = sorted(set(dates.year))
    splits = []
    for start_idx in range(len(years) - train_years - test_years + 1):
        train_years_set = years[start_idx : start_idx + train_years]
        test_years_set = years[
            start_idx + train_years : start_idx + train_years + test_years
        ]
        train_mask = dates.year.isin(train_years_set)
        test_mask = dates.year.isin(test_years_set)
        splits.append((dates[train_mask], dates[test_mask]))
    return splits
