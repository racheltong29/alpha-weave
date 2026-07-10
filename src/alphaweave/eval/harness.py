"""Regression-gate harness: blocks engine/signal changes that silently
degrade backtest reliability (most often via a reintroduced lookahead bug).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaweave.backtest.engine import run_backtest


@dataclass
class RegressionGateResult:
    passed: bool
    baseline_sharpe: float
    current_sharpe: float
    tolerance: float


def check_benchmark_factor_regression(
    signal: pd.Series,
    prices: pd.DataFrame,
    baseline_sharpe: float,
    tolerance: float = 0.15,
) -> RegressionGateResult:
    """Re-run a frozen benchmark factor and compare its Sharpe to a stored
    baseline. A drop beyond `tolerance` fails CI.

    This is the gate that catches, e.g., an accidental change to
    `signal.shift(1)` in engine.py that would silently introduce lookahead
    bias and inflate every future factor's backtest results.
    """
    result = run_backtest(signal, prices)
    current = result.sharpe
    passed = not np.isnan(current) and (baseline_sharpe - current) <= tolerance
    return RegressionGateResult(
        passed=passed,
        baseline_sharpe=baseline_sharpe,
        current_sharpe=current,
        tolerance=tolerance,
    )
