"""Hand-written 12-1 month price momentum factor.

The first factor run end-to-end through the engine (README weeks 3-4),
before hypothesis generation is automated. Kept separate from
`signals.py`'s `FactorHypothesis` dispatch since it isn't LLM-proposed.
"""

import numpy as np
import pandas as pd


def compute_momentum_signal(
    close: pd.Series, lookback: int = 126, skip: int = 21
) -> pd.Series:
    """Cross-sectional-free (single-ticker) momentum: past return over the
    `lookback` window ending `skip` periods ago, so the most recent `skip`
    periods (proxying for short-term reversal / not-yet-settled data) are
    excluded. Scaled to [-1, 1] via tanh.

    Signal at date T uses only prices strictly before T (via `.shift(skip)`
    on top of the lookback return), so it is point-in-time correct.
    """
    past_return = close.pct_change(lookback).shift(skip)
    return np.tanh(past_return).rename("signal")
