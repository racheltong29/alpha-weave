"""Turn a factor hypothesis into a computable per-ticker signal series."""

import pandas as pd

from alphaweave.factors.registry import FactorHypothesis


def compute_signal(hypothesis: FactorHypothesis, data: pd.DataFrame) -> pd.Series:
    """Compute a position-weight signal in [-1, 1] for `hypothesis` from `data`.

    Stub: real implementation dispatches on the hypothesis (e.g. FinBERT
    sentiment score, NER-derived entity counts, zero-shot event tags)
    normalized into a comparable signal scale. Must only use data available
    as of each date (point-in-time) — see engine.py's lookahead-bias note.
    """
    raise NotImplementedError("Wire up signal computation dispatch (README weeks 5-6).")
