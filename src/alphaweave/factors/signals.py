"""Turn a factor hypothesis into a computable per-ticker signal series."""

from functools import lru_cache

import pandas as pd

from alphaweave.factors.registry import FactorHypothesis


@lru_cache(maxsize=1)
def _finbert_pipeline():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def _finbert_net_sentiment(texts: pd.Series) -> pd.Series:
    """Per-row FinBERT sentiment in [-1, 1]: P(positive) - P(negative)."""
    classifier = _finbert_pipeline()
    scores = classifier(texts.tolist(), top_k=None, truncation=True, max_length=512)
    net = []
    for row in scores:
        by_label = {entry["label"]: entry["score"] for entry in row}
        net.append(by_label.get("positive", 0.0) - by_label.get("negative", 0.0))
    return pd.Series(net, index=texts.index)


def compute_signal(hypothesis: FactorHypothesis, data: pd.DataFrame) -> pd.Series:
    """Compute a position-weight signal in [-1, 1] for `hypothesis` from `data`.

    `data` is indexed by (date, ticker) and must have a 'text' column (e.g.
    a transcript excerpt or news snippet known as of that date). Every
    generated hypothesis is currently dispatched to FinBERT sentiment
    scoring -- the cheapest enrichment signal in the README's HF task list.
    NER-derived entity counts and zero-shot event tags are a separate
    dispatch branch for a future hypothesis category, not implemented yet.

    The result already only reflects text known as of each row's date, so
    it's point-in-time correct without an extra shift here -- `engine.py`
    applies the one-period execution-lag shift at backtest time.
    """
    if "text" not in data.columns:
        raise ValueError(
            f"compute_signal for hypothesis '{hypothesis.id}' requires a 'text' "
            "column (transcript/news snippet known as of that date)."
        )
    return _finbert_net_sentiment(data["text"]).rename("signal")
