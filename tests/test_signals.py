from unittest.mock import patch

import pandas as pd
import pytest

from alphaweave.factors.registry import FactorHypothesis
from alphaweave.factors.signals import compute_signal


def _fake_classifier(texts, top_k=None, truncation=None, max_length=None):
    # Mimic transformers' top_k=None output shape: one label/score list per input.
    outputs = []
    for text in texts:
        if "raised guidance" in text:
            outputs.append([{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.05}])
        else:
            outputs.append([{"label": "positive", "score": 0.1}, {"label": "negative", "score": 0.8}])
    return outputs


def test_compute_signal_requires_text_column():
    hypothesis = FactorHypothesis(id="h1", description="d", source_context="c")
    data = pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2023-01-01")])
    with pytest.raises(ValueError):
        compute_signal(hypothesis, data)


def test_compute_signal_dispatches_to_finbert_sentiment():
    hypothesis = FactorHypothesis(id="h1", description="sentiment factor", source_context="c")
    dates = pd.date_range("2023-01-01", periods=2)
    data = pd.DataFrame(
        {"text": ["company raised guidance for the year", "results missed expectations"]},
        index=dates,
    )
    with patch(
        "alphaweave.factors.signals._finbert_pipeline",
        return_value=_fake_classifier,
    ):
        signal = compute_signal(hypothesis, data)

    assert signal.name == "signal"
    assert signal.iloc[0] == pytest.approx(0.85)
    assert signal.iloc[1] == pytest.approx(-0.7)
