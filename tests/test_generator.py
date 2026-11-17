from unittest.mock import MagicMock, patch

import pytest

from alphaweave.factors.generator import is_near_duplicate, propose_factors


def test_is_near_duplicate_true_for_identical_text():
    assert is_near_duplicate(
        "companies with rising insider buying outperform",
        ["companies with rising insider buying outperform"],
    )


def test_is_near_duplicate_false_for_unrelated_text():
    assert not is_near_duplicate(
        "supply chain disruption mentions predict margin misses",
        ["job posting growth predicts revenue surprises"],
    )


def test_is_near_duplicate_false_when_registry_empty():
    assert not is_near_duplicate("any description", [])


def test_is_near_duplicate_respects_threshold():
    new = "insider buying activity signals future outperformance"
    existing = ["insider buying activity strongly signals outperformance"]
    assert is_near_duplicate(new, existing, threshold=0.5)
    assert not is_near_duplicate(new, existing, threshold=0.999)


def test_propose_factors_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        propose_factors("some transcript context")


def test_propose_factors_parses_tool_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.input = {
        "factors": [
            {"id": "guidance-tone-shift", "description": "desc one"},
            {"id": "supplier-mentions", "description": "desc two"},
        ]
    }
    mock_response = MagicMock()
    mock_response.content = [tool_use_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        factors = propose_factors("context text", n=2)

    assert len(factors) == 2
    assert factors[0].id == "guidance-tone-shift"
    assert factors[0].source_context == "context text"
