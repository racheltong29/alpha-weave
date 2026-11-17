"""LLM-driven factor hypothesis generation."""

import re
from collections import Counter

import numpy as np

from alphaweave.config import load_settings
from alphaweave.factors.registry import FactorHypothesis

_PROPOSE_FACTORS_TOOL = {
    "name": "propose_factors",
    "description": "Propose candidate quant trading-signal factor hypotheses.",
    "input_schema": {
        "type": "object",
        "properties": {
            "factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short kebab-case slug, e.g. 'guidance-tone-shift'.",
                        },
                        "description": {
                            "type": "string",
                            "description": "One or two sentences: the hypothesized signal and why it might predict returns.",
                        },
                    },
                    "required": ["id", "description"],
                },
            }
        },
        "required": ["factors"],
    },
}

_SYSTEM_PROMPT = (
    "You are a quant researcher proposing candidate trading-signal factors "
    "from unstructured context (earnings transcripts, filings, news). "
    "Do not restate well-known factors (momentum, value, size, quality, "
    "low-volatility) in different words -- a downstream embedding-similarity "
    "gate will reject those, wasting backtest compute. Each factor must be "
    "specific enough to be computed as a per-ticker, per-period numeric "
    "signal from the given context."
)


def propose_factors(context: str, n: int = 5) -> list[FactorHypothesis]:
    """Ask Claude to propose `n` candidate factor hypotheses from `context`
    (e.g. a batch of recent transcripts/filings/news summaries).

    Uses structured tool-call output so the response is a typed list rather
    than free-form prose that would need re-parsing.
    """
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot call the hypothesis-generation API."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        tools=[_PROPOSE_FACTORS_TOOL],
        tool_choice={"type": "tool", "name": "propose_factors"},
        messages=[
            {
                "role": "user",
                "content": f"Propose {n} candidate factors from this context:\n\n{context}",
            }
        ],
    )

    tool_use = next(block for block in response.content if block.type == "tool_use")
    factors = tool_use.input["factors"]
    return [
        FactorHypothesis(
            id=factor["id"],
            description=factor["description"],
            source_context=context,
        )
        for factor in factors[:n]
    ]


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _bag_of_words_vector(text: str) -> Counter:
    return Counter(_TOKEN_RE.findall(text.lower()))


def _cosine_similarity(a: Counter, b: Counter) -> float:
    shared = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in shared)
    norm_a = np.sqrt(sum(count**2 for count in a.values()))
    norm_b = np.sqrt(sum(count**2 for count in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(numerator / (norm_a * norm_b))


def is_near_duplicate(
    new_description: str, existing_descriptions: list[str], threshold: float = 0.9
) -> bool:
    """Reject a hypothesis that's just a reworded version of one already in
    the registry.

    Implemented as bag-of-words cosine similarity so the dedup gate needs no
    extra ML dependencies beyond what's already installed. Swapping in a real
    sentence embedding (e.g. bge-m3 via `transformers`) is a drop-in
    replacement: keep this function's signature, replace
    `_bag_of_words_vector` with an embedding call and `_cosine_similarity`
    with a dot-product over normalized embedding vectors.
    """
    if not existing_descriptions:
        return False
    new_vector = _bag_of_words_vector(new_description)
    similarities = [
        _cosine_similarity(new_vector, _bag_of_words_vector(existing))
        for existing in existing_descriptions
    ]
    return max(similarities) >= threshold
