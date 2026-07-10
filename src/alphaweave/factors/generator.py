"""LLM-driven factor hypothesis generation."""

from alphaweave.factors.registry import FactorHypothesis


def propose_factors(context: str, n: int = 5) -> list[FactorHypothesis]:
    """Ask an LLM to propose `n` candidate factor hypotheses from `context`
    (e.g. a batch of recent transcripts/filings/news summaries).

    Stub: real implementation calls the Anthropic API with a structured
    output schema (id, description, rationale) and should explicitly ask
    the model to avoid restating well-known factors (momentum, value, size)
    since the dedup gate downstream is a second line of defense, not the
    first (README weeks 5-6).
    """
    raise NotImplementedError("Wire up Claude API call with structured output.")


def is_near_duplicate(
    new_description: str, existing_descriptions: list[str], threshold: float = 0.9
) -> bool:
    """Embedding-similarity check to reject a hypothesis that's just a
    reworded version of one already in the registry.

    Stub: real implementation embeds all descriptions (e.g. bge-m3) and
    compares cosine similarity against `threshold`.
    """
    raise NotImplementedError("Wire up embedding similarity check.")
