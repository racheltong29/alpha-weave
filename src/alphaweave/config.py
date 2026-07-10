"""Environment-based settings for AlphaWeave."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    alpha_vantage_key: str | None
    data_dir: str
    trading_days_per_year: int = 252
    risk_free_rate_annual: float = 0.0


def load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        alpha_vantage_key=os.environ.get("ALPHA_VANTAGE_KEY"),
        data_dir=os.environ.get("ALPHAWEAVE_DATA_DIR", "./data"),
    )
