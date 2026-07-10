"""Factor hypothesis registry: versioned record of every factor tried."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FactorHypothesis:
    id: str
    description: str
    source_context: str
    created_at: float = field(default_factory=time.time)
    status: str = "proposed"  # proposed | kept | rejected
    sharpe: float | None = None
    max_drawdown: float | None = None
    t_stat: float | None = None
    rejection_reason: str | None = None


class FactorRegistry:
    """JSON-file-backed store of factor hypotheses and their outcomes.

    A real database (Postgres) is the scale-out path; a flat file is
    sufficient and fully reproducible for a solo project.
    """

    def __init__(self, path: str = "./data/factor_registry.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text())

    def _save(self, records: list[dict]) -> None:
        self.path.write_text(json.dumps(records, indent=2))

    def add(self, hypothesis: FactorHypothesis) -> None:
        records = self._load()
        records.append(asdict(hypothesis))
        self._save(records)

    def all(self) -> list[dict]:
        return self._load()

    def kept(self) -> list[dict]:
        return [r for r in self._load() if r["status"] == "kept"]

    def rejected(self) -> list[dict]:
        return [r for r in self._load() if r["status"] == "rejected"]
