# Factor Experiment Log

Every factor hypothesis tested, with its stats and the keep/reject decision.
This log is a first-class portfolio artifact: it shows research process
discipline, not just cherry-picked winners. Append a row per experiment —
including (especially) the failures.

Columns:
- **Factor**: short name + one-line description of the hypothesis
- **Source**: what data/context the LLM generated it from
- **OOS Sharpe**: mean out-of-sample Sharpe across walk-forward windows
- **Max DD**: worst drawdown across windows
- **t-stat / BH-sig**: raw t-stat and whether it survives Benjamini-Hochberg
- **Corr (max)**: highest |correlation| to any already-kept factor
- **Decision**: kept / rejected + reason

| Date | Factor | Source | OOS Sharpe | Max DD | t-stat / BH-sig | Corr (max) | Decision |
|---|---|---|---|---|---|---|---|
| _2026-__-___ | _example: CFO hedging-language intensity_ | _Q3 transcripts, 30 tickers_ | _—_ | _—_ | _—_ | _—_ | _—_ |

## Notes / lessons

Record lookahead-bias bugs caught, data-quality issues found, and prompt
changes to the hypothesis generator here, with dates. These notes are what
a quant interviewer will read most closely.
