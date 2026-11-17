"""End-to-end generated-factor pipeline (README weeks 5-8):

    real earnings text (EDGAR) -> LLM hypothesis generation -> dedup gate
    -> FinBERT signal computation -> backtest -> Benjamini-Hochberg
    correction across the batch -> factor registry + eval-log.

Unlike `run_pipeline.py` (one hand-written momentum factor), every
hypothesis here is machine-generated and must independently survive the
same statistical gates before being kept.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from alphaweave.backtest.engine import run_backtest
from alphaweave.backtest.metrics import benjamini_hochberg, max_drawdown, p_value, sharpe_ratio, t_stat
from alphaweave.data.edgar import fetch_earnings_release_text
from alphaweave.data.ingest import fetch_ohlcv
from alphaweave.factors.generator import is_near_duplicate, propose_factors
from alphaweave.factors.registry import FactorHypothesis, FactorRegistry
from alphaweave.factors.signals import compute_signal

_EVAL_LOG_PATH = Path("docs/eval-log.md")
_SIGNAL_STALE_AFTER_DAYS = 90


def _build_context(texts: pd.DataFrame, snippet_chars: int = 500) -> str:
    parts = []
    for (date, ticker), row in texts.iterrows():
        parts.append(f"[{ticker} {date.date()}] {row['text'][:snippet_chars]}")
    return "\n\n".join(parts)


def _signal_for_ticker(
    hypothesis: FactorHypothesis, texts: pd.DataFrame, ticker: str, price_index: pd.DatetimeIndex
) -> pd.Series:
    ticker_texts = texts.xs(ticker, level="ticker") if ticker in texts.index.get_level_values("ticker") else texts.iloc[0:0]
    if ticker_texts.empty:
        return pd.Series(0.0, index=price_index)
    sparse_signal = compute_signal(hypothesis, ticker_texts)
    dense = sparse_signal.reindex(price_index.union(sparse_signal.index)).sort_index()
    dense = dense.ffill(limit=_SIGNAL_STALE_AFTER_DAYS).fillna(0.0)
    return dense.reindex(price_index).fillna(0.0)


def _append_eval_log_row(row: dict) -> None:
    if not _EVAL_LOG_PATH.exists():
        return
    lines = _EVAL_LOG_PATH.read_text().splitlines(keepends=True)
    insert_at = next(
        (i for i, line in enumerate(lines) if line.startswith("## Notes")), len(lines)
    )
    formatted = (
        f"| {row['date']} | {row['factor']} | {row['source']} | "
        f"{row['oos_sharpe']:.2f} | {row['max_dd']:.2%} | "
        f"{row['t_stat']:.2f} / {row['bh_sig']} | {row['corr_max']} | {row['decision']} |\n"
    )
    lines.insert(insert_at, formatted)
    _EVAL_LOG_PATH.write_text("".join(lines))


def run(tickers: list[str], start: str, end: str, n_factors: int, data_dir: str) -> None:
    prices = fetch_ohlcv(tickers, start=start, end=end)
    texts = fetch_earnings_release_text(tickers, start=start, end=end)
    if texts.empty:
        raise RuntimeError(
            f"No 8-K item-2.02 earnings releases found for {tickers} in {start}..{end}"
        )

    registry = FactorRegistry(path=f"{data_dir}/factor_registry.json")
    existing_descriptions = [record["description"] for record in registry.all()]

    context = _build_context(texts)
    proposed = propose_factors(context, n=n_factors)
    print(f"Proposed {len(proposed)} candidate factors.\n")

    surviving = []
    for hypothesis in proposed:
        if is_near_duplicate(hypothesis.description, existing_descriptions):
            hypothesis.status = "rejected"
            hypothesis.rejection_reason = "near-duplicate of an existing factor"
            registry.add(hypothesis)
            _append_eval_log_row(
                {
                    "date": pd.Timestamp.now().date(),
                    "factor": f"{hypothesis.id}: {hypothesis.description}",
                    "source": f"tickers={tickers}, {start}..{end}",
                    "oos_sharpe": float("nan"),
                    "max_dd": float("nan"),
                    "t_stat": float("nan"),
                    "bh_sig": "n/a",
                    "corr_max": "n/a",
                    "decision": "rejected (near-duplicate)",
                }
            )
            print(f"[reject] {hypothesis.id}: near-duplicate")
            continue
        existing_descriptions.append(hypothesis.description)
        surviving.append(hypothesis)

    if not surviving:
        print("No hypotheses survived the dedup gate.")
        return

    batch_results = []
    for hypothesis in surviving:
        portfolio_returns = []
        for ticker in tickers:
            close = prices.xs(ticker, level="ticker")["close"]
            signal = _signal_for_ticker(hypothesis, texts, ticker, close.index)
            result = run_backtest(signal, close.to_frame("close"))
            portfolio_returns.append(result.returns.rename(ticker))
        portfolio = pd.concat(portfolio_returns, axis=1).mean(axis=1).dropna()
        returns_array = portfolio.to_numpy()
        batch_results.append(
            {
                "hypothesis": hypothesis,
                "returns": portfolio,
                "sharpe": sharpe_ratio(returns_array),
                "max_dd": max_drawdown(returns_array),
                "t_stat": t_stat(returns_array),
                "p_value": p_value(returns_array),
            }
        )

    bh_significant = benjamini_hochberg([r["p_value"] for r in batch_results])

    for result, significant in zip(batch_results, bh_significant):
        hypothesis = result["hypothesis"]
        kept = significant and result["sharpe"] > 0
        hypothesis.status = "kept" if kept else "rejected"
        hypothesis.sharpe = result["sharpe"]
        hypothesis.max_drawdown = result["max_dd"]
        hypothesis.t_stat = result["t_stat"]
        if not kept:
            hypothesis.rejection_reason = (
                "failed Benjamini-Hochberg significance" if not significant else "non-positive Sharpe"
            )
        registry.add(hypothesis)
        _append_eval_log_row(
            {
                "date": pd.Timestamp.now().date(),
                "factor": f"{hypothesis.id}: {hypothesis.description}",
                "source": f"tickers={tickers}, {start}..{end}",
                "oos_sharpe": result["sharpe"],
                "max_dd": result["max_dd"],
                "t_stat": result["t_stat"],
                "bh_sig": significant,
                "corr_max": "n/a",
                "decision": "kept" if kept else f"rejected ({hypothesis.rejection_reason})",
            }
        )
        print(
            f"[{'keep' if kept else 'reject'}] {hypothesis.id}: sharpe={result['sharpe']:.2f} "
            f"t_stat={result['t_stat']:.2f} bh_sig={significant}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--n-factors", type=int, default=5)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    run(args.tickers, args.start, args.end, args.n_factors, args.data_dir)
