"""End-to-end pipeline run: ingest -> compute the hand-written momentum
factor -> backtest -> walk-forward validate -> report.

This is the "prove the pipeline before automating hypothesis generation"
milestone (README weeks 3-4): one factor, real data, running start to finish.
"""

import argparse

import pandas as pd

from alphaweave.backtest.engine import run_backtest, walk_forward_splits
from alphaweave.backtest.report import save_equity_curve_plot
from alphaweave.data.ingest import fetch_ohlcv
from alphaweave.factors.momentum import compute_momentum_signal
from alphaweave.factors.registry import FactorHypothesis, FactorRegistry


def run(tickers: list[str], start: str, end: str, data_dir: str) -> None:
    prices = fetch_ohlcv(tickers, start=start, end=end)
    registry = FactorRegistry(path=f"{data_dir}/factor_registry.json")

    portfolio_returns = []
    for ticker in tickers:
        close = prices.xs(ticker, level="ticker")["close"]
        signal = compute_momentum_signal(close)
        ticker_prices = close.to_frame("close")
        result = run_backtest(signal, ticker_prices)
        print(
            f"{ticker}: sharpe={result.sharpe:.2f} "
            f"max_dd={result.max_dd:.2%} t_stat={result.t_stat:.2f}"
        )
        portfolio_returns.append(result.returns.rename(ticker))

    portfolio = pd.concat(portfolio_returns, axis=1).mean(axis=1).dropna()
    splits = walk_forward_splits(portfolio.index)
    print(f"\nWalk-forward windows: {len(splits)}")
    for train_dates, test_dates in splits:
        oos_returns = portfolio.reindex(test_dates).dropna()
        if oos_returns.empty:
            continue
        from alphaweave.backtest.metrics import sharpe_ratio

        print(
            f"  train {train_dates.min().date()}-{train_dates.max().date()} "
            f"-> test {test_dates.min().date()}-{test_dates.max().date()}: "
            f"oos_sharpe={sharpe_ratio(oos_returns.to_numpy()):.2f}"
        )

    from alphaweave.backtest.metrics import max_drawdown, sharpe_ratio, t_stat

    portfolio_sharpe = sharpe_ratio(portfolio.to_numpy())
    hypothesis = FactorHypothesis(
        id="momentum-12-1",
        description="12-1 month price momentum, equal-weighted across tickers",
        source_context=f"tickers={tickers}, {start}..{end}",
        status="kept" if portfolio_sharpe > 0 else "rejected",
        sharpe=portfolio_sharpe,
        max_drawdown=max_drawdown(portfolio.to_numpy()),
        t_stat=t_stat(portfolio.to_numpy()),
    )
    registry.add(hypothesis)
    save_equity_curve_plot(portfolio, f"{data_dir}/momentum_equity_curve.png")
    print(f"\nPortfolio sharpe={portfolio_sharpe:.2f}, logged to registry.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "SPY"])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    run(args.tickers, args.start, args.end, args.data_dir)
