"""Point-in-time OHLCV ingestion.

Daily close prices from yfinance are only known as of market close on their
date, so a frame indexed this way is already point-in-time correct: nothing
here reveals information before the date it's attached to.
"""

import pandas as pd


def fetch_ohlcv(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV for `tickers` between `start` and `end`.

    Returns a long frame indexed by (date, ticker) with a 'close' column,
    matching the index shape `run_backtest` expects for `prices`.
    """
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, group_by="ticker",
        progress=False,
    )
    frames = []
    for ticker in tickers:
        sub = raw[ticker] if len(tickers) > 1 else raw
        close = sub["Close"].rename("close").dropna()
        frame = close.to_frame()
        frame["ticker"] = ticker
        frames.append(frame)
    combined = pd.concat(frames)
    combined = combined.set_index("ticker", append=True)
    combined.index.set_names(["date", "ticker"], inplace=True)
    return combined.sort_index()
