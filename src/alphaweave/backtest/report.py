"""Equity-curve and factor-correlation reporting."""

import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative growth-of-$1 curve from a periodic return series."""
    return (1.0 + returns).cumprod()


def save_equity_curve_plot(returns: pd.Series, out_path: str) -> None:
    """Render and save an equity curve plot.

    Deferred import of matplotlib keeps it out of the core dependency path
    for environments that only need the numeric engine.
    """
    import matplotlib.pyplot as plt

    curve = equity_curve(returns)
    fig, ax = plt.subplots(figsize=(10, 4))
    curve.plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Growth of $1")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def factor_correlation_matrix(factor_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise correlation between candidate factors' return series.

    Used as an input to the dedup/novelty gate before a new factor is kept.
    """
    df = pd.DataFrame(factor_returns)
    return df.corr()
