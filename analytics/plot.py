"""Visualization for backtest results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from engine.backtest import BacktestResult


def plot_backtest(result: BacktestResult, title: str = "Pairs Trading Backtest") -> plt.Figure:
    """Plot comprehensive backtest results.

    Creates a 4-panel figure:
    1. Equity curve
    2. Spread and z-score
    3. Positions
    4. Drawdown
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # 只绘制有数据的部分
    valid = result.spread.dropna().index
    if len(valid) == 0:
        return fig

    start = valid[0]
    eq = result.equity_curve.loc[start:]
    spread = result.spread.loc[start:]
    zscore = result.zscore.loc[start:]
    positions = result.positions.loc[start:]

    # 1. Equity curve
    ax = axes[0]
    ax.plot(eq.index, eq.values, "b-", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve")
    ax.grid(True, alpha=0.3)

    # 2. Spread + Z-score
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(spread.index, spread.values, "b-", linewidth=0.5, alpha=0.7, label="Spread")
    ax2.plot(zscore.index, zscore.values, "r-", linewidth=0.5, alpha=0.7, label="Z-score")
    ax2.axhline(2.0, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.axhline(-2.0, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.axhline(0, color="gray", linestyle="-", linewidth=0.3)
    ax.set_ylabel("Spread", color="blue")
    ax2.set_ylabel("Z-score", color="red")
    ax.set_title("Spread & Z-score")
    ax.grid(True, alpha=0.3)

    # 3. Positions
    ax = axes[2]
    ax.fill_between(positions.index, positions.values, 0, alpha=0.5, step="post")
    ax.set_ylabel("Position")
    ax.set_title("Positions")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)

    # 4. Drawdown
    ax = axes[3]
    cummax = eq.cummax()
    drawdown = (eq / cummax - 1.0) * 100
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pair_prices(
    prices_a: pd.Series,
    prices_b: pd.Series,
    title: str = "Pair Prices (Normalized)",
) -> plt.Figure:
    """Plot normalized prices of the pair."""
    fig, ax = plt.subplots(figsize=(12, 4))

    norm_a = prices_a / prices_a.iloc[0]
    norm_b = prices_b / prices_b.iloc[0]

    ax.plot(norm_a.index, norm_a.values, label=prices_a.name or "Asset A", linewidth=0.8)
    ax.plot(norm_b.index, norm_b.values, label=prices_b.name or "Asset B", linewidth=0.8)
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Normalized Price")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_monthly_returns(pnl: pd.Series, title: str = "Monthly Returns") -> plt.Figure:
    """Plot monthly returns heatmap."""
    # 聚合为月度收益
    monthly = pnl.resample("ME").sum()
    monthly_df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )

    pivot = monthly_df.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)

    # 标注数值
    for i in range(len(pivot)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Return (%)")
    ax.set_title(title)
    plt.tight_layout()
    return fig
