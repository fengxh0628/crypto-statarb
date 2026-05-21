"""Performance metrics."""

import numpy as np
import pandas as pd


def compute_metrics(
    pnl: pd.Series,
    equity_curve: pd.Series,
    positions: pd.Series,
    trades: pd.Series,
    bars_per_year: int = 105120,  # 5分钟级：365*24*60/5
) -> dict:
    """Compute comprehensive performance metrics.

    Args:
        pnl: Per-bar PnL series
        equity_curve: Cumulative equity curve (starting at 1.0)
        positions: Position series
        trades: Trade series (position changes)
        bars_per_year: Number of bars in a year (for annualization)

    Returns:
        Dict of metric name -> value
    """
    pnl_clean = pnl.dropna()
    n_bars = len(pnl_clean)

    # 基本收益
    total_return = equity_curve.iloc[-1] - 1.0
    ann_factor = bars_per_year / n_bars if n_bars > 0 else 1
    # Handle negative total return (can't raise negative to fractional power)
    if total_return > -1:
        ann_return = (1 + total_return) ** ann_factor - 1
    else:
        ann_return = -1.0  # Cap at -100%

    # 波动率
    ann_vol = pnl_clean.std() * np.sqrt(bars_per_year)

    # Sharpe Ratio (假设无风险利率=0)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio
    downside = pnl_clean[pnl_clean < 0]
    downside_vol = downside.std() * np.sqrt(bars_per_year) if len(downside) > 0 else 0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    # 最大回撤
    cummax = equity_curve.cummax()
    drawdown = equity_curve / cummax - 1.0
    max_drawdown = drawdown.min()

    # 回撤持续时间（bar数）
    dd_duration = _max_drawdown_duration(equity_curve)

    # 胜率
    trade_bars = pnl_clean[trades != 0]
    if len(trade_bars) > 0:
        win_rate = (trade_bars > 0).sum() / len(trade_bars)
    else:
        # 用所有有仓位的bar
        active_pnl = pnl_clean[positions.shift(1) != 0]
        win_rate = (active_pnl > 0).sum() / len(active_pnl) if len(active_pnl) > 0 else 0

    # 盈亏比
    gains = pnl_clean[pnl_clean > 0]
    losses = pnl_clean[pnl_clean < 0]
    profit_factor = gains.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf

    # 交易统计
    n_trades = (trades != 0).sum()
    n_round_trips = n_trades // 2

    # 持仓占比
    time_in_market = (positions != 0).sum() / n_bars if n_bars > 0 else 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_bars": dd_duration,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": int(n_trades),
        "n_round_trips": int(n_round_trips),
        "time_in_market": time_in_market,
        "n_bars": n_bars,
    }


def _max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Compute max drawdown duration in number of bars."""
    cummax = equity_curve.cummax()
    in_drawdown = equity_curve < cummax

    max_duration = 0
    current_duration = 0
    for dd in in_drawdown:
        if dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration


def print_metrics(metrics: dict, capital: float = 0) -> None:
    """Pretty-print metrics.

    Args:
        metrics: Dict from compute_metrics()
        capital: If > 0, also show dollar PnL for this pair's allocation
    """
    print("=" * 50)
    print("         BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Total Return:         {metrics['total_return']:.4%}")
    if capital > 0:
        print(f"  Total PnL:            {metrics['total_return'] * capital:.2f} USDT")
    print(f"  Annualized Return:    {metrics['annualized_return']:.4%}")
    print(f"  Annualized Volatility:{metrics['annualized_volatility']:.4%}")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']:.4%}")
    if capital > 0:
        print(f"  Max DD (USDT):        {metrics['max_drawdown'] * capital:.2f} USDT")
    print(f"  Max DD Duration:      {metrics['max_drawdown_duration_bars']} bars")
    print(f"  Win Rate:             {metrics['win_rate']:.2%}")
    print(f"  Profit Factor:        {metrics['profit_factor']:.2f}")
    print(f"  Number of Trades:     {metrics['n_trades']}")
    print(f"  Round Trips:          {metrics['n_round_trips']}")
    print(f"  Time in Market:       {metrics['time_in_market']:.2%}")
    if "total_trades" in metrics:
        print(f"  Total Trades:         {metrics['total_trades']}")
    if "unique_pairs_traded" in metrics:
        print(f"  Unique Pairs Traded:  {metrics['unique_pairs_traded']}")
    print("=" * 50)


def print_monthly_returns(monthly_df: pd.DataFrame) -> None:
    """Print monthly returns table."""
    print("\n" + "=" * 70)
    print("  MONTHLY RETURNS")
    print("=" * 70)

    # 表头
    cols = monthly_df.columns.tolist()
    header = f"{'Month':<10}"
    for col in cols:
        header += f"{col:>12}"
    print(header)
    print("-" * (10 + 12 * len(cols)))

    # 数据行
    for month, row in monthly_df.iterrows():
        line = f"{month:<10}"
        for col in cols:
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                line += f"{'---':>12}"
            else:
                line += f"{val:>+11.2%} "
        print(line)

    print("-" * (10 + 12 * len(cols)))
    print("=" * 70)
