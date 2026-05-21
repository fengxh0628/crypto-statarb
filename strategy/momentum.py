"""Momentum strategy for crypto futures.

Logic:
1. Calculate momentum score for each symbol (past N bars return)
2. Go long top K symbols, go short bottom K symbols
3. Rebalance every M bars
4. Equal weight or volatility weight
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class MomentumConfig:
    # 动量参数
    momentum_window: int = 72  # 动量计算窗口（3天 for 1h）
    skip_recent: int = 12  # 跳过最近 N 个 bar（避免短期反转）
    
    # 交易参数
    top_k: int = 5  # 做多前 K 个
    bottom_k: int = 5  # 做空后 K 个
    rebalance_bars: int = 24  # 调仓间隔（1天 for 1h）
    
    # 仓位管理
    position_size: float = 0.10  # 每个符号仓位占比
    leverage: float = 1.0
    
    # 成本
    taker_fee: float = 0.0005  # Taker fee 0.05%
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 24
    
    # 过滤
    vol_filter_window: int = 72  # 波动率过滤窗口
    min_vol_threshold: float = 0.001  # 最小波动率（排除低波动币种）
    max_vol_threshold: float = 0.1  # 最大波动率（排除极端波动币种）


def compute_momentum(
    prices: pd.DataFrame,
    momentum_window: int = 72,
    skip_recent: int = 12,
) -> pd.Series:
    """计算每个币种的动量得分。
    
    momentum = log(price[t-skip] / price[t-skip-window])
    """
    n = len(prices)
    if n < momentum_window + skip_recent:
        return pd.Series(np.nan, index=prices.columns)
    
    # 使用对数收益率计算动量
    start_idx = n - momentum_window - skip_recent
    end_idx = n - skip_recent
    
    log_prices = np.log(prices.values)
    momentum = log_prices[end_idx] - log_prices[start_idx]
    
    return pd.Series(momentum, index=prices.columns)


def compute_volatility(
    prices: pd.DataFrame,
    window: int = 72,
) -> pd.Series:
    """计算每个币种的波动率（用于过滤和加权）。"""
    returns = np.log(prices).diff().iloc[-window:]
    vol = returns.std()
    return vol


def generate_momentum_signals(
    prices: pd.DataFrame,
    config: MomentumConfig,
) -> dict:
    """生成动量信号。
    
    Returns:
        {symbol: weight} 正数=做多，负数=做空，0=不持仓
    """
    momentum = compute_momentum(prices, config.momentum_window, config.skip_recent)
    vol = compute_volatility(prices, config.vol_filter_window)
    
    # 过滤低波动和极端波动币种
    valid = (vol > config.min_vol_threshold) & (vol < config.max_vol_threshold)
    valid = valid & ~momentum.isna()
    
    if valid.sum() < config.top_k + config.bottom_k:
        return {}
    
    # 排序
    valid_momentum = momentum[valid].dropna()
    sorted_symbols = valid_momentum.sort_values(ascending=False)
    
    # 选择 top K 和 bottom K
    long_symbols = sorted_symbols.head(config.top_k).index.tolist()
    short_symbols = sorted_symbols.tail(config.bottom_k).index.tolist()
    
    # 构建权重
    weights = {}
    for sym in long_symbols:
        weights[sym] = config.position_size * config.leverage
    for sym in short_symbols:
        weights[sym] = -config.position_size * config.leverage
    
    return weights


def run_momentum_backtest(
    prices: pd.DataFrame,
    config: Optional[MomentumConfig] = None,
    volumes: Optional[pd.DataFrame] = None,
) -> dict:
    """运行动量策略回测。
    
    Returns:
        dict with equity_curve, trades, metrics
    """
    if config is None:
        config = MomentumConfig()
    
    n_bars = len(prices)
    index = prices.index
    
    # 状态
    positions = {}  # {symbol: weight}
    portfolio_pnl = np.zeros(n_bars)
    equity_curve = np.ones(n_bars)
    trades = []
    
    total_fees = 0.0
    total_funding = 0.0
    total_gross_pnl = 0.0
    
    fee_rate = (config.taker_fee + config.slippage_bps / 10000.0) * 2 * config.leverage
    
    print(f"  Running momentum backtest: {n_bars} bars, {len(prices.columns)} symbols")
    last_progress = 0
    
    for t in range(config.momentum_window + config.skip_recent, n_bars):
        # 进度
        progress = (t - config.momentum_window - config.skip_recent) * 100 // (n_bars - config.momentum_window - config.skip_recent)
        if progress >= last_progress + 10:
            print(f"    {progress}% ({t}/{n_bars})", end="\r")
            last_progress = progress
        
        # 调仓
        if (t - config.momentum_window - config.skip_recent) % config.rebalance_bars == 0:
            # 计算新信号
            window_prices = prices.iloc[:t]
            new_weights = generate_momentum_signals(window_prices, config)
            
            # 计算调仓成本
            old_symbols = set(positions.keys())
            new_symbols = set(new_weights.keys())
            n_turnover = len(old_symbols.symmetric_difference(new_symbols))
            bar_fees = n_turnover * fee_rate * config.position_size
            total_fees += bar_fees
            
            # 更新持仓
            positions = new_weights
        
        # 计算 PnL
        bar_pnl = 0.0
        for sym, weight in positions.items():
            if sym in prices.columns:
                ret = np.log(prices.values[t, prices.columns.get_loc(sym)] / 
                           prices.values[t-1, prices.columns.get_loc(sym)])
                bar_pnl += ret * weight
        
        # Funding rate（做空支付，做多收取）
        bar_funding = 0.0
        if config.funding_interval_bars > 0 and t % config.funding_interval_bars == 0:
            for sym, weight in positions.items():
                if weight < 0:  # 空头支付 funding
                    bar_funding += config.funding_rate * abs(weight)
                else:  # 多头收取 funding
                    bar_funding -= config.funding_rate * weight
            total_funding += bar_funding
        
        bar_pnl -= bar_funding
        total_gross_pnl += bar_pnl
        
        portfolio_pnl[t] = bar_pnl
        equity_curve[t] = equity_curve[t-1] + bar_pnl
        
        if equity_curve[t] <= 0:
            print(f"\n    Liquidation at bar {t} ({index[t]})")
            equity_curve[t:] = 0
            break
    
    print()  # newline
    
    # 构建结果
    equity_series = pd.Series(equity_curve, index=index)
    pnl_series = pd.Series(portfolio_pnl, index=index)
    
    # 计算指标
    total_return = equity_series.iloc[-1] - 1.0
    n_bars_active = (pnl_series != 0).sum()
    
    metrics = {
        "total_return": total_return,
        "gross_pnl": total_gross_pnl,
        "trading_fees": total_fees,
        "funding_cost": total_funding,
        "net_pnl": total_gross_pnl - total_fees - total_funding,
        "n_bars": n_bars,
        "n_trades": len(trades),
    }
    
    # 打印成本分解
    print(f"\n  === Cost Breakdown ===")
    print(f"  Gross PnL:      {total_gross_pnl:+.4f} ({total_gross_pnl*100:+.2f}%)")
    print(f"  Trading Fees:   -{total_fees:.4f} (-{total_fees*100:.2f}%)")
    print(f"  Funding Rate:   -{total_funding:.4f} (-{total_funding*100:.2f}%)")
    print(f"  Net PnL:        {total_gross_pnl - total_fees - total_funding:+.4f} ({(total_gross_pnl - total_fees - total_funding)*100:+.2f}%)")
    print(f"  ======================\n")
    
    return {
        "equity_curve": equity_series,
        "pnl_series": pnl_series,
        "metrics": metrics,
        "positions": positions,
    }
