"""Enhanced momentum strategy with multi-factor signal.

Improvements over basic momentum:
1. Multi-timeframe momentum (short + long)
2. Volume-weighted momentum
3. Volatility-adjusted momentum
4. Trend filter (only trade when market is trending)
5. Momentum acceleration (second derivative)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnhancedMomentumConfig:
    # 动量参数
    momentum_short: int = 72  # 短期动量（3天 for 1h）
    momentum_long: int = 336  # 长期动量（14天 for 1h）
    momentum_weight_short: float = 0.3  # 短期动量权重
    momentum_weight_long: float = 0.7  # 长期动量权重
    
    skip_recent: int = 12  # 跳过最近 N 个 bar
    
    # 波动率调整
    vol_adjustment: bool = True  # 是否使用波动率调整动量
    vol_window: int = 168  # 波动率计算窗口（7天）
    
    # 成交量确认
    volume_confirm: bool = True  # 是否使用成交量确认
    volume_window: int = 168  # 成交量计算窗口
    
    # 趋势过滤
    trend_filter: bool = True  # 是否使用趋势过滤
    trend_window: int = 336  # 趋势计算窗口
    trend_threshold: float = 0.0  # 趋势阈值
    
    # 动量加速度
    momentum_accel: bool = True  # 是否使用动量加速度
    accel_window: int = 72  # 加速度计算窗口
    
    # 交易参数
    top_k: int = 5
    bottom_k: int = 5
    rebalance_bars: int = 24
    
    # 仓位管理
    position_size: float = 0.10
    leverage: float = 1.0
    
    # 成本
    taker_fee: float = 0.0005
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 24
    
    # 过滤
    min_vol_threshold: float = 0.001
    max_vol_threshold: float = 0.1


def compute_multi_timeframe_momentum(
    prices: pd.DataFrame,
    config: EnhancedMomentumConfig,
) -> pd.Series:
    """计算多时间框架动量得分。"""
    n = len(prices)
    if n < config.momentum_long + config.skip_recent:
        return pd.Series(np.nan, index=prices.columns)
    
    log_prices = np.log(prices.values)
    
    # 短期动量
    short_start = n - config.momentum_short - config.skip_recent
    short_end = n - config.skip_recent
    mom_short = log_prices[short_end] - log_prices[short_start]
    
    # 长期动量
    long_start = n - config.momentum_long - config.skip_recent
    long_end = n - config.skip_recent
    mom_long = log_prices[long_end] - log_prices[long_start]
    
    # 加权组合
    momentum = (config.momentum_weight_short * mom_short + 
                config.momentum_weight_long * mom_long)
    
    return pd.Series(momentum, index=prices.columns)


def compute_vol_adjusted_momentum(
    prices: pd.DataFrame,
    config: EnhancedMomentumConfig,
) -> pd.Series:
    """计算波动率调整的动量得分。"""
    momentum = compute_multi_timeframe_momentum(prices, config)
    
    if not config.vol_adjustment:
        return momentum
    
    # 计算波动率
    returns = np.log(prices).diff().iloc[-config.vol_window:]
    vol = returns.std()
    
    # 波动率调整：动量 / 波动率（风险调整后收益）
    vol_adjusted = momentum / vol.replace(0, np.nan)
    
    return vol_adjusted


def compute_volume_weighted_momentum(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    config: EnhancedMomentumConfig,
) -> pd.Series:
    """计算成交量加权的动量得分。"""
    momentum = compute_vol_adjusted_momentum(prices, config)
    
    if not config.volume_confirm or volumes is None:
        return momentum
    
    # 计算相对成交量
    vol_ma = volumes.rolling(config.volume_window).mean().iloc[-1]
    vol_recent = volumes.iloc[-config.skip_recent:].mean()
    vol_ratio = vol_recent / vol_ma.replace(0, np.nan)
    
    # 成交量确认：成交量放大时，动量信号更强
    volume_weight = np.clip(vol_ratio, 0.5, 1.5)
    
    return momentum * volume_weight


def compute_trend_filter(
    prices: pd.DataFrame,
    config: EnhancedMomentumConfig,
) -> pd.Series:
    """计算趋势过滤信号。"""
    if not config.trend_filter:
        return pd.Series(True, index=prices.columns)
    
    n = len(prices)
    if n < config.trend_window:
        return pd.Series(False, index=prices.columns)
    
    # 计算趋势：价格在趋势窗口内的位置
    log_prices = np.log(prices.values)
    trend_start = n - config.trend_window
    trend_end = n
    
    # 线性回归斜率
    x = np.arange(config.trend_window)
    slope = np.zeros(len(prices.columns))
    
    for i in range(len(prices.columns)):
        y = log_prices[trend_start:trend_end, i]
        # 简单线性回归
        x_mean = x.mean()
        y_mean = y.mean()
        slope[i] = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)
    
    # 趋势过滤：斜率大于阈值
    trend = pd.Series(slope > config.trend_threshold, index=prices.columns)
    
    return trend


def compute_momentum_acceleration(
    prices: pd.DataFrame,
    config: EnhancedMomentumConfig,
) -> pd.Series:
    """计算动量加速度（二阶导数）。"""
    if not config.momentum_accel:
        return pd.Series(0.0, index=prices.columns)
    
    n = len(prices)
    if n < config.momentum_short * 2 + config.skip_recent:
        return pd.Series(np.nan, index=prices.columns)
    
    log_prices = np.log(prices.values)
    
    # 近期动量
    recent_start = n - config.momentum_short - config.skip_recent
    recent_end = n - config.skip_recent
    mom_recent = log_prices[recent_end] - log_prices[recent_start]
    
    # 前期动量
    prev_start = n - config.momentum_short * 2 - config.skip_recent
    prev_end = n - config.momentum_short - config.skip_recent
    mom_prev = log_prices[prev_end] - log_prices[prev_start]
    
    # 加速度
    accel = mom_recent - mom_prev
    
    return pd.Series(accel, index=prices.columns)


def generate_enhanced_momentum_signals(
    prices: pd.DataFrame,
    config: EnhancedMomentumConfig,
    volumes: Optional[pd.DataFrame] = None,
) -> dict:
    """生成增强动量信号。"""
    # 基础动量
    momentum = compute_volume_weighted_momentum(prices, volumes, config)
    
    # 趋势过滤
    trend = compute_trend_filter(prices, config)
    
    # 动量加速度
    accel = compute_momentum_acceleration(prices, config)
    
    # 波动率过滤
    returns = np.log(prices).diff().iloc[-config.vol_window:]
    vol = returns.std()
    valid = (vol > config.min_vol_threshold) & (vol < config.max_vol_threshold)
    valid = valid & ~momentum.isna() & trend
    
    if valid.sum() < config.top_k + config.bottom_k:
        return {}
    
    # 组合信号：动量 + 加速度
    signal = momentum.copy()
    if config.momentum_accel:
        accel_valid = accel[valid].dropna()
        if len(accel_valid) > 0:
            # 加速度归一化
            accel_norm = (accel_valid - accel_valid.mean()) / (accel_valid.std() + 1e-10)
            signal[valid] = momentum[valid] + 0.2 * accel_norm
    
    # 排序
    valid_signal = signal[valid].dropna()
    sorted_symbols = valid_signal.sort_values(ascending=False)
    
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


def run_enhanced_momentum_backtest(
    prices: pd.DataFrame,
    config: Optional[EnhancedMomentumConfig] = None,
    volumes: Optional[pd.DataFrame] = None,
) -> dict:
    """运行增强动量策略回测。"""
    if config is None:
        config = EnhancedMomentumConfig()
    
    n_bars = len(prices)
    index = prices.index
    
    # 状态
    positions = {}
    portfolio_pnl = np.zeros(n_bars)
    equity_curve = np.ones(n_bars)
    trades = []
    
    total_fees = 0.0
    total_funding = 0.0
    total_gross_pnl = 0.0
    
    fee_rate = (config.taker_fee + config.slippage_bps / 10000.0) * 2 * config.leverage
    
    print(f"  Running enhanced momentum backtest: {n_bars} bars, {len(prices.columns)} symbols")
    last_progress = 0
    
    warmup = config.momentum_long + config.skip_recent + config.vol_window
    
    for t in range(warmup, n_bars):
        # 进度
        progress = (t - warmup) * 100 // (n_bars - warmup)
        if progress >= last_progress + 10:
            print(f"    {progress}% ({t}/{n_bars})", end="\r")
            last_progress = progress
        
        # 调仓
        if (t - warmup) % config.rebalance_bars == 0:
            window_prices = prices.iloc[:t]
            window_volumes = volumes.iloc[:t] if volumes is not None else None
            new_weights = generate_enhanced_momentum_signals(window_prices, config, window_volumes)
            
            # 计算调仓成本
            old_symbols = set(positions.keys())
            new_symbols = set(new_weights.keys())
            n_turnover = len(old_symbols.symmetric_difference(new_symbols))
            bar_fees = n_turnover * fee_rate * config.position_size
            total_fees += bar_fees
            
            positions = new_weights
        
        # 计算 PnL
        bar_pnl = 0.0
        for sym, weight in positions.items():
            if sym in prices.columns:
                ret = np.log(prices.values[t, prices.columns.get_loc(sym)] / 
                           prices.values[t-1, prices.columns.get_loc(sym)])
                bar_pnl += ret * weight
        
        # Funding rate
        bar_funding = 0.0
        if config.funding_interval_bars > 0 and t % config.funding_interval_bars == 0:
            for sym, weight in positions.items():
                if weight < 0:
                    bar_funding += config.funding_rate * abs(weight)
                else:
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
    
    print()
    
    # 构建结果
    equity_series = pd.Series(equity_curve, index=index)
    pnl_series = pd.Series(portfolio_pnl, index=index)
    
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
