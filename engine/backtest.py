"""Vectorized backtest engine."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from config import BacktestConfig
from engine.costs import compute_simple_costs
from pairs.spread import compute_hedge_ratio, compute_spread, estimate_ou_params, rolling_spread
from strategy.rules import generate_positions
from strategy.signals import compute_zscore


@dataclass
class BacktestResult:
    """Container for backtest results."""

    pnl: pd.Series  # 逐bar收益率
    cumulative_pnl: pd.Series  # 累计收益率
    equity_curve: pd.Series  # 净值曲线
    positions: pd.Series  # 仓位序列
    trades: pd.Series  # 交易（仓位变动）
    costs: pd.Series  # 交易成本
    spread: pd.Series  # 价差
    zscore: pd.Series  # z-score
    hedge_ratio: Union[float, pd.Series]  # hedge ratio
    prices_a: pd.Series
    prices_b: pd.Series


def _compute_funding_costs(
    positions: pd.Series,
    index: pd.DatetimeIndex,
    funding_rate_a: Optional[pd.Series],
    funding_rate_b: Optional[pd.Series],
    config: 'BacktestConfig',
) -> pd.Series:
    """Compute funding rate costs per bar.

    Logic:
    - position > 0: long A (pay fr_a if fr_a > 0) + short B (receive fr_b if fr_b > 0)
    - position < 0: short A (receive fr_a if fr_a > 0) + long B (pay fr_b if fr_b > 0)

    With real funding data: costs applied at each settlement timestamp present in
    the funding_rate series index.

    With fixed rate (fallback): costs applied every funding_interval_bars.
    """
    funding_costs = pd.Series(0.0, index=index)

    if funding_rate_a is not None or funding_rate_b is not None:
        # 真实 funding rate 模式
        # 将funding rate reindex到价格的时间轴，只在结算时刻有值
        if funding_rate_a is not None:
            # 找到价格index中与funding结算时刻匹配的bar
            fr_a = funding_rate_a.reindex(index, method=None).fillna(0)
        else:
            fr_a = pd.Series(0.0, index=index)

        if funding_rate_b is not None:
            fr_b = funding_rate_b.reindex(index, method=None).fillna(0)
        else:
            fr_b = pd.Series(0.0, index=index)

        # position > 0: long A, short B
        #   A腿: long 付 fr_a (正rate多头付费)
        #   B腿: short 收 fr_b (正rate空头收费) → 成本为 -fr_b
        # position < 0: short A, long B
        #   A腿: short 收 fr_a → 成本为 -fr_a
        #   B腿: long 付 fr_b
        # 统一公式: cost = position * (fr_a - fr_b)
        # 注意：这里 cost>0 表示支出
        funding_costs = positions * (fr_a - fr_b)
        # 取绝对值不对，应该保留方向：正=付费，负=收费
        # funding_costs 可以为负（净收到funding）

    else:
        # 固定 funding rate fallback
        interval = config.funding_interval_bars
        if interval > 0 and config.funding_rate > 0:
            funding_mask = np.zeros(len(positions), dtype=bool)
            funding_mask[interval::interval] = True
            # 固定模式：假设两腿都付出固定费率
            funding_costs[funding_mask] = (
                positions.abs().iloc[funding_mask.nonzero()[0]].values
                * config.funding_rate * 2
            )

    return funding_costs


def run_backtest(
    prices_a: pd.Series,
    prices_b: pd.Series,
    config: Optional[BacktestConfig] = None,
    rolling_beta: bool = True,
    adaptive_window: bool = True,
    funding_rate_a: Optional[pd.Series] = None,
    funding_rate_b: Optional[pd.Series] = None,
) -> BacktestResult:
    """Run a complete pairs trading backtest.

    Args:
        prices_a: Close prices of asset A
        prices_b: Close prices of asset B
        config: Backtest configuration
        rolling_beta: Whether to use rolling hedge ratio
        adaptive_window: If True, estimate half-life from spread and use it
                         as z-score window (overrides config.zscore_window)
        funding_rate_a: Historical funding rate series for asset A.
                        Index=settlement timestamps, values=rate per period.
                        If None, uses config.funding_rate as fixed rate.
        funding_rate_b: Historical funding rate series for asset B.
                        Same format as funding_rate_a.

    Returns:
        BacktestResult with all backtest outputs
    """
    if config is None:
        config = BacktestConfig()

    # 1. 计算价差
    if rolling_beta:
        spread, hedge_ratios = rolling_spread(
            prices_a, prices_b, window=config.hedge_ratio_window
        )
        hedge_ratio = hedge_ratios
    else:
        # 用前半段估计 hedge ratio
        train_len = min(config.hedge_ratio_window, len(prices_a) // 4)
        hr = compute_hedge_ratio(prices_a.iloc[:train_len], prices_b.iloc[:train_len])
        spread = compute_spread(prices_a, prices_b, hr)
        hedge_ratio = hr

    # 2. 自适应窗口：用半衰期作为z-score滚动窗口
    zscore_window = config.zscore_window
    if adaptive_window:
        # 用前段数据估计半衰期
        warmup = min(config.hedge_ratio_window * 2, len(spread) // 3)
        spread_warmup = spread.iloc[:warmup].dropna()
        if len(spread_warmup) > 30:
            ou = estimate_ou_params(spread_warmup)
            if ou["stationary"] and 10 < ou["half_life"] < 500:
                zscore_window = int(round(ou["half_life"]))

    # 3. 计算 z-score
    zscore = compute_zscore(spread, window=zscore_window)

    # 3. 生成仓位
    positions = generate_positions(zscore, config)

    # 4. 计算交易（仓位变动）
    trades = positions.diff().fillna(0)

    # 5. 计算PnL
    # 使用对数收益率方式：
    # 做多spread = 做多A + 做空B（按hedge ratio加权）
    # PnL(per bar) = position * (ret_A - β * ret_B)
    ret_a = np.log(prices_a / prices_a.shift(1))
    ret_b = np.log(prices_b / prices_b.shift(1))

    # 当前bar的β（如果是rolling）
    if isinstance(hedge_ratio, pd.Series):
        beta = hedge_ratio.shift(1).bfill()
    else:
        beta = hedge_ratio

    # 组合收益率 = ret_A - β * ret_B (做多spread方向)
    pair_return = ret_a - beta * ret_b
    raw_pnl = positions.shift(1) * pair_return  # 用前一bar的仓位

    # 6. 计算成本（以收益率计）
    # 每次交易两腿各付一次手续费+滑点
    fee_rate = config.taker_fee + config.slippage_bps / 10000.0
    trade_costs = trades.abs() * fee_rate * 2

    # 7. Funding rate 成本
    # 配对交易永续合约：做多A + 做空B
    # funding cost 在结算时刻扣除，方向：
    #   做多方付 funding_rate > 0 时付费，< 0 时收费
    #   做空方反过来
    funding_costs = _compute_funding_costs(
        positions, prices_a.index,
        funding_rate_a, funding_rate_b, config,
    )

    costs = trade_costs + funding_costs

    # 8. 净PnL（收益率 × 杠杆 = 保证金收益率）
    pnl = (raw_pnl - costs) * config.leverage
    pnl = pnl.fillna(0)

    # 9. 累计PnL和净值（基于保证金）
    cumulative_pnl = pnl.cumsum()
    equity_curve = 1.0 + cumulative_pnl  # 保证金净值从1开始

    return BacktestResult(
        pnl=pnl,
        cumulative_pnl=cumulative_pnl,
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
        costs=costs,
        spread=spread,
        zscore=zscore,
        hedge_ratio=hedge_ratio,
        prices_a=prices_a,
        prices_b=prices_b,
    )


def run_multi_pair_backtest(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    config: Optional[BacktestConfig] = None,
) -> dict[tuple[str, str], BacktestResult]:
    """Run backtest on multiple pairs.

    Args:
        prices: DataFrame with columns = symbol names
        pairs: List of (symbol_a, symbol_b) tuples
        config: Backtest configuration

    Returns:
        Dict mapping pair -> BacktestResult
    """
    results = {}
    for sym_a, sym_b in pairs:
        result = run_backtest(prices[sym_a], prices[sym_b], config)
        results[(sym_a, sym_b)] = result
    return results
