"""Transaction cost models."""

from typing import Union

import numpy as np
import pandas as pd

from ..config import BacktestConfig


def compute_trading_costs(
    trades: pd.Series,
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: Union[float, pd.Series],
    config: BacktestConfig,
) -> pd.Series:
    """Compute transaction costs for each bar.

    Args:
        trades: Position changes (diff of positions), e.g. +1, -1, +2, -2
        prices_a: Price series of asset A
        prices_b: Price series of asset B
        hedge_ratio: Hedge ratio (scalar or series)
        config: Backtest config with fee parameters

    Returns:
        Series of costs (positive = cost)
    """
    abs_trades = trades.abs()

    # 每单位仓位的名义金额
    notional_a = prices_a * config.position_size * config.capital / prices_a
    notional_b = prices_b * abs(hedge_ratio) * config.position_size * config.capital / prices_b

    # 简化：对notional用固定比例
    # 实际成本 = |trade_size| * (taker_fee + slippage) * notional_per_leg * 2legs
    fee_rate = config.taker_fee + config.slippage_bps / 10000.0
    costs = abs_trades * fee_rate * 2  # 两腿各收一次

    costs.name = "costs"
    return costs


def compute_simple_costs(
    position_changes: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """Simplified cost model: fixed percentage per trade on notional.

    Cost = |position_change| * (taker_fee + slippage) * 2 (两腿)
    Applied as fraction of position_size * capital.
    """
    fee_rate = config.taker_fee + config.slippage_bps / 10000.0
    costs = position_changes.abs() * fee_rate * 2
    costs.name = "costs"
    return costs
