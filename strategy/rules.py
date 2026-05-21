"""Position rules for pairs trading."""

import numpy as np
import pandas as pd

from config import BacktestConfig


def generate_positions(
    zscore: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """Generate target positions from z-score using threshold rules.

    Position logic (vectorized with state tracking):
        - Enter long spread when zscore < -entry_threshold
        - Enter short spread when zscore > +entry_threshold
        - Exit when |zscore| < exit_threshold
        - Stop loss when |zscore| > stop_loss_threshold
        - Max holding time limit

    Returns:
        Series of positions: +1 (long spread), -1 (short spread), 0 (flat)
    """
    n = len(zscore)
    positions = np.zeros(n)
    hold_count = 0

    for i in range(1, n):
        z = zscore.iloc[i]
        prev_pos = positions[i - 1]

        if np.isnan(z):
            positions[i] = prev_pos
            if prev_pos != 0:
                hold_count += 1
            continue

        # 止损
        if abs(z) > config.stop_loss_threshold and prev_pos != 0:
            positions[i] = 0
            hold_count = 0
            continue

        # 超时平仓
        if prev_pos != 0 and hold_count >= config.max_hold_bars:
            positions[i] = 0
            hold_count = 0
            continue

        # 平仓信号
        if prev_pos != 0 and abs(z) < config.exit_threshold:
            positions[i] = 0
            hold_count = 0
            continue

        # 开仓信号
        if prev_pos == 0:
            if z < -config.entry_threshold:
                positions[i] = 1  # spread过低，做多spread (买A卖B)
                hold_count = 1
            elif z > config.entry_threshold:
                positions[i] = -1  # spread过高，做空spread (卖A买B)
                hold_count = 1
            else:
                positions[i] = 0
        else:
            # 持仓中，保持仓位
            positions[i] = prev_pos
            hold_count += 1

    result = pd.Series(positions, index=zscore.index, name="position")
    return result
