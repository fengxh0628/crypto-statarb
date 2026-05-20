"""Signal generation for pairs trading."""

from typing import Optional

import numpy as np
import pandas as pd


def compute_zscore(
    spread: pd.Series,
    window: int = 30,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling z-score of the spread.

    z = (spread - rolling_mean) / rolling_std
    """
    if min_periods is None:
        min_periods = max(window // 2, 5)

    rolling_mean = spread.rolling(window, min_periods=min_periods).mean()
    rolling_std = spread.rolling(window, min_periods=min_periods).std()

    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"
    return zscore


def compute_bollinger_signal(
    spread: pd.Series,
    window: int = 30,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Band based signals on spread.

    Returns DataFrame with columns: [spread, mean, upper, lower, signal]
    signal: +1 when spread < lower (buy spread), -1 when spread > upper (sell spread)
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()

    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std

    signal = pd.Series(0.0, index=spread.index, name="signal")
    signal[spread > upper] = -1.0  # spread过高，做空spread
    signal[spread < lower] = 1.0  # spread过低，做多spread

    return pd.DataFrame(
        {
            "spread": spread,
            "mean": rolling_mean,
            "upper": upper,
            "lower": lower,
            "signal": signal,
        }
    )
