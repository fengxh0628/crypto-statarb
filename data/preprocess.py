"""Data preprocessing and alignment."""

import pandas as pd
import numpy as np


def align_prices(
    dfs: dict[str, pd.DataFrame],
    freq: str = "1min",
    method: str = "ffill",
    column: str = "close",
) -> pd.DataFrame:
    """Align multiple price/volume series to common time index.

    Args:
        dfs: Dict of {symbol: DataFrame}
        freq: Resample frequency
        method: Fill method for missing values ('ffill' or 'drop')
        column: Column name to extract (default 'close', use 'volume' for volumes)

    Returns:
        DataFrame with columns = symbol names
    """
    prices = {}
    for symbol, df in dfs.items():
        s = df[column].resample(freq).last()
        prices[symbol] = s

    result = pd.DataFrame(prices)

    if method == "ffill":
        result = result.ffill()
    elif method == "drop":
        result = result.dropna()

    # 去掉全为NaN的行（开头部分）
    result = result.dropna(how="all")
    return result


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute returns from price DataFrame.

    Args:
        prices: DataFrame of prices
        method: 'log' for log returns, 'simple' for simple returns
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def filter_outliers(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 60,
    n_std: float = 5.0,
) -> pd.DataFrame:
    """Filter outlier bars based on rolling z-score.

    Replaces outlier values with NaN, then forward fills.
    """
    rolling_mean = df[column].rolling(window, min_periods=10).mean()
    rolling_std = df[column].rolling(window, min_periods=10).std()
    zscore = (df[column] - rolling_mean) / rolling_std

    mask = zscore.abs() > n_std
    if mask.any():
        df = df.copy()
        df.loc[mask, column] = np.nan
        df[column] = df[column].ffill()
    return df
