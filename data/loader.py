"""Data loading utilities for crypto OHLCV data."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_csv(filepath: Union[str, Path], symbol: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV data from CSV file.

    Expected columns: timestamp/datetime, open, high, low, close, volume.
    Returns DataFrame with DatetimeIndex.
    """
    df = pd.read_csv(filepath)

    # 自动识别时间列
    time_col = None
    for col in ["timestamp", "datetime", "date", "time", "open_time"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]

    # 处理时间戳（毫秒或秒）
    if df[time_col].dtype in ("int64", "float64"):
        ts = df[time_col]
        if ts.iloc[0] > 1e12:  # 毫秒
            df[time_col] = pd.to_datetime(ts, unit="ms")
        else:
            df[time_col] = pd.to_datetime(ts, unit="s")
    else:
        df[time_col] = pd.to_datetime(df[time_col])

    df = df.set_index(time_col)
    df.index.name = "datetime"

    # 标准化列名
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "open" in lower and "time" not in lower:
            col_map[col] = "open"
        elif "high" in lower:
            col_map[col] = "high"
        elif "low" in lower:
            col_map[col] = "low"
        elif "close" in lower and "time" not in lower:
            col_map[col] = "close"
        elif "vol" in lower:
            col_map[col] = "volume"
    df = df.rename(columns=col_map)

    # 只保留OHLCV列
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].astype(float)
    df = df.sort_index()

    if symbol:
        df.attrs["symbol"] = symbol
    return df


def load_parquet(filepath: Union[str, Path], symbol: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV data from Parquet file."""
    df = pd.read_parquet(filepath)
    if not isinstance(df.index, pd.DatetimeIndex):
        # 尝试找时间列
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                df = df.set_index(col)
                df.index = pd.to_datetime(df.index)
                break
    df.index.name = "datetime"
    df = df.sort_index()
    if symbol:
        df.attrs["symbol"] = symbol
    return df


def fetch_binance(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    since: Optional[str] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch historical OHLCV data from Binance via ccxt.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT"
        timeframe: Candle period, e.g. "1m", "5m", "1h"
        since: Start time string, e.g. "2024-01-01"
        limit: Number of candles to fetch (max 1000 per request)
    """
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})

    since_ts = None
    if since:
        since_ts = int(pd.Timestamp(since).timestamp() * 1000)

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df.index.name = "datetime"
    df.attrs["symbol"] = symbol
    return df


def load_funding_rate_csv(filepath: Union[str, Path], symbol: Optional[str] = None) -> pd.Series:
    """Load historical funding rate data from CSV.

    Expected format:
        timestamp (or datetime), fundingRate
        1704067200000, 0.0001
        1704096000000, -0.00005
        ...

    Each row = one settlement (every 8 hours on Binance).
    Returns Series with DatetimeIndex, values = funding rate per period.
    """
    df = pd.read_csv(filepath)

    # 识别时间列
    time_col = None
    for col in ["timestamp", "datetime", "calcTime", "fundingTime"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        time_col = df.columns[0]

    if df[time_col].dtype in ("int64", "float64"):
        ts = df[time_col]
        if ts.iloc[0] > 1e12:
            df[time_col] = pd.to_datetime(ts, unit="ms")
        else:
            df[time_col] = pd.to_datetime(ts, unit="s")
    else:
        df[time_col] = pd.to_datetime(df[time_col])

    df = df.set_index(time_col)
    df.index.name = "datetime"

    # 识别 funding rate 列
    rate_col = None
    for col in ["fundingRate", "funding_rate", "rate", "fr"]:
        if col in df.columns:
            rate_col = col
            break
    if rate_col is None:
        rate_col = df.columns[0]

    series = df[rate_col].astype(float).sort_index()
    series.name = symbol or "funding_rate"
    return series


def fetch_binance_funding_rate(
    symbol: str = "BTC/USDT",
    since: Optional[str] = None,
    limit: int = 1000,
) -> pd.Series:
    """Fetch historical funding rate from Binance via ccxt.

    Args:
        symbol: Perpetual pair, e.g. "BTC/USDT"
        since: Start time string, e.g. "2024-01-01"
        limit: Number of records to fetch
    """
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    since_ts = None
    if since:
        since_ts = int(pd.Timestamp(since).timestamp() * 1000)

    params = {"symbol": symbol.replace("/", ""), "limit": limit}
    if since_ts:
        params["startTime"] = since_ts

    response = exchange.fapiPublicGetFundingRate(params)

    records = []
    for r in response:
        records.append({
            "datetime": pd.to_datetime(int(r["fundingTime"]), unit="ms"),
            "funding_rate": float(r["fundingRate"]),
        })

    df = pd.DataFrame(records).set_index("datetime").sort_index()
    series = df["funding_rate"]
    series.name = symbol
    return series
