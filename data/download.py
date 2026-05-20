"""Batch download crypto data (OHLCV + funding rate) via ccxt."""

import time
from pathlib import Path
from typing import List, Optional

import pandas as pd


def get_top_symbols(n: int = 80) -> List[str]:
    """Get top N perpetual contract symbols by 24h volume from Binance.

    Returns list of symbols like ["BTC/USDT", "ETH/USDT", ...].
    """
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    tickers = exchange.fetch_tickers()

    usdt_perps = []
    for symbol, ticker in tickers.items():
        if symbol.endswith("/USDT") and ticker.get("quoteVolume"):
            usdt_perps.append((symbol, ticker["quoteVolume"]))

    usdt_perps.sort(key=lambda x: x[1], reverse=True)

    symbols = [s[0] for s in usdt_perps[:n]]
    print(f"  Found {len(symbols)} symbols by volume (top {n})")
    return symbols


def get_all_perpetual_symbols() -> List[str]:
    """Get ALL USDT perpetual contract symbols from Binance.

    Returns all available symbols regardless of volume,
    to avoid survivorship bias in historical backtests.
    """
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    markets = exchange.load_markets()

    symbols = []
    for symbol, market in markets.items():
        if (market.get("swap")
            and market.get("quote") == "USDT"
            and market.get("active")):
            symbols.append(symbol)

    symbols.sort()
    print(f"  Found {len(symbols)} USDT perpetual symbols")
    return symbols


def download_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    since: str = "2024-01-01",
    until: Optional[str] = None,
    output_dir: str = "./data",
) -> Path:
    """Download historical OHLCV data from Binance futures.

    Handles pagination automatically (1000 bars per request).

    Args:
        symbol: Trading pair, e.g. "BTC/USDT"
        timeframe: Candle period, e.g. "5m"
        since: Start date string
        until: End date string (default: now)
        output_dir: Directory to save CSV

    Returns:
        Path to saved CSV file
    """
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    since_ts = int(pd.Timestamp(since).timestamp() * 1000)
    until_ts = int(pd.Timestamp(until).timestamp() * 1000) if until else int(time.time() * 1000)

    # 每根bar的时间间隔（ms）
    tf_ms = exchange.parse_timeframe(timeframe) * 1000

    all_ohlcv = []
    current_ts = since_ts

    print(f"  Downloading {symbol} {timeframe} from {since}...")
    while current_ts < until_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe, since=current_ts, limit=1000
            )
        except Exception as e:
            print(f"    Error at {pd.Timestamp(current_ts, unit='ms')}: {e}")
            break

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        current_ts = ohlcv[-1][0] + tf_ms  # 下一个bar的起始时间

        # 进度提示
        if len(all_ohlcv) % 5000 == 0:
            print(f"    ... {len(all_ohlcv)} bars downloaded")

        time.sleep(0.1)  # rate limit

    if not all_ohlcv:
        print(f"    No data for {symbol}")
        return None

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # 去重
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # 截止到 until
    df = df[df["timestamp"] <= until_ts]

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol.replace('/', '')}_{timeframe}.csv"
    filepath = output_path / filename
    df.to_csv(filepath, index=False)

    print(f"    Saved {len(df)} bars to {filepath}")
    return filepath


def download_funding_rate(
    symbol: str = "BTC/USDT",
    since: str = "2024-01-01",
    until: Optional[str] = None,
    output_dir: str = "./data",
) -> Path:
    """Download historical funding rate from Binance futures.

    Args:
        symbol: Perpetual pair, e.g. "BTC/USDT"
        since: Start date string
        until: End date string (default: now)
        output_dir: Directory to save CSV

    Returns:
        Path to saved CSV file
    """
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    since_ts = int(pd.Timestamp(since).timestamp() * 1000)
    until_ts = int(pd.Timestamp(until).timestamp() * 1000) if until else int(time.time() * 1000)

    all_records = []
    current_ts = since_ts

    print(f"  Downloading {symbol} funding rate from {since}...")
    while current_ts < until_ts:
        try:
            params = {
                "symbol": symbol.replace("/", ""),
                "startTime": current_ts,
                "limit": 1000,
            }
            response = exchange.fapiPublicGetFundingRate(params)
        except Exception as e:
            print(f"    Error: {e}")
            break

        if not response:
            break

        for r in response:
            all_records.append({
                "timestamp": int(r["fundingTime"]),
                "fundingRate": float(r["fundingRate"]),
            })

        # 下一页起始
        last_ts = int(response[-1]["fundingTime"])
        if last_ts == current_ts:
            break
        current_ts = last_ts + 1

        time.sleep(0.1)

    if not all_records:
        print(f"    No funding data for {symbol}")
        return None

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] <= until_ts]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol.replace('/', '')}_funding.csv"
    filepath = output_path / filename
    df.to_csv(filepath, index=False)

    print(f"    Saved {len(df)} records to {filepath}")
    return filepath


def download_universe(
    symbols: list = None,
    timeframe: str = "5m",
    since: str = "2024-01-01",
    until: Optional[str] = None,
    output_dir: str = "./data",
) -> dict:
    """Download OHLCV + funding rate for multiple symbols.

    Args:
        symbols: List of trading pairs. Default: major cryptos.
        timeframe: K-line period
        since: Start date
        until: End date
        output_dir: Output directory

    Returns:
        Dict with paths: {"ohlcv": {symbol: path}, "funding": {symbol: path}}
    """
    if symbols is None:
        symbols = get_top_symbols(80)

    print(f"Downloading {len(symbols)} symbols, {timeframe}, from {since}")
    print("=" * 50)

    paths = {"ohlcv": {}, "funding": {}}

    for sym in symbols:
        # K线数据
        p = download_ohlcv(sym, timeframe, since, until, output_dir)
        if p:
            paths["ohlcv"][sym] = p

        # Funding rate
        p = download_funding_rate(sym, since, until, output_dir)
        if p:
            paths["funding"][sym] = p

        print()

    print("=" * 50)
    print(f"Done. Files saved to {output_dir}/")
    return paths


if __name__ == "__main__":
    download_universe(
        timeframe="5m",
        since="2024-01-01",
        until="2024-07-01",
        output_dir="./data",
    )
