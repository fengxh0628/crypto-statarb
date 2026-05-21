#!/usr/bin/env python3
"""Run momentum strategy backtest on real downloaded data.

Usage:
    python3 run_momentum.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig
from data.loader import load_csv, load_funding_rate_csv
from data.preprocess import align_prices
from strategy.momentum import MomentumConfig, run_momentum_backtest


def load_all_data(data_dir: str = "~/binance_data/futures/um/monthly/klines", timeframe: str = "5m", symbols: list = None):
    """Load all OHLCV CSVs from binance_historical_data format and align into price + volume DataFrames.

    Data format:
        {data_dir}/{SYMBOL}/{timeframe}/{SYMBOL}-{timeframe}-{YYYY-MM}.csv

    Args:
        data_dir: Base directory containing symbol subdirectories
        timeframe: Candle period (e.g. "5m")
        symbols: Optional list of symbols to load. If None, loads all available.

    Returns:
        (prices, volumes) tuple of DataFrames with same index and columns.
    """
    data_path = Path(data_dir).expanduser()

    if symbols is None:
        # Auto-detect all available symbols
        symbol_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        symbols = [d.name for d in symbol_dirs]

    if not symbols:
        raise FileNotFoundError(f"No symbol directories found in {data_dir}/")

    print(f"  Found {len(symbols)} symbols, loading...")
    price_dfs = {}
    volume_dfs = {}
    total = len(symbols)

    # Binance historical data columns
    binance_cols = ["open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "count", "taker_buy_volume",
                    "taker_buy_quote_volume", "ignore"]

    for i, symbol in enumerate(symbols, 1):
        symbol_dir = data_path / symbol / timeframe
        if not symbol_dir.exists():
            continue

        csv_files = sorted(symbol_dir.glob(f"{symbol}-{timeframe}-*.csv"))
        if not csv_files:
            continue

        try:
            print(f"  [{i}/{total}] Loading {symbol} ({len(csv_files)} files)...", end="\r")

            # Read all files efficiently
            all_times = []
            all_closes = []
            all_volumes = []

            for f in csv_files:
                # Fast header check: read first few bytes
                with open(f, 'rb') as fh:
                    has_header = fh.read(10).startswith(b"open_time")
                if has_header:
                    df = pd.read_csv(f, usecols=[0, 4, 5], names=["open_time", "close", "volume"], header=0, dtype={"open_time": "int64"})
                else:
                    df = pd.read_csv(f, usecols=[0, 4, 5], names=["open_time", "close", "volume"], header=None, dtype={"open_time": "int64"})
                all_times.append(df["open_time"].values)
                all_closes.append(df["close"].values)
                all_volumes.append(df["volume"].values)

            if not all_times:
                continue

            # Concatenate numpy arrays (much faster than pd.concat)
            times = np.concatenate(all_times)
            closes = np.concatenate(all_closes)
            volumes = np.concatenate(all_volumes)

            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            closes = closes[sort_idx]
            volumes = volumes[sort_idx]

            # Convert timestamps once
            idx = pd.to_datetime(times, unit="ms")
            price_dfs[symbol] = pd.DataFrame({"close": closes}, index=idx)
            volume_dfs[symbol] = pd.DataFrame({"volume": volumes}, index=idx)

        except Exception as e:
            print(f"    Warning: failed to load {symbol}: {e}")

    print()  # Newline after progress
    print(f"  Loaded {len(price_dfs)} symbols successfully")

    # Align to timeframe frequency
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(timeframe, "5min")

    print(f"  Aligning prices and volumes to {freq}...")
    prices = align_prices(price_dfs, freq=freq, method="ffill")

    if volume_dfs:
        volumes = align_prices(volume_dfs, freq=freq, method="ffill", column="volume")
        # Ensure consistent columns
        common_cols = prices.columns.intersection(volumes.columns)
        prices = prices[common_cols]
        volumes = volumes[common_cols]
    else:
        volumes = None

    # Remove leading all-NaN rows
    prices = prices.dropna(how="all")
    if volumes is not None:
        volumes = volumes.loc[prices.index]

    print(f"  Total symbols: {len(prices.columns)}")
    print(f"  Total bars: {len(prices)}")

    return prices, volumes


def main():
    print("=" * 70)
    print("  Crypto Momentum Strategy - Backtest")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] Loading data...")
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT",
        "DOTUSDT", "LTCUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT",
        "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT", "SUIUSDT",
        "PEPEUSDT", "WIFUSDT", "FETUSDT", "RENDERUSDT", "INJUSDT",
        "TIAUSDT", "SEIUSDT", "TRXUSDT", "SHIBUSDT", "UNIUSDT",
        "AAVEUSDT", "ALGOUSDT", "APEUSDT", "AXSUSDT", "BCHUSDT",
        "CRVUSDT", "DYDXUSDT", "EGLDUSDT", "ENSUSDT", "FLOWUSDT",
        "GALAUSDT", "GMXUSDT", "GRTUSDT", "IMXUSDT", "LDOUSDT",
        "MKRUSDT", "ORDIUSDT", "RUNEUSDT", "SANDUSDT", "STXUSDT",
        "ACEUSDT", "ACHUSDT", "ACTUSDT", "AERGOUSDT", "AEROUSDT",
        "AEVOUSDT", "AGIXUSDT", "AGLDUSDT", "AKTUSDT", "ALICEUSDT",
        "ALPHAUSDT", "ALTUSDT", "AMBUSDT", "ANKRUSDT", "ARPAUSDT",
        "ASTRUSDT", "ATAUSDT", "AUCTIONUSDT", "BADGERUSDT", "BAKEUSDT",
        "BALUSDT", "BANDUSDT", "BELUSDT", "BERAUSDT", "BICOUSDT",
        "BIGTIMEUSDT", "BLURUSDT", "BLZUSDT", "BNTUSDT", "BNXUSDT",
        "BOBUSDT", "BOMEUSDT", "BONDUSDT", "BRETTUSDT", "CELOUSDT",
        "CELRUSDT", "CFXUSDT", "CHRUSDT", "CHZUSDT", "CITYUSDT",
        "CKBUSDT", "COMPUSDT", "COTIUSDT", "CROUSDT", "CTKUSDT",
        "CTSIUSDT", "CVCUSDT", "CVXUSDT", "DASHUSDT", "DEGENUSDT",
        "DENTUSDT", "DEXEUSDT", "DGBUSDT", "DOCKUSDT", "DODOUSDT",
        "DOGSUSDT", "DUSKUSDT", "EDUUSDT", "EIGENUSDT", "ENAUSDT",
    ]
    prices, volumes = load_all_data(symbols=symbols)

    # Resample to 1h
    print("\n  Resampling to 1h...")
    prices = prices.resample("1h").last().dropna(how="all")
    if volumes is not None:
        volumes = volumes.resample("1h").sum().reindex(prices.index)
    print(f"  After resample: {len(prices)} bars from {prices.index[0]} to {prices.index[-1]}")

    # 2. 配置
    config = MomentumConfig(
        momentum_window=336,  # 14 天动量
        skip_recent=12,  # 跳过最近 12 小时
        top_k=5,  # 做多前 5
        bottom_k=5,  # 做空后 5
        rebalance_bars=24,  # 每天调仓
        position_size=0.10,  # 每个符号 10% 仓位
        leverage=1.0,  # 1 倍杠杆
        taker_fee=0.0005,  # Taker fee 0.05%
        slippage_bps=1.0,
        funding_rate=0.0001,
        funding_interval_bars=24,
    )

    print(f"\n[2] Config:")
    print(f"    Momentum window: {config.momentum_window}h ({config.momentum_window/24:.0f} days)")
    print(f"    Skip recent: {config.skip_recent}h")
    print(f"    Long/Short: {config.top_k}/{config.bottom_k}")
    print(f"    Rebalance: every {config.rebalance_bars}h ({config.rebalance_bars/24:.0f} days)")
    print(f"    Leverage: {config.leverage}x | Position size: {config.position_size:.0%}")

    # 3. 运行回测
    print(f"\n[3] Running momentum backtest...")
    result = run_momentum_backtest(prices, config, volumes=volumes)

    # 4. 绩效
    metrics = result["metrics"]
    print(f"\n[4] Performance:")
    print(f"    Gross PnL:    {metrics['gross_pnl']:+.2%}")
    print(f"    Trading Fees: -{metrics['trading_fees']:.2%}")
    print(f"    Funding:      -{metrics['funding_cost']:.2%}")
    print(f"    Net PnL:      {metrics['net_pnl']:+.2%}")
    
    eq = result["equity_curve"]
    print(f"    Final Equity: {eq.iloc[-1]:.4f}")
    print(f"    Max Equity:   {eq.max():.4f}")
    print(f"    Min Equity:   {eq.min():.4f}")
    print(f"    Max Drawdown: {(eq / eq.cummax() - 1).min():.2%}")
    print(f"    Period:       {eq.index[0]} to {eq.index[-1]}")
    print(f"    Duration:     {(eq.index[-1] - eq.index[0]).days} days")

    # 5. 保存净值曲线
    eq.to_csv("./data/momentum_equity_curve.csv")
    print(f"\n    Saved: ./data/momentum_equity_curve.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
