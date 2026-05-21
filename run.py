#!/usr/bin/env python3
"""Run backtest on real downloaded data.

Usage:
    python3 -m cst.run

Expects data in ~/binance_data/futures/um/monthly/klines/ directory (from download_data_full.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig
from data.loader import load_csv, load_funding_rate_csv
from data.preprocess import align_prices
from engine.backtest_dynamic import run_dynamic_backtest
from analytics.metrics import print_metrics, print_monthly_returns


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
    print("  Crypto Stat Arb - Real Data Backtest")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] Loading data...")
    # 30个主流币种，可形成435对
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT",
        "DOTUSDT", "LTCUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT",
        "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT", "SUIUSDT",
        "PEPEUSDT", "WIFUSDT", "FETUSDT", "RENDERUSDT", "INJUSDT",
        "TIAUSDT", "SEIUSDT", "TRXUSDT", "SHIBUSDT", "UNIUSDT",
    ]
    prices, volumes = load_all_data(symbols=symbols)
    print(f"    Symbols: {len(prices.columns)}")
    print(f"    Period: {prices.index[0]} to {prices.index[-1]}")
    print(f"    Duration: {(prices.index[-1] - prices.index[0]).days} days")
    print(f"    Total bars: {len(prices)}")

    # 2. 配置
    config = BacktestConfig(
        # 信号参数
        zscore_window=24,  # 24 bars (1 day for 1h)
        hedge_ratio_window=168,  # 168 bars (1 week for 1h)
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.5,
        max_hold_bars=72,  # 最大持仓 3 天 (72h)

        # 成本
        taker_fee=0.0002,  # Maker fee
        slippage_bps=0.5,
        funding_rate=0.0001,
        funding_interval_bars=24,  # 8h for 1h data

        # 资金
        capital=10000,
        position_size=0.10,  # 每对 10% 资金
        max_pairs=3,  # 最多 3 对
        leverage=1.0,

        # 动态选对
        coint_lookback=504,  # 3 周 (504h)
        coint_recheck_bars=168,  # 每周重新扫描
        coint_significance=0.05,
        max_half_life=48,  # 半衰期不超过 48 bar (2 天)

        # 过滤条件
        vol_filter_window=48,  # 波动率过滤窗口
        min_vol_percentile=0.2,  # 排除低波动率时段
        trend_filter_window=168,  # 趋势过滤窗口
        max_trend_threshold=0.02,  # 排除强趋势时段
    )

    print(f"\n[2] Config:")
    print(f"    Capital: {config.capital} USDT | Leverage: {config.leverage}x | Max pairs: {config.max_pairs}")
    print(f"    Coint lookback: {config.coint_lookback} bars ({config.coint_lookback*5/60:.0f}h)")
    print(f"    Recheck every: {config.coint_recheck_bars} bars ({config.coint_recheck_bars*5/60:.0f}h)")

    # 3. 运行回测
    print(f"\n[3] Running dynamic backtest...")
    result = run_dynamic_backtest(prices, config, volumes=volumes)

    # 4. 绩效
    print(f"\n[4] Performance:")
    print_metrics(result.total_metrics, capital=config.capital)

    # 5. 月度收益
    print_monthly_returns(result.monthly_returns)

    # 6. 交易统计
    print("\n[6] Trade Statistics:")
    opens = [t for t in result.trades if t.action == "open"]
    closes = [t for t in result.trades if t.action == "close"]
    print(f"    Total opens: {len(opens)}")
    print(f"    Total closes: {len(closes)}")
    if closes:
        reasons = {}
        for t in closes:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        print(f"    Close reasons: {reasons}")

    avg_active = result.active_pairs_count.iloc[config.coint_lookback:].mean()
    max_active = result.active_pairs_count.max()
    print(f"    Avg active pairs: {avg_active:.1f}")
    print(f"    Max active pairs: {max_active}")

    print(f"\n    Portfolio PnL: {result.total_metrics['total_return']*config.capital:+.2f} USDT")
    print(f"    Max Drawdown: {result.total_metrics['max_drawdown']*config.capital:.2f} USDT")

    # 7. 保存净值曲线
    result.equity_curve.to_csv("./data/equity_curve.csv")
    result.monthly_returns.to_csv("./data/monthly_returns.csv")
    print("\n    Saved: ./data/equity_curve.csv, ./data/monthly_returns.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
