#!/usr/bin/env python3
"""Run alpha strategy backtest with walk-forward validation.

Usage:
    python3 run_alpha.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from alpha.base import AlphaConfig
from alpha.momentum import MomentumAlpha, MomentumConfig
from alpha.backtest import run_alpha_backtest
from alpha.walk_forward import run_walk_forward


def load_all_data(data_dir: str = "~/binance_data/futures/um/monthly/klines", timeframe: str = "5m", symbols: list = None):
    """Load all OHLCV CSVs from binance_historical_data format."""
    data_path = Path(data_dir).expanduser()

    if symbols is None:
        symbol_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        symbols = [d.name for d in symbol_dirs]

    if not symbols:
        raise FileNotFoundError(f"No symbol directories found in {data_dir}/")

    print(f"  Found {len(symbols)} symbols, loading...")
    price_dfs = {}
    volume_dfs = {}
    total = len(symbols)

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

            all_times = []
            all_closes = []
            all_volumes = []

            for f in csv_files:
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

            times = np.concatenate(all_times)
            closes = np.concatenate(all_closes)
            volumes = np.concatenate(all_volumes)

            sort_idx = np.argsort(times)
            times = times[sort_idx]
            closes = closes[sort_idx]
            volumes = volumes[sort_idx]

            idx = pd.to_datetime(times, unit="ms")
            price_dfs[symbol] = pd.DataFrame({"close": closes}, index=idx)
            volume_dfs[symbol] = pd.DataFrame({"volume": volumes}, index=idx)

        except Exception as e:
            print(f"    Warning: failed to load {symbol}: {e}")

    print()
    print(f"  Loaded {len(price_dfs)} symbols successfully")

    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(timeframe, "5min")

    print(f"  Aligning prices and volumes to {freq}...")
    
    from data.preprocess import align_prices
    prices = align_prices(price_dfs, freq=freq, method="ffill")

    if volume_dfs:
        volumes = align_prices(volume_dfs, freq=freq, method="ffill", column="volume")
        common_cols = prices.columns.intersection(volumes.columns)
        prices = prices[common_cols]
        volumes = volumes[common_cols]
    else:
        volumes = None

    prices = prices.dropna(how="all")
    if volumes is not None:
        volumes = volumes.loc[prices.index]

    print(f"  Total symbols: {len(prices.columns)}")
    print(f"  Total bars: {len(prices)}")

    return prices, volumes


def main():
    print("=" * 70)
    print("  Crypto Alpha Framework - Backtest")
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
    ]
    prices, volumes = load_all_data(symbols=symbols)

    # Resample to 1h
    print("\n  Resampling to 1h...")
    prices = prices.resample("1h").last().dropna(how="all")
    if volumes is not None:
        volumes = volumes.resample("1h").sum().reindex(prices.index)
    print(f"  After resample: {len(prices)} bars from {prices.index[0]} to {prices.index[-1]}")

    # 2. 配置
    base_config = {
        'momentum_window': 336,  # 14 天
        'skip_recent': 12,
        'top_k': 5,
        'bottom_k': 5,
        'rebalance_bars': 24,  # 每天
        'position_size': 0.10,
        'leverage': 1.0,
        'taker_fee': 0.0005,
        'slippage_bps': 1.0,
        'funding_rate': 0.0001,
        'funding_interval_bars': 24,
        'symbols': symbols,
    }

    config = MomentumConfig(**base_config)
    strategy = MomentumAlpha(config)

    print(f"\n[2] Config:")
    print(f"    Strategy: Momentum Alpha")
    print(f"    Momentum window: {config.momentum_window}h ({config.momentum_window/24:.0f} days)")
    print(f"    Skip recent: {config.skip_recent}h")
    print(f"    Long/Short: {config.top_k}/{config.bottom_k}")
    print(f"    Rebalance: every {config.rebalance_bars}h")
    print(f"    Leverage: {config.leverage}x | Position size: {config.position_size:.0%}")

    # 3. 完整回测
    print(f"\n[3] Running full backtest...")
    result = run_alpha_backtest(prices, strategy, volumes=volumes)

    metrics = result['metrics']
    print(f"\n[4] Full Backtest Performance:")
    print(f"    Total Return:     {metrics['total_return']:+.2%}")
    print(f"    Benchmark Return: {metrics['benchmark_return']:+.2%}")
    print(f"    Alpha:            {metrics['alpha']:+.2%}")
    print(f"    Net PnL:          {metrics['net_pnl']:+.2%}")
    print(f"    Max Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"    Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}")
    print(f"    Market Beta:      {result['factor_exposures'].get('market_beta', 'N/A')}")

    # 5. Walk-forward 验证
    print(f"\n[5] Running walk-forward validation...")
    wf_result = run_walk_forward(
        prices,
        MomentumAlpha,
        MomentumConfig,
        base_config,
        n_splits=5,
        train_bars=8760,  # 1 year
        volumes=volumes,
    )

    # 6. 保存结果
    result['equity_curve'].to_csv("./data/alpha_equity_curve.csv")
    result['benchmark_curve'].to_csv("./data/benchmark_curve.csv")
    result['weights_history'].to_csv("./data/alpha_weights.csv")
    if not result['factors'].empty:
        result['factors'].to_csv("./data/alpha_factors.csv")
        print(f"\n    Factors saved: {result['factors'].shape[0]} snapshots, {result['factors'].shape[1]} columns")
    if 'combined_equity' in wf_result:
        wf_result['combined_equity'].to_csv("./data/wf_equity_curve.csv")
    
    print(f"\n    Saved: ./data/alpha_equity_curve.csv")
    print(f"    Saved: ./data/benchmark_curve.csv")
    print(f"    Saved: ./data/alpha_weights.csv")
    if not result['factors'].empty:
        print(f"    Saved: ./data/alpha_factors.csv")
    print(f"    Saved: ./data/wf_equity_curve.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
