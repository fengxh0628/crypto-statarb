"""Run backtest on real downloaded data.

Usage:
    python3 -m cst.run_real

Expects data in ./data/ directory (from download_data.py).
"""

from pathlib import Path

import pandas as pd

from cst.config import BacktestConfig
from cst.data.loader import load_csv, load_funding_rate_csv
from cst.data.preprocess import align_prices
from cst.engine.backtest_dynamic import run_dynamic_backtest
from cst.analytics.metrics import print_metrics, print_monthly_returns


def load_all_data(data_dir: str = "./data", timeframe: str = "5m"):
    """Load all OHLCV CSVs and align into price + volume DataFrames.

    Returns:
        (prices, volumes) tuple of DataFrames with same index and columns.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob(f"*_{timeframe}.csv"))

    if not files:
        raise FileNotFoundError(f"No {timeframe} CSV files found in {data_dir}/")

    print(f"  Loading {len(files)} symbols...")
    price_dfs = {}
    volume_dfs = {}
    for f in files:
        # 从文件名提取币种: BTCUSDT_5m.csv -> BTC
        symbol = f.stem.replace(f"USDT_{timeframe}", "").replace(f"usdt_{timeframe}", "")
        try:
            df = load_csv(f, symbol=symbol)
            if len(df) > 1000:
                price_dfs[symbol] = df[["close"]]
                if "volume" in df.columns:
                    volume_dfs[symbol] = df[["volume"]]
        except Exception as e:
            print(f"    Warning: failed to load {f.name}: {e}")

    print(f"  Loaded {len(price_dfs)} symbols successfully")

    # 对齐到5min
    print("  Aligning prices and volumes...")
    prices = align_prices(price_dfs, freq="5min", method="ffill")

    if volume_dfs:
        volumes = align_prices(volume_dfs, freq="5min", method="ffill")
        # 确保列一致
        common_cols = prices.columns.intersection(volumes.columns)
        prices = prices[common_cols]
        volumes = volumes[common_cols]
    else:
        volumes = None

    # 去掉前面全NaN的部分
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
    prices, volumes = load_all_data("./data", "5m")
    print(f"    Symbols: {len(prices.columns)}")
    print(f"    Period: {prices.index[0]} to {prices.index[-1]}")
    print(f"    Duration: {(prices.index[-1] - prices.index[0]).days} days")
    print(f"    Total bars: {len(prices)}")

    # 2. 配置
    config = BacktestConfig(
        # 信号参数
        zscore_window=60,
        hedge_ratio_window=288,  # 1天估计hedge ratio
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=4.0,
        max_hold_bars=576,  # 最大持仓2天

        # 成本
        taker_fee=0.0004,
        slippage_bps=1.0,
        funding_rate=0.0001,
        funding_interval_bars=96,

        # 资金
        capital=10000,
        position_size=0.25,
        max_pairs=4,
        leverage=3.0,

        # 动态选对
        coint_lookback=2016,  # 7天
        coint_recheck_bars=288,  # 每天
        coint_significance=0.05,
        max_half_life=200,
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
