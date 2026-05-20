"""Example: Dynamic pairs trading backtest with continuous monitoring."""

import numpy as np
import pandas as pd

from cst.config import BacktestConfig
from cst.engine.backtest_dynamic import run_dynamic_backtest
from cst.analytics.metrics import print_metrics, print_monthly_returns
from cst.analytics.plot import plot_backtest


def generate_synthetic_universe(
    n_bars: int = 20160,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic crypto price universe with known cointegration structure.

    Creates 6 assets:
    - Group 1 (cointegrated): BTC, ETH, MATIC - share a common stochastic trend
    - Group 2 (cointegrated): SOL, AVAX - share a different common trend
    - BNB: independent, not cointegrated with others
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

    # 两个共同随机游走
    trend1 = np.cumsum(rng.normal(0, 0.0003, n_bars))
    trend2 = np.cumsum(rng.normal(0, 0.0003, n_bars))

    # OU过程（均值回归噪声）
    def ou_process(n, theta=0.02, sigma=0.0004):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i-1] + theta * (0 - x[i-1]) + sigma * rng.normal()
        return x

    prices_dict = {}

    # Group 1: BTC, ETH, MATIC
    prices_dict["BTC"] = np.exp(np.log(30000) + trend1 + ou_process(n_bars))
    prices_dict["ETH"] = np.exp(np.log(2000) + 0.85 * trend1 + ou_process(n_bars))
    prices_dict["MATIC"] = np.exp(np.log(0.8) + 0.9 * trend1 + ou_process(n_bars))

    # Group 2: SOL, AVAX
    prices_dict["SOL"] = np.exp(np.log(100) + trend2 + ou_process(n_bars))
    prices_dict["AVAX"] = np.exp(np.log(35) + 0.95 * trend2 + ou_process(n_bars))

    # BNB: 独立趋势
    trend3 = np.cumsum(rng.normal(0, 0.00025, n_bars))
    prices_dict["BNB"] = np.exp(np.log(300) + trend3 + ou_process(n_bars))

    return pd.DataFrame(prices_dict, index=index)


def main():
    print("=" * 70)
    print("  Crypto Stat Arb - Dynamic Pairs Trading Backtest")
    print("=" * 70)

    # 1. 生成数据（5min频率，约70天）
    n_bars = 20160
    print(f"\n[1] Generating synthetic 6-asset universe ({n_bars} bars @ 5min)...")
    prices = generate_synthetic_universe(n_bars=n_bars)
    print(f"    Assets: {list(prices.columns)}")
    print(f"    Period: {prices.index[0]} to {prices.index[-1]}")
    print(f"    Duration: {(prices.index[-1] - prices.index[0]).days} days")

    # 2. 配置
    config = BacktestConfig(
        # 信号参数
        zscore_window=60,
        hedge_ratio_window=240,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=4.0,
        max_hold_bars=576,  # 最大持仓2天(576*5min)

        # 成本
        taker_fee=0.0004,
        slippage_bps=1.0,
        funding_rate=0.0001,
        funding_interval_bars=96,  # 8h

        # 资金
        capital=10000,
        position_size=0.25,
        max_pairs=4,
        leverage=3.0,

        # 动态选对
        coint_lookback=2016,  # 7天
        coint_recheck_bars=288,  # 每天重新扫描
        coint_significance=0.05,
        max_half_life=200,
    )

    print(f"\n[2] Config:")
    print(f"    Capital: {config.capital} USDT | Leverage: {config.leverage}x | Max pairs: {config.max_pairs}")
    print(f"    Coint lookback: {config.coint_lookback} bars ({config.coint_lookback*5/60:.0f}h)")
    print(f"    Recheck every: {config.coint_recheck_bars} bars ({config.coint_recheck_bars*5/60:.0f}h)")
    print(f"    Max half-life: {config.max_half_life} bars ({config.max_half_life*5/60:.0f}h)")

    # 3. 运行动态回测
    print(f"\n[3] Running dynamic backtest...")
    result = run_dynamic_backtest(prices, config)

    # 4. 绩效
    print(f"\n[4] Performance:")
    pair_capital = config.capital * config.position_size * config.max_pairs
    print_metrics(result.total_metrics, capital=config.capital)

    # 5. 月度收益
    print_monthly_returns(result.monthly_returns)

    # 6. 交易统计
    print(f"\n[6] Trade Statistics:")
    opens = [t for t in result.trades if t.action == "open"]
    closes = [t for t in result.trades if t.action == "close"]
    print(f"    Total opens: {len(opens)}")
    print(f"    Total closes: {len(closes)}")
    if closes:
        reasons = {}
        for t in closes:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        print(f"    Close reasons: {reasons}")

    # 7. 活跃配对数
    avg_active = result.active_pairs_count.iloc[config.coint_lookback:].mean()
    max_active = result.active_pairs_count.max()
    print(f"    Avg active pairs: {avg_active:.1f}")
    print(f"    Max active pairs: {max_active}")

    print(f"\n    Portfolio PnL: {result.total_metrics['total_return']*config.capital:+.2f} USDT")
    print(f"    Max Drawdown: {result.total_metrics['max_drawdown']*config.capital:.2f} USDT")

    print("\nDone!")


if __name__ == "__main__":
    main()
