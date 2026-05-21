"""Dynamic backtest engine with continuous pair monitoring."""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from config import BacktestConfig
from pairs.fast_adf import batch_adf, batch_cointegration_test
from pairs.spread import compute_hedge_ratio, compute_spread, estimate_ou_params
from strategy.signals import compute_zscore


@dataclass
class TradeRecord:
    """Single trade (open or close) record."""

    pair: tuple
    side: int  # +1 long spread, -1 short spread
    action: str  # "open" or "close"
    reason: str  # "signal", "coint_fail", "stop_loss", "timeout"
    bar_idx: int
    timestamp: object
    zscore: float
    pnl: float = 0.0  # PnL at close (0 for open)


@dataclass
class ActivePosition:
    """State of an active pair position."""

    pair: tuple
    side: int  # +1 or -1
    entry_bar: int
    entry_zscore: float
    hedge_ratio: float
    cumulative_pnl: float = 0.0
    position_size: float = 0.10  # 动态仓位大小


@dataclass
class DynamicBacktestResult:
    """Results from dynamic backtest."""

    equity_curve: pd.Series  # 组合净值
    portfolio_pnl: pd.Series  # 逐bar PnL
    monthly_returns: pd.DataFrame  # 月度收益拆分
    trades: list  # 所有交易记录
    active_pairs_count: pd.Series  # 每bar活跃配对数
    total_metrics: dict  # 总绩效指标


class _SignalCache:
    """On-demand signal computation and caching for active/candidate pairs."""

    def __init__(self, prices: pd.DataFrame, config: BacktestConfig):
        self.prices = prices
        self.config = config
        self.n_bars = len(prices)
        # 预计算所有币种的 log price
        self.log_prices = np.log(prices.values)  # (n_bars, n_symbols)
        self.symbol_idx = {sym: i for i, sym in enumerate(prices.columns)}
        # 缓存: {pair: {"hr": ndarray, "spread": ndarray, "zscore": ndarray}}
        self._cache = {}

    def get_signal(self, pair: tuple, t: int):
        """Get z-score and hedge ratio for a pair at bar t.

        Computes and caches if not already done.
        Returns (zscore, hedge_ratio) or (nan, nan) if not computable.
        """
        if pair not in self._cache:
            self._compute_pair(pair)

        cache = self._cache[pair]
        z = cache["zscore"][t]
        hr = cache["hr"][t]
        return z, hr

    def _compute_pair(self, pair: tuple):
        """Compute rolling hedge ratio, spread, zscore for a pair."""
        sym_a, sym_b = pair
        ia = self.symbol_idx[sym_a]
        ib = self.symbol_idx[sym_b]

        log_a = self.log_prices[:, ia]
        log_b = self.log_prices[:, ib]

        window = self.config.hedge_ratio_window
        n = self.n_bars

        # Pure numpy rolling hedge ratio: cov(a,b) / var(b)
        # Uses cumulative sums for O(n) computation
        hr = np.full(n, np.nan)
        spread = np.full(n, np.nan)

        if n > window:
            # Shift by 1 so window at t covers [t-window, t)
            a = np.empty(n)
            b = np.empty(n)
            a[1:] = log_a[:-1]
            b[1:] = log_b[:-1]
            a[0] = 0.0  # Will be excluded by window cutoff
            b[0] = 0.0

            # Rolling sums using cumsum
            def rolling_sum(x, w):
                cs = np.cumsum(x)
                result = np.empty(n)
                result[:w] = np.nan
                result[w:] = cs[w:] - cs[:-w]
                return result

            sum_a = rolling_sum(a, window)
            sum_b = rolling_sum(b, window)
            sum_ab = rolling_sum(a * b, window)
            sum_b2 = rolling_sum(b * b, window)

            # cov(a,b) = E[ab] - E[a]E[b], var(b) = E[b^2] - E[b]^2
            inv_w = 1.0 / window
            inv_ddof = 1.0 / (window - 1)

            cov = (sum_ab - sum_a * sum_b * inv_w) * inv_ddof
            var_b = (sum_b2 - sum_b * sum_b * inv_w) * inv_ddof

            valid = var_b > 0
            hr = np.where(valid, cov / var_b, np.nan)
            # First window element includes artificial 0 at index 0, set to NaN
            hr[window] = np.nan
            spread = np.where(~np.isnan(hr), log_a - hr * log_b, np.nan)

        # Vectorized z-score
        zw = self.config.zscore_window
        zscore = np.full(n, np.nan)

        if n > window + zw:
            sp = pd.Series(spread)
            rolling_mean = sp.rolling(zw, min_periods=zw // 2).mean()
            rolling_std = sp.rolling(zw, min_periods=zw // 2).std()

            valid_z = ~np.isnan(spread) & (rolling_std.values > 0)
            zscore = np.where(
                valid_z,
                (spread - rolling_mean.values) / rolling_std.values,
                np.nan,
            )

        self._cache[pair] = {"hr": hr, "spread": spread, "zscore": zscore}

    def get_pair_return(self, pair: tuple, t: int):
        """Get log return of pair at bar t."""
        sym_a, sym_b = pair
        ia = self.symbol_idx[sym_a]
        ib = self.symbol_idx[sym_b]

        if t < 1:
            return 0.0, 0.0

        ret_a = self.log_prices[t, ia] - self.log_prices[t - 1, ia]
        ret_b = self.log_prices[t, ib] - self.log_prices[t - 1, ib]

        if np.isnan(ret_a) or np.isnan(ret_b):
            return 0.0, 0.0

        return ret_a, ret_b

    def evict(self, pair: tuple):
        """Remove pair from cache to free memory."""
        self._cache.pop(pair, None)


def _check_cointegration_batch(
    prices: pd.DataFrame,
    start: int,
    end: int,
    config: BacktestConfig,
    volumes: Optional[pd.DataFrame] = None,
    min_volume_quantile: float = 0.5,
) -> dict:
    """Run cointegration test on all pairs for a given window.

    Uses batch_adf for fast vectorized testing.

    Returns dict: {(sym_a, sym_b): {"pvalue": float, "half_life": float}}
    """
    symbols = prices.columns.tolist()
    window_prices = prices.iloc[start:end]

    # 流动性筛选
    if volumes is not None:
        window_vol = volumes.iloc[start:end]
        turnover = window_vol * window_prices
        avg_turnover = turnover.mean()
        # 排除全NaN的
        avg_turnover = avg_turnover.dropna()
        if len(avg_turnover) < 2:
            return {}
        vol_threshold = avg_turnover.quantile(min_volume_quantile)
        liquid_symbols = avg_turnover[avg_turnover >= vol_threshold].index.tolist()
        valid_count = window_prices[liquid_symbols].notna().sum()
        liquid_symbols = [s for s in liquid_symbols if valid_count[s] > len(window_prices) * 0.8]
    else:
        valid_count = window_prices.notna().sum()
        liquid_symbols = [s for s in symbols if valid_count[s] > len(window_prices) * 0.8]

    if len(liquid_symbols) < 2:
        return {}

    # 相关系数预筛
    corr_matrix = window_prices[liquid_symbols].corr()
    n = len(liquid_symbols)

    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        if abs(corr_matrix.iloc[i, j]) >= 0.5:
            candidate_pairs.append((liquid_symbols[i], liquid_symbols[j]))

    if not candidate_pairs:
        return {}

    # 用 batch_cointegration_test 批量检验
    coint_results = batch_cointegration_test(
        window_prices[liquid_symbols],
        start=0, end=len(window_prices),
        pairs=candidate_pairs,
        lags=1,
        significance="5%",
    )

    # 估计半衰期，过滤
    results = {}
    for pair, info in coint_results.items():
        hr = info["hedge_ratio"]
        sym_a, sym_b = pair
        log_a = np.log(window_prices[sym_a].values)
        log_b = np.log(window_prices[sym_b].values)
        spread = log_a - hr * log_b

        # 去除NaN
        spread_clean = spread[~np.isnan(spread)]
        if len(spread_clean) < 50:
            continue

        try:
            ou = estimate_ou_params(pd.Series(spread_clean))
            half_life = ou["half_life"] if ou["stationary"] else np.inf
        except Exception:
            half_life = np.inf

        if half_life > config.max_half_life:
            continue

        results[pair] = {
            "half_life": half_life,
            "hedge_ratio": hr,
        }

    return results


def run_dynamic_backtest(
    prices: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    volumes: Optional[pd.DataFrame] = None,
) -> DynamicBacktestResult:
    """Run dynamic pairs trading backtest with continuous monitoring.

    Logic per bar:
      1. Every coint_recheck_bars: re-scan pairs for cointegration (batch_adf)
      2. Check active positions: close if coint fails, signal triggers, or timeout
      3. If slots available: open new positions on valid pairs with entry signal

    Signals are computed on-demand and cached (no full precompute).

    Args:
        prices: DataFrame with columns = symbol names, index = datetime, values = close prices
        config: Backtest configuration
        volumes: Optional DataFrame with volume data for liquidity filtering.

    Returns:
        DynamicBacktestResult
    """
    if config is None:
        config = BacktestConfig()

    n_bars = len(prices)
    index = prices.index

    # 按需信号缓存
    print("  Initializing signal cache...")
    sig_cache = _SignalCache(prices, config)
    print(f"  {len(prices.columns)} symbols, {n_bars} bars")

    # 状态
    active_positions = {}  # {(sym_a, sym_b): ActivePosition}
    valid_pairs = {}  # 当前协整有效的配对
    trades = []
    portfolio_pnl = np.zeros(n_bars)
    active_count = np.zeros(n_bars, dtype=int)
    pair_monthly_pnl = {}

    # 手续费统计
    total_fees = 0.0
    total_funding = 0.0
    total_gross_pnl = 0.0

    start_bar = config.coint_lookback
    fee_rate = (config.taker_fee + config.slippage_bps / 10000.0) * 2 * config.leverage

    print(f"  Running from bar {start_bar} to {n_bars}...")
    last_progress = 0
    equity = 1.0  # Track equity for early exit
    for t in range(start_bar, n_bars):
        # 进度提示
        progress = (t - start_bar) * 100 // (n_bars - start_bar)
        if progress >= last_progress + 5:
            print(f"    {progress}% ({t}/{n_bars})", end="\r")
            last_progress = progress

        # --- 1. 定期重新扫描协整 ---
        if (t - start_bar) % config.coint_recheck_bars == 0:
            valid_pairs = _check_cointegration_batch(
                prices, t - config.coint_lookback, t, config,
                volumes=volumes,
            )
            # 为 valid pairs 预热信号缓存
            for pair in valid_pairs:
                sig_cache.get_signal(pair, t)

        # --- 2. 管理现有持仓 ---
        n_closes_this_bar = 0
        for pair in list(active_positions.keys()):
            pos = active_positions[pair]
            z, _ = sig_cache.get_signal(pair, t)
            hold_bars = t - pos.entry_bar

            close_reason = None

            if pair not in valid_pairs:
                close_reason = "coint_fail"
            elif not np.isnan(z) and abs(z) > config.stop_loss_threshold:
                close_reason = "stop_loss"
            elif hold_bars >= config.max_hold_bars:
                close_reason = "timeout"
            elif not np.isnan(z) and abs(z) < config.exit_threshold:
                close_reason = "signal"

            if close_reason:
                trades.append(TradeRecord(
                    pair=pair, side=pos.side, action="close",
                    reason=close_reason, bar_idx=t,
                    timestamp=index[t], zscore=z if not np.isnan(z) else 0,
                    pnl=pos.cumulative_pnl,
                ))
                del active_positions[pair]
                n_closes_this_bar += 1
                # 如果配对不再有效，可以释放缓存
                if pair not in valid_pairs:
                    sig_cache.evict(pair)

        # --- 3. 寻找新开仓机会 ---
        if len(active_positions) < config.max_pairs:
            candidates = sorted(valid_pairs.items(), key=lambda x: x[1]["half_life"])

            for pair, info in candidates:
                if len(active_positions) >= config.max_pairs:
                    break
                if pair in active_positions:
                    continue

                z, hr = sig_cache.get_signal(pair, t)

                if np.isnan(z) or np.isnan(hr):
                    continue

                side = 0
                if z > config.entry_threshold:
                    side = -1
                elif z < -config.entry_threshold:
                    side = +1

                if side != 0:
                    # 动态仓位：根据信号强度调整
                    signal_strength = min(abs(z) / config.entry_threshold, 2.0)
                    dynamic_size = config.position_size * signal_strength

                    active_positions[pair] = ActivePosition(
                        pair=pair, side=side, entry_bar=t,
                        entry_zscore=z, hedge_ratio=hr,
                        position_size=dynamic_size,
                    )
                    trades.append(TradeRecord(
                        pair=pair, side=side, action="open",
                        reason="signal", bar_idx=t,
                        timestamp=index[t], zscore=z,
                    ))

        # --- 4. 计算当前bar的PnL ---
        bar_pnl = 0.0
        n_new_trades = 0

        for pair, pos in active_positions.items():
            _, hr = sig_cache.get_signal(pair, t)
            if np.isnan(hr):
                hr = pos.hedge_ratio

            ret_a, ret_b = sig_cache.get_pair_return(pair, t)
            # 使用动态仓位
            pos_size = getattr(pos, 'position_size', config.position_size)
            pair_ret = (ret_a - hr * ret_b) * pos.side * config.leverage * pos_size
            bar_pnl += pair_ret
            pos.cumulative_pnl += pair_ret

            # 月度PnL
            month_key = index[t].strftime("%Y-%m")
            if pair not in pair_monthly_pnl:
                pair_monthly_pnl[pair] = {}
            pair_monthly_pnl[pair][month_key] = (
                pair_monthly_pnl[pair].get(month_key, 0.0) + pair_ret
            )

            if pos.entry_bar == t:
                n_new_trades += 1

        # 交易成本
        bar_fees = (n_new_trades + n_closes_this_bar) * fee_rate * config.position_size
        bar_pnl -= bar_fees
        total_fees += bar_fees

        # Funding rate
        bar_funding = 0.0
        if config.funding_interval_bars > 0 and t % config.funding_interval_bars == 0:
            n_active = len(active_positions)
            if n_active > 0:
                bar_funding = config.funding_rate * 2 * n_active * config.leverage * config.position_size
                bar_pnl -= bar_funding
                total_funding += bar_funding

        total_gross_pnl += bar_pnl + bar_fees + bar_funding

        portfolio_pnl[t] = bar_pnl
        active_count[t] = len(active_positions)
        equity += bar_pnl
        if equity <= 0:
            # 爆仓，提前结束
            print(f"\n    Liquidation at bar {t} ({index[t]})")
            break

    # --- 构建结果 ---
    print()  # Newline after progress
    print("  Building results...")
    pnl_series = pd.Series(portfolio_pnl, index=index)
    equity_curve = 1.0 + pnl_series.cumsum()
    # Floor equity at 0 (realistic: liquidation before negative equity)
    equity_curve = equity_curve.clip(lower=0)

    # 打印手续费分解
    net_pnl = total_gross_pnl - total_fees - total_funding
    print(f"\n  === Cost Breakdown ===")
    print(f"  Gross PnL:      {total_gross_pnl:+.4f} ({total_gross_pnl*100:+.2f}%)")
    print(f"  Trading Fees:   -{total_fees:.4f} (-{total_fees*100:.2f}%)")
    print(f"  Funding Rate:   -{total_funding:.4f} (-{total_funding*100:.2f}%)")
    print(f"  Net PnL:        {net_pnl:+.4f} ({net_pnl*100:+.2f}%)")
    print(f"  ======================\n")
    active_series = pd.Series(active_count, index=index)

    monthly_returns = _build_monthly_table(pnl_series, pair_monthly_pnl, index)

    from analytics.metrics import compute_metrics
    total_metrics = compute_metrics(
        pnl_series, equity_curve,
        pd.Series(active_count, index=index),
        pnl_series.diff().fillna(0),
    )
    total_metrics["total_trades"] = len(trades)
    total_metrics["unique_pairs_traded"] = len(set(t.pair for t in trades))

    return DynamicBacktestResult(
        equity_curve=equity_curve,
        portfolio_pnl=pnl_series,
        monthly_returns=monthly_returns,
        trades=trades,
        active_pairs_count=active_series,
        total_metrics=total_metrics,
    )


def _build_monthly_table(
    pnl_series: pd.Series,
    pair_monthly_pnl: dict,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build monthly returns table with per-pair breakdown."""
    months = sorted(set(index.strftime("%Y-%m")))
    all_pairs = sorted(pair_monthly_pnl.keys())

    data = {}
    for pair in all_pairs:
        col_name = "{}/{}".format(pair[0], pair[1])
        monthly_vals = []
        for month in months:
            val = pair_monthly_pnl[pair].get(month, None)
            monthly_vals.append(val)
        data[col_name] = monthly_vals

    portfolio_monthly = pnl_series.resample("ME").sum()
    data["Portfolio"] = [
        portfolio_monthly.get(pd.Timestamp(m + "-01") + pd.offsets.MonthEnd(0), 0)
        for m in months
    ]

    df = pd.DataFrame(data, index=months)
    df.index.name = "Month"
    return df
