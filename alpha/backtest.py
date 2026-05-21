"""Alpha backtest engine.

Runs alpha strategy backtest with:
- Benchmark comparison
- Factor exposure analysis
- Transaction cost modeling
"""

from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


def run_alpha_backtest(
    prices: pd.DataFrame,
    strategy: AlphaStrategy,
    volumes: Optional[pd.DataFrame] = None,
    benchmark_symbols: list = None,
) -> dict:
    """Run alpha strategy backtest.
    
    Args:
        prices: DataFrame of close prices
        strategy: AlphaStrategy instance
        volumes: DataFrame of volumes (optional)
        benchmark_symbols: Symbols for benchmark (default: ['BTCUSDT', 'ETHUSDT'])
        
    Returns:
        Dict with:
        - equity_curve: Strategy equity curve
        - benchmark_curve: Benchmark equity curve
        - weights_history: Historical portfolio weights
        - metrics: Performance metrics
        - factor_exposures: Factor exposure analysis
    """
    if benchmark_symbols is None:
        benchmark_symbols = ['BTCUSDT', 'ETHUSDT']
    
    n_bars = len(prices)
    index = prices.index
    
    # State
    positions = {}  # {symbol: weight}
    portfolio_pnl = np.zeros(n_bars)
    equity_curve = np.ones(n_bars)
    weights_history = []
    factor_history = []  # Store factor snapshots
    
    total_fees = 0.0
    total_funding = 0.0
    total_gross_pnl = 0.0
    
    fee_rate = (strategy.config.taker_fee + strategy.config.slippage_bps / 10000.0) * 2 * strategy.config.leverage
    
    # Benchmark
    valid_bench = [s for s in benchmark_symbols if s in prices.columns]
    if valid_bench:
        bench_prices = prices[valid_bench]
        bench_returns = np.log(bench_prices).diff().mean(axis=1).fillna(0)
        bench_equity = np.exp(np.cumsum(bench_returns))
        bench_equity = bench_equity / bench_equity.iloc[0]  # Normalize to 1
    else:
        bench_equity = pd.Series(1.0, index=index)
    
    print(f"  Running alpha backtest: {n_bars} bars, {len(prices.columns)} symbols")
    last_progress = 0
    
    warmup = strategy.config.momentum_window if hasattr(strategy.config, 'momentum_window') else 100
    
    for t in range(warmup, n_bars):
        # Progress
        progress = (t - warmup) * 100 // (n_bars - warmup)
        if progress >= last_progress + 10:
            print(f"    {progress}% ({t}/{n_bars})", end="\r")
            last_progress = progress
        
        bar_fees = 0.0
        
        # Rebalance
        if (t - warmup) % strategy.config.rebalance_bars == 0:
            window_prices = prices.iloc[:t]
            window_volumes = volumes.iloc[:t] if volumes is not None else None
            
            new_weights = strategy.generate_weights(window_prices, window_volumes)
            
            # Calculate turnover based on actual weight changes
            all_syms = set(positions.keys()) | set(new_weights.keys())
            turnover = sum(abs(new_weights.get(s, 0.0) - positions.get(s, 0.0)) for s in all_syms)
            bar_fees = turnover * fee_rate
            total_fees += bar_fees
            
            positions = new_weights
        
        # Store weights
        weights_history.append({
            'datetime': index[t],
            'weights': positions.copy(),
            'n_long': sum(1 for w in positions.values() if w > 0),
            'n_short': sum(1 for w in positions.values() if w < 0),
        })
        
        # Store factor snapshots at rebalance
        if (t - warmup) % strategy.config.rebalance_bars == 0:
            # Compute factors for all symbols
            factor_snapshot = {'datetime': index[t]}
            
            # Momentum factor
            if hasattr(strategy.config, 'momentum_window'):
                mom_win = strategy.config.momentum_window
                skip = strategy.config.skip_recent
                if t >= mom_win + skip:
                    log_p = np.log(prices.values)
                    mom_scores = log_p[t - skip] - log_p[t - mom_win - skip]
                    for j, sym in enumerate(prices.columns):
                        factor_snapshot[f'momentum_{sym}'] = mom_scores[j]
            
            # Volatility factor (30-day)
            vol_window = min(720, t)  # 30 days of 1h
            if t >= vol_window:
                returns = np.log(prices.iloc[t-vol_window:t]).diff().dropna(axis=0, how='all')
                vol = returns.std()
                for sym in prices.columns:
                    factor_snapshot[f'volatility_{sym}'] = vol.get(sym, np.nan)
            
            # Volume factor (7-day avg)
            if volumes is not None:
                vol_avg_window = min(168, t)  # 7 days
                if t >= vol_avg_window:
                    avg_vol = volumes.iloc[t-vol_avg_window:t].mean()
                    for sym in prices.columns:
                        factor_snapshot[f'volume_{sym}'] = avg_vol.get(sym, np.nan)
            
            factor_history.append(factor_snapshot)
        
        # Calculate PnL
        bar_pnl = 0.0
        for sym, weight in positions.items():
            if sym in prices.columns:
                ret = np.log(prices.values[t, prices.columns.get_loc(sym)] / 
                           prices.values[t-1, prices.columns.get_loc(sym)])
                if not np.isnan(ret):
                    bar_pnl += ret * weight
        
        # Funding rate
        bar_funding = 0.0
        if strategy.config.funding_interval_bars > 0 and t % strategy.config.funding_interval_bars == 0:
            for sym, weight in positions.items():
                if weight < 0:
                    bar_funding += strategy.config.funding_rate * abs(weight)
                else:
                    bar_funding -= strategy.config.funding_rate * weight
            total_funding += bar_funding
        
        bar_pnl -= bar_funding
        bar_pnl -= bar_fees  # Deduct fees from daily PnL
        total_gross_pnl += bar_pnl
        
        portfolio_pnl[t] = bar_pnl
        equity_curve[t] = equity_curve[t-1] + bar_pnl
        
        if equity_curve[t] <= 0:
            print(f"\n    Liquidation at bar {t} ({index[t]})")
            equity_curve[t:] = 0
            break
    
    print()  # newline
    
    # Build results
    equity_series = pd.Series(equity_curve, index=index)
    pnl_series = pd.Series(portfolio_pnl, index=index)
    weights_df = pd.DataFrame(weights_history).set_index('datetime')
    factors_df = pd.DataFrame(factor_history).set_index('datetime') if factor_history else pd.DataFrame()
    
    # Metrics
    total_return = equity_series.iloc[-1] - 1.0
    bench_return = bench_equity.iloc[-1] - 1.0
    
    # Drawdown
    cummax = equity_series.cummax()
    drawdown = equity_series / cummax - 1
    max_drawdown = drawdown.min()
    
    # Volatility and Sharpe
    daily_pnl = pnl_series.resample('1D').sum()
    ann_vol = daily_pnl.std() * np.sqrt(365)
    ann_return = (1 + total_return) ** (365 / max((equity_series.index[-1] - equity_series.index[0]).days, 1)) - 1
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Alpha vs benchmark
    alpha = total_return - bench_return
    
    metrics = {
        'total_return': total_return,
        'benchmark_return': bench_return,
        'alpha': alpha,
        'gross_pnl': total_gross_pnl,
        'trading_fees': total_fees,
        'funding_cost': total_funding,
        'net_pnl': total_gross_pnl - total_fees - total_funding,
        'max_drawdown': max_drawdown,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'n_bars': n_bars,
    }
    
    # Factor exposure analysis
    factor_exposures = analyze_factor_exposures(prices, weights_df, equity_series)
    
    # Print cost breakdown
    print(f"\n  === Cost Breakdown ===")
    print(f"  Gross PnL:      {total_gross_pnl:+.4f} ({total_gross_pnl*100:+.2f}%)")
    print(f"  Trading Fees:   -{total_fees:.4f} (-{total_fees*100:.2f}%)")
    print(f"  Funding Rate:   -{total_funding:.4f} (-{total_funding*100:.2f}%)")
    print(f"  Net PnL:        {total_gross_pnl - total_fees - total_funding:+.4f} ({(total_gross_pnl - total_fees - total_funding)*100:+.2f}%)")
    print(f"  ======================\n")
    
    return {
        'equity_curve': equity_series,
        'benchmark_curve': bench_equity,
        'pnl_series': pnl_series,
        'weights_history': weights_df,
        'factors': factors_df,
        'metrics': metrics,
        'factor_exposures': factor_exposures,
    }


def analyze_factor_exposures(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    equity_curve: pd.Series,
) -> dict:
    """Analyze factor exposures of the strategy.
    
    Factors:
    - Market beta (vs BTC)
    - Size (market cap proxy: volume)
    - Momentum (past return)
    - Volatility
    """
    exposures = {}
    
    # Market beta (vs BTC)
    if 'BTCUSDT' in prices.columns:
        btc_returns = np.log(prices['BTCUSDT']).diff().dropna()
        strat_returns = equity_curve.pct_change().dropna()
        
        # Align
        common_idx = btc_returns.index.intersection(strat_returns.index)
        if len(common_idx) > 100:
            btc_aligned = btc_returns.loc[common_idx]
            strat_aligned = strat_returns.loc[common_idx]
            
            # Regression
            cov = np.cov(btc_aligned.values, strat_aligned.values)
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0
            exposures['market_beta'] = beta
    
    # Size factor (volume-weighted)
    # High volume vs low volume performance
    # Simplified: just report average volume of held positions
    
    # Momentum factor
    # Check if strategy is actually capturing momentum
    # Compare returns of high-score vs low-score symbols
    
    # Volatility factor
    # Check if strategy is exposed to high/low vol symbols
    
    return exposures
