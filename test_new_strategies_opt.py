#!/usr/bin/env python3
"""Test optimized new strategies: CorrMR, CrossSecVol, Basis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from alpha.correlation_mr import CorrMRAlpha, CorrMRConfig
from alpha.cross_sectional_vol import CrossSectionalVolAlpha, CrossSectionalVolConfig
from alpha.basis_reversion import BasisAlpha, BasisConfig
from alpha.backtest import run_alpha_backtest


def load_data(symbols):
    from run_alpha import load_all_data
    
    print("[1] Loading price data...")
    prices, volumes = load_all_data(symbols=symbols)
    
    print("  Resampling to 1h...")
    prices = prices.resample("1h").last().dropna(how="all")
    if volumes is not None:
        volumes = volumes.resample("1h").sum().reindex(prices.index)
    
    print(f"  Loaded: {len(prices)} bars, {len(prices.columns)} symbols")
    return prices, volumes


def load_mark_prices(symbols):
    """Load Mark Price data from premiumIndexKlines."""
    from pathlib import Path
    
    data_path = Path("~/binance_data/futures/um/monthly/premiumIndexKlines").expanduser()
    
    mark_dfs = {}
    total = len(symbols)
    
    print(f"\n[2] Loading Mark Price data for {total} symbols...")
    
    for i, symbol in enumerate(symbols, 1):
        symbol_dir = data_path / symbol / "8h"
        if not symbol_dir.exists():
            continue
        
        csv_files = sorted(symbol_dir.glob(f"{symbol}-8h-premiumIndexKlines-*.csv"))
        if not csv_files:
            continue
        
        try:
            print(f"  [{i}/{total}] Loading {symbol} mark price...", end="\r")
            
            all_times = []
            all_closes = []
            
            for f in csv_files:
                df = pd.read_csv(f, usecols=[0, 4], names=["open_time", "close"], header=None, dtype={"open_time": "int64"})
                all_times.append(df["open_time"].values)
                all_closes.append(df["close"].values)
            
            if not all_times:
                continue
            
            times = np.concatenate(all_times)
            closes = np.concatenate(all_closes)
            
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            closes = closes[sort_idx]
            
            idx = pd.to_datetime(times, unit="ms")
            mark_dfs[symbol] = pd.DataFrame({"mark": closes}, index=idx)
        
        except Exception as e:
            pass
    
    print()
    
    if not mark_dfs:
        print("  No Mark Price data found.")
        return None
    
    print(f"  Loaded {len(mark_dfs)} symbols Mark Price data")
    
    # Align to 1h (forward fill from 8h)
    print("  Aligning Mark Prices to 1h...")
    freq = "1h"
    aligned = {}
    for sym, df in mark_dfs.items():
        resampled = df.resample(freq).last().ffill()
        if len(resampled) > 100:
            aligned[sym] = resampled["mark"]
    
    mark_df = pd.DataFrame(aligned)
    mark_df = mark_df.dropna(how="all")
    
    print(f"  Mark Price data: {len(mark_df)} bars, {len(mark_df.columns)} symbols")
    
    return mark_df


def main():
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
    
    prices, volumes = load_data(symbols)
    mark_df = load_mark_prices(symbols)
    
    # Use last 1.5 years for faster testing
    start_date = '2024-10-01'
    prices = prices.loc[prices.index >= start_date]
    if volumes is not None:
        volumes = volumes.loc[volumes.index >= start_date]
    if mark_df is not None:
        mark_df = mark_df.loc[mark_df.index >= start_date]
    
    print(f"\n  Test period: {prices.index[0]} to {prices.index[-1]} ({len(prices)} bars)")
    
    base_config = {
        'top_k': 5, 'bottom_k': 5, 'rebalance_bars': 24,
        'position_size': 0.10, 'leverage': 1.0,
        'taker_fee': 0.0005, 'slippage_bps': 1.0,
        'funding_rate': 0.0001, 'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    results = {}
    
    # Test 1: Correlation MR (Optimized: No threshold)
    print("\n" + "="*70)
    print("  CORRELATION MEAN REVERSION (Optimized: No Threshold)")
    print("="*70)
    
    corr_config = CorrMRConfig(**{**base_config, 'corr_window': 168, 'corr_threshold': 0.6, 'spread_window': 72})
    corr_strat = CorrMRAlpha(corr_config)
    corr_result = run_alpha_backtest(prices, corr_strat, volumes=volumes)
    results['CorrMR'] = corr_result['metrics']
    
    # Test 2: Cross-Sectional Vol (Optimized: Rebalance 48h)
    print("\n" + "="*70)
    print("  CROSS-SECTIONAL VOLATILITY (Optimized: Rebalance 48h)")
    print("="*70)
    
    csv_config = CrossSectionalVolConfig(**{**base_config, 'vol_window': 168, 'rebalance_bars': 48})
    csv_strat = CrossSectionalVolAlpha(csv_config)
    csv_result = run_alpha_backtest(prices, csv_strat, volumes=volumes)
    results['CrossSecVol'] = csv_result['metrics']
    
    # Test 3: Basis Reversion (Mark Price Premium)
    if mark_df is not None:
        print("\n" + "="*70)
        print("  BASIS REVERSION (Mark Price Premium)")
        print("="*70)
        
        basis_config = BasisConfig(**{**base_config, 'basis_window': 72})
        basis_strat = BasisAlpha(basis_config)
        basis_strat.set_mark_prices(mark_df)
        basis_result = run_alpha_backtest(prices, basis_strat, volumes=volumes)
        results['Basis'] = basis_result['metrics']
    else:
        print("\n  Skipping Basis (no Mark Price data)")
    
    # Comparison
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    
    print(f"  {'Strategy':<20} {'Return':>10} {'Gross':>10} {'Fees':>10} {'Net':>10} {'Sharpe':>8}")
    print(f"  {'-'*72}")
    
    for name, m in results.items():
        print(f"  {name:<20} {m['total_return']:>9.2%} {m['gross_pnl']:>9.2%} {m['trading_fees']:>9.2%} {m['net_pnl']:>9.2%} {m['sharpe_ratio']:>8.3f}")
    
    print()


if __name__ == "__main__":
    main()
