#!/usr/bin/env python3
"""Test three new strategies: Correlation MR, Cross-Sectional Vol, Funding Rate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from alpha.correlation_mr import CorrMRAlpha, CorrMRConfig
from alpha.cross_sectional_vol import CrossSectionalVolAlpha, CrossSectionalVolConfig
from alpha.funding_rate import FundingRateAlpha, FundingRateConfig
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


def load_funding_data(symbols):
    """Load funding rate data from premiumIndexKlines."""
    from pathlib import Path
    
    data_path = Path("~/binance_data/futures/um/monthly/premiumIndexKlines").expanduser()
    
    funding_dfs = {}
    total = len(symbols)
    
    print(f"\n[2] Loading funding rate data for {total} symbols...")
    
    for i, symbol in enumerate(symbols, 1):
        symbol_dir = data_path / symbol / "8h"
        if not symbol_dir.exists():
            continue
        
        csv_files = sorted(symbol_dir.glob(f"{symbol}-8h-premiumIndexKlines-*.csv"))
        if not csv_files:
            continue
        
        try:
            print(f"  [{i}/{total}] Loading {symbol} funding...", end="\r")
            
            all_times = []
            all_marks = []
            
            for f in csv_files:
                df = pd.read_csv(f, usecols=[0, 4], names=["open_time", "mark_price"], header=None, dtype={"open_time": "int64"})
                all_times.append(df["open_time"].values)
                all_marks.append(df["mark_price"].values)
            
            if not all_times:
                continue
            
            times = np.concatenate(all_times)
            marks = np.concatenate(all_marks)
            
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            marks = marks[sort_idx]
            
            idx = pd.to_datetime(times, unit="ms")
            funding_dfs[symbol] = pd.DataFrame({"funding": marks}, index=idx)
        
        except Exception as e:
            print(f"    Warning: failed to load {symbol} funding: {e}")
    
    print()
    
    if not funding_dfs:
        print("  No funding rate data found.")
        return None
    
    print(f"  Loaded {len(funding_dfs)} symbols funding data")
    
    # Align to 1h (forward fill from 8h)
    print("  Aligning funding rates to 1h...")
    freq = "1h"
    aligned = {}
    for sym, df in funding_dfs.items():
        resampled = df.resample(freq).last().ffill()
        if len(resampled) > 100:
            aligned[sym] = resampled["funding"]
    
    funding_df = pd.DataFrame(aligned)
    funding_df = funding_df.dropna(how="all")
    
    print(f"  Funding data: {len(funding_df)} bars, {len(funding_df.columns)} symbols")
    
    return funding_df


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
    funding_df = load_funding_data(symbols)
    
    # Use last 2 years for faster testing
    prices = prices.loc[prices.index >= '2024-01-01']
    if volumes is not None:
        volumes = volumes.loc[volumes.index >= '2024-01-01']
    if funding_df is not None:
        funding_df = funding_df.loc[funding_df.index >= '2024-01-01']
    
    print(f"\n  Test period: {prices.index[0]} to {prices.index[-1]} ({len(prices)} bars)")
    
    base_config = {
        'top_k': 5, 'bottom_k': 5, 'rebalance_bars': 24,
        'position_size': 0.10, 'leverage': 1.0,
        'taker_fee': 0.0005, 'slippage_bps': 1.0,
        'funding_rate': 0.0001, 'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    results = {}
    
    # Test 1: Correlation MR
    print("\n" + "="*70)
    print("  CORRELATION MEAN REVERSION")
    print("="*70)
    
    corr_config = CorrMRConfig(**{**base_config, 'corr_window': 168, 'corr_threshold': 0.7, 'spread_window': 72, 'entry_zscore': 2.0})
    corr_strat = CorrMRAlpha(corr_config)
    corr_result = run_alpha_backtest(prices, corr_strat, volumes=volumes)
    results['CorrMR'] = corr_result['metrics']
    
    # Test 2: Cross-Sectional Vol
    print("\n" + "="*70)
    print("  CROSS-SECTIONAL VOLATILITY")
    print("="*70)
    
    csv_config = CrossSectionalVolConfig(**{**base_config, 'vol_window': 168})
    csv_strat = CrossSectionalVolAlpha(csv_config)
    csv_result = run_alpha_backtest(prices, csv_strat, volumes=volumes)
    results['CrossSecVol'] = csv_result['metrics']
    
    # Test 3: Funding Rate (if data available)
    if funding_df is not None:
        print("\n" + "="*70)
        print("  FUNDING RATE ALPHA")
        print("="*70)
        
        fr_config = FundingRateConfig(**{**base_config, 'funding_window': 72, 'entry_threshold': 1.5})
        fr_strat = FundingRateAlpha(fr_config)
        fr_strat.set_funding_data(funding_df)
        fr_result = run_alpha_backtest(prices, fr_strat, volumes=volumes)
        results['FundingRate'] = fr_result['metrics']
    else:
        print("\n  Skipping Funding Rate (no data)")
    
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
