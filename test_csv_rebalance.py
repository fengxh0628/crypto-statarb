#!/usr/bin/env python3
"""Test CSV rebalance frequency to optimize fees."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpha.cross_sectional_vol import CrossSectionalVolAlpha, CrossSectionalVolConfig
from alpha.backtest import run_alpha_backtest


def load_data(symbols):
    from run_alpha import load_all_data
    
    print("[1] Loading data...")
    prices, volumes = load_all_data(symbols=symbols)
    
    print("  Resampling to 1h...")
    prices = prices.resample("1h").last().dropna(how="all")
    if volumes is not None:
        volumes = volumes.resample("1h").sum().reindex(prices.index)
    
    print(f"  Loaded: {len(prices)} bars, {len(prices.columns)} symbols")
    return prices, volumes


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
    
    # Use same period
    prices = prices.loc[prices.index >= '2024-10-01']
    if volumes is not None:
        volumes = volumes.loc[volumes.index >= '2024-10-01']
    
    print(f"  Test period: {prices.index[0]} to {prices.index[-1]} ({len(prices)} bars)")
    
    base_config = {
        'top_k': 5, 'bottom_k': 5,
        'position_size': 0.10, 'leverage': 1.0,
        'taker_fee': 0.0005, 'slippage_bps': 1.0,
        'funding_rate': 0.0001, 'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    results = {}
    
    rebalance_hours = [24, 48, 72, 96, 120]
    
    print("\n" + "="*70)
    print("  CSV REBALANCE FREQUENCY SWEEP")
    print("="*70)
    
    for hours in rebalance_hours:
        print(f"\n  Testing Rebalance: {hours}h")
        
        config = CrossSectionalVolConfig(**{**base_config, 'vol_window': 168, 'rebalance_bars': hours})
        strat = CrossSectionalVolAlpha(config)
        result = run_alpha_backtest(prices, strat, volumes=volumes)
        results[f'CSV ({hours}h)'] = result['metrics']
    
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
