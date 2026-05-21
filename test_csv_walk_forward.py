#!/usr/bin/env python3
"""Walk-forward validation for Cross-Sectional Volatility (120h rebalance)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpha.cross_sectional_vol import CrossSectionalVolAlpha, CrossSectionalVolConfig
from alpha.walk_forward import run_walk_forward


def load_data(symbols):
    from run_alpha import load_all_data
    
    print("[1] Loading data...")
    prices, volumes = load_all_data(symbols=symbols)
    
    print("  Resampling to 1h...")
    prices = prices.resample("1h").last().dropna(how="all")
    if volumes is not None:
        volumes = volumes.resample("1h").sum().reindex(prices.index)
    
    print(f"  Loaded: {len(prices)} bars, {len(prices.columns)} symbols")
    print(f"  Period: {prices.index[0]} to {prices.index[-1]}")
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
    
    base_config = {
        'vol_window': 168,
        'top_k': 5, 'bottom_k': 5, 
        'rebalance_bars': 120,  # 5 days
        'position_size': 0.10, 'leverage': 1.0,
        'taker_fee': 0.0005, 'slippage_bps': 1.0,
        'funding_rate': 0.0001, 'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    print("\n" + "="*70)
    print("  WALK-FORWARD: Cross-Sectional Volatility (120h Rebalance)")
    print("="*70)
    
    wf_result = run_walk_forward(
        prices,
        CrossSectionalVolAlpha,
        CrossSectionalVolConfig,
        base_config,
        n_splits=5,
        train_bars=8760,
        volumes=volumes,
    )
    
    # Per-period breakdown
    print("\n  === Per-Period Breakdown ===")
    print(f"  {'Split':<8} {'Net Return':>12} {'Gross':>12} {'Fees':>12}")
    print(f"  {'-'*48}")
    
    for i, r in enumerate(wf_result['results']):
        m = r['metrics']
        print(f"  {i+1:<8} {m['net_pnl']:>11.2%} {m['gross_pnl']:>11.2%} {m['trading_fees']:>11.2%}")
    
    print()


if __name__ == "__main__":
    main()
