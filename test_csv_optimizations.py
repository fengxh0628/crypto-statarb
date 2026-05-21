#!/usr/bin/env python3
"""Test CSV optimizations and CSV+Momentum ensemble."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpha.cross_sectional_vol import CrossSectionalVolAlpha, CrossSectionalVolConfig
from alpha.csv_enhanced import CSVEnhancedAlpha, CSVEnhancedConfig
from alpha.csv_momentum_ensemble import CSVMomentumEnsemble, CSVMomentumConfig
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
    
    # Use same period as previous CSV test
    prices = prices.loc[prices.index >= '2024-10-01']
    if volumes is not None:
        volumes = volumes.loc[volumes.index >= '2024-10-01']
    
    print(f"  Test period: {prices.index[0]} to {prices.index[-1]} ({len(prices)} bars)")
    
    base_config = {
        'top_k': 5, 'bottom_k': 5, 'rebalance_bars': 24,
        'position_size': 0.10, 'leverage': 1.0,
        'taker_fee': 0.0005, 'slippage_bps': 1.0,
        'funding_rate': 0.0001, 'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    results = {}
    
    # Test 1: CSV Baseline
    print("\n" + "="*70)
    print("  CSV BASELINE (Rebalance 48h)")
    print("="*70)
    
    csv_config = CrossSectionalVolConfig(**{**base_config, 'vol_window': 168, 'rebalance_bars': 48})
    csv_strat = CrossSectionalVolAlpha(csv_config)
    csv_result = run_alpha_backtest(prices, csv_strat, volumes=volumes)
    results['CSV (Base)'] = csv_result['metrics']
    
    # Test 2: CSV + Hysteresis
    print("\n" + "="*70)
    print("  CSV + HYSTERESIS (Entry 0.3 / Exit 0.1)")
    print("="*70)
    
    csv_h_config = CSVEnhancedConfig(**{**base_config, 'vol_window': 168, 'rebalance_bars': 48,
                                        'hysteresis_entry_threshold': 0.3, 'hysteresis_exit_threshold': 0.1})
    csv_h_strat = CSVEnhancedAlpha(csv_h_config)
    csv_h_result = run_alpha_backtest(prices, csv_h_strat, volumes=volumes)
    results['CSV + Hyst'] = csv_h_result['metrics']
    
    # Test 3: CSV + Momentum
    print("\n" + "="*70)
    print("  CSV + MOMENTUM ENSEMBLE (50/50)")
    print("="*70)
    
    ensemble_config = CSVMomentumConfig(**{
        **base_config,
        'momentum_window': 336, 'momentum_skip': 12,
        'momentum_trend_window': 336, 'momentum_trend_threshold': 0.0,
        'momentum_vol_window': 720, 'momentum_vol_threshold_high': 0.05,
        'momentum_vol_threshold_low': 0.01, 'momentum_mom_window': 720,
        'momentum_mom_threshold': 0.0, 'momentum_bull_multiplier': 1.0,
        'momentum_bear_multiplier': 0.0, 'momentum_range_multiplier': 0.75,
        'momentum_flat_in_bear': True,
        'csv_vol_window': 168,
        'momentum_weight': 0.5, 'csv_weight': 0.5,
        'rebalance_bars': 24,  # Momentum needs 24h
    })
    
    ensemble_strat = CSVMomentumEnsemble(ensemble_config)
    ens_result = run_alpha_backtest(prices, ensemble_strat, volumes=volumes)
    results['CSV+Mom'] = ens_result['metrics']
    
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
