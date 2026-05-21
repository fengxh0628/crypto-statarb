#!/usr/bin/env python3
"""Compare original vs enhanced ensemble with fixed turnover calculation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpha.momentum_squeeze_ensemble import MomentumSqueezeEnsemble, MomentumSqueezeConfig
from alpha.momentum_squeeze_enhanced import EnhancedMomentumSqueeze, EnhancedEnsembleConfig
from alpha.backtest import run_alpha_backtest
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
    
    # Use 1 year for faster testing
    prices = prices.loc[prices.index >= '2025-01-01']
    if volumes is not None:
        volumes = volumes.loc[volumes.index >= '2025-01-01']
    
    print(f"  Test period: {prices.index[0]} to {prices.index[-1]} ({len(prices)} bars)")
    
    base_config = {
        'momentum_window': 336, 'momentum_skip': 12,
        'momentum_trend_window': 336, 'momentum_trend_threshold': 0.0,
        'momentum_vol_window': 720, 'momentum_vol_threshold_high': 0.05,
        'momentum_vol_threshold_low': 0.01, 'momentum_mom_window': 720,
        'momentum_mom_threshold': 0.0, 'momentum_bull_multiplier': 1.0,
        'momentum_bear_multiplier': 0.0, 'momentum_range_multiplier': 0.75,
        'momentum_flat_in_bear': True,
        'squeeze_bb_window': 20, 'squeeze_bb_std': 2.0,
        'squeeze_percentile': 0.1, 'squeeze_lookback_percentile': 336,
        'squeeze_breakout_window': 24, 'squeeze_volume_confirm': True,
        'squeeze_volume_ma_window': 72,
        'momentum_weight': 0.5, 'squeeze_weight': 0.5,
        'top_k': 5, 'bottom_k': 5, 'rebalance_bars': 24,
        'position_size': 0.10, 'leverage': 1.0,
        'taker_fee': 0.0005, 'slippage_bps': 1.0,
        'funding_rate': 0.0001, 'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    # Test 1: Original
    print("\n" + "="*70)
    print("  ORIGINAL: Momentum + Squeeze (Fixed Turnover)")
    print("="*70)
    
    orig_strat = MomentumSqueezeEnsemble(MomentumSqueezeConfig(**base_config))
    orig_result = run_alpha_backtest(prices, orig_strat, volumes=volumes)
    
    # Test 2: Enhanced
    print("\n" + "="*70)
    print("  ENHANCED: + Hysteresis (0.15/0.05) + Vol Target (20%, 0.5-2.0x)")
    print("="*70)
    
    enh_config = {
        **base_config,
        'hysteresis_entry_threshold': 0.15,
        'hysteresis_exit_threshold': 0.05,
        'vol_target': 0.20,
        'vol_lookback': 720,
        'vol_scaling_min': 0.5,
        'vol_scaling_max': 2.0,
    }
    
    enh_strat = EnhancedMomentumSqueeze(EnhancedEnsembleConfig(**enh_config))
    enh_result = run_alpha_backtest(prices, enh_strat, volumes=volumes)
    
    # Comparison
    print("\n" + "="*70)
    print("  COMPARISON")
    print("="*70)
    
    orig_m = orig_result['metrics']
    enh_m = enh_result['metrics']
    
    print(f"  {'Metric':<20} {'Original':>12} {'Enhanced':>12} {'Change':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Total Return':<20} {orig_m['total_return']:>11.2%} {enh_m['total_return']:>11.2%} {enh_m['total_return']-orig_m['total_return']:>+9.2%}")
    print(f"  {'Gross PnL':<20} {orig_m['gross_pnl']:>11.2%} {enh_m['gross_pnl']:>11.2%} {enh_m['gross_pnl']-orig_m['gross_pnl']:>+9.2%}")
    print(f"  {'Trading Fees':<20} {orig_m['trading_fees']:>11.2%} {enh_m['trading_fees']:>11.2%} {enh_m['trading_fees']-orig_m['trading_fees']:>+9.2%}")
    print(f"  {'Net PnL':<20} {orig_m['net_pnl']:>11.2%} {enh_m['net_pnl']:>11.2%} {enh_m['net_pnl']-orig_m['net_pnl']:>+9.2%}")
    print(f"  {'Sharpe':<20} {orig_m['sharpe_ratio']:>12.3f} {enh_m['sharpe_ratio']:>12.3f} {enh_m['sharpe_ratio']-orig_m['sharpe_ratio']:>+9.3f}")
    print(f"  {'Max Drawdown':<20} {orig_m['max_drawdown']:>11.2%} {enh_m['max_drawdown']:>11.2%} {enh_m['max_drawdown']-orig_m['max_drawdown']:>+9.2%}")


if __name__ == "__main__":
    main()
