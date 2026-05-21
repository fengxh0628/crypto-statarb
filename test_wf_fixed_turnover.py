#!/usr/bin/env python3
"""Full walk-forward validation with fixed turnover calculation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpha.momentum_squeeze_ensemble import MomentumSqueezeEnsemble, MomentumSqueezeConfig
from alpha.momentum_squeeze_enhanced import EnhancedMomentumSqueeze, EnhancedEnsembleConfig
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
    
    # Test 1: Original Walk-Forward
    print("\n" + "="*70)
    print("  ORIGINAL: Momentum + Squeeze (Fixed Turnover)")
    print("="*70)
    
    orig_wf = run_walk_forward(
        prices,
        MomentumSqueezeEnsemble,
        MomentumSqueezeConfig,
        base_config,
        n_splits=5,
        train_bars=8760,
        volumes=volumes,
    )
    
    # Test 2: Enhanced Walk-Forward
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
    
    enh_wf = run_walk_forward(
        prices,
        EnhancedMomentumSqueeze,
        EnhancedEnsembleConfig,
        enh_config,
        n_splits=5,
        train_bars=8760,
        volumes=volumes,
    )
    
    # Final Comparison
    print("\n" + "="*70)
    print("  WALK-FORWARD COMPARISON SUMMARY")
    print("="*70)
    
    orig_s = orig_wf['summary']
    enh_s = enh_wf['summary']
    
    print(f"  {'Metric':<20} {'Original':>12} {'Enhanced':>12} {'Change':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Avg Return':<20} {orig_s['avg_return']:>11.2%} {enh_s['avg_return']:>11.2%} {enh_s['avg_return']-orig_s['avg_return']:>+9.2%}")
    print(f"  {'Consistency':<20} {orig_s['consistency']:>11.0%} {enh_s['consistency']:>11.0%} {enh_s['consistency']-orig_s['consistency']:>+9.0%}")
    print(f"  {'Avg Sharpe':<20} {orig_s['avg_sharpe']:>12.3f} {enh_s['avg_sharpe']:>12.3f} {enh_s['avg_sharpe']-orig_s['avg_sharpe']:>+9.3f}")
    print(f"  {'Std Return':<20} {orig_s['std_return']:>11.2%} {enh_s['std_return']:>11.2%} {enh_s['std_return']-orig_s['std_return']:>+9.2%}")
    print(f"  {'Min Return':<20} {orig_s['min_return']:>11.2%} {enh_s['min_return']:>11.2%} {enh_s['min_return']-orig_s['min_return']:>+9.2%}")
    print(f"  {'Max Return':<20} {orig_s['max_return']:>11.2%} {enh_s['max_return']:>11.2%} {enh_s['max_return']-orig_s['max_return']:>+9.2%}")
    print()
    
    # Per-period breakdown
    print("  Per-Period Breakdown:")
    print(f"  {'Split':<8} {'Orig Return':>12} {'Orig Fees':>12} {'Enh Return':>12} {'Enh Fees':>12}")
    print(f"  {'-'*60}")
    
    for i in range(len(orig_wf['results'])):
        orig_r = orig_wf['results'][i]
        enh_r = enh_wf['results'][i]
        print(f"  {i+1:<8} {orig_r['metrics']['total_return']:>11.2%} {orig_r['metrics']['trading_fees']:>11.2%} {enh_r['metrics']['total_return']:>11.2%} {enh_r['metrics']['trading_fees']:>11.2%}")
    
    print()


if __name__ == "__main__":
    main()
