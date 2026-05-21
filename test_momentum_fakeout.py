#!/usr/bin/env python3
"""Test Momentum + Fakeout Ensemble vs Momentum + Squeeze."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpha.momentum_squeeze_ensemble import MomentumSqueezeEnsemble, MomentumSqueezeConfig
from alpha.momentum_fakeout_ensemble import MomentumFakeoutEnsemble, MomentumFakeoutConfig
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
        'top_k': 5, 'bottom_k': 5, 'rebalance_bars': 24,
        'position_size': 0.10, 'leverage': 1.0,
        'taker_fee': 0.0005, 'slippage_bps': 1.0,
        'funding_rate': 0.0001, 'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    # Test 1: Momentum + Squeeze
    print("\n" + "="*70)
    print("  MOMENTUM + SQUEEZE (Baseline)")
    print("="*70)
    
    sq_config = {
        **base_config,
        'squeeze_bb_window': 20, 'squeeze_bb_std': 2.0,
        'squeeze_percentile': 0.1, 'squeeze_lookback_percentile': 336,
        'squeeze_breakout_window': 24, 'squeeze_volume_confirm': True,
        'squeeze_volume_ma_window': 72,
        'momentum_weight': 0.5, 'squeeze_weight': 0.5,
    }
    
    sq_wf = run_walk_forward(
        prices, MomentumSqueezeEnsemble, MomentumSqueezeConfig,
        sq_config, n_splits=5, train_bars=8760, volumes=volumes,
    )
    
    # Test 2: Momentum + Fakeout
    print("\n" + "="*70)
    print("  MOMENTUM + FAKEOUT REVERSAL")
    print("="*70)
    
    fo_config = {
        **base_config,
        'fakeout_lookback': 48,
        'fakeout_confirmation_bars': 1,
        'momentum_weight': 0.5, 'fakeout_weight': 0.5,
    }
    
    fo_wf = run_walk_forward(
        prices, MomentumFakeoutEnsemble, MomentumFakeoutConfig,
        fo_config, n_splits=5, train_bars=8760, volumes=volumes,
    )
    
    # Comparison
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    
    sq_s = sq_wf['summary']
    fo_s = fo_wf['summary']
    
    print(f"  {'Metric':<20} {'Mom+Squeeze':>12} {'Mom+Fakeout':>12} {'Change':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Avg Return':<20} {sq_s['avg_return']:>11.2%} {fo_s['avg_return']:>11.2%} {fo_s['avg_return']-sq_s['avg_return']:>+9.2%}")
    print(f"  {'Consistency':<20} {sq_s['consistency']:>11.0%} {fo_s['consistency']:>11.0%} {fo_s['consistency']-sq_s['consistency']:>+9.0%}")
    print(f"  {'Avg Sharpe':<20} {sq_s['avg_sharpe']:>12.3f} {fo_s['avg_sharpe']:>12.3f} {fo_s['avg_sharpe']-sq_s['avg_sharpe']:>+9.3f}")
    print(f"  {'Std Return':<20} {sq_s['std_return']:>11.2%} {fo_s['std_return']:>11.2%} {fo_s['std_return']-sq_s['std_return']:>+9.2%}")
    print(f"  {'Min Return':<20} {sq_s['min_return']:>11.2%} {fo_s['min_return']:>11.2%} {fo_s['min_return']-sq_s['min_return']:>+9.2%}")
    print(f"  {'Max Return':<20} {sq_s['max_return']:>11.2%} {fo_s['max_return']:>11.2%} {fo_s['max_return']-sq_s['max_return']:>+9.2%}")
    print()
    
    # Per-period
    print("  Per-Period Breakdown:")
    print(f"  {'Split':<8} {'Sq Return':>10} {'Sq Fees':>10} {'Fo Return':>10} {'Fo Fees':>10}")
    print(f"  {'-'*52}")
    
    for i in range(len(sq_wf['results'])):
        sq_r = sq_wf['results'][i]
        fo_r = fo_wf['results'][i]
        print(f"  {i+1:<8} {sq_r['metrics']['total_return']:>9.2%} {sq_r['metrics']['trading_fees']:>9.2%} {fo_r['metrics']['total_return']:>9.2%} {fo_r['metrics']['trading_fees']:>9.2%}")
    
    print()


if __name__ == "__main__":
    main()
