#!/usr/bin/env python3
"""Test enhanced ensemble with hysteresis and volatility targeting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from alpha.momentum_squeeze_ensemble import MomentumSqueezeEnsemble, MomentumSqueezeConfig
from alpha.momentum_squeeze_enhanced import EnhancedMomentumSqueeze, EnhancedEnsembleConfig
from alpha.backtest import run_alpha_backtest
from alpha.walk_forward import run_walk_forward


def load_data(symbols):
    """Load and preprocess data."""
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


def run_parameter_sweep(prices, volumes, symbols):
    """Sweep hysteresis and vol targeting parameters."""
    
    base_config = {
        'momentum_window': 336,
        'momentum_skip': 12,
        'momentum_trend_window': 336,
        'momentum_trend_threshold': 0.0,
        'momentum_vol_window': 720,
        'momentum_vol_threshold_high': 0.05,
        'momentum_vol_threshold_low': 0.01,
        'momentum_mom_window': 720,
        'momentum_mom_threshold': 0.0,
        'momentum_bull_multiplier': 1.0,
        'momentum_bear_multiplier': 0.0,
        'momentum_range_multiplier': 0.75,
        'momentum_flat_in_bear': True,
        'squeeze_bb_window': 20,
        'squeeze_bb_std': 2.0,
        'squeeze_percentile': 0.1,
        'squeeze_lookback_percentile': 336,
        'squeeze_breakout_window': 24,
        'squeeze_volume_confirm': True,
        'squeeze_volume_ma_window': 72,
        'momentum_weight': 0.5,
        'squeeze_weight': 0.5,
        'top_k': 5,
        'bottom_k': 5,
        'rebalance_bars': 24,
        'position_size': 0.10,
        'leverage': 1.0,
        'taker_fee': 0.0005,
        'slippage_bps': 1.0,
        'funding_rate': 0.0001,
        'funding_interval_bars': 24,
        'symbols': symbols,
    }
    
    # Parameter grid
    hysteresis_params = [
        (0.15, 0.05),  # Low thresholds
        (0.10, 0.03),  # Very low thresholds
        (0.20, 0.08),  # Medium thresholds
    ]
    
    vol_params = [
        (0.20, 0.3, 2.0),  # Higher vol target, wider scaling
        (0.25, 0.5, 1.5),  # Moderate vol target
        (0.30, 0.5, 2.0),  # High vol target, wide scaling
    ]
    
    print("\n" + "="*90)
    print("  PARAMETER SWEEP: Hysteresis + Vol Targeting")
    print("="*90)
    print(f"  {'Entry':>6} {'Exit':>5} {'VolTgt':>7} {'VolMin':>7} {'VolMax':>7} | {'AvgRet':>8} {'Consist':>8} {'Sharpe':>7}")
    print(f"  {'-'*90}")
    
    best_result = None
    best_config = None
    best_sharpe = -999
    
    for entry_thresh, exit_thresh in hysteresis_params:
        for vol_target, vol_min, vol_max in vol_params:
            config = {
                **base_config,
                'hysteresis_entry_threshold': entry_thresh,
                'hysteresis_exit_threshold': exit_thresh,
                'vol_target': vol_target,
                'vol_lookback': 720,
                'vol_scaling_min': vol_min,
                'vol_scaling_max': vol_max,
            }
            
            # Run single split for quick evaluation
            test_prices = prices.iloc[-4000:]  # Last ~6 months
            test_volumes = volumes.iloc[-4000:] if volumes is not None else None
            
            strat = EnhancedMomentumSqueeze(EnhancedEnsembleConfig(**config))
            from alpha.backtest import run_alpha_backtest
            result = run_alpha_backtest(test_prices, strat, volumes=test_volumes)
            
            metrics = result['metrics']
            avg_ret = metrics['total_return']
            sharpe = metrics['sharpe_ratio']
            fees = metrics['trading_fees']
            
            print(f"  {entry_thresh:>6.2f} {exit_thresh:>5.2f} {vol_target:>7.2f} {vol_min:>7.2f} {vol_max:>7.2f} | {avg_ret:>7.2%} {'N/A':>8} {sharpe:>7.3f} (Fees: {fees:.2%})")
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = config
                best_result = result
    
    print()
    return best_config, best_result


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
    
    # Use full data for parameter sweep
    run_parameter_sweep(prices, volumes, symbols)


if __name__ == "__main__":
    main()
