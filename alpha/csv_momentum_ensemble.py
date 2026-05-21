"""CSV + Momentum Ensemble.

Combines Cross-Sectional Volatility (Mean Reversion) and Momentum.
- Allocates 50% to Momentum (Trend Following)
- Allocates 50% to CSV (Volatility Management)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig
from alpha.timed_momentum import TimedMomentumAlpha, TimedMomentumConfig
from alpha.cross_sectional_vol import CrossSectionalVolAlpha, CrossSectionalVolConfig


@dataclass
class CSVMomentumConfig(AlphaConfig):
    """Configuration for CSV + Momentum Ensemble."""
    # Momentum params
    momentum_window: int = 336
    momentum_skip: int = 12
    momentum_trend_window: int = 336
    momentum_trend_threshold: float = 0.0
    momentum_vol_window: int = 720
    momentum_vol_threshold_high: float = 0.05
    momentum_vol_threshold_low: float = 0.01
    momentum_mom_window: int = 720
    momentum_mom_threshold: float = 0.0
    momentum_bull_multiplier: float = 1.0
    momentum_bear_multiplier: float = 0.0
    momentum_range_multiplier: float = 0.75
    momentum_flat_in_bear: bool = True
    
    # CSV params
    csv_vol_window: int = 168
    
    # Weights
    momentum_weight: float = 0.5
    csv_weight: float = 0.5
    
    # Trade params
    top_k: int = 5
    bottom_k: int = 5
    rebalance_bars: int = 24
    
    # Sizing/Costs
    position_size: float = 0.10
    leverage: float = 1.0
    taker_fee: float = 0.0005
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 24
    
    @property
    def skip_recent(self):
        return self.momentum_skip


class CSVMomentumEnsemble(AlphaStrategy):
    """CSV + Momentum Ensemble."""
    
    def __init__(self, config: CSVMomentumConfig):
        self.config = config
        
        self.momentum = TimedMomentumAlpha(TimedMomentumConfig(
            momentum_window=config.momentum_window,
            skip_recent=config.momentum_skip,
            regime_trend_window=config.momentum_trend_window,
            regime_trend_threshold=config.momentum_trend_threshold,
            regime_vol_window=config.momentum_vol_window,
            regime_vol_threshold_high=config.momentum_vol_threshold_high,
            regime_vol_threshold_low=config.momentum_vol_threshold_low,
            regime_mom_window=config.momentum_mom_window,
            regime_mom_threshold=config.momentum_mom_threshold,
            bull_multiplier=config.momentum_bull_multiplier,
            bear_multiplier=config.momentum_bear_multiplier,
            range_multiplier=config.momentum_range_multiplier,
            flat_in_bear=config.momentum_flat_in_bear,
            top_k=config.top_k,
            bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size * config.momentum_weight,
            leverage=config.leverage,
            taker_fee=config.taker_fee,
            slippage_bps=config.slippage_bps,
            funding_rate=config.funding_rate,
            funding_interval_bars=config.funding_interval_bars,
        ))
        
        self.csv = CrossSectionalVolAlpha(CrossSectionalVolConfig(
            vol_window=config.csv_vol_window,
            top_k=config.top_k,
            bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size * config.csv_weight,
            leverage=config.leverage,
            taker_fee=config.taker_fee,
            slippage_bps=config.slippage_bps,
            funding_rate=config.funding_rate,
            funding_interval_bars=config.funding_interval_bars,
        ))
    
    def generate_weights(self, prices, volumes=None):
        mom_w = self.momentum.generate_weights(prices, volumes)
        csv_w = self.csv.generate_weights(prices, volumes)
        
        combined = {}
        all_syms = set(mom_w.keys()) | set(csv_w.keys())
        for s in all_syms:
            combined[s] = mom_w.get(s, 0.0) + csv_w.get(s, 0.0)
        
        return combined
    
    def compute_scores(self, prices, volumes=None):
        return pd.Series(0.0, index=prices.columns)
