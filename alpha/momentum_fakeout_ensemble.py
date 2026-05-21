"""Momentum + Fakeout Reversal Ensemble.

Combines Timed Momentum and Fakeout Reversal strategies.
- Allocates 50% capital to Momentum
- Allocates 50% capital to Fakeout Reversal
- Diversifies risk: Momentum captures trends, Fakeout captures reversals
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig
from alpha.timed_momentum import TimedMomentumAlpha, TimedMomentumConfig
from alpha.fakeout_reversal import FakeoutAlpha, FakeoutConfig


@dataclass
class MomentumFakeoutConfig(AlphaConfig):
    """Configuration for Momentum + Fakeout Ensemble."""
    # Momentum parameters
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
    
    # Fakeout parameters
    fakeout_lookback: int = 48
    fakeout_confirmation_bars: int = 1
    
    # Weights
    momentum_weight: float = 0.5
    fakeout_weight: float = 0.5
    
    # Trade parameters
    top_k: int = 5
    bottom_k: int = 5
    rebalance_bars: int = 24
    
    # Position sizing
    position_size: float = 0.10
    leverage: float = 1.0
    
    # Costs
    taker_fee: float = 0.0005
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 24
    
    @property
    def skip_recent(self):
        return self.momentum_skip


class MomentumFakeoutEnsemble(AlphaStrategy):
    """Momentum + Fakeout Reversal Ensemble."""
    
    def __init__(self, config: MomentumFakeoutConfig):
        self.config = config
        
        # Sub-strategies
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
        
        self.fakeout = FakeoutAlpha(FakeoutConfig(
            lookback_window=config.fakeout_lookback,
            confirmation_bars=config.fakeout_confirmation_bars,
            top_k=config.top_k,
            bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size * config.fakeout_weight,
            leverage=config.leverage,
            taker_fee=config.taker_fee,
            slippage_bps=config.slippage_bps,
            funding_rate=config.funding_rate,
            funding_interval_bars=config.funding_interval_bars,
        ))
    
    def generate_weights(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Generate combined weights from both strategies."""
        mom_weights = self.momentum.generate_weights(prices, volumes)
        fo_weights = self.fakeout.generate_weights(prices, volumes)
        
        all_symbols = set(mom_weights.keys()) | set(fo_weights.keys())
        combined_weights = {}
        
        for sym in all_symbols:
            w_mom = mom_weights.get(sym, 0.0)
            w_fo = fo_weights.get(sym, 0.0)
            combined_weights[sym] = w_mom + w_fo
        
        return combined_weights
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute combined scores (for compatibility)."""
        return pd.Series(0.0, index=prices.columns)
