"""Momentum + Volatility Squeeze Ensemble.

Combines Timed Momentum and Volatility Squeeze strategies.
- Allocates 50% capital to Momentum
- Allocates 50% capital to Squeeze
- Diversifies risk and reduces overall turnover
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig
from alpha.timed_momentum import TimedMomentumAlpha, TimedMomentumConfig
from alpha.vol_squeeze import VolSqueezeAlpha, VolSqueezeConfig


@dataclass
class MomentumSqueezeConfig(AlphaConfig):
    """Configuration for Momentum + Squeeze Ensemble."""
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
    
    # Squeeze parameters
    squeeze_bb_window: int = 20
    squeeze_bb_std: float = 2.0
    squeeze_percentile: float = 0.1
    squeeze_lookback_percentile: int = 336
    squeeze_breakout_window: int = 24
    squeeze_volume_confirm: bool = True
    squeeze_volume_ma_window: int = 72
    
    # Weights
    momentum_weight: float = 0.5
    squeeze_weight: float = 0.5
    
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
    
    # Aliases for backtest engine compatibility
    @property
    def skip_recent(self):
        return self.momentum_skip


class MomentumSqueezeEnsemble(AlphaStrategy):
    """Momentum + Volatility Squeeze Ensemble."""
    
    def __init__(self, config: MomentumSqueezeConfig):
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
            position_size=config.position_size * config.momentum_weight,  # Split position size
            leverage=config.leverage,
            taker_fee=config.taker_fee,
            slippage_bps=config.slippage_bps,
            funding_rate=config.funding_rate,
            funding_interval_bars=config.funding_interval_bars,
        ))
        
        self.squeeze = VolSqueezeAlpha(VolSqueezeConfig(
            bb_window=config.squeeze_bb_window,
            bb_std=config.squeeze_bb_std,
            squeeze_percentile=config.squeeze_percentile,
            lookback_percentile=config.squeeze_lookback_percentile,
            breakout_window=config.squeeze_breakout_window,
            volume_confirm=config.squeeze_volume_confirm,
            volume_ma_window=config.squeeze_volume_ma_window,
            top_k=config.top_k,
            bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size * config.squeeze_weight,  # Split position size
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
        # Get weights from Momentum
        mom_weights = self.momentum.generate_weights(prices, volumes)
        
        # Get weights from Squeeze
        sq_weights = self.squeeze.generate_weights(prices, volumes)
        
        # Combine weights
        all_symbols = set(mom_weights.keys()) | set(sq_weights.keys())
        combined_weights = {}
        
        for sym in all_symbols:
            w_mom = mom_weights.get(sym, 0.0)
            w_sq = sq_weights.get(sym, 0.0)
            
            # If both strategies agree, add weights
            # If they disagree, they partially cancel out (reducing risk)
            combined_weights[sym] = w_mom + w_sq
        
        return combined_weights
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute combined scores (for compatibility)."""
        # This method is not used in generate_weights, but required by base class
        return pd.Series(0.0, index=prices.columns)
