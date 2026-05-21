"""Momentum alpha with market timing.

Adds regime detection to momentum strategy:
- Bull: Full position
- Bear: Half position or flat
- Range: Reduced position
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig
from alpha.momentum import MomentumConfig
from alpha.market_regime import MarketRegime, RegimeConfig


@dataclass
class TimedMomentumConfig(AlphaConfig):
    """Configuration for timed momentum strategy."""
    # Momentum parameters
    momentum_window: int = 336
    skip_recent: int = 12
    
    # Regime detection
    regime_trend_window: int = 336
    regime_trend_threshold: float = 0.0
    regime_vol_window: int = 720
    regime_vol_threshold_high: float = 0.05
    regime_vol_threshold_low: float = 0.01
    regime_mom_window: int = 720
    regime_mom_threshold: float = 0.0
    
    # Position sizing by regime
    bull_multiplier: float = 1.0
    bear_multiplier: float = 0.5
    range_multiplier: float = 0.75
    
    # Flat in bear market (optional)
    flat_in_bear: bool = False


class TimedMomentumAlpha(AlphaStrategy):
    """Momentum alpha with market timing."""
    
    def __init__(self, config: TimedMomentumConfig):
        self.config = config
        
        # Regime detector
        self.regime_config = RegimeConfig(
            trend_window=config.regime_trend_window,
            trend_threshold=config.regime_trend_threshold,
            vol_window=config.regime_vol_window,
            vol_threshold_high=config.regime_vol_threshold_high,
            vol_threshold_low=config.regime_vol_threshold_low,
            mom_window=config.regime_mom_window,
            mom_threshold=config.regime_mom_threshold,
        )
        self.regime_detector = MarketRegime(self.regime_config)
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute momentum scores (same as base momentum)."""
        n = len(prices)
        if n < self.config.momentum_window + self.config.skip_recent:
            return pd.Series(np.nan, index=prices.columns)
        
        start_idx = n - self.config.momentum_window - self.config.skip_recent
        end_idx = n - self.config.skip_recent
        
        log_prices = np.log(prices.values)
        momentum = log_prices[end_idx] - log_prices[start_idx]
        
        return pd.Series(momentum, index=prices.columns)
    
    def generate_weights(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Generate weights with regime-based position sizing."""
        # Detect market regime
        regime = self.regime_detector.detect_regime(prices, volumes)
        
        # Get position multiplier
        if regime == 'bear' and self.config.flat_in_bear:
            return {}  # Flat in bear market
        
        multiplier = self.regime_detector.get_position_multiplier(regime)
        
        # Override config multiplier temporarily
        original_size = self.config.position_size
        self.config.position_size *= multiplier
        
        # Generate weights
        weights = super().generate_weights(prices, volumes)
        
        # Restore original size
        self.config.position_size = original_size
        
        return weights
