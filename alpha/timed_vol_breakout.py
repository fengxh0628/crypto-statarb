"""Volatility breakout with market timing.

Adds regime detection to volatility breakout strategy:
- Bull: Full position
- Bear: Flat (or reduced)
- Range: Reduced position
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig
from alpha.vol_breakout import VolBreakoutConfig
from alpha.market_regime import MarketRegime, RegimeConfig


@dataclass
class TimedVolBreakoutConfig(AlphaConfig):
    """Configuration for timed volatility breakout strategy."""
    # Breakout parameters
    breakout_window: int = 168  # 7 days for 1h
    vol_filter_window: int = 72  # 3 days
    
    # Volatility threshold
    min_vol: float = 0.005  # Minimum daily vol
    max_vol: float = 0.15  # Maximum daily vol
    
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
    bear_multiplier: float = 0.0
    range_multiplier: float = 0.75
    
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


class TimedVolBreakoutAlpha(AlphaStrategy):
    """Volatility breakout alpha with market timing."""
    
    def __init__(self, config: TimedVolBreakoutConfig):
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
        """Compute breakout scores."""
        n = len(prices)
        if n < self.config.breakout_window + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # Calculate recent high/low
        window = self.config.breakout_window
        recent_high = prices.iloc[-window:].max()
        recent_low = prices.iloc[-window:].min()
        current_price = prices.iloc[-1]
        
        # Breakout signal: how far price is from range
        range_size = recent_high - recent_low
        range_size = range_size.replace(0, np.nan)
        
        # Normalized breakout: (price - mid) / (range/2)
        mid = (recent_high + recent_low) / 2
        breakout_score = (current_price - mid) / (range_size / 2 + 1e-10)
        
        # Volatility filter
        vol = self._compute_volatility(prices, self.config.vol_filter_window)
        valid_vol = (vol > self.config.min_vol) & (vol < self.config.max_vol)
        breakout_score[~valid_vol] = np.nan
        
        return breakout_score
    
    def generate_weights(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Generate weights with regime-based position sizing."""
        # Detect market regime
        regime = self.regime_detector.detect_regime(prices, volumes)
        
        # Get position multiplier
        multiplier = {
            'bull': self.config.bull_multiplier,
            'bear': self.config.bear_multiplier,
            'range': self.config.range_multiplier,
        }.get(regime, 0.5)
        
        if multiplier == 0:
            return {}
        
        # Override config multiplier temporarily
        original_size = self.config.position_size
        self.config.position_size *= multiplier
        
        # Generate weights
        weights = super().generate_weights(prices, volumes)
        
        # Restore original size
        self.config.position_size = original_size
        
        return weights
    
    @staticmethod
    def _compute_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute realized volatility."""
        returns = np.log(prices).diff().iloc[-window:]
        return returns.std()
