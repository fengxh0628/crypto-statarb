"""Market regime detection and timing.

Detects market states:
- Bull: Strong uptrend, high momentum
- Bear: Strong downtrend, low momentum
- Range: Sideways, mean-reverting

Uses BTC as market proxy.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Trend detection
    trend_window: int = 336  # 14 days for 1h
    trend_threshold: float = 0.0  # Slope threshold
    
    # Volatility regime
    vol_window: int = 720  # 30 days
    vol_threshold_high: float = 0.05  # High vol threshold
    vol_threshold_low: float = 0.01  # Low vol threshold
    
    # Momentum regime
    mom_window: int = 720  # 30 days
    mom_threshold: float = 0.0  # Momentum threshold


class MarketRegime:
    """Detect market regime using BTC price action."""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
    
    def detect_regime(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> str:
        """Detect current market regime.
        
        Returns:
            'bull', 'bear', or 'range'
        """
        if 'BTCUSDT' not in prices.columns:
            return 'range'
        
        btc = prices['BTCUSDT']
        n = len(btc)
        
        # Need enough data for all indicators
        min_bars = max(
            self.config.trend_window,
            self.config.vol_window,
            self.config.mom_window,
        )
        if n < min_bars:
            return 'range'
        
        # 1. Trend detection (linear regression slope)
        trend = self._compute_trend(btc, self.config.trend_window)
        
        # 2. Volatility regime
        vol = self._compute_volatility(btc, self.config.vol_window)
        
        # 3. Momentum
        mom = self._compute_momentum(btc, self.config.mom_window)
        
        # Regime classification
        if trend > self.config.trend_threshold and mom > self.config.mom_threshold:
            return 'bull'
        elif trend < -self.config.trend_threshold and mom < -self.config.mom_threshold:
            return 'bear'
        else:
            return 'range'
    
    def get_position_multiplier(self, regime: str) -> float:
        """Get position size multiplier based on regime.
        
        Returns:
            Multiplier (0.0 to 1.0)
        """
        multipliers = {
            'bull': 1.0,    # Full position
            'bear': 0.5,    # Half position
            'range': 0.75,  # Reduced position
        }
        return multipliers.get(regime, 0.5)
    
    @staticmethod
    def _compute_trend(series: pd.Series, window: int) -> float:
        """Compute linear regression slope."""
        y = np.log(series.values[-window:])
        x = np.arange(window)
        
        x_mean = x.mean()
        y_mean = y.mean()
        
        slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)
        return slope
    
    @staticmethod
    def _compute_volatility(series: pd.Series, window: int) -> float:
        """Compute realized volatility."""
        returns = np.log(series).diff().iloc[-window:]
        return returns.std()
    
    @staticmethod
    def _compute_momentum(series: pd.Series, window: int) -> float:
        """Compute log return momentum."""
        log_p = np.log(series.values)
        return log_p[-1] - log_p[-window]
