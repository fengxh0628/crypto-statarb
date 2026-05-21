"""Volatility breakout strategy.

Logic:
- Go long when price breaks above recent high
- Go short when price breaks below recent low
- Rebalance when breakout signals change
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class VolBreakoutConfig(AlphaConfig):
    # Breakout parameters
    breakout_window: int = 168  # 7 days for 1h
    vol_filter_window: int = 72  # 3 days
    
    # Volatility threshold
    min_vol: float = 0.005  # Minimum daily vol
    max_vol: float = 0.15  # Maximum daily vol
    
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


class VolBreakoutAlpha(AlphaStrategy):
    """Volatility breakout alpha."""
    
    def __init__(self, config: VolBreakoutConfig):
        self.config = config
    
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
        # Positive = above high (long), Negative = below low (short)
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
    
    @staticmethod
    def _compute_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute realized volatility."""
        returns = np.log(prices).diff().iloc[-window:]
        return returns.std()
