"""Fakeout Reversal (Liquidity Sweep).

Logic:
- Identify breakouts of recent High/Low.
- If price breaks out but closes back inside the range (Fakeout), trade the reversal.
- "Stop hunt" detection.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class FakeoutConfig(AlphaConfig):
    # Range parameters
    lookback_window: int = 48  # 2 days
    
    # Fakeout definition
    # We need Open/High/Low/Close data.
    # Since we only have 'prices' (Close) in the standard interface,
    # we have to approximate or assume prices contains OHLC?
    # Looking at the loader, it usually loads Close.
    # If we only have Close, we can't detect intrabar fakeouts easily.
    # However, we can detect "Breakout then Reversal" over 2 bars.
    # Bar 1: Breaks High. Bar 2: Closes back below High.
    
    confirmation_bars: int = 1  # Wait 1 bar to confirm fakeout
    
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


class FakeoutAlpha(AlphaStrategy):
    """Fakeout Reversal alpha."""
    
    def __init__(self, config: FakeoutConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute fakeout reversal scores."""
        n = len(prices)
        if n < self.config.lookback_window + self.config.confirmation_bars + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # Rolling High/Low of Close (approximation)
        rolling_high = prices.rolling(self.config.lookback_window).max()
        rolling_low = prices.rolling(self.config.lookback_window).min()
        
        # Previous High/Low (shifted by 1 to avoid look-ahead)
        prev_high = rolling_high.shift(1).iloc[-1]
        prev_low = rolling_low.shift(1).iloc[-1]
        
        # Current Price
        current_price = prices.iloc[-1]
        prev_price = prices.iloc[-2]
        
        score = pd.Series(0.0, index=prices.columns)
        
        # 1. Bullish Fakeout (Price broke Low, but closed back above)
        # Logic: 
        # Bar t-1 (or earlier): Price < Prev Low
        # Bar t: Price > Prev Low
        
        # Since we only have Close, let's look at the sequence:
        # Close[t-2] < Low_Lookback[t-2] (Breakdown)
        # Close[t-1] > Low_Lookback[t-2] (Recovery)
        
        # Actually, simpler:
        # Check if we are currently "Inside" the range, but were "Outside" recently.
        
        # Let's use a simpler heuristic:
        # Momentum is negative (Price dropped), but Price is now bouncing off support.
        # This is hard to define purely on Close without OHLC.
        
        # Let's try "Reversal from Extreme":
        # Price dropped X% in Y bars, now RSI is low.
        # This is similar to Trend Pullback.
        
        # Let's stick to the "Breakout Failure" logic using Close only:
        # 1. Price broke out of range N bars ago.
        # 2. Price is now back inside range.
        
        # Check if Price is inside range
        inside_range = (current_price > prev_low) & (current_price < prev_high)
        
        # Check if Price was outside range recently (e.g. 1-3 bars ago)
        # We check max/min of last few bars
        recent_extreme_high = prices.iloc[-(self.config.confirmation_bars+3):-(self.config.confirmation_bars)].max()
        recent_extreme_low = prices.iloc[-(self.config.confirmation_bars+3):-(self.config.confirmation_bars)].min()
        
        broke_high = (recent_extreme_high > prev_high)
        broke_low = (recent_extreme_low < prev_low)
        
        # Bullish Fakeout: Broke Low, now Inside -> Long
        bullish_fakeout = broke_low & inside_range
        
        # Bearish Fakeout: Broke High, now Inside -> Short
        bearish_fakeout = broke_high & inside_range
        
        # Score magnitude: How deep was the fakeout?
        # (Prev Low - Low_point) / Range
        # Approximate with how far back inside we are
        
        if bullish_fakeout.any():
            # Score based on how close to Low we were vs now
            # Simplified: Distance from Low
            score[bullish_fakeout] = (current_price[bullish_fakeout] - prev_low[bullish_fakeout]) / (prev_high[bullish_fakeout] - prev_low[bullish_fakeout] + 1e-10)
        
        if bearish_fakeout.any():
            score[bearish_fakeout] = -(prev_high[bearish_fakeout] - current_price[bearish_fakeout]) / (prev_high[bearish_fakeout] - prev_low[bearish_fakeout] + 1e-10)
        
        return score
