"""Volatility squeeze breakout strategy.

Logic:
- Identify periods of low volatility (Squeeze)
- Enter when price breaks out of the squeeze range
- Direction determined by breakout direction
- "Volatility contraction leads to expansion"
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class VolSqueezeConfig(AlphaConfig):
    # Squeeze detection
    bb_window: int = 20
    bb_std: float = 2.0
    squeeze_percentile: float = 0.1  # Bottom 10% of BB width
    lookback_percentile: int = 336  # ~14 days for percentile calc
    
    # Breakout confirmation
    breakout_window: int = 24  # Range to break
    volume_confirm: bool = True
    volume_ma_window: int = 72
    
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


class VolSqueezeAlpha(AlphaStrategy):
    """Volatility squeeze breakout alpha."""
    
    def __init__(self, config: VolSqueezeConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute squeeze breakout scores."""
        n = len(prices)
        if n < self.config.lookback_percentile + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # 1. Calculate Bollinger Band Width
        bb_mean = prices.rolling(self.config.bb_window).mean()
        bb_std = prices.rolling(self.config.bb_window).std()
        bb_upper = bb_mean + self.config.bb_std * bb_std
        bb_lower = bb_mean - self.config.bb_std * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mean
        
        # 2. Check if in Squeeze (Width is at historical low)
        current_width = bb_width.iloc[-1]
        width_percentile = bb_width.rolling(self.config.lookback_percentile).rank(pct=True).iloc[-1]
        is_squeeze = width_percentile < self.config.squeeze_percentile
        
        # 3. Breakout Signal
        # If in squeeze, look for breakout above/below recent range
        recent_high = prices.iloc[-self.config.breakout_window:].max()
        recent_low = prices.iloc[-self.config.breakout_window:].min()
        current_price = prices.iloc[-1]
        
        score = pd.Series(0.0, index=prices.columns)
        
        # Breakout Up
        breakout_up = is_squeeze & (current_price > recent_high)
        # Breakout Down
        breakout_down = is_squeeze & (current_price < recent_low)
        
        # Volume Confirmation (optional)
        if self.config.volume_confirm and volumes is not None:
            vol_ma = volumes.rolling(self.config.volume_ma_window).mean().iloc[-1]
            current_vol = volumes.iloc[-1]
            vol_spike = current_vol > vol_ma * 1.5
            breakout_up = breakout_up & vol_spike
            breakout_down = breakout_down & vol_spike
        
        # Score magnitude: how far price broke out
        range_size = (recent_high - recent_low).replace(0, np.nan)
        
        if breakout_up.any():
            score[breakout_up] = (current_price[breakout_up] - recent_high[breakout_up]) / (range_size[breakout_up] / 2 + 1e-10)
        
        if breakout_down.any():
            score[breakout_down] = (current_price[breakout_down] - recent_low[breakout_down]) / (range_size[breakout_down] / 2 + 1e-10)
            score[breakout_down] = -score[breakout_down]  # Negative for short
        
        return score
