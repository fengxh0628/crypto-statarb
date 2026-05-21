"""Cross-Sectional Volatility Strategy.

Logic:
- Calculate realized volatility for all symbols over a lookback window.
- Rank symbols by volatility.
- Short the highest volatility symbols (expecting vol compression).
- Long the lowest volatility symbols (expecting vol expansion or stability).
- This is a "Volatility Risk Premium" style strategy.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class CrossSectionalVolConfig(AlphaConfig):
    # Volatility parameters
    vol_window: int = 168  # 7 days
    vol_rank_method: str = 'percentile'  # 'percentile' or 'raw'
    
    # Trade parameters
    top_k: int = 5  # Low vol (long)
    bottom_k: int = 5  # High vol (short)
    rebalance_bars: int = 24
    
    # Position sizing
    position_size: float = 0.10
    leverage: float = 1.0
    
    # Costs
    taker_fee: float = 0.0005
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 24


class CrossSectionalVolAlpha(AlphaStrategy):
    """Cross-Sectional Volatility alpha."""
    
    def __init__(self, config: CrossSectionalVolConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute cross-sectional volatility scores."""
        n = len(prices)
        if n < self.config.vol_window + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # Calculate realized volatility
        returns = np.log(prices).diff().dropna()
        vol = returns.rolling(self.config.vol_window).std().iloc[-1]
        
        # Remove NaN
        vol = vol.dropna()
        
        if len(vol) < self.config.top_k + self.config.bottom_k:
            return pd.Series(0.0, index=prices.columns)
        
        # Score: Low vol = positive score (long), High vol = negative score (short)
        # Use percentile rank for robustness
        score = pd.Series(0.0, index=prices.columns)
        
        # Rank from 0 (low vol) to 1 (high vol)
        vol_rank = vol.rank(pct=True)
        
        # Invert: we want to long low vol, short high vol
        score[vol_rank.index] = -vol_rank
        
        return score
