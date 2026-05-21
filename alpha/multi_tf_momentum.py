"""Multi-timeframe momentum strategy.

Logic:
- Long-term momentum (7d) determines trend direction
- Short-term momentum (1d) determines entry timing
- Only trade when both timeframes agree
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class MultiTFMomentumConfig(AlphaConfig):
    # Momentum parameters
    long_window: int = 168  # 7 days for 1h
    short_window: int = 24  # 1 day for 1h
    skip_recent: int = 12  # Skip recent bars
    
    # Agreement threshold
    agreement_threshold: float = 0.0  # Both must be positive/negative
    
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


class MultiTFMomentumAlpha(AlphaStrategy):
    """Multi-timeframe momentum alpha."""
    
    def __init__(self, config: MultiTFMomentumConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute multi-timeframe momentum scores."""
        n = len(prices)
        if n < self.config.long_window + self.config.skip_recent + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        log_prices = np.log(prices.values)
        
        # Long-term momentum
        long_start = n - self.config.long_window - self.config.skip_recent
        long_end = n - self.config.skip_recent
        mom_long = log_prices[long_end] - log_prices[long_start]
        
        # Short-term momentum
        short_start = n - self.config.short_window - self.config.skip_recent
        short_end = n - self.config.skip_recent
        mom_short = log_prices[short_end] - log_prices[short_start]
        
        # Agreement filter: only trade when both agree
        # Score = long_mom * sign(short_mom) if they agree, else 0
        agreement = np.sign(mom_long) == np.sign(mom_short)
        
        # Combined score: weighted average, but only if they agree
        combined = 0.7 * mom_long + 0.3 * mom_short
        combined[~agreement] = 0  # No signal if disagreement
        
        return pd.Series(combined, index=prices.columns)
