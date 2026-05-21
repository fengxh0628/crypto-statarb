"""Momentum alpha strategy.

Computes momentum scores based on past returns.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class MomentumConfig(AlphaConfig):
    """Configuration for momentum strategy."""
    momentum_window: int = 336  # Lookback window (14 days for 1h)
    skip_recent: int = 12  # Skip recent bars to avoid reversal


class MomentumAlpha(AlphaStrategy):
    """Momentum alpha: long winners, short losers."""
    
    def __init__(self, config: MomentumConfig):
        super().__init__(config)
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute momentum scores.
        
        Score = log(price[t-skip] / price[t-skip-window])
        """
        n = len(prices)
        if n < self.config.momentum_window + self.config.skip_recent:
            return pd.Series(np.nan, index=prices.columns)
        
        start_idx = n - self.config.momentum_window - self.config.skip_recent
        end_idx = n - self.config.skip_recent
        
        log_prices = np.log(prices.values)
        momentum = log_prices[end_idx] - log_prices[start_idx]
        
        return pd.Series(momentum, index=prices.columns)
