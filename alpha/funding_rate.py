"""Funding Rate Alpha Strategy.

Logic:
- Load funding rate data (from premiumIndexKlines).
- High funding rate = crowded longs, expect reversal (short).
- Low/negative funding rate = crowded shorts, expect reversal (long).
- Low turnover: funding rates change slowly.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class FundingRateConfig(AlphaConfig):
    # Funding rate parameters
    funding_window: int = 72  # 3 days for mean/std
    entry_threshold: float = 1.5  # Z-score threshold to enter
    
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


class FundingRateAlpha(AlphaStrategy):
    """Funding Rate alpha."""
    
    def __init__(self, config: FundingRateConfig):
        self.config = config
        self.funding_data = None  # Will be set externally
    
    def set_funding_data(self, funding_df: pd.DataFrame):
        """Set funding rate data (index=datetime, columns=symbols)."""
        self.funding_data = funding_df
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute funding rate scores."""
        if self.funding_data is None:
            return pd.Series(0.0, index=prices.columns)
        
        # Align funding data with prices
        common_idx = self.funding_data.index.intersection(prices.index)
        if len(common_idx) < self.config.funding_window + 10:
            return pd.Series(0.0, index=prices.columns)
        
        # Get latest funding rates
        funding = self.funding_data.loc[common_idx]
        current_funding = funding.iloc[-1]
        
        # Calculate Z-score of funding rate
        funding_mean = funding.rolling(self.config.funding_window).mean().iloc[-1]
        funding_std = funding.rolling(self.config.funding_window).std().iloc[-1]
        
        z_score = (current_funding - funding_mean) / (funding_std + 1e-10)
        
        # Signal: High funding (crowded longs) -> Short, Low funding -> Long
        # Invert: negative z_score -> long, positive z_score -> short
        score = -z_score
        
        # Filter by threshold
        score[abs(score) < self.config.entry_threshold] = 0.0
        
        return score
