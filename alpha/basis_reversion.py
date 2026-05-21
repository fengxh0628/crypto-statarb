"""Basis Reversion (Mark Price vs Contract Price).

Logic:
- Basis = Mark Price - Contract Price (Close).
- High Basis (Premium) usually implies positive Funding Rate (Longs pay Shorts).
- Strategy: Short High Premium (collect funding + expect reversion), Long Low Premium.
- Data Source: premiumIndexKlines (Mark Price) + klines (Close Price).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class BasisConfig(AlphaConfig):
    # Basis parameters
    basis_window: int = 72  # 3 days for mean/std
    
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


class BasisAlpha(AlphaStrategy):
    """Basis Reversion alpha."""
    
    def __init__(self, config: BasisConfig):
        self.config = config
        self.mark_prices = None  # Will be set externally
    
    def set_mark_prices(self, mark_df: pd.DataFrame):
        """Set Mark Price data."""
        self.mark_prices = mark_df
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute basis reversion scores."""
        if self.mark_prices is None:
            return pd.Series(0.0, index=prices.columns)
        
        # Align data
        common_idx = self.mark_prices.index.intersection(prices.index)
        common_cols = self.mark_prices.columns.intersection(prices.columns)
        
        if len(common_idx) < 100 or len(common_cols) < 2:
            return pd.Series(0.0, index=prices.columns)
        
        mark = self.mark_prices.loc[common_idx, common_cols]
        close = prices.loc[common_idx, common_cols]
        
        # Calculate Basis (Premium)
        basis = mark - close
        
        # Z-score of Basis
        basis_mean = basis.rolling(self.config.basis_window).mean().iloc[-1]
        basis_std = basis.rolling(self.config.basis_window).std().iloc[-1]
        
        z_score = (basis.iloc[-1] - basis_mean) / (basis_std + 1e-10)
        
        # Signal: High Premium (Positive Z) -> Short. Low Premium -> Long.
        # Score = -Z-score
        score = -z_score
        
        return score
