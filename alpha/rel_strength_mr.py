"""Relative strength mean reversion strategy.

Logic:
- Calculate relative strength vs BTC (excess return)
- Short outperformers (expect reversion)
- Long underperformers (expect bounce)
- Low correlation with pure momentum
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class RelStrengthMRConfig(AlphaConfig):
    # Parameters
    rs_window: int = 72  # 3 days for 1h
    skip_recent: int = 12
    
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


class RelStrengthMRAlpha(AlphaStrategy):
    """Relative strength mean reversion alpha."""
    
    def __init__(self, config: RelStrengthMRConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute relative strength mean reversion scores."""
        n = len(prices)
        if n < self.config.rs_window + self.config.skip_recent + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        if 'BTCUSDT' not in prices.columns:
            return pd.Series(np.nan, index=prices.columns)
        
        log_prices = np.log(prices.values)
        
        # Calculate excess return vs BTC
        start_idx = n - self.config.rs_window - self.config.skip_recent
        end_idx = n - self.config.skip_recent
        
        btc_col = prices.columns.get_loc('BTCUSDT')
        btc_return = log_prices[end_idx, btc_col] - log_prices[start_idx, btc_col]
        
        # Excess return for each symbol
        excess_return = (log_prices[end_idx] - log_prices[start_idx]) - btc_return
        
        # Mean reversion signal: negative of excess return
        # High excess return -> short (expect reversion)
        # Low excess return -> long (expect bounce)
        score = -excess_return
        
        return pd.Series(score, index=prices.columns)
