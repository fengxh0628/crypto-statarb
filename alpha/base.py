"""Alpha strategy base class.

All alpha strategies should inherit from this class and implement:
- compute_scores(): Generate alpha scores for each symbol
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AlphaConfig:
    """Base configuration for alpha strategies."""
    # Universe
    symbols: list = None  # List of symbols to trade
    
    # Position sizing
    top_k: int = 5  # Number of symbols to go long
    bottom_k: int = 5  # Number of symbols to go short
    position_size: float = 0.10  # Position size per symbol
    leverage: float = 1.0
    
    # Rebalancing
    rebalance_bars: int = 24  # Rebalance frequency in bars
    
    # Costs
    taker_fee: float = 0.0005
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 24
    
    # Filtering
    min_volume_percentile: float = 0.2  # Exclude low volume symbols
    vol_filter_window: int = 72


class AlphaStrategy(ABC):
    """Base class for alpha strategies."""
    
    def __init__(self, config: AlphaConfig):
        self.config = config
    
    @abstractmethod
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute alpha scores for each symbol.
        
        Args:
            prices: DataFrame of close prices (index=datetime, columns=symbols)
            volumes: DataFrame of volumes (optional)
            
        Returns:
            Series of scores (index=symbols), higher = more bullish
        """
        pass
    
    def filter_universe(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Index:
        """Filter symbols based on liquidity and data quality."""
        # Remove symbols with too many NaN
        valid_data = prices.notna().sum() > len(prices) * 0.8
        
        # Volume filter
        if volumes is not None:
            vol_window = self.config.vol_filter_window
            if len(volumes) >= vol_window:
                avg_vol = volumes.iloc[-vol_window:].mean()
                vol_threshold = avg_vol.quantile(self.config.min_volume_percentile)
                sufficient_vol = avg_vol > vol_threshold
                valid_data = valid_data & sufficient_vol
        
        return prices.columns[valid_data]
    
    def generate_weights(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Generate portfolio weights from alpha scores.
        
        Returns:
            Dict of {symbol: weight}, positive = long, negative = short
        """
        # Filter universe
        valid_symbols = self.filter_universe(prices, volumes)
        if len(valid_symbols) < self.config.top_k + self.config.bottom_k:
            return {}
        
        # Compute scores
        valid_prices = prices[valid_symbols]
        scores = self.compute_scores(valid_prices, volumes[valid_symbols] if volumes is not None else None)
        scores = scores.dropna()
        
        if len(scores) < self.config.top_k + self.config.bottom_k:
            return {}
        
        # Select top and bottom
        sorted_scores = scores.sort_values(ascending=False)
        long_symbols = sorted_scores.head(self.config.top_k).index.tolist()
        short_symbols = sorted_scores.tail(self.config.bottom_k).index.tolist()
        
        # Generate weights
        weights = {}
        for sym in long_symbols:
            weights[sym] = self.config.position_size * self.config.leverage
        for sym in short_symbols:
            weights[sym] = -self.config.position_size * self.config.leverage
        
        return weights
