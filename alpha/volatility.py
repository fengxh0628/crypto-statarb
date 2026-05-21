"""Volatility risk premium alpha strategy.

Logic:
1. Calculate realized volatility for each symbol
2. Go long low volatility symbols, short high volatility symbols
3. Rationale: High vol symbols tend to mean-revert, low vol symbols provide stable returns
4. Rebalance when volatility ranking changes
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class VolatilityConfig(AlphaConfig):
    # 波动率参数
    vol_window: int = 72  # 波动率计算窗口（3天 for 1h）
    vol_ma_window: int = 336  # 波动率移动平均窗口（14天）
    
    # 信号参数
    vol_rank_method: str = 'percentile'  # 'percentile' or 'zscore'
    
    # 交易参数
    top_k: int = 5
    bottom_k: int = 5
    rebalance_bars: int = 24
    
    # 仓位管理
    position_size: float = 0.10
    leverage: float = 1.0
    
    # 成本
    taker_fee: float = 0.0005
    slippage_bps: float = 1.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 24
    
    # 过滤
    min_vol_threshold: float = 0.001
    max_vol_threshold: float = 0.1


class VolatilityAlpha(AlphaStrategy):
    """Volatility risk premium strategy."""
    
    def __init__(self, config: VolatilityConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute volatility-based scores."""
        n = len(prices)
        if n < max(self.config.vol_window, self.config.vol_ma_window) + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # Calculate realized volatility
        vol = self._compute_volatility(prices, self.config.vol_window)
        vol_ma = self._compute_volatility(prices, self.config.vol_ma_window)
        
        # Volatility ratio (current vs long-term)
        vol_ratio = vol / (vol_ma + 1e-10)
        
        # Signal: low vol = long, high vol = short
        # Use z-score or percentile ranking
        if self.config.vol_rank_method == 'zscore':
            signal = (vol_ratio - vol_ratio.mean()) / (vol_ratio.std() + 1e-10)
        else:  # percentile
            signal = vol_ratio.rank(pct=True) * 2 - 1  # -1 to 1
        
        return signal
    
    @staticmethod
    def _compute_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute realized volatility."""
        returns = np.log(prices).diff().iloc[-window:]
        return returns.std()
