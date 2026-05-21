"""Volume/flow alpha strategy.

Logic:
1. Calculate volume breakout signal (current volume vs average)
2. Calculate volume-price divergence (price up but volume down = bearish)
3. Go long positive divergence, short negative divergence
4. Rationale: Volume confirms price moves, divergence signals reversals
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class VolumeConfig(AlphaConfig):
    # 成交量参数
    vol_ma_window: int = 72  # 成交量移动平均窗口（3天）
    vol_breakout_threshold: float = 1.5  # 成交量突破阈值
    
    # 价格参数
    price_window: int = 24  # 价格趋势窗口（1天）
    
    # 信号组合
    vol_weight: float = 0.4  # 成交量权重
    divergence_weight: float = 0.6  # 背离权重
    
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
    vol_filter_window: int = 72


class VolumeAlpha(AlphaStrategy):
    """Volume/flow alpha strategy."""
    
    def __init__(self, config: VolumeConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute volume-based scores."""
        if volumes is None:
            return pd.Series(np.nan, index=prices.columns)
        
        n = len(prices)
        if n < max(self.config.vol_ma_window, self.config.price_window) + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # Volume breakout signal
        vol_ratio = self._compute_volume_ratio(volumes, self.config.vol_ma_window)
        
        # Price trend
        price_trend = self._compute_price_trend(prices, self.config.price_window)
        
        # Volume-price divergence
        # Positive divergence: price down but volume up (accumulation) = bullish
        # Negative divergence: price up but volume down (distribution) = bearish
        divergence = -price_trend * vol_ratio  # inverted for mean reversion
        
        # Combined signal
        # High vol_ratio + negative trend = strong buy signal
        # Low vol_ratio + positive trend = strong sell signal
        signal = (self.config.vol_weight * vol_ratio + 
                  self.config.divergence_weight * divergence)
        
        return signal
    
    @staticmethod
    def _compute_volume_ratio(volumes: pd.DataFrame, window: int) -> pd.Series:
        """Compute volume ratio (current vs average)."""
        vol_ma = volumes.rolling(window).mean()
        vol_recent = volumes.iloc[-1]
        return vol_recent / (vol_ma.iloc[-1] + 1e-10)
    
    @staticmethod
    def _compute_price_trend(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute price trend (log return over window)."""
        log_prices = np.log(prices)
        return log_prices.iloc[-1] - log_prices.iloc[-window]
    
    @staticmethod
    def _compute_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute volatility for filtering."""
        returns = np.log(prices).diff().iloc[-window:]
        return returns.std()
