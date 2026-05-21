"""Mean reversion alpha strategy.

Logic:
1. Calculate RSI for each symbol
2. Calculate Bollinger Band position
3. Go long oversold (low RSI, below BB), go short overbought (high RSI, above BB)
4. Rebalance when signal changes significantly
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class MeanReversionConfig(AlphaConfig):
    # RSI 参数
    rsi_window: int = 14  # RSI 计算周期
    rsi_overbought: float = 70  # 超买阈值
    rsi_oversold: float = 30  # 超卖阈值
    
    # 布林带参数
    bb_window: int = 20  # 布林带周期
    bb_std: float = 2.0  # 标准差倍数
    
    # 信号组合
    rsi_weight: float = 0.6  # RSI 权重
    bb_weight: float = 0.4  # 布林带权重
    
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
    vol_window: int = 72


class MeanReversionAlpha(AlphaStrategy):
    """Mean reversion strategy using RSI and Bollinger Bands."""
    
    def __init__(self, config: MeanReversionConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute mean reversion scores."""
        n = len(prices)
        if n < max(self.config.rsi_window, self.config.bb_window) + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # Calculate RSI
        rsi = self._compute_rsi(prices, self.config.rsi_window)
        
        # Calculate Bollinger Band position
        bb_pos = self._compute_bb_position(prices, self.config.bb_window, self.config.bb_std)
        
        # Combine signals (normalized to -1 to 1)
        # RSI: 0-100 -> -1 to 1 (low=oversold=long, high=overbought=short)
        rsi_signal = -(rsi - 50) / 50  # -1 (oversold) to +1 (overbought)
        
        # BB position: -1 (below lower) to +1 (above upper)
        bb_signal = bb_pos  # already normalized
        
        # Combined signal (mean reversion: negative = long, positive = short)
        signal = (self.config.rsi_weight * rsi_signal + 
                  self.config.bb_weight * bb_signal)
        
        return signal
    
    @staticmethod
    def _compute_rsi(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute RSI for all symbols."""
        returns = np.log(prices).diff()
        gain = returns.clip(lower=0)
        loss = -returns.clip(upper=0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    @staticmethod
    def _compute_bb_position(
        prices: pd.DataFrame,
        window: int,
        std_mult: float,
    ) -> pd.Series:
        """Compute Bollinger Band position (-1 to 1)."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        
        # Position: -1 (at lower) to +1 (at upper)
        bb_range = upper - lower
        position = 2 * (prices.iloc[-1] - lower.iloc[-1]) / (bb_range.iloc[-1] + 1e-10) - 1
        
        return position
    
    @staticmethod
    def _compute_volatility(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute volatility for filtering."""
        returns = np.log(prices).diff().iloc[-window:]
        return returns.std()
