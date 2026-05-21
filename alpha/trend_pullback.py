"""Trend pullback strategy.

Logic:
- Identify strong trend (Price > MA_Long)
- Wait for pullback (RSI < Oversold or Price Drop)
- Enter in direction of trend
- "Buy the dip" in uptrends, "Sell the rally" in downtrends
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class TrendPullbackConfig(AlphaConfig):
    # Trend filter
    trend_ma_window: int = 504  # ~21 days
    trend_threshold: float = 0.0  # Price must be above/below MA
    
    # Pullback signal
    pullback_rsi_window: int = 14
    pullback_rsi_oversold: float = 35  # Buy dip when RSI < this
    pullback_rsi_overbought: float = 65  # Sell rally when RSI > this
    
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


class TrendPullbackAlpha(AlphaStrategy):
    """Trend pullback alpha."""
    
    def __init__(self, config: TrendPullbackConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute trend pullback scores."""
        n = len(prices)
        if n < self.config.trend_ma_window + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        # 1. Trend Filter
        ma_trend = prices.rolling(self.config.trend_ma_window).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        # Uptrend: Price > MA, Downtrend: Price < MA
        is_uptrend = current_price > ma_trend
        is_downtrend = current_price < ma_trend
        
        # 2. Pullback Signal (RSI)
        rsi = self._compute_rsi(prices, self.config.pullback_rsi_window)
        
        # Score:
        # Uptrend + Oversold -> Strong Buy (High positive score)
        # Downtrend + Overbought -> Strong Sell (High negative score)
        # Else -> 0
        
        score = pd.Series(0.0, index=prices.columns)
        
        # Buy dips in uptrend
        buy_signal = is_uptrend & (rsi < self.config.pullback_rsi_oversold)
        score[buy_signal] = (self.config.pullback_rsi_oversold - rsi[buy_signal]) / 100
        
        # Sell rallies in downtrend
        sell_signal = is_downtrend & (rsi > self.config.pullback_rsi_overbought)
        score[sell_signal] = -(rsi[sell_signal] - self.config.pullback_rsi_overbought) / 100
        
        return score
    
    @staticmethod
    def _compute_rsi(prices: pd.DataFrame, window: int) -> pd.Series:
        """Compute RSI."""
        returns = np.log(prices).diff()
        gain = returns.clip(lower=0)
        loss = -returns.clip(upper=0)
        avg_gain = gain.rolling(window).mean().iloc[-1]
        avg_loss = loss.rolling(window).mean().iloc[-1]
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
