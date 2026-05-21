"""Multi-factor alpha strategy.

Combines:
1. Momentum (trend following)
2. Mean Reversion (RSI + Bollinger)
3. Volatility (low vol long, high vol short)
4. Volume (volume breakout + divergence)

Logic:
- Weighted combination of all factors
- Dynamic weighting based on recent factor performance
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class MultiFactorConfig(AlphaConfig):
    # 因子权重
    momentum_weight: float = 0.4
    mean_reversion_weight: float = 0.2
    volatility_weight: float = 0.2
    volume_weight: float = 0.2
    
    # 动态权重
    dynamic_weighting: bool = True
    lookback_performance: int = 336
    
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


class MultiFactorAlpha(AlphaStrategy):
    """Multi-factor alpha strategy."""
    
    def __init__(self, config: MultiFactorConfig):
        self.config = config
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute multi-factor scores."""
        from alpha.momentum import MomentumAlpha, MomentumConfig
        from alpha.mean_reversion import MeanReversionAlpha, MeanReversionConfig
        from alpha.volatility import VolatilityAlpha, VolatilityConfig
        from alpha.volume import VolumeAlpha, VolumeConfig
        
        # 计算各因子得分
        mom_config = MomentumConfig(
            momentum_window=336, skip_recent=12,
            top_k=self.config.top_k, bottom_k=self.config.bottom_k,
            rebalance_bars=self.config.rebalance_bars,
            position_size=self.config.position_size, leverage=self.config.leverage,
            taker_fee=self.config.taker_fee, slippage_bps=self.config.slippage_bps,
        )
        mr_config = MeanReversionConfig(
            rsi_window=14, rsi_overbought=70, rsi_oversold=30,
            bb_window=20, bb_std=2.0,
            rsi_weight=0.6, bb_weight=0.4,
            top_k=self.config.top_k, bottom_k=self.config.bottom_k,
            rebalance_bars=self.config.rebalance_bars,
            position_size=self.config.position_size, leverage=self.config.leverage,
            taker_fee=self.config.taker_fee, slippage_bps=self.config.slippage_bps,
        )
        vol_config = VolatilityConfig(
            vol_window=72, vol_ma_window=336,
            vol_rank_method='percentile',
            top_k=self.config.top_k, bottom_k=self.config.bottom_k,
            rebalance_bars=self.config.rebalance_bars,
            position_size=self.config.position_size, leverage=self.config.leverage,
            taker_fee=self.config.taker_fee, slippage_bps=self.config.slippage_bps,
        )
        volume_config = VolumeConfig(
            vol_ma_window=72, vol_breakout_threshold=1.5,
            price_window=24,
            vol_weight=0.4, divergence_weight=0.6,
            top_k=self.config.top_k, bottom_k=self.config.bottom_k,
            rebalance_bars=self.config.rebalance_bars,
            position_size=self.config.position_size, leverage=self.config.leverage,
            taker_fee=self.config.taker_fee, slippage_bps=self.config.slippage_bps,
        )
        
        mom_scores = MomentumAlpha(mom_config).compute_scores(prices, volumes)
        mr_scores = MeanReversionAlpha(mr_config).compute_scores(prices, volumes)
        vol_scores = VolatilityAlpha(vol_config).compute_scores(prices, volumes)
        volume_scores = VolumeAlpha(volume_config).compute_scores(prices, volumes)
        
        # 归一化各因子得分
        def normalize(s):
            s = s.dropna()
            if len(s) < 2:
                return s
            return (s - s.mean()) / (s.std() + 1e-10)
        
        mom_norm = normalize(mom_scores)
        mr_norm = normalize(mr_scores)
        vol_norm = normalize(vol_scores)
        volume_norm = normalize(volume_scores)
        
        # 组合得分
        combined = pd.Series(0.0, index=prices.columns)
        combined += self.config.momentum_weight * mom_norm
        combined += self.config.mean_reversion_weight * mr_norm
        combined += self.config.volatility_weight * vol_norm
        combined += self.config.volume_weight * volume_norm
        
        return combined
