"""Dynamic strategy ensemble.

Combines multiple strategies with regime-based weighting:
- Bull market: Momentum-heavy
- Bear market: Mean reversion or flat
- Range: Balanced mix
- High vol: Reduce all positions
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig
from alpha.momentum import MomentumAlpha, MomentumConfig
from alpha.mean_reversion import MeanReversionAlpha, MeanReversionConfig
from alpha.volatility import VolatilityAlpha, VolatilityConfig
from alpha.volume import VolumeAlpha, VolumeConfig
from alpha.market_regime import MarketRegime, RegimeConfig


@dataclass
class EnsembleConfig(AlphaConfig):
    """Configuration for strategy ensemble."""
    # Strategy weights by regime
    # Bull: momentum works best
    bull_weights: dict = None
    # Bear: mean reversion or flat
    bear_weights: dict = None
    # Range: balanced
    range_weights: dict = None
    
    # Volatility adjustment
    vol_adjustment: bool = True
    vol_threshold_high: float = 0.05
    vol_multiplier_low: float = 1.0
    vol_multiplier_high: float = 0.5
    
    # Minimum signal agreement
    min_agreement: int = 2  # At least N strategies must agree
    
    # Regime detection
    regime_trend_window: int = 336
    regime_trend_threshold: float = 0.0
    regime_vol_window: int = 720
    regime_mom_window: int = 720
    regime_mom_threshold: float = 0.0


class StrategyEnsemble(AlphaStrategy):
    """Dynamic strategy ensemble with regime-based weighting."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        
        # Default weights
        if config.bull_weights is None:
            self.config.bull_weights = {
                'momentum': 0.6,
                'mean_reversion': 0.1,
                'volatility': 0.1,
                'volume': 0.2,
            }
        if config.bear_weights is None:
            self.config.bear_weights = {
                'momentum': 0.2,
                'mean_reversion': 0.4,
                'volatility': 0.2,
                'volume': 0.2,
            }
        if config.range_weights is None:
            self.config.range_weights = {
                'momentum': 0.3,
                'mean_reversion': 0.3,
                'volatility': 0.2,
                'volume': 0.2,
            }
        
        # Regime detector
        self.regime_config = RegimeConfig(
            trend_window=config.regime_trend_window,
            trend_threshold=config.regime_trend_threshold,
            vol_window=config.regime_vol_window,
            vol_threshold_high=0.05,
            vol_threshold_low=0.01,
            mom_window=config.regime_mom_window,
            mom_threshold=config.regime_mom_threshold,
        )
        self.regime_detector = MarketRegime(self.regime_config)
        
        # Sub-strategies
        self.momentum = MomentumAlpha(MomentumConfig(
            momentum_window=336, skip_recent=12,
            top_k=config.top_k, bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size, leverage=config.leverage,
            taker_fee=config.taker_fee, slippage_bps=config.slippage_bps,
        ))
        self.mean_reversion = MeanReversionAlpha(MeanReversionConfig(
            rsi_window=14, rsi_overbought=70, rsi_oversold=30,
            bb_window=20, bb_std=2.0,
            rsi_weight=0.6, bb_weight=0.4,
            top_k=config.top_k, bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size, leverage=config.leverage,
            taker_fee=config.taker_fee, slippage_bps=config.slippage_bps,
        ))
        self.volatility = VolatilityAlpha(VolatilityConfig(
            vol_window=72, vol_ma_window=336,
            vol_rank_method='percentile',
            top_k=config.top_k, bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size, leverage=config.leverage,
            taker_fee=config.taker_fee, slippage_bps=config.slippage_bps,
        ))
        self.volume = VolumeAlpha(VolumeConfig(
            vol_ma_window=72, vol_breakout_threshold=1.5,
            price_window=24,
            vol_weight=0.4, divergence_weight=0.6,
            top_k=config.top_k, bottom_k=config.bottom_k,
            rebalance_bars=config.rebalance_bars,
            position_size=config.position_size, leverage=config.leverage,
            taker_fee=config.taker_fee, slippage_bps=config.slippage_bps,
        ))
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute ensemble scores."""
        # Detect regime
        regime = self.regime_detector.detect_regime(prices, volumes)
        
        # Get weights for regime
        weights = {
            'bull': self.config.bull_weights,
            'bear': self.config.bear_weights,
            'range': self.config.range_weights,
        }.get(regime, self.config.range_weights)
        
        # Compute individual scores
        scores = {}
        scores['momentum'] = self.momentum.compute_scores(prices, volumes)
        scores['mean_reversion'] = self.mean_reversion.compute_scores(prices, volumes)
        scores['volatility'] = self.volatility.compute_scores(prices, volumes)
        scores['volume'] = self.volume.compute_scores(prices, volumes)
        
        # Normalize scores
        def normalize(s):
            s = s.dropna()
            if len(s) < 2:
                return s
            return (s - s.mean()) / (s.std() + 1e-10)
        
        normalized = {k: normalize(v) for k, v in scores.items()}
        
        # Weighted combination
        combined = pd.Series(0.0, index=prices.columns)
        for name, score in normalized.items():
            combined += weights.get(name, 0) * score
        
        # Volatility adjustment
        if self.config.vol_adjustment:
            if 'BTCUSDT' in prices.columns:
                btc_vol = self.regime_detector._compute_volatility(
                    prices['BTCUSDT'], self.config.regime_vol_window
                )
                if btc_vol > self.config.vol_threshold_high:
                    combined *= self.config.vol_multiplier_high
        
        return combined
