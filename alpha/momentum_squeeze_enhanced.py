"""Enhanced Momentum + Volatility Squeeze Ensemble.

Adds:
1. Hysteresis Entry/Exit: Reduces whipsaw trades by requiring signal changes to exceed a threshold.
2. Volatility Targeting: Scales position sizes inversely to recent realized volatility.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig
from alpha.momentum_squeeze_ensemble import MomentumSqueezeEnsemble, MomentumSqueezeConfig


@dataclass
class EnhancedEnsembleConfig(MomentumSqueezeConfig):
    """Configuration for Enhanced Ensemble with Hysteresis and Vol Targeting."""
    # Hysteresis parameters
    hysteresis_entry_threshold: float = 0.3  # Min signal change to open/increase position (fraction of pos_size)
    hysteresis_exit_threshold: float = 0.1   # Min signal change to reduce/close position (fraction of pos_size)
    
    # Volatility targeting parameters
    vol_target: float = 0.15                 # Target annualized portfolio volatility (15%)
    vol_lookback: int = 720                  # Lookback window for realized vol (30 days)
    vol_scaling_min: float = 0.5             # Minimum vol scaling factor
    vol_scaling_max: float = 1.5             # Maximum vol scaling factor


class EnhancedMomentumSqueeze(MomentumSqueezeEnsemble):
    """Momentum + Squeeze with Hysteresis and Volatility Targeting."""
    
    def __init__(self, config: EnhancedEnsembleConfig):
        super().__init__(config)
        self.config = config
        self.prev_weights = {}  # Store previous weights for hysteresis
        self.return_history = []  # Store recent returns for vol targeting
    
    def generate_weights(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Generate combined weights with hysteresis and vol targeting."""
        # Get raw weights from base ensemble
        raw_weights = super().generate_weights(prices, volumes)
        
        # Apply hysteresis filter
        filtered_weights = self._apply_hysteresis(raw_weights)
        
        # Apply volatility targeting
        final_weights = self._apply_vol_targeting(filtered_weights, prices)
        
        # Update state
        self.prev_weights = final_weights.copy()
        
        # Track returns for vol calculation
        if len(prices) >= 2:
            # Portfolio return approximation (equal weight of all symbols)
            ret = prices.iloc[-1].pct_change().mean()
            if not np.isnan(ret):
                self.return_history.append(ret)
                # Keep only recent history
                if len(self.return_history) > self.config.vol_lookback:
                    self.return_history = self.return_history[-self.config.vol_lookback:]
        
        return final_weights
    
    def _apply_hysteresis(self, new_weights: dict) -> dict:
        """Apply hysteresis filter to reduce unnecessary rebalancing."""
        if not self.prev_weights:
            return new_weights
        
        filtered = {}
        entry_thresh = self.config.hysteresis_entry_threshold * self.config.position_size
        exit_thresh = self.config.hysteresis_exit_threshold * self.config.position_size
        
        all_symbols = set(new_weights.keys()) | set(self.prev_weights.keys())
        
        for sym in all_symbols:
            new_w = new_weights.get(sym, 0.0)
            old_w = self.prev_weights.get(sym, 0.0)
            delta = new_w - old_w
            
            # Determine threshold based on whether we're opening/increasing or closing/reducing
            if abs(old_w) < 1e-9:  # Currently flat
                threshold = entry_thresh
            elif (new_w > old_w > 0) or (new_w < old_w < 0):  # Increasing position
                threshold = entry_thresh
            else:  # Reducing or reversing position
                threshold = exit_thresh
            
            # Only update if change exceeds threshold
            if abs(delta) > threshold:
                filtered[sym] = new_w
            else:
                filtered[sym] = old_w
        
        # Remove near-zero weights
        return {sym: w for sym, w in filtered.items() if abs(w) > 1e-9}
    
    def _apply_vol_targeting(self, weights: dict, prices: pd.DataFrame) -> dict:
        """Scale weights based on recent realized volatility."""
        if len(self.return_history) < 30:  # Need at least 30 bars
            return weights
        
        # Calculate annualized realized volatility
        recent_returns = np.array(self.return_history)
        daily_vol = np.std(recent_returns) * np.sqrt(24)  # 24h per day
        ann_vol = daily_vol * np.sqrt(365)
        
        if ann_vol < 1e-9:
            return weights
        
        # Calculate scaling factor
        scale = self.config.vol_target / ann_vol
        
        # Clamp scaling factor
        scale = np.clip(scale, self.config.vol_scaling_min, self.config.vol_scaling_max)
        
        # Apply scaling
        return {sym: w * scale for sym, w in weights.items()}
