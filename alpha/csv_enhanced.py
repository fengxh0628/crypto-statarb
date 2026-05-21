"""Enhanced Cross-Sectional Volatility with Hysteresis."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.cross_sectional_vol import CrossSectionalVolAlpha, CrossSectionalVolConfig


@dataclass
class CSVEnhancedConfig(CrossSectionalVolConfig):
    """Configuration for CSV with Hysteresis."""
    hysteresis_entry_threshold: float = 0.3
    hysteresis_exit_threshold: float = 0.1


class CSVEnhancedAlpha(CrossSectionalVolAlpha):
    """Cross-Sectional Volatility with Hysteresis."""
    
    def __init__(self, config: CSVEnhancedConfig):
        super().__init__(config)
        self.config = config
        self.prev_weights = {}
    
    def generate_weights(self, prices, volumes=None):
        raw_weights = super().generate_weights(prices, volumes)
        return self._apply_hysteresis(raw_weights)
    
    def _apply_hysteresis(self, new_weights):
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
            
            threshold = entry_thresh if abs(old_w) < 1e-9 else exit_thresh
            
            if abs(delta) > threshold:
                filtered[sym] = new_w
            else:
                filtered[sym] = old_w
        
        self.prev_weights = {sym: w for sym, w in filtered.items() if abs(w) > 1e-9}
        return self.prev_weights
