"""Dynamic Correlation Mean Reversion.

Logic:
- Calculate rolling correlation with BTC for each symbol.
- If correlation is high (> threshold), calculate the spread (residual).
- If spread deviates significantly from mean (Z-score), trade the reversion.
- "Stat Arb" on the fly.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy, AlphaConfig


@dataclass
class CorrMRConfig(AlphaConfig):
    # Correlation parameters
    corr_window: int = 168  # 7 days
    corr_threshold: float = 0.7  # Minimum correlation to trade
    
    # Spread parameters
    spread_window: int = 72  # 3 days for mean/std
    entry_zscore: float = 2.0  # Enter when Z > 2
    
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


class CorrMRAlpha(AlphaStrategy):
    """Dynamic Correlation Mean Reversion alpha."""
    
    def __init__(self, config: CorrMRConfig):
        self.config = config
    
    def filter_universe(self, prices, volumes=None):
        """Override to keep all symbols (we need BTC for correlation)."""
        # Only drop symbols with all NaN
        return prices.columns[prices.notna().sum() > 0]
    
    def compute_scores(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute correlation mean reversion scores."""
        n = len(prices)
        
        # Debug: print first 5 calls
        if not hasattr(self, '_debug_calls'):
            self._debug_calls = 0
        if self._debug_calls < 5:
            print(f"\n  [CorrMR Debug] Call {self._debug_calls+1}: n={n}, cols={list(prices.columns)[:5]}...")
            self._debug_calls += 1

        if n < self.config.corr_window + 10:
            return pd.Series(np.nan, index=prices.columns)
        
        if 'BTCUSDT' not in prices.columns:
            # Debug: print available columns if BTC missing
            # print(f"  [CorrMR] BTCUSDT not found. Available: {prices.columns[:5]}...")
            return pd.Series(np.nan, index=prices.columns)
        
        btc = prices['BTCUSDT']
        returns = np.log(prices).diff()
        btc_ret = returns['BTCUSDT']
        
        # Calculate rolling correlation
        try:
            corr = returns.rolling(self.config.corr_window).corr(btc_ret).iloc[-1]
        except Exception as e:
            if self._debug_calls <= 5:
                print(f"  [CorrMR Error] corr calculation failed: {e}")
            return pd.Series(0.0, index=prices.columns)
        
        # Debug
        # print(f"  [CorrMR] Corr stats: min={corr.min():.3f}, max={corr.max():.3f}, NaN={corr.isna().sum()}")
        
        # Filter high correlation symbols
        # Handle NaN corr values
        high_corr = (corr > self.config.corr_threshold) & corr.notna()
        
        if self._debug_calls <= 5:
            print(f"  [CorrMR Debug] corr type={type(corr)}, shape={corr.shape if hasattr(corr, 'shape') else 'N/A'}", flush=True)
            print(f"  [CorrMR Debug] corr values: {corr.dropna().head()}", flush=True)
        
        if high_corr.sum() < 2:
            return pd.Series(0.0, index=prices.columns)
        
        # Calculate Spread (Residual) for high corr symbols
        # Simple model: Ret_sym = Beta * Ret_btc + Alpha
        # We want to trade Alpha
        # Alpha = Ret_sym - Beta * Ret_btc
        # Beta = Cov(Sym, BTC) / Var(BTC)
        
        # Fix: Select columns correctly
        cols = corr[high_corr].index
        cov = returns[cols].rolling(self.config.corr_window).cov(btc_ret).iloc[-1]
        var_btc = btc_ret.rolling(self.config.corr_window).var().iloc[-1]
        
        beta = cov / var_btc
        
        # Current residual
        current_ret = returns.iloc[-1][high_corr]
        current_btc_ret = btc_ret.iloc[-1]
        
        residual = current_ret - beta * current_btc_ret
        
        # Normalize by volatility to get "Z-score like" metric
        vol = returns[cols].rolling(self.config.corr_window).std().iloc[-1]
        
        # Z-score approx
        z_score = residual / (vol + 1e-10)
        
        # Signal: Negative Z -> Long, Positive Z -> Short
        # We want to trade the extremes
        score = pd.Series(0.0, index=prices.columns)
        score[high_corr] = -z_score  # Invert for mean reversion
        
        # REMOVED: Hard threshold filtering. 
        # Let the portfolio construction (Top/Bottom K) handle selection.
        # score[abs(score) < self.config.entry_zscore] = 0.0
        
        return score
