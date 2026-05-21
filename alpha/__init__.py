"""Crypto Alpha Backtesting Framework."""

from alpha.momentum import MomentumAlpha, MomentumConfig
from alpha.mean_reversion import MeanReversionAlpha, MeanReversionConfig
from alpha.volatility import VolatilityAlpha, VolatilityConfig
from alpha.volume import VolumeAlpha, VolumeConfig
from alpha.multi_factor import MultiFactorAlpha, MultiFactorConfig
from alpha.ensemble import StrategyEnsemble, EnsembleConfig
from alpha.timed_momentum import TimedMomentumAlpha, TimedMomentumConfig
from alpha.vol_breakout import VolBreakoutAlpha, VolBreakoutConfig
from alpha.multi_tf_momentum import MultiTFMomentumAlpha, MultiTFMomentumConfig
from alpha.rel_strength_mr import RelStrengthMRAlpha, RelStrengthMRConfig
from alpha.trend_pullback import TrendPullbackAlpha, TrendPullbackConfig
from alpha.vol_squeeze import VolSqueezeAlpha, VolSqueezeConfig
from alpha.momentum_squeeze_ensemble import MomentumSqueezeEnsemble, MomentumSqueezeConfig
from alpha.correlation_mr import CorrMRAlpha, CorrMRConfig
from alpha.fakeout_reversal import FakeoutAlpha, FakeoutConfig
from alpha.holy_trinity import HolyTrinityEnsemble, HolyTrinityConfig
from alpha.backtest import run_alpha_backtest
from alpha.walk_forward import run_walk_forward

__all__ = [
    'MomentumAlpha', 'MomentumConfig',
    'MeanReversionAlpha', 'MeanReversionConfig',
    'VolatilityAlpha', 'VolatilityConfig',
    'VolumeAlpha', 'VolumeConfig',
    'MultiFactorAlpha', 'MultiFactorConfig',
    'StrategyEnsemble', 'EnsembleConfig',
    'TimedMomentumAlpha', 'TimedMomentumConfig',
    'VolBreakoutAlpha', 'VolBreakoutConfig',
    'MultiTFMomentumAlpha', 'MultiTFMomentumConfig',
    'RelStrengthMRAlpha', 'RelStrengthMRConfig',
    'TrendPullbackAlpha', 'TrendPullbackConfig',
    'VolSqueezeAlpha', 'VolSqueezeConfig',
    'MomentumSqueezeEnsemble', 'MomentumSqueezeConfig',
    'CorrMRAlpha', 'CorrMRConfig',
    'FakeoutAlpha', 'FakeoutConfig',
    'HolyTrinityEnsemble', 'HolyTrinityConfig',
    'run_alpha_backtest',
    'run_walk_forward',
]
