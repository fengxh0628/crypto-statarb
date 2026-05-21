"""Backtest configuration."""

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    # 配对参数
    lookback_window: int = 60  # 协整检验回看窗口（bar数）
    zscore_window: int = 30  # z-score滚动窗口
    hedge_ratio_window: int = 120  # hedge ratio 估计窗口

    # 交易参数
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_threshold: float = 4.0
    max_hold_bars: int = 480  # 最大持仓时间（bar数）

    # 成本参数
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    slippage_bps: float = 1.0  # 滑点 basis points

    # Funding rate 参数
    funding_rate: float = 0.0001  # 默认 0.01% 每次结算
    funding_interval_bars: int = 96  # 结算间隔（5min bar: 8h = 96 bars）

    # 资金管理
    capital: float = 10000.0  # USDT
    position_size: float = 0.25  # 每对使用保证金占比
    max_pairs: int = 4  # 同时最大持仓配对数
    leverage: float = 3.0  # 杠杆倍数（影响实际敞口和爆仓距离）

    # 动态回测参数
    coint_lookback: int = 2016  # 协整检验回看窗口（5min bar: 7天=2016）
    coint_recheck_bars: int = 288  # 全量重新扫描间隔（5min: 1天=288）
    coint_significance: float = 0.05  # 协整p-value阈值
    max_half_life: float = 200.0  # 最大半衰期（bars），超过则不选

    # 过滤条件
    vol_filter_window: int = 48  # 波动率过滤窗口
    min_vol_percentile: float = 0.2  # 排除低波动率时段
    trend_filter_window: int = 168  # 趋势过滤窗口
    max_trend_threshold: float = 0.02  # 排除强趋势时段（绝对收益率）
