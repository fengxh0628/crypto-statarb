"""Walk-forward validation for alpha strategies.

Splits data into rolling train/test periods to validate out-of-sample performance.
"""

from typing import Optional

import numpy as np
import pandas as pd

from alpha.base import AlphaStrategy


def run_walk_forward(
    prices: pd.DataFrame,
    strategy_class,
    config_class,
    base_config: dict,
    n_splits: int = 5,
    train_bars: int = 8760,  # 1 year of 1h data
    volumes: Optional[pd.DataFrame] = None,
) -> dict:
    """Run walk-forward validation.
    
    Args:
        prices: DataFrame of close prices
        strategy_class: AlphaStrategy subclass
        config_class: Config dataclass
        base_config: Base configuration dict
        n_splits: Number of train/test splits
        train_bars: Training window size in bars
        volumes: DataFrame of volumes
        
    Returns:
        Dict with:
        - results: List of per-period results
        - combined_equity: Combined equity curve
        - summary: Summary statistics
    """
    n_bars = len(prices)
    test_bars = (n_bars - train_bars) // n_splits
    
    if test_bars < 1000:
        raise ValueError(f"Test period too short: {test_bars} bars")
    
    results = []
    all_equity = []
    
    print(f"\n  Walk-Forward Validation: {n_splits} splits")
    print(f"  Train: {train_bars} bars, Test: {test_bars} bars")
    print(f"  Total: {n_bars} bars from {prices.index[0]} to {prices.index[-1]}")
    
    for i in range(n_splits):
        # Define train/test periods
        train_start = max(0, i * test_bars)
        train_end = train_start + train_bars
        test_start = train_end
        test_end = min(test_start + test_bars, n_bars)
        
        if test_end - test_start < 100:
            continue
        
        train_prices = prices.iloc[train_start:train_end]
        test_prices = prices.iloc[test_start:test_end]
        train_volumes = volumes.iloc[train_start:train_end] if volumes is not None else None
        test_volumes = volumes.iloc[test_start:test_end] if volumes is not None else None
        
        print(f"\n  Split {i+1}/{n_splits}:")
        print(f"    Train: {train_prices.index[0]} to {train_prices.index[-1]}")
        print(f"    Test:  {test_prices.index[0]} to {test_prices.index[-1]}")
        
        # Create strategy with config
        config = config_class(**base_config)
        strategy = strategy_class(config)
        
        # Run backtest on test period
        from alpha.backtest import run_alpha_backtest
        result = run_alpha_backtest(test_prices, strategy, volumes=test_volumes)
        
        results.append({
            'split': i + 1,
            'train_period': (train_prices.index[0], train_prices.index[-1]),
            'test_period': (test_prices.index[0], test_prices.index[-1]),
            'metrics': result['metrics'],
            'equity_curve': result['equity_curve'],
        })
        
        all_equity.append(result['equity_curve'])
    
    # Combine equity curves
    if all_equity:
        combined_equity = pd.concat(all_equity)
    else:
        combined_equity = pd.Series(dtype=float)
    
    # Summary statistics
    returns = [r['metrics']['total_return'] for r in results]
    sharpe_ratios = [r['metrics'].get('sharpe_ratio', 0) for r in results]
    
    summary = {
        'n_splits': len(results),
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'positive_periods': sum(1 for r in returns if r > 0),
        'avg_sharpe': np.mean(sharpe_ratios),
        'consistency': sum(1 for r in returns if r > 0) / len(results) if results else 0,
    }
    
    print(f"\n  === Walk-Forward Summary ===")
    print(f"  Periods tested:     {summary['n_splits']}")
    print(f"  Avg Return:         {summary['avg_return']:+.2%}")
    print(f"  Std Return:         {summary['std_return']:.2%}")
    print(f"  Min/Max Return:     {summary['min_return']:+.2%} / {summary['max_return']:+.2%}")
    print(f"  Positive Periods:   {summary['positive_periods']}/{summary['n_splits']} ({summary['consistency']:.0%})")
    print(f"  Avg Sharpe:         {summary['avg_sharpe']:.3f}")
    print(f"  ===========================\n")
    
    return {
        'results': results,
        'combined_equity': combined_equity,
        'summary': summary,
    }
