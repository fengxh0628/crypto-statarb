"""Pair selection via cointegration and correlation analysis."""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .spread import compute_hedge_ratio, compute_spread, estimate_ou_params


def test_cointegration(
    prices_a: pd.Series,
    prices_b: pd.Series,
    significance: float = 0.05,
) -> dict:
    """Test cointegration between two price series using Engle-Granger method.

    Returns dict with test statistic, p-value, and whether cointegrated.
    """
    stat, pvalue, crit_values = coint(prices_a.values, prices_b.values)
    return {
        "statistic": stat,
        "pvalue": pvalue,
        "critical_values": dict(zip(["1%", "5%", "10%"], crit_values)),
        "cointegrated": pvalue < significance,
    }


def test_spread_stationarity(spread: pd.Series, significance: float = 0.05) -> dict:
    """Test if spread is stationary using ADF test."""
    result = adfuller(spread.dropna(), autolag="AIC")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "stationary": result[1] < significance,
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
    min_correlation: float = 0.7,
    max_half_life: float = np.inf,
) -> list:
    """Find all cointegrated pairs from a price DataFrame.

    Args:
        prices: DataFrame where each column is a symbol's close price
        significance: p-value threshold for cointegration test
        min_correlation: minimum correlation filter (pre-screen)
        max_half_life: maximum half-life in bars (filter out slow-reverting pairs)

    Returns:
        List of dicts with pair info, sorted by half_life (shortest first)
    """
    symbols = prices.columns.tolist()
    n = len(symbols)
    results = []

    # 预筛选：相关系数
    corr_matrix = prices.corr()

    for i, j in combinations(range(n), 2):
        sym_a, sym_b = symbols[i], symbols[j]

        # 跳过低相关性配对
        if abs(corr_matrix.iloc[i, j]) < min_correlation:
            continue

        # 协整检验
        coint_result = test_cointegration(prices[sym_a], prices[sym_b], significance)

        if not coint_result["cointegrated"]:
            continue

        # 估计OU参数（theta, half_life）
        hr = compute_hedge_ratio(prices[sym_a], prices[sym_b])
        spread = compute_spread(prices[sym_a], prices[sym_b], hr)
        ou = estimate_ou_params(spread)

        # 过滤半衰期过长的配对
        if ou["half_life"] > max_half_life:
            continue

        results.append(
            {
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "correlation": corr_matrix.iloc[i, j],
                "coint_pvalue": coint_result["pvalue"],
                "coint_statistic": coint_result["statistic"],
                "hedge_ratio": hr,
                "half_life": ou["half_life"],
                "theta": ou["theta"],
                "ou_mu": ou["mu"],
                "ou_sigma": ou["sigma"],
            }
        )

    # 按半衰期排序（越短越好 → 回归越快 → 套利机会越好）
    results.sort(key=lambda x: x["half_life"])
    return results


def rolling_cointegration(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = 1440,
    step: int = 60,
) -> pd.DataFrame:
    """Rolling cointegration test over time.

    Useful for monitoring pair stability.

    Args:
        window: Rolling window size (bars)
        step: Step size between tests (bars)

    Returns:
        DataFrame with columns [timestamp, pvalue, stationary]
    """
    n = len(prices_a)
    records = []

    for end in range(window, n, step):
        start = end - window
        a = prices_a.iloc[start:end]
        b = prices_b.iloc[start:end]

        try:
            result = test_cointegration(a, b)
            records.append(
                {
                    "datetime": prices_a.index[end - 1],
                    "pvalue": result["pvalue"],
                    "cointegrated": result["cointegrated"],
                }
            )
        except Exception:
            continue

    return pd.DataFrame(records).set_index("datetime") if records else pd.DataFrame()
