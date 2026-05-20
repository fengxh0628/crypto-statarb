"""Fast vectorized ADF test for batch cointegration screening.

Computes ADF t-statistics for multiple spread series simultaneously
using batched linear algebra (numpy BLAS). ~50-100x faster than
statsmodels adfuller for large batches.

Usage:
    from cst.pairs.fast_adf import batch_adf, fast_adf_single

    # Single series
    t_stat = fast_adf_single(spread_series, lags=1)
    is_stationary = t_stat < -2.86  # 5% critical value

    # Batch: 3160 pairs at once
    spreads = np.stack([spread1, spread2, ...])  # (n_pairs, n_bars)
    t_stats = batch_adf(spreads, lags=1)
    stationary_mask = t_stats < -2.86

ADF critical values (with constant, no trend):
    1%:  -3.43
    5%:  -2.86
    10%: -2.57
"""

import numpy as np
import pandas as pd


# ADF critical values (asymptotic, with constant term)
ADF_CRITICAL_VALUES = {
    "1%": -3.43,
    "5%": -2.86,
    "10%": -2.57,
}


def fast_adf_single(series: pd.Series, lags: int = 1) -> float:
    """Fast ADF test for a single series.

    Args:
        series: Time series (e.g. spread)
        lags: Number of lagged differences to include (fixed, no autolag)

    Returns:
        ADF t-statistic. Compare with critical values:
        t < -2.86 → stationary at 5% level
    """
    y = series.dropna().values
    n = len(y)
    if n < lags + 10:
        return 0.0  # 数据不足

    dy = np.diff(y)
    m = n - 1 - lags  # 有效样本数

    # 构建回归矩阵: Δy_t = α + γ*y_{t-1} + Σ(β_i * Δy_{t-i}) + ε
    Y = dy[lags:]  # (m,)

    # X columns: [constant, y_{t-1}, Δy_{t-1}, Δy_{t-2}, ...]
    X = np.empty((m, 2 + lags))
    X[:, 0] = 1.0  # 常数项
    X[:, 1] = y[lags:-1]  # y_{t-1}
    for i in range(lags):
        X[:, 2 + i] = dy[lags - i - 1: n - 2 - i]  # Δy_{t-i-1}

    # OLS
    XtX = X.T @ X
    XtY = X.T @ Y

    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        return 0.0

    # t-statistic for gamma (beta[1])
    resid = Y - X @ beta
    k = X.shape[1]
    sigma2 = (resid @ resid) / (m - k)

    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return 0.0

    se_gamma = np.sqrt(XtX_inv[1, 1] * sigma2)
    if se_gamma == 0:
        return 0.0

    t_stat = beta[1] / se_gamma
    return t_stat


def batch_adf(spreads: np.ndarray, lags: int = 1) -> np.ndarray:
    """Vectorized batch ADF test for multiple series simultaneously.

    Uses batched matrix operations (BLAS) for maximum performance.
    ~50-100x faster than calling statsmodels adfuller in a loop.

    Args:
        spreads: 2D array of shape (n_pairs, n_bars). Each row is a spread series.
                 NaN values should be handled before calling.
        lags: Number of lagged differences (fixed for all series)

    Returns:
        1D array of ADF t-statistics, shape (n_pairs,).
        Compare with -2.86 for 5% significance.
    """
    n_pairs, n_bars = spreads.shape
    if n_bars < lags + 10:
        return np.zeros(n_pairs)

    # First differences: (n_pairs, n_bars-1)
    dy = np.diff(spreads, axis=1)
    m = n_bars - 1 - lags  # 有效样本量

    # Dependent variable Y: Δy_t for t = lags+1 ... n-1
    # Shape: (n_pairs, m)
    Y = dy[:, lags:]

    # Build regressor matrix X: (n_pairs, m, 2+lags)
    k = 2 + lags  # number of regressors
    X = np.empty((n_pairs, m, k))

    # Column 0: constant
    X[:, :, 0] = 1.0

    # Column 1: y_{t-1}
    X[:, :, 1] = spreads[:, lags:-1]

    # Columns 2...: lagged differences Δy_{t-1}, Δy_{t-2}, ...
    for i in range(lags):
        X[:, :, 2 + i] = dy[:, lags - i - 1: lags - i - 1 + m]

    # Batch OLS: solve (X'X) @ beta = X'Y for each pair
    # X'X: (n_pairs, k, k)
    XtX = np.einsum("pik,pij->pkj", X, X)

    # X'Y: (n_pairs, k, 1) — need 3D for batched solve
    XtY = np.einsum("pik,pi->pk", X, Y)

    # Solve all systems at once: (n_pairs, k, k) @ (n_pairs, k, 1) = (n_pairs, k, 1)
    try:
        betas = np.linalg.solve(XtX, XtY[:, :, None])[:, :, 0]  # (n_pairs, k)
    except np.linalg.LinAlgError:
        # Fallback: solve one by one
        betas = np.zeros((n_pairs, k))
        for i in range(n_pairs):
            try:
                betas[i] = np.linalg.solve(XtX[i], XtY[i])
            except np.linalg.LinAlgError:
                pass

    # Residuals: (n_pairs, m)
    resid = Y - np.einsum("pij,pj->pi", X, betas)

    # Residual variance: sigma^2 = ||resid||^2 / (m - k)
    sigma2 = (resid ** 2).sum(axis=1) / (m - k)  # (n_pairs,)

    # Standard error of gamma (beta[:, 1])
    # se = sqrt(diag((X'X)^{-1})[1] * sigma^2)
    try:
        XtX_inv = np.linalg.inv(XtX)  # (n_pairs, k, k)
    except np.linalg.LinAlgError:
        XtX_inv = np.zeros_like(XtX)
        for i in range(n_pairs):
            try:
                XtX_inv[i] = np.linalg.inv(XtX[i])
            except np.linalg.LinAlgError:
                pass

    var_gamma = XtX_inv[:, 1, 1] * sigma2  # (n_pairs,)
    se_gamma = np.sqrt(np.maximum(var_gamma, 1e-30))

    # t-statistics
    t_stats = betas[:, 1] / se_gamma

    return t_stats


def batch_cointegration_test(
    prices: pd.DataFrame,
    start: int,
    end: int,
    pairs: list,
    lags: int = 1,
    significance: str = "5%",
) -> dict:
    """Fast batch cointegration test using vectorized ADF.

    Tests cointegration for multiple pairs simultaneously:
    1. For each pair, compute OLS hedge ratio
    2. Compute spread = log(A) - β * log(B)
    3. Run batch ADF on all spreads at once

    Args:
        prices: Price DataFrame (columns = symbols)
        start: Start index (inclusive)
        end: End index (exclusive)
        pairs: List of (symbol_a, symbol_b) tuples to test
        lags: ADF lag order
        significance: "1%", "5%", or "10%"

    Returns:
        Dict {(sym_a, sym_b): {"t_stat": float, "stationary": bool, "hedge_ratio": float}}
    """
    if not pairs:
        return {}

    critical_value = ADF_CRITICAL_VALUES[significance]
    window_prices = prices.iloc[start:end]
    n_samples = end - start

    # 批量计算 hedge ratio 和 spread
    log_prices = np.log(window_prices.values)  # (n_samples, n_symbols)
    symbol_idx = {sym: i for i, sym in enumerate(prices.columns)}

    spreads = np.empty((len(pairs), n_samples))
    hedge_ratios = np.empty(len(pairs))

    for idx, (sym_a, sym_b) in enumerate(pairs):
        ia = symbol_idx[sym_a]
        ib = symbol_idx[sym_b]
        log_a = log_prices[:, ia]
        log_b = log_prices[:, ib]

        # OLS: log_a = α + β * log_b
        X = np.column_stack([np.ones(n_samples), log_b])
        beta = np.linalg.lstsq(X, log_a, rcond=None)[0]
        hedge_ratios[idx] = beta[1]
        spreads[idx] = log_a - beta[1] * log_b

    # 批量 ADF
    t_stats = batch_adf(spreads, lags=lags)

    # 构建结果
    results = {}
    for idx, pair in enumerate(pairs):
        is_stationary = t_stats[idx] < critical_value
        if is_stationary:
            results[pair] = {
                "t_stat": t_stats[idx],
                "stationary": True,
                "hedge_ratio": hedge_ratios[idx],
            }

    return results
