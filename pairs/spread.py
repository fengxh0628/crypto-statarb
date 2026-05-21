"""Spread calculation for pairs trading."""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def compute_hedge_ratio(
    prices_a: pd.Series,
    prices_b: pd.Series,
    use_log: bool = True,
) -> float:
    """Estimate hedge ratio via OLS regression."""
    if use_log:
        y = np.log(prices_a.values)
        x = np.log(prices_b.values)
    else:
        y = prices_a.values
        x = prices_b.values

    # Remove NaN
    mask = ~(np.isnan(y) | np.isnan(x))
    y = y[mask]
    x = x[mask]
    if len(y) < 10:
        return np.nan

    x_const = add_constant(x)
    try:
        model = OLS(y, x_const).fit()
        return model.params[1]
    except Exception:
        return np.nan


def compute_spread(
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: float,
    use_log: bool = True,
) -> pd.Series:
    """Compute spread given hedge ratio.

    spread = log(A) - β * log(B)  (if use_log=True)
    spread = A - β * B            (if use_log=False)
    """
    if use_log:
        spread = np.log(prices_a) - hedge_ratio * np.log(prices_b)
    else:
        spread = prices_a - hedge_ratio * prices_b
    spread.name = "spread"
    return spread


def rolling_hedge_ratio(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = 120,
    use_log: bool = True,
) -> pd.Series:
    """Compute rolling hedge ratio using expanding/rolling OLS.

    Returns Series of β values aligned to the price index.
    """
    if use_log:
        a = np.log(prices_a)
        b = np.log(prices_b)
    else:
        a = prices_a
        b = prices_b

    n = len(a)
    betas = pd.Series(np.nan, index=prices_a.index)

    for i in range(window, n):
        y = a.iloc[i - window : i].values
        x = b.iloc[i - window : i].values
        # Remove NaN
        mask = ~(np.isnan(y) | np.isnan(x))
        y_clean = y[mask]
        x_clean = x[mask]
        if len(y_clean) < 10:
            continue
        x_const = add_constant(x_clean)
        try:
            model = OLS(y_clean, x_const).fit()
            betas.iloc[i] = model.params[1]
        except Exception:
            pass

    return betas


def rolling_spread(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = 120,
    use_log: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Compute spread with rolling hedge ratio.

    Returns:
        (spread, hedge_ratios) tuple
    """
    betas = rolling_hedge_ratio(prices_a, prices_b, window, use_log)

    if use_log:
        spread = np.log(prices_a) - betas * np.log(prices_b)
    else:
        spread = prices_a - betas * prices_b

    spread.name = "spread"
    return spread, betas


def estimate_ou_params(spread: pd.Series) -> dict:
    """Estimate OU process parameters from spread series.

    Fits AR(1) model: spread[t] = a + b * spread[t-1] + noise
    Then derives OU parameters:
        theta = -ln(b)         (mean reversion speed, per bar)
        mu = a / (1 - b)      (long-term mean)
        sigma = std(residuals) * sqrt(2*theta / (1 - b^2))
        half_life = ln(2) / theta  (bars to revert halfway)

    Returns:
        Dict with keys: theta, mu, sigma, half_life, b (AR1 coeff)
    """
    spread_clean = spread.dropna()
    if len(spread_clean) < 10:
        return {
            "theta": 0.0,
            "mu": spread_clean.mean() if len(spread_clean) > 0 else 0.0,
            "sigma": spread_clean.std() if len(spread_clean) > 0 else 0.0,
            "half_life": np.inf,
            "b": 0.0,
            "stationary": False,
        }
    y = spread_clean.iloc[1:].values
    x = spread_clean.iloc[:-1].values

    x_const = add_constant(x)
    try:
        model = OLS(y, x_const).fit()
    except Exception:
        return {
            "theta": 0.0,
            "mu": spread_clean.mean(),
            "sigma": spread_clean.std(),
            "half_life": np.inf,
            "b": 0.0,
            "stationary": False,
        }

    a = model.params[0]  # intercept
    b = model.params[1]  # AR(1) coefficient

    if b <= 0 or b >= 1:
        # 非平稳或边界情况
        return {
            "theta": 0.0,
            "mu": spread_clean.mean(),
            "sigma": spread_clean.std(),
            "half_life": np.inf,
            "b": b,
            "stationary": False,
        }

    theta = -np.log(b)
    mu = a / (1.0 - b)
    residual_std = model.resid.std()
    sigma = residual_std * np.sqrt(2.0 * theta / (1.0 - b**2))
    half_life = np.log(2) / theta

    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
        "b": b,
        "stationary": True,
    }
