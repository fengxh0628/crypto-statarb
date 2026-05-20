"""Benchmark: fast_adf vs statsmodels."""

import numpy as np
import pandas as pd
import time
from statsmodels.tsa.stattools import adfuller
from cst.pairs.fast_adf import fast_adf_single, batch_adf

rng = np.random.default_rng(42)
n = 2000

# OU过程（平稳）
ou = np.zeros(n)
for i in range(1, n):
    ou[i] = ou[i-1] * 0.98 + rng.normal() * 0.01

# 随机游走（非平稳）
rw = np.cumsum(rng.normal(0, 0.01, n))

print("=== Accuracy Test ===")
print("%-18s %12s %12s %12s" % ("Series", "statsmodels", "fast_single", "batch"))
print("-" * 58)

# statsmodels
t_ou_sm = adfuller(ou, maxlag=1, autolag=None)[0]
t_rw_sm = adfuller(rw, maxlag=1, autolag=None)[0]

# fast single
t_ou_fast = fast_adf_single(pd.Series(ou), lags=1)
t_rw_fast = fast_adf_single(pd.Series(rw), lags=1)

# batch
spreads = np.stack([ou, rw])
t_batch = batch_adf(spreads, lags=1)

print("%-18s %12.4f %12.4f %12.4f" % ("OU (stationary)", t_ou_sm, t_ou_fast, t_batch[0]))
print("%-18s %12.4f %12.4f %12.4f" % ("RW (non-stat)", t_rw_sm, t_rw_fast, t_batch[1]))
print("Critical 5%%: -2.86")
print()

# 性能测试
print("=== Performance Test (1000 pairs x 2000 bars) ===")
n_pairs = 1000
spreads_large = rng.normal(0, 1, (n_pairs, 2000))
for i in range(500):
    x = np.zeros(2000)
    for j in range(1, 2000):
        x[j] = x[j-1] * 0.97 + rng.normal() * 0.01
    spreads_large[i] = x

# statsmodels loop
t0 = time.time()
results_sm = [adfuller(spreads_large[i], maxlag=1, autolag=None)[0] for i in range(n_pairs)]
time_sm = time.time() - t0

# batch
t0 = time.time()
results_batch = batch_adf(spreads_large, lags=1)
time_batch = time.time() - t0

print("statsmodels loop: %.3fs" % time_sm)
print("batch_adf:        %.3fs" % time_batch)
print("Speedup:          %.0fx" % (time_sm / time_batch))
print("Results match:    %s" % np.allclose(results_sm, results_batch, atol=1e-6))

# 3160 pairs (80 coins)
print()
print("=== Performance Test (3160 pairs x 2016 bars) ===")
spreads_3160 = rng.normal(0, 1, (3160, 2016))
t0 = time.time()
results_3160 = batch_adf(spreads_3160, lags=1)
time_3160 = time.time() - t0
print("batch_adf (3160 pairs): %.3fs" % time_3160)
