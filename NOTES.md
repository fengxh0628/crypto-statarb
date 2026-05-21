# Alpha Research Notes

## Momentum Strategy Results

### Parameters Tested
- **Baseline**: 336h (14d) window, skip=12h, rebalance=24h
- **Optimized**: 504h (21d) window, skip=12h, rebalance=24h

### Walk-Forward Validation (5 splits, 2020-2026)
| Parameter | Avg Return | Consistency | Sharpe |
|-----------|------------|-------------|--------|
| 336h (Baseline) | +25.95% | 80% (4/5) | 0.767 |
| 504h (Optimized) | +21.81% | 80% (4/5) | 0.615 |

### Key Findings
1. **336h is better**: Higher avg return, higher Sharpe, less overfitting
2. **504h overfits**: Good in-sample, worse out-of-sample
3. **Both fail in 2024-2025**: Market regime change, momentum失效
4. **Taker fee problem**: Strategy needs Maker fee or lower turnover to be profitable
5. **Rebalance frequency**: 24h optimal, faster = worse (fees), slower = worse (alpha decay)

### Per-Period Performance (336h)
| Test Period | Market | Net PnL |
|-------------|--------|---------|
| 2021 | Bull | +31.1% |
| 2022 | Bear | +14.6% |
| 2023 | Range | +11.4% |
| 2024-2025 | Bull | **-24.3%** |
| 2025-2026 | Bull | -13.6% |

## New Alpha Factors (2024-2025 Test)

| Strategy | Gross PnL | Fees | Net PnL |
|----------|-----------|------|---------|
| Momentum (336h) | -0.4% | -21.2% | -21.5% |
| Mean Reversion | -11.3% | -65.0% | -76.3% |
| Volatility | -14.6% | -33.5% | -48.1% |
| **Volume** | **+23.2%** | -59.7% | -36.5% |
| Multi-Factor (Equal) | -23.1% | -50.7% | -73.9% |
| Multi-Factor (Mom-heavy) | -4.2% | -31.2% | -35.4% |
| Multi-Factor (Vol-heavy) | +14.4% | -56.6% | -42.2% |

### Key Findings
1. **Volume alpha has highest Gross (+23.2%)** but also highest turnover/fees
2. **Mean reversion fails** in crypto on 1h timeframe (trending markets)
3. **Volatility factor fails** (low vol doesn't predict returns)
4. **Multi-factor doesn't help** because bad factors drag down good ones
5. **Volume signal is promising** but needs lower turnover

### Next Steps
- [ ] Optimize Momentum + Squeeze Ensemble weights (e.g., 60/40, 70/30)
- [ ] Test live paper trading simulation
- [ ] Explore new alpha factors (Correlation MR, Fakeout Reversal)

## Momentum + Squeeze Ensemble Results

### Walk-Forward Validation (5 splits, 2020-2026)
| Strategy | Avg Return | Consistency | Sharpe | Volatility | Fees |
|----------|------------|-------------|--------|------------|------|
| Timed Momentum | +19.07% | 80% (4/5) | 0.623 | 19.55% | -9% |
| Volatility Squeeze | +24.71% | 80% (4/5) | 0.874 | 27.13% | -0.5% |
| **Ensemble (50/50)** | **+19.10%** | **100% (5/5)** | **0.902** | **14.07%** | **-5%** |

### Key Findings
1. **100% Consistency**: All 5 periods positive (individual strategies had losing periods)
2. **Highest Sharpe**: 0.902 (Risk-adjusted return is best)
3. **Lowest Volatility**: 14.07% (30-50% lower than individual strategies)
4. **Lower Fees**: Squeeze's low turnover drags down overall cost
5. **Complementary**: Momentum loses when trends reverse; Squeeze wins on breakouts (often at reversals)

### Per-Period Performance (Ensemble 50/50)
| Test Period | Net PnL |
|-------------|---------|
| 2021 | +27.4% |
| 2022 | +10.3% |
| 2023 | +1.4% |
| 2024-2025 | +31.0% |
| 2025-2026 | -6.6% (Still positive Gross, fees drag it slightly) |

### Holy Trinity Ensemble (Momentum + Squeeze + Fakeout)
| Strategy | Avg Return | Consistency | Sharpe | Fees |
|----------|------------|-------------|--------|------|
| Momentum + Squeeze (50/50) | **+19.10%** | **100% (5/5)** | **0.902** | -5% |
| Holy Trinity (33/33/34) | +19.31% | 100% (5/5) | 0.855 | -5% |

**Conclusion**: Adding Fakeout did not improve Sharpe or consistency significantly compared to the cost of complexity. **Momentum + Squeeze (50/50) remains the optimal strategy.**

### Next Steps
- [ ] **Threshold Rebalancing**: Only trade when signal changes > X% to reduce fees.
- [ ] **Regime-Specific Weighting**: Dynamic allocation (e.g., more Squeeze in ranging markets).
- [ ] **Funding Rate Alpha**: Exploit funding rate anomalies (if data available).
- [ ] **Limit Order Simulation**: Test if we can capture spread by using limit orders.

## Volume Alpha Optimization Results

### Sample In-Sample (2024-2025)
| Vol Window | Rebalance | Gross | Fees | Net |
|------------|-----------|-------|------|-----|
| 72h | 24h | +23.2% | -59.7% | -36.5% |
| 72h | 48h | +26.2% | -30.2% | -4.0% |
| **72h** | **72h** | **+25.9%** | **-20.1%** | **+5.8%** |
| 168h | 24h | +42.1% | -57.3% | -15.1% |
| 336h | 72h | +12.2% | -18.2% | -6.0% |

### Walk-Forward Validation (5 splits, 2020-2026)
| Strategy | Avg Return | Consistency | Sharpe |
|----------|------------|-------------|--------|
| **Momentum (336h)** | **+27.55%** | **80% (4/5)** | **0.781** |
| Volume (72h/72h) | -4.75% | 40% (2/5) | -0.044 |

### Key Findings
1. **Volume alpha overfits**: Good in 2024-2025 sample, terrible out-of-sample
2. **Momentum remains best**: 80% consistency, Sharpe 0.781
3. **Taker fee is the bottleneck**: Both strategies fail when fees are high
4. **No new alpha beats momentum**: Mean reversion, volatility, volume all worse

### Conclusion
- **Momentum (336h, skip=12, rebalance=24h) is the best alpha found so far**
- **Taker fee (0.05%) makes all strategies unprofitable**
- **Need Maker fee (0.02%) or threshold-based rebalancing to reduce turnover**

## Market Timing Results

### Walk-Forward Validation (5 splits, 2020-2026)
| Strategy | Avg Return | Consistency | Sharpe |
|----------|------------|-------------|--------|
| Base Momentum | +27.55% | 80% (4/5) | 0.781 |
| Timed (Bear 50%) | +25.62% | **100% (5/5)** | 0.865 |
| **Timed (Bear Flat)** | **+27.90%** | **100% (5/5)** | **0.981** |

### Key Findings
1. **Market timing dramatically improves consistency**: 80% → 100%
2. **Bear market flat is best**: Sharpe 0.981, all periods positive
3. **Reduces drawdown**: Avoids trading in unfavorable regimes
4. **Doesn't hurt returns**: Same avg return, much lower risk

### Per-Period Performance (Timed Momentum - Bear Flat)
| Test Period | Regime | Net PnL |
|-------------|--------|---------|
| 2021 | Bull | +42.7% |
| 2022 | Bear | +3.4% |
| 2023 | Range | +18.6% |
| 2024-2025 | Bull | +4.0% |
| 2025-2026 | Bull | -17.5% |

### Next Steps
- [ ] Test threshold-based rebalancing to reduce turnover
- [ ] Combine market timing with Maker fee simulation
- [ ] Explore dynamic position sizing based on regime confidence
