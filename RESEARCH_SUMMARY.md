# Crypto Alpha Research Summary

## 1. Environment & Setup
- **Data Source**: Binance Futures (UM) 1h Klines.
- **Universe**: 50 Mainstream Symbols (BTC, ETH, SOL, etc.).
- **Backtest Period**: 2020-01-01 to 2026-04-30 (Walk-Forward: 5 splits, 1y train / 1y test).
- **Fee Model**:
  - **Taker**: 0.05% (0.0005).
  - **Maker**: 0.02% (0.0002).
  - **Slippage**: 1 bps.
  - **Funding**: 0.01% per 8h (approx).
- **Engine Update**: Fixed turnover calculation to use **absolute weight changes** (`sum(|new_w - old_w|)`) instead of symbol count, providing accurate fee estimation.

---

## 2. Effective Alphas

### 🏆 Champion: Momentum + Volatility Squeeze Ensemble (50/50)
- **Logic**:
  - **Momentum**: 336h (14d) window, 12h skip. Captures trends.
  - **Squeeze**: Bollinger Band width compression + Breakout. Captures volatility expansion.
- **Performance (Walk-Forward)**:
  - **Avg Return**: **+17.24%**
  - **Consistency**: **80%** (4/5 periods positive).
  - **Sharpe**: **0.800**.
  - **Fees**: ~8.1% (Taker).
- **Why it works**: Low correlation between strategies. Momentum wins in trends; Squeeze wins in breakouts/ranges.
- **Status**: **Production Ready**.

### ⚠️ Conditional: Cross-Sectional Volatility (CSV)
- **Logic**: Short highest realized vol, Long lowest realized vol (Mean Reversion of Vol).
- **Performance**:
  - **Bull Markets**: **-33% to -37%** (Fails as high-vol assets keep rallying).
  - **Bear/Ranging**: **+18% to +25%** (Works as volatility reverts).
- **Status**: **Not standalone**. Potential use as a **Risk Filter** (reduce exposure when vol expands).

---

## 3. Failed / Abandoned Strategies

| Strategy | Reason for Failure |
|----------|--------------------|
| **Correlation MR** | Liquidated in strong trends. High correlation means everything moves together; betting against outliers fails in momentum-driven markets. |
| **Fakeout Reversal** | Signal too weak. When combined with Momentum, it was drowned out. |
| **Holy Trinity** | Adding a 3rd strategy increased fees without adding uncorrelated alpha. |
| **Basis / Funding** | Data format issues (`premiumIndexKlines` structure mismatch). |
| **Hysteresis / Vol Targeting** | Reduced fees but reduced Gross PnL even more. Net negative impact on Momentum. |

---

## 4. Key Findings

1. **Turnover Matters**: The previous "symbol count" fee model underestimated costs. Accurate weight-based turnover shows fees consume ~40-50% of Gross PnL for active strategies.
2. **Rebalancing Frequency**:
   - **Momentum**: Optimal at **24h**. Faster = fee drag; Slower = alpha decay.
   - **CSV**: Optimal at **120h (5 days)**. Volatility is persistent; frequent rebalancing hurts performance.
3. **Regime Sensitivity**:
   - Momentum works in Bull/Bear but suffers in Chop.
   - CSV works in Chop/Bear but dies in Bull.
   - **Conclusion**: Combining them might smooth equity, but CSV drags down returns in Bull markets.

---

## 5. Next Steps

### A. Optimize Current Champion (Momentum + Squeeze)
1. **Regime Filter**: Use CSV logic as a filter.
   - *Idea*: If Market Volatility > Threshold (Bull run phase), **reduce Squeeze weight** or **flatten Squeeze**.
   - *Goal*: Avoid the "Shorting High Vol" trap during parabolic moves.
2. **Dynamic Weighting**:
   - *Idea*: Weight based on recent volatility-adjusted returns of each sub-strategy.
3. **Maker Fee Simulation**:
   - *Idea*: Test if entry signals allow for Limit Orders (Maker fee 0.02%).
   - *Impact*: Could reduce fees from 8% -> 3%, boosting Net PnL to ~22%+.

### B. New Alpha Directions
1. **Funding Rate Arbitrage**:
   - Fix data loading.
   - Test: Short when Funding > X (crowded longs), Long when Funding < Y.
2. **Liquidity Sweep / Stop Hunt**:
   - Detect rapid wicks (High/Low rejection) on lower timeframes (if data available) or approximate with 1h OHLC.

### C. Infrastructure
1. **Fix Data Loaders**: Ensure `premiumIndexKlines` and `markPrice` are correctly aligned with 1h candles.
2. **Limit Order Engine**: Add simulation for Maker fees to see if strategies survive better with passive execution.
