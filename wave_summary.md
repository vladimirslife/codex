# Trading Strategy Search - Wave 18-20 Results

## Wave 18: Volume-Weighted Indicators Extended
**Best Sharpe**: 0.464 (VWAP_ZSCORE)
- VWAP_ZSCORE with period 50, k=-0.5: Sharpe 0.464, 3017 trades
- VOL_FLOW_INDEX (Volume Flow Index): Sharpe 0.463, 5634 trades
- Performance significantly declined from previous waves

**Key Findings**:
- Z-score based VWAP indicators underperformed expectations
- Volume flow indicators showed modest performance
- None approached the target Sharpe of 1.3

## Wave 19: Market Microstructure Indicators
**Best Sharpe**: 0.655 (VWAP_ACCEL)
- VWAP_ACCEL (VWAP Acceleration) with period 100: Sharpe 0.655, 7557 trades
- Intraday momentum indicators showed promise
- Volume-price correlation strategies underperformed

**Key Findings**:
- VWAP acceleration (second derivative) showed improvement
- High trade frequency (7557) with moderate Sharpe
- Market microstructure approach didn't achieve breakthrough

## Wave 20: Regime Detection & Market State
**Best Sharpe**: 0.765 (MOMENTUM_DIV)
- MOMENTUM_DIV (short 5, long 100): Sharpe 0.765, 5697 trades
- Market tension indicator also performed well
- Regime-based strategies showed mixed results

**Key Findings**:
- Momentum divergence between timeframes showed promise
- Still well below target Sharpe of 1.3
- Market state indicators didn't provide the breakthrough needed

## Overall Summary After 20 Waves

### Best Performers Across All Waves:
1. **EMA_GAP/EMA_RATIO** (Wave 4-6): Sharpe ~0.950, 5855-6044 trades
2. **VWAP_ANCHORED** (Wave 17): Sharpe 0.926, 6493 trades
3. **VWAP_TREND** (Wave 16): Sharpe 0.920, 6355 trades
4. **EMA_CROSS** (Wave 3): Sharpe 0.928, 6087 trades

### Key Insights:
1. **Best overall strategy**: EMA-based gap/ratio conditions consistently outperformed
2. **Volume indicators**: VWAP variants showed strong performance but couldn't exceed 0.95 Sharpe
3. **Optimal parameters**: 150-200 day lookback periods consistently optimal
4. **Trade frequency**: Best strategies maintained 5500-6500 trades
5. **Counter-intuitive**: Buying below moving averages/VWAP often outperformed buying above

### Challenges:
- Target Sharpe Ratio of 1.3 remains unachieved
- Single condition constraint limits strategy sophistication
- Market efficiency may prevent such high Sharpe ratios with simple rules
- Best achieved Sharpe ~0.95 appears to be a ceiling for single-condition strategies

### Conclusion:
After 20 waves of systematic testing, the target Sharpe Ratio of 1.3 with >3000 trades using a single technical condition appears unattainable. The best performing strategies cluster around 0.90-0.95 Sharpe, suggesting this may be the practical limit for single-condition, long-only overnight strategies on SPY.