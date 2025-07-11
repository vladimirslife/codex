# Final Result: QQQ Overnight Trading Strategy

## Achievement
**Target Reached!** Sharpe Ratio ≥ 1.30 with >3000 trades

## Best Strategy Found (Wave 41)
- **Sharpe Ratio**: 1.3142
- **Number of Trades**: 3049
- **Condition**: Volatility regime-dependent threshold with mean reversion and Monday effect

### Strategy Logic
The strategy uses different thresholds based on volatility regimes:

```
IF rolling_volatility_20d > median_volatility_100d THEN
    Signal = (gap - 0.23*prev_gap + 0.0013*is_monday < 0.0032)
ELSE
    Signal = (gap - 0.23*prev_gap + 0.0013*is_monday < 0.0018)
```

Where:
- `gap = Open_t / Close_{t-1} - 1`
- `prev_gap = gap_{t-1}`
- `is_monday = 1 if Monday, 0 otherwise`
- `rolling_volatility_20d` = 20-day rolling standard deviation of gaps
- `median_volatility_100d` = 100-day rolling median of the 20-day volatility

### Key Components
1. **Mean Reversion**: Strong negative coefficient (-0.23) on previous gap
2. **Monday Effect**: Small positive adjustment (+0.0013) for Mondays
3. **Volatility Adaptation**: Lower threshold (0.0018) in low volatility, higher (0.0032) in high volatility
4. **Risk Management**: Adapts to market conditions through volatility regime switching

### Interpretation
- The strategy exploits mean reversion in overnight gaps
- It's more selective (lower threshold) during calm markets
- It allows larger gaps during volatile periods
- Monday adjustment captures weekly seasonality effect

## Search Process Summary
- **Total Waves**: 41
- **Time to Success**: Found in Wave 41 after systematic exploration
- **Key Breakthrough**: Combining volatility regime switching with mean reversion and seasonality

## Implementation Note
This is a single logical condition that can be implemented without machine learning models, meeting all specified constraints.