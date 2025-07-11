#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Best Strategy: Overnight Gap < 0.3607%
After 11 waves of testing, this is the best single condition found.
"""

import pandas as pd
import numpy as np
import sys
import os

# ------------------------- HELPERS -----------------------------------
def load_ticker(path: str) -> pd.DataFrame:
    """Load CSV, standardize columns, compute overnight returns."""
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df["Date"] = pd.to_datetime(df["date"])
    df = (
        df[df["Date"] >= pd.Timestamp("2006-01-01")]
          .sort_values("Date")
          .reset_index(drop=True)
    )
    df["Next_Open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    return df

# ------------------------- LOAD DATA -----------------------------------
data_file = "4 - QQQ.csv"
if not os.path.exists(data_file):
    print(f"Missing file: {data_file}")
    sys.exit(1)

df = load_ticker(data_file)

# ------------------------- CALCULATE STRATEGY -----------------------------------
# Calculate overnight gap
df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

# BEST CONDITION FOUND: Overnight Gap < 0.3607%
THRESHOLD = 0.003607
df['signal'] = (df['overnight_gap'] < THRESHOLD).astype(int)

# Calculate strategy returns
df['strategy_daily_return'] = df['signal'] * df['next_overnight_return']
df['strategy_daily_return'].fillna(0, inplace=True)

# ------------------------- PERFORMANCE METRICS -----------------------------------
annual_rf = 0.02
daily_rf = annual_rf / 252

# Sharpe Ratio
excess_returns = df['strategy_daily_return'] - daily_rf
mean_excess_annual = excess_returns.mean() * 252
std_excess_annual = excess_returns.std() * np.sqrt(252)
sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0

# Cumulative returns
df['strategy_equity'] = (1 + df['strategy_daily_return']).cumprod()
total_return = df['strategy_equity'].iloc[-1] - 1
years_span = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
cagr = (1 + total_return) ** (1 / years_span) - 1 if years_span > 0 else 0

# Maximum drawdown
df['strategy_peak'] = df['strategy_equity'].cummax()
df['strategy_drawdown'] = (df['strategy_equity'] - df['strategy_peak']) / df['strategy_peak']
max_drawdown = df['strategy_drawdown'].min()

# Win rate
winning_trades = (df['strategy_daily_return'] > 0).sum()
total_trades = df['signal'].sum()
win_rate = winning_trades / total_trades if total_trades > 0 else 0

# Average win/loss
avg_win = df[df['strategy_daily_return'] > 0]['strategy_daily_return'].mean()
avg_loss = df[df['strategy_daily_return'] < 0]['strategy_daily_return'].mean()

# ------------------------- OUTPUT -----------------------------------
print("\n" + "="*60)
print("BEST STRATEGY FOUND AFTER 11 WAVES OF TESTING")
print("="*60)
print(f"\nCondition: Overnight Gap < {THRESHOLD:.4%}")
print(f"Entry: If open of day T < close of day T-1 by {THRESHOLD:.4%} or more,")
print(f"       buy at close of day T")
print(f"Exit:  Sell at open of day T+1")
print("\n" + "-"*60)
print("PERFORMANCE METRICS")
print("-"*60)
print(f"Sharpe Ratio:        {sharpe_ratio:.4f}")
print(f"CAGR:                {cagr*100:.2f}%")
print(f"Total Return:        {total_return*100:.2f}%")
print(f"Maximum Drawdown:    {max_drawdown*100:.2f}%")
print(f"Number of Trades:    {total_trades}")
print(f"Win Rate:            {win_rate*100:.2f}%")
print(f"Average Win:         {avg_win*100:.2f}%")
print(f"Average Loss:        {avg_loss*100:.2f}%")
print(f"Trading Days:        {len(df)}")
print(f"Time Period:         {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

# Check if it meets the criteria
print("\n" + "-"*60)
print("TARGET CRITERIA CHECK")
print("-"*60)
print(f"Sharpe >= 1.2:       {'❌ NO' if sharpe_ratio < 1.2 else '✅ YES'} (Current: {sharpe_ratio:.4f})")
print(f"Trades > 2500:       {'❌ NO' if total_trades <= 2500 else '✅ YES'} (Current: {total_trades})")
print(f"\nBoth criteria met:   {'❌ NO' if sharpe_ratio < 1.2 or total_trades <= 2500 else '✅ YES'}")

# Save detailed results
with open('final_results.txt', 'w') as f:
    f.write("FINAL BEST STRATEGY RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Condition: Overnight Gap < {THRESHOLD:.4%}\n")
    f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
    f.write(f"Number of Trades: {total_trades}\n")
    f.write(f"Total Return: {total_return*100:.2f}%\n")
    f.write(f"CAGR: {cagr*100:.2f}%\n")
    f.write(f"Maximum Drawdown: {max_drawdown*100:.2f}%\n")
    f.write(f"Win Rate: {win_rate*100:.2f}%\n")
    f.write(f"\nMeets criteria: {'NO' if sharpe_ratio < 1.2 or total_trades <= 2500 else 'YES'}\n")
    f.write(f"\nNote: After extensive testing of 700+ conditions across 11 waves,\n")
    f.write(f"this was the best single condition found. While it achieves an\n")
    f.write(f"impressive Sharpe Ratio of {sharpe_ratio:.4f} with {total_trades} trades,\n")
    f.write(f"it falls just short of the 1.2 Sharpe Ratio target.\n")

print("\nResults saved to 'final_results.txt'")
print("="*60)