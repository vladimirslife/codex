#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 9: Range expansion/contraction patterns and extreme conditions
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

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

def calculate_sharpe(returns, annual_rf=0.02):
    """Calculate Sharpe Ratio"""
    daily_rf = annual_rf / 252
    excess_returns = returns - daily_rf
    mean_excess_annual = excess_returns.mean() * 252
    std_excess_annual = excess_returns.std() * np.sqrt(252)
    return mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0

# ------------------------- LOAD DATA -----------------------------------
data_file = "4 - QQQ.csv"
if not os.path.exists(data_file):
    print(f"Missing file: {data_file}")
    sys.exit(1)

df = load_ticker(data_file)

# ------------------------- CALCULATE INDICATORS -------------------
# Core returns
df['daily_return'] = df['close'].pct_change()
df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
df['intraday_return'] = (df['close'] - df['open']) / df['open']

# Range metrics
df['range'] = (df['high'] - df['low']) / df['close']
df['true_range'] = pd.concat([
    df['high'] - df['low'],
    abs(df['high'] - df['close'].shift(1)),
    abs(df['low'] - df['close'].shift(1))
], axis=1).max(axis=1) / df['close']

# Range patterns
df['prev_range'] = df['range'].shift(1)
df['range_expansion'] = df['range'] / (df['prev_range'] + 0.0001)
df['range_3d_avg'] = df['range'].rolling(window=3).mean()
df['range_10d_avg'] = df['range'].rolling(window=10).mean()
df['range_vs_10d'] = df['range'] / (df['range_10d_avg'] + 0.0001)

# Consecutive patterns
df['narrow_range'] = (df['range'] < df['range_10d_avg'] * 0.7).astype(int)
df['narrow_range_streak'] = df['narrow_range'].groupby((df['narrow_range'] != df['narrow_range'].shift()).cumsum()).cumsum()

# Position in range
df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
df['open_to_high'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 0.0001)

# Overnight gap relative to range
df['gap_to_range'] = abs(df['overnight_gap']) / (df['prev_range'] + 0.0001)

# Inside/Outside days
df['inside_day'] = ((df['high'] <= df['high'].shift(1)) & 
                    (df['low'] >= df['low'].shift(1))).astype(int)
df['outside_day'] = ((df['high'] > df['high'].shift(1)) & 
                     (df['low'] < df['low'].shift(1))).astype(int)

# Doji patterns (small body relative to range)
df['body'] = abs(df['close'] - df['open']) / df['close']
df['body_to_range'] = df['body'] / (df['range'] + 0.0001)
df['doji'] = (df['body_to_range'] < 0.2).astype(int)

# Previous extremes
df['near_20d_low'] = (df['low'] - df['low'].rolling(window=20).min()) / df['low'] < 0.01
df['near_20d_high'] = (df['high'].rolling(window=20).max() - df['high']) / df['high'] < 0.01

# Reversal patterns
df['bullish_reversal'] = ((df['low'] < df['low'].shift(1)) & 
                          (df['close'] > df['open']) & 
                          (df['close_to_high'] > 0.7)).astype(int)

# Volatility squeeze
df['atr_10'] = df['true_range'].rolling(window=10).mean()
df['atr_percentile'] = df['atr_10'].rolling(window=50).rank(pct=True)

# More refined overnight gap thresholds based on previous analysis
df['optimal_gap_range'] = ((df['overnight_gap'] > 0.0033) & 
                           (df['overnight_gap'] < 0.0038)).astype(int)

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Range expansion/contraction
for threshold in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2]:
    signal = (df['range_expansion'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Range Expansion > {threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Narrow range conditions
for days in [2, 3, 4]:
    signal = (df['narrow_range_streak'] >= days).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'{days}+ Narrow Range Days',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 3: Gap relative to range
for threshold in [0.3, 0.5, 0.7, 1, 1.5]:
    signal = (df['gap_to_range'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap/Range > {threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Inside days
signal = df['inside_day'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Inside Day',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 5: ATR squeeze conditions
for threshold in [0.2, 0.3, 0.4]:
    signal = (df['atr_percentile'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'ATR in Bottom {int(threshold*100)}%',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: Near 20-day extremes
signal = df['near_20d_low'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Near 20-Day Low',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 7: Doji patterns
signal = df['doji'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Doji Pattern',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 8: Optimal gap range (refined from previous waves)
signal = df['optimal_gap_range'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': '0.33% < Gap < 0.38%',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 9: Close position in range with specific thresholds
for threshold in [0.15, 0.2, 0.25, 0.8, 0.85]:
    if threshold < 0.5:
        signal = (df['close_to_high'] < threshold).astype(int)
        condition = f'Close in Bottom {int(threshold*100)}% of Range'
    else:
        signal = (df['close_to_high'] > threshold).astype(int)
        condition = f'Close in Top {int((1-threshold)*100)}% of Range'
    
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': condition,
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 10: True range conditions
for threshold in [0.01, 0.015, 0.02, 0.025]:
    signal = (df['true_range'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'True Range > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 11: Very specific overnight gap thresholds (super fine-tuning)
for threshold in [0.00355, 0.00360, 0.00365]:
    signal = (df['overnight_gap'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Overnight Gap < {threshold:.4%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 9 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 20 by Sharpe Ratio:")
print(results_df.head(20).to_string(index=False))

# Show best high-trade conditions
high_trades = results_df[results_df['trades'] > 2500].sort_values('sharpe', ascending=False)
print(f"\nBest conditions with >2500 trades:")
print(high_trades.head(15).to_string(index=False))

# Check if we're getting closer
very_close = results_df[(results_df['sharpe'] >= 1.15) & (results_df['trades'] > 2000)]
print(f"\nConditions with Sharpe >= 1.15 and Trades > 2000:")
print(very_close.to_string(index=False))

# Filter for our target criteria
qualifying = results_df[(results_df['sharpe'] >= 1.2) & (results_df['trades'] > 2500)]
print(f"\n*** CONDITIONS MEETING CRITERIA (Sharpe >= 1.2 & Trades > 2500): {len(qualifying)} ***")
if len(qualifying) > 0:
    print("\n🎯 SUCCESS! Found qualifying conditions:")
    print(qualifying.to_string(index=False))
    
    # Save successful conditions
    with open('success.txt', 'w') as f:
        f.write("=== SUCCESSFUL CONDITIONS ===\n\n")
        for idx, row in qualifying.iterrows():
            f.write(f"Condition: {row['condition']}\n")
            f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
            f.write(f"Trades: {row['trades']}\n")
            f.write(f"Total Return: {row['total_return']:.2%}\n")
            f.write("-" * 40 + "\n")

# Append to history
with open('history.log', 'a') as f:
    f.write("\n\n=== WAVE 9 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")