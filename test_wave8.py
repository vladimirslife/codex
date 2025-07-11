#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 8: Time patterns, extreme gaps, and intraday conditions
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

# Day of week and month
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 4=Friday
df['month'] = df['Date'].dt.month
df['is_monday'] = (df['day_of_week'] == 0).astype(int)
df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
df['is_thursday'] = (df['day_of_week'] == 3).astype(int)
df['is_friday'] = (df['day_of_week'] == 4).astype(int)

# Previous values
df['prev_intraday_return'] = df['intraday_return'].shift(1)
df['prev_overnight_gap'] = df['overnight_gap'].shift(1)
df['prev_daily_return'] = df['daily_return'].shift(1)

# Gap characteristics
df['gap_size'] = abs(df['overnight_gap'])
df['prev_gap_size'] = df['gap_size'].shift(1)
df['gap_ratio'] = df['gap_size'] / (df['prev_gap_size'] + 0.0001)

# Rolling gap statistics
df['gap_mean_5d'] = df['overnight_gap'].rolling(window=5).mean()
df['gap_std_5d'] = df['overnight_gap'].rolling(window=5).std()
df['gap_zscore'] = (df['overnight_gap'] - df['gap_mean_5d']) / (df['gap_std_5d'] + 0.0001)

# Extreme gaps
df['gap_percentile'] = df['overnight_gap'].rolling(window=20).rank(pct=True)

# Previous close to open vs close to close
df['prev_co_return'] = df['prev_overnight_gap']
df['prev_cc_return'] = df['prev_daily_return']

# Multi-day patterns
df['sum_gap_3d'] = df['overnight_gap'].rolling(window=3).sum()
df['sum_intraday_3d'] = df['intraday_return'].rolling(window=3).sum()

# Volatility
df['volatility_10d'] = df['daily_return'].rolling(window=10).std()

# Moving averages
df['sma_20'] = df['close'].rolling(window=20).mean()
df['distance_from_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Previous intraday return conditions
for threshold in [-0.02, -0.015, -0.01, -0.008, -0.005, -0.003, 0, 0.003, 0.005]:
    signal = (df['prev_intraday_return'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Prev Intraday Return < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Day of week with gap conditions
for day, col in [('Monday', 'is_monday'), ('Tuesday', 'is_tuesday'), 
                 ('Wednesday', 'is_wednesday'), ('Thursday', 'is_thursday'), 
                 ('Friday', 'is_friday')]:
    # Day only
    signal = df[col].astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'{day} Entry',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 3: Gap Z-score (standardized gap)
for threshold in [-2, -1.5, -1, -0.5, 0, 0.5]:
    signal = (df['gap_zscore'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap Z-Score < {threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Gap percentile (extreme gaps)
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
    signal = (df['gap_percentile'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap in Bottom {int(threshold*100)}% of 20d',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 5: Gap ratio (current vs previous)
for threshold in [0.5, 1, 1.5, 2, 3]:
    signal = (df['gap_ratio'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap Ratio > {threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: Sum of gaps over 3 days
for threshold in [-0.02, -0.01, -0.005, 0, 0.005, 0.01]:
    signal = (df['sum_gap_3d'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'3d Gap Sum < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 7: Sum of intraday returns over 3 days
for threshold in [-0.03, -0.02, -0.01, 0, 0.01]:
    signal = (df['sum_intraday_3d'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'3d Intraday Sum < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 8: Large previous intraday move
for threshold in [0.01, 0.015, 0.02, 0.025]:
    signal = (abs(df['prev_intraday_return']) > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'|Prev Intraday| > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 9: Very small gaps (consolidation)
for threshold in [0.001, 0.002, 0.003, 0.004]:
    signal = (df['gap_size'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'|Gap| < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 10: Combined - Monday with negative gap
signal = ((df['is_monday'] == 1) & (df['overnight_gap'] < 0)).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Monday with Gap Down',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 8 RESULTS ===")
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
    f.write("\n\n=== WAVE 8 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")