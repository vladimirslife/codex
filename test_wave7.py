#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 7: Advanced gap-based conditions and ratios
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
df['overnight_return'] = df['overnight_gap']  # for clarity

# Previous values
df['prev_daily_return'] = df['daily_return'].shift(1)
df['prev_overnight_return'] = df['overnight_return'].shift(1)
df['prev_intraday_return'] = df['intraday_return'].shift(1)
df['prev_close'] = df['close'].shift(1)
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)

# Gap characteristics
df['gap_size'] = abs(df['overnight_gap'])
df['gap_direction'] = np.sign(df['overnight_gap'])
df['prev_gap_size'] = df['gap_size'].shift(1)
df['prev_gap_direction'] = df['gap_direction'].shift(1)

# Volatility measures
for period in [5, 10, 20]:
    df[f'volatility_{period}d'] = df['daily_return'].rolling(window=period).std()
    df[f'gap_volatility_{period}d'] = df['overnight_gap'].rolling(window=period).std()

# Gap relative to volatility
df['gap_to_vol_5d'] = df['overnight_gap'] / (df['volatility_5d'] + 0.0001)
df['gap_to_vol_10d'] = df['overnight_gap'] / (df['volatility_10d'] + 0.0001)

# Cumulative returns
df['cum_return_5d'] = df['daily_return'].rolling(window=5).sum()
df['cum_return_10d'] = df['daily_return'].rolling(window=10).sum()

# Gap after moves
df['gap_after_down'] = ((df['prev_daily_return'] < 0) & (df['overnight_gap'] < 0)).astype(int)
df['gap_after_up'] = ((df['prev_daily_return'] > 0) & (df['overnight_gap'] > 0)).astype(int)

# Open relative to previous levels
df['open_to_prev_high'] = (df['open'] - df['prev_high']) / df['prev_high']
df['open_to_prev_low'] = (df['open'] - df['prev_low']) / df['prev_low']
df['open_to_prev_close'] = (df['open'] - df['prev_close']) / df['prev_close']

# Moving averages for reference
for period in [20, 50]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

# Range
df['prev_range'] = (df['prev_high'] - df['prev_low']) / df['prev_close']
df['current_range'] = (df['high'] - df['low']) / df['close']

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Previous overnight return conditions
for threshold in [-0.01, -0.005, -0.003, -0.001, 0, 0.001, 0.003, 0.005]:
    signal = (df['prev_overnight_return'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Prev Overnight Return < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Gap relative to volatility
for period in [5, 10]:
    for threshold in [-2, -1.5, -1, -0.5, 0, 0.5, 1]:
        signal = (df[f'gap_to_vol_{period}d'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'Gap/{period}d Vol < {threshold}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 3: Open relative to previous high with fine thresholds
for threshold in [-0.02, -0.015, -0.01, -0.008, -0.006, -0.004, -0.002, 0]:
    signal = (df['open_to_prev_high'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Open < Prev High + {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Cumulative return conditions
for period in [5, 10]:
    for threshold in [-0.05, -0.03, -0.02, -0.01, 0, 0.01]:
        signal = (df[f'cum_return_{period}d'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'{period}d Cumulative Return < {threshold:.1%}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 5: Gap after down day
signal = df['gap_after_down'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Gap Down After Down Day',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 6: Open below previous low percentage
for threshold in [0, 0.002, 0.005, 0.01]:
    signal = ((df['prev_low'] - df['open']) / df['prev_low'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Open < Prev Low - {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 7: Gap volatility conditions
for period in [5, 10]:
    for threshold in [0.003, 0.005, 0.007, 0.01]:
        signal = (df[f'gap_volatility_{period}d'] > threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'{period}d Gap Volatility > {threshold:.1%}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 8: Overnight to intraday ratio
df['overnight_to_daily'] = df['overnight_gap'] / (df['prev_daily_return'] + 0.0001)
for threshold in [-2, -1, 0, 0.5, 1]:
    signal = (df['overnight_to_daily'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Overnight/Daily Ratio < {threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 9: Very specific overnight gap thresholds (refinement of best range)
for threshold in [0.0034, 0.0035, 0.0036, 0.0037, 0.0038]:
    signal = (df['overnight_gap'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Overnight Gap < {threshold:.3%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 10: Gap size (absolute) conditions
for threshold in [0.002, 0.003, 0.004, 0.005, 0.007]:
    signal = (df['gap_size'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'|Gap Size| > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 7 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 20 by Sharpe Ratio:")
print(results_df.head(20).to_string(index=False))

# Show best high-trade conditions
high_trades = results_df[results_df['trades'] > 2500].sort_values('sharpe', ascending=False)
print(f"\nBest conditions with >2500 trades:")
print(high_trades.head(15).to_string(index=False))

# Check if we're getting closer
very_close = results_df[(results_df['sharpe'] >= 1.18) & (results_df['trades'] > 2000)]
print(f"\nConditions with Sharpe >= 1.18 and Trades > 2000:")
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
    f.write("\n\n=== WAVE 7 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")