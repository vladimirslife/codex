#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 5: Ultra-fine tuning of overnight gap threshold around 0.3%
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

# Previous patterns
df['prev_overnight_gap'] = df['overnight_gap'].shift(1)
df['prev_daily_return'] = df['daily_return'].shift(1)
df['prev_intraday'] = df['intraday_return'].shift(1)
df['prev_close'] = df['close'].shift(1)
df['prev_open'] = df['open'].shift(1)

# Gap streaks
df['gap_down'] = (df['overnight_gap'] < 0).astype(int)
df['gap_down_streak'] = df['gap_down'].groupby((df['gap_down'] != df['gap_down'].shift()).cumsum()).cumsum()

# Moving averages
for period in [5, 10, 20]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

# High/Low patterns
df['high'] = df['high']
df['low'] = df['low']
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)

# Volatility
df['volatility_5d'] = df['daily_return'].rolling(window=5).std()

# Range
df['range_pct'] = (df['high'] - df['low']) / df['close']

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Ultra-fine tuning around 0.3% (the sweet spot from Wave 4)
for threshold in [0.0020, 0.0022, 0.0024, 0.0026, 0.0028, 0.0030, 0.0032, 
                  0.0034, 0.0036, 0.0038, 0.0040]:
    signal = (df['overnight_gap'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Overnight Gap < {threshold:.2%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Even finer granularity around best performers
for threshold in [0.0025, 0.0027, 0.0029, 0.0031, 0.0033, 0.0035]:
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

# Test 3: Previous overnight gap conditions
for threshold in [-0.005, -0.003, -0.001, 0, 0.001, 0.003, 0.005]:
    signal = (df['prev_overnight_gap'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Prev Overnight Gap < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Consecutive gap down days
for streak in [2, 3, 4]:
    signal = (df['gap_down_streak'] >= streak).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'{streak}+ Consecutive Gap Downs',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 5: Open below previous open
for pct in [-0.01, -0.005, 0, 0.005, 0.01]:
    signal = ((df['open'] - df['prev_open']) / df['prev_open'] < pct).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Open < Prev Open + {pct:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: Small positive gap (mean reversion after gap up)
for lower in [0.001, 0.002, 0.003]:
    for upper in [0.005, 0.007, 0.01]:
        if lower < upper:
            signal = ((df['overnight_gap'] > lower) & (df['overnight_gap'] < upper)).astype(int)
            strategy_return = signal * df['next_overnight_return']
            strategy_return.fillna(0, inplace=True)
            
            sharpe = calculate_sharpe(strategy_return)
            num_trades = signal.sum()
            
            results.append({
                'condition': f'{lower:.1%} < Gap < {upper:.1%}',
                'sharpe': sharpe,
                'trades': num_trades,
                'total_return': (1 + strategy_return).prod() - 1
            })

# Test 7: Volatility with gap conditions
for vol_threshold in [0.012, 0.015, 0.018]:
    signal = (df['volatility_5d'] > vol_threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'5d Volatility > {vol_threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 8: Range conditions
for threshold in [0.015, 0.02, 0.025, 0.03]:
    signal = (df['range_pct'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Daily Range > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 9: Open at specific position relative to previous day range
signal = (df['open'] < df['prev_close']).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Open < Previous Close',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 10: Previous day was down
for threshold in [-0.02, -0.015, -0.01, -0.005, 0]:
    signal = (df['prev_daily_return'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Prev Day < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 5 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 20 by Sharpe Ratio:")
print(results_df.head(20).to_string(index=False))

# Show best high-trade conditions
high_trades = results_df[results_df['trades'] > 2500].sort_values('sharpe', ascending=False)
print(f"\nBest conditions with >2500 trades:")
print(high_trades.head(15).to_string(index=False))

# Check conditions close to target
close_to_target = results_df[(results_df['sharpe'] >= 1.15) & (results_df['trades'] > 2500)]
print(f"\nConditions with Sharpe >= 1.15 and Trades > 2500:")
print(close_to_target.to_string(index=False))

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
    f.write("\n\n=== WAVE 5 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")