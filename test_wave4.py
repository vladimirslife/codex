#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 4: Fine-tuning overnight gap thresholds and related patterns
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

def calculate_rsi(prices, period):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
df['prev_overnight_gap'] = df['overnight_gap'].shift(1)

# Gap patterns
df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
df['gap_size'] = abs(df['overnight_gap'])

# Previous day patterns
df['prev_daily_return'] = df['daily_return'].shift(1)
df['prev_intraday'] = df['intraday_return'].shift(1)
df['prev_close'] = df['close'].shift(1)
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)

# Multi-day patterns
for i in range(2, 6):
    df[f'return_{i}d'] = df['close'].pct_change(i)

# Moving averages for context
for period in [5, 10, 20, 50]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

# RSI
for period in [5, 7, 10, 14]:
    df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

# Volatility
df['volatility_5d'] = df['daily_return'].rolling(window=5).std()
df['volatility_10d'] = df['daily_return'].rolling(window=10).std()

# Range measures
df['high_low_spread'] = (df['high'] - df['low']) / df['close']
df['prev_range'] = (df['prev_high'] - df['prev_low']) / df['prev_close']

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Very fine-tuned overnight gap thresholds (focusing around 0.5%)
for threshold in [-0.01, -0.005, -0.002, 0, 0.001, 0.002, 0.003, 0.004, 0.005, 
                  0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.015]:
    signal = (df['overnight_gap'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Overnight Gap < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Absolute gap size
for threshold in [0.003, 0.005, 0.007, 0.01, 0.015, 0.02]:
    signal = (df['gap_size'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'|Gap| > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 3: Previous day return conditions
for threshold in [-0.03, -0.02, -0.015, -0.01, -0.005, 0, 0.01, 0.02]:
    signal = (df['prev_daily_return'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Previous Day Return < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Multi-day returns
for days in [2, 3, 4, 5]:
    for threshold in [-0.03, -0.02, -0.01, 0, 0.01, 0.02]:
        signal = (df[f'return_{days}d'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'{days}-Day Return < {threshold:.1%}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 5: Open relative to previous day levels
signal = (df['open'] < df['prev_low']).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Open < Previous Low',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 6: Open relative to previous close with percentage
for pct in [-0.02, -0.015, -0.01, -0.005, 0]:
    signal = ((df['open'] - df['prev_close']) / df['prev_close'] < pct).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Open < Prev Close - {abs(pct):.1%}' if pct < 0 else 'Open < Prev Close',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 7: Volatility conditions with gap
for vol_threshold in [0.01, 0.015, 0.02]:
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

# Test 8: RSI conditions with finer thresholds
for period in [5, 7]:
    for threshold in [24, 25, 26, 27, 28, 29, 30]:
        signal = (df[f'rsi_{period}'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'RSI({period}) < {threshold}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 9: Previous intraday return
for threshold in [-0.02, -0.015, -0.01, -0.005, 0]:
    signal = (df['prev_intraday'] < threshold).astype(int)
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

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 4 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 15 by Sharpe Ratio:")
print(results_df.head(15).to_string(index=False))

# Show best high-trade conditions
high_trades = results_df[results_df['trades'] > 2500].sort_values('sharpe', ascending=False)
print(f"\nBest conditions with >2500 trades:")
print(high_trades.head(10).to_string(index=False))

# Check if we're getting closer
close_to_target = results_df[(results_df['sharpe'] >= 1.1) & (results_df['trades'] > 2000)]
print(f"\nConditions with Sharpe >= 1.1 and Trades > 2000:")
print(close_to_target.to_string(index=False))

# Filter for our target criteria
qualifying = results_df[(results_df['sharpe'] >= 1.2) & (results_df['trades'] > 2500)]
print(f"\nConditions meeting criteria (Sharpe >= 1.2 & Trades > 2500): {len(qualifying)}")
if len(qualifying) > 0:
    print(qualifying.to_string(index=False))

# Append to history
with open('history.log', 'a') as f:
    f.write("\n\n=== WAVE 4 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")