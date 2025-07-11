#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 3: Fine-tuning thresholds and exploring gap/momentum patterns
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

def calculate_rsi(prices, period):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_roc(prices, period):
    """Calculate Rate of Change"""
    return (prices - prices.shift(period)) / prices.shift(period) * 100

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
# Basic returns
df['daily_return'] = df['close'].pct_change()
df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
df['intraday_return'] = (df['close'] - df['open']) / df['open']

# High-Low spread
df['high_low_spread'] = (df['high'] - df['low']) / df['close']
df['high_close_dist'] = (df['high'] - df['close']) / df['close']
df['close_low_dist'] = (df['close'] - df['low']) / df['low']

# Momentum indicators
for period in [3, 5, 10, 20]:
    df[f'roc_{period}'] = calculate_roc(df['close'], period)
    df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

# RSI variants
for period in [5, 6, 7, 8, 9, 10, 12, 14]:
    df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

# Moving averages for reference
for period in [5, 10, 20, 50, 100, 200]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# Volatility measures
for period in [5, 10, 20]:
    df[f'volatility_{period}'] = df['daily_return'].rolling(window=period).std()
    df[f'range_avg_{period}'] = df['high_low_spread'].rolling(window=period).mean()

# Pattern indicators
df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
df['lower_close'] = (df['close'] < df['close'].shift(1)).astype(int)
df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)

# Shadow ratios
df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Fine-tuned daily return thresholds (focusing on 2-3% range)
for threshold in [0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.035]:
    signal = (df['daily_return'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Daily Return < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Overnight gap conditions
for threshold in [-0.015, -0.01, -0.005, 0, 0.005, 0.01]:
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

# Test 3: High-Low spread conditions
for threshold in [0.015, 0.02, 0.025, 0.03, 0.035]:
    signal = (df['high_low_spread'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'High-Low Spread > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Momentum conditions (ROC)
for period in [3, 5, 10]:
    for threshold in [-5, -3, -2, -1, 0, 1]:
        signal = (df[f'roc_{period}'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'ROC({period}) < {threshold}%',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 5: Close position within daily range
for threshold in [0.2, 0.3, 0.4, 0.5]:
    signal = (df['close_low_dist'] / df['high_low_spread'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Close in Bottom {int(threshold*100)}% of Range',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: RSI with very specific thresholds
for period in [6, 8, 10]:
    for threshold in [26, 27, 28, 29, 30, 31, 32, 33]:
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

# Test 7: Volatility-based conditions
for period in [5, 10]:
    for threshold in [0.015, 0.02, 0.025]:
        signal = (df[f'volatility_{period}'] > threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'{period}d Volatility > {threshold:.1%}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 8: Lower shadow conditions (hammer patterns)
for threshold in [0.01, 0.015, 0.02]:
    signal = ((df['lower_shadow'] > threshold) & (df['daily_return'] < 0)).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Lower Shadow > {threshold:.1%} on Down Day',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 9: Intraday reversal with specific thresholds
for threshold in [-0.018, -0.016, -0.014, -0.012, -0.008]:
    signal = (df['intraday_return'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Intraday Return < {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 10: Gap down conditions
signal = (df['gap_down'] == 1).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Gap Down Opening',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 3 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 10 by Sharpe Ratio:")
print(results_df.head(10).to_string(index=False))

# Show best high-trade conditions
high_trades = results_df[results_df['trades'] > 2500].sort_values('sharpe', ascending=False)
print(f"\nBest conditions with >2500 trades:")
print(high_trades.head(10).to_string(index=False))

# Filter for our target criteria
qualifying = results_df[(results_df['sharpe'] >= 1.2) & (results_df['trades'] > 2500)]
print(f"\nConditions meeting criteria (Sharpe >= 1.2 & Trades > 2500): {len(qualifying)}")
if len(qualifying) > 0:
    print(qualifying.to_string(index=False))

# Append to history
with open('history.log', 'a') as f:
    f.write("\n\n=== WAVE 3 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")