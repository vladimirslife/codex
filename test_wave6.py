#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 6: Exploring alternative patterns - candle size, extremes, weekly patterns
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
# Basic returns and gaps
df['daily_return'] = df['close'].pct_change()
df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
df['intraday_return'] = (df['close'] - df['open']) / df['open']

# Day of week
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 4=Friday
df['is_monday'] = (df['day_of_week'] == 0).astype(int)
df['is_friday'] = (df['day_of_week'] == 4).astype(int)
df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)

# Candle patterns
df['body_size'] = abs(df['close'] - df['open']) / df['open']
df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
df['full_range'] = (df['high'] - df['low']) / df['close']

# Previous candle
df['prev_body_size'] = df['body_size'].shift(1)
df['prev_full_range'] = df['full_range'].shift(1)

# Position within range
df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'])

# Multi-period highs/lows
for period in [5, 10, 20, 50]:
    df[f'high_{period}d'] = df['high'].rolling(window=period).max()
    df[f'low_{period}d'] = df['low'].rolling(window=period).min()
    df[f'is_at_{period}d_low'] = (df['low'] == df[f'low_{period}d']).astype(int)
    df[f'pct_from_{period}d_high'] = (df['close'] - df[f'high_{period}d']) / df[f'high_{period}d']
    df[f'pct_from_{period}d_low'] = (df['close'] - df[f'low_{period}d']) / df[f'low_{period}d']

# RSI
for period in [5, 7, 10, 14]:
    df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

# Moving averages
for period in [5, 10, 20, 50]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    df[f'distance_from_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']

# Volatility
df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
df['volatility_20d'] = df['daily_return'].rolling(window=20).std()

# Momentum
df['momentum_5d'] = df['close'].pct_change(5)
df['momentum_10d'] = df['close'].pct_change(10)

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Candle body size conditions
for threshold in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
    signal = (df['prev_body_size'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Prev Candle Body > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Full range conditions
for threshold in [0.015, 0.02, 0.025, 0.03, 0.035]:
    signal = (df['prev_full_range'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Prev Full Range > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 3: Close position within daily range
for threshold in [0.2, 0.25, 0.3, 0.35, 0.4]:
    signal = (df['close_position'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Close in Bottom {int(threshold*100)}% of Daily Range',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: At N-day lows
for period in [5, 10, 20]:
    signal = df[f'is_at_{period}d_low'].astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'At {period}-Day Low',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 5: Distance from N-day high
for period in [10, 20, 50]:
    for threshold in [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03]:
        signal = (df[f'pct_from_{period}d_high'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'Close < {period}d High - {abs(threshold):.0%}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 6: Day of week effects
for day_name, day_col in [('Monday', 'is_monday'), ('Friday', 'is_friday'), ('Tuesday', 'is_tuesday')]:
    signal = df[day_col].astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'{day_name} Entry',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 7: Lower shadow patterns (potential reversal)
for threshold in [0.008, 0.01, 0.012, 0.015]:
    signal = (df['lower_shadow'] > threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Lower Shadow > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 8: Distance from moving averages
for period in [10, 20, 50]:
    for threshold in [-0.03, -0.02, -0.015, -0.01]:
        signal = (df[f'distance_from_sma_{period}'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'Close < SMA({period}) - {abs(threshold):.1%}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 9: Momentum conditions
for period in [5, 10]:
    for threshold in [-0.05, -0.04, -0.03, -0.02, -0.01]:
        signal = (df[f'momentum_{period}d'] < threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'{period}d Momentum < {threshold:.1%}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 10: Enhanced volatility conditions
for period in [10, 20]:
    for threshold in [0.013, 0.015, 0.017, 0.02]:
        signal = (df[f'volatility_{period}d'] > threshold).astype(int)
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

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 6 RESULTS ===")
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
    f.write("\n\n=== WAVE 6 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")