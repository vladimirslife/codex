#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 1: Testing various single entry conditions
Goal: Sharpe Ratio >= 1.3 and Trades > 3000
"""

import pandas as pd
import numpy as np
import sys
import os
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ------------------------- HELPERS -----------------------------------
def load_ticker(path: str) -> pd.DataFrame:
    """Load CSV, standardize columns, compute overnight returns."""
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Next_Open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    return df

# Calculate RSI
def calculate_rsi(data, period):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Bollinger Bands
def calculate_bollinger(data, period, std_dev):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return sma, upper, lower

# Load data
data = load_ticker("4 - QQQ.csv")

# Pre-calculate various indicators
# RSI
for period in [5, 10, 14, 20, 30]:
    data[f'rsi_{period}'] = calculate_rsi(data['close'], period)

# Moving averages
for period in [5, 10, 20, 50, 100, 200]:
    data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
    data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()

# Bollinger Bands
for period in [10, 20, 30]:
    for std in [1, 2, 2.5]:
        sma, upper, lower = calculate_bollinger(data['close'], period, std)
        data[f'bb_upper_{period}_{std}'] = upper
        data[f'bb_lower_{period}_{std}'] = lower
        data[f'bb_sma_{period}'] = sma

# Momentum/ROC
for period in [1, 5, 10, 20]:
    data[f'roc_{period}'] = (data['close'] / data['close'].shift(period) - 1) * 100

# ATR
for period in [10, 14, 20]:
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data[f'atr_{period}'] = true_range.rolling(window=period).mean()

# Volume indicators
data['volume'] = data.get('volume', 1)  # If no volume, use 1
for period in [10, 20, 50]:
    data[f'volume_ma_{period}'] = data['volume'].rolling(window=period).mean()

# Simple patterns
data['green_candle'] = (data['close'] > data['open']).astype(int)
data['red_candle'] = (data['close'] < data['open']).astype(int)
data['daily_range'] = (data['high'] - data['low']) / data['close']
data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']

# Function to test a condition
def test_condition(data, condition_func, params):
    df = data.copy()
    
    # Generate signal based on condition
    df['signal'] = condition_func(df, **params).astype(int)
    
    # Shift signal to avoid look-ahead bias
    df['signal'] = df['signal'].shift(1)
    
    # Calculate strategy returns
    df['strategy_daily_return'] = df['signal'] * df['next_overnight_return']
    df['strategy_daily_return'].fillna(0, inplace=True)
    
    # Performance metrics
    annual_rf = 0.02
    daily_rf = annual_rf / 252
    
    excess_returns = df['strategy_daily_return'] - daily_rf
    mean_excess_annual = excess_returns.mean() * 252
    std_excess_annual = excess_returns.std() * np.sqrt(252)
    sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0
    
    num_trades = int(df['signal'].sum())
    
    return sharpe_ratio, num_trades

# Define condition functions
def rsi_below(df, period, threshold):
    return df[f'rsi_{period}'] < threshold

def rsi_above(df, period, threshold):
    return df[f'rsi_{period}'] > threshold

def price_above_sma(df, period):
    return df['close'] > df[f'sma_{period}']

def price_below_sma(df, period):
    return df['close'] < df[f'sma_{period}']

def price_above_ema(df, period):
    return df['close'] > df[f'ema_{period}']

def price_below_ema(df, period):
    return df['close'] < df[f'ema_{period}']

def price_below_bb_lower(df, period, std):
    return df['close'] < df[f'bb_lower_{period}_{std}']

def price_above_bb_upper(df, period, std):
    return df['close'] > df[f'bb_upper_{period}_{std}']

def roc_positive(df, period):
    return df[f'roc_{period}'] > 0

def roc_negative(df, period):
    return df[f'roc_{period}'] < 0

def roc_above_threshold(df, period, threshold):
    return df[f'roc_{period}'] > threshold

def roc_below_threshold(df, period, threshold):
    return df[f'roc_{period}'] < threshold

def green_candle_condition(df):
    return df['green_candle'] == 1

def red_candle_condition(df):
    return df['red_candle'] == 1

def high_daily_range(df, threshold):
    return df['daily_range'] > threshold

def high_atr(df, period, threshold):
    return df[f'atr_{period}'] > df[f'atr_{period}'].rolling(50).mean() * threshold

# Test conditions
results = []
print("Testing Wave 1 conditions...")

# Test RSI conditions
print("Testing RSI conditions...")
for period in [5, 10, 14, 20, 30]:
    for threshold in [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]:
        # RSI below (oversold)
        sharpe, trades = test_condition(data, rsi_below, {'period': period, 'threshold': threshold})
        results.append({
            'condition': f'RSI({period}) < {threshold}',
            'sharpe': sharpe,
            'trades': trades
        })
        
        # RSI above (overbought)
        sharpe, trades = test_condition(data, rsi_above, {'period': period, 'threshold': threshold})
        results.append({
            'condition': f'RSI({period}) > {threshold}',
            'sharpe': sharpe,
            'trades': trades
        })

# Test Moving Average conditions
print("Testing Moving Average conditions...")
for period in [5, 10, 20, 50, 100, 200]:
    # Price above SMA
    sharpe, trades = test_condition(data, price_above_sma, {'period': period})
    results.append({
        'condition': f'Close > SMA({period})',
        'sharpe': sharpe,
        'trades': trades
    })
    
    # Price below SMA
    sharpe, trades = test_condition(data, price_below_sma, {'period': period})
    results.append({
        'condition': f'Close < SMA({period})',
        'sharpe': sharpe,
        'trades': trades
    })
    
    # Price above EMA
    sharpe, trades = test_condition(data, price_above_ema, {'period': period})
    results.append({
        'condition': f'Close > EMA({period})',
        'sharpe': sharpe,
        'trades': trades
    })
    
    # Price below EMA
    sharpe, trades = test_condition(data, price_below_ema, {'period': period})
    results.append({
        'condition': f'Close < EMA({period})',
        'sharpe': sharpe,
        'trades': trades
    })

# Test Bollinger Bands conditions
print("Testing Bollinger Bands conditions...")
for period in [10, 20, 30]:
    for std in [1, 2, 2.5]:
        # Price below lower band
        sharpe, trades = test_condition(data, price_below_bb_lower, {'period': period, 'std': std})
        results.append({
            'condition': f'Close < BB_Lower({period},{std})',
            'sharpe': sharpe,
            'trades': trades
        })
        
        # Price above upper band
        sharpe, trades = test_condition(data, price_above_bb_upper, {'period': period, 'std': std})
        results.append({
            'condition': f'Close > BB_Upper({period},{std})',
            'sharpe': sharpe,
            'trades': trades
        })

# Test ROC conditions
print("Testing ROC/Momentum conditions...")
for period in [1, 5, 10, 20]:
    # ROC positive
    sharpe, trades = test_condition(data, roc_positive, {'period': period})
    results.append({
        'condition': f'ROC({period}) > 0',
        'sharpe': sharpe,
        'trades': trades
    })
    
    # ROC negative
    sharpe, trades = test_condition(data, roc_negative, {'period': period})
    results.append({
        'condition': f'ROC({period}) < 0',
        'sharpe': sharpe,
        'trades': trades
    })
    
    # ROC above/below thresholds
    for threshold in [-2, -1, 1, 2]:
        sharpe, trades = test_condition(data, roc_above_threshold, {'period': period, 'threshold': threshold})
        results.append({
            'condition': f'ROC({period}) > {threshold}%',
            'sharpe': sharpe,
            'trades': trades
        })
        
        sharpe, trades = test_condition(data, roc_below_threshold, {'period': period, 'threshold': threshold})
        results.append({
            'condition': f'ROC({period}) < {threshold}%',
            'sharpe': sharpe,
            'trades': trades
        })

# Test simple patterns
print("Testing simple pattern conditions...")
# Green candle
sharpe, trades = test_condition(data, green_candle_condition, {})
results.append({
    'condition': 'Green Candle (Close > Open)',
    'sharpe': sharpe,
    'trades': trades
})

# Red candle
sharpe, trades = test_condition(data, red_candle_condition, {})
results.append({
    'condition': 'Red Candle (Close < Open)',
    'sharpe': sharpe,
    'trades': trades
})

# High daily range
for threshold in [0.01, 0.015, 0.02, 0.025, 0.03]:
    sharpe, trades = test_condition(data, high_daily_range, {'threshold': threshold})
    results.append({
        'condition': f'Daily Range > {threshold*100:.1f}%',
        'sharpe': sharpe,
        'trades': trades
    })

# High ATR
print("Testing ATR conditions...")
for period in [10, 14, 20]:
    for threshold in [1.1, 1.2, 1.3, 1.5]:
        sharpe, trades = test_condition(data, high_atr, {'period': period, 'threshold': threshold})
        results.append({
            'condition': f'ATR({period}) > {threshold}x MA(ATR)',
            'sharpe': sharpe,
            'trades': trades
        })

# Convert results to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

# Print summary
print("\n" + "="*80)
print("WAVE 1 RESULTS SUMMARY")
print("="*80)
print(f"Total conditions tested: {len(results)}")
print(f"\nTop 10 conditions by Sharpe Ratio:")
print("-"*80)
for i, row in results_df.head(10).iterrows():
    print(f"{row['condition']:<50} | Sharpe: {row['sharpe']:>7.4f} | Trades: {row['trades']:>5}")

print(f"\nConditions meeting criteria (Sharpe >= 1.3 and Trades > 3000):")
print("-"*80)
qualifying = results_df[(results_df['sharpe'] >= 1.3) & (results_df['trades'] > 3000)]
if len(qualifying) > 0:
    for i, row in qualifying.iterrows():
        print(f"{row['condition']:<50} | Sharpe: {row['sharpe']:>7.4f} | Trades: {row['trades']:>5}")
else:
    print("No conditions meet the criteria yet.")

# Save top results to history
with open('history.log', 'w') as f:
    f.write("WAVE 1 TOP RESULTS\n")
    f.write("==================\n\n")
    for i, row in results_df.head(3).iterrows():
        f.write(f"Condition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Number of Trades: {row['trades']}\n")
        f.write("-"*50 + "\n\n")

print(f"\nResults saved to history.log")