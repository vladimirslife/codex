#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 1: Testing various simple entry conditions
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

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

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

# ------------------------- CALCULATE ALL INDICATORS -------------------
# RSI for different periods
for period in [5, 10, 14, 20, 30, 40, 50]:
    df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

# SMA for different periods
for period in [5, 10, 20, 30, 50, 100, 200]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

# EMA for different periods
for period in [5, 10, 20, 30, 50, 100, 200]:
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# MACD
df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

# Bollinger Bands
df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)

# Price action
df['daily_return'] = df['close'].pct_change()
df['high_low_range'] = (df['high'] - df['low']) / df['close']
df['close_open_pct'] = (df['close'] - df['open']) / df['open']

# Volume-based (if available)
if 'volume' in df.columns:
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: RSI conditions (oversold)
for period in [5, 10, 14, 20, 30]:
    for threshold in [20, 25, 30, 35, 40]:
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

# Test 2: RSI conditions (overbought - for mean reversion)
for period in [5, 10, 14, 20, 30]:
    for threshold in [60, 65, 70, 75, 80]:
        signal = (df[f'rsi_{period}'] > threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'RSI({period}) > {threshold}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 3: Price above SMA (trend following)
for period in [10, 20, 50, 100, 200]:
    signal = (df['close'] > df[f'sma_{period}']).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Close > SMA({period})',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Price below SMA (mean reversion)
for period in [10, 20, 50, 100]:
    signal = (df['close'] < df[f'sma_{period}']).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Close < SMA({period})',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 5: Price above EMA
for period in [10, 20, 50, 100]:
    signal = (df['close'] > df[f'ema_{period}']).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Close > EMA({period})',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: MACD conditions
signal = (df['macd'] > df['macd_signal']).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'MACD > Signal',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 7: MACD histogram positive
signal = (df['macd_hist'] > 0).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'MACD Histogram > 0',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 8: Bollinger Bands
signal = (df['close'] < df['bb_lower']).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Close < BB Lower',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 9: Daily return conditions
for threshold in [-0.03, -0.02, -0.01, 0.01, 0.02]:
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

# Test 10: Green/Red candles
signal = (df['close'] < df['open']).astype(int)  # Red candle
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Red Candle (Close < Open)',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 1 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 10 by Sharpe Ratio:")
print(results_df.head(10).to_string(index=False))

# Filter for our target criteria
qualifying = results_df[(results_df['sharpe'] >= 1.2) & (results_df['trades'] > 2500)]
print(f"\nConditions meeting criteria (Sharpe >= 1.2 & Trades > 2500): {len(qualifying)}")
if len(qualifying) > 0:
    print(qualifying.to_string(index=False))

# Save top results to history
with open('history.log', 'w') as f:
    f.write("=== WAVE 1 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")