#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 2: Extended testing with percentage deviations and new patterns
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

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

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
# Basic returns and price action
df['daily_return'] = df['close'].pct_change()
df['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
df['intraday_return'] = (df['close'] - df['open']) / df['open']
df['high_low_pct'] = (df['high'] - df['low']) / df['close']
df['body_size'] = abs(df['close'] - df['open']) / df['open']

# Moving averages
for period in [10, 20, 30, 50, 100, 150, 200]:
    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    # Percentage distance from MA
    df[f'pct_from_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
    df[f'pct_from_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100

# RSI
for period in [5, 7, 10, 14, 21, 30]:
    df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

# ATR
for period in [10, 14, 20]:
    df[f'atr_{period}'] = calculate_atr(df['high'], df['low'], df['close'], period)
    df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close'] * 100

# Consecutive patterns
df['red_candle'] = (df['close'] < df['open']).astype(int)
df['green_candle'] = (df['close'] > df['open']).astype(int)
df['down_day'] = (df['daily_return'] < 0).astype(int)
df['up_day'] = (df['daily_return'] > 0).astype(int)

# Count consecutive occurrences
for col in ['red_candle', 'down_day']:
    df[f'{col}_streak'] = df[col].groupby((df[col] != df[col].shift()).cumsum()).cumsum()

# Rolling statistics
for period in [5, 10, 20]:
    df[f'return_std_{period}'] = df['daily_return'].rolling(window=period).std()
    df[f'close_rank_{period}'] = df['close'].rolling(window=period).rank(pct=True)

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Refined daily return thresholds
for threshold in [-0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.03]:
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

# Test 2: Percentage below moving averages
for period in [20, 50, 100, 150, 200]:
    for pct_threshold in [-5, -3, -2, -1, 0, 1, 2]:
        signal = (df[f'pct_from_sma_{period}'] < pct_threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'Price < SMA({period}) - {abs(pct_threshold)}%' if pct_threshold < 0 else f'Price < SMA({period}) + {pct_threshold}%',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 3: RSI with finer thresholds
for period in [5, 7, 10, 14]:
    for threshold in [25, 28, 30, 32, 35, 38, 40, 45]:
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

# Test 4: Consecutive down days
for streak in [2, 3, 4, 5]:
    signal = (df['down_day_streak'] >= streak).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'{streak}+ Consecutive Down Days',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 5: Consecutive red candles
for streak in [2, 3, 4]:
    signal = (df['red_candle_streak'] >= streak).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'{streak}+ Consecutive Red Candles',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: ATR-based volatility conditions
for period in [10, 14, 20]:
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        signal = (df[f'atr_pct_{period}'] > threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'ATR({period})% > {threshold}',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 7: Intraday reversal patterns
for threshold in [-0.02, -0.015, -0.01, -0.005]:
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

# Test 8: Price rank within rolling window
for period in [10, 20]:
    for rank_threshold in [0.2, 0.3, 0.4]:
        signal = (df[f'close_rank_{period}'] < rank_threshold).astype(int)
        strategy_return = signal * df['next_overnight_return']
        strategy_return.fillna(0, inplace=True)
        
        sharpe = calculate_sharpe(strategy_return)
        num_trades = signal.sum()
        
        results.append({
            'condition': f'Price in Bottom {int(rank_threshold*100)}% of {period}d',
            'sharpe': sharpe,
            'trades': num_trades,
            'total_return': (1 + strategy_return).prod() - 1
        })

# Test 9: Large body candles
for threshold in [0.015, 0.02, 0.025, 0.03]:
    signal = ((df['body_size'] > threshold) & (df['red_candle'] == 1)).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Red Candle Body > {threshold:.1%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 2 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 10 by Sharpe Ratio:")
print(results_df.head(10).to_string(index=False))

# Filter for our target criteria
qualifying = results_df[(results_df['sharpe'] >= 1.2) & (results_df['trades'] > 2500)]
print(f"\nConditions meeting criteria (Sharpe >= 1.2 & Trades > 2500): {len(qualifying)}")
if len(qualifying) > 0:
    print(qualifying.to_string(index=False))

# Append to history
with open('history.log', 'a') as f:
    f.write("\n\n=== WAVE 2 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")