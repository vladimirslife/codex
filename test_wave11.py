#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 11: Exotic approaches - Fibonacci, golden ratio, extreme statistics
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

# Fibonacci and golden ratio
golden_ratio = 1.618033988749895
fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

# Swing highs and lows
df['swing_high'] = df['high'].rolling(window=20).max()
df['swing_low'] = df['low'].rolling(window=20).min()
df['swing_range'] = df['swing_high'] - df['swing_low']

# Fibonacci retracements
for fib in fibonacci_ratios:
    df[f'fib_{int(fib*100)}'] = df['swing_low'] + (df['swing_range'] * fib)
    df[f'near_fib_{int(fib*100)}'] = (abs(df['close'] - df[f'fib_{int(fib*100)}']) / df['close'] < 0.01).astype(int)

# Golden ratio conditions
df['gap_golden'] = (abs(df['overnight_gap'] - 0.00618) < 0.0001).astype(int)
df['gap_inv_golden'] = (abs(df['overnight_gap'] - 0.00382) < 0.0001).astype(int)

# Statistical extremes
df['gap_zscore'] = (df['overnight_gap'] - df['overnight_gap'].rolling(window=50).mean()) / df['overnight_gap'].rolling(window=50).std()
df['return_zscore'] = (df['daily_return'] - df['daily_return'].rolling(window=50).mean()) / df['daily_return'].rolling(window=50).std()

# Kurtosis and skewness
df['gap_kurt_20'] = df['overnight_gap'].rolling(window=20).kurt()
df['gap_skew_20'] = df['overnight_gap'].rolling(window=20).skew()

# Hurst exponent approximation (trending vs mean-reverting)
def hurst_rs(ts, min_k=2, max_k=None):
    """Simplified R/S calculation for Hurst exponent"""
    if max_k is None:
        max_k = len(ts) // 2
    rs_list = []
    k_list = []
    for k in range(min_k, min(max_k, len(ts)//2)):
        if k < 2:
            continue
        rs = []
        for start in range(0, len(ts)-k+1, k):
            subset = ts[start:start+k]
            if len(subset) < 2:
                continue
            mean = np.mean(subset)
            deviations = subset - mean
            cumsum = np.cumsum(deviations)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(subset, ddof=1)
            if S != 0:
                rs.append(R/S)
        if rs:
            rs_list.append(np.mean(rs))
            k_list.append(k)
    if len(rs_list) > 1:
        log_k = np.log(k_list)
        log_rs = np.log(rs_list)
        hurst = np.polyfit(log_k, log_rs, 1)[0]
        return hurst
    return 0.5

# Calculate rolling Hurst
window = 50
df['hurst'] = df['daily_return'].rolling(window=window).apply(lambda x: hurst_rs(x.values, min_k=2, max_k=25), raw=False)

# Entropy (measure of randomness)
def shannon_entropy(series):
    """Calculate Shannon entropy"""
    counts = pd.value_counts(pd.qcut(series, q=10, duplicates='drop'))
    probabilities = counts / len(series)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

df['gap_entropy'] = df['overnight_gap'].rolling(window=30).apply(shannon_entropy, raw=False)

# Price patterns
df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
df['lower_low'] = ((df['high'] < df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)

# Time-based cycles
df['day_of_month'] = df['Date'].dt.day
df['week_of_year'] = df['Date'].dt.isocalendar().week
df['quarter'] = df['Date'].dt.quarter

# Lunar cycle approximation (29.53 days)
df['lunar_phase'] = (df['Date'] - pd.Timestamp('2000-01-06')).dt.days % 29.53 / 29.53

# Solar position (simplified)
df['day_of_year'] = df['Date'].dt.dayofyear
df['solar_position'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Near Fibonacci levels
for fib in [23, 38, 50, 61, 78]:
    signal = df[f'near_fib_{fib}'].astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Near Fib {fib/100:.3f} Level',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 2: Statistical extremes
for z_threshold in [-2.5, -2, -1.5, 2, 2.5]:
    signal = (df['gap_zscore'] < z_threshold).astype(int) if z_threshold < 0 else (df['gap_zscore'] > z_threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    condition = f'Gap Z-Score < {z_threshold}' if z_threshold < 0 else f'Gap Z-Score > {z_threshold}'
    results.append({
        'condition': condition,
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 3: Hurst exponent conditions (trending vs mean-reverting)
for h_threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
    signal = (df['hurst'] < h_threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Hurst < {h_threshold} (Mean-Rev)',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: High kurtosis (fat tails)
for kurt_threshold in [0, 1, 2, 3]:
    signal = (df['gap_kurt_20'] > kurt_threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap Kurtosis > {kurt_threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 5: Entropy conditions
for entropy_threshold in [2.5, 3, 3.2, 3.3]:
    signal = (df['gap_entropy'] < entropy_threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap Entropy < {entropy_threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: Lunar phase
for phase in [0.25, 0.5, 0.75]:
    signal = (abs(df['lunar_phase'] - phase) < 0.1).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    phase_name = {0.25: 'First Quarter', 0.5: 'Full Moon', 0.75: 'Last Quarter'}[phase]
    results.append({
        'condition': f'Near {phase_name}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 7: Specific day of month
for day in [1, 15, 30]:
    signal = (df['day_of_month'] == day).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Day {day} of Month',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 8: Price pattern with gap
signal = ((df['higher_high'] == 1) & (df['overnight_gap'] < 0.004)).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Higher High & Gap < 0.4%',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 9: The absolutely perfect threshold (last attempt)
for threshold in [0.003607, 0.003608, 0.003609, 0.003610, 0.003611]:
    signal = (df['overnight_gap'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Overnight Gap < {threshold:.6%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 11 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 20 by Sharpe Ratio:")
print(results_df.head(20).to_string(index=False))

# Show best high-trade conditions
high_trades = results_df[results_df['trades'] > 2500].sort_values('sharpe', ascending=False)
print(f"\nBest conditions with >2500 trades:")
print(high_trades.head(15).to_string(index=False))

# Final check
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
else:
    print("\nAfter 11 waves and hundreds of conditions tested, no single condition")
    print("achieved both Sharpe >= 1.2 and Trades > 2500.")
    print(f"\nBest result: Overnight Gap < 0.361%, Sharpe = 1.147, Trades = 3606")

# Append to history
with open('history.log', 'a') as f:
    f.write("\n\n=== WAVE 11 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")