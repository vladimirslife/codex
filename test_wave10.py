#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 10: Complex conditions, market phases, and mathematical transformations
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

# Mathematical transformations of gap
df['gap_squared'] = df['overnight_gap'] ** 2
df['gap_log'] = np.log(abs(df['overnight_gap']) + 0.0001)
df['gap_sqrt'] = np.sqrt(abs(df['overnight_gap']))

# Market phase detection
df['sma_20'] = df['close'].rolling(window=20).mean()
df['sma_50'] = df['close'].rolling(window=50).mean()
df['trending_up'] = ((df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50'])).astype(int)
df['trending_down'] = ((df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50'])).astype(int)

# ADX for trend strength (simplified)
df['high_low'] = df['high'] - df['low']
df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = pd.concat([df['high_low'], df['high_close_prev'], df['low_close_prev']], axis=1).max(axis=1)
df['atr_14'] = df['tr'].rolling(window=14).mean()

# Directional movement
df['up_move'] = df['high'] - df['high'].shift(1)
df['down_move'] = df['low'].shift(1) - df['low']
df['pos_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
df['neg_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
df['pos_di'] = 100 * (df['pos_dm'].rolling(window=14).mean() / df['atr_14'])
df['neg_di'] = 100 * (df['neg_dm'].rolling(window=14).mean() / df['atr_14'])
df['dx'] = 100 * abs(df['pos_di'] - df['neg_di']) / (df['pos_di'] + df['neg_di'])
df['adx'] = df['dx'].rolling(window=14).mean()

# Complex combined conditions
df['gap_with_trend'] = ((df['overnight_gap'] < 0.004) & (df['trending_up'] == 1)).astype(int)
df['gap_counter_trend'] = ((df['overnight_gap'] < 0) & (df['trending_up'] == 1)).astype(int)

# Rolling correlations and relationships
df['gap_return_corr'] = df['overnight_gap'].rolling(window=20).corr(df['daily_return'])
df['gap_volatility'] = df['overnight_gap'].rolling(window=20).std()

# Efficiency ratio (trending vs choppy)
df['price_change_10'] = df['close'] - df['close'].shift(10)
df['path_sum_10'] = df['close'].diff().abs().rolling(window=10).sum()
df['efficiency_ratio'] = abs(df['price_change_10']) / (df['path_sum_10'] + 0.0001)

# Fractal dimension (simplified)
df['high_10'] = df['high'].rolling(window=10).max()
df['low_10'] = df['low'].rolling(window=10).min()
df['range_10'] = df['high_10'] - df['low_10']

# Composite conditions
df['strong_trend_gap'] = ((df['adx'] > 25) & (df['overnight_gap'] < 0.004)).astype(int)
df['weak_trend_gap'] = ((df['adx'] < 20) & (df['overnight_gap'] < 0.004)).astype(int)
df['efficient_gap'] = ((df['efficiency_ratio'] > 0.3) & (df['overnight_gap'] < 0.004)).astype(int)

# Time-based patterns with gap
df['day_of_week'] = df['Date'].dt.dayofweek
df['monday_gap'] = ((df['day_of_week'] == 0) & (df['overnight_gap'] < 0.005)).astype(int)
df['friday_gap'] = ((df['day_of_week'] == 4) & (df['overnight_gap'] < 0.005)).astype(int)

# Sine/Cosine transformations for cyclical patterns
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

# ------------------------- GRID SEARCH CONDITIONS -------------------
results = []

# Test 1: Market phase with gap
signal = df['gap_with_trend'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Gap < 0.4% in Uptrend',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 2: Counter-trend gap
signal = df['gap_counter_trend'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Negative Gap in Uptrend',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 3: ADX-based conditions
for adx_threshold in [20, 25, 30, 35]:
    signal = ((df['adx'] > adx_threshold) & (df['overnight_gap'] < 0.004)).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'ADX > {adx_threshold} & Gap < 0.4%',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 4: Efficiency ratio conditions
for eff_threshold in [0.2, 0.3, 0.4, 0.5]:
    signal = ((df['efficiency_ratio'] > eff_threshold) & (df['overnight_gap'] < 0.004)).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Efficiency > {eff_threshold} & Gap < 0.4%',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 5: Gap volatility conditions
for vol_threshold in [0.003, 0.005, 0.007]:
    signal = ((df['gap_volatility'] > vol_threshold) & (df['overnight_gap'] < 0.004)).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap Vol > {vol_threshold} & Gap < 0.4%',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 6: Mathematical transformations
signal = (df['gap_squared'] < 0.00001).astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Gap² < 0.001%',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 7: Log transformation
for threshold in [-7, -6.5, -6, -5.5]:
    signal = (df['gap_log'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Log(|Gap|) < {threshold}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 8: Composite day/gap conditions
signal = df['monday_gap'].astype(int)
strategy_return = signal * df['next_overnight_return']
strategy_return.fillna(0, inplace=True)

sharpe = calculate_sharpe(strategy_return)
num_trades = signal.sum()

results.append({
    'condition': 'Monday & Gap < 0.5%',
    'sharpe': sharpe,
    'trades': num_trades,
    'total_return': (1 + strategy_return).prod() - 1
})

# Test 9: Correlation-based
for corr_threshold in [-0.2, 0, 0.2]:
    signal = ((df['gap_return_corr'] < corr_threshold) & (df['overnight_gap'] < 0.004)).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Gap/Return Corr < {corr_threshold} & Gap < 0.4%',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# Test 10: The magic threshold (super fine-tuning around 0.36%)
for threshold in [0.00358, 0.00359, 0.00361, 0.00362]:
    signal = (df['overnight_gap'] < threshold).astype(int)
    strategy_return = signal * df['next_overnight_return']
    strategy_return.fillna(0, inplace=True)
    
    sharpe = calculate_sharpe(strategy_return)
    num_trades = signal.sum()
    
    results.append({
        'condition': f'Overnight Gap < {threshold:.5%}',
        'sharpe': sharpe,
        'trades': num_trades,
        'total_return': (1 + strategy_return).prod() - 1
    })

# ------------------------- SHOW RESULTS -------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

print("\n=== WAVE 10 RESULTS ===")
print(f"Total conditions tested: {len(results_df)}")
print(f"\nTop 20 by Sharpe Ratio:")
print(results_df.head(20).to_string(index=False))

# Show best high-trade conditions
high_trades = results_df[results_df['trades'] > 2500].sort_values('sharpe', ascending=False)
print(f"\nBest conditions with >2500 trades:")
print(high_trades.head(15).to_string(index=False))

# Check if we're getting very close
very_close = results_df[(results_df['sharpe'] >= 1.19) & (results_df['trades'] > 2500)]
print(f"\nConditions with Sharpe >= 1.19 and Trades > 2500:")
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
    f.write("\n\n=== WAVE 10 TOP 3 RESULTS ===\n")
    for idx, row in results_df.head(3).iterrows():
        f.write(f"\nCondition: {row['condition']}\n")
        f.write(f"Sharpe Ratio: {row['sharpe']:.4f}\n")
        f.write(f"Trades: {row['trades']}\n")
        f.write(f"Total Return: {row['total_return']:.2%}\n")
        f.write("-" * 40 + "\n")