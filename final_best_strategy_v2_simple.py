#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Best QQQ Trading Strategy - Version 2 (Simple)
After 20% improvement attempt

Best condition: SPY Gap < 0.194%
"""

import pandas as pd
import numpy as np

# Load data
def load_data():
    """Load QQQ and SPY data"""
    qqq = pd.read_csv("4 - QQQ.csv")
    spy = pd.read_csv("4 - SPY.csv")
    
    # Standardize columns
    qqq.columns = qqq.columns.str.lower()
    spy.columns = spy.columns.str.lower()
    
    # Convert dates
    qqq['date'] = pd.to_datetime(qqq['date'])
    spy['date'] = pd.to_datetime(spy['date'])
    
    # Filter from 2006
    qqq = qqq[qqq['date'] >= '2006-01-01'].reset_index(drop=True)
    spy = spy[spy['date'] >= '2006-01-01'].reset_index(drop=True)
    
    # Merge data
    df = qqq[['date', 'open', 'high', 'low', 'close']].copy()
    df = df.merge(spy[['date', 'open', 'close']], on='date', suffixes=('', '_spy'))
    
    return df

def calculate_strategy(df, spy_gap_threshold=0.00194):
    """
    Calculate strategy returns
    Entry: SPY Gap < threshold
    Buy: Close on day T
    Sell: Open on day T+1
    """
    # Calculate SPY overnight gap
    df['spy_overnight_gap'] = (df['open_spy'] - df['close_spy'].shift(1)) / df['close_spy'].shift(1)
    
    # Entry signal
    df['signal'] = (df['spy_overnight_gap'] < spy_gap_threshold).astype(int)
    
    # Calculate returns
    df['next_open'] = df['open'].shift(-1)
    df['overnight_return'] = df['next_open'] / df['close'] - 1
    df['strategy_return'] = df['signal'] * df['overnight_return']
    
    # Handle missing values
    df['strategy_return'].fillna(0, inplace=True)
    
    return df

def calculate_metrics(df, annual_rf=0.02):
    """Calculate performance metrics"""
    # Daily returns
    strategy_returns = df['strategy_return']
    
    # Sharpe Ratio
    daily_rf = annual_rf / 252
    excess_returns = strategy_returns - daily_rf
    sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    # Total return
    total_return = (1 + strategy_returns).prod() - 1
    
    # CAGR
    years = len(df) / 252
    cagr = (1 + total_return) ** (1/years) - 1
    
    # Max Drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    trades = df[df['signal'] == 1]
    win_rate = (trades['strategy_return'] > 0).sum() / len(trades) if len(trades) > 0 else 0
    
    # Number of trades
    num_trades = df['signal'].sum()
    
    return {
        'sharpe_ratio': sharpe,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades
    }

def main():
    """Run the analysis"""
    print("=== FINAL BEST QQQ TRADING STRATEGY V2 ===")
    print("Condition: SPY Gap < 0.194%")
    print("(Trading QQQ based on SPY gap)")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Calculate strategy
    df = calculate_strategy(df, spy_gap_threshold=0.00194)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Print results
    print(f"\nPerformance Metrics:")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Avg Trades per Year: {metrics['num_trades'] / (len(df) / 252):.1f}")
    
    # Compare with original strategy
    print("\n" + "=" * 50)
    print("Comparison with original strategy:")
    print("Original: QQQ Gap < 0.3607%, Sharpe 1.151")
    print(f"New: SPY Gap < 0.194%, Sharpe {metrics['sharpe_ratio']:.4f}")
    print(f"Improvement: {(metrics['sharpe_ratio'] / 1.151 - 1) * 100:.1f}%")
    
    # Save key trades
    trades = df[df['signal'] == 1][['date', 'spy_overnight_gap', 'overnight_return', 'strategy_return']]
    print(f"\n10 Most Recent Trades:")
    print(trades.tail(10).to_string(index=False))
    
    # Save trades to CSV
    trades.to_csv('spy_gap_trades.csv', index=False)
    print("\nFull trade history saved to 'spy_gap_trades.csv'")

if __name__ == "__main__":
    main()