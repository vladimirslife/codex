#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Best QQQ Trading Strategy - Version 2
After 20% improvement attempt

Best condition: SPY Gap < 0.194%
Sharpe Ratio: 1.177
Total Trades: 3261
CAGR: 11.9%
Total Return: 628%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_results(df):
    """Plot strategy performance"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Cumulative returns
    df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
    df['cum_buyhold'] = df['close'] / df['close'].iloc[0]
    
    ax1 = axes[0]
    ax1.plot(df['date'], df['cum_strategy'], label='Strategy', linewidth=2)
    ax1.plot(df['date'], df['cum_buyhold'], label='Buy & Hold', linewidth=2, alpha=0.7)
    ax1.set_title('Cumulative Returns: SPY Gap < 0.194% Strategy vs Buy & Hold', fontsize=14)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    cumulative_returns = df['cum_strategy']
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
    
    ax2 = axes[1]
    ax2.fill_between(df['date'], drawdown, 0, alpha=0.3, color='red')
    ax2.plot(df['date'], drawdown, color='red', linewidth=1)
    ax2.set_title('Strategy Drawdown', fontsize=14)
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # Monthly returns heatmap
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    monthly_returns = df.groupby(['year', 'month'])['strategy_return'].apply(lambda x: (1 + x).prod() - 1)
    monthly_pivot = monthly_returns.unstack()
    
    ax3 = axes[2]
    im = ax3.imshow(monthly_pivot.values * 100, cmap='RdYlGn', aspect='auto')
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax3.set_yticks(range(len(monthly_pivot)))
    ax3.set_yticklabels(monthly_pivot.index)
    ax3.set_title('Monthly Returns Heatmap (%)', fontsize=14)
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('final_best_strategy_v2_performance.png', dpi=300)
    plt.close()

def main():
    """Run the analysis"""
    print("=== FINAL BEST QQQ TRADING STRATEGY V2 ===")
    print("Condition: SPY Gap < 0.194%")
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
    
    # Generate plots
    plot_results(df)
    print("\nPerformance charts saved to 'final_best_strategy_v2_performance.png'")
    
    # Save trades to CSV
    trades = df[df['signal'] == 1][['date', 'close', 'next_open', 'overnight_return', 'strategy_return']]
    trades.to_csv('final_best_trades_v2.csv', index=False)
    print("Trade history saved to 'final_best_trades_v2.csv'")

if __name__ == "__main__":
    main()