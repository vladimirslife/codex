#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Trading Strategy Implementation
Sharpe Ratio: 1.1175 (79.8% of 1.4 target)
Condition: weighted_vol_ewm_25 < 0.0166
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OvernightVolatilityStrategy:
    """
    Trading strategy based on microstructure-weighted cross-ticker volatility.
    Enters long positions when volatility is low and expected to persist.
    """
    
    def __init__(self, alpha=0.25, threshold=0.0166, spy_weight=0.5, xlk_weight=0.3):
        """
        Initialize strategy parameters.
        
        Args:
            alpha: EWM decay factor (default: 0.25)
            threshold: Entry threshold (default: 0.0166)
            spy_weight: Weight for SPY in calculation (default: 0.5)
            xlk_weight: Weight for XLK in calculation (default: 0.3)
        """
        self.alpha = alpha
        self.threshold = threshold
        self.spy_weight = spy_weight
        self.xlk_weight = xlk_weight
        self.normalization = 1.8  # QQQ weight (1.0) + spy_weight + xlk_weight
        
    def calculate_noise_weight(self, hl_range, oc_range):
        """
        Calculate microstructure noise weight.
        
        Args:
            hl_range: High-Low range normalized by open
            oc_range: Open-Close range normalized by open
            
        Returns:
            Noise weight for volatility adjustment
        """
        noise_proxy = hl_range / (np.abs(oc_range) + 0.0001)
        noise_proxy_log = np.log1p(noise_proxy)
        noise_weight = 1 / (noise_proxy_log + 1)
        return noise_weight
    
    def calculate_weighted_volatility(self, qqq_data, spy_data, xlk_data):
        """
        Calculate microstructure-weighted cross-ticker volatility.
        
        Args:
            qqq_data: DataFrame with QQQ OHLC data
            spy_data: DataFrame with SPY OHLC data
            xlk_data: DataFrame with XLK OHLC data
            
        Returns:
            Series with weighted volatility values
        """
        # Calculate high-low range for each ticker
        qqq_hl = (qqq_data['high'] - qqq_data['low']) / qqq_data['open']
        spy_hl = (spy_data['high'] - spy_data['low']) / spy_data['open']
        xlk_hl = (xlk_data['high'] - xlk_data['low']) / xlk_data['open']
        
        # Calculate noise weight for QQQ (primary ticker)
        qqq_oc_range = (qqq_data['close'] - qqq_data['open']) / qqq_data['open']
        noise_weight = self.calculate_noise_weight(qqq_hl, qqq_oc_range)
        
        # Calculate weighted volatility
        weighted_vol = (
            qqq_hl * noise_weight + 
            spy_hl * self.spy_weight + 
            xlk_hl * self.xlk_weight
        ) / self.normalization
        
        return weighted_vol
    
    def generate_signals(self, qqq_data, spy_data, xlk_data):
        """
        Generate trading signals based on the strategy.
        
        Args:
            qqq_data: DataFrame with QQQ OHLC data
            spy_data: DataFrame with SPY OHLC data
            xlk_data: DataFrame with XLK OHLC data
            
        Returns:
            DataFrame with signals and strategy metrics
        """
        # Ensure data is aligned by date
        data = qqq_data.copy()
        data = data.merge(spy_data[['Date', 'open', 'high', 'low', 'close']], 
                         on='Date', suffixes=('', '_spy'))
        data = data.merge(xlk_data[['Date', 'open', 'high', 'low', 'close']], 
                         on='Date', suffixes=('', '_xlk'))
        
        # Calculate weighted volatility
        weighted_vol = self.calculate_weighted_volatility(
            data[['open', 'high', 'low', 'close']], 
            data[['open_spy', 'high_spy', 'low_spy', 'close_spy']].rename(columns=lambda x: x.replace('_spy', '')),
            data[['open_xlk', 'high_xlk', 'low_xlk', 'close_xlk']].rename(columns=lambda x: x.replace('_xlk', ''))
        )
        
        # Apply exponentially weighted moving average
        weighted_vol_ewm = weighted_vol.ewm(alpha=self.alpha, adjust=False).mean()
        
        # Generate signals (1 = long, 0 = no position)
        signals = (weighted_vol_ewm.shift(1) < self.threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': data['Date'],
            'weighted_vol': weighted_vol,
            'weighted_vol_ewm': weighted_vol_ewm,
            'signal': signals,
            'close': data['close']
        })
        
        # Calculate overnight returns for backtesting
        results['next_open'] = data['open'].shift(-1)
        results['overnight_return'] = (results['next_open'] - results['close']) / results['close']
        results['strategy_return'] = results['signal'] * results['overnight_return']
        
        return results
    
    def backtest(self, results, risk_free_rate=0.02):
        """
        Calculate strategy performance metrics.
        
        Args:
            results: DataFrame from generate_signals
            risk_free_rate: Annual risk-free rate (default: 0.02)
            
        Returns:
            Dictionary with performance metrics
        """
        # Filter valid returns
        valid_returns = results['strategy_return'].dropna()
        
        # Calculate metrics
        daily_rf = risk_free_rate / 252
        excess_returns = valid_returns - daily_rf
        
        # Sharpe Ratio
        mean_excess_annual = excess_returns.mean() * 252
        std_excess_annual = excess_returns.std() * np.sqrt(252)
        sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0
        
        # CAGR
        cumulative_return = (1 + valid_returns).prod() - 1
        years = len(valid_returns) / 252
        cagr = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Other metrics
        num_trades = results['signal'].sum()
        win_rate = (results[results['signal'] == 1]['strategy_return'] > 0).mean()
        
        # Maximum drawdown
        equity_curve = (1 + valid_returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'cagr': cagr,
            'total_return': cumulative_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'avg_trade_return': results[results['signal'] == 1]['strategy_return'].mean(),
            'best_trade': results[results['signal'] == 1]['strategy_return'].max(),
            'worst_trade': results[results['signal'] == 1]['strategy_return'].min()
        }

def main():
    """Example usage of the strategy."""
    print("=== Overnight Volatility Trading Strategy ===")
    print("Loading data...")
    
    # Load data (replace with actual data loading)
    qqq_data = pd.read_csv("4 - QQQ.csv")
    spy_data = pd.read_csv("4 - SPY.csv")
    xlk_data = pd.read_csv("4 - XLK.csv")
    
    # Standardize column names
    for df in [qqq_data, spy_data, xlk_data]:
        df.rename(columns=lambda x: x.lower(), inplace=True)
        df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
    
    # Initialize strategy
    strategy = OvernightVolatilityStrategy()
    
    # Generate signals
    print("\nGenerating signals...")
    results = strategy.generate_signals(qqq_data, spy_data, xlk_data)
    
    # Backtest
    print("\nBacktesting strategy...")
    metrics = strategy.backtest(results)
    
    # Display results
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"CAGR: {metrics['cagr']*100:.2f}%")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Average Trade Return: {metrics['avg_trade_return']*100:.3f}%")
    print(f"Best Trade: {metrics['best_trade']*100:.3f}%")
    print(f"Worst Trade: {metrics['worst_trade']*100:.3f}%")
    
    # Save trade log
    print("\nSaving trade log...")
    trade_log = results[results['signal'] == 1][['Date', 'weighted_vol_ewm', 'overnight_return', 'strategy_return']]
    trade_log.to_csv("final_trade_log.csv", index=False)
    print(f"Trade log saved to final_trade_log.csv")
    
    # Plot equity curve (optional)
    try:
        import matplotlib.pyplot as plt
        
        equity_curve = (1 + results['strategy_return'].fillna(0)).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(results['Date'], equity_curve, label='Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Overnight Volatility Strategy Equity Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('equity_curve.png')
        print("\nEquity curve saved to equity_curve.png")
    except ImportError:
        print("\nMatplotlib not available for plotting")

if __name__ == "__main__":
    main()