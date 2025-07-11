#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Trading Strategy Implementation
Strategy: Geometric mean of cross-ticker volatility < 0.025
Sharpe Ratio: 0.9680
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare_data(qqq_path, spy_path, xlk_path):
    """
    Load and prepare data for the trading strategy
    """
    # Load data
    qqq = pd.read_csv(qqq_path)
    spy = pd.read_csv(spy_path)
    xlk = pd.read_csv(xlk_path)
    
    # Standardize column names and dates
    for df in [qqq, spy, xlk]:
        df.rename(columns=lambda c: c.lower(), inplace=True)
        df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(by="Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # Calculate High-Low range for each ticker
    qqq["hl_range"] = (qqq["high"] - qqq["low"]) / qqq["open"]
    spy["hl_range"] = (spy["high"] - spy["low"]) / spy["open"]
    xlk["hl_range"] = (xlk["high"] - xlk["low"]) / xlk["open"]
    
    # Merge data
    data = qqq[["Date", "open", "close", "hl_range"]].copy()
    data = data.merge(spy[["Date", "hl_range"]], on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[["Date", "hl_range"]], on="Date", suffixes=("", "_xlk"))
    
    # Calculate geometric mean of volatility
    data["geom_mean_vol"] = np.cbrt(
        data["hl_range"] * data["hl_range_spy"] * data["hl_range_xlk"]
    )
    
    # Apply 5-day moving average
    data["geom_mean_vol_ma5"] = data["geom_mean_vol"].rolling(window=5).mean()
    
    # Calculate overnight returns for backtesting
    data["prev_close"] = data["close"].shift(1)
    data["next_open"] = data["open"].shift(-1)
    data["next_overnight_return"] = (data["next_open"] - data["close"]) / data["close"]
    
    # Filter data from 2006 onwards
    data = data[data["Date"] >= pd.Timestamp("2006-01-01")].copy()
    
    return data

def generate_signals(data, threshold=0.025):
    """
    Generate trading signals based on the strategy
    """
    # Signal: Buy when geometric mean volatility MA5 < threshold
    data["signal"] = (data["geom_mean_vol_ma5"].shift(1) < threshold).astype(int)
    
    # Remove NaN signals
    data["signal"] = data["signal"].fillna(0)
    
    return data

def backtest_strategy(data, initial_capital=100000, risk_free_rate=0.02):
    """
    Backtest the trading strategy
    """
    # Calculate strategy returns
    data["strategy_return"] = data["signal"] * data["next_overnight_return"]
    data["strategy_return"] = data["strategy_return"].fillna(0)
    
    # Calculate cumulative returns
    data["strategy_equity"] = initial_capital * (1 + data["strategy_return"]).cumprod()
    
    # Calculate performance metrics
    daily_rf = risk_free_rate / 252
    excess_returns = data["strategy_return"] - daily_rf
    
    # Sharpe Ratio
    mean_excess_annual = excess_returns.mean() * 252
    std_excess_annual = excess_returns.std() * np.sqrt(252)
    sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0
    
    # CAGR
    total_return = data["strategy_equity"].iloc[-1] / initial_capital - 1
    years = (data["Date"].iloc[-1] - data["Date"].iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Other metrics
    num_trades = data["signal"].sum()
    win_rate = (data[data["signal"] == 1]["strategy_return"] > 0).mean()
    max_drawdown = calculate_max_drawdown(data["strategy_equity"])
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "cagr": cagr,
        "total_return": total_return,
        "num_trades": int(num_trades),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "final_equity": data["strategy_equity"].iloc[-1]
    }

def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from equity curve
    """
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def print_performance_summary(metrics):
    """
    Print a formatted performance summary
    """
    print("\n" + "="*50)
    print("TRADING STRATEGY PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Strategy: Geometric Mean Volatility < 0.025")
    print(f"Period: 2006-2024")
    print("-"*50)
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"CAGR: {metrics['cagr']*100:.2f}%")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']:,}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Final Equity: ${metrics['final_equity']:,.2f}")
    print("="*50)

def save_trade_log(data, output_path="trade_log.csv"):
    """
    Save trade log for analysis
    """
    trades = data[data["signal"] == 1].copy()
    trades = trades[["Date", "signal", "next_overnight_return", "strategy_equity"]]
    trades.columns = ["Date", "Signal", "Return", "Equity"]
    trades.to_csv(output_path, index=False)
    print(f"\nTrade log saved to: {output_path}")

def main():
    """
    Main execution function
    """
    print("Loading data...")
    data = load_and_prepare_data(
        "4 - QQQ.csv",
        "4 - SPY.csv", 
        "4 - XLK.csv"
    )
    
    print("Generating signals...")
    data = generate_signals(data, threshold=0.025)
    
    print("Backtesting strategy...")
    metrics = backtest_strategy(data)
    
    print_performance_summary(metrics)
    
    # Optional: Save trade log
    save_trade_log(data)
    
    # Optional: Plot equity curve
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(data["Date"], data["strategy_equity"], label="Strategy Equity")
        plt.title("Trading Strategy Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("equity_curve.png")
        print("\nEquity curve saved to: equity_curve.png")
    except ImportError:
        print("\nMatplotlib not available for plotting")
    
    return data, metrics

if __name__ == "__main__":
    data, metrics = main()