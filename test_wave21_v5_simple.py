import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os

TICKER = "SPY"
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RF_ANNUAL = 0.02
RF_DAILY = RF_ANNUAL / 252.0

# -------------------- Data --------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load data with proper error handling"""
    cache_file = f"{ticker}_{start}_{end}.csv"
    
    try:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if isinstance(df, pd.DataFrame):
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
    except Exception as e:
        print(f"Error reading cache: {e}")
    
    return pd.DataFrame()

# -------------------- Best VOL_ADJ_TREND Variations --------------------

def enhanced_vol_adj_trend_best(df: pd.DataFrame, trend_period: int, vol_period: int, enhancement: str = 'standard') -> pd.Series:
    """Best performing VOL_ADJ_TREND with enhancements"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend using linear regression slope
    slopes = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
            except:
                slopes.iloc[i] = 0
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Apply enhancement
    if enhancement == 'standard':
        # Original best performer
        adjusted_trend = slopes * (1 - vol_percentile)
    
    elif enhancement == 'squared':
        # Squared adjustment for stronger effect
        adjusted_trend = slopes * ((1 - vol_percentile) ** 2)
    
    elif enhancement == 'exponential':
        # Exponential adjustment
        adjusted_trend = slopes * np.exp(-vol_percentile)
    
    elif enhancement == 'adaptive':
        # Adaptive based on volatility level
        low_vol = vol_percentile < 0.3
        mid_vol = (vol_percentile >= 0.3) & (vol_percentile < 0.7)
        high_vol = vol_percentile >= 0.7
        
        adjusted_trend = pd.Series(index=close.index, dtype=float)
        adjusted_trend[low_vol] = slopes[low_vol] * 1.2  # Boost in low vol
        adjusted_trend[mid_vol] = slopes[mid_vol] * (1 - vol_percentile[mid_vol])
        adjusted_trend[high_vol] = slopes[high_vol] * (1 - vol_percentile[high_vol]) * 0.5  # Reduce in high vol
    
    return adjusted_trend

# -------------------- Evaluation --------------------

def sharpe_ratio(returns: pd.Series) -> float:
    """Calculate Sharpe ratio"""
    if returns.empty or len(returns) < 20:
        return 0.0
    
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    
    std = clean_returns.std(ddof=1)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    
    excess = clean_returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

def evaluate(df: pd.DataFrame, params: dict) -> dict:
    """Evaluate strategy"""
    if df.empty:
        return {
            'params': params,
            'sharpe': 0.0,
            'trades': 0,
            'total_return': 0.0
        }
    
    close = df['Close']
    open_next = df['Open'].shift(-1)
    daily_returns = (open_next - close) / close
    
    # Get indicator
    indicator = enhanced_vol_adj_trend_best(
        df, 
        params['trend_period'], 
        params['vol_period'],
        params['enhancement']
    )
    
    # Generate signal
    signal = indicator > params['threshold']
    
    # Apply signal
    strategy_returns = daily_returns.copy()
    strategy_returns[~signal] = 0.0
    
    # Calculate metrics
    n_trades = int(signal.sum())
    sharpe = sharpe_ratio(strategy_returns.dropna())
    
    try:
        total_return = (1 + strategy_returns.fillna(0)).prod() - 1
    except:
        total_return = 0.0
    
    return {
        'params': params,
        'sharpe': sharpe,
        'trades': n_trades,
        'total_return': total_return
    }

# -------------------- Main --------------------

def main():
    print("Wave21_v5 Simplified: Best VOL_ADJ_TREND Parameters")
    print("=" * 60)
    
    # Load data
    prices = load_data(TICKER, START_DATE, END_DATE)
    
    if prices.empty:
        print("Error: Unable to load data")
        return
    
    print(f"Data loaded: {len(prices)} rows")
    
    # Test focused parameter sets around best performers
    results = []
    
    # Test parameters close to the best (150 trend, 40 vol, 0.0 threshold)
    trend_periods = [140, 145, 150, 155, 160]
    vol_periods = [35, 38, 40, 42, 45]
    thresholds = [-0.05, 0.0, 0.05]
    enhancements = ['standard', 'squared', 'exponential', 'adaptive']
    
    total_tests = len(trend_periods) * len(vol_periods) * len(thresholds) * len(enhancements)
    print(f"Total conditions to test: {total_tests}")
    
    test_count = 0
    for trend_p in trend_periods:
        for vol_p in vol_periods:
            for threshold in thresholds:
                for enhancement in enhancements:
                    test_count += 1
                    if test_count % 50 == 0:
                        print(f"Progress: {test_count}/{total_tests}")
                    
                    params = {
                        'trend_period': trend_p,
                        'vol_period': vol_p,
                        'threshold': threshold,
                        'enhancement': enhancement
                    }
                    
                    try:
                        result = evaluate(prices, params)
                        results.append(result)
                    except Exception as e:
                        print(f"Error with {params}: {e}")
    
    # Sort and display results
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\n" + "=" * 60)
    print("Top 20 Results:")
    print("=" * 60)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave21_v5_simplified ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave21_v5_simp | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 appended to history.log")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Best Sharpe: {res_sorted.iloc[0]['sharpe']:.3f}")
    print(f"Best Parameters: {res_sorted.iloc[0]['params']}")
    print(f"Strategies with Sharpe > 0.95: {len(res_sorted[res_sorted['sharpe'] > 0.95])}")
    print(f"Strategies with Sharpe > 1.0: {len(res_sorted[res_sorted['sharpe'] > 1.0])}")

if __name__ == "__main__":
    main()