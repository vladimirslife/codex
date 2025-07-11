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
MAX_COMBOS = 1000

# -------------------- Data --------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load data with proper error handling"""
    cache_file = f"{ticker}_{start}_{end}.csv"
    
    try:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure numeric types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except Exception as e:
        print(f"Error reading cache file: {e}")
    
    # Download fresh data
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            # Save to cache
            df.to_csv(cache_file)
            # Ensure numeric types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except Exception as e:
        print(f"Error downloading data: {e}")
    
    # Return empty DataFrame if all fails
    return pd.DataFrame()

# -------------------- Indicators --------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def volatility_adjusted_trend_v1(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Original VOL_ADJ_TREND - best performer from v2"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend using linear regression slope
    slopes = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            slope = np.polyfit(x, y, 1)[0]
            slopes.iloc[i] = slope / close.iloc[i] * 100  # Normalize by price
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Adjust trend by inverse volatility (stronger signal in low vol)
    adjusted_trend = slopes * (1 - vol_percentile)
    
    return adjusted_trend

def volatility_adjusted_trend_v2(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Enhanced version with EMA slope instead of linear regression"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend using EMA slope
    ema_trend = ema(close, trend_period)
    ema_slope = (ema_trend - ema_trend.shift(5)) / ema_trend.shift(5) * 100
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Adjust trend by inverse volatility
    adjusted_trend = ema_slope * (1 - vol_percentile)
    
    return adjusted_trend

def volatility_adjusted_trend_v3(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Version with squared volatility adjustment for stronger effect"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend using linear regression slope
    slopes = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            slope = np.polyfit(x, y, 1)[0]
            slopes.iloc[i] = slope / close.iloc[i] * 100
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Stronger adjustment: square the inverse volatility factor
    adjusted_trend = slopes * ((1 - vol_percentile) ** 2)
    
    return adjusted_trend

def volatility_adjusted_trend_v4(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Version with R-squared weighting for trend quality"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend with R-squared
    slopes = pd.Series(index=close.index, dtype=float)
    r_squared = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            # Linear regression
            x_mean = x.mean()
            y_mean = y.mean()
            
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()
            
            if denominator > 0:
                slope = numerator / denominator
                slopes.iloc[i] = slope / close.iloc[i] * 100
                
                # R-squared
                y_pred = slope * x + (y_mean - slope * x_mean)
                ss_res = ((y - y_pred) ** 2).sum()
                ss_tot = ((y - y_mean) ** 2).sum()
                
                if ss_tot > 0:
                    r_squared.iloc[i] = 1 - (ss_res / ss_tot)
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Adjust trend by both volatility and R-squared
    adjusted_trend = slopes * (1 - vol_percentile) * r_squared
    
    return adjusted_trend

def volatility_adjusted_trend_v5(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Version with adaptive threshold based on market regime"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend
    slopes = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            slope = np.polyfit(x, y, 1)[0]
            slopes.iloc[i] = slope / close.iloc[i] * 100
    
    # Calculate volatility metrics
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Calculate volatility regime (high/medium/low)
    vol_zscore = (vol - vol.rolling(200).mean()) / vol.rolling(200).std()
    
    # Adaptive adjustment: stronger in low vol, weaker in high vol
    base_adjustment = 1 - vol_percentile
    regime_factor = 1 - vol_zscore.clip(-1, 1) * 0.3  # Reduce signal in high vol
    
    adjusted_trend = slopes * base_adjustment * regime_factor
    
    return adjusted_trend

def volatility_adjusted_trend_v6(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Version using price-to-EMA deviation with volatility adjustment"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend as deviation from EMA
    ema_trend = ema(close, trend_period)
    price_deviation = (close - ema_trend) / ema_trend * 100
    
    # Smooth the deviation
    smooth_deviation = price_deviation.rolling(10).mean()
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Invert deviation and adjust by volatility
    # Negative deviation (below EMA) with low volatility = positive signal
    adjusted_trend = -smooth_deviation * (1 - vol_percentile)
    
    return adjusted_trend

# -------------------- Sharpe --------------------

def sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    std = returns.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    excess = returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

# -------------------- Evaluation --------------------

def evaluate(df: pd.DataFrame, cond_name: str, params: dict) -> dict:
    close = df['Close']
    open_next = df['Open'].shift(-1)
    daily_returns = (open_next - close) / close
    
    signal = pd.Series(False, index=df.index)
    
    if cond_name == 'VOL_ADJ_TREND_V1':
        trend_p = params['trend_period']
        vol_p = params['vol_period']
        k = params['k']
        indicator = volatility_adjusted_trend_v1(df, trend_p, vol_p)
        signal = indicator > k
    
    elif cond_name == 'VOL_ADJ_TREND_V2':
        trend_p = params['trend_period']
        vol_p = params['vol_period']
        k = params['k']
        indicator = volatility_adjusted_trend_v2(df, trend_p, vol_p)
        signal = indicator > k
    
    elif cond_name == 'VOL_ADJ_TREND_V3':
        trend_p = params['trend_period']
        vol_p = params['vol_period']
        k = params['k']
        indicator = volatility_adjusted_trend_v3(df, trend_p, vol_p)
        signal = indicator > k
    
    elif cond_name == 'VOL_ADJ_TREND_V4':
        trend_p = params['trend_period']
        vol_p = params['vol_period']
        k = params['k']
        indicator = volatility_adjusted_trend_v4(df, trend_p, vol_p)
        signal = indicator > k
    
    elif cond_name == 'VOL_ADJ_TREND_V5':
        trend_p = params['trend_period']
        vol_p = params['vol_period']
        k = params['k']
        indicator = volatility_adjusted_trend_v5(df, trend_p, vol_p)
        signal = indicator > k
    
    elif cond_name == 'VOL_ADJ_TREND_V6':
        trend_p = params['trend_period']
        vol_p = params['vol_period']
        k = params['k']
        indicator = volatility_adjusted_trend_v6(df, trend_p, vol_p)
        signal = indicator > k
    
    else:
        raise ValueError(f"Unknown condition: {cond_name}")
    
    # Apply signal
    strategy_returns = daily_returns.copy()
    strategy_returns[~signal] = 0.0
    
    # Calculate metrics
    n_trades = int(signal.sum())
    sharpe = sharpe_ratio(strategy_returns.dropna())
    total_return = (1 + strategy_returns.fillna(0)).prod() - 1
    
    return {
        'condition': cond_name,
        'params': params,
        'sharpe': sharpe,
        'trades': n_trades,
        'total_return': total_return
    }

# -------------------- Grid Search --------------------

conditions = []

# Test all versions with various parameters
versions = ['VOL_ADJ_TREND_V1', 'VOL_ADJ_TREND_V2', 'VOL_ADJ_TREND_V3', 
            'VOL_ADJ_TREND_V4', 'VOL_ADJ_TREND_V5', 'VOL_ADJ_TREND_V6']

# Focus on parameters near the best performing ones
trend_periods = [100, 125, 150, 175, 200]
vol_periods = [20, 30, 40, 50, 60]
thresholds = [-0.5, -0.25, 0.0, 0.25, 0.5]

for version in versions:
    for trend_p in trend_periods:
        for vol_p in vol_periods:
            for k in thresholds:
                conditions.append((version, {
                    'trend_period': trend_p, 
                    'vol_period': vol_p, 
                    'k': k
                }))

# Limit to MAX_COMBOS
conditions = conditions[:MAX_COMBOS]
print(f"Wave21_v3: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

def main():
    prices = load_data(TICKER, START_DATE, END_DATE)
    
    if prices.empty:
        print("Error: Unable to load data")
        return
    
    print(f"Data loaded: {len(prices)} rows")
    
    # Run evaluations
    results = []
    for i, (name, params) in enumerate(conditions):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(conditions)}")
        try:
            result = evaluate(prices, name, params)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {name} with {params}: {e}")
    
    # Sort and display results
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\nTop 15 Wave21_v3:")
    print(res_sorted.head(15).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave21_v3 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave21_v3 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave21_v3 appended to history.log")

if __name__ == "__main__":
    main()