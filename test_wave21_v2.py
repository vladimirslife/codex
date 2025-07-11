import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os
from typing import Optional

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

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def normalized_momentum_quality(df: pd.DataFrame, period: int) -> pd.Series:
    """Normalized momentum quality indicator"""
    close = df['Close']
    
    # Calculate momentum
    momentum = close.pct_change(period)
    
    # Calculate quality: consistency of momentum
    returns = close.pct_change()
    
    # Count positive returns in period
    positive_count = returns.rolling(period).apply(lambda x: (x > 0).sum())
    consistency = positive_count / period
    
    # Normalize momentum by its rolling std
    mom_std = momentum.rolling(period * 2).std()
    normalized_mom = momentum / mom_std.where(mom_std > 0, 1)
    
    # Quality score: normalized momentum * consistency
    quality = normalized_mom * consistency
    
    return quality

def volatility_adjusted_trend(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Trend strength adjusted by volatility regime"""
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

def momentum_divergence_normalized(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Normalized divergence between fast and slow momentum"""
    close = df['Close']
    
    # Calculate momentum
    fast_mom = close.pct_change(fast)
    slow_mom = close.pct_change(slow)
    
    # Normalize each by their own volatility
    fast_vol = fast_mom.rolling(fast * 2).std()
    slow_vol = slow_mom.rolling(slow * 2).std()
    
    fast_norm = fast_mom / fast_vol.where(fast_vol > 0, 1)
    slow_norm = slow_mom / slow_vol.where(slow_vol > 0, 1)
    
    # Divergence
    divergence = fast_norm - slow_norm
    
    return divergence

def relative_volume_momentum(df: pd.DataFrame, period: int) -> pd.Series:
    """Momentum weighted by relative volume strength"""
    close = df['Close']
    volume = df['Volume']
    
    # Price momentum
    momentum = close.pct_change(period)
    
    # Volume relative to its moving average
    vol_ma = volume.rolling(period * 2).mean()
    relative_vol = volume / vol_ma.where(vol_ma > 0, 1)
    
    # Volume momentum
    vol_momentum = relative_vol.rolling(period).mean()
    
    # Combined indicator
    combined = momentum * vol_momentum
    
    # Normalize by rolling std
    combined_std = combined.rolling(period * 2).std()
    normalized = combined / combined_std.where(combined_std > 0, 1)
    
    return normalized

def adaptive_momentum_oscillator(df: pd.DataFrame, min_period: int, max_period: int) -> pd.Series:
    """Momentum with adaptive period based on market conditions"""
    close = df['Close']
    returns = close.pct_change()
    
    # Use efficiency ratio to determine period
    change = (close - close.shift(20)).abs()
    path = returns.abs().rolling(20).sum()
    efficiency = change / path.where(path > 0, 1)
    
    # Map efficiency to period (high efficiency = shorter period)
    adaptive_period = min_period + (max_period - min_period) * (1 - efficiency)
    adaptive_period = adaptive_period.fillna(max_period).astype(int)
    
    # Calculate momentum with adaptive period
    momentum = pd.Series(index=close.index, dtype=float)
    
    for i in range(max_period, len(close)):
        period = min(max(adaptive_period.iloc[i], min_period), max_period)
        if not np.isnan(period) and i >= period:
            momentum.iloc[i] = close.iloc[i] / close.iloc[i-period] - 1
    
    # Normalize
    mom_std = momentum.rolling(max_period).std()
    normalized = momentum / mom_std.where(mom_std > 0, 1)
    
    return normalized

def volatility_regime_momentum(df: pd.DataFrame, mom_period: int, vol_period: int) -> pd.Series:
    """Momentum indicator that adapts to volatility regime"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate momentum
    momentum = close.pct_change(mom_period)
    
    # Volatility regime (z-score)
    vol = returns.rolling(vol_period).std()
    vol_mean = vol.rolling(vol_period * 2).mean()
    vol_std = vol.rolling(vol_period * 2).std()
    vol_zscore = (vol - vol_mean) / vol_std.where(vol_std > 0, 1)
    
    # Adjust momentum threshold based on volatility regime
    # In high vol, require stronger momentum signal
    adjustment_factor = 1 + vol_zscore.clip(-2, 2) * 0.5
    
    # Adjusted momentum
    adjusted_momentum = momentum / adjustment_factor
    
    return adjusted_momentum

def price_acceleration_normalized(df: pd.DataFrame, period: int) -> pd.Series:
    """Normalized price acceleration (2nd derivative)"""
    close = df['Close']
    
    # First derivative: rate of change
    roc1 = close.pct_change(1)
    
    # Second derivative: acceleration
    acceleration = roc1 - roc1.shift(1)
    
    # Smooth and normalize
    smooth_accel = acceleration.rolling(period).mean()
    accel_std = acceleration.rolling(period * 2).std()
    normalized = smooth_accel / accel_std.where(accel_std > 0, 1)
    
    return normalized

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
    
    if cond_name == 'NORM_MOM_QUALITY':
        period = params['period']
        k = params['k']
        quality = normalized_momentum_quality(df, period)
        signal = quality > k
    
    elif cond_name == 'VOL_ADJ_TREND':
        trend_p = params['trend_period']
        vol_p = params['vol_period']
        k = params['k']
        trend = volatility_adjusted_trend(df, trend_p, vol_p)
        signal = trend > k
    
    elif cond_name == 'MOM_DIV_NORM':
        fast = params['fast']
        slow = params['slow']
        k = params['k']
        div = momentum_divergence_normalized(df, fast, slow)
        signal = div > k
    
    elif cond_name == 'REL_VOL_MOM':
        period = params['period']
        k = params['k']
        rvm = relative_volume_momentum(df, period)
        signal = rvm > k
    
    elif cond_name == 'ADAPT_MOM_OSC':
        min_p = params['min_period']
        max_p = params['max_period']
        k = params['k']
        osc = adaptive_momentum_oscillator(df, min_p, max_p)
        signal = osc > k
    
    elif cond_name == 'VOL_REGIME_MOM':
        mom_p = params['mom_period']
        vol_p = params['vol_period']
        k = params['k']
        vrm = volatility_regime_momentum(df, mom_p, vol_p)
        signal = vrm > k
    
    elif cond_name == 'PRICE_ACCEL_NORM':
        period = params['period']
        k = params['k']
        accel = price_acceleration_normalized(df, period)
        signal = accel > k
    
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

# NORM_MOM_QUALITY - Normalized momentum quality
for period in (20, 50, 100, 150):
    for k in (-0.5, 0.0, 0.5, 1.0):
        conditions.append(('NORM_MOM_QUALITY', {'period': period, 'k': k}))

# VOL_ADJ_TREND - Volatility adjusted trend
for trend_p in (50, 100, 150):
    for vol_p in (20, 50):
        for k in (0.0, 0.5, 1.0):
            conditions.append(('VOL_ADJ_TREND', {'trend_period': trend_p, 'vol_period': vol_p, 'k': k}))

# MOM_DIV_NORM - Normalized momentum divergence
for fast in (5, 10, 20):
    for slow in (50, 100, 150):
        for k in (-1.0, -0.5, 0.0, 0.5):
            conditions.append(('MOM_DIV_NORM', {'fast': fast, 'slow': slow, 'k': k}))

# REL_VOL_MOM - Relative volume momentum
for period in (20, 50, 100):
    for k in (-0.5, 0.0, 0.5, 1.0):
        conditions.append(('REL_VOL_MOM', {'period': period, 'k': k}))

# ADAPT_MOM_OSC - Adaptive momentum oscillator
for min_p in (10, 20):
    for max_p in (50, 100):
        for k in (-1.0, -0.5, 0.0, 0.5):
            conditions.append(('ADAPT_MOM_OSC', {'min_period': min_p, 'max_period': max_p, 'k': k}))

# VOL_REGIME_MOM - Volatility regime momentum
for mom_p in (20, 50):
    for vol_p in (20, 50):
        for k in (-0.02, 0.0, 0.02, 0.04):
            conditions.append(('VOL_REGIME_MOM', {'mom_period': mom_p, 'vol_period': vol_p, 'k': k}))

# PRICE_ACCEL_NORM - Normalized price acceleration
for period in (10, 20, 50):
    for k in (-1.0, -0.5, 0.0, 0.5):
        conditions.append(('PRICE_ACCEL_NORM', {'period': period, 'k': k}))

# Limit to MAX_COMBOS
conditions = conditions[:MAX_COMBOS]
print(f"Wave21_v2: total conditions = {len(conditions)}")

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
        if i % 20 == 0:
            print(f"Progress: {i}/{len(conditions)}")
        try:
            result = evaluate(prices, name, params)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {name} with {params}: {e}")
    
    # Sort and display results
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\nTop 10 Wave21_v2:")
    print(res_sorted.head(10).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave21_v2 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave21_v2 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave21_v2 appended to history.log")

if __name__ == "__main__":
    main()