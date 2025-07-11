import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os
from typing import Optional, Tuple

TICKER = "SPY"
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RF_ANNUAL = 0.02
RF_DAILY = RF_ANNUAL / 252.0
MAX_COMBOS = 1000

# -------------------- Data --------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load data with proper error handling and type safety"""
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
            else:
                return pd.DataFrame()
    except Exception as e:
        print(f"Error reading cache file: {e}")
    
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(data, pd.DataFrame) and not data.empty:
            df = data.copy()
            df.to_csv(cache_file)
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except Exception as e:
        print(f"Error downloading data: {e}")
    
    return pd.DataFrame()

# -------------------- Enhanced VOL_ADJ_TREND Variations --------------------

def enhanced_vol_adj_trend_v1(df: pd.DataFrame, trend_period: int, vol_period: int, vol_weight: float) -> pd.Series:
    """Enhanced VOL_ADJ_TREND with adjustable volatility weight"""
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
    
    # Calculate volatility percentile with different window
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Adjust trend by inverse volatility with weight parameter
    adjusted_trend = slopes * (1 - vol_percentile * vol_weight)
    
    return adjusted_trend

def enhanced_vol_adj_trend_v2(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """VOL_ADJ_TREND with exponential volatility adjustment"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend using linear regression slope
    slopes = pd.Series(index=close.index, dtype=float)
    r_squared = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                # Linear regression
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
                
                # Calculate R-squared
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                if ss_tot > 0:
                    r_squared.iloc[i] = 1 - (ss_res / ss_tot)
                else:
                    r_squared.iloc[i] = 0
            except:
                slopes.iloc[i] = 0
                r_squared.iloc[i] = 0
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Exponential adjustment for stronger effect in low volatility
    vol_adjustment = np.exp(-2 * vol_percentile)
    
    # Combine with R-squared for quality filtering
    adjusted_trend = slopes * vol_adjustment * r_squared
    
    return adjusted_trend

def enhanced_vol_adj_trend_v3(df: pd.DataFrame, trend_period: int, vol_period: int, lookback_mult: float) -> pd.Series:
    """VOL_ADJ_TREND with dynamic volatility lookback"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend
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
    
    # Dynamic volatility lookback based on market conditions
    base_vol = returns.rolling(vol_period).std()
    vol_of_vol = base_vol.rolling(vol_period).std()
    
    # Adjust lookback: longer in stable markets, shorter in volatile markets
    vol_percentile_vov = vol_of_vol.rolling(100).rank(pct=True)
    dynamic_lookback = (vol_period * lookback_mult * (1 + vol_percentile_vov)).astype(int).clip(20, 200)
    
    # Calculate volatility percentile with dynamic lookback
    vol_percentiles = pd.Series(index=close.index, dtype=float)
    
    for i in range(200, len(close)):
        lookback = int(dynamic_lookback.iloc[i]) if not pd.isna(dynamic_lookback.iloc[i]) else vol_period * 2
        if i >= lookback:
            vol_window = base_vol.iloc[i-lookback:i+1]
            vol_percentiles.iloc[i] = vol_window.rank(pct=True).iloc[-1]
    
    # Adjust trend
    adjusted_trend = slopes * (1 - vol_percentiles)
    
    return adjusted_trend

def enhanced_vol_adj_trend_v4(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """VOL_ADJ_TREND with market regime adaptation"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend
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
    
    # Market regime detection
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()
    bull_market = ma_50 > ma_200
    
    # Volatility metrics
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Different adjustments for bull/bear markets
    adjusted_trend = pd.Series(index=close.index, dtype=float)
    
    # Bull market: standard adjustment
    adjusted_trend[bull_market] = slopes[bull_market] * (1 - vol_percentile[bull_market])
    
    # Bear market: stronger volatility penalty
    adjusted_trend[~bull_market] = slopes[~bull_market] * (1 - vol_percentile[~bull_market] * 1.5).clip(0, 1)
    
    return adjusted_trend

def enhanced_vol_adj_trend_v5(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """VOL_ADJ_TREND with volume confirmation"""
    close = df['Close']
    volume = df['Volume']
    returns = close.pct_change()
    
    # Calculate trend
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
    
    # Volatility adjustment
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Volume confirmation
    vol_ma = volume.rolling(vol_period).mean()
    relative_volume = volume / vol_ma.where(vol_ma > 0, 1)
    vol_percentile_volume = relative_volume.rolling(vol_period).rank(pct=True)
    
    # Combined adjustment: volatility and volume
    volatility_factor = 1 - vol_percentile
    volume_factor = 0.5 + 0.5 * vol_percentile_volume  # Range: 0.5 to 1.0
    
    adjusted_trend = slopes * volatility_factor * volume_factor
    
    return adjusted_trend

def enhanced_vol_adj_trend_v6(df: pd.DataFrame, trend_period: int, vol_period: int, smooth_period: int) -> pd.Series:
    """VOL_ADJ_TREND with smoothing and optimization"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend with smoothing
    slopes = pd.Series(index=close.index, dtype=float)
    
    # Use smoothed prices for trend calculation
    smooth_close = close.rolling(smooth_period).mean()
    
    for i in range(trend_period, len(close)):
        if i >= trend_period + smooth_period:
            y = smooth_close.iloc[i-trend_period:i].values
            x = np.arange(trend_period)
            if len(y) == trend_period and not pd.isna(y).any():
                try:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.iloc[i] = slope / close.iloc[i] * 100
                except:
                    slopes.iloc[i] = 0
    
    # Volatility adjustment with different calculation
    vol = returns.rolling(vol_period).std()
    
    # Use rank within expanding window for more stable percentiles
    vol_percentile = pd.Series(index=close.index, dtype=float)
    
    for i in range(vol_period * 2, len(close)):
        window_start = max(0, i - vol_period * 4)
        vol_window = vol.iloc[window_start:i+1]
        if len(vol_window) > 0:
            vol_percentile.iloc[i] = vol_window.rank(pct=True).iloc[-1]
    
    # Final adjustment
    adjusted_trend = slopes * (1 - vol_percentile)
    
    return adjusted_trend

# -------------------- Sharpe --------------------

def sharpe_ratio(returns: pd.Series) -> float:
    """Calculate Sharpe ratio with safety checks"""
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

# -------------------- Evaluation --------------------

def evaluate(df: pd.DataFrame, cond_name: str, params: dict) -> dict:
    """Evaluate strategy with error handling"""
    if df.empty:
        return {
            'condition': cond_name,
            'params': params,
            'sharpe': 0.0,
            'trades': 0,
            'total_return': 0.0
        }
    
    close = df['Close']
    open_next = df['Open'].shift(-1)
    daily_returns = (open_next - close) / close
    
    signal = pd.Series(False, index=df.index)
    
    try:
        if cond_name == 'ENH_VOL_ADJ_V1':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            vol_w = params['vol_weight']
            k = params['k']
            indicator = enhanced_vol_adj_trend_v1(df, trend_p, vol_p, vol_w)
            signal = indicator > k
        
        elif cond_name == 'ENH_VOL_ADJ_V2':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = enhanced_vol_adj_trend_v2(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'ENH_VOL_ADJ_V3':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            lb_mult = params['lookback_mult']
            k = params['k']
            indicator = enhanced_vol_adj_trend_v3(df, trend_p, vol_p, lb_mult)
            signal = indicator > k
        
        elif cond_name == 'ENH_VOL_ADJ_V4':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = enhanced_vol_adj_trend_v4(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'ENH_VOL_ADJ_V5':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = enhanced_vol_adj_trend_v5(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'ENH_VOL_ADJ_V6':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            smooth_p = params['smooth_period']
            k = params['k']
            indicator = enhanced_vol_adj_trend_v6(df, trend_p, vol_p, smooth_p)
            signal = indicator > k
        
        else:
            raise ValueError(f"Unknown condition: {cond_name}")
    
    except Exception as e:
        print(f"Error in {cond_name}: {e}")
        signal = pd.Series(False, index=df.index)
    
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
        'condition': cond_name,
        'params': params,
        'sharpe': sharpe,
        'trades': n_trades,
        'total_return': total_return
    }

# -------------------- Grid Search --------------------

def generate_conditions() -> list:
    """Generate test conditions focused on best parameters"""
    conditions = []
    
    # Focus on parameters near the best performing ones (150 trend, 40 vol)
    trend_periods = [125, 140, 150, 160, 175]
    vol_periods = [30, 35, 40, 45, 50]
    thresholds = [-0.25, -0.1, 0.0, 0.1, 0.25]
    
    # ENH_VOL_ADJ_V1 - With adjustable volatility weight
    for trend_p in trend_periods:
        for vol_p in vol_periods:
            for vol_w in [0.8, 1.0, 1.2]:
                for k in thresholds:
                    conditions.append(('ENH_VOL_ADJ_V1', {
                        'trend_period': trend_p,
                        'vol_period': vol_p,
                        'vol_weight': vol_w,
                        'k': k
                    }))
    
    # ENH_VOL_ADJ_V2 - Exponential adjustment
    for trend_p in trend_periods:
        for vol_p in vol_periods:
            for k in thresholds:
                conditions.append(('ENH_VOL_ADJ_V2', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # ENH_VOL_ADJ_V3 - Dynamic lookback
    for trend_p in [140, 150, 160]:
        for vol_p in [35, 40, 45]:
            for lb_mult in [1.5, 2.0, 2.5]:
                for k in [-0.1, 0.0, 0.1]:
                    conditions.append(('ENH_VOL_ADJ_V3', {
                        'trend_period': trend_p,
                        'vol_period': vol_p,
                        'lookback_mult': lb_mult,
                        'k': k
                    }))
    
    # ENH_VOL_ADJ_V4 - Market regime
    for trend_p in [140, 150, 160]:
        for vol_p in [35, 40, 45]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('ENH_VOL_ADJ_V4', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # ENH_VOL_ADJ_V5 - Volume confirmation
    for trend_p in [140, 150, 160]:
        for vol_p in [35, 40, 45]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('ENH_VOL_ADJ_V5', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # ENH_VOL_ADJ_V6 - Smoothed
    for trend_p in [140, 150, 160]:
        for vol_p in [35, 40, 45]:
            for smooth_p in [3, 5, 7]:
                for k in [-0.1, 0.0, 0.1]:
                    conditions.append(('ENH_VOL_ADJ_V6', {
                        'trend_period': trend_p,
                        'vol_period': vol_p,
                        'smooth_period': smooth_p,
                        'k': k
                    }))
    
    return conditions[:MAX_COMBOS]

# -------------------- Main --------------------

def main():
    """Main execution function"""
    print("Wave21_v5: Enhanced VOL_ADJ_TREND Variations")
    print("=" * 50)
    
    # Load data
    prices = load_data(TICKER, START_DATE, END_DATE)
    
    if prices.empty:
        print("Error: Unable to load data")
        return
    
    print(f"Data loaded: {len(prices)} rows")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Generate conditions
    conditions = generate_conditions()
    print(f"Total conditions to test: {len(conditions)}")
    
    # Run evaluations
    results = []
    for i, (name, params) in enumerate(conditions):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(conditions)} ({i/len(conditions)*100:.1f}%)")
        
        try:
            result = evaluate(prices, name, params)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {name} with {params}: {e}")
    
    # Sort and display results
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\n" + "=" * 50)
    print("Top 20 Results - Wave21_v5:")
    print("=" * 50)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave21_v5 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave21_v5 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave21_v5 appended to history.log")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print(f"Best Sharpe Ratio: {res_sorted.iloc[0]['sharpe']:.3f}")
    print(f"Best Strategy: {res_sorted.iloc[0]['condition']}")
    print(f"Best Parameters: {res_sorted.iloc[0]['params']}")
    print(f"Strategies with Sharpe > 0.9: {len(res_sorted[res_sorted['sharpe'] > 0.9])}")
    print(f"Strategies with Sharpe > 1.0: {len(res_sorted[res_sorted['sharpe'] > 1.0])}")
    print(f"Strategies with >3000 trades: {len(res_sorted[res_sorted['trades'] > 3000])}")

if __name__ == "__main__":
    main()