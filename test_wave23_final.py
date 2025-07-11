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

# -------------------- Data Loading --------------------

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

# -------------------- Price Acceleration Indicators --------------------

def price_acceleration_basic(df: pd.DataFrame, period: int) -> pd.Series:
    """Basic price acceleration - second derivative"""
    close = df['Close']
    
    # First derivative (velocity/momentum)
    velocity = close.pct_change(period)
    
    # Second derivative (acceleration)
    acceleration = velocity.diff(period)
    
    # Normalize by volatility
    vol = velocity.rolling(period).std()
    normalized_accel = acceleration / vol.where(vol > 0, 1)
    
    return normalized_accel

def price_acceleration_smooth(df: pd.DataFrame, period: int, smooth: int = 5) -> pd.Series:
    """Smoothed price acceleration"""
    close = df['Close']
    
    # Smooth prices first
    smooth_close = close.rolling(smooth).mean()
    
    # Calculate acceleration on smoothed prices
    velocity = smooth_close.pct_change(period)
    acceleration = velocity.diff(period)
    
    # Normalize
    vol = close.pct_change().rolling(period).std()
    normalized = acceleration / vol.where(vol > 0, 1)
    
    return normalized

def price_jerk(df: pd.DataFrame, period: int) -> pd.Series:
    """Third derivative - rate of change of acceleration"""
    close = df['Close']
    
    # First derivative
    velocity = close.pct_change(period)
    
    # Second derivative
    acceleration = velocity.diff(period)
    
    # Third derivative
    jerk = acceleration.diff(period // 2)
    
    # Normalize
    vol = velocity.rolling(period).std()
    normalized_jerk = jerk / vol.where(vol > 0, 1)
    
    return normalized_jerk

# -------------------- Enhanced VOL_ADJ_TREND Variations --------------------

def enhanced_vol_adj_trend_v1(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Enhanced VOL_ADJ_TREND with acceleration component"""
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
    
    # Volatility adjustment
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Add acceleration component
    trend_accel = slopes.diff(vol_period // 2)
    accel_factor = (1 + trend_accel * 10).clip(0.5, 1.5)
    
    # Combined adjustment
    adjusted_trend = slopes * (1 - vol_percentile) * accel_factor
    
    return adjusted_trend

def enhanced_vol_adj_trend_v2(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
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
    vol_ratio = volume / vol_ma.where(vol_ma > 0, 1)
    vol_factor = (0.5 + 0.5 * vol_ratio).clip(0.5, 1.5)
    
    # Combined adjustment
    adjusted_trend = slopes * (1 - vol_percentile) * vol_factor
    
    return adjusted_trend

def enhanced_vol_adj_trend_v3(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """VOL_ADJ_TREND with regime adaptation"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend
    slopes = pd.Series(index=close.index, dtype=float)
    r_squared = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
                
                # Calculate R-squared for trend quality
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
    
    # Volatility adjustment with regime
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Regime-based adjustment
    low_vol_regime = vol_percentile < 0.3
    high_vol_regime = vol_percentile > 0.7
    
    # Different adjustments for different regimes
    vol_factor = pd.Series(1 - vol_percentile, index=close.index)
    vol_factor[low_vol_regime] = (1 - vol_percentile[low_vol_regime]) * 1.2
    vol_factor[high_vol_regime] = (1 - vol_percentile[high_vol_regime]) * 0.8
    
    # Quality filter
    quality_factor = (0.5 + 0.5 * r_squared).clip(0.5, 1.0)
    
    # Combined adjustment
    adjusted_trend = slopes * vol_factor * quality_factor
    
    return adjusted_trend

# -------------------- Advanced Momentum Indicators --------------------

def momentum_quality_enhanced(df: pd.DataFrame, period: int) -> pd.Series:
    """Enhanced momentum quality score"""
    close = df['Close']
    returns = close.pct_change()
    
    # Basic momentum
    momentum = close.pct_change(period)
    
    # Quality metrics
    win_rate = (returns > 0).rolling(period).mean()
    avg_win = returns.where(returns > 0, 0).rolling(period).mean()
    avg_loss = returns.where(returns < 0, 0).rolling(period).mean().abs()
    
    # Consistency score
    rolling_std = returns.rolling(period).std()
    consistency = 1 / (1 + rolling_std * np.sqrt(252))
    
    # Combined quality score
    profit_factor = avg_win / avg_loss.where(avg_loss > 0, 1)
    quality = momentum * win_rate * profit_factor * consistency
    
    return quality

def adaptive_momentum(df: pd.DataFrame, base_period: int) -> pd.Series:
    """Momentum with adaptive lookback"""
    close = df['Close']
    volume = df['Volume']
    
    # Market activity
    vol_ratio = volume / volume.rolling(base_period).mean()
    price_range = (df['High'] - df['Low']) / close
    activity = (vol_ratio * price_range).rolling(10).mean()
    
    # Adaptive period
    activity_rank = activity.rolling(base_period * 2).rank(pct=True)
    adaptive_period = (base_period * (2 - activity_rank)).clip(10, 200)
    
    # Calculate adaptive momentum
    momentum = pd.Series(index=close.index, dtype=float)
    
    for i in range(200, len(close)):
        period = int(adaptive_period.iloc[i]) if not pd.isna(adaptive_period.iloc[i]) else base_period
        if i >= period and period > 0:
            momentum.iloc[i] = (close.iloc[i] / close.iloc[i-period] - 1) * 100
    
    return momentum

# -------------------- Volatility Regime Indicators --------------------

def volatility_regime_advanced(df: pd.DataFrame, period: int) -> pd.Series:
    """Advanced volatility regime detection"""
    returns = df['Close'].pct_change()
    
    # Multiple volatility measures
    std_vol = returns.rolling(period).std() * np.sqrt(252)
    
    # Parkinson volatility
    high_low_ratio = np.log(df['High'] / df['Low'])
    park_vol = (high_low_ratio / (2 * np.sqrt(np.log(2)))).rolling(period).mean() * np.sqrt(252)
    
    # Combined volatility
    combined_vol = (std_vol + park_vol) / 2
    
    # Regime thresholds
    vol_20 = combined_vol.rolling(252).quantile(0.2)
    vol_50 = combined_vol.rolling(252).quantile(0.5)
    vol_80 = combined_vol.rolling(252).quantile(0.8)
    
    # Regime scores
    regime_score = pd.Series(index=df.index, dtype=float)
    
    # Low vol regime (bullish)
    regime_score[combined_vol < vol_20] = 1.5
    regime_score[(combined_vol >= vol_20) & (combined_vol < vol_50)] = 1.0
    regime_score[(combined_vol >= vol_50) & (combined_vol < vol_80)] = 0.0
    regime_score[combined_vol >= vol_80] = -1.0
    
    # Smooth transitions
    return regime_score.rolling(5).mean()

# -------------------- Evaluation Functions --------------------

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

def evaluate(df: pd.DataFrame, cond_name: str, params: dict) -> dict:
    """Evaluate strategy"""
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
        if cond_name == 'PRICE_ACCEL_BASIC':
            period = params['period']
            k = params['k']
            indicator = price_acceleration_basic(df, period)
            signal = indicator > k
        
        elif cond_name == 'PRICE_ACCEL_SMOOTH':
            period = params['period']
            smooth = params['smooth']
            k = params['k']
            indicator = price_acceleration_smooth(df, period, smooth)
            signal = indicator > k
        
        elif cond_name == 'PRICE_JERK':
            period = params['period']
            k = params['k']
            indicator = price_jerk(df, period)
            signal = indicator > k
        
        elif cond_name == 'ENH_VOL_ADJ_V1':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = enhanced_vol_adj_trend_v1(df, trend_p, vol_p)
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
            k = params['k']
            indicator = enhanced_vol_adj_trend_v3(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'MOM_QUALITY_ENH':
            period = params['period']
            k = params['k']
            indicator = momentum_quality_enhanced(df, period)
            signal = indicator > k
        
        elif cond_name == 'ADAPTIVE_MOM':
            base_period = params['base_period']
            k = params['k']
            indicator = adaptive_momentum(df, base_period)
            signal = indicator > k
        
        elif cond_name == 'VOL_REGIME_ADV':
            period = params['period']
            k = params['k']
            indicator = volatility_regime_advanced(df, period)
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

# -------------------- Main Execution --------------------

def main():
    """Main execution function"""
    print("Wave23: Price Acceleration and Enhanced VOL_ADJ_TREND")
    print("=" * 70)
    
    # Load data
    prices = load_data(TICKER, START_DATE, END_DATE)
    
    if prices.empty:
        print("Error: Unable to load data")
        return
    
    print(f"Data loaded: {len(prices)} rows")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Generate test conditions
    conditions = []
    
    # PRICE_ACCEL_BASIC - Basic price acceleration
    for period in [10, 20, 30, 50]:
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            conditions.append(('PRICE_ACCEL_BASIC', {
                'period': period,
                'k': k
            }))
    
    # PRICE_ACCEL_SMOOTH - Smoothed price acceleration
    for period in [20, 30, 50]:
        for smooth in [3, 5, 7]:
            for k in [-0.5, 0.0, 0.5]:
                conditions.append(('PRICE_ACCEL_SMOOTH', {
                    'period': period,
                    'smooth': smooth,
                    'k': k
                }))
    
    # PRICE_JERK - Third derivative
    for period in [20, 40, 60]:
        for k in [-1.0, 0.0, 1.0]:
            conditions.append(('PRICE_JERK', {
                'period': period,
                'k': k
            }))
    
    # ENH_VOL_ADJ_V1 - Enhanced with acceleration
    for trend_p in [120, 130, 140, 150]:
        for vol_p in [40, 45, 50]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('ENH_VOL_ADJ_V1', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # ENH_VOL_ADJ_V2 - Enhanced with volume
    for trend_p in [125, 130, 135]:
        for vol_p in [42, 45, 48]:
            for k in [-0.05, 0.0, 0.05]:
                conditions.append(('ENH_VOL_ADJ_V2', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # ENH_VOL_ADJ_V3 - Enhanced with regime
    for trend_p in [128, 130, 132]:
        for vol_p in [44, 45, 46]:
            for k in [-0.02, 0.0, 0.02]:
                conditions.append(('ENH_VOL_ADJ_V3', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # MOM_QUALITY_ENH - Enhanced momentum quality
    for period in [20, 50, 100]:
        for k in [-0.01, 0.0, 0.01, 0.02]:
            conditions.append(('MOM_QUALITY_ENH', {
                'period': period,
                'k': k
            }))
    
    # ADAPTIVE_MOM - Adaptive momentum
    for base_period in [30, 50, 100]:
        for k in [0.0, 1.0, 2.0]:
            conditions.append(('ADAPTIVE_MOM', {
                'base_period': base_period,
                'k': k
            }))
    
    # VOL_REGIME_ADV - Advanced volatility regime
    for period in [20, 30, 50]:
        for k in [-0.5, 0.0, 0.5, 1.0]:
            conditions.append(('VOL_REGIME_ADV', {
                'period': period,
                'k': k
            }))
    
    print(f"Total conditions to test: {len(conditions)}")
    
    # Run evaluations
    results = []
    for i, (name, params) in enumerate(conditions):
        if i % 25 == 0:
            print(f"Progress: {i}/{len(conditions)} ({i/len(conditions)*100:.1f}%)")
        
        try:
            result = evaluate(prices, name, params)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {name} with {params}: {e}")
    
    # Sort and display results
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\n" + "=" * 70)
    print("Top 20 Results - Wave23:")
    print("=" * 70)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave23 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave23 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave23 appended to history.log")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print(f"Best Sharpe Ratio: {res_sorted.iloc[0]['sharpe']:.3f}")
    print(f"Best Strategy: {res_sorted.iloc[0]['condition']}")
    print(f"Best Parameters: {res_sorted.iloc[0]['params']}")
    print(f"Strategies with Sharpe > 0.9: {len(res_sorted[res_sorted['sharpe'] > 0.9])}")
    print(f"Strategies with Sharpe > 1.0: {len(res_sorted[res_sorted['sharpe'] > 1.0])}")
    print(f"Strategies with Sharpe > 1.1: {len(res_sorted[res_sorted['sharpe'] > 1.1])}")
    print(f"Strategies with >3000 trades: {len(res_sorted[res_sorted['trades'] > 3000])}")

if __name__ == "__main__":
    main()