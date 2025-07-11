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

# -------------------- Non-Linear Combinations --------------------

def nonlinear_vol_adj_trend(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Non-linear combination of trend and volatility"""
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
    
    # Volatility metrics
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Non-linear transformations
    # 1. Sigmoid transformation of trend
    trend_sigmoid = 2 / (1 + np.exp(-slopes * 0.5)) - 1
    
    # 2. Power transformation of volatility adjustment
    vol_power = (1 - vol_percentile) ** 1.5
    
    # 3. Interaction term
    interaction = trend_sigmoid * vol_power * np.sqrt(np.abs(trend_sigmoid))
    
    return interaction

def polynomial_trend_vol(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Polynomial combination of trend and volatility signals"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend
    slopes = pd.Series(index=close.index, dtype=float)
    curvature = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                # Fit polynomial (degree 2)
                coeffs = np.polyfit(x, y, 2)
                slopes.iloc[i] = coeffs[1] / close.iloc[i] * 100
                curvature.iloc[i] = coeffs[0] / close.iloc[i] * 10000
            except:
                slopes.iloc[i] = 0
                curvature.iloc[i] = 0
    
    # Volatility
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Polynomial combination
    signal = slopes * (1 - vol_percentile) + curvature * (1 - vol_percentile) ** 2
    
    return signal

def multiplicative_composite(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Multiplicative composite of multiple signals"""
    close = df['Close']
    volume = df['Volume']
    returns = close.pct_change()
    
    # Trend component
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
    
    # Normalize trend
    trend_norm = (slopes - slopes.rolling(trend_period).mean()) / slopes.rolling(trend_period).std()
    trend_factor = 1 / (1 + np.exp(-trend_norm))  # Sigmoid
    
    # Volatility component
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    vol_factor = np.exp(-(vol_percentile - 0.5) ** 2)  # Gaussian-like
    
    # Volume component
    vol_ma = volume.rolling(vol_period).mean()
    vol_ratio = volume / vol_ma.where(vol_ma > 0, 1)
    vol_factor_volume = 1 / (1 + np.exp(-(vol_ratio - 1) * 5))  # Sigmoid
    
    # Multiplicative composite
    composite = trend_factor * vol_factor * vol_factor_volume
    
    return composite

# -------------------- Alternative Smoothing Techniques --------------------

def kalman_smoothed_trend(df: pd.DataFrame, period: int) -> pd.Series:
    """Kalman filter-inspired smoothing for trend"""
    close = df['Close']
    
    # Initialize
    estimate = close.copy()
    error_estimate = pd.Series(1.0, index=close.index)
    
    # Parameters
    process_variance = 0.01
    measurement_variance = 0.1
    
    # Kalman-like filtering
    for i in range(1, len(close)):
        # Prediction
        prediction = estimate.iloc[i-1]
        prediction_error = error_estimate.iloc[i-1] + process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + measurement_variance)
        estimate.iloc[i] = prediction + kalman_gain * (close.iloc[i] - prediction)
        error_estimate.iloc[i] = (1 - kalman_gain) * prediction_error
    
    # Calculate trend on smoothed prices
    slopes = pd.Series(index=close.index, dtype=float)
    for i in range(period, len(close)):
        y = estimate.iloc[i-period:i].values
        x = np.arange(period)
        if len(y) == period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
            except:
                slopes.iloc[i] = 0
    
    return slopes

def hull_moving_average_trend(df: pd.DataFrame, period: int) -> pd.Series:
    """Hull Moving Average for smoother trend"""
    close = df['Close']
    
    # Hull MA calculation
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculations
    wma_half = close.rolling(half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
    )
    wma_full = close.rolling(period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
    )
    
    # Hull calculation
    raw_hull = 2 * wma_half - wma_full
    hull_ma = raw_hull.rolling(sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
    )
    
    # Trend on Hull MA
    hull_trend = (hull_ma / hull_ma.shift(period) - 1) * 100
    
    return hull_trend

def adaptive_smooth_trend(df: pd.DataFrame, base_period: int) -> pd.Series:
    """Adaptive smoothing based on market conditions"""
    close = df['Close']
    returns = close.pct_change()
    
    # Market noise measure
    noise = returns.rolling(20).std()
    noise_percentile = noise.rolling(100).rank(pct=True)
    
    # Adaptive smoothing period
    smooth_period = (base_period * (0.5 + noise_percentile)).astype(int).clip(5, 50)
    
    # Apply adaptive smoothing
    smoothed = pd.Series(index=close.index, dtype=float)
    for i in range(50, len(close)):
        period = int(smooth_period.iloc[i]) if not pd.isna(smooth_period.iloc[i]) else base_period
        smoothed.iloc[i] = close.iloc[i-period:i+1].mean()
    
    # Calculate trend
    trend = (smoothed / smoothed.shift(base_period) - 1) * 100
    
    return trend

# -------------------- New Price/Volume Relationships --------------------

def volume_momentum_divergence(df: pd.DataFrame, period: int) -> pd.Series:
    """Advanced volume-price momentum divergence"""
    close = df['Close']
    volume = df['Volume']
    
    # Price momentum
    price_mom = close / close.shift(period) - 1
    
    # Volume momentum (with direction)
    signed_volume = volume * np.sign(close - close.shift(1))
    vol_mom = signed_volume.rolling(period).sum() / volume.rolling(period).sum()
    
    # Divergence with non-linear scaling
    divergence = vol_mom - price_mom
    scaled_divergence = np.sign(divergence) * np.sqrt(np.abs(divergence))
    
    return scaled_divergence

def vwap_acceleration(df: pd.DataFrame, period: int) -> pd.Series:
    """VWAP acceleration indicator"""
    close = df['Close']
    volume = df['Volume']
    
    # Calculate VWAP
    typical_price = (df['High'] + df['Low'] + close) / 3
    vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
    
    # VWAP velocity and acceleration
    vwap_velocity = vwap.pct_change(period // 2)
    vwap_acceleration = vwap_velocity.diff(period // 4)
    
    # Normalize by price
    normalized_accel = vwap_acceleration / (close.rolling(period).std() / close)
    
    return normalized_accel

def volume_concentration_signal(df: pd.DataFrame, period: int) -> pd.Series:
    """Volume concentration and its impact on price"""
    close = df['Close']
    volume = df['Volume']
    
    # Volume concentration (Gini coefficient style)
    def gini_coefficient(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(n) + 1) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n
    
    vol_concentration = volume.rolling(period).apply(gini_coefficient)
    
    # Price efficiency under different concentration levels
    price_efficiency = close.pct_change(period) / (volume.rolling(period).sum() / 1e9)
    
    # Combined signal
    signal = price_efficiency * (1 - vol_concentration)
    
    return signal

# -------------------- Innovative Volatility Adjustments --------------------

def asymmetric_volatility_adjustment(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Asymmetric volatility adjustment for trend"""
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
    
    # Separate upside and downside volatility
    upside_returns = returns.where(returns > 0, 0)
    downside_returns = returns.where(returns < 0, 0)
    
    upside_vol = upside_returns.rolling(vol_period).std()
    downside_vol = downside_returns.abs().rolling(vol_period).std()
    
    # Asymmetric adjustment
    vol_ratio = upside_vol / downside_vol.where(downside_vol > 0, 1)
    vol_adjustment = 1 / (1 + np.exp(-(vol_ratio - 1) * 3))
    
    # Apply to trend
    adjusted_trend = slopes * vol_adjustment
    
    return adjusted_trend

def volatility_surface_signal(df: pd.DataFrame, short_vol: int, long_vol: int) -> pd.Series:
    """Volatility surface-based signal"""
    returns = df['Close'].pct_change()
    
    # Multiple volatility horizons
    vol_horizons = [short_vol, (short_vol + long_vol) // 2, long_vol]
    vols = [returns.rolling(h).std() * np.sqrt(252) for h in vol_horizons]
    
    # Volatility term structure slope
    vol_slope = (vols[0] - vols[2]) / vols[2].where(vols[2] > 0, 1)
    
    # Volatility curvature
    vol_curve = vols[1] - (vols[0] + vols[2]) / 2
    
    # Combined signal
    signal = -vol_slope + 2 * vol_curve  # Negative slope + positive curvature = bullish
    
    return signal

def dynamic_volatility_bands(df: pd.DataFrame, period: int) -> pd.Series:
    """Dynamic volatility bands with non-linear scaling"""
    close = df['Close']
    returns = close.pct_change()
    
    # Base volatility
    vol = returns.rolling(period).std()
    
    # Volatility of volatility
    vol_of_vol = vol.rolling(period).std()
    
    # Dynamic band width
    band_width = vol * (1 + vol_of_vol / vol.rolling(period * 2).mean())
    
    # Price position within bands
    ma = close.rolling(period).mean()
    position = (close - ma) / (band_width * close)
    
    # Non-linear transformation
    signal = -np.sign(position) * (np.abs(position) ** 0.5)
    
    return signal

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
        if cond_name == 'NONLINEAR_VOL_ADJ':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = nonlinear_vol_adj_trend(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'POLYNOMIAL_TREND':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = polynomial_trend_vol(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'MULTIPLICATIVE_COMP':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = multiplicative_composite(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'KALMAN_TREND':
            period = params['period']
            k = params['k']
            indicator = kalman_smoothed_trend(df, period)
            signal = indicator > k
        
        elif cond_name == 'HULL_MA_TREND':
            period = params['period']
            k = params['k']
            indicator = hull_moving_average_trend(df, period)
            signal = indicator > k
        
        elif cond_name == 'ADAPTIVE_SMOOTH':
            base_period = params['base_period']
            k = params['k']
            indicator = adaptive_smooth_trend(df, base_period)
            signal = indicator > k
        
        elif cond_name == 'VOL_MOM_DIV':
            period = params['period']
            k = params['k']
            indicator = volume_momentum_divergence(df, period)
            signal = indicator > k
        
        elif cond_name == 'VWAP_ACCEL':
            period = params['period']
            k = params['k']
            indicator = vwap_acceleration(df, period)
            signal = indicator > k
        
        elif cond_name == 'VOL_CONCENTRATION':
            period = params['period']
            k = params['k']
            indicator = volume_concentration_signal(df, period)
            signal = indicator > k
        
        elif cond_name == 'ASYM_VOL_ADJ':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = asymmetric_volatility_adjustment(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'VOL_SURFACE':
            short_vol = params['short_vol']
            long_vol = params['long_vol']
            k = params['k']
            indicator = volatility_surface_signal(df, short_vol, long_vol)
            signal = indicator > k
        
        elif cond_name == 'DYNAMIC_VOL_BANDS':
            period = params['period']
            k = params['k']
            indicator = dynamic_volatility_bands(df, period)
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
    print("Wave24: Non-Linear Combinations and Innovative Indicators")
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
    
    # NONLINEAR_VOL_ADJ - Non-linear volatility adjustment
    for trend_p in [120, 130, 140]:
        for vol_p in [40, 45, 50]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('NONLINEAR_VOL_ADJ', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # POLYNOMIAL_TREND - Polynomial trend with volatility
    for trend_p in [100, 130, 150]:
        for vol_p in [30, 45, 60]:
            for k in [-0.05, 0.0, 0.05]:
                conditions.append(('POLYNOMIAL_TREND', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # MULTIPLICATIVE_COMP - Multiplicative composite
    for trend_p in [120, 130, 140]:
        for vol_p in [40, 45, 50]:
            for k in [0.3, 0.4, 0.5]:
                conditions.append(('MULTIPLICATIVE_COMP', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # KALMAN_TREND - Kalman smoothed trend
    for period in [50, 100, 150]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('KALMAN_TREND', {
                'period': period,
                'k': k
            }))
    
    # HULL_MA_TREND - Hull MA trend
    for period in [50, 100, 150]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('HULL_MA_TREND', {
                'period': period,
                'k': k
            }))
    
    # ADAPTIVE_SMOOTH - Adaptive smoothing
    for base_period in [20, 50, 100]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('ADAPTIVE_SMOOTH', {
                'base_period': base_period,
                'k': k
            }))
    
    # VOL_MOM_DIV - Volume momentum divergence
    for period in [20, 50, 100]:
        for k in [-0.1, 0.0, 0.1]:
            conditions.append(('VOL_MOM_DIV', {
                'period': period,
                'k': k
            }))
    
    # VWAP_ACCEL - VWAP acceleration
    for period in [50, 100, 150]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('VWAP_ACCEL', {
                'period': period,
                'k': k
            }))
    
    # VOL_CONCENTRATION - Volume concentration
    for period in [20, 50, 100]:
        for k in [-0.01, 0.0, 0.01]:
            conditions.append(('VOL_CONCENTRATION', {
                'period': period,
                'k': k
            }))
    
    # ASYM_VOL_ADJ - Asymmetric volatility adjustment
    for trend_p in [120, 130, 140]:
        for vol_p in [40, 45, 50]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('ASYM_VOL_ADJ', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # VOL_SURFACE - Volatility surface
    for short_vol in [10, 20]:
        for long_vol in [50, 100]:
            for k in [-0.5, 0.0, 0.5]:
                conditions.append(('VOL_SURFACE', {
                    'short_vol': short_vol,
                    'long_vol': long_vol,
                    'k': k
                }))
    
    # DYNAMIC_VOL_BANDS - Dynamic volatility bands
    for period in [20, 50, 100]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('DYNAMIC_VOL_BANDS', {
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
    print("Top 20 Results - Wave24:")
    print("=" * 70)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave24 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave24 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave24 appended to history.log")
    
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