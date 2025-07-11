import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os
from typing import Optional, Tuple, Union

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

# -------------------- Helper Functions --------------------

def safe_division(numerator: pd.Series, denominator: pd.Series, fill_value: float = 1.0) -> pd.Series:
    """Safe division with zero handling"""
    return numerator / denominator.where(denominator != 0, fill_value)

def robust_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate robust z-score using median and MAD"""
    median = series.rolling(window).median()
    abs_dev = (series - median).abs()
    mad = abs_dev.rolling(window).median()
    mad_scaled = mad * 1.4826  # 1.4826 makes MAD comparable to std
    return safe_division(series - median, mad_scaled, 0)

def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Calculate percentile rank within rolling window"""
    return series.rolling(window).rank(pct=True)

# -------------------- Price Action + Volume Patterns --------------------

def volume_weighted_candle_pattern(df: pd.DataFrame, period: int) -> pd.Series:
    """Price action patterns weighted by relative volume"""
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    # Candle metrics
    body = close - open_price
    body_size = body.abs()
    range_size = high - low
    
    # Normalized body ratio
    body_ratio = safe_division(body_size, range_size, 0.5)
    
    # Volume weighting
    vol_ma = volume.rolling(period).mean()
    vol_weight = safe_division(volume, vol_ma, 1.0).clip(0, 3)
    
    # Bullish/bearish strength
    bullish_mask = (body > 0).astype(float)
    bearish_mask = (body < 0).astype(float)
    bullish_strength = bullish_mask * body_ratio * vol_weight
    bearish_strength = bearish_mask * body_ratio * vol_weight
    
    # Net pattern strength
    pattern_strength = (bullish_strength - bearish_strength).rolling(5).mean()
    
    return pattern_strength

def volume_price_efficiency(df: pd.DataFrame, period: int) -> pd.Series:
    """Measure price movement efficiency relative to volume"""
    close = df['Close']
    volume = df['Volume']
    
    # Price change
    price_change = close.pct_change(period).abs()
    
    # Volume used
    total_volume = volume.rolling(period).sum()
    avg_volume = volume.rolling(period * 2).mean()
    relative_volume = safe_division(total_volume, avg_volume * period, 1.0)
    
    # Efficiency: large price move with small volume = efficient
    efficiency = safe_division(price_change, relative_volume, 0)
    
    # Normalize
    efficiency_zscore = robust_zscore(efficiency, period * 2)
    
    return efficiency_zscore

def volume_breakout_strength(df: pd.DataFrame, period: int) -> pd.Series:
    """Breakout strength combining price and volume patterns"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    # Price breakout
    high_roll = high.rolling(period).max()
    low_roll = low.rolling(period).min()
    
    # Breakout detection
    upper_break = close > high_roll.shift(1)
    lower_break = close < low_roll.shift(1)
    
    # Volume confirmation
    vol_percentile = percentile_rank(volume, period * 2)
    
    # Breakout strength
    upper_mask = upper_break.astype(float)
    lower_mask = lower_break.astype(float)
    upper_strength = upper_mask * (close / high_roll - 1) * vol_percentile
    lower_strength = lower_mask * (1 - close / low_roll) * vol_percentile
    
    # Combined signal
    breakout_signal = upper_strength - lower_strength
    
    return breakout_signal

# -------------------- Alternative Normalization Techniques --------------------

def tanh_normalization(series: pd.Series, scale: float = 1.0) -> pd.Series:
    """Hyperbolic tangent normalization"""
    return np.tanh(series * scale)

def rank_normalization(series: pd.Series, window: int) -> pd.Series:
    """Normalize using rank within window"""
    rank = series.rolling(window).rank()
    count = series.rolling(window).count()
    return (rank - 1) / (count - 1).where(count > 1, 1)

def adaptive_zscore(series: pd.Series, base_window: int) -> pd.Series:
    """Z-score with adaptive window based on volatility"""
    returns = series.pct_change()
    vol = returns.rolling(base_window).std()
    vol_percentile = percentile_rank(vol, base_window * 4)
    
    # Adaptive window: shorter in high vol, longer in low vol
    window_factor = 0.5 + vol_percentile
    adaptive_window = (base_window * window_factor).clip(10, 200)
    
    # Calculate adaptive z-score
    zscore_values = pd.Series(index=series.index, dtype=float)
    
    for i in range(200, len(series)):
        window = int(adaptive_window.iloc[i]) if not pd.isna(adaptive_window.iloc[i]) else base_window
        if i >= window:
            subset = series.iloc[i-window:i+1]
            mean = subset.mean()
            std = subset.std()
            if std > 0:
                zscore_values.iloc[i] = (series.iloc[i] - mean) / std
    
    return zscore_values

def min_max_adaptive(series: pd.Series, window: int) -> pd.Series:
    """Min-max normalization with outlier handling"""
    # Calculate rolling min/max with percentiles to handle outliers
    rolling_5 = series.rolling(window).quantile(0.05)
    rolling_95 = series.rolling(window).quantile(0.95)
    
    # Normalize
    normalized = safe_division(series - rolling_5, rolling_95 - rolling_5, 0.5)
    
    # Clip to handle remaining outliers
    return normalized.clip(0, 1)

# -------------------- Alternative Volatility Measures --------------------

def garman_klass_volatility(df: pd.DataFrame, period: int) -> pd.Series:
    """Garman-Klass volatility estimator using OHLC"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    open_price = df['Open']
    
    # GK components
    hl_ratio = np.log(high / low) ** 2
    co_ratio = np.log(close / open_price) ** 2
    
    # GK volatility
    gk_vol = np.sqrt(0.5 * hl_ratio - (2 * np.log(2) - 1) * co_ratio)
    
    # Annualized rolling average
    return gk_vol.rolling(period).mean() * np.sqrt(252)

def parkinson_volatility(df: pd.DataFrame, period: int) -> pd.Series:
    """Parkinson volatility using high-low range"""
    high = df['High']
    low = df['Low']
    
    # Parkinson estimator
    hl_ratio = np.log(high / low)
    park_vol = hl_ratio / (2 * np.sqrt(np.log(2)))
    
    # Annualized rolling average
    return park_vol.rolling(period).mean() * np.sqrt(252)

def volatility_of_volatility(df: pd.DataFrame, period: int) -> pd.Series:
    """Volatility of volatility (vol of vol)"""
    returns = df['Close'].pct_change()
    
    # Rolling volatility
    vol = returns.rolling(period).std()
    
    # Volatility of volatility
    vol_of_vol = vol.rolling(period).std()
    
    # Normalize by mean volatility
    mean_vol = vol.rolling(period * 2).mean()
    
    return safe_division(vol_of_vol, mean_vol, 1.0)

# -------------------- Combined Indicators --------------------

def price_volume_divergence_normalized(df: pd.DataFrame, period: int, norm_type: str = 'zscore') -> pd.Series:
    """Price-volume divergence with various normalizations"""
    close = df['Close']
    volume = df['Volume']
    
    # Price momentum
    price_roc = close.pct_change(period)
    
    # Volume momentum
    vol_roc = volume.pct_change(period)
    
    # Apply normalization
    if norm_type == 'zscore':
        price_norm = robust_zscore(price_roc, period * 2)
        vol_norm = robust_zscore(vol_roc, period * 2)
    elif norm_type == 'rank':
        price_norm = rank_normalization(price_roc, period * 2)
        vol_norm = rank_normalization(vol_roc, period * 2)
    elif norm_type == 'tanh':
        price_norm = tanh_normalization(price_roc, 10)
        vol_norm = tanh_normalization(vol_roc, 10)
    else:  # minmax
        price_norm = min_max_adaptive(price_roc, period * 2)
        vol_norm = min_max_adaptive(vol_roc, period * 2)
    
    # Divergence
    divergence = vol_norm - price_norm
    
    return divergence

def volatility_adjusted_momentum(df: pd.DataFrame, mom_period: int, vol_type: str = 'standard') -> pd.Series:
    """Momentum adjusted by various volatility measures"""
    close = df['Close']
    
    # Momentum
    momentum = close.pct_change(mom_period)
    
    # Get volatility based on type
    if vol_type == 'standard':
        vol = close.pct_change().rolling(mom_period).std()
    elif vol_type == 'garman_klass':
        vol = garman_klass_volatility(df, mom_period)
    elif vol_type == 'parkinson':
        vol = parkinson_volatility(df, mom_period)
    else:  # vol_of_vol
        vol = volatility_of_volatility(df, mom_period)
    
    # Normalize volatility
    vol_percentile = percentile_rank(vol, mom_period * 2)
    
    # Adjust momentum
    adjusted_momentum = momentum * (1 - vol_percentile)
    
    return adjusted_momentum

def composite_price_action_score(df: pd.DataFrame, period: int) -> pd.Series:
    """Composite score combining multiple price action elements"""
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    # Components
    # 1. Candle strength
    body = close - open_price
    range_size = high - low
    candle_strength = safe_division(body, range_size, 0)
    
    # 2. Close position in range
    close_position = safe_division(close - low, range_size, 0.5)
    
    # 3. Volume confirmation
    vol_percentile = percentile_rank(volume, period)
    
    # 4. Trend alignment
    trend = close.rolling(period).mean()
    trend_position = (close > trend).astype(float)
    
    # Composite score
    composite = (
        candle_strength * 0.3 +
        (close_position - 0.5) * 0.2 +
        (vol_percentile - 0.5) * 0.3 +
        (trend_position - 0.5) * 0.2
    )
    
    # Smooth and normalize
    smoothed = composite.rolling(5).mean()
    return robust_zscore(smoothed, period)

# -------------------- Sharpe Ratio --------------------

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
        if cond_name == 'VOL_WGT_CANDLE':
            period = params['period']
            k = params['k']
            indicator = volume_weighted_candle_pattern(df, period)
            signal = indicator > k
        
        elif cond_name == 'VOL_PRICE_EFF':
            period = params['period']
            k = params['k']
            indicator = volume_price_efficiency(df, period)
            signal = indicator > k
        
        elif cond_name == 'VOL_BREAKOUT':
            period = params['period']
            k = params['k']
            indicator = volume_breakout_strength(df, period)
            signal = indicator > k
        
        elif cond_name == 'PRICE_VOL_DIV_NORM':
            period = params['period']
            norm_type = params['norm_type']
            k = params['k']
            indicator = price_volume_divergence_normalized(df, period, norm_type)
            signal = indicator > k
        
        elif cond_name == 'VOL_ADJ_MOM_ALT':
            mom_period = params['mom_period']
            vol_type = params['vol_type']
            k = params['k']
            indicator = volatility_adjusted_momentum(df, mom_period, vol_type)
            signal = indicator > k
        
        elif cond_name == 'COMPOSITE_PA':
            period = params['period']
            k = params['k']
            indicator = composite_price_action_score(df, period)
            signal = indicator > k
        
        elif cond_name == 'ADAPTIVE_Z':
            base_window = params['base_window']
            k = params['k']
            indicator = adaptive_zscore(df['Close'].pct_change(20), base_window)
            signal = indicator > k
        
        elif cond_name == 'TANH_MOM':
            period = params['period']
            scale = params['scale']
            k = params['k']
            momentum = df['Close'].pct_change(period)
            indicator = tanh_normalization(momentum, scale)
            signal = indicator > k
        
        elif cond_name == 'RANK_TREND':
            period = params['period']
            k = params['k']
            trend = df['Close'].rolling(period).mean()
            price_vs_trend = df['Close'] / trend - 1
            indicator = rank_normalization(price_vs_trend, period * 2)
            signal = indicator > k
        
        elif cond_name == 'MINMAX_VOL':
            period = params['period']
            k = params['k']
            returns = df['Close'].pct_change()
            vol = returns.rolling(period).std()
            indicator = min_max_adaptive(vol, period * 2)
            signal = indicator < k  # Low normalized vol
        
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
    """Generate test conditions"""
    conditions = []
    
    # VOL_WGT_CANDLE - Volume weighted candle patterns
    for period in [10, 20, 50]:
        for k in [-0.1, 0.0, 0.1, 0.2]:
            conditions.append(('VOL_WGT_CANDLE', {
                'period': period,
                'k': k
            }))
    
    # VOL_PRICE_EFF - Volume price efficiency
    for period in [20, 50, 100]:
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            conditions.append(('VOL_PRICE_EFF', {
                'period': period,
                'k': k
            }))
    
    # VOL_BREAKOUT - Volume breakout strength
    for period in [20, 50, 100]:
        for k in [-0.05, 0.0, 0.05, 0.1]:
            conditions.append(('VOL_BREAKOUT', {
                'period': period,
                'k': k
            }))
    
    # PRICE_VOL_DIV_NORM - Price volume divergence with different normalizations
    for period in [20, 50]:
        for norm_type in ['zscore', 'rank', 'tanh', 'minmax']:
            for k in [-1.0, -0.5, 0.0, 0.5]:
                conditions.append(('PRICE_VOL_DIV_NORM', {
                    'period': period,
                    'norm_type': norm_type,
                    'k': k
                }))
    
    # VOL_ADJ_MOM_ALT - Volatility adjusted momentum with alternative vol measures
    for mom_period in [20, 50, 100]:
        for vol_type in ['standard', 'garman_klass', 'parkinson', 'vol_of_vol']:
            for k in [-0.01, 0.0, 0.01]:
                conditions.append(('VOL_ADJ_MOM_ALT', {
                    'mom_period': mom_period,
                    'vol_type': vol_type,
                    'k': k
                }))
    
    # COMPOSITE_PA - Composite price action score
    for period in [20, 50, 100]:
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            conditions.append(('COMPOSITE_PA', {
                'period': period,
                'k': k
            }))
    
    # ADAPTIVE_Z - Adaptive z-score
    for base_window in [20, 50, 100]:
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            conditions.append(('ADAPTIVE_Z', {
                'base_window': base_window,
                'k': k
            }))
    
    # TANH_MOM - Tanh normalized momentum
    for period in [20, 50, 100]:
        for scale in [5, 10, 20]:
            for k in [-0.5, 0.0, 0.5]:
                conditions.append(('TANH_MOM', {
                    'period': period,
                    'scale': scale,
                    'k': k
                }))
    
    # RANK_TREND - Rank normalized trend
    for period in [50, 100, 150]:
        for k in [0.3, 0.5, 0.7]:
            conditions.append(('RANK_TREND', {
                'period': period,
                'k': k
            }))
    
    # MINMAX_VOL - Min-max normalized volatility
    for period in [20, 50]:
        for k in [0.2, 0.3, 0.5]:
            conditions.append(('MINMAX_VOL', {
                'period': period,
                'k': k
            }))
    
    return conditions

# -------------------- Main --------------------

def main():
    """Main execution function"""
    print("Wave22: Price Action + Volume, Alternative Normalizations, New Volatility Measures")
    print("=" * 80)
    
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
    
    print("\n" + "=" * 80)
    print("Top 20 Results - Wave22:")
    print("=" * 80)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave22 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave22 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave22 appended to history.log")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print(f"Best Sharpe Ratio: {res_sorted.iloc[0]['sharpe']:.3f}")
    print(f"Best Strategy: {res_sorted.iloc[0]['condition']}")
    print(f"Best Parameters: {res_sorted.iloc[0]['params']}")
    print(f"Strategies with Sharpe > 0.9: {len(res_sorted[res_sorted['sharpe'] > 0.9])}")
    print(f"Strategies with Sharpe > 1.0: {len(res_sorted[res_sorted['sharpe'] > 1.0])}")
    print(f"Strategies with >3000 trades: {len(res_sorted[res_sorted['trades'] > 3000])}")

if __name__ == "__main__":
    main()