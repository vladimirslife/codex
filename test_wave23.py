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

# -------------------- Indicators --------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window).mean()

def price_action_strength(df, period):
    """Measure strength of price action"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    open_price = df['Open']
    
    # Body size relative to range
    body = (close - open_price).abs()
    range_hl = high - low
    body_ratio = body / range_hl.where(range_hl > 0, 1)
    
    # Direction strength
    bullish = close > open_price
    strength = body_ratio * (2 * bullish - 1)  # +1 for bullish, -1 for bearish
    
    # Rolling average
    avg_strength = strength.rolling(period).mean()
    return avg_strength

def support_resistance_distance(df, period):
    """Distance from nearest support/resistance"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Rolling support and resistance
    resistance = high.rolling(period).max()
    support = low.rolling(period).min()
    
    # Distance to nearest level
    dist_to_resistance = (resistance - close) / close
    dist_to_support = (close - support) / close
    
    # Nearest level (smaller distance)
    nearest_distance = pd.concat([dist_to_resistance.abs(), dist_to_support.abs()], axis=1).min(axis=1)
    
    # Sign based on whether we're closer to support (-) or resistance (+)
    sign = (dist_to_resistance.abs() < dist_to_support.abs()) * 2 - 1
    
    return nearest_distance * sign

def candle_pattern_score(df, period):
    """Score based on candlestick patterns"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    open_price = df['Open']
    
    # Various candle characteristics
    body = close - open_price
    upper_shadow = high - pd.concat([close, open_price], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_price], axis=1).min(axis=1) - low
    range_hl = high - low
    
    # Normalized metrics
    body_norm = body / range_hl.where(range_hl > 0, 1)
    upper_norm = upper_shadow / range_hl.where(range_hl > 0, 1)
    lower_norm = lower_shadow / range_hl.where(range_hl > 0, 1)
    
    # Pattern scores
    # Hammer: small body at top, long lower shadow
    hammer_score = (lower_norm > 0.6) & (body_norm.abs() < 0.3) & (body > 0)
    
    # Doji: very small body
    doji_score = body_norm.abs() < 0.1
    
    # Combined pattern score
    pattern_score = hammer_score.astype(float) * 2 + doji_score.astype(float)
    
    # Rolling average
    avg_pattern = pattern_score.rolling(period).mean()
    return avg_pattern

def price_acceleration_2nd(df, period):
    """Second derivative of price (acceleration of acceleration)"""
    close = df['Close']
    
    # First derivative (velocity)
    velocity = close.pct_change()
    
    # Second derivative (acceleration)
    acceleration = velocity - velocity.shift(1)
    
    # Third derivative (jerk)
    jerk = acceleration - acceleration.shift(1)
    
    # Normalized by volatility
    vol = velocity.rolling(period).std()
    normalized_jerk = jerk / vol.where(vol > 0, 1)
    
    # Rolling average
    avg_jerk = normalized_jerk.rolling(period).mean()
    return avg_jerk

def volume_price_efficiency(df, period):
    """Price movement efficiency relative to volume"""
    close = df['Close']
    volume = df['Volume']
    
    # Price change
    price_change = close.pct_change(period).abs()
    
    # Volume effort (normalized)
    volume_sum = volume.rolling(period).sum()
    volume_avg = volume.rolling(period * 2).mean()
    volume_effort = volume_sum / (volume_avg * period)
    
    # Efficiency: price change per unit of volume effort
    efficiency = price_change / volume_effort.where(volume_effort > 0, 1)
    
    return efficiency

def trend_consistency_score(df, short_period, long_period):
    """Consistency between short and long term trends"""
    close = df['Close']
    
    # Short and long term EMAs
    ema_short = ema(close, short_period)
    ema_long = ema(close, long_period)
    
    # Slopes
    short_slope = (ema_short - ema_short.shift(5)) / 5
    long_slope = (ema_long - ema_long.shift(10)) / 10
    
    # Normalize slopes
    short_norm = short_slope / close * 100
    long_norm = long_slope / close * 100
    
    # Consistency score: both pointing same direction with similar magnitude
    direction_match = (short_norm * long_norm > 0).astype(float)
    magnitude_ratio = pd.concat([short_norm.abs() / long_norm.abs(), 
                                long_norm.abs() / short_norm.abs()], axis=1).min(axis=1)
    
    consistency = direction_match * magnitude_ratio.clip(0, 1)
    return consistency

def price_distribution_skew(df, period):
    """Skewness of price distribution"""
    close = df['Close']
    returns = close.pct_change()
    
    # Rolling skewness
    skew = returns.rolling(period).skew()
    
    # Normalized by rolling std
    std = returns.rolling(period).std()
    normalized_skew = skew / std.where(std > 0, 1)
    
    return normalized_skew

# -------------------- Alternative Price Momentum Measures --------------------

def price_acceleration(df: pd.DataFrame, period: int) -> pd.Series:
    """Second derivative of price - acceleration of price movement"""
    close = df['Close']
    
    # First derivative (velocity)
    velocity = close.pct_change(period)
    
    # Second derivative (acceleration)
    acceleration = velocity.diff(period)
    
    # Normalize by volatility
    vol = velocity.rolling(period).std()
    normalized_accel = acceleration / vol.where(vol > 0, 1)
    
    return normalized_accel

def momentum_quality_score(df: pd.DataFrame, period: int) -> pd.Series:
    """Quality of momentum based on consistency and strength"""
    close = df['Close']
    
    # Calculate daily returns
    returns = close.pct_change()
    
    # Rolling metrics
    momentum = close.pct_change(period)
    win_rate = (returns > 0).rolling(period).mean()
    avg_win = returns.where(returns > 0, 0).rolling(period).mean()
    avg_loss = returns.where(returns < 0, 0).rolling(period).mean().abs()
    
    # Quality score combining multiple factors
    quality = momentum * win_rate * (avg_win / avg_loss.where(avg_loss > 0, 1))
    
    return quality

def adaptive_roc(df: pd.DataFrame, base_period: int) -> pd.Series:
    """Rate of change with adaptive period based on market conditions"""
    close = df['Close']
    volume = df['Volume']
    
    # Market activity score
    vol_ratio = volume / volume.rolling(base_period).mean()
    price_range = (df['High'] - df['Low']) / close
    activity = (vol_ratio * price_range).rolling(10).mean()
    
    # Adaptive period: shorter when active, longer when quiet
    activity_rank = activity.rolling(base_period * 2).rank(pct=True)
    adaptive_period = (base_period * (2 - activity_rank)).clip(10, 200)
    
    # Calculate adaptive ROC
    roc_values = pd.Series(index=close.index, dtype=float)
    
    for i in range(200, len(close)):
        period = int(adaptive_period.iloc[i]) if not pd.isna(adaptive_period.iloc[i]) else base_period
        if i >= period and period > 0:
            roc_values.iloc[i] = (close.iloc[i] / close.iloc[i-period] - 1) * 100
    
    return roc_values

# -------------------- Advanced Volume-Price Relationships --------------------

def volume_price_correlation(df: pd.DataFrame, period: int) -> pd.Series:
    """Rolling correlation between volume and price changes"""
    close = df['Close']
    volume = df['Volume']
    
    # Calculate changes
    price_change = close.pct_change()
    volume_change = volume.pct_change()
    
    # Rolling correlation
    corr = price_change.rolling(period).corr(volume_change)
    
    # Smooth and normalize
    corr_smooth = corr.rolling(5).mean()
    corr_zscore = (corr_smooth - corr_smooth.rolling(period * 2).mean()) / corr_smooth.rolling(period * 2).std()
    
    return corr_zscore

def volume_weighted_trend_strength(df: pd.DataFrame, period: int) -> pd.Series:
    """Trend strength weighted by volume concentration"""
    close = df['Close']
    volume = df['Volume']
    
    # Calculate trend
    trend = pd.Series(index=close.index, dtype=float)
    for i in range(period, len(close)):
        y = close.iloc[i-period:i].values
        x = np.arange(period)
        if len(y) == period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                trend.iloc[i] = slope / close.iloc[i] * 100
            except:
                trend.iloc[i] = 0
    
    # Volume concentration (Herfindahl index style)
    vol_sum = volume.rolling(period).sum()
    vol_squares = (volume / vol_sum.where(vol_sum > 0, 1)) ** 2
    vol_concentration = vol_squares.rolling(period).sum()
    
    # Weight trend by volume concentration
    weighted_trend = trend * vol_concentration
    
    return weighted_trend

def obv_divergence(df: pd.DataFrame, period: int) -> pd.Series:
    """On-Balance Volume divergence from price"""
    close = df['Close']
    volume = df['Volume']
    
    # Calculate OBV
    obv = (volume * np.sign(close.diff())).cumsum()
    
    # Calculate divergence
    price_roc = close.pct_change(period)
    obv_roc = obv.pct_change(period)
    
    # Normalize both
    price_zscore = (price_roc - price_roc.rolling(period * 2).mean()) / price_roc.rolling(period * 2).std()
    obv_zscore = (obv_roc - obv_roc.rolling(period * 2).mean()) / obv_roc.rolling(period * 2).std()
    
    # Divergence signal
    divergence = obv_zscore - price_zscore
    
    return divergence

# -------------------- Volatility Regime Detection --------------------

def volatility_regime_markov(df: pd.DataFrame, period: int) -> pd.Series:
    """Markov regime detection for volatility states"""
    returns = df['Close'].pct_change()
    vol = returns.rolling(period).std() * np.sqrt(252)
    
    # Define regime thresholds
    vol_30 = vol.rolling(252).quantile(0.3)
    vol_70 = vol.rolling(252).quantile(0.7)
    
    # Regime classification
    regime = pd.Series(index=df.index, dtype=float)
    regime[vol < vol_30] = 1  # Low vol
    regime[(vol >= vol_30) & (vol < vol_70)] = 2  # Mid vol
    regime[vol >= vol_70] = 3  # High vol
    
    # Regime transition probability
    regime_change = regime.diff()
    transition_score = pd.Series(index=df.index, dtype=float)
    
    # Score based on regime and transition
    transition_score[regime == 1] = 1.0  # Bullish in low vol
    transition_score[regime == 2] = 0.0  # Neutral in mid vol
    transition_score[regime == 3] = -1.0  # Bearish in high vol
    
    # Boost score on transitions
    transition_score[regime_change == -1] = 1.5  # Transition to lower vol
    transition_score[regime_change == 1] = -1.5  # Transition to higher vol
    
    return transition_score.rolling(5).mean()

def garch_like_volatility(df: pd.DataFrame, period: int) -> pd.Series:
    """Simplified GARCH-like volatility with mean reversion"""
    returns = df['Close'].pct_change()
    
    # Current volatility
    vol = returns.rolling(period).std()
    
    # Long-term average volatility
    long_vol = vol.rolling(period * 4).mean()
    
    # Volatility of volatility
    vol_vol = vol.rolling(period).std()
    
    # Mean reversion score
    vol_zscore = (vol - long_vol) / vol_vol.where(vol_vol > 0, 1)
    
    # GARCH-like adjustment
    garch_score = -vol_zscore * (1 + vol_vol / long_vol.where(long_vol > 0, 1))
    
    return garch_score

def volatility_term_structure(df: pd.DataFrame, short_period: int, long_period: int) -> pd.Series:
    """Volatility term structure signal"""
    returns = df['Close'].pct_change()
    
    # Different period volatilities
    short_vol = returns.rolling(short_period).std() * np.sqrt(252)
    long_vol = returns.rolling(long_period).std() * np.sqrt(252)
    
    # Term structure
    term_structure = short_vol / long_vol.where(long_vol > 0, 1)
    
    # Signal based on term structure
    signal = 2 - term_structure  # Inverted: low short/long ratio is bullish
    
    return signal

# -------------------- New Normalization Approaches --------------------

def quantile_normalization(series: pd.Series, window: int, n_quantiles: int = 10) -> pd.Series:
    """Normalize to quantile buckets"""
    quantiles = pd.Series(index=series.index, dtype=float)
    
    for i in range(window, len(series)):
        window_data = series.iloc[i-window:i+1]
        if len(window_data) > 0:
            # Calculate quantile thresholds
            thresholds = [window_data.quantile(q/n_quantiles) for q in range(1, n_quantiles)]
            
            # Find which quantile current value falls into
            current_val = series.iloc[i]
            quantile = 0
            for threshold in thresholds:
                if current_val > threshold:
                    quantile += 1
            
            quantiles.iloc[i] = quantile / n_quantiles
    
    return quantiles

def sigmoid_normalization(series: pd.Series, scale: float = 1.0) -> pd.Series:
    """Sigmoid normalization for bounded output"""
    # Center the series
    centered = series - series.rolling(100).mean()
    
    # Scale by rolling std
    std = series.rolling(100).std()
    scaled = centered / std.where(std > 0, 1) * scale
    
    # Apply sigmoid
    return 1 / (1 + np.exp(-scaled))

def adaptive_minmax_normalization(series: pd.Series, window: int) -> pd.Series:
    """Min-max normalization with adaptive bounds"""
    # Use percentiles instead of min/max to handle outliers
    rolling_10 = series.rolling(window).quantile(0.1)
    rolling_90 = series.rolling(window).quantile(0.9)
    
    # Normalize
    normalized = (series - rolling_10) / (rolling_90 - rolling_10).where(rolling_90 > rolling_10, 1)
    
    # Clip to [0, 1] with small buffer
    return normalized.clip(-0.1, 1.1)

# -------------------- Combined Indicators --------------------

def momentum_acceleration_composite(df: pd.DataFrame, period: int) -> pd.Series:
    """Combine momentum with acceleration"""
    close = df['Close']
    
    # Momentum
    momentum = close.pct_change(period)
    
    # Acceleration
    accel = price_acceleration(df, period // 2)
    
    # Normalize both
    mom_norm = sigmoid_normalization(momentum, 10)
    accel_norm = sigmoid_normalization(accel, 5)
    
    # Composite with acceleration leading
    composite = mom_norm * 0.6 + accel_norm * 0.4
    
    return composite

def volume_regime_adjusted_trend(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Trend adjusted by volume regime"""
    close = df['Close']
    volume = df['Volume']
    
    # Calculate trend
    trend = pd.Series(index=close.index, dtype=float)
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                trend.iloc[i] = slope / close.iloc[i] * 100
            except:
                trend.iloc[i] = 0
    
    # Volume regime
    vol_ma = volume.rolling(vol_period).mean()
    vol_std = volume.rolling(vol_period).std()
    vol_zscore = (volume - vol_ma) / vol_std.where(vol_std > 0, 1)
    
    # Adjust trend by volume regime
    adjusted_trend = trend * (1 + vol_zscore.clip(-1, 1) * 0.5)
    
    return adjusted_trend

# -------------------- Sharpe --------------------

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

# -------------------- Evaluation --------------------

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
        if cond_name == 'PRICE_ACTION':
            period = params['period']
            k = params['k']
            strength = price_action_strength(df, period)
            signal = strength < k  # Buy on weak/bearish price action
        
        elif cond_name == 'SR_DISTANCE':
            period = params['period']
            k = params['k']
            distance = support_resistance_distance(df, period)
            signal = distance < k  # Buy near support
        
        elif cond_name == 'CANDLE_PATTERN':
            period = params['period']
            k = params['k']
            pattern = candle_pattern_score(df, period)
            signal = pattern > k  # Buy on bullish patterns
        
        elif cond_name == 'PRICE_JERK':
            period = params['period']
            k = params['k']
            jerk = price_acceleration_2nd(df, period)
            signal = jerk < k  # Buy on negative jerk (deceleration)
        
        elif cond_name == 'VOL_PRICE_EFF':
            period = params['period']
            k = params['k']
            efficiency = volume_price_efficiency(df, period)
            signal = efficiency > k  # Buy on high efficiency
        
        elif cond_name == 'TREND_CONSIST':
            short = params['short']
            long = params['long']
            k = params['k']
            consistency = trend_consistency_score(df, short, long)
            signal = consistency > k  # Buy on high consistency
        
        elif cond_name == 'PRICE_SKEW':
            period = params['period']
            k = params['k']
            skew = price_distribution_skew(df, period)
            signal = skew < k  # Buy on negative skew
        
        elif cond_name == 'PRICE_ACCEL':
            period = params['period']
            k = params['k']
            indicator = price_acceleration(df, period)
            signal = indicator > k
        
        elif cond_name == 'MOM_QUALITY':
            period = params['period']
            k = params['k']
            indicator = momentum_quality_score(df, period)
            signal = indicator > k
        
        elif cond_name == 'ADAPTIVE_ROC':
            base_period = params['base_period']
            k = params['k']
            indicator = adaptive_roc(df, base_period)
            signal = indicator > k
        
        elif cond_name == 'VOL_PRICE_CORR':
            period = params['period']
            k = params['k']
            indicator = volume_price_correlation(df, period)
            signal = indicator > k
        
        elif cond_name == 'VOL_WGT_TREND_STR':
            period = params['period']
            k = params['k']
            indicator = volume_weighted_trend_strength(df, period)
            signal = indicator > k
        
        elif cond_name == 'OBV_DIV':
            period = params['period']
            k = params['k']
            indicator = obv_divergence(df, period)
            signal = indicator > k
        
        elif cond_name == 'VOL_REGIME_MARKOV':
            period = params['period']
            k = params['k']
            indicator = volatility_regime_markov(df, period)
            signal = indicator > k
        
        elif cond_name == 'GARCH_VOL':
            period = params['period']
            k = params['k']
            indicator = garch_like_volatility(df, period)
            signal = indicator > k
        
        elif cond_name == 'VOL_TERM_STRUCT':
            short_p = params['short_period']
            long_p = params['long_period']
            k = params['k']
            indicator = volatility_term_structure(df, short_p, long_p)
            signal = indicator > k
        
        elif cond_name == 'MOM_ACCEL_COMP':
            period = params['period']
            k = params['k']
            indicator = momentum_acceleration_composite(df, period)
            signal = indicator > k
        
        elif cond_name == 'VOL_REG_ADJ_TREND':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = volume_regime_adjusted_trend(df, trend_p, vol_p)
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
    """Generate test conditions"""
    conditions = []
    
    # PRICE_ACTION - Price action strength
    for period in (10, 20, 50):
        for k in (-0.2, -0.1, 0.0, 0.1):
            conditions.append(('PRICE_ACTION', {'period': period, 'k': k}))

    # SR_DISTANCE - Support/Resistance distance
    for period in (20, 50, 100):
        for k in (-0.02, -0.01, 0.0):
            conditions.append(('SR_DISTANCE', {'period': period, 'k': k}))

    # CANDLE_PATTERN - Candlestick patterns
    for period in (10, 20):
        for k in (0.1, 0.2, 0.3, 0.4):
            conditions.append(('CANDLE_PATTERN', {'period': period, 'k': k}))

    # PRICE_JERK - Price acceleration 2nd derivative
    for period in (10, 20, 50):
        for k in (-0.01, 0.0, 0.01):
            conditions.append(('PRICE_JERK', {'period': period, 'k': k}))

    # VOL_PRICE_EFF - Volume price efficiency
    for period in (20, 50):
        for k in (0.01, 0.02, 0.03):
            conditions.append(('VOL_PRICE_EFF', {'period': period, 'k': k}))

    # TREND_CONSIST - Trend consistency
    for short in (10, 20):
        for long in (50, 100):
            for k in (0.5, 0.7, 0.9):
                conditions.append(('TREND_CONSIST', {'short': short, 'long': long, 'k': k}))

    # PRICE_SKEW - Price distribution skew
    for period in (20, 50, 100):
        for k in (-0.5, 0.0, 0.5):
            conditions.append(('PRICE_SKEW', {'period': period, 'k': k}))

    # PRICE_ACCEL - Price acceleration
    for period in [10, 20, 50]:
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            conditions.append(('PRICE_ACCEL', {
                'period': period,
                'k': k
            }))
    
    # MOM_QUALITY - Momentum quality score
    for period in [20, 50, 100]:
        for k in [-0.01, 0.0, 0.01, 0.02]:
            conditions.append(('MOM_QUALITY', {
                'period': period,
                'k': k
            }))
    
    # ADAPTIVE_ROC - Adaptive rate of change
    for base_period in [20, 50, 100]:
        for k in [-1.0, 0.0, 1.0, 2.0]:
            conditions.append(('ADAPTIVE_ROC', {
                'base_period': base_period,
                'k': k
            }))
    
    # VOL_PRICE_CORR - Volume price correlation
    for period in [20, 50, 100]:
        for k in [-1.0, -0.5, 0.0, 0.5]:
            conditions.append(('VOL_PRICE_CORR', {
                'period': period,
                'k': k
            }))
    
    # VOL_WGT_TREND_STR - Volume weighted trend strength
    for period in [50, 100, 150]:
        for k in [-0.01, 0.0, 0.01]:
            conditions.append(('VOL_WGT_TREND_STR', {
                'period': period,
                'k': k
            }))
    
    # OBV_DIV - OBV divergence
    for period in [20, 50, 100]:
        for k in [-1.0, -0.5, 0.0, 0.5]:
            conditions.append(('OBV_DIV', {
                'period': period,
                'k': k
            }))
    
    # VOL_REGIME_MARKOV - Volatility regime Markov
    for period in [20, 30, 50]:
        for k in [-0.5, 0.0, 0.5, 1.0]:
            conditions.append(('VOL_REGIME_MARKOV', {
                'period': period,
                'k': k
            }))
    
    # GARCH_VOL - GARCH-like volatility
    for period in [20, 50]:
        for k in [-1.0, -0.5, 0.0, 0.5]:
            conditions.append(('GARCH_VOL', {
                'period': period,
                'k': k
            }))
    
    # VOL_TERM_STRUCT - Volatility term structure
    for short_p in [10, 20]:
        for long_p in [50, 100]:
            for k in [0.8, 1.0, 1.2]:
                conditions.append(('VOL_TERM_STRUCT', {
                    'short_period': short_p,
                    'long_period': long_p,
                    'k': k
                }))
    
    # MOM_ACCEL_COMP - Momentum acceleration composite
    for period in [20, 50, 100]:
        for k in [0.4, 0.5, 0.6]:
            conditions.append(('MOM_ACCEL_COMP', {
                'period': period,
                'k': k
            }))
    
    # VOL_REG_ADJ_TREND - Volume regime adjusted trend
    for trend_p in [100, 130, 150]:
        for vol_p in [20, 30, 50]:
            for k in [-0.01, 0.0, 0.01]:
                conditions.append(('VOL_REG_ADJ_TREND', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    return conditions

# -------------------- Main --------------------

def main():
    """Main execution function"""
    print("Wave23: Alternative Momentum, Advanced Volume-Price, New Volatility Regimes")
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
    print("Top 20 Results - Wave23:")
    print("=" * 80)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave23 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave23 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave23 appended to history.log")
    
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