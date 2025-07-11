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

def load_data(ticker, start, end):
    cache_file = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors='coerce')
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is not None and not df.empty:
        df.to_csv(cache_file)
    return df.apply(pd.to_numeric, errors='coerce') if df is not None else pd.DataFrame()

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

# -------------------- Sharpe --------------------

def sharpe_ratio(returns):
    if returns.empty:
        return 0.0
    std = returns.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    excess = returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

# -------------------- Evaluation --------------------

def evaluate(df, cond_name, params):
    close = df['Close']
    open_next = df['Open'].shift(-1)
    daily_returns = (open_next - close) / close
    
    signal = pd.Series(False, index=df.index)
    
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
    
    else:
        raise ValueError(f"Unknown condition: {cond_name}")
    
    strategy_returns = daily_returns.copy()
    strategy_returns[~signal] = 0.0
    
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

conditions = conditions[:MAX_COMBOS]
print(f"Wave23: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
if not prices.empty:
    results = [evaluate(prices, name, params) for name, params in conditions]
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\nTop 10 Wave23:")
    print(res_sorted.head(10).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave23 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("Top 3 Wave23 appended to history.log")
else:
    print("Error: Unable to load data")