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

def mean_reversion_zscore(df, period):
    """Z-score from moving average"""
    close = df['Close']
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    
    zscore = (close - ma) / std.where(std > 0, np.nan)
    return zscore

def consecutive_moves(df, period):
    """Count consecutive up/down days"""
    close = df['Close']
    returns = close.pct_change()
    
    # Count consecutive positive returns
    is_positive = returns > 0
    consecutive = pd.Series(0, index=close.index)
    
    for i in range(1, len(close)):
        if is_positive.iloc[i] == is_positive.iloc[i-1]:
            consecutive.iloc[i] = consecutive.iloc[i-1] + 1
        else:
            consecutive.iloc[i] = 1
    
    # Normalize by period
    normalized = consecutive / period
    return normalized

def price_channel_position(df, period):
    """Position within Donchian channel"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    
    # Position in channel (0 = bottom, 1 = top)
    position = (close - lowest) / (highest - lowest).where(highest > lowest, 1)
    return position

def volume_weighted_reversal(df, period):
    """Volume-weighted mean reversion indicator"""
    close = df['Close']
    volume = df['Volume']
    
    # Volume-weighted average price
    vwap = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
    
    # Deviation weighted by relative volume
    deviation = (close - vwap) / vwap
    rel_volume = volume / volume.rolling(period).mean()
    
    weighted_deviation = deviation * rel_volume
    return weighted_deviation

def gap_reversal(df, period):
    """Overnight gap reversal pattern"""
    open_price = df['Open']
    close = df['Close']
    
    # Overnight gap
    gap = (open_price - close.shift(1)) / close.shift(1)
    
    # Average gap size
    avg_gap = gap.rolling(period).mean().abs()
    
    # Normalized gap
    normalized_gap = gap / avg_gap.where(avg_gap > 0, 1)
    return normalized_gap

def intraday_reversal_strength(df, period):
    """Strength of intraday reversals"""
    high = df['High']
    low = df['Low']
    open_price = df['Open']
    close = df['Close']
    
    # Intraday range
    daily_range = high - low
    
    # Reversal strength: how far close moved from open relative to range
    reversal = (close - open_price) / daily_range.where(daily_range > 0, 1)
    
    # Rolling average of absolute reversal
    avg_reversal = reversal.abs().rolling(period).mean()
    
    # Current reversal relative to average
    relative_reversal = reversal / avg_reversal.where(avg_reversal > 0, 1)
    return relative_reversal

def market_structure_break(df, period):
    """Detect breaks in market structure"""
    close = df['Close']
    
    # Recent highs and lows
    recent_high = close.rolling(period).max()
    recent_low = close.rolling(period).min()
    
    # Previous period highs and lows
    prev_high = recent_high.shift(period)
    prev_low = recent_low.shift(period)
    
    # Structure break: current close vs previous range
    above_prev_high = (close - prev_high) / prev_high
    below_prev_low = (prev_low - close) / prev_low
    
    # Combined structure break indicator
    structure_break = above_prev_high - below_prev_low
    return structure_break

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
    
    if cond_name == 'MEAN_REV_Z':
        period = params['period']
        k = params['k']
        zscore = mean_reversion_zscore(df, period)
        signal = zscore < k  # Buy on oversold
    
    elif cond_name == 'CONSEC_MOVES':
        period = params['period']
        k = params['k']
        consec = consecutive_moves(df, period)
        signal = consec > k  # Buy after consecutive moves
    
    elif cond_name == 'CHANNEL_POS':
        period = params['period']
        k = params['k']
        position = price_channel_position(df, period)
        signal = position < k  # Buy near channel bottom
    
    elif cond_name == 'VOL_WT_REV':
        period = params['period']
        k = params['k']
        vwr = volume_weighted_reversal(df, period)
        signal = vwr < k  # Buy on negative weighted deviation
    
    elif cond_name == 'GAP_REVERSAL':
        period = params['period']
        k = params['k']
        gap = gap_reversal(df, period)
        signal = gap < k  # Buy on down gaps
    
    elif cond_name == 'INTRADAY_REV':
        period = params['period']
        k = params['k']
        rev_strength = intraday_reversal_strength(df, period)
        signal = rev_strength < k  # Buy on negative reversals
    
    elif cond_name == 'STRUCT_BREAK':
        period = params['period']
        k = params['k']
        struct = market_structure_break(df, period)
        signal = struct > k  # Buy on upward structure breaks
    
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

# MEAN_REV_Z - Mean reversion z-score
for period in (20, 50, 100, 200):
    for k in (-2.0, -1.5, -1.0, -0.5):
        conditions.append(('MEAN_REV_Z', {'period': period, 'k': k}))

# CONSEC_MOVES - Consecutive moves
for period in (10, 20):
    for k in (0.3, 0.4, 0.5, 0.6):
        conditions.append(('CONSEC_MOVES', {'period': period, 'k': k}))

# CHANNEL_POS - Channel position
for period in (20, 50, 100):
    for k in (0.2, 0.3, 0.4):
        conditions.append(('CHANNEL_POS', {'period': period, 'k': k}))

# VOL_WT_REV - Volume weighted reversal
for period in (20, 50, 100):
    for k in (-0.02, -0.01, 0.0):
        conditions.append(('VOL_WT_REV', {'period': period, 'k': k}))

# GAP_REVERSAL - Gap reversal
for period in (20, 50):
    for k in (-1.0, -0.5, 0.0):
        conditions.append(('GAP_REVERSAL', {'period': period, 'k': k}))

# INTRADAY_REV - Intraday reversal
for period in (10, 20, 50):
    for k in (-1.0, -0.5, 0.0):
        conditions.append(('INTRADAY_REV', {'period': period, 'k': k}))

# STRUCT_BREAK - Market structure break
for period in (20, 50):
    for k in (0.0, 0.01, 0.02, 0.03):
        conditions.append(('STRUCT_BREAK', {'period': period, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave22: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
if not prices.empty:
    results = [evaluate(prices, name, params) for name, params in conditions]
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\nTop 10 Wave22:")
    print(res_sorted.head(10).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave22 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("Top 3 Wave22 appended to history.log")
else:
    print("Error: Unable to load data")