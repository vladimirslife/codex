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
    df.to_csv(cache_file)
    return df.apply(pd.to_numeric, errors='coerce')

# -------------------- Indicators --------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window).mean()

def regime_detection(df, period):
    """Detect market regime based on volatility clustering"""
    close = df['Close']
    returns = close.pct_change()
    
    # Rolling volatility
    vol = returns.rolling(period).std()
    
    # Volatility regime (high vs low)
    vol_percentile = vol.rolling(period * 2).rank(pct=True)
    return vol_percentile

def market_breadth(df, period):
    """Market breadth indicator based on up/down days"""
    close = df['Close']
    volume = df['Volume']
    
    # Up/down classification
    is_up = close > close.shift(1)
    
    # Volume-weighted breadth
    up_volume = volume.where(is_up, 0).rolling(period).sum()
    down_volume = volume.where(~is_up, 0).rolling(period).sum()
    
    breadth = (up_volume - down_volume) / (up_volume + down_volume)
    return breadth

def price_momentum_divergence(df, short_period, long_period):
    """Divergence between short and long term momentum"""
    close = df['Close']
    
    short_mom = close.pct_change(short_period)
    long_mom = close.pct_change(long_period)
    
    divergence = short_mom - long_mom
    return divergence

def volume_price_divergence(df, period):
    """Divergence between volume trend and price trend"""
    close = df['Close']
    volume = df['Volume']
    
    # Normalize trends
    price_trend = (close / close.rolling(period).mean() - 1)
    vol_trend = (volume / volume.rolling(period).mean() - 1)
    
    divergence = vol_trend - price_trend
    return divergence

def range_position_indicator(df, period):
    """Where price sits within its recent range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Rolling range
    period_high = high.rolling(period).max()
    period_low = low.rolling(period).min()
    
    # Position within range (0 = bottom, 1 = top)
    position = (close - period_low) / (period_high - period_low)
    return position

def volatility_adjusted_momentum(df, mom_period, vol_period):
    """Momentum scaled by inverse volatility"""
    close = df['Close']
    returns = close.pct_change()
    
    momentum = close.pct_change(mom_period)
    volatility = returns.rolling(vol_period).std()
    
    # Scale momentum by inverse volatility
    adj_momentum = momentum / volatility.where(volatility > 0, np.nan)
    return adj_momentum

def market_tension_indicator(df, period):
    """Measure of market tension using high-low-close relationships"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    open_price = df['Open']
    
    # Daily tension: how far close is from the extremes
    daily_range = high - low
    close_to_high = (high - close) / daily_range.where(daily_range > 0, 1)
    close_to_low = (close - low) / daily_range.where(daily_range > 0, 1)
    
    # Tension score
    tension = close_to_high - close_to_low  # Negative when close near high
    
    # Rolling average
    avg_tension = tension.rolling(period).mean()
    return avg_tension

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
    
    if cond_name == 'REGIME_VOL':
        period = params['period']
        k = params['k']
        regime = regime_detection(df, period)
        signal = regime < k  # Buy in low volatility regime
    
    elif cond_name == 'MARKET_BREADTH':
        period = params['period']
        k = params['k']
        breadth = market_breadth(df, period)
        signal = breadth < k  # Buy on negative breadth
    
    elif cond_name == 'MOMENTUM_DIV':
        short = params['short']
        long = params['long']
        k = params['k']
        div = price_momentum_divergence(df, short, long)
        signal = div < k  # Buy on negative divergence
    
    elif cond_name == 'VOL_PRICE_DIV':
        period = params['period']
        k = params['k']
        div = volume_price_divergence(df, period)
        signal = div > k  # Buy on positive divergence
    
    elif cond_name == 'RANGE_POS':
        period = params['period']
        k = params['k']
        pos = range_position_indicator(df, period)
        signal = pos < k  # Buy near bottom of range
    
    elif cond_name == 'VOL_ADJ_MOM':
        mom_p = params['mom_period']
        vol_p = params['vol_period']
        k = params['k']
        adj_mom = volatility_adjusted_momentum(df, mom_p, vol_p)
        signal = adj_mom > k
    
    elif cond_name == 'MARKET_TENSION':
        period = params['period']
        k = params['k']
        tension = market_tension_indicator(df, period)
        signal = tension > k  # Buy on positive tension
    
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

# REGIME_VOL
for period in (20, 50, 100):
    for k in (0.2, 0.3, 0.4, 0.5):
        conditions.append(('REGIME_VOL', {'period': period, 'k': k}))

# MARKET_BREADTH
for period in (10, 20, 50):
    for k in (-0.2, -0.1, 0.0, 0.1):
        conditions.append(('MARKET_BREADTH', {'period': period, 'k': k}))

# MOMENTUM_DIV
for short in (5, 10):
    for long in (20, 50, 100):
        for k in (-0.02, -0.01, 0.0):
            conditions.append(('MOMENTUM_DIV', {'short': short, 'long': long, 'k': k}))

# VOL_PRICE_DIV
for period in (20, 50, 100):
    for k in (0.0, 0.1, 0.2, 0.3):
        conditions.append(('VOL_PRICE_DIV', {'period': period, 'k': k}))

# RANGE_POS
for period in (20, 50, 100):
    for k in (0.2, 0.3, 0.4):
        conditions.append(('RANGE_POS', {'period': period, 'k': k}))

# VOL_ADJ_MOM
for mom_p in (10, 20):
    for vol_p in (20, 50):
        for k in (0.0, 0.5, 1.0):
            conditions.append(('VOL_ADJ_MOM', {'mom_period': mom_p, 'vol_period': vol_p, 'k': k}))

# MARKET_TENSION
for period in (10, 20, 50):
    for k in (-0.1, 0.0, 0.1, 0.2):
        conditions.append(('MARKET_TENSION', {'period': period, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave20: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave20:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave20 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave20 appended to history.log")