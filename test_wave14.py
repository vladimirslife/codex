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

def roc(series, period):
    return series.pct_change(period)

def atr(df, period):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def hull_ma(series, period):
    """Hull Moving Average"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = 2 * series.rolling(half_period).mean()
    wma_full = series.rolling(period).mean()
    return (wma_half - wma_full).rolling(sqrt_period).mean()

def kama(series, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    direction = (series - series.shift(period)).abs()
    volatility = (series.diff().abs()).rolling(period).sum()
    efficiency_ratio = direction / volatility
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    smooth = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama_series = pd.Series(index=series.index, dtype=float)
    kama_series.iloc[period] = series.iloc[period]
    
    for i in range(period + 1, len(series)):
        kama_series.iloc[i] = kama_series.iloc[i-1] + smooth.iloc[i] * (series.iloc[i] - kama_series.iloc[i-1])
    
    return kama_series

def linear_reg_slope(series, period):
    """Linear regression slope"""
    def slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    return series.rolling(period).apply(slope, raw=True)

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
    
    if cond_name == 'HULL_TREND':
        period = params['period']
        k = params['k']
        hull = hull_ma(close, period)
        signal = (close / hull - 1) > k
    
    elif cond_name == 'KAMA_DEV':
        period = params['period']
        k = params['k']
        kama_val = kama(close, period)
        signal = (close / kama_val - 1) > k
    
    elif cond_name == 'LR_SLOPE':
        period = params['period']
        k = params['k']
        slope = linear_reg_slope(close, period)
        signal = slope > k
    
    elif cond_name == 'VOL_RATIO':
        short = params['short']
        long = params['long']
        k = params['k']
        vol_short = close.rolling(short).std()
        vol_long = close.rolling(long).std()
        signal = (vol_short / vol_long) < k
    
    elif cond_name == 'PRICE_ACCEL':
        period = params['period']
        k = params['k']
        roc1 = roc(close, period)
        roc2 = roc(close, period).shift(period)
        accel = roc1 - roc2
        signal = accel > k
    
    elif cond_name == 'EMA_CURVE':
        short = params['short']
        mid = params['mid']
        long = params['long']
        k = params['k']
        curve = (ema(close, short) - ema(close, mid)) / (ema(close, mid) - ema(close, long))
        signal = curve > k
    
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

# HULL_TREND
for period in (16, 20, 30):
    for k in (0.0, 0.005, 0.01, 0.015):
        conditions.append(('HULL_TREND', {'period': period, 'k': k}))

# KAMA_DEV
for period in (10, 20, 30):
    for k in (0.0, 0.005, 0.01, 0.015):
        conditions.append(('KAMA_DEV', {'period': period, 'k': k}))

# LR_SLOPE
for period in (20, 50, 100):
    for k in (0.0, 0.001, 0.002):
        conditions.append(('LR_SLOPE', {'period': period, 'k': k}))

# VOL_RATIO
for short in (10, 20):
    for long in (50, 100):
        for k in (0.7, 0.8, 0.9):
            conditions.append(('VOL_RATIO', {'short': short, 'long': long, 'k': k}))

# PRICE_ACCEL
for period in (5, 10, 20):
    for k in (0.0, 0.005, 0.01):
        conditions.append(('PRICE_ACCEL', {'period': period, 'k': k}))

# EMA_CURVE
for short in (10, 20):
    for mid in (50, 100):
        for long in (150, 200):
            if short < mid < long:
                for k in (1.0, 1.5, 2.0):
                    conditions.append(('EMA_CURVE', {'short': short, 'mid': mid, 'long': long, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave14: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave14:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave14 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave14 appended to history.log")