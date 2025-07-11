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

def vwap_anchored(df, anchor_period):
    """Anchored VWAP - resets every N periods"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = pd.Series(index=df.index, dtype=float)
    
    for i in range(anchor_period, len(df)):
        start_idx = i - anchor_period
        tp_slice = typical_price.iloc[start_idx:i+1]
        vol_slice = df['Volume'].iloc[start_idx:i+1]
        vwap.iloc[i] = (tp_slice * vol_slice).sum() / vol_slice.sum()
    
    return vwap

def volume_price_trend(df, period):
    """Volume Price Trend indicator"""
    close = df['Close']
    volume = df['Volume']
    vpt = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
    return vpt / vpt.rolling(period).mean()

def on_balance_volume_norm(df, period):
    """Normalized On-Balance Volume"""
    close = df['Close']
    volume = df['Volume']
    
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv / obv.rolling(period).mean()

def volume_weighted_rsi(df, period):
    """RSI weighted by volume"""
    close = df['Close']
    volume = df['Volume']
    
    delta = close.diff()
    gain = (delta * volume).clip(lower=0)
    loss = (-delta * volume).clip(lower=0)
    
    avg_gain = gain.rolling(period).sum() / volume.rolling(period).sum()
    avg_loss = loss.rolling(period).sum() / volume.rolling(period).sum()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def accumulation_distribution(df, period):
    """Accumulation/Distribution Line normalized"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    
    ad = (clv * volume).cumsum()
    return ad / ad.rolling(period).mean()

def volume_momentum(df, short, long):
    """Volume momentum - short vs long volume ratio"""
    vol_short = df['Volume'].rolling(short).mean()
    vol_long = df['Volume'].rolling(long).mean()
    return vol_short / vol_long

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
    
    if cond_name == 'VWAP_ANCHORED':
        anchor = params['anchor']
        k = params['k']
        vwap = vwap_anchored(df, anchor)
        signal = (close / vwap - 1) > k
    
    elif cond_name == 'VPT_TREND':
        period = params['period']
        k = params['k']
        vpt = volume_price_trend(df, period)
        signal = vpt > k
    
    elif cond_name == 'OBV_NORM':
        period = params['period']
        k = params['k']
        obv = on_balance_volume_norm(df, period)
        signal = obv > k
    
    elif cond_name == 'VOL_RSI':
        period = params['period']
        k = params['k']
        vrsi = volume_weighted_rsi(df, period)
        signal = vrsi > k
    
    elif cond_name == 'ACC_DIST':
        period = params['period']
        k = params['k']
        ad = accumulation_distribution(df, period)
        signal = ad > k
    
    elif cond_name == 'VOL_MOM':
        short = params['short']
        long = params['long']
        k = params['k']
        vol_mom = volume_momentum(df, short, long)
        signal = vol_mom > k
    
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

# VWAP_ANCHORED - based on Wave 16 success
for anchor in (50, 100, 150, 200):
    for k in (-0.015, -0.01, -0.005, 0.0, 0.005):
        conditions.append(('VWAP_ANCHORED', {'anchor': anchor, 'k': k}))

# VPT_TREND
for period in (20, 50, 100):
    for k in (0.95, 1.0, 1.05):
        conditions.append(('VPT_TREND', {'period': period, 'k': k}))

# OBV_NORM
for period in (20, 50, 100):
    for k in (0.95, 1.0, 1.05, 1.1):
        conditions.append(('OBV_NORM', {'period': period, 'k': k}))

# VOL_RSI
for period in (14, 21, 28):
    for k in (45, 50, 55, 60):
        conditions.append(('VOL_RSI', {'period': period, 'k': k}))

# ACC_DIST
for period in (20, 50, 100):
    for k in (0.95, 1.0, 1.05):
        conditions.append(('ACC_DIST', {'period': period, 'k': k}))

# VOL_MOM
for short in (5, 10, 20):
    for long in (50, 100):
        for k in (1.0, 1.2, 1.5):
            conditions.append(('VOL_MOM', {'short': short, 'long': long, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave17: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave17:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave17 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave17 appended to history.log")