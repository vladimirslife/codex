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

def vwap(df, period):
    """Volume Weighted Average Price"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()

def relative_volume(df, period):
    """Current volume relative to average volume"""
    return df['Volume'] / df['Volume'].rolling(period).mean()

def price_density(df, period, bins=10):
    """Price density - how much time price spends at current level"""
    close = df['Close']
    density = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        window = close.iloc[i-period:i]
        # Skip if window contains NaN values
        if window.isna().any():
            density.iloc[i] = np.nan
            continue
        hist, edges = np.histogram(window, bins=bins)
        current_bin = np.digitize(close.iloc[i], edges) - 1
        current_bin = min(max(0, current_bin), bins-1)
        density.iloc[i] = hist[current_bin] / period
    
    return density

def trend_consistency(series, period):
    """Measures how consistent the trend is"""
    changes = series.pct_change()
    positive = (changes > 0).rolling(period).sum()
    return positive / period

def volatility_rank(df, period, lookback):
    """Rank current volatility vs historical"""
    vol = atr(df, period) / df['Close']
    rank = pd.Series(index=vol.index, dtype=float)
    
    for i in range(lookback, len(vol)):
        window = vol.iloc[max(0, i-lookback):i]
        rank.iloc[i] = (window < vol.iloc[i]).sum() / len(window)
    
    return rank

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
    
    if cond_name == 'VWAP_TREND':
        period = params['period']
        k = params['k']
        vwap_val = vwap(df, period)
        signal = (close / vwap_val - 1) > k
    
    elif cond_name == 'REL_VOL_HIGH':
        period = params['period']
        k = params['k']
        rel_vol = relative_volume(df, period)
        signal = rel_vol > k
    
    elif cond_name == 'PRICE_DENSITY_LOW':
        period = params['period']
        k = params['k']
        density = price_density(df, period)
        signal = density < k
    
    elif cond_name == 'TREND_CONSISTENCY':
        period = params['period']
        k = params['k']
        consistency = trend_consistency(close, period)
        signal = consistency > k
    
    elif cond_name == 'VOL_RANK_LOW':
        period = params['period']
        lookback = params['lookback']
        k = params['k']
        rank = volatility_rank(df, period, lookback)
        signal = rank < k
    
    elif cond_name == 'COMPOSITE_MOM':
        short = params['short']
        long = params['long']
        k = params['k']
        # Composite momentum: short-term momentum adjusted by long-term trend
        short_mom = roc(close, short)
        long_trend = close / ema(close, long) - 1
        composite = short_mom * (1 + long_trend)
        signal = composite > k
    
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

# VWAP_TREND
for period in (20, 50, 100, 200):
    for k in (-0.01, 0.0, 0.005, 0.01):
        conditions.append(('VWAP_TREND', {'period': period, 'k': k}))

# REL_VOL_HIGH
for period in (20, 50):
    for k in (1.5, 2.0, 2.5):
        conditions.append(('REL_VOL_HIGH', {'period': period, 'k': k}))

# PRICE_DENSITY_LOW
for period in (50, 100):
    for k in (0.1, 0.15, 0.2):
        conditions.append(('PRICE_DENSITY_LOW', {'period': period, 'k': k}))

# TREND_CONSISTENCY
for period in (20, 50, 100):
    for k in (0.6, 0.65, 0.7, 0.75):
        conditions.append(('TREND_CONSISTENCY', {'period': period, 'k': k}))

# VOL_RANK_LOW
for period in (14, 20):
    for lookback in (100, 200):
        for k in (0.2, 0.3, 0.4):
            conditions.append(('VOL_RANK_LOW', {'period': period, 'lookback': lookback, 'k': k}))

# COMPOSITE_MOM
for short in (5, 10):
    for long in (100, 150, 200):
        for k in (0.0, 0.005, 0.01):
            conditions.append(('COMPOSITE_MOM', {'short': short, 'long': long, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave16: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave16:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave16 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave16 appended to history.log")