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

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def stoch(df, period=14, k_period=3):
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    return k.rolling(k_period).mean()

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
    
    if cond_name == 'NORM_MOM':
        period = params['period']
        window = params['window']
        k = params['k']
        mom = roc(close, period)
        vol = close.rolling(window).std() / close
        norm_mom = mom / vol.replace(0, np.nan)
        signal = norm_mom > k
    
    elif cond_name == 'RSI_EMA_RATIO':
        rsi_period = params['rsi_period']
        ema_period = params['ema_period']
        k = params['k']
        rsi_val = rsi(close, rsi_period)
        ema_val = ema(close, ema_period)
        ratio = close / ema_val
        signal = (rsi_val > 50) & (ratio > k)
    
    elif cond_name == 'STOCH_TREND':
        stoch_period = params['stoch_period']
        ema_period = params['ema_period']
        k = params['k']
        stoch_val = stoch(df, stoch_period)
        trend = close / ema(close, ema_period) - 1
        signal = (stoch_val > 50) & (trend > k)
    
    elif cond_name == 'VOL_CONTRACTION':
        atr_period = params['atr_period']
        window = params['window']
        k = params['k']
        vol_ratio = atr(df, atr_period) / atr(df, atr_period).rolling(window).mean()
        signal = vol_ratio < k
    
    elif cond_name == 'ADAPTIVE_MOM':
        short = params['short']
        long = params['long']
        k = params['k']
        short_roc = roc(close, short)
        long_roc = roc(close, long)
        adaptive = short_roc - long_roc
        signal = adaptive > k
    
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

# NORM_MOM
for period in (5, 10, 20):
    for window in (20, 50):
        for k in (0.5, 1.0, 1.5):
            conditions.append(('NORM_MOM', {'period': period, 'window': window, 'k': k}))

# RSI_EMA_RATIO (combined single condition)
for rsi_period in (14, 21):
    for ema_period in (50, 100, 200):
        for k in (1.0, 1.01, 1.02):
            conditions.append(('RSI_EMA_RATIO', {'rsi_period': rsi_period, 'ema_period': ema_period, 'k': k}))

# STOCH_TREND (combined single condition)
for stoch_period in (14, 21):
    for ema_period in (50, 100, 200):
        for k in (0.0, 0.01, 0.02):
            conditions.append(('STOCH_TREND', {'stoch_period': stoch_period, 'ema_period': ema_period, 'k': k}))

# VOL_CONTRACTION
for atr_period in (14, 20):
    for window in (50, 100):
        for k in (0.7, 0.8, 0.9):
            conditions.append(('VOL_CONTRACTION', {'atr_period': atr_period, 'window': window, 'k': k}))

# ADAPTIVE_MOM
for short in (5, 10):
    for long in (20, 50):
        for k in (0.0, 0.005, 0.01):
            conditions.append(('ADAPTIVE_MOM', {'short': short, 'long': long, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave13: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave13:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave13 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave13 appended to history.log")