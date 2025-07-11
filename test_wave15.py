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

def efficiency_ratio(series, period):
    """Efficiency Ratio - measures trend strength"""
    change = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(period).sum()
    return change / volatility

def range_position(df, period):
    """Position of close within the range (0 to 1)"""
    high = df['High'].rolling(period).max()
    low = df['Low'].rolling(period).min()
    return (df['Close'] - low) / (high - low)

def volume_weighted_ma(df, period):
    """Volume Weighted Moving Average"""
    return (df['Close'] * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()

def money_flow_index(df, period=14):
    """Money Flow Index"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)
    
    mask = typical_price > typical_price.shift(1)
    positive_flow[mask] = money_flow[mask]
    negative_flow[~mask] = money_flow[~mask]
    
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

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
    
    if cond_name == 'EFF_RATIO':
        period = params['period']
        k = params['k']
        eff = efficiency_ratio(close, period)
        signal = eff > k
    
    elif cond_name == 'RANGE_POS':
        period = params['period']
        k = params['k']
        rp = range_position(df, period)
        signal = rp > k
    
    elif cond_name == 'VWMA_DEV':
        period = params['period']
        k = params['k']
        vwma = volume_weighted_ma(df, period)
        signal = (close / vwma - 1) > k
    
    elif cond_name == 'MFI_TREND':
        period = params['period']
        k = params['k']
        mfi = money_flow_index(df, period)
        signal = mfi > k
    
    elif cond_name == 'NORM_VOL_MOM':
        mom_period = params['mom_period']
        vol_period = params['vol_period']
        k = params['k']
        mom = roc(close, mom_period)
        vol = atr(df, vol_period) / close
        vol_ma = vol.rolling(vol_period).mean()
        norm_mom = mom / vol_ma
        signal = norm_mom > k
    
    elif cond_name == 'ADAPTIVE_TREND':
        fast = params['fast']
        slow = params['slow']
        k = params['k']
        eff = efficiency_ratio(close, slow)
        adaptive_period = fast + (slow - fast) * (1 - eff)
        adaptive_ma = close.copy()
        for i in range(slow, len(close)):
            if not np.isnan(adaptive_period.iloc[i]):
                period = max(1, int(adaptive_period.iloc[i]))
                adaptive_ma.iloc[i] = close.iloc[max(0, i-period):i].mean()
        signal = (close / adaptive_ma - 1) > k
    
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

# EFF_RATIO
for period in (10, 20, 30):
    for k in (0.3, 0.4, 0.5, 0.6):
        conditions.append(('EFF_RATIO', {'period': period, 'k': k}))

# RANGE_POS
for period in (20, 50, 100):
    for k in (0.7, 0.8, 0.85, 0.9):
        conditions.append(('RANGE_POS', {'period': period, 'k': k}))

# VWMA_DEV
for period in (20, 50, 100, 200):
    for k in (0.0, 0.005, 0.01):
        conditions.append(('VWMA_DEV', {'period': period, 'k': k}))

# MFI_TREND
for period in (14, 21, 28):
    for k in (50, 60, 70):
        conditions.append(('MFI_TREND', {'period': period, 'k': k}))

# NORM_VOL_MOM
for mom_period in (5, 10, 20):
    for vol_period in (20, 30):
        for k in (1.0, 1.5, 2.0):
            conditions.append(('NORM_VOL_MOM', {'mom_period': mom_period, 'vol_period': vol_period, 'k': k}))

# ADAPTIVE_TREND
for fast in (10, 20):
    for slow in (50, 100):
        for k in (0.0, 0.005, 0.01):
            conditions.append(('ADAPTIVE_TREND', {'fast': fast, 'slow': slow, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave15: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave15:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave15 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave15 appended to history.log")