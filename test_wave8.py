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

# ---------------------- Data Loader -------------------------

def load_prices(ticker, start, end):
    cache = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors='coerce')
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache)
    return df.apply(pd.to_numeric, errors='coerce')

# ---------------------- Indicators --------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def roc(series, period):
    return series.pct_change(period)

def rolling_std(series, window):
    return series.rolling(window).std()

def percentile(series, window):
    rmin = series.rolling(window).min()
    rmax = series.rolling(window).max()
    denom = (rmax - rmin).replace(0, np.nan)
    return (series - rmin) / denom

def adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ---------------------- Metric ------------------------------

def sharpe(ret):
    if ret.empty:
        return 0.0
    s = ret.std(ddof=0)
    if s == 0 or np.isclose(s,0):
        return 0.0
    return math.sqrt(252) * (ret - RF_DAILY).mean() / s

# ---------------------- Evaluation --------------------------

def evaluate(df, name, params):
    close = df['Close']
    open_next = df['Open'].shift(-1)
    dret = (open_next - close) / close

    if name == 'MOM_VOL_ADJ':
        period = params['period']
        window = params['window']
        k = params['k']
        norm_mom = roc(close, period) / rolling_std(close, window)
        signal = norm_mom > k
    elif name == 'VOL_PCTL_LOW':
        window = params['window']
        thr = params['thr']
        signal = percentile(rolling_std(close, window), window) < thr
    elif name == 'RES_BREAK':
        period = params['period']
        buffer = params['buffer']
        res = close.rolling(period).max().shift(1)
        signal = close > res * (1 + buffer)
    elif name == 'TREND_STRENGTH':
        short = params['short']
        long = params['long']
        delta = params['delta']
        signal = (ema(close, short)/ema(close, long)-1) > delta
    else:
        raise ValueError('Unknown condition')

    strat = dret.copy()
    strat[~signal] = 0.0
    trades = int(signal.sum())
    sr = sharpe(strat.dropna())
    tot = (1+strat.fillna(0)).prod()-1
    return {'condition':name,'params':params,'sharpe':sr,'trades':trades,'total_return':tot}

# ---------------------- Grid ----------------------------
conds=[]
# MOM_VOL_ADJ
for period in (5,10,20):
    for window in (20,50):
        for k in (0.5,1.0,1.5,2.0):
            conds.append(('MOM_VOL_ADJ',{'period':period,'window':window,'k':k}))
# VOL_PCTL_LOW
for window in (50,100,200):
    for thr in (0.2,0.3,0.4):
        conds.append(('VOL_PCTL_LOW',{'window':window,'thr':thr}))
# RES_BREAK
for period in (50,100,200):
    for buffer in (0.0,0.005,0.01):
        conds.append(('RES_BREAK',{'period':period,'buffer':buffer}))
# TREND_STRENGTH composite
for short in (5,10,15,20):
    for long in (150,200):
        for delta in (0.005,0.01,0.015):
            if long>short:
                conds.append(('TREND_STRENGTH',{'short':short,'long':long,'delta':delta}))

conds = conds[:MAX_COMBOS]
print(f"Wave8: total conditions {len(conds)}")

# ---------------------- Run ---------------------------
prices=load_prices(TICKER,START_DATE,END_DATE)
results=[evaluate(prices,n,p) for n,p in conds]
res=pd.DataFrame(results)
res_sorted=res.sort_values(['sharpe','trades'],ascending=[False,False])
print("\nTop 10 Wave8:")
print(res_sorted.head(10).to_string(index=False))

with open('history.log','a') as f:
    for _,row in res_sorted.head(3).iterrows():
        f.write(f"Wave8 | Condition: {row['condition']} | Params: {row['params']} | Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
print('Top 3 Wave8 appended to history.log')