import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math, os

TICKER="SPY"
START_DATE="1990-01-01"
END_DATE=datetime.today().strftime("%Y-%m-%d")
RF_ANNUAL=0.02
RF_DAILY=RF_ANNUAL/252
MAX_COMBOS=1000

# --------- Data loader ---------

def load_data(ticker,start,end):
    cache=f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache):
        df=pd.read_csv(cache,index_col=0,parse_dates=True)
        return df.apply(pd.to_numeric,errors='coerce')
    df=yf.download(ticker,start=start,end=end,progress=False,auto_adjust=False)
    df.to_csv(cache)
    return df.apply(pd.to_numeric,errors='coerce')

# --------- Indicators ----------

def ema(s,span):
    return s.ewm(span=span,adjust=False).mean()

def roc(s,period):
    return s.pct_change(period)

def atr(df,period):
    h,l,c=df['High'],df['Low'],df['Close']
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore(series,window):
    m=series.rolling(window).mean(); sd=series.rolling(window).std()
    return (series-m)/sd

def percentile(series,window):
    rmin=series.rolling(window).min(); rmax=series.rolling(window).max();
    return (series-rmin)/(rmax-rmin).replace(0,np.nan)

# -------- Sharpe --------------

def sharpe(ret):
    if ret.empty: return 0.0
    sd=ret.std(ddof=0)
    if sd==0 or np.isclose(sd,0): return 0.0
    return math.sqrt(252)*(ret-RF_DAILY).mean()/sd

# -------- Evaluation ----------

def evaluate(df,name,p):
    close=df['Close']; open_next=df['Open'].shift(-1); day_ret=(open_next-close)/close
    signal=pd.Series(False,index=df.index)

    if name=='VOL_SCALED_MOM':
        period=p['period']; k=p['k']
        mom=roc(close,period)
        vol=atr(df,period)/close
        score=mom/vol.replace(0,np.nan)
        signal=score>k
    elif name=='REL_MOM_RATIO':
        short=p['short']; long=p['long']; thr=p['thr']
        ratio=roc(close,short)/roc(close,long).replace(0,np.nan)
        signal=ratio>thr
    elif name=='TREND_Z':
        short=p['short']; long=p['long']; win=p['window']; k=p['k']
        trend=ema(close,short)/ema(close,long)-1
        signal=zscore(trend,win)>k
    elif name=='DEV_PCTL':
        long=p['long']; win=p['window']; thr=p['thr']
        dev=close/ema(close,long)-1
        signal=percentile(dev,win)>thr
    else:
        raise ValueError

    strat=day_ret.copy(); strat[~signal]=0.0
    n=int(signal.sum()); sr=sharpe(strat.dropna()); tot=(1+strat.fillna(0)).prod()-1
    return {'condition':name,'params':p,'sharpe':sr,'trades':n,'total_return':tot}

# -------- Grid ---------------
conds=[]
# VOL_SCALED_MOM
for period in (5,10,20):
    for k in (1.5,2,2.5,3):
        conds.append(('VOL_SCALED_MOM',{'period':period,'k':k}))
# REL_MOM_RATIO
for short in (5,10):
    for long in (50,100):
        if long>short:
            for thr in (1.0,1.2,1.4):
                conds.append(('REL_MOM_RATIO',{'short':short,'long':long,'thr':thr}))
# TREND_Z
for short in (10,20):
    for long in (150,200):
        for win in (100,200):
            for k in (1,1.5,2):
                conds.append(('TREND_Z',{'short':short,'long':long,'window':win,'k':k}))
# DEV_PCTL
for long in (150,200):
    for win in (100,200):
        for thr in (0.8,0.85,0.9):
            conds.append(('DEV_PCTL',{'long':long,'window':win,'thr':thr}))

conds=conds[:MAX_COMBOS]
print(f"Wave10: total conditions {len(conds)}")

# -------- Run ---------------
prices=load_data(TICKER,START_DATE,END_DATE)
results=[evaluate(prices,n,p) for n,p in conds]
res=pd.DataFrame(results)
res_sorted=res.sort_values(['sharpe','trades'],ascending=[False,False])
print('\nTop 10 Wave10:')
print(res_sorted.head(10).to_string(index=False))

with open('history.log','a') as f:
    for _,row in res_sorted.head(3).iterrows():
        f.write(f"Wave10 | Condition: {row['condition']} | Params: {row['params']} | Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
print('Top 3 Wave10 appended to history.log')