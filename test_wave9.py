import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math, os

TICKER="SPY"
START_DATE="1990-01-01"
END_DATE=datetime.today().strftime("%Y-%m-%d")
RF_ANNUAL=0.02
RF_DAILY=RF_ANNUAL/252.0
MAX_COMBOS=1000

#------------------- Data -------------------

def load_data(ticker,start,end):
    cache=f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache):
        df=pd.read_csv(cache,index_col=0,parse_dates=True)
        return df.apply(pd.to_numeric,errors='coerce')
    df=yf.download(ticker,start=start,end=end,progress=False,auto_adjust=False)
    df.to_csv(cache)
    return df.apply(pd.to_numeric,errors='coerce')

#------------------- Indicators -------------

def ema(s,span):
    return s.ewm(span=span,adjust=False).mean()

def roc(s,period):
    return s.pct_change(period)

def atr(df,period):
    h,l,c=df['High'],df['Low'],df['Close']
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore(series,window):
    mean=series.rolling(window).mean()
    std=series.rolling(window).std()
    return (series-mean)/std

def percentile(series,window):
    roll_min=series.rolling(window).min()
    roll_max=series.rolling(window).max()
    return (series-roll_min)/(roll_max-roll_min).replace(0,np.nan)

#------------------- Metric -----------------

def sharpe(returns):
    if returns.empty:
        return 0.0
    std=returns.std(ddof=0)
    if std==0 or np.isclose(std,0):
        return 0.0
    excess=returns-RF_DAILY
    return math.sqrt(252)*excess.mean()/std

#------------------- Evaluate ---------------

def evaluate(df,name,p):
    close=df['Close']
    open_next=df['Open'].shift(-1)
    r_day=(open_next-close)/close

    if name=='VOL_ADJ_MOM':
        period=p['period']
        k=p['k']
        mom=roc(close,period)
        vol=atr(df,period)/close
        ratio=mom/vol
        signal=ratio>k
    elif name=='ROC_PCTL':
        period=p['period']
        window=p['window']
        thr=p['thr']
        signal=percentile(roc(close,period),window)>thr
    elif name=='ZPRICE':
        window=p['window']
        k=p['k']
        signal=zscore(close,window)>k
    elif name=='EMA_SPREAD':
        short=p['short']; long=p['long']; delta=p['delta']
        signal=(ema(close,short)/ema(close,long)-1)>delta
    else:
        raise ValueError('unknown')

    strat=r_day.copy(); strat[~signal]=0.0
    trades=int(signal.sum())
    sr=sharpe(strat.dropna())
    total=(1+strat.fillna(0)).prod()-1
    return{'condition':name,'params':p,'sharpe':sr,'trades':trades,'total_return':total}

#------------------- Grid -------------------
conds=[]
# VOL_ADJ_MOM
for period in (5,10,20):
    for k in (1,1.5,2,2.5):
        conds.append(('VOL_ADJ_MOM',{'period':period,'k':k}))
# ROC_PCTL
for period in (5,10,20):
    for window in (50,100):
        for thr in (0.8,0.85,0.9):
            conds.append(('ROC_PCTL',{'period':period,'window':window,'thr':thr}))
# ZPRICE
for window in (50,100,200):
    for k in (1.5,2,2.5):
        conds.append(('ZPRICE',{'window':window,'k':k}))
# EMA_SPREAD dynamic scaling
for short in (5,10,15):
    for long in (150,200):
        for delta in (0.008,0.012,0.016):
            if long>short:
                conds.append(('EMA_SPREAD',{'short':short,'long':long,'delta':delta}))

conds=conds[:MAX_COMBOS]
print(f"Wave9: total conditions {len(conds)}")

#------------------ Run --------------------
prices=load_data(TICKER,START_DATE,END_DATE)
results=[evaluate(prices,n,p) for n,p in conds]
res=pd.DataFrame(results)
res_sorted=res.sort_values(['sharpe','trades'],ascending=[False,False])
print("\nTop 10 Wave9:")
print(res_sorted.head(10).to_string(index=False))

with open('history.log','a') as f:
    for _,row in res_sorted.head(3).iterrows():
        f.write(f"Wave9 | Condition: {row['condition']} | Params: {row['params']} | Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
print('Top 3 Wave9 appended to history.log')