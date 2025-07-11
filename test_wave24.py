#!/usr/bin/env python3
"""
Wave 24 – ATR(14)/Close ratio (volatility compression / expansion).
Single-condition rules: atr_ratio < low_thr or > high_thr with various thresholds.
"""
import os, pandas as pd, numpy as np
CSV=os.environ.get('QQQ_CSV','4 - QQQ.csv')
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)

high,low,close=df['high'],df['low'],df['close']
tr=pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
atr=tr.rolling(14).mean()
atr_ratio=atr/close

low_th=[0.005,0.007,0.01,0.012]
high_th=[0.025,0.03,0.035]
rules=[]
for t in low_th:
    rules.append((f'atr_ratio_lt_{t}',atr_ratio<t))
for t in high_th:
    rules.append((f'atr_ratio_gt_{t}',atr_ratio>t))
print('Wave24 rules',len(rules))

DAILY_RF=0.02/252

def eval(mask):
    strat=mask.astype(int)*df['next_ov_ret']; strat=strat.dropna()
    excess=strat-DAILY_RF; mean=excess.mean()*252; sd=excess.std()*np.sqrt(252)
    return (mean/sd if sd>0 else 0,int(mask.sum()))

best=None
for name,mask in rules:
    sh,tr=eval(mask)
    print(f"{name:20s} | Sharpe {sh:.3f} | trades {tr}")
    if best is None or sh>best[1]: best=(name,sh,tr)
print('Best',best)