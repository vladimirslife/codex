#!/usr/bin/env python3
"""
Wave 23 – расстояние цены от SMA.
Условие: (close / SMA(n) - 1) < -thr  или > thr.
"""
import os, pandas as pd
CSV=os.environ.get('QQQ_CSV','4 - QQQ.csv')
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)
close=df['close']
periods=[10,20,30,40]
ths=[0.0,0.01,0.02,0.03]
rules=[]
for n in periods:
  sma=close.rolling(n,min_periods=n).mean()
  dist=close/sma-1
  for t in ths:
    rules.append((f'dist{n}_gt_{t}',dist>t))
    rules.append((f'dist{n}_lt_-{t}',dist<-t))
print('Wave23 rules',len(rules))
DAILY_RF=0.02/252

def eval(mask):
  strat=mask.astype(int)*df['next_ov_ret']; strat=strat.dropna()
  excess=strat-DAILY_RF; m=excess.mean()*252; sd=excess.std()*252**0.5
  return m/sd if sd>0 else 0,int(mask.sum())
for name,mask in rules:
  sh,tr=eval(mask)
  print(name,sh,tr)