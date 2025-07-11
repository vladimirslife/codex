#!/usr/bin/env python3
"""Wave 21 – длинные скользящие средние (100–250 дней). Single-condition rules."""
import os, pandas as pd, numpy as np
CSV=os.environ.get('QQQ_CSV','4 - QQQ.csv')
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)
close=df['close']
periods=[100,150,200,250]
rules=[]
for n in periods:
  sma=close.rolling(n,min_periods=n).mean()
  rules.append((f'close_gt_sma{n}',(close>sma)))
  rules.append((f'close_lt_sma{n}',(close<sma)))
  ema=close.ewm(span=n,adjust=False,min_periods=n).mean()
  rules.append((f'close_gt_ema{n}',(close>ema)))
  rules.append((f'close_lt_ema{n}',(close<ema)))
print('Wave21 rules',len(rules))
DAILY_RF=0.02/252

def eval(mask):
  strat=mask.astype(int)*df['next_ov_ret']; strat=strat.dropna()
  excess=strat-DAILY_RF; mean=excess.mean()*252; sd=excess.std()*np.sqrt(252)
  return (mean/sd if sd>0 else 0,int(mask.sum()))
results=[(name,*eval(msk)) for name,msk in rules]
results.sort(key=lambda x:x[1],reverse=True)
for n,s,t in results: print(n,s,t)
print('best',results[0])