#!/usr/bin/env python3
"""
Wave 22 – Close Location Value (CLV) style: position of close in rolling high-low range.
Single-condition rules: pct_range < low_thr or > high_thr.
"""
import os, pandas as pd, numpy as np
CSV=os.environ.get('QQQ_CSV','4 - QQQ.csv')

df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)

close,high,low=df['close'],df['high'],df['low']

windows=[5,10,20,30]
thresholds=[0.05,0.1,0.9,0.95]
rules=[]
for n in windows:
  roll_high=high.rolling(n,min_periods=n).max()
  roll_low=low.rolling(n,min_periods=n).min()
  pct=(close-roll_low)/(roll_high-roll_low+1e-9)
  for thr in thresholds:
    if thr<0.5:
      rules.append((f'pct{n}_lt_{thr}',pct<thr))
    else:
      rules.append((f'pct{n}_gt_{thr}',pct>thr))

print('Wave22 rules',len(rules))
DAILY_RF=0.02/252

def eval(mask):
  strat=mask.astype(int)*df['next_ov_ret']; strat=strat.dropna()
  excess=strat-DAILY_RF; mean=excess.mean()*252; sd=excess.std()*np.sqrt(252)
  return (mean/sd if sd>0 else 0,int(mask.sum()))

best=None
for name,mask in rules:
  sh,tr=eval(mask)
  print(f"{name:15s} | Sharpe {sh:.3f} | trades {tr}")
  if best is None or sh>best[1]: best=(name,sh,tr)
print('Best',best)