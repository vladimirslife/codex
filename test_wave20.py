#!/usr/bin/env python3
# Wave 20 – rolling percentile rank of close price.
import os, pandas as pd, numpy as np
CSV=os.environ.get('QQQ_CSV','4 - QQQ.csv')
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)
close=df['close']
windows=[20,40,60]
qs=[0.65,0.7,0.75,0.8]
rules=[]
for n in windows:
  rank=close.rolling(n).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
  for q in qs:
    rules.append((f"prank{n}_gt_{q}",(rank>q)))
print('Wave20 rules',len(rules))
DAILY_RF=0.02/252

def eval(mask):
  strat=mask.astype(int)*df['next_ov_ret']; strat=strat.dropna()
  excess=strat-DAILY_RF; m=excess.mean()*252; sd=excess.std()*np.sqrt(252)
  return (m/sd if sd>0 else 0,int(mask.sum()))
for name,msk in rules:
  sh,tr=eval(msk)
  print(name,sh,tr)