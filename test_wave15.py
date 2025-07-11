#!/usr/bin/env python3
"""
Wave 15 – эксперимент: Ichimoku Cloud (price vs Senkou Span A/B) и узкая/широкая Bollinger Bandwidth.
"""

import os, pandas as pd, numpy as np
from datetime import datetime
from typing import List, Tuple, Callable

FILE=os.environ.get("QQQ_CSV","4 - QQQ.csv")
START="2006-01-01"
ANNUAL_RF=0.02; DAILY_RF=ANNUAL_RF/252
TOP=20

df=pd.read_csv(FILE)
df.columns=df.columns.str.lower()
df['date']=pd.to_datetime(df['date'])
df=df[df['date']>=START].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)

close=df['close']; high=df['high']; low=df['low']

Rule=Tuple[str,Callable[[pd.DataFrame],pd.Series]]
rules:List[Rule]=[]

# Ichimoku parameters default 9,26,52
conv=(high.rolling(9).max()+low.rolling(9).min())/2
base=(high.rolling(26).max()+low.rolling(26).min())/2
senkou_a=((conv+base)/2).shift(26)
senkou_b=((high.rolling(52).max()+low.rolling(52).min())/2).shift(26)

rules.append(("price_gt_kumo",lambda df,c=close,a=senkou_a,b=senkou_b:(c>np.maximum(a,b))))
rules.append(("price_lt_kumo",lambda df,c=close,a=senkou_a,b=senkou_b:(c<np.minimum(a,b))))

# Bollinger bandwidth
for n in [10,20]:
  mid=close.rolling(n).mean(); sd=close.rolling(n).std();
  bw=(sd*4)/mid # width
  for thr in [0.04,0.06,0.08]:
    rules.append((f"bbwidth{n}_lt_{thr}",lambda df,b=bw,t=thr:(b<t)))
  for thr in [0.12,0.15]:
    rules.append((f"bbwidth{n}_gt_{thr}",lambda df,b=bw,t=thr:(b>t)))

rules=rules[:100]
print(f"Wave15 rules: {len(rules)}")

def evaluate(sig):
  strat=sig.shift(0)*df['next_ov_ret']
  strat=strat.dropna();excess=strat-DAILY_RF
  sr=excess.mean()*252; sd=excess.std()*np.sqrt(252)
  sharpe=sr/sd if sd>0 else 0; trades=int(sig.sum());
  return sharpe,trades

results=[]
for name,func in rules:
  sig=func(df).astype(int);
  sh,tr=evaluate(sig)
  results.append((name,sh,tr))
results.sort(key=lambda x:x[1],reverse=True)
print('Top results Wave15:')
for i,(n,sh,tr) in enumerate(results[:TOP],1):
  print(f"{i:2d}. {n:20s} | Sharpe {sh:.3f} | Trades {tr}")

a=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
with open('history.log','a') as f:
  for n,sh,tr in results[:3]:
    f.write(f"[{a}] Wave15 | {n} | Sharpe={sh:.3f} | Trades={tr}\n")
print('Logged top3.')