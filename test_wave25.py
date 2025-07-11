#!/usr/bin/env python3
"""
Wave 25 – Know Sure Thing (KST) oscillator filters.
Single-condition: KST > 0 or KST < 0 with various parameter sets.
"""
import os, pandas as pd, numpy as np
CSV=os.environ.get('QQQ_CSV','4 - QQQ.csv')
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)
close=df['close']

param_sets=[(10,10,15,20,30,30,40,65,9), (10,15,20,30,30,40,45,65,9)]
rules=[]
for p in param_sets:
  r1,r2,r3,r4,s1,s2,s3,s4,signal=p
  roc1=close.pct_change(r1)
  roc2=close.pct_change(r2)
  roc3=close.pct_change(r3)
  roc4=close.pct_change(r4)
  sma1=roc1.rolling(s1).mean()
  sma2=roc2.rolling(s2).mean()
  sma3=roc3.rolling(s3).mean()
  sma4=roc4.rolling(s4).mean()
  kst=100*(sma1+ sma2*2 + sma3*3 + sma4*4)
  rules.append((f'kst_{r1}_{signal}_gt0',kst>0))
  rules.append((f'kst_{r1}_{signal}_lt0',kst<0))

print('Wave25 rules',len(rules))
DAILY_RF=0.02/252

def eval(mask):
    strat=mask.astype(int)*df['next_ov_ret']; strat=strat.dropna()
    excess=strat-DAILY_RF; m=excess.mean()*252; sd=excess.std()*np.sqrt(252)
    return (m/sd if sd>0 else 0, int(mask.sum()))
for name,mask in rules:
    sh,tr=eval(mask)
    print(name,sh,tr)