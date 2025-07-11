#!/usr/bin/env python3
"""
Wave 17 – гипотеза: сильная медвежья дневная свеча (большая красная body) ведёт к положительному овернайту.
Одно условие: (close/open - 1) <= -thr, thr в [0.5%, 1%, 1.5%, 2%].
Проверим и шире (до -0.1%).
"""
import os, pandas as pd, numpy as np
from datetime import datetime

CSV=os.environ.get("QQQ_CSV","4 - QQQ.csv")
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1); df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)

body_ret=df['close']/df['open']-1
THR=[0.005,0.01,0.015,0.02,0.03,0.04]
rules=[]
for t in THR:
    rules.append((f"body_le_-{int(t*1000)}bp", (body_ret <= -t)))

print(f"Wave17 rules: {len(rules)}")
ANNUAL_RF=0.02; DAILY_RF=ANNUAL_RF/252

def eval(sig):
    strat=sig.shift(0)*df['next_ov_ret']; strat=strat.dropna()
    excess=strat-DAILY_RF; sr=excess.mean()*252; sd=excess.std()*np.sqrt(252)
    return sr/sd if sd>0 else 0, int(sig.sum())

best=None
for name,mask in rules:
    sh,tr=eval(mask.astype(int))
    print(f"{name:20s} | Sharpe {sh:.3f} | trades {tr}")
    if best is None or sh>best[1]: best=(name,sh,tr)
print("Best:",best)