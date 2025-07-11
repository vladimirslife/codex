#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 26: (gap + alpha*d_gap)/(EMA_abs_gap + gamma) < thr
where d_gap = gap - prev_gap (1-day ROC).
One logical condition, searching for Sharpe >=1.3, trades >3000.
"""
import pandas as pd, numpy as np, os, datetime, math
FILE='4 - QQQ.csv'
ANNUAL_RF=0.02
DAILY_RF=ANNUAL_RF/252
if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

df=pd.read_csv(FILE)
df.columns=[c.lower() for c in df.columns]
df['date']=pd.to_datetime(df['date'])
df=df[df['date']>=pd.Timestamp('2006-01-01')].sort_values('date').reset_index(drop=True)
prev_close=df['close'].shift(1)
gap=df['open']/prev_close-1
prev_gap=gap.shift(1)
d_gap=gap-prev_gap  # first derivative
next_ov=df['open'].shift(-1)/df['close']-1

span=10  # fixed EMA span
ema_abs=gap.abs().ewm(span=span,adjust=False).mean().shift(1)
ema_abs=ema_abs.replace(0,np.nan)

a_vals=np.round(np.arange(-1.0,1.05,0.1),2)
thr_vals=np.round(np.arange(0.002,0.01,0.0005),4)
gamma_vals=[1e-4,2e-4,5e-4]

best=None
for alpha in a_vals:
    for gamma in gamma_vals:
        metric=(gap + alpha*d_gap)/(ema_abs+gamma)
        for thr in thr_vals:
            sig=(metric<thr).astype(int)
            trades=int(sig.sum())
            if trades<3000:
                continue
            excess=sig*next_ov-DAILY_RF
            sd=excess.std()
            if sd==0:
                continue
            sr=(excess.mean()/sd)*math.sqrt(252)
            if best is None or sr>best[0]:
                best=(sr,trades,alpha,gamma,thr)

if best:
    sr,tr,alpha,gamma,thr=best
    print(f"Wave26 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: (gap + {alpha}*d_gap)/(EMA_abs_gap+{gamma}) < {thr}")
    with open('history.log','a',encoding='utf-8') as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} | Wave 26 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: alpha={alpha} gamma={gamma} thr={thr}\n")
else:
    print('Wave26: no combo >3000 trades')