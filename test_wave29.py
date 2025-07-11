#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 29: (gap + alpha*d_gap)/(EMA_abs_fast) - beta*(EMA_abs_slow/EMA_abs_fast) < thr
Проверяем Sharpe >=1.3 и >3000 сделок.
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
d_gap=gap-prev_gap
next_ov=df['open'].shift(-1)/df['close']-1

fast_spans=[3,5,8]
slow_spans=[15,20,30]
alpha_vals=[-0.5,0.0,0.5]
beta_vals=[0.0,0.5,1.0]
thr_vals=np.round(np.arange(0.002,0.0061,0.0005),4)

best=None
for fast in fast_spans:
    ema_fast=gap.abs().ewm(span=fast,adjust=False).mean().shift(1)
    for slow in slow_spans:
        if slow<=fast:
            continue
        ema_slow=gap.abs().ewm(span=slow,adjust=False).mean().shift(1)
        ratio=ema_slow/ema_fast.replace(0,np.nan)
        for alpha in alpha_vals:
            num=(gap + alpha*d_gap)/ema_fast
            for beta in beta_vals:
                metric=num - beta*ratio
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
                        best=(sr,trades,alpha,beta,fast,slow,thr)

if best:
    sr,tr,alpha,beta,fast,slow,thr=best
    print(f"Wave29 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: metric(alpha={alpha},beta={beta},fast={fast},slow={slow}) < {thr}")
    with open('history.log','a',encoding='utf-8') as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} | Wave 29 | best | Sharpe={sr:.4f} | Trades={tr} | alpha={alpha} beta={beta} fast={fast} slow={slow} thr={thr}\n")
else:
    print('Wave29: no combo >3000 trades')