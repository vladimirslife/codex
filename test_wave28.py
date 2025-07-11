#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 28: one condition
 (gap + alpha*d_gap + beta*d2_gap) / (ema_gap_N + k*ema_std_N + gamma) < thr
where ema_gap_N, ema_std_N are EWMA mean and std of gap (shifted).
"""
import pandas as pd, numpy as np, os, datetime, math
FILE="4 - QQQ.csv"
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
prev_d_gap=d_gap.shift(1)
d2_gap=d_gap-prev_d_gap
next_ov=df['open'].shift(-1)/df['close']-1

gamma=1e-4
N_vals=[5,10,20]
alpha_vals=[-0.5,0.0,0.5]
beta_vals=[-0.3,0.0,0.3]
ks=[0.5,1.0,2.0]
thr_vals=np.round(np.arange(0.002,0.0061,0.0005),4)

best=None
for N in N_vals:
    ema_mean=gap.ewm(span=N,adjust=False).mean().shift(1)
    ema_sq=(gap**2).ewm(span=N,adjust=False).mean().shift(1)
    ema_var=ema_sq - ema_mean**2
    ema_var[ema_var<0]=np.nan
    ema_std=np.sqrt(ema_var)
    for alpha in alpha_vals:
        for beta in beta_vals:
            num=gap + alpha*d_gap + beta*d2_gap
            for k in ks:
                denom=ema_mean.abs()+k*ema_std+gamma
                metric=num/denom
                for thr in thr_vals:
                    sig=(metric < thr).astype(int)
                    trades=int(sig.sum())
                    if trades<3000:
                        continue
                    excess=sig*next_ov-DAILY_RF
                    sd=excess.std()
                    if sd==0:
                        continue
                    sr=(excess.mean()/sd)*math.sqrt(252)
                    if best is None or sr>best[0]:
                        best=(sr,trades,alpha,beta,k,N,thr)

if best:
    sr,tr,alpha,beta,k,N,thr=best
    print(f"Wave28 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: metric(alpha={alpha},beta={beta},k={k},N={N}) < {thr}")
    with open('history.log','a',encoding='utf-8') as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} | Wave 28 | best | Sharpe={sr:.4f} | Trades={tr} | alpha={alpha} beta={beta} k={k} N={N} thr={thr}\n")
else:
    print('Wave28: no combo >3000 trades')