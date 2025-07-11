#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 31: (gap + alpha*d_gap)/(ema_abs_weekday_N + gamma) < thr
ema_abs_weekday_N: для каждого weekday считается EWMA(|gap|) со span=N, затем shift(1).
Ищем Sharpe>=1.3 при >3000 сделок.
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

df['wday']=df['date'].dt.weekday
prev_close=df['close'].shift(1)
gap=df['open']/prev_close-1
prev_gap=gap.shift(1)
d_gap=gap-prev_gap
next_ov=df['open'].shift(-1)/df['close']-1

N_vals=[5,10]
alpha_vals=[-0.5,0,0.5]
gamma_vals=[1e-4,2e-4]
thr_vals=np.round(np.arange(0.002,0.0051,0.0005),4)

best=None
for N in N_vals:
    # compute weekday-specific EMA of |gap|
    ema_abs=pd.Series(index=df.index,dtype=float)
    for wd in range(5):
        mask=df['wday']==wd
        ema=gap.abs().where(mask).ewm(span=N,adjust=False).mean()
        ema_abs.update(ema)
    ema_abs=ema_abs.shift(1).replace(0,np.nan)
    for alpha in alpha_vals:
        num=gap + alpha*d_gap
        for gamma in gamma_vals:
            metric=num/(ema_abs+gamma)
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
                    best=(sr,trades,alpha,N,gamma,thr)

if best:
    sr,tr,alpha,N,gamma,thr=best
    print(f"Wave31 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: alpha={alpha}, N={N}, gamma={gamma}, thr={thr}")
    with open('history.log','a',encoding='utf-8') as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} | Wave 31 | best | Sharpe={sr:.4f} | Trades={tr} | alpha={alpha} N={N} gamma={gamma} thr={thr}\n")
else:
    print('Wave31: no combo >3000 trades')