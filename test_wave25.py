#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 25: (gap - beta*prev_gap)/(EWMA_std_gap_N_prev + gamma) < thr
EWMA_std_gap_N_prev — эксп. скользящее среднее rolling-std(|gap|) за N дней, сдвинутое на 1 день.
Цель: Sharpe >= 1.3 при >3000 сделок.
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
next_ov=df['open'].shift(-1)/df['close']-1

N_vals=[10,20,30]
beta_vals=np.round(np.arange(-0.23,-0.20,0.002),4)
thr_vals=np.round(np.arange(0.0025,0.0041,0.0003),4)
gamma_vals=[1e-4,2e-4,5e-4]

best=None
for N in N_vals:
    roll_std=gap.rolling(N).std(ddof=0)
    ewma_std=roll_std.ewm(span=N,adjust=False).mean().shift(1)
    ewma_std=ewma_std.replace(0,np.nan)
    for gamma in gamma_vals:
        denom=ewma_std+gamma
        for beta in beta_vals:
            metric=(gap - beta*prev_gap)/denom
            for thr in thr_vals:
                sig=(metric < thr).astype(int)
                trades=int(sig.sum())
                if trades<3000:
                    continue
                strat=sig*next_ov
                excess=strat-DAILY_RF
                sd=excess.std()
                if sd==0:
                    continue
                sr=(excess.mean()/sd)*math.sqrt(252)
                if best is None or sr>best[0]:
                    best=(sr,trades,beta,N,gamma,thr)

if best:
    sr,tr,beta,N,gamma,thr=best
    print(f"Wave25 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: (gap - {beta}*prev_gap)/(EWMA_std_gap_N={N}+{gamma}) < {thr}")
    with open('history.log','a',encoding='utf-8') as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} | Wave 25 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: beta={beta} N={N} gamma={gamma} thr={thr}\n")
else:
    print('Wave25: no combo >3000 trades')