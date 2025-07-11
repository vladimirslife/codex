#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 21: z-score метрика одного условия
 (gap - beta*prev_gap - delta*isMon - MA_gap_N_prev) / STD_gap_N_prev < thr
Ограничение по количеству сделок >3000.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

FILE="4 - QQQ.csv"
ANNUAL_RF=0.02
DAILY_RF=ANNUAL_RF/252

if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

df=pd.read_csv(FILE)
df.columns=[c.lower() for c in df.columns]
df["date"]=pd.to_datetime(df["date"])
df=df[df["date"]>=pd.Timestamp("2006-01-01")].sort_values("date").reset_index(drop=True)

prev_close=df["close"].shift(1)
gap=df["open"]/prev_close-1
prev_gap=gap.shift(1)

df["is_mon"]=(df["date"].dt.weekday==0).astype(int)
next_ov=df["open"].shift(-1)/df["close"]-1

N_vals=[5,10,20]
beta_vals=np.round(np.arange(-0.23,-0.205,0.004),5)
delta_vals=np.round(np.arange(-0.0015,0.0016,0.0005),6)
thr_vals=np.round(np.arange(-1.0,1.01,0.05),3)

best=None
for N in N_vals:
    ma=gap.rolling(N).mean().shift(1)
    std=gap.rolling(N).std(ddof=0).shift(1)
    std=std.replace(0,np.nan)
    for beta in beta_vals:
        base=gap - beta*prev_gap
        for delta in delta_vals:
            metric=(base - delta*df["is_mon"] - ma)/std
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
                sr=(excess.mean()/sd)*np.sqrt(252)
                if best is None or sr>best[0]:
                    best=(sr,trades,beta,delta,N,thr)

if best:
    sr,tr,beta,delta,N,thr=best
    print(f"Wave21 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: zScore_N={N} < {thr:.2f} (beta={beta:.3f}, delta={delta:.4f})")
    with open("history.log","a",encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | Wave 21 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: zScore_N{N}<{thr:.2f} beta={beta:.3f} delta={delta:.4f}\n")
else:
    print("Wave21: нет комбинации с >3000 сделок")