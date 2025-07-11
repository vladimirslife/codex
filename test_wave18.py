#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 18: тонкая настройка (gap - beta*prev_gap - delta*isMon) < thr"""
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

beta_center=-0.216
beta_vals=np.arange(beta_center-0.002, beta_center+0.0021, 0.0005) # 9 уровней

delta_center=-0.0008
delta_vals=np.arange(delta_center-0.0002, delta_center+0.00021, 0.00005) # 9 уровней

thr_center=0.0030
thr_vals=np.arange(thr_center-0.00005, thr_center+0.000051, 0.00001) # 11 уровней

best=None
for beta in beta_vals:
    metric_base=gap - beta*prev_gap  # compute once
    for delta in delta_vals:
        metric=metric_base - delta*df["is_mon"]
        abs_metric=metric  # asymmetric condition, keep < thr
        for thr in thr_vals:
            sig=(abs_metric < thr).astype(int)
            trades=int(sig.sum())
            if trades<3000:
                continue
            strat=sig*next_ov
            excess=strat-DAILY_RF
            if excess.std()==0:
                continue
            sr=(excess.mean()/excess.std())*np.sqrt(252)
            if best is None or sr>best[0]:
                best=(sr,trades,beta,delta,thr)

if best:
    sr,tr,beta,delta,thr=best
    print(f"Лучшее: Sharpe={sr:.4f}, Trades={tr}, Cond: gap -({beta:.5f})*prev_gap -({delta:.5f})*isMon < {thr:.5f}")
    with open("history.log","a",encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | Wave 18 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: gap - {beta:.5f}*prev_gap - {delta:.5f}*isMon < {thr:.5f}\n")
else:
    print("Нет подходящей комбинации >3000 сделок")