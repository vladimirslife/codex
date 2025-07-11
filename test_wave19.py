#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 19: ещё более тонкая настройка для достижения Sharpe >=1.3"""
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

beta_center=-0.2175
beta_vals=np.round(np.arange(beta_center-0.005, beta_center+0.00501, 0.00025),5)  # 41

delta_center=-0.00075
delta_vals=np.round(np.arange(delta_center-0.0002, delta_center+0.000201, 0.000025),6)  # 17

thr_center=0.0030
thr_vals=np.round(np.arange(thr_center-0.0001, thr_center+0.000101, 0.00001),6)  # 21

best=None
for beta in beta_vals:
    metric_base=gap - beta*prev_gap
    for delta in delta_vals:
        metric=metric_base - delta*df["is_mon"]
        for thr in thr_vals:
            sig=(metric < thr).astype(int)
            trades=int(sig.sum())
            if trades<3000:
                continue
            strat=sig*next_ov
            excess=strat-DAILY_RF
            std=excess.std()
            if std==0:
                continue
            sr=(excess.mean()/std)*np.sqrt(252)
            if best is None or sr>best[0]:
                best=(sr,trades,beta,delta,thr)

if best:
    sr,tr,beta,delta,thr=best
    print(f"Wave19 BEST Sharpe={sr:.4f}, Trades={tr}, Condition: gap -({beta:.5f})*prev_gap -({delta:.5f})*isMon < {thr:.5f}")
    with open("history.log","a",encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | Wave 19 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: gap - {beta:.5f}*prev_gap - {delta:.5f}*isMon < {thr:.5f}\n")
else:
    print("Wave19: no combo with >3000 trades")