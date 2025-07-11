#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 14: одно неявное условие (gap + alpha*weekday) < thr"""
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

df["weekday"]=df["date"].dt.weekday
prev_close=df["close"].shift(1)
gap=df["open"]/prev_close-1
next_ov=df["open"].shift(-1)/df["close"]-1

alpha_values=np.arange(-0.0015,0.0016,0.0001)
thr_values=np.arange(-0.005,0.006,0.0001)

best=None
for a in alpha_values:
    metric=gap + a*df["weekday"]
    for thr in thr_values:
        sig=(metric < thr).astype(int)
        trades=int(sig.sum())
        if trades<3000:
            continue
        strat=sig*next_ov
        ex=strat-DAILY_RF
        if ex.std()==0:
            continue
        sr=(ex.mean()/ex.std())*np.sqrt(252)
        if best is None or sr>best[0]:
            best=(sr,trades,a,thr)

print("=== Wave 14 best ===")
if best:
    sr,tr,a_val,thr_val=best
    print(f"Sharpe={sr:.4f}, Trades={tr}, Cond: gap + {a_val:.4f}*weekday < {thr_val:.4f}")
    with open("history.log","a",encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | Wave 14 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: gap + {a_val:.4f}*weekday < {thr_val:.4f}\n")
else:
    print("No combination satisfied trade count >>3000")