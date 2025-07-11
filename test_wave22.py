#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 22: условие (gap - beta*prev_gap - delta*isMon)/EMA_abs_gap_N_prev < thr"""
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

beta_vals=np.round(np.arange(-0.23,-0.205,0.002),5)
delta_vals=np.round(np.arange(-0.0012,0.0013,0.0004),6)
thr_vals=np.round(np.arange(0.0025,0.0041,0.0001),5)
span_vals=[5,10,20]

best=None
for span in span_vals:
    ema_gap=gap.abs().ewm(span=span, adjust=False).mean().shift(1)
    ema_gap=ema_gap.replace(0,np.nan)
    for beta in beta_vals:
        base=gap - beta*prev_gap
        for delta in delta_vals:
            metric=(base - delta*df["is_mon"])/ema_gap
            for thr in thr_vals:
                sig=(metric < thr).astype(int)
                trades=int(sig.sum())
                if trades<3000:
                    continue
                excess=sig*next_ov-DAILY_RF
                sd=excess.std()
                if sd==0:
                    continue
                sr=(excess.mean()/sd)*np.sqrt(252)
                if best is None or sr>best[0]:
                    best=(sr,trades,beta,delta,thr,span)

if best:
    sr,tr,beta,delta,thr,span=best
    print(f"Wave22 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: (gap - {beta:.3f}*prev_gap - {delta:.4f}*isMon)/EMA_abs_gap(span={span}) < {thr:.4f}")
    with open("history.log","a",encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | Wave 22 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: gapNormEMA span{span} < {thr:.4f} beta={beta:.3f} delta={delta:.4f}\n")
else:
    print("Wave22: no combo >3000 trades")