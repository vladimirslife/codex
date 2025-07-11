#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 13: одно условие — abs(gap)/ATR14_prev < thr"""
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

# True Range & ATR14
prev_close=df["close"].shift(1)
tr=pd.concat([
    df["high"]-df["low"],
    (df["high"]-prev_close).abs(),
    (df["low"]-prev_close).abs()
],axis=1).max(axis=1)
atr14=tr.rolling(14).mean().shift(1)  # shift to avoid look-ahead

# gap
gap=(df["open"]/prev_close-1).abs()
normalized=gap/atr14

# next overnight
next_open=df["open"].shift(-1)
df["next_ov"]=next_open/df["close"]-1

thresholds=np.arange(0.0,1.01,0.05)
results=[]
for thr in thresholds:
    sig=(normalized<thr).astype(int)
    strat=sig*df["next_ov"]
    strat.fillna(0,inplace=True)
    excess=strat-DAILY_RF
    if excess.std()==0:
        continue
    sr=(excess.mean()/excess.std())*np.sqrt(252)
    trades=int(sig.sum())
    results.append((sr,trades,thr))

results.sort(key=lambda x: x[0],reverse=True)
print("=== Wave 13 top 3 ===")
for i,(sr,tr,thr) in enumerate(results[:3],1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: |gap|/ATR14_prev < {thr:.2f}")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,thr) in enumerate(results[:3],1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 13 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: abs(gap)/ATR14_prev < {thr:.2f}\n")