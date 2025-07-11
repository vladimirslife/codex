#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 6: условие abs(gap) < threshold"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

FILE = "4 - QQQ.csv"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252

if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

df = pd.read_csv(FILE)
df.rename(columns=lambda c: c.lower(), inplace=True)
df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] >= pd.Timestamp("2006-01-01")].sort_values("Date").reset_index(drop=True)

df["prev_close"] = df["close"].shift(1)
df["gap"] = df["open"] / df["prev_close"] - 1

df["Next_Open"] = df["open"].shift(-1)
df["next_ov"] = df["Next_Open"] / df["close"] - 1

thresholds = np.arange(0.0005, 0.01, 0.0005)
results = []
for thr in thresholds:
    sig = (df["gap"].abs() < thr).astype(int)
    strat = sig * df["next_ov"]
    strat.fillna(0, inplace=True)
    excess = strat - DAILY_RF
    sr = 0 if excess.std() == 0 else (excess.mean()*252)/(excess.std()*np.sqrt(252))
    trades = int(sig.sum())
    results.append((sr, trades, thr))

results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 6 top 3 ===")
for i,(sr,tr,thr) in enumerate(top3,1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: abs(gap) < {thr:.4f}")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,thr) in enumerate(top3,1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 6 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: abs(gap) < {thr:.4f}\n")