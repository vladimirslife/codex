#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 4: условие gap_prev < threshold, где gap= open_i / close_{i-1} -1"""
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

# prev close
df["prev_close"] = df["close"].shift(1)
# gap
df["gap"] = df["open"] / df["prev_close"] - 1
# next overnight return
df["Next_Open"] = df["open"].shift(-1)
df["next_ov"] = df["Next_Open"] / df["close"] - 1

thresholds = np.arange(-0.04, 0.041, 0.002)  # -4% .. +4%
results = []
for thr in thresholds:
    sig = (df["gap"] < thr).astype(int)  # buy after negative gap beyond thr
    strat = sig * df["next_ov"]
    strat.fillna(0, inplace=True)
    excess = strat - DAILY_RF
    sr = 0 if excess.std() == 0 else (excess.mean()*252)/(excess.std()*np.sqrt(252))
    trades = int(sig.sum())
    results.append((sr, trades, thr))

results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 4 top 3 ===")
for i,(sr,tr,thr) in enumerate(top3,1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: gap < {thr:.3f}")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,thr) in enumerate(top3,1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 4 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: gap < {thr:.3f}\n")