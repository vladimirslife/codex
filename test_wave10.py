#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 10: сложное условие gap < g_thr И daily_return > dr_thr"""
import pandas as pd
import numpy as np
import os
from itertools import product
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

# Features
df["prev_close"] = df["close"].shift(1)
df["gap"] = df["open"] / df["prev_close"] - 1
df["daily_return"] = df["close"] / df["open"] - 1

df["Next_Open"] = df["open"].shift(-1)
df["next_ov"] = df["Next_Open"] / df["close"] - 1

# Grid
gap_thresholds = np.arange(0.001, 0.01, 0.001)  # 0.1%..1%
ret_thresholds = np.arange(-0.005, 0.006, 0.001)  # -0.5%..0.5%

best = []
for g_thr, r_thr in product(gap_thresholds, ret_thresholds):
    sig = ((df["gap"] < g_thr) & (df["daily_return"] > r_thr)).astype(int)
    strat = sig * df["next_ov"]
    strat.fillna(0, inplace=True)
    excess = strat - DAILY_RF
    if excess.std() == 0:
        continue
    sr = (excess.mean() / excess.std()) * np.sqrt(252)
    trades = int(sig.sum())
    best.append((sr, trades, g_thr, r_thr))

best.sort(key=lambda x: x[0], reverse=True)

top3 = best[:3]
print("=== Wave 10 top 3 ===")
for i,(sr,tr,g,r) in enumerate(top3,1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: (gap < {g:.4f}) & (daily_return > {r:.4f})")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,g,r) in enumerate(top3,1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 10 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: gap < {g:.4f} AND daily_return > {r:.4f}\n")