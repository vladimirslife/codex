#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 2: условие daily_return_prev < threshold (threshold отрицательный).
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

data_file = "4 - QQQ.csv"
annual_rf = 0.02
daily_rf = annual_rf / 252

if not os.path.exists(data_file):
    raise FileNotFoundError(data_file)

df = pd.read_csv(data_file)

df.rename(columns=lambda c: c.lower(), inplace=True)
df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] >= pd.Timestamp("2006-01-01")].sort_values("Date").reset_index(drop=True)

# prepare returns
df["Next_Open"] = df["open"].shift(-1)
df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1

df["daily_return"] = df["close"] / df["open"] - 1

thresholds = np.arange(0.0, -0.031, -0.001)  # 0 to -3%
results = []
for thr in thresholds:
    sig = (df["daily_return"].shift(1) < thr).astype(int)
    strat = sig * df["next_overnight_return"].shift(1)
    strat.fillna(0, inplace=True)
    excess = strat - daily_rf
    sr = 0 if excess.std() == 0 else (excess.mean()*252)/(excess.std()*np.sqrt(252))
    trades = int(sig.sum())
    results.append((sr, trades, thr))

results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 2 top 3 ===")
for i,(sr,tr,thr) in enumerate(top3,1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: daily_return_prev < {thr:.3f}")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,thr) in enumerate(top3,1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 2 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: daily_return_prev < {thr:.3f}\n")