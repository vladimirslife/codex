#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 8: условие относительной силы: (ratio_change_prev > thr)"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

files = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv"}
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252

for p in files.values():
    if not os.path.exists(p):
        raise FileNotFoundError(p)


def load(path):
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.Timestamp("2006-01-01")].sort_values("Date").reset_index(drop=True)
    return df[["Date", "close", "open"]]

qqq = load(files["QQQ"]).rename(columns={"close": "qqq_close", "open": "qqq_open"})
spy = load(files["SPY"]).rename(columns={"close": "spy_close", "open": "spy_open"})

merged = qqq.merge(spy, on="Date", how="inner")

# compute ratio
merged["ratio"] = merged["qqq_close"] / merged["spy_close"]
merged["ratio_change"] = merged["ratio"].pct_change()

# QQQ next overnight
merged["Next_Open"] = merged["qqq_open"].shift(-1)
merged["next_ov"] = merged["Next_Open"] / merged["qqq_close"] - 1

thresholds = np.arange(-0.005, 0.006, 0.0005)
results = []
for thr in thresholds:
    sig = (merged["ratio_change"].shift(1) > thr).astype(int)
    strat = sig * merged["next_ov"]
    strat.fillna(0, inplace=True)
    excess = strat - DAILY_RF
    sr = 0 if excess.std()==0 else (excess.mean()*252)/(excess.std()*np.sqrt(252))
    trades = int(sig.sum())
    results.append((sr, trades, thr))

results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 8 top 3 ===")
for i,(sr,tr,thr) in enumerate(top3,1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: ratio_change_prev > {thr:.4f}")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,thr) in enumerate(top3,1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 8 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: ratio_change_prev > {thr:.4f}\n")