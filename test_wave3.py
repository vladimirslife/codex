#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 3: условие SPY_next_overnight_prev > threshold
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252

files = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv"}
for p in files.values():
    if not os.path.exists(p):
        raise FileNotFoundError(p)

# load function
def load(path):
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.Timestamp("2006-01-01")].sort_values("Date").reset_index(drop=True)
    df["Next_Open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    return df[["Date", "open", "close", "next_overnight_return"]]

qqq = load(files["QQQ"]).rename(columns={"next_overnight_return": "QQQ_next_ov"})
spy = load(files["SPY"]).rename(columns={"next_overnight_return": "SPY_next_ov"})

# merge
common = qqq.merge(spy[["Date", "SPY_next_ov"]], on="Date", how="left")

thresholds = np.arange(-0.01, 0.011, 0.001)  # -1% .. +1%
results = []
for thr in thresholds:
    sig = (common["SPY_next_ov"].shift(1) > thr).astype(int)
    strat = sig * common["QQQ_next_ov"]
    strat.fillna(0, inplace=True)
    excess = strat - DAILY_RF
    sr = 0 if excess.std() == 0 else (excess.mean()*252)/(excess.std()*np.sqrt(252))
    trades = int(sig.sum())
    results.append((sr, trades, thr))

results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 3 top 3 ===")
for i,(sr,tr,thr) in enumerate(top3,1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: SPY_next_overnight_prev > {thr:.3f}")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,thr) in enumerate(top3,1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 3 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: SPY_next_ov_prev > {thr:.3f}\n")