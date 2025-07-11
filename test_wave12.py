#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 12: условие SPY предыдущий дневной доходности (close/open -1) > thr"""
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
    df["daily_ret"] = df["close"] / df["open"] - 1
    df["next_open"] = df["open"].shift(-1)
    df["next_ov"] = df["next_open"] / df["close"] - 1
    return df

qqq = load(files["QQQ"]).rename(columns={"next_ov": "qqq_next_ov"})
spy = load(files["SPY"]).rename(columns={"daily_ret": "spy_daily_ret"})[["Date", "spy_daily_ret"]]

df = qqq.merge(spy, on="Date", how="left")

thresholds = np.arange(-0.002, 0.005, 0.001)
results = []
for thr in thresholds:
    sig = (df["spy_daily_ret"].shift(1) > thr).astype(int)
    strat = sig * df["qqq_next_ov"]
    strat.fillna(0, inplace=True)
    excess = strat - DAILY_RF
    sr = 0 if excess.std()==0 else (excess.mean()/excess.std())*np.sqrt(252)
    trades = int(sig.sum())
    results.append((sr,trades,thr))

results.sort(key=lambda x: x[0], reverse=True)

top3=results[:3]
print("=== Wave 12 top 3 ===")
for i,(sr,tr,thr) in enumerate(top3,1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: SPY_daily_ret_prev > {thr:.4f}")

with open("history.log","a",encoding="utf-8") as f:
    for i,(sr,tr,thr) in enumerate(top3,1):
        f.write(f"{datetime.utcnow().isoformat()} | Wave 12 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: SPY_daily_ret_prev > {thr:.4f}\n")