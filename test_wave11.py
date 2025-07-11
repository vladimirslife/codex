#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 11: условие close_prev < smaN_prev"""
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

for n in [5,10,20,50,100]:
    df[f"sma{n}"] = df["close"].rolling(window=n).mean()

df["Next_Open"] = df["open"].shift(-1)
df["next_ov"] = df["Next_Open"] / df["close"] - 1

results=[]
for n in [5,10,20,50,100]:
    sig = (df["close"].shift(0) < df[f"sma{n}"].shift(0)).astype(int)
    strat = sig * df["next_ov"]
    strat.fillna(0, inplace=True)
    excess = strat - DAILY_RF
    if excess.std()==0:
        continue
    sr = (excess.mean()/excess.std())*np.sqrt(252)
    trades = int(sig.sum())
    results.append((sr,trades,n))

results.sort(key=lambda x: x[0],reverse=True)
print("=== Wave 11 top ===")
for sr,tr,n in results:
    print(f"SMA{n}: Sharpe={sr:.4f}, Trades={tr}")

with open("history.log","a",encoding="utf-8") as f:
    for sr,tr,n in results[:3]:
        f.write(f"{datetime.utcnow().isoformat()} | Wave 11 | SMA{n} | Sharpe={sr:.4f} | Trades={tr} | Condition: close < SMA{n}\n")