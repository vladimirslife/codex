#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 16: уточнённое одно условие  abs(gap - beta*prev_gap) < threshold"""
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
df.columns = [c.lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= pd.Timestamp("2006-01-01")].sort_values("date").reset_index(drop=True)

prev_close = df["close"].shift(1)
gap = df["open"] / prev_close - 1
prev_gap = gap.shift(1)
next_ov = df["open"].shift(-1) / df["close"] - 1

beta_values = np.arange(-0.25, -0.149, 0.005)  # [-0.25, -0.15] шаг 0.005
thr_values = np.arange(0.0020, 0.0051, 0.0001)  # 0.0020 .. 0.0050

best = None
results = []
for beta in beta_values:
    metric = gap - beta * prev_gap
    abs_metric = metric.abs()
    for thr in thr_values:
        sig = (abs_metric < thr).astype(int)
        trades = int(sig.sum())
        if trades < 3000:
            continue
        strat = sig * next_ov
        excess = strat - DAILY_RF
        if excess.std() == 0:
            continue
        sr = (excess.mean() / excess.std()) * np.sqrt(252)
        results.append((sr, trades, beta, thr))
        if best is None or sr > best[0]:
            best = (sr, trades, beta, thr)

results.sort(key=lambda x: x[0], reverse=True)

print("=== Wave 16: top 5 ===")
for i, (sr, tr, beta, thr) in enumerate(results[:5], 1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Condition: |gap - ({beta:.3f})*prev_gap| < {thr:.4f}")

if best:
    sr, tr, beta, thr = best
    with open("history.log", "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.utcnow().isoformat()} | Wave 16 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: |gap - {beta:.3f}*prev_gap| < {thr:.4f}\n"
        )
    print("\nЛучший результат Wave 16:")
    print(f"Sharpe={sr:.4f}, Trades={tr}, Condition: |gap - ({beta:.3f})*prev_gap| < {thr:.4f}")
else:
    print("Нет комбинации, удовлетворяющей количеству сделок > 3000")