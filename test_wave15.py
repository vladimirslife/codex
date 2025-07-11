#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 15: одно условие (gap - beta*prev_gap) < threshold"""
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
# сегодняшнее gap
gap = df["open"] / prev_close - 1
prev_gap = gap.shift(1)  # gap предыдущего дня

next_ov = df["open"].shift(-1) / df["close"] - 1

beta_values = np.arange(-1.0, 1.05, 0.05)  # -1 .. 1
thr_values = np.arange(-0.02, 0.0205, 0.0005)  # -2% .. 2%

results = []
for beta in beta_values:
    metric = gap - beta * prev_gap
    for thr in thr_values:
        sig = (metric < thr).astype(int)
        trades = int(sig.sum())
        if trades < 3000:
            continue
        strat = sig * next_ov
        excess = strat - DAILY_RF
        if excess.std() == 0:
            continue
        sr = (excess.mean() / excess.std()) * np.sqrt(252)
        results.append((sr, trades, beta, thr))

results.sort(key=lambda x: x[0], reverse=True)

print("=== Wave 15: top 3 ===")
for i, (sr, tr, beta, thr) in enumerate(results[:3], 1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Condition: (gap - {beta:.2f}*prev_gap) < {thr:.4f}")

with open("history.log", "a", encoding="utf-8") as f:
    for i, (sr, tr, beta, thr) in enumerate(results[:3], 1):
        f.write(
            f"{datetime.utcnow().isoformat()} | Wave 15 | rank {i} | Sharpe={sr:.4f} | Trades={tr} | Condition: (gap - {beta:.2f}*prev_gap) < {thr:.4f}\n"
        )

# вывод лучшего результата
best = results[0] if results else None
if best:
    sr, tr, beta, thr = best
    print("\nЛучший результат Wave 15:")
    print(f"Sharpe={sr:.4f}, Trades={tr}, Condition: (gap - {beta:.2f}*prev_gap) < {thr:.4f}")
else:
    print("Нет комбинации с >3000 сделок")