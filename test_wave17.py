#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 17: одно условие (gap - beta*prev_gap - delta*is_monday) < thr"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

FILE = "4 - QQQ.csv"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252

if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

# load data
df = pd.read_csv(FILE)
df.columns = [c.lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= pd.Timestamp("2006-01-01")].sort_values("date").reset_index(drop=True)

prev_close = df["close"].shift(1)
gap = df["open"] / prev_close - 1
prev_gap = gap.shift(1)

# Monday flag
df["is_mon"] = (df["date"].dt.weekday == 0).astype(int)

next_ov = df["open"].shift(-1) / df["close"] - 1

beta_values = np.arange(-0.24, -0.159, 0.004)     # -0.24 .. -0.16
delta_values = np.arange(-0.002, 0.0021, 0.0004)  # -0.002 .. 0.002
thr_values = np.arange(0.0025, 0.00351, 0.00005)  # 0.0025 .. 0.0035

best = None
results = []
for beta in beta_values:
    for delta in delta_values:
        metric = gap - beta * prev_gap - delta * df["is_mon"]
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
            results.append((sr, trades, beta, delta, thr))
            if best is None or sr > best[0]:
                best = (sr, trades, beta, delta, thr)

# sort top results
results.sort(key=lambda x: x[0], reverse=True)
print("=== Wave 17: top 5 ===")
for i, (sr, tr, beta, delta, thr) in enumerate(results[:5], 1):
    print(f"{i}. Sharpe={sr:.4f}, Trades={tr}, Cond: gap -({beta:.3f})*prev_gap -({delta:.4f})*isMon < {thr:.4f}")

if best:
    sr, tr, beta, delta, thr = best
    with open("history.log", "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.utcnow().isoformat()} | Wave 17 | best | Sharpe={sr:.4f} | Trades={tr} | Condition: gap - {beta:.3f}*prev_gap - {delta:.4f}*isMon < {thr:.4f}\n"
        )
    print("\nЛучший результат Wave 17:")
    print(f"Sharpe={sr:.4f}, Trades={tr}, Condition: gap -({beta:.3f})*prev_gap -({delta:.4f})*isMon < {thr:.4f}")
else:
    print("Нет комбинации с >3000 сделок")