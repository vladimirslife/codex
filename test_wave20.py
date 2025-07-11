#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 20: Условие (gap - beta*prev_gap - delta*isMon) / maGapN_prev < thr
Одна логическая проверка, где maGapN_prev — среднее |gap| за последние N дней (shifted 1).
"""
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
# весь доступный период после 2006
df = df[df["date"] >= pd.Timestamp("2006-01-01")].sort_values("date").reset_index(drop=True)

# базовые признаки
df["is_mon"] = (df["date"].dt.weekday == 0).astype(int)
prev_close = df["close"].shift(1)
gap = df["open"] / prev_close - 1
prev_gap = gap.shift(1)
next_ov = df["open"].shift(-1) / df["close"] - 1

# параметрические сетки
beta_vals = np.arange(-0.25, -0.149, 0.01)          # 11 значений
delta_vals = np.arange(-0.0015, 0.0011, 0.0005)      # 6 значений
thr_vals = np.arange(0.002, 0.0105, 0.0005)          # 17 значений
n_vals = [5, 10, 15, 20]                             # 4 значения

best = None
for N in n_vals:
    ma_gap = gap.abs().rolling(N).mean().shift(1)
    # избегаем деления на 0
    ma_gap = ma_gap.replace(0, np.nan)
    for beta in beta_vals:
        metric_base = gap - beta * prev_gap  # вектор
        for delta in delta_vals:
            metric = metric_base - delta * df["is_mon"]
            norm_metric = metric / ma_gap
            for thr in thr_vals:
                sig = (norm_metric < thr).astype(int)
                trades = int(sig.sum())
                if trades < 3000:
                    continue
                strat = sig * next_ov
                excess = strat - DAILY_RF
                std = excess.std()
                if std == 0:
                    continue
                sr = (excess.mean() / std) * np.sqrt(252)
                if best is None or sr > best[0]:
                    best = (sr, trades, beta, delta, thr, N)

if best:
    sr, trades, beta, delta, thr, N = best
    print(f"Wave20 BEST Sharpe={sr:.4f}, Trades={trades}, Cond: (gap -({beta:.3f})*prev_gap -({delta:.4f})*isMon)/MA|gap|_{N} < {thr:.4f}")
    with open("history.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | Wave 20 | best | Sharpe={sr:.4f} | Trades={trades} | Condition: (gap - {beta:.3f}*prev_gap - {delta:.4f}*isMon)/MAabsGap{N} < {thr:.4f}\n")
else:
    print("Wave20: не найдено комбинаций с >3000 сделок")