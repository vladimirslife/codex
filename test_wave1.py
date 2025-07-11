#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 1 condition sweep.
Тестируем одно простое условие: дневная доходность (close/open - 1) предыдущего дня > threshold.

Стратегия: держим позицию overnight (close_i -> open_{i+1}) только если условие выполняется.

Печатаем 3 лучших результата (Sharpe Ratio) и пишем в history.log.

Используем timeout при запуске извне.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# -------------------- CONSTANTS --------------------
DATA_FILE = "4 - QQQ.csv"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252

# -------------------- LOAD -------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in workspace")

df = pd.read_csv(DATA_FILE)
# Стандартизация
df.rename(columns=lambda c: c.lower(), inplace=True)
df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] >= pd.Timestamp("2006-01-01")].sort_values("Date").reset_index(drop=True)

# overnight return close_i -> open_{i+1}
df["Next_Open"] = df["open"].shift(-1)
df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1

# вычисляем дневную доходность (intraday close/open - 1)
df["daily_return"] = df["close"] / df["open"] - 1

# ---------------------------------------------------
thresholds = np.arange(0.0, 0.021, 0.001)  # 0% .. 2%
results = []

for thr in thresholds:
    # сигнал в день t формируется на данных t (daily_return.shift(0)), но для справедливости используем shift(1)
    signal = (df["daily_return"].shift(1) > thr).astype(int)
    strat_ret = signal * df["next_overnight_return"].shift(1)
    strat_ret.fillna(0, inplace=True)

    excess = strat_ret - DAILY_RF
    if excess.std() == 0:
        sharpe = 0
    else:
        sharpe = (excess.mean() * 252) / (excess.std() * np.sqrt(252))

    trades = int(signal.sum())
    results.append((sharpe, trades, thr))

# sort by Sharpe desc
results.sort(key=lambda x: x[0], reverse=True)

best3 = results[:3]
print("=== Wave 1: top 3 ===")
for i, (sharpe, trades, thr) in enumerate(best3, 1):
    print(f"{i}. Sharpe={sharpe:.4f}, Trades={trades}, Condition: daily_return_prev > {thr:.3f}")

# append to history.log
log_lines = [
    f"{datetime.utcnow().isoformat()} | Wave 1 | rank {i} | Sharpe={s:.4f} | Trades={t} | Condition: daily_return_prev > {thr:.3f}\n"
    for i, (s, t, thr) in enumerate(best3, 1)
]
with open("history.log", "a", encoding="utf-8") as f:
    f.writelines(log_lines)

# Вывод лучшего результата по условиям задачи
best = best3[0]
print("\nЛучший результат Wave 1:")
print(f"Sharpe={best[0]:.4f}, Trades={best[1]}, Condition: daily_return_prev > {best[2]:.3f}")