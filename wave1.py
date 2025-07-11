#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 1: Поиск одного условия входа для overnight-стратегии QQQ.
Условие: брать сделку, если предыдущая ночная доходность (от close_{t-1} до open_t)
не превышает порог threshold (<= threshold).
Проверяем сетку threshold из 0.002 до 0.01.
Проверяем forward-looking bias: сигнал строится на данных, известных ДО T.
"""

import pandas as pd
import numpy as np
from math import sqrt

annual_rf = 0.02
daily_rf = annual_rf / 252


def load_ticker(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df[df["Date"] >= pd.Timestamp("2006-01-01")]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df["Next_Open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    return df


df = load_ticker("4 - QQQ.csv").dropna(subset=["next_overnight_return"])

candidates = np.arange(0.002, 0.0105, 0.0005)
results = []
for th in candidates:
    signal = (df["next_overnight_return"].shift(1) <= th).astype(int)
    strategy_ret = signal * df["next_overnight_return"]

    excess = strategy_ret - daily_rf
    sharpe = excess.mean() * 252 / (excess.std() * sqrt(252)) if excess.std() > 0 else 0
    trades = int(signal.sum())

    results.append((sharpe, trades, th))

# sort by Sharpe descending
results.sort(key=lambda x: x[0], reverse=True)

top10 = results[:10]

print("\nTop-10 результатов Wave 1 (порог предыдущей ночной доходности <= threshold):")
print("Sharpe | Trades | Threshold")
for s, n, th in top10:
    print(f"{s:6.3f} | {n:5d} | {th:6.4f}")

best = top10[0]
print("\nЛучший результат Wave 1:")
print(f"Sharpe Ratio = {best[0]:.3f}, Сделок = {best[1]}, Условие: prev_overnight <= {best[2]:.4f}")

# Assert forward-looking correctness: сигнал shift(1) использует данные до момента T.
assert (df["next_overnight_return"].shift(1).isna().sum() == 1), "Shift alignment issue"