#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 7: новые однофакторные фильтры
  1. Разница ночных доходностей SPY и QQQ (diff = SPY_ov - QQQ_ov)
     Условие: diff ≤ thr.
  2. Длина последней серии отрицательных ночей SPY (streak_neg ≥ k).
  3. Фильтр по дню недели предыдущей свечи (weekday == w) – одно сравнение.

Цель: Sharpe ≥1.34 при >3800 сделок.
"""

import numpy as np
import pandas as pd
from math import sqrt

annual_rf = 0.02
daily_rf = annual_rf / 252


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df[df["Date"] >= pd.Timestamp("2006-01-01")]
        .sort_values(by=["Date"])  # type: ignore[arg-type]
        .reset_index(drop=True)
    )
    df["Next_Open"] = df["open"].shift(-1)
    df["ov"] = df["Next_Open"] / df["close"] - 1
    return df

qqq = load("4 - QQQ.csv")
spy = load("4 - SPY.csv")

df = qqq[["Date", "ov"]].rename(columns={"ov": "qqq_ov"}).merge(
    spy[["Date", "ov"]].rename(columns={"ov": "spy_ov"}),  # type: ignore[arg-type]
    on="Date",
)

# Helper for performance

def perf(sig: pd.Series):
    strat = sig * df["qqq_ov"].values
    excess = strat - daily_rf
    if strat.std() == 0:
        return 0.0, int(sig.sum())
    sr = excess.mean() * 252 / (excess.std() * sqrt(252))
    return sr, int(sig.sum())

# ------ 1. diff filter ------
print("=== Diff filter SPY_ov - QQQ_ov ≤ thr ===")
diff = (df["spy_ov"].shift(1) - df["qqq_ov"].shift(1))
diff_res = []
for thr in np.arange(-0.002, 0.00501, 0.0001):
    sig = (diff <= thr).astype(int)
    sr, tr = perf(sig)
    diff_res.append((sr, tr, thr))

diff_res.sort(key=lambda x: x[0], reverse=True)
print("Top 10 diff")
print("Sharpe | Trades | thr")
for s, n, thr in diff_res[:10]:
    flag = "*" if (s >= 1.34 and n > 3800) else " "
    print(f"{s:6.3f} | {n:5d} | {thr:6.4f} {flag}")

# ------ 2. streak negative nights ------
print("\n=== Streak negative SPY overnight >= k ===")
neg = (df["spy_ov"].shift(1) < 0).astype(int)
streak_len = neg * (neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1)
# compute consecutive negatives up to prev day
streak_res = []
for k in range(1, 6):
    sig = (streak_len >= k).astype(int)
    sr, tr = perf(sig)
    streak_res.append((sr, tr, k))

streak_res.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | k")
for s, n, k in streak_res:
    flag = "*" if (s >= 1.34 and n > 3800) else " "
    print(f"{s:6.3f} | {n:5d} | {k} {flag}")

# ------ 3. Weekday filter ------
print("\n=== Weekday(prev) == w filter ===")
weekday = df["Date"].dt.weekday.shift(1)
weekday_res = []
for w in range(5):  # 0=Mon ...4=Fri
    sig = (weekday == w).astype(int)
    sr, tr = perf(sig)
    weekday_res.append((sr, tr, w))
weekday_res.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | weekday(prev)")
for s, n, w in weekday_res:
    flag = "*" if (s >= 1.34 and n > 3800) else " "
    print(f"{s:6.3f} | {n:5d} | {w} {flag}")

# Bias check
assert diff.isna().sum() == 1