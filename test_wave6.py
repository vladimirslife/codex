#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 6 grid search – календарные закономерности.

Проверяются однофакторные условия вида:
  • День недели (weekday == k)
  • День месяца (day == d)
  • Диапазон дня месяца (day <= d, day >= d)
  • Месяц (month == m)

Цель – найти условие с Sharpe ≥ 1.4 и > 2500 сделок.
"""

import os
from datetime import datetime
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

# ---------------- CONFIG ------------------------------------------------------------------------
INPUT_FILE = os.environ.get("QQQ_CSV", "4 - QQQ.csv")
DATE_START = "2006-01-01"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
TOP_N = 20

# ---------------- DATA --------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= DATE_START].copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_ov_ret"] = df["next_open"] / df["close"] - 1
    df.dropna(inplace=True)
    # calendar columns
    df["weekday"] = df["date"].dt.weekday  # Monday 0
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    return df.reset_index(drop=True)

qqq = load(INPUT_FILE)

# ---------------- RULE GENERATION ---------------------------------------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# 1. Weekday rules (exact)
for wd in range(5):  # 0=Mon .. 4=Fri
    rules.append((f"weekday_eq_{wd}", lambda df, w=wd: (df["weekday"] == w)))

# 1b. Weekday range (weekday<=k or >=k)
for k in range(1,5):
    rules.append((f"weekday_le_{k}", lambda df, k=k: (df["weekday"] <= k)))
    rules.append((f"weekday_ge_{k}", lambda df, k=k: (df["weekday"] >= k)))

# 2. Day-of-month exact (1..31)
for d in range(1,32):
    rules.append((f"day_eq_{d}", lambda df, d=d: (df["day"] == d)))

# 2b. Day-of-month thresholds
for thr in [3,5,10,15,20,25,28]:
    rules.append((f"day_le_{thr}", lambda df, t=thr: (df["day"] <= t)))
    rules.append((f"day_ge_{thr}", lambda df, t=thr: (df["day"] >= t)))

# 3. Month exact (1..12)
for m in range(1,13):
    rules.append((f"month_eq_{m}", lambda df, m=m: (df["month"] == m)))

# Limit to ≤1000
rules = rules[:1000]
print(f"Wave 6: generated {len(rules)} calendar-based rules.")

# ---------------- EVALUATION --------------------------------------------------------------------

def evaluate(sig: pd.Series) -> Tuple[float, int]:
    strat_ret = sig.shift(0) * qqq["next_ov_ret"]
    strat_ret = strat_ret.dropna()
    excess = strat_ret - DAILY_RF
    mean_ex = excess.mean() * 252
    std_ex = excess.std() * np.sqrt(252)
    sharpe = mean_ex / std_ex if std_ex > 0 else 0.0
    trades = int(sig.sum())
    return sharpe, trades

results: List[Tuple[str, float, int]] = []
for name, func in rules:
    sig = func(qqq).astype(int)
    sh, tr = evaluate(sig)
    results.append((name, sh, tr))

results.sort(key=lambda x: x[1], reverse=True)

print("Top results (Wave 6):")
for i, (cond, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {cond:25s} | Sharpe: {sh:6.3f} | Trades: {tr}")

# ---------------- HISTORY LOG -------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for cond, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave6 | {cond} | Sharpe={sh:.3f} | Trades={tr}\n")
print("\nTop 3 записаны в history.log")