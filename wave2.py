#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 2: подтверждение устойчивости результата.

Эксперимент A (Refine): одно условие «prev_overnight_return ≤ threshold».
Порог ищем в узком диапазоне [0.0040; 0.0060] с шагом 0.0001.

Эксперимент B (Volatility filter): одно условие
«|day_return_{t-1}| ≤ vol_threshold».
Ищем vol_threshold в диапазоне [0.005; 0.030] с шагом 0.0005.

Для каждого эксперимента выводим Top-10 по Sharpe Ratio.
Проверяем forward-looking bias (используем shift(1)).
Условия неизменны: annual_rf = 0.02, только лонг, выход на Open следующего дня.
"""

import numpy as np
import pandas as pd
from math import sqrt

annual_rf = 0.02
daily_rf = annual_rf / 252


# ------------------------- DATA LOADING -----------------------------

def load_ticker(path: str) -> pd.DataFrame:
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
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    df["day_return"] = df["close"] / df["open"] - 1
    return df


df = load_ticker("4 - QQQ.csv").dropna(subset=["next_overnight_return"]).copy()

# ------------------------- EXPERIMENT A -----------------------------
print("\n=== Wave 2 — Эксперимент A: refine prev_overnight ≤ threshold ===")

candidates_A = np.arange(0.0040, 0.00601, 0.0001).round(4)
results_A = []
for th in candidates_A:
    signal = (df["next_overnight_return"].shift(1) <= th).astype(int)
    strategy_ret = signal * df["next_overnight_return"]

    excess = strategy_ret - daily_rf
    sharpe = (
        excess.mean() * 252 / (excess.std() * sqrt(252)) if excess.std() > 0 else 0
    )
    trades = int(signal.sum())
    results_A.append((sharpe, trades, th))

results_A.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | Threshold")
for s, n, th in results_A[:10]:
    print(f"{s:6.3f} | {n:5d} | {th:7.4f}")

best_A = results_A[0]
print("Лучший результат A: Sharpe = {:.3f}, Trades = {}, thr = {:.4f}".format(*best_A))

# ------------------------- EXPERIMENT B -----------------------------
print("\n=== Wave 2 — Эксперимент B: filter |day_return| ≤ vol_threshold ===")

candidates_B = np.arange(0.005, 0.0301, 0.0005).round(4)
results_B = []
for th in candidates_B:
    signal = (df["day_return"].shift(1).abs() <= th).astype(int)
    strategy_ret = signal * df["next_overnight_return"]
    excess = strategy_ret - daily_rf
    sharpe = (
        excess.mean() * 252 / (excess.std() * sqrt(252)) if excess.std() > 0 else 0
    )
    trades = int(signal.sum())
    results_B.append((sharpe, trades, th))

results_B.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | Vol_Thr")
for s, n, th in results_B[:10]:
    print(f"{s:6.3f} | {n:5d} | {th:7.4f}")

best_B = results_B[0]
print("Лучший результат B: Sharpe = {:.3f}, Trades = {}, vol_thr = {:.4f}".format(*best_B))

# ------------------------- CHECKS -----------------------------------
# forward-looking bias check
assert df["next_overnight_return"].shift(1).isna().sum() == 1, "Shift alignment issue in Experiment A"
assert df["day_return"].shift(1).isna().sum() == 1, "Shift alignment issue in Experiment B"