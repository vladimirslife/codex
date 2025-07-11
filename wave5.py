#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 5: Дальнейший поиск улучшений однофакторного фильтра.

Эксперимент A (fine): ещё более точная настройка порога по предыдущей ночной доходности SPY.
  • Диапазон 0.0035–0.0045 с шагом 0.00001 (1 bp)

Эксперимент B (idea check): сумма двух предыдущих ночных доходностей SPY ≤ thr.
  • Диапазон 0.004–0.008 с шагом 0.0001
  • Проверяем на всякий случай — держим одно условие.

Выводим top-10 лучших результатов и помечаем те, что Sharpe ≥ 1 и > 3800 сделок.
"""

import numpy as np
import pandas as pd
from math import sqrt

annual_rf = 0.02
daily_rf = annual_rf / 252


# ---------------------- LOAD DATA -----------------------------------

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
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    return df

qqq = load("4 - QQQ.csv")
spy = load("4 - SPY.csv")

base = qqq[["Date", "next_overnight_return"]].merge(
    spy[["Date", "next_overnight_return"]].rename(columns={"next_overnight_return": "spy_ov"}),  # type: ignore[arg-type]
    on="Date",
    how="left",
)

# ---------------------- HELPERS -------------------------------------

def perf(signal: pd.Series) -> tuple[float, int]:
    strat = signal * base["next_overnight_return"]
    excess = strat - daily_rf
    sr = excess.mean() * 252 / (excess.std() * sqrt(252)) if excess.std() > 0 else 0
    return sr, int(signal.sum())

# ---------------------- EXPERIMENT A -----------------------------
print("\n=== Wave 5 — Эксперимент A: Fine threshold SPY prev overnight ===")

a_candidates = np.round(np.arange(0.0035, 0.004501, 0.00001), 5)
a_results = []
for th in a_candidates:
    sig = (base["spy_ov"].shift(1) <= th).astype(int)
    sr, tr = perf(sig)
    a_results.append((sr, tr, th))

a_results.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | Threshold")
for s, n, th in a_results[:10]:
    flag = "*" if (s >= 1 and n > 3800) else " "
    print(f"{s:6.3f} | {n:5d} | {th:7.5f} {flag}")

best_A = next((r for r in a_results if r[1] > 3800 and r[0] >= 1), a_results[0])
print("Лучший A: Sharpe = {:.3f}, Trades = {}, thr = {:.5f}".format(*best_A))

# ---------------------- EXPERIMENT B -----------------------------
print("\n=== Wave 5 — Эксперимент B: Σ2 ночей SPY ≤ thr ===")

base["spy_ov_sum2"] = base["spy_ov"].shift(1).rolling(2).sum()

b_candidates = np.round(np.arange(0.004, 0.0081, 0.0001), 4)
b_results = []
for th in b_candidates:
    sig = (base["spy_ov_sum2"] <= th).astype(int)
    sr, tr = perf(sig)
    b_results.append((sr, tr, th))

b_results.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | ThrΣ2")
for s, n, th in b_results[:10]:
    flag = "*" if (s >= 1 and n > 3800) else " "
    print(f"{s:6.3f} | {n:5d} | {th:7.4f} {flag}")

best_B = next((r for r in b_results if r[1] > 3800 and r[0] >= 1), (0, 0, 0))
if best_B[0] > 0:
    print("Лучший B: Sharpe = {:.3f}, Trades = {}, thr = {:.4f}".format(*best_B))
else:
    print("Эксперимент B не нашёл фильтра, удовлетворяющего ограничениям.")

# ---------------------- BIAS CHECK -------------------------------
assert base["spy_ov"].shift(1).isna().sum() == 1