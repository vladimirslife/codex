#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 9 – Percentile-based и сезонные однофакторные фильтры.

A. Percentile: вычисляем rolling-перцентиль предыдущей ночной доходности SPY
   (rank среди последних N дней). Сигнал, если perc <= p_thr.
   • окна N ∈ {126, 252, 504}
   • p_thr ∈ {0.05 … 0.50}

B. Сезонный фильтр Month ∈ selected_set (одно условие проверяет, входит ли месяц
   предыдущей даты в фиксированный список). Подбираем лучшие наборы месяцев, но
   отчёт выводим для топ-5 встреченных combination sizes 3–6.

Цель: Sharpe ≥ 1.34, Trades > 2500.
"""
import numpy as np
import pandas as pd
from math import sqrt
import itertools

annual_rf = 0.02
daily_rf = annual_rf / 252

# ---------------- загрузка ----------------

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

base = qqq[["Date", "ov"]].merge(
    spy[["Date", "ov"]].rename(columns={"ov": "spy_ov"}),  # type: ignore[arg-type]
    on="Date",
)


def perf(sig: pd.Series):
    strat = sig * base["ov"]
    if strat.std() == 0:
        return 0.0, int(sig.sum())
    sr = (strat - daily_rf).mean() * 252 / (strat.std() * sqrt(252))
    return sr, int(sig.sum())

# ---------------- A. Percentile --------------
print("=== Wave 9 – A. Rolling percentile on SPY_ov_prev ===")
windows = [126, 252, 504]
pthrs = np.arange(0.05, 0.501, 0.02).round(2)
results_pct = []
for win in windows:
    rolling_window = base["spy_ov"].shift(1).rolling(win)
    # Percentile rank: position / window
    # Use rolling.apply with lambda for efficiency? We'll loop simple.
    # We'll compute ranks via expanding sorted indexes using pandas percentileofscore would be slow; approximate with rank.
    ranks = base["spy_ov"].shift(1).rolling(win).apply(lambda s: s.rank(pct=True).iloc[-1] if len(s.dropna()) == win else np.nan, raw=False)
    for p in pthrs:
        sig = (ranks <= p).astype(int)
        sr, tr = perf(sig)
        results_pct.append((sr, tr, f"win={win}, p<={p:.2f}"))

results_pct.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | Param")
for s, n, name in results_pct[:10]:
    flag = "*" if (s >= 1.34 and n > 2500) else " "
    print(f"{s:6.3f} | {n:5d} | {name} {flag}")

best_pct = next((r for r in results_pct if r[0] >= 1.34 and r[1] > 2500), None)
if best_pct:
    print(f"\nGoal achieved with percentile filter: {best_pct}")
else:
    print("\nPercentile filters did not hit target yet.")

# ---------------- B. Month seasonal ----------
print("\n=== Wave 9 – B. Month-set seasonal filter ===")
month_prev = base["Date"].dt.month.shift(1)
months = list(range(1, 13))
month_results = []
for k in range(3, 7):  # size of set
    for combo in itertools.combinations(months, k):
        combo_set = set(combo)
        sig = month_prev.isin(combo_set).astype(int)
        sr, tr = perf(sig)
        month_results.append((sr, tr, combo))

month_results.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | Months")
for s, n, mset in month_results[:10]:
    flag = "*" if (s >= 1.34 and n > 2500) else " "
    print(f"{s:6.3f} | {n:5d} | {mset} {flag}")

best_month = next((r for r in month_results if r[0] >= 1.34 and r[1] > 2500), None)
if best_month:
    print(f"\nGoal met with seasonal month filter: {best_month}")
else:
    print("\nSeasonal month filters did not achieve target.")