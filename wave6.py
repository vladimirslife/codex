#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 6: Нелинейные однофакторные фильтры

1. Z-score предыдущей ночной доходности SPY: z = (ov - mean)/std, где mean/std – rolling.
   Условие: z ≤ z_thr (одно числовое сравнение).
   Проверяем окна [20, 60, 120] и z_thr ∈ [0.0, 1.5] шаг 0.05.

2. Отношение SPY_prev_overnight / ATR, где ATR – rolling True Range (high-low
   approximation) c окнами [14, 20, 50].  Условие: ratio ≤ thr, thr ∈ [0.25, 1] шаг 0.05.

3. Ratio = SPY_prev_overnight / day_range_prev (high-low)/close.  Условие: ratio ≤ thr,
   thr ∈ [-0.5, 0.5] шаг 0.05.

Цель: Sharpe ≥ 1.34 и > 3800 сделок.
Все проверки для сигнала используют shift(1).
"""

import numpy as np
import pandas as pd
from math import sqrt

annual_rf = 0.02
daily_rf = annual_rf / 252


# ---------------- Data load ----------------

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
    spy[["Date", "next_overnight_return", "high", "low", "close"]].rename(columns={"next_overnight_return": "spy_ov"}),  # type: ignore[arg-type]
    on="Date",
    how="left",
)

# ---------------- helper -----------------

def performance(signal: pd.Series) -> tuple[float, int]:
    strat = signal * base["next_overnight_return"]
    excess = strat - daily_rf
    if strat.std() == 0:
        return 0.0, int(signal.sum())
    sr = excess.mean() * 252 / (excess.std() * sqrt(252))
    return sr, int(signal.sum())


def evaluate_and_log(results: list[tuple[float, int, str]], label: str):
    results.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTop-10 результатов для {label}")
    print("Sharpe | Trades | Param")
    for s, n, name in results[:10]:
        flag = "*" if (s >= 1.34 and n > 3800) else " "
        print(f"{s:6.3f} | {n:5d} | {name} {flag}")
    best = next((r for r in results if r[0] >= 1.34 and r[1] > 3800), None)
    if best:
        print(f"\nДостигнута цель! Лучший фильтр: {best[2]}, Sharpe = {best[0]:.3f}, Trades = {best[1]}")
    else:
        print("\nЦель Sharpe ≥ 1.34 пока не достигнута для данного набора.")

# ------------- 1. Z-score ----------------
print("=== 1. Z-score SPY_prev_overnight ===")
z_results: list[tuple[float, int, str]] = []
for win in [20, 60, 120]:
    rolling_mean = base["spy_ov"].shift(1).rolling(win).mean()
    rolling_std = base["spy_ov"].shift(1).rolling(win).std(ddof=0)
    zscore = (base["spy_ov"].shift(1) - rolling_mean) / rolling_std
    for thr in np.arange(0.0, 1.501, 0.05):
        sig = (zscore <= thr).astype(int)
        sr, tr = performance(sig)
        z_results.append((sr, tr, f"win={win}, thr={thr:.2f}"))

evaluate_and_log(z_results, "Z-score")

# ------------- 2. Ratio to ATR ----------------
print("\n=== 2. Ratio SPY_prev_overnight / ATR ===")
atr_results: list[tuple[float, int, str]] = []
# True Range approximation using high-low
hl_range = base["high"] - base["low"]
for win in [14, 20, 50]:
    atr = hl_range.shift(1).rolling(win).mean()
    ratio = base["spy_ov"].shift(1) / atr
    for thr in np.arange(0.25, 1.001, 0.05):
        sig = (ratio <= thr).astype(int)
        sr, tr = performance(sig)
        atr_results.append((sr, tr, f"win={win}, thr={thr:.2f}"))

evaluate_and_log(atr_results, "Overnight/ATR")

# ------------- 3. Ratio to day range ----------------
print("\n=== 3. Ratio SPY_prev_overnight / day_range_prev ===")
ratio_results: list[tuple[float, int, str]] = []
spy_day_range = hl_range / base["close"]  # percentage range
ratio2 = base["spy_ov"].shift(1) / spy_day_range.shift(1)
for thr in np.arange(-0.5, 0.501, 0.05):
    sig = (ratio2 <= thr).astype(int)
    sr, tr = performance(sig)
    ratio_results.append((sr, tr, f"thr={thr:.2f}"))

evaluate_and_log(ratio_results, "Overnight/DayRange")

# ---------------- bias check ----------------
assert base["spy_ov"].shift(1).isna().sum() == 1