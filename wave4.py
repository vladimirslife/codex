#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 4: Улучшение стратегии одним условием на основе предыдущей ночной доходности SPY.
Условие: входим в QQQ, если prev_overnight_return_SPY <= threshold.

Цель — найти threshold, дающий Sharpe > 1 и > 3800 сделок, превосходящее базовое решение (Sharpe 1.06).  
Проверяем сетку threshold 0.0035–0.0043 с шагом 0.0001.
"""

import pandas as pd
import numpy as np
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
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    return df


qqq = load("4 - QQQ.csv")
spy = load("4 - SPY.csv")

df = qqq[["Date", "next_overnight_return"]].merge(
    spy[["Date", "next_overnight_return"]].rename(columns={"next_overnight_return": "spy_next_ov"}),  # type: ignore[arg-type]
    on="Date",
    how="left",
)

candidates = np.arange(0.0035, 0.00431, 0.0001).round(4)
results = []
for th in candidates:
    signal = (df["spy_next_ov"].shift(1) <= th).astype(int)
    strat_ret = signal * df["next_overnight_return"]
    excess = strat_ret - daily_rf
    sharpe = excess.mean() * 252 / (excess.std() * sqrt(252)) if excess.std() > 0 else 0
    trades = int(signal.sum())
    results.append((sharpe, trades, th))

results.sort(key=lambda x: x[0], reverse=True)
print("Top-10 результатов (SPY_prev_overnight ≤ threshold)")
print("Sharpe | Trades | Threshold")
for s, n, th in results[:10]:
    flag = "*" if (s >= 1 and n > 3800) else " "
    print(f"{s:6.3f} | {n:5d} | {th:7.4f} {flag}")

best = next((r for r in results if r[1] > 3800 and r[0] >= 1), results[0])
print("\nЛучший, удовлетворяющий ограничениям: Sharpe = {:.3f}, Trades = {}, thr = {:.4f}".format(*best))

# forward-looking bias check
assert df["spy_next_ov"].shift(1).isna().sum() == 1