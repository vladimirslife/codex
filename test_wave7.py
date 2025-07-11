#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 7 grid search – кумулятивное изменение цены и волатильные всплески.

Однофакторные правила:
 1. N-дневное накопленное изменение цены (close_t / close_{t-n} - 1) < -thr  (падение)
 2. То же > +thr                            (рост)
 3. Rolling std(N) > thr                    (волатильность)

Цель: Sharpe ≥ 1.4, сделок > 2500.
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
TOP_N = 25

# ---------------- DATA --------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= DATE_START].copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_ov_ret"] = df["next_open"] / df["close"] - 1
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

qqq = load(INPUT_FILE)
close = qqq["close"]

# ---------------- RULE GENERATION ---------------------------------------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# 1. N-day cumulative pct change
N_PERIODS = range(2, 11)  # 2…10 дней
THR_DOWN = [0.01, 0.02, 0.03, 0.04]  # 1–4 % падение
THR_UP   = [0.01, 0.02, 0.03, 0.04]  # рост

for n in N_PERIODS:
    cum_change = close / close.shift(n) - 1
    for thr in THR_DOWN:
        rules.append((f"cum{n}_lt_-{int(thr*100)}bp", lambda df, s=cum_change, t=thr: (s < -t)))
    for thr in THR_UP:
        rules.append((f"cum{n}_gt_{int(thr*100)}bp", lambda df, s=cum_change, t=thr: (s > t)))

# 2. Rolling volatility breakout (std of daily pct change)
RET = close.pct_change()
VOL_PERIODS = [5, 10, 15, 20]
VOL_THR = [0.01, 0.015, 0.02, 0.025]  # 1–2.5 % daily std
for n in VOL_PERIODS:
    roll_std = RET.rolling(window=n, min_periods=n).std()
    for thr in VOL_THR:
        rules.append((f"std{n}_gt_{int(thr*1000)}bp", lambda df, s=roll_std, t=thr: (s > t)))

# Limit ≤1000
rules = rules[:1000]
print(f"Wave 7: generated {len(rules)} cumulative/vol rules.")

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

print("Top results (Wave 7):")
for i, (cond, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {cond:25s} | Sharpe: {sh:6.3f} | Trades: {tr}")

# ---------------- HISTORY LOG -------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for cond, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave7 | {cond} | Sharpe={sh:.3f} | Trades={tr}\n")
print("\nTop 3 записаны в history.log")