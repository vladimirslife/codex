#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 8 grid search — прорывы многодневных экстремумов.

Правила (одно условие):
  • open < rolling_min_low(n) * (1+margin)
  • open < rolling_min_close(n) * (1+margin)
  • close < rolling_min_close(n) * (1+margin)
  • open > rolling_max_high(n) * (1-margin)
  • open > rolling_max_close(n) * (1-margin)
  • close > rolling_max_close(n) * (1-margin)

Периоды n ∈ {5, 10, 15, 20, 30}  ; margin ∈ {0, 0.002, 0.005, 0.01}
Все сигналы вычисляются на закрытии T, позиция long до открытия T+1.
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

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= DATE_START].copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_ov_ret"] = df["next_open"] / df["close"] - 1
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

qqq = load_csv(INPUT_FILE)
open_ = qqq["open"]
close = qqq["close"]
high = qqq["high"] if "high" in qqq.columns else close
low  = qqq["low"]  if "low" in qqq.columns else close

# ---------------- RULE GENERATION ---------------------------------------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

WINDOWS = [5, 10, 15, 20, 30]
MARGINS = [0.0, 0.002, 0.005, 0.01]  # 0–1 %

for n in WINDOWS:
    min_low   = low.rolling(window=n, min_periods=n).min()
    min_close = close.rolling(window=n, min_periods=n).min()
    max_high  = high.rolling(window=n, min_periods=n).max()
    max_close = close.rolling(window=n, min_periods=n).max()

    for m in MARGINS:
        # Downside extremes
        thr_low_open   = min_low   * (1 + m)
        thr_low_close  = min_close * (1 + m)
        rules.append((f"open_lt_lowmin{n}_m{int(m*1000):03d}",  lambda df, o=open_, t=thr_low_open: (o < t)))
        rules.append((f"open_lt_clomin{n}_m{int(m*1000):03d}", lambda df, o=open_, t=thr_low_close: (o < t)))
        rules.append((f"close_lt_clomin{n}_m{int(m*1000):03d}", lambda df, c=close, t=thr_low_close: (c < t)))

        # Upside extremes
        thr_high_open  = max_high  * (1 - m)
        thr_high_close = max_close * (1 - m)
        rules.append((f"open_gt_highmax{n}_m{int(m*1000):03d}", lambda df, o=open_, t=thr_high_open: (o > t)))
        rules.append((f"open_gt_comax{n}_m{int(m*1000):03d}",  lambda df, o=open_, t=thr_high_close: (o > t)))
        rules.append((f"close_gt_comax{n}_m{int(m*1000):03d}", lambda df, c=close, t=thr_high_close: (c > t)))

# Ограничиваем ≤1000
rules = rules[:1000]
print(f"Wave 8: generated {len(rules)} extreme-level rules.")

# ---------------- EVALUATION --------------------------------------------------------------------

def evaluate(sig: pd.Series) -> Tuple[float, int]:
    strat = sig.shift(0) * qqq["next_ov_ret"]
    strat = strat.dropna()
    excess = strat - DAILY_RF
    mean_ex = excess.mean() * 252
    std_ex  = excess.std() * np.sqrt(252)
    sharpe  = mean_ex / std_ex if std_ex > 0 else 0.0
    trades  = int(sig.sum())
    return sharpe, trades

results: List[Tuple[str, float, int]] = []
for name, func in rules:
    sig = func(qqq).astype(int)
    sh, tr = evaluate(sig)
    results.append((name, sh, tr))

results.sort(key=lambda x: x[1], reverse=True)

print("Top results (Wave 8):")
for i, (cond, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {cond:35s} | Sharpe: {sh:6.3f} | Trades: {tr}")

# ---------------- HISTORY LOG -------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for cond, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave8 | {cond} | Sharpe={sh:.3f} | Trades={tr}\n")
print("\nTop 3 записаны в history.log")