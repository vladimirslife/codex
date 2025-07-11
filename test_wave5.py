#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 5 grid search.

• Refines normalised negative gap thresholds (by ATR).
• Adds rolling max/min breakout and near-extreme conditions with small margins.
Goal: find a SINGLE entry rule with Sharpe ≥ 1.4 and > 2500 trades.
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
TOP_N = 30

# ---------------- DATA --------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= DATE_START].copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_ov_ret"] = df["next_open"] / df["close"] - 1
    return df.dropna().reset_index(drop=True)

qqq = load_data(INPUT_FILE)
open_ = qqq["open"]
close = qqq["close"]
high = qqq["high"] if "high" in qqq.columns else close
low = qqq["low"] if "low" in qqq.columns else close
close_prev = close.shift(1)

# ---------------- ATR helper --------------------------------------------------------------------

def atr(high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([
        high_s - low_s,
        (high_s - close_s.shift(1)).abs(),
        (low_s - close_s.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=n).mean()

atr5 = atr(high, low, close, 5)
atr10 = atr(high, low, close, 10)

# ---------------- RULE GENERATION ---------------------------------------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# 1. Normalised negative gap thresholds (ATR5 & ATR10)
THRS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
for n_label, atr_series in [(5, atr5), (10, atr10)]:
    norm_gap = (open_ - close_prev) / atr_series
    for thr in THRS:
        rules.append((f"gap_norm{n_label}_lt_-{thr:.2f}", lambda df, g=norm_gap, t=thr: (g < -t)))

# 2. Rolling max breakout / near-high conditions
ROLL_N = [20, 30, 40, 50, 60]
MARGINS = [0.0, 0.005, 0.01, 0.02]  # 0%, 0.5%, 1%, 2%
for n in ROLL_N:
    roll_max = close.rolling(window=n, min_periods=n).max()
    roll_min = close.rolling(window=n, min_periods=n).min()
    for m in MARGINS:
        factor_high = 1 - m  # close > max * (1-m)
        factor_low = 1 + m   # close < min * (1+m)
        rules.append((f"close_gt_max{n}_m{int(m*1000):03d}", lambda df, c=close, thr=roll_max*factor_high: (c > thr)))
        rules.append((f"close_lt_min{n}_m{int(m*1000):03d}", lambda df, c=close, thr=roll_min*factor_low: (c < thr)))

# Ensure we do not exceed 1000
rules = rules[:1000]
print(f"Wave 5: generated {len(rules)} single-condition rules.")

# ---------------- EVALUATION --------------------------------------------------------------------

def evaluate(sig: pd.Series) -> Tuple[float, int]:
    strat_ret = sig.shift(0) * qqq["next_ov_ret"]
    strat_ret = strat_ret.dropna()
    excess = strat_ret - DAILY_RF
    mean_exc = excess.mean() * 252
    std_exc = excess.std() * np.sqrt(252)
    sharpe = mean_exc / std_exc if std_exc > 0 else 0.0
    trades = int(sig.sum())
    return sharpe, trades

results: List[Tuple[str, float, int]] = []
for name, func in rules:
    sig = func(qqq).astype(int)
    sh, tr = evaluate(sig)
    results.append((name, sh, tr))

results.sort(key=lambda x: x[1], reverse=True)

print("Top results (Wave 5):")
for i, (cond, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {cond:35s} | Sharpe: {sh:6.3f} | Trades: {tr}")

# ---------------- HISTORY LOG -------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for cond, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave5 | {cond} | Sharpe={sh:.3f} | Trades={tr}\n")
print("\nTop 3 записаны в history.log")