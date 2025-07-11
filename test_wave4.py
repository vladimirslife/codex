#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 4 script – исследование «нормализованных» гэпов, прорывов диапазона и
квантильных фильтров.  Цель: найти ОДНО условие с Sharpe ≥ 1.4 и >2500
сделок.

Сигнал формируется на close T, позиции long держатся до open T+1.
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
    return df.dropna().reset_index(drop=True)

qqq = load_csv(INPUT_FILE)
close = qqq["close"]
open_ = qqq["open"]
high = qqq["high"] if "high" in qqq.columns else close
low = qqq["low"] if "low" in qqq.columns else close
close_prev = close.shift(1)
high_prev = high.shift(1)
low_prev = low.shift(1)

# ---------------- HELPERS -----------------------------------------------------------------------

def atr(high_s: pd.Series, low_s: pd.Series, close_s: pd.Series, n: int) -> pd.Series:
    high_low = high_s - low_s
    high_close_prev = (high_s - close_s.shift(1)).abs()
    low_close_prev = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=n).mean()

# Pre-calculate ATRs
atr5 = atr(high, low, close, 5)
atr10 = atr(high, low, close, 10)

# ---------------- RULE GENERATION ---------------------------------------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# 1. Normalised gap (open-close_prev)/ATR_n < -thr
GAP_THR = [0.0, 0.25, 0.5, 0.75, 1.0]  # in ATR units
for n, atr_series in [(5, atr5), (10, atr10)]:
    gap = (open_ - close_prev) / atr_series
    for thr in GAP_THR:
        rules.append((f"gap_norm{n}_lt_-{thr:.2f}", lambda df, g=gap, t=thr: (g < -t)))
        rules.append((f"gap_norm{n}_gt_{thr:.2f}",  lambda df, g=gap, t=thr: (g > t)))

# 2. Open breaks yesterday range
rules.append(("open_lt_prev_low",  lambda df, o=open_, l=low_prev: (o < l)))
rules.append(("open_gt_prev_high", lambda df, o=open_, h=high_prev: (o > h)))

# 3. Range/ATR breakout (today range / ATR_n > thr)
RANGE_THR = [1.0, 1.25, 1.5, 2.0]
for n, atr_series in [(5, atr5), (10, atr10)]:
    range_today = (high - low) / atr_series
    for thr in RANGE_THR:
        rules.append((f"range{n}_gt_{thr:.2f}", lambda df, r=range_today, t=thr: (r > t)))

# 4. Rolling percentile (quantile) of close
ROLL_N = [20, 50, 100]
QUANTS_LOW = [0.1, 0.2, 0.3]
QUANTS_HIGH = [0.7, 0.8, 0.9]
for n in ROLL_N:
    roll = close.rolling(window=n, min_periods=n)
    for q in QUANTS_LOW:
        thr_series = roll.quantile(q)
        rules.append((f"close_lt_q{int(q*100)}_{n}", lambda df, c=close, th=thr_series: (c < th)))
    for q in QUANTS_HIGH:
        thr_series = roll.quantile(q)
        rules.append((f"close_gt_q{int(q*100)}_{n}", lambda df, c=close, th=thr_series: (c > th)))

# Ограничиваем ≤1000
rules = rules[:1000]
print(f"Wave 4: generated {len(rules)} candidate rules.")

# ---------------- EVALUATION --------------------------------------------------------------------

def evaluate(sig: pd.Series) -> Tuple[float, int]:
    strat_r = sig.shift(0) * qqq["next_ov_ret"]
    strat_r = strat_r.dropna()
    excess = strat_r - DAILY_RF
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

print("Top results (Wave 4):")
for i, (cond, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {cond:28s} | Sharpe: {sh:6.3f} | Trades: {tr}")

# ---------------- HISTORY LOG -------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for cond, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave4 | {cond} | Sharpe={sh:.3f} | Trades={tr}\n")
print("\nTop 3 добавлены в history.log")