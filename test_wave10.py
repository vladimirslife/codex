#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 10 grid search – TRIX (triple-smoothed EMA rate-of-change) и ADX (Average Directional Index).
Цель: одно условие, Sharpe ≥1.4 и >2500 сделок.
"""

import os
from datetime import datetime
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

INPUT_FILE = os.environ.get("QQQ_CSV", "4 - QQQ.csv")
DATE_START = "2006-01-01"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
TOP_N = 20

# ----------------- DATA --------------------------------------------

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
close = qqq["close"]
high  = qqq["high"] if "high" in qqq.columns else close
low   = qqq["low"]  if "low" in qqq.columns else close

# ----------------- INDICATORS --------------------------------------

def trix(series: pd.Series, n: int) -> pd.Series:
    ema1 = series.ewm(span=n, adjust=False, min_periods=n).mean()
    ema2 = ema1.ewm(span=n, adjust=False, min_periods=n).mean()
    ema3 = ema2.ewm(span=n, adjust=False, min_periods=n).mean()
    return ema3.pct_change()

# ADX helper functions
def adx(high: pd.Series, low: pd.Series, close_s: pd.Series, n: int) -> pd.Series:
    plus_dm = (high.diff()).where((high.diff() > low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((low.diff() > high.diff()) & (low.diff() > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close_s.shift(1)).abs(),
        (low - close_s.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=n, min_periods=n).mean()
    plus_di = 100 * (plus_dm.rolling(n, min_periods=n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n, min_periods=n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.rolling(window=n, min_periods=n).mean()
    return adx_val

# ----------------- RULE GENERATION ---------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# TRIX conditions
TRIX_N = [10, 15, 20, 30]
TRIX_THRESH = [0.0, 0.001, -0.001]  # cross + small bands (~0.1%)
for n in TRIX_N:
    t = trix(close, n)
    for thr in TRIX_THRESH:
        if thr >= 0:
            rules.append((f"trix{n}_gt_{thr:.3f}", lambda df, s=t, th=thr: (s > th)))
        if thr <= 0:
            rules.append((f"trix{n}_lt_{thr:.3f}", lambda df, s=t, th=thr: (s < th)))

# ADX conditions
ADX_N = [10, 14, 20]
ADX_THRESH = [20, 25, 30, 35]
for n in ADX_N:
    adx_series = adx(high, low, close, n)
    for thr in ADX_THRESH:
        rules.append((f"adx{n}_gt_{thr}", lambda df, s=adx_series, th=thr: (s > th)))
        rules.append((f"adx{n}_lt_{thr}", lambda df, s=adx_series, th=thr: (s < th)))

rules = rules[:500]
print(f"Wave 10: generated {len(rules)} TRIX/ADX rules.")

# ----------------- EVALUATION --------------------------------------

def evaluate(sig: pd.Series):
    strat = sig.shift(0) * qqq["next_ov_ret"]
    strat = strat.dropna()
    excess = strat - DAILY_RF
    mean_ex = excess.mean() * 252
    std_ex = excess.std() * np.sqrt(252)
    sharpe = mean_ex / std_ex if std_ex > 0 else 0
    trades = int(sig.sum())
    return sharpe, trades

results = []
for name, func in rules:
    sig = func(qqq).astype(int)
    sh, tr = evaluate(sig)
    results.append((name, sh, tr))

results.sort(key=lambda x: x[1], reverse=True)
print("Top results (Wave 10):")
for i, (name, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {name:20s} | Sharpe: {sh:6.3f} | Trades: {tr}")

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for name, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave10 | {name} | Sharpe={sh:.3f} | Trades={tr}\n")
print("Top 3 записаны в history.log")