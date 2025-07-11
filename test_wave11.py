#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 11 grid search – новые индикаторы: MACD-гистограмма, Z-score цены, DPO, Heikin-Ashi свечи.
Один фильтр, ≤ 300 правил.
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

# ---------------- data -----------------

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
open_ = qqq["open"]
high = qqq["high"] if "high" in qqq.columns else close
low  = qqq["low"]  if "low" in qqq.columns else close

Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# -------- MACD histogram --------
FAST, SLOW, SIGNAL = 12, 26, 9
ema_fast = close.ewm(span=FAST, adjust=False, min_periods=SLOW).mean()
ema_slow = close.ewm(span=SLOW, adjust=False, min_periods=SLOW).mean()
macd = ema_fast - ema_slow
macd_signal = macd.ewm(span=SIGNAL, adjust=False, min_periods=SIGNAL).mean()
hist = macd - macd_signal
rules.append(("macd_hist_gt0", lambda df, h=hist: (h > 0)))
rules.append(("macd_hist_lt0", lambda df, h=hist: (h < 0)))

# -------- Z-score of price vs SMA --------
Z_N = [20, 50]
Z_THR = [1.0, 1.5, 2.0]
for n in Z_N:
    sma_n = close.rolling(n, min_periods=n).mean()
    std_n = close.rolling(n, min_periods=n).std()
    z = (close - sma_n) / std_n
    for thr in Z_THR:
        rules.append((f"z{n}_gt_{thr}", lambda df, z=z, t=thr: (z > t)))
        rules.append((f"z{n}_lt_-{thr}", lambda df, z=z, t=thr: (z < -t)))

# -------- DPO (detrended price oscillator) --------
DPO_N = [20, 30]
for n in DPO_N:
    sma_n = close.rolling(n, min_periods=n).mean()
    dpo = close - sma_n
    rules.append((f"dpo{n}_gt0", lambda df, d=dpo: (d > 0)))
    rules.append((f"dpo{n}_lt0", lambda df, d=dpo: (d < 0)))

# -------- Heikin-Ashi candle color --------
ha_close = (open_ + high + low + close) / 4
ha_open = (open_.shift(1) + close.shift(1)) / 2
rules.append(("ha_green", lambda df, o=ha_open, c=ha_close: (c > o)))
rules.append(("ha_red",   lambda df, o=ha_open, c=ha_close: (c < o)))

# cap 300
rules = rules[:300]
print(f"Wave 11: generated {len(rules)} rules (MACD, Z, DPO, HA).")

# ---------- evaluation ------------
DAILY_RF = ANNUAL_RF / 252

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
    s = func(qqq).astype(int)
    sh, tr = evaluate(s)
    results.append((name, sh, tr))

results.sort(key=lambda x: x[1], reverse=True)
print("Top results (Wave 11):")
for i, (name, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {name:20s} | Sharpe: {sh:6.3f} | Trades: {tr}")

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for name, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave11 | {name} | Sharpe={sh:.3f} | Trades={tr}\n")
print("Top 3 записаны в history.log")