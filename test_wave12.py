#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 12 grid search – новые однофакторные идеи:
 1. N последовательных красных свечей (close<open) → mean reversion.
 2. Цена выше/ниже rolling median (робустная альтернатива SMA).
 3. Цена закрытия > open (зелёная) для M подряд? (single cond using run-length >= k).
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

# ------------ data --------------

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

Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# --- consecutive down candles --------------------------------------
red = (close < open_).astype(int)
consec = red.groupby((red != red.shift()).cumsum()).cumsum() * red  # running count only on reds
K = [2,3,4,5]
for k in K:
    rules.append((f"consec_red_ge{k}", lambda df, c=consec, k=k: (c >= k)))

# --- consecutive green candles -------------------------------------
green = (close > open_).astype(int)
consec_g = green.groupby((green != green.shift()).cumsum()).cumsum() * green
for k in K:
    rules.append((f"consec_green_ge{k}", lambda df, c=consec_g, k=k: (c >= k)))

# --- rolling median breakout ---------------------------------------
MED_N = [5,10,15,20]
for n in MED_N:
    med = close.rolling(n, min_periods=n).median()
    rules.append((f"close_gt_med{n}", lambda df, c=close, m=med: (c > m)))
    rules.append((f"close_lt_med{n}", lambda df, c=close, m=med: (c < m)))

rules = rules[:300]
print(f"Wave 12: generated {len(rules)} rules (runs & median).")

# evaluation

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
print("Top results (Wave 12):")
for i, (name, sh, tr) in enumerate(results[:TOP_N],1):
    print(f"{i:2d}. {name:25s} | Sharpe: {sh:6.3f} | Trades: {tr}")

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for name, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave12 | {name} | Sharpe={sh:.3f} | Trades={tr}\n")
print("Top 3 записаны в history.log")