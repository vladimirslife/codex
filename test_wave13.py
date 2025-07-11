#!/usr/bin/env python3
"""
Wave 13 grid search – Percent-B (Bollinger %) и диапазон True Range в процентилях.
Одно условие, ≤200 правил.
"""

import os
from datetime import datetime
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

FILE = os.environ.get("QQQ_CSV", "4 - QQQ.csv")
DATE_START = "2006-01-01"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
TOP_N = 20


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= DATE_START].copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_ov_ret"] = df["next_open"] / df["close"] - 1
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

qqq = load(FILE)
close = qqq["close"]
open_ = qqq["open"]
high = qqq["high"] if "high" in qqq else close
low  = qqq["low"]  if "low" in qqq else close

Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# --- Percent-B -----------------------------------------------------
BB_N = [10, 20]
DEV = 2
PB_THRESH = [0.0, 0.05, 0.1, 0.9, 0.95, 1.0]
for n in BB_N:
    sma_n = close.rolling(n, min_periods=n).mean()
    std_n = close.rolling(n, min_periods=n).std()
    upper = sma_n + DEV * std_n
    lower = sma_n - DEV * std_n
    pb = (close - lower) / (upper - lower)
    for thr in PB_THRESH:
        if thr < 0.5:
            rules.append((f"pb{n}_lt_{thr}", lambda df, p=pb, t=thr: (p < t)))
        else:
            rules.append((f"pb{n}_gt_{thr}", lambda df, p=pb, t=thr: (p > t)))

# --- True Range percentile ----------------------------------------
tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
TR_WINDOW = 20
tr_pct = tr.rolling(TR_WINDOW, min_periods=TR_WINDOW).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
PCT_THRESH = [0.8, 0.9]
for thr in PCT_THRESH:
    rules.append((f"tr_pct_gt_{thr}", lambda df, p=tr_pct, t=thr: (p > t)))

rules = rules[:200]
print(f"Wave 13: generated {len(rules)} rules (Percent-B & TR percentile).")

# Evaluation

def evaluate(sig):
    strat = sig.shift(0) * qqq["next_ov_ret"]
    strat = strat.dropna()
    excess = strat - DAILY_RF
    mean_ex = excess.mean() * 252
    std_ex = excess.std() * np.sqrt(252)
    sharpe = mean_ex / std_ex if std_ex>0 else 0
    trades = int(sig.sum())
    return sharpe, trades

results=[]
for name,func in rules:
    sig=func(qqq).astype(int)
    sh,tr=evaluate(sig)
    results.append((name,sh,tr))
results.sort(key=lambda x:x[1],reverse=True)
print("Top results (Wave 13):")
for i,(n,s,t) in enumerate(results[:TOP_N],1):
    print(f"{i:2d}. {n:20s} | Sharpe: {s:6.3f} | Trades: {t}")

now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log","a") as f:
    for n,s,t in results[:3]:
        f.write(f"[{now}] Wave13 | {n} | Sharpe={s:.3f} | Trades={t}\n")
print("Top 3 записаны в history.log")