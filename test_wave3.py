#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 3: фокус на часто возникающих однофакторных правилах — дневная
моментум-свеча, гэпы и волатильные разрывы — чтобы приблизиться к
требованию >2500 сделок и Sharpe ≥ 1.4.

Каждый кандидат — ровно одно логическое выражение, оцениваемое на close T,
с последующим overnight-переходом (long-only).
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
TOP_N = 20

# ---------------- DATA --------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= DATE_START].copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_ov_ret"] = df["next_open"] / df["close"] - 1
    return df.dropna().reset_index(drop=True)

qqq = load(INPUT_FILE)
close = qqq["close"]
open_ = qqq["open"]
close_prev = close.shift(1)

# Basic derived series
candle_green = close > open_
candle_red   = close < open_
close_up     = close > close_prev
close_down   = close < close_prev

daily_ret = close / open_ - 1  # внутридневная свеча, известна к close

# ---------------- RULE GENERATION ---------------------------------------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# 1. Candle colour
rules.append(("candle_green", lambda df, s=candle_green: s))
rules.append(("candle_red",   lambda df, s=candle_red: s))

# 2. Momentum vs close_prev
rules.append(("close_gt_prev", lambda df, s=close_up: s))
rules.append(("close_lt_prev", lambda df, s=close_down: s))

# 3. Gaps (open vs prev close) with thresholds
GAP_THRESH = [0.0, 0.002, 0.005, 0.01]  # 0%, 0.2%, 0.5%, 1%
gap_up_base = open_ / close_prev - 1
for thr in GAP_THRESH:
    rules.append((f"gap_up_gt_{thr:.3f}",   lambda df, g=gap_up_base, t=thr: (g > t)))
    rules.append((f"gap_down_lt_-{thr:.3f}", lambda df, g=gap_up_base, t=thr: (g < -t)))

# 4. Volatility breakout (|daily_ret| > thr)
VOL_THRESH = [0.005, 0.01, 0.015, 0.02]  # 0.5% ... 2%
abs_ret = daily_ret.abs()
for thr in VOL_THRESH:
    rules.append((f"abs_daily_ret_gt_{thr:.3f}", lambda df, a=abs_ret, t=thr: (a > t)))

# 5. Close at extreme of day (high==close / low==close)
high = qqq["high"] if "high" in qqq.columns else close
low  = qqq["low"]  if "low" in qqq.columns else close
rules.append(("close_at_high", lambda df, h=high, c=close: (c >= h)))
rules.append(("close_at_low",  lambda df, l=low,  c=close: (c <= l)))

# 6. Short EMA trend (3-10)
EMA_P = range(3, 11)
for n in EMA_P:
    ema_n = close.ewm(span=n, adjust=False, min_periods=n).mean()
    rules.append((f"close_gt_ema{n}", lambda df, e=ema_n: (df["close"] > e)))
    rules.append((f"close_lt_ema{n}", lambda df, e=ema_n: (df["close"] < e)))

# Ensure limit ≤ 1000
rules = rules[:1000]
print(f"Wave 3: generated {len(rules)} single-condition candidates.")

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
    s = func(qqq).astype(int)
    sh, tr = evaluate(s)
    results.append((name, sh, tr))

results.sort(key=lambda x: x[1], reverse=True)

print("Top results (Wave 3):")
for i, (cond, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {cond:25s} | Sharpe: {sh:6.3f} | Trades: {tr}")

# ---------------- HISTORY LOG -------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for cond, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave3 | {cond} | Sharpe={sh:.3f} | Trades={tr}\n")
print("\nTop 3 добавлены в history.log")