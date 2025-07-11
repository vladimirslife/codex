#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 9 grid search – новые индикаторы: Стохастик, Keltner, PPO, Дончианский канал.

Все правила однофакторные. Кол-во ≤ 500.
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

# --- Stochastic %K --------------------------------------------------
STO_N = [5, 9, 14]
STO_THRESH = [20, 30, 70, 80]
for n in STO_N:
    highest_high = high.rolling(n, min_periods=n).max()
    lowest_low = low.rolling(n, min_periods=n).min()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    for thr in STO_THRESH:
        rules.append((f"sto{k.name if False else ''}{n}_lt_{thr}", lambda df, s=k, t=thr: (s < t)))
        rules.append((f"sto{n}_gt_{thr}", lambda df, s=k, t=thr: (s > t)))

# --- Keltner Channel ----------------------------------------------
KC_N = [10, 20]
KC_ATR_MUL = [1.5, 2.0]

def atr(n: int) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

for n in KC_N:
    ema_c = close.ewm(span=n, adjust=False, min_periods=n).mean()
    atr_n = atr(n)
    for mult in KC_ATR_MUL:
        upper = ema_c + mult * atr_n
        lower = ema_c - mult * atr_n
        rules.append((f"close_lt_kclow_{n}_{mult}", lambda df, c=close, l=lower: (c < l)))
        rules.append((f"close_gt_kcupper_{n}_{mult}", lambda df, c=close, u=upper: (c > u)))

# --- PPO (Percentage Price Oscillator) -----------------------------
PPO_FAST = [12]
PPO_SLOW = [26]
PPO_SIGNAL = 9
for fast in PPO_FAST:
    for slow in PPO_SLOW:
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        ppo = 100 * (ema_fast - ema_slow) / ema_slow
        signal = ppo.ewm(span=PPO_SIGNAL, adjust=False, min_periods=PPO_SIGNAL).mean()
        rules.append(("ppo_gt_signal", lambda df, p=ppo, s=signal: (p > s)))
        rules.append(("ppo_lt_signal", lambda df, p=ppo, s=signal: (p < s)))

# --- Donchian channel breakout -------------------------------------
DC_N = [5, 10, 20]
for n in DC_N:
    dc_high = high.rolling(n, min_periods=n).max()
    dc_low = low.rolling(n, min_periods=n).min()
    rules.append((f"close_gt_dchigh{n}", lambda df, c=close, h=dc_high: (c > h)))
    rules.append((f"close_lt_dclow{n}",  lambda df, c=close, l=dc_low: (c < l)))

# cap 500
rules = rules[:500]
print(f"Wave 9: generated {len(rules)} new indicator rules.")

# evaluation
DAILY_RF = ANNUAL_RF / 252

def evaluate(sig: pd.Series):
    sr = sig.shift(0) * qqq["next_ov_ret"]
    sr = sr.dropna()
    excess = sr - DAILY_RF
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
print("Top results (Wave 9):")
for i, (name, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {name:25s} | Sharpe: {sh:6.3f} | Trades: {tr}")

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
with open("history.log", "a", encoding="utf-8") as fh:
    for name, sh, tr in results[:3]:
        fh.write(f"[{now}] Wave9 | {name} | Sharpe={sh:.3f} | Trades={tr}\n")
print("Top 3 записаны в history.log")