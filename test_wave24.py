#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 24: (gap - beta*EMA_fast_prev) / (EMA_slow_prev + gamma) < thr
EMA_fast < EMA_slow; одна проверка; цель Sharpe>=1.3 и >3000 сделок
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

FILE = "4 - QQQ.csv"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252

if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

df = pd.read_csv(FILE)
df.columns = [c.lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= pd.Timestamp("2006-01-01")].sort_values("date").reset_index(drop=True)

prev_close = df["close"].shift(1)
gap = df["open"] / prev_close - 1
next_ov = df["open"].shift(-1) / df["close"] - 1

fast_spans = [3, 5, 8]
slow_spans = [15, 20, 30, 40]

beta_vals = np.round(np.arange(-0.3, -0.09, 0.03), 3)
thr_vals = np.round(np.arange(0.002, 0.0105, 0.0005), 4)
gamma_vals = [1e-4, 2e-4, 5e-4]

best = None
for fast in fast_spans:
    ema_fast = gap.ewm(span=fast, adjust=False).mean().shift(1)
    for slow in slow_spans:
        if slow <= fast:
            continue
        ema_slow = gap.ewm(span=slow, adjust=False).mean().shift(1)
        for gamma in gamma_vals:
            denom = (ema_slow.abs() + gamma).replace(0, np.nan)
            for beta in beta_vals:
                numer = gap - beta * ema_fast
                metric = numer / denom
                for thr in thr_vals:
                    sig = (metric < thr).astype(int)
                    trades = int(sig.sum())
                    if trades < 3000:
                        continue
                    excess = sig * next_ov - DAILY_RF
                    sd = excess.std()
                    if sd == 0:
                        continue
                    sr = (excess.mean() / sd) * np.sqrt(252)
                    if best is None or sr > best[0]:
                        best = (sr, trades, beta, fast, slow, gamma, thr)

if best:
    sr, trades, beta, fast, slow, gamma, thr = best
    print(
        f"Wave24 BEST Sharpe={sr:.4f}, Trades={trades}, Cond: (gap - {beta:.3f}*EMA_fast(span={fast}))/(|EMA_slow(span={slow})| + {gamma}) < {thr:.4f}"
    )
    with open("history.log", "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.utcnow().isoformat()} | Wave 24 | best | Sharpe={sr:.4f} | Trades={trades} | Condition: EMAfast{fast} EMAslow{slow} beta={beta:.3f} thr={thr:.4f} gamma={gamma}\n"
        )
else:
    print("Wave24: no combo >3000 trades")