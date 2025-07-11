#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 23: условие (gap - beta*ewma_gap_N1_prev)/(ewma_abs_gap_N2_prev + gamma) < thr
Одна проверка, >3000 сделок, цель Sharpe >=1.3
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

# load
_df = pd.read_csv(FILE)
_df.columns = [c.lower() for c in _df.columns]
_df["date"] = pd.to_datetime(_df["date"])
_df = _df[_df["date"] >= pd.Timestamp("2006-01-01")].sort_values("date").reset_index(drop=True)

prev_close = _df["close"].shift(1)
gap = _df["open"] / prev_close - 1

next_ov = _df["open"].shift(-1) / _df["close"] - 1

# parameter grids (keep small)
N1_vals = [5, 10, 20]
N2_vals = [5, 10, 20]

beta_vals = np.round(np.arange(-0.30, -0.09, 0.03), 4)   # -0.30 .. -0.12
thr_vals = np.round(np.arange(0.002, 0.0105, 0.001), 4)  # 0.002 .. 0.01

gamma_vals = [1e-4, 3e-4, 5e-4]

best = None
for N1 in N1_vals:
    ewma_gap = gap.ewm(span=N1, adjust=False).mean().shift(1)
    for N2 in N2_vals:
        ewma_abs = gap.abs().ewm(span=N2, adjust=False).mean().shift(1)
        for gamma in gamma_vals:
            denom = ewma_abs + gamma
            # avoid division by zero
            denom = denom.replace(0, np.nan)
            for beta in beta_vals:
                numer = gap - beta * ewma_gap
                metric = numer / denom
                for thr in thr_vals:
                    sig = (metric < thr).astype(int)
                    trades = int(sig.sum())
                    if trades < 3000:
                        continue
                    strat = sig * next_ov
                    excess = strat - DAILY_RF
                    sd = excess.std()
                    if sd == 0:
                        continue
                    sr = (excess.mean() / sd) * np.sqrt(252)
                    if best is None or sr > best[0]:
                        best = (sr, trades, beta, N1, N2, gamma, thr)

if best:
    sr, trades, beta, N1, N2, gamma, thr = best
    print(
        f"Wave23 BEST Sharpe={sr:.4f}, Trades={trades}, Cond: (gap - {beta:.3f}*EWMA_gap(span={N1})) / (EWMA_abs_gap(span={N2}) + {gamma}) < {thr:.4f}"
    )
    with open("history.log", "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.utcnow().isoformat()} | Wave 23 | best | Sharpe={sr:.4f} | Trades={trades} | Condition: (gap - {beta:.3f}*ewma_gap_{N1})/(ewma_abs_{N2}+{gamma}) < {thr:.4f}\n"
        )
else:
    print("Wave23: no combo with >3000 trades")