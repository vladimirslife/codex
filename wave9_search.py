# wave9_search.py
"""Wave 9 – расширенный random-search с нелинейными и робастными трансформациями.
Остаётся одно условие: f_t-1(indicator) < thr   ИЛИ   |indicator| > thr.
Цель: Sharpe ≥ 1.4, trades > 2500.
"""
import math, random, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
FILES = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv", "XLK": "4 - XLK.csv"}
RAND = np.random.default_rng(42)

# ---------- load ----------
raw = {}
for sym, path in FILES.items():
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    else:
        df.rename(columns={"date": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= "2006-01-01"].copy()
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ret"] = df["open"].shift(-1) / df["close"] - 1
    raw[sym] = df[["Date", "ret"]]

base = raw["QQQ"].rename(columns={"ret": "qqq"})
for sym in ("SPY", "XLK"):
    base = base.merge(raw[sym].rename(columns={"ret": sym.lower()}), on="Date")

prev = base[["qqq", "spy", "xlk"]].shift(1)
max3   = prev.max(axis=1).values.astype(float)
min3   = prev.min(axis=1).values.astype(float)
range3 = max3 - min3
ret_arr = base["qqq"].values.astype(float)
mask = np.isfinite(max3) & np.isfinite(range3) & np.isfinite(ret_arr)

# -------- pre-compute rolling ranks + MAD -----------------
windows = [30, 60, 90]
roll_rank_dict = {}
mad_dict = {}
series_max = pd.Series(max3)
series_range = pd.Series(range3)
for w in windows:
    roll_rank_dict[("max3", w)] = series_max.rolling(w).rank(pct=True).shift(0).values
    roll_rank_dict[("range3", w)] = series_range.rolling(w).rank(pct=True).shift(0).values
    median_max = series_max.rolling(w).median()
    mad_max = series_max.rolling(w).apply(lambda x: np.median(np.abs(x-np.median(x))), raw=True)
    mad_dict[("max3", w)] = ((series_max - median_max) / (mad_max + 1e-8)).shift(0).values

# ---------- random search ----------
BEST = []
N_TRIES = 200_000
start = time.time()

for _ in range(N_TRIES):
    ind_choice = RAND.integers(0, 7)
    if ind_choice == 0:  # tanh
        k = RAND.uniform(80, 200)
        indicator = np.tanh(k * max3)
        thr = RAND.uniform(0.1, 0.3)
        cmp = ">"
    elif ind_choice == 1:  # sinh
        k = RAND.uniform(30, 80)
        indicator = np.sign(max3) * (np.sinh(k * np.abs(max3)))
        thr = RAND.uniform(0.5, 2.0)
        cmp = ">"
    elif ind_choice == 2:  # rolling rank max3
        w = int(RAND.choice(windows))
        indicator = roll_rank_dict[("max3", w)]
        thr = RAND.uniform(0.6, 0.9)
        cmp = ">"
    elif ind_choice == 3:  # rolling rank range3
        w = int(RAND.choice(windows))
        indicator = roll_rank_dict[("range3", w)]
        thr = RAND.uniform(0.6, 0.9)
        cmp = ">"
    elif ind_choice == 4:  # MAD-norm
        w = int(RAND.choice(windows))
        indicator = mad_dict[("max3", w)]
        thr = RAND.uniform(2.0, 4.0)
        cmp = ">"
    elif ind_choice == 5:  # exponential weighted comb
        alpha = RAND.uniform(0.7, 0.95)
        indicator = alpha * max3 + (1 - alpha) * range3 * RAND.uniform(0.5, 1.5)
        thr = RAND.uniform(0.0025, 0.0035)
        cmp = "<"
    else:  # log ratio
        indicator = np.log(max3 / (range3 + 1e-6))
        thr = RAND.uniform(-0.2, 0.1)
        cmp = "<"

    ind = indicator
    if cmp == "<":
        signal = (ind < thr) & mask
    else:
        signal = (ind > thr) & mask
    trades = int(signal.sum())
    if trades < 2500:
        continue
    strat = signal.astype(float) * ret_arr
    std = strat.std()
    if std == 0:
        continue
    sharpe = (strat.mean() - DAILY_RF) * 252 / (std * math.sqrt(252))
    if sharpe >= 1.4:
        print(f"FOUND Sharpe≥1.4: {sharpe:.4f}, trades={trades}, choice={ind_choice}, thr={thr:.6f}, cmp={cmp}")
        break
    BEST.append((sharpe, trades, ind_choice, thr, cmp))
else:
    BEST.sort(reverse=True)
    print("Top-5 after search (Sharpe, trades, choice, thr, cmp):")
    for row in BEST[:5]:
        print(row)

print(f"Elapsed {time.time()-start:.1f}s, evaluated {len(BEST)} combos.")