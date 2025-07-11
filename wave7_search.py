# wave7_search.py
"""Wave 7 – 100k random-search одного условия.
Индикаторы:
 1. compA = α·max3 + (1−α)·range3, α∈[0.75,0.95]
 2. compB = max3 − γ·range3,      γ∈[−0.5, +0.5]
Порог thr в лог-масштабе: 10^u, u ∈ [log10(0.0020), log10(0.0038)].
Проверяются варианты '< thr' для compA и compB, а также '> thr' для compB.
Цель: Sharpe ≥1.4, trades>2500. Всего 100 000 случайных комбинаций.
"""
import pandas as pd
import numpy as np
import random, math, time
from datetime import datetime
from pathlib import Path

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
FILES = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv", "XLK": "4 - XLK.csv"}
RND = np.random.default_rng(2025)

# ---------- load ----------
raw = {}
for sym, path in FILES.items():
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
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
max3 = prev.max(axis=1)
range3 = max3 - prev.min(axis=1)

# keep numpy arrays for speed
ret_arr = base["qqq"].values.astype(float)
max3_arr = max3.values.astype(float)
range3_arr = range3.values.astype(float)
mask_valid = np.isfinite(max3_arr)

BEST = []
N_TOTAL = 100_000
log_lo, log_hi = math.log10(0.0020), math.log10(0.0038)
start = time.time()
for _ in range(N_TOTAL):
    # randomly choose variant
    if RND.random() < 0.5:  # compA variant
        alpha = RND.uniform(0.75, 0.95)
        comp = alpha * max3_arr + (1 - alpha) * range3_arr
        thr = 10 ** RND.uniform(log_lo, log_hi)
        cmp_op = "<"
    else:  # compB variant
        gamma = RND.uniform(-0.5, 0.5)
        comp = max3_arr - gamma * range3_arr
        thr = 10 ** RND.uniform(log_lo, log_hi)
        cmp_op = "<" if RND.random() < 0.7 else ">"
    # construct signal
    if cmp_op == "<":
        signal = (comp < thr) & mask_valid
    else:
        signal = (comp > thr) & mask_valid
    trades = int(signal.sum())
    if trades < 2500:
        continue
    strat = signal.astype(float) * ret_arr
    std = strat.std()
    if std == 0:
        continue
    sharpe = (strat.mean() - DAILY_RF) * 252 / (std * math.sqrt(252))
    BEST.append((sharpe, trades, alpha if 'alpha' in locals() else None, gamma if 'gamma' in locals() else None, thr, cmp_op))

BEST.sort(reverse=True, key=lambda x: x[0])

top3 = BEST[:3]
print("=== Wave 7 top hits ===")
for i, (s, n, a, g, t, cmp_op) in enumerate(top3, 1):
    desc = f"alpha={a:.4f}" if a is not None else f"gamma={g:.4f}"
    print(f"#{i}: Sharpe={s:.4f}, trades={n}, {desc}, cond='{cmp_op} {t:.6f}'")

# log
history = Path("history.log")
line = f"{datetime.utcnow().isoformat()}\tWave7\t" + " | ".join([
    f"Sharpe={s:.4f},N={n},{'alpha' if a is not None else 'gamma'}={a if a is not None else g:.4f},cmp={cmp_op},thr={t:.6f}"
    for s, n, a, g, t, cmp_op in top3]) + "\n"
history.write_text(history.read_text()+line if history.exists() else line, encoding="utf-8")
print(f"Processed {len(BEST)} valid combos out of {N_TOTAL} in {time.time()-start:.1f}s")