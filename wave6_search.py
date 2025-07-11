# wave6_search.py
"""Wave 6 – расширенный random-search одного условия для композита comp = α·max3 + (1−α)·range3.
Поиск:
  • α ∈ [0.7, 1.3]  (α>1 ⇒ отрицательный вес range3)
  • thr ∈ [0.0020, 0.0038]
  • 20 000 случайных комбинаций
Проверяются оба сравнения: comp < thr и comp > thr.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime
from pathlib import Path

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
FILES = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv", "XLK": "4 - XLK.csv"}
random.seed(999)
np.random.seed(999)

# ---------- load returns ----------
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
    base = base.merge(raw[sym].rename(columns={"ret": sym.lower()}), on="Date", how="left")

prev = base[["qqq", "spy", "xlk"]].shift(1)
max3 = prev.max(axis=1)
min3 = prev.min(axis=1)
range3 = max3 - min3

BEST = []  # (sharpe, trades, alpha, thr, cmp)
N_TRIES = 20000
for _ in range(N_TRIES):
    alpha = random.uniform(0.7, 1.3)
    thr = random.uniform(0.0020, 0.0038)
    comp = alpha * max3 + (1 - alpha) * range3
    for cmp in ("<", ">"):
        if cmp == "<":
            signal = (comp < thr).astype(int)
        else:
            signal = (comp > thr).astype(int)
        trades = int(signal.sum())
        if trades < 2500:
            continue
        strat = signal * base["qqq"]
        if strat.std() == 0:
            continue
        sharpe = (strat - DAILY_RF).mean() * 252 / (strat.std() * np.sqrt(252))
        BEST.append((sharpe, trades, alpha, thr, cmp))

BEST.sort(reverse=True)

top3 = BEST[:3]
print("=== Wave 6: extended random-search (α,thr,cmp) ===")
for i, (s, n, a, t, cmp) in enumerate(top3, 1):
    print(f"#{i}: Sharpe={s:.4f}, trades={n}, alpha={a:.4f}, cond='{cmp} {t:.6f}'")

# history log
line = f"{datetime.utcnow().isoformat()}\tWave6\t" + " | ".join([
    f"Sharpe={s:.4f},N={n},alpha={a:.4f},cmp={cmp},thr={t:.6f}" for s, n, a, t, cmp in top3]) + "\n"
Path("history.log").write_text(Path("history.log").read_text()+line if Path("history.log").exists() else line, encoding="utf-8")