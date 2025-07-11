# wave4_search.py
"""Wave 4: Поиск одного условия с новыми композитами и random-search порога.

Новые индикаторы (все вычисляются на данных, известных к моменту T):
  • lag_max12        = max( max3_{t-1} , max3_{t-2} )
  • roll_mean_max3_2 = среднее max3 за 2 ночи
  • roll_mean_max3_3 = среднее max3 за 3 ночи
  • composite_w      = α * max3 + β * range3  (α ∈ [0,1])
Условие одно: indicator < thr.
Random-search использует 1000 случайных пар (indicator, threshold).
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import random

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
FILES = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv", "XLK": "4 - XLK.csv"}
random.seed(42)
np.random.seed(42)

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

prev1 = base[["qqq", "spy", "xlk"]].shift(1)
prev2 = base[["qqq", "spy", "xlk"]].shift(2)

max3 = prev1.max(axis=1)
min3 = prev1.min(axis=1)
range3 = max3 - min3
max3_prev2 = prev2.max(axis=1)
lag_max12 = np.maximum(max3, max3_prev2)
roll_mean_max3_2 = max3.rolling(2).mean()
roll_mean_max3_3 = max3.rolling(3).mean()

indicators = {
    "lag_max12": lag_max12,
    "roll_mean_max3_2": roll_mean_max3_2,
    "roll_mean_max3_3": roll_mean_max3_3,
}

# precompute composite weights
for i in range(100):  # 100 random weight combos
    alpha = random.random()
    beta = 1 - alpha
    indicators[f"comp_w_{i:03d}"] = alpha * max3 + beta * range3

results = []  # (Sharpe, trades, ind_name, thr)

# random sampling of thresholds around 0–0.01
for name, series in indicators.items():
    series = series.astype(float)
    finite_vals = series[np.isfinite(series)]
    if finite_vals.empty:
        continue
    lo, hi = finite_vals.quantile(0.01), finite_vals.quantile(0.99)
    # расширим чуть
    thr_candidates = np.random.uniform(lo, hi, 100)  # 100 random thresholds per indicator
    for thr in thr_candidates:
        signal = (series < thr).astype(int)
        trades = int(signal.sum())
        if trades < 2500:
            continue
        strat = signal * base["qqq"]
        if strat.std() == 0:
            continue
        sharpe = (strat - DAILY_RF).mean() * 252 / (strat.std() * np.sqrt(252))
        results.append((sharpe, trades, name, thr))

results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 4: random-search одиночного условия ===")
for i, (s, n, ind, thr) in enumerate(top3, 1):
    print(f"#{i}: Sharpe={s:.4f}, trades={n}, ind={ind}, thr={thr:.6f}")

# append to history
log_line = f"{datetime.utcnow().isoformat()}\tWave4\t" + " | ".join([
    f"Sharpe={s:.4f},N={n},ind={ind},thr={thr:.6f}" for s, n, ind, thr in top3]) + "\n"
if Path("history.log").exists():
    Path("history.log").write_text(Path("history.log").read_text()+log_line, encoding="utf-8")
else:
    Path("history.log").write_text(log_line, encoding="utf-8")