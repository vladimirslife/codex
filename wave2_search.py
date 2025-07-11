# wave2_search.py
"""Wave 2: Поиск одного условия на основе композитных предыдущих overnight-доходностей.
Индикаторы:
  • prev_qqq, prev_spy, prev_xlk – одиночные тикеры
  • max3  – максимальная из трёх
  • mean3 – средняя из трёх
  • min3  – минимальная из трёх
Условие всегда одно: indicator < threshold.
Для честности сигналы считаются на r_{t-1}, прибыль – на r_t.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
TICKERS = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv", "XLK": "4 - XLK.csv"}

# -------- load all returns --------
raw = {}
for sym, path in TICKERS.items():
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    if "time" in df.columns:
        df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    else:
        df.rename(columns={"date": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.Timestamp("2006-01-01")].copy()
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ret"] = df["open"].shift(-1) / df["close"] - 1
    raw[sym] = df[["Date", "ret"]]

# merge
base = raw["QQQ"].rename(columns={"ret": "qqq"})
for sym in ("SPY", "XLK"):
    base = base.merge(raw[sym].rename(columns={"ret": sym.lower()}), on="Date", how="left")

# previous returns (t-1)
prev = base[["qqq", "spy", "xlk"]].shift(1)

indicators = {
    "prev_qqq": prev["qqq"],
    "prev_spy": prev["spy"],
    "prev_xlk": prev["xlk"],
    "max3": prev.max(axis=1),
    "mean3": prev.mean(axis=1),
    "min3": prev.min(axis=1),
}

THRESH_RANGE = np.arange(-0.02, 0.02 + 1e-9, 0.00025)

results = []  # (Sharpe, trades, indicator_name, threshold)
for name, series in indicators.items():
    for thr in THRESH_RANGE:
        signal = (series < thr).astype(int)
        trades = int(signal.sum())
        if trades < 2500:
            continue
        strat_ret = signal * base["qqq"]  # прибыль в период (t, t+1)
        if strat_ret.std() == 0:
            continue
        excess = strat_ret - DAILY_RF
        sharpe = excess.mean() * 252 / (strat_ret.std() * np.sqrt(252))
        results.append((sharpe, trades, name, thr))

results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 2: Композитные индикаторы, одно условие ===")
for i, (s, n, name, thr) in enumerate(top3, 1):
    print(f"#{i}: Sharpe={s:.4f}, trades={n}, ind={name}, thr={thr:.6f}")

# log
history_line = (
    f"{datetime.utcnow().isoformat()}\tWave2\t" +
    " | ".join([f"Sharpe={s:.4f}, N={n}, ind={name}, thr={thr:.6f}" for s, n, name, thr in top3]) +
    "\n"
)
with open("history.log", "a", encoding="utf-8") as fh:
    fh.write(history_line)