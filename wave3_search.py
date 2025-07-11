# wave3_search.py
"""Wave 3: Расширенный перебор одного условия на сложных композитных индикаторах.

Индикаторы из прошлых overnight-доходностей (доступно до T):
 1. max3       – максимум r_{t-1} по QQQ, SPY, XLK
 2. range3     – max3 - min3 (спред)
 3. std3       – стандартное отклонение по трём тикерам
 4. rolling_std_qqq_3 – std за 3 дня по QQQ
 5. composite1 – max3 - std3
 6. composite2 – max3 / (std3 + 1e-6)
 7. rolling_range_5 – диапазон max-min последних 5 ночей QQQ

Условие одно: indicator < threshold (или > threshold для диапазона проверяем отдельно).
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
FILES = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv", "XLK": "4 - XLK.csv"}

# ------------- LOAD -----------------
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

# ---------- PREVIOUS RETURNS ----------
prev1 = base[["qqq", "spy", "xlk"]].shift(1)
prev2 = base[["qqq", "spy", "xlk"]].shift(2)

min3 = prev1.min(axis=1)
max3 = prev1.max(axis=1)
range3 = max3 - min3
std3 = prev1.std(axis=1)
# Rolling stats on QQQ
rolling_std_qqq_3 = base["qqq"].shift(1).rolling(3).std()
# Composite
composite1 = max3 - std3
composite2 = max3 / (std3 + 1e-6)
# rolling range 5 nights QQQ
rolling_range_5 = base["qqq"].shift(1).rolling(5).apply(lambda x: x.max() - x.min(), raw=True)

indicators = {
    "max3": max3,
    "range3": range3,
    "std3": std3,
    "rolling_std_qqq_3": rolling_std_qqq_3,
    "comp1_max_minus_std": composite1,
    "comp2_ratio": composite2,
    "rolling_range5": rolling_range_5,
}

results = []  # (Sharpe, trades, ind_name, cmp, thr)

for name, series in indicators.items():
    series = series.astype(float)
    # Сканируем диапазон порогов, зависящий от распределения индикатора
    finite_vals = series[np.isfinite(series)]
    lo, hi = np.nanpercentile(finite_vals, [1, 99]) if len(finite_vals) else (-0.02, 0.02)
    # слегка расширяем
    rng_min, rng_max = lo - abs(lo) * 0.1, hi + abs(hi) * 0.1
    # определяем шаг
    step = (rng_max - rng_min) / 400  # максимум ~400 тестов на индикатор
    if step <= 0:
        continue
    thresholds = np.arange(rng_min, rng_max + step / 2, step)

    # вариант «< threshold»
    for thr in thresholds:
        signal = (series < thr).astype(int)
        trades = int(signal.sum())
        if trades < 2500:
            continue
        strat = signal * base["qqq"]
        if strat.std() == 0:
            continue
        sharpe = (strat - DAILY_RF).mean() * 252 / (strat.std() * np.sqrt(252))
        results.append((sharpe, trades, name, "<", thr))

    # вариант «> threshold» – только для диапазонов / std, где имеет смысл
    if name in {"range3", "std3", "rolling_std_qqq_3", "rolling_range5"}:
        for thr in thresholds:
            signal = (series > thr).astype(int)
            trades = int(signal.sum())
            if trades < 2500:
                continue
            strat = signal * base["qqq"]
            if strat.std() == 0:
                continue
            sharpe = (strat - DAILY_RF).mean() * 252 / (strat.std() * np.sqrt(252))
            results.append((sharpe, trades, name, ">", thr))

# ----------------- SELECT TOP -----------------
results.sort(key=lambda x: x[0], reverse=True)

top3 = results[:3]
print("=== Wave 3: сложные композиты, одно условие ===")
for i, (s, n, ind, cmp, thr) in enumerate(top3, 1):
    print(f"#{i}: Sharpe={s:.4f}, trades={n}, ind={ind}, cond='{cmp} {thr:.6f}'")

# log history
line = f"{datetime.utcnow().isoformat()}\tWave3\t" + " | ".join([
    f"Sharpe={s:.4f},N={n},ind={ind},cmp={cmp},thr={thr:.6f}" for s, n, ind, cmp, thr in top3]) + "\n"
Path("history.log").write_text(Path("history.log").read_text() + line if Path("history.log").exists() else line, encoding="utf-8")