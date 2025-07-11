#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 1.
Один критерий входа: прошлая овернайт-доходность QQQ должна быть выше порога th.
Подбираем th из ограниченного диапазона, чтобы найти Sharpe ≥ 1.3 и >3000 сделок.
Результаты (топ-3 по Sharpe) записываются в history.log.
Работает <180 сек.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252

# ---------- data helpers ----------

def load_ticker(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df[df["Date"] >= pd.Timestamp("2006-01-01")]  # полный датасет без ограничений по времени
        .sort_values(by="Date")  # type: ignore[arg-type]
        .reset_index(drop=True)
    )
    df["next_open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["next_open"] / df["close"] - 1
    return df

# ---------- load ----------
path = "4 - QQQ.csv"
if not os.path.exists(path):
    raise FileNotFoundError(f"Не найден файл {path}")

df = load_ticker(path)

# ---------- grid search ----------
results = []
thresholds = np.linspace(-0.02, 0.0, 41)  # 41 значений (шаг ~0.0005)
for th in thresholds:
    # сигнал формируем в момент закрытия дня T на основе данных, доступных до T (ov_{T-1→T})
    signal = (df["next_overnight_return"].shift(1) > th).astype(int)
    strat_ret = signal * df["next_overnight_return"].shift(1)
    strat_ret = strat_ret.fillna(0)

    excess = strat_ret - DAILY_RF
    ann_mean = excess.mean() * 252
    ann_std = excess.std() * np.sqrt(252)
    sharpe = ann_mean / ann_std if ann_std != 0 else 0

    trades = int(signal.sum())

    results.append((sharpe, trades, th))

# ---------- выбрать топ-3 по Sharpe при условии >3000 сделок ----------
qualified = [r for r in results if r[1] > 3000 and r[0] >= 1.3]
qualified.sort(key=lambda x: x[0], reverse=True)

top3 = qualified[:3]

# ---------- вывод ----------
print("=== Wave 1 итоги ===")
if not top3:
    print("Условие не найдено, продолжаем в следующей волне…")
else:
    for idx, (sr, trades, th) in enumerate(top3, 1):
        print(f"{idx}) Sharpe={sr:.3f}  Trades={trades}  Условие: предыдущий овернайт > {th:.4%}")

# ---------- логирование ----------
if top3:
    with open("history.log", "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.utcnow().isoformat()}] Wave 1 – топ 3 результатов\n")
        for idx, (sr, trades, th) in enumerate(top3, 1):
            f.write(
                f"{idx}) Sharpe={sr:.4f}  Trades={trades}  Условие: previous_overnight_return > {th:.4%}\n"
            )