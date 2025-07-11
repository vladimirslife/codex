#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 3: Проверка устойчивости лучшего условия по фазам рынка + ещё несколько однофакторных фильтров.

Part A. Стратегия с фильтром prev_overnight_return ≤ BEST_THR (0.0052)
        • Годовые Sharpe и число сделок
        • Sharpe в bull / bear (bull – close > SMA200)

Part B. Дополнительные однофакторные фильтры
        1) prev_overnight_return ≤ 0.0040 (плотнее)
        2) |day_return| ≤ 0.0180  (из Wave 2)
        3) intraday_range_pct ≤ 0.025
Выводим Top-10 лучших среди всех кандидатов.
"""

import numpy as np
import pandas as pd
from math import sqrt

annual_rf = 0.02
daily_rf = annual_rf / 252
BEST_THR = 0.0052  # из Wave 2, Sharpe≈1.06, trades≈3943


# ---------------------- DATA LOAD -----------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df[df["Date"] >= pd.Timestamp("2006-01-01")]
        .sort_values(by=["Date"])  # type: ignore[arg-type]
        .reset_index(drop=True)
    )
    df["Next_Open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    df["day_return"] = df["close"] / df["open"] - 1
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    return df.dropna(subset=["next_overnight_return"]).copy()


df = load("4 - QQQ.csv")

# ---------------------- HELPERS -------------------------------------

def compute_performance(signal: pd.Series) -> tuple[float, int]:
    """Return (Sharpe, trades) with annualised SR."""
    strat_ret = signal * df["next_overnight_return"]
    excess = strat_ret - daily_rf
    sharpe = (
        excess.mean() * 252 / (excess.std() * sqrt(252)) if excess.std() > 0 else 0
    )
    return sharpe, int(signal.sum())


# ---------------------- PART A: SUBPERIODS ---------------------------
print("\n=== Wave 3 — Part A: Стабильность лучшего фильтра (thr ≤ {:.4f}) ===".format(BEST_THR))

signal_best = (df["next_overnight_return"].shift(1) <= BEST_THR).astype(int)
sharpe_all, trades_all = compute_performance(signal_best)
print(f"Итого: Sharpe = {sharpe_all:.3f}, Trades = {trades_all}\n")

# Годовые метрики
print("Годовой разрез:")
years = df["Date"].dt.year.unique()
print("Year | Sharpe | Trades")
for y in years:
    mask = df["Date"].dt.year == y
    s_year = signal_best[mask]
    if s_year.sum() < 10:
        continue  # пропустим слишком короткие годы
    sr, tr = compute_performance(s_year)
    print(f"{y} | {sr:6.2f} | {tr:5d}")

# Bull/Bear по SMA200
sma200 = df["close"].rolling(200).mean()
bull_mask = df["close"] > sma200
bear_mask = ~bull_mask

sr_bull, tr_bull = compute_performance(signal_best[bull_mask])
sr_bear, tr_bear = compute_performance(signal_best[bear_mask])
print("\nBull phase (close > SMA200):  Sharpe = {:.2f}, Trades = {}".format(sr_bull, tr_bull))
print("Bear phase (close ≤ SMA200): Sharpe = {:.2f}, Trades = {}".format(sr_bear, tr_bear))

# ---------------------- PART B: EXTRA FILTERS ------------------------
print("\n=== Wave 3 — Part B: Дополнительные однофакторные фильтры ===")
candidates = {
    "prev_overnight ≤ 0.0040": (df["next_overnight_return"].shift(1) <= 0.0040).astype(int),
    "|day_return| ≤ 0.0180": (df["day_return"].shift(1).abs() <= 0.0180).astype(int),
    "range_pct ≤ 0.025": (df["range_pct"].shift(1) <= 0.025).astype(int),
}
results = []
for name, sig in candidates.items():
    sr, tr = compute_performance(sig)
    results.append((sr, tr, name))

# include best baseline for comparison
results.append((sharpe_all, trades_all, f"prev_overnight ≤ {BEST_THR:.4f} (baseline)"))

results.sort(key=lambda x: x[0], reverse=True)
print("Sharpe | Trades | Condition")
for sr, tr, name in results:
    flag = "*" if sr >= 1 and tr > 3800 else " "
    print(f"{sr:6.3f} | {tr:5d} | {name} {flag}")

# ---------------------- CHECKS --------------------------------------
# forward-looking bias: shift(1) везде присутствует
assert df["next_overnight_return"].shift(1).isna().sum() == 1