#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 2 grid-search script.

Охватывает более «креативные» однофакторные правила, чтобы достигнуть
Sharpe ≥ 1.4 при > 2500 сделках.

Используемые индикаторы / правила (каждое — ровно одно логическое условие):
  • Короткий RSI (2-10) с низкими/высокими порогами.
  • Расширенный Bollinger Bands (σ до 4).
  • CCI (Commodity Channel Index) экстремумы.
  • Williams %R экстремумы.
  • ATR-каналы: цена ниже нижнего канала / выше верхнего канала.
  • ROC (Rate of Change): знак доходности.
  • %B (Bollinger percent-band) < 0.1 или > 0.9.

Всего генерируется ≤ 1000 вариантов.
Как и прежде: long-only, вход по close T, выход open T+1, без look-ahead.
"""

import os
from datetime import datetime
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd

# ---------------- CONFIG ------------------------------------------------------------------------
INPUT_FILE = os.environ.get("QQQ_CSV", "4 - QQQ.csv")
DATE_START = "2006-01-01"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
TOP_N = 15  # выводим чуть больше

# ---------------- DATA --------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= DATE_START].copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_ov_return"] = df["next_open"] / df["close"] - 1
    return df.dropna().reset_index(drop=True)

qqq = load_data(INPUT_FILE)
close = qqq["close"]
high = qqq["high"] if "high" in qqq.columns else close
low = qqq["low"] if "low" in qqq.columns else close

# ---------------- INDICATORS --------------------------------------------------------------------

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(window=n, min_periods=n).mean()
    roll_down = pd.Series(down).rolling(window=n, min_periods=n).mean()
    rs = roll_up / roll_down
    return 100.0 - 100.0 / (1.0 + rs)

def bollinger(series: pd.Series, n: int, n_std: float):
    mid = sma(series, n)
    std = series.rolling(window=n, min_periods=n).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, lower

def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = sma(tp, n)
    mad = (tp - sma_tp).abs().rolling(window=n, min_periods=n).mean()
    return (tp - sma_tp) / (0.015 * mad)

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    highest_high = high.rolling(window=n, min_periods=n).max()
    lowest_low = low.rolling(window=n, min_periods=n).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    high_low = high - low
    high_close_prev = (high - close.shift(1)).abs()
    low_close_prev = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.rolling(window=n, min_periods=n).mean()

def roc(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(periods=n)

# ---------------- RULE GENERATION ---------------------------------------------------------------
Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

# 1. Short-term RSI
RSI_PERIODS = list(range(2, 11))  # 2-10
RSI_LOW = [10, 15, 20, 25, 30]
RSI_HIGH = [70, 75, 80, 85, 90]
for n in RSI_PERIODS:
    rsi_series = rsi(close, n)
    for thr in RSI_LOW:
        rules.append((f"rsi{n}_lt_{thr}", lambda df, s=rsi_series, t=thr: (s < t)))
    for thr in RSI_HIGH:
        rules.append((f"rsi{n}_gt_{thr}", lambda df, s=rsi_series, t=thr: (s > t)))

# 2. Bollinger wide σ
BOLL_PERIODS = [5, 10, 15, 20, 30]
BOLL_STD = [2, 2.5, 3, 3.5, 4]
for n in BOLL_PERIODS:
    for sd in BOLL_STD:
        upper, lower = bollinger(close, n, sd)
        rules.append((f"close_lt_lowerBB_{n}_{sd}", lambda df, l=lower: (df["close"] < l)))
        rules.append((f"close_gt_upperBB_{n}_{sd}", lambda df, u=upper: (df["close"] > u)))

# 3. CCI extremes
CCI_PERIODS = [5, 10, 14, 20, 30]
CCI_THR = [100, 150, 200]
for n in CCI_PERIODS:
    cci_series = cci(high, low, close, n)
    for thr in CCI_THR:
        rules.append((f"cci{n}_gt_{thr}", lambda df, s=cci_series, t=thr: (s > t)))
        rules.append((f"cci{n}_lt_-{thr}", lambda df, s=cci_series, t=thr: (s < -t)))

# 4. Williams %R
WR_PERIODS = [5, 10, 14, 20]
for n in WR_PERIODS:
    wr = williams_r(high, low, close, n)
    rules.append((f"wr{n}_lt_-80", lambda df, w=wr: (w < -80)))
    rules.append((f"wr{n}_lt_-90", lambda df, w=wr: (w < -90)))
    rules.append((f"wr{n}_gt_-20", lambda df, w=wr: (w > -20)))
    rules.append((f"wr{n}_gt_-10", lambda df, w=wr: (w > -10)))

# 5. ATR channels
ATR_PERIODS = [10, 20, 30]
ATR_MULT = [1.0, 1.5, 2.0]
for n in ATR_PERIODS:
    atr_val = atr(high, low, close, n)
    sma_n = sma(close, n)
    lower_chan = sma_n - ATR_MULT[0] * atr_val  # will vary later in lambda
    upper_chan = sma_n + ATR_MULT[0] * atr_val
    for m in ATR_MULT:
        low_ch = sma_n - m * atr_val
        up_ch = sma_n + m * atr_val
        rules.append((f"close_lt_ATRchanLow_{n}_{m}", lambda df, lc=low_ch: (df["close"] < lc)))
        rules.append((f"close_gt_ATRchanUp_{n}_{m}", lambda df, uc=up_ch: (df["close"] > uc)))

# 6. ROC sign
ROC_PERIODS = [1, 2, 3, 5, 10]
for n in ROC_PERIODS:
    roc_series = roc(close, n)
    rules.append((f"roc{n}_gt0", lambda df, s=roc_series: (s > 0)))
    rules.append((f"roc{n}_lt0", lambda df, s=roc_series: (s < 0)))

# 7. Percent-B of Bollinger
PB_PERIODS = [5, 10, 20]
for n in PB_PERIODS:
    up, lo = bollinger(close, n, 2)
    pb = (close - lo) / (up - lo)
    rules.append((f"pb{n}_lt_0.1", lambda df, p=pb: (p < 0.1)))
    rules.append((f"pb{n}_gt_0.9", lambda df, p=pb: (p > 0.9)))

# Ограничиваем до 1000 правил
rules = rules[:1000]

print(f"Generated {len(rules)} candidate single-condition rules (Wave 2).")

# ---------------- EVALUATION --------------------------------------------------------------------

def evaluate(sig: pd.Series) -> Tuple[float, int]:
    strat_ret = sig.shift(0) * qqq["next_ov_return"]
    strat_ret = strat_ret.dropna()
    excess = strat_ret - DAILY_RF
    mean_excess = excess.mean() * 252
    std_excess = excess.std() * np.sqrt(252)
    sharpe = mean_excess / std_excess if std_excess > 0 else 0.0
    trades = int(sig.sum())
    return sharpe, trades

results: List[Tuple[str, float, int]] = []
for name, func in rules:
    sig = func(qqq).astype(int)
    sh, tr = evaluate(sig)
    results.append((name, sh, tr))

results.sort(key=lambda x: x[1], reverse=True)

print("Top results (Wave 2):")
for i, (cond, sh, tr) in enumerate(results[:TOP_N], 1):
    print(f"{i:2d}. {cond:35s} | Sharpe: {sh:6.3f} | Trades: {tr}")

# ---------------- HISTORY LOG -------------------------------------------------------------------
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
hist_lines = [f"[{now}] Wave2 | {cond} | Sharpe={sh:.3f} | Trades={tr}\n" for cond, sh, tr in results[:3]]
with open("history.log", "a", encoding="utf-8") as fh:
    fh.writelines(hist_lines)
print("\nTop 3 записаны в history.log")