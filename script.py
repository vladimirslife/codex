#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base script for QQQ strategy performance measurement without entry filters.
"""

import pandas as pd
import numpy as np
import sys
import os

# ------------------------- HELPERS -----------------------------------
def load_ticker(path: str) -> pd.DataFrame:
    """Load CSV, standardize columns, compute overnight returns."""
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df[df["Date"] >= pd.Timestamp("2006-01-01")]
          .sort_values("Date")
          .reset_index(drop=True)
    )
    df["Next_Open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    return df

# ------------------------- I/O ---------------------------------------
default_files = {
    "QQQ": "4 - QQQ.csv",
    "SPY": "4 - SPY.csv",
    "XLK": "4 - XLK.csv"
}

for i, key in enumerate(default_files.keys(), start=1):
    if len(sys.argv) > i:
        default_files[key] = sys.argv[i]

missing = [p for p in default_files.values() if not os.path.exists(p)]
if missing:
    print("Missing file(s):", ", ".join(missing))
    sys.exit(1)

data = {t: load_ticker(p) for t, p in default_files.items()}

# ------------------------- MERGE -------------------------------------
common = data["QQQ"][["Date", "open", "close", "next_overnight_return"]].rename(
    columns={"next_overnight_return": "QQQ_next_ov"}
)

common.sort_values("Date", inplace=True)
common.reset_index(drop=True, inplace=True)

# ------------------------- STRATEGY & ENTRY FILTER -------------------
# compute 20-day simple moving average on close prices
common["sma20"] = common["close"].rolling(window=20).mean()
# merge SPY and XLK next overnight returns for additional entry filters
common = common.merge(
    data["SPY"][['Date', 'next_overnight_return']].rename(columns={'next_overnight_return': 'SPY_next_ov'}),
    on='Date', how='left'
)
common = common.merge(
    data["XLK"][['Date', 'next_overnight_return']].rename(columns={'next_overnight_return': 'XLK_next_ov'}),
    on='Date', how='left'
)
# signal: yesterday's close above SMA20 AND (yesterday's daily bar was green OR yesterday's overnight return was positive)
#        AND (SPY or XLK yesterday overnight return was positive)
common["signal"] = (
    (common["close"].shift(1) > common["sma20"].shift(1))
    & (
        (common["close"].shift(1) > common["open"].shift(1))
        | (common["QQQ_next_ov"].shift(1) > 0)
    )
    & (
        (common["SPY_next_ov"].shift(1) > 0)
        | (common["XLK_next_ov"].shift(1) > 0)
    )
).astype(int)
# strategy return: if signal==1, capture overnight return from yesterday close to today open
common["strategy_daily_return"] = common["signal"] * common["QQQ_next_ov"].shift(1)
common["strategy_daily_return"].fillna(0, inplace=True)

# ------------------------- PERFORMANCE -------------------------------
annual_rf = 0.02
daily_rf  = annual_rf / 252

excess_returns = common["strategy_daily_return"] - daily_rf
mean_excess_annual = excess_returns.mean() * 252
std_excess_annual = excess_returns.std() * np.sqrt(252)
sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0

common["strategy_equity"] = (1 + common["strategy_daily_return"]).cumprod()
total_return = common["strategy_equity"].iloc[-1] - 1
years_span = (common["Date"].iloc[-1] - common["Date"].iloc[0]).days / 365.25
cagr = (1 + total_return) ** (1 / years_span) - 1 if years_span > 0 else 0

# ------------------------- OUTPUT ------------------------------------
print("\n=== Strategy results (SMA20 + SPY/XLK overnight filter) ===")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"CAGR:         {cagr*100:.2f}%")
print(f"Days:         {len(common)}")
# count actual entry signals as trades
num_trades = int(common["signal"].sum())
print(f"Trades:       {num_trades}")
