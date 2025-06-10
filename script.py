#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overnight-return strategy for QQQ confirmed by SPY and XLK.
Entry: overnight returns of all three ETFs < 0.35 %.
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

    # Shifted references
    df["prev_close"] = df["close"].shift(1)
    df["Next_Open"]  = df["open"].shift(-1)

    # Overnight returns
    df["today_overnight_return"] = df["open"] / df["prev_close"] - 1
    df["next_overnight_return"]  = df["Next_Open"] / df["close"] - 1
    return df


# ------------------------- I/O ---------------------------------------
default_files = {
    "QQQ": "4 - QQQ.csv",
    "SPY": "4 - SPY.csv",
    "XLK": "4 - XLK.csv"
}

# CLI overrides: qqq_path, spy_path, xlk_path
for i, key in enumerate(default_files.keys(), start=1):
    if len(sys.argv) > i:
        default_files[key] = sys.argv[i]

missing = [p for p in default_files.values() if not os.path.exists(p)]
if missing:
    print("Missing file(s):", ", ".join(missing))
    sys.exit(1)

data = {t: load_ticker(p) for t, p in default_files.items()}

# ------------------------- MERGE -------------------------------------
common = data["QQQ"][["Date",
                       "today_overnight_return",
                       "next_overnight_return"]].rename(
    columns={
        "today_overnight_return": "QQQ_ov",
        "next_overnight_return":  "QQQ_next_ov"
    }
)

for t in ("SPY", "XLK"):
    tmp = data[t][["Date",
                   "today_overnight_return"]].rename(
        columns={"today_overnight_return": f"{t}_ov"}
    )
    common = common.merge(tmp, on="Date", how="inner")

common.sort_values("Date", inplace=True)
common.reset_index(drop=True, inplace=True)


# ------------------------- STRATEGY ----------------------------------
threshold = 0.0035   # 0.35 %

common["signal"] = (
    (common["QQQ_ov"] < threshold) &
    (common["SPY_ov"] < threshold) &
    (common["XLK_ov"] < threshold)
).astype(int)

common["strategy_daily_return"] = 0.0
for idx in common.index[common["signal"] == 1]:
    if idx + 1 < len(common):
        common.loc[idx + 1,
                   "strategy_daily_return"] = common.loc[idx, "QQQ_next_ov"]


# ------------------------- PERFORMANCE -------------------------------
annual_rf = 0.02
daily_rf  = annual_rf / 252

common["excess_return"] = common["strategy_daily_return"] - daily_rf
mean_excess_daily = common["excess_return"].mean()
std_excess_daily  = common["excess_return"].std()

mean_excess_annual = mean_excess_daily * 252
std_excess_annual  = std_excess_daily * np.sqrt(252)
sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0

common["strategy_equity"] = (1 + common["strategy_daily_return"]).cumprod()
total_return = common["strategy_equity"].iloc[-1] - 1
years_span = (common["Date"].iloc[-1] - common["Date"].iloc[0]).days / 365.25
cagr = (1 + total_return) ** (1 / years_span) - 1 if years_span > 0 else 0


# ------------------------- OUTPUT ------------------------------------
print("\n=== Strategy results (threshold 0.35 %) ===")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"CAGR:         {cagr*100:.2f}%")
print(f"Days:         {len(common)}")
print(f"Trades:       {common['signal'].sum()}")