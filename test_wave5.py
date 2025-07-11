import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os

TICKER = "SPY"
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RF_ANNUAL = 0.02
RF_DAILY = RF_ANNUAL / 252.0
MAX_COMBOS = 1000

# --------------------------------------------------
# Data utility
# --------------------------------------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache_file = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache_file)
    return df.apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------
# Indicator helpers (no look-ahead)
# --------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def boll_width(series: pd.Series, window: int, num_std: float = 2):
    mavg = sma(series, window)
    std = series.rolling(window).std()
    upper = mavg + num_std * std
    lower = mavg - num_std * std
    return (upper - lower) / series

# --------------------------------------------------
# Performance metric
# --------------------------------------------------

def sharpe_ratio(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    std = ret.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    excess = ret - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

# --------------------------------------------------
# Condition evaluator
# --------------------------------------------------

def evaluate(df: pd.DataFrame, cond_name: str, params: dict):
    close = df["Close"]
    open_next = df["Open"].shift(-1)
    day_ret = (open_next - close) / close

    if cond_name == "EMA_GAP":
        s = params["short"]
        l = params["long"]
        d = params["delta"]
        signal = ema(close, s) > (1 + d) * ema(close, l)
    elif cond_name == "CLOSE_EMA_DEV":
        l = params["long"]
        d = params["delta"]
        signal = (close / ema(close, l) - 1) > d
    elif cond_name == "ATR_ABS_LOW":
        period = params["period"]
        thr = params["thr"]
        signal = (atr(df, period) / close) < thr
    elif cond_name == "BOLL_WIDTH":
        window = params["window"]
        thr = params["thr"]
        signal = boll_width(close, window) < thr
    else:
        raise ValueError("Unknown condition type")

    strat_ret = day_ret.copy()
    strat_ret[~signal] = 0.0

    trades = int(signal.sum())
    sr = sharpe_ratio(strat_ret.dropna())
    total_r = (1 + strat_ret.fillna(0)).prod() - 1

    return {
        "condition": cond_name,
        "params": params,
        "sharpe": sr,
        "trades": trades,
        "total_return": total_r,
    }

# --------------------------------------------------
# Build grid
# --------------------------------------------------
conditions = []

# Stronger EMA_GAP with higher deltas 0.015-0.03
for short in (5, 10, 15, 20, 25, 30):
    for long in (150, 200):
        if long > short:
            for delta in (0.015, 0.02, 0.025, 0.03):
                conditions.append(("EMA_GAP", {"short": short, "long": long, "delta": delta}))

# Price deviation above long EMA
for long in (100, 150, 200):
    for delta in (0.01, 0.02, 0.03, 0.04, 0.05):
        conditions.append(("CLOSE_EMA_DEV", {"long": long, "delta": delta}))

# ATR absolute low (volatility contraction)
for period in (14, 20):
    for thr in (0.004, 0.005, 0.006, 0.007):  # 0.4% - 0.7%
        conditions.append(("ATR_ABS_LOW", {"period": period, "thr": thr}))

# Bollinger width ultra narrow
for window in (20, 30):
    for thr in (0.02, 0.025, 0.03):
        conditions.append(("BOLL_WIDTH", {"window": window, "thr": thr}))

# Trim list
conditions = conditions[:MAX_COMBOS]
print(f"Wave5: total conditions = {len(conditions)}")

# --------------------------------------------------
# Execute tests
# --------------------------------------------------
prices = load_data(TICKER, START_DATE, END_DATE)

results = [evaluate(prices, name, p) for name, p in conditions]
res_df = pd.DataFrame(results)
res_df_sorted = res_df.sort_values(by=["sharpe", "trades"], ascending=[False, False])

print("\nTop 10 Wave5:")
print(res_df_sorted.head(10).to_string(index=False))

# Log top 3
with open("history.log", "a") as f:
    for _, row in res_df_sorted.head(3).iterrows():
        f.write(
            f"Wave5 | Condition: {row['condition']} | Params: {row['params']} | "
            f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n"
        )
print("Top 3 Wave5 appended to history.log")