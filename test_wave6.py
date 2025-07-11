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
# Data loader with caching
# --------------------------------------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache_path = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache_path)
    return df.apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------
# Indicator helpers (no look-ahead)
# --------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return true_range(df).rolling(period).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0)
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)

    tr_series = true_range(df)
    atr_series = tr_series.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_series)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_series)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_series = dx.rolling(period).mean()
    return adx_series


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()
    denom = (rolling_max - rolling_min).replace(0, np.nan)
    pr = (series - rolling_min) / denom
    return pr


def z_score(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

# --------------------------------------------------
# Sharpe calculation
# --------------------------------------------------

def sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    std = returns.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    excess = returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

# --------------------------------------------------
# Condition evaluation
# --------------------------------------------------

def evaluate(df: pd.DataFrame, cond_name: str, params: dict):
    close = df["Close"]
    open_next = df["Open"].shift(-1)
    daily_ret = (open_next - close) / close

    signal = pd.Series(False, index=df.index)

    if cond_name == "PERCENT_RANK_HIGH":
        window = params["window"]
        thr = params["thr"]
        signal = percentile_rank(close, window) > thr
    elif cond_name == "ADX_GT":
        period = params["period"]
        thr = params["thr"]
        signal = adx(df, period) > thr
    elif cond_name == "ZSCORE_HIGH":
        window = params["window"]
        k = params["k"]
        signal = z_score(close, window) > k
    elif cond_name == "EMA_RATIO":
        short = params["short"]
        long = params["long"]
        delta = params["delta"]
        signal = (ema(close, short) / ema(close, long) - 1) > delta
    else:
        raise ValueError("Unknown condition")

    strat_ret = daily_ret.copy()
    strat_ret[~signal] = 0.0

    trades = int(signal.sum())
    sr = sharpe_ratio(strat_ret.dropna())
    total_ret = (1 + strat_ret.fillna(0)).prod() - 1

    return {
        "condition": cond_name,
        "params": params,
        "sharpe": sr,
        "trades": trades,
        "total_return": total_ret,
    }

# --------------------------------------------------
# Build grid of conditions
# --------------------------------------------------
conds = []

# 1) Percentile Rank High
for window in (50, 100, 200):
    for thr in (0.8, 0.85, 0.9, 0.95):
        conds.append(("PERCENT_RANK_HIGH", {"window": window, "thr": thr}))

# 2) ADX strength
for period in (14, 20, 28):
    for thr in (20, 25, 30, 35):
        conds.append(("ADX_GT", {"period": period, "thr": thr}))

# 3) Z-score high relative to SMA
for window in (20, 50, 100):
    for k in (1.5, 2.0, 2.5):
        conds.append(("ZSCORE_HIGH", {"window": window, "k": k}))

# 4) EMA ratio (adaptive scaling)
for short in (5, 10, 15, 20):
    for long in (50, 100, 150, 200):
        if long > short:
            for delta in (0.005, 0.01, 0.015):
                conds.append(("EMA_RATIO", {"short": short, "long": long, "delta": delta}))

conds = conds[:MAX_COMBOS]
print(f"Wave6: total conditions = {len(conds)}")

# --------------------------------------------------
# Execute tests
# --------------------------------------------------
prices = load_data(TICKER, START_DATE, END_DATE)

results = [evaluate(prices, n, p) for n, p in conds]
res_df = pd.DataFrame(results)
res_df_sorted = res_df.sort_values(by=["sharpe", "trades"], ascending=[False, False])

print("\nTop 10 Wave6:")
print(res_df_sorted.head(10).to_string(index=False))

with open("history.log", "a") as f:
    for _, row in res_df_sorted.head(3).iterrows():
        f.write(
            f"Wave6 | Condition: {row['condition']} | Params: {row['params']} | "
            f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n"
        )
print("Top 3 Wave6 appended to history.log")