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
# Data download & cache
# --------------------------------------------------

def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache)
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

def bollinger_width(series: pd.Series, window: int, num_std: float = 2) -> pd.Series:
    ma = sma(series, window)
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / series

def roc(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change(period)

# --------------------------------------------------
# Metric
# --------------------------------------------------

def sharpe(returns: pd.Series) -> float:
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

def evaluate(df: pd.DataFrame, cond_name: str, p: dict):
    close = df["Close"]
    open_next = df["Open"].shift(-1)
    daily_ret = (open_next - close) / close
    signal = pd.Series(False, index=df.index)

    if cond_name == "EMA_GAP":
        s = p["short"]
        l = p["long"]
        d = p["delta"]
        signal = ema(close, s) > (1 + d) * ema(close, l)
    elif cond_name == "ATR_LOW":
        period = p["period"]
        k = p["k"]
        atr_series = atr(df, period)
        atr_ma = atr_series.rolling(period * 2).mean()
        signal = atr_series < k * atr_ma
    elif cond_name == "BOLL_NARROW":
        window = p["window"]
        thr = p["thr"]
        signal = bollinger_width(close, window) < thr
    elif cond_name == "ROC_STRONG":
        period = p["period"]
        thr = p["thr"]
        signal = roc(close, period) > thr
    elif cond_name == "DEV_ABS":
        l = p["long"]
        d = p["delta"]
        signal = (close / ema(close, l) - 1) > d
    else:
        raise ValueError("Unknown condition")

    strat_returns = daily_ret.copy()
    strat_returns[~signal] = 0.0

    n_trades = int(signal.sum())
    sharpe_ratio = sharpe(strat_returns.dropna())
    total_ret = (1 + strat_returns.fillna(0)).prod() - 1

    return {
        "condition": cond_name,
        "params": p,
        "sharpe": sharpe_ratio,
        "trades": n_trades,
        "total_return": total_ret,
    }

# --------------------------------------------------
# Build grid
# --------------------------------------------------
conds = []

# EMA_GAP combinations
for short in range(5, 46, 5):  # 5..45
    for long in (100, 150, 200):
        if long > short:
            for delta in (0.002, 0.005, 0.01):
                conds.append(("EMA_GAP", {"short": short, "long": long, "delta": delta}))

# ATR_LOW tighter
for period in (10, 14, 20, 30):
    for k in (0.6, 0.7, 0.8):
        conds.append(("ATR_LOW", {"period": period, "k": k}))

# Narrow Bollinger width
for window in (20, 30):
    for thr in (0.03, 0.04, 0.05):
        conds.append(("BOLL_NARROW", {"window": window, "thr": thr}))

# Strong ROC
for period in (5, 10, 20):
    for thr in (0.005, 0.01, 0.015):
        conds.append(("ROC_STRONG", {"period": period, "thr": thr}))

# Deviation above EMA long
for long in (100, 150, 200):
    for delta in (0.003, 0.005, 0.01, 0.02):
        conds.append(("DEV_ABS", {"long": long, "delta": delta}))

# Trim
conds = conds[:MAX_COMBOS]
print(f"Wave4: total test conditions = {len(conds)}")

# --------------------------------------------------
# Execute
# --------------------------------------------------
prices = get_price_data(TICKER, START_DATE, END_DATE)

results = [evaluate(prices, n, p) for n, p in conds]
res_df = pd.DataFrame(results)
res_df_sorted = res_df.sort_values(by=["sharpe", "trades"], ascending=[False, False])

print("\nTop 10 Wave4:")
print(res_df_sorted.head(10).to_string(index=False))

# log top3
with open("history.log", "a") as f:
    for _, row in res_df_sorted.head(3).iterrows():
        f.write(
            f"Wave4 | Condition: {row['condition']} | Params: {row['params']} | "
            f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n"
        )
print("Top 3 Wave4 appended to history.log")