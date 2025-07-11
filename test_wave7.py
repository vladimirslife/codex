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
# Data loader
# --------------------------------------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache)
    return df.apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------
# Indicators
# --------------------------------------------------

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int):
    return series.rolling(window).mean()


def roc(series: pd.Series, period: int):
    return series.pct_change(period)


def zscore(series: pd.Series, window: int):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def percentile(series: pd.Series, window: int):
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return (series - roll_min) / denom


def atr(df: pd.DataFrame, period: int):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --------------------------------------------------
# Metrics
# --------------------------------------------------

def sharpe(ret: pd.Series):
    if ret.empty:
        return 0.0
    std = ret.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    excess = ret - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(df: pd.DataFrame, name: str, p: dict):
    close = df["Close"]
    open_next = df["Open"].shift(-1)
    day_ret = (open_next - close) / close

    if name == "ROC_Z":
        period = p["period"]
        window = p["window"]
        k = p["k"]
        signal = zscore(roc(close, period), window) > k
    elif name == "ATR_PCTL_LOW":
        period = p["period"]
        window = p["window"]
        thr = p["thr"]
        atr_pct = atr(df, period) / close
        signal = percentile(atr_pct, window) < thr
    elif name == "EMA_BASKET":
        delta = p["delta"]
        ema_short = ema(close, 20)
        ema_mid = ema(close, 50)
        ema_long = ema(close, 200)
        composite = (ema_short + ema_mid + ema_long) / 3
        signal = (close / composite - 1) > delta
    elif name == "MOM_PCTL":
        period = p["period"]
        window = p["window"]
        thr = p["thr"]
        mom = roc(close, period)
        signal = percentile(mom, window) > thr
    else:
        raise ValueError("Unknown condition")

    strat_ret = day_ret.copy()
    strat_ret[~signal] = 0.0

    trades = int(signal.sum())
    sr = sharpe(strat_ret.dropna())
    total_r = (1 + strat_ret.fillna(0)).prod() - 1

    return {"condition": name, "params": p, "sharpe": sr, "trades": trades, "total_return": total_r}

# --------------------------------------------------
# Build grid
# --------------------------------------------------
conds = []

# ROC Z-score filters
for period in (5, 10, 20):
    for window in (50, 100):
        for k in (1.0, 1.5, 2.0):
            conds.append(("ROC_Z", {"period": period, "window": window, "k": k}))

# ATR percentile low (volatility contraction)
for period in (14, 20):
    for window in (100, 200):
        for thr in (0.2, 0.3, 0.4):
            conds.append(("ATR_PCTL_LOW", {"period": period, "window": window, "thr": thr}))

# EMA basket ratio
for delta in (0.001, 0.002, 0.003, 0.004):
    conds.append(("EMA_BASKET", {"delta": delta}))

# Momentum percentile
for period in (5, 10, 20):
    for window in (50, 100):
        for thr in (0.8, 0.85, 0.9):
            conds.append(("MOM_PCTL", {"period": period, "window": window, "thr": thr}))

# trim
conds = conds[:MAX_COMBOS]
print(f"Wave7: total conditions = {len(conds)}")

# --------------------------------------------------
# Run tests
# --------------------------------------------------
prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, n, p) for n, p in conds]
res_df = pd.DataFrame(results)
res_df_sorted = res_df.sort_values(by=["sharpe", "trades"], ascending=[False, False])

print("\nTop 10 Wave7:")
print(res_df_sorted.head(10).to_string(index=False))

with open("history.log", "a") as f:
    for _, row in res_df_sorted.head(3).iterrows():
        f.write(
            f"Wave7 | Condition: {row['condition']} | Params: {row['params']} | "
            f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n"
        )
print("Top 3 Wave7 appended to history.log")