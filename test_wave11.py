import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math, os

TICKER = "SPY"
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RF = 0.02
RF_DAILY = RF / 252.0
MAX_COMBOS = 1000

# ---------------- Data ----------------

def load(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache)
    return df.apply(pd.to_numeric, errors="coerce")

# ---------------- Indicators ----------

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def roc(series: pd.Series, period: int):
    return series.pct_change(period)

def atr(df: pd.DataFrame, period: int):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore(series: pd.Series, window: int):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def percentile(series: pd.Series, window: int):
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    return (series - roll_min) / (roll_max - roll_min).replace(0, np.nan)

# ---------------- Metric --------------

def sharpe(ret: pd.Series):
    if ret.empty:
        return 0.0
    std = ret.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    return math.sqrt(252) * (ret - RF_DAILY).mean() / std

# ---------------- Evaluation ----------

def evaluate(df: pd.DataFrame, name: str, params: dict):
    close = df["Close"]
    open_next = df["Open"].shift(-1)
    daily_ret = (open_next - close) / close

    if name == "DEV_VOL_RATIO":
        long = params["long"]
        period = params["atr"]
        k = params["k"]
        dev = close / ema(close, long) - 1
        vol = atr(df, period) / close
        ratio = dev / vol.replace(0, np.nan)
        signal = ratio > k
    elif name == "MOM_VOL_PCTL":
        period = params["period"]
        window = params["window"]
        thr = params["thr"]
        mom_vol = roc(close, period) / (atr(df, period) / close)
        signal = percentile(mom_vol, window) > thr
    elif name == "REL_STRENGTH_RATIO":
        short = params["short"]
        long = params["long"]
        thr = params["thr"]
        rs = roc(close, short) / roc(close, long).replace(0, np.nan)
        signal = rs > thr
    elif name == "COMPOSITE_Z":
        period = params["period"]
        window = params["window"]
        k = params["k"]
        comp = roc(close, period) + (close / ema(close, 200) - 1)
        signal = zscore(comp, window) > k
    else:
        raise ValueError("Unknown condition")

    strat = daily_ret.copy()
    strat[~signal] = 0.0

    trades = int(signal.sum())
    sr = sharpe(strat.dropna())
    total = (1 + strat.fillna(0)).prod() - 1

    return {"condition": name, "params": params, "sharpe": sr, "trades": trades, "total_return": total}

# ---------------- Grid -----------------
conds = []
# DEV_VOL_RATIO grid
for long in (150, 200):
    for atr_p in (14, 20):
        for k in (1.0, 1.2, 1.5):
            conds.append(("DEV_VOL_RATIO", {"long": long, "atr": atr_p, "k": k}))
# MOM_VOL_PCTL grid
for period in (5, 10, 20):
    for window in (50, 100):
        for thr in (0.8, 0.85, 0.9):
            conds.append(("MOM_VOL_PCTL", {"period": period, "window": window, "thr": thr}))
# REL_STRENGTH_RATIO grid
for short in (5, 10):
    for long in (50, 100):
        if long > short:
            for thr in (1.0, 1.2, 1.4):
                conds.append(("REL_STRENGTH_RATIO", {"short": short, "long": long, "thr": thr}))
# COMPOSITE_Z grid
for period in (5, 10, 20):
    for window in (50, 100):
        for k in (1.0, 1.5, 2.0):
            conds.append(("COMPOSITE_Z", {"period": period, "window": window, "k": k}))

conds = conds[:MAX_COMBOS]
print(f"Wave11: total conditions {len(conds)}")

# --------------- Run -------------------
prices = load(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, n, p) for n, p in conds]
res = pd.DataFrame(results)
res_sorted = res.sort_values(["sharpe", "trades"], ascending=[False, False])

print("\nTop 10 Wave11:")
print(res_sorted.head(10).to_string(index=False))

with open("history.log", "a") as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(
            f"Wave11 | Condition: {row['condition']} | Params: {row['params']} | "
            f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n"
        )
print("Top 3 Wave11 appended to history.log")