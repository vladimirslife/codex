import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os

TICKER = "SPY"
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ANNUAL_RF = 0.02
RF_DAILY = ANNUAL_RF / 252.0
MAX_COMBINATIONS = 1000  # safeguard

# ---------------------------------------------
# Data utilities
# ---------------------------------------------

def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache_path = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

# ---------------------------------------------
# Indicator helpers (no look-ahead)
# ---------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def roc(series: pd.Series, period: int):
    return series.pct_change(period)


def atr(df: pd.DataFrame, period: int):
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


def bollinger_bands(series: pd.Series, window: int, num_std: float):
    ma = sma(series, window)
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower

# ---------------------------------------------
# Performance metrics
# ---------------------------------------------

def sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    if returns.std(ddof=0) == 0 or np.isclose(returns.std(ddof=0), 0):
        return 0.0
    excess = returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / excess.std(ddof=0)

# ---------------------------------------------
# Condition evaluation
# ---------------------------------------------

def evaluate(df: pd.DataFrame, name: str, params: dict):
    close = df["Close"]
    open_next = df["Open"].shift(-1)
    returns = (open_next - close) / close

    signal = pd.Series(False, index=df.index)

    if name == "EMA_CROSS":
        short = params["short"]
        long = params["long"]
        signal = ema(close, short) > ema(close, long)
    elif name == "ATR_LOW":
        period = params["period"]
        k = params["k"]
        atr_series = atr(df, period)
        atr_sma = atr_series.rolling(period * 2).mean()
        signal = atr_series < k * atr_sma
    elif name == "BOLL_WIDTH_LOW":
        window = params["window"]
        thresh = params["thresh"]
        upper, lower = bollinger_bands(close, window, 2)
        width = (upper - lower) / close
        signal = width < thresh
    elif name == "ROC_POS":
        period = params["period"]
        signal = roc(close, period) > 0
    elif name == "SMA_SLOPE_UP":
        period = params["period"]
        sma_series = sma(close, period)
        signal = sma_series > sma_series.shift(1)
    elif name == "RSI_GT":
        period = params["period"]
        signal = rsi(close, period) > 50
    elif name == "BREAKOUT_HIGH":
        period = params["period"]
        prev_high = close.rolling(period).max().shift(1)
        signal = close > prev_high
    else:
        raise ValueError("Unknown condition type")

    strat_returns = returns.copy()
    strat_returns[~signal] = 0.0

    n_trades = int(signal.sum())
    sharpe = sharpe_ratio(strat_returns.dropna())
    total_ret = (1 + strat_returns.fillna(0)).prod() - 1

    return {
        "condition": name,
        "params": params,
        "sharpe": sharpe,
        "trades": n_trades,
        "total_return": total_ret,
    }

# ---------------------------------------------
# Build grid of conditions for Wave 3
# ---------------------------------------------
conditions = []

# EMA crossovers
short_periods = list(range(5, 50, 5))  # 5..45
long_periods = [50, 100, 150, 200]
for s in short_periods:
    for l in long_periods:
        if l > s:
            conditions.append(("EMA_CROSS", {"short": s, "long": l}))

# ATR low volatility
for period in (10, 14, 20, 30):
    for k in (0.8, 0.9):
        conditions.append(("ATR_LOW", {"period": period, "k": k}))

# Bollinger width narrow
for window in (20, 30):
    for thresh in (0.05, 0.06, 0.07):
        conditions.append(("BOLL_WIDTH_LOW", {"window": window, "thresh": thresh}))

# ROC positive
for period in (5, 10, 15, 20, 30):
    conditions.append(("ROC_POS", {"period": period}))

# SMA slope up
for period in (10, 20, 50, 200):
    conditions.append(("SMA_SLOPE_UP", {"period": period}))

# RSI above neutral (bullish momentum)
for period in (5, 14, 21, 30):
    conditions.append(("RSI_GT", {"period": period}))

# Breakout higher close than previous N-day max
for period in (50, 100, 200):
    conditions.append(("BREAKOUT_HIGH", {"period": period}))

# Trim to safeguard
conditions = conditions[:MAX_COMBINATIONS]
print(f"Wave3: total conditions = {len(conditions)}")

# ---------------------------------------------
# Run evaluation
# ---------------------------------------------
price_data = download_price_data(TICKER, START_DATE, END_DATE)

results = [evaluate(price_data, name, p) for name, p in conditions]
res_df = pd.DataFrame(results)
res_df_sorted = res_df.sort_values(by=["sharpe", "trades"], ascending=[False, False])

print("\nTop 10 Wave3:")
print(res_df_sorted.head(10).to_string(index=False))

# Append top 3 to history.log
log_lines = []
for _, row in res_df_sorted.head(3).iterrows():
    log_lines.append(
        f"Wave3 | Condition: {row['condition']} | Params: {row['params']} | Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}"
    )

with open("history.log", "a") as f:
    f.write("\n".join(log_lines) + "\n")

print("Top 3 Wave3 appended to history.log")