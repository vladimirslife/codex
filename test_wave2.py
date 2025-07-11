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
MAX_COMBINATIONS = 1000

# Helper functions

def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache_path = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def bollinger_lower(series: pd.Series, window: int, num_std: float):
    ma = sma(series, window)
    std = series.rolling(window=window).std()
    return ma - num_std * std


def macd(series: pd.Series, fast: int, slow: int) -> pd.Series:
    return ema(series, fast) - ema(series, slow)


def sharpe_ratio(returns: pd.Series, rf_daily: float = RF_DAILY) -> float:
    if returns.empty or returns.std(ddof=0) == 0 or np.isclose(returns.std(ddof=0), 0):
        return 0.0
    excess = returns - rf_daily
    return math.sqrt(252) * excess.mean() / excess.std(ddof=0)


def evaluate(df: pd.DataFrame, cond_name: str, params: dict):
    close = df["Close"]
    open_next = df["Open"].shift(-1)
    returns = (open_next - close) / close

    signal = pd.Series(False, index=df.index)

    if cond_name == "RSI_LT":
        series = rsi(close, params["period"])
        signal = series < params["threshold"]
    elif cond_name == "PRICE_GT_SMA":
        series = sma(close, params["period"])
        signal = close > series
    elif cond_name == "PRICE_GT_EMA":
        series = ema(close, params["period"])
        signal = close > series
    elif cond_name == "CLOSE_LT_BBL":
        lower = bollinger_lower(close, params["period"], params["std"])
        signal = close < lower
    elif cond_name == "MACD_POS":
        diff = macd(close, params["fast"], params["slow"])
        signal = diff > 0
    else:
        raise ValueError("Unknown condition")

    strat_returns = returns.copy()
    strat_returns[~signal] = 0.0

    n_trades = int(signal.sum())
    sharpe = sharpe_ratio(strat_returns.dropna())
    total_ret = (1 + strat_returns.fillna(0)).prod() - 1

    return {
        "condition": cond_name,
        "params": params,
        "sharpe": sharpe,
        "trades": n_trades,
        "total_return": total_ret,
    }

# Build grid
conditions = []

# RSI oversold
for period in range(5, 31, 5):
    for threshold in (10, 15, 20, 25, 30):
        conditions.append(("RSI_LT", {"period": period, "threshold": threshold}))

# Price above short SMA
for period in range(5, 61, 5):
    conditions.append(("PRICE_GT_SMA", {"period": period}))

# Bollinger Band oversold
for period in (10, 20, 30):
    for std in (1.5, 2.0, 2.5):
        conditions.append(("CLOSE_LT_BBL", {"period": period, "std": std}))

# MACD positive
for fast in (8, 12, 16):
    for slow in (20, 26, 32):
        if slow > fast:
            conditions.append(("MACD_POS", {"fast": fast, "slow": slow}))

# Trim to MAX
conditions = conditions[:MAX_COMBINATIONS]
print(f"Wave2: Total conditions generated: {len(conditions)}")

price_data = download_price_data(TICKER, START_DATE, END_DATE)

results = [evaluate(price_data, n, p) for n, p in conditions]
res_df = pd.DataFrame(results)
res_df_sorted = res_df.sort_values(by=["sharpe", "trades"], ascending=[False, False])

print("\nTop 10 Wave2:")
print(res_df_sorted.head(10).to_string(index=False))

log_lines = []
for _, row in res_df_sorted.head(3).iterrows():
    log_lines.append(
        f"Wave2 | Condition: {row['condition']} | Params: {row['params']} | Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}"
    )
with open("history.log", "a") as f:
    f.write("\n".join(log_lines) + "\n")
print("Top 3 Wave2 appended to history.log")