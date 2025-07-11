import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os

# -----------------------------
# Configurable parameters
# -----------------------------
TICKER = "SPY"  # broad market ETF, daily candles
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ANNUAL_RF = 0.02  # fixed per task
RF_DAILY = ANNUAL_RF / 252.0

# Maximum combinations – safeguard
MAX_COMBINATIONS = 1000

# -----------------------------
# Helper functions
# -----------------------------

def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV daily data using yfinance; cache locally for speed."""
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
    """Simple RSI implementation (no look-ahead)."""
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


def sharpe_ratio(returns: pd.Series, rf_daily: float = RF_DAILY) -> float:
    if returns.empty or returns.std(ddof=0) == 0 or np.isclose(returns.std(ddof=0), 0):
        return 0.0
    excess = returns - rf_daily
    return math.sqrt(252) * excess.mean() / excess.std(ddof=0)


def evaluate_condition(df: pd.DataFrame, condition_name: str, params: dict) -> dict:
    """Generate entry signals, compute next-day open returns, and performance metrics."""
    close = df["Close"]
    open_next = df["Open"].shift(-1)  # open on T+1
    returns = (open_next - close) / close  # raw overnight return

    signal = pd.Series(False, index=df.index)

    if condition_name == "RSI_LT":
        rsi_series = rsi(close, params["period"])
        signal = rsi_series < params["threshold"]
    elif condition_name == "PRICE_GT_SMA":
        sma_series = sma(close, params["period"])
        signal = close > sma_series
    elif condition_name == "PRICE_GT_EMA":
        ema_series = ema(close, params["period"])
        signal = close > ema_series
    else:
        raise ValueError(f"Unknown condition {condition_name}")

    # Strategy daily returns: apply return * signal (lag 0, entry at close T, exit next open)
    strat_returns = returns.copy()
    strat_returns[~signal] = 0.0

    n_trades = int(signal.sum())
    sharpe = sharpe_ratio(strat_returns.dropna())
    total_ret = (1 + strat_returns.fillna(0)).prod() - 1

    return {
        "condition": condition_name,
        "params": params,
        "sharpe": sharpe,
        "trades": n_trades,
        "total_return": total_ret,
    }

# -----------------------------
# Generate grid of conditions (up to MAX_COMBINATIONS)
# -----------------------------
conditions_grid = []

# 1) RSI < threshold
for period in range(5, 31, 5):  # 5,10,...,30
    for threshold in (20, 25, 30):
        conditions_grid.append(("RSI_LT", {"period": period, "threshold": threshold}))

# 2) Price > SMA(period)
for period in range(10, 201, 10):
    conditions_grid.append(("PRICE_GT_SMA", {"period": period}))

# 3) Price > EMA(period)
for period in range(10, 201, 10):
    conditions_grid.append(("PRICE_GT_EMA", {"period": period}))

# Trim to MAX_COMBINATIONS if necessary
conditions_grid = conditions_grid[:MAX_COMBINATIONS]

print(f"Total conditions generated: {len(conditions_grid)}")

# -----------------------------
# Main evaluation loop
# -----------------------------
price_data = download_price_data(TICKER, START_DATE, END_DATE)

results = []
for cond_name, params in conditions_grid:
    res = evaluate_condition(price_data, cond_name, params)
    results.append(res)

# Convert to DataFrame for easy sorting
res_df = pd.DataFrame(results)
res_df_sorted = res_df.sort_values(by=["sharpe", "trades"], ascending=[False, False])

# Output summary
print("\nTop 10 strategies by Sharpe Ratio (Wave 1):")
print(res_df_sorted.head(10).to_string(index=False))

# Log top 3 to history.log
history_lines = []
for _, row in res_df_sorted.head(3).iterrows():
    line = (
        f"Wave1 | Condition: {row['condition']} | Params: {row['params']} | "
        f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}"
    )
    history_lines.append(line)

with open("history.log", "a") as f:
    f.write("\n".join(history_lines) + "\n")

print("\nTop 3 entries appended to history.log")