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

# -------------------- Data --------------------

def load_data(ticker, start, end):
    cache_file = f"{ticker}_{start}_{end}.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors='coerce')
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.to_csv(cache_file)
    return df.apply(pd.to_numeric, errors='coerce')

# -------------------- Indicators --------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window).mean()

def atr(df, period):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def intraday_momentum(df, period):
    """Momentum based on intraday range vs close position"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Where close is within the day's range (0 = low, 1 = high)
    daily_range = high - low
    close_position = (close - low) / daily_range.where(daily_range > 0, 1)
    
    # Momentum of close position
    momentum = close_position - close_position.rolling(period).mean()
    return momentum

def volume_price_correlation(df, period):
    """Rolling correlation between volume and price changes"""
    price_change = df['Close'].pct_change()
    volume_change = df['Volume'].pct_change()
    
    correlation = price_change.rolling(period).corr(volume_change)
    return correlation

def open_close_ratio(df, period):
    """Ratio of open-to-close returns vs close-to-close returns"""
    open_close_ret = (df['Close'] - df['Open']) / df['Open']
    close_close_ret = df['Close'].pct_change()
    
    oc_avg = open_close_ret.rolling(period).mean()
    cc_avg = close_close_ret.rolling(period).mean()
    
    ratio = oc_avg / cc_avg.where(cc_avg != 0, np.nan)
    return ratio

def high_low_spread(df, period):
    """Normalized high-low spread as volatility measure"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    spread = (high - low) / close
    normalized = (spread - spread.rolling(period).mean()) / spread.rolling(period).std()
    return normalized

def volume_surge_indicator(df, period, multiplier):
    """Detects volume surges relative to average"""
    volume = df['Volume']
    avg_volume = volume.rolling(period).mean()
    surge = volume / avg_volume
    return surge

def price_efficiency_ratio(df, period):
    """Kaufman's Efficiency Ratio - directional movement vs volatility"""
    close = df['Close']
    change = close - close.shift(period)
    volatility = (close - close.shift()).abs().rolling(period).sum()
    
    efficiency = change.abs() / volatility.where(volatility > 0, np.nan)
    return efficiency

def vwap_acceleration(df, period):
    """Rate of change in VWAP"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
    
    # Second derivative of VWAP
    vwap_change = vwap.pct_change()
    acceleration = vwap_change - vwap_change.shift(1)
    return acceleration

# -------------------- Sharpe --------------------

def sharpe_ratio(returns):
    if returns.empty:
        return 0.0
    std = returns.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    excess = returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

# -------------------- Evaluation --------------------

def evaluate(df, cond_name, params):
    close = df['Close']
    open_next = df['Open'].shift(-1)
    daily_returns = (open_next - close) / close
    
    signal = pd.Series(False, index=df.index)
    
    if cond_name == 'INTRADAY_MOM':
        period = params['period']
        k = params['k']
        momentum = intraday_momentum(df, period)
        signal = momentum < k  # Buy on low intraday momentum
    
    elif cond_name == 'VOL_PRICE_CORR':
        period = params['period']
        k = params['k']
        corr = volume_price_correlation(df, period)
        signal = corr < k  # Buy on negative correlation
    
    elif cond_name == 'OC_RATIO':
        period = params['period']
        k = params['k']
        ratio = open_close_ratio(df, period)
        signal = ratio < k
    
    elif cond_name == 'HL_SPREAD':
        period = params['period']
        k = params['k']
        spread = high_low_spread(df, period)
        signal = spread > k  # Buy on high normalized spread
    
    elif cond_name == 'VOL_SURGE':
        period = params['period']
        mult = params['mult']
        surge = volume_surge_indicator(df, period, mult)
        signal = surge > mult
    
    elif cond_name == 'PRICE_EFFICIENCY':
        period = params['period']
        k = params['k']
        efficiency = price_efficiency_ratio(df, period)
        signal = efficiency > k
    
    elif cond_name == 'VWAP_ACCEL':
        period = params['period']
        k = params['k']
        accel = vwap_acceleration(df, period)
        signal = accel < k  # Buy on negative acceleration
    
    else:
        raise ValueError(f"Unknown condition: {cond_name}")
    
    strategy_returns = daily_returns.copy()
    strategy_returns[~signal] = 0.0
    
    n_trades = int(signal.sum())
    sharpe = sharpe_ratio(strategy_returns.dropna())
    total_return = (1 + strategy_returns.fillna(0)).prod() - 1
    
    return {
        'condition': cond_name,
        'params': params,
        'sharpe': sharpe,
        'trades': n_trades,
        'total_return': total_return
    }

# -------------------- Grid Search --------------------

conditions = []

# INTRADAY_MOM
for period in (10, 20, 30, 50):
    for k in (-0.2, -0.1, 0.0, 0.1):
        conditions.append(('INTRADAY_MOM', {'period': period, 'k': k}))

# VOL_PRICE_CORR
for period in (20, 50, 100):
    for k in (-0.3, -0.2, -0.1, 0.0):
        conditions.append(('VOL_PRICE_CORR', {'period': period, 'k': k}))

# OC_RATIO
for period in (20, 50, 100):
    for k in (-1.0, -0.5, 0.0, 0.5):
        conditions.append(('OC_RATIO', {'period': period, 'k': k}))

# HL_SPREAD
for period in (20, 50, 100):
    for k in (0.5, 1.0, 1.5, 2.0):
        conditions.append(('HL_SPREAD', {'period': period, 'k': k}))

# VOL_SURGE
for period in (20, 50):
    for mult in (1.5, 2.0, 2.5, 3.0):
        conditions.append(('VOL_SURGE', {'period': period, 'mult': mult}))

# PRICE_EFFICIENCY
for period in (10, 20, 30):
    for k in (0.3, 0.4, 0.5, 0.6):
        conditions.append(('PRICE_EFFICIENCY', {'period': period, 'k': k}))

# VWAP_ACCEL
for period in (20, 50, 100):
    for k in (-0.001, -0.0005, 0.0, 0.0005):
        conditions.append(('VWAP_ACCEL', {'period': period, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave19: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave19:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave19 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave19 appended to history.log")