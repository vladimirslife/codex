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

def roc(series, period):
    return series.pct_change(period)

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

def vwap_deviation_zscore(df, period):
    """Z-score of price deviation from VWAP"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
    deviation = (df['Close'] - vwap) / vwap
    mean_dev = deviation.rolling(period).mean()
    std_dev = deviation.rolling(period).std()
    return (deviation - mean_dev) / std_dev

def volume_weighted_momentum(df, period):
    """Momentum weighted by relative volume"""
    close = df['Close']
    volume = df['Volume']
    avg_volume = volume.rolling(period).mean()
    rel_volume = volume / avg_volume
    momentum = roc(close, period)
    return momentum * rel_volume

def vwap_bands(df, period, multiplier):
    """VWAP with bands based on volume-weighted standard deviation"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
    
    # Volume-weighted standard deviation
    squared_dev = ((typical_price - vwap) ** 2) * df['Volume']
    vw_variance = squared_dev.rolling(period).sum() / df['Volume'].rolling(period).sum()
    vw_std = np.sqrt(vw_variance)
    
    upper_band = vwap + multiplier * vw_std
    lower_band = vwap - multiplier * vw_std
    
    return vwap, upper_band, lower_band

def adaptive_vwap(df, min_period, max_period):
    """VWAP with adaptive period based on volatility"""
    close = df['Close']
    volume = df['Volume']
    
    # Use ATR as volatility measure
    vol = atr(df, 20) / close
    vol_rank = vol.rolling(100).rank(pct=True)
    
    # High volatility = shorter period, Low volatility = longer period
    adaptive_period = min_period + (max_period - min_period) * (1 - vol_rank)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = pd.Series(index=df.index, dtype=float)
    
    for i in range(max_period, len(df)):
        if not np.isnan(adaptive_period.iloc[i]):
            period = max(1, int(adaptive_period.iloc[i]))
            start_idx = max(0, i - period + 1)
            tp_slice = typical_price.iloc[start_idx:i+1]
            vol_slice = volume.iloc[start_idx:i+1]
            if vol_slice.sum() > 0:
                vwap.iloc[i] = (tp_slice * vol_slice).sum() / vol_slice.sum()
    
    return vwap

def volume_flow_index(df, period):
    """Modified Money Flow Index using volume flow"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_sum = positive_flow.rolling(period).sum()
    negative_sum = negative_flow.rolling(period).sum()
    
    flow_ratio = positive_sum / (positive_sum + negative_sum)
    return flow_ratio

def price_volume_oscillator(df, short, long):
    """Oscillator based on price-volume relationship"""
    pv = df['Close'] * df['Volume']
    short_ma = pv.rolling(short).mean()
    long_ma = pv.rolling(long).mean()
    return (short_ma - long_ma) / long_ma

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
    
    if cond_name == 'VWAP_ZSCORE':
        period = params['period']
        k = params['k']
        zscore = vwap_deviation_zscore(df, period)
        signal = zscore < k  # Buy when significantly below VWAP
    
    elif cond_name == 'VOL_WEIGHTED_MOM':
        period = params['period']
        k = params['k']
        vw_mom = volume_weighted_momentum(df, period)
        signal = vw_mom > k
    
    elif cond_name == 'VWAP_BAND_LOWER':
        period = params['period']
        mult = params['mult']
        vwap, upper, lower = vwap_bands(df, period, mult)
        signal = close < lower  # Buy at lower band
    
    elif cond_name == 'ADAPTIVE_VWAP_DEV':
        min_p = params['min_period']
        max_p = params['max_period']
        k = params['k']
        avwap = adaptive_vwap(df, min_p, max_p)
        signal = (close / avwap - 1) < k  # Buy below adaptive VWAP
    
    elif cond_name == 'VOL_FLOW_INDEX':
        period = params['period']
        k = params['k']
        vfi = volume_flow_index(df, period)
        signal = vfi > k
    
    elif cond_name == 'PV_OSCILLATOR':
        short = params['short']
        long = params['long']
        k = params['k']
        pvo = price_volume_oscillator(df, short, long)
        signal = pvo > k
    
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

# VWAP_ZSCORE - based on Wave 17 insights
for period in (50, 100, 150, 200):
    for k in (-2.0, -1.5, -1.0, -0.5):
        conditions.append(('VWAP_ZSCORE', {'period': period, 'k': k}))

# VOL_WEIGHTED_MOM
for period in (10, 20, 30):
    for k in (0.0, 0.01, 0.02):
        conditions.append(('VOL_WEIGHTED_MOM', {'period': period, 'k': k}))

# VWAP_BAND_LOWER
for period in (50, 100, 200):
    for mult in (1.0, 1.5, 2.0, 2.5):
        conditions.append(('VWAP_BAND_LOWER', {'period': period, 'mult': mult}))

# ADAPTIVE_VWAP_DEV
for min_p in (20, 50):
    for max_p in (100, 150, 200):
        if max_p > min_p:
            for k in (-0.02, -0.015, -0.01, -0.005):
                conditions.append(('ADAPTIVE_VWAP_DEV', {'min_period': min_p, 'max_period': max_p, 'k': k}))

# VOL_FLOW_INDEX
for period in (14, 21, 28):
    for k in (0.45, 0.5, 0.55):
        conditions.append(('VOL_FLOW_INDEX', {'period': period, 'k': k}))

# PV_OSCILLATOR
for short in (5, 10):
    for long in (20, 50):
        for k in (0.0, 0.02, 0.05):
            conditions.append(('PV_OSCILLATOR', {'short': short, 'long': long, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave18: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
results = [evaluate(prices, name, params) for name, params in conditions]
res_df = pd.DataFrame(results)
res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])

print("\nTop 10 Wave18:")
print(res_sorted.head(10).to_string(index=False))

# Log top 3
with open('history.log', 'a') as f:
    for _, row in res_sorted.head(3).iterrows():
        f.write(f"Wave18 | Condition: {row['condition']} | Params: {row['params']} | "
                f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")

print("Top 3 Wave18 appended to history.log")