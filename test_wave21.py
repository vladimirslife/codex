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
    if df is not None and not df.empty:
        df.to_csv(cache_file)
    return df.apply(pd.to_numeric, errors='coerce') if df is not None else pd.DataFrame()

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

def momentum_volatility_ratio(df, mom_period, vol_period):
    """Momentum divided by volatility - risk-adjusted momentum"""
    close = df['Close']
    returns = close.pct_change()
    
    momentum = close.pct_change(mom_period)
    volatility = returns.rolling(vol_period).std() * np.sqrt(252)
    
    ratio = momentum / volatility.where(volatility > 0, np.nan)
    return ratio

def volatility_breakout(df, period, multiplier):
    """Detect volatility expansion beyond normal range"""
    close = df['Close']
    returns = close.pct_change()
    
    vol = returns.rolling(period).std()
    vol_ma = vol.rolling(period * 2).mean()
    vol_std = vol.rolling(period * 2).std()
    
    # Z-score of volatility
    vol_zscore = (vol - vol_ma) / vol_std.where(vol_std > 0, np.nan)
    return vol_zscore

def momentum_acceleration(df, short_period, long_period):
    """Rate of change of momentum"""
    close = df['Close']
    
    short_mom = close.pct_change(short_period)
    long_mom = close.pct_change(long_period)
    
    # Acceleration = change in momentum
    acceleration = (short_mom - short_mom.shift(short_period)) / short_period
    return acceleration

def volatility_adjusted_rsi(df, rsi_period, vol_period):
    """RSI adjusted by volatility regime"""
    close = df['Close']
    returns = close.pct_change()
    
    # Standard RSI calculation
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    
    rs = avg_gain / avg_loss.where(avg_loss > 0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Volatility adjustment
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Adjust RSI thresholds based on volatility regime
    adjusted_rsi = rsi * (1 + (vol_percentile - 0.5))
    return adjusted_rsi

def momentum_consistency(df, period, threshold):
    """Measure consistency of momentum direction"""
    close = df['Close']
    returns = close.pct_change()
    
    # Count positive returns in rolling window
    positive_days = returns.rolling(period).apply(lambda x: (x > threshold).sum())
    consistency_ratio = positive_days / period
    
    return consistency_ratio

def volatility_term_structure(df, short_vol, long_vol):
    """Ratio of short-term to long-term volatility"""
    close = df['Close']
    returns = close.pct_change()
    
    short_term_vol = returns.rolling(short_vol).std()
    long_term_vol = returns.rolling(long_vol).std()
    
    term_structure = short_term_vol / long_term_vol.where(long_term_vol > 0, np.nan)
    return term_structure

def momentum_quality(df, period):
    """Quality of momentum - smoothness of trend"""
    close = df['Close']
    
    # Linear regression over period
    x = np.arange(period)
    slopes = pd.Series(index=close.index, dtype=float)
    r_squared = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        y = close.iloc[i-period:i].values
        if len(y) == period:
            # Calculate slope and R-squared
            x_mean = x.mean()
            y_mean = y.mean()
            
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()
            
            if denominator > 0:
                slope = numerator / denominator
                slopes.iloc[i] = slope
                
                # R-squared calculation
                y_pred = slope * x + (y_mean - slope * x_mean)
                ss_res = ((y - y_pred) ** 2).sum()
                ss_tot = ((y - y_mean) ** 2).sum()
                
                if ss_tot > 0:
                    r_squared.iloc[i] = 1 - (ss_res / ss_tot)
    
    # Quality = slope * R-squared (momentum with high confidence)
    quality = slopes * r_squared
    return quality

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
    
    if cond_name == 'MOM_VOL_RATIO':
        mom_p = params['mom_period']
        vol_p = params['vol_period']
        k = params['k']
        ratio = momentum_volatility_ratio(df, mom_p, vol_p)
        signal = ratio > k
    
    elif cond_name == 'VOL_BREAKOUT':
        period = params['period']
        mult = params['mult']
        zscore = volatility_breakout(df, period, mult)
        signal = zscore > mult  # Buy on volatility expansion
    
    elif cond_name == 'MOM_ACCEL':
        short = params['short']
        long = params['long']
        k = params['k']
        accel = momentum_acceleration(df, short, long)
        signal = accel > k
    
    elif cond_name == 'VOL_ADJ_RSI':
        rsi_p = params['rsi_period']
        vol_p = params['vol_period']
        k = params['k']
        adj_rsi = volatility_adjusted_rsi(df, rsi_p, vol_p)
        signal = adj_rsi < k  # Buy on oversold
    
    elif cond_name == 'MOM_CONSISTENCY':
        period = params['period']
        thr = params['threshold']
        k = params['k']
        consistency = momentum_consistency(df, period, thr)
        signal = consistency < k  # Buy on low consistency (reversal)
    
    elif cond_name == 'VOL_TERM_STRUCT':
        short = params['short_vol']
        long = params['long_vol']
        k = params['k']
        term_struct = volatility_term_structure(df, short, long)
        signal = term_struct < k  # Buy when short vol < long vol
    
    elif cond_name == 'MOM_QUALITY':
        period = params['period']
        k = params['k']
        quality = momentum_quality(df, period)
        signal = quality > k  # Buy on high quality momentum
    
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

# MOM_VOL_RATIO - Risk-adjusted momentum
for mom_p in (10, 20, 50):
    for vol_p in (20, 50):
        for k in (0.0, 0.1, 0.2, 0.3):
            conditions.append(('MOM_VOL_RATIO', {'mom_period': mom_p, 'vol_period': vol_p, 'k': k}))

# VOL_BREAKOUT - Volatility expansion
for period in (20, 50):
    for mult in (1.0, 1.5, 2.0, 2.5):
        conditions.append(('VOL_BREAKOUT', {'period': period, 'mult': mult}))

# MOM_ACCEL - Momentum acceleration
for short in (5, 10):
    for long in (20, 50):
        for k in (0.0, 0.001, 0.002):
            conditions.append(('MOM_ACCEL', {'short': short, 'long': long, 'k': k}))

# VOL_ADJ_RSI - Volatility-adjusted RSI
for rsi_p in (14, 21):
    for vol_p in (20, 50):
        for k in (30, 40, 50):
            conditions.append(('VOL_ADJ_RSI', {'rsi_period': rsi_p, 'vol_period': vol_p, 'k': k}))

# MOM_CONSISTENCY - Momentum consistency
for period in (20, 50):
    for thr in (0.0, 0.001):
        for k in (0.4, 0.5, 0.6):
            conditions.append(('MOM_CONSISTENCY', {'period': period, 'threshold': thr, 'k': k}))

# VOL_TERM_STRUCT - Volatility term structure
for short in (10, 20):
    for long in (50, 100):
        for k in (0.8, 1.0, 1.2):
            conditions.append(('VOL_TERM_STRUCT', {'short_vol': short, 'long_vol': long, 'k': k}))

# MOM_QUALITY - Momentum quality
for period in (20, 50, 100):
    for k in (0.0, 0.001, 0.002):
        conditions.append(('MOM_QUALITY', {'period': period, 'k': k}))

conditions = conditions[:MAX_COMBOS]
print(f"Wave21: total conditions = {len(conditions)}")

# -------------------- Run Tests --------------------

prices = load_data(TICKER, START_DATE, END_DATE)
if not prices.empty:
    results = [evaluate(prices, name, params) for name, params in conditions]
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\nTop 10 Wave21:")
    print(res_sorted.head(10).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave21 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("Top 3 Wave21 appended to history.log")
else:
    print("Error: Unable to load data")