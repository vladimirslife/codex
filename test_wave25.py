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

# -------------------- Data Loading --------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load data with proper error handling"""
    cache_file = f"{ticker}_{start}_{end}.csv"
    
    try:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if isinstance(df, pd.DataFrame):
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
    except Exception as e:
        print(f"Error reading cache: {e}")
    
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(data, pd.DataFrame) and not data.empty:
            df = data.copy()
            df.to_csv(cache_file)
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except Exception as e:
        print(f"Error downloading data: {e}")
    
    return pd.DataFrame()

# -------------------- DeMark Indicators --------------------

def demark_sequential(df: pd.DataFrame, setup_period: int = 9) -> pd.Series:
    """DeMark Sequential indicator"""
    close = df['Close']
    
    # Setup phase
    setup_buy = pd.Series(0, index=close.index)
    setup_sell = pd.Series(0, index=close.index)
    
    for i in range(4, len(close)):
        # Buy setup: close < close[4 bars ago]
        if close.iloc[i] < close.iloc[i-4]:
            setup_buy.iloc[i] = setup_buy.iloc[i-1] + 1 if setup_buy.iloc[i-1] > 0 else 1
        else:
            setup_buy.iloc[i] = 0
            
        # Sell setup: close > close[4 bars ago]
        if close.iloc[i] > close.iloc[i-4]:
            setup_sell.iloc[i] = setup_sell.iloc[i-1] + 1 if setup_sell.iloc[i-1] > 0 else 1
        else:
            setup_sell.iloc[i] = 0
    
    # Combined signal
    signal = setup_buy - setup_sell
    
    # Normalize by range
    signal_norm = signal / setup_period
    
    return signal_norm

def demark_pressure(df: pd.DataFrame, period: int) -> pd.Series:
    """DeMark Pressure Ratio"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Buying pressure
    bp = close - low.rolling(period).min()
    
    # Selling pressure
    sp = high.rolling(period).max() - close
    
    # Pressure ratio
    pressure = bp / (bp + sp).where((bp + sp) > 0, 1)
    
    # Trend adjustment
    trend = close / close.shift(period) - 1
    
    # Combined signal
    signal = pressure * (1 + trend)
    
    return signal

# -------------------- Williams VIX Fix --------------------

def williams_vix_fix(df: pd.DataFrame, period: int = 22) -> pd.Series:
    """Williams VIX Fix - synthetic VIX"""
    close = df['Close']
    low = df['Low']
    
    # Highest close over period
    highest_close = close.rolling(period).max()
    
    # VIX Fix calculation
    vix_fix = ((highest_close - low) / highest_close) * 100
    
    # Percentile rank
    vix_percentile = vix_fix.rolling(period * 3).rank(pct=True)
    
    # Inverted signal (high VIX = fear = buying opportunity)
    signal = 1 - vix_percentile
    
    return signal

def williams_ultimate_oscillator(df: pd.DataFrame, p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Williams Ultimate Oscillator with modifications"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True range
    prev_close = close.shift(1)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - prev_close).abs(),
        'lc': (low - prev_close).abs()
    }).max(axis=1)
    
    # Buying pressure
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    
    # Average over different periods
    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    
    # Weighted average
    uo = (4 * avg1 + 2 * avg2 + avg3) / 7
    
    # Trend adjustment
    trend = close / close.shift(p2) - 1
    signal = uo * (1 + trend * 5)
    
    return signal

# -------------------- Elder Force Index --------------------

def elder_force_index(df: pd.DataFrame, period: int) -> pd.Series:
    """Elder Force Index with enhancements"""
    close = df['Close']
    volume = df['Volume']
    
    # Raw force index
    force = (close - close.shift(1)) * volume
    
    # Smoothed force
    force_ema = force.ewm(span=period, adjust=False).mean()
    
    # Normalize by average volume
    avg_volume = volume.rolling(period * 2).mean()
    force_norm = force_ema / avg_volume.where(avg_volume > 0, 1)
    
    # Add acceleration component
    force_accel = force_norm.diff(period // 2)
    
    # Combined signal
    signal = force_norm + force_accel * 0.5
    
    return signal

def elder_impulse_system(df: pd.DataFrame, ema_period: int, macd_fast: int = 12, macd_slow: int = 26) -> pd.Series:
    """Elder Impulse System"""
    close = df['Close']
    
    # EMA slope
    ema = close.ewm(span=ema_period, adjust=False).mean()
    ema_slope = ema - ema.shift(1)
    
    # MACD histogram
    exp1 = close.ewm(span=macd_fast, adjust=False).mean()
    exp2 = close.ewm(span=macd_slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal_line
    macd_hist_slope = macd_hist - macd_hist.shift(1)
    
    # Impulse: both rising = bullish
    impulse = ((ema_slope > 0) & (macd_hist_slope > 0)).astype(float)
    
    # Add strength component
    strength = (ema_slope / close * 100) * (macd_hist_slope / close * 1000)
    
    # Combined signal
    signal = impulse + strength.clip(-1, 1)
    
    return signal

# -------------------- Innovative Momentum Measures --------------------

def momentum_quality_index(df: pd.DataFrame, period: int) -> pd.Series:
    """Momentum Quality Index - consistency of momentum"""
    close = df['Close']
    returns = close.pct_change()
    
    # Rolling momentum
    momentum = close / close.shift(period) - 1
    
    # Quality metrics
    # 1. Directional consistency
    positive_days = (returns > 0).rolling(period).sum()
    consistency = (positive_days / period - 0.5) * 2
    
    # 2. Smoothness (inverse of volatility)
    volatility = returns.rolling(period).std()
    smoothness = 1 / (1 + volatility * np.sqrt(252))
    
    # 3. Acceleration
    accel = momentum.diff(period // 4)
    
    # Combined quality score
    quality = momentum * consistency * smoothness * (1 + accel * 10)
    
    return quality

def adaptive_momentum_oscillator(df: pd.DataFrame, min_period: int, max_period: int) -> pd.Series:
    """Adaptive Momentum Oscillator"""
    close = df['Close']
    returns = close.pct_change()
    
    # Market efficiency ratio (Kaufman)
    direction = (close - close.shift(min_period)).abs()
    volatility = returns.abs().rolling(min_period).sum()
    efficiency = direction / volatility.where(volatility > 0, 1)
    
    # Adaptive period
    period = (min_period + (max_period - min_period) * (1 - efficiency)).astype(int)
    
    # Calculate adaptive momentum
    momentum = pd.Series(index=close.index, dtype=float)
    for i in range(max_period, len(close)):
        p = int(period.iloc[i]) if not pd.isna(period.iloc[i]) else min_period
        p = min(max(p, min_period), max_period)
        momentum.iloc[i] = close.iloc[i] / close.iloc[i-p] - 1
    
    # Normalize with adaptive bands
    mom_std = momentum.rolling(max_period).std()
    mom_mean = momentum.rolling(max_period).mean()
    signal = (momentum - mom_mean) / mom_std.where(mom_std > 0, 1)
    
    return signal

# -------------------- Cyclical Components --------------------

def hilbert_transform_cycle(df: pd.DataFrame, period: int) -> pd.Series:
    """Hilbert Transform for cycle detection"""
    close = df['Close']
    
    # Detrend price
    ma = close.rolling(period).mean()
    detrended = close - ma
    
    # Hilbert Transform (simplified)
    # Real part
    real = detrended
    
    # Imaginary part (90-degree phase shift)
    imag = pd.Series(index=close.index, dtype=float)
    for i in range(period, len(close)):
        weights = np.array([1 if j % 2 == 1 else 0 for j in range(period)])
        weights = weights / weights.sum()
        imag.iloc[i] = np.sum(detrended.iloc[i-period:i].values * weights)
    
    # Phase angle
    phase = np.arctan2(imag, real)
    
    # Rate of change of phase (instantaneous frequency)
    freq = phase.diff()
    
    # Cycle indicator
    cycle = np.sin(phase) * (1 + freq.abs())
    
    return cycle

def fourier_dominant_cycle(df: pd.DataFrame, window: int) -> pd.Series:
    """Fourier analysis for dominant cycle"""
    close = df['Close']
    returns = close.pct_change()
    
    # Rolling FFT to find dominant frequency
    dominant_period = pd.Series(index=close.index, dtype=float)
    cycle_strength = pd.Series(index=close.index, dtype=float)
    
    for i in range(window, len(close)):
        # Get window of returns
        window_data = returns.iloc[i-window:i].values
        
        # Remove NaN
        window_data = window_data[~np.isnan(window_data)]
        
        if len(window_data) > window // 2:
            # Apply FFT
            fft = np.fft.fft(window_data)
            freqs = np.fft.fftfreq(len(window_data))
            
            # Find dominant frequency (excluding DC component)
            magnitudes = np.abs(fft[1:len(fft)//2])
            if len(magnitudes) > 0:
                dominant_idx = np.argmax(magnitudes) + 1
                dominant_period.iloc[i] = 1 / freqs[dominant_idx] if freqs[dominant_idx] != 0 else window
                cycle_strength.iloc[i] = magnitudes[dominant_idx - 1] / np.sum(magnitudes)
    
    # Generate cycle signal
    cycle_signal = pd.Series(index=close.index, dtype=float)
    for i in range(window, len(close)):
        if not pd.isna(dominant_period.iloc[i]):
            period = int(dominant_period.iloc[i])
            phase = 2 * np.pi * (i % period) / period
            cycle_signal.iloc[i] = np.sin(phase) * cycle_strength.iloc[i]
    
    return cycle_signal

# -------------------- Advanced Scaling Methods --------------------

def rank_normalization(series: pd.Series, window: int) -> pd.Series:
    """Rank-based normalization"""
    return series.rolling(window).rank(pct=True) * 2 - 1

def robust_zscore(series: pd.Series, window: int) -> pd.Series:
    """Robust Z-score using median and MAD"""
    median = series.rolling(window).median()
    mad = (series - median).abs().rolling(window).median()
    return (series - median) / mad.where(mad > 0, 1)

def arcsinh_transform(series: pd.Series) -> pd.Series:
    """Inverse hyperbolic sine transformation"""
    return np.arcsinh(series * 10)

# -------------------- Evaluation Functions --------------------

def sharpe_ratio(returns: pd.Series) -> float:
    """Calculate Sharpe ratio"""
    if returns.empty or len(returns) < 20:
        return 0.0
    
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    
    std = clean_returns.std(ddof=1)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    
    excess = clean_returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

def evaluate(df: pd.DataFrame, cond_name: str, params: dict) -> dict:
    """Evaluate strategy"""
    if df.empty:
        return {
            'condition': cond_name,
            'params': params,
            'sharpe': 0.0,
            'trades': 0,
            'total_return': 0.0
        }
    
    close = df['Close']
    open_next = df['Open'].shift(-1)
    daily_returns = (open_next - close) / close
    
    signal = pd.Series(False, index=df.index)
    
    try:
        if cond_name == 'DEMARK_SEQ':
            setup_period = params['setup_period']
            k = params['k']
            indicator = demark_sequential(df, setup_period)
            signal = indicator > k
        
        elif cond_name == 'DEMARK_PRESSURE':
            period = params['period']
            k = params['k']
            indicator = demark_pressure(df, period)
            signal = indicator > k
        
        elif cond_name == 'WILLIAMS_VIX_FIX':
            period = params['period']
            k = params['k']
            indicator = williams_vix_fix(df, period)
            signal = indicator > k
        
        elif cond_name == 'WILLIAMS_ULT_OSC':
            p1 = params['p1']
            p2 = params['p2']
            p3 = params['p3']
            k = params['k']
            indicator = williams_ultimate_oscillator(df, p1, p2, p3)
            signal = indicator > k
        
        elif cond_name == 'ELDER_FORCE':
            period = params['period']
            k = params['k']
            indicator = elder_force_index(df, period)
            signal = indicator > k
        
        elif cond_name == 'ELDER_IMPULSE':
            ema_period = params['ema_period']
            k = params['k']
            indicator = elder_impulse_system(df, ema_period)
            signal = indicator > k
        
        elif cond_name == 'MOM_QUALITY':
            period = params['period']
            k = params['k']
            indicator = momentum_quality_index(df, period)
            signal = indicator > k
        
        elif cond_name == 'ADAPTIVE_MOM_OSC':
            min_p = params['min_period']
            max_p = params['max_period']
            k = params['k']
            indicator = adaptive_momentum_oscillator(df, min_p, max_p)
            signal = indicator > k
        
        elif cond_name == 'HILBERT_CYCLE':
            period = params['period']
            k = params['k']
            indicator = hilbert_transform_cycle(df, period)
            signal = indicator > k
        
        elif cond_name == 'FOURIER_CYCLE':
            window = params['window']
            k = params['k']
            indicator = fourier_dominant_cycle(df, window)
            signal = indicator > k
        
        # Combined indicators with different normalizations
        elif cond_name == 'ELDER_FORCE_RANK':
            period = params['period']
            k = params['k']
            raw = elder_force_index(df, period)
            indicator = rank_normalization(raw, period * 2)
            signal = indicator > k
        
        elif cond_name == 'MOM_QUALITY_ROBUST':
            period = params['period']
            k = params['k']
            raw = momentum_quality_index(df, period)
            indicator = robust_zscore(raw, period)
            signal = indicator > k
        
        elif cond_name == 'DEMARK_PRESSURE_ARCSINH':
            period = params['period']
            k = params['k']
            raw = demark_pressure(df, period)
            indicator = arcsinh_transform(raw)
            signal = indicator > k
        
        else:
            raise ValueError(f"Unknown condition: {cond_name}")
    
    except Exception as e:
        print(f"Error in {cond_name}: {e}")
        signal = pd.Series(False, index=df.index)
    
    # Apply signal
    strategy_returns = daily_returns.copy()
    strategy_returns[~signal] = 0.0
    
    # Calculate metrics
    n_trades = int(signal.sum())
    sharpe = sharpe_ratio(strategy_returns.dropna())
    
    try:
        total_return = (1 + strategy_returns.fillna(0)).prod() - 1
    except:
        total_return = 0.0
    
    return {
        'condition': cond_name,
        'params': params,
        'sharpe': sharpe,
        'trades': n_trades,
        'total_return': total_return
    }

# -------------------- Main Execution --------------------

def main():
    """Main execution function"""
    print("Wave25: DeMark, Williams, Elder and Cyclical Components")
    print("=" * 70)
    
    # Load data
    prices = load_data(TICKER, START_DATE, END_DATE)
    
    if prices.empty:
        print("Error: Unable to load data")
        return
    
    print(f"Data loaded: {len(prices)} rows")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Generate test conditions
    conditions = []
    
    # DEMARK_SEQ - DeMark Sequential
    for setup_period in [9, 13, 21]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('DEMARK_SEQ', {
                'setup_period': setup_period,
                'k': k
            }))
    
    # DEMARK_PRESSURE - DeMark Pressure
    for period in [50, 100, 150]:
        for k in [0.8, 1.0, 1.2]:
            conditions.append(('DEMARK_PRESSURE', {
                'period': period,
                'k': k
            }))
    
    # WILLIAMS_VIX_FIX - Williams VIX Fix
    for period in [14, 22, 30]:
        for k in [0.5, 0.6, 0.7]:
            conditions.append(('WILLIAMS_VIX_FIX', {
                'period': period,
                'k': k
            }))
    
    # WILLIAMS_ULT_OSC - Williams Ultimate Oscillator
    for p1, p2, p3 in [(7, 14, 28), (5, 10, 20), (10, 20, 40)]:
        for k in [0.4, 0.5, 0.6]:
            conditions.append(('WILLIAMS_ULT_OSC', {
                'p1': p1,
                'p2': p2,
                'p3': p3,
                'k': k
            }))
    
    # ELDER_FORCE - Elder Force Index
    for period in [13, 20, 30]:
        for k in [-0.0001, 0.0, 0.0001]:
            conditions.append(('ELDER_FORCE', {
                'period': period,
                'k': k
            }))
    
    # ELDER_IMPULSE - Elder Impulse System
    for ema_period in [13, 20, 30]:
        for k in [0.0, 0.5, 1.0]:
            conditions.append(('ELDER_IMPULSE', {
                'ema_period': ema_period,
                'k': k
            }))
    
    # MOM_QUALITY - Momentum Quality Index
    for period in [50, 100, 150]:
        for k in [-0.01, 0.0, 0.01]:
            conditions.append(('MOM_QUALITY', {
                'period': period,
                'k': k
            }))
    
    # ADAPTIVE_MOM_OSC - Adaptive Momentum Oscillator
    for min_p, max_p in [(10, 50), (20, 100), (30, 150)]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('ADAPTIVE_MOM_OSC', {
                'min_period': min_p,
                'max_period': max_p,
                'k': k
            }))
    
    # HILBERT_CYCLE - Hilbert Transform Cycle
    for period in [20, 30, 40]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('HILBERT_CYCLE', {
                'period': period,
                'k': k
            }))
    
    # FOURIER_CYCLE - Fourier Dominant Cycle
    for window in [50, 100, 150]:
        for k in [-0.1, 0.0, 0.1]:
            conditions.append(('FOURIER_CYCLE', {
                'window': window,
                'k': k
            }))
    
    # Combined with different normalizations
    # ELDER_FORCE_RANK
    for period in [20, 30]:
        for k in [0.5, 0.6, 0.7]:
            conditions.append(('ELDER_FORCE_RANK', {
                'period': period,
                'k': k
            }))
    
    # MOM_QUALITY_ROBUST
    for period in [100, 150]:
        for k in [-0.5, 0.0, 0.5]:
            conditions.append(('MOM_QUALITY_ROBUST', {
                'period': period,
                'k': k
            }))
    
    # DEMARK_PRESSURE_ARCSINH
    for period in [100, 150]:
        for k in [0.0, 0.5, 1.0]:
            conditions.append(('DEMARK_PRESSURE_ARCSINH', {
                'period': period,
                'k': k
            }))
    
    print(f"Total conditions to test: {len(conditions)}")
    
    # Run evaluations
    results = []
    for i, (name, params) in enumerate(conditions):
        if i % 25 == 0:
            print(f"Progress: {i}/{len(conditions)} ({i/len(conditions)*100:.1f}%)")
        
        try:
            result = evaluate(prices, name, params)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {name} with {params}: {e}")
    
    # Sort and display results
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\n" + "=" * 70)
    print("Top 20 Results - Wave25:")
    print("=" * 70)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave25 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave25 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave25 appended to history.log")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print(f"Best Sharpe Ratio: {res_sorted.iloc[0]['sharpe']:.3f}")
    print(f"Best Strategy: {res_sorted.iloc[0]['condition']}")
    print(f"Best Parameters: {res_sorted.iloc[0]['params']}")
    print(f"Strategies with Sharpe > 0.9: {len(res_sorted[res_sorted['sharpe'] > 0.9])}")
    print(f"Strategies with Sharpe > 1.0: {len(res_sorted[res_sorted['sharpe'] > 1.0])}")
    print(f"Strategies with Sharpe > 1.1: {len(res_sorted[res_sorted['sharpe'] > 1.1])}")
    print(f"Strategies with >3000 trades: {len(res_sorted[res_sorted['trades'] > 3000])}")

if __name__ == "__main__":
    main()