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

# -------------------- Enhanced Non-Linear Combinations --------------------

def ultra_nonlinear_vol_adj(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Ultra non-linear combination with multiple transformations"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend with acceleration
    slopes = pd.Series(index=close.index, dtype=float)
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
            except:
                slopes.iloc[i] = 0
    
    # Trend acceleration
    trend_accel = slopes.diff(vol_period // 2)
    accel_factor = (1 + trend_accel * 10).clip(0.5, 1.5)
    
    # Volatility metrics with non-linear transformations
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Multiple non-linear transformations
    # 1. Sigmoid with acceleration
    trend_sigmoid = 2 / (1 + np.exp(-slopes * 0.5 * accel_factor)) - 1
    
    # 2. Power transformation with adaptive exponent
    adaptive_power = 1.5 - 0.5 * vol_percentile
    vol_power = (1 - vol_percentile) ** adaptive_power
    
    # 3. Hyperbolic tangent interaction
    interaction = np.tanh(trend_sigmoid * vol_power * 2)
    
    # 4. Final composite with momentum
    momentum = close / close.shift(vol_period) - 1
    mom_factor = 1 / (1 + np.exp(-momentum * 20))
    
    composite = interaction * mom_factor * accel_factor
    
    return composite

def quantum_trend_oscillator(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Quantum-inspired oscillator with wave functions"""
    close = df['Close']
    returns = close.pct_change()
    
    # Base trend calculation
    slopes = pd.Series(index=close.index, dtype=float)
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
            except:
                slopes.iloc[i] = 0
    
    # Wave function components
    # Primary wave
    primary_freq = 2 * np.pi / trend_period
    primary_wave = np.sin(slopes * primary_freq)
    
    # Secondary wave with volatility modulation
    vol = returns.rolling(vol_period).std()
    vol_norm = (vol - vol.rolling(252).mean()) / vol.rolling(252).std()
    secondary_freq = 2 * np.pi / vol_period
    secondary_wave = np.cos(vol_norm * secondary_freq)
    
    # Interference pattern
    interference = primary_wave * secondary_wave
    
    # Probability amplitude (squared magnitude)
    amplitude = interference ** 2
    
    # Phase adjustment based on momentum
    momentum = close / close.shift(vol_period // 2) - 1
    phase = np.arctan(momentum * 10)
    
    # Final quantum signal
    quantum_signal = amplitude * np.cos(phase)
    
    return quantum_signal

def fractal_dimension_trend(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Fractal dimension-based trend indicator"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend
    slopes = pd.Series(index=close.index, dtype=float)
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
            except:
                slopes.iloc[i] = 0
    
    # Fractal dimension estimation (simplified Hurst exponent)
    def hurst_exponent(ts, lags=20):
        if len(ts) < lags * 2:
            return 0.5
        
        tau = []
        lagvec = []
        
        for lag in range(2, min(lags, len(ts) // 2)):
            pp = ts[lag:] - ts[:-lag]
            if len(pp) > 0 and np.std(pp) > 0:
                tau.append(np.std(pp))
                lagvec.append(lag)
        
        if len(tau) > 2:
            m = np.polyfit(np.log(lagvec), np.log(tau), 1)
            return m[0]
        return 0.5
    
    # Rolling Hurst exponent
    hurst = pd.Series(index=close.index, dtype=float)
    for i in range(trend_period, len(close)):
        window = returns.iloc[i-trend_period:i].values
        hurst.iloc[i] = hurst_exponent(window)
    
    # Fractal adjustment
    fractal_factor = 2 ** (2 - hurst)  # Higher dimension = more noise = lower factor
    
    # Volatility adjustment
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Combined signal
    signal = slopes * (1 - vol_percentile) * fractal_factor.fillna(1)
    
    return signal

def neural_composite_signal(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Neural network-inspired composite signal"""
    close = df['Close']
    volume = df['Volume']
    returns = close.pct_change()
    
    # Input layer: multiple features
    # 1. Trend
    slopes = pd.Series(index=close.index, dtype=float)
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
            except:
                slopes.iloc[i] = 0
    
    # 2. Volatility
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # 3. Volume ratio
    vol_ma = volume.rolling(vol_period).mean()
    vol_ratio = volume / vol_ma.where(vol_ma > 0, 1)
    
    # 4. Price position
    ma = close.rolling(trend_period).mean()
    price_position = (close - ma) / ma
    
    # Normalize inputs
    trend_norm = (slopes - slopes.rolling(252).mean()) / slopes.rolling(252).std()
    vol_norm = (vol_percentile - 0.5) * 2
    volratio_norm = (vol_ratio - 1).clip(-1, 1)
    position_norm = price_position.clip(-0.1, 0.1) * 10
    
    # Hidden layer with activation functions
    # Node 1: Trend-volatility interaction
    h1 = np.tanh(trend_norm.fillna(0) * (1 - vol_norm.fillna(0.5)))
    
    # Node 2: Volume-position interaction
    h2 = 1 / (1 + np.exp(-(volratio_norm.fillna(0) + position_norm.fillna(0))))
    
    # Node 3: Non-linear trend transformation
    h3 = np.sin(trend_norm.fillna(0) * np.pi / 2) * (1 - vol_percentile.fillna(0.5))
    
    # Output layer: weighted combination
    w1, w2, w3 = 0.5, 0.3, 0.2
    output = w1 * h1 + w2 * h2 + w3 * h3
    
    # Final activation
    signal = np.tanh(output * 2)
    
    return signal

def entropy_adjusted_trend(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Entropy-adjusted trend indicator"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend
    slopes = pd.Series(index=close.index, dtype=float)
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        x = np.arange(trend_period)
        if len(y) == trend_period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope / close.iloc[i] * 100
            except:
                slopes.iloc[i] = 0
    
    # Shannon entropy of returns distribution
    def shannon_entropy(data):
        if len(data) < 10:
            return 1.0
        
        # Create histogram
        hist, _ = np.histogram(data, bins=10)
        hist = hist + 1  # Avoid log(0)
        prob = hist / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(prob * np.log(prob))
        return entropy
    
    # Rolling entropy
    entropy = returns.rolling(vol_period).apply(shannon_entropy)
    entropy_norm = (entropy - entropy.rolling(252).mean()) / entropy.rolling(252).std()
    
    # Low entropy = more predictable = stronger signal
    entropy_factor = np.exp(-entropy_norm.fillna(0))
    
    # Volatility adjustment
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Combined signal with entropy weighting
    signal = slopes * (1 - vol_percentile) * entropy_factor
    
    return signal

def phase_space_momentum(df: pd.DataFrame, embed_dim: int, delay: int) -> pd.Series:
    """Phase space reconstruction momentum indicator"""
    close = df['Close']
    
    # Create phase space embedding
    embedded = pd.DataFrame(index=close.index)
    for i in range(embed_dim):
        embedded[f'dim_{i}'] = close.shift(i * delay)
    
    # Calculate phase space trajectory length
    trajectory_length = pd.Series(index=close.index, dtype=float)
    
    for i in range(embed_dim * delay, len(close)):
        # Get current and previous points in phase space
        curr_point = embedded.iloc[i].values
        prev_point = embedded.iloc[i-1].values
        
        if not np.any(np.isnan(curr_point)) and not np.any(np.isnan(prev_point)):
            # Euclidean distance in phase space
            dist = np.sqrt(np.sum((curr_point - prev_point) ** 2))
            trajectory_length.iloc[i] = dist
    
    # Normalize by price
    norm_trajectory = trajectory_length / close
    
    # Momentum based on trajectory acceleration
    trajectory_momentum = norm_trajectory.pct_change(delay)
    
    # Smooth with adaptive window
    vol = close.pct_change().rolling(delay * 2).std()
    vol_rank = vol.rolling(100).rank(pct=True)
    smooth_window = (delay * (1 + vol_rank)).astype(int).clip(5, 50)
    
    smoothed_signal = pd.Series(index=close.index, dtype=float)
    for i in range(50, len(close)):
        window = int(smooth_window.iloc[i]) if not pd.isna(smooth_window.iloc[i]) else delay
        smoothed_signal.iloc[i] = trajectory_momentum.iloc[i-window:i+1].mean()
    
    return smoothed_signal

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
        if cond_name == 'ULTRA_NONLINEAR':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = ultra_nonlinear_vol_adj(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'QUANTUM_TREND':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = quantum_trend_oscillator(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'FRACTAL_DIM':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = fractal_dimension_trend(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'NEURAL_COMP':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = neural_composite_signal(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'ENTROPY_ADJ':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = entropy_adjusted_trend(df, trend_p, vol_p)
            signal = indicator > k
        
        elif cond_name == 'PHASE_SPACE':
            embed_dim = params['embed_dim']
            delay = params['delay']
            k = params['k']
            indicator = phase_space_momentum(df, embed_dim, delay)
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
    print("Wave24 Enhanced: Aggressive Non-Linear Combinations")
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
    
    # ULTRA_NONLINEAR - Ultra non-linear combination
    for trend_p in [120, 130, 140, 150]:
        for vol_p in [40, 45, 50]:
            for k in [-0.1, 0.0, 0.1, 0.2]:
                conditions.append(('ULTRA_NONLINEAR', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # QUANTUM_TREND - Quantum oscillator
    for trend_p in [100, 130, 150]:
        for vol_p in [30, 45, 60]:
            for k in [-0.2, 0.0, 0.2]:
                conditions.append(('QUANTUM_TREND', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # FRACTAL_DIM - Fractal dimension trend
    for trend_p in [120, 130, 140]:
        for vol_p in [40, 45, 50]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('FRACTAL_DIM', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # NEURAL_COMP - Neural composite
    for trend_p in [120, 130, 140]:
        for vol_p in [40, 45, 50]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('NEURAL_COMP', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # ENTROPY_ADJ - Entropy adjusted
    for trend_p in [120, 130, 140]:
        for vol_p in [40, 45, 50]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('ENTROPY_ADJ', {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }))
    
    # PHASE_SPACE - Phase space momentum
    for embed_dim in [3, 4, 5]:
        for delay in [5, 10, 15]:
            for k in [-0.0001, 0.0, 0.0001]:
                conditions.append(('PHASE_SPACE', {
                    'embed_dim': embed_dim,
                    'delay': delay,
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
    print("Top 20 Results - Wave24 Enhanced:")
    print("=" * 70)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave24 Enhanced ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave24 Enhanced | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave24 Enhanced appended to history.log")
    
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