import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
import os
from typing import Optional, Union, Tuple

TICKER = "SPY"
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RF_ANNUAL = 0.02
RF_DAILY = RF_ANNUAL / 252.0
MAX_COMBOS = 1000

# -------------------- Data --------------------

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load data with proper error handling and type safety"""
    cache_file = f"{ticker}_{start}_{end}.csv"
    
    try:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure we have a DataFrame
            if isinstance(df, pd.DataFrame):
                # Ensure numeric types
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                return pd.DataFrame()
    except Exception as e:
        print(f"Error reading cache file: {e}")
    
    # Download fresh data
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        # Ensure we have a DataFrame
        if isinstance(data, pd.DataFrame) and not data.empty:
            df = data.copy()
            # Save to cache
            df.to_csv(cache_file)
            # Ensure numeric types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except Exception as e:
        print(f"Error downloading data: {e}")
    
    # Return empty DataFrame if all fails
    return pd.DataFrame()

# -------------------- Helper Functions --------------------

def safe_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA with type safety"""
    if not isinstance(series, pd.Series):
        return pd.Series()
    return series.ewm(span=span, adjust=False).mean()

def safe_sma(series: pd.Series, window: int) -> pd.Series:
    """SMA with type safety"""
    if not isinstance(series, pd.Series):
        return pd.Series()
    return series.rolling(window).mean()

def safe_division(numerator: pd.Series, denominator: pd.Series, fill_value: float = 1.0) -> pd.Series:
    """Safe division with zero handling"""
    return numerator / denominator.where(denominator != 0, fill_value)

# -------------------- Indicators --------------------

def adaptive_trend_momentum(df: pd.DataFrame, base_period: int, adapt_factor: float) -> pd.Series:
    """Adaptive trend with momentum-based period adjustment"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate momentum strength
    momentum = close.pct_change(20)
    mom_percentile = momentum.rolling(100).rank(pct=True)
    
    # Adaptive period: shorter in strong trends, longer in weak trends
    adaptive_period = base_period * (1 + adapt_factor * (1 - mom_percentile))
    adaptive_period = adaptive_period.clip(10, 200).fillna(base_period)
    
    # Calculate adaptive trend
    trend_values = pd.Series(index=close.index, dtype=float)
    
    for i in range(200, len(close)):
        period = int(adaptive_period.iloc[i])
        if period > 0 and i >= period:
            # Linear regression over adaptive period
            y = close.iloc[i-period:i].values
            x = np.arange(period)
            if len(y) == period:
                slope = np.polyfit(x, y, 1)[0]
                trend_values.iloc[i] = slope / close.iloc[i] * 100
    
    # Volatility adjustment
    vol = returns.rolling(20).std()
    vol_rank = vol.rolling(100).rank(pct=True)
    
    # Final signal: trend adjusted by volatility
    adjusted_signal = trend_values * (1 - vol_rank * 0.5)
    
    return adjusted_signal

def volume_weighted_momentum_dynamic(df: pd.DataFrame, min_period: int, max_period: int) -> pd.Series:
    """Volume-weighted momentum with dynamic lookback based on volume patterns"""
    close = df['Close']
    volume = df['Volume']
    
    # Volume patterns determine lookback period
    vol_ma_short = volume.rolling(10).mean()
    vol_ma_long = volume.rolling(50).mean()
    vol_ratio = safe_division(vol_ma_short, vol_ma_long)
    
    # High relative volume = shorter period (more responsive)
    vol_percentile = vol_ratio.rolling(100).rank(pct=True)
    dynamic_period = max_period - (max_period - min_period) * vol_percentile
    dynamic_period = dynamic_period.clip(min_period, max_period).fillna(max_period)
    
    # Calculate momentum with dynamic period
    momentum_values = pd.Series(index=close.index, dtype=float)
    
    for i in range(max_period, len(close)):
        period = int(dynamic_period.iloc[i])
        if period > 0 and i >= period:
            # Volume-weighted price change
            price_change = close.iloc[i] / close.iloc[i-period] - 1
            
            # Average volume over period
            avg_vol = volume.iloc[i-period:i].mean()
            current_vol = volume.iloc[i]
            
            # Weight by relative volume
            if avg_vol > 0:
                vol_weight = current_vol / avg_vol
                momentum_values.iloc[i] = price_change * vol_weight
    
    # Normalize by rolling std
    mom_std = momentum_values.rolling(50).std()
    normalized = safe_division(momentum_values, mom_std)
    
    return normalized

def volatility_regime_adaptive_signal(df: pd.DataFrame, trend_period: int, vol_lookback: int) -> pd.Series:
    """Signal that adapts to volatility regime with proper error handling"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate base trend
    trend = pd.Series(index=close.index, dtype=float)
    
    for i in range(trend_period, len(close)):
        y = close.iloc[i-trend_period:i].values
        if len(y) == trend_period:
            x = np.arange(trend_period)
            # Robust linear regression
            try:
                coeffs = np.polyfit(x, y, 1)
                trend.iloc[i] = coeffs[0] / close.iloc[i] * 100
            except:
                trend.iloc[i] = 0
    
    # Volatility regime classification
    vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    
    # Historical volatility percentiles
    vol_20 = vol.rolling(252).quantile(0.2)
    vol_50 = vol.rolling(252).quantile(0.5)
    vol_80 = vol.rolling(252).quantile(0.8)
    
    # Regime classification
    low_vol = vol < vol_20
    mid_vol = (vol >= vol_20) & (vol < vol_80)
    high_vol = vol >= vol_80
    
    # Adaptive signal based on regime
    signal = pd.Series(index=close.index, dtype=float)
    
    # Low vol: aggressive trend following
    signal[low_vol] = trend[low_vol] * 2.0
    
    # Mid vol: normal trend following
    signal[mid_vol] = trend[mid_vol] * 1.0
    
    # High vol: conservative/contrarian
    signal[high_vol] = trend[high_vol] * 0.5
    
    return signal

def market_microstructure_signal(df: pd.DataFrame, period: int) -> pd.Series:
    """Signal based on market microstructure patterns"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    open_price = df['Open']
    volume = df['Volume']
    
    # Intraday patterns
    daily_range = high - low
    body = (close - open_price).abs()
    upper_shadow = high - pd.concat([close, open_price], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_price], axis=1).min(axis=1) - low
    
    # Microstructure metrics
    body_ratio = safe_division(body, daily_range, 0.5)
    shadow_ratio = safe_division(upper_shadow + lower_shadow, daily_range, 0.5)
    
    # Volume patterns
    vol_ma = volume.rolling(period).mean()
    relative_vol = safe_division(volume, vol_ma)
    
    # Close position in daily range
    close_position = safe_division(close - low, daily_range, 0.5)
    
    # Combined microstructure score
    micro_score = (
        body_ratio * 0.3 +                    # Body size importance
        (1 - shadow_ratio) * 0.2 +            # Low shadow = strong
        relative_vol.clip(0, 2) * 0.3 +       # Volume confirmation
        (1 - close_position) * 0.2            # Close near lows = bullish reversal
    )
    
    # Smooth and normalize
    smooth_score = micro_score.rolling(5).mean()
    score_std = smooth_score.rolling(period).std()
    normalized = safe_division(smooth_score - smooth_score.rolling(period).mean(), score_std)
    
    return normalized

def adaptive_volatility_breakout(df: pd.DataFrame, lookback: int, multiplier: float) -> pd.Series:
    """Volatility breakout with adaptive thresholds"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate volatility metrics
    vol = returns.rolling(lookback).std()
    
    # Adaptive threshold based on volatility regime
    vol_percentile = vol.rolling(lookback * 4).rank(pct=True)
    
    # Dynamic multiplier: higher in low vol, lower in high vol
    dynamic_mult = multiplier * (1.5 - vol_percentile)
    
    # Bollinger-style bands with adaptive width
    ma = close.rolling(lookback).mean()
    upper_band = ma + vol * close * dynamic_mult
    lower_band = ma - vol * close * dynamic_mult
    
    # Breakout signals
    upper_breakout = close > upper_band
    lower_breakout = close < lower_band
    
    # Signal strength based on breakout magnitude
    upper_strength = safe_division(close - upper_band, upper_band) * upper_breakout
    lower_strength = safe_division(lower_band - close, lower_band) * lower_breakout
    
    # Combined signal (positive for upper breakout, negative for lower)
    signal = upper_strength - lower_strength
    
    return signal

def volume_price_divergence_adaptive(df: pd.DataFrame, period: int) -> pd.Series:
    """Adaptive divergence between volume and price trends"""
    close = df['Close']
    volume = df['Volume']
    
    # Price trend
    price_roc = close.pct_change(period)
    price_trend = price_roc.rolling(period).mean()
    
    # Volume trend
    vol_roc = volume.pct_change(period)
    vol_trend = vol_roc.rolling(period).mean()
    
    # Normalize trends
    price_zscore = (price_trend - price_trend.rolling(period * 2).mean()) / price_trend.rolling(period * 2).std()
    vol_zscore = (vol_trend - vol_trend.rolling(period * 2).mean()) / vol_trend.rolling(period * 2).std()
    
    # Divergence with adaptive weighting
    divergence = vol_zscore - price_zscore
    
    # Weight by trend strength
    trend_strength = price_roc.abs().rolling(period).mean()
    trend_percentile = trend_strength.rolling(period * 2).rank(pct=True)
    
    # Stronger weight in trending markets
    weighted_divergence = divergence * (0.5 + trend_percentile)
    
    return weighted_divergence

# -------------------- Sharpe --------------------

def sharpe_ratio(returns: pd.Series) -> float:
    """Calculate Sharpe ratio with safety checks"""
    if returns.empty or len(returns) < 20:
        return 0.0
    
    # Remove NaN values
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    
    std = clean_returns.std(ddof=1)
    if std == 0 or np.isclose(std, 0):
        return 0.0
    
    excess = clean_returns - RF_DAILY
    return math.sqrt(252) * excess.mean() / std

# -------------------- Evaluation --------------------

def evaluate(df: pd.DataFrame, cond_name: str, params: dict) -> dict:
    """Evaluate strategy with error handling"""
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
        if cond_name == 'ADAPT_TREND_MOM':
            base_p = params['base_period']
            adapt_f = params['adapt_factor']
            k = params['k']
            indicator = adaptive_trend_momentum(df, base_p, adapt_f)
            signal = indicator > k
        
        elif cond_name == 'VOL_WGT_MOM_DYN':
            min_p = params['min_period']
            max_p = params['max_period']
            k = params['k']
            indicator = volume_weighted_momentum_dynamic(df, min_p, max_p)
            signal = indicator > k
        
        elif cond_name == 'VOL_REGIME_ADAPT':
            trend_p = params['trend_period']
            vol_lb = params['vol_lookback']
            k = params['k']
            indicator = volatility_regime_adaptive_signal(df, trend_p, vol_lb)
            signal = indicator > k
        
        elif cond_name == 'MARKET_MICRO':
            period = params['period']
            k = params['k']
            indicator = market_microstructure_signal(df, period)
            signal = indicator > k
        
        elif cond_name == 'ADAPT_VOL_BREAK':
            lookback = params['lookback']
            mult = params['multiplier']
            k = params['k']
            indicator = adaptive_volatility_breakout(df, lookback, mult)
            signal = indicator > k
        
        elif cond_name == 'VOL_PRICE_DIV_ADAPT':
            period = params['period']
            k = params['k']
            indicator = volume_price_divergence_adaptive(df, period)
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
    
    # Total return calculation with safety
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

# -------------------- Grid Search --------------------

def generate_conditions() -> list:
    """Generate test conditions"""
    conditions = []
    
    # ADAPT_TREND_MOM - Adaptive trend momentum
    for base_p in [50, 100, 150, 200]:
        for adapt_f in [0.5, 1.0, 1.5]:
            for k in [-0.5, 0.0, 0.5, 1.0]:
                conditions.append(('ADAPT_TREND_MOM', {
                    'base_period': base_p,
                    'adapt_factor': adapt_f,
                    'k': k
                }))
    
    # VOL_WGT_MOM_DYN - Volume weighted momentum dynamic
    for min_p in [10, 20]:
        for max_p in [50, 100, 150]:
            for k in [-0.5, 0.0, 0.5, 1.0]:
                conditions.append(('VOL_WGT_MOM_DYN', {
                    'min_period': min_p,
                    'max_period': max_p,
                    'k': k
                }))
    
    # VOL_REGIME_ADAPT - Volatility regime adaptive
    for trend_p in [50, 100, 150]:
        for vol_lb in [20, 30, 50]:
            for k in [-1.0, -0.5, 0.0, 0.5]:
                conditions.append(('VOL_REGIME_ADAPT', {
                    'trend_period': trend_p,
                    'vol_lookback': vol_lb,
                    'k': k
                }))
    
    # MARKET_MICRO - Market microstructure
    for period in [10, 20, 50]:
        for k in [-1.0, -0.5, 0.0, 0.5]:
            conditions.append(('MARKET_MICRO', {
                'period': period,
                'k': k
            }))
    
    # ADAPT_VOL_BREAK - Adaptive volatility breakout
    for lookback in [20, 50, 100]:
        for mult in [1.5, 2.0, 2.5]:
            for k in [-0.1, 0.0, 0.1]:
                conditions.append(('ADAPT_VOL_BREAK', {
                    'lookback': lookback,
                    'multiplier': mult,
                    'k': k
                }))
    
    # VOL_PRICE_DIV_ADAPT - Volume price divergence adaptive
    for period in [20, 50, 100]:
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            conditions.append(('VOL_PRICE_DIV_ADAPT', {
                'period': period,
                'k': k
            }))
    
    return conditions[:MAX_COMBOS]

# -------------------- Main --------------------

def main():
    """Main execution function"""
    print("Wave21_v4: Advanced Adaptive Indicators")
    print("=" * 50)
    
    # Load data
    prices = load_data(TICKER, START_DATE, END_DATE)
    
    if prices.empty:
        print("Error: Unable to load data")
        return
    
    print(f"Data loaded: {len(prices)} rows")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Generate conditions
    conditions = generate_conditions()
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
    
    print("\n" + "=" * 50)
    print("Top 15 Results - Wave21_v4:")
    print("=" * 50)
    print(res_sorted.head(15).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave21_v4 ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave21_v4 | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 Wave21_v4 appended to history.log")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print(f"Best Sharpe Ratio: {res_sorted.iloc[0]['sharpe']:.3f}")
    print(f"Best Strategy: {res_sorted.iloc[0]['condition']}")
    print(f"Best Parameters: {res_sorted.iloc[0]['params']}")
    print(f"Strategies with Sharpe > 0.9: {len(res_sorted[res_sorted['sharpe'] > 0.9])}")
    print(f"Strategies with >3000 trades: {len(res_sorted[res_sorted['trades'] > 3000])}")

if __name__ == "__main__":
    main()