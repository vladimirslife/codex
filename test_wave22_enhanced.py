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

# -------------------- Data --------------------

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
    
    return pd.DataFrame()

# -------------------- Enhanced Indicators Combining Best Elements --------------------

def vol_adj_trend_with_normalization(df: pd.DataFrame, trend_period: int, vol_period: int, norm_type: str = 'standard') -> pd.Series:
    """VOL_ADJ_TREND with different normalization techniques"""
    close = df['Close']
    returns = close.pct_change()
    
    # Calculate trend using linear regression slope
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
    
    # Calculate volatility percentile
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    
    # Apply normalization to trend
    if norm_type == 'tanh':
        normalized_slopes = np.tanh(slopes * 10)
    elif norm_type == 'rank':
        normalized_slopes = slopes.rolling(trend_period).rank(pct=True)
    else:  # standard
        normalized_slopes = slopes
    
    # Adjust by volatility
    adjusted_trend = normalized_slopes * (1 - vol_percentile)
    
    return adjusted_trend

def price_action_vol_trend(df: pd.DataFrame, period: int) -> pd.Series:
    """Combine price action patterns with volatility-adjusted trend"""
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    returns = close.pct_change()
    
    # Price action component
    body = close - open_price
    range_size = high - low
    body_ratio = body / range_size.where(range_size > 0, 1)
    
    # Volume component
    vol_ma = volume.rolling(period).mean()
    vol_weight = (volume / vol_ma.where(vol_ma > 0, 1)).clip(0, 2)
    
    # Trend component
    trend_slope = pd.Series(index=close.index, dtype=float)
    for i in range(period, len(close)):
        y = close.iloc[i-period:i].values
        x = np.arange(period)
        if len(y) == period:
            try:
                slope = np.polyfit(x, y, 1)[0]
                trend_slope.iloc[i] = slope / close.iloc[i] * 100
            except:
                trend_slope.iloc[i] = 0
    
    # Volatility adjustment
    vol = returns.rolling(period // 2).std()
    vol_percentile = vol.rolling(period).rank(pct=True)
    
    # Combined signal
    pa_component = body_ratio * vol_weight
    trend_component = trend_slope * (1 - vol_percentile)
    
    # Normalize and combine
    pa_norm = (pa_component - pa_component.rolling(period).mean()) / pa_component.rolling(period).std()
    trend_norm = (trend_component - trend_component.rolling(period).mean()) / trend_component.rolling(period).std()
    
    combined = (pa_norm * 0.3 + trend_norm * 0.7).fillna(0)
    
    return combined

def adaptive_vol_momentum(df: pd.DataFrame, base_period: int, vol_measure: str = 'standard') -> pd.Series:
    """Momentum with adaptive period based on volatility regime"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    returns = close.pct_change()
    
    # Calculate volatility based on measure
    if vol_measure == 'garman_klass':
        # Garman-Klass volatility
        hl_ratio = np.log(high / low) ** 2
        co_ratio = np.log(close / df['Open']) ** 2
        vol = np.sqrt(0.5 * hl_ratio - (2 * np.log(2) - 1) * co_ratio).rolling(20).mean()
    elif vol_measure == 'parkinson':
        # Parkinson volatility
        hl_ratio = np.log(high / low)
        vol = (hl_ratio / (2 * np.sqrt(np.log(2)))).rolling(20).mean()
    else:
        # Standard volatility
        vol = returns.rolling(20).std()
    
    # Volatility regime
    vol_percentile = vol.rolling(base_period * 2).rank(pct=True)
    
    # Adaptive period: shorter in low vol, longer in high vol
    adaptive_period = (base_period * (0.5 + vol_percentile)).clip(10, 200)
    
    # Calculate adaptive momentum
    momentum = pd.Series(index=close.index, dtype=float)
    
    for i in range(200, len(close)):
        period = int(adaptive_period.iloc[i]) if not pd.isna(adaptive_period.iloc[i]) else base_period
        if i >= period and period > 0:
            momentum.iloc[i] = close.iloc[i] / close.iloc[i-period] - 1
    
    # Volatility adjustment
    adjusted_momentum = momentum * (1 - vol_percentile)
    
    return adjusted_momentum

def composite_enhanced_signal(df: pd.DataFrame, trend_period: int, vol_period: int) -> pd.Series:
    """Composite signal combining best elements"""
    close = df['Close']
    volume = df['Volume']
    returns = close.pct_change()
    
    # 1. Volatility-adjusted trend (best performer)
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
    
    vol = returns.rolling(vol_period).std()
    vol_percentile = vol.rolling(vol_period * 2).rank(pct=True)
    vol_adj_trend = slopes * (1 - vol_percentile)
    
    # 2. Volume confirmation
    vol_ma = volume.rolling(vol_period).mean()
    vol_ratio = volume / vol_ma.where(vol_ma > 0, 1)
    vol_signal = (vol_ratio - 1).rolling(10).mean()
    
    # 3. Price position
    ma = close.rolling(trend_period).mean()
    price_position = (close - ma) / ma
    
    # Normalize components
    def safe_zscore(series, window):
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std.where(std > 0, 1)
    
    trend_z = safe_zscore(vol_adj_trend, trend_period)
    vol_z = safe_zscore(vol_signal, vol_period)
    pos_z = safe_zscore(price_position, trend_period)
    
    # Weighted combination
    composite = trend_z * 0.6 + vol_z * 0.2 + pos_z * 0.2
    
    return composite

# -------------------- Evaluation --------------------

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
        if cond_name == 'VOL_ADJ_TREND_NORM':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            norm_type = params['norm_type']
            k = params['k']
            indicator = vol_adj_trend_with_normalization(df, trend_p, vol_p, norm_type)
            signal = indicator > k
        
        elif cond_name == 'PA_VOL_TREND':
            period = params['period']
            k = params['k']
            indicator = price_action_vol_trend(df, period)
            signal = indicator > k
        
        elif cond_name == 'ADAPTIVE_VOL_MOM':
            base_p = params['base_period']
            vol_measure = params['vol_measure']
            k = params['k']
            indicator = adaptive_vol_momentum(df, base_p, vol_measure)
            signal = indicator > k
        
        elif cond_name == 'COMPOSITE_ENH':
            trend_p = params['trend_period']
            vol_p = params['vol_period']
            k = params['k']
            indicator = composite_enhanced_signal(df, trend_p, vol_p)
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

# -------------------- Main --------------------

def main():
    print("Wave22 Enhanced: Combining Best Elements")
    print("=" * 60)
    
    # Load data
    prices = load_data(TICKER, START_DATE, END_DATE)
    
    if prices.empty:
        print("Error: Unable to load data")
        return
    
    print(f"Data loaded: {len(prices)} rows")
    
    # Test focused parameter sets
    results = []
    
    # VOL_ADJ_TREND_NORM - Best performer with normalization
    for trend_p in [130, 140, 150]:
        for vol_p in [40, 45, 50]:
            for norm_type in ['standard', 'tanh', 'rank']:
                for k in [-0.1, 0.0, 0.1]:
                    params = {
                        'trend_period': trend_p,
                        'vol_period': vol_p,
                        'norm_type': norm_type,
                        'k': k
                    }
                    result = evaluate(prices, 'VOL_ADJ_TREND_NORM', params)
                    results.append(result)
    
    # PA_VOL_TREND - Price action with vol trend
    for period in [50, 100, 150]:
        for k in [-0.5, 0.0, 0.5]:
            params = {
                'period': period,
                'k': k
            }
            result = evaluate(prices, 'PA_VOL_TREND', params)
            results.append(result)
    
    # ADAPTIVE_VOL_MOM - Adaptive volatility momentum
    for base_p in [50, 100, 150]:
        for vol_measure in ['standard', 'garman_klass', 'parkinson']:
            for k in [-0.01, 0.0, 0.01]:
                params = {
                    'base_period': base_p,
                    'vol_measure': vol_measure,
                    'k': k
                }
                result = evaluate(prices, 'ADAPTIVE_VOL_MOM', params)
                results.append(result)
    
    # COMPOSITE_ENH - Composite enhanced signal
    for trend_p in [140, 150, 160]:
        for vol_p in [40, 45, 50]:
            for k in [-0.5, 0.0, 0.5]:
                params = {
                    'trend_period': trend_p,
                    'vol_period': vol_p,
                    'k': k
                }
                result = evaluate(prices, 'COMPOSITE_ENH', params)
                results.append(result)
    
    print(f"Total conditions tested: {len(results)}")
    
    # Sort and display results
    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(['sharpe', 'trades'], ascending=[False, False])
    
    print("\n" + "=" * 60)
    print("Top 20 Results:")
    print("=" * 60)
    print(res_sorted.head(20).to_string(index=False))
    
    # Log top 3
    with open('history.log', 'a') as f:
        f.write("\n--- Wave22_Enhanced ---\n")
        for _, row in res_sorted.head(3).iterrows():
            f.write(f"Wave22_Enh | Condition: {row['condition']} | Params: {row['params']} | "
                    f"Sharpe: {row['sharpe']:.3f} | Trades: {row['trades']} | TotalRet: {row['total_return']:.2%}\n")
    
    print("\nTop 3 appended to history.log")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Best Sharpe: {res_sorted.iloc[0]['sharpe']:.3f}")
    print(f"Best Strategy: {res_sorted.iloc[0]['condition']}")
    print(f"Best Parameters: {res_sorted.iloc[0]['params']}")
    print(f"Strategies with Sharpe > 0.95: {len(res_sorted[res_sorted['sharpe'] > 0.95])}")
    print(f"Strategies with Sharpe > 1.0: {len(res_sorted[res_sorted['sharpe'] > 1.0])}")

if __name__ == "__main__":
    main()