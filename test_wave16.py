#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 16: Advanced statistical measures and cross-ticker analysis
"""

import pandas as pd
import numpy as np
from scipy import stats

# ------------------------- HELPERS -----------------------------------
def load_ticker(path: str) -> pd.DataFrame:
    """Load CSV, standardize columns, compute overnight returns."""
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.Timestamp("2006-01-01")]
    df = df.sort_values(by="Date")  # type: ignore
    df = df.reset_index(drop=True)
    
    # Вычисляем ночную доходность: от close до open следующего дня
    df["prev_close"] = df["close"].shift(1)
    df["overnight_return"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    # Для стратегии нужна доходность от сегодняшнего close до завтрашнего open
    df["next_open"] = df["open"].shift(-1)
    df["next_overnight_return"] = (df["next_open"] - df["close"]) / df["close"]
    
    # Base measures
    df["overnight_squared"] = df["overnight_return"] ** 2
    df["hl_range"] = (df["high"] - df["low"]) / df["open"]
    
    # Advanced statistical measures
    # 1. Normalized ATR (without volume)
    df["atr"] = df[["high", "low", "close"]].apply(
        lambda x: max(x["high"] - x["low"], 
                     abs(x["high"] - df["close"].shift(1).loc[x.name]) if x.name > 0 else x["high"] - x["low"],
                     abs(x["low"] - df["close"].shift(1).loc[x.name]) if x.name > 0 else x["high"] - x["low"]),
        axis=1
    ) / df["open"]
    df["atr_normalized"] = df["atr"] / df["atr"].rolling(window=20).mean()
    
    # 2. Kurtosis of returns
    df["close_pct"] = df["close"].pct_change()
    df["return_kurtosis"] = df["close_pct"].rolling(window=20).kurt()
    
    # 3. Entropy-based volatility
    def entropy(series):
        """Calculate Shannon entropy of returns"""
        if len(series) < 5:
            return np.nan
        hist, _ = np.histogram(series, bins=5)
        hist = hist[hist > 0]  # Remove zero bins
        probs = hist / hist.sum()
        return -np.sum(probs * np.log(probs))
    
    df["return_entropy"] = df["close_pct"].rolling(window=20).apply(entropy, raw=True)
    
    # 4. Fractal dimension (Higuchi method approximation)
    def hurst_exponent(series):
        """Simplified Hurst exponent calculation"""
        if len(series) < 10:
            return np.nan
        lags = range(2, min(10, len(series)//2))
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    df["hurst_exp"] = df["close"].rolling(window=20).apply(lambda x: hurst_exponent(x.values), raw=False)
    
    # 5. Volatility asymmetry
    df["up_moves"] = (df["close_pct"] > 0).astype(float)
    df["down_moves"] = (df["close_pct"] < 0).astype(float)
    df["up_vol"] = df["up_moves"] * df["hl_range"]
    df["down_vol"] = df["down_moves"] * df["hl_range"]
    df["vol_asymmetry"] = df["up_vol"].rolling(window=10).mean() / (df["down_vol"].rolling(window=10).mean() + 0.0001)
    
    # 6. Consecutive calm days
    df["is_calm"] = (df["hl_range"] < df["hl_range"].rolling(window=20).quantile(0.4)).fillna(False).astype(int)
    df["consecutive_calm"] = df["is_calm"].groupby((df["is_calm"] != df["is_calm"].shift()).cumsum()).cumsum()
    
    # 7. Time-weighted volatility
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_monday"] = (df["day_of_week"] == 0).fillna(False).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).fillna(False).astype(int)
    
    # 8. Volatility momentum
    df["vol_momentum"] = df["hl_range"].rolling(window=5).mean() - df["hl_range"].rolling(window=20).mean()
    df["vol_momentum_ma5"] = df["vol_momentum"].rolling(window=5).mean()
    
    # 9. Percentile rank of current volatility
    df["vol_percentile_rank"] = df["hl_range"].rolling(window=50).rank(pct=True)
    
    # 10. Exponentially weighted statistics
    df["hl_range_ema"] = df["hl_range"].ewm(span=10, adjust=False).mean()
    df["hl_range_emstd"] = df["hl_range"].ewm(span=10, adjust=False).std()
    df["vol_zscore_ema"] = (df["hl_range"] - df["hl_range_ema"]) / (df["hl_range_emstd"] + 0.0001)
    
    # 11. Range efficiency
    df["range_efficiency"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 0.0001)
    df["range_efficiency_ma5"] = df["range_efficiency"].rolling(window=5).mean()
    
    # 12. Volatility clustering measure
    df["vol_autocorr"] = df["hl_range"].rolling(window=20).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Merge on Date
    data = qqq.copy()
    data = data.merge(spy[["Date", "hl_range", "overnight_squared", "close_pct", "atr"]], 
                      on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[["Date", "hl_range", "overnight_squared", "close_pct", "atr"]], 
                      on="Date", suffixes=("", "_xlk"))
    
    # Cross-ticker measures
    # 1. Average volatility across tickers
    data["avg_hl_range"] = (data["hl_range"] + data["hl_range_spy"] + data["hl_range_xlk"]) / 3
    data["avg_hl_range_ma5"] = data["avg_hl_range"].rolling(window=5).mean()
    
    # 2. Volatility dispersion
    data["vol_dispersion"] = data[["hl_range", "hl_range_spy", "hl_range_xlk"]].std(axis=1)
    data["vol_dispersion_ma5"] = data["vol_dispersion"].rolling(window=5).mean()
    
    # 3. QQQ relative volatility
    data["qqq_rel_vol"] = data["hl_range"] / (data["avg_hl_range"] + 0.0001)
    data["qqq_rel_vol_ma5"] = data["qqq_rel_vol"].rolling(window=5).mean()
    
    # 4. Market-wide calm days
    data["market_calm"] = (
        (data["hl_range"] < data["hl_range"].rolling(window=20).quantile(0.3)) &
        (data["hl_range_spy"] < data["hl_range_spy"].rolling(window=20).quantile(0.3))
    ).fillna(False).astype(int)
    data["market_calm_ma5"] = data["market_calm"].rolling(window=5).mean()
    
    # 5. Cross-correlation of volatilities
    data["vol_corr_spy"] = data["hl_range"].rolling(window=20).corr(data["hl_range_spy"])
    data["vol_corr_xlk"] = data["hl_range"].rolling(window=20).corr(data["hl_range_xlk"])
    
    # 6. Beta-adjusted volatility
    data["beta_spy"] = data["close_pct"].rolling(window=60).cov(data["close_pct_spy"]) / data["close_pct_spy"].rolling(window=60).var()
    data["beta_adj_vol"] = data["hl_range"] / (data["beta_spy"].abs() + 0.5)
    data["beta_adj_vol_ma5"] = data["beta_adj_vol"].rolling(window=5).mean()
    
    # 7. Sector vs Market volatility
    data["tech_vs_market"] = (data["hl_range"] + data["hl_range_xlk"]) / (2 * data["hl_range_spy"] + 0.0001)
    data["tech_vs_market_ma5"] = data["tech_vs_market"].rolling(window=5).mean()
    
    return data

def calculate_strategy_performance(df, signal_column):
    """Рассчитать производительность стратегии"""
    df = df.copy()
    
    # Доходность стратегии
    df["strategy_return"] = df[signal_column] * df["next_overnight_return"]
    df["strategy_return"] = df["strategy_return"].fillna(0)
    
    # Расчет Sharpe Ratio
    annual_rf = 0.02
    daily_rf = annual_rf / 252
    
    excess_returns = df["strategy_return"] - daily_rf
    mean_excess_annual = excess_returns.mean() * 252
    std_excess_annual = excess_returns.std() * np.sqrt(252)
    sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0
    
    # Расчет общей доходности и CAGR
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    total_return = df["strategy_equity"].iloc[-1] - 1
    years_span = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years_span) - 1 if years_span > 0 else 0
    
    # Количество сделок
    num_trades = int(df[signal_column].sum())
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "cagr": cagr,
        "num_trades": num_trades,
        "total_return": total_return
    }

# ------------------------- ЗАГРУЗКА ДАННЫХ -----------------------------------
print("Загрузка данных...")
data = load_all_tickers()
print(f"Загружено {len(data)} строк данных")

# ------------------------- ТЕСТИРОВАНИЕ СТРАТЕГИЙ -----------------------------------
print("\n=== Wave 16: Advanced statistical measures ===\n")

results = []

# 1. Normalized ATR
print("Тестирую normalized ATR...")
thresholds = [0.7, 0.8, 0.9, 1.0]
for threshold in thresholds:
    data["signal"] = (data["atr_normalized"].shift(1) < threshold).fillna(False).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"atr_normalized < {threshold:.1f}", perf))

# 2. Return kurtosis
print("\nТестирую return kurtosis...")
thresholds = [-1, 0, 1, 2]
for threshold in thresholds:
    data["signal"] = (data["return_kurtosis"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"return_kurtosis < {threshold}", perf))

# 3. Return entropy
print("\nТестирую return entropy...")
thresholds = [1.0, 1.2, 1.4, 1.6]
for threshold in thresholds:
    data["signal"] = (data["return_entropy"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"return_entropy < {threshold:.1f}", perf))

# 4. Hurst exponent
print("\nТестирую Hurst exponent...")
thresholds = [0.4, 0.5, 0.6, 0.7]
for threshold in thresholds:
    data["signal"] = (data["hurst_exp"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hurst_exp > {threshold:.1f}", perf))

# 5. Volatility asymmetry
print("\nТестирую volatility asymmetry...")
thresholds = [0.8, 0.9, 1.0, 1.1]
for threshold in thresholds:
    data["signal"] = (data["vol_asymmetry"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_asymmetry < {threshold:.1f}", perf))

# 6. Consecutive calm days
print("\nТестирую consecutive calm days...")
thresholds = [3, 5, 7, 10]
for threshold in thresholds:
    data["signal"] = (data["consecutive_calm"].shift(1) >= threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"consecutive_calm >= {threshold}", perf))

# 7. Day of week effects
print("\nТестирую day of week effects...")
data["signal"] = data["is_monday"].shift(1).fillna(0).fillna(False).astype(int)
perf = calculate_strategy_performance(data, "signal")
results.append(("is_monday", perf))

data["signal"] = data["is_friday"].shift(1).fillna(0).fillna(False).astype(int)
perf = calculate_strategy_performance(data, "signal")
results.append(("is_friday", perf))

# 8. Volatility momentum
print("\nТестирую volatility momentum...")
thresholds = [-0.005, -0.003, -0.001, 0]
for threshold in thresholds:
    data["signal"] = (data["vol_momentum_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_momentum_ma5 < {threshold:.3f}", perf))

# 9. Volatility percentile rank
print("\nТестирую volatility percentile rank...")
thresholds = [0.2, 0.3, 0.4, 0.5]
for threshold in thresholds:
    data["signal"] = (data["vol_percentile_rank"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_percentile_rank < {threshold:.1f}", perf))

# 10. Volatility z-score (EMA)
print("\nТестирую volatility z-score (EMA)...")
thresholds = [-1.5, -1.0, -0.5, 0]
for threshold in thresholds:
    data["signal"] = (data["vol_zscore_ema"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_zscore_ema < {threshold:.1f}", perf))

# 11. Range efficiency
print("\nТестирую range efficiency...")
thresholds = [0.3, 0.4, 0.5, 0.6]
for threshold in thresholds:
    data["signal"] = (data["range_efficiency_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"range_efficiency_ma5 < {threshold:.1f}", perf))

# 12. Volatility autocorrelation
print("\nТестирую volatility autocorrelation...")
thresholds = [0.1, 0.2, 0.3, 0.4]
for threshold in thresholds:
    data["signal"] = (data["vol_autocorr"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_autocorr > {threshold:.1f}", perf))

# 13. Cross-ticker measures
print("\nТестирую cross-ticker measures...")
# Average volatility across markets
thresholds = [0.015, 0.02, 0.025, 0.03]
for threshold in thresholds:
    data["signal"] = (data["avg_hl_range_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"avg_hl_range_ma5 < {threshold:.3f}", perf))

# Volatility dispersion
thresholds = [0.002, 0.003, 0.004, 0.005]
for threshold in thresholds:
    data["signal"] = (data["vol_dispersion_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_dispersion_ma5 < {threshold:.3f}", perf))

# Market-wide calm
thresholds = [0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    data["signal"] = (data["market_calm_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"market_calm_ma5 > {threshold:.1f}", perf))

# Beta-adjusted volatility
thresholds = [0.02, 0.025, 0.03, 0.035]
for threshold in thresholds:
    data["signal"] = (data["beta_adj_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"beta_adj_vol_ma5 < {threshold:.3f}", perf))

# Tech vs Market volatility
thresholds = [0.9, 1.0, 1.1, 1.2]
for threshold in thresholds:
    data["signal"] = (data["tech_vs_market_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"tech_vs_market_ma5 < {threshold:.1f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 16 ===")
sorted_results = sorted(results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)

print("\nТоп 10 результатов по Sharpe Ratio:")
for i, (condition, perf) in enumerate(sorted_results[:10], 1):
    print(f"\n{i}. Условие: {condition}")
    print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
    print(f"   CAGR: {perf['cagr']*100:.2f}%")
    print(f"   Количество сделок: {perf['num_trades']}")

# Фильтруем результаты с количеством сделок > 2500
high_trades_results = [(c, p) for c, p in results if p['num_trades'] > 2500]
if high_trades_results:
    print("\n\nЛучшие результаты с количеством сделок > 2500:")
    sorted_high_trades = sorted(high_trades_results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    for i, (condition, perf) in enumerate(sorted_high_trades[:5], 1):
        print(f"\n{i}. Условие: {condition}")
        print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
        print(f"   CAGR: {perf['cagr']*100:.2f}%")
        print(f"   Количество сделок: {perf['num_trades']}")

# Проверка на достижение цели
goal_achieved = False
for condition, perf in results:
    if perf['sharpe_ratio'] >= 1.4 and perf['num_trades'] > 2500:
        print(f"\n🎯 ЦЕЛЬ ДОСТИГНУТА! Условие: {condition}")
        print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
        print(f"   CAGR: {perf['cagr']*100:.2f}%") 
        print(f"   Количество сделок: {perf['num_trades']}")
        goal_achieved = True
        break

if not goal_achieved:
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 17...")
    print(f"\nЛучший результат: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")