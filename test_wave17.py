#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 17: Non-linear transformations and advanced regime detection
"""

import pandas as pd
import numpy as np
from scipy import stats, signal

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
    df["close_pct"] = df["close"].pct_change()
    
    # Non-linear transformations
    # 1. Power transformations of H-L range
    df["hl_range_sqrt"] = np.sqrt(df["hl_range"])
    df["hl_range_cbrt"] = np.cbrt(df["hl_range"])  # Cube root
    df["hl_range_squared"] = df["hl_range"] ** 2
    df["hl_range_log"] = np.log1p(df["hl_range"])
    
    # 2. Sigmoid transformation
    df["hl_range_sigmoid"] = 1 / (1 + np.exp(-100 * (df["hl_range"] - 0.02)))
    
    # 3. Tanh transformation
    df["hl_range_tanh"] = np.tanh(50 * (df["hl_range"] - 0.02))
    
    # 4. Polynomial features
    df["hl_range_poly2"] = df["hl_range"] - 0.02 * df["hl_range"] ** 2
    df["hl_range_poly3"] = df["hl_range"] - 0.02 * df["hl_range"] ** 2 + 0.001 * df["hl_range"] ** 3
    
    # 5. Exponential decay of volatility
    alpha_decay = 0.1
    df["hl_range_decay"] = 0.0
    for i in range(1, len(df)):
        df.loc[i, "hl_range_decay"] = df.loc[i-1, "hl_range_decay"] * (1 - alpha_decay) + df.loc[i, "hl_range"] * alpha_decay
    
    # 6. Kernel-based transformations (RBF-like)
    df["hl_range_rbf"] = np.exp(-((df["hl_range"] - 0.015) ** 2) / (2 * 0.005 ** 2))
    
    # 7. Distance from median volatility
    df["hl_range_median_dist"] = np.abs(df["hl_range"] - df["hl_range"].rolling(window=50).median())
    
    # 8. Percentile transformations
    df["hl_range_percentile"] = df["hl_range"].rolling(window=100).rank(pct=True)
    df["hl_range_percentile_squared"] = df["hl_range_percentile"] ** 2
    
    # Regime detection
    # 9. Simple regime detection using thresholds
    df["low_vol_regime"] = (df["hl_range"].rolling(window=20).mean() < df["hl_range"].rolling(window=100).quantile(0.3)).astype(int)
    df["high_vol_regime"] = (df["hl_range"].rolling(window=20).mean() > df["hl_range"].rolling(window=100).quantile(0.7)).astype(int)
    
    # 10. Regime duration
    df["regime_duration"] = 0
    current_regime = 0
    duration = 0
    for i in range(len(df)):
        if i == 0:
            current_regime = df.loc[i, "low_vol_regime"]
            duration = 1
        else:
            if df.loc[i, "low_vol_regime"] == current_regime:
                duration += 1
            else:
                current_regime = df.loc[i, "low_vol_regime"]
                duration = 1
        df.loc[i, "regime_duration"] = duration if current_regime == 1 else 0
    
    # 11. Change point detection (simplified)
    df["vol_change_point"] = 0.0
    window = 20
    for i in range(window*2, len(df)):
        before = df["hl_range"].iloc[i-window*2:i-window].mean()
        after = df["hl_range"].iloc[i-window:i].mean()
        df.loc[i, "vol_change_point"] = np.abs(after - before) / (before + 0.0001)
    
    # 12. Volatility momentum with non-linear scaling
    df["vol_momentum"] = df["hl_range"].rolling(window=5).mean() - df["hl_range"].rolling(window=20).mean()
    df["vol_momentum_scaled"] = np.sign(df["vol_momentum"]) * np.sqrt(np.abs(df["vol_momentum"]))
    
    # Market microstructure proxies
    # 13. Amihud illiquidity proxy (using range as proxy for volume)
    df["amihud_proxy"] = np.abs(df["close_pct"]) / (df["hl_range"] + 0.0001)
    df["amihud_proxy_ma5"] = df["amihud_proxy"].rolling(window=5).mean()
    
    # 14. Bid-ask spread proxy
    df["ba_spread_proxy"] = 2 * np.sqrt(df["hl_range"] * np.abs(df["close_pct"]))
    df["ba_spread_proxy_ma5"] = df["ba_spread_proxy"].rolling(window=5).mean()
    
    # Time series decomposition (simplified)
    # 15. Detrended volatility
    df["hl_range_trend"] = df["hl_range"].rolling(window=50).mean()
    df["hl_range_detrended"] = df["hl_range"] / (df["hl_range_trend"] + 0.0001)
    
    # 16. Seasonal component (day of week effect)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["weekly_vol_avg"] = df.groupby("day_of_week")["hl_range"].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
    df["vol_seasonal_adj"] = df["hl_range"] / (df["weekly_vol_avg"] + 0.0001)
    
    # Apply moving averages to key indicators
    for col in ["hl_range_sqrt", "hl_range_cbrt", "hl_range_log", "hl_range_sigmoid", 
                "hl_range_rbf", "hl_range_percentile", "hl_range_decay"]:
        df[f"{col}_ma5"] = df[col].rolling(window=5).mean()
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Merge on Date
    data = qqq.copy()
    data = data.merge(spy[["Date", "hl_range", "hl_range_sqrt", "hl_range_log", "hl_range_rbf"]], 
                      on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[["Date", "hl_range", "hl_range_sqrt", "hl_range_log", "hl_range_rbf"]], 
                      on="Date", suffixes=("", "_xlk"))
    
    # Cross-ticker measures with non-linear transformations
    # 1. Average volatility with different transformations
    data["avg_hl_range"] = (data["hl_range"] + data["hl_range_spy"] + data["hl_range_xlk"]) / 3
    data["avg_hl_range_sqrt"] = (data["hl_range_sqrt"] + data["hl_range_sqrt_spy"] + data["hl_range_sqrt_xlk"]) / 3
    data["avg_hl_range_log"] = (data["hl_range_log"] + data["hl_range_log_spy"] + data["hl_range_log_xlk"]) / 3
    
    # 2. Geometric mean of volatilities
    data["geom_mean_vol"] = np.cbrt(data["hl_range"] * data["hl_range_spy"] * data["hl_range_xlk"])
    
    # 3. Harmonic mean of volatilities
    data["harm_mean_vol"] = 3 / (1/(data["hl_range"]+0.0001) + 1/(data["hl_range_spy"]+0.0001) + 1/(data["hl_range_xlk"]+0.0001))
    
    # 4. Max and min volatility across tickers
    data["max_vol"] = data[["hl_range", "hl_range_spy", "hl_range_xlk"]].max(axis=1)
    data["min_vol"] = data[["hl_range", "hl_range_spy", "hl_range_xlk"]].min(axis=1)
    data["vol_range"] = data["max_vol"] - data["min_vol"]
    
    # 5. Volatility synchronization (all low)
    data["all_low_vol"] = (
        (data["hl_range"] < data["hl_range"].rolling(window=20).quantile(0.3)) &
        (data["hl_range_spy"] < data["hl_range_spy"].rolling(window=20).quantile(0.3)) &
        (data["hl_range_xlk"] < data["hl_range_xlk"].rolling(window=20).quantile(0.3))
    ).astype(int)
    
    # 6. Non-linear cross-ticker interaction
    data["vol_interaction"] = data["hl_range"] * data["hl_range_spy"] / (data["hl_range_xlk"] + 0.0001)
    
    # 7. Principal component approximation (simplified)
    # Standardize volatilities
    vol_matrix = data[["hl_range", "hl_range_spy", "hl_range_xlk"]].values
    vol_matrix_np = np.array(vol_matrix)  # Ensure numpy array
    vol_std = (vol_matrix_np - np.nanmean(vol_matrix_np, axis=0)) / (np.nanstd(vol_matrix_np, axis=0) + 0.0001)
    # First PC approximation (equal weights as simplification)
    data["vol_pc1"] = np.nanmean(vol_std, axis=1)
    
    # Apply moving averages to cross-ticker measures
    for col in ["avg_hl_range", "avg_hl_range_sqrt", "avg_hl_range_log", "geom_mean_vol", 
                "harm_mean_vol", "vol_range", "vol_interaction", "vol_pc1"]:
        data[f"{col}_ma5"] = data[col].rolling(window=5).mean()
    
    data["all_low_vol_ma5"] = data["all_low_vol"].rolling(window=5).mean()
    
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
print("\n=== Wave 17: Non-linear transformations ===\n")

results = []

# 1. Square root transformation
print("Тестирую square root transformation...")
thresholds = [0.12, 0.14, 0.16, 0.18]
for threshold in thresholds:
    data["signal"] = (data["avg_hl_range_sqrt_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"avg_hl_range_sqrt_ma5 < {threshold:.2f}", perf))

# 2. Log transformation
print("\nТестирую log transformation...")
thresholds = [-4.5, -4.0, -3.5, -3.0]
for threshold in thresholds:
    data["signal"] = (data["avg_hl_range_log_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"avg_hl_range_log_ma5 < {threshold:.1f}", perf))

# 3. Geometric mean
print("\nТестирую geometric mean...")
thresholds = [0.015, 0.02, 0.025, 0.03]
for threshold in thresholds:
    data["signal"] = (data["geom_mean_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"geom_mean_vol_ma5 < {threshold:.3f}", perf))

# 4. Harmonic mean
print("\nТестирую harmonic mean...")
thresholds = [0.015, 0.02, 0.025, 0.03]
for threshold in thresholds:
    data["signal"] = (data["harm_mean_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"harm_mean_vol_ma5 < {threshold:.3f}", perf))

# 5. RBF transformation (single ticker)
print("\nТестирую RBF transformation...")
thresholds = [0.5, 0.6, 0.7, 0.8]
for threshold in thresholds:
    data["signal"] = (data["hl_range_rbf_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_rbf_ma5 > {threshold:.1f}", perf))

# 6. Volatility percentile squared
print("\nТестирую volatility percentile squared...")
thresholds = [0.1, 0.15, 0.2, 0.25]
for threshold in thresholds:
    data["signal"] = (data["hl_range_percentile_squared"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_percentile_squared < {threshold:.2f}", perf))

# 7. Volatility decay
print("\nТестирую volatility decay...")
thresholds = [0.015, 0.02, 0.025, 0.03]
for threshold in thresholds:
    data["signal"] = (data["hl_range_decay_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_decay_ma5 < {threshold:.3f}", perf))

# 8. Regime duration
print("\nТестирую regime duration...")
thresholds = [5, 10, 15, 20]
for threshold in thresholds:
    data["signal"] = (data["regime_duration"].shift(1) >= threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"regime_duration >= {threshold}", perf))

# 9. All tickers low volatility
print("\nТестирую all tickers low volatility...")
thresholds = [0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    data["signal"] = (data["all_low_vol_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"all_low_vol_ma5 > {threshold:.1f}", perf))

# 10. Volatility interaction
print("\nТестирую volatility interaction...")
thresholds = [0.0002, 0.0003, 0.0004, 0.0005]
for threshold in thresholds:
    data["signal"] = (data["vol_interaction_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_interaction_ma5 < {threshold:.4f}", perf))

# 11. First principal component
print("\nТестирую first principal component...")
thresholds = [-1.0, -0.5, 0, 0.5]
for threshold in thresholds:
    data["signal"] = (data["vol_pc1_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_pc1_ma5 < {threshold:.1f}", perf))

# 12. Min volatility across tickers
print("\nТестирую min volatility across tickers...")
thresholds = [0.01, 0.015, 0.02, 0.025]
for threshold in thresholds:
    data["signal"] = (data["min_vol"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"min_vol < {threshold:.3f}", perf))

# 13. Volatility range (max - min)
print("\nТестирую volatility range...")
thresholds = [0.005, 0.01, 0.015, 0.02]
for threshold in thresholds:
    data["signal"] = (data["vol_range_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_range_ma5 < {threshold:.3f}", perf))

# 14. Detrended volatility
print("\nТестирую detrended volatility...")
thresholds = [0.7, 0.8, 0.9, 1.0]
for threshold in thresholds:
    data["signal"] = (data["hl_range_detrended"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_detrended < {threshold:.1f}", perf))

# 15. Amihud illiquidity proxy
print("\nТестирую Amihud illiquidity proxy...")
thresholds = [0.5, 1.0, 1.5, 2.0]
for threshold in thresholds:
    data["signal"] = (data["amihud_proxy_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"amihud_proxy_ma5 < {threshold:.1f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 17 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 18...")
    print(f"\nЛучший результат: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")