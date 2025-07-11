#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 12: Радикальные новые подходы - более сложные статистические функции
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
    df = (
        df[df["Date"] >= pd.Timestamp("2006-01-01")]
          .sort_values("Date")
          .reset_index(drop=True)
    )
    # Вычисляем ночную доходность: от close до open следующего дня
    df["prev_close"] = df["close"].shift(1)
    df["overnight_return"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    # Для стратегии нужна доходность от сегодняшнего close до завтрашнего open
    df["next_open"] = df["open"].shift(-1)
    df["next_overnight_return"] = (df["next_open"] - df["close"]) / df["close"]
    
    # Квадрат ночной доходности
    df["overnight_squared"] = df["overnight_return"] ** 2
    
    # Entropy-based measure (энтропия последних N значений)
    def rolling_entropy(series, window):
        def entropy(x):
            if len(x) == 0:
                return 0
            counts = pd.Series(x).value_counts()
            probs = counts / len(x)
            return -np.sum(probs * np.log2(probs + 1e-10))
        return series.rolling(window=window).apply(entropy, raw=True)
    
    # Дискретизируем доходность для расчета энтропии
    df["overnight_discrete"] = pd.cut(df["overnight_return"], bins=5, labels=False)
    df["overnight_entropy5"] = rolling_entropy(df["overnight_discrete"], 5)
    
    # Mean Absolute Deviation (MAD)
    df["overnight_mad5"] = df["overnight_return"].rolling(window=5).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df["overnight_squared_mad5"] = df["overnight_squared"].rolling(window=5).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    
    # Gini coefficient для измерения неравенства в распределении
    def gini(x):
        sorted_x = np.sort(np.abs(x))
        n = len(sorted_x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(n) + 1) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] != 0 else 0
    
    df["overnight_gini5"] = df["overnight_return"].rolling(window=5).apply(gini)
    
    # Hurst exponent (мера персистентности)
    def hurst_exponent(ts, max_lag=5):
        if len(ts) < max_lag:
            return 0.5
        lags = range(2, min(max_lag, len(ts)))
        tau = []
        for lag in lags:
            pp = np.array(ts[:-lag])
            y = ts[lag:]
            tau.append(np.std(y - pp))
        if len(tau) > 1:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        return 0.5
    
    df["overnight_hurst10"] = df["overnight_return"].rolling(window=10).apply(lambda x: hurst_exponent(x.values))
    
    # Trimmed mean (усеченное среднее)
    df["overnight_squared_trimmed5"] = df["overnight_squared"].rolling(window=5).apply(
        lambda x: stats.trim_mean(x, 0.2)  # обрезаем 20% с каждой стороны
    )
    
    # Количество "спокойных" дней из последних N
    df["calm_days_5"] = (df["overnight_squared"] < 0.0001).rolling(window=5).sum()
    df["calm_days_10"] = (df["overnight_squared"] < 0.0001).rolling(window=10).sum()
    
    # Z-score squared
    df["overnight_zscore5"] = (df["overnight_return"] - df["overnight_return"].rolling(5).mean()) / df["overnight_return"].rolling(5).std()
    df["overnight_zscore_squared5"] = df["overnight_zscore5"] ** 2
    
    return df

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
data = load_ticker("4 - QQQ.csv")
print(f"Загружено {len(data)} строк данных")

# ------------------------- ТЕСТИРОВАНИЕ СТРАТЕГИЙ -----------------------------------
print("\n=== Wave 12: Радикальные подходы - entropy, MAD, Gini, Hurst ===\n")

results = []

# Тестирование MAD
print("Тестирую Mean Absolute Deviation...")
mad_thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]
for threshold in mad_thresholds:
    # MAD overnight
    data["signal"] = (data["overnight_mad5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_mad5 < {threshold:.4f}", perf))
    
    # MAD squared
    data["signal"] = (data["overnight_squared_mad5"].shift(1) < threshold/10).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_mad5 < {threshold/10:.5f}", perf))

# Тестирование Gini coefficient
print("\nТестирую Gini coefficient...")
gini_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in gini_thresholds:
    data["signal"] = (data["overnight_gini5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_gini5 < {threshold:.1f}", perf))

# Тестирование Hurst exponent
print("\nТестирую Hurst exponent...")
hurst_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in hurst_thresholds:
    data["signal"] = (data["overnight_hurst10"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_hurst10 < {threshold:.1f}", perf))

# Тестирование trimmed mean
print("\nТестирую Trimmed mean...")
trimmed_thresholds = [0.00008, 0.0001, 0.00012, 0.00014, 0.00015, 0.00016]
for threshold in trimmed_thresholds:
    data["signal"] = (data["overnight_squared_trimmed5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_trimmed5 < {threshold:.5f}", perf))

# Тестирование calm days
print("\nТестирую количество спокойных дней...")
calm_thresholds = [2, 3, 4, 5]
for threshold in calm_thresholds:
    # 5-day window
    data["signal"] = (data["calm_days_5"].shift(1) >= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"calm_days_5 >= {threshold}", perf))
    
    # 10-day window
    data["signal"] = (data["calm_days_10"].shift(1) >= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"calm_days_10 >= {threshold}", perf))

# Тестирование Z-score squared
print("\nТестирую Z-score squared...")
zscore_thresholds = [0.5, 1.0, 1.5, 2.0]
for threshold in zscore_thresholds:
    data["signal"] = (data["overnight_zscore_squared5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_zscore_squared5 < {threshold:.1f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 12 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 13...")