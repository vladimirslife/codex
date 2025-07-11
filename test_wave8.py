#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 8: Оптимизация скользящих средних квадрата и поиск лучшего окна
"""

import pandas as pd
import numpy as np

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
    
    # Скользящие средние квадрата с разными окнами
    windows = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    for w in windows:
        df[f"overnight_squared_ma{w}"] = df["overnight_squared"].rolling(window=w).mean()
        df[f"overnight_squared_max{w}"] = df["overnight_squared"].rolling(window=w).max()
        df[f"overnight_squared_min{w}"] = df["overnight_squared"].rolling(window=w).min()
        df[f"overnight_squared_std{w}"] = df["overnight_squared"].rolling(window=w).std()
    
    # Взвешенное скользящее среднее
    df["overnight_squared_ewm5"] = df["overnight_squared"].ewm(span=5, adjust=False).mean()
    df["overnight_squared_ewm10"] = df["overnight_squared"].ewm(span=10, adjust=False).mean()
    
    # Комбинированные метрики
    df["overnight_squared_range5"] = df["overnight_squared_max5"] - df["overnight_squared_min5"]
    df["overnight_squared_cv5"] = df["overnight_squared_std5"] / (df["overnight_squared_ma5"] + 0.000001)
    
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
print("\n=== Wave 8: Оптимизация скользящих средних квадрата ===\n")

results = []

# Очень детальная настройка около 0.00015 для MA5
print("Детальная настройка порога для MA5...")
detailed_thresholds = np.arange(0.00010, 0.00020, 0.000002)  # От 0.01% до 0.02% с шагом 0.0002%
for threshold in detailed_thresholds[:25]:  # Ограничиваем для скорости
    data["signal"] = (data["overnight_squared_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ma5 < {threshold:.6f}", perf))

# Тестирование разных окон с оптимальным порогом около 0.00015
print("\nТестирую разные окна для MA...")
windows = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
test_thresholds = [0.00012, 0.00014, 0.00015, 0.00016, 0.00018]
for w in windows:
    for threshold in test_thresholds:
        col_name = f"overnight_squared_ma{w}"
        if col_name in data.columns:
            data["signal"] = (data[col_name].shift(1) < threshold).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"overnight_squared_ma{w} < {threshold:.5f}", perf))

# Тестирование максимумов с разными окнами
print("\nТестирую максимумы с разными окнами...")
max_thresholds = [0.00020, 0.00025, 0.00030, 0.00035]
for w in [3, 4, 5, 6, 7, 8]:
    for threshold in max_thresholds:
        col_name = f"overnight_squared_max{w}"
        if col_name in data.columns:
            data["signal"] = (data[col_name].shift(1) < threshold).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"overnight_squared_max{w} < {threshold:.5f}", perf))

# Тестирование EWM
print("\nТестирую экспоненциальное взвешенное среднее...")
ewm_thresholds = [0.00010, 0.00012, 0.00014, 0.00015, 0.00016, 0.00018]
for threshold in ewm_thresholds:
    # EWM5
    data["signal"] = (data["overnight_squared_ewm5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ewm5 < {threshold:.5f}", perf))
    
    # EWM10
    data["signal"] = (data["overnight_squared_ewm10"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ewm10 < {threshold:.5f}", perf))

# Тестирование комбинированных метрик
print("\nТестирую комбинированные метрики...")
# Коэффициент вариации
cv_thresholds = [1, 2, 3, 4, 5]
for threshold in cv_thresholds:
    data["signal"] = (data["overnight_squared_cv5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_cv5 < {threshold}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 8 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 9...")