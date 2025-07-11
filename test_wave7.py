#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 7: Детальное исследование квадратичных условий и новые подходы
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
    
    # Квадрат и другие степени
    df["overnight_squared"] = df["overnight_return"] ** 2
    df["overnight_cubed"] = df["overnight_return"] ** 3
    df["overnight_sqrt"] = np.sign(df["overnight_return"]) * np.sqrt(np.abs(df["overnight_return"]))
    
    # Скользящие статистики квадрата
    df["overnight_squared_ma5"] = df["overnight_squared"].rolling(window=5).mean()
    df["overnight_squared_std5"] = df["overnight_squared"].rolling(window=5).std()
    
    # Медиана и персентили
    df["overnight_median5"] = df["overnight_return"].rolling(window=5).median()
    df["overnight_q25_5"] = df["overnight_return"].rolling(window=5).quantile(0.25)
    df["overnight_q75_5"] = df["overnight_return"].rolling(window=5).quantile(0.75)
    
    # Skewness и kurtosis
    df["overnight_skew5"] = df["overnight_return"].rolling(window=5).skew()
    df["overnight_kurt5"] = df["overnight_return"].rolling(window=5).kurt()
    
    # Максимум квадрата за период
    df["overnight_squared_max5"] = df["overnight_squared"].rolling(window=5).max()
    df["overnight_squared_min5"] = df["overnight_squared"].rolling(window=5).min()
    
    # Отношение к историческому среднему квадрату
    df["overnight_squared_ma20"] = df["overnight_squared"].rolling(window=20).mean()
    df["overnight_squared_ratio"] = df["overnight_squared"] / (df["overnight_squared_ma20"] + 0.000001)
    
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
print("\n=== Wave 7: Детальное исследование квадратичных условий ===\n")

results = []

# Очень детальная настройка квадрата около 0.0001
print("Тестирую детальные пороги для квадрата...")
squared_thresholds = np.arange(0.00005, 0.00015, 0.000001)  # От 0.005% до 0.015% с шагом 0.0001%
for i, threshold in enumerate(squared_thresholds[:30]):  # Ограничиваем для скорости
    data["signal"] = (data["overnight_squared"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared < {threshold:.6f}", perf))
    
    if (i + 1) % 10 == 0:
        print(f"Завершено {i+1} тестов квадратичных порогов...")

# Тесты со скользящим средним квадрата
print("\nТестирую условия со скользящим средним квадрата...")
ma_squared_thresholds = [0.00005, 0.00008, 0.0001, 0.00012, 0.00015]
for threshold in ma_squared_thresholds:
    data["signal"] = (data["overnight_squared_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ma5 < {threshold:.5f}", perf))

# Тесты с отношением квадрата к среднему
print("\nТестирую условия с отношением квадрата...")
ratio_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
for threshold in ratio_thresholds:
    data["signal"] = (data["overnight_squared_ratio"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ratio < {threshold:.1f}", perf))

# Тесты с медианой
print("\nТестирую условия с медианой...")
median_thresholds = [-0.002, -0.001, 0, 0.001, 0.002]
for threshold in median_thresholds:
    data["signal"] = (data["overnight_median5"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_median5 > {threshold:.4f}", perf))

# Тесты со skewness
print("\nТестирую условия со skewness...")
skew_thresholds = [-1, -0.5, 0, 0.5, 1]
for threshold in skew_thresholds:
    data["signal"] = (data["overnight_skew5"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_skew5 > {threshold:.1f}", perf))

# Тесты с максимумом квадрата
print("\nТестирую условия с максимумом квадрата...")
max_squared_thresholds = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
for threshold in max_squared_thresholds:
    data["signal"] = (data["overnight_squared_max5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_max5 < {threshold:.5f}", perf))

# Комбинированные условия с квадратом
print("\nТестирую комбинированные условия...")
# Квадрат < порог И сама доходность > другого порога
combo_conditions = [
    ("squared<0.0001 & return>-0.015", 
     (data["overnight_squared"].shift(1) < 0.0001) & (data["overnight_return"].shift(1) > -0.015)),
    ("squared<0.0001 & return>-0.01", 
     (data["overnight_squared"].shift(1) < 0.0001) & (data["overnight_return"].shift(1) > -0.01)),
    ("squared<0.00008 & return>-0.01", 
     (data["overnight_squared"].shift(1) < 0.00008) & (data["overnight_return"].shift(1) > -0.01)),
    ("squared<0.00012 & return>-0.01", 
     (data["overnight_squared"].shift(1) < 0.00012) & (data["overnight_return"].shift(1) > -0.01)),
]
for condition_name, condition in combo_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# Тесты с корнем
print("\nТестирую условия с корнем...")
sqrt_thresholds = [-0.1, -0.08, -0.06, -0.04, -0.02, 0]
for threshold in sqrt_thresholds:
    data["signal"] = (data["overnight_sqrt"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_sqrt > {threshold:.2f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 7 ===")
sorted_results = sorted(results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)

print("\nТоп 5 результатов по Sharpe Ratio:")
for i, (condition, perf) in enumerate(sorted_results[:5], 1):
    print(f"\n{i}. Условие: {condition}")
    print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
    print(f"   CAGR: {perf['cagr']*100:.2f}%")
    print(f"   Количество сделок: {perf['num_trades']}")

# Фильтруем результаты с количеством сделок > 2500
high_trades_results = [(c, p) for c, p in results if p['num_trades'] > 2500]
if high_trades_results:
    print("\n\nЛучшие результаты с количеством сделок > 2500:")
    sorted_high_trades = sorted(high_trades_results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    for i, (condition, perf) in enumerate(sorted_high_trades[:3], 1):
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 8...")