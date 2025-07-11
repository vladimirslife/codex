#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 10: Новые математические преобразования и временные паттерны
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
    
    # Временные признаки
    df["weekday"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day_of_month"] = df["Date"].dt.day
    df["is_month_end"] = (df["Date"].dt.day >= 25).astype(int)
    df["is_month_start"] = (df["Date"].dt.day <= 5).astype(int)
    
    # Квадрат и производные
    df["overnight_squared"] = df["overnight_return"] ** 2
    df["overnight_squared_ma5"] = df["overnight_squared"].rolling(window=5).mean()
    
    # Новые математические преобразования
    # Обратная величина (с защитой от деления на ноль)
    df["overnight_inv"] = 1 / (df["overnight_return"].abs() + 0.001)
    df["overnight_squared_inv"] = 1 / (df["overnight_squared"] + 0.000001)
    
    # Сигмоид-подобная функция
    df["overnight_sigmoid"] = 1 / (1 + np.exp(-df["overnight_return"] * 100))
    
    # Гармоническое среднее квадратов
    df["overnight_squared_harmonic5"] = 5 / (
        1/(df["overnight_squared"] + 0.000001) + 
        1/(df["overnight_squared"].shift(1) + 0.000001) + 
        1/(df["overnight_squared"].shift(2) + 0.000001) + 
        1/(df["overnight_squared"].shift(3) + 0.000001) + 
        1/(df["overnight_squared"].shift(4) + 0.000001)
    )
    
    # Геометрическое среднее квадратов
    df["overnight_squared_geom5"] = (
        df["overnight_squared"] * 
        df["overnight_squared"].shift(1) * 
        df["overnight_squared"].shift(2) * 
        df["overnight_squared"].shift(3) * 
        df["overnight_squared"].shift(4)
    ) ** 0.2
    
    # Ранг квадрата среди последних N дней
    df["overnight_squared_rank10"] = df["overnight_squared"].rolling(window=10).rank()
    df["overnight_squared_rank20"] = df["overnight_squared"].rolling(window=20).rank()
    
    # Детальная оптимизация лучшего условия
    df["overnight_squared_ma4"] = df["overnight_squared"].rolling(window=4).mean()
    df["overnight_squared_ma6"] = df["overnight_squared"].rolling(window=6).mean()
    
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
print("\n=== Wave 10: Новые преобразования и оптимизация ===\n")

results = []

# Детальная оптимизация лучшего комбинированного условия
print("Оптимизирую лучшее комбинированное условие...")
squared_thresholds = np.arange(0.00008, 0.00012, 0.000002)
ma5_thresholds = np.arange(0.00013, 0.00017, 0.000002)
for sq_thresh in squared_thresholds[:10]:
    for ma_thresh in ma5_thresholds[:10]:
        condition = (data["overnight_squared"].shift(1) < sq_thresh) & (data["overnight_squared_ma5"].shift(1) < ma_thresh)
        data["signal"] = condition.astype(int)
        perf = calculate_strategy_performance(data, "signal")
        results.append((f"squared<{sq_thresh:.6f} & ma5<{ma_thresh:.6f}", perf))

# Тестирование гармонического среднего
print("\nТестирую гармоническое среднее...")
harmonic_thresholds = [0.00005, 0.00008, 0.0001, 0.00012, 0.00015, 0.0002]
for threshold in harmonic_thresholds:
    data["signal"] = (data["overnight_squared_harmonic5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_harmonic5 < {threshold:.5f}", perf))

# Тестирование рангов
print("\nТестирую ранги...")
rank_thresholds = [3, 4, 5, 6, 7, 8, 9, 10]
for threshold in rank_thresholds:
    # Rank10
    data["signal"] = (data["overnight_squared_rank10"].shift(1) <= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_rank10 <= {threshold}", perf))
    
    # Rank20
    data["signal"] = (data["overnight_squared_rank20"].shift(1) <= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_rank20 <= {threshold}", perf))

# Тестирование обратных величин
print("\nТестирую обратные величины...")
inv_thresholds = [5000, 8000, 10000, 12000, 15000, 20000]
for threshold in inv_thresholds:
    data["signal"] = (data["overnight_squared_inv"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_inv > {threshold}", perf))

# Тестирование MA4 и MA6
print("\nТестирую MA4 и MA6...")
ma_thresholds = [0.00012, 0.00014, 0.00015, 0.00016, 0.00018]
for threshold in ma_thresholds:
    # MA4
    data["signal"] = (data["overnight_squared_ma4"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ma4 < {threshold:.5f}", perf))
    
    # MA6
    data["signal"] = (data["overnight_squared_ma6"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ma6 < {threshold:.5f}", perf))

# Временные условия в комбинации с лучшим
print("\nТестирую временные условия...")
best_condition = data["overnight_squared_ma5"].shift(1) < 0.00015
temporal_conditions = [
    ("best & not_monday", best_condition & (data["weekday"] != 0)),
    ("best & not_friday", best_condition & (data["weekday"] != 4)),
    ("best & mid_week", best_condition & data["weekday"].isin([1, 2, 3])),
    ("best & month_start", best_condition & (data["is_month_start"] == 1)),
    ("best & month_end", best_condition & (data["is_month_end"] == 1)),
]
for condition_name, condition in temporal_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 10 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 11...")