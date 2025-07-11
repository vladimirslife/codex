#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 5: Межрыночные связи и использование SPY/XLK
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
    
    # Дневная доходность
    df["daily_return"] = (df["close"] - df["open"]) / df["open"]
    
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
qqq_data = load_ticker("4 - QQQ.csv")
spy_data = load_ticker("4 - SPY.csv")
xlk_data = load_ticker("4 - XLK.csv")
print(f"Загружено: QQQ - {len(qqq_data)}, SPY - {len(spy_data)}, XLK - {len(xlk_data)} строк")

# Объединяем данные
data = qqq_data[["Date", "overnight_return", "next_overnight_return", "daily_return"]].copy()
data = data.rename(columns={"overnight_return": "qqq_overnight", "daily_return": "qqq_daily"})

# Добавляем SPY
spy_cols = spy_data[["Date", "overnight_return", "daily_return"]].rename(
    columns={"overnight_return": "spy_overnight", "daily_return": "spy_daily"}
)
data = data.merge(spy_cols, on="Date", how="left")

# Добавляем XLK
xlk_cols = xlk_data[["Date", "overnight_return", "daily_return"]].rename(
    columns={"overnight_return": "xlk_overnight", "daily_return": "xlk_daily"}
)
data = data.merge(xlk_cols, on="Date", how="left")

# Удаляем строки с NaN
data = data.dropna()
print(f"После объединения: {len(data)} строк")

# ------------------------- ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ -----------------------------------
# Средняя ночная доходность по всем тикерам
data["avg_overnight"] = (data["qqq_overnight"] + data["spy_overnight"] + data["xlk_overnight"]) / 3

# Максимальная и минимальная ночная доходность
data["max_overnight"] = data[["qqq_overnight", "spy_overnight", "xlk_overnight"]].max(axis=1)
data["min_overnight"] = data[["qqq_overnight", "spy_overnight", "xlk_overnight"]].min(axis=1)

# Количество положительных ночных доходностей
data["count_positive_overnight"] = (
    (data["qqq_overnight"] > 0).astype(int) + 
    (data["spy_overnight"] > 0).astype(int) + 
    (data["xlk_overnight"] > 0).astype(int)
)

# Разброс ночных доходностей
data["overnight_spread"] = data["max_overnight"] - data["min_overnight"]

# Отношение QQQ к SPY
data["qqq_spy_ratio"] = data["qqq_overnight"] / (data["spy_overnight"] + 0.0001)

# ------------------------- ТЕСТИРОВАНИЕ СТРАТЕГИЙ -----------------------------------
print("\n=== Wave 5: Межрыночные связи ===\n")

results = []

# Тесты со средней ночной доходностью
print("Тестирую условия со средней ночной доходностью...")
avg_thresholds = [-0.01, -0.008, -0.006, -0.004, -0.002, 0, 0.002]
for threshold in avg_thresholds:
    data["signal"] = (data["avg_overnight"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"avg_overnight > {threshold:.4f}", perf))

# Тесты с SPY ночной доходностью
print("\nТестирую условия с SPY...")
spy_thresholds = [-0.01, -0.008, -0.006, -0.004, -0.002, 0, 0.002]
for threshold in spy_thresholds:
    data["signal"] = (data["spy_overnight"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"spy_overnight > {threshold:.4f}", perf))

# Тесты с XLK ночной доходностью
print("\nТестирую условия с XLK...")
xlk_thresholds = [-0.01, -0.008, -0.006, -0.004, -0.002, 0, 0.002]
for threshold in xlk_thresholds:
    data["signal"] = (data["xlk_overnight"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"xlk_overnight > {threshold:.4f}", perf))

# Тесты с количеством положительных
print("\nТестирую условия с количеством положительных...")
for count in [0, 1, 2, 3]:
    data["signal"] = (data["count_positive_overnight"].shift(1) >= count).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"count_positive_overnight >= {count}", perf))

# Тесты с минимальной ночной доходностью
print("\nТестирую условия с минимальной ночной доходностью...")
min_thresholds = [-0.015, -0.01, -0.008, -0.006, -0.004, -0.002]
for threshold in min_thresholds:
    data["signal"] = (data["min_overnight"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"min_overnight > {threshold:.4f}", perf))

# Тесты с разбросом
print("\nТестирую условия с разбросом...")
spread_thresholds = [0.002, 0.004, 0.006, 0.008, 0.01]
for threshold in spread_thresholds:
    # Малый разброс
    data["signal"] = (data["overnight_spread"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_spread < {threshold:.4f}", perf))

# Тесты с отношением QQQ/SPY
print("\nТестирую условия с отношением QQQ/SPY...")
ratio_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5]
for threshold in ratio_thresholds:
    data["signal"] = (data["qqq_spy_ratio"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"qqq_spy_ratio > {threshold:.1f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 5 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 6...")