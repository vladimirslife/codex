#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 4: Статистические преобразования и процентильные ранги
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
    
    # Дополнительные показатели для Wave 4
    # Z-score (стандартизация)
    df["overnight_return_mean20"] = df["overnight_return"].rolling(window=20).mean()
    df["overnight_return_std20"] = df["overnight_return"].rolling(window=20).std()
    df["overnight_return_zscore"] = (df["overnight_return"] - df["overnight_return_mean20"]) / df["overnight_return_std20"]
    
    # Процентильный ранг за последние 20 дней
    df["overnight_return_rank20"] = df["overnight_return"].rolling(window=20).rank(pct=True)
    df["overnight_return_rank50"] = df["overnight_return"].rolling(window=50).rank(pct=True)
    
    # EMA (экспоненциально взвешенное среднее)
    df["overnight_return_ema5"] = df["overnight_return"].ewm(span=5, adjust=False).mean()
    df["overnight_return_ema10"] = df["overnight_return"].ewm(span=10, adjust=False).mean()
    
    # Отношение к волатильности
    df["overnight_return_sharpe5"] = df["overnight_return"] / df["overnight_return"].rolling(window=5).std()
    
    # Дневная доходность для комбинирования
    df["daily_return"] = (df["close"] - df["open"]) / df["open"]
    df["total_daily_return"] = df["overnight_return"] + df["daily_return"]
    
    # Квадрат и куб ночной доходности
    df["overnight_return_squared"] = df["overnight_return"] ** 2
    df["overnight_return_cubed"] = df["overnight_return"] ** 3
    
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
print("\n=== Wave 4: Статистические преобразования ===\n")

results = []

# Тесты с Z-score
print("Тестирую условия с Z-score...")
zscore_thresholds = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
for threshold in zscore_thresholds:
    data["signal_zscore"] = (data["overnight_return_zscore"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_zscore")
    results.append((f"overnight_return_zscore > {threshold:.1f}", perf))

# Тесты с процентильными рангами
print("\nТестирую условия с процентильными рангами...")
rank_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for threshold in rank_thresholds:
    # Rank20
    data["signal_rank20"] = (data["overnight_return_rank20"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_rank20")
    results.append((f"overnight_return_rank20 > {threshold:.1f}", perf))
    
    # Rank50
    data["signal_rank50"] = (data["overnight_return_rank50"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_rank50")
    results.append((f"overnight_return_rank50 > {threshold:.1f}", perf))

# Тесты с EMA
print("\nТестирую условия с EMA...")
ema_thresholds = [-0.01, -0.005, -0.002, -0.001, 0, 0.001, 0.002]
for threshold in ema_thresholds:
    # EMA5
    data["signal_ema5"] = (data["overnight_return_ema5"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_ema5")
    results.append((f"overnight_return_ema5 > {threshold:.4f}", perf))
    
    # EMA10
    data["signal_ema10"] = (data["overnight_return_ema10"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_ema10")
    results.append((f"overnight_return_ema10 > {threshold:.4f}", perf))

# Тесты с Sharpe ratio ночной доходности
print("\nТестирую условия с Sharpe ratio...")
sharpe_thresholds = [-1, -0.5, 0, 0.5, 1]
for threshold in sharpe_thresholds:
    data["signal_sharpe5"] = (data["overnight_return_sharpe5"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_sharpe5")
    results.append((f"overnight_return_sharpe5 > {threshold:.1f}", perf))

# Тесты с общей дневной доходностью
print("\nТестирую условия с общей дневной доходностью...")
total_thresholds = [-0.02, -0.015, -0.01, -0.005, 0]
for threshold in total_thresholds:
    data["signal_total"] = (data["total_daily_return"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_total")
    results.append((f"total_daily_return > {threshold:.4f}", perf))

# Тесты с квадратичными преобразованиями
print("\nТестирую условия с квадратичными преобразованиями...")
squared_thresholds = [0.00001, 0.00005, 0.0001, 0.0002]
for threshold in squared_thresholds:
    # Квадрат < порог (малая волатильность)
    data["signal_squared"] = (data["overnight_return_squared"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal_squared")
    results.append((f"overnight_return_squared < {threshold:.5f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 4 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 5...")