#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 9: Адаптивные пороги и продвинутые комбинации
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
    
    # Квадрат ночной доходности
    df["overnight_squared"] = df["overnight_return"] ** 2
    df["overnight_squared_ma5"] = df["overnight_squared"].rolling(window=5).mean()
    df["overnight_squared_max5"] = df["overnight_squared"].rolling(window=5).max()
    
    # Адаптивные пороги на основе персентилей
    df["overnight_squared_p30_20"] = df["overnight_squared"].rolling(window=20).quantile(0.30)
    df["overnight_squared_p40_20"] = df["overnight_squared"].rolling(window=20).quantile(0.40)
    df["overnight_squared_p50_20"] = df["overnight_squared"].rolling(window=20).quantile(0.50)
    df["overnight_squared_p60_20"] = df["overnight_squared"].rolling(window=20).quantile(0.60)
    
    df["overnight_squared_ma5_p30_50"] = df["overnight_squared_ma5"].rolling(window=50).quantile(0.30)
    df["overnight_squared_ma5_p40_50"] = df["overnight_squared_ma5"].rolling(window=50).quantile(0.40)
    df["overnight_squared_ma5_p50_50"] = df["overnight_squared_ma5"].rolling(window=50).quantile(0.50)
    
    # Отношение дневной к ночной доходности
    df["day_night_ratio"] = df["daily_return"] / (df["overnight_return"] + 0.0001)
    
    # Комбинированная метрика волатильности
    df["volatility_score"] = df["overnight_squared"] + df["daily_return"].abs()
    df["volatility_score_ma5"] = df["volatility_score"].rolling(window=5).mean()
    
    # Относительная позиция квадрата
    df["overnight_squared_rel_ma20"] = df["overnight_squared"] / (df["overnight_squared"].rolling(window=20).mean() + 0.000001)
    df["overnight_squared_ma5_rel_ma20"] = df["overnight_squared_ma5"] / (df["overnight_squared_ma5"].rolling(window=20).mean() + 0.000001)
    
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
print("\n=== Wave 9: Адаптивные пороги и продвинутые комбинации ===\n")

results = []

# Тесты с адаптивными порогами на основе персентилей
print("Тестирую адаптивные пороги на основе персентилей...")
# Квадрат меньше своего персентиля
adaptive_conditions = [
    ("overnight_squared < p30_20", data["overnight_squared"].shift(1) < data["overnight_squared_p30_20"].shift(1)),
    ("overnight_squared < p40_20", data["overnight_squared"].shift(1) < data["overnight_squared_p40_20"].shift(1)),
    ("overnight_squared < p50_20", data["overnight_squared"].shift(1) < data["overnight_squared_p50_20"].shift(1)),
    ("overnight_squared < p60_20", data["overnight_squared"].shift(1) < data["overnight_squared_p60_20"].shift(1)),
]
for condition_name, condition in adaptive_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# MA5 меньше своего персентиля
adaptive_ma_conditions = [
    ("overnight_squared_ma5 < p30_50", data["overnight_squared_ma5"].shift(1) < data["overnight_squared_ma5_p30_50"].shift(1)),
    ("overnight_squared_ma5 < p40_50", data["overnight_squared_ma5"].shift(1) < data["overnight_squared_ma5_p40_50"].shift(1)),
    ("overnight_squared_ma5 < p50_50", data["overnight_squared_ma5"].shift(1) < data["overnight_squared_ma5_p50_50"].shift(1)),
]
for condition_name, condition in adaptive_ma_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# Тесты с относительными позициями
print("\nТестирую относительные позиции...")
rel_thresholds = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
for threshold in rel_thresholds:
    # Квадрат относительно MA20
    data["signal"] = (data["overnight_squared_rel_ma20"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_rel_ma20 < {threshold:.1f}", perf))
    
    # MA5 относительно своего MA20
    data["signal"] = (data["overnight_squared_ma5_rel_ma20"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_ma5_rel_ma20 < {threshold:.1f}", perf))

# Комбинированные условия с дневной доходностью
print("\nТестирую комбинированные условия с дневной доходностью...")
combo_conditions = [
    ("squared_ma5<0.00015 & daily>0", 
     (data["overnight_squared_ma5"].shift(1) < 0.00015) & (data["daily_return"].shift(1) > 0)),
    ("squared_ma5<0.00015 & daily<0", 
     (data["overnight_squared_ma5"].shift(1) < 0.00015) & (data["daily_return"].shift(1) < 0)),
    ("squared_ma5<0.00015 & abs(daily)<0.01", 
     (data["overnight_squared_ma5"].shift(1) < 0.00015) & (data["daily_return"].shift(1).abs() < 0.01)),
    ("squared_ma5<0.00015 & day_night_ratio>0", 
     (data["overnight_squared_ma5"].shift(1) < 0.00015) & (data["day_night_ratio"].shift(1) > 0)),
]
for condition_name, condition in combo_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# Тесты с комбинированной волатильностью
print("\nТестирую комбинированную волатильность...")
vol_thresholds = [0.002, 0.003, 0.004, 0.005, 0.006]
for threshold in vol_thresholds:
    data["signal"] = (data["volatility_score_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"volatility_score_ma5 < {threshold:.4f}", perf))

# Специальные условия
print("\nТестирую специальные условия...")
special_conditions = [
    ("squared<0.0001 & squared_ma5<0.00015", 
     (data["overnight_squared"].shift(1) < 0.0001) & (data["overnight_squared_ma5"].shift(1) < 0.00015)),
    ("squared_ma5<0.00015 & squared_max5<0.0003", 
     (data["overnight_squared_ma5"].shift(1) < 0.00015) & (data["overnight_squared_max5"].shift(1) < 0.0003)),
    ("squared_ma5<0.00015 & overnight>-0.005", 
     (data["overnight_squared_ma5"].shift(1) < 0.00015) & (data["overnight_return"].shift(1) > -0.005)),
]
for condition_name, condition in special_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 9 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 10...")