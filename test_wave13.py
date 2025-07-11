#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 13: Экзотические подходы - rolling quantiles, сложные комбинации
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
    df["overnight_squared_ma5"] = df["overnight_squared"].rolling(window=5).mean()
    
    # Rolling quantiles квадрата
    df["overnight_squared_q10_20"] = df["overnight_squared"].rolling(window=20).quantile(0.10)
    df["overnight_squared_q20_20"] = df["overnight_squared"].rolling(window=20).quantile(0.20)
    df["overnight_squared_q25_20"] = df["overnight_squared"].rolling(window=20).quantile(0.25)
    df["overnight_squared_q30_20"] = df["overnight_squared"].rolling(window=20).quantile(0.30)
    
    # Interquartile range
    df["overnight_squared_iqr20"] = df["overnight_squared"].rolling(window=20).quantile(0.75) - df["overnight_squared"].rolling(window=20).quantile(0.25)
    
    # Сложная комбинация - расстояние от медианы в единицах IQR
    df["overnight_squared_median20"] = df["overnight_squared"].rolling(window=20).median()
    df["overnight_squared_z_iqr"] = (df["overnight_squared"] - df["overnight_squared_median20"]) / (df["overnight_squared_iqr20"] + 0.000001)
    
    # Количество "спокойных" дней (как в Wave 12)
    df["calm_days_5"] = (df["overnight_squared"] < 0.0001).rolling(window=5).sum()
    df["calm_days_3"] = (df["overnight_squared"] < 0.0001).rolling(window=3).sum()
    df["calm_days_7"] = (df["overnight_squared"] < 0.0001).rolling(window=7).sum()
    
    # Новый подход - относительный ранг в скользящем окне
    df["overnight_squared_rank_pct20"] = df["overnight_squared"].rolling(window=20).rank(pct=True)
    df["overnight_squared_rank_pct10"] = df["overnight_squared"].rolling(window=10).rank(pct=True)
    
    # Логит-преобразование процентильного ранга
    df["overnight_squared_logit_rank20"] = np.log((df["overnight_squared_rank_pct20"] + 0.01) / (1 - df["overnight_squared_rank_pct20"] + 0.01))
    
    # Мультипликативная комбинация
    df["calm_squared_product"] = df["calm_days_5"] * (1 - df["overnight_squared"] * 10000)
    
    # Экспоненциальное взвешивание спокойных дней
    df["calm_days_exp"] = 0.0
    alpha = 0.3
    for i in range(len(df)):
        if i == 0:
            df.loc[i, "calm_days_exp"] = 0
        else:
            if df.loc[i, "overnight_squared"] < 0.0001:
                df.loc[i, "calm_days_exp"] = df.loc[i-1, "calm_days_exp"] * (1 - alpha) + alpha
            else:
                df.loc[i, "calm_days_exp"] = df.loc[i-1, "calm_days_exp"] * (1 - alpha)
    
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
print("\n=== Wave 13: Экзотические подходы ===\n")

results = []

# Тестирование quantiles
print("Тестирую условия на основе квантилей...")
quantile_conditions = [
    ("squared < q10_20", data["overnight_squared"].shift(1) < data["overnight_squared_q10_20"].shift(1)),
    ("squared < q20_20", data["overnight_squared"].shift(1) < data["overnight_squared_q20_20"].shift(1)),
    ("squared < q25_20", data["overnight_squared"].shift(1) < data["overnight_squared_q25_20"].shift(1)),
    ("squared < q30_20", data["overnight_squared"].shift(1) < data["overnight_squared_q30_20"].shift(1)),
]
for condition_name, condition in quantile_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# Тестирование IQR-based conditions
print("\nТестирую условия на основе IQR...")
iqr_thresholds = [-1, -0.5, 0, 0.5, 1]
for threshold in iqr_thresholds:
    data["signal"] = (data["overnight_squared_z_iqr"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"squared_z_iqr < {threshold:.1f}", perf))

# Тестирование calm days с разными окнами
print("\nТестирую calm days с разными окнами...")
calm_conditions = [
    ("calm_days_3 >= 3", data["calm_days_3"].shift(1) >= 3),
    ("calm_days_3 >= 2", data["calm_days_3"].shift(1) >= 2),
    ("calm_days_5 >= 5", data["calm_days_5"].shift(1) >= 5),
    ("calm_days_7 >= 5", data["calm_days_7"].shift(1) >= 5),
    ("calm_days_7 >= 6", data["calm_days_7"].shift(1) >= 6),
]
for condition_name, condition in calm_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# Тестирование процентильных рангов
print("\nТестирую процентильные ранги...")
rank_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
for threshold in rank_thresholds:
    # 10-day window
    data["signal"] = (data["overnight_squared_rank_pct10"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"squared_rank_pct10 < {threshold:.2f}", perf))
    
    # 20-day window
    data["signal"] = (data["overnight_squared_rank_pct20"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"squared_rank_pct20 < {threshold:.2f}", perf))

# Тестирование экспоненциального взвешивания
print("\nТестирую экспоненциальное взвешивание...")
exp_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
for threshold in exp_thresholds:
    data["signal"] = (data["calm_days_exp"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"calm_days_exp > {threshold:.1f}", perf))

# Комбинированные условия лучших результатов
print("\nТестирую комбинации лучших условий...")
best_combinations = [
    ("calm_days_5>=4 & squared<0.00008", 
     (data["calm_days_5"].shift(1) >= 4) & (data["overnight_squared"].shift(1) < 0.00008)),
    ("calm_days_5>=4 & squared_ma5<0.00015", 
     (data["calm_days_5"].shift(1) >= 4) & (data["overnight_squared_ma5"].shift(1) < 0.00015)),
    ("calm_days_5>=3 & squared<0.00005", 
     (data["calm_days_5"].shift(1) >= 3) & (data["overnight_squared"].shift(1) < 0.00005)),
    ("calm_days_3>=3 & squared_ma5<0.00012", 
     (data["calm_days_3"].shift(1) >= 3) & (data["overnight_squared_ma5"].shift(1) < 0.00012)),
]
for condition_name, condition in best_combinations:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 13 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 14...")