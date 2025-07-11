#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 6: Нелинейные преобразования и специальные условия
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
    
    # День недели (0 = понедельник, 4 = пятница)
    df["weekday"] = df["Date"].dt.dayofweek
    
    # Логарифмическое преобразование (безопасное)
    df["log_overnight"] = np.sign(df["overnight_return"]) * np.log1p(np.abs(df["overnight_return"]))
    
    # Синусоидальное преобразование
    df["sin_overnight"] = np.sin(df["overnight_return"] * 100)  # Масштабируем для лучшего эффекта
    
    # Кумулятивная сумма ночных доходностей за последние 5 дней
    df["cumsum_5d"] = df["overnight_return"].rolling(window=5).sum()
    
    # Отношение текущей ночной доходности к средней за 10 дней
    df["overnight_ma10"] = df["overnight_return"].rolling(window=10).mean()
    df["overnight_ratio_ma10"] = df["overnight_return"] / (df["overnight_ma10"] + 0.0001)
    
    # Количество последовательных положительных/отрицательных ночных доходностей
    df["consecutive_sign"] = 0
    current_sign = 0
    current_count = 0
    for i in range(len(df)):
        if i == 0:
            continue
        if df.loc[i, "overnight_return"] > 0:
            if current_sign == 1:
                current_count += 1
            else:
                current_sign = 1
                current_count = 1
        elif df.loc[i, "overnight_return"] < 0:
            if current_sign == -1:
                current_count -= 1
            else:
                current_sign = -1
                current_count = -1
        df.loc[i, "consecutive_sign"] = current_count
    
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
print("\n=== Wave 6: Нелинейные преобразования ===\n")

results = []

# Тесты с логарифмическим преобразованием
print("Тестирую условия с логарифмическим преобразованием...")
log_thresholds = [-0.01, -0.008, -0.006, -0.004, -0.002, 0]
for threshold in log_thresholds:
    data["signal"] = (data["log_overnight"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"log_overnight > {threshold:.4f}", perf))

# Тесты с синусоидальным преобразованием
print("\nТестирую условия с синусоидальным преобразованием...")
sin_thresholds = [-0.5, -0.2, 0, 0.2, 0.5]
for threshold in sin_thresholds:
    data["signal"] = (data["sin_overnight"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"sin_overnight > {threshold:.1f}", perf))

# Тесты с кумулятивной суммой
print("\nТестирую условия с кумулятивной суммой...")
cumsum_thresholds = [-0.03, -0.02, -0.015, -0.01, -0.005, 0]
for threshold in cumsum_thresholds:
    data["signal"] = (data["cumsum_5d"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"cumsum_5d > {threshold:.4f}", perf))

# Тесты с отношением к MA10
print("\nТестирую условия с отношением к MA10...")
ratio_thresholds = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3]
for threshold in ratio_thresholds:
    data["signal"] = (data["overnight_ratio_ma10"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_ratio_ma10 > {threshold:.1f}", perf))

# Тесты с последовательными знаками
print("\nТестирую условия с последовательными знаками...")
consec_conditions = [
    ("consecutive_sign > -2", data["consecutive_sign"].shift(1) > -2),
    ("consecutive_sign > -3", data["consecutive_sign"].shift(1) > -3),
    ("consecutive_sign < 2", data["consecutive_sign"].shift(1) < 2),
    ("consecutive_sign < 3", data["consecutive_sign"].shift(1) < 3),
    ("abs(consecutive_sign) < 2", np.abs(data["consecutive_sign"].shift(1)) < 2),
    ("abs(consecutive_sign) < 3", np.abs(data["consecutive_sign"].shift(1)) < 3),
]
for condition_name, condition in consec_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# Тесты с днями недели
print("\nТестирую условия с днями недели...")
weekday_conditions = [
    ("not_monday", data["weekday"] != 0),
    ("not_friday", data["weekday"] != 4),
    ("mid_week", data["weekday"].isin([1, 2, 3])),
    ("week_start", data["weekday"].isin([0, 1])),
    ("week_end", data["weekday"].isin([3, 4])),
]
for condition_name, condition in weekday_conditions:
    # Комбинируем с лучшим условием из предыдущих волн
    data["signal"] = (condition & (data["overnight_return"].shift(1) > -0.01)).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"{condition_name} & overnight > -0.01", perf))

# Экстремальные условия
print("\nТестирую экстремальные условия...")
extreme_conditions = [
    ("overnight^3 > -0.000001", data["overnight_return"].shift(1) ** 3 > -0.000001),
    ("overnight^4 < 0.0000001", data["overnight_return"].shift(1) ** 4 < 0.0000001),
    ("exp(overnight*10) > 0.9", np.exp(data["overnight_return"].shift(1) * 10) > 0.9),
    ("tanh(overnight*100) > -0.5", np.tanh(data["overnight_return"].shift(1) * 100) > -0.5),
]
for condition_name, condition in extreme_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 6 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 7...")