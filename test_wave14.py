#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 14: Оптимизация экспоненциального взвешивания и использование high/low данных
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
    
    # High-Low range как мера волатильности
    df["hl_range"] = (df["high"] - df["low"]) / df["open"]
    df["hl_range_squared"] = df["hl_range"] ** 2
    df["hl_range_ma5"] = df["hl_range"].rolling(window=5).mean()
    
    # True Range
    df["true_range"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            np.abs(df["high"] - df["close"].shift(1)),
            np.abs(df["low"] - df["close"].shift(1))
        )
    ) / df["open"]
    df["true_range_ma5"] = df["true_range"].rolling(window=5).mean()
    
    # Количество "спокойных" дней
    df["calm_days_5"] = (df["overnight_squared"] < 0.0001).rolling(window=5).sum()
    
    # Экспоненциальное взвешивание с разными alpha
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        col_name = f"calm_days_exp_{int(alpha*100)}"
        df[col_name] = 0.0
        for i in range(len(df)):
            if i == 0:
                df.loc[i, col_name] = 0
            else:
                if df.loc[i, "overnight_squared"] < 0.0001:
                    df.loc[i, col_name] = df.loc[i-1, col_name] * (1 - alpha) + alpha
                else:
                    df.loc[i, col_name] = df.loc[i-1, col_name] * (1 - alpha)
    
    # Двойное экспоненциальное сглаживание (для трендов)
    alpha_trend = 0.3
    beta_trend = 0.1
    df["calm_days_double_exp"] = 0.0
    df["calm_days_trend"] = 0.0
    for i in range(len(df)):
        if i == 0:
            df.loc[i, "calm_days_double_exp"] = 0
            df.loc[i, "calm_days_trend"] = 0
        else:
            is_calm = 1 if df.loc[i, "overnight_squared"] < 0.0001 else 0
            # Level
            level = alpha_trend * is_calm + (1 - alpha_trend) * (df.loc[i-1, "calm_days_double_exp"] + df.loc[i-1, "calm_days_trend"])
            # Trend
            trend = beta_trend * (level - df.loc[i-1, "calm_days_double_exp"]) + (1 - beta_trend) * df.loc[i-1, "calm_days_trend"]
            df.loc[i, "calm_days_double_exp"] = level
            df.loc[i, "calm_days_trend"] = trend
    
    # Комбинация с High-Low range
    df["calm_hl_score"] = df["calm_days_5"] * (1 - df["hl_range_ma5"] * 10)
    
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
print("\n=== Wave 14: Оптимизация экспоненциального взвешивания ===\n")

results = []

# Тестирование разных alpha для экспоненциального взвешивания
print("Тестирую различные alpha для экспоненциального взвешивания...")
alphas = [10, 20, 30, 40, 50, 60, 70]
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for alpha in alphas:
    col_name = f"calm_days_exp_{alpha}"
    for threshold in thresholds:
        data["signal"] = (data[col_name].shift(1) > threshold).astype(int)
        perf = calculate_strategy_performance(data, "signal")
        results.append((f"calm_exp_a{alpha/100:.1f} > {threshold:.1f}", perf))

# Тестирование двойного экспоненциального сглаживания
print("\nТестирую двойное экспоненциальное сглаживание...")
double_exp_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in double_exp_thresholds:
    data["signal"] = (data["calm_days_double_exp"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"calm_double_exp > {threshold:.1f}", perf))

# Тестирование High-Low range
print("\nТестирую условия на основе High-Low range...")
hl_thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]
for threshold in hl_thresholds:
    # Low volatility based on H-L range
    data["signal"] = (data["hl_range_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_ma5 < {threshold:.3f}", perf))
    
    # True range
    data["signal"] = (data["true_range_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"true_range_ma5 < {threshold:.3f}", perf))

# Комбинированные условия с High-Low
print("\nТестирую комбинированные условия...")
combo_conditions = [
    ("calm_exp>0.5 & hl_range<0.02", 
     (data["calm_days_exp_30"].shift(1) > 0.5) & (data["hl_range_ma5"].shift(1) < 0.02)),
    ("calm_exp>0.4 & hl_range<0.015", 
     (data["calm_days_exp_30"].shift(1) > 0.4) & (data["hl_range_ma5"].shift(1) < 0.015)),
    ("calm_days_5>=4 & true_range<0.02", 
     (data["calm_days_5"].shift(1) >= 4) & (data["true_range_ma5"].shift(1) < 0.02)),
]
for condition_name, condition in combo_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 14 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 15...")