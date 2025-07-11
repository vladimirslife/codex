#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 11: Кардинально новые подходы - rolling correlation, weighted averages
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
    
    # Weighted Moving Average с линейными весами
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    df["overnight_squared_wma5"] = wma(df["overnight_squared"], 5)
    df["overnight_squared_wma10"] = wma(df["overnight_squared"], 10)
    
    # Hull Moving Average
    def hma(series, period):
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        wma_half = wma(series, half_period)
        wma_full = wma(series, period)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_period)
    
    df["overnight_squared_hma5"] = hma(df["overnight_squared"], 5)
    
    # Rolling correlation с будущей доходностью
    df["overnight_future_corr10"] = df["overnight_return"].rolling(window=10).corr(df["next_overnight_return"])
    
    # Количество дней с низкой волатильностью подряд
    df["low_vol_streak"] = 0
    low_vol_threshold = 0.0001
    streak = 0
    for i in range(len(df)):
        if df.loc[i, "overnight_squared"] < low_vol_threshold:
            streak += 1
        else:
            streak = 0
        df.loc[i, "low_vol_streak"] = streak
    
    # Расстояние от последнего пика волатильности
    df["days_since_vol_spike"] = 0
    spike_threshold = 0.001
    days = 0
    for i in range(len(df)):
        if df.loc[i, "overnight_squared"] > spike_threshold:
            days = 0
        else:
            days += 1
        df.loc[i, "days_since_vol_spike"] = days
    
    # Percentile rank за разные периоды
    df["overnight_squared_pct10"] = df["overnight_squared"].rolling(window=10).rank(pct=True)
    df["overnight_squared_pct30"] = df["overnight_squared"].rolling(window=30).rank(pct=True)
    
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
print("\n=== Wave 11: Новые подходы - weighted averages, streaks, correlation ===\n")

results = []

# Тестирование WMA
print("Тестирую Weighted Moving Average...")
wma_thresholds = [0.00008, 0.0001, 0.00012, 0.00014, 0.00015, 0.00016, 0.00018]
for threshold in wma_thresholds:
    # WMA5
    data["signal"] = (data["overnight_squared_wma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_wma5 < {threshold:.5f}", perf))
    
    # WMA10
    data["signal"] = (data["overnight_squared_wma10"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_wma10 < {threshold:.5f}", perf))

# Тестирование Hull MA
print("\nТестирую Hull Moving Average...")
hma_thresholds = [0.00008, 0.0001, 0.00012, 0.00014, 0.00015]
for threshold in hma_thresholds:
    data["signal"] = (data["overnight_squared_hma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_hma5 < {threshold:.5f}", perf))

# Тестирование low volatility streaks
print("\nТестирую полосы низкой волатильности...")
streak_thresholds = [2, 3, 4, 5, 6, 7, 8, 10]
for threshold in streak_thresholds:
    data["signal"] = (data["low_vol_streak"].shift(1) >= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"low_vol_streak >= {threshold}", perf))

# Тестирование дней с последнего пика волатильности
print("\nТестирую дни с последнего пика волатильности...")
days_thresholds = [3, 5, 7, 10, 15, 20]
for threshold in days_thresholds:
    data["signal"] = (data["days_since_vol_spike"].shift(1) >= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"days_since_vol_spike >= {threshold}", perf))

# Тестирование percentile ranks
print("\nТестирую percentile ranks...")
pct_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
for threshold in pct_thresholds:
    # 10-day percentile
    data["signal"] = (data["overnight_squared_pct10"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_pct10 < {threshold:.1f}", perf))
    
    # 30-day percentile
    data["signal"] = (data["overnight_squared_pct30"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_squared_pct30 < {threshold:.1f}", perf))

# Инверсные стратегии (продаем вместо покупаем)
print("\nТестирую инверсные стратегии...")
inv_conditions = [
    ("NOT squared_ma5<0.00015", ~(data["overnight_squared"].rolling(window=5).mean().shift(1) < 0.00015)),
    ("overnight_squared > 0.0002", data["overnight_squared"].shift(1) > 0.0002),
    ("overnight_squared > 0.0003", data["overnight_squared"].shift(1) > 0.0003),
]
for condition_name, condition in inv_conditions:
    data["signal"] = condition.astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((condition_name, perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 11 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 12...")