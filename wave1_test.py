#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAVE 1: Тестирование простых пороговых условий на основе предыдущей ночной доходности
"""

import pandas as pd
import numpy as np
import sys
import os

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
    df["Next_Open"] = df["open"].shift(-1)
    df["next_overnight_return"] = df["Next_Open"] / df["close"] - 1
    df["prev_overnight_return"] = df["next_overnight_return"].shift(1)
    return df

def test_strategy(df, condition_func, condition_desc):
    """Тестирует стратегию с заданным условием"""
    df = df.copy()
    
    # Генерируем сигнал на основе условия
    df["signal"] = condition_func(df).astype(int)
    
    # Расчет доходности стратегии
    df["strategy_daily_return"] = df["signal"] * df["next_overnight_return"].shift(1)
    df["strategy_daily_return"].fillna(0, inplace=True)
    
    # Метрики производительности
    annual_rf = 0.02
    daily_rf = annual_rf / 252
    
    excess_returns = df["strategy_daily_return"] - daily_rf
    mean_excess_annual = excess_returns.mean() * 252
    std_excess_annual = excess_returns.std() * np.sqrt(252)
    sharpe_ratio = mean_excess_annual / std_excess_annual if std_excess_annual != 0 else 0
    
    df["strategy_equity"] = (1 + df["strategy_daily_return"]).cumprod()
    total_return = df["strategy_equity"].iloc[-1] - 1
    years_span = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years_span) - 1 if years_span > 0 else 0
    
    num_trades = int(df["signal"].sum())
    
    return {
        'condition': condition_desc,
        'sharpe_ratio': sharpe_ratio,
        'cagr': cagr,
        'num_trades': num_trades,
        'total_return': total_return
    }

# Загружаем данные
qqq_data = load_ticker("4 - QQQ.csv")

print("=== WAVE 1: Простые пороговые условия ===\n")

# Список результатов
results = []

# 1. Тестируем положительную ночную доходность (разные пороги)
positive_thresholds = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]
for threshold in positive_thresholds:
    condition_func = lambda df, t=threshold: df["prev_overnight_return"] > t
    condition_desc = f"prev_overnight_return > {threshold}"
    result = test_strategy(qqq_data, condition_func, condition_desc)
    results.append(result)
    print(f"Условие: {condition_desc}")
    print(f"Sharpe: {result['sharpe_ratio']:.4f}, CAGR: {result['cagr']*100:.2f}%, Сделок: {result['num_trades']}")
    print("-" * 60)

# 2. Тестируем отрицательную ночную доходность (разные пороги)
negative_thresholds = [-0.001, -0.002, -0.003, -0.005, -0.007, -0.01, -0.015, -0.02, -0.025]
for threshold in negative_thresholds:
    condition_func = lambda df, t=threshold: df["prev_overnight_return"] < t
    condition_desc = f"prev_overnight_return < {threshold}"
    result = test_strategy(qqq_data, condition_func, condition_desc)
    results.append(result)
    print(f"Условие: {condition_desc}")
    print(f"Sharpe: {result['sharpe_ratio']:.4f}, CAGR: {result['cagr']*100:.2f}%, Сделок: {result['num_trades']}")
    print("-" * 60)

# 3. Тестируем значительную ночную доходность (по модулю)
abs_thresholds = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025]
for threshold in abs_thresholds:
    condition_func = lambda df, t=threshold: np.abs(df["prev_overnight_return"]) > t
    condition_desc = f"abs(prev_overnight_return) > {threshold}"
    result = test_strategy(qqq_data, condition_func, condition_desc)
    results.append(result)
    print(f"Условие: {condition_desc}")
    print(f"Sharpe: {result['sharpe_ratio']:.4f}, CAGR: {result['cagr']*100:.2f}%, Сделок: {result['num_trades']}")
    print("-" * 60)

# Находим ТОП-3 результата
print("\n=== ТОП-3 ЛУЧШИХ РЕЗУЛЬТАТА ПО SHARPE RATIO ===")
sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)

for i, result in enumerate(sorted_results[:3], 1):
    print(f"\n{i}. {result['condition']}")
    print(f"   Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"   CAGR: {result['cagr']*100:.2f}%")
    print(f"   Количество сделок: {result['num_trades']}")
    print(f"   Общая доходность: {result['total_return']*100:.2f}%")

# Проверяем, достигли ли мы цели
target_achieved = False
for result in sorted_results:
    if result['sharpe_ratio'] >= 1.4 and result['num_trades'] > 2500:
        target_achieved = True
        print(f"\n🎯 ЦЕЛЬ ДОСТИГНУТА! Условие: {result['condition']}")
        print(f"   Sharpe: {result['sharpe_ratio']:.4f}, Сделок: {result['num_trades']}")
        break

if not target_achieved:
    print(f"\n❌ Цель не достигнута в Wave 1. Лучший Sharpe: {sorted_results[0]['sharpe_ratio']:.4f}")
    print("Переходим к Wave 2...")

print("\nWAVE 1 завершена.")