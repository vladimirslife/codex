#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 15: Advanced volatility measures and transformations
"""

import pandas as pd
import numpy as np
from scipy import stats

# ------------------------- HELPERS -----------------------------------
def load_ticker(path: str) -> pd.DataFrame:
    """Load CSV, standardize columns, compute overnight returns."""
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.Timestamp("2006-01-01")]
    df = df.sort_values(by="Date")
    df = df.reset_index(drop=True)
    # Вычисляем ночную доходность: от close до open следующего дня
    df["prev_close"] = df["close"].shift(1)
    df["overnight_return"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    # Для стратегии нужна доходность от сегодняшнего close до завтрашнего open
    df["next_open"] = df["open"].shift(-1)
    df["next_overnight_return"] = (df["next_open"] - df["close"]) / df["close"]
    
    # Base volatility measures
    df["overnight_squared"] = df["overnight_return"] ** 2
    df["hl_range"] = (df["high"] - df["low"]) / df["open"]
    
    # Advanced volatility transformations
    # 1. Log-transformed H-L range
    df["log_hl_range"] = np.log1p(df["hl_range"])
    df["log_hl_range_ma5"] = df["log_hl_range"].rolling(window=5).mean()
    
    # 2. Parkinson volatility (using H-L)
    df["parkinson_vol"] = np.sqrt(np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2)))
    df["parkinson_vol_ma5"] = df["parkinson_vol"].rolling(window=5).mean()
    
    # 3. Garman-Klass volatility
    df["gk_vol"] = np.sqrt(
        0.5 * np.log(df["high"] / df["low"]) ** 2 - 
        (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
    )
    df["gk_vol_ma5"] = df["gk_vol"].rolling(window=5).mean()
    
    # 4. Combined overnight and intraday volatility
    df["combined_vol"] = np.sqrt(df["overnight_squared"] + df["hl_range"] ** 2)
    df["combined_vol_ma5"] = df["combined_vol"].rolling(window=5).mean()
    
    # 5. Volatility percentiles
    df["hl_range_pct20"] = df["hl_range"].rolling(window=20).quantile(0.2)
    df["hl_range_pct50"] = df["hl_range"].rolling(window=20).quantile(0.5)
    df["overnight_squared_pct20"] = df["overnight_squared"].rolling(window=20).quantile(0.2)
    
    # 6. Volatility acceleration (change in volatility)
    df["hl_range_change"] = df["hl_range"] - df["hl_range"].shift(1)
    df["vol_deceleration"] = (df["hl_range_change"] < 0).rolling(window=5).sum()
    
    # 7. Ratio-based measures
    df["overnight_to_intraday"] = np.abs(df["overnight_return"]) / (df["hl_range"] + 0.0001)
    df["overnight_to_intraday_ma5"] = df["overnight_to_intraday"].rolling(window=5).mean()
    
    # 8. Z-score of volatility
    df["hl_range_zscore"] = (df["hl_range"] - df["hl_range"].rolling(window=20).mean()) / df["hl_range"].rolling(window=20).std()
    
    # 9. Exponentially weighted volatility with optimal alpha from Wave 14
    alpha = 0.4
    df["hl_range_ewm"] = df["hl_range"].ewm(alpha=alpha, adjust=False).mean()
    
    # 10. Days since high volatility
    df["high_vol_flag"] = (df["hl_range"] > df["hl_range"].rolling(window=20).quantile(0.8)).astype(int)
    df["days_since_high_vol"] = 0
    counter = 0
    for i in range(len(df)):
        if df.loc[i, "high_vol_flag"] == 1:
            counter = 0
        else:
            counter += 1
        df.loc[i, "days_since_high_vol"] = counter
    
    # 11. Volatility regime (using percentiles)
    df["vol_regime_low"] = (df["hl_range"] < df["hl_range"].rolling(window=50).quantile(0.3)).astype(int)
    df["vol_regime_low_ma5"] = df["vol_regime_low"].rolling(window=5).mean()
    
    # 12. ATR-based measure
    df["atr"] = df[["high", "low", "close"]].apply(
        lambda x: max(x["high"] - x["low"], 
                     abs(x["high"] - df["close"].shift(1).loc[x.name]) if x.name > 0 else x["high"] - x["low"],
                     abs(x["low"] - df["close"].shift(1).loc[x.name]) if x.name > 0 else x["high"] - x["low"]),
        axis=1
    ) / df["open"]
    df["atr_ma14"] = df["atr"].rolling(window=14).mean()
    
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
print("\n=== Wave 15: Advanced volatility measures ===\n")

results = []

# 1. Log-transformed H-L range
print("Тестирую log-transformed H-L range...")
thresholds = [-4.5, -4.0, -3.5, -3.0, -2.5]
for threshold in thresholds:
    data["signal"] = (data["log_hl_range_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"log_hl_range_ma5 < {threshold:.1f}", perf))

# 2. Parkinson volatility
print("\nТестирую Parkinson volatility...")
thresholds = [0.005, 0.01, 0.015, 0.02, 0.025]
for threshold in thresholds:
    data["signal"] = (data["parkinson_vol_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"parkinson_vol_ma5 < {threshold:.3f}", perf))

# 3. Garman-Klass volatility
print("\nТестирую Garman-Klass volatility...")
thresholds = [0.005, 0.01, 0.015, 0.02, 0.025]
for threshold in thresholds:
    data["signal"] = (data["gk_vol_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"gk_vol_ma5 < {threshold:.3f}", perf))

# 4. Combined volatility
print("\nТестирую combined volatility...")
thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]
for threshold in thresholds:
    data["signal"] = (data["combined_vol_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"combined_vol_ma5 < {threshold:.3f}", perf))

# 5. Volatility percentiles
print("\nТестирую volatility percentiles...")
# H-L range below its 20th percentile
data["signal"] = (data["hl_range"].shift(1) < data["hl_range_pct20"].shift(1)).astype(int)
perf = calculate_strategy_performance(data, "signal")
results.append(("hl_range < hl_range_pct20", perf))

# Overnight squared below its 20th percentile
data["signal"] = (data["overnight_squared"].shift(1) < data["overnight_squared_pct20"].shift(1)).astype(int)
perf = calculate_strategy_performance(data, "signal")
results.append(("overnight_squared < overnight_squared_pct20", perf))

# 6. Volatility deceleration
print("\nТестирую volatility deceleration...")
thresholds = [3, 4, 5]
for threshold in thresholds:
    data["signal"] = (data["vol_deceleration"].shift(1) >= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_deceleration >= {threshold}", perf))

# 7. Overnight to intraday ratio
print("\nТестирую overnight to intraday ratio...")
thresholds = [0.5, 1.0, 1.5, 2.0]
for threshold in thresholds:
    data["signal"] = (data["overnight_to_intraday_ma5"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"overnight_to_intraday_ma5 < {threshold:.1f}", perf))

# 8. Z-score of volatility
print("\nТестирую Z-score of volatility...")
thresholds = [-1.5, -1.0, -0.5, 0]
for threshold in thresholds:
    data["signal"] = (data["hl_range_zscore"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_zscore < {threshold:.1f}", perf))

# 9. Days since high volatility
print("\nТестирую days since high volatility...")
thresholds = [3, 5, 7, 10]
for threshold in thresholds:
    data["signal"] = (data["days_since_high_vol"].shift(1) >= threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"days_since_high_vol >= {threshold}", perf))

# 10. Volatility regime
print("\nТестирую volatility regime...")
thresholds = [0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    data["signal"] = (data["vol_regime_low_ma5"].shift(1) > threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_regime_low_ma5 > {threshold:.1f}", perf))

# 11. ATR-based
print("\nТестирую ATR-based conditions...")
thresholds = [0.015, 0.02, 0.025, 0.03]
for threshold in thresholds:
    data["signal"] = (data["atr_ma14"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"atr_ma14 < {threshold:.3f}", perf))

# 12. Exponentially weighted H-L range
print("\nТестирую exponentially weighted H-L range...")
thresholds = [0.015, 0.02, 0.025, 0.03]
for threshold in thresholds:
    data["signal"] = (data["hl_range_ewm"].shift(1) < threshold).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_ewm < {threshold:.3f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 15 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 16...")
    print(f"\nЛучший результат: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")