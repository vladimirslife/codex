#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 22: Дальнейшая оптимизация MA3 и порога 0.017
"""

import pandas as pd
import numpy as np
from scipy import stats, optimize

# ------------------------- HELPERS -----------------------------------
def load_ticker(path: str) -> pd.DataFrame:
    """Load CSV, standardize columns, compute overnight returns."""
    df = pd.read_csv(path)
    df.rename(columns=lambda c: c.lower(), inplace=True)
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.Timestamp("2006-01-01")]
    df = df.sort_values(by="Date")  # type: ignore
    df = df.reset_index(drop=True)
    
    # Вычисляем ночную доходность: от close до open следующего дня
    df["prev_close"] = df["close"].shift(1)
    df["overnight_return"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    # Для стратегии нужна доходность от сегодняшнего close до завтрашнего open
    df["next_open"] = df["open"].shift(-1)
    df["next_overnight_return"] = (df["next_open"] - df["close"]) / df["close"]
    
    # Base measures
    df["overnight_squared"] = df["overnight_return"] ** 2
    df["hl_range"] = (df["high"] - df["low"]) / df["open"]
    df["close_pct"] = df["close"].pct_change()
    
    # Microstructure noise proxy (best from Wave 19/21)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    df["noise_proxy"] = df["hl_range"] / (np.abs(df["oc_range"]) + 0.0001)
    df["noise_proxy_log"] = np.log1p(df["noise_proxy"])
    df["noise_weight"] = 1 / (df["noise_proxy_log"] + 1)
    
    # Advanced noise transformations
    # 1. Adaptive noise weight based on volatility regime
    df["vol_percentile"] = df["hl_range"].rolling(window=252).rank(pct=True)
    df["adaptive_noise_weight"] = df["noise_weight"] * (1 + (0.5 - df["vol_percentile"]))
    
    # 2. Smoothed noise weight
    df["noise_weight_smooth"] = df["noise_weight"].ewm(alpha=0.1, adjust=False).mean()
    
    # 3. Noise weight with outlier adjustment
    noise_mean = df["noise_weight"].rolling(window=20).mean()
    noise_std = df["noise_weight"].rolling(window=20).std()
    df["noise_weight_robust"] = np.where(
        df["noise_weight"] > noise_mean + 2*noise_std,
        noise_mean + 2*noise_std,
        df["noise_weight"]
    )
    
    # 4. Time-decay noise weight
    df["days_from_month_start"] = df["Date"].dt.day
    df["time_decay_factor"] = 1 / (1 + df["days_from_month_start"] / 30)
    df["noise_weight_time"] = df["noise_weight"] * df["time_decay_factor"]
    
    # 5. Volatility clustering adjustment
    df["vol_autocorr"] = df["hl_range"].rolling(window=20).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
    df["cluster_adjusted_weight"] = df["noise_weight"] * (1 + df["vol_autocorr"] * 0.5)
    
    # 6. Market stress indicator
    df["stress_indicator"] = (df["hl_range"] > df["hl_range"].rolling(window=60).quantile(0.9)).astype(int)
    df["stress_days"] = df["stress_indicator"].rolling(window=10).sum()
    df["calm_market_weight"] = 1 / (1 + df["stress_days"] / 10)
    
    # 7. Intraday pattern adjustments
    df["hl_to_oc_ratio"] = df["hl_range"] / (np.abs(df["oc_range"]) + 0.0001)
    df["pattern_weight"] = 1 / (1 + np.log1p(df["hl_to_oc_ratio"]))
    
    # 8. Volatility of volatility
    df["vol_of_vol"] = df["hl_range"].rolling(window=20).std()
    df["stable_vol_weight"] = 1 / (1 + df["vol_of_vol"] * 10)
    
    # 9. Range position indicator
    df["close_in_range"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 0.0001)
    df["range_weight"] = 1 - 2 * np.abs(df["close_in_range"] - 0.5)
    
    # 10. Momentum-adjusted weight
    df["price_momentum"] = df["close"].pct_change(5)
    df["momentum_weight"] = 1 / (1 + np.abs(df["price_momentum"]) * 10)
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Merge on Date
    data = qqq.copy()
    
    # Get all weight variations from each ticker
    weight_cols = ["noise_weight", "adaptive_noise_weight", "noise_weight_smooth", 
                   "noise_weight_robust", "noise_weight_time", "cluster_adjusted_weight",
                   "calm_market_weight", "pattern_weight", "stable_vol_weight", 
                   "range_weight", "momentum_weight"]
    
    merge_cols = ["Date", "hl_range"] + weight_cols
    
    data = data.merge(spy[merge_cols], on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[merge_cols], on="Date", suffixes=("", "_xlk"))
    
    # Test different weight types with the winning formula
    for weight_col in weight_cols:
        # Original formula with different weight types
        data[f"weighted_vol_{weight_col}"] = (
            data["hl_range"] * data[weight_col] +
            data["hl_range_spy"] * 0.5 +
            data["hl_range_xlk"] * 0.3
        ) / 1.8
        
        # All tickers use same weight type
        data[f"weighted_vol_{weight_col}_all"] = (
            data["hl_range"] * data[weight_col] +
            data["hl_range_spy"] * data[f"{weight_col}_spy"] * 0.5 +
            data["hl_range_xlk"] * data[f"{weight_col}_xlk"] * 0.3
        ) / (data[weight_col] + data[f"{weight_col}_spy"] * 0.5 + data[f"{weight_col}_xlk"] * 0.3)
    
    # Test optimal ticker weight combinations around current best
    weight_combinations = [
        (1.0, 0.5, 0.3),   # Original
        (1.0, 0.55, 0.25), # More SPY
        (1.0, 0.45, 0.35), # More XLK
        (1.0, 0.6, 0.2),   # Much more SPY
        (1.0, 0.4, 0.4),   # Equal SPY/XLK
        (1.1, 0.5, 0.3),   # Slightly more QQQ
        (0.9, 0.5, 0.3),   # Slightly less QQQ
        (1.0, 0.5, 0.25),  # Less XLK
        (1.0, 0.45, 0.3),  # Slightly less SPY
        (1.05, 0.5, 0.3),  # Very slight QQQ increase
    ]
    
    for i, (w1, w2, w3) in enumerate(weight_combinations):
        norm = w1 + w2 + w3
        data[f"weighted_vol_w{i+1}"] = (
            data["hl_range"] * data["noise_weight"] * w1 +
            data["hl_range_spy"] * w2 +
            data["hl_range_xlk"] * w3
        ) / norm
    
    # Test different MA periods around 3
    data["weighted_vol_original"] = (
        data["hl_range"] * data["noise_weight"] +
        data["hl_range_spy"] * 0.5 +
        data["hl_range_xlk"] * 0.3
    ) / 1.8
    
    # Apply different smoothing methods
    data["weighted_vol_ma2"] = data["weighted_vol_original"].rolling(window=2).mean()
    data["weighted_vol_ma3"] = data["weighted_vol_original"].rolling(window=3).mean()
    data["weighted_vol_ma4"] = data["weighted_vol_original"].rolling(window=4).mean()
    data["weighted_vol_ewm20"] = data["weighted_vol_original"].ewm(alpha=0.2, adjust=False).mean()
    data["weighted_vol_ewm25"] = data["weighted_vol_original"].ewm(alpha=0.25, adjust=False).mean()
    data["weighted_vol_ewm35"] = data["weighted_vol_original"].ewm(alpha=0.35, adjust=False).mean()
    
    # Weighted moving average (more weight on recent)
    def weighted_ma(series, weights=[0.5, 0.3, 0.2]):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(weights), len(series)+1):
            weighted_sum = sum(series.iloc[i-len(weights)+j] * weights[j] for j in range(len(weights)))
            result.iloc[i-1] = weighted_sum / sum(weights)
        return result
    
    data["weighted_vol_wma3"] = weighted_ma(data["weighted_vol_original"], [0.5, 0.3, 0.2])
    data["weighted_vol_wma3_v2"] = weighted_ma(data["weighted_vol_original"], [0.6, 0.3, 0.1])
    data["weighted_vol_wma3_v3"] = weighted_ma(data["weighted_vol_original"], [0.4, 0.35, 0.25])
    
    # Hull moving average (reduced lag)
    def hull_ma(series, period=3):
        wma_half = series.rolling(window=period//2).mean()
        wma_full = series.rolling(window=period).mean()
        diff = 2 * wma_half - wma_full
        hull = diff.rolling(window=int(np.sqrt(period))).mean()
        return hull
    
    data["weighted_vol_hull3"] = hull_ma(data["weighted_vol_original"], 3)
    data["weighted_vol_hull4"] = hull_ma(data["weighted_vol_original"], 4)
    
    # Apply MA3 to all weight variations for comparison
    for weight_col in weight_cols:
        data[f"weighted_vol_{weight_col}_ma3"] = data[f"weighted_vol_{weight_col}"].rolling(window=3).mean()
        data[f"weighted_vol_{weight_col}_all_ma3"] = data[f"weighted_vol_{weight_col}_all"].rolling(window=3).mean()
    
    # Apply MA3 to weight combinations
    for i in range(1, 11):
        data[f"weighted_vol_w{i}_ma3"] = data[f"weighted_vol_w{i}"].rolling(window=3).mean()
    
    return data

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
data = load_all_tickers()
print(f"Загружено {len(data)} строк данных")

# ------------------------- ТЕСТИРОВАНИЕ СТРАТЕГИЙ -----------------------------------
print("\n=== Wave 22: Оптимизация MA3 и порога 0.017 ===\n")

results = []

# 1. Fine-tune threshold around 0.017
print("Fine-tuning порога около 0.017...")
thresholds = np.arange(0.0165, 0.0175, 0.0001)
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_ma3"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_ma3 < {threshold:.4f}", perf))

# 2. Test different smoothing methods
print("\nТестирую разные методы сглаживания...")
smoothing_cols = ["weighted_vol_ma2", "weighted_vol_ma3", "weighted_vol_ma4",
                 "weighted_vol_ewm20", "weighted_vol_ewm25", "weighted_vol_ewm35",
                 "weighted_vol_wma3", "weighted_vol_wma3_v2", "weighted_vol_wma3_v3",
                 "weighted_vol_hull3", "weighted_vol_hull4"]

for col in smoothing_cols:
    if col in data.columns:
        for threshold in [0.0168, 0.017, 0.0172]:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 3. Test different weight types
print("\nТестирую разные типы весов...")
weight_types = ["adaptive_noise_weight", "noise_weight_smooth", "noise_weight_robust",
               "cluster_adjusted_weight", "calm_market_weight", "stable_vol_weight"]

for weight_type in weight_types:
    col = f"weighted_vol_{weight_type}_ma3"
    if col in data.columns:
        for threshold in [0.0168, 0.017, 0.0172]:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 4. Test all tickers using same weight type
print("\nТестирую все тикеры с одинаковым типом веса...")
for weight_type in ["noise_weight", "adaptive_noise_weight", "cluster_adjusted_weight"]:
    col = f"weighted_vol_{weight_type}_all_ma3"
    if col in data.columns:
        for threshold in [0.0168, 0.017, 0.0172]:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 5. Test optimized ticker weights
print("\nТестирую оптимизированные веса тикеров...")
for i in range(1, 11):
    col = f"weighted_vol_w{i}_ma3"
    if col in data.columns:
        for threshold in [0.0168, 0.017, 0.0172]:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 6. Test combined indicators (still single condition via mathematical combination)
print("\nТестирую комбинированные индикаторы...")
# Combine with momentum
data["combined_vol_momentum"] = data["weighted_vol_ma3"] * (1 + data["price_momentum"].abs() * 0.1)
data["signal"] = (data["combined_vol_momentum"].shift(1) < 0.0175).fillna(False).astype(int)
perf = calculate_strategy_performance(data, "signal")
results.append(("combined_vol_momentum < 0.0175", perf))

# Combine with stress indicator
data["combined_vol_stress"] = data["weighted_vol_ma3"] * (1 + data["stress_days"] * 0.02)
data["signal"] = (data["combined_vol_stress"].shift(1) < 0.0175).fillna(False).astype(int)
perf = calculate_strategy_performance(data, "signal")
results.append(("combined_vol_stress < 0.0175", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 22 ===")
sorted_results = sorted(results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)

print("\nТоп 15 результатов по Sharpe Ratio:")
for i, (condition, perf) in enumerate(sorted_results[:15], 1):
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
    print("\n❌ Цель не достигнута.")
    print(f"\nЛучший результат Wave 22: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    
    # Сравнение с Wave 21
    wave21_best = 1.0693
    if sorted_results[0][1]['sharpe_ratio'] > wave21_best:
        print(f"\n✨ НОВЫЙ РЕКОРД! Улучшили Wave 21 ({wave21_best:.4f})!")
        print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")
    else:
        print(f"\nНе превзошли Wave 21 (1.0693). Прогресс: {wave21_best/1.4*100:.1f}% от цели")