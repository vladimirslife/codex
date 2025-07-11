#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 21: Оптимизация лучшего индикатора weighted_vol_micro из Wave 19
"""

import pandas as pd
import numpy as np
from scipy import stats, optimize, signal

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
    
    # Microstructure noise proxy (from Wave 19)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    df["noise_proxy"] = df["hl_range"] / (np.abs(df["oc_range"]) + 0.0001)
    df["noise_proxy_log"] = np.log1p(df["noise_proxy"])
    df["noise_weight"] = 1 / (df["noise_proxy_log"] + 1)
    
    # Advanced noise measures
    # 1. Amihud illiquidity proxy (simplified without volume)
    df["price_impact"] = np.abs(df["close_pct"]) / (df["hl_range"] + 0.0001)
    df["illiquidity_proxy"] = df["price_impact"].rolling(window=20).mean()
    
    # 2. Kyle's lambda approximation
    df["kyle_lambda"] = np.abs(df["close_pct"]) / np.sqrt(df["hl_range"])
    
    # 3. Realized efficiency ratio
    df["realized_path"] = df["hl_range"].rolling(window=5).sum()
    df["realized_displacement"] = np.abs(df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    df["realized_efficiency"] = df["realized_displacement"] / (df["realized_path"] + 0.0001)
    
    # 4. Microstructure volatility decomposition
    df["ms_permanent"] = df["close_pct"].rolling(window=20).std()
    df["ms_temporary"] = df["hl_range"] - np.abs(df["close_pct"])
    df["ms_ratio"] = df["ms_temporary"] / (df["ms_permanent"] + 0.0001)
    
    # 5. Advanced weighting schemes
    # Exponential decay of noise impact
    df["noise_exp_weight"] = np.exp(-df["noise_proxy_log"])
    
    # Sigmoid transformation for smooth weighting
    df["noise_sigmoid_weight"] = 1 / (1 + np.exp(5 * (df["noise_proxy_log"] - 2)))
    
    # Power law weighting
    df["noise_power_weight"] = 1 / (df["noise_proxy_log"] ** 1.5 + 1)
    
    # 6. Time-varying adjustments
    # Intraday volatility pattern
    df["time_factor"] = 1.0  # Placeholder since we don't have intraday data
    
    # Day of week effects
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    
    # 7. Volatility term structure
    df["vol_slope"] = (df["hl_range"].rolling(window=5).mean() - 
                      df["hl_range"].rolling(window=20).mean()) / df["hl_range"].rolling(window=20).mean()
    
    # 8. Jump detection
    df["jump_indicator"] = (np.abs(df["overnight_return"]) > 2 * df["overnight_return"].rolling(window=20).std()).astype(int)
    df["jump_filtered_hl"] = df["hl_range"] * (1 - df["jump_indicator"])
    
    # 9. Robust volatility measures
    # Median absolute deviation
    df["hl_mad"] = df["hl_range"].rolling(window=20).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Interquartile range
    df["hl_iqr"] = df["hl_range"].rolling(window=20).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    
    # 10. Volatility persistence score
    df["vol_acf1"] = df["hl_range"].rolling(window=30).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
    df["vol_acf2"] = df["hl_range"].rolling(window=30).apply(lambda x: x.autocorr(lag=2) if len(x) > 2 else 0)
    df["vol_persistence"] = df["vol_acf1"] + 0.5 * df["vol_acf2"]
    
    # Apply moving averages to key indicators
    for col in ["noise_weight", "noise_exp_weight", "noise_sigmoid_weight", "noise_power_weight",
                "illiquidity_proxy", "kyle_lambda", "realized_efficiency", "ms_ratio",
                "vol_slope", "jump_filtered_hl", "hl_mad", "hl_iqr", "vol_persistence"]:
        if col in df.columns:
            df[f"{col}_ma5"] = df[col].rolling(window=5).mean()
            df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
            df[f"{col}_ewm"] = df[col].ewm(alpha=0.3, adjust=False).mean()
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Merge on Date
    data = qqq.copy()
    data = data.merge(spy[["Date", "hl_range", "noise_weight", "noise_exp_weight", "noise_sigmoid_weight", 
                           "noise_power_weight", "illiquidity_proxy", "kyle_lambda"]], 
                      on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[["Date", "hl_range", "noise_weight", "noise_exp_weight", "noise_sigmoid_weight",
                           "noise_power_weight", "illiquidity_proxy", "kyle_lambda"]], 
                      on="Date", suffixes=("", "_xlk"))
    
    # Recreate best indicator from Wave 19 with variations
    # 1. Original weighted_vol_micro (for reference)
    data["weighted_vol_micro_original"] = (
        data["hl_range"] * (1 / (data["noise_proxy_log"] + 1)) +
        data["hl_range_spy"] * 0.5 +
        data["hl_range_xlk"] * 0.3
    ) / 1.8
    
    # 2. With exponential noise weights
    data["weighted_vol_micro_exp"] = (
        data["hl_range"] * data["noise_exp_weight"] +
        data["hl_range_spy"] * data["noise_exp_weight_spy"] * 0.5 +
        data["hl_range_xlk"] * data["noise_exp_weight_xlk"] * 0.3
    ) / (data["noise_exp_weight"] + data["noise_exp_weight_spy"] * 0.5 + data["noise_exp_weight_xlk"] * 0.3)
    
    # 3. With sigmoid noise weights
    data["weighted_vol_micro_sigmoid"] = (
        data["hl_range"] * data["noise_sigmoid_weight"] +
        data["hl_range_spy"] * data["noise_sigmoid_weight_spy"] * 0.5 +
        data["hl_range_xlk"] * data["noise_sigmoid_weight_xlk"] * 0.3
    ) / (data["noise_sigmoid_weight"] + data["noise_sigmoid_weight_spy"] * 0.5 + data["noise_sigmoid_weight_xlk"] * 0.3)
    
    # 4. With power law weights
    data["weighted_vol_micro_power"] = (
        data["hl_range"] * data["noise_power_weight"] +
        data["hl_range_spy"] * data["noise_power_weight_spy"] * 0.5 +
        data["hl_range_xlk"] * data["noise_power_weight_xlk"] * 0.3
    ) / (data["noise_power_weight"] + data["noise_power_weight_spy"] * 0.5 + data["noise_power_weight_xlk"] * 0.3)
    
    # 5. Optimized ticker weights (based on historical performance)
    # Testing different weight combinations
    weights_to_test = [
        (0.6, 0.3, 0.1),  # More QQQ weight
        (0.5, 0.35, 0.15),  # Balanced
        (0.4, 0.4, 0.2),  # Equal QQQ/SPY
        (0.45, 0.35, 0.2),  # Slight QQQ bias
        (0.55, 0.3, 0.15),  # Moderate QQQ bias
    ]
    
    for i, (w1, w2, w3) in enumerate(weights_to_test):
        data[f"weighted_vol_micro_w{i+1}"] = (
            data["hl_range"] * data["noise_weight"] * w1 +
            data["hl_range_spy"] * data["noise_weight_spy"] * w2 +
            data["hl_range_xlk"] * data["noise_weight_xlk"] * w3
        ) / (data["noise_weight"] * w1 + data["noise_weight_spy"] * w2 + data["noise_weight_xlk"] * w3)
    
    # 6. With illiquidity adjustment
    data["weighted_vol_micro_illiq"] = (
        data["hl_range"] * data["noise_weight"] / (data["illiquidity_proxy"] + 1) +
        data["hl_range_spy"] * data["noise_weight_spy"] / (data["illiquidity_proxy_spy"] + 1) * 0.5 +
        data["hl_range_xlk"] * data["noise_weight_xlk"] / (data["illiquidity_proxy_xlk"] + 1) * 0.3
    ) / (data["noise_weight"] / (data["illiquidity_proxy"] + 1) + 
         data["noise_weight_spy"] / (data["illiquidity_proxy_spy"] + 1) * 0.5 + 
         data["noise_weight_xlk"] / (data["illiquidity_proxy_xlk"] + 1) * 0.3)
    
    # 7. Dynamic weight based on correlation
    corr_window = 30
    data["corr_spy_qqq"] = data["hl_range"].rolling(window=corr_window).corr(data["hl_range_spy"])
    data["corr_xlk_qqq"] = data["hl_range"].rolling(window=corr_window).corr(data["hl_range_xlk"])
    
    # Higher weight to less correlated assets
    data["dyn_weight_spy"] = 0.5 * (1 - data["corr_spy_qqq"].abs())
    data["dyn_weight_xlk"] = 0.3 * (1 - data["corr_xlk_qqq"].abs())
    data["dyn_weight_qqq"] = 1 - data["dyn_weight_spy"] - data["dyn_weight_xlk"]
    
    data["weighted_vol_micro_dynamic"] = (
        data["hl_range"] * data["noise_weight"] * data["dyn_weight_qqq"] +
        data["hl_range_spy"] * data["noise_weight_spy"] * data["dyn_weight_spy"] +
        data["hl_range_xlk"] * data["noise_weight_xlk"] * data["dyn_weight_xlk"]
    ) / (data["noise_weight"] * data["dyn_weight_qqq"] + 
         data["noise_weight_spy"] * data["dyn_weight_spy"] + 
         data["noise_weight_xlk"] * data["dyn_weight_xlk"])
    
    # 8. Threshold optimization - test finer granularity around 0.018
    # Will test thresholds from 0.016 to 0.020 with 0.0002 steps
    
    # 9. Non-linear transformations of the best indicator
    data["weighted_vol_micro_original_log"] = np.log1p(data["weighted_vol_micro_original"] * 100)
    data["weighted_vol_micro_original_sqrt"] = np.sqrt(data["weighted_vol_micro_original"])
    data["weighted_vol_micro_original_cbrt"] = np.cbrt(data["weighted_vol_micro_original"])
    data["weighted_vol_micro_original_squared"] = data["weighted_vol_micro_original"] ** 2
    
    # 10. Polynomial transformation
    # Using polynomial features instead of isotonic regression
    data["weighted_vol_micro_poly2"] = data["weighted_vol_micro_original"] ** 2
    data["weighted_vol_micro_poly3"] = data["weighted_vol_micro_original"] ** 3
    data["weighted_vol_micro_inv"] = 1 / (data["weighted_vol_micro_original"] + 0.001)
    
    # Apply moving averages
    indicators = ["weighted_vol_micro_original", "weighted_vol_micro_exp", "weighted_vol_micro_sigmoid",
                 "weighted_vol_micro_power", "weighted_vol_micro_illiq", "weighted_vol_micro_dynamic",
                 "weighted_vol_micro_original_log", "weighted_vol_micro_original_sqrt", 
                 "weighted_vol_micro_original_cbrt", "weighted_vol_micro_original_squared",
                 "weighted_vol_micro_poly2", "weighted_vol_micro_poly3", "weighted_vol_micro_inv"]
    
    # Add weight variations
    indicators.extend([f"weighted_vol_micro_w{i+1}" for i in range(5)])
    
    for col in indicators:
        if col in data.columns:
            data[f"{col}_ma5"] = data[col].rolling(window=5).mean()
            data[f"{col}_ma3"] = data[col].rolling(window=3).mean()
            data[f"{col}_ma4"] = data[col].rolling(window=4).mean()
            data[f"{col}_ewm30"] = data[col].ewm(alpha=0.3, adjust=False).mean()
            data[f"{col}_ewm25"] = data[col].ewm(alpha=0.25, adjust=False).mean()
    
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
print("\n=== Wave 21: Оптимизация weighted_vol_micro ===\n")

results = []

# 1. Fine-tuning threshold for original indicator
print("Fine-tuning оригинального индикатора...")
thresholds = np.arange(0.016, 0.020, 0.0002)
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_original_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_original_ma5 < {threshold:.4f}", perf))

# 2. Test different MA periods
print("\nТестирую разные MA периоды...")
for ma_period in [3, 4]:
    thresholds = [0.017, 0.018, 0.019]
    for threshold in thresholds:
        col = f"weighted_vol_micro_original_ma{ma_period}"
        data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
        perf = calculate_strategy_performance(data, "signal")
        results.append((f"{col} < {threshold:.3f}", perf))

# 3. Test EWM variations
print("\nТестирую EWM вариации...")
for ewm_type in ["ewm30", "ewm25"]:
    thresholds = [0.017, 0.018, 0.019]
    for threshold in thresholds:
        col = f"weighted_vol_micro_original_{ewm_type}"
        data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
        perf = calculate_strategy_performance(data, "signal")
        results.append((f"{col} < {threshold:.3f}", perf))

# 4. Test noise weight variations
print("\nТестирую вариации с разными noise weights...")
for weight_type in ["exp", "sigmoid", "power"]:
    thresholds = [0.017, 0.018, 0.019]
    for threshold in thresholds:
        col = f"weighted_vol_micro_{weight_type}_ma5"
        data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
        perf = calculate_strategy_performance(data, "signal")
        results.append((f"{col} < {threshold:.3f}", perf))

# 5. Test different ticker weights
print("\nТестирую разные веса тикеров...")
for i in range(1, 6):
    thresholds = [0.017, 0.018, 0.019]
    for threshold in thresholds:
        col = f"weighted_vol_micro_w{i}_ma5"
        data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
        perf = calculate_strategy_performance(data, "signal")
        results.append((f"{col} < {threshold:.3f}", perf))

# 6. Test illiquidity-adjusted version
print("\nТестирую illiquidity-adjusted версию...")
thresholds = [0.017, 0.018, 0.019]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_illiq_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_illiq_ma5 < {threshold:.3f}", perf))

# 7. Test dynamic correlation weights
print("\nТестирую динамические веса...")
thresholds = [0.017, 0.018, 0.019]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_dynamic_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_dynamic_ma5 < {threshold:.3f}", perf))

# 8. Test non-linear transformations
print("\nТестирую нелинейные трансформации...")
# Log transform
thresholds = [2.85, 2.90, 2.95]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_original_log_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_original_log_ma5 < {threshold:.2f}", perf))

# Square root
thresholds = [0.130, 0.135, 0.140]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_original_sqrt_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_original_sqrt_ma5 < {threshold:.3f}", perf))

# Cube root
thresholds = [0.260, 0.265, 0.270]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_original_cbrt_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_original_cbrt_ma5 < {threshold:.3f}", perf))

# Squared
thresholds = [0.00030, 0.00032, 0.00034]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_original_squared_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_original_squared_ma5 < {threshold:.5f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 21 ===")
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
    print("\n❌ Цель не достигнута. Лучший результат всё ещё из Wave 19...")
    print(f"\nЛучший результат Wave 21: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    
    # Сравнение с Wave 19
    wave19_best = 1.0023
    if sorted_results[0][1]['sharpe_ratio'] > wave19_best:
        print(f"\n✨ НОВЫЙ РЕКОРД! Улучшили Wave 19 ({wave19_best:.4f})!")
        print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")
    else:
        print(f"\nНе превзошли Wave 19 (1.0023). Прогресс: {wave19_best/1.4*100:.1f}% от цели")