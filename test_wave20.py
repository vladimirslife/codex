#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 20: Оптимизация весов и продвинутые комбинации индикаторов
"""

import pandas as pd
import numpy as np
from scipy import stats, optimize, special

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
    
    # Best indicators from previous waves
    # 1. High-Low range transformations
    df["hl_range_sqrt"] = np.sqrt(df["hl_range"])
    df["hl_range_log"] = np.log1p(df["hl_range"])
    df["hl_range_cbrt"] = np.cbrt(df["hl_range"])
    
    # 2. Exponentially weighted measures
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        df[f"hl_range_ewm_{int(alpha*100)}"] = df["hl_range"].ewm(alpha=alpha, adjust=False).mean()
    
    # 3. Microstructure noise proxy
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    df["noise_proxy"] = df["hl_range"] / (np.abs(df["oc_range"]) + 0.0001)
    df["noise_proxy_log"] = np.log1p(df["noise_proxy"])
    df["noise_weight"] = 1 / (df["noise_proxy_log"] + 1)
    
    # 4. Volatility percentiles
    for window in [20, 50, 100]:
        for pct in [0.2, 0.3, 0.4]:
            df[f"hl_pct_{window}_{int(pct*100)}"] = df["hl_range"] / df["hl_range"].rolling(window=window).quantile(pct)
    
    # 5. Volatility clustering
    df["vol_autocorr"] = df["hl_range"].rolling(window=20).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
    df["vol_cluster_strength"] = df["vol_autocorr"] * (1 - df["hl_range"] / df["hl_range"].rolling(window=50).mean())
    
    # 6. Regime indicators
    df["low_vol_regime"] = (df["hl_range"] < df["hl_range"].rolling(window=100).quantile(0.3)).astype(int)
    df["regime_persistence"] = df["low_vol_regime"].rolling(window=10).mean()
    
    # 7. Advanced transformations
    # Sigmoid transformation with optimized parameters
    df["hl_sigmoid"] = 1 / (1 + np.exp(-200 * (df["hl_range"] - 0.02)))
    
    # Box-Cox transformation (approximation)
    lambda_param = 0.2  # Optimized from testing
    df["hl_boxcox"] = (df["hl_range"] ** lambda_param - 1) / lambda_param if lambda_param != 0 else np.log(df["hl_range"])
    
    # 8. Volatility momentum
    df["vol_momentum"] = df["hl_range"].rolling(window=5).mean() - df["hl_range"].rolling(window=20).mean()
    df["vol_momentum_scaled"] = np.tanh(df["vol_momentum"] * 100)
    
    # 9. Time-based adjustments
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_month_end"] = (df["Date"].dt.day > 25).astype(int)
    df["is_month_start"] = (df["Date"].dt.day < 6).astype(int)
    
    # 10. Volatility surface approximation
    df["vol_term_structure"] = df["hl_range"].rolling(window=5).mean() / df["hl_range"].rolling(window=20).mean()
    df["vol_curvature"] = (df["hl_range"].rolling(window=5).mean() + df["hl_range"].rolling(window=20).mean()) / (2 * df["hl_range"].rolling(window=10).mean() + 0.0001)
    
    # Apply moving averages to key indicators
    for col in ["hl_range", "hl_range_sqrt", "hl_range_log", "hl_range_cbrt", 
                "noise_weight", "vol_cluster_strength", "regime_persistence",
                "hl_sigmoid", "hl_boxcox", "vol_momentum_scaled", "vol_term_structure"]:
        if col in df.columns:
            df[f"{col}_ma5"] = df[col].rolling(window=5).mean()
            df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Merge on Date
    data = qqq.copy()
    data = data.merge(spy[["Date", "hl_range", "noise_weight", "vol_cluster_strength"]], 
                      on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[["Date", "hl_range", "noise_weight", "vol_cluster_strength"]], 
                      on="Date", suffixes=("", "_xlk"))
    
    # Optimal weighted combinations based on Wave 19 success
    # 1. Microstructure-weighted volatility (refined)
    data["weighted_vol_micro_v2"] = (
        data["hl_range"] * data["noise_weight"] * 0.5 +
        data["hl_range_spy"] * data["noise_weight_spy"] * 0.3 +
        data["hl_range_xlk"] * data["noise_weight_xlk"] * 0.2
    ) / (data["noise_weight"] * 0.5 + data["noise_weight_spy"] * 0.3 + data["noise_weight_xlk"] * 0.2)
    
    # 2. Optimal weights based on inverse variance
    data["var_qqq"] = data["hl_range"].rolling(window=20).var()
    data["var_spy"] = data["hl_range_spy"].rolling(window=20).var()
    data["var_xlk"] = data["hl_range_xlk"].rolling(window=20).var()
    
    data["w_qqq"] = (1/data["var_qqq"]) / (1/data["var_qqq"] + 1/data["var_spy"] + 1/data["var_xlk"])
    data["w_spy"] = (1/data["var_spy"]) / (1/data["var_qqq"] + 1/data["var_spy"] + 1/data["var_xlk"])
    data["w_xlk"] = (1/data["var_xlk"]) / (1/data["var_qqq"] + 1/data["var_spy"] + 1/data["var_xlk"])
    
    data["optimal_vol_invvar"] = (
        data["hl_range"] * data["w_qqq"] +
        data["hl_range_spy"] * data["w_spy"] +
        data["hl_range_xlk"] * data["w_xlk"]
    )
    
    # 3. Cluster-strength weighted volatility
    data["cluster_weighted_vol"] = (
        data["hl_range"] * (1 + data["vol_cluster_strength"]) +
        data["hl_range_spy"] * (1 + data["vol_cluster_strength_spy"]) +
        data["hl_range_xlk"] * (1 + data["vol_cluster_strength_xlk"])
    ) / (3 + data["vol_cluster_strength"] + data["vol_cluster_strength_spy"] + data["vol_cluster_strength_xlk"])
    
    # 4. Harmonic-geometric mean hybrid
    data["harm_geom_hybrid"] = 2 / (
        1/np.sqrt(data["hl_range"] * data["hl_range_spy"] * data["hl_range_xlk"]) +
        1/(3 / (1/data["hl_range"] + 1/data["hl_range_spy"] + 1/data["hl_range_xlk"]))
    )
    
    # 5. Rank-based combination
    data["rank_qqq"] = data["hl_range"].rank(pct=True)
    data["rank_spy"] = data["hl_range_spy"].rank(pct=True)
    data["rank_xlk"] = data["hl_range_xlk"].rank(pct=True)
    data["rank_avg"] = (data["rank_qqq"] + data["rank_spy"] + data["rank_xlk"]) / 3
    
    # 6. Entropy-weighted combination
    def rolling_entropy(series, window=20):
        def entropy(x):
            if len(x) < 5:
                return 1
            hist, _ = np.histogram(x, bins=5)
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            return -np.sum(probs * np.log(probs))
        return series.rolling(window=window).apply(entropy)
    
    data["entropy_qqq"] = rolling_entropy(data["hl_range"])
    data["entropy_spy"] = rolling_entropy(data["hl_range_spy"])
    data["entropy_xlk"] = rolling_entropy(data["hl_range_xlk"])
    
    data["entropy_weighted_vol"] = (
        data["hl_range"] * (1/data["entropy_qqq"]) +
        data["hl_range_spy"] * (1/data["entropy_spy"]) +
        data["hl_range_xlk"] * (1/data["entropy_xlk"])
    ) / (1/data["entropy_qqq"] + 1/data["entropy_spy"] + 1/data["entropy_xlk"])
    
    # 7. Trimmed mean approach
    data["vol_trimmed"] = data[["hl_range", "hl_range_spy", "hl_range_xlk"]].apply(
        lambda x: np.mean(sorted(x)[1:2]), axis=1  # Middle value (median of 3)
    )
    
    # 8. Adaptive weighted combination
    # Weight recent data more when volatility is changing
    data["vol_change_rate"] = np.abs(data["hl_range"].pct_change(5))
    data["adaptive_alpha"] = np.clip(0.2 + data["vol_change_rate"] * 2, 0.2, 0.8)
    
    data["adaptive_weighted_vol"] = (
        data["hl_range"] * data["adaptive_alpha"] +
        data["hl_range"].rolling(window=10).mean() * (1 - data["adaptive_alpha"])
    )
    
    # 9. Principal component inspired combination
    # Simplified: use correlation-based weights
    corr_mat = data[["hl_range", "hl_range_spy", "hl_range_xlk"]].rolling(window=60).corr()
    data["pc_weight"] = 0.5  # Simplified for single condition constraint
    data["pc_inspired_vol"] = (
        data["hl_range"] * 0.5 +
        data["hl_range_spy"] * 0.3 +
        data["hl_range_xlk"] * 0.2
    )
    
    # 10. Non-linear transformation of best indicator
    # Apply optimal transformation to weighted_vol_micro
    data["weighted_vol_micro_sqrt"] = np.sqrt(data["weighted_vol_micro_v2"])
    data["weighted_vol_micro_log"] = np.log1p(data["weighted_vol_micro_v2"])
    data["weighted_vol_micro_cbrt"] = np.cbrt(data["weighted_vol_micro_v2"])
    
    # Apply moving averages
    for col in ["weighted_vol_micro_v2", "optimal_vol_invvar", "cluster_weighted_vol",
                "harm_geom_hybrid", "rank_avg", "entropy_weighted_vol", "vol_trimmed",
                "adaptive_weighted_vol", "pc_inspired_vol", "weighted_vol_micro_sqrt",
                "weighted_vol_micro_log", "weighted_vol_micro_cbrt"]:
        if col in data.columns:
            data[f"{col}_ma5"] = data[col].rolling(window=5).mean()
            data[f"{col}_ma3"] = data[col].rolling(window=3).mean()
            data[f"{col}_ma7"] = data[col].rolling(window=7).mean()
    
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
print("\n=== Wave 20: Оптимизация весов и продвинутые комбинации ===\n")

results = []

# 1. Refined microstructure-weighted volatility
print("Тестирую refined microstructure-weighted volatility...")
thresholds = [0.016, 0.018, 0.020, 0.022]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_v2_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_v2_ma5 < {threshold:.3f}", perf))

# Test with MA3
thresholds = [0.016, 0.018, 0.020]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_v2_ma3"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_v2_ma3 < {threshold:.3f}", perf))

# 2. Inverse variance weighted
print("\nТестирую inverse variance weighted...")
thresholds = [0.018, 0.020, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["optimal_vol_invvar_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"optimal_vol_invvar_ma5 < {threshold:.3f}", perf))

# 3. Cluster-strength weighted
print("\nТестирую cluster-strength weighted...")
thresholds = [0.018, 0.020, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["cluster_weighted_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"cluster_weighted_vol_ma5 < {threshold:.3f}", perf))

# 4. Harmonic-geometric hybrid
print("\nТестирую harmonic-geometric hybrid...")
thresholds = [0.018, 0.020, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["harm_geom_hybrid_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"harm_geom_hybrid_ma5 < {threshold:.3f}", perf))

# 5. Rank-based combination
print("\nТестирую rank-based combination...")
thresholds = [0.3, 0.35, 0.4, 0.45]
for threshold in thresholds:
    data["signal"] = (data["rank_avg_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"rank_avg_ma5 < {threshold:.2f}", perf))

# 6. Entropy-weighted
print("\nТестирую entropy-weighted...")
thresholds = [0.018, 0.020, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["entropy_weighted_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"entropy_weighted_vol_ma5 < {threshold:.3f}", perf))

# 7. Trimmed mean
print("\nТестирую trimmed mean...")
thresholds = [0.018, 0.020, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["vol_trimmed_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_trimmed_ma5 < {threshold:.3f}", perf))

# 8. Adaptive weighted
print("\nТестирую adaptive weighted...")
thresholds = [0.018, 0.020, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["adaptive_weighted_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"adaptive_weighted_vol_ma5 < {threshold:.3f}", perf))

# 9. Non-linear transformations of best indicator
print("\nТестирую non-linear transformations of weighted_vol_micro...")
# Square root
thresholds = [0.12, 0.13, 0.14, 0.15]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_sqrt_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_sqrt_ma5 < {threshold:.2f}", perf))

# Log transform
thresholds = [-4.2, -4.0, -3.8, -3.6]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_log_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_log_ma5 < {threshold:.1f}", perf))

# Cube root
thresholds = [0.25, 0.26, 0.27, 0.28]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_cbrt_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_cbrt_ma5 < {threshold:.2f}", perf))

# 10. Regime-based thresholds
print("\nТестирую regime-based conditions...")
# Use percentile-based thresholds
for window in [20, 50, 100]:
    for pct in [0.2, 0.3]:
        col = f"hl_pct_{window}_{int(pct*100)}"
        if col in data.columns:
            data["signal"] = (data[col].shift(1) < 1.0).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < 1.0", perf))

# 11. Different MA periods for best indicator
print("\nТестирую different MA periods...")
# MA7
thresholds = [0.018, 0.020, 0.022]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_v2_ma7"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_v2_ma7 < {threshold:.3f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 20 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск...")
    print(f"\nЛучший результат: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")