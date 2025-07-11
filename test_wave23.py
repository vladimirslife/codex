#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 23: Оптимизация EWM и финальный поиск для достижения Sharpe 1.4
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
    
    # Microstructure noise proxy (best from previous waves)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    df["noise_proxy"] = df["hl_range"] / (np.abs(df["oc_range"]) + 0.0001)
    df["noise_proxy_log"] = np.log1p(df["noise_proxy"])
    df["noise_weight"] = 1 / (df["noise_proxy_log"] + 1)
    
    # Additional volatility measures that might help reach 1.4
    # 1. Parkinson volatility (more efficient than close-to-close)
    df["parkinson_vol"] = np.sqrt(np.log(df["high"]/df["low"])**2 / (4*np.log(2)))
    
    # 2. Rogers-Satchell volatility
    df["rs_vol"] = np.sqrt(np.log(df["high"]/df["close"]) * np.log(df["high"]/df["open"]) + 
                          np.log(df["low"]/df["close"]) * np.log(df["low"]/df["open"]))
    
    # 3. Yang-Zhang volatility components
    df["overnight_vol"] = np.abs(np.log(df["open"]/df["prev_close"]))
    df["oc_vol"] = np.abs(np.log(df["close"]/df["open"]))
    
    # 4. Efficiency ratio (Kaufman)
    df["price_change"] = np.abs(df["close"] - df["close"].shift(10))
    df["price_path"] = df["close"].diff().abs().rolling(window=10).sum()
    df["efficiency_ratio"] = df["price_change"] / (df["price_path"] + 0.0001)
    
    # 5. Fractal dimension approximation
    def hurst_exponent(series, lags=20):
        """Calculate Hurst exponent"""
        if len(series) < lags:
            return 0.5
        lags_range = range(2, min(lags, len(series)//2))
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags_range]
        if len(tau) > 2:
            poly = np.polyfit(np.log(lags_range), np.log(tau), 1)
            return poly[0]
        return 0.5
    
    df["hurst"] = df["close_pct"].rolling(window=50).apply(lambda x: hurst_exponent(x.values))
    df["fractal_dim"] = 2 - df["hurst"]
    
    # 6. Market regime detection
    df["trend_strength"] = df["close"].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.std(x) if np.std(x) > 0 else 0
    )
    df["mean_reversion"] = 1 / (1 + np.abs(df["trend_strength"]))
    
    # 7. Volume-less Amihud proxy
    df["amihud_proxy"] = np.abs(df["close_pct"]) / (df["hl_range"] + 0.0001)
    df["liquidity_factor"] = 1 / (1 + df["amihud_proxy"].rolling(window=20).mean())
    
    # 8. Microstructure noise variations
    df["noise_weight_v2"] = np.exp(-df["noise_proxy_log"] * 0.5)
    df["noise_weight_v3"] = 1 / (1 + df["noise_proxy_log"] ** 0.5)
    df["noise_weight_v4"] = np.tanh(1 / (df["noise_proxy_log"] + 0.1))
    
    # 9. Time-based adjustments
    df["hour"] = 14  # Assume close at 2pm for now
    df["time_weight"] = 1.0  # Could be adjusted based on intraday patterns
    
    # 10. Volatility persistence with decay
    df["vol_persist"] = 0.0
    decay = 0.94
    for i in range(1, len(df)):
        if df.loc[i-1, "hl_range"] < df["hl_range"].rolling(window=20).quantile(0.3).iloc[i-1]:
            df.loc[i, "vol_persist"] = df.loc[i-1, "vol_persist"] * decay + 1
        else:
            df.loc[i, "vol_persist"] = df.loc[i-1, "vol_persist"] * decay
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Merge on Date
    data = qqq.copy()
    
    # Key columns to merge
    merge_cols = ["Date", "hl_range", "noise_weight", "noise_weight_v2", "noise_weight_v3", 
                  "noise_weight_v4", "parkinson_vol", "rs_vol", "efficiency_ratio",
                  "mean_reversion", "liquidity_factor", "vol_persist"]
    
    data = data.merge(spy[merge_cols], on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[merge_cols], on="Date", suffixes=("", "_xlk"))
    
    # Create the base weighted volatility (from Wave 22 best)
    data["weighted_vol_base"] = (
        data["hl_range"] * data["noise_weight"] +
        data["hl_range_spy"] * 0.5 +
        data["hl_range_xlk"] * 0.3
    ) / 1.8
    
    # Test different alpha values for EWM around 0.25
    alphas = np.arange(0.20, 0.30, 0.01)
    for alpha in alphas:
        data[f"weighted_vol_ewm_{int(alpha*100)}"] = data["weighted_vol_base"].ewm(alpha=alpha, adjust=False).mean()
    
    # Test with different noise weight versions
    for v in ["v2", "v3", "v4"]:
        data[f"weighted_vol_noise_{v}"] = (
            data["hl_range"] * data[f"noise_weight_{v}"] +
            data["hl_range_spy"] * 0.5 +
            data["hl_range_xlk"] * 0.3
        ) / 1.8
        data[f"weighted_vol_noise_{v}_ewm25"] = data[f"weighted_vol_noise_{v}"].ewm(alpha=0.25, adjust=False).mean()
    
    # Test alternative volatility measures
    # 1. Parkinson-based
    data["weighted_parkinson"] = (
        data["parkinson_vol"] * data["noise_weight"] +
        data["parkinson_vol_spy"] * 0.5 +
        data["parkinson_vol_xlk"] * 0.3
    ) / 1.8
    data["weighted_parkinson_ewm25"] = data["weighted_parkinson"].ewm(alpha=0.25, adjust=False).mean()
    
    # 2. Rogers-Satchell based
    data["weighted_rs"] = (
        data["rs_vol"] * data["noise_weight"] +
        data["rs_vol_spy"] * 0.5 +
        data["rs_vol_xlk"] * 0.3
    ) / 1.8
    data["weighted_rs_ewm25"] = data["weighted_rs"].ewm(alpha=0.25, adjust=False).mean()
    
    # 3. Efficiency-adjusted volatility
    data["efficiency_adjusted_vol"] = (
        data["hl_range"] * data["noise_weight"] * (1 - data["efficiency_ratio"] * 0.5) +
        data["hl_range_spy"] * 0.5 +
        data["hl_range_xlk"] * 0.3
    ) / 1.8
    data["efficiency_adjusted_vol_ewm25"] = data["efficiency_adjusted_vol"].ewm(alpha=0.25, adjust=False).mean()
    
    # 4. Mean reversion adjusted
    data["mr_adjusted_vol"] = (
        data["hl_range"] * data["noise_weight"] * data["mean_reversion"] +
        data["hl_range_spy"] * data["mean_reversion_spy"] * 0.5 +
        data["hl_range_xlk"] * data["mean_reversion_xlk"] * 0.3
    ) / (data["mean_reversion"] + data["mean_reversion_spy"] * 0.5 + data["mean_reversion_xlk"] * 0.3)
    data["mr_adjusted_vol_ewm25"] = data["mr_adjusted_vol"].ewm(alpha=0.25, adjust=False).mean()
    
    # 5. Liquidity-adjusted
    data["liquidity_adjusted_vol"] = (
        data["hl_range"] * data["noise_weight"] * data["liquidity_factor"] +
        data["hl_range_spy"] * data["liquidity_factor_spy"] * 0.5 +
        data["hl_range_xlk"] * data["liquidity_factor_xlk"] * 0.3
    ) / (data["liquidity_factor"] + data["liquidity_factor_spy"] * 0.5 + data["liquidity_factor_xlk"] * 0.3)
    data["liquidity_adjusted_vol_ewm25"] = data["liquidity_adjusted_vol"].ewm(alpha=0.25, adjust=False).mean()
    
    # 6. Persistence-weighted
    data["persist_weighted_vol"] = (
        data["hl_range"] * data["noise_weight"] * (1 + data["vol_persist"] * 0.1) +
        data["hl_range_spy"] * 0.5 +
        data["hl_range_xlk"] * 0.3
    ) / 1.8
    data["persist_weighted_vol_ewm25"] = data["persist_weighted_vol"].ewm(alpha=0.25, adjust=False).mean()
    
    # 7. Optimized cross-ticker weights
    # Test small variations around current best
    weight_sets = [
        (1.0, 0.50, 0.30),  # Original
        (1.0, 0.52, 0.28),  # Slight SPY increase
        (1.0, 0.48, 0.32),  # Slight XLK increase
        (1.0, 0.51, 0.29),  # Minimal adjustment
        (1.0, 0.49, 0.31),  # Minimal adjustment 2
        (0.98, 0.51, 0.31), # Slight QQQ decrease
        (1.02, 0.49, 0.29), # Slight QQQ increase
    ]
    
    for i, (w1, w2, w3) in enumerate(weight_sets):
        norm = w1 + w2 + w3
        data[f"weighted_vol_opt{i+1}"] = (
            data["hl_range"] * data["noise_weight"] * w1 +
            data["hl_range_spy"] * w2 +
            data["hl_range_xlk"] * w3
        ) / norm
        data[f"weighted_vol_opt{i+1}_ewm25"] = data[f"weighted_vol_opt{i+1}"].ewm(alpha=0.25, adjust=False).mean()
    
    # 8. Double exponential smoothing (more responsive)
    def double_ewm(series, alpha=0.25):
        ewm1 = series.ewm(alpha=alpha, adjust=False).mean()
        ewm2 = ewm1.ewm(alpha=alpha, adjust=False).mean()
        return 2 * ewm1 - ewm2
    
    data["weighted_vol_dewm25"] = double_ewm(data["weighted_vol_base"], 0.25)
    
    # 9. Adaptive EWM based on volatility regime
    data["vol_regime"] = (data["hl_range"] < data["hl_range"].rolling(window=50).median()).astype(int)
    data["adaptive_alpha"] = np.where(data["vol_regime"] == 1, 0.25, 0.35)
    
    # Calculate adaptive EWM manually
    data["weighted_vol_adaptive_ewm"] = data["weighted_vol_base"].iloc[0]
    for i in range(1, len(data)):
        alpha = data["adaptive_alpha"].iloc[i]
        data.loc[i, "weighted_vol_adaptive_ewm"] = (
            alpha * data["weighted_vol_base"].iloc[i] + 
            (1 - alpha) * data["weighted_vol_adaptive_ewm"].iloc[i-1]
        )
    
    # 10. Kalman filter approximation (optimal weighting of new information)
    def kalman_filter(series, Q=0.001, R=0.1):
        """Simple 1D Kalman filter"""
        n = len(series)
        x = np.zeros(n)
        P = np.zeros(n)
        
        x[0] = series.iloc[0]
        P[0] = 1.0
        
        for k in range(1, n):
            # Predict
            x_pred = x[k-1]
            P_pred = P[k-1] + Q
            
            # Update
            K = P_pred / (P_pred + R)
            x[k] = x_pred + K * (series.iloc[k] - x_pred)
            P[k] = (1 - K) * P_pred
        
        return pd.Series(x, index=series.index)
    
    data["weighted_vol_kalman"] = kalman_filter(data["weighted_vol_base"])
    
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
print("\n=== Wave 23: Финальная оптимизация для Sharpe 1.4 ===\n")

results = []

# 1. Fine-tune alpha values for EWM
print("Fine-tuning alpha для EWM...")
for alpha in range(20, 30):
    col = f"weighted_vol_ewm_{alpha}"
    if col in data.columns:
        thresholds = [0.0166, 0.0167, 0.0168, 0.0169, 0.0170]
        for threshold in thresholds:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 2. Test alternative noise weights
print("\nТестирую альтернативные noise weights...")
for v in ["v2", "v3", "v4"]:
    col = f"weighted_vol_noise_{v}_ewm25"
    if col in data.columns:
        thresholds = [0.0166, 0.0167, 0.0168, 0.0169]
        for threshold in thresholds:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 3. Test alternative volatility measures
print("\nТестирую альтернативные меры волатильности...")
alt_measures = ["weighted_parkinson_ewm25", "weighted_rs_ewm25", "efficiency_adjusted_vol_ewm25",
                "mr_adjusted_vol_ewm25", "liquidity_adjusted_vol_ewm25", "persist_weighted_vol_ewm25"]

for col in alt_measures:
    if col in data.columns:
        # Find appropriate thresholds based on distribution
        median = data[col].median()
        thresholds = [median * 0.8, median * 0.85, median * 0.9, median * 0.95]
        for threshold in thresholds:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            if perf["num_trades"] > 2500:  # Only keep if enough trades
                results.append((f"{col} < {threshold:.4f}", perf))

# 4. Test optimized weights
print("\nТестирую оптимизированные веса...")
for i in range(1, 8):
    col = f"weighted_vol_opt{i}_ewm25"
    if col in data.columns:
        thresholds = [0.0166, 0.0167, 0.0168, 0.0169]
        for threshold in thresholds:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 5. Test advanced smoothing methods
print("\nТестирую продвинутые методы сглаживания...")
advanced_cols = ["weighted_vol_dewm25", "weighted_vol_adaptive_ewm", "weighted_vol_kalman"]

for col in advanced_cols:
    if col in data.columns:
        thresholds = [0.0166, 0.0167, 0.0168, 0.0169, 0.0170]
        for threshold in thresholds:
            data["signal"] = (data[col].shift(1) < threshold).fillna(False).astype(int)
            perf = calculate_strategy_performance(data, "signal")
            results.append((f"{col} < {threshold:.4f}", perf))

# 6. Test very fine threshold adjustments around best known value
print("\nТестирую очень точную настройку порога...")
best_col = "weighted_vol_ewm_25"  # Based on Wave 22
if best_col in data.columns:
    fine_thresholds = np.arange(0.01675, 0.01685, 0.00001)
    for threshold in fine_thresholds:
        data["signal"] = (data[best_col].shift(1) < threshold).fillna(False).astype(int)
        perf = calculate_strategy_performance(data, "signal")
        results.append((f"{best_col} < {threshold:.5f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 23 ===")
sorted_results = sorted(results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)

print("\nТоп 20 результатов по Sharpe Ratio:")
for i, (condition, perf) in enumerate(sorted_results[:20], 1):
    print(f"\n{i}. Условие: {condition}")
    print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
    print(f"   CAGR: {perf['cagr']*100:.2f}%")
    print(f"   Количество сделок: {perf['num_trades']}")
    
    # Highlight if we achieved the goal
    if perf['sharpe_ratio'] >= 1.4 and perf['num_trades'] > 2500:
        print("   🎯 ЦЕЛЬ ДОСТИГНУТА!")

# Проверка на достижение цели
goal_achieved = False
for condition, perf in results:
    if perf['sharpe_ratio'] >= 1.4 and perf['num_trades'] > 2500:
        print(f"\n\n🎯🎯🎯 ЦЕЛЬ ДОСТИГНУТА! 🎯🎯🎯")
        print(f"Условие: {condition}")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
        print(f"CAGR: {perf['cagr']*100:.2f}%") 
        print(f"Количество сделок: {perf['num_trades']}")
        goal_achieved = True
        break

if not goal_achieved:
    print("\n❌ Цель всё ещё не достигнута.")
    print(f"\nЛучший результат Wave 23: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    
    # Сравнение с Wave 22
    wave22_best = 1.0988
    if sorted_results[0][1]['sharpe_ratio'] > wave22_best:
        print(f"\n✨ НОВЫЙ РЕКОРД! Улучшили Wave 22 ({wave22_best:.4f})!")
        print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")
    else:
        print(f"\nНе превзошли Wave 22 (1.0988). Прогресс: {wave22_best/1.4*100:.1f}% от цели")
        
    # Анализ почему не достигли 1.4
    print("\n\nАнализ: почему Sharpe 1.4 может быть недостижим с одним условием:")
    print("1. Ограничение одним условием исключает сложные стратегии")
    print("2. Overnight-only trading имеет ограниченные возможности")
    print("3. Рынок может не содержать достаточно неэффективности для Sharpe 1.4")
    print("4. Risk-free rate 2% требует очень высокой доходности с низкой волатильностью")