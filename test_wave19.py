#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 19: Market microstructure patterns and extreme value theory
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
    
    # Market microstructure patterns
    # 1. Open-Close range (directional movement)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    df["oc_range_abs"] = np.abs(df["oc_range"])
    
    # 2. Close position within day's range
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 0.0001)
    
    # 3. Gap analysis
    df["gap"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    df["gap_filled"] = ((df["gap"] > 0) & (df["low"] <= df["prev_close"])) | \
                       ((df["gap"] < 0) & (df["high"] >= df["prev_close"]))
    df["gap_filled_ma5"] = df["gap_filled"].rolling(window=5).mean()
    
    # 4. Price efficiency ratio
    df["price_path"] = df["hl_range"]
    df["price_displacement"] = np.abs(df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    df["efficiency_ratio"] = df["price_displacement"] / (df["price_path"].rolling(window=5).sum() + 0.0001)
    
    # 5. Microstructure noise proxy
    df["noise_proxy"] = df["hl_range"] / (np.abs(df["oc_range"]) + 0.0001)
    df["noise_proxy_log"] = np.log1p(df["noise_proxy"])
    
    # Extreme value indicators
    # 6. Pareto tail index
    def pareto_tail_index(series, threshold_pct=0.95):
        """Estimate Pareto tail index for extreme values"""
        threshold = series.quantile(threshold_pct)
        exceedances = series[series > threshold] - threshold
        if len(exceedances) > 10:
            # Hill estimator
            return len(exceedances) / np.sum(np.log(exceedances / exceedances.min()))
        return np.nan
    
    df["vol_tail_index"] = df["hl_range"].rolling(window=100).apply(pareto_tail_index)
    
    # 7. Extreme value occurrence
    df["extreme_low_vol"] = (df["hl_range"] < df["hl_range"].rolling(window=252).quantile(0.05)).astype(int)
    df["extreme_low_vol_ma5"] = df["extreme_low_vol"].rolling(window=5).mean()
    
    # 8. Return time to extremes
    df["days_since_extreme_low"] = 0
    counter = 0
    for i in range(len(df)):
        if df.loc[i, "extreme_low_vol"] == 1:
            counter = 0
        else:
            counter += 1
        df.loc[i, "days_since_extreme_low"] = counter
    
    # Pattern recognition
    # 9. Consolidation pattern (low volatility + low directional movement)
    df["consolidation"] = ((df["hl_range"] < df["hl_range"].rolling(window=20).quantile(0.4)) & 
                          (df["oc_range_abs"] < df["oc_range_abs"].rolling(window=20).quantile(0.4))).astype(int)
    df["consolidation_days"] = df["consolidation"].rolling(window=10).sum()
    
    # 10. Volatility contraction pattern
    df["vol_contraction"] = (df["hl_range"].rolling(window=5).mean() / 
                            df["hl_range"].rolling(window=20).mean())
    
    # 11. Bollinger Band width
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_width"] = (df["bb_std"] * 2) / df["bb_middle"]
    
    # 12. Keltner Channel width
    df["kc_atr"] = df["hl_range"].rolling(window=20).mean()
    df["kc_width"] = (df["kc_atr"] * 2) / df["bb_middle"]
    
    # 13. Squeeze indicator (BB inside KC)
    df["squeeze"] = (df["bb_width"] < df["kc_width"]).astype(int)
    df["squeeze_ma5"] = df["squeeze"].rolling(window=5).mean()
    
    # Information theory
    # 14. Approximate entropy of price movements
    def approx_entropy(series, m=2, r=0.2):
        """Calculate approximate entropy"""
        N = len(series)
        if N < m + 1:
            return np.nan
        
        def _maxdist(xi, xj, m):
            return max([abs(float(xi[k]) - float(xj[k])) for k in range(m)])
        
        def _phi(m):
            patterns = np.array([series[i:i+m] for i in range(N-m+1)])
            C = np.zeros(N-m+1)
            for i in range(N-m+1):
                template = patterns[i]
                for j in range(N-m+1):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1
            phi = (N-m+1)**(-1) * np.sum(np.log(C/(N-m+1)))
            return phi
        
        try:
            return _phi(m) - _phi(m+1)
        except:
            return np.nan
    
    df["price_entropy"] = df["close_pct"].rolling(window=20).apply(lambda x: approx_entropy(x.values))
    
    # 15. Detrended fluctuation analysis (simplified)
    def dfa_alpha(series, window=20):
        """Simplified DFA to measure long-range dependence"""
        if len(series) < window:
            return np.nan
        try:
            # Integrate series
            Y = np.cumsum(series - np.mean(series))
            # Divide into boxes and detrend
            n_boxes = len(Y) // window
            F = []
            for i in range(n_boxes):
                box = Y[i*window:(i+1)*window]
                x = np.arange(len(box))
                coef = np.polyfit(x, box, 1)
                fit = np.polyval(coef, x)
                F.append(np.sqrt(np.mean((box - fit)**2)))
            # Return scaling exponent (simplified)
            return np.mean(F) / np.std(series) if np.std(series) > 0 else np.nan
        except:
            return np.nan
    
    df["dfa_alpha"] = df["close_pct"].rolling(window=50).apply(lambda x: dfa_alpha(x.values))
    
    # Apply moving averages to key indicators
    for col in ["oc_range_abs", "close_position", "efficiency_ratio", "noise_proxy_log",
                "vol_contraction", "bb_width", "kc_width", "consolidation_days",
                "price_entropy", "dfa_alpha"]:
        if col in df.columns:
            df[f"{col}_ma5"] = df[col].rolling(window=5).mean()
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Merge on Date
    data = qqq.copy()
    data = data.merge(spy[["Date", "hl_range", "bb_width", "squeeze", "vol_contraction"]], 
                      on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[["Date", "hl_range", "bb_width", "squeeze", "vol_contraction"]], 
                      on="Date", suffixes=("", "_xlk"))
    
    # Cross-market microstructure
    # 1. Market-wide squeeze
    data["market_squeeze"] = (data["squeeze"] + data["squeeze_spy"] + data["squeeze_xlk"]) / 3
    data["market_squeeze_ma5"] = data["market_squeeze"].rolling(window=5).mean()
    
    # 2. Volatility contraction synchronization
    data["vol_contraction_sync"] = np.minimum(
        np.minimum(data["vol_contraction"], data["vol_contraction_spy"]),
        data["vol_contraction_xlk"]
    )
    data["vol_contraction_sync_ma5"] = data["vol_contraction_sync"].rolling(window=5).mean()
    
    # 3. Cross-market efficiency
    data["market_efficiency"] = (data["efficiency_ratio"] + 
                                data.merge(spy[["Date", "efficiency_ratio"]], on="Date", suffixes=("", "_spy2"))["efficiency_ratio_spy2"] +
                                data.merge(xlk[["Date", "efficiency_ratio"]], on="Date", suffixes=("", "_xlk2"))["efficiency_ratio_xlk2"]) / 3
    
    # 4. Extreme value synchronization
    data["extreme_sync"] = (
        (data["hl_range"] < data["hl_range"].rolling(window=100).quantile(0.1)) &
        (data["hl_range_spy"] < data["hl_range_spy"].rolling(window=100).quantile(0.1))
    ).astype(int)
    data["extreme_sync_ma5"] = data["extreme_sync"].rolling(window=5).mean()
    
    # 5. Microstructure divergence
    data["micro_divergence"] = np.std([data["bb_width"], data["bb_width_spy"], data["bb_width_xlk"]], axis=0)
    data["micro_divergence_ma5"] = data["micro_divergence"].rolling(window=5).mean()
    
    # 6. Weighted volatility using microstructure
    # Weight by inverse of noise proxy
    data["weighted_vol_micro"] = (
        data["hl_range"] * (1 / (data["noise_proxy_log"] + 1)) +
        data["hl_range_spy"] * 0.5 +
        data["hl_range_xlk"] * 0.3
    ) / 1.8
    data["weighted_vol_micro_ma5"] = data["weighted_vol_micro"].rolling(window=5).mean()
    
    # 7. Pattern-based indicator
    data["pattern_score"] = (
        data["consolidation_days"] * 0.3 +
        data["squeeze_ma5"] * 5 +
        (1 - data["vol_contraction"]) * 2
    )
    data["pattern_score_ma5"] = data["pattern_score"].rolling(window=5).mean()
    
    # 8. Optimal weighted combination based on correlation
    # Use rolling correlation to weight tickers
    corr_window = 60
    data["corr_spy"] = data["hl_range"].rolling(window=corr_window).corr(data["hl_range_spy"])
    data["corr_xlk"] = data["hl_range"].rolling(window=corr_window).corr(data["hl_range_xlk"])
    
    # Inverse correlation weighting
    data["inv_corr_weight"] = 1 / (data["corr_spy"].abs() + data["corr_xlk"].abs() + 0.1)
    data["optimal_vol"] = (
        data["hl_range"] * data["inv_corr_weight"] +
        data["hl_range_spy"] * (1 - data["corr_spy"].abs()) +
        data["hl_range_xlk"] * (1 - data["corr_xlk"].abs())
    ) / (data["inv_corr_weight"] + 2 - data["corr_spy"].abs() - data["corr_xlk"].abs())
    data["optimal_vol_ma5"] = data["optimal_vol"].rolling(window=5).mean()
    
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
print("\n=== Wave 19: Market microstructure & extreme values ===\n")

results = []

# 1. Market-wide squeeze
print("Тестирую market-wide squeeze...")
thresholds = [0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    data["signal"] = (data["market_squeeze_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"market_squeeze_ma5 > {threshold:.1f}", perf))

# 2. Volatility contraction synchronization
print("\nТестирую volatility contraction sync...")
thresholds = [0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    data["signal"] = (data["vol_contraction_sync_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_contraction_sync_ma5 < {threshold:.1f}", perf))

# 3. Weighted volatility microstructure
print("\nТестирую weighted volatility microstructure...")
thresholds = [0.018, 0.02, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["weighted_vol_micro_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_vol_micro_ma5 < {threshold:.3f}", perf))

# 4. Optimal volatility (correlation-weighted)
print("\nТестирую optimal volatility...")
thresholds = [0.018, 0.02, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["optimal_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"optimal_vol_ma5 < {threshold:.3f}", perf))

# 5. Pattern score
print("\nТестирую pattern score...")
thresholds = [2, 3, 4, 5]
for threshold in thresholds:
    data["signal"] = (data["pattern_score_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"pattern_score_ma5 > {threshold}", perf))

# 6. Extreme synchronization
print("\nТестирую extreme synchronization...")
thresholds = [0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    data["signal"] = (data["extreme_sync_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"extreme_sync_ma5 > {threshold:.1f}", perf))

# 7. Microstructure divergence (low divergence = high agreement)
print("\nТестирую microstructure divergence...")
thresholds = [0.002, 0.003, 0.004, 0.005]
for threshold in thresholds:
    data["signal"] = (data["micro_divergence_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"micro_divergence_ma5 < {threshold:.3f}", perf))

# 8. Bollinger Band width
print("\nТестирую Bollinger Band width...")
thresholds = [0.02, 0.025, 0.03, 0.035]
for threshold in thresholds:
    data["signal"] = (data["bb_width_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"bb_width_ma5 < {threshold:.3f}", perf))

# 9. Consolidation days
print("\nТестирую consolidation days...")
thresholds = [5, 6, 7, 8]
for threshold in thresholds:
    data["signal"] = (data["consolidation_days_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"consolidation_days_ma5 > {threshold}", perf))

# 10. Efficiency ratio
print("\nТестирую efficiency ratio...")
thresholds = [0.3, 0.4, 0.5, 0.6]
for threshold in thresholds:
    data["signal"] = (data["efficiency_ratio_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"efficiency_ratio_ma5 > {threshold:.1f}", perf))

# 11. Noise proxy (low noise = clear trend)
print("\nТестирую noise proxy...")
thresholds = [2.0, 2.5, 3.0, 3.5]
for threshold in thresholds:
    data["signal"] = (data["noise_proxy_log_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"noise_proxy_log_ma5 < {threshold:.1f}", perf))

# 12. Close position
print("\nТестирую close position...")
thresholds = [0.3, 0.4, 0.5, 0.6]
for threshold in thresholds:
    data["signal"] = ((data["close_position_ma5"].shift(1) > threshold) & 
                     (data["close_position_ma5"].shift(1) < 1-threshold)).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"close_position_ma5 in [{threshold:.1f}, {1-threshold:.1f}]", perf))

# 13. Days since extreme low volatility
print("\nТестирую days since extreme low...")
thresholds = [10, 20, 30, 40]
for threshold in thresholds:
    data["signal"] = (data["days_since_extreme_low"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"days_since_extreme_low < {threshold}", perf))

# 14. Price entropy
print("\nТестирую price entropy...")
thresholds = [0.1, 0.2, 0.3, 0.4]
for threshold in thresholds:
    data["signal"] = (data["price_entropy_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"price_entropy_ma5 < {threshold:.1f}", perf))

# 15. DFA alpha
print("\nТестирую DFA alpha...")
thresholds = [0.5, 1.0, 1.5, 2.0]
for threshold in thresholds:
    data["signal"] = (data["dfa_alpha_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"dfa_alpha_ma5 < {threshold:.1f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 19 ===")
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
    print("\n❌ Цель не достигнута. Необходимо продолжить поиск в Wave 20...")
    print(f"\nЛучший результат: {sorted_results[0][0]}")
    print(f"Sharpe Ratio: {sorted_results[0][1]['sharpe_ratio']:.4f}")
    print(f"Прогресс: {sorted_results[0][1]['sharpe_ratio']/1.4*100:.1f}% от цели")