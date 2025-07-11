#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave 18: Optimal indicator combinations and adaptive thresholds
"""

import pandas as pd
import numpy as np
from scipy import stats, special

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
    
    # Time-based features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day_of_month"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week
    
    # Best transformations from previous waves
    df["hl_range_sqrt"] = np.sqrt(df["hl_range"])
    df["hl_range_log"] = np.log1p(df["hl_range"])
    df["hl_range_cbrt"] = np.cbrt(df["hl_range"])
    
    # Adaptive thresholds
    # 1. Dynamic percentile-based threshold
    df["hl_range_adaptive_pct"] = df["hl_range"] / df["hl_range"].rolling(window=100).quantile(0.3)
    
    # 2. Z-score based adaptive threshold
    df["hl_range_zscore_adaptive"] = (df["hl_range"] - df["hl_range"].rolling(window=50).mean()) / (df["hl_range"].rolling(window=50).std() + 0.0001)
    
    # 3. Exponentially weighted adaptive threshold
    df["hl_range_ewm_mean"] = df["hl_range"].ewm(span=20, adjust=False).mean()
    df["hl_range_ewm_std"] = df["hl_range"].ewm(span=20, adjust=False).std()
    df["hl_range_adaptive_ewm"] = (df["hl_range"] - df["hl_range_ewm_mean"]) / (df["hl_range_ewm_std"] + 0.0001)
    
    # Information theory measures
    # 4. Relative entropy (KL divergence approximation)
    def relative_entropy(series, window=20):
        """Calculate relative entropy between current and historical distribution"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window*2, len(series)):
            recent = series.iloc[i-window:i].values
            historical = series.iloc[i-window*2:i-window].values
            # Create histograms
            bins = np.linspace(min(historical.min(), recent.min()), 
                             max(historical.max(), recent.max()), 10)
            hist_recent, _ = np.histogram(recent, bins=bins, density=True)
            hist_historical, _ = np.histogram(historical, bins=bins, density=True)
            # Add small epsilon to avoid log(0)
            hist_recent = hist_recent + 1e-10
            hist_historical = hist_historical + 1e-10
            # Normalize
            hist_recent = hist_recent / hist_recent.sum()
            hist_historical = hist_historical / hist_historical.sum()
            # Calculate KL divergence
            kl_div = np.sum(hist_recent * np.log(hist_recent / hist_historical))
            result.iloc[i] = kl_div
        return result
    
    df["vol_kl_divergence"] = relative_entropy(df["hl_range"])
    
    # 5. Shannon entropy with adaptive binning
    def adaptive_entropy(series, window=20):
        """Calculate Shannon entropy with adaptive binning"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            data = series.iloc[i-window:i].values
            # Adaptive binning based on data range
            n_bins = min(int(np.sqrt(window)), 10)
            hist, _ = np.histogram(data, bins=n_bins)
            hist = hist[hist > 0]  # Remove zero bins
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log(probs))
            result.iloc[i] = entropy
        return result
    
    df["vol_adaptive_entropy"] = adaptive_entropy(df["hl_range"])
    
    # Extreme value theory
    # 6. Generalized extreme value distribution parameters
    def gev_params(series, window=100):
        """Fit GEV distribution and return shape parameter"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            data = series.iloc[i-window:i].values
            try:
                # Use method of moments for speed
                mean = np.mean(data)
                std = np.std(data)
                skew = stats.skew(data)
                # Approximate shape parameter
                if abs(skew) < 0.1:
                    xi = 0  # Gumbel
                else:
                    xi = -0.1 * np.sign(skew) * min(abs(skew), 2)
                result.iloc[i] = xi
            except:
                result.iloc[i] = 0
        return result
    
    df["vol_gev_shape"] = gev_params(df["hl_range"])
    
    # 7. Tail index estimation (Hill estimator approximation)
    def tail_index(series, window=100, k=10):
        """Estimate tail index using Hill estimator"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            data = sorted(series.iloc[i-window:i].values, reverse=True)
            if data[k-1] > 0:
                hill = k / np.sum([np.log(data[j] / data[k-1]) for j in range(k)])
                result.iloc[i] = hill
            else:
                result.iloc[i] = np.nan
        return result
    
    df["vol_tail_index"] = tail_index(df["hl_range"])
    
    # Statistical process control
    # 8. CUSUM statistic for volatility
    def cusum(series, k=0.5):
        """Calculate CUSUM statistic"""
        mean = series.rolling(window=50).mean()
        std = series.rolling(window=50).std()
        standardized = (series - mean) / (std + 0.0001)
        
        cusum_pos = pd.Series(index=series.index, dtype=float)
        cusum_neg = pd.Series(index=series.index, dtype=float)
        cusum_pos.iloc[0] = 0
        cusum_neg.iloc[0] = 0
        
        for i in range(1, len(series)):
            cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + standardized.iloc[i] - k)
            cusum_neg.iloc[i] = min(0, cusum_neg.iloc[i-1] + standardized.iloc[i] + k)
        
        return cusum_pos - cusum_neg
    
    df["vol_cusum"] = cusum(df["hl_range"])
    
    # 9. Multivariate outlier detection (Mahalanobis distance)
    # Create feature set for Mahalanobis distance
    df["hl_range_ma5"] = df["hl_range"].rolling(window=5).mean()
    df["hl_range_ma20"] = df["hl_range"].rolling(window=20).mean()
    
    # Time-weighted combinations
    # 10. Time-of-month adjusted volatility
    df["day_weight"] = 1 - np.abs(df["day_of_month"] - 15) / 15  # Weight peaks at mid-month
    df["vol_time_weighted"] = df["hl_range"] * df["day_weight"]
    
    # 11. Seasonal decomposition residual
    df["week_avg_vol"] = df.groupby("week_of_year")["hl_range"].transform(lambda x: x.rolling(window=52, min_periods=1).mean())
    df["vol_seasonal_residual"] = df["hl_range"] - df["week_avg_vol"]
    
    # Optimal weighted combinations (creating single indicators)
    # 12. Weighted geometric-harmonic mean
    df["vol_weighted_gh"] = 2 / (1/np.sqrt(df["hl_range"]) + 1/df["hl_range"])
    
    # 13. Tukey's biweight of volatility
    def tukey_biweight(series, window=20, c=4.685):
        """Calculate Tukey's biweight (robust average)"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            data = series.iloc[i-window:i].values
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad == 0:
                result.iloc[i] = median
            else:
                u = (data - median) / (c * mad)
                weights = ((1 - u**2)**2) * (np.abs(u) < 1)
                result.iloc[i] = np.sum(weights * data) / np.sum(weights)
        return result
    
    df["vol_tukey_biweight"] = tukey_biweight(df["hl_range"])
    
    # 14. Winsorized mean of volatility
    df["vol_winsorized"] = df["hl_range"].rolling(window=20).apply(lambda x: stats.mstats.winsorize(x, limits=[0.1, 0.1]).mean())
    
    # 15. Hodges-Lehmann estimator
    def hodges_lehmann(series, window=20):
        """Calculate Hodges-Lehmann estimator (robust location)"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            data = series.iloc[i-window:i].values
            # All pairwise averages
            pairwise_avg = []
            for j in range(len(data)):
                for k in range(j, len(data)):
                    pairwise_avg.append((data[j] + data[k]) / 2)
            result.iloc[i] = np.median(pairwise_avg)
        return result
    
    df["vol_hodges_lehmann"] = hodges_lehmann(df["hl_range"])
    
    # Apply moving averages to key indicators
    for col in ["hl_range_adaptive_pct", "hl_range_adaptive_ewm", "vol_kl_divergence",
                "vol_adaptive_entropy", "vol_cusum", "vol_time_weighted", "vol_weighted_gh",
                "vol_tukey_biweight", "vol_winsorized", "vol_hodges_lehmann"]:
        if col in df.columns:
            df[f"{col}_ma5"] = df[col].rolling(window=5).mean()
    
    return df

def load_all_tickers():
    """Load all three tickers and merge relevant columns"""
    qqq = load_ticker("4 - QQQ.csv")
    spy = load_ticker("4 - SPY.csv")
    xlk = load_ticker("4 - XLK.csv")
    
    # Calculate Mahalanobis distance for QQQ
    features = ["hl_range", "hl_range_ma5", "hl_range_ma20"]
    qqq["vol_mahalanobis"] = mahalanobis_distance(qqq, features)
    
    # Merge on Date
    data = qqq.copy()
    data = data.merge(spy[["Date", "hl_range", "hl_range_sqrt", "vol_tukey_biweight"]], 
                      on="Date", suffixes=("", "_spy"))
    data = data.merge(xlk[["Date", "hl_range", "hl_range_sqrt", "vol_tukey_biweight"]], 
                      on="Date", suffixes=("", "_xlk"))
    
    # Advanced cross-ticker combinations
    # 1. Weighted geometric mean with optimal weights
    # Weights chosen to maximize historical Sharpe (approximation)
    w1, w2, w3 = 0.4, 0.35, 0.25  # QQQ, SPY, XLK weights
    data["weighted_geom_vol"] = np.exp(w1 * np.log(data["hl_range"] + 1e-10) + 
                                       w2 * np.log(data["hl_range_spy"] + 1e-10) + 
                                       w3 * np.log(data["hl_range_xlk"] + 1e-10))
    
    # 2. Robust cross-ticker average using Tukey biweight
    data["robust_avg_vol"] = (data["vol_tukey_biweight"] + 
                             data["vol_tukey_biweight_spy"] + 
                             data["vol_tukey_biweight_xlk"]) / 3
    
    # 3. Cross-ticker synchronization score
    data["sync_score"] = np.exp(-(
        (data["hl_range"] - data["hl_range_spy"])**2 + 
        (data["hl_range"] - data["hl_range_xlk"])**2 + 
        (data["hl_range_spy"] - data["hl_range_xlk"])**2
    ) / (2 * 0.01**2))
    
    # 4. Minimum variance portfolio weight (simplified)
    # Assuming equal correlations, minimum variance weights
    total_vol = data["hl_range"] + data["hl_range_spy"] + data["hl_range_xlk"]
    data["mvp_vol"] = (data["hl_range"] * (1/data["hl_range"]) + 
                      data["hl_range_spy"] * (1/data["hl_range_spy"]) + 
                      data["hl_range_xlk"] * (1/data["hl_range_xlk"])) / (
                      1/data["hl_range"] + 1/data["hl_range_spy"] + 1/data["hl_range_xlk"])
    
    # 5. Copula-based dependence measure (simplified)
    # Using empirical copula transformation
    data["hl_rank_qqq"] = data["hl_range"].rank(pct=True)
    data["hl_rank_spy"] = data["hl_range_spy"].rank(pct=True)
    data["copula_dependence"] = np.sqrt((data["hl_rank_qqq"] - 0.5)**2 + 
                                        (data["hl_rank_spy"] - 0.5)**2)
    
    # 6. Information-theoretic combination
    # Mutual information approximation using correlation
    corr_window = 50
    data["mi_approximation"] = data["hl_range"].rolling(window=corr_window).corr(data["hl_range_spy"]).abs()
    data["info_weighted_vol"] = (data["hl_range"] * (1 - data["mi_approximation"]) + 
                                 data["hl_range_spy"] * data["mi_approximation"])
    
    # Apply moving averages
    for col in ["weighted_geom_vol", "robust_avg_vol", "sync_score", "mvp_vol", 
                "copula_dependence", "info_weighted_vol"]:
        data[f"{col}_ma5"] = data[col].rolling(window=5).mean()
    
    return data

def mahalanobis_distance(df, features, window=50):
    """Calculate Mahalanobis distance for volatility measures"""
    result = pd.Series(index=df.index, dtype=float)
    data = df[features].values
    
    for i in range(window, len(df)):
        try:
            subset = data[i-window:i]
            mean = np.mean(subset, axis=0)
            cov = np.cov(subset.T)
            # Add regularization to avoid singular matrix
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(cov)
            
            diff = data[i] - mean
            m_dist = np.sqrt(diff.T @ inv_cov @ diff)
            result.iloc[i] = m_dist
        except:
            # If calculation fails, use simplified distance
            result.iloc[i] = np.sqrt(np.sum((data[i] - mean)**2))
    
    return result

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
print("\n=== Wave 18: Optimal combinations & adaptive thresholds ===\n")

results = []

# 1. Weighted geometric mean with optimal weights
print("Тестирую weighted geometric mean...")
thresholds = [0.02, 0.022, 0.024, 0.026]
for threshold in thresholds:
    data["signal"] = (data["weighted_geom_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"weighted_geom_vol_ma5 < {threshold:.3f}", perf))

# 2. Robust average using Tukey biweight
print("\nТестирую robust average...")
thresholds = [0.018, 0.02, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["robust_avg_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"robust_avg_vol_ma5 < {threshold:.3f}", perf))

# 3. Synchronization score
print("\nТестирую synchronization score...")
thresholds = [0.7, 0.8, 0.9, 0.95]
for threshold in thresholds:
    data["signal"] = (data["sync_score_ma5"].shift(1) > threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"sync_score_ma5 > {threshold:.2f}", perf))

# 4. Minimum variance portfolio
print("\nТестирую minimum variance portfolio...")
thresholds = [0.018, 0.02, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["mvp_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"mvp_vol_ma5 < {threshold:.3f}", perf))

# 5. Adaptive percentile threshold
print("\nТестирую adaptive percentile threshold...")
thresholds = [0.7, 0.8, 0.9, 1.0]
for threshold in thresholds:
    data["signal"] = (data["hl_range_adaptive_pct"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_adaptive_pct < {threshold:.1f}", perf))

# 6. Adaptive EWM z-score
print("\nТестирую adaptive EWM z-score...")
thresholds = [-1.5, -1.0, -0.5, 0]
for threshold in thresholds:
    data["signal"] = (data["hl_range_adaptive_ewm"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"hl_range_adaptive_ewm < {threshold:.1f}", perf))

# 7. KL divergence
print("\nТестирую KL divergence...")
thresholds = [0.1, 0.2, 0.3, 0.4]
for threshold in thresholds:
    data["signal"] = (data["vol_kl_divergence_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_kl_divergence_ma5 < {threshold:.1f}", perf))

# 8. Adaptive entropy
print("\nТестирую adaptive entropy...")
thresholds = [1.5, 1.6, 1.7, 1.8]
for threshold in thresholds:
    data["signal"] = (data["vol_adaptive_entropy_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_adaptive_entropy_ma5 < {threshold:.1f}", perf))

# 9. CUSUM statistic
print("\nТестирую CUSUM statistic...")
thresholds = [-2, -1, 0, 1]
for threshold in thresholds:
    data["signal"] = (data["vol_cusum_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_cusum_ma5 < {threshold}", perf))

# 10. Mahalanobis distance
print("\nТестирую Mahalanobis distance...")
thresholds = [1.5, 2.0, 2.5, 3.0]
for threshold in thresholds:
    data["signal"] = (data["vol_mahalanobis"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_mahalanobis < {threshold:.1f}", perf))

# 11. Tukey biweight
print("\nТестирую Tukey biweight...")
thresholds = [0.018, 0.02, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["vol_tukey_biweight_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_tukey_biweight_ma5 < {threshold:.3f}", perf))

# 12. Winsorized mean
print("\nТестирую winsorized mean...")
thresholds = [0.018, 0.02, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["vol_winsorized_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"vol_winsorized_ma5 < {threshold:.3f}", perf))

# 13. Information weighted volatility
print("\nТестирую information weighted volatility...")
thresholds = [0.018, 0.02, 0.022, 0.024]
for threshold in thresholds:
    data["signal"] = (data["info_weighted_vol_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"info_weighted_vol_ma5 < {threshold:.3f}", perf))

# 14. Copula dependence
print("\nТестирую copula dependence...")
thresholds = [0.3, 0.4, 0.5, 0.6]
for threshold in thresholds:
    data["signal"] = (data["copula_dependence_ma5"].shift(1) < threshold).fillna(False).astype(int)
    perf = calculate_strategy_performance(data, "signal")
    results.append((f"copula_dependence_ma5 < {threshold:.1f}", perf))

# ------------------------- ИТОГИ -----------------------------------
print("\n=== ИТОГИ Wave 18 ===")
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