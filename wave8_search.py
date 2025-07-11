# wave8_search.py
"""Wave 8 – Bayesian Optimization для одного условия.
Индикатор: comp = α·max3 + (1−α)·range3  (α∈[0.6,1.3])
Порог thr ∈ [0.0015, 0.0045]. Условие: comp < thr.
Оптимизируем Sharpe (max). Останов при Sharpe>=1.4 и trades>2500.
Используется scikit-optimize (Gaussian Process)."""
import warnings, math
warnings.filterwarnings("ignore")
try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError:
    print("scikit-optimize не установлен. Установите: pip install --break-system-packages scikit-optimize")
    exit()

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
FILES = {"QQQ": "4 - QQQ.csv", "SPY": "4 - SPY.csv", "XLK": "4 - XLK.csv"}

# ---------- load ----------
raw = {}
for sym, path in FILES.items():
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
    else:
        df.rename(columns={"date": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= "2006-01-01"].copy()
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ret"] = df["open"].shift(-1) / df["close"] - 1
    raw[sym] = df[["Date", "ret"]]

base = raw["QQQ"].rename(columns={"ret": "qqq"})
for sym in ("SPY", "XLK"):
    base = base.merge(raw[sym].rename(columns={"ret": sym.lower()}), on="Date")
prev = base[["qqq", "spy", "xlk"]].shift(1)
max3 = prev.max(axis=1).values.astype(float)
range3 = (max3 - prev.min(axis=1).values.astype(float))
ret_arr = base["qqq"].values.astype(float)
mask = np.isfinite(max3) & np.isfinite(range3) & np.isfinite(ret_arr)

# ---------- objective ----------

def evaluate(params):
    """Returns negative Sharpe for minimization"""
    alpha, thr = params
    comp = alpha * max3 + (1 - alpha) * range3
    signal = (comp < thr) & mask
    trades = int(signal.sum())
    if trades < 2500:
        return 10  # penalize
    strat = signal.astype(float) * ret_arr
    std = strat.std()
    if std == 0:
        return 10
    sharpe = (strat.mean() - DAILY_RF) * 252 / (std * math.sqrt(252))
    # we need negative for gp_minimize
    return -sharpe

space = [Real(0.6, 1.3, name="alpha"), Real(0.0015, 0.0045, name="thr")]

res = gp_minimize(evaluate, space, n_calls=120, n_initial_points=30, random_state=2025)

best_alpha, best_thr = res.x
best_sharpe = -res.fun
# compute trades
comp_best = best_alpha * max3 + (1 - best_alpha) * range3
signal_best = (comp_best < best_thr) & mask
best_trades = int(signal_best.sum())

print("=== Wave 8 – Bayesian result ===")
print(f"Sharpe={best_sharpe:.4f}, trades={best_trades}, alpha={best_alpha:.4f}, thr={best_thr:.6f}")

# log
hist = Path("history.log")
line = f"{datetime.utcnow().isoformat()}\tWave8\tSharpe={best_sharpe:.4f},N={best_trades},alpha={best_alpha:.4f},thr={best_thr:.6f}\n"
if hist.exists():
    hist.write_text(hist.read_text() + line, encoding="utf-8")
else:
    hist.write_text(line, encoding="utf-8")