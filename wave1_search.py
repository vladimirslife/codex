# wave1_search.py
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------- CONSTANTS ----------------------
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
DATA_FILE = "4 - QQQ.csv"
STEP = 0.0001  # шаг сетки
LOW, HIGH = -0.02, 0.02  # диапазон порога

# ---------------------- LOAD DATA ----------------------
df = pd.read_csv(DATA_FILE)
# стандартизация колонок и вычисление ночной доходности

df.rename(columns=lambda c: c.lower(), inplace=True)
if "time" in df.columns:
    df.rename(columns={"date": "Date", "time": "Date"}, inplace=True)
else:
    df.rename(columns={"date": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df = (
    df[df["Date"] >= pd.Timestamp("2006-01-01")]
    .sort_values("Date")
    .reset_index(drop=True)
)

# close t -> open t+1

df["next_open"] = df["open"].shift(-1)
df["next_ov_return"] = df["next_open"] / df["close"] - 1
prev_ov = df["next_ov_return"].shift(1)

# ---------------------- GRID SEARCH --------------------
best = []  # хранит кортежи (Sharpe, trades, threshold)
thresholds = np.arange(LOW, HIGH + STEP, STEP)
for thr in thresholds:
    signal = (prev_ov < thr).astype(int)
    trades = int(signal.sum())
    if trades < 2500:
        continue
    strat_ret = signal * df["next_ov_return"]
    excess = strat_ret - DAILY_RF
    mean_excess_ann = excess.mean() * 252
    std_excess_ann = excess.std() * np.sqrt(252)
    if std_excess_ann == 0:
        continue
    sharpe = mean_excess_ann / std_excess_ann
    best.append((sharpe, trades, thr))

best.sort(reverse=True, key=lambda x: x[0])

top3 = best[:3]

# ---------------------- OUTPUT -------------------------
print("=== Wave 1: Одно условие — prev_overnight_return < threshold ===")
for i, (s, n, t) in enumerate(top3, 1):
    print(f"#{i}: Sharpe={s:.4f}, trades={n}, threshold={t:.6f}")

# ---------------------- HISTORY LOG --------------------
history_line = (
    f"{datetime.utcnow().isoformat()}\tWave1\t" +
    " | ".join([f"Sharpe={s:.4f}, N={n}, thr={t:.6f}" for s, n, t in top3]) +
    "\n"
)
with open("history.log", "a", encoding="utf-8") as f:
    f.write(history_line)