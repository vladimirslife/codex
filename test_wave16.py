#!/usr/bin/env python3
"""
Wave 16 grid search — исследуем объёмные показатели Chaikin Money Flow (CMF) и Money Flow Index (MFI).
При наличии столбца volume. Если volume отсутствует — пропускаем.
"""

import os, pandas as pd, numpy as np
from datetime import datetime
from typing import List, Tuple, Callable

CSV = os.environ.get("QQQ_CSV", "4 - QQQ.csv")
START = "2006-01-01"
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF/252
TOP = 20

df = pd.read_csv(CSV)
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= START].copy()
df['next_open'] = df['open'].shift(-1)
df['next_ov_ret'] = df['next_open']/df['close'] - 1

df.dropna(inplace=True)

have_vol = 'volume' in df.columns

Rule = Tuple[str, Callable[[pd.DataFrame], pd.Series]]
rules: List[Rule] = []

if have_vol:
    high, low, close, vol = df['high'], df['low'], df['close'], df['volume']
    # CMF
    CMF_N = [10, 20]
    for n in CMF_N:
        mf = ((close - low) - (high - close)) / (high - low + 1e-9) * vol
        cmf = mf.rolling(n).sum() / vol.rolling(n).sum()
        for thr in [0.1, 0.2, -0.1, -0.2]:
            if thr > 0:
                rules.append((f"cmf{n}_gt_{thr}", lambda d, c=cmf, t=thr: (c > t)))
            else:
                rules.append((f"cmf{n}_lt_{thr}", lambda d, c=cmf, t=thr: (c < t)))
    # MFI
    tp = (high + low + close)/3
    raw_mf = tp * vol
    pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
    MFI_N = [14, 20]
    for n in MFI_N:
        mfi = 100 - 100 / (1 + (pos_mf.rolling(n).sum() / (neg_mf.rolling(n).sum()+1e-9)))
        for thr in [20,30,70,80]:
            if thr < 50:
                rules.append((f"mfi{n}_lt_{thr}", lambda d, m=mfi, t=thr: (m < t)))
            else:
                rules.append((f"mfi{n}_gt_{thr}", lambda d, m=mfi, t=thr: (m > t)))
else:
    print("Volume column not found — skipping CMF/MFI rules.")

# Ensure limit
rules = rules[:200]
print(f"Wave16 generated {len(rules)} volume-based rules.")

# Evaluation
def eval_sig(sig):
    strat = sig.shift(0)*df['next_ov_ret']
    strat = strat.dropna()
    excess = strat - DAILY_RF
    mean = excess.mean()*252
    std = excess.std()*np.sqrt(252)
    sharpe = mean/std if std>0 else 0
    trades = int(sig.sum())
    return sharpe, trades

results = []
for name, func in rules:
    sig = func(df).astype(int)
    sh, tr = eval_sig(sig)
    results.append((name, sh, tr))
results.sort(key=lambda x: x[1], reverse=True)
print("Top results Wave16:")
for i, (n, sh, tr) in enumerate(results[:TOP],1):
    print(f"{i:2d}. {n:25s} | Sharpe {sh:.3f} | Trades {tr}")

a = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
with open('history.log','a') as f:
    for n,sh,tr in results[:3]:
        f.write(f"[{a}] Wave16 | {n} | Sharpe={sh:.3f} | Trades={tr}\n")
print('Logged top3.')