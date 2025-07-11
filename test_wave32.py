#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave 32: adaptive EMA depending on volatility regime (quantiles)
Condition: (gap + alpha*d_gap)/(ema_regime + gamma) < thr
"""
import pandas as pd, numpy as np, os, datetime, math
FILE='4 - QQQ.csv'
ANNUAL_RF=0.02
DAILY_RF=ANNUAL_RF/252
if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

df=pd.read_csv(FILE)
df.columns=[c.lower() for c in df.columns]
df['date']=pd.to_datetime(df['date'])
df=df[df['date']>=pd.Timestamp('2006-01-01')].sort_values('date').reset_index(drop=True)

prev_close=df['close'].shift(1)
gap=df['open']/prev_close-1
prev_gap=gap.shift(1)
d_gap=gap-prev_gap
next_ov=df['open'].shift(-1)/df['close']-1

# volatility proxy: rolling 20d std of gap
vol=gap.rolling(20).std(ddof=0)
low_q, high_q = vol.quantile([0.33,0.66])

# two spans
span_low=5
span_high=15
ema_low=gap.abs().ewm(span=span_low,adjust=False).mean().shift(1)
ema_high=gap.abs().ewm(span=span_high,adjust=False).mean().shift(1)

ema_regime = np.where(vol<=low_q, ema_low, np.where(vol>=high_q, ema_high*1.5, ema_low*0.75+ema_high*0.25))
ema_regime = pd.Series(ema_regime, index=df.index).replace(0,np.nan)

alpha_vals=[-0.5,0,0.5]
thr_vals=np.round(np.arange(0.002,0.0051,0.0003),4)
gamma=1e-4

best=None
for alpha in alpha_vals:
    metric=(gap + alpha*d_gap)/(ema_regime + gamma)
    for thr in thr_vals:
        sig=(metric<thr).astype(int)
        trades=int(sig.sum())
        if trades<3000:
            continue
        excess=sig*next_ov-DAILY_RF
        sd=excess.std()
        if sd==0:
            continue
        sr=(excess.mean()/sd)*math.sqrt(252)
        if best is None or sr>best[0]:
            best=(sr,trades,alpha,thr)

if best:
    sr,tr,alpha,thr=best
    print(f"Wave32 BEST Sharpe={sr:.4f}, Trades={tr}, Cond: alpha={alpha}, thr={thr}")
    with open('history.log','a',encoding='utf-8') as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} | Wave 32 | best | Sharpe={sr:.4f} | Trades={tr} | alpha={alpha} thr={thr}\n")
else:
    print('Wave32: no combo >3000 trades')