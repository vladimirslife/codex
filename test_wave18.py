#!/usr/bin/env python3
"""
Wave 18 grid search – Aroon oscillator and simple gap/high-low breakout conditions.
Single-condition rules, ≤150 variants.
Target (updated): Sharpe > 1.2 and trades > 2500.
"""
import os, pandas as pd, numpy as np
from datetime import datetime

CSV=os.environ.get("QQQ_CSV","4 - QQQ.csv")
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1

df.dropna(inplace=True)

close,high,low,open_=df['close'],df['high'],df['low'],df['open']

Rule=list[tuple[str,pd.Series]]
rules:Rule=[]

# Aroon
WINDOW=[14,20,25]
THR=[50,70,80]
for n in WINDOW:
    aroon_up=100*(n-high.rolling(n).apply(lambda x: n-1-np.argmax(x.iloc[::-1])))/n
    aroon_down=100*(n-low.rolling(n).apply(lambda x: n-1-np.argmin(x.iloc[::-1])))/n
    for t in THR:
        rules.append((f"aroon_up{n}_gt_{t}", (aroon_up>t)))
        rules.append((f"aroon_down{n}_gt_{t}", (aroon_down>t)))

# Price close above previous high / below previous low
prev_high=high.shift(1); prev_low=low.shift(1)
rules.append(("close_gt_prev_high", close>prev_high))
rules.append(("close_lt_prev_low", close<prev_low))

# Gap up/down threshold relative prev close
prev_close=close.shift(1)
gap=(open_-prev_close)/prev_close
for thr in [0.0,0.002,0.005,0.01]:
    rules.append((f"gap_up_gt_{thr}", gap>thr))
    rules.append((f"gap_down_lt_-{thr}", gap<-thr))

# limit
rules=rules[:150]
print("Wave18 rules:",len(rules))

ANNUAL_RF=0.02; DAILY_RF=ANNUAL_RF/252

def evaluate(mask:pd.Series):
    strat=mask.astype(int).shift(0)*df['next_ov_ret']
    strat=strat.dropna(); excess=strat-DAILY_RF
    mean=excess.mean()*252; sd=excess.std()*np.sqrt(252)
    sharpe=mean/sd if sd>0 else 0; trades=int(mask.sum())
    return sharpe,trades

best=None
for name,mask in rules:
    sh,tr=evaluate(mask)
    print(f"{name:25s} | Sharpe {sh:.3f} | Trades {tr}")
    if best is None or sh>best[1]: best=(name,sh,tr)
print("Best:",best)