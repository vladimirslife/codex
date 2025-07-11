#!/usr/bin/env python3
"""
Wave 19 – тест True Strength Index (TSI) и Chande Momentum Oscillator (CMO).
Single-condition rules, <= 100.
Target: Sharpe > 1.2, trades > 2500.
"""
import os, pandas as pd, numpy as np
from datetime import datetime

CSV=os.environ.get('QQQ_CSV','4 - QQQ.csv')
df=pd.read_csv(CSV)
df.columns=df.columns.str.lower(); df['date']=pd.to_datetime(df['date'])
df=df[df['date']>='2006-01-01'].copy()
df['next_open']=df['open'].shift(-1)
df['next_ov_ret']=df['next_open']/df['close']-1
df.dropna(inplace=True)

close=df['close']

rules=[]

# --- TSI ---
# TSI = EMA(EMA(momentum, r), s) / EMA(EMA(|momentum|, r), s) *100
R,S=25,13
mom=close.diff()
ema1_m=mom.ewm(span=R,adjust=False).mean(); ema2_m=ema1_m.ewm(span=S,adjust=False).mean()
ema1_a=mom.abs().ewm(span=R,adjust=False).mean(); ema2_a=ema1_a.ewm(span=S,adjust=False).mean()
tsi=100*ema2_m/ema2_a
for thr in [-5,0,5]:
    if thr>=0:
        rules.append((f"tsi_gt_{thr}",(tsi>thr)))
    else:
        rules.append((f"tsi_lt_{thr}",(tsi<thr)))

# --- CMO ---
N=14
up=close.diff().clip(lower=0)
dn=-close.diff().clip(upper=0)
up_sum=up.rolling(N).sum(); dn_sum=dn.rolling(N).sum()
cmo=100*(up_sum-dn_sum)/(up_sum+dn_sum+1e-9)
for thr in [-50,-30,30,50]:
    if thr<0:
        rules.append((f"cmo_lt_{thr}",(cmo<thr)))
    else:
        rules.append((f"cmo_gt_{thr}",(cmo>thr)))

rules=rules[:100]
print('Wave19 rules',len(rules))

DAILY_RF=0.02/252

def eval(mask):
    strat=mask.astype(int)*df['next_ov_ret']; strat=strat.dropna()
    excess=strat-DAILY_RF
    mean=excess.mean()*252; sd=excess.std()*np.sqrt(252)
    sh=mean/sd if sd>0 else 0; trades=int(mask.sum())
    return sh,trades

best=None
for name,mask in rules:
    sh,tr=eval(mask)
    print(f"{name:15s} | Sharpe {sh:.3f} | trades {tr}")
    if best is None or sh>best[1]: best=(name,sh,tr)
print('Best',best)