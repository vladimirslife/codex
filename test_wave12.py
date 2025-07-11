import pandas as pd, numpy as np, yfinance as yf, math, os
from datetime import datetime

TICKER = "SPY"
START = "1990-01-01"
END = datetime.today().strftime("%Y-%m-%d")
RF = 0.02 / 252
MAX = 1000

# ---------- Data ----------

def load(ticker):
    fn = f"{ticker}_{START}_{END}.csv"
    if os.path.exists(fn):
        df = pd.read_csv(fn, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce")
    df = yf.download(ticker, start=START, end=END, progress=False)
    df.to_csv(fn)
    return df.apply(pd.to_numeric, errors="coerce")

# ---------- Indicators ----------

def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()

def roc(x, n):
    return x.pct_change(n)

def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def zscore(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

# ---------- Sharpe ----------

def sharpe(r):
    if r.empty: return 0.0
    sd = r.std(ddof=0)
    if sd == 0 or np.isclose(sd, 0): return 0.0
    return math.sqrt(252) * (r - RF).mean() / sd

# ---------- Evaluate ----------

def eval_cond(df, name, p):
    close, open_next = df["Close"], df["Open"].shift(-1)
    daily = (open_next - close) / close
    sig = pd.Series(False, index=df.index)

    if name == "VOL_DEV_RATIO":
        long = p["long"]; n = p["atr"]; k = p["k"]
        dev = close/ema(close, long) - 1
        vol = atr(df, n) / close
        sig = (dev/vol.replace(0, np.nan)) > k
    elif name == "PRICE_Z_EMA":
        long = p["long"]; w = p["win"]; k = p["k"]
        dev = close/ema(close, long) - 1
        sig = zscore(dev, w) > k
    elif name == "MOM_Z_VOL":
        n = p["period"]; w = p["win"]; k = p["k"]
        mom_vol = roc(close, n) / (atr(df, n)/close)
        sig = zscore(mom_vol, w) > k
    elif name == "KELTNER_BREAK":
        mult = p["mult"]
        upper = ema(close, 20) + mult * atr(df, 20)
        sig = close > upper
    else:
        raise ValueError

    strat = daily.copy(); strat[~sig] = 0.0
    trades = int(sig.sum())
    sr = sharpe(strat.dropna())
    tot = (1+strat.fillna(0)).prod()-1
    return {"condition": name, "params": p, "sharpe": sr, "trades": trades, "total_return": tot}

# ---------- Grid ----------
conds = []
for long in (150,200):
    for atr_n in (14,20):
        for k in (1.3,1.5,1.7,2.0):
            conds.append(("VOL_DEV_RATIO", {"long": long, "atr": atr_n, "k": k}))
for long in (150,200):
    for w in (100,200):
        for k in (1.0,1.5,2.0):
            conds.append(("PRICE_Z_EMA", {"long": long, "win": w, "k": k}))
for n in (5,10,20):
    for w in (50,100):
        for k in (1.0,1.5,2.0):
            conds.append(("MOM_Z_VOL", {"period": n, "win": w, "k": k}))
for mult in (1.5,2.0,2.5,3.0):
    conds.append(("KELTNER_BREAK", {"mult": mult}))

conds = conds[:MAX]
print(f"Wave12: total {len(conds)} conditions")

df = load(TICKER)
results = [eval_cond(df, n, p) for n, p in conds]
res = pd.DataFrame(results)
res = res.sort_values(["sharpe", "trades"], ascending=[False, False])
print("\nTop 10 Wave12:")
print(res.head(10).to_string(index=False))

with open("history.log", "a") as f:
    for _, r in res.head(3).iterrows():
        f.write(f"Wave12 | Condition: {r.condition} | Params: {r.params} | Sharpe: {r.sharpe:.3f} | Trades: {r.trades} | TotalRet: {r.total_return:.2%}\n")
print("Top 3 Wave12 appended to history.log")