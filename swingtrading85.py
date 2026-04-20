"""
Smart Investing – Professional Algorithmic Trading Platform
===========================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
import random
import requests
import itertools
from datetime import datetime, timedelta
import pytz

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

IST  = pytz.timezone("Asia/Kolkata")

PRESET_TICKERS = {
    "Nifty 50":  "^NSEI",
    "BankNifty": "^NSEBANK",
    "Sensex":    "^BSESN",
    "BTC-USD":   "BTC-USD",
    "ETH-USD":   "ETH-USD",
    "Gold":      "GC=F",
    "Silver":    "SI=F",
    "Custom":    "",
}

TIMEFRAME_PERIODS = {
    "1m":  ["1d","5d","7d"],
    "5m":  ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo"],
    "1h":  ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":  ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk": ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}

PERIOD_ORDER = ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]
WARMUP_CANDLES = 200

STRATEGIES = [
    "EMA Crossover",
    "Anticipatory EMA Crossover",
    "Elliott Wave (Auto)",
    "Simple Buy",
    "Simple Sell",
]

SL_TYPES = [
    "Custom Points",
    "ATR Based",
    "Risk Reward Based",
    "Trailing SL",
    "Trailing SL – Swing Low/High",
    "Trailing SL – Candle Low/High",
    "EMA Reverse Crossover",
    "Auto SL",
    "Volatility Based",
    "Nearest Support/Resistance",
]

TARGET_TYPES = [
    "Custom Points",
    "ATR Based",
    "Risk Reward Based",
    "Trailing Target (Display Only)",
    "Trailing Target – Swing Low/High",
    "Trailing Target – Candle Low/High",
    "EMA Crossover Exit",
    "Auto Target",
    "Volatility Based",
    "Nearest Support/Resistance",
]

TF_MINUTES = {
    "1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080,
}

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE STATE
# ─────────────────────────────────────────────────────────────────────────────
_LOCK = threading.RLock()
_TS: dict = {}

def ts_get(k, default=None):
    with _LOCK: return _TS.get(k, default)

def ts_set(k, v):
    with _LOCK: _TS[k] = v

def ts_append(k, v):
    with _LOCK:
        if k not in _TS: _TS[k] = []
        _TS[k].append(v)

def ts_clear(k):
    with _LOCK: _TS[k] = []

# ─────────────────────────────────────────────────────────────────────────────
# SESSION INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    for k, v in [
        ("backtest_results", None),
        ("backtest_df",      None),
        ("opt_results",      None),
        ("live_thread",      None),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
code,.log-entry{font-family:'JetBrains Mono',monospace;}
.stApp{background:#050A14;}
section[data-testid="stSidebar"]>div{background:#060D1C;border-right:1px solid #0f2040;}
.stTabs [data-baseweb="tab-list"]{background:#0A1628;border-radius:12px;padding:4px;gap:4px;border:1px solid #0f2040;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#5a7fa8;font-weight:600;font-size:0.85rem;padding:8px 18px;transition:all .2s;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0f3460,#1a4a80)!important;color:#00E5CC!important;box-shadow:0 2px 12px rgba(0,229,204,.2);}
.m-card{background:linear-gradient(135deg,#0A1628,#0d1f3c);border:1px solid #0f3060;border-radius:12px;padding:14px 16px;text-align:center;margin-bottom:6px;}
.mc-label{color:#5a7fa8;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin:0;}
.mc-val{color:#e8f4ff;font-size:1.25rem;font-weight:700;margin:4px 0 0;font-family:'JetBrains Mono',monospace;}
.ltp-banner{background:linear-gradient(90deg,#050A14 0%,#091527 40%,#050A14 100%);border:1px solid #0f3060;border-radius:14px;padding:12px 24px;display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}
.ltp-val{color:#00E5CC;font-size:2rem;font-weight:800;font-family:'JetBrains Mono',monospace;}
.ticker-name{color:#5a7fa8;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;}
.ltp-time{color:#3a5a78;font-size:.75rem;}
.pos-card{background:linear-gradient(135deg,#0A1628,#0d2040);border-radius:14px;padding:18px;border:1px solid;}
.buy-card{border-color:#00c896;box-shadow:0 0 20px rgba(0,200,150,.08);}
.sell-card{border-color:#ff4d6d;box-shadow:0 0 20px rgba(255,77,109,.08);}
.pos-label{font-size:.7rem;color:#5a7fa8;text-transform:uppercase;letter-spacing:.06em;}
.pos-val{font-size:1.05rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#e8f4ff;}
.wave-card{background:linear-gradient(135deg,#0A1628,#0c1a38);border:1px solid #1a3060;border-radius:12px;padding:16px;margin-bottom:10px;}
.wave-status{color:#4FC3F7;font-size:.85rem;font-weight:600;margin-bottom:8px;}
.wave-row{display:flex;justify-content:space-between;align-items:center;margin:4px 0;font-size:.82rem;}
.wave-key{color:#5a7fa8;}
.wave-val{color:#e8f4ff;font-family:'JetBrains Mono',monospace;font-weight:600;}
.log-box{background:#030810;border:1px solid #0f2040;border-radius:10px;padding:12px;max-height:220px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:.78rem;}
.log-box::-webkit-scrollbar{width:4px;}
.log-box::-webkit-scrollbar-track{background:#030810;}
.log-box::-webkit-scrollbar-thumb{background:#1a3060;border-radius:2px;}
.violation-box{background:#1a0a0a;border:1px solid #8B2500;border-radius:10px;padding:12px 16px;margin:10px 0;color:#ff8c69;font-size:.85rem;}
.best-result{background:linear-gradient(135deg,#051a10,#082a18);border:1px solid #1a7a50;border-radius:12px;padding:18px;margin:12px 0;}
.status-live{display:inline-block;width:8px;height:8px;background:#00E5CC;border-radius:50%;margin-right:6px;animation:pulse-g 1.5s infinite;}
.status-off{display:inline-block;width:8px;height:8px;background:#3a5a78;border-radius:50%;margin-right:6px;}
@keyframes pulse-g{0%,100%{box-shadow:0 0 0 0 rgba(0,229,204,.4);}50%{box-shadow:0 0 0 6px rgba(0,229,204,0);}}
.cfg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px;margin:10px 0;}
.cfg-item{background:#080F20;border:1px solid #0f2040;border-radius:8px;padding:8px 12px;}
.cfg-key{color:#5a7fa8;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;}
.cfg-val{color:#00E5CC;font-size:.85rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
div[data-testid="stDecoration"]{display:none;}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(ticker, timeframe, period, warmup=True):
    try:
        warm_map = {"1m":"5d","5m":"5d","15m":"7d","1h":"1mo","1d":"1y","1wk":"2y"}
        warm_p   = warm_map.get(timeframe,"1mo")
        try:    ri = PERIOD_ORDER.index(period)
        except: ri = 0
        try:    wi = PERIOD_ORDER.index(warm_p)
        except: wi = 0
        fetch_p = PERIOD_ORDER[max(ri, wi)] if warmup else period

        raw = yf.download(ticker, period=fetch_p, interval=timeframe,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty: return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC").tz_convert(IST)
        else:
            raw.index = raw.index.tz_convert(IST)
        raw = raw.sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        raw = raw.dropna(subset=["Close"])
        return raw
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def calc_ema(series, period):
    """TradingView-accurate EMA."""
    return series.ewm(span=period, adjust=False, min_periods=1).mean()

def calc_atr(df, period=14):
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl,hpc,lpc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()

def add_ema_cols(df, fast, slow):
    df = df.copy()
    df["ema_fast"] = calc_ema(df["Close"], fast)
    df["ema_slow"] = calc_ema(df["Close"], slow)
    df["atr"]      = calc_atr(df)
    return df

def find_swing_pivots(df, left=5, right=5):
    highs, lows = [], []
    h = df["High"].values; l = df["Low"].values; n = len(df)
    for i in range(left, n-right):
        if all(h[i]>=h[i-j] for j in range(1,left+1)) and all(h[i]>=h[i+j] for j in range(1,right+1)):
            highs.append((i, df.index[i], h[i]))
        if all(l[i]<=l[i-j] for j in range(1,left+1)) and all(l[i]<=l[i+j] for j in range(1,right+1)):
            lows.append((i, df.index[i], l[i]))
    return highs, lows

def nearest_sr(df, lookback=60):
    recent = df.tail(lookback)
    highs, lows = find_swing_pivots(recent, left=3, right=3)
    price = float(df["Close"].iloc[-1])
    res = sorted([h[2] for h in highs if h[2]>price])
    sup = sorted([l[2] for l in lows  if l[2]<price], reverse=True)
    return (float(sup[0]) if sup else price*0.98), (float(res[0]) if res else price*1.02)

# ─────────────────────────────────────────────────────────────────────────────
# ELLIOTT WAVE
# ─────────────────────────────────────────────────────────────────────────────
def _zigzag(prices, pct=0.03):
    pivots=[]; direction=None
    pi,pv = 0,prices[0]; ti,tv = 0,prices[0]
    for i in range(1,len(prices)):
        p=prices[i]
        if direction is None:
            if p>pv*(1+pct):   direction="up";  pivots.append((ti,tv,"L")); pi,pv=i,p
            elif p<tv*(1-pct): direction="down"; pivots.append((pi,pv,"H")); ti,tv=i,p
        elif direction=="up":
            if p>pv:          pi,pv=i,p
            elif p<pv*(1-pct): pivots.append((pi,pv,"H")); direction="down"; ti,tv=i,p
        else:
            if p<tv:          ti,tv=i,p
            elif p>tv*(1+pct): pivots.append((ti,tv,"L")); direction="up"; pi,pv=i,p
    if direction=="up":   pivots.append((pi,pv,"H"))
    elif direction=="down": pivots.append((ti,tv,"L"))
    return pivots

def detect_elliott_waves(df):
    res={"pivots":[],"pattern":None,"wave_status":"Analyzing…","current_wave":None,
         "signal":None,"next_target":None,"completed_waves":[],"wave_projections":{},
         "auto_sl":None,"auto_target":None}
    if len(df)<50:
        res["wave_status"]="Insufficient data"; return res
    closes=df["Close"].values
    for pct in [0.02,0.03,0.05,0.08]:
        pivots=_zigzag(closes,pct); res["pivots"]=pivots
        if len(pivots)<6: continue
        for start in range(max(0,len(pivots)-12), len(pivots)-5):
            seg=pivots[start:start+6]; types=[p[2] for p in seg]; vals=[p[1] for p in seg]
            atr=float(calc_atr(df).iloc[-1])
            if types==["L","H","L","H","L","H"]:
                w0,w1,w2,w3,w4,w5=vals
                if w2>w0 and w3>w1 and w4>w2:
                    w1l=w1-w0
                    abc_a=w5-(w5-w4)*0.618; abc_b=abc_a+(w5-abc_a)*0.618; abc_c=abc_a-(abc_b-abc_a)
                    res.update({"pattern":"bullish_impulse",
                        "wave_status":"✅ 5-Wave Bullish Impulse complete → ABC Correction",
                        "current_wave":"Wave A (Correction – Down)","completed_waves":["W1","W2","W3","W4","W5"],
                        "signal":"sell","next_target":round(abc_a,2),
                        "wave_projections":{"Wave A":round(abc_a,2),"Wave B":round(abc_b,2),"Wave C":round(abc_c,2)},
                        "auto_sl":round(w5+atr*1.5,2),"auto_target":round(abc_a,2)}); return res
            if types==["H","L","H","L","H","L"]:
                w0,w1,w2,w3,w4,w5=vals
                if w2<w0 and w3<w1 and w4<w2:
                    abc_a=w5+(w4-w5)*0.618; abc_b=abc_a-(abc_a-w5)*0.618; abc_c=abc_a+(abc_a-abc_b)
                    res.update({"pattern":"bearish_impulse",
                        "wave_status":"✅ 5-Wave Bearish Impulse complete → ABC Bounce",
                        "current_wave":"Wave A (Correction – Up)","completed_waves":["W1","W2","W3","W4","W5"],
                        "signal":"buy","next_target":round(abc_a,2),
                        "wave_projections":{"Wave A":round(abc_a,2),"Wave B":round(abc_b,2),"Wave C":round(abc_c,2)},
                        "auto_sl":round(w5-atr*1.5,2),"auto_target":round(abc_a,2)}); return res
        if len(pivots)>=4:
            seg=pivots[-4:]; types=[p[2] for p in seg]; vals=[p[1] for p in seg]
            atr=float(calc_atr(df).iloc[-1])
            if types==["L","H","L","H"] and vals[1]>vals[0] and vals[3]>vals[1]:
                w1l=vals[1]-vals[0]; w3p=vals[2]+w1l*1.618
                res.update({"wave_status":"📊 Wave 3 forming (Bullish)","current_wave":"Wave 3 (In Progress)",
                    "completed_waves":["W1","W2"],"signal":"buy","next_target":round(w3p,2),
                    "wave_projections":{"W3 (1.618)":round(vals[2]+w1l*1.618,2),"W3 (2.618)":round(vals[2]+w1l*2.618,2)},
                    "auto_sl":round(vals[2]-atr*1.5,2),"auto_target":round(w3p,2)}); return res
            if types==["H","L","H","L"] and vals[1]<vals[0] and vals[3]<vals[1]:
                w1l=vals[0]-vals[1]; w3p=vals[2]-w1l*1.618
                res.update({"wave_status":"📊 Wave 3 forming (Bearish)","current_wave":"Wave 3 (In Progress)",
                    "completed_waves":["W1","W2"],"signal":"sell","next_target":round(w3p,2),
                    "wave_projections":{"W3 (1.618)":round(vals[2]-w1l*1.618,2),"W3 (2.618)":round(vals[2]-w1l*2.618,2)},
                    "auto_sl":round(vals[2]+atr*1.5,2),"auto_target":round(w3p,2)}); return res
    res["wave_status"]="⏳ No clear pattern yet – watching pivots…"; return res

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def gen_ema_signals(df, fast=9, slow=15, xover_type="Simple Crossover",
                    candle_size=10.0, min_angle=0.0, check_angle=False):
    df = add_ema_cols(df, fast, slow)
    sigs = pd.Series("", index=df.index)
    for i in range(1, len(df)):
        ef_n,ef_p = df["ema_fast"].iloc[i], df["ema_fast"].iloc[i-1]
        es_n,es_p = df["ema_slow"].iloc[i], df["ema_slow"].iloc[i-1]
        bull = ef_p<=es_p and ef_n>es_n
        bear = ef_p>=es_p and ef_n<es_n
        if not (bull or bear): continue
        if check_angle and min_angle>0:
            angle=abs((ef_n-es_n)-(ef_p-es_p))
            if angle<min_angle: continue
        body=abs(df["Close"].iloc[i]-df["Open"].iloc[i])
        if xover_type=="Custom Candle Size" and body<candle_size: continue
        if xover_type=="ATR Based Candle Size" and body<df["atr"].iloc[i]: continue
        sigs.iloc[i]="buy" if bull else "sell"
    df["signal"]=sigs; return df

def gen_anticipatory_signals(df, fast=9, slow=15):
    df = add_ema_cols(df, fast, slow)
    sigs = pd.Series("", index=df.index)
    for i in range(3,len(df)):
        ef=df["ema_fast"].iloc[i];   es=df["ema_slow"].iloc[i]
        ef1=df["ema_fast"].iloc[i-1]; es1=df["ema_slow"].iloc[i-1]
        ef2=df["ema_fast"].iloc[i-2]; es2=df["ema_slow"].iloc[i-2]
        atr=df["atr"].iloc[i]; cl=df["Close"].iloc[i]; op=df["Open"].iloc[i]
        cl1=df["Close"].iloc[i-1]; op1=df["Open"].iloc[i-1]
        gn=abs(ef-es); gp=abs(ef1-es1); gpp=abs(ef2-es2)
        conv=gn<gp<gpp; rapid=(gpp-gn)>0.05*atr
        body=cl-op; body1=cl1-op1; mid=(ef+es)/2
        sbull=body>0.45*atr and body1>0; sbear=body<-0.45*atr and body1<0
        if ef<es and conv and rapid and sbull and cl>mid: sigs.iloc[i]="buy"
        elif ef>es and conv and rapid and sbear and cl<mid: sigs.iloc[i]="sell"
    df["signal"]=sigs; return df

# ─────────────────────────────────────────────────────────────────────────────
# SL / TARGET
# ─────────────────────────────────────────────────────────────────────────────
def calc_sl(df, idx, trade_type, sl_type, custom_pts=10.0, atr_mult=2.0, rr=2.0):
    if "atr" not in df.columns:
        df = df.copy(); df["atr"]=calc_atr(df)
    price=float(df["Close"].iloc[idx]); atr=float(df["atr"].iloc[idx])
    sign = -1 if trade_type=="buy" else +1
    if sl_type=="Custom Points":   return round(price+sign*custom_pts,2)
    if sl_type=="ATR Based":       return round(price+sign*atr*atr_mult,2)
    if sl_type in ("Risk Reward Based","EMA Reverse Crossover"): return round(price+sign*atr*atr_mult,2)
    if "Trailing" in sl_type or sl_type=="Trailing SL":
        if trade_type=="buy": return round(float(df["Low"].iloc[idx])-atr*0.25,2)
        else:                 return round(float(df["High"].iloc[idx])+atr*0.25,2)
    if sl_type=="Auto SL":
        lb=min(20,idx)
        if trade_type=="buy": return round(float(df["Low"].iloc[max(0,idx-lb):idx+1].min())-atr*0.2,2)
        else:                 return round(float(df["High"].iloc[max(0,idx-lb):idx+1].max())+atr*0.2,2)
    if sl_type=="Volatility Based":
        std=float(df["Close"].iloc[max(0,idx-20):idx+1].std())
        return round(price+sign*std*2.0,2)
    if sl_type=="Nearest Support/Resistance":
        sup,res=nearest_sr(df.iloc[:idx+1])
        if trade_type=="buy": return round(sup-atr*0.1,2)
        else:                 return round(res+atr*0.1,2)
    return round(price+sign*custom_pts,2)

def calc_target(df, idx, trade_type, target_type, entry, sl,
                custom_pts=20.0, atr_mult=3.0, rr=2.0):
    if "atr" not in df.columns:
        df = df.copy(); df["atr"]=calc_atr(df)
    atr=float(df["atr"].iloc[idx]); sign=+1 if trade_type=="buy" else -1
    if target_type=="Custom Points":  return round(entry+sign*custom_pts,2)
    if target_type=="ATR Based":      return round(entry+sign*atr*atr_mult,2)
    if target_type=="Risk Reward Based": return round(entry+sign*abs(entry-sl)*rr,2)
    if "Trailing" in target_type:    return round(entry+sign*atr*atr_mult,2)
    if target_type=="EMA Crossover Exit": return round(entry+sign*atr*2.5,2)
    if target_type=="Auto Target":
        sup,res=nearest_sr(df.iloc[:idx+1])
        raw=(res if trade_type=="buy" else sup)
        mn=entry+sign*abs(entry-sl)*1.5
        return round(max(raw,mn) if trade_type=="buy" else min(raw,mn),2)
    if target_type=="Volatility Based":
        std=float(df["Close"].iloc[max(0,idx-20):idx+1].std())
        return round(entry+sign*std*3.0,2)
    if target_type=="Nearest Support/Resistance":
        sup,res=nearest_sr(df.iloc[:idx+1])
        return round(res if trade_type=="buy" else sup,2)
    return round(entry+sign*custom_pts,2)

def update_trailing_sl(cur_sl, ltp, trade_type, sl_type, df, idx, atr_mult=2.0):
    if "atr" not in df.columns:
        df=df.copy(); df["atr"]=calc_atr(df)
    atr=float(df["atr"].iloc[idx])
    if "Swing Low/High" in sl_type:
        lb=min(10,idx)
        if trade_type=="buy":
            ns=float(df["Low"].iloc[max(0,idx-lb):idx+1].min())-atr*0.1
            return max(cur_sl,ns)
        else:
            ns=float(df["High"].iloc[max(0,idx-lb):idx+1].max())+atr*0.1
            return min(cur_sl,ns)
    if "Candle Low/High" in sl_type:
        if trade_type=="buy":
            return max(cur_sl, float(df["Low"].iloc[idx])-atr*0.05)
        else:
            return min(cur_sl, float(df["High"].iloc[idx])+atr*0.05)
    if "Trailing" in sl_type:
        if trade_type=="buy":
            return max(cur_sl, ltp-atr*atr_mult)
        else:
            return min(cur_sl, ltp+atr*atr_mult)
    return cur_sl

# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, cfg):
    strat   = cfg["strategy"]; tf_filt = cfg["trade_filter"]
    sl_type = cfg["sl_type"];  tgt_type= cfg["target_type"]
    qty     = cfg["quantity"];  fast    = cfg["fast_ema"]; slow = cfg["slow_ema"]
    csl     = cfg["custom_sl"]; ctgt    = cfg["custom_target"]
    atr_sl  = cfg.get("atr_mult_sl",2.0); atr_tgt=cfg.get("atr_mult_target",3.0)
    rr      = cfg.get("rr_ratio",2.0)
    xtype   = cfg.get("xover_type","Simple Crossover")
    csz     = cfg.get("candle_size",10.0)
    mang    = cfg.get("min_angle",0.0); cang=cfg.get("check_angle",False)
    immediate = strat in ("Simple Buy","Simple Sell")

    if strat=="EMA Crossover":
        df=gen_ema_signals(df,fast,slow,xtype,csz,mang,cang)
    elif strat=="Anticipatory EMA Crossover":
        df=gen_anticipatory_signals(df,fast,slow)
        if "ema_fast" not in df.columns: df=add_ema_cols(df,fast,slow)
    elif strat=="Elliott Wave (Auto)":
        df=add_ema_cols(df,fast,slow); ew=detect_elliott_waves(df)
        df["signal"]=""
        if ew.get("signal") and len(df)>=2:
            df.iloc[-2, df.columns.get_loc("signal")]=ew["signal"]
    elif strat=="Simple Buy":
        df=add_ema_cols(df,fast,slow); df["signal"]=""
        df.iloc[0, df.columns.get_loc("signal")]="buy"
    elif strat=="Simple Sell":
        df=add_ema_cols(df,fast,slow); df["signal"]=""
        df.iloc[0, df.columns.get_loc("signal")]="sell"
    else:
        df["signal"]=""

    if "atr" not in df.columns: df["atr"]=calc_atr(df)

    trades=[]; violations=[]; position=None

    for i in range(len(df)):
        row=df.iloc[i]
        if position is not None:
            clo=float(row["Low"]); chi=float(row["High"]); cc=float(row["Close"])
            ep=None; er=None; vio=False
            if position["type"]=="buy":
                sl_hit=clo<=position["sl"]; tgt_hit=chi>=position["target"]
                if sl_hit and tgt_hit: ep=position["sl"]; er="SL Hit – VIOLATION(both reachable)"; vio=True
                elif sl_hit:            ep=position["sl"]; er="SL Hit (candle low)"
                elif tgt_hit:           ep=position["target"]; er="Target Hit (candle high)"
            else:
                sl_hit=chi>=position["sl"]; tgt_hit=clo<=position["target"]
                if sl_hit and tgt_hit: ep=position["sl"]; er="SL Hit – VIOLATION(both reachable)"; vio=True
                elif sl_hit:            ep=position["sl"]; er="SL Hit (candle high)"
                elif tgt_hit:           ep=position["target"]; er="Target Hit (candle low)"
            if ep is None and tgt_type=="EMA Crossover Exit" and "ema_fast" in df.columns and i>0:
                ef_n=df["ema_fast"].iloc[i]; es_n=df["ema_slow"].iloc[i]
                ef_p=df["ema_fast"].iloc[i-1]; es_p=df["ema_slow"].iloc[i-1]
                if position["type"]=="buy"  and ef_p>=es_p and ef_n<es_n: ep=cc; er="EMA Reverse Cross"
                if position["type"]=="sell" and ef_p<=es_p and ef_n>es_n: ep=cc; er="EMA Reverse Cross"
            if ep is not None:
                pnl=round((ep-position["entry_price"])*(1 if position["type"]=="buy" else -1)*qty,2)
                rec={"entry_datetime":position["entry_time"],"exit_datetime":df.index[i],
                     "type":position["type"],"entry_price":round(position["entry_price"],2),
                     "exit_price":round(ep,2),"sl":round(position["initial_sl"],2),
                     "target":round(position["target"],2),"candle_high":round(chi,2),
                     "candle_low":round(clo,2),"entry_reason":position["entry_reason"],
                     "exit_reason":er,"pnl":pnl,"quantity":qty,"sl_violation":vio}
                trades.append(rec)
                if vio: violations.append(rec)
                position=None
            else:
                position["sl"]=update_trailing_sl(position["sl"],cc,position["type"],sl_type,df,i,atr_sl)
            continue

        sig=row.get("signal","")
        if not sig: continue
        if tf_filt=="Buy Only"  and sig!="buy":  continue
        if tf_filt=="Sell Only" and sig!="sell": continue

        if immediate:
            eidx=i; ep=float(row["Close"])
        else:
            eidx=i+1
            if eidx>=len(df): continue
            ep=float(df["Open"].iloc[eidx])

        sl=calc_sl(df,eidx,sig,sl_type,csl,atr_sl,rr)
        tgt=calc_target(df,eidx,sig,tgt_type,ep,sl,ctgt,atr_tgt,rr)
        if strat=="Elliott Wave (Auto)":
            ew=detect_elliott_waves(df.iloc[:eidx+1])
            if ew.get("auto_sl"):     sl=ew["auto_sl"]
            if ew.get("auto_target"): tgt=ew["auto_target"]

        position={"entry_time":df.index[eidx],"entry_price":ep,"type":sig,
                  "sl":sl,"initial_sl":sl,"target":tgt,"entry_reason":f"{strat} Signal"}

    if position is not None:
        last=df.iloc[-1]; cc=float(last["Close"])
        pnl=round((cc-position["entry_price"])*(1 if position["type"]=="buy" else -1)*qty,2)
        trades.append({"entry_datetime":position["entry_time"],"exit_datetime":df.index[-1],
                        "type":position["type"],"entry_price":round(position["entry_price"],2),
                        "exit_price":round(cc,2),"sl":round(position["initial_sl"],2),
                        "target":round(position["target"],2),"candle_high":round(float(last["High"]),2),
                        "candle_low":round(float(last["Low"]),2),"entry_reason":position["entry_reason"],
                        "exit_reason":"End of Data","pnl":pnl,"quantity":qty,"sl_violation":False})

    winners  = sum(1 for t in trades if t["pnl"]>0)
    accuracy = (winners/len(trades)*100) if trades else 0.0
    return trades, violations, accuracy, df

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
_DK = dict(template="plotly_dark",paper_bgcolor="#050A14",plot_bgcolor="#080F20",
           font=dict(family="JetBrains Mono",color="#7a9fc0",size=11),
           legend=dict(bgcolor="rgba(5,10,20,.8)",bordercolor="#0f2040",borderwidth=1))

def build_bt_chart(df, trades, fast=9, slow=15):
    df=df.tail(800)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="OHLC",increasing_line_color="#00c896",decreasing_line_color="#ff4d6d",line=dict(width=1)),row=1,col=1)
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_fast"],name=f"EMA{fast}",
            line=dict(color="#FF9F43",width=1.8)),row=1,col=1)
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_slow"],name=f"EMA{slow}",
            line=dict(color="#4ECDC4",width=1.8)),row=1,col=1)
    for t in trades:
        clre="#00e5a0" if t["type"]=="buy" else "#ff4d6d"
        syme="triangle-up" if t["type"]=="buy" else "triangle-down"
        clrx="#00c896" if t["pnl"]>0 else "#ff4d6d"
        fig.add_trace(go.Scatter(x=[t["entry_datetime"]],y=[t["entry_price"]],mode="markers",
            marker=dict(symbol=syme,size=13,color=clre,line=dict(color="white",width=1)),
            showlegend=False,hovertemplate=f"ENTRY {t['type'].upper()}<br>Price:{t['entry_price']}<extra></extra>"),row=1,col=1)
        fig.add_trace(go.Scatter(x=[t["exit_datetime"]],y=[t["exit_price"]],mode="markers",
            marker=dict(symbol="x",size=10,color=clrx,line=dict(color="white",width=1)),
            showlegend=False,hovertemplate=f"EXIT<br>Price:{t['exit_price']}<br>PnL:{t['pnl']}<extra></extra>"),row=1,col=1)
        fig.add_shape(type="line",x0=t["entry_datetime"],x1=t["exit_datetime"],
            y0=t["sl"],y1=t["sl"],line=dict(color="#ff4d6d",width=1,dash="dot"),row=1,col=1)
        fig.add_shape(type="line",x0=t["entry_datetime"],x1=t["exit_datetime"],
            y0=t["target"],y1=t["target"],line=dict(color="#00c896",width=1,dash="dot"),row=1,col=1)
    vc=["#00c896" if float(df["Close"].iloc[i])>=float(df["Open"].iloc[i]) else "#ff4d6d" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,name="Vol"),row=2,col=1)
    fig.update_layout(height=680,xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=30,b=0),**_DK)
    fig.update_xaxes(gridcolor="#0f2040"); fig.update_yaxes(gridcolor="#0f2040")
    return fig

def build_live_chart(df, position, fast=9, slow=15):
    df=df.tail(300)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="OHLC",increasing_line_color="#00c896",decreasing_line_color="#ff4d6d",line=dict(width=1)),row=1,col=1)
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_fast"],name=f"EMA{fast}",
            line=dict(color="#FF9F43",width=2)),row=1,col=1)
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_slow"],name=f"EMA{slow}",
            line=dict(color="#4ECDC4",width=2)),row=1,col=1)
    if position:
        ep=position.get("entry_price",0); sl=position.get("sl",0); tgt=position.get("target",0)
        clr="#00c896" if position["type"]=="buy" else "#ff4d6d"
        sym="triangle-up" if position["type"]=="buy" else "triangle-down"
        fig.add_trace(go.Scatter(x=[position["entry_time"]],y=[ep],mode="markers",
            marker=dict(symbol=sym,size=14,color=clr,line=dict(color="white",width=1)),
            showlegend=False),row=1,col=1)
        for yv,color,label in [(sl,"#ff4d6d",f"SL {sl:.2f}"),(tgt,"#00c896",f"TGT {tgt:.2f}"),(ep,"#FFD93D",f"Entry {ep:.2f}")]:
            fig.add_hline(y=yv,line_color=color,line_dash="dash",
                annotation_text=label,annotation_font_color=color,row=1,col=1)
    vc=["#00c896" if float(df["Close"].iloc[i])>=float(df["Open"].iloc[i]) else "#ff4d6d" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,name="Vol"),row=2,col=1)
    fig.update_layout(height=560,xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=20,b=0),**_DK)
    fig.update_xaxes(gridcolor="#0f2040"); fig.update_yaxes(gridcolor="#0f2040")
    return fig

def build_pnl_chart(trades):
    cumul=[]; run=0.0
    for t in trades: run+=t["pnl"]; cumul.append(run)
    clrs=["#00c896" if v>=0 else "#ff4d6d" for v in cumul]
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=cumul,mode="lines+markers",
        line=dict(color="#4ECDC4",width=2),marker=dict(color=clrs,size=7),
        fill="tozeroy",fillcolor="rgba(0,200,150,.06)",name="Cumulative P&L"))
    fig.add_hline(y=0,line_color="#3a5a78",line_dash="dot")
    fig.update_layout(title="Cumulative P&L",height=240,margin=dict(l=0,r=0,t=35,b=0),**_DK)
    fig.update_xaxes(gridcolor="#0f2040"); fig.update_yaxes(gridcolor="#0f2040")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DHAN BROKER
# ─────────────────────────────────────────────────────────────────────────────
_dhan_hq = None

def init_dhan(client_id, access_token):
    global _dhan_hq
    try:
        from dhanhq import dhanhq
        _dhan_hq = dhanhq(client_id, access_token); return True
    except: return False

def register_ip(client_id, access_token):
    try:
        ip=requests.get("https://api.ipify.org",timeout=5).text.strip()
        r=requests.post("https://api.dhan.co/edis/form",
            headers={"access-token":access_token,"client-id":client_id,"Content-Type":"application/json"},
            json={"ip":ip},timeout=10)
        return f"✅ IP {ip} registered (HTTP {r.status_code})"
    except Exception as e: return f"⚠️ {e}"

def place_equity(security_id, txn, qty, product, order_type, exchange, price):
    seg={"NSE":"NSE_EQ","BSE":"BSE_EQ"}.get(exchange,"NSE_EQ")
    if _dhan_hq:
        try:
            return _dhan_hq.place_order(transactionType=txn,exchangeSegment=seg,
                productType="INTRADAY" if product=="INTRADAY" else "CNC",
                orderType=order_type,validity="DAY",securityId=str(security_id),
                quantity=int(qty),price=float(price) if order_type=="LIMIT" else 0,triggerPrice=0)
        except Exception as e: return {"error":str(e)}
    return {"status":"simulated","message":f"{txn} {qty}×{security_id} @{'MKT' if order_type=='MARKET' else price}"}

def place_options(security_id, txn, qty, segment, order_type, price):
    if _dhan_hq:
        try:
            return _dhan_hq.place_order(transactionType=txn,exchangeSegment=segment,
                productType="INTRADAY",orderType=order_type,validity="DAY",
                securityId=str(security_id),quantity=int(qty),
                price=float(price) if order_type=="LIMIT" else 0,triggerPrice=0)
        except Exception as e: return {"error":str(e)}
    return {"status":"simulated","message":f"OPT {txn} {qty}×{security_id}"}

# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRADING ENGINE  (background thread)
# ─────────────────────────────────────────────────────────────────────────────
def _log(msg):
    now=datetime.now(IST).strftime("%H:%M:%S")
    ts_append("log", f"[{now}]  {msg}")

def live_engine(cfg):
    """
    Background thread.
    ─ Simple Buy/Sell → fire immediately on FIRST loop iteration.
    ─ EMA strategies  → wait for candle boundary, entry on N+1 open.
    ─ SL/Target       → checked vs LTP every tick (not candle low/high).
    ─ Rate limit      → ≥1.5s between yfinance requests.
    """
    ticker   = cfg["ticker"];        tf       = cfg["timeframe"]
    strategy = cfg["strategy"];      fast     = cfg["fast_ema"];  slow  = cfg["slow_ema"]
    sl_type  = cfg["sl_type"];       tgt_type = cfg["target_type"]
    csl      = cfg["custom_sl"];     ctgt     = cfg["custom_target"]
    asl      = cfg.get("atr_mult_sl",2.0); atgt=cfg.get("atr_mult_target",3.0); rr=cfg.get("rr_ratio",2.0)
    qty      = cfg["quantity"];      cd_on    = cfg.get("cooldown_enabled",True); cd_s=cfg.get("cooldown_secs",5)
    no_ovlp  = cfg.get("no_overlap",True); dhan_on=cfg.get("dhan_enabled",False); opts_on=cfg.get("options_enabled",False)
    tf_min   = TF_MINUTES.get(tf,5)
    immediate= strategy in ("Simple Buy","Simple Sell")
    first_run= True

    ts_set("running",True); ts_set("position",None)
    ts_clear("log"); ts_clear("completed_trades")
    ts_set("last_trade_time",None)
    _log(f"🚀 STARTED  {ticker} | {tf} | {strategy}")
    _log(f"   SL:{sl_type}  TGT:{tgt_type}  QTY:{qty}")

    last_fetch   = 0.0
    last_sig_candle = None

    while ts_get("running"):
        try:
            # ── Rate limit ──
            wait=max(0.0, 1.5-(time.time()-last_fetch))
            if wait>0: time.sleep(wait)

            fetch_p = "5d" if tf in ("1m","5m","15m") else "1mo"
            df=fetch_data(ticker, tf, fetch_p, warmup=True)
            last_fetch=time.time()

            if df is None or len(df)<10:
                _log("⚠️ No data – retrying…"); time.sleep(3); continue

            df["ema_fast"] = calc_ema(df["Close"], fast)
            df["ema_slow"] = calc_ema(df["Close"], slow)
            df["atr"]      = calc_atr(df)

            ltp = float(df["Close"].iloc[-1])
            ts_set("ltp",          ltp)
            ts_set("df",           df)
            ts_set("last_candle",  df.iloc[-1].to_dict())
            ts_set("ema_fast_val", float(df["ema_fast"].iloc[-1]))
            ts_set("ema_slow_val", float(df["ema_slow"].iloc[-1]))
            ts_set("atr_val",      float(df["atr"].iloc[-1]))

            if strategy=="Elliott Wave (Auto)":
                ts_set("ew_status", detect_elliott_waves(df))

            # ── Monitor open position (vs LTP every tick) ──
            pos=ts_get("position")
            if pos is not None:
                sl=pos["sl"]; tgt=pos["target"]; tt=pos["type"]
                sl_hit  = ltp<=sl  if tt=="buy" else ltp>=sl
                tgt_hit = (ltp>=tgt if tt=="buy" else ltp<=tgt) and "Display Only" not in tgt_type
                ep=None; er=None
                if sl_hit:  ep=sl;  er="SL Hit (LTP)"
                elif tgt_hit: ep=tgt; er="Target Hit (LTP)"
                if ep is None and tgt_type=="EMA Crossover Exit" and len(df)>2:
                    ef_n=float(df["ema_fast"].iloc[-1]); es_n=float(df["ema_slow"].iloc[-1])
                    ef_p=float(df["ema_fast"].iloc[-2]); es_p=float(df["ema_slow"].iloc[-2])
                    if tt=="buy"  and ef_p>=es_p and ef_n<es_n: ep=ltp; er="EMA Reverse Cross"
                    if tt=="sell" and ef_p<=es_p and ef_n>es_n: ep=ltp; er="EMA Reverse Cross"
                if ep is not None:
                    pnl=round((ep-pos["entry_price"])*(1 if tt=="buy" else -1)*qty,2)
                    rec={"entry_datetime":pos["entry_time"],"exit_datetime":datetime.now(IST),
                         "type":tt,"entry_price":pos["entry_price"],"exit_price":ep,
                         "sl":pos["initial_sl"],"target":pos["target"],
                         "entry_reason":pos["entry_reason"],"exit_reason":er,
                         "pnl":pnl,"quantity":qty}
                    ts_append("completed_trades",rec); ts_set("position",None)
                    emoji="✅" if pnl>=0 else "🔴"
                    _log(f"{emoji} EXIT {tt.upper()} @ {ep:.2f} | P&L ₹{pnl:.2f} | {er}")
                    if dhan_on:
                        if opts_on:
                            sec=cfg.get("ce_security_id" if tt=="buy" else "pe_security_id","")
                            place_options(sec,"SELL",cfg.get("options_qty",65),cfg.get("exchange_segment","NSE_FNO"),cfg.get("opt_exit_order","MARKET"),ltp)
                        else:
                            etxn="SELL" if tt=="buy" else "BUY"
                            place_equity(cfg.get("security_id","1594"),etxn,qty,cfg.get("product_type","INTRADAY"),cfg.get("exit_order_type","MARKET"),cfg.get("exchange","NSE"),ltp)
                else:
                    new_sl=update_trailing_sl(sl,ltp,tt,sl_type,df,len(df)-1,asl)
                    if new_sl!=sl: pos["sl"]=new_sl; ts_set("position",pos)
                continue  # don't look for entries while in trade

            # ── Candle boundary check ──
            now_ist = datetime.now(IST)
            if immediate:
                is_boundary = first_run  # only fire once immediately
            else:
                cur_candle = df.index[-2]
                if tf_min < 1440:
                    total_min = now_ist.hour*60 + now_ist.minute
                    is_boundary = (total_min % tf_min == 0 and
                                   now_ist.second < 15 and
                                   last_sig_candle != cur_candle)
                else:
                    is_boundary = last_sig_candle != cur_candle

            if not is_boundary:
                first_run = False
                time.sleep(1.5); continue

            if not immediate:
                last_sig_candle = df.index[-2]

            # ── Cooldown ──
            if cd_on and not immediate:
                lt=ts_get("last_trade_time")
                if lt and (now_ist-lt).total_seconds()<cd_s:
                    _log(f"⏳ Cooldown ({cd_s}s)"); first_run=False; continue

            # ── No-overlap ──
            if no_ovlp and not immediate:
                done=ts_get("completed_trades") or []
                if done:
                    xt=done[-1].get("exit_datetime")
                    if xt:
                        if isinstance(xt,datetime) and xt.tzinfo is None: xt=IST.localize(xt)
                        if (now_ist-xt).total_seconds()<30: first_run=False; continue

            # ── Generate signal ──
            signal=None; entry_price=ltp

            if strategy=="Simple Buy":
                signal="buy"; entry_price=ltp; _log("📌 Simple Buy triggered immediately")
            elif strategy=="Simple Sell":
                signal="sell"; entry_price=ltp; _log("📌 Simple Sell triggered immediately")
            elif strategy=="EMA Crossover":
                if len(df)>=3:
                    ef_n=float(df["ema_fast"].iloc[-2]); es_n=float(df["ema_slow"].iloc[-2])
                    ef_p=float(df["ema_fast"].iloc[-3]); es_p=float(df["ema_slow"].iloc[-3])
                    if ef_p<=es_p and ef_n>es_n:   signal="buy"
                    elif ef_p>=es_p and ef_n<es_n: signal="sell"
                    if signal: entry_price=float(df["Open"].iloc[-1])
            elif strategy=="Anticipatory EMA Crossover":
                sig_df=gen_anticipatory_signals(df,fast,slow)
                s=sig_df["signal"].iloc[-2] if len(sig_df)>=2 else ""
                if s in ("buy","sell"): signal=s; entry_price=float(df["Open"].iloc[-1])
            elif strategy=="Elliott Wave (Auto)":
                ew=ts_get("ew_status") or {}
                if ew.get("signal"): signal=ew["signal"]; entry_price=ltp

            first_run=False

            if not signal: continue

            tf_filter=cfg.get("trade_filter","Both")
            if tf_filter=="Buy Only"  and signal!="buy":  continue
            if tf_filter=="Sell Only" and signal!="sell": continue

            # ── SL / Target ──
            sl=calc_sl(df,len(df)-1,signal,sl_type,csl,asl,rr)
            tgt=calc_target(df,len(df)-1,signal,tgt_type,entry_price,sl,ctgt,atgt,rr)
            if strategy=="Elliott Wave (Auto)":
                ew=ts_get("ew_status") or {}
                if ew.get("auto_sl"):     sl=ew["auto_sl"]
                if ew.get("auto_target"): tgt=ew["auto_target"]

            pos={"entry_time":now_ist,"entry_price":entry_price,"type":signal,
                 "sl":sl,"initial_sl":sl,"target":tgt,"entry_reason":f"{strategy} Signal","quantity":qty}
            ts_set("position",pos); ts_set("last_trade_time",now_ist)
            _log(f"📈 ENTRY {signal.upper()} @ {entry_price:.2f} | SL {sl:.2f} | TGT {tgt:.2f}")

            if dhan_on:
                if opts_on:
                    sec=cfg.get("ce_security_id","") if signal=="buy" else cfg.get("pe_security_id","")
                    r=place_options(sec,"BUY",cfg.get("options_qty",65),cfg.get("exchange_segment","NSE_FNO"),cfg.get("opt_entry_order","MARKET"),ltp)
                else:
                    txn="BUY" if signal=="buy" else "SELL"
                    r=place_equity(cfg.get("security_id","1594"),txn,qty,cfg.get("product_type","INTRADAY"),cfg.get("entry_order_type","LIMIT"),cfg.get("exchange","NSE"),ltp)
                _log(f"   Order: {r.get('status','?')} – {r.get('message',r.get('error',''))}")

        except Exception as e:
            _log(f"❌ Engine error: {e}"); time.sleep(3)

    _log("🛑 STOPPED")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def mcard(label, value, color="#e8f4ff"):
    return (f'<div class="m-card"><p class="mc-label">{label}</p>'
            f'<p class="mc-val" style="color:{color}">{value}</p></div>')

def cfg_item(label, value):
    return (f'<div class="cfg-item"><div class="cfg-key">{label}</div>'
            f'<div class="cfg-val">{value}</div></div>')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_session()

    # ═════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 8px 8px">
            <div style="font-size:2.2rem">📈</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;
                        color:#00E5CC;letter-spacing:.06em">SMART INVESTING</div>
            <div style="color:#3a5a78;font-size:.72rem;letter-spacing:.12em;
                        text-transform:uppercase;margin-top:2px">Algorithmic Trading Platform</div>
        </div>
        <hr style="border-color:#0f2040;margin:10px 0 14px">
        """, unsafe_allow_html=True)

        with st.expander("📊 Market Setup", expanded=True):
            ticker_label = st.selectbox("Instrument", list(PRESET_TICKERS.keys()), key="sb_tl")
            selected_ticker = st.text_input("Ticker Symbol","RELIANCE.NS",key="sb_ct") \
                if ticker_label=="Custom" else PRESET_TICKERS[ticker_label]
            timeframe = st.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()), index=2, key="sb_tf")
            period    = st.selectbox("Period", TIMEFRAME_PERIODS[timeframe], key="sb_pd")
            quantity  = st.number_input("Quantity", min_value=1, value=1, key="sb_qty")

        with st.expander("🎯 Strategy", expanded=True):
            strategy     = st.selectbox("Strategy", STRATEGIES, key="sb_strat")
            trade_filter = st.selectbox("Direction", ["Both","Buy Only","Sell Only"], key="sb_dir")
            fast_ema=slow_ema=9; xover_type="Simple Crossover"; candle_size=10.0
            check_angle=False; min_angle=0.0
            if strategy in ("EMA Crossover","Anticipatory EMA Crossover"):
                c1,c2=st.columns(2)
                fast_ema=c1.number_input("Fast EMA",2,200,9,key="sb_fe")
                slow_ema=c2.number_input("Slow EMA",3,500,15,key="sb_se")
            if strategy=="EMA Crossover":
                xover_type=st.selectbox("Crossover Filter",
                    ["Simple Crossover","Custom Candle Size","ATR Based Candle Size"],key="sb_xt")
                if xover_type=="Custom Candle Size":
                    candle_size=st.number_input("Min Candle Body (pts)",value=10.0,key="sb_cs")
                check_angle=st.checkbox("Angle Filter",value=False,key="sb_ca")
                min_angle=st.number_input("Min Angle",value=0.0,step=0.1,key="sb_ma") if check_angle else 0.0

        with st.expander("🛡️ Stop Loss & Target", expanded=True):
            sl_type     = st.selectbox("SL Type",     SL_TYPES,     key="sb_sl")
            target_type = st.selectbox("Target Type", TARGET_TYPES, key="sb_tt")
            custom_sl   = st.number_input("Custom SL (pts)",    value=10.0,key="sb_csl") if sl_type=="Custom Points"    else 10.0
            custom_tgt  = st.number_input("Custom Target (pts)",value=20.0,key="sb_ctg") if target_type=="Custom Points" else 20.0
            atr_mult_sl=2.0; atr_mult_tgt=3.0
            if "ATR" in sl_type or "ATR" in target_type:
                c1,c2=st.columns(2)
                atr_mult_sl =c1.number_input("ATR×SL",value=2.0,step=0.5,key="sb_as")
                atr_mult_tgt=c2.number_input("ATR×Tgt",value=3.0,step=0.5,key="sb_at")
            rr_ratio=2.0
            if "Risk Reward" in sl_type or "Risk Reward" in target_type:
                rr_ratio=st.number_input("RR Ratio",value=2.0,step=0.5,key="sb_rr")

        with st.expander("⚡ Live Settings", expanded=False):
            cooldown_on = st.checkbox("Cooldown between trades",value=True,key="sb_cd")
            cooldown_s  = st.number_input("Cooldown (secs)",value=5,min_value=1,key="sb_cs2") if cooldown_on else 5
            no_overlap  = st.checkbox("Prevent overlapping trades",value=True,key="sb_no")

        with st.expander("🏦 Dhan Broker", expanded=False):
            dhan_enabled=st.checkbox("Enable Dhan Broker",value=False,key="sb_dhan")
            client_id=access_token=""
            opts_on=False; security_id="1594"; product_type="INTRADAY"; exchange="NSE"
            entry_order="LIMIT"; exit_order="MARKET"
            ce_sec_id=pe_sec_id=""; opts_qty=65; exc_segment="NSE_FNO"; opt_entry="MARKET"; opt_exit="MARKET"
            if dhan_enabled:
                client_id=st.text_input("Client ID",type="password",key="sb_cid")
                access_token=st.text_input("Access Token",type="password",key="sb_tok")
                if st.button("🔑 Register IP (SEBI)",key="sb_rip"):
                    if client_id and access_token:
                        init_dhan(client_id,access_token)
                        st.info(register_ip(client_id,access_token))
                opts_on=st.checkbox("Options Trading",value=False,key="sb_opts")
                if not opts_on:
                    product_type=st.selectbox("Product",["INTRADAY","DELIVERY"],key="sb_pr")
                    exchange=st.selectbox("Exchange",["NSE","BSE"],key="sb_ex")
                    security_id=st.text_input("Security ID",value="1594",key="sb_sid")
                    entry_order=st.selectbox("Entry Order",["LIMIT","MARKET"],key="sb_eo")
                    exit_order=st.selectbox("Exit Order",["MARKET","LIMIT"],key="sb_xo")
                else:
                    exc_segment=st.selectbox("Segment",["NSE_FNO","BSE_FNO"],key="sb_seg")
                    ce_sec_id=st.text_input("CE Security ID","",key="sb_ce")
                    pe_sec_id=st.text_input("PE Security ID","",key="sb_pe")
                    opts_qty=st.number_input("Options Qty",value=65,min_value=1,key="sb_oq")
                    opt_entry=st.selectbox("Entry Order",["MARKET","LIMIT"],key="sb_oen")
                    opt_exit=st.selectbox("Exit Order",["MARKET","LIMIT"],key="sb_oex")

    # ═════════════════════════════════════════════════════════════════
    # LTP BANNER
    # ═════════════════════════════════════════════════════════════════
    ltp_val  = ts_get("ltp")
    ltp_disp = f"₹{ltp_val:,.2f}" if ltp_val else "— —"
    now_ist  = datetime.now(IST).strftime("%d %b %Y  %H:%M:%S IST")
    st.markdown(f"""
    <div class="ltp-banner">
        <div>
            <div class="ticker-name">{selected_ticker}</div>
            <div class="ltp-val">{ltp_disp}</div>
        </div>
        <div style="text-align:right">
            <div class="ltp-time">{now_ist}</div>
            <div style="color:#1a4060;font-size:.7rem;margin-top:3px">yfinance delayed data</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════
    # TABS
    # ═════════════════════════════════════════════════════════════════
    tab_bt, tab_lt, tab_hist, tab_opt = st.tabs(
        ["📊  Backtesting","⚡  Live Trading","📁  Trade History","⚙️  Optimization"])

    # ─────────────────────────────────────────────────────
    # BACKTESTING
    # ─────────────────────────────────────────────────────
    with tab_bt:
        st.markdown("### 📊 Strategy Backtesting")
        st.caption("Conservative: SL checked against candle Low/High first, then Target")

        if st.button("▶  Run Backtest", type="primary", key="run_bt"):
            with st.spinner("Fetching data…"):
                df_raw=fetch_data(selected_ticker, timeframe, period, warmup=True)
            if df_raw is None or df_raw.empty:
                st.error("No data returned. Check ticker / period.")
            else:
                bt_cfg=dict(strategy=strategy,trade_filter=trade_filter,
                    sl_type=sl_type,target_type=target_type,quantity=quantity,
                    fast_ema=fast_ema,slow_ema=slow_ema,custom_sl=custom_sl,
                    custom_target=custom_tgt,atr_mult_sl=atr_mult_sl,
                    atr_mult_target=atr_mult_tgt,rr_ratio=rr_ratio,
                    xover_type=xover_type,candle_size=candle_size,
                    min_angle=min_angle,check_angle=check_angle)
                with st.spinner("Running strategy…"):
                    trades,violations,accuracy,df_sig=run_backtest(df_raw.copy(),bt_cfg)
                st.session_state["backtest_results"]=(trades,violations,accuracy)
                st.session_state["backtest_df"]=df_sig

        if st.session_state.get("backtest_results"):
            trades,violations,accuracy=st.session_state["backtest_results"]
            df_sig=st.session_state.get("backtest_df")
            if not trades:
                st.info("No trades generated. Try different EMA parameters or a longer period.")
            else:
                total_pnl=sum(t["pnl"] for t in trades)
                winners=sum(1 for t in trades if t["pnl"]>0)
                losers=len(trades)-winners
                avg_win=sum(t["pnl"] for t in trades if t["pnl"]>0)/max(winners,1)
                avg_loss=sum(t["pnl"] for t in trades if t["pnl"]<=0)/max(losers,1)

                cols=st.columns(6)
                for col,(lbl,val,clr) in zip(cols,[
                    ("Trades",str(len(trades)),"#e8f4ff"),
                    ("P&L",f"₹{total_pnl:,.0f}","#00c896" if total_pnl>=0 else "#ff4d6d"),
                    ("Accuracy",f"{accuracy:.1f}%","#FFD93D"),
                    ("Winners",str(winners),"#00c896"),
                    ("Losers",str(losers),"#ff4d6d"),
                    ("Violations",str(len(violations)),"#FF9F43"),
                ]):
                    col.markdown(mcard(lbl,val,clr),unsafe_allow_html=True)

                st.markdown("<br>",unsafe_allow_html=True)
                if df_sig is not None:
                    st.plotly_chart(build_bt_chart(df_sig,trades,fast_ema,slow_ema),use_container_width=True)
                st.plotly_chart(build_pnl_chart(trades),use_container_width=True)

                st.markdown("#### 📋 Trade Log")
                tdf=pd.DataFrame(trades)
                for c in ("entry_datetime","exit_datetime"):
                    if c in tdf.columns:
                        tdf[c]=pd.to_datetime(tdf[c]).dt.strftime("%Y-%m-%d %H:%M")

                def _sty(row):
                    out=[]
                    for col in row.index:
                        if col=="pnl": out.append("color:#00c896;font-weight:600" if row[col]>0 else "color:#ff4d6d;font-weight:600")
                        elif col=="type": out.append("color:#00c896" if row[col]=="buy" else "color:#ff4d6d")
                        elif col=="sl_violation": out.append("color:#FF9F43;font-weight:700" if row[col] else "")
                        else: out.append("")
                    return out
                st.dataframe(tdf.style.apply(_sty,axis=1),use_container_width=True,height=420)

                if violations:
                    st.markdown(f"""<div class="violation-box">
                        ⚠️ <strong>{len(violations)} trades</strong> had SL & Target both reachable in same candle.
                        Conservative SL exit applied.</div>""",unsafe_allow_html=True)

                with st.expander("📈 Extended Stats"):
                    sc1,sc2,sc3,sc4=st.columns(4)
                    for sc,(lbl,val,clr) in zip([sc1,sc2,sc3,sc4],[
                        ("Avg Win",f"₹{avg_win:,.2f}","#00c896"),
                        ("Avg Loss",f"₹{avg_loss:,.2f}","#ff4d6d"),
                        ("Best",f"₹{max(t['pnl'] for t in trades):,.2f}","#00c896"),
                        ("Worst",f"₹{min(t['pnl'] for t in trades):,.2f}","#ff4d6d"),
                    ]):
                        sc.markdown(mcard(lbl,val,clr),unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # LIVE TRADING
    # ─────────────────────────────────────────────────────
    with tab_lt:
        is_running = bool(ts_get("running"))

        # ── Control buttons – always visible ──
        btn_col1, btn_col2, spacer = st.columns([1, 1, 4])

        with btn_col1:
            start_clicked = st.button(
                "▶  START" if not is_running else "▶  RUNNING…",
                type="primary", use_container_width=True, key="lt_start",
                disabled=is_running,
            )

        with btn_col2:
            stop_clicked = st.button(
                "⏹  STOP",
                use_container_width=True, key="lt_stop",
                disabled=not is_running,
            )

        # Status badge
        dot  = '<span class="status-live"></span>' if is_running else '<span class="status-off"></span>'
        stat = "● LIVE" if is_running else "○ Stopped"
        clr  = "#00E5CC" if is_running else "#3a5a78"
        st.markdown(f'{dot}<span style="color:{clr};font-weight:700;font-size:.9rem">{stat}</span>',
                     unsafe_allow_html=True)

        # ── Handle button clicks ──
        if start_clicked and not is_running:
            live_cfg = dict(
                ticker=selected_ticker, timeframe=timeframe, period=period,
                strategy=strategy, trade_filter=trade_filter,
                fast_ema=fast_ema, slow_ema=slow_ema,
                sl_type=sl_type, target_type=target_type,
                custom_sl=custom_sl, custom_target=custom_tgt,
                atr_mult_sl=atr_mult_sl, atr_mult_target=atr_mult_tgt,
                rr_ratio=rr_ratio, quantity=quantity,
                cooldown_enabled=cooldown_on, cooldown_secs=cooldown_s,
                no_overlap=no_overlap, dhan_enabled=dhan_enabled,
                options_enabled=opts_on,
                xover_type=xover_type, candle_size=candle_size,
                min_angle=min_angle, check_angle=check_angle,
                security_id=security_id, product_type=product_type,
                exchange=exchange, entry_order_type=entry_order,
                exit_order_type=exit_order,
                ce_security_id=ce_sec_id, pe_security_id=pe_sec_id,
                options_qty=opts_qty, exchange_segment=exc_segment,
                opt_entry_order=opt_entry, opt_exit_order=opt_exit,
            )
            ts_set("config", live_cfg)
            t = threading.Thread(target=live_engine, args=(live_cfg,), daemon=True)
            t.start()
            st.session_state["live_thread"] = t
            st.rerun()

        if stop_clicked and is_running:
            ts_set("running", False)
            st.rerun()

        st.divider()

        # ══ Active config display (shown once started) ══
        cfg_live = ts_get("config")
        if cfg_live:
            st.markdown("#### 🔧 Active Configuration")
            items = [
                ("Ticker",      cfg_live.get("ticker","-")),
                ("Timeframe",   cfg_live.get("timeframe","-")),
                ("Period",      cfg_live.get("period","-")),
                ("Strategy",    cfg_live.get("strategy","-")),
                ("Direction",   cfg_live.get("trade_filter","-")),
                ("Fast EMA",    str(cfg_live.get("fast_ema","-"))),
                ("Slow EMA",    str(cfg_live.get("slow_ema","-"))),
                ("SL Type",     cfg_live.get("sl_type","-")),
                ("Target Type", cfg_live.get("target_type","-")),
                ("Custom SL",   str(cfg_live.get("custom_sl","-"))),
                ("Custom Tgt",  str(cfg_live.get("custom_target","-"))),
                ("Quantity",    str(cfg_live.get("quantity","-"))),
                ("Cooldown",    f"{cfg_live.get('cooldown_secs',5)}s" if cfg_live.get("cooldown_enabled") else "Off"),
                ("No Overlap",  "✅" if cfg_live.get("no_overlap") else "❌"),
                ("Dhan Broker", "✅ On" if cfg_live.get("dhan_enabled") else "❌ Off"),
                ("Options",     "✅ On" if cfg_live.get("options_enabled") else "❌ Off"),
            ]
            grid_html = '<div class="cfg-grid">' + "".join(cfg_item(k,v) for k,v in items) + '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # ══ Live metrics row (LTP, Entry, PnL, SL, Target, EMA values) ══
        ltp_now = ts_get("ltp")
        pos     = ts_get("position")
        ef_val  = ts_get("ema_fast_val")
        es_val  = ts_get("ema_slow_val")
        atr_v   = ts_get("atr_val")

        if ltp_now or pos:
            st.markdown("#### 📊 Live Metrics")
            entry_val   = f"₹{pos['entry_price']:.2f}" if pos else "—"
            sl_val_disp = f"₹{pos['sl']:.2f}"          if pos else "—"
            tgt_disp    = f"₹{pos['target']:.2f}"      if pos else "—"
            ltp_disp2   = f"₹{ltp_now:.2f}"            if ltp_now else "—"
            ef_disp     = f"{ef_val:.2f}"               if ef_val else "—"
            es_disp     = f"{es_val:.2f}"               if es_val else "—"
            atr_disp    = f"{atr_v:.2f}"                if atr_v else "—"

            upnl_val = None
            if pos and ltp_now:
                upnl_val = (ltp_now - pos["entry_price"]) * (1 if pos["type"]=="buy" else -1) * quantity
            pnl_disp = f"₹{upnl_val:.2f}" if upnl_val is not None else "—"
            pnl_clr  = ("#00c896" if upnl_val>=0 else "#ff4d6d") if upnl_val is not None else "#e8f4ff"
            tt_clr   = ("#00c896" if pos["type"]=="buy" else "#ff4d6d") if pos else "#7a9fc0"
            tt_disp  = pos["type"].upper() if pos else "—"

            m1,m2,m3,m4,m5,m6,m7,m8 = st.columns(8)
            m1.markdown(mcard("LTP",        ltp_disp2,  "#00E5CC"), unsafe_allow_html=True)
            m2.markdown(mcard("Trade Type", tt_disp,    tt_clr),   unsafe_allow_html=True)
            m3.markdown(mcard("Entry",      entry_val,  "#FFD93D"), unsafe_allow_html=True)
            m4.markdown(mcard("Stop Loss",  sl_val_disp,"#ff4d6d"), unsafe_allow_html=True)
            m5.markdown(mcard("Target",     tgt_disp,   "#00c896"), unsafe_allow_html=True)
            m6.markdown(mcard("Unreal P&L", pnl_disp,   pnl_clr),  unsafe_allow_html=True)
            m7.markdown(mcard(f"EMA {fast_ema}", ef_disp, "#FF9F43"), unsafe_allow_html=True)
            m8.markdown(mcard(f"EMA {slow_ema}", es_disp, "#4ECDC4"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # ══ Main layout: chart + position card ══
        lc, rc = st.columns([3, 1])

        with lc:
            df_live = ts_get("df")
            if df_live is not None:
                st.plotly_chart(build_live_chart(df_live, pos, fast_ema, slow_ema),
                                 use_container_width=True)
                # Last candle
                lc_row = ts_get("last_candle")
                if lc_row:
                    disp = {k:round(v,4) if isinstance(v,float) else v
                            for k,v in lc_row.items() if not k.startswith("_")}
                    st.caption("🕯️ Last Fetched Candle")
                    st.dataframe(pd.DataFrame([disp]), use_container_width=True)
            else:
                st.markdown("""<div style="height:280px;display:flex;align-items:center;
                    justify-content:center;background:#080F20;border-radius:14px;
                    border:1px solid #0f2040;color:#3a5a78;font-size:1rem">
                    Press START to begin live trading</div>""", unsafe_allow_html=True)

        with rc:
            # Position card
            if pos:
                ltp_now = ts_get("ltp") or pos["entry_price"]
                upnl = (ltp_now - pos["entry_price"]) * (1 if pos["type"]=="buy" else -1) * quantity
                upnl_clr = "#00c896" if upnl>=0 else "#ff4d6d"
                card_cls = "buy-card" if pos["type"]=="buy" else "sell-card"
                badge    = "🟢 BUY"  if pos["type"]=="buy"  else "🔴 SELL"
                ttc      = "#00c896" if pos["type"]=="buy"  else "#ff4d6d"
                st.markdown("#### 🎯 Open Position")
                st.markdown(f"""
                <div class="pos-card {card_cls}">
                    <div style="color:{ttc};font-weight:800;font-size:1rem;margin-bottom:10px">{badge}</div>
                    <div class="wave-row"><span class="pos-label">Entry</span><span class="pos-val">₹{pos['entry_price']:.2f}</span></div>
                    <div class="wave-row"><span class="pos-label">LTP</span><span class="pos-val">₹{ltp_now:.2f}</span></div>
                    <div class="wave-row"><span class="pos-label">Stop Loss</span><span class="pos-val" style="color:#ff4d6d">₹{pos['sl']:.2f}</span></div>
                    <div class="wave-row"><span class="pos-label">Target</span><span class="pos-val" style="color:#00c896">₹{pos['target']:.2f}</span></div>
                    <div class="wave-row"><span class="pos-label">Quantity</span><span class="pos-val">{pos.get('quantity',quantity)}</span></div>
                    <hr style="border-color:#1a3060;margin:8px 0">
                    <div class="wave-row"><span class="pos-label">Unrealised P&L</span>
                        <span class="pos-val" style="color:{upnl_clr}">₹{upnl:.2f}</span></div>
                    <div style="color:#3a5a78;font-size:.68rem;margin-top:6px">{pos['entry_reason']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background:#080F20;border:1px dashed #0f2040;
                    border-radius:12px;padding:20px;text-align:center;color:#3a5a78;font-size:.85rem">
                    No open position</div>""", unsafe_allow_html=True)

            # Elliott Wave panel
            if strategy=="Elliott Wave (Auto)":
                ew = ts_get("ew_status") or {}
                st.markdown("#### 🌊 Elliott Wave")
                comp=", ".join(ew.get("completed_waves",[])) or "—"
                nt=ew.get("next_target"); ntd=f"₹{nt:.2f}" if nt else "—"
                sd=ew.get("signal","—").upper(); sc="#00c896" if ew.get("signal")=="buy" else "#ff4d6d" if ew.get("signal")=="sell" else "#7a9fc0"
                st.markdown(f"""<div class="wave-card">
                    <div class="wave-status">{ew.get("wave_status","Analyzing…")}</div>
                    <div class="wave-row"><span class="wave-key">Current Wave</span><span class="wave-val">{ew.get("current_wave","—")}</span></div>
                    <div class="wave-row"><span class="wave-key">Completed</span><span class="wave-val">{comp}</span></div>
                    <div class="wave-row"><span class="wave-key">Signal</span><span class="wave-val" style="color:{sc}">{sd}</span></div>
                    <div class="wave-row"><span class="wave-key">Next Target</span><span class="wave-val" style="color:#FFD93D">{ntd}</span></div>
                </div>""", unsafe_allow_html=True)
                proj=ew.get("wave_projections",{})
                if proj:
                    for wn,wl in proj.items():
                        st.markdown(f"""<div class="wave-row"><span class="wave-key" style="font-size:.75rem">{wn}</span>
                            <span class="wave-val" style="font-size:.8rem">₹{wl:.2f}</span></div>""", unsafe_allow_html=True)

            # ATR indicator
            if atr_v:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(mcard("ATR", f"{atr_v:.2f}", "#FFD93D"), unsafe_allow_html=True)

        # ── Live Log ──
        st.markdown("#### 📋 Activity Log")
        logs = ts_get("log") or []
        if logs:
            lines=""
            for e in reversed(logs[-80:]):
                clr=("#00E5CC" if "ENTRY" in e or "started" in e.lower()
                     else "#ff4d6d" if "❌" in e or "🔴" in e
                     else "#FFD93D" if "⚠️" in e
                     else "#FF9F43" if "STOP" in e
                     else "#00c896" if "✅" in e
                     else "#5a7fa8")
                lines+=f'<div class="log-entry" style="color:{clr};margin:1px 0">{e}</div>'
            st.markdown(f'<div class="log-box">{lines}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="log-box"><span style="color:#1a3060">Log empty – press START</span></div>',
                         unsafe_allow_html=True)

        # ── Auto-refresh every 1.5 s (only while running, no full flicker) ──
        if is_running:
            time.sleep(1.5)
            st.rerun()

    # ─────────────────────────────────────────────────────
    # TRADE HISTORY
    # ─────────────────────────────────────────────────────
    with tab_hist:
        st.markdown("### 📁 Trade History")
        st.caption("Updates automatically while live trading is running.")

        completed = ts_get("completed_trades") or []

        if not completed:
            st.markdown("""<div style="background:#080F20;border:1px dashed #0f2040;
                border-radius:14px;padding:40px;text-align:center;color:#3a5a78">
                <div style="font-size:2rem;margin-bottom:8px">📭</div>
                No completed trades yet. They appear here in real-time as trades close.
            </div>""", unsafe_allow_html=True)
        else:
            total_pnl=sum(t["pnl"] for t in completed)
            winners=sum(1 for t in completed if t["pnl"]>0)
            n=len(completed); acc=winners/n*100 if n else 0
            hc=st.columns(5)
            for c,(lbl,val,clr) in zip(hc,[
                ("Trades",str(n),"#e8f4ff"),
                ("Total P&L",f"₹{total_pnl:,.2f}","#00c896" if total_pnl>=0 else "#ff4d6d"),
                ("Accuracy",f"{acc:.1f}%","#FFD93D"),
                ("Winners",str(winners),"#00c896"),
                ("Losers",str(n-winners),"#ff4d6d"),
            ]):
                c.markdown(mcard(lbl,val,clr),unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            if n>1: st.plotly_chart(build_pnl_chart(completed),use_container_width=True)

            hist_df=pd.DataFrame(completed)
            for c in ("entry_datetime","exit_datetime"):
                if c in hist_df.columns:
                    hist_df[c]=pd.to_datetime(hist_df[c]).dt.strftime("%Y-%m-%d %H:%M:%S")

            def _sh(row):
                out=[]
                for col in row.index:
                    if col=="pnl": out.append("color:#00c896;font-weight:600" if row[col]>0 else "color:#ff4d6d;font-weight:600")
                    elif col=="type": out.append("color:#00c896" if row[col]=="buy" else "color:#ff4d6d")
                    else: out.append("")
                return out
            st.dataframe(hist_df.style.apply(_sh,axis=1),use_container_width=True,height=480)
            st.download_button("📥 Download CSV",data=hist_df.to_csv(index=False),
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",mime="text/csv")

    # ─────────────────────────────────────────────────────
    # OPTIMIZATION
    # ─────────────────────────────────────────────────────
    with tab_opt:
        st.markdown("### ⚙️ Strategy Optimization")
        st.caption("Grid-search EMA / SL / Target combinations. All results shown even if target not met.")

        oc1, oc2 = st.columns([1, 2])
        with oc1:
            tgt_acc    = st.number_input("Target Accuracy (%)", 0.0, 100.0, 60.0, 5.0, key="opt_acc")
            min_pts    = st.number_input("Min Total P&L (points)", value=0.0, step=10.0, key="opt_minpts",
                                          help="Only show results where total P&L ≥ this value")
            fast_r     = st.slider("Fast EMA Range",   3,  50, (5,  20), key="opt_fr")
            slow_r     = st.slider("Slow EMA Range",   10,100, (15, 50), key="opt_sr")
            sl_r       = st.slider("SL Points Range",  3, 100, (5,  30), key="opt_slr")
            tgt_r      = st.slider("Target Pts Range", 5, 200, (10, 60), key="opt_tgtr")
            max_combos = st.number_input("Max Combinations", 10, 2000, 100, key="opt_mc")
            run_opt    = st.button("🔍 Run Optimization", type="primary",
                                    use_container_width=True, key="run_opt")

        with oc2:
            if run_opt:
                df_opt=fetch_data(selected_ticker, timeframe, period, warmup=True)
                if df_opt is None or df_opt.empty:
                    st.error("Failed to fetch data.")
                else:
                    fast_v=list(range(fast_r[0], fast_r[1]+1, max(1,(fast_r[1]-fast_r[0])//8)))
                    slow_v=list(range(slow_r[0], slow_r[1]+1, max(1,(slow_r[1]-slow_r[0])//8)))
                    sl_v  =list(range(sl_r[0],   sl_r[1]+1,   max(1,(sl_r[1]-sl_r[0])//5)))
                    tgt_v =list(range(tgt_r[0],  tgt_r[1]+1,  max(1,(tgt_r[1]-tgt_r[0])//5)))
                    combos=[(f,s,sl,tg) for f,s,sl,tg in itertools.product(fast_v,slow_v,sl_v,tgt_v) if f<s]
                    random.shuffle(combos); combos=combos[:int(max_combos)]
                    prog=st.progress(0.0,"Optimizing…"); results=[]
                    for idx_o,(f,s,sl,tg) in enumerate(combos):
                        try:
                            ocfg=dict(strategy="EMA Crossover",trade_filter="Both",
                                sl_type="Custom Points",target_type="Custom Points",
                                quantity=1,fast_ema=f,slow_ema=s,
                                custom_sl=float(sl),custom_target=float(tg),
                                atr_mult_sl=2.0,atr_mult_target=3.0,rr_ratio=2.0,
                                xover_type="Simple Crossover",candle_size=10.0,
                                min_angle=0.0,check_angle=False)
                            ts_o,_,acc_o,_=run_backtest(df_opt.copy(),ocfg)
                            if ts_o:
                                pnl_o=sum(t["pnl"] for t in ts_o)
                                results.append({"Fast EMA":f,"Slow EMA":s,"SL (pts)":sl,"Target (pts)":tg,
                                    "Trades":len(ts_o),"Accuracy (%)":round(acc_o,1),
                                    "Total P&L":round(pnl_o,2),
                                    "Meets Accuracy":acc_o>=tgt_acc,
                                    "Meets Min PnL":pnl_o>=min_pts})
                        except: pass
                        prog.progress((idx_o+1)/len(combos),f"Testing {idx_o+1}/{len(combos)}…")
                    prog.empty()
                    if results:
                        res_df=pd.DataFrame(results).sort_values("Total P&L",ascending=False)
                        st.session_state["opt_results"]=res_df
                    else:
                        st.warning("No valid combinations.")

            if st.session_state.get("opt_results") is not None:
                res_df=st.session_state["opt_results"]
                both  =res_df[res_df["Meets Accuracy"] & res_df["Meets Min PnL"]]
                n_meet=len(both); n_all=len(res_df)

                if n_meet:
                    st.success(f"✅ {n_meet}/{n_all} meet accuracy ≥{tgt_acc:.0f}% AND P&L ≥{min_pts:.0f}")
                    st.dataframe(both,use_container_width=True)
                else:
                    st.warning(f"⚠️ 0 combinations met both filters – showing all {n_all} results:")

                def _so(row):
                    out=[]
                    for col in row.index:
                        if col=="Total P&L": out.append("color:#00c896" if row[col]>0 else "color:#ff4d6d")
                        elif col=="Accuracy (%)": out.append("color:#FFD93D")
                        elif col in ("Meets Accuracy","Meets Min PnL"):
                            out.append("color:#00c896;font-weight:700" if row[col] else "color:#3a5a78")
                        else: out.append("")
                    return out
                st.markdown(f"**All {n_all} results (sorted by P&L):**")
                st.dataframe(res_df.style.apply(_so,axis=1),use_container_width=True,height=480)

                best=res_df.iloc[0]
                st.markdown(f"""<div class="best-result">
                    <div style="color:#00E5CC;font-weight:800;margin-bottom:10px">🏆 Best Configuration</div>
                    <div class="wave-row"><span class="wave-key">Fast × Slow EMA</span><span class="wave-val">{int(best['Fast EMA'])} × {int(best['Slow EMA'])}</span></div>
                    <div class="wave-row"><span class="wave-key">SL / Target (pts)</span><span class="wave-val">{int(best['SL (pts)'])} / {int(best['Target (pts)'])}</span></div>
                    <div class="wave-row"><span class="wave-key">Accuracy</span><span class="wave-val" style="color:#FFD93D">{best['Accuracy (%)']:.1f}%</span></div>
                    <div class="wave-row"><span class="wave-key">Total P&L</span><span class="wave-val" style="color:{'#00c896' if best['Total P&L']>=0 else '#ff4d6d'}">₹{best['Total P&L']:,.2f}</span></div>
                    <div class="wave-row"><span class="wave-key">Trades</span><span class="wave-val">{int(best['Trades'])}</span></div>
                </div>""", unsafe_allow_html=True)
                st.download_button("📥 Download CSV",data=res_df.to_csv(index=False),
                    file_name="optimization.csv",mime="text/csv")

if __name__=="__main__":
    main()
