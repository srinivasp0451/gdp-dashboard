#!/usr/bin/env python3
"""
Smart Investing — Production-Grade Algorithmic Trading Platform
Single-file Streamlit app | All timestamps IST | v3.1.0
Run: streamlit run smart_investing.py --server.address 0.0.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import threading, time, requests, traceback
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

IST = ZoneInfo("Asia/Kolkata")
APP_VERSION = "3.1.0"

STRATEGIES = ["EMA Crossover","Anticipatory EMA","Elliott Wave","Wave Extrema","Simple Buy","Simple Sell"]

SL_TYPES = [
    "Custom Points",
    "ATR",
    "Trailing SL",
    "Trailing SL (3-Phase)",
    "Auto SL",
    "Adaptive SL",
    "ADV Volatility Based",
    "EMA Reverse Crossover",
    "Trailing Swing Low/High",
    "Candle Low/High",
    "Support/Resistance Trailing",
    "Volatility Based",
    "Cost-to-Cost + N Points",
    "Shift to Cost-to-Cost - N pts",
    "Ratchet SL (50% target → trail)",
    "Exit if SL past N pts stagnant",
    "Strategy Reverse Signal",
    "Smart SL (auto)",
]

TARGET_TYPES = [
    "Custom Points",
    "Risk:Reward",
    "ATR Multiple",
    "Adaptive Target",
    "Partial Exit + Ride",
    "Exit if Target fell 50%",
    "Smart Target (auto)",
]

TRAILING_SL_TYPES = {
    "Trailing SL","Trailing SL (3-Phase)","Trailing Swing Low/High",
    "Support/Resistance Trailing","Cost-to-Cost + N Points",
    "Adaptive SL","ADV Volatility Based","Ratchet SL (50% target → trail)",
    "Shift to Cost-to-Cost - N pts",
}

TIMEFRAMES = ["1m","5m","15m","30m","1h","1d"]
PERIODS    = ["1d","5d","7d","1mo","3mo","6mo","1y"]
FIB_RATIOS = [0.236,0.382,0.500,0.618,0.786,1.000,1.272,1.618,2.618]
_REFRESH   = {"1m":15,"5m":30,"15m":60,"30m":120,"1h":300,"1d":600}

NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS",
    "ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS",
    "BAJFINANCE.NS","WIPRO.NS","HCLTECH.NS","NTPC.NS","POWERGRID.NS","TATAMOTORS.NS",
    "ONGC.NS","TATASTEEL.NS","M&M.NS","TECHM.NS","ADANIENT.NS","ADANIPORTS.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS","JSWSTEEL.NS","NESTLEIND.NS",
    "SBILIFE.NS","SHREECEM.NS","TATACONSUM.NS","CIPLA.NS","APOLLOHOSP.NS",
    "BAJAJFINSV.NS","BPCL.NS","BRITANNIA.NS","HDFCLIFE.NS","UPL.NS","KAYNES.NS",
    "^NSEI","BTC-USD","ETH-USD",
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Investing", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');
:root{--bg:#0d1117;--surf:#161b22;--surf2:#21262d;--border:#30363d;
      --text:#e6edf3;--muted:#8b949e;--green:#3fb950;--red:#f85149;
      --blue:#58a6ff;--yellow:#d29922;--purple:#bc8cff;}
html,body,[class*="css"],.stApp,.main,.block-container{
  background-color:var(--bg)!important;color:var(--text)!important;
  font-family:'Space Mono',monospace!important;}
h1,h2,h3,h4,h5,h6{font-family:'Syne',sans-serif!important;color:var(--text)!important;}
div,span,p,label,li,td,th{color:var(--text)!important;}
a{color:var(--blue)!important;}
[data-testid="stSidebar"]{background-color:var(--surf)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="metric-container"]{background-color:var(--surf)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:14px 16px!important;}
[data-testid="metric-container"] *,[data-testid="stMetricValue"]{color:var(--text)!important;font-size:.95rem!important;overflow:hidden;text-overflow:clip!important;white-space:nowrap}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.65rem!important;overflow:hidden;text-overflow:clip!important;white-space:nowrap}
input,select,textarea{background-color:var(--surf2)!important;border-color:var(--border)!important;color:var(--text)!important;}
[data-baseweb="input"],[data-baseweb="select"],[data-baseweb="textarea"]{background-color:var(--surf2)!important;border-color:var(--border)!important;}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{color:var(--text)!important;background-color:var(--surf2)!important;}
/* selectbox trigger */
[data-testid="stSelectbox"] [data-baseweb="select"] div,[data-testid="stSelectbox"] [data-baseweb="select"] span{color:var(--text)!important;background-color:var(--surf2)!important;}
/* selectbox/multiselect dropdown popover list */
[data-baseweb="popover"],[data-baseweb="menu"],[data-baseweb="list"]{background-color:var(--surf2)!important;border:1px solid var(--border)!important;}
[data-baseweb="popover"] *,[data-baseweb="menu"] *,[data-baseweb="list"] *{background-color:var(--surf2)!important;color:var(--text)!important;}
[data-baseweb="option"]{background-color:var(--surf2)!important;color:var(--text)!important;}
[data-baseweb="option"]:hover,[data-baseweb="option"][aria-selected="true"]{background-color:#2d333b!important;color:#58a6ff!important;}
/* multiselect tags */
[data-baseweb="tag"]{background-color:#1f3a5f!important;color:#58a6ff!important;}
[data-baseweb="tag"] span{color:#58a6ff!important;}
/* radio + checkbox */
[data-testid="stRadio"] *,[data-testid="stCheckbox"] *{color:var(--text)!important;}
/* slider */
[data-testid="stSlider"] *{color:var(--text)!important;}
/* info/warning boxes text */
[data-testid="stAlert"] *{color:var(--text)!important;}
.stButton>button{background-color:var(--blue)!important;color:#0d1117!important;border:none!important;
  border-radius:6px!important;font-family:'Space Mono',monospace!important;font-weight:700!important;}
.stButton>button:hover{opacity:.82!important;}
.stTabs [data-baseweb="tab-list"]{background-color:var(--surf)!important;border-bottom:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-family:'Syne',sans-serif!important;font-weight:600!important;padding:8px 20px!important;}
.stTabs [aria-selected="true"]{color:var(--blue)!important;border-bottom:2px solid var(--blue)!important;background-color:var(--surf2)!important;}
.stDataFrame,[data-testid="stDataFrame"]{background-color:var(--surf)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
[data-testid="stDataFrameResizable"] *{color:var(--text)!important;}
.streamlit-expanderHeader{color:var(--text)!important;background-color:var(--surf2)!important;}
.streamlit-expanderContent{background-color:var(--surf)!important;}
[data-testid="stAlert"]{background-color:var(--surf2)!important;border-color:var(--border)!important;}
.stMarkdown *{color:var(--text)!important;}
hr{border-color:var(--border)!important;margin:12px 0!important;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,185,80,.75);}70%{box-shadow:0 0 0 9px rgba(63,185,80,0);}100%{box-shadow:0 0 0 0 rgba(63,185,80,0);}}
.live-dot{display:inline-block;width:10px;height:10px;background:#3fb950;border-radius:50%;animation:pulse 1.4s infinite;margin-right:7px;vertical-align:middle;}
.idle-dot{display:inline-block;width:10px;height:10px;background:#8b949e;border-radius:50%;margin-right:7px;vertical-align:middle;}
.card{background:var(--surf);border:1px solid var(--border);border-radius:8px;padding:16px 20px;margin-bottom:10px;}
.rec-card{background:linear-gradient(135deg,#161b22,#21262d);border:1px solid #30363d;border-left:4px solid #58a6ff;border-radius:8px;padding:14px 18px;margin-bottom:10px;}
.viol-card{background:#1a1010;border:1px solid #5a1f1f;border-radius:6px;padding:8px 14px;margin-bottom:6px;font-size:.84em;}
.status-bar{background:var(--surf);border:1px solid var(--border);border-radius:8px;padding:10px 16px;font-size:.84rem;display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

# ── Thread-safe state store — persists across Streamlit module reloads ──────────
import sys as _sys

_SKEY = "__smart_investing_state__"
if _SKEY not in _sys.modules:
    import types as _types
    _m = _types.ModuleType(_SKEY)
    _m.LOCK = threading.Lock()
    _m.TS   = {
        "running":False,"live_ltp":None,"live_signal":None,"live_position":None,
        "live_pnl":0.0,"live_pnl_pct":0.0,"live_trades":[],"live_df":None,
        "thread_heartbeat":None,"error":None,"log":deque(maxlen=300),
        "squareoff_requested":False,"trailing_sl":None,"phase":1,"locked_sl":None,
        "config":{},"candle_count":0,"daily_pnl":0.0,
        "last_bar_time":None,"live_ew_info":{},
    }
    _sys.modules[_SKEY] = _m

_LOCK = _sys.modules[_SKEY].LOCK
_TS   = _sys.modules[_SKEY].TS

def ts_get(k):
    with _LOCK: return _TS.get(k)
def ts_set(k, v):
    with _LOCK: _TS[k] = v
def ts_update(d):
    with _LOCK: _TS.update(d)
def ts_log(msg):
    t = datetime.now(IST).strftime("%H:%M:%S")
    with _LOCK: _TS["log"].appendleft(f"[{t}]  {msg}")

# ── Session state init ─────────────────────────────────────────────────────────
for _k,_v in {"bt_trades":[],"bt_result":None,"bt_cfg":None,
               "opt_fast":None,"opt_slow":None,"all_trades":[]}.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ── Data fetch — never reuse cached Ticker ─────────────────────────────────────
_YF_H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":"application/json"}

def _parse_chart(data):
    try:
        r=data["chart"]["result"][0]; q=r["indicators"]["quote"][0]
        df=pd.DataFrame({"Open":q["open"],"High":q["high"],"Low":q["low"],
                          "Close":q["close"],"Volume":q["volume"]},
                         index=pd.to_datetime(r["timestamp"],unit="s",utc=True))
        df.index=df.index.tz_convert(IST)
        df.dropna(subset=["Open","High","Low","Close"],inplace=True)
        df=df[df["Close"]>0]; df.sort_index(inplace=True); return df
    except: return None

def fetch_direct(sym,interval,period):
    try:
        r=requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                       params={"interval":interval,"range":period,"includePrePost":"false"},
                       headers=_YF_H,timeout=15)
        r.raise_for_status(); return _parse_chart(r.json())
    except: return None

def fetch_yf(sym,interval,period):
    if not HAS_YF: return None
    try:
        df=yf.download(sym,interval=interval,period=period,auto_adjust=True,
                       progress=False,timeout=15)
        if df is None or df.empty: return None
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        if df.index.tz is None: df.index=df.index.tz_localize("UTC")
        df.index=df.index.tz_convert(IST); df.sort_index(inplace=True); return df
    except: return None

def fetch_ohlcv(sym,interval,period):
    df=fetch_direct(sym,interval,period)
    if df is None or df.empty: df=fetch_yf(sym,interval,period)
    return df

# ── Indicators ─────────────────────────────────────────────────────────────────
def ema(s,p): return s.ewm(span=p,adjust=False,min_periods=1).mean()

def calc_atr(df,p=14):
    h,l,c=df["High"],df["Low"],df["Close"]; pc=c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False,min_periods=1).mean()

def calc_adx(df,p=14):
    h,l,c=df["High"],df["Low"],df["Close"]; ph,pl,pc=h.shift(1),l.shift(1),c.shift(1)
    tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    dmp=pd.Series(np.where((h-ph)>(pl-l),np.maximum(h-ph,0),0),index=df.index)
    dmm=pd.Series(np.where((pl-l)>(h-ph),np.maximum(pl-l,0),0),index=df.index)
    atr_s=tr.ewm(span=p,adjust=False,min_periods=1).mean().replace(0,np.nan)
    dip=100*dmp.ewm(span=p,adjust=False,min_periods=1).mean()/atr_s
    dim=100*dmm.ewm(span=p,adjust=False,min_periods=1).mean()/atr_s
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(span=p,adjust=False,min_periods=1).mean()

def calc_rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(span=p,adjust=False,min_periods=1).mean()
    ls=(-d.clip(upper=0)).ewm(span=p,adjust=False,min_periods=1).mean()
    return 100-100/(1+g/ls.replace(0,np.nan))

def angle_series(fast,slow):
    diff=fast-slow; sl=diff.diff()
    return np.degrees(np.arctan(sl/slow.replace(0,np.nan)*100))

def sup_res(df,w=20):
    return (float(df["Low"].rolling(w,min_periods=1).min().iloc[-1]),
            float(df["High"].rolling(w,min_periods=1).max().iloc[-1]))

# ── Strategies ─────────────────────────────────────────────────────────────────
def strat_ema(df,fast=9,slow=21):
    f,s=ema(df["Close"],fast),ema(df["Close"],slow)
    sig=pd.Series(0,index=df.index,dtype=int)
    sig[(f>s)&(f.shift(1)<=s.shift(1))]=1
    sig[(f<s)&(f.shift(1)>=s.shift(1))]=-1
    return sig

def ema_live_check(df,fast=9,slow=21):
    need=max(fast,slow)*3
    if len(df)<need: return 0
    sig=strat_ema(df.iloc[-need:],fast,slow)
    v=sig.iloc[-2:].values
    if 1 in v: return 1
    if -1 in v: return -1
    return 0

def strat_ant_ema(df,fast=9,slow=21,ant=3):
    f,s=ema(df["Close"],fast),ema(df["Close"],slow)
    g=f-s; gp=g.shift(ant)
    sig=pd.Series(0,index=df.index,dtype=int)
    sig[(g>0)&(gp<=0)]=1; sig[(g<0)&(gp>=0)]=-1
    return sig

def _pivots(s,order=5):
    hi=argrelextrema(s.values,np.greater_equal,order=order)[0]
    lo=argrelextrema(s.values,np.less_equal,   order=order)[0]
    return hi,lo

def _fib(p0,p1):
    rng=abs(p1-p0)
    if p1>p0: return {r:round(p1-rng*r,4) for r in FIB_RATIOS}
    return {r:round(p1+rng*r,4) for r in FIB_RATIOS}

def _merge_pivots(cl,hi,lo):
    raw=([(i,float(cl.iloc[i]),"H") for i in hi]+
         [(i,float(cl.iloc[i]),"L") for i in lo])
    raw.sort(key=lambda x:x[0])
    if not raw: return []
    f=[list(raw[0])]
    for p in raw[1:]:
        if p[2]!=f[-1][2]: f.append(list(p))
        elif p[2]=="H" and p[1]>f[-1][1]: f[-1]=list(p)
        elif p[2]=="L" and p[1]<f[-1][1]: f[-1]=list(p)
    return f

def detect_ew(df,order=5,no_lookahead=True):
    """
    no_lookahead=True: trim last `order` bars so argrelextrema cannot
    use future candles. This makes backtest and live identical.
    """
    empty={"pattern":None,"waves":[],"signal":0,"fib":{},"confidence":0,"in_progress":False}
    if len(df)<order*6: return empty
    # Remove lookahead: argrelextrema needs `order` bars on each side.
    # In live we only have confirmed bars, so trim the last `order` bars.
    safe_df = df.iloc[:-order] if (no_lookahead and len(df)>order*2) else df
    cl=safe_df["Close"]; hi,lo=_pivots(cl,order)
    if len(hi)<2 or len(lo)<2: return empty
    pv=_merge_pivots(cl,hi,lo)
    if len(pv)<4: return empty
    def b5(s): return (s[0][2]=="L" and s[1][2]=="H" and s[2][2]=="L" and s[3][2]=="H"
                       and s[4][2]=="L" and s[5][2]=="H" and s[1][1]>s[0][1]
                       and s[2][1]>s[0][1] and s[3][1]>s[1][1] and s[4][1]>s[2][1]
                       and s[5][1]>s[3][1] and s[4][1]>s[0][1])
    def br5(s): return (s[0][2]=="H" and s[1][2]=="L" and s[2][2]=="H" and s[3][2]=="L"
                        and s[4][2]=="H" and s[5][2]=="L" and s[1][1]<s[0][1]
                        and s[2][1]<s[0][1] and s[3][1]<s[1][1] and s[4][1]<s[2][1]
                        and s[5][1]<s[3][1] and s[4][1]<s[0][1])
    def abc_b(s): return s[0][2]=="L" and s[1][2]=="H" and s[2][2]=="L" and s[2][1]>s[0][1]
    def abc_br(s): return s[0][2]=="H" and s[1][2]=="L" and s[2][2]=="H" and s[2][1]<s[0][1]
    if len(pv)>=6:
        seg=pv[-6:]
        if b5(seg):  return {"pattern":"Bull Impulse 1-5","waves":[(p[0],p[1]) for p in seg],"signal":1,"confidence":82,"in_progress":False,"fib":_fib(seg[0][1],seg[5][1])}
        if br5(seg): return {"pattern":"Bear Impulse 1-5","waves":[(p[0],p[1]) for p in seg],"signal":-1,"confidence":82,"in_progress":False,"fib":_fib(seg[0][1],seg[5][1])}
    if len(pv)>=3:
        s3=pv[-3:]
        if abc_b(s3):  return {"pattern":"ABC Corrective Bull","waves":[(p[0],p[1]) for p in s3],"signal":1,"confidence":62,"in_progress":False,"fib":_fib(s3[0][1],s3[2][1])}
        if abc_br(s3): return {"pattern":"ABC Corrective Bear","waves":[(p[0],p[1]) for p in s3],"signal":-1,"confidence":62,"in_progress":False,"fib":_fib(s3[0][1],s3[2][1])}
    return {"pattern":"In Progress","waves":[(p[0],p[1],p[2]) for p in pv[-4:]],"signal":0,"confidence":30,"in_progress":True,"fib":{}}

def strat_ew(df,order=5):
    sig=pd.Series(0,index=df.index,dtype=int); ew=detect_ew(df,order)
    if ew["signal"]!=0 and ew["waves"]:
        idx=ew["waves"][-1][0]
        if 0<=idx<len(sig): sig.iloc[idx]=ew["signal"]
    return sig

def strat_wave_extrema(df,order=5,min_wp=0.5):
    sig=pd.Series(0,index=df.index,dtype=int); cl=df["Close"]
    if len(cl)<order*3: return sig
    hi=argrelextrema(cl.values,np.greater_equal,order=order)[0]
    lo=argrelextrema(cl.values,np.less_equal,   order=order)[0]
    if len(lo)>=2:
        i,pi=lo[-1],lo[-2]
        if cl.iloc[pi]>0 and abs(cl.iloc[i]-cl.iloc[pi])/cl.iloc[pi]*100>=min_wp: sig.iloc[i]=1
    if len(hi)>=2:
        i,pi=hi[-1],hi[-2]
        if cl.iloc[pi]>0 and abs(cl.iloc[i]-cl.iloc[pi])/cl.iloc[pi]*100>=min_wp: sig.iloc[i]=-1
    return sig

def strat_simple_buy(df):
    s=pd.Series(0,index=df.index,dtype=int)
    if not df.empty: s.iloc[-1]=1
    return s

def strat_simple_sell(df):
    s=pd.Series(0,index=df.index,dtype=int)
    if not df.empty: s.iloc[-1]=-1
    return s

def run_strategy(df,strategy,p):
    if strategy=="EMA Crossover": return strat_ema(df,p.get("fast",9),p.get("slow",21))
    if strategy=="Anticipatory EMA": return strat_ant_ema(df,p.get("fast",9),p.get("slow",21),p.get("anticipate_bars",3))
    if strategy=="Elliott Wave": return strat_ew(df,p.get("ew_order",5))
    if strategy=="Wave Extrema": return strat_wave_extrema(df,p.get("ew_order",5),p.get("min_wave_pct",0.5))
    if strategy=="Simple Buy": return strat_simple_buy(df)
    if strategy=="Simple Sell": return strat_simple_sell(df)
    return pd.Series(0,index=df.index,dtype=int)

# ── Filters ────────────────────────────────────────────────────────────────────
def apply_filters(df,signals,cfg):
    sig=signals.copy()
    dow=cfg.get("filter_dow",[])
    if dow: sig[~df.index.dayofweek.isin(dow)]=0
    if cfg.get("filter_adx",False):
        sig[calc_adx(df,14)<cfg.get("adx_min",20)]=0
    if cfg.get("filter_rsi",False):
        rsi=calc_rsi(df["Close"],14)
        sig[(rsi<cfg.get("rsi_lo",30))|(rsi>cfg.get("rsi_hi",70))]=0
    if cfg.get("filter_time", False):
        try:
            h0,m0=map(int,cfg.get("trade_time_start","09:15").split(":"))
            h1,m1=map(int,cfg.get("trade_time_end","15:20").split(":"))
            mins=df.index.hour*60+df.index.minute
            sig[~((mins>=h0*60+m0)&(mins<=h1*60+m1))]=0
        except: pass
    st=cfg.get("strategy","")
    if cfg.get("filter_angle",False) and st in ("EMA Crossover","Anticipatory EMA"):
        ang=angle_series(ema(df["Close"],cfg.get("fast",9)),ema(df["Close"],cfg.get("slow",21)))
        sig[(ang<cfg.get("angle_min",-90))|(ang>cfg.get("angle_max",90))]=0
    if cfg.get("filter_delta",False) and st in ("EMA Crossover","Anticipatory EMA"):
        fe=ema(df["Close"],cfg.get("fast",9)); se=ema(df["Close"],cfg.get("slow",21))
        dlt=((fe-se)/se.replace(0,np.nan)*100).abs()
        sig[dlt<cfg.get("delta_min",0)]=0
        if cfg.get("delta_max",0)>0: sig[dlt>cfg.get("delta_max",100)]=0
    return sig

# ── SL / Target helpers ────────────────────────────────────────────────────────
def _adaptive_sl_mult(df, idx, window=20):
    """Volatility regime ratio: high vol → wider SL, low vol → tighter."""
    if df is None or idx < window: return 1.5
    recent = df["Close"].iloc[max(0,idx-window):idx+1]
    vol    = recent.std()
    mean   = recent.mean()
    ratio  = vol / mean if mean > 0 else 0.01
    # scale between 1.0x (quiet) and 3.0x (volatile)
    return float(np.clip(ratio * 100, 1.0, 3.0))

def _adv_vol(df, idx, window=14):
    """Average Directional Volatility = mean(|close - prev_close|) * ATR weight."""
    if df is None or idx < 3: return None
    cl  = df["Close"].iloc[max(0,idx-window):idx+1]
    adv = cl.diff().abs().mean()
    return float(adv) if not np.isnan(adv) else None

def calc_sl(entry, side, sl_type, sl_param, df, idx, atr_v, fast=9, slow=21):
    if sl_type == "Custom Points":
        return entry - side * sl_param
    if sl_type == "ATR":
        return entry - side * atr_v * (sl_param if sl_param > 0 else 1.5)
    if sl_type == "Strategy Reverse Signal":
        return entry - side * sl_param
    if sl_type in ("Trailing SL","Auto SL"):
        return entry - side * atr_v * 1.5
    if sl_type == "Trailing SL (3-Phase)":
        return entry - side * atr_v * 1.5
    if sl_type == "Adaptive SL":
        mult = _adaptive_sl_mult(df, idx)
        return entry - side * atr_v * mult
    if sl_type == "ADV Volatility Based":
        adv = _adv_vol(df, idx) or atr_v
        mult = sl_param if sl_param > 0 else 2.0
        return entry - side * adv * mult
    if sl_type == "Volatility Based":
        vol = (df["Close"].iloc[max(0,idx-20):idx+1].std()
               if df is not None else atr_v)
        return entry - side * vol * (sl_param if sl_param > 0 else 2.0)
    if sl_type == "EMA Reverse Crossover":
        if df is not None:
            return float(ema(df["Close"], slow).iloc[min(idx, len(df)-1)])
        return entry - side * atr_v * 1.5
    if sl_type == "Trailing Swing Low/High":
        if df is not None:
            w = max(1, int(sl_param) if sl_param > 0 else 5)
            s = max(0, idx - w)
            return (float(df["Low"].iloc[s:idx+1].min()) if side == 1
                    else float(df["High"].iloc[s:idx+1].max()))
        return entry - side * atr_v * 1.5
    if sl_type == "Candle Low/High":
        if df is not None:
            return (float(df["Low"].iloc[min(idx, len(df)-1)]) if side == 1
                    else float(df["High"].iloc[min(idx, len(df)-1)]))
        return entry - side * atr_v
    if sl_type == "Support/Resistance Trailing":
        if df is not None:
            sup, res = sup_res(df.iloc[:idx+1])
            return sup if side == 1 else res
        return entry - side * atr_v * 1.5
    if sl_type in ("Cost-to-Cost + N Points","Shift to Cost-to-Cost - N pts"):
        return entry - side * atr_v * 1.5
    if sl_type in ("Ratchet SL (50% target → trail)","Exit if SL past N pts stagnant"):
        return entry - side * atr_v * 1.5
    if sl_type == "Smart SL (auto)":
        # Auto: use tighter of ATR*1.5 or recent swing low/high
        base = entry - side * atr_v * 1.5
        if df is not None and idx >= 5:
            swing = (float(df["Low"].iloc[max(0,idx-5):idx+1].min()) if side == 1
                     else float(df["High"].iloc[max(0,idx-5):idx+1].max()))
            # pick the one closer to entry (tighter)
            base = (max(base, swing) if side == 1 else min(base, swing))
        return base
    return entry - side * atr_v * 1.5

def calc_target(entry, side, sl, target_type, tp, rr, atr_v=None, df=None, idx=0):
    if target_type == "Custom Points":
        return entry + side * tp
    if target_type == "Risk:Reward":
        return entry + side * abs(entry - sl) * rr
    if target_type == "ATR Multiple":
        mult = rr if rr > 0 else 2.0
        return entry + side * (atr_v or abs(entry - sl)) * mult
    if target_type == "Adaptive Target":
        mult = _adaptive_sl_mult(df, idx) * rr if df is not None else rr
        return entry + side * abs(entry - sl) * mult
    if target_type in ("Partial Exit + Ride", "Exit if Target fell 50%"):
        return entry + side * tp
    if target_type == "Smart Target (auto)":
        # Auto: 2x ATR from entry
        return entry + side * (atr_v or abs(entry - sl)) * 2.0
    return entry + side * tp

def update_trail(cur, ltp, side, sl_type, sp, entry, atr_v, phase, locked,
                 tgt=None, extra=None):
    """
    extra = dict with trade-level state (best_profit, sl_stagnant_count, etc.)
    Returns: (new_sl, new_phase, new_locked, extra)
    """
    ns, np_, nl = cur, phase, locked
    if extra is None: extra = {}

    profit = (ltp - entry) * side

    if sl_type == "Trailing SL":
        trail = sp if sp > 0 else atr_v * 1.5
        prop  = ltp - side * trail
        ns    = max(cur, prop) if side == 1 else min(cur, prop)

    elif sl_type == "Trailing SL (3-Phase)":
        lp = sp if sp > 0 else atr_v
        if np_ == 1:
            prop = ltp - side * atr_v * 1.5
            ns   = max(cur, prop) if side == 1 else min(cur, prop)
            if profit >= lp:
                np_ = 2; nl = entry + side * lp
                ns  = max(ns, nl) if side == 1 else min(ns, nl)
        elif np_ == 2:
            if nl: ns = max(cur, nl) if side == 1 else min(cur, nl)
            if profit >= lp * 2: np_ = 3
        elif np_ == 3:
            prop = ltp - side * atr_v * 0.5
            ns   = max(cur, prop) if side == 1 else min(cur, prop)

    elif sl_type == "Adaptive SL":
        # Widen in volatile moves, tighten as profit grows
        vol_mult = max(0.5, 1.5 - profit / (atr_v * 3) if atr_v > 0 else 1.5)
        prop = ltp - side * atr_v * vol_mult
        ns   = max(cur, prop) if side == 1 else min(cur, prop)

    elif sl_type == "ADV Volatility Based":
        prop = ltp - side * atr_v * 1.2
        ns   = max(cur, prop) if side == 1 else min(cur, prop)

    elif sl_type in ("Trailing Swing Low/High", "Support/Resistance Trailing"):
        prop = ltp - side * atr_v * 1.2
        ns   = max(cur, prop) if side == 1 else min(cur, prop)

    elif sl_type == "Cost-to-Cost + N Points":
        n = sp if sp > 0 else atr_v * 0.5
        if profit >= n:
            lk = entry + side * n
            ns = max(cur, lk) if side == 1 else min(cur, lk)

    elif sl_type == "Shift to Cost-to-Cost - N pts":
        # When 50% of target reached, shift SL to entry - N pts
        if tgt is not None:
            half_tgt = abs(tgt - entry) * 0.5
            n_pts    = sp if sp > 0 else 5.0
            if profit >= half_tgt and not extra.get("shifted"):
                ns = entry - side * n_pts
                extra["shifted"] = True
                # Still ratchet if already better
                ns = max(cur, ns) if side == 1 else min(cur, ns)

    elif sl_type == "Ratchet SL (50% target → trail)":
        # After 50% target reached: trail by sp points every tick
        if tgt is not None:
            half_tgt = abs(tgt - entry) * 0.5
            trail_pts = sp if sp > 0 else atr_v
            if profit >= half_tgt:
                prop = ltp - side * trail_pts
                ns   = max(cur, prop) if side == 1 else min(cur, prop)

    elif sl_type == "Exit if SL past N pts stagnant":
        # Track best profit; if profit retreats by sp from peak, exit signal
        best = extra.get("best_profit", profit)
        if profit > best:
            extra["best_profit"] = profit
        # SL stays fixed — exit logic handled in main loop via extra flag
        stagnant_threshold = sp if sp > 0 else atr_v * 2
        if best > stagnant_threshold and (best - profit) >= stagnant_threshold * 0.5:
            extra["force_exit"] = True

    elif sl_type == "Smart SL (auto)":
        # Tighten trail as profit increases
        trail = max(atr_v * 0.5, atr_v * 1.5 - profit * 0.1)
        prop  = ltp - side * trail
        ns    = max(cur, prop) if side == 1 else min(cur, prop)

    return ns, np_, nl, extra

# ── Backtest engine ────────────────────────────────────────────────────────────
def _append_trade(pos,ep,reason,ets,trades,daily_map,day,qty):
    side=pos["side"]; ep_=pos.get("entry_time")
    es=ep_.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ep_,datetime) else str(ep_)
    ex=ets.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ets,"strftime") else str(ets)
    try: dur=(pd.Timestamp(ex)-pd.Timestamp(es)).total_seconds()/60
    except: dur=0.0
    pnl=(ep-pos["entry"])*side*qty
    trades.append({"entry_time":es,"exit_time":ex,"side":"BUY" if side==1 else "SELL",
                   "entry":round(pos["entry"],4),"exit":round(ep,4),
                   "sl":round(pos["sl"],4),"target":round(pos["target"],4),
                   "pnl":round(pnl,4),"exit_reason":reason,"duration_m":round(dur,1),
                   "atr":round(pos.get("atr",0),4),"pattern":pos.get("pattern",""),
                   "source":"backtest"})
    daily_map[day]=daily_map.get(day,0.0)+pnl

def run_backtest(df_raw,cfg):
    strategy=cfg["strategy"]; sl_type=cfg["sl_type"]
    sp=float(cfg.get("sl_param",20)); tp=float(cfg.get("target_param",40))
    rr=float(cfg.get("rr_ratio",2)); qty=int(cfg.get("qty",1))
    fast=int(cfg.get("fast",9)); slow=int(cfg.get("slow",21))
    ew_order=int(cfg.get("ew_order",5)); min_wp=float(cfg.get("min_wave_pct",0.5))
    max_loss=float(cfg.get("max_daily_loss",0)); max_profit=float(cfg.get("max_daily_profit",0))
    ant=int(cfg.get("anticipate_bars",3)); simple_imm=strategy in ("Simple Buy","Simple Sell")
    df=df_raw.copy()
    df["EMA_fast"]=ema(df["Close"],fast); df["EMA_slow"]=ema(df["Close"],slow)
    df["ATR"]=calc_atr(df,int(cfg.get("atr_period",14)))
    df["ADX"]=calc_adx(df,14); df["RSI"]=calc_rsi(df["Close"],14)
    if strategy in ("Elliott Wave","Wave Extrema"):
        mb=len(df); step=max(1,mb//6); raw_sig=pd.Series(0,index=df.index,dtype=int)
        for start in range(0,mb-step,step):
            end=min(start+step*2,mb); chunk=df.iloc[start:end]
            csig=run_strategy(chunk,strategy,{"ew_order":ew_order,"min_wave_pct":min_wp})
            sl=start; el=min(start+step,mb); n=el-sl; v=csig.values
            raw_sig.iloc[sl:el]=v[-n:] if len(v)>=n else np.pad(v,(n-len(v),0))
    else:
        raw_sig=run_strategy(df,strategy,{"fast":fast,"slow":slow,"ew_order":ew_order,
                                           "min_wave_pct":min_wp,"anticipate_bars":ant})
    signals=apply_filters(df,raw_sig,{**cfg,"strategy":strategy})
    trades=[]; position=None; equity=0.0; daily_map={}; violations=[]
    for i in range(len(df)):
        row=df.iloc[i]; ts=df.index[i]; sig=signals.iloc[i]; day=ts.date()
        dpnl=daily_map.get(day,0.0)
        if max_loss>0 and dpnl<=-max_loss: position=None; continue
        if max_profit>0 and dpnl>=max_profit: position=None; continue
        if position:
            side=position["side"]; ep=position["entry"]; sl_p=position["sl"]
            tgt=position["target"]; sl_t=position["sl_type"]; sl_par=position["sl_param"]
            atr_v=position["atr"]; trail=position.get("trailing_sl",sl_p)
            ph=position.get("phase",1); lk=position.get("locked_sl",None)
            mid=(row["High"]+row["Low"])/2
            trail,ph,lk,extra_=update_trail(trail,mid,side,sl_t,sl_par,ep,atr_v,ph,lk,
                                              tgt=tgt,extra=position.get("extra",{}))
            position.update({"trailing_sl":trail,"phase":ph,"locked_sl":lk,"extra":extra_})
            if extra_.get("force_exit"):
                _append_trade(position,float(row["Close"]),"Stagnant Exit",ts,trades,daily_map,day,qty)
                equity+=trades[-1]["pnl"]; position=None; continue
            eff=trail if sl_t in TRAILING_SL_TYPES else sl_p
            if sl_t=="EMA Reverse Crossover":
                rev=ema_live_check(df.iloc[:i+1],fast,slow)
                if rev!=0 and rev!=side:
                    _append_trade(position,float(row["Close"]),"EMA Reverse",ts,trades,daily_map,day,qty)
                    equity+=trades[-1]["pnl"]; position=None; continue
            if sl_t=="Strategy Reverse Signal":
                rs=run_strategy(df.iloc[:i+1],strategy,{"fast":fast,"slow":slow,"ew_order":ew_order,"min_wave_pct":min_wp,"anticipate_bars":ant})
                if int(rs.iloc[-1]) not in (0,side):
                    _append_trade(position,float(row["Close"]),"Strategy Reverse",ts,trades,daily_map,day,qty)
                    equity+=trades[-1]["pnl"]; position=None; continue
            exited=False; exit_p=None; exit_r=None
            if side==1:
                if row["Low"]<=eff: exit_p,exit_r,exited=eff,"SL",True
                elif row["High"]>=tgt: exit_p,exit_r,exited=tgt,"Target",True
            else:
                if row["High"]>=eff: exit_p,exit_r,exited=eff,"SL",True
                elif row["Low"]<=tgt: exit_p,exit_r,exited=tgt,"Target",True
            if exited:
                _append_trade(position,exit_p,exit_r,ts,trades,daily_map,day,qty)
                equity+=trades[-1]["pnl"]; position=None
        if not position and sig!=0:
            if simple_imm: e_idx,e_ts,e_p=i,ts,float(row["Close"])
            else:
                e_idx=i+1
                if e_idx>=len(df): continue
                e_ts=df.index[e_idx]; e_p=float(df["Open"].iloc[e_idx])
            atr_v=float(df["ATR"].iloc[min(i,len(df)-1)]); side=1 if sig==1 else -1
            sl_p=calc_sl(e_p,side,sl_type,sp,df,min(e_idx,len(df)-1),atr_v,fast,slow)
            tgt_p=calc_target(e_p,side,sl_p,sl_type,tp,rr)
            ew_inf=detect_ew(df.iloc[:i+1],ew_order) if strategy=="Elliott Wave" else {}
            position={"side":side,"entry":e_p,"sl":sl_p,"target":tgt_p,"sl_type":sl_type,
                      "sl_param":sp,"entry_time":e_ts,"atr":atr_v,"trailing_sl":sl_p,
                      "phase":1,"locked_sl":None,"qty":qty,"pattern":ew_inf.get("pattern","")}
            risk=abs(e_p-sl_p)
            if risk<0.01: violations.append(f"Tiny risk {risk:.4f} @ bar {i} [{e_ts}]")
            if side==1 and sl_p>=e_p: violations.append(f"LONG SL>=entry @ {e_ts}")
            if side==-1 and sl_p<=e_p: violations.append(f"SHORT SL<=entry @ {e_ts}")
    if position:
        ep=float(df.iloc[-1]["Close"]); side=position["side"]
        pnl=(ep-position["entry"])*side*qty
        trades.append({"entry_time":str(position["entry_time"]),"exit_time":str(df.index[-1]),
                       "side":"BUY" if side==1 else "SELL","entry":round(position["entry"],4),
                       "exit":round(ep,4),"sl":round(position["sl"],4),"target":round(position["target"],4),
                       "pnl":round(pnl,4),"exit_reason":"EOD","duration_m":0.0,
                       "atr":round(position.get("atr",0),4),"pattern":position.get("pattern",""),
                       "source":"backtest"})
        equity+=pnl
    eq_arr=np.array([0.0]+[t["pnl"] for t in trades]).cumsum()
    pk=np.maximum.accumulate(eq_arr); dd_arr=pk-eq_arr
    return {"trades":trades,"equity":round(float(eq_arr[-1]),4) if len(eq_arr)>0 else 0.0,
            "equity_arr":eq_arr,"dd_arr":dd_arr,"violations":violations,"signals":signals,"df":df}

# ── Live signal helper ─────────────────────────────────────────────────────────
def get_live_signal(df,strategy,cfg):
    f=cfg.get("fast",9); s=cfg.get("slow",21); o=cfg.get("ew_order",5)
    mw=cfg.get("min_wave_pct",0.5); an=cfg.get("anticipate_bars",3)
    if strategy=="EMA Crossover": return ema_live_check(df,f,s)
    if strategy=="Anticipatory EMA":
        sig=strat_ant_ema(df,f,s,an); v=sig.values[-2:]
        if 1 in v: return 1
        if -1 in v: return -1
        return 0
    if strategy=="Elliott Wave": return detect_ew(df,o)["signal"]
    if strategy=="Wave Extrema":
        sig=strat_wave_extrema(df,o,mw); return int(sig.iloc[-1]) if not sig.empty else 0
    if strategy=="Simple Buy": return 1
    if strategy=="Simple Sell": return -1
    return 0


# ── Live thread ────────────────────────────────────────────────────────────────
def _rec_live(pos, ep, reason, pnl):
    et = pos.get("entry_time")
    es = et.strftime("%Y-%m-%d %H:%M:%S") if isinstance(et, datetime) else str(et)
    with _LOCK:
        _TS["live_trades"].append({
            "entry_time":  es,
            "exit_time":   datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "side":        "BUY" if pos["side"]==1 else "SELL",
            "entry":       round(pos["entry"], 4),
            "exit":        round(ep, 4),
            "sl":          round(pos["sl"], 4),
            "target":      round(pos["target"], 4),
            "pnl":         round(pnl, 4),
            "exit_reason": reason,
            "duration_m":  0.0,
            "atr":         round(pos.get("atr", 0), 4),
            "pattern":     pos.get("pattern", ""),
            "source":      "live",
        })


def _fetch_ltp_only(sym):
    """Lightweight LTP fetch — just the current price."""
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
            params={"interval": "1m", "range": "1d", "includePrePost": "false"},
            headers=_YF_H, timeout=5)
        r.raise_for_status()
        meta = r.json()["chart"]["result"][0]["meta"]
        p = meta.get("regularMarketPrice") or meta.get("previousClose")
        return float(p) if p else None
    except:
        return None


def live_thread():
    """
    Simple loop:
      1. Fetch candles ONCE at start → compute signal
      2. Every 1.5 s: fetch LTP only → update PnL / check SL-Target
      3. Every 60 s: refresh candles → re-check signal
    """
    ts_log("▶ Thread started")
    try:
        cfg      = ts_get("config") or {}
        sym      = cfg["symbol"]
        iv       = cfg["interval"]
        per      = cfg["period"]
        strat    = cfg["strategy"]
        sl_type  = cfg["sl_type"]
        sp       = float(cfg.get("sl_param",     20))
        tp_pts   = float(cfg.get("target_param", 40))
        rr       = float(cfg.get("rr_ratio",      2))
        qty      = int(cfg.get("qty",   1))
        fast     = int(cfg.get("fast",  9))
        slow     = int(cfg.get("slow", 21))
        atr_p    = int(cfg.get("atr_period", 14))
        max_loss = float(cfg.get("max_daily_loss",   0))
        max_prof = float(cfg.get("max_daily_profit", 0))
        t_start  = cfg.get("trade_time_start", "09:15")
        t_end    = cfg.get("trade_time_end",   "15:20")
        simple   = strat in ("Simple Buy", "Simple Sell")
        ts_log(f"Config OK: {sym} | {strat} | {sl_type} | qty={qty}")
    except Exception as e:
        ts_set("error", f"Config parse failed: {e}\n" + traceback.format_exc())
        ts_set("running", False)
        ts_log(f"FATAL config error: {e}")
        return

    def in_window():
        try:
            now = datetime.now(IST)
            h0,m0 = map(int, t_start.split(":"))
            h1,m1 = map(int, t_end.split(":"))
            m = now.hour*60 + now.minute
            return h0*60+m0 <= m <= h1*60+m1
        except:
            return True

    def dow_ok():
        dows = cfg.get("filter_dow", [])
        return not dows or datetime.now(IST).weekday() in dows

    # ── Step 1: Fetch initial candle data ─────────────────────────────────────
    ts_log(f"Fetching {sym} {iv} {per} candles…")
    df = fetch_ohlcv(sym, iv, per)
    if df is None or df.empty:
        ts_log("⚠ Initial candle fetch failed — will retry on loop")
        df = None
    else:
        df["EMA_fast"] = ema(df["Close"], fast)
        df["EMA_slow"] = ema(df["Close"], slow)
        df["ATR"]      = calc_atr(df, atr_p)
        ts_set("live_df", df)
        ts_log(f"Candles loaded: {len(df)} bars")

    # ── Step 2: Compute initial signal ────────────────────────────────────────
    pending_signal = 0
    if simple:
        pending_signal = 1 if strat == "Simple Buy" else -1
        ts_log(f"Signal: {strat} → immediate entry")
    elif df is not None:
        pending_signal = get_live_signal(df, strat, cfg)
        ts_log(f"Signal from candles: {pending_signal}")

    ts_set("live_signal", pending_signal)

    position  = None
    daily_pnl = 0.0
    last_date = datetime.now(IST).date()
    last_candle_refresh = time.time()

    # ── Main loop: runs every 1.5 s ───────────────────────────────────────────
    while ts_get("running"):
        try:
            ts_set("thread_heartbeat", datetime.now(IST).isoformat())
            now_ist = datetime.now(IST)

            # Daily reset
            today = now_ist.date()
            if today != last_date:
                daily_pnl = 0.0
                last_date = today
                ts_log("Daily PnL reset")

            # Daily caps
            if max_loss > 0 and daily_pnl <= -max_loss:
                time.sleep(1.5); continue
            if max_prof > 0 and daily_pnl >= max_prof:
                time.sleep(1.5); continue

            # ── Fetch LTP (fast, every tick) ──────────────────────────────────
            ltp = _fetch_ltp_only(sym)
            if ltp is None or ltp <= 0:
                # fallback to last known
                ltp = ts_get("live_ltp")
                if not ltp:
                    ts_log("LTP unavailable, waiting…")
                    time.sleep(1.5); continue

            ts_update({
                "live_ltp":   ltp,
                "daily_pnl":  round(daily_pnl, 4),
                "candle_count": (ts_get("candle_count") or 0) + 1,
            })

            # ── Push live strategy indicator values every tick ─────────────────
            strat_info = {"strategy": strat, "fast": fast, "slow": slow}
            if df is not None and not df.empty:
                try:
                    fe_val = round(float(df["EMA_fast"].iloc[-1]), 4) if "EMA_fast" in df.columns else None
                    se_val = round(float(df["EMA_slow"].iloc[-1]),  4) if "EMA_slow" in df.columns else None
                    atr_val = round(float(df["ATR"].iloc[-1]),       4) if "ATR"      in df.columns else None
                    gap = round(fe_val - se_val, 4) if fe_val and se_val else None
                    trend = "Fast > Slow (Bullish)" if gap and gap > 0 else "Fast < Slow (Bearish)" if gap else "—"
                    # Crossover angle
                    ang_val = None
                    try:
                        ang_s   = angle_series(df["EMA_fast"], df["EMA_slow"])
                        ang_val = round(float(ang_s.iloc[-1]), 2)
                    except Exception:
                        pass
                    strat_info.update({
                        "ema_fast_val": fe_val, "ema_slow_val": se_val,
                        "atr_val": atr_val, "ema_gap": gap, "trend": trend,
                        "angle": ang_val, "bars": len(df),
                        "last_bar": str(df.index[-1])[:16],
                    })
                    # EW info
                    if strat == "Elliott Wave":
                        ew_now = detect_ew(df, cfg.get("ew_order", 5))
                        strat_info["ew_pattern"]    = ew_now.get("pattern") or "None detected"
                        strat_info["ew_confidence"] = ew_now.get("confidence", 0)
                        strat_info["ew_signal"]     = ew_now.get("signal", 0)
                except Exception:
                    pass
            ts_set("live_strat_info", strat_info)

            # ── Refresh candles every 60 s (or every tick for EMA strategies) ──
            ema_strat = strat in ("EMA Crossover","Anticipatory EMA")
            # EMA crossover: check every tick on latest candles so we never miss
            do_candle_refresh = (
                time.time() - last_candle_refresh >= 60
                or (ema_strat and time.time() - last_candle_refresh >= 5)
            )
            if do_candle_refresh:
                new_df = fetch_ohlcv(sym, iv, per)
                if new_df is not None and not new_df.empty:
                    new_df["EMA_fast"] = ema(new_df["Close"], fast)
                    new_df["EMA_slow"] = ema(new_df["Close"], slow)
                    new_df["ATR"]      = calc_atr(new_df, atr_p)
                    df = new_df
                    ts_set("live_df", df)
                    last_candle_refresh = time.time()
                    if not simple and not position:
                        pending_signal = get_live_signal(df, strat, cfg)
                        ts_set("live_signal", pending_signal)
                        if pending_signal != 0:
                            ts_log(f"Signal detected: {'BUY' if pending_signal==1 else 'SELL'} [{strat}]")

            # ── Squareoff ─────────────────────────────────────────────────────
            if ts_get("squareoff_requested") and position:
                pnl = (ltp - position["entry"]) * position["side"] * qty
                _rec_live(position, ltp, "Squareoff", pnl)
                daily_pnl += pnl
                position = None
                ts_update({
                    "daily_pnl":           round(daily_pnl, 4),
                    "live_pnl":            round(daily_pnl, 4),
                    "live_position":       None,
                    "trailing_sl":         None,
                    "squareoff_requested": False,
                })
                ts_log(f"Squareoff @ {ltp:.4f}  PnL={pnl:+.4f}")
                time.sleep(1.5); continue

            # ── Manage open position ──────────────────────────────────────────
            if position:
                side   = position["side"]
                entry  = position["entry"]
                sl_p   = position["sl"]
                tgt_p  = position["target"]
                sl_t   = position["sl_type"]
                sp_    = position["sl_param"]
                atr_   = position["atr"]
                trail  = position.get("trailing_sl", sl_p)
                phase  = position.get("phase", 1)
                locked = position.get("locked_sl", None)

                # Update trailing SL using live LTP
                _extra = position.get("extra", {})
                trail, phase, locked, _extra = update_trail(
                    trail, ltp, side, sl_t, sp_, entry, atr_, phase, locked,
                    tgt=tgt_p, extra=_extra)
                position.update({"trailing_sl": trail, "phase": phase,
                                  "locked_sl": locked, "extra": _extra})
                if _extra.get("force_exit"):
                    pnl = round((ltp - entry) * side * qty, 4)
                    _rec_live(position, ltp, "Stagnant Exit", pnl)
                    daily_pnl += pnl; position = None
                    ts_update({"daily_pnl": round(daily_pnl,4), "live_pnl": round(daily_pnl,4),
                               "live_position": None, "trailing_sl": None})
                    ts_log(f"Stagnant exit @ {ltp:.4f}  PnL={pnl:+.4f}")
                    time.sleep(1.5); continue

                eff_sl = trail if sl_t in TRAILING_SL_TYPES else sl_p
                unreal = round((ltp - entry) * side * qty, 4)

                # Push live position to UI
                ts_update({
                    "live_pnl":     round(daily_pnl + unreal, 4),
                    "daily_pnl":    round(daily_pnl, 4),
                    "trailing_sl":  round(trail, 4),
                    "phase":        phase,
                    "live_position": {
                        "side":           side,
                        "entry":          round(entry, 4),
                        "sl":             round(sl_p, 4),
                        "target":         round(tgt_p, 4),
                        "eff_sl":         round(eff_sl, 4),
                        "unrealized_pnl": unreal,
                        "pattern":        position.get("pattern", ""),
                        "entry_time":     str(position.get("entry_time", "")),
                    },
                })

                # Check SL
                if (side == 1 and ltp <= eff_sl) or (side == -1 and ltp >= eff_sl):
                    pnl = round((eff_sl - entry) * side * qty, 4)
                    _rec_live(position, eff_sl, "SL", pnl)
                    daily_pnl += pnl
                    position = None
                    ts_update({"daily_pnl": round(daily_pnl,4), "live_pnl": round(daily_pnl,4),
                               "live_position": None, "trailing_sl": None})
                    ts_log(f"SL hit @ {eff_sl:.4f}  PnL={pnl:+.4f}")
                    # After SL, re-check signal for next entry
                    if simple:
                        pending_signal = 1 if strat == "Simple Buy" else -1
                    elif df is not None:
                        pending_signal = get_live_signal(df, strat, cfg)
                    ts_set("live_signal", pending_signal)
                    time.sleep(1.5); continue

                # Check Target
                tgt_hit = (side == 1 and ltp >= tgt_p) or (side == -1 and ltp <= tgt_p)

                # "Exit if Target fell 50%" — if we hit target then fell back 50%
                target_type_pos = position.get("target_type","Custom Points")
                if target_type_pos == "Exit if Target fell 50%":
                    best_profit = position.get("extra",{}).get("best_profit",0)
                    if profit > best_profit:
                        if "extra" not in position: position["extra"] = {}
                        position["extra"]["best_profit"] = profit
                    half_tgt_profit = abs(tgt_p - entry) * 0.5
                    if best_profit >= half_tgt_profit and profit <= best_profit * 0.5:
                        tgt_hit = True

                if tgt_hit:
                    # Partial Exit: exit partial_pct % now, ride rest
                    p_pct = position.get("partial_pct", 100)
                    if target_type_pos == "Partial Exit + Ride" and not position.get("partial_done"):
                        exit_qty  = max(1, int(round(qty * p_pct / 100)))
                        ride_qty  = max(0, qty - exit_qty)
                        pnl = round((tgt_p - entry) * side * exit_qty, 4)
                        _rec_live({**position,"qty":exit_qty}, tgt_p, f"Partial Exit {p_pct}%", pnl)
                        daily_pnl += pnl
                        if ride_qty > 0:
                            position["qty"]          = ride_qty
                            position["partial_done"] = True
                            # New target = entry + 2x original distance
                            position["target"]       = entry + side * abs(tgt_p - entry) * 2
                            position["extra"]        = position.get("extra", {})
                            ts_log(f"Partial exit {exit_qty} @ {tgt_p:.4f}  Riding {ride_qty}")
                            ts_update({"daily_pnl": round(daily_pnl,4), "live_pnl": round(daily_pnl,4)})
                            time.sleep(1.5); continue
                        else:
                            position = None
                    else:
                        pnl = round((tgt_p - entry) * side * qty, 4)
                        _rec_live(position, tgt_p, "Target", pnl)
                        daily_pnl += pnl
                        position = None
                    ts_update({"daily_pnl": round(daily_pnl,4), "live_pnl": round(daily_pnl,4),
                               "live_position": None, "trailing_sl": None})
                    ts_log(f"Target hit @ {tgt_p:.4f}  PnL={pnl:+.4f}")
                    if simple:
                        pending_signal = 1 if strat == "Simple Buy" else -1
                    elif df is not None:
                        pending_signal = get_live_signal(df, strat, cfg)
                    ts_set("live_signal", pending_signal)
                    time.sleep(1.5); continue

            # ── Enter new position ────────────────────────────────────────────
            if not position and pending_signal != 0:
                # Time/day filters only apply to EMA strategies in auto mode
                # EW / Wave Extrema fire on confirmed pattern — respect signal immediately
                # Simple Buy/Sell always enter
                if simple:
                    can_enter = True
                elif strat in ("Elliott Wave", "Wave Extrema"):
                    can_enter = True   # pattern-based: enter on confirmed signal
                else:
                    time_ok = (not cfg.get("filter_time", False)) or in_window()
                    day_ok_ = (not cfg.get("filter_dow",  [])) or dow_ok()
                    can_enter = time_ok and day_ok_

                if not can_enter:
                    reason = "Outside trade window" if not in_window() else "Day filter active"
                    ts_set("live_entry_blocked", reason)
                else:
                    ts_set("live_entry_blocked", None)
                    side  = 1 if pending_signal == 1 else -1
                    atr_v = (float(df["ATR"].iloc[-1])
                             if df is not None and not df.empty else abs(ltp * 0.005))
                    sl_p  = calc_sl(ltp, side, sl_type, sp,
                                    df, (len(df)-1) if df is not None else 0,
                                    atr_v, fast, slow)
                    tgt_p = calc_target(ltp, side, sl_p, sl_type, tp_pts, rr)
                    ew_inf = (detect_ew(df, cfg.get("ew_order", 5))
                              if strat == "Elliott Wave" and df is not None else {})
                    target_type = cfg.get("target_type", "Custom Points")
                    partial_pct  = int(cfg.get("partial_pct", 70))
                    smart_on     = cfg.get("smart_sl_target", False)
                    if smart_on:
                        # Override: auto SL and target from ATR
                        sl_p  = entry - side * atr_v * 1.5
                        tgt_p = entry + side * atr_v * 2.5
                    position = {
                        "side": side, "entry": ltp, "sl": sl_p, "target": tgt_p,
                        "sl_type": sl_type, "sl_param": sp,
                        "target_type": target_type, "partial_pct": partial_pct,
                        "entry_time": datetime.now(IST),
                        "atr": atr_v, "qty": qty,
                        "trailing_sl": sl_p, "phase": 1, "locked_sl": None,
                        "pattern": ew_inf.get("pattern", ""),
                        "extra": {},
                    }
                    ts_update({"live_ew_info": ew_inf})
                    ts_log(f"{'BUY' if side==1 else 'SELL'} @ {ltp:.4f}  "
                           f"SL={sl_p:.4f}  TGT={tgt_p:.4f}")
                    if not simple:
                        pending_signal = 0
                        ts_set("live_signal", 0)
            else:
                ts_set("live_entry_blocked", None)

            # ── Auto EOD exit ─────────────────────────────────────────────
            if position and cfg.get("filter_time", False):
                try:
                    now2 = datetime.now(IST)
                    h1e, m1e = map(int, t_end.split(":"))
                    if now2.hour * 60 + now2.minute >= h1e * 60 + m1e:
                        pnl = round((ltp - position["entry"]) * position["side"] * qty, 4)
                        _rec_live(position, ltp, "EOD Auto Exit", pnl)
                        daily_pnl += pnl
                        position = None
                        ts_update({"daily_pnl": round(daily_pnl,4), "live_pnl": round(daily_pnl,4),
                                   "live_position": None, "trailing_sl": None})
                        ts_log(f"Auto EOD exit @ {ltp:.4f}  PnL={pnl:+.4f}")
                except Exception:
                    pass

        except Exception as exc:
            ts_log(f"ERROR: {exc}")
            ts_set("error", traceback.format_exc())

        time.sleep(1.5)

    ts_log("⏹ Thread stopped")
    ts_update({"live_position": None, "trailing_sl": None, "running": False})

# ── Analysis ───────────────────────────────────────────────────────────────────
def build_analysis(trades):
    if not trades: return {}
    df=pd.DataFrame(trades)
    df["entry_dt"]=pd.to_datetime(df["entry_time"],errors="coerce")
    df["exit_dt"] =pd.to_datetime(df["exit_time"], errors="coerce")
    w=df[df["pnl"]>0]; lo=df[df["pnl"]<=0]
    total=round(float(df["pnl"].sum()),4); wr=len(w)/len(df)*100 if len(df)>0 else 0
    aw=float(w["pnl"].mean()) if not w.empty else 0.0
    al=float(lo["pnl"].mean()) if not lo.empty else 0.0
    pfd=abs(float(lo["pnl"].sum())); pf=abs(float(w["pnl"].sum()))/pfd if pfd>0 else np.inf
    mw=float(w["pnl"].median()) if not w.empty else 0.0
    ml=float(lo["pnl"].median()) if not lo.empty else 0.0
    eq=np.array([0.0]+list(df["pnl"].cumsum())); pk=np.maximum.accumulate(eq); dd=pk-eq
    if not lo.empty and "entry" in lo.columns and "sl" in lo.columns:
        rsl=float((lo["entry"]-lo["sl"]).abs().median())
    else: rsl=0.0
    if not w.empty and "entry" in w.columns and "exit" in w.columns:
        sgn=1 if df["side"].iloc[0]=="BUY" else -1
        rtgt=float(((w["exit"]-w["entry"])*sgn).abs().median())
    else: rtgt=0.0
    rg=df.groupby("exit_reason")["pnl"].agg(["sum","count"]).reset_index()
    df["date"]=df["entry_dt"].dt.date; dp=df.groupby("date")["pnl"].sum()
    return {"df":df,"total_pnl":total,"win_rate":wr,"avg_win":aw,"avg_loss":al,
            "profit_factor":pf,"max_drawdown":float(dd.max()),"total_trades":len(df),
            "winners":len(w),"losers":len(lo),"median_win":mw,"median_loss":ml,
            "rec_sl_pts":round(rsl,4),"rec_tgt_pts":round(rtgt,4),
            "equity_arr":eq,"dd_arr":dd,"reason_grp":rg,"daily_pnl":dp}

# ── Plotly theme ───────────────────────────────────────────────────────────────
_CL=dict(paper_bgcolor="#0d1117",plot_bgcolor="#161b22",
         font=dict(family="Space Mono",color="#e6edf3",size=11),
         xaxis=dict(gridcolor="#30363d",showgrid=True,zeroline=False,color="#e6edf3"),
         yaxis=dict(gridcolor="#30363d",showgrid=True,zeroline=False,color="#e6edf3"),
         margin=dict(l=52,r=24,t=46,b=42),
         legend=dict(bgcolor="#161b22",bordercolor="#30363d",font=dict(color="#e6edf3",size=10)))

def candle_chart(df,signals=None,trades=None,ew_info=None,title=""):
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.60,0.20,0.20],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="Price",increasing_line_color="#3fb950",decreasing_line_color="#f85149",
        increasing_fillcolor="#3fb950",decreasing_fillcolor="#f85149"),row=1,col=1)
    if "EMA_fast" in df.columns: fig.add_trace(go.Scatter(x=df.index,y=df["EMA_fast"],line=dict(color="#58a6ff",width=1.4),name="EMA Fast"),row=1,col=1)
    if "EMA_slow" in df.columns: fig.add_trace(go.Scatter(x=df.index,y=df["EMA_slow"],line=dict(color="#d29922",width=1.4),name="EMA Slow"),row=1,col=1)
    if signals is not None:
        buys=signals[signals==1]; sells=signals[signals==-1]
        if not buys.empty:
            by=df["Low"].reindex(buys.index,fill_value=df["Low"].min())*0.999
            fig.add_trace(go.Scatter(x=buys.index,y=by,mode="markers",marker=dict(symbol="triangle-up",size=11,color="#3fb950"),name="BUY"),row=1,col=1)
        if not sells.empty:
            sy=df["High"].reindex(sells.index,fill_value=df["High"].max())*1.001
            fig.add_trace(go.Scatter(x=sells.index,y=sy,mode="markers",marker=dict(symbol="triangle-down",size=11,color="#f85149"),name="SELL"),row=1,col=1)
    if trades:
        for t in trades[:150]:
            c="rgba(63,185,80,.08)" if t["pnl"]>0 else "rgba(248,81,73,.08)"
            try: fig.add_vrect(x0=pd.Timestamp(t["entry_time"]),x1=pd.Timestamp(t["exit_time"]),fillcolor=c,line_width=0,row=1,col=1)
            except: pass
    if ew_info and ew_info.get("fib"):
        fc={0.236:"#58a6ff",0.382:"#d29922",0.500:"#e6edf3",0.618:"#d29922",0.786:"#bc8cff",1.000:"#f85149"}
        for r,lvl in ew_info["fib"].items():
            if r in fc: fig.add_hline(y=lvl,line_color=fc[r],line_dash="dot",line_width=1,annotation_text=f"Fib {r}",annotation_font_color=fc[r],row=1,col=1)
    vc=["#3fb950" if c>=o else "#f85149" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,name="Volume",opacity=0.7),row=2,col=1)
    if "ATR" in df.columns: fig.add_trace(go.Scatter(x=df.index,y=df["ATR"],line=dict(color="#bc8cff",width=1.2),name="ATR"),row=3,col=1)
    fig.update_layout(title=title,xaxis_rangeslider_visible=False,height=640,**_CL,showlegend=True)
    return fig

def eq_chart(eq,dd):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.6,0.4],vertical_spacing=0.04)
    x=list(range(len(eq)))
    fig.add_trace(go.Scatter(x=x,y=eq,line=dict(color="#58a6ff",width=2),fill="tozeroy",fillcolor="rgba(88,166,255,.10)",name="Equity"),row=1,col=1)
    fig.add_hline(y=0,line_color="#30363d",row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=-dd,line=dict(color="#f85149",width=1.5),fill="tozeroy",fillcolor="rgba(248,81,73,.12)",name="Drawdown"),row=2,col=1)
    fig.update_layout(title="Equity Curve & Drawdown",height=400,**_CL,showlegend=True)
    return fig

def heatmap_chart(df_t):
    df=df_t.copy()
    df["dow"]=pd.to_datetime(df["entry_time"],errors="coerce").dt.day_name()
    df["hour"]=pd.to_datetime(df["entry_time"],errors="coerce").dt.hour
    pv=df.pivot_table(values="pnl",index="dow",columns="hour",aggfunc="sum",fill_value=0)
    order=[d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] if d in pv.index]
    pv=pv.reindex(order)
    fig=go.Figure(go.Heatmap(z=pv.values,x=[f"{h}:00" for h in pv.columns],y=pv.index.tolist(),zmid=0,
        colorscale=[[0,"#f85149"],[0.5,"#161b22"],[1,"#3fb950"]],
        text=np.round(pv.values,1),texttemplate="%{text}",colorbar=dict(tickfont=dict(color="#e6edf3"))))
    fig.update_layout(title="PnL Heatmap  (Day × Hour IST)",height=310,**_CL)
    return fig

def ls_chart(df_t):
    grp=df_t.groupby("side")["pnl"].agg(["sum","count","mean"]).reset_index()
    fig=go.Figure(go.Bar(x=grp["side"],y=grp["sum"],
        text=[f"n={int(c)}<br>avg={a:.2f}" for c,a in zip(grp["count"],grp["mean"])],textposition="outside",
        marker_color=["#3fb950" if v>=0 else "#f85149" for v in grp["sum"]]))
    fig.update_layout(title="Long vs Short PnL",height=300,**_CL); return fig

def reason_chart(rg):
    colors={"SL":"#f85149","Target":"#3fb950","EOD":"#58a6ff","Squareoff":"#d29922","Strategy Reverse":"#bc8cff","EMA Reverse":"#e3b341"}
    fig=go.Figure(go.Bar(x=rg["exit_reason"],y=rg["sum"],
        text=[f"n={int(c)}" for c in rg["count"]],textposition="outside",
        marker_color=[colors.get(r,"#8b949e") for r in rg["exit_reason"]]))
    fig.update_layout(title="PnL by Exit Reason",height=280,**_CL); return fig

def daily_chart(dp):
    vc=["#3fb950" if v>=0 else "#f85149" for v in dp.values]
    fig=go.Figure(go.Bar(x=[str(d) for d in dp.index],y=dp.values,marker_color=vc,
        text=np.round(dp.values,2),textposition="outside"))
    fig.update_layout(title="Daily PnL",height=280,**_CL); return fig

def ang_scatter(df,fast,slow):
    fe=ema(df["Close"],fast); se=ema(df["Close"],slow)
    ang=angle_series(fe,se); dlt=(fe-se)/se.replace(0,np.nan)*100
    fig=go.Figure(go.Scatter(x=dlt.values,y=ang.values,mode="markers",
        marker=dict(color=df["Close"].values,colorscale="Plasma",size=4,opacity=0.55,
                    colorbar=dict(title="Close",tickfont=dict(color="#e6edf3")))))
    fig.update_layout(title="Crossover Angle vs EMA Δ%",xaxis_title="Delta %",yaxis_title="Angle °",height=340,**_CL)
    return fig

def ew_chart(trades):
    df=pd.DataFrame(trades); fig=go.Figure()
    if "pattern" in df.columns:
        df=df[df["pattern"].fillna("").ne("")]
        if not df.empty:
            g=df.groupby("pattern")["pnl"].agg(["sum","count"]).reset_index()
            fig=go.Figure(go.Bar(x=g["pattern"],y=g["sum"],text=[f"n={int(c)}" for c in g["count"]],textposition="outside",
                marker_color=["#3fb950" if v>=0 else "#f85149" for v in g["sum"]]))
    fig.update_layout(title="EW Pattern PnL Breakdown",height=280,**_CL); return fig

def dur_chart(df_t):
    fig=go.Figure(go.Histogram(x=df_t["duration_m"],nbinsx=30,marker_color="#58a6ff",opacity=0.8))
    fig.update_layout(title="Trade Duration (min)",xaxis_title="Minutes",height=280,**_CL)
    return fig

# ── Sidebar config ─────────────────────────────────────────────────────────────
def sidebar_config():
    with st.sidebar:
        st.markdown("<h2 style='font-family:Syne;color:#e6edf3'>⚙ Config</h2>",unsafe_allow_html=True)
        di=NIFTY50.index("KAYNES.NS") if "KAYNES.NS" in NIFTY50 else 0
        sym=st.selectbox("Symbol",NIFTY50,index=di)
        cust=st.text_input("Custom Symbol",placeholder="AAPL, ETH-USD…")
        if cust.strip(): sym=cust.strip().upper()
        interval=st.selectbox("Interval",TIMEFRAMES,index=0)
        period=st.selectbox("Period",PERIODS,index=2)
        st.markdown("---")
        strategy=st.selectbox("Strategy",STRATEGIES)
        qty=st.number_input("Quantity",min_value=1,value=1,step=1)
        st.markdown("**EMA Parameters**")
        df_fast=int(st.session_state.get("opt_fast") or 9)
        df_slow=int(st.session_state.get("opt_slow") or 21)
        fast=st.number_input("Fast EMA",min_value=2,value=df_fast,step=1)
        slow=st.number_input("Slow EMA",min_value=3,value=df_slow,step=1)
        ant=3
        if strategy=="Anticipatory EMA": ant=st.number_input("Anticipate Bars",min_value=1,value=3,step=1)
        ew_o=5; min_wp=0.5
        if strategy in ("Elliott Wave","Wave Extrema"):
            ew_o=st.number_input("EW Order",min_value=2,value=5,step=1)
            min_wp=st.number_input("Min Wave %",min_value=0.0,value=0.5,step=0.1)
        st.markdown("---")
        st.markdown("**Stop-Loss**")
        sl_type  = st.selectbox("SL Type",  SL_TYPES,  key="sl_type_sel")
        sl_param = st.number_input("SL Points / Param", min_value=0.0, value=20.0, step=0.5)

        st.markdown("**Target**")
        target_type = st.selectbox("Target Type", TARGET_TYPES, key="tgt_type_sel")
        tp          = st.number_input("Target Points", min_value=0.0, value=40.0, step=0.5)
        rr          = st.number_input("R:R Ratio",     min_value=0.1, value=2.0,  step=0.5)
        if target_type == "Partial Exit + Ride":
            partial_pct = st.number_input("Exit % at 1st target", min_value=10, max_value=90, value=70, step=5)
        else:
            partial_pct = 70

        atr_p = st.number_input("ATR Period", min_value=2, value=14, step=1)

        smart_sl_target = st.checkbox("🧠 Smart SL + Target (auto, overrides above)", value=False)
        if smart_sl_target:
            st.caption("SL = ATR×1.5 below/above entry. Target = ATR×2.5 above/below entry. Fully automatic.")
        st.markdown("---")
        st.markdown("**Filters** *(all disabled by default)*")

        # Day-of-week filter
        filter_dow_on = st.checkbox("Day-of-Week Filter", value=False)
        dow_ints = []
        if filter_dow_on:
            day_opts=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            sel_days=st.multiselect("Days",day_opts,default=["Mon","Tue","Wed","Thu","Fri"])
            dow_map={d:i for i,d in enumerate(day_opts)}
            dow_ints=[dow_map[d] for d in sel_days]

        # Trade time window
        filter_time=st.checkbox("Trade Time Window", value=False)
        t_start,t_end="09:15","15:30"
        if filter_time:
            fc1,fc2=st.columns(2)
            t_start=fc1.text_input("Start HH:MM",value="09:15",key="ts_start")
            t_end  =fc2.text_input("End   HH:MM",value="15:30",key="ts_end")

        # ADX filter
        filter_adx=st.checkbox("ADX Filter", value=False)
        adx_min=20
        if filter_adx:
            adx_min=st.number_input("ADX Min",0,100,20)

        # RSI filter
        filter_rsi=st.checkbox("RSI Filter", value=False)
        rsi_lo,rsi_hi=30,70
        if filter_rsi:
            fc1,fc2=st.columns(2)
            rsi_lo=fc1.number_input("RSI Low", 0,100,30)
            rsi_hi=fc2.number_input("RSI High",0,100,70)

        # Crossover angle filter
        filter_ang=st.checkbox("Crossover Angle Filter", value=False)
        amin,amax=-90.0,90.0
        if filter_ang:
            fc1,fc2=st.columns(2)
            amin=fc1.number_input("Angle Min°",value=-90.0)
            amax=fc2.number_input("Angle Max°",value=90.0)

        # EMA delta filter
        filter_dlt=st.checkbox("EMA Delta Filter", value=False)
        dmin,dmax=0.0,0.0
        if filter_dlt:
            fc1,fc2=st.columns(2)
            dmin=fc1.number_input("Delta Min%",value=0.0,step=0.1)
            dmax=fc2.number_input("Delta Max%",value=5.0,step=0.1)

        # Daily PnL caps
        filter_caps=st.checkbox("Daily PnL Caps", value=False)
        max_loss,max_profit=0.0,0.0
        if filter_caps:
            fc1,fc2=st.columns(2)
            max_loss  =fc1.number_input("Max Loss",  0.0,step=100.0)
            max_profit=fc2.number_input("Max Profit",0.0,step=100.0)
        st.markdown("---")
        st.markdown(f"<div style='font-size:.74rem;color:#8b949e;text-align:center'>{datetime.now(IST).strftime('%d %b %Y  %H:%M:%S IST')}</div>",unsafe_allow_html=True)
    return {"symbol":sym,"interval":interval,"period":period,"strategy":strategy,"qty":int(qty),
            "fast":int(fast),"slow":int(slow),"anticipate_bars":int(ant),"ew_order":int(ew_o),
            "min_wave_pct":float(min_wp),"sl_type":sl_type,"sl_param":float(sl_param),
            "rr_ratio":float(rr),"target_param":float(tp),"target_type":target_type,
            "partial_pct":int(partial_pct),"smart_sl_target":smart_sl_target,"atr_period":int(atr_p),
            "filter_dow":dow_ints,"filter_adx":filter_adx,"adx_min":int(adx_min),
            "filter_rsi":filter_rsi,"rsi_lo":int(rsi_lo),"rsi_hi":int(rsi_hi),
            "filter_angle":filter_ang,"angle_min":float(amin),"angle_max":float(amax),
            "filter_delta":filter_dlt,"delta_min":float(dmin),"delta_max":float(dmax),
            "filter_time":filter_time,"trade_time_start":t_start,"trade_time_end":t_end,
            "max_daily_loss":float(max_loss),"max_daily_profit":float(max_profit)}

# ── Merge helper ───────────────────────────────────────────────────────────────
def merge_all():
    bt=st.session_state.get("bt_trades",[]); lv=list(ts_get("live_trades") or [])
    merged=bt+lv; seen=set(); unique=[]
    for t in merged:
        k=(t.get("entry_time",""),t.get("exit_time",""),t.get("side",""),t.get("entry",""))
        if k not in seen: seen.add(k); unique.append(t)
    st.session_state["all_trades"]=unique

# ── PnL styler ─────────────────────────────────────────────────────────────────
def _ps(v):
    if isinstance(v,(int,float)) and not pd.isna(v):
        return "color:#3fb950;font-weight:700" if v>0 else "color:#f85149"
    return ""

# ── Tab 1: Backtest ────────────────────────────────────────────────────────────
def tab_backtest(cfg):
    st.markdown("<h2 style='font-family:Syne'>📊 Backtest</h2>",unsafe_allow_html=True)
    c1,c2=st.columns([2,3])
    with c1: run=st.button("▶ Run Backtest",key="run_bt",use_container_width=True)
    with c2: st.markdown(f"<div class='card' style='padding:8px 14px;margin:0'><b>{cfg['symbol']}</b> · {cfg['interval']} · {cfg['period']} · {cfg['strategy']} · {cfg['sl_type']} · fast={cfg['fast']} slow={cfg['slow']}</div>",unsafe_allow_html=True)
    if not run:
        if st.session_state.get("bt_result"): _render_bt(st.session_state["bt_result"],cfg)
        return
    with st.spinner("Fetching data…"): df=fetch_ohlcv(cfg["symbol"],cfg["interval"],cfg["period"])
    if df is None or df.empty: st.error("Could not fetch data. Check symbol / interval / period."); return
    st.success(f"{len(df)} bars · {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')} IST")
    with st.spinner("Running backtest…"): result=run_backtest(df,cfg)
    st.session_state["bt_result"]=result; st.session_state["bt_trades"]=result["trades"]; st.session_state["bt_cfg"]=cfg; merge_all()
    _render_bt(result,cfg)

def _render_bt(result,cfg):
    trades=result["trades"]; an=build_analysis(trades)
    st.plotly_chart(candle_chart(result["df"],result["signals"],trades,title=cfg.get("symbol","")),use_container_width=True,key="pc_bt_1")
    if not trades: st.warning("No trades generated — adjust strategy params or filters."); return
    if result.get("violations"):
        with st.expander(f"⚠ {len(result['violations'])} violation(s)"):
            for v in result["violations"]: st.markdown(f"<div class='viol-card'>{v}</div>",unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3,c4,c5,c6=st.columns(6)
    pfs=f"{an['profit_factor']:.2f}" if np.isfinite(an["profit_factor"]) else "∞"
    c1.metric("Total PnL",f"{an['total_pnl']:.4f}"); c2.metric("Win Rate",f"{an['win_rate']:.1f}%")
    c3.metric("Trades",an["total_trades"]); c4.metric("Profit Factor",pfs)
    c5.metric("Max DD",f"{an['max_drawdown']:.4f}"); c6.metric("Avg W/L",f"{an['avg_win']:.4f} / {an['avg_loss']:.4f}")
    c7,c8,c9,c10=st.columns(4)
    c7.metric("Winners",an["winners"]); c8.metric("Losers",an["losers"])
    c9.metric("Median Win",f"{an['median_win']:.4f}"); c10.metric("Median Loss",f"{an['median_loss']:.4f}")
    rsl=an["rec_sl_pts"]; rtgt=an["rec_tgt_pts"]; impl=f"1:{round(rtgt/rsl,2)}" if rsl>0 else "—"
    st.markdown(f"<div class='rec-card'>📌 <b>Data-driven Recommendations</b>  Suggested SL: <b>{rsl:.4f} pts</b> &nbsp;|&nbsp; Target: <b>{rtgt:.4f} pts</b> &nbsp;|&nbsp; R:R: <b>{impl}</b></div>",unsafe_allow_html=True)
    st.plotly_chart(eq_chart(an["equity_arr"],an["dd_arr"]),use_container_width=True,key="pc_bt_2")
    ca,cb=st.columns(2)
    with ca: st.plotly_chart(heatmap_chart(an["df"]),use_container_width=True,key="pc_bt_3")
    with cb: st.plotly_chart(ls_chart(an["df"]),use_container_width=True,key="pc_bt_4")
    cc,cd=st.columns(2)
    with cc: st.plotly_chart(ang_scatter(result["df"],cfg.get("fast",9),cfg.get("slow",21)),use_container_width=True,key="pc_bt_5")
    with cd: st.plotly_chart(ew_chart(trades),use_container_width=True,key="pc_bt_6")
    ce,cf=st.columns(2)
    with ce: st.plotly_chart(reason_chart(an["reason_grp"]),use_container_width=True,key="pc_bt_7")
    with cf: st.plotly_chart(dur_chart(an["df"]),use_container_width=True,key="pc_bt_8")
    if len(an["daily_pnl"])>1: st.plotly_chart(daily_chart(an["daily_pnl"]),use_container_width=True,key="pc_bt_9")
    with st.expander("📐 Angle & Delta Distribution"):
        fe=ema(result["df"]["Close"],cfg.get("fast",9)); se_=ema(result["df"]["Close"],cfg.get("slow",21))
        ang=angle_series(fe,se_).dropna(); dlt=((fe-se_)/se_.replace(0,np.nan)*100).dropna()
        xa,xb=st.columns(2)
        with xa:
            fa=go.Figure(go.Histogram(x=ang,nbinsx=40,marker_color="#58a6ff",opacity=0.8)); fa.update_layout(title="Angle Dist °",height=260,**_CL); st.plotly_chart(fa,use_container_width=True,key="pc_10")
        with xb:
            fd=go.Figure(go.Histogram(x=dlt,nbinsx=40,marker_color="#d29922",opacity=0.8)); fd.update_layout(title="Delta Dist %",height=260,**_CL); st.plotly_chart(fd,use_container_width=True,key="pc_11")
    st.markdown("---"); st.markdown("**📋 Trade Log**")
    cols=["entry_time","exit_time","side","entry","exit","sl","target","pnl","exit_reason","duration_m","atr","pattern"]
    df_s=pd.DataFrame(trades); avail=[c for c in cols if c in df_s.columns]
    st.dataframe(df_s[avail].style.map(_ps,subset=["pnl"]),use_container_width=True,height=440)
    st.download_button("⬇ Export CSV",df_s[avail].to_csv(index=False),"backtest_trades.csv","text/csv")


# ── Tab 2: Live Trading ────────────────────────────────────────────────────────
def tab_live(cfg):
    st.markdown("<h2 style='font-family:Syne'>⚡ Live Trading</h2>",
                unsafe_allow_html=True)

    running = ts_get("running") or False

    # ── Always-visible control buttons ────────────────────────────────────────
    b1, b2, b3 = st.columns([1, 1, 1])
    start_clicked    = b1.button("▶  START",     key="btn_start",
                                 use_container_width=True, disabled=running,
                                 type="primary")
    stop_clicked     = b2.button("⏹  STOP",      key="btn_stop",
                                 use_container_width=True, disabled=not running)
    sq_clicked       = b3.button("⬛  SQUAREOFF", key="btn_sq",
                                 use_container_width=True, disabled=not running)

    if start_clicked:
        ts_update({
            "running": True, "config": cfg, "error": None,
            "live_position": None, "live_ltp": None, "live_df": None,
            "live_trades": [], "live_pnl": 0.0, "daily_pnl": 0.0,
            "candle_count": 0, "squareoff_requested": False,
            "trailing_sl": None, "phase": 1, "locked_sl": None,
            "live_signal": None, "live_ew_info": {},
            "log": deque(maxlen=300),
        })
        threading.Thread(target=live_thread, daemon=True).start()
        st.rerun()

    if stop_clicked:
        ts_set("running", False)
        ts_update({"live_position": None, "trailing_sl": None})
        st.rerun()

    if sq_clicked:
        ts_set("squareoff_requested", True)
        st.toast("Squareoff sent to thread", icon="⬛")

    # Error
    err = ts_get("error")
    if err:
        st.error(f"Thread error: {str(err)[:300]}")
        if st.button("Clear", key="btn_clr"):
            ts_set("error", None); st.rerun()

    st.markdown("---")

    # ── Live metrics ───────────────────────────────────────────────────────────
    ltp   = ts_get("live_ltp")
    pos   = ts_get("live_position")
    dp    = ts_get("daily_pnl") or 0.0
    sig   = ts_get("live_signal")
    trail = ts_get("trailing_sl")
    phase = ts_get("phase") or 1
    hb    = str(ts_get("thread_heartbeat") or "—")[:19]
    cnc   = ts_get("candle_count") or 0
    unreal = pos.get("unrealized_pnl", 0.0) if pos else 0.0

    dot  = "<span class='live-dot'></span>" if running else "<span class='idle-dot'></span>"
    dpc  = "#3fb950" if dp >= 0 else "#f85149"
    sigl = "🟢 BUY" if sig == 1 else ("🔴 SELL" if sig == -1 else "—")

    st.markdown(
        f"<div class='status-bar'>"
        f"{dot}<b>{'LIVE' if running else 'IDLE'}</b>"
        f"&nbsp;&nbsp;Heartbeat: <code>{hb}</code>"
        f"&nbsp;&nbsp;Ticks: <b>{cnc}</b>"
        f"&nbsp;&nbsp;Daily PnL: <b style='color:{dpc}'>{dp:+.4f}</b>"
        f"</div>", unsafe_allow_html=True)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("LTP",         f"{ltp:.4f}" if ltp else "—")
    m2.metric("Daily PnL",   f"{dp:+.4f}")
    m3.metric("Signal",      sigl)
    m4.metric("Position",    "OPEN 🔵" if pos else "FLAT ⬜")
    m5.metric("Trailing SL", f"{trail:.4f}" if trail else "—")
    m6.metric("SL Phase",    str(phase))

    if pos:
        st.markdown("**Open Position**")
        p1,p2,p3,p4,p5,p6 = st.columns(6)
        p1.metric("Side",    "BUY 🟢" if pos.get("side")==1 else "SELL 🔴")
        p2.metric("Entry",   f"{pos.get('entry',0):.4f}")
        p3.metric("SL",      f"{pos.get('sl',0):.4f}")
        p4.metric("Target",  f"{pos.get('target',0):.4f}")
        p5.metric("Eff SL",  f"{pos.get('eff_sl',0):.4f}")
        p6.metric("Unreal.", f"{unreal:+.4f}")

    # ── Strategy status panel ─────────────────────────────────────────────────
    si      = ts_get("live_strat_info") or {}
    blocked = ts_get("live_entry_blocked")
    if si or running:
        with st.expander("📡 Strategy Status", expanded=True):
            sc1,sc2,sc3,sc4,sc5 = st.columns(5)
            sc1.metric("Strategy",   si.get("strategy", cfg["strategy"]))
            fv = si.get("ema_fast_val")
            sv = si.get("ema_slow_val")
            sc2.metric(f"Fast EMA({cfg['fast']})", f"{fv:.4f}" if fv else "—")
            sc3.metric(f"Slow EMA({cfg['slow']})", f"{sv:.4f}" if sv else "—")
            av = si.get("atr_val")
            sc4.metric("ATR",        f"{av:.4f}" if av else "—")
            ang = si.get("angle")
            sc5.metric("Crossover°", f"{ang:.1f}°" if ang is not None else "—")

            sc6,sc7,sc8,sc9 = st.columns(4)
            gv = si.get("ema_gap")
            sc6.metric("EMA Gap",  f"{gv:.4f}" if gv is not None else "—")
            sc7.metric("Trend",    si.get("trend","—"))
            sc8.metric("Bars",     str(si.get("bars","—")))
            sc9.metric("Last Bar", si.get("last_bar","—"))

            if cfg["strategy"] == "Elliott Wave":
                st.markdown("**🌊 Elliott Wave Analysis**")
                ew1,ew2,ew3 = st.columns(3)
                pat  = si.get("ew_pattern","Computing…")
                ewsig = si.get("ew_signal",0)
                conf  = si.get("ew_confidence",0)
                ew1.metric("Pattern",    pat)
                ew2.metric("Confidence", f"{conf}%")
                ew3.metric("EW Signal",  "🟢 BUY" if ewsig==1 else "🔴 SELL" if ewsig==-1 else "—")

                # EW wave range table
                ew_live = ts_get("live_ew_info") or {}
                waves = ew_live.get("waves",[])
                fib   = ew_live.get("fib",{})
                if waves:
                    _df_live = ts_get("live_df")
                    wave_rows = ""
                    for wi, w in enumerate(waves):
                        bar_i = w[0] if len(w)>0 else "—"
                        price = f"{w[1]:.4f}" if len(w)>1 else "—"
                        wtype = w[2] if len(w)>2 else ("H" if wi%2==0 else "L")
                        bar_t = "—"
                        if _df_live is not None and isinstance(bar_i,int) and bar_i < len(_df_live):
                            bar_t = str(_df_live.index[bar_i])[:16]
                        wave_rows += (f"<tr><td>W{wi+1}</td><td>{wtype}</td>"
                                      f"<td>{bar_t}</td><td>{price}</td></tr>")
                    st.markdown(
                        f"<table style='font-size:.75rem;width:100%'>"
                        f"<tr><th>Wave</th><th>Type</th><th>Time</th><th>Price</th></tr>"
                        f"{wave_rows}</table>",
                        unsafe_allow_html=True)
                if fib:
                    ltp_now = ts_get("live_ltp") or 0
                    fib_rows = ""
                    for r,lvl in sorted(fib.items()):
                        c = "#3fb950" if lvl < ltp_now else "#f85149"
                        d = "↓ Below" if lvl < ltp_now else "↑ Above"
                        fib_rows += (f"<tr><td>{r}</td><td style='color:{c}'>"
                                     f"{lvl:.4f}</td><td style='color:{c}'>{d}</td></tr>")
                    st.markdown(
                        f"<table style='font-size:.75rem'>"
                        f"<tr><th>Fib</th><th>Level</th><th>vs LTP</th></tr>"
                        f"{fib_rows}</table>",
                        unsafe_allow_html=True)

                # Buy/Sell suitability
                suit_color = "#3fb950" if ewsig==1 else "#f85149" if ewsig==-1 else "#8b949e"
                suit_text  = ("✅ BUY suitable now" if ewsig==1
                              else "✅ SELL suitable now" if ewsig==-1
                              else "⏳ No clear directional signal")
                st.markdown(
                    f"<div style='background:#161b22;border-left:4px solid {suit_color};"
                    f"padding:8px 14px;border-radius:4px;font-size:.88rem;margin-top:6px'>"
                    f"<b style='color:{suit_color}'>{suit_text}</b> "
                    f"(Pattern: {pat}  |  Confidence: {conf}%)</div>",
                    unsafe_allow_html=True)

            if blocked:
                st.warning(f"⚠ Entry blocked: {blocked}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    _df = ts_get("live_df")
    if _df is not None and not _df.empty:
        df2 = _df.copy()
        if "EMA_fast" not in df2.columns:
            df2["EMA_fast"] = ema(df2["Close"], cfg["fast"])
            df2["EMA_slow"] = ema(df2["Close"], cfg["slow"])
            df2["ATR"]      = calc_atr(df2, cfg["atr_period"])
        ew = ts_get("live_ew_info") or {}
        st.plotly_chart(candle_chart(df2, ew_info=ew,
            title=f"{cfg['symbol']}  {cfg['interval']}  (Live)"),
            use_container_width=True, key="live_candle_chart")
    elif running:
        st.info("⏳ Loading chart data…")

    # ── Session trades ─────────────────────────────────────────────────────────
    live_trades = list(ts_get("live_trades") or [])
    if live_trades:
        merge_all()
        an = build_analysis(live_trades)
        if an:
            st.markdown("**Session Trades**")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Session PnL",   f"{an['total_pnl']:.4f}")
            t2.metric("Win Rate",      f"{an['win_rate']:.1f}%")
            t3.metric("Trades",        an["total_trades"])
            pfs = f"{an['profit_factor']:.2f}" if np.isfinite(an["profit_factor"]) else "∞"
            t4.metric("Profit Factor", pfs)
            df_t  = pd.DataFrame(live_trades)
            cols_ = ["entry_time","exit_time","side","entry","exit","sl",
                     "target","pnl","exit_reason","atr","pattern"]
            avail = [c for c in cols_ if c in df_t.columns]
            st.dataframe(df_t[avail].style.map(_ps, subset=["pnl"]),
                         use_container_width=True, height=280)

    # ── Compact scrollable log (last 12 lines) ───────────────────────────────────
    logs = list(ts_get("log") or [])
    if logs:
        def _lc(l):
            if any(x in l for x in ["BUY","SELL","Target","started","Config","loaded"]): return "#3fb950"
            if any(x in l for x in ["ERROR","SL hit","Stagnant","EOD"]): return "#f85149"
            return "#8b949e"
        rows = "".join(
            f"<div style='color:{_lc(l)};padding:1px 0;white-space:nowrap'>{l}</div>"
            for l in logs[:12])
        st.markdown(
            f"<div style='background:#0d1117;border:1px solid #30363d;border-radius:6px;"
            f"padding:6px 10px;font-size:.72rem;font-family:monospace;"
            f"max-height:120px;overflow-y:auto'>{rows}</div>",
            unsafe_allow_html=True)

    # ── Auto-refresh: 1.5 s when running ─────────────────────────────────────
    if ts_get("running"):
        time.sleep(1.5)
        st.rerun()

# ── Tab 3: Trade History ───────────────────────────────────────────────────────
def tab_history():
    st.markdown("<h2 style='font-family:Syne'>📜 Trade History</h2>",unsafe_allow_html=True)
    merge_all()
    src=st.radio("Source",["All (Backtest + Live)","Backtest Only","Live Only"],horizontal=True)
    all_t=st.session_state.get("all_trades",[])
    live_t=list(ts_get("live_trades") or [])
    if src=="Backtest Only": trades=[t for t in all_t if t.get("source","backtest")=="backtest"]
    elif src=="Live Only": trades=live_t
    else: trades=all_t
    if not trades: st.info("No trades yet. Run a Backtest or start Live Trading."); return
    an=build_analysis(trades)
    if not an: return
    df=an["df"]
    c1,c2,c3,c4,c5,c6=st.columns(6)
    pfs=f"{an['profit_factor']:.2f}" if np.isfinite(an["profit_factor"]) else "∞"
    c1.metric("Total PnL",f"{an['total_pnl']:.4f}"); c2.metric("Win Rate",f"{an['win_rate']:.1f}%")
    c3.metric("Trades",an["total_trades"]); c4.metric("Max DD",f"{an['max_drawdown']:.4f}")
    c5.metric("PF",pfs); c6.metric("Median Win",f"{an['median_win']:.4f}")
    rsl=an["rec_sl_pts"]; rtgt=an["rec_tgt_pts"]; impl=f"1:{round(rtgt/rsl,2)}" if rsl>0 else "—"
    st.markdown(f"<div class='rec-card'>📌 Rec SL: <b>{rsl:.4f}</b> &nbsp;|&nbsp; Target: <b>{rtgt:.4f}</b> &nbsp;|&nbsp; R:R: <b>{impl}</b></div>",unsafe_allow_html=True)
    st.markdown("---")
    fc1,fc2,fc3=st.columns(3)
    with fc1: sf=st.selectbox("Side",["All","BUY","SELL"])
    with fc2: ro=["All"]+sorted(df["exit_reason"].dropna().unique().tolist()); rf=st.selectbox("Exit Reason",ro)
    with fc3: po=["All"]+sorted(df["pattern"].dropna().unique().tolist()); pf_=st.selectbox("EW Pattern",po)
    ds=df.copy()
    if sf!="All": ds=ds[ds["side"]==sf]
    if rf!="All": ds=ds[ds["exit_reason"]==rf]
    if pf_!="All": ds=ds[ds["pattern"]==pf_]
    st.markdown(f"Showing **{len(ds)}** of {len(df)} trades")
    vc=["entry_time","exit_time","side","entry","exit","sl","target","pnl","exit_reason","duration_m","atr","pattern","source"]
    av=[c for c in vc if c in ds.columns]
    st.dataframe(ds[av].style.map(_ps,subset=["pnl"]),use_container_width=True,height=440)
    st.download_button("⬇ Export CSV",ds[av].to_csv(index=False),"trade_history.csv","text/csv")
    st.markdown("---")
    if len(an["equity_arr"])>1: st.plotly_chart(eq_chart(an["equity_arr"],an["dd_arr"]),use_container_width=True,key="pc_hi_1")
    if len(trades)>=3:
        ca,cb=st.columns(2)
        with ca: st.plotly_chart(heatmap_chart(df),use_container_width=True,key="pc_hi_2")
        with cb: st.plotly_chart(ls_chart(df),use_container_width=True,key="pc_hi_3")
        cc,cd=st.columns(2)
        with cc: st.plotly_chart(reason_chart(an["reason_grp"]),use_container_width=True,key="pc_hi_4")
        with cd: st.plotly_chart(ew_chart(trades),use_container_width=True,key="pc_hi_5")
        if len(an["daily_pnl"])>1: st.plotly_chart(daily_chart(an["daily_pnl"]),use_container_width=True,key="pc_hi_6")
        st.plotly_chart(dur_chart(df),use_container_width=True,key="pc_hi_7")

# ── Tab 4: Optimization ────────────────────────────────────────────────────────
def tab_optimization(cfg):
    st.markdown("<h2 style='font-family:Syne'>🔬 Strategy Optimization</h2>",unsafe_allow_html=True)
    with st.expander("ℹ How it works",expanded=False):
        st.markdown("Grid Search over Fast × Slow EMA combinations. Each pair runs a full backtest with current SL/filter settings.")
    c1,c2=st.columns(2)
    with c1: fr=st.slider("Fast EMA range",3,50,(5,20)); fs=st.number_input("Fast step",min_value=1,max_value=20,value=2)
    with c2: sr=st.slider("Slow EMA range",10,120,(15,55)); ss=st.number_input("Slow step",min_value=1,max_value=20,value=3)
    metric=st.selectbox("Optimise for",["Total PnL","Win Rate","Profit Factor","Max Drawdown (minimise)"])
    max_c=st.number_input("Max combinations",min_value=4,max_value=500,value=80,step=10)
    run_=st.button("🚀 Run Grid Search",use_container_width=True)
    if not run_: return
    with st.spinner("Fetching data…"): df=fetch_ohlcv(cfg["symbol"],cfg["interval"],cfg["period"])
    if df is None or df.empty: st.error("Could not fetch data."); return
    fv=list(range(int(fr[0]),int(fr[1])+1,int(fs))); sv=list(range(int(sr[0]),int(sr[1])+1,int(ss)))
    combos=[(f,s) for f in fv for s in sv if f<s]
    if not combos: st.warning("No valid combinations — Fast must be < Slow."); return
    if len(combos)>max_c:
        import random; combos=random.sample(combos,int(max_c)); st.info(f"Sampled {len(combos)} combinations.")
    else: st.info(f"Testing {len(combos)} combinations…")
    prog=st.progress(0.0); stat=st.empty(); results=[]
    for idx,(f,s) in enumerate(combos):
        stat.markdown(f"Testing Fast={f}, Slow={s}…")
        try:
            res=run_backtest(df,{**cfg,"fast":f,"slow":s}); an=build_analysis(res["trades"])
            if an: pf=float(an["profit_factor"]) if np.isfinite(an["profit_factor"]) else 0.0; results.append({"fast":f,"slow":s,"pnl":an["total_pnl"],"win_rate":an["win_rate"],"profit_factor":pf,"max_dd":an["max_drawdown"],"trades":an["total_trades"],"med_win":an["median_win"],"med_loss":an["median_loss"]})
        except: pass
        prog.progress((idx+1)/len(combos))
    stat.empty()
    if not results: st.warning("No results generated."); return
    sm={"Total PnL":("pnl",False),"Win Rate":("win_rate",False),"Profit Factor":("profit_factor",False),"Max Drawdown (minimise)":("max_dd",True)}
    sc,asc=sm[metric]
    dr=pd.DataFrame(results).sort_values(sc,ascending=asc).reset_index(drop=True)
    best=dr.iloc[0]
    st.success(f"🏆 Best: Fast=**{int(best['fast'])}** Slow=**{int(best['slow'])}** | PnL={best['pnl']:.4f} | WR={best['win_rate']:.1f}% | Trades={int(best['trades'])}")
    if st.button(f"✅ Apply Best (Fast={int(best['fast'])}, Slow={int(best['slow'])})"):
        st.session_state["opt_fast"]=int(best["fast"]); st.session_state["opt_slow"]=int(best["slow"])
        st.success("Applied! Sidebar will update on next interaction.")
    st.markdown("---")
    try:
        pv=dr.pivot_table(values="pnl",index="slow",columns="fast",aggfunc="mean",fill_value=0)
        fh=go.Figure(go.Heatmap(z=pv.values,x=[str(c) for c in pv.columns],y=[str(r) for r in pv.index],zmid=0,colorscale=[[0,"#f85149"],[0.5,"#161b22"],[1,"#3fb950"]],text=np.round(pv.values,1),texttemplate="%{text}",colorbar=dict(tickfont=dict(color="#e6edf3"))))
        fh.update_layout(title="PnL Heatmap (Fast × Slow EMA)",xaxis_title="Fast EMA",yaxis_title="Slow EMA",height=440,**_CL); st.plotly_chart(fh,use_container_width=True,key="pc_20")
    except: pass
    try:
        wr=dr.pivot_table(values="win_rate",index="slow",columns="fast",aggfunc="mean",fill_value=0)
        fw=go.Figure(go.Heatmap(z=wr.values,x=[str(c) for c in wr.columns],y=[str(r) for r in wr.index],colorscale="RdYlGn",text=np.round(wr.values,1),texttemplate="%{text}%",colorbar=dict(tickfont=dict(color="#e6edf3"))))
        fw.update_layout(title="Win-Rate Heatmap",xaxis_title="Fast EMA",yaxis_title="Slow EMA",height=380,**_CL); st.plotly_chart(fw,use_container_width=True,key="pc_21")
    except: pass
    try:
        fp=go.Figure(go.Scatter(x=dr["max_dd"],y=dr["pnl"],mode="markers+text",text=[f"{int(r.fast)}/{int(r.slow)}" for _,r in dr.iterrows()],textposition="top center",marker=dict(color=dr["win_rate"],colorscale="RdYlGn",size=8,opacity=0.8,colorbar=dict(title="Win%",tickfont=dict(color="#e6edf3")))))
        fp.update_layout(title="Pareto — PnL vs Max Drawdown (colour=Win%)",xaxis_title="Max Drawdown",yaxis_title="Total PnL",height=380,**_CL); st.plotly_chart(fp,use_container_width=True,key="pc_22")
    except: pass
    try:
        ftc=go.Figure(go.Histogram(x=dr["trades"],nbinsx=20,marker_color="#58a6ff",opacity=0.8))
        ftc.update_layout(title="Trade Count Distribution",xaxis_title="# Trades",height=260,**_CL); st.plotly_chart(ftc,use_container_width=True,key="pc_23")
    except: pass
    st.markdown("**Top 20 Combinations**")
    st.dataframe(dr.head(20).style.format({"pnl":".4f","win_rate":".1f","profit_factor":".2f","max_dd":".4f","med_win":".4f","med_loss":".4f"}),use_container_width=True)
    st.download_button("⬇ Export Optimization CSV",dr.to_csv(index=False),"optimization.csv","text/csv")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    running=ts_get("running")
    dot="<span class='live-dot'></span>" if running else "<span class='idle-dot'></span>"
    now=datetime.now(IST).strftime('%d %b %Y  %H:%M:%S IST')
    st.markdown(f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'><span style='font-size:2.4rem'>📈</span><div><h1 style='font-family:Syne;font-size:2rem;margin:0'>Smart Investing</h1><span style='color:#8b949e;font-size:.76rem'>Production-Grade Algorithmic Trading Platform &nbsp;·&nbsp; v{APP_VERSION} &nbsp;·&nbsp; {now} &nbsp;{dot}<b style='color:{'#3fb950' if running else '#8b949e'}'>{'LIVE' if running else 'IDLE'}</b></span></div></div>",unsafe_allow_html=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    cfg=sidebar_config()
    t1,t2,t3,t4=st.tabs(["📊  Backtest","⚡  Live Trading","📜  Trade History","🔬  Optimization"])
    with t1: tab_backtest(cfg)
    with t2: tab_live(cfg)
    with t3: tab_history()
    with t4: tab_optimization(cfg)

if __name__=="__main__":
    main()
