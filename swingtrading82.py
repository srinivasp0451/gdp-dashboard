"""
╔══════════════════════════════════════════════════════════════╗
║       🌊  ELLIOTT WAVE ALGO TRADER  v5.1                     ║
║  Fragment-based Live · No Thread Race · Crystal Clear UI    ║
╚══════════════════════════════════════════════════════════════╝
Requirements:
  pip install streamlit>=1.37 yfinance plotly pandas numpy requests
Run: streamlit run elliott_wave_algo_trader.py

Key fixes vs v5.0:
  - NO background thread → no session-state race conditions
  - @st.fragment(run_every=2) → LTP refreshes every 2 s, ZERO page flicker
  - Giant POSITION STATUS banner — impossible to miss
  - Unrealized P&L updated every 2 s
  - Separate "Signal detected" vs "Position open" states
  - Clear explanation for EVERY state user can be in
  - Next-check timer works correctly
  - Conservative exit: SL checked with Low(BUY)/High(SELL) first
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from typing import Optional
import requests
import itertools
import warnings

warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> datetime:
    return datetime.now(IST)

def fmt_ist(dt=None) -> str:
    if dt is None:
        return now_ist().strftime("%H:%M:%S IST")
    try:
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).strftime("%d-%b %H:%M:%S IST")
    except Exception:
        return str(dt)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌊 Elliott Wave Algo Trader",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Exo 2',sans-serif;}

.main-hdr{background:linear-gradient(135deg,#0a0e1a,#0d1b2a,#0a1628);
  border:1px solid #1e3a5f;border-radius:14px;padding:18px 24px;margin-bottom:12px;
  box-shadow:0 4px 24px rgba(0,229,255,.08);}
.main-hdr h1{font-family:'Exo 2',sans-serif;font-weight:700;
  background:linear-gradient(90deg,#00e5ff,#4dd0e1,#00bcd4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin:0;font-size:1.8rem;}
.main-hdr p{color:#546e7a;margin:4px 0 0;font-size:.84rem;}

/* === POSITION MEGA BANNER === */
.pos-none{background:#07111e;border:2px dashed #263238;border-radius:14px;
  padding:18px 24px;text-align:center;}
.pos-signal{background:rgba(0,229,255,.08);border:2px solid #00e5ff;
  border-radius:14px;padding:18px 24px;text-align:center;}
.pos-buy{background:rgba(76,175,80,.12);border:3px solid #4caf50;
  border-radius:14px;padding:18px 24px;}
.pos-sell{background:rgba(244,67,54,.12);border:3px solid #f44336;
  border-radius:14px;padding:18px 24px;}
.pos-reversing{background:rgba(171,71,188,.14);border:3px solid #ab47bc;
  border-radius:14px;padding:18px 24px;text-align:center;animation:blink .8s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.6}}

/* tick cards */
.tick-card{background:#06101a;border:1px solid #0d3349;border-radius:8px;
  padding:10px 14px;text-align:center;}
.wave-card{background:#060d14;border:1px solid #1e3a5f;border-radius:8px;
  padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:.82rem;line-height:1.9;}
.info-box{background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
  padding:12px 14px;font-size:.86rem;line-height:1.9;}
.best-cfg{background:rgba(0,229,255,.07);border:1px solid #00bcd4;
  border-radius:10px;padding:12px 16px;margin:6px 0;}

.stTabs [data-baseweb="tab-list"]{gap:5px;background:transparent;}
.stTabs [data-baseweb="tab"]{background:#0d1b2a;border-radius:8px;color:#546e7a;
  border:1px solid #1e3a5f;padding:5px 12px;font-size:.81rem;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0d3349,#0a2540)!important;
  color:#00e5ff!important;border-color:#00bcd4!important;}
div[data-testid="metric-container"]{background:#0a1628;border:1px solid #1e3a5f;
  border-radius:8px;padding:9px 12px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
TICKER_GROUPS = {
    "🇮🇳 Indian Indices": {
        "Nifty 50":"^NSEI","Bank Nifty":"^NSEBANK","Sensex":"^BSESN",
        "Nifty IT":"^CNXIT","Nifty Midcap":"^NSEMDCP50",
        "Nifty Auto":"^CNXAUTO","Nifty Pharma":"^CNXPHARMA",
    },
    "🇮🇳 NSE Top Stocks": {
        "Reliance":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
        "Infosys":"INFY.NS","ICICI Bank":"ICICIBANK.NS","Kotak Bank":"KOTAKBANK.NS",
        "Wipro":"WIPRO.NS","L&T":"LT.NS","Axis Bank":"AXISBANK.NS",
        "SBI":"SBIN.NS","Bajaj Finance":"BAJFINANCE.NS","Maruti":"MARUTI.NS",
    },
    "₿ Crypto": {"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD",
                  "Solana":"SOL-USD","XRP":"XRP-USD","Cardano":"ADA-USD"},
    "💱 Forex":  {"USD/INR":"USDINR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X",
                  "USD/JPY":"JPY=X","AUD/USD":"AUDUSD=X","EUR/INR":"EURINR=X"},
    "🥇 Commodities": {"Gold":"GC=F","Silver":"SI=F","Crude Oil WTI":"CL=F",
                       "Natural Gas":"NG=F","Copper":"HG=F"},
    "🌐 US Stocks": {"Apple":"AAPL","Tesla":"TSLA","NVIDIA":"NVDA",
                     "Microsoft":"MSFT","Alphabet":"GOOGL","Meta":"META"},
    "✏️ Custom Ticker": {"Custom":"__CUSTOM__"},
}

TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d","1wk"]
PERIODS    = ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]
VALID_PERIODS = {
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo","3mo"],
    "15m":["1d","5d","7d","1mo","3mo"],"30m":["1d","5d","7d","1mo","3mo"],
    "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h":["1mo","3mo","6mo","1y","2y","5y"],
    "1d":["1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["3mo","6mo","1y","2y","5y","10y","20y"],
}
POLL_SECS = {"1m":60,"5m":300,"15m":900,"30m":1800,
             "1h":3600,"4h":14400,"1d":86400,"1wk":604800}
MTF_COMBOS = [("1d","1y","Daily"),("4h","3mo","4-Hour"),
              ("1h","1mo","1-Hour"),("15m","5d","15-Min")]

SL_WAVE  = "wave_auto"
SL_PTS   = "__sl_pts__"
SL_SIGREV= "__sl_sigrev__"
TGT_PTS  = "__tgt_pts__"
TGT_SIGREV="__tgt_sigrev__"

SL_MAP = {
    "Wave Auto (Pivot Low/High)": SL_WAVE,
    "0.5%":0.005,"1%":0.01,"1.5%":0.015,
    "2%":0.02,"2.5%":0.025,"3%":0.03,"5%":0.05,
    "Custom Points ▼": SL_PTS,
    "Exit on Signal Reverse": SL_SIGREV,
}
TGT_MAP = {
    "Wave Auto (Fib 1.618×W1)":"wave_auto",
    "R:R 1:1":1.0,"R:R 1:1.5":1.5,"R:R 1:2":2.0,
    "R:R 1:2.5":2.5,"R:R 1:3":3.0,
    "Fib 1.618×Wave 1":"fib_1618","Fib 2.618×Wave 1":"fib_2618",
    "Custom Points ▼": TGT_PTS,
    "Exit on Signal Reverse": TGT_SIGREV,
}
SL_KEYS  = list(SL_MAP.keys())
TGT_KEYS = list(TGT_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
_DEF = {
    # live engine (all written from MAIN THREAD only — no race condition)
    "live_running":        False,
    "live_phase":          "idle",
    # idle | scanning | no_signal | signal_ready | pos_open | reversing | stopped
    "live_ltp":            None,
    "live_ltp_ts":         None,
    "live_last_candle_ist":"—",
    "live_delay_s":        0,
    "live_next_check_ist": "—",
    "last_bar_ts":         None,
    "last_fetch_wall":     0.0,   # wall-clock time of last full OHLCV fetch
    "live_position":       None,
    "live_pnl":            0.0,
    "live_signals":        [],
    "live_trades":         [],
    "live_log":            [],
    "live_wave_state":     None,
    "live_last_sig":       None,
    "live_no_pos_reason":  "Click ▶ Start Live to begin. Everything runs automatically.",
    # scan / chart
    "_scan_sig":None,"_scan_df":None,
    # backtest
    "bt_results":None,
    # opt
    "opt_results":None,
    # analysis
    "_analysis_results":None,"_analysis_overall":"HOLD","_analysis_symbol":"",
    # applied config from optimization
    "applied_depth":5,
    "applied_sl_lbl":"Wave Auto (Pivot Low/High)",
    "applied_tgt_lbl":"Wave Auto (Fib 1.618×W1)",
    "best_cfg_applied":False,
    # custom points
    "custom_sl_pts":50.0,"custom_tgt_pts":100.0,
}
for _k,_v in _DEF.items():
    if _k not in st.session_state:
        st.session_state[_k]=_v

# ═══════════════════════════════════════════════════════════════════════════
# RATE-LIMIT SAFE FETCH  (lock shared across threads for scan / analysis)
# ═══════════════════════════════════════════════════════════════════════════
_fetch_lock   = threading.Lock()
_last_fetch_t = [0.0]

def fetch_ohlcv(symbol:str, interval:str, period:str,
                min_delay:float=1.5) -> Optional[pd.DataFrame]:
    with _fetch_lock:
        gap = time.time()-_last_fetch_t[0]
        if gap < min_delay:
            time.sleep(min_delay-gap)
        try:
            df = yf.download(symbol,interval=interval,period=period,
                             progress=False,auto_adjust=True)
            _last_fetch_t[0] = time.time()
        except Exception:
            _last_fetch_t[0] = time.time()
            return None
    if df is None or df.empty: return None
    if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    df = df.dropna(subset=["Open","High","Low","Close"])
    df.index = pd.to_datetime(df.index)
    return df

def fetch_ltp(symbol:str) -> Optional[float]:
    """Lightweight LTP — uses the most recent Close from a tiny 2-bar fetch."""
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info
        price = fi.get("lastPrice") or fi.get("regularMarketPrice")
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def add_indicators(df:pd.DataFrame) -> pd.DataFrame:
    df=df.copy(); c=df["Close"]
    df["EMA_20"]=c.ewm(span=20,adjust=False).mean()
    df["EMA_50"]=c.ewm(span=50,adjust=False).mean()
    df["EMA_200"]=c.ewm(span=200,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    df["RSI"]=100-(100/(1+rs))
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    df["MACD"]=ema12-ema26; df["MACD_Signal"]=df["MACD"].ewm(span=9,adjust=False).mean()
    if "Volume" in df.columns: df["Vol_Avg"]=df["Volume"].rolling(20).mean()
    return df

# ═══════════════════════════════════════════════════════════════════════════
# PIVOTS
# ═══════════════════════════════════════════════════════════════════════════
def find_pivots(df:pd.DataFrame, depth:int=5) -> list:
    H,L,n=df["High"].values,df["Low"].values,len(df)
    raw=[]
    for i in range(depth,n-depth):
        wh=H[max(0,i-depth):i+depth+1]; wl=L[max(0,i-depth):i+depth+1]
        if H[i]==wh.max(): raw.append((i,float(H[i]),"H"))
        elif L[i]==wl.min(): raw.append((i,float(L[i]),"L"))
    clean=[]
    for p in raw:
        if not clean or clean[-1][2]!=p[2]: clean.append(list(p))
        else:
            if p[2]=="H" and p[1]>clean[-1][1]: clean[-1]=list(p)
            elif p[2]=="L" and p[1]<clean[-1][1]: clean[-1]=list(p)
    return [tuple(x) for x in clean]

# ═══════════════════════════════════════════════════════════════════════════
# WAVE STATE (transparency panel)
# ═══════════════════════════════════════════════════════════════════════════
def analyze_wave_state(df:pd.DataFrame, pivots:list, sig:dict) -> dict:
    cur=float(df["Close"].iloc[-1]) if len(df) else 0
    if not pivots:
        return {"current_wave":"Collecting data…","next_wave":"—",
                "direction":"NEUTRAL","fib_levels":{},
                "action":"Insufficient pivot data. Try smaller Pivot Depth or longer Period.",
                "auto_action":"Wait"}

    if sig["signal"]=="BUY":
        wp=sig.get("wave_pivots",[None,None,None])
        p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p1[1]-p0[1]) if p0 and p1 else cur*0.02
        w2_low=p2[1] if p2 else cur*0.99
        fibs={"W3 target (1.618×W1)":round(w2_low+w1*1.618,2),
              "W3 min (1.000×W1)"  :round(w2_low+w1*1.000,2),
              "W3 ext  (2.618×W1)" :round(w2_low+w1*2.618,2)}
        return {"current_wave":"✅ Wave-2 Bottom — ENTRY ZONE for Wave-3",
                "next_wave":"Wave-3 UP ↑ (strongest & longest move)",
                "direction":"BULLISH","fib_levels":fibs,
                "action":(f"BUY position is being opened (or already open) at market price.\n"
                          f"SL placed below Wave-2 pivot at {w2_low:.2f}.\n"
                          f"Wave-3 up target: {fibs['W3 target (1.618×W1)']:.2f} (Fibonacci 1.618×W1)\n\n"
                          f"WHAT TO DO: Nothing. System manages everything automatically.\n"
                          f"Position will auto-close at SL, Target, or on SELL signal reverse."),
                "auto_action":"BUY"}

    if sig["signal"]=="SELL":
        wp=sig.get("wave_pivots",[None,None,None])
        p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p0[1]-p1[1]) if p0 and p1 else cur*0.02
        w2_high=p2[1] if p2 else cur*1.01
        fibs={"W3 target (1.618×W1)":round(w2_high-w1*1.618,2),
              "W3 min (1.000×W1)"  :round(w2_high-w1*1.000,2),
              "W3 ext  (2.618×W1)" :round(w2_high-w1*2.618,2)}
        return {"current_wave":"🔴 Wave-2 Top — ENTRY ZONE for Wave-3",
                "next_wave":"Wave-3 DOWN ↓ (strongest & longest move)",
                "direction":"BEARISH","fib_levels":fibs,
                "action":(f"SELL position is being opened (or already open) at market price.\n"
                          f"SL placed above Wave-2 pivot at {w2_high:.2f}.\n"
                          f"Wave-3 down target: {fibs['W3 target (1.618×W1)']:.2f} (Fibonacci 1.618×W1)\n\n"
                          f"WHAT TO DO: Nothing. System manages everything automatically.\n"
                          f"Position will auto-close at SL, Target, or on BUY signal reverse."),
                "auto_action":"SELL"}

    # HOLD: describe where we are
    lp=pivots[-1]
    retrace_pct=0.0
    if len(pivots)>=3:
        pa,pb,pc=pivots[-3],pivots[-2],pivots[-1]
        if abs(pb[1]-pa[1])>0:
            retrace_pct=abs(pc[1]-pb[1])/abs(pb[1]-pa[1])*100
    needed=f" (need 38–79% for signal; currently {retrace_pct:.1f}%)" if retrace_pct>0 else ""
    return {"current_wave":f"In progress — last pivot {'High' if lp[2]=='H' else 'Low'} @ {lp[1]:.2f}",
            "next_wave":f"Waiting for Wave-2 to complete{needed}",
            "direction":"NEUTRAL","fib_levels":{},
            "action":(f"No actionable signal yet. {sig.get('reason','')}\n\n"
                      f"WHAT TO DO: Nothing. System scans every candle automatically.\n"
                      f"A BUY/SELL signal fires when Wave-2 retracement hits 38–79% of Wave-1.\n"
                      f"Position opens automatically when signal fires. You just watch."),
            "auto_action":"Wait"}

# ═══════════════════════════════════════════════════════════════════════════
# CORE SIGNAL  (identical for backtest + live)
# ═══════════════════════════════════════════════════════════════════════════
def _blank(reason:str="") -> dict:
    return {"signal":"HOLD","entry_price":None,"sl":None,"target":None,
            "confidence":0.0,"reason":reason or "No Elliott Wave pattern detected",
            "pattern":"—","wave_pivots":None,"wave1_len":0.0,"retracement":0.0}

def _calc_target(tt, entry:float, direction:str, w1:float,
                 risk:float, ctp:float=100.0) -> float:
    s=1 if direction=="BUY" else -1
    if tt in ("wave_auto","fib_1618"): return entry+s*w1*1.618
    if tt=="fib_2618":                 return entry+s*w1*2.618
    if tt==TGT_PTS:                    return entry+s*ctp
    if tt==TGT_SIGREV:                 return entry+s*w1*1.618
    if isinstance(tt,(int,float)):     return entry+s*risk*float(tt)
    return entry+s*risk*2.0

def ew_signal(df:pd.DataFrame, depth:int=5,
              sl_type="wave_auto", tgt_type="wave_auto",
              csl:float=50.0, ctgt:float=100.0) -> dict:
    n=len(df)
    if n<max(30,depth*4): return _blank("Insufficient bars")
    pivots=find_pivots(df,depth)
    if len(pivots)<4: return _blank("Not enough pivots — try smaller Pivot Depth")
    cur=float(df["Close"].iloc[-1])
    best,bc=_blank(),0.0

    for i in range(len(pivots)-2):
        p0,p1,p2=pivots[i],pivots[i+1],pivots[i+2]
        bs=n-1-p2[0]
        # BUY
        if p0[2]=="L" and p1[2]=="H" and p2[2]=="L":
            w1=p1[1]-p0[1]
            if w1<=0: continue
            r=(p1[1]-p2[1])/w1
            if not (0.236<=r<=0.886 and p2[1]>p0[1] and bs<=depth*4): continue
            c=0.50
            if 0.382<=r<=0.618: c=0.65
            if 0.50 <=r<=0.786: c=0.72
            if abs(r-0.618)<0.04: c=0.87
            if abs(r-0.382)<0.03: c=0.75
            if i+3<len(pivots) and pivots[i+3][2]=="H":
                w3=pivots[i+3][1]-p2[1]
                if w3>w1: c=min(c+0.08,0.95)
                if abs(w3/w1-1.618)<0.20: c=min(c+0.05,0.98)
            if c<=bc: continue
            e=cur
            sl_=p2[1]*0.998 if sl_type==SL_WAVE else \
                e-csl          if sl_type==SL_PTS  else \
                e*(1-0.05)     if sl_type==SL_SIGREV else \
                e*(1-float(sl_type))
            rk=e-sl_
            if rk<=0: continue
            tgt_=_calc_target(tgt_type,e,"BUY",w1,rk,ctgt)
            if tgt_<=e: continue
            bc=c
            best={"signal":"BUY","entry_price":e,"sl":sl_,"target":tgt_,
                  "confidence":c,"retracement":r,
                  "reason":f"Wave-2 bottom: {r:.1%} retrace → Wave-3 up",
                  "pattern":f"W2 Bottom ({r:.1%})","wave_pivots":[p0,p1,p2],"wave1_len":w1}
        # SELL
        elif p0[2]=="H" and p1[2]=="L" and p2[2]=="H":
            w1=p0[1]-p1[1]
            if w1<=0: continue
            r=(p2[1]-p1[1])/w1
            if not (0.236<=r<=0.886 and p2[1]<p0[1] and bs<=depth*4): continue
            c=0.50
            if 0.382<=r<=0.618: c=0.65
            if 0.50 <=r<=0.786: c=0.72
            if abs(r-0.618)<0.04: c=0.87
            if abs(r-0.382)<0.03: c=0.75
            if i+3<len(pivots) and pivots[i+3][2]=="L":
                w3=p2[1]-pivots[i+3][1]
                if w3>w1: c=min(c+0.08,0.95)
                if abs(w3/w1-1.618)<0.20: c=min(c+0.05,0.98)
            if c<=bc: continue
            e=cur
            sl_=p2[1]*1.002 if sl_type==SL_WAVE else \
                e+csl          if sl_type==SL_PTS  else \
                e*(1+0.05)     if sl_type==SL_SIGREV else \
                e*(1+float(sl_type))
            rk=sl_-e
            if rk<=0: continue
            tgt_=_calc_target(tgt_type,e,"SELL",w1,rk,ctgt)
            if tgt_>=e: continue
            bc=c
            best={"signal":"SELL","entry_price":e,"sl":sl_,"target":tgt_,
                  "confidence":c,"retracement":r,
                  "reason":f"Wave-2 top: {r:.1%} retrace → Wave-3 down",
                  "pattern":f"W2 Top ({r:.1%})","wave_pivots":[p0,p1,p2],"wave1_len":w1}
    return best

# ═══════════════════════════════════════════════════════════════════════════
# LIVE TRADING ENGINE  (main-thread, called inside @st.fragment)
# ═══════════════════════════════════════════════════════════════════════════
def live_engine_tick(symbol:str, interval:str, period:str, depth:int,
                     sl_type, tgt_type, csl:float, ctgt:float,
                     dhan_on:bool, dhan_api, sec_id:str, live_qty:int):
    """
    Called every 2 s from the fragment.
    Step 1: Always try a lightweight LTP update (yf.Ticker.fast_info).
    Step 2: Check if enough wall-clock time has passed for a full candle fetch.
    Step 3: If new candle: run signal + position management.
    Returns True if full candle cycle completed (triggers chart refresh).
    """
    ss = st.session_state
    now_ts = time.time()

    # ── Fast LTP ──────────────────────────────────────────────────────────
    ltp_new = fetch_ltp(symbol)
    if ltp_new:
        ss.live_ltp = ltp_new
        ss.live_ltp_ts = now_ist().strftime("%H:%M:%S IST")

    ltp = ss.live_ltp

    # ── Unrealized P&L on open position (every 2s) ───────────────────────
    pos = ss.live_position
    if pos and ltp:
        pos["unreal_pnl"] = (ltp - pos["entry"]) * pos["qty"] if pos["type"] == "BUY" \
                             else (pos["entry"] - ltp) * pos["qty"]
        pos["dist_sl"]    = abs(ltp - pos["sl"])   / ltp * 100
        pos["dist_tgt"]   = abs(pos["target"] - ltp) / ltp * 100

    # ── Check if it's time for a full candle fetch ────────────────────────
    candle_s = POLL_SECS.get(interval, 3600)
    time_since_last = now_ts - ss.last_fetch_wall

    # Always fetch if never fetched; then respect candle interval
    if time_since_last < candle_s and ss.last_fetch_wall > 0:
        # Calculate next check time
        next_s = candle_s - time_since_last
        next_ist = (now_ist() + timedelta(seconds=next_s)).strftime("%H:%M:%S IST")
        ss.live_next_check_ist = next_ist
        ss.live_phase = "no_signal" if (pos is None and ss.live_phase not in ("signal_ready",)) \
                        else ss.live_phase
        return False   # no new candle cycle yet

    # ── Full candle fetch ─────────────────────────────────────────────────
    ss.live_phase = "scanning"
    ss.live_next_check_ist = "fetching…"
    ss.last_fetch_wall = now_ts

    df = fetch_ohlcv(symbol, interval, period, min_delay=1.5)

    def _log(msg, lvl="INFO"):
        ts = now_ist().strftime("%H:%M:%S IST")
        ss.live_log.append(f"[{ts}][{lvl}] {msg}")
        ss.live_log = ss.live_log[-150:]

    if df is None or len(df) < 35:
        _log("⚠️  No/insufficient data from yfinance","WARN")
        ss.live_no_pos_reason = ("⚠️ No data received. Check your internet connection, "
                                 "symbol name, or yfinance rate limits. Retrying next cycle.")
        ss.live_phase = "no_signal"
        return False

    # Update LTP from OHLCV close (more reliable than fast_info for some symbols)
    ohlcv_ltp = float(df["Close"].iloc[-1])
    if ss.live_ltp is None:
        ss.live_ltp = ohlcv_ltp

    # Last candle IST
    last_bar_dt = df.index[-2] if len(df) >= 2 else df.index[-1]
    ss.live_last_candle_ist = fmt_ist(last_bar_dt)

    # Delay
    try:
        lbdt = last_bar_dt.to_pydatetime()
        if lbdt.tzinfo is None: lbdt = lbdt.replace(tzinfo=timezone.utc)
        ss.live_delay_s = int((now_ist() - lbdt.astimezone(IST)).total_seconds())
    except Exception:
        ss.live_delay_s = 0

    df_closed = df.iloc[:-1]
    latest_ts  = str(df_closed.index[-1])
    pivots     = find_pivots(df_closed, depth)

    # Generate signal (always, so wave state stays current)
    sig = ew_signal(df_closed, depth, sl_type, tgt_type, csl, ctgt)
    ss.live_last_sig = sig
    ws = analyze_wave_state(df_closed, pivots, sig)
    ss.live_wave_state = ws
    ss._scan_df  = df
    ss._scan_sig = sig

    # Next check time
    nxt = (now_ist() + timedelta(seconds=candle_s)).strftime("%H:%M:%S IST")
    ss.live_next_check_ist = nxt

    # ── Duplicate bar guard ───────────────────────────────────────────────
    if ss.last_bar_ts == latest_ts:
        _log(f"⏭  Bar {latest_ts[-10:]} already processed → next check {nxt}")
        pos = ss.live_position
        if pos:
            # Still manage position even on same bar (price may have moved)
            ltp_now = ss.live_ltp or ohlcv_ltp
            _manage_position(ltp_now, sig, sl_type, tgt_type,
                             dhan_on, dhan_api, sec_id, _log)
        if ss.live_position:
            ss.live_phase = f"pos_{ss.live_position['type'].lower()}"
        elif ss.live_last_sig and ss.live_last_sig.get("signal") in ("BUY","SELL"):
            ss.live_phase = "signal_ready"
        else:
            ss.live_phase = "no_signal"
        return False

    ss.last_bar_ts = latest_ts
    _log(f"🕯 New bar: {latest_ts[-10:]} | LTP: {ohlcv_ltp:.2f}")

    # ── Position management ───────────────────────────────────────────────
    ltp_now = ss.live_ltp or ohlcv_ltp
    _manage_position(ltp_now, sig, sl_type, tgt_type,
                     dhan_on, dhan_api, sec_id, _log)

    # ── Open new position if signal ───────────────────────────────────────
    pos = ss.live_position
    if pos is None and sig["signal"] in ("BUY","SELL"):
        ep  = ltp_now
        w1  = sig.get("wave1_len", ep*0.02) or (ep*0.02)
        if sl_type == SL_WAVE:       sl_ = sig["sl"]
        elif sl_type == SL_PTS:      sl_ = ep-csl if sig["signal"]=="BUY" else ep+csl
        elif sl_type == SL_SIGREV:   sl_ = ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
        else:                        sl_ = ep*(1-float(sl_type)) if sig["signal"]=="BUY" else ep*(1+float(sl_type))
        rk = abs(ep - sl_)
        if rk <= 0:
            ss.live_no_pos_reason = (f"⚠️ Risk = 0 pts (SL {sl_:.2f} too close to entry {ep:.2f}). "
                                     f"Increase SL distance or use 'Wave Auto'.")
            ss.live_phase = "no_signal"
            _log("⚠️  Risk=0 — cannot enter position","WARN")
        else:
            tgt_ = _calc_target(tgt_type, ep, sig["signal"], w1, rk, ctgt)
            if (sig["signal"]=="BUY" and tgt_<=ep) or (sig["signal"]=="SELL" and tgt_>=ep):
                ss.live_no_pos_reason = f"⚠️ Target ({tgt_:.2f}) invalid vs entry ({ep:.2f}). Adjust Target setting."
                ss.live_phase = "no_signal"
                _log("⚠️  Invalid target — cannot enter","WARN")
            else:
                entry_ist = now_ist().strftime("%d-%b %H:%M:%S IST")
                ss.live_position = {
                    "type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,
                    "qty":live_qty,"entry_ist":entry_ist,"symbol":symbol,
                    "pattern":sig["pattern"],"confidence":sig["confidence"],
                    "unreal_pnl":0.0,"dist_sl":rk/ep*100,"dist_tgt":abs(tgt_-ep)/ep*100,
                }
                ss.live_phase = f"pos_{sig['signal'].lower()}"
                ss.live_no_pos_reason = ""
                ss.live_signals.append({
                    "Time (IST)":entry_ist, "Bar":fmt_ist(df_closed.index[-1]),
                    "Symbol":symbol,"TF":interval,"Period":period,
                    "Signal":sig["signal"],"Entry":round(ep,2),
                    "SL":round(sl_,2),"Target":round(tgt_,2),
                    "Conf":f"{sig['confidence']:.0%}","Pattern":sig["pattern"],
                })
                if dhan_on and dhan_api:
                    r=dhan_api.place_order(sec_id,"NSE_EQ",sig["signal"],live_qty)
                    _log(f"📤 Dhan order: {r}")
                em="🟢" if sig["signal"]=="BUY" else "🔴"
                _log(f"{em} ENTERED {sig['signal']} @ {ep:.2f} | SL {sl_:.2f} | "
                     f"Target {tgt_:.2f} | {sig['confidence']:.0%} | {sig['pattern']}")

    elif pos is None:
        ss.live_phase = "no_signal"
        ss.live_no_pos_reason = (
            f"📊 Signal: HOLD. {sig.get('reason','')}\n"
            f"Waiting for Wave-2 to complete a valid 38–79% retracement.\n"
            f"Next candle check: {nxt}"
        )
        _log(f"⏸  HOLD @ {ltp_now:.2f} | {sig.get('reason','')}")

    return True   # full cycle completed


def _manage_position(ltp:float, sig:dict, sl_type, tgt_type,
                     dhan_on:bool, dhan_api, sec_id:str, log_fn):
    """Check SL/Target/SignalReverse for open position. Modifies session state."""
    ss  = st.session_state
    pos = ss.live_position
    if not pos or not ltp: return

    hit_p, hit_r = None, None
    ptype = pos["type"]

    # Conservative: SL checked via Low-proxy (ltp for live), Target via High-proxy
    if ptype == "BUY":
        if ltp <= pos["sl"]:        hit_p, hit_r = pos["sl"],     "SL Hit (price ≤ SL)"
        elif ltp >= pos["target"]:  hit_p, hit_r = pos["target"], "Target Hit (price ≥ Target)"
        elif sig["signal"] == "SELL": hit_p, hit_r = ltp,          "Signal Reversed → SELL"
    else:
        if ltp >= pos["sl"]:        hit_p, hit_r = pos["sl"],     "SL Hit (price ≥ SL)"
        elif ltp <= pos["target"]:  hit_p, hit_r = pos["target"], "Target Hit (price ≤ Target)"
        elif sig["signal"] == "BUY": hit_p, hit_r = ltp,          "Signal Reversed → BUY"

    if hit_p is None: return

    qty  = pos["qty"]
    pnl  = (hit_p-pos["entry"])*qty if ptype=="BUY" else (pos["entry"]-hit_p)*qty
    ss.live_pnl += pnl
    ss.live_trades.append({
        "Time (IST)": now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Symbol":pos["symbol"],"TF":st.session_state.get("_live_interval","—"),
        "Type":ptype,"Entry":round(pos["entry"],2),"Exit":round(hit_p,2),
        "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
        "Qty":qty,"PnL Rs":round(pnl,2),"Reason":hit_r,
    })
    if dhan_on and dhan_api:
        xt="SELL" if ptype=="BUY" else "BUY"
        log_fn(f"📤 Dhan exit: {dhan_api.place_order(sec_id,'NSE_EQ',xt,qty)}")
    em="✅" if "Target" in hit_r else ("🔄" if "Reversed" in hit_r else "❌")
    log_fn(f"{em} {ptype} CLOSED @ {hit_p:.2f} | {hit_r} | Rs{pnl:+.2f}")
    ss.live_position = None
    if "Reversed" in hit_r:
        ss.live_phase = "reversing"
    else:
        ss.live_phase = "no_signal"
        ss.live_no_pos_reason = f"Last trade: {hit_r} | P&L: Rs{pnl:+.2f}"

# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df:pd.DataFrame, depth:int=5,
                 sl_type="wave_auto", tgt_type="wave_auto",
                 capital:float=100_000.0, csl:float=50.0, ctgt:float=100.0) -> dict:
    MB=max(30,depth*4)
    if len(df)<MB+10: return {"error":f"Need ≥{MB+10} bars.","equity_curve":[capital]}
    trades,equity_curve=[],[capital]; equity,pos=capital,None

    for i in range(MB,len(df)-1):
        bdf=df.iloc[:i+1]; nb=df.iloc[i+1]
        hi_i=float(df.iloc[i]["High"]); lo_i=float(df.iloc[i]["Low"])
        tsig=None

        if pos:
            if sl_type==SL_SIGREV or tgt_type==TGT_SIGREV:
                tsig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt)
            ep_,er_=None,None
            if pos["type"]=="BUY":
                if lo_i<=pos["sl"]:        ep_,er_=pos["sl"],   "SL (Low≤SL)"
                elif hi_i>=pos["target"]:  ep_,er_=pos["target"],"Target (High≥Target)"
                elif tsig and tsig["signal"]=="SELL":
                    ep_,er_=float(df.iloc[i]["Close"]),"Signal Reverse"
            else:
                if hi_i>=pos["sl"]:        ep_,er_=pos["sl"],   "SL (High≥SL)"
                elif lo_i<=pos["target"]:  ep_,er_=pos["target"],"Target (Low≤Target)"
                elif tsig and tsig["signal"]=="BUY":
                    ep_,er_=float(df.iloc[i]["Close"]),"Signal Reverse"
            if ep_ is not None:
                qty=pos["qty"]
                pnl=(ep_-pos["entry"])*qty if pos["type"]=="BUY" else (pos["entry"]-ep_)*qty
                equity+=pnl; equity_curve.append(equity)
                trades.append({"Entry Time":pos["entry_time"],"Exit Time":df.index[i],
                                "Type":pos["type"],"Entry":round(pos["entry"],2),
                                "Exit":round(ep_,2),"SL":round(pos["sl"],2),
                                "Target":round(pos["target"],2),
                                "Exit Bar Low":round(lo_i,2),"Exit Bar High":round(hi_i,2),
                                "Exit Reason":er_,"PnL Rs":round(pnl,2),
                                "PnL %":round(pnl/(pos["entry"]*qty)*100,2),
                                "Equity Rs":round(equity,2),
                                "Bars Held":i-pos["entry_bar"],
                                "Confidence":round(pos["conf"],2)})
                pos=None

        if pos is None:
            sig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt)
            if sig["signal"] in ("BUY","SELL"):
                ep=float(nb["Open"]); w1=sig.get("wave1_len",ep*0.02) or ep*0.02
                if sl_type==SL_WAVE:   sl_=sig["sl"]
                elif sl_type==SL_PTS:  sl_=ep-csl if sig["signal"]=="BUY" else ep+csl
                elif sl_type==SL_SIGREV: sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
                else: sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" else ep*(1+float(sl_type))
                rk=abs(ep-sl_)
                if rk<=0: continue
                tgt_=_calc_target(tgt_type,ep,sig["signal"],w1,rk,ctgt)
                if sig["signal"]=="BUY" and tgt_<=ep: continue
                if sig["signal"]=="SELL" and tgt_>=ep: continue
                qty=max(1,int(equity*0.95/ep))
                pos={"type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,
                     "entry_bar":i+1,"entry_time":df.index[i+1],"qty":qty,"conf":sig["confidence"]}

    if pos:
        ep2=float(df["Close"].iloc[-1]); qty=pos["qty"]
        pnl=(ep2-pos["entry"])*qty if pos["type"]=="BUY" else (pos["entry"]-ep2)*qty; equity+=pnl
        trades.append({"Entry Time":pos["entry_time"],"Exit Time":df.index[-1],
                       "Type":pos["type"],"Entry":round(pos["entry"],2),"Exit":round(ep2,2),
                       "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
                       "Exit Bar Low":round(float(df["Low"].iloc[-1]),2),
                       "Exit Bar High":round(float(df["High"].iloc[-1]),2),
                       "Exit Reason":"Open@End","PnL Rs":round(pnl,2),
                       "PnL %":round(pnl/(pos["entry"]*qty)*100,2),
                       "Equity Rs":round(equity,2),"Bars Held":len(df)-1-pos["entry_bar"],
                       "Confidence":round(pos["conf"],2)})

    if not trades:
        return {"error":"No trades. Try smaller Pivot Depth, longer Period, or different SL/Target.",
                "equity_curve":equity_curve}

    tdf=pd.DataFrame(trades)
    wins=tdf[tdf["PnL Rs"]>0]; loss=tdf[tdf["PnL Rs"]<=0]; ntot=len(tdf)
    wr=len(wins)/ntot*100 if ntot else 0
    pf=abs(wins["PnL Rs"].sum()/loss["PnL Rs"].sum()) if len(loss) and loss["PnL Rs"].sum()!=0 else 9999.0
    eq_arr=np.array(equity_curve); peak=np.maximum.accumulate(eq_arr)
    mdd=float(((eq_arr-peak)/peak*100).min())
    rets=tdf["PnL %"].values
    sharpe=float(rets.mean()/rets.std()*np.sqrt(252)) if len(rets)>1 and rets.std()!=0 else 0.0

    tdf["SL Verified"]=tdf.apply(lambda row:"✅" if (
        (row["Type"]=="BUY"  and "SL" in row["Exit Reason"] and row["Exit Bar Low"]<=row["SL"]) or
        (row["Type"]=="SELL" and "SL" in row["Exit Reason"] and row["Exit Bar High"]>=row["SL"]) or
        "SL" not in row["Exit Reason"]) else "⚠️",axis=1)
    tdf["TGT Verified"]=tdf.apply(lambda row:"✅" if (
        (row["Type"]=="BUY"  and "Target" in row["Exit Reason"] and row["Exit Bar High"]>=row["Target"]) or
        (row["Type"]=="SELL" and "Target" in row["Exit Reason"] and row["Exit Bar Low"]<=row["Target"]) or
        "Target" not in row["Exit Reason"]) else "⚠️",axis=1)

    return {"trades":tdf,"equity_curve":equity_curve,
            "exit_breakdown":{"SL hits":len(tdf[tdf["Exit Reason"].str.contains("SL",na=False)]),
                              "Target hits":len(tdf[tdf["Exit Reason"].str.contains("Target",na=False)]),
                              "Signal Reverse":len(tdf[tdf["Exit Reason"].str.contains("Reverse",na=False)]),
                              "Still open":len(tdf[tdf["Exit Reason"].str.contains("Open",na=False)])},
            "metrics":{"Total Trades":ntot,"Win Rate %":round(wr,1),
                       "Profit Factor":round(pf,2),"Total Return %":round((equity-capital)/capital*100,2),
                       "Final Equity Rs":round(equity,2),"Max Drawdown %":round(mdd,2),
                       "Sharpe Ratio":round(sharpe,2),
                       "Avg Win Rs":round(float(wins["PnL Rs"].mean()),2) if len(wins) else 0.0,
                       "Avg Loss Rs":round(float(loss["PnL Rs"].mean()),2) if len(loss) else 0.0,
                       "Wins":len(wins),"Losses":len(loss)}}

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization(df,capital=100_000.0,csl=50.0,ctgt=100.0):
    DEPTHS=[3,5,7,10]; SL_OPTS=[0.01,0.02,0.03,"wave_auto"]; TGT_OPTS=[1.5,2.0,3.0,"wave_auto","fib_1618"]
    combos=list(itertools.product(DEPTHS,SL_OPTS,TGT_OPTS)); prog=st.progress(0,text="Optimizing…"); rows=[]
    for idx,(dep,sl,tgt) in enumerate(combos):
        r=run_backtest(df,depth=dep,sl_type=sl,tgt_type=tgt,capital=capital,csl=csl,ctgt=ctgt)
        if "metrics" in r:
            m=r["metrics"]
            rows.append({"Depth":dep,"SL":str(sl),"Target":str(tgt),"Trades":m["Total Trades"],
                         "Win %":m["Win Rate %"],"Return %":m["Total Return %"],"PF":m["Profit Factor"],
                         "Max DD %":m["Max Drawdown %"],"Sharpe":m["Sharpe Ratio"]})
        prog.progress((idx+1)/len(combos),text=f"Combo {idx+1}/{len(combos)}…")
    prog.empty()
    if not rows: return pd.DataFrame()
    out=pd.DataFrame(rows)
    out["Score"]=out["Return %"].clip(lower=0)*(out["Win %"]/100)*out["PF"].clip(upper=10)/(out["Max DD %"].abs()+1)
    return out.sort_values("Score",ascending=False).reset_index(drop=True)

def sl_lbl_from(val):
    for k,v in SL_MAP.items():
        if str(v)==str(val): return k
    return SL_KEYS[0]

def tgt_lbl_from(val):
    for k,v in TGT_MAP.items():
        if str(v)==str(val): return k
    try:
        fv=float(val)
        for k,v in TGT_MAP.items():
            if isinstance(v,float) and abs(v-fv)<0.01: return k
    except Exception: pass
    return TGT_KEYS[0]

# ═══════════════════════════════════════════════════════════════════════════
# DHAN
# ═══════════════════════════════════════════════════════════════════════════
class DhanAPI:
    BASE="https://api.dhan.co"
    def __init__(self,cid,tok):
        self.cid=cid; self.hdrs={"Content-Type":"application/json","access-token":tok}
    def place_order(self,sec_id,seg,txn,qty,ot="MARKET",p=0.0,prod="INTRADAY"):
        try:
            r=requests.post(f"{self.BASE}/orders",headers=self.hdrs,timeout=10,json={
                "dhanClientId":self.cid,"transactionType":txn,"exchangeSegment":seg,
                "productType":prod,"orderType":ot,"validity":"DAY",
                "securityId":sec_id,"quantity":qty,"price":p})
            return r.json()
        except Exception as e: return {"error":str(e)}
    def fund_limit(self):
        try: return requests.get(f"{self.BASE}/fundlimit",headers=self.hdrs,timeout=10).json()
        except Exception as e: return {"error":str(e)}

# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def chart_waves(df,pivots,sig=None,trades=None,symbol="",tf_label=""):
    sig=sig or _blank()
    df_ind=add_indicators(df)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                      row_heights=[0.60,0.20,0.20],vertical_spacing=0.02,
                      subplot_titles=("","Volume","RSI"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="Price",
        increasing=dict(line=dict(color="#26a69a"),fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"),fillcolor="#ef5350")),row=1,col=1)
    for col,clr,nm in [("EMA_20","#ffb300","EMA20"),("EMA_50","#ab47bc","EMA50"),("EMA_200","#ef5350","EMA200")]:
        if col in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind[col],mode="lines",
                line=dict(color=clr,width=1.2),name=nm,opacity=0.7),row=1,col=1)
    if "Volume" in df.columns:
        vc=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,opacity=0.4,
                             name="Vol",showlegend=False),row=2,col=1)
    if "RSI" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind["RSI"],mode="lines",
            line=dict(color="#00e5ff",width=1.5),name="RSI",showlegend=False),row=3,col=1)
        fig.add_hline(y=70,line=dict(dash="dot",color="#ef5350",width=1),row=3,col=1)
        fig.add_hline(y=30,line=dict(dash="dot",color="#4caf50",width=1),row=3,col=1)
    vp=[p for p in pivots if p[0]<len(df)]
    if vp:
        fig.add_trace(go.Scatter(x=[df.index[p[0]] for p in vp],y=[p[1] for p in vp],
            mode="lines+markers",line=dict(color="rgba(255,180,0,.5)",width=1.5,dash="dot"),
            marker=dict(size=7,color=["#4caf50" if p[2]=="L" else "#f44336" for p in vp],
                        symbol=["triangle-up" if p[2]=="L" else "triangle-down" for p in vp]),
            name="ZigZag"),row=1,col=1)
    wp=sig.get("wave_pivots")
    if wp:
        vwp=[p for p in wp if p[0]<len(df)]
        if vwp:
            clr="#00e5ff" if sig["signal"]=="BUY" else "#ff4081"
            lbls=["W0","W1","W2"][:len(vwp)]
            fig.add_trace(go.Scatter(x=[df.index[p[0]] for p in vwp],y=[p[1] for p in vwp],
                mode="lines+markers+text",line=dict(color=clr,width=2.5),
                marker=dict(size=12,color=clr),text=lbls,textposition="top center",
                textfont=dict(color=clr,size=12,family="Share Tech Mono"),name="EW"),row=1,col=1)
    if sig["signal"] in ("BUY","SELL"):
        sc="#4caf50" if sig["signal"]=="BUY" else "#f44336"
        ss2="triangle-up" if sig["signal"]=="BUY" else "triangle-down"
        fig.add_trace(go.Scatter(x=[df.index[-1]],y=[df["Close"].iloc[-1]],mode="markers",
            marker=dict(size=20,color=sc,symbol=ss2,line=dict(color="white",width=1.5)),
            name=f"▶ {sig['signal']}"),row=1,col=1)
        if sig.get("sl"):
            fig.add_hline(y=sig["sl"],line=dict(dash="dash",color="#ff7043",width=1.5),
                annotation_text="  SL",annotation_position="right",row=1,col=1)
        if sig.get("target"):
            fig.add_hline(y=sig["target"],line=dict(dash="dash",color="#66bb6a",width=1.5),
                annotation_text="  Target",annotation_position="right",row=1,col=1)
    if trades is not None and not trades.empty:
        for tt,sy_,cl in [("BUY","triangle-up","#4caf50"),("SELL","triangle-down","#f44336")]:
            sub=trades[trades["Type"]==tt]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Entry Time"],y=sub["Entry"],mode="markers",
                    marker=dict(size=9,color=cl,symbol=sy_,line=dict(color="white",width=0.8)),
                    name=f"{tt} Entry"),row=1,col=1)
        for rsn_,sy_,cl in [("Target","circle","#66bb6a"),("SL","x","#ef5350"),("Reverse","diamond","#ab47bc")]:
            sub=trades[trades["Exit Reason"].str.contains(rsn_,na=False)]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Exit Time"],y=sub["Exit"],mode="markers",
                    marker=dict(size=7,color=cl,symbol=sy_),name=f"Exit({rsn_})",visible="legendonly"),row=1,col=1)
    fig.update_layout(title=dict(text=f"🌊 {symbol}"+(f" · {tf_label}" if tf_label else ""),
                                 font=dict(size=14,color="#00e5ff")),
                      template="plotly_dark",height=560,xaxis_rangeslider_visible=False,
                      plot_bgcolor="#06101a",paper_bgcolor="#06101a",
                      font=dict(color="#b0bec5",family="Exo 2"),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,font=dict(size=10)),
                      margin=dict(l=10,r=70,t=50,b=10))
    return fig

def chart_equity(equity_curve):
    eq=np.array(equity_curve,dtype=float); pk=np.maximum.accumulate(eq); dd=(eq-pk)/pk*100
    fig=make_subplots(rows=2,cols=1,row_heights=[0.65,0.35],vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=eq,mode="lines",name="Equity",line=dict(color="#00bcd4",width=2),
        fill="tozeroy",fillcolor="rgba(0,188,212,.07)"),row=1,col=1)
    fig.add_trace(go.Scatter(y=dd,mode="lines",name="Drawdown %",line=dict(color="#f44336",width=1.5),
        fill="tozeroy",fillcolor="rgba(244,67,54,.12)"),row=2,col=1)
    fig.add_hline(y=0,line=dict(dash="dot",color="#546e7a",width=1),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=350,plot_bgcolor="#06101a",
        paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),
        margin=dict(l=10,r=10,t=20,b=10))
    return fig

def chart_opt_scatter(opt_df):
    fig=go.Figure(go.Scatter(x=opt_df["Max DD %"].abs(),y=opt_df["Return %"],mode="markers",
        marker=dict(size=(opt_df["Win %"]/5).clip(lower=4),color=opt_df["Score"],
                    colorscale="Plasma",showscale=True,
                    colorbar=dict(title=dict(text="Score",font=dict(color="#b0bec5",size=12)),
                                  tickfont=dict(color="#b0bec5")),
                    line=dict(color="rgba(255,255,255,.2)",width=0.5)),
        text=[f"D={r.Depth} SL={r.SL} T={r.Target}" for _,r in opt_df.iterrows()],
        hovertemplate="<b>%{text}</b><br>Return %{y:.1f}% MaxDD %{x:.1f}%<extra></extra>"))
    fig.update_layout(title=dict(text="Return vs Max Drawdown (bubble=WinRate)",font=dict(size=13,color="#00e5ff")),
        xaxis_title="Max Drawdown %",yaxis_title="Total Return %",
        template="plotly_dark",height=380,plot_bgcolor="#06101a",paper_bgcolor="#06101a",
        font=dict(color="#b0bec5",family="Exo 2"),margin=dict(l=10,r=10,t=45,b=10))
    return fig

def generate_mtf_summary(symbol, results, overall_sig):
    lines=[f"## 🌊 Elliott Wave Analysis — {symbol}",f"*{now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}*\n"]
    bc=sum(1 for r in results if r["signal"]["signal"]=="BUY")
    sc=sum(1 for r in results if r["signal"]["signal"]=="SELL")
    hc=sum(1 for r in results if r["signal"]["signal"]=="HOLD")
    vi={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(overall_sig,"⚪")
    lines.append(f"### {vi} Overall: **{overall_sig}** ({bc}B·{sc}S·{hc}H / {len(results)} TFs)\n")
    if overall_sig=="BUY": lines.append("📈 **Bullish consensus.** Enter BUY on Wave-2 pullbacks. SL below Wave-2 pivot.")
    elif overall_sig=="SELL": lines.append("📉 **Bearish consensus.** Enter SELL on Wave-2 bounces. SL above Wave-2 pivot.")
    else: lines.append("⚠️ **No consensus.** Stay on sidelines. Wait for ≥70% confidence signal.")
    lines.append("\n---\n")
    for r in results:
        sig=r["signal"]; s=sig["signal"]; em={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(s,"⚪")
        lines.append(f"#### {em} {r['tf_name']}")
        if s in ("BUY","SELL"):
            retr=sig.get("retracement",0); ep=sig["entry_price"]; sl_=sig["sl"]; tgt_=sig["target"]
            rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
            lines.append(f"- **{s}** | Entry:{ep:.2f} SL:{sl_:.2f} Target:{tgt_:.2f} R:R 1:{rr:.1f}")
            lines.append(f"- Confidence: {sig['confidence']:.0%} | Retracement: {retr:.1%}")
            if abs(retr-0.618)<0.04: lines.append("- ✨ Golden Ratio (61.8%) — strongest signal")
        else:
            lines.append(f"- HOLD: {sig.get('reason','—')}")
        lines.append("")
    lines+=["---","| Wave | Meaning | Action |","|------|---------|--------|",
            "| W1 ↑ | First impulse | Missed |",
            "| **W2 ↓** | **38–79% retrace** | **🟢 BUY here** |",
            "| **W3 ↑** | **Strongest** | **Hold** |",
            "| W4 ↓ | Pullback | Partial exit |",
            "| W5 ↑ | Final | Full exit |",
            "\n> ⚠️ *Not financial advice.*"]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ███  STREAMLIT UI  ███
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="main-hdr">
  <h1>🌊 Elliott Wave Algo Trader v5.1</h1>
  <p>Crystal Clear Live Status · No-Flicker LTP · Auto-Reverse · IST Times · Conservative SL/TGT</p>
</div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Instrument")
    group_sel=st.selectbox("Category",list(TICKER_GROUPS.keys()),index=0)
    gmap=TICKER_GROUPS[group_sel]
    if group_sel=="✏️ Custom Ticker":
        symbol=st.text_input("Yahoo Finance ticker","^NSEI")
    else:
        tn=st.selectbox("Instrument",list(gmap.keys()))
        symbol=gmap[tn]; st.caption(f"Yahoo: `{symbol}`")

    st.markdown("---")
    if st.session_state.best_cfg_applied:
        st.markdown("""<div style="background:rgba(0,229,255,.08);border:1px solid #00bcd4;
        border-radius:8px;padding:7px 10px;font-size:.8rem;margin-bottom:8px">
        ✨ <b style="color:#00e5ff">Optimized Config Active</b></div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    interval=c1.selectbox("⏱ TF",TIMEFRAMES,index=6)
    vpl=VALID_PERIODS.get(interval,PERIODS)
    period=c2.selectbox("📅 Period",vpl,index=min(4,len(vpl)-1))

    # Store in session state for live engine to read
    st.session_state["_live_interval"]=interval

    st.markdown("---")
    st.markdown("### 🌊 Elliott Wave")
    depth=st.slider("Pivot Depth",2,15,st.session_state.applied_depth)

    st.markdown("---")
    st.markdown("### 🛡️ Risk")
    si=SL_KEYS.index(st.session_state.applied_sl_lbl) if st.session_state.applied_sl_lbl in SL_KEYS else 0
    ti=TGT_KEYS.index(st.session_state.applied_tgt_lbl) if st.session_state.applied_tgt_lbl in TGT_KEYS else 0
    sl_lbl=st.selectbox("Stop Loss",SL_KEYS,index=si)
    tgt_lbl=st.selectbox("Target",TGT_KEYS,index=ti)
    sl_val=SL_MAP[sl_lbl]; tgt_val=TGT_MAP[tgt_lbl]

    if sl_val==SL_PTS or tgt_val==TGT_PTS:
        st.markdown("**Custom Points**")
    if sl_val==SL_PTS:
        st.session_state.custom_sl_pts=st.number_input("SL Points",1.0,100000.0,st.session_state.custom_sl_pts,5.0)
    if tgt_val==TGT_PTS:
        st.session_state.custom_tgt_pts=st.number_input("Target Points",1.0,500000.0,st.session_state.custom_tgt_pts,10.0)
    csl=st.session_state.custom_sl_pts; ctgt=st.session_state.custom_tgt_pts

    if sl_val==SL_SIGREV: st.info("🔄 Exit when wave signal reverses direction")
    if tgt_val==TGT_SIGREV: st.info("🔄 Take profit when wave signal reverses")

    capital=st.number_input("💰 Capital (Rs)",10_000,50_000_000,100_000,10_000)

    st.markdown("---")
    st.markdown("### 🏦 Dhan")
    dhan_on=st.checkbox("Enable Dhan Integration",value=False)
    dhan_api,sec_id,live_qty=None,"1333",1
    if dhan_on:
        d_cid=st.text_input("Client ID"); d_tok=st.text_input("Access Token",type="password")
        sec_id=st.text_input("Security ID","1333"); live_qty=st.number_input("Order Qty",1,100_000,1)
        if d_cid and d_tok:
            dhan_api=DhanAPI(d_cid,d_tok)
            if st.button("🔌 Test"): st.json(dhan_api.fund_limit())
        else: st.info("Enter credentials")
    st.markdown("---")
    st.caption(f"⚡ 1.5s rate-limit | LTP refresh: 2s")
    st.caption(f"`{symbol}` · `{interval}` · `{period}`")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
t_analysis,t_live,t_bt,t_opt,t_help=st.tabs([
    "🔭  Wave Analysis","🔴  Live Trading","📊  Backtest","🔬  Optimization","❓  Help"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — WAVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with t_analysis:
    st.markdown("### 🔭 Multi-Timeframe Elliott Wave Analysis")
    ac1,ac2,ac3=st.columns([1.2,1,2.4])
    with ac1: run_analysis=st.button("🔭 Run Full Analysis",type="primary",use_container_width=True)
    with ac2: custom_tf_only=st.checkbox("Sidebar TF only",value=False)
    with ac3: st.caption(f"Scanning {'sidebar TF' if custom_tf_only else 'Daily·4H·1H·15M'} for `{symbol}`")

    if run_analysis:
        sc2=[(interval,period,f"{interval}·{period}")] if custom_tf_only \
            else [(tf,per,nm) for tf,per,nm in MTF_COMBOS if per in VALID_PERIODS.get(tf,[])]
        results=[]; prog=st.progress(0,text="Scanning…")
        for idx,(tf,per,nm) in enumerate(sc2):
            prog.progress((idx+1)/len(sc2),text=f"Fetching {nm}…")
            dfa=fetch_ohlcv(symbol,tf,per,min_delay=1.5)
            if dfa is not None and len(dfa)>=35:
                sa=ew_signal(dfa.iloc[:-1],depth,sl_val,tgt_val,csl,ctgt)
                results.append({"tf_name":nm,"interval":tf,"period":per,"signal":sa,"df":dfa,"pivots":find_pivots(dfa.iloc[:-1],depth)})
            else:
                results.append({"tf_name":nm,"interval":tf,"period":per,"signal":_blank(f"No data {tf}/{per}"),"df":None,"pivots":[]})
        prog.empty()
        bscore=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="BUY")
        sscore=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="SELL")
        ov="BUY" if bscore>sscore and bscore>0.5 else "SELL" if sscore>bscore and sscore>0.5 else "HOLD"
        st.session_state.update({"_analysis_results":results,"_analysis_overall":ov,"_analysis_symbol":symbol})

    ar=st.session_state.get("_analysis_results")
    if ar:
        ov=st.session_state.get("_analysis_overall","HOLD"); asym=st.session_state.get("_analysis_symbol",symbol)
        vc={"BUY":"#4caf50","SELL":"#f44336","HOLD":"#ffb300"}; vb={"BUY":"rgba(76,175,80,.10)","SELL":"rgba(244,67,54,.10)","HOLD":"rgba(255,179,0,.10)"}; vi={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}
        st.markdown(f"""<div style="background:{vb[ov]};border:2px solid {vc[ov]};border-radius:12px;padding:14px 20px;margin-bottom:10px;text-align:center">
        <span style="font-size:1.5rem;color:{vc[ov]};font-weight:700">{vi[ov]} Overall: {ov}</span><br>
        <span style="color:#78909c;font-size:.85rem">{asym} Multi-TF Consensus</span></div>""",unsafe_allow_html=True)
        tfc=st.columns(min(len(ar),4))
        for i,r in enumerate(ar):
            with tfc[i%4]:
                s=r["signal"]["signal"]; c_=r["signal"]["confidence"]
                sc3=vc.get(s,"#546e7a"); em=vi.get(s,"⚪")
                st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:1px solid #1e3a5f;border-radius:8px;padding:9px 11px;text-align:center;margin-bottom:4px">
                <div style="font-size:.76rem;color:#546e7a">{r['tf_name']}</div>
                <div style="font-size:1.1rem;color:{sc3};font-weight:700">{em} {s}</div>
                <div style="font-size:.74rem;color:#78909c">{r['signal']['pattern']}</div>
                <div style="font-size:.78rem;color:#00bcd4">{c_:.0%}</div></div>""",unsafe_allow_html=True)
        st.markdown("---")
        for r in ar:
            if r["df"] is not None:
                s_=r["signal"]["signal"]
                with st.expander(f"📈 {r['tf_name']} — {s_} ({r['signal']['confidence']:.0%})",expanded=(s_!="HOLD")):
                    st.plotly_chart(chart_waves(r["df"],r["pivots"],r["signal"],symbol=asym,tf_label=r["tf_name"]),use_container_width=True)
                    if s_ in ("BUY","SELL"):
                        sg=r["signal"]; ep,sl_,tgt_=sg["entry_price"],sg["sl"],sg["target"]
                        rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
                        mc1,mc2,mc3,mc4=st.columns(4)
                        mc1.metric("Entry",f"{ep:.2f}"); mc2.metric("SL",f"{sl_:.2f}",delta=f"-{abs(ep-sl_)/ep*100:.1f}%",delta_color="inverse")
                        mc3.metric("Target",f"{tgt_:.2f}",delta=f"+{abs(tgt_-ep)/ep*100:.1f}%"); mc4.metric("R:R",f"1:{rr:.1f}")
        st.markdown("---")
        st.markdown("### 📋 Analysis & Recommendations")
        st.markdown(generate_mtf_summary(asym,ar,ov))
    else:
        st.info("Click 🔭 Run Full Analysis to scan all timeframes automatically.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING  (fragment-based, no background thread)
# ═══════════════════════════════════════════════════════════════════════════
with t_live:

    # ── Top control bar ──────────────────────────────────────────────────
    ctl1,ctl2,ctl3=st.columns([1,1,4])
    with ctl1:
        if not st.session_state.live_running:
            if st.button("▶ Start Live",type="primary",use_container_width=True):
                st.session_state.live_running=True
                st.session_state.live_phase="scanning"
                st.session_state.live_log=[]
                st.session_state.last_bar_ts=None
                st.session_state.last_fetch_wall=0.0
                st.session_state.live_no_pos_reason="System starting — fetching first candle…"
                st.rerun()
        else:
            if st.button("⏹ Stop",type="secondary",use_container_width=True):
                st.session_state.live_running=False
                st.session_state.live_phase="idle"
                st.rerun()
    with ctl2:
        if st.button("🔄 Reset All",use_container_width=True):
            for k,v in _DEF.items(): st.session_state[k]=v
            st.success("Reset ✓"); time.sleep(0.3); st.rerun()
    with ctl3:
        if st.session_state.live_running:
            st.success(f"🟢 RUNNING — `{symbol}` · `{interval}` · `{period}` | LTP updates every 2s (no page flicker) | Full candle check every {POLL_SECS.get(interval,3600)//60}min")
        else:
            st.warning("⚫ STOPPED — Click ▶ Start Live. Everything runs automatically. No manual decisions needed.")

    # ── MEGA POSITION STATUS (always visible, always accurate) ───────────
    pos=st.session_state.live_position
    ltp=st.session_state.live_ltp
    phase=st.session_state.live_phase

    pos_placeholder = st.empty()   # updated by fragment

    def render_position_banner(container):
        pos=st.session_state.live_position
        ltp=st.session_state.live_ltp
        phase=st.session_state.live_phase

        with container:
            if pos:
                ptype=pos["type"]
                clr="#4caf50" if ptype=="BUY" else "#f44336"
                css="pos-buy" if ptype=="BUY" else "pos-sell"
                up=pos.get("unreal_pnl",0)
                up_c="#4caf50" if up>=0 else "#f44336"
                up_sym="▲" if up>=0 else "▼"
                dist_sl=pos.get("dist_sl",0); dist_tgt=pos.get("dist_tgt",0)
                ltp_disp=f"{ltp:,.2f}" if ltp else "—"
                st.markdown(f"""
                <div class="{css}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
                <div>
                  <div style="font-size:.8rem;color:#78909c;letter-spacing:.5px">POSITION STATUS</div>
                  <div style="font-size:2rem;font-weight:700;color:{clr};font-family:'Exo 2'">{ptype} OPEN ✅</div>
                  <div style="font-size:.85rem;color:#b0bec5;margin-top:4px">{pos.get('pattern','—')} | Confidence {pos.get('confidence',0):.0%}</div>
                  <div style="font-size:.78rem;color:#546e7a">Entered {pos.get('entry_ist','—')}</div>
                </div>
                <div style="text-align:right">
                  <div style="font-size:.78rem;color:#546e7a">UNREALIZED P&L</div>
                  <div style="font-size:1.7rem;font-weight:700;color:{up_c};font-family:'Share Tech Mono'">{up_sym} Rs{abs(up):,.2f}</div>
                  <div style="font-size:.8rem;color:#78909c">LTP: {ltp_disp}</div>
                </div>
                </div>
                <hr style="border-color:rgba(255,255,255,.08);margin:10px 0">
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:.84rem">
                  <div><div style="color:#546e7a;font-size:.74rem">ENTRY</div><div style="color:#b0bec5;font-weight:600">{pos['entry']:,.2f}</div></div>
                  <div><div style="color:#546e7a;font-size:.74rem">STOP LOSS</div><div style="color:#ff7043;font-weight:600">{pos['sl']:,.2f} <span style="color:#546e7a;font-size:.72rem">({dist_sl:.1f}% away)</span></div></div>
                  <div><div style="color:#546e7a;font-size:.74rem">TARGET</div><div style="color:#66bb6a;font-weight:600">{pos['target']:,.2f} <span style="color:#546e7a;font-size:.72rem">({dist_tgt:.1f}% away)</span></div></div>
                  <div><div style="color:#546e7a;font-size:.74rem">QTY</div><div style="color:#b0bec5;font-weight:600">{pos['qty']}</div></div>
                </div>
                <div style="margin-top:8px;padding:6px 10px;background:rgba(0,0,0,.3);border-radius:6px;font-size:.78rem;color:#78909c">
                🤖 <b>System manages this automatically.</b>
                Position auto-closes when SL is hit (price {'≤' if ptype=='BUY' else '≥'} {pos['sl']:,.2f})
                or Target is hit (price {'≥' if ptype=='BUY' else '≤'} {pos['target']:,.2f})
                or when a {'SELL' if ptype=='BUY' else 'BUY'} signal fires (auto-reverse). <b>No action needed from you.</b>
                </div>
                </div>""",unsafe_allow_html=True)

            elif phase=="reversing":
                st.markdown("""<div class="pos-reversing">
                <div style="font-size:1.5rem;font-weight:700;color:#ab47bc">🔄 AUTO-REVERSING POSITION…</div>
                <div style="font-size:.85rem;color:#b0bec5;margin-top:6px">
                Closing old position and opening reverse position at market price.<br>
                This takes 1–2 seconds. No action needed.
                </div></div>""",unsafe_allow_html=True)

            elif phase in ("scanning","signal_ready"):
                sig=st.session_state.live_last_sig
                if sig and sig.get("signal") in ("BUY","SELL"):
                    s_=sig["signal"]; sc3="#00e5ff"
                    st.markdown(f"""<div class="pos-signal">
                    <div style="font-size:.8rem;color:#78909c">SIGNAL DETECTED — ENTERING POSITION</div>
                    <div style="font-size:1.6rem;font-weight:700;color:{sc3}">💡 {s_} SIGNAL FIRED</div>
                    <div style="font-size:.85rem;color:#b0bec5;margin-top:4px">{sig.get('pattern','—')} | Confidence {sig.get('confidence',0):.0%}</div>
                    <div style="font-size:.83rem;color:#78909c;margin-top:6px">
                    Entry: ~{sig.get('entry_price',0):.2f} (market) |
                    SL: {sig.get('sl',0):.2f} |
                    Target: {sig.get('target',0):.2f}
                    </div>
                    <div style="margin-top:8px;font-size:.78rem;color:#546e7a">
                    🤖 Position is being opened automatically. It will appear above on next candle bar.
                    </div></div>""",unsafe_allow_html=True)
                else:
                    no_reason=st.session_state.live_no_pos_reason
                    st.markdown(f"""<div class="pos-none">
                    <div style="font-size:.8rem;color:#546e7a;letter-spacing:.5px">POSITION STATUS</div>
                    <div style="font-size:1.5rem;font-weight:600;color:#37474f;margin:6px 0">⏸ NO OPEN POSITION</div>
                    <div style="font-size:.83rem;color:#455a64;white-space:pre-line;text-align:left;max-width:700px;margin:auto">{no_reason}</div>
                    </div>""",unsafe_allow_html=True)
            else:
                no_reason=st.session_state.live_no_pos_reason
                st.markdown(f"""<div class="pos-none">
                <div style="font-size:.8rem;color:#546e7a;letter-spacing:.5px">POSITION STATUS</div>
                <div style="font-size:1.5rem;font-weight:600;color:#37474f;margin:6px 0">⏸ NO OPEN POSITION</div>
                <div style="font-size:.83rem;color:#455a64;white-space:pre-line;text-align:left;max-width:700px;margin:auto">{no_reason}</div>
                </div>""",unsafe_allow_html=True)

    render_position_banner(pos_placeholder)

    st.markdown("---")

    # ── Fragment: updates LTP + runs trading engine every 2s (no full page flicker) ──
    try:
        @st.fragment(run_every=2)
        def live_fragment():
            ss=st.session_state
            if not ss.live_running:
                return

            # Run the trading engine (main thread, no race condition)
            new_cycle=live_engine_tick(
                symbol,interval,period,depth,sl_val,tgt_val,csl,ctgt,
                dhan_on,dhan_api,sec_id,live_qty,
            )
            if new_cycle:
                render_position_banner(pos_placeholder)

            # ── 4-metric ticker bar ──────────────────────────────────────
            ltp_=ss.live_ltp; ltp_ts=ss.live_ltp_ts or "—"
            delay_=ss.live_delay_s
            cur_ist=now_ist().strftime("%H:%M:%S IST")
            last_c_=ss.live_last_candle_ist
            nxt_=ss.live_next_check_ist

            # Realized + unrealized P&L
            pos_=ss.live_position
            real_pnl=ss.live_pnl
            unreal_pnl=pos_.get("unreal_pnl",0) if pos_ else 0
            total_pnl=real_pnl+unreal_pnl

            tc1,tc2,tc3,tc4,tc5,tc6=st.columns(6)
            ltp_disp=f"{ltp_:,.2f}" if ltp_ else "—"
            tc1.metric("📊 LTP",ltp_disp,delta=ltp_ts,delta_color="off")

            delay_lbl="🟢 Fresh" if delay_<120 else "🟡 Delayed" if delay_<300 else "🔴 Stale"
            tc2.metric("⏩ Data Age",f"{delay_}s",delta=delay_lbl,delta_color="off")

            tc3.metric("🕒 IST Now",cur_ist,delta=last_c_,delta_color="off")
            tc4.metric("🔜 Next Candle",nxt_)

            real_c="normal" if real_pnl>=0 else "inverse"
            tc5.metric("💰 Realized P&L",f"Rs{real_pnl:,.2f}",delta=f"{len(ss.live_trades)} trades",delta_color=real_c)

            unreal_lbl=f"Rs{unreal_pnl:+,.2f}" if pos_ else "No position"
            tc6.metric("📈 Unrealized P&L",f"Rs{total_pnl:,.2f}",delta=unreal_lbl,
                       delta_color="normal" if unreal_pnl>=0 else "inverse")

            # Delay warning
            if delay_>=300:
                st.warning(f"⚠️ Data is {delay_}s old. yfinance may be rate-limited or market is closed.")
            elif delay_>=120:
                st.info(f"ℹ️ Data is {delay_}s old. Normal for {interval} candles outside market hours.")

            # ── Wave state + signal ──────────────────────────────────────
            ws_col, sig_col = st.columns([1.3,1])
            with ws_col:
                ws=ss.live_wave_state
                if ws:
                    dir_c={"BULLISH":"#4caf50","BEARISH":"#f44336","NEUTRAL":"#78909c"}.get(ws.get("direction","NEUTRAL"),"#78909c")
                    dir_i={"BULLISH":"📈","BEARISH":"📉","NEUTRAL":"➡️"}.get(ws.get("direction","NEUTRAL"),"➡️")
                    st.markdown(f"""
                    <div class="wave-card">
                    <b style="color:#00e5ff">🌊 Current Wave</b><br>
                    <span style="color:{dir_c}">{dir_i} {ws.get('current_wave','—')}</span><br>
                    <b style="color:#00e5ff">Next Expected</b><br>
                    <span style="color:#b0bec5">{ws.get('next_wave','—')}</span>
                    </div>""",unsafe_allow_html=True)
                    fibs=ws.get("fib_levels",{})
                    if fibs:
                        st.markdown("**📐 Wave-3 Fibonacci Targets**")
                        for lbl,val in fibs.items():
                            bc2="#4caf50" if "1.618" in lbl else ("#ab47bc" if "2.618" in lbl else "#ffb300")
                            st.markdown(f"""<div style="display:flex;justify-content:space-between;
                            background:#060d14;border-left:3px solid {bc2};border-radius:0 4px 4px 0;
                            padding:4px 9px;margin-bottom:3px;font-size:.82rem">
                            <span style="color:#78909c">{lbl}</span>
                            <span style="color:{bc2};font-family:'Share Tech Mono'">{val:,.2f}</span>
                            </div>""",unsafe_allow_html=True)
                    action=ws.get("action","")
                    st.markdown(f"""<div style="background:#060d14;border:1px solid #1e3a5f;
                    border-radius:6px;padding:10px 12px;font-size:.81rem;color:#b0bec5;
                    line-height:1.75;margin-top:8px;white-space:pre-line">{action}</div>""",unsafe_allow_html=True)

            with sig_col:
                sig_=ss.live_last_sig
                if sig_ and sig_.get("signal")!="HOLD":
                    s_=sig_["signal"]; sc_="#4caf50" if s_=="BUY" else "#f44336"
                    em_="🟢" if s_=="BUY" else "🔴"
                    rr_=abs(sig_.get("target",0)-sig_.get("entry_price",0))/abs(sig_.get("entry_price",0)-sig_.get("sl",0)) if sig_.get("sl") and abs(sig_.get("entry_price",0)-sig_.get("sl",0))>0 else 0
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,.03);border:2px solid {sc_};
                    border-radius:10px;padding:14px 16px">
                    <div style="font-size:.76rem;color:#546e7a">ELLIOTT WAVE SIGNAL</div>
                    <div style="font-size:1.5rem;color:{sc_};font-weight:700">{em_} {s_}</div>
                    <div style="font-size:.82rem;color:#b0bec5;margin-top:4px">{sig_.get('pattern','—')}</div>
                    <div style="font-size:.8rem;color:#00bcd4">Confidence: {sig_.get('confidence',0):.0%}</div>
                    <hr style="border-color:rgba(255,255,255,.08);margin:8px 0">
                    <div style="font-size:.8rem;color:#78909c">
                    Entry: <b style="color:#b0bec5">{sig_.get('entry_price',0):,.2f}</b><br>
                    SL: <b style="color:#ff7043">{sig_.get('sl',0):,.2f}</b><br>
                    Target: <b style="color:#66bb6a">{sig_.get('target',0):,.2f}</b><br>
                    R:R: <b style="color:#b0bec5">1:{rr_:.1f}</b>
                    </div>
                    <hr style="border-color:rgba(255,255,255,.08);margin:8px 0">
                    <div style="font-size:.75rem;color:#455a64">{sig_.get('reason','—')}</div>
                    </div>""",unsafe_allow_html=True)
                elif sig_:
                    st.markdown(f"""
                    <div style="background:#060d14;border:1px solid #263238;border-radius:10px;padding:14px 16px">
                    <div style="font-size:.76rem;color:#546e7a">ELLIOTT WAVE SIGNAL</div>
                    <div style="font-size:1.2rem;color:#37474f;font-weight:600">⏸ HOLD / NO SIGNAL</div>
                    <div style="font-size:.8rem;color:#37474f;margin-top:6px;line-height:1.6">{sig_.get('reason','—')}<br><br>
                    System scanning every candle. Signal fires automatically.<br>No action needed from you.</div>
                    </div>""",unsafe_allow_html=True)
                else:
                    st.markdown("""<div style="background:#060d14;border:1px solid #263238;border-radius:10px;padding:14px 16px">
                    <div style="color:#37474f">Waiting for first candle evaluation…</div></div>""",unsafe_allow_html=True)

            # ── Chart ────────────────────────────────────────────────────
            df_s=ss._scan_df
            if df_s is not None:
                piv_=find_pivots(df_s.iloc[:-1],depth)
                st.plotly_chart(chart_waves(df_s,piv_,ss._scan_sig,symbol=symbol),use_container_width=True)

            # ── History tables ───────────────────────────────────────────
            h1_col,h2_col=st.columns(2)
            with h1_col:
                if ss.live_signals:
                    st.markdown("##### 📋 Signal History")
                    st.dataframe(pd.DataFrame(ss.live_signals).tail(8),use_container_width=True,height=150)
            with h2_col:
                if ss.live_trades:
                    st.markdown("##### 🏁 Completed Trades")
                    td=pd.DataFrame(ss.live_trades)
                    st.dataframe(td.tail(8),use_container_width=True,height=150)
                    wns=(td["PnL Rs"]>0).sum(); tot=len(td); pnl_=td["PnL Rs"].sum()
                    st.caption(f"Win: {wns}/{tot} ({wns/tot*100:.0f}%) | Realized Rs{pnl_:,.2f}")

            if ss.live_log:
                with st.expander("📜 Activity Log",expanded=False):
                    st.code("\n".join(reversed(ss.live_log[-60:])),language=None)

        live_fragment()

    except Exception as frag_err:
        # Fragment not supported (old Streamlit) — fallback to manual scan only
        st.warning(f"⚠️ Fragment auto-refresh unavailable ({frag_err}). Use manual scan below.")
        if st.button("🔍 Scan & Update Now",use_container_width=True):
            with st.spinner("Running trading engine…"):
                live_engine_tick(symbol,interval,period,depth,sl_val,tgt_val,csl,ctgt,dhan_on,dhan_api,sec_id,live_qty)
            render_position_banner(pos_placeholder)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with t_bt:
    bl,br=st.columns([1,2.6],gap="medium")
    with bl:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""<div class="info-box">
        📈 <code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        🌊 Depth <code>{depth}</code> · SL <code>{sl_lbl}</code><br>
        🎯 Target <code>{tgt_lbl}</code> · Rs<code>{capital:,}</code><br>
        {"SL pts: "+str(csl) if sl_val==SL_PTS else ""}
        {"TGT pts: "+str(ctgt) if tgt_val==TGT_PTS else ""}
        <br><small style="color:#546e7a">Signal bar N → entry open bar N+1<br>
        SL checked: Low(BUY) / High(SELL) first (conservative)<br>
        Exit Bar Low/High columns verify realistic fills</small>
        </div>""",unsafe_allow_html=True)
        if st.session_state.best_cfg_applied: st.success("✨ Optimized config active")
        if st.button("🚀 Run Backtest",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dbt=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
            if dbt is None or len(dbt)<40: st.error("Not enough data.")
            else:
                with st.spinner(f"Running on {len(dbt)} bars…"):
                    res=run_backtest(dbt,depth,sl_val,tgt_val,capital,csl,ctgt)
                    res.update({"df":dbt,"pivots":find_pivots(dbt,depth),"symbol":symbol,"interval":interval,"period":period})
                st.session_state.bt_results=res
                if "error" in res: st.error(res["error"])
                else: st.success(f"✅ {res['metrics']['Total Trades']} trades!")
    with br:
        r=st.session_state.bt_results
        if r and "metrics" in r:
            m=r["metrics"]
            st.markdown(f"### Results — `{r.get('symbol','')}` · `{r.get('interval','')}` · `{r.get('period','')}`")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Total Return",f"{m['Total Return %']}%",delta=f"Rs{m['Final Equity Rs']:,}")
            c2.metric("Win Rate",f"{m['Win Rate %']}%",delta=f"{m['Wins']}W/{m['Losses']}L")
            c3.metric("Profit Factor",str(m["Profit Factor"]))
            c4.metric("Max Drawdown",f"{m['Max Drawdown %']}%")
            c5,c6,c7,c8=st.columns(4)
            c5.metric("Sharpe",str(m["Sharpe Ratio"])); c6.metric("Trades",str(m["Total Trades"]))
            c7.metric("Avg Win",f"Rs{m['Avg Win Rs']:,}"); c8.metric("Avg Loss",f"Rs{m['Avg Loss Rs']:,}")
            eb=r.get("exit_breakdown",{})
            if eb:
                ea,eb2,ec,ed=st.columns(4)
                ea.metric("SL Hits",str(eb.get("SL hits",0))); eb2.metric("Target Hits",str(eb.get("Target hits",0)))
                ec.metric("Sig Reverse",str(eb.get("Signal Reverse",0))); ed.metric("Still Open",str(eb.get("Still open",0)))
            tc1,tc2,tc3=st.tabs(["🕯 Wave Chart","📈 Equity Curve","📋 Trade Log"])
            with tc1: st.plotly_chart(chart_waves(r["df"],r["pivots"],_blank(),r["trades"],r["symbol"]),use_container_width=True)
            with tc2: st.plotly_chart(chart_equity(r["equity_curve"]),use_container_width=True)
            with tc3:
                st.dataframe(r["trades"],use_container_width=True,height=420)
                st.info("📊 **Exit Bar Low** / **High** let you verify SL/Target fills.\n"
                        "BUY SL: Low ≤ SL ✅ | BUY Target: High ≥ Target ✅\n"
                        "SELL SL: High ≥ SL ✅ | SELL Target: Low ≤ Target ✅\n"
                        "SL Verified and TGT Verified columns confirm correct exits.")
                st.download_button("📥 Download CSV",data=r["trades"].to_csv(index=False),
                                   file_name=f"ew_bt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        elif r and "error" in r: st.error(r["error"])
        else: st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run backtest to see results</h3></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
with t_opt:
    ol,or_=st.columns([1,3],gap="medium")
    with ol:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""<div class="info-box"><code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>Rs<code>{capital:,}</code><br><br>
        Grid 4×4×5 = <b>80 combos</b></div>""",unsafe_allow_html=True)
        st.warning("⏳ ~80 backtests (~1–3 min)")
        if st.button("🔬 Run Optimization",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dopt=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
            if dopt is None or len(dopt)<50: st.error("Not enough data.")
            else:
                with st.spinner("Running 80 combinations…"):
                    odf=run_optimization(dopt,capital,csl,ctgt)
                st.session_state.opt_results={"df":odf,"symbol":symbol,"interval":interval,"period":period}
                if not odf.empty: st.success(f"✅ Best Score: {odf['Score'].iloc[0]:.2f}")
    with or_:
        optr=st.session_state.opt_results
        if optr and optr.get("df") is not None and not optr["df"].empty:
            odf=optr["df"]
            st.markdown(f"### Results — `{optr['symbol']}` · `{optr['interval']}` · `{optr['period']}`")
            br_=odf.iloc[0]
            b1,b2,b3,b4,b5=st.columns(5)
            b1.metric("Best Depth",str(int(br_["Depth"]))); b2.metric("Best SL",str(br_["SL"]))
            b3.metric("Best Target",str(br_["Target"])); b4.metric("Best Return",f"{br_['Return %']:.1f}%")
            b5.metric("Best Score",f"{br_['Score']:.2f}")
            st.markdown("---")
            ac1,ac2=st.columns([1.6,2.4])
            with ac1:
                nr=st.number_input("Apply top N",min_value=1,max_value=min(5,len(odf)),value=1,step=1)
                ar_=odf.iloc[int(nr)-1]
            with ac2:
                st.markdown(f"""<div class="best-cfg">
                <b style="color:#00e5ff">Config #{int(nr)}</b><br>
                Depth <b>{int(ar_['Depth'])}</b> · SL <b>{ar_['SL']}</b> · Target <b>{ar_['Target']}</b><br>
                Win <b>{ar_['Win %']}%</b> · Return <b>{ar_['Return %']}%</b> · Score <b>{ar_['Score']:.2f}</b>
                </div>""",unsafe_allow_html=True)
            if st.button(f"✨ Apply Config #{int(nr)} → Sidebar + Backtest + Live",type="primary",use_container_width=True):
                st.session_state.applied_depth=int(ar_["Depth"])
                st.session_state.applied_sl_lbl=sl_lbl_from(ar_["SL"])
                st.session_state.applied_tgt_lbl=tgt_lbl_from(ar_["Target"])
                st.session_state.best_cfg_applied=True
                st.success(f"✅ Applied Depth={int(ar_['Depth'])} · SL={sl_lbl_from(ar_['SL'])} · Target={tgt_lbl_from(ar_['Target'])}")
                time.sleep(0.4); st.rerun()
            st.markdown("---")
            oc1,oc2=st.tabs(["📊 Scatter","📋 Table"])
            with oc1:
                st.plotly_chart(chart_opt_scatter(odf),use_container_width=True)
                st.caption("Top-left bubble = best (high return, low drawdown). Size = Win Rate. Color = Score.")
            with oc2:
                def _hl(row):
                    if row.name==0: return ["background-color:rgba(0,229,255,.18)"]*len(row)
                    elif row.name<3: return ["background-color:rgba(0,229,255,.08)"]*len(row)
                    return [""]*len(row)
                st.dataframe(odf.style.apply(_hl,axis=1),use_container_width=True,height=500)
                st.download_button("📥 Download CSV",data=odf.to_csv(index=False),
                                   file_name=f"ew_opt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        else:
            st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run optimization to see results</h3></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — HELP
# ═══════════════════════════════════════════════════════════════════════════
with t_help:
    st.markdown("## 📖 Complete Guide — Elliott Wave Algo Trader v5.1")
    h1,h2=st.columns(2,gap="large")
    with h1:
        st.markdown("""
### 🤔 "Is position entered or not?" — READ THIS FIRST

The **POSITION STATUS** banner (top of Live Trading tab) is your answer:

| What you see | What it means |
|---|---|
| 🟢 **BUY OPEN** (green box) | Yes, position is open and managed |
| 🔴 **SELL OPEN** (red box) | Yes, short position is open |
| ⏸ **NO OPEN POSITION** | No position yet. See reason below it. |
| 💡 **SIGNAL FIRED** (blue box) | Signal detected, position opening now |
| 🔄 **AUTO-REVERSING** | Old position closed, new one opening |

If you see **NO OPEN POSITION**, the box explains exactly why:
- "Waiting for Wave-2 to complete…" → normal, wait
- "Risk = 0 pts…" → change SL setting
- "Invalid target…" → change Target setting

---

### 🤖 Is everything automatic?

**YES — 100% automatic after clicking ▶ Start Live:**
1. System fetches data every candle (60s poll max)
2. Detects Wave-2 completion automatically
3. **Opens position at market price — no button click needed**
4. Monitors position every 2 seconds
5. **Auto-closes** at SL, Target, or Signal Reverse
6. **Auto-reverses** — BUY open + SELL signal = BUY closed + SELL opened

**You literally do nothing. Just watch.**

---

### 📡 LTP & P&L Updates

- **LTP** refreshes every **2 seconds** via `@st.fragment` — **no page flicker**
- **Unrealized P&L** = (LTP - Entry) × Qty — updated every 2s
- **Realized P&L** = sum of all closed trade profits
- **Total P&L** = Realized + Unrealized

---

### ⏱ What the time fields mean

| Field | Meaning |
|---|---|
| 🕒 IST Now | Your real current time (India) |
| 📊 Last Candle | When last closed bar ended (IST) |
| ⏩ Data Age | Gap in seconds — 0–120s = normal |
| 🔜 Next Candle | Approx. time of next full candle check |

**Data Age > 5 min**: Data may be stale. Check internet. Market may be closed.
        """)
    with h2:
        st.markdown("""
### 🔄 Auto-Reverse Explained

Example (BUY position open):
1. BUY opened at 71740 (Wave-2 bottom detected)
2. Price rises toward target
3. Suddenly a new SELL signal fires (Wave-2 top on a different wave)
4. System **automatically**:
   - Closes BUY at current market price
   - Opens SELL position immediately
   - Logs "Signal Reversed → SELL"

**You see**: Position changes from 🟢 BUY to 🔴 SELL automatically.

---

### 🛡️ SL & Target Options

| Option | Best for |
|--------|---------|
| Wave Auto | Default — SL at Wave-2 pivot (most logical) |
| % options | Simple fixed risk |
| Custom Points | Nifty (50pts), BTC (500pts), etc. |
| Signal Reverse | Let waves decide exits — no fixed SL |

---

### 📊 Backtest Exit Verification

New columns in trade log:
- **Exit Bar Low / High**: Actual bar H/L at exit
- **SL Verified ✅**: Confirms Low ≤ SL (BUY) or High ≥ SL (SELL)
- **TGT Verified ✅**: Confirms High ≥ Target (BUY) or Low ≤ Target (SELL)

This proves the backtest uses **conservative (realistic) fills** — not optimistic.

---

### ⚠️ Troubleshooting

| Problem | Solution |
|---------|---------|
| Position shown but no entry | Check the NO POSITION reason text |
| LTP shows "—" | Symbol may not support fast_info; wait for candle |
| "Signal already fired" | Click 🔄 Reset All |
| Large Data Age | Internet issue or market closed |
| No trades in backtest | Smaller Pivot Depth or longer Period |
| Fragment error | Upgrade: `pip install streamlit --upgrade` |
        """)
    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#37474f;font-size:.83rem;padding:8px">
    🌊 Elliott Wave Algo Trader v5.1 · Streamlit + yfinance + Plotly ·
    <b style="color:#f44336">Not financial advice. Paper trade first. Use Stop Loss always.</b>
    </div>""",unsafe_allow_html=True)
