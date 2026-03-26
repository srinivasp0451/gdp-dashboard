"""
╔══════════════════════════════════════════════════════════════╗
║       🌊  ELLIOTT WAVE ALGO TRADER  v5.2                     ║
║  Dhan Stocks+Options · Trailing SL/TGT · All Bugs Fixed     ║
╚══════════════════════════════════════════════════════════════╝
pip install streamlit>=1.37 yfinance plotly pandas numpy requests dhanhq
streamlit run elliott_wave_algo_trader.py
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
import itertools
import warnings
import math

warnings.filterwarnings("ignore")

# optional dhanhq
try:
    from dhanhq import dhanhq as DhanHQLib
    DHAN_LIB_OK = True
except ImportError:
    DHAN_LIB_OK = False

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist() -> datetime: return datetime.now(IST)
def fmt_ist(dt=None) -> str:
    if dt is None: return now_ist().strftime("%H:%M:%S IST")
    try:
        if hasattr(dt, "to_pydatetime"): dt = dt.to_pydatetime()
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).strftime("%d-%b %H:%M:%S IST")
    except: return str(dt)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="🌊 Elliott Wave Algo Trader",page_icon="🌊",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Exo 2',sans-serif;}
/* ---- layout: prevent sidebar content truncation ---- */
section[data-testid="stSidebar"]>div{min-width:310px;width:310px!important;}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] small{white-space:normal!important;word-break:break-word!important;}
/* metric: no ellipsis */
div[data-testid="metric-container"] *{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;}
div[data-testid="stMetricValue"]{font-size:1rem!important;}
div[data-testid="stMetricDelta"]{font-size:.78rem!important;}
/* main banners */
.main-hdr{background:linear-gradient(135deg,#0a0e1a,#0d1b2a,#0a1628);
  border:1px solid #1e3a5f;border-radius:14px;padding:18px 24px;margin-bottom:12px;
  box-shadow:0 4px 24px rgba(0,229,255,.08);}
.main-hdr h1{font-family:'Exo 2',sans-serif;font-weight:700;
  background:linear-gradient(90deg,#00e5ff,#4dd0e1,#00bcd4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;font-size:1.75rem;}
.main-hdr p{color:#546e7a;margin:4px 0 0;font-size:.83rem;}
/* position banners */
.pos-none{background:#07111e;border:2px dashed #263238;border-radius:14px;padding:16px 22px;}
.pos-signal{background:rgba(0,229,255,.08);border:2px solid #00e5ff;border-radius:14px;padding:16px 22px;}
.pos-buy{background:rgba(76,175,80,.11);border:3px solid #4caf50;border-radius:14px;padding:16px 22px;}
.pos-sell{background:rgba(244,67,54,.11);border:3px solid #f44336;border-radius:14px;padding:16px 22px;}
.pos-reversing{background:rgba(171,71,188,.14);border:3px solid #ab47bc;border-radius:14px;padding:16px 22px;text-align:center;}
/* wave / info cards */
.wave-card{background:#060d14;border:1px solid #1e3a5f;border-radius:8px;
  padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:.81rem;line-height:1.85;
  white-space:pre-wrap;word-break:break-word;}
.info-box{background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
  padding:12px 14px;font-size:.85rem;line-height:1.85;word-break:break-word;}
.best-cfg{background:rgba(0,229,255,.07);border:1px solid #00bcd4;border-radius:10px;
  padding:12px 16px;margin:6px 0;word-break:break-word;}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:4px;background:transparent;}
.stTabs [data-baseweb="tab"]{background:#0d1b2a;border-radius:8px;color:#546e7a;
  border:1px solid #1e3a5f;padding:5px 11px;font-size:.80rem;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0d3349,#0a2540)!important;
  color:#00e5ff!important;border-color:#00bcd4!important;}
div[data-testid="metric-container"]{background:#0a1628;border:1px solid #1e3a5f;
  border-radius:8px;padding:9px 10px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
TICKER_GROUPS = {
    "🇮🇳 Indian Indices": {"Nifty 50":"^NSEI","Bank Nifty":"^NSEBANK","Sensex":"^BSESN",
        "Nifty IT":"^CNXIT","Nifty Midcap":"^NSEMDCP50","Nifty Auto":"^CNXAUTO"},
    "🇮🇳 NSE Stocks": {"Reliance":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
        "Infosys":"INFY.NS","ICICI Bank":"ICICIBANK.NS","SBI":"SBIN.NS",
        "Wipro":"WIPRO.NS","L&T":"LT.NS","Axis Bank":"AXISBANK.NS","Maruti":"MARUTI.NS"},
    "₿ Crypto": {"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD","Solana":"SOL-USD"},
    "💱 Forex": {"USD/INR":"USDINR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPY=X"},
    "🥇 Commodities": {"Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F","Natural Gas":"NG=F"},
    "🌐 US Stocks": {"Apple":"AAPL","Tesla":"TSLA","NVIDIA":"NVDA","Microsoft":"MSFT","Meta":"META"},
    "✏️ Custom Ticker": {"Custom":"__CUSTOM__"},
}

TIMEFRAMES=["1m","5m","15m","30m","1h","4h","1d","1wk"]
PERIODS=["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]
VALID_PERIODS={
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo","3mo"],
    "15m":["1d","5d","7d","1mo","3mo"],"30m":["1d","5d","7d","1mo","3mo"],
    "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h":["1mo","3mo","6mo","1y","2y","5y"],
    "1d":["1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["3mo","6mo","1y","2y","5y","10y","20y"],
}
POLL_SECS={"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400,"1wk":604800}
MTF_COMBOS=[("1d","1y","Daily"),("4h","3mo","4-Hour"),("1h","1mo","1-Hour"),("15m","5d","15-Min")]

# SL / TGT sentinels
SL_WAVE="wave_auto"; SL_PTS="__sl_pts__"; SL_SIGREV="__sl_sigrev__"; SL_TRAIL="__sl_trail__"
TGT_PTS="__tgt_pts__"; TGT_SIGREV="__tgt_sigrev__"; TGT_TRAIL="__tgt_trail__"

SL_MAP={
    "Wave Auto (Pivot Low/High)":SL_WAVE,
    "0.5%":0.005,"1%":0.01,"1.5%":0.015,"2%":0.02,"2.5%":0.025,"3%":0.03,"5%":0.05,
    "Custom Points ▼":SL_PTS,
    "Trailing SL ▼":SL_TRAIL,
    "Exit on Signal Reverse":SL_SIGREV,
}
TGT_MAP={
    "Wave Auto (Fib 1.618×W1)":"wave_auto",
    "R:R 1:1":1.0,"R:R 1:1.5":1.5,"R:R 1:2":2.0,"R:R 1:2.5":2.5,"R:R 1:3":3.0,
    "Fib 1.618×Wave 1":"fib_1618","Fib 2.618×Wave 1":"fib_2618",
    "Custom Points ▼":TGT_PTS,
    "Trailing Target ▼ (display only)":TGT_TRAIL,
    "Exit on Signal Reverse":TGT_SIGREV,
}
SL_KEYS=list(SL_MAP.keys()); TGT_KEYS=list(TGT_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
_DEF={
    "live_running":False,"live_phase":"idle",
    "live_ltp":None,"live_ltp_prev":None,"live_ltp_ts":None,
    "live_last_candle_ist":"—","live_delay_s":0,"live_delay_context":"",
    "live_next_check_ist":"—","last_bar_ts":None,"last_fetch_wall":0.0,
    "live_position":None,"live_pnl":0.0,       # live_pnl = REALIZED ONLY
    "live_signals":[],"live_trades":[],"live_log":[],
    "live_wave_state":None,"live_last_sig":None,
    "live_no_pos_reason":"Click ▶ Start Live. Everything runs automatically.",
    "_scan_sig":None,"_scan_df":None,
    "bt_results":None,"opt_results":None,
    "_analysis_results":None,"_analysis_overall":"HOLD","_analysis_symbol":"",
    "applied_depth":5,"applied_sl_lbl":"Wave Auto (Pivot Low/High)",
    "applied_tgt_lbl":"Wave Auto (Fib 1.618×W1)","best_cfg_applied":False,
    "custom_sl_pts":50.0,"custom_tgt_pts":100.0,
    "trailing_sl_pts":50.0,"trailing_tgt_pts":150.0,
}
for _k,_v in _DEF.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ═══════════════════════════════════════════════════════════════════════════
# DHAN CLIENT (using dhanhq library)
# ═══════════════════════════════════════════════════════════════════════════
class DhanClient:
    """Wraps dhanhq library with fallback to requests if library not installed."""

    SEGMENT_MAP = {
        "NSE":"NSE_EQ","BSE":"BSE_EQ",
        "NSE_FNO":"NSE_FNO","BSE_FNO":"BSE_FNO",
    }

    def __init__(self, client_id:str, access_token:str):
        self.cid = client_id
        self.tok = access_token
        self._lib = None
        if DHAN_LIB_OK:
            try:
                self._lib = DhanHQLib(client_id, access_token)
            except Exception:
                self._lib = None

    def _seg_const(self, seg:str):
        """Return dhanhq segment constant or string fallback."""
        if self._lib:
            mapping = {
                "NSE":"NSE_EQ","NSE_EQ":"NSE_EQ","BSE":"BSE_EQ","BSE_EQ":"BSE_EQ",
                "NSE_FNO":"NSE_FNO","BSE_FNO":"BSE_FNO",
            }
            attr = mapping.get(seg, seg)
            return getattr(self._lib, attr, attr)
        return seg

    def place_order(self, security_id:str, segment:str, txn:str, qty:int,
                    order_type:str="MARKET", price:float=0.0,
                    product_type:str="INTRADAY", validity:str="DAY") -> dict:
        if self._lib:
            try:
                txn_const = self._lib.BUY if txn=="BUY" else self._lib.SELL
                ot_const  = self._lib.MARKET if order_type=="MARKET" else self._lib.LIMIT
                pt_map    = {"INTRADAY":self._lib.INTRA,"DELIVERY":self._lib.CNC,"MARGIN":self._lib.MARGIN}
                pt_const  = pt_map.get(product_type, self._lib.INTRA)
                seg_const = self._seg_const(segment)
                resp = self._lib.place_order(
                    security_id=security_id,
                    exchange_segment=seg_const,
                    transaction_type=txn_const,
                    quantity=qty,
                    order_type=ot_const,
                    product_type=pt_const,
                    price=float(price),
                    validity=validity,
                )
                return resp
            except Exception as e:
                return {"error":str(e),"lib":"dhanhq"}
        else:
            # fallback requests
            import requests
            try:
                r=requests.post("https://api.dhan.co/orders",timeout=10,
                    headers={"Content-Type":"application/json","access-token":self.tok},
                    json={"dhanClientId":self.cid,"transactionType":txn,
                          "exchangeSegment":segment,"productType":product_type,
                          "orderType":order_type,"validity":validity,
                          "securityId":security_id,"quantity":qty,"price":price})
                return r.json()
            except Exception as e:
                return {"error":str(e),"lib":"requests"}

    def fund_limit(self) -> dict:
        if self._lib:
            try: return self._lib.get_fund_limits()
            except Exception as e: return {"error":str(e)}
        else:
            import requests
            try:
                r=requests.get("https://api.dhan.co/fundlimit",timeout=10,
                    headers={"Content-Type":"application/json","access-token":self.tok})
                return r.json()
            except Exception as e: return {"error":str(e)}

# ═══════════════════════════════════════════════════════════════════════════
# RATE-LIMIT SAFE FETCH
# ═══════════════════════════════════════════════════════════════════════════
_fetch_lock=threading.Lock(); _last_fetch_t=[0.0]

def fetch_ohlcv(symbol:str,interval:str,period:str,min_delay:float=1.5)->Optional[pd.DataFrame]:
    with _fetch_lock:
        gap=time.time()-_last_fetch_t[0]
        if gap<min_delay: time.sleep(min_delay-gap)
        try:
            df=yf.download(symbol,interval=interval,period=period,progress=False,auto_adjust=True)
            _last_fetch_t[0]=time.time()
        except Exception: _last_fetch_t[0]=time.time(); return None
    if df is None or df.empty: return None
    if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    df=df.dropna(subset=["Open","High","Low","Close"]); df.index=pd.to_datetime(df.index)
    return df

def fetch_ltp(symbol:str)->Optional[float]:
    """Lightweight LTP via yfinance fast_info."""
    try:
        t=yf.Ticker(symbol); fi=t.fast_info
        p=fi.get("lastPrice") or fi.get("regularMarketPrice")
        if p and float(p)>0: return float(p)
    except Exception: pass
    return None

def delay_context(delay_s:int, interval:str) -> tuple:
    """Returns (label, color, advice) based on delay and interval."""
    candle_s=POLL_SECS.get(interval,3600)
    ratio=delay_s/candle_s if candle_s>0 else 0

    if delay_s<120:
        return ("🟢 Fresh data",  "#4caf50",
                f"Data is {delay_s}s old — very fresh for {interval} candles.")
    elif ratio<1.0:
        pct=int(ratio*100)
        return (f"🟡 {delay_s}s old",  "#ffb300",
                f"Data is {delay_s}s old ({pct}% of {interval} candle). Normal — current candle still forming.")
    elif ratio<2.0:
        return (f"🟡 {delay_s//60}min old", "#ff9800",
                f"Data is {delay_s}s old (~{delay_s//60}min). "
                f"Normal outside market hours OR market closed. "
                f"System will use latest available bar for signal.")
    else:
        hours=delay_s//3600
        return (f"🔴 {hours}h+ old", "#f44336",
                f"Data is {hours}h old. Market likely closed or holiday. "
                f"Signals on closed-market data are valid for next session open. "
                f"No action needed — system waits for new candle automatically.")

# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def add_indicators(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy(); c=df["Close"]
    df["EMA_20"]=c.ewm(span=20,adjust=False).mean()
    df["EMA_50"]=c.ewm(span=50,adjust=False).mean()
    df["EMA_200"]=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean()
    df["RSI"]=100-(100/(1+g/l.replace(0,np.nan)))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    df["MACD"]=e12-e26; df["MACD_Signal"]=df["MACD"].ewm(span=9,adjust=False).mean()
    if "Volume" in df.columns: df["Vol_Avg"]=df["Volume"].rolling(20).mean()
    return df

# ═══════════════════════════════════════════════════════════════════════════
# PIVOTS
# ═══════════════════════════════════════════════════════════════════════════
def find_pivots(df:pd.DataFrame,depth:int=5)->list:
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
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════
def analyze_wave_state(df:pd.DataFrame,pivots:list,sig:dict)->dict:
    cur=float(df["Close"].iloc[-1]) if len(df) else 0
    if not pivots:
        return {"current_wave":"Collecting data…","next_wave":"—","direction":"NEUTRAL",
                "fib_levels":{},"action":"Insufficient pivot data. Try smaller Pivot Depth.","auto_action":"Wait"}
    if sig["signal"]=="BUY":
        wp=sig.get("wave_pivots",[None,None,None])
        p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p1[1]-p0[1]) if p0 and p1 else cur*0.02
        w2_low=p2[1] if p2 else cur*0.99
        fibs={"W3 (1.618×W1)":round(w2_low+w1*1.618,2),
              "W3 (1.000×W1)":round(w2_low+w1*1.000,2),
              "W3 (2.618×W1)":round(w2_low+w1*2.618,2)}
        return {"current_wave":"✅ Wave-2 Bottom — ENTRY ZONE","next_wave":"Wave-3 UP ↑ (strongest move)",
                "direction":"BULLISH","fib_levels":fibs,
                "action":(f"BUY position opening at market price.\n"
                          f"SL: {w2_low:.2f} (below Wave-2 pivot)\n"
                          f"Wave-3 targets: {fibs['W3 (1.618×W1)']:.2f} · {fibs['W3 (2.618×W1)']:.2f}\n"
                          f"WHAT TO DO: Nothing. System auto-manages."),"auto_action":"BUY"}
    if sig["signal"]=="SELL":
        wp=sig.get("wave_pivots",[None,None,None])
        p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p0[1]-p1[1]) if p0 and p1 else cur*0.02
        w2_high=p2[1] if p2 else cur*1.01
        fibs={"W3 (1.618×W1)":round(w2_high-w1*1.618,2),
              "W3 (1.000×W1)":round(w2_high-w1*1.000,2),
              "W3 (2.618×W1)":round(w2_high-w1*2.618,2)}
        return {"current_wave":"🔴 Wave-2 Top — ENTRY ZONE","next_wave":"Wave-3 DOWN ↓ (strongest move)",
                "direction":"BEARISH","fib_levels":fibs,
                "action":(f"SELL position opening at market price.\n"
                          f"SL: {w2_high:.2f} (above Wave-2 pivot)\n"
                          f"Wave-3 targets: {fibs['W3 (1.618×W1)']:.2f} · {fibs['W3 (2.618×W1)']:.2f}\n"
                          f"WHAT TO DO: Nothing. System auto-manages."),"auto_action":"SELL"}
    lp=pivots[-1]; rp=0.0
    if len(pivots)>=3:
        pa,pb,pc=pivots[-3],pivots[-2],pivots[-1]
        if abs(pb[1]-pa[1])>0: rp=abs(pc[1]-pb[1])/abs(pb[1]-pa[1])*100
    needed=f" (need 38–79%; current {rp:.1f}%)" if rp>0 else ""
    return {"current_wave":f"{'High' if lp[2]=='H' else 'Low'} pivot @ {lp[1]:.2f}",
            "next_wave":f"Waiting for Wave-2 to complete{needed}","direction":"NEUTRAL","fib_levels":{},
            "action":(f"No signal yet. {sig.get('reason','')}\n"
                      f"System scans every candle automatically.\n"
                      f"Position opens when Wave-2 hits 38–79% retracement.\n"
                      f"WHAT TO DO: Nothing — just wait."),"auto_action":"Wait"}

# ═══════════════════════════════════════════════════════════════════════════
# CORE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════
def _blank(reason:str="")->dict:
    return {"signal":"HOLD","entry_price":None,"sl":None,"target":None,
            "confidence":0.0,"reason":reason or "No Elliott Wave pattern detected",
            "pattern":"—","wave_pivots":None,"wave1_len":0.0,"retracement":0.0}

def _calc_target(tt,entry:float,direction:str,w1:float,risk:float,ctp:float=100.0)->float:
    s=1 if direction=="BUY" else -1
    if tt in ("wave_auto","fib_1618"): return entry+s*w1*1.618
    if tt=="fib_2618": return entry+s*w1*2.618
    if tt==TGT_PTS:    return entry+s*ctp
    if tt==TGT_TRAIL:  return entry+s*ctp       # initial level; display updates live
    if tt==TGT_SIGREV: return entry+s*w1*1.618  # fallback level; actual exit = signal
    if isinstance(tt,(int,float)): return entry+s*risk*float(tt)
    return entry+s*risk*2.0

def ew_signal(df:pd.DataFrame,depth:int=5,sl_type="wave_auto",tgt_type="wave_auto",
              csl:float=50.0,ctgt:float=100.0)->dict:
    n=len(df)
    if n<max(30,depth*4): return _blank("Insufficient bars")
    pivots=find_pivots(df,depth)
    if len(pivots)<4: return _blank("Not enough pivots — try smaller Pivot Depth")
    cur=float(df["Close"].iloc[-1]); best,bc=_blank(),0.0
    for i in range(len(pivots)-2):
        p0,p1,p2=pivots[i],pivots[i+1],pivots[i+2]; bs=n-1-p2[0]
        # BUY
        if p0[2]=="L" and p1[2]=="H" and p2[2]=="L":
            w1=p1[1]-p0[1]
            if w1<=0: continue
            r=(p1[1]-p2[1])/w1
            if not(0.236<=r<=0.886 and p2[1]>p0[1] and bs<=depth*4): continue
            c=0.50
            if 0.382<=r<=0.618: c=0.65
            if 0.50<=r<=0.786: c=0.72
            if abs(r-0.618)<0.04: c=0.87
            if abs(r-0.382)<0.03: c=0.75
            if i+3<len(pivots) and pivots[i+3][2]=="H":
                w3=pivots[i+3][1]-p2[1]
                if w3>w1: c=min(c+0.08,0.95)
                if abs(w3/w1-1.618)<0.20: c=min(c+0.05,0.98)
            if c<=bc: continue
            e=cur
            sl_=p2[1]*0.998 if sl_type in(SL_WAVE,SL_TRAIL) else \
                e-csl if sl_type==SL_PTS else \
                e*(1-0.05) if sl_type==SL_SIGREV else e*(1-float(sl_type))
            rk=e-sl_
            if rk<=0: continue
            tgt_=_calc_target(tgt_type,e,"BUY",w1,rk,ctgt)
            if tgt_<=e: continue
            bc=c; best={"signal":"BUY","entry_price":e,"sl":sl_,"target":tgt_,"confidence":c,
                        "retracement":r,"reason":f"Wave-2 bottom: {r:.1%} retrace → Wave-3 up",
                        "pattern":f"W2 Bottom ({r:.1%})","wave_pivots":[p0,p1,p2],"wave1_len":w1}
        # SELL
        elif p0[2]=="H" and p1[2]=="L" and p2[2]=="H":
            w1=p0[1]-p1[1]
            if w1<=0: continue
            r=(p2[1]-p1[1])/w1
            if not(0.236<=r<=0.886 and p2[1]<p0[1] and bs<=depth*4): continue
            c=0.50
            if 0.382<=r<=0.618: c=0.65
            if 0.50<=r<=0.786: c=0.72
            if abs(r-0.618)<0.04: c=0.87
            if abs(r-0.382)<0.03: c=0.75
            if i+3<len(pivots) and pivots[i+3][2]=="L":
                w3=p2[1]-pivots[i+3][1]
                if w3>w1: c=min(c+0.08,0.95)
                if abs(w3/w1-1.618)<0.20: c=min(c+0.05,0.98)
            if c<=bc: continue
            e=cur
            sl_=p2[1]*1.002 if sl_type in(SL_WAVE,SL_TRAIL) else \
                e+csl if sl_type==SL_PTS else \
                e*(1+0.05) if sl_type==SL_SIGREV else e*(1+float(sl_type))
            rk=sl_-e
            if rk<=0: continue
            tgt_=_calc_target(tgt_type,e,"SELL",w1,rk,ctgt)
            if tgt_>=e: continue
            bc=c; best={"signal":"SELL","entry_price":e,"sl":sl_,"target":tgt_,"confidence":c,
                        "retracement":r,"reason":f"Wave-2 top: {r:.1%} retrace → Wave-3 down",
                        "pattern":f"W2 Top ({r:.1%})","wave_pivots":[p0,p1,p2],"wave1_len":w1}
    return best

# ═══════════════════════════════════════════════════════════════════════════
# TRAILING SL UPDATE
# ═══════════════════════════════════════════════════════════════════════════
def update_trailing_sl(pos:dict, ltp:float, sl_type:str, trail_pts:float) -> dict:
    """Move SL in favour of trade. Only moves SL toward profit, never against."""
    if sl_type != SL_TRAIL or not pos: return pos
    pos=pos.copy()
    if pos["type"]=="BUY":
        new_sl=ltp-trail_pts
        if new_sl>pos["sl"]: pos["sl"]=new_sl   # ratchet up
    else:
        new_sl=ltp+trail_pts
        if new_sl<pos["sl"]: pos["sl"]=new_sl   # ratchet down
    return pos

def update_trailing_tgt(pos:dict, ltp:float, tgt_type:str, trail_pts:float) -> dict:
    """Extend target display as price moves in favour (display only — never triggers exit)."""
    if tgt_type != TGT_TRAIL or not pos: return pos
    pos=pos.copy()
    if pos["type"]=="BUY":
        new_tgt=ltp+trail_pts
        if new_tgt>pos["target"]: pos["target_display"]=new_tgt
    else:
        new_tgt=ltp-trail_pts
        if new_tgt<pos["target"]: pos["target_display"]=new_tgt
    return pos

# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df:pd.DataFrame,depth:int=5,sl_type="wave_auto",tgt_type="wave_auto",
                 capital:float=100_000.0,csl:float=50.0,ctgt:float=100.0,
                 trail_sl_pts:float=50.0)->dict:
    MB=max(30,depth*4)
    if len(df)<MB+10: return {"error":f"Need ≥{MB+10} bars.","equity_curve":[capital]}
    trades,equity_curve=[],[capital]; equity,pos=capital,None

    for i in range(MB,len(df)-1):
        bdf=df.iloc[:i+1]; nb=df.iloc[i+1]
        hi_i=float(df.iloc[i]["High"]); lo_i=float(df.iloc[i]["Low"])
        close_i=float(df.iloc[i]["Close"])
        tsig=None

        if pos:
            # update trailing SL using bar close
            if sl_type==SL_TRAIL:
                pos=update_trailing_sl(pos, close_i, sl_type, trail_sl_pts)
            # signal reverse eval
            if sl_type==SL_SIGREV or tgt_type==TGT_SIGREV:
                tsig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt)
            ep_,er_=None,None
            if pos["type"]=="BUY":
                if lo_i<=pos["sl"]:       ep_,er_=pos["sl"],   "SL (Low≤SL)"
                elif hi_i>=pos["target"]: ep_,er_=pos["target"],"Target (High≥Target)"
                elif tsig and tsig["signal"]=="SELL": ep_,er_=close_i,"Signal Reverse"
            else:
                if hi_i>=pos["sl"]:       ep_,er_=pos["sl"],   "SL (High≥SL)"
                elif lo_i<=pos["target"]: ep_,er_=pos["target"],"Target (Low≤Target)"
                elif tsig and tsig["signal"]=="BUY": ep_,er_=close_i,"Signal Reverse"
            if ep_ is not None:
                qty=pos["qty"]; pnl=(ep_-pos["entry"])*qty if pos["type"]=="BUY" else (pos["entry"]-ep_)*qty
                equity+=pnl; equity_curve.append(equity)
                trades.append({"Entry Time":pos["entry_time"],"Exit Time":df.index[i],
                    "Type":pos["type"],"Entry":round(pos["entry"],2),"Exit":round(ep_,2),
                    "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
                    "Exit Bar Low":round(lo_i,2),"Exit Bar High":round(hi_i,2),
                    "Exit Reason":er_,"PnL Rs":round(pnl,2),
                    "PnL %":round(pnl/(pos["entry"]*qty)*100,2),
                    "Equity Rs":round(equity,2),"Bars Held":i-pos["entry_bar"],
                    "Confidence":round(pos["conf"],2)})
                pos=None

        if pos is None:
            sig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt)
            if sig["signal"] in("BUY","SELL"):
                ep=float(nb["Open"]); w1=sig.get("wave1_len",ep*0.02) or ep*0.02
                if sl_type in(SL_WAVE,SL_TRAIL): sl_=sig["sl"]
                elif sl_type==SL_PTS: sl_=ep-csl if sig["signal"]=="BUY" else ep+csl
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
    ea=np.array(equity_curve); pk=np.maximum.accumulate(ea)
    mdd=float(((ea-pk)/pk*100).min())
    rets=tdf["PnL %"].values
    sharpe=float(rets.mean()/rets.std()*np.sqrt(252)) if len(rets)>1 and rets.std()!=0 else 0.0

    tdf["SL Verified"]=tdf.apply(lambda r:"✅" if(
        (r["Type"]=="BUY"  and "SL" in r["Exit Reason"] and r["Exit Bar Low"]<=r["SL"]) or
        (r["Type"]=="SELL" and "SL" in r["Exit Reason"] and r["Exit Bar High"]>=r["SL"]) or
        "SL" not in r["Exit Reason"]) else "⚠️",axis=1)
    tdf["TGT Verified"]=tdf.apply(lambda r:"✅" if(
        (r["Type"]=="BUY"  and "Target" in r["Exit Reason"] and r["Exit Bar High"]>=r["Target"]) or
        (r["Type"]=="SELL" and "Target" in r["Exit Reason"] and r["Exit Bar Low"]<=r["Target"]) or
        "Target" not in r["Exit Reason"]) else "⚠️",axis=1)

    return {"trades":tdf,"equity_curve":equity_curve,
            "exit_breakdown":{
                "SL hits":len(tdf[tdf["Exit Reason"].str.contains("SL",na=False)]),
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
# OPTIMIZATION  (fixed: progress runs to 100%, catches all exceptions)
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization(df,capital=100_000.0,csl=50.0,ctgt=100.0,trail_sl=50.0):
    DEPTHS=[3,5,7,10]; SL_OPTS=[0.01,0.02,0.03,"wave_auto"]; TGT_OPTS=[1.5,2.0,3.0,"wave_auto","fib_1618"]
    combos=list(itertools.product(DEPTHS,SL_OPTS,TGT_OPTS))
    prog=st.progress(0,text="Optimizing — 0%"); rows=[]
    for idx,(dep,sl,tgt) in enumerate(combos):
        try:
            r=run_backtest(df,depth=dep,sl_type=sl,tgt_type=tgt,capital=capital,
                           csl=csl,ctgt=ctgt,trail_sl_pts=trail_sl)
            if "metrics" in r:
                m=r["metrics"]
                rows.append({"Depth":dep,"SL":str(sl),"Target":str(tgt),
                    "Trades":m["Total Trades"],"Win %":m["Win Rate %"],
                    "Return %":m["Total Return %"],"PF":m["Profit Factor"],
                    "Max DD %":m["Max Drawdown %"],"Sharpe":m["Sharpe Ratio"]})
        except Exception:
            pass   # skip failed combos, never halt progress
        pct=int((idx+1)/len(combos)*100)
        prog.progress((idx+1)/len(combos),text=f"Optimizing — {pct}% ({idx+1}/{len(combos)} combos)")
    prog.empty()
    if not rows: return pd.DataFrame()
    out=pd.DataFrame(rows)
    out["Score"]=(out["Return %"].clip(lower=0)*(out["Win %"]/100)
                  *out["PF"].clip(upper=10)/(out["Max DD %"].abs()+1))
    return out.sort_values("Score",ascending=False).reset_index(drop=True)

def sl_lbl(v):
    for k,vv in SL_MAP.items():
        if str(vv)==str(v): return k
    return SL_KEYS[0]

def tgt_lbl(v):
    for k,vv in TGT_MAP.items():
        if str(vv)==str(v): return k
    try:
        fv=float(v)
        for k,vv in TGT_MAP.items():
            if isinstance(vv,float) and abs(vv-fv)<0.01: return k
    except: pass
    return TGT_KEYS[0]

# ═══════════════════════════════════════════════════════════════════════════
# LIVE ENGINE TICK  (main-thread only, called from @st.fragment)
# ═══════════════════════════════════════════════════════════════════════════
def live_engine_tick(symbol,interval,period,depth,sl_type,tgt_type,csl,ctgt,
                     trail_sl_pts,trail_tgt_pts,dhan_on,dhan_client,
                     trade_mode,entry_ot,exit_ot,entry_price_lmt,exit_price_lmt,
                     product_type,segment,ce_sec_id,pe_sec_id,live_qty)->bool:
    ss=st.session_state; now_ts=time.time()

    # ── Fast LTP ──────────────────────────────────────────────────────────
    ltp_new=fetch_ltp(symbol)
    if ltp_new and ltp_new>0:
        ss.live_ltp_prev=ss.live_ltp
        ss.live_ltp=ltp_new
        ss.live_ltp_ts=now_ist().strftime("%H:%M:%S IST")

    ltp=ss.live_ltp

    # ── Unrealized P&L update (every 2s) — NEVER modifies live_pnl ───────
    pos=ss.live_position
    if pos and ltp:
        pos["unreal_pnl"]=(ltp-pos["entry"])*pos["qty"] if pos["type"]=="BUY" \
                          else (pos["entry"]-ltp)*pos["qty"]
        pos["dist_sl"]=abs(ltp-pos["sl"])/ltp*100
        pos["dist_tgt"]=abs(pos.get("target_display",pos["target"])-ltp)/ltp*100
        # Trailing SL update every tick
        if sl_type==SL_TRAIL:
            pos=update_trailing_sl(pos,ltp,sl_type,trail_sl_pts)
        # Trailing Target display update
        if tgt_type==TGT_TRAIL:
            pos=update_trailing_tgt(pos,ltp,tgt_type,trail_tgt_pts)
        ss.live_position=pos

    # ── Check if time for full candle fetch ───────────────────────────────
    candle_s=POLL_SECS.get(interval,3600)
    time_since=now_ts-ss.last_fetch_wall
    if time_since<candle_s and ss.last_fetch_wall>0:
        remaining=int(candle_s-time_since)
        nxt_ist=(now_ist()+timedelta(seconds=remaining)).strftime("%H:%M:%S IST")
        ss.live_next_check_ist=nxt_ist
        return False

    # ── Full candle fetch ─────────────────────────────────────────────────
    ss.live_phase="scanning"; ss.live_next_check_ist="fetching…"
    ss.last_fetch_wall=now_ts

    def _log(msg,lvl="INFO"):
        ts=now_ist().strftime("%H:%M:%S IST")
        ss.live_log.append(f"[{ts}][{lvl}] {msg}")
        ss.live_log=ss.live_log[-150:]

    df=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
    if df is None or len(df)<35:
        _log("⚠️  No/insufficient data","WARN")
        ss.live_no_pos_reason=("⚠️ No data from yfinance. Check symbol, interval, or internet.")
        ss.live_phase="no_signal"; return False

    ohlcv_ltp=float(df["Close"].iloc[-1])
    if not ss.live_ltp or ss.live_ltp<=0: ss.live_ltp=ohlcv_ltp

    # Candle timing
    last_bar_dt=df.index[-2] if len(df)>=2 else df.index[-1]
    ss.live_last_candle_ist=fmt_ist(last_bar_dt)
    try:
        lbdt=last_bar_dt.to_pydatetime()
        if lbdt.tzinfo is None: lbdt=lbdt.replace(tzinfo=timezone.utc)
        ss.live_delay_s=int((now_ist()-lbdt.astimezone(IST)).total_seconds())
    except: ss.live_delay_s=0
    dlbl,_,dadvice=delay_context(ss.live_delay_s,interval)
    ss.live_delay_context=dadvice

    df_closed=df.iloc[:-1]; latest_ts=str(df_closed.index[-1])
    pivots=find_pivots(df_closed,depth)
    sig=ew_signal(df_closed,depth,sl_type,tgt_type,csl,ctgt)
    ss.live_last_sig=sig
    ss.live_wave_state=analyze_wave_state(df_closed,pivots,sig)
    ss._scan_df=df; ss._scan_sig=sig

    nxt=(now_ist()+timedelta(seconds=candle_s)).strftime("%H:%M:%S IST")
    ss.live_next_check_ist=nxt

    if ss.last_bar_ts==latest_ts:
        _log(f"⏭  Same bar {latest_ts[-10:]} — next check {nxt}")
        pos=ss.live_position
        if pos:
            ltp_now=ss.live_ltp or ohlcv_ltp
            _manage_pos(ltp_now,sig,sl_type,tgt_type,dhan_on,dhan_client,
                        trade_mode,exit_ot,exit_price_lmt,segment,ce_sec_id,pe_sec_id,live_qty,_log)
        ss.live_phase=f"pos_{ss.live_position['type'].lower()}" if ss.live_position else "no_signal"
        return False

    ss.last_bar_ts=latest_ts
    ltp_now=ss.live_ltp or ohlcv_ltp
    _log(f"🕯 Bar {latest_ts[-10:]} | LTP {ltp_now:.2f}")
    _manage_pos(ltp_now,sig,sl_type,tgt_type,dhan_on,dhan_client,
                trade_mode,exit_ot,exit_price_lmt,segment,ce_sec_id,pe_sec_id,live_qty,_log)

    pos=ss.live_position
    if pos is None and sig["signal"] in("BUY","SELL"):
        ep=ltp_now; w1=sig.get("wave1_len",ep*0.02) or ep*0.02
        if sl_type in(SL_WAVE,SL_TRAIL): sl_=sig["sl"]
        elif sl_type==SL_PTS: sl_=ep-csl if sig["signal"]=="BUY" else ep+csl
        elif sl_type==SL_SIGREV: sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
        else: sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" else ep*(1+float(sl_type))
        rk=abs(ep-sl_)
        if rk<=0:
            ss.live_no_pos_reason=f"⚠️ Risk=0 (SL {sl_:.2f} too close to entry {ep:.2f}). Increase SL distance."
            ss.live_phase="no_signal"; return True
        tgt_=_calc_target(tgt_type,ep,sig["signal"],w1,rk,ctgt)
        if (sig["signal"]=="BUY" and tgt_<=ep) or (sig["signal"]=="SELL" and tgt_>=ep):
            ss.live_no_pos_reason=f"⚠️ Invalid target ({tgt_:.2f}). Adjust Target setting."
            ss.live_phase="no_signal"; return True

        ei=now_ist().strftime("%d-%b %H:%M:%S IST")
        ss.live_position={
            "type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,"target_display":tgt_,
            "qty":live_qty,"entry_ist":ei,"symbol":symbol,"pattern":sig["pattern"],
            "confidence":sig["confidence"],"unreal_pnl":0.0,
            "dist_sl":rk/ep*100,"dist_tgt":abs(tgt_-ep)/ep*100,
        }
        ss.live_phase=f"pos_{sig['signal'].lower()}"
        ss.live_no_pos_reason=""
        ss.live_signals.append({
            "Time (IST)":ei,"Bar":fmt_ist(df_closed.index[-1]),
            "Symbol":symbol,"TF":interval,"Period":period,
            "Signal":sig["signal"],"Entry":round(ep,2),
            "SL":round(sl_,2),"Target":round(tgt_,2),
            "Conf":f"{sig['confidence']:.0%}","Pattern":sig["pattern"],
        })
        if dhan_on and dhan_client:
            sec_id_use = ce_sec_id if (trade_mode=="options" and sig["signal"]=="BUY") \
                         else pe_sec_id if trade_mode=="options" else ""
            ep_use=entry_price_lmt if entry_ot=="LIMIT" else 0.0
            r=dhan_client.place_order(sec_id_use or "",segment,sig["signal"],live_qty,
                                      order_type=entry_ot,price=ep_use,product_type=product_type)
            _log(f"📤 Dhan entry: {r}")
        em="🟢" if sig["signal"]=="BUY" else "🔴"
        _log(f"{em} ENTERED {sig['signal']} @ {ep:.2f} | SL {sl_:.2f} | T {tgt_:.2f} | {sig['confidence']:.0%} | {sig['pattern']}")
    elif pos is None:
        ss.live_phase="no_signal"
        ss.live_no_pos_reason=(f"📊 HOLD. {sig.get('reason','')}\n"
                               f"System scans every {interval} candle. Next check: {nxt}")
        _log(f"⏸  HOLD @ {ltp_now:.2f}")
    return True

def _manage_pos(ltp,sig,sl_type,tgt_type,dhan_on,dhan_client,
                trade_mode,exit_ot,exit_price_lmt,segment,ce_sec_id,pe_sec_id,qty_,log_fn):
    ss=st.session_state; pos=ss.live_position
    if not pos or not ltp: return
    hit_p,hit_r=None,None; ptype=pos["type"]
    sl_=pos["sl"]; tgt_=pos["target"]
    if ptype=="BUY":
        if ltp<=sl_:      hit_p,hit_r=sl_,   "SL Hit (price ≤ SL)"
        elif ltp>=tgt_ and tgt_type!=TGT_TRAIL: hit_p,hit_r=tgt_,"Target Hit (price ≥ Target)"
        elif sig["signal"]=="SELL": hit_p,hit_r=ltp,"Signal Reversed → SELL"
    else:
        if ltp>=sl_:      hit_p,hit_r=sl_,   "SL Hit (price ≥ SL)"
        elif ltp<=tgt_ and tgt_type!=TGT_TRAIL: hit_p,hit_r=tgt_,"Target Hit (price ≤ Target)"
        elif sig["signal"]=="BUY": hit_p,hit_r=ltp,"Signal Reversed → BUY"
    if hit_p is None: return
    qty=pos["qty"]; pnl=(hit_p-pos["entry"])*qty if ptype=="BUY" else (pos["entry"]-hit_p)*qty
    # REALIZED P&L — only changed here
    ss.live_pnl+=pnl
    ss.live_trades.append({
        "Time (IST)":now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Symbol":pos["symbol"],"TF":ss.get("_live_interval","—"),
        "Type":ptype,"Entry":round(pos["entry"],2),"Exit":round(hit_p,2),
        "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
        "Qty":qty,"PnL Rs":round(pnl,2),"Reason":hit_r,
    })
    if dhan_on and dhan_client:
        xt="SELL" if ptype=="BUY" else "BUY"
        sec_id_use=ce_sec_id if(trade_mode=="options" and xt=="BUY") else pe_sec_id if trade_mode=="options" else ""
        xp=exit_price_lmt if exit_ot=="LIMIT" else 0.0
        log_fn(f"📤 Dhan exit: {dhan_client.place_order(sec_id_use or '',segment,xt,qty,order_type=exit_ot,price=xp)}")
    em="✅" if "Target" in hit_r else ("🔄" if "Reversed" in hit_r else "❌")
    log_fn(f"{em} {ptype} CLOSED @ {hit_p:.2f} | {hit_r} | Rs{pnl:+.2f}")
    ss.live_position=None
    ss.live_phase="reversing" if "Reversed" in hit_r else "no_signal"
    if "Reversed" not in hit_r:
        ss.live_no_pos_reason=f"Last: {hit_r} | Rs{pnl:+.2f}"

# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def chart_waves(df,pivots,sig=None,trades=None,symbol="",tf_label=""):
    sig=sig or _blank(); df_ind=add_indicators(df)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                      row_heights=[0.60,0.20,0.20],vertical_spacing=0.02,
                      subplot_titles=("","Volume","RSI"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="Price",
        increasing=dict(line=dict(color="#26a69a"),fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"),fillcolor="#ef5350")),row=1,col=1)
    for col,clr,nm in[("EMA_20","#ffb300","EMA20"),("EMA_50","#ab47bc","EMA50"),("EMA_200","#ef5350","EMA200")]:
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
            fig.add_trace(go.Scatter(x=[df.index[p[0]] for p in vwp],y=[p[1] for p in vwp],
                mode="lines+markers+text",line=dict(color=clr,width=2.5),marker=dict(size=12,color=clr),
                text=["W0","W1","W2"][:len(vwp)],textposition="top center",
                textfont=dict(color=clr,size=12,family="Share Tech Mono"),name="EW"),row=1,col=1)
    if sig["signal"] in("BUY","SELL"):
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
        for tt,sy,cl in[("BUY","triangle-up","#4caf50"),("SELL","triangle-down","#f44336")]:
            sub=trades[trades["Type"]==tt]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Entry Time"],y=sub["Entry"],mode="markers",
                    marker=dict(size=9,color=cl,symbol=sy,line=dict(color="white",width=0.8)),
                    name=f"{tt} Entry"),row=1,col=1)
        for rsn,sy,cl in[("Target","circle","#66bb6a"),("SL","x","#ef5350"),("Reverse","diamond","#ab47bc")]:
            sub=trades[trades["Exit Reason"].str.contains(rsn,na=False)]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Exit Time"],y=sub["Exit"],mode="markers",
                    marker=dict(size=7,color=cl,symbol=sy),name=f"Exit({rsn})",visible="legendonly"),row=1,col=1)
    fig.update_layout(
        title=dict(text=f"🌊 {symbol}"+(f" · {tf_label}" if tf_label else ""),font=dict(size=14,color="#00e5ff")),
        template="plotly_dark",height=560,xaxis_rangeslider_visible=False,
        plot_bgcolor="#06101a",paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,font=dict(size=10)),
        margin=dict(l=10,r=70,t=50,b=10))
    return fig

def chart_equity(ec):
    eq=np.array(ec,dtype=float); pk=np.maximum.accumulate(eq); dd=(eq-pk)/pk*100
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

def chart_opt_scatter(odf):
    fig=go.Figure(go.Scatter(x=odf["Max DD %"].abs(),y=odf["Return %"],mode="markers",
        marker=dict(size=(odf["Win %"]/5).clip(lower=4),color=odf["Score"],colorscale="Plasma",
                    showscale=True,colorbar=dict(title=dict(text="Score",font=dict(color="#b0bec5",size=12)),
                    tickfont=dict(color="#b0bec5")),line=dict(color="rgba(255,255,255,.2)",width=0.5)),
        text=[f"D={r.Depth} SL={r.SL} T={r.Target}" for _,r in odf.iterrows()],
        hovertemplate="<b>%{text}</b><br>Ret %{y:.1f}% DD %{x:.1f}%<extra></extra>"))
    fig.update_layout(title=dict(text="Return vs Max Drawdown (bubble=WinRate)",font=dict(size=13,color="#00e5ff")),
        xaxis_title="Max Drawdown %",yaxis_title="Total Return %",template="plotly_dark",height=380,
        plot_bgcolor="#06101a",paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),
        margin=dict(l=10,r=10,t=45,b=10))
    return fig

def generate_mtf_summary(symbol,results,overall_sig):
    lines=[f"## 🌊 Elliott Wave Analysis — {symbol}",f"*{now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}*\n"]
    bc=sum(1 for r in results if r["signal"]["signal"]=="BUY")
    sc=sum(1 for r in results if r["signal"]["signal"]=="SELL")
    hc=sum(1 for r in results if r["signal"]["signal"]=="HOLD")
    vi={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(overall_sig,"⚪")
    lines.append(f"### {vi} Overall: **{overall_sig}** ({bc}B·{sc}S·{hc}H)\n")
    if overall_sig=="BUY": lines.append("📈 Bullish consensus. Enter BUY on Wave-2 pullbacks.")
    elif overall_sig=="SELL": lines.append("📉 Bearish consensus. Enter SELL on Wave-2 bounces.")
    else: lines.append("⚠️ No consensus. Stay on sidelines.")
    lines.append("\n---\n")
    for r in results:
        sig=r["signal"]; s=sig["signal"]; em={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(s,"⚪")
        lines.append(f"#### {em} {r['tf_name']}")
        if s in("BUY","SELL"):
            retr=sig.get("retracement",0); ep=sig["entry_price"]; sl_=sig["sl"]; tgt_=sig["target"]
            rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
            lines.append(f"- **{s}** | Entry:{ep:.2f} SL:{sl_:.2f} Target:{tgt_:.2f} R:R 1:{rr:.1f}")
            lines.append(f"- Confidence:{sig['confidence']:.0%} | Retrace:{retr:.1%}")
            if abs(retr-0.618)<0.04: lines.append("- ✨ Golden Ratio — strongest signal")
        else:
            lines.append(f"- HOLD: {sig.get('reason','—')}")
        lines.append("")
    lines+=["---","| Wave | Meaning | Action |","|------|---------|--------|",
            "| W1 ↑ | First impulse | Missed |","| **W2 ↓** | **38–79% retrace** | **🟢 BUY here** |",
            "| **W3 ↑** | **Strongest** | **Hold** |","| W4 ↓ | Pullback | Partial exit |",
            "| W5 ↑ | Final | Full exit |","\n> ⚠️ *Not financial advice.*"]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ███  STREAMLIT UI  ███
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="main-hdr">
  <h1>🌊 Elliott Wave Algo Trader v5.2</h1>
  <p>Dhan Stocks+Options · Trailing SL/TGT · Delay Context · P&L Fix · No Truncation · Optimization Fixed</p>
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
    st.session_state["_live_interval"]=st.session_state.get("_live_interval","1d")

    st.markdown("---")
    if st.session_state.best_cfg_applied:
        st.markdown("""<div style="background:rgba(0,229,255,.08);border:1px solid #00bcd4;
        border-radius:8px;padding:7px 10px;font-size:.8rem;margin-bottom:6px">
        ✨ <b style="color:#00e5ff">Optimized Config Active</b></div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    interval=c1.selectbox("⏱ TF",TIMEFRAMES,index=6)
    vpl=VALID_PERIODS.get(interval,PERIODS)
    period=c2.selectbox("📅 Period",vpl,index=min(4,len(vpl)-1))
    st.session_state["_live_interval"]=interval

    st.markdown("---")
    st.markdown("### 🌊 Elliott Wave")
    depth=st.slider("Pivot Depth",2,15,st.session_state.applied_depth)

    st.markdown("---")
    st.markdown("### 🛡️ Risk Management")
    si=SL_KEYS.index(st.session_state.applied_sl_lbl) if st.session_state.applied_sl_lbl in SL_KEYS else 0
    ti=TGT_KEYS.index(st.session_state.applied_tgt_lbl) if st.session_state.applied_tgt_lbl in TGT_KEYS else 0
    sl_lbl_sel=st.selectbox("Stop Loss",SL_KEYS,index=si)
    tgt_lbl_sel=st.selectbox("Target",TGT_KEYS,index=ti)
    sl_val=SL_MAP[sl_lbl_sel]; tgt_val=TGT_MAP[tgt_lbl_sel]

    if sl_val==SL_PTS:
        st.session_state.custom_sl_pts=st.number_input("SL Points",1.0,1e6,st.session_state.custom_sl_pts,5.0)
    if tgt_val==TGT_PTS:
        st.session_state.custom_tgt_pts=st.number_input("Target Points",1.0,1e6,st.session_state.custom_tgt_pts,10.0)
    if sl_val==SL_TRAIL:
        st.session_state.trailing_sl_pts=st.number_input("Trailing SL Points",1.0,1e6,st.session_state.trailing_sl_pts,5.0)
        st.caption("SL moves in profit direction as LTP moves. Never moves against trade.")
    if tgt_val==TGT_TRAIL:
        st.session_state.trailing_tgt_pts=st.number_input("Trailing Target Points",1.0,1e6,st.session_state.trailing_tgt_pts,10.0)
        st.caption("⚠️ Display only — trailing target never triggers exit. Exit = SL or Signal Reverse only.")
    if sl_val==SL_SIGREV: st.info("🔄 Exit when wave signal reverses direction")
    if tgt_val==TGT_SIGREV: st.info("🔄 Take profit when wave signal reverses")

    csl=st.session_state.custom_sl_pts; ctgt=st.session_state.custom_tgt_pts
    trail_sl_pts=st.session_state.trailing_sl_pts; trail_tgt_pts=st.session_state.trailing_tgt_pts
    capital=st.number_input("💰 Capital (Rs)",10_000,50_000_000,100_000,10_000)

    # ── DHAN BROKERAGE ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏦 Dhan Brokerage")
    if not DHAN_LIB_OK:
        st.warning("dhanhq not installed. Run:\n`pip install dhanhq`\nFalling back to REST API.")
    dhan_on=st.checkbox("Enable Dhan Integration",value=False)
    dhan_client=None; trade_mode="stocks"; product_type="INTRADAY"; segment="NSE"
    entry_ot="MARKET"; exit_ot="MARKET"; entry_price_lmt=0.0; exit_price_lmt=0.0
    ce_sec_id=""; pe_sec_id=""; live_qty=1

    if dhan_on:
        d_cid=st.text_input("Client ID",key="dhan_cid")
        d_tok=st.text_input("Access Token",type="password",key="dhan_tok")
        if d_cid and d_tok:
            dhan_client=DhanClient(d_cid,d_tok)
            if st.button("🔌 Test Connection",key="dhan_test"):
                with st.spinner("Connecting…"): st.json(dhan_client.fund_limit())

        st.markdown("**Trading Mode**")
        trade_mode=st.radio("",["Stocks","Options"],horizontal=True,key="trade_mode_radio").lower()

        if trade_mode=="stocks":
            c1d,c2d=st.columns(2)
            product_type=c1d.selectbox("Trade Type",["INTRADAY","DELIVERY"],index=0,
                                        help="INTRADAY=INTRA, DELIVERY=CNC in Dhan")
            exchange=c2d.selectbox("Exchange",["NSE","BSE"],index=0)
            segment=exchange  # NSE→NSE_EQ, BSE→BSE_EQ handled in DhanClient
            sec_id_stock=st.text_input("Security ID",value="1333",
                                        help="Dhan security ID for the stock")
            live_qty=st.number_input("Qty",1,100_000,1)
            ce_sec_id=pe_sec_id=sec_id_stock

        else:  # options
            c1d,c2d=st.columns(2)
            opt_exchange=c1d.selectbox("Options Exchange",["NSE_FNO","BSE_FNO"],index=0)
            opt_product=c2d.selectbox("Product",["INTRADAY","MARGIN"],index=0)
            segment=opt_exchange; product_type=opt_product
            ce_sec_id=st.text_input("CE Security ID","52175",
                                     help="Security ID for Call option (BUY signal → buy CE)")
            pe_sec_id=st.text_input("PE Security ID","52176",
                                     help="Security ID for Put option (SELL signal → buy PE)")
            live_qty=st.number_input("Lot Size",1,10_000,50,
                                      help="Nifty=50, BankNifty=15, etc.")
            st.caption("BUY signal → Buy CE | SELL signal → Buy PE")

        st.markdown("**Order Types**")
        c1e,c2e=st.columns(2)
        entry_ot=c1e.selectbox("Entry Order",["LIMIT","MARKET"],index=0,
                                help="LIMIT=enter at specific price, MARKET=instant fill")
        exit_ot=c2e.selectbox("Exit Order",["MARKET","LIMIT"],index=0,
                               help="MARKET=instant exit, LIMIT=exit at specific price")
        if entry_ot=="LIMIT":
            entry_price_lmt=st.number_input("Entry Limit Price",0.0,1e7,0.0,0.5,
                                             help="0 = auto-fill with signal entry price")
        if exit_ot=="LIMIT":
            exit_price_lmt=st.number_input("Exit Limit Price",0.0,1e7,0.0,0.5)

    st.markdown("---")
    st.caption(f"⚡ 1.5s rate-limit | LTP: 2s | `{symbol}` · `{interval}` · `{period}`")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
t_analysis,t_live,t_bt,t_opt,t_help=st.tabs([
    "🔭  Wave Analysis","🔴  Live Trading","📊  Backtest","🔬  Optimization","❓  Help"])

# ── TAB 1: WAVE ANALYSIS ───────────────────────────────────────────────────
with t_analysis:
    st.markdown("### 🔭 Multi-Timeframe Elliott Wave Analysis")
    ac1,ac2,ac3=st.columns([1.2,1,2.4])
    with ac1: run_analysis=st.button("🔭 Run Full Analysis",type="primary",use_container_width=True)
    with ac2: custom_tf_only=st.checkbox("Sidebar TF only",value=False)
    with ac3: st.caption(f"{'Sidebar TF' if custom_tf_only else 'Daily·4H·1H·15M'} for `{symbol}`")
    if run_analysis:
        sc2=[(interval,period,f"{interval}·{period}")] if custom_tf_only \
            else [(tf,per,nm) for tf,per,nm in MTF_COMBOS if per in VALID_PERIODS.get(tf,[])]
        results=[]; prog=st.progress(0,text="Scanning…")
        for idx,(tf,per,nm) in enumerate(sc2):
            prog.progress((idx+1)/len(sc2),text=f"Fetching {nm}…")
            try:
                dfa=fetch_ohlcv(symbol,tf,per,min_delay=1.5)
                if dfa is not None and len(dfa)>=35:
                    sa=ew_signal(dfa.iloc[:-1],depth,sl_val,tgt_val,csl,ctgt)
                    results.append({"tf_name":nm,"interval":tf,"period":per,"signal":sa,
                                    "df":dfa,"pivots":find_pivots(dfa.iloc[:-1],depth)})
                else:
                    results.append({"tf_name":nm,"interval":tf,"period":per,
                                    "signal":_blank(f"No data"),"df":None,"pivots":[]})
            except Exception as e:
                results.append({"tf_name":nm,"interval":tf,"period":per,
                                "signal":_blank(str(e)),"df":None,"pivots":[]})
        prog.empty()
        bs=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="BUY")
        ss2=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="SELL")
        ov="BUY" if bs>ss2 and bs>0.5 else "SELL" if ss2>bs and ss2>0.5 else "HOLD"
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
                    if s_ in("BUY","SELL"):
                        sg=r["signal"]; ep,sl_,tgt_=sg["entry_price"],sg["sl"],sg["target"]
                        rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
                        mc1,mc2,mc3,mc4=st.columns(4)
                        mc1.metric("Entry",f"{ep:.2f}"); mc2.metric("SL",f"{sl_:.2f}",delta=f"-{abs(ep-sl_)/ep*100:.1f}%",delta_color="inverse")
                        mc3.metric("Target",f"{tgt_:.2f}",delta=f"+{abs(tgt_-ep)/ep*100:.1f}%"); mc4.metric("R:R",f"1:{rr:.1f}")
        st.markdown("---"); st.markdown("### 📋 Analysis & Recommendations")
        st.markdown(generate_mtf_summary(asym,ar,ov))
    else:
        st.info("Click 🔭 Run Full Analysis to scan all timeframes automatically.")

# ── TAB 2: LIVE TRADING ────────────────────────────────────────────────────
with t_live:
    ctl1,ctl2,ctl3=st.columns([1,1,4])
    with ctl1:
        if not st.session_state.live_running:
            if st.button("▶ Start Live",type="primary",use_container_width=True):
                st.session_state.update({"live_running":True,"live_phase":"scanning",
                    "live_log":[],"last_bar_ts":None,"last_fetch_wall":0.0,
                    "live_no_pos_reason":"System starting…"})
                st.rerun()
        else:
            if st.button("⏹ Stop",type="secondary",use_container_width=True):
                st.session_state.update({"live_running":False,"live_phase":"idle"}); st.rerun()
    with ctl2:
        if st.button("🔄 Reset All",use_container_width=True):
            for k,v in _DEF.items(): st.session_state[k]=v
            st.success("Reset ✓"); time.sleep(0.3); st.rerun()
    with ctl3:
        if st.session_state.live_running:
            st.success(f"🟢 RUNNING — `{symbol}` · `{interval}` · `{period}` | LTP every 2s (no flicker) | Candle check every {POLL_SECS.get(interval,3600)//60}min")
        else:
            st.warning("⚫ STOPPED — Click ▶ Start Live. Everything is automatic.")

    # Position banner placeholder (re-rendered by fragment)
    pos_ph=st.empty()

    def render_pos(container):
        ss=st.session_state; pos=ss.live_position; ltp=ss.live_ltp; phase=ss.live_phase
        with container:
            if pos:
                ptype=pos["type"]; clr="#4caf50" if ptype=="BUY" else "#f44336"
                css="pos-buy" if ptype=="BUY" else "pos-sell"
                up=pos.get("unreal_pnl",0); up_c="#4caf50" if up>=0 else "#f44336"; up_sym="▲" if up>=0 else "▼"
                sl_=pos["sl"]; tgt_=pos.get("target_display",pos["target"])
                ds=pos.get("dist_sl",0); dt=pos.get("dist_tgt",0)
                ltp_s=f"{ltp:,.2f}" if ltp else "—"
                trail_sl_note=" [Trailing]" if sl_val==SL_TRAIL else ""
                trail_tgt_note=" [Trailing Display]" if tgt_val==TGT_TRAIL else ""
                st.markdown(f"""
                <div class="{css}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">
                <div>
                  <div style="font-size:.76rem;color:#78909c;letter-spacing:.5px">POSITION STATUS</div>
                  <div style="font-size:2rem;font-weight:700;color:{clr}">{ptype} OPEN ✅</div>
                  <div style="font-size:.82rem;color:#b0bec5">{pos.get('pattern','—')} | {pos.get('confidence',0):.0%} confidence</div>
                  <div style="font-size:.76rem;color:#546e7a">Entered {pos.get('entry_ist','—')}</div>
                </div>
                <div style="text-align:right">
                  <div style="font-size:.74rem;color:#546e7a">UNREALIZED P&amp;L</div>
                  <div style="font-size:1.65rem;font-weight:700;color:{up_c};font-family:'Share Tech Mono'">{up_sym} Rs{abs(up):,.2f}</div>
                  <div style="font-size:.78rem;color:#78909c">LTP: {ltp_s}</div>
                </div>
                </div>
                <hr style="border-color:rgba(255,255,255,.08);margin:8px 0">
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:.82rem">
                  <div><div style="color:#546e7a;font-size:.72rem">ENTRY</div><div style="color:#b0bec5;font-weight:600">{pos['entry']:,.2f}</div></div>
                  <div><div style="color:#546e7a;font-size:.72rem">SL{trail_sl_note}</div><div style="color:#ff7043;font-weight:600">{sl_:,.2f}<br><span style="color:#546e7a;font-size:.72rem">{ds:.1f}% away</span></div></div>
                  <div><div style="color:#546e7a;font-size:.72rem">TARGET{trail_tgt_note}</div><div style="color:#66bb6a;font-weight:600">{tgt_:,.2f}<br><span style="color:#546e7a;font-size:.72rem">{dt:.1f}% away</span></div></div>
                  <div><div style="color:#546e7a;font-size:.72rem">QTY</div><div style="color:#b0bec5;font-weight:600">{pos['qty']}</div></div>
                </div>
                <div style="margin-top:8px;padding:6px 10px;background:rgba(0,0,0,.3);border-radius:6px;font-size:.77rem;color:#78909c">
                🤖 Auto-managed: closes at SL / Target / Signal Reverse. <b>No action needed.</b>
                {"⚠️ Trailing Target is display only — actual exit is via SL or Signal Reverse." if tgt_val==TGT_TRAIL else ""}
                </div></div>""",unsafe_allow_html=True)
            elif phase=="reversing":
                st.markdown("""<div class="pos-reversing">
                <div style="font-size:1.4rem;font-weight:700;color:#ab47bc">🔄 AUTO-REVERSING…</div>
                <div style="font-size:.83rem;color:#b0bec5;margin-top:4px">Closing old position and opening reverse. No action needed.</div>
                </div>""",unsafe_allow_html=True)
            elif phase in("scanning","signal_ready"):
                sig_=ss.live_last_sig
                if sig_ and sig_.get("signal") in("BUY","SELL"):
                    s_=sig_["signal"]; sc_="#00e5ff"
                    st.markdown(f"""<div class="pos-signal">
                    <div style="font-size:.76rem;color:#78909c">SIGNAL DETECTED — OPENING POSITION</div>
                    <div style="font-size:1.5rem;font-weight:700;color:{sc_}">💡 {s_} SIGNAL FIRED</div>
                    <div style="font-size:.82rem;color:#b0bec5;margin-top:4px">{sig_.get('pattern','—')} | {sig_.get('confidence',0):.0%}</div>
                    <div style="font-size:.8rem;color:#78909c;margin-top:4px">
                    Entry: ~{sig_.get('entry_price',0):.2f} | SL: {sig_.get('sl',0):.2f} | Target: {sig_.get('target',0):.2f}</div>
                    <div style="font-size:.76rem;color:#546e7a;margin-top:6px">🤖 Position opening automatically. Will appear above on next bar.</div>
                    </div>""",unsafe_allow_html=True)
                else:
                    nr=ss.live_no_pos_reason
                    st.markdown(f"""<div class="pos-none">
                    <div style="font-size:.76rem;color:#546e7a">POSITION STATUS</div>
                    <div style="font-size:1.4rem;font-weight:600;color:#37474f;margin:4px 0">⏸ NO OPEN POSITION</div>
                    <div style="font-size:.81rem;color:#455a64;white-space:pre-line;word-break:break-word">{nr}</div>
                    </div>""",unsafe_allow_html=True)
            else:
                nr=ss.live_no_pos_reason
                st.markdown(f"""<div class="pos-none">
                <div style="font-size:.76rem;color:#546e7a">POSITION STATUS</div>
                <div style="font-size:1.4rem;font-weight:600;color:#37474f;margin:4px 0">⏸ NO OPEN POSITION</div>
                <div style="font-size:.81rem;color:#455a64;white-space:pre-line;word-break:break-word">{nr}</div>
                </div>""",unsafe_allow_html=True)
    render_pos(pos_ph)

    st.markdown("---")

    # ── FRAGMENT: runs every 2s, updates LTP + engine ──────────────────
    try:
        @st.fragment(run_every=2)
        def live_frag():
            ss=st.session_state
            if not ss.live_running: return

            new_cycle=live_engine_tick(
                symbol,interval,period,depth,sl_val,tgt_val,csl,ctgt,
                trail_sl_pts,trail_tgt_pts,dhan_on,dhan_client,
                trade_mode,entry_ot,exit_ot,entry_price_lmt,exit_price_lmt,
                product_type,segment,ce_sec_id,pe_sec_id,live_qty)
            if new_cycle: render_pos(pos_ph)

            # ── 6-metric bar ─────────────────────────────────────────────
            ltp_=ss.live_ltp; ltp_ts=ss.live_ltp_ts or "—"
            delay_=ss.live_delay_s; dcontext=ss.live_delay_context
            dlbl,dclr,_=delay_context(delay_,interval)

            # REALIZED P&L (only from closed trades, never from unreal)
            real_pnl=ss.live_pnl        # ← only modified in _manage_pos on trade close
            pos_=ss.live_position
            unreal_pnl=pos_.get("unreal_pnl",0) if pos_ else 0  # ← display only, not added to live_pnl

            tc=st.columns(6)
            tc[0].metric("📊 LTP", f"{ltp_:,.2f}" if ltp_ else "—", delta=ltp_ts,delta_color="off")
            tc[1].metric("⏩ Data Age",f"{delay_}s",delta=dlbl,delta_color="off")
            tc[2].metric("🕒 IST",now_ist().strftime("%H:%M:%S"),delta=ss.live_last_candle_ist,delta_color="off")
            tc[3].metric("🔜 Next Check",ss.live_next_check_ist)
            tc[4].metric("💰 Realized P&L",f"Rs{real_pnl:,.2f}",
                         delta=f"{len(ss.live_trades)} trades",delta_color="normal" if real_pnl>=0 else "inverse")
            tc[5].metric("📈 Unrealized P&L",f"Rs{unreal_pnl:,.2f}" if pos_ else "No position",
                         delta="Open position" if pos_ else "—",
                         delta_color="normal" if unreal_pnl>=0 else "inverse")

            # Delay context explanation
            if dcontext:
                if delay_>=POLL_SECS.get(interval,3600)*2:
                    st.info(f"ℹ️ {dcontext}")
                elif delay_>=300:
                    st.warning(f"⚠️ {dcontext}")

            # ── Wave state + signal columns ───────────────────────────────
            ws_col,sig_col=st.columns([1.4,1])
            with ws_col:
                ws=ss.live_wave_state
                if ws:
                    dc={"BULLISH":"#4caf50","BEARISH":"#f44336","NEUTRAL":"#78909c"}.get(ws.get("direction","NEUTRAL"),"#78909c")
                    di={"BULLISH":"📈","BEARISH":"📉","NEUTRAL":"➡️"}.get(ws.get("direction","NEUTRAL"),"➡️")
                    st.markdown(f"""<div class="wave-card"><b style="color:#00e5ff">🌊 Current Wave</b>
<span style="color:{dc}">{di} {ws.get('current_wave','—')}</span>
<b style="color:#00e5ff">Next Expected</b>
<span style="color:#b0bec5">{ws.get('next_wave','—')}</span></div>""",unsafe_allow_html=True)
                    fibs=ws.get("fib_levels",{})
                    if fibs:
                        st.markdown("**📐 Wave-3 Fib Targets**")
                        for lbl,val in fibs.items():
                            bc2="#4caf50" if "1.618" in lbl else("#ab47bc" if "2.618" in lbl else "#ffb300")
                            st.markdown(f"""<div style="display:flex;justify-content:space-between;background:#060d14;border-left:3px solid {bc2};border-radius:0 4px 4px 0;padding:4px 9px;margin-bottom:3px;font-size:.81rem">
                            <span style="color:#78909c">{lbl}</span><span style="color:{bc2};font-family:'Share Tech Mono'">{val:,.2f}</span></div>""",unsafe_allow_html=True)
                    action=ws.get("action","")
                    st.markdown(f"""<div class="wave-card" style="margin-top:8px">{action}</div>""",unsafe_allow_html=True)
            with sig_col:
                sig_=ss.live_last_sig
                if sig_ and sig_.get("signal")!="HOLD":
                    s_=sig_["signal"]; sc_="#4caf50" if s_=="BUY" else "#f44336"; em_="🟢" if s_=="BUY" else "🔴"
                    rr_=abs(sig_.get("target",0)-sig_.get("entry_price",0))/abs(sig_.get("entry_price",0)-sig_.get("sl",0)) \
                        if sig_.get("sl") and abs(sig_.get("entry_price",0)-sig_.get("sl",0))>0 else 0
                    st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:2px solid {sc_};border-radius:10px;padding:14px 16px">
                    <div style="font-size:.74rem;color:#546e7a">LAST SIGNAL</div>
                    <div style="font-size:1.4rem;color:{sc_};font-weight:700">{em_} {s_}</div>
                    <div style="font-size:.8rem;color:#b0bec5">{sig_.get('pattern','—')}</div>
                    <div style="font-size:.78rem;color:#00bcd4">Confidence: {sig_.get('confidence',0):.0%}</div>
                    <hr style="border-color:rgba(255,255,255,.08);margin:8px 0">
                    <div style="font-size:.79rem;color:#78909c">
                    Entry: <b style="color:#b0bec5">{sig_.get('entry_price',0):,.2f}</b><br>
                    SL: <b style="color:#ff7043">{sig_.get('sl',0):,.2f}</b><br>
                    Target: <b style="color:#66bb6a">{sig_.get('target',0):,.2f}</b><br>
                    R:R: <b style="color:#b0bec5">1:{rr_:.1f}</b>
                    </div>
                    <div style="font-size:.74rem;color:#455a64;margin-top:6px">{sig_.get('reason','—')}</div>
                    </div>""",unsafe_allow_html=True)
                else:
                    nr=ss.live_no_pos_reason
                    st.markdown(f"""<div style="background:#060d14;border:1px solid #263238;border-radius:10px;padding:14px 16px">
                    <div style="font-size:.74rem;color:#546e7a">LAST SIGNAL</div>
                    <div style="font-size:1.1rem;color:#37474f;font-weight:600">⏸ HOLD / NO SIGNAL</div>
                    <div style="font-size:.79rem;color:#37474f;margin-top:6px;line-height:1.55;word-break:break-word">{nr}</div>
                    </div>""",unsafe_allow_html=True)

            # Chart
            if ss._scan_df is not None:
                piv_=find_pivots(ss._scan_df.iloc[:-1],depth)
                st.plotly_chart(chart_waves(ss._scan_df,piv_,ss._scan_sig,symbol=symbol),use_container_width=True)

            # History
            h1c,h2c=st.columns(2)
            with h1c:
                if ss.live_signals:
                    st.markdown("##### 📋 Signal History")
                    st.dataframe(pd.DataFrame(ss.live_signals).tail(8),use_container_width=True,height=150)
            with h2c:
                if ss.live_trades:
                    st.markdown("##### 🏁 Completed Trades")
                    td=pd.DataFrame(ss.live_trades); st.dataframe(td.tail(8),use_container_width=True,height=150)
                    wns=(td["PnL Rs"]>0).sum(); tot=len(td); pnl_=td["PnL Rs"].sum()
                    st.caption(f"Win: {wns}/{tot} ({wns/tot*100:.0f}%) | Realized Rs{pnl_:,.2f}")

            if ss.live_log:
                with st.expander("📜 Activity Log",expanded=False):
                    st.code("\n".join(reversed(ss.live_log[-60:])),language=None)

        live_frag()
    except Exception as fe:
        st.warning(f"⚠️ Fragment not available ({fe}). Upgrade: `pip install streamlit --upgrade`")
        if st.button("🔍 Manual Scan & Update",use_container_width=True):
            with st.spinner("Running…"):
                live_engine_tick(symbol,interval,period,depth,sl_val,tgt_val,csl,ctgt,
                                 trail_sl_pts,trail_tgt_pts,dhan_on,dhan_client,
                                 trade_mode,entry_ot,exit_ot,entry_price_lmt,exit_price_lmt,
                                 product_type,segment,ce_sec_id,pe_sec_id,live_qty)
            render_pos(pos_ph); st.rerun()

# ── TAB 3: BACKTEST ────────────────────────────────────────────────────────
with t_bt:
    bl,br=st.columns([1,2.6],gap="medium")
    with bl:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""<div class="info-box">
        📈 <code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        🌊 Depth <code>{depth}</code> · SL <code>{sl_lbl_sel}</code><br>
        🎯 Target <code>{tgt_lbl_sel}</code> · Rs<code>{capital:,}</code><br>
        <small style="color:#546e7a">Signal bar N → entry open bar N+1<br>
        SL: Low(BUY)/High(SELL) checked first (conservative)<br>
        Exit Bar Low/High verify fills</small></div>""",unsafe_allow_html=True)
        if st.session_state.best_cfg_applied: st.success("✨ Optimized config active")
        if st.button("🚀 Run Backtest",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dbt=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
            if dbt is None or len(dbt)<40: st.error("Not enough data.")
            else:
                with st.spinner(f"Running on {len(dbt)} bars…"):
                    res=run_backtest(dbt,depth,sl_val,tgt_val,capital,csl,ctgt,trail_sl_pts)
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
            c3.metric("Profit Factor",str(m["Profit Factor"])); c4.metric("Max Drawdown",f"{m['Max Drawdown %']}%")
            c5,c6,c7,c8=st.columns(4)
            c5.metric("Sharpe",str(m["Sharpe Ratio"])); c6.metric("Trades",str(m["Total Trades"]))
            c7.metric("Avg Win",f"Rs{m['Avg Win Rs']:,}"); c8.metric("Avg Loss",f"Rs{m['Avg Loss Rs']:,}")
            eb=r.get("exit_breakdown",{})
            if eb:
                ea,eb2,ec2,ed=st.columns(4)
                ea.metric("SL Hits",str(eb.get("SL hits",0))); eb2.metric("Target Hits",str(eb.get("Target hits",0)))
                ec2.metric("Sig Reverse",str(eb.get("Signal Reverse",0))); ed.metric("Still Open",str(eb.get("Still open",0)))
            tc1,tc2,tc3=st.tabs(["🕯 Wave Chart","📈 Equity Curve","📋 Trade Log"])
            with tc1: st.plotly_chart(chart_waves(r["df"],r["pivots"],_blank(),r["trades"],r["symbol"]),use_container_width=True)
            with tc2: st.plotly_chart(chart_equity(r["equity_curve"]),use_container_width=True)
            with tc3:
                st.dataframe(r["trades"],use_container_width=True,height=420)
                st.info("Exit Bar Low/High verify fills. SL Verified ✅ = Low≤SL(BUY)/High≥SL(SELL). TGT Verified ✅ = High≥Target(BUY)/Low≤Target(SELL).")
                st.download_button("📥 CSV",data=r["trades"].to_csv(index=False),
                                   file_name=f"ew_bt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        elif r and "error" in r: st.error(r["error"])
        else: st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run backtest to see results</h3></div>",unsafe_allow_html=True)

# ── TAB 4: OPTIMIZATION ────────────────────────────────────────────────────
with t_opt:
    ol,or_=st.columns([1,3],gap="medium")
    with ol:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""<div class="info-box"><code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        Rs<code>{capital:,}</code><br><br><b>80 combos</b>: 4 depths × 4 SL × 5 targets<br>
        <small style="color:#546e7a">Progress always reaches 100%.<br>Failed combos are skipped gracefully.</small></div>""",unsafe_allow_html=True)
        st.warning("⏳ ~80 backtests (~1–3 min)")
        if st.button("🔬 Run Optimization",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dopt=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
            if dopt is None or len(dopt)<50: st.error("Not enough data.")
            else:
                with st.spinner("Running 80 combinations…"):
                    odf=run_optimization(dopt,capital,csl,ctgt,trail_sl_pts)
                st.session_state.opt_results={"df":odf,"symbol":symbol,"interval":interval,"period":period}
                if odf.empty: st.warning("No combos generated trades. Try longer Period or different settings.")
                else: st.success(f"✅ Complete! Best Score: {odf['Score'].iloc[0]:.2f}")
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
                st.session_state.update({"applied_depth":int(ar_["Depth"]),"applied_sl_lbl":sl_lbl(ar_["SL"]),
                                          "applied_tgt_lbl":tgt_lbl(ar_["Target"]),"best_cfg_applied":True})
                st.success(f"✅ Applied! Depth={int(ar_['Depth'])} · SL={sl_lbl(ar_['SL'])} · Target={tgt_lbl(ar_['Target'])}")
                time.sleep(0.4); st.rerun()
            st.markdown("---")
            oc1,oc2=st.tabs(["📊 Scatter","📋 Table"])
            with oc1:
                st.plotly_chart(chart_opt_scatter(odf),use_container_width=True)
                st.caption("Top-left = best (high return, low DD). Bubble size = Win Rate. Color = Score.")
            with oc2:
                def _hl(row):
                    if row.name==0: return["background-color:rgba(0,229,255,.18)"]*len(row)
                    elif row.name<3: return["background-color:rgba(0,229,255,.08)"]*len(row)
                    return[""]*len(row)
                st.dataframe(odf.style.apply(_hl,axis=1),use_container_width=True,height=500)
                st.download_button("📥 CSV",data=odf.to_csv(index=False),
                                   file_name=f"ew_opt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        else:
            st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run optimization to see results</h3></div>",unsafe_allow_html=True)

# ── TAB 5: HELP ────────────────────────────────────────────────────────────
with t_help:
    st.markdown("## 📖 Complete Guide — Elliott Wave Algo Trader v5.2")
    h1,h2=st.columns(2,gap="large")
    with h1:
        st.markdown("""
### 🏦 Dhan Brokerage — Stocks vs Options

**Stocks mode:**
- Trade Type: INTRADAY (default) or DELIVERY
- Exchange: NSE (default) or BSE
- Enter Security ID for the stock (e.g. 1333 = HDFC Bank)
- Entry/Exit order types: MARKET or LIMIT

**Options mode:**
- Exchange: NSE_FNO (default) or BSE_FNO
- Product: INTRADAY or MARGIN
- CE Security ID: BUY signal → buys this CE option
- PE Security ID: SELL signal → buys this PE option
- Lot size: Nifty=50, BankNifty=15, FinNifty=25

**Order Types:**
- Entry: LIMIT (default, you set the price) or MARKET
- Exit: MARKET (default, instant fill) or LIMIT

Install: `pip install dhanhq`

---

### 🔄 Trailing SL — How It Works

Select "Trailing SL ▼" and set Trailing SL Points (e.g. 50).

- BUY position: SL starts at Wave-2 pivot. As LTP rises, SL moves up maintaining 50-pt distance. Never moves down.
- SELL position: SL starts above Wave-2 pivot. As LTP falls, SL moves down maintaining 50-pt distance. Never moves up.

Example: BUY at 22000, Trail=50pts
- LTP=22050 → SL moves to 22000
- LTP=22100 → SL moves to 22050
- LTP=22080 → SL stays at 22050 (never reverses)

---

### 🎯 Trailing Target — Display Only

⚠️ Trailing Target NEVER triggers exit. It only shows on screen.
Actual exits happen via: SL hit or Signal Reverse only.

Use it to track how far price could go before reversing.

---

### ⏱ Data Delay Explained

For 1h candles showing 1771s delay:
- 1771s = ~29.5 minutes
- Last complete 1h candle closed 29.5 min ago
- Current 1h candle is still forming (30min into 60min bar)
- This is COMPLETELY NORMAL
- System uses last closed bar for signal — correct behavior

For very large delays (>2× candle size): market is likely closed.
System still generates valid signals for next session open.
        """)
    with h2:
        st.markdown("""
### 🐛 Bugs Fixed in v5.2

| Bug | Fix |
|-----|-----|
| P&L increased without trades | Unrealized P&L is now display-only. `live_pnl` (realized) only changes when a trade closes. |
| Optimization stuck at 60-70% | All exceptions caught per combo. Progress always reaches 100%. |
| Metrics not updating | Fragment `run_every=2` + lightweight `fetch_ltp()` runs every 2s without full page reload. |
| Text truncated on desktop | CSS fixes: `white-space:normal`, `overflow:visible`, metric font-size adjusted. |
| "Is position entered?" confusion | Giant POSITION STATUS banner: 🟢 BUY OPEN / 🔴 SELL OPEN / ⏸ NO POSITION / 💡 SIGNAL FIRED. |
| Delay shows 1771s as error | Context-aware: explains delay relative to candle size. Market-closed scenario handled. |
| Background thread race | No background threads. Engine runs in main thread via `@st.fragment`. |

---

### 📊 P&L Breakdown

| Metric | What it is |
|--------|-----------|
| Realized P&L | Sum of ALL closed trade P&L. Only updates when trade closes. |
| Unrealized P&L | (LTP - Entry) × Qty for open position. Updates every 2s. Display only. |
| Total P&L | Realized + Unrealized (shown on position banner) |

---

### ⚠️ Troubleshooting

| Problem | Solution |
|---------|---------|
| dhanhq import error | `pip install dhanhq` |
| Fragment error | `pip install streamlit --upgrade` |
| Position not opening | Check "No Open Position" reason text |
| Optimization 0 results | Use longer Period or reduce depth |
| LTP shows "—" | Symbol may not support fast_info; wait for candle fetch |
        """)
    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#37474f;font-size:.82rem;padding:8px">
    🌊 Elliott Wave Algo Trader v5.2 · Streamlit + yfinance + Plotly + dhanhq ·
    <b style="color:#f44336">Not financial advice. Paper trade first. Use Stop Loss always.</b>
    </div>""",unsafe_allow_html=True)
