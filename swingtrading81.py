"""
╔══════════════════════════════════════════════════════════════╗
║       🌊  ELLIOTT WAVE ALGO TRADER  v5.4                     ║
║  Stale Signal Guard · Quick/Deep Opt · Full Auto Fix        ║
╚══════════════════════════════════════════════════════════════╝
pip install streamlit>=1.37 yfinance plotly pandas numpy dhanhq requests
streamlit run elliott_wave_algo_trader.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# ▼▼▼  CHANGE YOUR CREDENTIALS HERE  ▼▼▼
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CLIENT_ID    = "1104779876"
DEFAULT_ACCESS_TOKEN = ("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9"
                        ".eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc0NjY3MTA4"
                        "LCJpYXQiOjE3NzQ1ODA3MDgsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIs"
                        "IndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA0Nzc5ODc2In0"
                        ".tkaJSjQTuku8cS_lDrI6Y__7grZv6lsA_Sc4BGuRA_T4yMlj_"
                        "hCNtXQRYB4g3uMVva6z66nYDgpy6z6nibBo8Q")
# ▲▲▲  END CREDENTIALS  ▲▲▲

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, threading, itertools, warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

try:
    from dhanhq import dhanhq as DhanHQLib
    DHAN_LIB_OK = True
except ImportError:
    DHAN_LIB_OK = False

IST = timezone(timedelta(hours=5, minutes=30))
def now_ist(): return datetime.now(IST)
def fmt_ist(dt=None):
    if dt is None: return now_ist().strftime("%H:%M:%S IST")
    try:
        if hasattr(dt,"to_pydatetime"): dt=dt.to_pydatetime()
        if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).strftime("%d-%b %H:%M:%S IST")
    except: return str(dt)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="🌊 EW Algo Trader",page_icon="🌊",
                   layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Exo 2',sans-serif;}
section[data-testid="stSidebar"]>div{min-width:320px;width:320px!important;}
section[data-testid="stSidebar"] *{white-space:normal!important;word-break:break-word!important;}
div[data-testid="metric-container"] *{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;}
div[data-testid="stMetricValue"]{font-size:.93rem!important;}
div[data-testid="stMetricDelta"]{font-size:.74rem!important;}
.main-hdr{background:linear-gradient(135deg,#0a0e1a,#0d1b2a,#0a1628);border:1px solid #1e3a5f;
  border-radius:14px;padding:18px 24px;margin-bottom:10px;box-shadow:0 4px 24px rgba(0,229,255,.08);}
.main-hdr h1{font-family:'Exo 2';font-weight:700;background:linear-gradient(90deg,#00e5ff,#4dd0e1,#00bcd4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;font-size:1.7rem;}
.main-hdr p{color:#546e7a;margin:4px 0 0;font-size:.82rem;}
.pos-none{background:#07111e;border:2px dashed #263238;border-radius:14px;padding:14px 18px;}
.pos-buy{background:rgba(76,175,80,.11);border:3px solid #4caf50;border-radius:14px;padding:14px 18px;}
.pos-sell{background:rgba(244,67,54,.11);border:3px solid #f44336;border-radius:14px;padding:14px 18px;}
.pos-signal{background:rgba(0,229,255,.08);border:2px solid #00e5ff;border-radius:14px;padding:14px 18px;}
.pos-reversing{background:rgba(171,71,188,.14);border:3px solid #ab47bc;border-radius:14px;padding:14px 18px;text-align:center;}
.stale-warn{background:rgba(255,152,0,.12);border:2px solid #ff9800;border-radius:12px;padding:12px 16px;margin:6px 0;}
.sig-snap{background:#060d14;border:1px solid #1e3a5f;border-radius:10px;padding:12px 16px;margin:8px 0;}
.auto-banner{background:linear-gradient(135deg,rgba(0,229,255,.10),rgba(76,175,80,.08));
  border:2px solid #00e5ff;border-radius:14px;padding:14px 18px;margin-bottom:10px;}
.reliability-warn{background:rgba(244,67,54,.08);border:2px solid #ef5350;border-radius:12px;padding:14px 18px;margin:8px 0;}
.wave-card{background:#060d14;border:1px solid #1e3a5f;border-radius:8px;padding:10px 14px;
  font-family:'Share Tech Mono',monospace;font-size:.80rem;line-height:1.85;white-space:pre-wrap;word-break:break-word;}
.info-box{background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:11px 13px;font-size:.84rem;line-height:1.85;word-break:break-word;}
.best-cfg{background:rgba(0,229,255,.07);border:1px solid #00bcd4;border-radius:10px;padding:11px 15px;margin:5px 0;word-break:break-word;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:transparent;}
.stTabs [data-baseweb="tab"]{background:#0d1b2a;border-radius:8px;color:#546e7a;border:1px solid #1e3a5f;padding:5px 11px;font-size:.79rem;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0d3349,#0a2540)!important;color:#00e5ff!important;border-color:#00bcd4!important;}
div[data-testid="metric-container"]{background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;padding:8px 10px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS  — yfinance hard limits corrected
# ═══════════════════════════════════════════════════════════════════════════
TICKER_GROUPS={
    "🇮🇳 Indian Indices":{"Nifty 50":"^NSEI","Bank Nifty":"^NSEBANK","Sensex":"^BSESN",
        "Nifty IT":"^CNXIT","Nifty Midcap":"^NSEMDCP50","Nifty Auto":"^CNXAUTO"},
    "🇮🇳 NSE Stocks":{"Reliance":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
        "Infosys":"INFY.NS","ICICI Bank":"ICICIBANK.NS","SBI":"SBIN.NS",
        "Wipro":"WIPRO.NS","Axis Bank":"AXISBANK.NS","Maruti":"MARUTI.NS"},
    "₿ Crypto":{"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD","Solana":"SOL-USD"},
    "💱 Forex":{"USD/INR":"USDINR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X"},
    "🥇 Commodities":{"Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F"},
    "🌐 US Stocks":{"Apple":"AAPL","Tesla":"TSLA","NVIDIA":"NVDA","Microsoft":"MSFT"},
    "✏️ Custom Ticker":{"Custom":"__CUSTOM__"},
}

TIMEFRAMES=["1m","5m","15m","30m","1h","4h","1d","1wk"]
PERIODS=["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]

# CORRECTED yfinance limits (tested against actual API)
# 1m:7d, 5m:60d, 15m:60d, 30m:60d, 1h:730d, 4h:unlimited, 1d/1wk:unlimited
VALID_PERIODS={
    "1m" :["1d","5d","7d"],
    "5m" :["1d","5d","7d","1mo"],          # max ~60 days; 3mo fails
    "15m":["1d","5d","7d","1mo"],          # max ~60 days; 3mo fails
    "30m":["1d","5d","7d","1mo"],          # max ~60 days
    "1h" :["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h" :["1mo","3mo","6mo","1y","2y","5y"],
    "1d" :["1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["3mo","6mo","1y","2y","5y","10y","20y"],
}
POLL_SECS={"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400,"1wk":604800}

# Auto scan grid — only valid combos
AUTO_SCAN_GRID=[
    ("1d","2y"),("1d","1y"),("4h","6mo"),("4h","3mo"),
    ("1h","3mo"),("1h","1mo"),
]

# Optimization modes
OPT_MODES={
    "⚡ Quick (8 combos)": {
        "depths":[5,7],"sl_opts":["wave_auto",0.02],"tgt_opts":["wave_auto",2.0],
        "desc":"Tests 8 highest-probability combos. Fast, focused."},
    "⚖️ Balanced (24 combos)": {
        "depths":[5,7,10],"sl_opts":["wave_auto",0.015,0.025],"tgt_opts":["wave_auto",1.5,2.0,3.0],
        "desc":"24 combos covering most useful parameter space."},
    "🔬 Deep (80 combos)": {
        "depths":[3,5,7,10],"sl_opts":[0.01,0.02,0.03,"wave_auto"],"tgt_opts":[1.5,2.0,3.0,"wave_auto","fib_1618"],
        "desc":"Full 80-combo grid search. Thorough but slow (~3-5 min)."},
}

MIN_TRADES_FILTER = 8   # Ignore results with fewer than this many trades

SL_WAVE="wave_auto"; SL_PTS="__sl_pts__"; SL_SIGREV="__sl_sigrev__"; SL_TRAIL="__sl_trail__"
TGT_PTS="__tgt_pts__"; TGT_SIGREV="__tgt_sigrev__"; TGT_TRAIL="__tgt_trail__"
SL_MAP={
    "Wave Auto (Pivot)":SL_WAVE,
    "0.5%":0.005,"1%":0.01,"1.5%":0.015,"2%":0.02,"2.5%":0.025,"3%":0.03,"5%":0.05,
    "Custom Points ▼":SL_PTS,"Trailing SL ▼":SL_TRAIL,"Exit on Signal Reverse":SL_SIGREV,
}
TGT_MAP={
    "Wave Auto (Fib 1.618×W1)":"wave_auto",
    "R:R 1:1":1.0,"R:R 1:1.5":1.5,"R:R 1:2":2.0,"R:R 1:2.5":2.5,"R:R 1:3":3.0,
    "Fib 1.618×Wave 1":"fib_1618","Fib 2.618×Wave 1":"fib_2618",
    "Custom Points ▼":TGT_PTS,"Trailing Target ▼ (display only)":TGT_TRAIL,
    "Exit on Signal Reverse":TGT_SIGREV,
}
SL_KEYS=list(SL_MAP.keys()); TGT_KEYS=list(TGT_MAP.keys())
MIN_CONFIDENCE=0.68

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
_DEF={
    "live_running":False,"live_phase":"idle",
    "live_ltp":None,"live_ltp_prev":None,"live_ltp_ts":None,
    "live_last_candle_ist":"—","live_delay_s":0,"live_delay_ctx":"",
    "live_next_check_ist":"—","last_bar_ts":None,"last_fetch_wall":0.0,
    "live_position":None,"live_pnl":0.0,
    "live_signals":[],"live_trades":[],"live_log":[],
    "live_wave_state":None,"live_last_sig":None,
    "live_no_pos_reason":"Click ▶ Start Live. Everything runs automatically.",
    # Signal snapshot: what the signal looked like when it fired vs now
    "signal_snapshot":None,   # {bar_ts, ltp_at_signal, sl_at_signal, tgt_at_signal, direction, time_ist, df_after}
    "stale_check_enabled":False,
    "_scan_sig":None,"_scan_df":None,
    "bt_results":None,"opt_results":None,
    "_analysis_results":None,"_analysis_overall":"HOLD","_analysis_symbol":"",
    # Applied config from optimization / full auto
    "applied_depth":5,
    "applied_sl_lbl":"Wave Auto (Pivot)",
    "applied_tgt_lbl":"Wave Auto (Fib 1.618×W1)",
    "applied_interval":"1d",
    "applied_period":"1y",
    "best_cfg_applied":False,
    "custom_sl_pts":50.0,"custom_tgt_pts":100.0,
    "trailing_sl_pts":50.0,"trailing_tgt_pts":150.0,
    "auto_mode":False,"auto_running":False,
    "auto_status":"","auto_best_cfg":None,"auto_all_results":None,"auto_log":[],
}
for _k,_v in _DEF.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ═══════════════════════════════════════════════════════════════════════════
# DHAN CLIENT
# ═══════════════════════════════════════════════════════════════════════════
class DhanClient:
    def __init__(self,cid,tok):
        self.cid=cid; self.tok=tok; self._lib=None
        if DHAN_LIB_OK:
            try: self._lib=DhanHQLib(cid,tok)
            except: pass
    def place_order(self,sec_id,segment,txn,qty,order_type="MARKET",price=0.0,product_type="INTRADAY",validity="DAY"):
        if self._lib:
            try:
                tc=self._lib.BUY if txn=="BUY" else self._lib.SELL
                ot=self._lib.MARKET if order_type=="MARKET" else self._lib.LIMIT
                pt={"INTRADAY":self._lib.INTRA,"DELIVERY":self._lib.CNC,"MARGIN":self._lib.MARGIN}.get(product_type,self._lib.INTRA)
                sm={"NSE":"NSE_EQ","BSE":"BSE_EQ","NSE_FNO":"NSE_FNO","BSE_FNO":"BSE_FNO"}
                seg=getattr(self._lib,sm.get(segment,segment),segment)
                return self._lib.place_order(security_id=sec_id,exchange_segment=seg,
                    transaction_type=tc,quantity=qty,order_type=ot,product_type=pt,price=float(price),validity=validity)
            except Exception as e: return{"error":str(e)}
        import requests
        try:
            r=requests.post("https://api.dhan.co/orders",timeout=10,
                headers={"Content-Type":"application/json","access-token":self.tok},
                json={"dhanClientId":self.cid,"transactionType":txn,"exchangeSegment":segment,
                      "productType":product_type,"orderType":order_type,"validity":validity,
                      "securityId":sec_id,"quantity":qty,"price":price})
            return r.json()
        except Exception as e: return{"error":str(e)}
    def fund_limit(self):
        if self._lib:
            try: return self._lib.get_fund_limits()
            except Exception as e: return{"error":str(e)}
        import requests
        try:
            return requests.get("https://api.dhan.co/fundlimit",timeout=10,
                headers={"Content-Type":"application/json","access-token":self.tok}).json()
        except Exception as e: return{"error":str(e)}

# ═══════════════════════════════════════════════════════════════════════════
# FETCH
# ═══════════════════════════════════════════════════════════════════════════
_flock=threading.Lock(); _fts=[0.0]

def fetch_ohlcv(sym,iv,per,md=1.5):
    # Enforce period limits before calling
    allowed=VALID_PERIODS.get(iv,[])
    if allowed and per not in allowed:
        per=allowed[-1]  # clamp to max valid period
    with _flock:
        gap=time.time()-_fts[0]
        if gap<md: time.sleep(md-gap)
        try:
            df=yf.download(sym,interval=iv,period=per,progress=False,auto_adjust=True)
            _fts[0]=time.time()
        except: _fts[0]=time.time(); return None
    if df is None or df.empty: return None
    if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
    df=df.dropna(subset=["Open","High","Low","Close"]); df.index=pd.to_datetime(df.index)
    return df

def fetch_ltp(sym):
    try:
        fi=yf.Ticker(sym).fast_info
        p=fi.get("lastPrice") or fi.get("regularMarketPrice")
        if p and float(p)>0: return float(p)
    except: pass
    return None

def delay_ctx(delay_s,interval):
    cs=POLL_SECS.get(interval,3600); ratio=delay_s/cs if cs>0 else 0
    if delay_s<120: return "🟢 Fresh","#4caf50",f"Data is {delay_s}s old."
    elif ratio<1.0: return f"🟡 {delay_s}s","#ffb300",f"Candle still forming ({int(ratio*100)}%). Normal."
    elif ratio<2.0: return f"🟡 {delay_s//60}m","#ff9800",f"{delay_s//60}min old. Market between sessions."
    else:
        h=delay_s//3600
        return f"🔴 {h}h","#f44336",f"{h}h old. Market likely closed. Signal valid for next open."

# ═══════════════════════════════════════════════════════════════════════════
# STALE SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════
def check_signal_staleness(df: pd.DataFrame, signal_bar_idx: int,
                           signal_dir: str, sl: float, target: float) -> dict:
    """
    Checks bars AFTER the signal bar to see if SL or Target was already hit.
    Returns: {is_stale, reason, bars_elapsed, max_adverse, max_favorable}
    """
    if signal_bar_idx >= len(df)-1:
        return {"is_stale":False,"reason":"Signal is on latest bar","bars_elapsed":0,
                "max_adverse":0,"max_favorable":0,"sl_hit":False,"tgt_hit":False}

    post_bars = df.iloc[signal_bar_idx+1:]
    bars_elapsed = len(post_bars)
    sl_hit = False; tgt_hit = False; sl_hit_bar = None; tgt_hit_bar = None

    for i,(idx,row) in enumerate(post_bars.iterrows()):
        lo,hi = float(row["Low"]),float(row["High"])
        if signal_dir=="BUY":
            if lo<=sl:      sl_hit=True;  sl_hit_bar=i+1;  break
            if hi>=target:  tgt_hit=True; tgt_hit_bar=i+1; break
        else:
            if hi>=sl:      sl_hit=True;  sl_hit_bar=i+1;  break
            if lo<=target:  tgt_hit=True; tgt_hit_bar=i+1; break

    cur_price = float(df["Close"].iloc[-1])

    if signal_dir=="BUY":
        entry_ref = float(post_bars["Open"].iloc[0]) if len(post_bars)>0 else float(df["Close"].iloc[signal_bar_idx])
        max_hi = float(post_bars["High"].max()) if len(post_bars)>0 else entry_ref
        max_lo = float(post_bars["Low"].min())  if len(post_bars)>0 else entry_ref
        max_favorable = (max_hi - entry_ref)/entry_ref*100
        max_adverse   = (entry_ref - max_lo)/entry_ref*100
    else:
        entry_ref = float(post_bars["Open"].iloc[0]) if len(post_bars)>0 else float(df["Close"].iloc[signal_bar_idx])
        max_hi = float(post_bars["High"].max()) if len(post_bars)>0 else entry_ref
        max_lo = float(post_bars["Low"].min())  if len(post_bars)>0 else entry_ref
        max_favorable = (entry_ref - max_lo)/entry_ref*100
        max_adverse   = (max_hi - entry_ref)/entry_ref*100

    is_stale = sl_hit or tgt_hit
    if sl_hit:
        reason = f"❌ SL already hit at bar +{sl_hit_bar} after signal. Entry would be at a loss immediately."
    elif tgt_hit:
        reason = f"✅ Target already reached at bar +{tgt_hit_bar}. Opportunity has passed — entering now is late."
    elif bars_elapsed > 5:
        reason = f"⚠️ Signal is {bars_elapsed} bars old. Entry level may have drifted significantly."
    else:
        reason = f"Signal {bars_elapsed} bar(s) old. Entry still valid."

    return {"is_stale":is_stale,"reason":reason,"bars_elapsed":bars_elapsed,
            "max_adverse":round(max_adverse,2),"max_favorable":round(max_favorable,2),
            "sl_hit":sl_hit,"tgt_hit":tgt_hit,"entry_ref":entry_ref,"cur_price":cur_price}

# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def add_indicators(df):
    df=df.copy(); c=df["Close"]
    df["EMA_9"]=c.ewm(span=9,adjust=False).mean()
    df["EMA_21"]=c.ewm(span=21,adjust=False).mean()
    df["EMA_50"]=c.ewm(span=50,adjust=False).mean()
    df["EMA_200"]=c.ewm(span=200,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean()
    df["RSI"]=100-(100/(1+g/l.replace(0,np.nan)))
    rsi=df["RSI"]; rsi_min=rsi.rolling(14).min(); rsi_max=rsi.rolling(14).max()
    df["StochRSI"]=(rsi-rsi_min)/(rsi_max-rsi_min+1e-10)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    df["MACD"]=e12-e26; df["MACD_Signal"]=df["MACD"].ewm(span=9,adjust=False).mean()
    df["MACD_Hist"]=df["MACD"]-df["MACD_Signal"]
    hl=df["High"]-df["Low"]; hc=(df["High"]-c.shift()).abs(); lc=(df["Low"]-c.shift()).abs()
    df["ATR"]=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    ma20=c.rolling(20).mean(); std20=c.rolling(20).std()
    df["BB_Upper"]=ma20+2*std20; df["BB_Lower"]=ma20-2*std20
    if "Volume" in df.columns:
        df["Vol_MA"]=df["Volume"].rolling(20).mean()
        df["Vol_Ratio"]=df["Volume"]/df["Vol_MA"].replace(0,1)
    else:
        df["Vol_MA"]=0; df["Vol_Ratio"]=1
    return df

def find_pivots(df,depth=5):
    H,L,n=df["High"].values,df["Low"].values,len(df); raw=[]
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
# 8-FACTOR SCORING
# ═══════════════════════════════════════════════════════════════════════════
def score_factors(df_ind, direction, retr, w1):
    n=len(df_ind)
    if n<30: return{"score":0.0,"factors":{},"details":"Insufficient data"}
    cur=float(df_ind["Close"].iloc[-1])
    _f=lambda col,fallback=50: float(df_ind[col].iloc[-1]) if col in df_ind.columns and not df_ind[col].isna().iloc[-1] else fallback
    rsi=_f("RSI",50); srsi=_f("StochRSI",0.5); mh=_f("MACD_Hist",0)
    ema9=_f("EMA_9",cur); ema21=_f("EMA_21",cur); ema50=_f("EMA_50",cur); ema200=_f("EMA_200",cur)
    atr=max(_f("ATR",cur*0.01),1e-6); vr=_f("Vol_Ratio",1.0)
    bbl=_f("BB_Lower",cur*0.98); bbu=_f("BB_Upper",cur*1.02)
    mh_prev=[float(df_ind["MACD_Hist"].iloc[max(0,n-1-j)]) for j in range(1,4)] if "MACD_Hist" in df_ind.columns else [0,0,0]
    factors={}; score=0.0

    if direction=="BUY":
        f1=1.0 if rsi<=35 else 0.85 if rsi<=45 else 0.65 if rsi<=55 else 0.40 if rsi<=65 else 0.10
        factors["RSI"]=f"{rsi:.0f} ({'oversold' if rsi<45 else 'neutral' if rsi<60 else 'overbought'})"; score+=f1*0.18
        f2=1.0 if srsi<=0.20 else 0.75 if srsi<=0.40 else 0.45 if srsi<=0.60 else 0.20
        factors["StochRSI"]=f"{srsi:.2f}"; score+=f2*0.12
        f3=(1.0 if mh>0 and any(h<0 for h in mh_prev[:2]) else 0.80 if mh>0 else 0.65 if mh>mh_prev[0] else 0.25)
        factors["MACD"]=f"{mh:+.3f} ({'bull cross' if f3==1.0 else '↑' if mh>0 else '↓'})"; score+=f3*0.15
        f4=(1.0 if cur>ema9>ema21>ema50 else 0.80 if cur>ema21>ema50 else 0.60 if cur>ema50 else 0.40 if cur>ema200 else 0.15)
        factors["EMA"]=f"{'Up ↑' if f4>0.8 else 'Partial' if f4>0.5 else 'Down ↓'}"; score+=f4*0.15
        f5=(1.0 if abs(retr-0.618)<0.04 else 0.88 if abs(retr-0.500)<0.04 else 0.80 if abs(retr-0.382)<0.04 else 0.72 if 0.50<=retr<=0.786 else 0.55 if 0.382<=retr<=0.886 else 0.30)
        factors["Fib"]=f"{retr:.1%} ({'Golden ✨' if f5>=1.0 else '50%' if f5>=0.88 else 'Valid' if f5>=0.55 else 'Weak'})"; score+=f5*0.18
        f6=(1.0 if cur<=bbl*1.005 else 0.75 if cur<=bbl*1.02 else 0.50 if cur<=(bbl+bbu)/2 else 0.20)
        factors["BB"]=f"{'Near lower' if f6>0.7 else 'Mid' if f6>0.4 else 'Upper'}"; score+=f6*0.10
        f7=1.0 if vr>=1.5 else 0.80 if vr>=1.2 else 0.60 if vr>=0.8 else 0.40
        factors["Volume"]=f"{vr:.1f}×"; score+=f7*0.07
        f8=1.0 if w1/atr>=3 else 0.80 if w1/atr>=2 else 0.60 if w1/atr>=1 else 0.35
        factors["W1/ATR"]=f"{w1/atr:.1f}×"; score+=f8*0.05
    else:
        f1=1.0 if rsi>=65 else 0.85 if rsi>=55 else 0.65 if rsi>=45 else 0.40 if rsi>=35 else 0.10
        factors["RSI"]=f"{rsi:.0f} ({'overbought' if rsi>65 else 'neutral' if rsi>45 else 'oversold'})"; score+=f1*0.18
        f2=1.0 if srsi>=0.80 else 0.75 if srsi>=0.60 else 0.45 if srsi>=0.40 else 0.20
        factors["StochRSI"]=f"{srsi:.2f}"; score+=f2*0.12
        f3=(1.0 if mh<0 and any(h>0 for h in mh_prev[:2]) else 0.80 if mh<0 else 0.65 if mh<mh_prev[0] else 0.25)
        factors["MACD"]=f"{mh:+.3f} ({'bear cross' if f3==1.0 else '↓' if mh<0 else '↑'})"; score+=f3*0.15
        f4=(1.0 if cur<ema9<ema21<ema50 else 0.80 if cur<ema21<ema50 else 0.60 if cur<ema50 else 0.40 if cur<ema200 else 0.15)
        factors["EMA"]=f"{'Down ↓' if f4>0.8 else 'Partial' if f4>0.5 else 'Up ↑'}"; score+=f4*0.15
        f5=(1.0 if abs(retr-0.618)<0.04 else 0.88 if abs(retr-0.500)<0.04 else 0.80 if abs(retr-0.382)<0.04 else 0.72 if 0.50<=retr<=0.786 else 0.55 if 0.382<=retr<=0.886 else 0.30)
        factors["Fib"]=f"{retr:.1%}"; score+=f5*0.18
        f6=(1.0 if cur>=bbu*0.995 else 0.75 if cur>=bbu*0.98 else 0.50 if cur>=(bbl+bbu)/2 else 0.20)
        factors["BB"]=f"{'Near upper' if f6>0.7 else 'Mid' if f6>0.4 else 'Lower'}"; score+=f6*0.10
        f7=1.0 if vr>=1.5 else 0.80 if vr>=1.2 else 0.60 if vr>=0.8 else 0.40
        factors["Volume"]=f"{vr:.1f}×"; score+=f7*0.07
        f8=1.0 if w1/atr>=3 else 0.80 if w1/atr>=2 else 0.60 if w1/atr>=1 else 0.35
        factors["W1/ATR"]=f"{w1/atr:.1f}×"; score+=f8*0.05

    return{"score":min(score,1.0),"factors":factors}

# ═══════════════════════════════════════════════════════════════════════════
# CORE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════
def _blank(r=""): return{"signal":"HOLD","entry_price":None,"sl":None,"target":None,
    "confidence":0.0,"mf_score":0.0,"reason":r or "No pattern","pattern":"—",
    "wave_pivots":None,"wave1_len":0.0,"retracement":0.0,"factors":{},"rr":0.0,
    "signal_bar_idx":None}

def _ctgt(tt,e,d,w1,risk,ctp=100.0):
    s=1 if d=="BUY" else -1
    if tt in("wave_auto","fib_1618"): return e+s*w1*1.618
    if tt=="fib_2618": return e+s*w1*2.618
    if tt in(TGT_PTS,TGT_TRAIL): return e+s*ctp
    if tt==TGT_SIGREV: return e+s*w1*1.618
    if isinstance(tt,(int,float)): return e+s*risk*float(tt)
    return e+s*risk*2.0

def ew_signal(df,depth=5,sl_type="wave_auto",tgt_type="wave_auto",
              csl=50.0,ctgt=100.0,min_conf=MIN_CONFIDENCE):
    n=len(df)
    if n<max(40,depth*5): return _blank("Need more bars")
    pivots=find_pivots(df,depth)
    if len(pivots)<4: return _blank("Not enough pivots — try smaller depth")
    df_ind=add_indicators(df); cur=float(df["Close"].iloc[-1])
    atr=float(df_ind["ATR"].iloc[-1]) if "ATR" in df_ind.columns and not df_ind["ATR"].isna().iloc[-1] else cur*0.01
    best,bs=_blank(),0.0

    for i in range(len(pivots)-2):
        p0,p1,p2=pivots[i],pivots[i+1],pivots[i+2]; bars=n-1-p2[0]
        # BUY
        if p0[2]=="L" and p1[2]=="H" and p2[2]=="L":
            w1=p1[1]-p0[1]
            if w1<=0: continue
            r=(p1[1]-p2[1])/w1
            if not(0.236<=r<=0.886 and p2[1]>p0[1] and bars<=depth*5): continue
            if i+3<len(pivots) and pivots[i+3][2]=="H":
                w3=pivots[i+3][1]-p2[1]
                if w3<w1*0.90: continue
            ew=0.55
            if abs(r-0.618)<0.04: ew=0.85
            elif abs(r-0.500)<0.04: ew=0.75
            elif 0.50<=r<=0.786: ew=0.72
            elif 0.382<=r<=0.618: ew=0.65
            if i+3<len(pivots) and pivots[i+3][2]=="H":
                w3v=pivots[i+3][1]-p2[1]
                if w3v>w1: ew=min(ew+0.08,0.95)
                if abs(w3v/w1-1.618)<0.15: ew=min(ew+0.06,0.98)
            mf=score_factors(df_ind,"BUY",r,w1); mfs=mf["score"]
            combined=ew*0.55+mfs*0.45
            e=cur
            sl_=p2[1]*0.998 if sl_type in(SL_WAVE,SL_TRAIL) else e-csl if sl_type==SL_PTS else e*(1-0.05) if sl_type==SL_SIGREV else e*(1-float(sl_type))
            rk=e-sl_
            if rk<=0: continue
            tgt_=_ctgt(tgt_type,e,"BUY",w1,rk,ctgt)
            if tgt_<=e: continue
            rr=(tgt_-e)/rk
            if rr<1.2: continue
            if combined<=bs: continue
            bs=combined; best={
                "signal":"BUY","entry_price":e,"sl":sl_,"target":tgt_,
                "confidence":round(combined,3),"mf_score":round(mfs,3),"ew_conf":round(ew,3),
                "retracement":r,"reason":f"W2 bottom {r:.1%} | Score {combined:.0%} | R:R 1:{rr:.1f}",
                "pattern":f"W2 Bottom ({r:.1%})","wave_pivots":[p0,p1,p2],"wave1_len":w1,
                "factors":mf["factors"],"rr":round(rr,2),"signal_bar_idx":p2[0]}
        # SELL
        elif p0[2]=="H" and p1[2]=="L" and p2[2]=="H":
            w1=p0[1]-p1[1]
            if w1<=0: continue
            r=(p2[1]-p1[1])/w1
            if not(0.236<=r<=0.886 and p2[1]<p0[1] and bars<=depth*5): continue
            if i+3<len(pivots) and pivots[i+3][2]=="L":
                w3=p2[1]-pivots[i+3][1]
                if w3<w1*0.90: continue
            ew=0.55
            if abs(r-0.618)<0.04: ew=0.85
            elif abs(r-0.500)<0.04: ew=0.75
            elif 0.50<=r<=0.786: ew=0.72
            elif 0.382<=r<=0.618: ew=0.65
            if i+3<len(pivots) and pivots[i+3][2]=="L":
                w3v=p2[1]-pivots[i+3][1]
                if w3v>w1: ew=min(ew+0.08,0.95)
                if abs(w3v/w1-1.618)<0.15: ew=min(ew+0.06,0.98)
            mf=score_factors(df_ind,"SELL",r,w1); mfs=mf["score"]
            combined=ew*0.55+mfs*0.45
            e=cur
            sl_=p2[1]*1.002 if sl_type in(SL_WAVE,SL_TRAIL) else e+csl if sl_type==SL_PTS else e*(1+0.05) if sl_type==SL_SIGREV else e*(1+float(sl_type))
            rk=sl_-e
            if rk<=0: continue
            tgt_=_ctgt(tgt_type,e,"SELL",w1,rk,ctgt)
            if tgt_>=e: continue
            rr=(e-tgt_)/rk
            if rr<1.2: continue
            if combined<=bs: continue
            bs=combined; best={
                "signal":"SELL","entry_price":e,"sl":sl_,"target":tgt_,
                "confidence":round(combined,3),"mf_score":round(mfs,3),"ew_conf":round(ew,3),
                "retracement":r,"reason":f"W2 top {r:.1%} | Score {combined:.0%} | R:R 1:{rr:.1f}",
                "pattern":f"W2 Top ({r:.1%})","wave_pivots":[p0,p1,p2],"wave1_len":w1,
                "factors":mf["factors"],"rr":round(rr,2),"signal_bar_idx":p2[0]}

    if best["signal"]!="HOLD" and best["confidence"]<min_conf:
        return _blank(f"Signal found ({best['pattern']}) but score {best['confidence']:.0%} < min {min_conf:.0%}. "
                     f"Waiting for stronger confluence. {best.get('reason','')}")
    return best

def update_trailing_sl(pos,ltp,sl_type,trail_pts):
    if sl_type!=SL_TRAIL or not pos: return pos
    pos=pos.copy()
    if pos["type"]=="BUY":
        ns=ltp-trail_pts
        if ns>pos["sl"]: pos["sl"]=ns
    else:
        ns=ltp+trail_pts
        if ns<pos["sl"]: pos["sl"]=ns
    return pos

def update_trailing_tgt(pos,ltp,tgt_type,trail_pts):
    if tgt_type!=TGT_TRAIL or not pos: return pos
    pos=pos.copy()
    if pos["type"]=="BUY":
        nt=ltp+trail_pts
        if nt>pos.get("target_display",pos["target"]): pos["target_display"]=nt
    else:
        nt=ltp-trail_pts
        if nt<pos.get("target_display",pos["target"]): pos["target_display"]=nt
    return pos

def analyze_wave_state(df,pivots,sig):
    cur=float(df["Close"].iloc[-1]) if len(df) else 0
    if not pivots: return{"current_wave":"Collecting data…","next_wave":"—","direction":"NEUTRAL","fib_levels":{},"action":"Need more bars.","auto_action":"Wait"}
    if sig["signal"]=="BUY":
        wp=sig.get("wave_pivots",[None,None,None]); p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p1[1]-p0[1]) if p0 and p1 else cur*0.02; w2l=p2[1] if p2 else cur*0.99
        fibs={"W3 (1.618×W1)":round(w2l+w1*1.618,2),"W3 (1.0×W1)":round(w2l+w1,2),"W3 (2.618×W1)":round(w2l+w1*2.618,2)}
        return{"current_wave":"✅ Wave-2 Bottom — ENTRY ZONE","next_wave":"Wave-3 UP ↑","direction":"BULLISH","fib_levels":fibs,
               "action":f"BUY at market. SL: {w2l:.2f}\nW3 targets: {fibs['W3 (1.618×W1)']:.2f}·{fibs['W3 (2.618×W1)']:.2f}\nAuto-managed — no action needed.","auto_action":"BUY"}
    if sig["signal"]=="SELL":
        wp=sig.get("wave_pivots",[None,None,None]); p0,p1,p2=wp[0],wp[1],wp[2]
        w1=(p0[1]-p1[1]) if p0 and p1 else cur*0.02; w2h=p2[1] if p2 else cur*1.01
        fibs={"W3 (1.618×W1)":round(w2h-w1*1.618,2),"W3 (1.0×W1)":round(w2h-w1,2),"W3 (2.618×W1)":round(w2h-w1*2.618,2)}
        return{"current_wave":"🔴 Wave-2 Top — ENTRY ZONE","next_wave":"Wave-3 DOWN ↓","direction":"BEARISH","fib_levels":fibs,
               "action":f"SELL at market. SL: {w2h:.2f}\nW3 targets: {fibs['W3 (1.618×W1)']:.2f}·{fibs['W3 (2.618×W1)']:.2f}\nAuto-managed — no action needed.","auto_action":"SELL"}
    lp=pivots[-1]; rp=0.0
    if len(pivots)>=3:
        pa,pb,pc=pivots[-3],pivots[-2],pivots[-1]
        if abs(pb[1]-pa[1])>0: rp=abs(pc[1]-pb[1])/abs(pb[1]-pa[1])*100
    return{"current_wave":f"{'High' if lp[2]=='H' else 'Low'} @ {lp[1]:.2f}",
           "next_wave":f"Waiting for W2 ({rp:.1f}% retrace, need 38–79%)" if rp else "Waiting for Wave-2",
           "direction":"NEUTRAL","fib_levels":{},
           "action":f"HOLD. {sig.get('reason','')}\nSystem scans every candle. Nothing to do.","auto_action":"Wait"}

# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df,depth=5,sl_type="wave_auto",tgt_type="wave_auto",
                 capital=100_000.0,csl=50.0,ctgt=100.0,trail_sl=50.0,min_conf=MIN_CONFIDENCE):
    MB=max(40,depth*5)
    if len(df)<MB+10: return{"error":f"Need ≥{MB+10} bars.","equity_curve":[capital]}
    trades,equity_curve=[],[capital]; equity,pos=capital,None
    for i in range(MB,len(df)-1):
        bdf=df.iloc[:i+1]; nb=df.iloc[i+1]
        hi_i,lo_i,cl_i=float(df.iloc[i]["High"]),float(df.iloc[i]["Low"]),float(df.iloc[i]["Close"])
        tsig=None
        if pos:
            if sl_type==SL_TRAIL: pos=update_trailing_sl(pos,cl_i,sl_type,trail_sl)
            if sl_type==SL_SIGREV or tgt_type==TGT_SIGREV:
                tsig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt,min_conf)
            ep_,er_=None,None
            if pos["type"]=="BUY":
                if lo_i<=pos["sl"]:       ep_,er_=pos["sl"],   "SL (Low≤SL)"
                elif hi_i>=pos["target"]: ep_,er_=pos["target"],"Target (High≥Target)"
                elif tsig and tsig["signal"]=="SELL": ep_,er_=cl_i,"Signal Reverse"
            else:
                if hi_i>=pos["sl"]:       ep_,er_=pos["sl"],   "SL (High≥SL)"
                elif lo_i<=pos["target"]: ep_,er_=pos["target"],"Target (Low≤Target)"
                elif tsig and tsig["signal"]=="BUY": ep_,er_=cl_i,"Signal Reverse"
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
                    "Confidence":round(pos["conf"],2),"MF Score":round(pos.get("mf",0),2)})
                pos=None
        if pos is None:
            sig=ew_signal(bdf,depth,sl_type,tgt_type,csl,ctgt,min_conf)
            if sig["signal"] in("BUY","SELL"):
                ep=float(nb["Open"]); w1=sig.get("wave1_len",ep*0.02) or ep*0.02
                atr_=float(add_indicators(bdf)["ATR"].iloc[-1]) if len(bdf)>20 else ep*0.01
                if sl_type in(SL_WAVE,SL_TRAIL): sl_=sig["sl"]
                elif sl_type==SL_PTS: sl_=ep-csl if sig["signal"]=="BUY" else ep+csl
                elif sl_type==SL_SIGREV: sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
                else: sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" else ep*(1+float(sl_type))
                rk=abs(ep-sl_)
                if rk<=0: continue
                tgt_=_ctgt(tgt_type,ep,sig["signal"],w1,rk,ctgt)
                if sig["signal"]=="BUY" and tgt_<=ep: continue
                if sig["signal"]=="SELL" and tgt_>=ep: continue
                rr2=(tgt_-ep)/rk if sig["signal"]=="BUY" else (ep-tgt_)/rk
                if rr2<1.2: continue
                qty=max(1,int(equity*0.95/ep))
                pos={"type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,
                     "entry_bar":i+1,"entry_time":df.index[i+1],"qty":qty,
                     "conf":sig["confidence"],"mf":sig.get("mf_score",0)}
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
            "Confidence":round(pos["conf"],2),"MF Score":round(pos.get("mf",0),2)})
    if not trades: return{"error":"No trades. Try smaller depth, longer period, or lower confidence.","equity_curve":equity_curve}
    tdf=pd.DataFrame(trades)
    wins=tdf[tdf["PnL Rs"]>0]; loss=tdf[tdf["PnL Rs"]<=0]; ntot=len(tdf)
    wr=len(wins)/ntot*100 if ntot else 0
    pf=abs(wins["PnL Rs"].sum()/loss["PnL Rs"].sum()) if len(loss) and loss["PnL Rs"].sum()!=0 else 9999.0
    ea=np.array(equity_curve); pk=np.maximum.accumulate(ea)
    mdd=float(((ea-pk)/pk*100).min())
    rets=tdf["PnL %"].values
    sharpe=float(rets.mean()/rets.std()*np.sqrt(252)) if len(rets)>1 and rets.std()!=0 else 0.0
    tdf["SL Verified"]=tdf.apply(lambda r:"✅" if(
        (r["Type"]=="BUY" and "SL" in r["Exit Reason"] and r["Exit Bar Low"]<=r["SL"]) or
        (r["Type"]=="SELL" and "SL" in r["Exit Reason"] and r["Exit Bar High"]>=r["SL"]) or
        "SL" not in r["Exit Reason"]) else "⚠️",axis=1)
    tdf["TGT Verified"]=tdf.apply(lambda r:"✅" if(
        (r["Type"]=="BUY" and "Target" in r["Exit Reason"] and r["Exit Bar High"]>=r["Target"]) or
        (r["Type"]=="SELL" and "Target" in r["Exit Reason"] and r["Exit Bar Low"]<=r["Target"]) or
        "Target" not in r["Exit Reason"]) else "⚠️",axis=1)
    return{"trades":tdf,"equity_curve":equity_curve,
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
# OPTIMIZATION  (Quick / Balanced / Deep modes + min-trades filter)
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization(df,capital=100_000.0,csl=50.0,ctgt=100.0,trail_sl=50.0,
                     opt_mode="⚖️ Balanced (24 combos)",min_conf=MIN_CONFIDENCE,
                     min_trades=MIN_TRADES_FILTER,prog_obj=None):
    cfg=OPT_MODES.get(opt_mode,OPT_MODES["⚖️ Balanced (24 combos)"])
    combos=list(itertools.product(cfg["depths"],cfg["sl_opts"],cfg["tgt_opts"]))
    rows=[]
    for idx,(dep,sl,tgt) in enumerate(combos):
        try:
            r=run_backtest(df,depth=dep,sl_type=sl,tgt_type=tgt,capital=capital,
                           csl=csl,ctgt=ctgt,trail_sl=trail_sl,min_conf=min_conf)
            if "metrics" in r and r["metrics"]["Total Trades"]>=min_trades:
                m=r["metrics"]
                rows.append({"Depth":dep,"SL":str(sl),"Target":str(tgt),
                    "Trades":m["Total Trades"],"Win %":m["Win Rate %"],
                    "Return %":m["Total Return %"],"PF":m["Profit Factor"],
                    "Max DD %":m["Max Drawdown %"],"Sharpe":m["Sharpe Ratio"]})
        except: pass
        if prog_obj:
            prog_obj.progress((idx+1)/len(combos),text=f"Opt {idx+1}/{len(combos)} [{opt_mode.split('(')[0].strip()}]…")
    if prog_obj: prog_obj.empty()
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

def apply_best_cfg(entry:dict):
    """Apply all best-config fields to session state so sidebar/backtest/live all use it."""
    ss=st.session_state
    ss.applied_depth   = int(entry.get("depth",5))
    ss.applied_sl_lbl  = sl_lbl(entry.get("sl","wave_auto"))
    ss.applied_tgt_lbl = tgt_lbl(entry.get("target","wave_auto"))
    ss.applied_interval= entry.get("tf","1d")
    ss.applied_period  = entry.get("period","1y")
    ss.best_cfg_applied= True

# ═══════════════════════════════════════════════════════════════════════════
# FULL AUTO MODE
# ═══════════════════════════════════════════════════════════════════════════
def run_full_auto(symbol,capital,csl,ctgt,opt_mode="⚖️ Balanced (24 combos)"):
    ss=st.session_state
    def alog(m): ss.auto_log.append(f"[{now_ist().strftime('%H:%M:%S IST')}] {m}"); ss.auto_log=ss.auto_log[-200:]
    alog("🚀 Full Auto started")
    ss.auto_status="Scanning TF/period combinations…"
    all_results=[]; best_overall=None; best_score_overall=-999
    prog=st.progress(0,text="Full Auto: starting…")
    valid_grid=[(tf,per) for tf,per in AUTO_SCAN_GRID if per in VALID_PERIODS.get(tf,[])]
    for gi,(tf,per) in enumerate(valid_grid):
        nm=f"{tf}·{per}"
        alog(f"📡 Fetching {nm}…"); ss.auto_status=f"Fetching {nm} ({gi+1}/{len(valid_grid)})…"
        prog.progress((gi/len(valid_grid))*0.3,text=f"Fetching {nm}…")
        df=fetch_ohlcv(symbol,tf,per,md=1.5)
        if df is None or len(df)<55:
            alog(f"⚠️ {nm}: insufficient data — skipping"); continue
        try:
            sig=ew_signal(df.iloc[:-1],5,"wave_auto","wave_auto",csl,ctgt)
            alog(f"✅ {nm}: {len(df)} bars | {sig['signal']} ({sig.get('confidence',0):.0%})")
        except: sig=_blank("error"); alog(f"⚠️ {nm}: signal error")
        prog_fraction=(gi/len(valid_grid)*0.3)+0.3/len(valid_grid)
        inner_prog=st.empty()
        inner_prog_bar=inner_prog.progress(0,text=f"Optimizing {nm}…")
        try:
            odf=run_optimization(df,capital,csl,ctgt,50.0,opt_mode,MIN_CONFIDENCE,MIN_TRADES_FILTER,inner_prog_bar)
            inner_prog.empty()
            if odf.empty:
                alog(f"⚠️ {nm}: no valid optimization results"); continue
            br_=odf.iloc[0]; bonus=sig.get("confidence",0)*20 if sig["signal"] in("BUY","SELL") and sig.get("confidence",0)>=MIN_CONFIDENCE else 0
            weighted=br_["Score"]+bonus
            entry={"tf":tf,"period":per,"tf_name":nm,
                   "depth":int(br_["Depth"]),"sl":br_["SL"],"target":br_["Target"],
                   "trades":br_["Trades"],"win_pct":br_["Win %"],
                   "return_pct":br_["Return %"],"pf":br_["PF"],
                   "max_dd":br_["Max DD %"],"sharpe":br_["Sharpe"],"score":br_["Score"],
                   "weighted_score":weighted,"current_signal":sig["signal"],
                   "current_conf":sig.get("confidence",0),"opt_df":odf}
            all_results.append(entry)
            alog(f"📊 {nm}: score={br_['Score']:.2f} win={br_['Win %']:.0f}% return={br_['Return %']:.1f}%")
            if weighted>best_score_overall:
                best_score_overall=weighted; best_overall=entry.copy()
                alog(f"⭐ New best: {nm}")
        except Exception as e:
            inner_prog.empty(); alog(f"❌ {nm}: {e}")
        prog.progress(min((gi+1)/len(valid_grid)*0.9,0.99),text=f"Done {gi+1}/{len(valid_grid)}")

    prog.progress(1.0,text="Full Auto: done!"); time.sleep(0.3); prog.empty()
    if best_overall is None:
        ss.auto_status="❌ No valid results found. Try different symbol."; ss.auto_running=False; return
    apply_best_cfg(best_overall)   # ← single function call, applies everything
    ss.auto_best_cfg=best_overall; ss.auto_all_results=all_results
    ss.auto_status=(f"✅ Best: {best_overall['tf_name']} | Win {best_overall['win_pct']:.0f}% | "
                    f"Return {best_overall['return_pct']:.1f}% | Depth {best_overall['depth']}")
    alog(f"✅ Applied: {best_overall['tf']} {best_overall['period']} D={best_overall['depth']} "
         f"SL={best_overall['sl']} TGT={best_overall['target']}")
    alog(f"📌 Win={best_overall['win_pct']:.0f}% Return={best_overall['return_pct']:.1f}% Sharpe={best_overall['sharpe']:.2f}")
    ss.auto_running=False

# ═══════════════════════════════════════════════════════════════════════════
# LIVE ENGINE TICK
# ═══════════════════════════════════════════════════════════════════════════
def live_engine_tick(symbol,interval,period,depth,sl_type,tgt_type,csl,ctgt,
                     trail_sl,trail_tgt,dhan_on,dhan_client,trade_mode,entry_ot,exit_ot,
                     product_type,segment,ce_sec_id,pe_sec_id,live_qty,
                     stale_check,min_conf)->bool:
    ss=st.session_state; now_ts=time.time()
    ltp_new=fetch_ltp(symbol)
    if ltp_new and ltp_new>0:
        ss.live_ltp_prev=ss.live_ltp; ss.live_ltp=ltp_new
        ss.live_ltp_ts=now_ist().strftime("%H:%M:%S IST")
    ltp=ss.live_ltp
    pos=ss.live_position
    if pos and ltp:
        pos["unreal_pnl"]=(ltp-pos["entry"])*pos["qty"] if pos["type"]=="BUY" else (pos["entry"]-ltp)*pos["qty"]
        pos["dist_sl"]=abs(ltp-pos["sl"])/ltp*100
        pos["dist_tgt"]=abs(pos.get("target_display",pos["target"])-ltp)/ltp*100
        if sl_type==SL_TRAIL: pos=update_trailing_sl(pos,ltp,sl_type,trail_sl)
        if tgt_type==TGT_TRAIL: pos=update_trailing_tgt(pos,ltp,tgt_type,trail_tgt)
        ss.live_position=pos
    cs=POLL_SECS.get(interval,3600); ts2=now_ts-ss.last_fetch_wall
    if ts2<cs and ss.last_fetch_wall>0:
        ss.live_next_check_ist=(now_ist()+timedelta(seconds=int(cs-ts2))).strftime("%H:%M:%S IST")
        return False
    ss.live_phase="scanning"; ss.last_fetch_wall=now_ts
    def _log(m,lv="INFO"):
        ss.live_log.append(f"[{now_ist().strftime('%H:%M:%S IST')}][{lv}] {m}")
        ss.live_log=ss.live_log[-150:]
    df=fetch_ohlcv(symbol,interval,period,md=1.5)
    if df is None or len(df)<45:
        _log("⚠️  Insufficient data","WARN"); ss.live_no_pos_reason="⚠️ No data."; ss.live_phase="no_signal"; return False
    ohlcv_ltp=float(df["Close"].iloc[-1])
    if not ss.live_ltp or ss.live_ltp<=0: ss.live_ltp=ohlcv_ltp
    lbdt=df.index[-2] if len(df)>=2 else df.index[-1]
    ss.live_last_candle_ist=fmt_ist(lbdt)
    try:
        lbdt2=lbdt.to_pydatetime()
        if lbdt2.tzinfo is None: lbdt2=lbdt2.replace(tzinfo=timezone.utc)
        ss.live_delay_s=int((now_ist()-lbdt2.astimezone(IST)).total_seconds())
    except: ss.live_delay_s=0
    _,_,ss.live_delay_ctx=delay_ctx(ss.live_delay_s,interval)
    df_closed=df.iloc[:-1]; latest_ts=str(df_closed.index[-1])
    pivots=find_pivots(df_closed,depth)
    sig=ew_signal(df_closed,depth,sl_type,tgt_type,csl,ctgt,min_conf)
    ss.live_last_sig=sig; ss.live_wave_state=analyze_wave_state(df_closed,pivots,sig)
    ss._scan_df=df; ss._scan_sig=sig
    nxt=(now_ist()+timedelta(seconds=cs)).strftime("%H:%M:%S IST"); ss.live_next_check_ist=nxt
    if ss.last_bar_ts==latest_ts:
        _log(f"⏭  Same bar {latest_ts[-10:]} next {nxt}")
        if ss.live_position:
            _manage_pos(ss.live_ltp or ohlcv_ltp,sig,sl_type,tgt_type,dhan_on,dhan_client,
                        trade_mode,exit_ot,segment,ce_sec_id,pe_sec_id,live_qty,_log)
        ss.live_phase=f"pos_{ss.live_position['type'].lower()}" if ss.live_position else "no_signal"
        return False
    ss.last_bar_ts=latest_ts; ltp_now=ss.live_ltp or ohlcv_ltp
    _log(f"🕯 Bar {latest_ts[-10:]} | LTP {ltp_now:.2f} | {sig['signal']} ({sig.get('confidence',0):.0%})")
    _manage_pos(ltp_now,sig,sl_type,tgt_type,dhan_on,dhan_client,
                trade_mode,exit_ot,segment,ce_sec_id,pe_sec_id,live_qty,_log)
    pos=ss.live_position
    if pos is None and sig["signal"] in("BUY","SELL"):
        ep=ltp_now; w1=sig.get("wave1_len",ep*0.02) or ep*0.02
        df_ind_=add_indicators(df_closed)
        atr_=float(df_ind_["ATR"].iloc[-1]) if "ATR" in df_ind_.columns and not df_ind_["ATR"].isna().iloc[-1] else ep*0.01
        if sl_type in(SL_WAVE,SL_TRAIL): sl_=sig["sl"]
        elif sl_type==SL_PTS: sl_=ep-csl if sig["signal"]=="BUY" else ep+csl
        elif sl_type==SL_SIGREV: sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
        else: sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" else ep*(1+float(sl_type))
        rk=abs(ep-sl_)
        if rk<=0:
            ss.live_no_pos_reason=f"⚠️ Risk=0 (SL {sl_:.2f} ≈ entry {ep:.2f})."; ss.live_phase="no_signal"; return True
        tgt_=_ctgt(tgt_type,ep,sig["signal"],w1,rk,ctgt)
        if (sig["signal"]=="BUY" and tgt_<=ep) or (sig["signal"]=="SELL" and tgt_>=ep):
            ss.live_no_pos_reason=f"⚠️ Invalid target ({tgt_:.2f})."; ss.live_phase="no_signal"; return True
        rr_=(tgt_-ep)/rk if sig["signal"]=="BUY" else (ep-tgt_)/rk
        if rr_<1.2:
            ss.live_no_pos_reason=f"⚠️ R:R {rr_:.2f}:1 below minimum 1.2:1."; ss.live_phase="no_signal"; return True

        # ── Stale signal check ────────────────────────────────────────────
        sig_bar_idx=sig.get("signal_bar_idx",len(df_closed)-1)
        stale_info=check_signal_staleness(df_closed,sig_bar_idx,sig["signal"],sl_,tgt_)
        # Always store snapshot for display
        ss.signal_snapshot={
            "bar_ts":latest_ts,"time_ist":fmt_ist(df_closed.index[sig_bar_idx]),
            "ltp_at_signal":float(df_closed["Close"].iloc[sig_bar_idx]),
            "sl_at_signal":sl_,"tgt_at_signal":tgt_,"direction":sig["signal"],
            "cur_ltp":ltp_now,"cur_sl":sl_,"cur_tgt":tgt_,
            "stale_info":stale_info,"pattern":sig["pattern"],"conf":sig["confidence"],
        }
        if stale_check and stale_info["is_stale"]:
            ss.live_no_pos_reason=(f"🚫 STALE SIGNAL BLOCKED.\n{stale_info['reason']}\n"
                                   f"Bars elapsed since signal: {stale_info['bars_elapsed']}\n"
                                   f"Max adverse move: {stale_info['max_adverse']:.2f}%\n"
                                   f"System will wait for a fresh signal.")
            ss.live_phase="no_signal"
            _log(f"🚫 Stale signal blocked: {stale_info['reason']}","WARN")
            return True

        ei=now_ist().strftime("%d-%b %H:%M:%S IST")
        limit_price=sig["entry_price"] if entry_ot=="LIMIT" else 0.0
        ss.live_position={"type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,"target_display":tgt_,
            "qty":live_qty,"entry_ist":ei,"symbol":symbol,"pattern":sig["pattern"],
            "confidence":sig["confidence"],"mf_score":sig.get("mf_score",0),
            "rr":rr_,"unreal_pnl":0.0,"dist_sl":rk/ep*100,"dist_tgt":abs(tgt_-ep)/ep*100}
        ss.live_phase=f"pos_{sig['signal'].lower()}"; ss.live_no_pos_reason=""
        ss.live_signals.append({"Time (IST)":ei,"Bar":fmt_ist(df_closed.index[-1]),
            "TF":interval,"Period":period,"Signal":sig["signal"],
            "Entry":round(ep,2),"SL":round(sl_,2),"Target":round(tgt_,2),
            "Conf":f"{sig['confidence']:.0%}","MF":f"{sig.get('mf_score',0):.0%}",
            "R:R":f"1:{rr_:.1f}","Pattern":sig["pattern"]})
        if dhan_on and dhan_client:
            sec=ce_sec_id if(trade_mode=="options" and sig["signal"]=="BUY") else pe_sec_id if trade_mode=="options" else ""
            r2=dhan_client.place_order(sec or "",segment,sig["signal"],live_qty,
                order_type=entry_ot,price=limit_price,product_type=product_type)
            _log(f"📤 Dhan entry: {r2}")
        em="🟢" if sig["signal"]=="BUY" else "🔴"
        _log(f"{em} ENTERED {sig['signal']} @ {ep:.2f} | SL {sl_:.2f} | T {tgt_:.2f} | Conf {sig['confidence']:.0%}")
    elif pos is None:
        ss.live_phase="no_signal"
        ss.live_no_pos_reason=f"📊 HOLD. {sig.get('reason','')}\nNext check: {nxt}"
    return True

def _manage_pos(ltp,sig,sl_type,tgt_type,dhan_on,dhan_client,trade_mode,exit_ot,segment,ce_sec_id,pe_sec_id,qty_,log_fn):
    ss=st.session_state; pos=ss.live_position
    if not pos or not ltp: return
    hit_p,hit_r=None,None; pt=pos["type"]; sl_=pos["sl"]; tgt_=pos["target"]
    if pt=="BUY":
        if ltp<=sl_:       hit_p,hit_r=sl_,  "SL Hit (≤SL)"
        elif ltp>=tgt_ and tgt_type!=TGT_TRAIL: hit_p,hit_r=tgt_,"Target Hit (≥Target)"
        elif sig["signal"]=="SELL": hit_p,hit_r=ltp,"Signal Reversed→SELL"
    else:
        if ltp>=sl_:       hit_p,hit_r=sl_,  "SL Hit (≥SL)"
        elif ltp<=tgt_ and tgt_type!=TGT_TRAIL: hit_p,hit_r=tgt_,"Target Hit (≤Target)"
        elif sig["signal"]=="BUY": hit_p,hit_r=ltp,"Signal Reversed→BUY"
    if hit_p is None: return
    qty=pos["qty"]; pnl=(hit_p-pos["entry"])*qty if pt=="BUY" else (pos["entry"]-hit_p)*qty
    ss.live_pnl+=pnl  # ONLY place live_pnl is modified
    ss.live_trades.append({"Time (IST)":now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Symbol":pos["symbol"],"TF":ss.get("_live_interval","—"),
        "Type":pt,"Entry":round(pos["entry"],2),"Exit":round(hit_p,2),
        "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
        "Qty":qty,"PnL Rs":round(pnl,2),"Reason":hit_r})
    if dhan_on and dhan_client:
        xt="SELL" if pt=="BUY" else "BUY"
        sec=ce_sec_id if(trade_mode=="options" and xt=="BUY") else pe_sec_id if trade_mode=="options" else ""
        xp=hit_p if exit_ot=="LIMIT" else 0.0
        log_fn(f"📤 Dhan exit: {dhan_client.place_order(sec or '',segment,xt,qty,order_type=exit_ot,price=xp)}")
    em="✅" if "Target" in hit_r else("🔄" if "Reversed" in hit_r else "❌")
    log_fn(f"{em} {pt} CLOSED @ {hit_p:.2f} | {hit_r} | Rs{pnl:+.2f}")
    ss.live_position=None; ss.signal_snapshot=None
    ss.live_phase="reversing" if "Reversed" in hit_r else "no_signal"
    if "Reversed" not in hit_r: ss.live_no_pos_reason=f"Last: {hit_r} | Rs{pnl:+.2f}"

# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def chart_waves(df,pivots,sig=None,trades=None,symbol="",tf_label=""):
    sig=sig or _blank(); df_ind=add_indicators(df)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.60,0.20,0.20],
                      vertical_spacing=0.02,subplot_titles=("","Volume","RSI/StochRSI"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name="Price",
        increasing=dict(line=dict(color="#26a69a"),fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"),fillcolor="#ef5350")),row=1,col=1)
    for col,clr,nm in[("EMA_9","#ffeb3b","EMA9"),("EMA_21","#ffb300","EMA21"),("EMA_50","#ab47bc","EMA50"),("EMA_200","#ef5350","EMA200")]:
        if col in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind[col],mode="lines",line=dict(color=clr,width=1.0),name=nm,opacity=0.65),row=1,col=1)
    if "Volume" in df.columns:
        vc=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,opacity=0.4,name="Vol",showlegend=False),row=2,col=1)
    if "RSI" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind["RSI"],mode="lines",line=dict(color="#00e5ff",width=1.5),name="RSI",showlegend=False),row=3,col=1)
        fig.add_hline(y=70,line=dict(dash="dot",color="#ef5350",width=1),row=3,col=1)
        fig.add_hline(y=30,line=dict(dash="dot",color="#4caf50",width=1),row=3,col=1)
    if "StochRSI" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind["StochRSI"]*100,mode="lines",
            line=dict(color="#ff9800",width=1.0,dash="dot"),name="StochRSI×100",showlegend=False),row=3,col=1)
    vp=[p for p in pivots if p[0]<len(df)]
    if vp:
        fig.add_trace(go.Scatter(x=[df.index[p[0]] for p in vp],y=[p[1] for p in vp],mode="lines+markers",
            line=dict(color="rgba(255,180,0,.5)",width=1.5,dash="dot"),
            marker=dict(size=7,color=["#4caf50" if p[2]=="L" else "#f44336" for p in vp],
                        symbol=["triangle-up" if p[2]=="L" else "triangle-down" for p in vp]),name="ZigZag"),row=1,col=1)
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
        fig.add_trace(go.Scatter(x=[df.index[-1]],y=[df["Close"].iloc[-1]],mode="markers",
            marker=dict(size=20,color=sc,symbol="triangle-up" if sig["signal"]=="BUY" else "triangle-down",
                        line=dict(color="white",width=1.5)),name=f"▶ {sig['signal']}"),row=1,col=1)
        if sig.get("sl"):
            fig.add_hline(y=sig["sl"],line=dict(dash="dash",color="#ff7043",width=1.5),annotation_text="  SL",annotation_position="right",row=1,col=1)
        if sig.get("target"):
            fig.add_hline(y=sig["target"],line=dict(dash="dash",color="#66bb6a",width=1.5),annotation_text="  Target",annotation_position="right",row=1,col=1)
    if trades is not None and not trades.empty:
        for tt,sy,cl in[("BUY","triangle-up","#4caf50"),("SELL","triangle-down","#f44336")]:
            sub=trades[trades["Type"]==tt]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Entry Time"],y=sub["Entry"],mode="markers",
                    marker=dict(size=9,color=cl,symbol=sy,line=dict(color="white",width=0.8)),name=f"{tt} Entry"),row=1,col=1)
        for rsn,sy,cl in[("Target","circle","#66bb6a"),("SL","x","#ef5350"),("Reverse","diamond","#ab47bc")]:
            sub=trades[trades["Exit Reason"].str.contains(rsn,na=False)]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Exit Time"],y=sub["Exit"],mode="markers",
                    marker=dict(size=7,color=cl,symbol=sy),name=f"Exit({rsn})",visible="legendonly"),row=1,col=1)
    fig.update_layout(title=dict(text=f"🌊 {symbol}"+(f" · {tf_label}" if tf_label else ""),font=dict(size=13,color="#00e5ff")),
        template="plotly_dark",height=550,xaxis_rangeslider_visible=False,
        plot_bgcolor="#06101a",paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,font=dict(size=10)),
        margin=dict(l=10,r=70,t=45,b=10))
    return fig

def chart_equity(ec):
    eq=np.array(ec,dtype=float); pk=np.maximum.accumulate(eq); dd=(eq-pk)/pk*100
    fig=make_subplots(rows=2,cols=1,row_heights=[0.65,0.35],vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=eq,mode="lines",name="Equity",line=dict(color="#00bcd4",width=2),fill="tozeroy",fillcolor="rgba(0,188,212,.07)"),row=1,col=1)
    fig.add_trace(go.Scatter(y=dd,mode="lines",name="Drawdown %",line=dict(color="#f44336",width=1.5),fill="tozeroy",fillcolor="rgba(244,67,54,.12)"),row=2,col=1)
    fig.add_hline(y=0,line=dict(dash="dot",color="#546e7a",width=1),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=330,plot_bgcolor="#06101a",paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),margin=dict(l=10,r=10,t=15,b=10))
    return fig

def chart_opt_scatter(odf):
    fig=go.Figure(go.Scatter(x=odf["Max DD %"].abs(),y=odf["Return %"],mode="markers",
        marker=dict(size=(odf["Win %"]/5).clip(lower=4),color=odf["Score"],colorscale="Plasma",showscale=True,
                    colorbar=dict(title=dict(text="Score",font=dict(color="#b0bec5",size=11)),tickfont=dict(color="#b0bec5")),
                    line=dict(color="rgba(255,255,255,.2)",width=0.5)),
        text=[f"D={r.Depth} SL={r.SL} T={r.Target}" for _,r in odf.iterrows()],
        hovertemplate="<b>%{text}</b><br>Ret %{y:.1f}% DD %{x:.1f}%<extra></extra>"))
    fig.update_layout(title=dict(text="Return vs Max Drawdown (bubble=WinRate)",font=dict(size=12,color="#00e5ff")),
        xaxis_title="Max Drawdown %",yaxis_title="Total Return %",template="plotly_dark",height=360,
        plot_bgcolor="#06101a",paper_bgcolor="#06101a",font=dict(color="#b0bec5",family="Exo 2"),margin=dict(l=10,r=10,t=40,b=10))
    return fig

def generate_mtf_summary(symbol,results,overall_sig):
    lines=[f"## 🌊 {symbol} — Elliott Wave Analysis",f"*{now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}*\n"]
    bc=sum(1 for r in results if r["signal"]["signal"]=="BUY")
    sc=sum(1 for r in results if r["signal"]["signal"]=="SELL")
    vi={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(overall_sig,"⚪")
    lines.append(f"### {vi} Overall: **{overall_sig}** ({bc}B·{sc}S·{len(results)-bc-sc}H)\n")
    for r in results:
        sig=r["signal"]; s=sig["signal"]; em={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(s,"⚪")
        lines.append(f"#### {em} {r['tf_name']}")
        if s in("BUY","SELL"):
            ep,sl_,tgt_=sig["entry_price"],sig["sl"],sig["target"]
            rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
            lines.append(f"- {s} | Entry:{ep:.2f} SL:{sl_:.2f} Target:{tgt_:.2f} R:R 1:{rr:.1f}")
            lines.append(f"- Conf:{sig['confidence']:.0%} | MF:{sig.get('mf_score',0):.0%} | {sig['pattern']}")
        else:
            lines.append(f"- HOLD: {sig.get('reason','—')}")
        lines.append("")
    lines+=["---","| Wave | Meaning | Action |","|------|---------|--------|",
            "| W1 | First impulse | Missed |","| **W2** | **38–79% retrace** | **🟢 BUY/SELL here** |",
            "| **W3** | **Strongest** | **Hold** |","| W4 | Pullback | Partial exit |","| W5 | Final | Full exit |",
            "\n> ⚠️ *Not financial advice.*"]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ███  STREAMLIT UI  ███
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="main-hdr">
  <h1>🌊 Elliott Wave Algo Trader v5.4</h1>
  <p>Stale Signal Guard · Quick/Deep Opt · Full Auto Fix · Reliability Assessment · yfinance Limits Fixed</p>
</div>""",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Instrument")
    group_sel=st.selectbox("Category",list(TICKER_GROUPS.keys()),index=0)
    gmap=TICKER_GROUPS[group_sel]
    if group_sel=="✏️ Custom Ticker":
        symbol=st.text_input("Yahoo Finance ticker","KAYNES.NS")
    else:
        tn=st.selectbox("Instrument",list(gmap.keys()))
        symbol=gmap[tn]; st.caption(f"Yahoo: `{symbol}`")

    st.markdown("---")
    # Full Auto Mode
    auto_mode=st.checkbox("🤖 Full Auto Mode",value=st.session_state.auto_mode)
    st.session_state.auto_mode=auto_mode
    if auto_mode:
        st.markdown("""<div style="background:rgba(0,229,255,.1);border:1px solid #00e5ff;border-radius:8px;padding:8px 11px;font-size:.79rem;color:#00e5ff">
        🤖 <b>Full Auto Mode ON</b><br><span style="color:#78909c;font-size:.76rem">Scans 6 TF combos → optimizes → picks best → applies to sidebar/backtest/live</span></div>""",unsafe_allow_html=True)
        opt_mode_auto=st.selectbox("Optimization Speed",list(OPT_MODES.keys()),index=1,key="auto_opt_mode")
        if st.button("🚀 Run Full Auto",type="primary",use_container_width=True):
            if not st.session_state.auto_running:
                st.session_state.auto_running=True; st.session_state.auto_log=[]; st.session_state.auto_status="Starting…"
                run_full_auto(symbol,100_000.0,st.session_state.custom_sl_pts,st.session_state.custom_tgt_pts,opt_mode_auto)
                st.rerun()
        if st.session_state.auto_status: st.caption(st.session_state.auto_status)

    if st.session_state.best_cfg_applied:
        ss=st.session_state
        st.markdown(f"""<div style="background:rgba(0,229,255,.08);border:1px solid #00bcd4;border-radius:8px;padding:7px 10px;font-size:.79rem;margin-top:4px">
        ✨ <b style="color:#00e5ff">Optimized Config Active</b><br>
        <span style="color:#546e7a">TF:{ss.applied_interval} Per:{ss.applied_period} D:{ss.applied_depth}</span></div>""",unsafe_allow_html=True)

    st.markdown("---")
    c1,c2=st.columns(2)
    # Use applied interval/period from full auto if available, else use selectbox
    iv_idx=TIMEFRAMES.index(st.session_state.applied_interval) if st.session_state.applied_interval in TIMEFRAMES else 0
    interval=c1.selectbox("⏱ TF",TIMEFRAMES,index=iv_idx)
    vpl=VALID_PERIODS.get(interval,PERIODS)
    ap=st.session_state.applied_period
    per_idx=vpl.index(ap) if ap in vpl else min(4,len(vpl)-1)
    period=c2.selectbox("📅 Period",vpl,index=per_idx)
    st.session_state["_live_interval"]=interval

    st.markdown("---")
    st.markdown("### 🌊 Elliott Wave")
    depth=st.slider("Pivot Depth",2,15,st.session_state.applied_depth)
    conf_thresh=st.slider("Min Confidence %",40,90,int(MIN_CONFIDENCE*100),5)/100

    st.markdown("---")
    st.markdown("### 🛡️ Risk")
    si=SL_KEYS.index(st.session_state.applied_sl_lbl) if st.session_state.applied_sl_lbl in SL_KEYS else 0
    ti=TGT_KEYS.index(st.session_state.applied_tgt_lbl) if st.session_state.applied_tgt_lbl in TGT_KEYS else 0
    sl_lbl_sel=st.selectbox("Stop Loss",SL_KEYS,index=si)
    tgt_lbl_sel=st.selectbox("Target",TGT_KEYS,index=ti)
    sl_val=SL_MAP[sl_lbl_sel]; tgt_val=TGT_MAP[tgt_lbl_sel]
    if sl_val==SL_PTS: st.session_state.custom_sl_pts=st.number_input("SL Points",1.0,1e6,st.session_state.custom_sl_pts,5.0)
    if tgt_val==TGT_PTS: st.session_state.custom_tgt_pts=st.number_input("Target Points",1.0,1e6,st.session_state.custom_tgt_pts,10.0)
    if sl_val==SL_TRAIL: st.session_state.trailing_sl_pts=st.number_input("Trailing SL Pts",1.0,1e6,st.session_state.trailing_sl_pts,5.0)
    if tgt_val==TGT_TRAIL: st.session_state.trailing_tgt_pts=st.number_input("Trailing TGT Pts (display)",1.0,1e6,st.session_state.trailing_tgt_pts,10.0)
    csl=st.session_state.custom_sl_pts; ctgt=st.session_state.custom_tgt_pts
    trail_sl=st.session_state.trailing_sl_pts; trail_tgt=st.session_state.trailing_tgt_pts
    capital=st.number_input("💰 Capital (Rs)",10_000,50_000_000,100_000,10_000)

    st.markdown("---")
    st.markdown("### ⚠️ Stale Signal Guard")
    stale_check=st.checkbox("🛡 Prevent Stale Signal Entry",value=st.session_state.stale_check_enabled,
        help="When enabled: if SL or Target was already hit since signal fired, position will NOT be entered. Prevents entering stale/delayed signals.")
    st.session_state.stale_check_enabled=stale_check
    if stale_check:
        st.success("🛡 Active: System will skip stale signals where SL/Target already hit.")

    st.markdown("---")
    st.markdown("### 🏦 Dhan Brokerage")
    if not DHAN_LIB_OK: st.warning("Install: `pip install dhanhq`")
    dhan_on=st.checkbox("Enable Dhan Integration",value=False)
    dhan_client=None; trade_mode="stocks"; product_type="INTRADAY"
    segment="NSE"; entry_ot="MARKET"; exit_ot="MARKET"; ce_sec_id=""; pe_sec_id=""; live_qty=1
    if dhan_on:
        d_cid=st.text_input("Client ID",value=DEFAULT_CLIENT_ID)
        d_tok=st.text_area("Access Token",value=DEFAULT_ACCESS_TOKEN,height=70)
        if d_cid and d_tok:
            dhan_client=DhanClient(d_cid,d_tok)
            if st.button("🔌 Test"): st.json(dhan_client.fund_limit())
        trade_mode=st.radio("Mode",["Stocks","Options"],horizontal=True).lower()
        if trade_mode=="stocks":
            c1d,c2d=st.columns(2); product_type=c1d.selectbox("Type",["INTRADAY","DELIVERY"],index=0)
            exchange=c2d.selectbox("Exchange",["NSE","BSE"],index=0); segment=exchange
            sec_id_s=st.text_input("Security ID","1333"); live_qty=st.number_input("Qty",1,100_000,1)
            ce_sec_id=pe_sec_id=sec_id_s
        else:
            c1d,c2d=st.columns(2); segment=c1d.selectbox("Segment",["NSE_FNO","BSE_FNO"],index=0)
            product_type=c2d.selectbox("Product",["INTRADAY","MARGIN"],index=0)
            ce_sec_id=st.text_input("CE Security ID","52175"); pe_sec_id=st.text_input("PE Security ID","52176")
            live_qty=st.number_input("Lot Size",1,10_000,50)
            st.caption("BUY→CE | SELL→PE. Limit price from signal.")
        c1o,c2o=st.columns(2)
        entry_ot=c1o.selectbox("Entry Order",["LIMIT","MARKET"],index=0)
        exit_ot=c2o.selectbox("Exit Order",["MARKET","LIMIT"],index=0)
        if entry_ot=="LIMIT": st.caption("📌 Limit price = signal entry level")
        if exit_ot=="LIMIT": st.caption("📌 Limit exit = SL/Target level")

    st.markdown("---")
    st.caption(f"⚡ Rate-limit 1.5s | LTP 2s | Conf≥{conf_thresh:.0%}")
    st.caption(f"`{symbol}` · `{interval}` · `{period}`")
    st.caption(f"5m/15m max period: **1mo** (yfinance limit)")

# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────
t_analysis,t_live,t_bt,t_opt,t_reliability,t_help=st.tabs([
    "🔭  Wave Analysis","🔴  Live Trading","📊  Backtest","🔬  Optimization","🔍  Reliability","❓  Help"])

# ── WAVE ANALYSIS ──────────────────────────────────────────────────────────
with t_analysis:
    st.markdown("### 🔭 Multi-Timeframe Elliott Wave Analysis")
    ac1,ac2,ac3=st.columns([1.2,1,2.4])
    with ac1: run_analysis=st.button("🔭 Run Full Analysis",type="primary",use_container_width=True)
    with ac2: custom_tf_only=st.checkbox("Sidebar TF only",value=False)
    with ac3: st.caption(f"{'Sidebar TF' if custom_tf_only else 'Daily·4H·1H·15M'} | Min conf {conf_thresh:.0%} | 8-factor scoring")
    if run_analysis:
        MTF_COMBOS=[("1d","1y","Daily"),("4h","3mo","4-Hour"),("1h","1mo","1-Hour"),("15m","7d","15-Min")]
        sc2=[(interval,period,f"{interval}·{period}")] if custom_tf_only \
            else [(tf,per,nm) for tf,per,nm in MTF_COMBOS if per in VALID_PERIODS.get(tf,[])]
        results=[]; prog=st.progress(0,text="Scanning…")
        for idx,(tf,per,nm) in enumerate(sc2):
            prog.progress((idx+1)/len(sc2),text=f"Fetching {nm}…")
            try:
                dfa=fetch_ohlcv(symbol,tf,per,md=1.5)
                if dfa is not None and len(dfa)>=45:
                    sa=ew_signal(dfa.iloc[:-1],depth,sl_val,tgt_val,csl,ctgt,conf_thresh)
                    results.append({"tf_name":nm,"interval":tf,"period":per,"signal":sa,"df":dfa,"pivots":find_pivots(dfa.iloc[:-1],depth)})
                else:
                    results.append({"tf_name":nm,"interval":tf,"period":per,"signal":_blank("No data"),"df":None,"pivots":[]})
            except Exception as e:
                results.append({"tf_name":nm,"interval":tf,"period":per,"signal":_blank(str(e)),"df":None,"pivots":[]})
        prog.empty()
        bs2=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="BUY")
        ss2=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="SELL")
        ov="BUY" if bs2>ss2 and bs2>0.5 else "SELL" if ss2>bs2 and ss2>0.5 else "HOLD"
        st.session_state.update({"_analysis_results":results,"_analysis_overall":ov,"_analysis_symbol":symbol})
    ar=st.session_state.get("_analysis_results")
    if ar:
        ov=st.session_state.get("_analysis_overall","HOLD"); asym=st.session_state.get("_analysis_symbol",symbol)
        vc={"BUY":"#4caf50","SELL":"#f44336","HOLD":"#ffb300"}; vb={"BUY":"rgba(76,175,80,.10)","SELL":"rgba(244,67,54,.10)","HOLD":"rgba(255,179,0,.10)"}; vi={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}
        st.markdown(f"""<div style="background:{vb[ov]};border:2px solid {vc[ov]};border-radius:12px;padding:14px 20px;margin-bottom:10px;text-align:center">
        <span style="font-size:1.5rem;color:{vc[ov]};font-weight:700">{vi[ov]} Overall: {ov}</span><br>
        <span style="color:#78909c;font-size:.84rem">{asym} · Multi-TF Consensus</span></div>""",unsafe_allow_html=True)
        tfc=st.columns(min(len(ar),4))
        for i,r in enumerate(ar):
            with tfc[i%4]:
                s=r["signal"]["signal"]; c_=r["signal"]["confidence"]; mf_=r["signal"].get("mf_score",0)
                sc3=vc.get(s,"#546e7a"); em=vi.get(s,"⚪")
                st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:1px solid #1e3a5f;border-radius:8px;padding:9px 11px;text-align:center;margin-bottom:4px">
                <div style="font-size:.75rem;color:#546e7a">{r['tf_name']}</div>
                <div style="font-size:1.05rem;color:{sc3};font-weight:700">{em} {s}</div>
                <div style="font-size:.73rem;color:#78909c">{r['signal']['pattern']}</div>
                <div style="font-size:.77rem;color:#00bcd4">Conf {c_:.0%} | MF {mf_:.0%}</div></div>""",unsafe_allow_html=True)
        st.markdown("---")
        for r in ar:
            if r["df"] is not None:
                s_=r["signal"]["signal"]
                with st.expander(f"📈 {r['tf_name']} — {s_} (Conf {r['signal']['confidence']:.0%} | MF {r['signal'].get('mf_score',0):.0%})",expanded=(s_!="HOLD")):
                    st.plotly_chart(chart_waves(r["df"],r["pivots"],r["signal"],symbol=asym,tf_label=r["tf_name"]),use_container_width=True)
                    if s_ in("BUY","SELL"):
                        sg=r["signal"]; ep,sl_,tgt_=sg["entry_price"],sg["sl"],sg["target"]
                        rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
                        mc1,mc2,mc3,mc4,mc5=st.columns(5)
                        mc1.metric("Entry",f"{ep:.2f}"); mc2.metric("SL",f"{sl_:.2f}",delta=f"-{abs(ep-sl_)/ep*100:.1f}%",delta_color="inverse")
                        mc3.metric("Target",f"{tgt_:.2f}",delta=f"+{abs(tgt_-ep)/ep*100:.1f}%"); mc4.metric("R:R",f"1:{rr:.1f}"); mc5.metric("MF Score",f"{sg.get('mf_score',0):.0%}")
                        if sg.get("factors"):
                            st.markdown("**8-Factor Analysis:**")
                            fc=st.columns(4)
                            for j,(fk,fv) in enumerate(sg["factors"].items()):
                                fc[j%4].markdown(f"<small style='color:#78909c'><b style='color:#00bcd4'>{fk}</b>: {fv}</small>",unsafe_allow_html=True)
        st.markdown("---"); st.markdown("### 📋 Analysis & Recommendations"); st.markdown(generate_mtf_summary(asym,ar,ov))
    else:
        st.info("Click 🔭 Run Full Analysis to scan all timeframes with 8-factor confirmation.")

# ── LIVE TRADING ───────────────────────────────────────────────────────────
with t_live:
    ctl1,ctl2,ctl3=st.columns([1,1,4])
    with ctl1:
        if not st.session_state.live_running:
            if st.button("▶ Start Live",type="primary",use_container_width=True):
                st.session_state.update({"live_running":True,"live_phase":"scanning","live_log":[],
                    "last_bar_ts":None,"last_fetch_wall":0.0,"live_no_pos_reason":"Starting…"}); st.rerun()
        else:
            if st.button("⏹ Stop",type="secondary",use_container_width=True):
                st.session_state.update({"live_running":False,"live_phase":"idle"}); st.rerun()
    with ctl2:
        if st.button("🔄 Reset All",use_container_width=True):
            for k,v in _DEF.items(): st.session_state[k]=v
            st.success("Reset ✓"); time.sleep(0.3); st.rerun()
    with ctl3:
        if st.session_state.live_running:
            st.success(f"🟢 RUNNING — `{symbol}` · `{interval}` · `{period}` | LTP 2s | MinConf {conf_thresh:.0%} | Stale Guard {'🛡 ON' if stale_check else 'OFF'}")
        else:
            st.warning("⚫ STOPPED — Click ▶ Start Live. Everything runs automatically.")

    # Auto best config banner
    auto_best=st.session_state.auto_best_cfg
    if auto_best:
        st.markdown(f"""<div class="auto-banner">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">
        <div><div style="font-size:.73rem;color:#78909c">🤖 FULL AUTO — BEST CONFIG APPLIED</div>
        <div style="font-size:1.2rem;font-weight:700;color:#00e5ff">{auto_best['tf_name']} · Depth {auto_best['depth']} · SL: {auto_best['sl']} · Target: {auto_best['target']}</div>
        <div style="font-size:.80rem;color:#b0bec5">Win <b style="color:#4caf50">{auto_best['win_pct']:.0f}%</b> · Return <b style="color:#4caf50">{auto_best['return_pct']:.1f}%</b> · Sharpe <b>{auto_best['sharpe']:.2f}</b></div></div>
        <div style="text-align:right"><div style="font-size:.72rem;color:#546e7a">Current Signal</div>
        <div style="font-size:1.05rem;font-weight:700;color:{'#4caf50' if auto_best['current_signal']=='BUY' else '#f44336' if auto_best['current_signal']=='SELL' else '#78909c'}">
        {'🟢' if auto_best['current_signal']=='BUY' else '🔴' if auto_best['current_signal']=='SELL' else '⏸'} {auto_best['current_signal']} ({auto_best['current_conf']:.0%})</div></div></div>
        <div style="font-size:.75rem;color:#546e7a;margin-top:6px">✅ Config applied to sidebar + backtest + live. System will trade based on this config automatically.</div>
        </div>""",unsafe_allow_html=True)

    pos_ph=st.empty()

    def render_pos(container):
        ss=st.session_state; pos=ss.live_position; ltp=ss.live_ltp; phase=ss.live_phase
        with container:
            if pos:
                pt=pos["type"]; clr="#4caf50" if pt=="BUY" else "#f44336"; css="pos-buy" if pt=="BUY" else "pos-sell"
                up=pos.get("unreal_pnl",0); up_c="#4caf50" if up>=0 else "#f44336"; up_sym="▲" if up>=0 else "▼"
                sl_=pos["sl"]; tgt_=pos.get("target_display",pos["target"]); ds=pos.get("dist_sl",0); dt=pos.get("dist_tgt",0)
                ltp_s=f"{ltp:,.2f}" if ltp else "—"
                st.markdown(f"""<div class="{css}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px">
                <div><div style="font-size:.72rem;color:#78909c">POSITION STATUS</div>
                <div style="font-size:1.85rem;font-weight:700;color:{clr}">{pt} OPEN ✅</div>
                <div style="font-size:.79rem;color:#b0bec5">{pos.get('pattern','—')} | Conf {pos.get('confidence',0):.0%} | MF {pos.get('mf_score',0):.0%}</div>
                <div style="font-size:.73rem;color:#546e7a">R:R 1:{pos.get('rr',0):.1f} | Entered {pos.get('entry_ist','—')}</div></div>
                <div style="text-align:right"><div style="font-size:.71rem;color:#546e7a">UNREALIZED P&amp;L</div>
                <div style="font-size:1.55rem;font-weight:700;color:{up_c};font-family:'Share Tech Mono'">{up_sym} Rs{abs(up):,.2f}</div>
                <div style="font-size:.76rem;color:#78909c">LTP: {ltp_s}</div></div></div>
                <hr style="border-color:rgba(255,255,255,.08);margin:8px 0">
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:.80rem">
                <div><div style="color:#546e7a;font-size:.70rem">ENTRY</div><div style="color:#b0bec5;font-weight:600">{pos['entry']:,.2f}</div></div>
                <div><div style="color:#546e7a;font-size:.70rem">SL</div><div style="color:#ff7043;font-weight:600">{sl_:,.2f}<br><span style="color:#546e7a;font-size:.70rem">{ds:.1f}% away</span></div></div>
                <div><div style="color:#546e7a;font-size:.70rem">TARGET</div><div style="color:#66bb6a;font-weight:600">{tgt_:,.2f}<br><span style="color:#546e7a;font-size:.70rem">{dt:.1f}% away</span></div></div>
                <div><div style="color:#546e7a;font-size:.70rem">QTY</div><div style="color:#b0bec5;font-weight:600">{pos['qty']}</div></div></div>
                <div style="margin-top:8px;padding:6px 10px;background:rgba(0,0,0,.3);border-radius:6px;font-size:.75rem;color:#78909c">
                🤖 Auto-managed. Closes at SL/Target/Signal Reverse. <b>No action needed.</b></div></div>""",unsafe_allow_html=True)
            elif phase=="reversing":
                st.markdown("""<div class="pos-reversing"><div style="font-size:1.3rem;font-weight:700;color:#ab47bc">🔄 AUTO-REVERSING…</div>
                <div style="font-size:.81rem;color:#b0bec5;margin-top:4px">Closing old position and opening reverse. No action needed.</div></div>""",unsafe_allow_html=True)
            else:
                sig_=ss.live_last_sig
                if phase in("scanning","signal_ready") and sig_ and sig_.get("signal") in("BUY","SELL"):
                    sc_="#00e5ff"
                    st.markdown(f"""<div class="pos-signal"><div style="font-size:.72rem;color:#78909c">SIGNAL DETECTED — OPENING POSITION</div>
                    <div style="font-size:1.45rem;font-weight:700;color:{sc_}">💡 {sig_['signal']} FIRED</div>
                    <div style="font-size:.79rem;color:#b0bec5">{sig_.get('pattern','—')} | Conf {sig_.get('confidence',0):.0%}</div>
                    <div style="font-size:.77rem;color:#78909c;margin-top:4px">Entry: ~{sig_.get('entry_price',0):,.2f} | SL: {sig_.get('sl',0):,.2f} | Target: {sig_.get('target',0):,.2f}</div>
                    <div style="font-size:.74rem;color:#546e7a;margin-top:5px">🤖 Opening automatically on next bar.</div></div>""",unsafe_allow_html=True)
                else:
                    nr=ss.live_no_pos_reason
                    st.markdown(f"""<div class="pos-none"><div style="font-size:.72rem;color:#546e7a">POSITION STATUS</div>
                    <div style="font-size:1.3rem;font-weight:600;color:#37474f;margin:4px 0">⏸ NO OPEN POSITION</div>
                    <div style="font-size:.79rem;color:#455a64;white-space:pre-line;word-break:break-word">{nr}</div></div>""",unsafe_allow_html=True)

    render_pos(pos_ph)
    st.markdown("---")

    # ── Signal Snapshot panel ─────────────────────────────────────────────
    snap=st.session_state.signal_snapshot
    if snap:
        si_=snap["stale_info"]; bg="#060d14"; brd="#1e3a5f"
        if si_["sl_hit"]: bg="rgba(244,67,54,.08)"; brd="#ef5350"
        elif si_["tgt_hit"]: bg="rgba(76,175,80,.08)"; brd="#4caf50"
        elif si_["bars_elapsed"]>5: bg="rgba(255,152,0,.08)"; brd="#ff9800"
        d_ltp=snap["cur_ltp"]-snap["ltp_at_signal"]
        d_ltp_c="#4caf50" if(snap["direction"]=="BUY" and d_ltp>0) or(snap["direction"]=="SELL" and d_ltp<0) else "#f44336"
        st.markdown(f"""
        <div class="sig-snap" style="background:{bg};border-color:{brd}">
        <div style="font-size:.73rem;color:#546e7a;letter-spacing:.4px">📸 SIGNAL SNAPSHOT — THEN vs NOW</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:8px;font-size:.82rem">
          <div>
            <div style="color:#546e7a;font-size:.70rem">WHEN SIGNAL FIRED</div>
            <div style="color:#b0bec5">📅 {snap['time_ist']}</div>
            <div style="color:#b0bec5">LTP: <b>{snap['ltp_at_signal']:,.2f}</b></div>
            <div style="color:#ff7043">SL: {snap['sl_at_signal']:,.2f}</div>
            <div style="color:#66bb6a">Target: {snap['tgt_at_signal']:,.2f}</div>
          </div>
          <div>
            <div style="color:#546e7a;font-size:.70rem">RIGHT NOW</div>
            <div style="color:#b0bec5">📅 {now_ist().strftime('%d-%b %H:%M:%S IST')}</div>
            <div style="color:{d_ltp_c}">LTP: <b>{snap['cur_ltp']:,.2f}</b> ({d_ltp:+.2f})</div>
            <div style="color:#ff7043">SL if entering: {snap['cur_sl']:,.2f}</div>
            <div style="color:#66bb6a">Target if entering: {snap['cur_tgt']:,.2f}</div>
          </div>
          <div>
            <div style="color:#546e7a;font-size:.70rem">STALENESS CHECK</div>
            <div style="color:#b0bec5">Bars since signal: {si_['bars_elapsed']}</div>
            <div style="color:#b0bec5">Max adverse: {si_['max_adverse']:.2f}%</div>
            <div style="color:#b0bec5">Max favorable: {si_['max_favorable']:.2f}%</div>
          </div>
        </div>
        <div style="margin-top:8px;font-size:.79rem;color:{'#ef5350' if si_['is_stale'] else '#78909c'};border-top:1px solid rgba(255,255,255,.06);padding-top:6px">
        {'🚫 STALE: ' if si_['is_stale'] else '✅ '}{si_['reason']}
        {'<br>⚠️ <b>Stale Guard is OFF</b> — position will still open. Enable checkbox in sidebar to block.' if si_['is_stale'] and not stale_check else ''}
        </div></div>""",unsafe_allow_html=True)

    try:
        @st.fragment(run_every=2)
        def live_frag():
            ss=st.session_state
            if not ss.live_running: return
            new_cycle=live_engine_tick(symbol,interval,period,depth,sl_val,tgt_val,
                csl,ctgt,trail_sl,trail_tgt,dhan_on,dhan_client,
                trade_mode,entry_ot,exit_ot,product_type,segment,ce_sec_id,pe_sec_id,live_qty,
                stale_check,conf_thresh)
            if new_cycle: render_pos(pos_ph)
            ltp_=ss.live_ltp; real_pnl=ss.live_pnl
            pos_=ss.live_position; unreal=pos_.get("unreal_pnl",0) if pos_ else 0
            tc=st.columns(6)
            tc[0].metric("📊 LTP",f"{ltp_:,.2f}" if ltp_ else "—",delta=ss.live_ltp_ts or "—",delta_color="off")
            tc[1].metric("⏩ Data Age",f"{ss.live_delay_s}s",delta=ss.live_delay_ctx[:28] if ss.live_delay_ctx else "—",delta_color="off")
            tc[2].metric("🕒 IST",now_ist().strftime("%H:%M:%S"),delta=ss.live_last_candle_ist,delta_color="off")
            tc[3].metric("🔜 Next Check",ss.live_next_check_ist)
            tc[4].metric("💰 Realized P&L",f"Rs{real_pnl:,.2f}",delta=f"{len(ss.live_trades)} trades",delta_color="normal" if real_pnl>=0 else "inverse")
            tc[5].metric("📈 Unrealized",f"Rs{unreal:,.2f}" if pos_ else "No pos",delta_color="normal" if unreal>=0 else "inverse")
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
                        for lbl_f,val_f in fibs.items():
                            bc2="#4caf50" if "1.618" in lbl_f else("#ab47bc" if "2.618" in lbl_f else "#ffb300")
                            st.markdown(f"""<div style="display:flex;justify-content:space-between;background:#060d14;border-left:3px solid {bc2};border-radius:0 4px 4px 0;padding:3px 9px;margin-bottom:3px;font-size:.79rem">
                            <span style="color:#78909c">{lbl_f}</span><span style="color:{bc2};font-family:'Share Tech Mono'">{val_f:,.2f}</span></div>""",unsafe_allow_html=True)
                    st.markdown(f"""<div class="wave-card" style="margin-top:8px">{ws.get('action','')}</div>""",unsafe_allow_html=True)
            with sig_col:
                sig_=ss.live_last_sig
                if sig_ and sig_.get("signal")!="HOLD":
                    s_=sig_["signal"]; sc_="#4caf50" if s_=="BUY" else "#f44336"; em_="🟢" if s_=="BUY" else "🔴"
                    rr_=sig_.get("rr",0); mf_=sig_.get("mf_score",0); mf_c="#4caf50" if mf_>=0.7 else "#ffb300" if mf_>=0.55 else "#f44336"
                    st.markdown(f"""<div style="background:rgba(255,255,255,.03);border:2px solid {sc_};border-radius:10px;padding:13px 15px">
                    <div style="font-size:.72rem;color:#546e7a">LAST SIGNAL</div>
                    <div style="font-size:1.35rem;color:{sc_};font-weight:700">{em_} {s_}</div>
                    <div style="font-size:.78rem;color:#b0bec5">{sig_.get('pattern','—')}</div>
                    <div style="font-size:.76rem;color:#00bcd4">Conf {sig_.get('confidence',0):.0%} | MF <span style="color:{mf_c}">{mf_:.0%}</span></div>
                    <hr style="border-color:rgba(255,255,255,.08);margin:7px 0">
                    <div style="font-size:.77rem;color:#78909c">
                    Entry: <b style="color:#b0bec5">{sig_.get('entry_price',0):,.2f}</b><br>
                    SL: <b style="color:#ff7043">{sig_.get('sl',0):,.2f}</b><br>
                    Target: <b style="color:#66bb6a">{sig_.get('target',0):,.2f}</b><br>
                    R:R: <b>1:{rr_:.1f}</b></div>
                    <div style="font-size:.72rem;color:#455a64;margin-top:5px">{sig_.get('reason','—')[:110]}</div>
                    </div>""",unsafe_allow_html=True)
                    if sig_.get("factors"):
                        with st.expander("📊 8-Factor Breakdown",expanded=False):
                            for fk,fv in sig_["factors"].items(): st.caption(f"**{fk}**: {fv}")
                else:
                    nr=ss.live_no_pos_reason
                    st.markdown(f"""<div style="background:#060d14;border:1px solid #263238;border-radius:10px;padding:13px 15px">
                    <div style="font-size:.72rem;color:#546e7a">LAST SIGNAL</div>
                    <div style="font-size:1.05rem;color:#37474f;font-weight:600">⏸ HOLD / NO SIGNAL</div>
                    <div style="font-size:.77rem;color:#37474f;margin-top:5px;line-height:1.55;word-break:break-word">{nr}</div></div>""",unsafe_allow_html=True)
            if ss._scan_df is not None:
                piv_=find_pivots(ss._scan_df.iloc[:-1],depth)
                st.plotly_chart(chart_waves(ss._scan_df,piv_,ss._scan_sig,symbol=symbol),use_container_width=True)
            h1c,h2c=st.columns(2)
            with h1c:
                if ss.live_signals:
                    st.markdown("##### 📋 Signals"); st.dataframe(pd.DataFrame(ss.live_signals).tail(8),use_container_width=True,height=140)
            with h2c:
                if ss.live_trades:
                    st.markdown("##### 🏁 Trades"); td=pd.DataFrame(ss.live_trades); st.dataframe(td.tail(8),use_container_width=True,height=140)
                    wns=(td["PnL Rs"]>0).sum(); tot=len(td); pnl_=td["PnL Rs"].sum()
                    st.caption(f"Win {wns}/{tot} ({wns/tot*100:.0f}%) | Realized Rs{pnl_:,.2f}")
            if ss.live_log:
                with st.expander("📜 Log",expanded=False): st.code("\n".join(reversed(ss.live_log[-60:])),language=None)
            if ss.auto_log:
                with st.expander("🤖 Auto Mode Log",expanded=False): st.code("\n".join(ss.auto_log[-30:]),language=None)
        live_frag()
    except Exception as fe:
        st.warning(f"⚠️ Fragment unavailable: {fe}. Upgrade: `pip install streamlit --upgrade`")
        if st.button("🔍 Manual Scan",use_container_width=True):
            with st.spinner("Running…"):
                live_engine_tick(symbol,interval,period,depth,sl_val,tgt_val,csl,ctgt,trail_sl,trail_tgt,
                    dhan_on,dhan_client,trade_mode,entry_ot,exit_ot,product_type,segment,
                    ce_sec_id,pe_sec_id,live_qty,stale_check,conf_thresh)
            render_pos(pos_ph); st.rerun()

# ── BACKTEST ───────────────────────────────────────────────────────────────
with t_bt:
    bl,br=st.columns([1,2.6],gap="medium")
    with bl:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""<div class="info-box">
        📈 <code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        🌊 Depth <code>{depth}</code> · Conf≥<code>{conf_thresh:.0%}</code><br>
        🛡 SL <code>{sl_lbl_sel}</code> · 🎯 <code>{tgt_lbl_sel}</code><br>
        💰 Rs<code>{capital:,}</code><br>
        <small style="color:#546e7a">Signal N → entry open N+1<br>
        Min R:R 1.2 · W3>W1 rule enforced<br>SL: Low(BUY)/High(SELL) first</small></div>""",unsafe_allow_html=True)
        if st.session_state.best_cfg_applied:
            st.success(f"✨ Optimized: {st.session_state.applied_interval}/{st.session_state.applied_period} D={st.session_state.applied_depth}")
        if st.button("🚀 Run Backtest",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dbt=fetch_ohlcv(symbol,interval,period,md=1.5)
            if dbt is None or len(dbt)<50: st.error("Not enough data. Use longer period.")
            else:
                with st.spinner(f"Running on {len(dbt)} bars…"):
                    res=run_backtest(dbt,depth,sl_val,tgt_val,capital,csl,ctgt,trail_sl,conf_thresh)
                    res.update({"df":dbt,"pivots":find_pivots(dbt,depth),"symbol":symbol,"interval":interval,"period":period})
                st.session_state.bt_results=res
                if "error" in res: st.error(res["error"])
                else: st.success(f"✅ {res['metrics']['Total Trades']} trades | Win Rate: {res['metrics']['Win Rate %']}%")
    with br:
        r=st.session_state.bt_results
        if r and "metrics" in r:
            m=r["metrics"]
            st.markdown(f"### Results — `{r.get('symbol','')}` · `{r.get('interval','')}` · `{r.get('period','')}`")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Win Rate",f"{m['Win Rate %']}%",delta=f"{m['Wins']}W/{m['Losses']}L")
            c2.metric("Total Return",f"{m['Total Return %']}%",delta=f"Rs{m['Final Equity Rs']:,}")
            c3.metric("Profit Factor",str(m["Profit Factor"])); c4.metric("Max Drawdown",f"{m['Max Drawdown %']}%")
            c5,c6,c7,c8=st.columns(4)
            c5.metric("Sharpe",str(m["Sharpe Ratio"])); c6.metric("Trades",str(m["Total Trades"]))
            c7.metric("Avg Win",f"Rs{m['Avg Win Rs']:,}"); c8.metric("Avg Loss",f"Rs{m['Avg Loss Rs']:,}")
            eb=r.get("exit_breakdown",{})
            if eb:
                ea,eb2,ec2,ed=st.columns(4)
                ea.metric("SL Hits",str(eb.get("SL hits",0))); eb2.metric("Target Hits",str(eb.get("Target hits",0)))
                ec2.metric("Sig Reverse",str(eb.get("Signal Reverse",0))); ed.metric("Still Open",str(eb.get("Still open",0)))
            tc1,tc2,tc3=st.tabs(["🕯 Wave Chart","📈 Equity","📋 Trades"])
            with tc1: st.plotly_chart(chart_waves(r["df"],r["pivots"],_blank(),r["trades"],r["symbol"]),use_container_width=True)
            with tc2: st.plotly_chart(chart_equity(r["equity_curve"]),use_container_width=True)
            with tc3:
                st.dataframe(r["trades"],use_container_width=True,height=400)
                st.info("SL Verified ✅=Low≤SL(BUY)/High≥SL(SELL). TGT Verified ✅=High≥Target(BUY)/Low≤Target(SELL).")
                st.download_button("📥 CSV",data=r["trades"].to_csv(index=False),file_name=f"ew_bt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        elif r and "error" in r: st.error(r["error"])
        else: st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run backtest to see results</h3></div>",unsafe_allow_html=True)

# ── OPTIMIZATION ───────────────────────────────────────────────────────────
with t_opt:
    ol,or_=st.columns([1,3],gap="medium")
    with ol:
        st.markdown("### ⚙️ Config")
        opt_mode=st.selectbox("Optimization Mode",list(OPT_MODES.keys()),index=1)
        st.caption(OPT_MODES[opt_mode]["desc"])
        min_trades_ui=st.number_input("Min Trades Filter",1,50,MIN_TRADES_FILTER,1,
            help="Exclude results with fewer than this many trades. Prevents 1-trade 100% win rate nonsense.")
        st.markdown(f"""<div class="info-box"><code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        Conf≥<code>{conf_thresh:.0%}</code> · Min trades: <code>{min_trades_ui}</code><br>
        <b>{len(list(itertools.product(OPT_MODES[opt_mode]['depths'],OPT_MODES[opt_mode]['sl_opts'],OPT_MODES[opt_mode]['tgt_opts'])))} combos</b>
        </div>""",unsafe_allow_html=True)
        if st.button("🔬 Run Optimization",type="primary",use_container_width=True):
            with st.spinner("Fetching…"):
                dopt=fetch_ohlcv(symbol,interval,period,md=1.5)
            if dopt is None or len(dopt)<55: st.error("Not enough data.")
            else:
                prog_opt=st.progress(0,text="Optimizing…")
                with st.spinner(f"Running {opt_mode}…"):
                    odf=run_optimization(dopt,capital,csl,ctgt,trail_sl,opt_mode,conf_thresh,min_trades_ui,prog_opt)
                st.session_state.opt_results={"df":odf,"symbol":symbol,"interval":interval,"period":period}
                if odf.empty: st.warning("No results. Try longer period, lower conf threshold, or reduce min trades.")
                else: st.success(f"✅ Best: Win {odf['Win %'].iloc[0]:.0f}% | Score {odf['Score'].iloc[0]:.2f}")
    with or_:
        optr=st.session_state.opt_results
        if optr and optr.get("df") is not None and not optr["df"].empty:
            odf=optr["df"]
            st.markdown(f"### Results — `{optr['symbol']}` · `{optr['interval']}` · `{optr['period']}`")
            br_=odf.iloc[0]
            b1,b2,b3,b4,b5=st.columns(5)
            b1.metric("Best Depth",str(int(br_["Depth"]))); b2.metric("Best SL",str(br_["SL"]))
            b3.metric("Best Target",str(br_["Target"])); b4.metric("Win Rate",f"{br_['Win %']:.0f}%"); b5.metric("Score",f"{br_['Score']:.2f}")
            st.markdown("---")
            ac1,ac2=st.columns([1.6,2.4])
            with ac1:
                nr=st.number_input("Apply top N",min_value=1,max_value=min(5,len(odf)),value=1,step=1)
                ar_=odf.iloc[int(nr)-1]
            with ac2:
                st.markdown(f"""<div class="best-cfg"><b style="color:#00e5ff">Config #{int(nr)}</b><br>
                Depth <b>{int(ar_['Depth'])}</b> · SL <b>{ar_['SL']}</b> · Target <b>{ar_['Target']}</b><br>
                Win <b>{ar_['Win %']:.0f}%</b> · Return <b>{ar_['Return %']:.1f}%</b> · Trades <b>{int(ar_['Trades'])}</b> · Score <b>{ar_['Score']:.2f}</b></div>""",unsafe_allow_html=True)
            if st.button(f"✨ Apply Config #{int(nr)} → Sidebar + Backtest + Live",type="primary",use_container_width=True):
                apply_best_cfg({"depth":int(ar_["Depth"]),"sl":ar_["SL"],"target":ar_["Target"],
                                "tf":interval,"period":period})
                st.success(f"✅ Applied! Depth={int(ar_['Depth'])} · SL={sl_lbl(ar_['SL'])} · Target={tgt_lbl(ar_['Target'])}")
                time.sleep(0.4); st.rerun()
            st.markdown("---")
            oc1,oc2=st.tabs(["📊 Scatter","📋 Table"])
            with oc1:
                st.plotly_chart(chart_opt_scatter(odf),use_container_width=True)
                st.caption("Top-left = best. Bubble = WinRate. Color = Score. Min trades filter removes 1-trade outliers.")
            with oc2:
                def _hl(row):
                    if row.name==0: return["background-color:rgba(0,229,255,.18)"]*len(row)
                    elif row.name<3: return["background-color:rgba(0,229,255,.08)"]*len(row)
                    return[""]*len(row)
                st.dataframe(odf.style.apply(_hl,axis=1),use_container_width=True,height=500)
                st.download_button("📥 CSV",data=odf.to_csv(index=False),file_name=f"ew_opt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        else:
            st.markdown("<div style='text-align:center;padding:60px;color:#37474f'><h3>Run optimization to see results</h3></div>",unsafe_allow_html=True)

# ── RELIABILITY ASSESSMENT ─────────────────────────────────────────────────
with t_reliability:
    st.markdown("## 🔍 Reliability Assessment — Honest Evaluation")
    st.markdown("""<div class="reliability-warn">
    <b style="color:#ef5350;font-size:1.1rem">⚠️ READ THIS BEFORE LIVE TRADING</b><br>
    <span style="font-size:.85rem;color:#ef9a9a">This is an honest assessment of what this app can and cannot do reliably.</span>
    </div>""",unsafe_allow_html=True)
    col1,col2=st.columns(2,gap="large")
    with col1:
        st.markdown("""
### ✅ What This App Does RELIABLY

**Pattern Detection:**
- Correctly identifies ZigZag pivots using proven algorithm
- Fibonacci retracement calculation is mathematically accurate
- Wave-2 detection logic follows classical Elliott Wave rules
- 8-factor confluence scoring adds meaningful signal filtering

**Backtesting:**
- Walk-forward simulation (no future data leakage)
- Conservative fills: SL checked with Low (BUY) before Target with High
- Min R:R 1.2:1 filter removes poor setups
- Min trade count prevents 1-trade "100% accuracy" illusions

**Infrastructure:**
- 1.5s rate-limit respects yfinance limits
- Stale signal detection prevents entering dead signals
- Fragment-based LTP refresh (no page flicker)
- Realized/Unrealized P&L correctly separated

---

### ⚠️ Known Limitations

**yfinance Delays:**
- Free yfinance data is delayed 15-20 minutes for most exchanges
- NSE/BSE intraday data may be delayed or unavailable
- 5m/15m data limited to 60 days (~1mo)
- Data gaps during pre/post-market hours

**Elliott Wave Subjectivity:**
- Elliott Wave has inherent subjectivity — different traders count waves differently
- Algorithm may identify valid patterns that don't play out as expected
- Wave labels (W1, W2, W3) are probabilistic, not definitive
- No single indicator is 100% reliable in all market conditions

**Backtesting vs Reality:**
- Backtest assumes you can always get filled at next bar's open
- Real execution has slippage, commissions, partial fills
- Market conditions change — past performance ≠ future results
- Small sample sizes (< 30 trades) produce unreliable statistics
        """)
    with col2:
        st.markdown("""
### 🚫 What This App CANNOT Do

- Predict the future with certainty
- Replace professional trading judgment
- Account for fundamental/news events (earnings, RBI decisions, geopolitical)
- Handle circuit breakers, trading halts, or illiquid markets
- Guarantee profits even with high backtest win rates
- Access real-time NSE/BSE data (yfinance limitation)

---

### 📊 Realistic Expectations

| Metric | What to Expect | Red Flag |
|--------|---------------|----------|
| Win Rate | 55–70% with good config | >85% = overfit |
| Profit Factor | 1.3–2.0 | >3.0 with <20 trades = unreliable |
| Trades per year | 10–40 (daily TF) | 1–3 trades = not statistically valid |
| Max Drawdown | 15–35% | >50% = too risky |
| Sharpe Ratio | 0.8–2.0 | <0.5 = not worth the risk |

---

### 🏆 My Recommendation

**TRUST LEVEL: Use as a SIGNAL GENERATOR, not fully automated trader.**

```
DO:
✅ Use to identify Elliott Wave setups
✅ Use multi-TF analysis for confluence
✅ Use backtesting to validate a config
✅ Enable stale signal guard always
✅ Paper trade for 30+ trades before going live
✅ Use as one input among multiple analysis tools

DON'T:
❌ Fully automate with real money immediately
❌ Trust a config with <15 backtest trades
❌ Use on 5m/15m without monitoring the screen
❌ Ignore fundamental context (earnings, events)
❌ Risk >1-2% of capital per trade
❌ Treat 70% backtest win rate as 70% live win rate
   (expect 10-20% degradation in live trading)
```

**Bottom Line:** This tool gives you an *edge*, not a *guarantee*. It automates pattern recognition that would otherwise take hours manually. But the market is not a formula — use this as a decision support tool, not a decision replacement tool.

---

### 🧪 How to Validate Before Going Live

1. **Run Full Auto** → get best config
2. **Run Backtest** → need 15+ trades, win rate 60%+, Sharpe 1.0+
3. **Paper trade for 2 weeks** using Manual Scan
4. **Compare live signals to backtest signals** — they should roughly match
5. **Only then** consider enabling Dhan integration with tiny qty
6. **Monitor for first 10 live trades** before trusting automation
        """)

    st.markdown("---")
    st.markdown("""<div style="background:rgba(0,229,255,.05);border:1px solid #00bcd4;border-radius:10px;padding:14px 18px;text-align:center">
    <b style="color:#00e5ff">Final Verdict</b><br>
    <span style="color:#78909c;font-size:.88rem">
    This app is a <b>research-grade signal generator</b> with serious infrastructure (8-factor scoring, stale guard, conservative fills).<br>
    It is <b>more reliable than most free tools</b>, but <b>not suitable for fully automated trading without human oversight</b>.<br>
    Use it to <b>find setups faster</b>, not to <b>replace your judgment</b>.
    </span>
    </div>""",unsafe_allow_html=True)

# ── HELP ───────────────────────────────────────────────────────────────────
with t_help:
    st.markdown("## 📖 Help — Elliott Wave Algo Trader v5.4")
    h1,h2=st.columns(2,gap="large")
    with h1:
        st.markdown(f"""
### 🛡️ Stale Signal Guard (New in v5.4)

**Problem**: yfinance data is delayed 15-20 min. When you see a signal, it may have fired 30 min ago. By now, price may have already hit SL or Target.

**Solution**: Enable **"Prevent Stale Signal Entry"** checkbox in sidebar.

**How it works:**
1. When signal fires on bar N, system checks bars N+1 onwards
2. If any later bar has Low ≤ SL (BUY) → signal is STALE, SL already hit
3. If any later bar has High ≥ Target (BUY) → signal is STALE, target passed
4. Stale signal is BLOCKED (no position opened)
5. System waits for next fresh signal

**Signal Snapshot panel** shows:
- When signal fired vs right now
- LTP at signal vs LTP now
- SL/Target then vs now if entering
- Whether signal is stale or fresh

---

### 📅 yfinance Period Limits (Fixed in v5.4)

| Interval | Max Period |
|----------|-----------|
| 1m | 7 days |
| 5m | **1 month** (was 3mo, now corrected) |
| 15m | **1 month** (was 3mo, now corrected) |
| 30m | 1 month |
| 1h | 2 years |
| 4h, 1d, 1wk | Unlimited |

5m/15m with 3mo was failing silently — now corrected.

---

### ⚡ Quick / Balanced / Deep Optimization

| Mode | Combos | Best for |
|------|--------|---------|
| Quick | 8 | First pass, rough idea |
| Balanced | 24 | Daily use (recommended) |
| Deep | 80 | Thorough research |

Min Trades Filter: set to ≥8 to avoid 1-trade 100% win rate nonsense.

---

### 🤖 Full Auto Mode Fixed (v5.4)

Now applies: TF, Period, Depth, SL, Target to sidebar + backtest + live.
Uses `apply_best_cfg()` which correctly sets all 5 parameters.
        """)
    with h2:
        st.markdown("""
### 📈 Accuracy Improvement Tips

1. **Use Daily (1d) timeframe** — cleanest Elliott Waves
2. **Min confidence ≥68%** (default) — rejects weak setups
3. **Min R:R ≥1.2** (enforced) — only asymmetric trades
4. **W3 > W1 rule enforced** — removes invalid patterns
5. **8-factor scoring** (RSI + StochRSI + MACD + EMA + Fib + BB + Volume + ATR)
6. **Multi-TF confluence** — signal on multiple TFs = stronger signal
7. **Golden ratio 61.8%** retracement = highest confidence

---

### ⚠️ Common Issues & Fixes

| Problem | Solution |
|---------|---------|
| 5m/15m no data for 3mo | Fixed in v5.4: max 1mo for these TFs |
| 100% win rate 1 trade | Set min trades filter ≥8 in Optimization |
| Full auto config not applied | Fixed: `apply_best_cfg()` sets all 5 params |
| Stale signal entered | Enable 🛡 Prevent Stale Signal Entry checkbox |
| Position opened on stale signal | Stale guard was OFF — enable it |
| Fragment error | `pip install streamlit --upgrade` |
| dhanhq missing | `pip install dhanhq` |

---

### 🔑 Credentials
Change at top of file:
```python
DEFAULT_CLIENT_ID    = "your_client_id"
DEFAULT_ACCESS_TOKEN = "your_token"
```

---

### 📊 Minimum Reliable Backtest

Before trusting a config:
- ≥15 trades (ideally 30+)
- Win rate ≥55%
- Profit Factor ≥1.3
- Sharpe Ratio ≥0.8
- Max Drawdown ≤35%
        """)
    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#37474f;font-size:.81rem;padding:8px">
    🌊 Elliott Wave Algo Trader v5.4 · Stale Guard · Quick/Deep Opt · Full Auto Fixed ·
    <b style="color:#f44336">Not financial advice. Paper trade first. Use Stop Loss.</b>
    </div>""",unsafe_allow_html=True)
