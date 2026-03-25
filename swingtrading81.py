"""
╔══════════════════════════════════════════════════════════════╗
║       🌊  ELLIOTT WAVE ALGO TRADER  v5.0                     ║
║  Full Transparency · Auto-Reverse · IST Times · Wave State  ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run elliott_wave_algo_trader.py
pip install streamlit yfinance plotly pandas numpy requests
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

# ─── IST timezone ─────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> datetime:
    return datetime.now(IST)

def to_ist_str(dt) -> str:
    """Convert any datetime / pandas Timestamp to IST string."""
    if dt is None:
        return "—"
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
.main-hdr p{color:#546e7a;margin:4px 0 0;font-size:.85rem;}

/* Status banners */
.status-scanning{background:rgba(255,179,0,.10);border:2px solid #ffb300;border-radius:10px;padding:14px 18px;}
.status-signal  {background:rgba(0,229,255,.10);border:2px solid #00e5ff;border-radius:10px;padding:14px 18px;}
.status-buy     {background:rgba(76,175,80,.12) ;border:2px solid #4caf50;border-radius:10px;padding:14px 18px;}
.status-sell    {background:rgba(244,67,54,.12) ;border:2px solid #f44336;border-radius:10px;padding:14px 18px;}
.status-stopped {background:rgba(80,80,100,.10) ;border:2px solid #455a64;border-radius:10px;padding:14px 18px;}
.status-reverse {background:rgba(171,71,188,.14);border:2px solid #ab47bc;border-radius:10px;padding:14px 18px;}

.wave-card{background:#060d14;border:1px solid #1e3a5f;border-radius:8px;
  padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:.82rem;line-height:1.9;}
.info-box{background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
  padding:12px 14px;font-size:.86rem;line-height:1.9;}
.ltp-box{background:#060d14;border:1px solid #0d3349;border-radius:8px;
  padding:10px 14px;text-align:center;}
.best-cfg{background:rgba(0,229,255,.07);border:1px solid #00bcd4;
  border-radius:10px;padding:12px 16px;margin:6px 0;}

.sig-buy {background:rgba(76,175,80,.12);border:1.5px solid #4caf50;border-radius:10px;padding:12px 14px;}
.sig-sell{background:rgba(244,67,54,.12);border:1.5px solid #f44336;border-radius:10px;padding:12px 14px;}
.sig-hold{background:rgba(100,100,120,.10);border:1.5px solid #455a64;border-radius:10px;padding:12px 14px;}

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
# TICKER GROUPS
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
    "₿ Crypto": {
        "Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD",
        "Solana":"SOL-USD","XRP":"XRP-USD","Cardano":"ADA-USD",
    },
    "💱 Forex": {
        "USD/INR":"USDINR=X","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X",
        "USD/JPY":"JPY=X","AUD/USD":"AUDUSD=X","EUR/INR":"EURINR=X",
    },
    "🥇 Commodities": {
        "Gold":"GC=F","Silver":"SI=F","Crude Oil WTI":"CL=F",
        "Natural Gas":"NG=F","Copper":"HG=F",
    },
    "🌐 US Stocks": {
        "Apple":"AAPL","Tesla":"TSLA","NVIDIA":"NVDA",
        "Microsoft":"MSFT","Alphabet":"GOOGL","Meta":"META",
    },
    "✏️ Custom Ticker": {"Custom":"__CUSTOM__"},
}

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d","1wk"]
PERIODS    = ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]
VALID_PERIODS = {
    "1m" :["1d","5d","7d"],
    "5m" :["1d","5d","7d","1mo","3mo"],
    "15m":["1d","5d","7d","1mo","3mo"],
    "30m":["1d","5d","7d","1mo","3mo"],
    "1h" :["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h" :["1mo","3mo","6mo","1y","2y","5y"],
    "1d" :["1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["3mo","6mo","1y","2y","5y","10y","20y"],
}
MTF_COMBOS = [
    ("1d","1y","Daily (1y)"),("4h","3mo","4-Hour (3mo)"),
    ("1h","1mo","1-Hour (1mo)"),("15m","5d","15-Min (5d)"),
]

# SL options – special sentinels
SL_WAVE_AUTO    = "wave_auto"
SL_CUSTOM_PTS   = "__custom_pts__"
SL_SIG_REVERSE  = "__sig_reverse__"

SL_MAP = {
    "Wave Auto (Pivot Low/High)": SL_WAVE_AUTO,
    "0.5%":0.005,"1%":0.01,"1.5%":0.015,
    "2%":0.02,"2.5%":0.025,"3%":0.03,"5%":0.05,
    "Custom Points (enter below)": SL_CUSTOM_PTS,
    "Exit on Signal Reverse": SL_SIG_REVERSE,
}
TGT_MAP = {
    "Wave Auto (Fib 1.618×W1)": "wave_auto",
    "R:R 1:1":1.0,"R:R 1:1.5":1.5,"R:R 1:2":2.0,
    "R:R 1:2.5":2.5,"R:R 1:3":3.0,
    "Fib 1.618×Wave 1":"fib_1618","Fib 2.618×Wave 1":"fib_2618",
    "Custom Points (enter below)": "__custom_pts__",
    "Exit on Signal Reverse": "__sig_reverse__",
}
SL_KEYS  = list(SL_MAP.keys())
TGT_KEYS = list(TGT_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    # live engine
    "live_running":     False,
    "live_status":      "stopped",   # stopped|scanning|signal_fired|pos_buy|pos_sell|reversing
    "live_signals":     [],
    "live_log":         [],
    "last_bar_ts":      None,
    "live_position":    None,
    "live_pnl":         0.0,
    "live_trades":      [],
    # transparency data
    "live_ltp":         None,
    "live_ltp_prev":    None,
    "live_last_candle_ist": None,
    "live_delay_s":     0,
    "live_wave_state":  None,   # dict: current_wave, next_wave, fib_levels, action_text
    "live_sig_pending": None,   # signal that fired, waiting for next bar entry
    "live_no_pos_reason": "",
    "live_last_sig":    None,
    "live_next_check_ist": None,
    # scan
    "_scan_sig": None,
    "_scan_df":  None,
    # backtest
    "bt_results": None,
    # opt
    "opt_results": None,
    # analysis
    "_analysis_results": None,
    "_analysis_overall": "HOLD",
    "_analysis_symbol":  "",
    # applied best config
    "applied_depth":   5,
    "applied_sl_lbl":  "Wave Auto (Pivot Low/High)",
    "applied_tgt_lbl": "Wave Auto (Fib 1.618×W1)",
    "best_cfg_applied": False,
    # custom points
    "custom_sl_pts":  50.0,
    "custom_tgt_pts": 100.0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════
# RATE-LIMIT SAFE FETCH
# ═══════════════════════════════════════════════════════════════════════════
_fetch_lock    = threading.Lock()
_last_fetch_ts = [0.0]

def fetch_ohlcv(symbol:str, interval:str, period:str,
                min_delay:float=1.5) -> Optional[pd.DataFrame]:
    with _fetch_lock:
        gap = time.time() - _last_fetch_ts[0]
        if gap < min_delay:
            time.sleep(min_delay - gap)
        try:
            df = yf.download(symbol, interval=interval, period=period,
                             progress=False, auto_adjust=True)
            _last_fetch_ts[0] = time.time()
        except Exception:
            _last_fetch_ts[0] = time.time()
            return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna(subset=["Open","High","Low","Close"])
    df.index = pd.to_datetime(df.index)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def add_indicators(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["EMA_20"]  = c.ewm(span=20,  adjust=False).mean()
    df["EMA_50"]  = c.ewm(span=50,  adjust=False).mean()
    df["EMA_200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100/(1+rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    if "Volume" in df.columns:
        df["Vol_Avg"] = df["Volume"].rolling(20).mean()
    return df

# ═══════════════════════════════════════════════════════════════════════════
# PIVOTS
# ═══════════════════════════════════════════════════════════════════════════
def find_pivots(df:pd.DataFrame, depth:int=5) -> list:
    H,L,n = df["High"].values, df["Low"].values, len(df)
    raw = []
    for i in range(depth, n-depth):
        wh = H[max(0,i-depth):i+depth+1]
        wl = L[max(0,i-depth):i+depth+1]
        if H[i] == wh.max():
            raw.append((i, float(H[i]), "H"))
        elif L[i] == wl.min():
            raw.append((i, float(L[i]), "L"))
    clean = []
    for p in raw:
        if not clean or clean[-1][2] != p[2]:
            clean.append(list(p))
        else:
            if p[2]=="H" and p[1]>clean[-1][1]: clean[-1]=list(p)
            elif p[2]=="L" and p[1]<clean[-1][1]: clean[-1]=list(p)
    return [tuple(x) for x in clean]

# ═══════════════════════════════════════════════════════════════════════════
# WAVE STATE ANALYZER  (transparency: tell user which wave they are in)
# ═══════════════════════════════════════════════════════════════════════════
def analyze_wave_state(df:pd.DataFrame, pivots:list, sig:dict) -> dict:
    """
    Returns a dict describing current wave position, next expected move,
    Fibonacci levels, and recommended action text.
    """
    if not pivots or len(df) < 10:
        return {
            "current_wave": "Unknown", "next_wave": "—",
            "direction": "—", "fib_levels": {},
            "action": "Insufficient data to determine wave position.",
            "auto_action": "Wait",
        }

    cur = float(df["Close"].iloc[-1])
    last_p = pivots[-1]
    state  = {}

    if sig["signal"] == "BUY":
        p2 = sig.get("wave_pivots", [None,None,last_p])[2]
        p1 = sig.get("wave_pivots", [None,last_p,None])[1]
        p0 = sig.get("wave_pivots", [last_p,None,None])[0]
        w1 = p1[1] - p0[1] if p0 and p1 else cur*0.02
        w2_low = p2[1] if p2 else cur*0.99
        fibs = {
            "W3 (1.618×W1)": round(w2_low + w1*1.618, 2),
            "W3 (1.000×W1)": round(w2_low + w1*1.000, 2),
            "W3 (2.618×W1)": round(w2_low + w1*2.618, 2),
        }
        state = {
            "current_wave": "Wave 2 bottom (entry zone)",
            "next_wave": "Wave 3 UP (strongest impulse)",
            "direction": "BULLISH",
            "fib_levels": fibs,
            "action": (
                f"✅ **Wave-2 bottom detected.** Price is at the ideal entry zone for Wave-3 up.\n\n"
                f"**Auto action**: BUY order will be placed (or is already placed) at market price.\n"
                f"**SL**: Below Wave-2 low at {w2_low:.2f}\n"
                f"**Wave-3 targets**: {fibs['W3 (1.618×W1)']:.2f} (1.618×) · {fibs['W3 (2.618×W1)']:.2f} (2.618×)\n\n"
                f"**What happens next**: Hold position as Wave-3 develops. "
                f"If a SELL signal fires, position will **auto-close and reverse**."
            ),
            "auto_action": "BUY",
        }

    elif sig["signal"] == "SELL":
        p2 = sig.get("wave_pivots", [None,None,last_p])[2]
        p1 = sig.get("wave_pivots", [None,last_p,None])[1]
        p0 = sig.get("wave_pivots", [last_p,None,None])[0]
        w1 = p0[1] - p1[1] if p0 and p1 else cur*0.02
        w2_high = p2[1] if p2 else cur*1.01
        fibs = {
            "W3 (1.618×W1)": round(w2_high - w1*1.618, 2),
            "W3 (1.000×W1)": round(w2_high - w1*1.000, 2),
            "W3 (2.618×W1)": round(w2_high - w1*2.618, 2),
        }
        state = {
            "current_wave": "Wave 2 top (entry zone)",
            "next_wave": "Wave 3 DOWN (strongest impulse)",
            "direction": "BEARISH",
            "fib_levels": fibs,
            "action": (
                f"🔴 **Wave-2 top detected.** Price is at the ideal entry zone for Wave-3 down.\n\n"
                f"**Auto action**: SELL order placed at market price.\n"
                f"**SL**: Above Wave-2 high at {w2_high:.2f}\n"
                f"**Wave-3 targets**: {fibs['W3 (1.618×W1)']:.2f} (1.618×) · {fibs['W3 (2.618×W1)']:.2f} (2.618×)\n\n"
                f"**What happens next**: Hold short position as Wave-3 down develops. "
                f"If a BUY signal fires, position will **auto-close and reverse**."
            ),
            "auto_action": "SELL",
        }

    else:
        # Determine position in wave cycle from pivots
        if len(pivots) >= 2:
            lp  = pivots[-1]
            lp2 = pivots[-2]
            move = "up" if lp[2] == "H" else "down"
            retrace_pct = 0
            if len(pivots) >= 3:
                p_a, p_b, p_c = pivots[-3], pivots[-2], pivots[-1]
                if p_b[1] != p_a[1]:
                    retrace_pct = abs(p_c[1]-p_b[1]) / abs(p_b[1]-p_a[1]) * 100
            state = {
                "current_wave": f"In progress — last pivot was a {'High' if lp[2]=='H' else 'Low'} ({move} leg)",
                "next_wave": "Waiting for Wave-2 retracement to complete",
                "direction": "NEUTRAL",
                "fib_levels": {},
                "action": (
                    f"⏸ **No actionable Elliott Wave signal yet.**\n\n"
                    f"Last pivot: {'High' if lp[2]=='H' else 'Low'} at {lp[1]:.2f}  "
                    f"(bar {lp[0]})\n"
                    + (f"Current retracement: {retrace_pct:.1f}% of previous wave\n\n" if retrace_pct else "\n")
                    + f"**What to do**: Wait. The system scans every candle automatically.\n"
                    f"A signal will fire when Wave-2 completes a valid pivot at "
                    f"38.2%–88.6% retracement.\n"
                    f"**No manual intervention needed — everything is automatic.**"
                ),
                "auto_action": "Wait",
            }
        else:
            state = {
                "current_wave": "Insufficient pivot data",
                "next_wave": "—", "direction": "NEUTRAL", "fib_levels": {},
                "action": "Collecting data… increase Period or reduce Pivot Depth.",
                "auto_action": "Wait",
            }

    return state

# ═══════════════════════════════════════════════════════════════════════════
# CORE SIGNAL  (identical for backtest + live)
# ═══════════════════════════════════════════════════════════════════════════
def _blank(reason:str="") -> dict:
    return {
        "signal":"HOLD","entry_price":None,"sl":None,"target":None,
        "confidence":0.0,"reason":reason or "No Elliott Wave pattern detected",
        "pattern":"—","wave_pivots":None,"wave1_len":0.0,"retracement":0.0,
    }

def ew_signal(df:pd.DataFrame, depth:int=5,
              sl_type="wave_auto", tgt_type="wave_auto",
              custom_sl_pts:float=50.0, custom_tgt_pts:float=100.0) -> dict:
    n = len(df)
    if n < max(30, depth*4): return _blank("Insufficient bars")
    pivots = find_pivots(df, depth)
    if len(pivots) < 4: return _blank("Not enough pivots — try smaller Pivot Depth")

    cur = float(df["Close"].iloc[-1])
    best, best_conf = _blank(), 0.0

    for i in range(len(pivots)-2):
        p0,p1,p2 = pivots[i], pivots[i+1], pivots[i+2]
        bars_since = n-1-p2[0]

        # BUY: Low→High→Low
        if p0[2]=="L" and p1[2]=="H" and p2[2]=="L":
            w1 = p1[1]-p0[1]
            if w1<=0: continue
            retr = (p1[1]-p2[1])/w1
            if not (0.236<=retr<=0.886 and p2[1]>p0[1] and bars_since<=depth*4): continue
            conf = 0.50
            if 0.382<=retr<=0.618: conf=0.65
            if 0.50 <=retr<=0.786: conf=0.72
            if abs(retr-0.618)<0.04: conf=0.87
            if abs(retr-0.382)<0.03: conf=0.75
            if i+3<len(pivots) and pivots[i+3][2]=="H":
                w3=pivots[i+3][1]-p2[1]
                if w3>w1: conf=min(conf+0.08,0.95)
                if abs(w3/w1-1.618)<0.20: conf=min(conf+0.05,0.98)
            if conf<=best_conf: continue
            entry=cur
            if sl_type==SL_WAVE_AUTO:
                sl_=p2[1]*0.998
            elif sl_type==SL_CUSTOM_PTS:
                sl_=entry-custom_sl_pts
            elif sl_type==SL_SIG_REVERSE:
                sl_=entry*(1-0.05)   # wide fallback for backtest; live uses signal reverse
            else:
                sl_=entry*(1-float(sl_type))
            risk=entry-sl_
            if risk<=0: continue
            tgt_=_calc_target(tgt_type, entry,"BUY",w1,risk, custom_tgt_pts)
            if tgt_<=entry: continue
            best_conf=conf
            best={
                "signal":"BUY","entry_price":entry,"sl":sl_,"target":tgt_,
                "confidence":conf,"retracement":retr,
                "reason":f"Wave-2 bottom: {retr:.1%} retrace → Wave-3 up",
                "pattern":f"W2 Bottom ({retr:.1%})",
                "wave_pivots":[p0,p1,p2],"wave1_len":w1,
            }

        # SELL: High→Low→High
        elif p0[2]=="H" and p1[2]=="L" and p2[2]=="H":
            w1=p0[1]-p1[1]
            if w1<=0: continue
            retr=(p2[1]-p1[1])/w1
            if not (0.236<=retr<=0.886 and p2[1]<p0[1] and bars_since<=depth*4): continue
            conf=0.50
            if 0.382<=retr<=0.618: conf=0.65
            if 0.50 <=retr<=0.786: conf=0.72
            if abs(retr-0.618)<0.04: conf=0.87
            if abs(retr-0.382)<0.03: conf=0.75
            if i+3<len(pivots) and pivots[i+3][2]=="L":
                w3=p2[1]-pivots[i+3][1]
                if w3>w1: conf=min(conf+0.08,0.95)
                if abs(w3/w1-1.618)<0.20: conf=min(conf+0.05,0.98)
            if conf<=best_conf: continue
            entry=cur
            if sl_type==SL_WAVE_AUTO:
                sl_=p2[1]*1.002
            elif sl_type==SL_CUSTOM_PTS:
                sl_=entry+custom_sl_pts
            elif sl_type==SL_SIG_REVERSE:
                sl_=entry*(1+0.05)
            else:
                sl_=entry*(1+float(sl_type))
            risk=sl_-entry
            if risk<=0: continue
            tgt_=_calc_target(tgt_type, entry,"SELL",w1,risk, custom_tgt_pts)
            if tgt_>=entry: continue
            best_conf=conf
            best={
                "signal":"SELL","entry_price":entry,"sl":sl_,"target":tgt_,
                "confidence":conf,"retracement":retr,
                "reason":f"Wave-2 top: {retr:.1%} retrace → Wave-3 down",
                "pattern":f"W2 Top ({retr:.1%})",
                "wave_pivots":[p0,p1,p2],"wave1_len":w1,
            }
    return best

def _calc_target(tgt_type, entry:float, direction:str,
                 w1:float, risk:float, custom_pts:float=100.0) -> float:
    sign = 1 if direction=="BUY" else -1
    if tgt_type in ("wave_auto","fib_1618"): return entry+sign*w1*1.618
    elif tgt_type=="fib_2618":              return entry+sign*w1*2.618
    elif tgt_type=="__custom_pts__":        return entry+sign*custom_pts
    elif tgt_type=="__sig_reverse__":       return entry+sign*w1*1.618  # fallback
    elif isinstance(tgt_type,(int,float)):  return entry+sign*risk*float(tgt_type)
    return entry+sign*risk*2.0

# ═══════════════════════════════════════════════════════════════════════════
# HELPER: sl/tgt label lookups for applied config
# ═══════════════════════════════════════════════════════════════════════════
def sl_label_from_val(val) -> str:
    for k,v in SL_MAP.items():
        if str(v)==str(val): return k
    return SL_KEYS[0]

def tgt_label_from_val(val) -> str:
    for k,v in TGT_MAP.items():
        if str(v)==str(val): return k
    try:
        fv=float(val)
        for k,v in TGT_MAP.items():
            if isinstance(v,float) and abs(v-fv)<0.01: return k
    except Exception: pass
    return TGT_KEYS[0]

# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df:pd.DataFrame, depth:int=5,
                 sl_type="wave_auto", tgt_type="wave_auto",
                 capital:float=100_000.0,
                 custom_sl_pts:float=50.0,
                 custom_tgt_pts:float=100.0) -> dict:
    MIN_BARS = max(30, depth*4)
    if len(df) < MIN_BARS+10:
        return {"error":f"Need ≥{MIN_BARS+10} bars. Use longer period or smaller depth."}

    trades, equity_curve = [], [capital]
    equity, pos = capital, None

    for i in range(MIN_BARS, len(df)-1):
        bar_df   = df.iloc[:i+1]
        next_bar = df.iloc[i+1]
        hi_i     = float(df.iloc[i]["High"])
        lo_i     = float(df.iloc[i]["Low"])
        this_sig = None  # for signal-reverse exit logic

        if pos:
            # For signal-reverse exit, evaluate signal BEFORE checking price
            if sl_type==SL_SIG_REVERSE or tgt_type=="__sig_reverse__":
                this_sig = ew_signal(bar_df, depth, sl_type, tgt_type, custom_sl_pts, custom_tgt_pts)

            exit_p, exit_r, exit_hi, exit_lo = None, None, hi_i, lo_i

            # Conservative: check SL with Low(BUY)/High(SELL) first
            if pos["type"]=="BUY":
                if lo_i <= pos["sl"]:       exit_p,exit_r = pos["sl"],   "SL (Low≤SL)"
                elif hi_i >= pos["target"]: exit_p,exit_r = pos["target"],"Target (High≥Target)"
                # Signal reverse exit
                elif (sl_type==SL_SIG_REVERSE or tgt_type=="__sig_reverse__") \
                     and this_sig and this_sig["signal"]=="SELL":
                    exit_p,exit_r = float(df.iloc[i]["Close"]),"Signal Reverse"
            else:
                if hi_i >= pos["sl"]:       exit_p,exit_r = pos["sl"],   "SL (High≥SL)"
                elif lo_i <= pos["target"]: exit_p,exit_r = pos["target"],"Target (Low≤Target)"
                elif (sl_type==SL_SIG_REVERSE or tgt_type=="__sig_reverse__") \
                     and this_sig and this_sig["signal"]=="BUY":
                    exit_p,exit_r = float(df.iloc[i]["Close"]),"Signal Reverse"

            if exit_p is not None:
                qty = pos["qty"]
                pnl = (exit_p-pos["entry"])*qty if pos["type"]=="BUY" \
                      else (pos["entry"]-exit_p)*qty
                equity += pnl
                equity_curve.append(equity)
                trades.append({
                    "Entry Time": pos["entry_time"],"Exit Time": df.index[i],
                    "Type": pos["type"],
                    "Entry": round(pos["entry"],2),"Exit": round(exit_p,2),
                    "SL": round(pos["sl"],2),"Target": round(pos["target"],2),
                    "Exit Bar Low": round(lo_i,2),"Exit Bar High": round(hi_i,2),
                    "Exit Reason": exit_r,
                    "PnL Rs": round(pnl,2),
                    "PnL %": round(pnl/(pos["entry"]*qty)*100,2),
                    "Equity Rs": round(equity,2),
                    "Bars Held": i-pos["entry_bar"],
                    "Confidence": round(pos["conf"],2),
                })
                pos = None

        if pos is None:
            sig = ew_signal(bar_df, depth, sl_type, tgt_type, custom_sl_pts, custom_tgt_pts)
            if sig["signal"] in ("BUY","SELL"):
                ep   = float(next_bar["Open"])
                w1   = sig.get("wave1_len", ep*0.02) or (ep*0.02)
                if sl_type==SL_WAVE_AUTO:
                    sl_=sig["sl"]
                elif sl_type==SL_CUSTOM_PTS:
                    sl_=ep-custom_sl_pts if sig["signal"]=="BUY" else ep+custom_sl_pts
                elif sl_type==SL_SIG_REVERSE:
                    sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
                else:
                    sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" \
                        else ep*(1+float(sl_type))
                risk=abs(ep-sl_)
                if risk<=0: continue
                tgt_=_calc_target(tgt_type, ep, sig["signal"], w1, risk, custom_tgt_pts)
                if sig["signal"]=="BUY"  and tgt_<=ep: continue
                if sig["signal"]=="SELL" and tgt_>=ep: continue
                qty=max(1, int(equity*0.95/ep))
                pos={
                    "type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,
                    "entry_bar":i+1,"entry_time":df.index[i+1],
                    "qty":qty,"conf":sig["confidence"],
                }

    if pos:
        ep2=float(df["Close"].iloc[-1])
        qty=pos["qty"]
        pnl=(ep2-pos["entry"])*qty if pos["type"]=="BUY" else (pos["entry"]-ep2)*qty
        equity+=pnl
        trades.append({
            "Entry Time":pos["entry_time"],"Exit Time":df.index[-1],
            "Type":pos["type"],
            "Entry":round(pos["entry"],2),"Exit":round(ep2,2),
            "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
            "Exit Bar Low":round(float(df["Low"].iloc[-1]),2),
            "Exit Bar High":round(float(df["High"].iloc[-1]),2),
            "Exit Reason":"Open@End",
            "PnL Rs":round(pnl,2),
            "PnL %":round(pnl/(pos["entry"]*qty)*100,2),
            "Equity Rs":round(equity,2),
            "Bars Held":len(df)-1-pos["entry_bar"],
            "Confidence":round(pos["conf"],2),
        })

    if not trades:
        return {"error":"No trades. Try smaller Pivot Depth, longer Period, or different SL/Target.",
                "equity_curve":equity_curve}

    tdf  = pd.DataFrame(trades)
    wins = tdf[tdf["PnL Rs"]>0]
    loss = tdf[tdf["PnL Rs"]<=0]
    ntot = len(tdf)
    wr   = len(wins)/ntot*100 if ntot else 0
    pf   = abs(wins["PnL Rs"].sum()/loss["PnL Rs"].sum()) \
           if len(loss) and loss["PnL Rs"].sum()!=0 else 9999.0
    eq_arr = np.array(equity_curve)
    peak   = np.maximum.accumulate(eq_arr)
    mdd    = float(((eq_arr-peak)/peak*100).min())
    rets   = tdf["PnL %"].values
    sharpe = float(rets.mean()/rets.std()*np.sqrt(252)) \
             if len(rets)>1 and rets.std()!=0 else 0.0

    # SL/Target hit verification text
    sl_hits = tdf[tdf["Exit Reason"].str.contains("SL",na=False)]
    tgt_hits= tdf[tdf["Exit Reason"].str.contains("Target",na=False)]
    rev_hits= tdf[tdf["Exit Reason"].str.contains("Reverse",na=False)]
    open_hits=tdf[tdf["Exit Reason"].str.contains("Open",na=False)]

    return {
        "trades":tdf,"equity_curve":equity_curve,
        "exit_breakdown":{
            "SL hits":len(sl_hits),"Target hits":len(tgt_hits),
            "Signal Reverse exits":len(rev_hits),"Still open":len(open_hits),
        },
        "metrics":{
            "Total Trades":ntot,"Win Rate %":round(wr,1),
            "Profit Factor":round(pf,2),
            "Total Return %":round((equity-capital)/capital*100,2),
            "Final Equity Rs":round(equity,2),
            "Max Drawdown %":round(mdd,2),"Sharpe Ratio":round(sharpe,2),
            "Avg Win Rs":round(float(wins["PnL Rs"].mean()),2) if len(wins) else 0.0,
            "Avg Loss Rs":round(float(loss["PnL Rs"].mean()),2) if len(loss) else 0.0,
            "Wins":len(wins),"Losses":len(loss),
        },
    }

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization(df:pd.DataFrame, capital:float=100_000.0,
                     custom_sl_pts:float=50.0, custom_tgt_pts:float=100.0) -> pd.DataFrame:
    DEPTHS   = [3,5,7,10]
    SL_OPTS  = [0.01,0.02,0.03,"wave_auto"]
    TGT_OPTS = [1.5,2.0,3.0,"wave_auto","fib_1618"]
    combos   = list(itertools.product(DEPTHS,SL_OPTS,TGT_OPTS))
    prog     = st.progress(0,text="Optimizing…")
    rows     = []
    for idx,(dep,sl,tgt) in enumerate(combos):
        r=run_backtest(df,depth=dep,sl_type=sl,tgt_type=tgt,capital=capital,
                       custom_sl_pts=custom_sl_pts,custom_tgt_pts=custom_tgt_pts)
        if "metrics" in r:
            m=r["metrics"]
            rows.append({
                "Depth":dep,"SL":str(sl),"Target":str(tgt),
                "Trades":m["Total Trades"],"Win %":m["Win Rate %"],
                "Return %":m["Total Return %"],"PF":m["Profit Factor"],
                "Max DD %":m["Max Drawdown %"],"Sharpe":m["Sharpe Ratio"],
            })
        prog.progress((idx+1)/len(combos),text=f"Combo {idx+1}/{len(combos)}…")
    prog.empty()
    if not rows: return pd.DataFrame()
    out=pd.DataFrame(rows)
    out["Score"]=(
        out["Return %"].clip(lower=0)
        *(out["Win %"]/100)
        *out["PF"].clip(upper=10)
        /(out["Max DD %"].abs()+1)
    )
    return out.sort_values("Score",ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# DHAN API
# ═══════════════════════════════════════════════════════════════════════════
class DhanAPI:
    BASE="https://api.dhan.co"
    def __init__(self,cid,tok):
        self.cid=cid
        self.hdrs={"Content-Type":"application/json","access-token":tok}
    def place_order(self,sec_id,segment,txn,qty,
                    order_type="MARKET",price=0.0,product="INTRADAY"):
        try:
            r=requests.post(f"{self.BASE}/orders",headers=self.hdrs,timeout=10,json={
                "dhanClientId":self.cid,"transactionType":txn,
                "exchangeSegment":segment,"productType":product,
                "orderType":order_type,"validity":"DAY",
                "securityId":sec_id,"quantity":qty,"price":price,
            })
            return r.json()
        except Exception as e: return {"error":str(e)}
    def fund_limit(self):
        try: return requests.get(f"{self.BASE}/fundlimit",headers=self.hdrs,timeout=10).json()
        except Exception as e: return {"error":str(e)}

# ═══════════════════════════════════════════════════════════════════════════
# LIVE TRADING LOOP  — fully transparent, auto-reverse, IST times
# ═══════════════════════════════════════════════════════════════════════════
def live_loop(symbol,interval,period,depth,sl_type,tgt_type,
              dhan_on,dhan_api,sec_id,live_qty,custom_sl_pts,custom_tgt_pts):

    POLL={"1m":60,"5m":300,"15m":900,"30m":1800,
          "1h":3600,"4h":3600,"1d":3600,"1wk":3600}
    sleep_s=min(POLL.get(interval,60),60)
    poll_label=f"{sleep_s}s" if sleep_s<120 else f"{sleep_s//60}m"

    def log(msg,lvl="INFO"):
        ts=now_ist().strftime("%H:%M:%S IST")
        if "live_log" in st.session_state:
            st.session_state.live_log.append(f"[{ts}][{lvl}] {msg}")
            st.session_state.live_log=st.session_state.live_log[-150:]

    def set_status(s): st.session_state.live_status=s

    def close_position(exit_price:float, reason:str):
        pos=st.session_state.live_position
        if not pos: return
        qty_=pos["qty"]
        pnl=(exit_price-pos["entry"])*qty_ if pos["type"]=="BUY" \
            else (pos["entry"]-exit_price)*qty_
        st.session_state.live_pnl+=pnl
        st.session_state.live_trades.append({
            "Time":now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
            "Symbol":symbol,"TF":interval,"Period":period,
            "Type":pos["type"],
            "Entry":round(pos["entry"],2),"Exit":round(exit_price,2),
            "SL":round(pos["sl"],2),"Target":round(pos["target"],2),
            "Qty":qty_,"PnL Rs":round(pnl,2),"Reason":reason,
        })
        if dhan_on and dhan_api:
            xt="SELL" if pos["type"]=="BUY" else "BUY"
            log(f"📤 Dhan exit: {dhan_api.place_order(sec_id,'NSE_EQ',xt,qty_)}")
        em="✅" if "Target" in reason else ("🔄" if "Reverse" in reason else "❌")
        log(f"{em} {pos['type']} closed @ {exit_price:.2f} | {reason} | Rs{pnl:.2f}")
        st.session_state.live_position=None

    log(f"🚀 Live started | {symbol} | {interval} | {period} | poll every {poll_label}")
    set_status("scanning")

    while st.session_state.get("live_running",False):
        try:
            curr_ist=now_ist()
            st.session_state.live_current_time_ist=curr_ist.strftime("%H:%M:%S IST")
            st.session_state.live_next_check_ist=(curr_ist+timedelta(seconds=sleep_s)).strftime("%H:%M:%S IST")

            log(f"📡 Fetching {symbol} [{interval}/{period}] — 1.5s rate-limit enforced")
            set_status("scanning")
            df=fetch_ohlcv(symbol,interval,period,min_delay=1.5)

            if df is None or len(df)<35:
                log("⚠️  No/insufficient data — will retry","WARN")
                st.session_state.live_no_pos_reason="⚠️ No data received from yfinance. Check symbol/interval/period."
                time.sleep(sleep_s)
                continue

            # LTP and timing
            ltp_prev=st.session_state.live_ltp
            ltp=float(df["Close"].iloc[-1])
            st.session_state.live_ltp_prev=ltp_prev
            st.session_state.live_ltp=ltp

            last_candle_ts=df.index[-2] if len(df)>=2 else df.index[-1]
            last_candle_ist_str=to_ist_str(last_candle_ts)
            st.session_state.live_last_candle_ist=last_candle_ist_str

            # Delay calculation
            try:
                if last_candle_ts.tzinfo is None:
                    last_candle_dt=last_candle_ts.to_pydatetime().replace(tzinfo=timezone.utc)
                else:
                    last_candle_dt=last_candle_ts.to_pydatetime()
                delay_s=int((curr_ist-last_candle_dt.astimezone(IST)).total_seconds())
                st.session_state.live_delay_s=delay_s
            except Exception:
                st.session_state.live_delay_s=0

            df_closed=df.iloc[:-1]
            latest_ts=str(df_closed.index[-1])

            # Duplicate bar guard
            if st.session_state.get("last_bar_ts")==latest_ts:
                next_chk=st.session_state.get("live_next_check_ist","—")
                log(f"⏭  Bar {latest_ts[-19:]} already processed. Next check ≈ {next_chk}")
                pos_now=st.session_state.live_position
                if pos_now:
                    set_status(f"pos_{pos_now['type'].lower()}")
                time.sleep(sleep_s)
                continue

            st.session_state.last_bar_ts=latest_ts

            # Get pivots + wave state for UI
            pivs=find_pivots(df_closed,depth)

            # ── Generate signal ──────────────────────────────────────────
            sig=ew_signal(df_closed,depth,sl_type,tgt_type,custom_sl_pts,custom_tgt_pts)
            st.session_state.live_last_sig=sig
            ws=analyze_wave_state(df_closed,pivs,sig)
            st.session_state.live_wave_state=ws

            pos=st.session_state.live_position

            # ── Auto-close if SL/Target hit or Signal Reversal ───────────
            if pos:
                hit=None; hit_reason=None
                if pos["type"]=="BUY":
                    # Conservative: check SL via low first
                    if ltp<=pos["sl"]:       hit=pos["sl"],   "SL Hit (price≤SL)"
                    elif ltp>=pos["target"]: hit=pos["target"],"Target Hit (price≥Target)"
                    # Auto-reverse on signal flip
                    elif sig["signal"]=="SELL":
                        hit=ltp,"Signal Reversed SELL — auto closing BUY"
                        set_status("reversing")
                else:
                    if ltp>=pos["sl"]:       hit=pos["sl"],   "SL Hit (price≥SL)"
                    elif ltp<=pos["target"]: hit=pos["target"],"Target Hit (price≤Target)"
                    elif sig["signal"]=="BUY":
                        hit=ltp,"Signal Reversed BUY — auto closing SELL"
                        set_status("reversing")

                # Signal reverse forced exit regardless of SL/TGT setting
                if "Reversed" in (hit[1] if hit else ""):
                    close_position(hit[0],hit[1])
                    pos=None
                    # Fall through to open new position below
                elif hit:
                    close_position(hit[0],hit[1])
                    pos=None
                    st.session_state.live_no_pos_reason=""

            # ── Open new position ────────────────────────────────────────
            if pos is None and sig["signal"] in ("BUY","SELL"):
                ep=ltp
                w1=sig.get("wave1_len",ep*0.02) or (ep*0.02)
                # SL
                if sl_type==SL_WAVE_AUTO:
                    sl_=sig["sl"]
                elif sl_type==SL_CUSTOM_PTS:
                    sl_=ep-custom_sl_pts if sig["signal"]=="BUY" else ep+custom_sl_pts
                elif sl_type==SL_SIG_REVERSE:
                    sl_=ep*(1-0.05) if sig["signal"]=="BUY" else ep*(1+0.05)
                else:
                    sl_=ep*(1-float(sl_type)) if sig["signal"]=="BUY" \
                        else ep*(1+float(sl_type))
                risk=abs(ep-sl_)
                if risk<=0:
                    st.session_state.live_no_pos_reason=f"⚠️ Risk=0: SL ({sl_:.2f}) too close to entry ({ep:.2f}). Adjust SL setting."
                    log("⚠️  Risk=0 — skipping entry","WARN")
                    time.sleep(sleep_s)
                    continue
                tgt_=_calc_target(tgt_type,ep,sig["signal"],w1,risk,custom_tgt_pts)
                if (sig["signal"]=="BUY" and tgt_<=ep) or (sig["signal"]=="SELL" and tgt_>=ep):
                    st.session_state.live_no_pos_reason=f"⚠️ Invalid target ({tgt_:.2f}) vs entry ({ep:.2f}). Adjust Target setting."
                    time.sleep(sleep_s)
                    continue

                st.session_state.live_position={
                    "type":sig["signal"],"entry":ep,"sl":sl_,"target":tgt_,
                    "qty":live_qty,"entry_time":now_ist().strftime("%H:%M:%S IST"),
                    "entry_ist":now_ist().strftime("%d-%b %H:%M:%S IST"),
                    "symbol":symbol,"pattern":sig["pattern"],
                    "confidence":sig["confidence"],
                }
                set_status(f"pos_{sig['signal'].lower()}")
                st.session_state.live_no_pos_reason=""
                st.session_state.live_signals.append({
                    "Time (IST)":now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
                    "Bar TS":to_ist_str(df_closed.index[-1]),
                    "Symbol":symbol,"TF":interval,"Period":period,
                    "Signal":sig["signal"],"Entry":round(ep,2),
                    "SL":round(sl_,2),"Target":round(tgt_,2),
                    "Conf":f"{sig['confidence']:.0%}",
                    "Pattern":sig["pattern"],
                })
                if dhan_on and dhan_api:
                    log(f"📤 Dhan: {dhan_api.place_order(sec_id,'NSE_EQ',sig['signal'],live_qty)}")
                em="🟢" if sig["signal"]=="BUY" else "🔴"
                log(f"{em} {sig['signal']} @ {ep:.2f} | SL {sl_:.2f} | T {tgt_:.2f} | {sig['confidence']:.0%} | {sig['pattern']}")

            elif pos is None and sig["signal"]=="HOLD":
                set_status("scanning")
                reason=sig.get("reason","No pattern")
                st.session_state.live_no_pos_reason=(
                    f"📊 No signal yet. {reason}\n\n"
                    f"System is scanning every {poll_label} automatically.\n"
                    f"No action needed — position will open automatically when Wave-2 completes."
                )
                log(f"⏸  HOLD @ {ltp:.2f} | {reason}")

            time.sleep(sleep_s)

        except Exception as exc:
            log(f"💥 {exc}","ERROR")
            time.sleep(sleep_s)

    set_status("stopped")
    log("🛑 Live trading stopped")

# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def chart_waves(df:pd.DataFrame, pivots:list,
                sig:Optional[dict]=None,
                trades:Optional[pd.DataFrame]=None,
                symbol:str="", tf_label:str="") -> go.Figure:
    sig=sig or _blank()
    df_ind=add_indicators(df)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                      row_heights=[0.60,0.20,0.20],vertical_spacing=0.02,
                      subplot_titles=("","Volume","RSI"))

    fig.add_trace(go.Candlestick(
        x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="Price",
        increasing=dict(line=dict(color="#26a69a"),fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"),fillcolor="#ef5350"),
    ),row=1,col=1)

    for col,clr,nm in [("EMA_20","#ffb300","EMA20"),("EMA_50","#ab47bc","EMA50"),("EMA_200","#ef5350","EMA200")]:
        if col in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind[col],mode="lines",
                                     line=dict(color=clr,width=1.2),name=nm,opacity=0.7),row=1,col=1)

    if "Volume" in df.columns:
        vcol=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vcol,opacity=0.4,
                             name="Vol",showlegend=False),row=2,col=1)

    if "RSI" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind["RSI"],mode="lines",
                                 line=dict(color="#00e5ff",width=1.5),name="RSI",showlegend=False),row=3,col=1)
        fig.add_hline(y=70,line=dict(dash="dot",color="#ef5350",width=1),row=3,col=1)
        fig.add_hline(y=30,line=dict(dash="dot",color="#4caf50",width=1),row=3,col=1)

    vp=[p for p in pivots if p[0]<len(df)]
    if vp:
        fig.add_trace(go.Scatter(
            x=[df.index[p[0]] for p in vp],y=[p[1] for p in vp],
            mode="lines+markers",
            line=dict(color="rgba(255,180,0,.5)",width=1.5,dash="dot"),
            marker=dict(size=7,
                        color=["#4caf50" if p[2]=="L" else "#f44336" for p in vp],
                        symbol=["triangle-up" if p[2]=="L" else "triangle-down" for p in vp]),
            name="ZigZag",
        ),row=1,col=1)

    wp=sig.get("wave_pivots")
    if wp:
        valid_wp=[p for p in wp if p[0]<len(df)]
        if valid_wp:
            clr="#00e5ff" if sig["signal"]=="BUY" else "#ff4081"
            lbls=["W0","W1","W2"][:len(valid_wp)]
            fig.add_trace(go.Scatter(
                x=[df.index[p[0]] for p in valid_wp],y=[p[1] for p in valid_wp],
                mode="lines+markers+text",
                line=dict(color=clr,width=2.5),
                marker=dict(size=12,color=clr),
                text=lbls,textposition="top center",
                textfont=dict(color=clr,size=12,family="Share Tech Mono"),
                name="EW Pattern",
            ),row=1,col=1)

    if sig["signal"] in ("BUY","SELL"):
        sc="#4caf50" if sig["signal"]=="BUY" else "#f44336"
        ss="triangle-up" if sig["signal"]=="BUY" else "triangle-down"
        fig.add_trace(go.Scatter(x=[df.index[-1]],y=[df["Close"].iloc[-1]],mode="markers",
                                 marker=dict(size=20,color=sc,symbol=ss,line=dict(color="white",width=1.5)),
                                 name=f"▶ {sig['signal']}"),row=1,col=1)
        if sig.get("sl"):
            fig.add_hline(y=sig["sl"],line=dict(dash="dash",color="#ff7043",width=1.5),
                          annotation_text="  SL",annotation_position="right",row=1,col=1)
        if sig.get("target"):
            fig.add_hline(y=sig["target"],line=dict(dash="dash",color="#66bb6a",width=1.5),
                          annotation_text="  Target",annotation_position="right",row=1,col=1)

    if trades is not None and not trades.empty:
        for ttype,sym_,clr in [("BUY","triangle-up","#4caf50"),("SELL","triangle-down","#f44336")]:
            sub=trades[trades["Type"]==ttype]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Entry Time"],y=sub["Entry"],mode="markers",
                                         marker=dict(size=9,color=clr,symbol=sym_,
                                                     line=dict(color="white",width=0.8)),
                                         name=f"{ttype} Entry"),row=1,col=1)
        for rsn_,sym_,clr in [("Target","circle","#66bb6a"),("SL","x","#ef5350"),("Reverse","diamond","#ab47bc")]:
            sub=trades[trades["Exit Reason"].str.contains(rsn_,na=False)]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Exit Time"],y=sub["Exit"],mode="markers",
                                         marker=dict(size=7,color=clr,symbol=sym_),
                                         name=f"Exit({rsn_})",visible="legendonly"),row=1,col=1)

    title_str=f"🌊 {symbol}"+(f" · {tf_label}" if tf_label else "")
    fig.update_layout(title=dict(text=title_str,font=dict(size=14,color="#00e5ff")),
                      template="plotly_dark",height=580,
                      xaxis_rangeslider_visible=False,
                      plot_bgcolor="#06101a",paper_bgcolor="#06101a",
                      font=dict(color="#b0bec5",family="Exo 2"),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,font=dict(size=10)),
                      margin=dict(l=10,r=70,t=50,b=10))
    return fig

def chart_equity(equity_curve:list) -> go.Figure:
    eq=np.array(equity_curve,dtype=float)
    pk=np.maximum.accumulate(eq)
    dd=(eq-pk)/pk*100
    fig=make_subplots(rows=2,cols=1,row_heights=[0.65,0.35],vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=eq,mode="lines",name="Equity",
                             line=dict(color="#00bcd4",width=2),
                             fill="tozeroy",fillcolor="rgba(0,188,212,.07)"),row=1,col=1)
    fig.add_trace(go.Scatter(y=dd,mode="lines",name="Drawdown %",
                             line=dict(color="#f44336",width=1.5),
                             fill="tozeroy",fillcolor="rgba(244,67,54,.12)"),row=2,col=1)
    fig.add_hline(y=0,line=dict(dash="dot",color="#546e7a",width=1),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=360,
                      plot_bgcolor="#06101a",paper_bgcolor="#06101a",
                      font=dict(color="#b0bec5",family="Exo 2"),
                      margin=dict(l=10,r=10,t=20,b=10))
    return fig

def chart_opt_scatter(opt_df:pd.DataFrame) -> go.Figure:
    fig=go.Figure(go.Scatter(
        x=opt_df["Max DD %"].abs(),y=opt_df["Return %"],mode="markers",
        marker=dict(
            size=(opt_df["Win %"]/5).clip(lower=4),
            color=opt_df["Score"],colorscale="Plasma",showscale=True,
            colorbar=dict(title=dict(text="Score",font=dict(color="#b0bec5",size=12)),
                          tickfont=dict(color="#b0bec5")),
            line=dict(color="rgba(255,255,255,.2)",width=0.5),
        ),
        text=[f"Depth={r.Depth} SL={r.SL} T={r.Target}" for _,r in opt_df.iterrows()],
        hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%  MaxDD %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Return vs Max Drawdown (bubble = Win Rate)",font=dict(size=13,color="#00e5ff")),
        xaxis_title="Max Drawdown %",yaxis_title="Total Return %",
        template="plotly_dark",height=400,
        plot_bgcolor="#06101a",paper_bgcolor="#06101a",
        font=dict(color="#b0bec5",family="Exo 2"),
        margin=dict(l=10,r=10,t=45,b=10))
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MTF SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def generate_mtf_summary(symbol:str, results:list, overall_sig:str) -> str:
    lines=[f"## 🌊 Elliott Wave Analysis — {symbol}",
           f"*{now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}*\n"]
    buy_c =sum(1 for r in results if r["signal"]["signal"]=="BUY")
    sell_c=sum(1 for r in results if r["signal"]["signal"]=="SELL")
    hold_c=sum(1 for r in results if r["signal"]["signal"]=="HOLD")
    v_icon={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(overall_sig,"⚪")
    lines.append(f"### {v_icon} Overall: **{overall_sig}**")
    lines.append(f"*{buy_c} BUY · {sell_c} SELL · {hold_c} HOLD across {len(results)} timeframes*\n")
    if overall_sig=="BUY":
        lines.append("**📈 Bullish consensus.** Wave-3 up likely. Enter BUY on pullbacks. SL below Wave-2 pivot.")
    elif overall_sig=="SELL":
        lines.append("**📉 Bearish consensus.** Wave-3 down likely. Enter SELL on bounces. SL above Wave-2 pivot.")
    else:
        lines.append("**⚠️ No consensus.** Wait on sidelines for ≥70% confidence signal.")
    lines.append("\n---\n### 📊 Timeframe Breakdown\n")
    for r in results:
        sig=r["signal"]; s=sig["signal"]
        em={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(s,"⚪")
        lines.append(f"#### {em} {r['tf_name']}")
        if s in ("BUY","SELL"):
            retr=sig.get("retracement",0); conf=sig["confidence"]
            ep=sig["entry_price"]; sl_=sig["sl"]; tgt_=sig["target"]
            w1=sig.get("wave1_len",0)
            rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
            lines.append(f"- **Signal**: {s} | Pattern: {sig['pattern']}")
            lines.append(f"- Entry: {ep:.2f} | SL: {sl_:.2f} | Target: {tgt_:.2f} | R:R 1:{rr:.1f}")
            lines.append(f"- Confidence: {conf:.0%} | Retracement: {retr:.1%} of Wave-1")
            if w1>0: lines.append(f"- Wave-1: {w1:.2f}pts → W3 proj: {w1*1.618:.2f}pts (1.618×)")
            if abs(retr-0.618)<0.04: lines.append("- ✨ Golden Ratio (61.8%) — strongest signal")
            elif retr>=0.786: lines.append("- ⚠️ Deep retrace (78.6%) — valid but watch W0 level")
            df_r=r.get("df")
            if df_r is not None and len(df_r)>55:
                try:
                    di=add_indicators(df_r)
                    rsi=float(di["RSI"].iloc[-1]) if not pd.isna(di["RSI"].iloc[-1]) else 50
                    lines.append(f"- RSI {rsi:.1f}: "+("Overbought" if rsi>70 else "Oversold" if rsi<30 else "Neutral"))
                except Exception: pass
            lines.append(f"\n📋 **Action**: {'BUY' if s=='BUY' else 'SELL'} @ {ep:.2f} | SL {sl_:.2f} | Target {tgt_:.2f}")
        else:
            lines.append(f"- {sig.get('reason','No pattern')} — **WAIT**")
        lines.append("")
    lines+=["---","### 📚 Wave Reference",
            "| Wave | Meaning | Action |",
            "|------|---------|--------|",
            "| W1 ↑ | First impulse | Missed — watch W2 |",
            "| **W2 ↓** | **Retracement 38–79%** | **🟢 BUY here** |",
            "| **W3 ↑** | **Strongest move** | **Hold for max profit** |",
            "| W4 ↓ | Minor pullback | Partial exit |",
            "| W5 ↑ | Final extension | Full exit |",
            "\n> ⚠️ *Not financial advice. Always use Stop Loss.*"]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# ███  STREAMLIT UI  ███
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-hdr">
  <h1>🌊 Elliott Wave Algo Trader v5.0</h1>
  <p>Full Transparency · Auto-Reverse · IST Times · Wave State · Conservative SL/TGT · Custom Points</p>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Instrument")
    group_sel=st.selectbox("Category",list(TICKER_GROUPS.keys()),index=0)
    group_map=TICKER_GROUPS[group_sel]
    if group_sel=="✏️ Custom Ticker":
        symbol=st.text_input("Enter Yahoo Finance ticker","^NSEI")
    else:
        ticker_name=st.selectbox("Instrument",list(group_map.keys()))
        symbol=group_map[ticker_name]
        st.caption(f"Yahoo Finance: `{symbol}`")

    st.markdown("---")
    if st.session_state.best_cfg_applied:
        st.markdown("""<div style="background:rgba(0,229,255,.08);border:1px solid #00bcd4;
        border-radius:8px;padding:8px 11px;font-size:.82rem;margin-bottom:8px">
        ✨ <b style="color:#00e5ff">Optimized Config Active</b></div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    interval=c1.selectbox("⏱ Timeframe",TIMEFRAMES,index=6)
    vp_list=VALID_PERIODS.get(interval,PERIODS)
    period=c2.selectbox("📅 Period",vp_list,index=min(4,len(vp_list)-1))

    st.markdown("---")
    st.markdown("### 🌊 Elliott Wave")
    depth=st.slider("Pivot Depth",2,15,st.session_state.applied_depth,
                    help="Lower=more signals. Higher=cleaner signals.")

    st.markdown("---")
    st.markdown("### 🛡️ Risk Management")
    sl_idx  = SL_KEYS.index(st.session_state.applied_sl_lbl) if st.session_state.applied_sl_lbl in SL_KEYS else 0
    tgt_idx = TGT_KEYS.index(st.session_state.applied_tgt_lbl) if st.session_state.applied_tgt_lbl in TGT_KEYS else 0
    sl_lbl  = st.selectbox("Stop Loss",  SL_KEYS,  index=sl_idx)
    tgt_lbl = st.selectbox("Target",     TGT_KEYS, index=tgt_idx)
    sl_val  = SL_MAP[sl_lbl]
    tgt_val = TGT_MAP[tgt_lbl]

    # Custom points inputs
    if sl_val == SL_CUSTOM_PTS or tgt_val == "__custom_pts__":
        st.markdown("**Custom Points Settings**")
    if sl_val == SL_CUSTOM_PTS:
        st.session_state.custom_sl_pts = st.number_input(
            "SL Points (absolute)", min_value=1.0, max_value=100000.0,
            value=st.session_state.custom_sl_pts, step=5.0,
            help="e.g. 50 means SL is 50 points below entry for BUY")
    if tgt_val == "__custom_pts__":
        st.session_state.custom_tgt_pts = st.number_input(
            "Target Points (absolute)", min_value=1.0, max_value=500000.0,
            value=st.session_state.custom_tgt_pts, step=10.0,
            help="e.g. 100 means Target is 100 points above entry for BUY")
    custom_sl_pts  = st.session_state.custom_sl_pts
    custom_tgt_pts = st.session_state.custom_tgt_pts

    if sl_val == SL_SIG_REVERSE:
        st.info("🔄 SL = Signal Reverse: position exits automatically when Elliott Wave signal flips direction.")
    if tgt_val == "__sig_reverse__":
        st.info("🔄 Target = Signal Reverse: takes profit when signal reverses.")

    capital=st.number_input("💰 Capital (Rs)",10_000,50_000_000,100_000,10_000)

    st.markdown("---")
    st.markdown("### 🏦 Dhan Brokerage")
    dhan_on=st.checkbox("Enable Dhan Integration",value=False)
    dhan_api,sec_id,live_qty=None,"1333",1
    if dhan_on:
        d_cid=st.text_input("Client ID")
        d_tok=st.text_input("Access Token",type="password")
        sec_id=st.text_input("Security ID","1333")
        live_qty=st.number_input("Order Qty",1,100_000,1)
        if d_cid and d_tok:
            dhan_api=DhanAPI(d_cid,d_tok)
            if st.button("🔌 Test Dhan"): st.json(dhan_api.fund_limit())
        else: st.info("Enter credentials to activate")
    st.markdown("---")
    st.caption(f"⚡ Rate-limit: **1.5 s** · Polls every candle")
    st.caption(f"📌 `{symbol}` · `{interval}` · `{period}`")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
t_analysis,t_live,t_bt,t_opt,t_help=st.tabs([
    "🔭  Wave Analysis","🔴  Live Trading","📊  Backtest","🔬  Optimization","❓  Help",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — WAVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with t_analysis:
    st.markdown("### 🔭 Multi-Timeframe Elliott Wave Analysis")
    ac1,ac2,ac3=st.columns([1.2,1,2.4])
    with ac1:
        run_analysis=st.button("🔭 Run Full Analysis",type="primary",use_container_width=True)
    with ac2:
        custom_tf_only=st.checkbox("Sidebar TF only",value=False)
    with ac3:
        st.caption(f"{'Sidebar TF' if custom_tf_only else 'Daily · 4H · 1H · 15M'} for `{symbol}`")

    if run_analysis:
        scan_combos=[(interval,period,f"{interval.upper()}·{period}")] if custom_tf_only \
            else [(tf,per,nm) for tf,per,nm in MTF_COMBOS if per in VALID_PERIODS.get(tf,[])]
        results=[]
        prog=st.progress(0,text="Scanning…")
        for idx,(tf,per,nm) in enumerate(scan_combos):
            prog.progress((idx+1)/len(scan_combos),text=f"Fetching {nm}…")
            df_a=fetch_ohlcv(symbol,tf,per,min_delay=1.5)
            if df_a is not None and len(df_a)>=35:
                sig_a=ew_signal(df_a.iloc[:-1],depth,sl_val,tgt_val,custom_sl_pts,custom_tgt_pts)
                results.append({"tf_name":nm,"interval":tf,"period":per,
                                 "signal":sig_a,"df":df_a,"pivots":find_pivots(df_a.iloc[:-1],depth)})
            else:
                results.append({"tf_name":nm,"interval":tf,"period":per,
                                 "signal":_blank(f"No data for {tf}/{per}"),"df":None,"pivots":[]})
        prog.empty()
        buy_score =sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="BUY")
        sell_score=sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"]=="SELL")
        if   buy_score>sell_score  and buy_score>0.5:  overall="BUY"
        elif sell_score>buy_score  and sell_score>0.5: overall="SELL"
        else:                                           overall="HOLD"
        st.session_state["_analysis_results"]=results
        st.session_state["_analysis_overall"]=overall
        st.session_state["_analysis_symbol"]=symbol

    ar=st.session_state.get("_analysis_results")
    if ar:
        overall=st.session_state.get("_analysis_overall","HOLD")
        a_sym  =st.session_state.get("_analysis_symbol",symbol)
        v_colors={"BUY":"#4caf50","SELL":"#f44336","HOLD":"#ffb300"}
        v_bg    ={"BUY":"rgba(76,175,80,.10)","SELL":"rgba(244,67,54,.10)","HOLD":"rgba(255,179,0,.10)"}
        v_icons ={"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}
        st.markdown(f"""
        <div style="background:{v_bg[overall]};border:2px solid {v_colors[overall]};
        border-radius:12px;padding:14px 20px;margin-bottom:10px;text-align:center">
        <span style="font-size:1.55rem;color:{v_colors[overall]};font-weight:700">
          {v_icons[overall]} Overall: {overall}
        </span><br>
        <span style="color:#78909c;font-size:.86rem">{a_sym} — Multi-TF Elliott Wave Consensus</span>
        </div>""",unsafe_allow_html=True)
        tf_cols=st.columns(min(len(ar),4))
        for i,r in enumerate(ar):
            with tf_cols[i%4]:
                s=r["signal"]["signal"]; c_=r["signal"]["confidence"]; pat=r["signal"]["pattern"]
                sc=v_colors.get(s,"#546e7a"); em=v_icons.get(s,"⚪")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.03);border:1px solid #1e3a5f;
                border-radius:8px;padding:9px 11px;text-align:center;margin-bottom:4px">
                <div style="font-size:.77rem;color:#546e7a">{r['tf_name']}</div>
                <div style="font-size:1.15rem;color:{sc};font-weight:700">{em} {s}</div>
                <div style="font-size:.74rem;color:#78909c">{pat}</div>
                <div style="font-size:.79rem;color:#00bcd4">{c_:.0%}</div>
                </div>""",unsafe_allow_html=True)
        st.markdown("---")
        for r in ar:
            if r["df"] is not None:
                s_=r["signal"]["signal"]
                with st.expander(f"📈 {r['tf_name']}  —  {s_}  ({r['signal']['confidence']:.0%})",expanded=(s_!="HOLD")):
                    st.plotly_chart(chart_waves(r["df"],r["pivots"],r["signal"],symbol=a_sym,tf_label=r["tf_name"]),use_container_width=True)
                    if s_ in ("BUY","SELL"):
                        sig_=r["signal"]
                        ep,sl_,tgt_=sig_["entry_price"],sig_["sl"],sig_["target"]
                        rr=abs(tgt_-ep)/abs(ep-sl_) if abs(ep-sl_)>0 else 0
                        mc1,mc2,mc3,mc4=st.columns(4)
                        mc1.metric("Entry",f"{ep:.2f}")
                        mc2.metric("SL",f"{sl_:.2f}",delta=f"-{abs(ep-sl_)/ep*100:.1f}%",delta_color="inverse")
                        mc3.metric("Target",f"{tgt_:.2f}",delta=f"+{abs(tgt_-ep)/ep*100:.1f}%")
                        mc4.metric("R:R",f"1:{rr:.1f}")
        st.markdown("---")
        st.markdown("### 📋 Detailed Analysis & Recommendations")
        st.markdown(generate_mtf_summary(a_sym,ar,overall))
    else:
        st.markdown("<div style='text-align:center;padding:60px 20px;color:#37474f'>"
                    "<h3>Click 🔭 Run Full Analysis</h3>"
                    "<p>Daily · 4H · 1H · 15M scanned automatically</p></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING  (fully transparent)
# ═══════════════════════════════════════════════════════════════════════════
with t_live:
    # ── Top controls row ─────────────────────────────────────────────────
    ctrl1,ctrl2,ctrl3=st.columns([1,1,4])
    with ctrl1:
        if not st.session_state.live_running:
            if st.button("▶ Start Live",type="primary",use_container_width=True):
                st.session_state.live_running=True
                st.session_state.live_log=[]
                st.session_state.last_bar_ts=None
                st.session_state.live_status="scanning"
                threading.Thread(
                    target=live_loop,
                    args=(symbol,interval,period,depth,sl_val,tgt_val,
                          dhan_on,dhan_api,sec_id,live_qty,custom_sl_pts,custom_tgt_pts),
                    daemon=True,
                ).start()
                st.rerun()
        else:
            if st.button("⏹ Stop",type="secondary",use_container_width=True):
                st.session_state.live_running=False
                st.rerun()
    with ctrl2:
        if st.button("🔄 Reset All",use_container_width=True,
                     help="Clears all state — fixes duplicate signal issues"):
            for k,v in _DEFAULTS.items():
                st.session_state[k]=v
            st.success("State cleared ✓"); time.sleep(0.3); st.rerun()
    with ctrl3:
        if st.session_state.live_running:
            st.success(f"🟢 **RUNNING** — `{symbol}` · `{interval}` · `{period}` · polls every candle close | 1.5s rate-limit safe")
        else:
            st.warning("⚫ **STOPPED** — Click ▶ Start Live to begin. Everything is automatic.")

    st.markdown("---")

    # ── Main live dashboard: 3 columns ──────────────────────────────────
    col_status, col_wave, col_chart = st.columns([1.1, 1.2, 2.5], gap="medium")

    # ── COL 1: Real-time status panel ────────────────────────────────────
    with col_status:
        st.markdown("#### 📡 Live Status")

        status = st.session_state.live_status
        ltp    = st.session_state.live_ltp
        ltp_p  = st.session_state.live_ltp_prev
        delay  = st.session_state.live_delay_s

        # LTP display
        if ltp is not None:
            ltp_delta = ltp - ltp_p if ltp_p else 0
            ltp_color = "#4caf50" if ltp_delta >= 0 else "#f44336"
            ltp_arrow = "▲" if ltp_delta >= 0 else "▼"
            st.markdown(f"""
            <div class="ltp-box">
            <div style="font-size:.78rem;color:#546e7a">LTP (Last Traded Price)</div>
            <div style="font-size:1.7rem;color:{ltp_color};font-weight:700;font-family:'Share Tech Mono'">{ltp:,.2f}</div>
            <div style="font-size:.82rem;color:{ltp_color}">{ltp_arrow} {ltp_delta:+.2f}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ltp-box'><div style='color:#546e7a;font-size:.85rem'>LTP: waiting for data…</div></div>",unsafe_allow_html=True)

        st.markdown("")

        # Status banner
        STATUS_CFG = {
            "stopped":     ("⚫","STOPPED","status-stopped","System idle. Click ▶ Start Live."),
            "scanning":    ("🟡","SCANNING","status-scanning","Fetching candles & evaluating waves…"),
            "signal_fired":("💡","SIGNAL FIRED","status-signal","Signal detected — position opening at market price"),
            "pos_buy":     ("🟢","LONG (BUY) OPEN","status-buy","BUY position active. Auto-closes at SL/Target/Reverse."),
            "pos_sell":    ("🔴","SHORT (SELL) OPEN","status-sell","SELL position active. Auto-closes at SL/Target/Reverse."),
            "reversing":   ("🔄","AUTO-REVERSING","status-reverse","Closing old position & opening reverse position automatically"),
        }
        icon,label,css,desc=STATUS_CFG.get(status,STATUS_CFG["scanning"])
        st.markdown(f"""
        <div class="{css}" style="margin-bottom:10px">
        <div style="font-size:1.1rem;font-weight:700">{icon} {label}</div>
        <div style="font-size:.8rem;color:#b0bec5;margin-top:4px">{desc}</div>
        </div>""", unsafe_allow_html=True)

        # Time info
        curr_ist=now_ist().strftime("%H:%M:%S IST")
        last_c  =st.session_state.live_last_candle_ist or "—"
        next_chk=st.session_state.get("live_next_check_ist","—")
        delay_s =st.session_state.live_delay_s

        delay_color="#4caf50" if delay_s<120 else ("#ffb300" if delay_s<300 else "#f44336")
        delay_advice=""
        if delay_s>=300:
            delay_advice="⚠️ Large delay! Data may be stale. Check internet/yfinance."
        elif delay_s>=120:
            delay_advice="ℹ️ Moderate delay. Normal for lower frequency intervals."

        st.markdown(f"""
        <div class="wave-card" style="margin-bottom:8px">
        🕒 <b>Current</b> : {curr_ist}<br>
        📊 <b>Last Candle</b>: {last_c}<br>
        ⏩ <b>Delay</b>   : <span style="color:{delay_color}">{delay_s}s</span><br>
        🔜 <b>Next Check</b>: {next_chk}
        </div>""", unsafe_allow_html=True)

        if delay_advice:
            st.warning(delay_advice)

        # Open position card
        pos=st.session_state.live_position
        if pos:
            ptype=pos["type"]; clr="#4caf50" if ptype=="BUY" else "#f44336"
            cur_pnl=(ltp-pos["entry"])*pos["qty"] if ltp and ptype=="BUY" \
                    else (pos["entry"]-ltp)*pos["qty"] if ltp else 0
            pnl_c="#4caf50" if cur_pnl>=0 else "#f44336"
            dist_sl  =abs(ltp-pos["sl"])/ltp*100 if ltp else 0
            dist_tgt =abs(pos["target"]-ltp)/ltp*100 if ltp else 0
            st.markdown(f"""
            <div class="wave-card" style="border-color:{clr};margin-bottom:8px">
            📍 <b style="color:{clr}">{ptype} OPEN</b><br>
            Entry  : <b>{pos['entry']:.2f}</b> · Qty: <b>{pos['qty']}</b><br>
            LTP    : <b>{ltp:.2f}</b><br>
            SL     : <b style="color:#ff7043">{pos['sl']:.2f}</b> ({dist_sl:.1f}% away)<br>
            Target : <b style="color:#66bb6a">{pos['target']:.2f}</b> ({dist_tgt:.1f}% away)<br>
            Unreal P&L: <b style="color:{pnl_c}">Rs{cur_pnl:+.2f}</b><br>
            Pattern: {pos.get('pattern','—')}<br>
            Conf   : {pos.get('confidence',0):.0%}<br>
            Entered: {pos.get('entry_ist','—')}
            </div>""", unsafe_allow_html=True)
            st.success("✅ Position is **open and managed automatically**. No action needed.")
        else:
            no_pos_reason=st.session_state.live_no_pos_reason
            if no_pos_reason:
                st.markdown(f"""
                <div class="wave-card" style="border-color:#546e7a">
                <b style="color:#78909c">⏸ No Open Position</b><br><br>
                <span style="font-size:.82rem;color:#546e7a">{no_pos_reason}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<div class='wave-card'><b style='color:#546e7a'>⏸ No open position yet</b></div>",unsafe_allow_html=True)

        # Total P&L
        total_pnl=st.session_state.live_pnl
        pnl_c2="#4caf50" if total_pnl>=0 else "#f44336"
        st.markdown(f"""
        <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:6px;
        padding:8px 12px;text-align:center;margin-top:6px">
        <div style="font-size:.78rem;color:#546e7a">Realized P&L (Session)</div>
        <div style="font-size:1.3rem;color:{pnl_c2};font-weight:700">Rs{total_pnl:,.2f}</div>
        <div style="font-size:.76rem;color:#546e7a">Trades: {len(st.session_state.live_trades)} | Signals: {len(st.session_state.live_signals)}</div>
        </div>""", unsafe_allow_html=True)

    # ── COL 2: Wave state panel ───────────────────────────────────────────
    with col_wave:
        st.markdown("#### 🌊 Wave Position")
        ws=st.session_state.live_wave_state
        last_sig=st.session_state.live_last_sig

        if ws:
            dir_color={"BULLISH":"#4caf50","BEARISH":"#f44336","NEUTRAL":"#78909c"}.get(ws.get("direction","NEUTRAL"),"#78909c")
            dir_icon ={"BULLISH":"📈","BEARISH":"📉","NEUTRAL":"➡️"}.get(ws.get("direction","NEUTRAL"),"➡️")

            st.markdown(f"""
            <div class="wave-card" style="margin-bottom:8px">
            <b style="color:#00e5ff">Current Wave</b><br>
            <span style="color:{dir_color}">{dir_icon} {ws.get('current_wave','—')}</span><br><br>
            <b style="color:#00e5ff">Next Expected</b><br>
            <span style="color:#b0bec5">{ws.get('next_wave','—')}</span>
            </div>""", unsafe_allow_html=True)

            # Fibonacci levels
            fibs=ws.get("fib_levels",{})
            if fibs:
                st.markdown("**📐 Wave-3 Fibonacci Targets**")
                for lbl,val in fibs.items():
                    bar_color="#4caf50" if "1.618" in lbl else ("#ab47bc" if "2.618" in lbl else "#ffb300")
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;background:#060d14;
                    border-left:3px solid {bar_color};border-radius:0 4px 4px 0;
                    padding:4px 8px;margin-bottom:3px;font-size:.82rem;">
                    <span style="color:#78909c">{lbl}</span>
                    <span style="color:{bar_color};font-family:'Share Tech Mono'">{val:.2f}</span>
                    </div>""", unsafe_allow_html=True)

            # Action text
            st.markdown("**🤖 Auto System Says:**")
            action_text=ws.get("action","")
            # Render action as markdown
            st.markdown(f"""<div style="background:#060d14;border:1px solid #1e3a5f;border-radius:6px;
            padding:10px 12px;font-size:.82rem;color:#b0bec5;line-height:1.7">{action_text.replace(chr(10),'<br>')}</div>""",unsafe_allow_html=True)

        else:
            st.info("Wave state appears after first scan. Click ▶ Start or Scan Signal Now.")

        # Last signal
        if last_sig and last_sig.get("signal")!="HOLD":
            s_=last_sig["signal"]
            sc_="#4caf50" if s_=="BUY" else "#f44336"
            st.markdown(f"""
            <div style="margin-top:10px;background:rgba(255,255,255,.03);border:1px solid #1e3a5f;
            border-radius:8px;padding:9px 12px">
            <div style="font-size:.78rem;color:#546e7a">Last Elliott Wave Signal</div>
            <div style="font-size:1rem;color:{sc_};font-weight:700">{s_} — {last_sig.get('pattern','—')}</div>
            <div style="font-size:.8rem;color:#00bcd4">Confidence: {last_sig.get('confidence',0):.0%}</div>
            <div style="font-size:.78rem;color:#546e7a">{last_sig.get('reason','—')}</div>
            </div>""", unsafe_allow_html=True)
        elif last_sig:
            st.markdown(f"""
            <div style="margin-top:10px;background:rgba(255,255,255,.02);border:1px solid #37474f;
            border-radius:8px;padding:9px 12px">
            <div style="font-size:.78rem;color:#546e7a">Last Signal: HOLD</div>
            <div style="font-size:.78rem;color:#455a64">{last_sig.get('reason','—')}</div>
            </div>""", unsafe_allow_html=True)

        # Manual scan
        st.markdown("---")
        if st.button("🔍 Scan Signal Now", use_container_width=True):
            with st.spinner("Fetching (1.5s rate-limit)…"):
                df_=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
            if df_ is not None and len(df_)>=35:
                sig_=ew_signal(df_.iloc[:-1],depth,sl_val,tgt_val,custom_sl_pts,custom_tgt_pts)
                pivs_=find_pivots(df_.iloc[:-1],depth)
                ws_=analyze_wave_state(df_.iloc[:-1],pivs_,sig_)
                st.session_state._scan_sig=sig_
                st.session_state._scan_df=df_
                st.session_state.live_wave_state=ws_
                st.session_state.live_last_sig=sig_
                st.session_state.live_ltp=float(df_["Close"].iloc[-1])
                st.session_state.live_last_candle_ist=to_ist_str(df_.index[-2])
                st.rerun()
            else:
                st.error("No data / too few bars")

        # Signal card
        sc_=st.session_state._scan_sig
        if sc_:
            s=sc_["signal"]
            if s=="BUY":
                st.markdown(f"""
                <div class="sig-buy" style="margin-top:8px">
                <div style="font-size:1.2rem;color:#4caf50;font-weight:700">🟢 BUY SIGNAL</div>
                <div style="font-size:.83rem;margin-top:4px">
                Entry <b>{sc_['entry_price']:.2f}</b> · SL <b style="color:#ff7043">{sc_['sl']:.2f}</b> · T <b style="color:#66bb6a">{sc_['target']:.2f}</b><br>
                Confidence <b>{sc_['confidence']:.0%}</b> · {sc_['pattern']}<br>
                <small style="color:#78909c">{sc_['reason']}</small>
                </div></div>""", unsafe_allow_html=True)
            elif s=="SELL":
                st.markdown(f"""
                <div class="sig-sell" style="margin-top:8px">
                <div style="font-size:1.2rem;color:#f44336;font-weight:700">🔴 SELL SIGNAL</div>
                <div style="font-size:.83rem;margin-top:4px">
                Entry <b>{sc_['entry_price']:.2f}</b> · SL <b style="color:#ff7043">{sc_['sl']:.2f}</b> · T <b style="color:#66bb6a">{sc_['target']:.2f}</b><br>
                Confidence <b>{sc_['confidence']:.0%}</b> · {sc_['pattern']}<br>
                <small style="color:#78909c">{sc_['reason']}</small>
                </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sig-hold" style="margin-top:8px">
                <div style="font-size:1rem;color:#78909c;font-weight:600">⏸ HOLD — No Signal</div>
                <div style="font-size:.8rem;color:#546e7a;margin-top:3px">{sc_['reason']}</div>
                </div>""", unsafe_allow_html=True)

    # ── COL 3: Chart + history ────────────────────────────────────────────
    with col_chart:
        st.markdown("#### 📈 Live Chart")
        df_s=st.session_state._scan_df
        if df_s is not None:
            piv_=find_pivots(df_s.iloc[:-1],depth)
            st.plotly_chart(chart_waves(df_s,piv_,st.session_state._scan_sig,symbol=symbol),
                            use_container_width=True)
        else:
            st.info("Click 🔍 Scan Signal Now to load the chart.")

        # Signal + trade history
        if st.session_state.live_signals:
            with st.expander(f"📋 Signal History ({len(st.session_state.live_signals)})",expanded=False):
                st.dataframe(pd.DataFrame(st.session_state.live_signals).tail(15),
                             use_container_width=True,height=160)

        if st.session_state.live_trades:
            with st.expander(f"🏁 Completed Trades ({len(st.session_state.live_trades)})",expanded=True):
                trd_df=pd.DataFrame(st.session_state.live_trades)
                st.dataframe(trd_df,use_container_width=True,height=150)
                wins_=(trd_df["PnL Rs"]>0).sum(); tot_=len(trd_df); pnl_=trd_df["PnL Rs"].sum()
                st.caption(f"Win rate: {wins_}/{tot_} ({wins_/tot_*100:.0f}%)  ·  P&L Rs{pnl_:,.2f}")

        # Activity log
        if st.session_state.live_log:
            with st.expander("📜 Activity Log (background thread events)",expanded=False):
                st.code("\n".join(reversed(st.session_state.live_log[-60:])),language=None)

    # Auto-refresh while running
    if st.session_state.live_running:
        time.sleep(3); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with t_bt:
    bl,br=st.columns([1,2.6],gap="medium")
    with bl:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""
        <div class="info-box">
        📈 <b>Symbol</b>: <code>{symbol}</code><br>
        ⏱  <code>{interval}</code> · <code>{period}</code><br>
        🌊 Depth <code>{depth}</code> · Rs<code>{capital:,}</code><br>
        🛡  SL: <code>{sl_lbl}</code><br>
        🎯 Target: <code>{tgt_lbl}</code><br>
        {"📍 SL pts: "+str(custom_sl_pts) if sl_val==SL_CUSTOM_PTS else ""}
        {"📍 TGT pts: "+str(custom_tgt_pts) if tgt_val=="__custom_pts__" else ""}<br>
        <small style="color:#546e7a">
        Signal bar N → entry at open bar N+1<br>
        SL checked with Low(BUY)/High(SELL) first (conservative)<br>
        Then Target checked with High(BUY)/Low(SELL)<br>
        Signal Reverse exit included in Exit Reason
        </small>
        </div>""", unsafe_allow_html=True)
        if st.session_state.best_cfg_applied:
            st.success("✨ Optimized config active")
        if st.button("🚀 Run Backtest",type="primary",use_container_width=True):
            with st.spinner("Fetching (1.5s rate-limit)…"):
                df_bt=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
            if df_bt is None or len(df_bt)<40:
                st.error("Not enough data.")
            else:
                with st.spinner(f"Running on {len(df_bt)} bars…"):
                    res=run_backtest(df_bt,depth,sl_val,tgt_val,capital,custom_sl_pts,custom_tgt_pts)
                    res.update({"df":df_bt,"pivots":find_pivots(df_bt,depth),
                                "symbol":symbol,"interval":interval,"period":period})
                st.session_state.bt_results=res
                if "error" in res: st.error(res["error"])
                else: st.success(f"✅ {res['metrics']['Total Trades']} trades generated!")

    with br:
        r=st.session_state.bt_results
        if r and "metrics" in r:
            m=r["metrics"]
            st.markdown(f"### Results — `{r.get('symbol','')}` · `{r.get('interval','')}` · `{r.get('period','')}`")
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Total Return",f"{m['Total Return %']}%",delta=f"Rs{m['Final Equity Rs']:,}")
            c2.metric("Win Rate",    f"{m['Win Rate %']}%",   delta=f"{m['Wins']}W/{m['Losses']}L")
            c3.metric("Profit Factor",str(m["Profit Factor"]))
            c4.metric("Max Drawdown",f"{m['Max Drawdown %']}%")
            c5,c6,c7,c8=st.columns(4)
            c5.metric("Sharpe",str(m["Sharpe Ratio"]))
            c6.metric("Trades",str(m["Total Trades"]))
            c7.metric("Avg Win", f"Rs{m['Avg Win Rs']:,}")
            c8.metric("Avg Loss",f"Rs{m['Avg Loss Rs']:,}")

            # Exit breakdown
            eb=r.get("exit_breakdown",{})
            if eb:
                ea,eb_,ec,ed=st.columns(4)
                ea.metric("SL Hits",    str(eb.get("SL hits",0)))
                eb_.metric("Target Hits",str(eb.get("Target hits",0)))
                ec.metric("Sig Reverse",str(eb.get("Signal Reverse exits",0)))
                ed.metric("Still Open", str(eb.get("Still open",0)))

            tc1,tc2,tc3=st.tabs(["🕯 Wave Chart","📈 Equity Curve","📋 Trade Log"])
            with tc1:
                st.plotly_chart(chart_waves(r["df"],r["pivots"],_blank(),r["trades"],r["symbol"]),use_container_width=True)
            with tc2:
                st.plotly_chart(chart_equity(r["equity_curve"]),use_container_width=True)
            with tc3:
                trades_disp=r["trades"].copy()
                # Verification column
                trades_disp["SL Verified"]=trades_disp.apply(
                    lambda row: "✅" if (
                        (row["Type"]=="BUY"  and "SL" in row["Exit Reason"] and row["Exit Bar Low"] <=row["SL"]) or
                        (row["Type"]=="SELL" and "SL" in row["Exit Reason"] and row["Exit Bar High"]>=row["SL"]) or
                        "SL" not in row["Exit Reason"]
                    ) else "⚠️", axis=1
                )
                trades_disp["TGT Verified"]=trades_disp.apply(
                    lambda row: "✅" if (
                        (row["Type"]=="BUY"  and "Target" in row["Exit Reason"] and row["Exit Bar High"]>=row["Target"]) or
                        (row["Type"]=="SELL" and "Target" in row["Exit Reason"] and row["Exit Bar Low"] <=row["Target"]) or
                        "Target" not in row["Exit Reason"]
                    ) else "⚠️", axis=1
                )
                st.dataframe(trades_disp,use_container_width=True,height=420)
                st.info(
                    "📊 **Exit Bar Low/High** columns let you verify exits:\n\n"
                    "- **BUY SL exit**: Exit Bar Low ≤ SL ✅ (conservative — low checked first)\n"
                    "- **BUY Target exit**: Exit Bar High ≥ Target ✅\n"
                    "- **SELL SL exit**: Exit Bar High ≥ SL ✅\n"
                    "- **SELL Target exit**: Exit Bar Low ≤ Target ✅\n"
                    "- **Signal Reverse**: Closed at close of reversal bar\n\n"
                    "A completed live trade will appear here with same Symbol·TF·Period."
                )
                st.download_button("📥 Download CSV",data=r["trades"].to_csv(index=False),
                                   file_name=f"ew_bt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        elif r and "error" in r:
            st.error(r["error"])
        else:
            st.markdown("<div style='text-align:center;padding:80px 20px;color:#37474f'>"
                        "<h3>Run backtest to see results</h3></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
with t_opt:
    ol,or_=st.columns([1,3],gap="medium")
    with ol:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""
        <div class="info-box">
        📈 <code>{symbol}</code> · <code>{interval}</code> · <code>{period}</code><br>
        💰 Rs<code>{capital:,}</code><br><br>
        Grid: 4×4×5 = <b>80 combos</b>
        </div>""", unsafe_allow_html=True)
        st.warning("⏳ ~80 backtests (~1–3 min)")
        if st.button("🔬 Run Optimization",type="primary",use_container_width=True):
            with st.spinner("Fetching data…"):
                df_opt=fetch_ohlcv(symbol,interval,period,min_delay=1.5)
            if df_opt is None or len(df_opt)<50:
                st.error("Not enough data.")
            else:
                with st.spinner("Running 80 combinations…"):
                    opt_df=run_optimization(df_opt,capital,custom_sl_pts,custom_tgt_pts)
                st.session_state.opt_results={"df":opt_df,"symbol":symbol,"interval":interval,"period":period}
                if not opt_df.empty: st.success(f"✅ Best Score: {opt_df['Score'].iloc[0]:.2f}")

    with or_:
        opt_r=st.session_state.opt_results
        if opt_r and opt_r.get("df") is not None and not opt_r["df"].empty:
            opt_df=opt_r["df"]
            st.markdown(f"### Results — `{opt_r['symbol']}` · `{opt_r['interval']}` · `{opt_r['period']}`")
            best_row=opt_df.iloc[0]
            b1,b2,b3,b4,b5=st.columns(5)
            b1.metric("Best Depth",str(int(best_row["Depth"])))
            b2.metric("Best SL",   str(best_row["SL"]))
            b3.metric("Best Target",str(best_row["Target"]))
            b4.metric("Best Return",f"{best_row['Return %']:.1f}%")
            b5.metric("Best Score", f"{best_row['Score']:.2f}")

            st.markdown("---")
            ac1,ac2=st.columns([1.6,2.4])
            with ac1:
                n_rows=st.number_input("Apply top N config",min_value=1,max_value=min(5,len(opt_df)),value=1,step=1)
                apply_row=opt_df.iloc[int(n_rows)-1]
            with ac2:
                st.markdown(f"""
                <div class="best-cfg">
                <b style="color:#00e5ff">Config #{int(n_rows)}</b><br>
                Depth <b>{int(apply_row['Depth'])}</b> · SL <b>{apply_row['SL']}</b> · Target <b>{apply_row['Target']}</b><br>
                Win <b>{apply_row['Win %']}%</b> · Return <b>{apply_row['Return %']}%</b> · Score <b>{apply_row['Score']:.2f}</b>
                </div>""", unsafe_allow_html=True)

            if st.button(f"✨ Apply Config #{int(n_rows)} → Sidebar + Backtest + Live",type="primary",use_container_width=True):
                new_depth=int(apply_row["Depth"])
                new_sl   =sl_label_from_val(apply_row["SL"])
                new_tgt  =tgt_label_from_val(apply_row["Target"])
                st.session_state.applied_depth  =new_depth
                st.session_state.applied_sl_lbl =new_sl
                st.session_state.applied_tgt_lbl=new_tgt
                st.session_state.best_cfg_applied=True
                st.success(f"✅ Applied! Depth={new_depth} · SL={new_sl} · Target={new_tgt}")
                time.sleep(0.4); st.rerun()

            st.markdown("---")
            oc1,oc2=st.tabs(["📊 Scatter","📋 Full Table"])
            with oc1:
                st.plotly_chart(chart_opt_scatter(opt_df),use_container_width=True)
                st.caption("Top-left = best. Bubble size = Win Rate. Color = Score.")
            with oc2:
                def _hl(row):
                    if row.name==0:   return ["background-color:rgba(0,229,255,.18)"]*len(row)
                    elif row.name<3:  return ["background-color:rgba(0,229,255,.08)"]*len(row)
                    return [""]*len(row)
                st.dataframe(opt_df.style.apply(_hl,axis=1),use_container_width=True,height=500)
                st.download_button("📥 Download CSV",data=opt_df.to_csv(index=False),
                                   file_name=f"ew_opt_{symbol}_{interval}_{period}.csv",mime="text/csv")
        else:
            st.markdown("<div style='text-align:center;padding:80px;color:#37474f'>"
                        "<h3>Run optimization to see results</h3></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — HELP
# ═══════════════════════════════════════════════════════════════════════════
with t_help:
    st.markdown("## 📖 Complete Guide — Elliott Wave Algo Trader v5.0")
    h1,h2=st.columns(2,gap="large")
    with h1:
        st.markdown("""
### 🤖 Is Everything Automatic?

**YES. Once you click ▶ Start Live:**
- System fetches data every candle automatically
- Detects Elliott Wave-2 completion
- Opens BUY/SELL position at market price
- Monitors SL and Target every candle
- **Auto-closes** position when SL or Target is hit
- **Auto-reverses** — if a BUY is open and SELL signal fires, it closes BUY and opens SELL
- **No manual decisions needed**

---

### 🌊 Which Wave Am I In? (Wave State Panel)

The **Wave Position** panel (Live Trading tab, middle column) shows:
- **Current wave** (e.g., "Wave 2 bottom — entry zone")
- **Next expected wave** (e.g., "Wave 3 UP — strongest impulse")
- **Fibonacci targets** for Wave 3 (1×, 1.618×, 2.618× extensions)
- **Auto System Says** text explaining exactly what to do

---

### 📡 What Do Live Status States Mean?

| Status | Meaning |
|--------|---------|
| ⚫ STOPPED | Engine not running |
| 🟡 SCANNING | Fetching data, evaluating wave pattern |
| 💡 SIGNAL FIRED | Wave-2 detected, entering position |
| 🟢 LONG OPEN | BUY position active, auto-managed |
| 🔴 SHORT OPEN | SELL position active, auto-managed |
| 🔄 AUTO-REVERSING | Closing old position, opening reverse |

---

### ⏱️ IST Time & Delay

- **Current IST**: Your real current time in India
- **Last Candle IST**: When the last completed candle closed
- **Delay**: Gap between last candle and now
  - < 2 min: 🟢 Normal
  - 2–5 min: 🟡 Slight delay
  - > 5 min: 🔴 Check internet / yfinance

---

### 🛡️ SL & Target Options

| Option | How It Works |
|--------|-------------|
| Wave Auto | SL = 0.2% below/above Wave-2 pivot (best) |
| % options | Fixed % from entry |
| Custom Points | Fixed point distance (e.g. 50 pts) |
| Signal Reverse | Exit ONLY when wave signal flips direction |
        """)

    with h2:
        st.markdown("""
### 🔄 Auto-Reverse Logic

If a **BUY** position is open and a **SELL signal** fires:
1. BUY position closed at current market price
2. Reason logged as "Signal Reversed SELL — auto closing BUY"
3. New **SELL** position opened immediately at market price

This happens automatically — no action needed from you.

---

### 📊 Backtest Exit Verification

New columns in Trade Log:
- **Exit Bar Low**: Lowest price of the exit bar
- **Exit Bar High**: Highest price of the exit bar

Verification logic:
- **BUY SL exit** ✅ if Exit Bar Low ≤ SL (conservative — low is checked first)
- **BUY Target exit** ✅ if Exit Bar High ≥ Target
- **SELL SL exit** ✅ if Exit Bar High ≥ SL
- **SELL Target exit** ✅ if Exit Bar Low ≤ Target

This proves the backtest exits are realistic — SL is not checked via high price, only low (BUY), preventing unrealistic fills.

---

### ❓ Why No Position Despite Signal?

The system will explain in the "No Open Position" box:
- Risk=0: SL too close to entry → adjust SL setting
- Invalid target → adjust Target setting
- No pattern yet → waiting, will fire automatically
- Large delay → data may be stale

---

### ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| "Signal already fired" | Click 🔄 Reset All |
| No trades in backtest | Reduce depth or use longer period |
| Large delay warning | Check internet connection |
| 4h interval issues | Use 1h or 1d instead |
| Dhan order rejected | Verify Security ID, check market hours |

---

### 📊 Trusted Signal Priority

Highest trust: **Multiple TFs aligned + 61.8% retrace + conf ≥ 85%**

Use **Wave Analysis tab** to see all TF alignment before starting live trading.
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#37474f;font-size:.83rem;padding:8px">
    🌊 Elliott Wave Algo Trader v5.0 ·
    Streamlit + yfinance + Plotly ·
    <b style="color:#f44336">Not financial advice. Use Stop Loss. Verify in paper trading first.</b>
    </div>""", unsafe_allow_html=True)
