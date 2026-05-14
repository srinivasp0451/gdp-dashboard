#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   ⚡  PROFESSIONAL ALGORITHMIC TRADING PLATFORM  v3.0        ║
║   Backtesting · Live Trading · Trade History                 ║
║   ORB · Simple Buy · Dhan (pydhan + dhanhq)                  ║
╚══════════════════════════════════════════════════════════════╝
Fixes in this version:
 • All Plotly fillcolor use rgba() — no 8-digit hex
 • All indicators calculated manually (no ta/talib/pandas_ta)
 • Backtest uses FIXED SL only (no trailing — no tick data)
 • Sidebar fully functional
 • Dhan: pydhan for equity, dhanhq for options
 • Options: Buy signal→BUY CE, Sell signal→BUY PE (always buyer)
 • Equity: direction mirrors main algo (BUY/SELL)
 • Placeholder/paper-trade mode checkbox
"""

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, uuid, math, warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except Exception:
    IST = None

try:
    from pydhan import pydhan as _Pydhan
    PYDHAN_OK = True
except Exception:
    PYDHAN_OK = False

try:
    from dhanhq import dhanhq as _Dhanhq
    DHANHQ_OK = True
except Exception:
    DHANHQ_OK = False

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Pro Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS  — dark terminal aesthetic, no 8-digit hex
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:#0a0e17!important; color:#e2e8f0!important; font-family:'Inter',sans-serif;
}
[data-testid="stSidebar"],[data-testid="stSidebarContent"]{
  background:#111827!important; border-right:1px solid #1e2d45!important;
}
[data-testid="stSidebar"] *{ color:#e2e8f0!important; }
[data-testid="stSidebar"] label{ color:#94a3b8!important; font-size:11px!important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] input{
  background:#1a2235!important; color:#e2e8f0!important;
  border:1px solid #1e2d45!important;
}
.stTabs [data-baseweb="tab-list"]{background:#111827;border-bottom:1px solid #1e2d45;}
.stTabs [data-baseweb="tab"]{
  color:#94a3b8!important; font-family:'JetBrains Mono',monospace;
  font-size:12px; font-weight:600; padding:10px 22px;
}
.stTabs [aria-selected="true"]{
  background:#1a2235!important; color:#00d4aa!important;
  border:1px solid #1e2d45!important; border-bottom-color:#1a2235!important;
}
.stButton>button{
  background:#1a2235; color:#00d4aa; border:1px solid #00d4aa;
  border-radius:6px; font-family:'JetBrains Mono',monospace;
  font-size:12px; font-weight:600; transition:all .15s; padding:6px 16px;
}
.stButton>button:hover{background:#00d4aa;color:#0a0e17;}
.stButton>button[kind="primary"]{background:#00d4aa;color:#0a0e17;}
div[data-testid="metric-container"]{
  background:#1a2235;border:1px solid #1e2d45;border-radius:8px;padding:12px 16px;
}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:10px!important;font-family:'JetBrains Mono',monospace!important;}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important;}
.mcard{background:#1a2235;border:1px solid #1e2d45;border-radius:10px;padding:14px 18px;margin-bottom:8px;}
.mlabel{font-size:10px;color:#475569;font-family:'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase;}
.mval{font-size:21px;font-family:'JetBrains Mono',monospace;font-weight:700;margin-top:4px;}
.cgreen{color:#10b981!important;} .cred{color:#ef4444!important;}
.cgold{color:#f59e0b!important;}  .cblue{color:#3b82f6!important;}
.cacc{color:#00d4aa!important;}
.sbadge{display:inline-block;padding:3px 10px;border-radius:20px;
  font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.08em;}
.bwin{background:rgba(16,185,129,.15);color:#10b981;border:1px solid #10b981;}
.bloss{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid #ef4444;}
.bbuy{background:rgba(16,185,129,.15);color:#10b981;border:1px solid #10b981;}
.trow{background:#1a2235;border:1px solid #1e2d45;border-radius:8px;
  padding:10px 14px;margin-bottom:6px;font-family:'JetBrains Mono',monospace;font-size:12px;}
.sh{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.15em;
  text-transform:uppercase;color:#475569;border-bottom:1px solid #1e2d45;
  padding-bottom:6px;margin-bottom:12px;margin-top:8px;}
.ldot{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:#10b981;margin-right:6px;animation:pulse 1.4s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(1.4)}}
.ibox{background:rgba(59,130,246,.08);border-left:3px solid #3b82f6;
  border-radius:0 6px 6px 0;padding:10px 14px;font-size:12px;color:#93c5fd;margin-bottom:10px;}
.wbox{background:rgba(245,158,11,.08);border-left:3px solid #f59e0b;
  border-radius:0 6px 6px 0;padding:10px 14px;font-size:12px;color:#fbbf24;margin-bottom:10px;}
.pbox{background:rgba(139,92,246,.08);border-left:3px solid #8b5cf6;
  border-radius:0 6px 6px 0;padding:10px 14px;font-size:12px;color:#c4b5fd;margin-bottom:10px;}
.ocard{background:#1a2235;border:1px solid #312e81;border-radius:8px;
  padding:9px 13px;margin-bottom:5px;font-family:'JetBrains Mono',monospace;font-size:11px;}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
TICKER_MAP = {
    "NIFTY 50"     :"^NSEI",      "BANK NIFTY"   :"^NSEBANK",
    "SENSEX"       :"^BSESN",     "NIFTY IT"     :"^CNXIT",
    "NIFTY PHARMA" :"^CNXPHARMA", "NIFTY AUTO"   :"^CNXAUTO",
    "NIFTY FMCG"   :"^CNXFMCG",   "BTC/USD"      :"BTC-USD",
    "ETH/USD"      :"ETH-USD",    "BNB/USD"      :"BNB-USD",
    "USD/INR"      :"USDINR=X",   "EUR/USD"      :"EURUSD=X",
    "GBP/USD"      :"GBPUSD=X",   "GOLD"         :"GC=F",
    "SILVER"       :"SI=F",       "CRUDE OIL"    :"CL=F",
    "RELIANCE"     :"RELIANCE.NS","TCS"           :"TCS.NS",
    "HDFC BANK"    :"HDFCBANK.NS","INFOSYS"       :"INFY.NS",
    "ICICI BANK"   :"ICICIBANK.NS","SBI"          :"SBIN.NS",
    "WIPRO"        :"WIPRO.NS",   "BAJAJ FINANCE" :"BAJFINANCE.NS",
    "CUSTOM"       :"CUSTOM",
}
TF_MAP = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h","1d":"1d","1wk":"1wk"}
PR_MAP = {
    "1 Day":"1d","5 Days":"5d","7 Days":"7d","1 Month":"1mo",
    "3 Months":"3mo","6 Months":"6mo","1 Year":"1y",
    "2 Years":"2y","5 Years":"5y","10 Years":"10y","20 Years":"max",
}
VALID = {
    "1m" :["1d","2d","5d","7d"],
    "5m" :["1d","5d","7d","1mo"],
    "15m":["1d","5d","7d","1mo","3mo"],
    "30m":["1d","5d","7d","1mo","3mo","6mo"],
    "1h" :["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h" :["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y"],
    "1d" :["1mo","3mo","6mo","1y","2y","5y","10y","max"],
    "1wk":["3mo","6mo","1y","2y","5y","10y","max"],
}
NSE_SECTORS = {
    "🏦 Banking"  :["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","BANDHANBNK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS"],
    "💻 IT/Tech"  :["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "💊 Pharma"   :["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","TORNTPHARM.NS","AUROPHARMA.NS"],
    "🚗 Auto"     :["MARUTI.NS","TATAMOTORS.NS","M&M.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS"],
    "⚡ Energy"   :["RELIANCE.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS"],
    "🛒 FMCG"     :["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS"],
    "💰 Finance"  :["BAJFINANCE.NS","BAJAJFINSV.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","RECLTD.NS"],
    "🏗️ Infra"   :["LT.NS","ADANIPORTS.NS","SIEMENS.NS","ABB.NS","BHEL.NS","IRFC.NS"],
}
STRATEGIES  = ["ORB (Opening Range Breakout)","Simple Buy","Simple Sell"]
SL_BT       = ["Fixed Points","ATR Based","Risk-Reward Based","Volatility (BB) Based"]
SL_LT       = ["Fixed Points","ATR Based","Risk-Reward Based","Trailing (Points)",
                "Trailing (Candle Low/High)","Trailing (Swing Low/High)",
                "Signal Reversal","Volatility (BB) Based"]
TGT_TYPES   = ["Fixed Points","ATR Based","Risk-Reward Based",
                "Trailing Target (Display Only)","Volatility (BB) Based"]
IND_LIST    = ["RSI","MACD","Volume","ADX","ATR Filter","Order Block","Liquidity Hunt",
               "Volatility Filter","EMA","SMA","VWAP","Fibonacci","EMA/SMA Crossover"]

# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
def _init():
    D = dict(
        trade_history=[],  live_position=None, live_active=False,
        live_ticker="",    live_ltp=0.0,       live_signals=None,
        last_fetch=0.0,    bt_results=None,    bt_trades=[],
        bt_equity=[],      bt_df=None,         bt_sigs=None,
        partial_closed=False, trailing_high=0.0,
        live_log=deque(maxlen=60), order_log=[],
        dhan_cid="",       dhan_tok="",
    )
    for k,v in D.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ──────────────────────────────────────────────────────────────
# DATA FETCHER  (1.5 s rate limit)
# ──────────────────────────────────────────────────────────────
class DF:
    DELAY = 1.5

    @staticmethod
    def _wait():
        gap = time.time() - st.session_state.last_fetch
        if gap < DF.DELAY:
            time.sleep(DF.DELAY - gap)
        st.session_state.last_fetch = time.time()

    @staticmethod
    def resolve(name, custom=""):
        if name == "CUSTOM":
            return custom.strip().upper()
        return TICKER_MAP.get(name, name)

    @staticmethod
    def fetch(ticker, interval, period, retries=3):
        for attempt in range(retries):
            try:
                DF._wait()
                df = yf.download(ticker, interval=interval, period=period,
                                 auto_adjust=True, progress=False, timeout=15)
                if df is None or df.empty:
                    return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.dropna(subset=["Open","High","Low","Close"], inplace=True)
                if IST:
                    try:
                        if df.index.tz is None:
                            df.index = df.index.tz_localize("UTC").tz_convert(IST)
                        else:
                            df.index = df.index.tz_convert(IST)
                    except Exception:
                        pass
                return df
            except Exception as e:
                if attempt < retries-1:
                    time.sleep(2.0)
                else:
                    st.error(f"Fetch error: {e}")
        return None

    @staticmethod
    def get_ltp(ticker):
        try:
            DF._wait()
            info = yf.Ticker(ticker).fast_info
            p = getattr(info, "last_price", None)
            if p and p > 0:
                return float(p)
        except Exception:
            pass
        try:
            df = DF.fetch(ticker, "1m", "1d")
            if df is not None and not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def mom_pct(ticker):
        try:
            DF._wait()
            df = yf.download(ticker, interval="1m", period="2d",
                             auto_adjust=True, progress=False, timeout=10)
            if df is None or df.empty:
                return 0.0
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            c = df["Close"].dropna()
            return float((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100) if len(c)>=2 else 0.0
        except Exception:
            return 0.0

# ──────────────────────────────────────────────────────────────
# MANUAL INDICATORS  (zero external TA deps)
# ──────────────────────────────────────────────────────────────
class Calc:

    @staticmethod
    def ema(s, p):
        return s.ewm(span=p, adjust=False).mean()

    @staticmethod
    def sma(s, p):
        return s.rolling(window=p, min_periods=1).mean()

    @staticmethod
    def atr(high, low, close, p=14):
        tr = pd.concat([
            high-low,
            (high-close.shift(1)).abs(),
            (low-close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=p, adjust=False).mean()

    @staticmethod
    def rsi(close, p=14):
        d  = close.diff()
        g  = d.clip(lower=0)
        l  = (-d).clip(lower=0)
        ag = g.ewm(com=p-1, adjust=False).mean()
        al = l.ewm(com=p-1, adjust=False).mean()
        rs = ag / al.replace(0, np.nan)
        return 100 - 100/(1+rs)

    @staticmethod
    def macd(close, fast=12, slow=26, sig=9):
        m = Calc.ema(close, fast) - Calc.ema(close, slow)
        s = Calc.ema(m, sig)
        return m, s, m-s

    @staticmethod
    def adx(high, low, close, p=14):
        up   = high.diff()
        dn   = -low.diff()
        pdm  = pd.Series(np.where((up>dn)&(up>0), up, 0.0), index=close.index)
        ndm  = pd.Series(np.where((dn>up)&(dn>0), dn, 0.0), index=close.index)
        at   = Calc.atr(high, low, close, p)
        pdi  = 100*Calc.ema(pdm, p)/at.replace(0,np.nan)
        ndi  = 100*Calc.ema(ndm, p)/at.replace(0,np.nan)
        dx   = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
        return Calc.ema(dx, p), pdi, ndi

    @staticmethod
    def bb(close, p=20, d=2.0):
        mid = Calc.sma(close, p)
        std = close.rolling(p, min_periods=1).std()
        return mid+d*std, mid, mid-d*std

    @staticmethod
    def vwap(high, low, close, vol):
        tp = (high+low+close)/3
        if vol is not None and vol.sum()>0:
            return (tp*vol).cumsum()/vol.cumsum().replace(0,np.nan)
        return tp

    @staticmethod
    def obv(close, vol):
        return (np.sign(close.diff()).fillna(0)*vol).cumsum()

    @staticmethod
    def add_all(df):
        df = df.copy()
        c=df["Close"]; h=df["High"]; l=df["Low"]
        vol = df.get("Volume", pd.Series(0,index=df.index))
        has_vol = vol is not None and vol.sum()>0

        df["ATR"] = Calc.atr(h,l,c)
        df["RSI"] = Calc.rsi(c)
        m,ms,mh   = Calc.macd(c)
        df["MACD"]=m; df["MACD_S"]=ms; df["MACD_H"]=mh
        adx_,pdi,ndi = Calc.adx(h,l,c)
        df["ADX"]=adx_; df["PDI"]=pdi; df["NDI"]=ndi
        for p in [9,20,50,200]:
            df[f"EMA{p}"] = Calc.ema(c,p)
            df[f"SMA{p}"] = Calc.sma(c,p)
        bu,bm,bl = Calc.bb(c)
        df["BB_U"]=bu; df["BB_M"]=bm; df["BB_L"]=bl; df["BB_W"]=bu-bl
        if has_vol:
            df["VOL_MA"]    = Calc.sma(vol,20)
            df["VOL_RATIO"] = vol/df["VOL_MA"].replace(0,np.nan)
            df["OBV"]       = Calc.obv(c,vol)
        else:
            df["VOL_MA"]=df["VOL_RATIO"]=df["OBV"]=np.nan
        df["VWAP"] = Calc.vwap(h,l,c,vol if has_vol else pd.Series(0,index=df.index))
        w = min(100,len(df))
        rh = h.rolling(w,min_periods=1).max()
        rl = l.rolling(w,min_periods=1).min()
        rng = rh-rl
        for k,f in [("F236",.236),("F382",.382),("F500",.5),("F618",.618),("F786",.786)]:
            df[k] = rh-f*rng
        # Order blocks
        ob_b = pd.Series(np.nan,index=df.index)
        ob_r = pd.Series(np.nan,index=df.index)
        for i in range(2,len(df)-1):
            if df["Open"].iloc[i]>df["Close"].iloc[i] and df["Close"].iloc[i+1]>df["Open"].iloc[i]:
                ob_b.iloc[i] = df["Low"].iloc[i]
            if df["Close"].iloc[i]>df["Open"].iloc[i] and df["Close"].iloc[i+1]<df["Open"].iloc[i]:
                ob_r.iloc[i] = df["High"].iloc[i]
        df["OB_B"]=ob_b; df["OB_R"]=ob_r
        df["LIQ_H"]=h.rolling(20,min_periods=1).max()
        df["LIQ_L"]=l.rolling(20,min_periods=1).min()
        return df

# ──────────────────────────────────────────────────────────────
# SUPPORT / RESISTANCE
# ──────────────────────────────────────────────────────────────
def find_sr(df, win=10):
    h,l = df["High"].values, df["Low"].values
    lvls = []
    for i in range(win, len(df)-win):
        if h[i]==max(h[i-win:i+win+1]): lvls.append(float(h[i]))
        if l[i]==min(l[i-win:i+win+1]): lvls.append(float(l[i]))
    if not lvls: return []
    lvls.sort()
    atr_v = float(df["ATR"].median()) if "ATR" in df else 1.0
    cl=[lvls[0]]
    for x in lvls[1:]:
        if x-cl[-1]>atr_v*0.5: cl.append(x)
    return cl

# ──────────────────────────────────────────────────────────────
# INDICATOR CONFIRMATION
# ──────────────────────────────────────────────────────────────
def ind_ok(df, i, cfg):
    c = df.iloc[i]
    def g(k):
        return float(c[k]) if k in c and pd.notna(c.get(k,np.nan)) else None

    if cfg.get("RSI"):
        r=g("RSI")
        if r is not None and r<50: return False
    if cfg.get("MACD"):
        m,s=g("MACD"),g("MACD_S")
        if m is not None and s is not None and m<s: return False
    if cfg.get("ADX"):
        a=g("ADX")
        if a is not None and a<20: return False
    if cfg.get("EMA"):
        e,cl=g("EMA20"),g("Close")
        if e and cl and cl<e: return False
    if cfg.get("SMA"):
        s,cl=g("SMA20"),g("Close")
        if s and cl and cl<s: return False
    if cfg.get("EMA/SMA Crossover"):
        e,s=g("EMA9"),g("SMA20")
        if e and s and e<s: return False
    if cfg.get("Volume"):
        vr=g("VOL_RATIO")
        if vr is not None and vr<1.0: return False
    if cfg.get("Order Block"):
        ob,cl=g("OB_B"),g("Close")
        if ob and cl and cl<ob: return False
    if cfg.get("VWAP"):
        vw,cl=g("VWAP"),g("Close")
        if vw and cl and cl<vw: return False
    if cfg.get("Volatility Filter"):
        bw,at=g("BB_W"),g("ATR")
        if bw and at and at>0 and bw<at*0.5: return False
    if cfg.get("Fibonacci"):
        f,cl=g("F618"),g("Close")
        if f and cl and cl<f: return False
    if cfg.get("Liquidity Hunt"):
        lh,cl=g("LIQ_H"),g("Close")
        if lh and cl and cl<lh*0.995: return False
    return True

# ──────────────────────────────────────────────────────────────
# ORB STRATEGY
# ──────────────────────────────────────────────────────────────
def orb_signals(df, ind_cfg, or_period=1):
    """
    Rules from strategy image:
    • Stock 1–1.5% momentum  • Entry 09:15-09:45  • No entry after 10:00
    • Exit by 14:30  • RR 1:2  • ½ at 1:1 + SL→CTC
    Anti-fake: close>OR_high, body>40%, ATR filter, volume, resistance check
    """
    sig = pd.Series("", index=df.index)
    try:
        df=df.copy()
        df["_d"]=df.index.date
        df["_h"]=df.index.hour
        df["_m"]=df.index.minute
    except Exception:
        return sig

    for day, ddf in df.groupby("_d"):
        ddf = ddf.sort_index()
        if len(ddf)<or_period+2: continue
        or_bars = ddf.iloc[:or_period]
        or_hi   = float(or_bars["High"].max())
        or_lo   = float(or_bars["Low"].min())
        atr_v   = float(ddf["ATR"].median()) if "ATR" in ddf else max(or_hi-or_lo,1.0)
        sr_lvls = find_sr(ddf, win=min(5,max(2,len(ddf)//4)))
        failed  = 0

        for i in range(or_period, len(ddf)):
            ts = ddf.index[i]
            c  = ddf.iloc[i]
            hr,mn = int(c["_h"]),int(c["_m"])
            if hr>10 or (hr==10 and mn>=0): break
            if hr>9  or (hr==9  and mn>45): break
            if (sig[ddf.index[:i]]=="BUY").any(): break

            cl=float(c["Close"]); op=float(c["Open"])
            hi=float(c["High"]);  lo=float(c["Low"])
            body=abs(cl-op); rng=max(hi-lo,1e-9)

            if hi>or_hi and cl<=or_hi: failed+=1; continue
            if cl<=or_hi: continue
            if (cl-or_hi)<0.25*atr_v: continue
            if body/rng<0.40: continue
            if failed>=2: continue
            vr=float(c.get("VOL_RATIO",1.0) or 1.0)
            if vr>0 and vr<1.1: continue
            above=[x for x in sr_lvls if x>cl]
            if above and (min(above)-cl)<atr_v*1.5: continue
            if not ind_ok(ddf,i,ind_cfg): continue
            sig.at[ts]="BUY"; break
    return sig

def simple_buy_signals(df, ind_cfg):
    sig=pd.Series("",index=df.index)
    for i in range(2,len(df)):
        if float(df["Close"].iloc[i])>float(df["High"].iloc[i-1]):
            if ind_ok(df,i,ind_cfg): sig.iloc[i]="BUY"
    return sig

# ──────────────────────────────────────────────────────────────
# SL / TARGET MANAGER
# ──────────────────────────────────────────────────────────────
def compute_sl(entry, sl_type, p, df=None, idx=-1):
    atr=float(df["ATR"].iloc[idx]) if df is not None and "ATR" in df else entry*0.01
    if atr<=0: atr=entry*0.01
    if sl_type=="Fixed Points":           return entry-p.get("sl_pts",5)
    if sl_type=="ATR Based":              return entry-p.get("sl_atr",1.5)*atr
    if sl_type=="Risk-Reward Based":
        return entry-(entry+p.get("tgt_pts",10)-entry)/max(p.get("rr",2.0),0.1)
    if sl_type=="Volatility (BB) Based":
        if df is not None and "BB_L" in df and idx>=0:
            v=float(df["BB_L"].iloc[idx])
            if not np.isnan(v): return v
        return entry-p.get("sl_atr",2.0)*atr
    # Trailing types — initial = fixed (trailing happens live only)
    if "Candle" in sl_type and df is not None and idx>=1:
        return float(df["Low"].iloc[idx-1])
    if "Swing"  in sl_type and df is not None and idx>=5:
        return float(df["Low"].iloc[max(0,idx-5):idx].min())
    return entry-p.get("sl_pts",5)

def compute_tgt(entry, sl, tgt_type, p, df=None, idx=-1):
    atr=float(df["ATR"].iloc[idx]) if df is not None and "ATR" in df else entry*0.01
    if atr<=0: atr=entry*0.01
    risk=max(entry-sl,0.01)
    if tgt_type=="Fixed Points":           return entry+p.get("tgt_pts",10)
    if tgt_type=="ATR Based":              return entry+p.get("tgt_atr",2.0)*atr
    if tgt_type in ("Risk-Reward Based","Trailing Target (Display Only)"):
        return entry+p.get("rr",2.0)*risk
    if tgt_type=="Volatility (BB) Based":
        if df is not None and "BB_U" in df and idx>=0:
            v=float(df["BB_U"].iloc[idx])
            if not np.isnan(v): return v
        return entry+p.get("rr",2.0)*risk
    return entry+p.get("tgt_pts",10)

def trail_sl(cur_sl, sl_type, ltp, entry, p, df=None, idx=-1, hi=0.0):
    """Live trailing only."""
    if sl_type=="Trailing (Points)":       return max(cur_sl, ltp-p.get("sl_pts",5))
    if sl_type=="Trailing (Candle Low/High)":
        if df is not None and idx>=1: return max(cur_sl, float(df["Low"].iloc[idx-1]))
    if sl_type=="Trailing (Swing Low/High)":
        if df is not None and idx>=5: return max(cur_sl, float(df["Low"].iloc[max(0,idx-5):idx].min()))
    if sl_type=="ATR Based":
        atr=float(df["ATR"].iloc[idx]) if df is not None and "ATR" in df else 0
        if atr>0: return max(cur_sl, ltp-p.get("sl_atr",1.5)*atr)
    return cur_sl

# ──────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────
def run_backtest(df, strategy, sl_type, tgt_type, params, ind_cfg):
    """
    • Entry on candle N+1 OPEN
    • SL checked vs LOW first (conservative)
    • Target checked vs HIGH
    • NO trailing SL (no tick data in backtest)
    • Partial ½ at 1:1 + SL→CTC
    • Force exit 14:30
    """
    df = Calc.add_all(df)
    bt_sl = sl_type if sl_type in SL_BT else "Fixed Points"

    if strategy=="ORB (Opening Range Breakout)": sigs=orb_signals(df,ind_cfg)
    else: sigs=simple_buy_signals(df,ind_cfg)

    trades=[]; capital=float(params.get("init_cap",100000)); equity=[capital]
    qty=int(params.get("qty",1)); in_trade=False
    ep=sl=tgt=t1=0.0; et=None; part=False

    try:
        df["_h"]=df.index.hour; df["_m"]=df.index.minute
    except Exception:
        pass

    for i in range(len(df)):
        c=df.iloc[i]; ts=df.index[i]
        op=float(c["Open"]); hi=float(c["High"]); lo=float(c["Low"]); cl=float(c["Close"])
        hr=int(c.get("_h",0)); mn=int(c.get("_m",0))

        # Force exit 14:30
        if in_trade and hr==14 and mn>=30:
            pnl=(cl-ep)*qty*0.99; capital+=pnl
            trades.append(_mk(et,ts,ep,cl,sl,tgt,pnl,"Force 14:30",capital))
            equity.append(capital); in_trade=part=False; continue

        if in_trade:
            # ½ at 1:1
            if not part and hi>=t1:
                part=True; capital+=(t1-ep)*(qty//2); sl=ep  # CTC

            # SL vs LOW first
            if lo<=sl:
                rem=(qty//2) if part else qty
                pnl=(sl-ep)*rem*0.99; capital+=pnl
                trades.append(_mk(et,ts,ep,sl,sl,tgt,pnl,"SL Hit",capital))
                equity.append(capital); in_trade=part=False; continue

            # Target vs HIGH
            if hi>=tgt:
                rem=(qty//2) if part else qty
                pnl=(tgt-ep)*rem*0.99; capital+=pnl
                trades.append(_mk(et,ts,ep,tgt,sl,tgt,pnl,"Target Hit",capital))
                equity.append(capital); in_trade=part=False; continue

        # Entry on N+1 open
        if not in_trade and i>0 and sigs.iloc[i-1]=="BUY":
            ep=op; et=ts
            sl =compute_sl(ep,bt_sl,params,df,i)
            tgt=compute_tgt(ep,sl,tgt_type,params,df,i)
            t1 =ep+(ep-sl); part=False; in_trade=True

    if in_trade and len(df):
        cl_=float(df["Close"].iloc[-1]); pnl=(cl_-ep)*qty*0.99; capital+=pnl
        trades.append(_mk(et,df.index[-1],ep,cl_,sl,tgt,pnl,"End of Data",capital))
        equity.append(capital)

    return _summarise(trades, equity, params.get("init_cap",100000))

def _mk(et,xt,ep,xp,sl,tgt,pnl,reason,cap):
    return dict(entry_time=str(et)[:19],exit_time=str(xt)[:19],
                entry_price=round(ep,4),exit_price=round(xp,4),
                sl=round(sl,4),target=round(tgt,4),pnl=round(pnl,2),
                reason=reason,capital=round(cap,2),
                status="WIN" if pnl>=0 else "LOSS")

def _summarise(trades, equity, init):
    if not trades: return dict(trades=[],equity=equity,summary={})
    df=pd.DataFrame(trades)
    wins=int((df["pnl"]>=0).sum()); total=len(df); losses=total-wins
    gross=df["pnl"].sum(); final=equity[-1] if equity else init
    peak=init; dd=0.0
    for v in equity: peak=max(peak,v); dd=max(dd,peak-v)
    gw=df.loc[df["pnl"]>=0,"pnl"].sum(); gl=abs(df.loc[df["pnl"]<0,"pnl"].sum())
    pf=gw/gl if gl>0 else float("inf")
    arr=df["pnl"].values
    sharpe=(arr.mean()/arr.std()*math.sqrt(252)) if len(arr)>1 and arr.std()>0 else 0.0
    return dict(trades=trades,equity=equity,summary=dict(
        total_trades=total,wins=wins,losses=losses,
        win_rate=round(wins/total*100,1) if total else 0,
        total_pnl=round(gross,2),init_cap=round(init,2),
        final_cap=round(final,2),return_pct=round((final-init)/init*100,2),
        max_dd=round(dd,2),profit_factor=round(pf,2),
        avg_win=round(df.loc[df["pnl"]>=0,"pnl"].mean(),2) if wins else 0,
        avg_loss=round(df.loc[df["pnl"]<0,"pnl"].mean(),2) if losses else 0,
        sharpe=round(sharpe,2)))

# ──────────────────────────────────────────────────────────────
# DHAN ORDER MANAGER
# ──────────────────────────────────────────────────────────────
def place_equity_order(dcfg, direction, ltp=0.0):
    """
    Uses pydhan. direction='BUY' or 'SELL'.
    For intraday/delivery. Main algo BUY→order BUY, SELL→order SELL.
    """
    cid  = st.session_state.dhan_cid
    tok  = st.session_state.dhan_tok
    secid= str(dcfg.get("eq_sec","1594"))
    qty  = int(dcfg.get("eq_qty",1))
    prod = dcfg.get("eq_prod","INTRADAY")    # INTRADAY or DELIVERY
    exch = dcfg.get("eq_exch","NSE")         # NSE or BSE
    # Entry order type for BUY, exit order type for SELL
    otype= dcfg.get("eq_entry_ot","LIMIT") if direction=="BUY" else dcfg.get("eq_exit_ot","MARKET")
    price= round(ltp,2) if otype=="LIMIT" else 0.0

    info = dict(type="EQUITY",direction=direction,security_id=secid,
                qty=qty,product=prod,exchange=exch,order_type=otype,
                price=price,timestamp=str(datetime.now())[:19])

    if dcfg.get("paper_mode"):
        info["status"]="PAPER_TRADE"
        info["order_id"]=f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        return info

    if not PYDHAN_OK:
        info["status"]="ERROR"; info["message"]="pydhan not installed"; return info
    if not cid or not tok:
        info["status"]="ERROR"; info["message"]="Credentials missing"; return info
    try:
        d = _Pydhan(client_id=cid, access_token=tok)
        exch_seg  = d.NSE if exch=="NSE" else d.BSE
        prod_type = d.INTRADAY if prod=="INTRADAY" else d.DELIVERY
        ot        = d.LIMIT if otype=="LIMIT" else d.MARKET
        txn       = d.BUY  if direction=="BUY" else d.SELL
        resp = d.place_order(security_id=secid, exchange_segment=exch_seg,
                             transaction_type=txn, quantity=qty,
                             order_type=ot, product_type=prod_type, price=price)
        info["status"]="PLACED"
        info["order_id"]=resp.get("orderId","") if isinstance(resp,dict) else ""
        info["response"]=str(resp)
        return info
    except Exception as e:
        info["status"]="ERROR"; info["message"]=str(e); return info

def place_option_order(dcfg, direction, ltp=0.0):
    """
    Uses dhanhq. direction='BUY'→buy CE, 'SELL'→buy PE. Always buyer.
    """
    cid = st.session_state.dhan_cid
    tok = st.session_state.dhan_tok
    exch= dcfg.get("fno_exch","NSE_FNO")
    if direction=="BUY":
        secid=str(dcfg.get("ce_sec",""));  opt="CE"
    else:
        secid=str(dcfg.get("pe_sec",""));  opt="PE"
    qty  = int(dcfg.get("fno_qty",65))
    otype= dcfg.get("fno_entry_ot","MARKET") if direction=="BUY" else dcfg.get("fno_exit_ot","MARKET")
    price= round(ltp,2) if otype=="LIMIT" else 0.0

    info = dict(type=f"OPTIONS_{opt}",direction="BUY",option=opt,
                security_id=secid,qty=qty,exchange=exch,
                order_type=otype,price=price,timestamp=str(datetime.now())[:19])

    if dcfg.get("paper_mode"):
        info["status"]="PAPER_TRADE"
        info["order_id"]=f"PAPER-OPT-{uuid.uuid4().hex[:6].upper()}"
        return info

    if not DHANHQ_OK:
        info["status"]="ERROR"; info["message"]="dhanhq not installed"; return info
    if not cid or not tok:
        info["status"]="ERROR"; info["message"]="Credentials missing"; return info
    if not secid:
        info["status"]="ERROR"; info["message"]=f"{opt} Security ID missing"; return info
    try:
        d = _Dhanhq(cid, tok)
        resp = d.place_order(transactionType="BUY",exchangeSegment=exch,
                             productType="INTRADAY",orderType=otype,
                             validity="DAY",securityId=secid,
                             quantity=qty,price=price,triggerPrice=0)
        info["status"]="PLACED"
        info["order_id"]=resp.get("orderId","") if isinstance(resp,dict) else ""
        info["response"]=str(resp)
        return info
    except Exception as e:
        info["status"]="ERROR"; info["message"]=str(e); return info

def execute_dhan(direction, dcfg, ltp=0.0):
    """Route to correct order function based on config."""
    if not dcfg.get("enable_dhan"): return None
    if dcfg.get("enable_options"):  return place_option_order(dcfg,direction,ltp)
    else:                           return place_equity_order(dcfg,direction,ltp)

def log_order(result):
    if result: st.session_state.order_log.append(result)

# ──────────────────────────────────────────────────────────────
# rgba helper (avoids 8-digit hex in Plotly)
# ──────────────────────────────────────────────────────────────
def rgba(hex6, a):
    h=hex6.lstrip("#")
    r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

# ──────────────────────────────────────────────────────────────
# CHART BUILDER
# ──────────────────────────────────────────────────────────────
def build_chart(df, trades=None, sigs=None, ind_cfg=None, title="Price"):
    if ind_cfg is None: ind_cfg={}
    rows=1; specs=[[{"type":"candlestick"}]]; rh=[0.55]
    sv=ind_cfg.get("Volume") and "Volume" in df and df["Volume"].sum()>0
    sr=ind_cfg.get("RSI")    and "RSI" in df
    sm=ind_cfg.get("MACD")   and "MACD" in df
    if sv: rows+=1; specs.append([{"type":"bar"}]);     rh.append(0.15)
    if sr: rows+=1; specs.append([{"type":"scatter"}]); rh.append(0.15)
    if sm: rows+=1; specs.append([{"type":"scatter"}]); rh.append(0.15)
    tot=sum(rh); rh=[x/tot for x in rh]

    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,
                      vertical_spacing=0.03,row_heights=rh,specs=specs)

    fig.add_trace(go.Candlestick(
        x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        increasing_line_color="#10b981",decreasing_line_color="#ef4444",
        increasing_fillcolor=rgba("10b981",.18),decreasing_fillcolor=rgba("ef4444",.18),
        name="Price",line_width=1,
    ),row=1,col=1)

    if ind_cfg.get("EMA"):
        for p,clr in [(9,"#f59e0b"),(20,"#3b82f6"),(50,"#8b5cf6"),(200,"#ec4899")]:
            k=f"EMA{p}"
            if k in df:
                fig.add_trace(go.Scatter(x=df.index,y=df[k],name=k,
                    line=dict(color=clr,width=1),opacity=0.85),row=1,col=1)

    if ind_cfg.get("SMA"):
        for p,clr in [(20,"#06b6d4"),(50,"#a78bfa")]:
            k=f"SMA{p}"
            if k in df:
                fig.add_trace(go.Scatter(x=df.index,y=df[k],name=k,
                    line=dict(color=clr,width=1,dash="dash"),opacity=0.75),row=1,col=1)

    if ind_cfg.get("Volatility Filter") and "BB_U" in df:
        for k,clr in [("BB_U","#94a3b8"),("BB_M","#64748b"),("BB_L","#94a3b8")]:
            fig.add_trace(go.Scatter(x=df.index,y=df[k],name=k,
                line=dict(color=clr,width=1,dash="dot"),opacity=0.45),row=1,col=1)

    if ind_cfg.get("VWAP") and "VWAP" in df:
        fig.add_trace(go.Scatter(x=df.index,y=df["VWAP"],name="VWAP",
            line=dict(color="#f97316",width=1.5)),row=1,col=1)

    if ind_cfg.get("Fibonacci"):
        for fk,clr in [("F382","#fbbf24"),("F500","#fb923c"),("F618","#f87171")]:
            if fk in df:
                fig.add_trace(go.Scatter(x=df.index,y=df[fk],
                    name=fk.replace("F","Fib."),
                    line=dict(color=clr,width=1,dash="dot"),opacity=0.6),row=1,col=1)

    if ind_cfg.get("Order Block") and "OB_B" in df:
        ob=df["OB_B"].dropna()
        if len(ob):
            fig.add_trace(go.Scatter(x=ob.index,y=ob.values,mode="markers",
                marker=dict(symbol="square",size=8,color="#10b981",opacity=0.7),
                name="Bull OB"),row=1,col=1)

    if sigs is not None:
        bi=sigs[sigs=="BUY"].index
        if len(bi):
            fig.add_trace(go.Scatter(x=bi,y=df.loc[bi,"Low"]*0.999,mode="markers",
                marker=dict(symbol="triangle-up",size=12,color="#10b981"),
                name="BUY"),row=1,col=1)

    cur=2
    if sv:
        clrs=["#10b981" if float(df["Close"].iloc[j])>=float(df["Open"].iloc[j])
              else "#ef4444" for j in range(len(df))]
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=clrs,
            opacity=0.5,name="Vol"),row=cur,col=1)
        if "VOL_MA" in df:
            fig.add_trace(go.Scatter(x=df.index,y=df["VOL_MA"],name="VolMA",
                line=dict(color="#f59e0b",width=1)),row=cur,col=1)
        fig.update_yaxes(title_text="Vol",row=cur,col=1); cur+=1

    if sr:
        fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],name="RSI",
            line=dict(color="#a78bfa",width=1.5)),row=cur,col=1)
        for lvl,clr in [(70,"#ef4444"),(50,"#94a3b8"),(30,"#10b981")]:
            fig.add_hline(y=lvl,line=dict(color=clr,width=0.8,dash="dot"),row=cur,col=1)
        fig.update_yaxes(title_text="RSI",range=[0,100],row=cur,col=1); cur+=1

    if sm:
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],name="MACD",
            line=dict(color="#3b82f6",width=1.5)),row=cur,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["MACD_S"],name="Sig",
            line=dict(color="#f59e0b",width=1,dash="dash")),row=cur,col=1)
        hc=["#10b981" if (df["MACD_H"].iloc[j] or 0)>=0 else "#ef4444" for j in range(len(df))]
        fig.add_trace(go.Bar(x=df.index,y=df["MACD_H"],marker_color=hc,
            opacity=0.5,name="Hist"),row=cur,col=1)
        fig.update_yaxes(title_text="MACD",row=cur,col=1)

    fig.update_layout(
        title=dict(text=title,font=dict(family="JetBrains Mono",size=13,color="#e2e8f0")),
        paper_bgcolor="#0a0e17",plot_bgcolor="#0a0e17",
        font=dict(color="#94a3b8",family="JetBrains Mono",size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#111827",bordercolor="#1e2d45",borderwidth=1,font=dict(size=10)),
        margin=dict(l=50,r=20,t=40,b=20),
        height=570 if rows==1 else 570+(rows-1)*130,
    )
    for r in range(1,rows+1):
        fig.update_xaxes(showgrid=True,gridcolor="#1e2d45",linecolor="#1e2d45",row=r,col=1)
        fig.update_yaxes(showgrid=True,gridcolor="#1e2d45",linecolor="#1e2d45",row=r,col=1)
    return fig

def equity_chart(eq, init):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=eq,mode="lines",
        line=dict(color="#00d4aa",width=2),
        fill="tozeroy",fillcolor=rgba("00d4aa",.07)))
    fig.add_hline(y=init,line=dict(color="#94a3b8",width=1,dash="dot"))
    fig.update_layout(
        title=dict(text="Equity Curve",font=dict(family="JetBrains Mono",size=12,color="#e2e8f0")),
        paper_bgcolor="#0a0e17",plot_bgcolor="#0a0e17",
        font=dict(color="#94a3b8",family="JetBrains Mono",size=10),
        margin=dict(l=40,r=10,t=35,b=15),height=240,
    )
    fig.update_xaxes(showgrid=True,gridcolor="#1e2d45")
    fig.update_yaxes(showgrid=True,gridcolor="#1e2d45",tickprefix="₹")
    return fig

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:6px 0 14px'>
          <div style='font-family:JetBrains Mono;font-size:16px;font-weight:700;
                      color:#00d4aa;letter-spacing:.1em'>⚡ TRADING LAB</div>
          <div style='font-size:9px;color:#334155;font-family:JetBrains Mono;
                      letter-spacing:.2em'>PROFESSIONAL PLATFORM</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Instrument ──
        st.markdown("**📌 INSTRUMENT**")
        tk_name = st.selectbox("Ticker", list(TICKER_MAP.keys()), index=0, key="sb_tk")
        custom  = ""
        if tk_name=="CUSTOM":
            custom = st.text_input("Custom Symbol","AAPL",key="sb_cu")
        sym = DF.resolve(tk_name, custom)
        st.caption(f"Symbol → `{sym}`")

        tf_keys=list(TF_MAP.keys()); pr_keys=list(PR_MAP.keys())
        tf = st.selectbox("Timeframe",tf_keys,index=tf_keys.index("5m"),key="sb_tf")
        pr = st.selectbox("Period",   pr_keys,index=pr_keys.index("1 Month"),key="sb_pr")

        tf_code=TF_MAP[tf]; pr_code=PR_MAP[pr]
        allowed=VALID.get(tf_code,list(PR_MAP.values()))
        if pr_code not in allowed:
            pr_code=allowed[-1]
            fk=[k for k,v in PR_MAP.items() if v==pr_code]
            if fk: st.warning(f"Period auto → **{fk[0]}**")

        st.divider()
        # ── Strategy ──
        st.markdown("**🎯 STRATEGY**")
        strategy = st.selectbox("Strategy",STRATEGIES,key="sb_st")
        qty      = st.number_input("Quantity",1,100000,1,key="sb_qty")
        init_cap = st.number_input("Capital (₹)",1000,10000000,100000,step=10000,key="sb_cap")

        st.divider()
        # ── SL & Target ──
        st.markdown("**🛡️ SL & TARGET**")
        sl_type  = st.selectbox("Stop Loss Type",SL_LT,key="sb_sl")
        tgt_type = st.selectbox("Target Type",TGT_TYPES,key="sb_tgt")

        c1,c2=st.columns(2)
        sl_pts  = c1.number_input("SL Pts", 0.1,100000.0,5.0, step=0.5,key="sb_slp")
        tgt_pts = c2.number_input("Tgt Pts",0.1,100000.0,10.0,step=0.5,key="sb_tp")
        c3,c4=st.columns(2)
        sl_atr  = c3.number_input("SL ATR×", 0.1,10.0,1.5,step=0.1,key="sb_sa")
        tgt_atr = c4.number_input("Tgt ATR×",0.1,10.0,2.0,step=0.1,key="sb_ta")
        rr = st.slider("Risk:Reward 1:N",1.0,5.0,2.0,0.5,key="sb_rr")

        if sl_type not in SL_BT:
            st.markdown('<div class="wbox">ℹ️ Trailing SL works in <b>Live only</b>. '
                        'Backtest uses Fixed Points.</div>',unsafe_allow_html=True)
        if tgt_type=="Trailing Target (Display Only)":
            st.info("📊 Trailing Target: display only, never exits.")

        st.divider()
        # ── Indicators ──
        st.markdown("**📊 INDICATORS** *(all off = no filter)*")
        ind_cfg={}
        ic1,ic2=st.columns(2)
        for j,ind in enumerate(IND_LIST):
            safe=ind.replace("/","_").replace(" ","_")
            ind_cfg[ind]=(ic1 if j%2==0 else ic2).checkbox(
                ind,value=False,key=f"i_{safe}")

        st.divider()
        # ── Dhan Broker ──
        st.markdown("**🔑 DHAN BROKER**")
        en_dhan = st.checkbox("Enable Dhan Broker",value=False,key="sb_den")
        dcfg    = {"enable_dhan":en_dhan}

        if en_dhan:
            cid = st.text_input("Client ID",    st.session_state.dhan_cid, key="sb_cid")
            tok = st.text_input("Access Token", st.session_state.dhan_tok,
                                type="password", key="sb_tok")
            if cid: st.session_state.dhan_cid = cid
            if tok: st.session_state.dhan_tok  = tok

            paper = st.checkbox("📋 Placeholder / Paper Trade Mode",
                                value=False, key="sb_paper",
                                help="Simulates orders — nothing sent to Dhan")
            dcfg["paper_mode"] = paper
            if paper:
                st.markdown('<div class="pbox">🧾 <b>Paper Trade ON</b> — orders simulated, '
                            'NOT sent to broker.</div>',unsafe_allow_html=True)

            en_opt = st.checkbox("📈 Options Trading",value=False,key="sb_opt")
            dcfg["enable_options"] = en_opt

            if not en_opt:
                st.markdown("*── Equity Settings ──*")
                eq_prod  = st.selectbox("Product",["INTRADAY","DELIVERY"],index=0,key="sb_eprod")
                eq_exch  = st.selectbox("Exchange",["NSE","BSE"],index=0,key="sb_eexch")
                eq_sec   = st.text_input("Security ID","1594",key="sb_esec")
                eq_qty   = st.number_input("Qty",1,100000,1,key="sb_eqty")
                eq_entry = st.selectbox("Entry Order",["LIMIT","MARKET"],index=0,key="sb_eentry")
                eq_exit  = st.selectbox("Exit Order", ["MARKET","LIMIT"],index=0,key="sb_eexit")
                if eq_entry=="LIMIT": st.caption("ℹ️ Limit price = current LTP")
                dcfg.update(dict(eq_prod=eq_prod,eq_exch=eq_exch,eq_sec=eq_sec,
                                 eq_qty=eq_qty,eq_entry_ot=eq_entry,eq_exit_ot=eq_exit))
            else:
                st.markdown("*── Options Settings ──*")
                fno_exch  = st.selectbox("F&O Exch",["NSE_FNO","BSE_FNO"],index=0,key="sb_fexch")
                ce_sec    = st.text_input("CE Security ID","",key="sb_ce")
                pe_sec    = st.text_input("PE Security ID","",key="sb_pe")
                fno_qty   = st.number_input("Lot Qty",1,100000,65,key="sb_fqty")
                fno_entry = st.selectbox("Entry Order",["MARKET","LIMIT"],index=0,key="sb_fentry")
                fno_exit  = st.selectbox("Exit Order", ["MARKET","LIMIT"],index=0,key="sb_fexit")
                if fno_entry=="LIMIT": st.caption("ℹ️ Limit price = current LTP")
                st.caption("Buy signal→Buy CE  |  Sell signal→Buy PE")
                dcfg.update(dict(fno_exch=fno_exch,ce_sec=ce_sec,pe_sec=pe_sec,
                                 fno_qty=fno_qty,fno_entry_ot=fno_entry,fno_exit_ot=fno_exit))

            if st.button("🔗 Test Connection",key="sb_test"):
                if paper:
                    st.info("📋 Paper mode — no real connection needed.")
                elif cid and tok:
                    with st.spinner("Testing…"):
                        try:
                            if not en_opt and PYDHAN_OK:
                                _Pydhan(client_id=cid,access_token=tok)
                                st.success("✅ pydhan OK!")
                            elif en_opt and DHANHQ_OK:
                                _Dhanhq(cid,tok)
                                st.success("✅ dhanhq OK!")
                            else:
                                st.warning("Install pydhan/dhanhq first.")
                        except Exception as e:
                            st.error(f"❌ {e}")
                else:
                    st.warning("Enter credentials first.")

    return dict(sym=sym,tf=tf_code,pr=pr_code,strategy=strategy,
                qty=qty,init_cap=init_cap,sl_type=sl_type,tgt_type=tgt_type,
                sl_pts=sl_pts,tgt_pts=tgt_pts,sl_atr=sl_atr,tgt_atr=tgt_atr,
                rr=rr,ind_cfg=ind_cfg), dcfg

# ──────────────────────────────────────────────────────────────
# BACKTESTING TAB
# ──────────────────────────────────────────────────────────────
def mc(col,lbl,val,clr="cacc"):
    col.markdown(f'<div class="mcard"><div class="mlabel">{lbl}</div>'
                 f'<div class="mval {clr}">{val}</div></div>',unsafe_allow_html=True)

def tab_bt(tcfg, dcfg):
    st.markdown("""
    <div style='padding:2px 0 12px'>
      <span style='font-family:JetBrains Mono;font-size:15px;font-weight:700;color:#00d4aa'>
        ◈ BACKTESTING ENGINE</span>
      <span style='font-size:10px;color:#475569;font-family:JetBrains Mono;margin-left:10px'>
        Entry N+1 Open · SL vs Low · Target vs High · No Trailing SL</span>
    </div>""",unsafe_allow_html=True)

    c1,c2,_=st.columns([1,1,5])
    run=c1.button("▶ RUN BACKTEST",type="primary",key="bt_run")
    clr=c2.button("🗑 CLEAR",key="bt_clr")

    if clr:
        st.session_state.bt_results=None
        st.session_state.bt_trades=[]
        st.session_state.bt_equity=[]
        st.session_state.bt_df=None
        st.session_state.bt_sigs=None
        st.rerun()

    if run:
        with st.spinner("📡 Fetching data…"):
            df=DF.fetch(tcfg["sym"],tcfg["tf"],tcfg["pr"])
        if df is None or df.empty:
            st.error("❌ No data. Check ticker/timeframe/period."); return
        params=dict(init_cap=tcfg["init_cap"],qty=tcfg["qty"],
                    sl_pts=tcfg["sl_pts"],tgt_pts=tcfg["tgt_pts"],
                    sl_atr=tcfg["sl_atr"],tgt_atr=tcfg["tgt_atr"],rr=tcfg["rr"])
        with st.spinner("⚙️ Backtesting…"):
            res=run_backtest(df,tcfg["strategy"],tcfg["sl_type"],
                             tcfg["tgt_type"],params,tcfg["ind_cfg"])
        st.session_state.bt_results=res["summary"]
        st.session_state.bt_trades =res["trades"]
        st.session_state.bt_equity =res["equity"]
        st.session_state.bt_df     =df
        df_i=Calc.add_all(df)
        if tcfg["strategy"].startswith("ORB"):
            st.session_state.bt_sigs=orb_signals(df_i,tcfg["ind_cfg"])
        else:
            st.session_state.bt_sigs=simple_buy_signals(df_i,tcfg["ind_cfg"])

    if not st.session_state.bt_results:
        st.markdown('<div class="ibox">⬆️ Configure in sidebar → click <b>RUN BACKTEST</b>.'
                    '<br>ORB: entry 09:15–09:45 · exit 14:30 · ½ close at 1:1 · SL→CTC</div>',
                    unsafe_allow_html=True)
        return

    s=st.session_state.bt_results
    st.markdown('<div class="sh">📈 PERFORMANCE SUMMARY</div>',unsafe_allow_html=True)
    m=st.columns(8)
    mc(m[0],"Trades",    s["total_trades"])
    mc(m[1],"Win Rate",  f"{s['win_rate']}%","cgreen" if s["win_rate"]>=50 else "cred")
    mc(m[2],"Total P&L", f"₹{s['total_pnl']:,.0f}","cgreen" if s["total_pnl"]>=0 else "cred")
    mc(m[3],"Return",    f"{s['return_pct']}%","cgreen" if s["return_pct"]>=0 else "cred")
    mc(m[4],"Max DD",    f"₹{s['max_dd']:,.0f}","cred")
    mc(m[5],"Profit F.", f"{s['profit_factor']:.2f}","cgreen" if s["profit_factor"]>=1 else "cred")
    mc(m[6],"Sharpe",    f"{s['sharpe']:.2f}","cgreen" if s["sharpe"]>=1 else "cgold")
    mc(m[7],"Final Cap", f"₹{s['final_cap']:,.0f}")

    a1,a2=st.columns(2)
    a1.metric("Avg Win",  f"₹{s['avg_win']:,.2f}")
    a2.metric("Avg Loss", f"₹{s['avg_loss']:,.2f}")

    if st.session_state.bt_equity:
        st.plotly_chart(equity_chart(st.session_state.bt_equity,tcfg["init_cap"]),
                        use_container_width=True)

    if st.session_state.bt_df is not None:
        df_i=Calc.add_all(st.session_state.bt_df)
        fig=build_chart(df_i,st.session_state.bt_trades,
                        st.session_state.bt_sigs,tcfg["ind_cfg"],
                        f"{tcfg['sym']} · {tcfg['tf']} · {tcfg['strategy']}")
        st.plotly_chart(fig,use_container_width=True)

    if st.session_state.bt_trades:
        st.markdown('<div class="sh" style="margin-top:16px">📋 TRADE LOG</div>',unsafe_allow_html=True)
        df_t=pd.DataFrame(st.session_state.bt_trades)
        st.dataframe(df_t[["entry_time","exit_time","entry_price","exit_price",
                            "sl","target","pnl","reason","status","capital"]],
                     use_container_width=True,hide_index=True,
                     column_config={"pnl":st.column_config.NumberColumn("P&L ₹",format="%.2f")})
        st.download_button("⬇ CSV",df_t.to_csv(index=False),
                           f"bt_{tcfg['sym']}_{datetime.now():%Y%m%d}.csv",
                           "text/csv",key="bt_dl")

# ──────────────────────────────────────────────────────────────
# LIVE TRADING TAB
# ──────────────────────────────────────────────────────────────
def tab_live(tcfg, dcfg):
    st.markdown("""
    <div style='padding:2px 0 12px'>
      <span style='font-family:JetBrains Mono;font-size:15px;font-weight:700;color:#00d4aa'>
        ◈ LIVE TRADING</span>
      <span style='font-size:10px;color:#475569;font-family:JetBrains Mono;margin-left:10px'>
        Auto-refresh 1.5s · Trailing SL active · Dhan API</span>
    </div>""",unsafe_allow_html=True)

    # Config strip
    dhan_info = "✅ ON" if dcfg.get("enable_dhan") else "⛔ OFF"
    if dcfg.get("paper_mode"): dhan_info+=" 📋Paper"
    if dcfg.get("enable_options"): dhan_info+=" 📈Opts"
    st.markdown(
        f'<div class="ibox" style="font-family:JetBrains Mono;font-size:11px">'
        f'<b>Ticker:</b>{tcfg["sym"]} &nbsp;|&nbsp;<b>TF:</b>{tcfg["tf"]} &nbsp;|&nbsp;'
        f'<b>Strategy:</b>{tcfg["strategy"]} &nbsp;|&nbsp;'
        f'<b>Qty:</b>{tcfg["qty"]} &nbsp;|&nbsp;<b>SL:</b>{tcfg["sl_type"]} &nbsp;|&nbsp;'
        f'<b>Dhan:</b>{dhan_info}</div>',unsafe_allow_html=True)

    # Sector scanner
    with st.expander("🔭 NSE Sector Scanner (1–1.5% momentum)",expanded=False):
        sels=st.multiselect("Sectors",list(NSE_SECTORS.keys()),
                            default=["🏦 Banking","💻 IT/Tech"],key="lt_sec")
        if st.button("🔍 Scan",key="lt_sc"):
            results=[]
            tks=[t for s in sels for t in NSE_SECTORS.get(s,[])[:4]]
            prg=st.progress(0)
            for j,tk in enumerate(tks):
                prg.progress((j+1)/max(len(tks),1))
                m=DF.mom_pct(tk)
                if 1.0<=m<=1.5: results.append({"Ticker":tk,"Momentum%":round(m,2)})
            prg.empty()
            if results: st.dataframe(pd.DataFrame(results),use_container_width=True,hide_index=True)
            else: st.warning("No stocks in 1–1.5% range now.")

    st.divider()

    # Controls
    b1,b2,b3,b4=st.columns([1,1,1,1])
    if b1.button("▶ START",type="primary",key="lt_st",
                 disabled=st.session_state.live_active):
        st.session_state.live_active    = True
        st.session_state.live_ticker    = tcfg["sym"]
        st.session_state.partial_closed = False
        st.session_state.trailing_high  = 0.0
        st.session_state.live_log.clear()
        st.rerun()

    if b2.button("⏹ STOP",key="lt_sp",
                 disabled=not st.session_state.live_active):
        st.session_state.live_active=False; st.rerun()

    if b3.button("🔪 SQUARE OFF",key="lt_sq",
                 disabled=st.session_state.live_position is None):
        _sq_off(tcfg,dcfg); st.rerun()

    if b4.button("🔄 REFRESH",key="lt_rf"): st.rerun()

    if st.session_state.live_active:
        st.markdown('<span class="ldot"></span>'
                    '<span style="font-family:JetBrains Mono;font-size:11px;color:#10b981">'
                    'LIVE · Auto-refresh 1.5s</span>',unsafe_allow_html=True)
    else:
        st.markdown('<span style="font-size:11px;color:#475569;'
                    'font-family:JetBrains Mono">⏸  STOPPED</span>',unsafe_allow_html=True)

    _live_metrics(tcfg,dcfg)

    if st.session_state.live_log:
        with st.expander("📟 Activity Log",expanded=False):
            for msg in reversed(list(st.session_state.live_log)):
                st.caption(msg)

    if st.session_state.order_log:
        with st.expander(f"📦 Order Log ({len(st.session_state.order_log)} orders)",expanded=False):
            for o in reversed(st.session_state.order_log[-20:]):
                st_color = "#4ade80" if o.get("status")=="PLACED" else \
                           "#c4b5fd" if o.get("status")=="PAPER_TRADE" else "#f87171"
                st.markdown(
                    f'<div class="ocard">'
                    f'<b>{o.get("type","")}</b> {o.get("direction","")} '
                    f'{"["+o.get("option","")+"]" if o.get("option") else ""} '
                    f'| Sec:{o.get("security_id","")} | Qty:{o.get("qty","")} '
                    f'| {o.get("order_type","")} @ ₹{o.get("price",0):.2f} '
                    f'| <span style="color:{st_color}">{o.get("status","")}</span>'
                    f'| ID:{o.get("order_id","—")}'
                    f'<span style="float:right;color:#334155">{o.get("timestamp","")}</span>'
                    f'</div>',unsafe_allow_html=True)

    if st.session_state.live_active:
        time.sleep(1.5)
        st.rerun()


def _live_metrics(tcfg, dcfg):
    ltp=st.session_state.live_ltp; df_l=None

    if st.session_state.live_active:
        ticker=st.session_state.live_ticker or tcfg["sym"]
        df_l=DF.fetch(ticker,tcfg["tf"],tcfg["pr"])
        if df_l is not None and not df_l.empty:
            df_l=Calc.add_all(df_l)
            ltp=float(df_l["Close"].iloc[-1])
            st.session_state.live_ltp=ltp
            _process_live(df_l,ltp,tcfg,dcfg)

    pos=st.session_state.live_position
    st.markdown('<div class="sh">📡 LIVE METRICS</div>',unsafe_allow_html=True)
    m=st.columns(6)
    m[0].metric("LTP",f"₹{ltp:,.2f}" if ltp else "—")

    if pos:
        ep=pos["entry"]; sl_=pos["sl"]; tgt_=pos["target"]
        pnl_=(ltp-ep)*pos["qty"]
        m[1].metric("Entry",  f"₹{ep:,.2f}")
        m[2].metric("SL",     f"₹{sl_:,.2f}",delta=f"{ltp-sl_:+.2f}")
        m[3].metric("Target", f"₹{tgt_:,.2f}",delta=f"{tgt_-ltp:+.2f}")
        m[4].metric("P & L",  f"₹{pnl_:,.2f}",delta=f"{pnl_:+.2f}",
                    delta_color="normal" if pnl_>=0 else "inverse")
        m[5].metric("Qty",    pos["qty"])

        if tcfg["tgt_type"]=="Trailing Target (Display Only)" and df_l is not None:
            p=dict(sl_pts=tcfg["sl_pts"],tgt_pts=tcfg["tgt_pts"],
                   rr=tcfg["rr"],sl_atr=tcfg["sl_atr"],tgt_atr=tcfg["tgt_atr"])
            trail=compute_tgt(ltp,sl_,"Risk-Reward Based",p,df_l,-1)
            st.markdown(f'<div class="wbox">📊 <b>Trailing Target (Display):</b> '
                        f'₹{trail:,.2f} — tracks price, does NOT exit.</div>',unsafe_allow_html=True)

        pc=st.columns(4)
        for col,lbl,val,clr in [
            (pc[0],"ENTRY",    f"₹{ep:,.2f}",    "cacc"),
            (pc[1],"STOP LOSS",f"₹{sl_:,.2f}",   "cred"),
            (pc[2],"TARGET",   f"₹{tgt_:,.2f}",  "cgreen"),
            (pc[3],"UNREAL P&L",f"₹{pnl_:,.2f}", "cgreen" if pnl_>=0 else "cred"),
        ]:
            col.markdown(f'<div class="mcard"><div class="mlabel">{lbl}</div>'
                         f'<div class="mval {clr}">{val}</div></div>',unsafe_allow_html=True)

        if st.session_state.partial_closed:
            st.success("✅ ½ qty closed at 1:1 — SL trailed to CTC (breakeven)")
    else:
        for col,lbl in zip(m[1:],["Entry","SL","Target","P&L","Qty"]):
            col.metric(lbl,"—")

    if df_l is not None:
        sigs=st.session_state.live_signals
        fig=build_chart(df_l,[],
                        sigs if isinstance(sigs,pd.Series) else None,
                        tcfg["ind_cfg"],
                        f"LIVE · {st.session_state.live_ticker} · {tcfg['tf']}")
        st.plotly_chart(fig,use_container_width=True)


def _process_live(df, ltp, tcfg, dcfg):
    pos=st.session_state.live_position
    p=dict(sl_pts=tcfg["sl_pts"],tgt_pts=tcfg["tgt_pts"],
           rr=tcfg["rr"],sl_atr=tcfg["sl_atr"],tgt_atr=tcfg["tgt_atr"])
    now=datetime.now()

    if pos and now.hour==14 and now.minute>=30:
        pnl=(ltp-pos["entry"])*pos["qty"]
        _close(pos,ltp,"Force Exit 14:30",pnl,dcfg); return

    if pos:
        st.session_state.trailing_high=max(st.session_state.trailing_high,ltp)
        new_sl=trail_sl(pos["sl"],tcfg["sl_type"],ltp,pos["entry"],p,df,-1,
                        st.session_state.trailing_high)
        if new_sl>pos["sl"]:
            pos["sl"]=new_sl; st.session_state.live_position=pos
            st.session_state.live_log.append(f"[{now:%H:%M:%S}] SL trailed → ₹{new_sl:.2f}")

        t1=pos.get("t1",pos["entry"]+(pos["entry"]-pos["sl"]))
        if not st.session_state.partial_closed and ltp>=t1:
            st.session_state.partial_closed=True
            pos["sl"]=pos["entry"]; pos["qty"]=max(1,pos["qty"]//2)
            st.session_state.live_position=pos
            st.session_state.live_log.append(
                f"[{now:%H:%M:%S}] ½ closed @ ₹{t1:.2f}, SL→CTC ₹{pos['entry']:.2f}")
            log_order(execute_dhan("SELL",dcfg,ltp))

        if ltp<=pos["sl"]:
            pnl=(pos["sl"]-pos["entry"])*pos["qty"]
            _close(pos,pos["sl"],"SL Hit",pnl,dcfg); return

        if ltp>=pos["target"]:
            pnl=(pos["target"]-pos["entry"])*pos["qty"]
            _close(pos,pos["target"],"Target Hit",pnl,dcfg); return
        return

    if pos is None and now.hour<10:
        if tcfg["strategy"]=="ORB (Opening Range Breakout)":
            sigs=orb_signals(df,tcfg["ind_cfg"])
        elif tcfg["strategy"]=="Simple Buy":
            sigs=simple_buy_signals(df,tcfg["ind_cfg"])
        else:
            sigs=pd.Series("",index=df.index)

        st.session_state.live_signals=sigs
        latest=sigs.iloc[-1] if len(sigs) else ""
        if latest=="BUY":
            sl_ =compute_sl(ltp,tcfg["sl_type"],p,df,-1)
            tgt_=compute_tgt(ltp,sl_,tcfg["tgt_type"],p,df,-1)
            t1  =ltp+(ltp-sl_)
            st.session_state.live_position=dict(
                entry=ltp,sl=sl_,target=tgt_,t1=t1,
                qty=tcfg["qty"],time=str(now)[:19])
            st.session_state.trailing_high=ltp
            st.session_state.live_log.append(
                f"[{now:%H:%M:%S}] 🟢 BUY @ ₹{ltp:.2f}  SL=₹{sl_:.2f}  TGT=₹{tgt_:.2f}")
            log_order(execute_dhan("BUY",dcfg,ltp))


def _close(pos, xp, reason, pnl, dcfg):
    st.session_state.trade_history.append(dict(
        ticker=st.session_state.live_ticker,
        entry_time=pos["time"],exit_time=str(datetime.now())[:19],
        entry_price=round(pos["entry"],4),exit_price=round(xp,4),
        qty=pos["qty"],sl=round(pos["sl"],4),target=round(pos["target"],4),
        pnl=round(pnl,2),reason=reason,status="WIN" if pnl>=0 else "LOSS"))
    st.session_state.live_log.append(
        f"[{datetime.now():%H:%M:%S}] "
        f"{'✅' if pnl>=0 else '❌'} {reason} @ ₹{xp:.2f}  PnL=₹{pnl:+.2f}")
    log_order(execute_dhan("SELL",dcfg,xp))
    st.session_state.live_position=None
    st.session_state.partial_closed=False


def _sq_off(tcfg, dcfg):
    pos=st.session_state.live_position
    if not pos: return
    ltp=DF.get_ltp(st.session_state.live_ticker) or pos["entry"]
    pnl=(ltp-pos["entry"])*pos["qty"]
    _close(pos,ltp,"Manual Square Off",pnl,dcfg)

# ──────────────────────────────────────────────────────────────
# TRADE HISTORY TAB
# ──────────────────────────────────────────────────────────────
def tab_history():
    st.markdown("""
    <div style='padding:2px 0 12px'>
      <span style='font-family:JetBrains Mono;font-size:15px;font-weight:700;color:#00d4aa'>
        ◈ TRADE HISTORY</span>
      <span style='font-size:10px;color:#475569;font-family:JetBrains Mono;margin-left:10px'>
        Live session · In-memory · No database</span>
    </div>""",unsafe_allow_html=True)

    trades=st.session_state.trade_history
    c,_=st.columns([1,6])
    if c.button("🗑 Clear",key="hclr"):
        st.session_state.trade_history=[]; st.rerun()

    if not trades:
        st.markdown('<div class="ibox">No live trades yet. '
                    'Start live trading to populate history.</div>',unsafe_allow_html=True)
        return

    df_h=pd.DataFrame(trades)
    wins=int((df_h["pnl"]>=0).sum()); total=len(df_h)
    losses=total-wins; wr=round(wins/total*100,1); tp=df_h["pnl"].sum()

    m=st.columns(5)
    mc(m[0],"Trades",  total)
    mc(m[1],"Wins",    wins, "cgreen")
    mc(m[2],"Losses",  losses,"cred")
    mc(m[3],"Win Rate",f"{wr}%","cgreen" if wr>=50 else "cred")
    mc(m[4],"Total P&L",f"₹{tp:,.2f}","cgreen" if tp>=0 else "cred")

    cum=[0.0]
    for t in trades: cum.append(cum[-1]+t["pnl"])
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=cum,mode="lines+markers",
        line=dict(color="#00d4aa",width=2),
        fill="tozeroy",fillcolor=rgba("00d4aa",.07)))
    fig.add_hline(y=0,line=dict(color="#94a3b8",width=1,dash="dot"))
    fig.update_layout(
        title=dict(text="Cumulative P&L",
                   font=dict(family="JetBrains Mono",size=12,color="#e2e8f0")),
        paper_bgcolor="#0a0e17",plot_bgcolor="#0a0e17",
        font=dict(color="#94a3b8",family="JetBrains Mono",size=10),
        margin=dict(l=40,r=10,t=35,b=15),height=220)
    fig.update_xaxes(showgrid=True,gridcolor="#1e2d45")
    fig.update_yaxes(showgrid=True,gridcolor="#1e2d45",tickprefix="₹")
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="sh">📋 ALL TRADES</div>',unsafe_allow_html=True)
    for t in reversed(trades):
        brd="#10b981" if t["status"]=="WIN" else "#ef4444"
        bc ="bwin"    if t["status"]=="WIN" else "bloss"
        st.markdown(
            f'<div class="trow" style="border-left:3px solid {brd}">'
            f'<span class="sbadge {bc}">{t["status"]}</span>'
            f'<span style="margin-left:10px;color:#94a3b8">{t.get("ticker","")}</span>'
            f'<span style="margin-left:10px;color:#e2e8f0;font-weight:700">'
            f'₹{t["pnl"]:+,.2f}</span>'
            f'<span style="margin-left:12px;color:#64748b">'
            f'Entry ₹{t["entry_price"]:,.2f} → Exit ₹{t["exit_price"]:,.2f}'
            f' · Qty {t.get("qty",1)} · {t.get("reason","")}</span>'
            f'<span style="float:right;color:#334155;font-size:10px">'
            f'{t.get("exit_time","")}</span></div>',unsafe_allow_html=True)

    st.download_button("⬇ Download CSV",df_h.to_csv(index=False),
                       f"history_{datetime.now():%Y%m%d_%H%M}.csv","text/csv",key="hdl")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    tcfg, dcfg = render_sidebar()

    st.markdown("""
    <div style='padding:6px 0 4px'>
      <span style='font-family:JetBrains Mono;font-size:19px;font-weight:700;
                   background:linear-gradient(90deg,#00d4aa,#3b82f6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        ⚡ PROFESSIONAL ALGORITHMIC TRADING PLATFORM
      </span>
    </div>""",unsafe_allow_html=True)

    warns=[]
    if not PYDHAN_OK:   warns.append("`pip install pydhan` (equity orders)")
    if not DHANHQ_OK:   warns.append("`pip install dhanhq` (options orders)")
    if warns:
        st.warning("⚠️  Optional broker libs missing: " + "  •  ".join(warns) +
                   "   — All **indicators calculated built-in**, no ta/talib needed.")

    t1,t2,t3 = st.tabs(["📊  BACKTESTING","⚡  LIVE TRADING","📋  TRADE HISTORY"])
    with t1: tab_bt(tcfg,dcfg)
    with t2: tab_live(tcfg,dcfg)
    with t3: tab_history()

if __name__=="__main__":
    main()
