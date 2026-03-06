
# =============================================================================
# NIFTY / BANKNIFTY / MULTI-ASSET ALGO TRADING PLATFORM
# Single-file Streamlit Application | All indicators manual (no pandas-ta/talib)
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoTrader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL THEME  –  white background, black font only
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"], .stApp {
      background-color: #ffffff !important;
      color: #000000 !important;
      font-family: 'Segoe UI', sans-serif;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: #f5f5f5 !important;
      color: #000000 !important;
      border-radius: 6px 6px 0 0;
      font-weight: 600;
      padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
      background-color: #000000 !important;
      color: #ffffff !important;
  }
  .metric-card {
      background: #f9f9f9;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 12px 16px;
      text-align: center;
  }
  .metric-card .label { font-size: 12px; color: #555; margin-bottom: 4px; }
  .metric-card .value { font-size: 22px; font-weight: 700; color: #000; }
  .win  { color: #006400 !important; }
  .loss { color: #8b0000 !important; }
  .section-header {
      font-size: 18px; font-weight: 700;
      border-bottom: 2px solid #000;
      padding-bottom: 6px; margin-bottom: 14px;
  }
  div.stButton > button {
      background: #000000; color: #ffffff;
      border-radius: 6px; border: none;
      padding: 8px 22px; font-weight: 600;
      transition: background 0.2s;
  }
  div.stButton > button:hover { background: #333333; }
  .stSelectbox label, .stMultiSelect label,
  .stSlider label, .stNumberInput label,
  .stCheckbox label, .stRadio label { color: #000 !important; font-weight: 500; }
  .stDataFrame { border: 1px solid #ddd; border-radius: 6px; }
  .live-badge {
      display: inline-block;
      background: #000; color: #fff;
      border-radius: 12px; padding: 2px 12px;
      font-size: 12px; font-weight: 700;
      animation: blink 1.2s step-start infinite;
  }
  @keyframes blink { 50% { opacity: 0.4; } }
  .signal-long  { background:#e6ffe6; border-left:4px solid #006400; padding:8px 14px; border-radius:4px; }
  .signal-short { background:#ffe6e6; border-left:4px solid #8b0000; padding:8px 14px; border-radius:4px; }
  .signal-none  { background:#f5f5f5; border-left:4px solid #999;    padding:8px 14px; border-radius:4px; }
  .stSidebar { background-color: #fafafa !important; }
  .stSidebar * { color: #000 !important; }
  h1, h2, h3, h4, h5, h6 { color: #000 !important; }
  p, span, div, label { color: #000 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

TICKERS = {
    "Nifty 50"         : "^NSEI",
    "Bank Nifty"       : "^NSEBANK",
    "Sensex"           : "^BSESN",
    "Bitcoin (BTC)"    : "BTC-USD",
    "Ethereum (ETH)"   : "ETH-USD",
    "USD/INR"          : "USDINR=X",
    "Gold Futures"     : "GC=F",
    "Silver Futures"   : "SI=F",
    "Crude Oil"        : "CL=F",
    "Natural Gas"      : "NG=F",
    "EUR/USD"          : "EURUSD=X",
    "GBP/USD"          : "GBPUSD=X",
    "USD/JPY"          : "USDJPY=X",
    "Copper"           : "HG=F",
    "Platinum"         : "PL=F",
    "Custom Ticker"    : "__CUSTOM__",
}

TIMEFRAMES = ["1m","5m","15m","30m","1h","4h","1d","1wk"]
TF_PERIOD_MAP = {
    "1m":  ["1d","5d","7d"],
    "5m":  ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo","3mo"],
    "30m": ["1d","5d","7d","1mo","3mo","6mo"],
    "1h":  ["5d","7d","1mo","3mo","6mo","1y","2y"],
    "4h":  ["1mo","3mo","6mo","1y","2y","5y"],
    "1d":  ["1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk": ["3mo","6mo","1y","2y","5y","10y","20y"],
}
ALL_PERIODS = ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]

STRATEGIES = [
    "1.  EMA Crossover (9 / 21)",
    "2.  EMA Crossover (20 / 50)",
    "3.  Triple EMA (9 / 21 / 50)",
    "4.  MACD Signal Line Cross",
    "5.  RSI Oversold / Overbought",
    "6.  Bollinger Band Breakout",
    "7.  Bollinger Band Mean Reversion",
    "8.  Supertrend",
    "9.  ADX + DI Crossover",
    "10. Stochastic Crossover",
    "11. Stochastic + RSI Combined",
    "12. Ichimoku Cloud",
    "13. VWAP Cross",
    "14. Donchian Channel Breakout",
    "15. Keltner Channel Breakout",
    "16. CCI Strategy",
    "17. Williams %R Strategy",
    "18. Pivot Point Bounce",
    "19. Heikin Ashi + EMA",
    "20. RSI + MACD Combined",
    "21. EMA + RSI Filter",
    "22. Opening Range Breakout",
    "23. Inside Bar Breakout",
    "24. MACD Histogram Reversal",
    "25. Supertrend + RSI",
    "26. Volume Price Trend",
    "27. Swing High / Low Breakout",
    "28. EMA Ribbon",
    "29. Parabolic SAR",
    "30. Dual Momentum",
]

SL_TYPES = [
    "Custom Points",
    "Trailing SL (Fixed Points)",
    "Previous Candle Low/High",
    "Current Candle Low/High",
    "Prev/Curr Swing Low/High",
    "ATR Based",
    "Risk/Reward Based",
    "EMA Reverse Crossover",
    "Volatility Based",
    "Cost to Cost (Breakeven)",
    "Strategy Based",
    "Auto (AI Managed)",
]

TARGET_TYPES = [
    "Custom Points",
    "Trailing Target (Fixed Points)",
    "Previous Candle High/Low",
    "Current Candle High/Low",
    "Prev/Curr Swing High/Low",
    "ATR Based",
    "Risk/Reward Based",
    "EMA Reverse Crossover",
    "Partial Exits (Multi-Level)",
    "Strategy Based",
    "Auto (AI Managed)",
]

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT  – call once per run
# ─────────────────────────────────────────────────────────────
def init_session():
    keys = {
        "bt_results"       : None,
        "bt_trades"        : [],
        "bt_metrics"       : {},
        "live_trades"      : [],
        "live_open_pos"    : None,
        "live_running"     : False,
        "live_last_signal" : None,
        "live_capital"     : 100_000.0,
        "trade_history"    : [],
        "opt_results"      : None,
        "data_cache"       : {},
        "last_fetch"       : 0.0,
        "live_chart_data"  : pd.DataFrame(),
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────────────────────
# RATE-LIMITED DATA FETCHER
# ─────────────────────────────────────────────────────────────
def fetch_data(ticker: str, period: str, interval: str, force_fresh=False) -> pd.DataFrame | None:
    cache_key = f"{ticker}_{period}_{interval}"
    if not force_fresh and cache_key in st.session_state.data_cache:
        cached_ts, cached_df = st.session_state.data_cache[cache_key]
        if time.time() - cached_ts < 60:          # cache for 60 s
            return cached_df

    elapsed = time.time() - st.session_state.last_fetch
    if elapsed < 1.5:
        time.sleep(1.5 - elapsed)

    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        st.session_state.last_fetch = time.time()

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("Asia/Kolkata")

        for col in ["Open","High","Low","Close","Volume"]:
            if col not in df.columns:
                df[col] = 0.0
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df = df[df["Volume"] >= 0]

        st.session_state.data_cache[cache_key] = (time.time(), df)
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS  (pure numpy / pandas — no external TA)
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(com=n - 1, adjust=False).mean()
    al = l.ewm(com=n - 1, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(s: pd.Series, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow)
    si = _ema(m, sig)
    return m, si, m - si

def _bb(s: pd.Series, n=20, k=2):
    m = _sma(s, n)
    std = s.rolling(n).std()
    return m + k * std, m, m - k * std

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n=14) -> pd.Series:
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()

def _stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return k, _sma(k, dp)

def _adx(h, l, c, n=14):
    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    pdm[pdm < ndm] = 0
    ndm[ndm < pdm] = 0
    atr_ = _atr(h, l, c, n)
    pdi = 100 * _ema(pdm, n) / atr_.replace(0, np.nan)
    ndi = 100 * _ema(ndm, n) / atr_.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return _ema(dx, n), pdi, ndi

def _cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    ma = _sma(tp, n)
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def _wpr(h, l, c, n=14):
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)

def _supertrend(h, l, c, n=10, mult=3.0):
    atr_ = _atr(h, l, c, n)
    hl2 = (h + l) / 2
    ub = hl2 + mult * atr_
    lb = hl2 - mult * atr_

    ub_f = ub.copy(); lb_f = lb.copy()
    direction = pd.Series(1, index=c.index)
    st_line = lb.copy()

    for i in range(1, len(c)):
        prev_ub = ub_f.iloc[i-1]; prev_lb = lb_f.iloc[i-1]
        cur_ub  = ub.iloc[i];     cur_lb  = lb.iloc[i]
        prev_c  = c.iloc[i-1];    cur_c   = c.iloc[i]

        lb_f.iloc[i] = cur_lb if cur_lb > prev_lb or prev_c < prev_lb else prev_lb
        ub_f.iloc[i] = cur_ub if cur_ub < prev_ub or prev_c > prev_ub else prev_ub

        if direction.iloc[i-1] == 1:
            if cur_c < lb_f.iloc[i]:
                direction.iloc[i] = -1; st_line.iloc[i] = ub_f.iloc[i]
            else:
                direction.iloc[i] = 1;  st_line.iloc[i] = lb_f.iloc[i]
        else:
            if cur_c > ub_f.iloc[i]:
                direction.iloc[i] = 1;  st_line.iloc[i] = lb_f.iloc[i]
            else:
                direction.iloc[i] = -1; st_line.iloc[i] = ub_f.iloc[i]

    return st_line, direction

def _vwap(h, l, c, v):
    tp = (h + l + c) / 3
    cv = (tp * v).cumsum()
    sv = v.cumsum()
    return cv / sv.replace(0, np.nan)

def _donchian(h, l, n=20):
    return h.rolling(n).max(), l.rolling(n).min()

def _keltner(h, l, c, en=20, an=10, mult=1.5):
    mid = _ema(c, en)
    return mid + mult * _atr(h, l, c, an), mid, mid - mult * _atr(h, l, c, an)

def _pivot(h, l, c):
    p = (h + l + c) / 3
    r1 = 2*p - l; s1 = 2*p - h
    r2 = p + (h-l); s2 = p - (h-l)
    r3 = h + 2*(p-l); s3 = l - 2*(h-p)
    return p, r1, s1, r2, s2, r3, s3

def _heikin_ashi(o, h, l, c):
    ha_c = (o + h + l + c) / 4
    ha_o = ha_c.copy()
    ha_o.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(o)):
        ha_o.iloc[i] = (ha_o.iloc[i-1] + ha_c.iloc[i-1]) / 2
    ha_h = pd.concat([h, ha_o, ha_c], axis=1).max(axis=1)
    ha_l = pd.concat([l, ha_o, ha_c], axis=1).min(axis=1)
    return ha_o, ha_h, ha_l, ha_c

def _psar(h, l, c, af0=0.02, af_step=0.02, af_max=0.2):
    n = len(c)
    sar = np.full(n, np.nan); ep = np.full(n, np.nan)
    af = np.full(n, af0); trend = np.ones(n, dtype=int)
    sar[0] = l.iloc[0]; ep[0] = h.iloc[0]
    for i in range(1, n):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af[i-1]*(ep[i-1]-sar[i-1])
            sar[i] = min(sar[i], l.iloc[i-1], l.iloc[max(0,i-2)])
            if l.iloc[i] < sar[i]:
                trend[i]=-1; sar[i]=ep[i-1]; ep[i]=l.iloc[i]; af[i]=af0
            else:
                trend[i]=1
                if h.iloc[i] > ep[i-1]:
                    ep[i]=h.iloc[i]; af[i]=min(af[i-1]+af_step,af_max)
                else:
                    ep[i]=ep[i-1]; af[i]=af[i-1]
        else:
            sar[i] = sar[i-1] + af[i-1]*(ep[i-1]-sar[i-1])
            sar[i] = max(sar[i], h.iloc[i-1], h.iloc[max(0,i-2)])
            if h.iloc[i] > sar[i]:
                trend[i]=1; sar[i]=ep[i-1]; ep[i]=h.iloc[i]; af[i]=af0
            else:
                trend[i]=-1
                if l.iloc[i] < ep[i-1]:
                    ep[i]=l.iloc[i]; af[i]=min(af[i-1]+af_step,af_max)
                else:
                    ep[i]=ep[i-1]; af[i]=af[i-1]
    return (pd.Series(sar, index=c.index),
            pd.Series(trend, index=c.index))

def _swing_hl(h, l, lb=5):
    sh = pd.Series(np.nan, index=h.index)
    sl_ = pd.Series(np.nan, index=l.index)
    for i in range(lb, len(h)-lb):
        if h.iloc[i] == h.iloc[i-lb:i+lb+1].max():
            sh.iloc[i] = h.iloc[i]
        if l.iloc[i] == l.iloc[i-lb:i+lb+1].min():
            sl_.iloc[i] = l.iloc[i]
    return sh, sl_

# ─────────────────────────────────────────────────────────────
# 30 STRATEGY SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────
def compute_signals(df: pd.DataFrame, strategy: str, p: dict = None) -> pd.DataFrame:
    """Returns df copy with 'signal' column: 1=Long, -1=Short, 0=Flat"""
    if p is None:
        p = {}
    df = df.copy()
    df["signal"] = 0
    o = df["Open"]; h = df["High"]; l = df["Low"]
    c = df["Close"]; v = df["Volume"]
    n = len(c)
    if n < 60:
        return df

    try:
        num = int(strategy.split(".")[0].strip())
    except:
        return df

    try:
        if num == 1:
            f = _ema(c, p.get("fast",9)); s = _ema(c, p.get("slow",21))
            df["signal"] = np.where((f>s)&(f.shift()<=s.shift()),1,
                           np.where((f<s)&(f.shift()>=s.shift()),-1,0))

        elif num == 2:
            f = _ema(c, p.get("fast",20)); s = _ema(c, p.get("slow",50))
            df["signal"] = np.where((f>s)&(f.shift()<=s.shift()),1,
                           np.where((f<s)&(f.shift()>=s.shift()),-1,0))

        elif num == 3:
            e1=_ema(c,p.get("e1",9)); e2=_ema(c,p.get("e2",21)); e3=_ema(c,p.get("e3",50))
            df["signal"] = np.where((e1>e2)&(e2>e3)&((e1.shift()<=e2.shift())|(e2.shift()<=e3.shift())),1,
                           np.where((e1<e2)&(e2<e3)&((e1.shift()>=e2.shift())|(e2.shift()>=e3.shift())),-1,0))

        elif num == 4:
            ml,sl_,hi = _macd(c,p.get("fast",12),p.get("slow",26),p.get("sig",9))
            df["signal"] = np.where((ml>sl_)&(ml.shift()<=sl_.shift()),1,
                           np.where((ml<sl_)&(ml.shift()>=sl_.shift()),-1,0))

        elif num == 5:
            r = _rsi(c, p.get("period",14))
            ob=p.get("overbought",70); os_=p.get("oversold",30)
            df["signal"] = np.where((r<os_)&(r.shift()>=os_),1,
                           np.where((r>ob)&(r.shift()<=ob),-1,0))

        elif num == 6:
            ub,mb,lb_ = _bb(c,p.get("period",20),p.get("std",2))
            df["signal"] = np.where((c>ub)&(c.shift()<=ub.shift()),1,
                           np.where((c<lb_)&(c.shift()>=lb_.shift()),-1,0))

        elif num == 7:
            ub,mb,lb_ = _bb(c,p.get("period",20),p.get("std",2))
            df["signal"] = np.where((c<lb_)&(c.shift()>=lb_.shift()),1,
                           np.where((c>ub)&(c.shift()<=ub.shift()),-1,0))

        elif num == 8:
            st_,dir_ = _supertrend(h,l,c,p.get("period",10),p.get("mult",3.0))
            df["signal"] = np.where((dir_==1)&(dir_.shift()==-1),1,
                           np.where((dir_==-1)&(dir_.shift()==1),-1,0))

        elif num == 9:
            adx_,pdi,ndi = _adx(h,l,c,p.get("period",14))
            thresh = p.get("thresh",25)
            df["signal"] = np.where((pdi>ndi)&(pdi.shift()<=ndi.shift())&(adx_>thresh),1,
                           np.where((ndi>pdi)&(ndi.shift()<=pdi.shift())&(adx_>thresh),-1,0))

        elif num == 10:
            k,d = _stoch(h,l,c,p.get("k",14),p.get("d",3))
            df["signal"] = np.where((k>d)&(k.shift()<=d.shift())&(k<80),1,
                           np.where((k<d)&(k.shift()>=d.shift())&(k>20),-1,0))

        elif num == 11:
            k,d = _stoch(h,l,c,p.get("k",14),p.get("d",3))
            r = _rsi(c,p.get("rsi",14))
            df["signal"] = np.where((k>d)&(k.shift()<=d.shift())&(r<55),1,
                           np.where((k<d)&(k.shift()>=d.shift())&(r>45),-1,0))

        elif num == 12:
            ten = (h.rolling(9).max()+l.rolling(9).min())/2
            kij = (h.rolling(26).max()+l.rolling(26).min())/2
            sa  = ((ten+kij)/2).shift(26)
            sb  = ((h.rolling(52).max()+l.rolling(52).min())/2).shift(26)
            df["signal"] = np.where(
                (c>sa)&(c>sb)&(ten>kij)&(ten.shift()<=kij.shift()),1,
                np.where((c<sa)&(c<sb)&(ten<kij)&(ten.shift()>=kij.shift()),-1,0))

        elif num == 13:
            vw = _vwap(h,l,c,v)
            df["signal"] = np.where((c>vw)&(c.shift()<=vw.shift()),1,
                           np.where((c<vw)&(c.shift()>=vw.shift()),-1,0))

        elif num == 14:
            dc_h,dc_l = _donchian(h,l,p.get("period",20))
            df["signal"] = np.where(c>dc_h.shift(1),1,
                           np.where(c<dc_l.shift(1),-1,0))

        elif num == 15:
            ku,km,kl = _keltner(h,l,c)
            df["signal"] = np.where((c>ku)&(c.shift()<=ku.shift()),1,
                           np.where((c<kl)&(c.shift()>=kl.shift()),-1,0))

        elif num == 16:
            cc = _cci(h,l,c,p.get("period",20))
            df["signal"] = np.where((cc>-100)&(cc.shift()<=-100),1,
                           np.where((cc<100)&(cc.shift()>=100),-1,0))

        elif num == 17:
            wr = _wpr(h,l,c,p.get("period",14))
            df["signal"] = np.where((wr>-80)&(wr.shift()<=-80),1,
                           np.where((wr<-20)&(wr.shift()>=-20),-1,0))

        elif num == 18:
            pv,r1,s1,_,_,_,_ = _pivot(h.shift(),l.shift(),c.shift())
            df["signal"] = np.where((c>pv)&(c<r1)&(c.shift()<=pv.shift()),1,
                           np.where((c<pv)&(c>s1)&(c.shift()>=pv.shift()),-1,0))

        elif num == 19:
            hao,hah,hal,hac = _heikin_ashi(o,h,l,c)
            he = _ema(hac, p.get("ema",21))
            green = hac > hao; red = hac < hao
            df["signal"] = np.where(green&(hac>he)&~green.shift().fillna(False),1,
                           np.where(red&(hac<he)&~red.shift().fillna(False),-1,0))

        elif num == 20:
            r = _rsi(c,p.get("rsi",14))
            ml,sl_,_ = _macd(c,12,26,9)
            df["signal"] = np.where((r<55)&(ml>sl_)&(ml.shift()<=sl_.shift()),1,
                           np.where((r>45)&(ml<sl_)&(ml.shift()>=sl_.shift()),-1,0))

        elif num == 21:
            f=_ema(c,p.get("fast",9)); s=_ema(c,p.get("slow",21))
            r=_rsi(c,p.get("rsi",14))
            df["signal"] = np.where((f>s)&(f.shift()<=s.shift())&(r>40)&(r<70),1,
                           np.where((f<s)&(f.shift()>=s.shift())&(r<60)&(r>30),-1,0))

        elif num == 22:
            dates = df.index.date
            df["_date"] = dates
            first_h = df.groupby("_date")["High"].transform("first")
            first_l = df.groupby("_date")["Low"].transform("first")
            df["signal"] = np.where((c>first_h)&(c.shift()<=first_h.shift()),1,
                           np.where((c<first_l)&(c.shift()>=first_l.shift()),-1,0))
            df.drop("_date",axis=1,inplace=True)

        elif num == 23:
            inside = (h<h.shift())&(l>l.shift())
            df["signal"] = np.where((h>h.shift())&inside.shift().fillna(False),1,
                           np.where((l<l.shift())&inside.shift().fillna(False),-1,0))

        elif num == 24:
            _,_,hist = _macd(c,12,26,9)
            df["signal"] = np.where((hist>0)&(hist.shift()<=0),1,
                           np.where((hist<0)&(hist.shift()>=0),-1,0))

        elif num == 25:
            _,dir_ = _supertrend(h,l,c,p.get("period",10),p.get("mult",3.0))
            r = _rsi(c,p.get("rsi",14))
            df["signal"] = np.where((dir_==1)&(dir_.shift()==-1)&(r>40),1,
                           np.where((dir_==-1)&(dir_.shift()==1)&(r<60),-1,0))

        elif num == 26:
            vpt = (c.pct_change()*v).cumsum()
            ve = _ema(vpt,p.get("period",14))
            df["signal"] = np.where((vpt>ve)&(vpt.shift()<=ve.shift()),1,
                           np.where((vpt<ve)&(vpt.shift()>=ve.shift()),-1,0))

        elif num == 27:
            lb = p.get("lookback",10)
            rh = h.rolling(lb).max(); rl = l.rolling(lb).min()
            df["signal"] = np.where(c>rh.shift(1),1,
                           np.where(c<rl.shift(1),-1,0))

        elif num == 28:
            ps_ = [5,8,13,21,34,55]
            ems = [_ema(c,x) for x in ps_]
            bull = pd.Series(False,index=c.index)
            bear = pd.Series(False,index=c.index)
            for i in range(1,n):
                is_b = all(ems[j].iloc[i]>ems[j+1].iloc[i] for j in range(len(ems)-1))
                pr_b = all(ems[j].iloc[i-1]>ems[j+1].iloc[i-1] for j in range(len(ems)-1))
                is_e = all(ems[j].iloc[i]<ems[j+1].iloc[i] for j in range(len(ems)-1))
                pr_e = all(ems[j].iloc[i-1]<ems[j+1].iloc[i-1] for j in range(len(ems)-1))
                bull.iloc[i] = is_b and not pr_b
                bear.iloc[i] = is_e and not pr_e
            df["signal"] = np.where(bull,1,np.where(bear,-1,0))

        elif num == 29:
            sar,tr = _psar(h,l,c,p.get("af0",0.02),p.get("afstep",0.02),p.get("afmax",0.2))
            df["signal"] = np.where((tr==1)&(tr.shift()==-1),1,
                           np.where((tr==-1)&(tr.shift()==1),-1,0))

        elif num == 30:
            m1 = c.pct_change(p.get("p1",10))
            m2 = c.pct_change(p.get("p2",30))
            df["signal"] = np.where(
                (m1>0)&(m2>0)&((m1.shift()<=0)|(m2.shift()<=0)),1,
                np.where((m1<0)&(m2<0)&((m1.shift()>=0)|(m2.shift()>=0)),-1,0))

    except Exception as e:
        st.warning(f"Signal error [{strategy}]: {e}")

    return df

# ─────────────────────────────────────────────────────────────
# SL / TARGET CALCULATION
# ─────────────────────────────────────────────────────────────
def calc_sl(df, idx, entry, direction, sl_type, sp, atr_v=None):
    d = direction   # 1=long, -1=short
    if atr_v is None or np.isnan(atr_v):
        atr_v = entry * 0.01

    if sl_type == "Custom Points":
        return entry - d * sp.get("sl_pts", 20)
    elif sl_type == "Trailing SL (Fixed Points)":
        return entry - d * sp.get("sl_pts", 20)
    elif sl_type == "Previous Candle Low/High":
        return df["Low"].iloc[idx-1] if d==1 else df["High"].iloc[idx-1]
    elif sl_type == "Current Candle Low/High":
        return df["Low"].iloc[idx] if d==1 else df["High"].iloc[idx]
    elif sl_type == "Prev/Curr Swing Low/High":
        lb = sp.get("swing_lb",5); st=max(0,idx-lb*4)
        sub = df.iloc[st:idx+1]
        return sub["Low"].min() if d==1 else sub["High"].max()
    elif sl_type == "ATR Based":
        return entry - d * sp.get("atr_sl_mult",2.0) * atr_v
    elif sl_type == "Risk/Reward Based":
        tgt_pts = sp.get("rr_target_pts",40); rr = sp.get("rr_ratio",2.0)
        return entry - d * (tgt_pts/rr)
    elif sl_type == "EMA Reverse Crossover":
        return entry - d * sp.get("sl_pts",20)   # fallback; checked live
    elif sl_type == "Volatility Based":
        return entry - d * 1.5 * atr_v
    elif sl_type == "Cost to Cost (Breakeven)":
        return entry - d * atr_v  # initial; moves to breakeven
    elif sl_type == "Strategy Based":
        return entry - d * 2.0 * atr_v
    elif sl_type == "Auto (AI Managed)":
        # Dynamic: 1.5x ATR, never exceed 2% of price
        raw = 1.5 * atr_v
        cap = entry * 0.02
        return entry - d * min(raw, cap)
    return entry - d * atr_v

def calc_target(df, idx, entry, direction, tgt_type, tp, atr_v=None):
    d = direction
    if atr_v is None or np.isnan(atr_v):
        atr_v = entry * 0.01

    if tgt_type == "Custom Points":
        return entry + d * tp.get("tgt_pts",40)
    elif tgt_type == "Trailing Target (Fixed Points)":
        return entry + d * tp.get("tgt_pts",40)
    elif tgt_type == "Previous Candle High/Low":
        return df["High"].iloc[idx-1] if d==1 else df["Low"].iloc[idx-1]
    elif tgt_type == "Current Candle High/Low":
        return df["High"].iloc[idx] if d==1 else df["Low"].iloc[idx]
    elif tgt_type == "Prev/Curr Swing High/Low":
        lb=tp.get("swing_lb",5); st=max(0,idx-lb*4)
        sub=df.iloc[st:idx+1]
        return sub["High"].max() if d==1 else sub["Low"].min()
    elif tgt_type == "ATR Based":
        return entry + d * tp.get("atr_tgt_mult",3.0) * atr_v
    elif tgt_type == "Risk/Reward Based":
        sl_pts=tp.get("rr_sl_pts",20); rr=tp.get("rr_ratio",2.0)
        return entry + d * sl_pts * rr
    elif tgt_type == "EMA Reverse Crossover":
        return entry + d * tp.get("tgt_pts",40)   # fallback; checked live
    elif tgt_type == "Partial Exits (Multi-Level)":
        # Return first level; partial exit logic in backtest engine
        return entry + d * tp.get("tgt_pts",40)
    elif tgt_type == "Strategy Based":
        return entry + d * 3.0 * atr_v
    elif tgt_type == "Auto (AI Managed)":
        raw = 3.0 * atr_v
        return entry + d * raw
    return entry + d * atr_v

# ─────────────────────────────────────────────────────────────
# UPDATE TRAILING SL EACH BAR
# ─────────────────────────────────────────────────────────────
def update_trail_sl(pos, df, i, sl_type, sp, atr_v):
    d   = pos["direction"]
    cur = df["Close"].iloc[i]
    sl  = pos["sl"]

    if sl_type == "Trailing SL (Fixed Points)":
        pts = sp.get("sl_pts",20)
        new_sl = cur - d*pts
        pos["sl"] = max(sl,new_sl) if d==1 else min(sl,new_sl)

    elif sl_type == "Previous Candle Low/High":
        new_sl = df["Low"].iloc[i-1] if d==1 else df["High"].iloc[i-1]
        pos["sl"] = max(sl,new_sl) if d==1 else min(sl,new_sl)

    elif sl_type == "ATR Based":
        mult = sp.get("atr_sl_mult",2.0)
        new_sl = cur - d*mult*atr_v
        pos["sl"] = max(sl,new_sl) if d==1 else min(sl,new_sl)

    elif sl_type == "Volatility Based":
        new_sl = cur - d*1.5*atr_v
        pos["sl"] = max(sl,new_sl) if d==1 else min(sl,new_sl)

    elif sl_type == "Cost to Cost (Breakeven)":
        if atr_v:
            profit = (cur - pos["entry_price"]) * d
            if profit >= atr_v:
                be = pos["entry_price"]
                pos["sl"] = max(sl,be) if d==1 else min(sl,be)

    elif sl_type == "Auto (AI Managed)":
        raw = 1.5*atr_v; cap = cur*0.02
        new_sl = cur - d*min(raw,cap)
        pos["sl"] = max(sl,new_sl) if d==1 else min(sl,new_sl)

    return pos

# ─────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
def run_backtest(df_raw, strategy, sl_type, tgt_type, sp, tp,
                 allow_overlap=False, use_tf=False, tf_start=None, tf_end=None,
                 initial_capital=100_000.0, qty=1):

    df = compute_signals(df_raw, strategy)

    if use_tf and tf_start and tf_end:
        mask = (df.index.time >= tf_start) & (df.index.time <= tf_end)
        df = df[mask].copy()

    if len(df) < 10:
        return []

    atr_s   = _atr(df["High"],df["Low"],df["Close"],14)
    ema_f   = _ema(df["Close"], sp.get("ema_fast",9))
    ema_sl  = _ema(df["Close"], sp.get("ema_slow",21))

    trades  = []
    capital = initial_capital
    open_positions = []   # list of pos dicts (supports overlap)

    for i in range(1, len(df)):
        row     = df.iloc[i]
        cur_c   = row["Close"]
        cur_h   = row["High"]
        cur_l   = row["Low"]
        cur_t   = df.index[i]
        atr_v   = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else cur_c*0.01
        ema_f_v = float(ema_f.iloc[i])
        ema_s_v = float(ema_sl.iloc[i])
        ema_f_p = float(ema_f.iloc[i-1])
        ema_s_p = float(ema_sl.iloc[i-1])

        # ── Manage open positions ──────────────────────────────
        still_open = []
        for pos in open_positions:
            d = pos["direction"]

            # Update trailing SL
            pos = update_trail_sl(pos, df, i, sl_type, sp, atr_v)

            # EMA crossover exit
            ema_exit = False
            if sl_type == "EMA Reverse Crossover" or tgt_type == "EMA Reverse Crossover":
                if d==1 and ema_f_v < ema_s_v and ema_f_p >= ema_s_p:
                    ema_exit=True
                elif d==-1 and ema_f_v > ema_s_v and ema_f_p <= ema_s_p:
                    ema_exit=True

            # Partial exits
            if tgt_type == "Partial Exits (Multi-Level)":
                lv1 = pos["entry_price"] + d * tp.get("tgt_pts",40)
                lv2 = pos["entry_price"] + d * tp.get("tgt_pts",40) * 1.5
                if not pos.get("lv1_done") and ((d==1 and cur_h>=lv1)or(d==-1 and cur_l<=lv1)):
                    pnl_partial = (lv1 - pos["entry_price"]) * d * int(qty*0.5)
                    capital += pnl_partial
                    pos["lv1_done"] = True
                    pos["qty_remaining"] = int(qty*0.5)
                elif not pos.get("lv2_done") and pos.get("lv1_done") and \
                     ((d==1 and cur_h>=lv2)or(d==-1 and cur_l<=lv2)):
                    rem = pos.get("qty_remaining", qty)
                    pnl_partial = (lv2 - pos["entry_price"]) * d * rem
                    capital += pnl_partial
                    pos["lv2_done"] = True
                    still_open.append(pos)
                    continue

            # Check SL
            sl_hit = (d==1 and cur_l <= pos["sl"]) or (d==-1 and cur_h >= pos["sl"])
            # Check Target
            tgt_hit = (d==1 and cur_h >= pos["target"]) or (d==-1 and cur_l <= pos["target"])
            # Check opposite signal
            opp_sig = (row["signal"]==-1 and d==1) or (row["signal"]==1 and d==-1)

            exit_reason = None
            exit_price  = cur_c

            if ema_exit:
                exit_reason="EMA Cross Exit"; exit_price=cur_c
            elif sl_hit:
                exit_reason="SL Hit";         exit_price=pos["sl"]
            elif tgt_hit:
                exit_reason="Target Hit";     exit_price=pos["target"]
            elif opp_sig and not allow_overlap:
                exit_reason="Opposite Signal"; exit_price=cur_c

            if exit_reason:
                q = pos.get("qty_remaining", qty)
                pnl = (exit_price - pos["entry_price"]) * d * q
                capital += pnl
                trades.append({
                    "entry_time"  : pos["entry_time"],
                    "exit_time"   : cur_t,
                    "direction"   : "LONG" if d==1 else "SHORT",
                    "entry_price" : round(pos["entry_price"],4),
                    "exit_price"  : round(exit_price,4),
                    "sl"          : round(pos["sl"],4),
                    "target"      : round(pos["target"],4),
                    "pnl"         : round(pnl,2),
                    "exit_reason" : exit_reason,
                    "capital"     : round(capital,2),
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # ── Open new position ──────────────────────────────────
        sig = int(row["signal"])
        if sig != 0:
            if not allow_overlap and open_positions:
                pass  # skip if position already open
            else:
                ep  = cur_c
                sl_ = calc_sl(df, i, ep, sig, sl_type, sp, atr_v)
                tg_ = calc_target(df, i, ep, sig, tgt_type, tp, atr_v)
                open_positions.append({
                    "direction"   : sig,
                    "entry_price" : ep,
                    "entry_time"  : cur_t,
                    "sl"          : sl_,
                    "target"      : tg_,
                })

    # Close any remaining open positions
    if open_positions and len(df) > 0:
        last_c = df["Close"].iloc[-1]
        last_t = df.index[-1]
        for pos in open_positions:
            d = pos["direction"]
            q = pos.get("qty_remaining", qty)
            pnl = (last_c - pos["entry_price"]) * d * q
            capital += pnl
            trades.append({
                "entry_time"  : pos["entry_time"],
                "exit_time"   : last_t,
                "direction"   : "LONG" if d==1 else "SHORT",
                "entry_price" : round(pos["entry_price"],4),
                "exit_price"  : round(last_c,4),
                "sl"          : round(pos["sl"],4),
                "target"      : round(pos["target"],4),
                "pnl"         : round(pnl,2),
                "exit_reason" : "End of Data",
                "capital"     : round(capital,2),
            })

    return trades

# ─────────────────────────────────────────────────────────────
# METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────
def calc_metrics(trades, initial_capital=100_000.0):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    n = len(df)
    wins   = df[df["pnl"]>0]
    losses = df[df["pnl"]<=0]
    wr = len(wins)/n*100 if n>0 else 0
    tot_pnl = df["pnl"].sum()
    avg_w = wins["pnl"].mean() if len(wins)>0 else 0
    avg_l = losses["pnl"].mean() if len(losses)>0 else 0
    gross_w = wins["pnl"].sum(); gross_l = losses["pnl"].sum()
    pf = abs(gross_w/gross_l) if gross_l!=0 else float("inf")
    cum = df["pnl"].cumsum()
    peak = cum.cummax()
    dd = (cum-peak).min()
    # Sharpe (daily)
    sh = (df["pnl"].mean()/df["pnl"].std()*np.sqrt(252)
          if df["pnl"].std()>0 else 0)
    # Expectancy
    exp_ = (wr/100)*avg_w + (1-wr/100)*avg_l

    c_wins=c_loss=mx_cw=mx_cl=0
    for p in df["pnl"]:
        if p>0:
            c_wins+=1; c_loss=0; mx_cw=max(mx_cw,c_wins)
        else:
            c_loss+=1; c_wins=0; mx_cl=max(mx_cl,c_loss)

    return {
        "Total Trades"      : n,
        "Win Rate (%)"      : round(wr,2),
        "Total P&L"         : round(tot_pnl,2),
        "Return (%)"        : round(tot_pnl/initial_capital*100,2),
        "Avg Win"           : round(avg_w,2),
        "Avg Loss"          : round(avg_l,2),
        "Profit Factor"     : round(pf,2),
        "Max Drawdown"      : round(dd,2),
        "Sharpe Ratio"      : round(sh,2),
        "Expectancy"        : round(exp_,2),
        "Max Consec Wins"   : mx_cw,
        "Max Consec Losses" : mx_cl,
        "Gross Profit"      : round(gross_w,2),
        "Gross Loss"        : round(gross_l,2),
    }

# ─────────────────────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────────────────────
def build_chart(df, trades, strategy, show_indicators=True):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.60,0.20,0.20],
                        subplot_titles=["Price","Volume","RSI"],
                        vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="Price",
        increasing_line_color="#000000",
        decreasing_line_color="#888888",
        increasing_fillcolor="#000000",
        decreasing_fillcolor="#cccccc",
    ), row=1, col=1)

    if show_indicators:
        ema9  = _ema(df["Close"],9)
        ema21 = _ema(df["Close"],21)
        fig.add_trace(go.Scatter(x=df.index,y=ema9, name="EMA9",
            line=dict(color="#000000",width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index,y=ema21, name="EMA21",
            line=dict(color="#444444",width=1,dash="dash")), row=1, col=1)

    # Trade markers
    if trades:
        tdf = pd.DataFrame(trades)
        longs  = tdf[tdf["direction"]=="LONG"]
        shorts = tdf[tdf["direction"]=="SHORT"]
        wins   = tdf[tdf["pnl"]>0]
        losses = tdf[tdf["pnl"]<=0]

        if not longs.empty:
            fig.add_trace(go.Scatter(x=longs["entry_time"],y=longs["entry_price"],
                mode="markers",name="Long Entry",
                marker=dict(symbol="triangle-up",color="#000000",size=10)), row=1, col=1)
        if not shorts.empty:
            fig.add_trace(go.Scatter(x=shorts["entry_time"],y=shorts["entry_price"],
                mode="markers",name="Short Entry",
                marker=dict(symbol="triangle-down",color="#555555",size=10)), row=1, col=1)
        if not wins.empty:
            fig.add_trace(go.Scatter(x=wins["exit_time"],y=wins["exit_price"],
                mode="markers",name="Win Exit",
                marker=dict(symbol="circle",color="#006400",size=8)), row=1, col=1)
        if not losses.empty:
            fig.add_trace(go.Scatter(x=losses["exit_time"],y=losses["exit_price"],
                mode="markers",name="Loss Exit",
                marker=dict(symbol="x",color="#8b0000",size=8)), row=1, col=1)

    # Volume
    colors = ["#000000" if c>=o else "#cccccc"
              for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors, showlegend=False), row=2, col=1)

    # RSI
    rsi_v = _rsi(df["Close"],14)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_v, name="RSI",
        line=dict(color="#000000",width=1)), row=3, col=1)
    fig.add_hline(y=70,line_dash="dash",line_color="#888888",row=3,col=1)
    fig.add_hline(y=30,line_dash="dash",line_color="#888888",row=3,col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        template="plotly_white", plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#000000",family="Segoe UI"),
        legend=dict(bgcolor="#ffffff",font=dict(color="#000000")),
        title=dict(text=f"<b>{strategy}</b>",font=dict(color="#000000",size=14))
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig

def build_equity_curve(trades, initial_capital=100_000.0):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    cum_pnl = df["pnl"].cumsum() + initial_capital
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cum_pnl))), y=cum_pnl,
        fill="tozeroy", fillcolor="rgba(0,0,0,0.08)",
        line=dict(color="#000000",width=2), name="Equity"))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="#888888")
    fig.update_layout(
        title="Equity Curve", height=300,
        template="plotly_white", plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#000000"),
        xaxis_title="Trade #", yaxis_title="Capital"
    )
    return fig

# ─────────────────────────────────────────────────────────────
# METRIC DISPLAY HELPER
# ─────────────────────────────────────────────────────────────
def show_metrics(metrics: dict):
    if not metrics:
        st.info("No metrics available yet.")
        return
    items = list(metrics.items())
    cols = st.columns(4)
    for i,(k,v) in enumerate(items):
        c = cols[i%4]
        is_pnl = "P&L" in k or "Profit" in k or "Loss" in k or "Return" in k
        color_cls = ""
        if is_pnl:
            color_cls = "win" if isinstance(v,(int,float)) and v>0 else "loss"
        c.markdown(f"""
        <div class="metric-card">
          <div class="label">{k}</div>
          <div class="value {color_cls}">{v}</div>
        </div>""", unsafe_allow_html=True)
    st.write("")

# ─────────────────────────────────────────────────────────────
# SIDEBAR  –  Global Configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Instrument ────────────────────────────────────────────
    st.markdown("### Instrument")
    ticker_name = st.selectbox("Select Instrument", list(TICKERS.keys()), key="sb_ticker")
    if ticker_name == "Custom Ticker":
        custom_ticker = st.text_input("Custom Yahoo Finance Ticker", "RELIANCE.NS", key="sb_custom")
        ACTIVE_TICKER = custom_ticker
    else:
        ACTIVE_TICKER = TICKERS[ticker_name]
    st.caption(f"Symbol: `{ACTIVE_TICKER}`")

    # ── Timeframe / Period ────────────────────────────────────
    st.markdown("### Timeframe & Period")
    tf = st.selectbox("Timeframe", TIMEFRAMES, index=2, key="sb_tf")
    valid_periods = TF_PERIOD_MAP.get(tf, ALL_PERIODS)
    period = st.selectbox("Period", valid_periods,
                          index=min(3, len(valid_periods)-1), key="sb_period")

    # ── Strategy ─────────────────────────────────────────────
    st.markdown("### Strategy")
    strategy = st.selectbox("Strategy", STRATEGIES, key="sb_strategy")

    # ── Capital & Qty ─────────────────────────────────────────
    st.markdown("### Capital & Quantity")
    initial_cap = st.number_input("Initial Capital (₹)", 10_000, 10_000_000, 100_000, 10_000)
    qty = st.number_input("Lot / Qty", 1, 10_000, 1)

    # ── Trading Time Filter ───────────────────────────────────
    st.markdown("### Trading Time Filter")
    use_time_filter = st.checkbox("Enable Time Filter (IST)", value=True, key="sb_tf_en")
    col1, col2 = st.columns(2)
    with col1:
        t_start = st.time_input("From", datetime.time(9,15), key="sb_tstart")
    with col2:
        t_end   = st.time_input("To",   datetime.time(15,30), key="sb_tend")

    # ── Overlap Setting ───────────────────────────────────────
    st.markdown("### Backtest Settings")
    allow_overlap = st.checkbox("Allow Overlapping Trades", value=False, key="sb_overlap")

    # ── Stop Loss ─────────────────────────────────────────────
    st.markdown("### Stop Loss")
    sl_type = st.selectbox("SL Type", SL_TYPES, index=5, key="sb_sl_type")
    sl_pts     = st.number_input("SL Points (Custom/Trailing)", 1, 10_000, 20, key="sb_sl_pts")
    atr_sl_m   = st.slider("ATR SL Multiplier", 0.5, 5.0, 2.0, 0.1, key="sb_atr_sl")
    rr_ratio_sl= st.slider("R:R Ratio (for RR-based SL)", 1.0, 10.0, 2.0, 0.5, key="sb_rr_sl")
    ema_fast_s = st.number_input("EMA Fast (Crossover SL)", 3, 100, 9,  key="sb_ema_f")
    ema_slow_s = st.number_input("EMA Slow (Crossover SL)", 5, 200, 21, key="sb_ema_sl")
    swing_lb_s = st.number_input("Swing Lookback (bars)", 3, 50, 5, key="sb_swing_sl")

    SL_PARAMS = {
        "sl_pts"       : sl_pts,
        "atr_sl_mult"  : atr_sl_m,
        "rr_target_pts": 40,
        "rr_ratio"     : rr_ratio_sl,
        "ema_fast"     : ema_fast_s,
        "ema_slow"     : ema_slow_s,
        "swing_lb"     : swing_lb_s,
    }

    # ── Target ────────────────────────────────────────────────
    st.markdown("### Target")
    tgt_type   = st.selectbox("Target Type", TARGET_TYPES, index=6, key="sb_tgt_type")
    tgt_pts    = st.number_input("Target Points", 1, 50_000, 40, key="sb_tgt_pts")
    atr_tgt_m  = st.slider("ATR Target Multiplier", 0.5, 10.0, 3.0, 0.5, key="sb_atr_tgt")
    rr_ratio_t = st.slider("R:R Ratio (Target)", 1.0, 10.0, 2.0, 0.5, key="sb_rr_tgt")
    swing_lb_t = st.number_input("Swing Lookback (Target)", 3, 50, 5, key="sb_swing_tgt")

    TGT_PARAMS = {
        "tgt_pts"      : tgt_pts,
        "atr_tgt_mult" : atr_tgt_m,
        "rr_sl_pts"    : sl_pts,
        "rr_ratio"     : rr_ratio_t,
        "swing_lb"     : swing_lb_t,
    }

    st.markdown("---")
    if st.button("🗑 Clear All Session Data"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
st.markdown("# 📈 AlgoTrader Pro")
st.markdown(f"**Instrument:** `{ACTIVE_TICKER}` &nbsp;|&nbsp; **Timeframe:** `{tf}` &nbsp;|&nbsp; **Period:** `{period}` &nbsp;|&nbsp; **Strategy:** {strategy}")
st.markdown("---")

tab_bt, tab_live, tab_hist, tab_analysis, tab_opt, tab_dhan = st.tabs([
    "📊 Backtesting",
    "⚡ Live Trading",
    "📋 Trade History",
    "🔍 Analysis",
    "🎯 Optimization",
    "🏦 Dhan API",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 – BACKTESTING
# ═════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown('<div class="section-header">Backtesting Engine</div>', unsafe_allow_html=True)

    col_cfg, col_run = st.columns([3,1])
    with col_run:
        st.write("")
        run_bt = st.button("▶ Run Backtest", key="btn_run_bt")
        show_ind = st.checkbox("Show EMA on Chart", value=True, key="bt_show_ind")

    with col_cfg:
        st.info(f"📌 **Config:** Strategy `{strategy}` | SL: `{sl_type}` | Target: `{tgt_type}` | "
                f"Overlap: `{'Yes' if allow_overlap else 'No'}` | "
                f"Time Filter: `{'ON' if use_time_filter else 'OFF'}`")

    if run_bt:
        # Clear previous results
        st.session_state.bt_results  = None
        st.session_state.bt_trades   = []
        st.session_state.bt_metrics  = {}

        with st.spinner(f"Fetching {ACTIVE_TICKER} data…"):
            df_raw = fetch_data(ACTIVE_TICKER, period, tf, force_fresh=True)

        if df_raw is None or df_raw.empty:
            st.error("Failed to fetch data. Check ticker / period / timeframe combination.")
        else:
            with st.spinner("Running backtest…"):
                trades = run_backtest(
                    df_raw, strategy, sl_type, tgt_type,
                    SL_PARAMS, TGT_PARAMS,
                    allow_overlap=allow_overlap,
                    use_tf=use_time_filter,
                    tf_start=t_start, tf_end=t_end,
                    initial_capital=float(initial_cap),
                    qty=int(qty)
                )
                metrics = calc_metrics(trades, float(initial_cap))

            st.session_state.bt_results = df_raw
            st.session_state.bt_trades  = trades
            st.session_state.bt_metrics = metrics

            # Also add to trade history
            for t in trades:
                t["strategy"] = strategy
                t["ticker"]   = ACTIVE_TICKER
            st.session_state.trade_history.extend(trades)

    # Display results
    if st.session_state.bt_results is not None:
        df_res = st.session_state.bt_results
        trades = st.session_state.bt_trades
        metrics = st.session_state.bt_metrics

        if not trades:
            st.warning("No trades generated. Try different parameters or a longer period.")
        else:
            st.success(f"✅ Backtest complete — {len(trades)} trades found.")
            st.markdown("### Performance Metrics")
            show_metrics(metrics)

            st.markdown("### Equity Curve")
            st.plotly_chart(build_equity_curve(trades, float(initial_cap)),
                            use_container_width=True)

            st.markdown("### Price Chart with Trades")
            st.plotly_chart(build_chart(df_res, trades, strategy, show_ind),
                            use_container_width=True)

            st.markdown("### Trade Log")
            tdf = pd.DataFrame(trades)
            tdf["pnl_color"] = tdf["pnl"].apply(lambda x: "🟢" if x>0 else "🔴")
            st.dataframe(tdf[["entry_time","exit_time","direction",
                               "entry_price","exit_price","sl","target",
                               "pnl_color","pnl","exit_reason","capital"]],
                         use_container_width=True, height=320)

# ═════════════════════════════════════════════════════════════
# TAB 2 – LIVE TRADING
# ═════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="section-header">Live Trading Monitor</div>', unsafe_allow_html=True)

    st.markdown("#### Active Configuration")
    cfg_cols = st.columns(4)
    cfg_cols[0].markdown(f"**Ticker:** `{ACTIVE_TICKER}`")
    cfg_cols[1].markdown(f"**Timeframe:** `{tf}`")
    cfg_cols[2].markdown(f"**Strategy:** {strategy[:30]}…")
    cfg_cols[3].markdown(f"**Capital:** ₹{initial_cap:,}")
    cfg_cols2 = st.columns(4)
    cfg_cols2[0].markdown(f"**SL Type:** {sl_type}")
    cfg_cols2[1].markdown(f"**Target Type:** {tgt_type}")
    cfg_cols2[2].markdown(f"**Time Filter:** {'✅' if use_time_filter else '❌'} {t_start}–{t_end}")
    cfg_cols2[3].markdown(f"**Overlap:** {'✅' if allow_overlap else '❌'}")
    st.markdown("---")

    lc1, lc2, lc3 = st.columns(3)
    start_live = lc1.button("▶ Start Live Scan", key="btn_live_start")
    stop_live  = lc2.button("⏹ Stop", key="btn_live_stop")
    refresh_s  = lc3.number_input("Refresh (sec)", 10, 300, 60, key="live_refresh")

    if stop_live:
        st.session_state.live_running = False
        st.info("Live scanning stopped.")

    if start_live:
        st.session_state.live_running   = True
        st.session_state.live_open_pos  = None
        st.session_state.live_last_signal = None

    live_placeholder = st.empty()
    signal_box       = st.empty()

    if st.session_state.live_running:
        with live_placeholder.container():
            now_ist = datetime.datetime.now(IST)
            st.markdown(f'<span class="live-badge">● LIVE</span> &nbsp; Last scan: **{now_ist.strftime("%H:%M:%S IST")}**',
                        unsafe_allow_html=True)

            # Time filter check
            in_time = True
            if use_time_filter:
                in_time = t_start <= now_ist.time() <= t_end
                if not in_time:
                    st.warning(f"⏰ Outside trading hours ({t_start}–{t_end} IST). Waiting…")

            if in_time:
                with st.spinner("Fetching latest data…"):
                    df_live = fetch_data(ACTIVE_TICKER, period, tf, force_fresh=True)

                if df_live is not None and not df_live.empty:
                    df_sig = compute_signals(df_live, strategy)
                    atr_s  = _atr(df_sig["High"],df_sig["Low"],df_sig["Close"],14)
                    ema_f  = _ema(df_sig["Close"], SL_PARAMS.get("ema_fast",9))
                    ema_sl_ = _ema(df_sig["Close"], SL_PARAMS.get("ema_slow",21))

                    last_sig = int(df_sig["signal"].iloc[-1])
                    last_c   = float(df_sig["Close"].iloc[-1])
                    last_h   = float(df_sig["High"].iloc[-1])
                    last_l   = float(df_sig["Low"].iloc[-1])
                    atr_v    = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else last_c*0.01
                    last_t   = df_sig.index[-1]

                    st.session_state.live_last_signal = last_sig

                    # Current position management
                    pos = st.session_state.live_open_pos
                    if pos is not None:
                        d = pos["direction"]
                        pos = update_trail_sl(pos, df_sig, -1, sl_type, SL_PARAMS, atr_v)
                        pos["current_price"] = last_c
                        pos["unrealized_pnl"] = round((last_c - pos["entry_price"]) * d * qty, 2)

                        # EMA exit
                        ema_exit = False
                        if sl_type=="EMA Reverse Crossover" or tgt_type=="EMA Reverse Crossover":
                            if d==1 and float(ema_f.iloc[-1])<float(ema_sl_.iloc[-1]):
                                ema_exit=True
                            elif d==-1 and float(ema_f.iloc[-1])>float(ema_sl_.iloc[-1]):
                                ema_exit=True

                        sl_hit  = (d==1 and last_l<=pos["sl"]) or (d==-1 and last_h>=pos["sl"])
                        tgt_hit = (d==1 and last_h>=pos["target"]) or (d==-1 and last_l<=pos["target"])

                        if ema_exit or sl_hit or tgt_hit:
                            reason = "EMA Exit" if ema_exit else ("SL Hit" if sl_hit else "Target Hit")
                            ep = pos["sl"] if sl_hit else (pos["target"] if tgt_hit else last_c)
                            pnl = (ep - pos["entry_price"]) * d * qty
                            st.session_state.live_trades.append({
                                "entry_time"  : pos["entry_time"],
                                "exit_time"   : last_t,
                                "direction"   : "LONG" if d==1 else "SHORT",
                                "entry_price" : pos["entry_price"],
                                "exit_price"  : round(ep,4),
                                "pnl"         : round(pnl,2),
                                "exit_reason" : reason,
                            })
                            st.session_state.live_open_pos = None
                            pnl_color = "win" if pnl>0 else "loss"
                            st.markdown(f'<div class="signal-{"long" if pnl>0 else "short"}">'
                                        f'⚡ Position closed — {reason} | P&L: '
                                        f'<span class="{pnl_color}">₹{pnl:,.2f}</span></div>',
                                        unsafe_allow_html=True)

                        st.session_state.live_open_pos = pos

                    # Open new position on signal
                    if last_sig != 0 and st.session_state.live_open_pos is None:
                        sl_  = calc_sl(df_sig, -1, last_c, last_sig, sl_type, SL_PARAMS, atr_v)
                        tg_  = calc_target(df_sig, -1, last_c, last_sig, tgt_type, TGT_PARAMS, atr_v)
                        st.session_state.live_open_pos = {
                            "direction"   : last_sig,
                            "entry_price" : last_c,
                            "entry_time"  : last_t,
                            "sl"          : sl_,
                            "target"      : tg_,
                            "current_price": last_c,
                            "unrealized_pnl": 0.0,
                        }

                    # ── Signal display ─────────────────────────────────────
                    st.markdown("#### Latest Signal")
                    if last_sig == 1:
                        signal_box.markdown(
                            '<div class="signal-long">🟢 <b>LONG Signal</b> detected on last candle</div>',
                            unsafe_allow_html=True)
                    elif last_sig == -1:
                        signal_box.markdown(
                            '<div class="signal-short">🔴 <b>SHORT Signal</b> detected on last candle</div>',
                            unsafe_allow_html=True)
                    else:
                        signal_box.markdown(
                            '<div class="signal-none">⬜ No Signal on last candle</div>',
                            unsafe_allow_html=True)

                    # ── Open position display ───────────────────────────────
                    pos = st.session_state.live_open_pos
                    if pos:
                        st.markdown("#### Open Position")
                        p_cols = st.columns(5)
                        p_cols[0].metric("Direction", "🟢 LONG" if pos["direction"]==1 else "🔴 SHORT")
                        p_cols[1].metric("Entry", f"{pos['entry_price']:.4f}")
                        p_cols[2].metric("SL",    f"{pos['sl']:.4f}")
                        p_cols[3].metric("Target",f"{pos['target']:.4f}")
                        pnl_v = pos.get("unrealized_pnl",0)
                        p_cols[4].metric("Unrealized P&L", f"₹{pnl_v:,.2f}",
                                         delta=f"₹{pnl_v:,.2f}")
                    else:
                        st.info("No open position.")

                    # ── Recent live chart ───────────────────────────────────
                    st.markdown("#### Live Chart (Last 100 candles)")
                    chart_df = df_sig.tail(100)
                    st.plotly_chart(build_chart(chart_df, [], strategy, True),
                                    use_container_width=True)

                    # ── Live trades session ─────────────────────────────────
                    if st.session_state.live_trades:
                        st.markdown("#### Live Session Trades")
                        ltd = pd.DataFrame(st.session_state.live_trades)
                        st.dataframe(ltd, use_container_width=True, height=200)
                        total_live_pnl = ltd["pnl"].sum()
                        pnl_cls = "win" if total_live_pnl>0 else "loss"
                        st.markdown(f'Session P&L: <span class="{pnl_cls}"><b>₹{total_live_pnl:,.2f}</b></span>',
                                    unsafe_allow_html=True)

        st.info(f"⏱ Next refresh in {refresh_s}s. Press **Stop** to halt. Auto-refreshes require manual page reload or a loop.")

# ═════════════════════════════════════════════════════════════
# TAB 3 – TRADE HISTORY
# ═════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown('<div class="section-header">Trade History</div>', unsafe_allow_html=True)

    all_trades = (st.session_state.get("trade_history", []) +
                  st.session_state.get("live_trades", []))

    if not all_trades:
        st.info("No trades yet. Run a backtest or live session first.")
    else:
        hist_df = pd.DataFrame(all_trades)

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        if "strategy" in hist_df.columns:
            strats_avail = ["All"] + sorted(hist_df["strategy"].unique().tolist())
            sel_strat = fc1.selectbox("Filter Strategy", strats_avail, key="hist_strat_f")
        else:
            sel_strat = "All"
        if "ticker" in hist_df.columns:
            tickers_avail = ["All"] + sorted(hist_df["ticker"].unique().tolist())
            sel_tick = fc2.selectbox("Filter Ticker", tickers_avail, key="hist_tick_f")
        else:
            sel_tick = "All"
        dir_opts = ["All","LONG","SHORT"]
        sel_dir = fc3.selectbox("Direction", dir_opts, key="hist_dir_f")

        filt = hist_df.copy()
        if sel_strat != "All" and "strategy" in filt.columns:
            filt = filt[filt["strategy"]==sel_strat]
        if sel_tick != "All" and "ticker" in filt.columns:
            filt = filt[filt["ticker"]==sel_tick]
        if sel_dir != "All":
            filt = filt[filt["direction"]==sel_dir]

        st.markdown(f"**Showing {len(filt)} trades**")
        cols_show = [c for c in ["entry_time","exit_time","direction","ticker","strategy",
                                 "entry_price","exit_price","pnl","exit_reason"] if c in filt.columns]
        st.dataframe(filt[cols_show], use_container_width=True, height=400)

        if len(filt) > 0:
            m = calc_metrics(filt.to_dict("records"), float(initial_cap))
            st.markdown("### Aggregate Metrics")
            show_metrics(m)

        if st.button("🗑 Clear Trade History", key="btn_clear_hist"):
            st.session_state.trade_history = []
            st.session_state.live_trades   = []
            st.rerun()

# ═════════════════════════════════════════════════════════════
# TAB 4 – ANALYSIS
# ═════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown('<div class="section-header">Deep Analysis</div>', unsafe_allow_html=True)

    an_btn = st.button("🔄 Load & Analyse Data", key="btn_analysis")
    if an_btn:
        with st.spinner("Fetching data…"):
            df_an = fetch_data(ACTIVE_TICKER, period, tf, force_fresh=True)
        if df_an is not None and not df_an.empty:
            st.session_state["analysis_df"] = df_an

    df_an = st.session_state.get("analysis_df")
    if df_an is not None:
        st.markdown(f"**{len(df_an)} candles loaded** — {df_an.index[0].strftime('%Y-%m-%d %H:%M')} → {df_an.index[-1].strftime('%Y-%m-%d %H:%M')}")

        a_tabs = st.tabs(["OHLCV","Indicators","Strategy Signals","Statistics","Correlation"])

        with a_tabs[0]:
            st.plotly_chart(build_chart(df_an.tail(200), [], strategy, False), use_container_width=True)
            st.dataframe(df_an.tail(50), use_container_width=True)

        with a_tabs[1]:
            ind_sel = st.multiselect("Choose Indicators", 
                ["EMA(9,21,50)", "Bollinger Bands", "Supertrend", "MACD", "RSI",
                 "Stochastic", "ATR", "ADX", "CCI", "VWAP", "Ichimoku", "Parabolic SAR"],
                default=["EMA(9,21,50)","RSI","MACD"], key="ind_sel")

            sub_df = df_an.copy()
            fig_ind = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    row_heights=[0.5,0.17,0.17,0.16],
                                    vertical_spacing=0.03)
            fig_ind.add_trace(go.Candlestick(
                x=sub_df.index, open=sub_df["Open"], high=sub_df["High"],
                low=sub_df["Low"], close=sub_df["Close"], name="Price",
                increasing_line_color="#000", decreasing_line_color="#888",
                increasing_fillcolor="#000", decreasing_fillcolor="#ccc",
            ), row=1, col=1)

            colors_ = ["#000000","#444444","#888888"]
            if "EMA(9,21,50)" in ind_sel:
                for p_,c_ in zip([9,21,50],colors_):
                    fig_ind.add_trace(go.Scatter(x=sub_df.index, y=_ema(sub_df["Close"],p_),
                        name=f"EMA{p_}", line=dict(color=c_,width=1)), row=1, col=1)

            if "Bollinger Bands" in ind_sel:
                ub_,mb_,lb__ = _bb(sub_df["Close"],20,2)
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=ub_,name="BB Up",
                    line=dict(color="#999",width=1,dash="dot")), row=1, col=1)
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=lb__,name="BB Low",
                    line=dict(color="#999",width=1,dash="dot"),fill="tonexty",
                    fillcolor="rgba(0,0,0,0.04)"), row=1, col=1)

            if "Supertrend" in ind_sel:
                st_,dir__ = _supertrend(sub_df["High"],sub_df["Low"],sub_df["Close"])
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=st_,name="Supertrend",
                    line=dict(color="#333",width=1)), row=1, col=1)

            if "MACD" in ind_sel:
                ml_,sl__,hi_ = _macd(sub_df["Close"])
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=ml_,name="MACD",
                    line=dict(color="#000",width=1)), row=2, col=1)
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=sl__,name="Signal",
                    line=dict(color="#666",width=1,dash="dash")), row=2, col=1)
                fig_ind.add_trace(go.Bar(x=sub_df.index,y=hi_,name="Hist",
                    marker_color=["#000" if v>=0 else "#aaa" for v in hi_.fillna(0)]), row=2, col=1)

            if "RSI" in ind_sel:
                r_ = _rsi(sub_df["Close"],14)
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=r_,name="RSI",
                    line=dict(color="#000",width=1)), row=3, col=1)
                fig_ind.add_hline(y=70,line_dash="dash",line_color="#888",row=3,col=1)
                fig_ind.add_hline(y=30,line_dash="dash",line_color="#888",row=3,col=1)

            if "Stochastic" in ind_sel:
                k_,d_ = _stoch(sub_df["High"],sub_df["Low"],sub_df["Close"])
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=k_,name="%K",
                    line=dict(color="#000",width=1)), row=3, col=1)
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=d_,name="%D",
                    line=dict(color="#666",width=1,dash="dash")), row=3, col=1)

            if "ATR" in ind_sel:
                atr__ = _atr(sub_df["High"],sub_df["Low"],sub_df["Close"],14)
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=atr__,name="ATR",
                    line=dict(color="#000",width=1)), row=4, col=1)

            if "ADX" in ind_sel:
                adx__,pdi__,ndi__ = _adx(sub_df["High"],sub_df["Low"],sub_df["Close"])
                fig_ind.add_trace(go.Scatter(x=sub_df.index,y=adx__,name="ADX",
                    line=dict(color="#000",width=1)), row=4, col=1)

            fig_ind.update_layout(height=800, template="plotly_white",
                                  plot_bgcolor="#fff", paper_bgcolor="#fff",
                                  font=dict(color="#000"),
                                  xaxis_rangeslider_visible=False)
            fig_ind.update_xaxes(gridcolor="#eee"); fig_ind.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_ind, use_container_width=True)

        with a_tabs[2]:
            df_sigs = compute_signals(df_an, strategy)
            total_l = (df_sigs["signal"]==1).sum()
            total_s = (df_sigs["signal"]==-1).sum()
            sc1, sc2 = st.columns(2)
            sc1.metric("Long Signals",  total_l)
            sc2.metric("Short Signals", total_s)

            # Show signal distribution over time
            fig_sig = go.Figure()
            long_idx  = df_sigs[df_sigs["signal"]==1]
            short_idx = df_sigs[df_sigs["signal"]==-1]
            fig_sig.add_trace(go.Scatter(x=df_sigs.index, y=df_sigs["Close"],
                name="Price", line=dict(color="#000",width=1)))
            if not long_idx.empty:
                fig_sig.add_trace(go.Scatter(x=long_idx.index, y=long_idx["Close"],
                    mode="markers", name="LONG",
                    marker=dict(symbol="triangle-up",color="#000",size=9)))
            if not short_idx.empty:
                fig_sig.add_trace(go.Scatter(x=short_idx.index, y=short_idx["Close"],
                    mode="markers", name="SHORT",
                    marker=dict(symbol="triangle-down",color="#555",size=9)))
            fig_sig.update_layout(height=400,template="plotly_white",
                                  plot_bgcolor="#fff",paper_bgcolor="#fff",
                                  font=dict(color="#000"))
            st.plotly_chart(fig_sig, use_container_width=True)

        with a_tabs[3]:
            st.markdown("#### Descriptive Statistics")
            st.dataframe(df_an.describe(), use_container_width=True)
            # Returns distribution
            rets = df_an["Close"].pct_change().dropna()
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(x=rets*100, nbinsx=50, name="Returns",
                marker_color="#000", opacity=0.7))
            fig_ret.update_layout(title="Daily Returns Distribution (%)",
                                  height=300, template="plotly_white",
                                  plot_bgcolor="#fff", paper_bgcolor="#fff",
                                  font=dict(color="#000"),
                                  xaxis_title="Return (%)", yaxis_title="Frequency")
            st.plotly_chart(fig_ret, use_container_width=True)

        with a_tabs[4]:
            st.markdown("#### Indicator Correlation Heatmap")
            corr_df = pd.DataFrame({
                "Close"      : df_an["Close"],
                "EMA9"       : _ema(df_an["Close"],9),
                "EMA21"      : _ema(df_an["Close"],21),
                "RSI14"      : _rsi(df_an["Close"],14),
                "MACD"       : _macd(df_an["Close"])[0],
                "ATR14"      : _atr(df_an["High"],df_an["Low"],df_an["Close"],14),
                "CCI20"      : _cci(df_an["High"],df_an["Low"],df_an["Close"],20),
                "Volume"     : df_an["Volume"],
            }).dropna()
            corr_matrix = corr_df.corr()
            fig_hm = go.Figure(go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale=[[0,"#ffffff"],[0.5,"#999999"],[1,"#000000"]],
                text=corr_matrix.round(2).values, texttemplate="%{text}",
                showscale=True))
            fig_hm.update_layout(height=450, template="plotly_white",
                                  plot_bgcolor="#fff", paper_bgcolor="#fff",
                                  font=dict(color="#000"))
            st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Click **Load & Analyse Data** to begin.")

# ═════════════════════════════════════════════════════════════
# TAB 5 – OPTIMIZATION
# ═════════════════════════════════════════════════════════════
with tab_opt:
    st.markdown('<div class="section-header">Strategy Optimizer — Target 90%+ Win Rate</div>',
                unsafe_allow_html=True)
    st.warning("⚠️ Optimization runs many backtests. Large grids may take several minutes. "
               "Reduce parameter ranges if needed.")

    st.markdown("#### Optimization Parameters")
    oc1, oc2 = st.columns(2)
    with oc1:
        opt_strategy  = st.selectbox("Strategy to Optimize", STRATEGIES, key="opt_strat")
        opt_target_wr = st.slider("Target Win Rate (%)", 50, 95, 70, 5, key="opt_wr")
        opt_min_trades= st.number_input("Min Trades Required", 5, 500, 20, key="opt_min_tr")
        opt_ticker    = st.text_input("Ticker for Opt", ACTIVE_TICKER, key="opt_tick")
        opt_tf        = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(tf), key="opt_tf")
        opt_period    = st.selectbox("Period", valid_periods, key="opt_per")
    with oc2:
        st.markdown("**EMA Fast range:**")
        ef_lo,ef_hi = st.slider("EMA Fast",3,100,(5,20),key="opt_ef")
        ef_step     = st.number_input("Step",1,10,2,key="opt_ef_s")
        st.markdown("**EMA Slow range:**")
        es_lo,es_hi = st.slider("EMA Slow",10,200,(15,60),key="opt_es")
        es_step     = st.number_input("Step ",1,20,5,key="opt_es_s")
        opt_sl_type = st.selectbox("SL for Opt", SL_TYPES, index=5, key="opt_sl_t")
        opt_tgt_type= st.selectbox("Target for Opt", TARGET_TYPES, index=6, key="opt_tgt_t")
        opt_sl_pts  = st.number_input("SL Pts (opt)", 5, 500, 20, key="opt_sl_p")
        opt_tgt_pts = st.number_input("Target Pts (opt)", 5, 1000, 40, key="opt_tgt_p")

    run_opt = st.button("🚀 Run Optimization", key="btn_opt")

    if run_opt:
        st.session_state.opt_results = None
        with st.spinner("Fetching data for optimization…"):
            df_opt = fetch_data(opt_ticker, opt_period, opt_tf, force_fresh=True)
        if df_opt is None or df_opt.empty:
            st.error("Failed to fetch data.")
        else:
            ef_range = list(range(ef_lo, ef_hi+1, max(ef_step,1)))
            es_range = list(range(es_lo, es_hi+1, max(es_step,1)))
            combos   = [(ef,es) for ef in ef_range for es in es_range if ef < es]

            if len(combos) > 200:
                st.warning(f"Limiting to first 200 out of {len(combos)} combinations.")
                combos = combos[:200]

            best_results = []
            prog = st.progress(0)
            prog_text = st.empty()

            sp_ = {"sl_pts":opt_sl_pts,"atr_sl_mult":2.0,"rr_target_pts":opt_tgt_pts,
                   "rr_ratio":2.0,"ema_fast":9,"ema_slow":21,"swing_lb":5}
            tp_ = {"tgt_pts":opt_tgt_pts,"atr_tgt_mult":3.0,"rr_sl_pts":opt_sl_pts,
                   "rr_ratio":2.0,"swing_lb":5}

            for idx_,(ef,es) in enumerate(combos):
                prog_text.text(f"Testing EMA({ef},{es}) — {idx_+1}/{len(combos)}")
                prog.progress((idx_+1)/len(combos))

                try:
                    df_tmp = df_opt.copy()
                    df_s   = compute_signals(df_tmp, opt_strategy, {"fast":ef,"slow":es,"e1":ef,"e2":es})
                    tr_    = run_backtest(df_opt, opt_strategy, opt_sl_type, opt_tgt_type,
                                         sp_, tp_, allow_overlap=False, initial_capital=float(initial_cap))
                    mt_    = calc_metrics(tr_, float(initial_cap))

                    if mt_ and mt_.get("Total Trades",0) >= opt_min_trades:
                        best_results.append({
                            "EMA Fast"    : ef,
                            "EMA Slow"    : es,
                            "Win Rate (%)" : mt_["Win Rate (%)"],
                            "Total Trades" : mt_["Total Trades"],
                            "Total P&L"   : mt_["Total P&L"],
                            "Profit Factor": mt_["Profit Factor"],
                            "Max Drawdown" : mt_["Max Drawdown"],
                            "Return (%)"   : mt_["Return (%)"],
                            "Sharpe"       : mt_["Sharpe Ratio"],
                        })
                except:
                    continue

            prog.empty(); prog_text.empty()

            if best_results:
                opt_df = pd.DataFrame(best_results).sort_values("Win Rate (%)", ascending=False)
                st.session_state.opt_results = opt_df
            else:
                st.warning("No valid combinations found. Try wider ranges or more data.")

    # Display optimization results
    opt_df = st.session_state.get("opt_results")
    if opt_df is not None and not opt_df.empty:
        st.markdown("### Optimization Results")
        st.success(f"Found {len(opt_df)} valid parameter sets. Sorted by Win Rate.")

        # Top 10
        top10 = opt_df.head(10)
        st.markdown("#### Top 10 Configurations")
        st.dataframe(top10, use_container_width=True)

        # Best config
        best = opt_df.iloc[0]
        st.markdown("---")
        st.markdown("### ⭐ Best Configuration")
        bc1,bc2,bc3,bc4 = st.columns(4)
        bc1.metric("EMA Fast",  int(best["EMA Fast"]))
        bc2.metric("EMA Slow",  int(best["EMA Slow"]))
        bc3.metric("Win Rate",  f"{best['Win Rate (%)']:.1f}%")
        bc4.metric("Profit Factor", f"{best['Profit Factor']:.2f}")

        st.info(f"💡 **Recommended:** Use EMA Fast = **{int(best['EMA Fast'])}**, "
                f"EMA Slow = **{int(best['EMA Slow'])}** with **{opt_strategy}** strategy. "
                f"Win Rate: **{best['Win Rate (%)']:.1f}%** | Trades: **{int(best['Total Trades'])}** | "
                f"P&L: **₹{best['Total P&L']:,.0f}**")

        if best["Win Rate (%)"] >= opt_target_wr:
            st.success(f"✅ Target of {opt_target_wr}% win rate ACHIEVED!")
        else:
            st.warning(f"⚠️ Best win rate {best['Win Rate (%)']:.1f}% is below target {opt_target_wr}%. "
                       f"Consider a different strategy or instrument.")

        # Scatter plot
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(
            x=opt_df["EMA Fast"], y=opt_df["Win Rate (%)"],
            mode="markers",
            marker=dict(
                size=opt_df["Total Trades"].clip(1)/opt_df["Total Trades"].max()*20+5,
                color=opt_df["Profit Factor"].clip(0,5),
                colorscale=[[0,"#cccccc"],[0.5,"#666666"],[1,"#000000"]],
                colorbar=dict(title="Profit Factor"),
                showscale=True
            ),
            text=[f"EMA({r['EMA Fast']},{r['EMA Slow']})<br>WR:{r['Win Rate (%)']:.1f}%"
                  for _,r in opt_df.iterrows()],
            hovertemplate="%{text}<extra></extra>"
        ))
        fig_opt.update_layout(title="Optimization Landscape",
                              xaxis_title="EMA Fast", yaxis_title="Win Rate (%)",
                              height=400, template="plotly_white",
                              plot_bgcolor="#fff", paper_bgcolor="#fff",
                              font=dict(color="#000"))
        st.plotly_chart(fig_opt, use_container_width=True)

        if st.button("📋 Export Optimization Results as CSV", key="btn_opt_csv"):
            csv = opt_df.to_csv(index=False)
            st.download_button("⬇ Download CSV", csv, "optimization_results.csv",
                               "text/csv", key="dl_opt_csv")

# ═════════════════════════════════════════════════════════════
# TAB 6 – DHAN API
# ═════════════════════════════════════════════════════════════
with tab_dhan:
    st.markdown('<div class="section-header">Dhan API — Live Order Placement</div>',
                unsafe_allow_html=True)
    st.info("📌 Fill in your Dhan credentials to enable live order placement. "
            "Orders are placed based on signals from the Live Trading tab.")

    dc1, dc2 = st.columns(2)
    with dc1:
        dhan_client_id  = st.text_input("Dhan Client ID", type="password", key="dhan_cid")
        dhan_token      = st.text_input("Dhan Access Token", type="password", key="dhan_tok")
        dhan_enabled    = st.checkbox("Enable Live Order Placement", value=False, key="dhan_en")
    with dc2:
        dhan_product    = st.selectbox("Product Type", ["INTRADAY","CNC","MARGIN"], key="dhan_prod")
        dhan_order_type = st.selectbox("Order Type", ["MARKET","LIMIT","SL","SL-M"], key="dhan_ord_t")
        dhan_exchange   = st.selectbox("Exchange", ["NSE","BSE","MCX","NFO"], key="dhan_exch")
        dhan_security_id= st.text_input("Security ID (Dhan Symbol ID)", "13", key="dhan_sec_id")

    st.markdown("---")
    st.markdown("#### Order Placement Code (Dhan SDK)")

    dhan_code = '''
# ─────────────────────────────────────────────────────────────
# DHAN ORDER PLACEMENT  (Install: pip install dhanhq)
# ─────────────────────────────────────────────────────────────
# from dhanhq import dhanhq

# Initialize Dhan client
# dhan = dhanhq(client_id=DHAN_CLIENT_ID, access_token=DHAN_ACCESS_TOKEN)

def place_dhan_order(signal: int, qty: int, price: float = 0,
                     security_id: str = "13",
                     exchange_segment: str = "NSE_EQ",
                     product_type: str = "INTRADAY",
                     order_type: str = "MARKET") -> dict:
    """
    Place BUY or SELL order via Dhan API.
    signal:  1 = BUY (Long),  -1 = SELL (Short)
    """
    # transaction_type = dhan.BUY if signal == 1 else dhan.SELL

    # response = dhan.place_order(
    #     security_id     = security_id,
    #     exchange_segment= exchange_segment,     # NSE_EQ / NSE_FNO / MCX_COMM
    #     transaction_type= transaction_type,
    #     quantity        = qty,
    #     order_type      = order_type,           # MARKET / LIMIT / SL / SL-M
    #     product_type    = product_type,         # INTRADAY / CNC / MARGIN
    #     price           = price,                # 0 for MARKET
    #     trigger_price   = 0,                    # For SL orders
    #     disclosed_quantity = 0,
    #     after_market_order = False,
    #     validity        = "DAY",
    #     amo_time        = "OPEN",
    #     bo_profit_value = 0,
    #     bo_stop_loss_Value = 0,
    # )
    # return response

    # ── Simulated response for testing ──────────────────────
    action = "BUY" if signal == 1 else "SELL"
    return {
        "status"   : "success",
        "orderId"  : "SIMULATED_ORDER_12345",
        "action"   : action,
        "qty"      : qty,
        "price"    : price,
        "timestamp": str(datetime.datetime.now(IST)),
    }


def cancel_dhan_order(order_id: str) -> dict:
    """Cancel an existing Dhan order"""
    # response = dhan.cancel_order(order_id=order_id)
    # return response
    return {"status": "cancelled", "orderId": order_id}


def get_positions() -> list:
    """Fetch open positions from Dhan"""
    # return dhan.get_positions()
    return []


def get_order_status(order_id: str) -> dict:
    """Check order status"""
    # return dhan.get_order_by_id(order_id=order_id)
    return {"status": "TRADED", "orderId": order_id}


# ── Integration with AlgoTrader signals ─────────────────────
def execute_signal(signal: int, last_close: float, sl: float, target: float) -> dict:
    """
    Called by live trading loop when a new signal is generated.
    Handles entry order + SL/Target bracket orders.
    """
    if signal == 0:
        return {}

    entry_order = place_dhan_order(
        signal=signal, qty=1, price=0,  # MARKET order
        order_type="MARKET"
    )

    # ── After entry, place SL order ──────────────────────────
    # sl_order = place_dhan_order(
    #     signal=-signal,   # opposite direction for SL
    #     qty=1,
    #     price=sl,
    #     order_type="SL-M",
    #     trigger_price=sl,
    # )

    # ── Place Target (Limit) order ───────────────────────────
    # tgt_order = place_dhan_order(
    #     signal=-signal,
    #     qty=1,
    #     price=target,
    #     order_type="LIMIT",
    # )

    return {
        "entry" : entry_order,
        # "sl"    : sl_order,
        # "target": tgt_order,
    }
'''
    st.code(dhan_code, language="python")

    st.markdown("---")
    st.markdown("#### Test Signal Placement")
    tc1, tc2, tc3 = st.columns(3)
    test_signal = tc1.selectbox("Signal", ["BUY (Long)","SELL (Short)"], key="dhan_test_sig")
    test_price  = tc2.number_input("Price", 0.0, 1_000_000.0, 0.0, key="dhan_test_pr")
    test_qty    = tc3.number_input("Qty", 1, 10_000, int(qty), key="dhan_test_qty")

    if st.button("📤 Test Order (Simulated)", key="btn_dhan_test"):
        sig_val = 1 if "BUY" in test_signal else -1
        result = {
            "status"   : "success (simulated)",
            "action"   : test_signal,
            "qty"      : test_qty,
            "price"    : test_price if test_price > 0 else "MARKET",
            "timestamp": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            "note"     : "Enable Dhan credentials to place real orders."
        }
        st.json(result)

    st.markdown("---")
    st.markdown("#### Dhan SDK Installation")
    st.code("pip install dhanhq", language="bash")
    st.caption("Docs: https://dhanhq.co/docs/v2/")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#888;font-size:12px;'>"
    "AlgoTrader Pro · Built with Streamlit + yfinance · "
    "All indicators computed in pure Python/NumPy/Pandas · "
    "Not financial advice — trade at your own risk."
    "</center>",
    unsafe_allow_html=True
)
