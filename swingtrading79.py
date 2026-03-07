
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
      background: #000000 !important;
      color: #ffffff !important;
      border-radius: 6px; border: none !important;
      padding: 8px 22px; font-weight: 600;
      transition: background 0.2s;
  }
  div.stButton > button * { color: #ffffff !important; }
  div.stButton > button:hover { background: #333333 !important; color: #ffffff !important; }
  div.stButton > button:active { background: #111111 !important; color: #ffffff !important; }
  div.stDownloadButton > button {
      background: #000000 !important; color: #ffffff !important;
      border-radius: 6px; border: none !important; font-weight: 600;
  }
  div.stDownloadButton > button:hover { background: #333333 !important; }
  .stSelectbox label, .stMultiSelect label,
  .stNumberInput label,
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
  p { color: #000 !important; }
  label:not([data-baseweb]) { color: #000 !important; }
  .stMarkdown, .stText { color: #000 !important; }
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

def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA — matches TradingView ta.rma(). alpha = 1/n."""
    return s.ewm(alpha=1.0/n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """RSI using Wilder's RMA — identical to TradingView ta.rsi()."""
    delta = s.diff()
    avg_gain = _rma(delta.clip(lower=0), n)
    avg_loss = _rma((-delta).clip(lower=0), n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def _macd(s: pd.Series, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow)
    si = _ema(m, sig)
    return m, si, m - si

def _bb(s: pd.Series, n=20, k=2.0):
    """Bollinger Bands — matches TradingView ta.bb() (sample stdev, ddof=1)."""
    m   = _sma(s, n)
    std = s.rolling(n).std(ddof=1)
    return m + k * std, m, m - k * std

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n=14) -> pd.Series:
    """ATR using Wilder's RMA — matches TradingView ta.atr()."""
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return _rma(tr, n)

def _stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return k, _sma(k, dp)

def _adx(h, l, c, n=14):
    """ADX/DI — matches TradingView ta.adx() using Wilder's RMA."""
    up   = h.diff()
    down = -l.diff()
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    ndm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=l.index)
    atr_ = _atr(h, l, c, n)
    pdi  = 100.0 * _rma(pdm, n) / atr_.replace(0, np.nan)
    ndi  = 100.0 * _rma(ndm, n) / atr_.replace(0, np.nan)
    dx   = 100.0 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return _rma(dx, n), pdi, ndi

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
# ENTRY / EXIT LOGIC TEXT HELPERS
# ─────────────────────────────────────────────────────────────
def _entry_logic_text(strategy: str, direction: int, df, i: int, atr_v: float, ep: float) -> str:
    d_str = "LONG" if direction==1 else "SHORT"
    c = df["Close"]; h = df["High"]; l = df["Low"]
    try:
        num = int(strategy.split(".")[0].strip())
    except:
        num = 0

    try:
        if num in (1,2,21):
            ef = _ema(c,9).iloc[i]; es = _ema(c,21).iloc[i]
            return (f"{d_str}: EMA9({ef:.2f}) crossed {'above' if direction==1 else 'below'} "
                    f"EMA21({es:.2f}). Price: {ep:.2f}. ATR={atr_v:.2f}.")
        elif num == 3:
            return (f"{d_str}: Triple EMA stack {'bullish' if direction==1 else 'bearish'} "
                    f"(EMA9>EMA21>EMA50 or reverse). Price: {ep:.2f}.")
        elif num == 4:
            ml,sl_,_ = _macd(c); mv=ml.iloc[i]; sv=sl_.iloc[i]
            return (f"{d_str}: MACD({mv:.4f}) crossed {'above' if direction==1 else 'below'} "
                    f"Signal({sv:.4f}). Price: {ep:.2f}.")
        elif num == 5:
            rv = _rsi(c,14).iloc[i]
            cond = "exited oversold (<30)" if direction==1 else "exited overbought (>70)"
            return f"{d_str}: RSI({rv:.1f}) {cond}. Price: {ep:.2f}. ATR={atr_v:.2f}."
        elif num in (6,7):
            ub,mb,lb_ = _bb(c)
            return (f"{d_str}: Price {'broke above upper BB' if (num==6 and direction==1) else 'broke below lower BB' if (num==6 and direction==-1) else 'touched lower BB (mean-revert)' if direction==1 else 'touched upper BB (mean-revert)'}. "
                    f"BB Upper={ub.iloc[i]:.2f} Lower={lb_.iloc[i]:.2f}. Price: {ep:.2f}.")
        elif num == 8:
            _,dir_ = _supertrend(h,l,c)
            return (f"{d_str}: Supertrend flipped to {'bullish' if direction==1 else 'bearish'}. "
                    f"Price: {ep:.2f}. ATR={atr_v:.2f}.")
        elif num == 9:
            adx_,pdi,ndi = _adx(h,l,c)
            return (f"{d_str}: +DI({pdi.iloc[i]:.1f}) crossed {'above' if direction==1 else 'below'} "
                    f"-DI({ndi.iloc[i]:.1f}), ADX={adx_.iloc[i]:.1f}. Price: {ep:.2f}.")
        elif num in (10,11):
            k,d_ = _stoch(h,l,c)
            return (f"{d_str}: Stoch %K({k.iloc[i]:.1f}) crossed {'above' if direction==1 else 'below'} "
                    f"%D({d_.iloc[i]:.1f}). Price: {ep:.2f}.")
        elif num == 13:
            vw = _vwap(h,l,c,df["Volume"])
            return (f"{d_str}: Price({c.iloc[i]:.2f}) crossed {'above' if direction==1 else 'below'} "
                    f"VWAP({vw.iloc[i]:.2f}). Entry: {ep:.2f}.")
        elif num == 29:
            sar,_ = _psar(h,l,c)
            return (f"{d_str}: SAR flipped {'bullish (price above SAR)' if direction==1 else 'bearish (price below SAR)'}. "
                    f"SAR={sar.iloc[i]:.2f}. Price: {ep:.2f}.")
        else:
            return f"{d_str}: Strategy #{num} signal triggered at {ep:.2f}. ATR={atr_v:.2f}."
    except:
        return f"{d_str}: Signal triggered at {ep:.2f}."

def _exit_logic_text(reason: str, pos: dict, exit_price: float, d: int) -> str:
    ep = pos["entry_price"]
    pts = (exit_price - ep) * d
    if reason == "SL Hit":
        return (f"Stop Loss triggered at {exit_price:.2f}. "
                f"Lost {abs(pts):.2f} pts. SL was set at {pos['sl']:.2f} "
                f"({'trailing' if 'Trail' in pos.get('entry_type','') else 'fixed'}).")
    elif reason == "Target Hit":
        return (f"Target reached at {exit_price:.2f}. "
                f"Gained {abs(pts):.2f} pts. Target was {pos['target']:.2f}.")
    elif reason == "EMA Cross Exit":
        return (f"EMA crossover reversed — exit at {exit_price:.2f}. "
                f"P&L: {pts:+.2f} pts.")
    elif reason == "Opposite Signal":
        return (f"Opposite signal triggered — exit at {exit_price:.2f}. "
                f"Strategy flipped direction. P&L: {pts:+.2f} pts.")
    elif reason == "End of Data":
        return f"Backtest ended — position closed at {exit_price:.2f}. P&L: {pts:+.2f} pts."
    return f"Exit at {exit_price:.2f}. P&L: {pts:+.2f} pts."

# ─────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
def run_backtest(df_raw, strategy, sl_type, tgt_type, sp, tp,
                 allow_overlap=False, use_tf=False, tf_start=None, tf_end=None,
                 qty=1):

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
                    pos["lv1_done"] = True
                    pos["qty_remaining"] = max(1, int(qty * 0.5))
                elif not pos.get("lv2_done") and pos.get("lv1_done") and \
                     ((d==1 and cur_h>=lv2)or(d==-1 and cur_l<=lv2)):
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
                q   = pos.get("qty_remaining", qty)
                pnl = (exit_price - pos["entry_price"]) * d * q
                pts = (exit_price - pos["entry_price"]) * d   # points per unit
                trades.append({
                    # ── Timing ──────────────────────────────────────
                    "entry_time"      : pos["entry_time"],
                    "exit_time"       : cur_t,
                    # ── Direction & Type ────────────────────────────
                    "direction"       : "LONG" if d==1 else "SHORT",
                    "entry_type"      : pos.get("entry_type","Signal"),
                    "entry_logic"     : pos.get("entry_logic",""),
                    # ── Price Levels ────────────────────────────────
                    "entry_price"     : round(pos["entry_price"],4),
                    "exit_price"      : round(exit_price,4),
                    "sl_level"        : round(pos["sl_initial"],4),
                    "final_sl"        : round(pos["sl"],4),
                    "target_level"    : round(pos["target"],4),
                    # ── Points ──────────────────────────────────────
                    "points_captured" : round(pts,4),
                    "points_risked"   : round(abs(pos["entry_price"]-pos["sl_initial"]),4),
                    # ── P&L ─────────────────────────────────────────
                    "qty"             : q,
                    "pnl"             : round(pnl,2),
                    "result"          : "WIN" if pnl>0 else ("LOSS" if pnl<0 else "BE"),
                    # ── Exit ────────────────────────────────────────
                    "exit_reason"     : exit_reason,
                    "exit_logic"      : _exit_logic_text(exit_reason, pos, exit_price, d),
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
                # Use NEXT bar open for realistic entry (matches live trading)
                if i + 1 < len(df):
                    ep = float(df["Open"].iloc[i+1])
                    entry_t = df.index[i+1]
                else:
                    ep = cur_c
                    entry_t = cur_t
                sl_ = calc_sl(df, i, ep, sig, sl_type, sp, atr_v)
                tg_ = calc_target(df, i, ep, sig, tgt_type, tp, atr_v)
                entry_logic = _entry_logic_text(strategy, sig, df, i, atr_v, ep)
                open_positions.append({
                    "direction"   : sig,
                    "entry_price" : ep,
                    "entry_time"  : entry_t,
                    "sl"          : sl_,
                    "sl_initial"  : sl_,
                    "target"      : tg_,
                    "entry_type"  : "Signal-Bar+1 Open",
                    "entry_logic" : entry_logic,
                })

    # Close any remaining open positions
    if open_positions and len(df) > 0:
        last_c = df["Close"].iloc[-1]
        last_t = df.index[-1]
        for pos in open_positions:
            d   = pos["direction"]
            q   = pos.get("qty_remaining", qty)
            pnl = (last_c - pos["entry_price"]) * d * q
            pts = (last_c - pos["entry_price"]) * d
            trades.append({
                "entry_time"      : pos["entry_time"],
                "exit_time"       : last_t,
                "direction"       : "LONG" if d==1 else "SHORT",
                "entry_type"      : pos.get("entry_type","Signal"),
                "entry_logic"     : pos.get("entry_logic",""),
                "entry_price"     : round(pos["entry_price"],4),
                "exit_price"      : round(last_c,4),
                "sl_level"        : round(pos["sl_initial"],4),
                "final_sl"        : round(pos["sl"],4),
                "target_level"    : round(pos["target"],4),
                "points_captured" : round(pts,4),
                "points_risked"   : round(abs(pos["entry_price"]-pos["sl_initial"]),4),
                "qty"             : q,
                "pnl"             : round(pnl,2),
                "result"          : "WIN" if pnl>0 else ("LOSS" if pnl<0 else "BE"),
                "exit_reason"     : "End of Data",
                "exit_logic"      : "Backtest period ended — position closed at last bar close.",
            })

    return trades

# ─────────────────────────────────────────────────────────────
# METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────
def calc_metrics(trades, initial_capital=0.0):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    n  = len(df)

    # Resolve P&L column robustly
    pnl_col = "pnl" if "pnl" in df.columns else None
    if pnl_col is None:
        return {}

    wins   = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] <= 0]
    wr     = len(wins) / n * 100 if n > 0 else 0

    tot_pnl  = df[pnl_col].sum()
    avg_w    = wins[pnl_col].mean()   if len(wins)   > 0 else 0.0
    avg_l    = losses[pnl_col].mean() if len(losses) > 0 else 0.0
    gross_w  = wins[pnl_col].sum()
    gross_l  = losses[pnl_col].sum()
    pf       = abs(gross_w / gross_l) if gross_l != 0 else float("inf")

    # Points stats
    pts_col  = "points_captured" if "points_captured" in df.columns else pnl_col
    tot_pts  = df[pts_col].sum()
    pts_won  = df.loc[df[pnl_col]>0,  pts_col].sum() if len(wins)>0   else 0.0
    pts_lost = df.loc[df[pnl_col]<=0, pts_col].sum() if len(losses)>0 else 0.0
    avg_pts_w = df.loc[df[pnl_col]>0,  pts_col].mean() if len(wins)>0  else 0.0
    avg_pts_l = df.loc[df[pnl_col]<=0, pts_col].mean() if len(losses)>0 else 0.0

    cum  = df[pnl_col].cumsum()
    peak = cum.cummax()
    dd   = (cum - peak).min()
    sh   = (df[pnl_col].mean() / df[pnl_col].std() * np.sqrt(252)
            if df[pnl_col].std() > 0 else 0)
    exp_ = (wr/100) * avg_w + (1 - wr/100) * avg_l

    c_wins = c_loss = mx_cw = mx_cl = 0
    for p in df[pnl_col]:
        if p > 0:
            c_wins += 1; c_loss = 0; mx_cw = max(mx_cw, c_wins)
        else:
            c_loss += 1; c_wins = 0; mx_cl = max(mx_cl, c_loss)

    avg_dur = "N/A"
    try:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"]  = pd.to_datetime(df["exit_time"])
        durs    = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
        avg_dur = f"{durs.mean():.0f} min"
    except:
        pass

    return {
        "Total Trades"        : n,
        "Win Rate (%)"        : round(wr, 2),
        "Accuracy"            : f"{wr:.1f}% ({len(wins)}W / {len(losses)}L)",
        "Total P&L (pts×qty)" : round(tot_pnl, 2),
        "Total Points Gained" : round(pts_won, 2),
        "Total Points Lost"   : round(pts_lost, 2),
        "Net Points"          : round(tot_pts, 2),
        "Avg Win (pts)"       : round(avg_pts_w, 2),
        "Avg Loss (pts)"      : round(avg_pts_l, 2),
        "Gross Profit"        : round(gross_w, 2),
        "Gross Loss"          : round(gross_l, 2),
        "Profit Factor"       : round(pf, 2),
        "Max Drawdown"        : round(dd, 2),
        "Sharpe Ratio"        : round(sh, 2),
        "Expectancy"          : round(exp_, 2),
        "Max Consec Wins"     : mx_cw,
        "Max Consec Losses"   : mx_cl,
        "Avg Trade Duration"  : avg_dur,
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

def build_equity_curve(trades, initial_capital=0.0):
    if not trades:
        return go.Figure()
    df = pd.DataFrame(trades)
    cum_pnl = df["pnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cum_pnl))), y=cum_pnl,
        fill="tozeroy", fillcolor="rgba(0,0,0,0.08)",
        line=dict(color="#000000",width=2), name="Cumulative P&L"))
    fig.add_hline(y=0, line_dash="dash", line_color="#888888")
    fig.update_layout(
        title="Cumulative P&L Curve (Points × Qty)", height=300,
        template="plotly_white", plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#000000"),
        xaxis_title="Trade #", yaxis_title="Cumulative P&L"
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
# MARKET SUMMARY  –  human-readable analysis
# ─────────────────────────────────────────────────────────────
def generate_market_summary(df: pd.DataFrame, ticker: str) -> str:
    """Returns a plain-English paragraph describing the current market state."""
    if df is None or len(df) < 30:
        return "Insufficient data to generate a market summary."

    c   = df["Close"]
    h   = df["High"]
    l   = df["Low"]
    v   = df["Volume"]

    last_c   = float(c.iloc[-1])
    prev_c   = float(c.iloc[-2])
    chg_pct  = (last_c - prev_c) / prev_c * 100

    # Trend
    ema9_v  = float(_ema(c,9).iloc[-1])
    ema21_v = float(_ema(c,21).iloc[-1])
    ema50_v = float(_ema(c,50).iloc[-1]) if len(c)>=50 else ema21_v
    if ema9_v > ema21_v > ema50_v:
        trend = "in a strong UPTREND — all EMAs (9, 21, 50) are stacked bullishly"
        trend_action = "Prefer LONG setups. Look for dips to EMA9 or EMA21 as buying opportunities."
    elif ema9_v < ema21_v < ema50_v:
        trend = "in a strong DOWNTREND — all EMAs (9, 21, 50) are stacked bearishly"
        trend_action = "Prefer SHORT setups. Rallies to EMA9 or EMA21 are sell opportunities."
    elif ema9_v > ema21_v:
        trend = "in a short-term BULLISH phase (EMA9 above EMA21), though medium-term trend is mixed"
        trend_action = "Cautious LONG trades can be considered, but confirm with volume."
    else:
        trend = "in a short-term BEARISH phase (EMA9 below EMA21)"
        trend_action = "Avoid aggressive longs. Sideways or downward bias expected in the short term."

    # RSI
    rsi_v = float(_rsi(c,14).iloc[-1])
    if rsi_v > 70:
        rsi_msg = f"RSI is at {rsi_v:.1f} — OVERBOUGHT territory. The market may be overextended; avoid chasing longs. Watch for exhaustion or reversal candles."
    elif rsi_v < 30:
        rsi_msg = f"RSI is at {rsi_v:.1f} — OVERSOLD territory. Selling pressure may be exhausting; a bounce or reversal is possible. But in strong downtrends, oversold can stay oversold."
    elif rsi_v > 55:
        rsi_msg = f"RSI is at {rsi_v:.1f} — Moderately bullish momentum. Buyers are in control without being overstretched."
    elif rsi_v < 45:
        rsi_msg = f"RSI is at {rsi_v:.1f} — Moderately bearish momentum. Sellers have a slight edge."
    else:
        rsi_msg = f"RSI is at {rsi_v:.1f} — Neutral zone. No strong bias from momentum alone; wait for directional confirmation."

    # MACD
    ml,sl_,hi_ = _macd(c)
    macd_v = float(ml.iloc[-1]); macd_sig = float(sl_.iloc[-1]); hist_v = float(hi_.iloc[-1])
    if macd_v > macd_sig and hist_v > 0:
        macd_msg = "MACD is above its signal line with positive histogram — bullish momentum is building."
    elif macd_v < macd_sig and hist_v < 0:
        macd_msg = "MACD is below its signal line with negative histogram — bearish momentum is dominant."
    elif macd_v > macd_sig and hist_v < 0:
        macd_msg = "MACD crossed above signal but histogram is shrinking — momentum is slowing; watch carefully."
    else:
        macd_msg = "MACD just crossed below its signal line — early bearish signal, confirmation needed."

    # Bollinger Bands
    ub,mb,lb_ = _bb(c,20,2)
    ub_v=float(ub.iloc[-1]); lb_v=float(lb_.iloc[-1]); mb_v=float(mb.iloc[-1])
    band_width = (ub_v - lb_v) / mb_v * 100
    if last_c > ub_v:
        bb_msg = f"Price is ABOVE the upper Bollinger Band ({ub_v:.2f}) — price is stretched. Breakouts can continue but a mean-reversion pullback to the middle band ({mb_v:.2f}) is common."
    elif last_c < lb_v:
        bb_msg = f"Price is BELOW the lower Bollinger Band ({lb_v:.2f}) — deeply oversold on a statistical basis. A snap-back to the middle band ({mb_v:.2f}) is historically likely."
    elif band_width < 2:
        bb_msg = f"Bollinger Bands are SQUEEZING (width: {band_width:.1f}%) — volatility is very low. Expect a sharp breakout soon; direction is unknown until it triggers."
    else:
        bb_msg = f"Price is trading INSIDE the Bollinger Bands (upper: {ub_v:.2f}, lower: {lb_v:.2f}). No extreme stretching detected."

    # ATR / Volatility
    atr_v = float(_atr(h,l,c,14).iloc[-1])
    atr_pct = atr_v / last_c * 100
    if atr_pct > 2:
        vol_msg = f"Volatility is HIGH (ATR={atr_v:.2f}, {atr_pct:.1f}% of price). Use wider stops and reduce position size. Large moves are possible in both directions."
    elif atr_pct < 0.5:
        vol_msg = f"Volatility is LOW (ATR={atr_v:.2f}, {atr_pct:.1f}% of price). Tight stops are viable, but breakouts may be muted."
    else:
        vol_msg = f"Volatility is MODERATE (ATR={atr_v:.2f}, {atr_pct:.1f}% of price). Normal risk management applies."

    # Volume
    avg_vol = float(v.rolling(20).mean().iloc[-1]) if len(v)>=20 else float(v.mean())
    last_vol = float(v.iloc[-1])
    if avg_vol > 0:
        vol_ratio = last_vol / avg_vol
        if vol_ratio > 1.5:
            vol_msg2 = f"Volume on the last candle is {vol_ratio:.1f}x the 20-bar average — STRONG participation. This move has conviction behind it."
        elif vol_ratio < 0.5:
            vol_msg2 = f"Volume is only {vol_ratio:.1f}x the 20-bar average — LOW participation. Treat any breakout or breakdown with skepticism until volume confirms."
        else:
            vol_msg2 = "Volume is near its 20-bar average — normal participation, no special volume signal."
    else:
        vol_msg2 = "Volume data not available for this instrument."

    # Support / Resistance (simple)
    recent_high = float(h.rolling(20).max().iloc[-1])
    recent_low  = float(l.rolling(20).min().iloc[-1])
    dist_to_hi  = (recent_high - last_c) / last_c * 100
    dist_to_lo  = (last_c - recent_low)  / last_c * 100

    # Candle bias
    candle_dir = "BULLISH" if last_c >= prev_c else "BEARISH"
    candle_chg = abs(chg_pct)

    # Final recommendation
    bull_signals = sum([
        ema9_v > ema21_v,
        rsi_v > 50,
        macd_v > macd_sig,
        last_c > mb_v,
        last_c >= prev_c,
    ])
    if bull_signals >= 4:
        overall = "Overall bias: BULLISH. Multiple indicators align. Prefer long trades with proper risk management."
    elif bull_signals <= 1:
        overall = "Overall bias: BEARISH. Multiple indicators point down. Prefer short trades or stay in cash."
    else:
        overall = "Overall bias: MIXED/NEUTRAL. Conflicting signals. Wait for clearer directional confirmation before trading."

    summary = f"""
**{ticker} — Market Intelligence Summary**

The instrument is currently trading at **{last_c:.2f}**, {candle_dir} by **{candle_chg:.2f}%** from the previous candle close. It is {trend}.

**Trend:** {trend_action}

**Momentum (RSI):** {rsi_msg}

**MACD:** {macd_msg}

**Bollinger Bands:** {bb_msg}

**Volatility:** {vol_msg} {vol_msg2}

**Key Levels:** 20-bar resistance near **{recent_high:.2f}** (price is {dist_to_hi:.1f}% away). 20-bar support near **{recent_low:.2f}** (price is {dist_to_lo:.1f}% above it).

**⚡ {overall}**

*Note: This is a technical analysis summary based on historical price data. Always apply your own judgment, position sizing, and never risk more than you can afford to lose.*
"""
    return summary.strip()

# ─────────────────────────────────────────────────────────────
# LIVE COMMENTARY  –  real-time market reading
# ─────────────────────────────────────────────────────────────
def generate_live_commentary(df: pd.DataFrame, last_c: float,
                              last_sig: int, pos: dict | None,
                              atr_v: float, ticker: str) -> str:
    """Returns a real-time plain-English commentary for the live trading panel."""
    if df is None or len(df) < 20:
        return "Waiting for enough data to generate commentary…"

    c = df["Close"]; h = df["High"]; l = df["Low"]
    ema9_v  = float(_ema(c,9).iloc[-1])
    ema21_v = float(_ema(c,21).iloc[-1])
    rsi_v   = float(_rsi(c,14).iloc[-1])
    ml,sl_,_ = _macd(c)
    macd_v  = float(ml.iloc[-1]); macd_sig_v = float(sl_.iloc[-1])
    ub,mb,lb_ = _bb(c,20,2)
    ub_v=float(ub.iloc[-1]); lb_v=float(lb_.iloc[-1]); mb_v=float(mb.iloc[-1])

    # Price vs key levels
    if last_c > ema9_v > ema21_v:
        price_level = f"Price ({last_c:.2f}) is ABOVE both EMA9 ({ema9_v:.2f}) and EMA21 ({ema21_v:.2f}) — bullish structure intact."
    elif last_c < ema9_v < ema21_v:
        price_level = f"Price ({last_c:.2f}) is BELOW both EMA9 ({ema9_v:.2f}) and EMA21 ({ema21_v:.2f}) — bearish structure."
    elif last_c > ema21_v:
        price_level = f"Price ({last_c:.2f}) is between EMA9 ({ema9_v:.2f}) and EMA21 ({ema21_v:.2f}) — testing key EMA support/resistance."
    else:
        price_level = f"Price ({last_c:.2f}) is below EMA21 ({ema21_v:.2f}) — caution advised."

    # RSI reading
    if rsi_v > 70:
        rsi_note = f"RSI {rsi_v:.1f} — overbought. Do NOT add to longs here. Consider partial profit booking if long."
    elif rsi_v < 30:
        rsi_note = f"RSI {rsi_v:.1f} — oversold. Do NOT add to shorts here. Watch for reversal."
    elif rsi_v > 55:
        rsi_note = f"RSI {rsi_v:.1f} — bullish momentum."
    elif rsi_v < 45:
        rsi_note = f"RSI {rsi_v:.1f} — bearish momentum."
    else:
        rsi_note = f"RSI {rsi_v:.1f} — neutral."

    # BB position
    if last_c > ub_v:
        bb_note = f"Price above upper BB ({ub_v:.2f}) — statistically stretched. Avoid new longs."
    elif last_c < lb_v:
        bb_note = f"Price below lower BB ({lb_v:.2f}) — extreme oversold. Potential snap-back zone."
    else:
        bb_note = f"Price inside BB bands. Normal range. Mid-band at {mb_v:.2f}."

    # Signal
    if last_sig == 1:
        sig_note = "🟢 FRESH LONG SIGNAL on this candle. Strategy suggests entering a BUY position."
        action = "✅ ACTION: Consider entering LONG with appropriate SL and Target as configured."
    elif last_sig == -1:
        sig_note = "🔴 FRESH SHORT SIGNAL on this candle. Strategy suggests entering a SELL position."
        action = "✅ ACTION: Consider entering SHORT with appropriate SL and Target as configured."
    else:
        sig_note = "⬜ No new signal. Market is in a wait state per the selected strategy."
        action = "⏳ ACTION: Stay patient. No new trade trigger. Manage existing positions if any."

    # Position status
    if pos is not None:
        d = pos["direction"]
        ep = pos["entry_price"]
        sl = pos["sl"]
        tg = pos["target"]
        upnl = (last_c - ep) * d
        risk_pts = abs(ep - sl)
        rwd_pts  = abs(tg - ep)
        pos_note = (
            f"**Open Position:** {'LONG' if d==1 else 'SHORT'} entered at {ep:.2f}. "
            f"SL at {sl:.2f} (risk: {risk_pts:.2f} pts). Target: {tg:.2f} (reward: {rwd_pts:.2f} pts). "
            f"Unrealized: {upnl:+.2f} pts × qty. "
        )
        if d == 1:
            if last_c < ema9_v:
                pos_note += "⚠️ Price dipped below EMA9 — monitor closely, consider tightening SL."
            elif upnl > rwd_pts * 0.6:
                pos_note += "💡 Approaching 60% of target. Consider partial booking."
        else:
            if last_c > ema9_v:
                pos_note += "⚠️ Price recovered above EMA9 — monitor closely for short squeeze."
            elif abs(upnl) > rwd_pts * 0.6:
                pos_note += "💡 Approaching 60% of target. Consider partial booking."
    else:
        pos_note = "No open position. System is scanning for the next entry."

    commentary = f"""
{price_level}
{rsi_note} | {bb_note}
MACD: {'Bullish' if macd_v > macd_sig_v else 'Bearish'} — MACD {macd_v:.2f} vs Signal {macd_sig_v:.2f}.
ATR (volatility buffer): {atr_v:.2f} pts.

{sig_note}
{pos_note}

{action}
"""
    return commentary.strip()

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

    # ── Quantity ──────────────────────────────────────────────
    st.markdown("### Quantity")
    qty = st.number_input("Lot / Qty (default 1)", 1, 10_000, 1, key="sb_qty")
    st.caption("All P&L shown as: Points × Qty")

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
    sl_type    = st.selectbox("SL Type", SL_TYPES, index=5, key="sb_sl_type")
    sl_pts     = st.number_input("SL Points (Custom/Trailing)", 1, 10_000, 20, key="sb_sl_pts")
    atr_sl_m   = st.number_input("ATR SL Multiplier", min_value=0.1, max_value=10.0, value=2.0, step=0.1, format="%.1f", key="sb_atr_sl")
    rr_ratio_sl= st.number_input("R:R Ratio (RR-based SL)", min_value=0.5, max_value=20.0, value=2.0, step=0.5, format="%.1f", key="sb_rr_sl")
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
    atr_tgt_m  = st.number_input("ATR Target Multiplier", min_value=0.5, max_value=20.0, value=3.0, step=0.5, format="%.1f", key="sb_atr_tgt")
    rr_ratio_t = st.number_input("R:R Ratio (Target)", min_value=0.5, max_value=20.0, value=2.0, step=0.5, format="%.1f", key="sb_rr_tgt")
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
                    qty=int(qty)
                )
                metrics = calc_metrics(trades)

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
            st.plotly_chart(build_equity_curve(trades),
                            use_container_width=True)

            st.markdown("### Price Chart with Trades")
            st.plotly_chart(build_chart(df_res, trades, strategy, show_ind),
                            use_container_width=True)

            st.markdown("### Trade Log — Full Detail")
            tdf = pd.DataFrame(trades)
            # Cumulative running totals
            tdf["cum_pnl"]    = tdf["pnl"].cumsum().round(2)
            tdf["cum_pts"]    = tdf["points_captured"].cumsum().round(2) if "points_captured" in tdf.columns else tdf["pnl"].cumsum().round(2)
            tdf["result_icon"]= tdf["pnl"].apply(lambda x: "🟢 WIN" if x>0 else ("🔴 LOSS" if x<0 else "⬜ BE"))

            display_cols = [
                c for c in [
                    "entry_time","exit_time","direction","entry_type",
                    "entry_price","exit_price","sl_level","target_level","final_sl",
                    "points_captured","points_risked","qty","pnl","result_icon",
                    "cum_pts","cum_pnl","exit_reason","entry_logic","exit_logic"
                ] if c in tdf.columns
            ]
            st.dataframe(tdf[display_cols], use_container_width=True, height=380)

            # Summary totals row
            if "points_captured" in tdf.columns:
                pts_won_  = tdf.loc[tdf["pnl"]>0,"points_captured"].sum()
                pts_lost_ = tdf.loc[tdf["pnl"]<=0,"points_captured"].sum()
                net_pts_  = tdf["points_captured"].sum()
                sc1,sc2,sc3,sc4 = st.columns(4)
                sc1.metric("Total Points Gained", f"{pts_won_:.2f}")
                sc2.metric("Total Points Lost",   f"{pts_lost_:.2f}")
                sc3.metric("Net Points",           f"{net_pts_:.2f}")
                sc4.metric("Total P&L (pts×qty)",  f"{tdf['pnl'].sum():.2f}")

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
    cfg_cols[3].markdown(f"**Qty:** {qty}")
    cfg_cols2 = st.columns(4)
    cfg_cols2[0].markdown(f"**SL Type:** {sl_type}")
    cfg_cols2[1].markdown(f"**Target Type:** {tgt_type}")
    cfg_cols2[2].markdown(f"**Time Filter:** {'✅' if use_time_filter else '❌'} {t_start}–{t_end}")
    cfg_cols2[3].markdown(f"**Overlap:** {'✅' if allow_overlap else '❌'}")
    st.markdown("---")

    lc1, lc2, lc3 = st.columns(3)
    start_live = lc1.button("▶ Start Live Scan", key="btn_live_start")
    stop_live  = lc2.button("⏹ Stop", key="btn_live_stop")
    refresh_s  = lc3.number_input("Refresh interval (sec)", min_value=1.0, max_value=300.0,
                                   value=1.5, step=0.5, format="%.1f", key="live_refresh")

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

                    # ── Use CONFIRMED (closed) candle for signal — matches backtest bar+1 ──
                    # Signal from last CLOSED bar (iloc[-2]), price levels from current bar
                    sig_bar_idx = -2 if len(df_sig) >= 2 else -1
                    last_sig = int(df_sig["signal"].iloc[sig_bar_idx])
                    last_c   = float(df_sig["Close"].iloc[-1])   # current bar close
                    last_h   = float(df_sig["High"].iloc[-1])
                    last_l   = float(df_sig["Low"].iloc[-1])
                    last_o   = float(df_sig["Open"].iloc[-1])    # current bar open = entry price
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
                            ep_exit = pos["sl"] if sl_hit else (pos["target"] if tgt_hit else last_c)
                            pnl  = (ep_exit - pos["entry_price"]) * d * qty
                            pts  = (ep_exit - pos["entry_price"]) * d
                            st.session_state.live_trades.append({
                                "entry_time"      : pos["entry_time"],
                                "exit_time"       : last_t,
                                "direction"       : "LONG" if d==1 else "SHORT",
                                "entry_type"      : pos.get("entry_type","Signal"),
                                "entry_logic"     : pos.get("entry_logic",""),
                                "entry_price"     : pos["entry_price"],
                                "exit_price"      : round(ep_exit,4),
                                "sl_level"        : round(pos.get("sl_initial", pos["sl"]),4),
                                "final_sl"        : round(pos["sl"],4),
                                "target_level"    : round(pos["target"],4),
                                "points_captured" : round(pts,4),
                                "points_risked"   : round(abs(pos["entry_price"]-pos.get("sl_initial",pos["sl"])),4),
                                "qty"             : qty,
                                "pnl"             : round(pnl,2),
                                "result"          : "WIN" if pnl>0 else ("LOSS" if pnl<0 else "BE"),
                                "exit_reason"     : reason,
                                "exit_logic"      : _exit_logic_text(reason, pos, ep_exit, d),
                            })
                            st.session_state.live_open_pos = None
                            pnl_color = "win" if pnl>0 else "loss"
                            st.markdown(f'<div class="signal-{"long" if pnl>0 else "short"}">'
                                        f'⚡ Position closed — {reason} | P&L: '
                                        f'<span class="{pnl_color}">{pnl:+.2f} pts×qty</span></div>',
                                        unsafe_allow_html=True)

                        st.session_state.live_open_pos = pos

                    # Open new position on confirmed signal
                    if last_sig != 0 and st.session_state.live_open_pos is None:
                        entry_price_live = last_o   # current bar open = "next bar after signal"
                        sl_  = calc_sl(df_sig, -1, entry_price_live, last_sig, sl_type, SL_PARAMS, atr_v)
                        tg_  = calc_target(df_sig, -1, entry_price_live, last_sig, tgt_type, TGT_PARAMS, atr_v)
                        entry_logic_live = _entry_logic_text(strategy, last_sig, df_sig, sig_bar_idx, atr_v, entry_price_live)
                        st.session_state.live_open_pos = {
                            "direction"     : last_sig,
                            "entry_price"   : entry_price_live,
                            "entry_time"    : last_t,
                            "sl"            : sl_,
                            "sl_initial"    : sl_,
                            "target"        : tg_,
                            "current_price" : last_c,
                            "unrealized_pnl": round((last_c - entry_price_live) * last_sig * qty, 2),
                            "entry_type"    : "Signal-Bar+1 Open (Live)",
                            "entry_logic"   : entry_logic_live,
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

                    # ── Live Commentary ─────────────────────────────────────
                    st.markdown("#### 📢 Live Market Commentary")
                    commentary = generate_live_commentary(
                        df_sig, last_c, last_sig,
                        st.session_state.live_open_pos,
                        atr_v, ACTIVE_TICKER
                    )
                    st.info(commentary)

                    # ── Open position display ───────────────────────────────
                    pos = st.session_state.live_open_pos
                    if pos:
                        st.markdown("#### Open Position")
                        p_cols = st.columns(6)
                        p_cols[0].metric("Direction",    "🟢 LONG" if pos["direction"]==1 else "🔴 SHORT")
                        p_cols[1].metric("Entry Price",  f"{pos['entry_price']:.4f}")
                        p_cols[2].metric("Current",      f"{last_c:.4f}")
                        p_cols[3].metric("SL",           f"{pos['sl']:.4f}")
                        p_cols[4].metric("Target",       f"{pos['target']:.4f}")
                        pnl_v = pos.get("unrealized_pnl",0)
                        p_cols[5].metric("Unrealized (pts×qty)", f"{pnl_v:+.2f}", delta=f"{pnl_v:+.2f}")
                        if pos.get("entry_logic"):
                            st.caption(f"📌 Entry Logic: {pos['entry_logic']}")
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
                        ltd["cum_pnl"] = ltd["pnl"].cumsum().round(2)
                        live_display_cols = [c for c in [
                            "entry_time","exit_time","direction","entry_price","exit_price",
                            "sl_level","target_level","points_captured","points_risked",
                            "qty","pnl","result","cum_pnl","exit_reason","entry_logic"
                        ] if c in ltd.columns]
                        st.dataframe(ltd[live_display_cols], use_container_width=True, height=220)
                        total_live_pnl = ltd["pnl"].sum()
                        total_live_pts = ltd["points_captured"].sum() if "points_captured" in ltd.columns else total_live_pnl
                        pnl_cls = "win" if total_live_pnl > 0 else "loss"
                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric("Session P&L",    f"{total_live_pnl:+.2f}")
                        s2.metric("Net Points",     f"{total_live_pts:+.2f}")
                        s3.metric("Session Trades", len(ltd))
                        wr_live = len(ltd[ltd["pnl"]>0])/len(ltd)*100 if len(ltd)>0 else 0
                        s4.metric("Session Win Rate", f"{wr_live:.1f}%")

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
            m = calc_metrics(filt.to_dict("records"))
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
            # Quick stats bar
            last_c_an = float(df_an["Close"].iloc[-1])
            prev_c_an = float(df_an["Close"].iloc[-2]) if len(df_an)>1 else last_c_an
            chg_an = last_c_an - prev_c_an
            chg_pct_an = chg_an/prev_c_an*100 if prev_c_an>0 else 0
            qs1,qs2,qs3,qs4 = st.columns(4)
            qs1.metric("Last Close", f"{last_c_an:.4f}")
            qs2.metric("Change", f"{chg_an:+.4f}", delta=f"{chg_pct_an:+.2f}%")
            qs3.metric("52-bar High", f"{df_an['High'].tail(52).max():.4f}")
            qs4.metric("52-bar Low",  f"{df_an['Low'].tail(52).min():.4f}")
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
            st.markdown("#### 📝 Human-Readable Market Summary")
            summary_text = generate_market_summary(df_an, ACTIVE_TICKER)
            st.markdown(summary_text)
            st.markdown("---")
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
# TAB 5 – OPTIMIZATION  (full auto, all strategies)
# ═════════════════════════════════════════════════════════════
with tab_opt:
    st.markdown('<div class="section-header">🎯 Auto Strategy Optimizer</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Enter your ticker, choose a strategy, set the target accuracy, and hit **Run**. "
        "The optimizer automatically searches ALL relevant parameter combinations for that strategy, "
        "tests every SL/Target pairing, and ranks results by win rate. No manual ranges needed."
    )

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        opt_ticker   = st.text_input("Ticker", ACTIVE_TICKER, key="opt_tick")
        opt_tf_sel   = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(tf), key="opt_tf")
    with oc2:
        opt_strategy = st.selectbox("Strategy", STRATEGIES, key="opt_strat")
        vp2 = TF_PERIOD_MAP.get(opt_tf_sel, ALL_PERIODS)
        opt_period   = st.selectbox("Period", vp2, index=min(3,len(vp2)-1), key="opt_per")
    with oc3:
        opt_target_wr = st.number_input("Target Win Rate (%)", min_value=30.0, max_value=99.0,
                                         value=60.0, step=1.0, format="%.1f", key="opt_wr")
        opt_min_trades = st.number_input("Min Trades Threshold", 5, 500, 10, key="opt_min_tr")
        opt_max_combos = st.number_input("Max Combinations to Test", 50, 2000, 400, key="opt_max_c")

    run_opt = st.button("🚀 Run Full Auto-Optimization", key="btn_opt")

    # ── STRATEGY PARAM SPACE (per strategy num) ──────────────
    def _opt_param_space(num: int) -> list[dict]:
        """Return list of param dicts to sweep for a given strategy number."""
        grids = []
        if num in (1, 21):   # EMA Crossover
            for f in [5,7,9,12,14,17]:
                for s in [15,18,21,26,30,34,50]:
                    if f < s: grids.append({"fast":f,"slow":s})
        elif num == 2:
            for f in [10,15,20,25]:
                for s in [35,40,50,55,60]:
                    if f < s: grids.append({"fast":f,"slow":s})
        elif num == 3:
            for e1 in [5,7,9,12]:
                for e2 in [15,18,21,26]:
                    for e3 in [34,40,50,55]:
                        if e1<e2<e3: grids.append({"e1":e1,"e2":e2,"e3":e3})
        elif num == 4:   # MACD
            for f in [8,10,12,14]:
                for s in [20,24,26,28]:
                    for sg in [7,9,11]:
                        if f<s: grids.append({"fast":f,"slow":s,"sig":sg})
        elif num == 5:   # RSI
            for p in [9,11,14,18,21]:
                for ob in [65,70,75]:
                    for os_ in [25,30,35]:
                        grids.append({"period":p,"overbought":ob,"oversold":os_})
        elif num in (6, 7):  # Bollinger
            for p in [15,18,20,25]:
                for std in [1.5,2.0,2.5]:
                    grids.append({"period":p,"std":std})
        elif num == 8:   # Supertrend
            for p in [7,8,10,12,14]:
                for m in [2.0,2.5,3.0,3.5,4.0]:
                    grids.append({"period":p,"mult":m})
        elif num == 9:   # ADX
            for p in [10,12,14,18]:
                for t in [20,25,30]:
                    grids.append({"period":p,"thresh":t})
        elif num in (10, 11):  # Stochastic
            for k in [9,11,14,18]:
                for d in [3,5]:
                    grids.append({"k":k,"d":d})
        elif num == 12:  # Ichimoku – structural, fewer variations
            grids = [{}]
        elif num == 13:  # VWAP
            grids = [{}]
        elif num == 14:  # Donchian
            for p in [10,15,20,25,30]:
                grids.append({"period":p})
        elif num == 15:  # Keltner
            for en in [15,18,20,25]:
                for m in [1.0,1.5,2.0]:
                    grids.append({"en":en,"mult":m})
        elif num == 16:  # CCI
            for p in [14,16,20,24]:
                grids.append({"period":p})
        elif num == 17:  # WPR
            for p in [10,12,14,18]:
                grids.append({"period":p})
        elif num == 18:  # Pivot
            grids = [{}]
        elif num == 19:  # Heikin Ashi
            for e in [14,18,21,26]:
                grids.append({"ema":e})
        elif num == 20:  # RSI+MACD
            for rp in [9,14,18]:
                grids.append({"rsi":rp})
        elif num == 22:  # ORB
            grids = [{}]
        elif num == 23:  # Inside Bar
            grids = [{}]
        elif num == 24:  # MACD Histogram
            for f in [8,10,12,14]:
                for s in [20,24,26,28]:
                    if f<s: grids.append({"fast":f,"slow":s})
        elif num == 25:  # Supertrend+RSI
            for p in [7,10,12]:
                for m in [2.5,3.0,3.5]:
                    for rp in [10,14,18]:
                        grids.append({"period":p,"mult":m,"rsi":rp})
        elif num == 26:  # VPT
            for p in [10,12,14,18]:
                grids.append({"period":p})
        elif num == 27:  # Swing Breakout
            for lb in [5,8,10,15,20]:
                grids.append({"lookback":lb})
        elif num == 28:  # EMA Ribbon
            grids = [{}]
        elif num == 29:  # Parabolic SAR
            for af0 in [0.01,0.02,0.03]:
                for afm in [0.15,0.20,0.25]:
                    grids.append({"af0":af0,"afmax":afm})
        elif num == 30:  # Dual Momentum
            for p1 in [5,7,10,12]:
                for p2 in [20,25,30,35]:
                    grids.append({"p1":p1,"p2":p2})
        else:
            grids = [{}]
        return grids if grids else [{}]

    if run_opt:
        st.session_state.opt_results = None
        with st.spinner(f"Fetching data for {opt_ticker}…"):
            df_opt = fetch_data(opt_ticker, opt_period, opt_tf_sel, force_fresh=True)

        if df_opt is None or df_opt.empty:
            st.error("Failed to fetch data. Check ticker / period / timeframe.")
        else:
            try:
                strat_num = int(opt_strategy.split(".")[0].strip())
            except:
                strat_num = 1

            param_space = _opt_param_space(strat_num)

            # SL / Target combos to test
            sl_combos  = ["Custom Points","ATR Based","Trailing SL (Fixed Points)",
                          "Previous Candle Low/High","Volatility Based","Auto (AI Managed)"]
            tgt_combos = ["Custom Points","ATR Based","Risk/Reward Based",
                          "Trailing Target (Fixed Points)","Auto (AI Managed)"]

            # SL/Target params variations
            sl_pt_vals  = [10,20,30,50]
            tgt_pt_vals = [20,40,60,100]
            atr_sl_vals = [1.5,2.0,2.5,3.0]
            atr_tg_vals = [2.0,3.0,4.0]
            rr_vals     = [1.5,2.0,2.5,3.0]

            # Build full combination list
            all_combos = []
            for p_dict in param_space:
                for sl_t in sl_combos:
                    for tgt_t in tgt_combos:
                        for sl_pts_ in sl_pt_vals:
                            for tgt_pts_ in tgt_pt_vals:
                                for atr_sl_ in atr_sl_vals:
                                    all_combos.append({
                                        "params"   : p_dict,
                                        "sl_type"  : sl_t,
                                        "tgt_type" : tgt_t,
                                        "sl_pts"   : sl_pts_,
                                        "tgt_pts"  : tgt_pts_,
                                        "atr_sl"   : atr_sl_,
                                        "atr_tgt"  : 3.0,
                                        "rr"       : 2.0,
                                    })

            # Cap and shuffle for diversity
            import random
            random.seed(42)
            random.shuffle(all_combos)
            max_c = int(opt_max_combos)
            all_combos = all_combos[:max_c]

            best_results = []
            prog      = st.progress(0)
            prog_text = st.empty()

            for idx_, combo in enumerate(all_combos):
                pct = (idx_+1)/len(all_combos)
                prog.progress(pct)
                prog_text.text(
                    f"Testing combo {idx_+1}/{len(all_combos)}  "
                    f"| SL: {combo['sl_type'][:18]}  "
                    f"| Tgt: {combo['tgt_type'][:18]}  "
                    f"| Params: {combo['params']}"
                )
                try:
                    sp_ = {
                        "sl_pts"       : combo["sl_pts"],
                        "atr_sl_mult"  : combo["atr_sl"],
                        "rr_target_pts": combo["tgt_pts"],
                        "rr_ratio"     : combo["rr"],
                        "ema_fast"     : combo["params"].get("fast",9),
                        "ema_slow"     : combo["params"].get("slow",21),
                        "swing_lb"     : 5,
                    }
                    tp_ = {
                        "tgt_pts"     : combo["tgt_pts"],
                        "atr_tgt_mult": combo["atr_tgt"],
                        "rr_sl_pts"   : combo["sl_pts"],
                        "rr_ratio"    : combo["rr"],
                        "swing_lb"    : 5,
                    }
                    tr_ = run_backtest(
                        df_opt, opt_strategy, combo["sl_type"], combo["tgt_type"],
                        sp_, tp_, allow_overlap=False, qty=1
                    )
                    mt_ = calc_metrics(tr_)

                    if mt_ and mt_.get("Total Trades",0) >= int(opt_min_trades):
                        best_results.append({
                            "Win Rate (%)"  : mt_["Win Rate (%)"],
                            "Profit Factor" : mt_["Profit Factor"],
                            "Total Trades"  : mt_["Total Trades"],
                            "Total P&L"     : mt_["Total P&L (pts×qty)"],
                            "Avg Win"       : mt_["Avg Win (pts×qty)"],
                            "Avg Loss"      : mt_["Avg Loss (pts×qty)"],
                            "Max Drawdown"  : mt_["Max Drawdown"],
                            "Sharpe"        : mt_["Sharpe Ratio"],
                            "Expectancy"    : mt_["Expectancy"],
                            "SL Type"       : combo["sl_type"],
                            "Target Type"   : combo["tgt_type"],
                            "SL Points"     : combo["sl_pts"],
                            "Target Points" : combo["tgt_pts"],
                            "ATR SL Mult"   : combo["atr_sl"],
                            "R:R Ratio"     : combo["rr"],
                            "Strategy Params": str(combo["params"]),
                        })
                except:
                    continue

            prog.empty(); prog_text.empty()

            if best_results:
                opt_df_res = pd.DataFrame(best_results).sort_values(
                    ["Win Rate (%)","Profit Factor"], ascending=False
                ).reset_index(drop=True)
                st.session_state.opt_results = opt_df_res
            else:
                st.warning("No valid combinations found. Try a longer period or lower min trades.")

    # ── Display results ──────────────────────────────────────
    opt_df = st.session_state.get("opt_results")
    if opt_df is not None and not opt_df.empty:
        st.markdown("---")
        st.success(f"✅ Optimization complete — {len(opt_df)} valid configurations found out of tested combos.")

        target_wr_val = st.session_state.get("opt_wr_val", 60.0)

        best = opt_df.iloc[0]
        beat_target = best["Win Rate (%)"] >= opt_target_wr

        if beat_target:
            st.success(f"🏆 Target win rate of {opt_target_wr:.1f}% ACHIEVED! Best result: {best['Win Rate (%)']:.1f}%")
        else:
            st.warning(f"⚠️ Best win rate is {best['Win Rate (%)']:.1f}% — below target of {opt_target_wr:.1f}%. "
                       f"Try a different period, timeframe, or lower the target.")

        st.markdown("### ⭐ Best Configuration")
        bc_cols = st.columns(5)
        bc_cols[0].metric("Win Rate",      f"{best['Win Rate (%)']:.1f}%")
        bc_cols[1].metric("Profit Factor", f"{best['Profit Factor']:.2f}")
        bc_cols[2].metric("Total Trades",  int(best["Total Trades"]))
        bc_cols[3].metric("Total P&L",     f"{best['Total P&L']:.2f}")
        bc_cols[4].metric("Sharpe",        f"{best['Sharpe']:.2f}")

        st.markdown("#### Best Parameters to Use in Live Trading")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| **Strategy** | {st.session_state.get("opt_strat","—")} |
| **SL Type** | {best['SL Type']} |
| **Target Type** | {best['Target Type']} |
| **SL Points** | {best['SL Points']} |
| **Target Points** | {best['Target Points']} |
| **ATR SL Multiplier** | {best['ATR SL Mult']} |
| **R:R Ratio** | {best['R:R Ratio']} |
| **Strategy Params** | `{best['Strategy Params']}` |
        """)
        st.info("💡 Apply these exact settings in the sidebar, then go to Live Trading or Backtesting with full confidence.")

        # Tabs for results exploration
        res_tabs = st.tabs(["Top 30 Results","Win Rate Distribution","Parameter Heatmap","Export"])
        with res_tabs[0]:
            st.dataframe(opt_df.head(30), use_container_width=True)

        with res_tabs[1]:
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Histogram(
                x=opt_df["Win Rate (%)"], nbinsx=30,
                marker_color="#000000", opacity=0.7, name="Win Rate Distribution"
            ))
            fig_wr.add_vline(x=opt_target_wr, line_dash="dash", line_color="#555",
                             annotation_text=f"Target {opt_target_wr:.0f}%")
            fig_wr.update_layout(
                title="Win Rate Distribution Across All Tested Combinations",
                height=350, template="plotly_white",
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                font=dict(color="#000"),
                xaxis_title="Win Rate (%)", yaxis_title="Count"
            )
            st.plotly_chart(fig_wr, use_container_width=True)

            # PF vs WR scatter
            fig_pf = go.Figure()
            fig_pf.add_trace(go.Scatter(
                x=opt_df["Win Rate (%)"],
                y=opt_df["Profit Factor"].clip(0,10),
                mode="markers",
                marker=dict(
                    color=opt_df["Total P&L"],
                    colorscale=[[0,"#cccccc"],[0.5,"#666"],[1,"#000"]],
                    size=8, colorbar=dict(title="P&L"),
                    showscale=True
                ),
                text=[f"WR:{r['Win Rate (%)']:.1f}%  PF:{r['Profit Factor']:.2f}<br>"
                      f"SL:{r['SL Type']}<br>Tgt:{r['Target Type']}"
                      for _,r in opt_df.iterrows()],
                hovertemplate="%{text}<extra></extra>",
                name="All Results"
            ))
            fig_pf.update_layout(
                title="Win Rate vs Profit Factor",
                height=350, template="plotly_white",
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                font=dict(color="#000"),
                xaxis_title="Win Rate (%)", yaxis_title="Profit Factor"
            )
            st.plotly_chart(fig_pf, use_container_width=True)

        with res_tabs[2]:
            st.markdown("#### Top SL Types by Average Win Rate")
            sl_grp = opt_df.groupby("SL Type")["Win Rate (%)"].mean().sort_values(ascending=False).reset_index()
            fig_sl = go.Figure(go.Bar(
                x=sl_grp["SL Type"], y=sl_grp["Win Rate (%)"],
                marker_color="#000000"
            ))
            fig_sl.update_layout(height=300, template="plotly_white",
                                  plot_bgcolor="#fff", paper_bgcolor="#fff",
                                  font=dict(color="#000"),
                                  xaxis_title="SL Type", yaxis_title="Avg Win Rate (%)")
            st.plotly_chart(fig_sl, use_container_width=True)

            st.markdown("#### Top Target Types by Average Win Rate")
            tgt_grp = opt_df.groupby("Target Type")["Win Rate (%)"].mean().sort_values(ascending=False).reset_index()
            fig_tgt = go.Figure(go.Bar(
                x=tgt_grp["Target Type"], y=tgt_grp["Win Rate (%)"],
                marker_color="#444444"
            ))
            fig_tgt.update_layout(height=300, template="plotly_white",
                                   plot_bgcolor="#fff", paper_bgcolor="#fff",
                                   font=dict(color="#000"))
            st.plotly_chart(fig_tgt, use_container_width=True)

        with res_tabs[3]:
            csv_data = opt_df.to_csv(index=False)
            st.download_button(
                "⬇ Download All Results as CSV",
                csv_data,
                f"optimization_{opt_strategy[:15].strip()}.csv",
                "text/csv",
                key="dl_opt_csv"
            )
            st.markdown("**Summary Statistics:**")
            st.dataframe(opt_df[["Win Rate (%)","Profit Factor","Total P&L",
                                  "Sharpe","Expectancy"]].describe().round(2),
                         use_container_width=True)

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
