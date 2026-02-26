"""
ProTrader AI — Professional Algorithmic Trading Platform
Built with Streamlit | Free APIs (yfinance) | No talib/pandas_ta
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dtime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time as time_module
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ProTrader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# THEME CSS
# ══════════════════════════════════════════════════════════════
def apply_theme(dark: bool):
    if dark:
        BG   = "#0d1117"; CARD = "#161b22"; TEXT = "#e6edf3"
        ACC  = "#00d4aa"; SEC  = "#21262d"; BDR  = "#30363d"
        UP   = "#00d4aa"; DN   = "#ff4d6d"
    else:
        BG   = "#f8fafc"; CARD = "#ffffff"; TEXT = "#1a1a2e"
        ACC  = "#0055ff"; SEC  = "#eef2f7"; BDR  = "#d1d9e0"
        UP   = "#059669"; DN   = "#dc2626"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@600;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── GLOBAL ── */
html, body, .stApp                {{ background-color:{BG} !important; color:{TEXT} !important; font-family:'DM Sans',sans-serif; }}
.stApp > header                   {{ background-color:{BG} !important; border-bottom:1px solid {BDR}; }}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"]                 {{ background:{CARD} !important; border-right:1px solid {BDR} !important; }}
section[data-testid="stSidebar"] *               {{ color:{TEXT} !important; }}
section[data-testid="stSidebar"] .stSelectbox > div > div {{ background:{SEC}; border:1px solid {BDR}; border-radius:8px; }}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{ background:{CARD}; border-radius:12px; padding:4px; border:1px solid {BDR}; gap:2px; }}
.stTabs [data-baseweb="tab"]     {{ color:{TEXT} !important; border-radius:8px; font-weight:500; padding:8px 20px; transition:all .2s; }}
.stTabs [aria-selected="true"]   {{ background:{ACC} !important; color:#fff !important; font-weight:600; }}

/* ── METRICS ── */
[data-testid="stMetricValue"]    {{ color:{ACC} !important; font-size:1.35rem !important; font-weight:700; font-family:'JetBrains Mono',monospace; }}
[data-testid="stMetricLabel"]    {{ color:{TEXT} !important; opacity:.75; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }}
[data-testid="stMetricDelta"]    {{ font-family:'JetBrains Mono',monospace; font-size:.8rem; }}

/* ── BUTTONS ── */
.stButton > button               {{ background:{ACC}; color:#fff; border:none; border-radius:8px; font-weight:600; padding:10px 24px; transition:all .2s; font-family:'DM Sans',sans-serif; }}
.stButton > button:hover         {{ opacity:.88; transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,212,170,.35); }}

/* ── INPUTS ── */
.stSelectbox > div > div, .stNumberInput > div > div > input,
.stTextInput > div > div > input, .stSlider {{ color:{TEXT} !important; }}

/* ── CARDS ── */
.pt-card  {{ background:{CARD}; border:1px solid {BDR}; border-radius:14px; padding:18px 22px; margin:8px 0; }}
.pt-signal-long  {{ background:linear-gradient(135deg,#052e1a,#0a4d28); border:2px solid {UP}; border-radius:14px; padding:22px; color:#fff; }}
.pt-signal-short {{ background:linear-gradient(135deg,#2e0511,#4d0a1a); border:2px solid {DN}; border-radius:14px; padding:22px; color:#fff; }}
.pt-signal-hold  {{ background:linear-gradient(135deg,#1c1a05,#33300a); border:2px solid #f59e0b; border-radius:14px; padding:22px; color:#fff; }}

/* ── COMMENTARY ── */
.pt-commentary {{ background:{SEC}; border-left:4px solid {ACC}; border-radius:0 10px 10px 0; padding:14px 18px; margin:6px 0;
                  color:{TEXT}; font-family:'JetBrains Mono',monospace; font-size:.82rem; line-height:1.7; white-space:pre-wrap; }}

/* ── DATAFRAME ── */
.stDataFrame  {{ background:{CARD}; border-radius:10px; border:1px solid {BDR}; }}
iframe        {{ border-radius:10px; }}

/* ── DIVIDER ── */
hr {{ border-color:{BDR} !important; }}

/* ── TEXT ── */
h1,h2,h3,h4,h5,h6,p,label,span,div {{ color:{TEXT}; }}
.stMarkdown p  {{ color:{TEXT} !important; }}

/* ── SCROLLBAR ── */
::-webkit-scrollbar       {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{ACC}; border-radius:4px; }}

/* ── BADGE ── */
.badge {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; }}
.badge-up {{ background:{'rgba(0,212,170,.18)' if dark else 'rgba(5,150,105,.1)'}; color:{UP}; border:1px solid {UP}; }}
.badge-dn {{ background:{'rgba(255,77,109,.18)' if dark else 'rgba(220,38,38,.1)'}; color:{DN}; border:1px solid {DN}; }}
.badge-nt {{ background:rgba(245,158,11,.18); color:#f59e0b; border:1px solid #f59e0b; }}

/* ── LOGO ── */
.pt-logo {{ font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem;
            background:linear-gradient(135deg,{ACC},{ACC}aa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
TICKERS = {
    "NIFTY 50":     "^NSEI",
    "BANK NIFTY":   "^NSEBANK",
    "SENSEX":       "^BSESN",
    "BTC/USD":      "BTC-USD",
    "ETH/USD":      "ETH-USD",
    "USD/INR":      "INR=X",
    "GOLD":         "GC=F",
    "SILVER":       "SI=F",
    "CRUDE OIL":    "CL=F",
    "EUR/USD":      "EURUSD=X",
    "GBP/USD":      "GBPUSD=X",
    "USD/JPY":      "JPY=X",
    "APPLE":        "AAPL",
    "RELIANCE":     "RELIANCE.NS",
    "TCS":          "TCS.NS",
    "HDFC BANK":    "HDFCBANK.NS",
    "⚙ Custom":     "__custom__",
}

TIMEFRAMES = {
    "1 Minute":   "1m",
    "5 Minutes":  "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour":     "1h",
    "4 Hours":    "4h",
    "1 Day":      "1d",
    "1 Week":     "1wk",
}

PERIODS = {
    "1 Day":    "1d",
    "5 Days":   "5d",
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
    "2 Years":  "2y",
}

REC_TF = {
    "Scalping":    ("5 Minutes",  "5 Days"),
    "Intraday":    ("15 Minutes", "1 Month"),
    "Swing":       ("1 Hour",     "3 Months"),
    "Positional":  ("1 Day",      "1 Year"),
}

SL_TYPES = {
    "ATR-Based (Dynamic)":      "atr",
    "Fixed %":                  "pct",
    "Swing High/Low":           "swing",
    "Volatility (2×ATR)":       "vol",
    "Previous Candle H/L":      "candle",
}

TARGET_TYPES = {
    "ATR Multiple":             "atr",
    "Risk:Reward Ratio":        "rr",
    "Pivot Points (R1/S1)":     "pivot",
    "Bollinger Band Extreme":   "bb",
    "Fixed %":                  "pct",
}

STRATEGY_DESC = {
    "Scalping":   "EMA9/21 cross + RSI + Stochastic + Volume surge | Best on 1m–5m | Tight SLs",
    "Intraday":   "VWAP deviation + EMA21/50 + MACD + Bollinger Bands | Best on 15m–30m",
    "Swing":      "EMA21/50 Golden/Death Cross + ADX trend strength + Supertrend | Best on 1h–4h",
    "Positional": "Supertrend + EMA200 + ADX + MACD | Best on 1d–1wk | Wide SLs",
}


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "dark_mode":       False,   # default: light theme
        "trade_history":   [],
        "paper_position":  None,
        "paper_balance":   100_000.0,
        "commentary_log":  [],
        "auto_refresh":    False,
        "last_refresh":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# INDICATORS  (pure pandas/numpy — no talib)
# ══════════════════════════════════════════════════════════════
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _sma(s, n):
    return s.rolling(n).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _macd(s, f=12, sl=26, sig=9):
    m = _ema(s, f) - _ema(s, sl)
    si = _ema(m, sig)
    return m, si, m - si

def _boll(s, n=20, k=2):
    m = _sma(s, n)
    std = s.rolling(n).std()
    return m + k*std, m, m - k*std

def _atr(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n-1, min_periods=n).mean()

def _vwap(h, l, c, v):
    tp = (h + l + c) / 3
    return (tp * v).cumsum() / v.cumsum()

def _supertrend(h, l, c, n=10, mult=3.0):
    atr_v   = _atr(h, l, c, n).values
    hl2     = ((h + l) / 2).values
    c_arr   = c.values
    ub, lb  = hl2 + mult*atr_v, hl2 - mult*atr_v
    fub, flb = ub.copy(), lb.copy()
    trend = np.ones(len(c_arr))
    for i in range(1, len(c_arr)):
        fub[i] = ub[i] if (ub[i] < fub[i-1] or c_arr[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i] > flb[i-1] or c_arr[i-1] < flb[i-1]) else flb[i-1]
        if trend[i-1] == -1:
            trend[i] = 1  if c_arr[i] > fub[i] else -1
        else:
            trend[i] = -1 if c_arr[i] < flb[i] else  1
    st_line = np.where(trend == 1, flb, fub)
    return pd.Series(st_line, index=c.index), pd.Series(trend, index=c.index)

def _adx(h, l, c, n=14):
    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    dmp = h.diff().clip(lower=0)
    dmn = (-l.diff()).clip(lower=0)
    dmp = dmp.where(dmp > dmn, 0.0)
    dmn = dmn.where(dmn > dmp, 0.0)
    atr14  = tr.ewm(com=n-1, min_periods=n).mean()
    dip    = 100 * dmp.ewm(com=n-1, min_periods=n).mean() / atr14
    din    = 100 * dmn.ewm(com=n-1, min_periods=n).mean() / atr14
    dx     = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
    return dx.ewm(com=n-1, min_periods=n).mean(), dip, din

def _stoch(h, l, c, k=14, d=3):
    lo, hi = l.rolling(k).min(), h.rolling(k).max()
    sk = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return sk, sk.rolling(d).mean()

def _obv(c, v):
    return (np.sign(c.diff()).fillna(0) * v).cumsum()

def compute_indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    v = df.get("Volume", pd.Series(np.ones(len(df)), index=df.index))
    I = {}
    I["ema9"]   = _ema(c, 9);   I["ema21"]  = _ema(c, 21)
    I["ema50"]  = _ema(c, 50);  I["ema200"] = _ema(c, 200)
    I["sma20"]  = _sma(c, 20)
    I["rsi"]    = _rsi(c, 14)
    I["macd"], I["macd_sig"], I["macd_hist"] = _macd(c)
    I["bb_up"], I["bb_mid"], I["bb_lo"]      = _boll(c)
    I["atr"]    = _atr(h, l, c, 14)
    try:
        I["vwap"] = _vwap(h, l, c, v)
    except Exception:
        I["vwap"] = c.copy()
    I["st_line"], I["st_dir"]   = _supertrend(h, l, c)
    I["adx"], I["dip"], I["din"] = _adx(h, l, c)
    I["stoch_k"], I["stoch_d"]  = _stoch(h, l, c)
    I["obv"]    = _obv(c, v)
    pivot       = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    I["pivot"]  = pivot
    I["r1"]     = 2*pivot - l.shift(1)
    I["s1"]     = 2*pivot - h.shift(1)
    I["r2"]     = pivot + (h.shift(1) - l.shift(1))
    I["s2"]     = pivot - (h.shift(1) - l.shift(1))
    return I


# ══════════════════════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════════════════════
def _score_to_signal(score, threshold=50):
    if score >= threshold:   return "LONG"
    if score <= -threshold:  return "SHORT"
    return "HOLD"

def _strat_scalping(df, I):
    c = df["Close"]; v = df.get("Volume", pd.Series(np.ones(len(df)), index=df.index))
    e9, e21 = I["ema9"].values, I["ema21"].values
    r, sk, sd = I["rsi"].values, I["stoch_k"].values, I["stoch_d"].values
    vol, vol_ma = v.values, v.rolling(20).mean().values
    sigs, scores, reasons = (pd.Series("HOLD", index=df.index),
                             pd.Series(0.0, index=df.index), pd.Series("", index=df.index))
    for i in range(21, len(df)):
        sc, rs = 0, []
        # EMA cross
        if e9[i] > e21[i] and e9[i-1] <= e21[i-1]: sc += 40; rs.append("EMA9 crossed ↑ EMA21 (fresh bull cross)")
        elif e9[i] < e21[i] and e9[i-1] >= e21[i-1]: sc -= 40; rs.append("EMA9 crossed ↓ EMA21 (fresh bear cross)")
        elif e9[i] > e21[i]: sc += 15; rs.append("EMA9>EMA21 (uptrend)")
        else: sc -= 15; rs.append("EMA9<EMA21 (downtrend)")
        # RSI
        if not np.isnan(r[i]):
            if r[i] < 30: sc += 25; rs.append(f"RSI={r[i]:.1f} oversold")
            elif r[i] > 70: sc -= 25; rs.append(f"RSI={r[i]:.1f} overbought")
        # Stochastic
        if not (np.isnan(sk[i]) or np.isnan(sd[i])):
            if sk[i] < 20 and sd[i] < 20: sc += 20; rs.append("Stoch oversold")
            elif sk[i] > 80 and sd[i] > 80: sc -= 20; rs.append("Stoch overbought")
            if sk[i] > sd[i] and sk[i-1] <= sd[i-1]: sc += 15; rs.append("Stoch K↑D (buy)")
            elif sk[i] < sd[i] and sk[i-1] >= sd[i-1]: sc -= 15; rs.append("Stoch K↓D (sell)")
        # Volume
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            if vol[i] > 1.5*vol_ma[i]:
                addon = 10 if sc > 0 else -10
                sc += addon; rs.append("Volume surge confirms move")
        scores.iloc[i] = np.clip(sc, -100, 100)
        reasons.iloc[i] = " | ".join(rs)
        sigs.iloc[i] = _score_to_signal(sc, 50)
    return sigs, scores, reasons

def _strat_intraday(df, I):
    c = df["Close"]
    vw, e21, e50 = I["vwap"].values, I["ema21"].values, I["ema50"].values
    m, ms, mh = I["macd"].values, I["macd_sig"].values, I["macd_hist"].values
    r, bbu, bbl = I["rsi"].values, I["bb_up"].values, I["bb_lo"].values
    cv = c.values
    sigs, scores, reasons = (pd.Series("HOLD", index=df.index),
                             pd.Series(0.0, index=df.index), pd.Series("", index=df.index))
    for i in range(50, len(df)):
        sc, rs = 0, []
        # VWAP
        if not np.isnan(vw[i]):
            if cv[i] > vw[i]*1.001: sc += 25; rs.append(f"Price above VWAP (bull bias)")
            elif cv[i] < vw[i]*0.999: sc -= 25; rs.append(f"Price below VWAP (bear bias)")
        # EMA alignment
        if e21[i] > e50[i]: sc += 15; rs.append("EMA21>EMA50 uptrend")
        else: sc -= 15; rs.append("EMA21<EMA50 downtrend")
        # MACD
        if not (np.isnan(m[i]) or np.isnan(ms[i])):
            if m[i] > ms[i] and m[i-1] <= ms[i-1]: sc += 30; rs.append("MACD bull crossover")
            elif m[i] < ms[i] and m[i-1] >= ms[i-1]: sc -= 30; rs.append("MACD bear crossover")
            elif m[i] > ms[i]: sc += 10; rs.append("MACD bullish")
            else: sc -= 10; rs.append("MACD bearish")
            if mh[i] > 0 and mh[i] > mh[i-1]: sc += 8; rs.append("Histogram expanding ↑")
            elif mh[i] < 0 and mh[i] < mh[i-1]: sc -= 8; rs.append("Histogram expanding ↓")
        # RSI
        if not np.isnan(r[i]):
            if r[i] < 35: sc += 18; rs.append(f"RSI={r[i]:.1f} oversold")
            elif r[i] > 65: sc -= 18; rs.append(f"RSI={r[i]:.1f} overbought")
        # Bollinger
        if not np.isnan(bbl[i]):
            if cv[i] <= bbl[i]: sc += 14; rs.append("Price at BB lower (bounce zone)")
            elif cv[i] >= bbu[i]: sc -= 14; rs.append("Price at BB upper (reversal zone)")
        scores.iloc[i] = np.clip(sc, -100, 100)
        reasons.iloc[i] = " | ".join(rs)
        sigs.iloc[i] = _score_to_signal(sc, 55)
    return sigs, scores, reasons

def _strat_swing(df, I):
    c = df["Close"]
    e21, e50, e200 = I["ema21"].values, I["ema50"].values, I["ema200"].values
    m, ms = I["macd"].values, I["macd_sig"].values
    r, adx_v = I["rsi"].values, I["adx"].values
    dip, din = I["dip"].values, I["din"].values
    std = I["st_dir"].values; cv = c.values
    sigs, scores, reasons = (pd.Series("HOLD", index=df.index),
                             pd.Series(0.0, index=df.index), pd.Series("", index=df.index))
    for i in range(201, len(df)):
        sc, rs = 0, []
        # EMA200 context
        if cv[i] > e200[i]: sc += 12; rs.append("Above EMA200 (macro uptrend)")
        else: sc -= 12; rs.append("Below EMA200 (macro downtrend)")
        # EMA21/50 cross
        if e21[i] > e50[i] and e21[i-1] <= e50[i-1]: sc += 38; rs.append("🟡 Golden Cross: EMA21 above EMA50")
        elif e21[i] < e50[i] and e21[i-1] >= e50[i-1]: sc -= 38; rs.append("💀 Death Cross: EMA21 below EMA50")
        elif e21[i] > e50[i]: sc += 16; rs.append("EMA21>EMA50 (swing uptrend)")
        else: sc -= 16; rs.append("EMA21<EMA50 (swing downtrend)")
        # ADX
        if not np.isnan(adx_v[i]):
            mult = 1.5 if adx_v[i] > 25 else (1.0 if adx_v[i] > 20 else 0.4)
            if dip[i] > din[i]: sc += int(15*mult); rs.append(f"ADX={adx_v[i]:.0f} DI+ dominant (bull)")
            else: sc -= int(15*mult); rs.append(f"ADX={adx_v[i]:.0f} DI- dominant (bear)")
        # MACD
        if not np.isnan(m[i]):
            if m[i] > ms[i]: sc += 14; rs.append("MACD bullish")
            else: sc -= 14; rs.append("MACD bearish")
        # RSI momentum
        if not np.isnan(r[i]):
            if r[i] > 55: sc += 8; rs.append(f"RSI={r[i]:.0f} bullish momentum")
            elif r[i] < 45: sc -= 8; rs.append(f"RSI={r[i]:.0f} bearish momentum")
        # Supertrend
        if not np.isnan(std[i]):
            if std[i] == 1: sc += 14; rs.append("Supertrend BULLISH")
            else: sc -= 14; rs.append("Supertrend BEARISH")
        scores.iloc[i] = np.clip(sc, -100, 100)
        reasons.iloc[i] = " | ".join(rs)
        sigs.iloc[i] = _score_to_signal(sc, 60)
    return sigs, scores, reasons

def _strat_positional(df, I):
    c = df["Close"]
    e50, e200 = I["ema50"].values, I["ema200"].values
    stl, std = I["st_line"].values, I["st_dir"].values
    adx_v, dip, din = I["adx"].values, I["dip"].values, I["din"].values
    m, ms = I["macd"].values, I["macd_sig"].values
    r = I["rsi"].values; cv = c.values
    sigs, scores, reasons = (pd.Series("HOLD", index=df.index),
                             pd.Series(0.0, index=df.index), pd.Series("", index=df.index))
    for i in range(201, len(df)):
        sc, rs = 0, []
        # Supertrend — primary signal
        if not np.isnan(std[i]):
            if std[i] == 1: sc += 42; rs.append(f"Supertrend BULLISH (support at {stl[i]:.4f})")
            else: sc -= 42; rs.append(f"Supertrend BEARISH (resistance at {stl[i]:.4f})")
        # EMA200 major trend
        if cv[i] > e200[i]: sc += 22; rs.append(f"Above EMA200={e200[i]:.4f} (primary uptrend)")
        else: sc -= 22; rs.append(f"Below EMA200={e200[i]:.4f} (primary downtrend)")
        # ADX + directional
        if not np.isnan(adx_v[i]) and adx_v[i] > 18:
            if dip[i] > din[i]: sc += 18; rs.append(f"ADX={adx_v[i]:.0f} bullish direction")
            else: sc -= 18; rs.append(f"ADX={adx_v[i]:.0f} bearish direction")
        # MACD confirmation
        if not np.isnan(m[i]):
            if m[i] > ms[i]: sc += 12; rs.append("MACD confirms bull momentum")
            else: sc -= 12; rs.append("MACD confirms bear momentum")
        if not np.isnan(r[i]): sc += 6 if r[i] > 50 else -6
        scores.iloc[i] = np.clip(sc, -100, 100)
        reasons.iloc[i] = " | ".join(rs)
        sigs.iloc[i] = _score_to_signal(sc, 65)
    return sigs, scores, reasons

def run_strategy(df, strategy, I=None):
    if I is None: I = compute_indicators(df)
    fn = {"Scalping": _strat_scalping, "Intraday": _strat_intraday,
          "Swing": _strat_swing, "Positional": _strat_positional}[strategy]
    return fn(df, I)


# ══════════════════════════════════════════════════════════════
# SIGNAL BUILDER  (entry + all SL/target variants)
# ══════════════════════════════════════════════════════════════
def compute_sl_target(side, entry, atr_v, sl_type, tgt_type, sl_mult, t_mult, high, low, pivot, r1, s1, bb_up, bb_lo, pct=1.5):
    """Returns sl, t1, t2, t3, tsl_offset"""
    direction = 1 if side == "LONG" else -1

    # ── Stop Loss ──────────────────────────────────────────
    if sl_type == "atr":
        sl = entry - direction * sl_mult * atr_v
    elif sl_type == "pct":
        sl = entry * (1 - direction * pct/100)
    elif sl_type == "swing":
        sl = low - atr_v*0.2 if side == "LONG" else high + atr_v*0.2
    elif sl_type == "vol":
        sl = entry - direction * 2 * atr_v
    elif sl_type == "candle":
        sl = low if side == "LONG" else high
    else:
        sl = entry - direction * sl_mult * atr_v

    risk = abs(entry - sl) if abs(entry - sl) > 1e-9 else atr_v

    # ── Targets ────────────────────────────────────────────
    if tgt_type == "atr":
        t1 = entry + direction * t_mult * atr_v
        t2 = entry + direction * t_mult * 1.8 * atr_v
        t3 = entry + direction * t_mult * 3.0 * atr_v
    elif tgt_type == "rr":
        t1 = entry + direction * risk * t_mult
        t2 = entry + direction * risk * t_mult * 1.8
        t3 = entry + direction * risk * t_mult * 3.0
    elif tgt_type == "pivot" and r1 and s1:
        t1 = r1 if side == "LONG" else s1
        t2 = entry + direction * risk * 2.5
        t3 = entry + direction * risk * 4.0
    elif tgt_type == "bb":
        t1 = bb_up if side == "LONG" else bb_lo
        t2 = entry + direction * risk * 2.5
        t3 = entry + direction * risk * 4.0
    elif tgt_type == "pct":
        t1 = entry * (1 + direction * pct * t_mult / 100)
        t2 = entry * (1 + direction * pct * t_mult * 1.8 / 100)
        t3 = entry * (1 + direction * pct * t_mult * 3.0 / 100)
    else:
        t1 = entry + direction * t_mult * atr_v
        t2 = entry + direction * t_mult * 1.8 * atr_v
        t3 = entry + direction * t_mult * 3.0 * atr_v

    tsl_offset = sl_mult * atr_v
    return sl, t1, t2, t3, tsl_offset

def get_signal(df, strategy, sl_type="atr", tgt_type="atr", sl_mult=1.5, t_mult=2.0):
    if df is None or len(df) < 30: return None
    I   = compute_indicators(df)
    sigs, scores, reasons = run_strategy(df, strategy, I)
    i   = -1
    sig = sigs.iloc[i]; score = scores.iloc[i]; reason = reasons.iloc[i]
    c   = df["Close"].iloc[i]
    h   = df["High"].iloc[-5:].min();  l = df["Low"].iloc[-5:].max()
    h_c = df["High"].iloc[i]; l_c = df["Low"].iloc[i]
    atr_v  = I["atr"].iloc[i]
    if np.isnan(atr_v) or atr_v == 0: atr_v = c * 0.01
    r1_v = I["r1"].iloc[i]; s1_v = I["s1"].iloc[i]
    bbu  = I["bb_up"].iloc[i]; bbl = I["bb_lo"].iloc[i]
    pivot_v = I["pivot"].iloc[i]
    sl, t1, t2, t3, tsl = compute_sl_target(
        sig, c, atr_v, sl_type, tgt_type, sl_mult, t_mult,
        h_c, l_c, pivot_v, r1_v, s1_v, bbu, bbl)
    rr = abs(t1 - c) / max(abs(c - sl), 1e-9)
    return {
        "signal":     sig,
        "confidence": float(min(abs(score), 100)),
        "score":      float(score),
        "entry":      c,
        "sl":         sl, "t1": t1, "t2": t2, "t3": t3,
        "tsl_offset": tsl,
        "atr":        atr_v,
        "reason":     reason,
        "rr":         rr,
        "rsi":        float(I["rsi"].iloc[i]),
        "macd_v":     float(I["macd"].iloc[i]),
        "adx_v":      float(I["adx"].iloc[i]),
        "st_dir":     float(I["st_dir"].iloc[i]),
        "ema21":      float(I["ema21"].iloc[i]),
        "ema50":      float(I["ema50"].iloc[i]),
        "vwap_v":     float(I["vwap"].iloc[i]),
        "r1":         float(r1_v) if not np.isnan(r1_v) else None,
        "s1":         float(s1_v) if not np.isnan(s1_v) else None,
        "dip":        float(I["dip"].iloc[i]),
        "din":        float(I["din"].iloc[i]),
        "bb_up":      float(bbu),
        "bb_lo":      float(bbl),
        "obv_v":      float(I["obv"].iloc[i]),
    }


# ══════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ══════════════════════════════════════════════════════════════
def run_backtest(df, strategy, sl_mult=1.5, t_mult=2.0, sl_type="atr", tgt_type="atr",
                 use_window=False, win_start=None, win_end=None, capital=100_000, qty=1):
    I = compute_indicators(df)
    sigs, scores, reasons = run_strategy(df, strategy, I)
    c_arr   = df["Close"].values
    h_arr   = df["High"].values
    l_arr   = df["Low"].values
    atr_arr = I["atr"].values
    r1_arr  = I["r1"].values
    s1_arr  = I["s1"].values
    bbu_arr = I["bb_up"].values
    bbl_arr = I["bb_lo"].values
    piv_arr = I["pivot"].values
    idx     = df.index

    trades = []
    equity = [capital]
    bal    = capital
    peak   = capital
    max_dd = 0.0

    in_trade = False; side = None; entry = 0.0; sl = 0.0; target = 0.0
    tsl_off  = 0.0;  best_price = 0.0; t_entry = None; t_reason = ""

    for i in range(1, len(df)):
        # ── Trading window ──────────────────────────────
        if use_window and win_start and win_end:
            ts = pd.Timestamp(idx[i])
            if hasattr(ts, "time"):
                t = ts.time()
                in_window = win_start <= t <= win_end
                if not in_window:
                    if in_trade:
                        pnl = (c_arr[i] - entry)*qty if side == "LONG" else (entry - c_arr[i])*qty
                        bal += pnl; trades.append(_trade_rec(t_entry, idx[i], side, entry, c_arr[i], pnl, "Window Close", t_reason))
                        in_trade = False
                    equity.append(bal); peak = max(peak, bal); max_dd = max(max_dd, (peak-bal)/peak*100)
                    continue

        sig   = sigs.iloc[i]; score = abs(scores.iloc[i])
        atr_v = atr_arr[i] if not np.isnan(atr_arr[i]) else c_arr[i]*0.01
        r1_v  = r1_arr[i] if not np.isnan(r1_arr[i]) else None
        s1_v  = s1_arr[i] if not np.isnan(s1_arr[i]) else None

        # ── Manage open trade ────────────────────────────
        if in_trade:
            if side == "LONG":
                best_price = max(best_price, h_arr[i])
                new_tsl    = best_price - tsl_off
                sl         = max(sl, new_tsl)      # trail up only
                if l_arr[i] <= sl:
                    pnl = (sl - entry)*qty; bal += pnl
                    trades.append(_trade_rec(t_entry, idx[i], "LONG", entry, sl, pnl, "Stop Loss", t_reason))
                    in_trade = False
                elif h_arr[i] >= target:
                    pnl = (target - entry)*qty; bal += pnl
                    trades.append(_trade_rec(t_entry, idx[i], "LONG", entry, target, pnl, "Target Hit", t_reason))
                    in_trade = False
                elif sig == "SHORT" and score >= 55:
                    pnl = (c_arr[i] - entry)*qty; bal += pnl
                    trades.append(_trade_rec(t_entry, idx[i], "LONG", entry, c_arr[i], pnl, "Signal Flip", t_reason))
                    in_trade = False
            else:
                best_price = min(best_price, l_arr[i])
                new_tsl    = best_price + tsl_off
                sl         = min(sl, new_tsl)      # trail down only
                if h_arr[i] >= sl:
                    pnl = (entry - sl)*qty; bal += pnl
                    trades.append(_trade_rec(t_entry, idx[i], "SHORT", entry, sl, pnl, "Stop Loss", t_reason))
                    in_trade = False
                elif l_arr[i] <= target:
                    pnl = (entry - target)*qty; bal += pnl
                    trades.append(_trade_rec(t_entry, idx[i], "SHORT", entry, target, pnl, "Target Hit", t_reason))
                    in_trade = False
                elif sig == "LONG" and score >= 55:
                    pnl = (entry - c_arr[i])*qty; bal += pnl
                    trades.append(_trade_rec(t_entry, idx[i], "SHORT", entry, c_arr[i], pnl, "Signal Flip", t_reason))
                    in_trade = False
            peak   = max(peak, bal); max_dd = max(max_dd, (peak-bal)/peak*100)

        # ── Enter new trade ──────────────────────────────
        if not in_trade and sig in ("LONG","SHORT") and score >= 50:
            sl_c, t1_c, _, t3_c, tsl_c = compute_sl_target(
                sig, c_arr[i], atr_v, sl_type, tgt_type, sl_mult, t_mult,
                h_arr[i], l_arr[i], piv_arr[i], r1_v, s1_v, bbu_arr[i], bbl_arr[i])
            in_trade   = True; side  = sig; entry = c_arr[i]
            sl         = sl_c; target = t3_c; tsl_off = tsl_c
            best_price = entry if sig == "LONG" else entry
            t_entry    = idx[i]; t_reason = reasons.iloc[i]

        equity.append(bal)

    # ── Force close at end ──────────────────────────────
    if in_trade:
        pnl = (c_arr[-1]-entry)*qty if side=="LONG" else (entry-c_arr[-1])*qty
        bal += pnl
        trades.append(_trade_rec(t_entry, idx[-1], side, entry, c_arr[-1], pnl, "End of Data", t_reason))
        equity.append(bal)

    if not trades:
        return None, pd.DataFrame(), []

    tf = pd.DataFrame(trades)
    w  = tf[tf["pnl"] > 0]; lo = tf[tf["pnl"] <= 0]
    stats = {
        "trades":   len(tf),
        "win_rate": len(w)/len(tf)*100,
        "total_pnl":  tf["pnl"].sum(),
        "return_pct": tf["pnl"].sum()/capital*100,
        "avg_win":  w["pnl"].mean() if len(w) else 0,
        "avg_loss": lo["pnl"].mean() if len(lo) else 0,
        "pf":       abs(w["pnl"].sum()/lo["pnl"].sum()) if lo["pnl"].sum() != 0 else 999,
        "max_dd":   max_dd,
        "final":    bal,
        "sharpe":   tf["pnl"].mean()/(tf["pnl"].std()+1e-9)*np.sqrt(252) if len(tf)>1 else 0,
        "expectancy": tf["pnl"].mean(),
    }
    return stats, tf, equity

def _trade_rec(et, xt, side, entry, exit_, pnl, reason, t_reason):
    return {"entry_time": et, "exit_time": xt, "side": side,
            "entry": entry, "exit": exit_, "pnl": pnl,
            "exit_reason": reason, "analysis": t_reason}


# ══════════════════════════════════════════════════════════════
# DATA FETCHER
# ══════════════════════════════════════════════════════════════
_FETCH_MIN_GAP = 1.5   # minimum seconds between any yfinance requests

def _safe_download(ticker, period, interval, retries=3):
    """
    Wraps yf.download with:
      • 1.5 s minimum gap between calls (rate-limit friendly)
      • exponential back-off on failure (1.5 s → 3 s → 6 s)
      • up to `retries` attempts before returning None
    """
    now = time_module.time()
    last = st.session_state.get("_last_yf_call", 0)
    wait = _FETCH_MIN_GAP - (now - last)
    if wait > 0:
        time_module.sleep(wait)

    delay = _FETCH_MIN_GAP
    for attempt in range(retries):
        try:
            st.session_state["_last_yf_call"] = time_module.time()
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return df
        except Exception:
            if attempt < retries - 1:
                time_module.sleep(delay)
                delay *= 2          # exponential back-off
    return None


@st.cache_data(ttl=120)
def fetch(ticker, period, interval):
    return _safe_download(ticker, period, interval)


@st.cache_data(ttl=45)
def fetch_live(ticker, period, interval):
    return _safe_download(ticker, period, interval)



# ══════════════════════════════════════════════════════════════
# COLOR HELPERS
# ══════════════════════════════════════════════════════════════
def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex string to rgba(r,g,b,alpha) — Plotly-safe."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════
def build_chart(df, I, sig_data=None, dark=True, height=800):
    BG   = "#0d1117" if dark else "#f8fafc"
    GRID = "#1e2633" if dark else "#e2e8f0"
    TEXT = "#e6edf3" if dark else "#1a1a2e"
    UP   = "#00d4aa"; DN = "#ff4d6d"

    rows    = [0.50, 0.18, 0.15, 0.17]
    fig     = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=rows,
                            vertical_spacing=0.025,
                            subplot_titles=["Price", "MACD", "RSI", "Volume / OBV"])
    c_idx   = df.index

    # ── Candlestick ─────────────────────────────────────
    fig.add_trace(go.Candlestick(x=c_idx, open=df["Open"], high=df["High"],
                                  low=df["Low"], close=df["Close"], name="Price",
                                  increasing=dict(line_color=UP, fillcolor="rgba(0,212,170,0.55)"),
                                  decreasing=dict(line_color=DN, fillcolor="rgba(255,77,109,0.55)")), row=1, col=1)

    # ── EMAs ────────────────────────────────────────────
    for ema_p, col, nm in [(21,"#FFD700","EMA21"),(50,"#FF8C00","EMA50"),(200,"#A78BFA","EMA200")]:
        k = f"ema{ema_p}"
        if k in I:
            fig.add_trace(go.Scatter(x=c_idx, y=I[k], name=nm, line=dict(color=col,width=1), opacity=0.85), row=1, col=1)

    # ── VWAP ────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=c_idx, y=I["vwap"], name="VWAP",
                              line=dict(color="#00BFFF",width=1.5,dash="dash"), opacity=0.9), row=1, col=1)

    # ── Bollinger ───────────────────────────────────────
    fig.add_trace(go.Scatter(x=c_idx, y=I["bb_up"], name="BB Upper",
                              line=dict(color="rgba(156,163,175,0.45)",width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=c_idx, y=I["bb_lo"], name="BB Lower",
                              line=dict(color="rgba(156,163,175,0.45)",width=1),
                              fill="tonexty", fillcolor="rgba(156,163,175,0.06)", showlegend=False), row=1, col=1)

    # ── Supertrend ──────────────────────────────────────
    st_b = I["st_line"].where(I["st_dir"] == 1)
    st_r = I["st_line"].where(I["st_dir"] == -1)
    fig.add_trace(go.Scatter(x=c_idx, y=st_b, name="ST Bull", line=dict(color=UP,width=2.2), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=c_idx, y=st_r, name="ST Bear", line=dict(color=DN,width=2.2), mode="lines"), row=1, col=1)

    # ── Signal markers & levels ─────────────────────────
    if sig_data and sig_data["signal"] != "HOLD":
        lt = c_idx[-1]; lc = df["Close"].iloc[-1]
        mc = UP if sig_data["signal"]=="LONG" else DN
        ms = "triangle-up" if sig_data["signal"]=="LONG" else "triangle-down"
        fig.add_trace(go.Scatter(x=[lt], y=[lc], mode="markers",
                                  marker=dict(symbol=ms,size=16,color=mc,line=dict(color="#fff",width=1.5)),
                                  name=f"▶ {sig_data['signal']}"), row=1, col=1)
        lvls = [("SL",sig_data["sl"],"#ff4d6d","dash"),
                ("T1",sig_data["t1"],"#00d4aa","dot"),
                ("T2",sig_data["t2"],"#4ade80","dot"),
                ("T3",sig_data["t3"],"#86efac","dot")]
        for lbl, val, lc2, ld in lvls:
            if val: fig.add_hline(y=val, line_dash=ld, line_color=lc2,
                                    annotation_text=f" {lbl}:{val:.4f}",
                                    annotation_font_color=lc2, row=1, col=1)

    # ── MACD ────────────────────────────────────────────
    hist_col = [UP if v >= 0 else DN for v in I["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=c_idx, y=I["macd_hist"], name="Histogram",
                          marker_color=hist_col, opacity=0.75), row=2, col=1)
    fig.add_trace(go.Scatter(x=c_idx, y=I["macd"],     name="MACD",   line=dict(color="#FFD700",width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=c_idx, y=I["macd_sig"], name="Signal", line=dict(color="#FF6B35",width=1.5)), row=2, col=1)

    # ── RSI ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=c_idx, y=I["rsi"], name="RSI", line=dict(color="#A78BFA",width=1.8)), row=3, col=1)
    for lvl, col in [(70,"#ff4d6d"),(50,"#94a3b8"),(30,"#00d4aa")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, line_width=0.8, row=3, col=1)

    # ── Volume + OBV ────────────────────────────────────
    if "Volume" in df.columns:
        vcol = [UP if c >= o else DN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=c_idx, y=df["Volume"], name="Volume",
                              marker_color=vcol, opacity=0.55), row=4, col=1)
        vol_ma = df["Volume"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=c_idx, y=vol_ma, name="Vol MA20",
                                  line=dict(color="#FFD700",width=1.3)), row=4, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="DM Sans"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor=hex_rgba(BG, 0.80), bordercolor=GRID, borderwidth=1,
                    font=dict(color=TEXT, size=11), orientation="h",
                    x=0, y=1.02, xanchor="left"),
        margin=dict(l=60, r=60, t=50, b=20), height=height,
    )
    for r in range(1, 5):
        fig.update_xaxes(row=r, col=1, gridcolor=GRID, showgrid=True, zeroline=False,
                          showspikes=True, spikecolor=TEXT, spikethickness=1)
        fig.update_yaxes(row=r, col=1, gridcolor=GRID, showgrid=True, zeroline=False)
    return fig


# ══════════════════════════════════════════════════════════════
# COMMENTARY GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_commentary(sig_data, strategy, ticker):
    ts   = datetime.now().strftime("%H:%M:%S")
    sig  = sig_data["signal"]; conf = sig_data["confidence"]
    px   = sig_data["entry"];  rsi  = sig_data["rsi"]
    adx  = sig_data["adx_v"]; macd = sig_data["macd_v"]
    e21  = sig_data["ema21"]; e50  = sig_data["ema50"]
    vw   = sig_data["vwap_v"]; std  = sig_data["st_dir"]
    dip  = sig_data["dip"];   din  = sig_data["din"]

    L = []
    L.append(f"[{ts}] ══════════ MARKET SCAN ══════════")
    L.append(f"[{ts}] Asset: {ticker} | Strategy: {strategy} | Price: {px:.5f}")

    # Trend
    if e21 > e50: L.append(f"[{ts}] 📈 TREND BIAS: BULLISH  (EMA21={e21:.4f} > EMA50={e50:.4f})")
    else:          L.append(f"[{ts}] 📉 TREND BIAS: BEARISH  (EMA21={e21:.4f} < EMA50={e50:.4f})")

    # RSI
    if   rsi < 30: L.append(f"[{ts}] 🔴 RSI={rsi:.1f}  ⚡ OVERSOLD — strong bounce potential")
    elif rsi > 70: L.append(f"[{ts}] 🟡 RSI={rsi:.1f}  ⚡ OVERBOUGHT — profit-taking zone")
    else:           L.append(f"[{ts}] ⚪ RSI={rsi:.1f}  Neutral momentum")

    # ADX
    if   adx > 30: L.append(f"[{ts}] 💪 ADX={adx:.1f}  STRONG directional trend — follow signals")
    elif adx > 20: L.append(f"[{ts}] 👍 ADX={adx:.1f}  Moderate trend — signals reliable")
    else:           L.append(f"[{ts}] ⚠️  ADX={adx:.1f}  WEAK/RANGING — filter trades carefully")

    # DI
    if dip > din: L.append(f"[{ts}] DI+ ({dip:.1f}) > DI- ({din:.1f}) → Bullish directional edge")
    else:          L.append(f"[{ts}] DI- ({din:.1f}) > DI+ ({dip:.1f}) → Bearish directional edge")

    # VWAP
    if px > vw: L.append(f"[{ts}] ✅ Price ABOVE VWAP ({vw:.4f}) → institutional buy zone")
    else:        L.append(f"[{ts}] ❌ Price BELOW VWAP ({vw:.4f}) → institutional sell zone")

    # Supertrend
    if std == 1:  L.append(f"[{ts}] 🟢 SUPERTREND: BULLISH (price above trend line)")
    elif std == -1: L.append(f"[{ts}] 🔴 SUPERTREND: BEARISH (price below trend line)")

    # MACD
    if macd > 0: L.append(f"[{ts}] 📊 MACD positive ({macd:.4f}) → bullish momentum building")
    else:         L.append(f"[{ts}] 📊 MACD negative ({macd:.4f}) → bearish momentum building")

    L.append(f"[{ts}] ────────────────────────────────────")

    # Signal action
    if sig == "LONG":
        rr = sig_data["rr"]
        L.append(f"[{ts}] 🟢 SIGNAL: LONG  |  Confidence: {conf:.0f}%  |  R:R = 1:{rr:.2f}")
        L.append(f"[{ts}] 📍 Entry:  {px:.5f}")
        L.append(f"[{ts}] 🛑 Stop Loss:  {sig_data['sl']:.5f}  (risk = {abs(px-sig_data['sl']):.5f})")
        L.append(f"[{ts}] 🎯 Target 1:  {sig_data['t1']:.5f}  (+{abs(sig_data['t1']-px):.5f})")
        L.append(f"[{ts}] 🎯 Target 2:  {sig_data['t2']:.5f}")
        L.append(f"[{ts}] 🎯 Target 3:  {sig_data['t3']:.5f}  (full runner)")
        L.append(f"[{ts}] 🔒 Trailing SL offset: {sig_data['tsl_offset']:.5f} below peak")
        L.append(f"[{ts}] ► ACTION: ENTER LONG | Partial exit at T1, trail to T3")
    elif sig == "SHORT":
        rr = sig_data["rr"]
        L.append(f"[{ts}] 🔴 SIGNAL: SHORT  |  Confidence: {conf:.0f}%  |  R:R = 1:{rr:.2f}")
        L.append(f"[{ts}] 📍 Entry:  {px:.5f}")
        L.append(f"[{ts}] 🛑 Stop Loss:  {sig_data['sl']:.5f}")
        L.append(f"[{ts}] 🎯 Target 1:  {sig_data['t1']:.5f}")
        L.append(f"[{ts}] 🎯 Target 2:  {sig_data['t2']:.5f}")
        L.append(f"[{ts}] 🎯 Target 3:  {sig_data['t3']:.5f}")
        L.append(f"[{ts}] 🔒 Trailing SL offset: {sig_data['tsl_offset']:.5f} above trough")
        L.append(f"[{ts}] ► ACTION: ENTER SHORT | Partial exit at T1, trail to T3")
    else:
        L.append(f"[{ts}] ⏸️  SIGNAL: HOLD  |  Confidence: {conf:.0f}%")
        L.append(f"[{ts}] ► ACTION: WAIT — no high-probability setup. Monitor for breakout.")
        L.append(f"[{ts}]   Watch: price vs VWAP and EMA alignment for next signal.")

    L.append(f"[{ts}] ════════════════════════════════════")

    # Reasoning
    if sig_data.get("reason"):
        L.append(f"[{ts}] BREAKDOWN:")
        for r in sig_data["reason"].split(" | ")[:6]:
            if r.strip(): L.append(f"[{ts}]   → {r}")
    return L


# ══════════════════════════════════════════════════════════════
# ANALYSIS SUMMARY
# ══════════════════════════════════════════════════════════════
def analysis_summary(df, I, sig, ticker, strategy, tf_label, period_label):
    if sig is None: return "Insufficient data."
    px      = sig["entry"]
    chg_pct = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
    atr_pct = sig["atr"] / px * 100
    trend   = "UPTREND" if sig["ema21"] > sig["ema50"] else "DOWNTREND"
    vol_str = ""
    if "Volume" in df.columns:
        v_now, v_avg = df["Volume"].iloc[-1], df["Volume"].rolling(20).mean().iloc[-1]
        if   v_now > 1.5*v_avg: vol_str = "🔴 ABOVE AVERAGE — Volume surge (conviction move)"
        elif v_now < 0.7*v_avg: vol_str = "🟡 LOW — below average volume (weak conviction)"
        else:                    vol_str = "⚪ AVERAGE — normal volume"
    rr_str  = f"1:{sig['rr']:.2f}" if sig["signal"] != "HOLD" else "N/A"
    reasons_fmt = "\n".join(f"  • {r}" for r in sig["reason"].split(" | ") if r.strip()) if sig.get("reason") else "N/A"

    return f"""
## 📊 Technical Analysis — {ticker}

**Strategy:** {strategy} &nbsp;|&nbsp; **Timeframe:** {tf_label} &nbsp;|&nbsp; **Period:** {period_label}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

### 🔍 Market Overview
Current price **{px:.5f}** ({chg_pct:+.2f}% last candle). Primary structure: **{trend}**.  
EMA21 = {sig["ema21"]:.4f} | EMA50 = {sig["ema50"]:.4f} | EMA200 context: {"above" if px > sig.get("ema50",px) else "below"} medium-term average.

### 📈 Trend & Momentum
| Indicator | Value | Interpretation |
|-----------|-------|----------------|
| **RSI (14)** | {sig["rsi"]:.1f} | {"Oversold — bounce likely" if sig["rsi"]<30 else "Overbought — reversal risk" if sig["rsi"]>70 else "Neutral zone"} |
| **MACD** | {sig["macd_v"]:.4f} | {"Bullish momentum" if sig["macd_v"]>0 else "Bearish momentum"} |
| **ADX** | {sig["adx_v"]:.1f} | {"Strong trend (>25)" if sig["adx_v"]>25 else "Moderate trend" if sig["adx_v"]>20 else "⚠️ Weak/ranging"} |
| **Supertrend** | {"🟢 BULLISH" if sig["st_dir"]==1 else "🔴 BEARISH"} | Trend direction confirmed |
| **DI+/DI-** | {sig["dip"]:.1f} / {sig["din"]:.1f} | {"Bullish edge" if sig["dip"]>sig["din"] else "Bearish edge"} |
| **VWAP** | {sig["vwap_v"]:.4f} | Price {"above ✅" if px>sig["vwap_v"] else "below ❌"} VWAP |
| **ATR (14)** | {sig["atr"]:.4f} | {atr_pct:.2f}% of price (volatility) |

### 🎯 Key Levels
- **Resistance (R1):** {sig.get("r1", "N/A")}  
- **Support (S1):**    {sig.get("s1", "N/A")}  
- **BB Upper:**        {sig["bb_up"]:.4f}  
- **BB Lower:**        {sig["bb_lo"]:.4f}  

### 🚦 Signal
**Direction: {sig["signal"]}** | Confidence: **{sig["confidence"]:.0f}%** | Risk:Reward = **{rr_str}**

{"**Entry:** "+f'{sig["entry"]:.5f} | **SL:** '+f'{sig["sl"]:.5f} | **T1:** '+f'{sig["t1"]:.5f} | **T2:** '+f'{sig["t2"]:.5f} | **T3:** '+f'{sig["t3"]:.5f}' if sig["signal"] != "HOLD" else "No active trade setup — awaiting confirmation."}

### 📝 Reasoning
{reasons_fmt}

### 📦 Volume Analysis
{vol_str if vol_str else "Volume data unavailable"}

### ⚠️ Risk Management Notes
- Risk only **1–2% of capital** per trade
- Move SL to breakeven once T1 is reached
- Scale out at T1 (50%), T2 (30%), T3 (20%)
- Trailing SL locks in profits as price extends
- Avoid trading ADX < 20 with trend-following strategies
"""


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown('<div class="pt-logo">⚡ ProTrader AI</div>', unsafe_allow_html=True)
        st.caption("Professional Algo Trading Platform")
        st.divider()

        # Theme
        dark = st.toggle("🌙 Dark Theme", value=st.session_state.dark_mode, key="theme_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.divider()

        # Ticker
        tk_name = st.selectbox("📊 Asset", list(TICKERS.keys()), index=3)
        if tk_name == "⚙ Custom":
            ticker = st.text_input("Enter symbol (e.g. AAPL, ^NSEI)", "AAPL")
        else:
            ticker = TICKERS[tk_name]

        st.divider()

        # Strategy
        strategy = st.selectbox("🎯 Strategy", list(REC_TF.keys()), index=1)
        st.caption(f"💡 {STRATEGY_DESC[strategy]}")

        rec_tf, rec_p = REC_TF[strategy]
        tf_label  = st.selectbox("⏱ Timeframe", list(TIMEFRAMES.keys()),
                                  index=list(TIMEFRAMES.keys()).index(rec_tf))
        prd_label = st.selectbox("📅 Period",    list(PERIODS.keys()),
                                  index=list(PERIODS.keys()).index(rec_p))
        tf  = TIMEFRAMES[tf_label]
        prd = PERIODS[prd_label]

        st.divider()
        st.subheader("🛡 Risk / SL Settings")
        sl_type  = st.selectbox("SL Type",     list(SL_TYPES.keys()),  index=0)
        tgt_type = st.selectbox("Target Type", list(TARGET_TYPES.keys()), index=0)
        sl_mult  = st.slider("SL Multiplier (ATR×)",  0.5, 4.0, 1.5, 0.25)
        t_mult   = st.slider("Target Multiplier",      1.0, 8.0, 2.5, 0.25)

        st.divider()
        st.subheader("⏰ Trading Window")
        use_win  = st.checkbox("Restrict to Trading Hours", value=True)
        w_sh, w_sm = 9,  15
        w_eh, w_em = 15,  0
        if use_win:
            c1, c2 = st.columns(2)
            with c1: w_sh = st.number_input("Start Hr",  0, 23, 9);  w_sm = st.number_input("Start Min", 0, 59, 15)
            with c2: w_eh = st.number_input("End Hr",    0, 23, 15); w_em = st.number_input("End Min",   0, 59, 0)
        win_s = dtime(w_sh, w_sm); win_e = dtime(w_eh, w_em)

        st.divider()
        st.subheader("💼 Paper Trading")
        capital = st.number_input("Capital", 1_000, 10_000_000, 100_000, 1_000)
        qty     = st.number_input("Qty / Lot", 1, 10_000, 1, 1)
        st.metric("Balance", f"{st.session_state.paper_balance:,.0f}")
        if st.session_state.paper_position:
            p = st.session_state.paper_position
            st.success(f"Open {p['side']} @ {p['entry']:.5f}")
        if st.button("♻️ Reset Account"):
            st.session_state.paper_balance = capital
            st.session_state.paper_position = None
            st.session_state.trade_history  = []
            st.rerun()

    return dict(
        ticker=ticker, tk_name=tk_name, strategy=strategy,
        tf=tf, tf_label=tf_label, prd=prd, prd_label=prd_label,
        sl_type=SL_TYPES[sl_type], tgt_type=TARGET_TYPES[tgt_type],
        sl_mult=sl_mult, t_mult=t_mult,
        use_win=use_win, win_s=win_s, win_e=win_e,
        capital=capital, qty=qty,
    )


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    _init()
    cfg = sidebar()
    apply_theme(st.session_state.dark_mode)

    # ── Header ──────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0 4px 0;">
      <div>
        <div class="pt-logo" style="font-size:1.5rem;">⚡ ProTrader AI</div>
        <small style="opacity:.6;">{cfg['tk_name']} &nbsp;·&nbsp; {cfg['strategy']} &nbsp;·&nbsp; {cfg['tf_label']} &nbsp;·&nbsp; {cfg['prd_label']}</small>
      </div>
      <div>
        <span class="badge badge-{'up' if cfg['strategy'] in ('Swing','Positional') else 'nt'}">{cfg['strategy'].upper()}</span>
        &nbsp;<span class="badge badge-nt">{cfg['tf_label']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_live, tab_bt, tab_hist, tab_an = st.tabs(
        ["📡 Live Trading", "🔬 Backtesting", "📋 Trade History", "📊 Analysis"])

    # ══════════════════════════════════════════════════════
    # TAB 1 — LIVE TRADING
    # ══════════════════════════════════════════════════════
    with tab_live:
        col_r, col_a, col_info = st.columns([1, 1, 4])
        with col_r:
            last_ts = st.session_state.get("_last_refresh_ts", 0)
            since   = time_module.time() - last_ts
            btn_ok  = since >= _FETCH_MIN_GAP   # enforce gap on manual too
            if st.button("🔄 Refresh", use_container_width=True, disabled=not btn_ok):
                st.session_state["_last_refresh_ts"] = time_module.time()
                st.cache_data.clear(); st.rerun()
            if not btn_ok:
                st.caption(f"⏳ {_FETCH_MIN_GAP - since:.1f}s cooldown")
        with col_a:
            auto = st.checkbox("⚡ Auto (45s)", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto
        with col_info:
            st.caption(f"Last scan: {datetime.now().strftime('%H:%M:%S')} | {cfg['ticker']} | "
                       f"TF:{cfg['tf_label']} | Period:{cfg['prd_label']} | "
                       f"SL:{list(SL_TYPES.keys())[list(SL_TYPES.values()).index(cfg['sl_type'])]} | "
                       f"Target:{list(TARGET_TYPES.keys())[list(TARGET_TYPES.values()).index(cfg['tgt_type'])]}")

        df = fetch_live(cfg["ticker"], cfg["prd"], cfg["tf"])

        if df is None or len(df) < 30:
            st.error("⚠️ Data unavailable. Check ticker or network. Try a different period/timeframe.")
            st.stop()

        I   = compute_indicators(df)
        sig = get_signal(df, cfg["strategy"], cfg["sl_type"], cfg["tgt_type"], cfg["sl_mult"], cfg["t_mult"])

        if sig is None:
            st.warning("Computing indicators… not enough history yet."); st.stop()

        # ── Signal Banner ────────────────────────────────
        s_val = sig["signal"]; conf = sig["confidence"]; px = sig["entry"]
        if s_val == "LONG":
            css, icon = "pt-signal-long",  "🟢 LONG"
        elif s_val == "SHORT":
            css, icon = "pt-signal-short", "🔴 SHORT"
        else:
            css, icon = "pt-signal-hold",  "⏸️ HOLD"

        rr_str = f"1:{sig['rr']:.2f}" if s_val != "HOLD" else "—"
        sl_str = f"{sig['sl']:.5f}" if s_val != "HOLD" else "—"
        t1_str = f"{sig['t1']:.5f}" if s_val != "HOLD" else "—"
        t3_str = f"{sig['t3']:.5f}" if s_val != "HOLD" else "—"

        st.markdown(f"""
        <div class="{css}">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
            <div><span style="font-size:1.6rem;font-weight:800;font-family:Syne,sans-serif;">{icon}</span>
                 &nbsp;<span style="font-size:1.1rem;">Confidence: <b>{conf:.0f}%</b></span>
                 &nbsp;<span style="opacity:.75;font-size:.95rem;">R:R {rr_str}</span></div>
            <div style="font-family:JetBrains Mono,monospace;font-size:.9rem;line-height:2;">
              Entry: <b>{px:.5f}</b> &nbsp;|&nbsp; SL: <b>{sl_str}</b> &nbsp;|&nbsp;
              T1: <b>{t1_str}</b> &nbsp;|&nbsp; T2: <b>{sig.get("t2","—") if s_val!="HOLD" else "—"}</b> &nbsp;|&nbsp;
              T3: <b>{t3_str}</b>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ── Metrics Row ──────────────────────────────────
        chg = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Price",      f"{px:.4f}",         f"{chg:+.2f}%")
        m2.metric("RSI",        f"{sig['rsi']:.1f}",  "OS" if sig["rsi"]<30 else "OB" if sig["rsi"]>70 else "Neutral")
        m3.metric("ADX",        f"{sig['adx_v']:.1f}", "Strong" if sig["adx_v"]>25 else "Weak")
        m4.metric("MACD",       f"{sig['macd_v']:.4f}", "Bull" if sig["macd_v"]>0 else "Bear")
        m5.metric("Supertrend", "🟢 Bull" if sig["st_dir"]==1 else "🔴 Bear")
        m6.metric("VWAP",       f"{sig['vwap_v']:.4f}", "Above" if px > sig["vwap_v"] else "Below")
        m7.metric("ATR",        f"{sig['atr']:.4f}")

        # ── Chart ────────────────────────────────────────
        st.plotly_chart(build_chart(df, I, sig, st.session_state.dark_mode), use_container_width=True)

        # ── Commentary ───────────────────────────────────
        st.subheader("📡 Live Commentary")
        lines = generate_commentary(sig, cfg["strategy"], cfg["tk_name"])
        # Keep last 60 lines in log
        st.session_state.commentary_log = (st.session_state.commentary_log + lines)[-120:]
        block = "\n".join(st.session_state.commentary_log[-50:])
        st.markdown(f'<div class="pt-commentary">{block}</div>', unsafe_allow_html=True)

        # ── Paper Trade Controls ─────────────────────────
        st.divider()
        st.subheader("💼 Paper Trade Desk")

        def _enter(side):
            if st.session_state.paper_position is not None:
                st.warning("Close current position first."); return
            sl_p  = sig["sl"]; t3_p = sig["t3"]; tsl_p = sig["tsl_offset"]
            cost  = px * cfg["qty"]
            if cost > st.session_state.paper_balance:
                st.error("Insufficient balance."); return
            st.session_state.paper_position = {
                "side":       side, "entry": px, "qty": cfg["qty"],
                "sl":         sl_p, "target": t3_p, "tsl_offset": tsl_p,
                "best_price": px, "entry_time": datetime.now().isoformat(),
                "ticker":     cfg["ticker"], "strategy": cfg["strategy"],
            }
            st.success(f"✅ {side} entered @ {px:.5f} | SL: {sl_p:.5f} | T: {t3_p:.5f}")

        def _close_pos(reason="Manual Close"):
            pos = st.session_state.paper_position
            if pos is None: st.info("No open position."); return
            pnl = (px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]
            st.session_state.paper_balance += pnl
            st.session_state.trade_history.append({
                "entry_time":  pos["entry_time"],
                "exit_time":   datetime.now().isoformat(),
                "ticker":      pos.get("ticker", cfg["ticker"]),
                "strategy":    pos.get("strategy", cfg["strategy"]),
                "side":        pos["side"], "entry": pos["entry"], "exit": px,
                "qty":         pos["qty"], "pnl":  pnl, "exit_reason": reason,
            })
            st.session_state.paper_position = None
            (st.success if pnl > 0 else st.error)(f"Closed @ {px:.5f} | P&L: {pnl:+.2f}")

        col_l, col_s, col_c = st.columns(3)
        with col_l:
            if st.button("🟢 ENTER LONG",  use_container_width=True): _enter("LONG")
        with col_s:
            if st.button("🔴 ENTER SHORT", use_container_width=True): _enter("SHORT")
        with col_c:
            if st.button("❌ CLOSE POSITION", use_container_width=True): _close_pos()

        # Live P&L display
        pos = st.session_state.paper_position
        if pos:
            live_pnl = (px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]
            # Update trailing SL
            if pos["side"] == "LONG":
                pos["best_price"] = max(pos["best_price"], px)
                new_tsl = pos["best_price"] - pos["tsl_offset"]
                pos["sl"] = max(pos["sl"], new_tsl)
            else:
                pos["best_price"] = min(pos["best_price"], px)
                new_tsl = pos["best_price"] + pos["tsl_offset"]
                pos["sl"] = min(pos["sl"], new_tsl)

            color = "🟢" if live_pnl >= 0 else "🔴"
            st.info(f"{color} **{pos['side']}** @ {pos['entry']:.5f} | Current: {px:.5f} | "
                    f"Live P&L: **{live_pnl:+.2f}** | Trailing SL: {pos['sl']:.5f} | Target: {pos['target']:.5f}")
            # Auto SL/target check
            if pos["side"] == "LONG" and px <= pos["sl"]:
                st.error("⚠️ Trailing SL HIT — closing position"); _close_pos("Trailing SL")
            elif pos["side"] == "SHORT" and px >= pos["sl"]:
                st.error("⚠️ Trailing SL HIT — closing position"); _close_pos("Trailing SL")
            elif pos["side"] == "LONG" and px >= pos["target"]:
                st.success("🎯 Target HIT — closing position"); _close_pos("Target Hit")
            elif pos["side"] == "SHORT" and px <= pos["target"]:
                st.success("🎯 Target HIT — closing position"); _close_pos("Target Hit")

        # ── Auto-refresh with rate-limit guard ──────────
        if st.session_state.auto_refresh:
            REFRESH_EVERY = 45          # seconds between full refreshes
            last = st.session_state.get("_last_refresh_ts", 0)
            elapsed = time_module.time() - last
            remaining = max(0, int(REFRESH_EVERY - elapsed))

            if remaining > 0:
                st.caption(f"⏱ Next refresh in **{remaining}s** (rate-limit safe · min gap {_FETCH_MIN_GAP}s)")
                time_module.sleep(1)    # tick every 1 second
                st.rerun()
            else:
                st.session_state["_last_refresh_ts"] = time_module.time()
                st.cache_data.clear()
                st.rerun()

    # ══════════════════════════════════════════════════════
    # TAB 2 — BACKTESTING
    # ══════════════════════════════════════════════════════
    with tab_bt:
        st.subheader("🔬 Strategy Backtesting")
        st.caption("⚡ Uses identical signal logic as Live Trading for consistency")

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bt_prd_label = st.selectbox("Backtest Period", list(PERIODS.keys()), index=4, key="bt_prd")
            bt_prd       = PERIODS[bt_prd_label]
        with bc2:
            bt_tf_label  = st.selectbox("Backtest Timeframe", list(TIMEFRAMES.keys()), key="bt_tf",
                                         index=list(TIMEFRAMES.keys()).index(cfg["tf_label"]))
            bt_tf        = TIMEFRAMES[bt_tf_label]
        with bc3:
            bt_cap = st.number_input("Starting Capital", 1_000, 10_000_000, cfg["capital"], 1_000, key="bt_cap")
            bt_qty = st.number_input("Qty per Trade", 1, 10_000, cfg["qty"], 1, key="bt_qty")

        use_win_bt = st.checkbox("Use Trading Window Filter",
                                  value=cfg["use_win"], key="bt_win",
                                  help=f"Restrict to {cfg['win_s']} – {cfg['win_e']}")

        run_col, _ = st.columns([2, 5])
        with run_col:
            run_bt = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

        if run_bt:
            with st.spinner("Fetching data & running backtest…"):
                df_bt = fetch(cfg["ticker"], bt_prd, bt_tf)

            if df_bt is None or len(df_bt) < 60:
                st.error("Not enough data. Try a longer period or different timeframe.")
            else:
                with st.spinner("Simulating trades…"):
                    stats, t_df, equity = run_backtest(
                        df_bt, cfg["strategy"],
                        cfg["sl_mult"], cfg["t_mult"],
                        cfg["sl_type"], cfg["tgt_type"],
                        use_win_bt,
                        cfg["win_s"] if use_win_bt else None,
                        cfg["win_e"] if use_win_bt else None,
                        bt_cap, bt_qty,
                    )

                if stats is None:
                    st.warning("No trades generated. Try relaxing signal thresholds or a different period.")
                else:
                    # ── Performance Cards ────────────────────────
                    st.subheader("📈 Performance Summary")
                    p1,p2,p3,p4,p5,p6 = st.columns(6)
                    p1.metric("Trades",        stats["trades"])
                    p2.metric("Win Rate",       f"{stats['win_rate']:.1f}%",
                               "✅ Good" if stats["win_rate"]>=50 else "❌ Poor")
                    p3.metric("Total P&L",      f"{stats['total_pnl']:+,.2f}",
                               f"{stats['return_pct']:+.1f}%")
                    p4.metric("Profit Factor",  f"{min(stats['pf'],99):.2f}",
                               "✅" if stats["pf"]>=1.5 else "⚠️")
                    p5.metric("Max Drawdown",   f"{stats['max_dd']:.1f}%")
                    p6.metric("Sharpe Ratio",   f"{stats['sharpe']:.2f}")

                    p7,p8,p9,p10 = st.columns(4)
                    p7.metric("Avg Win",        f"{stats['avg_win']:+.2f}")
                    p8.metric("Avg Loss",       f"{stats['avg_loss']:+.2f}")
                    p9.metric("Expectancy",     f"{stats['expectancy']:+.2f}")
                    p10.metric("Final Capital", f"{stats['final']:,.0f}")

                    # ── Equity Curve ────────────────────────────
                    BG2 = "#0d1117" if st.session_state.dark_mode else "#f8fafc"
                    TC  = "#e6edf3" if st.session_state.dark_mode else "#1a1a2e"
                    eq_fig = go.Figure()
                    eq_fig.add_trace(go.Scatter(
                        x=list(range(len(equity))), y=equity,
                        mode="lines", name="Equity",
                        line=dict(color="#00d4aa", width=2.5),
                        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
                    ))
                    eq_fig.add_hline(y=bt_cap, line_dash="dash", line_color="#94a3b8",
                                      annotation_text=" Initial Capital")
                    eq_fig.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2,
                                         font=dict(color=TC), height=280,
                                         margin=dict(l=50,r=20,t=30,b=20),
                                         title=dict(text="Equity Curve", font=dict(color=TC)))
                    st.plotly_chart(eq_fig, use_container_width=True)

                    # ── Per-trade P&L bars ──────────────────────
                    pnl_colors = ["#00d4aa" if p>0 else "#ff4d6d" for p in t_df["pnl"]]
                    bar_fig = go.Figure(go.Bar(x=list(range(len(t_df))), y=t_df["pnl"],
                                               marker_color=pnl_colors, opacity=0.8, name="P&L"))
                    bar_fig.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2,
                                          font=dict(color=TC), height=220,
                                          margin=dict(l=50,r=20,t=30,b=20),
                                          title=dict(text="Trade-by-Trade P&L", font=dict(color=TC)))
                    st.plotly_chart(bar_fig, use_container_width=True)

                    # ── Exit Reason Pie ─────────────────────────
                    er = t_df["exit_reason"].value_counts()
                    pie_fig = go.Figure(go.Pie(labels=er.index, values=er.values, hole=0.4,
                                               marker=dict(colors=["#00d4aa","#ff4d6d","#FFD700","#A78BFA","#94a3b8"])))
                    pie_fig.update_layout(paper_bgcolor=BG2, font=dict(color=TC), height=250,
                                          margin=dict(l=20,r=20,t=30,b=20),
                                          title=dict(text="Exit Reasons", font=dict(color=TC)))
                    st.plotly_chart(pie_fig, use_container_width=True)

                    # ── Drawdown Curve ──────────────────────────
                    eq_s   = pd.Series(equity)
                    peak_s = eq_s.cummax()
                    dd_s   = (peak_s - eq_s) / peak_s * 100
                    dd_fig = go.Figure(go.Scatter(x=list(range(len(dd_s))), y=-dd_s,
                                                   mode="lines", fill="tozeroy",
                                                   fillcolor="rgba(255,77,109,0.12)",
                                                   line=dict(color="#ff4d6d", width=1.5), name="Drawdown %"))
                    dd_fig.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2, font=dict(color=TC),
                                         height=200, margin=dict(l=50,r=20,t=30,b=20),
                                         title=dict(text="Drawdown (%)", font=dict(color=TC)))
                    st.plotly_chart(dd_fig, use_container_width=True)

                    # ── Trade Log ───────────────────────────────
                    st.subheader("📋 Detailed Trade Log")
                    show_cols = ["entry_time","exit_time","side","entry","exit","pnl","exit_reason"]
                    st.dataframe(
                        t_df[show_cols].style
                          .format({"entry": "{:.5f}", "exit": "{:.5f}", "pnl": "{:+.2f}"})
                          .applymap(lambda v: "color:#00d4aa" if isinstance(v,(int,float)) and v>0
                                    else "color:#ff4d6d" if isinstance(v,(int,float)) and v<0 else "",
                                    subset=["pnl"]),
                        use_container_width=True, height=320
                    )

                    # ── Text Summary ─────────────────────────────
                    st.subheader("📝 Backtest Narrative")
                    verdict = "PROFITABLE ✅" if stats["total_pnl"] > 0 else "NOT PROFITABLE ❌"
                    er_dict = t_df["exit_reason"].value_counts().to_dict()
                    st.markdown(f"""
**Strategy:** {cfg["strategy"]} on **{cfg["tk_name"]}** ({bt_tf_label} | {bt_prd_label})  
Window filter: {"ON (" + str(cfg["win_s"]) + " – " + str(cfg["win_e"]) + ")" if use_win_bt else "OFF"}

The backtest simulated **{stats["trades"]} trades** achieving a win rate of **{stats["win_rate"]:.1f}%** with
a profit factor of **{min(stats["pf"],99):.2f}**. The strategy is **{verdict}** on this data,
returning **{stats["return_pct"]:.1f}%** ({stats["total_pnl"]:+,.2f}) from {bt_cap:,} capital.

Max drawdown was **{stats["max_dd"]:.1f}%** — {"acceptable" if stats["max_dd"]<20 else "consider tighter SLs"}.  
Sharpe ratio: **{stats["sharpe"]:.2f}** {"(excellent)" if stats["sharpe"]>2 else "(good)" if stats["sharpe"]>1 else "(needs improvement)"}.  
Expectancy per trade: **{stats["expectancy"]:+.2f}**.  
Exit breakdown: {er_dict}
                    """)

    # ══════════════════════════════════════════════════════
    # TAB 3 — TRADE HISTORY
    # ══════════════════════════════════════════════════════
    with tab_hist:
        st.subheader("📋 Paper Trade History")

        if not st.session_state.trade_history:
            st.info("No paper trades yet. Use the Live Trading tab to execute trades.")
        else:
            th = pd.DataFrame(st.session_state.trade_history)
            total = th["pnl"].sum(); wr = len(th[th["pnl"]>0])/len(th)*100

            hm1,hm2,hm3,hm4,hm5 = st.columns(5)
            hm1.metric("Total Trades",  len(th))
            hm2.metric("Total P&L",     f"{total:+,.2f}", "Profit" if total>0 else "Loss")
            hm3.metric("Win Rate",       f"{wr:.1f}%")
            hm4.metric("Balance",        f"{st.session_state.paper_balance:,.0f}")
            hm5.metric("Best Trade",     f"{th['pnl'].max():+.2f}")

            BG2 = "#0d1117" if st.session_state.dark_mode else "#f8fafc"
            TC  = "#e6edf3" if st.session_state.dark_mode else "#1a1a2e"

            # Cumulative P&L
            cum_fig = go.Figure(go.Scatter(y=th["pnl"].cumsum(), mode="lines+markers",
                                            line=dict(color="#00d4aa",width=2),
                                            marker=dict(size=5)))
            cum_fig.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2, font=dict(color=TC),
                                   height=220, margin=dict(l=50,r=20,t=30,b=20),
                                   title=dict(text="Cumulative P&L", font=dict(color=TC)))
            st.plotly_chart(cum_fig, use_container_width=True)

            # Table
            st.dataframe(
                th.style.format({"entry": "{:.5f}", "exit": "{:.5f}", "pnl": "{:+.2f}"})
                  .applymap(lambda v: "color:#00d4aa" if isinstance(v,(int,float)) and v>0
                            else "color:#ff4d6d" if isinstance(v,(int,float)) and v<0 else "",
                            subset=["pnl"]),
                use_container_width=True, height=380
            )

            cl1, cl2 = st.columns([1,5])
            with cl1:
                if st.button("🗑️ Clear History"):
                    st.session_state.trade_history = []; st.rerun()
            with cl2:
                csv = th.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export CSV", csv, "trade_history.csv", "text/csv")

    # ══════════════════════════════════════════════════════
    # TAB 4 — ANALYSIS
    # ══════════════════════════════════════════════════════
    with tab_an:
        st.subheader("📊 Deep Market Analysis")

        an_since = time_module.time() - st.session_state.get("_last_refresh_ts", 0)
        an_ok    = an_since >= _FETCH_MIN_GAP
        if st.button("🔄 Refresh Analysis", key="an_refresh", disabled=not an_ok):
            st.session_state["_last_refresh_ts"] = time_module.time()
            st.cache_data.clear(); st.rerun()

        df_an = fetch(cfg["ticker"], cfg["prd"], cfg["tf"])
        if df_an is None or len(df_an) < 30:
            st.error("Not enough data for analysis."); st.stop()

        I_an  = compute_indicators(df_an)
        sig_an = get_signal(df_an, cfg["strategy"], cfg["sl_type"], cfg["tgt_type"],
                             cfg["sl_mult"], cfg["t_mult"])

        # ── Multi-strategy consensus ─────────────────────
        st.subheader("🎯 Multi-Strategy Consensus")
        consensus = {}
        for strat in ["Scalping","Intraday","Swing","Positional"]:
            s = get_signal(df_an, strat, cfg["sl_type"], cfg["tgt_type"], cfg["sl_mult"], cfg["t_mult"])
            if s: consensus[strat] = s

        cv = {"LONG":0,"SHORT":0,"HOLD":0}
        cols_c = st.columns(4)
        for idx, (strat_n, s) in enumerate(consensus.items()):
            cv[s["signal"]] += 1
            col_c = "#00d4aa" if s["signal"]=="LONG" else "#ff4d6d" if s["signal"]=="SHORT" else "#f59e0b"
            with cols_c[idx]:
                st.markdown(f"""
                <div style="background:{col_c}1a;border:1.5px solid {col_c};border-radius:12px;
                             padding:14px;text-align:center;">
                  <div style="font-weight:700;font-size:.9rem;opacity:.75;">{strat_n}</div>
                  <div style="font-size:1.3rem;font-weight:800;color:{col_c};">{s["signal"]}</div>
                  <div style="font-size:.8rem;font-family:JetBrains Mono,monospace;">{s["confidence"]:.0f}%</div>
                </div>""", unsafe_allow_html=True)

        overall = max(cv, key=cv.get)
        badge   = "badge-up" if overall=="LONG" else "badge-dn" if overall=="SHORT" else "badge-nt"
        st.markdown(f'<br><b>Overall Consensus:</b> <span class="badge {badge}">{overall}</span> '
                    f'({cv[overall]}/4 strategies agree)', unsafe_allow_html=True)

        st.divider()

        # ── Indicator dashboard ──────────────────────────
        if sig_an:
            st.subheader("📊 Indicator Dashboard")
            ind_rows = [
                ("RSI (14)",     f"{sig_an['rsi']:.1f}",     "Oversold" if sig_an["rsi"]<30 else "Overbought" if sig_an["rsi"]>70 else "Neutral"),
                ("MACD",         f"{sig_an['macd_v']:.5f}",  "Bullish" if sig_an["macd_v"]>0 else "Bearish"),
                ("ADX",          f"{sig_an['adx_v']:.1f}",   "Strong" if sig_an["adx_v"]>25 else "Weak"),
                ("Supertrend",   "🟢 Bull" if sig_an["st_dir"]==1 else "🔴 Bear", ""),
                ("DI+ / DI-",    f"{sig_an['dip']:.1f}/{sig_an['din']:.1f}", "Bull edge" if sig_an["dip"]>sig_an["din"] else "Bear edge"),
                ("EMA21/EMA50",  f"{sig_an['ema21']:.4f}/{sig_an['ema50']:.4f}", "↑ Uptrend" if sig_an["ema21"]>sig_an["ema50"] else "↓ Downtrend"),
                ("VWAP",         f"{sig_an['vwap_v']:.4f}",  "Price above" if sig_an["entry"]>sig_an["vwap_v"] else "Price below"),
                ("ATR",          f"{sig_an['atr']:.5f}",     f"{sig_an['atr']/sig_an['entry']*100:.2f}% vol"),
                ("BB Upper/Lo",  f"{sig_an['bb_up']:.4f}/{sig_an['bb_lo']:.4f}", ""),
                ("Signal Score", f"{sig_an['score']:.0f}/100",""),
                ("R1/S1",        f"{sig_an.get('r1','N/A')}/{sig_an.get('s1','N/A')}", "Pivot levels"),
                ("OBV",          f"{sig_an['obv_v']:.0f}",   ""),
            ]
            # 12 items → 3 rows × 4 cols
            for row_start in range(0, len(ind_rows), 4):
                cols = st.columns(4)
                for col_i, item in enumerate(ind_rows[row_start:row_start+4]):
                    nm, vl, nt = item
                    with cols[col_i]:
                        st.metric(nm, vl, nt if nt else None)

        st.divider()

        # ── Full text analysis ───────────────────────────
        st.subheader("📝 Analysis Report")
        st.markdown(analysis_summary(df_an, I_an, sig_an, cfg["tk_name"],
                                      cfg["strategy"], cfg["tf_label"], cfg["prd_label"]))

        st.divider()

        # ── Chart ────────────────────────────────────────
        st.subheader("📈 Full Chart")
        st.plotly_chart(build_chart(df_an, I_an, sig_an, st.session_state.dark_mode, height=850),
                        use_container_width=True)


if __name__ == "__main__":
    main()
