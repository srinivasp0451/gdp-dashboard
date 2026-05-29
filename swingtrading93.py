# ═══════════════════════════════════════════════════════════════════
#  AlgoTrader Pro — Comprehensive Market Intelligence Dashboard
#  Powered by yfinance | Built with Streamlit + Plotly
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoTrader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
#MainMenu, footer, header { visibility: hidden; }

.stApp { background-color: #04080f; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #0a1428 100%);
    border-right: 1px solid #0e2040;
}

/* Sidebar button — broad selectors cover all Streamlit versions */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #0e2a44, #163550) !important;
    color: #00e5ff !important;
    border: 1px solid #1e5070 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 10px 16px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.4px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 42px !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] .stButton button:hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #123358, #1c4268) !important;
    border-color: #00e5ff !important;
    box-shadow: 0 0 14px rgba(0,229,255,0.25) !important;
    color: #ffffff !important;
}

.app-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    background: linear-gradient(135deg, #00e5ff 0%, #3b82f6 40%, #a855f7 70%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
    padding: 8px 0;
}
.app-sub {
    text-align: center;
    color: #1e4060;
    font-size: 0.72rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

.sh {
    font-size: 1rem;
    font-weight: 700;
    color: #c8ddf0;
    background: linear-gradient(90deg, #071828, #04080f);
    border-left: 3px solid #00e5ff;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    margin: 20px 0 12px 0;
    letter-spacing: 0.3px;
}

.pcard {
    background: linear-gradient(135deg, #071828, #0b1f35);
    border: 1px solid #0e2a42;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,.5);
    transition: border-color .2s;
}
.pcard:hover { border-color: #1e4060; }
.pcard-lbl { color: #2c5070; font-size: .68rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 3px; }
.pcard-val { color: #dceeff; font-size: 1.45rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.up  { color: #00e676 !important; }
.dn  { color: #ff4444 !important; }
.neu { color: #ffc107 !important; }

.ins { border-radius: 0 8px 8px 0; padding: 11px 15px; margin: 6px 0; font-size: .86rem; color: #b8cfe8; line-height: 1.7; }
.ins.bull { background: rgba(0,230,118,.05);  border-left: 3px solid #00e676; }
.ins.bear { background: rgba(255,68,68,.05);  border-left: 3px solid #ff4444; }
.ins.info { background: rgba(0,229,255,.05);  border-left: 3px solid #00e5ff; }
.ins.warn { background: rgba(255,193,7,.05);  border-left: 3px solid #ffc107; }

.pill { display:inline-block; padding:3px 11px; border-radius:20px; font-size:.72rem; font-weight:700; letter-spacing:.5px; }
.pill-bull { background:rgba(0,230,118,.12);  color:#00e676; border:1px solid rgba(0,230,118,.35); }
.pill-bear { background:rgba(255,68,68,.12);  color:#ff4444; border:1px solid rgba(255,68,68,.35); }
.pill-neu  { background:rgba(255,193,7,.12);  color:#ffc107; border:1px solid rgba(255,193,7,.35); }

.stTabs [data-baseweb="tab-list"] {
    background: #060d1a; border-radius: 8px; padding: 3px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-weight: 600; color: #2c5070;
    font-size: .83rem; padding: 8px 16px;
}
.stTabs [aria-selected="true"] { background: #0a1e35 !important; color: #00e5ff !important; }

div[data-testid="stDataFrame"] > div { border-radius: 8px !important; }

/* Primary button global override */
.stButton [kind="primary"],
button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0e2a44, #163550) !important;
    color: #00e5ff !important;
    border: 1px solid #1e5070 !important;
    font-weight: 700 !important;
}

.div { height:1px; background:linear-gradient(90deg,transparent,#0e2a42,transparent); margin:16px 0; }
.tbox {
    background: linear-gradient(135deg, #060e1c, #091a2e);
    border: 1px solid #0e2a42; border-radius: 14px; padding: 22px; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
TICKER_MAP = {
    "🇮🇳  Nifty 50":         "^NSEI",
    "🇮🇳  Bank Nifty":       "^NSEBANK",
    "🇮🇳  Sensex":           "^BSESN",
    "🇮🇳  Nifty IT":         "^CNXIT",
    "₿   Bitcoin / USD":    "BTC-USD",
    "Ξ   Ethereum / USD":   "ETH-USD",
    "💵  USD / INR":         "USDINR=X",
    "🥇  Gold Futures":      "GC=F",
    "🥈  Silver Futures":    "SI=F",
    "🛢️  Crude Oil WTI":     "CL=F",
    "⛽  Natural Gas":       "NG=F",
    "📊  S&P 500":           "^GSPC",
    "📊  NASDAQ 100":        "^NDX",
    "📊  Dow Jones":         "^DJI",
    "💴  EUR / USD":         "EURUSD=X",
    "💷  GBP / USD":         "GBPUSD=X",
    "🇯🇵  USD / JPY":         "JPY=X",
    "✏️  Custom Ticker":     "CUSTOM",
}

INTERVAL_PERIODS = {
    "1m":  ["1d", "5d"],
    "5m":  ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo"],
    "30m": ["1d", "5d", "1mo"],
    "1h":  ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    "4h":  ["5d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "1mo": ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
}

INTERVAL_LABELS = {
    "1m":"1 Min","5m":"5 Min","15m":"15 Min","30m":"30 Min",
    "1h":"1 Hr","4h":"4 Hr*","1d":"Daily","1wk":"Weekly","1mo":"Monthly",
}

INDIAN_TICKERS = {"^NSEI","^NSEBANK","^BSESN","^CNXIT"}
INTRADAY_IVLS  = {"1m","5m","15m","30m","1h","4h"}
MONTH_NAMES    = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
DAY_NAMES      = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday"}

PLOT_BG = "#04080f"; GRID_BG = "#060e1c"

# ──────────────────────────────────────────────────────────────────
# DATA FETCHING  (cached, rate-limited)
# ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlc(ticker: str, interval: str, period: str):
    try:
        actual = "1h" if interval == "4h" else interval
        time.sleep(1)
        df = yf.download(ticker, interval=actual, period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df.dropna(subset=["Close"])
        if interval == "4h":
            agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
            if "Volume" in df.columns:
                agg["Volume"] = "sum"
            df = df.resample("4h").agg(agg).dropna(subset=["Close"])
        return df
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(ticker: str, period: str = "2y"):
    try:
        time.sleep(1)
        df = yf.download(ticker, interval="1d", period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df.dropna(subset=["Close"])
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    try:
        time.sleep(1)
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_expiries(ticker: str) -> list:
    try:
        time.sleep(1)
        opts = yf.Ticker(ticker).options
        return list(opts) if opts else []
    except Exception:
        return []


def fetch_chain(ticker: str, expiry: str):
    try:
        time.sleep(1)
        return yf.Ticker(ticker).option_chain(expiry)
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    lo = df["Low"].astype(float)
    v  = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    # SMAs + EMAs
    for p in [9, 15, 20, 50, 100, 200]:
        df[f"SMA{p}"] = c.rolling(p).mean()
        df[f"EMA{p}"] = c.ewm(span=p, adjust=False).mean()

    # RSI-14
    d  = c.diff()
    g  = d.clip(lower=0).rolling(14).mean()
    ls = (-d).clip(lower=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    # MACD
    df["MACD"]      = c.ewm(12, adjust=False).mean() - c.ewm(26, adjust=False).mean()
    df["MACD_Sig"]  = df["MACD"].ewm(9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

    # Bollinger Bands (20, 2)
    df["BB_Mid"] = c.rolling(20).mean()
    bs = c.rolling(20).std()
    df["BB_Up"]  = df["BB_Mid"] + 2 * bs
    df["BB_Dn"]  = df["BB_Mid"] - 2 * bs
    df["BB_Pct"] = (c - df["BB_Dn"]) / (df["BB_Up"] - df["BB_Dn"]).replace(0, np.nan)

    # VWAP
    tp = (h + lo + c) / 3
    df["VWAP"] = (tp * v).cumsum() / v.cumsum().replace(0, np.nan)

    # ATR
    pc = c.shift(1)
    tr = pd.concat([(h - lo).abs(), (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ADX / +DI / -DI
    up_move   = h.diff()
    dn_move   = -lo.diff()
    pdm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    atr14    = tr.rolling(14).mean()
    pdi      = 100 * pdm.rolling(14).mean() / atr14.replace(0, np.nan)
    ndi      = 100 * ndm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()
    df["+DI"] = pdi
    df["-DI"] = ndi

    # Stochastic (14, 3)
    l14 = lo.rolling(14).min()
    h14 = h.rolling(14).max()
    df["Stoch_K"] = 100 * (c - l14) / (h14 - l14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # Pivot points (standard, previous candle)
    pp = (h.shift(1) + lo.shift(1) + c.shift(1)) / 3
    r  =  h.shift(1) - lo.shift(1)
    df["PP"] = pp
    df["R1"] = 2 * pp - lo.shift(1)
    df["S1"] = 2 * pp - h.shift(1)
    df["R2"] = pp + r
    df["S2"] = pp - r
    df["R3"] = h.shift(1) + 2 * (pp - lo.shift(1))
    df["S3"] = lo.shift(1) - 2 * (h.shift(1) - pp)

    # Change cols
    df["Chg_Abs"] = c.diff()
    df["Chg_Pct"] = c.pct_change() * 100

    return df

# ──────────────────────────────────────────────────────────────────
# RANGE ANALYSIS
# ──────────────────────────────────────────────────────────────────
def compute_ranges(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or len(daily) < 6:
        return pd.DataFrame()
    c  = daily["Close"].astype(float)
    tc = float(c.iloc[-1])
    rows = []

    def add_row(label, subset):
        if subset.empty:
            return
        rh = float(subset.max())
        rl = float(subset.min())
        rng = rh - rl
        pos = (tc - rl) / rng * 100 if rng > 0 else 50.0
        viol = tc > rh or tc < rl
        rows.append({
            "Period":          label,
            "Range High":      round(rh, 2),
            "Range Low":       round(rl, 2),
            "Range (pts)":     round(rng, 2),
            "Range (%)":       round(rng / rl * 100, 2) if rl else 0,
            "Today Close":     round(tc, 2),
            "Position (%)":    round(pos, 1),
            "Status":          "🔴 BREAKOUT" if viol else "🟢 Within Range",
            "_v":              viol,
        })

    # Day ranges 2–5
    for n in range(2, 6):
        if len(c) > n:
            add_row(f"{n}-Day", c.iloc[-(n + 1):-1])

    # Week ranges 2–3
    for n in [2, 3]:
        d = n * 5
        if len(c) > d:
            add_row(f"{n}-Week", c.iloc[-(d + 1):-1])

    # Month ranges 1–12
    for n in range(1, 13):
        d = n * 21
        if len(c) > d:
            add_row(f"{n}-Month", c.iloc[-(d + 1):-1])

    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────
# TECHNICAL INSIGHTS
# ──────────────────────────────────────────────────────────────────
def _g(series_row, col):
    v = series_row.get(col)
    try:
        fv = float(v)
        return None if np.isnan(fv) else fv
    except Exception:
        return None


def tech_insights(df: pd.DataFrame) -> list:
    if df is None or len(df) < 30:
        return []
    last = df.iloc[-1]
    close  = _g(last, "Close")
    rsi    = _g(last, "RSI")
    macd   = _g(last, "MACD")
    msig   = _g(last, "MACD_Sig")
    mhist  = _g(last, "MACD_Hist")
    adx    = _g(last, "ADX")
    pdi    = _g(last, "+DI")
    ndi    = _g(last, "-DI")
    bb_p   = _g(last, "BB_Pct")
    ema20  = _g(last, "EMA20")
    ema50  = _g(last, "EMA50")
    ema200 = _g(last, "EMA200")
    atr    = _g(last, "ATR")
    stk    = _g(last, "Stoch_K")
    std    = _g(last, "Stoch_D")

    out = []

    # ── Trend structure
    if close and ema20 and ema50 and ema200:
        if close > ema20 > ema50 > ema200:
            out.append(("bull", "🚀 <b>Strong Bull Trend:</b> Price > EMA20 > EMA50 > EMA200. All moving averages aligned bullishly. Trend-following longs are favoured."))
        elif close < ema20 < ema50 < ema200:
            out.append(("bear", "🔻 <b>Strong Bear Trend:</b> Price < EMA20 < EMA50 < EMA200. All moving averages aligned bearishly. Avoid longs; look for short opportunities."))
        elif close > ema200:
            out.append(("bull", f"📈 Price is <b>above EMA200</b> ({ema200:.2f}), indicating a long-term bullish structure."))
        else:
            out.append(("bear", f"📉 Price is <b>below EMA200</b> ({ema200:.2f}), indicating a long-term bearish structure."))
        if ema20 and ema50:
            if ema20 > ema50:
                out.append(("bull", f"✅ EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}): short-term momentum is <b>bullish</b>."))
            else:
                out.append(("bear", f"❌ EMA20 ({ema20:.2f}) < EMA50 ({ema50:.2f}): short-term momentum is <b>bearish</b>."))

    # ── RSI
    if rsi is not None:
        if rsi >= 75:
            out.append(("bear", f"⚠️ <b>RSI = {rsi:.1f}</b> — Highly overbought. Probability of pullback/mean-reversion is elevated. Watch for divergence."))
        elif rsi >= 60:
            out.append(("bull", f"📊 <b>RSI = {rsi:.1f}</b> — Bullish momentum zone (60–75). Healthy uptrend conditions."))
        elif rsi <= 25:
            out.append(("bull", f"⚡ <b>RSI = {rsi:.1f}</b> — Deeply oversold. High probability mean-reversion bounce likely. Watch for bullish engulfing candles."))
        elif rsi <= 40:
            out.append(("bear", f"📊 <b>RSI = {rsi:.1f}</b> — Bearish zone (25–40). Weak momentum; avoid aggressive longs."))
        else:
            out.append(("info", f"📊 <b>RSI = {rsi:.1f}</b> — Neutral zone (40–60). No strong momentum signal; wait for a breakout."))

    # ── MACD
    if macd is not None and msig is not None:
        if macd > msig and macd > 0:
            out.append(("bull", f"✅ <b>MACD Bullish:</b> MACD ({macd:.3f}) above signal ({msig:.3f}) in positive territory. Strong bullish momentum."))
        elif macd > msig and macd <= 0:
            out.append(("info", f"🔄 <b>MACD Bullish Crossover</b> (still negative). Early-stage reversal signal — wait for MACD to cross zero for confirmation."))
        elif macd < msig and macd < 0:
            out.append(("bear", f"❌ <b>MACD Bearish:</b> MACD ({macd:.3f}) below signal in negative territory. Sustained selling pressure."))
        else:
            out.append(("warn", f"🔄 <b>MACD Bearish Crossover</b> (still positive). Early warning — momentum may be turning down."))
        if mhist is not None:
            prev_hist = _g(df.iloc[-2] if len(df) > 1 else df.iloc[-1], "MACD_Hist")
            if prev_hist is not None:
                if mhist > 0 and mhist > prev_hist:
                    out.append(("bull", "📈 MACD histogram <b>expanding positively</b> — momentum is accelerating to the upside."))
                elif mhist < 0 and mhist < prev_hist:
                    out.append(("bear", "📉 MACD histogram <b>expanding negatively</b> — selling momentum is intensifying."))
                elif abs(mhist) < abs(prev_hist):
                    out.append(("warn", "⚠️ MACD histogram <b>contracting</b> — momentum is fading. Possible reversal or consolidation ahead."))

    # ── ADX
    if adx is not None and pdi is not None and ndi is not None:
        trend_dir = "bullish" if pdi > ndi else "bearish"
        if adx >= 40:
            out.append(("info", f"💪 <b>ADX = {adx:.1f}</b> — Very strong {trend_dir} trend. Trend-following strategies have high edge. +DI={pdi:.1f}, -DI={ndi:.1f}."))
        elif adx >= 25:
            out.append(("info", f"📐 <b>ADX = {adx:.1f}</b> — Trending market ({trend_dir}). +DI={pdi:.1f} vs -DI={ndi:.1f}. Trend strategies preferred over oscillators."))
        else:
            out.append(("warn", f"↔️ <b>ADX = {adx:.1f}</b> — Weak/ranging market. Trend-following strategies are risky; consider range trading. +DI={pdi:.1f}, -DI={ndi:.1f}."))

    # ── Bollinger Bands
    if bb_p is not None:
        if bb_p > 0.95:
            out.append(("bear", f"🎯 <b>BB %B = {bb_p:.2f}</b> — Price at upper Bollinger Band. Statistically overbought; mean reversion is likely."))
        elif bb_p < 0.05:
            out.append(("bull", f"🎯 <b>BB %B = {bb_p:.2f}</b> — Price at lower Bollinger Band. Statistically oversold; potential bounce."))
        elif 0.45 < bb_p < 0.55:
            out.append(("info", f"📏 <b>BB %B = {bb_p:.2f}</b> — Price at BB midline. Consolidating; directional breakout imminent."))
        if close and atr and ema20:
            bb_width = _g(last, "BB_Up")
            bb_dn    = _g(last, "BB_Dn")
            if bb_width and bb_dn and close:
                bw_pct = (bb_width - bb_dn) / close * 100
                if bw_pct < 2:
                    out.append(("warn", f"🔥 <b>Bollinger Band Squeeze!</b> Band width = {bw_pct:.1f}% — Volatility is historically low. Explosive move is building up."))

    # ── ATR
    if atr is not None and close:
        atr_pct = atr / close * 100
        if atr_pct > 3:
            out.append(("warn", f"⚡ <b>High Volatility:</b> ATR = {atr:.2f} ({atr_pct:.1f}% of price). Use wider stops. Intraday swings will be large."))
        elif atr_pct < 0.4:
            out.append(("info", f"😴 <b>Low Volatility:</b> ATR = {atr:.2f} ({atr_pct:.2f}%). Compression phase — a breakout move may follow."))

    # ── Stochastic
    if stk is not None:
        if stk >= 80:
            out.append(("bear", f"📈 <b>Stochastic %K = {stk:.1f}</b> — Overbought. Watch for %K crossing below %D as a sell trigger."))
        elif stk <= 20:
            out.append(("bull", f"📉 <b>Stochastic %K = {stk:.1f}</b> — Oversold. Watch for %K crossing above %D as a buy trigger."))
        if stk is not None and std is not None:
            if stk > std and stk < 80:
                out.append(("bull", f"✅ Stochastic <b>%K > %D</b> ({stk:.1f} > {std:.1f}) — Bullish cross, momentum building."))
            elif stk < std and stk > 20:
                out.append(("bear", f"❌ Stochastic <b>%K < %D</b> ({stk:.1f} < {std:.1f}) — Bearish cross, momentum weakening."))

    return out

# ──────────────────────────────────────────────────────────────────
# TECHNICAL CHART
# ──────────────────────────────────────────────────────────────────
def build_tech_chart(df: pd.DataFrame, name: str, interval: str) -> go.Figure:
    has_vol = "Volume" in df.columns and float(df["Volume"].sum()) > 0
    row_h = [0.44, 0.11, 0.15, 0.15, 0.15] if has_vol else [0.52, 0.0, 0.16, 0.16, 0.16]
    titles = ["Price · Bollinger Bands · Moving Averages",
              "Volume" if has_vol else "",
              "RSI (14)",
              "MACD (12 · 26 · 9)",
              "ADX · +DI · -DI (14)"]

    n_rows = 5 if has_vol else 5
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_h,
        subplot_titles=titles,
        vertical_spacing=0.022,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing=dict(line=dict(color="#00e676", width=1), fillcolor="#00e676"),
        decreasing=dict(line=dict(color="#ff4444", width=1), fillcolor="#ff4444"),
    ), row=1, col=1)

    # Bollinger Bands
    if "BB_Up" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Up"],
            line=dict(color="rgba(56,189,248,0.45)", width=1, dash="dot"),
            name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Dn"],
            line=dict(color="rgba(56,189,248,0.45)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(56,189,248,0.04)",
            name="BB Lower", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"],
            line=dict(color="rgba(56,189,248,0.25)", width=1),
            name="BB Mid", showlegend=False), row=1, col=1)

    # Moving Averages
    ma_palette = [
        ("EMA9",   "#fbbf24", 1.0), ("EMA20",  "#fb923c", 1.3),
        ("EMA50",  "#a78bfa", 1.6), ("SMA100", "#38bdf8", 1.3),
        ("SMA200", "#f43f5e", 1.8),
    ]
    for col, color, w in ma_palette:
        if col in df.columns and df[col].notna().sum() > 3:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                line=dict(color=color, width=w), opacity=0.85), row=1, col=1)

    # VWAP (intraday only)
    if interval in INTRADAY_IVLS and "VWAP" in df.columns and df["VWAP"].notna().sum() > 3:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#22d3ee", width=1.8, dash="dot")), row=1, col=1)

    # Volume
    if has_vol:
        vc = ["#00e676" if float(c) >= float(o) else "#ff4444"
              for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
            marker_color=vc, name="Volume", showlegend=False), row=2, col=1)
        vol_ma = df["Volume"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=vol_ma,
            line=dict(color="#fbbf24", width=1.1), showlegend=False), row=2, col=1)

    # RSI
    r = 3 if has_vol else 2
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color="#a78bfa", width=1.5), name="RSI",
        fill="tozeroy", fillcolor="rgba(167,139,250,0.05)", showlegend=False), row=r, col=1)
    for lvl, clr in [(70,"#ff4444"),(50,"#2c5070"),(30,"#00e676")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, line_width=1, row=r, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,68,68,0.04)",
                  line_width=0, row=r, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,230,118,0.04)",
                  line_width=0, row=r, col=1)

    # MACD
    m = 4 if has_vol else 3
    hc = ["#00e676" if float(v) >= 0 else "#ff4444"
          for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
        marker_color=hc, name="Hist", showlegend=False), row=m, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
        line=dict(color="#38bdf8", width=1.5), showlegend=False), row=m, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Sig"],
        line=dict(color="#fbbf24", width=1.5), showlegend=False), row=m, col=1)

    # ADX
    a = 5 if has_vol else 4
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX"],
        line=dict(color="#fb923c", width=1.6), showlegend=False), row=a, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["+DI"],
        line=dict(color="#00e676", width=1, dash="dot"), showlegend=False), row=a, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["-DI"],
        line=dict(color="#ff4444", width=1, dash="dot"), showlegend=False), row=a, col=1)
    fig.add_hline(y=25, line_dash="dot", line_color="#1e4060", line_width=1, row=a, col=1)

    fig.update_layout(
        title=dict(text=f"📈  Technical Analysis — {name}",
                   font=dict(size=16, color="#b8cfe8"), x=0.01),
        paper_bgcolor=PLOT_BG, plot_bgcolor=GRID_BG,
        font=dict(color="#4a7090", size=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9.5, color="#6a90b0")),
        height=920, margin=dict(t=65, l=55, r=15, b=30),
        hovermode="x unified",
    )
    for i in range(1, 6):
        fig.update_xaxes(gridcolor="#06101c", zerolinecolor="#07141f", row=i, col=1)
        fig.update_yaxes(gridcolor="#06101c", zerolinecolor="#07141f", row=i, col=1)
    return fig

# ──────────────────────────────────────────────────────────────────
# OPTIONS HELPERS
# ──────────────────────────────────────────────────────────────────
def calc_max_pain(chain) -> float:
    try:
        c = chain.calls[["strike","openInterest"]].dropna()
        p = chain.puts[["strike","openInterest"]].dropna()
        strikes = sorted(set(c["strike"].tolist() + p["strike"].tolist()))
        pain = []
        for s in strikes:
            cp = float(((c["strike"] < s)*(s - c["strike"])*c["openInterest"]).sum())
            pp = float(((p["strike"] > s)*(p["strike"] - s)*p["openInterest"]).sum())
            pain.append(cp + pp)
        return float(strikes[int(np.argmin(pain))]) if pain else 0.0
    except Exception:
        return 0.0


def analyze_chain(chain, spot: float) -> dict:
    c = chain.calls.copy()
    p = chain.puts.copy()
    tot_c = float(c["openInterest"].sum())
    tot_p = float(p["openInterest"].sum())
    pcr = tot_p / tot_c if tot_c > 0 else 0.0

    all_s = sorted(set(c["strike"].tolist() + p["strike"].tolist()))
    atm   = float(min(all_s, key=lambda x: abs(x - spot))) if all_s else spot

    ac = c[c["strike"] == atm]; ap = p[p["strike"] == atm]
    ce_p  = float(ac["lastPrice"].values[0]) if len(ac) and "lastPrice" in ac else 0.0
    pe_p  = float(ap["lastPrice"].values[0]) if len(ap) and "lastPrice" in ap else 0.0
    ce_iv = float(ac["impliedVolatility"].values[0]) if len(ac) and "impliedVolatility" in ac else 0.0
    pe_iv = float(ap["impliedVolatility"].values[0]) if len(ap) and "impliedVolatility" in ap else 0.0

    straddle = ce_p + pe_p
    atm_iv   = (ce_iv + pe_iv) / 2
    mp       = calc_max_pain(chain)
    exp_move = straddle * 0.68  # ±1 std dev approximation

    return dict(atm=atm, ce_prem=ce_p, pe_prem=pe_p, straddle=straddle,
                pcr=pcr, tot_c_oi=tot_c, tot_p_oi=tot_p,
                atm_iv=atm_iv, max_pain=mp, exp_move=exp_move, spot=spot)


def straddle_chart(chain, spot: float, name: str) -> go.Figure:
    c = chain.calls; p = chain.puts
    mc = c[["strike","lastPrice","openInterest"]].copy()
    mp2 = p[["strike","lastPrice","openInterest"]].copy()
    mc.columns = ["Strike","CE","CE_OI"]
    mp2.columns = ["Strike","PE","PE_OI"]
    df = mc.merge(mp2, on="Strike").sort_values("Strike")
    df["Straddle"] = df["CE"] + df["PE"]
    atm_idx = int((df["Strike"] - spot).abs().idxmin())
    pos = df.index.get_loc(atm_idx)
    df = df.iloc[max(0, pos - 10): pos + 11]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38],
        subplot_titles=["Option Premiums vs Strike","Open Interest"],
        shared_xaxes=True, vertical_spacing=0.08)

    for col, color, dash in [("CE","#00e676","solid"),("PE","#ff4444","solid"),
                               ("Straddle","#fbbf24","dot")]:
        fig.add_trace(go.Scatter(x=df["Strike"], y=df[col], name=col,
            line=dict(color=color, width=2, dash=dash), mode="lines+markers",
            marker=dict(size=5)), row=1, col=1)

    fig.add_vline(x=spot, line_dash="dash", line_color="#00e5ff", line_width=1.5)
    fig.add_annotation(x=spot, y=float(df["Straddle"].max()) * 1.05,
        text=f"Spot {spot:.2f}", font=dict(color="#00e5ff", size=10),
        showarrow=False, row=1, col=1)

    fig.add_trace(go.Bar(x=df["Strike"], y=df["CE_OI"],
        name="CE OI", marker_color="rgba(0,230,118,0.55)"), row=2, col=1)
    fig.add_trace(go.Bar(x=df["Strike"], y=df["PE_OI"],
        name="PE OI", marker_color="rgba(255,68,68,0.55)"), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"📊 Options Straddle Analysis — {name}",
                   font=dict(size=15, color="#b8cfe8"), x=0.01),
        paper_bgcolor=PLOT_BG, plot_bgcolor=GRID_BG,
        font=dict(color="#4a7090", size=10),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        height=540, margin=dict(t=60, l=55, r=15, b=30),
        barmode="group", hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#06101c"); fig.update_yaxes(gridcolor="#06101c")
    return fig


def straddle_insights(s: dict) -> list:
    ins = []
    spot, atm, prem = s["spot"], s["atm"], s["straddle"]
    pcr, mp, iv     = s["pcr"], s["max_pain"], s["atm_iv"] * 100
    exp             = s["exp_move"]

    ins.append(("info",
        f"🎯 <b>ATM Strike:</b> {atm:.0f} | <b>Spot:</b> {spot:.2f} | "
        f"<b>Straddle Premium:</b> {prem:.2f} | <b>ATM IV:</b> {iv:.1f}%"))
    ins.append(("info",
        f"📐 <b>Expected Move (±1σ, 68% probability):</b> ±{exp:.2f} "
        f"({exp / spot * 100:.1f}%) → Range: <b>{spot - exp:.2f}</b> to <b>{spot + exp:.2f}</b>"))

    if pcr > 1.3:
        ins.append(("bull",
            f"🐂 <b>PCR = {pcr:.2f}</b> (Bullish). Heavy put writing signals market makers "
            f"expect support. Bulls in control at option seller level."))
    elif pcr < 0.7:
        ins.append(("bear",
            f"🐻 <b>PCR = {pcr:.2f}</b> (Bearish). Heavy call writing signals resistance "
            f"overhead. Bears in control at option seller level."))
    else:
        ins.append(("warn",
            f"⚖️ <b>PCR = {pcr:.2f}</b> — Balanced. No strong directional bias from option writers."))

    mp_diff = mp - spot
    ins.append(("info",
        f"🧲 <b>Max Pain = {mp:.2f}</b> — Price tends to gravitate here near expiry "
        f"(option sellers' sweet spot). Delta from spot: {mp_diff:+.2f} ({mp_diff / spot * 100:+.1f}%)."))

    if abs(mp_diff / spot) < 0.005:
        ins.append(("warn",
            "⚡ <b>Spot ≈ Max Pain</b>. Expect pinning action. "
            "Short straddle/strangle sellers benefit from time decay here."))
    elif mp_diff > 0:
        ins.append(("bull",
            f"📈 Max Pain is <b>above spot</b> by {mp_diff:.2f}. "
            f"Price may drift UP toward {mp:.2f} as expiry approaches."))
    else:
        ins.append(("bear",
            f"📉 Max Pain is <b>below spot</b> by {abs(mp_diff):.2f}. "
            f"Price may drift DOWN toward {mp:.2f} as expiry approaches."))

    if iv > 35:
        ins.append(("bear",
            f"⚡ <b>High IV ({iv:.1f}%)</b>. Premiums are expensive — favour option <b>selling</b> "
            f"strategies (short straddle, iron condor, short strangle)."))
    elif iv < 20:
        ins.append(("bull",
            f"😴 <b>Low IV ({iv:.1f}%)</b>. Premiums are cheap — favour option <b>buying</b> "
            f"strategies (long straddle/strangle, buying calls/puts)."))
    else:
        ins.append(("info",
            f"📊 IV at {iv:.1f}% is moderate. Both buying and selling strategies are viable; "
            f"directional bias should drive selection."))

    # Mean reversion signal
    if abs(mp_diff / spot) > 0.02:
        ins.append(("warn",
            f"🔄 <b>Mean Reversion Likely</b> toward Max Pain ({mp:.2f}) as expiry nears. "
            f"Current deviation ({abs(mp_diff / spot) * 100:.1f}%) is significant."))

    return ins

# ──────────────────────────────────────────────────────────────────
# INTRADAY SIGNAL
# ──────────────────────────────────────────────────────────────────
def run_intraday_analysis(ticker: str) -> dict:
    """Multi-timeframe analysis with 1-second delay between fetches."""
    timeframes = [
        ("5m",  "1mo",  "5-Min"),
        ("15m", "1mo",  "15-Min"),
        ("1h",  "3mo",  "1-Hour"),
        ("1d",  "6mo",  "Daily"),
    ]
    tf_signals = {}
    prog = st.progress(0, "Initialising multi-timeframe scan…")

    for i, (iv, per, lbl) in enumerate(timeframes):
        prog.progress((i + 1) / len(timeframes), f"Scanning {lbl} timeframe…")
        try:
            time.sleep(1)
            raw = yf.download(ticker, interval=iv, period=per,
                              progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 30:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna(subset=["Close"])
            raw = calc_indicators(raw)
            last = raw.iloc[-1]

            def gv(col):
                return _g(last, col)

            close = gv("Close"); rsi   = gv("RSI");  macd  = gv("MACD")
            msig  = gv("MACD_Sig"); ema20 = gv("EMA20"); ema50 = gv("EMA50")
            adx   = gv("ADX");  pdi   = gv("+DI"); ndi  = gv("-DI")
            atr   = gv("ATR");  bb_p  = gv("BB_Pct"); stk  = gv("Stoch_K")

            score = 0; reasons = []

            if close and ema20:
                if close > ema20: score += 1; reasons.append(f"Price > EMA20")
                else:             score -= 1; reasons.append(f"Price < EMA20")
            if ema20 and ema50:
                if ema20 > ema50: score += 1; reasons.append("EMA20 > EMA50")
                else:             score -= 1; reasons.append("EMA20 < EMA50")
            if macd is not None and msig is not None:
                if macd > msig: score += 1; reasons.append("MACD bullish cross")
                else:           score -= 1; reasons.append("MACD bearish cross")
            if rsi is not None:
                if 50 < rsi < 70:   score += 1; reasons.append(f"RSI bullish ({rsi:.0f})")
                elif rsi >= 70:     score -= 1; reasons.append(f"RSI overbought ({rsi:.0f})")
                elif 30 < rsi <= 50: score -= 1; reasons.append(f"RSI bearish ({rsi:.0f})")
                elif rsi <= 30:     score += 1; reasons.append(f"RSI oversold bounce ({rsi:.0f})")
            if adx and pdi and ndi:
                if adx > 25 and pdi > ndi: score += 1; reasons.append(f"ADX trend up ({adx:.0f})")
                elif adx > 25:             score -= 1; reasons.append(f"ADX trend dn ({adx:.0f})")
            if bb_p is not None:
                if bb_p < 0.20:   score += 1; reasons.append("Near lower BB (buy zone)")
                elif bb_p > 0.80: score -= 1; reasons.append("Near upper BB (sell zone)")
            if stk is not None:
                if stk < 20:   score += 1; reasons.append(f"Stoch oversold ({stk:.0f})")
                elif stk > 80: score -= 1; reasons.append(f"Stoch overbought ({stk:.0f})")

            tf_signals[lbl] = dict(
                score=score, reasons=reasons,
                close=close, rsi=rsi, atr=atr, adx=adx, ema20=ema20,
            )
        except Exception as ex:
            tf_signals[lbl] = dict(score=0, reasons=[f"Error: {ex}"],
                                   close=None, rsi=None, atr=None, adx=None, ema20=None)

    prog.empty()
    if not tf_signals:
        return {}

    total_score = sum(v["score"] for v in tf_signals.values())
    base = (tf_signals.get("15-Min") or tf_signals.get("5-Min")
            or list(tf_signals.values())[0])
    price = base.get("close") or 0.0
    atr   = base.get("atr") or price * 0.005

    if total_score >= 4:
        direction, emoji = "LONG",  "🟢"
    elif total_score <= -4:
        direction, emoji = "SHORT", "🔴"
    else:
        direction, emoji = "NEUTRAL / WAIT", "🟡"

    if direction == "LONG":
        entry = price; sl = price - 1.5*atr
        t1 = price + 1.5*atr; t2 = price + 2.5*atr; t3 = price + 4.0*atr
    elif direction == "SHORT":
        entry = price; sl = price + 1.5*atr
        t1 = price - 1.5*atr; t2 = price - 2.5*atr; t3 = price - 4.0*atr
    else:
        entry = price; sl = price - atr; t1 = price + atr; t2 = t3 = None

    rsi_vals = [v["rsi"] for v in tf_signals.values() if v.get("rsi") is not None]
    adx_vals = [v["adx"] for v in tf_signals.values() if v.get("adx") is not None]
    avg_rsi  = float(np.mean(rsi_vals)) if rsi_vals else 50.0
    avg_adx  = float(np.mean(adx_vals)) if adx_vals else 20.0
    eligible = abs(total_score) >= 4 and 15 < avg_rsi < 85 and avg_adx > 18

    return dict(
        direction=direction, emoji=emoji, total_score=total_score,
        entry=round(entry,2), sl=round(sl,2),
        t1=round(t1,2) if t1 else None,
        t2=round(t2,2) if t2 else None,
        t3=round(t3,2) if t3 else None,
        eligible=eligible, price=price, atr=atr,
        tf_signals=tf_signals, avg_rsi=avg_rsi, avg_adx=avg_adx,
    )

# ──────────────────────────────────────────────────────────────────
# HEATMAPS
# ──────────────────────────────────────────────────────────────────
def monthly_heatmap(daily: pd.DataFrame, name: str) -> go.Figure:
    df = daily.copy()
    df.index = pd.to_datetime(df.index)
    df["_yr"] = df.index.year
    df["_mo"] = df.index.month

    # Last close of each (year, month) — sorted so pct_change is chronological
    monthly_close = (
        df.groupby(["_yr", "_mo"])["Close"]
        .last()
        .sort_index()          # ensures Jan follows Dec of prior year
    )
    # Month-over-month return: Jan correctly = Jan_close / Dec_prev_close - 1
    monthly_ret = monthly_close.pct_change() * 100
    # Reshape to year × month matrix
    mr = monthly_ret.unstack(level="_mo")   # index=year, columns=1..12
    years = sorted(mr.index.tolist())

    def _safe(yr, m):
        try:
            if yr not in mr.index or m not in mr.columns:
                return np.nan
            v = mr.at[yr, m]
            return float(v) if pd.notna(v) else np.nan
        except Exception:
            return np.nan

    z, txt = [], []
    for yr in years:
        row, trow = [], []
        for m in range(1, 13):
            v = _safe(yr, m)
            row.append(None if np.isnan(v) else v)
            trow.append(f"{v:.1f}%" if not np.isnan(v) else "")
        z.append(row); txt.append(trow)

    fig = go.Figure(go.Heatmap(
        z=z, x=list(MONTH_NAMES.values()), y=[str(y) for y in years],
        colorscale=[[0,"#7f1d1d"],[0.5,"#0a1428"],[1,"#064e3b"]],
        zmid=0, zmin=-12, zmax=12,
        text=txt, texttemplate="%{text}",
        textfont=dict(size=10.5, color="white"),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return %", ticksuffix="%",
                      title_font=dict(color="#6a90b0"), tickfont=dict(color="#6a90b0")),
    ))
    fig.update_layout(
        title=dict(text=f"📅 Monthly Returns Heatmap — {name}",
                   font=dict(size=15, color="#b8cfe8"), x=0.01),
        xaxis=dict(title="Month", tickfont=dict(color="#6a90b0")),
        yaxis=dict(title="Year", autorange="reversed", tickfont=dict(color="#6a90b0")),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        height=max(300, len(years) * 30 + 110),
        margin=dict(t=55, l=55, r=20, b=50),
    )
    return fig


def dow_heatmap(daily: pd.DataFrame, name: str) -> go.Figure:
    df = daily.copy()
    df.index = pd.to_datetime(df.index)
    df["_dow"] = df.index.dayofweek
    df["_mo"]  = df.index.month
    df["_ret"] = df["Close"].pct_change() * 100
    piv = df.groupby(["_dow", "_mo"])["_ret"].mean().unstack()

    def _safe(d, m):
        try:
            if d not in piv.index or m not in piv.columns:
                return np.nan
            v = piv.at[d, m]
            return float(v) if pd.notna(v) else np.nan
        except Exception:
            return np.nan

    z, txt = [], []
    for d in range(5):
        row, trow = [], []
        for m in range(1, 13):
            v = _safe(d, m)
            row.append(None if np.isnan(v) else v)
            trow.append(f"{v:.2f}%" if not np.isnan(v) else "")
        z.append(row); txt.append(trow)

    fig = go.Figure(go.Heatmap(
        z=z, x=list(MONTH_NAMES.values()),
        y=["Monday","Tuesday","Wednesday","Thursday","Friday"],
        colorscale=[[0,"#7f1d1d"],[0.5,"#0a1428"],[1,"#064e3b"]],
        zmid=0, text=txt, texttemplate="%{text}",
        textfont=dict(size=10.5, color="white"),
        hovertemplate="Day: %{y}<br>Month: %{x}<br>Avg Ret: %{z:.3f}%<extra></extra>",
        colorbar=dict(title="Avg Ret %", ticksuffix="%",
                      title_font=dict(color="#6a90b0"), tickfont=dict(color="#6a90b0")),
    ))
    fig.update_layout(
        title=dict(text=f"📊 Average Daily Return — Weekday × Month — {name}",
                   font=dict(size=15, color="#b8cfe8"), x=0.01),
        xaxis=dict(title="Month", tickfont=dict(color="#6a90b0")),
        yaxis=dict(title="Day of Week", tickfont=dict(color="#6a90b0")),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        height=340, margin=dict(t=55, l=95, r=20, b=50),
    )
    return fig

# ──────────────────────────────────────────────────────────────────
# HELPERS FOR DISPLAY
# ──────────────────────────────────────────────────────────────────
def fmt_ohlc(df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    d = df.tail(n).copy()
    for col in ["Open","High","Low","Close"]:
        d[col] = d[col].astype(float).round(2)
    d["Chg ($)"] = d["Close"].diff().round(2)
    d["Chg (%)"] = (d["Close"].pct_change() * 100).round(2)
    if "Volume" in d.columns:
        d["Volume"] = d["Volume"].apply(
            lambda x: f"{float(x):,.0f}" if pd.notna(x) and float(x) > 0 else "—")
    try:
        d.index = d.index.strftime("%Y-%m-%d %H:%M" if hasattr(d.index[0], "hour") else "%Y-%m-%d")
    except Exception:
        pass
    cols = [c for c in ["Open","High","Low","Close","Volume","Chg ($)","Chg (%)"] if c in d.columns]
    return d[cols].iloc[::-1]


def color_val(val):
    try:
        v = float(str(val).replace("%","").replace("+","").replace(",",""))
        if v > 0: return "color: #00e676; font-weight:600"
        if v < 0: return "color: #ff4444; font-weight:600"
        return "color: #4a7090"
    except Exception:
        return ""


def pcard(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="pcard"><div class="pcard-lbl">{label}</div>'
            f'<div class="pcard-val {cls}">{value}</div></div>')


def ins(kind: str, text: str) -> str:
    return f'<div class="ins {kind}">{text}</div>'

# ──────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="app-title">AlgoTrader Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Real-Time Market Intelligence · Multi-Timeframe · Options · Heatmaps</div>',
                unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        st.markdown('<div class="div"></div>', unsafe_allow_html=True)

        ticker_label = st.selectbox("📌 Instrument", list(TICKER_MAP.keys()), index=0)
        ticker = TICKER_MAP[ticker_label]
        if ticker == "CUSTOM":
            ct = st.text_input("Enter Symbol", placeholder="e.g. AAPL, TCS.NS, MSFT")
            ticker = ct.strip().upper() if ct.strip() else "AAPL"
            ticker_label = f"Custom: {ticker}"

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)

        interval = st.selectbox("⏱️ Interval", list(INTERVAL_PERIODS.keys()),
                                 format_func=lambda x: f"{x}  —  {INTERVAL_LABELS[x]}", index=6)
        period_list = INTERVAL_PERIODS[interval]
        period = st.selectbox("📅 Period", period_list,
                               index=min(3, len(period_list)-1))

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        if st.button("🔄  Refresh All Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        is_indian = ticker in INDIAN_TICKERS
        if is_indian:
            st.warning("🇮🇳 Indian index: options data unavailable via yfinance.")
        else:
            st.info("ℹ️ Options available for US-listed tickers.")

        st.markdown("---")
        st.caption("4h interval = resampled from 1h | "
                   "1-second delay between API calls (yfinance rate limit)")

    # ── Fetch Primary Data ──────────────────────────────────
    with st.spinner("📡 Fetching market data…"):
        df = fetch_ohlc(ticker, interval, period)

    if df is None or df.empty:
        st.error(f"❌ No data for **{ticker}**. Check the symbol and try again.")
        return

    df = calc_indicators(df)
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    chg_abs    = last_close - prev_close
    chg_pct    = chg_abs / prev_close * 100 if prev_close else 0.0
    arrow      = "▲" if chg_abs >= 0 else "▼"
    dir_cls    = "up" if chg_abs >= 0 else "dn"

    info = fetch_info(ticker)

    # ── Top Metrics ─────────────────────────────────────────
    cols6 = st.columns(6)
    hi52  = info.get("fiftyTwoWeekHigh")
    lo52  = info.get("fiftyTwoWeekLow")
    has_v = "Volume" in df.columns and float(df["Volume"].iloc[-1]) > 0

    cards = [
        ("Last Price",  f"{last_close:,.2f}",                               ""),
        ("Change",      f"{arrow} {abs(chg_abs):,.2f}",                     dir_cls),
        ("Change %",    f"{arrow} {abs(chg_pct):.2f}%",                     dir_cls),
        ("52W High",    f"{hi52:,.2f}" if hi52 else "—",                    ""),
        ("52W Low",     f"{lo52:,.2f}" if lo52 else "—",                    ""),
        ("Volume",      f"{float(df['Volume'].iloc[-1]):,.0f}" if has_v else "—", ""),
    ]
    for col, (lbl, val, cls) in zip(cols6, cards):
        col.markdown(pcard(lbl, val, cls), unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📋  Price & Ranges",
        "📈  Technical Snapshot",
        "🎯  Options & Straddle",
        "🚦  Intraday Intelligence",
        "🗓️   Return Heatmaps",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — OHLC + RANGE ANALYSIS
    # ════════════════════════════════════════════════════════
    with t1:
        st.markdown('<div class="sh">📋 Recent OHLC Data</div>', unsafe_allow_html=True)
        ohlc_disp = fmt_ohlc(df, 40)
        change_cols = [c for c in ["Chg ($)","Chg (%)"] if c in ohlc_disp.columns]
        fmt_map = {}
        for col in ["Open","High","Low","Close"]:
            if col in ohlc_disp.columns:
                fmt_map[col] = "{:,.2f}"
        for col in change_cols:
            fmt_map[col] = "{:+,.2f}" if "$" in col else "{:+,.2f}%"

        styled_ohlc = (
            ohlc_disp.style
            .map(color_val, subset=change_cols)
            .format(fmt_map, na_rep="—")
            .set_properties(**{"background-color":"#071828","color":"#b8cfe8","border-color":"#0e2a42"})
            .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),
                                                         ("color","#4a7090"),
                                                         ("font-size","0.78rem"),
                                                         ("text-transform","uppercase"),
                                                         ("letter-spacing","1px")]}])
        )
        st.dataframe(styled_ohlc, use_container_width=True, height=430)

        # Range analysis
        st.markdown('<div class="sh">📏 Historical Range Analysis & Breakout Detection</div>',
                    unsafe_allow_html=True)
        with st.spinner("Loading daily data for range analysis…"):
            daily_df = fetch_daily(ticker, "2y")

        if daily_df is not None and len(daily_df) >= 10:
            ranges = compute_ranges(daily_df)

            if not ranges.empty:
                today_c = float(daily_df["Close"].iloc[-1])

                # Visual range chart
                rng_fig = go.Figure()
                for _, row in ranges.iterrows():
                    color = "#ff4444" if row["_v"] else "#00e676"
                    rng_fig.add_trace(go.Bar(
                        x=[row["Range High"] - row["Range Low"]],
                        y=[row["Period"]],
                        base=[row["Range Low"]],
                        orientation="h",
                        marker=dict(
                            color=f"rgba({'255,68,68' if row['_v'] else '0,230,118'},0.25)",
                            line=dict(color=color, width=1.2)
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{row['Period']}</b><br>"
                            f"High: {row['Range High']:,.2f}<br>"
                            f"Low:  {row['Range Low']:,.2f}<br>"
                            f"Range: {row['Range (pts)']:,.2f} ({row['Range (%)']:.1f}%)<br>"
                            f"Status: {row['Status']}<extra></extra>"
                        ),
                    ))

                rng_fig.add_vline(x=today_c, line_dash="dash",
                                  line_color="#ffc107", line_width=2)
                rng_fig.add_annotation(
                    x=today_c, y=1.0, yref="paper",
                    text=f" Today: {today_c:,.2f}",
                    font=dict(color="#ffc107", size=11),
                    showarrow=False, xanchor="left",
                )
                rng_fig.update_layout(
                    title=dict(text="Price vs Historical Ranges (green = within, red = breakout)",
                               font=dict(size=13, color="#b8cfe8"), x=0.01),
                    paper_bgcolor=PLOT_BG, plot_bgcolor=GRID_BG,
                    font=dict(color="#4a7090", size=10),
                    height=520, margin=dict(t=48, l=90, r=20, b=30),
                    xaxis=dict(title="Price Level", gridcolor="#06101c"),
                    yaxis=dict(gridcolor="#06101c", autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(rng_fig, use_container_width=True)

                # Three sub-tables
                c_day, c_wk, c_mo = st.columns([1, 0.9, 1.2])

                def range_table(mask, col):
                    sub = ranges[mask].drop(columns=["_v"])
                    if sub.empty:
                        col.warning("No data")
                        return
                    sub_styled = (
                        sub.style
                        .format({
                            "Range High":   "{:,.2f}", "Range Low":    "{:,.2f}",
                            "Range (pts)":  "{:,.2f}", "Range (%)":    "{:.2f}%",
                            "Today Close":  "{:,.2f}", "Position (%)": "{:.1f}%",
                        })
                        .set_properties(**{"background-color":"#071828","color":"#b8cfe8"})
                        .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),
                                                                      ("color","#4a7090"),
                                                                      ("font-size","0.73rem")]}])
                    )
                    col.dataframe(sub_styled, use_container_width=True)

                with c_day:
                    st.markdown("##### 📅 Day Ranges (2 – 5 Days)")
                    range_table(ranges["Period"].str.contains("Day"), c_day)

                with c_wk:
                    st.markdown("##### 📅 Week Ranges (2 – 3 Weeks)")
                    range_table(ranges["Period"].str.contains("Week"), c_wk)

                with c_mo:
                    st.markdown("##### 🗓️ Month Ranges (1 – 12 Months)")
                    range_table(ranges["Period"].str.contains("Month"), c_mo)

                # Summary insight
                violations = ranges[ranges["_v"]]
                if not violations.empty:
                    st.markdown(ins("bear",
                        f"🔴 <b>Breakout Detected!</b> Today's close ({today_c:,.2f}) has broken "
                        f"outside <b>{len(violations)}</b> historical range(s): "
                        f"{', '.join(violations['Period'].tolist())}. "
                        "This signals significant momentum. Confirm with volume and trend indicators."),
                        unsafe_allow_html=True)
                else:
                    st.markdown(ins("bull",
                        f"🟢 <b>Range Intact.</b> Today's close ({today_c:,.2f}) remains within "
                        f"all {len(ranges)} analysed ranges. Price is consolidating — "
                        "no major breakout detected."),
                        unsafe_allow_html=True)
        else:
            st.warning("⚠️ Insufficient daily data for range analysis.")

    # ════════════════════════════════════════════════════════
    # TAB 2 — TECHNICAL SNAPSHOT
    # ════════════════════════════════════════════════════════
    with t2:
        st.markdown('<div class="sh">📈 Technical Indicators Chart</div>', unsafe_allow_html=True)
        fig_tech = build_tech_chart(df, ticker_label, interval)
        st.plotly_chart(fig_tech, use_container_width=True)

        # ── Indicator value table
        st.markdown('<div class="sh">📊 Current Indicator Values</div>', unsafe_allow_html=True)
        last = df.iloc[-1]
        cp = float(last["Close"])

        def fv(col, dec=2):
            v = _g(last, col)
            return f"{v:,.{dec}f}" if v is not None else "—"

        def sig_ma(col):
            v = _g(last, col)
            if v is None: return "—"
            return "🟢 Above" if cp > v else "🔴 Below"

        ic1, ic2, ic3 = st.columns(3)

        with ic1:
            st.markdown("**Moving Averages vs Price**")
            ma_rows = []
            for p_val in [9, 15, 20, 50, 100, 200]:
                for prefix in ["SMA","EMA"]:
                    col_name = f"{prefix}{p_val}"
                    v = _g(last, col_name)
                    ma_rows.append({
                        "MA": col_name,
                        "Value": f"{v:,.2f}" if v else "—",
                        "Signal": sig_ma(col_name),
                        "Distance %": f"{(cp/v-1)*100:+.2f}%" if v else "—",
                    })
            ma_df = pd.DataFrame(ma_rows)
            st.dataframe(ma_df.style
                .set_properties(**{"background-color":"#071828","color":"#b8cfe8"})
                .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),("color","#4a7090"),("font-size","0.73rem")]}]),
                use_container_width=True, height=460)

        with ic2:
            st.markdown("**Momentum & Oscillators**")
            def rsig(col, ob=70, os=30):
                v = _g(last, col)
                if v is None: return "—"
                if v >= ob: return "🔴 Overbought"
                if v <= os: return "🟢 Oversold"
                return "⚪ Neutral"

            mom_rows = [
                {"Indicator":"RSI (14)",    "Value":fv("RSI",1),       "Signal":rsig("RSI")},
                {"Indicator":"Stoch %K",    "Value":fv("Stoch_K",1),   "Signal":rsig("Stoch_K",80,20)},
                {"Indicator":"Stoch %D",    "Value":fv("Stoch_D",1),   "Signal":rsig("Stoch_D",80,20)},
                {"Indicator":"MACD",        "Value":fv("MACD",4),
                 "Signal":"🟢 Bullish" if (_g(last,"MACD") or 0)>0 else "🔴 Bearish"},
                {"Indicator":"MACD Signal", "Value":fv("MACD_Sig",4),
                 "Signal":"🟢 Buy" if (_g(last,"MACD") or 0) > (_g(last,"MACD_Sig") or 0) else "🔴 Sell"},
                {"Indicator":"MACD Hist",   "Value":fv("MACD_Hist",4),
                 "Signal":"🟢 +" if (_g(last,"MACD_Hist") or 0)>=0 else "🔴 −"},
                {"Indicator":"BB %B",       "Value":fv("BB_Pct",3),
                 "Signal":"🔴 Upper" if (_g(last,"BB_Pct") or 0.5)>0.8
                           else "🟢 Lower" if (_g(last,"BB_Pct") or 0.5)<0.2 else "⚪ Mid"},
                {"Indicator":"BB Upper",    "Value":fv("BB_Up"),        "Signal":"Resistance"},
                {"Indicator":"BB Mid",      "Value":fv("BB_Mid"),       "Signal":"Mean"},
                {"Indicator":"BB Lower",    "Value":fv("BB_Dn"),        "Signal":"Support"},
                {"Indicator":"ATR (14)",    "Value":fv("ATR"),
                 "Signal":f"{(_g(last,'ATR') or 0)/cp*100:.2f}% of price" if cp else "—"},
                {"Indicator":"VWAP",        "Value":fv("VWAP"),
                 "Signal":"🟢 Above" if cp>(_g(last,"VWAP") or 0) else "🔴 Below"},
            ]
            st.dataframe(pd.DataFrame(mom_rows).style
                .set_properties(**{"background-color":"#071828","color":"#b8cfe8"})
                .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),("color","#4a7090"),("font-size","0.73rem")]}]),
                use_container_width=True, height=460)

        with ic3:
            st.markdown("**Trend & Pivot Levels**")
            adx_v = _g(last,"ADX"); pdi_v = _g(last,"+DI"); ndi_v = _g(last,"-DI")
            if adx_v and pdi_v and ndi_v:
                trend_note = (f"{'Strong' if adx_v>25 else 'Weak'} "
                              f"{'Bullish' if pdi_v>ndi_v else 'Bearish'}")
            else:
                trend_note = "—"

            trend_rows = [
                {"Level":"ADX (14)",   "Value":fv("ADX",1),
                 "Note":"<25 Ranging · >25 Trending · >40 Strong"},
                {"Level":"+DI",        "Value":fv("+DI",1), "Note":"Bullish direction indicator"},
                {"Level":"-DI",        "Value":fv("-DI",1), "Note":"Bearish direction indicator"},
                {"Level":"Trend",      "Value":trend_note,  "Note":"ADX+DI assessment"},
                {"Level":"Pivot (PP)", "Value":fv("PP"),    "Note":"Standard pivot"},
                {"Level":"R1",         "Value":fv("R1"),    "Note":"Resistance 1"},
                {"Level":"R2",         "Value":fv("R2"),    "Note":"Resistance 2"},
                {"Level":"R3",         "Value":fv("R3"),    "Note":"Resistance 3"},
                {"Level":"S1",         "Value":fv("S1"),    "Note":"Support 1"},
                {"Level":"S2",         "Value":fv("S2"),    "Note":"Support 2"},
                {"Level":"S3",         "Value":fv("S3"),    "Note":"Support 3"},
            ]
            st.dataframe(pd.DataFrame(trend_rows).style
                .set_properties(**{"background-color":"#071828","color":"#b8cfe8"})
                .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),("color","#4a7090"),("font-size","0.73rem")]}]),
                use_container_width=True, height=420)

        # ── Key insights
        st.markdown('<div class="sh">💡 Key Technical Insights</div>', unsafe_allow_html=True)
        insights = tech_insights(df)
        if insights:
            for kind, text in insights:
                st.markdown(ins(kind, text), unsafe_allow_html=True)
        else:
            st.info("Not enough data to generate insights.")

    # ════════════════════════════════════════════════════════
    # TAB 3 — OPTIONS & STRADDLE
    # ════════════════════════════════════════════════════════
    with t3:
        if is_indian:
            st.markdown(
                ins("warn",
                    "⚠️ <b>Options data unavailable for Indian indices via yfinance.</b><br>"
                    "For live Indian options data, use:<br>"
                    "• <a href='https://www.nseindia.com' target='_blank'>NSE India</a> (free) | "
                    "• <a href='https://sensibull.com' target='_blank'>Sensibull</a> | "
                    "• <a href='https://opstra.definedge.com' target='_blank'>Opstra</a> | "
                    "• Zerodha Kite / Upstox / AngelOne broker APIs"
                ), unsafe_allow_html=True)
        else:
            with st.spinner("Checking available option expiries…"):
                expiries = fetch_option_expiries(ticker)

            if not expiries:
                st.warning(
                    f"⚠️ No options data available for **{ticker}** via yfinance. "
                    "Options data is available for most US-listed stocks and ETFs.")
            else:
                st.markdown('<div class="sh">🎯 Straddle Premium Analysis</div>',
                            unsafe_allow_html=True)
                sel_exp = st.selectbox("📅 Select Expiry Date", expiries)

                with st.spinner(f"Loading option chain for {sel_exp}…"):
                    chain = fetch_chain(ticker, sel_exp)

                if chain is None:
                    st.error("Could not load option chain. Please try another expiry.")
                else:
                    spot = float(df["Close"].iloc[-1])
                    try:
                        stats = analyze_chain(chain, spot)

                        # Metric cards
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        for col, lbl, val, cls in [
                            (mc1, "Spot Price",     f"{stats['spot']:,.2f}",      ""),
                            (mc2, "ATM Strike",     f"{stats['atm']:.0f}",        ""),
                            (mc3, "Straddle Prem",  f"{stats['straddle']:.2f}",   "neu"),
                            (mc4, "PCR",            f"{stats['pcr']:.2f}",
                             "up" if stats["pcr"]>1.2 else "dn" if stats["pcr"]<0.8 else ""),
                            (mc5, "Max Pain",       f"{stats['max_pain']:.2f}",   ""),
                        ]:
                            col.markdown(pcard(lbl, val, cls), unsafe_allow_html=True)

                        mc6, mc7, mc8, mc9 = st.columns(4)
                        for col, lbl, val, cls in [
                            (mc6, "CE Premium",  f"{stats['ce_prem']:.2f}",        "up"),
                            (mc7, "PE Premium",  f"{stats['pe_prem']:.2f}",        "dn"),
                            (mc8, "ATM IV",      f"{stats['atm_iv']*100:.1f}%",    "neu"),
                            (mc9, "Exp Move ±",  f"{stats['exp_move']:.2f} "
                                                 f"({stats['exp_move']/spot*100:.1f}%)", ""),
                        ]:
                            col.markdown(pcard(lbl, val, cls), unsafe_allow_html=True)

                        st.markdown('<div class="div"></div>', unsafe_allow_html=True)

                        # Straddle chart
                        st.plotly_chart(straddle_chart(chain, spot, ticker_label),
                                        use_container_width=True)

                        # Options chain table
                        st.markdown('<div class="sh">📋 Options Chain — Near ATM (±8 strikes)</div>',
                                    unsafe_allow_html=True)
                        calls = chain.calls.copy(); puts = chain.puts.copy()
                        all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
                        atm_s = stats["atm"]
                        if atm_s in all_strikes:
                            ai = all_strikes.index(atm_s)
                            near = all_strikes[max(0, ai-8): ai+9]
                        else:
                            near = all_strikes[:17]

                        c_near = calls[calls["strike"].isin(near)].copy()
                        p_near = puts[puts["strike"].isin(near)].copy()

                        greek_c = [c for c in ["delta","gamma","theta"] if c in c_near.columns]
                        greek_p = [c for c in ["delta","gamma","theta"] if c in p_near.columns]

                        c_base = ["strike","lastPrice","openInterest","impliedVolatility","volume"] + greek_c
                        p_base = ["strike","lastPrice","openInterest","impliedVolatility","volume"] + greek_p
                        c_near = c_near[[col for col in c_base if col in c_near.columns]]
                        p_near = p_near[[col for col in p_base if col in p_near.columns]]

                        c_near.columns = (["Strike","CE Last","CE OI","CE IV","CE Vol"]
                                          + [f"CE {g.title()}" for g in greek_c])
                        p_near.columns = (["Strike","PE Last","PE OI","PE IV","PE Vol"]
                                          + [f"PE {g.title()}" for g in greek_p])

                        combo = c_near.merge(p_near, on="Strike", how="outer").sort_values("Strike")
                        combo["Straddle"] = combo.get("CE Last", 0) + combo.get("PE Last", 0)
                        if "CE OI" in combo: combo["CE OI"] = combo["CE OI"].fillna(0).astype(int)
                        if "PE OI" in combo: combo["PE OI"] = combo["PE OI"].fillna(0).astype(int)
                        if "CE IV" in combo: combo["CE IV"] = (combo["CE IV"] * 100).round(1)
                        if "PE IV" in combo: combo["PE IV"] = (combo["PE IV"] * 100).round(1)

                        def hl_atm(row):
                            if row.get("Strike") == atm_s:
                                return ["background-color:rgba(0,229,255,0.1);font-weight:700"] * len(row)
                            return [""] * len(row)

                        num_cols = [c for c in combo.columns
                                    if c not in ["Strike","CE OI","PE OI"]]
                        fmt2 = {c: "{:.2f}" for c in num_cols if c in combo}
                        if "CE OI" in combo: fmt2["CE OI"] = "{:,}"
                        if "PE OI" in combo: fmt2["PE OI"] = "{:,}"

                        st.dataframe(
                            combo.style.apply(hl_atm, axis=1)
                            .format(fmt2, na_rep="—")
                            .set_properties(**{"background-color":"#071828","color":"#b8cfe8"})
                            .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),
                                                                          ("color","#4a7090"),
                                                                          ("font-size","0.73rem")]}]),
                            use_container_width=True, height=360,
                        )

                        # PCR / OI Summary
                        st.markdown('<div class="sh">📊 OI · PCR · Max Pain Summary</div>',
                                    unsafe_allow_html=True)
                        pcr_rows = [
                            {"Metric":"Total Call OI",    "Value":f"{stats['tot_c_oi']:,.0f}",
                             "Interpretation":"Total call open interest across all strikes"},
                            {"Metric":"Total Put OI",     "Value":f"{stats['tot_p_oi']:,.0f}",
                             "Interpretation":"Total put open interest across all strikes"},
                            {"Metric":"PCR (OI-based)",   "Value":f"{stats['pcr']:.3f}",
                             "Interpretation":"< 0.7 Bearish | 0.7–1.2 Neutral | > 1.2 Bullish"},
                            {"Metric":"Max Pain",         "Value":f"{stats['max_pain']:.2f}",
                             "Interpretation":"Price at which total option pain is minimised (expiry magnet)"},
                            {"Metric":"ATM IV",           "Value":f"{stats['atm_iv']*100:.1f}%",
                             "Interpretation":"Implied Volatility at the at-the-money strike"},
                            {"Metric":"Expected Move ±1σ","Value":f"±{stats['exp_move']:.2f} ({stats['exp_move']/spot*100:.1f}%)",
                             "Interpretation":"68% probability price stays within this range by expiry"},
                            {"Metric":"Upper Exp Bound",  "Value":f"{spot+stats['exp_move']:.2f}",
                             "Interpretation":"Upper bound of 1σ expected range"},
                            {"Metric":"Lower Exp Bound",  "Value":f"{spot-stats['exp_move']:.2f}",
                             "Interpretation":"Lower bound of 1σ expected range"},
                        ]
                        st.dataframe(
                            pd.DataFrame(pcr_rows).style
                            .set_properties(**{"background-color":"#071828","color":"#b8cfe8"})
                            .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),
                                                                          ("color","#4a7090"),
                                                                          ("font-size","0.73rem")]}]),
                            use_container_width=True, height=330,
                        )

                        # Insights
                        st.markdown('<div class="sh">💡 Straddle & Options Intelligence</div>',
                                    unsafe_allow_html=True)
                        for kind, text in straddle_insights(stats):
                            st.markdown(ins(kind, text), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error analysing options chain: {e}")

    # ════════════════════════════════════════════════════════
    # TAB 4 — INTRADAY INTELLIGENCE
    # ════════════════════════════════════════════════════════
    with t4:
        st.markdown('<div class="sh">🚦 Multi-Timeframe Intraday Signal Engine</div>',
                    unsafe_allow_html=True)
        st.markdown(ins("info",
            "ℹ️ Analyses <b>5-Min · 15-Min · 1-Hour · Daily</b> timeframes with 1-second "
            "delay between each API request. Results are stored until you re-run."),
            unsafe_allow_html=True)

        if "intra_signal" not in st.session_state:
            st.session_state.intra_signal = None
        if "intra_ticker" not in st.session_state:
            st.session_state.intra_ticker = None

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button("🔍  Run Intraday Analysis", key="run_intra",
                                use_container_width=True)

        if run_btn:
            st.session_state.intra_signal = run_intraday_analysis(ticker)
            st.session_state.intra_ticker = ticker

        sig = st.session_state.intra_signal

        if sig is None:
            st.info("Click **Run Intraday Analysis** to generate a signal.")
        elif not sig:
            st.error("Could not generate signal — insufficient data for this ticker/timeframe.")
        else:
            if st.session_state.intra_ticker != ticker:
                st.warning("⚠️ Ticker has changed since last analysis. Re-run for updated signal.")

            direction  = sig["direction"]
            emoji      = sig["emoji"]
            score      = sig["total_score"]
            eligible   = sig["eligible"]
            elig_kind  = "bull" if eligible else "warn"
            elig_text  = "✅ ELIGIBLE for Intraday Trade" if eligible else "⚠️ NOT RECOMMENDED — Signal strength too low or conditions unfavourable"

            st.markdown(ins(elig_kind,
                f"{elig_text}<br>"
                f"Direction: <b>{emoji} {direction}</b> &nbsp;|&nbsp; "
                f"Composite Score: <b>{score:+d} / {len(sig['tf_signals'])*7}</b> &nbsp;|&nbsp; "
                f"Avg RSI: <b>{sig['avg_rsi']:.1f}</b> &nbsp;|&nbsp; "
                f"Avg ADX: <b>{sig['avg_adx']:.1f}</b>"),
                unsafe_allow_html=True)

            # TF breakdown
            st.markdown("#### 📊 Timeframe Signal Breakdown")
            tf_cols = st.columns(len(sig["tf_signals"]))
            for (lbl, data), col in zip(sig["tf_signals"].items(), tf_cols):
                sc = data["score"]
                kind = "bull" if sc > 1 else ("bear" if sc < -1 else "warn")
                ic = "🟢" if sc > 1 else ("🔴" if sc < -1 else "🟡")
                rsi_s = f"{data['rsi']:.0f}" if data.get("rsi") is not None else "—"
                adx_s = f"{data['adx']:.0f}" if data.get("adx") is not None else "—"
                reasons_html = "<br>".join(f"• {r}" for r in data.get("reasons", [])[:5])
                col.markdown(
                    f'<div class="ins {kind}" style="height:190px;overflow:auto">'
                    f'<b>{ic} {lbl}</b><br>'
                    f'Score: <b>{sc:+d}</b><br>'
                    f'RSI: {rsi_s} &nbsp; ADX: {adx_s}<br>'
                    f'<small style="color:#6a90b0">{reasons_html}</small></div>',
                    unsafe_allow_html=True)

            # Trade levels
            st.markdown("#### 🎯 Trade Setup")
            entry, sl, t1_, t2_, t3_, atr_ = (
                sig["entry"], sig["sl"], sig["t1"], sig["t2"], sig["t3"], sig["atr"])
            is_long = direction == "LONG"

            lv_cols = st.columns(5)
            for col, lbl, val, cls in [
                (lv_cols[0], "Entry",     f"{entry:,.2f}",                  "neu"),
                (lv_cols[1], "Stop Loss", f"{sl:,.2f}",                     "dn" if is_long else "up"),
                (lv_cols[2], "Target 1",  f"{t1_:,.2f}" if t1_ else "—",   "up" if is_long else "dn"),
                (lv_cols[3], "Target 2",  f"{t2_:,.2f}" if t2_ else "—",   "up" if is_long else "dn"),
                (lv_cols[4], "Target 3",  f"{t3_:,.2f}" if t3_ else "—",   "up" if is_long else "dn"),
            ]:
                col.markdown(pcard(lbl, val, cls), unsafe_allow_html=True)

            risk = abs(entry - sl)
            rwd1 = abs(t1_ - entry) if t1_ else 0
            rr   = rwd1 / risk if risk > 0 else 0

            st.markdown(ins("info",
                f"📐 <b>Risk:</b> {risk:.2f} pts ({risk/entry*100:.2f}%) &nbsp;|&nbsp; "
                f"<b>Reward T1:</b> {rwd1:.2f} pts &nbsp;|&nbsp; "
                f"<b>R:R Ratio = 1:{rr:.1f}</b><br>"
                f"📏 ATR (15-Min): {atr_:.2f} — Levels are at ×1.5, ×2.5, ×4.0 ATR<br>"
                f"⚠️ <b>Risk Management:</b> Never risk more than 1–2% of capital on a single trade.<br>"
                f"🕐 <b>Entry Timing:</b> Wait for first 15-min candle to close in the direction of signal before entering.<br>"
                f"✅ <b>Confirmation Needed:</b> Volume expansion + candle close beyond key level + RSI alignment."),
                unsafe_allow_html=True)

            if direction != "NEUTRAL / WAIT":
                # Trade chart overlay
                recent = df.tail(80).copy()
                tc_fig = go.Figure()
                tc_fig.add_trace(go.Candlestick(
                    x=recent.index, open=recent["Open"], high=recent["High"],
                    low=recent["Low"], close=recent["Close"],
                    name="Price",
                    increasing=dict(line=dict(color="#00e676",width=1), fillcolor="#00e676"),
                    decreasing=dict(line=dict(color="#ff4444",width=1), fillcolor="#ff4444"),
                ))
                levels = [
                    (entry, "#ffc107", "solid",  f"Entry  {entry:,.2f}"),
                    (sl,    "#ff4444", "dash",   f"SL  {sl:,.2f}"),
                ]
                if t1_: levels.append((t1_, "#00e676", "dot",  f"T1  {t1_:,.2f}"))
                if t2_: levels.append((t2_, "#34d399", "dot",  f"T2  {t2_:,.2f}"))
                if t3_: levels.append((t3_, "#6ee7b7", "dot",  f"T3  {t3_:,.2f}"))

                for lvl, color, dash, txt in levels:
                    tc_fig.add_hline(y=lvl, line_dash=dash, line_color=color, line_width=1.5)
                    tc_fig.add_annotation(
                        x=recent.index[-1], y=lvl, text=f" {txt}",
                        font=dict(color=color, size=10.5), showarrow=False,
                        xanchor="left", bgcolor="rgba(4,8,15,0.75)")

                tc_fig.update_layout(
                    title=dict(text=f"Trade Setup — {ticker_label}  {emoji} {direction}",
                               font=dict(size=14, color="#b8cfe8"), x=0.01),
                    paper_bgcolor=PLOT_BG, plot_bgcolor=GRID_BG,
                    xaxis_rangeslider_visible=False,
                    font=dict(color="#4a7090", size=10),
                    height=420, margin=dict(t=50, l=55, r=150, b=30),
                    xaxis=dict(gridcolor="#06101c"), yaxis=dict(gridcolor="#06101c"),
                )
                st.plotly_chart(tc_fig, use_container_width=True)
            else:
                st.markdown(ins("warn",
                    "🟡 <b>NEUTRAL / WAIT</b> — Mixed signals across timeframes. "
                    "No high-confidence trade setup. Suggested actions:<br>"
                    "• Wait for ADX to rise above 25 (trend confirmation)<br>"
                    "• Look for RSI to exit extreme zones with volume confirmation<br>"
                    "• Watch for MACD crossover aligned with price breaking a key MA<br>"
                    "• Avoid trading in ranging/low-ADX environments with tight stops"),
                    unsafe_allow_html=True)

            st.markdown(ins("bear",
                "⚠️ <b>Disclaimer:</b> This analysis is for educational purposes only. "
                "It does not constitute financial advice. Past signals do not guarantee future performance. "
                "Always use proper risk management and consult a qualified advisor."),
                unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 5 — RETURN HEATMAPS
    # ════════════════════════════════════════════════════════
    with t5:
        st.markdown('<div class="sh">🗓️ Return Heatmaps — Seasonality Analysis</div>',
                    unsafe_allow_html=True)

        with st.spinner("Loading 5-year daily data for heatmaps…"):
            heat_df = fetch_daily(ticker, "5y")

        if heat_df is None or len(heat_df) < 60:
            st.warning("⚠️ Not enough historical data for heatmap analysis.")
        else:
            # Monthly heatmap
            st.plotly_chart(monthly_heatmap(heat_df, ticker_label), use_container_width=True)

            # Annual returns summary
            st.markdown('<div class="sh">📊 Year-by-Year Performance</div>',
                        unsafe_allow_html=True)
            heat_df2 = heat_df.copy()
            heat_df2["Year"]  = heat_df2.index.year
            heat_df2["Month"] = heat_df2.index.month
            ann = heat_df2.groupby("Year")["Close"].agg(First="first", Last="last")
            ann["Annual Return (%)"] = (ann["Last"] / ann["First"] - 1) * 100
            ann["Year High"] = heat_df2.groupby("Year")["High"].max()
            ann["Year Low"]  = heat_df2.groupby("Year")["Low"].min()
            ann["Range %"]   = (ann["Year High"] - ann["Year Low"]) / ann["Year Low"] * 100
            ann = ann.drop(columns=["First","Last"]).reset_index()
            ann = ann.sort_values("Year", ascending=False)

            ann_styled = (
                ann.style
                .map(color_val, subset=["Annual Return (%)"])
                .format({
                    "Annual Return (%)": "{:+.2f}%",
                    "Year High":   "{:,.2f}",
                    "Year Low":    "{:,.2f}",
                    "Range %":     "{:.1f}%",
                })
                .set_properties(**{"background-color":"#071828","color":"#b8cfe8"})
                .set_table_styles([{"selector":"th","props":[("background-color","#060d1a"),
                                                              ("color","#4a7090"),
                                                              ("font-size","0.73rem")]}])
            )
            st.dataframe(ann_styled, use_container_width=True, height=320)

            # DoW heatmap
            st.plotly_chart(dow_heatmap(heat_df, ticker_label), use_container_width=True)

            # Seasonality insights — derive everything from index, not extra columns
            st.markdown('<div class="sh">💡 Seasonality Insights</div>',
                        unsafe_allow_html=True)
            _h = heat_df.copy()
            _h.index = pd.to_datetime(_h.index)
            _h["_ret"] = _h["Close"].pct_change() * 100
            _h["_mo"]  = _h.index.month
            _h["_dow"] = _h.index.dayofweek

            monthly_avg = _h.groupby("_mo")["_ret"].mean()
            dow_avg     = _h.groupby("_dow")["_ret"].mean()
            best_m  = int(monthly_avg.idxmax()); worst_m = int(monthly_avg.idxmin())
            best_d  = int(dow_avg.idxmax());     worst_d = int(dow_avg.idxmin())

            # Overall stats
            total_ret = float((heat_df["Close"].iloc[-1] / heat_df["Close"].iloc[0] - 1) * 100)
            avg_ann   = total_ret / max(1, len(heat_df) / 252)
            max_dd    = float(((heat_df["Close"] / heat_df["Close"].cummax()) - 1).min() * 100)
            volatility = float(heat_df["Close"].pct_change().std() * np.sqrt(252) * 100)

            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, lbl, val, cls in [
                (sc1, "5Y Total Return", f"{total_ret:+.1f}%", "up" if total_ret>0 else "dn"),
                (sc2, "Avg Annual Ret",  f"{avg_ann:+.1f}%",   "up" if avg_ann>0 else "dn"),
                (sc3, "Max Drawdown",    f"{max_dd:.1f}%",      "dn"),
                (sc4, "Annual Volatility",f"{volatility:.1f}%", "neu"),
            ]:
                col.markdown(pcard(lbl, val, cls), unsafe_allow_html=True)

            st.markdown('<div class="div"></div>', unsafe_allow_html=True)

            seas_insights = [
                ("bull", f"📈 <b>Best Month:</b> {MONTH_NAMES.get(best_m,'?')} "
                         f"(avg daily return: {monthly_avg[best_m]:.3f}%). "
                         f"Historically, this is the strongest month of the year for {ticker_label}."),
                ("bear", f"📉 <b>Worst Month:</b> {MONTH_NAMES.get(worst_m,'?')} "
                         f"(avg daily return: {monthly_avg[worst_m]:.3f}%). "
                         f"Historically, exercise caution during this period."),
                ("bull", f"📅 <b>Best Weekday:</b> {DAY_NAMES.get(best_d,'?')} "
                         f"(avg return: {dow_avg[best_d]:.3f}%). "
                         f"Strongest day for positive closes historically."),
                ("bear", f"📅 <b>Worst Weekday:</b> {DAY_NAMES.get(worst_d,'?')} "
                         f"(avg return: {dow_avg[worst_d]:.3f}%). "
                         f"Most prone to selling pressure historically."),
                ("info", f"📊 <b>Annual Volatility:</b> {volatility:.1f}% — "
                         + ("High volatility; suitable for options strategies and wider stops."
                            if volatility > 30 else
                            "Moderate volatility; standard position sizing applies."
                            if volatility > 15 else
                            "Low volatility; favour breakout strategies and tight stops.")),
                ("warn", f"⚠️ <b>Max Drawdown (5Y):</b> {max_dd:.1f}%. "
                         f"This is the largest peak-to-trough decline observed in 5 years. "
                         f"Ensure your risk tolerance and position size account for moves of this magnitude."),
            ]
            for kind, text in seas_insights:
                st.markdown(ins(kind, text), unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.caption(f"AlgoTrader Pro · Data via yfinance · Last rendered: "
               f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} · "
               "Not financial advice — educational use only.")


if __name__ == "__main__":
    main()
