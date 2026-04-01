# ══════════════════════════════════════════════════════════════════════════════
#  SMART INVESTING — Algorithmic Trading Platform  v2.0
#
#  Install:
#    pip install streamlit yfinance pandas numpy plotly pytz
#    pip install dhanhq                   # optional – Dhan broker live orders
#    pip install streamlit-autorefresh    # optional – live tab auto-refresh
#
#  Run:
#    streamlit run smart_investing.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings("ignore")

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_OK = True
except ImportError:
    AUTO_REFRESH_OK = False

try:
    from dhanhq import dhanhq
    DHAN_OK = True
except ImportError:
    DHAN_OK = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

IST = pytz.timezone("Asia/Kolkata")

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
_SS = {
    "light_theme":          False,
    "live_running":         False,
    "live_position":        None,
    "trade_history":        [],
    "live_log":             [],
    "live_chart_df":        None,
    "live_ema_fast_val":    None,
    "live_ema_slow_val":    None,
    "live_atr_val":         None,
    "live_ltp":             None,
    "last_trade_exit_time": None,
    "live_cfg":             None,
    "bt_results":           None,
    "bt_violations":        [],
    "bt_chart_df":          None,
}
for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════════════
# THEME CSS  (dark default | light optional)
# ════════════════════════════════════════════════════════════════════════════
def inject_css(light: bool) -> None:
    if light:
        # ── Light theme ──────────────────────────────────────────────────
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');
        html,body,[data-testid="stApp"]  { background:#f4f7fb !important; color:#1a2236; font-family:'Syne',sans-serif; }
        [data-testid="stSidebar"]        { background:#eef1f7 !important; border-right:1px solid #d4dbe8; }
        .stTabs [data-baseweb="tab-list"]{ background:transparent; border-bottom:1px solid #d4dbe8; }
        .stTabs [data-baseweb="tab"]     { font-family:'Syne',sans-serif; font-weight:700; font-size:13px; color:#8899aa; padding:10px 22px; border-radius:6px 6px 0 0; }
        .stTabs [aria-selected="true"]   { color:#007a6a !important; background:rgba(0,122,106,.07) !important; }
        [data-testid="metric-container"] { background:#fff; border:1px solid #d4dbe8; border-radius:10px; padding:12px 16px; }
        [data-testid="stMetricLabel"]    { font-size:10px; color:#8899aa !important; letter-spacing:.6px; text-transform:uppercase; }
        [data-testid="stMetricValue"]    { font-family:'JetBrains Mono',monospace; font-size:17px; color:#1a2236; }
        .stButton button                 { border-radius:8px; font-family:'Syne',sans-serif; font-weight:700; border:1px solid #d4dbe8; }
        .stButton button[kind="primary"] { background:linear-gradient(135deg,#00b896,#007a6a); color:#fff; border:none; }
        .stSelectbox>div,.stNumberInput>div input,.stTextInput>div input { background:#fff !important; border:1px solid #d4dbe8 !important; border-radius:8px !important; color:#1a2236 !important; }
        .shdr { font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#8899aa;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #d4dbe8; }
        .card { background:#fff; border:1px solid #d4dbe8; border-radius:12px; padding:14px 18px; margin-bottom:10px; }
        .card-red   { background:rgba(220,50,80,.06); border-color:rgba(220,50,80,.3); }
        .card-green { background:rgba(0,160,120,.06); border-color:rgba(0,160,120,.3); }
        .logbox { background:#f4f7fb; border:1px solid #d4dbe8; border-radius:8px; padding:12px; height:260px; overflow-y:auto; font-family:'JetBrains Mono',monospace; font-size:11px; line-height:1.7; color:#1a2236; }
        hr { border-color:#d4dbe8 !important; }
        ::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:#f4f7fb}::-webkit-scrollbar-thumb{background:#d4dbe8;border-radius:2px}
        </style>""", unsafe_allow_html=True)
    else:
        # ── Dark theme (default) ─────────────────────────────────────────
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');
        html,body,[data-testid="stApp"]  { background:#080b12 !important; color:#c8d0e0; font-family:'Syne',sans-serif; }
        [data-testid="stSidebar"]        { background:#0c1018 !important; border-right:1px solid #1a2236; }
        .stTabs [data-baseweb="tab-list"]{ background:transparent; border-bottom:1px solid #1a2236; gap:4px; }
        .stTabs [data-baseweb="tab"]     { font-family:'Syne',sans-serif; font-weight:700; font-size:13px; color:#4a5568; padding:10px 22px; border-radius:6px 6px 0 0; border:1px solid transparent; transition:all .2s; }
        .stTabs [aria-selected="true"]   { color:#00e5b4 !important; background:rgba(0,229,180,.06) !important; border-color:#1a2236 #1a2236 transparent !important; }
        .stTabs [data-baseweb="tab"]:hover{ color:#8899bb !important; }
        [data-testid="metric-container"] { background:#0f1623; border:1px solid #1a2236; border-radius:10px; padding:12px 16px; }
        [data-testid="stMetricLabel"]    { font-size:10px; color:#4a5568 !important; letter-spacing:.6px; text-transform:uppercase; }
        [data-testid="stMetricValue"]    { font-family:'JetBrains Mono',monospace; font-size:17px; color:#c8d0e0; }
        [data-testid="stMetricDelta"]    { font-family:'JetBrains Mono',monospace; font-size:12px; }
        .stButton button                 { border-radius:8px; font-family:'Syne',sans-serif; font-weight:700; font-size:13px; border:1px solid #1a2236; transition:all .2s; color:#c8d0e0; }
        .stButton button[kind="primary"] { background:linear-gradient(135deg,#00e5b4,#00a88a); color:#000; border:none; }
        .stButton button[kind="primary"]:hover{ opacity:.9; transform:translateY(-1px); box-shadow:0 6px 20px rgba(0,229,180,.25); }
        .stButton button:not([kind="primary"]):hover{ border-color:#00e5b4; color:#00e5b4; }
        .stSelectbox>div,.stNumberInput>div input,.stTextInput>div input { background:#0f1623 !important; border:1px solid #1a2236 !important; border-radius:8px !important; color:#c8d0e0 !important; font-family:'JetBrains Mono',monospace !important; font-size:13px !important; }
        .shdr{ font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4a5568;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #1a2236; }
        .card{ background:#0f1623;border:1px solid #1a2236;border-radius:12px;padding:14px 18px;margin-bottom:10px; }
        .card-red  { background:rgba(255,77,109,.07);border-color:rgba(255,77,109,.3); }
        .card-green{ background:rgba(0,229,180,.06);border-color:rgba(0,229,180,.25); }
        .logbox{ background:#080b12;border:1px solid #1a2236;border-radius:8px;padding:12px;height:260px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.7; }
        hr{ border-color:#1a2236 !important;margin:10px 0 !important; }
        ::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:#080b12}::-webkit-scrollbar-thumb{background:#1a2236;border-radius:2px}
        [data-testid="stExpander"]{ background:#0f1623;border:1px solid #1a2236;border-radius:10px; }
        </style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
TICKER_MAP = {
    "Nifty 50":       "^NSEI",
    "Bank Nifty":     "^NSEBANK",
    "Sensex":         "^BSESN",
    "Bitcoin (BTC)":  "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Gold":           "GC=F",
    "Silver":         "SI=F",
}

TIMEFRAME_PERIODS = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

WARMUP_MAP = {
    "1d":"5d", "5d":"1mo", "7d":"1mo", "1mo":"3mo", "3mo":"6mo",
    "6mo":"1y", "1y":"2y", "2y":"5y", "5y":"max", "10y":"max", "20y":"max",
}

MAX_PERIOD_FOR_INTERVAL = {
    "1m":"7d", "5m":"60d", "15m":"60d", "1h":"730d", "1d":"max", "1wk":"max",
}

PERIOD_DAYS = {
    "1d":1, "5d":5, "7d":7, "1mo":30, "3mo":90,
    "6mo":180, "1y":365, "2y":730, "5y":1825, "10y":3650, "20y":7300,
}

INTERVAL_MINUTES = {
    "1m":1, "5m":5, "15m":15, "1h":60, "1d":1440, "1wk":10080,
}

STRATEGIES   = ["EMA Crossover", "Simple Buy", "Simple Sell"]
SL_TYPES     = ["Custom Points", "Trailing SL", "Reverse EMA Crossover", "Risk Reward Based", "ATR Based"]
TARGET_TYPES = ["Custom Points", "Trailing Target", "EMA Crossover", "Risk Reward Based", "ATR Based"]

# ─────────────────────────────────────────────────────────────────────────────
#  COMMENTED STRATEGY — Price Crosses Threshold (plug-in ready, not in dropdown)
#
#  def price_crosses_threshold(prev_close, curr_close, threshold, direction, action):
#      """
#      direction : "above" | "below"   — which side the price crosses
#      action    : "buy"   | "sell"    — what order to place (from dropdown)
#      Returns   : action string if triggered, else None
#      """
#      if direction == "above" and prev_close < threshold <= curr_close:
#          return action
#      if direction == "below" and prev_close > threshold >= curr_close:
#          return action
#      return None
#
#  To activate:
#    1. Add "Price Crosses Threshold" to STRATEGIES list above.
#    2. Add sidebar widgets: threshold (float input), direction (selectbox above/below),
#       action (selectbox buy/sell).
#    3. In run_backtest() and live_engine() call this function when strategy matches.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  DHAN CANDLESTICK DATA FETCH  (commented — plug in to replace yfinance)
#
#  from dhanhq import dhanhq
#
#  DHAN_INTERVAL_MAP = {"1m":"1", "5m":"5", "15m":"15", "1h":"60", "1d":"D"}
#
#  def fetch_candles_dhan(client_id, access_token, security_id,
#                         exchange_segment, instrument_type,
#                         interval, from_date, to_date):
#      """
#      Real-time / zero-delay candles from Dhan.
#
#      security_id      : e.g. "13"  (Nifty50 index), "1333" (Infosys equity)
#      exchange_segment : "NSE_EQ" | "BSE_EQ" | "NSE_FNO" | "IDX_I"
#      instrument_type  : "EQUITY" | "INDEX" | "FUTIDX" | "OPTIDX"
#      interval         : "1" | "5" | "15" | "25" | "60" | "D"
#      from_date        : "YYYY-MM-DD"
#      to_date          : "YYYY-MM-DD"
#      """
#      dhan = dhanhq(client_id, access_token)
#      if interval != "D":
#          resp = dhan.intraday_minute_data(
#              security_id=security_id,
#              exchange_segment=exchange_segment,
#              instrument_type=instrument_type,
#              interval=interval,
#              from_date=from_date,
#              to_date=to_date,
#          )
#      else:
#          resp = dhan.historical_daily_data(
#              security_id=security_id,
#              exchange_segment=exchange_segment,
#              instrument_type=instrument_type,
#              expiry_code=0,
#              from_date=from_date,
#              to_date=to_date,
#          )
#      records = resp.get("data", [])
#      if not records:
#          return pd.DataFrame()
#      df = pd.DataFrame(records)
#      df.rename(columns={"timestamp":"dt","open":"Open","high":"High",
#                         "low":"Low","close":"Close","volume":"Volume"}, inplace=True)
#      df["dt"] = pd.to_datetime(df["dt"], unit="s", utc=True)
#      df.set_index("dt", inplace=True)
#      df.index = df.index.tz_convert("Asia/Kolkata")
#      return df[["Open","High","Low","Close","Volume"]].sort_index()
#
#  HOW TO PLUG IN — replace fetch_ohlcv() body with:
#    from_dt = (datetime.now() - timedelta(days=PERIOD_DAYS[period])).strftime("%Y-%m-%d")
#    to_dt   = datetime.now().strftime("%Y-%m-%d")
#    df_full = fetch_candles_dhan(CLIENT_ID, ACCESS_TOKEN,
#                                 security_id=YOUR_SECURITY_ID,
#                                 exchange_segment="NSE_EQ",
#                                 instrument_type="EQUITY",
#                                 interval=DHAN_INTERVAL_MAP[interval],
#                                 from_date=from_dt, to_date=to_dt)
#    df_display = df_full.copy()
#    return df_full, df_display
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def _clean(raw) -> pd.DataFrame | None:
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    need = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in need):
        return None
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[cols].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return df if not df.empty else None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str):
    """
    Returns (df_full, df_display) where df_full covers a warmup window
    (so EMA/ATR are never NaN in the display period) and df_display is
    trimmed to the user-selected period.
    Returns None on failure.
    """
    try:
        warmup = WARMUP_MAP.get(period, period)
        max_p  = MAX_PERIOD_FOR_INTERVAL.get(interval, "max")
        wu_days = PERIOD_DAYS.get(warmup, 99999)
        mx_days = PERIOD_DAYS.get(max_p,  99999)
        if max_p != "max" and wu_days > mx_days:
            warmup = max_p

        df_full = _clean(yf.download(
            ticker, period=warmup, interval=interval,
            auto_adjust=True, progress=False, prepost=False,
        ))
        if df_full is None:
            return None

        days = PERIOD_DAYS.get(period)
        if days:
            cutoff     = datetime.now(IST) - timedelta(days=days)
            df_display = df_full[df_full.index >= cutoff].copy()
            if df_display.empty:
                df_display = df_full.copy()
        else:
            df_display = df_full.copy()

        return df_full, df_display
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ltp_cached(ticker: str) -> dict | None:
    try:
        df = _clean(yf.download(ticker, period="5d", interval="1d",
                                auto_adjust=True, progress=False))
        if df is None or len(df) < 2:
            return None
        prev = float(df["Close"].iloc[-2])
        ltp  = float(df["Close"].iloc[-1])
        chg  = ltp - prev
        pct  = chg / prev * 100 if prev else 0
        return {"price": ltp, "change": chg, "pct": pct, "prev": prev}
    except:
        return None

# ════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def tv_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Matches TradingView EMA exactly.
    alpha = 2/(period+1), seeds from bar 0, no adjust (adjust=False).
    """
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi = df["High"].astype(float)
    lo = df["Low"].astype(float)
    pc = df["Close"].astype(float).shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()


def compute_ema_angle(ema_series: pd.Series, idx: int) -> float:
    """
    Approximate angle (°) of the EMA slope at bar idx.
    Uses percentage slope to be price-scale independent:
      slope_pct = (ema[i] - ema[i-1]) / ema[i-1] * 100
      angle     = |arctan(slope_pct)|  (in degrees)
    """
    if idx < 1 or idx >= len(ema_series):
        return 0.0
    prev = float(ema_series.iloc[idx - 1])
    curr = float(ema_series.iloc[idx])
    if prev == 0:
        return 0.0
    slope_pct = (curr - prev) / prev * 100.0
    return abs(float(np.degrees(np.arctan(slope_pct))))


def add_indicators(df: pd.DataFrame, fast: int, slow: int, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    ef = tv_ema(df["Close"].astype(float), fast)
    es = tv_ema(df["Close"].astype(float), slow)
    df[f"EMA_{fast}"] = ef
    df[f"EMA_{slow}"] = es
    df["ATR"]         = compute_atr(df, atr_period)

    pef, pes = ef.shift(1), es.shift(1)
    df["Signal"] = 0
    df.loc[(ef > es) & (pef <= pes), "Signal"] =  1   # bullish cross
    df.loc[(ef < es) & (pef >= pes), "Signal"] = -1   # bearish cross
    return df

# ════════════════════════════════════════════════════════════════════════════
# SL / TARGET CALCULATORS
# ════════════════════════════════════════════════════════════════════════════

def calc_sl(entry: float, tt: str, sl_type: str, sl_pts: float,
            ema_f: float, ema_s: float, rr: float,
            atr_val: float = 0.0, atr_mult: float = 1.5) -> float | None:
    sign = 1 if tt == "buy" else -1
    if sl_type == "Custom Points":
        return entry - sign * sl_pts
    elif sl_type == "Trailing SL":
        return entry - sign * sl_pts      # initial; updated tick-by-tick
    elif sl_type == "Reverse EMA Crossover":
        base = ema_s if tt == "buy" else ema_f
        fb   = entry - sign * sl_pts
        return min(base, fb) if tt == "buy" else max(base, fb)
    elif sl_type == "Risk Reward Based":
        return entry - sign * sl_pts
    elif sl_type == "ATR Based":
        atr_v = atr_val if atr_val > 0 else sl_pts
        return entry - sign * atr_v * atr_mult
    return entry - sign * sl_pts


def calc_tgt(entry: float, tt: str, tgt_type: str, tgt_pts: float,
             sl: float | None, ema_f: float, ema_s: float, rr: float,
             atr_val: float = 0.0, atr_tgt_mult: float = 2.0) -> float | None:
    sign = 1 if tt == "buy" else -1
    if tgt_type == "Custom Points":
        return entry + sign * tgt_pts
    elif tgt_type == "Trailing Target":
        return entry + sign * tgt_pts    # display only — never forces exit
    elif tgt_type == "EMA Crossover":
        return None                       # exit on signal
    elif tgt_type == "Risk Reward Based":
        risk = abs(entry - sl) if sl is not None else tgt_pts
        return entry + sign * risk * rr
    elif tgt_type == "ATR Based":
        atr_v = atr_val if atr_val > 0 else tgt_pts
        return entry + sign * atr_v * atr_tgt_mult
    return entry + sign * tgt_pts

# ════════════════════════════════════════════════════════════════════════════
# ──────────────────────   BACKTEST ENGINE   ──────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
#
#  CORRECT EMA CROSSOVER BACKTEST LOGIC (and why):
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  CANDLE N   : crossover is DETECTED here (at close of candle N)    │
#  │  CANDLE N+1 : entry is PLACED at OPEN of candle N+1                │
#  │               (you cannot trade candle N because you only know the  │
#  │                crossover happened AFTER it closes)                  │
#  │                                                                     │
#  │  EXIT (Backtest — candle data only, no tick feed):                  │
#  │   • BUY  : SL checked vs candle LOW,  Target vs candle HIGH         │
#  │   • SELL : SL checked vs candle HIGH, Target vs candle LOW          │
#  │   • Conservative rule: SL is evaluated FIRST for both types.        │
#  │     If both SL and Target breach in the same candle, SL wins.       │
#  │     This is the worst-case (most conservative) assumption.          │
#  │                                                                     │
#  │  EXIT (Live trading — tick-by-tick LTP available):                  │
#  │   • SL and Target are compared against current LTP every tick.      │
#  │   • Conservative: SL still checked first.                           │
#  │   • Signal check (for EMA cross exit) only at candle close.         │
#  └─────────────────────────────────────────────────────────────────────┘
#
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df_full, df_display, strategy, fast, slow,
    sl_type, sl_pts, tgt_type, tgt_pts,
    qty, cd_en, cd_s, no_overlap, rr,
    atr_period, atr_sl_mult, atr_tgt_mult,
    min_angle, crossover_candle, candle_size_pts, candle_atr_mult,
):
    # ── 1. Compute indicators on FULL df (no NaN in display window) ───────
    df = add_indicators(df_full.copy(), fast, slow, atr_period)
    if strategy != "EMA Crossover":
        df["Signal"] = 0

    # ── 2. Find warmup boundary ───────────────────────────────────────────
    display_start = df_display.index[0] if not df_display.empty else df.index[0]
    warmup_rows   = max(slow * 3, int(df.index.searchsorted(display_start)))
    warmup_rows   = min(warmup_rows, len(df) - 1)

    trades:         list[dict] = []
    violations:     list[dict] = []
    active:         dict | None = None
    pending_entry:  dict | None = None   # ← set on signal candle N, executed at N+1 open
    last_exit_ts                = None

    # ── 3. Bar loop ───────────────────────────────────────────────────────
    for i in range(warmup_rows, len(df)):
        row   = df.iloc[i]
        ts    = df.index[i]
        open_ = float(row["Open"])
        hi    = float(row["High"])
        lo    = float(row["Low"])
        cl    = float(row["Close"])
        sig   = int(row.get("Signal", 0))
        ef_v  = float(row[f"EMA_{fast}"])
        es_v  = float(row[f"EMA_{slow}"])
        atr_v = float(row["ATR"]) if "ATR" in df.columns else sl_pts

        # ── 3a. EXECUTE PENDING ENTRY at this candle's OPEN ──────────────
        #   Entry happens at the OPEN of the bar AFTER the crossover candle.
        #   Reason: crossover is only confirmed at the CLOSE of the signal
        #   candle — the earliest possible entry is the next bar's open.
        if pending_entry is not None and active is None:
            # Cooldown check
            if cd_en and last_exit_ts is not None:
                if (ts - last_exit_ts).total_seconds() < cd_s:
                    pending_entry = None
                    continue

            ep = open_   # ← ENTRY at N+1 candle OPEN price
            tt = pending_entry["trade_type"]

            # Recompute SL/Target from actual entry price
            sl_p  = calc_sl(ep, tt, sl_type, sl_pts, ef_v, es_v, rr, atr_v, atr_sl_mult)
            tgt_p = calc_tgt(ep, tt, tgt_type, tgt_pts, sl_p, ef_v, es_v, rr, atr_v, atr_tgt_mult)

            # Skip if open already past SL (gap scenario — gapped through SL)
            if tt == "buy"  and sl_p is not None and ep <= sl_p:
                pending_entry = None
            elif tt == "sell" and sl_p is not None and ep >= sl_p:
                pending_entry = None
            else:
                active = {
                    "entry_time":   ts,
                    "entry_price":  ep,
                    "trade_type":   tt,
                    "sl":           sl_p,
                    "target":       tgt_p,
                    "entry_reason": pending_entry["reason"],
                }
                pending_entry = None

        # ── 3b. EXIT LOGIC ────────────────────────────────────────────────
        if active is not None:
            ep  = active["entry_price"]
            tt  = active["trade_type"]
            sl  = active["sl"]
            tgt = active["target"]

            # Compare SL/Target vs candle H or L
            if tt == "buy":
                sl_hit  = (sl  is not None and lo  <= sl)
                tgt_hit = (tgt is not None and hi  >= tgt)
            else:  # sell
                sl_hit  = (sl  is not None and hi  >= sl)
                tgt_hit = (tgt is not None and lo  <= tgt)

            # EMA crossover exit (from completed bar signal)
            ema_exit = False
            if tgt_type == "EMA Crossover":
                if tt == "buy"  and sig == -1: ema_exit = True
                if tt == "sell" and sig ==  1: ema_exit = True

            exit_px  = None
            exit_rsn = None
            viol     = False

            # ── CONSERVATIVE RULE (BUY and SELL): SL first ──────────────
            if sl_hit:
                exit_px  = sl
                exit_rsn = "Stop Loss Hit"
                if tgt_hit:
                    viol     = True
                    exit_rsn = "SL Hit ⚠ (ambiguous: both SL & Target in same candle, SL taken first)"
            elif tgt_hit:
                exit_px  = tgt
                exit_rsn = "Target Hit"
            elif ema_exit:
                exit_px  = cl
                exit_rsn = "EMA Crossover Exit"

            # Trailing SL update (no exit yet)
            if sl_type == "Trailing SL" and exit_px is None:
                if tt == "buy":
                    nsl = cl - sl_pts
                    if nsl > active["sl"]: active["sl"] = nsl
                else:
                    nsl = cl + sl_pts
                    if nsl < active["sl"]: active["sl"] = nsl

            if exit_px is not None:
                pnl = ((exit_px - ep) if tt == "buy" else (ep - exit_px)) * qty
                rec = {
                    "Entry Time":   active["entry_time"],
                    "Exit Time":    ts,
                    "Trade Type":   tt.upper(),
                    "Entry Price":  round(ep, 2),
                    "Exit Price":   round(float(exit_px), 2),
                    "SL":           round(float(active["sl"]), 2)  if active["sl"]  is not None else "—",
                    "Target":       round(float(tgt), 2)            if tgt           is not None else "—",
                    "Candle High":  round(hi, 2),
                    "Candle Low":   round(lo, 2),
                    "Entry Reason": active["entry_reason"],
                    "Exit Reason":  exit_rsn,
                    "PnL (Rs)":     round(pnl, 2),
                    "Qty":          qty,
                    "Violation":    "Yes" if viol else "",
                }
                trades.append(rec)
                if viol: violations.append(rec)
                last_exit_ts = ts
                active       = None

        # ── 3c. SIGNAL DETECTION (N+1 open entry) ─────────────────────────
        #   We set pending_entry here so it fires at the NEXT bar's open.
        if active is None and pending_entry is None:
            trade_type = None
            entry_rsn  = ""

            if strategy == "EMA Crossover" and sig != 0:
                # ── Crossover angle filter ───────────────────────────────
                angle = compute_ema_angle(df[f"EMA_{fast}"], i)
                if angle < min_angle:
                    continue  # Crossover angle too flat — skip

                # ── Crossover candle body size filter ────────────────────
                body = abs(cl - open_)
                if crossover_candle == "Custom Candle Size":
                    if body < candle_size_pts: continue
                elif crossover_candle == "ATR Based Candle Size":
                    if body < candle_atr_mult * atr_v: continue

                if sig ==  1: trade_type = "buy";  entry_rsn = f"EMA{fast} crossed above EMA{slow} (angle {angle:.1f}°)"
                if sig == -1: trade_type = "sell"; entry_rsn = f"EMA{fast} crossed below EMA{slow} (angle {angle:.1f}°)"

            elif strategy == "Simple Buy"  and i == warmup_rows:
                trade_type = "buy";  entry_rsn = "Simple Buy (market order)"
            elif strategy == "Simple Sell" and i == warmup_rows:
                trade_type = "sell"; entry_rsn = "Simple Sell (market order)"

            if trade_type is not None:
                pending_entry = {
                    "trade_type": trade_type,
                    "reason":     entry_rsn,
                    "ema_f":      ef_v,
                    "ema_s":      es_v,
                    "atr":        atr_v,
                }

    # ── 4. Force close open trade at end of data ──────────────────────────
    if active is not None:
        lr  = df.iloc[-1]
        epx = float(lr["Close"])
        pnl = ((epx - active["entry_price"]) if active["trade_type"] == "buy"
               else (active["entry_price"] - epx)) * qty
        trades.append({
            "Entry Time":   active["entry_time"],
            "Exit Time":    df.index[-1],
            "Trade Type":   active["trade_type"].upper(),
            "Entry Price":  round(active["entry_price"], 2),
            "Exit Price":   round(epx, 2),
            "SL":           round(float(active["sl"]), 2)     if active["sl"]     is not None else "—",
            "Target":       round(float(active["target"]), 2) if active["target"] is not None else "—",
            "Candle High":  round(float(lr["High"]), 2),
            "Candle Low":   round(float(lr["Low"]), 2),
            "Entry Reason": active["entry_reason"],
            "Exit Reason":  "End of Data (position closed)",
            "PnL (Rs)":     round(pnl, 2),
            "Qty":          qty,
            "Violation":    "",
        })

    return pd.DataFrame(trades), violations


# ════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ════════════════════════════════════════════════════════════════════════════

def _plotly_base(light: bool) -> dict:
    bg   = "#f4f7fb" if light else "#080b12"
    card = "#ffffff"  if light else "#080b12"
    grid = "#e8edf5"  if light else "#0f1623"
    txt  = "#334155"  if light else "#6b7a99"
    return dict(
        template="plotly_white" if light else "plotly_dark",
        paper_bgcolor=bg,
        plot_bgcolor=card,
        font=dict(family="JetBrains Mono", color=txt, size=11),
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=16, l=48, r=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#fff" if light else "#0f1623",
                        bordercolor="#d4dbe8" if light else "#1a2236",
                        font=dict(family="JetBrains Mono", size=11)),
        xaxis=dict(gridcolor=grid, zeroline=False, showspikes=True,
                   spikecolor="#94a3b8" if light else "#2a3456", spikethickness=1),
        yaxis=dict(gridcolor=grid, zeroline=False, showspikes=True,
                   spikecolor="#94a3b8" if light else "#2a3456", spikethickness=1),
    )


def build_chart(df: pd.DataFrame, fast: int, slow: int,
                trades_df=None, position=None, title: str = "",
                light: bool = False) -> go.Figure:

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02)

    # Candles
    fig.add_trace(go.Candlestick(
        x=list(df.index),
        open=list(df["Open"].astype(float)),
        high=list(df["High"].astype(float)),
        low=list(df["Low"].astype(float)),
        close=list(df["Close"].astype(float)),
        name="OHLC",
        increasing=dict(line=dict(color="#00b896" if light else "#00e5b4", width=1.2),
                        fillcolor="#00b896" if light else "#00e5b4"),
        decreasing=dict(line=dict(color="#e0284a" if light else "#ff4d6d", width=1.2),
                        fillcolor="#e0284a" if light else "#ff4d6d"),
        whiskerwidth=0.4,
    ), row=1, col=1)

    # EMAs — always show if columns present
    ef_col = f"EMA_{fast}"
    es_col = f"EMA_{slow}"
    if ef_col in df.columns:
        fig.add_trace(go.Scatter(
            x=list(df.index), y=list(df[ef_col].astype(float)),
            name=f"EMA {fast}", line=dict(color="#f59e0b", width=1.8),
        ), row=1, col=1)
    if es_col in df.columns:
        fig.add_trace(go.Scatter(
            x=list(df.index), y=list(df[es_col].astype(float)),
            name=f"EMA {slow}", line=dict(color="#3b82f6", width=1.8),
        ), row=1, col=1)

    # Backtest trade markers
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            tt  = str(t["Trade Type"])
            clr = ("#00b896" if light else "#00e5b4") if tt == "BUY" else ("#e0284a" if light else "#ff4d6d")
            en_sym = "triangle-up"   if tt == "BUY" else "triangle-down"
            ex_sym = "triangle-down" if tt == "BUY" else "triangle-up"
            ep_v   = float(t["Entry Price"]); xp_v = float(t["Exit Price"])
            pnl_v  = float(t["PnL (Rs)"])

            fig.add_trace(go.Scatter(
                x=[t["Entry Time"]], y=[ep_v], mode="markers",
                name="Entry", showlegend=False,
                marker=dict(symbol=en_sym, size=13, color=clr,
                            line=dict(width=1.5, color="#fff")),
                hovertemplate=(f"<b>{tt} ENTRY</b><br>Price: {ep_v:.2f}<br>"
                               f"{t.get('Entry Reason','')}<extra></extra>"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t["Exit Time"]], y=[xp_v], mode="markers",
                name="Exit", showlegend=False,
                marker=dict(symbol=ex_sym, size=11, color=clr, opacity=0.6,
                            line=dict(width=1, color="#000" if light else "#080b12")),
                hovertemplate=(f"<b>{tt} EXIT</b><br>Price: {xp_v:.2f}<br>"
                               f"{t.get('Exit Reason','')}<br>PnL: Rs{pnl_v:+.2f}<extra></extra>"),
            ), row=1, col=1)

    # Live position lines
    if position is not None:
        for lvl, col, lbl in [
            (position.get("entry_price"), "#ffffff" if not light else "#334155", "Entry"),
            (position.get("sl"),          "#e0284a" if light else "#ff4d6d",     "SL"),
            (position.get("target"),      "#00b896" if light else "#00e5b4",     "Target"),
        ]:
            if lvl is not None:
                fig.add_hline(
                    y=float(lvl), row=1, col=1,
                    line=dict(color=col, width=1.4,
                              dash="dash" if lbl != "Entry" else "solid"),
                    annotation_text=f" {lbl} {float(lvl):.2f}",
                    annotation_font=dict(color=col, size=11),
                )

    # Volume
    if "Volume" in df.columns:
        vol_clr = [("#00b896" if light else "#00e5b4") if float(c) >= float(o) else ("#e0284a" if light else "#ff4d6d")
                   for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=list(df.index),
            y=list(df["Volume"].fillna(0).astype(float)),
            marker_color=vol_clr, name="Vol", showlegend=False,
        ), row=2, col=1)

    layout = _plotly_base(light)
    # Extract xaxis/yaxis from base and apply per-row
    xax = layout.pop("xaxis", {})
    yax = layout.pop("yaxis", {})
    fig.update_layout(
        height=560,
        title=dict(text=title, x=0.01,
                   font=dict(family="Syne", size=13,
                             color="#334155" if light else "#6b7a99")),
        **layout,
    )
    fig.update_xaxes(**xax)
    fig.update_yaxes(**yax)
    return fig

# ════════════════════════════════════════════════════════════════════════════
# UI WIDGETS
# ════════════════════════════════════════════════════════════════════════════

def ltp_widget(ticker: str, label: str, light: bool) -> None:
    info = fetch_ltp_cached(ticker)
    if info:
        p, c, pct = info["price"], info["change"], info["pct"]
        up_clr    = "#00b896" if light else "#00e5b4"
        dn_clr    = "#e0284a" if light else "#ff4d6d"
        col       = up_clr if c >= 0 else dn_clr
        arr       = "▲" if c >= 0 else "▼"
        bg        = "#ffffff" if light else "#0f1623"
        brd       = "#d4dbe8" if light else "#1a2236"
        txt       = "#1a2236" if light else "#e2e8f0"
        muted     = "#8899aa" if light else "#4a5568"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {brd};border-radius:12px;
                    padding:14px 22px;margin-bottom:14px;display:flex;
                    align-items:center;gap:28px;flex-wrap:wrap">
            <div>
                <div style="font-family:Syne,sans-serif;font-size:10px;font-weight:700;
                            letter-spacing:2px;text-transform:uppercase;color:{muted}">{label}</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:28px;
                            font-weight:700;color:{txt};line-height:1.1">{p:,.2f}</div>
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:17px;font-weight:600;color:{col}">{arr} {c:+.2f}</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:600;color:{col}">{pct:+.2f}%</div>
            <div style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:11px;color:{muted}">
                Prev Close: {info['prev']:,.2f}
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"LTP unavailable for {label}.")

# ════════════════════════════════════════════════════════════════════════════
# DHAN ORDER PLACEMENT
# ════════════════════════════════════════════════════════════════════════════

def _dhan(cfg):
    return dhanhq(str(cfg["dhan_client_id"]), str(cfg["dhan_access_token"]))


def place_entry_order(cfg: dict, tt: str, ltp: float) -> str:
    if not DHAN_OK: return "dhanhq not installed"
    try:
        dhan = _dhan(cfg)
        if cfg.get("options_trading"):
            sec   = cfg["ce_security_id"] if tt == "buy" else cfg["pe_security_id"]
            ot    = "MARKET" if cfg.get("opts_entry_otype") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0.0
            r = dhan.place_order(
                transactionType="BUY",
                exchangeSegment=str(cfg.get("opts_exchange", "NSE_FNO")),
                productType="INTRADAY", orderType=ot,
                validity="DAY", securityId=str(sec),
                quantity=int(cfg.get("opts_qty", 65)),
                price=price, triggerPrice=0,
            )
        else:
            ex = "NSE_EQ" if cfg.get("eq_exchange", "NSE") == "NSE" else "BSE"
            pt = "INTRADAY" if cfg.get("eq_product") == "Intraday" else "CNC"
            ot = "MARKET" if cfg.get("eq_entry_otype") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0.0
            r = dhan.place_order(
                security_id=str(cfg.get("eq_sec_id", "1594")),
                exchange_segment=ex,
                transaction_type="BUY" if tt == "buy" else "SELL",
                quantity=int(cfg.get("eq_qty", 1)),
                order_type=ot, product_type=pt, price=price,
            )
        return f"Entry OK: {r}"
    except Exception as e:
        return f"Entry FAILED: {e}"


def place_exit_order(cfg: dict, tt: str, ltp: float) -> str:
    if not DHAN_OK: return "dhanhq not installed"
    try:
        dhan = _dhan(cfg)
        if cfg.get("options_trading"):
            sec   = cfg["ce_security_id"] if tt == "buy" else cfg["pe_security_id"]
            ot    = "MARKET" if cfg.get("opts_exit_otype", "Market Order") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0.0
            r = dhan.place_order(
                transactionType="SELL",
                exchangeSegment=str(cfg.get("opts_exchange", "NSE_FNO")),
                productType="INTRADAY", orderType=ot,
                validity="DAY", securityId=str(sec),
                quantity=int(cfg.get("opts_qty", 65)),
                price=price, triggerPrice=0,
            )
        else:
            ex = "NSE_EQ" if cfg.get("eq_exchange", "NSE") == "NSE" else "BSE"
            pt = "INTRADAY" if cfg.get("eq_product") == "Intraday" else "CNC"
            ot = "MARKET" if cfg.get("eq_exit_otype", "Market Order") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0.0
            r = dhan.place_order(
                security_id=str(cfg.get("eq_sec_id", "1594")),
                exchange_segment=ex,
                transaction_type="SELL" if tt == "buy" else "BUY",
                quantity=int(cfg.get("eq_qty", 1)),
                order_type=ot, product_type=pt, price=price,
            )
        return f"Exit OK: {r}"
    except Exception as e:
        return f"Exit FAILED: {e}"

# ════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE (background thread)
# ════════════════════════════════════════════════════════════════════════════
#
#  Live trading EMA crossover flow:
#   1. Every tick (1.5 s poll): fetch latest OHLCV, update LTP, check SL/Target vs LTP.
#   2. On each NEW completed candle (detected by timestamp change of df.index[-2]):
#      - Compute EMA values on the COMPLETED candle (index[-2], not the forming bar).
#      - Check if a crossover occurred on that completed candle.
#      - If yes, place entry at CURRENT LTP (equivalent to next tick's "open").
#   3. SL and Target are always checked vs LTP (not H/L) in live trading.
#      Conservative rule still applies: SL checked first.
# ════════════════════════════════════════════════════════════════════════════

def live_engine(cfg: dict) -> None:
    ticker      = cfg["ticker"];    interval    = cfg["interval"];  period = cfg["period"]
    strategy    = cfg["strategy"];  fast        = cfg["fast_ema"]; slow   = cfg["slow_ema"]
    sl_type     = cfg["sl_type"];   sl_pts      = cfg["sl_pts"]
    tgt_type    = cfg["tgt_type"];  tgt_pts     = cfg["tgt_pts"]
    qty         = cfg["qty"];       cd_en       = cfg["cd_en"];    cd_s   = cfg["cd_s"]
    rr          = cfg["rr"]
    atr_period  = cfg.get("atr_period", 14)
    atr_sl_m    = cfg.get("atr_sl_mult", 1.5)
    atr_tgt_m   = cfg.get("atr_tgt_mult", 2.0)
    min_angle   = cfg.get("min_angle", 0.0)
    co_candle   = cfg.get("crossover_candle", "Simple Crossover")
    c_size_pts  = cfg.get("candle_size_pts", 10.0)
    c_atr_m     = cfg.get("candle_atr_mult", 1.0)
    int_mins    = INTERVAL_MINUTES.get(interval, 5)

    def _log(msg: str) -> None:
        ts    = datetime.now(IST).strftime("%H:%M:%S")
        entry = f"<span style='color:#8899aa'>[{ts}]</span> {msg}"
        logs  = st.session_state.get("live_log", [])
        logs.append(entry)
        st.session_state["live_log"] = logs[-200:]

    _log("🚀 <b>Engine started</b>")
    last_closed_ts = None

    while st.session_state.get("live_running", False):
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=True, progress=False, prepost=False)
            df  = _clean(raw)

            if df is None or len(df) < max(slow * 2 + 5, 10):
                _log("Insufficient data — waiting…")
                time.sleep(1.5)
                continue

            # Add indicators
            df[f"EMA_{fast}"] = tv_ema(df["Close"].astype(float), fast)
            df[f"EMA_{slow}"] = tv_ema(df["Close"].astype(float), slow)
            df["ATR"]         = compute_atr(df, atr_period)

            # Store for UI (use COMPLETED candle values = index[-2])
            ef_completed = float(df[f"EMA_{fast}"].iloc[-2])
            es_completed = float(df[f"EMA_{slow}"].iloc[-2])
            atr_completed= float(df["ATR"].iloc[-2])

            st.session_state["live_chart_df"]     = df.copy()
            st.session_state["live_ema_fast_val"] = ef_completed
            st.session_state["live_ema_slow_val"] = es_completed
            st.session_state["live_atr_val"]      = atr_completed

            ltp = float(df["Close"].iloc[-1])   # forming / latest bar LTP
            st.session_state["live_ltp"] = ltp

            # ── EXIT: SL/Target vs LTP (conservative: SL first) ──────────
            pos = st.session_state.get("live_position")
            if pos is not None:
                tt  = pos["trade_type"]
                ep  = pos["entry_price"]
                sl  = pos["sl"]
                tgt = pos["target"]

                exited     = False
                exit_px    = None
                exit_reason= None

                # Conservative SL-first
                if tt == "buy":
                    if sl  is not None and ltp <= sl:
                        exit_px = ltp; exit_reason = "SL Hit (LTP ≤ SL)"; exited = True
                    elif tgt is not None and ltp >= tgt:
                        exit_px = ltp; exit_reason = "Target Hit (LTP ≥ Target)"; exited = True
                else:
                    if sl  is not None and ltp >= sl:
                        exit_px = ltp; exit_reason = "SL Hit (LTP ≥ SL)"; exited = True
                    elif tgt is not None and ltp <= tgt:
                        exit_px = ltp; exit_reason = "Target Hit (LTP ≤ Target)"; exited = True

                # EMA crossover exit (completed candle only)
                if not exited and tgt_type == "EMA Crossover" and len(df) >= 3:
                    pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])
                    if tt == "buy"  and ef_completed < es_completed and pf >= ps:
                        exit_px = ltp; exit_reason = "EMA Crossover Exit"; exited = True
                    if tt == "sell" and ef_completed > es_completed and pf <= ps:
                        exit_px = ltp; exit_reason = "EMA Crossover Exit"; exited = True

                # Trailing SL
                if sl_type == "Trailing SL" and not exited:
                    if tt == "buy":
                        nsl = ltp - sl_pts
                        if nsl > pos["sl"]: pos["sl"] = nsl; _log(f"Trailing SL → <b>{nsl:.2f}</b>")
                    else:
                        nsl = ltp + sl_pts
                        if nsl < pos["sl"]: pos["sl"] = nsl; _log(f"Trailing SL → <b>{nsl:.2f}</b>")
                    st.session_state["live_position"] = pos

                if exited and exit_px is not None:
                    pnl = ((exit_px - ep) if tt == "buy" else (ep - exit_px)) * qty
                    rec = {
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   tt.upper(),
                        "Entry Price":  round(ep, 2),
                        "Exit Price":   round(float(exit_px), 2),
                        "SL":           round(float(sl), 2)  if sl  is not None else "—",
                        "Target":       round(float(tgt), 2) if tgt is not None else "—",
                        "Entry Reason": pos.get("entry_reason", ""),
                        "Exit Reason":  exit_reason,
                        "PnL (Rs)":     round(pnl, 2),
                        "Qty":          qty,
                        "Mode":         "Live",
                        "Violation":    "",
                    }
                    st.session_state["trade_history"].append(rec)
                    st.session_state["live_position"]        = None
                    st.session_state["last_trade_exit_time"] = datetime.now(IST)
                    if cfg.get("dhan_en"):
                        _log(f"Broker: {place_exit_order(cfg, tt, float(exit_px))}")
                    pnl_clr = "color:#00e5b4" if pnl >= 0 else "color:#ff4d6d"
                    _log(f"EXIT <b>{tt.upper()}</b> @ {float(exit_px):.2f} | {exit_reason} | "
                         f"<span style='{pnl_clr}'>PnL Rs{pnl:+.2f}</span>")

            # ── ENTRY: check on NEW completed candle only ─────────────────
            current_closed_ts = df.index[-2] if len(df) >= 2 else None
            is_new_candle     = (current_closed_ts is not None and
                                 current_closed_ts != last_closed_ts)

            if is_new_candle and st.session_state.get("live_position") is None:
                last_closed_ts = current_closed_ts

                # Cooldown
                if cd_en:
                    last_exit = st.session_state.get("last_trade_exit_time")
                    if last_exit and (datetime.now(IST) - last_exit).total_seconds() < cd_s:
                        remain = cd_s - (datetime.now(IST) - last_exit).total_seconds()
                        _log(f"Cooldown: {remain:.0f}s remaining")
                        time.sleep(1.5)
                        continue

                if len(df) < 3:
                    time.sleep(1.5)
                    continue

                pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])

                signal     = None
                entry_rsn  = ""

                if strategy == "EMA Crossover":
                    bullish = ef_completed > es_completed and pf <= ps
                    bearish = ef_completed < es_completed and pf >= ps

                    if bullish or bearish:
                        # Angle check
                        angle = compute_ema_angle(df[f"EMA_{fast}"], len(df) - 2)
                        if angle >= min_angle:
                            # Candle body check on completed candle
                            body = abs(float(df["Close"].iloc[-2]) - float(df["Open"].iloc[-2]))
                            size_ok = True
                            if co_candle == "Custom Candle Size":
                                size_ok = body >= c_size_pts
                            elif co_candle == "ATR Based Candle Size":
                                size_ok = body >= c_atr_m * atr_completed
                            if size_ok:
                                if bullish: signal = "buy";  entry_rsn = f"EMA{fast} x EMA{slow} ↑ (angle {angle:.1f}°)"
                                if bearish: signal = "sell"; entry_rsn = f"EMA{fast} x EMA{slow} ↓ (angle {angle:.1f}°)"

                elif strategy == "Simple Buy":  signal = "buy";  entry_rsn = "Simple Buy"
                elif strategy == "Simple Sell": signal = "sell"; entry_rsn = "Simple Sell"

                if signal:
                    sl_p  = calc_sl(ltp, signal, sl_type, sl_pts, ef_completed, es_completed, rr, atr_completed, atr_sl_m)
                    tgt_p = calc_tgt(ltp, signal, tgt_type, tgt_pts, sl_p, ef_completed, es_completed, rr, atr_completed, atr_tgt_m)
                    st.session_state["live_position"] = {
                        "trade_type":   signal,
                        "entry_price":  ltp,
                        "entry_time":   datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "sl":           sl_p,
                        "target":       tgt_p,
                        "entry_reason": entry_rsn,
                    }
                    if cfg.get("dhan_en"):
                        _log(f"Broker: {place_entry_order(cfg, signal, ltp)}")
                    clr2 = "color:#00e5b4" if signal == "buy" else "color:#ff4d6d"
                    sl_s  = f"{sl_p:.2f}"  if sl_p  is not None else "None"
                    tgt_s = f"{tgt_p:.2f}" if tgt_p is not None else "None"
                    _log(f"ENTRY <b><span style='{clr2}'>{signal.upper()}</span></b> @ {ltp:.2f} "
                         f"| SL: {sl_s} | Target: {tgt_s} | {entry_rsn}")

        except Exception as exc:
            _log(f"<span style='color:#ff4d6d'>Error: {exc}</span>")

        time.sleep(1.5)   # ← yfinance rate-limit guard — do NOT remove

    _log("Engine stopped")


# ════════════════════════════════════════════════════════════════════════════
# ██████████████████████████   MAIN APP   ████████████████████████████████████
# ════════════════════════════════════════════════════════════════════════════

def main():
    light = st.session_state.get("light_theme", False)
    inject_css(light)

    # ════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════════════════
    with st.sidebar:
        logo_clr = "#007a6a" if light else "#00e5b4"
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0 10px">
            <div style="font-size:38px">📈</div>
            <div style="font-family:Syne,sans-serif;font-size:18px;font-weight:800;
                        letter-spacing:3px;color:{logo_clr}">SMART INVESTING</div>
            <div style="font-size:9px;color:#8899aa;letter-spacing:2px;margin-top:2px">
                ALGORITHMIC TRADING PLATFORM
            </div>
        </div>""", unsafe_allow_html=True)

        # Theme toggle
        light_on = st.checkbox("⬜ Light Theme", value=light, key="light_theme")
        if light_on != light:
            st.rerun()

        st.divider()

        # ── ASSET ────────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Asset</div>', unsafe_allow_html=True)
        choice = st.selectbox("Ticker", list(TICKER_MAP.keys()) + ["Custom →"],
                              label_visibility="collapsed")
        if choice == "Custom →":
            ticker_sym   = st.text_input("Symbol (e.g. RELIANCE.NS)", "RELIANCE.NS")
            ticker_label = ticker_sym
        else:
            ticker_sym   = TICKER_MAP[choice]
            ticker_label = choice

        # ── TIMEFRAME — default 5m / 1mo ─────────────────────────────────
        st.markdown('<div class="shdr">Timeframe</div>', unsafe_allow_html=True)
        tf_keys = list(TIMEFRAME_PERIODS.keys())
        c1, c2  = st.columns(2)
        with c1:
            interval = st.selectbox("Interval", tf_keys,
                                    index=tf_keys.index("5m"),
                                    label_visibility="collapsed")
        with c2:
            prd_opts    = TIMEFRAME_PERIODS[interval]
            default_prd = prd_opts.index("1mo") if "1mo" in prd_opts else min(1, len(prd_opts)-1)
            period      = st.selectbox("Period", prd_opts, index=default_prd,
                                       label_visibility="collapsed")

        st.divider()

        # ── STRATEGY ─────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Strategy</div>', unsafe_allow_html=True)
        strategy = st.selectbox("Strategy", STRATEGIES, label_visibility="collapsed")

        fast_ema = 9; slow_ema = 15
        min_angle        = 0.0
        crossover_candle = "Simple Crossover"
        candle_size_pts  = 10.0
        candle_atr_mult  = 1.0

        if strategy == "EMA Crossover":
            e1, e2 = st.columns(2)
            with e1: fast_ema = st.number_input("Fast EMA", 2, 200, 9)
            with e2: slow_ema = st.number_input("Slow EMA", 2, 500, 15)

            st.markdown("**Crossover Conditions**")
            min_angle = st.number_input(
                "Min Crossover Angle (°)",
                min_value=0.0, max_value=90.0, value=0.0, step=0.5,
                help="Minimum angle of the fast EMA at the crossover bar. "
                     "0 = no filter. Higher values require a steeper / stronger crossover.",
            )
            crossover_candle = st.selectbox(
                "Crossover Candle Filter",
                ["Simple Crossover", "Custom Candle Size", "ATR Based Candle Size"],
                help="Minimum body size of the crossover candle (optional filter).",
            )
            if crossover_candle == "Custom Candle Size":
                candle_size_pts = st.number_input("Min Body (points)", 0.1, 1e6, 10.0, 0.5)
            elif crossover_candle == "ATR Based Candle Size":
                candle_atr_mult = st.number_input("Min Body (ATR mult)", 0.1, 10.0, 1.0, 0.1)

        st.divider()

        # ── STOP LOSS ────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Stop Loss</div>', unsafe_allow_html=True)
        sl_type  = st.selectbox("SL Type", SL_TYPES, label_visibility="collapsed")
        s1, s2   = st.columns(2)
        with s1: sl_pts = st.number_input("SL Points", 0.1, 1e6, 10.0, step=0.5)
        with s2: rr     = st.number_input("R:R", 0.5, 20.0, 2.0, step=0.5,
                                          help="Risk:Reward ratio (for Risk Reward Based SL/Target)")

        atr_period   = 14
        atr_sl_mult  = 1.5
        atr_tgt_mult = 2.0

        # ── TARGET ───────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Target</div>', unsafe_allow_html=True)
        tgt_type = st.selectbox("Target Type", TARGET_TYPES, label_visibility="collapsed")
        tgt_pts  = st.number_input("Target Points", 0.1, 1e6, 20.0, step=0.5)

        # ATR config (show if SL or Target is ATR Based)
        if sl_type == "ATR Based" or tgt_type == "ATR Based":
            st.markdown("**ATR Configuration**")
            atr_period = st.number_input("ATR Period", 2, 200, 14)
            if sl_type == "ATR Based":
                atr_sl_mult  = st.number_input("ATR Mult (SL)", 0.1, 10.0, 1.5, 0.1)
            if tgt_type == "ATR Based":
                atr_tgt_mult = st.number_input("ATR Mult (Target)", 0.1, 10.0, 2.0, 0.1)

        st.divider()

        # ── TRADE SETTINGS ───────────────────────────────────────────────
        st.markdown('<div class="shdr">Trade Settings</div>', unsafe_allow_html=True)
        qty = st.number_input("Quantity", 1, 10_000_000, 1)
        cd1, cd2 = st.columns([1.4, 1])
        with cd1: cd_en = st.checkbox("Cooldown Period", value=True)
        with cd2: cd_s  = st.number_input("Secs", 1, 86400, 5,
                                           disabled=not cd_en,
                                           label_visibility="visible")
        no_overlap = st.checkbox("No Overlapping Trades", value=True)

        st.divider()

        # ── DHAN BROKER ──────────────────────────────────────────────────
        st.markdown('<div class="shdr">Dhan Broker</div>', unsafe_allow_html=True)
        dhan_en  = st.checkbox("Enable Dhan Broker", value=False)
        dhan_cfg: dict = {"dhan_en": dhan_en}

        if dhan_en:
            if not DHAN_OK:
                st.warning("Run: pip install dhanhq")
            dhan_cfg["dhan_client_id"]    = st.text_input("Client ID",    "", type="password")
            dhan_cfg["dhan_access_token"] = st.text_input("Access Token", "", type="password")
            opts_en = st.checkbox("Options Trading", value=False)
            dhan_cfg["options_trading"] = opts_en
            if not opts_en:
                dhan_cfg["eq_product"]      = st.selectbox("Product",      ["Intraday", "Delivery"])
                dhan_cfg["eq_exchange"]     = st.selectbox("Exchange",     ["NSE", "BSE"])
                dhan_cfg["eq_sec_id"]       = st.text_input("Security ID", "1594")
                dhan_cfg["eq_qty"]          = st.number_input("Broker Qty", 1, 1_000_000, 1)
                dhan_cfg["eq_entry_otype"]  = st.selectbox("Entry Order",  ["Limit Order", "Market Order"])
                dhan_cfg["eq_exit_otype"]   = st.selectbox("Exit Order",   ["Market Order", "Limit Order"])
            else:
                dhan_cfg["opts_exchange"]     = st.selectbox("FnO Exchange", ["NSE_FNO", "BSE_FNO"])
                dhan_cfg["ce_security_id"]    = st.text_input("CE Sec ID", "")
                dhan_cfg["pe_security_id"]    = st.text_input("PE Sec ID", "")
                dhan_cfg["opts_qty"]          = st.number_input("Lots/Qty", 1, 1_000_000, 65)
                dhan_cfg["opts_entry_otype"]  = st.selectbox("Entry Order", ["Market Order", "Limit Order"])
                dhan_cfg["opts_exit_otype"]   = st.selectbox("Exit Order",  ["Market Order", "Limit Order"])

        st.divider()
        st.caption("Smart Investing v2.0  •  Educational use only")

    # Full cfg bundle
    cfg: dict = {
        "ticker": ticker_sym, "ticker_label": ticker_label,
        "interval": interval, "period": period,
        "strategy": strategy, "fast_ema": fast_ema, "slow_ema": slow_ema,
        "sl_type": sl_type, "sl_pts": sl_pts,
        "tgt_type": tgt_type, "tgt_pts": tgt_pts,
        "qty": qty, "cd_en": cd_en, "cd_s": cd_s,
        "no_overlap": no_overlap, "rr": rr,
        "atr_period": atr_period, "atr_sl_mult": atr_sl_mult, "atr_tgt_mult": atr_tgt_mult,
        "min_angle": min_angle, "crossover_candle": crossover_candle,
        "candle_size_pts": candle_size_pts, "candle_atr_mult": candle_atr_mult,
        **dhan_cfg,
    }

    # ════════════════════════════════════════════════════════════════════
    # PAGE HEADER
    # ════════════════════════════════════════════════════════════════════
    hdr_clr = "#007a6a" if light else "#00e5b4"
    st.markdown(f"""
    <div style="padding:4px 0 10px">
        <span style="font-family:Syne,sans-serif;font-size:30px;font-weight:800;color:{hdr_clr}">
            📈 Smart Investing
        </span>
        <span style="font-family:Syne,sans-serif;font-size:12px;color:#8899aa;
                     margin-left:14px;vertical-align:middle">
            Algorithmic Trading Platform
        </span>
    </div>""", unsafe_allow_html=True)

    tab_bt, tab_live, tab_hist = st.tabs([
        "🔬  Backtesting",
        "⚡  Live Trading",
        "📋  Trade History",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — BACKTESTING
    # ════════════════════════════════════════════════════════════════════
    with tab_bt:
        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### 🔬 Backtest Engine")

        acc_clr = "#007a6a" if light else "#00e5b4"
        st.markdown(f"""
        <div style="background:{'rgba(0,122,106,.07)' if light else 'rgba(0,229,180,.05)'};
                    border-left:3px solid {acc_clr};padding:10px 14px;border-radius:6px;
                    margin-bottom:12px;font-size:12px;font-family:Syne,sans-serif">
            <b style="color:{acc_clr}">Entry at N+1 Open · Conservative SL-First Exit</b><br>
            <span style="color:{'#555' if light else '#7a9988'}">
            Crossover detected at <b>close of candle N</b> → entry placed at
            <b>open of candle N+1</b> (you cannot trade the signal candle itself).
            SL is compared vs candle <b>Low</b> (buy) / <b>High</b> (sell).
            Target vs candle <b>High</b> (buy) / <b>Low</b> (sell).
            When both breach in the same bar, <b>SL wins</b> (worst-case conservative).
            </span>
        </div>""", unsafe_allow_html=True)

        r1, r2 = st.columns([4, 1])
        with r1:
            run_btn = st.button("▶  Run Backtest", type="primary",
                                use_container_width=True, key="run_bt")
        with r2:
            if st.button("↺ Clear", use_container_width=True, key="clr_bt"):
                st.session_state.update(bt_results=None, bt_violations=[], bt_chart_df=None)
                st.rerun()

        if run_btn:
            with st.spinner(f"Fetching {ticker_label}  {interval}/{period}…"):
                res = fetch_ohlcv(ticker_sym, interval, period)
            if res is None:
                st.error("Data fetch failed. Check ticker or network.")
            else:
                df_full, df_display = res
                with st.spinner("Running backtest…"):
                    tdf, viols = run_backtest(
                        df_full, df_display, strategy, fast_ema, slow_ema,
                        sl_type, sl_pts, tgt_type, tgt_pts,
                        qty, cd_en, cd_s, no_overlap, rr,
                        atr_period, atr_sl_mult, atr_tgt_mult,
                        min_angle, crossover_candle, candle_size_pts, candle_atr_mult,
                    )
                df_chart = add_indicators(df_full.copy(), fast_ema, slow_ema, atr_period)
                if not df_display.empty:
                    df_chart = df_chart.loc[df_display.index[0]:]
                st.session_state["bt_results"]    = tdf
                st.session_state["bt_violations"] = viols
                st.session_state["bt_chart_df"]   = df_chart

        tdf   = st.session_state.get("bt_results")
        viols = st.session_state.get("bt_violations", [])
        dfc   = st.session_state.get("bt_chart_df")

        if tdf is not None:
            if tdf.empty:
                st.warning("No trades generated. Adjust strategy parameters.")
            else:
                n       = len(tdf)
                wins    = int((tdf["PnL (Rs)"] > 0).sum())
                losses  = n - wins
                tot_pnl = float(tdf["PnL (Rs)"].sum())
                acc     = wins / n * 100 if n else 0
                avg_w   = float(tdf.loc[tdf["PnL (Rs)"] > 0, "PnL (Rs)"].mean()) if wins   else 0.0
                avg_l   = float(tdf.loc[tdf["PnL (Rs)"] < 0, "PnL (Rs)"].mean()) if losses else 0.0
                best    = float(tdf["PnL (Rs)"].max())
                worst   = float(tdf["PnL (Rs)"].min())

                m = st.columns(8)
                for col, label, val in zip(m, [
                    "Trades","Wins","Losses","Accuracy",
                    "Total PnL","Avg Win","Best","Violations"
                ], [n, wins, losses, f"{acc:.1f}%",
                    f"Rs{tot_pnl:+,.0f}", f"Rs{avg_w:+.0f}", f"Rs{best:+.0f}", len(viols)]):
                    col.metric(label, val)

                st.divider()

                if dfc is not None:
                    st.plotly_chart(
                        build_chart(dfc, fast_ema, slow_ema, trades_df=tdf,
                                    title=f"{ticker_label} · {strategy} · {interval}/{period}",
                                    light=light),
                        use_container_width=True,
                    )

                st.markdown("#### 📊 Trade Log")
                disp = tdf.copy()
                disp["Entry Time"] = disp["Entry Time"].astype(str)
                disp["Exit Time"]  = disp["Exit Time"].astype(str)

                # Color scheme: subtle tints, readable font
                win_bg  = "rgba(0,180,140,0.12)"  if light else "rgba(0,229,180,0.10)"
                loss_bg = "rgba(220,50,80,0.10)"  if light else "rgba(255,77,109,0.10)"
                win_fg  = "#005a47" if light else "#00e5b4"
                loss_fg = "#8b0000" if light else "#ff7799"

                def _style_row(row):
                    pnl = row["PnL (Rs)"]
                    if pnl > 0:
                        return [f"background-color:{win_bg};color:{win_fg}" for _ in row]
                    elif pnl < 0:
                        return [f"background-color:{loss_bg};color:{loss_fg}" for _ in row]
                    return ["" for _ in row]

                styled = (disp.style
                          .apply(_style_row, axis=1)
                          .format({"PnL (Rs)": "Rs{:+.2f}",
                                   "Entry Price": "{:.2f}", "Exit Price": "{:.2f}"}))
                st.dataframe(styled, use_container_width=True, height=420,
                             column_config={
                                 "Trade Type": st.column_config.TextColumn(width=70),
                                 "Violation":  st.column_config.TextColumn(width=60),
                             })

                if viols:
                    st.markdown(f"""
                    <div style="background:rgba(220,50,80,.08);border-left:4px solid #dc3250;
                                padding:12px 16px;border-radius:8px;margin:10px 0;
                                font-family:Syne,sans-serif;font-size:12px">
                        ⚠ <b>{len(viols)} candle(s)</b> where BOTH SL and Target were
                        hit in the same bar. SL applied (conservative). These trades are
                        most likely to differ from live results — tick data would resolve
                        which was actually hit first.
                    </div>""", unsafe_allow_html=True)
                    with st.expander(f"View {len(viols)} violation(s)"):
                        vdf = pd.DataFrame(viols).copy()
                        vdf["Entry Time"] = vdf["Entry Time"].astype(str)
                        vdf["Exit Time"]  = vdf["Exit Time"].astype(str)
                        st.dataframe(vdf, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — LIVE TRADING
    # ════════════════════════════════════════════════════════════════════
    with tab_live:
        if st.session_state["live_running"] and AUTO_REFRESH_OK:
            st_autorefresh(interval=2000, key="live_ar")

        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### ⚡ Live Trading")

        running = st.session_state["live_running"]

        # Status
        if running:
            st.markdown("""
            <div style="display:inline-flex;align-items:center;gap:8px;
                        background:rgba(0,229,180,.1);border:1px solid rgba(0,229,180,.3);
                        border-radius:20px;padding:6px 16px;margin-bottom:10px">
                <span style="width:8px;height:8px;border-radius:50%;background:#00e5b4;
                             display:inline-block;animation:pulse 1s infinite"></span>
                <span style="font-family:Syne,sans-serif;font-size:12px;font-weight:700;
                             color:#00e5b4;letter-spacing:1px">LIVE RUNNING</span>
            </div>
            <style>@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(1.4)}}</style>
            """, unsafe_allow_html=True)

        # Control buttons
        b1, b2, b3, b4, _ = st.columns([1.2, 1.2, 1.4, 1.4, 2])
        with b1:
            if st.button("▶ Start", type="primary", use_container_width=True,
                         disabled=running, key="live_start"):
                st.session_state["live_running"]  = True
                st.session_state["live_log"]      = []
                st.session_state["live_cfg"]      = cfg.copy()
                st.session_state["live_position"] = None
                t = threading.Thread(target=live_engine, args=(cfg,), daemon=True)
                t.start()
                st.rerun()
        with b2:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not running, key="live_stop"):
                st.session_state["live_running"] = False
                time.sleep(0.4)
                st.rerun()
        with b3:
            if st.button("⚡ Square Off", use_container_width=True, key="live_sq"):
                pos = st.session_state.get("live_position")
                if pos:
                    ltp_now = st.session_state.get("live_ltp") or pos["entry_price"]
                    pnl  = ((ltp_now - pos["entry_price"]) if pos["trade_type"] == "buy"
                            else (pos["entry_price"] - ltp_now)) * qty
                    rec = {
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   pos["trade_type"].upper(),
                        "Entry Price":  round(float(pos["entry_price"]), 2),
                        "Exit Price":   round(float(ltp_now), 2),
                        "SL":           round(float(pos["sl"]), 2)     if pos["sl"]     is not None else "—",
                        "Target":       round(float(pos["target"]), 2) if pos["target"] is not None else "—",
                        "Entry Reason": pos.get("entry_reason", ""),
                        "Exit Reason":  "Manual Square Off",
                        "PnL (Rs)":     round(pnl, 2),
                        "Qty":          qty, "Mode": "Live", "Violation": "",
                    }
                    st.session_state["trade_history"].append(rec)
                    st.session_state["live_position"] = None
                    if cfg.get("dhan_en"):
                        place_exit_order(cfg, pos["trade_type"], float(ltp_now))
                    st.success(f"Squared off @ {float(ltp_now):.2f}  |  PnL Rs{pnl:+.2f}")
                else:
                    st.info("No open position.")
                st.rerun()
        with b4:
            if st.button("🔄 Refresh Display", use_container_width=True, key="live_refresh"):
                # Manually fetch latest data and update session state
                try:
                    raw = yf.download(ticker_sym, period=period, interval=interval,
                                      auto_adjust=True, progress=False, prepost=False)
                    df_r = _clean(raw)
                    if df_r is not None and len(df_r) >= slow_ema + 5:
                        df_r[f"EMA_{fast_ema}"] = tv_ema(df_r["Close"].astype(float), fast_ema)
                        df_r[f"EMA_{slow_ema}"] = tv_ema(df_r["Close"].astype(float), slow_ema)
                        df_r["ATR"]             = compute_atr(df_r, atr_period)
                        st.session_state["live_chart_df"]     = df_r
                        st.session_state["live_ema_fast_val"] = float(df_r[f"EMA_{fast_ema}"].iloc[-2])
                        st.session_state["live_ema_slow_val"] = float(df_r[f"EMA_{slow_ema}"].iloc[-2])
                        st.session_state["live_atr_val"]      = float(df_r["ATR"].iloc[-2])
                        st.session_state["live_ltp"]          = float(df_r["Close"].iloc[-1])
                        st.success("Refreshed.")
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
                st.rerun()

        st.divider()

        # ── Active config (updates without full page reload via session_state) ──
        live_cfg_shown = st.session_state.get("live_cfg") or cfg
        with st.expander("⚙ Active Configuration", expanded=True):
            cc = st.columns(4)
            cc[0].metric("Ticker",   live_cfg_shown.get("ticker_label", "—"))
            cc[1].metric("Interval", live_cfg_shown.get("interval", "—"))
            cc[2].metric("Period",   live_cfg_shown.get("period", "—"))
            cc[3].metric("Strategy", live_cfg_shown.get("strategy", "—"))
            cc2 = st.columns(5)
            cc2[0].metric(f"Fast EMA", live_cfg_shown.get("fast_ema", "—"))
            cc2[1].metric(f"Slow EMA", live_cfg_shown.get("slow_ema", "—"))
            cc2[2].metric("SL",   f"{live_cfg_shown.get('sl_type','—')} / {live_cfg_shown.get('sl_pts','—')}pt")
            cc2[3].metric("TGT",  f"{live_cfg_shown.get('tgt_type','—')} / {live_cfg_shown.get('tgt_pts','—')}pt")
            cc2[4].metric("Qty",  live_cfg_shown.get("qty", "—"))

        st.divider()

        # ── Main layout: chart + position  |  log ────────────────────────
        main_col, log_col = st.columns([1.7, 1])

        with main_col:
            ef_v   = st.session_state.get("live_ema_fast_val")
            es_v   = st.session_state.get("live_ema_slow_val")
            atr_v  = st.session_state.get("live_atr_val")
            ltp_c  = st.session_state.get("live_ltp")
            pos    = st.session_state.get("live_position")

            # EMA + LTP metrics (always show if data available)
            if ef_v is not None and es_v is not None:
                diff = ef_v - es_v
                trend = "Bullish ↑" if diff > 0 else "Bearish ↓"
                t_delta = "normal" if diff > 0 else "inverse"
                mv1, mv2, mv3, mv4, mv5 = st.columns(5)
                mv1.metric(f"EMA {fast_ema}",    f"{ef_v:.2f}")
                mv2.metric(f"EMA {slow_ema}",    f"{es_v:.2f}")
                mv3.metric("EMA Diff",  f"{diff:+.2f}", delta=trend, delta_color=t_delta)
                mv4.metric("ATR",       f"{atr_v:.2f}" if atr_v else "—")
                mv5.metric("LTP",       f"{ltp_c:.2f}" if ltp_c else "—")
            else:
                st.info("Click **▶ Start** or **🔄 Refresh Display** to see live EMA values.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Open position card + live PnL
            if pos is not None:
                tt   = pos["trade_type"]
                ep   = float(pos["entry_price"])
                sl   = pos.get("sl")
                tgt  = pos.get("target")
                pnl_live = ((ltp_c - ep) if (ltp_c is not None and tt == "buy")
                            else ((ep - ltp_c) if ltp_c is not None else 0.0)) * qty
                pos_clr  = "card-green" if pnl_live >= 0 else "card-red"
                tt_clr   = ("#00b896" if light else "#00e5b4") if tt == "buy" else ("#e0284a" if light else "#ff4d6d")
                pnl_clr  = ("#00b896" if light else "#00e5b4") if pnl_live >= 0 else ("#e0284a" if light else "#ff4d6d")
                txt_clr  = "#1a2236" if light else "#c8d0e0"
                mut_clr  = "#8899aa" if light else "#4a5568"

                sl_s  = f"{float(sl):.2f}"  if sl  is not None else "—"
                tgt_s = f"{float(tgt):.2f}" if tgt is not None else "N/A"
                ltp_s = f"{ltp_c:.2f}"      if ltp_c is not None else "—"

                st.markdown(f"""
                <div class="card {pos_clr}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                    <span style="font-family:Syne,sans-serif;font-size:13px;font-weight:800;
                                 letter-spacing:2px;color:{tt_clr}">● OPEN {tt.upper()} POSITION</span>
                    <span style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:{pnl_clr}">
                        Rs{pnl_live:+.2f}
                    </span>
                  </div>
                  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;
                              font-family:JetBrains Mono,monospace;font-size:13px;color:{txt_clr}">
                    <div><div style="color:{mut_clr};font-size:10px;text-transform:uppercase">Entry</div>{ep:.2f}</div>
                    <div><div style="color:{mut_clr};font-size:10px;text-transform:uppercase">LTP</div>{ltp_s}</div>
                    <div><div style="color:#e0284a;font-size:10px;text-transform:uppercase">SL</div>{sl_s}</div>
                    <div><div style="color:{("#00b896" if light else "#00e5b4")};font-size:10px;text-transform:uppercase">Target</div>{tgt_s}</div>
                    <div><div style="color:{mut_clr};font-size:10px;text-transform:uppercase">Qty</div>{qty}</div>
                  </div>
                  <div style="margin-top:10px;font-family:Syne,sans-serif;font-size:11px;color:{mut_clr}">
                    Entry: {pos["entry_time"]}  ·  {pos.get("entry_reason","")}
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                bg2  = "#ffffff" if light else "#0f1623"
                brd2 = "#d4dbe8" if light else "#1a2236"
                st.markdown(f"""
                <div style="background:{bg2};border:1px solid {brd2};border-radius:12px;
                            text-align:center;padding:24px;color:#8899aa;
                            font-family:Syne,sans-serif;font-size:13px">
                    No open position
                </div>""", unsafe_allow_html=True)

            # Chart — always show if data is available
            df_live = st.session_state.get("live_chart_df")
            if df_live is not None and not df_live.empty:
                st.plotly_chart(
                    build_chart(df_live, fast_ema, slow_ema, position=pos,
                                title=f"Live · {ticker_label} · {interval}",
                                light=light),
                    use_container_width=True,
                )

                # Last fetched candle
                st.markdown("**Last Fetched Candle** *(yfinance may have 15-min delay)*")
                lr   = df_live.iloc[-1]
                ef_c = f"EMA_{fast_ema}"
                es_c = f"EMA_{slow_ema}"
                last_row_data = {
                    "Time":  str(df_live.index[-1]),
                    "Open":  round(float(lr["Open"]),  2),
                    "High":  round(float(lr["High"]),  2),
                    "Low":   round(float(lr["Low"]),   2),
                    "Close": round(float(lr["Close"]), 2),
                    "Vol":   int(lr.get("Volume", 0)),
                    f"EMA{fast_ema}": round(float(df_live[ef_c].iloc[-1]), 2) if ef_c in df_live.columns else "—",
                    f"EMA{slow_ema}": round(float(df_live[es_c].iloc[-1]), 2) if es_c in df_live.columns else "—",
                    "ATR":   round(float(df_live["ATR"].iloc[-1]), 2) if "ATR" in df_live.columns else "—",
                }
                st.dataframe(pd.DataFrame([last_row_data]), use_container_width=True,
                             hide_index=True)

        with log_col:
            st.markdown("**Activity Log**")
            if not running and not st.session_state.get("live_log"):
                st.caption("Start the engine to see activity.")
            logs = st.session_state.get("live_log", [])
            log_html = (
                "".join(f"<div style='padding:2px 0;border-bottom:1px solid {'#eef1f7' if light else '#0f1623'}'>{l}</div>"
                        for l in reversed(logs))
                if logs else "<div style='color:#8899aa'>No activity yet…</div>"
            )
            st.markdown(f'<div class="logbox">{log_html}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — TRADE HISTORY
    # ════════════════════════════════════════════════════════════════════
    with tab_hist:
        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### 📋 Trade History")
        st.caption("All completed trades — updates in real-time even while live is running.")

        hist = st.session_state.get("trade_history", [])

        _, hcol = st.columns([5, 1])
        with hcol:
            if st.button("🗑 Clear", use_container_width=True, key="clr_hist"):
                st.session_state["trade_history"] = []
                st.rerun()

        if not hist:
            bg3 = "#ffffff" if light else "#0f1623"
            brd3= "#d4dbe8" if light else "#1a2236"
            st.markdown(f"""
            <div style="background:{bg3};border:1px solid {brd3};border-radius:12px;
                        text-align:center;padding:40px;color:#8899aa;font-family:Syne,sans-serif">
                No completed trades yet.
            </div>""", unsafe_allow_html=True)
        else:
            hdf     = pd.DataFrame(hist)
            tot_pnl = float(hdf["PnL (Rs)"].sum())
            wins    = int((hdf["PnL (Rs)"] > 0).sum())
            n_h     = len(hdf)
            acc_h   = wins / n_h * 100 if n_h else 0
            avg_pnl = float(hdf["PnL (Rs)"].mean())
            best_h  = float(hdf["PnL (Rs)"].max())
            worst_h = float(hdf["PnL (Rs)"].min())

            hm = st.columns(6)
            for col, label, val in zip(hm, [
                "Trades","Win Rate","Total PnL","Avg PnL","Best","Worst"
            ], [n_h, f"{acc_h:.1f}%",
                f"Rs{tot_pnl:+,.0f}", f"Rs{avg_pnl:+.0f}",
                f"Rs{best_h:+.0f}", f"Rs{worst_h:+.0f}"]):
                col.metric(label, val)

            st.divider()

            win_bg  = "rgba(0,180,140,0.12)"  if light else "rgba(0,229,180,0.10)"
            loss_bg = "rgba(220,50,80,0.10)"  if light else "rgba(255,77,109,0.10)"
            win_fg  = "#005a47" if light else "#00e5b4"
            loss_fg = "#8b0000" if light else "#ff7799"

            def _hstyle(row):
                p = row["PnL (Rs)"]
                if p > 0:   return [f"background-color:{win_bg};color:{win_fg}" for _ in row]
                elif p < 0: return [f"background-color:{loss_bg};color:{loss_fg}" for _ in row]
                return ["" for _ in row]

            styled_h = (hdf.style.apply(_hstyle, axis=1)
                        .format({"PnL (Rs)": "Rs{:+.2f}",
                                 "Entry Price": "{:.2f}", "Exit Price": "{:.2f}"}))
            st.dataframe(styled_h, use_container_width=True, height=420)

            # Cumulative PnL curve
            st.markdown("#### Cumulative PnL Curve")
            hdf["Cum PnL"] = hdf["PnL (Rs)"].cumsum()
            base = _plotly_base(light)
            xax  = base.pop("xaxis", {})
            yax  = base.pop("yaxis", {})
            fp   = go.Figure()
            fp.add_trace(go.Scatter(
                x=list(range(1, len(hdf) + 1)),
                y=list(hdf["Cum PnL"].astype(float)),
                mode="lines+markers",
                line=dict(color="#00b896" if light else "#00e5b4", width=2),
                marker=dict(size=5,
                            color=[("#00b896" if light else "#00e5b4") if p >= 0
                                   else ("#e0284a" if light else "#ff4d6d")
                                   for p in hdf["PnL (Rs)"]]),
                fill="tozeroy",
                fillcolor="rgba(0,180,140,0.08)" if light else "rgba(0,229,180,0.06)",
            ))
            fp.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))
            fp.update_layout(height=260, showlegend=False,
                             xaxis_title="Trade #", yaxis_title="PnL (Rs)", **base)
            fp.update_xaxes(**xax)
            fp.update_yaxes(**yax)
            st.plotly_chart(fp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
