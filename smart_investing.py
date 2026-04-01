# ══════════════════════════════════════════════════════════════════════════════
#  SMART INVESTING — Algorithmic Trading Platform  v4.0
#
#  Install:
#    pip install streamlit yfinance pandas numpy plotly pytz
#    pip install dhanhq                  # optional – Dhan broker live orders
#    pip install streamlit-autorefresh   # REQUIRED for live auto-refresh
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
# SESSION STATE  — each user gets their own isolated state (Streamlit guarantees this)
# ════════════════════════════════════════════════════════════════════════════
_SS = {
    "light_theme":   True,
    # backtest (completely isolated from live)
    "bt_results":    None,
    "bt_violations": [],
    "bt_chart_df":   None,
    "bt_run_key":    None,   # "<ticker>|<interval>|<period>|<strategy>"
    # live state (a plain mutable dict; the thread holds a reference to it)
    "live_shared":   None,
    # threading.Event — more reliable stop than a flag in a dict
    "live_stop_evt": None,
}
for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _new_live_shared() -> dict:
    return {
        "running":        False,
        "position":       None,
        "pending_entry":  None,   # waiting for N+1 open (EMA crossover only)
        "trade_history":  [],
        "log":            [],
        "chart_df":       None,
        "ema_fast_val":   None,
        "ema_slow_val":   None,
        "atr_val":        None,
        "ltp":            None,
        "last_exit_ts":   None,
        "cfg":            None,
    }


if st.session_state["live_shared"]  is None:
    st.session_state["live_shared"]  = _new_live_shared()
if st.session_state["live_stop_evt"] is None:
    st.session_state["live_stop_evt"] = threading.Event()

# ════════════════════════════════════════════════════════════════════════════
# THEME CSS
# ════════════════════════════════════════════════════════════════════════════
def inject_css(light: bool) -> None:
    if light:
        st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');
html,body,[data-testid="stApp"]   {background:#f5f7fa !important;color:#1a2236;font-family:'Syne',sans-serif;}
[data-testid="stSidebar"]         {background:#eef1f7 !important;border-right:1px solid #d0d7e6;}
.stTabs [data-baseweb="tab-list"] {background:transparent;border-bottom:1px solid #d0d7e6;}
.stTabs [data-baseweb="tab"]      {font-family:'Syne',sans-serif;font-weight:700;font-size:13px;color:#7a8899;padding:10px 22px;border-radius:6px 6px 0 0;}
.stTabs [aria-selected="true"]    {color:#007a6a !important;background:rgba(0,122,106,.07) !important;}
[data-testid="metric-container"]  {background:#fff;border:1px solid #d0d7e6;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricLabel"]     {font-size:10px;color:#7a8899 !important;letter-spacing:.6px;text-transform:uppercase;}
[data-testid="stMetricValue"]     {font-family:'JetBrains Mono',monospace;font-size:17px;color:#1a2236;}
[data-testid="stMetricDelta"]     {font-family:'JetBrains Mono',monospace;font-size:12px;}
.stButton button                  {border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;border:1px solid #d0d7e6;transition:all .2s;}
.stButton button[kind="primary"]  {background:linear-gradient(135deg,#00b896,#007a6a);color:#fff;border:none;}
.stSelectbox>div,.stNumberInput>div input,.stTextInput>div input {background:#fff !important;border:1px solid #d0d7e6 !important;border-radius:8px !important;color:#1a2236 !important;font-family:'JetBrains Mono',monospace !important;}
.shdr{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#7a8899;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #d0d7e6;}
.card{background:#fff;border:1px solid #d0d7e6;border-radius:12px;padding:14px 18px;margin-bottom:10px;}
.card-red{background:rgba(220,50,80,.05);border-color:rgba(220,50,80,.35);}
.card-green{background:rgba(0,160,120,.05);border-color:rgba(0,160,120,.35);}
.logbox{background:#f5f7fa;border:1px solid #d0d7e6;border-radius:8px;padding:12px;height:280px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.7;color:#1a2236;}
hr{border-color:#d0d7e6 !important;margin:10px 0 !important;}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:#f5f7fa}::-webkit-scrollbar-thumb{background:#d0d7e6;border-radius:2px}
[data-testid="stExpander"]{background:#fff;border:1px solid #d0d7e6;border-radius:10px;}
</style>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');
html,body,[data-testid="stApp"]   {background:#080b12 !important;color:#c8d0e0;font-family:'Syne',sans-serif;}
[data-testid="stSidebar"]         {background:#0c1018 !important;border-right:1px solid #1a2236;}
.stTabs [data-baseweb="tab-list"] {background:transparent;border-bottom:1px solid #1a2236;gap:4px;}
.stTabs [data-baseweb="tab"]      {font-family:'Syne',sans-serif;font-weight:700;font-size:13px;color:#4a5568;padding:10px 22px;border-radius:6px 6px 0 0;border:1px solid transparent;transition:all .2s;}
.stTabs [aria-selected="true"]    {color:#00e5b4 !important;background:rgba(0,229,180,.06) !important;border-color:#1a2236 #1a2236 transparent !important;}
.stTabs [data-baseweb="tab"]:hover{color:#8899bb !important;}
[data-testid="metric-container"]  {background:#0f1623;border:1px solid #1a2236;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricLabel"]     {font-size:10px;color:#4a5568 !important;letter-spacing:.6px;text-transform:uppercase;}
[data-testid="stMetricValue"]     {font-family:'JetBrains Mono',monospace;font-size:17px;color:#c8d0e0;}
[data-testid="stMetricDelta"]     {font-family:'JetBrains Mono',monospace;font-size:12px;}
.stButton button                  {border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;font-size:13px;border:1px solid #1a2236;transition:all .2s;color:#c8d0e0;}
.stButton button[kind="primary"]  {background:linear-gradient(135deg,#00e5b4,#00a88a);color:#000;border:none;}
.stButton button[kind="primary"]:hover{opacity:.9;transform:translateY(-1px);}
.stButton button:not([kind="primary"]):hover{border-color:#00e5b4;color:#00e5b4;}
.stSelectbox>div,.stNumberInput>div input,.stTextInput>div input{background:#0f1623 !important;border:1px solid #1a2236 !important;border-radius:8px !important;color:#c8d0e0 !important;font-family:'JetBrains Mono',monospace !important;font-size:13px !important;}
.shdr{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4a5568;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #1a2236;}
.card{background:#0f1623;border:1px solid #1a2236;border-radius:12px;padding:14px 18px;margin-bottom:10px;}
.card-red{background:rgba(255,77,109,.07);border-color:rgba(255,77,109,.3);}
.card-green{background:rgba(0,229,180,.06);border-color:rgba(0,229,180,.25);}
.logbox{background:#080b12;border:1px solid #1a2236;border-radius:8px;padding:12px;height:280px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.7;}
hr{border-color:#1a2236 !important;margin:10px 0 !important;}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:#080b12}::-webkit-scrollbar-thumb{background:#1a2236;border-radius:2px}
[data-testid="stExpander"]{background:#0f1623;border:1px solid #1a2236;border-radius:10px;}
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

PERIOD_DAYS = {
    "1d":1,"5d":5,"7d":7,"1mo":30,"3mo":90,"6mo":180,
    "1y":365,"2y":730,"5y":1825,"10y":3650,"20y":7300,
}

# Hard max days yfinance allows per interval
MAX_DAYS = {"1m":7,"5m":60,"15m":60,"1h":730,"1d":36500,"1wk":36500}

INTERVAL_MINUTES = {"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080}

STRATEGIES   = ["EMA Crossover", "Simple Buy", "Simple Sell"]
SL_TYPES     = ["Custom Points", "Trailing SL", "Reverse EMA Crossover", "Risk Reward Based", "ATR Based"]
TARGET_TYPES = ["Custom Points", "Trailing Target", "EMA Crossover", "Risk Reward Based", "ATR Based"]

# ─────────────────────────────────────────────────────────────────────────────
#  COMMENTED STRATEGY — Price Crosses Threshold (not in dropdown, plug-in ready)
#
#  def price_crosses_threshold(prev_close, curr_close, threshold, direction, action):
#      """direction: "above"|"below"   action: "buy"|"sell" """
#      if direction == "above" and prev_close < threshold <= curr_close: return action
#      if direction == "below" and prev_close > threshold >= curr_close: return action
#      return None
#
#  Activate: add to STRATEGIES, add sidebar widgets, call in run_backtest/live_engine.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  DHAN CANDLESTICK FETCH  (commented — drop-in yfinance replacement)
#
#  from dhanhq import dhanhq
#  DHAN_INTERVAL_MAP = {"1m":"1","5m":"5","15m":"15","1h":"60","1d":"D"}
#
#  def fetch_candles_dhan(client_id, access_token, security_id,
#                         exchange_segment, instrument_type,
#                         interval, from_date, to_date):
#      dhan = dhanhq(client_id, access_token)
#      if interval != "D":
#          resp = dhan.intraday_minute_data(
#              security_id=security_id, exchange_segment=exchange_segment,
#              instrument_type=instrument_type, interval=interval,
#              from_date=from_date, to_date=to_date)
#      else:
#          resp = dhan.historical_daily_data(
#              security_id=security_id, exchange_segment=exchange_segment,
#              instrument_type=instrument_type, expiry_code=0,
#              from_date=from_date, to_date=to_date)
#      records = resp.get("data", [])
#      if not records: return pd.DataFrame()
#      df = pd.DataFrame(records)
#      df.rename(columns={"timestamp":"dt","open":"Open","high":"High",
#                         "low":"Low","close":"Close","volume":"Volume"}, inplace=True)
#      df["dt"] = pd.to_datetime(df["dt"], unit="s", utc=True)
#      df.set_index("dt", inplace=True)
#      df.index = df.index.tz_convert("Asia/Kolkata")
#      return df[["Open","High","Low","Close","Volume"]].sort_index()
#
#  Plug-in: in fetch_ohlcv(), replace yf.download() call with fetch_candles_dhan().
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def _clean(raw) -> pd.DataFrame | None:
    """Flatten MultiIndex cols, force string col names, ensure 1-D numeric series, convert to IST."""
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return None
    df = raw.copy()
    # Flatten MultiIndex (yfinance v0.2+ returns MultiIndex for single ticker too)
    if isinstance(df.columns, pd.MultiIndex):
        # Take first level which contains Open/High/Low/Close/Volume
        df.columns = [str(c[0]) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    need = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in need):
        return None
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[keep].copy()
    # Ensure each column is a plain 1-D float Series (NOT a DataFrame)
    for c in keep:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df[c] = pd.to_numeric(col, errors="coerce")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df.dropna(subset=need, inplace=True)
    df.sort_index(inplace=True)
    return df if not df.empty else None


def _best_warmup(display_period: str, interval: str) -> str:
    """Clamp warmup to yfinance's hard max for this interval (fixes 5m/15m + 1mo)."""
    max_d     = MAX_DAYS.get(interval, 36500)
    preferred = WARMUP_MAP.get(display_period, display_period)
    pref_d    = PERIOD_DAYS.get(preferred, PERIOD_DAYS.get(display_period, 30))
    if pref_d <= max_d:
        return preferred
    # Find largest valid period within max_d
    ok = {p: d for p, d in PERIOD_DAYS.items() if d <= max_d}
    return max(ok, key=lambda p: ok[p]) if ok else display_period


@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str):
    """Returns (df_full, df_display) or None on failure."""
    try:
        warmup = _best_warmup(period, interval)
        df_full = _clean(yf.download(
            ticker, period=warmup, interval=interval,
            auto_adjust=True, progress=False, prepost=False,
        ))
        if df_full is None or df_full.empty:
            return None
        d = PERIOD_DAYS.get(period)
        if d:
            cut        = datetime.now(IST) - timedelta(days=d)
            df_display = df_full[df_full.index >= cut].copy()
            if df_display.empty:
                df_display = df_full.copy()
        else:
            df_display = df_full.copy()
        return df_full, df_display
    except Exception:
        return None


@st.cache_data(ttl=20, show_spinner=False)
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
# INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def tv_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exact TradingView EMA match.
    TV formula: alpha = 2/(length+1), seed = first bar close, adjust=False.
    pandas ewm(span=period, adjust=False, min_periods=1) replicates this exactly.
    Deviation from TV arises only if we have fewer warmup bars than TV (TV uses
    all-time history). Fetch as much warmup as allowed to minimise this gap.
    """
    s = series.copy()
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return s.ewm(span=period, adjust=False, min_periods=1).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi = df["High"].astype(float)
    lo = df["Low"].astype(float)
    pc = df["Close"].astype(float).shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()


def ema_angle_deg(ema_s: pd.Series, idx: int) -> float:
    """Scale-independent EMA slope angle in degrees."""
    if idx < 1 or idx >= len(ema_s):
        return 0.0
    prev = float(ema_s.iloc[idx - 1])
    curr = float(ema_s.iloc[idx])
    if prev == 0:
        return 0.0
    return abs(float(np.degrees(np.arctan((curr - prev) / prev * 100.0))))


def add_indicators(df: pd.DataFrame, fast: int, slow: int,
                   atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    ef = tv_ema(df["Close"].astype(float), fast)
    es = tv_ema(df["Close"].astype(float), slow)
    df[f"EMA_{fast}"] = ef
    df[f"EMA_{slow}"] = es
    df["ATR"]         = compute_atr(df, atr_period)
    # Crossover signals on COMPLETED bars
    df["Signal"] = 0
    df.loc[(ef > es) & (ef.shift(1) <= es.shift(1)), "Signal"] =  1   # bullish cross
    df.loc[(ef < es) & (ef.shift(1) >= es.shift(1)), "Signal"] = -1   # bearish cross
    return df

# ════════════════════════════════════════════════════════════════════════════
# SL / TARGET
# ════════════════════════════════════════════════════════════════════════════

def calc_sl(entry, tt, sl_type, sl_pts, ef, es, rr, atr=0.0, atr_m=1.5):
    s = 1 if tt == "buy" else -1
    if sl_type == "Custom Points":         return entry - s * sl_pts
    if sl_type == "Trailing SL":           return entry - s * sl_pts
    if sl_type == "Reverse EMA Crossover":
        base = es if tt == "buy" else ef
        fb   = entry - s * sl_pts
        return min(base, fb) if tt == "buy" else max(base, fb)
    if sl_type == "Risk Reward Based":     return entry - s * sl_pts
    if sl_type == "ATR Based":
        v = atr if atr > 0 else sl_pts
        return entry - s * v * atr_m
    return entry - s * sl_pts


def calc_tgt(entry, tt, tgt_type, tgt_pts, sl, ef, es, rr, atr=0.0, atr_m=2.0):
    s = 1 if tt == "buy" else -1
    if tgt_type == "Custom Points":    return entry + s * tgt_pts
    if tgt_type == "Trailing Target":  return entry + s * tgt_pts   # display only
    if tgt_type == "EMA Crossover":    return None
    if tgt_type == "Risk Reward Based":
        risk = abs(entry - sl) if sl is not None else tgt_pts
        return entry + s * risk * rr
    if tgt_type == "ATR Based":
        v = atr if atr > 0 else tgt_pts
        return entry + s * v * atr_m
    return entry + s * tgt_pts

# ════════════════════════════════════════════════════════════════════════════
# ─────────────────────────  BACKTEST ENGINE  ─────────────────────────────────
#
#  EMA CROSSOVER  — how it works (and why it matches live trading):
#
#   Candle N closes  → crossover detected on CLOSE of bar N
#   Candle N+1 opens → entry at OPEN of bar N+1  (first tradeable price)
#
#   This matches live trading where:
#     • crossover is confirmed at candle N close
#     • entry order is placed at the OPEN of the very next candle
#
#  SIMPLE BUY / SIMPLE SELL — enter immediately, no signal wait:
#
#   Entry at CLOSE of the current bar (as soon as no position exists).
#   Equivalent to "enter at market now" in live trading.
#   After each exit, re-enter immediately on the next bar.
#
#  EXIT LOGIC (same for all strategies):
#
#   BUY  : SL vs candle LOW first → if LOW ≤ SL, exit at SL price.
#           Otherwise check Target vs candle HIGH → if HIGH ≥ Target, exit.
#   SELL : SL vs candle HIGH first → if HIGH ≥ SL, exit at SL price.
#           Otherwise check Target vs candle LOW → if LOW ≤ Target, exit.
#
#   This is conservative (SL first). If both breach the same candle,
#   SL is taken. This is the closest to live tick behaviour without real
#   tick data — in live trading the SL/Target are checked vs LTP every poll.
#
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df_full, df_display, strategy, fast, slow,
    sl_type, sl_pts, tgt_type, tgt_pts,
    qty, cd_en, cd_s, no_overlap, rr,
    atr_period, atr_sl_m, atr_tgt_m,
    adv_filter, min_angle, co_candle, candle_size_pts, candle_atr_m,
):
    # ── 1. Indicators on full df (warmup ensures no NaN in display window) ─
    df = add_indicators(df_full.copy(), fast, slow, atr_period)

    # ── 2. Warmup boundary ────────────────────────────────────────────────
    display_start = df_display.index[0] if not df_display.empty else df.index[0]
    warmup_rows   = max(slow * 3, int(df.index.searchsorted(display_start)))
    warmup_rows   = min(warmup_rows, len(df) - 2)

    trades:        list[dict] = []
    active:        dict | None = None
    pending_entry: dict | None = None   # EMA crossover: execute at N+1 open
    last_exit_ts               = None

    # ── 3. Bar-by-bar loop ────────────────────────────────────────────────
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
        atr_v = float(row["ATR"]) if pd.notna(row.get("ATR")) else sl_pts

        # ── 3a. EMA CROSSOVER: execute pending entry at THIS bar's OPEN ──
        #   (signal was on candle N; this is candle N+1 open)
        if pending_entry is not None and active is None:
            ep   = open_
            tt   = pending_entry["trade_type"]
            ef_p = pending_entry["ef"]; es_p = pending_entry["es"]; av_p = pending_entry["atr"]

            if cd_en and last_exit_ts is not None:
                if (ts - last_exit_ts).total_seconds() < cd_s:
                    pending_entry = None
                    continue

            sl_p  = calc_sl(ep, tt, sl_type, sl_pts, ef_p, es_p, rr, av_p, atr_sl_m)
            tgt_p = calc_tgt(ep, tt, tgt_type, tgt_pts, sl_p, ef_p, es_p, rr, av_p, atr_tgt_m)

            # Skip if open already gapped past SL
            gap_sl = (tt == "buy"  and sl_p is not None and ep <= sl_p) or \
                     (tt == "sell" and sl_p is not None and ep >= sl_p)
            if not gap_sl:
                active = {"entry_time": ts, "entry_price": ep,
                          "trade_type": tt, "sl": sl_p, "target": tgt_p,
                          "entry_reason": pending_entry["reason"]}
            pending_entry = None

        # ── 3b. EXIT LOGIC ────────────────────────────────────────────────
        #   BUY  : check SL vs LOW, then Target vs HIGH
        #   SELL : check SL vs HIGH, then Target vs LOW
        #   SL is always evaluated FIRST (conservative / worst-case).
        if active is not None:
            ep  = active["entry_price"]
            tt  = active["trade_type"]
            sl  = active["sl"]
            tgt = active["target"]

            if tt == "buy":
                sl_hit  = sl  is not None and lo <= sl
                tgt_hit = tgt is not None and hi >= tgt
            else:
                sl_hit  = sl  is not None and hi >= sl
                tgt_hit = tgt is not None and lo <= tgt

            ema_exit = (tgt_type == "EMA Crossover" and
                        ((tt == "buy" and sig == -1) or (tt == "sell" and sig == 1)))

            exit_px  = None
            exit_rsn = None

            # Conservative: SL first — same for BUY and SELL
            if sl_hit:
                exit_px  = sl
                exit_rsn = "Stop Loss Hit"
            elif tgt_hit:
                exit_px  = tgt
                exit_rsn = "Target Hit"
            elif ema_exit:
                exit_px  = cl
                exit_rsn = "EMA Crossover Exit"

            # Trailing SL update (when no exit yet)
            if sl_type == "Trailing SL" and exit_px is None:
                if tt == "buy":
                    nsl = cl - sl_pts
                    if nsl > active["sl"]: active["sl"] = nsl
                else:
                    nsl = cl + sl_pts
                    if nsl < active["sl"]: active["sl"] = nsl

            if exit_px is not None:
                pnl = ((float(exit_px) - ep) if tt == "buy" else (ep - float(exit_px))) * qty
                trades.append({
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
                })
                last_exit_ts = ts
                active       = None

        # ── 3c. ENTRY SIGNAL DETECTION ────────────────────────────────────
        if active is None and pending_entry is None:

            # Cooldown guard
            if cd_en and last_exit_ts is not None:
                if (ts - last_exit_ts).total_seconds() < cd_s:
                    continue

            trade_type = None
            reason     = ""

            if strategy == "EMA Crossover" and sig != 0:
                # ── Advanced filters (only when checkbox enabled) ─────────
                allow = True
                if adv_filter:
                    angle = ema_angle_deg(df[f"EMA_{fast}"], i)
                    if angle < min_angle:
                        allow = False
                    else:
                        body = abs(cl - open_)
                        if co_candle == "Custom Candle Size" and body < candle_size_pts:
                            allow = False
                        elif co_candle == "ATR Based Candle Size" and body < candle_atr_m * atr_v:
                            allow = False
                    angle_info = f" | angle {angle:.1f}°" if adv_filter else ""
                else:
                    angle_info = ""

                if allow:
                    if sig ==  1:
                        trade_type = "buy"
                        reason = f"EMA{fast} crossed above EMA{slow}{angle_info}"
                    else:
                        trade_type = "sell"
                        reason = f"EMA{fast} crossed below EMA{slow}{angle_info}"

                if trade_type is not None:
                    # ── EMA CROSSOVER: pending entry → executes at N+1 OPEN ──
                    pending_entry = {"trade_type": trade_type, "reason": reason,
                                     "ef": ef_v, "es": es_v, "atr": atr_v}
                    continue   # move to next bar for entry

            elif strategy == "Simple Buy":
                # ── SIMPLE BUY: enter at CLOSE of THIS bar (immediate) ────
                # Re-enters after every exit. Suitable for testing SL/Target behaviour.
                trade_type = "buy"
                reason     = "Simple Buy"

            elif strategy == "Simple Sell":
                trade_type = "sell"
                reason     = "Simple Sell"

            # Simple Buy/Sell: enter at current bar's CLOSE directly (no pending)
            if trade_type is not None and strategy in ["Simple Buy", "Simple Sell"]:
                sl_p  = calc_sl(cl, trade_type, sl_type, sl_pts, ef_v, es_v, rr, atr_v, atr_sl_m)
                tgt_p = calc_tgt(cl, trade_type, tgt_type, tgt_pts, sl_p, ef_v, es_v, rr, atr_v, atr_tgt_m)
                active = {"entry_time": ts, "entry_price": cl,
                          "trade_type": trade_type, "sl": sl_p, "target": tgt_p,
                          "entry_reason": reason}

    # ── 4. Close any open trade at end of data ────────────────────────────
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
            "Candle Low":   round(float(lr["Low"]),  2),
            "Entry Reason": active["entry_reason"],
            "Exit Reason":  "End of Data",
            "PnL (Rs)":     round(pnl, 2),
            "Qty":          qty,
        })

    return pd.DataFrame(trades)

# ════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ════════════════════════════════════════════════════════════════════════════

def _base(light: bool) -> dict:
    bg  = "#f5f7fa" if light else "#080b12"
    plt = "#ffffff"  if light else "#080b12"
    grd = "#e8edf5"  if light else "#0f1623"
    txt = "#334155"  if light else "#6b7a99"
    spk = "#94a3b8"  if light else "#2a3456"
    return dict(
        paper_bgcolor=bg, plot_bgcolor=plt,
        font=dict(family="JetBrains Mono", color=txt, size=11),
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=16, l=48, r=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#fff" if light else "#0f1623",
                        bordercolor="#d0d7e6" if light else "#1a2236",
                        font=dict(family="JetBrains Mono", size=11)),
        xaxis=dict(gridcolor=grd, zeroline=False, showspikes=True,
                   spikecolor=spk, spikethickness=1),
        yaxis=dict(gridcolor=grd, zeroline=False, showspikes=True,
                   spikecolor=spk, spikethickness=1),
    )


def build_chart(df: pd.DataFrame, fast: int, slow: int,
                trades_df=None, position=None, title="", light=False) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02)
    up  = "#00b896" if light else "#00e5b4"
    dn  = "#e0284a" if light else "#ff4d6d"
    bg0 = "#000"    if light else "#080b12"

    # Candles
    fig.add_trace(go.Candlestick(
        x=list(df.index),
        open=list(df["Open"].astype(float)),
        high=list(df["High"].astype(float)),
        low=list(df["Low"].astype(float)),
        close=list(df["Close"].astype(float)),
        name="OHLC",
        increasing=dict(line=dict(color=up, width=1.2), fillcolor=up),
        decreasing=dict(line=dict(color=dn, width=1.2), fillcolor=dn),
        whiskerwidth=0.4,
    ), row=1, col=1)

    # EMAs (always visible)
    for col_, clr, nm in [
        (f"EMA_{fast}", "#f59e0b", f"EMA {fast}"),
        (f"EMA_{slow}", "#3b82f6", f"EMA {slow}"),
    ]:
        if col_ in df.columns:
            fig.add_trace(go.Scatter(
                x=list(df.index), y=list(df[col_].astype(float)),
                name=nm, line=dict(color=clr, width=1.8),
            ), row=1, col=1)

    # Trade markers
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            tt_ = str(t["Trade Type"])
            c_  = up if tt_ == "BUY" else dn
            esy = "triangle-up"   if tt_ == "BUY" else "triangle-down"
            exy = "triangle-down" if tt_ == "BUY" else "triangle-up"
            ep_ = float(t["Entry Price"])
            xp_ = float(t["Exit Price"])
            p_  = float(t["PnL (Rs)"])
            fig.add_trace(go.Scatter(
                x=[t["Entry Time"]], y=[ep_], mode="markers", showlegend=False,
                marker=dict(symbol=esy, size=13, color=c_, line=dict(width=1.5, color="#fff")),
                hovertemplate=f"<b>{tt_} ENTRY</b><br>{ep_:.2f}<br>{t.get('Entry Reason','')}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t["Exit Time"]], y=[xp_], mode="markers", showlegend=False,
                marker=dict(symbol=exy, size=11, color=c_, opacity=0.6,
                            line=dict(width=1, color=bg0)),
                hovertemplate=f"<b>{tt_} EXIT</b><br>{xp_:.2f}<br>{t.get('Exit Reason','')}<br>PnL Rs{p_:+.2f}<extra></extra>",
            ), row=1, col=1)

    # Live position lines
    if position is not None:
        for lvl, clr_, lbl in [
            (position.get("entry_price"), "#334155" if light else "#ffffff", "Entry"),
            (position.get("sl"),          dn,                                "SL"),
            (position.get("target"),      up,                                "Target"),
        ]:
            if lvl is not None:
                fig.add_hline(y=float(lvl), row=1, col=1,
                              line=dict(color=clr_, width=1.4,
                                        dash="dash" if lbl != "Entry" else "solid"),
                              annotation_text=f" {lbl} {float(lvl):.2f}",
                              annotation_font=dict(color=clr_, size=11))

    # Volume
    if "Volume" in df.columns:
        vc = [up if float(c) >= float(o) else dn
              for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=list(df.index),
            y=list(df["Volume"].fillna(0).astype(float)),
            marker_color=vc, name="Vol", showlegend=False,
        ), row=2, col=1)

    base = _base(light)
    xax  = base.pop("xaxis", {})
    yax  = base.pop("yaxis", {})
    fig.update_layout(height=570,
                      title=dict(text=title, x=0.01,
                                 font=dict(family="Syne", size=13,
                                           color="#334155" if light else "#6b7a99")),
                      **base)
    fig.update_xaxes(**xax)
    fig.update_yaxes(**yax)
    return fig

# ════════════════════════════════════════════════════════════════════════════
# LTP WIDGET
# ════════════════════════════════════════════════════════════════════════════

def ltp_widget(ticker: str, label: str, light: bool) -> None:
    info = fetch_ltp_cached(ticker)
    if not info:
        st.warning(f"LTP unavailable for {label}.")
        return
    p, c, pct = info["price"], info["change"], info["pct"]
    up  = "#007a6a" if light else "#00e5b4"
    dn  = "#c0152d" if light else "#ff4d6d"
    col = up if c >= 0 else dn
    arr = "▲" if c >= 0 else "▼"
    bg  = "#ffffff" if light else "#0f1623"
    brd = "#d0d7e6" if light else "#1a2236"
    txt = "#1a2236" if light else "#e2e8f0"
    mut = "#7a8899" if light else "#4a5568"
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {brd};border-radius:12px;
                padding:14px 22px;margin-bottom:14px;display:flex;
                align-items:center;gap:28px;flex-wrap:wrap">
        <div>
            <div style="font-size:10px;font-weight:700;letter-spacing:2px;
                        text-transform:uppercase;color:{mut};font-family:Syne,sans-serif">{label}</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:28px;
                        font-weight:700;color:{txt};line-height:1.1">{p:,.2f}</div>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:17px;font-weight:600;color:{col}">{arr} {c:+.2f}</div>
        <div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:600;color:{col}">{pct:+.2f}%</div>
        <div style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:11px;color:{mut}">
            Prev Close: {info['prev']:,.2f}
        </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DHAN ORDERS
# ════════════════════════════════════════════════════════════════════════════

def _dhan(cfg): return dhanhq(str(cfg["dhan_client_id"]), str(cfg["dhan_access_token"]))


def place_entry_order(cfg: dict, tt: str, ltp: float) -> str:
    if not DHAN_OK: return "dhanhq not installed"
    try:
        dhan = _dhan(cfg)
        if cfg.get("options_trading"):
            sec = cfg["ce_security_id"] if tt == "buy" else cfg["pe_security_id"]
            ot  = "MARKET" if cfg.get("opts_entry_otype") == "Market Order" else "LIMIT"
            r   = dhan.place_order(transactionType="BUY",
                                   exchangeSegment=str(cfg.get("opts_exchange","NSE_FNO")),
                                   productType="INTRADAY", orderType=ot, validity="DAY",
                                   securityId=str(sec), quantity=int(cfg.get("opts_qty",65)),
                                   price=float(ltp) if ot=="LIMIT" else 0.0, triggerPrice=0)
        else:
            ex = "NSE_EQ" if cfg.get("eq_exchange","NSE") == "NSE" else "BSE"
            pt = "INTRADAY" if cfg.get("eq_product") == "Intraday" else "CNC"
            ot = "MARKET" if cfg.get("eq_entry_otype") == "Market Order" else "LIMIT"
            r  = dhan.place_order(security_id=str(cfg.get("eq_sec_id","1594")),
                                  exchange_segment=ex,
                                  transaction_type="BUY" if tt=="buy" else "SELL",
                                  quantity=int(cfg.get("eq_qty",1)),
                                  order_type=ot, product_type=pt,
                                  price=float(ltp) if ot=="LIMIT" else 0.0)
        return f"Entry OK: {r}"
    except Exception as e: return f"Entry FAILED: {e}"


def place_exit_order(cfg: dict, tt: str, ltp: float) -> str:
    if not DHAN_OK: return "dhanhq not installed"
    try:
        dhan = _dhan(cfg)
        if cfg.get("options_trading"):
            sec = cfg["ce_security_id"] if tt == "buy" else cfg["pe_security_id"]
            ot  = "MARKET" if cfg.get("opts_exit_otype","Market Order") == "Market Order" else "LIMIT"
            r   = dhan.place_order(transactionType="SELL",
                                   exchangeSegment=str(cfg.get("opts_exchange","NSE_FNO")),
                                   productType="INTRADAY", orderType=ot, validity="DAY",
                                   securityId=str(sec), quantity=int(cfg.get("opts_qty",65)),
                                   price=float(ltp) if ot=="LIMIT" else 0.0, triggerPrice=0)
        else:
            ex = "NSE_EQ" if cfg.get("eq_exchange","NSE") == "NSE" else "BSE"
            pt = "INTRADAY" if cfg.get("eq_product") == "Intraday" else "CNC"
            ot = "MARKET" if cfg.get("eq_exit_otype","Market Order") == "Market Order" else "LIMIT"
            r  = dhan.place_order(security_id=str(cfg.get("eq_sec_id","1594")),
                                  exchange_segment=ex,
                                  transaction_type="SELL" if tt=="buy" else "BUY",
                                  quantity=int(cfg.get("eq_qty",1)),
                                  order_type=ot, product_type=pt,
                                  price=float(ltp) if ot=="LIMIT" else 0.0)
        return f"Exit OK: {r}"
    except Exception as e: return f"Exit FAILED: {e}"

# ════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE  (background daemon thread)
# ════════════════════════════════════════════════════════════════════════════
#
#  Multi-user safety:
#    Thread only reads/writes `shared` (a plain dict from this user's
#    session_state) and checks `stop_event` (a threading.Event created
#    per-session). No global state touched. Two users run entirely separate
#    threads with separate dicts — zero cross-contamination.
#
#  Stop reliability:
#    Uses threading.Event.is_set() which is atomically readable across threads.
#    The thread exits within 1.5 s of stop_event.set() being called.
#
#  Entry logic (matches backtest):
#    EMA Crossover : signal on candle N → pending_entry → execute at candle N+1 OPEN
#    Simple Buy/Sell : enter immediately at current LTP (no candle wait)
#
#  Exit logic:
#    SL / Target checked vs LTP (tick data) every ~1.5 s
#    SL is checked FIRST (conservative, same as backtest)
# ════════════════════════════════════════════════════════════════════════════

def live_engine(cfg: dict, shared: dict, stop_event: threading.Event) -> None:
    ticker    = cfg["ticker"];    interval = cfg["interval"];  period = cfg["period"]
    strategy  = cfg["strategy"];  fast     = cfg["fast_ema"]; slow   = cfg["slow_ema"]
    sl_type   = cfg["sl_type"];   sl_pts   = cfg["sl_pts"]
    tgt_type  = cfg["tgt_type"];  tgt_pts  = cfg["tgt_pts"]
    qty       = cfg["qty"];       cd_en    = cfg["cd_en"];    cd_s   = cfg["cd_s"]
    rr        = cfg["rr"]
    atr_period= cfg.get("atr_period", 14)
    atr_sl_m  = cfg.get("atr_sl_mult", 1.5)
    atr_tgt_m = cfg.get("atr_tgt_mult", 2.0)
    adv_filt  = cfg.get("adv_filter", False)
    min_angle = cfg.get("min_angle", 0.0)
    co_candle = cfg.get("crossover_candle", "Simple Crossover")
    cs_pts    = cfg.get("candle_size_pts", 10.0)
    ca_mult   = cfg.get("candle_atr_mult", 1.0)

    def _log(msg: str) -> None:
        ts = datetime.now(IST).strftime("%H:%M:%S")
        shared["log"].append(f"[{ts}] {msg}")
        shared["log"] = shared["log"][-300:]

    _log("Engine started")
    last_closed_ts = None

    while not stop_event.is_set():
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=True, progress=False, prepost=False)
            df  = _clean(raw)

            if df is None or len(df) < max(slow * 2 + 5, 10):
                _log("Insufficient data — waiting…")
                stop_event.wait(timeout=1.5)
                continue

            # Add indicators
            df[f"EMA_{fast}"] = tv_ema(df["Close"].astype(float), fast)
            df[f"EMA_{slow}"] = tv_ema(df["Close"].astype(float), slow)
            df["ATR"]         = compute_atr(df, atr_period)

            # Values from COMPLETED candle (index[-2])
            ef_comp  = float(df[f"EMA_{fast}"].iloc[-2])
            es_comp  = float(df[f"EMA_{slow}"].iloc[-2])
            atr_comp = float(df["ATR"].iloc[-2])

            shared["chart_df"]     = df.copy()
            shared["ema_fast_val"] = ef_comp
            shared["ema_slow_val"] = es_comp
            shared["atr_val"]      = atr_comp

            ltp = float(df["Close"].iloc[-1])
            shared["ltp"] = ltp

            # ─── STEP A: Simple Buy/Sell — enter immediately at LTP ───────
            # No candle close wait required for these strategies.
            if strategy in ["Simple Buy", "Simple Sell"]:
                if shared.get("position") is None and shared.get("pending_entry") is None:
                    # Cooldown check
                    go_entry = True
                    if cd_en:
                        last_ex = shared.get("last_exit_ts")
                        if last_ex and (datetime.now(IST) - last_ex).total_seconds() < cd_s:
                            go_entry = False

                    if go_entry:
                        sig_tt = "buy" if strategy == "Simple Buy" else "sell"
                        sl_p  = calc_sl(ltp, sig_tt, sl_type, sl_pts, ef_comp, es_comp, rr, atr_comp, atr_sl_m)
                        tgt_p = calc_tgt(ltp, sig_tt, tgt_type, tgt_pts, sl_p, ef_comp, es_comp, rr, atr_comp, atr_tgt_m)
                        shared["position"] = {
                            "trade_type":   sig_tt,
                            "entry_price":  ltp,
                            "entry_time":   datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                            "sl":           sl_p,
                            "target":       tgt_p,
                            "entry_reason": strategy,
                        }
                        sl_s  = f"{sl_p:.2f}"  if sl_p  is not None else "None"
                        tgt_s = f"{tgt_p:.2f}" if tgt_p is not None else "None"
                        _log(f"ENTRY {sig_tt.upper()} @ LTP {ltp:.2f} | SL:{sl_s} | Tgt:{tgt_s}")
                        if cfg.get("dhan_en"):
                            _log(f"Broker: {place_entry_order(cfg, sig_tt, ltp)}")

            # ─── STEP B: EMA Crossover pending entry at N+1 OPEN ─────────
            pen = shared.get("pending_entry")
            if pen is not None and shared.get("position") is None:
                entry_open = float(df["Open"].iloc[-1])   # N+1 candle open
                tt = pen["trade_type"]
                sl_p  = calc_sl(entry_open, tt, sl_type, sl_pts, pen["ef"], pen["es"], rr, pen["atr"], atr_sl_m)
                tgt_p = calc_tgt(entry_open, tt, tgt_type, tgt_pts, sl_p, pen["ef"], pen["es"], rr, pen["atr"], atr_tgt_m)
                gap_sl = (tt == "buy"  and sl_p is not None and entry_open <= sl_p) or \
                         (tt == "sell" and sl_p is not None and entry_open >= sl_p)
                if not gap_sl:
                    shared["position"] = {
                        "trade_type":   tt,
                        "entry_price":  entry_open,
                        "entry_time":   datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "sl":           sl_p,
                        "target":       tgt_p,
                        "entry_reason": pen["reason"],
                    }
                    sl_s  = f"{sl_p:.2f}"  if sl_p  is not None else "None"
                    tgt_s = f"{tgt_p:.2f}" if tgt_p is not None else "None"
                    _log(f"ENTRY {tt.upper()} @ N+1 Open {entry_open:.2f} | SL:{sl_s} | Tgt:{tgt_s} | {pen['reason']}")
                    if cfg.get("dhan_en"):
                        _log(f"Broker: {place_entry_order(cfg, tt, entry_open)}")
                else:
                    _log(f"Entry skipped — open {entry_open:.2f} gapped past SL {sl_p:.2f}")
                shared["pending_entry"] = None

            # ─── STEP C: SL/Target check vs LTP (every tick) ─────────────
            pos = shared.get("position")
            if pos is not None:
                tt  = pos["trade_type"]
                ep  = float(pos["entry_price"])
                sl  = pos.get("sl")
                tgt = pos.get("target")

                exited     = False
                exit_px    = None
                exit_reason= None

                # Conservative SL first
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

                # EMA crossover exit (completed candle)
                if not exited and tgt_type == "EMA Crossover" and len(df) >= 3:
                    pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])
                    if tt == "buy"  and ef_comp < es_comp and pf >= ps:
                        exit_px = ltp; exit_reason = "EMA Crossover Exit"; exited = True
                    if tt == "sell" and ef_comp > es_comp and pf <= ps:
                        exit_px = ltp; exit_reason = "EMA Crossover Exit"; exited = True

                # Trailing SL update
                if sl_type == "Trailing SL" and not exited:
                    if tt == "buy":
                        nsl = ltp - sl_pts
                        if nsl > pos["sl"]:
                            pos["sl"] = nsl
                            shared["position"] = pos
                            _log(f"Trailing SL → {nsl:.2f}")
                    else:
                        nsl = ltp + sl_pts
                        if nsl < pos["sl"]:
                            pos["sl"] = nsl
                            shared["position"] = pos
                            _log(f"Trailing SL → {nsl:.2f}")

                if exited and exit_px is not None:
                    pnl = ((float(exit_px) - ep) if tt == "buy" else (ep - float(exit_px))) * qty
                    rec = {
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   tt.upper(),
                        "Entry Price":  round(ep, 2),
                        "Exit Price":   round(float(exit_px), 2),
                        "SL":           round(float(sl), 2)  if sl  is not None else "—",
                        "Target":       round(float(tgt), 2) if tgt is not None else "—",
                        "Entry Reason": pos.get("entry_reason",""),
                        "Exit Reason":  exit_reason,
                        "PnL (Rs)":     round(pnl, 2),
                        "Qty":          qty,
                        "Mode":         "Live",
                    }
                    shared["trade_history"].append(rec)
                    shared["position"]     = None
                    shared["last_exit_ts"] = datetime.now(IST)
                    if cfg.get("dhan_en"):
                        _log(f"Broker exit: {place_exit_order(cfg, tt, float(exit_px))}")
                    sign = "+" if pnl >= 0 else ""
                    _log(f"EXIT {tt.upper()} @ {float(exit_px):.2f} | {exit_reason} | PnL Rs{sign}{pnl:.2f}")

            # ─── STEP D: EMA crossover signal detection (new completed candle) ─
            current_closed_ts = df.index[-2] if len(df) >= 2 else None
            is_new_candle     = (current_closed_ts is not None and
                                 current_closed_ts != last_closed_ts)

            if (is_new_candle and strategy == "EMA Crossover" and
                    shared.get("position") is None and shared.get("pending_entry") is None):
                last_closed_ts = current_closed_ts

                if cd_en:
                    last_ex = shared.get("last_exit_ts")
                    if last_ex and (datetime.now(IST) - last_ex).total_seconds() < cd_s:
                        remain = cd_s - (datetime.now(IST) - last_ex).total_seconds()
                        _log(f"Cooldown: {remain:.0f}s remaining")
                        stop_event.wait(timeout=1.5)
                        continue

                if len(df) >= 3:
                    pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])
                    bullish = ef_comp > es_comp and pf <= ps
                    bearish = ef_comp < es_comp and pf >= ps

                    if bullish or bearish:
                        allow = True
                        angle_info = ""
                        if adv_filt:
                            angle = ema_angle_deg(df[f"EMA_{fast}"], len(df) - 2)
                            angle_info = f" ({angle:.1f}°)"
                            if angle < min_angle:
                                allow = False
                            elif co_candle != "Simple Crossover":
                                body = abs(float(df["Close"].iloc[-2]) - float(df["Open"].iloc[-2]))
                                if co_candle == "Custom Candle Size" and body < cs_pts:
                                    allow = False
                                elif co_candle == "ATR Based Candle Size" and body < ca_mult * atr_comp:
                                    allow = False

                        if allow:
                            sig_tt = "buy" if bullish else "sell"
                            rsn    = f"EMA{fast}xEMA{slow} {'↑' if bullish else '↓'}{angle_info}"
                            _log(f"Signal: {sig_tt.upper()} on {str(current_closed_ts)[:16]} — entry at next candle open")
                            shared["pending_entry"] = {
                                "trade_type": sig_tt, "reason": rsn,
                                "ef": ef_comp, "es": es_comp, "atr": atr_comp,
                            }
            elif is_new_candle:
                last_closed_ts = current_closed_ts  # keep tracking even if no signal

        except Exception as exc:
            _log(f"Error: {exc}")

        stop_event.wait(timeout=1.5)   # ← replaces time.sleep; responds to stop instantly

    _log("Engine stopped")
    shared["running"] = False   # ensure UI shows stopped state

# ════════════════════════════════════════════════════════════════════════════
# ████████████████████████   MAIN APP   ██████████████████████████████████████
# ════════════════════════════════════════════════════════════════════════════

def main():
    light  = st.session_state.get("light_theme", True)
    inject_css(light)
    shared = st.session_state["live_shared"]

    # ════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════════════════
    with st.sidebar:
        lc = "#007a6a" if light else "#00e5b4"
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0 10px">
            <div style="font-size:38px">📈</div>
            <div style="font-family:Syne,sans-serif;font-size:18px;font-weight:800;
                        letter-spacing:3px;color:{lc}">SMART INVESTING</div>
            <div style="font-size:9px;color:#8899aa;letter-spacing:2px;margin-top:2px">
                ALGORITHMIC TRADING PLATFORM
            </div>
        </div>""", unsafe_allow_html=True)

        lt = st.checkbox("☀ Light Theme", value=light, key="light_theme")
        if lt != light:
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

        # ── TIMEFRAME  (default 5m / 1mo) ────────────────────────────────
        st.markdown('<div class="shdr">Timeframe</div>', unsafe_allow_html=True)
        tf_keys = list(TIMEFRAME_PERIODS.keys())
        c1, c2  = st.columns(2)
        with c1:
            interval = st.selectbox("Interval", tf_keys,
                                    index=tf_keys.index("5m"),
                                    label_visibility="collapsed")
        with c2:
            prd_opts  = TIMEFRAME_PERIODS[interval]
            dflt_prd  = prd_opts.index("1mo") if "1mo" in prd_opts else min(1, len(prd_opts)-1)
            period    = st.selectbox("Period", prd_opts, index=dflt_prd,
                                     label_visibility="collapsed")

        st.divider()

        # ── STRATEGY ─────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Strategy</div>', unsafe_allow_html=True)
        strategy = st.selectbox("Strategy", STRATEGIES, label_visibility="collapsed")

        fast_ema = 9; slow_ema = 15
        adv_filter = False; min_angle = 0.0
        crossover_candle = "Simple Crossover"
        candle_size_pts  = 10.0; candle_atr_mult = 1.0

        if strategy == "EMA Crossover":
            e1, e2 = st.columns(2)
            with e1: fast_ema = st.number_input("Fast EMA", 2, 200, 9)
            with e2: slow_ema = st.number_input("Slow EMA", 2, 500, 15)

            adv_filter = st.checkbox("Advanced Entry Filters", value=False,
                                     help="Disable = plain EMA crossover. Enable to add angle/candle size filters.")
            if adv_filter:
                min_angle = st.number_input("Min Crossover Angle (°)", 0.0, 90.0, 0.0, 0.5,
                                            help="0 = no angle filter. Increase to require steeper crossover.")
                crossover_candle = st.selectbox("Candle Size Filter",
                                                ["Simple Crossover",
                                                 "Custom Candle Size",
                                                 "ATR Based Candle Size"])
                if crossover_candle == "Custom Candle Size":
                    candle_size_pts = st.number_input("Min Body (pts)", 0.1, 1e6, 10.0, 0.5)
                elif crossover_candle == "ATR Based Candle Size":
                    candle_atr_mult = st.number_input("Min Body (ATR×)", 0.1, 10.0, 1.0, 0.1)

        st.divider()

        # ── STOP LOSS ────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Stop Loss</div>', unsafe_allow_html=True)
        sl_type = st.selectbox("SL Type", SL_TYPES, label_visibility="collapsed")
        s1, s2  = st.columns(2)
        with s1: sl_pts = st.number_input("SL Points", 0.1, 1e6, 10.0, 0.5)
        with s2: rr     = st.number_input("R:R", 0.5, 20.0, 2.0, 0.5,
                                          help="Risk:Reward ratio for RR-based SL/Target")

        # ── TARGET ───────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Target</div>', unsafe_allow_html=True)
        tgt_type = st.selectbox("Target Type", TARGET_TYPES, label_visibility="collapsed")
        tgt_pts  = st.number_input("Target Points", 0.1, 1e6, 20.0, 0.5)

        atr_period = 14; atr_sl_mult = 1.5; atr_tgt_mult = 2.0
        if sl_type == "ATR Based" or tgt_type == "ATR Based":
            st.markdown("**ATR Config**")
            atr_period = st.number_input("ATR Period", 2, 200, 14)
            if sl_type  == "ATR Based": atr_sl_mult  = st.number_input("ATR × SL",  0.1, 10.0, 1.5, 0.1)
            if tgt_type == "ATR Based": atr_tgt_mult = st.number_input("ATR × TGT", 0.1, 10.0, 2.0, 0.1)

        st.divider()

        # ── TRADE SETTINGS ───────────────────────────────────────────────
        st.markdown('<div class="shdr">Trade Settings</div>', unsafe_allow_html=True)
        qty = st.number_input("Quantity", 1, 10_000_000, 1)
        d1, d2 = st.columns([1.4, 1])
        with d1: cd_en = st.checkbox("Cooldown", value=True)
        with d2: cd_s  = st.number_input("Secs", 1, 86400, 5, disabled=not cd_en,
                                          label_visibility="visible")
        no_overlap = st.checkbox("No Overlapping Trades", value=True)

        st.divider()

        # ── DHAN BROKER ──────────────────────────────────────────────────
        st.markdown('<div class="shdr">Dhan Broker</div>', unsafe_allow_html=True)
        dhan_en  = st.checkbox("Enable Dhan Broker", value=False)
        dhan_cfg: dict = {"dhan_en": dhan_en}

        if dhan_en:
            if not DHAN_OK: st.warning("pip install dhanhq")
            dhan_cfg["dhan_client_id"]    = st.text_input("Client ID",    "", type="password")
            dhan_cfg["dhan_access_token"] = st.text_input("Access Token", "", type="password")
            opts_en = st.checkbox("Options Trading", value=False)
            dhan_cfg["options_trading"] = opts_en
            if not opts_en:
                dhan_cfg["eq_product"]     = st.selectbox("Product",     ["Intraday","Delivery"])
                dhan_cfg["eq_exchange"]    = st.selectbox("Exchange",    ["NSE","BSE"])
                dhan_cfg["eq_sec_id"]      = st.text_input("Sec ID",     "1594")
                dhan_cfg["eq_qty"]         = st.number_input("Broker Qty",1,1_000_000,1)
                dhan_cfg["eq_entry_otype"] = st.selectbox("Entry Order", ["Limit Order","Market Order"])
                dhan_cfg["eq_exit_otype"]  = st.selectbox("Exit Order",  ["Market Order","Limit Order"])
            else:
                dhan_cfg["opts_exchange"]    = st.selectbox("FnO Exchange",["NSE_FNO","BSE_FNO"])
                dhan_cfg["ce_security_id"]   = st.text_input("CE Sec ID","")
                dhan_cfg["pe_security_id"]   = st.text_input("PE Sec ID","")
                dhan_cfg["opts_qty"]         = st.number_input("Lots",1,1_000_000,65)
                dhan_cfg["opts_entry_otype"] = st.selectbox("Entry Order",["Market Order","Limit Order"])
                dhan_cfg["opts_exit_otype"]  = st.selectbox("Exit Order", ["Market Order","Limit Order"])

        st.divider()
        st.caption("Smart Investing v4.0  •  Educational use only")

    # Full config bundle
    cfg: dict = {
        "ticker": ticker_sym, "ticker_label": ticker_label,
        "interval": interval, "period": period,
        "strategy": strategy, "fast_ema": fast_ema, "slow_ema": slow_ema,
        "sl_type": sl_type, "sl_pts": sl_pts,
        "tgt_type": tgt_type, "tgt_pts": tgt_pts,
        "qty": qty, "cd_en": cd_en, "cd_s": cd_s,
        "no_overlap": no_overlap, "rr": rr,
        "atr_period": atr_period, "atr_sl_mult": atr_sl_mult, "atr_tgt_mult": atr_tgt_mult,
        "adv_filter": adv_filter, "min_angle": min_angle,
        "crossover_candle": crossover_candle,
        "candle_size_pts": candle_size_pts, "candle_atr_mult": candle_atr_mult,
        **dhan_cfg,
    }

    # ── Page header ───────────────────────────────────────────────────────
    hc = "#007a6a" if light else "#00e5b4"
    st.markdown(f"""
    <div style="padding:4px 0 10px">
        <span style="font-family:Syne,sans-serif;font-size:30px;font-weight:800;color:{hc}">
            📈 Smart Investing
        </span>
        <span style="font-family:Syne,sans-serif;font-size:12px;color:#7a8899;
                     margin-left:14px;vertical-align:middle">
            Algorithmic Trading Platform
        </span>
    </div>""", unsafe_allow_html=True)

    if not AUTO_REFRESH_OK:
        st.info("Install `streamlit-autorefresh` for live tab auto-refresh.")

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

        r1, r2 = st.columns([4, 1])
        with r1:
            run_btn = st.button("▶  Run Backtest", type="primary",
                                use_container_width=True, key="run_bt")
        with r2:
            if st.button("↺ Clear", use_container_width=True, key="clr_bt"):
                st.session_state.update(bt_results=None, bt_violations=[], bt_chart_df=None, bt_run_key=None)
                st.rerun()

        if run_btn:
            with st.spinner(f"Fetching {ticker_label}  {interval}/{period}…"):
                res = fetch_ohlcv(ticker_sym, interval, period)
            if res is None:
                st.error(f"Data fetch failed for {ticker_sym} {interval}/{period}.")
            else:
                df_full, df_display = res
                with st.spinner("Running backtest…"):
                    tdf = run_backtest(
                        df_full, df_display, strategy, fast_ema, slow_ema,
                        sl_type, sl_pts, tgt_type, tgt_pts,
                        qty, cd_en, cd_s, no_overlap, rr,
                        atr_period, atr_sl_mult, atr_tgt_mult,
                        adv_filter, min_angle, crossover_candle,
                        candle_size_pts, candle_atr_mult,
                    )
                df_chart = add_indicators(df_full.copy(), fast_ema, slow_ema, atr_period)
                if not df_display.empty:
                    df_chart = df_chart.loc[df_display.index[0]:]

                # Results stored in isolated bt_ keys (not in live_shared)
                st.session_state["bt_results"]  = tdf
                st.session_state["bt_chart_df"] = df_chart
                st.session_state["bt_run_key"]  = f"{ticker_sym}|{interval}|{period}|{strategy}"

        tdf = st.session_state.get("bt_results")
        dfc = st.session_state.get("bt_chart_df")
        run_key = st.session_state.get("bt_run_key")

        if tdf is not None:
            if run_key:
                st.caption(f"Results for: `{run_key}`")

            if tdf.empty:
                st.warning("No trades generated. Adjust strategy parameters or check ticker data.")
            else:
                n    = len(tdf)
                wins = int((tdf["PnL (Rs)"] > 0).sum())
                loss = n - wins
                tot  = float(tdf["PnL (Rs)"].sum())
                acc  = wins / n * 100 if n else 0
                avgw = float(tdf.loc[tdf["PnL (Rs)"] > 0, "PnL (Rs)"].mean()) if wins else 0.0
                best = float(tdf["PnL (Rs)"].max())
                wst  = float(tdf["PnL (Rs)"].min())

                m = st.columns(8)
                for c_, l_, v_ in zip(m,
                    ["Trades","Wins","Losses","Accuracy",
                     "Total PnL","Avg Win","Best","Worst"],
                    [n, wins, loss, f"{acc:.1f}%",
                     f"Rs{tot:+,.0f}", f"Rs{avgw:+.0f}",
                     f"Rs{best:+.0f}", f"Rs{wst:+.0f}"]):
                    c_.metric(l_, v_)

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

                wbg = "rgba(0,150,120,0.10)" if light else "rgba(0,229,180,0.09)"
                lbg = "rgba(220,50,80,0.08)"  if light else "rgba(255,77,109,0.09)"
                wfg = "#004d3a" if light else "#00e5b4"
                lfg = "#7a0000" if light else "#ff8fa3"
                nfg = "#1a2236" if light else "#c8d0e0"

                def _srow(row):
                    p = row["PnL (Rs)"]
                    if p > 0:   return [f"background-color:{wbg};color:{wfg}" for _ in row]
                    elif p < 0: return [f"background-color:{lbg};color:{lfg}" for _ in row]
                    return [f"color:{nfg}" for _ in row]

                styled = (disp.style
                          .apply(_srow, axis=1)
                          .format({"PnL (Rs)":"Rs{:+.2f}",
                                   "Entry Price":"{:.2f}","Exit Price":"{:.2f}"}))
                st.dataframe(styled, use_container_width=True, height=420,
                             column_config={
                                 "Trade Type": st.column_config.TextColumn(width=70),
                             })

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — LIVE TRADING
    # ════════════════════════════════════════════════════════════════════
    with tab_live:
        # Auto-refresh every 2 s while running
        if shared.get("running") and AUTO_REFRESH_OK:
            st_autorefresh(interval=2000, key="live_ar")

        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### ⚡ Live Trading")

        running = shared.get("running", False)

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

        # ── Control buttons ───────────────────────────────────────────────
        b1, b2, b3, _ = st.columns([1.2, 1.2, 1.5, 3])
        with b1:
            if st.button("▶ Start", type="primary", use_container_width=True,
                         disabled=running, key="live_start"):
                # Create fresh stop event (ensures previous thread is dead)
                new_evt = threading.Event()
                st.session_state["live_stop_evt"] = new_evt
                # Fresh live shared dict
                st.session_state["live_shared"] = _new_live_shared()
                shared = st.session_state["live_shared"]
                shared["running"] = True
                shared["cfg"]     = cfg.copy()
                threading.Thread(
                    target=live_engine,
                    args=(cfg, shared, new_evt),
                    daemon=True,
                ).start()
                st.rerun()

        with b2:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not running, key="live_stop"):
                # Signal the thread via threading.Event (reliable across threads)
                evt = st.session_state.get("live_stop_evt")
                if evt is not None:
                    evt.set()
                shared["running"] = False
                time.sleep(0.3)
                st.rerun()

        with b3:
            if st.button("⚡ Square Off", use_container_width=True, key="live_sq"):
                pos = shared.get("position")
                if pos:
                    ltp_now = shared.get("ltp") or float(pos["entry_price"])
                    tt_sq   = pos["trade_type"]
                    pnl = ((ltp_now - float(pos["entry_price"])) if tt_sq == "buy"
                           else (float(pos["entry_price"]) - ltp_now)) * qty
                    shared["trade_history"].append({
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   tt_sq.upper(),
                        "Entry Price":  round(float(pos["entry_price"]), 2),
                        "Exit Price":   round(float(ltp_now), 2),
                        "SL":           round(float(pos["sl"]), 2)     if pos.get("sl")     is not None else "—",
                        "Target":       round(float(pos["target"]), 2) if pos.get("target") is not None else "—",
                        "Entry Reason": pos.get("entry_reason",""),
                        "Exit Reason":  "Manual Square Off",
                        "PnL (Rs)":     round(pnl, 2),
                        "Qty":          qty, "Mode": "Live",
                    })
                    shared["position"]     = None
                    shared["pending_entry"]= None
                    shared["last_exit_ts"] = datetime.now(IST)
                    if cfg.get("dhan_en"):
                        place_exit_order(cfg, tt_sq, float(ltp_now))
                    st.success(f"Squared off @ {float(ltp_now):.2f}  |  PnL Rs{pnl:+.2f}")
                else:
                    st.info("No open position.")
                st.rerun()

        st.divider()

        # Active config
        live_cfg_shown = shared.get("cfg") or cfg
        with st.expander("⚙ Active Configuration", expanded=True):
            cc = st.columns(5)
            cc[0].metric("Ticker",   live_cfg_shown.get("ticker_label","—"))
            cc[1].metric("Interval", live_cfg_shown.get("interval","—"))
            cc[2].metric("Period",   live_cfg_shown.get("period","—"))
            cc[3].metric("Strategy", live_cfg_shown.get("strategy","—"))
            cc[4].metric("Qty",      live_cfg_shown.get("qty","—"))
            cc2 = st.columns(4)
            cc2[0].metric(f"EMA {live_cfg_shown.get('fast_ema','—')} (fast)", "—" if shared.get("ema_fast_val") is None else f"{shared['ema_fast_val']:.2f}")
            cc2[1].metric(f"EMA {live_cfg_shown.get('slow_ema','—')} (slow)", "—" if shared.get("ema_slow_val") is None else f"{shared['ema_slow_val']:.2f}")
            cc2[2].metric("SL",  f"{live_cfg_shown.get('sl_type','—')} / {live_cfg_shown.get('sl_pts','—')} pts")
            cc2[3].metric("TGT", f"{live_cfg_shown.get('tgt_type','—')} / {live_cfg_shown.get('tgt_pts','—')} pts")

        st.divider()

        main_col, log_col = st.columns([1.7, 1])

        with main_col:
            ef_v  = shared.get("ema_fast_val")
            es_v  = shared.get("ema_slow_val")
            atr_v = shared.get("atr_val")
            ltp_c = shared.get("ltp")
            pos   = shared.get("position")
            pen   = shared.get("pending_entry")

            # EMA metrics row
            if ef_v is not None and es_v is not None:
                diff  = ef_v - es_v
                trend = "Bullish ↑" if diff > 0 else "Bearish ↓"
                mv    = st.columns(5)
                mv[0].metric(f"EMA {fast_ema}",  f"{ef_v:.2f}")
                mv[1].metric(f"EMA {slow_ema}",  f"{es_v:.2f}")
                mv[2].metric("EMA Diff",  f"{diff:+.2f}",
                             delta=trend, delta_color="normal" if diff > 0 else "inverse")
                mv[3].metric("ATR",  f"{atr_v:.2f}" if atr_v else "—")
                mv[4].metric("LTP",  f"{ltp_c:.2f}" if ltp_c else "—")
            else:
                st.info("Start the engine to see live EMA and LTP values.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Pending entry notice
            if pen is not None:
                pa = "#007a6a" if light else "#00e5b4"
                pb = "rgba(0,120,100,.07)" if light else "rgba(0,229,180,.07)"
                st.markdown(f"""
                <div style="background:{pb};border:1px solid {pa};border-radius:8px;
                            padding:10px 16px;margin-bottom:10px;font-family:Syne,sans-serif;font-size:12px">
                    ⏳ <b style="color:{pa}">Pending {pen['trade_type'].upper()}</b>
                    — signal confirmed. Entry at next candle open.<br>
                    <small style="color:#8899aa">{pen.get('reason','')}</small>
                </div>""", unsafe_allow_html=True)

            # Open position card
            if pos is not None:
                tt    = pos["trade_type"]
                ep    = float(pos["entry_price"])
                sl_   = pos.get("sl")
                tgt_  = pos.get("target")
                pnl_live = ((ltp_c - ep) if (ltp_c is not None and tt == "buy")
                            else ((ep - ltp_c) if ltp_c is not None else 0.0)) * qty
                pos_cls = "card-green" if pnl_live >= 0 else "card-red"
                tt_clr  = ("#00b896" if light else "#00e5b4") if tt == "buy" else ("#e0284a" if light else "#ff4d6d")
                pnl_clr = ("#007a6a" if light else "#00e5b4") if pnl_live >= 0 else ("#c0152d" if light else "#ff4d6d")
                tx      = "#1a2236" if light else "#c8d0e0"
                mt      = "#7a8899" if light else "#4a5568"
                sl_s  = f"{float(sl_):.2f}"  if sl_  is not None else "—"
                tgt_s = f"{float(tgt_):.2f}" if tgt_ is not None else "N/A"
                ltp_s = f"{ltp_c:.2f}"        if ltp_c is not None else "—"

                st.markdown(f"""
                <div class="card {pos_cls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                    <span style="font-family:Syne,sans-serif;font-size:13px;font-weight:800;
                                 letter-spacing:2px;color:{tt_clr}">● OPEN {tt.upper()}</span>
                    <div style="text-align:right">
                      <div style="font-size:10px;color:{mt};font-family:Syne,sans-serif">LIVE PnL</div>
                      <div style="font-family:JetBrains Mono,monospace;font-size:22px;
                                  font-weight:700;color:{pnl_clr}">Rs{pnl_live:+.2f}</div>
                    </div>
                  </div>
                  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;
                              font-family:JetBrains Mono,monospace;font-size:13px;color:{tx}">
                    <div><div style="color:{mt};font-size:10px">ENTRY</div>{ep:.2f}</div>
                    <div><div style="color:{mt};font-size:10px">LTP</div>{ltp_s}</div>
                    <div><div style="color:#e0284a;font-size:10px">SL</div>{sl_s}</div>
                    <div><div style="color:{("#007a6a" if light else "#00e5b4")};font-size:10px">TARGET</div>{tgt_s}</div>
                    <div><div style="color:{mt};font-size:10px">QTY</div>{qty}</div>
                  </div>
                  <div style="margin-top:10px;font-family:Syne,sans-serif;font-size:11px;color:{mt}">
                    {pos["entry_time"]}  ·  {pos.get("entry_reason","")}
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                bg2 = "#fff" if light else "#0f1623"
                bd2 = "#d0d7e6" if light else "#1a2236"
                st.markdown(f"""
                <div style="background:{bg2};border:1px solid {bd2};border-radius:12px;
                            text-align:center;padding:22px;color:#7a8899;
                            font-family:Syne,sans-serif;font-size:13px">
                    No open position
                </div>""", unsafe_allow_html=True)

            # Chart
            df_live = shared.get("chart_df")
            if df_live is not None and not df_live.empty:
                st.plotly_chart(
                    build_chart(df_live, fast_ema, slow_ema, position=pos,
                                title=f"Live · {ticker_label} · {interval}",
                                light=light),
                    use_container_width=True,
                )
                # Last candle row
                lr   = df_live.iloc[-1]
                ec_f = f"EMA_{fast_ema}"; ec_s = f"EMA_{slow_ema}"
                st.markdown("**Last Fetched Candle**")
                st.dataframe(pd.DataFrame([{
                    "Time":  str(df_live.index[-1]),
                    "Open":  round(float(lr["Open"]),  2),
                    "High":  round(float(lr["High"]),  2),
                    "Low":   round(float(lr["Low"]),   2),
                    "Close": round(float(lr["Close"]), 2),
                    "Vol":   int(float(lr.get("Volume", 0))),
                    f"EMA{fast_ema}": round(float(df_live[ec_f].iloc[-1]),2) if ec_f in df_live.columns else "—",
                    f"EMA{slow_ema}": round(float(df_live[ec_s].iloc[-1]),2) if ec_s in df_live.columns else "—",
                    "ATR": round(float(df_live["ATR"].iloc[-1]),2) if "ATR" in df_live.columns else "—",
                }]), use_container_width=True, hide_index=True)

        with log_col:
            st.markdown("**Activity Log**")
            logs = shared.get("log", [])
            sep  = "#eef1f7" if light else "#0f1623"
            tx2  = "#1a2236" if light else "#c8d0e0"
            mt2  = "#7a8899" if light else "#4a5568"
            log_html = (
                "".join(f"<div style='padding:2px 0;border-bottom:1px solid {sep};color:{tx2}'>{l}</div>"
                        for l in reversed(logs))
                if logs else f"<div style='color:{mt2}'>No activity — start the engine.</div>"
            )
            st.markdown(f'<div class="logbox">{log_html}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — TRADE HISTORY  (live trades only)
    # ════════════════════════════════════════════════════════════════════
    with tab_hist:
        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### 📋 Trade History")
        st.caption("All completed live trades — auto-updates while engine is running.")

        hist = shared.get("trade_history", [])

        _, hcol = st.columns([5, 1])
        with hcol:
            if st.button("🗑 Clear", use_container_width=True, key="clr_hist"):
                shared["trade_history"] = []
                st.rerun()

        bg3 = "#fff" if light else "#0f1623"; brd3 = "#d0d7e6" if light else "#1a2236"

        if not hist:
            st.markdown(f"""
            <div style="background:{bg3};border:1px solid {brd3};border-radius:12px;
                        text-align:center;padding:40px;color:#7a8899;
                        font-family:Syne,sans-serif">
                No live trades yet.
            </div>""", unsafe_allow_html=True)
        else:
            hdf     = pd.DataFrame(hist)
            tot_pnl = float(hdf["PnL (Rs)"].sum())
            wins_h  = int((hdf["PnL (Rs)"] > 0).sum())
            n_h     = len(hdf)
            acc_h   = wins_h / n_h * 100 if n_h else 0
            avg_p   = float(hdf["PnL (Rs)"].mean())
            best_h  = float(hdf["PnL (Rs)"].max())
            worst_h = float(hdf["PnL (Rs)"].min())

            hm = st.columns(6)
            for c_, l_, v_ in zip(hm,
                ["Trades","Win Rate","Total PnL","Avg PnL","Best","Worst"],
                [n_h, f"{acc_h:.1f}%", f"Rs{tot_pnl:+,.0f}",
                 f"Rs{avg_p:+.0f}", f"Rs{best_h:+.0f}", f"Rs{worst_h:+.0f}"]):
                c_.metric(l_, v_)

            st.divider()

            wbg = "rgba(0,150,120,0.10)" if light else "rgba(0,229,180,0.09)"
            lbg = "rgba(220,50,80,0.08)"  if light else "rgba(255,77,109,0.09)"
            wfg = "#004d3a" if light else "#00e5b4"
            lfg = "#7a0000" if light else "#ff8fa3"
            nfg = "#1a2236" if light else "#c8d0e0"

            def _hs(row):
                p = row["PnL (Rs)"]
                if p > 0:   return [f"background-color:{wbg};color:{wfg}" for _ in row]
                elif p < 0: return [f"background-color:{lbg};color:{lfg}" for _ in row]
                return [f"color:{nfg}" for _ in row]

            styled_h = (hdf.style.apply(_hs, axis=1)
                        .format({"PnL (Rs)":"Rs{:+.2f}",
                                 "Entry Price":"{:.2f}","Exit Price":"{:.2f}"}))
            st.dataframe(styled_h, use_container_width=True, height=420)

            st.markdown("#### Cumulative PnL")
            hdf["Cum PnL"] = hdf["PnL (Rs)"].cumsum()
            up_c = "#00b896" if light else "#00e5b4"
            dn_c = "#e0284a" if light else "#ff4d6d"
            base = _base(light); xax = base.pop("xaxis",{}); yax = base.pop("yaxis",{})
            fp   = go.Figure()
            fp.add_trace(go.Scatter(
                x=list(range(1, n_h + 1)),
                y=list(hdf["Cum PnL"].astype(float)),
                mode="lines+markers",
                line=dict(color=up_c, width=2),
                marker=dict(size=5, color=[up_c if p >= 0 else dn_c for p in hdf["PnL (Rs)"]]),
                fill="tozeroy",
                fillcolor="rgba(0,180,140,0.08)" if light else "rgba(0,229,180,0.06)",
            ))
            fp.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dash"))
            fp.update_layout(height=260, showlegend=False,
                             xaxis_title="Trade #", yaxis_title="PnL (Rs)", **base)
            fp.update_xaxes(**xax); fp.update_yaxes(**yax)
            st.plotly_chart(fp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
