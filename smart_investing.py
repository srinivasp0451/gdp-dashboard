# ══════════════════════════════════════════════════════════════════════════════
#  SMART INVESTING — Algorithmic Trading Platform
#  Version 1.0.0
#
#  Install:
#    pip install streamlit yfinance pandas numpy plotly pytz
#    pip install dhanhq           # optional – Dhan broker
#    pip install streamlit-autorefresh  # optional – live tab auto-refresh
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

# ── Optional libraries ───────────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [data-testid="stApp"]      { background:#080b12 !important; color:#c8d0e0; font-family:'Syne',sans-serif; }
[data-testid="stSidebar"]              { background:#0c1018 !important; border-right:1px solid #1a2236; }
[data-testid="stSidebar"] *            { font-family:'Syne',sans-serif; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]      { background:transparent; border-bottom:1px solid #1a2236; gap:4px; }
.stTabs [data-baseweb="tab"]           { font-family:'Syne',sans-serif; font-weight:700; font-size:13px; color:#4a5568; padding:10px 22px; border-radius:6px 6px 0 0; border:1px solid transparent; transition:all .2s; }
.stTabs [aria-selected="true"]         { color:#00e5b4 !important; background:rgba(0,229,180,.06) !important; border-color:#1a2236 #1a2236 transparent !important; }
.stTabs [data-baseweb="tab"]:hover     { color:#8899bb !important; }

/* ── Metrics ── */
[data-testid="metric-container"]       { background:#0f1623; border:1px solid #1a2236; border-radius:10px; padding:12px 16px; }
[data-testid="stMetricLabel"]          { font-family:'Syne',sans-serif; font-size:10px; color:#4a5568 !important; letter-spacing:.6px; text-transform:uppercase; }
[data-testid="stMetricValue"]          { font-family:'JetBrains Mono',monospace; font-size:17px; color:#c8d0e0; }
[data-testid="stMetricDelta"]          { font-family:'JetBrains Mono',monospace; font-size:12px; }

/* ── Buttons ── */
.stButton button                       { border-radius:8px; font-family:'Syne',sans-serif; font-weight:700; font-size:13px; border:1px solid #1a2236; transition:all .2s; }
.stButton button[kind="primary"]       { background:linear-gradient(135deg,#00e5b4,#00a88a); color:#000; border:none; }
.stButton button[kind="primary"]:hover { opacity:.9; transform:translateY(-1px); box-shadow:0 6px 20px rgba(0,229,180,.25); }
.stButton button:not([kind="primary"]):hover { border-color:#00e5b4; color:#00e5b4; }

/* ── Inputs ── */
.stSelectbox > div, .stNumberInput > div input, .stTextInput > div input { background:#0f1623 !important; border:1px solid #1a2236 !important; border-radius:8px !important; color:#c8d0e0 !important; font-family:'JetBrains Mono',monospace !important; font-size:13px !important; }

/* ── Sidebar section headers ── */
.shdr { font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#4a5568; margin:14px 0 6px; padding-bottom:5px; border-bottom:1px solid #1a2236; }

/* ── Cards ── */
.card { background:#0f1623; border:1px solid #1a2236; border-radius:12px; padding:14px 18px; margin-bottom:10px; }
.card-red   { background:rgba(255,77,109,.07); border-color:rgba(255,77,109,.3); }
.card-green { background:rgba(0,229,180,.06);  border-color:rgba(0,229,180,.25); }

/* ── Log box ── */
.logbox { background:#080b12; border:1px solid #1a2236; border-radius:8px; padding:12px; height:260px; overflow-y:auto; font-family:'JetBrains Mono',monospace; font-size:11px; line-height:1.7; }

/* ── Misc ── */
hr { border-color:#1a2236 !important; margin:10px 0 !important; }
::-webkit-scrollbar { width:4px; height:4px; } ::-webkit-scrollbar-track { background:#080b12; } ::-webkit-scrollbar-thumb { background:#1a2236; border-radius:2px; }
[data-testid="stExpander"] { background:#0f1623; border:1px solid #1a2236; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
IST = pytz.timezone("Asia/Kolkata")

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

# For each display period, fetch a larger warmup so EMA never starts as NaN
WARMUP_MAP = {
    "1d": "5d",   "5d": "1mo",  "7d": "1mo",
    "1mo": "3mo", "3mo": "6mo", "6mo": "1y",
    "1y": "2y",   "2y": "5y",   "5y": "max",
    "10y": "max", "20y": "max",
}

MAX_PERIOD_FOR_INTERVAL = {
    "1m": "7d", "5m": "60d", "15m": "60d",
    "1h": "730d", "1d": "max", "1wk": "max",
}

INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440, "1wk": 10080,
}

STRATEGIES   = ["EMA Crossover", "Simple Buy", "Simple Sell"]
SL_TYPES     = ["Custom Points", "Trailing SL", "Reverse EMA Crossover", "Risk Reward Based"]
TARGET_TYPES = ["Custom Points", "Trailing Target", "EMA Crossover", "Risk Reward Based"]

# ────────────────────────────────────────────────────────────────────────────
#  COMMENTED STRATEGY — Price Crosses Threshold
#  (intentionally excluded from STRATEGIES dropdown; use as reference/future plug-in)
#
#  Config needed:
#    threshold : float   — price level to watch
#    direction : "above" | "below"
#    action    : "buy"  | "sell"
#
#  Entry logic (evaluate on COMPLETED candle close):
#
#  def price_crosses_threshold(prev_close, curr_close, threshold, direction, action):
#      if direction == "above":
#          if prev_close < threshold and curr_close >= threshold:
#              return action       # "buy" or "sell" from dropdown
#      elif direction == "below":
#          if prev_close > threshold and curr_close <= threshold:
#              return action       # "buy" or "sell" from dropdown
#      return None
#
#  To activate:
#    1. Add "Price Crosses Threshold" to STRATEGIES list.
#    2. Add UI widgets in sidebar for threshold / direction / action dropdowns.
#    3. Call price_crosses_threshold() inside run_backtest() and live_engine().
# ────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS = {
    "live_running":         False,
    "live_position":        None,
    "trade_history":        [],
    "live_log":             [],
    "live_chart_df":        None,
    "live_ema_fast_val":    None,
    "live_ema_slow_val":    None,
    "live_ltp":             None,
    "last_trade_exit_time": None,
    "live_cfg":             None,
    "bt_results":           None,
    "bt_violations":        [],
    "bt_chart_df":          None,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════════════
# ──────────────────────────── DATA LAYER ────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  DHAN CANDLESTICK FETCH  (commented reference — plug in to replace yfinance)
#
#  from dhanhq import dhanhq
#
#  def fetch_candles_dhan(client_id, access_token, security_id, exchange_segment,
#                         instrument_type, interval, from_date, to_date):
#      """
#      Fetch OHLCV candles from Dhan broker API — zero delay, real-time data.
#
#      Parameters
#      ----------
#      security_id       : str   e.g. "1333" (Infosys), "13" (Nifty-50 index)
#      exchange_segment  : str   "NSE_EQ" | "BSE_EQ" | "NSE_FNO" | "BSE_FNO" | "IDX_I"
#      instrument_type   : str   "EQUITY" | "INDEX" | "FUTIDX" | "OPTIDX"
#      interval          : str   "1"(1m) | "5"(5m) | "15"(15m) | "25"(25m) | "60"(1h) | "D"(1d)
#      from_date         : str   "YYYY-MM-DD"
#      to_date           : str   "YYYY-MM-DD"
#
#      Returns
#      -------
#      pd.DataFrame with columns: Open, High, Low, Close, Volume, index=DatetimeIndex(IST)
#      """
#      dhan = dhanhq(client_id, access_token)
#
#      # Intraday candles (1m, 5m, 15m, 25m, 60m)
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
#          # Daily candles
#          resp = dhan.historical_daily_data(
#              security_id=security_id,
#              exchange_segment=exchange_segment,
#              instrument_type=instrument_type,
#              expiry_code=0,           # 0 = current/spot; use 1/2/3 for futures expiry
#              from_date=from_date,
#              to_date=to_date,
#          )
#
#      # resp["data"] is a list of {timestamp, open, high, low, close, volume}
#      records = resp.get("data", [])
#      if not records:
#          return pd.DataFrame()
#
#      df = pd.DataFrame(records)
#      df.rename(columns={"timestamp":"datetime","open":"Open","high":"High",
#                          "low":"Low","close":"Close","volume":"Volume"}, inplace=True)
#      df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
#      df.set_index("datetime", inplace=True)
#      df.index = df.index.tz_convert("Asia/Kolkata")
#      df = df[["Open","High","Low","Close","Volume"]].copy()
#      df.sort_index(inplace=True)
#      return df
#
#  HOW TO PLUG IN:
#  ───────────────
#  Replace the body of fetch_ohlcv() with:
#
#  def fetch_ohlcv(ticker, interval, period):
#      from_date = (datetime.now() - timedelta(days=period_days[period])).strftime("%Y-%m-%d")
#      to_date   = datetime.now().strftime("%Y-%m-%d")
#      df_full   = fetch_candles_dhan(CLIENT_ID, ACCESS_TOKEN,
#                                     security_id=DHAN_SEC_ID,
#                                     exchange_segment="NSE_EQ",
#                                     instrument_type="EQUITY",
#                                     interval=DHAN_INTERVAL_MAP[interval],
#                                     from_date=from_date, to_date=to_date)
#      df_display = df_full.copy()
#      return df_full, df_display
#
#  DHAN_INTERVAL_MAP = {"1m":"1","5m":"5","15m":"15","1h":"60","1d":"D"}
# ─────────────────────────────────────────────────────────────────────────────


def _clean(raw) -> pd.DataFrame | None:
    """Flatten MultiIndex, keep OHLCV, convert to IST."""
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
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
    return df if not df.empty else None


PERIOD_DAYS = {
    "1d": 1,   "5d": 5,   "7d": 7,   "1mo": 30,  "3mo": 90,
    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "20y": 7300,
}


@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str):
    """
    Fetch OHLCV with warmup window → EMA is never NaN in display window.
    Returns (df_full, df_display) or None on failure.
    """
    try:
        warmup = WARMUP_MAP.get(period, period)

        # Respect yfinance hard limits per interval
        max_p   = MAX_PERIOD_FOR_INTERVAL.get(interval, "max")
        warmup_days = PERIOD_DAYS.get(warmup, 9999)
        max_days    = PERIOD_DAYS.get(max_p, 9999)
        if max_p != "max" and warmup_days > max_days:
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
    except Exception as e:
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


def fetch_live_ltp(ticker: str) -> float | None:
    try:
        df = _clean(yf.download(ticker, period="1d", interval="1m",
                                auto_adjust=True, progress=False))
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except:
        pass
    return None

# ════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def tv_ema(series: pd.Series, period: int) -> pd.Series:
    """
    EMA matching TradingView exactly.
    TV uses alpha = 2/(n+1), seeds from bar 0 (no SMA seed), adjust=False.
    pandas ewm(span=n, adjust=False, min_periods=1) replicates this perfectly.
    """
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def add_indicators(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df = df.copy()
    ef = tv_ema(df["Close"], fast)
    es = tv_ema(df["Close"], slow)
    df[f"EMA_{fast}"] = ef
    df[f"EMA_{slow}"] = es
    pef, pes = ef.shift(1), es.shift(1)
    df["Signal"] = 0
    df.loc[(ef > es) & (pef <= pes), "Signal"] =  1   # bullish cross
    df.loc[(ef < es) & (pef >= pes), "Signal"] = -1   # bearish cross
    return df

# ════════════════════════════════════════════════════════════════════════════
# SL / TARGET CALCULATORS
# ════════════════════════════════════════════════════════════════════════════

def calc_sl(entry, tt, sl_type, sl_pts, ema_f, ema_s, rr):
    sign = 1 if tt == "buy" else -1
    if sl_type == "Custom Points":
        return entry - sign * sl_pts
    elif sl_type == "Trailing SL":
        return entry - sign * sl_pts         # initial; updated live each tick
    elif sl_type == "Reverse EMA Crossover":
        base = ema_s if tt == "buy" else ema_f
        fb   = entry - sign * sl_pts
        return min(base, fb) if tt == "buy" else max(base, fb)
    elif sl_type == "Risk Reward Based":
        return entry - sign * sl_pts
    return entry - sign * sl_pts


def calc_tgt(entry, tt, tgt_type, tgt_pts, sl, ema_f, ema_s, rr):
    sign = 1 if tt == "buy" else -1
    if tgt_type == "Custom Points":
        return entry + sign * tgt_pts
    elif tgt_type == "Trailing Target":
        return entry + sign * tgt_pts       # display only — never forces exit
    elif tgt_type == "EMA Crossover":
        return None                          # exit on crossover signal
    elif tgt_type == "Risk Reward Based":
        risk = abs(entry - sl) if sl is not None else tgt_pts
        return entry + sign * risk * rr
    return entry + sign * tgt_pts

# ════════════════════════════════════════════════════════════════════════════
# ──────────────────────────── BACKTEST ENGINE ────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(df_full, df_display, strategy, fast, slow,
                 sl_type, sl_pts, tgt_type, tgt_pts,
                 qty, cd_en, cd_s, no_overlap, rr):
    """
    Conservative SL-first rule for ALL trade types (buy AND sell):
      1. Check SL first  → if hit, exit at SL price
      2. Check Target    → if hit, exit at Target price
      3. If BOTH hit on same candle, SL wins (worst-case conservative assumption).

    This approach is the most conservative and closest to what live trading
    would produce (you never know which was hit first intra-candle).

    Backtest uses candle High/Low for comparison:
      BUY  : SL vs candle Low,  Target vs candle High
      SELL : SL vs candle High, Target vs candle Low
    """

    # ── 1. Compute indicators on full dataset ─────────────────────────────
    df = add_indicators(df_full.copy(), fast, slow)
    if strategy != "EMA Crossover":
        df["Signal"] = 0

    # ── 2. Find warmup boundary ───────────────────────────────────────────
    display_start = df_display.index[0] if not df_display.empty else df.index[0]
    warmup_rows   = max(slow * 3, df.index.searchsorted(display_start))
    warmup_rows   = min(warmup_rows, len(df) - 1)

    trades:     list[dict] = []
    violations: list[dict] = []
    active:     dict | None = None
    last_exit_ts = None

    # ── 3. Bar-by-bar loop ────────────────────────────────────────────────
    for i in range(warmup_rows, len(df)):
        row = df.iloc[i]
        ts  = df.index[i]
        hi  = float(row["High"])
        lo  = float(row["Low"])
        cl  = float(row["Close"])
        sig = int(row.get("Signal", 0))

        ef_val = float(row.get(f"EMA_{fast}", cl))
        es_val = float(row.get(f"EMA_{slow}", cl))

        # ── 3a. EXIT LOGIC ────────────────────────────────────────────────
        if active is not None:
            ep  = active["entry_price"]
            tt  = active["trade_type"]
            sl  = active["sl"]
            tgt = active["target"]

            # Evaluate hit conditions using candle prices
            if tt == "buy":
                sl_hit  = (sl  is not None and lo  <= sl)
                tgt_hit = (tgt is not None and hi  >= tgt)
            else:  # sell
                sl_hit  = (sl  is not None and hi  >= sl)
                tgt_hit = (tgt is not None and lo  <= tgt)

            # EMA crossover exit (from completed candle signal)
            ema_exit = False
            if tgt_type == "EMA Crossover":
                if tt == "buy"  and sig == -1: ema_exit = True
                if tt == "sell" and sig ==  1: ema_exit = True

            exit_px  = None
            exit_rsn = None
            viol     = False

            # ── CONSERVATIVE RULE (same for BUY and SELL):
            #    SL is checked FIRST — this is the worst-case assumption.
            #    If both SL and Target are breached on the same candle,
            #    we assume SL was hit first (most conservative / closest to live).
            if sl_hit:
                exit_px  = sl
                exit_rsn = "Stop Loss Hit"
                if tgt_hit:
                    # Both breached on same candle → flag as violation
                    viol     = True
                    exit_rsn = "SL Hit ⚠ (both SL & Target breached — SL taken first, conservative)"
            elif tgt_hit:
                exit_px  = tgt
                exit_rsn = "Target Hit"
            elif ema_exit:
                exit_px  = cl
                exit_rsn = "EMA Crossover Exit"

            # Trailing SL update (only when no exit this bar)
            if sl_type == "Trailing SL" and exit_px is None:
                if tt == "buy":
                    new_sl = cl - sl_pts
                    if new_sl > active["sl"]: active["sl"] = new_sl
                else:
                    new_sl = cl + sl_pts
                    if new_sl < active["sl"]: active["sl"] = new_sl

            if exit_px is not None:
                pnl = ((exit_px - ep) if tt == "buy" else (ep - exit_px)) * qty
                rec = {
                    "Entry Time":   active["entry_time"],
                    "Exit Time":    ts,
                    "Trade Type":   tt.upper(),
                    "Entry Price":  round(ep, 2),
                    "Exit Price":   round(exit_px, 2),
                    "SL":           round(active["sl"], 2) if active["sl"]  is not None else "—",
                    "Target":       round(tgt, 2)          if tgt           is not None else "—",
                    "Candle High":  round(hi, 2),
                    "Candle Low":   round(lo, 2),
                    "Entry Reason": active["entry_reason"],
                    "Exit Reason":  exit_rsn,
                    "PnL (₹)":      round(pnl, 2),
                    "Qty":          qty,
                    "Violation":    "⚠ Yes" if viol else "✓",
                }
                trades.append(rec)
                if viol: violations.append(rec)
                last_exit_ts = ts
                active        = None

        # ── 3b. ENTRY LOGIC ───────────────────────────────────────────────
        if active is None:
            # Cooldown guard
            if cd_en and last_exit_ts is not None:
                if (ts - last_exit_ts).total_seconds() < cd_s:
                    continue

            trade_type = None
            entry_rsn  = ""

            if strategy == "EMA Crossover":
                if   sig ==  1: trade_type = "buy";  entry_rsn = f"EMA{fast} crossed above EMA{slow}"
                elif sig == -1: trade_type = "sell"; entry_rsn = f"EMA{fast} crossed below EMA{slow}"
            elif strategy == "Simple Buy":
                if i == warmup_rows: trade_type = "buy";  entry_rsn = "Simple Buy (market entry)"
            elif strategy == "Simple Sell":
                if i == warmup_rows: trade_type = "sell"; entry_rsn = "Simple Sell (market entry)"

            if trade_type is None:
                continue

            sl_p  = calc_sl(cl, trade_type, sl_type, sl_pts, ef_val, es_val, rr)
            tgt_p = calc_tgt(cl, trade_type, tgt_type, tgt_pts, sl_p, ef_val, es_val, rr)

            active = {
                "entry_time":   ts,
                "entry_price":  cl,
                "trade_type":   trade_type,
                "sl":           sl_p,
                "target":       tgt_p,
                "entry_reason": entry_rsn,
            }

    # ── 4. Force-close open trade at end of data ─────────────────────────
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
            "SL":           round(active["sl"], 2)     if active["sl"]     is not None else "—",
            "Target":       round(active["target"], 2) if active["target"] is not None else "—",
            "Candle High":  round(float(lr["High"]), 2),
            "Candle Low":   round(float(lr["Low"]), 2),
            "Entry Reason": active["entry_reason"],
            "Exit Reason":  "End of Data (open position closed)",
            "PnL (₹)":      round(pnl, 2),
            "Qty":          qty,
            "Violation":    "✓",
        })

    return pd.DataFrame(trades), violations

# ════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ════════════════════════════════════════════════════════════════════════════
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#080b12",
    plot_bgcolor="#080b12",
    font=dict(family="JetBrains Mono", color="#6b7a99", size=11),
    xaxis_rangeslider_visible=False,
    margin=dict(t=40, b=16, l=48, r=24),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0f1623", bordercolor="#1a2236",
                    font=dict(family="JetBrains Mono", size=11)),
)


def build_chart(df, fast, slow, trades_df=None, position=None, title=""):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02)

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing=dict(line=dict(color="#00e5b4", width=1.2), fillcolor="#00e5b4"),
        decreasing=dict(line=dict(color="#ff4d6d", width=1.2), fillcolor="#ff4d6d"),
        whiskerwidth=0.4,
    ), row=1, col=1)

    # EMAs
    if f"EMA_{fast}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{fast}"],
                                 name=f"EMA {fast}",
                                 line=dict(color="#f59e0b", width=1.8)), row=1, col=1)
    if f"EMA_{slow}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{slow}"],
                                 name=f"EMA {slow}",
                                 line=dict(color="#3b82f6", width=1.8)), row=1, col=1)

    # Trade markers (backtest)
    if trades_df is not None and not trades_df.empty:
        buy_entries  = trades_df[trades_df["Trade Type"] == "BUY"]
        sell_entries = trades_df[trades_df["Trade Type"] == "SELL"]

        for _, t in buy_entries.iterrows():
            fig.add_trace(go.Scatter(
                x=[t["Entry Time"]], y=[t["Entry Price"]],
                mode="markers", name="Buy Entry", showlegend=False,
                marker=dict(symbol="triangle-up", size=13, color="#00e5b4",
                            line=dict(width=1.5, color="#fff")),
                hovertemplate=f"<b>BUY ENTRY</b><br>Price: {t['Entry Price']}<br>{t['Entry Reason']}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t["Exit Time"]], y=[t["Exit Price"]],
                mode="markers", name="Buy Exit", showlegend=False,
                marker=dict(symbol="triangle-down", size=11, color="#00e5b4", opacity=0.55,
                            line=dict(width=1, color="#080b12")),
                hovertemplate=f"<b>BUY EXIT</b><br>Price: {t['Exit Price']}<br>{t['Exit Reason']}<br>PnL: ₹{t['PnL (₹)']:+.2f}<extra></extra>",
            ), row=1, col=1)

        for _, t in sell_entries.iterrows():
            fig.add_trace(go.Scatter(
                x=[t["Entry Time"]], y=[t["Entry Price"]],
                mode="markers", name="Sell Entry", showlegend=False,
                marker=dict(symbol="triangle-down", size=13, color="#ff4d6d",
                            line=dict(width=1.5, color="#fff")),
                hovertemplate=f"<b>SELL ENTRY</b><br>Price: {t['Entry Price']}<br>{t['Entry Reason']}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[t["Exit Time"]], y=[t["Exit Price"]],
                mode="markers", name="Sell Exit", showlegend=False,
                marker=dict(symbol="triangle-up", size=11, color="#ff4d6d", opacity=0.55,
                            line=dict(width=1, color="#080b12")),
                hovertemplate=f"<b>SELL EXIT</b><br>Price: {t['Exit Price']}<br>{t['Exit Reason']}<br>PnL: ₹{t['PnL (₹)']:+.2f}<extra></extra>",
            ), row=1, col=1)

    # Live position lines
    if position is not None:
        for level, color, label in [
            (position.get("entry_price"), "#ffffff", "Entry"),
            (position.get("sl"),          "#ff4d6d", "SL"),
            (position.get("target"),      "#00e5b4", "Target"),
        ]:
            if level is not None:
                fig.add_hline(y=level,
                              line=dict(color=color, width=1.4, dash="dash" if label != "Entry" else "solid"),
                              annotation_text=f" {label} {level:.2f}",
                              annotation_font=dict(color=color, size=11), row=1, col=1)

    # Volume
    if "Volume" in df.columns:
        vol_clr = ["#00e5b4" if c >= o else "#ff4d6d"
                   for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                             marker_color=vol_clr, name="Vol", showlegend=False), row=2, col=1)

    fig.update_layout(height=560, title=dict(text=title, x=0.01,
                      font=dict(family="Syne", size=13, color="#6b7a99")), **_PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#0f1623", zeroline=False,
                     showspikes=True, spikecolor="#2a3456", spikethickness=1)
    fig.update_yaxes(gridcolor="#0f1623", zeroline=False,
                     showspikes=True, spikecolor="#2a3456", spikethickness=1)
    return fig

# ════════════════════════════════════════════════════════════════════════════
# LTP WIDGET
# ════════════════════════════════════════════════════════════════════════════

def ltp_widget(ticker, label):
    info = fetch_ltp_cached(ticker)
    if info:
        p, c, pct = info["price"], info["change"], info["pct"]
        col = "#00e5b4" if c >= 0 else "#ff4d6d"
        arr = "▲" if c >= 0 else "▼"
        st.markdown(f"""
        <div class="card" style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;margin-bottom:14px">
            <div>
                <div style="font-family:Syne,sans-serif;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4a5568">{label}</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:#e2e8f0;line-height:1.1">{p:,.2f}</div>
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:17px;font-weight:600;color:{col}">{arr} {c:+.2f}</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:600;color:{col}">{pct:+.2f}%</div>
            <div style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:11px;color:#4a5568">Prev Close: {info['prev']:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"⚡ LTP unavailable for {label}. Check ticker or network.")

# ════════════════════════════════════════════════════════════════════════════
# DHAN ORDER PLACEMENT
# ════════════════════════════════════════════════════════════════════════════

def _dhan(cfg):
    return dhanhq(cfg["dhan_client_id"], cfg["dhan_access_token"])


def place_entry_order(cfg, tt, ltp):
    if not DHAN_OK: return "⚠ dhanhq not installed"
    try:
        dhan = _dhan(cfg)
        if cfg.get("options_trading"):
            sec   = cfg["ce_security_id"] if tt == "buy" else cfg["pe_security_id"]
            ot    = "MARKET" if cfg.get("opts_entry_otype") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0
            r = dhan.place_order(transactionType="BUY",
                                 exchangeSegment=cfg.get("opts_exchange", "NSE_FNO"),
                                 productType="INTRADAY", orderType=ot,
                                 validity="DAY", securityId=str(sec),
                                 quantity=int(cfg.get("opts_qty", 65)),
                                 price=price, triggerPrice=0)
        else:
            ex    = "NSE_EQ" if cfg.get("eq_exchange", "NSE") == "NSE" else "BSE"
            pt    = "INTRADAY" if cfg.get("eq_product") == "Intraday" else "CNC"
            ot    = "MARKET" if cfg.get("eq_entry_otype") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0
            r = dhan.place_order(security_id=str(cfg.get("eq_sec_id", "1594")),
                                 exchange_segment=ex,
                                 transaction_type="BUY" if tt == "buy" else "SELL",
                                 quantity=int(cfg.get("eq_qty", 1)),
                                 order_type=ot, product_type=pt, price=price)
        return f"✅ Entry placed: {r}"
    except Exception as e:
        return f"❌ Entry failed: {e}"


def place_exit_order(cfg, tt, ltp):
    if not DHAN_OK: return "⚠ dhanhq not installed"
    try:
        dhan = _dhan(cfg)
        if cfg.get("options_trading"):
            sec   = cfg["ce_security_id"] if tt == "buy" else cfg["pe_security_id"]
            ot    = "MARKET" if cfg.get("opts_exit_otype", "Market Order") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0
            r = dhan.place_order(transactionType="SELL",
                                 exchangeSegment=cfg.get("opts_exchange", "NSE_FNO"),
                                 productType="INTRADAY", orderType=ot,
                                 validity="DAY", securityId=str(sec),
                                 quantity=int(cfg.get("opts_qty", 65)),
                                 price=price, triggerPrice=0)
        else:
            ex    = "NSE_EQ" if cfg.get("eq_exchange", "NSE") == "NSE" else "BSE"
            pt    = "INTRADAY" if cfg.get("eq_product") == "Intraday" else "CNC"
            ot    = "MARKET" if cfg.get("eq_exit_otype", "Market Order") == "Market Order" else "LIMIT"
            price = float(ltp) if ot == "LIMIT" else 0
            r = dhan.place_order(security_id=str(cfg.get("eq_sec_id", "1594")),
                                 exchange_segment=ex,
                                 transaction_type="SELL" if tt == "buy" else "BUY",
                                 quantity=int(cfg.get("eq_qty", 1)),
                                 order_type=ot, product_type=pt, price=price)
        return f"✅ Exit placed: {r}"
    except Exception as e:
        return f"❌ Exit failed: {e}"

# ════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE (background thread)
# ════════════════════════════════════════════════════════════════════════════

def live_engine(cfg):
    """
    Background thread loop:
    - Polls yfinance every 1.5 s (avoids API rate-limit bans)
    - SL / Target compared with LTP each tick (not candle H/L)
    - EMA cross-signal evaluated only on COMPLETED candles
    - New candle detected by comparing last closed bar timestamp
    """
    ticker   = cfg["ticker"];    interval = cfg["interval"];  period = cfg["period"]
    strategy = cfg["strategy"];  fast     = cfg["fast_ema"]; slow   = cfg["slow_ema"]
    sl_type  = cfg["sl_type"];   sl_pts   = cfg["sl_pts"]
    tgt_type = cfg["tgt_type"];  tgt_pts  = cfg["tgt_pts"]
    qty      = cfg["qty"];       cd_en    = cfg["cd_en"];    cd_s   = cfg["cd_s"]
    rr       = cfg["rr"]
    int_mins = INTERVAL_MINUTES.get(interval, 5)

    def _log(msg):
        ts    = datetime.now(IST).strftime("%H:%M:%S")
        entry = f"<span style='color:#4a5568'>[{ts}]</span> {msg}"
        logs  = st.session_state.get("live_log", [])
        logs.append(entry)
        st.session_state["live_log"] = logs[-200:]

    _log("🚀 <b>Engine started</b>")
    last_closed_candle_ts = None

    while st.session_state.get("live_running", False):
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=True, progress=False, prepost=False)
            df  = _clean(raw)

            if df is None or len(df) < max(slow * 2, 5):
                _log("⚠ Insufficient data — waiting…"); time.sleep(1.5); continue

            df[f"EMA_{fast}"] = tv_ema(df["Close"], fast)
            df[f"EMA_{slow}"] = tv_ema(df["Close"], slow)

            st.session_state["live_chart_df"]     = df.copy()
            st.session_state["live_ema_fast_val"] = float(df[f"EMA_{fast}"].iloc[-2])
            st.session_state["live_ema_slow_val"] = float(df[f"EMA_{slow}"].iloc[-2])

            ltp = float(df["Close"].iloc[-1])
            st.session_state["live_ltp"] = ltp

            # ── EXIT: compare SL/Target against LTP (tick-level) ─────────
            pos = st.session_state.get("live_position")
            if pos is not None:
                tt  = pos["trade_type"]
                ep  = pos["entry_price"]
                sl  = pos["sl"]
                tgt = pos["target"]

                exited      = False
                exit_px     = None
                exit_reason = None

                # Conservative SL-first on live tick too
                if tt == "buy":
                    if sl  is not None and ltp <= sl:
                        exit_px = ltp; exit_reason = "Stop Loss Hit (LTP ≤ SL)"; exited = True
                    elif tgt is not None and ltp >= tgt:
                        exit_px = ltp; exit_reason = "Target Hit (LTP ≥ Target)"; exited = True
                else:  # sell
                    if sl  is not None and ltp >= sl:
                        exit_px = ltp; exit_reason = "Stop Loss Hit (LTP ≥ SL)"; exited = True
                    elif tgt is not None and ltp <= tgt:
                        exit_px = ltp; exit_reason = "Target Hit (LTP ≤ Target)"; exited = True

                # EMA crossover exit (completed candle only)
                if not exited and tgt_type == "EMA Crossover" and len(df) >= 3:
                    cf = float(df[f"EMA_{fast}"].iloc[-2]); cs = float(df[f"EMA_{slow}"].iloc[-2])
                    pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])
                    if tt == "buy"  and cf < cs and pf >= ps: exit_px = ltp; exit_reason = "EMA Crossover Exit"; exited = True
                    if tt == "sell" and cf > cs and pf <= ps: exit_px = ltp; exit_reason = "EMA Crossover Exit"; exited = True

                # Trailing SL update
                if sl_type == "Trailing SL" and not exited:
                    if tt == "buy":
                        nsl = ltp - sl_pts
                        if nsl > pos["sl"]: pos["sl"] = nsl; _log(f"📊 Trailing SL → <b>{nsl:.2f}</b>")
                    else:
                        nsl = ltp + sl_pts
                        if nsl < pos["sl"]: pos["sl"] = nsl; _log(f"📊 Trailing SL → <b>{nsl:.2f}</b>")
                    st.session_state["live_position"] = pos

                if exited:
                    pnl = ((exit_px - ep) if tt == "buy" else (ep - exit_px)) * qty
                    rec = {
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   tt.upper(),
                        "Entry Price":  round(ep, 2),
                        "Exit Price":   round(exit_px, 2),
                        "SL":           round(sl, 2)  if sl  is not None else "—",
                        "Target":       round(tgt, 2) if tgt is not None else "—",
                        "Entry Reason": pos.get("entry_reason", ""),
                        "Exit Reason":  exit_reason,
                        "PnL (₹)":      round(pnl, 2),
                        "Qty":          qty,
                        "Mode":         "Live",
                        "Violation":    "✓",
                    }
                    st.session_state["trade_history"].append(rec)
                    st.session_state["live_position"]        = None
                    st.session_state["last_trade_exit_time"] = datetime.now(IST)
                    if cfg.get("dhan_en"):
                        _log(f"📤 {place_exit_order(cfg, tt, exit_px)}")
                    c2 = "color:#00e5b4" if pnl >= 0 else "color:#ff4d6d"
                    _log(f"✅ <b>EXIT {tt.upper()}</b> @ {exit_px:.2f} | {exit_reason} | <span style='{c2}'>PnL ₹{pnl:+.2f}</span>")

            # ── ENTRY: only on completed candle boundary ──────────────────
            current_closed_ts = df.index[-2] if len(df) >= 2 else None
            is_new_candle     = (current_closed_ts is not None and
                                 current_closed_ts != last_closed_candle_ts)

            # Also allow trigger on time boundary (multiple of interval)
            now_ist      = datetime.now(IST)
            cur_total_m  = now_ist.hour * 60 + now_ist.minute
            is_boundary  = (cur_total_m % int_mins == 0 and now_ist.second < 5)

            if (is_new_candle or is_boundary) and st.session_state.get("live_position") is None:
                last_closed_candle_ts = current_closed_ts

                # Cooldown
                if cd_en:
                    last_exit = st.session_state.get("last_trade_exit_time")
                    if last_exit and (now_ist - last_exit).total_seconds() < cd_s:
                        remain = cd_s - (now_ist - last_exit).total_seconds()
                        _log(f"⏳ Cooldown: <b>{remain:.0f}s</b> remaining"); time.sleep(1.5); continue

                if len(df) < 3: time.sleep(1.5); continue

                cf = float(df[f"EMA_{fast}"].iloc[-2]); cs = float(df[f"EMA_{slow}"].iloc[-2])
                pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])

                signal    = None
                entry_rsn = ""
                if strategy == "EMA Crossover":
                    if   cf > cs and pf <= ps: signal = "buy";  entry_rsn = f"EMA{fast} crossed above EMA{slow}"
                    elif cf < cs and pf >= ps: signal = "sell"; entry_rsn = f"EMA{fast} crossed below EMA{slow}"
                elif strategy == "Simple Buy":
                    signal = "buy";  entry_rsn = "Simple Buy"
                elif strategy == "Simple Sell":
                    signal = "sell"; entry_rsn = "Simple Sell"

                if signal:
                    sl_p  = calc_sl(ltp, signal, sl_type, sl_pts, cf, cs, rr)
                    tgt_p = calc_tgt(ltp, signal, tgt_type, tgt_pts, sl_p, cf, cs, rr)
                    st.session_state["live_position"] = {
                        "trade_type":   signal,
                        "entry_price":  ltp,
                        "entry_time":   now_ist.strftime("%Y-%m-%d %H:%M:%S"),
                        "sl":           sl_p,
                        "target":       tgt_p,
                        "entry_reason": entry_rsn,
                    }
                    if cfg.get("dhan_en"):
                        _log(f"📤 {place_entry_order(cfg, signal, ltp)}")
                    cs2 = "color:#00e5b4" if signal == "buy" else "color:#ff4d6d"
                    _log(f"🎯 <b><span style='{cs2}'>{signal.upper()}</span></b> @ {ltp:.2f} "
                         f"| SL: {sl_p:.2f if sl_p else 'None'} "
                         f"| Target: {tgt_p:.2f if tgt_p else 'None'} | {entry_rsn}")

        except Exception as exc:
            _log(f"❌ Error: <span style='color:#ff4d6d'>{exc}</span>")

        time.sleep(1.5)   # ← rate-limit guard: never remove this

    _log("⏹ <b>Engine stopped</b>")

# ════════════════════════════════════════════════════════════════════════════
# ████████████████████████   MAIN APP   ██████████████████████████████████████
# ════════════════════════════════════════════════════════════════════════════

def main():

    # ════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:18px 0 12px">
            <div style="font-size:40px">📈</div>
            <div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;letter-spacing:3px;
                        background:linear-gradient(90deg,#00e5b4,#3b82f6);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                SMART INVESTING
            </div>
            <div style="font-size:9px;color:#4a5568;letter-spacing:2px;margin-top:3px">
                ALGORITHMIC TRADING PLATFORM
            </div>
        </div>""", unsafe_allow_html=True)
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

        # ── TIMEFRAME ────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Timeframe</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            interval = st.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()),
                                    index=2, label_visibility="collapsed")
        with c2:
            prd_opts = TIMEFRAME_PERIODS[interval]
            period   = st.selectbox("Period", prd_opts,
                                    index=min(1, len(prd_opts) - 1),
                                    label_visibility="collapsed")

        # ── STRATEGY ─────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Strategy</div>', unsafe_allow_html=True)
        strategy = st.selectbox("Strategy", STRATEGIES, label_visibility="collapsed")
        fast_ema, slow_ema = 9, 15
        if strategy == "EMA Crossover":
            e1, e2 = st.columns(2)
            with e1: fast_ema = st.number_input("Fast EMA", 2, 200, 9)
            with e2: slow_ema = st.number_input("Slow EMA", 2, 500, 15)

        # ── STOP LOSS ────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Stop Loss</div>', unsafe_allow_html=True)
        sl_type = st.selectbox("SL Type", SL_TYPES, label_visibility="collapsed")
        s1, s2  = st.columns(2)
        with s1: sl_pts = st.number_input("SL Points", 0.1, 1e6, 10.0, step=0.5)
        with s2: rr     = st.number_input("R:R Ratio", 0.5, 20.0, 2.0, step=0.5,
                                          help="Risk:Reward — used when SL/Target = 'Risk Reward Based'")

        # ── TARGET ───────────────────────────────────────────────────────
        st.markdown('<div class="shdr">Target</div>', unsafe_allow_html=True)
        tgt_type = st.selectbox("Target Type", TARGET_TYPES, label_visibility="collapsed")
        tgt_pts  = st.number_input("Target Points", 0.1, 1e6, 20.0, step=0.5)

        # ── TRADE SETTINGS ───────────────────────────────────────────────
        st.markdown('<div class="shdr">Trade Settings</div>', unsafe_allow_html=True)
        qty = st.number_input("Quantity", 1, 10_000_000, 1)
        cd_col1, cd_col2 = st.columns([1.4, 1])
        with cd_col1: cd_en = st.checkbox("Cooldown Period", value=True)
        with cd_col2: cd_s  = st.number_input("Secs", 1, 86400, 5,
                                               disabled=not cd_en, label_visibility="visible")
        no_overlap = st.checkbox("No Overlapping Trades", value=True,
                                 help="Only one trade open at a time; new signals are ignored until current position is closed.")

        # ── DHAN BROKER ──────────────────────────────────────────────────
        st.markdown('<div class="shdr">Dhan Broker</div>', unsafe_allow_html=True)
        dhan_en = st.checkbox("Enable Dhan Broker", value=False)
        dhan_cfg: dict = {"dhan_en": dhan_en}

        if dhan_en:
            if not DHAN_OK:
                st.warning("Install dhanhq: `pip install dhanhq`")
            dhan_cfg["dhan_client_id"]    = st.text_input("Client ID",    "", type="password")
            dhan_cfg["dhan_access_token"] = st.text_input("Access Token", "", type="password")
            opts_en = st.checkbox("Options Trading", value=False,
                                  help="BUY CE on bullish signal · BUY PE on bearish signal")
            dhan_cfg["options_trading"] = opts_en

            if not opts_en:
                dhan_cfg["eq_product"]      = st.selectbox("Product",       ["Intraday", "Delivery"])
                dhan_cfg["eq_exchange"]     = st.selectbox("Exchange",      ["NSE", "BSE"])
                dhan_cfg["eq_sec_id"]       = st.text_input("Security ID",  "1594")
                dhan_cfg["eq_qty"]          = st.number_input("Broker Qty", 1, 1_000_000, 1)
                dhan_cfg["eq_entry_otype"]  = st.selectbox("Entry Order",   ["Limit Order", "Market Order"])
                dhan_cfg["eq_exit_otype"]   = st.selectbox("Exit Order",    ["Market Order", "Limit Order"])
            else:
                dhan_cfg["opts_exchange"]      = st.selectbox("FnO Exchange", ["NSE_FNO", "BSE_FNO"])
                dhan_cfg["ce_security_id"]     = st.text_input("CE Security ID", "")
                dhan_cfg["pe_security_id"]     = st.text_input("PE Security ID", "")
                dhan_cfg["opts_qty"]           = st.number_input("Lots/Qty", 1, 1_000_000, 65)
                dhan_cfg["opts_entry_otype"]   = st.selectbox("Entry Order",["Market Order", "Limit Order"])
                dhan_cfg["opts_exit_otype"]    = st.selectbox("Exit Order", ["Market Order", "Limit Order"])

        st.divider()
        st.caption("Smart Investing v1.0  •  Educational use only  •  Trade at own risk")

    # Full config bundle
    cfg: dict = {
        "ticker": ticker_sym, "ticker_label": ticker_label,
        "interval": interval, "period": period,
        "strategy": strategy, "fast_ema": fast_ema, "slow_ema": slow_ema,
        "sl_type": sl_type, "sl_pts": sl_pts,
        "tgt_type": tgt_type, "tgt_pts": tgt_pts,
        "qty": qty, "cd_en": cd_en, "cd_s": cd_s, "no_overlap": no_overlap, "rr": rr,
        **dhan_cfg,
    }

    # ════════════════════════════════════════════════════════════════════
    # PAGE HEADER
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="padding:4px 0 10px">
        <span style="font-family:Syne,sans-serif;font-size:32px;font-weight:800;
                     background:linear-gradient(90deg,#00e5b4 30%,#3b82f6 80%);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            📈 Smart Investing
        </span>
        <span style="font-family:Syne,sans-serif;font-size:12px;color:#4a5568;
                     margin-left:14px;vertical-align:middle">
            Algorithmic Trading Platform
        </span>
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════════════════
    tab_bt, tab_live, tab_hist = st.tabs([
        "🔬  Backtesting",
        "⚡  Live Trading",
        "📋  Trade History",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — BACKTESTING
    # ════════════════════════════════════════════════════════════════════
    with tab_bt:
        ltp_widget(ticker_sym, ticker_label)

        st.markdown("### 🔬 Backtest Engine")
        st.markdown(
            f"<small style='color:#4a5568'>Strategy: <b style='color:#c8d0e0'>{strategy}</b> &nbsp;·&nbsp; "
            f"Ticker: <b style='color:#c8d0e0'>{ticker_label}</b> &nbsp;·&nbsp; "
            f"Interval: <b style='color:#c8d0e0'>{interval}</b> &nbsp;·&nbsp; "
            f"Period: <b style='color:#c8d0e0'>{period}</b> &nbsp;·&nbsp; "
            f"SL: <b style='color:#ff4d6d'>{sl_type} ({sl_pts} pts)</b> &nbsp;·&nbsp; "
            f"Target: <b style='color:#00e5b4'>{tgt_type} ({tgt_pts} pts)</b></small>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Conservative SL-first explanation box
        st.markdown("""
        <div style="background:rgba(0,229,180,0.05);border-left:3px solid #00e5b4;
                    padding:10px 14px;border-radius:6px;margin-bottom:12px;font-size:12px;
                    font-family:Syne,sans-serif;color:#7a9988">
            <b style="color:#00e5b4">Conservative Exit Rule (Both BUY & SELL)</b><br>
            SL is always checked <b>first</b> on every candle for all trade types.
            If both SL and Target are breached on the same candle, SL is assumed hit first
            (worst-case / most conservative assumption, closest to live trading reality).
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
                st.error("❌ Data fetch failed. Check ticker symbol or network.")
            else:
                df_full, df_display = res
                with st.spinner("Running backtest…"):
                    tdf, viols = run_backtest(
                        df_full, df_display, strategy, fast_ema, slow_ema,
                        sl_type, sl_pts, tgt_type, tgt_pts,
                        qty, cd_en, cd_s, no_overlap, rr,
                    )
                # Store
                df_chart = add_indicators(df_full.copy(), fast_ema, slow_ema)
                if not df_display.empty:
                    df_chart = df_chart.loc[df_display.index[0]:]
                st.session_state["bt_results"]    = tdf
                st.session_state["bt_violations"] = viols
                st.session_state["bt_chart_df"]   = df_chart

        # ── Display results ───────────────────────────────────────────────
        tdf   = st.session_state.get("bt_results")
        viols = st.session_state.get("bt_violations", [])
        dfc   = st.session_state.get("bt_chart_df")

        if tdf is not None:
            if tdf.empty:
                st.warning("No trades generated. Try adjusting strategy parameters.")
            else:
                # ── Summary metrics ───────────────────────────────────────
                n       = len(tdf)
                wins    = int((tdf["PnL (₹)"] > 0).sum())
                losses  = n - wins
                tot_pnl = float(tdf["PnL (₹)"].sum())
                acc     = wins / n * 100 if n else 0
                avg_w   = float(tdf.loc[tdf["PnL (₹)"] > 0, "PnL (₹)"].mean()) if wins   else 0.0
                avg_l   = float(tdf.loc[tdf["PnL (₹)"] < 0, "PnL (₹)"].mean()) if losses else 0.0
                best    = float(tdf["PnL (₹)"].max())
                worst   = float(tdf["PnL (₹)"].min())

                cols = st.columns(8)
                metrics = [
                    ("Trades",      n,               None),
                    ("Wins ✓",      wins,            None),
                    ("Losses ✗",    losses,          None),
                    ("Accuracy",    f"{acc:.1f}%",   None),
                    ("Total PnL",   f"₹{tot_pnl:+,.2f}", "normal"),
                    ("Avg Win",     f"₹{avg_w:+.2f}", "normal"),
                    ("Best Trade",  f"₹{best:+.2f}",  "normal"),
                    ("Violations",  len(viols),      None),
                ]
                for col, (label, val, dt) in zip(cols, metrics):
                    col.metric(label, val)

                st.divider()

                # ── Chart ─────────────────────────────────────────────────
                if dfc is not None:
                    st.plotly_chart(
                        build_chart(dfc, fast_ema, slow_ema, trades_df=tdf,
                                    title=f"{ticker_label}  ·  {strategy}  ·  {interval}/{period}"),
                        use_container_width=True,
                    )

                # ── Trade table ───────────────────────────────────────────
                st.markdown("#### 📊 Trade Log")

                disp = tdf.copy()
                disp["Entry Time"] = disp["Entry Time"].astype(str)
                disp["Exit Time"]  = disp["Exit Time"].astype(str)

                def _style(row):
                    bg = "#0d2a1a" if row["PnL (₹)"] > 0 else ("#2a0d14" if row["PnL (₹)"] < 0 else "")
                    return [f"background-color:{bg}" for _ in row]

                styled = (disp.style
                          .apply(_style, axis=1)
                          .format({"PnL (₹)": "₹{:+.2f}", "Entry Price": "{:.2f}",
                                   "Exit Price": "{:.2f}"}))
                st.dataframe(styled, use_container_width=True, height=420,
                             column_config={
                                 "Trade Type": st.column_config.TextColumn(width=70),
                                 "Violation":  st.column_config.TextColumn(width=60),
                             })

                # ── Violations ────────────────────────────────────────────
                if viols:
                    st.markdown(f"""
                    <div style="background:rgba(255,77,109,.08);border-left:4px solid #ff4d6d;
                                padding:12px 16px;border-radius:8px;margin:10px 0;
                                font-family:Syne,sans-serif;font-size:12px">
                        ⚠ <b>{len(viols)} candle(s)</b> where both SL and Target were breached in the
                        same bar. SL exit was applied for all (conservative). These trades are the
                        most likely to differ from live results.
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
            st_autorefresh(interval=3000, key="live_ar")

        ltp_widget(ticker_sym, ticker_label)
        st.markdown("### ⚡ Live Trading")

        # ── Status badge ─────────────────────────────────────────────────
        running = st.session_state["live_running"]
        if running:
            st.markdown("""
            <div style="display:inline-flex;align-items:center;gap:8px;
                        background:rgba(0,229,180,.1);border:1px solid rgba(0,229,180,.3);
                        border-radius:20px;padding:6px 16px;margin-bottom:12px">
                <span style="width:8px;height:8px;border-radius:50%;background:#00e5b4;
                             display:inline-block;animation:pulse 1s infinite"></span>
                <span style="font-family:Syne,sans-serif;font-size:12px;font-weight:700;
                             color:#00e5b4;letter-spacing:1px">LIVE RUNNING</span>
            </div>
            <style>@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.3)}}</style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:inline-flex;align-items:center;gap:8px;
                        background:#0f1623;border:1px solid #1a2236;
                        border-radius:20px;padding:6px 16px;margin-bottom:12px">
                <span style="width:8px;height:8px;border-radius:50%;background:#4a5568;display:inline-block"></span>
                <span style="font-family:Syne,sans-serif;font-size:12px;font-weight:700;color:#4a5568;letter-spacing:1px">STOPPED</span>
            </div>""", unsafe_allow_html=True)

        # ── Control buttons ───────────────────────────────────────────────
        b1, b2, b3, _ = st.columns([1.4, 1.4, 1.4, 3])
        with b1:
            if st.button("▶  Start", type="primary", use_container_width=True,
                         disabled=running, key="live_start"):
                st.session_state["live_running"]  = True
                st.session_state["live_log"]      = []
                st.session_state["live_cfg"]      = cfg.copy()
                t = threading.Thread(target=live_engine, args=(cfg,), daemon=True)
                t.start()
                st.rerun()

        with b2:
            if st.button("⏹  Stop", use_container_width=True,
                         disabled=not running, key="live_stop"):
                st.session_state["live_running"] = False
                time.sleep(0.3)
                st.rerun()

        with b3:
            if st.button("⚡ Square Off", use_container_width=True,
                         disabled=not running, key="live_sq"):
                pos = st.session_state.get("live_position")
                if pos:
                    ltp_now = st.session_state.get("live_ltp") or fetch_live_ltp(ticker_sym) or pos["entry_price"]
                    pnl = ((ltp_now - pos["entry_price"]) if pos["trade_type"] == "buy"
                           else (pos["entry_price"] - ltp_now)) * qty
                    rec = {
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   pos["trade_type"].upper(),
                        "Entry Price":  round(pos["entry_price"], 2),
                        "Exit Price":   round(ltp_now, 2),
                        "SL":           round(pos["sl"], 2)     if pos["sl"]     is not None else "—",
                        "Target":       round(pos["target"], 2) if pos["target"] is not None else "—",
                        "Entry Reason": pos.get("entry_reason", ""),
                        "Exit Reason":  "Manual Square Off",
                        "PnL (₹)":      round(pnl, 2),
                        "Qty":          qty,
                        "Mode":         "Live",
                        "Violation":    "✓",
                    }
                    st.session_state["trade_history"].append(rec)
                    st.session_state["live_position"] = None
                    if cfg.get("dhan_en"):
                        place_exit_order(cfg, pos["trade_type"], ltp_now)
                    st.success(f"✅ Squared off at {ltp_now:.2f}  |  PnL ₹{pnl:+.2f}")
                else:
                    st.info("No open position to square off.")
                st.rerun()

        st.divider()

        # ── Config summary ────────────────────────────────────────────────
        live_cfg_shown = st.session_state.get("live_cfg") or cfg
        with st.expander("⚙  Active Configuration", expanded=running):
            cc = st.columns(4)
            cc[0].metric("Ticker",     live_cfg_shown.get("ticker_label", "—"))
            cc[1].metric("Interval",   live_cfg_shown.get("interval", "—"))
            cc[2].metric("Period",     live_cfg_shown.get("period", "—"))
            cc[3].metric("Strategy",   live_cfg_shown.get("strategy", "—"))
            cc2 = st.columns(4)
            cc2[0].metric("Fast EMA",  live_cfg_shown.get("fast_ema", "—"))
            cc2[1].metric("Slow EMA",  live_cfg_shown.get("slow_ema", "—"))
            cc2[2].metric("SL",        f"{live_cfg_shown.get('sl_type','—')} / {live_cfg_shown.get('sl_pts','—')} pts")
            cc2[3].metric("Target",    f"{live_cfg_shown.get('tgt_type','—')} / {live_cfg_shown.get('tgt_pts','—')} pts")
            cc3 = st.columns(4)
            cc3[0].metric("Qty",       live_cfg_shown.get("qty", "—"))
            cc3[1].metric("Cooldown",  f"{live_cfg_shown.get('cd_s','—')}s" if live_cfg_shown.get("cd_en") else "Off")
            cc3[2].metric("No Overlap",live_cfg_shown.get("no_overlap", "—"))
            cc3[3].metric("Dhan",      "Enabled" if live_cfg_shown.get("dhan_en") else "Disabled")

        st.divider()

        # ── Two-column layout: position + EMA vs log ──────────────────────
        main_col, log_col = st.columns([1.6, 1])

        with main_col:
            pos = st.session_state.get("live_position")
            ltp_cur = st.session_state.get("live_ltp")
            ef_v = st.session_state.get("live_ema_fast_val")
            es_v = st.session_state.get("live_ema_slow_val")

            # EMA values
            if ef_v is not None:
                ev1, ev2, ev3, ev4 = st.columns(4)
                ev1.metric(f"EMA {fast_ema}",  f"{ef_v:.2f}")
                ev2.metric(f"EMA {slow_ema}",  f"{es_v:.2f}")
                if ef_v is not None and es_v is not None:
                    diff = ef_v - es_v
                    ev3.metric("EMA Diff", f"{diff:+.2f}",
                               delta="Bullish" if diff > 0 else "Bearish")
                if ltp_cur:
                    ev4.metric("Last LTP", f"{ltp_cur:.2f}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Position card
            if pos is not None:
                tt  = pos["trade_type"]
                ep  = pos["entry_price"]
                sl  = pos["sl"]
                tgt = pos["target"]
                pnl_live = ((ltp_cur - ep) if (ltp_cur and tt == "buy")
                            else ((ep - ltp_cur) if ltp_cur else 0)) * qty
                clr = "card-green" if pnl_live >= 0 else "card-red"
                tt_clr = "#00e5b4" if tt == "buy" else "#ff4d6d"
                st.markdown(f"""
                <div class="card {clr}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                        <span style="font-family:Syne,sans-serif;font-size:13px;font-weight:800;
                                     letter-spacing:2px;color:{tt_clr}">● OPEN {tt.upper()} POSITION</span>
                        <span style="font-family:JetBrains Mono,monospace;font-size:18px;font-weight:700;
                                     color:{'#00e5b4' if pnl_live>=0 else '#ff4d6d'}">
                            ₹{pnl_live:+.2f}
                        </span>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
                                font-family:JetBrains Mono,monospace;font-size:12px">
                        <div><div style="color:#4a5568;font-size:10px">ENTRY</div>{ep:.2f}</div>
                        <div><div style="color:#4a5568;font-size:10px">LTP</div>{ltp_cur:.2f if ltp_cur else '—'}</div>
                        <div><div style="color:#ff4d6d;font-size:10px">SL</div>{sl:.2f if sl else '—'}</div>
                        <div><div style="color:#00e5b4;font-size:10px">TARGET</div>{tgt:.2f if tgt else 'N/A'}</div>
                    </div>
                    <div style="margin-top:10px;font-family:Syne,sans-serif;font-size:11px;color:#4a5568">
                        📅 Entry: {pos['entry_time']} &nbsp;·&nbsp; {pos.get('entry_reason','')}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:28px;color:#4a5568;
                                          font-family:Syne,sans-serif;font-size:13px">
                    No open position
                </div>""", unsafe_allow_html=True)

            # Chart
            df_live = st.session_state.get("live_chart_df")
            if df_live is not None and not df_live.empty:
                st.plotly_chart(
                    build_chart(df_live, fast_ema, slow_ema, position=pos,
                                title=f"Live  ·  {ticker_label}  ·  {interval}"),
                    use_container_width=True,
                )

                # Last fetched candle row
                st.markdown("**Last Fetched Candle** (⚠ yfinance data may have delay)")
                last_row = df_live.iloc[-1]
                lc = pd.DataFrame([{
                    "Time":   str(df_live.index[-1]),
                    "Open":   round(float(last_row["Open"]), 2),
                    "High":   round(float(last_row["High"]), 2),
                    "Low":    round(float(last_row["Low"]),  2),
                    "Close":  round(float(last_row["Close"]),2),
                    "Volume": int(last_row.get("Volume", 0)),
                    f"EMA{fast_ema}": round(float(df_live[f"EMA_{fast_ema}"].iloc[-1]), 2) if f"EMA_{fast_ema}" in df_live.columns else "—",
                    f"EMA{slow_ema}": round(float(df_live[f"EMA_{slow_ema}"].iloc[-1]), 2) if f"EMA_{slow_ema}" in df_live.columns else "—",
                }])
                st.dataframe(lc, use_container_width=True, hide_index=True)

        with log_col:
            st.markdown("**Activity Log**")
            logs = st.session_state.get("live_log", [])
            log_html = "".join(f"<div style='border-bottom:1px solid #0f1623;padding:3px 0'>{l}</div>"
                               for l in reversed(logs)) if logs else "<div style='color:#4a5568'>No activity yet…</div>"
            st.markdown(f'<div class="logbox">{log_html}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — TRADE HISTORY
    # ════════════════════════════════════════════════════════════════════
    with tab_hist:
        ltp_widget(ticker_sym, ticker_label)
        st.markdown("### 📋 Trade History")
        st.caption("All completed trades — updates in real-time even while live trading is active.")

        hist = st.session_state.get("trade_history", [])

        h1, h2 = st.columns([4, 1])
        with h2:
            if st.button("🗑  Clear History", use_container_width=True, key="clr_hist"):
                st.session_state["trade_history"] = []
                st.rerun()

        if not hist:
            st.markdown("""
            <div class="card" style="text-align:center;padding:40px;color:#4a5568;
                                      font-family:Syne,sans-serif">
                No completed trades yet. Run a backtest or start live trading.
            </div>""", unsafe_allow_html=True)
        else:
            hdf = pd.DataFrame(hist)
            # Summary
            tot_pnl = float(hdf["PnL (₹)"].sum())
            wins    = int((hdf["PnL (₹)"] > 0).sum())
            n_h     = len(hdf)
            acc_h   = wins / n_h * 100 if n_h else 0
            avg_pnl = float(hdf["PnL (₹)"].mean())
            best_h  = float(hdf["PnL (₹)"].max())
            worst_h = float(hdf["PnL (₹)"].min())

            hm = st.columns(6)
            hm[0].metric("Total Trades", n_h)
            hm[1].metric("Win Rate",     f"{acc_h:.1f}%")
            hm[2].metric("Total PnL",    f"₹{tot_pnl:+,.2f}")
            hm[3].metric("Avg PnL",      f"₹{avg_pnl:+.2f}")
            hm[4].metric("Best",         f"₹{best_h:+.2f}")
            hm[5].metric("Worst",        f"₹{worst_h:+.2f}")

            st.divider()

            disp_h = hdf.copy()

            def _hstyle(row):
                pnl = row["PnL (₹)"]
                bg  = "#0d2a1a" if pnl > 0 else ("#2a0d14" if pnl < 0 else "")
                return [f"background-color:{bg}" for _ in row]

            styled_h = (disp_h.style
                        .apply(_hstyle, axis=1)
                        .format({"PnL (₹)": "₹{:+.2f}",
                                 "Entry Price": "{:.2f}", "Exit Price": "{:.2f}"}))
            st.dataframe(styled_h, use_container_width=True, height=500)

            # PnL Curve
            st.markdown("#### Cumulative PnL Curve")
            hdf["Cum PnL"] = hdf["PnL (₹)"].cumsum()
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=list(range(1, len(hdf) + 1)), y=hdf["Cum PnL"],
                mode="lines+markers",
                line=dict(color="#00e5b4", width=2),
                marker=dict(size=5, color=["#00e5b4" if p >= 0 else "#ff4d6d"
                                           for p in hdf["PnL (₹)"]]),
                fill="tozeroy",
                fillcolor="rgba(0,229,180,0.06)",
                name="Cumulative PnL",
            ))
            fig_pnl.add_hline(y=0, line=dict(color="#4a5568", width=1, dash="dash"))
            fig_pnl.update_layout(
                height=280, **_PLOTLY_LAYOUT,
                xaxis_title="Trade #", yaxis_title="PnL (₹)",
                showlegend=False,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
