"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SMART INVESTING - Professional Algorithmic Trading Platform     ║
║      EMA Crossover · Elliott Wave · Live Trading · Dhan Broker Integration   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
import datetime
from datetime import timezone, timedelta
import pytz
import warnings
import traceback

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main-title  { font-size:2.2rem; font-weight:700; color:#00D4AA; letter-spacing:-1px; }
  .ltp-card    { background:linear-gradient(135deg,#1a1f2e,#141922);
                 padding:10px 18px; border-radius:10px; margin-bottom:14px;
                 border:1px solid #2a3040; display:inline-block; }
  .pos-card    { background:#1a1f2e; padding:12px; border-radius:10px; margin-top:6px; }
  .wave-card   { background:#111827; padding:10px; border-radius:8px; font-size:.82rem; margin-top:6px; }
  .log-box     { background:#080c14; padding:8px; border-radius:6px; height:210px;
                 overflow-y:auto; font-size:.75rem; font-family:monospace; }
  .metric-row  { display:flex; gap:10px; flex-wrap:wrap; }
  .stTabs [data-baseweb="tab"] { font-size:.92rem; font-weight:600; padding:8px 18px; }
  .stTabs [aria-selected="true"]{ color:#00D4AA !important; border-bottom-color:#00D4AA !important; }
  div[data-testid="stSidebar"] { background:#0a0d14; }
  .stButton>button { border-radius:8px; font-weight:600; }
  .stButton>button[kind="primary"]{ background:#00D4AA; color:#000; border:none; }
  .stButton>button[kind="primary"]:hover{ background:#00f0c0; }
  .violation-badge{ background:#ff4757; color:white; padding:2px 8px;
                    border-radius:12px; font-size:.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "Nifty 50":  "^NSEI",
    "BankNifty": "^NSEBANK",
    "Sensex":    "^BSESN",
    "BTC-USD":   "BTC-USD",
    "ETH-USD":   "ETH-USD",
    "Gold":      "GC=F",
    "Silver":    "SI=F",
    "Custom":    "CUSTOM",
}

TIMEFRAME_PERIODS = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

# Larger periods used to warm-up EMA so gap-up/down never shows NaN
WARMUP_MAP = {
    "1m":  {"1d": "7d",   "5d": "7d",   "7d": "7d"},
    "5m":  {"1d": "1mo",  "5d": "1mo",  "7d": "1mo",  "1mo": "1mo"},
    "15m": {"1d": "1mo",  "5d": "1mo",  "7d": "1mo",  "1mo": "1mo"},
    "1h":  {"1d": "3mo",  "5d": "3mo",  "7d": "3mo",  "1mo": "3mo",
             "3mo": "6mo","6mo": "1y",  "1y": "2y",   "2y": "2y"},
    "1d":  {"5d": "3mo",  "7d": "3mo",  "1mo": "3mo", "3mo": "6mo",
             "6mo": "1y","1y": "2y",    "2y": "5y",   "5y": "10y",
             "10y": "20y","20y": "20y"},
    "1wk": {"1mo": "2y",  "3mo": "2y",  "6mo": "2y",  "1y": "2y",
             "2y": "5y", "5y": "10y",   "10y": "20y","20y": "20y"},
}

PERIOD_DAYS = {
    "1d": 1,   "5d": 5,  "7d": 7,   "1mo": 30, "3mo": 90,
    "6mo": 180,"1y": 365,"2y": 730,"5y": 1825,"10y": 3650,"20y": 7300,
}

STRATEGIES    = ["EMA Crossover", "Elliott Wave", "Simple Buy", "Simple Sell"]
SL_TYPES      = ["Custom Points", "ATR Based", "Trailing SL",
                  "Reverse EMA Crossover", "Risk Reward Based"]
TARGET_TYPES  = ["Custom Points", "ATR Based", "Trailing Target",
                  "EMA Crossover", "Risk Reward Based"]
CROSS_TYPES   = ["Simple Crossover", "Custom Candle Size", "ATR Based Candle Size"]


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        live_running=False,
        live_log=[],
        live_data=None,
        live_ltp=None,
        live_prev_close=None,
        live_ema_fast=None,
        live_ema_slow=None,
        live_elliott={},
        current_position=None,
        completed_trades=[],
        bt_results=None,
        bt_df=None,
        bt_fig=None,
        opt_results=None,
        opt_strategy=None,
        cooldown_ts=0.0,
        last_completed_candle=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder smoothing — matches TradingView."""
    prev_c = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - prev_c).abs(),
         (df["Low"]  - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """
    EMA that EXACTLY matches TradingView:
    ewm(span=n, adjust=False, min_periods=1)
    This ensures warm-up values are correct even for the very first bar.
    """
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


@st.cache_data(ttl=20, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str,
               fast: int = 9, slow: int = 26) -> pd.DataFrame:
    """
    Fetch OHLCV and add EMA/ATR indicators.
    Uses a WARMUP period (larger than requested) so EMA is fully seeded
    before the user's visible window — avoids NaN on gap-up / gap-down days.
    """
    warmup = WARMUP_MAP.get(interval, {}).get(period, period)
    try:
        raw = yf.download(symbol, interval=interval, period=warmup,
                          auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            return pd.DataFrame()
        raw = _flatten_cols(raw)

        # Compute on FULL warmup so first visible bar has correct EMA
        raw["EMA_fast"] = calc_ema(raw["Close"], fast)
        raw["EMA_slow"] = calc_ema(raw["Close"], slow)
        raw["ATR"]      = compute_atr(raw, 14)

        # Localise index
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert(IST)

        # Trim to requested period
        cutoff = pd.Timestamp.now(tz=IST) - pd.Timedelta(days=PERIOD_DAYS.get(period, 30))
        df = raw[raw.index >= cutoff].copy()
        return df if not df.empty else raw.tail(200).copy()
    except Exception as exc:
        st.error(f"Data fetch error: {exc}")
        return pd.DataFrame()


def fetch_ltp(symbol: str):
    """Quick last-price + prev-close fetch."""
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info
        ltp  = float(fi.last_price)
        prev = float(fi.previous_close) if hasattr(fi, "previous_close") else ltp
        return ltp, prev
    except Exception:
        pass
    try:
        df = yf.download(symbol, period="5d", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        df = _flatten_cols(df)
        if len(df) >= 2:
            return float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
        if len(df) == 1:
            return float(df["Close"].iloc[-1]), float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# EMA CROSSOVER SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def get_ema_signals(df: pd.DataFrame, fast: int, slow: int,
                    min_angle: float = 0.0,
                    cross_type: str = "Simple Crossover",
                    custom_size: float = 10.0) -> pd.Series:
    """
    Returns Series: +1 = buy signal, -1 = sell signal, 0 = none.
    Signal at index N  →  entry at index N+1 open price.
    Crossover detection matches TradingView exactly because we use
    the same EMA formula.
    """
    fe = calc_ema(df["Close"], fast)
    se = calc_ema(df["Close"], slow)

    prev_fe = fe.shift(1)
    prev_se = se.shift(1)

    cross_up   = (fe >  se) & (prev_fe <= prev_se)
    cross_down = (fe <  se) & (prev_fe >= prev_se)

    signals = pd.Series(0, index=df.index, dtype=int)

    for i in range(1, len(df)):
        if not (cross_up.iloc[i] or cross_down.iloc[i]):
            continue

        # Angle filter (absolute price-change of fast EMA per bar)
        angle = abs(fe.iloc[i] - fe.iloc[i - 1])
        if angle < min_angle:
            continue

        # Candle-size filter
        candle_sz = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
        atr_val   = float(df["ATR"].iloc[i]) if "ATR" in df.columns else custom_size

        if cross_type == "Custom Candle Size" and candle_sz < custom_size:
            continue
        if cross_type == "ATR Based Candle Size" and candle_sz < atr_val:
            continue

        signals.iloc[i] = 1 if cross_up.iloc[i] else -1

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# ELLIOTT WAVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _detect_zigzag(df: pd.DataFrame, window: int = 5) -> list:
    """
    Return alternating list of (timestamp, price, 'H'|'L') pivot points.
    """
    n = len(df)
    highs, lows = df["High"].values, df["Low"].values
    ph = [False] * n
    pl = [False] * n

    for i in range(window, n - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) \
        and all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
            ph[i] = True
        if all(lows[i]  <= lows[i - j]  for j in range(1, window + 1)) \
        and all(lows[i]  <= lows[i + j]  for j in range(1, window + 1)):
            pl[i] = True

    raw = []
    for i in range(n):
        if ph[i]:
            raw.append((df.index[i], highs[i], "H"))
        if pl[i]:
            raw.append((df.index[i], lows[i],  "L"))

    # Deduplicate: keep dominant pivot of same type in sequence
    filtered = []
    for p in sorted(raw, key=lambda x: x[0]):
        if not filtered or filtered[-1][2] != p[2]:
            filtered.append(p)
        else:
            if p[2] == "H" and p[1] > filtered[-1][1]:
                filtered[-1] = p
            elif p[2] == "L" and p[1] < filtered[-1][1]:
                filtered[-1] = p
    return filtered


def analyze_elliott_waves(df: pd.DataFrame) -> dict:
    """
    Full Elliott Wave analysis following professional rules:
      - Wave 2 never retraces > 100% of Wave 1
      - Wave 3 is never the shortest impulse wave  
      - Wave 4 never overlaps Wave 1 price territory
      - Fibonacci projections for targets
    Returns comprehensive wave information dict.
    """
    base = dict(status="Insufficient data", waves=[], current_wave=None,
                next_wave=None, next_target=None, sl_level=None,
                trend="Unknown", fib_levels={}, signal=0)

    if len(df) < 25:
        return base

    w_size = max(3, len(df) // 25)
    pivots = _detect_zigzag(df, window=w_size)

    if len(pivots) < 6:
        base["status"] = f"Only {len(pivots)} pivots — building..."
        base["waves"]  = pivots
        return base

    best_seq  = None
    best_trend = "Unknown"

    # Slide a 6-pivot window to find the most recent valid impulse
    for start in range(len(pivots) - 5, -1, -1):
        pts = pivots[start: start + 6]
        p0, p1, p2, p3, p4, p5 = [x[1] for x in pts]
        t0 = pts[0][2]

        if t0 == "L":  # Bullish impulse candidate
            w1 = p1 - p0;  w2_ret = p1 - p2
            w3 = p3 - p2;  w4_ret = p3 - p4
            w5 = p5 - p4

            ok  = w1  > 0 and w3 > 0 and w5 > 0           # positive waves
            ok &= w2_ret / w1 < 1.0 if w1 else False       # rule: W2 < 100%
            ok &= p2 > p0                                   # rule: W2 > W0
            ok &= p4 > p1                                   # rule: W4 ≠ overlap W1
            # rule: W3 not shortest
            ok &= w3 >= min(w1, w5)
            if ok:
                best_seq   = pts
                best_trend = "Bullish"
                break

        elif t0 == "H":  # Bearish impulse candidate
            w1 = p0 - p1;  w2_ret = p2 - p1
            w3 = p2 - p3;  w4_ret = p4 - p3
            w5 = p4 - p5

            ok  = w1  > 0 and w3 > 0 and w5 > 0
            ok &= w2_ret / w1 < 1.0 if w1 else False
            ok &= p2 < p0
            ok &= p4 < p1
            ok &= w3 >= min(w1, w5)
            if ok:
                best_seq   = pts
                best_trend = "Bearish"
                break

    if best_seq is None:
        # Fall back to last 6 pivots for display
        best_seq   = pivots[-6:]
        best_trend = "Unclear"

    wave_labels = ["W0", "W1", "W2", "W3", "W4", "W5"]
    wave_info = [
        {"label": wave_labels[i] if i < 6 else f"W{i}",
         "price": p[1], "time": p[0], "type": p[2]}
        for i, p in enumerate(best_seq)
    ]

    # ── Next wave target & SL via Fibonacci projections ──
    next_target  = None
    sl_level     = None
    next_wave_lbl = None
    signal       = 0
    completed_cnt = len(best_seq) - 1  # number of completed swings

    prices = [p[1] for p in best_seq]

    if best_trend == "Bullish":
        if completed_cnt == 4:   # In Wave 4, expecting Wave 5
            w1_len = prices[1] - prices[0]
            next_target  = prices[4] + w1_len           # W5 = W1 length projection
            sl_level     = prices[3]                     # W4 low is SL
            next_wave_lbl = "Wave 5 ↑"
            signal = 1
        elif completed_cnt == 2: # In Wave 2, expecting Wave 3
            w1_len = prices[1] - prices[0]
            next_target  = prices[2] + 1.618 * w1_len   # W3 = 1.618 × W1
            sl_level     = prices[1]                     # W1 high (invalidation)
            next_wave_lbl = "Wave 3 ↑ (strongest)"
            signal = 1
        elif completed_cnt >= 5: # Expecting correction
            w5_h = prices[5]; w0_l = prices[0]
            next_target  = w5_h - 0.382 * (w5_h - w0_l)  # 38.2% correction
            sl_level     = w5_h
            next_wave_lbl = "Wave A ↓ (correction)"
            signal = -1
    elif best_trend == "Bearish":
        if completed_cnt == 4:
            w1_len = prices[0] - prices[1]
            next_target  = prices[4] - w1_len
            sl_level     = prices[3]
            next_wave_lbl = "Wave 5 ↓"
            signal = -1
        elif completed_cnt == 2:
            w1_len = prices[0] - prices[1]
            next_target  = prices[2] - 1.618 * w1_len
            sl_level     = prices[1]
            next_wave_lbl = "Wave 3 ↓ (strongest)"
            signal = -1
        elif completed_cnt >= 5:
            w5_l = prices[5]; w0_h = prices[0]
            next_target  = w5_l + 0.382 * (w0_h - w5_l)
            sl_level     = w5_l
            next_wave_lbl = "Wave A ↑ (correction)"
            signal = 1

    # Fibonacci retracement/extension levels from last completed wave
    fib_levels = {}
    if len(prices) >= 4:
        a, b = prices[-2], prices[-1]
        rng  = abs(b - a)
        direction = 1 if b > a else -1
        for pct, label in [(0.236,"23.6%"),(0.382,"38.2%"),(0.500,"50%"),
                            (0.618,"61.8%"),(1.0,"100%"),(1.618,"161.8%")]:
            fib_levels[label] = round(b + direction * pct * rng, 2)

    completed_waves = {f"W{i}→W{i+1}": f"{prices[i]:.2f} → {prices[i+1]:.2f}"
                       for i in range(len(prices) - 1)}

    return dict(
        status=f"{best_trend} impulse · {completed_cnt} waves identified",
        waves=wave_info,
        current_wave=f"Wave {completed_cnt}",
        next_wave=next_wave_lbl,
        next_target=next_target,
        sl_level=sl_level,
        trend=best_trend,
        fib_levels=fib_levels,
        completed_waves=completed_waves,
        signal=signal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SL / TARGET HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_sl(entry: float, ttype: str, sl_type: str,
           sl_pts: float, atr: float, rr: float,
           fe: float, se: float) -> float:
    if ttype == "Buy":
        if sl_type == "Custom Points":          return entry - sl_pts
        if sl_type == "ATR Based":              return entry - atr
        if sl_type == "Trailing SL":            return entry - sl_pts
        if sl_type == "Reverse EMA Crossover":  return min(se, entry - 0.5 * sl_pts)
        if sl_type == "Risk Reward Based":      return entry - sl_pts
    else:  # Sell
        if sl_type == "Custom Points":          return entry + sl_pts
        if sl_type == "ATR Based":              return entry + atr
        if sl_type == "Trailing SL":            return entry + sl_pts
        if sl_type == "Reverse EMA Crossover":  return max(se, entry + 0.5 * sl_pts)
        if sl_type == "Risk Reward Based":      return entry + sl_pts
    return entry - sl_pts if ttype == "Buy" else entry + sl_pts


def get_target(entry: float, ttype: str, tgt_type: str,
               tgt_pts: float, atr: float, rr: float,
               risk_pts: float, fe: float, se: float) -> float:
    if ttype == "Buy":
        if tgt_type == "Custom Points":         return entry + tgt_pts
        if tgt_type == "ATR Based":             return entry + atr * rr
        if tgt_type == "Trailing Target":       return entry + tgt_pts   # display only
        if tgt_type == "EMA Crossover":         return entry + tgt_pts   # exit on crossover
        if tgt_type == "Risk Reward Based":     return entry + risk_pts * rr
    else:
        if tgt_type == "Custom Points":         return entry - tgt_pts
        if tgt_type == "ATR Based":             return entry - atr * rr
        if tgt_type == "Trailing Target":       return entry - tgt_pts
        if tgt_type == "EMA Crossover":         return entry - tgt_pts
        if tgt_type == "Risk Reward Based":     return entry - risk_pts * rr
    return entry + tgt_pts if ttype == "Buy" else entry - tgt_pts


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Rules (conservative, as close to live as possible):
      • EMA Crossover / Elliott Wave  → signal on candle N, ENTRY at candle N+1 open
      • Simple Buy / Simple Sell      → immediate entry at current close (no wait)
      • SL: compared against candle Low  (Buy) / High (Sell)  first  ← conservative
      • Target: compared against High (Buy) / Low (Sell)
      • If BOTH hit on same candle → violation logged; SL exit used
      • No cooldown in backtesting
    """
    if df.empty or len(df) < 4:
        return {"trades": [], "violations": [], "metrics": {}}

    strategy    = cfg["strategy"]
    fast_n      = cfg["fast_ema"]
    slow_n      = cfg["slow_ema"]
    qty         = cfg["quantity"]
    sl_type     = cfg["sl_type"]
    sl_pts      = cfg["sl_points"]
    tgt_type    = cfg["target_type"]
    tgt_pts     = cfg["target_points"]
    rr          = cfg["rr_ratio"]
    min_angle   = cfg.get("min_angle", 0.0)
    cross_type  = cfg.get("crossover_type", "Simple Crossover")
    custom_sz   = cfg.get("custom_candle_size", 10.0)
    no_overlap  = cfg.get("no_overlap", True)

    fe_s = calc_ema(df["Close"], fast_n)
    se_s = calc_ema(df["Close"], slow_n)

    # Pre-compute EMA crossover reverse flags (for EMA-based SL/Target exits)
    fe_cross_down = (fe_s <  se_s) & (fe_s.shift(1) >= se_s.shift(1))
    fe_cross_up   = (fe_s >  se_s) & (fe_s.shift(1) <= se_s.shift(1))

    # Generate primary entry signals
    if strategy == "EMA Crossover":
        raw_sig = get_ema_signals(df, fast_n, slow_n, min_angle, cross_type, custom_sz)
    elif strategy == "Elliott Wave":
        ew = analyze_elliott_waves(df)
        raw_sig = pd.Series(0, index=df.index, dtype=int)
        if ew and ew.get("signal") != 0 and len(df) > 5:
            raw_sig.iloc[-4] = ew["signal"]   # anchor signal a few candles back
    elif strategy == "Simple Buy":
        raw_sig = pd.Series(1, index=df.index, dtype=int)
    elif strategy == "Simple Sell":
        raw_sig = pd.Series(-1, index=df.index, dtype=int)
    else:
        raw_sig = pd.Series(0, index=df.index, dtype=int)

    # Elliott Wave auto SL/Target
    ew_analysis = analyze_elliott_waves(df) if strategy == "Elliott Wave" else None

    trades, violations = [], []
    in_trade  = False
    trade_rec = {}
    peak_ltp  = 0.0  # for trailing

    for i in range(1, len(df)):
        row    = df.iloc[i]
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        atr    = float(df["ATR"].iloc[i]) if "ATR" in df.columns else sl_pts
        fe     = float(fe_s.iloc[i])
        se_val = float(se_s.iloc[i])

        if in_trade:
            tt    = trade_rec["Type"]
            entry = trade_rec["Entry Price"]
            sl_v  = trade_rec["_sl"]
            tgt_v = trade_rec["_tgt"]

            # Trailing SL update
            if sl_type == "Trailing SL":
                if tt == "Buy":
                    new_sl = c - sl_pts
                    if new_sl > sl_v:
                        trade_rec["_sl"] = sl_v = new_sl
                else:
                    new_sl = c + sl_pts
                    if new_sl < sl_v:
                        trade_rec["_sl"] = sl_v = new_sl

            exit_price  = None
            exit_reason = None
            violated    = False

            # ── BUY exit logic ──
            if tt == "Buy":
                sl_hit  = l <= sl_v
                tgt_hit = h >= tgt_v and tgt_type != "Trailing Target"
                ema_exit = fe_cross_down.iloc[i] and sl_type == "Reverse EMA Crossover"
                ema_tgt  = fe_cross_down.iloc[i] and tgt_type == "EMA Crossover"

                if sl_hit and tgt_hit:
                    violated   = True          # Both hit → SL wins (conservative)
                    exit_price  = sl_v
                    exit_reason = "SL Hit ⚠️ (violation)"
                elif sl_hit:
                    exit_price  = sl_v;        exit_reason = "SL Hit"
                elif tgt_hit:
                    exit_price  = tgt_v;       exit_reason = "Target Hit ✅"
                elif ema_exit:
                    exit_price  = c;           exit_reason = "EMA Reversal SL"
                elif ema_tgt:
                    exit_price  = c;           exit_reason = "EMA Reversal Target"

            # ── SELL exit logic ──
            else:
                sl_hit  = h >= sl_v
                tgt_hit = l <= tgt_v and tgt_type != "Trailing Target"
                ema_exit = fe_cross_up.iloc[i]   and sl_type == "Reverse EMA Crossover"
                ema_tgt  = fe_cross_up.iloc[i]   and tgt_type == "EMA Crossover"

                if sl_hit and tgt_hit:
                    violated   = True
                    exit_price  = sl_v
                    exit_reason = "SL Hit ⚠️ (violation)"
                elif sl_hit:
                    exit_price  = sl_v;        exit_reason = "SL Hit"
                elif tgt_hit:
                    exit_price  = tgt_v;       exit_reason = "Target Hit ✅"
                elif ema_exit:
                    exit_price  = c;           exit_reason = "EMA Reversal SL"
                elif ema_tgt:
                    exit_price  = c;           exit_reason = "EMA Reversal Target"

            if exit_price is not None:
                pnl = ((exit_price - entry) if tt == "Buy"
                       else (entry - exit_price)) * qty
                trade_rec.update(
                    {"Exit Time": row.name, "Exit Price": round(exit_price, 2),
                     "High": round(h, 2),   "Low": round(l, 2),
                     "Exit Reason": exit_reason, "PnL": round(pnl, 2)}
                )
                # Clean private keys
                for k in ("_sl", "_tgt"):
                    trade_rec.pop(k, None)
                trades.append(trade_rec)
                if violated:
                    violations.append(trade_rec)
                in_trade = False
                trade_rec = {}
                continue

        # ── Entry ──
        if not in_trade:
            if strategy in ("EMA Crossover", "Elliott Wave"):
                sig = int(raw_sig.iloc[i - 1])   # signal on N-1, enter at N open
                entry_p = o
            else:
                sig = int(raw_sig.iloc[i])         # Simple: immediate close
                entry_p = c

            if sig == 0:
                continue

            ttype = "Buy" if sig == 1 else "Sell"
            sl_v  = get_sl(entry_p, ttype, sl_type, sl_pts, atr, rr, fe, se_val)
            if ew_analysis and strategy == "Elliott Wave":
                if ew_analysis.get("sl_level"):
                    sl_v = ew_analysis["sl_level"]
            risk   = abs(entry_p - sl_v)
            tgt_v  = get_target(entry_p, ttype, tgt_type, tgt_pts, atr, rr, risk, fe, se_val)
            if ew_analysis and strategy == "Elliott Wave":
                if ew_analysis.get("next_target"):
                    tgt_v = ew_analysis["next_target"]

            reason = (f"EMA {fast_n}/{slow_n} {'Cross Up' if sig==1 else 'Cross Down'}"
                      if strategy == "EMA Crossover"
                      else f"{strategy} {'Buy' if sig==1 else 'Sell'}")

            in_trade  = True
            trade_rec = {
                "Entry Time":   row.name,
                "Exit Time":    None,
                "Type":         ttype,
                "Entry Price":  round(entry_p, 2),
                "Exit Price":   None,
                "SL":           round(sl_v, 2),
                "Target":       round(tgt_v, 2),
                "High":         None,
                "Low":          None,
                "Signal Reason":reason,
                "Exit Reason":  None,
                "PnL":          None,
                "_sl":          sl_v,
                "_tgt":         tgt_v,
            }
            peak_ltp = entry_p

    # Force-close open trade at last bar
    if in_trade:
        last = df.iloc[-1]
        ep   = float(last["Close"])
        tt   = trade_rec["Type"]
        pnl  = ((ep - trade_rec["Entry Price"]) if tt == "Buy"
                else (trade_rec["Entry Price"] - ep)) * qty
        trade_rec.update(
            {"Exit Time": df.index[-1], "Exit Price": round(ep, 2),
             "High": round(float(last["High"]), 2),
             "Low":  round(float(last["Low"]),  2),
             "Exit Reason": "End of Data", "PnL": round(pnl, 2)}
        )
        for k in ("_sl", "_tgt"):
            trade_rec.pop(k, None)
        trades.append(trade_rec)

    metrics = {}
    if trades:
        pnls = [t["PnL"] for t in trades if t["PnL"] is not None]
        wins = sum(1 for p in pnls if p > 0)
        metrics = {
            "Total Trades": len(trades),
            "Wins":   wins,
            "Losses": len(trades) - wins,
            "Accuracy (%)": round(wins / len(trades) * 100, 1),
            "Total PnL":    round(sum(pnls), 2),
            "Avg PnL":      round(np.mean(pnls), 2),
            "Max Win":      round(max(pnls), 2),
            "Max Loss":     round(min(pnls), 2),
            "Violations":   len(violations),
        }

    return {"trades": trades, "violations": violations, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────
def build_backtest_chart(df: pd.DataFrame, trades: list, cfg: dict) -> go.Figure:
    fe_line = calc_ema(df["Close"], cfg["fast_ema"])
    se_line = calc_ema(df["Close"], cfg["slow_ema"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28],
                        vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color="#00ff88"), fillcolor="#00ff8833"),
        decreasing=dict(line=dict(color="#ff4757"), fillcolor="#ff475733"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=fe_line, name=f"EMA {cfg['fast_ema']}",
        line=dict(color="#FFD700", width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=se_line, name=f"EMA {cfg['slow_ema']}",
        line=dict(color="#00BFFF", width=1.5)
    ), row=1, col=1)

    buy_e  = [t for t in trades if t["Type"] == "Buy"]
    sell_e = [t for t in trades if t["Type"] == "Sell"]

    # Entry markers
    if buy_e:
        fig.add_trace(go.Scatter(
            x=[t["Entry Time"] for t in buy_e],
            y=[t["Entry Price"] for t in buy_e],
            mode="markers", name="Buy Entry",
            marker=dict(symbol="triangle-up", size=13, color="#00ff88",
                        line=dict(color="#fff", width=1)),
        ), row=1, col=1)
    if sell_e:
        fig.add_trace(go.Scatter(
            x=[t["Entry Time"] for t in sell_e],
            y=[t["Entry Price"] for t in sell_e],
            mode="markers", name="Sell Entry",
            marker=dict(symbol="triangle-down", size=13, color="#ff4757",
                        line=dict(color="#fff", width=1)),
        ), row=1, col=1)

    # Exit markers
    win_trades  = [t for t in trades if (t["PnL"] or 0) >= 0]
    loss_trades = [t for t in trades if (t["PnL"] or 0) <  0]
    if win_trades:
        fig.add_trace(go.Scatter(
            x=[t["Exit Time"] for t in win_trades],
            y=[t["Exit Price"] for t in win_trades],
            mode="markers", name="Win Exit",
            marker=dict(symbol="circle", size=9, color="#00ff88",
                        line=dict(color="#000", width=1)),
        ), row=1, col=1)
    if loss_trades:
        fig.add_trace(go.Scatter(
            x=[t["Exit Time"] for t in loss_trades],
            y=[t["Exit Price"] for t in loss_trades],
            mode="markers", name="Loss Exit",
            marker=dict(symbol="x", size=10, color="#ff4757"),
        ), row=1, col=1)

    # Volume
    vol_colors = ["#00ff8866" if c >= o else "#ff475766"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=620,
        paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0, font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2330")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER — LIVE TRADING
# ─────────────────────────────────────────────────────────────────────────────
def build_live_chart(df: pd.DataFrame, cfg: dict) -> go.Figure:
    fe_line = calc_ema(df["Close"], cfg["fast_ema"])
    se_line = calc_ema(df["Close"], cfg["slow_ema"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color="#00ff88"), fillcolor="#00ff8822"),
        decreasing=dict(line=dict(color="#ff4757"), fillcolor="#ff475722"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=fe_line, name=f"EMA {cfg['fast_ema']}",
        line=dict(color="#FFD700", width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=se_line, name=f"EMA {cfg['slow_ema']}",
        line=dict(color="#00BFFF", width=1.5)
    ), row=1, col=1)

    # Horizontal lines for current position
    pos = st.session_state.current_position
    if pos:
        for lvl, color, lbl in [
            (pos["entry_price"], "#ffffff", f"Entry {pos['entry_price']:.2f}"),
            (pos["sl"],          "#ff4757", f"SL {pos['sl']:.2f}"),
            (pos["target"],      "#00ff88", f"Tgt {pos['target']:.2f}"),
        ]:
            fig.add_hline(y=lvl, line_color=color, line_dash="dash",
                          line_width=1.2,
                          annotation_text=lbl,
                          annotation_font=dict(size=10, color=color),
                          row=1, col=1)

    # Volume
    vol_colors = ["#00ff8855" if c >= o else "#ff475555"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=480,
        paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0, font=dict(size=11)),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2330")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DHAN ORDER PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
def place_dhan_order(cfg: dict, trade_type: str, phase: str,
                      price: float, log_fn) -> None:
    """
    phase: 'ENTRY' or 'EXIT'
    For options: BUY CE on algo-Buy, BUY PE on algo-Sell (pure buyer).
    """
    try:
        from dhanhq import dhanhq  # type: ignore
        dhan = dhanhq(cfg["dhan_client_id"], cfg["dhan_access_token"])

        is_opt = cfg.get("options_trading", False)

        if is_opt:
            sec_id = cfg["ce_security_id"] if trade_type == "Buy" else cfg["pe_security_id"]
            txn    = "BUY" if phase == "ENTRY" else "SELL"
            seg    = cfg.get("options_exchange", "NSE_FNO")
            qty    = cfg.get("options_qty", 65)
            ot     = cfg.get("entry_order_type" if phase == "ENTRY" else "exit_order_type", "MARKET")
            lp     = price if ot == "LIMIT" else 0
            resp   = dhan.place_order(transactionType=txn, exchangeSegment=seg,
                                      productType="INTRADAY", orderType=ot,
                                      validity="DAY", securityId=str(sec_id),
                                      quantity=qty, price=lp, triggerPrice=0)
            log_fn(f"📤 Options {txn} {sec_id} ×{qty} @{lp} → {resp}")

        else:
            product  = cfg.get("product_type", "INTRADAY")
            exchange = cfg.get("exchange", "NSE")
            sec_id   = cfg.get("security_id", "1594")
            qty      = cfg.get("quantity", 1)
            ot       = cfg.get("entry_order_type" if phase == "ENTRY" else "exit_order_type", "MARKET")
            lp       = price if ot == "LIMIT" else 0
            if phase == "ENTRY":
                txn = "BUY" if trade_type == "Buy" else "SELL"
            else:
                txn = "SELL" if trade_type == "Buy" else "BUY"  # close
            resp = dhan.place_order(transactionType=txn,
                                    exchangeSegment=f"{exchange}_EQ",
                                    productType=product, orderType=ot,
                                    validity="DAY", securityId=str(sec_id),
                                    quantity=qty, price=lp, triggerPrice=0)
            log_fn(f"📤 Equity {txn} {sec_id} ×{qty} @{lp} → {resp}")

    except ImportError:
        log_fn("⚠️ dhanhq not installed. Run: pip install dhanhq")
    except Exception as exc:
        log_fn(f"❌ Dhan order error: {str(exc)[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRADING LOOP  (background thread)
# ─────────────────────────────────────────────────────────────────────────────
def _live_loop(symbol: str, interval: str, cfg: dict):
    log = st.session_state.live_log

    def _log(msg):
        ts = datetime.datetime.now(IST).strftime("%H:%M:%S")
        log.insert(0, f"[{ts}] {msg}")
        if len(log) > 120:
            log.pop()

    _log(f"🚀 Started | {symbol} {interval} | {cfg['strategy']}")

    fast_n     = cfg["fast_ema"]
    slow_n     = cfg["slow_ema"]
    sl_type    = cfg["sl_type"]
    sl_pts     = cfg["sl_points"]
    tgt_type   = cfg["target_type"]
    tgt_pts    = cfg["target_points"]
    rr         = cfg["rr_ratio"]
    qty        = cfg["quantity"]
    cooldown   = cfg.get("cooldown", 5)
    use_cd     = cfg.get("use_cooldown", True)
    no_ov      = cfg.get("no_overlap", True)
    strategy   = cfg["strategy"]

    last_completed_candle = None

    while st.session_state.live_running:
        try:
            time.sleep(1.5)  # Respect yfinance rate limits
            warmup = WARMUP_MAP.get(interval, {}).get(cfg.get("period", "1d"), "1mo")
            raw = yf.download(symbol, interval=interval, period=warmup,
                              auto_adjust=True, progress=False, threads=False)

            if raw.empty:
                _log("⚠️ Empty data — retrying")
                time.sleep(3)
                continue

            raw = _flatten_cols(raw)
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            raw.index = raw.index.tz_convert(IST)

            raw["EMA_fast"] = calc_ema(raw["Close"], fast_n)
            raw["EMA_slow"] = calc_ema(raw["Close"], slow_n)
            raw["ATR"]      = compute_atr(raw, 14)

            ltp       = float(raw["Close"].iloc[-1])
            prev_cl   = float(raw["Close"].iloc[-2]) if len(raw) > 1 else ltp
            fe_live   = float(raw["EMA_fast"].iloc[-1])
            se_live   = float(raw["EMA_slow"].iloc[-1])
            atr_live  = float(raw["ATR"].iloc[-1])

            st.session_state.live_ltp        = ltp
            st.session_state.live_prev_close = prev_cl
            st.session_state.live_data       = raw
            st.session_state.live_ema_fast   = fe_live
            st.session_state.live_ema_slow   = se_live

            if strategy == "Elliott Wave":
                st.session_state.live_elliott = analyze_elliott_waves(raw)

            # ── Manage open position ──
            pos = st.session_state.current_position
            if pos:
                ttype = pos["type"]
                entry = pos["entry_price"]
                sl_v  = pos["sl"]
                tgt_v = pos["target"]

                # Update trailing SL with LTP
                if sl_type == "Trailing SL":
                    if ttype == "Buy":
                        new_sl = ltp - sl_pts
                        if new_sl > sl_v:
                            sl_v = new_sl
                            st.session_state.current_position["sl"] = sl_v
                            _log(f"↑ Trail SL → {sl_v:.2f}")
                    else:
                        new_sl = ltp + sl_pts
                        if new_sl < sl_v:
                            sl_v = new_sl
                            st.session_state.current_position["sl"] = sl_v
                            _log(f"↓ Trail SL → {sl_v:.2f}")

                exit_price  = None
                exit_reason = None

                if ttype == "Buy":
                    if ltp <= sl_v:
                        exit_price = ltp; exit_reason = "SL Hit"
                    elif tgt_type != "Trailing Target" and ltp >= tgt_v:
                        exit_price = ltp; exit_reason = "Target Hit ✅"
                    elif sl_type == "Reverse EMA Crossover":
                        if len(raw) >= 3:
                            fe2 = float(raw["EMA_fast"].iloc[-2])
                            se2 = float(raw["EMA_slow"].iloc[-2])
                            fe3 = float(raw["EMA_fast"].iloc[-3])
                            se3 = float(raw["EMA_slow"].iloc[-3])
                            if fe2 < se2 and fe3 >= se3:
                                exit_price = ltp; exit_reason = "EMA Reversal SL"
                    elif tgt_type == "EMA Crossover":
                        if len(raw) >= 3:
                            fe2 = float(raw["EMA_fast"].iloc[-2])
                            se2 = float(raw["EMA_slow"].iloc[-2])
                            fe3 = float(raw["EMA_fast"].iloc[-3])
                            se3 = float(raw["EMA_slow"].iloc[-3])
                            if fe2 < se2 and fe3 >= se3:
                                exit_price = ltp; exit_reason = "EMA Target Crossover"
                else:
                    if ltp >= sl_v:
                        exit_price = ltp; exit_reason = "SL Hit"
                    elif tgt_type != "Trailing Target" and ltp <= tgt_v:
                        exit_price = ltp; exit_reason = "Target Hit ✅"
                    elif sl_type == "Reverse EMA Crossover":
                        if len(raw) >= 3:
                            fe2 = float(raw["EMA_fast"].iloc[-2])
                            se2 = float(raw["EMA_slow"].iloc[-2])
                            fe3 = float(raw["EMA_fast"].iloc[-3])
                            se3 = float(raw["EMA_slow"].iloc[-3])
                            if fe2 > se2 and fe3 <= se3:
                                exit_price = ltp; exit_reason = "EMA Reversal SL"

                if exit_price is not None:
                    pnl = ((exit_price - entry) if ttype == "Buy"
                           else (entry - exit_price)) * qty
                    completed = {**pos,
                                 "exit_price": round(exit_price, 2),
                                 "exit_time":  datetime.datetime.now(IST),
                                 "exit_reason": exit_reason,
                                 "pnl":         round(pnl, 2)}
                    st.session_state.completed_trades.append(completed)
                    st.session_state.current_position = None
                    st.session_state.cooldown_ts = time.time()
                    _log(f"🔴 {ttype} EXIT @ {exit_price:.2f} | {exit_reason} | P&L {pnl:+.2f}")

                    if cfg.get("dhan_enabled"):
                        place_dhan_order(cfg, ttype, "EXIT", exit_price, _log)
                    continue

            # ── Look for new entry ──
            if st.session_state.current_position is None:
                if use_cd and (time.time() - st.session_state.cooldown_ts) < cooldown:
                    continue

                sig        = 0
                entry_p    = ltp
                completed_candle = raw.index[-2] if len(raw) >= 2 else None

                if strategy == "Simple Buy":
                    sig = 1
                elif strategy == "Simple Sell":
                    sig = -1
                elif strategy in ("EMA Crossover", "Elliott Wave"):
                    # Only fire when a NEW completed candle is available
                    if completed_candle and completed_candle != last_completed_candle:
                        last_completed_candle = completed_candle
                        if len(raw) >= 3:
                            fe_c1 = float(raw["EMA_fast"].iloc[-2])
                            se_c1 = float(raw["EMA_slow"].iloc[-2])
                            fe_c2 = float(raw["EMA_fast"].iloc[-3])
                            se_c2 = float(raw["EMA_slow"].iloc[-3])
                            if strategy == "EMA Crossover":
                                if fe_c1 > se_c1 and fe_c2 <= se_c2:
                                    sig = 1;  _log("📊 EMA Cross UP")
                                elif fe_c1 < se_c1 and fe_c2 >= se_c2:
                                    sig = -1; _log("📊 EMA Cross DOWN")
                            else:  # Elliott Wave
                                ew = st.session_state.live_elliott
                                sig = ew.get("signal", 0) if ew else 0
                        # Entry at current bar's open (next candle approximation)
                        entry_p = float(raw["Open"].iloc[-1])

                if sig != 0:
                    ttype = "Buy" if sig == 1 else "Sell"
                    sl_v  = get_sl(entry_p, ttype, sl_type, sl_pts, atr_live, rr, fe_live, se_live)
                    ew_a  = st.session_state.live_elliott
                    if strategy == "Elliott Wave" and ew_a and ew_a.get("sl_level"):
                        sl_v = ew_a["sl_level"]
                    risk  = abs(entry_p - sl_v)
                    tgt_v = get_target(entry_p, ttype, tgt_type, tgt_pts, atr_live, rr, risk, fe_live, se_live)
                    if strategy == "Elliott Wave" and ew_a and ew_a.get("next_target"):
                        tgt_v = ew_a["next_target"]

                    st.session_state.current_position = {
                        "type":        ttype,
                        "entry_price": round(entry_p, 2),
                        "entry_time":  datetime.datetime.now(IST),
                        "sl":          round(sl_v, 2),
                        "target":      round(tgt_v, 2),
                        "qty":         qty,
                        "strategy":    strategy,
                    }
                    st.session_state.cooldown_ts = time.time()
                    _log(f"🟢 {ttype} ENTRY @ {entry_p:.2f} | SL {sl_v:.2f} | Tgt {tgt_v:.2f}")

                    if cfg.get("dhan_enabled"):
                        place_dhan_order(cfg, ttype, "ENTRY", entry_p, _log)

        except Exception as exc:
            _log(f"❌ Error: {str(exc)[:100]}")
            time.sleep(5)

    _log("⛔ Stopped")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────
def run_optimization(df: pd.DataFrame, strategy: str,
                     min_acc: float, cfg_base: dict) -> list:
    results = []

    if strategy == "EMA Crossover":
        for f in range(3, 26, 2):
            for s in range(10, 61, 3):
                if f >= s:
                    continue
                cfg = {**cfg_base, "fast_ema": f, "slow_ema": s}
                bt  = run_backtest(df, cfg)
                m   = bt.get("metrics", {})
                if m.get("Total Trades", 0) >= 3:
                    results.append({
                        "Fast EMA": f, "Slow EMA": s,
                        "Accuracy (%)": m["Accuracy (%)"],
                        "Total PnL":    m["Total PnL"],
                        "Total Trades": m["Total Trades"],
                        "Avg PnL":      m["Avg PnL"],
                        "Max Win":      m["Max Win"],
                        "Max Loss":     m["Max Loss"],
                    })

    elif strategy == "Elliott Wave":
        for w in [3, 5, 7, 10, 15]:
            # We can't vary EW window in cfg easily; vary min_angle instead
            for ma in [0.0, 0.05, 0.1, 0.2]:
                cfg = {**cfg_base, "strategy": "Elliott Wave",
                       "fast_ema": cfg_base["fast_ema"],
                       "slow_ema": cfg_base["slow_ema"],
                       "min_angle": ma}
                bt  = run_backtest(df, cfg)
                m   = bt.get("metrics", {})
                if m.get("Total Trades", 0) >= 2:
                    results.append({
                        "EW Pivot Window": w, "Min Angle": ma,
                        "Accuracy (%)": m["Accuracy (%)"],
                        "Total PnL":    m["Total PnL"],
                        "Total Trades": m["Total Trades"],
                    })

    filtered = [r for r in results if r["Accuracy (%)"] >= min_acc]
    filtered.sort(key=lambda x: (x["Accuracy (%)"], x["Total PnL"]), reverse=True)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# LTP HEADER WIDGET
# ─────────────────────────────────────────────────────────────────────────────
def ltp_header(symbol: str):
    ltp, prev = fetch_ltp(symbol)
    if ltp:
        chg  = ltp - (prev or ltp)
        pct  = chg / (prev or ltp) * 100
        arr  = "▲" if chg >= 0 else "▼"
        col  = "#00ff88" if chg >= 0 else "#ff4757"
        st.markdown(
            f'<div class="ltp-card">'
            f'<span style="color:#8892a4;font-size:.8rem;">{symbol}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#fff;font-size:1.35rem;font-weight:700;">{ltp:,.2f}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{col};font-size:1rem;">'
            f'{arr} {abs(chg):,.2f} ({abs(pct):.2f}%)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="ltp-card" style="color:#666;">LTP unavailable for {symbol}</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar() -> dict:
    sb = st.sidebar
    sb.markdown('<p class="main-title">📈 Smart Investing</p>', unsafe_allow_html=True)
    sb.divider()

    # ── Instrument ──
    sb.markdown("### 📌 Instrument")
    tname  = sb.selectbox("Ticker", list(TICKERS.keys()), key="sb_ticker")
    if tname == "Custom":
        sym = sb.text_input("Custom Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS", key="sb_custom")
        ticker_sym = sym.strip().upper()
    else:
        ticker_sym = TICKERS[tname]

    # ── Timeframe ──
    sb.markdown("### ⏱️ Timeframe")
    interval   = sb.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()), index=2, key="sb_interval")
    periods_ok = TIMEFRAME_PERIODS[interval]
    period     = sb.selectbox("Period", periods_ok,
                               index=min(1, len(periods_ok) - 1), key="sb_period")
    quantity   = sb.number_input("Quantity", min_value=1, value=1, step=1, key="sb_qty")

    # ── Strategy ──
    sb.markdown("### 🎯 Strategy")
    strategy = sb.selectbox("Strategy", STRATEGIES, key="sb_strategy")

    # EMA config
    sb.markdown("#### EMA Parameters")
    fast_ema = sb.number_input("Fast EMA", min_value=2, max_value=100, value=9,  key="sb_fast")
    slow_ema = sb.number_input("Slow EMA", min_value=3, max_value=300, value=15, key="sb_slow")

    min_angle   = 0.0
    cross_type  = "Simple Crossover"
    custom_sz   = 10.0

    if strategy == "EMA Crossover":
        use_ang = sb.checkbox("Min Crossover Angle Filter", value=False, key="sb_ang_en")
        if use_ang:
            min_angle = sb.number_input("Min Angle (price units)", 0.0, value=0.0,
                                         step=0.05, key="sb_ang")
        cross_type = sb.selectbox("Crossover Type", CROSS_TYPES, key="sb_ctype")
        if cross_type == "Custom Candle Size":
            custom_sz = sb.number_input("Min Candle Size (pts)", 0.1, value=10.0, key="sb_csz")

    # ── Stop Loss ──
    sb.markdown("### 🛡️ Stop Loss")
    sl_type = sb.selectbox("SL Type", SL_TYPES, key="sb_sltype")
    sl_pts  = sb.number_input("SL Points", 0.1, value=10.0, step=0.5, key="sb_slpts")

    # ── Target ──
    sb.markdown("### 🎯 Target")
    tgt_type = sb.selectbox("Target Type", TARGET_TYPES, key="sb_tgttype")
    tgt_pts  = sb.number_input("Target Points", 0.1, value=20.0, step=0.5, key="sb_tgtpts")
    rr_ratio = sb.number_input("Risk : Reward", 0.1, value=2.0, step=0.1, key="sb_rr")

    # ── Trade Management ──
    sb.markdown("### ⚙️ Trade Management")
    use_cd  = sb.checkbox("Cooldown Between Trades", value=True, key="sb_usecd")
    cd_secs = sb.number_input("Cooldown (sec)", 0, value=5, key="sb_cd") if use_cd else 5
    no_ovlp = sb.checkbox("Prevent Overlapping Trades", value=True, key="sb_noovlp")

    # ── Dhan Broker ──
    sb.markdown("### 🏦 Dhan Broker")
    dhan_en = sb.checkbox("Enable Dhan Broker", value=False, key="sb_dhan")
    dhan_cfg: dict = {}

    if dhan_en:
        dhan_cid = sb.text_input("Client ID",     key="sb_dcid")
        dhan_tok = sb.text_input("Access Token",  key="sb_dtok", type="password")
        is_opt   = sb.checkbox("Options Trading", value=False, key="sb_opt")

        if is_opt:
            opt_exch = sb.selectbox("Options Exchange", ["NSE_FNO", "BSE_FNO"], key="sb_oexch")
            ce_sid   = sb.text_input("CE Security ID", value="57749", key="sb_ce")
            pe_sid   = sb.text_input("PE Security ID", value="57716", key="sb_pe")
            opt_qty  = sb.number_input("Options Qty", 1, value=65, key="sb_oqty")
            ent_ot   = sb.selectbox("Entry Order Type", ["MARKET", "LIMIT"], key="sb_oeot")
            ext_ot   = sb.selectbox("Exit Order Type",  ["MARKET", "LIMIT"], key="sb_oxot")
            dhan_cfg = dict(options_trading=True, options_exchange=opt_exch,
                             ce_security_id=ce_sid, pe_security_id=pe_sid,
                             options_qty=opt_qty, entry_order_type=ent_ot,
                             exit_order_type=ext_ot)
        else:
            prod_type = sb.selectbox("Product Type", ["INTRADAY", "DELIVERY"], key="sb_prod")
            exchange  = sb.selectbox("Exchange",     ["NSE", "BSE"], key="sb_exch")
            sec_id    = sb.text_input("Security ID", value="1594", key="sb_secid")
            ord_qty   = sb.number_input("Order Qty", 1, value=1, key="sb_ordqty")
            ent_ot    = sb.selectbox("Entry Order Type", ["LIMIT", "MARKET"], key="sb_eot")
            ext_ot    = sb.selectbox("Exit Order Type",  ["MARKET", "LIMIT"], key="sb_xot")
            dhan_cfg  = dict(options_trading=False, product_type=prod_type,
                              exchange=exchange, security_id=sec_id,
                              quantity=ord_qty, entry_order_type=ent_ot,
                              exit_order_type=ext_ot)

        dhan_cfg.update(dhan_client_id=dhan_cid, dhan_access_token=dhan_tok)

    cfg = dict(
        ticker=ticker_sym, ticker_name=tname,
        interval=interval, period=period, quantity=quantity,
        strategy=strategy, fast_ema=fast_ema, slow_ema=slow_ema,
        sl_type=sl_type, sl_points=sl_pts,
        target_type=tgt_type, target_points=tgt_pts, rr_ratio=rr_ratio,
        min_angle=min_angle, crossover_type=cross_type, custom_candle_size=custom_sz,
        use_cooldown=use_cd, cooldown=cd_secs, no_overlap=no_ovlp,
        dhan_enabled=dhan_en, **dhan_cfg,
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────
def tab_backtest(cfg: dict):
    ltp_header(cfg["ticker"])

    run_btn = st.button("▶️  Run Backtest", type="primary", key="bt_run")

    if run_btn:
        with st.spinner("Fetching data & running backtest..."):
            df = fetch_data(cfg["ticker"], cfg["interval"], cfg["period"],
                            cfg["fast_ema"], cfg["slow_ema"])
            if df.empty:
                st.error("No data returned. Verify ticker / period combination.")
                return
            res = run_backtest(df, cfg)
            st.session_state.bt_results = res
            st.session_state.bt_df      = df
            st.session_state.bt_fig     = (
                build_backtest_chart(df, res["trades"], cfg)
                if res["trades"] else None
            )

    res = st.session_state.bt_results
    if not res:
        st.info("Configure parameters in the sidebar then press **Run Backtest**.")
        return

    m = res["metrics"]
    if m:
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Trades",         m.get("Total Trades", 0))
        c2.metric("Wins",           m.get("Wins", 0))
        c3.metric("Losses",         m.get("Losses", 0))
        c4.metric("Accuracy",       f"{m.get('Accuracy (%)', 0):.1f}%")
        c5.metric("Total PnL",      f"₹{m.get('Total PnL', 0):,.0f}")
        c6.metric("Avg PnL / Trade",f"₹{m.get('Avg PnL', 0):,.0f}")
        c7.metric("Violations ⚠️",  m.get("Violations", 0))
        st.divider()

    if st.session_state.bt_fig:
        st.plotly_chart(st.session_state.bt_fig,
                        use_container_width=True, config={"displayModeBar": False})

    trades = res["trades"]
    viol   = res["violations"]

    if viol:
        st.warning(
            f"⚠️ **{len(viol)} violation(s)** — SL *and* Target both hit on same candle. "
            "SL exit used (conservative). These bars may diverge from live trading."
        )

    if trades:
        st.subheader(f"📋 Trade Log — {len(trades)} trades")
        df_t = pd.DataFrame(trades)
        for c in ("Entry Time", "Exit Time"):
            if c in df_t.columns:
                df_t[c] = pd.to_datetime(df_t[c]).dt.strftime("%Y-%m-%d %H:%M")

        def _pnl_color(v):
            if v is None: return ""
            return "color:#00ff88;font-weight:600" if v > 0 else "color:#ff4757;font-weight:600"

        styled = df_t.style.applymap(_pnl_color, subset=["PnL"])
        st.dataframe(styled, use_container_width=True, height=420)

        if viol:
            with st.expander(f"🚨 View {len(viol)} Violation Trades"):
                st.dataframe(pd.DataFrame(viol), use_container_width=True)
    else:
        st.info("No trades generated. Adjust strategy parameters or period.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LIVE TRADING
# ─────────────────────────────────────────────────────────────────────────────
def tab_live(cfg: dict):
    # LTP (real-time if running, else static)
    if st.session_state.live_ltp:
        ltp   = st.session_state.live_ltp
        prev  = st.session_state.live_prev_close or ltp
        chg   = ltp - prev
        pct   = chg / prev * 100 if prev else 0
        arr   = "▲" if chg >= 0 else "▼"
        col   = "#00ff88" if chg >= 0 else "#ff4757"
        st.markdown(
            f'<div class="ltp-card">'
            f'<span style="color:#8892a4;font-size:.8rem;">{cfg["ticker"]} &nbsp;🔴 LIVE</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#fff;font-size:1.35rem;font-weight:700;">{ltp:,.2f}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{col};font-size:1rem;">{arr} {abs(chg):,.2f} ({abs(pct):.2f}%)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        ltp_header(cfg["ticker"])

    # Control buttons
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 4])
    running = st.session_state.live_running

    if c1.button("▶️ START",      type="primary",  disabled=running,  key="lt_start"):
        st.session_state.live_running = True
        st.session_state.live_log     = []
        t = threading.Thread(
            target=_live_loop,
            args=(cfg["ticker"], cfg["interval"], cfg),
            daemon=True,
        )
        t.start()
        st.rerun()

    if c2.button("⏹️ STOP",       disabled=not running, key="lt_stop"):
        st.session_state.live_running = False
        st.rerun()

    if c3.button("🔄 SQUARE OFF", key="lt_sq"):
        pos = st.session_state.current_position
        if pos:
            ep  = st.session_state.live_ltp or pos["entry_price"]
            pnl = ((ep - pos["entry_price"]) if pos["type"] == "Buy"
                   else (pos["entry_price"] - ep)) * pos["qty"]
            st.session_state.completed_trades.append(
                {**pos, "exit_price": round(ep, 2),
                 "exit_time": datetime.datetime.now(IST),
                 "exit_reason": "Manual Square Off", "pnl": round(pnl, 2)}
            )
            st.session_state.current_position = None
            st.rerun()
        else:
            st.toast("No open position to square off.")

    # Status badge
    badge = ("🟢 **RUNNING**" if running else "🔴 **STOPPED**")
    st.markdown(badge)

    # Active config summary
    if running:
        with st.expander("📋 Active Configuration", expanded=False):
            rows = [
                ("Ticker",       cfg["ticker"]),
                ("Interval",     cfg["interval"]),
                ("Period",       cfg["period"]),
                ("Strategy",     cfg["strategy"]),
                (f"Fast EMA",    cfg["fast_ema"]),
                (f"Slow EMA",    cfg["slow_ema"]),
                ("SL Type",      cfg["sl_type"]),
                ("SL Points",    cfg["sl_points"]),
                ("Target Type",  cfg["target_type"]),
                ("Target Points",cfg["target_points"]),
                ("RR Ratio",     cfg["rr_ratio"]),
                ("Quantity",     cfg["quantity"]),
                ("Cooldown",     f"{cfg['cooldown']} s"),
                ("No Overlap",   cfg["no_overlap"]),
                ("Dhan Enabled", cfg["dhan_enabled"]),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Parameter", "Value"]),
                         use_container_width=True, hide_index=True)

    # Main layout
    col_chart, col_info = st.columns([3, 1], gap="medium")

    with col_chart:
        df = st.session_state.live_data
        if df is not None and not df.empty:
            st.plotly_chart(build_live_chart(df, cfg),
                            use_container_width=True,
                            config={"displayModeBar": False})

            # Last fetched candle row
            last = df.iloc[-1]
            fe_v  = st.session_state.live_ema_fast
            se_v  = st.session_state.live_ema_slow
            st.markdown("**📊 Last Fetched Candle**")
            lc = st.columns(7)
            lc[0].metric("Time",  df.index[-1].strftime("%H:%M"))
            lc[1].metric("O",  f"{last['Open']:.2f}")
            lc[2].metric("H",  f"{last['High']:.2f}")
            lc[3].metric("L",  f"{last['Low']:.2f}")
            lc[4].metric("C",  f"{last['Close']:.2f}")
            lc[5].metric(f"EMA {cfg['fast_ema']}", f"{fe_v:.2f}" if fe_v else "—")
            lc[6].metric(f"EMA {cfg['slow_ema']}", f"{se_v:.2f}" if se_v else "—")
        else:
            st.info("Start live trading to see the chart.")

    with col_info:
        # ── Current Position ──
        st.markdown("**📌 Position**")
        pos = st.session_state.current_position
        if pos:
            ltp  = st.session_state.live_ltp or pos["entry_price"]
            upnl = ((ltp - pos["entry_price"]) if pos["type"] == "Buy"
                    else (pos["entry_price"] - ltp)) * pos["qty"]
            col  = "#00ff88" if upnl >= 0 else "#ff4757"
            pct_move = (upnl / (pos["entry_price"] * pos["qty"])) * 100 if pos["entry_price"] else 0
            st.markdown(
                f'<div class="pos-card" style="border-left:3px solid {col};">'
                f'<b style="color:{col};font-size:1rem;">{pos["type"]}</b>'
                f' &nbsp;<span style="color:#aaa;font-size:.8rem;">{pos["strategy"]}</span><br>'
                f'<b>Entry:</b> {pos["entry_price"]:.2f}<br>'
                f'<b>LTP:</b> {ltp:.2f}<br>'
                f'<span style="color:#ff4757;"><b>SL:</b> {pos["sl"]:.2f}</span><br>'
                f'<span style="color:#00ff88;"><b>Tgt:</b> {pos["target"]:.2f}</span><br>'
                f'<b>Qty:</b> {pos["qty"]}<br>'
                f'<b>Unrealised P&L:</b><br>'
                f'<b style="color:{col};font-size:1.1rem;">{upnl:+.2f} ({pct_move:+.2f}%)</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div style="color:#666;font-style:italic;">No open position</div>',
                        unsafe_allow_html=True)

        # ── Elliott Wave panel ──
        if cfg["strategy"] == "Elliott Wave":
            st.markdown("**🌊 Elliott Wave**")
            ew = st.session_state.live_elliott
            if ew and ew.get("trend") != "Unknown":
                tc = "#00ff88" if ew["trend"] == "Bullish" else "#ff4757"
                fibs_html = "".join(
                    f'<tr><td style="color:#8892a4">{k}</td><td style="color:#fff">{v}</td></tr>'
                    for k, v in ew.get("fib_levels", {}).items()
                )
                cw_html = "".join(
                    f'<tr><td style="color:#8892a4">{k}</td><td style="color:#ccc">{v}</td></tr>'
                    for k, v in ew.get("completed_waves", {}).items()
                )
                st.markdown(
                    f'<div class="wave-card">'
                    f'<b style="color:{tc}">{ew["trend"]}</b>'
                    f' &nbsp;<span style="color:#aaa">{ew.get("status","")}</span><br><br>'
                    f'<b>Current Wave:</b> {ew.get("current_wave","—")}<br>'
                    f'<b>Next:</b> {ew.get("next_wave","—")}<br>'
                    f'<b>Next Target:</b> '
                    + (f'{ew["next_target"]:.2f}' if ew.get("next_target") else "—")
                    + '<br>'
                    f'<b>Wave SL:</b> '
                    + (f'{ew["sl_level"]:.2f}' if ew.get("sl_level") else "—")
                    + '<br><br>'
                    f'<details><summary style="cursor:pointer;color:#00D4AA">Completed Waves</summary>'
                    f'<table style="width:100%">{cw_html}</table></details>'
                    f'<details><summary style="cursor:pointer;color:#00D4AA">Fibonacci Levels</summary>'
                    f'<table style="width:100%">{fibs_html}</table></details>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="wave-card" style="color:#666">Analysing waves...</div>',
                            unsafe_allow_html=True)

        # ── Activity Log ──
        st.markdown("**📜 Activity Log**")
        logs = st.session_state.live_log[:20]
        rows = "".join(
            f'<div style="border-bottom:1px solid #1e2330;padding:2px 0;color:#aaa">{l}</div>'
            for l in logs
        )
        st.markdown(f'<div class="log-box">{rows}</div>', unsafe_allow_html=True)

    # Completed trades (always visible, even while running)
    ct = st.session_state.completed_trades
    if ct:
        st.divider()
        st.subheader(f"✅ Completed Trades — {len(ct)}")
        rows_ct = []
        for t in ct:
            et = t.get("entry_time", ""); xt = t.get("exit_time", "")
            rows_ct.append({
                "Type":       t.get("type", ""),
                "Entry Time": et.strftime("%H:%M:%S") if hasattr(et, "strftime") else str(et),
                "Exit Time":  xt.strftime("%H:%M:%S") if hasattr(xt, "strftime") else str(xt),
                "Entry":      t.get("entry_price", ""),
                "Exit":       t.get("exit_price", ""),
                "SL":         t.get("sl", ""),
                "Target":     t.get("target", ""),
                "Reason":     t.get("exit_reason", ""),
                "PnL":        t.get("pnl", ""),
            })
        st.dataframe(pd.DataFrame(rows_ct), use_container_width=True)

    # Auto-refresh while running
    if running:
        time.sleep(2)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def tab_history():
    all_t = st.session_state.completed_trades
    ltp_text = ""

    st.subheader("📚 Full Trade History")
    if not all_t:
        st.info("No completed trades yet.")
        return

    rows = []
    for t in all_t:
        et = t.get("entry_time", t.get("Entry Time", ""))
        xt = t.get("exit_time",  t.get("Exit Time",  ""))
        rows.append({
            "Type":       t.get("type",        t.get("Type", "")),
            "Entry Time": et.strftime("%Y-%m-%d %H:%M:%S") if hasattr(et, "strftime") else str(et),
            "Exit Time":  xt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(xt, "strftime") else str(xt),
            "Entry Price":t.get("entry_price", t.get("Entry Price", "")),
            "Exit Price": t.get("exit_price",  t.get("Exit Price", "")),
            "SL":         t.get("sl",          t.get("SL", "")),
            "Target":     t.get("target",      t.get("Target", "")),
            "Exit Reason":t.get("exit_reason", t.get("Exit Reason", "")),
            "Strategy":   t.get("strategy",    ""),
            "PnL":        round(float(t.get("pnl", t.get("PnL", 0)) or 0), 2),
        })

    df_h = pd.DataFrame(rows)
    tot   = df_h["PnL"].sum()
    wins  = (df_h["PnL"] > 0).sum()
    loss  = (df_h["PnL"] < 0).sum()
    acc   = wins / len(df_h) * 100 if len(df_h) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades",  len(df_h))
    c2.metric("Wins",          wins)
    c3.metric("Losses",        loss)
    c4.metric("Accuracy",      f"{acc:.1f}%")
    c5.metric("Total PnL",     f"₹{tot:,.2f}")

    # Cumulative PnL chart
    if len(df_h) > 1:
        cum = df_h["PnL"].cumsum()
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=list(range(1, len(cum) + 1)), y=cum,
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.12)",
            line=dict(color="#00D4AA", width=2),
        ))
        fig_c.update_layout(
            template="plotly_dark", height=220,
            paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
            margin=dict(l=0, r=0, t=20, b=0),
            title=dict(text="Cumulative P&L", font=dict(color="#aaa", size=12)),
            xaxis=dict(title="Trade #", showgrid=False),
            yaxis=dict(title="₹", gridcolor="#1e2330"),
        )
        st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})

    # Table
    def _col(v):
        if v is None: return ""
        try:
            v = float(v)
        except Exception:
            return ""
        return "color:#00ff88;font-weight:600" if v > 0 else "color:#ff4757;font-weight:600"

    styled = df_h.style.applymap(_col, subset=["PnL"])
    st.dataframe(styled, use_container_width=True, height=450)

    if st.button("🗑️ Clear History"):
        st.session_state.completed_trades = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
def tab_optimize(cfg: dict):
    st.subheader("⚙️ Strategy Optimisation")

    c1, c2, c3 = st.columns(3)
    min_acc  = c1.number_input("Min Accuracy (%)", 1.0, 100.0, 60.0, 1.0)
    opt_stgy = c2.selectbox("Optimise Strategy", ["EMA Crossover", "Elliott Wave"])
    run_btn  = c3.button("🔍 Run Optimisation", type="primary")

    st.caption(
        "EMA Crossover tests all fast/slow combinations in range 3–25 / 10–60. "
        "Results sorted by accuracy ↓ then total PnL ↓. Click **Apply** to push to sidebar."
    )

    if run_btn:
        with st.spinner("Running optimisation — this may take a minute..."):
            df = fetch_data(cfg["ticker"], cfg["interval"], cfg["period"],
                            cfg["fast_ema"], cfg["slow_ema"])
            if df.empty:
                st.error("No data to optimise on.")
                return
            results = run_optimization(df, opt_stgy, min_acc, cfg)
            st.session_state.opt_results  = results
            st.session_state.opt_strategy = opt_stgy

    res = st.session_state.opt_results
    if res is None:
        st.info("Configure and press **Run Optimisation**.")
        return

    if not res:
        st.warning(f"No results with accuracy ≥ {min_acc}%. Try lowering the threshold.")
        return

    st.success(f"✅ {len(res)} configuration(s) meet the criteria.")

    for i, row in enumerate(res[:25]):
        col_a, col_b = st.columns([5, 1])
        with col_a:
            if st.session_state.opt_strategy == "EMA Crossover":
                st.markdown(
                    f"**#{i+1}** &nbsp; EMA `{row['Fast EMA']}` / `{row['Slow EMA']}`"
                    f" &nbsp;|&nbsp; Acc **{row['Accuracy (%)']:.1f}%**"
                    f" &nbsp;|&nbsp; PnL **₹{row['Total PnL']:,.0f}**"
                    f" &nbsp;|&nbsp; Trades {row['Total Trades']}"
                    f" &nbsp;|&nbsp; Avg {row['Avg PnL']:,.0f}"
                )
            else:
                st.markdown(
                    f"**#{i+1}** &nbsp; Acc **{row['Accuracy (%)']:.1f}%**"
                    f" &nbsp;|&nbsp; PnL **₹{row['Total PnL']:,.0f}**"
                    f" &nbsp;|&nbsp; Trades {row['Total Trades']}"
                )
        with col_b:
            if st.button("✅ Apply", key=f"opt_apply_{i}"):
                if st.session_state.opt_strategy == "EMA Crossover":
                    st.session_state["sb_fast"] = row["Fast EMA"]
                    st.session_state["sb_slow"] = row["Slow EMA"]
                st.success("Applied! Sidebar will refresh on next interaction.")
                st.rerun()

    with st.expander("📄 Full Results Table"):
        st.dataframe(pd.DataFrame(res), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Backtesting",
        "⚡  Live Trading",
        "📚  Trade History",
        "⚙️  Optimisation",
    ])

    with tab1:
        tab_backtest(cfg)

    with tab2:
        tab_live(cfg)

    with tab3:
        tab_history()

    with tab4:
        tab_optimize(cfg)


if __name__ == "__main__":
    main()
