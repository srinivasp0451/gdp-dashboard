"""
Smart Investing - Professional Algorithmic Trading Platform
===========================================================
Author: Smart Investing Team
Features: Multi-strategy backtesting & live trading | Elliott Wave | Dhan Broker
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
import datetime
import requests
import warnings
import math
from collections import deque
import json

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

APP_VERSION = "2.0.0"

TICKERS = {
    "Nifty 50":    "^NSEI",
    "BankNifty":   "^NSEBANK",
    "Sensex":      "^BSESN",
    "BTC/USD":     "BTC-USD",
    "ETH/USD":     "ETH-USD",
    "Gold":        "GC=F",
    "Silver":      "SI=F",
    "Custom":      "__CUSTOM__",
}

TIMEFRAME_PERIODS: dict[str, list[str]] = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

# Warmup period map: extra historical data to avoid NaN in indicators
WARMUP_MAP = {
    "1m":  "7d",
    "5m":  "1mo",
    "15m": "1mo",
    "1h":  "6mo",
    "1d":  "5y",
    "1wk": "10y",
}

STRATEGIES = [
    "EMA Crossover",
    "Anticipatory EMA",
    "Elliott Wave",
    "Simple Buy",
    "Simple Sell",
]

SL_TYPES = [
    "Custom Points",
    "ATR Based",
    "Risk Reward Based",
    "Trailing SL",
    "Auto SL",
    "EMA Reverse Crossover",
    "Swing Low/High",
    "Candle Low/High",
    "Support / Resistance",
    "Volatility Based",
]

TARGET_TYPES = [
    "Custom Points",
    "ATR Based",
    "Risk Reward Based",
    "Trailing Target (Display Only)",
    "Auto Target",
    "EMA Reverse Crossover",
    "Swing High/Low",
    "Candle High/Low",
    "Support / Resistance",
    "Volatility Based",
]

CROSSOVER_TYPES = ["Simple Crossover", "Custom Candle Size", "ATR Based Candle Size"]

# ══════════════════════════════════════════════════════════════════════════════
# THREAD-SAFE STATE STORE
# ══════════════════════════════════════════════════════════════════════════════

_TS_LOCK  = threading.Lock()
_TS: dict = {}

def _ts_get(key, default=None):
    with _TS_LOCK:
        return _TS.get(key, default)

def _ts_set(key, value):
    with _TS_LOCK:
        _TS[key] = value

def _ts_append(key, value):
    with _TS_LOCK:
        if key not in _TS:
            _TS[key] = []
        _TS[key].append(value)

def _ts_clear(key):
    with _TS_LOCK:
        _TS[key] = []

# yfinance rate-limit guard
_YF_LOCK      = threading.Lock()
_LAST_YF_CALL = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

_SS_DEFAULTS = {
    "live_running":       False,
    "stop_event":         None,
    "live_thread":        None,
    "backtest_results":   None,
    "backtest_fig":       None,
    "opt_results":        None,
    "auto_refresh":       False,
}

def init_ss():
    for k, v in _SS_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# IP REGISTRATION (SEBI)
# ══════════════════════════════════════════════════════════════════════════════

def get_my_ip() -> str:
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=5)
        return r.json().get("ip", "Unknown")
    except Exception:
        return "Unknown"

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _rate_limited_download(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """yfinance download with 1.5 s global rate limit."""
    global _LAST_YF_CALL
    with _YF_LOCK:
        gap = time.time() - _LAST_YF_CALL
        if gap < 1.5:
            time.sleep(1.5 - gap)
        _LAST_YF_CALL = time.time()
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        return df
    except Exception:
        return None


def fetch_with_warmup(ticker: str, period: str, interval: str,
                      min_bars: int = 200) -> tuple[pd.DataFrame | None, int]:
    """
    Fetch data, prepending warmup bars so indicators never start with NaN.
    Returns (df_combined, n_main_bars).
    """
    df_main = _rate_limited_download(ticker, period, interval)
    if df_main is None:
        return None, 0

    n_main = len(df_main)
    if n_main >= min_bars:
        return df_main, n_main

    # Need more historical data
    warmup_period = WARMUP_MAP.get(interval, "1y")
    if warmup_period == period:
        return df_main, n_main

    df_warm = _rate_limited_download(ticker, warmup_period, interval)
    if df_warm is None or len(df_warm) <= n_main:
        return df_main, n_main

    # Locate where df_main starts inside df_warm
    try:
        start_loc = df_warm.index.get_indexer([df_main.index[0]], method="nearest")[0]
        prefix = df_warm.iloc[max(0, start_loc - min_bars): start_loc]
        combined = pd.concat([prefix, df_main])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined, n_main
    except Exception:
        return df_main, n_main

# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    """TradingView-accurate EMA: adjust=False, min_periods=1 (no NaN at start)."""
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()


def crossover_angle_deg(fast: pd.Series, slow: pd.Series, i: int) -> float:
    if i < 1:
        return 0.0
    df_fast = fast.iloc[i] - fast.iloc[i - 1]
    df_slow = slow.iloc[i] - slow.iloc[i - 1]
    return float(np.degrees(np.arctan(abs(df_fast - df_slow))))


def swing_highs_lows(df: pd.DataFrame, left: int = 5, right: int = 5
                     ) -> tuple[list, list]:
    """Return (swing_highs, swing_lows) as lists of (bar_index, datetime, price)."""
    n = len(df)
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values
    for i in range(left, n - right):
        is_sh = all(h[i] >= h[i - j] for j in range(1, left + 1)) and \
                all(h[i] >= h[i + j] for j in range(1, right + 1))
        is_sl = all(l[i] <= l[i - j] for j in range(1, left + 1)) and \
                all(l[i] <= l[i + j] for j in range(1, right + 1))
        if is_sh:
            highs.append((i, df.index[i], float(h[i])))
        if is_sl:
            lows.append((i, df.index[i], float(l[i])))
    return highs, lows


def nearest_support_resistance(df: pd.DataFrame, idx: int, window: int = 20
                               ) -> tuple[float, float]:
    start = max(0, idx - window)
    sub = df.iloc[start: idx + 1]
    return float(sub["Low"].min()), float(sub["High"].max())

# ══════════════════════════════════════════════════════════════════════════════
# ELLIOTT WAVE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _clean_pivots(raw: list[dict], min_wave_pct: float = 0.5) -> list[dict]:
    """Remove consecutive same-type pivots; keep more extreme one. Filter by %."""
    if not raw:
        return []
    cleaned: list[dict] = [raw[0]]
    for p in raw[1:]:
        last = cleaned[-1]
        if p["type"] == last["type"]:
            # keep more extreme
            if p["type"] == "H" and p["price"] > last["price"]:
                cleaned[-1] = p
            elif p["type"] == "L" and p["price"] < last["price"]:
                cleaned[-1] = p
        else:
            pct = abs(p["price"] - last["price"]) / max(last["price"], 1e-9) * 100
            if pct >= min_wave_pct:
                cleaned.append(p)
    return cleaned


def detect_elliott_waves(df: pd.DataFrame, min_wave_pct: float = 0.5,
                         left: int = 4, right: int = 4) -> dict:
    """
    Full Elliott Wave engine. Returns rich dict with:
      completed_waves, current_wave, next_target, fibonacci_levels,
      wave_labels, signal (BUY/SELL/NONE), entry, sl, target, pattern,
      wave_direction, wave_points.
    """
    default = {
        "pattern": "Insufficient Data",
        "wave_direction": None,
        "completed_waves": [],
        "current_wave": "Analyzing…",
        "wave_labels": [],
        "wave_points": {},
        "fibonacci_levels": {},
        "next_target": None,
        "signal": "NONE",
        "entry": None,
        "sl": None,
        "target": None,
        "pivots": [],
    }

    if len(df) < (left + right + 1) * 2:
        return default

    sh, sl_pts = swing_highs_lows(df, left, right)

    raw = []
    for idx, dt, price in sh:
        raw.append({"idx": idx, "dt": dt, "price": price, "type": "H"})
    for idx, dt, price in sl_pts:
        raw.append({"idx": idx, "dt": dt, "price": price, "type": "L"})
    raw.sort(key=lambda x: x["idx"])

    pivots = _clean_pivots(raw, min_wave_pct)
    default["pivots"] = pivots

    if len(pivots) < 4:
        return default

    # ── Try 5-wave impulse patterns on the last 9 pivots ─────────────────────
    for start in range(max(0, len(pivots) - 10), len(pivots) - 5):
        sub = pivots[start: start + 6]
        if len(sub) < 6:
            continue
        res = _check_impulse_up(sub)
        if res:
            res["pivots"] = pivots
            return res
        res = _check_impulse_down(sub)
        if res:
            res["pivots"] = pivots
            return res

    # ── Try in-progress waves ─────────────────────────────────────────────────
    for start in range(max(0, len(pivots) - 8), len(pivots) - 3):
        sub = pivots[start:]
        res = _check_inprogress(sub, df)
        if res:
            res["pivots"] = pivots
            return res

    # ── Try ABC corrective ────────────────────────────────────────────────────
    for start in range(max(0, len(pivots) - 6), len(pivots) - 3):
        sub = pivots[start:]
        res = _check_abc(sub)
        if res:
            res["pivots"] = pivots
            return res

    default["pattern"] = "No Clear Pattern Detected"
    return default


def _fib_levels(base: float, peak: float, direction: int = 1) -> dict:
    """Fib retracement levels from base to peak."""
    rng = peak - base
    return {
        "23.6%": round(peak - rng * 0.236, 2),
        "38.2%": round(peak - rng * 0.382, 2),
        "50.0%": round(peak - rng * 0.500, 2),
        "61.8%": round(peak - rng * 0.618, 2),
        "78.6%": round(peak - rng * 0.786, 2),
    }


def _check_impulse_up(sub: list[dict]) -> dict | None:
    types = [p["type"] for p in sub]
    if types != ["L", "H", "L", "H", "L", "H"]:
        return None
    p0, p1, p2, p3, p4, p5 = [p["price"] for p in sub]
    # Elliott rules
    if p2 <= p0: return None          # W2 cannot go below W0
    if p3 <= p1: return None          # W3 must exceed W1
    if p4 <= p1: return None          # W4 cannot enter W1 territory (simplified)
    w1 = p1 - p0; w3 = p3 - p2; w5 = p5 - p4
    if w3 < w1 and w3 < w5: return None  # W3 cannot be shortest

    fib = _fib_levels(p0, p5)
    w5_ext_1 = p4 + w1
    w5_ext_618 = p4 + w3 * 0.618

    return {
        "wave_direction": "Bullish",
        "pattern": "Completed 5-Wave Impulse (Bullish)",
        "completed_waves": ["Wave 1", "Wave 2", "Wave 3", "Wave 4", "Wave 5"],
        "current_wave": "Correction (A-B-C) Expected",
        "wave_points": {
            "Wave 0 (Base)": p0, "Wave 1 Top": p1,
            "Wave 2 Bottom": p2, "Wave 3 Top": p3,
            "Wave 4 Bottom": p4, "Wave 5 Top": p5,
        },
        "wave_labels": [
            {"label": "0", "idx": sub[0]["idx"], "price": p0, "dt": sub[0]["dt"]},
            {"label": "1", "idx": sub[1]["idx"], "price": p1, "dt": sub[1]["dt"]},
            {"label": "2", "idx": sub[2]["idx"], "price": p2, "dt": sub[2]["dt"]},
            {"label": "3", "idx": sub[3]["idx"], "price": p3, "dt": sub[3]["dt"]},
            {"label": "4", "idx": sub[4]["idx"], "price": p4, "dt": sub[4]["dt"]},
            {"label": "5", "idx": sub[5]["idx"], "price": p5, "dt": sub[5]["dt"]},
        ],
        "fibonacci_levels": {**fib,
            "Wave-A Target (38.2% retrace)": round(p5 - (p5-p0)*0.382, 2),
            "Wave-C Target (61.8% retrace)": round(p5 - (p5-p0)*0.618, 2),
        },
        "next_target": round(p5 - (p5 - p0) * 0.618, 2),
        "signal": "SELL",  # After 5-wave up, sell correction
        "entry": p5, "sl": round(p5 + w1 * 0.5, 2),
        "target": round(p5 - (p5 - p0) * 0.618, 2),
    }


def _check_impulse_down(sub: list[dict]) -> dict | None:
    types = [p["type"] for p in sub]
    if types != ["H", "L", "H", "L", "H", "L"]:
        return None
    p0, p1, p2, p3, p4, p5 = [p["price"] for p in sub]
    if p2 >= p0: return None
    if p3 >= p1: return None
    if p4 >= p1: return None
    w1 = p0 - p1; w3 = p2 - p3; w5 = p4 - p5
    if w3 < w1 and w3 < w5: return None

    fib = _fib_levels(p5, p0, -1)
    return {
        "wave_direction": "Bearish",
        "pattern": "Completed 5-Wave Impulse (Bearish)",
        "completed_waves": ["Wave 1", "Wave 2", "Wave 3", "Wave 4", "Wave 5"],
        "current_wave": "Correction (A-B-C) Expected (Upward)",
        "wave_points": {
            "Wave 0 (Top)": p0, "Wave 1 Bottom": p1,
            "Wave 2 Top": p2, "Wave 3 Bottom": p3,
            "Wave 4 Top": p4, "Wave 5 Bottom": p5,
        },
        "wave_labels": [
            {"label": "0", "idx": sub[0]["idx"], "price": p0, "dt": sub[0]["dt"]},
            {"label": "1", "idx": sub[1]["idx"], "price": p1, "dt": sub[1]["dt"]},
            {"label": "2", "idx": sub[2]["idx"], "price": p2, "dt": sub[2]["dt"]},
            {"label": "3", "idx": sub[3]["idx"], "price": p3, "dt": sub[3]["dt"]},
            {"label": "4", "idx": sub[4]["idx"], "price": p4, "dt": sub[4]["dt"]},
            {"label": "5", "idx": sub[5]["idx"], "price": p5, "dt": sub[5]["dt"]},
        ],
        "fibonacci_levels": {**fib,
            "Wave-A Target (38.2%)": round(p5 + (p0-p5)*0.382, 2),
            "Wave-C Target (61.8%)": round(p5 + (p0-p5)*0.618, 2),
        },
        "next_target": round(p5 + (p0 - p5) * 0.618, 2),
        "signal": "BUY",
        "entry": p5, "sl": round(p5 - w1 * 0.5, 2),
        "target": round(p5 + (p0 - p5) * 0.618, 2),
    }


def _check_inprogress(sub: list[dict], df: pd.DataFrame) -> dict | None:
    """Detect in-progress waves (waves 1+2 done, entering wave 3 etc.)."""
    if len(sub) < 3:
        return None
    types = [p["type"] for p in sub[:3]]
    last_close = float(df["Close"].iloc[-1])

    # Uptrend: 0=L, 1=H, 2=L → in wave 3 up
    if types == ["L", "H", "L"]:
        p0, p1, p2 = sub[0]["price"], sub[1]["price"], sub[2]["price"]
        if p2 <= p0:
            return None
        w1 = p1 - p0
        w3_1618 = p2 + w1 * 1.618
        w3_2618 = p2 + w1 * 2.618
        w4_support = p2 + w1 * 0.382
        w5_target  = p2 + w1 * 2.0

        # Are we currently in wave 3? Price should be above W2 and rising
        in_wave3 = last_close > p2

        labels = [
            {"label": "0", "idx": sub[0]["idx"], "price": p0, "dt": sub[0]["dt"]},
            {"label": "1", "idx": sub[1]["idx"], "price": p1, "dt": sub[1]["dt"]},
            {"label": "2", "idx": sub[2]["idx"], "price": p2, "dt": sub[2]["dt"]},
        ]
        if len(sub) >= 4 and sub[3]["type"] == "H":
            labels.append({"label": "3?", "idx": sub[3]["idx"], "price": sub[3]["price"], "dt": sub[3]["dt"]})

        completed = ["Wave 1", "Wave 2"]
        current_w = "Wave 3 In Progress (Buy Zone)" if in_wave3 else "Wave 2 Bottom (Buy Opportunity)"

        return {
            "wave_direction": "Bullish",
            "pattern": "In Progress – Waves 1 & 2 Complete",
            "completed_waves": completed,
            "current_wave": current_w,
            "wave_points": {
                "Wave 0 (Base)": p0, "Wave 1 Top": p1, "Wave 2 Bottom (Buy)": p2
            },
            "wave_labels": labels,
            "fibonacci_levels": {
                "Wave 3 Target (1.618×W1)": round(w3_1618, 2),
                "Wave 3 Extended (2.618×W1)": round(w3_2618, 2),
                "Wave 4 Support (38.2%)":   round(w4_support, 2),
                "Wave 5 Final Target":       round(w5_target, 2),
            },
            "next_target": round(w3_1618, 2),
            "signal": "BUY",
            "entry": p2,
            "sl": round(p0 - (p1 - p0) * 0.1, 2),
            "target": round(w3_1618, 2),
        }

    # Downtrend: 0=H, 1=L, 2=H → in wave 3 down
    if types == ["H", "L", "H"]:
        p0, p1, p2 = sub[0]["price"], sub[1]["price"], sub[2]["price"]
        if p2 >= p0:
            return None
        w1 = p0 - p1
        w3_1618 = p2 - w1 * 1.618
        w3_2618 = p2 - w1 * 2.618
        w4_resist = p2 - w1 * 0.382

        labels = [
            {"label": "0", "idx": sub[0]["idx"], "price": p0, "dt": sub[0]["dt"]},
            {"label": "1", "idx": sub[1]["idx"], "price": p1, "dt": sub[1]["dt"]},
            {"label": "2", "idx": sub[2]["idx"], "price": p2, "dt": sub[2]["dt"]},
        ]
        if len(sub) >= 4 and sub[3]["type"] == "L":
            labels.append({"label": "3?", "idx": sub[3]["idx"], "price": sub[3]["price"], "dt": sub[3]["dt"]})

        return {
            "wave_direction": "Bearish",
            "pattern": "In Progress – Waves 1 & 2 Complete (Bearish)",
            "completed_waves": ["Wave 1", "Wave 2"],
            "current_wave": "Wave 3 In Progress (Sell Zone)",
            "wave_points": {
                "Wave 0 (Top)": p0, "Wave 1 Bottom": p1, "Wave 2 Top (Sell)": p2
            },
            "wave_labels": labels,
            "fibonacci_levels": {
                "Wave 3 Target (1.618×W1)": round(w3_1618, 2),
                "Wave 3 Extended (2.618×W1)": round(w3_2618, 2),
                "Wave 4 Resistance (38.2%)":  round(w4_resist, 2),
            },
            "next_target": round(w3_1618, 2),
            "signal": "SELL",
            "entry": p2,
            "sl": round(p0 + (p0 - p1) * 0.1, 2),
            "target": round(w3_1618, 2),
        }

    return None


def _check_abc(sub: list[dict]) -> dict | None:
    if len(sub) < 3:
        return None
    # ABC up: L H L
    if sub[0]["type"] == "L" and sub[1]["type"] == "H" and sub[2]["type"] == "L":
        a_s, a_e, b_e = sub[0]["price"], sub[1]["price"], sub[2]["price"]
        wa = a_e - a_s
        wb = a_e - b_e
        if wa <= 0: return None
        if not (0.382 <= wb / wa <= 0.886): return None
        wc_eq  = b_e + wa
        wc_ext = b_e + wa * 1.618
        return {
            "wave_direction": "Bullish ABC",
            "pattern": "ABC Corrective (Bullish – Wave C Up)",
            "completed_waves": ["Wave A", "Wave B"],
            "current_wave": "Wave C In Progress (Upward)",
            "wave_points": {"A Start": a_s, "A End / B Start": a_e, "B End (Buy)": b_e},
            "wave_labels": [
                {"label": "A", "idx": sub[1]["idx"], "price": a_e, "dt": sub[1]["dt"]},
                {"label": "B", "idx": sub[2]["idx"], "price": b_e, "dt": sub[2]["dt"]},
            ],
            "fibonacci_levels": {
                "C = A Target": round(wc_eq, 2),
                "C = 1.618×A":  round(wc_ext, 2),
            },
            "next_target": round(wc_eq, 2),
            "signal": "BUY",
            "entry": b_e,
            "sl": round(a_s - abs(wa) * 0.1, 2),
            "target": round(wc_eq, 2),
        }
    # ABC down: H L H
    if sub[0]["type"] == "H" and sub[1]["type"] == "L" and sub[2]["type"] == "H":
        a_s, a_e, b_e = sub[0]["price"], sub[1]["price"], sub[2]["price"]
        wa = a_s - a_e
        wb = b_e - a_e
        if wa <= 0: return None
        if not (0.382 <= wb / wa <= 0.886): return None
        wc_eq  = b_e - wa
        wc_ext = b_e - wa * 1.618
        return {
            "wave_direction": "Bearish ABC",
            "pattern": "ABC Corrective (Bearish – Wave C Down)",
            "completed_waves": ["Wave A", "Wave B"],
            "current_wave": "Wave C In Progress (Downward)",
            "wave_points": {"A Start": a_s, "A End / B Start": a_e, "B End (Sell)": b_e},
            "wave_labels": [
                {"label": "A", "idx": sub[1]["idx"], "price": a_e, "dt": sub[1]["dt"]},
                {"label": "B", "idx": sub[2]["idx"], "price": b_e, "dt": sub[2]["dt"]},
            ],
            "fibonacci_levels": {
                "C = A Target": round(wc_eq, 2),
                "C = 1.618×A":  round(wc_ext, 2),
            },
            "next_target": round(wc_eq, 2),
            "signal": "SELL",
            "entry": b_e,
            "sl": round(a_s + abs(wa) * 0.1, 2),
            "target": round(wc_eq, 2),
        }
    return None

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, config: dict
                     ) -> tuple[pd.Series, pd.Series, dict]:
    """
    Returns (signals, signal_reasons, indicators_dict).
    signals: +1 BUY, -1 SELL, 0 HOLD
    For Simple Buy/Sell entry is immediate (not N+1).
    For EMA/Elliott/Anticipatory → signal at bar N, entry at bar N+1 open.
    """
    n       = len(df)
    signals = pd.Series(0, index=df.index, dtype=int)
    reasons = pd.Series("", index=df.index, dtype=str)
    strat   = config.get("strategy", "EMA Crossover")
    inds    = {}

    if strat == "Simple Buy":
        signals.iloc[0] = 1
        reasons.iloc[0] = "Simple Buy – Immediate Market Entry"
        return signals, reasons, inds

    if strat == "Simple Sell":
        signals.iloc[0] = -1
        reasons.iloc[0] = "Simple Sell – Immediate Market Entry"
        return signals, reasons, inds

    fast_p  = config.get("fast_ema", 9)
    slow_p  = config.get("slow_ema", 15)
    fast_e  = ema(df["Close"], fast_p)
    slow_e  = ema(df["Close"], slow_p)
    atr_s   = atr(df)
    inds["fast_ema"] = fast_e
    inds["slow_ema"] = slow_e
    inds["atr"]      = atr_s

    if strat == "EMA Crossover":
        min_angle    = config.get("min_angle", 0.0)
        cx_type      = config.get("crossover_type", "Simple Crossover")
        cx_candle_sz = config.get("custom_candle_size", 10)

        for i in range(1, n):
            f0, f1 = fast_e.iloc[i], fast_e.iloc[i - 1]
            s0, s1 = slow_e.iloc[i], slow_e.iloc[i - 1]
            candle_sz = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
            angle     = crossover_angle_deg(fast_e, slow_e, i)

            if angle < min_angle:
                continue
            if cx_type == "Custom Candle Size" and candle_sz < cx_candle_sz:
                continue
            if cx_type == "ATR Based Candle Size" and candle_sz < atr_s.iloc[i]:
                continue

            if f1 <= s1 and f0 > s0:
                signals.iloc[i] = 1
                reasons.iloc[i] = (f"EMA Bullish Crossover | Fast={f0:.2f} Slow={s0:.2f} "
                                   f"Δ={f0-s0:.2f} Angle={angle:.1f}°")
            elif f1 >= s1 and f0 < s0:
                signals.iloc[i] = -1
                reasons.iloc[i] = (f"EMA Bearish Crossover | Fast={f0:.2f} Slow={s0:.2f} "
                                   f"Δ={s0-f0:.2f} Angle={angle:.1f}°")

    elif strat == "Anticipatory EMA":
        # Predict imminent crossover: large acceleration towards crossover
        for i in range(3, n):
            f0, f1, f2 = fast_e.iloc[i], fast_e.iloc[i-1], fast_e.iloc[i-2]
            s0, s1, s2 = slow_e.iloc[i], slow_e.iloc[i-1], slow_e.iloc[i-2]
            gap_now  = f0 - s0
            gap_prev = f1 - s1
            gap2     = f2 - s2
            a_val    = float(atr_s.iloc[i])

            # Converging fast → slow from below (bullish anticipation)
            if gap2 < 0 and gap_prev < 0 and gap_now < 0:
                gap_shrink = abs(gap2) - abs(gap_now)
                if gap_shrink > 0 and abs(gap_now) < a_val * 1.2:
                    fast_accel = (f0 - f1) - (f1 - f2)
                    if fast_accel > 0:
                        signals.iloc[i] = 1
                        reasons.iloc[i] = (f"Anticipatory EMA BUY | Gap={gap_now:.2f} "
                                           f"Shrink={gap_shrink:.2f} ATR={a_val:.2f}")
                        continue

            # Converging fast → slow from above (bearish anticipation)
            if gap2 > 0 and gap_prev > 0 and gap_now > 0:
                gap_shrink = abs(gap2) - abs(gap_now)
                if gap_shrink > 0 and abs(gap_now) < a_val * 1.2:
                    fast_accel = (f0 - f1) - (f1 - f2)
                    if fast_accel < 0:
                        signals.iloc[i] = -1
                        reasons.iloc[i] = (f"Anticipatory EMA SELL | Gap={gap_now:.2f} "
                                           f"Shrink={gap_shrink:.2f} ATR={a_val:.2f}")

    elif strat == "Elliott Wave":
        ew = detect_elliott_waves(df,
                                  min_wave_pct=config.get("min_wave_pct", 0.5),
                                  left=config.get("ew_left", 4),
                                  right=config.get("ew_right", 4))
        inds["elliott"] = ew
        sig_ew = ew.get("signal", "NONE")
        if sig_ew in ("BUY", "SELL") and ew.get("wave_labels"):
            last_label = ew["wave_labels"][-1]
            anchor_idx = last_label.get("idx", n - 1)
            anchor_idx = min(anchor_idx, n - 1)
            sig_val = 1 if sig_ew == "BUY" else -1
            signals.iloc[anchor_idx] = sig_val
            reasons.iloc[anchor_idx] = f"Elliott Wave | {ew.get('pattern','')} | {ew.get('current_wave','')}"

    return signals, reasons, inds

# ══════════════════════════════════════════════════════════════════════════════
# SL / TARGET CALCULATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_sl(entry: float, sig: int, config: dict,
            df: pd.DataFrame, idx: int) -> float:
    sl_type = config.get("sl_type", "Custom Points")
    a_val   = float(atr(df).iloc[min(idx, len(df)-1)])
    direction = sig  # +1 buy, -1 sell

    if sl_type == "Custom Points":
        pts = config.get("sl_points", 10)
        return entry - pts if sig == 1 else entry + pts

    if sl_type == "ATR Based":
        mult = config.get("atr_sl_mult", 1.5)
        return entry - a_val * mult if sig == 1 else entry + a_val * mult

    if sl_type == "Risk Reward Based":
        rr  = config.get("risk_reward", 2.0)
        tpt = config.get("target_points", 20)
        pts = tpt / rr
        return entry - pts if sig == 1 else entry + pts

    if sl_type == "Volatility Based":
        vol = df["Close"].pct_change().std() * float(df["Close"].iloc[min(idx, len(df)-1)])
        mult = config.get("vol_sl_mult", 2.0)
        return entry - vol * mult if sig == 1 else entry + vol * mult

    if sl_type in ("Swing Low/High", "Support / Resistance"):
        win   = config.get("swing_window", 20)
        start = max(0, idx - win)
        if sig == 1:
            return float(df["Low"].iloc[start: idx + 1].min()) - a_val * 0.5
        else:
            return float(df["High"].iloc[start: idx + 1].max()) + a_val * 0.5

    if sl_type == "Candle Low/High":
        if sig == 1:
            return float(df["Low"].iloc[min(idx, len(df)-1)]) - a_val * 0.1
        else:
            return float(df["High"].iloc[min(idx, len(df)-1)]) + a_val * 0.1

    if sl_type == "Auto SL":
        win   = 5
        start = max(0, idx - win)
        if sig == 1:
            swing = float(df["Low"].iloc[start: idx + 1].min())
            return min(swing, entry - a_val * 1.5)
        else:
            swing = float(df["High"].iloc[start: idx + 1].max())
            return max(swing, entry + a_val * 1.5)

    # EMA Reverse Crossover / Trailing SL → use custom as initial
    pts = config.get("sl_points", 10)
    return entry - pts if sig == 1 else entry + pts


def calc_target(entry: float, sig: int, config: dict,
                df: pd.DataFrame, idx: int) -> float:
    tgt_type = config.get("target_type", "Custom Points")
    a_val    = float(atr(df).iloc[min(idx, len(df)-1)])

    if tgt_type == "Custom Points":
        pts = config.get("target_points", 20)
        return entry + pts if sig == 1 else entry - pts

    if tgt_type == "ATR Based":
        mult = config.get("atr_target_mult", 2.0)
        return entry + a_val * mult if sig == 1 else entry - a_val * mult

    if tgt_type == "Risk Reward Based":
        rr  = config.get("risk_reward", 2.0)
        pts = config.get("sl_points", 10)
        return entry + pts * rr if sig == 1 else entry - pts * rr

    if tgt_type == "Volatility Based":
        vol  = df["Close"].pct_change().std() * float(df["Close"].iloc[min(idx, len(df)-1)])
        mult = config.get("vol_target_mult", 3.0)
        return entry + vol * mult if sig == 1 else entry - vol * mult

    if tgt_type in ("Swing High/Low", "Support / Resistance"):
        win   = config.get("swing_window", 20)
        start = max(0, idx - win)
        if sig == 1:
            swing = float(df["High"].iloc[start: idx + 1].max())
            return max(swing, entry + a_val)
        else:
            swing = float(df["Low"].iloc[start: idx + 1].min())
            return min(swing, entry - a_val)

    if tgt_type == "Candle High/Low":
        if sig == 1:
            return float(df["High"].iloc[min(idx, len(df)-1)]) + a_val * 0.1
        else:
            return float(df["Low"].iloc[min(idx, len(df)-1)]) - a_val * 0.1

    if tgt_type == "Auto Target":
        win   = 10
        start = max(0, idx - win)
        if sig == 1:
            swing = float(df["High"].iloc[start: idx + 1].max())
            return max(entry + a_val * 2.5, swing)
        else:
            swing = float(df["Low"].iloc[start: idx + 1].min())
            return min(entry - a_val * 2.5, swing)

    # EMA Reverse Crossover / Trailing Target → use custom as display
    pts = config.get("target_points", 20)
    return entry + pts if sig == 1 else entry - pts


def update_trailing_sl(position: dict, ltp: float, config: dict,
                       df: pd.DataFrame, i: int) -> dict:
    """Update SL for trailing variants. Modifies position in place."""
    sl_type = config.get("sl_type", "Custom Points")
    sig     = position["signal_type"]

    if sl_type == "Trailing SL":
        trail = config.get("sl_points", 10)
        if sig == 1:
            new_sl = ltp - trail
            if new_sl > position["current_sl"]:
                position["current_sl"] = new_sl
        else:
            new_sl = ltp + trail
            if new_sl < position["current_sl"]:
                position["current_sl"] = new_sl

    elif sl_type == "Swing Low/High":
        win = config.get("swing_window", 20)
        a_v = float(atr(df).iloc[min(i, len(df)-1)])
        start = max(0, i - win)
        if sig == 1:
            swing = float(df["Low"].iloc[start: i + 1].min())
            new_sl = swing - a_v * 0.5
            if new_sl > position["current_sl"]:
                position["current_sl"] = new_sl
        else:
            swing = float(df["High"].iloc[start: i + 1].max())
            new_sl = swing + a_v * 0.5
            if new_sl < position["current_sl"]:
                position["current_sl"] = new_sl

    elif sl_type == "Candle Low/High":
        a_v = float(atr(df).iloc[min(i, len(df)-1)])
        if sig == 1:
            candle_sl = float(df["Low"].iloc[i]) - a_v * 0.1
            if candle_sl > position["current_sl"]:
                position["current_sl"] = candle_sl
        else:
            candle_sl = float(df["High"].iloc[i]) + a_v * 0.1
            if candle_sl < position["current_sl"]:
                position["current_sl"] = candle_sl

    return position

# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _build_trade(pos: dict, exit_row: pd.Series, exit_dt,
                 exit_price: float, exit_reason: str,
                 quantity: int, is_violation: bool) -> dict:
    sig = pos["signal_type"]
    pnl = ((exit_price - pos["entry_price"]) if sig == 1
           else (pos["entry_price"] - exit_price)) * quantity
    return {
        "Entry DateTime": str(pos["entry_dt"]),
        "Exit DateTime":  str(exit_dt),
        "Signal":         "BUY" if sig == 1 else "SELL",
        "Entry Price":    round(pos["entry_price"], 2),
        "Exit Price":     round(exit_price, 2),
        "SL":             round(pos["initial_sl"], 2),
        "Final SL":       round(pos["current_sl"], 2),
        "Target":         round(pos["target"], 2),
        "Candle High":    round(float(exit_row["High"]), 2),
        "Candle Low":     round(float(exit_row["Low"]),  2),
        "Entry Reason":   pos.get("signal_reason", ""),
        "Exit Reason":    exit_reason,
        "PnL":            round(pnl, 2),
        "Is Violation":   is_violation,
        "Mode":           "Backtest",
    }


def run_backtest(df_full: pd.DataFrame, config: dict,
                 progress_cb=None) -> tuple[list, list, dict]:
    """
    Conservative backtesting engine.
    BUY:  check SL vs candle Low first, then Target vs candle High.
    SELL: check SL vs candle High first, then Target vs candle Low.
    N+1 entry rule for indicator-based strategies.
    Returns (trades, violations, indicators).
    """
    strat     = config.get("strategy", "EMA Crossover")
    qty       = config.get("quantity", 1)
    immediate = strat in ("Simple Buy", "Simple Sell")
    tgt_type  = config.get("target_type", "Custom Points")

    signals, reasons, inds = generate_signals(df_full, config)

    fast_e = inds.get("fast_ema", pd.Series(dtype=float))
    slow_e = inds.get("slow_ema", pd.Series(dtype=float))
    atr_s  = inds.get("atr", atr(df_full))

    trades: list     = []
    violations: list = []
    position: dict | None = None
    n = len(df_full)

    i = 0
    while i < n:
        row = df_full.iloc[i]

        # ── MANAGE OPEN POSITION ─────────────────────────────────────────────
        if position is not None:
            position = update_trailing_sl(position, float(row["Close"]), config, df_full, i)

            # EMA reverse crossover SL / target exit
            if (config.get("sl_type") == "EMA Reverse Crossover" or
                    config.get("target_type") == "EMA Reverse Crossover"):
                if len(fast_e) > i and len(slow_e) > i and i > 0:
                    f0, f1 = fast_e.iloc[i], fast_e.iloc[i-1]
                    s0, s1 = slow_e.iloc[i], slow_e.iloc[i-1]
                    sig = position["signal_type"]
                    if sig == 1 and f1 >= s1 and f0 < s0:
                        ep = float(row["Open"])
                        t  = _build_trade(position, row, df_full.index[i],
                                          ep, "EMA Reverse Crossover Exit", qty, False)
                        trades.append(t)
                        position = None
                        i += 1
                        continue
                    elif sig == -1 and f1 <= s1 and f0 > s0:
                        ep = float(row["Open"])
                        t  = _build_trade(position, row, df_full.index[i],
                                          ep, "EMA Reverse Crossover Exit", qty, False)
                        trades.append(t)
                        position = None
                        i += 1
                        continue

            sl   = position["current_sl"]
            tgt  = position["target"]
            sig  = position["signal_type"]
            c_hi = float(row["High"])
            c_lo = float(row["Low"])

            exit_price  = None
            exit_reason = None
            is_viol     = False

            if sig == 1:  # BUY → SL=Low, Target=High  (conservative: check SL first)
                sl_hit  = c_lo <= sl
                tgt_hit = c_hi >= tgt

                if sl_hit and tgt_hit:
                    # Both hit same candle → violation, conservative = SL first
                    exit_price  = sl
                    exit_reason = "SL Hit first (Violation: SL & Target same candle)"
                    is_viol     = True
                elif sl_hit:
                    exit_price  = sl
                    exit_reason = "SL Hit"
                elif tgt_hit:
                    if tgt_type != "Trailing Target (Display Only)":
                        exit_price  = tgt
                        exit_reason = "Target Hit"

            else:  # SELL → SL=High, Target=Low  (conservative: check SL first)
                sl_hit  = c_hi >= sl
                tgt_hit = c_lo <= tgt

                if sl_hit and tgt_hit:
                    exit_price  = sl
                    exit_reason = "SL Hit first (Violation: SL & Target same candle)"
                    is_viol     = True
                elif sl_hit:
                    exit_price  = sl
                    exit_reason = "SL Hit"
                elif tgt_hit:
                    if tgt_type != "Trailing Target (Display Only)":
                        exit_price  = tgt
                        exit_reason = "Target Hit"

            if exit_price is not None:
                t = _build_trade(position, row, df_full.index[i],
                                 exit_price, exit_reason, qty, is_viol)
                trades.append(t)
                if is_viol:
                    violations.append(t)
                position = None

        # ── CHECK NEW SIGNAL ─────────────────────────────────────────────────
        if position is None and signals.iloc[i] != 0:
            sig_val = int(signals.iloc[i])
            reason  = str(reasons.iloc[i])

            if immediate:
                entry_idx   = i
                entry_dt    = df_full.index[i]
                entry_price = float(df_full.iloc[i]["Open"])
            else:
                # N+1 entry rule
                if i + 1 >= n:
                    i += 1
                    continue
                entry_idx   = i + 1
                entry_dt    = df_full.index[entry_idx]
                entry_price = float(df_full.iloc[entry_idx]["Open"])

            sl_price  = calc_sl(entry_price, sig_val, config, df_full, entry_idx)
            tgt_price = calc_target(entry_price, sig_val, config, df_full, entry_idx)

            # Elliott wave overrides target/SL
            if strat == "Elliott Wave" and "elliott" in inds:
                ew = inds["elliott"]
                if ew.get("target") is not None:
                    tgt_price = float(ew["target"])
                if ew.get("sl") is not None:
                    sl_price = float(ew["sl"])

            position = {
                "signal_idx":   i,
                "entry_idx":    entry_idx,
                "entry_dt":     entry_dt,
                "entry_price":  entry_price,
                "signal_type":  sig_val,
                "initial_sl":   sl_price,
                "current_sl":   sl_price,
                "target":       tgt_price,
                "signal_reason": reason,
            }

            if not immediate:
                i = entry_idx   # jump to entry bar

        if progress_cb:
            progress_cb(i / n)
        i += 1

    # Close any open position at end of data
    if position is not None:
        last_row = df_full.iloc[-1]
        ep  = float(last_row["Close"])
        t   = _build_trade(position, last_row, df_full.index[-1],
                            ep, "End of Data (Force Close)", qty, False)
        trades.append(t)

    return trades, violations, inds


def calc_accuracy(trades: list) -> tuple[int, int, float]:
    if not trades:
        return 0, 0, 0.0
    wins = sum(1 for t in trades if t["PnL"] > 0)
    return wins, len(trades), round(wins / len(trades) * 100, 2)

# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

_DARK_BG   = "rgba(13,17,23,1)"
_GRID_CLR  = "rgba(255,255,255,0.07)"
_GREEN     = "#00e676"
_RED       = "#ff1744"
_ORANGE    = "#ff9800"
_BLUE      = "#40c4ff"
_YELLOW    = "#ffea00"
_PURPLE    = "#ce93d8"


def _base_layout(title: str, height: int = 700) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="white", size=14)),
        height=height,
        template="plotly_dark",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color="white", family="JetBrains Mono, monospace"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.15)",
                    borderwidth=1),
    )


def build_backtest_chart(df: pd.DataFrame, trades: list,
                         inds: dict, config: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22],
                        vertical_spacing=0.02,
                        subplot_titles=("", "Volume"))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing=dict(line=dict(color=_GREEN), fillcolor="rgba(0,230,118,0.2)"),
        decreasing=dict(line=dict(color=_RED),   fillcolor="rgba(255,23,68,0.2)"),
    ), row=1, col=1)

    # EMA lines
    if "fast_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=inds["fast_ema"],
            name=f"EMA {config.get('fast_ema',9)}",
            line=dict(color=_ORANGE, width=1.5)), row=1, col=1)
    if "slow_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=inds["slow_ema"],
            name=f"EMA {config.get('slow_ema',15)}",
            line=dict(color=_BLUE, width=1.5)), row=1, col=1)

    # Elliott wave labels
    if "elliott" in inds and inds["elliott"]:
        ew = inds["elliott"]
        for wl in ew.get("wave_labels", []):
            try:
                fig.add_annotation(x=wl["dt"], y=wl["price"],
                    text=wl["label"], showarrow=True, arrowhead=2,
                    arrowcolor=_YELLOW, font=dict(color=_YELLOW, size=11),
                    bgcolor="rgba(0,0,0,0.6)", row=1, col=1)
            except Exception:
                pass

    # Trade markers
    buy_x, buy_y   = [], []
    sell_x, sell_y = [], []
    ex_x, ex_y, ex_clr, ex_txt = [], [], [], []

    for t in trades:
        try:
            edt = pd.to_datetime(t["Entry DateTime"])
            xdt = pd.to_datetime(t["Exit DateTime"])
        except Exception:
            continue
        if t["Signal"] == "BUY":
            buy_x.append(edt); buy_y.append(t["Entry Price"])
        else:
            sell_x.append(edt); sell_y.append(t["Entry Price"])
        clr = "rgba(0,230,118,0.9)" if t["PnL"] >= 0 else "rgba(255,23,68,0.9)"
        ex_x.append(xdt); ex_y.append(t["Exit Price"])
        ex_clr.append(clr)
        ex_txt.append(f"PnL: {t['PnL']:+.2f}<br>{t['Exit Reason']}")

    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="Buy Entry",
            marker=dict(symbol="triangle-up", size=13, color=_GREEN,
                        line=dict(color="white", width=0.5))), row=1, col=1)
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="Sell Entry",
            marker=dict(symbol="triangle-down", size=13, color=_RED,
                        line=dict(color="white", width=0.5))), row=1, col=1)
    if ex_x:
        fig.add_trace(go.Scatter(x=ex_x, y=ex_y, mode="markers", name="Exit",
            marker=dict(symbol="x-thin-open", size=11, color=ex_clr,
                        line=dict(width=2)),
            text=ex_txt, hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    # SL / Target horizontal lines per trade (last 10 to avoid clutter)
    for t in trades[-10:]:
        try:
            e_dt = pd.to_datetime(t["Entry DateTime"])
            x_dt = pd.to_datetime(t["Exit DateTime"])
            fig.add_shape(type="line", x0=e_dt, x1=x_dt, y0=t["SL"], y1=t["SL"],
                line=dict(color=_RED, width=1, dash="dot"), row=1, col=1)
            fig.add_shape(type="line", x0=e_dt, x1=x_dt, y0=t["Target"], y1=t["Target"],
                line=dict(color=_GREEN, width=1, dash="dot"), row=1, col=1)
        except Exception:
            pass

    # Volume bars
    vol_clr = [_GREEN if c >= o else _RED
                for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_clr, opacity=0.5), row=2, col=1)

    layout = _base_layout("Backtest Results")
    layout["xaxis2"] = dict(showgrid=True, gridcolor=_GRID_CLR)
    layout["yaxis"]  = dict(showgrid=True, gridcolor=_GRID_CLR)
    layout["yaxis2"] = dict(showgrid=True, gridcolor=_GRID_CLR)
    fig.update_layout(**layout)
    return fig


def build_live_chart(df: pd.DataFrame, position: dict | None,
                     inds: dict, config: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22],
                        vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color=_GREEN), fillcolor="rgba(0,230,118,0.15)"),
        decreasing=dict(line=dict(color=_RED),   fillcolor="rgba(255,23,68,0.15)"),
    ), row=1, col=1)

    if "fast_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=inds["fast_ema"],
            name=f"EMA {config.get('fast_ema',9)}",
            line=dict(color=_ORANGE, width=1.5)), row=1, col=1)
    if "slow_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=inds["slow_ema"],
            name=f"EMA {config.get('slow_ema',15)}",
            line=dict(color=_BLUE, width=1.5)), row=1, col=1)

    # Elliott Wave labels
    if "elliott" in inds and inds["elliott"]:
        ew = inds["elliott"]
        for wl in ew.get("wave_labels", []):
            try:
                fig.add_annotation(x=wl["dt"], y=wl["price"],
                    text=wl["label"], showarrow=True, arrowhead=2,
                    arrowcolor=_YELLOW, font=dict(color=_YELLOW, size=10),
                    bgcolor="rgba(0,0,0,0.6)", row=1, col=1)
            except Exception:
                pass

    if position is not None:
        ep  = position["entry_price"]
        sl  = position["current_sl"]
        tgt = position["target"]
        fig.add_hline(y=ep,  line=dict(color="white", width=1.5, dash="dash"),
                      annotation_text=f"Entry {ep:.2f}", annotation_position="right",
                      row=1, col=1)
        fig.add_hline(y=sl,  line=dict(color=_RED, width=1.5, dash="dash"),
                      annotation_text=f"SL {sl:.2f}", annotation_position="right",
                      row=1, col=1)
        fig.add_hline(y=tgt, line=dict(color=_GREEN, width=1.5, dash="dash"),
                      annotation_text=f"Tgt {tgt:.2f}", annotation_position="right",
                      row=1, col=1)

    vol_clr = [_GREEN if c >= o else _RED
                for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_clr, opacity=0.5), row=2, col=1)

    layout = _base_layout("Live Trading", 580)
    layout["yaxis"]  = dict(showgrid=True, gridcolor=_GRID_CLR)
    layout["yaxis2"] = dict(showgrid=True, gridcolor=_GRID_CLR)
    fig.update_layout(**layout)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# DHAN BROKER
# ══════════════════════════════════════════════════════════════════════════════

def _dhan_client(cfg: dict):
    """Return dhanhq client or None."""
    try:
        from dhanhq import dhanhq
        return dhanhq(cfg["client_id"], cfg["access_token"])
    except Exception:
        return None


def place_equity_order(cfg: dict, sig: int, ltp: float = 0.0) -> dict:
    dhan = _dhan_client(cfg)
    if dhan is None:
        return {"error": "dhanhq not installed or bad credentials"}
    try:
        tx  = "BUY" if sig == 1 else "SELL"
        seg = {"NSE": "NSE_EQ", "BSE": "BSE_EQ"}.get(cfg.get("exchange","NSE"), "NSE_EQ")
        ot  = cfg.get("entry_order_type", "MARKET")
        p   = float(ltp) if ot == "LIMIT" else 0
        return dhan.place_order(
            transactionType=tx, exchangeSegment=seg,
            productType=cfg.get("product_type","INTRADAY"),
            orderType=ot, validity="DAY",
            securityId=str(cfg.get("security_id","1594")),
            quantity=int(cfg.get("quantity",1)),
            price=p, triggerPrice=0)
    except Exception as e:
        return {"error": str(e)}


def place_options_order(cfg: dict, sig: int, ltp: float = 0.0) -> dict:
    dhan = _dhan_client(cfg)
    if dhan is None:
        return {"error": "dhanhq not installed or bad credentials"}
    try:
        sec = str(cfg.get("ce_security_id","57749")) if sig == 1 \
              else str(cfg.get("pe_security_id","57716"))
        ot  = cfg.get("options_entry_order_type", "MARKET")
        p   = float(ltp) if ot == "LIMIT" else 0
        return dhan.place_order(
            transactionType="BUY",
            exchangeSegment=cfg.get("options_exchange","NSE_FNO"),
            productType="INTRADAY", orderType=ot, validity="DAY",
            securityId=sec, quantity=int(cfg.get("options_quantity",65)),
            price=p, triggerPrice=0)
    except Exception as e:
        return {"error": str(e)}


def place_exit_order(cfg: dict, pos: dict, ltp: float = 0.0,
                     is_options: bool = False) -> dict:
    dhan = _dhan_client(cfg)
    if dhan is None:
        return {"error": "dhanhq not installed"}
    try:
        sig = pos.get("signal_type", 1)
        ot  = cfg.get("exit_order_type", "MARKET")
        p   = float(ltp) if ot == "LIMIT" else 0
        if is_options:
            sec = str(cfg.get("ce_security_id","57749")) if sig == 1 \
                  else str(cfg.get("pe_security_id","57716"))
            return dhan.place_order(
                transactionType="SELL",
                exchangeSegment=cfg.get("options_exchange","NSE_FNO"),
                productType="INTRADAY", orderType=ot, validity="DAY",
                securityId=sec, quantity=int(cfg.get("options_quantity",65)),
                price=p, triggerPrice=0)
        else:
            tx  = "SELL" if sig == 1 else "BUY"
            seg = {"NSE": "NSE_EQ", "BSE": "BSE_EQ"}.get(cfg.get("exchange","NSE"), "NSE_EQ")
            return dhan.place_order(
                transactionType=tx, exchangeSegment=seg,
                productType=cfg.get("product_type","INTRADAY"),
                orderType=ot, validity="DAY",
                securityId=str(cfg.get("security_id","1594")),
                quantity=int(cfg.get("quantity",1)),
                price=p, triggerPrice=0)
    except Exception as e:
        return {"error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# LIVE TRADING THREAD
# ══════════════════════════════════════════════════════════════════════════════

def _tf_minutes(interval: str) -> int:
    return {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"1d":1440,"1wk":10080}.get(interval,5)


def _bar_complete(interval: str) -> bool:
    """True if current clock time is at a completed bar boundary (±3 s)."""
    mins = _tf_minutes(interval)
    if mins >= 1440:
        return True
    now  = datetime.datetime.now()
    tot  = now.hour * 60 + now.minute
    return (tot % mins == 0) and (now.second < 5)


def live_thread_fn(ticker: str, period: str, interval: str,
                   config: dict, stop_event: threading.Event,
                   dhan_cfg: dict):
    """Background thread: fetch → signal → manage position → place orders."""
    _ts_set("live_running", True)
    _ts_set("current_position", None)
    _ts_set("live_log", [])
    _ts_set("live_data", None)
    _ts_set("current_pnl", 0.0)

    def log(msg: str):
        ist = datetime.datetime.now().strftime("%H:%M:%S")
        _ts_append("live_log", f"[{ist}] {msg}")

    log(f"▶ Started | {ticker} | {interval}/{period} | {config.get('strategy')}")
    log(f"  SL={config.get('sl_type')} | Target={config.get('target_type')}")

    cooldown    = config.get("cooldown", 5)
    use_cool    = config.get("enable_cooldown", True)
    last_exit   = 0.0
    last_bar_dt = None
    cached_df: pd.DataFrame | None = None
    last_fetch  = 0.0
    global _LAST_YF_CALL

    while True:
        # Exit loop ONLY when: stop requested AND no open position
        stop_requested = stop_event.is_set()
        has_position   = _ts_get("current_position") is not None
        if stop_requested and not has_position:
            break

        try:
            # ── Fetch data ─────────────────────────────────────────────────
            now_t = time.time()
            if now_t - last_fetch >= 3.0:
                with _YF_LOCK:
                    gap = time.time() - _LAST_YF_CALL
                    if gap < 1.5:
                        time.sleep(1.5 - gap)
                    _LAST_YF_CALL = time.time()
                try:
                    df = yf.download(ticker, period=period, interval=interval,
                                     auto_adjust=True, progress=False, threads=False)
                    if df is not None and len(df) > 1:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df = df.dropna(how="all")
                        cached_df = df
                        last_fetch = time.time()
                        lc = df.iloc[-1]
                        _ts_set("last_candle", {
                            "datetime": str(df.index[-1]),
                            "open":  round(float(lc["Open"]),  2),
                            "high":  round(float(lc["High"]),  2),
                            "low":   round(float(lc["Low"]),   2),
                            "close": round(float(lc["Close"]), 2),
                            "volume": int(lc["Volume"]) if not np.isnan(lc["Volume"]) else 0,
                        })
                        _ts_set("live_data", {
                            "open":  df["Open"].tolist(),
                            "high":  df["High"].tolist(),
                            "low":   df["Low"].tolist(),
                            "close": df["Close"].tolist(),
                            "volume":df["Volume"].tolist(),
                            "index": [str(x) for x in df.index],
                        })
                except Exception as e:
                    log(f"[FETCH ERROR] {e}")
                    stop_event.wait(2.0)
                    continue

            if cached_df is None or len(cached_df) < 3:
                stop_event.wait(1.5)
                continue

            df  = cached_df
            ltp = float(df["Close"].iloc[-1])

            # ── Manage open position ───────────────────────────────────────
            pos = _ts_get("current_position")
            if pos is not None:
                # Update trailing SL using LTP
                sl_type = config.get("sl_type","Custom Points")
                sig     = pos["signal_type"]

                if sl_type == "Trailing SL":
                    trail = config.get("sl_points", 10)
                    if sig == 1:
                        new_sl = ltp - trail
                        if new_sl > pos["current_sl"]:
                            pos["current_sl"] = new_sl
                            _ts_set("current_position", pos)
                    else:
                        new_sl = ltp + trail
                        if new_sl < pos["current_sl"]:
                            pos["current_sl"] = new_sl
                            _ts_set("current_position", pos)

                # Update PnL display
                ep  = pos["entry_price"]
                pnl = (ltp - ep) if sig == 1 else (ep - ltp)
                _ts_set("current_pnl", round(pnl, 2))

                sl  = pos["current_sl"]
                tgt = pos["target"]
                tgt_type = config.get("target_type","Custom Points")

                exit_price  = None
                exit_reason = None

                # EMA reverse crossover check
                if sl_type == "EMA Reverse Crossover" or tgt_type == "EMA Reverse Crossover":
                    fe  = ema(df["Close"], config.get("fast_ema",9))
                    se  = ema(df["Close"], config.get("slow_ema",15))
                    if len(fe) >= 2:
                        f0, f1 = float(fe.iloc[-1]), float(fe.iloc[-2])
                        s0, s1 = float(se.iloc[-1]), float(se.iloc[-2])
                        if sig == 1 and f1 >= s1 and f0 < s0:
                            exit_price  = ltp
                            exit_reason = "EMA Reverse Crossover Exit"
                        elif sig == -1 and f1 <= s1 and f0 > s0:
                            exit_price  = ltp
                            exit_reason = "EMA Reverse Crossover Exit"

                # SL/Target vs LTP (tick-level)
                if exit_price is None:
                    if sig == 1:
                        if ltp <= sl:
                            exit_price  = ltp; exit_reason = "SL Hit"
                        elif tgt_type != "Trailing Target (Display Only)" and ltp >= tgt:
                            exit_price  = ltp; exit_reason = "Target Hit"
                    else:
                        if ltp >= sl:
                            exit_price  = ltp; exit_reason = "SL Hit"
                        elif tgt_type != "Trailing Target (Display Only)" and ltp <= tgt:
                            exit_price  = ltp; exit_reason = "Target Hit"

                if exit_price is not None:
                    pnl_val = (exit_price - ep) if sig == 1 else (ep - exit_price)
                    qty     = config.get("quantity", 1)
                    log(f"◼ EXIT {exit_reason} | EP={ep:.2f} XP={exit_price:.2f} PnL={pnl_val*qty:+.2f}")

                    if dhan_cfg.get("enabled"):
                        resp = place_exit_order(dhan_cfg, pos, ltp,
                                                dhan_cfg.get("options_trading",False))
                        log(f"  Dhan exit: {resp}")

                    trade_rec = {
                        "Entry DateTime": pos["entry_dt"],
                        "Exit DateTime":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Signal":         "BUY" if sig == 1 else "SELL",
                        "Entry Price":    round(ep, 2),
                        "Exit Price":     round(exit_price, 2),
                        "SL":             round(pos["initial_sl"], 2),
                        "Target":         round(tgt, 2),
                        "Entry Reason":   pos.get("signal_reason",""),
                        "Exit Reason":    exit_reason,
                        "PnL":            round(pnl_val * qty, 2),
                        "Mode":           "Live",
                    }
                    hist = _ts_get("trade_history", [])
                    hist.append(trade_rec)
                    _ts_set("trade_history", hist)
                    _ts_set("current_position", None)
                    _ts_set("current_pnl", 0.0)
                    last_exit = time.time()

            # ── Check for new signal (skip if stop requested) ─────────────
            pos = _ts_get("current_position")
            if pos is None and stop_event.is_set():
                stop_event.wait(1.5)
                continue

            if pos is None:
                if use_cool and (time.time() - last_exit) < cooldown:
                    stop_event.wait(0.5)
                    continue

                strat = config.get("strategy","EMA Crossover")
                bar_ok = True
                if strat in ("EMA Crossover","Anticipatory EMA","Elliott Wave"):
                    bar_ok = _bar_complete(interval)

                if not bar_ok:
                    stop_event.wait(0.5)
                    continue

                # Don't re-check the same completed bar
                cur_bar = str(df.index[-2]) if len(df) >= 2 else str(df.index[-1])
                if cur_bar == last_bar_dt and strat not in ("Simple Buy","Simple Sell"):
                    stop_event.wait(0.5)
                    continue

                # Signal on last COMPLETED bar (df[-2]), entry at current ltp
                sigs, reas, live_inds = generate_signals(df, config)

                if strat in ("Simple Buy","Simple Sell"):
                    sig_idx = len(sigs) - 1
                else:
                    sig_idx = len(sigs) - 2

                sig_val = int(sigs.iloc[sig_idx])

                # Store Elliott state always
                if "elliott" in live_inds:
                    _ts_set("elliott_wave_state", live_inds["elliott"])

                if sig_val != 0:
                    last_bar_dt = cur_bar
                    entry_price = ltp
                    sl_price    = calc_sl(entry_price, sig_val, config, df, len(df)-1)
                    tgt_price   = calc_target(entry_price, sig_val, config, df, len(df)-1)

                    if strat == "Elliott Wave" and "elliott" in live_inds:
                        ew = live_inds["elliott"]
                        if ew.get("sl"):    sl_price  = float(ew["sl"])
                        if ew.get("target"):tgt_price = float(ew["target"])

                    reason = str(reas.iloc[sig_idx])
                    log(f"◆ {'BUY' if sig_val==1 else 'SELL'} Signal | Entry={entry_price:.2f} "
                        f"SL={sl_price:.2f} Tgt={tgt_price:.2f}")
                    log(f"  {reason}")

                    new_pos = {
                        "entry_dt":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price":  entry_price,
                        "signal_type":  sig_val,
                        "initial_sl":   sl_price,
                        "current_sl":   sl_price,
                        "target":       tgt_price,
                        "signal_reason":reason,
                    }
                    _ts_set("current_position", new_pos)

                    if dhan_cfg.get("enabled"):
                        if dhan_cfg.get("options_trading"):
                            resp = place_options_order(dhan_cfg, sig_val, ltp)
                        else:
                            resp = place_equity_order(dhan_cfg, sig_val, ltp)
                        log(f"  Dhan entry: {resp}")

        except Exception as e:
            log(f"[ERROR] {e}")

        stop_event.wait(1.5)

    _ts_set("live_running", False)
    log("⏹ Live trading stopped.")

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_optimization(df: pd.DataFrame, base_cfg: dict,
                     target_acc: float = 0.0,
                     progress_ph=None) -> pd.DataFrame:
    rows = []
    total_runs = len(range(5,22,2)) * len(range(10,32,2)) * len([5,10,15,20,25]) * len([10,20,30,50])
    done = 0

    for fast in range(5, 22, 2):
        for slow in range(10, 32, 2):
            if fast >= slow:
                done += len([5,10,15,20,25]) * len([10,20,30,50])
                continue
            for sl_pts in [5, 10, 15, 20, 25]:
                for tgt_pts in [10, 20, 30, 50]:
                    cfg = {**base_cfg,
                           "fast_ema": fast, "slow_ema": slow,
                           "sl_points": sl_pts, "target_points": tgt_pts}
                    try:
                        trades, _, _ = run_backtest(df, cfg)
                        if len(trades) >= 3:
                            wins, n_t, acc = calc_accuracy(trades)
                            pnl = sum(t["PnL"] for t in trades)
                            rows.append({
                                "Fast EMA": fast, "Slow EMA": slow,
                                "SL Pts": sl_pts, "Tgt Pts": tgt_pts,
                                "Trades": n_t, "Wins": wins,
                                "Accuracy %": acc, "Total PnL": round(pnl, 2),
                                "Avg PnL": round(pnl / n_t, 2) if n_t else 0,
                            })
                    except Exception:
                        pass
                    done += 1
                    if progress_ph is not None and done % 20 == 0:
                        progress_ph.progress(min(done / total_runs, 1.0))

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("Accuracy %", ascending=False).reset_index(drop=True)
    if target_acc > 0:
        filtered = result[result["Accuracy %"] >= target_acc]
        return filtered if not filtered.empty else result
    return result

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:       #0d1117;
  --surface:  #161b22;
  --surface2: #1f2733;
  --border:   rgba(255,255,255,0.08);
  --green:    #00e676;
  --red:      #ff1744;
  --orange:   #ff9800;
  --blue:     #40c4ff;
  --yellow:   #ffea00;
  --text:     #e6edf3;
  --muted:    #8b949e;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
}

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4, h5 {
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}

.metric-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
  text-align: center;
  font-family: 'Space Mono', monospace;
}
.metric-card .label  { font-size:11px; color: var(--muted); text-transform:uppercase; letter-spacing:1px; }
.metric-card .value  { font-size:24px; font-weight:700; margin-top:4px; }
.metric-card .green  { color: var(--green); }
.metric-card .red    { color: var(--red);   }
.metric-card .orange { color: var(--orange);}
.metric-card .blue   { color: var(--blue);  }

.log-box {
  background: #0a0e13;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  height: 220px;
  overflow-y: auto;
  font-family: 'Space Mono', monospace;
  font-size: 11px;
  color: #a0aab4;
  line-height: 1.7;
}

.wave-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  margin: 6px 0;
}
.wave-card h4 { margin:0 0 8px; font-family:'Syne',sans-serif; font-size:13px; color:var(--yellow); }
.wave-card p  { margin:2px 0; font-size:11px; color:var(--muted); }

.position-card {
  background: linear-gradient(135deg, rgba(0,230,118,0.07), rgba(64,196,255,0.07));
  border: 1px solid rgba(0,230,118,0.25);
  border-radius: 12px;
  padding: 18px 22px;
}

.violation-badge {
  background: rgba(255,23,68,0.15);
  border: 1px solid rgba(255,23,68,0.4);
  border-radius: 6px;
  padding: 8px 14px;
  font-size: 13px;
  color: #ff1744;
  font-family: 'Space Mono', monospace;
}

.config-display {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 12px;
  margin-bottom: 12px;
}

.stButton>button {
  font-family: 'Space Mono', monospace !important;
  font-weight: 700;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 14px !important;
  font-weight: 600;
}

[data-testid="metric-container"] {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px !important;
}

.header-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 18px;
}
.header-bar .logo {
  font-family: 'Syne', sans-serif;
  font-size: 22px;
  font-weight: 800;
  color: var(--green);
}
.header-bar .version {
  font-size: 10px;
  color: var(--muted);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 6px;
}
.live-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
             background:var(--red); animation:blink 1s step-end infinite; }
@keyframes blink { 50% { opacity:0; } }
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# HELPER UI
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value, color: str = "blue") -> str:
    return (f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value {color}">{value}</div>'
            f'</div>')


def _fmt_pnl(v: float) -> str:
    return f"{'↑' if v>=0 else '↓'} {abs(v):,.2f}"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Smart Investing",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_ss()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 16px;">
          <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
                      color:#00e676;letter-spacing:-1px;">📈 Smart Investing</div>
          <div style="font-size:10px;color:#8b949e;margin-top:4px;">
            Algorithmic Trading Platform v2.0
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Instrument")
        ticker_name = st.selectbox("Ticker", list(TICKERS.keys()), key="_tk")
        if ticker_name == "Custom":
            custom_sym  = st.text_input("Custom Symbol", "RELIANCE.NS", key="_cust")
            ticker_sym  = custom_sym
        else:
            ticker_sym  = TICKERS[ticker_name]

        interval = st.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()),
                                 index=1, key="_iv")
        period   = st.selectbox("Period", TIMEFRAME_PERIODS[interval],
                                 index=min(1, len(TIMEFRAME_PERIODS[interval])-1),
                                 key="_pd")

        st.markdown("---")
        st.markdown("### 📐 Strategy")
        strategy  = st.selectbox("Strategy", STRATEGIES, key="_strat")

        fast_ema = slow_ema = 9
        min_angle = 0.0
        crossover_type   = "Simple Crossover"
        custom_candle_sz = 10
        min_wave_pct     = 0.5

        if strategy in ("EMA Crossover","Anticipatory EMA","Elliott Wave"):
            c1, c2 = st.columns(2)
            fast_ema = c1.number_input("Fast EMA", 2, 50, 9, key="_fe")
            slow_ema = c2.number_input("Slow EMA", 3, 200, 15, key="_se")

        if strategy in ("EMA Crossover","Anticipatory EMA"):
            use_angle = st.checkbox("Min Crossover Angle Filter", False, key="_ua")
            if use_angle:
                min_angle = st.number_input("Min Angle (°)", 0.0, 89.0, 0.0, 1.0, key="_ma")
            crossover_type = st.selectbox("Crossover Type", CROSSOVER_TYPES, key="_cxt")
            if crossover_type == "Custom Candle Size":
                custom_candle_sz = st.number_input("Min Candle Size (pts)", 1, 1000, 10, key="_ccs")

        if strategy == "Elliott Wave":
            min_wave_pct = st.number_input("Min Wave %", 0.1, 10.0, 0.5, 0.1, key="_mwp")

        st.markdown("---")
        st.markdown("### 🛡️ Stop Loss")
        sl_type   = st.selectbox("SL Type", SL_TYPES, key="_slt")
        sl_points = st.number_input("SL Points", 1, 10000, 10, key="_slp")

        atr_sl_mult  = 1.5
        rr_ratio     = 2.0
        swing_window = 20
        vol_sl_mult  = 2.0

        if "ATR" in sl_type:
            atr_sl_mult = st.number_input("ATR SL Multiplier", 0.5, 10.0, 1.5, 0.1, key="_asm")
        if "Risk Reward" in sl_type:
            rr_ratio = st.number_input("Risk:Reward", 0.5, 20.0, 2.0, 0.5, key="_rr")
        if "Swing" in sl_type or "Support" in sl_type:
            swing_window = st.number_input("Swing Lookback (bars)", 5, 100, 20, key="_sw")
        if "Volatility" in sl_type:
            vol_sl_mult = st.number_input("Vol SL Multiplier", 0.5, 10.0, 2.0, 0.1, key="_vsm")

        st.markdown("---")
        st.markdown("### 🎯 Target")
        target_type   = st.selectbox("Target Type", TARGET_TYPES, key="_tgt")
        target_points = st.number_input("Target Points", 1, 10000, 20, key="_tgtp")

        atr_target_mult = 2.0
        vol_target_mult = 3.0
        if "ATR" in target_type:
            atr_target_mult = st.number_input("ATR Tgt Multiplier", 0.5, 20.0, 2.0, 0.1, key="_atm")
        if "Risk Reward" in target_type:
            rr_ratio = st.number_input("Risk:Reward (Target)", 0.5, 20.0, 2.0, 0.5, key="_rrt")
        if "Volatility" in target_type:
            vol_target_mult = st.number_input("Vol Tgt Multiplier", 0.5, 20.0, 3.0, 0.1, key="_vtm")

        # Partial booking
        enable_partial = st.checkbox("Partial Profit Booking", False, key="_pb")
        partial_pct    = 50
        if enable_partial:
            partial_pct = st.number_input("Book % at Target 1", 10, 90, 50, 10, key="_pbp")

        st.markdown("---")
        st.markdown("### ⚙️ Trade Settings")
        quantity       = st.number_input("Quantity", 1, 100000, 1, key="_qty")
        enable_cool    = st.checkbox("Cooldown Between Trades", True, key="_ecl")
        cooldown_secs  = 5
        if enable_cool:
            cooldown_secs = st.number_input("Cooldown (s)", 1, 3600, 5, key="_cls")
        prevent_overlap = st.checkbox("Prevent Overlapping Trades", True, key="_po")

        st.markdown("---")
        st.markdown("### 🏦 Dhan Broker")
        enable_dhan = st.checkbox("Enable Dhan", False, key="_edhan")

        dhan_cfg: dict = {"enabled": False}
        if enable_dhan:
            dhan_cfg["enabled"]      = True
            dhan_cfg["client_id"]    = st.text_input("Client ID", "1104779876", key="_did")
            dhan_cfg["access_token"] = st.text_input("Access Token", "", type="password", key="_dat")
            options_trading          = st.checkbox("Options Trading", False, key="_ot")
            dhan_cfg["options_trading"] = options_trading

            if not options_trading:
                dhan_cfg["exchange"]       = st.selectbox("Exchange", ["NSE","BSE"], key="_dex")
                dhan_cfg["product_type"]   = st.selectbox("Product", ["INTRADAY","DELIVERY"], key="_dpt")
                dhan_cfg["security_id"]    = st.text_input("Security ID", "1594", key="_dsid")
                dhan_cfg["quantity"]       = st.number_input("Order Qty", 1, 100000, 1, key="_dqty")
                dhan_cfg["entry_order_type"] = st.selectbox("Entry Order", ["MARKET","LIMIT"], index=1, key="_det")
                dhan_cfg["exit_order_type"]  = st.selectbox("Exit Order",  ["MARKET","LIMIT"], key="_dxt")
            else:
                dhan_cfg["options_exchange"]      = st.selectbox("Opt Exchange", ["NSE_FNO","BSE_FNO"], key="_oex")
                dhan_cfg["ce_security_id"]        = st.text_input("CE Security ID", "57749", key="_ce")
                dhan_cfg["pe_security_id"]        = st.text_input("PE Security ID", "57716", key="_pe")
                dhan_cfg["options_quantity"]      = st.number_input("Opt Qty", 1, 100000, 65, key="_oqty")
                dhan_cfg["options_entry_order_type"] = st.selectbox("Opt Entry", ["MARKET","LIMIT"], key="_oet")
                dhan_cfg["options_exit_order_type"]  = st.selectbox("Opt Exit",  ["MARKET","LIMIT"], key="_oxt")

            if st.button("🌐 Register IP (SEBI)", key="_regip"):
                ip = get_my_ip()
                st.info(f"Your IP: **{ip}** – Add this to Dhan whitelist.")

    # Assemble config
    config: dict = {
        "strategy":        strategy,
        "fast_ema":        fast_ema,
        "slow_ema":        slow_ema,
        "sl_type":         sl_type,
        "sl_points":       sl_points,
        "target_type":     target_type,
        "target_points":   target_points,
        "quantity":        quantity,
        "enable_cooldown": enable_cool,
        "cooldown":        cooldown_secs,
        "prevent_overlap": prevent_overlap,
        "min_angle":       min_angle,
        "crossover_type":  crossover_type,
        "custom_candle_size": custom_candle_sz,
        "min_wave_pct":    min_wave_pct,
        "atr_sl_mult":     atr_sl_mult,
        "atr_target_mult": atr_target_mult,
        "risk_reward":     rr_ratio,
        "swing_window":    swing_window,
        "vol_sl_mult":     vol_sl_mult,
        "vol_target_mult": vol_target_mult,
        "enable_partial":  enable_partial,
        "partial_pct":     partial_pct,
    }

    # ── HEADER ───────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([5,1])
    with col_h1:
        live_dot = '<span class="live-dot"></span>' if st.session_state.live_running else ""
        st.markdown(
            f'<div class="header-bar">'
            f'  <div class="logo">📈 Smart Investing</div>'
            f'  <span class="version">v{APP_VERSION}</span>'
            f'  {live_dot}'
            f'  <span style="font-size:12px;color:#8b949e;">'
            f'    {ticker_sym} · {interval} · {period}'
            f'  </span>'
            f'</div>', unsafe_allow_html=True)
    with col_h2:
        if st.session_state.live_running:
            if st.button("🔄 Refresh", key="_ref"):
                st.rerun()

    # ── TABS ─────────────────────────────────────────────────────────────────
    t_bt, t_lt, t_th, t_opt = st.tabs([
        "📊 Backtest", "🔴 Live Trading", "📋 Trade History", "⚙️ Optimization"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – BACKTESTING
    # ══════════════════════════════════════════════════════════════════════════
    with t_bt:
        st.markdown("### Backtesting Engine")

        brun = st.button("▶ Run Backtest", type="primary", key="_brun")

        if brun:
            with st.spinner("Fetching data…"):
                df_full, n_main = fetch_with_warmup(ticker_sym, period, interval)

            if df_full is None or len(df_full) < 5:
                st.error("❌ Could not fetch data. Check ticker / period / internet.")
            else:
                prog_bar = st.progress(0.0, text="Running backtest…")
                def _prog(v): prog_bar.progress(min(v, 1.0))
                trades, violations, inds = run_backtest(df_full.copy(), config, _prog)
                prog_bar.empty()

                st.session_state.backtest_results = {
                    "trades": trades, "violations": violations,
                    "inds": inds, "df": df_full
                }
                st.session_state.backtest_fig = build_backtest_chart(df_full, trades, inds, config)
                st.success(f"✅ Backtest complete – {len(trades)} trades analysed.")

        if st.session_state.backtest_results:
            res    = st.session_state.backtest_results
            trades = res["trades"]
            viol   = res["violations"]
            df_bt  = res["df"]
            inds   = res["inds"]

            wins, n_t, acc = calc_accuracy(trades)
            total_pnl = sum(t["PnL"] for t in trades)
            avg_pnl   = total_pnl / n_t if n_t else 0

            # Metrics
            cols = st.columns(6)
            mets = [
                ("Total Trades", n_t, "blue"),
                ("Wins",        wins, "green"),
                ("Losses",      n_t - wins, "red"),
                ("Accuracy",    f"{acc}%", "green" if acc >= 50 else "red"),
                ("Total PnL",   _fmt_pnl(total_pnl), "green" if total_pnl >= 0 else "red"),
                ("Avg PnL",     f"{avg_pnl:+.2f}", "green" if avg_pnl >= 0 else "red"),
            ]
            for col, (lbl, val, clr) in zip(cols, mets):
                col.markdown(metric_card(lbl, val, clr), unsafe_allow_html=True)

            # Violation banner
            if viol:
                st.markdown(
                    f'<div class="violation-badge">⚠️ {len(viol)} candles where '
                    f'SL & Target both hit in same bar (Conservative: SL taken first)</div>',
                    unsafe_allow_html=True)

            st.markdown("---")

            # Chart
            if st.session_state.backtest_fig:
                st.plotly_chart(st.session_state.backtest_fig,
                                use_container_width=True, key="_btfig")

            # Trade table
            st.markdown("#### 📋 Trade Log")
            if trades:
                df_tr = pd.DataFrame(trades)
                # Color coding
                def style_row(row):
                    clr = "background-color:rgba(0,230,118,0.08)" if row["PnL"] >= 0 \
                          else "background-color:rgba(255,23,68,0.08)"
                    if row.get("Is Violation"):
                        clr += ";border-left:3px solid #ff9800"
                    return [clr] * len(row)
                styled = df_tr.style.apply(style_row, axis=1)\
                               .format({"PnL": "{:+.2f}", "Entry Price": "{:.2f}",
                                        "Exit Price": "{:.2f}", "SL": "{:.2f}",
                                        "Target": "{:.2f}"})
                st.dataframe(styled, use_container_width=True, height=380)

            # Violation table
            if viol:
                st.markdown(f"#### ⚠️ Violation Trades ({len(viol)})")
                st.dataframe(pd.DataFrame(viol), use_container_width=True)

            # Elliott wave summary
            if strategy == "Elliott Wave" and "elliott" in inds and inds["elliott"]:
                ew = inds["elliott"]
                st.markdown("#### 🌊 Elliott Wave Analysis")
                _render_ew_info(ew)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – LIVE TRADING
    # ══════════════════════════════════════════════════════════════════════════
    with t_lt:
        st.markdown("### Live Trading")

        # Control buttons
        ctrl_cols = st.columns([1.5,1.5,1.5,4])
        start_btn   = ctrl_cols[0].button("▶ Start",    type="primary",   key="_start")
        stop_btn    = ctrl_cols[1].button("⏹ Stop",     type="secondary", key="_stop")
        squareoff_b = ctrl_cols[2].button("⚡ Squareoff", type="secondary", key="_sq")

        if start_btn and not st.session_state.live_running:
            se = threading.Event()
            st.session_state.stop_event  = se
            st.session_state.live_running = True
            _ts_set("trade_history", [])  # fresh session history
            t = threading.Thread(
                target=live_thread_fn,
                args=(ticker_sym, period, interval, config, se, dhan_cfg),
                daemon=True
            )
            st.session_state.live_thread = t
            t.start()
            st.toast("🟢 Live trading started!", icon="✅")

        if stop_btn and st.session_state.live_running:
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.session_state.live_running = False
            st.toast("⏹ Stop signal sent.", icon="⏹")

        if squareoff_b:
            pos = _ts_get("current_position")
            if pos is not None and dhan_cfg.get("enabled"):
                ltp = _ts_get("last_candle", {}).get("close", 0)
                resp = place_exit_order(dhan_cfg, pos, ltp,
                                         dhan_cfg.get("options_trading", False))
                _ts_set("current_position", None)
                st.toast(f"⚡ Squareoff sent: {resp}", icon="⚡")
            elif pos is not None:
                _ts_set("current_position", None)
                st.toast("Position squared off (paper).", icon="✅")
            else:
                st.toast("No open position.", icon="ℹ️")

        # Config display (refreshes in-place)
        st.markdown(
            f'<div class="config-display">'
            f'<b>Active Config</b> &nbsp;|&nbsp; '
            f'Ticker: <b>{ticker_sym}</b> &nbsp;·&nbsp; '
            f'TF: <b>{interval}</b> &nbsp;·&nbsp; '
            f'Period: <b>{period}</b> &nbsp;·&nbsp; '
            f'Strategy: <b>{strategy}</b> &nbsp;·&nbsp; '
            f'EMA: <b>{fast_ema}/{slow_ema}</b> &nbsp;·&nbsp; '
            f'SL: <b>{sl_type} ({sl_points}pts)</b> &nbsp;·&nbsp; '
            f'Tgt: <b>{target_type} ({target_points}pts)</b> &nbsp;·&nbsp; '
            f'Qty: <b>{quantity}</b>'
            f'</div>', unsafe_allow_html=True)

        # Live status row
        pos      = _ts_get("current_position")
        live_pnl = _ts_get("current_pnl", 0.0)
        lc       = _ts_get("last_candle")

        lc_cols = st.columns(6)
        if lc:
            lc_cols[0].metric("Last Close", f"{lc.get('close',0):.2f}")
            lc_cols[1].metric("Last High",  f"{lc.get('high',0):.2f}")
            lc_cols[2].metric("Last Low",   f"{lc.get('low',0):.2f}")
            lc_cols[3].metric("Last Open",  f"{lc.get('open',0):.2f}")
            lc_cols[4].metric("Volume",     f"{lc.get('volume',0):,}")
            lc_cols[5].metric("Bar Time",   str(lc.get("datetime",""))[-8:-3])

        st.markdown("---")

        chart_col, info_col = st.columns([2.5, 1])

        with chart_col:
            # Live chart
            ld = _ts_get("live_data")
            if ld and len(ld.get("close",[])) >= 3:
                df_live = pd.DataFrame({
                    "Open":   ld["open"],  "High":  ld["high"],
                    "Low":    ld["low"],   "Close": ld["close"],
                    "Volume": ld["volume"],
                }, index=pd.to_datetime(ld["index"]))
                live_inds = {}
                if strategy in ("EMA Crossover","Anticipatory EMA","Elliott Wave"):
                    live_inds["fast_ema"] = ema(df_live["Close"], fast_ema)
                    live_inds["slow_ema"] = ema(df_live["Close"], slow_ema)
                ew_state = _ts_get("elliott_wave_state")
                if ew_state:
                    live_inds["elliott"] = ew_state
                fig_live = build_live_chart(df_live, pos, live_inds, config)
                st.plotly_chart(fig_live, use_container_width=True, key="_livechart")
            else:
                st.info("⏳ Waiting for live data…")

        with info_col:
            # Current position card
            st.markdown("**📌 Current Position**")
            if pos:
                sig_str = "🟢 BUY" if pos["signal_type"] == 1 else "🔴 SELL"
                pnl_clr = "#00e676" if live_pnl >= 0 else "#ff1744"
                st.markdown(f"""
                <div class="position-card">
                  <div style="font-size:18px;font-weight:700;">{sig_str}</div>
                  <div style="margin-top:10px;font-size:12px;">
                    <div>Entry: <b>{pos['entry_price']:.2f}</b></div>
                    <div>SL: <b style="color:#ff1744">{pos['current_sl']:.2f}</b></div>
                    <div>Target: <b style="color:#00e676">{pos['target']:.2f}</b></div>
                    <div>Since: {pos['entry_dt']}</div>
                    <div style="margin-top:8px;font-size:16px;color:{pnl_clr};">
                      P&L: <b>{live_pnl:+.2f} pts</b>
                    </div>
                    <div style="font-size:10px;color:#8b949e;margin-top:4px;">
                      {pos.get('signal_reason','')[:60]}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#8b949e;font-size:13px;padding:16px;">'
                    'No open position</div>', unsafe_allow_html=True)

            # Elliott Wave Info
            st.markdown("**🌊 Elliott Wave**")
            ew_s = _ts_get("elliott_wave_state")
            if ew_s and ew_s.get("pattern","") != "Insufficient Data":
                _render_ew_info(ew_s)
            else:
                st.markdown(
                    '<div style="color:#8b949e;font-size:12px;padding:8px;">—</div>',
                    unsafe_allow_html=True)

        # Trade log
        st.markdown("**📋 Live Session Log**")
        logs = _ts_get("live_log", [])
        log_html = "<br>".join(reversed(logs[-60:]))
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – TRADE HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with t_th:
        st.markdown("### Trade History")

        # Merge live history + backtest history
        live_hist = _ts_get("trade_history", [])
        bt_hist   = st.session_state.backtest_results["trades"] \
                    if st.session_state.backtest_results else []

        mode_filter = st.radio("Show", ["All","Live","Backtest"],
                               horizontal=True, key="_hf")
        all_hist = live_hist + bt_hist
        if mode_filter == "Live":
            all_hist = live_hist
        elif mode_filter == "Backtest":
            all_hist = bt_hist

        if not all_hist:
            st.info("No trades recorded yet.")
        else:
            df_hist  = pd.DataFrame(all_hist)
            wins_h, n_h, acc_h = calc_accuracy(all_hist)
            tot_pnl  = sum(t["PnL"] for t in all_hist)
            max_win  = max((t["PnL"] for t in all_hist), default=0)
            max_loss = min((t["PnL"] for t in all_hist), default=0)

            hc = st.columns(6)
            hc[0].metric("Total Trades", n_h)
            hc[1].metric("Wins",         wins_h)
            hc[2].metric("Losses",       n_h - wins_h)
            hc[3].metric("Accuracy",     f"{acc_h}%")
            hc[4].metric("Total PnL",    f"{tot_pnl:+.2f}")
            hc[5].metric("Best Trade",   f"{max_win:+.2f}")

            # PnL chart
            pnl_vals  = [t["PnL"] for t in all_hist]
            cum_pnl   = list(pd.Series(pnl_vals).cumsum())
            bar_clrs  = [_GREEN if p >= 0 else _RED for p in pnl_vals]

            fig_pnl = make_subplots(rows=1, cols=2,
                subplot_titles=("Cumulative PnL", "Per-Trade PnL"))
            fig_pnl.add_trace(go.Scatter(
                y=cum_pnl, mode="lines+markers", name="Cum PnL",
                line=dict(color=_BLUE, width=2),
                marker=dict(color=[_GREEN if v>=0 else _RED for v in cum_pnl], size=6)
            ), row=1, col=1)
            fig_pnl.add_trace(go.Bar(
                y=pnl_vals, name="Trade PnL",
                marker_color=bar_clrs), row=1, col=2)
            fig_pnl.update_layout(**_base_layout("PnL Analysis", 320))
            st.plotly_chart(fig_pnl, use_container_width=True, key="_pnlfig")

            st.dataframe(df_hist.style.apply(
                lambda row: ["background-color:rgba(0,230,118,0.08)"
                             if row["PnL"] >= 0
                             else "background-color:rgba(255,23,68,0.08)"] * len(row),
                axis=1).format({"PnL": "{:+.2f}"}),
                use_container_width=True, height=400)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════════
    with t_opt:
        st.markdown("### Strategy Optimization")
        st.markdown("Grid-search over EMA periods and SL/Target points.")

        oc1, oc2, oc3 = st.columns(3)
        target_acc = oc1.number_input("Min Accuracy % Filter", 0.0, 100.0, 0.0, 5.0, key="_tacc")
        min_trades  = oc2.number_input("Min Trades", 1, 200, 5, key="_mtr")

        if oc3.button("🚀 Run Optimization", type="primary", key="_optrun"):
            with st.spinner("Fetching data for optimization…"):
                df_opt, _ = fetch_with_warmup(ticker_sym, period, interval)

            if df_opt is None or len(df_opt) < 20:
                st.error("Not enough data for optimization.")
            else:
                prog_opt = st.progress(0.0, text="Optimizing…")
                opt_cfg  = {**config, "strategy": "EMA Crossover"}
                opt_res  = run_optimization(df_opt, opt_cfg, target_acc, prog_opt)
                prog_opt.empty()

                if opt_res.empty:
                    st.warning("No results found.")
                else:
                    opt_res = opt_res[opt_res["Trades"] >= min_trades]
                    st.session_state.opt_results = opt_res
                    st.success(f"Found {len(opt_res)} parameter combinations.")

        if st.session_state.opt_results is not None:
            opt_df = st.session_state.opt_results

            if not opt_df.empty:
                best = opt_df.iloc[0]
                bc = st.columns(4)
                bc[0].metric("Best Accuracy",  f"{best['Accuracy %']:.1f}%")
                bc[1].metric("Best Fast EMA",  int(best["Fast EMA"]))
                bc[2].metric("Best Slow EMA",  int(best["Slow EMA"]))
                bc[3].metric("Best PnL",       f"{best['Total PnL']:+.2f}")

                # Heatmap: Fast × Slow → Accuracy
                pivot = opt_df.pivot_table(
                    values="Accuracy %", index="Slow EMA",
                    columns="Fast EMA", aggfunc="max")
                fig_heat = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale="RdYlGn", showscale=True,
                    text=pivot.values.round(1), texttemplate="%{text}%",
                ))
                fig_heat.update_layout(**_base_layout("EMA Accuracy Heatmap", 380))
                st.plotly_chart(fig_heat, use_container_width=True, key="_heat")

                # Apply best params button
                if st.button("⚡ Apply Best Parameters to Sidebar", key="_applyopt"):
                    st.info(
                        f"Set Fast EMA={int(best['Fast EMA'])}, "
                        f"Slow EMA={int(best['Slow EMA'])}, "
                        f"SL Points={int(best['SL Pts'])}, "
                        f"Target Points={int(best['Tgt Pts'])} in the sidebar."
                    )

                st.dataframe(opt_df.head(50).style.apply(
                    lambda row: ["background-color:rgba(0,230,118,0.08)"
                                 if row["Accuracy %"] >= 60 else ""] * len(row),
                    axis=1).format({"Accuracy %": "{:.1f}%", "Total PnL": "{:+.2f}",
                                    "Avg PnL": "{:+.2f}"}),
                    use_container_width=True, height=380)


def _render_ew_info(ew: dict):
    """Render Elliott Wave info card."""
    if not ew:
        return
    direction_clr = {"Bullish":"#00e676","Bearish":"#ff1744",
                      "Bullish ABC":"#ffea00","Bearish ABC":"#ff9800"}.get(
                      ew.get("wave_direction",""), "#40c4ff")

    completed_str = " → ".join(ew.get("completed_waves",[]) or ["–"])
    signal_clr    = "#00e676" if ew.get("signal")=="BUY" else \
                    "#ff1744" if ew.get("signal")=="SELL" else "#8b949e"
    sig_str       = ew.get("signal","NONE")

    fib_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:11px;padding:2px 0;color:#a0aab4;">'
        f'<span>{k}</span><span style="color:#e6edf3;">{v}</span></div>'
        for k, v in (ew.get("fibonacci_levels") or {}).items()
    )

    wp_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:11px;padding:1px 0;color:#a0aab4;">'
        f'<span>{k}</span><span style="color:#e6edf3;">{v}</span></div>'
        for k, v in (ew.get("wave_points") or {}).items()
    )

    tgt = ew.get("next_target")
    tgt_str = f"{tgt:.2f}" if tgt else "–"

    st.markdown(f"""
    <div class="wave-card">
      <h4>🌊 {ew.get("pattern","–")}</h4>
      <div style="font-size:11px;color:{direction_clr};font-weight:700;margin-bottom:6px;">
        {ew.get("wave_direction","–")}
      </div>
      <div style="font-size:11px;color:#8b949e;">Completed: {completed_str}</div>
      <div style="font-size:12px;color:#e6edf3;margin:4px 0;">
        ▶ {ew.get("current_wave","–")}
      </div>
      <div style="font-size:12px;margin:4px 0;">
        Signal: <b style="color:{signal_clr};">{sig_str}</b>
        &nbsp;|&nbsp; Next Target: <b style="color:#ffea00;">{tgt_str}</b>
      </div>
      <div style="margin-top:8px;">{wp_rows}</div>
      <div style="margin-top:8px;border-top:1px solid rgba(255,255,255,0.07);
                  padding-top:8px;">{fib_rows}</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
