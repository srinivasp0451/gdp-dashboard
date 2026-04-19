"""
Smart Investing - Professional Algorithmic Trading Platform
===========================================================
Author: Smart Investing Platform
Description: Full-featured algo trading platform with yfinance, Dhan broker,
             Elliott Wave analysis, multiple strategies, backtesting & live trading.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
import math
import random
import requests
import itertools
from datetime import datetime, timedelta
from collections import deque
import pytz

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc

PRESET_TICKERS = {
    "Nifty 50":   "^NSEI",
    "BankNifty":  "^NSEBANK",
    "Sensex":     "^BSESN",
    "BTC-USD":    "BTC-USD",
    "ETH-USD":    "ETH-USD",
    "Gold":       "GC=F",
    "Silver":     "SI=F",
    "Custom":     "",
}

TIMEFRAME_PERIODS = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

# Period ordering for warmup calculation
PERIOD_ORDER = ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]

STRATEGIES = [
    "EMA Crossover",
    "Anticipatory EMA Crossover",
    "Elliott Wave (Auto)",
    "Simple Buy",
    "Simple Sell",
    # "Price Crosses Threshold",   ← commented out intentionally, not shown in dropdown
]

SL_TYPES = [
    "Custom Points",
    "ATR Based",
    "Risk Reward Based",
    "Trailing SL",
    "Trailing SL – Swing Low/High",
    "Trailing SL – Candle Low/High",
    "EMA Reverse Crossover",
    "Auto SL",
    "Volatility Based",
    "Nearest Support/Resistance",
]

TARGET_TYPES = [
    "Custom Points",
    "ATR Based",
    "Risk Reward Based",
    "Trailing Target (Display Only)",
    "Trailing Target – Swing Low/High",
    "Trailing Target – Candle Low/High",
    "EMA Crossover Exit",
    "Auto Target",
    "Volatility Based",
    "Nearest Support/Resistance",
]

TF_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15,
    "1h": 60, "1d": 1440, "1wk": 10080,
}

WARMUP_CANDLES = 200   # Extra candles fetched for indicator accuracy

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE STATE (module-level dict + RLock)
# ─────────────────────────────────────────────────────────────────────────────
_LOCK = threading.RLock()
_TS: dict = {}


def ts_get(key, default=None):
    with _LOCK:
        return _TS.get(key, default)


def ts_set(key, value):
    with _LOCK:
        _TS[key] = value


def ts_append(key, value):
    with _LOCK:
        if key not in _TS:
            _TS[key] = []
        _TS[key].append(value)


def ts_clear(key):
    with _LOCK:
        _TS[key] = []


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "live_thread":        None,
        "backtest_results":   None,
        "backtest_df":        None,
        "opt_results":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# CSS STYLES
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, .log-entry {
    font-family: 'JetBrains Mono', monospace;
}

/* ── Overall background ── */
.stApp { background: #050A14; }
section[data-testid="stSidebar"] > div { background: #060D1C; border-right: 1px solid #0f2040; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0A1628;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #0f2040;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #5a7fa8;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 16px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0f3460, #1a4a80) !important;
    color: #00E5CC !important;
    box-shadow: 0 2px 12px rgba(0,229,204,0.2);
}

/* ── Metric cards ── */
.m-card {
    background: linear-gradient(135deg, #0A1628, #0d1f3c);
    border: 1px solid #0f3060;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.m-card:hover { border-color: #1a5080; }
.m-card .mc-label { color: #5a7fa8; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin: 0; }
.m-card .mc-val   { color: #e8f4ff; font-size: 1.35rem; font-weight: 700; margin: 4px 0 0 0; font-family:'JetBrains Mono',monospace; }

/* ── LTP Banner ── */
.ltp-banner {
    background: linear-gradient(90deg, #050A14 0%, #091527 40%, #0a1e35 60%, #050A14 100%);
    border: 1px solid #0f3060;
    border-radius: 14px;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}
.ltp-banner .ticker-name { color: #5a7fa8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
.ltp-banner .ltp-val     { color: #00E5CC; font-size: 2rem; font-weight: 800; font-family:'JetBrains Mono',monospace; }
.ltp-banner .ltp-time    { color: #3a5a78; font-size: 0.75rem; }

/* ── Position card ── */
.pos-card {
    background: linear-gradient(135deg, #0A1628, #0d2040);
    border: 1px solid;
    border-radius: 14px;
    padding: 18px;
}
.pos-card.buy-card  { border-color: #00c896; box-shadow: 0 0 20px rgba(0,200,150,0.08); }
.pos-card.sell-card { border-color: #ff4d6d; box-shadow: 0 0 20px rgba(255,77,109,0.08); }
.pos-label { font-size: 0.7rem; color: #5a7fa8; text-transform: uppercase; letter-spacing: 0.08em; }
.pos-val   { font-size: 1.05rem; font-weight: 700; font-family:'JetBrains Mono',monospace; color: #e8f4ff; }

/* ── Wave card ── */
.wave-card {
    background: linear-gradient(135deg, #0A1628, #0c1a38);
    border: 1px solid #1a3060;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}
.wave-status { color: #4FC3F7; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px; }
.wave-row    { display: flex; justify-content: space-between; align-items: center; margin: 4px 0; font-size: 0.82rem; }
.wave-key    { color: #5a7fa8; }
.wave-val    { color: #e8f4ff; font-family:'JetBrains Mono',monospace; font-weight: 600; }

/* ── Log container ── */
.log-box {
    background: #030810;
    border: 1px solid #0f2040;
    border-radius: 10px;
    padding: 12px;
    max-height: 220px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.log-box::-webkit-scrollbar { width: 4px; }
.log-box::-webkit-scrollbar-track { background: #030810; }
.log-box::-webkit-scrollbar-thumb { background: #1a3060; border-radius: 2px; }

/* ── Violation alert ── */
.violation-box {
    background: #1a0a0a;
    border: 1px solid #8B2500;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 10px 0;
    color: #ff8c69;
    font-size: 0.85rem;
}

/* ── Best result box ── */
.best-result {
    background: linear-gradient(135deg, #051a10, #082a18);
    border: 1px solid #1a7a50;
    border-radius: 12px;
    padding: 18px;
    margin: 12px 0;
}

/* ── Status dot ── */
.status-live { display:inline-block; width:8px; height:8px; background:#00E5CC; border-radius:50%; margin-right:6px; animation: pulse-green 1.5s infinite; }
.status-off  { display:inline-block; width:8px; height:8px; background:#3a5a78; border-radius:50%; margin-right:6px; }

@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,229,204,0.4); }
    50%      { box-shadow: 0 0 0 6px rgba(0,229,204,0); }
}

/* ── Sidebar sections ── */
.sb-section {
    background: #070F20;
    border: 1px solid #0f2040;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 8px;
}
.sb-header { color: #00E5CC; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }

/* ── Buttons ── */
button[kind="primary"]    { background: linear-gradient(135deg, #006644, #00a86b) !important; border: none !important; color: white !important; font-weight: 700 !important; border-radius: 8px !important; }
button[kind="secondary"]  { background: #0A1628 !important; border: 1px solid #0f3060 !important; color: #5a7fa8 !important; border-radius: 8px !important; }

/* ── Streamlit overrides ── */
.stSelectbox label, .stNumberInput label, .stCheckbox label, .stSlider label { color: #7a9fc0 !important; font-size: 0.8rem !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stDecoration"] { display: none; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def _get_warmup_period(timeframe: str) -> str:
    """Return a period large enough to have WARMUP_CANDLES for EMA seeding."""
    m = {
        "1m": "5d", "5m": "5d", "15m": "7d",
        "1h": "1mo", "1d": "1y", "1wk": "2y",
    }
    return m.get(timeframe, "1mo")


def fetch_data(ticker: str, timeframe: str, period: str, warmup: bool = True) -> pd.DataFrame | None:
    """
    Fetch OHLCV data from yfinance with adequate warmup candles so that EMA
    calculations never produce NaN at the display window start (gap-up / gap-down safe).
    Always uses adjust=True (auto-adjust) and sorts ascending.
    """
    try:
        if warmup:
            warm_p = _get_warmup_period(timeframe)
            try:
                req_idx  = PERIOD_ORDER.index(period)
            except ValueError:
                req_idx = 0
            try:
                warm_idx = PERIOD_ORDER.index(warm_p)
            except ValueError:
                warm_idx = 0
            fetch_period = PERIOD_ORDER[max(req_idx, warm_idx)]
        else:
            fetch_period = period

        raw = yf.download(
            ticker,
            period=fetch_period,
            interval=timeframe,
            progress=False,
            auto_adjust=True,
        )

        if raw is None or raw.empty:
            return None

        # Flatten MultiIndex columns (yfinance ≥ 0.2.x sometimes adds ticker level)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]

        raw.index = pd.to_datetime(raw.index)

        # Ensure IST timezone
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC").tz_convert(IST)
        else:
            raw.index = raw.index.tz_convert(IST)

        raw = raw.sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        raw = raw.dropna(subset=["Close"])

        # Tag warmup rows so we can separate display vs. calculation data
        raw["_warmup"] = False
        if warmup and len(raw) > WARMUP_CANDLES:
            raw.iloc[:WARMUP_CANDLES, raw.columns.get_loc("_warmup")] = True

        return raw

    except Exception as exc:
        st.error(f"Data fetch error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """
    TradingView-accurate EMA:  ewm(span=n, adjust=False, min_periods=1)
    min_periods=1 ensures NO NaN even on first bar (gap-up / sparse data safe).
    """
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR – matches TradingView."""
    hl   = df["High"] - df["Low"]
    hpc  = (df["High"] - df["Close"].shift(1)).abs()
    lpc  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()


def calc_bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std(ddof=0)
    return mid - std_mult * std, mid, mid + std_mult * std


def crossover_angle(fast: pd.Series, slow: pd.Series, i: int) -> float:
    """Absolute gap change at crossover point – proxy for crossover angle."""
    if i < 1:
        return 0.0
    return abs((fast.iloc[i] - slow.iloc[i]) - (fast.iloc[i - 1] - slow.iloc[i - 1]))


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORT / RESISTANCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def find_swing_pivots(df: pd.DataFrame, left: int = 5, right: int = 5):
    """Return lists of (index_pos, timestamp, price) swing highs and swing lows."""
    highs, lows = [], []
    h_arr = df["High"].values
    l_arr = df["Low"].values
    n     = len(df)

    for i in range(left, n - right):
        if all(h_arr[i] >= h_arr[i - j] for j in range(1, left + 1)) and \
           all(h_arr[i] >= h_arr[i + j] for j in range(1, right + 1)):
            highs.append((i, df.index[i], h_arr[i]))
        if all(l_arr[i] <= l_arr[i - j] for j in range(1, left + 1)) and \
           all(l_arr[i] <= l_arr[i + j] for j in range(1, right + 1)):
            lows.append((i, df.index[i], l_arr[i]))
    return highs, lows


def nearest_support_resistance(df: pd.DataFrame, lookback: int = 60):
    """Return (support_price, resistance_price) nearest to current price."""
    recent = df.tail(lookback)
    highs, lows = find_swing_pivots(recent, left=3, right=3)
    price = float(df["Close"].iloc[-1])

    resistances = sorted([h[2] for h in highs if h[2] > price])
    supports    = sorted([l[2] for l in lows  if l[2] < price], reverse=True)

    support    = supports[0]    if supports    else price * 0.98
    resistance = resistances[0] if resistances else price * 1.02
    return float(support), float(resistance)


# ─────────────────────────────────────────────────────────────────────────────
# ELLIOTT WAVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def _zigzag(prices: np.ndarray, pct: float = 0.03):
    """
    Build ZigZag pivot list from price array.
    Returns list of (array_index, price, 'H'|'L').
    """
    pivots = []
    direction = None
    peak_idx, peak_val = 0, prices[0]
    trough_idx, trough_val = 0, prices[0]

    for i in range(1, len(prices)):
        p = prices[i]
        if direction is None:
            if p > peak_val * (1 + pct):
                direction = "up"
                pivots.append((trough_idx, trough_val, "L"))
                peak_idx, peak_val = i, p
            elif p < trough_val * (1 - pct):
                direction = "down"
                pivots.append((peak_idx, peak_val, "H"))
                trough_idx, trough_val = i, p
        elif direction == "up":
            if p > peak_val:
                peak_idx, peak_val = i, p
            elif p < peak_val * (1 - pct):
                pivots.append((peak_idx, peak_val, "H"))
                direction = "down"
                trough_idx, trough_val = i, p
        else:  # down
            if p < trough_val:
                trough_idx, trough_val = i, p
            elif p > trough_val * (1 + pct):
                pivots.append((trough_idx, trough_val, "L"))
                direction = "up"
                peak_idx, peak_val = i, p

    # Append last pivot
    if direction == "up":
        pivots.append((peak_idx, peak_val, "H"))
    elif direction == "down":
        pivots.append((trough_idx, trough_val, "L"))

    return pivots


def detect_elliott_waves(df: pd.DataFrame) -> dict:
    """
    Detect Elliott 5-wave impulse and ABC correction.
    Returns a rich dict used by both live trading and backtesting.
    """
    closes = df["Close"].values
    result = {
        "pivots":          [],
        "pattern":         None,
        "wave_status":     "Analyzing…",
        "current_wave":    None,
        "signal":          None,
        "next_target":     None,
        "completed_waves": [],
        "wave_projections": {},
        "auto_sl":         None,
        "auto_target":     None,
    }

    if len(closes) < 50:
        result["wave_status"] = "Insufficient data for Elliott Wave"
        return result

    # Try different sensitivities to find a pattern
    for pct in [0.02, 0.03, 0.05, 0.08]:
        pivots = _zigzag(closes, pct=pct)
        result["pivots"] = pivots
        if len(pivots) < 6:
            continue

        # Scan for 5-wave impulse in last N pivots
        for start in range(max(0, len(pivots) - 12), len(pivots) - 5):
            seg   = pivots[start: start + 6]
            types = [p[2] for p in seg]
            vals  = [p[1] for p in seg]

            # ── Bullish Impulse: L H L H L H ──
            if types == ["L", "H", "L", "H", "L", "H"]:
                w0,w1,w2,w3,w4,w5 = vals
                rule1 = w2 > w0          # Wave 2 > Wave 0
                rule2 = w3 > w1          # Wave 3 > Wave 1 peak
                rule3 = w4 > w2          # Wave 4 > Wave 2 trough  (simplified)
                rule5_ok = (w3 - w2) > (w1 - w0) or (w5 - w4) > (w1 - w0)  # Wave 3 not shortest
                if rule1 and rule2 and rule3:
                    # Fibonacci projections
                    wave1 = w1 - w0
                    wave3_tgt = w2 + wave1 * 1.618
                    wave5_tgt = w4 + wave1 * 1.000
                    abc_a_tgt = w5 - (w5 - w4) * 0.618
                    abc_b_tgt = abc_a_tgt + (w5 - abc_a_tgt) * 0.618
                    abc_c_tgt = abc_a_tgt - (abc_b_tgt - abc_a_tgt) * 1.0

                    atr = float(calc_atr(df).iloc[-1])
                    result.update({
                        "pattern":        "bullish_impulse",
                        "wave_status":    "✅ 5-Wave Bullish Impulse complete → ABC Correction expected",
                        "current_wave":   "Wave A (Corrective – Down)",
                        "completed_waves": ["Wave 1","Wave 2","Wave 3","Wave 4","Wave 5"],
                        "signal":         "sell",
                        "next_target":    round(abc_a_tgt, 2),
                        "wave_projections": {
                            "Wave A target": round(abc_a_tgt, 2),
                            "Wave B target": round(abc_b_tgt, 2),
                            "Wave C target": round(abc_c_tgt, 2),
                        },
                        "auto_sl":    round(w5 + atr * 1.5, 2),
                        "auto_target":round(abc_a_tgt, 2),
                    })
                    return result

            # ── Bearish Impulse: H L H L H L ──
            elif types == ["H", "L", "H", "L", "H", "L"]:
                w0,w1,w2,w3,w4,w5 = vals
                rule1 = w2 < w0
                rule2 = w3 < w1
                rule3 = w4 < w2
                if rule1 and rule2 and rule3:
                    wave1     = w0 - w1
                    abc_a_tgt = w5 + (w4 - w5) * 0.618
                    abc_b_tgt = abc_a_tgt - (abc_a_tgt - w5) * 0.618
                    abc_c_tgt = abc_a_tgt + (abc_a_tgt - abc_b_tgt) * 1.0

                    atr = float(calc_atr(df).iloc[-1])
                    result.update({
                        "pattern":        "bearish_impulse",
                        "wave_status":    "✅ 5-Wave Bearish Impulse complete → ABC Bounce expected",
                        "current_wave":   "Wave A (Corrective – Up)",
                        "completed_waves": ["Wave 1","Wave 2","Wave 3","Wave 4","Wave 5"],
                        "signal":         "buy",
                        "next_target":    round(abc_a_tgt, 2),
                        "wave_projections": {
                            "Wave A target": round(abc_a_tgt, 2),
                            "Wave B target": round(abc_b_tgt, 2),
                            "Wave C target": round(abc_c_tgt, 2),
                        },
                        "auto_sl":    round(w5 - atr * 1.5, 2),
                        "auto_target":round(abc_a_tgt, 2),
                    })
                    return result

        # ── Partial wave detection (in-progress) ──
        if len(pivots) >= 4:
            seg   = pivots[-4:]
            types = [p[2] for p in seg]
            vals  = [p[1] for p in seg]
            atr   = float(calc_atr(df).iloc[-1])
            price = float(df["Close"].iloc[-1])

            if types == ["L", "H", "L", "H"] and vals[1] > vals[0] and vals[3] > vals[1]:
                wave1 = vals[1] - vals[0]
                wave3_proj = vals[2] + wave1 * 1.618
                result.update({
                    "wave_status":    "📊 Possible Wave 3 forming (Bullish Impulse in progress)",
                    "current_wave":   "Wave 3 (In progress)",
                    "completed_waves": ["Wave 1","Wave 2"],
                    "signal":         "buy",
                    "next_target":    round(wave3_proj, 2),
                    "wave_projections": {
                        "Wave 3 target (1.618)": round(vals[2] + wave1 * 1.618, 2),
                        "Wave 3 target (2.618)": round(vals[2] + wave1 * 2.618, 2),
                        "Wave 5 projection":     round(vals[2] + wave1 * 3.236, 2),
                    },
                    "auto_sl":    round(vals[2] - atr * 1.5, 2),
                    "auto_target":round(vals[2] + wave1 * 1.618, 2),
                })
                return result

            if types == ["H", "L", "H", "L"] and vals[1] < vals[0] and vals[3] < vals[1]:
                wave1 = vals[0] - vals[1]
                wave3_proj = vals[2] - wave1 * 1.618
                result.update({
                    "wave_status":    "📊 Possible Wave 3 forming (Bearish Impulse in progress)",
                    "current_wave":   "Wave 3 (In progress)",
                    "completed_waves": ["Wave 1","Wave 2"],
                    "signal":         "sell",
                    "next_target":    round(wave3_proj, 2),
                    "wave_projections": {
                        "Wave 3 target (1.618)": round(vals[2] - wave1 * 1.618, 2),
                        "Wave 3 target (2.618)": round(vals[2] - wave1 * 2.618, 2),
                        "Wave 5 projection":     round(vals[2] - wave1 * 3.236, 2),
                    },
                    "auto_sl":    round(vals[2] + atr * 1.5, 2),
                    "auto_target":round(vals[2] - wave1 * 1.618, 2),
                })
                return result

    result["wave_status"] = "⏳ No clear Elliott Wave pattern yet – watching for pivots…"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EMA CROSSOVER STRATEGY SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
def _add_ema_cols(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = calc_ema(df["Close"], fast)
    df["ema_slow"] = calc_ema(df["Close"], slow)
    df["atr"]      = calc_atr(df)
    return df


def generate_ema_crossover_signals(df: pd.DataFrame, fast: int = 9, slow: int = 15,
                                   xover_type: str = "Simple Crossover",
                                   candle_size: float = 10.0,
                                   min_angle: float = 0.0,
                                   check_angle: bool = False) -> pd.DataFrame:
    """
    Generate EMA crossover signals – matches TradingView crossover exactly.
    Signal column: '' / 'buy' / 'sell'
    """
    df = _add_ema_cols(df, fast, slow)
    signals = pd.Series("", index=df.index)

    for i in range(1, len(df)):
        ef_now, ef_prv = df["ema_fast"].iloc[i], df["ema_fast"].iloc[i - 1]
        es_now, es_prv = df["ema_slow"].iloc[i], df["ema_slow"].iloc[i - 1]

        bull = ef_prv <= es_prv and ef_now > es_now
        bear = ef_prv >= es_prv and ef_now < es_now

        if not (bull or bear):
            continue

        # Angle filter
        if check_angle and min_angle > 0:
            angle = abs((ef_now - es_now) - (ef_prv - es_prv))
            if angle < min_angle:
                continue

        # Candle-size filter
        body = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
        if xover_type == "Custom Candle Size" and body < candle_size:
            continue
        if xover_type == "ATR Based Candle Size" and body < df["atr"].iloc[i]:
            continue

        signals.iloc[i] = "buy" if bull else "sell"

    df["signal"] = signals
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANTICIPATORY EMA CROSSOVER
# ─────────────────────────────────────────────────────────────────────────────
def generate_anticipatory_ema_signals(df: pd.DataFrame,
                                      fast: int = 9, slow: int = 15) -> pd.DataFrame:
    """
    Anticipates an imminent EMA crossover BEFORE it happens using:
    1. Rapid gap convergence between fast & slow EMA
    2. Strong candle body in the direction of anticipated crossover
    3. Price already crossing the midpoint of fast/slow range
    4. Momentum: close > open (bull) or close < open (bear) for last 2 candles
    Risk is lower (entry before the cross) and reward is higher.
    """
    df = _add_ema_cols(df, fast, slow)
    signals = pd.Series("", index=df.index)

    for i in range(3, len(df)):
        ef   = df["ema_fast"].iloc[i]
        es   = df["ema_slow"].iloc[i]
        ef1  = df["ema_fast"].iloc[i - 1]
        es1  = df["ema_slow"].iloc[i - 1]
        ef2  = df["ema_fast"].iloc[i - 2]
        es2  = df["ema_slow"].iloc[i - 2]
        atr  = df["atr"].iloc[i]
        cl   = df["Close"].iloc[i]
        op   = df["Open"].iloc[i]
        cl1  = df["Close"].iloc[i - 1]
        op1  = df["Open"].iloc[i - 1]

        gap_now  = abs(ef  - es)
        gap_prev = abs(ef1 - es1)
        gap_pp   = abs(ef2 - es2)

        converging       = gap_now < gap_prev < gap_pp
        rapid_conv       = (gap_pp - gap_now) > 0.05 * atr
        body             = cl - op
        body1            = cl1 - op1
        strong_bull      = body  > 0.45 * atr and body1 > 0
        strong_bear      = body  < -0.45 * atr and body1 < 0
        midpoint         = (ef + es) / 2

        # Below slow but fast approaching from below → anticipate bullish cross
        if ef < es and converging and rapid_conv and strong_bull and cl > midpoint:
            signals.iloc[i] = "buy"

        # Above slow but fast approaching from above → anticipate bearish cross
        elif ef > es and converging and rapid_conv and strong_bear and cl < midpoint:
            signals.iloc[i] = "sell"

    df["signal"] = signals
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SL / TARGET CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    if "atr" not in df.columns:
        df = df.copy()
        df["atr"] = calc_atr(df)
    return df


def calc_sl(df: pd.DataFrame, idx: int, trade_type: str, sl_type: str,
            custom_pts: float = 10.0, atr_mult: float = 2.0,
            rr_ratio: float = 2.0) -> float:
    df = _ensure_atr(df)
    price = float(df["Close"].iloc[idx])
    atr   = float(df["atr"].iloc[idx])
    sign  = -1 if trade_type == "buy" else +1   # SL direction

    if sl_type == "Custom Points":
        return round(price + sign * custom_pts, 2)

    elif sl_type == "ATR Based":
        return round(price + sign * atr * atr_mult, 2)

    elif sl_type in ("Risk Reward Based", "EMA Reverse Crossover"):
        # For RR: SL will be set such that risk = target/rr_ratio (placeholder)
        return round(price + sign * atr * atr_mult, 2)

    elif sl_type in ("Trailing SL", "Trailing SL – Swing Low/High",
                     "Trailing SL – Candle Low/High"):
        if trade_type == "buy":
            return round(float(df["Low"].iloc[idx]) - atr * 0.25, 2)
        else:
            return round(float(df["High"].iloc[idx]) + atr * 0.25, 2)

    elif sl_type == "Auto SL":
        lb = min(20, idx)
        if trade_type == "buy":
            return round(float(df["Low"].iloc[max(0, idx - lb):idx + 1].min()) - atr * 0.2, 2)
        else:
            return round(float(df["High"].iloc[max(0, idx - lb):idx + 1].max()) + atr * 0.2, 2)

    elif sl_type == "Volatility Based":
        std = float(df["Close"].iloc[max(0, idx - 20):idx + 1].std())
        return round(price + sign * std * 2.0, 2)

    elif sl_type == "Nearest Support/Resistance":
        sup, res = nearest_support_resistance(df.iloc[:idx + 1])
        if trade_type == "buy":
            return round(sup - atr * 0.1, 2)
        else:
            return round(res + atr * 0.1, 2)

    return round(price + sign * custom_pts, 2)


def calc_target(df: pd.DataFrame, idx: int, trade_type: str, target_type: str,
                entry: float, sl: float,
                custom_pts: float = 20.0, atr_mult: float = 3.0,
                rr_ratio: float = 2.0) -> float:
    df = _ensure_atr(df)
    atr  = float(df["atr"].iloc[idx])
    sign = +1 if trade_type == "buy" else -1   # Target direction

    if target_type == "Custom Points":
        return round(entry + sign * custom_pts, 2)

    elif target_type == "ATR Based":
        return round(entry + sign * atr * atr_mult, 2)

    elif target_type == "Risk Reward Based":
        risk = abs(entry - sl)
        return round(entry + sign * risk * rr_ratio, 2)

    elif target_type in ("Trailing Target (Display Only)",
                         "Trailing Target – Swing Low/High",
                         "Trailing Target – Candle Low/High"):
        return round(entry + sign * atr * atr_mult, 2)

    elif target_type == "EMA Crossover Exit":
        return round(entry + sign * atr * 2.5, 2)

    elif target_type == "Auto Target":
        sup, res = nearest_support_resistance(df.iloc[:idx + 1])
        raw_tgt  = res if trade_type == "buy" else sup
        min_tgt  = entry + sign * abs(entry - sl) * 1.5
        if trade_type == "buy":
            return round(max(raw_tgt, min_tgt), 2)
        else:
            return round(min(raw_tgt, min_tgt), 2)

    elif target_type == "Volatility Based":
        std = float(df["Close"].iloc[max(0, idx - 20):idx + 1].std())
        return round(entry + sign * std * 3.0, 2)

    elif target_type == "Nearest Support/Resistance":
        sup, res = nearest_support_resistance(df.iloc[:idx + 1])
        return round(res if trade_type == "buy" else sup, 2)

    return round(entry + sign * custom_pts, 2)


def update_trailing_sl(current_sl: float, ltp: float, trade_type: str,
                        sl_type: str, df: pd.DataFrame, idx: int,
                        atr_mult: float = 2.0) -> float:
    """Ratchet trailing SL – only moves in favour of the trade."""
    if "atr" not in df.columns:
        df = df.copy(); df["atr"] = calc_atr(df)
    atr = float(df["atr"].iloc[idx])

    if "Swing Low/High" in sl_type:
        lb = min(10, idx)
        if trade_type == "buy":
            new_sl = float(df["Low"].iloc[max(0, idx - lb):idx + 1].min()) - atr * 0.1
            return max(current_sl, new_sl)
        else:
            new_sl = float(df["High"].iloc[max(0, idx - lb):idx + 1].max()) + atr * 0.1
            return min(current_sl, new_sl)

    elif "Candle Low/High" in sl_type:
        if trade_type == "buy":
            new_sl = float(df["Low"].iloc[idx]) - atr * 0.05
            return max(current_sl, new_sl)
        else:
            new_sl = float(df["High"].iloc[idx]) + atr * 0.05
            return min(current_sl, new_sl)

    elif sl_type == "Trailing SL":
        if trade_type == "buy":
            new_sl = ltp - atr * atr_mult
            return max(current_sl, new_sl)
        else:
            new_sl = ltp + atr * atr_mult
            return min(current_sl, new_sl)

    return current_sl


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, cfg: dict) -> tuple[list, list, float, pd.DataFrame]:
    """
    Full conservative backtesting engine.

    BUY  trades: check SL vs candle Low FIRST → then Target vs candle High
    SELL trades: check SL vs candle High FIRST → then Target vs candle Low

    EMA/Anticipatory strategies: entry on candle N+1 open (signal on candle N).
    Simple Buy/Sell: entry immediately at current close.

    Returns (trades, violations, accuracy_pct, df_with_signals)
    """
    strategy      = cfg["strategy"]
    trade_filter  = cfg["trade_filter"]
    sl_type       = cfg["sl_type"]
    target_type   = cfg["target_type"]
    quantity      = cfg["quantity"]
    fast          = cfg["fast_ema"]
    slow          = cfg["slow_ema"]
    custom_sl     = cfg["custom_sl"]
    custom_target = cfg["custom_target"]
    atr_mult_sl   = cfg.get("atr_mult_sl", 2.0)
    atr_mult_tgt  = cfg.get("atr_mult_target", 3.0)
    rr_ratio      = cfg.get("rr_ratio", 2.0)
    xover_type    = cfg.get("xover_type", "Simple Crossover")
    candle_size   = cfg.get("candle_size", 10.0)
    min_angle     = cfg.get("min_angle", 0.0)
    check_angle   = cfg.get("check_angle", False)

    # ── Generate signals ──
    if strategy == "EMA Crossover":
        df = generate_ema_crossover_signals(df, fast, slow, xover_type,
                                            candle_size, min_angle, check_angle)
    elif strategy == "Anticipatory EMA Crossover":
        df = generate_anticipatory_ema_signals(df, fast, slow)
        if "ema_fast" not in df.columns:
            df = _add_ema_cols(df, fast, slow)
    elif strategy == "Elliott Wave (Auto)":
        df = _add_ema_cols(df, fast, slow)
        ew  = detect_elliott_waves(df)
        df["signal"] = ""
        if ew.get("signal") and len(df) >= 2:
            df.iloc[-2, df.columns.get_loc("signal")] = ew["signal"]
    elif strategy == "Simple Buy":
        df = _add_ema_cols(df, fast, slow)
        df["signal"] = ""
        df.iloc[0, df.columns.get_loc("signal")] = "buy"
    elif strategy == "Simple Sell":
        df = _add_ema_cols(df, fast, slow)
        df["signal"] = ""
        df.iloc[0, df.columns.get_loc("signal")] = "sell"
    else:
        df["signal"] = ""

    if "atr" not in df.columns:
        df["atr"] = calc_atr(df)

    immediate_entry = strategy in ("Simple Buy", "Simple Sell")

    trades, violations = [], []
    position = None

    for i in range(len(df)):
        row = df.iloc[i]

        # ── Check open position ──
        if position is not None:
            c_low   = float(row["Low"])
            c_high  = float(row["High"])
            c_close = float(row["Close"])
            exit_price  = None
            exit_reason = None
            violated    = False

            if position["type"] == "buy":
                sl_hit  = c_low  <= position["sl"]
                tgt_hit = c_high >= position["target"]

                if sl_hit and tgt_hit:
                    # Both in same candle → conservative = SL wins
                    exit_price  = position["sl"]
                    exit_reason = "SL Hit (candle low) – VIOLATION: Target also reachable same candle"
                    violated    = True
                elif sl_hit:
                    exit_price  = position["sl"]
                    exit_reason = "SL Hit (candle low)"
                elif tgt_hit:
                    exit_price  = position["target"]
                    exit_reason = "Target Hit (candle high)"

            else:  # sell
                sl_hit  = c_high >= position["sl"]
                tgt_hit = c_low  <= position["target"]

                if sl_hit and tgt_hit:
                    exit_price  = position["sl"]
                    exit_reason = "SL Hit (candle high) – VIOLATION: Target also reachable same candle"
                    violated    = True
                elif sl_hit:
                    exit_price  = position["sl"]
                    exit_reason = "SL Hit (candle high)"
                elif tgt_hit:
                    exit_price  = position["target"]
                    exit_reason = "Target Hit (candle low)"

            # EMA Crossover Exit
            if exit_price is None and target_type == "EMA Crossover Exit" \
                    and "ema_fast" in df.columns and i > 0:
                ef_n = df["ema_fast"].iloc[i];   es_n = df["ema_slow"].iloc[i]
                ef_p = df["ema_fast"].iloc[i-1]; es_p = df["ema_slow"].iloc[i-1]
                if position["type"] == "buy"  and ef_p >= es_p and ef_n < es_n:
                    exit_price  = c_close; exit_reason = "EMA Reverse Cross Exit"
                if position["type"] == "sell" and ef_p <= es_p and ef_n > es_n:
                    exit_price  = c_close; exit_reason = "EMA Reverse Cross Exit"

            if exit_price is None:
                # Update trailing SL
                position["sl"] = update_trailing_sl(
                    position["sl"], c_close, position["type"],
                    sl_type, df, i, atr_mult_sl)
                continue

            pnl_pts = (exit_price - position["entry_price"]) \
                      * (1 if position["type"] == "buy" else -1)
            pnl = round(pnl_pts * quantity, 2)

            rec = {
                "entry_datetime": position["entry_time"],
                "exit_datetime":  df.index[i],
                "type":           position["type"],
                "entry_price":    round(position["entry_price"], 2),
                "exit_price":     round(exit_price, 2),
                "sl":             round(position["initial_sl"], 2),
                "target":         round(position["target"], 2),
                "candle_high":    round(c_high, 2),
                "candle_low":     round(c_low, 2),
                "entry_reason":   position["entry_reason"],
                "exit_reason":    exit_reason,
                "pnl":            pnl,
                "quantity":       quantity,
                "sl_violation":   violated,
            }
            trades.append(rec)
            if violated:
                violations.append(rec)
            position = None

        # ── Look for new signal ──
        if position is not None:
            continue

        sig = row.get("signal", "")
        if not sig:
            continue

        if trade_filter == "Buy Only"  and sig != "buy":  continue
        if trade_filter == "Sell Only" and sig != "sell": continue

        # Determine entry candle
        if immediate_entry:
            entry_idx   = i
            entry_price = float(row["Close"])
        else:
            entry_idx = i + 1
            if entry_idx >= len(df):
                continue
            entry_price = float(df["Open"].iloc[entry_idx])

        # Calculate SL & Target
        sl     = calc_sl(df, entry_idx, sig, sl_type, custom_sl, atr_mult_sl, rr_ratio)
        target = calc_target(df, entry_idx, sig, target_type, entry_price,
                             sl, custom_target, atr_mult_tgt, rr_ratio)

        # For Elliott Wave: use auto SL/target if available
        if strategy == "Elliott Wave (Auto)":
            ew = detect_elliott_waves(df.iloc[:entry_idx + 1])
            if ew.get("auto_sl"):     sl     = ew["auto_sl"]
            if ew.get("auto_target"): target = ew["auto_target"]

        position = {
            "entry_time":   df.index[entry_idx],
            "entry_price":  entry_price,
            "type":         sig,
            "sl":           sl,
            "initial_sl":   sl,
            "target":       target,
            "entry_reason": f"{strategy} Signal",
        }

    # Close any leftover position at end of data
    if position is not None:
        last  = df.iloc[-1]
        close = float(last["Close"])
        pnl   = round((close - position["entry_price"])
                      * (1 if position["type"] == "buy" else -1) * quantity, 2)
        trades.append({
            "entry_datetime": position["entry_time"],
            "exit_datetime":  df.index[-1],
            "type":           position["type"],
            "entry_price":    round(position["entry_price"], 2),
            "exit_price":     round(close, 2),
            "sl":             round(position["initial_sl"], 2),
            "target":         round(position["target"], 2),
            "candle_high":    round(float(last["High"]), 2),
            "candle_low":     round(float(last["Low"]), 2),
            "entry_reason":   position["entry_reason"],
            "exit_reason":    "End of Data",
            "pnl":            pnl,
            "quantity":       quantity,
            "sl_violation":   False,
        })

    winners  = sum(1 for t in trades if t["pnl"] > 0)
    accuracy = (winners / len(trades) * 100) if trades else 0.0

    return trades, violations, accuracy, df


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#050A14",
    plot_bgcolor="#080F20",
    font=dict(family="JetBrains Mono", color="#7a9fc0", size=11),
    legend=dict(bgcolor="rgba(5,10,20,0.8)", bordercolor="#0f2040", borderwidth=1),
)


def build_backtest_chart(df: pd.DataFrame, trades: list,
                          fast: int = 9, slow: int = 15) -> go.Figure:
    df = df.tail(800)  # limit display

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.78, 0.22],
                         vertical_spacing=0.02)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#00c896", decreasing_line_color="#ff4d6d",
        increasing_fillcolor="#00c896", decreasing_fillcolor="#ff4d6d",
        line=dict(width=1),
    ), row=1, col=1)

    # EMAs
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"],
                                  name=f"EMA {fast}",
                                  line=dict(color="#FF9F43", width=1.8)), row=1, col=1)
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"],
                                  name=f"EMA {slow}",
                                  line=dict(color="#4ECDC4", width=1.8)), row=1, col=1)

    # Trade markers
    for t in trades:
        clr_e = "#00e5a0" if t["type"] == "buy" else "#ff4d6d"
        sym_e = "triangle-up" if t["type"] == "buy" else "triangle-down"
        clr_x = "#00c896" if t["pnl"] > 0 else "#ff4d6d"

        fig.add_trace(go.Scatter(
            x=[t["entry_datetime"]], y=[t["entry_price"]],
            mode="markers",
            marker=dict(symbol=sym_e, size=13, color=clr_e,
                        line=dict(color="white", width=1)),
            name="Entry", showlegend=False,
            hovertemplate=(f"<b>{'BUY' if t['type']=='buy' else 'SELL'} ENTRY</b><br>"
                           f"Price: {t['entry_price']}<br>"
                           f"SL: {t['sl']}<br>Target: {t['target']}<extra></extra>"),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[t["exit_datetime"]], y=[t["exit_price"]],
            mode="markers",
            marker=dict(symbol="x", size=10, color=clr_x,
                        line=dict(color="white", width=1)),
            name="Exit", showlegend=False,
            hovertemplate=(f"<b>EXIT</b><br>Price: {t['exit_price']}<br>"
                           f"PnL: {t['pnl']}<br>Reason: {t['exit_reason']}<extra></extra>"),
        ), row=1, col=1)

        # SL / Target lines between entry & exit
        fig.add_shape(type="line",
                       x0=t["entry_datetime"], x1=t["exit_datetime"],
                       y0=t["sl"], y1=t["sl"],
                       line=dict(color="#ff4d6d", width=1, dash="dot"),
                       row=1, col=1)
        fig.add_shape(type="line",
                       x0=t["entry_datetime"], x1=t["exit_datetime"],
                       y0=t["target"], y1=t["target"],
                       line=dict(color="#00c896", width=1, dash="dot"),
                       row=1, col=1)

    # Volume
    vol_colors = ["#00c896" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i])
                  else "#ff4d6d" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                          marker_color=vol_colors, name="Volume"), row=2, col=1)

    fig.update_layout(
        height=680, xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        **_DARK,
    )
    fig.update_xaxes(gridcolor="#0f2040", showgrid=True)
    fig.update_yaxes(gridcolor="#0f2040", showgrid=True)
    return fig


def build_live_chart(df: pd.DataFrame, position: dict | None,
                      fast: int = 9, slow: int = 15) -> go.Figure:
    df = df.tail(300)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.78, 0.22], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#00c896", decreasing_line_color="#ff4d6d",
        line=dict(width=1),
    ), row=1, col=1)

    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"],
                                  name=f"EMA {fast}",
                                  line=dict(color="#FF9F43", width=2)), row=1, col=1)
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"],
                                  name=f"EMA {slow}",
                                  line=dict(color="#4ECDC4", width=2)), row=1, col=1)

    if position:
        ep  = position.get("entry_price", 0)
        sl  = position.get("sl", 0)
        tgt = position.get("target", 0)
        clr = "#00c896" if position["type"] == "buy" else "#ff4d6d"
        sym = "triangle-up" if position["type"] == "buy" else "triangle-down"

        fig.add_trace(go.Scatter(
            x=[position["entry_time"]], y=[ep], mode="markers",
            marker=dict(symbol=sym, size=14, color=clr, line=dict(color="white", width=1)),
            name="Entry", showlegend=False,
        ), row=1, col=1)

        x0, x1 = df.index[0], df.index[-1]
        for y_val, color, label in [(sl, "#ff4d6d", f"SL {sl:.2f}"),
                                     (tgt, "#00c896", f"TGT {tgt:.2f}"),
                                     (ep, "#FFD93D", f"Entry {ep:.2f}")]:
            fig.add_hline(y=y_val, line_color=color, line_dash="dash",
                           annotation_text=label,
                           annotation_font_color=color, row=1, col=1)

    vol_colors = ["#00c896" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i])
                  else "#ff4d6d" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                          marker_color=vol_colors, name="Volume"), row=2, col=1)

    fig.update_layout(
        height=580, xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
        **_DARK,
    )
    fig.update_xaxes(gridcolor="#0f2040")
    fig.update_yaxes(gridcolor="#0f2040")
    return fig


def build_pnl_chart(trades: list) -> go.Figure:
    cumul = []
    running = 0.0
    for t in trades:
        running += t["pnl"]
        cumul.append(running)

    clrs = ["#00c896" if v >= 0 else "#ff4d6d" for v in cumul]
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        y=cumul, mode="lines+markers",
        line=dict(color="#4ECDC4", width=2),
        marker=dict(color=clrs, size=7),
        fill="tozeroy", fillcolor="rgba(0,200,150,0.06)",
        name="Cumulative P&L",
    ))
    fig.add_hline(y=0, line_color="#3a5a78", line_dash="dot")
    fig.update_layout(
        title="Cumulative P&L", height=260,
        margin=dict(l=0, r=0, t=35, b=0),
        **_DARK,
    )
    fig.update_xaxes(gridcolor="#0f2040")
    fig.update_yaxes(gridcolor="#0f2040")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DHAN BROKER INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
_dhan_hq  = None   # dhanhq instance (options-grade API)


def init_dhan_client(client_id: str, access_token: str) -> bool:
    global _dhan_hq
    try:
        from dhanhq import dhanhq
        _dhan_hq = dhanhq(client_id, access_token)
        return True
    except ImportError:
        return False
    except Exception:
        return False


def register_sebi_ip(client_id: str, access_token: str) -> str:
    """SEBI mandatory: register current public IP with Dhan before placing orders."""
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
        headers = {
            "access-token": access_token,
            "client-id":    client_id,
            "Content-Type": "application/json",
        }
        r = requests.post("https://api.dhan.co/edis/form",
                          headers=headers, json={"ip": ip}, timeout=10)
        return f"✅ IP {ip} registered (HTTP {r.status_code})"
    except Exception as exc:
        return f"⚠️ IP registration: {exc}"


def _place_equity(security_id: str, txn: str, qty: int, product: str,
                   order_type: str, exchange: str, price: float) -> dict:
    """Place intraday/delivery order. Falls back to simulation if dhanhq unavailable."""
    seg_map = {"NSE": "NSE_EQ", "BSE": "BSE_EQ"}
    if _dhan_hq:
        try:
            return _dhan_hq.place_order(
                transactionType=txn,
                exchangeSegment=seg_map.get(exchange, "NSE_EQ"),
                productType="INTRADAY" if product == "INTRADAY" else "CNC",
                orderType=order_type,
                validity="DAY",
                securityId=str(security_id),
                quantity=int(qty),
                price=float(price) if order_type == "LIMIT" else 0,
                triggerPrice=0,
            )
        except Exception as exc:
            return {"error": str(exc)}
    # Simulation
    return {"status": "simulated",
            "message": f"{txn} {qty}×{security_id} @ {'MKT' if order_type=='MARKET' else price}"}


def _place_options(security_id: str, txn: str, qty: int, segment: str,
                    order_type: str, price: float) -> dict:
    if _dhan_hq:
        try:
            return _dhan_hq.place_order(
                transactionType=txn,
                exchangeSegment=segment,
                productType="INTRADAY",
                orderType=order_type,
                validity="DAY",
                securityId=str(security_id),
                quantity=int(qty),
                price=float(price) if order_type == "LIMIT" else 0,
                triggerPrice=0,
            )
        except Exception as exc:
            return {"error": str(exc)}
    return {"status": "simulated",
            "message": f"OPTIONS {txn} {qty}×{security_id} @ {'MKT' if order_type=='MARKET' else price}"}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRADING ENGINE  (background thread)
# ─────────────────────────────────────────────────────────────────────────────
def _log_live(msg: str):
    now = datetime.now(IST).strftime("%H:%M:%S")
    ts_append("log", f"[{now}]  {msg}")


def live_trading_engine(cfg: dict):
    """
    Background thread – runs until ts_get('running') is False.

    Key rules:
    • Rate-limited data fetch: ≥1.5 s between requests.
    • EMA signals evaluated only at exact candle boundaries (multiples of TF).
    • SL/Target monitored every tick (each fetch cycle) vs LTP.
    • Simple Buy/Sell: immediate entry on current LTP, no candle boundary check.
    • No cooldown in backtest; cooldown applied here.
    • No overlapping trades if flag set.
    """
    ticker       = cfg["ticker"]
    tf           = cfg["timeframe"]
    strategy     = cfg["strategy"]
    fast         = cfg["fast_ema"]
    slow         = cfg["slow_ema"]
    sl_type      = cfg["sl_type"]
    target_type  = cfg["target_type"]
    custom_sl    = cfg["custom_sl"]
    custom_tgt   = cfg["custom_target"]
    atr_sl       = cfg.get("atr_mult_sl", 2.0)
    atr_tgt      = cfg.get("atr_mult_target", 3.0)
    rr_ratio     = cfg.get("rr_ratio", 2.0)
    qty          = cfg["quantity"]
    cooldown_on  = cfg.get("cooldown_enabled", True)
    cooldown_s   = cfg.get("cooldown_secs", 5)
    no_overlap   = cfg.get("no_overlap", True)
    dhan_on      = cfg.get("dhan_enabled", False)
    opts_on      = cfg.get("options_enabled", False)
    tf_min       = TF_MINUTES.get(tf, 5)
    immediate    = strategy in ("Simple Buy", "Simple Sell")

    ts_set("running", True)
    ts_set("position", None)
    ts_clear("log")
    ts_clear("completed_trades")
    ts_set("last_trade_time", None)

    _log_live("🚀 Live trading STARTED")
    _log_live(f"   {ticker} | {tf} | {strategy}")
    _log_live(f"   SL: {sl_type} | Target: {target_type}")

    last_signal_candle = None
    last_fetch_time    = 0.0

    while ts_get("running"):
        try:
            now  = datetime.now(IST)
            wait = max(0.0, 1.5 - (time.time() - last_fetch_time))
            if wait > 0:
                time.sleep(wait)

            # ── Fetch data ──
            fetch_period = "5d" if tf in ("1m", "5m", "15m") else "1mo"
            df = fetch_data(ticker, tf, fetch_period, warmup=True)
            last_fetch_time = time.time()

            if df is None or len(df) < 20:
                _log_live("⚠️ Insufficient data – retrying…")
                time.sleep(3)
                continue

            df["ema_fast"] = calc_ema(df["Close"], fast)
            df["ema_slow"] = calc_ema(df["Close"], slow)
            df["atr"]      = calc_atr(df)

            ltp = float(df["Close"].iloc[-1])
            ts_set("ltp",          ltp)
            ts_set("df",           df)
            ts_set("last_candle",  df.iloc[-1])
            ts_set("ema_fast_val", float(df["ema_fast"].iloc[-1]))
            ts_set("ema_slow_val", float(df["ema_slow"].iloc[-1]))
            ts_set("atr_val",      float(df["atr"].iloc[-1]))

            if strategy == "Elliott Wave (Auto)":
                ts_set("elliott_status", detect_elliott_waves(df))

            # ── Monitor open position (SL/Target vs LTP every tick) ──
            position = ts_get("position")
            if position is not None:
                sl  = position["sl"]
                tgt = position["target"]
                tt  = position["type"]

                sl_hit  = (ltp <= sl)  if tt == "buy"  else (ltp >= sl)
                tgt_hit = (ltp >= tgt) if tt == "buy"  else (ltp <= tgt)

                exit_price  = None
                exit_reason = None

                # Conservative: SL first, then target
                if sl_hit:
                    exit_price  = sl
                    exit_reason = "SL Hit (LTP)"
                elif tgt_hit and "Display Only" not in target_type:
                    exit_price  = tgt
                    exit_reason = "Target Hit (LTP)"

                # EMA crossover exit
                if exit_price is None and target_type == "EMA Crossover Exit" and len(df) > 2:
                    ef_n = float(df["ema_fast"].iloc[-1]); es_n = float(df["ema_slow"].iloc[-1])
                    ef_p = float(df["ema_fast"].iloc[-2]); es_p = float(df["ema_slow"].iloc[-2])
                    if tt == "buy"  and ef_p >= es_p and ef_n < es_n:
                        exit_price = ltp; exit_reason = "EMA Reverse Cross"
                    if tt == "sell" and ef_p <= es_p and ef_n > es_n:
                        exit_price = ltp; exit_reason = "EMA Reverse Cross"

                if exit_price is not None:
                    pnl = round((exit_price - position["entry_price"])
                                * (1 if tt == "buy" else -1) * qty, 2)
                    rec = {
                        "entry_datetime": position["entry_time"],
                        "exit_datetime":  now,
                        "type":           tt,
                        "entry_price":    position["entry_price"],
                        "exit_price":     exit_price,
                        "sl":             position["initial_sl"],
                        "target":         position["target"],
                        "entry_reason":   position["entry_reason"],
                        "exit_reason":    exit_reason,
                        "pnl":            pnl,
                        "quantity":       qty,
                    }
                    ts_append("completed_trades", rec)
                    ts_set("position", None)
                    emoji = "✅" if pnl >= 0 else "🔴"
                    _log_live(f"{emoji} EXIT {tt.upper()} @ {exit_price:.2f} | P&L ₹{pnl:.2f} | {exit_reason}")

                    # Dhan exit order
                    if dhan_on:
                        e_txn = "SELL" if tt == "buy" else "BUY"
                        if opts_on:
                            sec = cfg.get("ce_security_id" if tt == "buy" else "pe_security_id", "")
                            _place_options(sec, "SELL", cfg.get("options_qty", 65),
                                           cfg.get("exchange_segment", "NSE_FNO"),
                                           cfg.get("exit_order_type", "MARKET"), ltp)
                        else:
                            _place_equity(cfg.get("security_id","1594"), e_txn,
                                          qty, cfg.get("product_type","INTRADAY"),
                                          cfg.get("exit_order_type","MARKET"),
                                          cfg.get("exchange","NSE"), ltp)
                else:
                    # Trail SL
                    new_sl = update_trailing_sl(sl, ltp, tt, sl_type, df, len(df)-1, atr_sl)
                    if new_sl != sl:
                        position["sl"] = new_sl
                        ts_set("position", position)

                continue  # don't look for new entry while in position

            # ── Determine if we are at a candle boundary ──
            current_candle = df.index[-2]   # signal on previous candle (closed)

            if immediate:
                is_boundary = True
            else:
                if tf_min < 1440:
                    total_min = now.hour * 60 + now.minute
                    is_boundary = (total_min % tf_min == 0 and now.second < 12
                                   and last_signal_candle != current_candle)
                else:
                    is_boundary = last_signal_candle != current_candle

            if not is_boundary:
                time.sleep(1.5)
                continue

            last_signal_candle = current_candle

            # ── Cooldown check ──
            if cooldown_on:
                last_tt = ts_get("last_trade_time")
                if last_tt:
                    if (now - last_tt).total_seconds() < cooldown_s:
                        _log_live(f"⏳ Cooldown ({cooldown_s}s)…")
                        continue

            # ── No-overlap check ──
            if no_overlap:
                completed = ts_get("completed_trades") or []
                if completed:
                    lt = completed[-1]
                    xt = lt.get("exit_datetime")
                    if xt and isinstance(xt, datetime):
                        if xt.tzinfo is None:
                            xt = IST.localize(xt)
                        if (now - xt).total_seconds() < 30:
                            continue

            # ── Generate signal ──
            signal      = None
            entry_price = ltp

            if strategy == "EMA Crossover":
                ef_n = float(df["ema_fast"].iloc[-2]); es_n = float(df["ema_slow"].iloc[-2])
                ef_p = float(df["ema_fast"].iloc[-3]); es_p = float(df["ema_slow"].iloc[-3])
                if ef_p <= es_p and ef_n > es_n:
                    signal = "buy"
                elif ef_p >= es_p and ef_n < es_n:
                    signal = "sell"
                # Entry on N+1 (current open)
                if signal:
                    entry_price = float(df["Open"].iloc[-1])

            elif strategy == "Anticipatory EMA Crossover":
                sig_df = generate_anticipatory_ema_signals(df, fast, slow)
                s = sig_df["signal"].iloc[-2]
                if s in ("buy", "sell"):
                    signal      = s
                    entry_price = float(df["Open"].iloc[-1])

            elif strategy == "Elliott Wave (Auto)":
                ew = ts_get("elliott_status") or {}
                if ew.get("signal"):
                    signal      = ew["signal"]
                    entry_price = ltp

            elif strategy == "Simple Buy":
                signal = "buy"; entry_price = ltp
            elif strategy == "Simple Sell":
                signal = "sell"; entry_price = ltp

            if not signal:
                continue

            # ── Trade direction filter ──
            tf_filter = cfg.get("trade_filter", "Both")
            if tf_filter == "Buy Only"  and signal != "buy":  continue
            if tf_filter == "Sell Only" and signal != "sell": continue

            # ── Compute SL / Target ──
            sl     = calc_sl(df, len(df)-1, signal, sl_type, custom_sl, atr_sl, rr_ratio)
            target = calc_target(df, len(df)-1, signal, target_type, entry_price,
                                 sl, custom_tgt, atr_tgt, rr_ratio)

            if strategy == "Elliott Wave (Auto)":
                ew = ts_get("elliott_status") or {}
                if ew.get("auto_sl"):     sl     = ew["auto_sl"]
                if ew.get("auto_target"): target = ew["auto_target"]

            position = {
                "entry_time":   now,
                "entry_price":  entry_price,
                "type":         signal,
                "sl":           sl,
                "initial_sl":   sl,
                "target":       target,
                "entry_reason": f"{strategy} Signal",
                "quantity":     qty,
            }
            ts_set("position", position)
            ts_set("last_trade_time", now)
            _log_live(f"📈 ENTRY {signal.upper()} @ {entry_price:.2f} | SL {sl:.2f} | TGT {target:.2f}")

            # ── Dhan entry order ──
            if dhan_on:
                if opts_on:
                    if signal == "buy":
                        sec = cfg.get("ce_security_id", "")
                    else:
                        sec = cfg.get("pe_security_id", "")
                    r = _place_options(sec, "BUY", cfg.get("options_qty", 65),
                                       cfg.get("exchange_segment", "NSE_FNO"),
                                       cfg.get("entry_order_type", "MARKET"), ltp)
                else:
                    txn = "BUY" if signal == "buy" else "SELL"
                    r = _place_equity(cfg.get("security_id","1594"), txn,
                                      qty, cfg.get("product_type","INTRADAY"),
                                      cfg.get("entry_order_type","LIMIT"),
                                      cfg.get("exchange","NSE"), ltp)
                _log_live(f"   Order: {r.get('status','?')} – {r.get('message',r.get('error',''))}")

        except Exception as exc:
            _log_live(f"❌ Engine error: {exc}")
            time.sleep(3)

    _log_live("🛑 Live trading STOPPED")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Metric Card HTML
# ─────────────────────────────────────────────────────────────────────────────
def mcard(label: str, value: str, color: str = "#e8f4ff") -> str:
    return (f'<div class="m-card">'
            f'<p class="mc-label">{label}</p>'
            f'<p class="mc-val" style="color:{color}">{value}</p>'
            f'</div>')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session()

    # ═══════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:16px 8px 8px;">
            <div style="font-size:2.2rem">📈</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.25rem;
                        font-weight:800; color:#00E5CC; letter-spacing:0.06em;">
                SMART INVESTING
            </div>
            <div style="color:#3a5a78; font-size:0.72rem; letter-spacing:0.12em;
                        text-transform:uppercase; margin-top:2px;">
                Algorithmic Trading Platform
            </div>
        </div>
        <hr style="border-color:#0f2040; margin:10px 0 14px;">
        """, unsafe_allow_html=True)

        # ── Market Setup ──
        with st.expander("📊  Market Setup", expanded=True):
            ticker_label = st.selectbox("Instrument", list(PRESET_TICKERS.keys()),
                                         key="sb_ticker_label")
            if ticker_label == "Custom":
                selected_ticker = st.text_input("Ticker Symbol", "RELIANCE.NS",
                                                 key="sb_custom_ticker")
            else:
                selected_ticker = PRESET_TICKERS[ticker_label]

            timeframe = st.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()),
                                      index=2, key="sb_tf")
            period    = st.selectbox("Period", TIMEFRAME_PERIODS[timeframe], key="sb_period")
            quantity  = st.number_input("Quantity", min_value=1, value=1, step=1, key="sb_qty")

        # ── Strategy ──
        with st.expander("🎯  Strategy", expanded=True):
            strategy      = st.selectbox("Strategy", STRATEGIES, key="sb_strat")
            trade_filter  = st.selectbox("Direction", ["Both","Buy Only","Sell Only"],
                                          key="sb_direction")

            fast_ema = slow_ema = 9
            xover_type = "Simple Crossover"
            candle_size = check_angle = min_angle = None

            if strategy in ("EMA Crossover", "Anticipatory EMA Crossover"):
                c1, c2 = st.columns(2)
                fast_ema = c1.number_input("Fast EMA", 2, 200, 9,  key="sb_fast")
                slow_ema = c2.number_input("Slow EMA", 3, 500, 15, key="sb_slow")

            if strategy == "EMA Crossover":
                xover_type = st.selectbox("Crossover Filter",
                    ["Simple Crossover","Custom Candle Size","ATR Based Candle Size"],
                    key="sb_xover")
                if xover_type == "Custom Candle Size":
                    candle_size = st.number_input("Min Candle Body (pts)", value=10.0,
                                                   key="sb_csize")
                else:
                    candle_size = 10.0

                check_angle = st.checkbox("Crossover Angle Filter", value=False,
                                           key="sb_angle_on")
                min_angle = st.number_input("Min Angle (abs, default 0)",
                                             value=0.0, step=0.1, key="sb_angle") \
                            if check_angle else 0.0
            else:
                candle_size = 10.0; check_angle = False; min_angle = 0.0

        # ── SL & Target ──
        with st.expander("🛡️  Stop Loss & Target", expanded=True):
            sl_type     = st.selectbox("SL Type",     SL_TYPES,     key="sb_sl_type")
            target_type = st.selectbox("Target Type", TARGET_TYPES, key="sb_tgt_type")

            custom_sl  = st.number_input("Custom SL (pts)",     value=10.0, key="sb_csl") \
                         if sl_type == "Custom Points" else 10.0
            custom_tgt = st.number_input("Custom Target (pts)", value=20.0, key="sb_ctgt") \
                         if target_type == "Custom Points" else 20.0

            atr_mult_sl  = 2.0; atr_mult_tgt = 3.0
            if "ATR" in sl_type or "ATR" in target_type:
                c1, c2 = st.columns(2)
                atr_mult_sl  = c1.number_input("ATR×SL",  value=2.0, step=0.5, key="sb_atr_sl")
                atr_mult_tgt = c2.number_input("ATR×Tgt", value=3.0, step=0.5, key="sb_atr_tgt")

            rr_ratio = 2.0
            if "Risk Reward" in sl_type or "Risk Reward" in target_type:
                rr_ratio = st.number_input("Risk:Reward Ratio", value=2.0, step=0.5, key="sb_rr")

            book_profit = st.checkbox("Partial Book Profit at Target 1", value=False, key="sb_bp")
            book_pct    = st.number_input("% to book at T1", 10, 90, 50, key="sb_bppct") \
                          if book_profit else 50

        # ── Live Trading Settings ──
        with st.expander("⚡  Live Settings", expanded=False):
            cooldown_on  = st.checkbox("Cooldown between trades", value=True, key="sb_cd_on")
            cooldown_s   = st.number_input("Cooldown (seconds)", value=5, min_value=1,
                                            key="sb_cd_s") if cooldown_on else 5
            no_overlap   = st.checkbox("Prevent overlapping trades", value=True, key="sb_noovlp")

        # ── Dhan Broker ──
        with st.expander("🏦  Dhan Broker", expanded=False):
            dhan_enabled = st.checkbox("Enable Dhan Broker", value=False, key="sb_dhan")

            client_id    = access_token = ""
            opts_on      = False
            security_id  = "1594"; product_type = "INTRADAY"; exchange = "NSE"
            entry_order  = "LIMIT"; exit_order   = "MARKET"
            ce_sec_id = pe_sec_id = ""; opts_qty = 65
            exc_segment = "NSE_FNO"; opt_entry = "MARKET"; opt_exit = "MARKET"

            if dhan_enabled:
                client_id    = st.text_input("Client ID",     type="password", key="sb_cid")
                access_token = st.text_input("Access Token",  type="password", key="sb_tok")

                if st.button("🔑 Register IP (SEBI Mandatory)", key="sb_regip"):
                    if client_id and access_token:
                        init_dhan_client(client_id, access_token)
                        msg = register_sebi_ip(client_id, access_token)
                        st.info(msg)

                opts_on = st.checkbox("Options Trading", value=False, key="sb_opts")

                if not opts_on:
                    product_type = st.selectbox("Product",  ["INTRADAY","DELIVERY"], key="sb_prod")
                    exchange     = st.selectbox("Exchange", ["NSE","BSE"],            key="sb_exc")
                    security_id  = st.text_input("Security ID", value="1594",         key="sb_secid")
                    entry_order  = st.selectbox("Entry Order",  ["LIMIT","MARKET"],   key="sb_enty")
                    exit_order   = st.selectbox("Exit Order",   ["MARKET","LIMIT"],   key="sb_exit")
                else:
                    exc_segment  = st.selectbox("Segment", ["NSE_FNO","BSE_FNO"],    key="sb_seg")
                    ce_sec_id    = st.text_input("CE Security ID", value="",          key="sb_ce")
                    pe_sec_id    = st.text_input("PE Security ID", value="",          key="sb_pe")
                    opts_qty     = st.number_input("Options Qty", value=65, min_value=1, key="sb_oqty")
                    opt_entry    = st.selectbox("Entry Order", ["MARKET","LIMIT"],    key="sb_oenty")
                    opt_exit     = st.selectbox("Exit Order",  ["MARKET","LIMIT"],    key="sb_oexit")

    # ═══════════════════════════════════════════════════════
    # LTP BANNER  (top of every tab)
    # ═══════════════════════════════════════════════════════
    ltp_val  = ts_get("ltp")
    ltp_disp = f"₹{ltp_val:,.2f}" if ltp_val else "— —"
    now_ist  = datetime.now(IST).strftime("%d %b %Y  %H:%M:%S IST")

    st.markdown(f"""
    <div class="ltp-banner">
        <div>
            <div class="ticker-name">{selected_ticker}</div>
            <div class="ltp-val">{ltp_disp}</div>
        </div>
        <div style="text-align:right">
            <div class="ltp-time">{now_ist}</div>
            <div style="color:#1a4060; font-size:0.7rem; margin-top:3px;">yfinance delayed data</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════
    tab_bt, tab_lt, tab_hist, tab_opt = st.tabs(
        ["📊  Backtesting", "⚡  Live Trading", "📁  Trade History", "⚙️  Optimization"])

    # ─────────────────────────────────────────────────────
    # TAB 1  –  BACKTESTING
    # ─────────────────────────────────────────────────────
    with tab_bt:
        st.markdown("### 📊 Strategy Backtesting")
        st.caption("Conservative mode: SL checked first (via candle Low/High) before Target")

        run_btn = st.button("▶  Run Backtest", type="primary", key="run_bt")

        if run_btn:
            with st.spinner("Fetching data & running backtest…"):
                df_raw = fetch_data(selected_ticker, timeframe, period, warmup=True)

            if df_raw is None or df_raw.empty:
                st.error("❌ No data returned. Check ticker symbol / period.")
            else:
                bt_cfg = dict(
                    strategy=strategy,
                    trade_filter=trade_filter,
                    sl_type=sl_type,
                    target_type=target_type,
                    quantity=quantity,
                    fast_ema=fast_ema, slow_ema=slow_ema,
                    custom_sl=custom_sl, custom_target=custom_tgt,
                    atr_mult_sl=atr_mult_sl, atr_mult_target=atr_mult_tgt,
                    rr_ratio=rr_ratio,
                    xover_type=xover_type, candle_size=candle_size,
                    min_angle=min_angle, check_angle=check_angle,
                )
                with st.spinner("Running strategy…"):
                    trades, violations, accuracy, df_sig = run_backtest(df_raw.copy(), bt_cfg)

                st.session_state["backtest_results"] = (trades, violations, accuracy)
                st.session_state["backtest_df"]      = df_sig

        # ── Display results ──
        if st.session_state.get("backtest_results"):
            trades, violations, accuracy = st.session_state["backtest_results"]
            df_sig = st.session_state.get("backtest_df")

            if not trades:
                st.info("⚠️ No trades generated with current configuration. "
                         "Try a different period, timeframe, or EMA parameters.")
            else:
                total_pnl = sum(t["pnl"] for t in trades)
                winners   = sum(1 for t in trades if t["pnl"] > 0)
                losers    = len(trades) - winners
                avg_win   = sum(t["pnl"] for t in trades if t["pnl"] > 0) / max(winners, 1)
                avg_loss  = sum(t["pnl"] for t in trades if t["pnl"] <= 0) / max(losers, 1)
                best_t    = max(t["pnl"] for t in trades)
                worst_t   = min(t["pnl"] for t in trades)

                # Metrics row
                cols = st.columns(6)
                metric_data = [
                    ("Trades",     str(len(trades)),                "#e8f4ff"),
                    ("P&L",        f"₹{total_pnl:,.0f}",           "#00c896" if total_pnl>=0 else "#ff4d6d"),
                    ("Accuracy",   f"{accuracy:.1f}%",              "#FFD93D"),
                    ("Winners",    str(winners),                    "#00c896"),
                    ("Losers",     str(losers),                     "#ff4d6d"),
                    ("Violations", str(len(violations)),            "#FF9F43"),
                ]
                for col, (lbl, val, clr) in zip(cols, metric_data):
                    col.markdown(mcard(lbl, val, clr), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Chart
                if df_sig is not None:
                    fig_bt = build_backtest_chart(df_sig, trades, fast_ema, slow_ema)
                    st.plotly_chart(fig_bt, use_container_width=True)

                # P&L Curve
                st.plotly_chart(build_pnl_chart(trades), use_container_width=True)

                # Trade table
                st.markdown("#### 📋 Trade Log")
                tdf = pd.DataFrame(trades)
                for col in ("entry_datetime", "exit_datetime"):
                    if col in tdf.columns:
                        tdf[col] = pd.to_datetime(tdf[col]).dt.strftime("%Y-%m-%d %H:%M")

                def _style_row(row):
                    styles = []
                    for col in row.index:
                        if col == "pnl":
                            styles.append("color:#00c896;font-weight:600" if row[col] > 0
                                          else "color:#ff4d6d;font-weight:600")
                        elif col == "type":
                            styles.append("color:#00c896" if row[col] == "buy"
                                          else "color:#ff4d6d")
                        elif col == "sl_violation":
                            styles.append("color:#FF9F43;font-weight:700" if row[col] else "")
                        else:
                            styles.append("")
                    return styles

                st.dataframe(tdf.style.apply(_style_row, axis=1),
                              use_container_width=True, height=420)

                # Violation panel
                if violations:
                    st.markdown(f"""
                    <div class="violation-box">
                        ⚠️ <strong>{len(violations)} trades</strong> had both SL and Target
                        reachable within the same candle.<br>
                        Conservative approach applied: SL exit used (worst case for buyer,
                        best case for backtesting accuracy vs live).
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("📋 View Violation Trades"):
                        vdf = pd.DataFrame(violations)
                        st.dataframe(vdf, use_container_width=True)

                # Extended stats
                with st.expander("📈 Extended Statistics"):
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.markdown(mcard("Avg Win",  f"₹{avg_win:,.2f}",  "#00c896"), unsafe_allow_html=True)
                    sc2.markdown(mcard("Avg Loss", f"₹{avg_loss:,.2f}", "#ff4d6d"), unsafe_allow_html=True)
                    sc3.markdown(mcard("Best",     f"₹{best_t:,.2f}",   "#00c896"), unsafe_allow_html=True)
                    sc4.markdown(mcard("Worst",    f"₹{worst_t:,.2f}",  "#ff4d6d"), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # TAB 2  –  LIVE TRADING
    # ─────────────────────────────────────────────────────
    with tab_lt:
        st.markdown("### ⚡ Live Trading")

        is_running = bool(ts_get("running"))

        # Control buttons
        b1, b2, b3, b4 = st.columns([1, 1, 1, 3])

        with b1:
            if not is_running:
                if st.button("▶  START", type="primary", use_container_width=True, key="lt_start"):
                    live_cfg = dict(
                        ticker=selected_ticker,
                        timeframe=timeframe, period=period,
                        strategy=strategy,
                        trade_filter=trade_filter,
                        fast_ema=fast_ema, slow_ema=slow_ema,
                        sl_type=sl_type, target_type=target_type,
                        custom_sl=custom_sl, custom_target=custom_tgt,
                        atr_mult_sl=atr_mult_sl, atr_mult_target=atr_mult_tgt,
                        rr_ratio=rr_ratio,
                        quantity=quantity,
                        cooldown_enabled=cooldown_on, cooldown_secs=cooldown_s,
                        no_overlap=no_overlap,
                        dhan_enabled=dhan_enabled,
                        options_enabled=opts_on,
                        xover_type=xover_type, candle_size=candle_size,
                        min_angle=min_angle, check_angle=check_angle,
                        # Dhan equity
                        security_id=security_id,
                        product_type=product_type,
                        exchange=exchange,
                        entry_order_type=entry_order,
                        exit_order_type=exit_order,
                        # Dhan options
                        ce_security_id=ce_sec_id,
                        pe_security_id=pe_sec_id,
                        options_qty=opts_qty,
                        exchange_segment=exc_segment,
                        opt_entry_order=opt_entry,
                        opt_exit_order=opt_exit,
                    )
                    ts_set("config", live_cfg)
                    ts_set("running", True)
                    t = threading.Thread(target=live_trading_engine,
                                         args=(live_cfg,), daemon=True)
                    t.start()
                    st.session_state["live_thread"] = t
                    st.success("Live trading started ✅")
                    st.rerun()

        with b2:
            if is_running:
                if st.button("⏹  STOP", use_container_width=True, key="lt_stop"):
                    ts_set("running", False)
                    st.warning("Stopping…")
                    st.rerun()

        with b3:
            if st.button("🔄  Square Off", use_container_width=True, key="lt_sqoff"):
                pos = ts_get("position")
                if pos:
                    ltp_now = ts_get("ltp") or pos["entry_price"]
                    pnl = round((ltp_now - pos["entry_price"])
                                * (1 if pos["type"] == "buy" else -1) * qty, 2)
                    ts_append("completed_trades", {
                        "entry_datetime": pos["entry_time"],
                        "exit_datetime":  datetime.now(IST),
                        "type":           pos["type"],
                        "entry_price":    pos["entry_price"],
                        "exit_price":     ltp_now,
                        "sl":             pos["initial_sl"],
                        "target":         pos["target"],
                        "entry_reason":   pos["entry_reason"],
                        "exit_reason":    "Manual Square Off",
                        "pnl":            pnl,
                        "quantity":       qty,
                    })
                    ts_set("position", None)
                    st.success(f"Squared off! P&L: ₹{pnl:.2f}")
                    st.rerun()
                else:
                    st.info("No open position")

        # Status
        dot  = '<span class="status-live"></span>' if is_running else '<span class="status-off"></span>'
        stat = "LIVE" if is_running else "Stopped"
        clr  = "#00E5CC" if is_running else "#3a5a78"
        st.markdown(f'{dot}<span style="color:{clr};font-weight:700;font-size:0.9rem">{stat}</span>',
                     unsafe_allow_html=True)

        # ── Active Config ──
        cfg_live = ts_get("config")
        if cfg_live:
            with st.expander("🔧 Active Configuration", expanded=False):
                items = [
                    ("Ticker",      cfg_live.get("ticker",     "-")),
                    ("Timeframe",   cfg_live.get("timeframe",  "-")),
                    ("Strategy",    cfg_live.get("strategy",   "-")),
                    ("Direction",   cfg_live.get("trade_filter","-")),
                    ("Fast EMA",    str(cfg_live.get("fast_ema", "-"))),
                    ("Slow EMA",    str(cfg_live.get("slow_ema", "-"))),
                    ("SL Type",     cfg_live.get("sl_type",    "-")),
                    ("Target Type", cfg_live.get("target_type","-")),
                    ("Custom SL",   str(cfg_live.get("custom_sl",   "-"))),
                    ("Custom Tgt",  str(cfg_live.get("custom_target","-"))),
                    ("Quantity",    str(cfg_live.get("quantity","-"))),
                    ("Dhan",        "✅" if cfg_live.get("dhan_enabled") else "❌"),
                ]
                cols = st.columns(4)
                for idx_i, (lbl, val) in enumerate(items):
                    cols[idx_i % 4].markdown(mcard(lbl, val), unsafe_allow_html=True)
                    if idx_i % 4 == 3:
                        st.markdown("<br>", unsafe_allow_html=True)

        st.divider()

        # ── Main live layout ──
        lc, rc = st.columns([3, 1])

        with lc:
            df_live = ts_get("df")
            pos     = ts_get("position")

            if df_live is not None:
                fig_live = build_live_chart(df_live, pos, fast_ema, slow_ema)
                st.plotly_chart(fig_live, use_container_width=True)

                # Last candle row
                st.markdown("#### 📊 Last Fetched Candle")
                lc_row = ts_get("last_candle")
                if lc_row is not None:
                    if hasattr(lc_row, "to_dict"):
                        lc_dict = lc_row.to_dict()
                    else:
                        lc_dict = dict(lc_row)
                    # Clean display
                    disp = {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in lc_dict.items() if not k.startswith("_")}
                    st.dataframe(pd.DataFrame([disp]), use_container_width=True)
            else:
                st.markdown("""
                <div style="height:300px; display:flex; align-items:center; justify-content:center;
                             background:#080F20; border-radius:14px; border:1px solid #0f2040;
                             color:#3a5a78; font-size:1rem;">
                    Start live trading to see the chart
                </div>""", unsafe_allow_html=True)

        with rc:
            # Indicator Values
            ef_val  = ts_get("ema_fast_val")
            es_val  = ts_get("ema_slow_val")
            atr_v   = ts_get("atr_val")
            if ef_val:
                st.markdown("#### 📈 Indicators")
                st.markdown(mcard(f"EMA {fast_ema}", f"{ef_val:.2f}", "#FF9F43"),
                             unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(mcard(f"EMA {slow_ema}", f"{es_val:.2f}", "#4ECDC4"),
                             unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(mcard("ATR", f"{atr_v:.2f}", "#FFD93D"),
                             unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Active Position ──
            pos = ts_get("position")
            if pos:
                ltp_now  = ts_get("ltp") or pos["entry_price"]
                upnl     = (ltp_now - pos["entry_price"]) \
                            * (1 if pos["type"] == "buy" else -1) * qty
                upnl_clr = "#00c896" if upnl >= 0 else "#ff4d6d"
                card_cls = "buy-card" if pos["type"] == "buy" else "sell-card"
                badge    = "🟢 BUY" if pos["type"] == "buy" else "🔴 SELL"
                ttype_clr= "#00c896" if pos["type"]=="buy" else "#ff4d6d"

                st.markdown("#### 🎯 Open Position")
                st.markdown(f"""
                <div class="pos-card {card_cls}">
                    <div style="color:{ttype_clr};font-weight:800;font-size:1rem;margin-bottom:10px">
                        {badge}
                    </div>
                    <div class="wave-row">
                        <span class="pos-label">Entry</span>
                        <span class="pos-val">₹{pos['entry_price']:.2f}</span>
                    </div>
                    <div class="wave-row">
                        <span class="pos-label">LTP</span>
                        <span class="pos-val">₹{ltp_now:.2f}</span>
                    </div>
                    <div class="wave-row">
                        <span class="pos-label">Stop Loss</span>
                        <span class="pos-val" style="color:#ff4d6d">₹{pos['sl']:.2f}</span>
                    </div>
                    <div class="wave-row">
                        <span class="pos-label">Target</span>
                        <span class="pos-val" style="color:#00c896">₹{pos['target']:.2f}</span>
                    </div>
                    <div class="wave-row">
                        <span class="pos-label">Quantity</span>
                        <span class="pos-val">{pos.get('quantity', qty)}</span>
                    </div>
                    <hr style="border-color:#1a3060;margin:8px 0">
                    <div class="wave-row">
                        <span class="pos-label">Unrealised P&L</span>
                        <span class="pos-val" style="color:{upnl_clr}">₹{upnl:.2f}</span>
                    </div>
                    <div style="color:#3a5a78;font-size:0.7rem;margin-top:6px">
                        {pos['entry_reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#080F20;border:1px dashed #0f2040;border-radius:12px;
                             padding:20px;text-align:center;color:#3a5a78;font-size:0.85rem;">
                    No open position
                </div>""", unsafe_allow_html=True)

            # ── Elliott Wave Status ──
            if strategy == "Elliott Wave (Auto)":
                ew = ts_get("elliott_status") or {}
                st.markdown("#### 🌊 Elliott Wave")
                completed_str = ", ".join(ew.get("completed_waves", [])) or "—"
                proj          = ew.get("wave_projections", {})
                next_tgt      = ew.get("next_target")
                next_tgt_str  = f"₹{next_tgt:.2f}" if next_tgt else "—"
                signal_str    = (ew.get("signal") or "—").upper()
                sig_clr       = "#00c896" if ew.get("signal")=="buy" else \
                                "#ff4d6d" if ew.get("signal")=="sell" else "#7a9fc0"

                st.markdown(f"""
                <div class="wave-card">
                    <div class="wave-status">{ew.get("wave_status","Analyzing…")}</div>
                    <div class="wave-row">
                        <span class="wave-key">Current Wave</span>
                        <span class="wave-val">{ew.get("current_wave","—")}</span>
                    </div>
                    <div class="wave-row">
                        <span class="wave-key">Completed</span>
                        <span class="wave-val">{completed_str}</span>
                    </div>
                    <div class="wave-row">
                        <span class="wave-key">Signal</span>
                        <span class="wave-val" style="color:{sig_clr}">{signal_str}</span>
                    </div>
                    <div class="wave-row">
                        <span class="wave-key">Next Target</span>
                        <span class="wave-val" style="color:#FFD93D">{next_tgt_str}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                if proj:
                    st.markdown("**Projections:**")
                    for wname, wlvl in proj.items():
                        st.markdown(f"""
                        <div class="wave-row" style="padding:2px 0">
                            <span class="wave-key" style="font-size:0.78rem">{wname}</span>
                            <span class="wave-val" style="font-size:0.82rem">₹{wlvl:.2f}</span>
                        </div>""", unsafe_allow_html=True)

        # ── Trading Log ──
        st.markdown("#### 📋 Live Log")
        logs = ts_get("log") or []
        if logs:
            log_lines = ""
            for entry in reversed(logs[-60:]):
                if "ENTRY" in entry or "✅" in entry:
                    clr = "#00E5CC"
                elif "❌" in entry or "🔴" in entry:
                    clr = "#ff4d6d"
                elif "⚠️" in entry:
                    clr = "#FFD93D"
                elif "STOP" in entry:
                    clr = "#FF9F43"
                else:
                    clr = "#5a7fa8"
                log_lines += f'<div class="log-entry" style="color:{clr};margin:1px 0">{entry}</div>'
            st.markdown(f'<div class="log-box">{log_lines}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="log-box"><span style="color:#1a3060">No log entries yet…</span></div>',
                         unsafe_allow_html=True)

        # Auto-refresh
        if is_running:
            time.sleep(0.15)
            st.rerun()

    # ─────────────────────────────────────────────────────
    # TAB 3  –  TRADE HISTORY
    # ─────────────────────────────────────────────────────
    with tab_hist:
        st.markdown("### 📁 Trade History")
        st.caption("Updates in real-time while live trading is running.")

        completed = ts_get("completed_trades") or []

        if not completed:
            st.markdown("""
            <div style="background:#080F20;border:1px dashed #0f2040;border-radius:14px;
                         padding:40px;text-align:center;color:#3a5a78">
                <div style="font-size:2rem;margin-bottom:8px">📭</div>
                No completed trades yet.<br>They will appear here automatically as trades close.
            </div>""", unsafe_allow_html=True)
        else:
            total_pnl = sum(t["pnl"] for t in completed)
            winners   = sum(1 for t in completed if t["pnl"] > 0)
            n         = len(completed)
            acc       = winners / n * 100 if n else 0

            hc1, hc2, hc3, hc4, hc5 = st.columns(5)
            hc1.markdown(mcard("Trades",    str(n),                          "#e8f4ff"), unsafe_allow_html=True)
            hc2.markdown(mcard("Total P&L", f"₹{total_pnl:,.2f}",
                                "#00c896" if total_pnl>=0 else "#ff4d6d"),   unsafe_allow_html=True)
            hc3.markdown(mcard("Accuracy",  f"{acc:.1f}%",                   "#FFD93D"), unsafe_allow_html=True)
            hc4.markdown(mcard("Winners",   str(winners),                    "#00c896"), unsafe_allow_html=True)
            hc5.markdown(mcard("Losers",    str(n - winners),                "#ff4d6d"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if n > 1:
                st.plotly_chart(build_pnl_chart(completed), use_container_width=True)

            hist_df = pd.DataFrame(completed)
            for col in ("entry_datetime", "exit_datetime"):
                if col in hist_df.columns:
                    hist_df[col] = pd.to_datetime(hist_df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")

            def _style_hist(row):
                styles = []
                for col in row.index:
                    if col == "pnl":
                        styles.append("color:#00c896;font-weight:600" if row[col] > 0
                                       else "color:#ff4d6d;font-weight:600")
                    elif col == "type":
                        styles.append("color:#00c896" if row[col] == "buy" else "color:#ff4d6d")
                    else:
                        styles.append("")
                return styles

            st.dataframe(hist_df.style.apply(_style_hist, axis=1),
                          use_container_width=True, height=480)

            csv_data = hist_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # ─────────────────────────────────────────────────────
    # TAB 4  –  OPTIMIZATION
    # ─────────────────────────────────────────────────────
    with tab_opt:
        st.markdown("### ⚙️ Strategy Optimization")
        st.caption("Grid-search over EMA / SL / Target combinations. "
                    "Shows all results even if target accuracy is not reached.")

        oc1, oc2 = st.columns([1, 2])

        with oc1:
            tgt_acc     = st.number_input("Target Accuracy (%)", 0.0, 100.0, 60.0, 5.0, key="opt_acc")
            fast_r      = st.slider("Fast EMA Range",    3, 50,  (5, 20),  key="opt_fr")
            slow_r      = st.slider("Slow EMA Range",    10,100, (15, 50), key="opt_sr")
            sl_r        = st.slider("SL Points Range",   3, 100, (5, 30),  key="opt_slr")
            tgt_r       = st.slider("Target Points Range",5,200, (10, 60), key="opt_tgtr")
            max_combos  = st.number_input("Max Combinations", 10, 1000, 100, key="opt_mc")
            run_opt_btn = st.button("🔍 Run Optimization", type="primary",
                                     use_container_width=True, key="run_opt")

        with oc2:
            if run_opt_btn:
                df_opt = fetch_data(selected_ticker, timeframe, period, warmup=True)

                if df_opt is None or df_opt.empty:
                    st.error("Failed to fetch data for optimization.")
                else:
                    fast_vals = list(range(fast_r[0], fast_r[1]+1, max(1,(fast_r[1]-fast_r[0])//8)))
                    slow_vals = list(range(slow_r[0], slow_r[1]+1, max(1,(slow_r[1]-slow_r[0])//8)))
                    sl_vals   = list(range(sl_r[0],  sl_r[1]+1,  max(1,(sl_r[1]-sl_r[0])//5)))
                    tgt_vals  = list(range(tgt_r[0], tgt_r[1]+1, max(1,(tgt_r[1]-tgt_r[0])//5)))

                    combos = [(f, s, sl, tg)
                              for f, s, sl, tg in itertools.product(fast_vals, slow_vals, sl_vals, tgt_vals)
                              if f < s]
                    random.shuffle(combos)
                    combos = combos[:int(max_combos)]

                    prog = st.progress(0.0, text="Optimizing…")
                    results = []

                    for idx_o, (f, s, sl, tg) in enumerate(combos):
                        try:
                            o_cfg = dict(
                                strategy="EMA Crossover",
                                trade_filter="Both",
                                sl_type="Custom Points", target_type="Custom Points",
                                quantity=1, fast_ema=f, slow_ema=s,
                                custom_sl=float(sl), custom_target=float(tg),
                                atr_mult_sl=2.0, atr_mult_target=3.0, rr_ratio=2.0,
                                xover_type="Simple Crossover", candle_size=10.0,
                                min_angle=0.0, check_angle=False,
                            )
                            ts_o, _, acc_o, _ = run_backtest(df_opt.copy(), o_cfg)
                            if ts_o:
                                pnl_o = sum(t["pnl"] for t in ts_o)
                                results.append({
                                    "Fast EMA": f, "Slow EMA": s,
                                    "SL (pts)": sl, "Target (pts)": tg,
                                    "Trades": len(ts_o),
                                    "Accuracy (%)": round(acc_o, 1),
                                    "Total P&L": round(pnl_o, 2),
                                    "✓ Target Met": acc_o >= tgt_acc,
                                })
                        except Exception:
                            pass
                        prog.progress((idx_o + 1) / len(combos),
                                       text=f"Testing {idx_o+1}/{len(combos)}…")

                    prog.empty()

                    if not results:
                        st.warning("No valid combinations found.")
                    else:
                        res_df = pd.DataFrame(results).sort_values("Total P&L", ascending=False)
                        st.session_state["opt_results"] = res_df

            if st.session_state.get("opt_results") is not None:
                res_df  = st.session_state["opt_results"]
                meets   = res_df[res_df["✓ Target Met"] == True]
                n_meets = len(meets)
                n_all   = len(res_df)

                if n_meets:
                    st.success(f"✅ {n_meets}/{n_all} combinations meet {tgt_acc:.0f}% accuracy")
                    st.dataframe(meets, use_container_width=True)
                else:
                    st.warning(f"⚠️ 0/{n_all} combinations reached {tgt_acc:.0f}% – "
                                f"showing best available results:")

                st.markdown(f"**All {n_all} results (sorted by P&L):**")

                def _style_opt(row):
                    styles = []
                    for col in row.index:
                        if col == "Total P&L":
                            styles.append("color:#00c896" if row[col] > 0 else "color:#ff4d6d")
                        elif col == "✓ Target Met":
                            styles.append("color:#00c896;font-weight:700" if row[col] else "color:#3a5a78")
                        elif col == "Accuracy (%)":
                            styles.append("color:#FFD93D")
                        else:
                            styles.append("")
                    return styles

                st.dataframe(res_df.style.apply(_style_opt, axis=1),
                              use_container_width=True, height=500)

                best = res_df.iloc[0]
                st.markdown(f"""
                <div class="best-result">
                    <div style="color:#00E5CC;font-weight:800;margin-bottom:10px">🏆 Best Configuration</div>
                    <div class="wave-row">
                        <span class="wave-key">Fast EMA × Slow EMA</span>
                        <span class="wave-val">{int(best['Fast EMA'])} × {int(best['Slow EMA'])}</span>
                    </div>
                    <div class="wave-row">
                        <span class="wave-key">SL / Target (pts)</span>
                        <span class="wave-val">{int(best['SL (pts)'])} / {int(best['Target (pts)'])}</span>
                    </div>
                    <div class="wave-row">
                        <span class="wave-key">Accuracy</span>
                        <span class="wave-val" style="color:#FFD93D">{best['Accuracy (%)']:.1f}%</span>
                    </div>
                    <div class="wave-row">
                        <span class="wave-key">Total P&L</span>
                        <span class="wave-val" style="color:{'#00c896' if best['Total P&L']>=0 else '#ff4d6d'}">
                            ₹{best['Total P&L']:,.2f}
                        </span>
                    </div>
                    <div class="wave-row">
                        <span class="wave-key">Trades</span>
                        <span class="wave-val">{int(best['Trades'])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                csv_opt = res_df.to_csv(index=False)
                st.download_button("📥 Download Optimization Results",
                                    data=csv_opt,
                                    file_name="optimization_results.csv",
                                    mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
