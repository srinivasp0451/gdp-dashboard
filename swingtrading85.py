#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         SMART INVESTING — Algorithmic Trading Platform           ║
║  NSE | BSE | Crypto | Backtesting | Live | Elliott Wave | Dhan  ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ================================================================
# 1. IMPORTS
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading, time, math, warnings, json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import pytz

warnings.filterwarnings("ignore")

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Smart Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# 2. CSS STYLING
# ================================================================
st.markdown("""
<style>
/* ── Base ── */
.main .block-container { padding-top: 0.5rem; padding-bottom: 1rem; max-width: 100%; }
h1,h2,h3,h4 { color: #e6edf3; }

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg,#161b22,#1c2128);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 14px 18px; text-align: center; margin: 4px 0;
}
.metric-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #58a6ff; font-size: 22px; font-weight: 700; margin-top: 2px; }
.delta-pos { color: #3fb950; font-size: 12px; }
.delta-neg { color: #f85149; font-size: 12px; }

/* ── Badges ── */
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.badge-buy  { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.badge-sell { background:#2d1a1a; color:#f85149; border:1px solid #da3633; }
.badge-hold { background:#1c2128; color:#8b949e; border:1px solid #30363d; }
.badge-run  { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.badge-stop { background:#2d1a1a; color:#f85149; border:1px solid #da3633; }
.badge-warn { background:#2d1f0a; color:#e3b341; border:1px solid #9e6a03; }

/* ── LTP Bar ── */
.ltp-bar {
    background:linear-gradient(90deg,#161b22,#1c2128);
    border:1px solid #30363d; border-radius:10px;
    padding:10px 20px; display:flex; align-items:center;
    gap:24px; margin-bottom:12px; flex-wrap:wrap;
}
.ltp-ticker { color:#8b949e; font-size:12px; }
.ltp-price  { color:#58a6ff; font-size:22px; font-weight:700; }
.ltp-change-pos { color:#3fb950; font-size:14px; }
.ltp-change-neg { color:#f85149; font-size:14px; }

/* ── Config Box ── */
.config-box {
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:14px; margin:6px 0;
}
.config-row { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid #21262d; }
.config-row:last-child { border-bottom: none; }
.config-key { color:#8b949e; font-size:12px; }
.config-val { color:#e6edf3; font-size:12px; font-weight:600; }

/* ── Trade Table ── */
.trade-wrap { overflow-x:auto; }
table.trade-table { width:100%; border-collapse:collapse; font-size:12px; }
table.trade-table th {
    background:#161b22; color:#8b949e;
    padding:8px 10px; text-align:left;
    border-bottom:2px solid #30363d; white-space:nowrap;
}
table.trade-table td { padding:6px 10px; border-bottom:1px solid #21262d; white-space:nowrap; }
table.trade-table tr:hover td { background:#1c2128; }
.pnl-pos { color:#3fb950; font-weight:700; }
.pnl-neg { color:#f85149; font-weight:700; }
.violation-row td { background:#2d1a1a !important; }

/* ── Wave Cards ── */
.wave-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin:8px 0; }
.wave-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:10px; text-align:center;
}
.wave-complete  { border-color:#238636!important; }
.wave-active    { border-color:#e3b341!important; }
.wave-pending   { border-color:#30363d!important; }
.wave-name      { font-size:20px; font-weight:700; }
.wave-info      { color:#8b949e; font-size:11px; margin-top:4px; }

/* ── Alert Boxes ── */
.alert { padding:10px 14px; border-radius:8px; margin:6px 0; font-size:13px; }
.alert-info    { background:#1c2d3e; border-left:3px solid #58a6ff; color:#a0cfff; }
.alert-warn    { background:#2d2318; border-left:3px solid #e3b341; color:#f0c672; }
.alert-error   { background:#2d1a1a; border-left:3px solid #f85149; color:#ff8080; }
.alert-success { background:#1a2d1a; border-left:3px solid #3fb950; color:#70d080; }

/* ── Log ── */
.log-box {
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    padding:10px; height:180px; overflow-y:auto; font-family:monospace; font-size:11px;
}
.log-info  { color:#58a6ff; }
.log-buy   { color:#3fb950; }
.log-sell  { color:#f85149; }
.log-warn  { color:#e3b341; }
.log-exit  { color:#a371f7; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#0d1117!important; }
[data-testid="stSidebar"] label { color:#e6edf3!important; }
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stNumberInput > label,
[data-testid="stSidebar"] .stCheckbox > label { color:#8b949e!important; font-size:12px!important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#161b22; border-radius:10px; padding:4px; gap:2px;
}
.stTabs [data-baseweb="tab"] {
    background:transparent; color:#8b949e; border-radius:8px; font-size:13px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background:#21262d; color:#e6edf3;
}

/* ── Position Card ── */
.pos-card {
    background:linear-gradient(135deg,#1a2d1a,#161b22);
    border:1px solid #238636; border-radius:12px; padding:16px; margin:8px 0;
}
.pos-card.sell-pos {
    background:linear-gradient(135deg,#2d1a1a,#161b22);
    border-color:#da3633;
}
.pos-title { font-size:13px; font-weight:700; color:#e6edf3; margin-bottom:8px; }
.pos-grid  { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.pos-field { }
.pos-flabel { color:#8b949e; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
.pos-fval   { color:#e6edf3; font-size:14px; font-weight:600; margin-top:2px; }
.pos-fval.green { color:#3fb950; }
.pos-fval.red   { color:#f85149; }

/* ── Summary Cards ── */
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin:10px 0; }
.summary-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:12px; text-align:center;
}
.summary-label { color:#8b949e; font-size:11px; text-transform:uppercase; }
.summary-value { color:#e6edf3; font-size:20px; font-weight:700; margin-top:4px; }

/* ── Divider ── */
hr.section-divider { border:none; border-top:1px solid #21262d; margin:14px 0; }

/* ── Candle Row ── */
.candle-row {
    display:flex; gap:20px; align-items:center;
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:10px 16px; margin:6px 0;
    flex-wrap:wrap;
}
.candle-field { }
.candle-flabel { color:#8b949e; font-size:10px; text-transform:uppercase; }
.candle-fval   { color:#e6edf3; font-size:14px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# 3. CONSTANTS & MAPPINGS
# ================================================================
IST = pytz.timezone("Asia/Kolkata")

TICKERS: Dict[str, str] = {
    "Nifty 50":   "^NSEI",
    "BankNifty":  "^NSEBANK",
    "Sensex":     "^BSESN",
    "BTC-USD":    "BTC-USD",
    "ETH-USD":    "ETH-USD",
    "Gold":       "GC=F",
    "Silver":     "SI=F",
    "Custom":     "CUSTOM",
}

TIMEFRAME_PERIODS: Dict[str, List[str]] = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

# Extra period for warmup candles (to ensure TradingView-accurate EMA)
WARMUP_PERIOD: Dict[str, str] = {
    "1m":  "5d",
    "5m":  "1mo",
    "15m": "1mo",
    "1h":  "3mo",
    "1d":  "2y",
    "1wk": "5y",
}

STRATEGIES = [
    "EMA Crossover",
    "Simple Buy",
    "Simple Sell",
    "Anticipatory EMA Crossover",
    "Elliott Wave Auto",
]

SL_TYPES = [
    "Custom Points",
    "Trailing SL",
    "ATR Based",
    "Risk-Reward Based",
    "Auto SL",
    "EMA Reverse Crossover",
    "Volatility Based",
    "Trail with Swing Low/High",
    "Trail with Candle Low/High",
    "Support/Resistance Based",
]

TARGET_TYPES = [
    "Custom Points",
    "Trailing Target (Display Only)",
    "ATR Based",
    "Risk-Reward Based",
    "Auto Target",
    "EMA Crossover Exit",
    "Volatility Based",
    "Trail with Swing High/Low",
    "Trail with Candle High/Low",
    "Support/Resistance Based",
    "Book Profit: T1 then T2",
]

CROSSOVER_TYPES = ["Simple Crossover", "Custom Candle Size", "ATR Based Candle Size"]


# ================================================================
# 4. THREAD-SAFE STATE
# ================================================================
_TS_LOCK = threading.RLock()
_TS: Dict[str, Any] = {
    "live_running":    False,
    "live_position":   None,
    "live_trades":     [],
    "live_log":        [],
    "last_candle":     None,
    "ltp":             None,
    "ema_fast_val":    None,
    "ema_slow_val":    None,
    "wave_data":       {},
    "config":          {},
    "stop_event":      threading.Event(),
    "live_thread":     None,
    "cooldown_until":  None,
    "prev_candle_time": None,
    "live_error":      None,
    "tick_count":      0,
}

def ts_get(key: str, default=None):
    with _TS_LOCK:
        return _TS.get(key, default)

def ts_set(key: str, value):
    with _TS_LOCK:
        _TS[key] = value

def ts_append(key: str, value):
    with _TS_LOCK:
        if key not in _TS:
            _TS[key] = []
        _TS[key].append(value)

def ts_update_dict(key: str, updates: dict):
    with _TS_LOCK:
        if _TS.get(key) is None:
            _TS[key] = {}
        _TS[key].update(updates)

def ts_log(msg: str, level: str = "info"):
    timestamp = datetime.now(IST).strftime("%H:%M:%S")
    entry = {"time": timestamp, "msg": msg, "level": level}
    with _TS_LOCK:
        _TS["live_log"].append(entry)
        if len(_TS["live_log"]) > 200:
            _TS["live_log"] = _TS["live_log"][-200:]


# ================================================================
# 5. DATA FETCHING
# ================================================================
def get_yf_symbol(ticker_name: str, custom: str = "") -> str:
    if ticker_name == "Custom":
        return custom.strip().upper()
    return TICKERS.get(ticker_name, ticker_name)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    return df

def fetch_ohlcv(symbol: str, interval: str, period: str,
                warmup: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV with optional warmup candles.
    Returns full DataFrame (including warmup) so EMA calcs are accurate.
    The caller slices to display period after computing indicators.
    """
    try:
        if warmup:
            wp = WARMUP_PERIOD.get(interval, period)
            df = yf.download(symbol, interval=interval, period=wp,
                             auto_adjust=True, progress=False, threads=False)
        else:
            df = yf.download(symbol, interval=interval, period=period,
                             auto_adjust=True, progress=False, threads=False)

        if df is None or len(df) == 0:
            return None

        df = _flatten_columns(df)

        # Keep only OHLCV
        cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[cols].copy()
        df.dropna(subset=["Close"], inplace=True)

        if len(df) == 0:
            return None

        return df

    except Exception as e:
        return None

def fetch_ohlcv_with_slice(symbol: str, interval: str, period: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Returns (full_df_with_warmup, display_df_for_period).
    full_df is used for indicator calculations; display_df for charts/backtest.
    """
    full_df = fetch_ohlcv(symbol, interval, period, warmup=True)
    if full_df is None:
        return None, None

    display_df = fetch_ohlcv(symbol, interval, period, warmup=False)
    if display_df is None:
        display_df = full_df.copy()

    return full_df, display_df

def get_ltp(symbol: str) -> Optional[float]:
    """Get latest price quickly."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.fast_info
        price = getattr(data, 'last_price', None)
        if price is not None and not math.isnan(float(price)):
            return float(price)
        # Fallback
        df = yf.download(symbol, interval="1m", period="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df is not None and len(df) > 0:
            df = _flatten_columns(df)
            return float(df['Close'].iloc[-1])
    except:
        pass
    return None


# ================================================================
# 6. INDICATOR CALCULATIONS (TradingView-accurate)
# ================================================================
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    TradingView-accurate EMA:  EMA = ewm(span=n, adjust=False, min_periods=1)
    This ensures first EMA = first price (min_periods=1) and uses
    multiplicative smoothing factor k = 2/(n+1) with no bias correction.
    """
    return series.ewm(span=period, adjust=False, min_periods=1).mean()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=1).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(span=period, adjust=False, min_periods=1).mean()
    avg_l = loss.ewm(span=period, adjust=False, min_periods=1).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

def compute_std(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period, min_periods=1).std().fillna(0)

def compute_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Rolling minimum low over lookback period (recent swing low)."""
    return df['Low'].rolling(window=lookback, min_periods=1).min()

def compute_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Rolling maximum high over lookback period (recent swing high)."""
    return df['High'].rolling(window=lookback, min_periods=1).max()

def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
    """Simple support/resistance using rolling min/max."""
    n = min(window, len(df))
    recent = df.tail(n)
    support    = float(recent['Low'].min())
    resistance = float(recent['High'].max())
    return support, resistance

def compute_crossover_angle(ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """Approximate crossover angle in degrees."""
    diff = ema_fast - ema_slow
    angle = diff.diff().apply(lambda x: math.degrees(math.atan(x)) if not math.isnan(x) else 0)
    return angle


# ================================================================
# 7. ELLIOTT WAVE ANALYSIS
# ================================================================
def find_zigzag_pivots(df: pd.DataFrame, deviation: float = 0.03) -> pd.DataFrame:
    """
    ZigZag pivot detection for Elliott Wave.
    Returns DataFrame with columns: bar_idx, price, ptype (H/L), datetime.
    """
    if len(df) < 5:
        return pd.DataFrame(columns=["bar_idx", "price", "ptype", "datetime"])

    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    n     = len(close)

    pivots: List[Tuple] = []

    # Determine initial direction
    direction = 1 if close[min(5, n-1)] > close[0] else -1
    last_idx   = 0
    last_price = high[0] if direction == 1 else low[0]

    for i in range(1, n):
        if direction == 1:           # looking for high
            if high[i] > last_price:
                last_price = high[i]
                last_idx   = i
            elif close[i] < last_price * (1 - deviation):
                pivots.append((last_idx, last_price, "H"))
                direction  = -1
                last_idx   = i
                last_price = low[i]
        else:                        # looking for low
            if low[i] < last_price:
                last_price = low[i]
                last_idx   = i
            elif close[i] > last_price * (1 + deviation):
                pivots.append((last_idx, last_price, "L"))
                direction  = 1
                last_idx   = i
                last_price = high[i]

    # Add last pivot
    ptype = "H" if direction == 1 else "L"
    if not pivots or pivots[-1][0] != last_idx:
        pivots.append((last_idx, last_price, ptype))

    if not pivots:
        return pd.DataFrame(columns=["bar_idx", "price", "ptype", "datetime"])

    pdf = pd.DataFrame(pivots, columns=["bar_idx", "price", "ptype"])
    pdf["datetime"] = pdf["bar_idx"].apply(lambda i: df.index[i])
    return pdf

def validate_impulse_rules(prices: List[float]) -> Tuple[bool, str]:
    """
    Validate 5-wave impulse: [w0, w1, w2, w3, w4, w5].
    Returns (is_valid, reason).
    """
    if len(prices) < 6:
        return False, "Need 6 pivot prices"

    p = prices  # p[0]=w0, p[1]=w1 ... p[5]=w5
    bull = p[1] > p[0]

    if bull:
        if p[2] <= p[0]:
            return False, "Rule 1: Wave2 retraced past Wave1 start"
        len1 = abs(p[1] - p[0])
        len3 = abs(p[3] - p[2])
        len5 = abs(p[5] - p[4])
        if len3 < len1 and len3 < len5:
            return False, "Rule 2: Wave3 is shortest"
        if p[4] <= p[1]:
            return False, "Rule 3: Wave4 overlaps Wave1"
        if not (p[1]>p[0] and p[2]<p[1] and p[3]>p[1] and p[4]<p[3] and p[5]>p[3]):
            return False, "Direction check failed (bull)"
    else:
        if p[2] >= p[0]:
            return False, "Rule 1 (bear): Wave2 retraced past Wave1 start"
        len1 = abs(p[1] - p[0])
        len3 = abs(p[3] - p[2])
        len5 = abs(p[5] - p[4])
        if len3 < len1 and len3 < len5:
            return False, "Rule 2 (bear): Wave3 is shortest"
        if p[4] >= p[1]:
            return False, "Rule 3 (bear): Wave4 overlaps Wave1"
        if not (p[1]<p[0] and p[2]>p[1] and p[3]<p[1] and p[4]>p[3] and p[5]<p[3]):
            return False, "Direction check failed (bear)"

    return True, "Valid impulse"

def fib_levels_from_wave(start: float, end: float) -> Dict[str, float]:
    diff = end - start
    return {
        "0%":    end,
        "23.6%": end - 0.236 * diff,
        "38.2%": end - 0.382 * diff,
        "50.0%": end - 0.500 * diff,
        "61.8%": end - 0.618 * diff,
        "78.6%": end - 0.786 * diff,
        "100%":  start,
        "161.8%": start - 0.618 * diff,
        "261.8%": start - 1.618 * diff,
    }

def analyze_elliott_waves(df: pd.DataFrame) -> Dict:
    """
    Complete Elliott Wave analysis.
    Works identically for backtesting and live trading —
    just call with the appropriate DataFrame slice.
    """
    result = {
        "status":          "Analyzing...",
        "waves":           [],
        "current_wave":    None,
        "wave_status":     "Unknown",
        "direction":       None,
        "signal":          None,
        "sl":              None,
        "tp1":             None,
        "tp2":             None,
        "tp3":             None,
        "next_target":     None,
        "next_wave_range": None,
        "fib_levels":      {},
        "pivots":          pd.DataFrame(),
        "impulse_complete": False,
        "abc_waves":       [],
    }

    if len(df) < 20:
        result["status"] = "Insufficient data"
        return result

    # Adaptive deviation
    atr   = compute_atr(df, 14).iloc[-1]
    price = df["Close"].iloc[-1]
    dev   = max(0.015, min(0.07, (atr / price) * 2.5))

    pivots = find_zigzag_pivots(df, deviation=dev)
    result["pivots"] = pivots

    if len(pivots) < 4:
        result["status"] = "Not enough pivots detected"
        return result

    prices = pivots["price"].values
    ptypes = pivots["ptype"].values
    times  = pivots["datetime"].values
    n      = len(prices)

    # ── Search for best valid 5-wave impulse in recent pivots ──
    best_start = None
    for start in range(max(0, n - 16), max(0, n - 5)):
        cand_idx  = [start]
        exp_type  = "H" if ptypes[start] == "L" else "L"
        for j in range(start + 1, min(start + 12, n)):
            if ptypes[j] == exp_type:
                cand_idx.append(j)
                exp_type = "H" if exp_type == "L" else "L"
            if len(cand_idx) == 6:
                break

        if len(cand_idx) < 6:
            continue

        cp = [prices[i] for i in cand_idx]
        valid, reason = validate_impulse_rules(cp)
        if valid:
            best_start = cand_idx
            break      # take most recent valid pattern

    if best_start is not None and len(best_start) >= 5:
        idxs  = best_start
        wp    = [prices[i] for i in idxs]
        wt    = [times[i]  for i in idxs]
        bull  = wp[1] > wp[0]
        direc = 1 if bull else -1
        wave_labels = ["1", "2", "3", "4", "5"]

        waves_info = []
        for k in range(min(5, len(wp) - 1)):
            waves_info.append({
                "wave":        wave_labels[k],
                "start_price": wp[k],
                "end_price":   wp[k + 1],
                "start_time":  wt[k],
                "end_time":    wt[k + 1],
                "status":      "Complete",
                "direction":   "Up" if wp[k+1] > wp[k] else "Down",
                "pct_move":    round(abs(wp[k+1] - wp[k]) / wp[k] * 100, 2),
            })

        result["waves"]    = waves_info
        result["direction"] = "Up" if bull else "Down"

        if len(idxs) >= 6:
            # Full 5-wave complete → ABC correction
            result["impulse_complete"] = True
            result["current_wave"]     = "A"
            result["wave_status"]      = "Impulse complete — ABC correction forming"

            w5 = wp[5]
            w0 = wp[0]
            span = abs(w5 - w0)

            a_tgt = w5 - direc * span * 0.382
            b_tgt = w5 - direc * span * 0.618
            result["next_target"]     = a_tgt
            result["next_wave_range"] = (min(a_tgt, b_tgt), max(a_tgt, b_tgt))
            result["fib_levels"]      = fib_levels_from_wave(w0, w5)
            result["status"]          = "5-Wave Impulse Complete"

        elif len(idxs) == 5:
            # Wave 5 starting (after Wave 4 pivot)
            w0, w1, w2, w3, w4 = wp[0], wp[1], wp[2], wp[3], wp[4]
            len_w1 = abs(w1 - w0)
            len_w3 = abs(w3 - w2)

            w5_t1 = w4 + direc * len_w1 * 0.618
            w5_t2 = w4 + direc * len_w1 * 1.000
            w5_t3 = w4 + direc * len_w1 * 1.618

            result["current_wave"]     = "5"
            result["wave_status"]      = "Wave 5 forming — entry opportunity"
            result["next_target"]      = w5_t2
            result["next_wave_range"]  = (min(w5_t1, w5_t3), max(w5_t1, w5_t3))
            result["fib_levels"]       = fib_levels_from_wave(w0, w3)
            result["status"]           = "Wave 5 in progress"
            result["tp1"]              = w5_t1
            result["tp2"]              = w5_t2
            result["tp3"]              = w5_t3

            # Auto signal
            if bull:
                result["signal"] = "buy"
                result["sl"]     = w4 * 0.995
            else:
                result["signal"] = "sell"
                result["sl"]     = w4 * 1.005

        elif len(idxs) == 4:
            # Wave 4 forming
            result["current_wave"]  = "4"
            result["wave_status"]   = "Wave 4 correction in progress"
            result["status"]        = "Wave 4 forming — wait for completion"

    else:
        # Partial wave detection
        if n >= 3:
            last_dir = "Up" if prices[-1] > prices[-2] else "Down"
            wave_num = min(n - 1, 3)
            result["current_wave"] = str(wave_num)
            result["wave_status"]  = f"Partial wave {wave_num} — {last_dir} trend"
            result["direction"]    = last_dir
            result["status"]       = f"Partial wave detected ({wave_num} of 5)"
        else:
            result["status"] = "Insufficient pivot data"

    return result


# ================================================================
# 8. STRATEGY SIGNAL GENERATION
# ================================================================
def add_indicators(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Add all indicator columns needed by strategies."""
    fast = cfg.get("ema_fast", 9)
    slow = cfg.get("ema_slow", 15)

    df = df.copy()
    df["ema_fast"] = compute_ema(df["Close"], fast)
    df["ema_slow"] = compute_ema(df["Close"], slow)
    df["atr"]      = compute_atr(df, cfg.get("atr_period", 14))
    df["rsi"]      = compute_rsi(df["Close"], 14)
    df["std"]      = compute_std(df["Close"], 20)
    df["swing_lo"] = compute_swing_lows(df,  cfg.get("swing_lookback", 10))
    df["swing_hi"] = compute_swing_highs(df, cfg.get("swing_lookback", 10))
    return df

def signals_ema_crossover(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    """
    EMA Crossover signals.
    Signal on candle N → entry signal column set at N.
    Actual entry in backtesting/live is at candle N+1 open.
    Returns pd.Series of 1 (buy), -1 (sell), 0 (hold).
    """
    fast    = df["ema_fast"]
    slow    = df["ema_slow"]
    signals = pd.Series(0, index=df.index)

    cross_above = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_below = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    # Apply crossover filters
    cross_type  = cfg.get("crossover_type", "Simple Crossover")
    min_angle   = cfg.get("min_crossover_angle", 0.0)
    custom_size = cfg.get("custom_candle_size", 10.0)

    candle_body = (df["Close"] - df["Open"]).abs()

    if cross_type == "Custom Candle Size":
        cross_above = cross_above & (candle_body >= custom_size)
        cross_below = cross_below & (candle_body >= custom_size)
    elif cross_type == "ATR Based Candle Size":
        atr_filter  = df["atr"] * cfg.get("atr_candle_mult", 0.5)
        cross_above = cross_above & (candle_body >= atr_filter)
        cross_below = cross_below & (candle_body >= atr_filter)

    # Minimum angle filter
    if min_angle > 0:
        angle = compute_crossover_angle(fast, slow).abs()
        cross_above = cross_above & (angle >= min_angle)
        cross_below = cross_below & (angle >= min_angle)

    signals[cross_above] = 1
    signals[cross_below] = -1
    return signals

def signals_simple_buy(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    """Simple Buy: signal on every candle (immediate entry)."""
    sig = pd.Series(0, index=df.index)
    if len(df) > 0:
        sig.iloc[-1] = 1  # only latest candle for live; backtest engine iterates
    return sig

def signals_simple_sell(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    sig = pd.Series(0, index=df.index)
    if len(df) > 0:
        sig.iloc[-1] = -1
    return sig

def signals_anticipatory_ema(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    """
    Anticipatory EMA Crossover:
    Detect when crossover is LIKELY to happen before it actually does.
    Criteria:
    1. EMA gap narrowing rapidly (convergence rate)
    2. Price momentum in crossover direction
    3. Candle sustainability (wicks vs body ratio)
    4. RSI alignment
    """
    signals  = pd.Series(0, index=df.index)
    fast     = df["ema_fast"]
    slow     = df["ema_slow"]
    gap      = fast - slow
    gap_prev = gap.shift(1)
    gap_rate = gap - gap_prev      # rate of change of gap

    rsi = df["rsi"]

    for i in range(2, len(df)):
        g  = gap.iloc[i]
        gp = gap.iloc[i-1]
        gr = gap_rate.iloc[i]      # gap narrowing speed

        # Anticipate bullish crossover
        if (gp < 0 and g < 0 and abs(g) < abs(gp)):
            # Gap narrowing
            conv_rate = abs(gr) / (abs(gp) + 1e-8)

            # Price above recent SMA
            price_mom = df["Close"].iloc[i] > df["Close"].iloc[i-3:i].mean()

            # Candle body fraction (sustainability)
            body   = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
            candle = abs(df["High"].iloc[i]  - df["Low"].iloc[i])
            body_frac = body / (candle + 1e-8)

            # RSI alignment (above 45 for bullish anticipation)
            rsi_ok = rsi.iloc[i] > 45

            score = (conv_rate * 40) + (20 if price_mom else 0) + (body_frac * 20) + (20 if rsi_ok else 0)
            if score >= 60 and conv_rate > 0.15:
                signals.iloc[i] = 1

        # Anticipate bearish crossover
        elif (gp > 0 and g > 0 and abs(g) < abs(gp)):
            conv_rate = abs(gr) / (abs(gp) + 1e-8)
            price_mom = df["Close"].iloc[i] < df["Close"].iloc[i-3:i].mean()
            body      = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
            candle    = abs(df["High"].iloc[i]  - df["Low"].iloc[i])
            body_frac = body / (candle + 1e-8)
            rsi_ok    = rsi.iloc[i] < 55

            score = (conv_rate * 40) + (20 if price_mom else 0) + (body_frac * 20) + (20 if rsi_ok else 0)
            if score >= 60 and conv_rate > 0.15:
                signals.iloc[i] = -1

    return signals

def signals_elliott_wave(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    """
    Elliott Wave auto signals.
    Signal when Wave 5 entry is detected.
    """
    signals = pd.Series(0, index=df.index)

    # Need enough data for wave detection
    if len(df) < 30:
        return signals

    # Compute on rolling window for backtesting
    for i in range(29, len(df)):
        slice_df = df.iloc[max(0, i-100):i+1]
        wave_res  = analyze_elliott_waves(slice_df)
        sig = wave_res.get("signal")
        if sig == "buy":
            signals.iloc[i] = 1
        elif sig == "sell":
            signals.iloc[i] = -1

    return signals

def get_signals(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    """Route to correct strategy signal function."""
    strategy = cfg.get("strategy", "EMA Crossover")
    df = add_indicators(df, cfg)

    # For simple buy/sell in backtest, generate buy/sell on ALL candles
    if strategy == "Simple Buy":
        return pd.Series(1, index=df.index), df
    elif strategy == "Simple Sell":
        return pd.Series(-1, index=df.index), df
    elif strategy == "EMA Crossover":
        return signals_ema_crossover(df, cfg), df
    elif strategy == "Anticipatory EMA Crossover":
        return signals_anticipatory_ema(df, cfg), df
    elif strategy == "Elliott Wave Auto":
        return signals_elliott_wave(df, cfg), df
    else:
        return pd.Series(0, index=df.index), df


# ================================================================
# 9. SL / TARGET CALCULATION
# ================================================================
def calc_initial_sl(entry_price: float, trade_type: str,
                    df: pd.DataFrame, cfg: Dict,
                    entry_idx: int) -> float:
    sl_type   = cfg.get("sl_type", "Custom Points")
    sl_points = cfg.get("sl_points", 10.0)
    rr_ratio  = cfg.get("rr_ratio", 2.0)
    atr_mult  = cfg.get("atr_mult_sl", 1.5)
    vol_mult  = cfg.get("vol_mult_sl", 2.0)
    swing_lb  = cfg.get("swing_lookback", 10)
    sign      = -1 if trade_type == "buy" else 1

    try:
        row = df.iloc[entry_idx]
        atr = float(df["atr"].iloc[entry_idx])
    except:
        return entry_price + sign * sl_points

    if sl_type == "Custom Points":
        return entry_price + sign * sl_points

    elif sl_type == "ATR Based":
        return entry_price + sign * atr * atr_mult

    elif sl_type in ("Auto SL", "Volatility Based"):
        std = float(df["std"].iloc[entry_idx])
        return entry_price + sign * max(atr * 1.5, std * vol_mult)

    elif sl_type == "Risk-Reward Based":
        tgt_pts = cfg.get("target_points", 20.0)
        return entry_price + sign * (tgt_pts / rr_ratio)

    elif sl_type == "EMA Reverse Crossover":
        # SL at slow EMA level at entry
        return float(df["ema_slow"].iloc[entry_idx])

    elif sl_type == "Trail with Swing Low/High":
        lo = float(df["swing_lo"].iloc[entry_idx])
        hi = float(df["swing_hi"].iloc[entry_idx])
        if trade_type == "buy":
            return lo - atr * 0.2
        else:
            return hi + atr * 0.2

    elif sl_type == "Trail with Candle Low/High":
        if trade_type == "buy":
            return float(row["Low"]) - atr * 0.1
        else:
            return float(row["High"]) + atr * 0.1

    elif sl_type == "Support/Resistance Based":
        support, resistance = compute_support_resistance(df.iloc[:entry_idx+1])
        if trade_type == "buy":
            return support - atr * 0.1
        else:
            return resistance + atr * 0.1

    elif sl_type == "Trailing SL":
        return entry_price + sign * sl_points

    return entry_price + sign * sl_points

def calc_initial_target(entry_price: float, trade_type: str,
                        df: pd.DataFrame, cfg: Dict,
                        entry_idx: int, sl: float) -> Tuple[float, float, float]:
    """Returns (tp1, tp2, tp3)."""
    tgt_type   = cfg.get("target_type", "Custom Points")
    tgt_points = cfg.get("target_points", 20.0)
    rr_ratio   = cfg.get("rr_ratio", 2.0)
    atr_mult   = cfg.get("atr_mult_tgt", 2.5)
    vol_mult   = cfg.get("vol_mult_tgt", 3.0)
    sign       = 1 if trade_type == "buy" else -1

    try:
        atr = float(df["atr"].iloc[entry_idx])
        std = float(df["std"].iloc[entry_idx])
    except:
        atr = tgt_points / 2
        std = tgt_points / 2

    sl_dist = abs(entry_price - sl)

    if tgt_type == "Custom Points":
        tp1 = entry_price + sign * tgt_points
        tp2 = entry_price + sign * tgt_points * 1.5
        tp3 = entry_price + sign * tgt_points * 2.0

    elif tgt_type == "ATR Based":
        tp1 = entry_price + sign * atr * atr_mult
        tp2 = entry_price + sign * atr * atr_mult * 1.5
        tp3 = entry_price + sign * atr * atr_mult * 2.0

    elif tgt_type in ("Auto Target", "Volatility Based"):
        base = max(atr * atr_mult, std * vol_mult)
        tp1  = entry_price + sign * base
        tp2  = entry_price + sign * base * 1.618
        tp3  = entry_price + sign * base * 2.618

    elif tgt_type == "Risk-Reward Based":
        tp1 = entry_price + sign * sl_dist * rr_ratio
        tp2 = entry_price + sign * sl_dist * rr_ratio * 1.618
        tp3 = entry_price + sign * sl_dist * rr_ratio * 2.618

    elif tgt_type == "EMA Crossover Exit":
        # Target at slow EMA + ATR buffer
        ema_val = float(df["ema_slow"].iloc[entry_idx])
        tp1 = ema_val + sign * atr * 0.5
        tp2 = entry_price + sign * atr * atr_mult
        tp3 = entry_price + sign * atr * atr_mult * 2

    elif tgt_type in ("Trailing Target (Display Only)", "Trail with Swing High/Low",
                      "Trail with Candle High/Low"):
        tp1 = entry_price + sign * tgt_points
        tp2 = entry_price + sign * tgt_points * 1.5
        tp3 = entry_price + sign * tgt_points * 2.0

    elif tgt_type == "Support/Resistance Based":
        support, resistance = compute_support_resistance(df.iloc[:entry_idx+1])
        if trade_type == "buy":
            tp1 = resistance
            tp2 = resistance + atr * 1.0
            tp3 = resistance + atr * 2.0
        else:
            tp1 = support
            tp2 = support - atr * 1.0
            tp3 = support - atr * 2.0

    elif tgt_type == "Book Profit: T1 then T2":
        tp1 = entry_price + sign * tgt_points
        tp2 = entry_price + sign * tgt_points * 2.0
        tp3 = entry_price + sign * tgt_points * 3.0

    else:
        tp1 = entry_price + sign * tgt_points
        tp2 = entry_price + sign * tgt_points * 1.5
        tp3 = entry_price + sign * tgt_points * 2.0

    return tp1, tp2, tp3

def update_trailing_sl(current_sl: float, current_price: float,
                       trade_type: str, df: pd.DataFrame,
                       cfg: Dict, current_idx: int) -> float:
    """Update SL for trailing types during trade lifecycle."""
    sl_type  = cfg.get("sl_type", "Custom Points")
    sl_pts   = cfg.get("sl_points", 10.0)
    atr_mult = cfg.get("atr_mult_sl", 1.5)
    atr      = float(df["atr"].iloc[current_idx]) if current_idx < len(df) else sl_pts

    if sl_type == "Trailing SL":
        if trade_type == "buy":
            new_sl = current_price - sl_pts
            return max(current_sl, new_sl)
        else:
            new_sl = current_price + sl_pts
            return min(current_sl, new_sl)

    elif sl_type == "ATR Based" and "trail" in cfg.get("sl_trail", ""):
        if trade_type == "buy":
            new_sl = current_price - atr * atr_mult
            return max(current_sl, new_sl)
        else:
            new_sl = current_price + atr * atr_mult
            return min(current_sl, new_sl)

    elif sl_type == "Trail with Swing Low/High":
        swing_lo = float(df["swing_lo"].iloc[current_idx])
        swing_hi = float(df["swing_hi"].iloc[current_idx])
        if trade_type == "buy":
            return max(current_sl, swing_lo - atr * 0.1)
        else:
            return min(current_sl, swing_hi + atr * 0.1)

    elif sl_type == "Trail with Candle Low/High":
        row = df.iloc[current_idx]
        if trade_type == "buy":
            return max(current_sl, float(row["Low"]) - atr * 0.05)
        else:
            return min(current_sl, float(row["High"]) + atr * 0.05)

    return current_sl

def check_ema_reversal_exit(df: pd.DataFrame, idx: int, trade_type: str) -> bool:
    """Check if EMA has reversed (for EMA Reverse Crossover SL/Target)."""
    if idx < 1:
        return False
    fast_now  = df["ema_fast"].iloc[idx]
    fast_prev = df["ema_fast"].iloc[idx-1]
    slow_now  = df["ema_slow"].iloc[idx]
    slow_prev = df["ema_slow"].iloc[idx-1]

    if trade_type == "buy":
        return fast_now < slow_now and fast_prev >= slow_prev
    else:
        return fast_now > slow_now and fast_prev <= slow_prev


# ================================================================
# 10. BACKTESTING ENGINE
# ================================================================
def run_backtest(df_full: pd.DataFrame, df_display: pd.DataFrame,
                 cfg: Dict) -> Dict:
    """
    Core backtest engine.
    - EMA signal on candle N → entry at candle N+1 OPEN
    - For BUY: check SL against candle Low FIRST, then target against High (conservative)
    - For SELL: check SL against candle High FIRST, then target against Low (conservative)
    - Simple Buy/Sell: immediate entry on current close (no N+1 wait)
    Returns dict with trades, violations, summary stats.
    """
    strategy = cfg.get("strategy", "EMA Crossover")
    immediate_entry = strategy in ("Simple Buy", "Simple Sell")

    # Compute indicators on FULL df (with warmup) for accuracy
    df_full = add_indicators(df_full, cfg)

    # Get signal indices aligned to display_df
    signals_full, _ = get_signals(df_full, cfg)

    # Align signals to display slice
    # Find start of display period in full df
    if len(df_display) > 0 and len(df_full) >= len(df_display):
        start_iloc = len(df_full) - len(df_display)
    else:
        start_iloc = 0

    df = df_full.iloc[start_iloc:].copy()
    signals = signals_full.iloc[start_iloc:].copy()

    trades     = []
    violations = []
    position   = None   # active trade dict

    for i in range(len(df)):
        row    = df.iloc[i]
        idx    = df.index[i]
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])

        # ── Manage open position ──────────────────────────────────
        if position is not None:
            entry_price = position["entry_price"]
            sl          = position["sl"]
            tp1         = position["tp1"]
            tp2         = position["tp2"]
            tp3         = position["tp3"]
            trade_type  = position["type"]

            # Update trailing SL
            sl = update_trailing_sl(sl, c, trade_type, df, cfg, i)
            position["sl"] = sl

            exit_price  = None
            exit_reason = None
            violated    = False

            if trade_type == "buy":
                # Conservative: check SL (Low) FIRST, then Target (High)
                sl_hit  = l <= sl
                tp1_hit = h >= tp1

                if sl_hit and tp1_hit:
                    # Determine which came first (conservative: SL wins)
                    violated    = True
                    exit_price  = sl
                    exit_reason = "SL Hit (conservative — ambiguous candle)"
                elif sl_hit:
                    exit_price  = sl
                    exit_reason = "SL Hit"
                elif tp1_hit:
                    exit_price  = tp1
                    exit_reason = "TP1 Hit"

                # EMA reversal exit
                if exit_reason is None and cfg.get("target_type") == "EMA Crossover Exit":
                    if check_ema_reversal_exit(df, i, "buy"):
                        exit_price  = c
                        exit_reason = "EMA Reversal Exit"

            else:  # sell
                # Conservative: check SL (High) FIRST
                sl_hit  = h >= sl
                tp1_hit = l <= tp1

                if sl_hit and tp1_hit:
                    violated    = True
                    exit_price  = sl
                    exit_reason = "SL Hit (conservative — ambiguous candle)"
                elif sl_hit:
                    exit_price  = sl
                    exit_reason = "SL Hit"
                elif tp1_hit:
                    exit_price  = tp1
                    exit_reason = "TP1 Hit"

                if exit_reason is None and cfg.get("target_type") == "EMA Crossover Exit":
                    if check_ema_reversal_exit(df, i, "sell"):
                        exit_price  = c
                        exit_reason = "EMA Reversal Exit"

            if exit_price is not None:
                sign = 1 if trade_type == "buy" else -1
                pnl  = sign * (exit_price - entry_price) * cfg.get("quantity", 1)

                trade = {
                    **position,
                    "exit_time":   idx,
                    "exit_price":  round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "exit_high":   h,
                    "exit_low":    l,
                    "pnl":         round(pnl, 2),
                    "pnl_pct":     round(sign * (exit_price - entry_price) / entry_price * 100, 2),
                    "violated":    violated,
                }
                trades.append(trade)
                if violated:
                    violations.append(trade)
                position = None

        # ── Check for new entry signal ────────────────────────────
        if position is None:
            sig = signals.iloc[i]

            if immediate_entry:
                # Simple Buy/Sell: enter on close immediately
                if strategy == "Simple Buy":
                    sig = 1
                elif strategy == "Simple Sell":
                    sig = -1

            if sig != 0:
                # For EMA crossover & anticipatory: entry on N+1 open
                if not immediate_entry:
                    if i + 1 >= len(df):
                        continue  # no next candle
                    entry_row   = df.iloc[i + 1]
                    entry_price = float(entry_row["Open"])
                    entry_time  = df.index[i + 1]
                    entry_idx_i = i + 1
                else:
                    entry_price = c
                    entry_time  = idx
                    entry_idx_i = i

                trade_type = "buy" if sig == 1 else "sell"
                sl  = calc_initial_sl(entry_price, trade_type, df, cfg, entry_idx_i)
                tp1, tp2, tp3 = calc_initial_target(entry_price, trade_type, df, cfg, entry_idx_i, sl)

                entry_row = df.iloc[entry_idx_i]
                position = {
                    "entry_time":  entry_time,
                    "entry_price": round(entry_price, 2),
                    "type":        trade_type,
                    "sl":          round(sl, 2),
                    "tp1":         round(tp1, 2),
                    "tp2":         round(tp2, 2),
                    "tp3":         round(tp3, 2),
                    "entry_high":  float(entry_row["High"]),
                    "entry_low":   float(entry_row["Low"]),
                    "signal_reason": strategy,
                    "quantity":    cfg.get("quantity", 1),
                }

    # Close any open position at last candle
    if position is not None:
        last   = df.iloc[-1]
        c      = float(last["Close"])
        sign   = 1 if position["type"] == "buy" else -1
        pnl    = sign * (c - position["entry_price"]) * cfg.get("quantity", 1)
        trades.append({
            **position,
            "exit_time":   df.index[-1],
            "exit_price":  round(c, 2),
            "exit_reason": "End of Data",
            "exit_high":   float(last["High"]),
            "exit_low":    float(last["Low"]),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(sign * (c - position["entry_price"]) / position["entry_price"] * 100, 2),
            "violated":    False,
        })

    # ── Compute summary ──────────────────────────────────────────
    total   = len(trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers  = [t for t in trades if t["pnl"] <= 0]
    total_pnl   = sum(t["pnl"] for t in trades)
    accuracy    = round(len(winners) / total * 100, 1) if total > 0 else 0
    avg_win     = sum(t["pnl"] for t in winners) / len(winners) if winners else 0
    avg_loss    = sum(t["pnl"] for t in losers)  / len(losers)  if losers  else 0
    profit_factor = abs(avg_win * len(winners) / (avg_loss * len(losers))) if losers and losers[0]["pnl"] != 0 else 0

    return {
        "trades":       trades,
        "violations":   violations,
        "summary": {
            "total_trades":   total,
            "winners":        len(winners),
            "losers":         len(losers),
            "total_pnl":      round(total_pnl, 2),
            "accuracy":       accuracy,
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 2),
            "violation_count": len(violations),
            "max_win":        round(max((t["pnl"] for t in trades), default=0), 2),
            "max_loss":       round(min((t["pnl"] for t in trades), default=0), 2),
        },
        "df": df,  # with indicators
    }


# ================================================================
# 11. DHAN BROKER INTEGRATION
# ================================================================
def build_dhan_client(client_id: str, access_token: str):
    """Initialize Dhan client (uses dhanhq library)."""
    try:
        from dhanhq import dhanhq
        return dhanhq(client_id=client_id, access_token=access_token)
    except ImportError:
        return None
    except Exception:
        return None

def register_dhan_ip(dhan_client) -> Dict:
    """Register IP for order placement (SEBI mandate)."""
    try:
        if dhan_client is None:
            return {"success": False, "msg": "Dhan client not initialized"}
        # dhanhq handles IP registration internally on auth
        return {"success": True, "msg": "IP registered (handled by dhanhq on auth)"}
    except Exception as e:
        return {"success": False, "msg": str(e)}

def place_dhan_equity_order(dhan_client, cfg: Dict, trade_type: str,
                             entry_price: float) -> Dict:
    """
    Place equity intraday/delivery order via Dhan.
    trade_type: 'buy' or 'sell'
    """
    if dhan_client is None:
        return {"success": False, "msg": "Dhan not initialized"}

    try:
        from dhanhq import dhanhq as dhan_lib

        seg_map   = {"NSE": "NSE_EQ", "BSE": "BSE_EQ"}
        seg       = seg_map.get(cfg.get("exchange", "NSE"), "NSE_EQ")
        prod_map  = {"Intraday": "INTRADAY", "Delivery": "CNC"}
        prod      = prod_map.get(cfg.get("product_type", "Intraday"), "INTRADAY")
        ord_map   = {"Market Order": "MARKET", "Limit Order": "LIMIT"}
        ord_type  = ord_map.get(cfg.get("entry_order_type", "Market Order"), "MARKET")

        txn = "BUY" if trade_type == "buy" else "SELL"
        price = entry_price if ord_type == "LIMIT" else 0

        response = dhan_client.place_order(
            security_id    = str(cfg.get("security_id", "1594")),
            exchange_segment = seg,
            transaction_type = txn,
            quantity       = int(cfg.get("quantity", 1)),
            order_type     = ord_type,
            product_type   = prod,
            price          = price,
            trigger_price  = 0,
            validity       = "DAY",
        )
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "msg": str(e)}

def place_dhan_options_order(dhan_client, cfg: Dict,
                              algo_signal: str, entry_price: float) -> Dict:
    """
    Place options order:
    - algo_signal 'buy'  → CE BUY order
    - algo_signal 'sell' → PE BUY order
    (User is only a buyer, never a seller)
    """
    if dhan_client is None:
        return {"success": False, "msg": "Dhan not initialized"}

    try:
        seg_map   = {"NSE_FNO": "NSE_FNO", "BSE_FNO": "BSE_FNO"}
        seg       = seg_map.get(cfg.get("fno_segment", "NSE_FNO"), "NSE_FNO")
        ord_map   = {"Market Order": "MARKET", "Limit Order": "LIMIT"}
        ord_type  = ord_map.get(cfg.get("options_entry_order_type", "Market Order"), "MARKET")

        if algo_signal == "buy":
            sec_id = str(cfg.get("ce_security_id", ""))
        else:
            sec_id = str(cfg.get("pe_security_id", ""))

        price = entry_price if ord_type == "LIMIT" else 0

        response = dhan_client.place_order(
            security_id      = sec_id,
            exchange_segment  = seg,
            transaction_type  = "BUY",
            quantity          = int(cfg.get("options_quantity", 65)),
            order_type        = ord_type,
            product_type      = "INTRADAY",
            price             = price,
            trigger_price     = 0,
            validity          = "DAY",
        )
        return {"success": True, "response": response, "option": "CE" if algo_signal == "buy" else "PE"}
    except Exception as e:
        return {"success": False, "msg": str(e)}

def place_exit_order(dhan_client, cfg: Dict, position: Dict,
                     exit_price: float) -> Dict:
    """Place exit order (opposite of entry)."""
    if dhan_client is None:
        return {"success": False, "msg": "Dhan not initialized"}

    try:
        ord_map   = {"Market Order": "MARKET", "Limit Order": "LIMIT"}
        ord_type  = ord_map.get(cfg.get("exit_order_type", "Market Order"), "MARKET")
        price = exit_price if ord_type == "LIMIT" else 0

        trade_type = position.get("type", "buy")

        if cfg.get("options_enabled", False):
            sec_id = position.get("option_security_id", "")
            seg    = cfg.get("fno_segment", "NSE_FNO")
            txn    = "SELL"  # exit options position
            qty    = int(cfg.get("options_quantity", 65))
            prod   = "INTRADAY"
        else:
            seg_map = {"NSE": "NSE_EQ", "BSE": "BSE_EQ"}
            seg    = seg_map.get(cfg.get("exchange", "NSE"), "NSE_EQ")
            txn    = "SELL" if trade_type == "buy" else "BUY"
            qty    = int(cfg.get("quantity", 1))
            prod_map = {"Intraday": "INTRADAY", "Delivery": "CNC"}
            prod   = prod_map.get(cfg.get("product_type", "Intraday"), "INTRADAY")
            sec_id = str(cfg.get("security_id", "1594"))

        response = dhan_client.place_order(
            security_id      = sec_id,
            exchange_segment  = seg,
            transaction_type  = txn,
            quantity          = qty,
            order_type        = ord_type,
            product_type      = prod,
            price             = price,
            trigger_price     = 0,
            validity          = "DAY",
        )
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "msg": str(e)}


# ================================================================
# 12. LIVE TRADING ENGINE  (runs in background thread)
# ================================================================
def _get_candle_interval_seconds(interval: str) -> int:
    """Return interval in seconds."""
    mapping = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400, "1wk": 604800}
    return mapping.get(interval, 300)

def _is_candle_closed(interval: str) -> bool:
    """
    Check if the current time is at a candle boundary (multiple of interval).
    Only check strategy signals at exact candle close boundaries.
    """
    now = datetime.now(IST)
    if interval == "1m":
        return True  # every tick = 1 candle
    elif interval == "5m":
        return now.second == 0 and now.minute % 5 == 0
    elif interval == "15m":
        return now.second == 0 and now.minute % 15 == 0
    elif interval == "1h":
        return now.second == 0 and now.minute == 0
    elif interval == "1d":
        return now.second == 0 and now.minute == 0 and now.hour == 15  # NSE close
    elif interval == "1wk":
        return now.weekday() == 4 and now.hour == 15 and now.minute == 0 and now.second == 0
    return True

def live_trading_loop(stop_event: threading.Event):
    """
    Background thread for live trading.
    - Fetches data every 1.5s (respects yfinance rate limits)
    - Checks SL/Target against LTP (tick-by-tick)
    - Generates entry signals only at candle close boundaries
    - Entry at next candle's open (simulated from LTP)
    """
    cfg      = ts_get("config", {})
    symbol   = cfg.get("symbol", "^NSEI")
    interval = cfg.get("interval", "5m")
    period   = cfg.get("period", "1d")
    strategy = cfg.get("strategy", "EMA Crossover")
    immediate_entry = strategy in ("Simple Buy", "Simple Sell")

    dhan_enabled = cfg.get("dhan_enabled", False)
    dhan_client  = None
    if dhan_enabled:
        dhan_client = build_dhan_client(
            cfg.get("dhan_client_id", ""),
            cfg.get("dhan_access_token", "")
        )
        if dhan_client:
            ts_log("Dhan client initialized", "info")
            register_dhan_ip(dhan_client)

    # State for this session
    pending_entry_signal = None  # {type: 'buy'/'sell', at_next_open: True}
    last_signal_check_time = None

    ts_log(f"Live trading started | {symbol} | {interval} | {period} | {strategy}", "info")

    tick = 0
    while not stop_event.is_set():
        tick += 1
        ts_set("tick_count", tick)

        try:
            # ── 1. Fetch latest data ──────────────────────────────
            df_full = fetch_ohlcv(symbol, interval, period, warmup=True)
            if df_full is None or len(df_full) < 5:
                ts_log("Data fetch returned empty — retrying...", "warn")
                time.sleep(1.5)
                continue

            df_full = add_indicators(df_full, cfg)

            # Last candle
            last_row   = df_full.iloc[-1]
            ltp        = float(last_row["Close"])
            ema_f      = float(last_row["ema_fast"])
            ema_s      = float(last_row["ema_slow"])
            last_time  = df_full.index[-1]

            ts_set("ltp",          ltp)
            ts_set("ema_fast_val", ema_f)
            ts_set("ema_slow_val", ema_s)
            ts_set("last_candle", {
                "time":   str(last_time),
                "open":   round(float(last_row["Open"]),  2),
                "high":   round(float(last_row["High"]),  2),
                "low":    round(float(last_row["Low"]),   2),
                "close":  round(ltp, 2),
                "volume": int(last_row.get("Volume", 0)),
                "ema_fast": round(ema_f, 4),
                "ema_slow": round(ema_s, 4),
                "atr":    round(float(last_row["atr"]), 4),
            })

            # ── 2. Elliott Wave analysis ──────────────────────────
            if strategy == "Elliott Wave Auto" or cfg.get("show_waves", True):
                wave_res = analyze_elliott_waves(df_full)
                ts_set("wave_data", wave_res)

            # ── 3. Check SL/Target against LTP (tick-by-tick) ────
            position = ts_get("live_position")
            if position is not None:
                trade_type = position["type"]
                sl         = position["sl"]
                tp1        = position["tp1"]
                tp2        = position["tp2"]

                # Update trailing SL against LTP
                sl = update_trailing_sl(sl, ltp, trade_type, df_full, cfg, len(df_full)-1)
                position["sl"] = sl
                ts_set("live_position", position)

                exit_price  = None
                exit_reason = None

                if trade_type == "buy":
                    if ltp <= sl:
                        exit_price  = ltp
                        exit_reason = "SL Hit (LTP)"
                    elif ltp >= tp1:
                        exit_price  = ltp
                        exit_reason = "TP1 Hit (LTP)"
                else:
                    if ltp >= sl:
                        exit_price  = ltp
                        exit_reason = "SL Hit (LTP)"
                    elif ltp <= tp1:
                        exit_price  = ltp
                        exit_reason = "TP1 Hit (LTP)"

                # EMA reversal check at candle boundary
                if exit_reason is None and cfg.get("target_type") == "EMA Crossover Exit":
                    if check_ema_reversal_exit(df_full, len(df_full)-1, trade_type):
                        exit_price  = ltp
                        exit_reason = "EMA Reversal Exit"

                if exit_price is not None:
                    sign = 1 if trade_type == "buy" else -1
                    pnl  = sign * (exit_price - position["entry_price"]) * position.get("quantity", 1)
                    trade = {
                        **position,
                        "exit_time":   datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price":  round(exit_price, 2),
                        "exit_reason": exit_reason,
                        "pnl":         round(pnl, 2),
                        "pnl_pct":     round(sign * (exit_price - position["entry_price"]) / position["entry_price"] * 100, 2),
                        "source":      "live",
                    }
                    ts_append("live_trades", trade)
                    ts_set("live_position", None)
                    ts_log(f"EXIT {trade_type.upper()} @ {exit_price:.2f} | {exit_reason} | PnL: {pnl:.2f}", "exit")

                    # Place exit order via Dhan
                    if dhan_enabled and dhan_client:
                        ex_res = place_exit_order(dhan_client, cfg, position, exit_price)
                        ts_log(f"Dhan exit order: {ex_res.get('msg', 'OK')}", "info")

                    pending_entry_signal = None

            # ── 4. Execute pending entry (at open of new candle) ──
            if pending_entry_signal is not None and ts_get("live_position") is None:
                # Check candle has advanced
                if last_time != pending_entry_signal.get("signal_candle_time"):
                    # New candle opened — enter at LTP (approximates next open)
                    trade_type = pending_entry_signal["type"]
                    entry_price = ltp

                    # Cooldown check
                    cooldown_until = ts_get("cooldown_until")
                    if cooldown_until and datetime.now(IST) < cooldown_until:
                        ts_log("Cooldown active — skipping entry", "warn")
                        pending_entry_signal = None
                    else:
                        sl  = calc_initial_sl(entry_price, trade_type, df_full, cfg, len(df_full)-1)
                        tp1, tp2, tp3 = calc_initial_target(entry_price, trade_type, df_full, cfg, len(df_full)-1, sl)

                        pos = {
                            "entry_time":  datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                            "entry_price": round(entry_price, 2),
                            "type":        trade_type,
                            "sl":          round(sl, 2),
                            "tp1":         round(tp1, 2),
                            "tp2":         round(tp2, 2),
                            "tp3":         round(tp3, 2),
                            "quantity":    cfg.get("quantity", 1),
                            "signal_reason": strategy,
                        }
                        ts_set("live_position", pos)
                        ts_log(f"ENTRY {trade_type.upper()} @ {entry_price:.2f} | SL:{sl:.2f} | TP1:{tp1:.2f}", "buy" if trade_type == "buy" else "sell")

                        # Place entry order via Dhan
                        if dhan_enabled and dhan_client:
                            if cfg.get("options_enabled", False):
                                res = place_dhan_options_order(dhan_client, cfg, trade_type, entry_price)
                                if res.get("success"):
                                    pos["option_security_id"] = (
                                        cfg.get("ce_security_id") if trade_type == "buy"
                                        else cfg.get("pe_security_id")
                                    )
                            else:
                                res = place_dhan_equity_order(dhan_client, cfg, trade_type, entry_price)
                            ts_log(f"Dhan entry order: {res.get('msg', 'placed')}", "info")

                        pending_entry_signal = None

                        # Set cooldown
                        if cfg.get("cooldown_enabled", True):
                            cooldown_secs = cfg.get("cooldown_seconds", 5)
                            ts_set("cooldown_until",
                                   datetime.now(IST) + timedelta(seconds=cooldown_secs))

            # ── 5. Generate new entry signal at candle boundary ───
            if ts_get("live_position") is None and pending_entry_signal is None:
                candle_closed = _is_candle_closed(interval)
                if candle_closed or immediate_entry:
                    # Avoid re-checking same candle
                    if last_signal_check_time != last_time:
                        last_signal_check_time = last_time

                        if immediate_entry:
                            sig = 1 if strategy == "Simple Buy" else -1
                        else:
                            sigs, _ = get_signals(df_full, cfg)
                            sig = int(sigs.iloc[-1])

                        if sig != 0:
                            # Overlap check
                            if cfg.get("no_overlap", True):
                                last_trades = ts_get("live_trades", [])
                                if last_trades:
                                    last_trade = last_trades[-1]
                                    try:
                                        lt_exit = datetime.strptime(str(last_trade["exit_time"]), "%Y-%m-%d %H:%M:%S")
                                        lt_exit = IST.localize(lt_exit)
                                        if datetime.now(IST) < lt_exit:
                                            ts_log("Overlap prevented — last trade not yet exited", "warn")
                                            time.sleep(1.5)
                                            continue
                                    except:
                                        pass

                            trade_type = "buy" if sig == 1 else "sell"
                            ts_log(f"Signal: {trade_type.upper()} generated on {interval} candle", "info")

                            if immediate_entry:
                                # Enter immediately
                                entry_price = ltp
                                cooldown_until = ts_get("cooldown_until")
                                if not (cooldown_until and datetime.now(IST) < cooldown_until):
                                    sl  = calc_initial_sl(entry_price, trade_type, df_full, cfg, len(df_full)-1)
                                    tp1, tp2, tp3 = calc_initial_target(entry_price, trade_type, df_full, cfg, len(df_full)-1, sl)
                                    pos = {
                                        "entry_time":  datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                                        "entry_price": round(entry_price, 2),
                                        "type":        trade_type,
                                        "sl":          round(sl, 2),
                                        "tp1":         round(tp1, 2),
                                        "tp2":         round(tp2, 2),
                                        "tp3":         round(tp3, 2),
                                        "quantity":    cfg.get("quantity", 1),
                                        "signal_reason": strategy,
                                    }
                                    ts_set("live_position", pos)
                                    ts_log(f"ENTRY {trade_type.upper()} @ {entry_price:.2f} | SL:{sl:.2f}", "buy" if trade_type == "buy" else "sell")

                                    if dhan_enabled and dhan_client:
                                        if cfg.get("options_enabled", False):
                                            res = place_dhan_options_order(dhan_client, cfg, trade_type, entry_price)
                                        else:
                                            res = place_dhan_equity_order(dhan_client, cfg, trade_type, entry_price)
                                        ts_log(f"Dhan entry order: {res.get('msg', 'placed')}", "info")

                                    if cfg.get("cooldown_enabled", True):
                                        ts_set("cooldown_until",
                                               datetime.now(IST) + timedelta(seconds=cfg.get("cooldown_seconds", 5)))
                            else:
                                # Queue entry at next candle open
                                pending_entry_signal = {
                                    "type":             trade_type,
                                    "signal_candle_time": last_time,
                                }
                                ts_log(f"Signal queued: {trade_type.upper()} — waiting for next candle open", "info")

        except Exception as e:
            ts_log(f"Live loop error: {str(e)}", "warn")
            ts_set("live_error", str(e))

        time.sleep(1.5)  # Respect yfinance API rate limits

    ts_log("Live trading stopped", "warn")
    ts_set("live_running", False)

def start_live_trading(cfg: Dict):
    ts_set("config",        cfg)
    ts_set("live_running",  True)
    ts_set("live_error",    None)
    stop_ev = threading.Event()
    ts_set("stop_event",    stop_ev)
    t = threading.Thread(target=live_trading_loop, args=(stop_ev,), daemon=True)
    ts_set("live_thread",   t)
    t.start()

def stop_live_trading():
    ev = ts_get("stop_event")
    if ev:
        ev.set()
    ts_set("live_running", False)

def squareoff_position():
    pos = ts_get("live_position")
    if pos:
        ltp = ts_get("ltp") or pos["entry_price"]
        sign = 1 if pos["type"] == "buy" else -1
        pnl  = sign * (ltp - pos["entry_price"]) * pos.get("quantity", 1)
        trade = {
            **pos,
            "exit_time":   datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price":  round(float(ltp), 2),
            "exit_reason": "Manual Squareoff",
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(sign * (ltp - pos["entry_price"]) / pos["entry_price"] * 100, 2),
            "source":      "live",
        }
        ts_append("live_trades", trade)
        ts_set("live_position", None)
        ts_log(f"Manual SQUAREOFF {pos['type'].upper()} @ {ltp:.2f} | PnL: {pnl:.2f}", "exit")


# ================================================================
# 13. OPTIMIZATION ENGINE
# ================================================================
def run_optimization(df_full: pd.DataFrame, df_display: pd.DataFrame,
                     base_cfg: Dict, target_accuracy: float) -> List[Dict]:
    """
    Grid search over EMA periods, SL points, Target points.
    Returns all results sorted by accuracy (best first).
    Always shows results even if target accuracy not met.
    """
    results = []

    fast_range = range(5, 21, 2)
    slow_range = range(10, 41, 5)
    sl_range   = [5, 10, 15, 20, 30]
    tgt_range  = [10, 20, 30, 40, 60]

    total_combos = len(fast_range) * len(slow_range) * len(sl_range) * len(tgt_range)

    for fast in fast_range:
        for slow in slow_range:
            if slow <= fast:
                continue
            for sl_pts in sl_range:
                for tgt_pts in tgt_range:
                    cfg = {
                        **base_cfg,
                        "ema_fast":     fast,
                        "ema_slow":     slow,
                        "sl_points":    sl_pts,
                        "target_points": tgt_pts,
                    }
                    try:
                        res = run_backtest(df_full.copy(), df_display.copy(), cfg)
                        s   = res["summary"]
                        if s["total_trades"] < 2:
                            continue
                        results.append({
                            "ema_fast":    fast,
                            "ema_slow":    slow,
                            "sl_points":   sl_pts,
                            "tgt_points":  tgt_pts,
                            "total_trades": s["total_trades"],
                            "accuracy":    s["accuracy"],
                            "total_pnl":   s["total_pnl"],
                            "profit_factor": s["profit_factor"],
                            "max_win":     s["max_win"],
                            "max_loss":    s["max_loss"],
                            "meets_target": s["accuracy"] >= target_accuracy,
                        })
                    except:
                        pass

    results.sort(key=lambda x: (-x["accuracy"], -x["total_pnl"]))
    return results


# ================================================================
# 14. CHART GENERATION
# ================================================================
def build_chart(df: pd.DataFrame, trades: List[Dict],
                cfg: Dict, show_ema: bool = True,
                title: str = "Price Chart") -> go.Figure:
    """
    Build interactive Plotly chart with:
    - Candlestick OHLCV
    - EMA overlays
    - Trade entry/exit markers with SL/TP lines
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=["", "Volume"]
    )

    # ── Candlestick ──────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
        increasing_fillcolor="#0d2818",
        decreasing_fillcolor="#2d1a1a",
        line=dict(width=1),
    ), row=1, col=1)

    # ── EMA lines ─────────────────────────────────────────────
    if show_ema and "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_fast"],
            name=f"EMA {cfg.get('ema_fast', 9)}",
            line=dict(color="#f0a500", width=1.5),
            opacity=0.9,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_slow"],
            name=f"EMA {cfg.get('ema_slow', 15)}",
            line=dict(color="#58a6ff", width=1.5),
            opacity=0.9,
        ), row=1, col=1)

    # ── Volume bars ──────────────────────────────────────────────
    if "Volume" in df.columns:
        colors = ["#0d2818" if c >= o else "#2d1a1a"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.6,
        ), row=2, col=1)

    # ── Trade markers ─────────────────────────────────────────────
    for trade in trades:
        try:
            entry_t = pd.to_datetime(trade["entry_time"])
            exit_t  = pd.to_datetime(trade["exit_time"])
            ep      = trade["entry_price"]
            xp      = trade["exit_price"]
            tp      = trade["type"]
            pnl     = trade["pnl"]

            e_color = "#3fb950" if tp == "buy" else "#f85149"
            x_color = "#3fb950" if pnl >= 0 else "#f85149"
            e_sym   = "triangle-up"   if tp == "buy" else "triangle-down"
            x_sym   = "square"

            fig.add_trace(go.Scatter(
                x=[entry_t], y=[ep],
                mode="markers+text",
                marker=dict(symbol=e_sym, size=12, color=e_color,
                            line=dict(color="white", width=1)),
                text=[f"{'B' if tp=='buy' else 'S'} {ep:.1f}"],
                textposition="top center",
                textfont=dict(size=9, color=e_color),
                name=f"Entry {tp.upper()}",
                showlegend=False,
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[exit_t], y=[xp],
                mode="markers+text",
                marker=dict(symbol=x_sym, size=10, color=x_color,
                            line=dict(color="white", width=1)),
                text=[f"X {xp:.1f}"],
                textposition="bottom center",
                textfont=dict(size=9, color=x_color),
                name="Exit",
                showlegend=False,
            ), row=1, col=1)

            # SL line
            fig.add_shape(
                type="line",
                x0=entry_t, x1=exit_t,
                y0=trade["sl"], y1=trade["sl"],
                line=dict(color="#f85149", width=1, dash="dot"),
                row=1, col=1
            )
            # TP1 line
            fig.add_shape(
                type="line",
                x0=entry_t, x1=exit_t,
                y0=trade["tp1"], y1=trade["tp1"],
                line=dict(color="#3fb950", width=1, dash="dot"),
                row=1, col=1
            )
        except:
            pass

    # ── Layout ─────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e6edf3", size=14)),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(
            gridcolor="#21262d", zerolinecolor="#21262d",
            rangeslider=dict(visible=False),
            showline=True, linecolor="#30363d",
        ),
        xaxis2=dict(gridcolor="#21262d", showline=True, linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d",
                   showline=True, linecolor="#30363d"),
        yaxis2=dict(gridcolor="#21262d", showline=True, linecolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig

def build_live_chart(df: pd.DataFrame, position: Optional[Dict],
                     cfg: Dict) -> go.Figure:
    """Live chart with current position marked."""
    trades_list = []
    if position:
        trades_list = [dict(
            entry_time=position["entry_time"],
            exit_time=df.index[-1],
            entry_price=position["entry_price"],
            exit_price=ts_get("ltp") or position["entry_price"],
            type=position["type"],
            sl=position["sl"],
            tp1=position["tp1"],
            tp2=position.get("tp2", position["tp1"]),
            pnl=0,
        )]
    return build_chart(df, trades_list, cfg, show_ema=True, title="Live Chart")

def build_wave_chart(df: pd.DataFrame, wave_res: Dict) -> go.Figure:
    """Chart with Elliott Wave pivots and projections."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Close",
        line=dict(color="#58a6ff", width=1.5),
    ))

    pivots = wave_res.get("pivots", pd.DataFrame())
    if len(pivots) > 0:
        h_piv = pivots[pivots["ptype"] == "H"]
        l_piv = pivots[pivots["ptype"] == "L"]

        fig.add_trace(go.Scatter(
            x=h_piv["datetime"], y=h_piv["price"],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=10, color="#f85149"),
            text=["H"] * len(h_piv),
            textposition="top center",
            name="Swing High",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=l_piv["datetime"], y=l_piv["price"],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=10, color="#3fb950"),
            text=["L"] * len(l_piv),
            textposition="bottom center",
            name="Swing Low",
            showlegend=True,
        ))

    # Wave labels
    for w in wave_res.get("waves", []):
        try:
            mid_price = (w["start_price"] + w["end_price"]) / 2
            mid_time  = pd.to_datetime(w["start_time"]) + (
                pd.to_datetime(w["end_time"]) - pd.to_datetime(w["start_time"])
            ) / 2
            fig.add_annotation(
                x=mid_time, y=mid_price,
                text=f"W{w['wave']}",
                showarrow=False,
                font=dict(color="#e3b341", size=12, family="monospace"),
            )
        except:
            pass

    # Next target line
    nt = wave_res.get("next_target")
    if nt:
        fig.add_hline(y=nt,
                      line=dict(color="#e3b341", width=1, dash="dash"),
                      annotation_text=f"Wave Target: {nt:.1f}")

    fig.update_layout(
        title="Elliott Wave Analysis",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
    )
    return fig


# ================================================================
# 15. HTML HELPERS
# ================================================================
def ltp_bar_html(ticker_name: str, ltp: Optional[float],
                 prev: Optional[float] = None) -> str:
    price_str = f"{ltp:,.2f}" if ltp else "—"
    chg_str   = ""
    if ltp and prev and prev != 0:
        chg  = ltp - prev
        pct  = chg / prev * 100
        cls  = "ltp-change-pos" if chg >= 0 else "ltp-change-neg"
        sign = "+" if chg >= 0 else ""
        chg_str = f'<span class="{cls}">{sign}{chg:.2f} ({sign}{pct:.2f}%)</span>'

    ts_ist = datetime.now(IST).strftime("%H:%M:%S IST")
    return f"""
    <div class="ltp-bar">
      <div>
        <div class="ltp-ticker">{ticker_name}</div>
        <div class="ltp-price">{price_str}</div>
      </div>
      {chg_str}
      <div style="margin-left:auto;color:#8b949e;font-size:11px">{ts_ist}</div>
    </div>
    """

def config_box_html(items: Dict) -> str:
    rows = ""
    for k, v in items.items():
        rows += f'<div class="config-row"><span class="config-key">{k}</span><span class="config-val">{v}</span></div>'
    return f'<div class="config-box">{rows}</div>'

def badge(text: str, kind: str = "hold") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'

def alert_html(msg: str, kind: str = "info") -> str:
    return f'<div class="alert alert-{kind}">{msg}</div>'

def summary_grid_html(items: Dict) -> str:
    cards = ""
    for label, value in items.items():
        color = ""
        if isinstance(value, (int, float)):
            if "PnL" in label or "Profit" in label:
                color = "color:#3fb950" if value >= 0 else "color:#f85149"
        cards += f"""
        <div class="summary-card">
          <div class="summary-label">{label}</div>
          <div class="summary-value" style="{color}">{value}</div>
        </div>"""
    return f'<div class="summary-grid">{cards}</div>'

def position_card_html(pos: Dict, ltp: float) -> str:
    trade_type = pos.get("type", "buy")
    cls        = "" if trade_type == "buy" else "sell-pos"
    ep         = pos.get("entry_price", 0)
    sl         = pos.get("sl", 0)
    tp1        = pos.get("tp1", 0)
    qty        = pos.get("quantity", 1)
    sign       = 1 if trade_type == "buy" else -1
    unreal_pnl = sign * (ltp - ep) * qty
    pnl_cls    = "green" if unreal_pnl >= 0 else "red"
    badge_str  = badge("BUY", "buy") if trade_type == "buy" else badge("SELL", "sell")

    return f"""
    <div class="pos-card {cls}">
      <div class="pos-title">{badge_str} Active Position — {pos.get('signal_reason','')}</div>
      <div class="pos-grid">
        <div class="pos-field">
          <div class="pos-flabel">Entry</div>
          <div class="pos-fval">{ep:.2f}</div>
        </div>
        <div class="pos-field">
          <div class="pos-flabel">LTP</div>
          <div class="pos-fval">{ltp:.2f}</div>
        </div>
        <div class="pos-field">
          <div class="pos-flabel">Unrealized P&L</div>
          <div class="pos-fval {pnl_cls}">{unreal_pnl:+.2f}</div>
        </div>
        <div class="pos-field">
          <div class="pos-flabel">Stop Loss</div>
          <div class="pos-fval red">{sl:.2f}</div>
        </div>
        <div class="pos-field">
          <div class="pos-flabel">Target 1</div>
          <div class="pos-fval green">{tp1:.2f}</div>
        </div>
        <div class="pos-field">
          <div class="pos-flabel">Qty</div>
          <div class="pos-fval">{qty}</div>
        </div>
        <div class="pos-field">
          <div class="pos-flabel">Entry Time</div>
          <div class="pos-fval" style="font-size:11px">{pos.get('entry_time','—')}</div>
        </div>
      </div>
    </div>
    """

def wave_cards_html(wave_res: Dict) -> str:
    waves      = wave_res.get("waves", [])
    curr_wave  = wave_res.get("current_wave", "")
    completed  = {w["wave"] for w in waves}
    all_waves  = ["1","2","3","4","5","A","B","C"]
    cards      = ""

    wave_colors = {
        "1": "#58a6ff", "2": "#e3b341", "3": "#3fb950",
        "4": "#e3b341", "5": "#58a6ff", "A": "#f85149",
        "B": "#3fb950", "C": "#f85149",
    }

    for wn in all_waves:
        if wn in completed:
            w = next((x for x in waves if x["wave"] == wn), None)
            if w:
                pct  = w.get("pct_move", 0)
                dirn = "↑" if w["direction"] == "Up" else "↓"
                cls  = "wave-complete"
                info = f"{dirn} {pct:.1f}%"
            else:
                cls  = "wave-complete"
                info = "Done"
        elif wn == curr_wave:
            cls  = "wave-active"
            info = "In Progress"
        else:
            cls  = "wave-pending"
            info = "Pending"

        col = wave_colors.get(wn, "#8b949e")
        cards += f"""
        <div class="wave-card {cls}">
          <div class="wave-name" style="color:{col}">W{wn}</div>
          <div class="wave-info">{info}</div>
        </div>"""

    return f'<div class="wave-grid">{cards}</div>'

def trades_table_html(trades: List[Dict], show_violations: bool = True) -> str:
    if not trades:
        return '<div class="alert alert-info">No trades yet.</div>'

    headers = ["#","Type","Entry Time","Entry","SL","TP1","Exit Time","Exit","Exit Reason","High","Low","PnL","PnL%","Qty"]
    th_row  = "".join(f"<th>{h}</th>" for h in headers)
    rows    = ""

    for i, t in enumerate(trades, 1):
        viol_cls  = "violation-row" if t.get("violated") else ""
        pnl       = t.get("pnl", 0)
        pnl_cls   = "pnl-pos" if pnl >= 0 else "pnl-neg"
        trade_type = t.get("type", "")
        type_badge = badge("BUY","buy") if trade_type=="buy" else badge("SELL","sell")

        rows += f"""
        <tr class="{viol_cls}">
          <td>{i}</td>
          <td>{type_badge}</td>
          <td style="font-size:10px">{str(t.get('entry_time',''))[:16]}</td>
          <td>{t.get('entry_price','')}</td>
          <td style="color:#f85149">{t.get('sl','')}</td>
          <td style="color:#3fb950">{t.get('tp1','')}</td>
          <td style="font-size:10px">{str(t.get('exit_time',''))[:16]}</td>
          <td>{t.get('exit_price','')}</td>
          <td style="font-size:11px">{t.get('exit_reason','')}</td>
          <td>{t.get('exit_high', t.get('entry_high',''))}</td>
          <td>{t.get('exit_low',  t.get('entry_low',''))}</td>
          <td class="{pnl_cls}">{pnl:+.2f}</td>
          <td class="{pnl_cls}">{t.get('pnl_pct',0):+.2f}%</td>
          <td>{t.get('quantity',1)}</td>
        </tr>"""

    viol_count = sum(1 for t in trades if t.get("violated"))
    viol_note  = ""
    if show_violations and viol_count > 0:
        viol_note = f"""
        <div class="alert alert-warn">
          ⚠️ <strong>{viol_count} trade(s)</strong> had ambiguous candles where both SL and
          target were hit in the same bar — highlighted in red. Conservative (SL-first) exit applied.
          These diverge most from live trading results.
        </div>"""

    return f"""
    {viol_note}
    <div class="trade-wrap">
      <table class="trade-table">
        <thead><tr>{th_row}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """

def log_html(log_entries: List[Dict]) -> str:
    lines = ""
    for e in reversed(log_entries[-60:]):
        cls = {"buy":"log-buy","sell":"log-sell","exit":"log-exit",
               "warn":"log-warn","info":"log-info"}.get(e.get("level","info"),"log-info")
        lines += f'<div class="{cls}">[{e["time"]}] {e["msg"]}</div>'
    return f'<div class="log-box">{lines}</div>'


# ================================================================
# 16. SIDEBAR — Configuration Panel
# ================================================================
def render_sidebar() -> Dict:
    """Render sidebar and return config dict."""
    with st.sidebar:
        st.markdown("## 📈 Smart Investing")
        st.markdown("---")

        # ── Instrument ──────────────────────────────────────────
        st.markdown("### 🔭 Instrument")
        ticker_name = st.selectbox("Ticker", list(TICKERS.keys()), key="sb_ticker")
        custom_tick = ""
        if ticker_name == "Custom":
            custom_tick = st.text_input("Custom Symbol (e.g. RELIANCE.NS)", key="sb_custom")
        symbol = get_yf_symbol(ticker_name, custom_tick)

        interval = st.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()),
                                 index=1, key="sb_interval")
        period   = st.selectbox("Period",
                                 TIMEFRAME_PERIODS.get(interval, ["1d"]),
                                 key="sb_period")

        st.markdown("---")

        # ── Strategy ──────────────────────────────────────────
        st.markdown("### 🎯 Strategy")
        strategy = st.selectbox("Strategy", STRATEGIES, key="sb_strategy")

        ema_fast = 9
        ema_slow = 15
        crossover_type   = "Simple Crossover"
        custom_candle_sz = 10.0
        atr_candle_mult  = 0.5
        min_angle        = 0.0
        use_angle_filter = False

        if strategy in ("EMA Crossover", "Anticipatory EMA Crossover"):
            c1, c2 = st.columns(2)
            with c1:
                ema_fast = st.number_input("Fast EMA", value=9, min_value=2, max_value=50, key="sb_fast")
            with c2:
                ema_slow = st.number_input("Slow EMA", value=15, min_value=3, max_value=100, key="sb_slow")

            crossover_type = st.selectbox("Crossover Type", CROSSOVER_TYPES, key="sb_co_type")
            if crossover_type == "Custom Candle Size":
                custom_candle_sz = st.number_input("Min Candle Size (pts)", value=10.0, min_value=0.1, key="sb_candle_sz")
            elif crossover_type == "ATR Based Candle Size":
                atr_candle_mult = st.number_input("ATR Multiplier for Candle", value=0.5, min_value=0.1, key="sb_atr_co")

            use_angle_filter = st.checkbox("Min Crossover Angle Filter", value=False, key="sb_angle_en")
            if use_angle_filter:
                min_angle = st.number_input("Min Angle (°, abs value)", value=0.0, min_value=0.0, max_value=90.0, key="sb_angle")

        quantity = st.number_input("Quantity", value=1, min_value=1, key="sb_qty")

        st.markdown("---")

        # ── Stop Loss ──────────────────────────────────────────
        st.markdown("### 🛡️ Stop Loss")
        sl_type = st.selectbox("SL Type", SL_TYPES, key="sb_sl_type")

        sl_points    = 10.0
        rr_ratio     = 2.0
        atr_mult_sl  = 1.5
        vol_mult_sl  = 2.0
        swing_lb     = 10
        t1_pct       = 50.0

        if sl_type == "Custom Points":
            sl_points = st.number_input("SL Points", value=10.0, min_value=0.1, key="sb_sl_pts")
        elif sl_type == "ATR Based":
            atr_mult_sl = st.number_input("ATR Multiplier (SL)", value=1.5, min_value=0.1, key="sb_atr_sl")
        elif sl_type == "Risk-Reward Based":
            rr_ratio = st.number_input("Risk-Reward Ratio", value=2.0, min_value=0.1, key="sb_rr")
        elif sl_type == "Volatility Based":
            vol_mult_sl = st.number_input("Volatility Multiplier (SL)", value=2.0, min_value=0.1, key="sb_vol_sl")
        elif sl_type in ("Trail with Swing Low/High", "Support/Resistance Based"):
            swing_lb = st.number_input("Lookback Candles", value=10, min_value=3, key="sb_swing_lb")

        st.markdown("---")

        # ── Target ──────────────────────────────────────────────
        st.markdown("### 🎯 Target")
        target_type = st.selectbox("Target Type", TARGET_TYPES, key="sb_tgt_type")

        target_points = 20.0
        atr_mult_tgt  = 2.5
        vol_mult_tgt  = 3.0

        if target_type == "Custom Points":
            target_points = st.number_input("Target Points", value=20.0, min_value=0.1, key="sb_tgt_pts")
        elif target_type == "ATR Based":
            atr_mult_tgt = st.number_input("ATR Multiplier (Target)", value=2.5, min_value=0.1, key="sb_atr_tgt")
        elif target_type == "Risk-Reward Based":
            rr_ratio = st.number_input("R:R Ratio", value=2.0, min_value=0.1, key="sb_rr2")
        elif target_type == "Volatility Based":
            vol_mult_tgt = st.number_input("Volatility Multiplier (Target)", value=3.0, min_value=0.1, key="sb_vol_tgt")
        elif target_type == "Book Profit: T1 then T2":
            t1_pct = st.number_input("Book % at T1", value=50.0, min_value=1.0, max_value=100.0, key="sb_t1pct")

        st.markdown("---")

        # ── Live Trading Options ────────────────────────────────
        st.markdown("### ⚡ Live Settings")
        cooldown_enabled  = st.checkbox("Cooldown Between Trades", value=True, key="sb_cd_en")
        cooldown_seconds  = 5
        if cooldown_enabled:
            cooldown_seconds = st.number_input("Cooldown (seconds)", value=5, min_value=1, key="sb_cd_sec")

        no_overlap = st.checkbox("Prevent Overlapping Trades", value=True, key="sb_overlap")

        st.markdown("---")

        # ── Dhan Broker ─────────────────────────────────────────
        st.markdown("### 🏦 Dhan Broker")
        dhan_enabled = st.checkbox("Enable Dhan Broker", value=False, key="sb_dhan")

        options_enabled = False
        exchange        = "NSE"
        product_type    = "Intraday"
        security_id     = "1594"
        entry_ord_type  = "Market Order"
        exit_ord_type   = "Market Order"
        fno_segment     = "NSE_FNO"
        ce_security_id  = ""
        pe_security_id  = ""
        options_qty     = 65
        options_entry   = "Market Order"
        options_exit    = "Market Order"
        dhan_client_id  = ""
        dhan_token      = ""

        if dhan_enabled:
            dhan_client_id = st.text_input("Client ID",     key="sb_cid",   type="password")
            dhan_token     = st.text_input("Access Token",  key="sb_token", type="password")

            options_enabled = st.checkbox("Options Trading", value=False, key="sb_opts")

            if not options_enabled:
                exchange      = st.selectbox("Exchange",     ["NSE","BSE"],        key="sb_exch")
                product_type  = st.selectbox("Product Type", ["Intraday","Delivery"], key="sb_prod")
                security_id   = st.text_input("Security ID", value="1594",          key="sb_secid")
                entry_ord_type= st.selectbox("Entry Order",  ORDER_TYPES, index=0,  key="sb_ent_ord")
                exit_ord_type = st.selectbox("Exit Order",   ORDER_TYPES, index=0,  key="sb_ex_ord")
            else:
                fno_segment    = st.selectbox("FNO Segment", FNO_SEGMENTS,          key="sb_fno")
                ce_security_id = st.text_input("CE Security ID", value="",          key="sb_ce")
                pe_security_id = st.text_input("PE Security ID", value="",          key="sb_pe")
                options_qty    = st.number_input("Options Qty", value=65, min_value=1, key="sb_optqty")
                options_entry  = st.selectbox("Entry Order Type", ORDER_TYPES,      key="sb_opt_ent")
                options_exit   = st.selectbox("Exit Order Type",  ORDER_TYPES,      key="sb_opt_ex")

        st.markdown("---")
        st.markdown('<div style="color:#8b949e;font-size:10px;text-align:center">Smart Investing v2.0 | yfinance + Dhan</div>', unsafe_allow_html=True)

    # Build config dict
    cfg = {
        "ticker_name":    ticker_name,
        "symbol":         symbol,
        "interval":       interval,
        "period":         period,
        "strategy":       strategy,
        "quantity":       quantity,
        "ema_fast":       ema_fast,
        "ema_slow":       ema_slow,
        "crossover_type": crossover_type,
        "custom_candle_size":  custom_candle_sz,
        "atr_candle_mult":     atr_candle_mult,
        "min_crossover_angle": min_angle if use_angle_filter else 0.0,
        "sl_type":        sl_type,
        "sl_points":      sl_points,
        "rr_ratio":       rr_ratio,
        "atr_mult_sl":    atr_mult_sl,
        "vol_mult_sl":    vol_mult_sl,
        "swing_lookback": swing_lb,
        "target_type":    target_type,
        "target_points":  target_points,
        "atr_mult_tgt":   atr_mult_tgt,
        "vol_mult_tgt":   vol_mult_tgt,
        "t1_book_pct":    t1_pct,
        "cooldown_enabled": cooldown_enabled,
        "cooldown_seconds": cooldown_seconds,
        "no_overlap":     no_overlap,
        "dhan_enabled":   dhan_enabled,
        "options_enabled": options_enabled,
        "exchange":       exchange,
        "product_type":   product_type,
        "security_id":    security_id,
        "entry_order_type": entry_ord_type,
        "exit_order_type":  exit_ord_type,
        "fno_segment":    fno_segment,
        "ce_security_id": ce_security_id,
        "pe_security_id": pe_security_id,
        "options_quantity": options_qty,
        "options_entry_order_type": options_entry,
        "options_exit_order_type":  options_exit,
        "dhan_client_id": dhan_client_id,
        "dhan_access_token": dhan_token,
        "atr_period":     14,
    }
    return cfg


# ================================================================
# 17. TAB 1 — BACKTESTING
# ================================================================
def render_backtest_tab(cfg: Dict):
    # ── LTP bar ────────────────────────────────────────────────
    ltp_ph = st.empty()

    col1, col2 = st.columns([3, 1])
    with col2:
        run_btn = st.button("▶ Run Backtest", key="bt_run", use_container_width=True,
                            type="primary")
    with col1:
        st.markdown(f"**{cfg['ticker_name']}** | {cfg['interval']} | {cfg['period']} | Strategy: **{cfg['strategy']}**")

    if "bt_result" not in st.session_state:
        st.session_state.bt_result = None

    if run_btn:
        with st.spinner("Fetching data and running backtest..."):
            df_full, df_display = fetch_ohlcv_with_slice(
                cfg["symbol"], cfg["interval"], cfg["period"]
            )
            if df_full is None:
                st.error("Failed to fetch data. Check ticker symbol and period.")
                return

            result = run_backtest(df_full, df_display or df_full, cfg)
            st.session_state.bt_result  = result
            st.session_state.bt_df      = df_full
            st.session_state.bt_display = df_display or df_full

    # ── LTP ─────────────────────────────────────────────────
    ltp = get_ltp(cfg["symbol"])
    ltp_ph.markdown(ltp_bar_html(cfg["ticker_name"], ltp), unsafe_allow_html=True)

    result = st.session_state.get("bt_result")
    if result is None:
        st.markdown(alert_html("Configure parameters in the sidebar and click <strong>▶ Run Backtest</strong>.", "info"),
                    unsafe_allow_html=True)
        return

    trades  = result["trades"]
    summary = result["summary"]
    df      = result.get("df", st.session_state.get("bt_df"))

    # ── Summary cards ────────────────────────────────────────
    st.markdown(summary_grid_html({
        "Total Trades":   summary["total_trades"],
        "Winners":        summary["winners"],
        "Losers":         summary["losers"],
        "Accuracy":       f"{summary['accuracy']}%",
        "Total P&L":      summary["total_pnl"],
        "Avg Win":        summary["avg_win"],
        "Avg Loss":       summary["avg_loss"],
        "Profit Factor":  summary["profit_factor"],
        "Max Win":        summary["max_win"],
        "Max Loss":       summary["max_loss"],
        "SL/TGT Conflicts": summary["violation_count"],
    }), unsafe_allow_html=True)

    st.markdown("---")

    # ── Chart ─────────────────────────────────────────────────
    if df is not None and len(df) > 0:
        fig = build_chart(df, trades, cfg,
                          title=f"{cfg['ticker_name']} — {cfg['strategy']} Backtest")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Trade Table ──────────────────────────────────────────
    st.markdown("#### 📋 Trade Log")
    if trades:
        st.markdown(trades_table_html(trades, show_violations=True), unsafe_allow_html=True)
    else:
        st.markdown(alert_html("No trades generated with current configuration.", "warn"), unsafe_allow_html=True)


# ================================================================
# 18. TAB 2 — LIVE TRADING
# ================================================================
def render_live_tab(cfg: Dict):
    # ── LTP bar ────────────────────────────────────────────────
    ltp_ph = st.empty()

    # ── Control buttons ──────────────────────────────────────
    is_running = ts_get("live_running", False)
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        start_clicked = st.button("▶ Start", key="lv_start", disabled=is_running,
                                   use_container_width=True, type="primary")
    with b2:
        stop_clicked  = st.button("⏹ Stop",  key="lv_stop",  disabled=not is_running,
                                   use_container_width=True)
    with b3:
        sq_clicked    = st.button("⚡ Squareoff", key="lv_sq", use_container_width=True)
    with b4:
        refresh       = st.button("🔄 Refresh",   key="lv_ref", use_container_width=True)

    if start_clicked:
        start_live_trading(cfg)
        st.success("Live trading started!")

    if stop_clicked:
        stop_live_trading()
        st.warning("Live trading stopped.")

    if sq_clicked:
        squareoff_position()
        st.info("Position squared off.")

    st.markdown("---")

    # ── Config display (shown when live) ─────────────────────
    if is_running:
        st.markdown("#### ⚙️ Active Configuration")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(config_box_html({
                "Ticker":     cfg["ticker_name"],
                "Symbol":     cfg["symbol"],
                "Timeframe":  cfg["interval"],
                "Period":     cfg["period"],
                "Strategy":   cfg["strategy"],
                "Quantity":   cfg["quantity"],
            }), unsafe_allow_html=True)
        with c2:
            st.markdown(config_box_html({
                "EMA Fast":   cfg["ema_fast"],
                "EMA Slow":   cfg["ema_slow"],
                "SL Type":    cfg["sl_type"],
                "SL Points":  cfg["sl_points"],
                "Target":     cfg["target_type"],
                "Tgt Points": cfg["target_points"],
                "Cooldown":   f"{cfg['cooldown_seconds']}s" if cfg["cooldown_enabled"] else "Off",
                "No Overlap": "Yes" if cfg["no_overlap"] else "No",
                "Dhan":       "Enabled" if cfg["dhan_enabled"] else "Disabled",
            }), unsafe_allow_html=True)
        st.markdown("---")

    # ── Live metrics ─────────────────────────────────────────
    ltp      = ts_get("ltp")
    ema_f    = ts_get("ema_fast_val")
    ema_s    = ts_get("ema_slow_val")
    last_c   = ts_get("last_candle")
    position = ts_get("live_position")
    tick     = ts_get("tick_count", 0)

    ltp_ph.markdown(
        ltp_bar_html(cfg["ticker_name"], ltp),
        unsafe_allow_html=True
    )

    # Quick metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">LTP</div><div class="metric-value">{ltp:.2f if ltp else "—"}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">EMA {cfg["ema_fast"]}</div><div class="metric-value">{ema_f:.2f if ema_f else "—"}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">EMA {cfg["ema_slow"]}</div><div class="metric-value">{ema_s:.2f if ema_s else "—"}</div></div>', unsafe_allow_html=True)
    with m4:
        status_badge = badge("RUNNING","run") if is_running else badge("STOPPED","stop")
        tick_str = f"Tick #{tick}" if is_running else "—"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Status</div><div class="metric-value" style="font-size:14px">{status_badge} {tick_str}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Last fetched candle ───────────────────────────────────
    if last_c:
        st.markdown("#### 📊 Last Fetched Candle")
        st.markdown(f"""
        <div class="candle-row">
          <div class="candle-field"><div class="candle-flabel">Time</div><div class="candle-fval" style="font-size:12px">{last_c['time'][:16]}</div></div>
          <div class="candle-field"><div class="candle-flabel">Open</div><div class="candle-fval">{last_c['open']}</div></div>
          <div class="candle-field"><div class="candle-flabel">High</div><div class="candle-fval" style="color:#3fb950">{last_c['high']}</div></div>
          <div class="candle-field"><div class="candle-flabel">Low</div><div class="candle-fval" style="color:#f85149">{last_c['low']}</div></div>
          <div class="candle-field"><div class="candle-flabel">Close</div><div class="candle-fval">{last_c['close']}</div></div>
          <div class="candle-field"><div class="candle-flabel">EMA {cfg['ema_fast']}</div><div class="candle-fval" style="color:#f0a500">{last_c.get('ema_fast','—')}</div></div>
          <div class="candle-field"><div class="candle-flabel">EMA {cfg['ema_slow']}</div><div class="candle-fval" style="color:#58a6ff">{last_c.get('ema_slow','—')}</div></div>
          <div class="candle-field"><div class="candle-flabel">ATR</div><div class="candle-fval">{last_c.get('atr','—')}</div></div>
          <div class="candle-field"><div class="candle-flabel">Volume</div><div class="candle-fval" style="font-size:11px">{last_c.get('volume',0):,}</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Active Position ───────────────────────────────────────
    if position and ltp:
        st.markdown("#### 💼 Active Position")
        st.markdown(position_card_html(position, ltp), unsafe_allow_html=True)
    elif not position and is_running:
        st.markdown(alert_html("No active position — waiting for signal.", "info"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Elliott Wave display ──────────────────────────────────
    wave_res = ts_get("wave_data", {})
    if wave_res:
        st.markdown("#### 🌊 Elliott Wave Analysis")
        col_w, col_i = st.columns([2, 1])
        with col_w:
            st.markdown(wave_cards_html(wave_res), unsafe_allow_html=True)
        with col_i:
            st.markdown(f"""
            <div class="config-box">
              <div class="config-row"><span class="config-key">Status</span><span class="config-val">{wave_res.get('status','—')}</span></div>
              <div class="config-row"><span class="config-key">Current Wave</span><span class="config-val">{wave_res.get('current_wave','—')}</span></div>
              <div class="config-row"><span class="config-key">Direction</span><span class="config-val">{wave_res.get('direction','—')}</span></div>
              <div class="config-row"><span class="config-key">Next Target</span><span class="config-val">{f"{wave_res['next_target']:.2f}" if wave_res.get('next_target') else '—'}</span></div>
              <div class="config-row"><span class="config-key">Wave Signal</span><span class="config-val">{wave_res.get('signal','—')}</span></div>
              <div class="config-row"><span class="config-key">Wave SL</span><span class="config-val">{f"{wave_res['sl']:.2f}" if wave_res.get('sl') else '—'}</span></div>
              <div class="config-row"><span class="config-key">TP1</span><span class="config-val">{f"{wave_res['tp1']:.2f}" if wave_res.get('tp1') else '—'}</span></div>
              <div class="config-row"><span class="config-key">TP2</span><span class="config-val">{f"{wave_res['tp2']:.2f}" if wave_res.get('tp2') else '—'}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Wave chart
        if is_running or ltp:
            try:
                df_live = fetch_ohlcv(cfg["symbol"], cfg["interval"], cfg["period"], warmup=True)
                if df_live is not None and len(df_live) > 10:
                    df_live = add_indicators(df_live, cfg)
                    wfig = build_wave_chart(df_live, wave_res)
                    st.plotly_chart(wfig, use_container_width=True, config={"displayModeBar": False})
            except:
                pass

    st.markdown("---")

    # ── Live chart ────────────────────────────────────────────
    if is_running or ltp:
        try:
            df_live = fetch_ohlcv(cfg["symbol"], cfg["interval"], cfg["period"], warmup=True)
            if df_live is not None and len(df_live) > 5:
                df_live = add_indicators(df_live, cfg)
                lfig    = build_live_chart(df_live, position, cfg)
                st.plotly_chart(lfig, use_container_width=True, config={"displayModeBar": False})
        except:
            pass

    # ── Live log ──────────────────────────────────────────────
    st.markdown("#### 📝 Live Log")
    log_entries = ts_get("live_log", [])
    st.markdown(log_html(log_entries), unsafe_allow_html=True)

    err = ts_get("live_error")
    if err:
        st.markdown(alert_html(f"⚠️ Last error: {err}", "error"), unsafe_allow_html=True)


# ================================================================
# 19. TAB 3 — TRADE HISTORY
# ================================================================
def render_history_tab(cfg: Dict):
    ltp = get_ltp(cfg["symbol"])
    st.markdown(ltp_bar_html(cfg["ticker_name"], ltp), unsafe_allow_html=True)

    st.markdown("### 📚 Trade History")
    st.markdown(alert_html("Trade history updates in real-time — even while live trading is running.", "info"),
                unsafe_allow_html=True)

    live_trades = ts_get("live_trades", [])
    bt_trades   = []
    if st.session_state.get("bt_result"):
        bt_trades = st.session_state["bt_result"].get("trades", [])

    tab_lv, tab_bt = st.tabs(["Live Trades", "Backtest Trades"])

    with tab_lv:
        if live_trades:
            # Summary
            total  = len(live_trades)
            wins   = sum(1 for t in live_trades if t["pnl"] > 0)
            tot_pnl = sum(t["pnl"] for t in live_trades)
            acc    = round(wins / total * 100, 1) if total > 0 else 0
            st.markdown(summary_grid_html({
                "Total Trades": total,
                "Winners":      wins,
                "Losers":       total - wins,
                "Accuracy":     f"{acc}%",
                "Total P&L":    round(tot_pnl, 2),
            }), unsafe_allow_html=True)
            st.markdown(trades_table_html(live_trades, show_violations=False),
                        unsafe_allow_html=True)
        else:
            st.markdown(alert_html("No live trades yet.", "info"), unsafe_allow_html=True)

    with tab_bt:
        if bt_trades:
            st.markdown(trades_table_html(bt_trades, show_violations=True),
                        unsafe_allow_html=True)
        else:
            st.markdown(alert_html("Run a backtest first.", "info"), unsafe_allow_html=True)


# ================================================================
# 20. TAB 4 — OPTIMIZATION
# ================================================================
def render_optimization_tab(cfg: Dict):
    ltp = get_ltp(cfg["symbol"])
    st.markdown(ltp_bar_html(cfg["ticker_name"], ltp), unsafe_allow_html=True)

    st.markdown("### ⚙️ Strategy Optimization")
    st.markdown(alert_html(
        "Grid search over EMA periods, SL points, and Target points. "
        "Results shown even if target accuracy not achieved.", "info"
    ), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        target_acc = st.number_input("Target Accuracy (%)", value=50.0,
                                      min_value=0.0, max_value=100.0, key="opt_acc")
    with c2:
        max_results = st.number_input("Max Results to Show", value=20,
                                       min_value=5, max_value=200, key="opt_max")
    with c3:
        run_opt = st.button("🔍 Run Optimization", key="opt_run", type="primary",
                             use_container_width=True)

    if run_opt:
        with st.spinner("Running optimization grid search... (this may take a minute)"):
            df_full, df_display = fetch_ohlcv_with_slice(
                cfg["symbol"], cfg["interval"], cfg["period"]
            )
            if df_full is None:
                st.error("Data fetch failed.")
                return

            results = run_optimization(
                df_full, df_display or df_full, cfg, float(target_acc)
            )
            st.session_state.opt_results = results

    results = st.session_state.get("opt_results", [])
    if results:
        meets = [r for r in results if r["meets_target"]]
        fails = [r for r in results if not r["meets_target"]]

        st.markdown(f"**{len(results)} combinations tested | {len(meets)} meet {target_acc}% accuracy target**")

        disp = results[:max_results]

        rows = ""
        for i, r in enumerate(disp, 1):
            cls    = "" if r["meets_target"] else "style='opacity:0.6'"
            acc_c  = "#3fb950" if r["meets_target"] else "#e3b341"
            pnl_c  = "#3fb950" if r["total_pnl"] >= 0 else "#f85149"
            rows += f"""
            <tr {cls}>
              <td>{i}</td>
              <td>{r['ema_fast']}</td>
              <td>{r['ema_slow']}</td>
              <td>{r['sl_points']}</td>
              <td>{r['tgt_points']}</td>
              <td>{r['total_trades']}</td>
              <td style="color:{acc_c};font-weight:700">{r['accuracy']}%</td>
              <td style="color:{pnl_c};font-weight:700">{r['total_pnl']:.2f}</td>
              <td>{r['profit_factor']:.2f}</td>
              <td style="color:#3fb950">{r['max_win']:.2f}</td>
              <td style="color:#f85149">{r['max_loss']:.2f}</td>
            </tr>"""

        html = f"""
        <div class="trade-wrap">
          <table class="trade-table">
            <thead>
              <tr>
                <th>#</th><th>Fast EMA</th><th>Slow EMA</th>
                <th>SL Pts</th><th>Tgt Pts</th>
                <th>Trades</th><th>Accuracy</th><th>Total PnL</th>
                <th>Profit Factor</th><th>Max Win</th><th>Max Loss</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

        # Chart: accuracy distribution
        if len(results) > 3:
            accs = [r["accuracy"] for r in results]
            pnls = [r["total_pnl"] for r in results]
            fig  = go.Figure()
            fig.add_trace(go.Scatter(
                x=accs, y=pnls,
                mode="markers",
                marker=dict(
                    color=accs,
                    colorscale="RdYlGn",
                    size=8,
                    showscale=True,
                    colorbar=dict(title="Accuracy %"),
                ),
                text=[f"EMA {r['ema_fast']}/{r['ema_slow']} | SL:{r['sl_points']} Tgt:{r['tgt_points']}"
                      for r in results],
                hovertemplate="%{text}<br>Acc: %{x:.1f}%<br>PnL: %{y:.2f}<extra></extra>",
            ))
            fig.add_vline(x=target_acc, line=dict(color="#f0a500", width=1, dash="dash"),
                          annotation_text=f"Target: {target_acc}%")
            fig.update_layout(
                title="Optimization: Accuracy vs Total PnL",
                xaxis_title="Accuracy (%)",
                yaxis_title="Total PnL",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font=dict(color="#8b949e"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                margin=dict(l=0, r=0, t=40, b=0),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    elif st.session_state.get("opt_results") is not None:
        st.markdown(alert_html("No valid results found. Try different parameters or a longer period.", "warn"),
                    unsafe_allow_html=True)


# ================================================================
# 21. MAIN APP
# ================================================================
def main():
    cfg = render_sidebar()

    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
      <span style="font-size:28px">📈</span>
      <div>
        <h2 style="margin:0;padding:0;color:#e6edf3">Smart Investing</h2>
        <div style="color:#8b949e;font-size:12px">
          Algorithmic Trading Platform · NSE · BSE · Crypto · Elliott Wave · Dhan
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_bt, tab_lv, tab_hi, tab_op = st.tabs([
        "📊 Backtesting",
        "⚡ Live Trading",
        "📚 Trade History",
        "⚙️ Optimization",
    ])

    with tab_bt:
        render_backtest_tab(cfg)

    with tab_lv:
        render_live_tab(cfg)

    with tab_hi:
        render_history_tab(cfg)

    with tab_op:
        render_optimization_tab(cfg)


if __name__ == "__main__":
    main()
