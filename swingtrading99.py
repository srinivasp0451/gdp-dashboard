"""
AlgoTrader Pro — Single-file Streamlit algorithmic trading workbench.

Educational / research tool. Not investment advice. Past backtest performance
never guarantees future returns. Live "trading" in this app is a paper/
simulation layer unless you explicitly enable the Dhan broker checkbox and
wire in verified credentials — do that only after testing in a sandbox.
"""

import io
import json
import smtplib
import ssl
import time
from datetime import datetime, timedelta, date, time as dtime

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover - very old Python fallback
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from email.mime.text import MIMEText

st.set_page_config(page_title="AlgoTrader Pro", layout="wide", page_icon="📈")

# ============================================================================
# CONSTANTS
# ============================================================================

TICKER_MAP = {
    "Nifty50": "^NSEI",
    "BankNifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "USDINR": "USDINR=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Custom": None,
    "Options Trading": None,   # special mode: trade CE/PE option legs on an index/stock via Dhan
}

TF_PERIOD_MAP = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "7d", "1mo", "3mo", "6mo", "1y"],
    "1d": ["7d", "1mo", "6mo", "1y", "2y", "3y", "5y", "10y"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "30y"],
}

STRATEGIES = [
    "EMA Crossover",
    "Simple Buy Only",
    "Simple Sell Only",
    "Threshold Cross",
    "Price Action Support/Resistance",
    "Liquidity Grab Reversal",
    "RSI Cross",
    "Bollinger Bands",
    "Volume Breakout",
    "Elliott Wave (Zigzag)",
    "Pro: VWAP + Supertrend Trend",
    "Pro: Opening Range Breakout + Volume",
    "Pro: BB+RSI Mean Reversion (ATR filtered)",
    "Pro: EMA50 Trend + EMA9/15 Pullback",
    "Pro: MACD Crossover",
    "Pro: Donchian Channel Breakout",
    "Pro: Keltner Squeeze Breakout",
    "Pro: Stochastic Reversal",
    "Pro: TEMA Trend Flip",
    "Pro: CCI Extreme Reversal",
    "Pro: Parabolic SAR Flip",
    "Pro: ADX/DI Directional Entry",
    "Pro: Heikin-Ashi Trend Continuation",
    "Pro: Ichimoku Cloud Breakout",
]

PRO_STRATEGIES = {
    "Pro: VWAP + Supertrend Trend",
    "Pro: Opening Range Breakout + Volume",
    "Pro: BB+RSI Mean Reversion (ATR filtered)",
    "Pro: EMA50 Trend + EMA9/15 Pullback",
    "Pro: MACD Crossover",
    "Pro: Donchian Channel Breakout",
    "Pro: Keltner Squeeze Breakout",
    "Pro: Stochastic Reversal",
    "Pro: TEMA Trend Flip",
    "Pro: CCI Extreme Reversal",
    "Pro: Parabolic SAR Flip",
    "Pro: ADX/DI Directional Entry",
    "Pro: Heikin-Ashi Trend Continuation",
    "Pro: Ichimoku Cloud Breakout",
}

# Rough family classification used by the Regime Filter — trend-following
# strategies want ADX confirming a trend, mean-reversion strategies want the
# opposite (a non-trending / ranging tape). "neutral" strategies aren't gated.
STRATEGY_FAMILY = {
    "EMA Crossover": "trend",
    "Simple Buy Only": "neutral",
    "Simple Sell Only": "neutral",
    "Threshold Cross": "neutral",
    "Price Action Support/Resistance": "trend",
    "Liquidity Grab Reversal": "mean_reversion",
    "RSI Cross": "mean_reversion",
    "Bollinger Bands": "mean_reversion",
    "Volume Breakout": "trend",
    "Elliott Wave (Zigzag)": "trend",
    "Pro: VWAP + Supertrend Trend": "trend",
    "Pro: Opening Range Breakout + Volume": "trend",
    "Pro: BB+RSI Mean Reversion (ATR filtered)": "mean_reversion",
    "Pro: EMA50 Trend + EMA9/15 Pullback": "trend",
    "Pro: MACD Crossover": "trend",
    "Pro: Donchian Channel Breakout": "trend",
    "Pro: Keltner Squeeze Breakout": "trend",
    "Pro: Stochastic Reversal": "mean_reversion",
    "Pro: TEMA Trend Flip": "trend",
    "Pro: CCI Extreme Reversal": "mean_reversion",
    "Pro: Parabolic SAR Flip": "trend",
    "Pro: ADX/DI Directional Entry": "trend",
    "Pro: Heikin-Ashi Trend Continuation": "trend",
    "Pro: Ichimoku Cloud Breakout": "trend",
}

# These strategies react to a condition that's true or false AT A SINGLE
# PRICE POINT (previous close vs current price, or a price crossing a fixed
# threshold) — there's no "candle shape" to wait for, unlike an EMA/RSI/BB
# cross which genuinely needs a closed bar to compute reliably. So these fire
# immediately at the current price instead of waiting for next-candle-open.
IMMEDIATE_EXECUTION_STRATEGIES = {"Simple Buy Only", "Simple Sell Only", "Threshold Cross"}

SL_TYPES = [
    "Custom Points", "Trailing SL (Points)", "Trail Candle Low/High (Current)",
    "Trail Candle Low/High (Previous)", "Trail Swing Low/High (Current)",
    "Trail Swing Low/High (Previous)", "Strategy Signal Exit", "EMA Reverse Crossover Exit",
    "ATR Based SL", "Risk:Reward Based (min 1:2)", "Autopilot SL",
    "Loss Recovery SL (Give-back)",
]

TARGET_TYPES = [
    "Custom Points", "Trailing Target (Display Only)", "Trail Candle Low/High (Current)",
    "Trail Candle Low/High (Previous)", "Trail Swing Low/High (Current)",
    "Trail Swing Low/High (Previous)", "Strategy Signal Exit", "EMA Reverse Crossover Exit",
    "ATR Based Target", "Risk:Reward Based (min 1:2)", "Autopilot Target",
    "Profit Giveback Target", "Partial Book + Trail Remainder",
]

RATE_LIMIT_DELAY = 0.3  # seconds, mandatory pause between yfinance calls (Dhan path applies NO delay)

# ---------------------------------------------------------------------------
# DHAN CONSTANTS
# ---------------------------------------------------------------------------
DHAN_API_BASE = "https://api.dhan.co/v2"
DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_DEFAULT_CLIENT_ID = "1104779876"
EMAIL_DEFAULT_FROM = "srinivas.trml@gmail.com"

# Index underlyings Dhan can serve directly (index spot for data, FNO for orders)
DHAN_INDEX_MAP = {
    "Nifty50":   {"underlying": "NIFTY",     "security_id": "13", "segment": "IDX_I", "exchange": "NSE", "default_opt_qty": 65},
    "BankNifty": {"underlying": "BANKNIFTY", "security_id": "25", "segment": "IDX_I", "exchange": "NSE", "default_opt_qty": 35},
    "Sensex":    {"underlying": "SENSEX",    "security_id": "51", "segment": "IDX_I", "exchange": "BSE", "default_opt_qty": 20},
}

# Instrument dropdown → (F&O/EQ classification, product) mapping.
# Exchange (NSE/BSE) is a separate user-editable dropdown; segment resolves as:
#   equity  → NSE_EQ / BSE_EQ         futures & options → NSE_FNO / BSE_FNO
DHAN_INSTRUMENT_CHOICES = [
    "Stock Intraday", "Stock Delivery", "Stock Futures",
    "Index Futures", "Stock Options", "Index Options",
]
DHAN_INSTRUMENT_META = {
    "Stock Intraday": {"kind": "EQ",  "product": "INTRADAY", "scrip_instrument": "EQUITY"},
    "Stock Delivery": {"kind": "EQ",  "product": "CNC",      "scrip_instrument": "EQUITY"},
    "Stock Futures":  {"kind": "FNO", "product": "MARGIN",   "scrip_instrument": "FUTSTK"},
    "Index Futures":  {"kind": "FNO", "product": "MARGIN",   "scrip_instrument": "FUTIDX"},
    "Stock Options":  {"kind": "FNO", "product": "MARGIN",   "scrip_instrument": "OPTSTK"},
    "Index Options":  {"kind": "FNO", "product": "MARGIN",   "scrip_instrument": "OPTIDX"},
}

# yfinance-only tickers Dhan cannot serve — the data feed silently falls back
# to yfinance for these (with a notice on the Live tab).
DHAN_UNSUPPORTED_YF = {"BTC-USD", "ETH-USD", "USDINR=X", "GC=F", "SI=F"}

# Dhan intraday chart API accepted interval codes
DHAN_INTERVAL_CODE = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}

# Rough period-string → number of calendar days to request from Dhan
PERIOD_TO_DAYS = {
    "1d": 1, "5d": 5, "7d": 7, "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 366, "2y": 731, "3y": 1096, "5y": 1827, "10y": 3653,
    "20y": 7305, "30y": 10958,
}


def ist_now():
    """Current wall-clock time in IST (Dhan candles / trade windows / daily
    risk counters are all defined in IST)."""
    return datetime.now(IST)


def is_indian_ticker(ticker_choice, ticker):
    """Trade-window enforcement applies ONLY to Indian instruments
    (.NS/.BO/Nifty/BankNifty/Sensex); everything else trades 24h."""
    t = (ticker or "")
    return (
        ticker_choice in ("Nifty50", "BankNifty", "Sensex", "Options Trading")
        or t.endswith(".NS") or t.endswith(".BO")
        or t in ("^NSEI", "^NSEBANK", "^BSESN")
    )

# ============================================================================
# SESSION STATE
# ============================================================================

for key, default in {
    "live_positions": [],
    "live_history": [],
    "opt_results": {},
    "last_backtest": None,
    "last_backtest_df": None,
    "live_running": False,
    "last_acted_signal_marker": None,
    # --- shared config store (single source of truth for Sidebar + Admin Panel) ---
    "app_cfg": {},
    # --- Dhan data-feed / autofill bookkeeping ---
    "dhan_fallback_notice": None,
    "dhan_feed_warning": None,
    "dhan_autofill_sig": None,
    "dhan_autofill_last_try": 0.0,
    "dhan_opt_autofill_sig": None,
    "dhan_opt_autofill_last_try": 0.0,
    # --- daily risk-gate counters (reset on IST date change) ---
    "risk_day_key": None,
    "risk_day_entries": 0,
    "risk_last_event_ts": 0.0,
    "live_blocked_reason": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# One-off IST daily counter reset
_today_key = ist_now().strftime("%Y-%m-%d")
if st.session_state.risk_day_key != _today_key:
    st.session_state.risk_day_key = _today_key
    st.session_state.risk_day_entries = 0


# ============================================================================
# SHARED CONFIG STORE + TWO-WAY SYNCED WIDGET WRAPPERS
# ----------------------------------------------------------------------------
# The Sidebar and the "🛠 Admin Panel" tab are two live views of ONE store:
# st.session_state.app_cfg. Every control is rendered through a wrapper using
# a deterministic, callback-free sync (see _cfg_sync below): per run, the
# WIDGET wins when the user changed it since the last render (robust even
# when Streamlit misses a change event because a rerun was superseded — e.g.
# toggling a checkbox and clicking "Run Backtest" immediately), otherwise the
# STORE seeds the widget (cross-view edits, the Optimization tab's
# apply-config, autofills). Coerced values (option list changed, clamping,
# casting) are written straight back to the store so the store can never
# disagree with what the widget shows.
# ============================================================================

def _cfg_store():
    return st.session_state.app_cfg


# --- Deterministic two-way sync (no on_change callbacks) -------------------
# Streamlit callbacks can be MISSED when a rerun is superseded quickly (e.g.
# toggling a checkbox and clicking "Run Backtest" almost immediately). The
# old design then re-seeded the widget from the stale store — visibly
# un-ticking the checkbox and running the backtest with the OLD config.
# This version never relies on callbacks. Per widget, per run:
#   1) WIDGET WINS: if the widget's value differs from the snapshot of what
#      we rendered last time, the USER changed it (even if Streamlit missed
#      the change event) → write it into the store.
#   2) STORE SEEDS: otherwise, if the store differs from the widget, the
#      store was changed programmatically (the other view, the Optimization
#      tab's apply-config, an autofill) → push store → widget.
#   3) Render, then snapshot what was rendered and commit it to the store.
# Result: a fresh user change can never be clobbered, and cross-view edits
# still propagate instantly.

def _cfg_sync(wkey, cfg_key, default, coerce=None):
    store = _cfg_store()
    seen_key = "_seen_" + wkey
    if cfg_key not in store:
        store[cfg_key] = default
    # 1) widget wins over a stale store (covers missed change events)
    if wkey in st.session_state and seen_key in st.session_state \
            and st.session_state[wkey] != st.session_state[seen_key]:
        store[cfg_key] = st.session_state[wkey]
    cur = store.get(cfg_key, default)
    if coerce is not None:
        cur = coerce(cur)
    if store.get(cfg_key) != cur:
        store[cfg_key] = cur          # coercion write-back
    # 2) store seeds the widget (cross-view / programmatic changes)
    if st.session_state.get(wkey) != cur:
        st.session_state[wkey] = cur
    return cur


def _cfg_commit(wkey, cfg_key, val):
    st.session_state["_seen_" + wkey] = val
    _cfg_store()[cfg_key] = val
    return val


def cfg_checkbox(ui, label, cfg_key, default=False, prefix="sb", **kw):
    wkey = f"w_{prefix}_{cfg_key}"
    _cfg_sync(wkey, cfg_key, bool(default), coerce=lambda v: bool(v))
    val = ui.checkbox(label, key=wkey, **kw)
    return _cfg_commit(wkey, cfg_key, bool(val))


def cfg_selectbox(ui, label, cfg_key, options, default=None, prefix="sb", **kw):
    options = list(options)
    if not options:
        return None
    if default is None or default not in options:
        default = options[0]

    def _coerce(v):
        return v if v in options else default   # option list changed → coerce

    wkey = f"w_{prefix}_{cfg_key}"
    _cfg_sync(wkey, cfg_key, default, coerce=_coerce)
    val = ui.selectbox(label, options, key=wkey, **kw)
    return _cfg_commit(wkey, cfg_key, val)


def cfg_multiselect(ui, label, cfg_key, options, default=None, prefix="sb", **kw):
    options = list(options)

    def _coerce(v):
        return [c for c in (v or []) if c in options]

    wkey = f"w_{prefix}_{cfg_key}"
    _cfg_sync(wkey, cfg_key, list(default or []), coerce=_coerce)
    val = ui.multiselect(label, options, key=wkey, **kw)
    return _cfg_commit(wkey, cfg_key, list(val))


def cfg_number(ui, label, cfg_key, default, min_value=None, max_value=None,
               step=None, is_int=False, prefix="sb", **kw):
    def _coerce(v):
        try:
            v = int(v) if is_int else float(v)
        except (TypeError, ValueError):
            v = int(default) if is_int else float(default)
        if min_value is not None:
            v = max(v, min_value)     # clamping
        if max_value is not None:
            v = min(v, max_value)
        return v

    wkey = f"w_{prefix}_{cfg_key}"
    _cfg_sync(wkey, cfg_key, _coerce(default), coerce=_coerce)
    val = ui.number_input(label, min_value=min_value, max_value=max_value, step=step, key=wkey, **kw)
    return _cfg_commit(wkey, cfg_key, val)


def cfg_text(ui, label, cfg_key, default="", prefix="sb", **kw):
    wkey = f"w_{prefix}_{cfg_key}"
    _cfg_sync(wkey, cfg_key, str(default), coerce=lambda v: "" if v is None else str(v))
    val = ui.text_input(label, key=wkey, **kw)
    return _cfg_commit(wkey, cfg_key, val)


def cfg_slider(ui, label, cfg_key, min_value, max_value, default, step=None, prefix="sb", **kw):
    def _coerce(v):
        try:
            v = type(default)(v)
        except (TypeError, ValueError):
            v = default
        return max(min(v, max_value), min_value)

    wkey = f"w_{prefix}_{cfg_key}"
    _cfg_sync(wkey, cfg_key, default, coerce=_coerce)
    val = ui.slider(label, min_value, max_value, key=wkey, step=step, **kw)
    return _cfg_commit(wkey, cfg_key, val)


def cfg_time(ui, label, cfg_key, default, prefix="sb", **kw):
    def _coerce(v):
        if isinstance(v, dtime):
            return v
        try:
            hh, mm = str(v).split(":")[:2]
            return dtime(int(hh), int(mm))
        except Exception:
            return default

    wkey = f"w_{prefix}_{cfg_key}"
    _cfg_sync(wkey, cfg_key, default, coerce=_coerce)
    val = ui.time_input(label, key=wkey, **kw)
    return _cfg_commit(wkey, cfg_key, val)


def cfg_set(cfg_key, value):
    """Programmatic write into the shared store (used by the Optimization
    tab's 'apply config' — replaces the old sidebar_overrides mechanism).
    The store-seeds step in _cfg_sync propagates it into BOTH views' widgets
    on the next run."""
    st.session_state.app_cfg[cfg_key] = value


# ============================================================================
# INDICATORS
# ----------------------------------------------------------------------------
# TradingView-convention verification (documentation only — no formula
# changes were needed; every formula below already matches TV):
#   • RSI / ATR / ADX / ±DI ... Wilder's RMA smoothing, implemented here as
#     ewm(alpha=1/period, adjust=False) which is mathematically identical
#     to TradingView's ta.rma().
#   • EMA / MACD / TEMA ....... standard EMA: ewm(span=period, adjust=False),
#     identical to ta.ema() (MACD = EMA(fast)−EMA(slow), signal = EMA of MACD).
#   • Bollinger Bands ......... population stdev, std(ddof=0), matching
#     ta.stdev()'s biased=true default (pandas' own ddof=1 default would give
#     slightly wider bands — deliberately NOT used).
#   • CCI ..................... mean absolute deviation of typical price about
#     its SMA, matching ta.cci().
#   • Stochastic .............. raw %K with smoothing 1 (%K = 100·(C−LL)/(HH−LL)),
#     %D = SMA(%K, d_period) — matches TV's default "fast" stochastic.
#   • Supertrend .............. RMA-smoothed ATR bands with band carry-forward
#     (upper band can only ratchet down in downtrends / lower band up in
#     uptrends) — same band logic as TradingView's supertrend().
# ============================================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(series, period=20, std_mult=2):
    mid = series.rolling(period).mean()
    # ddof=0 (population stdev) matches TradingView's ta.stdev default (biased=true).
    # pandas' own default is ddof=1 (sample stdev) which gives slightly WIDER bands
    # than TradingView at the same settings — this is the #1 cause of BB mismatches.
    std = series.rolling(period).std(ddof=0)
    return mid + std_mult * std, mid, mid - std_mult * std


def adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)


def supertrend(df, period=10, mult=3):
    atr_ = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = (hl2 + mult * atr_).copy()
    lowerband = (hl2 - mult * atr_).copy()
    line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            line.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
            continue
        if df["Close"].iloc[i] > upperband.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lowerband.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]
        line.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]
    return line, direction


def swing_points(df, lookback=3):
    highs, lows = df["High"], df["Low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    n = len(df)
    for i in range(lookback, n - lookback):
        wh = highs.iloc[i - lookback: i + lookback + 1]
        wl = lows.iloc[i - lookback: i + lookback + 1]
        if highs.iloc[i] == wh.max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == wl.min():
            swing_low.iloc[i] = True
    return swing_high, swing_low


def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def donchian(df, period=20):
    upper = df["High"].rolling(period).max()
    lower = df["Low"].rolling(period).min()
    return upper, (upper + lower) / 2, lower


def keltner(df, period=20, atr_mult=1.5):
    mid = ema(df["Close"], period)
    a = atr(df, period)
    return mid + atr_mult * a, mid, mid - atr_mult * a


def stochastic(df, k_period=14, d_period=3):
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    return k, k.rolling(d_period).mean()


def tema(series, period=20):
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def cci(df, period=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def adx_di(df, period=14):
    """Like adx() but also returns +DI/-DI separately, needed for directional
    (not just strength) entries."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return plus_di, minus_di, dx.ewm(alpha=1 / period, adjust=False).mean()


def parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    high, low = df["High"].values, df["Low"].values
    n = len(df)
    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)
    ep = np.zeros(n)
    af = np.zeros(n)
    trend[0], sar[0], ep[0], af[0] = 1, low[0], high[0], af_start
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            s = prev_sar + af[i - 1] * (ep[i - 1] - prev_sar)
            s = min(s, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < s:
                trend[i], sar[i], ep[i], af[i] = -1, ep[i - 1], low[i], af_start
            else:
                trend[i], sar[i] = 1, s
                if high[i] > ep[i - 1]:
                    ep[i], af[i] = high[i], min(af[i - 1] + af_step, af_max)
                else:
                    ep[i], af[i] = ep[i - 1], af[i - 1]
        else:
            s = prev_sar + af[i - 1] * (ep[i - 1] - prev_sar)
            s = max(s, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > s:
                trend[i], sar[i], ep[i], af[i] = 1, ep[i - 1], high[i], af_start
            else:
                trend[i], sar[i] = -1, s
                if low[i] < ep[i - 1]:
                    ep[i], af[i] = low[i], min(af[i - 1] + af_step, af_max)
                else:
                    ep[i], af[i] = ep[i - 1], af[i - 1]
    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)


def heikin_ashi(df):
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


def ichimoku(df, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    tenkan = (df["High"].rolling(tenkan_p).max() + df["Low"].rolling(tenkan_p).min()) / 2
    kijun = (df["High"].rolling(kijun_p).max() + df["Low"].rolling(kijun_p).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
    senkou_b = ((df["High"].rolling(senkou_b_p).max() + df["Low"].rolling(senkou_b_p).min()) / 2).shift(kijun_p)
    return tenkan, kijun, senkou_a, senkou_b


# Gap handling note: True Range (used by atr()/adx()/supertrend()) is defined as
# max(high-low, |high-prev_close|, |low-prev_close|) — the prev_close terms are
# exactly what captures a gap-up/gap-down correctly, so ATR/ADX/Supertrend here
# already reflect gaps properly without special-casing. What DOES need explicit
# handling is simply not having enough bars yet (e.g. right after a fresh
# fetch, or a low-period intraday pull) — that's what MIN_BARS_REQUIRED and
# safe_indicator_value() below are for: show "N/A — insufficient data" instead
# of silently returning/using a NaN or a misleading half-warmed-up value.
MIN_BARS_REQUIRED = {
    "ema9": 9 * 3, "ema15": 15 * 3, "ema20": 20 * 3, "ema50": 50 * 3,
    "rsi": 14 * 3, "atr": 14 * 3, "adx": 14 * 4, "bollinger": 20 * 2, "supertrend": 10 * 4,
}


def safe_indicator_value(series, min_bars, label=""):
    """Returns (value, is_reliable). If there isn't enough history for the
    indicator to have warmed up, or the latest value is NaN, returns
    (None, False) so callers can render 'N/A — insufficient data' instead of
    a silently wrong number."""
    if series is None or len(series) < min_bars:
        return None, False
    val = series.iloc[-1]
    if pd.isna(val):
        return None, False
    return float(val), True


# ============================================================================
# DATA FETCH
# ----------------------------------------------------------------------------
# Two sources behind one router:
#   • yfinance (default) — keeps its mandatory 0.3s delay per API call.
#   • Dhan data feed (optional, checkbox) — NO delay at all (Dhan provides
#     zero-delay data). Serves candles (historical + intraday, IST timezone)
#     and live LTP. Tickers Dhan cannot serve (BTC-USD, ETH-USD, USDINR,
#     gold/silver futures, …) automatically fall back to yfinance with a
#     notice on the Live tab. Feed ON without an access token silently stays
#     on yfinance and shows a warning.
# ============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_data_yf(ticker, interval, period):
    """Original yfinance candle fetch — logic unchanged, mandatory delay kept."""
    time.sleep(RATE_LIMIT_DELAY)
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    return df


# ---------------------------------------------------------------------------
# DHAN SCRIP MASTER (downloaded once, cached 24h)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def load_dhan_scrip_master():
    """Dhan's public scrip master CSV. Cached for 24h. Returns an empty
    DataFrame on failure (callers treat that as 'lookup unavailable')."""
    try:
        resp = requests.get(DHAN_SCRIP_MASTER_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _scrip_cols(sm):
    """Best-effort column resolution across scrip-master schema variants."""
    def pick(*cands):
        for c in cands:
            if c in sm.columns:
                return c
        return None
    return {
        "exch": pick("SEM_EXM_EXCH_ID", "EXCH_ID"),
        "segment": pick("SEM_SEGMENT", "SEGMENT"),
        "secid": pick("SEM_SMST_SECURITY_ID", "SECURITY_ID"),
        "symbol": pick("SEM_TRADING_SYMBOL", "TRADING_SYMBOL"),
        "name": pick("SM_SYMBOL_NAME", "SYMBOL_NAME"),
        "instrument": pick("SEM_INSTRUMENT_NAME", "INSTRUMENT"),
        "expiry": pick("SEM_EXPIRY_DATE", "EXPIRY_DATE"),
        "strike": pick("SEM_STRIKE_PRICE", "STRIKE_PRICE"),
        "opt_type": pick("SEM_OPTION_TYPE", "OPTION_TYPE"),
        "lot": pick("SEM_LOT_UNITS", "LOT_UNITS", "SEM_LOT_SIZE"),
        "custom": pick("SEM_CUSTOM_SYMBOL", "CUSTOM_SYMBOL"),
    }


def _yf_symbol_to_plain(ticker):
    """RELIANCE.NS → RELIANCE, TCS.BO → TCS (scrip-master trading symbol)."""
    t = (ticker or "").upper()
    for suf in (".NS", ".BO"):
        if t.endswith(suf):
            return t[: -len(suf)]
    return t


def dhan_lookup_equity(symbol_plain, exchange="NSE"):
    """Equity security ID + lot size from the scrip master."""
    sm = load_dhan_scrip_master()
    if sm.empty:
        return None
    c = _scrip_cols(sm)
    if not (c["exch"] and c["secid"] and c["symbol"]):
        return None
    try:
        sub = sm[(sm[c["exch"]].astype(str).str.strip() == exchange)
                 & (sm[c["symbol"]].astype(str).str.strip().str.upper() == symbol_plain.upper())]
        if c["instrument"]:
            eq = sub[sub[c["instrument"]].astype(str).str.contains("EQUITY", case=False, na=False)]
            if not eq.empty:
                sub = eq
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {"security_id": str(int(float(row[c["secid"]]))),
                "lot_size": int(float(row[c["lot"]])) if c["lot"] and not pd.isna(row[c["lot"]]) else 1}
    except Exception:
        return None


def _fno_underlying_frame(instrument_code, underlying, exchange="NSE"):
    sm = load_dhan_scrip_master()
    if sm.empty:
        return pd.DataFrame(), {}
    c = _scrip_cols(sm)
    if not (c["instrument"] and c["secid"]):
        return pd.DataFrame(), c
    try:
        sub = sm[sm[c["instrument"]].astype(str).str.strip() == instrument_code]
        if c["exch"]:
            sub = sub[sub[c["exch"]].astype(str).str.strip() == exchange]
        u = underlying.upper()
        sym = sub[c["symbol"]].astype(str).str.upper() if c["symbol"] else ""
        name = sub[c["name"]].astype(str).str.upper() if c["name"] else ""
        mask = pd.Series(False, index=sub.index)
        if c["symbol"]:
            mask |= sym.str.startswith(u + "-") | sym.str.startswith(u + " ") | (sym == u) | sym.str.startswith(u)
        if c["name"]:
            mask |= (name == u)
        return sub[mask], c
    except Exception:
        return pd.DataFrame(), c


def dhan_get_expiries(underlying, instrument_code, exchange="NSE"):
    """Sorted (nearest-first) list of expiry date strings for an underlying's
    futures or options from the scrip master."""
    sub, c = _fno_underlying_frame(instrument_code, underlying, exchange)
    if sub.empty or not c.get("expiry"):
        return []
    try:
        exp = pd.to_datetime(sub[c["expiry"]], errors="coerce").dt.date.dropna().unique()
        today = ist_now().date()
        exp = sorted(d for d in exp if d >= today)
        return [d.strftime("%Y-%m-%d") for d in exp]
    except Exception:
        return []


def dhan_get_strikes(underlying, expiry_str, instrument_code, exchange="NSE"):
    """Sorted list of real strike prices for an underlying+expiry."""
    sub, c = _fno_underlying_frame(instrument_code, underlying, exchange)
    if sub.empty or not (c.get("expiry") and c.get("strike")):
        return []
    try:
        exp = pd.to_datetime(sub[c["expiry"]], errors="coerce").dt.date
        sub = sub[exp == pd.to_datetime(expiry_str).date()]
        strikes = pd.to_numeric(sub[c["strike"]], errors="coerce").dropna().unique()
        return sorted(float(s) for s in strikes if s > 0)
    except Exception:
        return []


def dhan_lookup_option(underlying, expiry_str, strike, opt_type, instrument_code, exchange="NSE"):
    """Security ID + lot size of one specific option contract."""
    sub, c = _fno_underlying_frame(instrument_code, underlying, exchange)
    if sub.empty or not (c.get("expiry") and c.get("strike") and c.get("opt_type")):
        return None
    try:
        exp = pd.to_datetime(sub[c["expiry"]], errors="coerce").dt.date
        sub = sub[exp == pd.to_datetime(expiry_str).date()]
        stk = pd.to_numeric(sub[c["strike"]], errors="coerce")
        sub = sub[np.isclose(stk, float(strike))]
        sub = sub[sub[c["opt_type"]].astype(str).str.strip().str.upper() == opt_type.upper()]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {"security_id": str(int(float(row[c["secid"]]))),
                "lot_size": int(float(row[c["lot"]])) if c["lot"] and not pd.isna(row[c["lot"]]) else 1}
    except Exception:
        return None


def dhan_lookup_future(underlying, expiry_str, instrument_code, exchange="NSE"):
    """Security ID + lot size of one futures contract."""
    sub, c = _fno_underlying_frame(instrument_code, underlying, exchange)
    if sub.empty or not c.get("expiry"):
        return None
    try:
        exp = pd.to_datetime(sub[c["expiry"]], errors="coerce").dt.date
        sub = sub[exp == pd.to_datetime(expiry_str).date()]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {"security_id": str(int(float(row[c["secid"]]))),
                "lot_size": int(float(row[c["lot"]])) if c["lot"] and not pd.isna(row[c["lot"]]) else 1}
    except Exception:
        return None


def round_to_nearest_strike(price, strikes):
    """ATM = live LTP rounded to the nearest REAL strike from the chain."""
    if not strikes or price is None:
        return None
    return min(strikes, key=lambda s: abs(s - price))


# ---------------------------------------------------------------------------
# DHAN FEED — instrument resolution, candles, live LTP  (NO delay applied)
# ---------------------------------------------------------------------------

def _dhan_creds():
    cfg = st.session_state.app_cfg
    return (str(cfg.get("dhan_client_id") or "").strip(),
            str(cfg.get("dhan_access_token") or "").strip())


def dhan_feed_active():
    """Data feed checkbox ON *and* a token present. ON without a token
    silently stays on yfinance (a warning is surfaced on the Live tab).
    Options Trading mode auto-activates the Dhan feed when a token exists,
    because option premiums must come from Dhan with zero delay — yfinance
    has no options data."""
    cfg = st.session_state.app_cfg
    options_mode = cfg.get("ticker_choice") == "Options Trading"
    if not (cfg.get("use_dhan_feed") or options_mode):
        return False
    _, token = _dhan_creds()
    if not token:
        st.session_state.dhan_feed_warning = (
            "Dhan Data Feed is ON but no Access Token is set — staying on yfinance until a token is provided."
        )
        return False
    st.session_state.dhan_feed_warning = None
    return True


def dhan_resolve_feed_instrument(ticker):
    """Maps a yfinance-style ticker to Dhan's (security_id, exchange_segment,
    instrument) for the DATA feed. Returns None for tickers Dhan can't serve.
    PREMIUM TRADING: a sentinel ticker "DHANOPT::<segment>::<security_id>::<instr>"
    resolves straight to that option contract, so fetch_data / get_live_ltp
    serve the option's OWN premium candles and premium LTP."""
    if not ticker or ticker in DHAN_UNSUPPORTED_YF:
        return None
    if ticker.startswith("DHANOPT::"):
        try:
            _, segment, sec_id, instr = ticker.split("::")
            if not sec_id:
                return None
            return {"security_id": sec_id, "segment": segment, "instrument": instr}
        except ValueError:
            return None
    if ticker == "^NSEI":
        return {"security_id": "13", "segment": "IDX_I", "instrument": "INDEX"}
    if ticker == "^NSEBANK":
        return {"security_id": "25", "segment": "IDX_I", "instrument": "INDEX"}
    if ticker == "^BSESN":
        return {"security_id": "51", "segment": "IDX_I", "instrument": "INDEX"}
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        exchange = "NSE" if ticker.endswith(".NS") else "BSE"
        info = dhan_lookup_equity(_yf_symbol_to_plain(ticker), exchange)
        if info:
            return {"security_id": info["security_id"],
                    "segment": f"{exchange}_EQ", "instrument": "EQUITY"}
    return None


def _dhan_headers():
    client_id, token = _dhan_creds()
    return {"access-token": token, "client-id": client_id, "Content-Type": "application/json"}


@st.cache_data(ttl=15, show_spinner=False)
def _dhan_fetch_candles_cached(security_id, segment, instrument, interval, period, _token_fingerprint):
    """Dhan candle fetch (historical + intraday). NO artificial delay.
    Timestamps are converted to IST. `_token_fingerprint` only busts the
    cache when credentials change — the token itself is never stored here."""
    try:
        headers = _dhan_headers()
        today = ist_now().date()
        days = PERIOD_TO_DAYS.get(period, 7)
        from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        if interval in DHAN_INTERVAL_CODE:
            url = f"{DHAN_API_BASE}/charts/intraday"
            payload = {
                "securityId": str(security_id), "exchangeSegment": segment,
                "instrument": instrument, "interval": DHAN_INTERVAL_CODE[interval],
                "fromDate": from_date, "toDate": to_date,
            }
        else:  # 1d / 1wk → daily history (weekly resampled below)
            url = f"{DHAN_API_BASE}/charts/historical"
            payload = {
                "securityId": str(security_id), "exchangeSegment": segment,
                "instrument": instrument, "expiryCode": 0,
                "fromDate": from_date, "toDate": to_date,
            }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data or "open" not in data or not data.get("open"):
            return pd.DataFrame()
        ts = pd.to_datetime(pd.Series(data.get("timestamp", data.get("start_Time", []))), unit="s", utc=True)
        idx = ts.dt.tz_convert("Asia/Kolkata")
        df = pd.DataFrame({
            "Open": pd.to_numeric(pd.Series(data["open"]), errors="coerce"),
            "High": pd.to_numeric(pd.Series(data["high"]), errors="coerce"),
            "Low": pd.to_numeric(pd.Series(data["low"]), errors="coerce"),
            "Close": pd.to_numeric(pd.Series(data["close"]), errors="coerce"),
            "Volume": pd.to_numeric(pd.Series(data.get("volume", [0] * len(data["open"]))), errors="coerce").fillna(0),
        })
        df.index = pd.DatetimeIndex(idx.values, tz="Asia/Kolkata")
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        if interval == "1wk" and not df.empty:
            df = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                           "Close": "last", "Volume": "sum"}).dropna()
        return df
    except Exception:
        return pd.DataFrame()


def fetch_data_dhan(ticker, interval, period):
    feed = dhan_resolve_feed_instrument(ticker)
    if feed is None:
        return None  # not servable by Dhan
    _, token = _dhan_creds()
    return _dhan_fetch_candles_cached(feed["security_id"], feed["segment"], feed["instrument"],
                                      interval, period, hash(token) % 10_000_019)


def fetch_data(ticker, interval, period):
    """ROUTER — every candle consumer in the app calls this. Chooses the Dhan
    feed when enabled+tokened+servable, otherwise the original yfinance path
    (with its mandatory 0.3s delay). Falling back for a Dhan-unservable
    ticker records a notice that the Live tab displays.
    PREMIUM TRADING sentinel tickers (DHANOPT::…) can ONLY be served by Dhan
    — yfinance has no options data — so without an active Dhan token they
    return empty with an explanatory notice instead of falling through."""
    if str(ticker).startswith("DHANOPT::"):
        if dhan_feed_active():
            dhan_df = fetch_data_dhan(ticker, interval, period)
            if dhan_df is not None and not dhan_df.empty:
                st.session_state.dhan_fallback_notice = None
                return dhan_df
        st.session_state.dhan_fallback_notice = (
            "Premium trading needs the Dhan data feed (option premiums are not available on yfinance) — "
            "enter a valid Dhan Access Token in '🔐 Dhan Account' to load the option's candles."
        )
        return pd.DataFrame()
    if dhan_feed_active():
        dhan_df = fetch_data_dhan(ticker, interval, period)
        if dhan_df is None:
            st.session_state.dhan_fallback_notice = (
                f"Dhan cannot serve '{ticker}' — automatically using yfinance (0.3s delay) for this ticker."
            )
        else:
            st.session_state.dhan_fallback_notice = None
            if not dhan_df.empty:
                return dhan_df
            # Empty Dhan response (off-hours gap, API hiccup) → fall through to yfinance
    return fetch_data_yf(ticker, interval, period)


def dhan_get_ltp(security_id, segment):
    """Zero-delay live LTP straight from Dhan's market-quote endpoint."""
    try:
        resp = requests.post(f"{DHAN_API_BASE}/marketfeed/ltp", headers=_dhan_headers(),
                             json={segment: [int(security_id)]}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        seg = data.get(segment, {})
        entry = seg.get(str(security_id)) or (next(iter(seg.values())) if seg else None)
        if entry and "last_price" in entry:
            return float(entry["last_price"])
    except Exception:
        pass
    return None


def dhan_get_ltp_for_ticker(ticker):
    feed = dhan_resolve_feed_instrument(ticker)
    if feed is None:
        return None
    return dhan_get_ltp(feed["security_id"], feed["segment"])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_vix_series(period="5y"):
    """Fetches India VIX (^INDIAVIX) daily closes. Used only to align against
    whatever timeframe the user is trading — VIX itself only publishes daily."""
    time.sleep(RATE_LIMIT_DELAY)
    try:
        d = yf.download("^INDIAVIX", interval="1d", period=period, progress=False, auto_adjust=True)
    except Exception:
        return pd.Series(dtype=float)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d["Close"].dropna()


def get_vix_aligned(target_index):
    """Maps each candle's timestamp in target_index to the most recent known
    India VIX daily close on/before that date (forward-fill by date)."""
    vix = fetch_vix_series("5y")
    if vix is None or vix.empty or len(target_index) == 0:
        return pd.Series(np.nan, index=target_index)

    vix = vix.sort_index()
    vix_idx = pd.DatetimeIndex(vix.index)
    if vix_idx.tz is not None:
        vix_idx = vix_idx.tz_localize(None)
    vix_clean = pd.Series(vix.values, index=vix_idx).sort_index()

    tgt_idx = pd.DatetimeIndex(target_index)
    tgt_naive = tgt_idx.tz_localize(None) if tgt_idx.tz is not None else tgt_idx

    left = pd.DataFrame({"t": tgt_naive})
    right = pd.DataFrame({"t": vix_clean.index, "vix": vix_clean.values}).sort_values("t")
    merged = pd.merge_asof(left, right, on="t", direction="backward")

    result = pd.Series(merged["vix"].values, index=target_index)
    return result


@st.fragment(run_every=2)
def live_position_fragment(ticker, label="LTP"):
    """
    Refreshes every ~2s on its own: live price, and — if a paper position is
    open — live points/PnL color-coded green (profit) or red (loss), plus
    Entry/SL/Target with their configured types, running highs/lows, and
    remaining quantity. This is the minimum live-trading readout; before this
    fix you'd have had to compute PnL in your head from LTP vs entry.
    """
    ltp = None
    if dhan_feed_active():
        # Zero-delay Dhan tick; previous close for the delta comes from the
        # (cached) candle feed so no extra latency is added.
        ltp = dhan_get_ltp_for_ticker(ticker)
        if ltp is not None:
            try:
                candles = fetch_data(ticker, "1m", "1d")
                prev = float(candles["Close"].iloc[-1]) if not candles.empty else ltp
            except Exception:
                prev = ltp
            st.metric(label, f"{ltp:,.2f}", f"{ltp - prev:+.2f}")
    if ltp is None:
        time.sleep(RATE_LIMIT_DELAY)
        try:
            data = yf.Ticker(ticker).history(period="1d", interval="1m")
            if data is None or data.empty:
                data = yf.Ticker(ticker).history(period="5d", interval="15m")
            if data is not None and not data.empty:
                ltp = float(data["Close"].iloc[-1])
                prev = float(data["Close"].iloc[-2]) if len(data) > 1 else ltp
                st.metric(label, f"{ltp:,.2f}", f"{ltp - prev:+.2f}")
            else:
                st.info("No live data returned yet.")
        except Exception as exc:
            st.warning(f"Fetch issue (rate limit or symbol): {exc}")

    positions = st.session_state.get("live_positions", [])
    if positions and ltp is not None:
        pos = positions[0]
        direction = pos["direction"]
        points = (ltp - pos["entry_price"]) * direction
        pnl = points * pos["remaining_qty"]

        st.markdown("###### 💰 Live Position P&L")
        c1, c2, c3 = st.columns(3)
        c1.metric("Entry Type", "LONG" if direction == 1 else "SHORT")
        c2.metric("Entry Price", f"{pos['entry_price']:.2f}")
        c3.metric("LTP", f"{ltp:,.2f}")

        c4, c5 = st.columns(2)
        c4.metric(f"SL ({pos['sl_type']})", f"{pos['sl']:.2f}")
        c5.metric(f"Target ({pos['target_type']})", f"{pos['target']:.2f}")

        # st.metric's delta is auto-colored green/red by sign — that IS the
        # green/red live PnL indicator, no manual color logic needed.
        c6, c7 = st.columns(2)
        c6.metric("Live Points", f"{points:+.2f}", f"{points:+.2f}")
        c7.metric("Live PnL", f"{pnl:+.2f}", f"{pnl:+.2f}")

        c8, c9, c10 = st.columns(3)
        c8.metric("Highest since entry", f"{pos['highest']:.2f}")
        c9.metric("Lowest since entry", f"{pos['lowest']:.2f}")
        c10.metric("Qty remaining", f"{pos['remaining_qty']}/{pos['original_qty']}")
    elif positions and ltp is None:
        st.caption("Position is open but couldn't fetch a live price this cycle — PnL will resume once the next tick comes in.")

    return ltp


@st.fragment(run_every=3)
def recent_trades_fragment():
    """
    Renders the last 10 completed live trades. Wrapped in its own fragment so
    it reflects a trade the instant SL/Target/manual-close fires — without
    this, a trade closed by live_signal_loop_fragment (a separate fragment)
    wouldn't show up here until a full page rerun happened (e.g. clicking
    Stop), even though it was already correctly recorded in session_state.
    """
    st.markdown("#### Recent Trades")
    if st.session_state.live_history:
        st.dataframe(pd.DataFrame(st.session_state.live_history[-10:]), use_container_width=True, hide_index=True)
    else:
        st.caption("No live trades yet.")


@st.fragment(run_every=5)
def trade_history_fragment():
    """Same reasoning as recent_trades_fragment — the whole Trade History tab
    now updates on its own instead of only reflecting reality after Stop."""
    hist_df = pd.DataFrame(st.session_state.live_history)
    if hist_df.empty:
        st.caption("No completed live trades yet.")
        return
    m = compute_metrics(hist_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("No. of Trades", m["total_trades"])
    c2.metric("Accuracy", f"{m['accuracy']}%")
    c3.metric("Points Gained/Lost", m["total_points"])
    c4.metric("Total PnL", m["total_pnl"])
    c5.metric("Expectancy", m["expectancy"])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear Trade History"):
        st.session_state.live_history = []
        st.rerun()


# ============================================================================
# STRATEGY SIGNAL GENERATION  (no look-ahead: signal at i uses data up to i)
# ============================================================================

def generate_signals(df, strategy, params):
    df = df.copy()
    df["signal"] = 0

    if strategy == "EMA Crossover":
        f, s = params.get("ema_fast", 9), params.get("ema_slow", 15)
        ef, es = ema(df["Close"], f), ema(df["Close"], s)
        df.loc[(ef > es) & (ef.shift(1) <= es.shift(1)), "signal"] = 1
        df.loc[(ef < es) & (ef.shift(1) >= es.shift(1)), "signal"] = -1

    elif strategy == "Simple Buy Only":
        df.loc[df["Close"] > df["Close"].shift(1), "signal"] = 1

    elif strategy == "Simple Sell Only":
        df.loc[df["Close"] < df["Close"].shift(1), "signal"] = -1

    elif strategy == "Threshold Cross":
        thr = params.get("threshold", float(df["Close"].iloc[0]))
        # Cross Direction (identical in backtest and live):
        #   "Below" (default) → price approaches from BELOW and crosses UP
        #                       through the threshold → LONG.
        #   "Above"           → price crosses DOWN from above → SHORT.
        cross_dir = params.get("threshold_direction", "Below")
        if cross_dir == "Above":
            df.loc[(df["Close"] < thr) & (df["Close"].shift(1) >= thr), "signal"] = -1
        else:
            df.loc[(df["Close"] > thr) & (df["Close"].shift(1) <= thr), "signal"] = 1

    elif strategy == "Price Action Support/Resistance":
        w = params.get("sr_window", 20)
        res = df["High"].rolling(w).max().shift(1)
        sup = df["Low"].rolling(w).min().shift(1)
        df.loc[df["Close"] > res, "signal"] = 1
        df.loc[df["Close"] < sup, "signal"] = -1

    elif strategy == "Liquidity Grab Reversal":
        w = params.get("liq_window", 20)
        vol_avg = df["Volume"].rolling(w).mean()
        rec_high = df["High"].rolling(w).max().shift(1)
        rec_low = df["Low"].rolling(w).min().shift(1)
        sweep_high = (df["High"] > rec_high) & (df["Close"] < rec_high) & (df["Volume"] > 1.5 * vol_avg)
        sweep_low = (df["Low"] < rec_low) & (df["Close"] > rec_low) & (df["Volume"] > 1.5 * vol_avg)
        df.loc[sweep_low, "signal"] = 1
        df.loc[sweep_high, "signal"] = -1

    elif strategy == "RSI Cross":
        r = rsi(df["Close"], params.get("rsi_period", 14))
        df["rsi"] = r
        df.loc[(r > 30) & (r.shift(1) <= 30), "signal"] = 1
        df.loc[(r < 70) & (r.shift(1) >= 70), "signal"] = -1

    elif strategy == "Bollinger Bands":
        upper, mid, lower = bollinger(df["Close"], params.get("bb_period", 20), params.get("bb_std", 2))
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = upper, mid, lower
        df.loc[(df["Close"] < lower) & (df["Close"].shift(1) >= lower.shift(1)), "signal"] = 1
        df.loc[(df["Close"] > upper) & (df["Close"].shift(1) <= upper.shift(1)), "signal"] = -1

    elif strategy == "Volume Breakout":
        w = params.get("vol_window", 20)
        factor = params.get("vol_factor", 2)
        vol_avg = df["Volume"].rolling(w).mean()
        rh = df["High"].rolling(w).max().shift(1)
        rl = df["Low"].rolling(w).min().shift(1)
        df.loc[(df["Close"] > rh) & (df["Volume"] > factor * vol_avg), "signal"] = 1
        df.loc[(df["Close"] < rl) & (df["Volume"] > factor * vol_avg), "signal"] = -1

    elif strategy == "Elliott Wave (Zigzag)":
        sh, sl_ = swing_points(df, params.get("zigzag_lookback", 3))
        df["swing_high"], df["swing_low"] = sh, sl_
        df.loc[sl_, "signal"] = 1
        df.loc[sh, "signal"] = -1

    elif strategy == "Pro: VWAP + Supertrend Trend":
        vw = vwap(df)
        st_line, direction = supertrend(df, params.get("st_period", 10), params.get("st_mult", 3))
        df["vwap"], df["supertrend"] = vw, st_line
        buy = (df["Close"] > vw) & (direction == 1) & (direction.shift(1) != 1)
        sell = (df["Close"] < vw) & (direction == -1) & (direction.shift(1) != -1)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Opening Range Breakout + Volume":
        n_candles = params.get("orb_candles", 5)
        dates = pd.Series(df.index.date, index=df.index)
        or_high = df.groupby(dates)["High"].transform(lambda x: x.iloc[:n_candles].max())
        or_low = df.groupby(dates)["Low"].transform(lambda x: x.iloc[:n_candles].min())
        vol_avg = df["Volume"].rolling(20).mean()
        buy = (df["Close"] > or_high) & (df["Volume"] > 1.3 * vol_avg)
        sell = (df["Close"] < or_low) & (df["Volume"] > 1.3 * vol_avg)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: BB+RSI Mean Reversion (ATR filtered)":
        upper, mid, lower = bollinger(df["Close"], 20, 2)
        r = rsi(df["Close"], 14)
        a = atr(df, 14)
        atr_pct = a / df["Close"]
        low_vol = atr_pct < atr_pct.rolling(50).mean() * 1.5
        buy = (df["Close"] < lower) & (r < 35) & low_vol.fillna(False)
        sell = (df["Close"] > upper) & (r > 65) & low_vol.fillna(False)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: EMA50 Trend + EMA9/15 Pullback":
        e50 = ema(df["Close"], 50)
        e9, e15 = ema(df["Close"], 9), ema(df["Close"], 15)
        uptrend, downtrend = df["Close"] > e50, df["Close"] < e50
        buy = uptrend & (e9 > e15) & (e9.shift(1) <= e15.shift(1))
        sell = downtrend & (e9 < e15) & (e9.shift(1) >= e15.shift(1))
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: MACD Crossover":
        macd_line, signal_line, hist = macd(df["Close"], params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9))
        df["macd"], df["macd_signal"] = macd_line, signal_line
        buy = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        sell = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Donchian Channel Breakout":
        upper, mid, lower = donchian(df, params.get("donchian_period", 20))
        buy = df["Close"] > upper.shift(1)
        sell = df["Close"] < lower.shift(1)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Keltner Squeeze Breakout":
        k_upper, k_mid, k_lower = keltner(df, params.get("keltner_period", 20), params.get("keltner_atr_mult", 1.5))
        bb_upper, bb_mid, bb_lower = bollinger(df["Close"], 20, 2)
        squeeze = (bb_upper < k_upper) & (bb_lower > k_lower)
        buy = squeeze.shift(1).fillna(False) & (df["Close"] > k_upper)
        sell = squeeze.shift(1).fillna(False) & (df["Close"] < k_lower)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Stochastic Reversal":
        k, d = stochastic(df, params.get("stoch_k", 14), params.get("stoch_d", 3))
        buy = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 30)
        sell = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 70)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: TEMA Trend Flip":
        t = tema(df["Close"], params.get("tema_period", 20))
        buy = (df["Close"] > t) & (df["Close"].shift(1) <= t.shift(1))
        sell = (df["Close"] < t) & (df["Close"].shift(1) >= t.shift(1))
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: CCI Extreme Reversal":
        c = cci(df, params.get("cci_period", 20))
        buy = (c > -100) & (c.shift(1) <= -100)
        sell = (c < 100) & (c.shift(1) >= 100)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Parabolic SAR Flip":
        sar, trend = parabolic_sar(df)
        df["sar"] = sar
        buy = (trend == 1) & (trend.shift(1) == -1)
        sell = (trend == -1) & (trend.shift(1) == 1)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: ADX/DI Directional Entry":
        plus_di, minus_di, adx_val = adx_di(df, 14)
        buy = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)) & (adx_val > 20)
        sell = (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1)) & (adx_val > 20)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Heikin-Ashi Trend Continuation":
        ha_open, ha_high, ha_low, ha_close = heikin_ashi(df)
        bullish, bearish = ha_close > ha_open, ha_close < ha_open
        buy = bullish & bullish.shift(1).fillna(False) & ~bullish.shift(2).fillna(False)
        sell = bearish & bearish.shift(1).fillna(False) & ~bearish.shift(2).fillna(False)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    elif strategy == "Pro: Ichimoku Cloud Breakout":
        tenkan, kijun, senkou_a, senkou_b = ichimoku(df)
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        buy = (df["Close"] > cloud_top) & (df["Close"].shift(1) <= cloud_top.shift(1))
        sell = (df["Close"] < cloud_bottom) & (df["Close"].shift(1) >= cloud_bottom.shift(1))
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

    df["signal"] = df["signal"].fillna(0)
    return apply_signal_direction_rules(df, params)


def apply_signal_direction_rules(df, params):
    """CENTRAL application (backtest, optimization, heatmaps, live all obey):
    1) Flip / Reverse Entries FIRST (Long ↔ Short) when enabled.
    2) THEN the Trade Direction filter (Both / Long Only / Short Only).
    Instrument mapping downstream is preserved automatically — flipped
    signals in options simply BUY the other leg."""
    params = params or {}
    if params.get("flip_signals"):
        df["signal"] = -df["signal"]
    td = params.get("trade_direction", "Both")
    if td == "Long Only":
        df.loc[df["signal"] == -1, "signal"] = 0
    elif td == "Short Only":
        df.loc[df["signal"] == 1, "signal"] = 0
    return df


def apply_direction_rules_to_scalar(sig, params):
    """Same flip-first-then-filter rule for a single live-computed signal
    (used by the immediate-execution LTP path)."""
    params = params or {}
    if params.get("flip_signals"):
        sig = -sig
    td = params.get("trade_direction", "Both")
    if td == "Long Only" and sig == -1:
        return 0
    if td == "Short Only" and sig == 1:
        return 0
    return sig


def apply_filters(df, filters, params=None):
    params = params or {}
    df = df.copy()
    mask_buy = df["signal"] == 1
    mask_sell = df["signal"] == -1

    if filters.get("adx_enabled"):
        a = adx(df, 14)
        df["adx_f"] = a
        ok = (a >= filters.get("adx_min", 0)) & (a <= filters.get("adx_max", 100))
        mask_buy &= ok.fillna(False)
        mask_sell &= ok.fillna(False)

    if filters.get("rsi_enabled"):
        r = rsi(df["Close"], 14)
        rsi_buy_ok = (r > 30) & (r.shift(1) <= 30)
        rsi_sell_ok = (r < 70) & (r.shift(1) >= 70)
        mask_buy &= rsi_buy_ok.fillna(False)
        mask_sell &= rsi_sell_ok.fillna(False)

    if filters.get("bb_enabled"):
        upper, mid, lower = bollinger(df["Close"], 20, 2)
        mask_buy &= (df["Close"] <= upper).fillna(False)
        mask_sell &= (df["Close"] >= lower).fillna(False)

    if filters.get("ema20_enabled"):
        e20 = ema(df["Close"], 20)
        mask_buy &= (df["Close"] > e20).fillna(False)
        mask_sell &= (df["Close"] < e20).fillna(False)

    if filters.get("sma20_enabled"):
        s20 = sma(df["Close"], 20)
        mask_buy &= (df["Close"] > s20).fillna(False)
        mask_sell &= (df["Close"] < s20).fillna(False)

    if filters.get("smc_enabled"):
        sh, sl_ = swing_points(df, 3)
        last_high = df["High"].where(sh).ffill()
        last_low = df["Low"].where(sl_).ffill()
        bos_up = (df["Close"] > last_high.shift(1)).fillna(False)
        bos_dn = (df["Close"] < last_low.shift(1)).fillna(False)
        mask_buy &= bos_up
        mask_sell &= bos_dn

    if filters.get("atr_enabled"):
        a = atr(df, 14)
        df["atr_f"] = a
        ok = (a >= filters.get("atr_min", 0.0)) & (a <= filters.get("atr_max", 1e9))
        mask_buy &= ok.fillna(False)
        mask_sell &= ok.fillna(False)

    if filters.get("supertrend_enabled"):
        st_line, st_dir = supertrend(df, filters.get("st_filter_period", 10), filters.get("st_filter_mult", 3.0))
        df["supertrend_f"], df["supertrend_dir_f"] = st_line, st_dir
        mask_buy &= (st_dir == 1)
        mask_sell &= (st_dir == -1)

    if filters.get("regime_enabled"):
        a = adx(df, 14)
        df["regime_adx"] = a
        trend_ok = a >= filters.get("regime_trend_min", 25)
        range_ok = a <= filters.get("regime_range_max", 20)
        family = STRATEGY_FAMILY.get(filters.get("current_strategy"), "neutral")
        if family == "trend":
            mask_buy &= trend_ok.fillna(False)
            mask_sell &= trend_ok.fillna(False)
        elif family == "mean_reversion":
            mask_buy &= range_ok.fillna(False)
            mask_sell &= range_ok.fillna(False)
        # "neutral" strategies are left ungated by regime

    if filters.get("vix_enabled"):
        vix_aligned = get_vix_aligned(df.index)
        df["vix_f"] = vix_aligned.values
        ok = (vix_aligned >= filters.get("vix_min", 0)) & (vix_aligned <= filters.get("vix_max", 100))
        ok = pd.Series(ok.values, index=df.index).fillna(False)
        mask_buy &= ok
        mask_sell &= ok

    # --- Crossover Angle / Crossover Quality filters ---
    # These only constrain entries that coincide with an actual EMA{fast}/{slow}
    # crossover in the SAME bar (using the fast/slow periods set for the main
    # strategy). If a signal fires from a strategy/bar where no such crossover
    # is happening, these two filters have no effect on it — "angle" and
    # "crossover candle size" are only meaningful at the moment of a cross.
    if filters.get("angle_enabled") or filters.get("crossover_quality_enabled"):
        f, s = params.get("ema_fast", 9), params.get("ema_slow", 15)
        ef, es = ema(df["Close"], f), ema(df["Close"], s)
        cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
        cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
        a_series_for_angle = atr(df, 14)

        angle_ok = pd.Series(True, index=df.index)
        if filters.get("angle_enabled"):
            # Angle is scale-dependent by nature (a raw price-difference slope has
            # no inherent "degrees"), so it's normalized against ATR — the
            # steepness of the EMA move relative to the instrument's own typical
            # bar range. This is a disclosed heuristic, not a standardized
            # industry figure; absolute value is used since a valid crossover in
            # either direction can produce a negative raw slope depending on sign
            # convention.
            ema_fast_delta = ef.diff()
            angle_deg = np.degrees(np.arctan2(ema_fast_delta.abs(), a_series_for_angle.replace(0, np.nan)))
            df["crossover_angle_deg"] = angle_deg
            angle_ok = (angle_deg >= filters.get("angle_min_deg", 0)).fillna(False)

        quality_ok = pd.Series(True, index=df.index)
        if filters.get("crossover_quality_enabled"):
            mode = filters.get("crossover_quality_mode", "Simple Crossover")
            candle_range = (df["High"] - df["Low"])
            if mode == "Crossover with Candle Size":
                quality_ok = (candle_range >= filters.get("crossover_min_points", 1.0)).fillna(False)
            elif mode == "Crossover with ATR-based Candle Size":
                quality_ok = (candle_range >= a_series_for_angle * filters.get("crossover_atr_mult", 1.0)).fillna(False)
            # "Simple Crossover" = no extra size requirement, quality_ok stays True

        cross_condition_ok = angle_ok & quality_ok
        # Only gate the bars that ARE crossovers; leave all other bars untouched.
        mask_buy &= (~cross_up) | (cross_up & cross_condition_ok)
        mask_sell &= (~cross_dn) | (cross_dn & cross_condition_ok)

    new_signal = pd.Series(0, index=df.index)
    new_signal[mask_buy] = 1
    new_signal[mask_sell] = -1
    df["signal"] = new_signal
    return df


# ============================================================================
# SL / TARGET ENGINE
# ============================================================================

def calc_initial_sl_target(direction, entry_price, atr_val, params, sl_type, target_type):
    sl_points = params.get("sl_points", 10.0)
    target_points = params.get("target_points", 20.0)
    rr_ratio = max(params.get("rr_ratio", 2.0), 2.0)

    if sl_type == "ATR Based SL":
        sl_dist = atr_val * params.get("atr_mult_sl", 1.5)
    elif sl_type == "Autopilot SL":
        sl_dist = max(atr_val * 1.2, sl_points)
    elif sl_type == "Loss Recovery SL (Give-back)":
        # This SL type's real exit logic is the give-back check in
        # check_special_exit_conditions(); this hard level is only a wide
        # backstop in case price gaps straight through it.
        sl_dist = max(atr_val * 3.0, params.get("loss_trigger_points", 20.0) * 1.5)
    else:
        sl_dist = sl_points

    if target_type == "ATR Based Target":
        target_dist = atr_val * params.get("atr_mult_target", 3.0)
    elif target_type == "Risk:Reward Based (min 1:2)":
        target_dist = sl_dist * rr_ratio
    elif target_type == "Autopilot Target":
        target_dist = max(atr_val * 2.5, sl_dist * 2)
    elif target_type == "Profit Giveback Target":
        # Real exit logic is the give-back check; this is a wide backstop.
        target_dist = max(atr_val * 4.0, params.get("profit_trigger_points", 50.0) * 1.5)
    elif target_type == "Partial Book + Trail Remainder":
        # This IS the real, actionable level — it's Target 1, the point at
        # which the first tranche gets booked (checked via the normal hard-exit
        # path, then intercepted in run_backtest to book partially instead of
        # closing fully).
        target_dist = params.get("partial_target1_points", target_points)
    else:
        target_dist = target_points

    if sl_type == "Risk:Reward Based (min 1:2)":
        target_dist = max(target_dist, sl_dist * rr_ratio)

    if direction == 1:
        sl, target = entry_price - sl_dist, entry_price + target_dist
    else:
        sl, target = entry_price + sl_dist, entry_price - target_dist
    return sl, target, sl_dist, target_dist


def check_special_exit_conditions(trade, candle):
    """Stateful exits that can't be expressed as a single fixed level:
    Loss Recovery SL (cut losers that don't bounce back enough) and Profit
    Giveback Target (lock in winners that give back too much of their peak)."""
    direction = trade["direction"]
    current_pl = (candle["Close"] - trade["entry_price"]) * direction

    if trade["sl_type"] == "Loss Recovery SL (Give-back)":
        trigger = trade.get("loss_trigger_points", 20.0)
        recovery_pct = trade.get("min_recovery_pct", 50.0) / 100.0
        prev_worst = trade.get("worst_pl_points", 0.0)
        worst = min(prev_worst, current_pl)
        trade["worst_pl_points"] = worst
        is_fresh_low = current_pl <= prev_worst
        # Only judge "did it recover enough" on candles where price has
        # actually ticked UP from its worst point — checking on the very
        # candle that SETS a new worst is tautological (current == worst
        # there), which would cut every loser the instant it first touched
        # the trigger with no chance to bounce at all.
        if worst <= -trigger and not is_fresh_low:
            required_level = worst + recovery_pct * trigger
            if current_pl <= required_level:
                return True, float(candle["Close"]), (
                    f"Loss Recovery SL (down {abs(worst):.1f} pts, recovered < {trade.get('min_recovery_pct',50):.0f}%)"
                )

    if trade["target_type"] == "Profit Giveback Target":
        trigger = trade.get("profit_trigger_points", 50.0)
        giveback_pct = trade.get("giveback_pct", 30.0) / 100.0
        peak = max(trade.get("peak_pl_points", 0.0), current_pl)
        trade["peak_pl_points"] = peak
        if peak >= trigger:
            giveback_level = peak * (1 - giveback_pct)
            if current_pl <= giveback_level:
                return True, float(candle["Close"]), (
                    f"Profit Giveback (peak {peak:.1f} pts, gave back > {trade.get('giveback_pct',30):.0f}%)"
                )

    return False, None, None


def check_time_based_exit(trade, candle_time, candle_close, min_minutes, max_minutes):
    """
    Exits a position once it has been in a continuous floating loss for at
    least `min_minutes`. `max_minutes` is a documented upper safety bound
    (mainly relevant to live polling where checks may lag) — set it >=
    min_minutes. Resets the loss-timer the instant the trade turns flat/green.
    """
    direction = trade["direction"]
    current_pl = (candle_close - trade["entry_price"]) * direction
    if current_pl < 0:
        if trade.get("loss_since") is None:
            trade["loss_since"] = candle_time
        elapsed_min = (candle_time - trade["loss_since"]).total_seconds() / 60.0
        if elapsed_min >= min_minutes:
            return True, float(candle_close), f"Time-Based Loss Exit (in loss {elapsed_min:.1f}m, threshold {min_minutes:.0f}-{max_minutes:.0f}m)"
    else:
        trade["loss_since"] = None
    return False, None, None




def update_trade_levels(trade, i, df, params, atr_series):
    direction = trade["direction"]
    candle = df.iloc[i]
    prev_candle = df.iloc[i - 1] if i > 0 else candle
    sl_type, target_type = trade["sl_type"], trade["target_type"]

    trade["highest"] = max(trade.get("highest", candle["High"]), candle["High"])
    trade["lowest"] = min(trade.get("lowest", candle["Low"]), candle["Low"])

    a_val = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else trade["sl_dist"]

    if sl_type == "Trailing SL (Points)":
        d = trade["sl_dist"]
        trade["sl"] = max(trade["sl"], trade["highest"] - d) if direction == 1 else min(trade["sl"], trade["lowest"] + d)
    elif sl_type == "Trail Candle Low/High (Current)":
        trade["sl"] = max(trade["sl"], candle["Low"]) if direction == 1 else min(trade["sl"], candle["High"])
    elif sl_type == "Trail Candle Low/High (Previous)":
        trade["sl"] = max(trade["sl"], prev_candle["Low"]) if direction == 1 else min(trade["sl"], prev_candle["High"])
    elif sl_type in ("Trail Swing Low/High (Current)", "Trail Swing Low/High (Previous)"):
        span = df.iloc[max(0, i - 10):(i if "Previous" in sl_type else i + 1)]
        if direction == 1 and not span.empty:
            trade["sl"] = max(trade["sl"], span["Low"].min())
        elif direction == -1 and not span.empty:
            trade["sl"] = min(trade["sl"], span["High"].max())
    elif sl_type == "ATR Based SL":
        mult = params.get("atr_mult_sl", 1.5)
        trade["sl"] = max(trade["sl"], candle["Close"] - a_val * mult) if direction == 1 else min(trade["sl"], candle["Close"] + a_val * mult)
    elif sl_type == "Autopilot SL":
        profit = (candle["Close"] - trade["entry_price"]) * direction
        tighten = 0.7 if profit > trade["sl_dist"] else 1.4
        trade["sl"] = max(trade["sl"], candle["Close"] - a_val * tighten) if direction == 1 else min(trade["sl"], candle["Close"] + a_val * tighten)

    if target_type == "Trail Candle Low/High (Current)":
        trade["target"] = max(trade["target"], candle["High"]) if direction == 1 else min(trade["target"], candle["Low"])
    elif target_type == "Trail Candle Low/High (Previous)":
        trade["target"] = max(trade["target"], prev_candle["High"]) if direction == 1 else min(trade["target"], prev_candle["Low"])
    elif target_type in ("Trail Swing Low/High (Current)", "Trail Swing Low/High (Previous)"):
        span = df.iloc[max(0, i - 10):(i if "Previous" in target_type else i + 1)]
        if direction == 1 and not span.empty:
            trade["target"] = max(trade["target"], span["High"].max())
        elif direction == -1 and not span.empty:
            trade["target"] = min(trade["target"], span["Low"].min())
    elif target_type == "Autopilot Target":
        trade["target"] = max(trade["target"], candle["Close"] + a_val * 2.5) if direction == 1 else min(trade["target"], candle["Close"] - a_val * 2.5)

    return trade


def detect_signal_exit_condition(trade, i, df, params):
    """
    Detects a strategy-reverse or EMA-reverse exit condition using data known
    at the CLOSE of candle i. This is only ever used to SCHEDULE an exit for
    execution at the OPEN of candle i+1 (see run_backtest / live tab) — never
    executed on candle i itself. Executing it on candle i would mean using
    that candle's own close to justify a fill at that candle's open, which is
    look-ahead bias no live system could actually achieve.
    """
    direction = trade["direction"]
    sl_type, target_type = trade["sl_type"], trade["target_type"]

    if sl_type == "Strategy Signal Exit" or target_type == "Strategy Signal Exit":
        sig = df["signal"].iloc[i]
        if (direction == 1 and sig == -1) or (direction == -1 and sig == 1):
            return True, "Strategy Reverse Signal"

    if sl_type == "EMA Reverse Crossover Exit" or target_type == "EMA Reverse Crossover Exit":
        f, s = params.get("ema_fast", 9), params.get("ema_slow", 15)
        close_slice = df["Close"].iloc[: i + 1]
        if len(close_slice) > max(f, s) + 1:
            ef, es = ema(close_slice, f), ema(close_slice, s)
            if direction == 1 and ef.iloc[-1] < es.iloc[-1] and ef.iloc[-2] >= es.iloc[-2]:
                return True, "EMA Reverse Crossover"
            if direction == -1 and ef.iloc[-1] > es.iloc[-1] and ef.iloc[-2] <= es.iloc[-2]:
                return True, "EMA Reverse Crossover"

    return False, None


def check_hard_exit(trade, candle):
    """
    BACKTEST-ONLY. Hard SL/Target check using only the CURRENT candle's own
    high/low against levels set from PAST data (entry price, ATR at signal
    time, trailing updates). No look-ahead here — these levels never depend
    on this candle's own close. Conservative fill order: longs check SL(low)
    before Target(high); shorts check SL(high) before Target(low).

    This candle-range approach exists because a backtest has no live ticks —
    only OHLC bars — so it can't know the exact path price took inside a
    candle, hence the conservative "assume the worse touch happened first"
    rule. Live trading uses check_hard_exit_ltp() below instead, which
    compares directly against the last-traded price — see that function's
    docstring for why that's the correct approach once you have real tick
    data (e.g. via Dhan) instead of polled candles.
    """
    direction = trade["direction"]
    target_display_only = trade["target_type"] == "Trailing Target (Display Only)"

    if direction == 1:
        if candle["Low"] <= trade["sl"]:
            return True, trade["sl"], "Stoploss Hit"
        if not target_display_only and candle["High"] >= trade["target"]:
            return True, trade["target"], "Target Hit"
    else:
        if candle["High"] >= trade["sl"]:
            return True, trade["sl"], "Stoploss Hit"
        if not target_display_only and candle["Low"] <= trade["target"]:
            return True, trade["target"], "Target Hit"

    return False, None, None


def check_hard_exit_ltp(trade, ltp):
    """
    LIVE-ONLY. Compares SL/Target directly against the last-traded price
    instead of a candle's high/low. SL is checked before Target for both
    directions — same conservative "risk first" ordering as backtest, just
    evaluated against a single live price point rather than a candle range,
    since with a real tick feed there's no ambiguity about what price was
    actually touched (unlike a completed OHLC bar, where the touch order of
    two levels hit in the same candle is genuinely unknowable).
    """
    direction = trade["direction"]
    target_display_only = trade["target_type"] == "Trailing Target (Display Only)"

    if direction == 1:
        if ltp <= trade["sl"]:
            return True, trade["sl"], "Stoploss Hit (LTP)"
        if not target_display_only and ltp >= trade["target"]:
            return True, trade["target"], "Target Hit (LTP)"
    else:
        if ltp >= trade["sl"]:
            return True, trade["sl"], "Stoploss Hit (LTP)"
        if not target_display_only and ltp <= trade["target"]:
            return True, trade["target"], "Target Hit (LTP)"

    return False, None, None


def get_live_ltp(ticker):
    """
    Fetches the freshest possible last-traded price, bypassing the cached
    fetch_data() used for candle/indicator data (that cache has a 30s TTL,
    fine for indicators, too stale for a live SL/Target trigger check).

    DHAN INTEGRATION (now wired): when the Dhan data feed is enabled with a
    token and the ticker is servable, the real Dhan tick is returned with NO
    delay. Every SL/Target check in evaluate_live_signal() automatically uses
    it. Otherwise this falls back to the original yfinance path (0.3s delay).
    """
    if dhan_feed_active():
        ltp = dhan_get_ltp_for_ticker(ticker)
        if ltp is not None:
            return ltp
    time.sleep(RATE_LIMIT_DELAY)
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data is None or data.empty:
            data = yf.Ticker(ticker).history(period="5d", interval="15m")
        if data is not None and not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(raw_df, strategy, sl_type, target_type, params, filters, qty, risk_ctrl=None):
    if raw_df.empty or len(raw_df) < 30:
        return pd.DataFrame(), raw_df

    risk_ctrl = risk_ctrl or {}
    filters = dict(filters or {})
    filters["current_strategy"] = strategy

    df = generate_signals(raw_df, strategy, params)
    df = apply_filters(df, filters, params)
    atr_series = atr(df, 14)

    trades = []
    open_trade = None

    def _exit_candle_ohlc(exit_time):
        """Exit candle's own Open/High/Low/Close for the trade-history row."""
        try:
            row = df.loc[exit_time]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {
                "Exit Open": round(float(row["Open"]), 2), "Exit High": round(float(row["High"]), 2),
                "Exit Low": round(float(row["Low"]), 2), "Exit Close": round(float(row["Close"]), 2),
            }
        except Exception:
            return {"Exit Open": np.nan, "Exit High": np.nan, "Exit Low": np.nan, "Exit Close": np.nan}

    def close_trade(exit_price, exit_time, reason, qty_to_close):
        points = (exit_price - open_trade["entry_price"]) * open_trade["direction"]
        trades.append({
            "Entry Time": open_trade["entry_time"], "Entry Price": round(open_trade["entry_price"], 2),
            "Direction": "LONG" if open_trade["direction"] == 1 else "SHORT",
            "Exit Time": exit_time, "Exit Price": round(float(exit_price), 2),
            "SL": round(open_trade["initial_sl"], 2), "Target": round(open_trade["initial_target"], 2),
            "Highest": round(open_trade["highest"], 2), "Lowest": round(open_trade["lowest"], 2),
            "Points": round(points, 2), "PnL": round(points * qty_to_close, 2),
            "Exit Reason": reason, "Qty": qty_to_close,
            **_exit_candle_ohlc(exit_time),
        })

    for i in range(1, len(df) - 1):
        if open_trade is None:
            sig = df["signal"].iloc[i]
            if sig == -1 and params.get("long_entries_only"):
                # Premium (options-buyer) mode: SHORT signals never OPEN a
                # position — the signal itself stays in the series so it can
                # still exit an open long via 'Strategy Signal Exit'.
                continue
            if sig != 0:
                if strategy in IMMEDIATE_EXECUTION_STRATEGIES:
                    # No candle shape to wait for — the condition (price vs
                    # prev close, or price crossing a threshold) is already
                    # fully known at this candle's close, so fill right here.
                    entry_idx = i
                    entry_price = float(df["Close"].iloc[i])
                else:
                    entry_idx = i + 1
                    if entry_idx >= len(df):
                        break
                    entry_price = float(df["Open"].iloc[entry_idx])
                a_val = atr_series.iloc[i]
                a_val = a_val if not np.isnan(a_val) else entry_price * 0.005
                sl, target, sl_dist, target_dist = calc_initial_sl_target(sig, entry_price, a_val, params, sl_type, target_type)
                open_trade = {
                    "entry_time": df.index[entry_idx], "entry_price": entry_price,
                    "direction": int(sig), "qty": qty, "sl": sl, "target": target,
                    "initial_sl": sl, "initial_target": target,
                    "sl_dist": sl_dist, "target_dist": target_dist,
                    "sl_type": sl_type, "target_type": target_type,
                    "highest": entry_price, "lowest": entry_price,
                    "signal_candle": df.index[i], "entry_idx": entry_idx,
                    "pending_exit_reason": None,
                    "peak_pl_points": 0.0, "worst_pl_points": 0.0, "loss_since": None,
                    "original_qty": qty, "remaining_qty": qty, "partial_booked": False,
                    "loss_trigger_points": params.get("loss_trigger_points", 20.0),
                    "min_recovery_pct": params.get("min_recovery_pct", 50.0),
                    "profit_trigger_points": params.get("profit_trigger_points", 50.0),
                    "giveback_pct": params.get("giveback_pct", 30.0),
                    "partial_book_pct": params.get("partial_book_pct", 50.0),
                }
        else:
            if i < open_trade["entry_idx"]:
                continue
            candle = df.iloc[i]
            open_trade = update_trade_levels(open_trade, i, df, params, atr_series)

            exited, exit_price, reason = False, None, None

            # 1) A signal/EMA-reverse exit detected on the PREVIOUS candle's
            #    close is executed here, at THIS candle's open.
            if open_trade.get("pending_exit_reason"):
                exited, exit_price, reason = True, candle["Open"], open_trade["pending_exit_reason"]

            # 2) Stateful special exits (Loss Recovery SL / Profit Giveback Target).
            if not exited:
                sp_exit, sp_price, sp_reason = check_special_exit_conditions(open_trade, candle)
                if sp_exit:
                    exited, exit_price, reason = True, sp_price, sp_reason

            # 2b) Time-based loss-holding-duration exit (if enabled).
            if not exited and risk_ctrl.get("loss_duration_enabled"):
                td_exit, td_price, td_reason = check_time_based_exit(
                    open_trade, df.index[i], candle["Close"],
                    risk_ctrl.get("loss_duration_min_minutes", 1), risk_ctrl.get("loss_duration_max_minutes", 5),
                )
                if td_exit:
                    exited, exit_price, reason = True, td_price, td_reason

            # 3) Hard SL/Target on this candle. For "Partial Book + Trail
            #    Remainder", a Target Hit here means Target 1 — book part of
            #    the quantity and keep the rest running under a trailing stop
            #    instead of a full close.
            if not exited:
                hard_exit, hard_price, hard_reason = check_hard_exit(open_trade, candle)
                if hard_exit:
                    if (open_trade["target_type"] == "Partial Book + Trail Remainder"
                            and hard_reason == "Target Hit" and not open_trade["partial_booked"]):
                        book_qty = max(1, round(open_trade["original_qty"] * open_trade["partial_book_pct"] / 100.0))
                        book_qty = min(book_qty, open_trade["remaining_qty"])
                        partial_points = (hard_price - open_trade["entry_price"]) * open_trade["direction"]
                        trades.append({
                            "Entry Time": open_trade["entry_time"], "Entry Price": round(open_trade["entry_price"], 2),
                            "Direction": "LONG" if open_trade["direction"] == 1 else "SHORT",
                            "Exit Time": df.index[i], "Exit Price": round(float(hard_price), 2),
                            "SL": round(open_trade["initial_sl"], 2), "Target": round(open_trade["initial_target"], 2),
                            "Highest": round(open_trade["highest"], 2), "Lowest": round(open_trade["lowest"], 2),
                            "Points": round(partial_points, 2), "PnL": round(partial_points * book_qty, 2),
                            "Exit Reason": f"Partial Book ({book_qty}/{open_trade['original_qty']} qty @ Target 1)",
                            "Qty": book_qty,
                            "Exit Open": round(float(candle["Open"]), 2), "Exit High": round(float(candle["High"]), 2),
                            "Exit Low": round(float(candle["Low"]), 2), "Exit Close": round(float(candle["Close"]), 2),
                        })
                        open_trade["remaining_qty"] -= book_qty
                        open_trade["partial_booked"] = True
                        if open_trade["remaining_qty"] <= 0:
                            open_trade = None
                        else:
                            # Remainder now runs on a trailing stop only — no
                            # fixed second target — and the SL is forced onto
                            # an ATR trail if it wasn't already trailing, so
                            # the remainder is never left unprotected.
                            open_trade["target_type"] = "Trailing Target (Display Only)"
                            if open_trade["sl_type"] not in ("Trailing SL (Points)", "ATR Based SL", "Autopilot SL"):
                                open_trade["sl_type"] = "ATR Based SL"
                        continue
                    else:
                        exited, exit_price, reason = True, hard_price, hard_reason

            if exited:
                close_trade(exit_price, df.index[i], reason, open_trade["remaining_qty"])
                open_trade = None
            elif open_trade is not None:
                # 4) Detect (but don't act on) a new signal/EMA-reverse exit
                #    using this candle's own close — scheduled for next candle.
                sig_exit, sig_reason = detect_signal_exit_condition(open_trade, i, df, params)
                if sig_exit:
                    open_trade["pending_exit_reason"] = sig_reason

    if open_trade is not None:
        last_i = len(df) - 1
        close_trade(df["Close"].iloc[last_i], df.index[last_i], "End of Data (Forced Close)", open_trade["remaining_qty"])

    return pd.DataFrame(trades), df


def recommend_sl_target_from_mae_mfe(sig_df, trades_df, lookahead=20):
    """
    Recommends SL/Target distances from the ACTUAL adverse/favorable price
    excursions your signals produced — not a guess, and not the same as
    "whatever SL/Target you happened to backtest with". For every trade:
      MAE (Max Adverse Excursion)  = worst move against you before exit/lookahead
      MFE (Max Favorable Excursion) = best move in your favor before exit/lookahead
    SL is suggested at a percentile of the MAE distribution (tight enough to
    matter, loose enough to survive normal noise). Target is suggested at a
    more conservative percentile of MFE (a realistically reachable level, not
    the best-case outlier). This is standard MAE/MFE analysis, a real
    technique used to size stops/targets off a strategy's own behavior.
    """
    if trades_df is None or trades_df.empty or sig_df is None or sig_df.empty:
        return None

    mae_list, mfe_list = [], []
    for _, row in trades_df.iterrows():
        entry_time = row["Entry Time"]
        if entry_time not in sig_df.index:
            continue
        entry_idx = sig_df.index.get_loc(entry_time)
        direction = 1 if row["Direction"] == "LONG" else -1
        entry_price = row["Entry Price"]
        window = sig_df.iloc[entry_idx: entry_idx + lookahead]
        if window.empty:
            continue
        if direction == 1:
            mae = entry_price - window["Low"].min()
            mfe = window["High"].max() - entry_price
        else:
            mae = window["High"].max() - entry_price
            mfe = entry_price - window["Low"].min()
        mae_list.append(max(mae, 0))
        mfe_list.append(max(mfe, 0))

    if len(mae_list) < 5:
        return None

    mae_arr, mfe_arr = np.array(mae_list), np.array(mfe_list)
    return {
        "n_trades": len(mae_arr),
        "mae_p50": float(np.percentile(mae_arr, 50)), "mae_p70": float(np.percentile(mae_arr, 70)), "mae_p90": float(np.percentile(mae_arr, 90)),
        "mfe_p50": float(np.percentile(mfe_arr, 50)), "mfe_p70": float(np.percentile(mfe_arr, 70)), "mfe_p90": float(np.percentile(mfe_arr, 90)),
        "suggested_sl": float(np.percentile(mae_arr, 70)),
        "suggested_target": float(np.percentile(mfe_arr, 50)),
    }


def compute_metrics(trades_df):
    if trades_df is None or trades_df.empty:
        return dict(total_trades=0, wins=0, losses=0, accuracy=0.0, total_points=0.0, total_pnl=0.0,
                    avg_win=0.0, avg_loss=0.0, expectancy=0.0, sharpe=0.0, max_drawdown=0.0)
    total = len(trades_df)
    wins = int((trades_df["Points"] > 0).sum())
    losses = total - wins
    accuracy = wins / total * 100
    avg_win = trades_df.loc[trades_df["Points"] > 0, "Points"].mean() if wins else 0.0
    avg_loss = trades_df.loc[trades_df["Points"] <= 0, "Points"].mean() if losses else 0.0
    win_rate = wins / total
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    returns = trades_df["Points"]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0.0
    cum = returns.cumsum()
    drawdown = (cum - cum.cummax()).min()
    return dict(total_trades=total, wins=wins, losses=losses, accuracy=round(accuracy, 2),
                total_points=round(returns.sum(), 2), total_pnl=round(trades_df["PnL"].sum(), 2),
                avg_win=round(avg_win, 2), avg_loss=round(avg_loss, 2), expectancy=round(expectancy, 2),
                sharpe=round(sharpe, 2), max_drawdown=round(float(drawdown), 2))


def apply_cost_model(trades_df, cost_cfg, qty):
    """Adds cost-adjusted Points/PnL columns. Costs are subtracted per round-trip
    trade: (slippage + spread) in points, plus brokerage charged on both legs."""
    if trades_df is None or trades_df.empty:
        return trades_df
    trades_df = trades_df.copy()
    per_trade_point_cost = cost_cfg.get("slippage_points", 0.0) + cost_cfg.get("spread_points", 0.0)
    per_trade_cash_cost = cost_cfg.get("brokerage_flat", 0.0) * 2
    trades_df["Points (Net)"] = (trades_df["Points"] - per_trade_point_cost).round(2)
    trades_df["PnL (Net)"] = (trades_df["Points (Net)"] * qty - per_trade_cash_cost).round(2)
    return trades_df


def compute_metrics_from_columns(trades_df, points_col="Points", pnl_col="PnL"):
    """Same as compute_metrics but usable on either raw or cost-adjusted columns."""
    if trades_df is None or trades_df.empty:
        return compute_metrics(trades_df)
    tmp = pd.DataFrame({"Points": trades_df[points_col].values, "PnL": trades_df[pnl_col].values})
    return compute_metrics(tmp)


def walk_forward_folds(trades_df, start_time, end_time, n_folds):
    """Splits the backtest's time range into n sequential folds and computes
    metrics for whichever trades fall (by entry time) into each fold. This is
    an out-of-sample CONSISTENCY check (does the edge hold up across sub
    periods), not a parameter re-optimization walk-forward."""
    if trades_df is None or trades_df.empty:
        return []
    edges = pd.date_range(start=start_time, end=end_time, periods=n_folds + 1)
    fold_rows = []
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        mask = (trades_df["Entry Time"] >= lo) & (trades_df["Entry Time"] < hi if k < n_folds - 1 else trades_df["Entry Time"] <= hi)
        sub = trades_df[mask]
        m = compute_metrics(sub)
        m["Fold"] = k + 1
        m["From"] = lo
        m["To"] = hi
        fold_rows.append(m)
    return fold_rows


def smart_verdict(metrics, wf_fold_metrics=None, cost_enabled=False, metrics_net=None):
    """
    Rule-based composite score across accuracy, expectancy, Sharpe, drawdown,
    out-of-sample fold consistency (if walk-forward was run), and cost-adjusted
    expectancy (if cost modeling was run). This is a transparent heuristic
    scorecard, NOT a trained machine-learning model — there is no hidden
    "learning" happening beyond aggregating the metrics you can already see
    into one verdict, so you can sanity-check it against the numbers yourself.
    """
    notes = []
    if metrics["total_trades"] < 10:
        return "⚪ NOT ENOUGH DATA", ["Fewer than 10 trades in this sample — verdict would be unreliable. Use a longer period or lower timeframe."]

    score = 0
    if metrics["expectancy"] > 0:
        score += 1; notes.append(f"✅ Positive raw expectancy ({metrics['expectancy']} pts/trade)")
    else:
        score -= 1; notes.append(f"❌ Negative raw expectancy ({metrics['expectancy']} pts/trade)")

    if metrics["accuracy"] >= 50:
        score += 1; notes.append(f"✅ Win rate ≥ 50% ({metrics['accuracy']}%)")
    else:
        notes.append(f"⚠️ Win rate below 50% ({metrics['accuracy']}%) — relies on winners being bigger than losers")

    if metrics["sharpe"] > 0.5:
        score += 1; notes.append(f"✅ Sharpe > 0.5 ({metrics['sharpe']})")
    elif metrics["sharpe"] > 0:
        notes.append(f"⚠️ Sharpe is positive but weak ({metrics['sharpe']})")
    else:
        score -= 1; notes.append(f"❌ Non-positive Sharpe ({metrics['sharpe']})")

    if wf_fold_metrics:
        profitable_folds = sum(1 for f in wf_fold_metrics if f["total_trades"] > 0 and f["expectancy"] > 0)
        valid_folds = sum(1 for f in wf_fold_metrics if f["total_trades"] > 0)
        if valid_folds > 0:
            consistency = profitable_folds / valid_folds
            if consistency >= 0.6:
                score += 2; notes.append(f"✅ Edge held up in {profitable_folds}/{valid_folds} out-of-sample folds ({consistency:.0%}) — reasonably consistent")
            elif consistency >= 0.4:
                notes.append(f"⚠️ Edge held up in only {profitable_folds}/{valid_folds} folds ({consistency:.0%}) — inconsistent across time")
            else:
                score -= 2; notes.append(f"❌ Edge held up in only {profitable_folds}/{valid_folds} folds ({consistency:.0%}) — looks like overfitting to one period, not a real edge")
        else:
            notes.append("⚪ Walk-forward folds had too few trades each to judge.")
    else:
        notes.append("⚪ Walk-forward validation not run — this verdict is based on a single sample only, which is the weakest form of evidence. Turn it on for a more trustworthy read.")

    if cost_enabled and metrics_net is not None:
        if metrics_net["expectancy"] > 0:
            score += 1; notes.append(f"✅ Edge survives realistic costs (net expectancy {metrics_net['expectancy']} pts/trade)")
        else:
            score -= 2; notes.append(f"❌ Edge DISAPPEARS after realistic costs (net expectancy {metrics_net['expectancy']} pts/trade) — this is the single most common reason retail systems look good on paper and lose money live")
    else:
        notes.append("⚪ Cost modeling not run — raw backtest numbers usually overstate real returns once slippage, spread, and brokerage are included.")

    if score >= 4:
        verdict = "🟢 LIKELY TO BE PROFITABLE — reasonable candidate for cautious, small-size live/paper testing"
    elif score >= 1:
        verdict = "🟡 MARGINAL EDGE — proceed only with small size, tight risk control, and continued monitoring"
    else:
        verdict = "🔴 LIKELY TO CAUSE LOSSES — this configuration does not show a reliable edge; not recommended for live deployment as-is"

    return verdict, notes


def recommend_from_metrics(m):
    if m["total_trades"] < 5:
        return "⚪ Not enough trades in this sample to judge the strategy — widen the period."
    if m["accuracy"] >= 55 and m["expectancy"] > 0 and m["sharpe"] > 0.3:
        return f"🟢 Reasonable edge on this sample: {m['accuracy']}% win rate, positive expectancy ({m['expectancy']} pts/trade), Sharpe {m['sharpe']}. Still validate on more regimes and with realistic slippage/costs before sizing up."
    if m["expectancy"] > 0:
        return f"🟡 Marginal edge: expectancy is positive ({m['expectancy']} pts/trade) but win rate ({m['accuracy']}%) or Sharpe ({m['sharpe']}) is weak. Consider tightening filters (ADX/RSI/volume) or a different timeframe."
    return f"🔴 No robust edge detected on this sample (expectancy {m['expectancy']}, accuracy {m['accuracy']}%). Try other strategy/timeframe combos in the Optimization tab before trading this config live."


# ============================================================================
# DHAN BROKER — REAL ORDER PLACEMENT (v2)
# ============================================================================

def dhan_exchange_segment(kind, exchange):
    """Instrument kind (EQ/FNO) + exchange (NSE/BSE) → Dhan exchange segment."""
    exchange = "BSE" if str(exchange).upper().startswith("BSE") or str(exchange).upper() == "BSE" else "NSE"
    return f"{exchange}_{'EQ' if kind == 'EQ' else 'FNO'}"


def place_dhan_order(client_id, access_token, security_id, txn_type, product_cfg, qty,
                     price=0.0, order_type="MARKET"):
    """
    REAL Dhan Broker API (v2) order call: POST {DHAN_API_BASE}/orders.

    Safety behavior:
      • Without an access token, NOTHING is sent — the full payload is
        returned with status SIMULATED_NOT_SENT (safe dry-run, displayed).
      • Network/API failures return status ERROR without crashing the app.
      • With "Use Broker SL/Target (Bracket Order)" enabled, entries go out
        as productType "BO" carrying boProfitValue / boStopLossValue (and
        trailingJump when > 0) so the broker manages the exit legs itself.
    """
    order_type = (order_type or "MARKET").upper()
    payload = {
        "dhanClientId": str(client_id or ""),
        "transactionType": txn_type,
        "exchangeSegment": product_cfg.get("exchange_segment"),
        "productType": product_cfg.get("product", "INTRADAY"),
        "orderType": order_type,
        "validity": "DAY",
        "securityId": str(security_id or ""),
        "quantity": int(qty),
        "disclosedQuantity": 0,
        "price": float(price) if order_type == "LIMIT" else 0.0,
        "afterMarketOrder": False,
    }
    if product_cfg.get("bo_enabled") and product_cfg.get("is_entry", True):
        payload["productType"] = "BO"
        payload["boProfitValue"] = float(product_cfg.get("bo_target_points", 0.0))
        payload["boStopLossValue"] = float(product_cfg.get("bo_sl_points", 0.0))
        trail = float(product_cfg.get("bo_trail_jump", 0.0))
        if trail > 0:
            payload["trailingJump"] = trail

    token = str(access_token or "").strip()
    if not token:
        return {"status": "SIMULATED_NOT_SENT", "payload": payload,
                "note": "No Dhan access token provided — payload shown as a safe dry-run, nothing was sent."}

    try:
        resp = requests.post(
            f"{DHAN_API_BASE}/orders",
            headers={"access-token": token, "client-id": str(client_id or ""),
                     "Content-Type": "application/json"},
            json=payload, timeout=15,
        )
        try:
            body = resp.json()
        except ValueError:
            body = {"raw": resp.text[:500]}
        if resp.status_code in (200, 201, 202):
            return {"status": "SENT", "http_status": resp.status_code,
                    "payload": payload, "response": body}
        return {"status": "ERROR", "http_status": resp.status_code,
                "payload": payload, "response": body,
                "note": "Dhan API rejected the order — see response. The app keeps running."}
    except Exception as exc:
        return {"status": "ERROR", "payload": payload, "error": str(exc),
                "note": "Network/API failure — order not confirmed. The app keeps running."}


def dhan_proportional_qty(dhan_qty, paper_qty_closed, paper_total_qty):
    """Partial books send a PROPORTIONAL share of the Dhan quantity."""
    try:
        if paper_total_qty <= 0:
            return int(dhan_qty)
        share = int(round(int(dhan_qty) * float(paper_qty_closed) / float(paper_total_qty)))
        return max(1, min(share, int(dhan_qty)))
    except Exception:
        return int(dhan_qty)


def dispatch_dhan_event(cfg, direction, is_entry, event_label, paper_qty, paper_total_qty,
                        price, exit_reason=None):
    """
    Single choke-point every live entry/exit/partial/square-off goes through.

    Options direction rule (all modes, including flipped signals):
        LONG signal → BUY the CE leg · SHORT signal → BUY the PE leg
        Exits SELL whichever leg is open. Options are always BOUGHT, never sold.
    Stocks/futures: long = BUY entry / SELL exit, short = SELL entry / BUY exit.

    Bracket orders: when BO is ON and this exit reason is "Stoploss Hit" /
    "Target Hit", the app SKIPS its own exit order — the broker's BO legs
    already closed the position (avoids double exits). Signal exits and
    manual square-offs are still sent.
    """
    if not cfg.get("dhan_enabled"):
        return None
    product_cfg = dict(cfg.get("product_cfg") or {})

    if (not is_entry and product_cfg.get("bo_enabled") and exit_reason
            and ("Stoploss Hit" in str(exit_reason) or "Target Hit" in str(exit_reason))):
        return {"status": "SKIPPED_BO_MANAGED",
                "note": f"Bracket Order active — broker legs already handled '{exit_reason}'; "
                        "own exit order deliberately skipped to avoid a double exit."}

    security_id, txn_type = resolve_dhan_order_leg(direction, is_entry, cfg.get("ticker"), product_cfg)
    order_type = cfg.get("entry_order_type", "MARKET") if is_entry else cfg.get("exit_order_type", "MARKET")

    dhan_qty = int(cfg.get("dhan_qty", 1) or 1)
    send_qty = dhan_proportional_qty(dhan_qty, paper_qty, paper_total_qty)

    product_cfg["is_entry"] = is_entry
    result = place_dhan_order(cfg.get("dhan_client_id"), cfg.get("dhan_access_token"),
                              security_id, txn_type, product_cfg, send_qty,
                              price=price or 0.0, order_type=order_type)
    if isinstance(result, dict):
        result["event"] = event_label
    return result


# ============================================================================
# EMAIL NOTIFICATIONS (Gmail SMTP, SSL 465) — a mail failure NEVER blocks
# trading, it only surfaces a warning.
# ============================================================================

def send_trade_email(cfg, subject, body_lines):
    if not cfg.get("email_enabled"):
        return
    sender = str(cfg.get("email_from") or "").strip()
    recipients = [r.strip() for r in str(cfg.get("email_to") or "").split(",") if r.strip()]
    app_password = str(cfg.get("email_app_password") or "").strip()
    if not (sender and recipients and app_password):
        st.warning("📧 Email notification skipped — From/To/App Password not fully configured.")
        return
    try:
        msg = MIMEText("\n".join(str(x) for x in body_lines))
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=15) as server:
            server.login(sender, app_password)
            server.sendmail(sender, recipients, msg.as_string())
    except Exception as exc:
        st.warning(f"📧 Email notification failed (trading unaffected): {exc}")


def email_trade_event(cfg, event, details):
    """entry / exit / partial book / manual square-off notification containing
    strategy, entry, SL, target, exit reason, points and PnL."""
    lines = [f"AlgoTrader Pro — {event}", "-" * 40]
    for k, v in details.items():
        lines.append(f"{k}: {v}")
    lines.append("-" * 40)
    lines.append(f"Time (IST): {ist_now().strftime('%Y-%m-%d %H:%M:%S')}")
    send_trade_email(cfg, f"[AlgoTrader] {event} — {details.get('Ticker', '')}", lines)


# ============================================================================
# RISK CONTROL GATES (live-trading only; all disabled by default)
# ============================================================================

def _today_realized_points():
    """Sum of realized points from live history rows whose exit date is
    TODAY in IST — the basis for daily loss/profit limits."""
    today = ist_now().date()
    total = 0.0
    for row in st.session_state.get("live_history", []):
        et = row.get("Exit Time")
        try:
            d = et.date() if hasattr(et, "date") else pd.to_datetime(et).date()
        except Exception:
            continue
        if d == today:
            total += float(row.get("Points", 0.0) or 0.0)
    return total


def check_entry_gates(gates, ticker_choice, ticker):
    """Returns (allowed, reason). Blocked entries display the specific gate
    reason on the Live tab. Daily counters reset at IST date change."""
    gates = gates or {}
    now = ist_now()

    # Daily counter reset on IST date change
    day_key = now.strftime("%Y-%m-%d")
    if st.session_state.risk_day_key != day_key:
        st.session_state.risk_day_key = day_key
        st.session_state.risk_day_entries = 0

    if gates.get("max_day_loss_enabled"):
        limit = float(gates.get("max_day_loss_points", 20.0))
        realized = _today_realized_points()
        if realized <= -limit:
            return False, f"Max Points Loss in a Day hit (realized {realized:+.2f} ≤ −{limit:.0f} pts) — new entries blocked for today."

    if gates.get("max_day_profit_enabled"):
        limit = float(gates.get("max_day_profit_points", 100.0))
        realized = _today_realized_points()
        if realized >= limit:
            return False, f"Max Points Profit in a Day reached (realized {realized:+.2f} ≥ +{limit:.0f} pts) — trading stopped for today."

    if gates.get("max_day_trades_enabled"):
        limit = int(gates.get("max_day_trades", 10))
        if st.session_state.risk_day_entries >= limit:
            return False, f"Max Number of Trades in a Day reached ({st.session_state.risk_day_entries}/{limit}) — entries blocked."

    if gates.get("trade_window_enabled") and is_indian_ticker(ticker_choice, ticker):
        start_t = gates.get("trade_window_start", dtime(9, 15))
        end_t = gates.get("trade_window_end", dtime(15, 30))
        now_t = now.time()
        if not (start_t <= now_t <= end_t):
            return False, (f"Outside Trade Window ({start_t.strftime('%H:%M')}–{end_t.strftime('%H:%M')} IST) "
                           "— enforced only for Indian tickers; non-Indian tickers trade all 24 hours.")

    if gates.get("cooldown_enabled"):
        cooldown = float(gates.get("cooldown_seconds", 1.0))
        elapsed = time.time() - float(st.session_state.get("risk_last_event_ts", 0.0) or 0.0)
        if elapsed < cooldown:
            return False, f"Entry Cooldown active — {cooldown - elapsed:.1f}s of {cooldown:.0f}s remaining after the last entry/exit event."

    return True, None


def note_trade_event(entered=False):
    """Record entry counts + the timestamp any entry/exit event happened
    (feeds the max-trades gate and the entry cooldown)."""
    if entered:
        st.session_state.risk_day_entries = int(st.session_state.get("risk_day_entries", 0)) + 1
    st.session_state.risk_last_event_ts = time.time()


def check_profitable_hold_exit(gates, pos, ltp, now=None):
    """'Max Hold Duration of Profitable Trade': if the open position has been
    held ≥ N minutes AND is currently in profit → exit immediately."""
    if not (gates or {}).get("profit_hold_enabled"):
        return False, None, None
    try:
        max_min = float(gates.get("profit_hold_minutes", 1.0))
        entry_time = pos.get("entry_time")
        entry_dt = entry_time if hasattr(entry_time, "tzinfo") else pd.to_datetime(entry_time)
        now = now or ist_now()
        entry_cmp = pd.Timestamp(entry_dt)
        now_cmp = pd.Timestamp(now)
        if entry_cmp.tzinfo is not None:
            entry_cmp = entry_cmp.tz_convert("Asia/Kolkata").tz_localize(None)
        if now_cmp.tzinfo is not None:
            now_cmp = now_cmp.tz_convert("Asia/Kolkata").tz_localize(None)
        held_min = (now_cmp - entry_cmp).total_seconds() / 60.0
        points_now = (ltp - pos["entry_price"]) * pos["direction"]
        if held_min >= max_min and points_now > 0:
            return True, float(ltp), (f"Max Profitable Hold Duration ({held_min:.1f}m ≥ {max_min:.0f}m, "
                                      f"in profit {points_now:+.2f} pts)")
    except Exception:
        pass
    return False, None, None


def resolve_dhan_order_leg(direction, is_entry, fallback_ticker, product_cfg):
    """
    Decides WHICH instrument to trade and which side (BUY/SELL) to send.

    If "Auto-select CE/PE by signal direction" is on and both security IDs are
    filled in: a LONG signal buys the CE leg, a SHORT signal buys the PE leg —
    both are entered by BUYING (not selling) an option, which keeps risk
    defined (no naked option writing baked into this default). Exiting always
    SELLs whichever leg is currently open.

    Otherwise, falls back to trading the underlying ticker directly: BUY to
    open long / SELL to close it, SELL to open short / BUY to close it.
    """
    is_options = "Options" in str(product_cfg.get("instrument", "")) or product_cfg.get("options_mode")
    use_ce_pe = (
        (is_options or product_cfg.get("auto_ce_pe"))
        and product_cfg.get("ce_security_id")
        and product_cfg.get("pe_security_id")
    )
    if use_ce_pe:
        # Applies in ALL modes, including flipped signals: by the time the
        # direction reaches here the flip has already happened, so a flipped
        # long simply BUYs the other (PE) leg automatically.
        security_id = product_cfg["ce_security_id"] if direction == 1 else product_cfg["pe_security_id"]
        txn_type = "BUY" if is_entry else "SELL"   # options always BOUGHT, never sold short
        return security_id, txn_type

    # Stocks / futures: the Security ID box value is used when present
    # (scrip-master lookup already auto-filled it); the raw ticker is only a
    # last-resort fallback.
    security_id = product_cfg.get("security_id") or fallback_ticker
    if is_entry:
        txn_type = "BUY" if direction == 1 else "SELL"
    else:
        txn_type = "SELL" if direction == 1 else "BUY"
    return security_id, txn_type


# ============================================================================
# CONFIGURATION CONTROLS (shared by Sidebar AND the "🛠 Admin Panel" tab)
# ----------------------------------------------------------------------------
# render_config_controls(ui, prefix) renders EVERY control through the
# two-way-synced cfg_* wrappers above. The sidebar calls it with
# (st.sidebar, "sb"); the Admin Panel tab calls it full-width with
# (container, "ad"). Both are live views of st.session_state.app_cfg —
# changing a value in either place updates the other instantly.
# The old sidebar_overrides mechanism is removed; the Optimization tab's
# "apply config" writes directly into this shared store.
# ============================================================================

def _underlying_for_fno(ticker_choice_v, ticker_v):
    """Scrip-master underlying symbol for F&O lookups."""
    if ticker_choice_v in DHAN_INDEX_MAP:
        return DHAN_INDEX_MAP[ticker_choice_v]["underlying"]
    return _yf_symbol_to_plain(ticker_v or "")


def _current_underlying_ltp(ticker_v):
    """Best-effort current price of the underlying used to compute ATM
    strikes (live LTP rounded to the nearest real strike)."""
    try:
        d = fetch_data(ticker_v, "1m", "1d")
        if d is not None and not d.empty:
            return float(d["Close"].iloc[-1])
        d = fetch_data(ticker_v, "1d", "1mo")
        if d is not None and not d.empty:
            return float(d["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _try_autofill(sig, fetch_fn, sig_key, try_key):
    """Autofill reliability contract:
      • a ticker/instrument/exchange (signature) change ALWAYS overwrites
        stale Security IDs — stale values are cleared on the first attempt
        for a new signature;
      • a failed fetch RETRIES (throttled to every 20s) instead of
        permanently giving up;
      • the signature is only locked in ON SUCCESS."""
    if st.session_state.get(sig_key) == sig:
        return  # already succeeded for this exact signature
    attempted_key = "_attempted_" + sig_key
    now = time.time()
    same_sig_retry = st.session_state.get(attempted_key) == sig
    if same_sig_retry and (now - float(st.session_state.get(try_key, 0.0) or 0.0)) < 20.0:
        return  # throttle retries to every 20s
    st.session_state[try_key] = now
    st.session_state[attempted_key] = sig
    if fetch_fn():
        st.session_state[sig_key] = sig  # lock in ONLY on success


def render_config_controls(ui, prefix):
    """Renders the full control set into `ui` (sidebar or Admin Panel) and
    returns the assembled config dict. All original controls, defaults, and
    captions are preserved — rendered through the synced store."""
    store = _cfg_store()
    ui.title("⚙️ Algo Configuration" if prefix == "sb" else "🛠 Admin Panel — full configuration")
    if prefix == "ad":
        ui.caption("Every sidebar control, full-width. This panel and the sidebar are two live views of the "
                   "same shared config store — change a value in either place and the other updates instantly.")

    # ------------------------------------------------------------ TICKER --
    ticker_names = list(TICKER_MAP.keys())
    ticker_choice = cfg_selectbox(ui, "Ticker", "ticker_choice", ticker_names, default="Nifty50", prefix=prefix)

    options_mode = ticker_choice == "Options Trading"
    premium_mode = False
    if options_mode:
        # ---- OPTIONS TRADING mode with three sub-modes:
        #  • Index / Stocks — the main algorithm runs on the UNDERLYING's
        #    candles; a LONG signal buys the CE leg, a SHORT signal buys the
        #    PE leg (both legs configured below).
        #  • Premium — you pick ONE leg (CE or PE + strike); the strategy
        #    runs directly on that option's OWN premium candles from Dhan.
        #    LONG signal on the premium → BUY that leg. SHORT signal → NO
        #    position is entered (options are only ever bought here), though
        #    an opposite signal can still exit an open long via
        #    'Strategy Signal Exit'.
        opt_underlying_kind = cfg_selectbox(ui, "Options Underlying", "opt_underlying_kind",
                                            ["Index", "Stocks", "Premium"], default="Index", prefix=prefix)
        premium_mode = opt_underlying_kind == "Premium"
        if opt_underlying_kind == "Index":
            opt_index = cfg_selectbox(ui, "Index", "opt_index", list(DHAN_INDEX_MAP.keys()),
                                      default="Nifty50", prefix=prefix)
            ticker = TICKER_MAP[opt_index]
            underlying_choice = opt_index
        elif opt_underlying_kind == "Stocks":
            opt_stock = cfg_text(ui, "Stock symbol (NSE, e.g. RELIANCE)", "opt_stock", "RELIANCE", prefix=prefix)
            ticker = f"{_yf_symbol_to_plain(opt_stock)}.NS"
            underlying_choice = "Custom"
        else:
            # ---------------- PREMIUM TRADING ----------------
            ui.markdown("#### 🎯 Premium Trading — trade the option's own candles")
            prem_u = cfg_selectbox(ui, "Premium Underlying", "prem_underlying",
                                   ["Nifty50", "BankNifty", "Sensex", "Custom Stock"],
                                   default="Nifty50", prefix=prefix)
            if prem_u == "Custom Stock":
                prem_stock = cfg_text(ui, "Stock symbol (NSE, e.g. RELIANCE)", "prem_stock", "RELIANCE", prefix=prefix)
                prem_underlying_sym = _yf_symbol_to_plain(prem_stock)
                prem_underlying_yf = f"{prem_underlying_sym}.NS"
                prem_exchange, prem_instr, prem_scrip_instr = "NSE", "OPTSTK", "OPTSTK"
                prem_default_qty = None
            else:
                prem_meta = DHAN_INDEX_MAP[prem_u]
                prem_underlying_sym = prem_meta["underlying"]
                prem_underlying_yf = TICKER_MAP[prem_u]
                prem_exchange = prem_meta["exchange"]
                prem_instr, prem_scrip_instr = "OPTIDX", "OPTIDX"
                prem_default_qty = prem_meta["default_opt_qty"]

            prem_opt_type = cfg_selectbox(ui, "Option Type (CE or PE — this exact leg is traded)",
                                          "prem_opt_type", ["CE", "PE"], default="CE", prefix=prefix)

            prem_expiries = dhan_get_expiries(prem_underlying_sym, prem_scrip_instr, prem_exchange)
            if prem_expiries:
                prem_expiry = cfg_selectbox(ui, "Expiry Date (auto-fetched, nearest pre-selected)",
                                            "prem_expiry", prem_expiries, default=prem_expiries[0], prefix=prefix)
            else:
                prem_expiry = cfg_text(ui, "Expiry (YYYY-MM-DD — auto-fetch unavailable, enter manually)",
                                       "prem_expiry_manual", "", prefix=prefix)

            prem_strikes = dhan_get_strikes(prem_underlying_sym, prem_expiry, prem_scrip_instr, prem_exchange) if prem_expiry else []
            if prem_strikes:
                prem_atm = round_to_nearest_strike(_current_underlying_ltp(prem_underlying_yf), prem_strikes)
                prem_strike = cfg_selectbox(ui, "Strike (ATM pre-selected)", "prem_strike", prem_strikes,
                                            default=prem_atm if prem_atm in prem_strikes else prem_strikes[len(prem_strikes) // 2],
                                            prefix=prefix)
            else:
                prem_strike = cfg_number(ui, "Strike (strike list unavailable — manual)", "prem_strike_manual",
                                         0.0, 0.0, 10000000.0, prefix=prefix)

            # Auto-fill the single leg's Security ID (same reliability rules:
            # signature change always overwrites, failures retry every 20s,
            # signature locks only on success). The box stays editable.
            prem_sig = ("PREM", prem_underlying_sym, prem_exchange, prem_expiry, prem_opt_type, prem_strike)

            def _fetch_prem_id():
                info = dhan_lookup_option(prem_underlying_sym, prem_expiry, prem_strike, prem_opt_type,
                                          prem_scrip_instr, prem_exchange) if (prem_expiry and prem_strike) else None
                if info:
                    store["prem_security_id"] = info["security_id"]
                    store["_prem_lot_size"] = info.get("lot_size")
                    return True
                return False

            if st.session_state.get("dhan_opt_autofill_sig") != prem_sig \
                    and st.session_state.get("_attempted_dhan_opt_autofill_sig") != prem_sig:
                store["prem_security_id"] = ""      # sig change ALWAYS clears stale IDs
            _try_autofill(prem_sig, _fetch_prem_id, "dhan_opt_autofill_sig", "dhan_opt_autofill_last_try")

            prem_id = cfg_text(ui, f"{prem_opt_type} Security ID (auto-filled, editable)",
                               "prem_security_id", "", prefix=prefix).strip()
            prem_segment = f"{prem_exchange}_FNO"
            # Sentinel ticker → fetch_data / get_live_ltp serve THIS option's
            # premium candles + premium LTP straight from Dhan (no delay).
            ticker = f"DHANOPT::{prem_segment}::{prem_id}::{prem_instr}"
            underlying_choice = "Options Trading"

            if store.get("_prem_qty_default_sig") != (prem_u, prem_expiry, prem_opt_type):
                store["dhan_qty"] = int(prem_default_qty or store.get("_prem_lot_size") or 1)
                store["_prem_qty_default_sig"] = (prem_u, prem_expiry, prem_opt_type)

            if not prem_id:
                ui.warning("Waiting for the option's Security ID (auto-fill in progress, or enter it manually) — "
                           "no premium data can load until it's set.")
            ui.caption(f"⚡ The selected strategy runs on this {prem_opt_type}'s premium candles. "
                       f"LONG signal → BUY the {prem_opt_type}. SHORT signal → NO entry (you're only an options "
                       "buyer), but an opposite signal can still EXIT an open position when 'Strategy Signal Exit' "
                       "is the SL/Target type.")
        ui.caption("🔐 Dhan Client ID / Access Token for options data & orders are entered once in the "
                   "'🔐 Dhan Account' section below — one set serves both the feed and order placement.")
    elif ticker_choice == "Custom":
        ticker = cfg_text(ui, "Custom Ticker (Yahoo Finance symbol)", "ticker_custom", "KAYNES.NS", prefix=prefix)
        underlying_choice = ticker_choice
    else:
        ticker = TICKER_MAP[ticker_choice]
        underlying_choice = ticker_choice

    intervals = list(TF_PERIOD_MAP.keys())
    interval = cfg_selectbox(ui, "Timeframe", "interval", intervals, default="1m", prefix=prefix)
    periods_available = TF_PERIOD_MAP[interval]
    _default_period = "7d" if "7d" in periods_available else periods_available[0]
    period = cfg_selectbox(ui, "Period", "period", periods_available, default=_default_period, prefix=prefix)

    qty = cfg_number(ui, "Quantity", "qty", 1, min_value=1, step=1, is_int=True, prefix=prefix)

    # ---------------------------------------------------------- STRATEGY --
    ui.markdown("### 📐 Strategy")
    strategy = cfg_selectbox(ui, "Strategy", "strategy", STRATEGIES, default=STRATEGIES[0], prefix=prefix)

    params = {}
    params["ema_fast"] = int(store.get("ema_fast", 9))
    params["ema_slow"] = int(store.get("ema_slow", 15))

    if strategy in ("EMA Crossover", "Pro: EMA50 Trend + EMA9/15 Pullback"):
        params["ema_fast"] = cfg_number(ui, "EMA Fast", "ema_fast", 9, 2, 100, is_int=True, prefix=prefix)
        params["ema_slow"] = cfg_number(ui, "EMA Slow", "ema_slow", 15, 3, 200, is_int=True, prefix=prefix)
    if strategy == "Threshold Cross":
        params["threshold"] = cfg_number(ui, "Threshold Price", "threshold", 0.0, prefix=prefix)
        params["threshold_direction"] = cfg_selectbox(
            ui, "Cross Direction", "threshold_direction", ["Below", "Above"], default="Below", prefix=prefix)
        ui.caption("Below = price approaches from below and crosses UP through the threshold → LONG. "
                   "Above = price crosses DOWN from above → SHORT. Applied identically in backtest and live.")
    if strategy == "Price Action Support/Resistance":
        params["sr_window"] = cfg_number(ui, "S/R Lookback", "sr_window", 20, 5, 200, is_int=True, prefix=prefix)
    if strategy == "Liquidity Grab Reversal":
        params["liq_window"] = cfg_number(ui, "Liquidity Lookback", "liq_window", 20, 5, 200, is_int=True, prefix=prefix)
    if strategy == "RSI Cross":
        params["rsi_period"] = cfg_number(ui, "RSI Period", "rsi_period", 14, 2, 50, is_int=True, prefix=prefix)
    if strategy in ("Bollinger Bands", "Pro: BB+RSI Mean Reversion (ATR filtered)"):
        params["bb_period"] = cfg_number(ui, "BB Period", "bb_period", 20, 5, 100, is_int=True, prefix=prefix)
        params["bb_std"] = cfg_number(ui, "BB Std Dev", "bb_std", 2.0, 1.0, 4.0, prefix=prefix)
    if strategy == "Volume Breakout":
        params["vol_window"] = cfg_number(ui, "Volume Lookback", "vol_window", 20, 5, 100, is_int=True, prefix=prefix)
        params["vol_factor"] = cfg_number(ui, "Volume Spike Factor", "vol_factor", 2.0, 1.0, 5.0, prefix=prefix)
    if strategy == "Elliott Wave (Zigzag)":
        params["zigzag_lookback"] = cfg_number(ui, "Zigzag Lookback", "zigzag_lookback", 3, 2, 20, is_int=True, prefix=prefix)
    if strategy == "Pro: VWAP + Supertrend Trend":
        params["st_period"] = cfg_number(ui, "Supertrend Period", "st_period", 10, 5, 50, is_int=True, prefix=prefix)
        params["st_mult"] = cfg_number(ui, "Supertrend Multiplier", "st_mult", 3.0, 1.0, 6.0, prefix=prefix)
    if strategy == "Pro: Opening Range Breakout + Volume":
        params["orb_candles"] = cfg_number(ui, "ORB Candles", "orb_candles", 5, 1, 30, is_int=True, prefix=prefix)
    if strategy == "Pro: MACD Crossover":
        c1, c2, c3 = ui.columns(3)
        params["macd_fast"] = cfg_number(c1, "MACD Fast", "macd_fast", 12, 2, 50, is_int=True, prefix=prefix)
        params["macd_slow"] = cfg_number(c2, "MACD Slow", "macd_slow", 26, 5, 100, is_int=True, prefix=prefix)
        params["macd_signal"] = cfg_number(c3, "MACD Signal", "macd_signal", 9, 2, 30, is_int=True, prefix=prefix)
    if strategy == "Pro: Donchian Channel Breakout":
        params["donchian_period"] = cfg_number(ui, "Donchian Period", "donchian_period", 20, 5, 100, is_int=True, prefix=prefix)
    if strategy == "Pro: Keltner Squeeze Breakout":
        c1, c2 = ui.columns(2)
        params["keltner_period"] = cfg_number(c1, "Keltner Period", "keltner_period", 20, 5, 50, is_int=True, prefix=prefix)
        params["keltner_atr_mult"] = cfg_number(c2, "Keltner ATR Mult", "keltner_atr_mult", 1.5, 0.5, 4.0, prefix=prefix)
    if strategy == "Pro: Stochastic Reversal":
        c1, c2 = ui.columns(2)
        params["stoch_k"] = cfg_number(c1, "Stochastic %K Period", "stoch_k", 14, 2, 50, is_int=True, prefix=prefix)
        params["stoch_d"] = cfg_number(c2, "Stochastic %D Period", "stoch_d", 3, 2, 20, is_int=True, prefix=prefix)
    if strategy == "Pro: TEMA Trend Flip":
        params["tema_period"] = cfg_number(ui, "TEMA Period", "tema_period", 20, 5, 100, is_int=True, prefix=prefix)
    if strategy == "Pro: CCI Extreme Reversal":
        params["cci_period"] = cfg_number(ui, "CCI Period", "cci_period", 20, 5, 100, is_int=True, prefix=prefix)

    if strategy in PRO_STRATEGIES:
        ui.caption("💡 Professional-grade composite strategy (trend/volatility/liquidity confluence). Not a guarantee of profitability — validate in the Optimization tab first.")
    if strategy in IMMEDIATE_EXECUTION_STRATEGIES:
        ui.caption("⚡ Immediate execution in LIVE trading: this strategy checks its condition against the live LTP "
                   "and enters IMMEDIATELY at LTP (no waiting for the next candle open) — it's a price condition, "
                   "not a candle-shape strategy.")

    # ------------------------------------------------ 🎚 TRADE DIRECTION --
    ui.markdown("### 🎚 Trade Direction")
    trade_direction = cfg_selectbox(ui, "Allowed direction", "trade_direction",
                                    ["Both", "Long Only", "Short Only"], default="Both", prefix=prefix)
    flip_signals = cfg_checkbox(ui, "Flip / Reverse Entries (Long ↔ Short)", "flip_signals", False, prefix=prefix)
    ui.caption("Applied centrally in signal generation, so backtest, optimization, heatmaps, and live all obey it. "
               "Flip happens FIRST, then the Trade Direction filter. Instrument mapping is preserved automatically "
               "(flipped signals in options just BUY the other leg).")
    params["trade_direction"] = trade_direction
    params["flip_signals"] = flip_signals
    # Premium mode is buyer-only: SHORT signals never OPEN a position (they
    # can still exit an open long via Strategy Signal Exit). Enforced at the
    # entry points of both the backtest engine and the live engine.
    params["long_entries_only"] = bool(premium_mode)
    if premium_mode:
        ui.caption("🎯 Premium mode note: SHORT entries are disabled here regardless of the Trade Direction "
                   "setting — as an options buyer you only ever BUY the selected leg on LONG signals.")

    # ---------------------------------------------------------- STOPLOSS --
    ui.markdown("### 🛑 Stoploss")
    sl_type = cfg_selectbox(ui, "Stoploss Type", "sl_type", SL_TYPES, default=SL_TYPES[0], prefix=prefix)
    _sl_explain = {
        "Custom Points": "Active SL = entry ∓ 'SL Points (base)'. Only that one fixed level exists and that's what hits.",
        "Trailing SL (Points)": "Initial SL = entry ∓ 'SL Points (base)'; it then trails, always staying that many points behind the best price reached. One level — the trailed one — is what hits.",
        "Trail Candle Low/High (Current)": "Initial SL = entry ∓ 'SL Points (base)' (a starting backstop). Every candle it RATCHETS to the current candle's low (longs) / high (shorts) whenever that is TIGHTER. There is only ONE active SL at any moment — the tighter of the two — and that single level is what hits.",
        "Trail Candle Low/High (Previous)": "Initial SL = entry ∓ 'SL Points (base)' (a starting backstop). Every candle it RATCHETS to the PREVIOUS candle's low/high whenever that is tighter. Only the single, current ratcheted level can hit.",
        "Trail Swing Low/High (Current)": "Initial SL = entry ∓ 'SL Points (base)' (a starting backstop). It then RATCHETS to the 10-bar swing low (longs) / swing high (shorts) including the current bar, whenever that is tighter. One active level — the ratcheted one — is what hits.",
        "Trail Swing Low/High (Previous)": "Initial SL = entry ∓ 'SL Points (base)' (a starting backstop). It then RATCHETS to the 10-bar swing low/high up to the PREVIOUS bar whenever that is tighter. One active level — the ratcheted one — is what hits.",
        "Strategy Signal Exit": "The exit fires when the strategy gives the REVERSE signal. 'SL Points (base)' still arms a hard backstop level at entry ∓ that many points, in case price runs before a reverse signal appears.",
        "EMA Reverse Crossover Exit": "The exit fires on the reverse EMA crossover. 'SL Points (base)' still arms a hard backstop level at entry ∓ that many points.",
        "ATR Based SL": "'SL Points (base)' is IGNORED for this type — SL = ATR × multiplier, trailed each candle.",
        "Risk:Reward Based (min 1:2)": "SL = entry ∓ 'SL Points (base)'; the target is then derived from it via the R:R ratio.",
        "Autopilot SL": "Initial SL distance = max(ATR × 1.2, 'SL Points (base)'), then adaptively tightens as profit builds.",
        "Loss Recovery SL (Give-back)": "The give-back recovery logic governs the exit; 'SL Points (base)' is IGNORED. A wide emergency backstop (max of 3×ATR and 1.5× the loss trigger) protects against gaps.",
    }
    if sl_type in _sl_explain:
        ui.caption("ℹ️ " + _sl_explain[sl_type])
    params["sl_points"] = cfg_number(ui, "SL Points (base)", "sl_points", 10.0, 0.1, 100000.0, prefix=prefix)
    if sl_type == "ATR Based SL":
        params["atr_mult_sl"] = cfg_number(ui, "ATR Multiplier (SL)", "atr_mult_sl", 1.5, 0.5, 5.0, prefix=prefix)
    if sl_type == "Loss Recovery SL (Give-back)":
        c1, c2 = ui.columns(2)
        params["loss_trigger_points"] = cfg_number(c1, "Loss trigger (points)", "loss_trigger_points", 20.0, 1.0, 100000.0, prefix=prefix)
        params["min_recovery_pct"] = cfg_number(c2, "Min recovery required (%)", "min_recovery_pct", 50.0, 1.0, 100.0, prefix=prefix)
        ui.caption(f"Once floating loss reaches {params['loss_trigger_points']:.0f} pts, exit if price hasn't recovered at least {params['min_recovery_pct']:.0f}% of that loss back toward entry.")

    # ------------------------------------------------------------ TARGET --
    ui.markdown("### 🎯 Target")
    target_type = cfg_selectbox(ui, "Target Type", "target_type", TARGET_TYPES, default=TARGET_TYPES[0], prefix=prefix)
    _tgt_explain = {
        "Custom Points": "Active target = entry ± 'Target Points (base)'. Only that fixed level exists and that's what hits.",
        "Trailing Target (Display Only)": "No fixed target hits — the position rides until the SL side (or a signal/risk exit) closes it. 'Target Points (base)' only sets the initial displayed level.",
        "Trail Candle Low/High (Current)": "Initial target = entry ± 'Target Points (base)'; it then EXTENDS with the current candle's high/low, so it keeps moving away — exits usually come from the SL side.",
        "Trail Candle Low/High (Previous)": "Initial target = entry ± 'Target Points (base)'; it then EXTENDS with the previous candle's high/low.",
        "Trail Swing Low/High (Current)": "Initial target = entry ± 'Target Points (base)'; it then EXTENDS to the 10-bar swing high/low including the current bar.",
        "Trail Swing Low/High (Previous)": "Initial target = entry ± 'Target Points (base)'; it then EXTENDS to the 10-bar swing high/low up to the previous bar.",
        "Strategy Signal Exit": "The exit fires on the strategy's REVERSE signal. 'Target Points (base)' still arms a hard take-profit level at entry ± that many points.",
        "EMA Reverse Crossover Exit": "The exit fires on the reverse EMA crossover. 'Target Points (base)' still arms a hard take-profit level.",
        "ATR Based Target": "'Target Points (base)' is IGNORED for this type — target = ATR × multiplier.",
        "Risk:Reward Based (min 1:2)": "'Target Points (base)' is IGNORED — target distance = SL distance × the R:R ratio.",
        "Autopilot Target": "Initial target distance = max(ATR × 2.5, 2× SL distance), then adaptively extends. 'Target Points (base)' is ignored.",
        "Profit Giveback Target": "The give-back logic governs the exit; 'Target Points (base)' is IGNORED. A wide backstop (max of 4×ATR and 1.5× the profit trigger) still exists.",
        "Partial Book + Trail Remainder": "'Target 1 (points)' below is the REAL actionable level for the first tranche; the remainder trails with no fixed second target.",
    }
    if target_type in _tgt_explain:
        ui.caption("ℹ️ " + _tgt_explain[target_type])
    params["target_points"] = cfg_number(ui, "Target Points (base)", "target_points", 20.0, 0.1, 200000.0, prefix=prefix)
    if target_type == "ATR Based Target":
        params["atr_mult_target"] = cfg_number(ui, "ATR Multiplier (Target)", "atr_mult_target", 3.0, 1.0, 8.0, prefix=prefix)
    if target_type == "Profit Giveback Target":
        c1, c2 = ui.columns(2)
        params["profit_trigger_points"] = cfg_number(c1, "Profit trigger (points)", "profit_trigger_points", 50.0, 1.0, 100000.0, prefix=prefix)
        params["giveback_pct"] = cfg_number(c2, "Max giveback allowed (%)", "giveback_pct", 30.0, 1.0, 100.0, prefix=prefix)
        ui.caption(f"Once floating profit peaks at ≥{params['profit_trigger_points']:.0f} pts, exit if it falls back by more than {params['giveback_pct']:.0f}% from that peak.")
    if target_type == "Partial Book + Trail Remainder":
        c1, c2 = ui.columns(2)
        params["partial_target1_points"] = cfg_number(c1, "Target 1 (points)", "partial_target1_points",
                                                      float(params.get("target_points", 20.0)), 0.1, 200000.0, prefix=prefix)
        params["partial_book_pct"] = cfg_number(c2, "Qty % to book at Target 1", "partial_book_pct", 50.0, 1.0, 99.0, prefix=prefix)
        ui.caption(
            f"Books {params['partial_book_pct']:.0f}% of quantity when Target 1 ({params['partial_target1_points']:.0f} pts) is hit; "
            "the remainder keeps running under an ATR trailing stop with no fixed second target. "
            "⚠️ With Quantity = 1, there's nothing left to trail after rounding — increase Quantity in the sidebar to actually see partial-booking behavior."
        )
    if sl_type == "Risk:Reward Based (min 1:2)" or target_type == "Risk:Reward Based (min 1:2)":
        params["rr_ratio"] = cfg_number(ui, "Risk:Reward Ratio (min 2)", "rr_ratio", 2.0, 2.0, 10.0, prefix=prefix)

    # ------------------------------------------- TIME-BASED RISK CONTROL --
    ui.markdown("### ⏱ Time-Based Risk Control")
    loss_duration_enabled = cfg_checkbox(ui, "Loss Holding Duration Exit", "loss_duration_enabled", False, prefix=prefix)
    loss_duration_min_minutes, loss_duration_max_minutes = 1.0, 5.0
    if loss_duration_enabled:
        c1, c2 = ui.columns(2)
        loss_duration_min_minutes = cfg_number(c1, "Min minutes in loss before acting", "loss_duration_min_minutes", 1.0, 0.0, step=1.0, prefix=prefix)
        loss_duration_max_minutes = cfg_number(c2, "Safety ceiling (minutes)", "loss_duration_max_minutes", 5.0, 0.0, step=1.0, prefix=prefix)
        ui.caption(
            "Exits as soon as the position has been continuously in a floating loss for at least the first number "
            "of minutes. The second number is just an upper safety bound (mainly relevant to live polling delays) — "
            "keep it ≥ the first. No cap is applied to how high you can set either value."
        )
    risk_ctrl = {
        "loss_duration_enabled": loss_duration_enabled,
        "loss_duration_min_minutes": loss_duration_min_minutes,
        "loss_duration_max_minutes": loss_duration_max_minutes,
    }

    # ------------------------------------------------------------ GATES  --
    ui.markdown("### 🚧 Risk Controls (Live-Trading Gates)")
    ui.caption("All disabled by default. These gate LIVE entries only; blocked entries display the specific "
               "gate reason on the Live tab. Daily counters reset at date change (IST).")
    gates = {}
    gates["max_day_loss_enabled"] = cfg_checkbox(ui, "Max Points Loss in a Day", "gate_day_loss_enabled", False, prefix=prefix)
    if gates["max_day_loss_enabled"]:
        gates["max_day_loss_points"] = cfg_number(ui, "Max loss (points)", "gate_day_loss_points", 20.0, 0.1, 1000000.0, prefix=prefix)
        ui.caption("Once today's realized points ≤ −limit, new entries are blocked for the day.")
    gates["max_day_profit_enabled"] = cfg_checkbox(ui, "Max Points Profit in a Day", "gate_day_profit_enabled", False, prefix=prefix)
    if gates["max_day_profit_enabled"]:
        gates["max_day_profit_points"] = cfg_number(ui, "Max profit (points)", "gate_day_profit_points", 100.0, 0.1, 1000000.0, prefix=prefix)
        ui.caption("Once today's realized points ≥ +limit, trading stops for the day.")
    gates["max_day_trades_enabled"] = cfg_checkbox(ui, "Max Number of Trades in a Day", "gate_day_trades_enabled", False, prefix=prefix)
    if gates["max_day_trades_enabled"]:
        gates["max_day_trades"] = cfg_number(ui, "Max entries per day", "gate_day_trades", 10, 1, 10000, is_int=True, prefix=prefix)
    gates["profit_hold_enabled"] = cfg_checkbox(ui, "Max Hold Duration of Profitable Trade", "gate_profit_hold_enabled", False, prefix=prefix)
    if gates["profit_hold_enabled"]:
        gates["profit_hold_minutes"] = cfg_number(ui, "Exit profitable trade after (minutes)", "gate_profit_hold_minutes", 1.0, 0.1, 100000.0, prefix=prefix)
        ui.caption("If the open position has been held at least this many minutes AND is currently in profit, it exits immediately.")
    gates["trade_window_enabled"] = cfg_checkbox(ui, "Trade Window (IST, Indian tickers only)", "gate_window_enabled", False, prefix=prefix)
    if gates["trade_window_enabled"]:
        c1, c2 = ui.columns(2)
        gates["trade_window_start"] = cfg_time(c1, "Window start (IST)", "gate_window_start", dtime(9, 15), prefix=prefix)
        gates["trade_window_end"] = cfg_time(c2, "Window end (IST)", "gate_window_end", dtime(15, 30), prefix=prefix)
        ui.caption("Entries only inside the window; enforced ONLY for Indian tickers (.NS/.BO/Nifty/BankNifty/Sensex). "
                   "All other tickers keep the full 24 hours — every hour and minute of the day.")
    gates["cooldown_enabled"] = cfg_checkbox(ui, "Enable Entry Cooldown", "gate_cooldown_enabled", False, prefix=prefix)
    if gates["cooldown_enabled"]:
        gates["cooldown_seconds"] = cfg_number(ui, "Cooldown (seconds)", "gate_cooldown_seconds", 1.0, 0.1, 86400.0, prefix=prefix)
        ui.caption("After any entry/exit event, block new entries for this many seconds.")

    # ---------------------------------------------------------- FILTERS  --
    ui.markdown("### 🔍 Additional Entry Filters")
    filters = {"adx_enabled": cfg_checkbox(ui, "ADX Filter", "adx_enabled", False, prefix=prefix)}
    if filters["adx_enabled"]:
        c1, c2 = ui.columns(2)
        filters["adx_min"] = cfg_number(c1, "ADX Min", "adx_min", 20, 0, 100, is_int=True, prefix=prefix)
        filters["adx_max"] = cfg_number(c2, "ADX Max", "adx_max", 100, 0, 100, is_int=True, prefix=prefix)
    filters["rsi_enabled"] = cfg_checkbox(ui, "RSI Filter (30 up-cross buy / 70 down-cross sell)", "rsi_enabled", False, prefix=prefix)
    filters["bb_enabled"] = cfg_checkbox(ui, "Bollinger Band Filter", "bb_enabled", False, prefix=prefix)
    filters["ema20_enabled"] = cfg_checkbox(ui, "EMA20 Filter", "ema20_enabled", False, prefix=prefix)
    filters["sma20_enabled"] = cfg_checkbox(ui, "SMA20 Filter", "sma20_enabled", False, prefix=prefix)
    filters["smc_enabled"] = cfg_checkbox(ui, "SMC (Structure Break) Filter", "smc_enabled", False, prefix=prefix)

    filters["atr_enabled"] = cfg_checkbox(ui, "ATR (Volatility) Filter", "atr_enabled", False, prefix=prefix)
    if filters["atr_enabled"]:
        c1, c2 = ui.columns(2)
        filters["atr_min"] = cfg_number(c1, "ATR Min (points)", "atr_min", 0.0, 0.0, 100000.0, prefix=prefix)
        filters["atr_max"] = cfg_number(c2, "ATR Max (points)", "atr_max", 100000.0, 0.0, 100000.0, prefix=prefix)
        ui.caption("Only trade when 14-period ATR is inside this band — avoids dead/illiquid tape and blow-off volatility spikes.")

    filters["supertrend_enabled"] = cfg_checkbox(ui, "Supertrend Filter", "supertrend_enabled", False, prefix=prefix)
    if filters["supertrend_enabled"]:
        c1, c2 = ui.columns(2)
        filters["st_filter_period"] = cfg_number(c1, "Supertrend Period (filter)", "st_filter_period", 10, 5, 50, is_int=True, prefix=prefix)
        filters["st_filter_mult"] = cfg_number(c2, "Supertrend Mult (filter)", "st_filter_mult", 3.0, 1.0, 6.0, prefix=prefix)
        ui.caption("Only takes buys when Supertrend is bullish, sells when Supertrend is bearish — independent of the main strategy.")

    filters["regime_enabled"] = cfg_checkbox(ui, "Regime Filter (Trend vs Range, adaptive)", "regime_enabled", False, prefix=prefix)
    if filters["regime_enabled"]:
        c1, c2 = ui.columns(2)
        filters["regime_trend_min"] = cfg_number(c1, "ADX ≥ this = Trending", "regime_trend_min", 25, 10, 60, is_int=True, prefix=prefix)
        filters["regime_range_max"] = cfg_number(c2, "ADX ≤ this = Ranging", "regime_range_max", 20, 5, 40, is_int=True, prefix=prefix)
        ui.caption(
            "Trend-type strategies (EMA/Supertrend/ORB/S-R/EW) only fire when ADX confirms a trend; "
            "mean-reversion strategies (RSI/Bollinger/Liquidity/BB+RSI) only fire when ADX confirms a range. "
            "This is the 'adapt to changing market regime' control — it doesn't switch strategies for you, "
            "it stops your chosen strategy from firing in the regime it's known to perform badly in."
        )

    filters["angle_enabled"] = cfg_checkbox(ui, "Angle of Crossover Filter", "angle_enabled", False, prefix=prefix)
    if filters["angle_enabled"]:
        filters["angle_min_deg"] = cfg_number(ui, "Minimum crossover angle (degrees, absolute value)",
                                              "angle_min_deg", 0.0, 0.0, step=1.0, prefix=prefix)
        ui.caption(
            f"Only accepts an EMA{params.get('ema_fast',9)}/{params.get('ema_slow',15)} crossover if it's steep enough. "
            "Angle is normalized against ATR (there's no universal 'degrees' for a raw price slope), so treat it as a "
            "relative steepness score, not a standardized industry figure. Absolute value is used since valid crosses "
            "can produce a negative raw slope depending on direction."
        )

    filters["crossover_quality_enabled"] = cfg_checkbox(ui, "Crossover Confirmation Filter", "crossover_quality_enabled", False, prefix=prefix)
    if filters["crossover_quality_enabled"]:
        filters["crossover_quality_mode"] = cfg_selectbox(
            ui, "Confirmation type", "crossover_quality_mode",
            ["Simple Crossover", "Crossover with Candle Size", "Crossover with ATR-based Candle Size"], prefix=prefix)
        if filters["crossover_quality_mode"] == "Crossover with Candle Size":
            filters["crossover_min_points"] = cfg_number(ui, "Min candle range (points)", "crossover_min_points", 1.0, 0.0, step=0.5, prefix=prefix)
        elif filters["crossover_quality_mode"] == "Crossover with ATR-based Candle Size":
            filters["crossover_atr_mult"] = cfg_number(ui, "Min candle range (× ATR)", "crossover_atr_mult", 1.0, 0.1, step=0.1, prefix=prefix)
        ui.caption(f"Only accepts an EMA{params.get('ema_fast',9)}/{params.get('ema_slow',15)} crossover bar that also clears this candle-size bar — filters out crosses on tiny, indecisive candles.")

    filters["vix_enabled"] = cfg_checkbox(ui, "India VIX Filter", "vix_enabled", False, prefix=prefix)
    if filters["vix_enabled"]:
        c1, c2 = ui.columns(2)
        filters["vix_min"] = cfg_number(c1, "VIX Min", "vix_min", 10.0, 0.0, 100.0, prefix=prefix)
        filters["vix_max"] = cfg_number(c2, "VIX Max", "vix_max", 25.0, 0.0, 100.0, prefix=prefix)
        ui.caption(
            "India VIX is a fear/expected-volatility gauge, not a price indicator — you don't need to be an expert to "
            "use it as a simple filter here. Rough rule of thumb: below ~15 = calm (often better for trend-following), "
            "15–20 = normal, 20–30 = elevated/nervous (often better for mean-reversion or smaller size), above ~30 = "
            "panic (many systems sit out entirely). Defaults above (10–25) are a conservative 'avoid extremes' band — "
            "adjust to taste. VIX only publishes daily, so intraday timeframes reuse the latest known daily value."
        )

    # ------------------------------------------------- SMART EVALUATION  --
    ui.markdown("### 🧠 Smart Evaluation (Recommended Before Going Live)")
    ui.caption("Off by default. Turn these on to get a more honest read on whether a config is likely to hold up out-of-sample and after real costs.")

    wf_enabled = cfg_checkbox(ui, "Enable Walk-Forward Validation", "wf_enabled", False, prefix=prefix)
    wf_folds = 5
    if wf_enabled:
        wf_folds = cfg_slider(ui, "Number of sequential out-of-sample folds", "wf_folds", 3, 20, 5, prefix=prefix)
        ui.caption("Splits the backtest period into N sequential chunks and checks whether the edge holds up across most of them, not just in aggregate.")

    cost_enabled = cfg_checkbox(ui, "Enable Realistic Cost Modeling", "cost_enabled", False, prefix=prefix)
    cost_cfg = {"slippage_points": 0.0, "spread_points": 0.0, "brokerage_flat": 0.0}
    if cost_enabled:
        cost_cfg["slippage_points"] = cfg_number(ui, "Slippage per trade (points)", "cost_slippage", 1.0, 0.0, 10000.0, prefix=prefix)
        cost_cfg["spread_points"] = cfg_number(ui, "Bid-Ask spread cost (points)", "cost_spread", 0.5, 0.0, 10000.0, prefix=prefix)
        cost_cfg["brokerage_flat"] = cfg_number(ui, "Brokerage per order leg (currency)", "cost_brokerage", 20.0, 0.0, 10000.0, prefix=prefix)
        ui.caption("Deducted from every trade: (slippage + spread) in points, plus brokerage charged twice per round trip (entry + exit).")

    # ------------------------------------------------------- DATA FEED   --
    ui.markdown("### 📡 Data Source")
    use_dhan_feed = cfg_checkbox(ui, "Use Dhan Data Feed (instead of yfinance)", "use_dhan_feed", False, prefix=prefix)
    if use_dhan_feed:
        ui.caption("Dhan serves candles (historical + intraday, IST) and live LTP with NO delay at all. Tickers Dhan "
                   "cannot serve (BTC-USD, ETH-USD, USDINR, gold/silver futures, …) automatically fall back to "
                   "yfinance with a notice on the Live tab. Feed ON without an access token silently stays on "
                   "yfinance and shows a warning.")
    else:
        ui.caption("yfinance path keeps its mandatory 0.3s delay per API call.")

    # -------------------------------------------------- ORDER PLACEMENT  --
    ui.markdown("### 🏦 Dhan Broker — Live Order Placement")
    dhan_enabled = cfg_checkbox(ui, "Enable Dhan Order Placement (LIVE)", "dhan_enabled", False, prefix=prefix)

    # -------- shared 🔐 Dhan Account credentials (one set serves both the
    # data feed and order placement) --------
    dhan_client_id, dhan_access_token = "", ""
    need_creds = use_dhan_feed or dhan_enabled or options_mode
    if need_creds:
        ui.markdown("#### 🔐 Dhan Account")
        dhan_client_id = cfg_text(ui, "Dhan Client ID", "dhan_client_id", DHAN_DEFAULT_CLIENT_ID, prefix=prefix)
        dhan_access_token = cfg_text(ui, "Dhan Access Token", "dhan_access_token", "", type="password", prefix=prefix)
        ui.caption("One set of credentials serves both the data feed and order placement.")
    else:
        dhan_client_id = str(store.get("dhan_client_id", "") or "")
        dhan_access_token = str(store.get("dhan_access_token", "") or "")

    product_cfg = {}
    entry_order_type, exit_order_type, dhan_qty = "MARKET", "MARKET", 1

    dhan_touchpoints_on = dhan_enabled or options_mode
    if dhan_touchpoints_on and premium_mode:
        # -------- PREMIUM MODE product config: the single selected leg. All
        # instrument details were already chosen in the 🎯 Premium Trading
        # section above; orders always BUY that leg on entry / SELL on exit.
        if dhan_enabled:
            ui.warning("Live orders will be attempted using the credentials above. Without a token, orders are only "
                       "SIMULATED (payload shown, nothing sent). Test in a sandbox first.")
        _p_exch = "BSE" if store.get("prem_underlying") == "Sensex" else "NSE"
        product_cfg = {
            "instrument": "Index Options" if store.get("prem_underlying", "Nifty50") != "Custom Stock" else "Stock Options",
            "exchange": _p_exch,
            "exchange_segment": f"{_p_exch}_FNO",
            "product": "MARGIN",
            "options_mode": True,
            "premium_mode": True,
            "security_id": str(store.get("prem_security_id", "") or "").strip(),
            "expiry": store.get("prem_expiry", store.get("prem_expiry_manual", "")),
            "opt_type": store.get("prem_opt_type", "CE"),
            "strike": store.get("prem_strike", store.get("prem_strike_manual")),
            "underlying": store.get("prem_underlying", "Nifty50"),
            "lot_size": store.get("_prem_lot_size"),
        }
        c1, c2 = ui.columns(2)
        entry_order_type = cfg_selectbox(c1, "Entry Order Type", "entry_order_type", ["MARKET", "LIMIT"], default="MARKET", prefix=prefix)
        exit_order_type = cfg_selectbox(c2, "Exit Order Type", "exit_order_type", ["MARKET", "LIMIT"], default="MARKET", prefix=prefix)
        dhan_qty = cfg_number(ui, "Dhan Quantity (real orders use this; paper P&L uses the paper Quantity above)",
                              "dhan_qty", 1, 1, 1000000, is_int=True, prefix=prefix)
        bo_enabled = cfg_checkbox(ui, "Use Broker SL/Target (Bracket Order)", "bo_enabled", False, prefix=prefix)
        product_cfg["bo_enabled"] = bo_enabled
        if bo_enabled:
            c1, c2, c3 = ui.columns(3)
            product_cfg["bo_sl_points"] = cfg_number(c1, "SL Points (boStopLossValue)", "bo_sl_points", 10.0, 0.1, 100000.0, prefix=prefix)
            product_cfg["bo_target_points"] = cfg_number(c2, "Target Points (boProfitValue)", "bo_target_points", 20.0, 0.1, 200000.0, prefix=prefix)
            product_cfg["bo_trail_jump"] = cfg_number(c3, "Trail SL Jump (0 = off)", "bo_trail_jump", 0.0, 0.0, 100000.0, prefix=prefix)
            ui.caption("Entries go out as productType \"BO\"; broker-managed Stoploss/Target hits skip the app's own "
                       "exit order to avoid double exits. Signal exits and manual square-offs are still sent.")
        if options_mode and not dhan_enabled:
            ui.info("📄 Premium Trading with Dhan Order Placement OFF = PAPER trading of the leg's premium. "
                    "Turn order placement ON to send REAL orders for the exact same leg.")
    elif dhan_touchpoints_on:
        if dhan_enabled:
            ui.warning("Live orders will be attempted using the credentials above. Without a token, orders are only "
                       "SIMULATED (payload shown, nothing sent). Test in a sandbox first.")

        # ---- Instrument dropdown (options mode pre-selects an Options type)
        default_instrument = "Index Options" if (options_mode and store.get("opt_underlying_kind", "Index") == "Index") \
            else ("Stock Options" if options_mode else "Stock Intraday")
        instrument_type = cfg_selectbox(ui, "Instrument", "dhan_instrument", DHAN_INSTRUMENT_CHOICES,
                                        default=default_instrument, prefix=prefix)
        meta = DHAN_INSTRUMENT_META[instrument_type]

        # ---- Exchange dropdown: auto-flips to BSE when Sensex or a .BO
        # ticker is selected, back to NSE otherwise; always user-editable.
        auto_exchange = "BSE" if (ticker_choice == "Sensex"
                                  or (ticker or "").endswith(".BO")
                                  or (options_mode and store.get("opt_index") == "Sensex")) else "NSE"
        if store.get("_last_auto_exchange") != auto_exchange:
            store["exchange"] = auto_exchange          # auto-flip writes to store…
            store["_last_auto_exchange"] = auto_exchange
        exchange = cfg_selectbox(ui, "Exchange", "exchange", ["NSE", "BSE"], default="NSE", prefix=prefix)  # …still user-editable

        underlying = _underlying_for_fno(underlying_choice if not options_mode else
                                         (store.get("opt_index", "Nifty50") if store.get("opt_underlying_kind", "Index") == "Index" else "Custom"),
                                         ticker)
        is_fno = meta["kind"] == "FNO"
        is_opts = "Options" in instrument_type
        creds_ok = bool(dhan_access_token) or bool(use_dhan_feed) or dhan_enabled

        # ---- Expiry (auto-fetched from the cached scrip master; nearest pre-selected)
        expiry = ""
        strikes = []
        lot_size = None
        if is_fno:
            expiries = dhan_get_expiries(underlying, meta["scrip_instrument"], exchange)
            if expiries:
                expiry = cfg_selectbox(ui, "Expiry Date (auto-fetched, nearest pre-selected)", "dhan_expiry",
                                       expiries, default=expiries[0], prefix=prefix)
            else:
                expiry = cfg_text(ui, "Expiry (YYYY-MM-DD — auto-fetch unavailable, enter manually)", "dhan_expiry_manual", "", prefix=prefix)

        # ---- Options: option type, ATM strikes from live LTP, CE/PE security IDs
        ce_strike = pe_strike = None
        if is_opts:
            opt_type_pref = cfg_selectbox(ui, "Option Type (CE/PE)", "opt_type_pref",
                                          ["Auto (CE on LONG / PE on SHORT)", "CE", "PE"],
                                          default="Auto (CE on LONG / PE on SHORT)", prefix=prefix)
            product_cfg["opt_type_pref"] = opt_type_pref

            if expiry:
                strikes = dhan_get_strikes(underlying, expiry, meta["scrip_instrument"], exchange)
            atm = None
            if strikes:
                ltp_u = _current_underlying_ltp(ticker)
                atm = round_to_nearest_strike(ltp_u, strikes)
            if strikes:
                ce_strike = cfg_selectbox(ui, "CE Strike (ATM pre-selected)", "ce_strike", strikes,
                                          default=atm if atm in strikes else strikes[len(strikes) // 2], prefix=prefix)
                pe_strike = cfg_selectbox(ui, "PE Strike (ATM pre-selected)", "pe_strike", strikes,
                                          default=atm if atm in strikes else strikes[len(strikes) // 2], prefix=prefix)
            else:
                ce_strike = cfg_number(ui, "CE Strike (strike list unavailable — manual)", "ce_strike_manual", 0.0, 0.0, 1000000.0, prefix=prefix)
                pe_strike = cfg_number(ui, "PE Strike (strike list unavailable — manual)", "pe_strike_manual", 0.0, 0.0, 1000000.0, prefix=prefix)

            # ---- CE/PE Security ID autofill — refreshes whenever expiry or
            # either strike changes; orders use the box values (scrip-master
            # lookup only as a fallback).
            sig = ("OPT", ticker, instrument_type, exchange, expiry, ce_strike, pe_strike)

            def _fetch_opt_ids():
                ok_any = False
                ce = dhan_lookup_option(underlying, expiry, ce_strike, "CE", meta["scrip_instrument"], exchange) if expiry and ce_strike else None
                pe = dhan_lookup_option(underlying, expiry, pe_strike, "PE", meta["scrip_instrument"], exchange) if expiry and pe_strike else None
                if ce:
                    store["ce_security_id"] = ce["security_id"]; ok_any = True
                    store["_opt_lot_size"] = ce.get("lot_size")
                if pe:
                    store["pe_security_id"] = pe["security_id"]; ok_any = True
                return bool(ce and pe) if (expiry and ce_strike and pe_strike) else ok_any

            if creds_ok or True:  # scrip master is public — autofill works even in paper mode
                if st.session_state.get("dhan_opt_autofill_sig") != sig:
                    # a signature change ALWAYS overwrites stale IDs
                    if st.session_state.get("_attempted_dhan_opt_autofill_sig") != sig:
                        store["ce_security_id"] = ""
                        store["pe_security_id"] = ""
                _try_autofill(sig, _fetch_opt_ids, "dhan_opt_autofill_sig", "dhan_opt_autofill_last_try")

            ce_id = cfg_text(ui, "CE Security ID (auto-filled, editable — used on LONG signals)", "ce_security_id", "", prefix=prefix)
            pe_id = cfg_text(ui, "PE Security ID (auto-filled, editable — used on SHORT signals)", "pe_security_id", "", prefix=prefix)
            product_cfg["ce_security_id"] = ce_id.strip()
            product_cfg["pe_security_id"] = pe_id.strip()
            lot_size = store.get("_opt_lot_size")

            # Default option quantities: NIFTY 65 · BANKNIFTY 35 · SENSEX 20;
            # stock options default to the contract lot size.
            if options_mode and store.get("opt_underlying_kind", "Index") == "Index":
                default_qty = DHAN_INDEX_MAP.get(store.get("opt_index", "Nifty50"), {}).get("default_opt_qty", 65)
            elif underlying in ("NIFTY", "BANKNIFTY", "SENSEX"):
                default_qty = {"NIFTY": 65, "BANKNIFTY": 35, "SENSEX": 20}[underlying]
            else:
                default_qty = int(lot_size or 1)
            if store.get("_opt_qty_default_sig") != (underlying, instrument_type):
                store["dhan_qty"] = int(default_qty)
                store["_opt_qty_default_sig"] = (underlying, instrument_type)

            ui.caption("Options direction rule (all modes, including flipped signals): LONG signal → BUY the CE leg; "
                       "SHORT signal → BUY the PE leg; exits SELL whichever leg is open. Options are always BOUGHT, "
                       "never sold.")

        elif is_fno:  # ------- futures: single security id, auto-filled
            sig = ("FUT", ticker, instrument_type, exchange, expiry)

            def _fetch_fut_id():
                info = dhan_lookup_future(underlying, expiry, meta["scrip_instrument"], exchange) if expiry else None
                if info:
                    store["dhan_security_id"] = info["security_id"]
                    if store.get("_fut_qty_default_sig") != sig:
                        store["dhan_qty"] = int(info.get("lot_size") or 1)   # futures default = contract lot size
                        store["_fut_qty_default_sig"] = sig
                    return True
                return False

            if st.session_state.get("dhan_autofill_sig") != sig:
                if st.session_state.get("_attempted_dhan_autofill_sig") != sig:
                    store["dhan_security_id"] = ""      # sig change ALWAYS clears stale IDs
            _try_autofill(sig, _fetch_fut_id, "dhan_autofill_sig", "dhan_autofill_last_try")

            sec_id = cfg_text(ui, "Security ID (always visible & mandatory — auto-filled when Dhan is enabled, "
                                  "manual entry in pure-yfinance mode)", "dhan_security_id", "", prefix=prefix)
            product_cfg["security_id"] = sec_id.strip()

        else:  # ----------------- equity: security id auto-filled
            sig = ("EQ", ticker, instrument_type, exchange)

            def _fetch_eq_id():
                info = dhan_lookup_equity(_yf_symbol_to_plain(ticker), exchange)
                if info:
                    store["dhan_security_id"] = info["security_id"]
                    return True
                return False

            if st.session_state.get("dhan_autofill_sig") != sig:
                if st.session_state.get("_attempted_dhan_autofill_sig") != sig:
                    store["dhan_security_id"] = ""      # sig change ALWAYS clears stale IDs
            if use_dhan_feed or dhan_enabled or options_mode:
                _try_autofill(sig, _fetch_eq_id, "dhan_autofill_sig", "dhan_autofill_last_try")

            sec_id = cfg_text(ui, "Security ID (always visible & mandatory — auto-filled when the Dhan feed or "
                                  "order placement is enabled, manual entry in pure-yfinance mode)",
                              "dhan_security_id", "", prefix=prefix)
            product_cfg["security_id"] = sec_id.strip()

        # ---- Order types, Dhan quantity, bracket orders ----
        c1, c2 = ui.columns(2)
        entry_order_type = cfg_selectbox(c1, "Entry Order Type", "entry_order_type", ["MARKET", "LIMIT"], default="MARKET", prefix=prefix)
        exit_order_type = cfg_selectbox(c2, "Exit Order Type", "exit_order_type", ["MARKET", "LIMIT"], default="MARKET", prefix=prefix)
        ui.caption("LIMIT carries the computed price (entry reference / exit level) on the order.")

        dhan_qty = cfg_number(ui, "Dhan Quantity (real orders use this; paper P&L uses the paper Quantity above)",
                              "dhan_qty", 1, 1, 1000000, is_int=True, prefix=prefix)
        ui.caption("Partial books send a proportional share of the Dhan quantity.")

        bo_enabled = cfg_checkbox(ui, "Use Broker SL/Target (Bracket Order)", "bo_enabled", False, prefix=prefix)
        product_cfg["bo_enabled"] = bo_enabled
        if bo_enabled:
            c1, c2, c3 = ui.columns(3)
            product_cfg["bo_sl_points"] = cfg_number(c1, "SL Points (boStopLossValue)", "bo_sl_points", 10.0, 0.1, 100000.0, prefix=prefix)
            product_cfg["bo_target_points"] = cfg_number(c2, "Target Points (boProfitValue)", "bo_target_points", 20.0, 0.1, 200000.0, prefix=prefix)
            product_cfg["bo_trail_jump"] = cfg_number(c3, "Trail SL Jump (0 = off)", "bo_trail_jump", 0.0, 0.0, 100000.0, prefix=prefix)
            ui.caption("Entries go out as productType \"BO\". When the app then detects \"Stoploss Hit\"/\"Target Hit\", "
                       "it SKIPS its own exit order — the broker's legs already closed the position (avoids double "
                       "exits). Signal exits and manual square-offs are still sent.")

        product_cfg["instrument"] = instrument_type
        product_cfg["exchange"] = exchange
        product_cfg["exchange_segment"] = dhan_exchange_segment(meta["kind"], exchange)
        product_cfg["product"] = meta["product"]
        product_cfg["options_mode"] = options_mode
        product_cfg["expiry"] = expiry
        product_cfg["ce_strike"] = ce_strike if is_opts else None
        product_cfg["pe_strike"] = pe_strike if is_opts else None
        product_cfg["underlying"] = underlying
        product_cfg["lot_size"] = lot_size

        if options_mode and not dhan_enabled:
            ui.info("📄 Options Trading with Dhan Order Placement OFF = PAPER trading of the option legs. "
                    "Turn order placement ON to send REAL orders using the exact same configuration values above.")
    else:
        ui.caption("Disabled by default. Live trading tab runs in paper/simulation mode until enabled.")

    # ----------------------------------------------------------- EMAIL   --
    ui.markdown("### 📧 Email Notifications")
    email_enabled = cfg_checkbox(ui, "Send Email Notification", "email_enabled", False, prefix=prefix)
    email_from = str(store.get("email_from", EMAIL_DEFAULT_FROM) or EMAIL_DEFAULT_FROM)
    email_to, email_app_password = "", ""
    if email_enabled:
        email_from = cfg_text(ui, "From (Gmail address)", "email_from", EMAIL_DEFAULT_FROM, prefix=prefix)
        email_to = cfg_text(ui, "To (comma-separated)", "email_to", "", prefix=prefix)
        email_app_password = cfg_text(ui, "Gmail App Password", "email_app_password", "", type="password", prefix=prefix)
        ui.caption("Emails via Gmail SMTP (SSL 465) on entry, exit, partial book, and manual square-off — containing "
                   "strategy/entry/SL/target/exit reason/points/PnL. A mail failure never blocks trading, it only "
                   "shows a warning.")
    else:
        email_to = str(store.get("email_to", "") or "")
        email_app_password = str(store.get("email_app_password", "") or "")

    return dict(
        ticker=ticker, ticker_choice=ticker_choice, interval=interval, period=period, qty=qty,
        strategy=strategy, sl_type=sl_type, target_type=target_type, params=params, filters=filters,
        wf_enabled=wf_enabled, wf_folds=wf_folds, cost_enabled=cost_enabled, cost_cfg=cost_cfg,
        risk_ctrl=risk_ctrl, gates=gates,
        options_mode=options_mode,
        premium_mode=premium_mode,
        use_dhan_feed=use_dhan_feed,
        dhan_enabled=dhan_enabled, dhan_client_id=dhan_client_id, dhan_access_token=dhan_access_token,
        product_cfg=product_cfg, entry_order_type=entry_order_type, exit_order_type=exit_order_type,
        dhan_qty=dhan_qty,
        email_enabled=email_enabled, email_from=email_from, email_to=email_to,
        email_app_password=email_app_password,
    )


# ============================================================================
# SIDEBAR (one of the two live views of the shared config store)
# ============================================================================

config = render_config_controls(st.sidebar, "sb")
if st.session_state.get("cfg_applied_msg"):
    st.sidebar.success(st.session_state.pop("cfg_applied_msg"))

ticker = config["ticker"]
ticker_choice = config["ticker_choice"]
interval = config["interval"]
period = config["period"]
qty = config["qty"]
strategy = config["strategy"]
sl_type = config["sl_type"]
target_type = config["target_type"]
params = config["params"]
filters = config["filters"]
wf_enabled = config["wf_enabled"]
wf_folds = config["wf_folds"]
cost_enabled = config["cost_enabled"]
cost_cfg = config["cost_cfg"]
risk_ctrl = config["risk_ctrl"]
gates = config["gates"]
dhan_enabled = config["dhan_enabled"]
dhan_client_id = config["dhan_client_id"]
dhan_access_token = config["dhan_access_token"]
product_cfg = config["product_cfg"]


# ============================================================================
# HELPERS SHARED ACROSS TABS
# ============================================================================

def price_chart(df, trades_df=None, title="", ema_overlay=None, extra_lines=None):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
    if ema_overlay:
        for period, color in ema_overlay:
            series = ema(df["Close"], period)
            fig.add_trace(go.Scatter(x=df.index, y=series, mode="lines",
                                      line=dict(width=1.6, color=color), name=f"EMA {period}"))
    if extra_lines:
        for label, y_val, color, dash in extra_lines:
            fig.add_hline(y=y_val, line=dict(color=color, dash=dash, width=1.2),
                          annotation_text=label, annotation_position="right")
    if trades_df is not None and not trades_df.empty:
        longs = trades_df[trades_df["Direction"] == "LONG"]
        shorts = trades_df[trades_df["Direction"] == "SHORT"]
        fig.add_trace(go.Scatter(x=longs["Entry Time"], y=longs["Entry Price"], mode="markers",
                                  marker=dict(symbol="triangle-up", color="lime", size=11), name="Long Entry"))
        fig.add_trace(go.Scatter(x=shorts["Entry Time"], y=shorts["Entry Price"], mode="markers",
                                  marker=dict(symbol="triangle-down", color="red", size=11), name="Short Entry"))
        fig.add_trace(go.Scatter(x=trades_df["Exit Time"], y=trades_df["Exit Price"], mode="markers",
                                  marker=dict(symbol="x", color="orange", size=9), name="Exit"))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520, margin=dict(t=40, b=10), legend=dict(orientation="h"))
    return fig


def describe_signal_status(df, strategy, params, filters):
    """
    Human-readable read of exactly where the current (latest closed) candle
    stands relative to what each active condition needs to fire a buy or a
    sell. Not a guarantee of a signal — just a transparent status board.
    Any indicator without enough warm-up history reports 'N/A' instead of a
    misleading half-computed number.
    """
    lines = []
    close = df["Close"]
    f, s = params.get("ema_fast", 9), params.get("ema_slow", 15)
    ef, es = ema(close, f), ema(close, s)
    ef_val, ef_ok = safe_indicator_value(ef, max(f * 3, f + 5))
    es_val, es_ok = safe_indicator_value(es, max(s * 3, s + 5))
    a_series = atr(df, 14)

    if strategy in ("EMA Crossover", "Pro: EMA50 Trend + EMA9/15 Pullback"):
        if ef_ok and es_ok:
            gap = ef_val - es_val
            state = "🟢 BULLISH (fast > slow)" if ef_val > es_val else "🔴 BEARISH (fast < slow)"
            lines.append(f"EMA{f}={ef_val:.2f} vs EMA{s}={es_val:.2f} → {state}, gap = {gap:+.2f}")
            if ef_val > es_val:
                lines.append(f"Needs EMA{f} to cross BELOW EMA{s} for a fresh SELL signal (currently {gap:+.2f} above).")
            else:
                lines.append(f"Needs EMA{f} to cross ABOVE EMA{s} for a fresh BUY signal (currently {gap:+.2f} below).")
        else:
            lines.append(f"EMA{f}/EMA{s}: N/A — need at least {max(f,s)*3} candles of history to warm up reliably, only have {len(df)}.")

    if strategy == "RSI Cross" or filters.get("rsi_enabled"):
        r = rsi(close, params.get("rsi_period", 14))
        r_val, r_ok = safe_indicator_value(r, params.get("rsi_period", 14) * 3)
        if r_ok:
            lines.append(f"RSI({params.get('rsi_period',14)}) = {r_val:.1f}. Buy needs RSI to cross UP through 30 (distance: {r_val-30:+.1f}). "
                         f"Sell needs RSI to cross DOWN through 70 (distance: {70-r_val:+.1f}).")
        else:
            lines.append(f"RSI: N/A — insufficient warm-up history ({len(df)} candles available).")

    if strategy in ("Bollinger Bands", "Pro: BB+RSI Mean Reversion (ATR filtered)") or filters.get("bb_enabled"):
        upper, mid, lower = bollinger(close, params.get("bb_period", 20), params.get("bb_std", 2))
        u_val, u_ok = safe_indicator_value(upper, params.get("bb_period", 20) * 2)
        l_val, l_ok = safe_indicator_value(lower, params.get("bb_period", 20) * 2)
        m_val, m_ok = safe_indicator_value(mid, params.get("bb_period", 20) * 2)
        if u_ok and l_ok and m_ok:
            c_now = float(close.iloc[-1])
            lines.append(f"Close {c_now:.2f} vs Bollinger band [{l_val:.2f} , {u_val:.2f}] (mid {m_val:.2f}). "
                         f"Distance to lower band: {c_now-l_val:+.2f}, to upper: {u_val-c_now:+.2f}.")
        else:
            lines.append("Bollinger Bands: N/A — insufficient warm-up history.")

    if filters.get("adx_enabled"):
        a_val, a_ok = safe_indicator_value(adx(df, 14), 14 * 4)
        adx_min, adx_max = filters.get("adx_min", 0), filters.get("adx_max", 100)
        if a_ok:
            ok = adx_min <= a_val <= adx_max
            lines.append(f"ADX filter: current ADX = {a_val:.1f}, needs [{adx_min}, {adx_max}] → {'✅ OK' if ok else '❌ blocking entries right now'}")
        else:
            lines.append("ADX filter: N/A — insufficient warm-up history (ADX needs roughly 3-4x its period to stabilize).")

    if filters.get("supertrend_enabled"):
        st_line, st_dir = supertrend(df, filters.get("st_filter_period", 10), filters.get("st_filter_mult", 3.0))
        if len(df) >= filters.get("st_filter_period", 10) * 4:
            d_now = st_dir.iloc[-1]
            lines.append(f"Supertrend filter: currently {'🟢 Bullish' if d_now == 1 else '🔴 Bearish'} → {'only BUY' if d_now==1 else 'only SELL'} entries allowed.")
        else:
            lines.append("Supertrend filter: N/A — insufficient warm-up history.")

    if filters.get("regime_enabled"):
        a_val, a_ok = safe_indicator_value(adx(df, 14), 14 * 4)
        family = STRATEGY_FAMILY.get(strategy, "neutral")
        if not a_ok:
            lines.append("Regime filter: N/A — insufficient warm-up history for ADX.")
        elif family == "trend":
            trend_min = filters.get("regime_trend_min", 25)
            ok = a_val >= trend_min
            lines.append(f"Regime filter (trend strategy): ADX {a_val:.1f} needs ≥ {trend_min} → {'✅ trending, OK' if ok else '❌ not trending enough, blocking entries'}")
        elif family == "mean_reversion":
            range_max = filters.get("regime_range_max", 20)
            ok = a_val <= range_max
            lines.append(f"Regime filter (mean-reversion strategy): ADX {a_val:.1f} needs ≤ {range_max} → {'✅ ranging, OK' if ok else '❌ trending too hard, blocking entries'}")

    if filters.get("angle_enabled") and ef_ok and es_ok:
        a_now = a_series.iloc[-1] if not pd.isna(a_series.iloc[-1]) else None
        if a_now:
            ema_fast_delta = ef.diff().iloc[-1]
            angle_now = np.degrees(np.arctan2(abs(ema_fast_delta), a_now)) if a_now > 0 else None
            if angle_now is not None:
                ok = angle_now >= filters.get("angle_min_deg", 0)
                lines.append(f"Crossover angle (ATR-normalized): {angle_now:.1f}°, needs ≥ {filters.get('angle_min_deg',0):.1f}° → {'✅ OK' if ok else '❌ too shallow right now'}")

    # ----- strategy-specific conditions not covered above ------------------
    c_now = float(close.iloc[-1])
    if strategy == "Threshold Cross":
        thr = params.get("threshold", c_now)
        cd = params.get("threshold_direction", "Below")
        if cd == "Above":
            lines.append(f"Threshold Cross (Above): close {c_now:.2f} vs threshold {thr:.2f} → needs price to cross DOWN "
                         f"through it for a SHORT (distance {c_now - thr:+.2f}).")
        else:
            lines.append(f"Threshold Cross (Below): close {c_now:.2f} vs threshold {thr:.2f} → needs price to cross UP "
                         f"through it for a LONG (distance {thr - c_now:+.2f}).")
    if strategy == "Simple Buy Only":
        prev_c = float(close.iloc[-2])
        lines.append(f"Simple Buy Only: LTP must be above previous close {prev_c:.2f} → currently {c_now:.2f} ({c_now - prev_c:+.2f}).")
    if strategy == "Simple Sell Only":
        prev_c = float(close.iloc[-2])
        lines.append(f"Simple Sell Only: LTP must be below previous close {prev_c:.2f} → currently {c_now:.2f} ({c_now - prev_c:+.2f}).")
    if strategy == "Pro: MACD Crossover":
        m_line, m_sig, _ = macd(close, params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9))
        if len(df) >= params.get("macd_slow", 26) * 3:
            gap = float(m_line.iloc[-1] - m_sig.iloc[-1])
            lines.append(f"MACD {m_line.iloc[-1]:+.2f} vs signal {m_sig.iloc[-1]:+.2f} → "
                         f"{'🟢 above (needs cross DOWN for a fresh SELL)' if gap > 0 else '🔴 below (needs cross UP for a fresh BUY)'}, gap {gap:+.2f}.")
        else:
            lines.append("MACD: N/A — insufficient warm-up history.")
    if strategy == "Pro: VWAP + Supertrend Trend":
        st_line, st_dir = supertrend(df, params.get("st_period", 10), params.get("st_mult", 3.0))
        if len(df) >= params.get("st_period", 10) * 4:
            v = vwap(df)
            lines.append(f"VWAP {float(v.iloc[-1]):.2f} vs close {c_now:.2f}; Supertrend {'🟢 Bullish' if st_dir.iloc[-1]==1 else '🔴 Bearish'} — "
                         "BUY needs close above VWAP with bullish Supertrend flip; SELL the reverse.")
    if strategy == "Pro: Donchian Channel Breakout":
        p = params.get("donchian_period", 20)
        if len(df) > p + 2:
            upper, _, lower = donchian(df, p)
            lines.append(f"Donchian({p}): close {c_now:.2f} vs upper {float(upper.iloc[-2]):.2f} (BUY breakout needs a close above) "
                         f"/ lower {float(lower.iloc[-2]):.2f} (SELL breakout needs a close below).")
    if strategy == "Pro: Stochastic Reversal":
        k, d = stochastic(df, params.get("stoch_k", 14), params.get("stoch_d", 3))
        if len(df) >= params.get("stoch_k", 14) * 3:
            lines.append(f"Stochastic %K {float(k.iloc[-1]):.1f} / %D {float(d.iloc[-1]):.1f} — BUY needs %K to cross up "
                         "through %D from oversold (<20); SELL a cross down from overbought (>80).")
    if strategy == "Pro: TEMA Trend Flip":
        t = tema(close, params.get("tema_period", 20))
        if len(df) >= params.get("tema_period", 20) * 3:
            lines.append(f"TEMA({params.get('tema_period',20)}) = {float(t.iloc[-1]):.2f} vs close {c_now:.2f} → "
                         f"{'🟢 price above (SELL needs a flip below)' if c_now > float(t.iloc[-1]) else '🔴 price below (BUY needs a flip above)'}.")
    if strategy == "Pro: CCI Extreme Reversal":
        c_ind = cci(df, params.get("cci_period", 20))
        if len(df) >= params.get("cci_period", 20) * 3:
            cv = float(c_ind.iloc[-1])
            lines.append(f"CCI({params.get('cci_period',20)}) = {cv:+.1f} — BUY needs a recover up through −100 "
                         f"(distance {cv + 100:+.1f}); SELL a fall down through +100 (distance {100 - cv:+.1f}).")
    if strategy == "Volume Breakout":
        vw, vf = params.get("vol_window", 20), params.get("vol_factor", 2.0)
        if "Volume" in df.columns and len(df) > vw + 2:
            v_now = float(df["Volume"].iloc[-1]); v_avg = float(df["Volume"].rolling(vw).mean().iloc[-1])
            need = v_avg * vf
            lines.append(f"Volume Breakout: current vol {v_now:,.0f} vs required {need:,.0f} ({vf}× the {vw}-bar avg {v_avg:,.0f}) → "
                         f"{'✅ spike present' if v_now >= need else '❌ no spike yet'} (plus a range breakout in price).")

    # ----- ALL remaining active entry filters (complete coverage) ----------
    if filters.get("ema20_enabled"):
        e20 = ema(close, 20)
        e_val, e_ok = safe_indicator_value(e20, 40)
        if e_ok:
            lines.append(f"EMA20 filter: close {c_now:.2f} vs EMA20 {e_val:.2f} → "
                         f"{'✅ BUYs allowed (close above)' if c_now > e_val else '❌ BUYs blocked'} · "
                         f"{'✅ SELLs allowed (close below)' if c_now < e_val else '❌ SELLs blocked'} (distance {c_now - e_val:+.2f}).")
        else:
            lines.append("EMA20 filter: N/A — insufficient warm-up history.")
    if filters.get("sma20_enabled"):
        s20 = sma(close, 20)
        s_val, s_ok = safe_indicator_value(s20, 25)
        if s_ok:
            lines.append(f"SMA20 filter: close {c_now:.2f} vs SMA20 {s_val:.2f} → "
                         f"{'✅ BUYs allowed (close above)' if c_now > s_val else '❌ BUYs blocked'} · "
                         f"{'✅ SELLs allowed (close below)' if c_now < s_val else '❌ SELLs blocked'} (distance {c_now - s_val:+.2f}).")
        else:
            lines.append("SMA20 filter: N/A — insufficient warm-up history.")
    if filters.get("bb_enabled"):
        upper_f, _, lower_f = bollinger(close, 20, 2)
        uf, uf_ok = safe_indicator_value(upper_f, 40)
        lf, lf_ok = safe_indicator_value(lower_f, 40)
        if uf_ok and lf_ok:
            buy_ok, sell_ok = c_now <= uf, c_now >= lf
            lines.append(f"Bollinger filter: close {c_now:.2f} must be ≤ upper {uf:.2f} for BUYs "
                         f"({'✅' if buy_ok else '❌'}) and ≥ lower {lf:.2f} for SELLs ({'✅' if sell_ok else '❌'}).")
        else:
            lines.append("Bollinger filter: N/A — insufficient warm-up history.")
    if filters.get("smc_enabled"):
        try:
            sh, sl_ = swing_points(df, 3)
            last_high = df["High"].where(sh).ffill()
            last_low = df["Low"].where(sl_).ffill()
            lh = float(last_high.shift(1).iloc[-1]) if not pd.isna(last_high.shift(1).iloc[-1]) else None
            ll = float(last_low.shift(1).iloc[-1]) if not pd.isna(last_low.shift(1).iloc[-1]) else None
            if lh is not None and ll is not None:
                bos_up_now = c_now > lh
                bos_dn_now = c_now < ll
                lines.append(f"SMC (Structure Break) filter: close {c_now:.2f} vs last swing high {lh:.2f} "
                             f"(break above = bullish BOS → {'✅ BUYs allowed NOW' if bos_up_now else f'❌ needs {lh - c_now:+.2f} more'}) "
                             f"and last swing low {ll:.2f} (break below = bearish BOS → "
                             f"{'✅ SELLs allowed NOW' if bos_dn_now else f'❌ needs {c_now - ll:+.2f} more down'}).")
            else:
                lines.append("SMC filter: N/A — no confirmed swing points yet in this window.")
        except Exception:
            lines.append("SMC filter: N/A — could not compute swing structure on this data.")
    if filters.get("atr_enabled"):
        a_val_f, a_ok_f = safe_indicator_value(a_series, 14 * 3)
        atr_min, atr_max = filters.get("atr_min", 0.0), filters.get("atr_max", 1e9)
        if a_ok_f:
            ok = atr_min <= a_val_f <= atr_max
            lines.append(f"ATR (Volatility) filter: ATR(14) = {a_val_f:.2f}, needs [{atr_min:.2f}, {atr_max:.2f}] → "
                         f"{'✅ OK' if ok else '❌ blocking entries right now'}.")
        else:
            lines.append("ATR filter: N/A — insufficient warm-up history.")
    if filters.get("crossover_quality_enabled"):
        mode = filters.get("crossover_quality_mode", "Simple Crossover")
        rng = float(df["High"].iloc[-1] - df["Low"].iloc[-1])
        if mode == "Crossover with Candle Size":
            need = filters.get("crossover_min_points", 1.0)
            lines.append(f"Crossover Confirmation ({mode}): current candle range {rng:.2f} pts, needs ≥ {need:.2f} → "
                         f"{'✅ OK' if rng >= need else '❌ candle too small'} (only checked on the crossover bar itself).")
        elif mode == "Crossover with ATR-based Candle Size":
            a_now2 = float(a_series.iloc[-1]) if not pd.isna(a_series.iloc[-1]) else None
            if a_now2:
                need = a_now2 * filters.get("crossover_atr_mult", 1.0)
                lines.append(f"Crossover Confirmation ({mode}): current candle range {rng:.2f} pts, needs ≥ {need:.2f} "
                             f"({filters.get('crossover_atr_mult',1.0)}×ATR) → {'✅ OK' if rng >= need else '❌ candle too small'} "
                             "(only checked on the crossover bar itself).")
        else:
            lines.append("Crossover Confirmation (Simple Crossover): no candle-size requirement — any genuine crossover bar passes.")

    if filters.get("vix_enabled"):
        vix_aligned = get_vix_aligned(df.index)
        vix_val = vix_aligned.iloc[-1] if len(vix_aligned) else np.nan
        vix_min, vix_max = filters.get("vix_min", 0), filters.get("vix_max", 100)
        if pd.isna(vix_val):
            lines.append("India VIX filter: N/A — couldn't fetch VIX data right now.")
        else:
            ok = vix_min <= vix_val <= vix_max
            lines.append(f"India VIX: {vix_val:.2f}, needs [{vix_min}, {vix_max}] → {'✅ OK' if ok else '❌ blocking entries right now'}")

    if not lines:
        lines.append("The selected strategy's condition is evaluated on each candle close — no additional live-readable "
                     "state to display for it, and no entry filters are active.")
    return lines


@st.fragment(run_every=3)
def live_dashboard_fragment(ticker, interval, period, strategy, params, filters):
    """
    Everything here re-renders on its own every ~3s WITHOUT rerunning the rest
    of the page — this is what makes the signal status board (RSI/EMA/ADX/etc.
    values) update live instead of only on button click or full page refresh.
    Only ever mounted while Live Monitoring is ON, so it costs zero extra API
    calls while stopped.
    """
    raw_status = fetch_data(ticker, interval, period)
    if raw_status.empty or len(raw_status) < 30:
        st.caption("Not enough data yet to compute signal status.")
        return

    st.markdown("###### 📊 Indicator Dashboard")
    close = raw_status["Close"]
    f, s = params.get("ema_fast", 9), params.get("ema_slow", 15)
    ef_val, ef_ok = safe_indicator_value(ema(close, f), max(f * 3, f + 5))
    es_val, es_ok = safe_indicator_value(ema(close, s), max(s * 3, s + 5))

    cols = st.columns(4)
    cols[0].metric(f"EMA {f} (fast)", f"{ef_val:.2f}" if ef_ok else "N/A")
    cols[1].metric(f"EMA {s} (slow)", f"{es_val:.2f}" if es_ok else "N/A")
    cols[2].metric("Gap", f"{(ef_val-es_val):+.2f}" if ef_ok and es_ok else "N/A")
    if filters.get("adx_enabled") or filters.get("regime_enabled"):
        adx_val, adx_ok = safe_indicator_value(adx(raw_status, 14), 14 * 4)
        cols[3].metric("ADX", f"{adx_val:.1f}" if adx_ok else "N/A")
    elif filters.get("vix_enabled"):
        vix_series = get_vix_aligned(raw_status.index)
        vix_val = vix_series.iloc[-1] if len(vix_series) else np.nan
        cols[3].metric("India VIX", f"{vix_val:.2f}" if not pd.isna(vix_val) else "N/A")
    else:
        rsi_val, rsi_ok = safe_indicator_value(rsi(close, params.get("rsi_period", 14)), params.get("rsi_period", 14) * 3)
        cols[3].metric("RSI", f"{rsi_val:.1f}" if rsi_ok else "N/A")

    st.markdown("###### 📟 Signal Status Board")
    st.caption("What the current (last closed) candle is showing vs. what's needed to trigger a fresh buy or sell. Updates automatically every ~3s while live monitoring is on.")
    for line in describe_signal_status(raw_status, strategy, params, filters):
        st.write("• " + line)


def _options_active(full_cfg):
    pc = (full_cfg or {}).get("product_cfg") or {}
    return bool(pc.get("options_mode") or "Options" in str(pc.get("instrument", ""))) \
        and pc.get("ce_security_id") and pc.get("pe_security_id")


def _live_capture_option_entry(new_pos, direction, full_cfg):
    """Options mode: record which leg is bought (LONG→CE, SHORT→PE — direction
    already includes any flip) and its Dhan premium at entry, with NO delay.
    Works in paper mode too: signals/SL/target run on the underlying (the main
    algorithm), while the premium is tracked for the records."""
    if not _options_active(full_cfg):
        return
    pc = full_cfg["product_cfg"]
    leg = "CE" if direction == 1 else "PE"
    sec_id = pc["ce_security_id"] if direction == 1 else pc["pe_security_id"]
    new_pos["opt_leg"] = leg
    new_pos["opt_security_id"] = sec_id
    prem = dhan_get_ltp(sec_id, pc.get("exchange_segment", "NSE_FNO"))
    new_pos["opt_entry_premium"] = round(prem, 2) if prem is not None else None


def _live_attach_option_premiums(row, pos, full_cfg, qty_closed, closing=True):
    """Adds Option Leg / Entry Premium / Exit Premium / Premium PnL columns to
    a live-history row when trading option legs (zero-delay Dhan premium)."""
    if not pos.get("opt_leg"):
        return
    row["Option Leg"] = pos.get("opt_leg")
    row["Option Security ID"] = pos.get("opt_security_id")
    row["Option Entry Premium"] = pos.get("opt_entry_premium")
    exit_prem = None
    if _options_active(full_cfg):
        pc = full_cfg["product_cfg"]
        exit_prem = dhan_get_ltp(pos.get("opt_security_id"), pc.get("exchange_segment", "NSE_FNO"))
    row["Option Exit Premium"] = round(exit_prem, 2) if exit_prem is not None else None
    if exit_prem is not None and pos.get("opt_entry_premium") is not None:
        # Options are always BOUGHT → premium PnL = (exit − entry) × qty
        row["Option Premium PnL"] = round((exit_prem - pos["opt_entry_premium"]) * qty_closed, 2)


def evaluate_live_signal(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                          dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl,
                          full_cfg=None):
    """
    Top-level (not nested-closure) live signal evaluator. This is deliberately
    a plain module-level function taking every input explicitly, rather than
    a function-inside-a-function relying on captured variables — nesting it
    inside `with tab_live:` and wrapping THAT in a fragment was fragile and is
    exactly what caused positions to silently stop updating. This version, and
    its fragment wrapper right below, follow the same plain top-level pattern
    already working fine for live_dashboard_fragment / live_position_fragment.
    """
    raw = fetch_data(ticker, interval, period)
    if raw.empty or len(raw) < 30:
        st.error("Not enough data to evaluate a signal.")
        return None
    live_filters = dict(filters)
    live_filters["current_strategy"] = strategy
    sig_df = apply_filters(generate_signals(raw, strategy, params), live_filters, params)
    a_series = atr(sig_df, 14)
    open_pos = st.session_state.live_positions

    # Fresh, uncached last-traded price — this is what SL/Target/exit checks
    # below are compared against, not the (possibly ~30s stale, cached)
    # candle data used for indicators/signals. Falls back to the last candle
    # close only if a live tick genuinely couldn't be fetched this cycle.
    ltp = get_live_ltp(ticker)
    if ltp is None:
        ltp = float(sig_df["Close"].iloc[-1])

    # Immediate-execution strategies (Simple Buy/Sell Only, Threshold Cross)
    # check the CURRENT price against the last CLOSED candle directly — no
    # "wait for this candle to close" delay, since there's no candle shape to
    # confirm, just a price level.
    if strategy in IMMEDIATE_EXECUTION_STRATEGIES:
        prev_close = float(sig_df["Close"].iloc[-2])
        if strategy == "Simple Buy Only":
            last_sig = 1 if ltp > prev_close else 0
        elif strategy == "Simple Sell Only":
            last_sig = -1 if ltp < prev_close else 0
        else:  # Threshold Cross — same Cross Direction rule as the backtest:
            thr = params.get("threshold", prev_close)
            if params.get("threshold_direction", "Below") == "Above":
                # "Above": price crosses DOWN from above → SHORT
                last_sig = -1 if (ltp < thr and prev_close >= thr) else 0
            else:
                # "Below" (default): approaches from below, crosses UP → LONG
                last_sig = 1 if (ltp > thr and prev_close <= thr) else 0
        # Flip FIRST, then the Trade Direction filter — same central rule as
        # candle-based strategies get inside generate_signals().
        last_sig = apply_direction_rules_to_scalar(last_sig, params)
        entry_reference_price = ltp
    else:
        last_sig = int(sig_df["signal"].iloc[-2])  # last CLOSED candle's signal
        entry_reference_price = float(sig_df["Open"].iloc[-1])  # next candle's open

    if open_pos:
        pos = open_pos[0]
        i = len(sig_df) - 1
        candle = sig_df.iloc[i]
        pos = update_trade_levels(pos, i, sig_df, params, a_series)
        pos["highest"] = max(pos["highest"], ltp)
        pos["lowest"] = min(pos["lowest"], ltp)

        exited, exit_price, reason = False, None, None
        if pos.get("pending_exit_reason"):
            exited, exit_price, reason = True, candle["Open"], pos["pending_exit_reason"]
        if not exited:
            sp_exit, sp_price, sp_reason = check_special_exit_conditions(pos, {"Close": ltp})
            if sp_exit:
                exited, exit_price, reason = True, sp_price, sp_reason
        if not exited and risk_ctrl.get("loss_duration_enabled"):
            td_exit, td_price, td_reason = check_time_based_exit(
                pos, sig_df.index[-1], ltp,
                risk_ctrl.get("loss_duration_min_minutes", 1), risk_ctrl.get("loss_duration_max_minutes", 5),
            )
            if td_exit:
                exited, exit_price, reason = True, td_price, td_reason
        if not exited:
            hard_exit, hard_price, hard_reason = check_hard_exit_ltp(pos, ltp)
            if hard_exit:
                if pos["target_type"] == "Partial Book + Trail Remainder" and "Target Hit" in hard_reason and not pos["partial_booked"]:
                    book_qty = max(1, round(pos["original_qty"] * pos["partial_book_pct"] / 100.0))
                    book_qty = min(book_qty, pos["remaining_qty"])
                    partial_points = (hard_price - pos["entry_price"]) * pos["direction"]
                    exit_candle = sig_df.iloc[-1]
                    partial_reason = f"Partial Book ({book_qty}/{pos['original_qty']} qty @ Target 1)"
                    row = {
                        "Entry Time": pos["entry_time"], "Entry Price": round(pos["entry_price"], 2),
                        "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
                        "Exit Time": sig_df.index[-1], "Exit Price": round(float(hard_price), 2),
                        "SL": round(pos["initial_sl"], 2), "Target": round(pos["initial_target"], 2),
                        "Highest": round(pos["highest"], 2), "Lowest": round(pos["lowest"], 2),
                        "Points": round(partial_points, 2), "PnL": round(partial_points * book_qty, 2),
                        "Exit Reason": partial_reason, "Qty": book_qty,
                        "Exit Open": round(float(exit_candle["Open"]), 2), "Exit High": round(float(exit_candle["High"]), 2),
                        "Exit Low": round(float(exit_candle["Low"]), 2), "Exit Close": round(float(exit_candle["Close"]), 2),
                    }
                    _live_attach_option_premiums(row, pos, full_cfg, book_qty, closing=False)
                    st.session_state.live_history.append(row)
                    pos["remaining_qty"] -= book_qty
                    pos["partial_booked"] = True
                    note_trade_event()  # feeds the entry-cooldown gate
                    if full_cfg:
                        res = dispatch_dhan_event(full_cfg, pos["direction"], False, "Partial Book",
                                                  book_qty, pos["original_qty"], hard_price,
                                                  exit_reason=partial_reason)
                        if res:
                            st.json(res)
                        email_trade_event(full_cfg, "Partial Book", {
                            "Ticker": ticker, "Strategy": strategy,
                            "Direction": row["Direction"], "Entry Price": row["Entry Price"],
                            "SL": row["SL"], "Target": row["Target"],
                            "Exit Price": row["Exit Price"], "Exit Reason": partial_reason,
                            "Points": row["Points"], "PnL": row["PnL"], "Qty": book_qty,
                        })
                    if pos["remaining_qty"] <= 0:
                        st.session_state.live_positions = []
                        st.success(f"Fully booked at Target 1 @ {hard_price:.2f}")
                        return sig_df
                    else:
                        pos["target_type"] = "Trailing Target (Display Only)"
                        if pos["sl_type"] not in ("Trailing SL (Points)", "ATR Based SL", "Autopilot SL"):
                            pos["sl_type"] = "ATR Based SL"
                        st.session_state.live_positions = [pos]
                        st.success(f"Partial booked ({book_qty} qty) @ {hard_price:.2f} — remaining {pos['remaining_qty']} qty now trailing.")
                        return sig_df
                else:
                    exited, exit_price, reason = True, hard_price, hard_reason
        if not exited and full_cfg:
            # 🚧 Risk gate: Max Hold Duration of Profitable Trade — if held
            # ≥ N minutes AND currently in profit → exit immediately.
            ph_exit, ph_price, ph_reason = check_profitable_hold_exit(full_cfg.get("gates"), pos, ltp)
            if ph_exit:
                exited, exit_price, reason = True, ph_price, ph_reason
        if not exited:
            sig_exit, sig_reason = detect_signal_exit_condition(pos, i, sig_df, params)
            if sig_exit:
                pos["pending_exit_reason"] = sig_reason

        pos["current_price"] = ltp
        if exited:
            points = (exit_price - pos["entry_price"]) * pos["direction"]
            exit_candle = sig_df.iloc[-1]
            row = {
                "Entry Time": pos["entry_time"], "Entry Price": round(pos["entry_price"], 2),
                "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
                "Exit Time": sig_df.index[-1], "Exit Price": round(float(exit_price), 2),
                "SL": round(pos["initial_sl"], 2), "Target": round(pos["initial_target"], 2),
                "Highest": round(pos["highest"], 2), "Lowest": round(pos["lowest"], 2),
                "Points": round(points, 2), "PnL": round(points * pos["remaining_qty"], 2),
                "Exit Reason": reason, "Qty": pos["remaining_qty"],
                "Exit Open": round(float(exit_candle["Open"]), 2), "Exit High": round(float(exit_candle["High"]), 2),
                "Exit Low": round(float(exit_candle["Low"]), 2), "Exit Close": round(float(exit_candle["Close"]), 2),
            }
            _live_attach_option_premiums(row, pos, full_cfg, pos["remaining_qty"], closing=True)
            st.session_state.live_history.append(row)
            st.session_state.live_positions = []
            note_trade_event()  # feeds the entry-cooldown gate
            st.success(f"Position closed: {reason} @ {exit_price:.2f}")
            if full_cfg:
                # BO-managed Stoploss/Target hits are automatically SKIPPED
                # inside dispatch_dhan_event to avoid a double exit.
                res = dispatch_dhan_event(full_cfg, pos["direction"], False, "Exit",
                                          pos["remaining_qty"], pos["remaining_qty"], exit_price,
                                          exit_reason=reason)
                if res:
                    st.json(res)
                email_trade_event(full_cfg, "Trade Exit", {
                    "Ticker": ticker, "Strategy": strategy,
                    "Direction": row["Direction"], "Entry Price": row["Entry Price"],
                    "SL": row["SL"], "Target": row["Target"],
                    "Exit Price": row["Exit Price"], "Exit Reason": reason,
                    "Points": row["Points"], "PnL": row["PnL"], "Qty": pos["remaining_qty"],
                })
        else:
            st.session_state.live_positions = [pos]
            st.info("Position still open — levels updated.")
    elif last_sig != 0:
        if last_sig == -1 and params.get("long_entries_only"):
            # Premium (options-buyer) mode: never OPEN on a SHORT signal.
            st.caption("🎯 Premium mode: SHORT signal detected but ignored for entries — options are only ever "
                       "BOUGHT here. (An opposite signal can still exit an open long via 'Strategy Signal Exit'.)")
            return sig_df
        # The candle/tick that produced this signal — used to make sure we
        # only ever act on it ONCE. Without this, a fast target-hit followed
        # by a re-check (every ~5s) would keep seeing the SAME unchanged
        # crossover as "last closed candle's signal" until a genuinely new
        # candle closes, and would re-open a fresh position every cycle —
        # which is exactly the bug that produced repeated instant re-entries.
        if strategy in IMMEDIATE_EXECUTION_STRATEGIES:
            signal_marker = (sig_df.index[-1], last_sig)
        else:
            signal_marker = (sig_df.index[-2], last_sig)

        if st.session_state.get("last_acted_signal_marker") == signal_marker:
            st.caption(f"Signal at {signal_marker[0]} already acted on — waiting for a genuinely new signal before re-entering.")
        else:
            # 🚧 Risk Control gates (live entries only, all default-disabled).
            # A blocked entry shows the specific gate reason right here and
            # on the Live tab header.
            if full_cfg:
                allowed, gate_reason = check_entry_gates(full_cfg.get("gates"),
                                                         full_cfg.get("ticker_choice"), ticker)
            else:
                allowed, gate_reason = True, None
            if not allowed:
                st.session_state.live_blocked_reason = gate_reason
                st.warning(f"🚧 Entry blocked: {gate_reason}")
                return sig_df
            st.session_state.live_blocked_reason = None
            entry_price = entry_reference_price
            a_val = a_series.iloc[-1] if not np.isnan(a_series.iloc[-1]) else entry_price * 0.005
            sl, target, sl_dist, target_dist = calc_initial_sl_target(last_sig, entry_price, a_val, params, sl_type, target_type)
            new_pos = {
                "entry_time": sig_df.index[-1], "entry_price": entry_price, "direction": last_sig,
                "qty": qty, "sl": sl, "target": target, "initial_sl": sl, "initial_target": target,
                "sl_dist": sl_dist, "target_dist": target_dist, "sl_type": sl_type, "target_type": target_type,
                "highest": entry_price, "lowest": entry_price, "current_price": entry_price,
                "pending_exit_reason": None,
                "peak_pl_points": 0.0, "worst_pl_points": 0.0, "loss_since": None,
                "original_qty": qty, "remaining_qty": qty, "partial_booked": False,
                "loss_trigger_points": params.get("loss_trigger_points", 20.0),
                "min_recovery_pct": params.get("min_recovery_pct", 50.0),
                "profit_trigger_points": params.get("profit_trigger_points", 50.0),
                "giveback_pct": params.get("giveback_pct", 30.0),
                "partial_book_pct": params.get("partial_book_pct", 50.0),
            }
            # ---- Options mode: capture the leg + its ZERO-DELAY Dhan premium
            # at entry (yfinance is never used for option premiums).
            _live_capture_option_entry(new_pos, last_sig, full_cfg)
            st.session_state.live_positions = [new_pos]
            st.session_state.last_acted_signal_marker = signal_marker
            note_trade_event(entered=True)  # feeds max-trades/day + cooldown gates
            st.success(f"New {'LONG' if last_sig == 1 else 'SHORT'} position opened @ {entry_price:.2f}")
            if full_cfg:
                res = dispatch_dhan_event(full_cfg, last_sig, True, "Entry", qty, qty, entry_price)
                if res:
                    st.json(res)
                email_trade_event(full_cfg, "Trade Entry", {
                    "Ticker": ticker, "Strategy": strategy,
                    "Direction": "LONG" if last_sig == 1 else "SHORT",
                    "Entry Price": round(entry_price, 2),
                    "SL": round(sl, 2), "Target": round(target, 2), "Qty": qty,
                    **({"Option Leg": new_pos.get("opt_leg"),
                        "Option Entry Premium": new_pos.get("opt_entry_premium")}
                       if new_pos.get("opt_leg") else {}),
                })
    else:
        st.caption("No new signal on the latest closed candle.")
    return sig_df


@st.fragment(run_every=5)
def live_signal_loop_fragment(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                               dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl,
                               full_cfg=None):
    """Re-runs evaluate_live_signal() every ~5s on its own, independent of the
    rest of the page — this is what makes entries/exits keep happening while
    Live Monitoring is on, instead of firing only once at the Start click."""
    evaluate_live_signal(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                          dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl,
                          full_cfg=full_cfg)


def apply_config_to_sidebar(cfg_row):
    """Write a chosen optimization result row DIRECTLY into the shared config
    store (st.session_state.app_cfg) — the Sidebar and Admin Panel are both
    live views of that store, so both pick the values up on the rerun.
    (Replaces the old sidebar_overrides mechanism.)"""
    cfg_set("ticker_choice", cfg_row.get("ticker_choice", ticker_choice))
    if cfg_row.get("ticker_choice", ticker_choice) == "Custom" and cfg_row.get("ticker"):
        cfg_set("ticker_custom", cfg_row["ticker"])
    cfg_set("interval", cfg_row["Timeframe"])
    cfg_set("period", cfg_row["Period"])
    cfg_set("strategy", cfg_row["Strategy"])
    cfg_set("sl_type", cfg_row.get("SL Type", sl_type))
    cfg_set("target_type", cfg_row.get("Target Type", target_type))
    cfg_set("qty", qty)
    st.session_state["cfg_applied_msg"] = "Optimized config applied ✅"
    st.rerun()


def render_range_insight_section(ticker, interval, period, section_title):
    """
    Fetches OHLC data for the given timeframe/period, shows a table with
    per-bar % and absolute change, and a plain-language read of where price
    currently sits within that period's range plus whether the latest move
    is unusually large. This is descriptive/statistical framing, not a
    prediction — phrased with appropriate hedging.
    """
    st.markdown(f"##### {section_title}")
    with st.spinner(f"Fetching {period} of {interval} data…"):
        raw = fetch_data(ticker, interval, period)

    if raw.empty or len(raw) < 5:
        st.warning("Not enough data returned for this timeframe/period.")
        return

    df = raw.copy()
    df["Change"] = df["Close"].diff()
    df["Change %"] = df["Close"].pct_change() * 100

    display_df = df[["Open", "High", "Low", "Close", "Volume", "Change", "Change %"]].round(2)

    def _color_change(val):
        if pd.isna(val) or val == 0:
            return ""
        return "color: #16c784; font-weight: 600;" if val > 0 else "color: #ea3943; font-weight: 600;"

    sorted_df = display_df.sort_index(ascending=False)
    styler = sorted_df.style
    # pandas >=2.1 renamed Styler.applymap -> Styler.map (and removed applymap
    # entirely in some newer releases) — support both instead of assuming one.
    style_fn = getattr(styler, "map", None) or styler.applymap
    styled = style_fn(_color_change, subset=["Change", "Change %"])
    st.dataframe(styled, use_container_width=True)

    period_high = float(df["High"].max())
    period_low = float(df["Low"].min())
    current_close = float(df["Close"].iloc[-1])
    latest_change_pct = float(df["Change %"].iloc[-1]) if not pd.isna(df["Change %"].iloc[-1]) else 0.0
    latest_change_abs = float(df["Change"].iloc[-1]) if not pd.isna(df["Change"].iloc[-1]) else 0.0

    rng = period_high - period_low
    position_pct = ((current_close - period_low) / rng * 100) if rng > 0 else 50.0

    if position_pct >= 80:
        position_desc = f"near the TOP of its range for this period ({position_pct:.0f}th percentile) — stretched to the upside"
    elif position_pct <= 20:
        position_desc = f"near the BOTTOM of its range for this period ({position_pct:.0f}th percentile) — stretched to the downside"
    else:
        position_desc = f"roughly in the MIDDLE of its range for this period ({position_pct:.0f}th percentile)"

    pct_std = df["Change %"].std()
    is_unusual = pct_std > 0 and abs(latest_change_pct) > 1.5 * pct_std

    lines = [
        f"**Range for this period:** Low `{period_low:.2f}` → High `{period_high:.2f}` (spread {rng:.2f}). Current close `{current_close:.2f}` is {position_desc}.",
        f"**Latest bar move:** {latest_change_abs:+.2f} ({latest_change_pct:+.2f}%)"
        + (f" — unusually large versus the typical ±{pct_std:.2f}% swing for this data, worth noting." if is_unusual else " — within a typical range for this data, nothing statistically unusual."),
    ]

    if position_pct >= 80:
        lines.append("Statistically, prices stretched to the top of a recent range sometimes see a pause or partial pullback before continuing — but strong trends can also keep extending. This isn't a sell signal by itself; treat it as one input alongside whatever strategy/indicators you're using.")
    elif position_pct <= 20:
        lines.append("Statistically, prices stretched to the bottom of a recent range sometimes see a bounce or basing period before continuing lower — but downtrends can also keep extending. This isn't a buy signal by itself.")
    else:
        lines.append("Sitting mid-range generally means less positional bias either way — range-bound/choppy behavior is at least as likely as a decisive breakout from here.")

    for line in lines:
        st.write(line)


def render_bin_analysis_section(t1, t2, t1_name, t2_name, p1, diff, fetch_interval, fetch_period, section_label, fwd_n=5):
    """
    Renders one full historical-bin-analysis block (bin table + empirical bias +
    ATR-sized reference levels) for a given timeframe/period. Used twice in the
    Spread tool: once on daily/2y for a stable statistical read, and once on
    whatever timeframe/period is selected in the sidebar for a read that matches
    how the user actually intends to trade.
    """
    st.markdown(f"##### {section_label}")
    with st.spinner(f"Fetching {fetch_period} of {fetch_interval} history for both tickers…"):
        h1 = fetch_data(t1, fetch_interval, fetch_period)
        h2 = fetch_data(t2, fetch_interval, fetch_period)

    if h1.empty or h2.empty:
        st.warning("Not enough historical data for one of the tickers at this timeframe/period.")
        return

    joined = pd.DataFrame({"c1": h1["Close"], "c2": h2["Close"]}).dropna()
    joined["diff"] = joined["c1"] - joined["c2"]
    joined["fwd_ret_1"] = joined["c1"].shift(-fwd_n) / joined["c1"] - 1
    joined["fwd_ret_2"] = joined["c2"].shift(-fwd_n) / joined["c2"] - 1
    joined = joined.dropna()

    if len(joined) < 30:
        st.warning("Not enough overlapping candles at this timeframe/period for a reliable bin analysis. Try a longer period.")
        return

    n_bins = min(8, max(3, len(joined) // 15))
    try:
        joined["bin"] = pd.qcut(joined["diff"], n_bins, duplicates="drop")
    except ValueError:
        joined["bin"] = pd.cut(joined["diff"], n_bins)

    bin_stats = joined.groupby("bin", observed=True).agg(
        n=("diff", "count"), diff_lo=("diff", "min"), diff_hi=("diff", "max"),
        t1_avg_fwd_pct=("fwd_ret_1", lambda x: round(x.mean() * 100, 2)),
        t1_pct_up=("fwd_ret_1", lambda x: round((x > 0).mean() * 100, 1)),
        t2_avg_fwd_pct=("fwd_ret_2", lambda x: round(x.mean() * 100, 2)),
        t2_pct_up=("fwd_ret_2", lambda x: round((x > 0).mean() * 100, 1)),
    ).reset_index(drop=True)

    current_bin_idx = None
    for idx, row in bin_stats.iterrows():
        if row["diff_lo"] <= diff <= row["diff_hi"]:
            current_bin_idx = idx
            break
    if current_bin_idx is None:
        current_bin_idx = 0 if diff < bin_stats["diff_lo"].min() else len(bin_stats) - 1

    display_stats = bin_stats.copy()
    display_stats.insert(0, "Bin", [f"#{i + 1}" for i in range(len(display_stats))])
    display_stats["← Today"] = ["👈" if i == current_bin_idx else "" for i in range(len(display_stats))]
    st.dataframe(
        display_stats.rename(columns={
            "diff_lo": "Diff Range Low", "diff_hi": "Diff Range High", "n": "# Occurrences",
            "t1_avg_fwd_pct": f"{t1_name} Avg Fwd {fwd_n}-bar %", "t1_pct_up": f"{t1_name} % Up",
            "t2_avg_fwd_pct": f"{t2_name} Avg Fwd {fwd_n}-bar %", "t2_pct_up": f"{t2_name} % Up",
        }),
        use_container_width=True, hide_index=True,
    )

    current_row = bin_stats.iloc[current_bin_idx]
    st.info(
        f"Today's difference ({diff:,.2f}) falls in **bin #{current_bin_idx + 1}** "
        f"[{current_row['diff_lo']:.1f} to {current_row['diff_hi']:.1f}], seen {int(current_row['n'])} times in this sample. "
        f"In the {fwd_n} bars *after* being in this bin, historically **{t1_name}** averaged "
        f"{current_row['t1_avg_fwd_pct']:+.2f}% (up {current_row['t1_pct_up']:.0f}% of occurrences), and "
        f"**{t2_name}** averaged {current_row['t2_avg_fwd_pct']:+.2f}% (up {current_row['t2_pct_up']:.0f}% of occurrences)."
    )

    if current_row["t1_avg_fwd_pct"] > 0.3 and current_row["t1_pct_up"] >= 55:
        bias = "UP"
    elif current_row["t1_avg_fwd_pct"] < -0.3 and current_row["t1_pct_up"] <= 45:
        bias = "DOWN"
    else:
        bias = "NEUTRAL"

    if bias == "UP":
        st.success(f"🟢 Empirical bias from this bin: {t1_name} has historically leaned UP from here.")
    elif bias == "DOWN":
        st.warning(f"🔴 Empirical bias from this bin: {t1_name} has historically leaned DOWN from here.")
    else:
        st.info("🟡 This bin shows no clear historical directional bias for a confident call either way.")

    if bias != "NEUTRAL":
        # SL/Target sized from ticker1's OWN volatility (ATR) on THIS timeframe —
        # never from the raw cross-instrument price gap. Using the raw diff as a
        # distance was the bug that produced nonsensical levels (SL far beyond
        # entry, negative targets) whenever the two instruments trade on very
        # different scales. The diff is only used to look up historical
        # conditional behavior, never to size risk.
        a_series = atr(h1, 14)
        a1 = a_series.iloc[-1] if len(h1) > 20 and not np.isnan(a_series.iloc[-1]) else p1 * 0.005
        direction = 1 if bias == "UP" else -1
        sl_dist, target_dist = a1 * 1.5, a1 * 3.0  # keeps ~1:2 R:R
        entry_ref = p1
        sl_ref = entry_ref - sl_dist if direction == 1 else entry_ref + sl_dist
        tgt_ref = entry_ref + target_dist if direction == 1 else entry_ref - target_dist
        st.markdown(
            f"**Reference levels for {t1_name}** (sized off its own 14-period ATR on this timeframe ≈ {a1:.2f}, "
            f"*not* the raw price gap): Entry ≈ `{entry_ref:.2f}` · SL ≈ `{sl_ref:.2f}` · Target ≈ `{tgt_ref:.2f}`"
        )

    st.caption(
        "Empirical conditional-return lookup, not a validated statistical-arbitrage model — small sample sizes "
        "(check '# Occurrences') make the average unreliable. Treat as a directional hint only."
    )


# ============================================================================
# TABS
# ============================================================================

tab_bt, tab_live, tab_hist, tab_heat, tab_opt, tab_spread, tab_ohlc, tab_admin = st.tabs(
    ["📊 Backtest", "🔴 Live Trading", "📜 Trade History", "🔥 Heatmaps", "🧪 Optimization", "🔀 Spread Tool", "📅 OHLC & Range", "🛠 Admin Panel"]
)

# ---------------------------------------------------------------- BACKTEST -
with tab_bt:
    st.subheader(f"Backtest — {ticker_choice} ({ticker}) · {interval} · {period} · {strategy}")
    st.caption("Entry rule: signal confirmed on candle *n* → position opened at the **open of candle n+1**. "
               "Longs check SL (candle low) before target (candle high); shorts check SL (candle high) before target (candle low) — conservative fill assumption.")

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Fetching data and running backtest…"):
            raw = fetch_data(ticker, interval, period)
            if raw.empty:
                st.error("No data returned. Check ticker/timeframe/period combination.")
            else:
                trades_df, sig_df = run_backtest(raw, strategy, sl_type, target_type, params, filters, qty, risk_ctrl)
                st.session_state.last_backtest = trades_df
                st.session_state.last_backtest_df = sig_df

    trades_df = st.session_state.last_backtest
    sig_df = st.session_state.last_backtest_df

    if trades_df is not None and sig_df is not None and not sig_df.empty:
        m = compute_metrics(trades_df)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades", m["total_trades"])
        c2.metric("Accuracy", f"{m['accuracy']}%")
        c3.metric("Total Points", m["total_points"])
        c4.metric("Total PnL", m["total_pnl"])
        c5.metric("Expectancy/Trade", m["expectancy"])
        c6.metric("Sharpe", m["sharpe"])
        st.info(recommend_from_metrics(m))

        # ---- Realistic cost modeling ----
        m_net, trades_display = None, trades_df
        if cost_enabled:
            trades_costed = apply_cost_model(trades_df, cost_cfg, qty)
            m_net = compute_metrics_from_columns(trades_costed, "Points (Net)", "PnL (Net)")
            trades_display = trades_costed
            st.markdown("#### 💸 Cost-Adjusted Results")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Net Accuracy", f"{m_net['accuracy']}%", delta=f"{m_net['accuracy']-m['accuracy']:.1f}pp")
            cc2.metric("Net Total Points", m_net["total_points"], delta=round(m_net["total_points"] - m["total_points"], 2))
            cc3.metric("Net Total PnL", m_net["total_pnl"])
            cc4.metric("Net Expectancy/Trade", m_net["expectancy"], delta=round(m_net["expectancy"] - m["expectancy"], 2))

        # ---- Walk-forward validation ----
        wf_folds_result = None
        if wf_enabled:
            wf_folds_result = walk_forward_folds(trades_df, sig_df.index[0], sig_df.index[-1], wf_folds)
            st.markdown("#### 🧪 Walk-Forward Consistency (Out-of-Sample Folds)")
            wf_table = pd.DataFrame(wf_folds_result)[["Fold", "From", "To", "total_trades", "accuracy", "expectancy", "sharpe", "total_pnl"]]
            st.dataframe(wf_table, use_container_width=True, hide_index=True)
            profitable = sum(1 for f in wf_folds_result if f["total_trades"] > 0 and f["expectancy"] > 0)
            valid = sum(1 for f in wf_folds_result if f["total_trades"] > 0)
            st.caption(f"Profitable in {profitable}/{valid} folds with trades. This checks whether the edge is consistent across time, not concentrated in one lucky stretch.")

        # ---- Smart verdict ----
        st.markdown("#### 🧠 Smart Evaluation Verdict")
        verdict, notes = smart_verdict(m, wf_fold_metrics=wf_folds_result, cost_enabled=cost_enabled, metrics_net=m_net)
        st.subheader(verdict)
        for n in notes:
            st.write(n)
        st.caption("This is a transparent rule-based scorecard over the metrics above — not a trained ML model. Enable Walk-Forward Validation and Cost Modeling in the sidebar for a materially more trustworthy verdict before risking real capital.")

        st.markdown("#### Trade Log")
        st.dataframe(trades_display, use_container_width=True, hide_index=True)

        st.markdown("#### 📐 SL/Target Recommendation (MAE/MFE Analysis)")
        mae_mfe = recommend_sl_target_from_mae_mfe(sig_df, trades_df, lookahead=20)
        if mae_mfe is None:
            st.caption("Need at least 5 trades to compute a reliable recommendation.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Suggested SL (70th pct. of adverse moves)", f"{mae_mfe['suggested_sl']:.2f} pts")
            c2.metric("Suggested Target (50th pct. of favorable moves)", f"{mae_mfe['suggested_target']:.2f} pts")
            c3.metric("Based on", f"{mae_mfe['n_trades']} trades")
            st.caption(
                f"MAE (adverse move before things went right/wrong) distribution: 50th pct {mae_mfe['mae_p50']:.2f}, "
                f"70th pct {mae_mfe['mae_p70']:.2f}, 90th pct {mae_mfe['mae_p90']:.2f}. "
                f"MFE (favorable move available) distribution: 50th pct {mae_mfe['mfe_p50']:.2f}, "
                f"70th pct {mae_mfe['mfe_p70']:.2f}, 90th pct {mae_mfe['mfe_p90']:.2f}. "
                "SL is set loose enough to survive most normal noise (70th percentile of adverse excursion) without "
                "being so wide it erases the point of having a stop. Target is set at the 50th percentile of favorable "
                "excursion — a realistically reachable level roughly half your winners could hit, not the best-case "
                "90th-percentile outlier. This is descriptive of what THIS strategy on THIS data actually did — re-run "
                "it after changing timeframe, period, or filters, since the right SL/Target changes with all of those."
            )

        st.markdown("#### Chart — Price with Entries/Exits")
        st.plotly_chart(
            price_chart(sig_df, trades_df, "Price with Entries/Exits",
                        ema_overlay=[(params.get("ema_fast", 9), "#3399ff"), (params.get("ema_slow", 15), "#ff9933")]),
            use_container_width=True,
        )
    else:
        st.caption("Run a backtest to see results here. (This never writes into Live Trading or Trade History.)")

# ------------------------------------------------------------- LIVE TRADE -
with tab_live:
    st.subheader(f"Live (Paper) Trading — {ticker_choice} ({ticker})")
    st.caption("This is a simulation layer driven by the latest candle signal. Nothing polls the API until you click Start — Stop (or leaving/closing this browser tab) halts it again.")

    # ---- Which data source & fill logic are actually active right now ----
    _feed_on = dhan_feed_active()
    _dhan_servable = _feed_on and dhan_resolve_feed_instrument(ticker) is not None
    if _dhan_servable:
        st.info("📡 **Data source: Dhan data feed** — candles + live LTP with **no delay at all**. "
                "⚙️ **Fill logic:** SL/Target checks compare against the live Dhan LTP tick (not stale candle data); "
                "candle-based strategies still enter at the next candle open, immediate-execution strategies enter at LTP.")
    else:
        st.info("📡 **Data source: yfinance** — mandatory 0.3s delay per API call. "
                "⚙️ **Fill logic:** SL/Target checks compare against the freshest fetched LTP; candle-based strategies "
                "enter at the next candle open, immediate-execution strategies enter at LTP.")
    if st.session_state.get("dhan_feed_warning"):
        st.warning("⚠️ " + st.session_state.dhan_feed_warning)
    if st.session_state.get("dhan_fallback_notice"):
        st.warning("↩️ " + st.session_state.dhan_fallback_notice)
    if st.session_state.get("live_blocked_reason"):
        st.warning(f"🚧 Last entry was blocked by a risk gate: {st.session_state.live_blocked_reason}")

    # ---- Start / Stop / Square-off controls ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        if not st.session_state.live_running:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                st.session_state.live_running = True
                st.rerun()
        else:
            st.button("▶️ Running…", disabled=True, use_container_width=True)
    with ctrl2:
        if st.button("⏸ Stop", use_container_width=True, disabled=not st.session_state.live_running):
            st.session_state.live_running = False
            st.rerun()
    with ctrl3:
        manual_eval = st.button("🔄 Evaluate Once", use_container_width=True)
    with ctrl4:
        squareoff_clicked = st.button("🟥 Square Off Now", use_container_width=True, disabled=not st.session_state.live_positions)

    if st.session_state.live_running:
        st.success("🟢 Live monitoring is ON — polling the API and re-checking signals every few seconds. Click Stop to halt it.")
    else:
        st.caption("⚪ Live monitoring is OFF — no background API calls are being made. Click Start to arm it, or use Evaluate Once for a single manual check.")
    st.caption("Note: a full browser close / new session always resets this to OFF. A plain in-tab refresh (F5) may preserve the ON state since Streamlit keeps the same session — click Stop first if you want a hard reset before refreshing.")

    if squareoff_clicked and st.session_state.live_positions:
        pos = st.session_state.live_positions[0]
        raw = fetch_data(ticker, interval, period)
        ltp_now = get_live_ltp(ticker)
        exit_price = ltp_now if ltp_now is not None else (float(raw["Close"].iloc[-1]) if not raw.empty else pos["current_price"])
        points = (exit_price - pos["entry_price"]) * pos["direction"]
        _sq_row = {
            "Entry Time": pos["entry_time"], "Entry Price": round(pos["entry_price"], 2),
            "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
            "Exit Time": datetime.now(), "Exit Price": round(exit_price, 2),
            "SL": round(pos["initial_sl"], 2), "Target": round(pos["initial_target"], 2),
            "Highest": round(pos["highest"], 2), "Lowest": round(pos["lowest"], 2),
            "Points": round(points, 2), "PnL": round(points * pos["remaining_qty"], 2),
            "Exit Reason": "Manual Square Off", "Qty": pos["remaining_qty"],
        }
        if raw is not None and not raw.empty:
            _c = raw.iloc[-1]
            _sq_row.update({"Exit Open": round(float(_c["Open"]), 2), "Exit High": round(float(_c["High"]), 2),
                            "Exit Low": round(float(_c["Low"]), 2), "Exit Close": round(float(_c["Close"]), 2)})
        _live_attach_option_premiums(_sq_row, pos, config, pos["remaining_qty"], closing=True)
        st.session_state.live_history.append(_sq_row)
        st.session_state.live_positions = []
        note_trade_event()  # feeds the entry-cooldown gate
        st.warning(f"Manually squared off @ {exit_price:.2f}")
        # Manual square-offs are ALWAYS sent, even with Bracket Orders on
        # (dispatch only skips broker-managed Stoploss/Target hits).
        _sq_res = dispatch_dhan_event(config, pos["direction"], False, "Manual Square Off",
                                      pos["remaining_qty"], pos["remaining_qty"], exit_price,
                                      exit_reason="Manual Square Off")
        if _sq_res:
            st.json(_sq_res)
        email_trade_event(config, "Manual Square Off", {
            "Ticker": ticker, "Strategy": strategy,
            "Direction": _sq_row["Direction"], "Entry Price": _sq_row["Entry Price"],
            "SL": _sq_row["SL"], "Target": _sq_row["Target"],
            "Exit Price": _sq_row["Exit Price"], "Exit Reason": "Manual Square Off",
            "Points": _sq_row["Points"], "PnL": _sq_row["PnL"], "Qty": pos["remaining_qty"],
        })
        st.rerun()

    st.markdown("**Live Price & Position P&L**")
    if st.session_state.live_running:
        live_position_fragment(ticker, "LTP")
    else:
        st.caption("Stopped — no LTP polling. Click Start to resume live price and P&L updates.")

    with st.expander("Selected Configuration"):
        st.json({
            "Ticker": ticker, "Timeframe": interval, "Period": period, "Quantity": qty,
            "Strategy": strategy, "Stoploss Type": sl_type, "Target Type": target_type,
            "Filters Active": [k for k, v in filters.items() if v is True],
            "Dhan Live Orders": dhan_enabled,
            "Data Source": ("Dhan data feed (no delay)" if _dhan_servable
                            else "yfinance (0.3s delay per call)"),
            "Fill Logic": ("SL/Target vs live LTP tick; candle strategies fill at next candle open; "
                           "immediate-execution strategies fill at LTP"),
            "Dhan Product Config": product_cfg,
            "Email Notifications": config.get("email_enabled", False),
        })

    if manual_eval:
        evaluate_live_signal(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                              dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl,
                              full_cfg=config)

    if st.session_state.live_running:
        # THIS is what makes trade entry/exit actually keep happening while
        # monitoring is on: a plain TOP-LEVEL fragment (same pattern as
        # live_dashboard_fragment / live_position_fragment below), not a
        # closure nested inside this tab — nesting it was fragile and is
        # exactly what caused positions to silently stop updating before.
        live_signal_loop_fragment(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                                   dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl,
                                   full_cfg=config)

    if st.session_state.live_running:
        live_dashboard_fragment(ticker, interval, period, strategy, params, filters)
    else:
        st.caption("📊 Indicator Dashboard / 📟 Signal Status Board are paused while monitoring is OFF. Click Start to see them update live, or Evaluate Once for a single snapshot below.")
        if manual_eval:
            raw_status = fetch_data(ticker, interval, period)
            if not raw_status.empty and len(raw_status) >= 30:
                for line in describe_signal_status(raw_status, strategy, params, filters):
                    st.write("• " + line)

    st.markdown("#### Chart — EMA Overlay")
    raw_status = fetch_data(ticker, interval, period)
    if not raw_status.empty:
        chart_df = raw_status.tail(150)
        extra_lines = []
        if st.session_state.live_positions:
            pos = st.session_state.live_positions[0]
            extra_lines = [
                ("Entry", pos["entry_price"], "white", "dot"),
                ("SL", pos["sl"], "red", "dash"),
                ("Target", pos["target"], "lime", "dash"),
            ]
        st.plotly_chart(
            price_chart(chart_df, None, "Recent Price Action",
                        ema_overlay=[(params.get("ema_fast", 9), "#3399ff"), (params.get("ema_slow", 15), "#ff9933")],
                        extra_lines=extra_lines),
            use_container_width=True,
        )

    if not st.session_state.live_running:
        st.markdown("#### Open Position (static snapshot — start live monitoring for live P&L)")
        if st.session_state.live_positions:
            pos = st.session_state.live_positions[0]
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Entry Type", "LONG" if pos["direction"] == 1 else "SHORT")
            c2.metric("Entry Price", f"{pos['entry_price']:.2f}")
            c3.metric(f"SL ({pos['sl_type']})", f"{pos['sl']:.2f}")
            c4.metric(f"Target ({pos['target_type']})", f"{pos['target']:.2f}")
            c5.metric("Highest", f"{pos['highest']:.2f}")
            c6.metric("Lowest", f"{pos['lowest']:.2f}")
            c7.metric("Remaining Qty", f"{pos['remaining_qty']}/{pos['original_qty']}")
        else:
            st.caption("No open paper position.")

    recent_trades_fragment()

# ------------------------------------------------------------- TRADE HIST -
with tab_hist:
    st.subheader("Trade History (Live/Paper only — never mixed with backtest)")
    trade_history_fragment()

# ---------------------------------------------------------------- HEATMAP -
with tab_heat:
    st.subheader("Return Heatmaps")

    st.markdown("##### 1) Monthly % Returns — Configurable Lookback")
    heatmap_years = st.number_input("Years of history to show", min_value=1, max_value=30, value=10, step=1)
    st.caption("Defaults to 10 years — the full 20-year grid can get cramped and hard to read; narrow it down here.")
    if st.button(f"Generate {int(heatmap_years)}Y Monthly Heatmap"):
        with st.spinner(f"Fetching {int(heatmap_years)} years of monthly candles…"):
            time.sleep(RATE_LIMIT_DELAY)
            monthly = yf.download(ticker, interval="1mo", period=f"{int(heatmap_years)}y", progress=False, auto_adjust=True)
            if isinstance(monthly.columns, pd.MultiIndex):
                monthly.columns = monthly.columns.get_level_values(0)
        if monthly is None or monthly.empty:
            st.error("No monthly data available for this ticker.")
        else:
            monthly = monthly.dropna()
            monthly["ret_pct"] = monthly["Close"].pct_change() * 100
            monthly["Year"] = monthly.index.year
            monthly["Month"] = monthly.index.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pivot = monthly.pivot_table(index="Year", columns="Month", values="ret_pct")
            pivot = pivot.reindex(columns=month_order)
            # Symmetric zmin/zmax around 0 (rather than relying on autorange)
            # so a red-to-green diverging scale is centered correctly even
            # when one outlier month skews the raw min/max — otherwise
            # ordinary negative months can render as pale yellow instead of
            # a clearly-red color.
            vmax = np.nanmax(np.abs(pivot.values)) if np.isfinite(pivot.values).any() else 1.0
            vmax = vmax if vmax > 0 else 1.0
            fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn", zmin=-vmax, zmax=vmax, aspect="auto",
                             labels=dict(color="% return"))
            fig.update_layout(height=max(400, 32 * len(pivot) + 150), title=f"{ticker_choice} — Monthly % Returns ({int(heatmap_years)}Y)")
            st.plotly_chart(fig, use_container_width=True)

            latest_month = datetime.now().strftime("%b")
            if latest_month in pivot.columns:
                hist_avg = pivot[latest_month].mean()
                if hist_avg > 0.5:
                    st.success(f"📈 Historically, {latest_month} has averaged {hist_avg:.2f}% return over 20 years — seasonally favorable.")
                elif hist_avg < -0.5:
                    st.warning(f"📉 Historically, {latest_month} has averaged {hist_avg:.2f}% return over 20 years — seasonally weak, consider waiting.")
                else:
                    st.info(f"➖ {latest_month} has averaged {hist_avg:.2f}% historically — no strong seasonal bias.")

    st.divider()
    st.markdown(f"##### 2) % Returns Heatmap — Selected Timeframe ({interval}) & Period ({period})")
    if st.button("Generate Selected Timeframe Heatmap"):
        with st.spinner("Fetching and computing…"):
            raw = fetch_data(ticker, interval, period)
        if raw.empty:
            st.error("No data returned for this timeframe/period.")
        else:
            raw = raw.copy()
            raw["ret_pct"] = raw["Close"].pct_change() * 100
            if interval in ("1m", "5m", "15m", "1h"):
                raw["bucket_row"] = raw.index.strftime("%A")
                raw["bucket_col"] = raw.index.hour
                pivot2 = raw.pivot_table(index="bucket_row", columns="bucket_col", values="ret_pct", aggfunc="mean")
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                pivot2 = pivot2.reindex([d for d in day_order if d in pivot2.index])
                x_label, y_label = "Hour of Day", "Day of Week"
            else:
                raw["bucket_row"] = raw.index.year
                raw["bucket_col"] = raw.index.strftime("%b")
                pivot2 = raw.pivot_table(index="bucket_row", columns="bucket_col", values="ret_pct", aggfunc="mean")
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot2 = pivot2.reindex(columns=[m for m in month_order if m in pivot2.columns])
                x_label, y_label = "Month", "Year"

            vmax2 = np.nanmax(np.abs(pivot2.values)) if np.isfinite(pivot2.values).any() else 1.0
            vmax2 = vmax2 if vmax2 > 0 else 1.0
            fig2 = px.imshow(pivot2, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-vmax2, zmax=vmax2, aspect="auto",
                              labels=dict(color="avg % return", x=x_label, y=y_label))
            fig2.update_layout(height=550, title=f"{ticker_choice} — Avg % Return · {interval}/{period}")
            st.plotly_chart(fig2, use_container_width=True)

            recent_momentum = raw["ret_pct"].tail(10).mean()
            overall_bias = pivot2.mean().mean()
            if recent_momentum > 0 and overall_bias > 0:
                st.success("🟢 Recent momentum and historical bias for this slot both lean positive — favorable window to consider buying, subject to your risk rules.")
            elif recent_momentum < 0 and overall_bias < 0:
                st.warning("🔴 Recent momentum and historical bias both lean negative — better to wait for confirmation.")
            else:
                st.info("🟡 Mixed signal between recent momentum and historical bias — no strong directional edge right now, wait for confluence.")

# ------------------------------------------------------------ OPTIMIZATION -
with tab_opt:
    st.subheader("Strategy Optimizer")
    st.caption("Runs backtests across chosen strategy / timeframe / period / stoploss / target / filter combinations. Each combo triggers a rate-limited yfinance call and its own backtest loop — bigger grids take longer.")

    EXCLUDE_FROM_SELECT_ALL = {"Simple Buy Only", "Simple Sell Only", "Threshold Cross"}

    if st.button("⚡ Select all strategies (except Simple Buy Only, Simple Sell Only & Threshold Cross)"):
        st.session_state["opt_strategies_ms"] = [s for s in STRATEGIES if s not in EXCLUDE_FROM_SELECT_ALL]
        st.rerun()

    opt_strategies = st.multiselect("Strategies to test", STRATEGIES, default=[strategy], key="opt_strategies_ms")
    opt_intervals = st.multiselect("Timeframes to test", list(TF_PERIOD_MAP.keys()), default=[interval])
    combo_periods = sorted({p for iv in opt_intervals for p in TF_PERIOD_MAP[iv]}, key=lambda x: x)
    opt_periods = st.multiselect("Periods to test (only valid ones per timeframe are used)", combo_periods, default=[period] if period in combo_periods else combo_periods[:1])

    st.markdown("**Stoploss & Target combinations** (defaults to your current sidebar selection — change here without leaving this tab)")
    c1, c2 = st.columns(2)
    opt_sl_types = c1.multiselect("SL types to test", SL_TYPES, default=[sl_type])
    opt_target_types = c2.multiselect("Target types to test", TARGET_TYPES, default=[target_type])

    st.markdown("**Additional filters to optimize ON/OFF** (uses the thresholds set in the sidebar for each filter; leaves untouched filters exactly as configured there)")
    FILTER_TOGGLE_LABELS = {
        "adx_enabled": "ADX Filter", "rsi_enabled": "RSI Filter", "bb_enabled": "Bollinger Filter",
        "ema20_enabled": "EMA20 Filter", "sma20_enabled": "SMA20 Filter", "smc_enabled": "SMC Filter",
        "atr_enabled": "ATR Filter", "supertrend_enabled": "Supertrend Filter", "regime_enabled": "Regime Filter",
    }
    opt_filter_labels_chosen = st.multiselect("Filters to test both WITH and WITHOUT", list(FILTER_TOGGLE_LABELS.values()))
    toggle_keys = [k for k, v in FILTER_TOGGLE_LABELS.items() if v in opt_filter_labels_chosen]

    def build_filter_variants(base_filters, keys):
        variants = [dict(base_filters)]
        for key in keys:
            expanded = []
            for v in variants:
                v_off, v_on = dict(v), dict(v)
                v_off[key], v_on[key] = False, True
                expanded.append(v_off)
                expanded.append(v_on)
            variants = expanded
        return variants

    filter_variants = build_filter_variants(filters, toggle_keys)

    def variant_label(fv, keys):
        if not keys:
            return "sidebar default"
        return ", ".join(f"{FILTER_TOGGLE_LABELS[k]}:{'ON' if fv.get(k) else 'OFF'}" for k in keys)

    st.markdown("### 🎯 Accuracy-Targeted Optimization")
    accuracy_target_enabled = st.checkbox("Enable — only surface configs meeting a minimum accuracy", value=False)
    target_accuracy = 60.0
    if accuracy_target_enabled:
        target_accuracy = st.number_input("Minimum accuracy required (%)", 0.0, 100.0, 60.0, step=1.0)
        st.caption("After running, results are filtered to combos meeting this accuracy. If none qualify, you'll still see the best combo(s) actually found, clearly labeled as below target.")

    n_combos = (len(opt_strategies) * len(opt_sl_types) * len(opt_target_types) * len(filter_variants)
                * sum(1 for iv in opt_intervals for p in TF_PERIOD_MAP[iv] if p in opt_periods))
    st.caption(f"Estimated backtest runs: **{n_combos}** (≈{n_combos * RATE_LIMIT_DELAY:.1f}s+ just for data-fetch delays, plus backtest compute time per run).")

    MAX_COMBOS = st.number_input(
        "Safety cap on number of combinations (no upper limit — raise as high as you want, but large grids can run for a long time and are more likely to hit yfinance rate limits)",
        min_value=50, value=400, step=50,
    )
    if n_combos > MAX_COMBOS:
        st.error(f"That's {n_combos} combinations — over your current cap of {int(MAX_COMBOS)}. Either narrow your selections or raise the cap above.")

    run_disabled = n_combos == 0 or n_combos > MAX_COMBOS
    if st.button("🧪 Run Optimization", type="primary", disabled=run_disabled):
        rows = []
        progress = st.progress(0.0, text="Starting…")
        combos = [
            (s, iv, p, slt, tgt, fv)
            for s in opt_strategies
            for iv in opt_intervals for p in TF_PERIOD_MAP[iv] if p in opt_periods
            for slt in opt_sl_types
            for tgt in opt_target_types
            for fv in filter_variants
        ]
        data_cache = {}
        for idx, (s, iv, p, slt, tgt, fv) in enumerate(combos):
            cache_key = (iv, p)
            if cache_key not in data_cache:
                data_cache[cache_key] = fetch_data(ticker, iv, p)
            raw = data_cache[cache_key]
            if not raw.empty and len(raw) >= 30:
                tdf, _ = run_backtest(raw, s, slt, tgt, params, fv, qty, risk_ctrl)
                m = compute_metrics(tdf)
                rows.append({
                    "Strategy": s, "Timeframe": iv, "Period": p, "SL Type": slt, "Target Type": tgt,
                    "Filters": variant_label(fv, toggle_keys), **m,
                })
            progress.progress((idx + 1) / max(len(combos), 1), text=f"{s} · {iv}/{p} · {slt[:12]}/{tgt[:12]}")
        progress.empty()
        st.session_state.opt_results = pd.DataFrame(rows)

    results = st.session_state.opt_results
    if isinstance(results, pd.DataFrame) and not results.empty:
        rank_metric = st.selectbox("Rank by", ["accuracy", "sharpe", "expectancy", "total_pnl", "total_points"], index=1)

        working = results
        if accuracy_target_enabled:
            qualifying = results[results["accuracy"] >= target_accuracy]
            if not qualifying.empty:
                st.success(f"✅ {len(qualifying)} combination(s) meet the ≥{target_accuracy}% accuracy target.")
                working = qualifying
            else:
                best_found = results.sort_values("accuracy", ascending=False).iloc[0]
                st.warning(f"⚠️ No combination reached {target_accuracy}% accuracy. Best found: **{best_found['Strategy']}** "
                           f"· {best_found['Timeframe']}/{best_found['Period']} at **{best_found['accuracy']}%** accuracy — showing full results below anyway.")

        best_overall = working.sort_values(rank_metric, ascending=False).iloc[0]
        st.success(f"🏆 Best by {rank_metric}: **{best_overall['Strategy']}** · {best_overall['Timeframe']}/{best_overall['Period']} · "
                   f"SL: {best_overall['SL Type']} · Target: {best_overall['Target Type']} · Filters: {best_overall['Filters']} "
                   f"(accuracy {best_overall['accuracy']}%, sharpe {best_overall['sharpe']}, expectancy {best_overall['expectancy']})")
        if st.button("✅ Apply BEST overall config to sidebar"):
            apply_config_to_sidebar(best_overall)

        st.markdown("---")
        st.caption("Every combination tested for a strategy shows up as its own row below — pick any rank you want, not just #1.")
        for s in working["Strategy"].unique():
            sub = working[working["Strategy"] == s].sort_values(rank_metric, ascending=False).reset_index(drop=True)
            sub.insert(0, "Rank", range(1, len(sub) + 1))
            st.markdown(f"**{s}** — {len(sub)} combination(s) tested")
            st.dataframe(sub, use_container_width=True, hide_index=True)
            rank_choice = st.number_input(f"Apply rank # for '{s}' (1 = best)", min_value=1, max_value=len(sub), value=1, key=f"rank_{s}")
            if st.button(f"Apply rank {rank_choice} config for '{s}'", key=f"apply_{s}"):
                apply_config_to_sidebar(sub.iloc[int(rank_choice) - 1])
    else:
        st.caption("Run the optimizer to see ranked results per strategy.")

# --------------------------------------------------------------- SPREAD --
with tab_spread:
    st.subheader("Cross-Asset Spread / Difference Tool")
    st.caption("Pick two (optionally a third) instruments, compare live prices, and get a simple directional read.")

    all_names = [n for n in TICKER_MAP if n != "Custom"] + ["Custom"]
    c1, c2, c3 = st.columns(3)
    with c1:
        t1_name = st.selectbox("Ticker 1", all_names, index=0, key="sp_t1")
        t1 = st.text_input("Custom symbol 1", "RELIANCE.NS", key="sp_t1_custom") if t1_name == "Custom" else TICKER_MAP[t1_name]
    with c2:
        t2_name = st.selectbox("Ticker 2", all_names, index=2, key="sp_t2")
        t2 = st.text_input("Custom symbol 2", "TCS.NS", key="sp_t2_custom") if t2_name == "Custom" else TICKER_MAP[t2_name]
    with c3:
        use_t3 = st.checkbox("Add Ticker 3 (optional)")
        t3 = None
        if use_t3:
            t3_name = st.selectbox("Ticker 3", all_names, index=3, key="sp_t3")
            t3 = st.text_input("Custom symbol 3", "HDFCBANK.NS", key="sp_t3_custom") if t3_name == "Custom" else TICKER_MAP[t3_name]

    if st.button("🔍 Fetch & Compare"):
        def get_ltp(sym):
            time.sleep(RATE_LIMIT_DELAY)
            try:
                d = yf.Ticker(sym).history(period="1d", interval="1m")
                if d.empty:
                    d = yf.Ticker(sym).history(period="5d", interval="15m")
                return float(d["Close"].iloc[-1]) if not d.empty else None
            except Exception:
                return None

        p1, p2 = get_ltp(t1), get_ltp(t2)
        p3 = get_ltp(t3) if t3 else None

        if p1 is None or p2 is None:
            st.error("Could not fetch one or more prices — check symbols.")
        else:
            diff = p1 - p2
            st.metric(f"{t1_name} price", f"{p1:,.2f}")
            st.metric(f"{t2_name} price", f"{p2:,.2f}")
            st.metric("Difference (T1 − T2)", f"{diff:,.2f}")
            if p3 is not None:
                st.metric(f"{t3_name} price", f"{p3:,.2f}")
                st.metric("Difference (T1 − T3)", f"{p1 - p3:,.2f}")

            st.markdown("---")
            st.markdown("#### 📊 Historical Bin Analysis")
            st.caption("Where does today's difference sit relative to its own history, and how did each ticker behave afterwards when the difference was in that same range?")

            render_bin_analysis_section(t1, t2, t1_name, t2_name, p1, diff, "1d", "2y",
                                         section_label="Baseline: Daily candles, 2-year history", fwd_n=5)

            st.markdown("---")
            render_bin_analysis_section(t1, t2, t1_name, t2_name, p1, diff, interval, period,
                                         section_label=f"Matched to your sidebar selection: {interval} candles, {period} history", fwd_n=5)

# ---------------------------------------------------------------- OHLC/RANGE
with tab_ohlc:
    st.subheader(f"OHLC & Range Insights — {ticker_choice} ({ticker})")
    st.caption("Raw candle data with per-bar % and absolute change, plus a plain-language read on where price sits in its range. Descriptive/statistical framing, not a prediction.")

    render_range_insight_section(ticker, "1d", "1y", "1) Fixed baseline: Daily candles, past 1 year")

    st.markdown("---")
    render_range_insight_section(ticker, interval, period, f"2) Matched to your sidebar selection: {interval} candles, {period}")

# ---------------------------------------------------------------- ADMIN ----
with tab_admin:
    # Full-width rendering of EVERY sidebar control. This tab and the sidebar
    # are two live views of the same shared config store — changing a value
    # here updates the sidebar instantly, and vice versa. Widget keys are
    # prefixed ("ad" vs "sb") so Streamlit never collides, while both write
    # into st.session_state.app_cfg via the cfg_* wrappers.
    render_config_controls(st.container(), "ad")

# ============================================================================
# FOOTER / GLOBAL DISCLAIMER
# ============================================================================

st.divider()
st.caption(
    "⚠️ Educational tool. Backtests use simplified conservative fill logic and ignore slippage, "
    "brokerage, taxes, and liquidity constraints — real results will differ. Verify any strategy on "
    "out-of-sample data and paper-trade before committing capital. 🏦 The Dhan integration performs "
    "REAL network calls: with Order Placement enabled and a valid access token, orders are actually "
    "sent to Dhan's live API; without a token they are only SIMULATED (payload shown, nothing sent). "
    "Always validate credentials and behavior in a sandbox before going live."
)
