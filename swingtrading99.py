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
import sqlite3
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
    "OI Based (CE/PE Open Interest)",
    "OI Change Based (ΔOI)",
    "OI + Volume Change Based",
    "PCR Based (Put-Call Ratio)",
    "Gamma Blast (Expiry Momentum)",
    "Multi-Strike OI (ATM ± N Levels)",
    "Hybrid (Combine Strategies)",
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
    "OI Based (CE/PE Open Interest)": "neutral",
    "OI Change Based (ΔOI)": "neutral",
    "OI + Volume Change Based": "neutral",
    "PCR Based (Put-Call Ratio)": "neutral",
    "Gamma Blast (Expiry Momentum)": "neutral",
    "Multi-Strike OI (ATM ± N Levels)": "neutral",
    "Hybrid (Combine Strategies)": "neutral",
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
# The OI strategy belongs here too: option-chain Open Interest is a live
# snapshot, not a candle-derived series, so there is nothing to wait for —
# and reading it from the last CLOSED candle (as candle strategies do) would
# always see zero, which is why it produced a visible signal but no entry.
IMMEDIATE_EXECUTION_STRATEGIES = {"Simple Buy Only", "Simple Sell Only", "Threshold Cross",
                                  "OI Based (CE/PE Open Interest)",
                                  "OI Change Based (ΔOI)",
                                  "OI + Volume Change Based",
                                  "PCR Based (Put-Call Ratio)",
                                  "Gamma Blast (Expiry Momentum)",
                                  "Multi-Strike OI (ATM ± N Levels)"}

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
EMAIL_DEFAULT_TO = "srinivasp451@gmail.com"

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
# Dhan's intraday chart API accepts 1, 5, 15, 25 and 60 minute intervals —
# finer granularity than yfinance offers. The extra grains are ADDITIVE: the
# base yfinance timeframes keep working exactly as before, and the Dhan-only
# ones simply become selectable when the Dhan feed is active.
DHAN_INTERVAL_CODE = {"1m": "1", "2m": "1", "3m": "1", "5m": "5", "10m": "5",
                      "15m": "15", "25m": "25", "30m": "15", "45m": "15",
                      "60m": "60", "1h": "60", "2h": "60", "4h": "60"}

# Dhan-only timeframes → the periods that make sense for them. Merged into
# TF_PERIOD_MAP only when the Dhan feed is active, so the yfinance-only
# experience is unchanged.
DHAN_EXTRA_TF_PERIODS = {
    "2m": ["1d", "5d", "7d", "1mo"],
    "3m": ["1d", "5d", "7d", "1mo"],
    "10m": ["1d", "5d", "7d", "1mo", "3mo"],
    "25m": ["1d", "5d", "7d", "1mo", "3mo"],
    "30m": ["1d", "5d", "7d", "1mo", "3mo", "6mo"],
    "45m": ["1d", "5d", "7d", "1mo", "3mo", "6mo"],
    "2h": ["5d", "7d", "1mo", "3mo", "6mo", "1y"],
    "4h": ["7d", "1mo", "3mo", "6mo", "1y", "2y"],
}

# Timeframes that Dhan serves by resampling a finer base interval, e.g. a 3m
# candle is built from 1m data. Keyed by timeframe → (base code, pandas rule).
DHAN_RESAMPLE_TF = {"2m": "2min", "3m": "3min", "10m": "10min", "30m": "30min",
                    "45m": "45min", "2h": "2h", "4h": "4h"}


def available_tf_period_map():
    """TF_PERIOD_MAP, extended with Dhan-only granularities when that feed is
    active. Callers keep using one map, so nothing downstream changes."""
    base = {k: list(v) for k, v in TF_PERIOD_MAP.items()}
    try:
        if dhan_feed_active():
            for tf, periods in DHAN_EXTRA_TF_PERIODS.items():
                base.setdefault(tf, list(periods))
    except Exception:
        pass
    # keep a sensible ascending order rather than dict insertion order
    def _mins(tf):
        try:
            if tf.endswith("m"):
                return int(tf[:-1])
            if tf.endswith("h"):
                return int(tf[:-1]) * 60
            if tf == "1d":
                return 60 * 24
            if tf == "1wk":
                return 60 * 24 * 7
        except Exception:
            pass
        return 10 ** 6
    return {k: base[k] for k in sorted(base, key=_mins)}

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
# CONFIG WIDGETS — SINGLE OWNER, SINGLE RENDER
# ----------------------------------------------------------------------------
# There is exactly ONE set of configuration widgets in this app: the sidebar.
# Streamlit owns each widget's value through its session key, and this module
# never re-assigns that key once it exists (see _cfg_init). st.session_state
# .app_cfg is a read-only MIRROR of those values for non-widget consumers
# (order routing, the Optimization tab, config snapshots) — the mirror never
# drives the widgets.
#
# Why it's built this way: earlier versions rendered the same controls twice
# (sidebar + a duplicate Admin Panel) and tried to keep the two in sync by
# writing widget keys on every run. Because the two views render one after
# the other, each run wrote a value the user had not chosen, and selections
# visibly snapped back to their previous state. The duplicate rendering has
# been removed entirely — the Admin Panel is now a read-only full-width
# summary — which eliminates that entire class of bug at the source rather
# than trying to arbitrate a race.
# ============================================================================

_CFG_PREFIX = "w_"


def _cfg_store():
    return st.session_state.app_cfg


def _wkey(cfg_key):
    return _CFG_PREFIX + cfg_key


def _cfg_init(cfg_key, default, coerce=None):
    """
    Initialise a widget's session key ONCE, then never touch it again.

    This is the single most important rule in this file. Streamlit already
    owns a widget's value once the widget exists, and it survives reruns by
    itself. Any code that re-assigns that key on later runs is fighting the
    user: whatever they just picked gets overwritten by whatever the code
    thinks the value should be, which is exactly how a selection appears to
    "snap back" to the previous one. So the ONLY writes here are:
      • first-ever creation (seed from the stored value or the default), and
      • a coercion when the live option list no longer contains the value
        (e.g. the period list changed with the timeframe) — otherwise
        Streamlit itself would raise.
    Everything else is read-only as far as widget state is concerned.
    """
    store = _cfg_store()
    wkey = _wkey(cfg_key)
    if wkey not in st.session_state:
        seed = store.get(cfg_key, default)
        st.session_state[wkey] = coerce(seed) if coerce else seed
    elif coerce is not None:
        fixed = coerce(st.session_state[wkey])
        if fixed != st.session_state[wkey]:
            st.session_state[wkey] = fixed      # option list changed under us
    return wkey


def _cfg_out(cfg_key, val):
    """Mirror the widget's value into the plain-dict store. The store is a
    read-only projection used by non-widget code (order routing, the
    Optimization tab, saved snapshots) — it never drives the widgets."""
    _cfg_store()[cfg_key] = val
    return val


def cfg_force(cfg_key, value):
    """
    Programmatic write (auto-filled security IDs, exchange auto-flip,
    apply-optimized-config). Legal because every caller runs BEFORE the
    corresponding widget is instantiated on this run; Streamlit only forbids
    assigning to a widget key AFTER its widget has been created in the same
    run. Used sparingly and only for values the user did not just type.
    """
    st.session_state[_wkey(cfg_key)] = value
    _cfg_store()[cfg_key] = value


def cfg_checkbox(ui, label, cfg_key, default=False, prefix="sb", **kw):
    wkey = _cfg_init(cfg_key, bool(default), coerce=lambda v: bool(v))
    return _cfg_out(cfg_key, ui.checkbox(label, key=wkey, **kw))


def cfg_selectbox(ui, label, cfg_key, options, default=None, prefix="sb", **kw):
    options = list(options)
    if not options:
        return None
    if default is None or default not in options:
        default = options[0]
    wkey = _cfg_init(cfg_key, default, coerce=lambda v: v if v in options else default)
    return _cfg_out(cfg_key, ui.selectbox(label, options, key=wkey, **kw))


def cfg_multiselect(ui, label, cfg_key, options, default=None, prefix="sb", **kw):
    options = list(options)
    wkey = _cfg_init(cfg_key, list(default or []),
                     coerce=lambda v: [c for c in (v or []) if c in options])
    return _cfg_out(cfg_key, list(ui.multiselect(label, options, key=wkey, **kw)))


def cfg_number(ui, label, cfg_key, default, min_value=None, max_value=None,
               step=None, is_int=False, prefix="sb", **kw):
    def _coerce(v):
        try:
            v = int(v) if is_int else float(v)
        except (TypeError, ValueError):
            v = int(default) if is_int else float(default)
        if min_value is not None:
            v = max(v, min_value)
        if max_value is not None:
            v = min(v, max_value)
        return v

    wkey = _cfg_init(cfg_key, _coerce(default), coerce=_coerce)
    return _cfg_out(cfg_key, ui.number_input(label, min_value=min_value, max_value=max_value,
                                             step=step, key=wkey, **kw))


def cfg_text(ui, label, cfg_key, default="", prefix="sb", **kw):
    wkey = _cfg_init(cfg_key, str(default), coerce=lambda v: "" if v is None else str(v))
    return _cfg_out(cfg_key, ui.text_input(label, key=wkey, **kw))


def cfg_slider(ui, label, cfg_key, min_value, max_value, default, step=None, prefix="sb", **kw):
    def _coerce(v):
        try:
            v = type(default)(v)
        except (TypeError, ValueError):
            v = default
        return max(min(v, max_value), min_value)

    wkey = _cfg_init(cfg_key, default, coerce=_coerce)
    return _cfg_out(cfg_key, ui.slider(label, min_value, max_value, key=wkey, step=step, **kw))


def cfg_time(ui, label, cfg_key, default, prefix="sb", **kw):
    def _coerce(v):
        if isinstance(v, dtime):
            return v
        try:
            hh, mm = str(v).split(":")[:2]
            return dtime(int(hh), int(mm))
        except Exception:
            return default

    wkey = _cfg_init(cfg_key, default, coerce=_coerce)
    return _cfg_out(cfg_key, ui.time_input(label, key=wkey, **kw))


def cfg_set(cfg_key, value):
    """Programmatic write used by the Optimization tab's apply-config."""
    cfg_force(cfg_key, value)


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
#   • VWAP .................... session-anchored: resets at each trading day's
#     first bar, matching TV's built-in VWAP (a whole-window cumulative VWAP
#     would drift away from TV on multi-day intraday data).
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
    """Session-anchored VWAP (TradingView convention): the cumulative
    typical-price×volume / volume RESETS at the start of each trading day,
    so gap-up/gap-down opens start a fresh session anchor exactly like TV's
    built-in VWAP. Falls back to a whole-window cumulative VWAP if the index
    isn't date-aware (shouldn't happen with yfinance/Dhan data)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = tp * df["Volume"]
    try:
        day = pd.Index(df.index).normalize()
        cum_pv = pv.groupby(day).cumsum()
        cum_v = df["Volume"].groupby(day).cumsum()
        return cum_pv / cum_v.replace(0, np.nan)
    except Exception:
        return pv.cumsum() / df["Volume"].cumsum().replace(0, np.nan)


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


def elliott_wave_state(df, lookback=3):
    """
    Zigzag pivot detection with EXPLICIT confirmation timing plus Elliott
    wave labelling.

    A swing pivot at bar i is only *knowable* once `lookback` bars have
    printed to its right. This function therefore returns, separately:
      • raw_high / raw_low   — where the pivot actually sits (for plotting)
      • confirm_high/low     — the bar where that pivot became knowable
                               (this is what signals are allowed to use)
      • pivot_price/kind     — the confirmed pivot's price and H/L type
      • wave_label           — running Elliott count (1..5 then A/B/C) applied
                               to the alternating confirmed pivot sequence
      • higher_low/lower_high— structure flags for impulse-only filtering
      • bars_to_confirm      — how many more bars the newest provisional
                               pivot still needs (drives the status board)
    """
    n = len(df)
    highs, lows = df["High"], df["Low"]
    raw_high = pd.Series(False, index=df.index)
    raw_low = pd.Series(False, index=df.index)
    confirm_high = pd.Series(False, index=df.index)
    confirm_low = pd.Series(False, index=df.index)
    pivot_price = pd.Series(np.nan, index=df.index)
    pivot_kind = pd.Series("", index=df.index)
    wave_label = pd.Series("", index=df.index)
    higher_low = pd.Series(False, index=df.index)
    lower_high = pd.Series(False, index=df.index)

    if n < (2 * lookback + 2):
        return {"raw_high": raw_high, "raw_low": raw_low,
                "confirm_high": confirm_high, "confirm_low": confirm_low,
                "pivot_price": pivot_price, "pivot_kind": pivot_kind,
                "wave_label": wave_label, "higher_low": higher_low,
                "lower_high": lower_high, "pivots": [], "bars_to_confirm": None,
                "provisional": None}

    seq = []          # confirmed alternating pivots: (idx, price, kind)
    labels = ["1", "2", "3", "4", "5", "A", "B", "C"]
    lab_i = 0

    for i in range(lookback, n - lookback):
        win_h = highs.iloc[i - lookback: i + lookback + 1]
        win_l = lows.iloc[i - lookback: i + lookback + 1]
        is_h = highs.iloc[i] == win_h.max()
        is_l = lows.iloc[i] == win_l.min()
        if is_h and is_l:                       # inside bar cluster — ambiguous
            continue
        ci = i + lookback                       # confirmation bar index
        if ci >= n:
            continue
        if is_h:
            raw_high.iloc[i] = True
            # enforce alternation: a new high replaces a weaker previous high
            if seq and seq[-1][2] == "H":
                if highs.iloc[i] > seq[-1][1]:
                    seq[-1] = (i, float(highs.iloc[i]), "H")
                continue
            prev_h = next((p for p in reversed(seq) if p[2] == "H"), None)
            if prev_h is not None and highs.iloc[i] < prev_h[1]:
                lower_high.iloc[ci] = True
            seq.append((i, float(highs.iloc[i]), "H"))
            confirm_high.iloc[ci] = True
            pivot_price.iloc[ci] = float(highs.iloc[i])
            pivot_kind.iloc[ci] = "H"
            wave_label.iloc[ci] = labels[lab_i % len(labels)]
            lab_i += 1
        elif is_l:
            raw_low.iloc[i] = True
            if seq and seq[-1][2] == "L":
                if lows.iloc[i] < seq[-1][1]:
                    seq[-1] = (i, float(lows.iloc[i]), "L")
                continue
            prev_l = next((p for p in reversed(seq) if p[2] == "L"), None)
            if prev_l is not None and lows.iloc[i] > prev_l[1]:
                higher_low.iloc[ci] = True
            seq.append((i, float(lows.iloc[i]), "L"))
            confirm_low.iloc[ci] = True
            pivot_price.iloc[ci] = float(lows.iloc[i])
            pivot_kind.iloc[ci] = "L"
            wave_label.iloc[ci] = labels[lab_i % len(labels)]
            lab_i += 1

    # ---- provisional (not yet confirmed) pivot forming in the final bars ----
    provisional, bars_to_confirm = None, None
    tail_start = max(0, n - 1 - 2 * lookback)
    for i in range(n - 1 - lookback, tail_start - 1, -1):
        if i < lookback:
            break
        win_h = highs.iloc[max(0, i - lookback): i + lookback + 1]
        win_l = lows.iloc[max(0, i - lookback): i + lookback + 1]
        right = n - 1 - i
        if right >= lookback:
            break
        if highs.iloc[i] == win_h.max():
            provisional, bars_to_confirm = ("H", float(highs.iloc[i]), df.index[i]), lookback - right
            break
        if lows.iloc[i] == win_l.min():
            provisional, bars_to_confirm = ("L", float(lows.iloc[i]), df.index[i]), lookback - right
            break

    return {"raw_high": raw_high, "raw_low": raw_low,
            "confirm_high": confirm_high, "confirm_low": confirm_low,
            "pivot_price": pivot_price, "pivot_kind": pivot_kind,
            "wave_label": wave_label, "higher_low": higher_low,
            "lower_high": lower_high, "pivots": seq,
            "bars_to_confirm": bars_to_confirm, "provisional": provisional}


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
def _session_fit(idx):
    """Fraction of timestamps that fall inside the 09:15–15:30 IST session."""
    try:
        t = pd.DatetimeIndex(idx).time
        if len(t) == 0:
            return 0.0
        inside = sum(1 for x in t if dtime(9, 15) <= x <= dtime(15, 30))
        return inside / len(t)
    except Exception:
        return 0.0


def normalize_index_to_ist(df, ticker):
    """
    Force a candle index to NAIVE IST.

    Two different failure modes produced UTC timestamps in the UI:
      • tz-AWARE data (Dhan, some yfinance responses) simply needs converting.
      • tz-NAIVE data — yfinance returns this for several interval/ticker
        combinations — carries no marker at all, and earlier code left it
        untouched, so UTC values were displayed verbatim (a 13:01 IST bar
        showing as 07:31).

    A naive index is therefore tested against the actual Indian session rather
    than trusted: whichever of "as-is" or "shifted by +5:30" puts more bars
    inside 09:15–15:30 wins. That is self-correcting — if a feed later starts
    returning proper IST, the as-is reading fits better and nothing is shifted.
    Daily/weekly bars carry no meaningful time-of-day, so they are left alone.
    """
    if df is None or df.empty:
        return df
    try:
        idx = pd.DatetimeIndex(df.index)
    except Exception:
        return df

    if idx.tz is not None:
        df = df.copy()
        df.index = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        return df

    # naive: only intraday Indian data can be diagnosed by session fit
    try:
        intraday = len(set(idx.time)) > 3
    except Exception:
        intraday = False
    if not (intraday and is_indian_ticker(None, ticker)):
        return df

    as_is = _session_fit(idx)
    shifted_idx = idx + pd.Timedelta(hours=5, minutes=30)
    shifted = _session_fit(shifted_idx)
    if shifted > as_is + 0.05:          # clearly a UTC series → convert to IST
        df = df.copy()
        df.index = shifted_idx
    return df


# Dhan-only grains → (yfinance base interval, resample rule) so that turning
# the Dhan feed off never leaves a selected timeframe unusable.
_YF_FALLBACK_TF = {"2m": ("1m", "2min"), "3m": ("1m", "3min"), "10m": ("5m", "10min"),
                   "25m": ("5m", "25min"), "30m": ("15m", "30min"), "45m": ("15m", "45min"),
                   "60m": ("1h", None), "2h": ("1h", "2h"), "4h": ("1h", "4h")}


def fetch_data_yf(ticker, interval, period):
    """Original yfinance candle fetch — logic unchanged, mandatory delay kept.
    Dhan-only granularities are served by resampling the nearest yfinance
    interval, so switching the feed off never breaks the current selection."""
    resample_rule = None
    if interval in _YF_FALLBACK_TF:
        interval, resample_rule = _YF_FALLBACK_TF[interval]
    time.sleep(RATE_LIMIT_DELAY)
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    if resample_rule and not df.empty:
        try:
            df = df.resample(resample_rule).agg({"Open": "first", "High": "max", "Low": "min",
                                                 "Close": "last", "Volume": "sum"})
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
        except Exception:
            pass
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
    # Option-chain strategies (OI, ΔOI, PCR, gamma, multi-strike) read Dhan's
    # option-chain API and cannot work without it, so selecting one implies the
    # feed — the user should not have to tick a separate checkbox first.
    chain_strategy = cfg.get("strategy") in OPTION_CHAIN_STRATEGIES
    if not (cfg.get("use_dhan_feed") or options_mode or chain_strategy):
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
        # `idx` is ALREADY tz-aware IST. Using idx.values here would hand back the
        # underlying UTC instants as naive values, and DatetimeIndex(..., tz=...)
        # LOCALISES naive input rather than converting it — stamping UTC wall
        # times as if they were IST and shifting every candle 5h30m earlier
        # (a 13:01 IST bar displayed as 07:31). Wrap the tz-aware index directly.
        df.index = pd.DatetimeIndex(idx)
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        if interval == "1wk" and not df.empty:
            df = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                           "Close": "last", "Volume": "sum"}).dropna()
        elif interval in DHAN_RESAMPLE_TF and not df.empty:
            # Dhan serves 1/5/15/25/60m natively; the intermediate grains are
            # built by resampling the nearest finer base interval.
            df = df.resample(DHAN_RESAMPLE_TF[interval]).agg(
                {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}).dropna(how="all")
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
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
                return normalize_index_to_ist(dhan_df, ticker)
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
                return normalize_index_to_ist(dhan_df, ticker)
            # Empty Dhan response (off-hours gap, API hiccup) → fall through to yfinance
    return normalize_index_to_ist(fetch_data_yf(ticker, interval, period), ticker)


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


@st.cache_data(ttl=60, show_spinner=False)
def dhan_get_option_chain(under_security_id, under_segment, expiry, _token_fp):
    """
    Dhan option chain (/v2/optionchain) → aggregate CE/PE Open Interest,
    OI change, VOLUME and volume change, plus the full PER-STRIKE table
    (OI, previous OI, volume, previous volume, LTP, IV and greeks) that the
    multi-strike, max-pain and gamma strategies need. Cached 60s (Dhan
    rate-limits this endpoint and OI is not a tick-level number anyway).
    Returns None when unavailable.
    """
    try:
        resp = requests.post(
            f"{DHAN_API_BASE}/optionchain", headers=_dhan_headers(),
            json={"UnderlyingScrip": int(under_security_id),
                  "UnderlyingSeg": under_segment, "Expiry": expiry},
            timeout=20,
        )
        resp.raise_for_status()
        data = (resp.json() or {}).get("data", {}) or {}
        oc = data.get("oc", {}) or {}

        def _f(d, *names):
            for nm in names:
                if nm in d and d[nm] is not None:
                    try:
                        return float(d[nm])
                    except (TypeError, ValueError):
                        pass
            return 0.0

        strikes = {}
        ce_oi = pe_oi = ce_prev = pe_prev = 0.0
        ce_vol = pe_vol = ce_pvol = pe_pvol = 0.0
        for strike_raw, legs in oc.items():
            try:
                strike = float(strike_raw)
            except (TypeError, ValueError):
                continue
            ce, pe = legs.get("ce") or {}, legs.get("pe") or {}
            c_greek, p_greek = ce.get("greeks") or {}, pe.get("greeks") or {}
            row = {
                "ce_oi": _f(ce, "oi"), "pe_oi": _f(pe, "oi"),
                "ce_prev_oi": _f(ce, "previous_oi", "previousOi"),
                "pe_prev_oi": _f(pe, "previous_oi", "previousOi"),
                "ce_vol": _f(ce, "volume"), "pe_vol": _f(pe, "volume"),
                "ce_prev_vol": _f(ce, "previous_volume", "previousVolume"),
                "pe_prev_vol": _f(pe, "previous_volume", "previousVolume"),
                "ce_ltp": _f(ce, "last_price"), "pe_ltp": _f(pe, "last_price"),
                "ce_iv": _f(ce, "implied_volatility", "impliedVolatility"),
                "pe_iv": _f(pe, "implied_volatility", "impliedVolatility"),
                "ce_gamma": _f(c_greek, "gamma"), "pe_gamma": _f(p_greek, "gamma"),
                "ce_delta": _f(c_greek, "delta"), "pe_delta": _f(p_greek, "delta"),
                "ce_vega": _f(c_greek, "vega"), "pe_vega": _f(p_greek, "vega"),
                "ce_theta": _f(c_greek, "theta"), "pe_theta": _f(p_greek, "theta"),
            }
            row["ce_oi_change"] = row["ce_oi"] - row["ce_prev_oi"]
            row["pe_oi_change"] = row["pe_oi"] - row["pe_prev_oi"]
            row["ce_vol_change"] = row["ce_vol"] - row["ce_prev_vol"]
            row["pe_vol_change"] = row["pe_vol"] - row["pe_prev_vol"]
            strikes[strike] = row
            ce_oi += row["ce_oi"]; pe_oi += row["pe_oi"]
            ce_prev += row["ce_prev_oi"]; pe_prev += row["pe_prev_oi"]
            ce_vol += row["ce_vol"]; pe_vol += row["pe_vol"]
            ce_pvol += row["ce_prev_vol"]; pe_pvol += row["pe_prev_vol"]

        if ce_oi == 0 and pe_oi == 0:
            return None
        return {
            "ce_oi": ce_oi, "pe_oi": pe_oi,
            "ce_oi_change": ce_oi - ce_prev, "pe_oi_change": pe_oi - pe_prev,
            "ce_volume": ce_vol, "pe_volume": pe_vol,
            "ce_volume_change": ce_vol - ce_pvol, "pe_volume_change": pe_vol - pe_pvol,
            "pcr": (pe_oi / ce_oi) if ce_oi else None,
            "pcr_volume": (pe_vol / ce_vol) if ce_vol else None,
            "underlying": data.get("last_price"),
            "expiry": expiry,
            "strikes": strikes,
            "fetched_at": ist_now().strftime("%H:%M:%S IST"),
            "fetched_ts": time.time(),
        }
    except Exception:
        return None


def get_chain_snapshot(underlying_name, expiry=None):
    """Fetch a chain snapshot for an EXPLICIT underlying/expiry. Used by the
    Option Chain Analysis tab so it works regardless of which strategy is
    selected in the sidebar."""
    meta = DHAN_INDEX_MAP.get(underlying_name)
    if not meta:
        return None
    _, token = _dhan_creds()
    if not token:
        return None
    if not expiry:
        exps = dhan_get_expiries(meta["underlying"], "OPTIDX", meta["exchange"])
        expiry = exps[0] if exps else None
    if not expiry:
        return None
    return dhan_get_option_chain(meta["security_id"], meta["segment"], expiry,
                                 hash(token) % 10_000_019)


def get_oi_snapshot():
    """Resolve the configured OI underlying + expiry and pull a live snapshot."""
    store = st.session_state.app_cfg
    return get_chain_snapshot(store.get("oi_underlying", "Nifty50"), store.get("oi_expiry"))


def evaluate_oi_signal(params, snap):
    """
    OI-based signal from the aggregate option-chain picture.

    Rule: when one side's absolute OI exceeds the other AND that side's
    CHANGE in OI is also larger, with both sides clearing their configured
    minimum OI thresholds → that side is 'dominant' and fires a signal.

    Interpretation: OI is written from the SELLER's perspective, so heavy CE
    writing (CE OI > PE OI with ΔCE > ΔPE) is conventionally read as bearish
    — resistance being built overhead — which by default BUYS PE. The
    'Flip OI interpretation' checkbox reverses that mapping so you can trade
    whichever reading you believe, rather than the app forcing one on you.

    Returns (signal, [explanation lines]).
    """
    if not snap:
        return 0, ["OI data unavailable — needs a valid Dhan token, a resolvable expiry, and live market hours."]
    ce_oi, pe_oi = snap["ce_oi"], snap["pe_oi"]
    d_ce, d_pe = snap["ce_oi_change"], snap["pe_oi_change"]
    min_ce = float(params.get("oi_ce_threshold", 0.0))
    min_pe = float(params.get("oi_pe_threshold", 0.0))
    flip = bool(params.get("oi_flip", False))

    base = (f"CE OI {ce_oi:,.0f} (min {min_ce:,.0f}) vs PE OI {pe_oi:,.0f} (min {min_pe:,.0f}); "
            f"ΔCE {d_ce:+,.0f} vs ΔPE {d_pe:+,.0f}")
    if snap.get("pcr"):
        base += f"; PCR {snap['pcr']:.2f}"
    lines = [base]

    if not (ce_oi >= min_ce and pe_oi >= min_pe):
        lines.append("❌ Minimum OI thresholds not met → no signal.")
        return 0, lines

    ce_dominant = (ce_oi > pe_oi) and (d_ce > d_pe)
    pe_dominant = (pe_oi > ce_oi) and (d_pe > d_ce)
    reading = "flipped reading" if flip else "standard seller-perspective reading"

    if ce_dominant:
        sig = 1 if flip else -1
        lines.append(f"✅ CE side dominant (CE OI > PE OI and ΔCE > ΔPE) → "
                     f"{'LONG / BUY CE' if flip else 'SHORT bias → BUY PE'} ({reading}).")
        return sig, lines
    if pe_dominant:
        sig = -1 if flip else 1
        lines.append(f"✅ PE side dominant (PE OI > CE OI and ΔPE > ΔCE) → "
                     f"{'SHORT bias → BUY PE' if flip else 'LONG / BUY CE'} ({reading}).")
        return sig, lines
    lines.append("❌ Neither side dominant on BOTH absolute OI and OI change → no signal.")
    return 0, lines


# ============================================================================
# OPTION-CHAIN ANALYTICS (shared by every chain-based strategy)
# ============================================================================

# Every one of these reads a LIVE option-chain snapshot rather than a candle
# series, so they all enter immediately at LTP (see
# IMMEDIATE_EXECUTION_STRATEGIES) and none of them can be backtested — Dhan
# exposes only the current chain, not historical OI/volume.
OPTION_CHAIN_STRATEGIES = {
    "OI Based (CE/PE Open Interest)",
    "OI Change Based (ΔOI)",
    "OI + Volume Change Based",
    "PCR Based (Put-Call Ratio)",
    "Gamma Blast (Expiry Momentum)",
    "Multi-Strike OI (ATM ± N Levels)",
}


def _side_dominance(ce_val, pe_val, mode, n_mult, return_detail=False):
    """
    Which side wins, under either comparison mode:
      • "Absolute"    → simply the larger value.
      • "N× multiple" → EITHER ratio may satisfy the threshold. Both are
                        computed explicitly and symmetrically:
                            CE/PE  (is CE writing n× the PE writing?)
                            PE/CE  (is PE writing n× the CE writing?)
                        With n = 5, ΔCE 600k vs ΔPE 100k gives CE/PE = 6× → CE
                        dominant; ΔCE 100k vs ΔPE 600k gives PE/CE = 6× → PE
                        dominant. Whichever ratio clears n wins; if somehow
                        both do, the LARGER ratio wins rather than whichever
                        happens to be tested first.

    Only a RISING side can dominate — a falling ΔOI is unwinding, not position
    building. When one side is rising and the other is flat or falling, the
    ratio is mathematically undefined (division by zero or a negative), so
    that case is treated as one-sided dominance in its own right rather than
    being forced through the n× test, and the reason string says so. The
    previous version floored the denominator at 1e-9, which made *any* rise on
    one side pass *any* n× threshold — a false positive that also short-
    circuited before the opposite ratio was ever considered.

    Returns "CE" / "PE" / None, or (side, detail_dict) when return_detail.
    """
    n = max(float(n_mult or 1.0), 1.0)
    ce_val, pe_val = float(ce_val or 0.0), float(pe_val or 0.0)
    detail = {"n": n, "ce": ce_val, "pe": pe_val, "ce_over_pe": None,
              "pe_over_ce": None, "basis": None}

    def _out(side):
        return (side, detail) if return_detail else side

    if str(mode).startswith("N"):
        both_positive = ce_val > 0 and pe_val > 0
        if both_positive:
            r_ce = ce_val / pe_val          # CE writing relative to PE writing
            r_pe = pe_val / ce_val          # PE writing relative to CE writing
            detail["ce_over_pe"], detail["pe_over_ce"] = r_ce, r_pe
            ce_ok, pe_ok = r_ce >= n, r_pe >= n
            if ce_ok and pe_ok:             # only possible when n <= 1
                detail["basis"] = "both ratios clear the threshold; larger ratio wins"
                return _out("CE" if r_ce >= r_pe else "PE")
            if ce_ok:
                detail["basis"] = f"CE/PE = {r_ce:.2f}× ≥ {n:.2f}×"
                return _out("CE")
            if pe_ok:
                detail["basis"] = f"PE/CE = {r_pe:.2f}× ≥ {n:.2f}×"
                return _out("PE")
            detail["basis"] = (f"neither ratio reaches {n:.2f}× "
                               f"(CE/PE = {r_ce:.2f}×, PE/CE = {r_pe:.2f}×)")
            return _out(None)

        # exactly one side rising → one-sided build, ratio undefined
        if ce_val > 0 >= pe_val:
            detail["basis"] = ("CE is building while PE is flat or unwinding — one-sided, "
                               "so the n× ratio does not apply")
            return _out("CE")
        if pe_val > 0 >= ce_val:
            detail["basis"] = ("PE is building while CE is flat or unwinding — one-sided, "
                               "so the n× ratio does not apply")
            return _out("PE")
        detail["basis"] = "neither side is building"
        return _out(None)

    # ---- Absolute mode ----
    if ce_val > 0 and pe_val > 0:
        detail["ce_over_pe"] = ce_val / pe_val
        detail["pe_over_ce"] = pe_val / ce_val
    if ce_val > pe_val and ce_val > 0:
        detail["basis"] = "CE change is larger"
        return _out("CE")
    if pe_val > ce_val and pe_val > 0:
        detail["basis"] = "PE change is larger"
        return _out("PE")
    detail["basis"] = "no side is larger and rising"
    return _out(None)


def _ratio_x(a, b):
    """Safe 'n×' ratio for display."""
    try:
        a, b = float(a), float(b)
        if abs(b) < 1e-9:
            return None
        return a / b
    except (TypeError, ValueError):
        return None


def _chain_side_to_signal(side, flip, ce_reading="bearish"):
    """
    Map a dominant chain side to a trade direction.

    OI is written from the SELLER's side, so heavy CE writing is conventionally
    read as bearish (resistance overhead) → BUY PE → SHORT signal. The flip
    checkbox inverts that mapping so you can trade the opposite reading.
    """
    if side is None:
        return 0
    base = -1 if side == "CE" else 1          # CE dominant → short bias
    if ce_reading == "bullish":
        base = -base
    return -base if flip else base


def compute_max_pain(strikes):
    """
    Max pain = the strike where option WRITERS lose the least if expiry
    settled there, i.e. the strike minimising total intrinsic payout:
        pain(K) = Σ_S [ CE_OI(S) · max(0, K−S) ] + Σ_S [ PE_OI(S) · max(0, S−K) ]
    Price is often argued to gravitate toward it near expiry.
    """
    if not strikes:
        return None, {}
    ks = sorted(strikes.keys())
    pain = {}
    for k in ks:
        total = 0.0
        for s in ks:
            row = strikes[s]
            if k > s:
                total += row["ce_oi"] * (k - s)
            if s > k:
                total += row["pe_oi"] * (s - k)
        pain[k] = total
    best = min(pain, key=pain.get)
    return best, pain


def multi_strike_band(snap, levels, spot=None):
    """
    Aggregate the chain across ATM ± `levels` strikes (so levels=3 sums seven
    strikes: ATM and three either side) and return the band's totals, its own
    PCR, and the max pain computed over the whole chain.
    """
    if not snap or not snap.get("strikes"):
        return None
    strikes = snap["strikes"]
    ks = sorted(strikes.keys())
    if not ks:
        return None
    spot = spot if spot is not None else snap.get("underlying")
    if spot is None:
        # Fall back to the strike where CE and PE premiums are closest —
        # that is effectively the market's own ATM.
        spot = min(ks, key=lambda s: abs(strikes[s]["ce_ltp"] - strikes[s]["pe_ltp"]))
    atm = min(ks, key=lambda s: abs(s - float(spot)))
    ai = ks.index(atm)
    lo, hi = max(0, ai - int(levels)), min(len(ks) - 1, ai + int(levels))
    band = ks[lo:hi + 1]

    agg = {k: 0.0 for k in ("ce_oi", "pe_oi", "ce_oi_change", "pe_oi_change",
                            "ce_vol", "pe_vol", "ce_vol_change", "pe_vol_change")}
    for s in band:
        for k in agg:
            agg[k] += strikes[s][k]
    mp, _pain = compute_max_pain(strikes)
    return {
        "atm": atm, "band": band, "levels": int(levels), "spot": float(spot),
        "ce_oi": agg["ce_oi"], "pe_oi": agg["pe_oi"],
        "ce_oi_change": agg["ce_oi_change"], "pe_oi_change": agg["pe_oi_change"],
        "ce_volume": agg["ce_vol"], "pe_volume": agg["pe_vol"],
        "ce_volume_change": agg["ce_vol_change"], "pe_volume_change": agg["pe_vol_change"],
        "pcr": (agg["pe_oi"] / agg["ce_oi"]) if agg["ce_oi"] else None,
        "pcr_volume": (agg["pe_vol"] / agg["ce_vol"]) if agg["ce_vol"] else None,
        "max_pain": mp,
    }


def days_to_expiry(expiry_str):
    try:
        return (pd.to_datetime(expiry_str).date() - ist_now().date()).days
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PCR HISTORY TRACKER — feeds the PCR strategy's table and its change columns
# ---------------------------------------------------------------------------

def _atm_row(snap):
    """ATM strike and its per-strike row (nearest strike to spot)."""
    if not snap or not snap.get("strikes"):
        return None, {}
    ks = sorted(snap["strikes"].keys())
    if not ks:
        return None, {}
    spot = snap.get("underlying")
    if spot is None:
        atm = min(ks, key=lambda s: abs(snap["strikes"][s]["ce_ltp"] - snap["strikes"][s]["pe_ltp"]))
    else:
        atm = min(ks, key=lambda s: abs(s - float(spot)))
    return atm, snap["strikes"][atm]


@st.cache_data(ttl=300, show_spinner=False)
def get_current_vix():
    """Latest India VIX close. VIX only publishes daily, so this is cached for
    5 minutes — it is context for the chain, not a tick-level input."""
    try:
        s = fetch_vix_series("1mo")
        if s is not None and len(s):
            return float(s.iloc[-1])
    except Exception:
        pass
    return None


def record_chain_history(snap, futures=None, underlying_label=None):
    """Append one row per distinct snapshot (deduped by fetch timestamp). This
    time series is what every plot on the Option Chain Analysis tab draws from,
    and what the change-vs-previous columns are computed against. A real
    Timestamp is stored alongside the display string so the history can be
    resampled into interval buckets or spanned across days from the database.

    Greeks note: CE and PE at the same strike share gamma and vega (put-call
    parity), so those take the ATM value directly; theta differs between the
    legs, so the STRADDLE theta (CE + PE) is stored, which is the decay number
    that actually matters to an ATM straddle buyer."""
    if not snap:
        return
    hist = st.session_state.setdefault("chain_history", [])
    stamp = snap.get("fetched_at")
    if hist and hist[-1].get("Time") == stamp:
        return
    mp, _pain = compute_max_pain(snap.get("strikes") or {})
    atm, arow = _atm_row(snap)
    hist.append({
        "Time": stamp,
        "Timestamp": ist_now(),
        "PCR": round(snap["pcr"], 4) if snap.get("pcr") else None,
        "PCR Volume": round(snap["pcr_volume"], 4) if snap.get("pcr_volume") else None,
        "Price": snap.get("underlying"),
        "Futures": futures,
        "CE OI": snap.get("ce_oi"), "PE OI": snap.get("pe_oi"),
        "Total OI": (snap.get("ce_oi") or 0) + (snap.get("pe_oi") or 0),
        "CE ΔOI": snap.get("ce_oi_change"), "PE ΔOI": snap.get("pe_oi_change"),
        "Net ΔOI (PE−CE)": (snap.get("pe_oi_change") or 0) - (snap.get("ce_oi_change") or 0),
        "CE Volume": snap.get("ce_volume"), "PE Volume": snap.get("pe_volume"),
        "Total Volume": (snap.get("ce_volume") or 0) + (snap.get("pe_volume") or 0),
        "CE ΔVolume": snap.get("ce_volume_change"), "PE ΔVolume": snap.get("pe_volume_change"),
        "Max Pain": mp,
        "ATM Strike": atm,
        "ATM Gamma": max(arow.get("ce_gamma", 0.0), arow.get("pe_gamma", 0.0)) if arow else None,
        "ATM Vega": max(arow.get("ce_vega", 0.0), arow.get("pe_vega", 0.0)) if arow else None,
        "ATM Theta": (arow.get("ce_theta", 0.0) + arow.get("pe_theta", 0.0)) if arow else None,
        "ATM Straddle": (arow.get("ce_ltp", 0.0) + arow.get("pe_ltp", 0.0)) if arow else None,
        "ATM IV": max(arow.get("ce_iv", 0.0), arow.get("pe_iv", 0.0)) if arow else None,
        "VIX": get_current_vix(),
    })
    if len(hist) > 1000:
        del hist[:-1000]
    if underlying_label:
        db_save_chain_snapshot(snap, underlying_label, futures)


def chain_history_df():
    """Chain history as a DataFrame with change columns (absolute, %, n×)."""
    hist = st.session_state.get("chain_history", [])
    if not hist:
        return pd.DataFrame()
    df = pd.DataFrame(hist)
    for col, short in (("PCR", "PCR"), ("Price", "Price"), ("Total OI", "OI"),
                       ("Total Volume", "Volume"), ("Max Pain", "MaxPain"),
                       ("ATM Gamma", "Gamma")):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        prev = s.shift(1)
        df[f"Δ{short} (abs)"] = (s - prev)
        df[f"Δ{short} (%)"] = ((s - prev) / prev.replace(0, np.nan) * 100)
        df[f"Δ{short} (n×)"] = (s / prev.replace(0, np.nan))
    return df


def build_chain_history_table():
    """
    PCR tracking table: each row's value plus its change from the PREVIOUS
    row expressed three ways — absolute, percentage, and n× multiple — for
    PCR, price, OI and volume.
    """
    df = chain_history_df()
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(4)
    ordered = ["Time", "PCR", "ΔPCR (abs)", "ΔPCR (%)", "ΔPCR (n×)",
               "Price", "ΔPrice (abs)", "ΔPrice (%)", "ΔPrice (n×)",
               "CE OI", "PE OI", "Total OI", "ΔOI (abs)", "ΔOI (%)", "ΔOI (n×)",
               "CE ΔOI", "PE ΔOI", "Net ΔOI (PE−CE)",
               "CE Volume", "PE Volume", "Total Volume",
               "ΔVolume (abs)", "ΔVolume (%)", "ΔVolume (n×)",
               "Max Pain", "ATM Strike", "ATM Gamma", "ATM Straddle", "ATM IV"]
    return df[[c for c in ordered if c in df.columns]].iloc[::-1]   # newest first


# ---------------------------------------------------------------------------
# CHAIN PLOTTING (Option Chain Analysis tab)
# ---------------------------------------------------------------------------

# label → (history column, axis family). Metrics sharing a family share a
# y-axis, so scales that belong together stay directly comparable.
CHAIN_METRICS = {
    "Price (index/underlying)": ("Price", "price"),
    "Futures Price": ("Futures", "price"),
    "Max Pain": ("Max Pain", "price"),
    "ATM Strike": ("ATM Strike", "price"),
    "PCR (OI)": ("PCR", "ratio"),
    "PCR (Volume)": ("PCR Volume", "ratio"),
    "Total OI": ("Total OI", "oi"),
    "CE OI": ("CE OI", "oi"),
    "PE OI": ("PE OI", "oi"),
    "Change in OI (total)": ("ΔOI (abs)", "oi_change"),
    "Change in CE OI": ("CE ΔOI", "oi_change"),
    "Change in PE OI": ("PE ΔOI", "oi_change"),
    "Net ΔOI (PE − CE)": ("Net ΔOI (PE−CE)", "oi_change"),
    "Total Volume": ("Total Volume", "volume"),
    "CE Volume": ("CE Volume", "volume"),
    "PE Volume": ("PE Volume", "volume"),
    "ATM Gamma": ("ATM Gamma", "greek"),
    "ATM Vega": ("ATM Vega", "greek"),
    "ATM Theta": ("ATM Theta", "greek"),
    "ATM IV": ("ATM IV", "greek"),
    "India VIX": ("VIX", "greek"),
    "ATM Straddle Premium": ("ATM Straddle", "premium"),
}

_FAMILY_TITLE = {
    "price": "Price / Strike", "ratio": "PCR", "oi": "Open Interest",
    "oi_change": "Change in OI", "volume": "Volume", "greek": "Gamma / IV",
    "premium": "Premium",
}
_SERIES_COLORS = ["#4c9be8", "#f0a202", "#38b000", "#e5383b", "#9d4edd",
                  "#00b4d8", "#ff7b00", "#c2185b", "#7cb342"]


CHART_TYPES = ["Line", "Area", "Bar (grouped)", "Bar (stacked)", "Scatter",
               "Line + Markers", "Pie (latest snapshot)"]


def _missing_metrics(hist, metric_labels):
    """Which of these metrics have no usable values — used to explain empty plots."""
    out = []
    for lbl in metric_labels:
        col, _fam = CHAIN_METRICS.get(lbl, (None, None))
        if hist is None or hist.empty or not col or col not in hist.columns \
                or pd.to_numeric(hist[col], errors="coerce").notna().sum() == 0:
            out.append(lbl)
    return out or ["(no snapshots recorded)"]


def chain_plot(hist, metric_labels, title, normalize=None, height=430, chart_type="Line"):
    """
    Multi-metric time-series plot.

    Metrics are grouped by axis family so like scales share an axis. With one
    or two families the real values are shown on twin axes; with three or more
    (e.g. PCR + price + ΔOI + volume, whose magnitudes differ by orders of
    magnitude) the series are indexed to 100 at the first reading so shape and
    turning points stay comparable — the raw values remain visible in the
    tooltip and in the summary beneath. `normalize` overrides that choice.
    """
    metric_labels = [m for m in metric_labels if m in CHAIN_METRICS]
    if hist is None or hist.empty or not metric_labels:
        return None, False
    cols = [(m, *CHAIN_METRICS[m]) for m in metric_labels]
    cols = [(m, c, fam) for m, c, fam in cols if c in hist.columns
            and pd.to_numeric(hist[c], errors="coerce").notna().any()]
    if not cols:
        return None, False

    families = list(dict.fromkeys(fam for _m, _c, fam in cols))
    auto_norm = len(families) >= 3
    do_norm = auto_norm if normalize is None else bool(normalize)

    x = hist["Time"] if "Time" in hist.columns else hist.index
    fig = go.Figure()
    fam_axis = {}
    axis_slots = ["y", "y2", "y3", "y4"]
    for i, fam in enumerate(families[:4]):
        fam_axis[fam] = axis_slots[i] if not do_norm else "y"

    # ---- Pie is a snapshot composition, not a time series ----
    if str(chart_type).startswith("Pie"):
        labels_p, values_p = [], []
        for label, col, _fam in cols:
            s = pd.to_numeric(hist[col], errors="coerce").dropna()
            if len(s):
                labels_p.append(label)
                values_p.append(abs(float(s.iloc[-1])))
        if not values_p or sum(values_p) == 0:
            return None, False
        fig = go.Figure(data=[go.Pie(labels=labels_p, values=values_p, hole=0.35,
                                     marker=dict(colors=_SERIES_COLORS[:len(values_p)]),
                                     textinfo="label+percent")])
        fig.update_layout(title=f"{title} — latest snapshot composition", height=height,
                          margin=dict(l=30, r=30, t=60, b=30))
        return fig, False

    _is_bar = str(chart_type).startswith("Bar")
    for i, (label, col, fam) in enumerate(cols):
        raw = pd.to_numeric(hist[col], errors="coerce")
        if do_norm:
            first_valid = raw.dropna()
            base = first_valid.iloc[0] if len(first_valid) else np.nan
            plotted = (raw / base * 100.0) if (base and not pd.isna(base) and base != 0) else raw
        else:
            plotted = raw
        colr = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        common = dict(x=x, y=plotted, name=label, yaxis=fam_axis.get(fam, "y"),
                      customdata=np.stack([raw.values], axis=-1),
                      hovertemplate=f"<b>{label}</b><br>%{{x}}<br>value: %{{customdata[0]:,.4f}}<extra></extra>")
        if _is_bar:
            fig.add_trace(go.Bar(marker=dict(color=colr), **common))
        elif chart_type == "Scatter":
            fig.add_trace(go.Scatter(mode="markers", marker=dict(size=8, color=colr), **common))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(mode="lines", fill="tozeroy", line=dict(width=2, color=colr), **common))
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(mode="lines", line=dict(width=2, color=colr), **common))
        else:  # Line + Markers (default look)
            fig.add_trace(go.Scatter(mode="lines+markers", line=dict(width=2, color=colr),
                                     marker=dict(size=5), **common))

    layout = dict(title=title, height=height, hovermode="x unified",
                  margin=dict(l=60, r=60, t=60, b=40),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    if str(chart_type).startswith("Bar"):
        layout["barmode"] = "stack" if "stacked" in str(chart_type) else "group"
    if do_norm:
        layout["yaxis"] = dict(title="Indexed to 100 at first reading")
    else:
        side_order = ["left", "right", "left", "right"]
        for i, fam in enumerate(families[:4]):
            key = "yaxis" if i == 0 else f"yaxis{i+1}"
            ax = dict(title=_FAMILY_TITLE.get(fam, fam), side=side_order[i])
            if i > 0:
                ax.update(overlaying="y", showgrid=False)
            if i >= 2:
                ax["anchor"] = "free"
                ax["position"] = 0.0 if side_order[i] == "left" else 1.0
            layout[key] = ax
        if len(families) >= 3:
            layout["xaxis"] = dict(domain=[0.08, 0.92])
    fig.update_layout(**layout)
    return fig, do_norm


def _trend(series, window=6):
    """Direction and magnitude of a series over the recent window."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None, None, None
    recent = s.iloc[-window:] if len(s) > window else s
    first, last = float(recent.iloc[0]), float(recent.iloc[-1])
    delta = last - first
    pct = (delta / first * 100.0) if first else None
    word = "rising" if delta > 0 else ("falling" if delta < 0 else "flat")
    return word, delta, pct


def chain_recommendation(hist, snap):
    """
    Overall CE/PE recommendation from the whole chain picture.

    Six independent votes, each a conventional option-chain read:
      1. PCR level        — high = heavy put writing = bullish
      2. PCR direction    — rising ratio = puts being added = bullish
      3. ΔOI dominance    — which side is being written more right now
      4. Spot vs max pain — price tends to drift toward max pain near expiry
      5. Volume PCR       — where today's actual participation is going
      6. Price trend      — the market's own answer, as confirmation
    The net score decides, so no single input can carry an entry alone. This
    is a mechanical reading of positioning, not a forecast.
    """
    votes, lines = 0, []
    if not snap:
        return "NO DATA", 0, ["Option chain unavailable — cannot form a recommendation."]

    pcr = snap.get("pcr")
    if pcr:
        if pcr >= 1.2:
            votes += 1; lines.append(f"🟢 +1 · PCR {pcr:.3f} ≥ 1.20 — puts written heavily, writers expect support to hold (bullish).")
        elif pcr <= 0.8:
            votes -= 1; lines.append(f"🔴 −1 · PCR {pcr:.3f} ≤ 0.80 — calls written heavily, resistance being built (bearish).")
        else:
            lines.append(f"⚪ 0 · PCR {pcr:.3f} sits in the neutral 0.80–1.20 band.")

    if hist is not None and not hist.empty and "PCR" in hist.columns:
        word, delta, _pct = _trend(hist["PCR"])
        if word == "rising":
            votes += 1; lines.append(f"🟢 +1 · PCR {word} ({delta:+.3f} over the recent window) — puts being added.")
        elif word == "falling":
            votes -= 1; lines.append(f"🔴 −1 · PCR {word} ({delta:+.3f}) — puts unwinding or calls being added.")
        elif word:
            lines.append("⚪ 0 · PCR flat over the window.")
        else:
            lines.append("⚪ 0 · not enough history yet to read a PCR trend.")

    d_ce, d_pe = snap.get("ce_oi_change", 0.0), snap.get("pe_oi_change", 0.0)
    if d_pe > d_ce:
        votes += 1; lines.append(f"🟢 +1 · PE ΔOI {d_pe:+,.0f} exceeds CE ΔOI {d_ce:+,.0f} — put writing dominant (support forming).")
    elif d_ce > d_pe:
        votes -= 1; lines.append(f"🔴 −1 · CE ΔOI {d_ce:+,.0f} exceeds PE ΔOI {d_pe:+,.0f} — call writing dominant (resistance forming).")

    mp, _p = compute_max_pain(snap.get("strikes") or {})
    spot = snap.get("underlying")
    if mp and spot:
        if spot < mp:
            votes += 1; lines.append(f"🟢 +1 · Spot {spot:,.2f} below max pain {mp:,.0f} — pull toward max pain is upward.")
        elif spot > mp:
            votes -= 1; lines.append(f"🔴 −1 · Spot {spot:,.2f} above max pain {mp:,.0f} — pull toward max pain is downward.")

    vpcr = snap.get("pcr_volume")
    if vpcr:
        if vpcr > 1.0:
            votes += 1; lines.append(f"🟢 +1 · Volume PCR {vpcr:.3f} > 1 — today's flow is concentrated in puts.")
        else:
            votes -= 1; lines.append(f"🔴 −1 · Volume PCR {vpcr:.3f} ≤ 1 — today's flow is concentrated in calls.")

    if hist is not None and not hist.empty and "Price" in hist.columns:
        word, delta, pct = _trend(hist["Price"])
        if word == "rising":
            votes += 1; lines.append(f"🟢 +1 · Price {word} ({delta:+,.2f}"
                                     + (f", {pct:+.2f}%" if pct is not None else "") + ") over the window.")
        elif word == "falling":
            votes -= 1; lines.append(f"🔴 −1 · Price {word} ({delta:+,.2f}"
                                     + (f", {pct:+.2f}%" if pct is not None else "") + ") over the window.")

    if votes >= 2:
        verdict = "BUY CE"
    elif votes <= -2:
        verdict = "BUY PE"
    else:
        verdict = "WAIT / NO EDGE"
    return verdict, votes, lines


def chain_plot_summary(hist, snap, metric_labels, normalized=False):
    """Per-plot summary: what each plotted series is doing, notable
    divergences between them, then the overall recommendation."""
    out = []
    if hist is None or hist.empty:
        return ["No snapshots recorded yet."], ("NO DATA", 0, [])
    for label in metric_labels:
        col, _fam = CHAIN_METRICS.get(label, (None, None))
        if not col or col not in hist.columns:
            continue
        s = pd.to_numeric(hist[col], errors="coerce").dropna()
        if s.empty:
            continue
        word, delta, pct = _trend(s)
        latest = float(s.iloc[-1])
        fmt = f"{latest:,.4f}" if abs(latest) < 100 else f"{latest:,.0f}"
        piece = f"**{label}**: {fmt}"
        if word:
            piece += f" · {word} ({delta:+,.4f}" + (f", {pct:+.2f}%" if pct is not None else "") + " over window)"
        out.append(piece)

    # divergence checks between price and positioning
    def _dir(col):
        if col in hist.columns:
            w, _d, _p = _trend(hist[col])
            return w
        return None
    p_dir, pcr_dir = _dir("Price"), _dir("PCR")
    if p_dir and pcr_dir and p_dir != "flat" and pcr_dir != "flat":
        if p_dir != pcr_dir:
            out.append(f"⚠️ **Divergence**: price is {p_dir} while PCR is {pcr_dir} — positioning is not confirming "
                       "the move, which often precedes a stall or reversal.")
        else:
            out.append(f"✅ **Confirmation**: price and PCR are both {p_dir} — positioning agrees with the move.")
    if normalized:
        out.append("_Series are indexed to 100 at the first reading because their raw scales differ by orders of "
                   "magnitude; hover any point for the true value._")
    return out, chain_recommendation(hist, snap)


# ---------------------------------------------------------------------------
# ANALYSIS TABLE (own timeframe/period, selectable + orderable columns)
# ---------------------------------------------------------------------------

TABLE_TIMEFRAMES = ["1m", "2m", "3m", "5m", "10m", "15m", "30m", "60m", "1h", "4h", "1d", "1wk"]
TABLE_PERIODS = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "4y", "5y",
                 "6y", "7y", "8y", "9y", "10y"]
_TF_RULE = {"1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min", "10m": "10min",
            "15m": "15min", "30m": "30min", "60m": "60min", "1h": "1h", "4h": "4h",
            "1d": "1D", "1wk": "1W"}
_PERIOD_DAYS = {"1d": 1, "5d": 5, "7d": 7, "1mo": 31, "3mo": 92, "6mo": 183, "1y": 366,
                "2y": 731, "3y": 1096, "4y": 1461, "5y": 1827, "6y": 2192, "7y": 2557,
                "8y": 2922, "9y": 3287, "10y": 3653}

# display column → source column in the history frame
TABLE_COLUMN_SOURCE = {
    "PCR": "PCR", "Spot": "Price", "Future": "Futures",
    "Total OI": "Total OI", "CE OI": "CE OI", "PE OI": "PE OI",
    "CE ΔOI": "CE ΔOI", "PE ΔOI": "PE ΔOI",
    "Volume": "Total Volume", "Max Pain": "Max Pain",
    "Straddle": "ATM Straddle", "Gamma": "ATM Gamma", "Vega": "ATM Vega",
    "Theta": "ATM Theta", "IV": "ATM IV", "VIX": "VIX",
}
# columns that also get a period-over-period change column
TABLE_DELTA_OF = ["PCR", "Spot", "Future", "Total OI", "CE OI", "PE OI", "Volume",
                  "Max Pain", "Straddle", "Gamma", "Vega", "Theta", "IV", "VIX"]

TABLE_ALL_COLUMNS = ["Time"]
for _c in ["PCR", "Spot", "Future", "Total OI", "CE OI", "PE OI", "CE ΔOI", "PE ΔOI",
           "Volume", "Max Pain", "Straddle", "Gamma", "Vega", "Theta", "IV", "VIX"]:
    TABLE_ALL_COLUMNS.append(_c)
    if _c in TABLE_DELTA_OF:
        TABLE_ALL_COLUMNS.append(f"Δ {_c}")

TABLE_DEFAULT_COLUMNS = ["Time", "PCR", "Δ PCR", "Spot", "Δ Spot", "Future", "Δ Future",
                         "Total OI", "CE ΔOI", "PE ΔOI", "Volume", "Max Pain", "Straddle", "IV"]


# yfinance history limits by interval: 1m ≈ 7 days, other intraday ≈ 60 days,
# daily is effectively unlimited. The table fetches at a base interval within
# those limits and resamples up to the requested timeframe.
_TF_BASE_INTERVAL = {"1m": "1m", "2m": "2m", "3m": "1m", "5m": "5m", "10m": "5m",
                     "15m": "15m", "30m": "30m", "60m": "60m", "1h": "60m",
                     "4h": "60m", "1d": "1d", "1wk": "1d"}
_BASE_LIMIT_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 60, "1d": 3650}
_YF_PERIOD_LADDER = [(1, "1d"), (5, "5d"), (7, "7d"), (31, "1mo"), (92, "3mo"),
                     (183, "6mo"), (366, "1y"), (731, "2y"), (1827, "5y"), (3653, "10y")]


def _norm_ts(values):
    """
    Normalise any datetime input to naive datetime64[ns].

    pandas 2/3 preserve the source resolution, so candle indexes can arrive as
    datetime64[s] while timestamps built from Python datetime objects are
    datetime64[us]. merge_asof requires both keys to have the SAME unit, so
    every merge key in this module is pushed through here first.
    """
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    try:
        if getattr(s.dtype, "tz", None) is not None:
            s = s.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass
    try:
        s = s.astype("datetime64[ns]")
    except Exception:
        pass
    return s


def _days_to_period_string(days):
    for lim, label in _YF_PERIOD_LADDER:
        if days <= lim:
            return label
    return "10y"


def _is_indian_underlying(uinfo):
    """Indian equity/index underlyings observe the 09:15–15:30 IST session."""
    if not uinfo:
        return False
    y = str(uinfo.get("yf") or "")
    return y.endswith((".NS", ".BO")) or y in ("^NSEI", "^NSEBANK", "^BSESN")


def _table_master_timeline(uinfo, timeframe, period):
    """
    Build the table's TIME AXIS from real candles rather than from recorded
    snapshots.

    This is the fix for a sparse table: chain snapshots only exist for the
    minutes analysis actually ran, so a snapshot-driven table shows a handful
    of rows. Candles exist for every traded minute, already exclude weekends,
    holidays and pre/post-market, and give a genuine Spot series — so the
    candle index becomes the spine, and chain values are attached to it.
    Returns (DataFrame indexed by Timestamp with a Spot column, note).
    """
    if not uinfo or not uinfo.get("yf"):
        return pd.DataFrame(), "No underlying resolved, so no market timeline could be built."
    base = _TF_BASE_INTERVAL.get(timeframe, "1m")
    want_days = _PERIOD_DAYS.get(period, 1)
    limit = _BASE_LIMIT_DAYS.get(base, 60)
    eff_days = min(want_days, limit)
    note = None
    if eff_days < want_days:
        note = (f"{timeframe} candles are only available for about {limit} days, so the timeline covers the last "
                f"{eff_days} days rather than {period}. Use a larger timeframe for longer history.")
    try:
        raw = fetch_data(uinfo["yf"], base, _days_to_period_string(eff_days))
    except Exception as exc:
        return pd.DataFrame(), f"Could not load candles for the timeline: {exc}"
    if raw is None or raw.empty:
        return pd.DataFrame(), "No candles returned for the timeline (check the data source and market hours)."

    df = raw.copy()
    try:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        df.index = idx
    except Exception:
        pass
    rule = _TF_RULE.get(timeframe, "1min")
    if rule not in ("1min",) or base != "1m":
        try:
            df = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                        "Close": "last", "Volume": "sum"}).dropna(how="all")
        except Exception:
            pass
    df = df.dropna(subset=["Close"])
    cutoff = pd.Timestamp(ist_now()).tz_localize(None) - pd.Timedelta(days=eff_days)
    df = df[df.index >= cutoff]
    # Keep only real Indian trading sessions: weekdays, 09:15–15:30 IST. Even
    # with correct timezones this drops any pre/post-market or padded rows a
    # feed might return, so the table can never show a time the market was shut.
    if timeframe not in ("1d", "1wk") and _is_indian_underlying(uinfo):
        try:
            tod = df.index.time
            in_session = (tod >= dtime(9, 15)) & (tod <= dtime(15, 30))
            df = df[in_session & (df.index.dayofweek < 5)]
        except Exception:
            pass
    out = pd.DataFrame({"Timestamp": _norm_ts(df.index),
                        "Spot": pd.to_numeric(df["Close"], errors="coerce").values})
    return out, note


def _futures_timeline(uinfo, timeframe, period):
    """True futures candles for the same timeline, when Dhan can serve them."""
    if not uinfo:
        return pd.DataFrame()
    _, token = _dhan_creds()
    if not token:
        return pd.DataFrame()
    try:
        exps = dhan_get_expiries(uinfo["underlying"], uinfo["fut_instrument"], uinfo["exchange"])
        if not exps:
            return pd.DataFrame()
        info = dhan_lookup_future(uinfo["underlying"], exps[0], uinfo["fut_instrument"], uinfo["exchange"])
        if not info:
            return pd.DataFrame()
        base = _TF_BASE_INTERVAL.get(timeframe, "1m")
        base = base if base in DHAN_INTERVAL_CODE or base == "1d" else "5m"
        days = min(_PERIOD_DAYS.get(period, 1), _BASE_LIMIT_DAYS.get(base, 60))
        df = _dhan_fetch_candles_cached(info["security_id"], f"{uinfo['exchange']}_FNO", "FUTIDX",
                                        base, _days_to_period_string(days), hash(token) % 10_000_019)
        if df is None or df.empty:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        df = df.copy(); df.index = idx
        rule = _TF_RULE.get(timeframe, "1min")
        df = df.resample(rule).agg({"Close": "last"}).dropna()
        return pd.DataFrame({"Timestamp": _norm_ts(df.index),
                             "Future": pd.to_numeric(df["Close"], errors="coerce").values})
    except Exception:
        return pd.DataFrame()


def _vix_timeline():
    """India VIX publishes daily, so it is mapped onto the grid by date."""
    try:
        s = fetch_vix_series("1y")
        if s is None or not len(s):
            return pd.DataFrame()
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        return pd.DataFrame({"Date": _norm_ts(idx).dt.normalize(),
                             "VIX": pd.to_numeric(s.values, errors="coerce")})
    except Exception:
        return pd.DataFrame()


def chain_readiness(uinfo, expiry):
    """
    Explain in plain terms why chain data may be missing, instead of leaving
    the UI full of None. Returns (ok, list_of_problems).
    """
    problems = []
    client, token = _dhan_creds()
    if not str(token or "").strip():
        problems.append(
            "**No Dhan Access Token.** Every option-chain value — PCR, CE/PE OI, ΔOI, volume, max pain, "
            "straddle, IV and the greeks — comes from Dhan's option-chain API, which requires a token. "
            "Without one, those columns stay empty and the chain plots have nothing to draw. "
            "Add it in the sidebar under **🔐 Dhan Account**. (Spot still works from candles, which is why "
            "only that column is populated.)")
    if not uinfo:
        problems.append("**Underlying not resolved** — the symbol was not found in Dhan's scrip master.")
    if not expiry:
        problems.append("**No expiry selected** — the chain endpoint needs one.")
    if not st.session_state.get("chain_history"):
        problems.append(
            "**No snapshots recorded yet.** Press **🔍 Analyze Once** or **▶ Analyze Continuously** above; "
            "the table and plots fill in from snapshots as they are taken.")
    now = ist_now()
    if now.weekday() >= 5 or not (dtime(9, 15) <= now.time() <= dtime(15, 30)):
        problems.append(
            f"**Market is closed right now** ({now.strftime('%a %H:%M')} IST). Dhan's chain endpoint generally "
            "returns nothing outside 09:15–15:30 IST on trading days, so live snapshots will not appear until "
            "the next session.")
    return (len(problems) == 0), problems


def build_chain_analysis_table(timeframe, period, underlying_label=None, expiry=None,
                               uinfo=None, carry_forward=True):
    """
    Build the analysis table at its OWN timeframe and period.

    The row grid comes from real market candles, so a 1m/5d table has a row for
    every traded minute (09:15–15:30, trading days only) rather than only the
    minutes a chain snapshot happened to be taken. Spot — and futures where
    Dhan can serve them — are true per-bucket values from those candles.

    Chain metrics (PCR, OI, max pain, straddle, greeks) are attached with a
    backward as-of join: each row carries the most recent snapshot at or before
    that time. That is the correct treatment because every one of these is a
    LEVEL or a cumulative day-to-date figure — Dhan's OI change is measured
    against the previous day's close, and chain volume is the day's running
    total — so carrying the last known value forward is accurate rather than
    invented, and the per-bucket Δ columns then fall out correctly. Rows before
    the first snapshot stay blank, and the caller is told how many rows carry a
    real snapshot versus a carried one.
    """
    days = _PERIOD_DAYS.get(period, 1)
    notes = []

    # ---- chain snapshots (session, or database for multi-day) ----
    snaps = chain_history_df()
    if days > 1:
        if db_enabled():
            dbdf = db_load_chain_history(underlying_label, expiry, since_days=days)
            if not dbdf.empty:
                snaps = dbdf
                notes.append(f"{len(dbdf)} stored snapshots loaded from the database.")
            else:
                notes.append(f"No stored snapshots yet for the last {period}; using this session only.")
        else:
            notes.append("Periods beyond 1d need Data Persistence enabled (Admin Panel) for chain history — "
                         "Dhan has no historical option-chain API. Spot and futures still cover the full period.")

    # ---- master timeline from candles ----
    grid, gnote = _table_master_timeline(uinfo, timeframe, period)
    if gnote:
        notes.append(gnote)

    if grid.empty:
        # No candles: fall back to the old snapshot-only behaviour.
        if snaps is None or snaps.empty or "Timestamp" not in snaps.columns:
            return pd.DataFrame(), " ".join(notes) if notes else None
        work = snaps.copy()
        work["Timestamp"] = _norm_ts(work["Timestamp"])
        work = work.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")
        res = work.resample(_TF_RULE.get(timeframe, "1min")).last().dropna(how="all").reset_index()
        grid = pd.DataFrame({"Timestamp": res["Timestamp"],
                             "Spot": pd.to_numeric(res.get("Price"), errors="coerce")})
        notes.append("Falling back to snapshot times because no candles were available for the timeline.")
        merged = res.rename(columns={"Price": "Spot"})
        merged["Timestamp"] = grid["Timestamp"]
    else:
        merged = grid.copy()
        if snaps is not None and not snaps.empty and "Timestamp" in snaps.columns:
            sn = snaps.copy()
            sn["Timestamp"] = _norm_ts(sn["Timestamp"])
            sn = sn.dropna(subset=["Timestamp"]).sort_values("Timestamp")
            keep = [c for c in ["Timestamp", "PCR", "Price", "Futures", "CE OI", "PE OI", "Total OI",
                                "CE ΔOI", "PE ΔOI", "Total Volume", "Max Pain", "ATM Straddle",
                                "ATM Gamma", "ATM Vega", "ATM Theta", "ATM IV", "VIX"]
                    if c in sn.columns]
            sn = sn[keep].rename(columns={"Futures": "Future_snap", "VIX": "VIX_snap"})
            merged["Timestamp"] = _norm_ts(merged["Timestamp"])
            merged = pd.merge_asof(merged.sort_values("Timestamp"), sn,
                                   on="Timestamp", direction="backward")
            merged["_has_snap"] = merged["PCR"].notna() if "PCR" in merged.columns else False
            if not carry_forward:
                # blank out carried rows, keeping only exact snapshot rows
                snap_times = set(sn["Timestamp"])
                mask = ~merged["Timestamp"].isin(snap_times)
                for c in [c for c in merged.columns if c not in ("Timestamp", "Spot", "_has_snap")]:
                    merged.loc[mask, c] = np.nan

    # ---- true futures series where available ----
    fut = _futures_timeline(uinfo, timeframe, period)
    if not fut.empty:
        merged["Timestamp"] = _norm_ts(merged["Timestamp"])
        fut["Timestamp"] = _norm_ts(fut["Timestamp"])
        merged = pd.merge_asof(merged.sort_values("Timestamp"), fut.sort_values("Timestamp"),
                               on="Timestamp", direction="backward")
        notes.append("Futures values come from real futures candles.")
    elif "Future_snap" in merged.columns:
        merged["Future"] = merged["Future_snap"]

    # ---- VIX by date ----
    vix = _vix_timeline()
    if not vix.empty:
        merged["Date"] = _norm_ts(merged["Timestamp"]).dt.normalize()
        vix["Date"] = _norm_ts(vix["Date"]).dt.normalize()
        merged = merged.merge(vix, on="Date", how="left").drop(columns=["Date"])
    elif "VIX_snap" in merged.columns:
        merged["VIX"] = merged["VIX_snap"]

    if merged.empty:
        return pd.DataFrame(), " ".join(notes) if notes else None

    # ---- assemble display columns ----
    out = pd.DataFrame()
    # Time label scales with the window: intraday needs no date, a multi-year
    # window needs the year to stay unambiguous.
    _fmt = ("%H:%M" if days <= 1 else ("%d-%b %H:%M" if days <= 366 else "%d-%b-%y %H:%M"))
    if timeframe in ("1d", "1wk"):
        _fmt = "%d-%b-%Y"
    out["Time"] = pd.to_datetime(merged["Timestamp"]).dt.strftime(_fmt)
    src_map = {"PCR": "PCR", "Spot": "Spot", "Future": "Future", "Total OI": "Total OI",
               "CE OI": "CE OI", "PE OI": "PE OI", "CE ΔOI": "CE ΔOI", "PE ΔOI": "PE ΔOI",
               "Volume": "Total Volume", "Max Pain": "Max Pain", "Straddle": "ATM Straddle",
               "Gamma": "ATM Gamma", "Vega": "ATM Vega", "Theta": "ATM Theta",
               "IV": "ATM IV", "VIX": "VIX"}
    for disp, srccol in src_map.items():
        out[disp] = pd.to_numeric(merged[srccol], errors="coerce") if srccol in merged.columns else np.nan
    for disp in TABLE_DELTA_OF:
        if disp in out.columns:
            out[f"Δ {disp}"] = out[disp].diff()
    out = out[[c for c in TABLE_ALL_COLUMNS if c in out.columns]]

    real = int(merged["_has_snap"].sum()) if "_has_snap" in merged.columns else 0
    notes.append(f"{len(out)} rows on the {timeframe} market grid; {real} carry a chain snapshot "
                 f"({'values carried forward between snapshots' if carry_forward else 'gaps left blank'}).")
    return out.iloc[::-1].reset_index(drop=True), " ".join(notes) if notes else None


def table_plot(table_df, columns, chart_type="Line", height=430, title="Table metrics"):
    """Plot selected TABLE columns. Because these columns can mix wildly
    different scales (VIX ~12, OI ~10^7), three or more selected columns are
    indexed to 100 at the first reading; raw values stay in the tooltip."""
    cols = [c for c in columns if c in table_df.columns and c != "Time"]
    if table_df is None or table_df.empty or not cols:
        return None
    plot_df = table_df.iloc[::-1].reset_index(drop=True)   # chronological for plotting
    x = plot_df["Time"] if "Time" in plot_df.columns else plot_df.index

    if str(chart_type).startswith("Pie"):
        vals, labs = [], []
        for c in cols:
            s = pd.to_numeric(plot_df[c], errors="coerce").dropna()
            if len(s):
                labs.append(c); vals.append(abs(float(s.iloc[-1])))
        if not vals or sum(vals) == 0:
            return None
        fig = go.Figure(data=[go.Pie(labels=labs, values=vals, hole=0.35,
                                     marker=dict(colors=_SERIES_COLORS[:len(vals)]),
                                     textinfo="label+percent")])
        fig.update_layout(title=f"{title} — latest row composition", height=height)
        return fig

    do_norm = len(cols) >= 3
    fig = go.Figure()
    for i, c in enumerate(cols):
        raw = pd.to_numeric(plot_df[c], errors="coerce")
        if do_norm:
            fv = raw.dropna()
            base = fv.iloc[0] if len(fv) else np.nan
            y = (raw / base * 100.0) if (base and not pd.isna(base) and base != 0) else raw
        else:
            y = raw
        colr = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        common = dict(x=x, y=y, name=c, customdata=np.stack([raw.values], axis=-1),
                      hovertemplate=f"<b>{c}</b><br>%{{x}}<br>value: %{{customdata[0]:,.4f}}<extra></extra>",
                      yaxis="y" if (do_norm or i == 0) else "y2")
        if str(chart_type).startswith("Bar"):
            fig.add_trace(go.Bar(marker=dict(color=colr), **common))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(mode="lines", fill="tozeroy", line=dict(width=2, color=colr), **common))
        elif chart_type == "Scatter":
            fig.add_trace(go.Scatter(mode="markers", marker=dict(size=8, color=colr), **common))
        else:
            fig.add_trace(go.Scatter(mode="lines+markers", line=dict(width=2, color=colr),
                                     marker=dict(size=4), **common))
    layout = dict(title=title, height=height, hovermode="x unified",
                  margin=dict(l=60, r=60, t=60, b=40),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    if str(chart_type).startswith("Bar"):
        layout["barmode"] = "stack" if "stacked" in str(chart_type) else "group"
    if do_norm:
        layout["yaxis"] = dict(title="Indexed to 100 at first reading")
    elif len(cols) == 2:
        layout["yaxis"] = dict(title=cols[0], side="left")
        layout["yaxis2"] = dict(title=cols[1], side="right", overlaying="y", showgrid=False)
    fig.update_layout(**layout)
    return fig


def intraday_slice(hist):
    """Rows recorded TODAY (IST) — the basis of the intraday section."""
    if hist is None or hist.empty or "Timestamp" not in hist.columns:
        return pd.DataFrame()
    try:
        ts = pd.to_datetime(hist["Timestamp"], errors="coerce")
        ts = ts.dt.tz_localize(None) if getattr(ts.dtype, "tz", None) else ts
        return hist[ts.dt.date == ist_now().date()].copy()
    except Exception:
        return hist.copy()


def intraday_stats(day_df):
    """Session-level summary of how positioning has evolved today."""
    if day_df is None or day_df.empty:
        return {}
    def _fl(col):
        s = pd.to_numeric(day_df.get(col), errors="coerce").dropna() if col in day_df.columns else pd.Series(dtype=float)
        if s.empty:
            return None, None, None, None
        return float(s.iloc[0]), float(s.iloc[-1]), float(s.min()), float(s.max())
    out = {}
    for label, col in (("PCR", "PCR"), ("Spot", "Price"), ("Futures", "Futures"),
                       ("Total OI", "Total OI"), ("Max Pain", "Max Pain"),
                       ("ATM Straddle", "ATM Straddle"), ("ATM IV", "ATM IV")):
        o, c, lo, hi = _fl(col)
        if o is None:
            continue
        out[label] = {"open": o, "last": c, "low": lo, "high": hi,
                      "change": c - o, "change_pct": ((c - o) / o * 100) if o else None}
    for label, col in (("CE OI built", "CE ΔOI"), ("PE OI built", "PE ΔOI"),
                       ("Volume", "Total Volume")):
        if col in day_df.columns:
            s = pd.to_numeric(day_df[col], errors="coerce").dropna()
            if len(s):
                out[label] = {"total": float(s.sum())}
    return out


# ---------------------------------------------------------------------------
# GROQ AI ANALYSIS
# ---------------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_CHOICES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "(custom — type below)",
]


def build_groq_chain_payload(snap, hist, levels=6, hist_rows=25):
    """
    Compact, complete-enough description of the chain for the model: headline
    aggregates, the ATM band strike by strike, max pain, and the recent
    snapshot history. Trimmed deliberately — sending 60+ strikes of raw JSON
    wastes tokens and buries the signal.
    """
    if not snap:
        return None
    mp, _pain = compute_max_pain(snap.get("strikes") or {})
    atm, arow = _atm_row(snap)
    band = multi_strike_band(snap, levels)
    payload = {
        "captured_at_ist": snap.get("fetched_at"),
        "expiry": snap.get("expiry"),
        "days_to_expiry": days_to_expiry(snap.get("expiry")),
        "spot": snap.get("underlying"),
        "atm_strike": atm,
        "max_pain": mp,
        "totals": {
            "ce_oi": snap.get("ce_oi"), "pe_oi": snap.get("pe_oi"),
            "ce_oi_change": snap.get("ce_oi_change"), "pe_oi_change": snap.get("pe_oi_change"),
            "ce_volume": snap.get("ce_volume"), "pe_volume": snap.get("pe_volume"),
            "ce_volume_change": snap.get("ce_volume_change"), "pe_volume_change": snap.get("pe_volume_change"),
            "pcr_oi": snap.get("pcr"), "pcr_volume": snap.get("pcr_volume"),
        },
        "atm_metrics": {
            "straddle_premium": (arow.get("ce_ltp", 0) + arow.get("pe_ltp", 0)) if arow else None,
            "ce_iv": arow.get("ce_iv") if arow else None,
            "pe_iv": arow.get("pe_iv") if arow else None,
            "ce_gamma": arow.get("ce_gamma") if arow else None,
            "pe_gamma": arow.get("pe_gamma") if arow else None,
        },
        "atm_band": ({
            "levels_each_side": band["levels"],
            "band_pcr_oi": band["pcr"], "band_pcr_volume": band["pcr_volume"],
            "band_ce_oi": band["ce_oi"], "band_pe_oi": band["pe_oi"],
            "band_ce_oi_change": band["ce_oi_change"], "band_pe_oi_change": band["pe_oi_change"],
        } if band else None),
        "strikes": [],
    }
    if band:
        for s in band["band"]:
            r = snap["strikes"][s]
            payload["strikes"].append({
                "strike": s,
                "ce": {"oi": r["ce_oi"], "oi_chg": r["ce_oi_change"], "vol": r["ce_vol"],
                       "ltp": r["ce_ltp"], "iv": r["ce_iv"], "gamma": r["ce_gamma"], "delta": r["ce_delta"]},
                "pe": {"oi": r["pe_oi"], "oi_chg": r["pe_oi_change"], "vol": r["pe_vol"],
                       "ltp": r["pe_ltp"], "iv": r["pe_iv"], "gamma": r["pe_gamma"], "delta": r["pe_delta"]},
            })
    if hist is not None and not hist.empty:
        keep = [c for c in ("Time", "PCR", "Price", "Total OI", "Total Volume",
                            "CE ΔOI", "PE ΔOI", "Max Pain", "ATM Gamma") if c in hist.columns]
        payload["recent_history"] = json.loads(
            hist[keep].tail(hist_rows).to_json(orient="records", date_format="iso"))
    return payload


def groq_analyze_chain(api_key, model, snap, hist, extra_instructions="", temperature=0.2):
    """
    Send the chain to Groq's OpenAI-compatible chat endpoint and return the
    analysis text. Never raises — any failure comes back as a readable message
    so a bad key or a rate limit can't interrupt the app.
    """
    if not str(api_key or "").strip():
        return "⚠️ No Groq API key set — enter one to enable AI analysis."
    payload = build_groq_chain_payload(snap, hist)
    if not payload:
        return "⚠️ No option-chain snapshot available to analyse."
    system = (
        "You are an experienced Indian index options analyst. You will receive a JSON snapshot of a live NSE/BSE "
        "option chain plus a short history of previous snapshots. Analyse positioning and produce a concise, "
        "structured read. Required sections, in order:\n"
        "1. VERDICT — exactly one of: BUY CE, BUY PE, or WAIT.\n"
        "2. CONFIDENCE — Low / Medium / High.\n"
        "3. WHY — 3 to 6 short bullets citing the specific numbers that drive the verdict.\n"
        "4. KEY LEVELS — support and resistance implied by OI walls, plus max pain.\n"
        "5. RISKS — what would invalidate this read.\n"
        "Rules: open interest is written from the SELLER's perspective, so heavy call writing implies resistance and "
        "heavy put writing implies support. Reason only from the data supplied; never invent numbers. If the data is "
        "inconclusive, say WAIT rather than forcing a directional call. Be brief and specific."
    )
    user = "Option chain snapshot:\n" + json.dumps(payload, default=str)
    if str(extra_instructions or "").strip():
        user += "\n\nAdditional analyst instructions: " + extra_instructions.strip()
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {str(api_key).strip()}",
                     "Content-Type": "application/json"},
            json={"model": model, "temperature": float(temperature), "max_tokens": 1400,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=60,
        )
        if resp.status_code != 200:
            try:
                err = resp.json().get("error", {}).get("message", resp.text[:400])
            except ValueError:
                err = resp.text[:400]
            return f"⚠️ Groq API error {resp.status_code}: {err}"
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return "⚠️ Groq returned no choices."
        return (choices[0].get("message", {}) or {}).get("content", "") or "⚠️ Empty response from Groq."
    except Exception as exc:
        return f"⚠️ Groq request failed: {exc}"


# ============================================================================
# PERSISTENCE (SQLite) — optional, disabled by default
# ----------------------------------------------------------------------------
# Without this everything lives in Streamlit session state, which dies when the
# browser tab is discarded or the machine sleeps — taking an OPEN position with
# it. With persistence on, the open position, closed trades, chain snapshots
# and screener runs are written to disk, so a trade that is still running when
# the session drops is restored on the next start and continues to be managed
# until a genuine exit (square-off, SL/target, or a signal exit) closes it.
# ============================================================================

DB_DEFAULT_PATH = "algotrader.db"
NIFTY50_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "LT", "SBIN",
    "BHARTIARTL", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC",
    "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "M&M", "HCLTECH",
    "TECHM", "ADANIENT", "ADANIPORTS", "COALINDIA", "GRASIM", "HINDALCO",
    "DRREDDY", "CIPLA", "DIVISLAB", "BRITANNIA", "EICHERMOT", "HEROMOTOCO",
    "BAJAJ-AUTO", "BAJAJFINSV", "INDUSINDBK", "APOLLOHOSP", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "BPCL", "UPL", "LTIM",
]


NIFTY_NEXT50_SYMBOLS = [
    "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "DMART", "BAJAJHLDNG", "BANKBARODA",
    "BERGEPAINT", "BEL", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DABUR",
    "DLF", "GAIL", "GODREJCP", "HAVELLS", "HAL", "ICICIGI", "ICICIPRULI",
    "IOC", "INDIGO", "NAUKRI", "JINDALSTEL", "JSWENERGY", "LICI", "MARICO",
    "MOTHERSON", "MUTHOOTFIN", "PIDILITIND", "PFC", "PNB", "RECLTD",
    "SIEMENS", "SHREECEM", "SHRIRAMFIN", "SRF", "TVSMOTOR", "TATAPOWER",
    "TORNTPHARM", "TRENT", "UNITDSPR", "VBL", "VEDL", "ZOMATO", "ZYDUSLIFE",
    "ABB", "IRFC", "JIOFIN", "POLYCAB",
]
NIFTY100_SYMBOLS = NIFTY50_SYMBOLS + NIFTY_NEXT50_SYMBOLS


def db_enabled():
    return bool(st.session_state.app_cfg.get("db_enabled", False))


def db_file():
    return str(st.session_state.app_cfg.get("db_path", DB_DEFAULT_PATH) or DB_DEFAULT_PATH).strip()


def db_connect():
    conn = sqlite3.connect(db_file(), check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    """Create tables if missing. Safe to call repeatedly."""
    try:
        with db_connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS open_position (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    saved_at TEXT, ticker TEXT, strategy TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    saved_at TEXT, ticker TEXT, strategy TEXT,
                    entry_time TEXT, exit_time TEXT, direction TEXT,
                    entry_price REAL, exit_price REAL, points REAL, pnl REAL,
                    exit_reason TEXT, qty REAL, payload TEXT);
                CREATE TABLE IF NOT EXISTS chain_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, underlying TEXT, expiry TEXT,
                    spot REAL, futures REAL, pcr REAL, pcr_volume REAL,
                    ce_oi REAL, pe_oi REAL, ce_oi_change REAL, pe_oi_change REAL,
                    ce_volume REAL, pe_volume REAL, max_pain REAL,
                    atm_strike REAL, atm_gamma REAL, atm_straddle REAL, atm_iv REAL,
                    payload TEXT);
                CREATE TABLE IF NOT EXISTS delivery_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    converted_at TEXT, ticker TEXT, strategy TEXT, instrument TEXT,
                    direction TEXT, entry_price REAL, qty REAL, sl REAL, target REAL,
                    status TEXT, resumed_at TEXT, closed_at TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS screener_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, strategy TEXT, interval TEXT, period TEXT,
                    universe TEXT, results TEXT);
                CREATE INDEX IF NOT EXISTS idx_chain_lookup
                    ON chain_snapshots (underlying, expiry, ts);
            """)
        return True, None
    except Exception as exc:
        return False, str(exc)


def db_save_open_position(pos, ticker, strategy):
    if not (db_enabled() and pos):
        return
    try:
        with db_connect() as conn:
            conn.execute("INSERT OR REPLACE INTO open_position (id, saved_at, ticker, strategy, payload) "
                         "VALUES (1, ?, ?, ?, ?)",
                         (ist_now().isoformat(), ticker, strategy, json.dumps(pos, default=str)))
    except Exception as exc:
        st.session_state["db_last_error"] = f"save position: {exc}"


def db_clear_open_position():
    if not db_enabled():
        return
    try:
        with db_connect() as conn:
            conn.execute("DELETE FROM open_position WHERE id = 1")
    except Exception as exc:
        st.session_state["db_last_error"] = f"clear position: {exc}"


def db_load_open_position():
    try:
        with db_connect() as conn:
            row = conn.execute("SELECT * FROM open_position WHERE id = 1").fetchone()
        if not row:
            return None, None, None
        pos = json.loads(row["payload"])
        for k in ("entry_time",):
            if pos.get(k):
                try:
                    pos[k] = pd.to_datetime(pos[k])
                except Exception:
                    pass
        return pos, row["ticker"], row["strategy"]
    except Exception:
        return None, None, None


def db_save_trade(row, ticker, strategy):
    if not db_enabled():
        return
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO trade_history (saved_at, ticker, strategy, entry_time, exit_time, direction, "
                "entry_price, exit_price, points, pnl, exit_reason, qty, payload) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ist_now().isoformat(), ticker, strategy,
                 str(row.get("Entry Time")), str(row.get("Exit Time")), row.get("Direction"),
                 float(row.get("Entry Price") or 0), float(row.get("Exit Price") or 0),
                 float(row.get("Points") or 0), float(row.get("PnL") or 0),
                 row.get("Exit Reason"), float(row.get("Qty") or 0),
                 json.dumps(row, default=str)))
    except Exception as exc:
        st.session_state["db_last_error"] = f"save trade: {exc}"


def db_load_trades(limit=1000):
    try:
        with db_connect() as conn:
            rows = conn.execute("SELECT payload FROM trade_history ORDER BY id DESC LIMIT ?",
                                (int(limit),)).fetchall()
        return [json.loads(r["payload"]) for r in rows][::-1]
    except Exception:
        return []


def db_save_chain_snapshot(snap, underlying, futures=None):
    if not (db_enabled() and snap):
        return
    try:
        mp, _p = compute_max_pain(snap.get("strikes") or {})
        atm, arow = _atm_row(snap)
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO chain_snapshots (ts, underlying, expiry, spot, futures, pcr, pcr_volume, ce_oi, pe_oi, "
                "ce_oi_change, pe_oi_change, ce_volume, pe_volume, max_pain, atm_strike, atm_gamma, atm_straddle, "
                "atm_iv, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ist_now().isoformat(), underlying, snap.get("expiry"),
                 snap.get("underlying"), futures, snap.get("pcr"), snap.get("pcr_volume"),
                 snap.get("ce_oi"), snap.get("pe_oi"), snap.get("ce_oi_change"), snap.get("pe_oi_change"),
                 snap.get("ce_volume"), snap.get("pe_volume"), mp, atm,
                 max(arow.get("ce_gamma", 0), arow.get("pe_gamma", 0)) if arow else None,
                 (arow.get("ce_ltp", 0) + arow.get("pe_ltp", 0)) if arow else None,
                 max(arow.get("ce_iv", 0), arow.get("pe_iv", 0)) if arow else None,
                 json.dumps({"strikes_count": len(snap.get("strikes") or {})})))
    except Exception as exc:
        st.session_state["db_last_error"] = f"save chain: {exc}"


def db_load_chain_history(underlying, expiry=None, since_days=None, limit=20000):
    """Persisted chain snapshots as a DataFrame shaped like the in-memory
    history, so plots can span days rather than just the current session."""
    try:
        q = "SELECT * FROM chain_snapshots WHERE underlying = ?"
        args = [underlying]
        if expiry:
            q += " AND expiry = ?"; args.append(expiry)
        if since_days:
            cutoff = (ist_now() - timedelta(days=int(since_days))).isoformat()
            q += " AND ts >= ?"; args.append(cutoff)
        q += " ORDER BY ts ASC LIMIT ?"; args.append(int(limit))
        with db_connect() as conn:
            rows = [dict(r) for r in conn.execute(q, args).fetchall()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "Timestamp": pd.to_datetime(df["ts"], errors="coerce"),
            "PCR": df["pcr"], "PCR Volume": df["pcr_volume"],
            "Price": df["spot"], "Futures": df["futures"],
            "CE OI": df["ce_oi"], "PE OI": df["pe_oi"],
            "Total OI": df["ce_oi"].fillna(0) + df["pe_oi"].fillna(0),
            "CE ΔOI": df["ce_oi_change"], "PE ΔOI": df["pe_oi_change"],
            "Net ΔOI (PE−CE)": df["pe_oi_change"].fillna(0) - df["ce_oi_change"].fillna(0),
            "CE Volume": df["ce_volume"], "PE Volume": df["pe_volume"],
            "Total Volume": df["ce_volume"].fillna(0) + df["pe_volume"].fillna(0),
            "Max Pain": df["max_pain"], "ATM Strike": df["atm_strike"],
            "ATM Gamma": df["atm_gamma"], "ATM Straddle": df["atm_straddle"],
            "ATM IV": df["atm_iv"],
        })
        out["Time"] = out["Timestamp"].dt.strftime("%d-%b %H:%M:%S")
        return out
    except Exception:
        return pd.DataFrame()


def db_save_delivery_position(pos, ticker, strategy, instrument):
    """Persist a position that was carried overnight as delivery, so it can be
    reviewed and resumed in the Admin Panel on a later day."""
    if not db_enabled():
        return None
    try:
        with db_connect() as conn:
            cur = conn.execute(
                "INSERT INTO delivery_positions (converted_at, ticker, strategy, instrument, direction, "
                "entry_price, qty, sl, target, status, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (ist_now().isoformat(), ticker, strategy, instrument,
                 "LONG" if pos.get("direction") == 1 else "SHORT",
                 float(pos.get("entry_price") or 0), float(pos.get("remaining_qty") or 0),
                 float(pos.get("sl") or 0), float(pos.get("target") or 0),
                 "OPEN", json.dumps(pos, default=str)))
            return cur.lastrowid
    except Exception as exc:
        st.session_state["db_last_error"] = f"save delivery: {exc}"
        return None


def db_load_delivery_positions(status=None):
    try:
        q = "SELECT * FROM delivery_positions"
        args = []
        if status:
            q += " WHERE status = ?"; args.append(status)
        q += " ORDER BY id DESC"
        with db_connect() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]
    except Exception:
        return []


def db_update_delivery_status(row_id, status, extra_field=None):
    try:
        with db_connect() as conn:
            if extra_field:
                conn.execute(f"UPDATE delivery_positions SET status = ?, {extra_field} = ? WHERE id = ?",
                             (status, ist_now().isoformat(), int(row_id)))
            else:
                conn.execute("UPDATE delivery_positions SET status = ? WHERE id = ?", (status, int(row_id)))
        return True
    except Exception as exc:
        st.session_state["db_last_error"] = f"update delivery: {exc}"
        return False


def db_save_screener_run(results_df, strategy, interval, period, universe):
    if not (db_enabled() and results_df is not None and not results_df.empty):
        return
    try:
        with db_connect() as conn:
            conn.execute("INSERT INTO screener_runs (ts, strategy, interval, period, universe, results) "
                         "VALUES (?,?,?,?,?,?)",
                         (ist_now().isoformat(), strategy, interval, period, universe,
                          results_df.to_json(orient="records", date_format="iso")))
    except Exception as exc:
        st.session_state["db_last_error"] = f"save screener: {exc}"


def db_stats():
    try:
        with db_connect() as conn:
            def cnt(t):
                try:
                    return conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
                except Exception:
                    return 0
            return {"trades": cnt("trade_history"), "chain_snapshots": cnt("chain_snapshots"),
                    "screener_runs": cnt("screener_runs"), "open_position": cnt("open_position")}
    except Exception:
        return {}


def db_bootstrap():
    """Run once per session: initialise the schema, restore any trade that was
    still open when the previous session ended, and reload closed trades."""
    if not db_enabled() or st.session_state.get("_db_booted"):
        return
    ok, err = db_init()
    if not ok:
        st.session_state["db_last_error"] = err
        return
    st.session_state["_db_booted"] = True
    if not st.session_state.get("live_positions"):
        pos, tkr, strat = db_load_open_position()
        if pos:
            st.session_state.live_positions = [pos]
            st.session_state["db_restored_note"] = (
                f"Restored an open {pos.get('direction') == 1 and 'LONG' or 'SHORT'} position on {tkr} "
                f"(entry {pos.get('entry_price')}) that was still running when the last session ended. "
                "It will keep being managed until a genuine exit closes it.")
    if not st.session_state.get("live_history"):
        trades = db_load_trades()
        if trades:
            st.session_state.live_history = trades


def db_persist_position_state(ticker, strategy):
    """Mirror the current open position (or its absence) into the database."""
    if not db_enabled():
        return
    pos_list = st.session_state.get("live_positions") or []
    if pos_list:
        db_save_open_position(pos_list[0], ticker, strategy)
    else:
        db_clear_open_position()


# ============================================================================
# FUTURES PRICE + STOCK-UNDERLYING CHAIN RESOLUTION
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def _dhan_future_ltp_cached(security_id, segment, _token_fp):
    return dhan_get_ltp(security_id, segment)


def resolve_chain_underlying(kind, name):
    """
    Resolve either an INDEX (from DHAN_INDEX_MAP) or a STOCK symbol into
    everything the chain/futures APIs need.
    """
    if kind == "Index":
        meta = DHAN_INDEX_MAP.get(name)
        if not meta:
            return None
        return {"label": name, "underlying": meta["underlying"], "security_id": meta["security_id"],
                "segment": meta["segment"], "exchange": meta["exchange"],
                "opt_instrument": "OPTIDX", "fut_instrument": "FUTIDX",
                "yf": TICKER_MAP.get(name)}
    sym = _yf_symbol_to_plain(name)
    eq = dhan_lookup_equity(sym, "NSE")
    if not eq:
        return None
    return {"label": sym, "underlying": sym, "security_id": eq["security_id"],
            "segment": "NSE_EQ", "exchange": "NSE",
            "opt_instrument": "OPTSTK", "fut_instrument": "FUTSTK",
            "yf": f"{sym}.NS"}


def get_futures_price(uinfo, expiry=None):
    """Nearest-expiry futures LTP for the underlying (index or stock)."""
    if not uinfo:
        return None
    _, token = _dhan_creds()
    if not token:
        return None
    try:
        exps = dhan_get_expiries(uinfo["underlying"], uinfo["fut_instrument"], uinfo["exchange"])
        fexp = expiry if (expiry and expiry in exps) else (exps[0] if exps else None)
        if not fexp:
            return None
        info = dhan_lookup_future(uinfo["underlying"], fexp, uinfo["fut_instrument"], uinfo["exchange"])
        if not info:
            return None
        return _dhan_future_ltp_cached(info["security_id"], f"{uinfo['exchange']}_FNO",
                                       hash(token) % 10_000_019)
    except Exception:
        return None


def get_chain_snapshot_for(uinfo, expiry=None):
    """Chain snapshot for a resolved underlying (index OR stock)."""
    if not uinfo:
        return None
    _, token = _dhan_creds()
    if not token:
        return None
    if not expiry:
        exps = dhan_get_expiries(uinfo["underlying"], uinfo["opt_instrument"], uinfo["exchange"])
        expiry = exps[0] if exps else None
    if not expiry:
        return None
    return dhan_get_option_chain(uinfo["security_id"], uinfo["segment"], expiry,
                                 hash(token) % 10_000_019)


# ============================================================================
# HISTORY AGGREGATION (interval / multi-day analysis)
# ============================================================================

CHAIN_AGG_CHOICES = [
    "Current session (every snapshot)",
    "1 min", "5 min", "15 min", "30 min", "60 min",
    "Last 7 days", "Last 30 days", "Last 180 days",
]

_AGG_RULE = {"1 min": "1min", "5 min": "5min", "15 min": "15min",
             "30 min": "30min", "60 min": "60min"}
_AGG_DAYS = {"Last 7 days": (7, "1h"), "Last 30 days": (30, "1D"), "Last 180 days": (180, "1D")}


def aggregate_chain_history(df, mode, underlying_label=None, expiry=None):
    """
    Reshape chain history for the selected analysis window.

    Intraday buckets resample the recorded snapshots; multi-day windows read
    from the database instead, because in-memory history only covers the
    current session. Level metrics (OI, PCR, price) take the LAST value in
    each bucket — the state at the end of the interval — while flow metrics
    (ΔOI, volume) are SUMMED, since those are per-period quantities.
    """
    if mode.startswith("Current"):
        return df, None

    note = None
    if mode in _AGG_DAYS:
        days, rule = _AGG_DAYS[mode]
        if not db_enabled():
            return df, ("Multi-day analysis needs the database enabled (Admin Panel → Data Persistence). "
                        "Showing the current session only — nothing is stored across sessions yet.")
        dbdf = db_load_chain_history(underlying_label, expiry, since_days=days)
        if dbdf.empty:
            return df, (f"No stored snapshots in the last {days} days for this underlying/expiry yet. "
                        "History accumulates while analysis runs with the database enabled.")
        df, note = dbdf, f"Loaded {len(dbdf)} stored snapshots from the database covering up to {days} days."
    else:
        rule = _AGG_RULE.get(mode)
        if not rule:
            return df, None

    if df is None or df.empty or "Timestamp" not in df.columns:
        return df, note
    work = df.copy()
    work["Timestamp"] = pd.to_datetime(work["Timestamp"], errors="coerce")
    work = work.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()
    if work.empty:
        return df, note

    last_cols = ["PCR", "PCR Volume", "Price", "Futures", "CE OI", "PE OI", "Total OI",
                 "Max Pain", "ATM Strike", "ATM Gamma", "ATM Straddle", "ATM IV"]
    sum_cols = ["CE ΔOI", "PE ΔOI", "Net ΔOI (PE−CE)", "CE Volume", "PE Volume",
                "Total Volume", "CE ΔVolume", "PE ΔVolume"]
    agg = {c: "last" for c in last_cols if c in work.columns}
    agg.update({c: "sum" for c in sum_cols if c in work.columns})
    if not agg:
        return df, note
    out = work.resample(rule).agg(agg).dropna(how="all").reset_index()
    out["Time"] = out["Timestamp"].dt.strftime("%d-%b %H:%M")
    # rebuild change columns on the aggregated series
    for col, short in (("PCR", "PCR"), ("Price", "Price"), ("Total OI", "OI"),
                       ("Total Volume", "Volume"), ("Max Pain", "MaxPain"), ("ATM Gamma", "Gamma")):
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce"); prev = s.shift(1)
            out[f"Δ{short} (abs)"] = s - prev
            out[f"Δ{short} (%)"] = (s - prev) / prev.replace(0, np.nan) * 100
            out[f"Δ{short} (n×)"] = s / prev.replace(0, np.nan)
    if note is None:
        note = f"Resampled {len(df)} snapshots into {len(out)} × {mode} buckets."
    return out, note


# ============================================================================
# SCREENER
# ============================================================================

def screener_fetch(symbol, interval, period, source):
    """
    Fetch candles for one screener symbol.

    yfinance is deliberately defensive here: a screener hits dozens of symbols
    in a row, which is exactly the pattern that trips Yahoo's rate limiter. Any
    failure returns an empty frame with a reason instead of raising, so one bad
    symbol can never abort the whole scan.
    """
    yf_symbol = symbol if symbol.endswith((".NS", ".BO")) else f"{symbol}.NS"
    try:
        if source == "Dhan":
            uinfo = resolve_chain_underlying("Stock", symbol)
            if not uinfo:
                return pd.DataFrame(), "not found in Dhan scrip master"
            _, token = _dhan_creds()
            if not token:
                return pd.DataFrame(), "no Dhan token"
            df = _dhan_fetch_candles_cached(uinfo["security_id"], "NSE_EQ", "EQUITY",
                                            interval, period, hash(token) % 10_000_019)
            if df is None or df.empty:
                return pd.DataFrame(), "no Dhan candles"
            return df, None
        if source == "yfinance":
            df = fetch_data_yf(yf_symbol, interval, period)
        else:  # Auto — honour the app's data-source setting
            df = fetch_data(yf_symbol, interval, period)
        if df is None or df.empty:
            return pd.DataFrame(), "no data returned"
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {str(exc)[:80]}"


def screener_scan(symbols, strategy, params, filters, interval, period,
                  source="Auto", before_bars=5, progress=None):
    """
    Run the sidebar's strategy + filters across a list of symbols and bucket
    each result by WHEN its signal fired:

      • Just Now    — signal on the most recently CLOSED candle. This is the
                      bar the live engine would act on right now.
      • Just Before — signal fired 2..N candles ago: already triggered, still
                      recent enough to be worth a look, but the move has begun.
      • Just After  — signal on the last closed candle AND the currently
                      forming candle has already moved further in the signal's
                      direction, i.e. early follow-through is confirming it.

    Every symbol is wrapped so a single failure (rate limit, delisting, bad
    symbol) is recorded and skipped rather than aborting the scan.
    """
    rows, errors = [], []
    total = max(len(symbols), 1)
    for i, sym in enumerate(symbols):
        if progress:
            progress(i / total, sym)
        df, err = screener_fetch(sym, interval, period, source)
        if err or df.empty or len(df) < 5:
            errors.append({"Symbol": sym, "Reason": err or "insufficient candles"})
            continue
        try:
            sig_df = generate_signals(df, strategy, params)
            sig_df = apply_filters(sig_df, filters, strategy)
            sig = sig_df["signal"]
            if len(sig) < 3:
                errors.append({"Symbol": sym, "Reason": "too few bars after indicators"})
                continue

            last_closed = int(sig.iloc[-2])
            window = sig.iloc[max(0, len(sig) - 2 - int(before_bars)):-2]
            prior = window[window != 0]
            close_closed = float(sig_df["Close"].iloc[-2])
            close_now = float(sig_df["Close"].iloc[-1])

            bucket, direction, when, moved = None, None, None, None
            if last_closed != 0:
                direction = "LONG" if last_closed == 1 else "SHORT"
                moved = (close_now - close_closed) * last_closed
                if moved > 0:
                    bucket, when = "Just After", "latest closed bar + follow-through"
                else:
                    bucket, when = "Just Now", "latest closed bar"
            elif len(prior):
                last_idx = prior.index[-1]
                direction = "LONG" if int(prior.iloc[-1]) == 1 else "SHORT"
                bars_ago = len(sig) - 1 - sig.index.get_loc(last_idx)
                bucket, when = "Just Before", f"{bars_ago} bars ago"
                moved = (close_now - float(sig_df["Close"].loc[last_idx])) * int(prior.iloc[-1])

            if not bucket:
                continue
            atr_now = safe_indicator_value(atr(sig_df, 14), 40)[0]
            rows.append({
                "Bucket": bucket, "Symbol": sym, "Direction": direction,
                "Signal When": when,
                "Signal Price": round(close_closed, 2),
                "Last Price": round(close_now, 2),
                "Move Since Signal": round(moved, 2) if moved is not None else None,
                "Move %": round((moved / close_closed * 100), 2) if (moved is not None and close_closed) else None,
                "ATR(14)": round(atr_now, 2) if atr_now else None,
                "Bars": len(sig_df),
                "Last Candle": str(sig_df.index[-1]),
            })
        except Exception as exc:
            errors.append({"Symbol": sym, "Reason": f"{type(exc).__name__}: {str(exc)[:80]}"})
    if progress:
        progress(1.0, "done")
    return pd.DataFrame(rows), pd.DataFrame(errors)


# ---------------------------------------------------------------------------
# STRATEGY EVALUATORS
# ---------------------------------------------------------------------------

def evaluate_oi_change_signal(params, snap):
    """ΔOI dominance, in either Absolute or N× mode."""
    if not snap:
        return 0, ["ΔOI data unavailable — needs a Dhan token, a resolvable expiry, and live market hours."]
    d_ce, d_pe = snap["ce_oi_change"], snap["pe_oi_change"]
    mode = params.get("oi_chg_mode", "Absolute")
    n = params.get("oi_chg_n", 2.0)
    min_chg = float(params.get("oi_chg_min", 0.0))
    flip = bool(params.get("oi_chg_flip", False))
    r_ce, r_pe = _ratio_x(d_ce, d_pe), _ratio_x(d_pe, d_ce)
    ratio_txt = ""
    if r_ce is not None and r_pe is not None:
        ratio_txt = f" → CE/PE = {r_ce:.2f}× · PE/CE = {r_pe:.2f}×"
    lines = [f"ΔCE OI {d_ce:+,.0f} vs ΔPE OI {d_pe:+,.0f}" + ratio_txt
             + f" · mode: {mode}"
             + (f" (either ratio must reach {float(n):.2f}×)" if str(mode).startswith('N') else "")]
    if max(abs(d_ce), abs(d_pe)) < min_chg:
        lines.append(f"❌ Neither side's ΔOI reaches the minimum of {min_chg:,.0f} → no signal.")
        return 0, lines
    side, detail = _side_dominance(d_ce, d_pe, mode, n, return_detail=True)
    if detail.get("basis"):
        lines.append(f"Test: {detail['basis']}.")
    if side is None:
        lines.append("❌ No side dominant under this mode → no signal.")
        return 0, lines
    sig = _chain_side_to_signal(side, flip)
    lines.append(f"✅ {side} ΔOI dominant → {'LONG (BUY CE)' if sig == 1 else 'SHORT (BUY PE)'}"
                 f" ({'flipped' if flip else 'standard seller-perspective'} reading).")
    return sig, lines


def evaluate_oi_volume_signal(params, snap):
    """ΔOI *and* ΔVolume must point at the same side — fresh positions being
    built with real participation behind them, not a stale OI drift."""
    if not snap:
        return 0, ["ΔOI/ΔVolume data unavailable — needs a Dhan token, a resolvable expiry, and live market hours."]
    d_ce, d_pe = snap["ce_oi_change"], snap["pe_oi_change"]
    v_ce, v_pe = snap.get("ce_volume_change", 0.0), snap.get("pe_volume_change", 0.0)
    oi_mode, oi_n = params.get("oiv_oi_mode", "Absolute"), params.get("oiv_oi_n", 2.0)
    vol_mode, vol_n = params.get("oiv_vol_mode", "Absolute"), params.get("oiv_vol_n", 2.0)
    flip = bool(params.get("oiv_flip", False))
    xo_ce, xo_pe = _ratio_x(d_ce, d_pe), _ratio_x(d_pe, d_ce)
    xv_ce, xv_pe = _ratio_x(v_ce, v_pe), _ratio_x(v_pe, v_ce)
    _rt = lambda a, b: (f" (CE/PE {a:.2f}× · PE/CE {b:.2f}×)" if (a is not None and b is not None) else "")
    lines = [
        f"ΔOI: CE {d_ce:+,.0f} vs PE {d_pe:+,.0f}" + _rt(xo_ce, xo_pe)
        + f" · mode {oi_mode}" + (f" (either ratio ≥ {float(oi_n):.2f}×)" if str(oi_mode).startswith('N') else ""),
        f"ΔVolume: CE {v_ce:+,.0f} vs PE {v_pe:+,.0f}" + _rt(xv_ce, xv_pe)
        + f" · mode {vol_mode}" + (f" (either ratio ≥ {float(vol_n):.2f}×)" if str(vol_mode).startswith('N') else ""),
    ]
    oi_side, oi_detail = _side_dominance(d_ce, d_pe, oi_mode, oi_n, return_detail=True)
    vol_side, vol_detail = _side_dominance(v_ce, v_pe, vol_mode, vol_n, return_detail=True)
    if oi_detail.get("basis"):
        lines.append(f"ΔOI test: {oi_detail['basis']}.")
    if vol_detail.get("basis"):
        lines.append(f"ΔVolume test: {vol_detail['basis']}.")
    if oi_side is None or vol_side is None:
        lines.append(f"❌ Needs BOTH: ΔOI side = {oi_side or 'none'}, ΔVolume side = {vol_side or 'none'} → no signal.")
        return 0, lines
    if oi_side != vol_side:
        lines.append(f"❌ Disagreement — ΔOI favours {oi_side} but ΔVolume favours {vol_side} → no signal.")
        return 0, lines
    sig = _chain_side_to_signal(oi_side, flip)
    lines.append(f"✅ Both ΔOI and ΔVolume favour {oi_side} → "
                 f"{'LONG (BUY CE)' if sig == 1 else 'SHORT (BUY PE)'}"
                 f" ({'flipped' if flip else 'standard'} reading).")
    return sig, lines


def evaluate_pcr_signal(params, snap):
    """
    PCR strategy.

    Design rationale: PCR = total PE OI / total CE OI. A HIGH PCR means puts
    are being written heavily — writers are confident price stays up — which
    is read as bullish; a LOW PCR is the mirror, bearish. Rather than trading
    a single line, this uses two bands with a deliberate no-trade zone in the
    middle (most of the session sits there and is noise), and can additionally
    require that PCR is moving the right way, which filters the common failure
    of entering a stretched ratio just as it starts unwinding.

    Optional extreme-reversal mode inverts the logic beyond very high/low
    readings, where the ratio is usually a crowded-positioning warning rather
    than a trend confirmation.
    """
    if not snap:
        return 0, ["PCR data unavailable — needs a Dhan token, a resolvable expiry, and live market hours."]
    pcr = snap.get("pcr")
    if not pcr:
        return 0, ["PCR could not be computed (no CE open interest in the chain)."]
    bull = float(params.get("pcr_bull", 1.2))
    bear = float(params.get("pcr_bear", 0.8))
    need_trend = bool(params.get("pcr_require_trend", False))
    extreme_mode = bool(params.get("pcr_extreme_reversal", False))
    ex_hi = float(params.get("pcr_extreme_high", 1.8))
    ex_lo = float(params.get("pcr_extreme_low", 0.5))
    flip = bool(params.get("pcr_flip", False))

    hist = st.session_state.get("chain_history", [])
    prev_pcr = None
    for row in reversed(hist[:-1] if hist else []):
        if row.get("PCR"):
            prev_pcr = row["PCR"]
            break
    d_pcr = (pcr - prev_pcr) if prev_pcr else None

    lines = [f"PCR = {pcr:.3f} (bullish ≥ {bull:.2f} · bearish ≤ {bear:.2f} · no-trade zone between)"
             + (f" · ΔPCR vs previous reading {d_pcr:+.3f}" if d_pcr is not None else " · no previous reading yet")]
    if snap.get("pcr_volume"):
        lines.append(f"Volume-based PCR = {snap['pcr_volume']:.3f} (context only — OI PCR drives the signal).")

    sig = 0
    if extreme_mode and pcr >= ex_hi:
        sig = -1
        lines.append(f"⚠️ PCR ≥ extreme {ex_hi:.2f} → treated as crowded put-writing, REVERSAL short bias.")
    elif extreme_mode and pcr <= ex_lo:
        sig = 1
        lines.append(f"⚠️ PCR ≤ extreme {ex_lo:.2f} → treated as crowded call-writing, REVERSAL long bias.")
    elif pcr >= bull:
        sig = 1
        lines.append("✅ PCR in the bullish band (heavy put writing) → LONG bias.")
    elif pcr <= bear:
        sig = -1
        lines.append("✅ PCR in the bearish band (heavy call writing) → SHORT bias.")
    else:
        lines.append("❌ PCR inside the no-trade zone → no signal.")
        return 0, lines

    if need_trend:
        if d_pcr is None:
            lines.append("❌ 'Require PCR trend confirmation' is ON but there is no previous reading yet → no signal.")
            return 0, lines
        if sig == 1 and d_pcr <= 0:
            lines.append(f"❌ Long bias needs PCR RISING, but ΔPCR is {d_pcr:+.3f} → no signal.")
            return 0, lines
        if sig == -1 and d_pcr >= 0:
            lines.append(f"❌ Short bias needs PCR FALLING, but ΔPCR is {d_pcr:+.3f} → no signal.")
            return 0, lines
        lines.append(f"✅ PCR trend confirms ({d_pcr:+.3f}).")

    if flip:
        sig = -sig
        lines.append("🔄 Flip enabled → direction inverted.")
    lines.append(f"Verdict: {'LONG (BUY CE)' if sig == 1 else 'SHORT (BUY PE)'}.")
    return sig, lines


def evaluate_gamma_blast_signal(params, snap, df=None):
    """
    Gamma blast.

    The setup this looks for is the expiry-day pattern where ATM options have
    collapsed to a small premium while gamma is at its highest: the writers'
    hedges become extremely sensitive to price, so once price breaks out of
    its compression range the delta-hedging feedback loop can expand the
    premium several-fold in minutes. All four conditions must line up:
      1. within N days of expiry (default 0 = expiry day only),
      2. combined ATM straddle premium below a ceiling (the compression),
      3. ATM gamma at or above a floor (the fuel),
      4. price breaking out of its recent range (the trigger, and the
         direction — a break up buys CE, a break down buys PE).
    Condition 4 is what makes this directional rather than a guess; without a
    break there is no signal at all.
    """
    lines = []
    if not snap or not snap.get("strikes"):
        return 0, ["Gamma Blast: option chain unavailable — needs a Dhan token, an expiry, and live market hours."]
    max_dte = int(params.get("gb_max_dte", 0))
    prem_cap = float(params.get("gb_premium_cap", 60.0))
    gamma_min = float(params.get("gb_gamma_min", 0.0))
    lookback = int(params.get("gb_range_lookback", 15))
    buffer_pts = float(params.get("gb_break_buffer", 0.0))
    flip = bool(params.get("gb_flip", False))

    dte = days_to_expiry(snap.get("expiry"))
    if dte is None:
        lines.append("Days-to-expiry unknown — skipping the expiry check.")
    else:
        lines.append(f"Days to expiry = {dte} (needs ≤ {max_dte})."
                     + (" ✅" if dte <= max_dte else " ❌"))
        if dte > max_dte:
            return 0, lines + ["❌ Too far from expiry for a gamma blast → no signal."]

    band = multi_strike_band(snap, 0)
    if not band:
        return 0, lines + ["Gamma Blast: could not locate the ATM strike."]
    atm = band["atm"]
    row = snap["strikes"][atm]
    straddle = row["ce_ltp"] + row["pe_ltp"]
    gamma = max(row["ce_gamma"], row["pe_gamma"])
    lines.append(f"ATM {atm:.0f}: CE {row['ce_ltp']:.2f} + PE {row['pe_ltp']:.2f} = straddle {straddle:.2f} "
                 f"(needs ≤ {prem_cap:.2f})" + (" ✅" if straddle <= prem_cap else " ❌"))
    lines.append(f"ATM gamma = {gamma:.5f} (needs ≥ {gamma_min:.5f})" + (" ✅" if gamma >= gamma_min else " ❌"))
    if straddle > prem_cap:
        return 0, lines + ["❌ Premium not compressed enough → no signal."]
    if gamma < gamma_min:
        return 0, lines + ["❌ ATM gamma below the floor → no signal."]

    if df is None or len(df) < lookback + 2:
        return 0, lines + [f"❌ Need at least {lookback + 2} candles to measure the compression range → no signal."]
    window = df.iloc[-(lookback + 1):-1]
    hi, lo = float(window["High"].max()), float(window["Low"].min())
    px = float(df["Close"].iloc[-1])
    lines.append(f"Compression range over last {lookback} candles: {lo:.2f} – {hi:.2f}; price {px:.2f} "
                 f"(break buffer {buffer_pts:.2f}).")
    sig = 0
    if px > hi + buffer_pts:
        sig = 1
        lines.append("✅ Upside break → BUY CE (gamma blast long).")
    elif px < lo - buffer_pts:
        sig = -1
        lines.append("✅ Downside break → BUY PE (gamma blast short).")
    else:
        lines.append("❌ Still inside the range — waiting for the break that triggers the blast.")
        return 0, lines
    if flip:
        sig = -sig
        lines.append("🔄 Flip enabled → direction inverted.")
    return sig, lines


def evaluate_multi_strike_signal(params, snap, spot=None):
    """
    Multi-strike ATM ± N levels.

    Sums CE and PE open interest, OI change and volume across the ATM strike
    and N strikes either side (so N=3 covers seven strikes), computes that
    band's own PCR, and locates max pain over the full chain. Trading only the
    strikes around the money keeps the read focused on where the action
    actually is, instead of letting far-OTM legs dominate the totals.

    Scoring combines three independent votes — band PCR, ΔOI dominance, and
    where spot sits relative to max pain — and requires a configurable minimum
    net score, so a single ambiguous input cannot trigger an entry on its own.
    """
    if not snap:
        return 0, ["Multi-strike: option chain unavailable — needs a Dhan token, an expiry, and live market hours."]
    levels = int(params.get("ms_levels", 3))
    band = multi_strike_band(snap, levels, spot)
    if not band:
        return 0, ["Multi-strike: could not build the strike band."]
    bull_pcr = float(params.get("ms_pcr_bull", 1.2))
    bear_pcr = float(params.get("ms_pcr_bear", 0.8))
    oi_mode, oi_n = params.get("ms_oi_mode", "Absolute"), params.get("ms_oi_n", 2.0)
    min_votes = int(params.get("ms_min_votes", 2))
    use_maxpain = bool(params.get("ms_use_max_pain", True))
    flip = bool(params.get("ms_flip", False))

    lines = [
        f"ATM {band['atm']:.0f} ± {levels} levels → {len(band['band'])} strikes "
        f"({min(band['band']):.0f}–{max(band['band']):.0f}), spot {band['spot']:.2f}",
        f"Band CE OI {band['ce_oi']:,.0f} vs PE OI {band['pe_oi']:,.0f} → band PCR "
        + (f"{band['pcr']:.3f}" if band['pcr'] else "n/a")
        + f" (bullish ≥ {bull_pcr:.2f} / bearish ≤ {bear_pcr:.2f})",
        f"Band ΔOI: CE {band['ce_oi_change']:+,.0f} vs PE {band['pe_oi_change']:+,.0f} · mode {oi_mode}"
        + (f" ≥{float(oi_n):.1f}×" if str(oi_mode).startswith('N') else ""),
        f"Band volume: CE {band['ce_volume']:,.0f} vs PE {band['pe_volume']:,.0f}"
        + (f" → volume PCR {band['pcr_volume']:.3f}" if band.get('pcr_volume') else ""),
        f"Max pain (full chain) = {band['max_pain']:.0f}" if band.get("max_pain") else "Max pain unavailable.",
    ]

    votes = 0
    if band["pcr"]:
        if band["pcr"] >= bull_pcr:
            votes += 1; lines.append("🟢 Vote +1: band PCR bullish (put writing dominant).")
        elif band["pcr"] <= bear_pcr:
            votes -= 1; lines.append("🔴 Vote −1: band PCR bearish (call writing dominant).")
        else:
            lines.append("⚪ Vote 0: band PCR inside the neutral zone.")

    oi_side = _side_dominance(band["ce_oi_change"], band["pe_oi_change"], oi_mode, oi_n)
    if oi_side == "PE":
        votes += 1; lines.append("🟢 Vote +1: PE ΔOI dominant (puts being written → support building).")
    elif oi_side == "CE":
        votes -= 1; lines.append("🔴 Vote −1: CE ΔOI dominant (calls being written → resistance building).")
    else:
        lines.append("⚪ Vote 0: no clear ΔOI dominance in the band.")

    if use_maxpain and band.get("max_pain"):
        if band["spot"] < band["max_pain"]:
            votes += 1; lines.append(f"🟢 Vote +1: spot {band['spot']:.2f} is BELOW max pain {band['max_pain']:.0f} "
                                     "(drift toward max pain is upward).")
        elif band["spot"] > band["max_pain"]:
            votes -= 1; lines.append(f"🔴 Vote −1: spot {band['spot']:.2f} is ABOVE max pain {band['max_pain']:.0f} "
                                     "(drift toward max pain is downward).")
        else:
            lines.append("⚪ Vote 0: spot sits at max pain.")

    lines.append(f"Net score {votes:+d}, needs |score| ≥ {min_votes}.")
    sig = 0
    if votes >= min_votes:
        sig = 1
    elif votes <= -min_votes:
        sig = -1
    else:
        lines.append("❌ Not enough agreement → no signal.")
        return 0, lines
    if flip:
        sig = -sig
        lines.append("🔄 Flip enabled → direction inverted.")
    lines.append(f"✅ Verdict: {'BUY CE (long)' if sig == 1 else 'BUY PE (short)'}.")
    return sig, lines


def evaluate_option_chain_signal(strategy, params, df=None):
    """Single dispatcher for every option-chain strategy. Returns (sig, lines)."""
    snap = get_oi_snapshot()
    record_chain_history(snap)
    spot = None
    if df is not None and len(df):
        try:
            spot = float(df["Close"].iloc[-1])
        except Exception:
            spot = None
    if strategy == "OI Based (CE/PE Open Interest)":
        return evaluate_oi_signal(params, snap)
    if strategy == "OI Change Based (ΔOI)":
        return evaluate_oi_change_signal(params, snap)
    if strategy == "OI + Volume Change Based":
        return evaluate_oi_volume_signal(params, snap)
    if strategy == "PCR Based (Put-Call Ratio)":
        return evaluate_pcr_signal(params, snap)
    if strategy == "Gamma Blast (Expiry Momentum)":
        return evaluate_gamma_blast_signal(params, snap, df)
    if strategy == "Multi-Strike OI (ATM ± N Levels)":
        return evaluate_multi_strike_signal(params, snap, spot)
    return 0, []


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

    # Both keys must share the same datetime unit — candle indexes and the VIX
    # series can arrive at different resolutions (s vs us vs ns) in pandas 2/3.
    left = pd.DataFrame({"t": _norm_ts(tgt_naive)})
    right = pd.DataFrame({"t": _norm_ts(vix_clean.index), "vix": vix_clean.values}).sort_values("t")
    merged = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward")

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

def generate_signals(df, strategy, params, _raw=False):
    df = df.copy()
    df["signal"] = 0

    if strategy in OPTION_CHAIN_STRATEGIES:
        # Live option-chain snapshot — there is no historical OI/volume series
        # available, so only the LATEST bar can carry a signal. These are
        # immediate-execution strategies: the live engine reads the snapshot
        # directly and enters at LTP, so this marking exists for charting and
        # status display rather than for backtesting (a backtest cannot
        # reconstruct past chains and will not produce trades).
        sig, _lines = evaluate_option_chain_signal(strategy, params, df)
        if sig != 0 and len(df):
            df.iloc[-1, df.columns.get_loc("signal")] = sig

    elif strategy == "Hybrid (Combine Strategies)":
        members = list(params.get("hybrid_members", []))
        mode = params.get("hybrid_mode", "AND")
        if members:
            longs, shorts = [], []
            for m in members:
                if m == "Hybrid (Combine Strategies)":
                    continue    # no self-recursion
                sub = generate_signals(df, m, params, _raw=True)
                s = sub["signal"].reindex(df.index).fillna(0)
                longs.append(s == 1)
                shorts.append(s == -1)
                # keep each member's indicator columns for charting/status
                for col in sub.columns:
                    if col not in df.columns and col != "signal":
                        df[col] = sub[col]
            if longs:
                if str(mode).upper().startswith("AND"):
                    # ALL selected strategies must agree on the same bar
                    long_mask = np.logical_and.reduce([m.values for m in longs])
                    short_mask = np.logical_and.reduce([m.values for m in shorts])
                else:  # OR — any single member firing is enough
                    long_mask = np.logical_or.reduce([m.values for m in longs])
                    short_mask = np.logical_or.reduce([m.values for m in shorts])
                # A bar where both sides qualify is contradictory → no trade
                both = long_mask & short_mask
                long_mask = long_mask & ~both
                short_mask = short_mask & ~both
                df.loc[long_mask, "signal"] = 1
                df.loc[short_mask, "signal"] = -1

    elif strategy == "EMA Crossover":
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
        #   "Below" (default) → LONG when price is/goes ABOVE the threshold.
        #   "Above"           → SHORT when price is/goes BELOW the threshold.
        # Trigger Mode:
        #   "Level (fire whenever price is beyond the threshold)" — DEFAULT.
        #     This is an order-style trigger: it does NOT require the app to
        #     have witnessed the exact crossing bar, so it still fires when
        #     price crossed before you started, or between polls. This is why
        #     the old cross-only behaviour looked dead.
        #   "Cross event (needs the actual crossing bar)" — the stricter,
        #     original behaviour: fires only on the bar where price crossed.
        cross_dir = params.get("threshold_direction", "Below")
        mode = params.get("threshold_trigger_mode", "Level")
        if str(mode).startswith("Cross"):
            if cross_dir == "Above":
                df.loc[(df["Close"] < thr) & (df["Close"].shift(1) >= thr), "signal"] = -1
            else:
                df.loc[(df["Close"] > thr) & (df["Close"].shift(1) <= thr), "signal"] = 1
        else:
            if cross_dir == "Above":
                df.loc[df["Close"] < thr, "signal"] = -1
            else:
                df.loc[df["Close"] > thr, "signal"] = 1

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
        buy_lvl = params.get("rsi_buy_level", 30.0)
        sell_lvl = params.get("rsi_sell_level", 70.0)
        # Cross direction is now explicit for each side:
        #   BUY  "Up-cross (from below)"  → RSI rises THROUGH the buy level
        #        "Down-cross (from above)" → RSI falls INTO oversold
        #   SELL "Down-cross (from above)" → RSI falls THROUGH the sell level
        #        "Up-cross (from below)"   → RSI rises INTO overbought
        buy_dir = params.get("rsi_buy_cross", "Up-cross (from below)")
        sell_dir = params.get("rsi_sell_cross", "Down-cross (from above)")
        if buy_dir.startswith("Up"):
            df.loc[(r > buy_lvl) & (r.shift(1) <= buy_lvl), "signal"] = 1
        else:
            df.loc[(r < buy_lvl) & (r.shift(1) >= buy_lvl), "signal"] = 1
        if sell_dir.startswith("Down"):
            df.loc[(r < sell_lvl) & (r.shift(1) >= sell_lvl), "signal"] = -1
        else:
            df.loc[(r > sell_lvl) & (r.shift(1) <= sell_lvl), "signal"] = -1

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
        lb = params.get("zigzag_lookback", 3)
        piv = elliott_wave_state(df, lb)
        df["swing_high"], df["swing_low"] = piv["raw_high"], piv["raw_low"]
        df["ew_pivot_price"] = piv["pivot_price"]
        df["ew_pivot_kind"] = piv["pivot_kind"]
        df["ew_wave_label"] = piv["wave_label"]
        # Signals fire on the CONFIRMATION bar, never retroactively on the
        # pivot bar itself. A pivot at bar i can only be known at bar i+lb
        # (it needs lb bars on its right), so the old version — which marked
        # the pivot bar — was using future information and could NEVER fire
        # on the newest bar, which is why it worked in backtest and was dead
        # in live. Firing at confirmation makes backtest and live identical.
        df.loc[piv["confirm_low"], "signal"] = 1     # confirmed trough → long
        df.loc[piv["confirm_high"], "signal"] = -1   # confirmed peak → short
        if params.get("ew_impulse_only", False):
            # Optional stricter mode: only trade pivots that continue the
            # impulse structure (higher-low in an uptrend / lower-high in a
            # downtrend), i.e. wave 3 / wave 5 starts rather than every swing.
            df.loc[(df["signal"] == 1) & (~piv["higher_low"]), "signal"] = 0
            df.loc[(df["signal"] == -1) & (~piv["lower_high"]), "signal"] = 0

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
    if _raw:
        # Sub-strategy call from the Hybrid combiner: return RAW signals so
        # flip/direction rules are applied ONCE, to the combined result.
        return df
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
    o = float(candle["Open"])

    # GAP HANDLING (important for Indian markets): if the candle OPENED
    # already beyond the level (overnight gap-up / gap-down, or an opening
    # gap on the first bar of the session), no fill at the level itself was
    # ever possible — the realistic fill is the gapped open. SLs therefore
    # fill at the WORSE of (level, open) and targets at the BETTER of
    # (level, open), with the reason annotated so gap exits are auditable.
    if direction == 1:
        if candle["Low"] <= trade["sl"]:
            if o < trade["sl"]:
                return True, o, "Stoploss Hit (gap-down — filled @ open)"
            return True, trade["sl"], "Stoploss Hit"
        if not target_display_only and candle["High"] >= trade["target"]:
            if o > trade["target"]:
                return True, o, "Target Hit (gap-up — filled @ open)"
            return True, trade["target"], "Target Hit"
    else:
        if candle["High"] >= trade["sl"]:
            if o > trade["sl"]:
                return True, o, "Stoploss Hit (gap-up — filled @ open)"
            return True, trade["sl"], "Stoploss Hit"
        if not target_display_only and candle["Low"] <= trade["target"]:
            if o < trade["target"]:
                return True, o, "Target Hit (gap-down — filled @ open)"
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
            if ltp < trade["sl"]:
                # Price is already BELOW the level (e.g. an overnight gap-down
                # or a fast move between polls) — a live exit can only fill at
                # the market price, not back at the level.
                return True, float(ltp), "Stoploss Hit (LTP gapped past level)"
            return True, trade["sl"], "Stoploss Hit (LTP)"
        if not target_display_only and ltp >= trade["target"]:
            if ltp > trade["target"]:
                return True, float(ltp), "Target Hit (LTP gapped past level)"
            return True, trade["target"], "Target Hit (LTP)"
    else:
        if ltp >= trade["sl"]:
            if ltp > trade["sl"]:
                return True, float(ltp), "Stoploss Hit (LTP gapped past level)"
            return True, trade["sl"], "Stoploss Hit (LTP)"
        if not target_display_only and ltp <= trade["target"]:
            if ltp < trade["target"]:
                return True, float(ltp), "Target Hit (LTP gapped past level)"
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

def _email_status(ok, message):
    """Record the outcome of the last send so the Live tab can show it —
    warnings raised inside a fragment are easy to miss entirely."""
    st.session_state["email_last_status"] = {
        "ok": ok, "message": message, "at": ist_now().strftime("%d-%b %H:%M:%S IST")}
    return ok


def send_trade_email(cfg, subject, body_lines, force=False):
    """
    Gmail SMTP over SSL:465. Never raises — a mail problem must not interrupt
    trading, so every outcome is recorded via _email_status() and surfaced on
    the Live tab instead of only being warned about inside a fragment.
    """
    if not (cfg.get("email_enabled") or force):
        return False
    sender = str(cfg.get("email_from") or "").strip()
    recipients = [r.strip() for r in str(cfg.get("email_to") or "").split(",") if r.strip()]
    # Google DISPLAYS app passwords in four space-separated blocks; pasting
    # that verbatim is the most common reason login fails, so strip whitespace
    # rather than rejecting a password the user copied exactly as shown.
    app_password = "".join(str(cfg.get("email_app_password") or "").split())

    if not sender:
        return _email_status(False, "No From address set.")
    if not recipients:
        return _email_status(False, "No To address set.")
    if not app_password:
        return _email_status(False, "No Gmail App Password set.")
    if len(app_password) != 16:
        return _email_status(False, (
            f"App Password is {len(app_password)} characters after removing spaces — Google app passwords are "
            "exactly 16. This is an App Password from myaccount.google.com/apppasswords, NOT your normal Gmail "
            "password (which SMTP always rejects)."))
    try:
        msg = MIMEText("\n".join(str(x) for x in body_lines))
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
            server.login(sender, app_password)
            server.sendmail(sender, recipients, msg.as_string())
        return _email_status(True, f"Sent to {', '.join(recipients)} — “{subject}”.")
    except smtplib.SMTPAuthenticationError as exc:
        return _email_status(False, (
            f"Gmail rejected the login ({exc.smtp_code}). Use a 16-character App Password from "
            "myaccount.google.com/apppasswords with 2-Step Verification enabled, and make sure the From "
            "address is that same Google account."))
    except smtplib.SMTPRecipientsRefused:
        return _email_status(False, f"Recipient address refused: {', '.join(recipients)}.")
    except Exception as exc:
        return _email_status(False, f"{type(exc).__name__}: {exc}")


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


def check_delivery_conversion(cfg, pos, ticker, strategy):
    """
    Intraday → delivery carry-over.

    Intraday products are auto-squared-off by the broker near the close, so a
    position that has hit neither SL nor target would be closed at whatever
    price the market happens to be at. When enabled (default), at the
    configured cut-off the position is instead marked as DELIVERY: it stays
    open, is written to the database, and can be reviewed and resumed from the
    Admin Panel on a later day.

    Applies to stock intraday, stock options and index options — anything held
    on an intraday product. Positions already converted are not converted
    twice, and nothing is converted if SL/target already closed the trade.
    """
    if not cfg or not pos:
        return False, None
    if not cfg.get("convert_to_delivery", True):
        return False, None
    if pos.get("converted_to_delivery"):
        return False, None
    cutoff = cfg.get("delivery_cutoff_time", dtime(15, 0))
    if not isinstance(cutoff, dtime):
        try:
            hh, mm = str(cutoff).split(":")[:2]
            cutoff = dtime(int(hh), int(mm))
        except Exception:
            cutoff = dtime(15, 0)
    now = ist_now()
    if now.weekday() >= 5 or now.time() < cutoff:
        return False, None
    product_cfg = cfg.get("product_cfg") or {}
    instrument = str(product_cfg.get("instrument") or ("Options" if product_cfg.get("options_mode") else "Equity"))
    pos["converted_to_delivery"] = True
    pos["delivery_converted_at"] = now.isoformat()
    row_id = db_save_delivery_position(pos, ticker, strategy, instrument)
    pos["delivery_row_id"] = row_id
    msg = (f"Converted to DELIVERY at {now.strftime('%H:%M IST')} (cut-off {cutoff.strftime('%H:%M')}) — "
           f"SL/target had not been hit. The position stays open and is stored"
           + (" in the database; resume it from the Admin Panel." if row_id else
              " in this session only (enable Data Persistence to store it)."))
    return True, msg


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
# CONFIGURATION CONTROLS — rendered exactly ONCE, into the sidebar
# ----------------------------------------------------------------------------
# render_config_controls() builds every control through the cfg_* wrappers
# above and returns the assembled config dict. It is called from exactly one
# place (the sidebar). The "🛠 Admin Panel" tab does NOT re-render these
# controls — it shows a read-only summary — because rendering two editable
# copies per run is what previously made selections snap back.
# The Optimization tab's "apply config" writes through cfg_set() + rerun.
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


def render_config_controls(ui, prefix="sb"):
    """Renders the full control set into `ui` (the sidebar) and returns the
    assembled config dict. All original controls, defaults, and captions are
    preserved. `prefix` is retained for signature stability but there is only
    one rendering of these controls, so widget keys never collide."""
    store = _cfg_store()
    ui.title("⚙️ Algo Configuration")

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
                    cfg_force("prem_security_id", info["security_id"])
                    store["_prem_lot_size"] = info.get("lot_size")
                    return True
                return False

            if st.session_state.get("dhan_opt_autofill_sig") != prem_sig \
                    and st.session_state.get("_attempted_dhan_opt_autofill_sig") != prem_sig:
                cfg_force("prem_security_id", "")   # sig change ALWAYS clears stale IDs
            _try_autofill(prem_sig, _fetch_prem_id, "dhan_opt_autofill_sig", "dhan_opt_autofill_last_try")

            prem_id = cfg_text(ui, f"{prem_opt_type} Security ID (auto-filled, editable)",
                               "prem_security_id", "", prefix=prefix).strip()
            prem_segment = f"{prem_exchange}_FNO"
            # Sentinel ticker → fetch_data / get_live_ltp serve THIS option's
            # premium candles + premium LTP straight from Dhan (no delay).
            ticker = f"DHANOPT::{prem_segment}::{prem_id}::{prem_instr}"
            underlying_choice = "Options Trading"

            if store.get("_prem_qty_default_sig") != (prem_u, prem_expiry, prem_opt_type):
                cfg_force("dhan_qty", int(prem_default_qty or store.get("_prem_lot_size") or 1))
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

    _tf_map = available_tf_period_map()
    intervals = list(_tf_map.keys())
    interval = cfg_selectbox(ui, "Timeframe", "interval", intervals, default="1m", prefix=prefix)
    if interval in DHAN_EXTRA_TF_PERIODS:
        ui.caption(f"⚡ {interval} is a Dhan-only granularity (yfinance does not offer it). It is served from Dhan "
                   "and will fall back to the nearest supported timeframe if the feed is turned off.")
    periods_available = _tf_map.get(interval, TF_PERIOD_MAP.get(interval, ["1d"]))
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
        params["threshold_trigger_mode"] = cfg_selectbox(
            ui, "Trigger Mode", "threshold_trigger_mode",
            ["Level (fire whenever price is beyond the threshold)",
             "Cross event (needs the actual crossing bar)"],
            default="Level (fire whenever price is beyond the threshold)", prefix=prefix)
        ui.caption("Below = LONG when price is/goes ABOVE the threshold. Above = SHORT when price is/goes BELOW it. "
                   "**Level mode (default)** behaves like a resting order: it fires the instant the LIVE LTP is on the "
                   "trigger side — no candle close needed and no requirement that the app witnessed the exact crossing "
                   "bar (that requirement is what made this look dead before). Cross-event mode is the stricter "
                   "original behaviour.")
    if strategy == "RSI Cross":
        params["rsi_period"] = cfg_number(ui, "RSI Period", "rsi_period", 14, 2, 50, is_int=True, prefix=prefix)
        c1, c2 = ui.columns(2)
        params["rsi_buy_level"] = cfg_number(c1, "RSI Buy Level", "rsi_buy_level", 30.0, 1.0, 99.0, prefix=prefix)
        params["rsi_sell_level"] = cfg_number(c2, "RSI Sell Level", "rsi_sell_level", 70.0, 1.0, 99.0, prefix=prefix)
        params["rsi_buy_cross"] = cfg_selectbox(
            ui, "BUY fires on", "rsi_buy_cross",
            ["Up-cross (from below)", "Down-cross (from above)"],
            default="Up-cross (from below)", prefix=prefix)
        params["rsi_sell_cross"] = cfg_selectbox(
            ui, "SELL fires on", "rsi_sell_cross",
            ["Down-cross (from above)", "Up-cross (from below)"],
            default="Down-cross (from above)", prefix=prefix)
        ui.caption(f"Default: BUY when RSI rises UP THROUGH {params['rsi_buy_level']:.0f} (recovering out of oversold) "
                   f"and SELL when RSI falls DOWN THROUGH {params['rsi_sell_level']:.0f}. Switch either dropdown to "
                   "trade the opposite crossing direction (e.g. buy as RSI drops INTO oversold) — both levels and both "
                   "directions are fully configurable.")
    if strategy == "Elliott Wave (Zigzag)":
        params["zigzag_lookback"] = cfg_number(ui, "Zigzag Lookback (bars each side of a pivot)",
                                               "zigzag_lookback", 3, 2, 20, is_int=True, prefix=prefix)
        params["ew_impulse_only"] = cfg_checkbox(ui, "Impulse structure only (higher-lows / lower-highs)",
                                                 "ew_impulse_only", False, prefix=prefix)
        ui.caption(f"A pivot needs {params['zigzag_lookback']} bars on each side, so it becomes tradeable "
                   f"{params['zigzag_lookback']} bars AFTER it forms — signals now fire on that confirmation bar in "
                   "both backtest and live (previously they were stamped on the pivot bar itself, which used future "
                   "bars and could never fire on the newest candle — that's why it worked only in backtest). "
                   "Impulse-only restricts entries to swings that continue the structure.")
    if strategy in OPTION_CHAIN_STRATEGIES:
        # ---- shared chain source (all option-chain strategies use this) ----
        params["oi_underlying"] = cfg_selectbox(ui, "Option Chain Underlying", "oi_underlying",
                                                list(DHAN_INDEX_MAP.keys()), default="Nifty50", prefix=prefix)
        _oi_meta = DHAN_INDEX_MAP[params["oi_underlying"]]
        _oi_exps = dhan_get_expiries(_oi_meta["underlying"], "OPTIDX", _oi_meta["exchange"])
        if _oi_exps:
            params["oi_expiry"] = cfg_selectbox(ui, "Chain Expiry (nearest pre-selected)", "oi_expiry",
                                                _oi_exps, default=_oi_exps[0], prefix=prefix)
        else:
            params["oi_expiry"] = cfg_text(ui, "Chain Expiry (YYYY-MM-DD)", "oi_expiry", "", prefix=prefix)
        ui.caption("⚡ All option-chain strategies enter IMMEDIATELY at LTP the moment their condition is met — no "
                   "candle close required, since the chain is a live snapshot. They need a Dhan token, and the chain "
                   "refreshes every 60s (Dhan rate-limits that endpoint). They cannot be backtested: Dhan exposes only "
                   "the current chain, never historical OI/volume.")

    if strategy == "OI Based (CE/PE Open Interest)":
        c1, c2 = ui.columns(2)
        params["oi_ce_threshold"] = cfg_number(c1, "Min CE OI", "oi_ce_threshold", 0.0, 0.0, 1e12, step=100000.0, prefix=prefix)
        params["oi_pe_threshold"] = cfg_number(c2, "Min PE OI", "oi_pe_threshold", 0.0, 0.0, 1e12, step=100000.0, prefix=prefix)
        params["oi_flip"] = cfg_checkbox(ui, "Flip OI interpretation (buy the other leg)", "oi_flip", False, prefix=prefix)
        ui.caption("Rule: a side is dominant when its absolute OI is higher AND its change in OI is larger, with both "
                   "sides clearing their minimum thresholds. Because OI is written from the SELLER's perspective, "
                   "heavy CE writing reads as bearish by default and BUYS PE — tick the flip box for the opposite "
                   "interpretation.")

    if strategy == "OI Change Based (ΔOI)":
        params["oi_chg_mode"] = cfg_selectbox(ui, "Comparison Mode", "oi_chg_mode",
                                              ["Absolute (larger ΔOI wins)", "N× multiple (must be n times the other)"],
                                              default="Absolute (larger ΔOI wins)", prefix=prefix)
        if str(params["oi_chg_mode"]).startswith("N"):
            params["oi_chg_n"] = cfg_number(ui, "N (either ratio may satisfy it — e.g. 5 fires when ΔCE is 5× ΔPE "
                                                "OR ΔPE is 5× ΔCE)",
                                            "oi_chg_n", 2.0, 1.0, 1000.0, step=1.0, prefix=prefix)
        params["oi_chg_min"] = cfg_number(ui, "Minimum ΔOI to consider (either side)", "oi_chg_min",
                                          0.0, 0.0, 1e12, step=100000.0, prefix=prefix)
        params["oi_chg_flip"] = cfg_checkbox(ui, "Flip interpretation (buy the other leg)", "oi_chg_flip", False, prefix=prefix)
        ui.caption("Trades the CHANGE in open interest rather than its absolute level — fresh positioning, which is "
                   "usually the more informative signal. **N× mode checks BOTH ratios symmetrically**: with n = 5 a "
                   "signal fires when ΔCE/ΔPE ≥ 5 (call writing dominant → bearish → BUY PE) *or* when ΔPE/ΔCE ≥ 5 "
                   "(put writing dominant → bullish → BUY CE). Only a RISING side can dominate, since a falling ΔOI "
                   "is unwinding rather than position building; when one side rises while the other is flat or "
                   "falling the ratio is undefined, so that counts as one-sided dominance in its own right and the "
                   "status board says so. Both ratios are always displayed, so you can see exactly what was tested.")

    if strategy == "OI + Volume Change Based":
        ui.markdown("**ΔOI condition**")
        params["oiv_oi_mode"] = cfg_selectbox(ui, "ΔOI Comparison Mode", "oiv_oi_mode",
                                              ["Absolute (larger ΔOI wins)", "N× multiple (must be n times the other)"],
                                              default="Absolute (larger ΔOI wins)", prefix=prefix)
        if str(params["oiv_oi_mode"]).startswith("N"):
            params["oiv_oi_n"] = cfg_number(ui, "N for ΔOI (either ratio may satisfy it)", "oiv_oi_n", 2.0, 1.0, 1000.0, step=1.0, prefix=prefix)
        ui.markdown("**ΔVolume condition**")
        params["oiv_vol_mode"] = cfg_selectbox(ui, "ΔVolume Comparison Mode", "oiv_vol_mode",
                                               ["Absolute (larger ΔVolume wins)", "N× multiple (must be n times the other)"],
                                               default="Absolute (larger ΔVolume wins)", prefix=prefix)
        if str(params["oiv_vol_mode"]).startswith("N"):
            params["oiv_vol_n"] = cfg_number(ui, "N for ΔVolume (either ratio may satisfy it)", "oiv_vol_n", 2.0, 1.0, 1000.0, step=1.0, prefix=prefix)
        params["oiv_flip"] = cfg_checkbox(ui, "Flip interpretation (buy the other leg)", "oiv_flip", False, prefix=prefix)
        ui.caption("Requires BOTH ΔOI and ΔVolume to favour the SAME side — positions being built with real "
                   "participation behind them. If the two disagree, no entry is taken, which filters out stale OI "
                   "drift on thin volume. Each condition has its own Absolute / N× mode and its own n.")

    if strategy == "PCR Based (Put-Call Ratio)":
        c1, c2 = ui.columns(2)
        params["pcr_bull"] = cfg_number(c1, "Bullish PCR ≥", "pcr_bull", 1.2, 0.1, 10.0, step=0.05, prefix=prefix)
        params["pcr_bear"] = cfg_number(c2, "Bearish PCR ≤", "pcr_bear", 0.8, 0.05, 10.0, step=0.05, prefix=prefix)
        params["pcr_require_trend"] = cfg_checkbox(ui, "Require PCR trend confirmation (rising for long / falling for short)",
                                                   "pcr_require_trend", False, prefix=prefix)
        params["pcr_extreme_reversal"] = cfg_checkbox(ui, "Extreme-reading reversal mode", "pcr_extreme_reversal",
                                                      False, prefix=prefix)
        if params["pcr_extreme_reversal"]:
            c1, c2 = ui.columns(2)
            params["pcr_extreme_high"] = cfg_number(c1, "Extreme HIGH PCR ≥ (short)", "pcr_extreme_high",
                                                    1.8, 1.0, 20.0, step=0.1, prefix=prefix)
            params["pcr_extreme_low"] = cfg_number(c2, "Extreme LOW PCR ≤ (long)", "pcr_extreme_low",
                                                   0.5, 0.01, 2.0, step=0.05, prefix=prefix)
        params["pcr_flip"] = cfg_checkbox(ui, "Flip interpretation", "pcr_flip", False, prefix=prefix)
        ui.caption("PCR = total PE OI / total CE OI. A HIGH ratio means puts are being written heavily (writers expect "
                   "price to hold) → read as bullish; a LOW ratio is the bearish mirror. The gap between the two "
                   "thresholds is a deliberate no-trade zone, because most of the session sits mid-range and that is "
                   "noise. Trend confirmation avoids entering a stretched ratio just as it unwinds. Extreme mode "
                   "inverts the logic beyond very high/low readings, where the ratio usually signals crowded "
                   "positioning rather than trend. A full PCR tracking table appears on the Live Trading tab.")

    if strategy == "Gamma Blast (Expiry Momentum)":
        c1, c2 = ui.columns(2)
        params["gb_max_dte"] = cfg_number(c1, "Max days to expiry (0 = expiry day only)", "gb_max_dte",
                                          0, 0, 30, is_int=True, prefix=prefix)
        params["gb_premium_cap"] = cfg_number(c2, "ATM straddle premium ceiling", "gb_premium_cap",
                                              60.0, 0.5, 100000.0, step=5.0, prefix=prefix)
        c1, c2 = ui.columns(2)
        params["gb_gamma_min"] = cfg_number(c1, "Minimum ATM gamma", "gb_gamma_min",
                                            0.0, 0.0, 10.0, step=0.0005, format="%.5f", prefix=prefix)
        params["gb_range_lookback"] = cfg_number(c2, "Compression range lookback (candles)", "gb_range_lookback",
                                                 15, 3, 500, is_int=True, prefix=prefix)
        params["gb_break_buffer"] = cfg_number(ui, "Breakout buffer (points beyond the range)", "gb_break_buffer",
                                               0.0, 0.0, 10000.0, step=1.0, prefix=prefix)
        params["gb_flip"] = cfg_checkbox(ui, "Flip direction", "gb_flip", False, prefix=prefix)
        ui.caption("Looks for the expiry-day setup where ATM premium has collapsed while gamma is at its peak: "
                   "writers' hedges become hypersensitive, so a break out of the compression range can expand premium "
                   "several-fold as delta-hedging feeds on itself. ALL four must align — near expiry, straddle below "
                   "the ceiling, gamma above the floor, and price breaking the recent range. The break also sets "
                   "direction: up buys CE, down buys PE. No break means no signal, which is what keeps this "
                   "directional rather than a coin flip.")

    if strategy == "Multi-Strike OI (ATM ± N Levels)":
        params["ms_levels"] = cfg_number(ui, "Levels either side of ATM (3 = seven strikes total)", "ms_levels",
                                         3, 1, 20, is_int=True, prefix=prefix)
        c1, c2 = ui.columns(2)
        params["ms_pcr_bull"] = cfg_number(c1, "Band PCR bullish ≥", "ms_pcr_bull", 1.2, 0.1, 10.0, step=0.05, prefix=prefix)
        params["ms_pcr_bear"] = cfg_number(c2, "Band PCR bearish ≤", "ms_pcr_bear", 0.8, 0.05, 10.0, step=0.05, prefix=prefix)
        params["ms_oi_mode"] = cfg_selectbox(ui, "Band ΔOI Comparison Mode", "ms_oi_mode",
                                             ["Absolute (larger ΔOI wins)", "N× multiple (must be n times the other)"],
                                             default="Absolute (larger ΔOI wins)", prefix=prefix)
        if str(params["ms_oi_mode"]).startswith("N"):
            params["ms_oi_n"] = cfg_number(ui, "N for band ΔOI (either ratio may satisfy it)", "ms_oi_n", 2.0, 1.0, 1000.0, step=1.0, prefix=prefix)
        params["ms_use_max_pain"] = cfg_checkbox(ui, "Include max-pain vote", "ms_use_max_pain", True, prefix=prefix)
        params["ms_min_votes"] = cfg_number(ui, "Minimum net score to trade (of 3 votes)", "ms_min_votes",
                                            2, 1, 3, is_int=True, prefix=prefix)
        params["ms_flip"] = cfg_checkbox(ui, "Flip direction", "ms_flip", False, prefix=prefix)
        ui.caption("Sums CE/PE open interest, ΔOI and volume across the ATM strike and N strikes either side, computes "
                   "that band's own PCR, and locates max pain across the full chain. Restricting to strikes around the "
                   "money keeps the read where the action is instead of letting far-OTM legs dominate. Three "
                   "independent votes — band PCR, ΔOI dominance, and spot versus max pain — must reach the minimum net "
                   "score, so one ambiguous input can't trigger an entry alone.")

    if strategy in OPTION_CHAIN_STRATEGIES:
        _snap_preview = get_oi_snapshot()
        if _snap_preview:
            _pv = [f"Live chain @ {_snap_preview['fetched_at']}",
                   f"CE OI {_snap_preview['ce_oi']:,.0f} (Δ{_snap_preview['ce_oi_change']:+,.0f})",
                   f"PE OI {_snap_preview['pe_oi']:,.0f} (Δ{_snap_preview['pe_oi_change']:+,.0f})"]
            if _snap_preview.get("pcr"):
                _pv.append(f"PCR {_snap_preview['pcr']:.3f}")
            ui.caption(" · ".join(_pv))
        else:
            ui.caption("Live chain preview unavailable — check the Dhan Access Token and expiry, and note the chain "
                       "endpoint only returns data during market hours.")

    if strategy == "Hybrid (Combine Strategies)":
        _members = [s for s in STRATEGIES if s != "Hybrid (Combine Strategies)"]
        params["hybrid_members"] = cfg_multiselect(ui, "Strategies to combine", "hybrid_members",
                                                   _members, default=["EMA Crossover"], prefix=prefix)
        params["hybrid_mode"] = cfg_selectbox(ui, "Combination Logic", "hybrid_mode",
                                              ["AND — every selected strategy must fire the same direction",
                                               "OR — any one selected strategy firing is enough"],
                                              default="AND — every selected strategy must fire the same direction",
                                              prefix=prefix)
        ui.caption("AND is a confluence filter: all selected strategies must signal the SAME direction on the same bar "
                   "(fewer, higher-conviction entries). OR is a broadener: any single member firing triggers an entry. "
                   "A bar where the members contradict each other is skipped. Each member uses its own parameters as "
                   "configured above/below; flip and Trade Direction are applied once to the combined result.")
    if strategy == "Price Action Support/Resistance":
        params["sr_window"] = cfg_number(ui, "S/R Lookback", "sr_window", 20, 5, 200, is_int=True, prefix=prefix)
    if strategy == "Liquidity Grab Reversal":
        params["liq_window"] = cfg_number(ui, "Liquidity Lookback", "liq_window", 20, 5, 200, is_int=True, prefix=prefix)
    if strategy in ("Bollinger Bands", "Pro: BB+RSI Mean Reversion (ATR filtered)"):
        params["bb_period"] = cfg_number(ui, "BB Period", "bb_period", 20, 5, 100, is_int=True, prefix=prefix)
        params["bb_std"] = cfg_number(ui, "BB Std Dev", "bb_std", 2.0, 1.0, 4.0, prefix=prefix)
    if strategy == "Volume Breakout":
        params["vol_window"] = cfg_number(ui, "Volume Lookback", "vol_window", 20, 5, 100, is_int=True, prefix=prefix)
        params["vol_factor"] = cfg_number(ui, "Volume Spike Factor", "vol_factor", 2.0, 1.0, 5.0, prefix=prefix)
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

    # ------------------------------------------- INTRADAY → DELIVERY -----
    ui.markdown("### 📦 Intraday → Delivery Carry-Over")
    convert_to_delivery = cfg_checkbox(ui, "Convert unresolved intraday positions to delivery",
                                       "convert_to_delivery", True, prefix=prefix)
    delivery_cutoff_time = dtime(15, 0)
    if convert_to_delivery:
        delivery_cutoff_time = cfg_time(ui, "Conversion cut-off (IST)", "delivery_cutoff_time",
                                        dtime(15, 0), prefix=prefix)
        ui.caption("If neither SL nor target has been hit by this time, the position is NOT squared off — it is "
                   "marked as delivery, kept open, and written to the database so you can review it and resume "
                   "trading it later from the Admin Panel. Applies to stock intraday, stock options and index "
                   "options. Enable Data Persistence (Admin Panel) for it to survive a restart.")
    else:
        ui.caption("Positions are left to the normal exit rules; note that brokers auto-square-off intraday "
                   "products near the close.")

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
    chain_strategy = strategy in OPTION_CHAIN_STRATEGIES
    need_creds = use_dhan_feed or dhan_enabled or options_mode or chain_strategy
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

    dhan_touchpoints_on = dhan_enabled or options_mode or chain_strategy
    if chain_strategy and not options_mode and not premium_mode:
        # ---- OPTION-CHAIN STRATEGY EXECUTION ----------------------------
        # These strategies read the chain but their signals must be TRADED as
        # option legs: a LONG buys CE, a SHORT buys PE. Without this the trade
        # would be placed on the index itself (and paper P&L tracked on index
        # points), which is not what the strategy is expressing. The ATM CE and
        # PE legs of the configured chain are resolved here so both live orders
        # and paper trading use the correct instrument.
        ui.markdown("#### 🎯 Option Leg Execution")
        _cs_trade_opts = cfg_checkbox(ui, "Trade CE/PE legs (LONG → buy CE, SHORT → buy PE)",
                                      "chain_trade_options", True, prefix=prefix)
        if _cs_trade_opts:
            _cs_und = params.get("oi_underlying", store.get("oi_underlying", "Nifty50"))
            _cs_meta = DHAN_INDEX_MAP.get(_cs_und, DHAN_INDEX_MAP["Nifty50"])
            _cs_exp = params.get("oi_expiry", store.get("oi_expiry"))
            _cs_strikes = dhan_get_strikes(_cs_meta["underlying"], _cs_exp, "OPTIDX",
                                           _cs_meta["exchange"]) if _cs_exp else []
            _cs_offset = cfg_number(ui, "Strike offset from ATM (0 = ATM, +1 = one strike OTM, −1 = ITM)",
                                    "chain_strike_offset", 0, -20, 20, is_int=True, prefix=prefix)
            _cs_atm = None
            if _cs_strikes:
                _cs_ltp = _current_underlying_ltp(TICKER_MAP.get(_cs_und, "^NSEI"))
                _cs_atm = round_to_nearest_strike(_cs_ltp, _cs_strikes)
            if _cs_atm is not None and _cs_strikes:
                _ai = _cs_strikes.index(_cs_atm)
                # An OTM call sits ABOVE spot and an OTM put BELOW, so the
                # offset is applied in opposite directions for the two legs.
                _ce_strike = _cs_strikes[min(len(_cs_strikes) - 1, max(0, _ai + int(_cs_offset)))]
                _pe_strike = _cs_strikes[min(len(_cs_strikes) - 1, max(0, _ai - int(_cs_offset)))]
                _sig = ("CHAIN", _cs_und, _cs_exp, _ce_strike, _pe_strike)

                def _fetch_chain_legs():
                    ce = dhan_lookup_option(_cs_meta["underlying"], _cs_exp, _ce_strike, "CE",
                                            "OPTIDX", _cs_meta["exchange"])
                    pe = dhan_lookup_option(_cs_meta["underlying"], _cs_exp, _pe_strike, "PE",
                                            "OPTIDX", _cs_meta["exchange"])
                    if ce:
                        cfg_force("ce_security_id", ce["security_id"])
                        store["_opt_lot_size"] = ce.get("lot_size")
                    if pe:
                        cfg_force("pe_security_id", pe["security_id"])
                    return bool(ce and pe)

                if st.session_state.get("dhan_opt_autofill_sig") != _sig \
                        and st.session_state.get("_attempted_dhan_opt_autofill_sig") != _sig:
                    cfg_force("ce_security_id", "")
                    cfg_force("pe_security_id", "")
                _try_autofill(_sig, _fetch_chain_legs, "dhan_opt_autofill_sig", "dhan_opt_autofill_last_try")
                ui.caption(f"ATM {_cs_atm:.0f} · CE strike {_ce_strike:.0f} · PE strike {_pe_strike:.0f} "
                           f"(expiry {_cs_exp})")
            else:
                ui.warning("Could not resolve strikes for this chain — check the Dhan token and expiry. "
                           "Until the CE/PE IDs are filled, entries would fall back to the underlying.")
            _ce_id = cfg_text(ui, "CE Security ID (auto-filled, used on LONG signals)",
                              "ce_security_id", "", prefix=prefix).strip()
            _pe_id = cfg_text(ui, "PE Security ID (auto-filled, used on SHORT signals)",
                              "pe_security_id", "", prefix=prefix).strip()
            _cs_qty_default = _cs_meta.get("default_opt_qty", 65)
            if store.get("_chain_qty_sig") != _cs_und:
                cfg_force("dhan_qty", int(_cs_qty_default))
                store["_chain_qty_sig"] = _cs_und
            c1, c2 = ui.columns(2)
            entry_order_type = cfg_selectbox(c1, "Entry Order Type", "entry_order_type",
                                             ["MARKET", "LIMIT"], default="MARKET", prefix=prefix)
            exit_order_type = cfg_selectbox(c2, "Exit Order Type", "exit_order_type",
                                            ["MARKET", "LIMIT"], default="MARKET", prefix=prefix)
            dhan_qty = cfg_number(ui, "Dhan Quantity (lots × lot size)", "dhan_qty",
                                  int(_cs_qty_default), 1, 1000000, is_int=True, prefix=prefix)
            product_cfg = {
                "instrument": "Index Options",
                "exchange": _cs_meta["exchange"],
                "exchange_segment": f"{_cs_meta['exchange']}_FNO",
                "product": "MARGIN",
                "options_mode": True,
                "chain_strategy_mode": True,
                "ce_security_id": _ce_id,
                "pe_security_id": _pe_id,
                "expiry": _cs_exp,
                "underlying": _cs_meta["underlying"],
                "lot_size": store.get("_opt_lot_size"),
                "bo_enabled": False,
            }
            ui.caption("Signals are computed from the option chain but EXECUTED on the option legs — a LONG buys the "
                       "CE, a SHORT buys the PE, and exits sell whichever leg is open. Paper trading records the "
                       "leg's premium too, so P&L reflects the option rather than index points. Untick the box above "
                       "to trade the underlying instead.")
        else:
            ui.caption("CE/PE execution disabled — signals will be traded on the underlying index instead.")
    elif dhan_touchpoints_on and premium_mode:
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
            cfg_force("exchange", auto_exchange)       # auto-flip writes the widget…
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
                    cfg_force("ce_security_id", ce["security_id"]); ok_any = True
                    store["_opt_lot_size"] = ce.get("lot_size")
                if pe:
                    cfg_force("pe_security_id", pe["security_id"]); ok_any = True
                return bool(ce and pe) if (expiry and ce_strike and pe_strike) else ok_any

            if creds_ok or True:  # scrip master is public — autofill works even in paper mode
                if st.session_state.get("dhan_opt_autofill_sig") != sig:
                    # a signature change ALWAYS overwrites stale IDs
                    if st.session_state.get("_attempted_dhan_opt_autofill_sig") != sig:
                        cfg_force("ce_security_id", "")
                        cfg_force("pe_security_id", "")
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
                cfg_force("dhan_qty", int(default_qty))
                store["_opt_qty_default_sig"] = (underlying, instrument_type)

            ui.caption("Options direction rule (all modes, including flipped signals): LONG signal → BUY the CE leg; "
                       "SHORT signal → BUY the PE leg; exits SELL whichever leg is open. Options are always BOUGHT, "
                       "never sold.")

        elif is_fno:  # ------- futures: single security id, auto-filled
            sig = ("FUT", ticker, instrument_type, exchange, expiry)

            def _fetch_fut_id():
                info = dhan_lookup_future(underlying, expiry, meta["scrip_instrument"], exchange) if expiry else None
                if info:
                    cfg_force("dhan_security_id", info["security_id"])
                    if store.get("_fut_qty_default_sig") != sig:
                        cfg_force("dhan_qty", int(info.get("lot_size") or 1))  # futures default = contract lot size
                        store["_fut_qty_default_sig"] = sig
                    return True
                return False

            if st.session_state.get("dhan_autofill_sig") != sig:
                if st.session_state.get("_attempted_dhan_autofill_sig") != sig:
                    cfg_force("dhan_security_id", "")   # sig change ALWAYS clears stale IDs
            _try_autofill(sig, _fetch_fut_id, "dhan_autofill_sig", "dhan_autofill_last_try")

            sec_id = cfg_text(ui, "Security ID (always visible & mandatory — auto-filled when Dhan is enabled, "
                                  "manual entry in pure-yfinance mode)", "dhan_security_id", "", prefix=prefix)
            product_cfg["security_id"] = sec_id.strip()

        else:  # ----------------- equity: security id auto-filled
            sig = ("EQ", ticker, instrument_type, exchange)

            def _fetch_eq_id():
                info = dhan_lookup_equity(_yf_symbol_to_plain(ticker), exchange)
                if info:
                    cfg_force("dhan_security_id", info["security_id"])
                    return True
                return False

            if st.session_state.get("dhan_autofill_sig") != sig:
                if st.session_state.get("_attempted_dhan_autofill_sig") != sig:
                    cfg_force("dhan_security_id", "")   # sig change ALWAYS clears stale IDs
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
    email_to, email_app_password = EMAIL_DEFAULT_TO, ""
    if email_enabled:
        email_from = cfg_text(ui, "From (Gmail address)", "email_from", EMAIL_DEFAULT_FROM, prefix=prefix)
        email_to = cfg_text(ui, "To (comma-separated)", "email_to", EMAIL_DEFAULT_TO, prefix=prefix)
        email_app_password = cfg_text(ui, "Gmail App Password", "email_app_password", "", type="password", prefix=prefix)
        ui.caption("Emails via Gmail SMTP (SSL 465) on entry, exit, partial book, and manual square-off — containing "
                   "strategy/entry/SL/target/exit reason/points/PnL. A mail failure never blocks trading. "
                   "The password must be a 16-character **App Password** from myaccount.google.com/apppasswords "
                   "(2-Step Verification required) — a normal Gmail password is always rejected by SMTP. "
                   "Spaces are stripped automatically, so you can paste it exactly as Google shows it.")
        if ui.button("📧 Send test email", key=f"btn_{prefix}_email_test"):
            _ok = send_trade_email(
                {"email_enabled": True, "email_from": email_from, "email_to": email_to,
                 "email_app_password": email_app_password},
                "[AlgoTrader] Test email",
                ["This is a test from AlgoTrader Pro.",
                 f"Sent at {ist_now().strftime('%d-%b-%Y %H:%M:%S IST')}.",
                 "If you received this, trade notifications will work."], force=True)
            _st = st.session_state.get("email_last_status", {})
            if _ok:
                ui.success("✅ " + _st.get("message", "Sent."))
            else:
                ui.error("❌ " + _st.get("message", "Failed."))
    else:
        email_to = str(store.get("email_to", EMAIL_DEFAULT_TO) or EMAIL_DEFAULT_TO)
        email_app_password = str(store.get("email_app_password", "") or "")

    return dict(
        ticker=ticker, ticker_choice=ticker_choice, interval=interval, period=period, qty=qty,
        strategy=strategy, sl_type=sl_type, target_type=target_type, params=params, filters=filters,
        wf_enabled=wf_enabled, wf_folds=wf_folds, cost_enabled=cost_enabled, cost_cfg=cost_cfg,
        risk_ctrl=risk_ctrl, gates=gates,
        convert_to_delivery=convert_to_delivery,
        delivery_cutoff_time=delivery_cutoff_time,
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

# Restore any trade that was still open when a previous session ended, before
# any tab renders, so the live engine picks it up and keeps managing it.
db_bootstrap()

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

def config_fingerprint(cfg):
    """Stable hash of everything that can change backtest output. Used to tell
    the user when displayed results predate their current settings."""
    try:
        relevant = {
            k: cfg.get(k) for k in
            ("ticker", "interval", "period", "qty", "strategy", "sl_type", "target_type",
             "cost_enabled", "wf_enabled", "wf_folds")
        }
        relevant["params"] = cfg.get("params")
        relevant["filters"] = cfg.get("filters")
        relevant["risk_ctrl"] = cfg.get("risk_ctrl")
        relevant["cost_cfg"] = cfg.get("cost_cfg")
        return json.dumps(relevant, sort_keys=True, default=str)
    except Exception:
        return None


def price_chart(df, trades_df=None, title="", ema_overlay=None, extra_lines=None, elliott=None):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
    if elliott:
        # Zigzag connecting confirmed alternating pivots, annotated with the
        # running Elliott count (1-2-3-4-5 → A-B-C).
        try:
            seq = elliott.get("pivots") or []
            if len(seq) >= 2:
                xs = [df.index[i] for i, _p, _k in seq]
                ys = [p for _i, p, _k in seq]
                labels = ["1", "2", "3", "4", "5", "A", "B", "C"]
                texts = [labels[n % len(labels)] for n in range(len(seq))]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers+text", text=texts, textposition="top center",
                    textfont=dict(size=12, color="#8ab4f8"),
                    line=dict(color="#8ab4f8", width=1.8, dash="dot"),
                    marker=dict(size=8, color="#8ab4f8", symbol="diamond"),
                    name="Elliott zigzag"))
            prov = elliott.get("provisional")
            if prov:
                kind, price, when = prov
                fig.add_trace(go.Scatter(
                    x=[when], y=[price], mode="markers+text",
                    text=[f"provisional {'L' if kind == 'L' else 'H'}"], textposition="bottom center",
                    textfont=dict(size=11, color="#f0a202"),
                    marker=dict(size=11, color="#f0a202", symbol="circle-open", line=dict(width=2)),
                    name="Pending pivot"))
        except Exception:
            pass
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
            if strategy == "RSI Cross":
                bl = params.get("rsi_buy_level", 30.0)
                sl_lvl = params.get("rsi_sell_level", 70.0)
                bd = params.get("rsi_buy_cross", "Up-cross (from below)")
                sd = params.get("rsi_sell_cross", "Down-cross (from above)")
                lines.append(
                    f"RSI({params.get('rsi_period',14)}) = {r_val:.1f}. "
                    f"BUY needs RSI to cross {'UP through' if bd.startswith('Up') else 'DOWN through'} {bl:.0f} "
                    f"(distance {r_val - bl:+.1f}). "
                    f"SELL needs RSI to cross {'DOWN through' if sd.startswith('Down') else 'UP through'} {sl_lvl:.0f} "
                    f"(distance {sl_lvl - r_val:+.1f}).")
            else:
                lines.append(f"RSI filter({params.get('rsi_period',14)}) = {r_val:.1f}. Buy needs an up-cross through 30 "
                             f"(distance {r_val-30:+.1f}); sell a down-cross through 70 (distance {70-r_val:+.1f}).")
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
    if strategy == "Elliott Wave (Zigzag)":
        lb = params.get("zigzag_lookback", 3)
        try:
            piv = elliott_wave_state(df, lb)
            seq = piv["pivots"]
            if seq:
                recent = seq[-4:]
                chain = " → ".join(f"{'H' if k=='H' else 'L'}@{p:.2f}" for _i, p, k in recent)
                last_idx, last_price, last_kind = seq[-1]
                lab = piv["wave_label"].iloc[min(last_idx + lb, len(df) - 1)]
                lines.append(f"Elliott structure (last {len(recent)} confirmed pivots): {chain}"
                             + (f" · current wave count ≈ {lab}" if lab else ""))
                if last_kind == "L":
                    lines.append(f"Last CONFIRMED pivot was a LOW @ {last_price:.2f} → a LONG already fired on its "
                                 f"confirmation bar. Next SHORT needs a swing HIGH to form and then survive {lb} bars.")
                else:
                    lines.append(f"Last CONFIRMED pivot was a HIGH @ {last_price:.2f} → a SHORT already fired on its "
                                 f"confirmation bar. Next LONG needs a swing LOW to form and then survive {lb} bars.")
            else:
                lines.append(f"Elliott Wave: no confirmed pivots yet — need at least {2*lb+2} candles of history "
                             f"(have {len(df)}).")
            prov, btc = piv["provisional"], piv["bars_to_confirm"]
            if prov and btc:
                kind, price, when = prov
                side = "LONG" if kind == "L" else "SHORT"
                lines.append(f"⏳ PENDING pivot: provisional swing {'LOW' if kind=='L' else 'HIGH'} @ {price:.2f} "
                             f"(formed {when}). It confirms in {btc} more candle(s) — a {side} fires then, "
                             f"provided price does not {'break below' if kind=='L' else 'break above'} {price:.2f} "
                             "first (which would invalidate it and start a new pivot).")
            else:
                lines.append(f"⏳ No provisional pivot in the last {lb} candles — price is mid-swing; a new "
                             "extreme must print before any signal can develop.")
            if params.get("ew_impulse_only"):
                lines.append("Impulse-only mode ON: a confirmed low must also be a HIGHER low (and a confirmed high a "
                             "LOWER high) or the signal is suppressed.")
        except Exception as exc:
            lines.append(f"Elliott Wave: could not compute wave state ({exc}).")

    if strategy in OPTION_CHAIN_STRATEGIES:
        oc_sig, oc_lines = evaluate_option_chain_signal(strategy, params, df)
        for l in oc_lines:
            lines.append("📊 " + l)
        lines.append(f"Option-chain verdict right now: "
                     f"{'🟢 LONG (BUY CE)' if oc_sig == 1 else ('🔴 SHORT (BUY PE)' if oc_sig == -1 else '⚪ no signal')} "
                     "— entry fires immediately at LTP when this turns non-flat.")

    if strategy == "Hybrid (Combine Strategies)":
        members = list(params.get("hybrid_members", []))
        mode = params.get("hybrid_mode", "AND")
        if not members:
            lines.append("Hybrid: no member strategies selected — nothing can fire. Pick at least one.")
        else:
            lines.append(f"Hybrid mode: {'AND (all must agree)' if str(mode).upper().startswith('AND') else 'OR (any one is enough)'} "
                         f"across {len(members)} strategies:")
            fired_long, fired_short = [], []
            for m in members:
                try:
                    sub = generate_signals(df, m, params, _raw=True)
                    s_now = int(sub["signal"].iloc[-1])
                except Exception:
                    s_now = 0
                mark = "🟢 LONG" if s_now == 1 else ("🔴 SHORT" if s_now == -1 else "⚪ flat")
                lines.append(f"   • {m}: {mark} on the latest bar")
                if s_now == 1:
                    fired_long.append(m)
                elif s_now == -1:
                    fired_short.append(m)
            if str(mode).upper().startswith("AND"):
                need = len(members)
                lines.append(f"   → AND needs all {need}: currently {len(fired_long)} long / {len(fired_short)} short. "
                             + ("✅ LONG would fire." if len(fired_long) == need else
                                ("✅ SHORT would fire." if len(fired_short) == need else "❌ not unanimous — no entry.")))
            else:
                lines.append("   → OR needs any one: "
                             + ("✅ LONG would fire." if fired_long else
                                ("✅ SHORT would fire." if fired_short else "❌ none firing — no entry.")))

    if strategy == "Threshold Cross":
        thr = params.get("threshold", c_now)
        cd = params.get("threshold_direction", "Below")
        tm = params.get("threshold_trigger_mode", "Level")
        level_mode = not str(tm).startswith("Cross")
        if cd == "Above":
            armed = c_now < thr
            lines.append(f"Threshold Cross (Above · {'Level' if level_mode else 'Cross-event'} mode): price {c_now:.2f} "
                         f"vs threshold {thr:.2f} → "
                         + (f"{'✅ SHORT condition met NOW' if armed else f'❌ needs {c_now - thr:+.2f} more to fall below'}"
                            if level_mode else
                            f"needs an actual DOWN-cross through it (currently {c_now - thr:+.2f} away)"))
        else:
            armed = c_now > thr
            lines.append(f"Threshold Cross (Below · {'Level' if level_mode else 'Cross-event'} mode): price {c_now:.2f} "
                         f"vs threshold {thr:.2f} → "
                         + (f"{'✅ LONG condition met NOW' if armed else f'❌ needs {thr - c_now:+.2f} more to rise above'}"
                            if level_mode else
                            f"needs an actual UP-cross through it (currently {thr - c_now:+.2f} away)"))
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
        elif strategy in OPTION_CHAIN_STRATEGIES:
            # Read the LIVE option-chain snapshot directly rather than any
            # candle column: OI/volume/PCR/gamma are not candle-derived, so
            # there is no bar to wait for and the entry happens at once, at LTP.
            last_sig, _oc_lines = evaluate_option_chain_signal(strategy, params, sig_df)
            if last_sig != 0:
                st.caption(f"📊 {strategy}: condition met → entering immediately at LTP (no candle close required). "
                           + (_oc_lines[-1] if _oc_lines else ""))
        else:  # Threshold Cross — evaluated against the LIVE LTP, no candle close needed
            thr = params.get("threshold", prev_close)
            cross_dir = params.get("threshold_direction", "Below")
            mode = params.get("threshold_trigger_mode", "Level")
            if str(mode).startswith("Cross"):
                if cross_dir == "Above":
                    last_sig = -1 if (ltp < thr and prev_close >= thr) else 0
                else:
                    last_sig = 1 if (ltp > thr and prev_close <= thr) else 0
            else:
                # Level mode: fires the instant the LTP is on the trigger side,
                # regardless of when the crossing happened.
                if cross_dir == "Above":
                    last_sig = -1 if ltp < thr else 0
                else:
                    last_sig = 1 if ltp > thr else 0
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
                    db_save_trade(row, ticker, strategy)
                    db_persist_position_state(ticker, strategy)
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
            _conv, _conv_msg = check_delivery_conversion(full_cfg, pos, ticker, strategy)
            if _conv:
                st.session_state["delivery_note"] = _conv_msg
                st.info("📦 " + _conv_msg)
                db_persist_position_state(ticker, strategy)
                if full_cfg.get("email_enabled"):
                    email_trade_event(full_cfg, "Converted to Delivery", {
                        "Ticker": ticker, "Strategy": strategy,
                        "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
                        "Entry Price": round(pos["entry_price"], 2),
                        "SL": round(pos["sl"], 2), "Target": round(pos["target"], 2),
                        "Qty": pos.get("remaining_qty"), "Note": _conv_msg,
                    })
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
        if not exited:
            # Keep the stored copy in step with trailed SL/target levels so a
            # restored position resumes with the correct risk, not stale levels.
            db_persist_position_state(ticker, strategy)
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
            db_save_trade(row, ticker, strategy)
            db_clear_open_position()
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
            db_persist_position_state(ticker, strategy)   # survives a session drop
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

(tab_bt, tab_live, tab_hist, tab_heat, tab_opt, tab_spread, tab_ohlc,
 tab_chain, tab_screen, tab_dl, tab_admin) = st.tabs(
    ["📊 Backtest", "🔴 Live Trading", "📜 Trade History", "🔥 Heatmaps", "🧪 Optimization",
     "🔀 Spread Tool", "📅 OHLC & Range", "🔗 Option Chain Analysis", "🔎 Screener",
     "⬇️ Data Download", "🛠 Admin Panel"]
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
                st.session_state.last_backtest_fp = config_fingerprint(config)

    trades_df = st.session_state.last_backtest
    sig_df = st.session_state.last_backtest_df

    # Results are computed on click and then persist. If the configuration has
    # changed since that click, say so loudly instead of letting old numbers
    # masquerade as results for the current settings.
    _cur_fp = config_fingerprint(config)
    if trades_df is not None and st.session_state.get("last_backtest_fp") not in (None, _cur_fp):
        st.warning("⚠️ The configuration has changed since these results were produced — they reflect the PREVIOUS "
                   "settings. Click **Run Backtest** to recompute with the current configuration.")

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
                        ema_overlay=[(params.get("ema_fast", 9), "#3399ff"), (params.get("ema_slow", 15), "#ff9933")],
                        elliott=(elliott_wave_state(sig_df, params.get("zigzag_lookback", 3))
                                 if strategy == "Elliott Wave (Zigzag)" else None)),
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
    if st.session_state.get("db_restored_note"):
        st.success("💾 " + st.session_state.pop("db_restored_note"))
    if st.session_state.get("live_blocked_reason"):
        st.warning(f"🚧 Last entry was blocked by a risk gate: {st.session_state.live_blocked_reason}")

    # ---- Run Once / Run Continuously / Stop / Square-off controls ----
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        manual_eval = st.button("🔍 Run Once", use_container_width=True,
                                help="Evaluate signals and manage any open position exactly once, right now.")
    with ctrl2:
        if not st.session_state.live_running:
            if st.button("▶ Run Continuously", type="primary", use_container_width=True,
                         help="Keep re-checking signals every few seconds until stopped."):
                st.session_state.live_running = True
                st.rerun()
        else:
            st.button("▶ Running…", disabled=True, use_container_width=True)
    with ctrl3:
        if st.button("⏹ Stop", use_container_width=True, disabled=not st.session_state.live_running):
            st.session_state.live_running = False
            st.rerun()
    with ctrl4:
        squareoff_clicked = st.button("🟥 Square Off Now", use_container_width=True, disabled=not st.session_state.live_positions)

    if st.session_state.live_running:
        st.success("🟢 Running continuously — polling the API and re-checking signals every few seconds. "
                   "Click Stop to halt it.")
    else:
        st.caption("⚪ Continuous running is OFF — no background API calls are being made. Use **Run Once** for a "
                   "single check, or **Run Continuously** to keep it evaluating.")

    _em = st.session_state.get("email_last_status")
    if _em:
        if _em.get("ok"):
            st.caption(f"📧 Last email: ✅ {_em['message']} ({_em['at']})")
        else:
            st.warning(f"📧 Last email FAILED at {_em['at']}: {_em['message']}")
    if st.session_state.get("delivery_note"):
        st.info("📦 " + st.session_state.pop("delivery_note"))
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
        db_save_trade(_sq_row, ticker, strategy)
        db_clear_open_position()
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

    # ---- Option-chain panel (all chain strategies) + PCR tracking table ----
    if strategy in OPTION_CHAIN_STRATEGIES:
        st.markdown("#### 📊 Live Option Chain")
        _snap = get_oi_snapshot()
        record_chain_history(_snap)
        if not _snap:
            st.warning("Option chain unavailable — verify the Dhan Access Token and the selected expiry. "
                       "Dhan's chain endpoint only returns data during market hours.")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Underlying", f"{_snap.get('underlying') or 0:,.2f}")
            m2.metric("CE OI", f"{_snap['ce_oi']:,.0f}", f"{_snap['ce_oi_change']:+,.0f}")
            m3.metric("PE OI", f"{_snap['pe_oi']:,.0f}", f"{_snap['pe_oi_change']:+,.0f}")
            m4.metric("PCR (OI)", f"{_snap['pcr']:.3f}" if _snap.get("pcr") else "n/a")
            _mp, _ = compute_max_pain(_snap.get("strikes") or {})
            m5.metric("Max Pain", f"{_mp:.0f}" if _mp else "n/a")
            v1, v2, v3 = st.columns(3)
            v1.metric("CE Volume", f"{_snap.get('ce_volume', 0):,.0f}", f"{_snap.get('ce_volume_change', 0):+,.0f}")
            v2.metric("PE Volume", f"{_snap.get('pe_volume', 0):,.0f}", f"{_snap.get('pe_volume_change', 0):+,.0f}")
            v3.metric("PCR (Volume)", f"{_snap['pcr_volume']:.3f}" if _snap.get("pcr_volume") else "n/a")
            st.caption(f"Snapshot @ {_snap['fetched_at']} · expiry {_snap.get('expiry')} · refreshes every 60s "
                       "(Dhan rate-limits the chain endpoint).")

            if strategy == "Multi-Strike OI (ATM ± N Levels)":
                _band = multi_strike_band(_snap, int(params.get("ms_levels", 3)))
                if _band:
                    st.markdown(f"**ATM {_band['atm']:.0f} ± {_band['levels']} levels "
                                f"({len(_band['band'])} strikes)**")
                    _rows = []
                    for s in _band["band"]:
                        r = _snap["strikes"][s]
                        _rows.append({
                            "Strike": s, "CE OI": r["ce_oi"], "ΔCE OI": r["ce_oi_change"],
                            "CE Vol": r["ce_vol"], "PE OI": r["pe_oi"], "ΔPE OI": r["pe_oi_change"],
                            "PE Vol": r["pe_vol"],
                            "Strike PCR": round(r["pe_oi"] / r["ce_oi"], 3) if r["ce_oi"] else None,
                        })
                    st.dataframe(pd.DataFrame(_rows), hide_index=True, use_container_width=True)

        if strategy == "PCR Based (Put-Call Ratio)":
            st.markdown("#### 📈 PCR Tracking Table")
            st.caption("One row per chain snapshot (newest first). Each metric shows its change from the PREVIOUS "
                       "reading three ways: absolute, percentage, and n× multiple.")
            _tbl = build_chain_history_table()
            if _tbl.empty:
                st.info("No snapshots recorded yet — the table fills in as the chain refreshes (every 60s).")
            else:
                st.dataframe(_tbl, hide_index=True, use_container_width=True)
                st.download_button("⬇ Download PCR history (CSV)", _tbl.to_csv(index=False).encode(),
                                   file_name="pcr_history.csv", mime="text/csv")

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
        st.caption("📊 Indicator Dashboard / 📟 Signal Status Board are paused while monitoring is OFF. Click Run Continuously to see them update live, or Run Once for a single snapshot below.")
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
                        extra_lines=extra_lines,
                        elliott=(elliott_wave_state(chart_df, params.get("zigzag_lookback", 3))
                                 if strategy == "Elliott Wave (Zigzag)" else None)),
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
    opt_intervals = st.multiselect("Timeframes to test", list(available_tf_period_map().keys()), default=[interval])
    combo_periods = sorted({p for iv in opt_intervals for p in available_tf_period_map().get(iv, [])}, key=lambda x: x)
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
                * sum(1 for iv in opt_intervals for p in available_tf_period_map().get(iv, []) if p in opt_periods))
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
            for iv in opt_intervals for p in available_tf_period_map().get(iv, []) if p in opt_periods
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

# -------------------------------------------------- OPTION CHAIN ANALYSIS ---
with tab_chain:
    st.subheader("🔗 Option Chain Analysis")
    st.caption("A full positioning picture built from live Dhan option-chain snapshots. Every plot draws on the "
               "snapshot history below it, so the charts fill out as analysis runs. Independent of the sidebar "
               "strategy — pick any underlying and expiry here.")

    # ---------------- source + refresh controls (outside the fragment) -----
    oc0, oc1, oc2, oc3 = st.columns([1, 1.2, 1.2, 1])
    oca_kind = cfg_selectbox(oc0, "Underlying type", "oca_kind", ["Index", "Stock"], default="Index")
    if oca_kind == "Index":
        oca_name = cfg_selectbox(oc1, "Index", "oca_underlying", list(DHAN_INDEX_MAP.keys()), default="Nifty50")
    else:
        oca_name = cfg_text(oc1, "Stock symbol (NSE, e.g. RELIANCE)", "oca_stock", "RELIANCE")
    _uinfo = resolve_chain_underlying(oca_kind, oca_name)
    oca_under = _uinfo["label"] if _uinfo else str(oca_name)
    if not _uinfo:
        st.warning(f"Could not resolve '{oca_name}' in Dhan's scrip master — check the symbol. "
                   "Stock option chains require the stock to have listed options (F&O stocks only).")
    _oca_exps = dhan_get_expiries(_uinfo["underlying"], _uinfo["opt_instrument"],
                                  _uinfo["exchange"]) if _uinfo else []
    if _oca_exps:
        oca_expiry = cfg_selectbox(oc2, "Expiry (nearest pre-selected)", "oca_expiry",
                                   _oca_exps, default=_oca_exps[0])
    else:
        oca_expiry = cfg_text(oc2, "Expiry (YYYY-MM-DD)", "oca_expiry", "")
    oca_interval = cfg_number(oc3, "Continuous refresh (seconds)", "oca_interval",
                              60, 15, 900, step=15, is_int=True)

    st.caption("Dhan caches the chain for 60s and rate-limits that endpoint, so refresh intervals below ~60s will "
               "mostly re-display the same snapshot rather than fetching a new one.")

    b1, b2, b3, b4 = st.columns(4)
    _once = b1.button("🔍 Analyze Once", use_container_width=True)
    _start = b2.button("▶ Analyze Continuously", type="primary", use_container_width=True)
    _stop = b3.button("⏹ Stop Analysis", use_container_width=True)
    _clear = b4.button("🗑 Clear History", use_container_width=True)

    if _once:
        _s = get_chain_snapshot_for(_uinfo, oca_expiry)
        record_chain_history(_s, get_futures_price(_uinfo), oca_under)
        st.session_state.oca_last_run = ist_now().strftime("%H:%M:%S IST")
        if not _s:
            st.warning("No snapshot returned — check the Dhan Access Token and expiry; the chain endpoint only "
                       "serves data during market hours.")
    if _start:
        st.session_state.oca_running = True
    if _stop:
        st.session_state.oca_running = False
    if _clear:
        st.session_state.chain_history = []
        st.session_state.pop("groq_last", None)

    _running = bool(st.session_state.get("oca_running", False))
    if _running:
        st.success(f"▶ Continuous analysis ON — every plot and summary refreshes automatically every "
                   f"{int(oca_interval)}s. Press ⏹ Stop Analysis to halt.")
    else:
        st.info("⏹ Continuous analysis is OFF. Use 🔍 Analyze Once for a single snapshot, or ▶ Analyze "
                "Continuously to keep the charts updating.")

    # ---------------- custom plot metric selection (checkboxes) ------------
    with st.expander("🎛 Custom plot — tick any metrics to chart together", expanded=False):
        st.caption("Any combination works. Metrics that share a scale share an axis; if you mix three or more "
                   "different scales the series are indexed to 100 so their shapes stay comparable (raw values "
                   "stay in the tooltip).")
        _labels = list(CHAIN_METRICS.keys())
        _picked = []
        _ccols = st.columns(3)
        for _i, _lbl in enumerate(_labels):
            _default = _lbl in ("Price (index/underlying)", "PCR (OI)")
            if cfg_checkbox(_ccols[_i % 3], _lbl, f"oca_m_{_i}", _default):
                _picked.append(_lbl)
        _norm_choice = cfg_selectbox(st, "Scaling", "oca_norm",
                                     ["Auto (index to 100 when scales differ)",
                                      "Always index to 100", "Never — use real values on twin axes"],
                                     default="Auto (index to 100 when scales differ)")
        st.session_state["oca_picked"] = _picked

    # ---------------- Groq controls ----------------------------------------
    with st.expander("🤖 Groq AI analysis of the option chain", expanded=False):
        g1, g2 = st.columns([2, 1])
        groq_key = cfg_text(g1, "Groq API Key", "groq_api_key", "", type="password")
        groq_model_pick = cfg_selectbox(g2, "Model", "groq_model_pick", GROQ_MODEL_CHOICES,
                                        default=GROQ_MODEL_CHOICES[0])
        groq_model = (cfg_text(st, "Custom model id", "groq_model_custom", "")
                      if groq_model_pick.startswith("(custom") else groq_model_pick)
        groq_extra = cfg_text(st, "Extra instructions (optional)", "groq_extra", "")
        groq_auto = cfg_checkbox(st, "Include Groq in continuous analysis (uses API quota every refresh)",
                                 "groq_auto", False)
        if st.button("🤖 Ask Groq now"):
            with st.spinner("Sending the chain to Groq…"):
                _snap_g = get_chain_snapshot_for(_uinfo, oca_expiry)
                record_chain_history(_snap_g, get_futures_price(_uinfo), oca_under)
                st.session_state["groq_last"] = groq_analyze_chain(
                    groq_key, groq_model, _snap_g, chain_history_df(), groq_extra)
                st.session_state["groq_last_at"] = ist_now().strftime("%H:%M:%S IST")
        st.caption("The model receives the headline aggregates, the ATM strike band, max pain, greeks and the recent "
                   "snapshot history. Model output is an opinion generated from that data — it is not financial "
                   "advice, and it can be wrong or inconsistent. Treat it as one more input, never as a trade "
                   "instruction. Your key is used only for these requests and is not stored anywhere by this app.")

    st.divider()

    # ---------------- everything below refreshes on the interval -----------
    @st.fragment(run_every=(int(oca_interval) if _running else None))
    def _chain_analysis_body():
        snap = get_chain_snapshot_for(_uinfo, oca_expiry)
        futures_px = get_futures_price(_uinfo)
        record_chain_history(snap, futures_px, oca_under)
        hist = chain_history_df()

        _ready, _problems = chain_readiness(_uinfo, oca_expiry)
        if not _ready:
            with st.container(border=True):
                st.markdown("### ⚠️ Why some values are empty")
                for _p in _problems:
                    st.markdown(f"- {_p}")
        if not snap:
            if hist.empty:
                st.info("Nothing to display yet — resolve the points above and run the analysis.")
                return

        # ---- headline metrics ----
        if snap:
            mp, _pain = compute_max_pain(snap.get("strikes") or {})
            atm, arow = _atm_row(snap)
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Spot", f"{snap.get('underlying') or 0:,.2f}")
            k2.metric("ATM", f"{atm:.0f}" if atm else "n/a")
            k3.metric("PCR (OI)", f"{snap['pcr']:.3f}" if snap.get("pcr") else "n/a")
            k4.metric("PCR (Vol)", f"{snap['pcr_volume']:.3f}" if snap.get("pcr_volume") else "n/a")
            k5.metric("Max Pain", f"{mp:.0f}" if mp else "n/a")
            k6.metric("Futures", f"{futures_px:,.2f}" if futures_px else "n/a",
                      f"{futures_px - (snap.get('underlying') or 0):+,.2f} basis" if futures_px and snap.get("underlying") else None)
            st.caption(f"Days to expiry: {days_to_expiry(snap.get('expiry'))}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CE OI", f"{snap['ce_oi']:,.0f}", f"{snap['ce_oi_change']:+,.0f}")
            k2.metric("PE OI", f"{snap['pe_oi']:,.0f}", f"{snap['pe_oi_change']:+,.0f}")
            k3.metric("CE Volume", f"{snap.get('ce_volume', 0):,.0f}", f"{snap.get('ce_volume_change', 0):+,.0f}")
            k4.metric("PE Volume", f"{snap.get('pe_volume', 0):,.0f}", f"{snap.get('pe_volume_change', 0):+,.0f}")
            _atm_bits = []
            if arow:
                _atm_bits.append(f"ATM straddle {arow.get('ce_ltp', 0) + arow.get('pe_ltp', 0):,.2f}")
                _atm_bits.append(f"ATM gamma {max(arow.get('ce_gamma', 0), arow.get('pe_gamma', 0)):.5f}")
                _atm_bits.append(f"ATM IV {max(arow.get('ce_iv', 0), arow.get('pe_iv', 0)):.2f}")
            st.caption(f"Snapshot @ {snap.get('fetched_at')} · expiry {snap.get('expiry')}"
                       + (" · " + " · ".join(_atm_bits) if _atm_bits else "")
                       + f" · {len(hist)} snapshot(s) in history")

        # ---- overall recommendation banner ----
        verdict, score, vote_lines = chain_recommendation(hist, snap)
        vc1, vc2 = st.columns([1, 3])
        with vc1:
            if verdict == "BUY CE":
                st.success(f"### 🟢 {verdict}")
            elif verdict == "BUY PE":
                st.error(f"### 🔴 {verdict}")
            else:
                st.warning(f"### ⚪ {verdict}")
            st.caption(f"Net score {score:+d}")
        with vc2:
            st.markdown("**Overall option-chain read**")
            for _l in vote_lines:
                st.markdown(f"- {_l}")

        if len(hist) < 2:
            st.info("📈 Plots need at least two snapshots to draw a line. Press 🔍 Analyze Once again in a minute, "
                    "or start continuous analysis and the charts will build themselves.")

        _norm_mode = st.session_state.app_cfg.get("oca_norm", "Auto (index to 100 when scales differ)")
        _norm_arg = None if _norm_mode.startswith("Auto") else (_norm_mode.startswith("Always"))

        # ---- preset plots ----
        # ---- dedicated comparison plots (fixed chart types where the shape
        # ---- of the data makes one type clearly right) ----
        FIXED_PLOTS = [
            ("📊 CE OI vs PE OI", ["CE OI", "PE OI"], "Bar (grouped)",
             "The two OI walls side by side — which side carries more written positions."),
            ("📊 Change in CE OI vs Change in PE OI", ["CE ΔOI", "PE ΔOI"], "Bar (grouped)",
             "Fresh writing this period. Bars make the sign and relative size obvious."),
            ("📈 PCR vs Spot", ["PCR (OI)", "Price (index/underlying)"], "Line",
             "Positioning ratio against the cash index/stock."),
            ("📈 Max Pain vs Spot", ["Max Pain", "Price (index/underlying)"], "Line",
             "How far price sits from the writers' least-pain strike, and whether the gap is closing."),
            ("📈 Futures vs PCR", ["Futures Price", "PCR (OI)"], "Line",
             "Futures lead the cash market; read against PCR this shows whether leverage agrees with positioning."),
            ("📈 Futures vs Spot", ["Futures Price", "Price (index/underlying)"], "Line",
             "The basis. A widening premium is bullish carry, a discount is bearish."),
            ("📉 Futures vs Volume (line)", ["Futures Price", "Total Volume"], "Line",
             "Is the futures move backed by option-market participation?"),
            ("📊 Futures vs Volume (bar)", ["Futures Price", "Total Volume"], "Bar (grouped)",
             "Same pair as bars, where per-bucket volume is easier to compare."),
            ("📈 Futures vs CE OI vs PE OI", ["Futures Price", "CE OI", "PE OI"], "Line",
             "Which wall builds as futures move."),
            ("📊 Futures vs ΔCE OI vs ΔPE OI (bar)", ["Futures Price", "CE ΔOI", "PE ΔOI"], "Bar (grouped)",
             "Fresh writing on each side against the futures move."),
            ("📈 Futures vs ΔCE OI vs ΔPE OI (line)", ["Futures Price", "CE ΔOI", "PE ΔOI"], "Line",
             "Same three series as lines, for reading the trend rather than the per-bucket size."),
            ("📈 Futures vs Max Pain", ["Futures Price", "Max Pain"], "Line",
             "Futures against the max-pain magnet — the pull is usually most visible here near expiry."),
        ]
        st.markdown("### 🎯 Dedicated comparisons")
        for _fi, (_ftitle, _fmetrics, _fchart, _fwhy) in enumerate(FIXED_PLOTS):
            st.markdown(f"#### {_ftitle}")
            st.caption(_fwhy)
            _ffig, _fnorm = chain_plot(hist, _fmetrics, _ftitle, normalize=_norm_arg, chart_type=_fchart)
            if _ffig is None:
                st.info("No data yet for: " + ", ".join(_missing_metrics(hist, _fmetrics))
                        + ". These come from the option chain (Dhan token + a snapshot during market hours).")
            else:
                st.plotly_chart(_ffig, use_container_width=True, key=f"oca_fixed_{_fi}")
                _sl, (_v, _s, _vl) = chain_plot_summary(hist, snap, _fmetrics, _fnorm)
                with st.container(border=True):
                    st.markdown("**Summary**")
                    for _x in _sl:
                        st.markdown(f"- {_x}")
                    _b = "🟢" if _v == "BUY CE" else ("🔴" if _v == "BUY PE" else "⚪")
                    st.markdown(f"**Recommendation: {_b} {_v}** _(net score {_s:+d})_")
            st.divider()

        # ================= INTRADAY ANALYSIS =================
        st.markdown("### 📉 Intraday Analysis")
        st.caption("Today's session only (IST). Shows how positioning has evolved since the open, independent of any "
                   "longer history held in the database.")
        _day = intraday_slice(hist)
        if _day.empty or len(_day) < 1:
            st.info("No snapshots recorded today yet — run the analysis and this section fills in through the session.")
        else:
            _stats = intraday_stats(_day)
            _order = ["Spot", "Futures", "PCR", "Max Pain", "ATM Straddle", "ATM IV"]
            _shown = [k for k in _order if k in _stats]
            if _shown:
                _cols = st.columns(len(_shown))
                for _i, _k in enumerate(_shown):
                    _v = _stats[_k]
                    _delta = (f"{_v['change']:+,.2f}"
                              + (f" ({_v['change_pct']:+.2f}%)" if _v.get("change_pct") is not None else ""))
                    _cols[_i].metric(_k, f"{_v['last']:,.2f}", _delta)
            _b = []
            for _k in ("CE OI built", "PE OI built", "Volume"):
                if _k in _stats:
                    _b.append(f"{_k}: {_stats[_k]['total']:+,.0f}")
            for _k in ("Spot", "PCR"):
                if _k in _stats:
                    _b.append(f"{_k} range {_stats[_k]['low']:,.2f}–{_stats[_k]['high']:,.2f}")
            if _b:
                st.caption("Session totals — " + " · ".join(_b))

            _iL, _iR = st.columns(2)
            _if1, _ = chain_plot(_day, ["PCR (OI)", "Price (index/underlying)", "Futures Price"],
                                 "Intraday — PCR vs Spot vs Futures")
            if _if1:
                _iL.plotly_chart(_if1, use_container_width=True, key="oca_intraday_1")
            _if2, _ = chain_plot(_day, ["Change in CE OI", "Change in PE OI"],
                                 "Intraday — OI built by side", chart_type="Bar (grouped)")
            if _if2:
                _iR.plotly_chart(_if2, use_container_width=True, key="oca_intraday_2")
            _if3, _ = chain_plot(_day, ["Max Pain", "Price (index/underlying)"],
                                 "Intraday — Max Pain vs Spot")
            if _if3:
                st.plotly_chart(_if3, use_container_width=True, key="oca_intraday_3")

            _iv, _is, _ivl = chain_recommendation(_day, snap)
            _ibadge = "🟢" if _iv == "BUY CE" else ("🔴" if _iv == "BUY PE" else "⚪")
            with st.container(border=True):
                st.markdown(f"**Intraday read: {_ibadge} {_iv}** _(net score {_is:+d}, computed on today's rows only)_")
                for _l in _ivl:
                    st.markdown(f"- {_l}")
        st.divider()

        # ================= ANALYSIS TABLE =================
        st.markdown("### 🧮 Analysis Table")
        _tt1, _tt2 = st.columns(2)
        _tf = cfg_selectbox(_tt1, "Table timeframe", "oca_tbl_tf", TABLE_TIMEFRAMES, default="1m")
        _tp = cfg_selectbox(_tt2, "Table period", "oca_tbl_period", TABLE_PERIODS, default="1d")
        st.caption("This table has its OWN timeframe and period, independent of the sidebar and of the plots above. "
                   "Rows come from real market candles, so 1m/5d gives a row for every traded minute (09:15–15:30, "
                   "trading days only) — not just the minutes a snapshot happened to be taken. Spot, and futures "
                   "where Dhan serves them, are true per-bucket values. Chain metrics (PCR, OI, max pain, straddle, "
                   "greeks) attach to each row from the most recent snapshot at or before it: these are levels or "
                   "cumulative day-to-date figures, so carrying them forward is accurate rather than invented. "
                   "Each Δ column is the change from the previous row at this timeframe.")

        _tfill = cfg_selectbox(st, "Rows between snapshots", "oca_tbl_fill",
                               ["Carry last known chain values forward (continuous)",
                                "Leave blank (only rows with a real snapshot)"],
                               default="Carry last known chain values forward (continuous)")
        _tbl, _tnote = build_chain_analysis_table(_tf, _tp, oca_under, oca_expiry, uinfo=_uinfo,
                                                  carry_forward=_tfill.startswith("Carry"))
        if _tnote:
            st.caption("🗂 " + _tnote)

        with st.expander("🧱 Columns — tick to include, then drag to reorder", expanded=False):
            _tc = st.columns(4)
            _chosen = []
            for _ci, _col in enumerate(TABLE_ALL_COLUMNS):
                _dflt = _col in TABLE_DEFAULT_COLUMNS
                if cfg_checkbox(_tc[_ci % 4], _col, f"oca_tc_{_ci}", _dflt):
                    _chosen.append(_col)
            _ordered = cfg_multiselect(st, "Column order (selection order = display order; "
                                           "anything ticked but not listed is appended)",
                                       "oca_tbl_order", _chosen, default=[])
            _final_cols = [c for c in _ordered if c in _chosen] + [c for c in _chosen if c not in _ordered]
            st.session_state["oca_tbl_cols"] = _final_cols

        _final_cols = st.session_state.get("oca_tbl_cols", TABLE_DEFAULT_COLUMNS)
        _hide_empty = cfg_checkbox(st, "Hide columns that are completely empty", "oca_tbl_hide_empty", True)
        if _tbl.empty:
            st.info("No data for this timeframe/period yet.")
        else:
            _show = [c for c in _final_cols if c in _tbl.columns] or [c for c in TABLE_DEFAULT_COLUMNS if c in _tbl.columns]
            _empty_cols = [c for c in _show if c != "Time" and _tbl[c].notna().sum() == 0]
            if _empty_cols:
                st.warning(f"**{len(_empty_cols)} column(s) have no data at all:** {', '.join(_empty_cols)}. "
                           "These come from the option chain, which needs a Dhan Access Token and a snapshot "
                           "taken during market hours — see the notes at the top of this tab."
                           + ("  They are hidden below." if _hide_empty else ""))
                if _hide_empty:
                    _show = [c for c in _show if c not in _empty_cols]
            if not _show or _show == ["Time"]:
                st.info("Every selected column is empty — nothing to tabulate.")
                _show = [c for c in ["Time", "Spot"] if c in _tbl.columns]
            _disp = _tbl[_show].copy()
            for _c in _disp.columns:
                if _disp[_c].dtype.kind == "f":
                    _disp[_c] = _disp[_c].round(4)
            st.dataframe(_disp, hide_index=True, use_container_width=True, height=420)
            st.download_button("⬇ Download analysis table (CSV)", _disp.to_csv(index=False).encode(),
                               file_name=f"chain_analysis_{oca_under}_{_tf}_{_tp}.csv",
                               mime="text/csv", key="oca_dl_tbl")

            st.markdown("##### 📈 Plot from the table")
            _pc1, _pc2 = st.columns([1, 2])
            _tbl_chart = cfg_selectbox(_pc1, "Chart type", "oca_tbl_chart", CHART_TYPES, default="Line")
            _plot_default = [c for c in ("PCR", "Spot") if c in _show]
            _tbl_plot_cols = cfg_multiselect(_pc2, "Columns to plot (from the table above)",
                                             "oca_tbl_plot_cols",
                                             [c for c in _show if c != "Time"], default=_plot_default)
            if not _tbl_plot_cols:
                st.info("Pick one or more columns to plot.")
            else:
                _tfig = table_plot(_tbl, _tbl_plot_cols, _tbl_chart,
                                   title=f"Table metrics — {_tf} / {_tp}")
                if _tfig is None:
                    st.info("Nothing plottable in the current selection.")
                else:
                    st.plotly_chart(_tfig, use_container_width=True, key="oca_tbl_plot")
                    if len(_tbl_plot_cols) >= 3:
                        st.caption("_Three or more columns selected, so series are indexed to 100 at the first "
                                   "reading (their raw scales differ too much to share an axis). Hover for true "
                                   "values._")
        st.divider()

        st.markdown("### 🧭 Composite views")
        PRESETS = [
            ("1️⃣ PCR vs Price",
             ["PCR (OI)", "Price (index/underlying)"],
             "The core relationship: is positioning confirming the price move or fighting it?"),
            ("2️⃣ PCR vs OI vs Price",
             ["PCR (OI)", "Total OI", "Price (index/underlying)"],
             "Adds total open interest — is the move backed by new positions or just existing ones shuffling?"),
            ("3️⃣ PCR vs Price vs Change in OI",
             ["PCR (OI)", "Price (index/underlying)", "Change in OI (total)"],
             "ΔOI shows where fresh positions are being added right now, rather than the accumulated total."),
            ("4️⃣ Index Price vs OI vs Change in OI",
             ["Price (index/underlying)", "Total OI", "Change in OI (total)"],
             "Price against both the stock and the flow of open interest — classic build-up vs unwinding read."),
            ("5️⃣ PCR vs Price vs Max Pain",
             ["PCR (OI)", "Price (index/underlying)", "Max Pain"],
             "Where price sits relative to the strike that hurts writers least, and how that gap is evolving."),
            ("6️⃣ PCR vs Price vs Change in OI vs Volume",
             ["PCR (OI)", "Price (index/underlying)", "Change in OI (total)", "Total Volume"],
             "Volume confirms whether the ΔOI build is backed by genuine participation."),
            ("7️⃣ PCR vs Change in OI vs Price vs Gamma",
             ["PCR (OI)", "Change in OI (total)", "Price (index/underlying)", "ATM Gamma"],
             "Gamma rises into expiry — high gamma with compressing premium is the blast-risk regime."),
            ("8️⃣ CE vs PE Open Interest",
             ["CE OI", "PE OI", "Price (index/underlying)"],
             "The two sides side by side: which wall is being built faster?"),
        ]
        for _pi, (_title, _metrics, _why) in enumerate(PRESETS):
            st.markdown(f"#### {_title}")
            st.caption(_why)
            _fig, _was_norm = chain_plot(hist, _metrics, _title, normalize=_norm_arg)
            if _fig is None:
                st.info("No data yet for: " + ", ".join(_missing_metrics(hist, _metrics))
                        + ". These come from the option chain (Dhan token + a snapshot during market hours).")
            else:
                st.plotly_chart(_fig, use_container_width=True, key=f"oca_preset_{_pi}")
                _sum_lines, (_v, _s, _vl) = chain_plot_summary(hist, snap, _metrics, _was_norm)
                with st.container(border=True):
                    st.markdown("**Summary**")
                    for _sl in _sum_lines:
                        st.markdown(f"- {_sl}")
                    _badge = "🟢" if _v == "BUY CE" else ("🔴" if _v == "BUY PE" else "⚪")
                    st.markdown(f"**Recommendation: {_badge} {_v}** _(net score {_s:+d} across the six chain votes)_")
            st.divider()

        # ---- custom plot ----
        st.markdown("#### 🎛 Custom Plot")
        _picked = st.session_state.get("oca_picked", [])
        if not _picked:
            st.info("Tick metrics in the '🎛 Custom plot' expander above to build your own chart.")
        else:
            st.caption("Plotting: " + ", ".join(_picked))
            _cfig, _cnorm = chain_plot(hist, _picked, "Custom metric comparison", normalize=_norm_arg, height=480)
            if _cfig is None:
                st.info("No data yet for the selected metrics.")
            else:
                st.plotly_chart(_cfig, use_container_width=True, key="oca_custom")
                _sum_lines, (_v, _s, _vl) = chain_plot_summary(hist, snap, _picked, _cnorm)
                with st.container(border=True):
                    st.markdown("**Summary**")
                    for _sl in _sum_lines:
                        st.markdown(f"- {_sl}")
                    _badge = "🟢" if _v == "BUY CE" else ("🔴" if _v == "BUY PE" else "⚪")
                    st.markdown(f"**Recommendation: {_badge} {_v}** _(net score {_s:+d})_")

        st.divider()

        # ---- Groq output ----
        st.markdown("#### 🤖 Groq AI Analysis")
        if _running and st.session_state.app_cfg.get("groq_auto") and st.session_state.app_cfg.get("groq_api_key"):
            _model = (st.session_state.app_cfg.get("groq_model_custom")
                      if str(st.session_state.app_cfg.get("groq_model_pick", "")).startswith("(custom")
                      else st.session_state.app_cfg.get("groq_model_pick"))
            st.session_state["groq_last"] = groq_analyze_chain(
                st.session_state.app_cfg.get("groq_api_key"), _model, snap, hist,
                st.session_state.app_cfg.get("groq_extra", ""))
            st.session_state["groq_last_at"] = ist_now().strftime("%H:%M:%S IST")
        if st.session_state.get("groq_last"):
            st.caption(f"Generated @ {st.session_state.get('groq_last_at', 'n/a')} · "
                       "AI-generated opinion, not financial advice.")
            st.markdown(st.session_state["groq_last"])
        else:
            st.info("No AI analysis yet — open the '🤖 Groq AI analysis' expander above, add your API key and "
                    "press **Ask Groq now** (or tick 'Include Groq in continuous analysis').")

        st.divider()

        # ---- full chain + history tables ----
        if snap and snap.get("strikes"):
            with st.expander("📋 Full option chain (all strikes)", expanded=False):
                _rows = []
                _mp2, _ = compute_max_pain(snap["strikes"])
                _atm2, _ = _atm_row(snap)
                for _s in sorted(snap["strikes"].keys()):
                    _r = snap["strikes"][_s]
                    _rows.append({
                        "CE OI": _r["ce_oi"], "CE ΔOI": _r["ce_oi_change"], "CE Vol": _r["ce_vol"],
                        "CE IV": _r["ce_iv"], "CE LTP": _r["ce_ltp"],
                        "Strike": _s,
                        "PE LTP": _r["pe_ltp"], "PE IV": _r["pe_iv"], "PE Vol": _r["pe_vol"],
                        "PE ΔOI": _r["pe_oi_change"], "PE OI": _r["pe_oi"],
                        "PCR": round(_r["pe_oi"] / _r["ce_oi"], 3) if _r["ce_oi"] else None,
                        "Marker": ("← ATM" if _s == _atm2 else ("← Max Pain" if _s == _mp2 else "")),
                    })
                _cdf = pd.DataFrame(_rows)
                st.dataframe(_cdf, hide_index=True, use_container_width=True)
                st.download_button("⬇ Download chain (CSV)", _cdf.to_csv(index=False).encode(),
                                   file_name=f"option_chain_{oca_under}_{oca_expiry}.csv", mime="text/csv",
                                   key="oca_dl_chain")

        with st.expander("🧾 Snapshot history (absolute / % / n× changes)", expanded=False):
            _htbl = build_chain_history_table()
            if _htbl.empty:
                st.info("No snapshots recorded yet.")
            else:
                st.dataframe(_htbl, hide_index=True, use_container_width=True)
                st.download_button("⬇ Download history (CSV)", _htbl.to_csv(index=False).encode(),
                                   file_name="chain_history.csv", mime="text/csv", key="oca_dl_hist")

        st.caption(f"Last render {ist_now().strftime('%H:%M:%S IST')}"
                   + (f" · auto-refreshing every {int(oca_interval)}s" if _running else " · idle"))

    _chain_analysis_body()


# -------------------------------------------------------------- SCREENER ----
with tab_screen:
    st.subheader("🔎 Screener")
    st.caption("Runs the EXACT configuration selected in the sidebar — strategy, its parameters, timeframe, period, "
               "entry filters, flip and Trade Direction — across a list of stocks, then groups the hits by how "
               "recently the signal fired.")

    sc1, sc2, sc3 = st.columns([1.2, 1, 1])
    scr_universe = cfg_selectbox(sc1, "Universe", "scr_universe",
                                 ["Nifty 50", "Nifty 100", "Custom list"], default="Nifty 50")
    scr_source = cfg_selectbox(sc2, "Data source", "scr_source",
                               ["Auto (follow Data Source setting)", "yfinance", "Dhan"],
                               default="Auto (follow Data Source setting)")
    scr_before = cfg_number(sc3, "'Just Before' window (candles)", "scr_before_bars",
                            5, 1, 100, is_int=True)

    if scr_universe == "Custom list":
        scr_custom = cfg_text(st, "Symbols (comma-separated, NSE — e.g. RELIANCE, TCS, INFY)",
                              "scr_custom", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK")
        _symbols = [s.strip().upper() for s in str(scr_custom).split(",") if s.strip()]
    elif scr_universe == "Nifty 100":
        _symbols = list(NIFTY100_SYMBOLS)
    else:
        _symbols = list(NIFTY50_SYMBOLS)

    scr_limit = cfg_number(st, "Maximum symbols to scan (protects against rate limits)", "scr_limit",
                           100, 1, 500, is_int=True)
    if len(_symbols) > int(scr_limit):
        st.caption(f"⚠️ {len(_symbols)} symbols in the universe but the cap is {int(scr_limit)} — "
                   "raise the cap to scan them all.")
    _symbols = _symbols[:int(scr_limit)]

    _src_label = ("Dhan" if scr_source == "Dhan"
                  else ("yfinance" if scr_source == "yfinance"
                        else ("Dhan" if dhan_feed_active() else "yfinance")))
    st.info(f"Ready to scan **{len(_symbols)}** symbols on **{interval} / {period}** using **{strategy}** "
            f"via **{_src_label}**."
            + (f"  ⏱ yfinance enforces a {RATE_LIMIT_DELAY}s pause per symbol, so this will take roughly "
               f"{len(_symbols) * RATE_LIMIT_DELAY:.0f}–{len(_symbols) * (RATE_LIMIT_DELAY + 0.7):.0f}s. "
               "Failures are skipped and listed rather than crashing the scan."
               if _src_label == "yfinance" else "  Dhan has no artificial delay, so this runs quickly."))

    b1, b2 = st.columns([1, 3])
    _run_scan = b1.button("▶ Run Analysis", type="primary", use_container_width=True)

    if _run_scan:
        _prog = st.progress(0.0, text="Starting scan…")

        def _cb(frac, sym):
            _prog.progress(min(max(frac, 0.0), 1.0), text=f"Scanning {sym} … ({int(frac * 100)}%)")

        with st.spinner("Screening…"):
            _res, _errs = screener_scan(
                _symbols, strategy, params, filters, interval, period,
                source=("Dhan" if scr_source == "Dhan" else
                        ("yfinance" if scr_source == "yfinance" else "Auto")),
                before_bars=int(scr_before), progress=_cb)
        _prog.empty()
        st.session_state["scr_results"] = _res
        st.session_state["scr_errors"] = _errs
        st.session_state["scr_run_at"] = ist_now().strftime("%d-%b-%Y %H:%M:%S IST")
        db_save_screener_run(_res, strategy, interval, period, scr_universe)

    _res = st.session_state.get("scr_results")
    _errs = st.session_state.get("scr_errors")

    if _res is None:
        st.info("Press **▶ Run Analysis** to scan. Change the strategy, timeframe, period or filters in the sidebar "
                "first — the screener uses whatever is configured there.")
    else:
        st.caption(f"Last run: {st.session_state.get('scr_run_at', 'n/a')} · {len(_symbols)} symbols requested · "
                   f"{0 if _res.empty else len(_res)} signals found · "
                   f"{0 if (_errs is None or _errs.empty) else len(_errs)} skipped")

        st.markdown("""
**How the three groups are defined**
- 🟢 **Just Now** — the signal is on the most recently *closed* candle. This is the bar the live engine would act on right now.
- 🔵 **Just After** — the signal is on that same closed candle **and** the currently forming candle has already moved further in the signal's direction, so early follow-through is confirming it.
- 🟡 **Just Before** — the signal fired within the previous few candles (your window above). Already triggered and the move has begun, so entry is later than ideal.
""")

        if _res.empty:
            st.warning("No symbol produced a signal in any of the three windows with the current configuration. "
                       "That is a normal outcome for a selective strategy — try a longer 'Just Before' window, a "
                       "different timeframe, or fewer entry filters.")
        else:
            _tot = st.columns(3)
            for _i, (_bk, _emoji) in enumerate([("Just Now", "🟢"), ("Just After", "🔵"), ("Just Before", "🟡")]):
                _tot[_i].metric(f"{_emoji} {_bk}", int((_res["Bucket"] == _bk).sum()))

            for _bk, _emoji, _blurb in [
                ("Just Now", "🟢", "Signal on the latest closed candle — actionable right now."),
                ("Just After", "🔵", "Signal on the latest closed candle with follow-through already underway."),
                ("Just Before", "🟡", "Signal fired in the last few candles — the move has already started."),
            ]:
                _sub = _res[_res["Bucket"] == _bk].drop(columns=["Bucket"])
                st.markdown(f"#### {_emoji} {_bk} — {len(_sub)} stock(s)")
                st.caption(_blurb)
                if _sub.empty:
                    st.caption("_None in this group._")
                else:
                    _sub = _sub.sort_values("Move %", ascending=False, na_position="last")
                    st.dataframe(_sub, hide_index=True, use_container_width=True)
                st.divider()

            st.download_button("⬇ Download screener results (CSV)",
                               _res.to_csv(index=False).encode(),
                               file_name=f"screener_{strategy.replace(' ', '_')}_{interval}.csv",
                               mime="text/csv", key="scr_dl")

        if _errs is not None and not _errs.empty:
            with st.expander(f"⚠️ {len(_errs)} symbol(s) skipped", expanded=False):
                st.caption("Common causes: yfinance rate limiting on a long scan, a symbol not listed on the chosen "
                           "source, or not enough candles for the strategy's indicators to warm up. The scan "
                           "continues past every one of these rather than aborting.")
                st.dataframe(_errs, hide_index=True, use_container_width=True)

    if strategy in OPTION_CHAIN_STRATEGIES:
        st.warning("⚠️ The selected strategy reads a live option chain, which exists per-underlying rather than "
                   "per-candle. Screening many stocks with it is not meaningful — pick a price-based strategy for "
                   "the screener, or use the Option Chain Analysis tab for chain work.")


# --------------------------------------------------------- DATA DOWNLOAD ----
with tab_dl:
    st.subheader("⬇️ Data Download")
    st.caption("Export candles as CSV or Excel for any instrument — equities, indices, index/stock options, or "
               "index/stock futures. Options and futures come from Dhan (yfinance has no derivatives data); "
               "equities and indices work on either source.")

    d1, d2 = st.columns([1, 2])
    dl_type = cfg_selectbox(d1, "Instrument type", "dl_type",
                            ["Stock (equity)", "Index", "Index Options", "Stock Options",
                             "Index Futures", "Stock Futures"], default="Stock (equity)")
    _is_deriv = dl_type not in ("Stock (equity)", "Index")
    _is_opt = "Options" in dl_type
    _is_idx_deriv = dl_type.startswith("Index") and _is_deriv

    dl_ticker, dl_sec_id, dl_segment, dl_instr = None, None, None, None
    _label = ""

    if dl_type == "Index":
        _idx = cfg_selectbox(d2, "Index", "dl_index", list(DHAN_INDEX_MAP.keys()), default="Nifty50")
        dl_ticker = TICKER_MAP.get(_idx)
        _label = _idx
    elif dl_type == "Stock (equity)":
        _sym = cfg_text(d2, "Stock symbol (NSE, e.g. RELIANCE)", "dl_stock", "RELIANCE")
        dl_ticker = f"{_yf_symbol_to_plain(_sym)}.NS"
        _label = _yf_symbol_to_plain(_sym)
    else:
        if _is_idx_deriv:
            _u = cfg_selectbox(d2, "Underlying index", "dl_deriv_index", list(DHAN_INDEX_MAP.keys()),
                               default="Nifty50")
            _uinfo_dl = resolve_chain_underlying("Index", _u)
        else:
            _u = cfg_text(d2, "Underlying stock (NSE, e.g. RELIANCE)", "dl_deriv_stock", "RELIANCE")
            _uinfo_dl = resolve_chain_underlying("Stock", _u)

        if not _uinfo_dl:
            st.warning(f"Could not resolve '{_u}' in Dhan's scrip master. Derivatives need a symbol that has "
                       "listed F&O contracts.")
        else:
            _scrip = _uinfo_dl["opt_instrument"] if _is_opt else _uinfo_dl["fut_instrument"]
            _exps = dhan_get_expiries(_uinfo_dl["underlying"], _scrip, _uinfo_dl["exchange"])
            e1, e2, e3 = st.columns(3)
            if _exps:
                _exp = cfg_selectbox(e1, "Expiry", "dl_expiry", _exps, default=_exps[0])
            else:
                _exp = cfg_text(e1, "Expiry (YYYY-MM-DD)", "dl_expiry_manual", "")
                st.caption("Expiry list unavailable — check the Dhan token, then enter the expiry manually.")
            if _is_opt:
                _otype = cfg_selectbox(e2, "Option type", "dl_opt_type", ["CE", "PE"], default="CE")
                _strikes = dhan_get_strikes(_uinfo_dl["underlying"], _exp, _scrip,
                                            _uinfo_dl["exchange"]) if _exp else []
                if _strikes:
                    _atm_dl = round_to_nearest_strike(_current_underlying_ltp(_uinfo_dl["yf"]), _strikes)
                    _strike = cfg_selectbox(e3, "Strike (ATM pre-selected)", "dl_strike", _strikes,
                                            default=_atm_dl if _atm_dl in _strikes else _strikes[len(_strikes) // 2])
                else:
                    _strike = cfg_number(e3, "Strike (manual)", "dl_strike_manual", 0.0, 0.0, 1e7)
                _info = (dhan_lookup_option(_uinfo_dl["underlying"], _exp, _strike, _otype, _scrip,
                                            _uinfo_dl["exchange"]) if (_exp and _strike) else None)
                _label = f"{_uinfo_dl['underlying']}_{_exp}_{_strike:g}{_otype}" if _exp and _strike else "option"
            else:
                _info = dhan_lookup_future(_uinfo_dl["underlying"], _exp, _scrip,
                                           _uinfo_dl["exchange"]) if _exp else None
                _label = f"{_uinfo_dl['underlying']}_{_exp}_FUT" if _exp else "future"
            if _info:
                dl_sec_id = _info["security_id"]
                dl_segment = f"{_uinfo_dl['exchange']}_FNO"
                dl_instr = _scrip
                st.caption(f"Resolved security ID **{dl_sec_id}** · segment {dl_segment} · lot size "
                           f"{_info.get('lot_size', 'n/a')}")
            else:
                st.warning("Contract not resolved yet — pick an expiry (and strike) that exists in the scrip master.")

    t1, t2, t3 = st.columns(3)
    _dl_tf_map = available_tf_period_map()
    dl_tf = cfg_selectbox(t1, "Timeframe", "dl_timeframe", list(_dl_tf_map.keys()), default="5m")
    dl_period = cfg_selectbox(t2, "Period", "dl_period", _dl_tf_map.get(dl_tf, ["1d"]),
                              default=_dl_tf_map.get(dl_tf, ["1d"])[0])
    dl_fmt = cfg_selectbox(t3, "File format", "dl_format", ["CSV", "Excel (.xlsx)"], default="CSV")

    if _is_deriv:
        st.info("Derivatives data is served by Dhan and needs a valid Access Token in the sidebar's "
                "🔐 Dhan Account section. yfinance has no options or futures candles.")

    if st.button("⬇️ Fetch data", type="primary"):
        _df_dl, _err = None, None
        try:
            if _is_deriv:
                _cid, _tok = _dhan_creds()
                if not _tok:
                    _err = "No Dhan Access Token — derivatives cannot be fetched."
                elif not dl_sec_id:
                    _err = "The contract has not been resolved yet."
                else:
                    _df_dl = _dhan_fetch_candles_cached(dl_sec_id, dl_segment, dl_instr,
                                                        dl_tf, dl_period, hash(_tok) % 10_000_019)
                    _df_dl = normalize_index_to_ist(_df_dl, "")
            else:
                _df_dl = fetch_data(dl_ticker, dl_tf, dl_period)
        except Exception as exc:
            _err = f"{type(exc).__name__}: {exc}"

        if _err:
            st.error(_err)
        elif _df_dl is None or _df_dl.empty:
            st.warning("No data returned. Common causes: the market has never traded this contract, the period "
                       "exceeds what the source retains for this timeframe, or the token/expiry is wrong.")
        else:
            _out = _df_dl.copy()
            _out.index.name = "Datetime (IST)"
            _out = _out.reset_index()
            st.session_state["dl_result"] = _out
            st.session_state["dl_name"] = f"{_label or 'data'}_{dl_tf}_{dl_period}".replace(" ", "_")

    _res_dl = st.session_state.get("dl_result")
    if _res_dl is not None and not _res_dl.empty:
        st.success(f"{len(_res_dl):,} rows fetched.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(_res_dl):,}")
        try:
            c2.metric("From", str(_res_dl.iloc[0, 0])[:16])
            c3.metric("To", str(_res_dl.iloc[-1, 0])[:16])
            c4.metric("Last close", f"{float(_res_dl['Close'].iloc[-1]):,.2f}")
        except Exception:
            pass
        st.dataframe(_res_dl.tail(200), hide_index=True, use_container_width=True, height=380)
        st.caption("Preview shows the most recent 200 rows; the download contains everything fetched.")

        _fname = st.session_state.get("dl_name", "data")
        if dl_fmt.startswith("CSV"):
            st.download_button("⬇️ Download CSV", _res_dl.to_csv(index=False).encode(),
                               file_name=f"{_fname}.csv", mime="text/csv", key="dl_btn_csv")
        else:
            try:
                _buf = io.BytesIO()
                with pd.ExcelWriter(_buf, engine="xlsxwriter") as _xw:
                    _res_dl.to_excel(_xw, index=False, sheet_name="Data")
                st.download_button("⬇️ Download Excel", _buf.getvalue(), file_name=f"{_fname}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="dl_btn_xlsx")
            except Exception as exc:
                st.warning(f"Excel export needs the xlsxwriter package ({exc}). Falling back to CSV.")
                st.download_button("⬇️ Download CSV instead", _res_dl.to_csv(index=False).encode(),
                                   file_name=f"{_fname}.csv", mime="text/csv", key="dl_btn_csv_fb")


# ---------------------------------------------------------------- ADMIN ----
with tab_admin:
    # READ-ONLY by design. This panel used to render a second, editable copy
    # of every sidebar control; because two widget sets rendered per run and
    # each tried to sync the other, selections snapped back to their previous
    # values. The duplicate widgets are gone — the sidebar is the single
    # owner of configuration — and this tab now presents the full active
    # configuration, full-width, for review.
    st.subheader("🛠 Admin Panel — active configuration")
    st.caption("Read-only view of every setting currently in force. Edit anything in the sidebar on the left; "
               "this panel reflects those values immediately.")

    _dsrc = ("Dhan data feed (no delay)"
             if (dhan_feed_active() and dhan_resolve_feed_instrument(ticker) is not None)
             else "yfinance (0.3s delay per call)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Ticker", f"{ticker_choice}")
    a2.metric("Timeframe / Period", f"{interval} / {period}")
    a3.metric("Paper Quantity", f"{qty}")
    a4.metric("Data Source", "Dhan" if _dsrc.startswith("Dhan") else "yfinance")

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("##### 📐 Strategy & Exits")
        _strategy_rows = [
            ("Strategy", strategy),
            ("Stoploss Type", sl_type),
            ("Target Type", target_type),
            ("Trade Direction", params.get("trade_direction", "Both")),
            ("Flip / Reverse Entries", "ON" if params.get("flip_signals") else "off"),
        ]
        if strategy == "Hybrid (Combine Strategies)":
            _strategy_rows.append(("Hybrid members", ", ".join(params.get("hybrid_members", [])) or "— none —"))
            _strategy_rows.append(("Hybrid logic", params.get("hybrid_mode", "AND")))
        if strategy == "OI Based (CE/PE Open Interest)":
            _strategy_rows.append(("OI underlying / expiry",
                                   f"{params.get('oi_underlying','')} · {params.get('oi_expiry','')}"))
            _strategy_rows.append(("OI interpretation", "FLIPPED" if params.get("oi_flip") else "standard"))
        st.dataframe(pd.DataFrame(_strategy_rows, columns=["Setting", "Value"]),
                     hide_index=True, use_container_width=True)

        st.markdown("##### 🎛 Strategy Parameters")
        _pp = {k: v for k, v in params.items() if k not in ("hybrid_members",)}
        st.dataframe(pd.DataFrame(sorted(_pp.items()), columns=["Parameter", "Value"]),
                     hide_index=True, use_container_width=True)

    with ac2:
        st.markdown("##### 🔍 Entry Filters")
        _active = [k.replace("_enabled", "") for k, v in filters.items() if v is True]
        st.write(("Active: " + ", ".join(_active)) if _active else "No entry filters active.")
        st.dataframe(pd.DataFrame(sorted((k, str(v)) for k, v in filters.items()),
                                  columns=["Filter setting", "Value"]),
                     hide_index=True, use_container_width=True)

        st.markdown("##### 🚧 Risk Gates")
        _g = [(k, str(v)) for k, v in (gates or {}).items()]
        st.dataframe(pd.DataFrame(sorted(_g) or [("(none configured)", "")],
                                  columns=["Gate", "Value"]),
                     hide_index=True, use_container_width=True)

    st.markdown("##### 💾 Data Persistence (SQLite)")
    st.caption("These controls are editable here. They live only in this panel — they are not duplicated in the "
               "sidebar, so there is exactly one widget per setting and no chance of the two fighting each other.")
    dp1, dp2 = st.columns([1, 2])
    _db_on = cfg_checkbox(dp1, "Store all data in a database", "db_enabled", False)
    _db_path = cfg_text(dp2, "Database file (SQLite)", "db_path", DB_DEFAULT_PATH)
    if _db_on:
        _ok, _err = db_init()
        if not _ok:
            st.error(f"Could not open the database: {_err}")
        else:
            db_bootstrap()
            _stats = db_stats()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Closed trades stored", _stats.get("trades", 0))
            s2.metric("Chain snapshots", _stats.get("chain_snapshots", 0))
            s3.metric("Screener runs", _stats.get("screener_runs", 0))
            s4.metric("Open position saved", _stats.get("open_position", 0))
            st.success("Persistence is ON. The open position, closed trades, option-chain snapshots and screener "
                       "runs are written to disk. If the session drops while a trade is still running, that trade "
                       "is restored on the next start and keeps being managed until a genuine exit closes it — "
                       "instead of vanishing at end of day.")
            if st.session_state.get("db_last_error"):
                st.warning(f"Last database error: {st.session_state['db_last_error']}")
    else:
        st.info("Persistence is OFF (default): everything lives in memory only, so an open position is lost if the "
                "browser tab is discarded or the machine sleeps. Enable it to survive restarts and to unlock the "
                "multi-day analysis windows on the Option Chain tab.")

    st.markdown("##### 📦 Delivery Positions (carried over from intraday)")
    if not _db_on:
        st.caption("Enable the database above to store and resume delivery conversions across sessions.")
    else:
        _dp_open = db_load_delivery_positions("OPEN")
        _dp_all = db_load_delivery_positions()
        if not _dp_all:
            st.caption("No positions have been converted to delivery yet. When an intraday position has hit neither "
                       "SL nor target by the configured cut-off, it is carried over and listed here.")
        else:
            _dp_df = pd.DataFrame([{
                "ID": r["id"], "Converted": str(r["converted_at"])[:19].replace("T", " "),
                "Ticker": r["ticker"], "Instrument": r["instrument"], "Strategy": r["strategy"],
                "Direction": r["direction"], "Entry": r["entry_price"], "Qty": r["qty"],
                "SL": r["sl"], "Target": r["target"], "Status": r["status"],
                "Resumed": str(r["resumed_at"] or "")[:19].replace("T", " "),
            } for r in _dp_all])
            st.dataframe(_dp_df, hide_index=True, use_container_width=True)

            if _dp_open:
                _opts = {f"#{r['id']} · {r['ticker']} · {r['direction']} {r['qty']} @ {r['entry_price']}": r
                         for r in _dp_open}
                _pick = st.selectbox("Select a delivery position", list(_opts.keys()), key="dp_pick")
                _row = _opts[_pick]
                r1, r2, r3 = st.columns(3)
                if r1.button("▶ Resume trading this position", use_container_width=True, key="dp_resume"):
                    try:
                        _pos = json.loads(_row["payload"])
                        # Clear the conversion flag so the normal exit rules
                        # (SL, target, signal exits) manage it again from now on.
                        _pos["converted_to_delivery"] = False
                        _pos.pop("delivery_converted_at", None)
                        _pos["resumed_from_delivery"] = True
                        if _pos.get("entry_time"):
                            try:
                                _pos["entry_time"] = pd.to_datetime(_pos["entry_time"])
                            except Exception:
                                pass
                        st.session_state.live_positions = [_pos]
                        db_update_delivery_status(_row["id"], "RESUMED", "resumed_at")
                        db_save_open_position(_pos, _row["ticker"], _row["strategy"])
                        st.success(f"Position #{_row['id']} on {_row['ticker']} is live again — the Live Trading tab "
                                   "will manage it under the current SL/target rules.")
                    except Exception as exc:
                        st.error(f"Could not resume: {exc}")
                if r2.button("✔ Mark as closed", use_container_width=True, key="dp_close"):
                    db_update_delivery_status(_row["id"], "CLOSED", "closed_at")
                    st.success(f"Position #{_row['id']} marked closed.")
                r3.caption("Resuming loads it as the active live position. Only one position is managed at a time, "
                           "so any currently open position is replaced.")
            else:
                st.caption("No delivery positions are currently open.")

    st.markdown("##### 🏦 Broker / Feed / Notifications")
    st.json({
        "Data Source": _dsrc,
        "Dhan Live Orders": dhan_enabled,
        "Dhan Client ID set": bool(str(dhan_client_id or "").strip()),
        "Dhan Access Token set": bool(str(dhan_access_token or "").strip()),
        "Dhan Quantity": config.get("dhan_qty"),
        "Entry / Exit Order Type": f"{config.get('entry_order_type')} / {config.get('exit_order_type')}",
        "Options Mode": config.get("options_mode"),
        "Premium Mode": config.get("premium_mode"),
        "Dhan Product Config": product_cfg,
        "Email Notifications": config.get("email_enabled"),
        "Email To": config.get("email_to") if config.get("email_enabled") else None,
        "Time-Based Risk Control": risk_ctrl,
        "Walk-Forward": {"enabled": wf_enabled, "folds": wf_folds},
        "Cost Model": {"enabled": cost_enabled, **(cost_cfg or {})},
    })

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
