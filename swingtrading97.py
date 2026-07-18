"""
AlgoTrader Pro — Single-file Streamlit algorithmic trading workbench.

Educational / research tool. Not investment advice. Past backtest performance
never guarantees future returns. Live "trading" in this app is a paper/
simulation layer unless you explicitly enable the Dhan broker checkbox and
wire in verified credentials — do that only after testing in a sandbox.
"""

import smtplib
import ssl
import time
from datetime import datetime, timedelta, time as dt_time
from email.mime.text import MIMEText

import numpy as np
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

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

RATE_LIMIT_DELAY = 0.3  # seconds, mandatory pause between YFINANCE calls only.
# Dhan has no comparable API rate-limit issue, so the Dhan data path deliberately
# applies NO delay at all — do not add one there.

# ---------------------------------------------------------------------------
# DHAN — data feed + order placement constants
# ---------------------------------------------------------------------------
DHAN_API_BASE = "https://api.dhan.co/v2"
DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# Well-known Dhan security IDs for the index tickers this app ships with.
INDEX_SECURITY_MAP = {
    "^NSEI": {"security_id": "13", "segment": "IDX_I", "instrument": "INDEX"},
    "^NSEBANK": {"security_id": "25", "segment": "IDX_I", "instrument": "INDEX"},
    "^BSESN": {"security_id": "51", "segment": "IDX_I", "instrument": "INDEX"},
}

# Default option quantities requested for index options (current lot sizes
# as configured by the user — editable in the Dhan section).
INDEX_OPTION_LOT_DEFAULTS = {"NIFTY": 65, "BANKNIFTY": 35, "SENSEX": 20}

DHAN_INSTRUMENTS = [
    "Stock Intraday", "Stock Delivery", "Stock Futures",
    "Index Futures", "Stock Options", "Index Options",
]

DHAN_INTERVAL_MAP = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}

PERIOD_DAYS = {
    "1d": 1, "5d": 5, "7d": 7, "1mo": 31, "3mo": 93, "6mo": 186,
    "1y": 366, "2y": 732, "3y": 1098, "5y": 1830, "10y": 3660,
    "20y": 7320, "30y": 10980,
}

# ============================================================================
# SESSION STATE
# ============================================================================

for key, default in {
    "live_positions": [],
    "live_history": [],
    "sidebar_overrides": {},
    "opt_results": {},
    "last_backtest": None,
    "last_backtest_df": None,
    "live_running": False,
    "last_acted_signal_marker": None,
    "app_cfg": {},          # single source of truth for ALL config widgets
    "_ltp_prev": None,      # previous LTP shown, for the live delta readout
    "live_day_stats": {},   # {"date": <date>, "entries": n} — for Max Trades/Day
    "last_trade_event_ts": None,  # time.time() of last entry/exit — entry cooldown
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================================
# INDICATORS
# All formulas follow TradingView's conventions so values match TV charts at
# the same settings: RSI/ATR/ADX/±DI use Wilder's RMA smoothing (ewm alpha=1/n,
# TV's ta.rma), EMA/MACD/TEMA use standard EMA (span=n, adjust=False), Bollinger
# uses population stdev (ddof=0, TV's ta.stdev biased=true), CCI uses mean
# absolute deviation about the SMA (TV's ta.dev), Stochastic matches TV's raw
# %K with default smoothing=1, and Supertrend uses RMA-ATR with TV's
# band-carry-forward rules. Residual differences vs a TV chart come from the
# DATA (session boundaries / feed), not the math.
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
# DATA FETCH (rate-limited)
# ============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_data_yf(ticker, interval, period):
    """yfinance path — keeps the mandatory 0.3s rate-limit delay intact."""
    time.sleep(RATE_LIMIT_DELAY)
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    return df


# ============================================================================
# DHAN DATA FEED + SCRIP MASTER + EMAIL HELPERS
# ============================================================================

def _dhan_headers(client_id, token):
    return {"access-token": str(token), "client-id": str(client_id), "Content-Type": "application/json"}


@st.cache_data(ttl=86400, show_spinner=False)
def load_dhan_scrip_master():
    """Downloads and caches (24h) Dhan's public instrument master CSV. Used to
    resolve security IDs, F&O expiries, strikes, and lot sizes automatically."""
    try:
        df = pd.read_csv(DHAN_SCRIP_MASTER_URL, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def lookup_equity_security_id(symbol, exchange="NSE"):
    """NSE/BSE cash-equity security ID for a plain symbol (e.g. 'RELIANCE')."""
    sm = load_dhan_scrip_master()
    if sm.empty:
        return None
    try:
        rows = sm[
            (sm["SEM_EXM_EXCH_ID"].astype(str).str.strip() == exchange)
            & (sm["SEM_INSTRUMENT_NAME"].astype(str).str.strip() == "EQUITY")
            & (sm["SEM_TRADING_SYMBOL"].astype(str).str.strip().str.upper() == str(symbol).upper())
        ]
        if "SEM_SERIES" in rows.columns:
            eq = rows[rows["SEM_SERIES"].astype(str).str.strip().isin(["EQ", "A", "B", "BE"])]
            if not eq.empty:
                rows = eq
        if rows.empty:
            return None
        return str(int(float(rows.iloc[0]["SEM_SMST_SECURITY_ID"])))
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def dhan_derivative_table(underlying, kind):
    """All FUT or OPT scrip-master rows for one underlying (e.g. 'NIFTY').
    kind: 'FUT' or 'OPT'. Trading symbols look like 'NIFTY-Dec2025-FUT' /
    'NIFTY-Dec2025-24000-CE', so an 'UNDERLYING-' prefix match is exact enough
    (it can't accidentally match NIFTYNXT50-, which has its own prefix)."""
    sm = load_dhan_scrip_master()
    if sm.empty:
        return pd.DataFrame()
    inst_names = {"FUT": ("FUTIDX", "FUTSTK"), "OPT": ("OPTIDX", "OPTSTK")}[kind]
    try:
        rows = sm[sm["SEM_INSTRUMENT_NAME"].astype(str).str.strip().isin(inst_names)].copy()
        pref = str(underlying).upper() + "-"
        rows = rows[rows["SEM_TRADING_SYMBOL"].astype(str).str.upper().str.startswith(pref)]
        rows["_expiry"] = pd.to_datetime(rows["SEM_EXPIRY_DATE"], errors="coerce").dt.date
        rows = rows.dropna(subset=["_expiry"])
        return rows
    except Exception:
        return pd.DataFrame()


def dhan_expiry_list(underlying, kind):
    """Sorted list of upcoming expiry dates ('YYYY-MM-DD') for FUT/OPT."""
    rows = dhan_derivative_table(underlying, kind)
    if rows.empty:
        return []
    today = datetime.now().date()
    return [d.isoformat() for d in sorted({d for d in rows["_expiry"] if d >= today})]


def dhan_option_strikes(underlying, expiry_str):
    """Sorted unique strike prices available for one option expiry."""
    rows = dhan_derivative_table(underlying, "OPT")
    if rows.empty or not expiry_str:
        return []
    try:
        exp = pd.to_datetime(expiry_str).date()
        sel = rows[rows["_expiry"] == exp]
        return sorted({float(s) for s in pd.to_numeric(sel["SEM_STRIKE_PRICE"], errors="coerce").dropna()})
    except Exception:
        return []


def lookup_future_security_id(underlying, expiry_str):
    rows = dhan_derivative_table(underlying, "FUT")
    if rows.empty or not expiry_str:
        return None
    try:
        exp = pd.to_datetime(expiry_str).date()
        sel = rows[rows["_expiry"] == exp]
        if sel.empty:
            return None
        return str(int(float(sel.iloc[0]["SEM_SMST_SECURITY_ID"])))
    except Exception:
        return None


def lookup_option_security_id(underlying, expiry_str, strike, opt_type):
    """Security ID for one exact (underlying, expiry, strike, CE/PE) leg."""
    rows = dhan_derivative_table(underlying, "OPT")
    if rows.empty or not expiry_str or not strike:
        return None
    try:
        exp = pd.to_datetime(expiry_str).date()
        sel = rows[rows["_expiry"] == exp].copy()
        sel["_strike"] = pd.to_numeric(sel["SEM_STRIKE_PRICE"], errors="coerce")
        sel = sel[(sel["_strike"] - float(strike)).abs() < 0.01]
        sel = sel[sel["SEM_OPTION_TYPE"].astype(str).str.strip().str.upper() == str(opt_type).upper()]
        if sel.empty:
            return None
        return str(int(float(sel.iloc[0]["SEM_SMST_SECURITY_ID"])))
    except Exception:
        return None


def dhan_lot_size(underlying, kind, expiry_str=""):
    rows = dhan_derivative_table(underlying, kind)
    if rows.empty:
        return None
    try:
        if expiry_str:
            exp = pd.to_datetime(expiry_str).date()
            sel = rows[rows["_expiry"] == exp]
            if not sel.empty:
                rows = sel
        return int(float(rows.iloc[0]["SEM_LOT_UNITS"]))
    except Exception:
        return None


def dhan_underlying_symbol(ticker_choice, ticker):
    """Maps the app's ticker selection to a Dhan underlying symbol."""
    if ticker_choice == "Nifty50":
        return "NIFTY"
    if ticker_choice == "BankNifty":
        return "BANKNIFTY"
    if ticker_choice == "Sensex":
        return "SENSEX"
    base = str(ticker).upper()
    for suf in (".NS", ".BO"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base


def dhan_data_meta(ticker):
    """Returns {security_id, segment, instrument} for tickers Dhan CAN serve,
    or None for tickers it can't (BTC-USD, ETH-USD, USDINR=X, GC=F, SI=F, …) —
    those automatically fall back to yfinance."""
    if ticker in INDEX_SECURITY_MAP:
        return dict(INDEX_SECURITY_MAP[ticker])
    t = str(ticker).upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        exch = "NSE" if t.endswith(".NS") else "BSE"
        sid = lookup_equity_security_id(t[:-3], exch)
        if sid:
            return {"security_id": sid, "segment": f"{exch}_EQ", "instrument": "EQUITY"}
    return None


def _dhan_arrays_to_df(js):
    """Converts a Dhan charts API response (arrays of o/h/l/c/v + epoch
    timestamps) into the same OHLCV DataFrame shape yfinance returns."""
    try:
        if not js or "open" not in js or not js.get("open"):
            return pd.DataFrame()
        ts = pd.to_datetime(js.get("timestamp", []), unit="s", utc=True).tz_convert("Asia/Kolkata")
        n = len(js["open"])
        df = pd.DataFrame(
            {
                "Open": pd.to_numeric(js["open"], errors="coerce"),
                "High": pd.to_numeric(js["high"], errors="coerce"),
                "Low": pd.to_numeric(js["low"], errors="coerce"),
                "Close": pd.to_numeric(js["close"], errors="coerce"),
                "Volume": pd.to_numeric(js.get("volume", [0] * n), errors="coerce"),
            },
            index=ts,
        )
        return df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    except Exception:
        return pd.DataFrame()


def dhan_fetch_candles(meta, interval, period, client_id, token):
    """Fetches OHLCV candles from Dhan's charts API with NO artificial delay.
    Intraday requests are chunked (Dhan caps a single intraday request at ~90
    days). '1wk' is built by resampling daily candles."""
    days = PERIOD_DAYS.get(period, 30)
    to_d = datetime.now().date() + timedelta(days=1)
    from_d = to_d - timedelta(days=days + 1)
    headers = _dhan_headers(client_id, token)
    try:
        if interval in ("1d", "1wk"):
            payload = {
                "securityId": str(meta["security_id"]), "exchangeSegment": meta["segment"],
                "instrument": meta["instrument"], "expiryCode": 0, "oi": False,
                "fromDate": from_d.isoformat(), "toDate": to_d.isoformat(),
            }
            r = requests.post(f"{DHAN_API_BASE}/charts/historical", headers=headers, json=payload, timeout=20)
            df = _dhan_arrays_to_df(r.json() if r.ok else None)
            if interval == "1wk" and not df.empty:
                df = df.resample("W-FRI").agg(
                    {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                ).dropna()
            return df

        iv = DHAN_INTERVAL_MAP.get(interval)
        if iv is None:
            return pd.DataFrame()
        chunks, cur_from = [], from_d
        while cur_from < to_d:
            cur_to = min(cur_from + timedelta(days=85), to_d)
            payload = {
                "securityId": str(meta["security_id"]), "exchangeSegment": meta["segment"],
                "instrument": meta["instrument"], "interval": iv, "oi": False,
                "fromDate": cur_from.isoformat(), "toDate": cur_to.isoformat(),
            }
            r = requests.post(f"{DHAN_API_BASE}/charts/intraday", headers=headers, json=payload, timeout=20)
            part = _dhan_arrays_to_df(r.json() if r.ok else None)
            if not part.empty:
                chunks.append(part)
            cur_from = cur_to
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks)
        return df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()


def dhan_fetch_ltp(meta, client_id, token):
    """Last-traded price via Dhan's market-feed LTP endpoint (no delay)."""
    try:
        seg = meta["segment"] if meta.get("instrument") != "INDEX" else "IDX_I"
        body = {seg: [int(meta["security_id"])]}
        r = requests.post(f"{DHAN_API_BASE}/marketfeed/ltp", headers=_dhan_headers(client_id, token), json=body, timeout=8)
        if not r.ok:
            return None
        seg_data = (r.json().get("data") or {}).get(seg) or {}
        row = seg_data.get(str(meta["security_id"]))
        if row is None and seg_data:
            row = list(seg_data.values())[0]
        if row and row.get("last_price") is not None:
            return float(row["last_price"])
    except Exception:
        pass
    return None


def fetch_data(ticker, interval, period):
    """DATA-SOURCE ROUTER. When 'Use Dhan Data Feed' is enabled (and a token
    is present) fetches candles from Dhan with NO delay; any ticker Dhan can't
    serve (BTC/ETH/commodities/…) or any failed Dhan call transparently falls
    back to the original, rate-limited yfinance path."""
    cfg = st.session_state.get("app_cfg", {})
    if cfg.get("use_dhan_data") and cfg.get("dhan_access_token"):
        meta = dhan_data_meta(ticker)
        if meta is not None:
            df = dhan_fetch_candles(meta, interval, period, cfg.get("dhan_client_id", ""), cfg["dhan_access_token"])
            if df is not None and not df.empty:
                return df
    return fetch_data_yf(ticker, interval, period)


def send_email_notification(subject, body):
    """Fires an email via Gmail SMTP (app password) when the email-notification
    checkbox is enabled. Never raises — a mail failure must not break trading."""
    cfg = st.session_state.get("app_cfg", {})
    if not cfg.get("email_enabled"):
        return
    sender = str(cfg.get("email_from", "")).strip()
    to = str(cfg.get("email_to", "")).strip()
    pwd = str(cfg.get("email_app_password", "")).strip()
    if not (sender and to and pwd):
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        recipients = [a.strip() for a in to.replace(";", ",").split(",") if a.strip()]
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=15) as server:
            server.login(sender, pwd)
            server.sendmail(sender, recipients, msg.as_string())
    except Exception as exc:
        st.warning(f"📧 Email notification failed: {exc}")


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
    ltp = get_live_ltp(ticker)  # routes to Dhan (no delay) or yfinance (0.3s delay)
    if ltp is not None:
        prev = st.session_state.get("_ltp_prev")
        st.metric(label, f"{ltp:,.2f}", f"{(ltp - prev):+.2f}" if prev is not None else None)
        st.session_state["_ltp_prev"] = ltp
    else:
        st.info("No live data returned yet.")

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
        thr_dir = params.get("threshold_direction", "Below")
        if thr_dir == "Below":
            # Price must approach FROM BELOW and cross up through the level.
            df.loc[(df["Close"] > thr) & (df["Close"].shift(1) <= thr), "signal"] = 1
        else:  # "Above" — price approaches from above and crosses down.
            df.loc[(df["Close"] < thr) & (df["Close"].shift(1) >= thr), "signal"] = -1

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

    # ---- Flip / Reverse signals (Long ↔ Short) ---------------------------
    # Off by default. When ON, every long signal becomes a short entry and
    # vice-versa — applied to backtest AND live identically. For options this
    # still maps direction→leg downstream (long→CE BUY, short→PE BUY): flipped
    # signals BUY the other leg — never any option selling.
    if params.get("flip_signals", False):
        df["signal"] = -df["signal"]

    # ---- Trade Direction filter (Both / Long Only / Short Only) ----------
    # Applied centrally so it affects backtests, optimization, heatmaps AND
    # live trading identically.
    trade_type = params.get("trade_type", "Both")
    if trade_type == "Long Only":
        df.loc[df["signal"] == -1, "signal"] = 0
    elif trade_type == "Short Only":
        df.loc[df["signal"] == 1, "signal"] = 0
    return df


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


def check_hard_exit(trade, candle, close_fill=False):
    """
    Hard SL/Target check using only the CURRENT candle's own high/low against
    levels set from PAST data (entry price, ATR at signal time, trailing
    updates). No look-ahead — these levels never depend on this candle's own
    close. Conservative order: longs check SL(low) before Target(high);
    shorts check SL(high) before Target(low).

    close_fill=False (DEFAULT — original behavior): a breached level fills AT
    THE LEVEL PRICE — LONG: Low <= SL → exit at SL; High >= Target → exit at
    Target (models a resting stop/limit order sitting at the broker).

    close_fill=True ("Candle-Close SL/Target Fills" checkbox ON): a breached
    level fills at the candle's CLOSE — the honest fill for a poll-based
    system that only DISCOVERS the breach when the candle completes. P&L /
    points then include the overshoot beyond the level in both directions.
    """
    direction = trade["direction"]
    target_display_only = trade["target_type"] == "Trailing Target (Display Only)"

    if direction == 1:
        if candle["Low"] <= trade["sl"]:
            return True, (float(candle["Close"]) if close_fill else trade["sl"]), "Stoploss Hit"
        if not target_display_only and candle["High"] >= trade["target"]:
            return True, (float(candle["Close"]) if close_fill else trade["target"]), "Target Hit"
    else:
        if candle["High"] >= trade["sl"]:
            return True, (float(candle["Close"]) if close_fill else trade["sl"]), "Stoploss Hit"
        if not target_display_only and candle["Low"] <= trade["target"]:
            return True, (float(candle["Close"]) if close_fill else trade["target"]), "Target Hit"

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

    When 'Use Dhan Data Feed' is enabled, this hits Dhan's market-feed LTP
    endpoint with NO delay; otherwise (or on any Dhan failure / unsupported
    ticker like BTC/ETH) it falls back to the original rate-limited yfinance
    path — every SL/Target check automatically uses whichever tick came back.
    """
    cfg = st.session_state.get("app_cfg", {})
    if cfg.get("use_dhan_data") and cfg.get("dhan_access_token"):
        meta = dhan_data_meta(ticker)
        if meta is not None:
            ltp = dhan_fetch_ltp(meta, cfg.get("dhan_client_id", ""), cfg["dhan_access_token"])
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

    def close_trade(exit_price, exit_time, reason, qty_to_close):
        points = (exit_price - open_trade["entry_price"]) * open_trade["direction"]
        try:
            _c = df.loc[exit_time]
            _ohlc = {"Candle Open": round(float(_c["Open"]), 2), "Candle High": round(float(_c["High"]), 2),
                     "Candle Low": round(float(_c["Low"]), 2), "Candle Close": round(float(_c["Close"]), 2)}
        except Exception:
            _ohlc = {"Candle Open": None, "Candle High": None, "Candle Low": None, "Candle Close": None}
        trades.append({
            "Entry Time": open_trade["entry_time"], "Entry Price": round(open_trade["entry_price"], 2),
            "Direction": "LONG" if open_trade["direction"] == 1 else "SHORT",
            "Exit Time": exit_time, "Exit Price": round(float(exit_price), 2),
            "SL": round(open_trade["initial_sl"], 2), "Target": round(open_trade["initial_target"], 2),
            "Highest": round(open_trade["highest"], 2), "Lowest": round(open_trade["lowest"], 2),
            "Points": round(points, 2), "PnL": round(points * qty_to_close, 2),
            "Exit Reason": reason, "Qty": qty_to_close, **_ohlc,
        })

    for i in range(1, len(df) - 1):
        if open_trade is None:
            sig = df["signal"].iloc[i]
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
                hard_exit, hard_price, hard_reason = check_hard_exit(
                    open_trade, candle, close_fill=params.get("close_fill_logic", False))
                if hard_exit:
                    if (open_trade["target_type"] == "Partial Book + Trail Remainder"
                            and hard_reason == "Target Hit" and not open_trade["partial_booked"]):
                        book_qty = max(1, round(open_trade["original_qty"] * open_trade["partial_book_pct"] / 100.0))
                        book_qty = min(book_qty, open_trade["remaining_qty"])
                        partial_points = (hard_price - open_trade["entry_price"]) * open_trade["direction"]
                        _pc = df.iloc[i]
                        trades.append({
                            "Entry Time": open_trade["entry_time"], "Entry Price": round(open_trade["entry_price"], 2),
                            "Direction": "LONG" if open_trade["direction"] == 1 else "SHORT",
                            "Exit Time": df.index[i], "Exit Price": round(float(hard_price), 2),
                            "SL": round(open_trade["initial_sl"], 2), "Target": round(open_trade["initial_target"], 2),
                            "Highest": round(open_trade["highest"], 2), "Lowest": round(open_trade["lowest"], 2),
                            "Points": round(partial_points, 2), "PnL": round(partial_points * book_qty, 2),
                            "Exit Reason": f"Partial Book ({book_qty}/{open_trade['original_qty']} qty @ Target 1)",
                            "Qty": book_qty,
                            "Candle Open": round(float(_pc["Open"]), 2), "Candle High": round(float(_pc["High"]), 2),
                            "Candle Low": round(float(_pc["Low"]), 2), "Candle Close": round(float(_pc["Close"]), 2),
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
# DHAN BROKER PLACEHOLDER
# ============================================================================

def place_dhan_order(client_id, access_token, security_id, txn_type, product_cfg, qty,
                     price=0.0, order_type="MARKET", is_entry=True):
    """
    Places a REAL order via Dhan Broker API v2 (POST /v2/orders) when the Dhan
    order-placement checkbox is enabled and an access token is present.
    - Entry/exit order type (MARKET/LIMIT) comes from the config dropdowns.
    - LIMIT orders carry the reference price; MARKET orders send price 0.
    - When 'Use Broker SL/Target (Bracket Order)' is ON, ENTRY orders go out as
      productType=BO with boStopLossValue / boProfitValue (+ trailingJump when
      Trail SL Jump > 0 — 0 means trailing off).
    Without a token, returns the fully-built payload as SIMULATED_NOT_SENT so
    everything can be dry-run safely.
    """
    use_bo = bool(product_cfg.get("bo_enabled")) and is_entry
    payload = {
        "dhanClientId": str(client_id),
        "correlationId": f"algo{int(time.time() * 1000) % 10**12}",
        "transactionType": txn_type,
        "exchangeSegment": product_cfg.get("exchange_segment", "NSE_EQ"),
        "productType": "BO" if use_bo else product_cfg.get("product", "INTRADAY"),
        "orderType": order_type if order_type in ("MARKET", "LIMIT") else "MARKET",
        "validity": "DAY",
        "securityId": str(security_id),
        "quantity": int(qty),
        "disclosedQuantity": 0,
        "price": float(price) if order_type == "LIMIT" else 0.0,
        "afterMarketOrder": False,
    }
    if use_bo:
        payload["boStopLossValue"] = float(product_cfg.get("bo_sl", 0.0))
        payload["boProfitValue"] = float(product_cfg.get("bo_target", 0.0))
        trail = float(product_cfg.get("bo_trail", 0.0))
        if trail > 0:
            payload["trailingJump"] = trail  # 0 = trailing off (field omitted)
    if not access_token:
        return {"status": "SIMULATED_NOT_SENT", "payload": payload,
                "note": "No Dhan access token entered — order was built but NOT sent."}
    try:
        resp = requests.post(f"{DHAN_API_BASE}/orders",
                             headers=_dhan_headers(client_id, access_token),
                             json=payload, timeout=12)
        try:
            body = resp.json()
        except ValueError:
            body = {"raw": resp.text}
        return {"status": "SENT", "http_status": resp.status_code, "request": payload, "response": body}
    except Exception as exc:
        return {"status": "ERROR", "error": str(exc), "payload": payload}


def dhan_order_qty(product_cfg):
    """Quantity to send on Dhan orders (the 'Dhan Quantity' input — independent
    of the paper-trading Quantity in the main config)."""
    try:
        return max(1, int(product_cfg.get("dhan_qty", 1)))
    except (TypeError, ValueError):
        return 1


def dhan_should_send_exit(product_cfg, reason):
    """When a Bracket Order manages SL/Target at the broker, the broker's own
    legs close the position on SL/Target — sending our own exit order too would
    double-exit. Manual square-offs / signal exits are still sent."""
    if product_cfg.get("bo_enabled") and reason and ("Stoploss Hit" in str(reason) or "Target Hit" in str(reason)):
        return False
    return True


def _now_ist():
    try:
        return pd.Timestamp.now(tz="Asia/Kolkata")
    except Exception:
        return pd.Timestamp.now()


def _day_realized_points():
    """Sum of realized Points across today's closed rows in live history
    (partial-book rows included — they're realized points too)."""
    today = _now_ist().date()
    total = 0.0
    for h in st.session_state.live_history:
        try:
            et = pd.Timestamp(h.get("Exit Time"))
            et_date = et.date()
        except Exception:
            continue
        if et_date == today:
            total += float(h.get("Points", 0.0) or 0.0)
    return total


def _entries_today():
    stats = st.session_state.get("live_day_stats") or {}
    if stats.get("date") != _now_ist().date():
        return 0
    return int(stats.get("entries", 0))


def _record_entry_taken():
    today = _now_ist().date()
    stats = st.session_state.get("live_day_stats") or {}
    if stats.get("date") != today:
        stats = {"date": today, "entries": 0}
    stats["entries"] = int(stats.get("entries", 0)) + 1
    st.session_state.live_day_stats = stats
    st.session_state.last_trade_event_ts = time.time()


def _is_indian_ticker(ticker):
    t = str(ticker).upper()
    return t.endswith(".NS") or t.endswith(".BO") or t in ("^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX")


def check_live_entry_gates(cfg_live, ticker):
    """All the pre-entry risk gates. Returns (allowed: bool, reason: str).
    Evaluated fresh on every live cycle so toggling a gate applies instantly."""
    # Max loss in day
    if cfg_live.get("max_loss_day_enabled"):
        day_pts = _day_realized_points()
        if day_pts <= -abs(float(cfg_live.get("max_loss_day_points", 20.0))):
            return False, f"Max loss/day hit ({day_pts:+.1f} pts) — no new entries today."
    # Max profit in day
    if cfg_live.get("max_profit_day_enabled"):
        day_pts = _day_realized_points()
        if day_pts >= abs(float(cfg_live.get("max_profit_day_points", 100.0))):
            return False, f"Max profit/day reached ({day_pts:+.1f} pts) — locking in gains, no new entries today."
    # Max number of trades
    if cfg_live.get("max_trades_enabled"):
        n = _entries_today()
        limit = int(cfg_live.get("max_trades_day", 10))
        if n >= limit:
            return False, f"Max trades/day reached ({n}/{limit}) — no new entries today."
    # Trade window (IST) — only enforced for Indian tickers
    if cfg_live.get("trade_window_enabled") and _is_indian_ticker(ticker):
        now_t = _now_ist().time()
        start_t = cfg_live.get("trade_window_start") or dt_time(9, 15)
        end_t = cfg_live.get("trade_window_end") or dt_time(15, 30)
        if not (start_t <= now_t <= end_t):
            return False, f"Outside trade window ({start_t.strftime('%H:%M')}–{end_t.strftime('%H:%M')} IST) — entries paused."
    # Entry cooldown after the previous entry/exit
    if cfg_live.get("entry_cooldown_enabled"):
        last_ts = st.session_state.get("last_trade_event_ts")
        cd = float(cfg_live.get("entry_cooldown_seconds", 1.0))
        if last_ts is not None and (time.time() - last_ts) < cd:
            return False, f"Entry cooldown active ({cd:.0f}s after the last trade event)."
    return True, ""


def resolve_dhan_order_leg(direction, is_entry, fallback_ticker, product_cfg):
    """
    Decides WHICH instrument to trade and which side (BUY/SELL) to send.

    Options instruments: a LONG signal BUYs the CE leg, a SHORT signal BUYs the
    PE leg — both are entered by BUYING (not selling) an option, which keeps
    risk defined (no naked option writing baked into this default). Exiting
    always SELLs whichever leg is currently open. CE/PE security IDs are
    resolved automatically from the scrip master (expiry + strike + type).

    Stocks / futures: uses the configured (auto-fetched) security ID — BUY to
    open long / SELL to close it, SELL to open short / BUY to close it.
    """
    inst = str(product_cfg.get("instrument", ""))
    if "Options" in inst:
        security_id = product_cfg.get("ce_security_id") if direction == 1 else product_cfg.get("pe_security_id")
        txn_type = "BUY" if is_entry else "SELL"
        return (security_id or fallback_ticker), txn_type

    security_id = product_cfg.get("security_id") or fallback_ticker
    if is_entry:
        txn_type = "BUY" if direction == 1 else "SELL"
    else:
        txn_type = "SELL" if direction == 1 else "BUY"
    return security_id, txn_type


# ============================================================================
# CONFIGURATION PANEL — a SINGLE definition rendered in TWO places: the
# sidebar AND the "🛠 Admin Panel" tab. Every widget is two-way synced through
# st.session_state.app_cfg, so changing a value in either place instantly
# updates the other (and everything downstream) — exactly the same controls,
# exactly the same behavior.
# ============================================================================

def _cfg_store(wkey, name):
    st.session_state.app_cfg[name] = st.session_state[wkey]


def _seed_widget(prefix, name, value):
    """Force the widget's session value from the central config store BEFORE
    the widget is instantiated — this is what keeps the sidebar copy and the
    Admin Panel copy of every control perfectly in sync. Only writes when the
    value actually differs: rewriting identical widget state on every rerun
    creates needless state churn that can make button clicks feel 'lost'."""
    wkey = f"{prefix}::{name}"
    if wkey not in st.session_state or st.session_state[wkey] != value:
        st.session_state[wkey] = value
    return wkey


def cfg_checkbox(ui, prefix, name, label, default=False, help=None):
    cur = bool(st.session_state.app_cfg.get(name, default))
    st.session_state.app_cfg[name] = cur
    wkey = _seed_widget(prefix, name, cur)
    return ui.checkbox(label, key=wkey, on_change=_cfg_store, args=(wkey, name), help=help)


def cfg_selectbox(ui, prefix, name, label, options, default=None, help=None):
    options = list(options)
    if not options:
        options = [""]
    cur = st.session_state.app_cfg.get(name, default if default is not None else options[0])
    if cur not in options:
        cur = default if default in options else options[0]
    # Write the (possibly coerced) value back so the store NEVER disagrees
    # with what the widget shows — a store/widget mismatch after an option
    # list changed was the "dropdown change not reflecting" bug.
    st.session_state.app_cfg[name] = cur
    wkey = _seed_widget(prefix, name, cur)
    return ui.selectbox(label, options, key=wkey, on_change=_cfg_store, args=(wkey, name), help=help)


def cfg_number(ui, prefix, name, label, min_value=None, max_value=None, default=0.0, step=None, help=None):
    as_float = any(isinstance(x, float) for x in (min_value, max_value, step, default))
    cur = st.session_state.app_cfg.get(name, default)
    try:
        cur = float(cur) if as_float else int(cur)
    except (TypeError, ValueError):
        cur = default
    if min_value is not None:
        cur = max(cur, min_value)
    if max_value is not None:
        cur = min(cur, max_value)
    st.session_state.app_cfg[name] = cur  # keep store == widget after cast/clamp
    wkey = _seed_widget(prefix, name, cur)
    return ui.number_input(label, min_value=min_value, max_value=max_value, step=step,
                           key=wkey, on_change=_cfg_store, args=(wkey, name), help=help)


def cfg_text(ui, prefix, name, label, default="", help=None, password=False):
    cur = str(st.session_state.app_cfg.get(name, default))
    st.session_state.app_cfg[name] = cur
    wkey = _seed_widget(prefix, name, cur)
    kwargs = dict(key=wkey, on_change=_cfg_store, args=(wkey, name), help=help)
    if password:
        kwargs["type"] = "password"
    return ui.text_input(label, **kwargs)


def cfg_slider(ui, prefix, name, label, min_value, max_value, default):
    try:
        cur = int(st.session_state.app_cfg.get(name, default))
    except (TypeError, ValueError):
        cur = default
    cur = min(max(cur, min_value), max_value)
    st.session_state.app_cfg[name] = cur
    wkey = _seed_widget(prefix, name, cur)
    return ui.slider(label, min_value, max_value, key=wkey, on_change=_cfg_store, args=(wkey, name))


def cfg_time(ui, prefix, name, label, default, help=None):
    cur = st.session_state.app_cfg.get(name, default)
    if not isinstance(cur, dt_time):
        cur = default
    st.session_state.app_cfg[name] = cur
    wkey = _seed_widget(prefix, name, cur)
    return ui.time_input(label, key=wkey, on_change=_cfg_store, args=(wkey, name), help=help)


def auto_populate_dhan_defaults():
    """Auto-fetches Dhan defaults whenever the (ticker, instrument, exchange,
    expiry, strikes) combination changes: security IDs from the scrip master
    (incl. CE/PE option legs), nearest F&O expiries, ATM strikes from live LTP,
    and default quantities (65 Nifty / 35 BankNifty / 20 Sensex for index
    options; contract lot size otherwise). Runs only when the Dhan DATA feed is
    on — with yfinance as data source, the fields stay manual. On a fetch
    failure it retries (throttled to every 20s) instead of giving up, and a
    ticker change always forces a refill so IDs can't go stale."""
    cfg = st.session_state.app_cfg
    if not (cfg.get("dhan_enabled") or cfg.get("use_dhan_data")):
        return
    tkr_choice = cfg.get("ticker_choice", "Nifty50")
    tkr = TICKER_MAP.get(tkr_choice) if tkr_choice != "Custom" else None
    if tkr is None:
        tkr = cfg.get("custom_ticker", "KAYNES.NS")
    inst = cfg.get("dhan_instrument", "Stock Intraday")

    # Exchange auto-defaults to NSE; flips to BSE automatically when a BSE
    # ticker (Sensex / *.BO) is picked. Always user-overridable via the
    # Exchange dropdown afterwards.
    tkr_sig = (tkr_choice, tkr)
    if cfg.get("_dhan_exch_auto_for") != tkr_sig:
        cfg["dhan_exchange"] = "BSE" if (tkr_choice == "Sensex" or str(tkr).upper().endswith(".BO")) else "NSE"
        cfg["_dhan_exch_auto_for"] = tkr_sig

    # Manual mode note: this function is only reached when the Dhan data feed
    # or Dhan order placement is enabled — pure-yfinance setups keep all
    # security-ID fields fully manual (the scrip master is public, no token
    # needed for the lookups below).

    underlying = dhan_underlying_symbol(tkr_choice, tkr)
    exch = cfg.get("dhan_exchange", "NSE")
    base_sig = (tkr_choice, tkr, inst, exch)
    full_sig = base_sig + (cfg.get("dhan_opt_expiry", ""),
                           cfg.get("dhan_ce_strike", 0), cfg.get("dhan_pe_strike", 0))
    if cfg.get("_dhan_auto_sig") == full_sig:
        return
    # Throttle retries after a failure so a broken network can't freeze the UI
    # on every rerun (this was one cause of buttons needing multiple clicks).
    last_try = cfg.get("_dhan_auto_last_try", 0.0)
    if cfg.get("_dhan_auto_failed") and (time.time() - last_try) < 20:
        return
    cfg["_dhan_auto_last_try"] = time.time()
    ok = True
    try:
        if inst in ("Stock Intraday", "Stock Delivery"):
            sid = lookup_equity_security_id(underlying, exch)
            if sid:
                # A ticker/instrument/exchange change ALWAYS overwrites — stale
                # IDs from the previous ticker were the "sometimes not filled"
                # bug. Only a user edit on the SAME signature is preserved.
                if cfg.get("_dhan_sid_auto_for") != base_sig or not cfg.get("dhan_security_id"):
                    cfg["dhan_security_id"] = sid
                    cfg["_dhan_sid_auto_for"] = base_sig
            else:
                ok = False
        elif inst in ("Stock Futures", "Index Futures"):
            expiries = dhan_expiry_list(underlying, "FUT")
            cfg["_dhan_fut_expiries"] = expiries
            if expiries and cfg.get("dhan_expiry") not in expiries:
                cfg["dhan_expiry"] = expiries[0]
            sid = lookup_future_security_id(underlying, cfg.get("dhan_expiry", ""))
            if sid:
                cfg["dhan_security_id"] = sid
                cfg["_dhan_sid_auto_for"] = base_sig
            else:
                ok = False
            if cfg.get("_dhan_qty_auto_for") != base_sig:
                lot = dhan_lot_size(underlying, "FUT", cfg.get("dhan_expiry", ""))
                if lot:
                    cfg["dhan_qty"] = lot
                cfg["_dhan_qty_auto_for"] = base_sig
        else:  # Stock Options / Index Options
            expiries = dhan_expiry_list(underlying, "OPT")
            cfg["_dhan_opt_expiries"] = expiries
            if expiries and cfg.get("dhan_opt_expiry") not in expiries:
                cfg["dhan_opt_expiry"] = expiries[0]
            strikes = dhan_option_strikes(underlying, cfg.get("dhan_opt_expiry", ""))
            if cfg.get("_dhan_strike_auto_for") != base_sig:
                ltp = get_live_ltp(tkr)
                if strikes and ltp is not None:
                    atm = float(min(strikes, key=lambda s: abs(s - ltp)))
                    cfg["dhan_ce_strike"] = atm
                    cfg["dhan_pe_strike"] = atm
                    cfg["_dhan_strike_auto_for"] = base_sig
                else:
                    ok = False
            # Resolve the CE/PE SECURITY IDs for the current expiry+strikes —
            # refreshed whenever expiry or either strike changes.
            leg_sig = (underlying, cfg.get("dhan_opt_expiry", ""),
                       cfg.get("dhan_ce_strike", 0), cfg.get("dhan_pe_strike", 0))
            if cfg.get("_dhan_optsid_auto_for") != leg_sig:
                ce_id = lookup_option_security_id(underlying, cfg.get("dhan_opt_expiry", ""),
                                                  cfg.get("dhan_ce_strike", 0), "CE")
                pe_id = lookup_option_security_id(underlying, cfg.get("dhan_opt_expiry", ""),
                                                  cfg.get("dhan_pe_strike", 0), "PE")
                if ce_id:
                    cfg["dhan_ce_security_id"] = ce_id
                if pe_id:
                    cfg["dhan_pe_security_id"] = pe_id
                if ce_id and pe_id:
                    cfg["_dhan_optsid_auto_for"] = leg_sig
                else:
                    ok = False
            if cfg.get("_dhan_qty_auto_for") != base_sig:
                default_lot = INDEX_OPTION_LOT_DEFAULTS.get(underlying)
                if not default_lot:
                    default_lot = dhan_lot_size(underlying, "OPT", cfg.get("dhan_opt_expiry", "")) or 1
                cfg["dhan_qty"] = default_lot
                cfg["_dhan_qty_auto_for"] = base_sig
    except Exception:
        ok = False
    cfg["_dhan_auto_failed"] = not ok
    if ok:
        cfg["_dhan_auto_sig"] = full_sig  # only lock in on SUCCESS — failures retry


def render_config_panel(ui, prefix):
    """Renders EVERY configuration control into `ui` (st.sidebar or the Admin
    Panel tab). `prefix` keeps the two copies' widget keys distinct while the
    shared app_cfg store keeps their VALUES identical."""
    cfg = st.session_state.app_cfg

    # ------------------------------------------------------------ Data source
    ui.markdown("### 📡 Data Source")
    use_dhan_data = cfg_checkbox(
        ui, prefix, "use_dhan_data", "Use Dhan Data Feed (instead of yfinance)", default=False,
        help="yfinance keeps its mandatory 0.3s delay per call (API rate limits). "
             "Dhan has no such limit, so the Dhan path applies NO delay at all.",
    )
    if use_dhan_data:
        ui.caption("⚡ Dhan feed active — zero delay. Tickers Dhan can't serve (BTC-USD, ETH-USD, "
                   "USDINR, Gold/Silver futures, …) automatically fall back to yfinance.")
    else:
        ui.caption("Default: yfinance with a fixed 0.3s delay per API call (rate-limit protection).")

    dhan_creds_needed = use_dhan_data or cfg.get("dhan_enabled", False)
    if dhan_creds_needed:
        ui.markdown("**🔐 Dhan Account** (shared by the data feed and order placement)")
        cfg_text(ui, prefix, "dhan_client_id", "Dhan Client ID", default="1104779876")
        cfg_text(ui, prefix, "dhan_access_token", "Dhan Access Token", default="", password=True)
        if use_dhan_data and not cfg.get("dhan_access_token"):
            ui.warning("Enter a Dhan access token — without it the app silently keeps using yfinance.")

    # -------------------------------------------------------------- Instrument
    ui.markdown("### 🎯 Instrument & Data")
    ticker_choice = cfg_selectbox(ui, prefix, "ticker_choice", "Ticker", list(TICKER_MAP.keys()), default="Nifty50")
    if ticker_choice == "Custom":
        cfg_text(ui, prefix, "custom_ticker", "Custom Ticker (Yahoo Finance symbol)", default="KAYNES.NS")

    interval = cfg_selectbox(ui, prefix, "interval", "Timeframe", list(TF_PERIOD_MAP.keys()), default="1m")
    periods_available = TF_PERIOD_MAP[interval]
    cfg_selectbox(ui, prefix, "period", "Period", periods_available,
                  default="7d" if "7d" in periods_available else periods_available[0])
    cfg_number(ui, prefix, "qty", "Quantity", min_value=1, default=1, step=1)

    # ---------------------------------------------------------------- Strategy
    ui.markdown("### 📐 Strategy")
    strategy = cfg_selectbox(ui, prefix, "strategy", "Strategy", STRATEGIES)
    if strategy in IMMEDIATE_EXECUTION_STRATEGIES:
        ui.caption("⚡ This is a price condition, not a candle strategy — in LIVE trading it enters IMMEDIATELY "
                   "at LTP the moment the condition is met (no waiting for the next candle open).")

    if strategy in ("EMA Crossover", "Pro: EMA50 Trend + EMA9/15 Pullback"):
        cfg_number(ui, prefix, "ema_fast", "EMA Fast", 2, 100, 9)
        cfg_number(ui, prefix, "ema_slow", "EMA Slow", 3, 200, 15)
    if strategy == "Threshold Cross":
        cfg_number(ui, prefix, "threshold", "Threshold Price", default=0.0)
        thr_dir = cfg_selectbox(ui, prefix, "threshold_direction", "Cross Direction", ["Below", "Above"], default="Below",
                                help="Below: price must approach the threshold FROM BELOW and cross UP through it (long). "
                                     "Above: price approaches from above and crosses DOWN through it (short).")
        ui.caption("↗️ Crossing UP from below → LONG entry." if thr_dir == "Below"
                   else "↘️ Crossing DOWN from above → SHORT entry.")
    if strategy == "Price Action Support/Resistance":
        cfg_number(ui, prefix, "sr_window", "S/R Lookback", 5, 200, 20)
    if strategy == "Liquidity Grab Reversal":
        cfg_number(ui, prefix, "liq_window", "Liquidity Lookback", 5, 200, 20)
    if strategy == "RSI Cross":
        cfg_number(ui, prefix, "rsi_period", "RSI Period", 2, 50, 14)
    if strategy in ("Bollinger Bands", "Pro: BB+RSI Mean Reversion (ATR filtered)"):
        cfg_number(ui, prefix, "bb_period", "BB Period", 5, 100, 20)
        cfg_number(ui, prefix, "bb_std", "BB Std Dev", 1.0, 4.0, 2.0)
    if strategy == "Volume Breakout":
        cfg_number(ui, prefix, "vol_window", "Volume Lookback", 5, 100, 20)
        cfg_number(ui, prefix, "vol_factor", "Volume Spike Factor", 1.0, 5.0, 2.0)
    if strategy == "Elliott Wave (Zigzag)":
        cfg_number(ui, prefix, "zigzag_lookback", "Zigzag Lookback", 2, 20, 3)
    if strategy == "Pro: VWAP + Supertrend Trend":
        cfg_number(ui, prefix, "st_period", "Supertrend Period", 5, 50, 10)
        cfg_number(ui, prefix, "st_mult", "Supertrend Multiplier", 1.0, 6.0, 3.0)
    if strategy == "Pro: Opening Range Breakout + Volume":
        cfg_number(ui, prefix, "orb_candles", "ORB Candles", 1, 30, 5)
    if strategy == "Pro: MACD Crossover":
        c1, c2, c3 = ui.columns(3)
        cfg_number(c1, prefix, "macd_fast", "MACD Fast", 2, 50, 12)
        cfg_number(c2, prefix, "macd_slow", "MACD Slow", 5, 100, 26)
        cfg_number(c3, prefix, "macd_signal", "MACD Signal", 2, 30, 9)
    if strategy == "Pro: Donchian Channel Breakout":
        cfg_number(ui, prefix, "donchian_period", "Donchian Period", 5, 100, 20)
    if strategy == "Pro: Keltner Squeeze Breakout":
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "keltner_period", "Keltner Period", 5, 50, 20)
        cfg_number(c2, prefix, "keltner_atr_mult", "Keltner ATR Mult", 0.5, 4.0, 1.5)
    if strategy == "Pro: Stochastic Reversal":
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "stoch_k", "Stochastic %K Period", 2, 50, 14)
        cfg_number(c2, prefix, "stoch_d", "Stochastic %D Period", 2, 20, 3)
    if strategy == "Pro: TEMA Trend Flip":
        cfg_number(ui, prefix, "tema_period", "TEMA Period", 5, 100, 20)
    if strategy == "Pro: CCI Extreme Reversal":
        cfg_number(ui, prefix, "cci_period", "CCI Period", 5, 100, 20)

    if strategy in PRO_STRATEGIES:
        ui.caption("💡 Professional-grade composite strategy (trend/volatility/liquidity confluence). Not a guarantee of profitability — validate in the Optimization tab first.")

    # ------------------------------------------------------- Trade direction
    ui.markdown("### 🎚 Trade Direction")
    tt_sel = cfg_selectbox(ui, prefix, "trade_type", "Trade Direction", ["Both", "Long Only", "Short Only"], default="Both",
                           help="Both (default): take long and short signals. Long Only / Short Only: signals in the "
                                "other direction are ignored everywhere — backtest, optimization, heatmaps AND live trading.")
    flip_on = cfg_checkbox(ui, prefix, "flip_signals", "Flip / Reverse Entries (Long ↔ Short)", default=False,
                           help="When ON: a long signal enters SHORT, a short signal enters LONG — everywhere (backtest, "
                                "optimization, live). Useful to trade the inverse of a consistently losing signal.")
    if flip_on:
        ui.caption("🔄 Signals are REVERSED (flip happens first, then the Trade Direction filter above is applied). "
                   "Stocks/futures: flipped long → SELL entry, flipped short → BUY entry. Options: flipped signals "
                   "just BUY the other leg (flipped long → PE BUY, flipped short → CE BUY) — options are always "
                   "BOUGHT, never sold, in every mode.")

    # ---------------------------------------------------------------- Stoploss
    ui.markdown("### 🛑 Stoploss")
    sl_type = cfg_selectbox(ui, prefix, "sl_type", "Stoploss Type", SL_TYPES)
    if sl_type not in ("ATR Based SL", "Loss Recovery SL (Give-back)"):
        cfg_number(ui, prefix, "sl_points", "SL Points", 0.1, 100000.0, 10.0,
                   help="The stoploss distance in points for points-based SL types, and the INITIAL SL distance "
                        "for trailing SL types (the trail then takes over).")
    if sl_type == "ATR Based SL":
        cfg_number(ui, prefix, "atr_mult_sl", "ATR Multiplier (SL)", 0.5, 5.0, 1.5)
    if sl_type == "Loss Recovery SL (Give-back)":
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "loss_trigger_points", "Loss trigger (points)", 1.0, 100000.0, 20.0)
        cfg_number(c2, prefix, "min_recovery_pct", "Min recovery required (%)", 1.0, 100.0, 50.0)
        ui.caption(f"Once floating loss reaches {cfg.get('loss_trigger_points', 20.0):.0f} pts, exit if price hasn't recovered at least {cfg.get('min_recovery_pct', 50.0):.0f}% of that loss back toward entry.")

    # ------------------------------------------------------------------ Target
    ui.markdown("### 🎯 Target")
    target_type = cfg_selectbox(ui, prefix, "target_type", "Target Type", TARGET_TYPES)
    if target_type not in ("ATR Based Target", "Risk:Reward Based (min 1:2)", "Autopilot Target",
                           "Profit Giveback Target", "Partial Book + Trail Remainder"):
        cfg_number(ui, prefix, "target_points", "Target Points", 0.1, 200000.0, 20.0,
                   help="The target distance in points for points-based target types, and the INITIAL target "
                        "distance for trailing target types. Hidden for target types that compute their own "
                        "level (ATR / R:R / Autopilot / Giveback / Partial Book).")
    if target_type == "ATR Based Target":
        cfg_number(ui, prefix, "atr_mult_target", "ATR Multiplier (Target)", 1.0, 8.0, 3.0)
    if target_type == "Profit Giveback Target":
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "profit_trigger_points", "Profit trigger (points)", 1.0, 100000.0, 50.0)
        cfg_number(c2, prefix, "giveback_pct", "Max giveback allowed (%)", 1.0, 100.0, 30.0)
        ui.caption(f"Once floating profit peaks at ≥{cfg.get('profit_trigger_points', 50.0):.0f} pts, exit if it falls back by more than {cfg.get('giveback_pct', 30.0):.0f}% from that peak.")
    if target_type == "Partial Book + Trail Remainder":
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "partial_target1_points", "Target 1 (points)", 0.1, 200000.0, 20.0)
        cfg_number(c2, prefix, "partial_book_pct", "Qty % to book at Target 1", 1.0, 99.0, 50.0)
        ui.caption(
            f"Books {cfg.get('partial_book_pct', 50.0):.0f}% of quantity when Target 1 ({cfg.get('partial_target1_points', 20.0):.0f} pts) is hit; "
            "the remainder keeps running under an ATR trailing stop with no fixed second target. "
            "⚠️ With Quantity = 1, there's nothing left to trail after rounding — increase Quantity to actually see partial-booking behavior."
        )
    if sl_type == "Risk:Reward Based (min 1:2)" or target_type == "Risk:Reward Based (min 1:2)":
        cfg_number(ui, prefix, "rr_ratio", "Risk:Reward Ratio (min 2)", 2.0, 10.0, 2.0)

    # -------------------------------------------------- SL/Target fill logic
    ui.markdown("### 🧮 SL/Target Fill Logic (Backtest + Live)")
    close_fill = cfg_checkbox(
        ui, prefix, "close_fill_logic", "Candle-Close SL/Target Fills (wait for candle close)", default=False,
        help="OFF (default — original behavior): BACKTEST — LONG checks SL vs candle Low first then Target vs High "
             "(SHORT: SL vs High then Target vs Low) and a breached level fills AT THE SL/TARGET PRICE; "
             "LIVE — SL is checked against the LTP first, then Target against the LTP, exits at the observed price. "
             "ON: both engines wait for the candle to CLOSE, run the same Low/High checks, and a breached SL or "
             "Target fills at the candle's CLOSING price — P&L and points are then calculated from that close.",
    )
    if close_fill:
        ui.caption("🕯️ ON — Backtest AND live: signal on candle N → entry at candle N+1 open; LONG → SL vs candle "
                   "Low first then Target vs candle High (SHORT: SL vs High then Target vs Low); when either level "
                   "is breached, the exit fills at that candle's CLOSE and P&L/points use the close.")
    else:
        ui.caption("⚡ OFF (default) — original logic. Backtest: same Low/High checks, but fills AT the SL/Target "
                   "price. Live: SL checked against LTP first, then Target against LTP; exits at the observed price.")

    # -------------------------------------------------------------- Time risk
    ui.markdown("### ⏱ Time-Based Risk Control")
    if cfg_checkbox(ui, prefix, "loss_duration_enabled", "Loss Holding Duration Exit", default=False):
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "loss_duration_min_minutes", "Min minutes in loss before acting", min_value=0.0, default=1.0, step=1.0)
        cfg_number(c2, prefix, "loss_duration_max_minutes", "Safety ceiling (minutes)", min_value=0.0, default=5.0, step=1.0)
        ui.caption(
            "Exits as soon as the position has been continuously in a floating loss for at least the first number "
            "of minutes. The second number is just an upper safety bound (mainly relevant to live polling delays) — "
            "keep it ≥ the first. No cap is applied to how high you can set either value."
        )

    # ------------------------------------------------ Daily limits & timing
    ui.markdown("### 🛡 Daily Risk Limits & Trade Timing (LIVE)")
    ui.caption("All off by default. These gates apply to LIVE trading entries/exits (realized points from today's closed trades).")
    if cfg_checkbox(ui, prefix, "max_loss_day_enabled", "Max Points Loss in a Day", default=False):
        cfg_number(ui, prefix, "max_loss_day_points", "Max points loss", min_value=1.0, default=20.0, step=5.0)
        ui.caption("Once today's realized points reach −this value, no new entries are taken for the rest of the day.")
    if cfg_checkbox(ui, prefix, "max_profit_day_enabled", "Max Points Profit in a Day", default=False):
        cfg_number(ui, prefix, "max_profit_day_points", "Max points profit", min_value=1.0, default=100.0, step=10.0)
        ui.caption("Once today's realized points reach +this value, trading stops for the day to lock in gains.")
    if cfg_checkbox(ui, prefix, "max_trades_enabled", "Max Number of Trades in a Day", default=False):
        cfg_number(ui, prefix, "max_trades_day", "Max trades", min_value=1, default=10, step=1)
    if cfg_checkbox(ui, prefix, "max_profit_hold_enabled", "Max Hold Duration of Profitable Trade", default=False):
        cfg_number(ui, prefix, "max_profit_hold_minutes", "Max hold while in profit (minutes)", min_value=0.1, default=1.0, step=0.5)
        ui.caption("If the open position has been held at least this long AND is currently in profit, it's booked immediately.")
    if cfg_checkbox(ui, prefix, "trade_window_enabled", "Trade Window (IST — Indian tickers only)", default=False):
        c1, c2 = ui.columns(2)
        cfg_time(c1, prefix, "trade_window_start", "Start time (IST)", dt_time(9, 15))
        cfg_time(c2, prefix, "trade_window_end", "End time (IST)", dt_time(15, 30))
        ui.caption("New entries are only taken between these times (IST). Enforced for Indian tickers "
                   "(.NS / .BO / Nifty / BankNifty / Sensex); non-Indian tickers like BTC-USD ignore it.")
    if cfg_checkbox(ui, prefix, "entry_cooldown_enabled", "Enable Entry Cooldown", default=False):
        cfg_number(ui, prefix, "entry_cooldown_seconds", "Cooldown (seconds)", min_value=0.1, default=1.0, step=0.5)
        ui.caption("After any entry/exit event, new entries are blocked for this many seconds.")

    # ----------------------------------------------------------------- Filters
    ui.markdown("### 🔍 Additional Entry Filters")
    if cfg_checkbox(ui, prefix, "adx_enabled", "ADX Filter", default=False):
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "adx_min", "ADX Min", 0, 100, 20)
        cfg_number(c2, prefix, "adx_max", "ADX Max", 0, 100, 100)
    cfg_checkbox(ui, prefix, "rsi_enabled", "RSI Filter (30 up-cross buy / 70 down-cross sell)", default=False)
    cfg_checkbox(ui, prefix, "bb_enabled", "Bollinger Band Filter", default=False)
    cfg_checkbox(ui, prefix, "ema20_enabled", "EMA20 Filter", default=False)
    cfg_checkbox(ui, prefix, "sma20_enabled", "SMA20 Filter", default=False)
    cfg_checkbox(ui, prefix, "smc_enabled", "SMC (Structure Break) Filter", default=False)

    if cfg_checkbox(ui, prefix, "atr_enabled", "ATR (Volatility) Filter", default=False):
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "atr_min", "ATR Min (points)", 0.0, 100000.0, 0.0)
        cfg_number(c2, prefix, "atr_max", "ATR Max (points)", 0.0, 100000.0, 100000.0)
        ui.caption("Only trade when 14-period ATR is inside this band — avoids dead/illiquid tape and blow-off volatility spikes.")

    if cfg_checkbox(ui, prefix, "supertrend_enabled", "Supertrend Filter", default=False):
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "st_filter_period", "Supertrend Period (filter)", 5, 50, 10)
        cfg_number(c2, prefix, "st_filter_mult", "Supertrend Mult (filter)", 1.0, 6.0, 3.0)
        ui.caption("Only takes buys when Supertrend is bullish, sells when Supertrend is bearish — independent of the main strategy.")

    if cfg_checkbox(ui, prefix, "regime_enabled", "Regime Filter (Trend vs Range, adaptive)", default=False):
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "regime_trend_min", "ADX ≥ this = Trending", 10, 60, 25)
        cfg_number(c2, prefix, "regime_range_max", "ADX ≤ this = Ranging", 5, 40, 20)
        ui.caption(
            "Trend-type strategies (EMA/Supertrend/ORB/S-R/EW) only fire when ADX confirms a trend; "
            "mean-reversion strategies (RSI/Bollinger/Liquidity/BB+RSI) only fire when ADX confirms a range. "
            "This is the 'adapt to changing market regime' control — it doesn't switch strategies for you, "
            "it stops your chosen strategy from firing in the regime it's known to perform badly in."
        )

    if cfg_checkbox(ui, prefix, "angle_enabled", "Angle of Crossover Filter", default=False):
        cfg_number(ui, prefix, "angle_min_deg", "Minimum crossover angle (degrees, absolute value)", min_value=0.0, default=0.0, step=1.0)
        ui.caption(
            f"Only accepts an EMA{cfg.get('ema_fast', 9)}/{cfg.get('ema_slow', 15)} crossover if it's steep enough. "
            "Angle is normalized against ATR (there's no universal 'degrees' for a raw price slope), so treat it as a "
            "relative steepness score, not a standardized industry figure. Absolute value is used since valid crosses "
            "can produce a negative raw slope depending on direction."
        )

    if cfg_checkbox(ui, prefix, "crossover_quality_enabled", "Crossover Confirmation Filter", default=False):
        mode = cfg_selectbox(ui, prefix, "crossover_quality_mode", "Confirmation type",
                             ["Simple Crossover", "Crossover with Candle Size", "Crossover with ATR-based Candle Size"])
        if mode == "Crossover with Candle Size":
            cfg_number(ui, prefix, "crossover_min_points", "Min candle range (points)", min_value=0.0, default=1.0, step=0.5)
        elif mode == "Crossover with ATR-based Candle Size":
            cfg_number(ui, prefix, "crossover_atr_mult", "Min candle range (× ATR)", min_value=0.1, default=1.0, step=0.1)
        ui.caption(f"Only accepts an EMA{cfg.get('ema_fast', 9)}/{cfg.get('ema_slow', 15)} crossover bar that also clears this candle-size bar — filters out crosses on tiny, indecisive candles.")

    if cfg_checkbox(ui, prefix, "vix_enabled", "India VIX Filter", default=False):
        c1, c2 = ui.columns(2)
        cfg_number(c1, prefix, "vix_min", "VIX Min", 0.0, 100.0, 10.0)
        cfg_number(c2, prefix, "vix_max", "VIX Max", 0.0, 100.0, 25.0)
        ui.caption(
            "India VIX is a fear/expected-volatility gauge, not a price indicator — you don't need to be an expert to "
            "use it as a simple filter here. Rough rule of thumb: below ~15 = calm (often better for trend-following), "
            "15–20 = normal, 20–30 = elevated/nervous (often better for mean-reversion or smaller size), above ~30 = "
            "panic (many systems sit out entirely). Defaults above (10–25) are a conservative 'avoid extremes' band — "
            "adjust to taste. VIX only publishes daily, so intraday timeframes reuse the latest known daily value."
        )

    # -------------------------------------------------------------- Smart eval
    ui.markdown("### 🧠 Smart Evaluation (Recommended Before Going Live)")
    ui.caption("Off by default. Turn these on to get a more honest read on whether a config is likely to hold up out-of-sample and after real costs.")

    if cfg_checkbox(ui, prefix, "wf_enabled", "Enable Walk-Forward Validation", default=False):
        cfg_slider(ui, prefix, "wf_folds", "Number of sequential out-of-sample folds", 3, 20, 5)
        ui.caption("Splits the backtest period into N sequential chunks and checks whether the edge holds up across most of them, not just in aggregate.")

    if cfg_checkbox(ui, prefix, "cost_enabled", "Enable Realistic Cost Modeling", default=False):
        cfg_number(ui, prefix, "slippage_points", "Slippage per trade (points)", 0.0, 10000.0, 1.0)
        cfg_number(ui, prefix, "spread_points", "Bid-Ask spread cost (points)", 0.0, 10000.0, 0.5)
        cfg_number(ui, prefix, "brokerage_flat", "Brokerage per order leg (currency)", 0.0, 10000.0, 20.0)
        ui.caption("Deducted from every trade: (slippage + spread) in points, plus brokerage charged twice per round trip (entry + exit).")

    # ------------------------------------------------------------- Dhan orders
    ui.markdown("### 🏦 Dhan Broker — Live Order Placement")
    dhan_enabled = cfg_checkbox(ui, prefix, "dhan_enabled", "Enable Dhan Order Placement (LIVE)", default=False)
    if dhan_enabled:
        ui.warning("⚠️ REAL orders will be sent to Dhan using the credentials above once an access token is entered. Without a token, orders are only simulated (payload shown, nothing sent).")
        if not use_dhan_data:
            ui.caption("Using the Dhan Client ID / Access Token from the 🔐 Dhan Account section above.")

        inst = cfg_selectbox(ui, prefix, "dhan_instrument", "Instrument", DHAN_INSTRUMENTS, default="Stock Intraday")
        cfg_selectbox(ui, prefix, "dhan_exchange", "Exchange", ["NSE", "BSE"], default="NSE",
                      help="NSE for Nifty / BankNifty / NSE stocks (default); BSE for Sensex / BSE-listed stocks. "
                           "Auto-flips to BSE when a Sensex or .BO ticker is selected — always editable.")
        _auto_note = ("auto-fetched from the Dhan scrip master — editable"
                      if (cfg.get("use_dhan_data") or cfg.get("dhan_enabled")) and cfg.get("_dhan_auto_failed") is not True
                      else "enter manually")

        if inst in ("Stock Intraday", "Stock Delivery", "Stock Futures", "Index Futures"):
            cfg_text(ui, prefix, "dhan_security_id", f"Security ID — mandatory ({_auto_note})", default="")
            if not str(cfg.get("dhan_security_id", "")).strip():
                ui.error("Security ID is mandatory — orders cannot be routed without it. "
                         + ("It auto-fills from the scrip master; if it stays empty, check connectivity or enter it manually."
                            if (cfg.get("use_dhan_data") or cfg.get("dhan_enabled")) else "Enter it manually (yfinance mode does not auto-fill)."))
            if "Futures" in inst:
                fut_opts = cfg.get("_dhan_fut_expiries") or []
                if fut_opts:
                    cfg_selectbox(ui, prefix, "dhan_expiry", "Expiry Date (auto-fetched)", fut_opts)
                else:
                    cfg_text(ui, prefix, "dhan_expiry", "Expiry Date (YYYY-MM-DD)", default="")
            c1, c2 = ui.columns(2)
            cfg_selectbox(c1, prefix, "dhan_entry_order_type", "Entry Order Type", ["MARKET", "LIMIT"], default="MARKET")
            cfg_selectbox(c2, prefix, "dhan_exit_order_type", "Exit Order Type", ["MARKET", "LIMIT"], default="MARKET")
            cfg_number(ui, prefix, "dhan_qty", "Dhan Quantity", min_value=1, default=1, step=1)
            if cfg_checkbox(ui, prefix, "dhan_bo_enabled", "Use Broker SL/Target (Bracket Order)", default=False):
                b1, b2, b3 = ui.columns(3)
                cfg_number(b1, prefix, "dhan_bo_sl", "SL Points (boStopLossValue)", min_value=0.0, default=10.0, step=0.5)
                cfg_number(b2, prefix, "dhan_bo_target", "Target Points (boProfitValue)", min_value=0.0, default=20.0, step=0.5)
                cfg_number(b3, prefix, "dhan_bo_trail", "Trail SL Jump (0 = off)", min_value=0.0, default=0.0, step=0.5)
                ui.caption("Entry orders go out as Bracket Orders (productType=BO) — the BROKER then manages SL/Target legs "
                           "with these point distances. The app skips sending its own exit order on SL/Target hits to avoid "
                           "double exits (manual square-offs and signal exits are still sent).")
        else:  # Stock Options / Index Options
            opt_opts = cfg.get("_dhan_opt_expiries") or []
            if opt_opts:
                cfg_selectbox(ui, prefix, "dhan_opt_expiry", "Expiry Date (auto-fetched)", opt_opts)
            else:
                cfg_text(ui, prefix, "dhan_opt_expiry", "Expiry Date (YYYY-MM-DD)", default="")
            c1, c2 = ui.columns(2)
            cfg_number(c1, prefix, "dhan_ce_strike", "CE Strike Price (default ATM)", min_value=0.0, default=0.0, step=50.0)
            cfg_number(c2, prefix, "dhan_pe_strike", "PE Strike Price (default ATM)", min_value=0.0, default=0.0, step=50.0)
            s1, s2 = ui.columns(2)
            cfg_text(s1, prefix, "dhan_ce_security_id", f"CE Security ID — mandatory ({_auto_note})", default="")
            cfg_text(s2, prefix, "dhan_pe_security_id", f"PE Security ID — mandatory ({_auto_note})", default="")
            if not str(cfg.get("dhan_ce_security_id", "")).strip() or not str(cfg.get("dhan_pe_security_id", "")).strip():
                ui.error("CE and PE Security IDs are mandatory — option orders cannot be routed without them. "
                         + ("They auto-fill for the selected expiry/strikes; if empty, check connectivity or enter manually."
                            if (cfg.get("use_dhan_data") or cfg.get("dhan_enabled")) else "Enter them manually (yfinance mode does not auto-fill)."))
            cfg_number(ui, prefix, "dhan_qty", "Dhan Quantity (lot size — 65 Nifty / 35 BankNifty / 20 Sensex by default)",
                       min_value=1, default=1, step=1)
            c3, c4 = ui.columns(2)
            cfg_selectbox(c3, prefix, "dhan_entry_order_type", "Entry Order Type", ["MARKET", "LIMIT"], default="MARKET")
            cfg_selectbox(c4, prefix, "dhan_exit_order_type", "Exit Order Type", ["MARKET", "LIMIT"], default="MARKET")
            ui.caption("LONG signal → BUYs the CE leg; SHORT signal → BUYs the PE leg (buying both ways keeps risk defined — "
                       "no naked option selling). Exits SELL whichever leg is open. Strikes default to ATM (from live LTP) "
                       "and expiry to the nearest available — both editable.")
    else:
        ui.caption("Disabled by default. Live trading tab runs in paper/simulation mode until enabled.")

    # ------------------------------------------------------------------ Email
    ui.markdown("### 📧 Email Notifications")
    if cfg_checkbox(ui, prefix, "email_enabled", "Send Email Notification (entries / exits / square-offs)", default=False):
        cfg_text(ui, prefix, "email_from", "From (Gmail address)", default="srinivas.trml@gmail.com")
        cfg_text(ui, prefix, "email_to", "To (comma-separated for multiple)", default="")
        cfg_text(ui, prefix, "email_app_password", "Gmail App Password", default="", password=True)
        ui.caption("Uses Gmail SMTP with an App Password (Google Account → Security → 2-Step Verification → App passwords). "
                   "A mail failure never blocks trading — it just shows a warning.")


# ---------------------------------------------------------------------------
# Render the SIDEBAR copy of the config panel, then derive every runtime value
# the rest of the app uses from the shared store.
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Algo Configuration")
if st.session_state.pop("_cfg_applied_msg", False):
    st.sidebar.success("Optimized config applied ✅")

auto_populate_dhan_defaults()
render_config_panel(st.sidebar, "sb")

cfg = st.session_state.app_cfg

ticker_choice = cfg.get("ticker_choice", "Nifty50")
ticker = TICKER_MAP[ticker_choice] if ticker_choice != "Custom" and ticker_choice in TICKER_MAP else None
if ticker is None:
    ticker = cfg.get("custom_ticker", "KAYNES.NS")
interval = cfg.get("interval", "1m")
_periods_avail = TF_PERIOD_MAP.get(interval, ["7d"])
period = cfg.get("period") if cfg.get("period") in _periods_avail else ("7d" if "7d" in _periods_avail else _periods_avail[0])
qty = max(1, int(cfg.get("qty", 1)))
strategy = cfg.get("strategy", STRATEGIES[0])
sl_type = cfg.get("sl_type", SL_TYPES[0])
target_type = cfg.get("target_type", TARGET_TYPES[0])

PARAM_DEFAULTS = {
    "ema_fast": 9, "ema_slow": 15, "threshold": 0.0, "threshold_direction": "Below",
    "trade_type": "Both", "flip_signals": False, "close_fill_logic": False, "sr_window": 20, "liq_window": 20,
    "rsi_period": 14, "bb_period": 20, "bb_std": 2.0, "vol_window": 20, "vol_factor": 2.0,
    "zigzag_lookback": 3, "st_period": 10, "st_mult": 3.0, "orb_candles": 5,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "donchian_period": 20,
    "keltner_period": 20, "keltner_atr_mult": 1.5, "stoch_k": 14, "stoch_d": 3,
    "tema_period": 20, "cci_period": 20,
    "sl_points": 10.0, "atr_mult_sl": 1.5, "loss_trigger_points": 20.0, "min_recovery_pct": 50.0,
    "target_points": 20.0, "atr_mult_target": 3.0, "profit_trigger_points": 50.0, "giveback_pct": 30.0,
    "partial_target1_points": 20.0, "partial_book_pct": 50.0, "rr_ratio": 2.0,
}
params = {k: cfg.get(k, d) for k, d in PARAM_DEFAULTS.items()}

risk_ctrl = {
    "loss_duration_enabled": bool(cfg.get("loss_duration_enabled", False)),
    "loss_duration_min_minutes": float(cfg.get("loss_duration_min_minutes", 1.0)),
    "loss_duration_max_minutes": float(cfg.get("loss_duration_max_minutes", 5.0)),
}

FILTER_DEFAULTS = {
    "adx_enabled": False, "adx_min": 20, "adx_max": 100,
    "rsi_enabled": False, "bb_enabled": False, "ema20_enabled": False,
    "sma20_enabled": False, "smc_enabled": False,
    "atr_enabled": False, "atr_min": 0.0, "atr_max": 100000.0,
    "supertrend_enabled": False, "st_filter_period": 10, "st_filter_mult": 3.0,
    "regime_enabled": False, "regime_trend_min": 25, "regime_range_max": 20,
    "angle_enabled": False, "angle_min_deg": 0.0,
    "crossover_quality_enabled": False, "crossover_quality_mode": "Simple Crossover",
    "crossover_min_points": 1.0, "crossover_atr_mult": 1.0,
    "vix_enabled": False, "vix_min": 10.0, "vix_max": 25.0,
}
filters = {k: cfg.get(k, d) for k, d in FILTER_DEFAULTS.items()}

wf_enabled = bool(cfg.get("wf_enabled", False))
wf_folds = int(cfg.get("wf_folds", 5))
cost_enabled = bool(cfg.get("cost_enabled", False))
cost_cfg = {"slippage_points": 0.0, "spread_points": 0.0, "brokerage_flat": 0.0}
if cost_enabled:
    cost_cfg = {
        "slippage_points": float(cfg.get("slippage_points", 1.0)),
        "spread_points": float(cfg.get("spread_points", 0.5)),
        "brokerage_flat": float(cfg.get("brokerage_flat", 20.0)),
    }


def build_product_cfg(cfg, ticker_choice, ticker):
    """Derives the full Dhan order-routing config (exchange segment, product
    type, security IDs — incl. auto-resolved CE/PE legs for options) from the
    shared config store."""
    inst = cfg.get("dhan_instrument", "Stock Intraday")
    underlying = dhan_underlying_symbol(ticker_choice, ticker)
    exch = cfg.get("dhan_exchange") or ("BSE" if (ticker_choice == "Sensex" or str(ticker).upper().endswith(".BO")) else "NSE")
    pc = {
        "instrument": inst,
        "underlying": underlying,
        "entry_order_type": cfg.get("dhan_entry_order_type", "MARKET"),
        "exit_order_type": cfg.get("dhan_exit_order_type", "MARKET"),
        "dhan_qty": int(cfg.get("dhan_qty", 1) or 1),
        "bo_enabled": bool(cfg.get("dhan_bo_enabled", False)),
        "bo_sl": float(cfg.get("dhan_bo_sl", 10.0) or 0.0),
        "bo_target": float(cfg.get("dhan_bo_target", 20.0) or 0.0),
        "bo_trail": float(cfg.get("dhan_bo_trail", 0.0) or 0.0),
    }
    if inst in ("Stock Intraday", "Stock Delivery"):
        pc["exchange_segment"] = f"{exch}_EQ"
        pc["product"] = "CNC" if inst == "Stock Delivery" else "INTRADAY"
        pc["security_id"] = str(cfg.get("dhan_security_id", "")).strip()
    elif inst in ("Stock Futures", "Index Futures"):
        pc["exchange_segment"] = f"{exch}_FNO"
        pc["product"] = "MARGIN"
        pc["security_id"] = str(cfg.get("dhan_security_id", "")).strip()
        pc["expiry"] = str(cfg.get("dhan_expiry", "")).strip()
    else:  # Stock Options / Index Options
        pc["exchange_segment"] = f"{exch}_FNO"
        pc["product"] = "INTRADAY"
        pc["expiry"] = str(cfg.get("dhan_opt_expiry", "")).strip()
        pc["ce_strike"] = float(cfg.get("dhan_ce_strike", 0) or 0)
        pc["pe_strike"] = float(cfg.get("dhan_pe_strike", 0) or 0)
        # The CE/PE Security ID input boxes are the source of truth (auto-filled
        # in Dhan mode, manual in yfinance mode); scrip-master lookup is only a
        # fallback if they were left empty.
        pc["ce_security_id"] = str(cfg.get("dhan_ce_security_id", "")).strip() or (
            lookup_option_security_id(underlying, pc["expiry"], pc["ce_strike"], "CE") or "")
        pc["pe_security_id"] = str(cfg.get("dhan_pe_security_id", "")).strip() or (
            lookup_option_security_id(underlying, pc["expiry"], pc["pe_strike"], "PE") or "")
    return pc


dhan_enabled = bool(cfg.get("dhan_enabled", False))
dhan_client_id = str(cfg.get("dhan_client_id", "1104779876")).strip()
dhan_access_token = str(cfg.get("dhan_access_token", "")).strip()
product_cfg = build_product_cfg(cfg, ticker_choice, ticker) if dhan_enabled else {}

config = dict(
    ticker=ticker, ticker_choice=ticker_choice, interval=interval, period=period, qty=qty,
    strategy=strategy, sl_type=sl_type, target_type=target_type, params=params, filters=filters,
    wf_enabled=wf_enabled, wf_folds=wf_folds, cost_enabled=cost_enabled, cost_cfg=cost_cfg,
    risk_ctrl=risk_ctrl,
)


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

    if filters.get("vix_enabled"):
        vix_aligned = get_vix_aligned(df.index)
        vix_val = vix_aligned.iloc[-1] if len(vix_aligned) else np.nan
        vix_min, vix_max = filters.get("vix_min", 0), filters.get("vix_max", 100)
        if pd.isna(vix_val):
            lines.append("India VIX filter: N/A — couldn't fetch VIX data right now.")
        else:
            ok = vix_min <= vix_val <= vix_max
            lines.append(f"India VIX: {vix_val:.2f}, needs [{vix_min}, {vix_max}] → {'✅ OK' if ok else '❌ blocking entries right now'}")

    return lines


def render_live_dashboard(ticker, interval, period, strategy, params, filters, live=True):
    """Renders the 📊 Indicator Dashboard + 📟 Signal Status Board (all
    confluence/filter statuses). Called by the auto-refreshing fragment while
    monitoring is ON, and directly (single static snapshot) while it's OFF —
    so the board is ALWAYS visible on the Live tab."""
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
    if live:
        st.caption("What the current (last closed) candle is showing vs. what's needed to trigger a fresh buy or sell. Updates automatically every ~3s while live monitoring is on.")
    else:
        st.caption("What the current (last closed) candle is showing vs. what's needed to trigger a fresh buy or sell. Static snapshot — click Start for live auto-refresh every ~3s.")
    for line in describe_signal_status(raw_status, strategy, params, filters):
        st.write("• " + line)


@st.fragment(run_every=3)
def live_dashboard_fragment(ticker, interval, period, strategy, params, filters):
    """
    Everything here re-renders on its own every ~3s WITHOUT rerunning the rest
    of the page — this is what makes the signal status board (RSI/EMA/ADX/etc.
    values) update live instead of only on button click or full page refresh.
    Only ever mounted while Live Monitoring is ON, so it costs zero extra API
    calls while stopped.
    """
    render_live_dashboard(ticker, interval, period, strategy, params, filters, live=True)


def evaluate_live_signal(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                          dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl):
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

    # ---- Live exit engine selection --------------------------------------
    # "Candle-Close SL/Target Fills" checkbox (OFF by default):
    #   OFF → ORIGINAL behavior: SL checked against LTP first, then Target
    #         against LTP; exits fill at the actually observed price.
    #   ON  → backtest-identical candle logic: wait for the candle to close,
    #         LONG checks SL vs candle Low then Target vs candle High (SHORT:
    #         SL vs High then Target vs Low), and a breached level fills at
    #         the candle's CLOSE — same as the backtest with the box ON.
    cfg_live = st.session_state.get("app_cfg", {})
    use_candle_logic = (
        bool(cfg_live.get("close_fill_logic", False))
        and strategy not in IMMEDIATE_EXECUTION_STRATEGIES
        and len(sig_df) >= 2
    )

    # Immediate-execution strategies (Simple Buy/Sell Only, Threshold Cross)
    # check the CURRENT price against the last CLOSED candle directly and enter
    # IMMEDIATELY at LTP — no "wait for next candle open" delay, since these
    # aren't candle-shape strategies, just price conditions.
    if strategy in IMMEDIATE_EXECUTION_STRATEGIES:
        prev_close = float(sig_df["Close"].iloc[-2])
        if strategy == "Simple Buy Only":
            last_sig = 1 if ltp > prev_close else 0
        elif strategy == "Simple Sell Only":
            last_sig = -1 if ltp < prev_close else 0
        else:  # Threshold Cross
            thr = cfg_live.get("threshold", params.get("threshold", prev_close))
            thr_dir = cfg_live.get("threshold_direction", params.get("threshold_direction", "Below"))
            last_sig = 0
            if thr_dir == "Below":
                # Cross the level from BELOW (upward) → long.
                if ltp > thr and prev_close <= thr:
                    last_sig = 1
            else:  # "Above" — cross from above (downward) → short.
                if ltp < thr and prev_close >= thr:
                    last_sig = -1
        # Flip/Reverse applies to immediate strategies too.
        if cfg_live.get("flip_signals", False):
            last_sig = -last_sig
        # Trade Direction filter also applies to immediate strategies.
        tt = cfg_live.get("trade_type", "Both")
        if (tt == "Long Only" and last_sig == -1) or (tt == "Short Only" and last_sig == 1):
            last_sig = 0
        entry_reference_price = ltp
    else:
        last_sig = int(sig_df["signal"].iloc[-2])  # last CLOSED candle's signal
        entry_reference_price = float(sig_df["Open"].iloc[-1])  # next candle's open

    if open_pos:
        pos = open_pos[0]
        if use_candle_logic:
            # Evaluate against the last CLOSED candle — identical to backtest.
            i = len(sig_df) - 2
            candle = sig_df.iloc[i]
            ref_price = float(candle["Close"])
            pos = update_trade_levels(pos, i, sig_df, params, a_series)
            pos["highest"] = max(pos["highest"], float(candle["High"]))
            pos["lowest"] = min(pos["lowest"], float(candle["Low"]))
        else:
            i = len(sig_df) - 1
            candle = sig_df.iloc[i]
            ref_price = ltp
            pos = update_trade_levels(pos, i, sig_df, params, a_series)
            pos["highest"] = max(pos["highest"], ltp)
            pos["lowest"] = min(pos["lowest"], ltp)

        exited, exit_price, reason = False, None, None
        if pos.get("pending_exit_reason"):
            # Pending signal-exit always executes at the NEXT candle's open —
            # same as the backtest engine's exit rule.
            exited, exit_price, reason = True, float(sig_df["Open"].iloc[-1]), pos["pending_exit_reason"]
        if not exited:
            sp_exit, sp_price, sp_reason = check_special_exit_conditions(pos, {"Close": ref_price})
            if sp_exit:
                exited, exit_price, reason = True, sp_price, sp_reason
        if not exited and cfg_live.get("max_profit_hold_enabled"):
            # Max hold duration for a PROFITABLE trade: once held ≥ N minutes
            # AND currently in profit, book it.
            try:
                entry_ts = pd.Timestamp(pos["entry_time"])
                now_ts = pd.Timestamp.now(tz=entry_ts.tz) if entry_ts.tz is not None else pd.Timestamp.now()
                held_min = (now_ts - entry_ts).total_seconds() / 60.0
            except Exception:
                held_min = 0.0
            pl_pts = (ref_price - pos["entry_price"]) * pos["direction"]
            if pl_pts > 0 and held_min >= float(cfg_live.get("max_profit_hold_minutes", 1.0)):
                exited, exit_price, reason = True, ref_price, f"Max Profitable Hold ({held_min:.1f} min in profit)"
        if not exited and risk_ctrl.get("loss_duration_enabled"):
            td_exit, td_price, td_reason = check_time_based_exit(
                pos, sig_df.index[i], ref_price,
                risk_ctrl.get("loss_duration_min_minutes", 1), risk_ctrl.get("loss_duration_max_minutes", 5),
            )
            if td_exit:
                exited, exit_price, reason = True, td_price, td_reason
        if not exited:
            if use_candle_logic:
                # Backtest-identical conservative fill: LONG checks SL (candle
                # Low) BEFORE Target (candle High); SHORT checks SL (candle
                # High) BEFORE Target (candle Low).
                hard_exit, hard_price, hard_reason = check_hard_exit(pos, candle, close_fill=True)
            else:
                hard_exit, hard_price, hard_reason = check_hard_exit_ltp(pos, ltp)
            if hard_exit:
                if pos["target_type"] == "Partial Book + Trail Remainder" and "Target Hit" in hard_reason and not pos["partial_booked"]:
                    book_qty = max(1, round(pos["original_qty"] * pos["partial_book_pct"] / 100.0))
                    book_qty = min(book_qty, pos["remaining_qty"])
                    partial_points = (hard_price - pos["entry_price"]) * pos["direction"]
                    st.session_state.live_history.append({
                        "Entry Time": pos["entry_time"], "Entry Price": round(pos["entry_price"], 2),
                        "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
                        "Exit Time": sig_df.index[-1], "Exit Price": round(float(hard_price), 2),
                        "SL": round(pos["initial_sl"], 2), "Target": round(pos["initial_target"], 2),
                        "Candle Open": round(float(candle["Open"]), 2), "Candle High": round(float(candle["High"]), 2),
                        "Candle Low": round(float(candle["Low"]), 2), "Candle Close": round(float(candle["Close"]), 2),
                        "Highest": round(pos["highest"], 2), "Lowest": round(pos["lowest"], 2),
                        "Points": round(partial_points, 2), "PnL": round(partial_points * book_qty, 2),
                        "Exit Reason": f"Partial Book ({book_qty}/{pos['original_qty']} qty @ Target 1)", "Qty": book_qty,
                    })
                    pos["remaining_qty"] -= book_qty
                    pos["partial_booked"] = True
                    if dhan_enabled:
                        leg_id, side = resolve_dhan_order_leg(pos["direction"], False, ticker, product_cfg)
                        dhan_part_qty = max(1, round(dhan_order_qty(product_cfg) * pos["partial_book_pct"] / 100.0))
                        st.json(place_dhan_order(dhan_client_id, dhan_access_token, leg_id, side, product_cfg,
                                                 dhan_part_qty, hard_price,
                                                 order_type=product_cfg.get("exit_order_type", "MARKET"), is_entry=False))
                    send_email_notification(
                        f"[AlgoTrader] PARTIAL BOOK {ticker} @ {float(hard_price):.2f}",
                        f"Booked {book_qty}/{pos['original_qty']} qty at Target 1.\n"
                        f"Entry: {pos['entry_price']:.2f}\nBooked at: {float(hard_price):.2f}\n"
                        f"Points: {partial_points:+.2f}\nRemaining qty now trailing: {pos['remaining_qty']}",
                    )
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
        if not exited:
            sig_exit, sig_reason = detect_signal_exit_condition(pos, i, sig_df, params)
            if sig_exit:
                pos["pending_exit_reason"] = sig_reason

        pos["current_price"] = ltp
        if exited:
            points = (exit_price - pos["entry_price"]) * pos["direction"]
            st.session_state.live_history.append({
                "Entry Time": pos["entry_time"], "Entry Price": round(pos["entry_price"], 2),
                "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
                "Exit Time": sig_df.index[-1], "Exit Price": round(float(exit_price), 2),
                "SL": round(pos["initial_sl"], 2), "Target": round(pos["initial_target"], 2),
                "Candle Open": round(float(candle["Open"]), 2), "Candle High": round(float(candle["High"]), 2),
                "Candle Low": round(float(candle["Low"]), 2), "Candle Close": round(float(candle["Close"]), 2),
                "Highest": round(pos["highest"], 2), "Lowest": round(pos["lowest"], 2),
                "Points": round(points, 2), "PnL": round(points * pos["remaining_qty"], 2),
                "Exit Reason": reason, "Qty": pos["remaining_qty"],
            })
            st.session_state.live_positions = []
            st.session_state.last_trade_event_ts = time.time()
            st.success(f"Position closed: {reason} @ {exit_price:.2f}")
            if dhan_enabled and dhan_should_send_exit(product_cfg, reason):
                leg_id, side = resolve_dhan_order_leg(pos["direction"], False, ticker, product_cfg)
                st.json(place_dhan_order(dhan_client_id, dhan_access_token, leg_id, side, product_cfg,
                                         dhan_order_qty(product_cfg), exit_price,
                                         order_type=product_cfg.get("exit_order_type", "MARKET"), is_entry=False))
            send_email_notification(
                f"[AlgoTrader] EXIT {'LONG' if pos['direction'] == 1 else 'SHORT'} {ticker} @ {float(exit_price):.2f}",
                f"Reason: {reason}\nEntry: {pos['entry_price']:.2f}\nExit: {float(exit_price):.2f}\n"
                f"Points: {points:+.2f}\nPnL: {points * pos['remaining_qty']:+.2f}\nQty: {pos['remaining_qty']}",
            )
        else:
            st.session_state.live_positions = [pos]
            st.info("Position still open — levels updated.")
    elif last_sig != 0:
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
            gate_ok, gate_reason = check_live_entry_gates(cfg_live, ticker)
            if not gate_ok:
                st.warning(f"🚧 Entry blocked: {gate_reason}")
                return sig_df
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
            st.session_state.live_positions = [new_pos]
            st.session_state.last_acted_signal_marker = signal_marker
            _record_entry_taken()
            st.success(f"New {'LONG' if last_sig == 1 else 'SHORT'} position opened @ {entry_price:.2f}")
            if dhan_enabled:
                leg_id, side = resolve_dhan_order_leg(last_sig, True, ticker, product_cfg)
                st.json(place_dhan_order(dhan_client_id, dhan_access_token, leg_id, side, product_cfg,
                                         dhan_order_qty(product_cfg), entry_price,
                                         order_type=product_cfg.get("entry_order_type", "MARKET"), is_entry=True))
            send_email_notification(
                f"[AlgoTrader] ENTRY {'LONG' if last_sig == 1 else 'SHORT'} {ticker} @ {entry_price:.2f}",
                f"Strategy: {strategy}\nTimeframe: {interval}\nEntry: {entry_price:.2f}\n"
                f"SL: {sl:.2f}\nTarget: {target:.2f}\nQty: {qty}",
            )
    else:
        st.caption("No new signal on the latest closed candle.")
    return sig_df


@st.fragment(run_every=5)
def live_signal_loop_fragment(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                               dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl):
    """Re-runs evaluate_live_signal() every ~5s on its own, independent of the
    rest of the page — this is what makes entries/exits keep happening while
    Live Monitoring is on, instead of firing only once at the Start click."""
    evaluate_live_signal(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                          dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl)


def apply_config_to_sidebar(cfg_row):
    """Push a chosen optimization result row into the shared config store —
    both the sidebar AND the Admin Panel instantly reflect it."""
    cfg = st.session_state.app_cfg
    row_choice = cfg_row.get("ticker_choice", ticker_choice)
    cfg["ticker_choice"] = row_choice
    if row_choice == "Custom":
        cfg["custom_ticker"] = cfg_row.get("ticker", ticker)
    cfg["interval"] = cfg_row["Timeframe"]
    cfg["period"] = cfg_row["Period"]
    cfg["strategy"] = cfg_row["Strategy"]
    cfg["sl_type"] = cfg_row.get("SL Type", sl_type)
    cfg["target_type"] = cfg_row.get("Target Type", target_type)
    st.session_state["_cfg_applied_msg"] = True
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

    _cfg_now = st.session_state.app_cfg
    if _cfg_now.get("use_dhan_data") and _cfg_now.get("dhan_access_token"):
        if dhan_data_meta(ticker) is None:
            st.warning(f"📡 Dhan can't serve **{ticker}** (crypto/commodity/forex) — automatically using yfinance "
                       "(0.3s rate-limit delay) for this ticker. Dhan will be used again for supported tickers.")
        else:
            st.caption("📡 Data source: **Dhan feed** — no API delay applied.")
    else:
        st.caption("📡 Data source: **yfinance** — a fixed 0.3s delay is applied to every API call (rate-limit protection).")
    if _cfg_now.get("close_fill_logic", False):
        st.caption("🧮 Exit engine: **Candle-Close fills** — waits for candle close; LONG: SL vs Low → Target vs High (SHORT: SL vs High → Target vs Low); breached SL/Target fills at the candle's CLOSE (P&L from close). Matches the backtest with the same box ON.")
    else:
        st.caption("🧮 Exit engine: **LTP logic (default/original)** — SL checked against live LTP first, then Target; exits at the observed price. Backtest fills at the SL/Target level.")

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
        exit_price = float(raw["Close"].iloc[-1]) if not raw.empty else pos["current_price"]
        if not raw.empty:
            _sc = raw.iloc[-1]
            _so = {"Candle Open": round(float(_sc["Open"]), 2), "Candle High": round(float(_sc["High"]), 2),
                   "Candle Low": round(float(_sc["Low"]), 2), "Candle Close": round(float(_sc["Close"]), 2)}
        else:
            _so = {"Candle Open": None, "Candle High": None, "Candle Low": None, "Candle Close": None}
        points = (exit_price - pos["entry_price"]) * pos["direction"]
        st.session_state.live_history.append({
            "Entry Time": pos["entry_time"], "Entry Price": round(pos["entry_price"], 2),
            "Direction": "LONG" if pos["direction"] == 1 else "SHORT",
            "Exit Time": datetime.now(), "Exit Price": round(exit_price, 2),
            "SL": round(pos["initial_sl"], 2), "Target": round(pos["initial_target"], 2),
            "Highest": round(pos["highest"], 2), "Lowest": round(pos["lowest"], 2),
            "Points": round(points, 2), "PnL": round(points * pos["remaining_qty"], 2),
            "Exit Reason": "Manual Square Off", "Qty": pos["remaining_qty"], **_so,
        })
        st.session_state.live_positions = []
        st.session_state.last_trade_event_ts = time.time()
        st.warning(f"Manually squared off @ {exit_price:.2f}")
        if dhan_enabled:
            leg_id, side = resolve_dhan_order_leg(pos["direction"], False, ticker, product_cfg)
            st.json(place_dhan_order(dhan_client_id, dhan_access_token, leg_id, side, product_cfg,
                                     dhan_order_qty(product_cfg), exit_price,
                                     order_type=product_cfg.get("exit_order_type", "MARKET"), is_entry=False))
        send_email_notification(
            f"[AlgoTrader] MANUAL SQUARE OFF {ticker} @ {exit_price:.2f}",
            f"Entry: {pos['entry_price']:.2f}\nExit: {exit_price:.2f}\nPoints: {points:+.2f}\n"
            f"PnL: {points * pos['remaining_qty']:+.2f}",
        )
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
            "Data Source": "Dhan (no delay)" if (_cfg_now.get("use_dhan_data") and _cfg_now.get("dhan_access_token") and dhan_data_meta(ticker)) else "yfinance (0.3s delay)",
            "SL/Target Fill Logic": "Candle-Close fills (backtest + live)" if _cfg_now.get("close_fill_logic", False) else "Original: level-price fills (backtest) / LTP checks (live)",
            "Dhan Live Orders": dhan_enabled,
            "Dhan Product Config": {k: v for k, v in product_cfg.items() if not str(k).startswith("_")} if dhan_enabled else "—",
            "Email Notifications": bool(_cfg_now.get("email_enabled")),
        })

    if manual_eval:
        evaluate_live_signal(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                              dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl)

    if st.session_state.live_running:
        # THIS is what makes trade entry/exit actually keep happening while
        # monitoring is on: a plain TOP-LEVEL fragment (same pattern as
        # live_dashboard_fragment / live_position_fragment below), not a
        # closure nested inside this tab — nesting it was fragile and is
        # exactly what caused positions to silently stop updating before.
        live_signal_loop_fragment(ticker, interval, period, strategy, params, filters, sl_type, target_type, qty,
                                   dhan_enabled, dhan_client_id, dhan_access_token, product_cfg, risk_ctrl)

    if st.session_state.live_running:
        live_dashboard_fragment(ticker, interval, period, strategy, params, filters)
    else:
        # Monitoring OFF: still show the full Indicator Dashboard + Signal
        # Status Board as a static snapshot, so the confluence/parameter
        # status is ALWAYS visible on this tab (it just doesn't auto-refresh
        # until you click Start).
        render_live_dashboard(ticker, interval, period, strategy, params, filters, live=False)

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

# ============================================================================
# TAB: ADMIN PANEL
# ============================================================================

with tab_admin:
    st.subheader("🛠 Admin Panel — Full Configuration")
    st.caption(
        "Every control from the sidebar, in one full-width panel. Both are live views of the SAME "
        "configuration — change a value here and the sidebar updates instantly (and vice versa)."
    )
    render_config_panel(st, "adm")

# ============================================================================
# FOOTER / GLOBAL DISCLAIMER
# ============================================================================

st.divider()
st.caption(
    "⚠️ Educational tool. Backtests use simplified conservative fill logic — real results will differ "
    "unless realistic cost modeling is enabled, and even then liquidity/queue effects are ignored. "
    "Verify any strategy on out-of-sample data and paper-trade before committing capital. "
    "When Dhan order placement is enabled WITH an access token, REAL orders are sent to Dhan's live API "
    "(api.dhan.co) — without a token, order payloads are only simulated and displayed."
)
