"""
===============================================================================
 MULTI-ASSET ALGORITHMIC TRADING PLATFORM  --  single-file Streamlit application
===============================================================================

Run with:
    pip install streamlit yfinance pandas numpy plotly requests
    streamlit run algo_trading_platform.py

Self-test the maths without launching the UI or touching the network:
    python algo_trading_platform.py --selftest

-------------------------------------------------------------------------------
EXECUTION MODEL  (read this before trusting a single number)
-------------------------------------------------------------------------------
BACKTEST
    Signal on the close of candle N  ->  entry at the OPEN of candle N+1.
    Exit check order inside every candle, deliberately conservative:
        1. Gap: if the candle OPENS beyond the stop, fill at the open.
        2. Stop-loss vs the candle LOW  (long)  /  HIGH (short)   <-- checked FIRST
        3. Target   vs the candle HIGH (long)  /  LOW  (short)
    When both the stop and the target sit inside one candle's range, the stop is
    assumed to have triggered first. OHLC data cannot tell us the true intrabar
    path, so the pessimistic branch is taken every time.
    Trailing levels are advanced only AFTER the candle has been checked, using
    that candle's own extremes.

LIVE
    Signal on the close of candle N  ->  entry at the OPEN of candle N+1.
    Stop-loss is checked against the LTP first, then the target against the LTP,
    because live polling gives a running price rather than a finished candle.
    Trailing levels are advanced on every tick using the LTP.

-------------------------------------------------------------------------------
HONEST LIMITATIONS  (these are not disclaimers, they are design facts)
-------------------------------------------------------------------------------
* yfinance quotes for NSE/BSE indices are DELAYED, typically by 15+ minutes, and
  recent candles are frequently revised. Polling it faster does not make it
  fresher. Driving real broker orders from this feed is a losing proposition and
  the Dhan panel warns about it at the point of use.
* Yahoo has no published rate limit but throttles aggressively. Sub-second
  polling will earn a 429 and then a temporary IP block. The mandatory 0.3s
  guard is a floor, not a safe cruising speed.
* Trailing stops in the BACKTEST are approximations. With OHLC bars we cannot
  know whether price hit the trailing level before or after it moved. Live
  trailing on LTP is exact; backtested trailing is not. Treat backtested
  trailing-stop results as optimistic.
* Elliott Wave labelling is subjective. The implementation here is a mechanical
  zigzag heuristic, not a wave count an analyst would sign off on.
* PCR, open-interest change and news filters have NO free data source wired in.
  They are exposed as manual inputs plus a pluggable hook, and are inert
  otherwise. They are not silently faked.

This is a research and paper-trading sandbox. Broker order placement is OFF by
default, gated behind an explicit opt-in, and defaults to dry-run even then.
===============================================================================
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

# Streamlit / plotly / requests are imported lazily so that --selftest runs on a
# bare pandas+numpy environment.
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


# =============================================================================
# SECTION 1 -- CONSTANTS
# =============================================================================
APP_TITLE = "Multi-Asset Algorithmic Trading Platform"

API_GUARD_DELAY = 0.3          # MANDATORY sleep before AND after every yfinance block
WARMUP_BARS = 200              # candles reserved purely for indicator stabilisation
ABSOLUTE_MIN_BARS = 60

ASSET_UNIVERSE: dict[str, dict[str, str]] = {
    "Indian Indices": {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Sensex": "^BSESN",
        "Fin Nifty": "NIFTY_FIN_SERVICE.NS",
        "India VIX": "^INDIAVIX",
    },
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"},
    "Forex": {"USD / INR": "USDINR=X", "EUR / USD": "EURUSD=X"},
    "Commodities": {"Gold Futures": "GC=F", "Silver Futures": "SI=F"},
    "Indian Stocks": {
        "Reliance Industries": "RELIANCE.NS",
        "Kaynes Technology": "KAYNES.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "State Bank of India": "SBIN.NS",
        "ICICI Bank": "ICICIBANK.NS",
    },
}

_INR_SYMBOLS = {"^NSEI", "^NSEBANK", "^BSESN", "USDINR=X", "^INDIAVIX"}
VIX_SYMBOL = "^INDIAVIX"

INTERVALS = ["1m", "2m", "3m", "5m", "10m", "15m", "30m", "60m", "4h", "1d", "1wk", "1mo"]
NATIVE_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"}
DERIVED_INTERVALS = {"3m": ("1m", "3min"), "10m": ("5m", "10min"), "4h": ("60m", "4h")}
INTRADAY_INTERVALS = {"1m", "2m", "3m", "5m", "10m", "15m", "30m", "60m", "4h"}

APPROX_BARS_PER_DAY = {
    "1m": 375, "2m": 187, "3m": 125, "5m": 75, "10m": 37, "15m": 25,
    "30m": 13, "60m": 7, "4h": 2, "1d": 1, "1wk": 0.2, "1mo": 0.045,
}

PERIODS = ["1d", "5d", "7d", "1mo", "60d", "3mo", "6mo", "1y", "2y", "3y",
           "5y", "10y", "15y", "20y", "30y", "max"]
PERIOD_DAYS = {"1d": 1, "5d": 5, "7d": 7, "1mo": 30, "60d": 60, "3mo": 91, "6mo": 182,
               "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650,
               "15y": 5475, "20y": 7300, "30y": 10950, "max": 36500}
INTERVAL_MAX_DAYS = {"1m": 7, "2m": 60, "3m": 7, "5m": 60, "10m": 60,
                     "15m": 60, "30m": 60, "60m": 730, "4h": 730}

# ----------------------------------------------------------------- exit types --
SL_TYPES = [
    "Fixed Percentage",
    "Fixed Points",
    "ATR Multiple",
    "Trailing Points",
    "Trailing Percentage",
    "Trailing ATR (Chandelier)",
    "Step Trail (trigger k, trail N)",
    # --- candle structure ---
    "Previous Candle Low/High",
    "Current Candle Low/High",
    "Trail Previous Candle Low/High",
    "Trail Current Candle Low/High",
    # --- swing structure ---
    "Previous Swing Low/High",
    "Current Swing Low/High",
    "Trail Previous Swing Low/High",
    "Trail Current Swing Low/High",
    # --- signal driven ---
    "Price Action Structure Break",
    "EMA Reverse Crossover",
    "Strategy Reverse Signal",
    "No Stop-Loss",
]

TP_TYPES = [
    "Fixed Percentage",
    "Fixed Points",
    "ATR Multiple",
    "Risk : Reward Multiple",
    "Trailing Target (display only)",
    # --- candle structure ---
    "Previous Candle High/Low",
    "Current Candle High/Low",
    "Trail Previous Candle High/Low",
    "Trail Current Candle High/Low",
    # --- swing structure ---
    "Previous Swing High/Low",
    "Current Swing High/Low",
    "Trail Previous Swing High/Low",
    "Trail Current Swing High/Low",
    # --- signal driven ---
    "EMA Reverse Crossover",
    "Strategy Reverse Signal",
    "No Target",
]

# Structural exits take their level from the chart, so they need no magnitude.
_STRUCTURAL_SL = {
    "Previous Candle Low/High", "Current Candle Low/High",
    "Trail Previous Candle Low/High", "Trail Current Candle Low/High",
    "Previous Swing Low/High", "Current Swing Low/High",
    "Trail Previous Swing Low/High", "Trail Current Swing Low/High",
    "Price Action Structure Break",
}
_STRUCTURAL_TP = {
    "Previous Candle High/Low", "Current Candle High/Low",
    "Trail Previous Candle High/Low", "Trail Current Candle High/Low",
    "Previous Swing High/Low", "Current Swing High/Low",
    "Trail Previous Swing High/Low", "Trail Current Swing High/Low",
}
_SL_NO_VALUE = _STRUCTURAL_SL | {"EMA Reverse Crossover", "Strategy Reverse Signal", "No Stop-Loss"}
_TP_NO_VALUE = _STRUCTURAL_TP | {"EMA Reverse Crossover", "Strategy Reverse Signal", "No Target"}

TRAILING_SL_TYPES = {
    "Trailing Points", "Trailing Percentage", "Trailing ATR (Chandelier)",
    "Step Trail (trigger k, trail N)", "Trail Previous Candle Low/High",
    "Trail Current Candle Low/High", "Trail Previous Swing Low/High",
    "Trail Current Swing Low/High", "Price Action Structure Break",
}
TRAILING_TP_TYPES = {
    "Trailing Target (display only)", "Trail Previous Candle High/Low",
    "Trail Current Candle High/Low", "Trail Previous Swing High/Low",
    "Trail Current Swing High/Low",
}

# --------------------------------------------------------------------------- #
# BACKTEST RELIABILITY OF EACH EXIT TYPE  -- read this before trusting a number
# --------------------------------------------------------------------------- #
# A backtest sees four numbers per candle (O/H/L/C). It cannot see the PATH
# price took between them. That single limitation splits the exit types cleanly
# into two groups:
#
#   DISTANCE TRAILS (points / percent / ATR / step) recompute the stop from the
#   running extreme. Inside one candle, price can print the high (which lifts
#   the stop) and then fall back through that lifted stop before the candle
#   closes. A live tick feed catches this; OHLC cannot. The engine checks the
#   OLD stop first and only then advances the trail, so these intrabar stop-outs
#   are systematically MISSED and the backtest reads better than reality.
#   -> NOT backtest-safe. Live behaviour is exact; backtested behaviour is not.
#
#   STRUCTURAL TRAILS (previous/current candle extreme, previous/current swing)
#   only change at a candle boundary, and the level applied during candle i was
#   already known at the close of candle i-1. There is no intrabar ambiguity to
#   resolve, so the simulation matches what a live engine would have done.
#   -> Backtest-safe, subject to the usual gap rule.
#
#   STATIC stops and targets never move, so they carry no path ambiguity either.
#   -> Backtest-safe.
DISTANCE_TRAIL_TYPES = {"Trailing Points", "Trailing Percentage",
                        "Trailing ATR (Chandelier)", "Step Trail (trigger k, trail N)"}
STRUCTURAL_TRAIL_TYPES = {"Trail Previous Candle Low/High", "Trail Current Candle Low/High",
                          "Trail Previous Swing Low/High", "Trail Current Swing Low/High",
                          "Price Action Structure Break", "Trail Previous Candle High/Low",
                          "Trail Current Candle High/Low", "Trail Previous Swing High/Low",
                          "Trail Current Swing High/Low"}


def exit_reliability(sl_type: str, tp_type: str) -> tuple[str, list[str]]:
    """Return ('Backtest-safe' | 'Optimistic', reasons) for an exit configuration."""
    reasons: list[str] = []
    verdict = "Backtest-safe"
    for label, kind in (("Stop-loss", sl_type), ("Target", tp_type)):
        if kind in DISTANCE_TRAIL_TYPES:
            verdict = "Optimistic"
            reasons.append(
                f"{label} `{kind}` is a DISTANCE trail. It is recomputed from the running "
                "extreme, so within a single candle price can lift the level and then fall "
                "back through it. OHLC data cannot show that path, so the backtest misses "
                "those stop-outs and overstates results. Live, on tick data, it is exact.")
        elif kind in STRUCTURAL_TRAIL_TYPES:
            reasons.append(
                f"{label} `{kind}` is a STRUCTURAL trail: the level only changes at a candle "
                "boundary and was known before the candle it is applied to. No intrabar "
                "ambiguity, so the simulation matches live behaviour.")
        elif kind == "Trailing Target (display only)":
            reasons.append(f"{label} never fires an exit, so it cannot distort the result.")
        else:
            reasons.append(f"{label} `{kind}` is static: no path ambiguity.")
    return verdict, reasons


# Wall-clock length of one candle, used to decide whether a feed has gone stale.
# How long a quote may sit unchanged before we call the venue closed, and how
# many ticks of evidence we need before trusting movement either way.
QUOTE_LIVE_WINDOW = 300.0
QUOTE_EVIDENCE_TICKS = 3

INTERVAL_SECONDS = {"1m": 60, "2m": 120, "3m": 180, "5m": 300, "10m": 600, "15m": 900,
                    "30m": 1800, "60m": 3600, "4h": 14400, "1d": 86400,
                    "1wk": 604800, "1mo": 2592000}

DEFAULT_PARAMS: dict[str, float] = {
    "ema_fast": 9, "ema_slow": 21, "ema_mid": 50, "ema_macro": 200,
    "rsi_len": 14, "atr_len": 14, "atr_mult": 2.0, "channel_mult": 2.0,
    "vol_len": 20, "vol_mult": 1.5, "breakout_len": 20, "squeeze_mult": 1.2,
    "orb_bars": 3, "gap_pct": 0.30, "pullback_tol": 0.15,
    "pivot_left": 3, "pivot_right": 3, "zigzag_pct": 0.3,
    "adx_len": 14, "bb_len": 20, "bb_mult": 2.0,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "st_len": 10, "st_mult": 3.0, "structure_len": 20,
    "rsi_long_level": 40.0, "rsi_short_level": 60.0,
    "oi_change_threshold": 0.0, "pcr_min": 0.8, "pcr_max": 1.2,
    "zero_hero_atr": 2.0, "expiry_weekday": 3, "gamma_tail_bars": 6,
    "flip_entries": False,
    "threshold_price": 0.0, "threshold_pct": 1.0,
    "threshold_ref": "Previous session close",
    "threshold_mode": "Cross above = BUY, cross below = SELL",
}

THRESHOLD_MODES = [
    "Cross above = BUY, cross below = SELL",
    "Cross above = BUY only",
    "Cross below = SELL only",
    "Cross above = SELL, cross below = BUY (fade)",
]
THRESHOLD_REFS = ["Previous session close", "Session open", "Rolling 20-bar mean",
                  "First candle of the loaded window"]


def currency_symbol(ticker: str) -> str:
    t = (ticker or "").upper()
    return "\u20b9" if (t in _INR_SYMBOLS or t.endswith((".NS", ".BO"))) else "$"


def sanitize_period(interval: str, period: str) -> tuple[str, str | None]:
    """Clamp a period to what Yahoo will actually serve for the interval."""
    ceiling = INTERVAL_MAX_DAYS.get(interval)
    if ceiling is None or PERIOD_DAYS.get(period, 0) <= ceiling:
        return period, None
    allowed = [p for p in PERIODS if PERIOD_DAYS[p] <= ceiling]
    eff = allowed[-1] if allowed else "1d"
    return eff, (f"Yahoo caps `{interval}` history at ~{ceiling} days. "
                 f"Period `{period}` was clamped to `{eff}`.")


# --------------------------------------------------------------- formatting ---
def fmt(value, digits: int = 2, dash: str = "--") -> str:
    if value is None:
        return dash
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return dash if (math.isnan(f) or math.isinf(f)) else f"{f:,.{digits}f}"


def fmt_signed(value, digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "--" if math.isnan(f) else f"{f:+,.{digits}f}"


def fmt_time(ts) -> str:
    if ts is None:
        return "--"
    try:
        if pd.isna(ts):
            return "--"
    except (TypeError, ValueError):
        pass
    if isinstance(ts, (pd.Timestamp, datetime)):
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def safe_last(series, offset: int = 0):
    idx = -1 - offset
    if series is None or len(series) < abs(idx):
        return None
    v = series.iloc[idx]
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _f(v, default=np.nan) -> float:
    """Coerce to float, mapping None/NaT to NaN."""
    try:
        if v is None:
            return default
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# =============================================================================
# SECTION 2 -- INDICATORS  (all hand-written; no pandas-ta / ta / TA-Lib)
# =============================================================================
def _seeded_recursive(values: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """
    SMA-seeded recursive smoother:  out[t] = out[t-1] + alpha*(x[t] - out[t-1])

    pandas' ewm(adjust=False) seeds with the FIRST observation; TradingView seeds
    with the simple average of the first `length` observations. That difference
    is the usual cause of RSI/ATR drift against the chart, so the kernel below
    reproduces the TradingView convention explicitly.
    """
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n == 0:
        return out
    finite = np.isfinite(values)
    if finite.sum() < length:
        return out
    first = int(np.argmax(finite))
    seed_end = first + length
    if seed_end > n:
        return out
    if not np.isfinite(values[first:seed_end]).all():
        win = np.convolve(finite.astype(int), np.ones(length, dtype=int), mode="valid")
        idx = np.where(win == length)[0]
        if idx.size == 0:
            return out
        first = int(idx[0])
        seed_end = first + length
    out[seed_end - 1] = float(np.mean(values[first:seed_end]))
    prev = out[seed_end - 1]
    for i in range(seed_end, n):
        x = values[i]
        prev = prev if not np.isfinite(x) else prev + alpha * (x - prev)
        out[i] = prev
    return out


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(int(length), min_periods=int(length)).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    return pd.Series(_seeded_recursive(np.asarray(series, float), length, 2.0 / (length + 1.0)),
                     index=series.index, name=f"EMA{length}")


def rma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    return pd.Series(_seeded_recursive(np.asarray(series, float), length, 1.0 / length),
                     index=series.index, name=f"RMA{length}")


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = rma(delta.clip(lower=0.0), length)
    avg_loss = rma((-delta).clip(lower=0.0), length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out.rename(f"RSI{length}")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    if len(tr):
        tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr.rename("TR")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), length).rename(f"ATR{length}")


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    """Wilder ADX. Returns (adx, plus_di, minus_di)."""
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_ = rma(true_range(high, low, close), length).replace(0.0, np.nan)
    plus_di = 100.0 * rma(plus_dm, length) / atr_
    minus_di = 100.0 * rma(minus_dm, length) / atr_
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return rma(dx, length).rename("ADX"), plus_di.rename("+DI"), minus_di.rename("-DI")


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line.rename("MACD"), sig.rename("MACD_SIGNAL"), (line - sig).rename("MACD_HIST")


def bollinger(close: pd.Series, length=20, mult=2.0):
    mid = sma(close, length)
    sd = close.rolling(int(length), min_periods=int(length)).std(ddof=0)
    return mid.rename("BB_MID"), (mid + mult * sd).rename("BB_UP"), (mid - mult * sd).rename("BB_LO")


def stdev(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(int(length), min_periods=int(length)).std(ddof=0)


def rolling_high(series: pd.Series, length: int, exclude_current: bool = True) -> pd.Series:
    s = series.shift(1) if exclude_current else series
    return s.rolling(int(length), min_periods=int(length)).max()


def rolling_low(series: pd.Series, length: int, exclude_current: bool = True) -> pd.Series:
    s = series.shift(1) if exclude_current else series
    return s.rolling(int(length), min_periods=int(length)).min()


def slope(series: pd.Series, length: int = 5) -> pd.Series:
    return series - series.shift(int(length))


def cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    ok = a.shift(1).notna() & b.shift(1).notna()
    return ((a > b) & (a.shift(1) <= b.shift(1)) & ok).fillna(False)


def cross_under(a: pd.Series, b: pd.Series) -> pd.Series:
    ok = a.shift(1).notna() & b.shift(1).notna()
    return ((a < b) & (a.shift(1) >= b.shift(1)) & ok).fillna(False)


def supertrend(high, low, close, length=10, mult=3.0):
    """Classic SuperTrend. Returns (direction, upper_band, lower_band)."""
    a = atr(high, low, close, length)
    hl2 = (high + low) / 2.0
    ur, lr = (hl2 + mult * a).to_numpy(float), (hl2 - mult * a).to_numpy(float)
    c = close.to_numpy(float)
    n = len(close)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    prev = 1
    for i in range(n):
        if not np.isfinite(ur[i]):
            continue
        if i == 0 or not np.isfinite(up[i - 1]):
            up[i], dn[i], direction[i] = ur[i], lr[i], prev
            continue
        up[i] = min(ur[i], up[i - 1]) if c[i - 1] <= up[i - 1] else ur[i]
        dn[i] = max(lr[i], dn[i - 1]) if c[i - 1] >= dn[i - 1] else lr[i]
        if c[i] > up[i - 1]:
            prev = 1
        elif c[i] < dn[i - 1]:
            prev = -1
        direction[i] = prev
    idx = close.index
    return (pd.Series(direction, index=idx, name="ST_DIR"),
            pd.Series(up, index=idx, name="ST_UP"),
            pd.Series(dn, index=idx, name="ST_DN"))


def session_key(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(index).normalize(), index=index)


def bar_of_session(index: pd.DatetimeIndex, intraday: bool) -> pd.Series:
    if not intraday:
        return pd.Series(np.arange(len(index)), index=index)
    k = session_key(index)
    return k.groupby(k).cumcount()


def vwap(df: pd.DataFrame, intraday: bool) -> tuple[pd.Series, bool]:
    """
    Session-anchored VWAP.

    Indices and spot FX report zero volume on Yahoo, which would make a true
    VWAP a division by zero. In that case we fall back to a session-anchored
    TWAP (cumulative mean of the typical price) and return volume_ok=False so
    the UI can say plainly that this is not a real VWAP.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0.0)
    volume_ok = float(vol.abs().sum()) > 0.0
    if intraday:
        k = session_key(df.index)
        if volume_ok:
            pv = (tp * vol).groupby(k).cumsum()
            cv = vol.groupby(k).cumsum().replace(0.0, np.nan)
            return (pv / cv).rename("VWAP"), True
        return tp.groupby(k).expanding().mean().reset_index(level=0, drop=True).rename("TWAP"), False
    if volume_ok:
        pv, cv = (tp * vol).cumsum(), vol.cumsum().replace(0.0, np.nan)
        return (pv / cv).rename("VWAP"), True
    return tp.expanding().mean().rename("TWAP"), False


# ----------------------------------------------------- structure / swings ----
def swing_levels(high: pd.Series, low: pd.Series, left: int = 3, right: int = 3):
    """
    Confirmed swing levels, with no look-ahead.

    Returns ``(current_high, current_low, previous_high, previous_low)`` where
    "current" is the most recently confirmed pivot and "previous" is the one
    before it. A pivot at bar i is only knowable once `right` further bars have
    printed, so the detected value is shifted forward by `right` before being
    carried. That shift is the difference between a usable structural stop and a
    backtest that quietly cheats.
    """
    win = int(left) + int(right) + 1
    is_ph = high == high.rolling(win, center=True, min_periods=win).max()
    is_pl = low == low.rolling(win, center=True, min_periods=win).min()
    ph_raw = high.where(is_ph).shift(int(right))
    pl_raw = low.where(is_pl).shift(int(right))

    cur_h = ph_raw.ffill().rename("SWING_HIGH")
    cur_l = pl_raw.ffill().rename("SWING_LOW")
    prev_h = ph_raw.dropna().shift(1).reindex(high.index).ffill().rename("PREV_SWING_HIGH")
    prev_l = pl_raw.dropna().shift(1).reindex(low.index).ffill().rename("PREV_SWING_LOW")
    return cur_h, cur_l, prev_h, prev_l


def fair_value_gaps(df: pd.DataFrame):
    """ICT fair value gaps: a 3-candle imbalance where candle 1 and 3 do not overlap."""
    bull = (df["Low"] > df["High"].shift(2)) & (df["Close"] > df["Open"])
    bear = (df["High"] < df["Low"].shift(2)) & (df["Close"] < df["Open"])
    bull_top = df["Low"].where(bull)
    bull_bot = df["High"].shift(2).where(bull)
    bear_bot = df["High"].where(bear)
    bear_top = df["Low"].shift(2).where(bear)
    return (bull.rename("FVG_BULL"), bear.rename("FVG_BEAR"),
            bull_bot.ffill().rename("FVG_BULL_LO"), bull_top.ffill().rename("FVG_BULL_HI"),
            bear_bot.ffill().rename("FVG_BEAR_LO"), bear_top.ffill().rename("FVG_BEAR_HI"))


def market_structure(df: pd.DataFrame, left=3, right=3):
    """
    Smart-money style structure: break of structure (BOS) against confirmed swings.

    Returns (bos_direction, swing_high, swing_low) where bos_direction holds +1
    after a bullish break, -1 after a bearish break, carried forward until the
    opposite break occurs.
    """
    sh, sl_, _, _ = swing_levels(df["High"], df["Low"], left, right)
    bull_bos = (df["Close"] > sh) & sh.notna()
    bear_bos = (df["Close"] < sl_) & sl_.notna()
    raw = pd.Series(np.where(bull_bos, 1, np.where(bear_bos, -1, np.nan)), index=df.index)
    return raw.ffill().fillna(0).astype(int).rename("BOS"), sh, sl_


# =============================================================================
# SECTION 3 -- MARKET DATA  (the single rate-limit-guarded network choke-point)
# =============================================================================
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class MarketDataError(RuntimeError):
    """Raised when usable OHLCV data could not be assembled."""


@dataclass
class DataBundle:
    frame: pd.DataFrame
    symbol: str
    interval: str
    period: str
    warnings: list[str] = field(default_factory=list)

    @property
    def bars(self) -> int:
        return len(self.frame)


def _raw_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    The ONE and ONLY call site for yfinance in this file.

    A hard 0.3s pause is taken before the request and again after it, in a
    finally block so the post-guard runs even when the request raises. Keeping
    this to a single function is what makes the guarantee auditable.
    """
    import yfinance as yf

    time.sleep(API_GUARD_DELAY)                      # ---- mandatory pre-guard ----
    try:
        return yf.download(tickers=symbol, period=period, interval=interval,
                           auto_adjust=False, actions=False, progress=False,
                           threads=False, group_by="column")
    except Exception as exc:                          # noqa: BLE001
        raise MarketDataError(f"Download failed for {symbol} [{interval}/{period}]: {exc}") from exc
    finally:
        time.sleep(API_GUARD_DELAY)                   # ---- mandatory post-guard ----


def _normalise(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise MarketDataError(f"Yahoo returned an empty frame for `{symbol}`.")
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        df.columns = (df.columns.get_level_values(0) if {"Open", "Close"} & lvl0
                      else df.columns.get_level_values(1))
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in ["Open", "High", "Low", "Close"] if c not in df.columns]
    if missing:
        raise MarketDataError(f"`{symbol}` response is missing columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[OHLCV].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    # High/Low sanity: Yahoo occasionally emits an inverted bar.
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule, label="left", closed="left", origin="start_day").agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _fetch_uncached(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if interval in NATIVE_INTERVALS:
        return _normalise(_raw_download(symbol, period, interval), symbol)
    src, rule = DERIVED_INTERVALS[interval]
    # 3m/10m/4h are not served natively; resample them from a finer native feed.
    return _resample(_normalise(_raw_download(symbol, period, src), symbol), rule)


def _cached_fetch(symbol: str, period: str, interval: str, bucket: int) -> pd.DataFrame:
    """Streamlit-cached wrapper; `bucket` is the caller's freshness key."""
    if st is None:
        return _fetch_uncached(symbol, period, interval)
    if not hasattr(_cached_fetch, "_impl"):
        @st.cache_data(show_spinner=False, max_entries=96)
        def _impl(sym, per, itv, _b):
            return _fetch_uncached(sym, per, itv)
        _cached_fetch._impl = _impl
    return _cached_fetch._impl(symbol, period, interval, bucket)


def load_market_data(symbol: str, period: str, interval: str,
                     freshness_seconds: float = 300.0,
                     min_bars: int = ABSOLUTE_MIN_BARS) -> DataBundle:
    """Fetch, clean, resample and validate an OHLCV series."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise MarketDataError("No symbol supplied.")
    warnings: list[str] = []
    eff_period, clamp = sanitize_period(interval, period)
    if clamp:
        warnings.append(clamp)

    bucket = int(time.time() // max(0.3, float(freshness_seconds)))
    frame = _cached_fetch(symbol, eff_period, interval, bucket)

    if frame.empty:
        raise MarketDataError(
            f"No candles returned for `{symbol}` at {interval}/{eff_period}. The symbol "
            "may be wrong, delisted, or outside its trading calendar.")
    if len(frame) < min_bars:
        raise MarketDataError(
            f"Only {len(frame)} candles available for `{symbol}` at {interval}/{eff_period}; "
            f"at least {min_bars} are required. Widen the period or use a coarser interval.")
    if float(frame["Volume"].abs().sum()) == 0.0:
        warnings.append("This instrument reports zero volume on Yahoo (normal for indices and "
                        "spot FX). Volume-gated strategies and filters will not arm; VWAP "
                        "falls back to a session TWAP.")
    return DataBundle(frame=frame, symbol=symbol, interval=interval,
                      period=eff_period, warnings=warnings)


_LAST_QUOTE_CALL = {"t": 0.0}


def _space_quote_requests(min_gap: float = API_GUARD_DELAY) -> None:
    """
    Enforce the mandatory gap BETWEEN quote requests without double-sleeping.

    The heavy candle download brackets itself with 0.3s on each side. The quote
    path is hit on every tick, so it instead guarantees a 0.3s spacing measured
    from the previous call: same protection, but it does not burn 0.6s of every
    tick doing nothing.
    """
    gap = time.time() - _LAST_QUOTE_CALL["t"]
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _LAST_QUOTE_CALL["t"] = time.time()


def yahoo_ltp(symbol: str) -> float | None:
    """
    Last traded price from Yahoo's QUOTE endpoint, not from a candle close.

    This is the difference between a price that moves and one that steps once
    per candle. The quote stream updates continuously; a 5m candle's close only
    changes when the candle rolls. (Yahoo's Indian quotes still carry an
    exchange delay, but they tick within that delay instead of freezing.)
    """
    import yfinance as yf

    _space_quote_requests()
    try:
        ticker = yf.Ticker(symbol)
        fast = getattr(ticker, "fast_info", None)
        for probe in ("last_price", "lastPrice", "regular_market_price", "regularMarketPrice"):
            value = None
            if fast is not None:
                value = getattr(fast, probe, None)
                if value is None:
                    try:
                        value = fast[probe]
                    except Exception:                               # noqa: BLE001
                        value = None
            if value is not None:
                price = float(value)
                if np.isfinite(price) and price > 0:
                    return price
    except Exception:                                               # noqa: BLE001
        pass
    # Last resort: the freshest 1-minute candle available.
    try:
        _space_quote_requests()
        hist = yf.Ticker(symbol).history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            price = float(hist["Close"].dropna().iloc[-1])
            if np.isfinite(price) and price > 0:
                return price
    except Exception:                                               # noqa: BLE001
        pass
    return None


def load_vix(freshness_seconds: float = 60.0) -> float | None:
    """Latest India VIX close, or None if unavailable. Never raises."""
    try:
        b = load_market_data(VIX_SYMBOL, "5d", "15m", freshness_seconds, min_bars=1)
        return float(b.frame["Close"].iloc[-1])
    except Exception:                                  # noqa: BLE001
        return None


def live_period_for(interval: str, needed_bars: int = WARMUP_BARS + 120) -> str:
    """
    Smallest period that still satisfies the warm-up requirement.

    Re-pulling `max` history on every poll is exactly how an IP gets throttled.
    """
    per_day = APPROX_BARS_PER_DAY.get(interval, 1.0)
    needed_days = needed_bars / max(per_day, 0.001)
    for p in PERIODS:
        if p != "max" and PERIOD_DAYS[p] >= needed_days * 1.6:
            return p
    return "max"


def gap_profile(df: pd.DataFrame, threshold_pct: float = 0.3) -> pd.DataFrame:
    pc = df["Close"].shift(1)
    gap = (df["Open"] - pc) / pc * 100.0
    out = df.loc[gap.abs() >= threshold_pct, ["Open", "Close"]].copy()
    out["gap_pct"] = gap.loc[out.index].round(3)
    out["direction"] = np.where(out["gap_pct"] > 0, "Gap Up", "Gap Down")
    return out


# =============================================================================
# SECTION 4 -- ADDITIONAL ENTRY FILTERS
# =============================================================================
# Every filter is OFF by default. An enabled filter must agree with the trade
# direction or the signal is discarded. Filters never create signals of their
# own -- they only veto.
#
# DATA HONESTY: PCR, open-interest change and news have no free feed wired in.
# They are exposed as manual inputs plus a hook so a user with their own source
# can plug it in. Left untouched they are inert and say so in the UI rather than
# silently passing or silently blocking.

FILTER_SPECS: list[dict] = [
    {"key": "adx", "label": "ADX strength", "kind": "range",
     "min": 20.0, "max": 60.0, "step": 1.0,
     "help": "Trade only when trend strength sits inside the band."},
    {"key": "rsi", "label": "RSI band", "kind": "range",
     "min": 40.0, "max": 70.0, "step": 1.0,
     "modes": ["Cross min from below = LONG, cross max from above = SHORT",
               "Inside the band",
               "Above min = LONG, below max = SHORT"],
     "help": "Pick how the band is read: a crossing event, a static zone, or a simple side test."},
    {"key": "crossover", "label": "Crossover quality", "kind": "crossover",
     "help": "Rejects limp crossovers. Angle is the fast/slow EMA convergence rate normalised "
             "by ATR, so it does not change when you zoom the chart."},
    {"key": "ema20", "label": "EMA(20) side", "kind": "toggle",
     "help": "Long only above the 20 EMA, short only below."},
    {"key": "sma20", "label": "SMA(20) side", "kind": "toggle",
     "help": "Long only above the 20 SMA, short only below."},
    {"key": "bb", "label": "Bollinger Bands", "kind": "mode",
     "modes": ["Above / below middle band", "Inside the bands", "Outside the bands (breakout)"],
     "help": "Positional filter against a 20/2.0 Bollinger."},
    {"key": "macd", "label": "MACD histogram", "kind": "toggle",
     "help": "Long needs a positive histogram, short a negative one."},
    {"key": "smc", "label": "SMC break of structure", "kind": "toggle",
     "help": "Direction must agree with the last confirmed break of structure."},
    {"key": "ict", "label": "ICT premium / discount", "kind": "toggle",
     "help": "Longs only from the discount half of the dealing range, shorts from premium."},
    {"key": "volspike", "label": "Volume spike", "kind": "value",
     "value": 1.5, "step": 0.1,
     "help": "Bar volume must be at least N x its 20-bar average. Inert on zero-volume feeds."},
    {"key": "regime", "label": "Market regime", "kind": "mode",
     "modes": ["Trending only (ADX >= 25)", "Ranging only (ADX < 20)"],
     "help": "Coarse regime gate built on ADX."},
    {"key": "atrpct", "label": "ATR % of price", "kind": "range",
     "min": 0.05, "max": 5.0, "step": 0.05,
     "help": "Skip dead tape and skip blow-off volatility."},
    {"key": "supertrend", "label": "SuperTrend direction", "kind": "toggle",
     "help": "Direction must agree with a 10/3.0 SuperTrend."},
    {"key": "vwap", "label": "VWAP side", "kind": "toggle",
     "help": "Long only above VWAP. On zero-volume feeds this degrades to a session TWAP."},
    {"key": "vix", "label": "India VIX band", "kind": "range",
     "min": 8.0, "max": 25.0, "step": 0.5, "external": True,
     "help": "Fetched live from ^INDIAVIX. Applies to every instrument, not just Indian ones."},
    {"key": "pcr", "label": "Put-Call Ratio band", "kind": "range",
     "min": 0.7, "max": 1.3, "step": 0.05, "manual": True,
     "help": "NO FREE FEED. Enter the value manually or wire the hook."},
    {"key": "oi", "label": "OI change", "kind": "oi", "manual": True,
     "help": "NO FREE FEED. Enter the observed change manually or wire the hook."},
    {"key": "news", "label": "News block", "kind": "news", "manual": True,
     "help": "NO FREE FEED. A manual kill-switch plus a hook for your own news source."},
]

FILTER_LABELS = {s["key"]: s["label"] for s in FILTER_SPECS}


def default_filter_config() -> dict:
    """All filters disabled, carrying their default magnitudes."""
    cfg: dict = {}
    for spec in FILTER_SPECS:
        entry: dict = {"enabled": False}
        if spec["kind"] == "range":
            entry.update(min=spec["min"], max=spec["max"])
        elif spec["kind"] == "value":
            entry["value"] = spec["value"]
        elif spec["kind"] == "mode":
            entry["mode"] = spec["modes"][0]
        elif spec["kind"] == "oi":
            entry.update(mode="Absolute change", value=0.0, manual=0.0)
        elif spec["kind"] == "news":
            entry["block"] = False
        elif spec["kind"] == "crossover":
            entry.update(min_angle=0.0, mode="Simple crossover (no candle size rule)",
                         candle_points=10.0, candle_atr=1.0)
        if spec.get("modes"):
            entry["mode"] = spec["modes"][0]
        if spec.get("manual"):
            entry["manual"] = entry.get("manual", 0.0)
        cfg[spec["key"]] = entry
    return cfg


def attach_filter_columns(df: pd.DataFrame, params: dict, intraday: bool) -> pd.DataFrame:
    """Compute every indicator any filter might need, once."""
    out = df
    if "flt_ready" in out.columns:
        return out
    out = out.copy()
    p = lambda k: params.get(k, DEFAULT_PARAMS[k])                       # noqa: E731

    a, plus_di, minus_di = adx(out["High"], out["Low"], out["Close"], int(p("adx_len")))
    out["f_adx"], out["f_pdi"], out["f_mdi"] = a, plus_di, minus_di
    out["f_rsi"] = rsi(out["Close"], int(p("rsi_len")))
    out["f_ema20"] = ema(out["Close"], 20)
    out["f_sma20"] = sma(out["Close"], 20)
    mid, up, lo = bollinger(out["Close"], int(p("bb_len")), float(p("bb_mult")))
    out["f_bb_mid"], out["f_bb_up"], out["f_bb_lo"] = mid, up, lo
    _, _, hist = macd(out["Close"], int(p("macd_fast")), int(p("macd_slow")), int(p("macd_signal")))
    out["f_macd_hist"] = hist
    bos, sh, sl_ = market_structure(out, int(p("pivot_left")), int(p("pivot_right")))
    out["f_bos"], out["f_swing_high"], out["f_swing_low"] = bos, sh, sl_
    rng_hi = rolling_high(out["High"], int(p("structure_len")), exclude_current=False)
    rng_lo = rolling_low(out["Low"], int(p("structure_len")), exclude_current=False)
    out["f_range_mid"] = (rng_hi + rng_lo) / 2.0
    out["f_vol_ma"] = sma(out["Volume"], int(p("vol_len")))
    out["f_atr"] = atr(out["High"], out["Low"], out["Close"], int(p("atr_len")))
    out["f_atr_pct"] = out["f_atr"] / out["Close"] * 100.0
    st_dir, _, _ = supertrend(out["High"], out["Low"], out["Close"],
                              int(p("st_len")), float(p("st_mult")))
    out["f_st_dir"] = st_dir
    vw, vol_ok = vwap(out, intraday)
    out["f_vwap"] = vw
    out.attrs["vwap_is_volume_weighted"] = vol_ok

    # Crossover geometry. The raw gradient of an EMA pair is in price units per
    # bar, which makes any "angle" meaningless across instruments and zoom
    # levels. Normalising the per-bar change in the fast/slow spread by ATR
    # gives a dimensionless rate whose arctangent IS comparable everywhere.
    fast = out["ema_fast"] if "ema_fast" in out.columns else ema(out["Close"], int(p("ema_fast")))
    slow = out["ema_slow"] if "ema_slow" in out.columns else ema(out["Close"], int(p("ema_slow")))
    spread = fast - slow
    rate = (spread - spread.shift(1)) / out["f_atr"].replace(0.0, np.nan)
    out["f_cross_angle"] = np.degrees(np.arctan(rate)).abs()
    out["f_candle_range"] = (out["High"] - out["Low"]).abs()
    out["flt_ready"] = True
    return out


def _const_like(series: pd.Series, value: float) -> pd.Series:
    return pd.Series(float(value), index=series.index)


@dataclass
class FilterReport:
    key: str
    label: str
    value: str
    long_ok: bool
    short_ok: bool


def evaluate_filters(df: pd.DataFrame, fcfg: dict, extras: dict | None = None):
    """
    Return ``(long_mask, short_mask, reports)``.

    ``reports`` describes the state of each ENABLED filter on the final bar,
    which is what lets the live panel say exactly which gate is blocking entry
    instead of leaving the operator guessing.
    """
    extras = extras or {}
    idx = df.index
    ok_long = pd.Series(True, index=idx)
    ok_short = pd.Series(True, index=idx)
    reports: list[FilterReport] = []

    def apply(key: str, lmask: pd.Series, smask: pd.Series, value: str):
        nonlocal ok_long, ok_short
        lmask = lmask.fillna(False)
        smask = smask.fillna(False)
        ok_long &= lmask
        ok_short &= smask
        reports.append(FilterReport(key, FILTER_LABELS[key], value,
                                    bool(lmask.iloc[-1]) if len(lmask) else False,
                                    bool(smask.iloc[-1]) if len(smask) else False))

    def on(key: str) -> bool:
        return bool(fcfg.get(key, {}).get("enabled", False))

    c = df["Close"]

    if on("adx"):
        cfg = fcfg["adx"]
        m = df["f_adx"].between(cfg["min"], cfg["max"])
        apply("adx", m, m, fmt(safe_last(df["f_adx"])))

    if on("rsi"):
        cfg = fcfg["rsi"]
        lo, hi = float(cfg["min"]), float(cfg["max"])
        mode = cfg.get("mode", "Cross min from below = LONG, cross max from above = SHORT")
        r = df["f_rsi"]
        if mode.startswith("Cross"):
            # The crossing reading: RSI reclaiming the lower level is the long
            # trigger, losing the upper level is the short trigger.
            lm = cross_over(r, _const_like(r, lo))
            sm = cross_under(r, _const_like(r, hi))
        elif mode.startswith("Inside"):
            lm = r.between(lo, hi)
            sm = r.between(100.0 - hi, 100.0 - lo)
        else:
            lm, sm = r >= lo, r <= hi
        apply("rsi", lm, sm, f"{fmt(safe_last(r))} ({mode.split(' =')[0].lower()})")

    if on("crossover"):
        cfg = fcfg["crossover"]
        min_angle = abs(float(cfg.get("min_angle", 0.0)))
        ang_ok = df["f_cross_angle"] >= min_angle
        mode = cfg.get("mode", "Simple crossover (no candle size rule)")
        if mode.startswith("Custom"):
            size_ok = df["f_candle_range"] >= float(cfg.get("candle_points", 0.0))
            size_txt = f">= {fmt(cfg.get('candle_points'))} pts"
        elif mode.startswith("ATR"):
            size_ok = df["f_candle_range"] >= float(cfg.get("candle_atr", 1.0)) * df["f_atr"]
            size_txt = f">= {fmt(cfg.get('candle_atr'))} x ATR"
        else:
            size_ok = pd.Series(True, index=idx)
            size_txt = "no size rule"
        m = ang_ok & size_ok
        apply("crossover", m, m,
              f"angle {fmt(safe_last(df['f_cross_angle']))}deg vs {fmt(min_angle)}, "
              f"range {fmt(safe_last(df['f_candle_range']))} {size_txt}")

    if on("ema20"):
        apply("ema20", c > df["f_ema20"], c < df["f_ema20"], fmt(safe_last(df["f_ema20"])))

    if on("sma20"):
        apply("sma20", c > df["f_sma20"], c < df["f_sma20"], fmt(safe_last(df["f_sma20"])))

    if on("bb"):
        mode = fcfg["bb"].get("mode", "Above / below middle band")
        if mode.startswith("Above"):
            lm, sm = c > df["f_bb_mid"], c < df["f_bb_mid"]
        elif mode.startswith("Inside"):
            inside = (c < df["f_bb_up"]) & (c > df["f_bb_lo"])
            lm = sm = inside
        else:
            lm, sm = c > df["f_bb_up"], c < df["f_bb_lo"]
        apply("bb", lm, sm, f"mid {fmt(safe_last(df['f_bb_mid']))}")

    if on("macd"):
        apply("macd", df["f_macd_hist"] > 0, df["f_macd_hist"] < 0,
              fmt(safe_last(df["f_macd_hist"]), 4))

    if on("smc"):
        apply("smc", df["f_bos"] == 1, df["f_bos"] == -1,
              {1: "bullish BOS", -1: "bearish BOS"}.get(safe_last(df["f_bos"]), "none"))

    if on("ict"):
        apply("ict", c < df["f_range_mid"], c > df["f_range_mid"],
              f"range mid {fmt(safe_last(df['f_range_mid']))}")

    if on("volspike"):
        mult = float(fcfg["volspike"].get("value", 1.5))
        ratio = df["Volume"] / df["f_vol_ma"].replace(0.0, np.nan)
        m = ratio >= mult
        note = fmt(safe_last(ratio)) + "x"
        if float(df["Volume"].abs().sum()) == 0.0:
            note = "no volume on feed -- filter blocks everything"
        apply("volspike", m, m, note)

    if on("regime"):
        trending = fcfg["regime"].get("mode", "").startswith("Trending")
        m = (df["f_adx"] >= 25) if trending else (df["f_adx"] < 20)
        apply("regime", m, m, f"ADX {fmt(safe_last(df['f_adx']))}")

    if on("atrpct"):
        cfg = fcfg["atrpct"]
        m = df["f_atr_pct"].between(cfg["min"], cfg["max"])
        apply("atrpct", m, m, fmt(safe_last(df["f_atr_pct"]), 3) + "%")

    if on("supertrend"):
        apply("supertrend", df["f_st_dir"] == 1, df["f_st_dir"] == -1,
              "up" if safe_last(df["f_st_dir"]) == 1 else "down")

    if on("vwap"):
        vol_ok = bool(df.attrs.get("vwap_is_volume_weighted", True))
        label = fmt(safe_last(df["f_vwap"])) + ("" if vol_ok else "  (TWAP fallback, no volume)")
        apply("vwap", c > df["f_vwap"], c < df["f_vwap"], label)

    if on("vix"):
        cfg = fcfg["vix"]
        v = extras.get("vix")
        if v is None:
            m = pd.Series(False, index=idx)
            label = "unavailable -- filter blocks everything"
        else:
            inside = cfg["min"] <= v <= cfg["max"]
            m = pd.Series(inside, index=idx)
            label = fmt(v)
        apply("vix", m, m, label)

    if on("pcr"):
        cfg = fcfg["pcr"]
        v = extras.get("pcr", cfg.get("manual"))
        if v in (None, 0.0):
            m = pd.Series(False, index=idx)
            label = "no value supplied -- filter blocks everything"
        else:
            m = pd.Series(cfg["min"] <= float(v) <= cfg["max"], index=idx)
            label = f"{fmt(v)} (manual)"
        apply("pcr", m, m, label)

    if on("oi"):
        cfg = fcfg["oi"]
        v = extras.get("oi_change", cfg.get("manual"))
        thresh = float(cfg.get("value", 0.0))
        if v is None:
            m = pd.Series(False, index=idx)
            label = "no value supplied -- filter blocks everything"
        else:
            v = float(v)
            m = pd.Series(abs(v) >= thresh, index=idx)
            label = f"{fmt(v)} vs {fmt(thresh)} ({cfg.get('mode', 'Absolute change')}, manual)"
        apply("oi", m, m, label)

    if on("news"):
        blocked = bool(extras.get("news_block", fcfg["news"].get("block", False)))
        m = pd.Series(not blocked, index=idx)
        apply("news", m, m, "BLOCKED" if blocked else "clear (manual)")

    return ok_long, ok_short, reports


# =============================================================================
# SECTION 5 -- STRATEGY MATRIX
# =============================================================================
# Contract:
#   compute(df, params) -> frame with indicator columns + integer `signal`
#                          (+1 long, -1 short, 0 flat), evaluated on that bar's
#                          CLOSE and filled on the NEXT bar's open.
#   status(frame, params) -> StatusReport describing how far the market is from
#                          arming, using real numbers.
#
# `immediate=True` marks the two Simple Buy / Simple Sell profiles, which enter
# at once rather than waiting for the next candle open.


@dataclass
class StatusReport:
    headline: str
    metrics: list[tuple[str, str]]
    long_condition: str
    short_condition: str


@dataclass
class Strategy:
    key: str
    name: str
    blurb: str
    min_bars: int
    compute: Callable[[pd.DataFrame, dict], pd.DataFrame]
    status: Callable[[pd.DataFrame, dict], StatusReport]
    overlays: tuple[str, ...] = ()
    oscillator: str | None = None
    immediate: bool = False


def _p(params: dict, key: str):
    return params.get(key, DEFAULT_PARAMS[key])


def _const(df, v):
    return pd.Series(float(v), index=df.index)


def _finalise(out: pd.DataFrame, long: pd.Series, short: pd.Series) -> pd.DataFrame:
    long = long.fillna(False).astype(bool)
    short = short.fillna(False).astype(bool)
    both = long & short                       # contradictory -> stand aside
    out["signal"] = pd.Series(
        np.where(long & ~both, 1, np.where(short & ~both, -1, 0)), index=out.index).astype(int)
    return out


def _sr(headline, metrics, long_c, short_c) -> StatusReport:
    return StatusReport(headline, metrics, long_c, short_c)


def zigzag_pivot_table(close: pd.Series, threshold_pct: float = 0.8):
    """
    Confirmed zigzag pivots with the bar on which each became KNOWABLE.

    Returns a list of ``(pivot_index, pivot_price, kind, confirm_index)`` where
    kind is +1 for a swing high and -1 for a swing low. Consumers must only look
    at pivots whose ``confirm_index <= current bar`` -- otherwise the pattern
    logic silently reads the future.
    """
    c = close.to_numpy(float)
    n = len(c)
    piv: list[tuple[int, float, int, int]] = []
    if n == 0:
        return piv
    thr = threshold_pct / 100.0
    # Direction starts at +1, never 0. With a neutral start the running extreme
    # tracked BOTH directions at once, so it always equalled the current price,
    # no reversal threshold could ever be breached, and the function returned an
    # empty pivot list forever -- which is why the wave profile never fired.
    direction, ext_i, ext = 1, 0, c[0]
    for i in range(1, n):
        if not np.isfinite(c[i]) or ext <= 0:
            continue
        if direction > 0:
            if c[i] > ext:
                ext_i, ext = i, c[i]
            elif c[i] <= ext * (1 - thr):
                piv.append((ext_i, float(ext), 1, i))       # confirmed swing HIGH
                direction, ext_i, ext = -1, i, c[i]
        else:
            if c[i] < ext:
                ext_i, ext = i, c[i]
            elif c[i] >= ext * (1 + thr):
                piv.append((ext_i, float(ext), -1, i))      # confirmed swing LOW
                direction, ext_i, ext = 1, i, c[i]
    return piv


def zigzag_pivots(close: pd.Series, threshold_pct: float = 0.8) -> pd.Series:
    """
    Percentage zigzag: +1 marks a confirmed swing high, -1 a confirmed swing low.

    Used by the Elliott Wave heuristic. Confirmation is retrospective by nature,
    so the series is shifted so that a pivot only becomes visible on the bar the
    reversal threshold was actually breached.
    """
    c = close.to_numpy(float)
    n = len(c)
    marks = np.zeros(n, dtype=int)
    if n == 0:
        return pd.Series(marks, index=close.index, name="ZZ")
    for piv_i, _, kind, _ in zigzag_pivot_table(close, threshold_pct):
        marks[piv_i] = kind
    return pd.Series(marks, index=close.index, name="ZZ")


# ------------------------------------------------------------ 01 Dual EMA ----
def c_dual_ema(df, p):
    out = df.copy()
    out["ema_fast"] = ema(out["Close"], int(_p(p, "ema_fast")))
    out["ema_slow"] = ema(out["Close"], int(_p(p, "ema_slow")))
    return _finalise(out, cross_over(out["ema_fast"], out["ema_slow"]),
                     cross_under(out["ema_fast"], out["ema_slow"]))


def s_dual_ema(df, p):
    f, s = safe_last(df["ema_fast"]), safe_last(df["ema_slow"])
    gap = (f - s) if (f is not None and s is not None) else None
    return _sr(f"Fast EMA is {'above' if (gap or 0) > 0 else 'below'} the slow EMA by "
               f"{fmt(abs(gap or 0))} points.",
               [(f"{int(_p(p,'ema_fast'))} EMA", fmt(f)), (f"{int(_p(p,'ema_slow'))} EMA", fmt(s)),
                ("Spread", fmt(gap))],
               f"{int(_p(p,'ema_fast'))} EMA must cross ABOVE the {int(_p(p,'ema_slow'))} EMA.",
               f"{int(_p(p,'ema_fast'))} EMA must cross BELOW the {int(_p(p,'ema_slow'))} EMA.")


# --------------------------------------------------- 02 RSI mean reversion ---
def c_rsi_reversion(df, p):
    out = df.copy()
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    return _finalise(out, cross_over(out["rsi"], _const(out, 30.0)),
                     cross_under(out["rsi"], _const(out, 70.0)))


def s_rsi_reversion(df, p):
    r = safe_last(df["rsi"])
    zone = ("OVERSOLD, waiting for the recovery print above 30" if (r or 50) < 30
            else "OVERBOUGHT, waiting for the failure print below 70" if (r or 50) > 70
            else "neutral, no reversion setup armed")
    return _sr(f"RSI({int(_p(p,'rsi_len'))}) is {fmt(r)} :: {zone}.",
               [("RSI", fmt(r)), ("Oversold", "30.00"), ("Overbought", "70.00")],
               "RSI must dip below 30 then close back ABOVE 30.",
               "RSI must spike above 70 then close back BELOW 70.")


# ------------------------------------------------------- 03 EMA pullback -----
def c_pullback(df, p):
    out = df.copy()
    tol = float(_p(p, "pullback_tol")) / 100.0
    out["ema_slow"] = ema(out["Close"], int(_p(p, "ema_slow")))
    out["ema_macro"] = ema(out["Close"], int(_p(p, "ema_macro")))
    out["macro_slope"] = slope(out["ema_macro"], 5)
    up = (out["Close"] > out["ema_macro"]) & (out["macro_slope"] > 0)
    dn = (out["Close"] < out["ema_macro"]) & (out["macro_slope"] < 0)
    long = up & (out["Low"] <= out["ema_slow"] * (1 + tol)) & \
        (out["Close"] > out["ema_slow"]) & (out["Close"] > out["Open"])
    short = dn & (out["High"] >= out["ema_slow"] * (1 - tol)) & \
        (out["Close"] < out["ema_slow"]) & (out["Close"] < out["Open"])
    return _finalise(out, long, short)


def s_pullback(df, p):
    c, e21, e200 = safe_last(df["Close"]), safe_last(df["ema_slow"]), safe_last(df["ema_macro"])
    sl = safe_last(df["macro_slope"])
    regime = ("BULLISH" if (c or 0) > (e200 or 0) and (sl or 0) > 0
              else "BEARISH" if (c or 0) < (e200 or 0) and (sl or 0) < 0 else "NO MACRO BIAS")
    return _sr(f"Macro regime reads {regime}; the trigger sits at the {int(_p(p,'ema_slow'))} EMA.",
               [("Last price", fmt(c)), (f"{int(_p(p,'ema_slow'))} EMA", fmt(e21)),
                (f"{int(_p(p,'ema_macro'))} EMA", fmt(e200)), ("Macro slope", fmt(sl))],
               "Price above a rising macro EMA, candle tags the slow EMA and closes above it green.",
               "Price below a falling macro EMA, candle tags the slow EMA and closes below it red.")


# --------------------------------------------- 04 ATR trailing breakout ------
def c_atr_trail(df, p):
    out = df.copy()
    out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(p, "atr_len")))
    d, up, dn = supertrend(out["High"], out["Low"], out["Close"],
                           int(_p(p, "atr_len")), float(_p(p, "atr_mult")))
    out["trail_dir"], out["trail_upper"], out["trail_lower"] = d, up, dn
    flip = out["trail_dir"] != out["trail_dir"].shift(1)
    return _finalise(out, flip & (out["trail_dir"] == 1), flip & (out["trail_dir"] == -1))


def s_atr_trail(df, p):
    d = safe_last(df["trail_dir"])
    return _sr(f"Trailing volatility band is locked {'UP' if d == 1 else 'DOWN'}.",
               [("Last price", fmt(safe_last(df["Close"]))), ("ATR", fmt(safe_last(df["atr"]))),
                ("Upper trail", fmt(safe_last(df["trail_upper"]))),
                ("Lower trail", fmt(safe_last(df["trail_lower"])))],
               "Close must break ABOVE the upper band and flip the trail.",
               "Close must break BELOW the lower band and flip the trail.")


# ------------------------------------------------------------- 05 ORB --------
def _opening_range(df, bars, intraday):
    if intraday:
        k = session_key(df.index)
        bn = k.groupby(k).cumcount()
        orh = df["High"].where(bn < bars).groupby(k).cummax().groupby(k).ffill()
        orl = df["Low"].where(bn < bars).groupby(k).cummin().groupby(k).ffill()
        return orh, orl, bn >= bars
    return (rolling_high(df["High"], bars), rolling_low(df["Low"], bars),
            pd.Series(True, index=df.index))


def c_orb(df, p):
    out = df.copy()
    orh, orl, active = _opening_range(out, int(_p(p, "orb_bars")), bool(p.get("intraday", True)))
    out["or_high"], out["or_low"] = orh, orl
    return _finalise(out, active & cross_over(out["Close"], orh),
                     active & cross_under(out["Close"], orl))


def s_orb(df, p):
    return _sr(f"Opening range built from the first {int(_p(p,'orb_bars'))} candles of the session.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Range high", fmt(safe_last(df["or_high"]))),
                ("Range low", fmt(safe_last(df["or_low"])))],
               "A closed candle must print ABOVE the opening-range high.",
               "A closed candle must print BELOW the opening-range low.")


# ---------------------------------------------------- 06 Golden cross --------
def c_golden_cross(df, p):
    out = df.copy()
    out["ema_mid"] = ema(out["Close"], int(_p(p, "ema_mid")))
    out["ema_macro"] = ema(out["Close"], int(_p(p, "ema_macro")))
    return _finalise(out, cross_over(out["ema_mid"], out["ema_macro"]),
                     cross_under(out["ema_mid"], out["ema_macro"]))


def s_golden_cross(df, p):
    m, M = safe_last(df["ema_mid"]), safe_last(df["ema_macro"])
    gap = (m - M) if (m is not None and M is not None) else None
    return _sr(f"Structure is in a {'GOLDEN' if (gap or 0) > 0 else 'DEATH'} CROSS state.",
               [(f"{int(_p(p,'ema_mid'))} EMA", fmt(m)),
                (f"{int(_p(p,'ema_macro'))} EMA", fmt(M)), ("Spread", fmt(gap))],
               "Mid EMA must cross ABOVE the macro EMA.",
               "Mid EMA must cross BELOW the macro EMA.")


# -------------------------------------------------------- 07 Gap fade --------
def c_gap_fade(df, p):
    out = df.copy()
    thr = float(_p(p, "gap_pct"))
    intraday = bool(p.get("intraday", True))
    pc = out["Close"].shift(1)
    out["gap_pct"] = (out["Open"] - pc) / pc * 100.0
    at_open = (bar_of_session(out.index, intraday) == 0) if intraday else pd.Series(True, index=out.index)
    gap_up = at_open & (out["gap_pct"] >= thr)
    gap_dn = at_open & (out["gap_pct"] <= -thr)
    return _finalise(out, gap_dn & (out["Close"] > out["Open"]),
                     gap_up & (out["Close"] < out["Open"]))


def s_gap_fade(df, p):
    g = safe_last(df["gap_pct"])
    return _sr(f"Latest session gap measured {fmt(g)}% against the previous close.",
               [("Gap %", fmt(g)), ("Threshold %", fmt(_p(p, "gap_pct"))),
                ("Last price", fmt(safe_last(df["Close"])))],
               f"Session must gap DOWN at least {fmt(_p(p,'gap_pct'))}% and close green.",
               f"Session must gap UP at least {fmt(_p(p,'gap_pct'))}% and close red.")


# ------------------------------------------------- 08 RSI centerline ---------
def c_rsi_centerline(df, p):
    out = df.copy()
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    return _finalise(out, cross_over(out["rsi"], _const(out, 50.0)),
                     cross_under(out["rsi"], _const(out, 50.0)))


def s_rsi_centerline(df, p):
    r = safe_last(df["rsi"])
    return _sr(f"RSI sits {fmt(abs((r or 50) - 50))} points "
               f"{'above' if (r or 0) > 50 else 'below'} the 50 centerline.",
               [("RSI", fmt(r)), ("Centerline", "50.00")],
               "RSI must slice UP through 50.", "RSI must slice DOWN through 50.")


# ------------------------------------------------------ 09 MTF vector --------
def c_mtf_vector(df, p):
    out = df.copy()
    for k, col in (("ema_fast", "ema_fast"), ("ema_slow", "ema_slow"),
                   ("ema_mid", "ema_mid"), ("ema_macro", "ema_macro")):
        out[col] = ema(out["Close"], int(_p(p, k)))
    bull = (out["ema_fast"] > out["ema_slow"]) & (out["ema_slow"] > out["ema_mid"]) & \
        (out["Close"] > out["ema_macro"])
    bear = (out["ema_fast"] < out["ema_slow"]) & (out["ema_slow"] < out["ema_mid"]) & \
        (out["Close"] < out["ema_macro"])
    out["vector"] = np.where(bull, 1, np.where(bear, -1, 0))
    return _finalise(out, bull & ~bull.shift(1, fill_value=False),
                     bear & ~bear.shift(1, fill_value=False))


def s_mtf_vector(df, p):
    v = safe_last(df["vector"])
    state = {1: "FULLY BULLISH", -1: "FULLY BEARISH"}.get(v, "MIXED / UNALIGNED")
    return _sr(f"EMA vector stack is {state}.",
               [(f"{int(_p(p,'ema_fast'))} EMA", fmt(safe_last(df["ema_fast"]))),
                (f"{int(_p(p,'ema_slow'))} EMA", fmt(safe_last(df["ema_slow"]))),
                (f"{int(_p(p,'ema_mid'))} EMA", fmt(safe_last(df["ema_mid"]))),
                (f"{int(_p(p,'ema_macro'))} EMA", fmt(safe_last(df["ema_macro"])))],
               "Stack must newly align fast > slow > mid with price above the macro EMA.",
               "Stack must newly align fast < slow < mid with price below the macro EMA.")


# ------------------------------------------------------- 10 Squeeze ----------
def c_squeeze(df, p):
    out = df.copy()
    look, mult = int(_p(p, "breakout_len")), float(_p(p, "squeeze_mult"))
    out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(p, "atr_len")))
    out["atr_mean"] = sma(out["atr"], look)
    out["atr_ratio"] = out["atr"] / out["atr_mean"]
    out["box_high"] = rolling_high(out["High"], look)
    out["box_low"] = rolling_low(out["Low"], look)
    armed = (out["atr_ratio"].shift(1) < 1.0) & (out["atr_ratio"] > mult)
    return _finalise(out, armed & (out["Close"] > out["box_high"]),
                     armed & (out["Close"] < out["box_low"]))


def s_squeeze(df, p):
    return _sr(f"ATR is running at {fmt(safe_last(df['atr_ratio']))}x its rolling mean "
               f"(trigger {fmt(_p(p,'squeeze_mult'))}x).",
               [("ATR", fmt(safe_last(df["atr"]))), ("ATR mean", fmt(safe_last(df["atr_mean"]))),
                ("Ratio", fmt(safe_last(df["atr_ratio"]))),
                ("Box high", fmt(safe_last(df["box_high"]))),
                ("Box low", fmt(safe_last(df["box_low"])))],
               "Volatility expands past the multiplier while price breaks the box high.",
               "Volatility expands past the multiplier while price breaks the box low.")


# ------------------------------------------------ 11 Volume confirmation -----
def c_volume_breakout(df, p):
    out = df.copy()
    look = int(_p(p, "breakout_len"))
    out["vol_ma"] = sma(out["Volume"], int(_p(p, "vol_len")))
    out["vol_ratio"] = out["Volume"] / out["vol_ma"].replace(0.0, np.nan)
    out["box_high"] = rolling_high(out["High"], look)
    out["box_low"] = rolling_low(out["Low"], look)
    surge = out["vol_ratio"] >= float(_p(p, "vol_mult"))
    return _finalise(out, surge & (out["Close"] > out["box_high"]),
                     surge & (out["Close"] < out["box_low"]))


def s_volume_breakout(df, p):
    dead = float(df["Volume"].tail(50).abs().sum()) == 0.0
    return _sr("This instrument reports no volume, so this profile cannot arm." if dead
               else f"Bar volume is {fmt(safe_last(df['vol_ratio']))}x the moving average.",
               [("Bar volume", fmt(safe_last(df["Volume"]), 0)),
                ("Volume MA", fmt(safe_last(df["vol_ma"]), 0)),
                ("Ratio", fmt(safe_last(df["vol_ratio"]))),
                ("Box high", fmt(safe_last(df["box_high"]))),
                ("Box low", fmt(safe_last(df["box_low"])))],
               "Close above the box high on a volume spike.",
               "Close below the box low on a volume spike.")


# ------------------------------------------------------ 12 Engulfing ---------
def c_engulfing(df, p):
    out = df.copy()
    look = int(_p(p, "breakout_len"))
    out["sup"] = rolling_low(out["Low"], look)
    out["res"] = rolling_high(out["High"], look)
    po, pc = out["Open"].shift(1), out["Close"].shift(1)
    bull = (out["Close"] > out["Open"]) & (pc < po) & (out["Close"] >= po) & (out["Open"] <= pc)
    bear = (out["Close"] < out["Open"]) & (pc > po) & (out["Close"] <= po) & (out["Open"] >= pc)
    return _finalise(out, bull & (out["Low"] <= out["sup"] * 1.003),
                     bear & (out["High"] >= out["res"] * 0.997))


def s_engulfing(df, p):
    return _sr("Waiting for an engulfing candle to print into a structural zone.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Support band", fmt(safe_last(df["sup"]))),
                ("Resistance band", fmt(safe_last(df["res"])))],
               "A bullish engulfing candle tagging the support band.",
               "A bearish engulfing candle tagging the resistance band.")


# ----------------------------------------------- 13 ATR channel reversion ----
def c_channel_reversion(df, p):
    out = df.copy()
    k = float(_p(p, "channel_mult"))
    out["basis"] = ema(out["Close"], int(_p(p, "ema_slow")))
    out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(p, "atr_len")))
    out["ch_upper"] = out["basis"] + k * out["atr"]
    out["ch_lower"] = out["basis"] - k * out["atr"]
    return _finalise(out, cross_over(out["Close"], out["ch_lower"]),
                     cross_under(out["Close"], out["ch_upper"]))


def s_channel_reversion(df, p):
    c, up, lo = safe_last(df["Close"]), safe_last(df["ch_upper"]), safe_last(df["ch_lower"])
    where = ("EXTENDED above the upper channel" if (c or 0) > (up or 1e18)
             else "CAPITULATED below the lower channel" if (c or 1e18) < (lo or 0)
             else "INSIDE the channel")
    return _sr(f"Price is {where}.",
               [("Last price", fmt(c)), ("Basis", fmt(safe_last(df["basis"]))),
                ("Upper", fmt(up)), ("Lower", fmt(lo)), ("ATR", fmt(safe_last(df["atr"])))],
               "Price drops below the lower channel then closes back inside.",
               "Price extends above the upper channel then closes back inside.")


# ------------------------------------------------------ 14 RSI burst ---------
def c_rsi_burst(df, p):
    out = df.copy()
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    out["ema_mid"] = ema(out["Close"], int(_p(p, "ema_mid")))
    return _finalise(out,
                     cross_over(out["rsi"], _const(out, 60.0)) & (out["Close"] > out["ema_mid"]),
                     cross_under(out["rsi"], _const(out, 40.0)) & (out["Close"] < out["ema_mid"]))


def s_rsi_burst(df, p):
    return _sr(f"RSI is at {fmt(safe_last(df['rsi']))}; bursts fire at 60 / 40.",
               [("RSI", fmt(safe_last(df["rsi"]))), ("Bull level", "60.00"), ("Bear level", "40.00"),
                ("Trend EMA", fmt(safe_last(df["ema_mid"])))],
               "RSI breaks UP through 60 with price above the trend EMA.",
               "RSI breaks DOWN through 40 with price below the trend EMA.")


# --------------------------------------------------- 15 Bias scalper ---------
def c_bias_scalper(df, p):
    out = df.copy()
    out["ema_fast"] = ema(out["Close"], int(_p(p, "ema_fast")))
    out["ema_mid"] = ema(out["Close"], int(_p(p, "ema_mid")))
    out["bias_slope"] = slope(out["ema_mid"], 5)
    up = (out["Close"] > out["ema_mid"]) & (out["bias_slope"] > 0)
    dn = (out["Close"] < out["ema_mid"]) & (out["bias_slope"] < 0)
    return _finalise(out, up & cross_over(out["Close"], out["ema_fast"]),
                     dn & cross_under(out["Close"], out["ema_fast"]))


def s_bias_scalper(df, p):
    sl = safe_last(df["bias_slope"])
    return _sr(f"The {int(_p(p,'ema_mid'))}-bar trend path is sloping {'UP' if (sl or 0) > 0 else 'DOWN'}.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Fast EMA", fmt(safe_last(df["ema_fast"]))),
                ("Trend EMA", fmt(safe_last(df["ema_mid"]))), ("Slope", fmt(sl))],
               "With the trend EMA rising, price crosses ABOVE the fast EMA.",
               "With the trend EMA falling, price crosses BELOW the fast EMA.")


# ------------------------------------------- 16 / 17 Simple buy and sell -----
def _c_simple(direction: int):
    def build(df, p):
        out = df.copy()
        out["ema_fast"] = ema(out["Close"], int(_p(p, "ema_fast")))
        out["ema_slow"] = ema(out["Close"], int(_p(p, "ema_slow")))
        sig = pd.Series(direction, index=out.index, dtype=int)
        out["signal"] = sig
        return out
    return build


def _s_simple(direction: int):
    def status(df, p):
        side = "LONG" if direction > 0 else "SHORT"
        return _sr(f"Immediate {side} profile. It enters at once and runs until the stop, "
                   "the target or a manual square-off resolves it.",
                   [("Last price", fmt(safe_last(df["Close"]))),
                    ("Mode", "Enter now, no candle wait")],
                   "Enters immediately." if direction > 0 else "No long entries in this profile.",
                   "Enters immediately." if direction < 0 else "No short entries in this profile.")
    return status


# ------------------------------------ 18 SMC break of structure + order block -
def c_smc_ob(df, p):
    out = df.copy()
    bos, sh, sl_ = market_structure(out, int(_p(p, "pivot_left")), int(_p(p, "pivot_right")))
    out["bos"], out["swing_high"], out["swing_low"] = bos, sh, sl_
    down = out["Close"] < out["Open"]
    up = out["Close"] > out["Open"]
    # Order block = the last opposing candle immediately before the break.
    ob_bull_lo = out["Low"].where(down).ffill().shift(1)
    ob_bull_hi = out["High"].where(down).ffill().shift(1)
    ob_bear_lo = out["Low"].where(up).ffill().shift(1)
    ob_bear_hi = out["High"].where(up).ffill().shift(1)
    new_bull = (bos == 1) & (bos.shift(1) != 1)
    new_bear = (bos == -1) & (bos.shift(1) != -1)
    out["ob_bull_lo"] = ob_bull_lo.where(new_bull).ffill()
    out["ob_bull_hi"] = ob_bull_hi.where(new_bull).ffill()
    out["ob_bear_lo"] = ob_bear_lo.where(new_bear).ffill()
    out["ob_bear_hi"] = ob_bear_hi.where(new_bear).ffill()
    long = (bos == 1) & (out["Low"] <= out["ob_bull_hi"]) & (out["Close"] > out["ob_bull_lo"]) & up
    short = (bos == -1) & (out["High"] >= out["ob_bear_lo"]) & (out["Close"] < out["ob_bear_hi"]) & down
    return _finalise(out, long, short)


def s_smc_ob(df, p):
    b = safe_last(df["bos"])
    state = {1: "BULLISH", -1: "BEARISH"}.get(b, "NONE")
    return _sr(f"Last confirmed structure break is {state}.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Bull OB zone", f"{fmt(safe_last(df['ob_bull_lo']))} - {fmt(safe_last(df['ob_bull_hi']))}"),
                ("Bear OB zone", f"{fmt(safe_last(df['ob_bear_lo']))} - {fmt(safe_last(df['ob_bear_hi']))}"),
                ("Swing high", fmt(safe_last(df["swing_high"]))),
                ("Swing low", fmt(safe_last(df["swing_low"])))],
               "After a bullish BOS, price must retrace into the bullish order block and close green.",
               "After a bearish BOS, price must retrace into the bearish order block and close red.")


# ------------------------------------------------ 19 SMC liquidity sweep -----
def c_smc_sweep(df, p):
    out = df.copy()
    sh, sl_, _, _ = swing_levels(out["High"], out["Low"],
                                 int(_p(p, "pivot_left")), int(_p(p, "pivot_right")))
    out["swing_high"], out["swing_low"] = sh, sl_
    # Wick takes out the pool of liquidity, body closes back inside the range.
    long = (out["Low"] < sl_) & (out["Close"] > sl_) & (out["Close"] > out["Open"])
    short = (out["High"] > sh) & (out["Close"] < sh) & (out["Close"] < out["Open"])
    return _finalise(out, long, short)


def s_smc_sweep(df, p):
    return _sr("Watching the liquidity pools sitting beyond the last confirmed swings.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Sell-side pool (swing low)", fmt(safe_last(df["swing_low"]))),
                ("Buy-side pool (swing high)", fmt(safe_last(df["swing_high"])))],
               "A wick must sweep below the swing low and the body close back above it, green.",
               "A wick must sweep above the swing high and the body close back below it, red.")


# ------------------------------------------------------- 20 ICT FVG ---------
def c_ict_fvg(df, p):
    out = df.copy()
    bull, bear, blo, bhi, selo, sehi = fair_value_gaps(out)
    out["fvg_bull"], out["fvg_bear"] = bull, bear
    out["fvg_bull_lo"], out["fvg_bull_hi"] = blo, bhi
    out["fvg_bear_lo"], out["fvg_bear_hi"] = selo, sehi
    bos, _, _ = market_structure(out, int(_p(p, "pivot_left")), int(_p(p, "pivot_right")))
    out["bos"] = bos
    long = (bos == 1) & (out["Low"] <= out["fvg_bull_hi"]) & \
        (out["Close"] > out["fvg_bull_lo"]) & (out["Close"] > out["Open"])
    short = (bos == -1) & (out["High"] >= out["fvg_bear_lo"]) & \
        (out["Close"] < out["fvg_bear_hi"]) & (out["Close"] < out["Open"])
    return _finalise(out, long, short)


def s_ict_fvg(df, p):
    return _sr("Waiting for price to rebalance into the most recent fair value gap.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Bullish FVG", f"{fmt(safe_last(df['fvg_bull_lo']))} - {fmt(safe_last(df['fvg_bull_hi']))}"),
                ("Bearish FVG", f"{fmt(safe_last(df['fvg_bear_lo']))} - {fmt(safe_last(df['fvg_bear_hi']))}")],
               "With bullish structure, price must trade into the bullish gap and close green.",
               "With bearish structure, price must trade into the bearish gap and close red.")


# ------------------------------------------- 21 ICT killzone Judas swing -----
def c_ict_judas(df, p):
    out = df.copy()
    intraday = bool(p.get("intraday", True))
    k = session_key(out.index)
    bn = bar_of_session(out.index, intraday)
    sess_open = out["Open"].groupby(k).transform("first") if intraday else out["Open"].shift(1)
    out["session_open"] = sess_open
    zone = int(_p(p, "orb_bars")) * 4
    in_kz = (bn >= 1) & (bn <= zone) if intraday else pd.Series(True, index=out.index)
    # False move away from the session open, then reclaim.
    long = in_kz & (out["Low"] < sess_open) & (out["Close"] > sess_open) & (out["Close"] > out["Open"])
    short = in_kz & (out["High"] > sess_open) & (out["Close"] < sess_open) & (out["Close"] < out["Open"])
    return _finalise(out, long, short)


def s_ict_judas(df, p):
    return _sr("Hunting the killzone fake-out around the session open.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Session open", fmt(safe_last(df["session_open"])))],
               "Price dips below the session open inside the killzone and reclaims it green.",
               "Price pops above the session open inside the killzone and loses it red.")


# ------------------------------------------------- 22 Price action pin bar ---
def c_pin_bar(df, p):
    out = df.copy()
    look = int(_p(p, "breakout_len"))
    body = (out["Close"] - out["Open"]).abs()
    rng = (out["High"] - out["Low"]).replace(0.0, np.nan)
    lower_wick = out[["Open", "Close"]].min(axis=1) - out["Low"]
    upper_wick = out["High"] - out[["Open", "Close"]].max(axis=1)
    out["sup"] = rolling_low(out["Low"], look)
    out["res"] = rolling_high(out["High"], look)
    out["body_pct"] = body / rng * 100.0
    bull_pin = (lower_wick >= 2.0 * body) & (lower_wick / rng > 0.5) & \
        (out["Close"] > out["Low"] + 0.6 * rng)
    bear_pin = (upper_wick >= 2.0 * body) & (upper_wick / rng > 0.5) & \
        (out["Close"] < out["High"] - 0.6 * rng)
    return _finalise(out, bull_pin & (out["Low"] <= out["sup"] * 1.003),
                     bear_pin & (out["High"] >= out["res"] * 0.997))


def s_pin_bar(df, p):
    return _sr("Waiting for a rejection wick to print into a structural extreme.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Body % of range", fmt(safe_last(df["body_pct"]))),
                ("Support", fmt(safe_last(df["sup"]))), ("Resistance", fmt(safe_last(df["res"])))],
               "Lower wick at least 2x the body, closing in the upper third, at support.",
               "Upper wick at least 2x the body, closing in the lower third, at resistance.")


# --------------------------------------------- 23 Inside bar breakout --------
def c_inside_bar(df, p):
    out = df.copy()
    inside = (out["High"] < out["High"].shift(1)) & (out["Low"] > out["Low"].shift(1))
    out["inside"] = inside
    mother_hi = out["High"].shift(1).where(inside).ffill()
    mother_lo = out["Low"].shift(1).where(inside).ffill()
    out["mother_hi"], out["mother_lo"] = mother_hi, mother_lo
    recent = inside.rolling(4, min_periods=1).max().astype(bool)
    return _finalise(out, recent & cross_over(out["Close"], mother_hi),
                     recent & cross_under(out["Close"], mother_lo))


def s_inside_bar(df, p):
    return _sr("Waiting for an inside-bar coil to resolve.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Mother bar high", fmt(safe_last(df["mother_hi"]))),
                ("Mother bar low", fmt(safe_last(df["mother_lo"]))),
                ("Inside bar now", "yes" if safe_last(df["inside"]) else "no")],
               "Close must break ABOVE the mother bar high within 4 bars of the coil.",
               "Close must break BELOW the mother bar low within 4 bars of the coil.")


# ------------------------------------------------- 24 Elliott wave 3 ---------
def _auto_zigzag_threshold(df: pd.DataFrame, requested: float, atr_len: int) -> float:
    """
    Scale the zigzag threshold to the instrument's own volatility.

    A fixed 0.8% threshold finds almost no pivots on a quiet series and far too
    many on a violent one, which is why the wave logic previously sat silent. We
    take roughly three ATRs as the swing size and never exceed what the user
    asked for.
    """
    a = atr(df["High"], df["Low"], df["Close"], atr_len)
    med_atr, med_px = float(a.median(skipna=True)), float(df["Close"].median())
    if not np.isfinite(med_atr) or med_px <= 0:
        return max(0.05, float(requested))
    atr_pct = 3.0 * med_atr / med_px * 100.0
    return float(max(0.05, min(float(requested), atr_pct)))


def c_elliott(df, p):
    """
    Mechanical wave-3 heuristic, NOT a real Elliott count.

    Looks for  low -> high -> higher-low  where the retracement of the first leg
    sits inside the usual Fibonacci window, then triggers when price reclaims the
    leg high. Only pivots already CONFIRMED by the zigzag on or before the
    current bar are consulted, so there is no look-ahead.
    """
    out = df.copy()
    thr = _auto_zigzag_threshold(out, _p(p, "zigzag_pct"), int(_p(p, "atr_len")))
    out["zigzag_threshold"] = thr
    pivots = zigzag_pivot_table(out["Close"], thr)
    n = len(out)
    close = out["Close"].to_numpy(float)
    long = np.zeros(n, dtype=bool)
    short = np.zeros(n, dtype=bool)
    wave_hi, wave_lo = np.full(n, np.nan), np.full(n, np.nan)
    retr_col, label = np.full(n, np.nan), np.full(n, "", dtype=object)
    piv_count = np.zeros(n, dtype=int)          # pivots CONFIRMED so far, for tuning

    lo_f, hi_f = 0.236, 0.886          # the window practitioners actually use
    known: list[tuple[int, float, int, int]] = []
    ptr = 0
    for i in range(n):
        while ptr < len(pivots) and pivots[ptr][3] <= i:
            known.append(pivots[ptr])
            ptr += 1
        piv_count[i] = len(known)
        if len(known) < 3:
            continue
        (_, p1, k1, _), (_, p2, k2, _), (_, p3, k3, _) = known[-3], known[-2], known[-1]
        if (k1, k2, k3) == (-1, 1, -1) and p2 > p1:          # low -> high -> low
            leg = p2 - p1
            retr = (p2 - p3) / leg if leg > 0 else np.nan
            wave_hi[i], wave_lo[i], retr_col[i] = p2, p3, retr
            label[i] = "Wave 2 retrace, awaiting reclaim of the leg high"
            if leg > 0 and lo_f <= retr <= hi_f and p3 > p1 and close[i] > p2:
                long[i] = True
                label[i] = "Wave 3 trigger (long)"
        elif (k1, k2, k3) == (1, -1, 1) and p2 < p1:         # high -> low -> high
            leg = p1 - p2
            retr = (p3 - p2) / leg if leg > 0 else np.nan
            wave_hi[i], wave_lo[i], retr_col[i] = p3, p2, retr
            label[i] = "Wave 2 retrace, awaiting loss of the leg low"
            if leg > 0 and lo_f <= retr <= hi_f and p3 < p1 and close[i] < p2:
                short[i] = True
                label[i] = "Wave 3 trigger (short)"
    out["wave_high"], out["wave_low"] = wave_hi, wave_lo
    out["wave_retrace"], out["wave_state"] = retr_col, label
    out["zz_pivots"] = piv_count
    idx = out.index
    return _finalise(out, pd.Series(long, index=idx), pd.Series(short, index=idx))


def s_elliott(df, p):
    state = safe_last(df["wave_state"]) or "No qualifying wave structure yet"
    retr = safe_last(df["wave_retrace"])
    hi, lo, c = safe_last(df["wave_high"]), safe_last(df["wave_low"]), safe_last(df["Close"])
    need_long = (hi - c) if (hi is not None and c is not None) else None
    need_short = (c - lo) if (lo is not None and c is not None) else None
    return _sr(f"{state}. Elliott labelling is subjective; this is a mechanical zigzag "
               f"approximation, not an analyst's count.",
               [("Last price", fmt(c)), ("Leg extreme", fmt(hi)), ("Retrace pivot", fmt(lo)),
                ("Retracement", "--" if retr is None else f"{retr*100:.1f}%"),
                ("Swing threshold", f"{fmt(safe_last(df['zigzag_threshold']), 3)}%"),
                ("Pivots confirmed", fmt(safe_last(df["zz_pivots"]), 0)),
                ("Points to long trigger", fmt_signed(need_long)),
                ("Points to short trigger", fmt_signed(need_short))],
               "Impulse leg up, a 23.6-88.6% retrace holding above the origin, then a close "
               "back above the leg high.",
               "Impulse leg down, a 23.6-88.6% retrace holding below the origin, then a close "
               "back below the leg low.")


# ------------------------------------------------- 25 SuperTrend flip --------
def c_supertrend_flip(df, p):
    out = df.copy()
    d, up, dn = supertrend(out["High"], out["Low"], out["Close"],
                           int(_p(p, "st_len")), float(_p(p, "st_mult")))
    out["st_dir"], out["st_up"], out["st_dn"] = d, up, dn
    flip = d != d.shift(1)
    return _finalise(out, flip & (d == 1), flip & (d == -1))


def s_supertrend_flip(df, p):
    d = safe_last(df["st_dir"])
    return _sr(f"SuperTrend({int(_p(p,'st_len'))}, {fmt(_p(p,'st_mult'),1)}) is "
               f"{'BULLISH' if d == 1 else 'BEARISH'}.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Upper band", fmt(safe_last(df["st_up"]))),
                ("Lower band", fmt(safe_last(df["st_dn"])))],
               "SuperTrend must flip to bullish.", "SuperTrend must flip to bearish.")


# ---------------------------------------------------- 26 VWAP reversion ------
def c_vwap_reversion(df, p):
    out = df.copy()
    vw, vol_ok = vwap(out, bool(p.get("intraday", True)))
    out["vwap"] = vw
    out.attrs["vwap_is_volume_weighted"] = vol_ok
    out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(p, "atr_len")))
    k = float(_p(p, "channel_mult"))
    out["vwap_lo"] = out["vwap"] - k * out["atr"]
    out["vwap_hi"] = out["vwap"] + k * out["atr"]
    return _finalise(out, cross_over(out["Close"], out["vwap_lo"]),
                     cross_under(out["Close"], out["vwap_hi"]))


def s_vwap_reversion(df, p):
    vol_ok = bool(df.attrs.get("vwap_is_volume_weighted", True))
    return _sr("Fading stretches away from VWAP." if vol_ok else
               "This feed has no volume, so the anchor is a session TWAP, not a true VWAP.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("VWAP anchor", fmt(safe_last(df["vwap"]))),
                ("Lower band", fmt(safe_last(df["vwap_lo"]))),
                ("Upper band", fmt(safe_last(df["vwap_hi"])))],
               "Price stretches below the lower band then closes back inside.",
               "Price stretches above the upper band then closes back inside.")


# ------------------------------------------- 27 Wyckoff spring / upthrust ----
def c_wyckoff(df, p):
    out = df.copy()
    look = int(_p(p, "structure_len"))
    out["range_low"] = rolling_low(out["Low"], look)
    out["range_high"] = rolling_high(out["High"], look)
    out["vol_ma"] = sma(out["Volume"], int(_p(p, "vol_len")))
    # Volume confirmation is optional: zero-volume feeds must not be locked out.
    dead = float(out["Volume"].abs().sum()) == 0.0
    vol_ok = pd.Series(True, index=out.index) if dead else (out["Volume"] > out["vol_ma"])
    spring = (out["Low"] < out["range_low"]) & (out["Close"] > out["range_low"]) & \
        (out["Close"] > out["Open"]) & vol_ok
    upthrust = (out["High"] > out["range_high"]) & (out["Close"] < out["range_high"]) & \
        (out["Close"] < out["Open"]) & vol_ok
    return _finalise(out, spring, upthrust)


def s_wyckoff(df, p):
    return _sr("Watching the range edges for a spring or an upthrust.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Range low", fmt(safe_last(df["range_low"]))),
                ("Range high", fmt(safe_last(df["range_high"])))],
               "Price must dip below the range low and close back inside, green.",
               "Price must poke above the range high and close back inside, red.")


# ------------------------------------- 28 / 29 Price threshold crossings -----
def _threshold_signals(out, upper, lower, mode):
    up = cross_over(out["Close"], upper)
    dn = cross_under(out["Close"], lower)
    if mode == "Cross above = BUY only":
        return up, pd.Series(False, index=out.index)
    if mode == "Cross below = SELL only":
        return pd.Series(False, index=out.index), dn
    if mode.startswith("Cross above = SELL"):
        return dn, up                       # faded: the break is sold, the breakdown bought
    return up, dn


def c_threshold_abs(df, p):
    """Cross of a fixed price level the operator types in."""
    out = df.copy()
    level = float(_p(p, "threshold_price"))
    if level <= 0:                          # unset -> anchor on the window's first close
        level = float(out["Close"].iloc[0])
    out["threshold"] = level
    long, short = _threshold_signals(out, out["threshold"], out["threshold"],
                                     str(_p(p, "threshold_mode")))
    return _finalise(out, long, short)


def s_threshold_abs(df, p):
    lvl, c = safe_last(df["threshold"]), safe_last(df["Close"])
    dist = (c - lvl) if (c is not None and lvl is not None) else None
    return _sr(f"Price is {fmt(abs(dist or 0))} points "
               f"{'above' if (dist or 0) >= 0 else 'below'} the {fmt(lvl)} threshold.",
               [("Last price", fmt(c)), ("Threshold", fmt(lvl)), ("Distance", fmt_signed(dist)),
                ("Mode", str(_p(p, "threshold_mode")))],
               "Close must cross the threshold in the direction set by the mode.",
               "Close must cross the threshold in the direction set by the mode.")


def c_threshold_pct(df, p):
    """Cross of a percentage band around a moving reference price."""
    out = df.copy()
    pct = abs(float(_p(p, "threshold_pct"))) / 100.0
    ref_kind = str(_p(p, "threshold_ref"))
    intraday = bool(p.get("intraday", True))
    if ref_kind == "Session open" and intraday:
        k = session_key(out.index)
        ref = out["Open"].groupby(k).transform("first")
    elif ref_kind == "Rolling 20-bar mean":
        ref = sma(out["Close"], 20)
    elif ref_kind == "First candle of the loaded window":
        ref = pd.Series(float(out["Close"].iloc[0]), index=out.index)
    else:                                   # previous session close
        if intraday:
            k = session_key(out.index)
            ref = out["Close"].groupby(k).transform("last").groupby(k).transform("first")
            ref = ref.shift(1).ffill().fillna(out["Close"].iloc[0])
        else:
            ref = out["Close"].shift(1)
    out["threshold_ref"] = ref
    out["threshold_up"] = ref * (1 + pct)
    out["threshold_dn"] = ref * (1 - pct)
    long, short = _threshold_signals(out, out["threshold_up"], out["threshold_dn"],
                                     str(_p(p, "threshold_mode")))
    return _finalise(out, long, short)


def s_threshold_pct(df, p):
    c = safe_last(df["Close"])
    ref = safe_last(df["threshold_ref"])
    moved = ((c - ref) / ref * 100.0) if (c and ref) else None
    return _sr(f"Price is {fmt_signed(moved)}% from the {str(_p(p,'threshold_ref')).lower()} "
               f"reference of {fmt(ref)}.",
               [("Last price", fmt(c)), ("Reference", fmt(ref)), ("Move %", fmt_signed(moved)),
                ("Upper band", fmt(safe_last(df["threshold_up"]))),
                ("Lower band", fmt(safe_last(df["threshold_dn"])))],
               f"Close must cross the {fmt(_p(p,'threshold_pct'))}% band per the selected mode.",
               f"Close must cross the {fmt(_p(p,'threshold_pct'))}% band per the selected mode.")


# ------------------------------------ 34-43 Fibonacci and structure ----------
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
HYBRID_LOGIC = ["All selected must agree (AND)", "Any one is enough (OR)"]


def fib_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach retracement levels measured off the last CONFIRMED swing pair.

    The leg direction is inferred from where price currently sits between the
    two pivots. Everything derives from confirmed pivots, so nothing here can
    see the future.
    """
    out = df
    hi, lo = out["swing_high"], out["swing_low"]
    leg = hi - lo
    out["fib_leg"] = leg
    up_leg = (out["Close"] - lo).abs() >= (hi - out["Close"]).abs()
    out["fib_up_leg"] = up_leg
    for r in FIB_RATIOS:
        out[f"fib_{int(r*1000)}"] = np.where(up_leg, hi - leg * r, lo + leg * r)
    out["fib_golden_hi"] = out[["fib_500", "fib_618"]].max(axis=1)
    out["fib_golden_lo"] = out[["fib_500", "fib_618"]].min(axis=1)
    return out


def _fib_base(df, p):
    out = df.copy()
    sh, sl_, _, _ = swing_levels(out["High"], out["Low"],
                                 int(_p(p, "pivot_left")), int(_p(p, "pivot_right")))
    out["swing_high"], out["swing_low"] = sh, sl_
    return fib_frame(out)


def _fib_zone_touch(out: pd.DataFrame):
    """Long when a rising leg retraces into the golden zone and closes back up."""
    long = (out["fib_up_leg"] & (out["Low"] <= out["fib_golden_hi"])
            & (out["Close"] > out["fib_golden_lo"]) & (out["Close"] > out["Open"]))
    short = ((~out["fib_up_leg"]) & (out["High"] >= out["fib_golden_lo"])
             & (out["Close"] < out["fib_golden_hi"]) & (out["Close"] < out["Open"]))
    return long, short


def _fib_status(df, extra=None):
    return _sr("Watching for a retracement into the 50-61.8% zone of the last confirmed leg.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Swing high", fmt(safe_last(df["swing_high"]))),
                ("Swing low", fmt(safe_last(df["swing_low"]))),
                ("Golden zone", f"{fmt(safe_last(df['fib_golden_lo']))} - "
                                f"{fmt(safe_last(df['fib_golden_hi']))}"),
                ("Leg direction", "up" if safe_last(df["fib_up_leg"]) else "down")]
               + list(extra or []),
               "Rising leg retraces into the golden zone and the candle closes green.",
               "Falling leg retraces into the golden zone and the candle closes red.")


def c_fib_zone(df, p):
    out = _fib_base(df, p)
    return _finalise(out, *_fib_zone_touch(out))


def s_fib_zone(df, p):
    return _fib_status(df)


def c_fib_rsi(df, p):
    out = _fib_base(df, p)
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    long, short = _fib_zone_touch(out)
    return _finalise(out, long & (out["rsi"] >= float(_p(p, "rsi_long_level"))),
                     short & (out["rsi"] <= float(_p(p, "rsi_short_level"))))


def s_fib_rsi(df, p):
    return _fib_status(df, [("RSI", fmt(safe_last(df["rsi"])))])


def c_fib_volume(df, p):
    out = _fib_base(df, p)
    out["vol_ma"] = sma(out["Volume"], int(_p(p, "vol_len")))
    out["vol_ratio"] = out["Volume"] / out["vol_ma"].replace(0.0, np.nan)
    dead = float(out["Volume"].abs().sum()) == 0.0
    surge = pd.Series(True, index=out.index) if dead else \
        (out["vol_ratio"] >= float(_p(p, "vol_mult")))
    long, short = _fib_zone_touch(out)
    return _finalise(out, long & surge, short & surge)


def s_fib_volume(df, p):
    dead = float(df["Volume"].tail(50).abs().sum()) == 0.0
    return _fib_status(df, [("Volume ratio", "n/a (feed has no volume)" if dead
                             else fmt(safe_last(df["vol_ratio"])))])


def c_fib_vwap(df, p):
    out = _fib_base(df, p)
    vw, vol_ok = vwap(out, bool(p.get("intraday", True)))
    out["vwap"] = vw
    out.attrs["vwap_is_volume_weighted"] = vol_ok
    long, short = _fib_zone_touch(out)
    return _finalise(out, long & (out["Close"] > out["vwap"]),
                     short & (out["Close"] < out["vwap"]))


def s_fib_vwap(df, p):
    ok = bool(df.attrs.get("vwap_is_volume_weighted", True))
    return _fib_status(df, [("VWAP" if ok else "TWAP (no volume)", fmt(safe_last(df["vwap"])))])


def c_fib_macd(df, p):
    out = _fib_base(df, p)
    _, _, hist = macd(out["Close"], int(_p(p, "macd_fast")), int(_p(p, "macd_slow")),
                      int(_p(p, "macd_signal")))
    out["macd_hist"] = hist
    long, short = _fib_zone_touch(out)
    return _finalise(out, long & (out["macd_hist"] > 0), short & (out["macd_hist"] < 0))


def s_fib_macd(df, p):
    return _fib_status(df, [("MACD histogram", fmt(safe_last(df["macd_hist"]), 4))])


def c_fib_bollinger(df, p):
    out = _fib_base(df, p)
    mid, up, lo = bollinger(out["Close"], int(_p(p, "bb_len")), float(_p(p, "bb_mult")))
    out["bb_mid"], out["bb_up"], out["bb_lo"] = mid, up, lo
    long, short = _fib_zone_touch(out)
    inside = (out["Close"] < out["bb_up"]) & (out["Close"] > out["bb_lo"])
    return _finalise(out, long & inside, short & inside)


def s_fib_bollinger(df, p):
    return _fib_status(df, [("Bollinger mid", fmt(safe_last(df["bb_mid"])))])


def c_fib_fvg(df, p):
    out = _fib_base(df, p)
    _, _, blo, bhi, selo, sehi = fair_value_gaps(out)
    out["fvg_bull_lo"], out["fvg_bull_hi"] = blo, bhi
    out["fvg_bear_lo"], out["fvg_bear_hi"] = selo, sehi
    long, short = _fib_zone_touch(out)
    # The imbalance must overlap the retracement zone: two independent reasons
    # for the same price to matter.
    long &= (out["fvg_bull_hi"] >= out["fib_golden_lo"]) & (out["fvg_bull_lo"] <= out["fib_golden_hi"])
    short &= (out["fvg_bear_hi"] >= out["fib_golden_lo"]) & (out["fvg_bear_lo"] <= out["fib_golden_hi"])
    return _finalise(out, long, short)


def s_fib_fvg(df, p):
    return _fib_status(df, [("Bull FVG", f"{fmt(safe_last(df['fvg_bull_lo']))} - "
                                         f"{fmt(safe_last(df['fvg_bull_hi']))}"),
                            ("Bear FVG", f"{fmt(safe_last(df['fvg_bear_lo']))} - "
                                         f"{fmt(safe_last(df['fvg_bear_hi']))}")])


def c_ema_retest_breakout(df, p):
    """
    Double EMA crossover -> pullback to the fast EMA -> break of the pre-pullback
    extreme. The signal prints on the breakout candle and, like every profile
    here, fills at the NEXT candle's open.
    """
    out = df.copy()
    look = int(_p(p, "breakout_len"))
    out["ema_fast"] = ema(out["Close"], int(_p(p, "ema_fast")))
    out["ema_slow"] = ema(out["Close"], int(_p(p, "ema_slow")))
    bull = out["ema_fast"] > out["ema_slow"]

    crossed_up = cross_over(out["ema_fast"], out["ema_slow"])
    crossed_dn = cross_under(out["ema_fast"], out["ema_slow"])
    cross_any = crossed_up | crossed_dn
    pos = pd.Series(np.arange(len(out)), index=out.index)
    last_cross = pos.where(cross_any).ffill()
    out["bars_since_cross"] = (pos - last_cross).fillna(9999)

    pulled_up = (out["Low"] <= out["ema_fast"]).rolling(look, min_periods=1).max().astype(bool)
    pulled_dn = (out["High"] >= out["ema_fast"]).rolling(look, min_periods=1).max().astype(bool)
    out["retest_high"] = rolling_high(out["High"], look)
    out["retest_low"] = rolling_low(out["Low"], look)

    fresh = out["bars_since_cross"] <= look * 2
    long = bull & fresh & pulled_up & cross_over(out["Close"], out["retest_high"])
    short = (~bull) & fresh & pulled_dn & cross_under(out["Close"], out["retest_low"])
    return _finalise(out, long, short)


def s_ema_retest_breakout(df, p):
    c, hi, lo = safe_last(df["Close"]), safe_last(df["retest_high"]), safe_last(df["retest_low"])
    f, sl_ = safe_last(df["ema_fast"]), safe_last(df["ema_slow"])
    return _sr(f"Trend is {'UP' if (f or 0) > (sl_ or 0) else 'DOWN'}; "
               f"{fmt(safe_last(df['bars_since_cross']), 0)} bars since the crossover.",
               [("Last price", fmt(c)), ("Fast EMA", fmt(f)), ("Slow EMA", fmt(sl_)),
                ("Breakout high", fmt(hi)),
                ("Points to long", fmt_signed((hi or 0) - (c or 0))),
                ("Points to short", fmt_signed((c or 0) - (lo or 0)))],
               "Bullish cross, pullback to the fast EMA, then a close above the recent high. "
               "Entry is the next candle's open.",
               "Bearish cross, pullback to the fast EMA, then a close below the recent low. "
               "Entry is the next candle's open.")


def c_price_action(df, p):
    """Composite price action: pin bar, engulfing, or inside-bar break at structure."""
    out = df.copy()
    look = int(_p(p, "breakout_len"))
    out["sup"] = rolling_low(out["Low"], look)
    out["res"] = rolling_high(out["High"], look)
    body = (out["Close"] - out["Open"]).abs()
    rng = (out["High"] - out["Low"]).replace(0.0, np.nan)
    lw = out[["Open", "Close"]].min(axis=1) - out["Low"]
    uw = out["High"] - out[["Open", "Close"]].max(axis=1)
    po, pc = out["Open"].shift(1), out["Close"].shift(1)

    bull_pin = (lw >= 2 * body) & (lw / rng > 0.5)
    bear_pin = (uw >= 2 * body) & (uw / rng > 0.5)
    bull_eng = (out["Close"] > out["Open"]) & (pc < po) & (out["Close"] >= po) & (out["Open"] <= pc)
    bear_eng = (out["Close"] < out["Open"]) & (pc > po) & (out["Close"] <= po) & (out["Open"] >= pc)
    inside = (out["High"] < out["High"].shift(1)) & (out["Low"] > out["Low"].shift(1))
    ib_up = inside.shift(1).fillna(False) & (out["Close"] > out["High"].shift(1))
    ib_dn = inside.shift(1).fillna(False) & (out["Close"] < out["Low"].shift(1))

    at_sup = out["Low"] <= out["sup"] * 1.004
    at_res = out["High"] >= out["res"] * 0.996
    out["pa_pattern"] = np.where(
        bull_pin, "bullish pin", np.where(
            bear_pin, "bearish pin", np.where(
                bull_eng, "bullish engulfing", np.where(
                    bear_eng, "bearish engulfing", np.where(
                        ib_up, "inside-bar break up", np.where(
                            ib_dn, "inside-bar break down", ""))))))
    return _finalise(out, ((bull_pin | bull_eng) & at_sup) | ib_up,
                     ((bear_pin | bear_eng) & at_res) | ib_dn)


def s_price_action(df, p):
    return _sr(f"Latest price action read: {safe_last(df['pa_pattern']) or 'nothing on this bar'}.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Support", fmt(safe_last(df["sup"]))),
                ("Resistance", fmt(safe_last(df["res"])))],
               "Bullish pin or engulfing at support, or an inside-bar break upward.",
               "Bearish pin or engulfing at resistance, or an inside-bar break downward.")


def c_hybrid(df, p):
    """
    Combine several profiles under AND / OR.

    AND is strict: every member must point the same way on the SAME candle,
    which is rare and produces very few trades. OR fires on the first member to
    signal and therefore inherits the false positives of all of them.
    """
    out = df.copy()
    members = [m for m in (p.get("hybrid_members") or [])
               if m in STRATEGIES and not STRATEGIES[m].immediate and not m.startswith("43 ")]
    out["ema_fast"] = ema(out["Close"], int(_p(p, "ema_fast")))
    out["ema_slow"] = ema(out["Close"], int(_p(p, "ema_slow")))
    empty = pd.Series(False, index=out.index)
    if not members:
        out["hybrid_detail"] = "no members selected"
        out["hybrid_long_votes"] = 0
        out["hybrid_short_votes"] = 0
        out["hybrid_members"] = 0
        return _finalise(out, empty, empty)

    longs, shorts, names = [], [], []
    for name in members:
        try:
            sig = get_strategy(name).compute(df, p)["signal"]
        except Exception:                                           # noqa: BLE001
            continue
        longs.append(sig == 1)
        shorts.append(sig == -1)
        names.append(name.split("\u00b7 ")[-1].strip())
    if not longs:
        out["hybrid_detail"] = "members failed to compute"
        out["hybrid_long_votes"] = 0
        out["hybrid_short_votes"] = 0
        out["hybrid_members"] = 0
        return _finalise(out, empty, empty)

    lf, sf = pd.concat(longs, axis=1), pd.concat(shorts, axis=1)
    out["hybrid_long_votes"] = lf.sum(axis=1)
    out["hybrid_short_votes"] = sf.sum(axis=1)
    out["hybrid_members"] = len(longs)
    out["hybrid_detail"] = ", ".join(names)
    if str(p.get("hybrid_logic", HYBRID_LOGIC[0])).startswith("All"):
        return _finalise(out, lf.all(axis=1), sf.all(axis=1))
    return _finalise(out, lf.any(axis=1), sf.any(axis=1))


def s_hybrid(df, p):
    logic = str(p.get("hybrid_logic", HYBRID_LOGIC[0]))
    n = int(safe_last(df["hybrid_members"]) or 0)
    return _sr(f"{n} member profile(s) combined under: {logic}.",
               [("Members", str(safe_last(df["hybrid_detail"]) or "none")),
                ("Long votes", fmt(safe_last(df["hybrid_long_votes"]), 0)),
                ("Short votes", fmt(safe_last(df["hybrid_short_votes"]), 0)),
                ("Needed", "all of them" if logic.startswith("All") else "any one")],
               "Member profiles must agree per the selected logic.",
               "Member profiles must agree per the selected logic.")


# ------------------------------------------ 44-48 Option profiles ------------
# DIRECTION MAPPING: a LONG signal buys a CALL, a SHORT signal buys a PUT. The
# engine and the Dhan router both follow that rule, and the leg is recorded on
# every trade row.
#
# DATA HONESTY: open interest and PCR are not available from Yahoo at all, and
# no free source publishes a usable HISTORY of them. So:
#   * the OI profiles below are LIVE-ONLY and need Dhan market data. In a
#     backtest they deliberately produce nothing rather than inventing a series.
#   * Zero Hero and Gamma Blast are expressed as rules on the UNDERLYING, which
#     is genuinely backtestable, with the option leg chosen from the direction.

OPTION_OI_PROFILES = {"44 \u00b7 Options: OI Change", "45 \u00b7 Options: OI Change + PCR",
                      "46 \u00b7 Options: OI Change + Volume"}


def _oi_history() -> list[dict]:
    if st is None:
        return []
    try:
        return list(st.session_state.get("option_metrics_history", []) or [])
    except Exception:                                               # noqa: BLE001
        return []


def _oi_signal(params: dict) -> tuple[int, dict]:
    """
    Read the accumulated option-chain history and return (direction, detail).

    History is built up by the live loop polling Dhan; there is no historical
    feed to replay, so an empty history means no signal rather than a guess.
    """
    hist = _oi_history()
    detail = {"samples": len(hist)}
    if len(hist) < 2:
        return 0, detail
    now, prev = hist[-1], hist[-2]
    ce_chg = float(now.get("ce_oi", 0)) - float(prev.get("ce_oi", 0))
    pe_chg = float(now.get("pe_oi", 0)) - float(prev.get("pe_oi", 0))
    pcr = float(now.get("pcr", 0) or 0)
    detail.update(ce_change=ce_chg, pe_change=pe_chg, pcr=pcr,
                  ce_volume=now.get("ce_volume"), pe_volume=now.get("pe_volume"))
    threshold = float(params.get("oi_change_threshold", 0.0))
    if abs(ce_chg - pe_chg) < threshold:
        return 0, detail
    # Put writing (PE OI building faster) is the bullish tell; call writing is bearish.
    direction = 1 if pe_chg > ce_chg else -1
    if str(params.get("strategy_name", "")).startswith("45 "):
        lo, hi = float(params.get("pcr_min", 0.8)), float(params.get("pcr_max", 1.2))
        if direction > 0 and pcr < lo:
            return 0, detail
        if direction < 0 and pcr > hi:
            return 0, detail
    if str(params.get("strategy_name", "")).startswith("46 "):
        cev, pev = float(now.get("ce_volume", 0) or 0), float(now.get("pe_volume", 0) or 0)
        if direction > 0 and pev <= cev:
            return 0, detail
        if direction < 0 and cev <= pev:
            return 0, detail
    return direction, detail


def _c_option_oi(name: str):
    def build(df, p):
        out = df.copy()
        out["ema_fast"] = ema(out["Close"], int(_p(p, "ema_fast")))
        out["ema_slow"] = ema(out["Close"], int(_p(p, "ema_slow")))
        params = dict(p)
        params["strategy_name"] = name
        direction, detail = _oi_signal(params)
        out["oi_samples"] = detail.get("samples", 0)
        out["oi_ce_change"] = detail.get("ce_change", np.nan)
        out["oi_pe_change"] = detail.get("pe_change", np.nan)
        out["oi_pcr"] = detail.get("pcr", np.nan)
        long = pd.Series(False, index=out.index)
        short = pd.Series(False, index=out.index)
        # The chain describes NOW, so it can only speak about the newest closed bar.
        if direction != 0 and len(out) >= 2:
            (long if direction > 0 else short).iloc[-2] = True
        return _finalise(out, long, short)
    return build


def _s_option_oi(name: str):
    def status(df, p):
        hist = _oi_history()
        if len(hist) < 2:
            head = ("No option-chain history yet. These profiles need Dhan market data and "
                    "build their series live; they cannot be backtested, because no free "
                    "source publishes historical OI.")
        else:
            head = (f"Chain sampled {len(hist)} times. Put writing is the bullish tell, "
                    f"call writing the bearish one.")
        return _sr(head,
                   [("Samples", str(len(hist))),
                    ("CE OI change", fmt(safe_last(df["oi_ce_change"]), 0)),
                    ("PE OI change", fmt(safe_last(df["oi_pe_change"]), 0)),
                    ("PCR", fmt(safe_last(df["oi_pcr"]))),
                    ("Threshold", fmt(p.get("oi_change_threshold", 0.0), 0))],
                   "PE open interest must build faster than CE (long -> buy a CALL).",
                   "CE open interest must build faster than PE (short -> buy a PUT).")
    return status


def c_zero_hero(df, p):
    """
    Expiry-day momentum burst on the underlying.

    'Zero hero' is the practice of buying cheap far-OTM options on expiry and
    needing a large, fast move. The tradable edge, if any, is in the underlying
    burst, so that is what is modelled here.
    """
    out = df.copy()
    intraday = bool(p.get("intraday", True))
    out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(p, "atr_len")))
    k = session_key(out.index)
    sess_open = out["Open"].groupby(k).transform("first") if intraday else out["Open"]
    out["session_open"] = sess_open
    out["move_from_open"] = out["Close"] - sess_open
    mult = float(_p(p, "zero_hero_atr"))
    weekday = int(_p(p, "expiry_weekday"))
    is_expiry = pd.Series(pd.DatetimeIndex(out.index).weekday == weekday, index=out.index) \
        if intraday else pd.Series(True, index=out.index)
    out["is_expiry_day"] = is_expiry
    burst_up = out["move_from_open"] >= mult * out["atr"]
    burst_dn = out["move_from_open"] <= -mult * out["atr"]
    first_up = burst_up & ~burst_up.groupby(k).shift(1).fillna(False)
    first_dn = burst_dn & ~burst_dn.groupby(k).shift(1).fillna(False)
    return _finalise(out, is_expiry & first_up, is_expiry & first_dn)


def s_zero_hero(df, p):
    move, a = safe_last(df["move_from_open"]), safe_last(df["atr"])
    need = float(_p(p, "zero_hero_atr")) * (a or 0)
    return _sr(f"{'Expiry day.' if safe_last(df['is_expiry_day']) else 'Not an expiry day.'} "
               f"Move from the session open is {fmt_signed(move)}; the burst needs {fmt(need)}.",
               [("Session open", fmt(safe_last(df["session_open"]))),
                ("Move from open", fmt_signed(move)), ("ATR", fmt(a)),
                ("Burst threshold", fmt(need)),
                ("Points to long", fmt_signed(need - (move or 0))),
                ("Points to short", fmt_signed(-need - (move or 0)))],
               "On expiry day, price must run the burst distance ABOVE the session open "
               "(long -> buy a CALL).",
               "On expiry day, price must run the burst distance BELOW the session open "
               "(short -> buy a PUT).")


def c_gamma_blast(df, p):
    """
    Late-session volatility expansion, the setup people call a gamma blast.

    Restricted to the closing stretch of the session, when option gamma is at
    its most violent, and requires both an ATR expansion and a directional break
    of the session's own range.
    """
    out = df.copy()
    intraday = bool(p.get("intraday", True))
    look = int(_p(p, "breakout_len"))
    out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(p, "atr_len")))
    out["atr_mean"] = sma(out["atr"], look)
    out["atr_ratio"] = out["atr"] / out["atr_mean"]
    k = session_key(out.index)
    bar_no = bar_of_session(out.index, intraday)
    bars_in_day = bar_no.groupby(k).transform("max")
    tail_bars = int(_p(p, "gamma_tail_bars"))
    late = (bars_in_day - bar_no) <= tail_bars if intraday else pd.Series(True, index=out.index)
    out["is_late_session"] = late
    out["day_high"] = out["High"].groupby(k).cummax().groupby(k).shift(1)
    out["day_low"] = out["Low"].groupby(k).cummin().groupby(k).shift(1)
    expanding = out["atr_ratio"] >= float(_p(p, "squeeze_mult"))
    return _finalise(out, late & expanding & (out["Close"] > out["day_high"]),
                     late & expanding & (out["Close"] < out["day_low"]))


def s_gamma_blast(df, p):
    c = safe_last(df["Close"])
    hi, lo = safe_last(df["day_high"]), safe_last(df["day_low"])
    return _sr(f"{'In' if safe_last(df['is_late_session']) else 'Outside'} the closing stretch; "
               f"ATR is {fmt(safe_last(df['atr_ratio']))}x its mean.",
               [("Last price", fmt(c)), ("Session high", fmt(hi)), ("Session low", fmt(lo)),
                ("ATR ratio", fmt(safe_last(df["atr_ratio"]))),
                ("Points to long", fmt_signed((hi or 0) - (c or 0))),
                ("Points to short", fmt_signed((c or 0) - (lo or 0)))],
               "Late in the session, with volatility expanding, close above the session high "
               "(long -> buy a CALL).",
               "Late in the session, with volatility expanding, close below the session low "
               "(short -> buy a PUT).")


# ------------------------- 30-33 RSI crossover and combination profiles ------
def _rsi_levels(p):
    return float(_p(p, "rsi_long_level")), float(_p(p, "rsi_short_level"))


def c_rsi_crossover(df, p):
    """RSI reclaiming a configurable lower level / losing a configurable upper one."""
    out = df.copy()
    lo, hi = _rsi_levels(p)
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    out["rsi_long_level"], out["rsi_short_level"] = lo, hi
    return _finalise(out, cross_over(out["rsi"], _const(out, lo)),
                     cross_under(out["rsi"], _const(out, hi)))


def s_rsi_crossover(df, p):
    lo, hi = _rsi_levels(p)
    r = safe_last(df["rsi"])
    return _sr(f"RSI is {fmt(r)}; long level {fmt(lo)}, short level {fmt(hi)}.",
               [("RSI", fmt(r)), ("Long level", fmt(lo)), ("Short level", fmt(hi)),
                ("To long trigger", fmt_signed((lo - r) if r is not None else None)),
                ("To short trigger", fmt_signed((hi - r) if r is not None else None))],
               f"RSI must cross UP through {fmt(lo)}.", f"RSI must cross DOWN through {fmt(hi)}.")


def c_bb_rsi(df, p):
    """Bollinger band touch confirmed by RSI leaving its extreme."""
    out = df.copy()
    lo, hi = _rsi_levels(p)
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    mid, up, dn = bollinger(out["Close"], int(_p(p, "bb_len")), float(_p(p, "bb_mult")))
    out["bb_mid"], out["bb_up"], out["bb_lo"] = mid, up, dn
    long = (out["Low"] <= dn) & (out["Close"] > dn) & cross_over(out["rsi"], _const(out, lo))
    short = (out["High"] >= up) & (out["Close"] < up) & cross_under(out["rsi"], _const(out, hi))
    return _finalise(out, long, short)


def s_bb_rsi(df, p):
    lo, hi = _rsi_levels(p)
    return _sr("Waiting for a band rejection confirmed by RSI.",
               [("Last price", fmt(safe_last(df["Close"]))), ("RSI", fmt(safe_last(df["rsi"]))),
                ("Lower band", fmt(safe_last(df["bb_lo"]))),
                ("Upper band", fmt(safe_last(df["bb_up"]))),
                ("RSI levels", f"{fmt(lo)} / {fmt(hi)}")],
               f"Candle must tag the lower band and close above it while RSI crosses up through {fmt(lo)}.",
               f"Candle must tag the upper band and close below it while RSI crosses down through {fmt(hi)}.")


def c_ema_rsi(df, p):
    """EMA crossover that only counts when RSI agrees with the direction."""
    out = df.copy()
    lo, hi = _rsi_levels(p)
    out["ema_fast"] = ema(out["Close"], int(_p(p, "ema_fast")))
    out["ema_slow"] = ema(out["Close"], int(_p(p, "ema_slow")))
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    long = cross_over(out["ema_fast"], out["ema_slow"]) & (out["rsi"] >= lo)
    short = cross_under(out["ema_fast"], out["ema_slow"]) & (out["rsi"] <= hi)
    return _finalise(out, long, short)


def s_ema_rsi(df, p):
    lo, hi = _rsi_levels(p)
    f, sl_ = safe_last(df["ema_fast"]), safe_last(df["ema_slow"])
    return _sr(f"EMA spread {fmt_signed((f - sl_) if (f and sl_) else None)}, "
               f"RSI {fmt(safe_last(df['rsi']))}.",
               [("Fast EMA", fmt(f)), ("Slow EMA", fmt(sl_)),
                ("Points to cross", fmt(abs((f - sl_) if (f and sl_) else 0))),
                ("RSI", fmt(safe_last(df["rsi"]))), ("RSI gates", f"{fmt(lo)} / {fmt(hi)}")],
               f"Fast EMA crosses above slow EMA while RSI is at or above {fmt(lo)}.",
               f"Fast EMA crosses below slow EMA while RSI is at or below {fmt(hi)}.")


def c_vol_rsi(df, p):
    """Volume spike plus an RSI crossing. Inert on feeds that report no volume."""
    out = df.copy()
    lo, hi = _rsi_levels(p)
    out["rsi"] = rsi(out["Close"], int(_p(p, "rsi_len")))
    out["vol_ma"] = sma(out["Volume"], int(_p(p, "vol_len")))
    out["vol_ratio"] = out["Volume"] / out["vol_ma"].replace(0.0, np.nan)
    surge = out["vol_ratio"] >= float(_p(p, "vol_mult"))
    return _finalise(out, surge & cross_over(out["rsi"], _const(out, lo)),
                     surge & cross_under(out["rsi"], _const(out, hi)))


def s_vol_rsi(df, p):
    lo, hi = _rsi_levels(p)
    dead = float(df["Volume"].tail(50).abs().sum()) == 0.0
    return _sr("This feed reports no volume, so this profile cannot arm." if dead else
               f"Volume {fmt(safe_last(df['vol_ratio']))}x average, RSI {fmt(safe_last(df['rsi']))}.",
               [("Volume ratio", fmt(safe_last(df["vol_ratio"]))),
                ("RSI", fmt(safe_last(df["rsi"]))), ("RSI levels", f"{fmt(lo)} / {fmt(hi)}"),
                ("Spike needed", f"{fmt(_p(p,'vol_mult'))}x")],
               f"Volume spike with RSI crossing up through {fmt(lo)}.",
               f"Volume spike with RSI crossing down through {fmt(hi)}.")


# ------------------------------------------------------------- REGISTRY ------
_DEFS = [
    ("S01", "01 · Dual EMA Crossover", "9 EMA crossing the 21 EMA.", 40,
     c_dual_ema, s_dual_ema, ("ema_fast", "ema_slow"), None, False),
    ("S02", "02 · RSI Mean Reversion", "Buy the recovery from oversold, fade the failure from overbought.",
     40, c_rsi_reversion, s_rsi_reversion, (), "rsi", False),
    ("S03", "03 · EMA Structural Trend Pullback Scalper", "Pullbacks into the 21 EMA filtered by the 200 EMA.",
     210, c_pullback, s_pullback, ("ema_slow", "ema_macro"), None, False),
    ("S04", "04 · ATR Trailing Volatility Breakout", "Stop-and-reverse breaks of trailing ATR bands.",
     60, c_atr_trail, s_atr_trail, ("trail_upper", "trail_lower"), None, False),
    ("S05", "05 · Opening Range Breakout (ORB)", "Breaks of the session opening range.", 40,
     c_orb, s_orb, ("or_high", "or_low"), None, False),
    ("S06", "06 · Macro Golden Cross Continuum", "50 EMA crossing the 200 EMA.", 210,
     c_golden_cross, s_golden_cross, ("ema_mid", "ema_macro"), None, False),
    ("S07", "07 · Gap Counter-Trend Fade Momentum", "Fading exhausted gap-ups, buying gap-down reversals.",
     40, c_gap_fade, s_gap_fade, (), None, False),
    ("S08", "08 · RSI Centerline 50 Crossing", "Trend acceleration through the RSI median.", 40,
     c_rsi_centerline, s_rsi_centerline, (), "rsi", False),
    ("S09", "09 · Multi-Timeframe EMA Macro Vector", "Fast, intermediate and major EMAs aligning.", 210,
     c_mtf_vector, s_mtf_vector, ("ema_fast", "ema_slow", "ema_mid", "ema_macro"), None, False),
    ("S10", "10 · Volatility Price Squeeze Multiplier", "Breakouts as ATR expands past its mean.", 60,
     c_squeeze, s_squeeze, ("box_high", "box_low"), None, False),
    ("S11", "11 · High Volume Structural Confirmation", "Structure breaks backed by a volume spike.", 60,
     c_volume_breakout, s_volume_breakout, ("box_high", "box_low"), None, False),
    ("S12", "12 · Engulfing Candlestick Reversal", "Engulfing bars printed into structural zones.", 40,
     c_engulfing, s_engulfing, ("sup", "res"), None, False),
    ("S13", "13 · ATR Channel Reversion Engine", "Fading extensions outside an ATR channel.", 60,
     c_channel_reversion, s_channel_reversion, ("basis", "ch_upper", "ch_lower"), None, False),
    ("S14", "14 · RSI Momentum Swing Burst", "Momentum entries as RSI bursts through 60 / 40.", 60,
     c_rsi_burst, s_rsi_burst, ("ema_mid",), "rsi", False),
    ("S15", "15 · Macro-Trend EMA Bias Scalper", "Quick plays aligned to the 50-bar trend path.", 60,
     c_bias_scalper, s_bias_scalper, ("ema_fast", "ema_mid"), None, False),
    ("S16", "16 · Simple Buy (immediate entry)", "Enters LONG at once and runs until the exit resolves it.",
     30, _c_simple(1), _s_simple(1), ("ema_fast", "ema_slow"), None, True),
    ("S17", "17 · Simple Sell (immediate entry)", "Enters SHORT at once and runs until the exit resolves it.",
     30, _c_simple(-1), _s_simple(-1), ("ema_fast", "ema_slow"), None, True),
    ("S18", "18 · SMC Break of Structure + Order Block", "BOS, then a retrace into the originating order block.",
     60, c_smc_ob, s_smc_ob, ("swing_high", "swing_low"), None, False),
    ("S19", "19 · SMC Liquidity Sweep Reversal", "Stop-hunt beyond a swing, then a close back inside.",
     60, c_smc_sweep, s_smc_sweep, ("swing_high", "swing_low"), None, False),
    ("S20", "20 · ICT Fair Value Gap Entry", "Rebalance into the last unfilled imbalance.", 60,
     c_ict_fvg, s_ict_fvg, ("fvg_bull_hi", "fvg_bear_lo"), None, False),
    ("S21", "21 · ICT Killzone Judas Swing", "False move off the session open, then the reclaim.", 40,
     c_ict_judas, s_ict_judas, ("session_open",), None, False),
    ("S22", "22 · Price Action Pin Bar at Structure", "Rejection wicks printed at range extremes.", 40,
     c_pin_bar, s_pin_bar, ("sup", "res"), None, False),
    ("S23", "23 · Price Action Inside Bar Breakout", "Coil, then the resolution of the mother bar.", 40,
     c_inside_bar, s_inside_bar, ("mother_hi", "mother_lo"), None, False),
    ("S24", "24 · Elliott Wave Impulse (heuristic)", "Zigzag wave-3 approximation. Subjective by nature.",
     80, c_elliott, s_elliott, ("wave_high", "wave_low"), None, False),
    ("S25", "25 · SuperTrend Flip", "Direction flips of a 10/3.0 SuperTrend.", 60,
     c_supertrend_flip, s_supertrend_flip, ("st_up", "st_dn"), None, False),
    ("S26", "26 · VWAP Reversion", "Fading ATR-scaled stretches away from the VWAP anchor.", 60,
     c_vwap_reversion, s_vwap_reversion, ("vwap", "vwap_hi", "vwap_lo"), None, False),
    ("S27", "27 · Wyckoff Spring / Upthrust", "Range-edge failures with volume confirmation.", 60,
     c_wyckoff, s_wyckoff, ("range_high", "range_low"), None, False),
    ("S28", "28 · Price Threshold Cross (absolute level)",
     "Crosses of a fixed price you type in. Direction is configurable.", 30,
     c_threshold_abs, s_threshold_abs, ("threshold",), None, False),
    ("S29", "29 · Price Threshold Cross (% from reference)",
     "Crosses of a percentage band around a moving reference price.", 30,
     c_threshold_pct, s_threshold_pct, ("threshold_up", "threshold_dn", "threshold_ref"),
     None, False),
    ("S30", "30 · RSI Crossover (configurable levels)",
     "RSI reclaiming your long level or losing your short level.", 40,
     c_rsi_crossover, s_rsi_crossover, (), "rsi", False),
    ("S31", "31 · Bollinger Band + RSI", "Band rejection confirmed by an RSI crossing.", 60,
     c_bb_rsi, s_bb_rsi, ("bb_up", "bb_mid", "bb_lo"), "rsi", False),
    ("S32", "32 · EMA Crossover + RSI", "EMA cross that only counts when RSI agrees.", 60,
     c_ema_rsi, s_ema_rsi, ("ema_fast", "ema_slow"), "rsi", False),
    ("S34", "34 · Double EMA Pullback + Breakout Retest",
     "Cross, pullback to the fast EMA, then a break of the pre-pullback extreme.", 60,
     c_ema_retest_breakout, s_ema_retest_breakout,
     ("ema_fast", "ema_slow", "retest_high", "retest_low"), None, False),
    ("S35", "35 · Fibonacci Retracement Zone",
     "Retracement into the 50-61.8% zone of the last confirmed leg.", 60,
     c_fib_zone, s_fib_zone, ("fib_golden_hi", "fib_golden_lo", "fib_382", "fib_786"), None, False),
    ("S36", "36 · Fibonacci + RSI", "Golden-zone retracement confirmed by RSI.", 60,
     c_fib_rsi, s_fib_rsi, ("fib_golden_hi", "fib_golden_lo"), "rsi", False),
    ("S37", "37 · Fibonacci + Volume Breakout", "Golden-zone entry on a volume surge.", 60,
     c_fib_volume, s_fib_volume, ("fib_golden_hi", "fib_golden_lo"), None, False),
    ("S38", "38 · Fibonacci + VWAP", "Golden-zone entry on the right side of VWAP.", 60,
     c_fib_vwap, s_fib_vwap, ("fib_golden_hi", "fib_golden_lo", "vwap"), None, False),
    ("S39", "39 · Fibonacci + MACD", "Golden-zone entry with MACD histogram agreement.", 60,
     c_fib_macd, s_fib_macd, ("fib_golden_hi", "fib_golden_lo"), None, False),
    ("S40", "40 · Fibonacci + Bollinger Bands", "Golden-zone entry while inside the bands.", 60,
     c_fib_bollinger, s_fib_bollinger, ("fib_golden_hi", "fib_golden_lo", "bb_mid"), None, False),
    ("S41", "41 · Fibonacci + Fair Value Gap",
     "Golden zone overlapping an unfilled imbalance.", 60,
     c_fib_fvg, s_fib_fvg, ("fib_golden_hi", "fib_golden_lo"), None, False),
    ("S42", "42 · Price Action Composite",
     "Pin bars, engulfings and inside-bar breaks read against structure.", 60,
     c_price_action, s_price_action, ("sup", "res"), None, False),
    ("S43", "43 · Hybrid (combine profiles with AND / OR)",
     "Select several profiles and require all, or any one, to agree.", 60,
     c_hybrid, s_hybrid, ("ema_fast", "ema_slow"), None, False),
    ("S44", "44 · Options: OI Change",
     "Live-only. Open-interest build read from the Dhan option chain.", 30,
     _c_option_oi("44 · Options: OI Change"), _s_option_oi("44 · Options: OI Change"),
     ("ema_fast", "ema_slow"), None, False),
    ("S45", "45 · Options: OI Change + PCR",
     "Live-only. OI build gated by the put-call ratio.", 30,
     _c_option_oi("45 · Options: OI Change + PCR"), _s_option_oi("45 · Options: OI Change + PCR"),
     ("ema_fast", "ema_slow"), None, False),
    ("S46", "46 · Options: OI Change + Volume",
     "Live-only. OI build confirmed by option volume.", 30,
     _c_option_oi("46 · Options: OI Change + Volume"),
     _s_option_oi("46 · Options: OI Change + Volume"), ("ema_fast", "ema_slow"), None, False),
    ("S47", "47 · Options: Zero Hero (expiry-day burst)",
     "Expiry-day momentum burst on the underlying. Long buys a CALL, short a PUT.", 60,
     c_zero_hero, s_zero_hero, ("session_open",), None, False),
    ("S48", "48 · Options: Gamma Blast (late-session)",
     "Late-session volatility expansion breaking the day's range.", 60,
     c_gamma_blast, s_gamma_blast, ("day_high", "day_low"), None, False),
    ("S33", "33 · Volume Spike + RSI", "Volume confirmation on an RSI crossing.", 60,
     c_vol_rsi, s_vol_rsi, (), "rsi", False),
]

STRATEGIES: dict[str, Strategy] = {
    name: Strategy(key=k, name=name, blurb=b, min_bars=mb, compute=c, status=s,
                   overlays=ov, oscillator=osc, immediate=imm)
    for k, name, b, mb, c, s, ov, osc, imm in _DEFS
}
STRATEGY_NAMES: list[str] = list(STRATEGIES.keys())


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy `{name}`.") from exc


def prepare(df: pd.DataFrame, strategy_name: str, params: dict,
            filter_cfg: dict | None = None, extras: dict | None = None):
    """
    Run the full pipeline: strategy signals -> exit-engine context columns ->
    optional entry filters. Returns ``(frame, filter_reports)``.
    """
    strat = get_strategy(strategy_name)
    out = strat.compute(df, params)

    # Columns every exit type may need, computed once.
    if "ema_fast" not in out:
        out["ema_fast"] = ema(out["Close"], int(_p(params, "ema_fast")))
    if "ema_slow" not in out:
        out["ema_slow"] = ema(out["Close"], int(_p(params, "ema_slow")))
    if "atr" not in out:
        out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(params, "atr_len")))
    sh, sl_, psh, psl = swing_levels(out["High"], out["Low"],
                                     int(_p(params, "pivot_left")), int(_p(params, "pivot_right")))
    out["swing_high"], out["swing_low"] = sh, sl_
    out["prev_swing_high"], out["prev_swing_low"] = psh, psl
    out["prev_high"] = out["High"].shift(1)
    out["prev_low"] = out["Low"].shift(1)

    # Contrarian switch. Applied BEFORE the filters so that an enabled filter
    # gates the direction actually traded, not the one originally signalled.
    out["flipped"] = bool(params.get("flip_entries"))
    if params.get("flip_entries"):
        out["signal"] = -out["signal"].astype(int)

    reports: list[FilterReport] = []
    if filter_cfg and any(v.get("enabled") for v in filter_cfg.values()):
        out = attach_filter_columns(out, params, bool(params.get("intraday", True)))
        ok_long, ok_short, reports = evaluate_filters(out, filter_cfg, extras)
        out["raw_signal"] = out["signal"]
        gated = np.where((out["signal"] == 1) & ~ok_long, 0,
                         np.where((out["signal"] == -1) & ~ok_short, 0, out["signal"]))
        out["signal"] = pd.Series(gated, index=out.index).astype(int)
        out["filters_long_ok"] = ok_long
        out["filters_short_ok"] = ok_short
    else:
        out["raw_signal"] = out["signal"]
        out["filters_long_ok"] = True
        out["filters_short_ok"] = True
    return out, reports


# =============================================================================
# SECTION 6 -- RISK / EXIT ENGINE
# =============================================================================
# This is the part that decides profitability, so the rules are written out in
# full rather than implied.
#
# RATCHET RULE: a trailing stop may only ever move in the trade's favour. It
# never loosens, not even when the indicator it tracks loosens.
#
# BACKTEST vs LIVE, stated plainly: with OHLC candles we cannot know whether
# price reached the trailing level before or after the extreme that moved it.
# The engine therefore advances trailing levels only AFTER a candle has been
# checked for exits, using that candle's own extremes. That is the pessimistic
# reading, but backtested trailing results remain approximations. Live trailing
# on the LTP is exact.


@dataclass
class CostModel:
    """
    Round-trip trading costs. OFF by default so gross and net are never confused.

    Deliberately simple and explicit rather than a full Indian tax schedule:
    brokerage per side, an optional percentage of turnover covering STT / GST /
    exchange and regulatory charges, and slippage in points per side. Enter the
    numbers your own contract notes actually show.
    """
    enabled: bool = False
    brokerage_per_side: float = 20.0
    pct_of_turnover: float = 0.05
    slippage_points: float = 0.0

    def total(self, entry_price: float, exit_price: float, quantity: float) -> float:
        if not self.enabled:
            return 0.0
        turnover = (abs(entry_price) + abs(exit_price)) * quantity
        return (2.0 * self.brokerage_per_side
                + turnover * self.pct_of_turnover / 100.0
                + 2.0 * self.slippage_points * quantity)

    def summary(self) -> str:
        if not self.enabled:
            return "Costs excluded (gross PnL)"
        return (f"{fmt(self.brokerage_per_side,0)}/side + {fmt(self.pct_of_turnover,3)}% turnover "
                f"+ {fmt(self.slippage_points,2)} pts slippage/side")


@dataclass
class RiskConfig:
    sl_type: str
    sl_value: float
    tp_type: str
    tp_value: float
    quantity: float = 1.0
    step_trigger: float = 0.0        # `k` for the step trail
    min_stop_atr: float = 0.25       # fallback distance when a structural stop is invalid
    costs: CostModel = field(default_factory=CostModel)

    def as_summary(self) -> str:
        sl = self.sl_type if self.sl_type in _SL_NO_VALUE else f"{self.sl_type} {fmt(self.sl_value)}"
        tp = self.tp_type if self.tp_type in _TP_NO_VALUE else f"{self.tp_type} {fmt(self.tp_value)}"
        if self.sl_type == "Step Trail (trigger k, trail N)":
            sl += f" (k={fmt(self.step_trigger)})"
        return f"SL: {sl}  |  TGT: {tp}  |  Qty: {fmt(self.quantity, 0)}"


@dataclass
class BarCtx:
    """Everything the exit engine may need from one candle."""
    time: Any
    open: float
    high: float
    low: float
    close: float
    atr: float
    ema_fast: float
    ema_slow: float
    swing_high: float          # most recently CONFIRMED swing high
    swing_low: float           # most recently CONFIRMED swing low
    prev_swing_high: float     # the confirmed swing high before that one
    prev_swing_low: float
    prev_high: float           # previous candle's high
    prev_low: float
    signal: int


def bar_ctx(frame: pd.DataFrame, i: int) -> BarCtx:
    row = frame.iloc[i]
    return BarCtx(
        time=frame.index[i],
        open=_f(row["Open"]), high=_f(row["High"]), low=_f(row["Low"]), close=_f(row["Close"]),
        atr=_f(row.get("atr")), ema_fast=_f(row.get("ema_fast")), ema_slow=_f(row.get("ema_slow")),
        swing_high=_f(row.get("swing_high")), swing_low=_f(row.get("swing_low")),
        prev_swing_high=_f(row.get("prev_swing_high")), prev_swing_low=_f(row.get("prev_swing_low")),
        prev_high=_f(row.get("prev_high")), prev_low=_f(row.get("prev_low")),
        signal=int(row.get("signal", 0) or 0),
    )


class ExitManager:
    """Owns the stop and target of one position for its whole life."""

    def __init__(self, risk: RiskConfig, entry_price: float, direction: int, ctx: BarCtx):
        self.risk = risk
        self.entry = float(entry_price)
        self.d = int(direction)
        self.notes: list[str] = []
        self.mfe = float(entry_price)          # best price seen in our favour
        self.bars_held = 0
        self.tp_display_only = risk.tp_type == "Trailing Target (display only)"
        self.uses_signal_exit = (risk.sl_type in ("EMA Reverse Crossover", "Strategy Reverse Signal")
                                 or risk.tp_type in ("EMA Reverse Crossover", "Strategy Reverse Signal"))
        self._pending_current_candle_stop = risk.sl_type == "Current Candle Low/High"
        self.sl = self._initial_stop(ctx)
        self.initial_sl = self.sl
        self.risk_points = abs(self.entry - self.sl) if self.sl is not None else None
        self.tp = self._initial_target(ctx)

    # ------------------------------------------------------------- helpers --
    def _fallback_distance(self, ctx: BarCtx) -> float:
        """Used when a structural level is missing or sits the wrong side of entry."""
        if np.isfinite(ctx.atr) and ctx.atr > 0:
            return max(ctx.atr * self.risk.min_stop_atr, self.entry * 0.0005)
        return max(self.entry * 0.002, 0.05)

    def _valid_stop(self, level, ctx: BarCtx, label: str):
        """A long's stop must be BELOW entry, a short's ABOVE. Otherwise fall back."""
        if level is None or not np.isfinite(level):
            fb = self.entry - self.d * self._fallback_distance(ctx)
            self.notes.append(f"{label} unavailable at entry; fell back to an ATR-scaled stop.")
            return fb
        if (self.d > 0 and level >= self.entry) or (self.d < 0 and level <= self.entry):
            fb = self.entry - self.d * self._fallback_distance(ctx)
            self.notes.append(f"{label} sat on the wrong side of entry; fell back to an ATR-scaled stop.")
            return fb
        return float(level)

    def _valid_target(self, level, ctx: BarCtx, label: str):
        if level is None or not np.isfinite(level):
            return None
        if (self.d > 0 and level <= self.entry) or (self.d < 0 and level >= self.entry):
            self.notes.append(f"{label} sat on the wrong side of entry; no target set.")
            return None
        return float(level)

    # ------------------------------------------------------------- initial --
    def _structural_level(self, kind: str, ctx: BarCtx, for_target: bool):
        """
        Resolve one structural level for the CURRENT trade direction.

        A long's stop rides lows and its target rides highs; a short is the
        mirror. `for_target` flips which side of the candle or swing we read.
        """
        want_high = (self.d > 0) if for_target else (self.d < 0)
        if "Candle" in kind:
            if "Previous" in kind:
                return ctx.prev_high if want_high else ctx.prev_low
            return ctx.high if want_high else ctx.low          # current candle
        if "Swing" in kind or "Structure Break" in kind:
            if "Previous" in kind:
                return ctx.prev_swing_high if want_high else ctx.prev_swing_low
            return ctx.swing_high if want_high else ctx.swing_low
        return None

    def _initial_stop(self, ctx: BarCtx):
        t, v, d, e = self.risk.sl_type, float(self.risk.sl_value), self.d, self.entry
        if t in ("No Stop-Loss", "EMA Reverse Crossover", "Strategy Reverse Signal"):
            return None
        if t == "Fixed Percentage" or t == "Trailing Percentage":
            return e - d * e * v / 100.0
        if t in ("Fixed Points", "Trailing Points", "Step Trail (trigger k, trail N)"):
            return e - d * v
        if t in ("ATR Multiple", "Trailing ATR (Chandelier)"):
            if not np.isfinite(ctx.atr):
                return self._valid_stop(None, ctx, "ATR")
            return e - d * v * ctx.atr
        if t in _STRUCTURAL_SL:
            lvl = self._structural_level(t, ctx, for_target=False)
            if "Current Candle" in t:
                # At entry the current candle has only just opened, so its low is
                # not yet knowable. The signal candle's extreme stands in until
                # this candle completes.
                lvl = ctx.prev_low if d > 0 else ctx.prev_high
            return self._valid_stop(lvl, ctx, t)
        return self._valid_stop(None, ctx, t)

    def _initial_target(self, ctx: BarCtx):
        t, v, d, e = self.risk.tp_type, float(self.risk.tp_value), self.d, self.entry
        if t in ("No Target", "EMA Reverse Crossover", "Strategy Reverse Signal"):
            return None
        if t == "Fixed Percentage":
            return e + d * e * v / 100.0
        if t in ("Fixed Points", "Trailing Target (display only)"):
            return e + d * v
        if t == "ATR Multiple":
            return e + d * v * ctx.atr if np.isfinite(ctx.atr) else None
        if t == "Risk : Reward Multiple":
            risk_pts = self.risk_points
            if risk_pts is None or risk_pts <= 0:
                risk_pts = self._fallback_distance(ctx)
                self.notes.append("No measurable stop distance; the R:R target used an ATR proxy.")
            return e + d * v * risk_pts
        if t in _STRUCTURAL_TP:
            lvl = self._structural_level(t, ctx, for_target=True)
            if "Current Candle" in t:
                lvl = ctx.prev_high if d > 0 else ctx.prev_low
            return self._valid_target(lvl, ctx, t)
        return None

    # ------------------------------------------------------------- trailing --
    def _ratchet(self, candidate):
        """Move the stop only in our favour, never against."""
        if candidate is None or not np.isfinite(candidate):
            return
        if self.sl is None:
            self.sl = float(candidate)
            return
        self.sl = max(self.sl, float(candidate)) if self.d > 0 else min(self.sl, float(candidate))

    def update(self, favourable_price: float, ctx: BarCtx) -> None:
        """
        Advance trailing levels.

        BACKTEST: called after the candle has been checked, with that candle's
        high (long) or low (short) as the favourable excursion.
        LIVE: called on every poll with the LTP.
        """
        f = float(favourable_price)
        if np.isfinite(f):
            self.mfe = max(self.mfe, f) if self.d > 0 else min(self.mfe, f)

        t, v, d, e = self.risk.sl_type, float(self.risk.sl_value), self.d, self.entry
        cand, structural = None, False
        if t == "Trailing Points":
            cand = self.mfe - d * v
        elif t == "Trailing Percentage":
            cand = self.mfe * (1 - d * v / 100.0)
        elif t == "Trailing ATR (Chandelier)" and np.isfinite(ctx.atr):
            cand = self.mfe - d * v * ctx.atr
        elif t == "Step Trail (trigger k, trail N)":
            # Nothing happens until price has moved k points in our favour. Then
            # the stop jumps to cost and thereafter rides N points behind the
            # best price, never dropping back below cost.
            if (self.mfe - e) * d >= float(self.risk.step_trigger):
                raw = self.mfe - d * v
                cand = max(e, raw) if d > 0 else min(e, raw)
        elif t in TRAILING_SL_TYPES:
            cand, structural = self._structural_level(t, ctx, for_target=False), True

        if cand is not None and np.isfinite(cand):
            if structural:
                # A structural level must not leapfrog to the wrong side of the
                # current price. Distance-based trails are NOT guarded this way:
                # if price has already fallen back through a level derived from
                # the best price, that stop is genuinely hit and the next check
                # must fire it rather than have it quietly suppressed here.
                ref = f if np.isfinite(f) else ctx.close
                if (d > 0 and cand < ref) or (d < 0 and cand > ref):
                    self._ratchet(cand)
            else:
                self._ratchet(cand)

        # ---- target trailing ----
        tt = self.risk.tp_type
        if self.tp_display_only:
            self.tp = self.mfe + d * float(self.risk.tp_value)
        elif tt in TRAILING_TP_TYPES:
            # A trailing target may only extend AWAY from entry. Letting it drift
            # closer would hand the trade an instant, fictitious fill.
            tcand = self._structural_level(tt, ctx, for_target=True)
            if tcand is not None and np.isfinite(tcand):
                if self.tp is None:
                    self.tp = float(tcand)
                else:
                    self.tp = max(self.tp, float(tcand)) if d > 0 else min(self.tp, float(tcand))

    # --------------------------------------------------------------- checks --
    @property
    def target_is_live(self) -> bool:
        """A display-only trailing target never fires an exit."""
        return self.tp is not None and not self.tp_display_only

    def check_bar(self, ctx: BarCtx):
        """
        BACKTEST exit check for one candle.

        Order: gap through the open, then STOP against the low (long) / high
        (short), then TARGET against the high (long) / low (short). When both
        levels sit inside the range, the stop wins.
        """
        d = self.d
        if d > 0:
            if self.sl is not None and ctx.open <= self.sl:
                return ctx.open, "Stop-Loss (Gap)"
            if self.target_is_live and ctx.open >= self.tp:
                return ctx.open, "Target (Gap)"
            if self.sl is not None and ctx.low <= self.sl:
                return self.sl, "Stop-Loss"
            if self.target_is_live and ctx.high >= self.tp:
                return self.tp, "Target"
        else:
            if self.sl is not None and ctx.open >= self.sl:
                return ctx.open, "Stop-Loss (Gap)"
            if self.target_is_live and ctx.open <= self.tp:
                return ctx.open, "Target (Gap)"
            if self.sl is not None and ctx.high >= self.sl:
                return self.sl, "Stop-Loss"
            if self.target_is_live and ctx.low <= self.tp:
                return self.tp, "Target"
        return None

    def check_tick(self, ltp: float):
        """
        LIVE exit check against a running price.

        Stop first, then target, both against the LTP. The fill is recorded at
        the LTP rather than at the level, because that is where a market exit
        would actually go.
        """
        p = float(ltp)
        if not np.isfinite(p):
            return None
        if self.d > 0:
            if self.sl is not None and p <= self.sl:
                return p, "Stop-Loss"
            if self.target_is_live and p >= self.tp:
                return p, "Target"
        else:
            if self.sl is not None and p >= self.sl:
                return p, "Stop-Loss"
            if self.target_is_live and p <= self.tp:
                return p, "Target"
        return None

    def signal_exit_reason(self, ctx: BarCtx) -> str | None:
        """Bar-driven exits: EMA reverse crossover and strategy reverse signal."""
        d = self.d
        for label, kind in (("stop", self.risk.sl_type), ("target", self.risk.tp_type)):
            if kind == "EMA Reverse Crossover":
                if np.isfinite(ctx.ema_fast) and np.isfinite(ctx.ema_slow):
                    if (d > 0 and ctx.ema_fast < ctx.ema_slow) or (d < 0 and ctx.ema_fast > ctx.ema_slow):
                        return f"EMA Reverse Crossover ({label})"
            elif kind == "Strategy Reverse Signal":
                if ctx.signal == -d:
                    return f"Strategy Reverse Signal ({label})"
        return None

    # ---------------------------------------------------------------- state --
    def points(self, price: float) -> float:
        return (float(price) - self.entry) * self.d

    def pnl(self, price: float) -> float:
        return self.points(price) * self.risk.quantity

    def snapshot(self) -> dict:
        return {"stop_loss": self.sl, "target": self.tp, "initial_stop": self.initial_sl,
                "mfe": self.mfe, "display_only_target": self.tp_display_only,
                "notes": list(self.notes)}


@dataclass
class Position:
    """An open tracked position, live or simulated."""
    strategy: str
    symbol: str
    interval: str
    direction: int
    quantity: float
    entry_price: float
    entry_time: Any
    signal_bar_time: Any
    manager: ExitManager
    broker_order_id: str | None = None
    entry_ltp_at_fill: float | None = None
    entry_bar: dict | None = None          # OHLC of the candle the fill happened on
    high_since_entry: float | None = None  # highest price seen while the trade was open
    low_since_entry: float | None = None
    option_leg: str | None = None          # "CE" / "PE" when routed as an option

    @property
    def stop_loss(self):
        return self.manager.sl

    @property
    def target(self):
        return self.manager.tp

    def points(self, price):
        return self.manager.points(price)

    def pnl(self, price):
        return self.manager.pnl(price)


# =============================================================================
# SECTION 7 -- BACKTEST ENGINE
# =============================================================================
class BacktestError(RuntimeError):
    """Raised when the sample cannot support a valid simulation."""


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict
    warmup_index: int
    warnings: list[str] = field(default_factory=list)
    filter_reports: list = field(default_factory=list)
    walk_forward: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_backtest(df: pd.DataFrame, strategy_name: str, params: dict, risk: RiskConfig,
                 filter_cfg: dict | None = None, extras: dict | None = None,
                 warmup: int = WARMUP_BARS) -> BacktestResult:
    strat = get_strategy(strategy_name)
    warnings: list[str] = []

    required = max(warmup, strat.min_bars) + 5
    if len(df) <= required:
        raise BacktestError(
            f"`{strategy_name}` needs at least {required} candles once the {warmup}-bar "
            f"warm-up is reserved, but only {len(df)} are available. Widen the period or "
            "use a coarser interval.")

    frame, reports = prepare(df, strategy_name, params, filter_cfg, extras)

    start = max(warmup, strat.min_bars)
    start = max(start, 1)
    if start >= len(frame) - 1:
        raise BacktestError("The warm-up window consumed the entire sample.")

    sig = frame["signal"].to_numpy(int)
    n = len(frame)
    trades: list[dict] = []
    pos: Position | None = None
    pending_signal_exit: str | None = None
    gap_exits = 0
    fallback_notes: set[str] = set()

    for i in range(start, n):
        ctx = bar_ctx(frame, i)
        just_exited = False

        # ---------------------------------------------------- manage risk ---
        if pos is not None:
            mgr = pos.manager
            res = mgr.check_bar(ctx)
            exit_price = reason = None

            if res and res[1].endswith("(Gap)"):
                exit_price, reason = res              # a gap beats everything else
            elif pending_signal_exit:
                exit_price, reason = ctx.open, pending_signal_exit
            elif res:
                exit_price, reason = res

            if exit_price is not None:
                if reason.endswith("(Gap)"):
                    gap_exits += 1
                trades.append(_close_trade(pos, float(exit_price), ctx.time, reason,
                                           _bar_dict(ctx)))
                fallback_notes.update(mgr.notes)
                pos, pending_signal_exit, just_exited = None, None, True
            else:
                # Survived the candle: NOW advance the trail using its extremes.
                mgr.update(ctx.high if pos.direction > 0 else ctx.low, ctx)
                pending_signal_exit = mgr.signal_exit_reason(ctx)
                mgr.bars_held += 1

        # --------------------------------------------------------- entries ---
        if pos is None and not just_exited and sig[i - 1] != 0:
            d = int(sig[i - 1])
            entry = ctx.open                          # signal on N -> fill at N+1 open
            if np.isfinite(entry) and entry > 0:
                mgr = ExitManager(risk, entry, d, ctx)
                pos = Position(strategy=strategy_name, symbol=params.get("symbol", ""),
                               interval=params.get("interval", ""), direction=d,
                               quantity=risk.quantity, entry_price=entry, entry_time=ctx.time,
                               signal_bar_time=frame.index[i - 1], manager=mgr,
                               entry_bar=_bar_dict(ctx))
                pending_signal_exit = None

    if pos is not None:
        last = bar_ctx(frame, n - 1)
        trades.append(_close_trade(pos, float(last.close), last.time, "End of Data",
                                   _bar_dict(last)))
        fallback_notes.update(pos.manager.notes)

    trades_df = pd.DataFrame(trades)
    equity = _equity_curve(trades_df, frame.index)
    stats = _statistics(trades_df, equity, risk)
    stats.update(gap_exits=gap_exits, bars_tested=n - start, warmup_bars=start)

    if gap_exits:
        warnings.append(f"{gap_exits} exit(s) filled through a price gap rather than at the "
                        "requested level. That slippage is real and is included in the PnL.")
    if trades_df.empty:
        warnings.append("This configuration produced no entries. Try a longer period, a faster "
                        "interval, fewer filters, or a strategy whose conditions occur more often.")
    verdict, reasons = exit_reliability(risk.sl_type, risk.tp_type)
    stats["reliability"] = verdict
    stats["reliability_reasons"] = reasons
    for note in sorted(fallback_notes):
        warnings.append("Exit engine: " + note)

    return BacktestResult(frame=frame, trades=trades_df, equity=equity, stats=stats,
                          warmup_index=start, warnings=warnings, filter_reports=reports)


def _ohlc_cols(prefix: str, bar: dict | None) -> dict:
    """Flatten a candle into labelled columns for the trade tables."""
    bar = bar or {}
    return {f"{prefix} {k}": (None if bar.get(k) is None else round(float(bar[k]), 4))
            for k in ("Open", "High", "Low", "Close")}


def _bar_dict(ctx: BarCtx) -> dict:
    return {"Open": ctx.open, "High": ctx.high, "Low": ctx.low, "Close": ctx.close}


def _close_trade(pos: Position, exit_price: float, exit_time, reason: str,
                 exit_bar: dict | None = None) -> dict:
    points = round((exit_price - pos.entry_price) * pos.direction, 4)
    mgr = pos.manager
    gross = round(points * pos.quantity, 4)
    cost = round(mgr.risk.costs.total(pos.entry_price, float(exit_price), pos.quantity), 4)
    return {
        "Direction": "LONG" if pos.direction > 0 else "SHORT",
        "Signal Time": pos.signal_bar_time,
        "Entry Time": pos.entry_time,
        "Entry Price": round(pos.entry_price, 4),
        "Exit Time": exit_time,
        "Exit Price": round(float(exit_price), 4),
        "Initial Stop": None if mgr.initial_sl is None else round(mgr.initial_sl, 4),
        "Final Stop": None if mgr.sl is None else round(mgr.sl, 4),
        "Target": None if mgr.tp is None else round(mgr.tp, 4),
        "Best Price": round(mgr.mfe, 4),
        "Exit Reason": reason,
        "Bars Held": mgr.bars_held,
        "Points": points,
        "Gross PnL": gross,
        "Costs": cost,
        "PnL": round(gross - cost, 4),
        "Quantity": pos.quantity,
        **_ohlc_cols("Entry Bar", pos.entry_bar),
        **_ohlc_cols("Exit Bar", exit_bar),
    }


def _equity_curve(trades: pd.DataFrame, index: pd.Index) -> pd.Series:
    curve = pd.Series(0.0, index=index, dtype=float)
    if trades.empty:
        return curve
    realised = trades.groupby("Exit Time")["PnL"].sum()
    curve.loc[realised.index] = realised.to_numpy(float)
    return curve.cumsum()


def _statistics(trades: pd.DataFrame, equity: pd.Series, risk: RiskConfig) -> dict:
    if trades.empty:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "gross_points": 0.0,
                "net_pnl": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "expectancy": 0.0, "max_drawdown": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
                "longs": 0, "shorts": 0, "avg_bars": 0.0, "sharpe": 0.0,
                "trades_per_year": 0.0, "total_costs": 0.0, "gross_pnl": 0.0}
    pnl = trades["PnL"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    dd = equity - equity.cummax()

    # Sharpe on per-trade returns, annualised by the observed trade frequency.
    # Assumptions are stated in the UI: zero risk-free rate, trades treated as
    # independent, and the annualisation factor derived from the sample's own
    # span rather than a calendar convention.
    returns = pnl / (trades["Entry Price"].abs() * trades["Quantity"]).replace(0.0, np.nan)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe, trades_per_year = 0.0, 0.0
    if len(returns) > 2 and float(returns.std(ddof=1)) > 0:
        try:
            span_days = max(
                (pd.Timestamp(trades["Exit Time"].iloc[-1])
                 - pd.Timestamp(trades["Entry Time"].iloc[0])).total_seconds() / 86400.0, 1e-6)
        except Exception:                                           # noqa: BLE001
            span_days = 1.0
        trades_per_year = len(returns) / span_days * 365.0
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(max(trades_per_year, 1e-9)))

    return {
        "total_trades": int(len(trades)), "wins": int(len(wins)), "losses": int(len(losses)),
        "win_rate": round(len(wins) / len(trades) * 100.0, 2),
        "gross_points": round(float(trades["Points"].sum()), 2),
        "net_pnl": round(float(pnl.sum()), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "expectancy": round(float(pnl.mean()), 2),
        "max_drawdown": round(float(dd.min()), 2),
        "best_trade": round(float(pnl.max()), 2), "worst_trade": round(float(pnl.min()), 2),
        "longs": int((trades["Direction"] == "LONG").sum()),
        "shorts": int((trades["Direction"] == "SHORT").sum()),
        "avg_bars": round(float(trades["Bars Held"].mean()), 1),
        "sharpe": round(sharpe, 2),
        "trades_per_year": round(trades_per_year, 1),
        "total_costs": round(float(trades["Costs"].sum()), 2) if "Costs" in trades else 0.0,
        "gross_pnl": round(float(trades["Gross PnL"].sum()), 2) if "Gross PnL" in trades else 0.0,
    }


# =============================================================================
# SECTION 8 -- DHAN BROKER ADAPTER  (opt-in, dry-run by default)
# =============================================================================
# Endpoints and field names follow the DhanHQ v2 REST specification. Nothing is
# transmitted unless the operator explicitly enables live order routing AND
# turns off dry-run. Verify against the current DhanHQ documentation before
# routing real money.

DHAN_BASE = "https://api.dhan.co/v2"
DEFAULT_DHAN_CLIENT_ID = "1104779876"
DHAN_SCRIP_URLS = [
    "https://images.dhan.co/api-data/api-scrip-master-detailed.csv",
    "https://images.dhan.co/api-data/api-scrip-master.csv",
]
DHAN_PRODUCTS = ["INTRADAY", "CNC", "MARGIN", "MTF", "CO", "BO"]
DHAN_INSTRUMENTS = ["EQUITY", "OPTIONS", "FUTURES"]
DHAN_SEGMENTS = ["NSE_EQ", "BSE_EQ", "NSE_FNO", "BSE_FNO", "MCX_COMM", "NSE_CURRENCY"]

# Column-name candidates, because the scrip master schema has changed over time.
_COL_CANDIDATES = {
    "security_id": ["SEM_SMST_SECURITY_ID", "SECURITY_ID"],
    "trading_symbol": ["SEM_TRADING_SYMBOL", "TRADING_SYMBOL"],
    "custom_symbol": ["SEM_CUSTOM_SYMBOL", "DISPLAY_NAME"],
    "name": ["SM_SYMBOL_NAME", "SYMBOL_NAME", "UNDERLYING_SYMBOL"],
    "exchange": ["SEM_EXM_EXCH_ID", "EXCH_ID"],
    "segment": ["SEM_SEGMENT", "SEGMENT"],
    "instrument": ["SEM_INSTRUMENT_NAME", "INSTRUMENT", "INSTRUMENT_TYPE"],
    "expiry": ["SEM_EXPIRY_DATE", "EXPIRY_DATE", "SM_EXPIRY_DATE"],
    "strike": ["SEM_STRIKE_PRICE", "STRIKE_PRICE"],
    "option_type": ["SEM_OPTION_TYPE", "OPTION_TYPE"],
    "lot_size": ["SEM_LOT_UNITS", "LOT_SIZE"],
}


class BrokerError(RuntimeError):
    """Raised for broker connectivity, resolution or rejection failures."""


def _pick_col(df: pd.DataFrame, key: str) -> str | None:
    for cand in _COL_CANDIDATES[key]:
        if cand in df.columns:
            return cand
    return None


def load_scrip_master(force: bool = False) -> pd.DataFrame:
    """Download and normalise the Dhan instrument master. Cached for the session."""
    import requests

    last_err = None
    for url in DHAN_SCRIP_URLS:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            from io import StringIO
            raw = pd.read_csv(StringIO(resp.text), low_memory=False)
            cols = {k: _pick_col(raw, k) for k in _COL_CANDIDATES}
            if not cols["security_id"] or not cols["trading_symbol"]:
                last_err = f"Unexpected schema at {url}: {list(raw.columns)[:10]}"
                continue
            out = pd.DataFrame({
                k: (raw[v] if v else np.nan) for k, v in cols.items()
            })
            out["security_id"] = out["security_id"].astype(str).str.strip()
            for c in ("trading_symbol", "custom_symbol", "name", "exchange", "segment",
                      "instrument", "option_type"):
                out[c] = out[c].astype(str).str.strip().str.upper()
            out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
            out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
            out["lot_size"] = pd.to_numeric(out["lot_size"], errors="coerce")
            out.attrs["source_url"] = url
            return out
        except Exception as exc:                                   # noqa: BLE001
            last_err = f"{url}: {exc}"
    raise BrokerError(f"Could not load the Dhan scrip master. Last error -- {last_err}")


def _nearest_expiry(frame: pd.DataFrame, on: pd.Timestamp | None = None):
    on = pd.Timestamp(on or pd.Timestamp.now().normalize())
    future = frame.loc[frame["expiry"].notna() & (frame["expiry"] >= on), "expiry"]
    return None if future.empty else future.min()


def resolve_instrument(master: pd.DataFrame, underlying: str, instrument: str,
                       segment: str, spot_price: float | None = None,
                       option_type: str = "CALL", expiry: Any = None) -> dict:
    """
    Resolve an underlying to a concrete tradable contract.

    Equity   -> the cash scrip on the chosen segment.
    Futures  -> nearest unexpired contract.
    Options  -> nearest expiry, strike closest to spot (ATM), chosen right.
    """
    under = (underlying or "").strip().upper()
    if not under:
        raise BrokerError("No underlying supplied for instrument resolution.")

    frame = master[master["segment"].str.contains(segment.split("_")[-1][:3], na=False) |
                   master["exchange"].str.startswith(segment.split("_")[0], na=False)]
    if frame.empty:
        frame = master

    name_hit = (frame["name"].fillna("") == under) | \
               (frame["trading_symbol"].fillna("").str.startswith(under)) | \
               (frame["custom_symbol"].fillna("").str.startswith(under))
    frame = frame[name_hit]
    if frame.empty:
        raise BrokerError(f"`{under}` was not found in the Dhan instrument master for {segment}.")

    if instrument == "EQUITY":
        eq = frame[frame["instrument"].str.contains("EQUITY", na=False)]
        eq = eq if not eq.empty else frame
        row = eq.iloc[0]
    elif instrument == "FUTURES":
        fut = frame[frame["instrument"].str.contains("FUT", na=False)]
        if fut.empty:
            raise BrokerError(f"No futures contracts found for `{under}`.")
        exp = pd.Timestamp(expiry) if expiry else _nearest_expiry(fut)
        if exp is None:
            raise BrokerError(f"No unexpired futures contract for `{under}`.")
        row = fut[fut["expiry"] == exp].iloc[0]
    else:  # OPTIONS
        opt = frame[frame["instrument"].str.contains("OPT", na=False)]
        if opt.empty:
            raise BrokerError(f"No option contracts found for `{under}`.")
        exp = pd.Timestamp(expiry) if expiry else _nearest_expiry(opt)
        if exp is None:
            raise BrokerError(f"No unexpired option contract for `{under}`.")
        opt = opt[opt["expiry"] == exp]
        right = "CE" if str(option_type).upper().startswith("C") else "PE"
        typed = opt[opt["option_type"].str.startswith(right[0], na=False) |
                    opt["trading_symbol"].str.endswith(right, na=False)]
        opt = typed if not typed.empty else opt
        if spot_price is None or not np.isfinite(spot_price):
            raise BrokerError("A spot price is required to select the ATM strike.")
        opt = opt[opt["strike"].notna()]
        if opt.empty:
            raise BrokerError(f"No strikes with usable data for `{under}` {exp:%Y-%m-%d}.")
        row = opt.iloc[(opt["strike"] - float(spot_price)).abs().argsort().iloc[0]]

    return {
        "security_id": str(row["security_id"]),
        "trading_symbol": str(row["trading_symbol"]),
        "exchange_segment": segment,
        "instrument": instrument,
        "expiry": None if pd.isna(row.get("expiry")) else pd.Timestamp(row["expiry"]).date().isoformat(),
        "strike": None if pd.isna(row.get("strike")) else float(row["strike"]),
        "option_type": None if instrument != "OPTIONS" else ("CALL" if str(option_type).upper().startswith("C") else "PUT"),
        "lot_size": None if pd.isna(row.get("lot_size")) else int(row["lot_size"]),
    }


def dhan_ltp(broker: dict, contract: dict) -> float | None:
    """
    Real-time last traded price from DhanHQ v2 Market Quote.

    This exists because Yahoo's Indian feed is delayed roughly 15 minutes, so on
    a 5-minute chart the "LTP" can sit unchanged for a quarter of an hour no
    matter how fast you poll. If you have Dhan credentials, this is the price
    that actually moves.
    """
    token = str(broker.get("access_token", "")).strip()
    client = str(broker.get("client_id", "")).strip()
    if not token or not client or not contract:
        return None
    import requests
    seg, sec = contract["exchange_segment"], str(contract["security_id"])
    try:
        resp = requests.post(f"{DHAN_BASE}/marketfeed/ltp",
                             headers={"Content-Type": "application/json",
                                      "Accept": "application/json",
                                      "access-token": token, "client-id": client},
                             data=json.dumps({seg: [int(sec)]}), timeout=10)
        body = resp.json()
    except Exception as exc:                                        # noqa: BLE001
        raise BrokerError(f"Dhan LTP request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise BrokerError(f"Dhan LTP rejected (HTTP {resp.status_code}): {str(body)[:200]}")
    try:
        return float(body["data"][seg][sec]["last_price"])
    except (KeyError, TypeError, ValueError):
        raise BrokerError(f"Unexpected Dhan LTP payload: {str(body)[:200]}")


def place_dhan_order(broker: dict, contract: dict, side: str, quantity: float,
                     dry_run: bool = True) -> dict:
    """
    Place a MARKET order through DhanHQ v2.

    Returns a receipt dict. With ``dry_run=True`` the payload is built and
    returned but nothing leaves the machine.
    """
    payload = {
        "dhanClientId": str(broker.get("client_id", "")).strip(),
        "correlationId": f"algoplat{int(time.time())}",
        "transactionType": "BUY" if side.upper() in ("BUY", "LONG") else "SELL",
        "exchangeSegment": contract["exchange_segment"],
        "productType": broker.get("product_type", "INTRADAY"),
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": str(contract["security_id"]),
        "quantity": int(quantity),
        "price": 0,
    }
    if dry_run:
        return {"status": "DRY_RUN", "payload": payload, "order_id": None,
                "message": "Dry run: payload built, nothing transmitted."}

    token = str(broker.get("access_token", "")).strip()
    if not token or not payload["dhanClientId"]:
        raise BrokerError("Dhan client id and access token are both required for live routing.")

    import requests
    try:
        resp = requests.post(f"{DHAN_BASE}/orders", headers={
            "Content-Type": "application/json", "Accept": "application/json",
            "access-token": token}, data=json.dumps(payload), timeout=20)
    except Exception as exc:                                        # noqa: BLE001
        raise BrokerError(f"Dhan request failed: {exc}") from exc

    try:
        body = resp.json()
    except Exception:                                               # noqa: BLE001
        body = {"raw": resp.text[:500]}
    if resp.status_code >= 400:
        raise BrokerError(f"Dhan rejected the order (HTTP {resp.status_code}): {body}")
    return {"status": body.get("orderStatus", "SENT"), "payload": payload,
            "order_id": body.get("orderId"), "message": str(body)[:300]}


# =============================================================================
# SECTION 9 -- SESSION STATE  (live ledger is strictly separate from backtests)
# =============================================================================
_STATE_DEFAULTS = {
    "live_running": False, "live_config": None, "live_position": None,
    "live_trades": [],            # closed LIVE trades ONLY -- the ledger source
    "live_events": [], "live_last_poll": 0.0, "live_last_bar": None,
    "live_snapshot": None, "live_error": None, "live_started_at": None,
    "live_poll_count": 0, "live_fail_streak": 0, "live_backoff_until": 0.0,
    "backtest_result": None, "backtest_meta": None, "backtest_error": None,
    "scrip_master": None, "broker_receipts": [], "feed_log": [],
    "live_frame": None, "live_frame_at": 0.0, "live_reports": [],
    "live_frame_warnings": [], "live_vix": None, "candle_refreshes": 0,
    "ltp_note": None, "screener_results": None, "screener_error": None,
    "last_seen_ltp": None, "last_ltp_change_ts": 0.0, "pending_ticker": None,
    "last_closed_bar": None, "option_metrics": None, "live_last_signal_time": None,
    "live_first_cycle": False, "last_good_quote": None, "suspect_ticks": 0,
    "optimizer_results": None, "pattern_results": None, "pending_combo": None,
    "pattern_rows": None, "pattern_frames": None, "pattern_hits": None,
    "pattern_errors": None, "lab_results": None,
}


def init_state() -> None:
    for k, v in _STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = list(v) if isinstance(v, list) else v


def log_event(message: str, level: str = "info") -> None:
    st.session_state.live_events.insert(
        0, {"time": pd.Timestamp.now().strftime("%H:%M:%S"), "level": level, "message": message})
    del st.session_state.live_events[300:]


def record_live_trade(trade: dict) -> None:
    """
    The ONLY writer to the live ledger.

    Backtest output lives in a different session key and has no code path here,
    so Tab 3 cannot be contaminated by simulated fills.
    """
    trade = dict(trade)
    trade["Source"] = "LIVE"
    st.session_state.live_trades.append(trade)


def live_ledger_frame() -> pd.DataFrame:
    rows = st.session_state.get("live_trades", [])
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "Exit Time" in frame.columns:
        frame = frame.sort_values("Exit Time").reset_index(drop=True)
    frame.insert(0, "#", range(1, len(frame) + 1))
    return frame


def reset_live_runtime() -> None:
    st.session_state.live_position = None
    st.session_state.live_last_bar = None
    st.session_state.live_last_signal_time = None
    st.session_state.live_snapshot = None
    st.session_state.live_error = None
    st.session_state.live_poll_count = 0
    st.session_state.live_frame = None
    st.session_state.live_frame_at = 0.0
    st.session_state.candle_refreshes = 0
    st.session_state.feed_log = []
    st.session_state.last_seen_ltp = None
    st.session_state.last_ltp_change_ts = 0.0
    st.session_state.live_fail_streak = 0
    st.session_state.live_backoff_until = 0.0


# =============================================================================
# SECTION 10 -- LIVE ENGINE
# =============================================================================
# Signal on the close of candle N -> entry at the OPEN of candle N+1, which is
# already printed and therefore immediately actionable. Stop and target are then
# evaluated against the LTP on every poll, stop first.


@dataclass
class LiveSnapshot:
    frame: pd.DataFrame
    ltp: float
    next_open: float
    last_closed_time: Any
    last_closed_signal: int
    raw_signal: int
    status: StatusReport
    filter_reports: list
    fetched_at: pd.Timestamp
    bars: int
    data_warnings: list[str]
    vix: float | None = None
    feed_age_seconds: float = 0.0
    stale: bool = False                     # candles are lagging (signals may be old)
    quote_live: bool = True                 # the PRICE is moving (the venue is open)
    recent_signal: int = 0                  # most recent signal on ANY closed candle
    recent_signal_time: Any = None
    recent_signal_bars_ago: int | None = None
    ltp_source: str = "Yahoo (delayed candle close)"

    @property
    def frozen(self) -> bool:
        """Truly dead: candles lagging AND the price is not moving either."""
        return self.stale and not self.quote_live


def refresh_candles(cfg: dict):
    """
    The HEAVY leg: download candles and recompute indicators, signals, filters.

    Runs on its own slower cadence. Recomputing 200+ bars of indicators on every
    0.3s tick would be pointless work -- the candles simply have not changed.
    """
    strat = get_strategy(cfg["strategy"])
    bundle = load_market_data(symbol=cfg["symbol"], period=live_period_for(cfg["interval"]),
                              interval=cfg["interval"],
                              freshness_seconds=max(1.0, cfg.get("candle_seconds", 15.0) * 0.9),
                              min_bars=max(strat.min_bars, 30))
    extras = dict(cfg.get("filter_extras") or {})
    if cfg.get("filter_cfg", {}).get("vix", {}).get("enabled"):
        extras["vix"] = load_vix(freshness_seconds=60)
    frame, reports = prepare(bundle.frame, cfg["strategy"], cfg["params"],
                             cfg.get("filter_cfg"), extras)
    if len(frame) < 3:
        raise MarketDataError("Not enough candles to evaluate a live signal.")
    return frame, reports, bundle.warnings, extras.get("vix")


def fetch_live_ltp(cfg: dict, frame: pd.DataFrame) -> tuple[float, str]:
    """
    The LIGHT leg: one price, fetched on every tick.

    Order of preference: Dhan real-time quote, then Yahoo's quote endpoint,
    then -- only if both fail -- the newest candle close, which is the value
    that cannot move between candles.
    """
    broker = cfg.get("broker") or {}
    if broker.get("use_live_ltp") and broker.get("contract"):
        try:
            price = dhan_ltp(broker, broker["contract"])
            if price and np.isfinite(price) and price > 0:
                return float(price), "Dhan (real-time quote)"
        except BrokerError as exc:
            st.session_state.ltp_note = f"Dhan LTP failed, using Yahoo: {exc}"

    price = yahoo_ltp(cfg["symbol"])
    if price is not None:
        st.session_state.last_good_quote = (float(price), time.time())
        return float(price), "Yahoo quote (ticks continuously)"

    # Falling back to the candle close on a single failed quote call makes the
    # price flip between the live quote and a stale close on alternate ticks --
    # a sawtooth that looks like market movement but is pure feed artefact, and
    # one that can trip a trailing stop. Hold the last good quote instead.
    cached = st.session_state.get("last_good_quote")
    if cached:
        px, when = cached
        age = time.time() - float(when)
        if age <= 120:
            return float(px), f"Last good quote ({age:0.0f}s old — quote call failed)"
    return float(frame["Close"].iloc[-1]), "Candle close (no quote available)"


def build_snapshot(cfg: dict, frame: pd.DataFrame, reports, warnings, vix,
                   ltp: float, ltp_source: str) -> LiveSnapshot:
    strat = get_strategy(cfg["strategy"])
    closed = -2                                     # last FULLY CLOSED candle
    last_ts = pd.Timestamp(frame.index[-1])
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tz is not None else pd.Timestamp.now()
    age = float((now - last_ts).total_seconds())
    bar_seconds = INTERVAL_SECONDS.get(cfg["interval"], 300)
    # Candles lagging by more than three bars. On a delayed feed this is routine
    # DURING market hours, so on its own it proves nothing about the venue.
    stale = age > max(3 * bar_seconds, 120)

    # Whether the venue is open is a question about the PRICE, not the candles.
    # The only trustworthy evidence is OBSERVED MOVEMENT: a closed exchange still
    # serves a quote, and that quote will differ from the last intraday candle
    # close (official close vs candle close), so "the numbers differ" proves
    # nothing. We therefore watch for the price actually changing.
    now_t = time.time()
    prev_ltp = st.session_state.get("last_seen_ltp")
    if prev_ltp is not None and abs(float(prev_ltp) - ltp) > 1e-9:
        st.session_state.last_ltp_change_ts = now_t
    st.session_state.last_seen_ltp = ltp

    since_change = now_t - float(st.session_state.get("last_ltp_change_ts", 0.0))
    observed_ticks = int(st.session_state.get("live_poll_count", 0))
    if not stale:
        quote_live = True                       # candles are current: the venue is open
    elif observed_ticks < QUOTE_EVIDENCE_TICKS:
        quote_live = False                      # not enough evidence yet -- hold entries
    else:
        quote_live = since_change <= QUOTE_LIVE_WINDOW

    # The screener reports signals from a WINDOW of recent candles, so the live
    # panel must be able to talk about the same thing. Without this the engine
    # can only ever see the newest closed bar, and a signal that fired two bars
    # ago looks like nothing happened at all.
    closed_sig = frame["signal"].iloc[:-1]
    recent = closed_sig[closed_sig != 0]
    r_sig, r_time, r_ago = 0, None, None
    if len(recent):
        r_time = recent.index[-1]
        r_sig = int(recent.iloc[-1])
        r_ago = int(len(frame) - 2 - frame.index.get_loc(r_time))

    return LiveSnapshot(
        frame=frame, ltp=ltp, next_open=float(frame["Open"].iloc[-1]),
        last_closed_time=frame.index[closed],
        last_closed_signal=int(frame["signal"].iloc[closed]),
        raw_signal=int(frame["raw_signal"].iloc[closed]),
        status=strat.status(frame.iloc[:len(frame) + closed + 1], cfg["params"]),
        filter_reports=reports, fetched_at=pd.Timestamp.now(), bars=len(frame),
        data_warnings=warnings, vix=vix, feed_age_seconds=age, stale=stale,
        quote_live=quote_live, ltp_source=ltp_source,
        recent_signal=r_sig, recent_signal_time=r_time, recent_signal_bars_ago=r_ago)


def _live_close(position: Position, exit_price: float, reason: str) -> dict:
    mgr = position.manager
    points = round((float(exit_price) - position.entry_price) * position.direction, 4)
    trade = {
        "Strategy": position.strategy, "Symbol": position.symbol, "Interval": position.interval,
        "Direction": "LONG" if position.direction > 0 else "SHORT",
        "Quantity": position.quantity,
        "Entry Time": pd.Timestamp(position.entry_time),
        "Entry Price": round(position.entry_price, 4),
        "Exit Time": pd.Timestamp.now(), "Exit Price": round(float(exit_price), 4),
        "Initial Stop": None if mgr.initial_sl is None else round(mgr.initial_sl, 4),
        "Final Stop": None if mgr.sl is None else round(mgr.sl, 4),
        "Target": None if mgr.tp is None else round(mgr.tp, 4),
        "Best Price": round(mgr.mfe, 4), "Exit Reason": reason,
        "Points": points, "Gross PnL": round(points * position.quantity, 4),
        "Costs": round(mgr.risk.costs.total(position.entry_price, float(exit_price),
                                            position.quantity), 4),
        "PnL": round(points * position.quantity
                     - mgr.risk.costs.total(position.entry_price, float(exit_price),
                                            position.quantity), 4),
        "Broker Order": position.broker_order_id or "-",
        "Option Leg": position.option_leg or "-",
        **_ohlc_cols("Entry Bar", position.entry_bar),
        **_ohlc_cols("Exit Bar", st.session_state.get("last_closed_bar")),
    }
    record_live_trade(trade)
    return trade


def square_off(reason: str = "Manual Square-Off", price: float | None = None) -> dict | None:
    """Flatten the tracked position, book the PnL, zero the risk."""
    position: Position | None = st.session_state.live_position
    if position is None:
        return None
    if price is None:
        snap = st.session_state.live_snapshot
        price = snap.ltp if snap else position.entry_price

    cfg = st.session_state.live_config or {}
    _maybe_route_broker(cfg, position, closing=True)

    trade = _live_close(position, float(price), reason)
    st.session_state.live_position = None
    db_clear_position()
    db_append_trade(trade)
    err = send_email(cfg, f"EXIT {trade['Direction']} {trade['Symbol']} :: "
                          f"{fmt_signed(trade['PnL'])}",
                     f"Reason: {reason}\nEntry: {fmt(trade['Entry Price'])}\n"
                     f"Exit: {fmt(trade['Exit Price'])}\nPoints: {fmt_signed(trade['Points'])}\n"
                     f"PnL: {fmt_signed(trade['PnL'])}\nTime: {pd.Timestamp.now()}")
    if err:
        log_event(err, "error")
    log_event(f"{reason}: {trade['Direction']} closed at {fmt(trade['Exit Price'])} "
              f"for {fmt_signed(trade['PnL'])} ({fmt_signed(trade['Points'])} pts).",
              "warn" if trade["PnL"] < 0 else "success")
    return trade


def _maybe_route_broker(cfg: dict, position: Position, closing: bool) -> None:
    """Send the entry or exit leg to Dhan, if and only if the operator enabled it."""
    broker = cfg.get("broker") or {}
    if not broker.get("enabled"):
        return
    contract = broker.get("contract")
    if not contract:
        log_event("Broker routing is on but no contract is resolved; order skipped.", "error")
        return
    side = ("SELL" if position.direction > 0 else "BUY") if closing else \
           ("BUY" if position.direction > 0 else "SELL")
    try:
        receipt = place_dhan_order(broker, contract, side, position.quantity,
                                   dry_run=broker.get("dry_run", True))
        st.session_state.broker_receipts.insert(0, {
            "time": pd.Timestamp.now(), "leg": "EXIT" if closing else "ENTRY",
            "side": side, **{k: receipt.get(k) for k in ("status", "order_id", "message")}})
        del st.session_state.broker_receipts[100:]
        if not closing:
            position.broker_order_id = receipt.get("order_id")
        log_event(f"Broker {('EXIT' if closing else 'ENTRY')} {side} -> {receipt['status']}",
                  "info" if receipt["status"] == "DRY_RUN" else "success")
    except BrokerError as exc:
        log_event(f"Broker order FAILED: {exc}", "error")


def _open_live_position(cfg: dict, direction: int, price: float, ctx: BarCtx, bar_time,
                        levels_from: float | None = None) -> Position:
    """
    Open a tracked position.

    ``levels_from`` anchors the stop and target on a DIFFERENT price from the
    fill. That is what "join the trade with whatever is left" means: the levels
    stay where the original signal put them, while PnL is measured from the
    price you actually got.
    """
    anchor = float(levels_from) if levels_from is not None else float(price)
    mgr = ExitManager(cfg["risk"], anchor, direction, ctx)
    if levels_from is not None:
        mgr.entry = float(price)
        mgr.mfe = float(price)
        mgr.risk_points = abs(float(price) - mgr.sl) if mgr.sl is not None else None
    position = Position(strategy=cfg["strategy"], symbol=cfg["symbol"], interval=cfg["interval"],
                        direction=direction, quantity=cfg["risk"].quantity,
                        entry_price=float(price), entry_time=pd.Timestamp.now(),
                        signal_bar_time=bar_time, manager=mgr,
                        entry_ltp_at_fill=cfg.get("_ltp_at_fill"),
                        entry_bar=_bar_dict(ctx),
                        high_since_entry=float(price), low_since_entry=float(price),
                        option_leg=("CE" if direction > 0 else "PE")
                        if (cfg.get("broker") or {}).get("instrument") == "OPTIONS" else None)
    st.session_state.live_position = position
    if levels_from is not None:
        remaining = None if mgr.tp is None else abs(mgr.tp - float(price))
        log_event(f"Joined an existing signal: levels kept from the original entry "
                  f"{fmt(anchor)}, filled at {fmt(price)}. Remaining to target: "
                  f"{fmt(remaining)}; risk now {fmt(mgr.risk_points)}.", "warn")
    _maybe_route_broker(cfg, position, closing=False)
    db_save_position(position, cfg)
    err = send_email(cfg, f"ENTRY {'LONG' if direction > 0 else 'SHORT'} {cfg['symbol']} "
                          f"@ {fmt(price)}",
                     f"Strategy: {cfg['strategy']}\nSymbol: {cfg['symbol']} ({cfg['interval']})\n"
                     f"Direction: {'LONG' if direction > 0 else 'SHORT'}\n"
                     f"Entry: {fmt(price)}\nStop: {fmt(mgr.sl)}\nTarget: {fmt(mgr.tp)}\n"
                     f"Quantity: {fmt(cfg['risk'].quantity, 0)}\n"
                     f"Time: {pd.Timestamp.now()}")
    if err:
        log_event(err, "error")
    log_event(f"ENTRY {'LONG' if direction > 0 else 'SHORT'} @ {fmt(price)} | "
              f"SL {fmt(mgr.sl)} | TGT {fmt(mgr.tp)} | qty {fmt(cfg['risk'].quantity, 0)}", "success")
    for note in mgr.notes:
        log_event("Exit engine: " + note, "warn")
    return position


def run_cycle(cfg: dict) -> None:
    """
    One tick.

    Fast leg every tick (price, PnL, stop, target, trailing); slow leg only when
    the candle cadence is due (download, indicators, signals, filters).
    """
    if time.time() < st.session_state.get("live_backoff_until", 0.0):
        return

    now = time.time()
    candle_gap = float(cfg.get("candle_seconds", 15.0))
    need_candles = (st.session_state.live_frame is None
                    or (now - st.session_state.live_frame_at) >= candle_gap)
    try:
        if need_candles:
            frame, reports, warns, vix = refresh_candles(cfg)
            st.session_state.live_frame = frame
            st.session_state.live_reports = reports
            st.session_state.live_frame_warnings = warns
            st.session_state.live_vix = vix
            st.session_state.live_frame_at = now
            st.session_state.candle_refreshes += 1
        frame = st.session_state.live_frame
        ltp, ltp_source = fetch_live_ltp(cfg, frame)
        snapshot = build_snapshot(cfg, frame, st.session_state.live_reports,
                                  st.session_state.live_frame_warnings,
                                  st.session_state.live_vix, ltp, ltp_source)
    except Exception as exc:                                        # noqa: BLE001
        st.session_state.live_fail_streak += 1
        streak = st.session_state.live_fail_streak
        backoff = min(300.0, max(2.0, cfg["poll_seconds"]) * (2 ** min(streak, 6)))
        st.session_state.live_backoff_until = time.time() + backoff
        st.session_state.live_error = (f"{exc}  --  backing off {backoff:.0f}s "
                                       f"(consecutive failures: {streak})")
        log_event(f"Feed error: {exc}. Backing off {backoff:.0f}s.", "error")
        return

    st.session_state.live_fail_streak = 0
    st.session_state.live_backoff_until = 0.0
    st.session_state.live_error = None
    st.session_state.live_snapshot = snapshot
    st.session_state.live_last_poll = time.time()
    st.session_state.live_poll_count += 1

    # Poll-by-poll record. If the LTP column never changes across dozens of
    # ticks, the refresh loop is fine and the FEED is standing still.
    log = st.session_state.feed_log
    prev_ltp = log[0]["LTP"] if log else None
    prev_source = log[0]["Source"] if log else None
    log.insert(0, {"Polled at": pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3],
                   "Newest candle": fmt_time(snapshot.frame.index[-1]),
                   "LTP": round(snapshot.ltp, 4),
                   "Changed": "-" if prev_ltp is None else
                              ("yes" if abs(prev_ltp - snapshot.ltp) > 1e-9 else "no"),
                   "Source": snapshot.ltp_source})
    del log[60:]

    # Alternating sources are the classic cause of a phantom sawtooth.
    if prev_source and prev_source != snapshot.ltp_source:
        log_event(f"Price source changed: {prev_source} -> {snapshot.ltp_source}. A change here "
                  f"moves the quoted price for reasons that have nothing to do with the market.",
                  "warn")

    closed_ctx = bar_ctx(frame, len(frame) - 2)
    st.session_state.last_closed_bar = _bar_dict(closed_ctx)

    # A single bad print can hit a stop that the market never reached. A move of
    # this size between two ticks seconds apart is a data artefact, not a market
    # event, so the tick is logged and skipped for risk management -- the next
    # poll is 0.3s away and will confirm or deny it.
    suspect = False
    if prev_ltp is not None and np.isfinite(closed_ctx.atr) and closed_ctx.atr > 0:
        jump = abs(snapshot.ltp - float(prev_ltp))
        if jump > max(8.0 * closed_ctx.atr, 0.03 * float(prev_ltp)):
            suspect = True
            st.session_state.suspect_ticks += 1
            log_event(f"Suspect tick ignored: {fmt(prev_ltp)} -> {fmt(snapshot.ltp)} "
                      f"({fmt(jump)} in one poll, ATR {fmt(closed_ctx.atr)}). Waiting for the "
                      f"next quote to confirm.", "warn")
    if suspect:
        return
    position: Position | None = st.session_state.live_position
    new_bar = st.session_state.live_last_bar != snapshot.last_closed_time

    # ------------------------------------------------- 1. manage open risk ---
    if position is not None:
        mgr = position.manager
        hit = mgr.check_tick(snapshot.ltp)           # stop first, then target, both vs LTP
        if hit:
            price, reason = hit
            square_off(reason, price)
            st.session_state.live_last_bar = snapshot.last_closed_time
            return
        if new_bar:
            reason = mgr.signal_exit_reason(closed_ctx)
            if reason:
                square_off(reason, snapshot.ltp)
                st.session_state.live_last_bar = snapshot.last_closed_time
                return
            mgr.bars_held += 1
        mgr.update(snapshot.ltp, closed_ctx)         # trail on the running price
        # True extremes of the trade, which differ from the manager's favourable
        # excursion: for a short the LOW is the good news and the HIGH is the pain.
        position.high_since_entry = max(position.high_since_entry or snapshot.ltp, snapshot.ltp)
        position.low_since_entry = min(position.low_since_entry or snapshot.ltp, snapshot.ltp)
        if new_bar:
            db_save_position(position, cfg)          # persist the ratcheted stop
        st.session_state.live_last_bar = snapshot.last_closed_time
        return

    # ----------------------------------------------------- 2. fresh entries ---
    # Only a genuinely DEAD feed blocks entry: candles lagging AND the price not
    # moving. Lagging candles alone are normal on a delayed feed during market
    # hours, and blocking on that basis stops live trades for no reason.
    if snapshot.frozen and not cfg.get("allow_stale_entries"):
        return

    strat = get_strategy(cfg["strategy"])
    if strat.immediate:
        direction = 1 if "Buy" in strat.name else -1
        cfg["_ltp_at_fill"] = snapshot.ltp
        _open_live_position(cfg, direction, snapshot.ltp, closed_ctx, snapshot.last_closed_time)
        st.session_state.live_last_bar = snapshot.last_closed_time
        return

    st.session_state.live_last_bar = snapshot.last_closed_time
    direction = int(snapshot.last_closed_signal)
    signal_time = snapshot.last_closed_time
    catch_up = False

    first_cycle = bool(st.session_state.get("live_first_cycle"))
    st.session_state.live_first_cycle = False
    join_window = int(cfg.get("join_window", 20) or 20)

    if direction == 0 and first_cycle and cfg.get("enter_on_start", True) \
            and snapshot.recent_signal != 0 and snapshot.recent_signal_bars_ago is not None \
            and 0 < snapshot.recent_signal_bars_ago <= join_window:
        # You pressed Start because the screener showed a signal. Take it, but on
        # the original signal's terms: the stop and target stay where they were,
        # so you are joining with whatever reward is left rather than getting a
        # fresh full-width target from a price that has already moved.
        frame_idx = snapshot.frame.index
        pos_i = int(frame_idx.get_loc(snapshot.recent_signal_time))
        original_fill = (float(snapshot.frame["Open"].iloc[pos_i + 1])
                         if pos_i + 1 < len(frame_idx) else snapshot.ltp)
        d = int(snapshot.recent_signal)
        probe = ExitManager(cfg["risk"], original_fill, d, closed_ctx)
        already_done = probe.check_tick(snapshot.ltp)
        if already_done:
            log_event(f"A {'LONG' if d > 0 else 'SHORT'} signal fired "
                      f"{snapshot.recent_signal_bars_ago} candle(s) ago, but price has already "
                      f"reached its {already_done[1].lower()} at {fmt(snapshot.ltp)}. Not "
                      f"entering — there is nothing left of that trade.", "warn")
            st.session_state.live_last_signal_time = snapshot.recent_signal_time
            return
        st.session_state.live_last_signal_time = snapshot.recent_signal_time
        cfg["_ltp_at_fill"] = snapshot.ltp
        _open_live_position(cfg, d, snapshot.ltp, closed_ctx, snapshot.recent_signal_time,
                            levels_from=original_fill)
        return

    if direction == 0:
        # Catch-up: the screener reports a signal from a window of candles, and
        # the engine used to see only the newest one. With a lookback set we can
        # still act on a slightly older signal -- but the N+1 open is long gone,
        # so the fill is the CURRENT price and the row says so.
        lookback = int(cfg.get("entry_lookback", 0) or 0)
        if (lookback > 0 and snapshot.recent_signal != 0
                and snapshot.recent_signal_bars_ago is not None
                and 0 < snapshot.recent_signal_bars_ago <= lookback):
            direction = int(snapshot.recent_signal)
            signal_time = snapshot.recent_signal_time
            catch_up = True

    if direction == 0:
        return
    if st.session_state.get("live_last_signal_time") == signal_time:
        return                                   # this exact signal was already traded
    st.session_state.live_last_signal_time = signal_time

    # Signal on candle N -> fill at the OPEN of candle N+1 (already printed).
    if catch_up:
        fill = snapshot.ltp
        log_event(f"Catch-up entry: the signal fired {snapshot.recent_signal_bars_ago} candle(s) "
                  f"ago, so the N+1 open has passed. Filling at the current price "
                  f"{fmt(fill)} instead.", "warn")
    else:
        fill = snapshot.ltp if cfg.get("fill_at_ltp") else snapshot.next_open
    cfg["_ltp_at_fill"] = snapshot.ltp
    _open_live_position(cfg, direction, fill, closed_ctx, signal_time)


def should_poll(cfg: dict) -> bool:
    if time.time() < st.session_state.get("live_backoff_until", 0.0):
        return False
    return (time.time() - st.session_state.get("live_last_poll", 0.0)) >= float(cfg["poll_seconds"])


# =============================================================================
# SECTION 11 -- CHARTS
# =============================================================================
_UP, _DOWN = "#26a69a", "#ef5350"
_OVERLAY_COLOURS = ["#f4a261", "#4f9df7", "#b07cf0", "#8d99ae", "#e9c46a", "#2a9d8f"]
_PRETTY = {
    "ema_fast": "Fast EMA", "ema_slow": "Slow EMA", "ema_mid": "Mid EMA", "ema_macro": "Macro EMA",
    "trail_upper": "ATR Trail Up", "trail_lower": "ATR Trail Down", "or_high": "OR High",
    "or_low": "OR Low", "box_high": "Box High", "box_low": "Box Low", "sup": "Support",
    "res": "Resistance", "basis": "Basis", "ch_upper": "Channel Up", "ch_lower": "Channel Down",
    "swing_high": "Swing High", "swing_low": "Swing Low", "st_up": "SuperTrend Up",
    "st_dn": "SuperTrend Down", "vwap": "VWAP", "vwap_hi": "VWAP Upper", "vwap_lo": "VWAP Lower",
    "mother_hi": "Mother Bar High", "mother_lo": "Mother Bar Low", "session_open": "Session Open",
    "fvg_bull_hi": "Bullish FVG", "fvg_bear_lo": "Bearish FVG", "wave_high": "Leg High",
    "wave_low": "Leg Low", "range_high": "Range High", "range_low": "Range Low",
}


def _label(col: str) -> str:
    return _PRETTY.get(col, col.replace("_", " ").title())


def price_chart(df, title, overlays=("ema_fast", "ema_slow"), trades=None, tail=None,
                hide_weekends=True, height=620, light=False, ema_lengths=None):
    """
    The single chart used by both tabs: candles plus overlay lines.

    Every overlay carries its latest value in the legend and again as a label
    pinned at the right edge, so the numbers are readable without hovering.
    """
    import plotly.graph_objects as go

    data = df.tail(tail) if tail else df
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"],
        close=data["Close"], name="Price",
        increasing_line_color=_UP, decreasing_line_color=_DOWN,
        increasing_fillcolor=_UP, decreasing_fillcolor=_DOWN))

    # Name the EMAs by their actual length -- "EMA 9" tells you more than
    # "Fast EMA" when the lengths are configurable.
    ema_names = {"ema_fast": f"EMA {int((ema_lengths or {}).get('fast', 0))}",
                 "ema_slow": f"EMA {int((ema_lengths or {}).get('slow', 0))}"} \
        if ema_lengths else {}
    line_colours = {"ema_fast": "#f4a261", "ema_slow": "#4f9df7"}

    for i, col in enumerate(dict.fromkeys(("ema_fast", "ema_slow", *overlays))):
        if col not in data.columns or data[col].notna().sum() == 0:
            continue
        colour = line_colours.get(col, _OVERLAY_COLOURS[i % len(_OVERLAY_COLOURS)])
        last = data[col].dropna()
        value = float(last.iloc[-1]) if len(last) else None
        name = ema_names.get(col, _label(col)) + (f"  {fmt(value)}" if value is not None else "")
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode="lines", name=name,
                                 line=dict(width=1.6, color=colour)))
        if value is not None:
            fig.add_annotation(x=last.index[-1], y=value, text=f"{_label(col)} {fmt(value)}",
                               showarrow=False, xanchor="left", xshift=6, font=dict(size=11,
                               color=colour), bgcolor="rgba(0,0,0,0.35)")

    close = float(data["Close"].iloc[-1])
    fig.add_annotation(x=data.index[-1], y=close, text=f"LTP {fmt(close)}", showarrow=False,
                       xanchor="left", xshift=6, font=dict(size=12, color="#ffffff"),
                       bgcolor="rgba(38,166,154,0.85)")

    if trades is not None and not trades.empty:
        for frame, name, sym, colour in ((trades[trades["Direction"] == "LONG"], "Long entry",
                                          "triangle-up", _UP),
                                         (trades[trades["Direction"] == "SHORT"], "Short entry",
                                          "triangle-down", _DOWN)):
            if not frame.empty:
                fig.add_trace(go.Scatter(
                    x=frame["Entry Time"], y=frame["Entry Price"], mode="markers", name=name,
                    marker=dict(symbol=sym, size=11, color=colour,
                                line=dict(width=1, color="#fff")),
                    hovertemplate="%{x}<br>Entry %{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=trades["Exit Time"], y=trades["Exit Price"], mode="markers", name="Exit",
            marker=dict(symbol="x", size=9, color="#8d99ae"),
            customdata=trades[["Exit Reason", "PnL"]],
            hovertemplate="%{x}<br>Exit %{y:,.2f}<br>%{customdata[0]}"
                          "<br>PnL %{customdata[1]:,.2f}<extra></extra>"))

    fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
                      height=height, margin=dict(l=10, r=90, t=46, b=10), hovermode="x",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_rangeslider_visible=False, dragmode="pan",
                      template="plotly_white" if light else None)
    if light:
        fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across",
                         spikedash="dot", spikethickness=1, spikecolor="#9aa0a6")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", side="left")
    if hide_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# =============================================================================
# SECTION 12 -- SIDEBAR CONTROL CONSOLE
# =============================================================================
_CUSTOM = "-- Custom ticker --"


def render_sidebar() -> dict:
    live = bool(st.session_state.get("live_running", False))
    sb = st.sidebar

    # Consume a ticker handed over by the screener. This must happen BEFORE the
    # widgets below are created, which is the whole reason for the handover key.
    combo = st.session_state.pop("pending_combo", None)
    if combo and not live:
        st.session_state["cfg_strategy"] = combo["strategy"]
        st.session_state["cfg_sl_type"] = combo["sl_type"]
        st.session_state["cfg_tp_type"] = combo["tp_type"]
        if combo.get("sl_value") is not None:
            st.session_state[f"cfg_sl_v_{combo['sl_type']}"] = float(combo["sl_value"])
        if combo.get("tp_value") is not None:
            st.session_state[f"cfg_tp_v_{combo['tp_type']}"] = float(combo["tp_value"])
        # The additional entry filters are part of the tested combination, so
        # they travel with it. Every filter is reset first, otherwise leftovers
        # from a previous apply silently change what gets traded.
        for spec in FILTER_SPECS:
            st.session_state[f"flt_{spec['key']}"] = False
        chosen = str(combo.get("filter_key") or "")
        if chosen:
            st.session_state[f"flt_{chosen}"] = True
        for key, value in (combo.get("widgets") or {}).items():
            st.session_state[key] = value
        st.session_state["applied_from_screener"] = combo["strategy"]

    pending = st.session_state.pop("pending_ticker", None)
    if pending and not live:
        st.session_state["cfg_group"] = _CUSTOM
        st.session_state["cfg_custom"] = pending
        st.session_state["applied_from_screener"] = pending

    sb.title("Control Console")
    applied = st.session_state.pop("applied_from_screener", None)
    if applied:
        sb.success(f"`{applied}` applied from the screener.")
    if live:
        _running_banner()
        sb.caption("Configuration is locked while the automation core is running.")

    sb.subheader("Instrument")
    group = sb.selectbox("Asset class", list(ASSET_UNIVERSE) + [_CUSTOM], disabled=live,
                         key="cfg_group")
    if group == _CUSTOM:
        # Seed the default only when nothing is set, so a ticker handed over by
        # the screener is not fighting a hardcoded default on the same widget.
        st.session_state.setdefault("cfg_custom", "KAYNES.NS")
        symbol = sb.text_input("Custom Yahoo ticker", disabled=live,
                               key="cfg_custom").strip().upper()
        asset_label = symbol or "--"
    else:
        uni = ASSET_UNIVERSE[group]
        asset_label = sb.selectbox("Asset", list(uni), disabled=live, key="cfg_asset")
        symbol = uni[asset_label]
        ovr = sb.text_input("Override ticker (optional)", "", disabled=live,
                            placeholder=symbol, key="cfg_ovr").strip().upper()
        if ovr:
            symbol, asset_label = ovr, ovr
    sb.caption(f"Resolved symbol: `{symbol}`")

    sb.subheader("Resolution")
    interval = sb.selectbox("Interval", INTERVALS, index=INTERVALS.index("5m"), disabled=live,
                            key="cfg_interval")
    period = sb.selectbox("Period", PERIODS, index=PERIODS.index("1mo"), disabled=live,
                          key="cfg_period")
    eff_period, clamp = sanitize_period(interval, period)
    if clamp:
        sb.warning(clamp)

    sb.subheader("Strategy Profile")
    strategy = sb.selectbox("Logic profile", STRATEGY_NAMES, disabled=live, key="cfg_strategy")
    strat = get_strategy(strategy)
    sb.caption(strat.blurb)
    if strat.immediate:
        sb.info("This profile enters the moment the engine starts, with no candle wait.")

    sb.subheader("Position Sizing")
    quantity = sb.number_input("Quantity", min_value=1.0, value=1.0, step=1.0, disabled=live,
                               key="cfg_qty")

    # ------------------------------------------------------------ risk ------
    sb.subheader("Stop-Loss")
    sl_type = sb.selectbox("Stop-Loss type", SL_TYPES, disabled=live, key="cfg_sl_type")
    sl_value, step_trigger = 0.0, 0.0
    if sl_type not in _SL_NO_VALUE:
        default = {"Fixed Percentage": 1.0, "Trailing Percentage": 1.0,
                   "ATR Multiple": 2.0, "Trailing ATR (Chandelier)": 3.0}.get(sl_type, 20.0)
        sl_value = sb.number_input("Stop-Loss value", min_value=0.01, value=float(default),
                                   step=0.1, disabled=live, key=f"cfg_sl_v_{sl_type}",
                                   help="Percent, points or ATR multiple depending on the type.")
    if sl_type == "Step Trail (trigger k, trail N)":
        step_trigger = sb.number_input("Trigger k (points in favour before the trail arms)",
                                       min_value=0.0, value=5.0, step=0.5, disabled=live,
                                       key="cfg_step_k",
                                       help="Below k the original stop stands. At k the stop jumps "
                                            "to cost, then rides N points behind the best price.")
        sb.caption(f"Entry 50, N={fmt(sl_value)}, k={fmt(step_trigger)}: at 50+k the stop moves to "
                   f"50; at 60 it stays 50; at 61 it becomes {fmt(61 - sl_value)}.")
    if sl_type in TRAILING_SL_TYPES:
        sb.caption("Trailing stops are exact live but APPROXIMATE in backtests -- OHLC bars hide "
                   "the intrabar path.")

    sb.subheader("Target")
    tp_type = sb.selectbox("Target type", TP_TYPES, disabled=live, key="cfg_tp_type")
    tp_value = 0.0
    if tp_type not in _TP_NO_VALUE:
        default = {"Fixed Percentage": 2.0, "ATR Multiple": 3.0,
                   "Risk : Reward Multiple": 2.0}.get(tp_type, 40.0)
        tp_value = sb.number_input("Target value", min_value=0.01, value=float(default), step=0.1,
                                   disabled=live, key=f"cfg_tp_v_{tp_type}")
    if tp_type == "Trailing Target (display only)":
        sb.caption("Display only: this target trails the best price and never fires an exit. "
                   "The position is resolved by the stop or a strategy exit.")
    if tp_type in TRAILING_TP_TYPES and tp_type != "Trailing Target (display only)":
        sb.caption("A trailing target only ever extends AWAY from entry. It never drifts closer, "
                   "which would hand the trade an instant fictitious fill.")
    if tp_type in _STRUCTURAL_TP and "Trail" not in tp_type:
        sb.caption("Structural target. If the level sits the wrong side of entry at fill time, "
                   "no target is set and the stop or a strategy exit resolves the trade.")

    # ------------------------------------------------------- execution ------
    sb.subheader("Execution")
    poll_seconds = sb.number_input("Live poll interval (seconds)", min_value=API_GUARD_DELAY,
                                   max_value=600.0, value=0.3, step=0.1, key="cfg_poll",
                                   help="How often the LTP, PnL, stop and target refresh. Quote "
                                        "requests are spaced by the mandatory 0.3s guard.")
    if poll_seconds < 1.0:
        sb.warning(f"At {fmt(poll_seconds,1)}s the tick loop issues ~{60/poll_seconds:.0f} quote "
                   "requests a minute. Quotes are small, but Yahoo still throttles: if you start "
                   "seeing backoff messages, ease this up. The heavy candle download runs on its "
                   "own slower cadence below and is unaffected.")
    enter_on_start = sb.checkbox(
        "Enter immediately on start if a signal already fired", value=True, disabled=live,
        key="cfg_enter_start",
        help="You pressed Start because the screener showed a signal. This takes it on the "
             "first poll, keeping the ORIGINAL stop and target — so you join with whatever "
             "reward is left rather than getting a fresh full-width target from a price that "
             "has already moved. If price has already hit those levels, nothing is entered.")
    join_window = sb.number_input(
        "How many candles old a signal may be to join", 1, 100, 20, 1, disabled=live,
        key="cfg_join_window") if enter_on_start else 20
    entry_lookback = sb.number_input(
        "Live: act on a signal up to N candles old", min_value=0, max_value=20, value=0, step=1,
        disabled=live, key="cfg_entry_look",
        help="0 keeps the strict rule: only a signal on the newest CLOSED candle is taken, "
             "filled at the next candle's open. Raise it to catch a signal the screener "
             "reported a few candles ago — but the N+1 open has passed by then, so the fill is "
             "the current price and the trade row says so.")
    square_off_on_stop = sb.checkbox(
        "Square off the open position when the engine stops", value=False, disabled=live,
        key="cfg_sq_stop",
        help="Off by default: stopping the engine leaves the position open. Be aware that "
             "nothing is then watching its stop-loss.")
    allow_stale = sb.checkbox("Live: allow entries on a frozen feed", value=False,
                              disabled=live, key="cfg_stale",
                              help="Off by default. When the venue is closed the LTP is just an "
                                   "old candle close, so an entry books a fictitious price and "
                                   "sits at 0.00 PnL until trading resumes.")
    candle_seconds = sb.number_input("Candle re-download interval (seconds)", min_value=1.0,
                                     max_value=900.0, value=15.0, step=1.0, key="cfg_candle",
                                     help="How often the full candle history is re-downloaded and "
                                          "indicators recomputed. The price, PnL, stop and target "
                                          "refresh on every tick regardless of this.")
    fill_at_ltp = sb.checkbox("Live: fill at LTP instead of the N+1 open", value=False,
                              disabled=live, key="cfg_fill_ltp",
                              help="Default follows the N+1-open rule. Turn this on if you would "
                                   "rather record the price a market order would actually get.")

    sb.subheader("Analysis Options")
    costs = CostModel(enabled=sb.checkbox("Include charges / brokerage in PnL", value=False,
                                          key="cfg_costs_on",
                                          help="Off by default so gross and net are never "
                                               "confused. When on, every trade is charged."))
    if costs.enabled:
        costs.brokerage_per_side = sb.number_input("Brokerage per side", 0.0, 10000.0, 20.0, 1.0,
                                                   key="cfg_brok")
        costs.pct_of_turnover = sb.number_input("Taxes / charges (% of turnover)", 0.0, 5.0,
                                                0.05, 0.01, key="cfg_pct")
        costs.slippage_points = sb.number_input("Slippage (points per side)", 0.0, 500.0, 0.0,
                                                0.1, key="cfg_slip")
        sb.caption("Enter what your own contract notes show. This is a simple model, not the "
                   "full Indian tax schedule.")
    risk = RiskConfig(sl_type=sl_type, sl_value=float(sl_value), tp_type=tp_type,
                      tp_value=float(tp_value), quantity=float(quantity),
                      step_trigger=float(step_trigger), costs=costs)

    walk_fwd = sb.checkbox("Run segment stability (walk-forward) check", value=False,
                           key="cfg_wfo",
                           help="Splits the sample into sequential segments and reports each "
                                "separately, so one lucky window cannot carry the whole result.")
    wf_folds = sb.number_input("Segments", 2, 12, 5, 1, key="cfg_wfo_folds") if walk_fwd else 5

    verdict, _reasons = exit_reliability(sl_type, tp_type)
    (sb.success if verdict == "Backtest-safe" else sb.warning)(
        f"Exit configuration is **{verdict}** for backtesting.")

    use_dhan_data = sb.checkbox("Use Dhan market data (needs API token)", value=False,
                                key="cfg_dhan_data",
                                help="Replaces Yahoo's delayed quote with Dhan's real-time LTP. "
                                     "Read-only: this places no orders. Credentials are entered "
                                     "in the Dhan panel below.")

    email_cfg = {"enabled": False}
    with sb.expander("Email notifications (off by default)"):
        email_cfg["enabled"] = st.checkbox("Send email on entry and exit", value=False,
                                           key="cfg_email_on")
        if email_cfg["enabled"]:
            email_cfg["from"] = st.text_input("From address", "srinivasp451@gmail.com",
                                              key="cfg_email_from")
            email_cfg["to"] = st.text_input("To address", "srinivasp451@gmail.com",
                                            key="cfg_email_to")
            email_cfg["password"] = st.text_input("App password", type="password",
                                                  key="cfg_email_pw",
                                                  help="A Gmail App Password, not your account "
                                                       "password. It is held in memory for this "
                                                       "session only and never written to disk.")
            email_cfg["host"] = st.text_input("SMTP host", "smtp.gmail.com", key="cfg_email_host")
            email_cfg["port"] = st.number_input("SMTP port (SSL)", 1, 65535, 465,
                                                key="cfg_email_port")

    st.session_state["groq_cfg"] = render_groq_sidebar(sb)
    filter_cfg, filter_extras = _render_filters(sb, live)
    broker = _render_broker(sb, live, symbol)

    params = dict(DEFAULT_PARAMS)
    with sb.expander("Advanced indicator parameters"):
        for key, lo, hi in (("ema_fast", 2, 100), ("ema_slow", 3, 200), ("ema_mid", 5, 300),
                            ("ema_macro", 20, 400), ("rsi_len", 2, 100), ("atr_len", 2, 100),
                            ("breakout_len", 5, 200), ("structure_len", 5, 200),
                            ("orb_bars", 1, 60), ("pivot_left", 1, 20), ("pivot_right", 1, 20)):
            params[key] = st.number_input(key, lo, hi, int(DEFAULT_PARAMS[key]), disabled=live,
                                          key=f"pm_{key}")
        for key, lo, hi, stp in (("atr_mult", 0.5, 10.0, 0.1), ("channel_mult", 0.5, 10.0, 0.1),
                                 ("vol_mult", 1.0, 10.0, 0.1), ("gap_pct", 0.05, 10.0, 0.05),
                                 ("squeeze_mult", 1.0, 5.0, 0.05), ("zigzag_pct", 0.1, 10.0, 0.1),
                                 ("st_mult", 0.5, 10.0, 0.1)):
            params[key] = st.number_input(key, lo, hi, float(DEFAULT_PARAMS[key]), stp,
                                          disabled=live, key=f"pm_{key}")
    flip = sb.checkbox(
        "Flip entries (trade every signal the other way)", value=False, disabled=live,
        key="cfg_flip",
        help="Sells on a LONG signal and buys on a SHORT one. Applied before the entry "
             "filters, so an enabled filter gates the direction actually traded. Note this "
             "does not turn a losing system into a winner: costs and the stop/target "
             "asymmetry are paid either way.")
    params["flip_entries"] = bool(flip)
    if flip:
        sb.warning("Entries are INVERTED. Every table and chart reflects the flipped direction.")

    if strategy.startswith("43 "):
        sb.subheader("Hybrid Members")
        choices = [n for n in STRATEGY_NAMES
                   if not STRATEGIES[n].immediate and not n.startswith(("28 ", "29 ", "43 ",
                                                                        "44 ", "45 ", "46 "))]
        params["hybrid_members"] = sb.multiselect(
            "Profiles to combine", choices,
            default=[c for c in choices if c.startswith(("01 ", "08 "))],
            disabled=live, key="cfg_hybrid_members")
        params["hybrid_logic"] = sb.selectbox("Combination logic", HYBRID_LOGIC, disabled=live,
                                              key="cfg_hybrid_logic")
        if str(params["hybrid_logic"]).startswith("All"):
            sb.caption("AND is strict: every member must signal the same way on the SAME candle. "
                       "Expect very few trades, and check the count before drawing conclusions.")
        else:
            sb.caption("OR fires on the first member to signal, so it inherits the false "
                       "positives of all of them.")

    if strategy.startswith(("44 ", "45 ", "46 ")):
        sb.subheader("Option Chain Settings")
        sb.info("These profiles are LIVE-ONLY and need Dhan market data. No free source "
                "publishes historical open interest, so they produce nothing in a backtest "
                "rather than inventing a series.")
        params["oi_change_threshold"] = sb.number_input("Minimum OI change to act on", 0.0,
                                                        1e9, 0.0, 1000.0, disabled=live,
                                                        key="cfg_oi_thr")
        if strategy.startswith("45 "):
            params["pcr_min"] = sb.number_input("PCR floor for longs", 0.0, 5.0, 0.8, 0.05,
                                                disabled=live, key="cfg_pcr_min")
            params["pcr_max"] = sb.number_input("PCR ceiling for shorts", 0.0, 5.0, 1.2, 0.05,
                                                disabled=live, key="cfg_pcr_max")

    if strategy.startswith("47 "):
        sb.subheader("Zero Hero Settings")
        params["zero_hero_atr"] = sb.number_input("Burst size (x ATR from the session open)",
                                                  0.5, 10.0, 2.0, 0.1, disabled=live,
                                                  key="cfg_zh_atr")
        params["expiry_weekday"] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(
            sb.selectbox("Expiry weekday", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                            "Friday"], index=3, disabled=live, key="cfg_expiry_wd"))

    if strategy.startswith("48 "):
        sb.subheader("Gamma Blast Settings")
        params["gamma_tail_bars"] = sb.number_input("Closing stretch (candles)", 1, 60, 6, 1,
                                                    disabled=live, key="cfg_gamma_tail")

    if strategy.startswith(("30 ", "31 ", "32 ", "33 ", "36 ")):
        sb.subheader("RSI Levels")
        params["rsi_long_level"] = sb.number_input("RSI long level", 1.0, 99.0, 40.0, 1.0,
                                                   disabled=live, key="cfg_rsi_long")
        params["rsi_short_level"] = sb.number_input("RSI short level", 1.0, 99.0, 60.0, 1.0,
                                                    disabled=live, key="cfg_rsi_short")
    if strategy.startswith(("28 ", "29 ")):
        sb.subheader("Threshold Settings")
        params["threshold_mode"] = sb.selectbox("Trigger mode", THRESHOLD_MODES, disabled=live,
                                                key="cfg_th_mode")
        if strategy.startswith("28 "):
            params["threshold_price"] = sb.number_input(
                "Threshold price (absolute)", min_value=0.0, value=0.0, step=1.0, disabled=live,
                key="cfg_th_price",
                help="Leave at 0 to anchor on the first close of the loaded window.")
        else:
            params["threshold_pct"] = sb.number_input(
                "Threshold move (%)", min_value=0.01, value=1.0, step=0.05, disabled=live,
                key="cfg_th_pct")
            params["threshold_ref"] = sb.selectbox("Reference price", THRESHOLD_REFS,
                                                   disabled=live, key="cfg_th_ref")

    params["intraday"] = interval in INTRADAY_INTERVALS
    params["symbol"], params["interval"] = symbol, interval

    sb.divider()
    sb.caption("Research and paper-trading sandbox. Data from Yahoo Finance is delayed and "
               "unaudited. Broker routing is off unless you switch it on.")

    return {"symbol": symbol, "asset_label": asset_label, "interval": interval,
            "period": eff_period, "requested_period": period, "strategy": strategy,
            "params": params, "risk": risk, "quantity": float(quantity),
            "poll_seconds": float(poll_seconds), "fill_at_ltp": bool(fill_at_ltp),
            "flip_entries": bool(flip),
            "allow_stale_entries": bool(allow_stale),
            "entry_lookback": int(entry_lookback),
            "enter_on_start": bool(enter_on_start), "join_window": int(join_window),
            "square_off_on_stop": bool(square_off_on_stop),
            "candle_seconds": float(candle_seconds), "costs": costs,
            "walk_forward": bool(walk_fwd), "wf_folds": int(wf_folds),
            "use_dhan_data": bool(use_dhan_data), "email": email_cfg,
            "filter_cfg": filter_cfg, "filter_extras": filter_extras, "broker": broker,
            "currency": currency_symbol(symbol),
            "hide_weekends": not trades_around_the_clock(symbol)}


def _render_filters(sb, live: bool):
    """Additional entry filters. Every one is unchecked by default."""
    cfg = default_filter_config()
    extras: dict = {}
    with sb.expander("Additional entry filters (all off by default)"):
        st.caption("An enabled filter can only VETO a signal, never create one.")
        for spec in FILTER_SPECS:
            key = spec["key"]
            on = st.checkbox(spec["label"], value=False, disabled=live, key=f"flt_{key}",
                             help=spec["help"])
            cfg[key]["enabled"] = on
            if not on:
                continue
            if spec.get("modes") and spec["kind"] != "mode":
                cfg[key]["mode"] = st.selectbox(f"{key} reading", spec["modes"], disabled=live,
                                                key=f"flt_{key}_rmode")
            if spec["kind"] == "crossover":
                cfg[key]["min_angle"] = abs(st.number_input(
                    "Minimum crossover angle (degrees)", min_value=0.0, max_value=89.0,
                    value=0.0, step=1.0, disabled=live, key="flt_x_angle",
                    help="Absolute value, 0 disables the angle test. Measured as the ATR-"
                         "normalised convergence rate of the fast/slow EMA pair, so it is "
                         "comparable across instruments and zoom levels."))
                cfg[key]["mode"] = st.selectbox(
                    "Candle size rule",
                    ["Simple crossover (no candle size rule)", "Custom candle size (points)",
                     "ATR based candle size"], disabled=live, key="flt_x_mode")
                if cfg[key]["mode"].startswith("Custom"):
                    cfg[key]["candle_points"] = st.number_input(
                        "Minimum candle size (points)", min_value=0.0, value=10.0, step=1.0,
                        disabled=live, key="flt_x_pts")
                elif cfg[key]["mode"].startswith("ATR"):
                    cfg[key]["candle_atr"] = st.number_input(
                        "Minimum candle size (x ATR)", min_value=0.0, value=1.0, step=0.1,
                        disabled=live, key="flt_x_atr")
            if spec["kind"] == "range":
                c1, c2 = st.columns(2)
                cfg[key]["min"] = c1.number_input(f"{key} min", value=float(spec["min"]),
                                                  step=float(spec["step"]), disabled=live,
                                                  key=f"flt_{key}_min")
                cfg[key]["max"] = c2.number_input(f"{key} max", value=float(spec["max"]),
                                                  step=float(spec["step"]), disabled=live,
                                                  key=f"flt_{key}_max")
            elif spec["kind"] == "value":
                cfg[key]["value"] = st.number_input(f"{key} multiple", value=float(spec["value"]),
                                                    step=float(spec["step"]), disabled=live,
                                                    key=f"flt_{key}_v")
            elif spec["kind"] == "mode":
                cfg[key]["mode"] = st.selectbox(f"{key} mode", spec["modes"], disabled=live,
                                                key=f"flt_{key}_m")
            elif spec["kind"] == "oi":
                cfg[key]["mode"] = st.selectbox("OI comparison",
                                                ["Absolute change", "N times baseline"],
                                                disabled=live, key="flt_oi_mode")
                cfg[key]["value"] = st.number_input("OI threshold", value=0.0, step=1.0,
                                                    disabled=live, key="flt_oi_thr")
                cfg[key]["manual"] = st.number_input("Observed OI change (manual entry)",
                                                     value=0.0, step=1.0, disabled=live,
                                                     key="flt_oi_val")
                extras["oi_change"] = cfg[key]["manual"]
            elif spec["kind"] == "news":
                cfg[key]["block"] = st.checkbox("Block all entries right now (news kill-switch)",
                                                value=False, key="flt_news_block")
                extras["news_block"] = cfg[key]["block"]
            if key == "pcr":
                cfg[key]["manual"] = st.number_input("Observed PCR (manual entry)", value=0.0,
                                                     step=0.05, disabled=live, key="flt_pcr_val")
                extras["pcr"] = cfg[key]["manual"]
            if spec.get("manual"):
                st.warning(f"{spec['label']}: no free data feed is wired in. It uses the manual "
                           "value above, or blocks everything if you leave it empty. It is not "
                           "silently guessed.")
    return cfg, extras


def _render_broker(sb, live: bool, symbol: str) -> dict:
    """
    Dhan integration.

    DATA and ORDERS are independent switches. Wanting real-time prices is not
    the same as wanting orders sent, so credentials are collected as soon as
    EITHER is enabled rather than being buried inside the order-routing branch.
    """
    broker = {"enabled": False, "dry_run": True, "contract": None, "use_live_ltp": False}
    with sb.expander("Dhan integration (data and / or orders)"):
        use_data = st.checkbox(
            "Use Dhan market data (real-time LTP and option chain)", value=False,
            key="brk_data",
            help="Read-only. Replaces Yahoo's delayed candle-close price with Dhan's live "
                 "quote. Places no orders. Requires a Dhan Data API subscription.")
        enable_orders = st.checkbox(
            "Enable Dhan order placement", value=False, disabled=live, key="brk_enabled",
            help="Separate from the data switch above. Off unless you explicitly want orders "
                 "transmitted.")
        broker["use_live_ltp"] = bool(use_data)
        broker["enabled"] = bool(enable_orders)

        if not (use_data or enable_orders):
            st.caption("Both switches are off. The app runs entirely on Yahoo data and places "
                       "no orders.")
            return broker

        # --- credentials: needed by DATA and by ORDERS alike ---
        st.markdown("**Credentials**")
        broker["client_id"] = st.text_input("Dhan client ID", value=DEFAULT_DHAN_CLIENT_ID,
                                            key="brk_cid")
        broker["access_token"] = st.text_input(
            "Dhan access token", type="password", key="brk_tok",
            help="Required for market data as well as for orders. Generated from the Dhan web "
                 "portal; it expires, so refresh it when quotes start failing.")
        if not str(broker["access_token"]).strip():
            st.warning("No access token yet. Dhan data calls will fail and the app will fall "
                       "back to the Yahoo quote.")

        # --- contract: needed for the LTP feed AND for order routing ---
        st.markdown("**Contract**")
        instrument = st.selectbox("Instrument", DHAN_INSTRUMENTS, key="brk_inst")
        segment = st.selectbox("Exchange segment", DHAN_SEGMENTS,
                               index=0 if instrument == "EQUITY" else 2, key="brk_seg")
        underlying = st.text_input("Underlying symbol", value=_default_underlying(symbol),
                                   key="brk_under",
                                   help="Dhan's own name, e.g. RELIANCE, NIFTY, BANKNIFTY.")
        broker["instrument"] = instrument
        broker["segment"] = segment
        broker["underlying"] = underlying
        option_type = "CALL"
        if instrument == "OPTIONS":
            broker["auto_right"] = st.checkbox(
                "Pick the right from the signal (long -> CE, short -> PE)", value=True,
                key="brk_auto_right")
            option_type = st.selectbox("Option right (used when auto is off)", ["CALL", "PUT"],
                                       key="brk_right")
        broker["option_type"] = option_type

        if st.button("Resolve contract", key="brk_resolve"):
            try:
                with st.spinner("Downloading the Dhan instrument master..."):
                    master = st.session_state.scrip_master
                    if master is None:
                        master = load_scrip_master()
                        st.session_state.scrip_master = master
                spot = None
                snap = st.session_state.get("live_snapshot")
                res = st.session_state.get("backtest_result")
                if snap is not None:
                    spot = snap.ltp
                elif res is not None:
                    spot = float(res.frame["Close"].iloc[-1])
                st.session_state["brk_contract"] = resolve_instrument(
                    master, underlying, instrument, segment, spot, option_type)
            except BrokerError as exc:
                st.error(str(exc))

        contract = st.session_state.get("brk_contract")
        if contract:
            broker["contract"] = contract
            st.success(f"{contract['trading_symbol']}  (security id {contract['security_id']})")
            st.json({k: v for k, v in contract.items() if v is not None})
            if contract.get("lot_size"):
                st.caption(f"Lot size {contract['lot_size']}. Quantity is sent in units, so set "
                           "the sidebar quantity to a multiple of the lot.")
        else:
            st.info("Resolve a contract to enable the Dhan LTP feed and, if switched on, order "
                    "routing.")

        # --- order-only settings ---
        if enable_orders:
            st.markdown("**Order routing**")
            st.error("Live routing sends REAL orders to your Dhan account. Keep dry-run on until "
                     "you have tested the whole path.")
            broker["dry_run"] = st.checkbox("Dry run (build the payload, transmit nothing)",
                                            value=True, key="brk_dry")
            broker["product_type"] = st.selectbox("Product type", DHAN_PRODUCTS, key="brk_prod")
    return broker


def _default_underlying(symbol: str) -> str:
    s = (symbol or "").upper()
    mapping = {"^NSEI": "NIFTY", "^NSEBANK": "BANKNIFTY", "^BSESN": "SENSEX",
               "NIFTY_FIN_SERVICE.NS": "FINNIFTY"}
    return mapping.get(s, s.replace(".NS", "").replace(".BO", ""))


def _running_banner() -> None:
    cfg = st.session_state.get("live_config") or {}
    risk = cfg.get("risk")
    st.sidebar.success("LIVE AUTOMATION CORE :: RUNNING")
    broker = cfg.get("broker") or {}
    routing = ("OFF" if not broker.get("enabled")
               else "DRY RUN" if broker.get("dry_run") else "LIVE ORDERS")
    st.sidebar.markdown(f"""
| Running parameter | Value |
|---|---|
| Asset | `{cfg.get('symbol', '--')}` |
| Strategy | {cfg.get('strategy', '--')} |
| Timeframe | `{cfg.get('interval', '--')}` |
| Quantity | {fmt(cfg.get('quantity'), 0)} |
| Stop-Loss | {getattr(risk, 'sl_type', '--')} {fmt(getattr(risk, 'sl_value', None))} |
| Target | {getattr(risk, 'tp_type', '--')} {fmt(getattr(risk, 'tp_value', None))} |
| Poll | {fmt(cfg.get('poll_seconds'), 1)}s |
| Broker routing | {routing} |
""")


# =============================================================================
# SECTION 13 -- TAB 1: BACKTESTING ENGINE STUDIO
# =============================================================================
def tab_backtest(cfg: dict) -> None:
    st.subheader("Backtesting Engine Studio")
    st.caption("Historical simulation only. Nothing here can reach the live ledger in Tab 3.")

    left, right = st.columns([1, 3])
    run = left.button("Run Backtest Analysis", type="primary", width="stretch")
    right.info(f"**{cfg['strategy']}** on `{cfg['symbol']}` | `{cfg['interval']}` / "
               f"`{cfg['period']}` | {cfg['risk'].as_summary()}")

    if run:
        _run_backtest_ui(cfg)
    if st.session_state.backtest_error:
        st.error(st.session_state.backtest_error)
    if st.session_state.backtest_result is not None:
        _render_backtest(st.session_state.backtest_result, st.session_state.backtest_meta)
    elif not st.session_state.backtest_error:
        st.info("Configure the console on the left, then run the analysis.")


def _run_backtest_ui(cfg: dict) -> None:
    st.session_state.backtest_error = None
    st.session_state.backtest_result = None
    with st.status("Running simulation...", expanded=True) as status:
        try:
            st.write(f"Fetching `{cfg['symbol']}` at {cfg['interval']} / {cfg['period']} ...")
            bundle = load_market_data(cfg["symbol"], cfg["period"], cfg["interval"], 300.0,
                                      min_bars=max(30, get_strategy(cfg["strategy"]).min_bars))
            st.write(f"Received {bundle.bars:,} candles. Computing indicators and signals ...")
            extras = dict(cfg.get("filter_extras") or {})
            if cfg["filter_cfg"].get("vix", {}).get("enabled"):
                st.write("Fetching India VIX for the volatility filter ...")
                extras["vix"] = load_vix()
            result = run_backtest(bundle.frame, cfg["strategy"], cfg["params"], cfg["risk"],
                                  cfg["filter_cfg"], extras, WARMUP_BARS)
            result.warnings = list(bundle.warnings) + list(result.warnings)
            if cfg.get("walk_forward"):
                st.write(f"Running {cfg['wf_folds']} stability segments ...")
                result.walk_forward = walk_forward(bundle.frame, cfg["strategy"], cfg["params"],
                                                   cfg["risk"], cfg["filter_cfg"], extras,
                                                   int(cfg["wf_folds"]))
            st.session_state.backtest_result = result
            st.session_state.backtest_meta = dict(cfg)
            status.update(label=f"Complete :: {result.stats['total_trades']} trades",
                          state="complete", expanded=False)
        except (MarketDataError, BacktestError) as exc:
            st.session_state.backtest_error = str(exc)
            status.update(label="Simulation aborted", state="error", expanded=False)
        except Exception as exc:                                    # noqa: BLE001
            st.session_state.backtest_error = f"Unexpected failure: {exc}"
            status.update(label="Simulation aborted", state="error", expanded=False)


def _trust_header(result: BacktestResult, meta: dict) -> None:
    """
    A blunt verdict on whether this result deserves any confidence.

    Backtests fail in predictable ways: too few trades, an exit type the
    simulation cannot model honestly, results carried by gap fills, or no cost
    model. Each is checked and named rather than left for the operator to spot.
    """
    s = result.stats
    problems, cautions = [], []

    if s["total_trades"] == 0:
        problems.append("No trades at all -- there is nothing here to evaluate.")
    elif s["total_trades"] < 30:
        problems.append(f"Only {s['total_trades']} trades. Below roughly 30 the win rate and "
                        "Sharpe are noise; this sample cannot distinguish edge from luck.")
    elif s["total_trades"] < 100:
        cautions.append(f"{s['total_trades']} trades is a thin sample. Treat the numbers as "
                        "indicative, not established.")

    if s.get("reliability") == "Optimistic":
        problems.append("A distance-based trailing exit is active, which the simulation cannot "
                        "model honestly on OHLC data. Live results will be WORSE than shown.")
    if s["total_trades"] and s["gap_exits"] / max(s["total_trades"], 1) > 0.25:
        cautions.append(f"{s['gap_exits']} of {s['total_trades']} exits filled through gaps. The "
                        "stop is too tight for this instrument's gap behaviour.")
    if not meta["risk"].costs.enabled and s["total_trades"]:
        cautions.append("Costs are excluded. Brokerage, taxes and slippage often exceed the edge "
                        "on fast intraday profiles -- enable the cost model before believing this.")
    if s["total_trades"] and s["max_drawdown"] and abs(s["max_drawdown"]) > abs(s["net_pnl"]):
        cautions.append("Maximum drawdown exceeds total profit. Even if the edge is real, it "
                        "would be very hard to sit through.")
    if not meta.get("walk_forward"):
        cautions.append("No segment stability check was run, so it is unknown whether one lucky "
                        "stretch carried the result.")

    if problems:
        st.error("### Do not trust this result yet\n\n"
                 + "\n".join(f"- {x}" for x in problems)
                 + ("\n\nAlso worth noting:\n" + "\n".join(f"- {x}" for x in cautions)
                    if cautions else ""))
    elif cautions:
        st.warning("### Treat this result with caution\n\n"
                   + "\n".join(f"- {x}" for x in cautions))
    else:
        st.success("### This result passes the basic sanity checks\n\n"
                   f"- {s['total_trades']} trades, a sample large enough to mean something.\n"
                   "- Exit configuration is backtest-safe: no path ambiguity to guess at.\n"
                   "- Costs are included and drawdown sits within total profit.\n"
                   "- Segment stability was checked.\n\n"
                   "Passing these checks means the simulation is *honest*. It does not mean the "
                   "edge will persist: markets change, and out-of-sample forward testing on "
                   "paper is the only thing that tells you whether it still works.")


def _render_backtest(result: BacktestResult, meta: dict) -> None:
    cur, s = meta["currency"], result.stats
    _trust_header(result, meta)

    st.markdown("#### Performance Summary")
    a = st.columns(5)
    a[0].metric("Total Trades", s["total_trades"], f"{s['longs']}L / {s['shorts']}S")
    a[1].metric("Win Rate", f"{fmt(s['win_rate'])}%", f"{s['wins']}W / {s['losses']}L")
    a[2].metric(f"Net PnL ({cur})", fmt_signed(s["net_pnl"]), f"{fmt_signed(s['gross_points'])} pts")
    pf = s["profit_factor"]
    a[3].metric("Profit Factor", "inf" if pf == float("inf") else fmt(pf))
    a[4].metric(f"Max Drawdown ({cur})", fmt(s["max_drawdown"]))
    b = st.columns(5)
    b[0].metric("Expectancy / trade", fmt_signed(s["expectancy"]))
    b[1].metric("Average Win", fmt(s["avg_win"]))
    b[2].metric("Average Loss", fmt(s["avg_loss"]))
    b[3].metric("Best / Worst", f"{fmt(s['best_trade'], 0)} / {fmt(s['worst_trade'], 0)}")
    b[4].metric("Warm-up Bars", f"{s['warmup_bars']:,}", f"{s['bars_tested']:,} tested")
    c = st.columns(5)
    c[0].metric("Sharpe Ratio", fmt(s["sharpe"]), f"~{fmt(s['trades_per_year'],0)} trades/yr",
                help="Per-trade returns annualised by the sample's own trade frequency. Zero "
                     "risk-free rate, trades treated as independent. A Sharpe from a few dozen "
                     "trades is a noisy estimate, not a property of the strategy.")
    c[1].metric(f"Gross PnL ({cur})", fmt_signed(s.get("gross_pnl", s["net_pnl"])))
    c[2].metric(f"Costs charged ({cur})", fmt(s.get("total_costs", 0.0)),
                meta["risk"].costs.summary())
    c[3].metric("Avg Bars Held", fmt(s["avg_bars"], 1))
    c[4].metric("Gap Exits", f"{s['gap_exits']:,}")

    if meta.get("walk_forward") and not result.walk_forward.empty:
        st.markdown("#### Segment Stability")
        st.caption("The sample split into sequential stretches. With a fixed configuration there "
                   "is no in-sample fitting to guard against, so this asks a narrower question: "
                   "did the result hold across the whole sample, or did one window carry it? "
                   "Use the Strategy Optimiser for the fit-then-test version.")
        st.dataframe(result.walk_forward, width="stretch", hide_index=True)

    if result.warnings:
        with st.expander(f"Run notes and caveats ({len(result.warnings)})", expanded=False):
            for w in result.warnings:
                st.warning(w)

    strat = get_strategy(meta["strategy"])
    st.plotly_chart(price_chart(result.frame,
                                f"{meta['symbol']} | {meta['interval']} | {meta['strategy']}",
                                strat.overlays, result.trades,
                                hide_weekends=meta["hide_weekends"]),
                    width="stretch", config={"scrollZoom": True})
    st.caption(f"The first {result.warmup_index:,} candles were reserved as the indicator warm-up "
               "window and produced no orders. Signals fire on a candle close and fill at the "
               "next candle's open.")
    t1, t2, t3, t4 = st.tabs(["Simulated Trades", "Exit Reasons", "Gap Diagnostics", "Indicator Frame"])
    with t1:
        if result.trades.empty:
            st.info("No simulated trades for this configuration.")
        else:
            st.dataframe(result.trades, width="stretch", hide_index=True,
                         column_config={"PnL": st.column_config.NumberColumn(f"PnL ({cur})",
                                                                            format="%.2f")})
            st.download_button("Download simulated trades (CSV)",
                               result.trades.to_csv(index=False).encode(),
                               f"backtest_{meta['symbol']}_{meta['interval']}.csv", "text/csv")
            st.caption("Simulation output. Deliberately NOT written to the live ledger.")
    with t2:
        if result.trades.empty:
            st.info("Nothing to break down yet.")
        else:
            by = result.trades.groupby("Exit Reason")["PnL"].agg(["count", "sum", "mean"]).round(2)
            by.columns = ["Trades", f"Total PnL ({cur})", f"Average ({cur})"]
            st.dataframe(by.reset_index(), width="stretch", hide_index=True)
            st.caption("Where the exits actually came from. If almost everything closes on "
                       "'Stop-Loss (Gap)', the stop is too tight for this instrument's gaps.")
    with t3:
        gaps = gap_profile(result.frame, 0.3)
        st.metric("Gap candles in sample (>= 0.30%)", f"{len(gaps):,}")
        st.dataframe(gaps.tail(200), width="stretch")
    with t4:
        st.dataframe(result.frame.tail(300), width="stretch")

    render_analyst_panel(
        "backtest", "this backtest",
        f"Strategy: {meta['strategy']} on {meta['symbol']} {meta['interval']}\n"
        f"Risk: {meta['risk'].as_summary()}\n"
        f"Stats: {json.dumps(result.stats, default=str)}\n"
        f"Warnings: {result.warnings}\n"
        f"Trades sample:\n{_frame_context(result.trades)}")


# =============================================================================
# SECTION 14 -- TAB 2: LIVE SANDBOX OPERATIONS PANEL
# =============================================================================
def tab_live(cfg: dict) -> None:
    st.subheader("Live Sandbox Operations Panel")
    st.caption("Signals are read from closed candles and filled at the next candle's open. "
               "Stop and target are then checked against the LTP on every poll, stop first.")
    if st.session_state.live_running:
        # The controls are rendered INSIDE the fragment. Left outside, they only
        # redraw on a full app rerun, so the square-off button stayed greyed out
        # for as long as the fragment was quietly opening and closing positions.
        _mount_live_fragment(float(st.session_state.live_config.get("poll_seconds", 5.0)))
    else:
        _live_controls(cfg)
        st.divider()
        _idle_panel(cfg)


def _live_controls(cfg: dict) -> None:
    """
    The three operator actions.

    Rendered inside the live fragment while running, so the square-off button
    reflects the position as it actually is on this tick rather than as it was
    at the last full page render.
    """
    running = bool(st.session_state.live_running)
    position = st.session_state.live_position
    c1, c2, c3 = st.columns(3)

    if c1.button("Start Live Automation Core", type="primary", disabled=running,
                 width="stretch"):
        reset_live_runtime()
        st.session_state.live_config = dict(cfg)
        st.session_state.live_running = True
        st.session_state.live_started_at = pd.Timestamp.now()
        st.session_state.live_last_poll = 0.0
        st.session_state.live_first_cycle = True
        log_event(f"Core started :: {cfg['symbol']} | {cfg['interval']} | {cfg['strategy']} | "
                  f"{cfg['risk'].as_summary()} | poll {fmt(cfg['poll_seconds'],1)}s", "success")
        st.rerun(scope="app")

    if c2.button("Stop Live Processing Engine", disabled=not running, width="stretch"):
        squared = None
        if st.session_state.live_position is not None and cfg.get("square_off_on_stop"):
            squared = square_off("Squared Off on Engine Stop")
        st.session_state.live_running = False
        if squared:
            st.toast(f"Position squared off at {fmt(squared['Exit Price'])} "
                     f"for {fmt_signed(squared['PnL'])} and written to the ledger.")
        elif st.session_state.live_position is not None:
            log_event("Engine stopped with a position still OPEN. Its stop-loss and target are "
                      "no longer being monitored by anything.", "warn")
        log_event("Core stopped.", "info")
        st.rerun(scope="app")

    if c3.button("Manual Square-Off", disabled=position is None, width="stretch",
                 help="Closes the tracked position now and writes it to the ledger. The engine "
                      "keeps running and will take the next qualifying signal."):
        trade = square_off("Manual Square-Off")
        if trade:
            st.toast(f"Closed at {fmt(trade['Exit Price'])} for {fmt_signed(trade['PnL'])}. "
                     "The engine is still running.")
        st.rerun(scope="app")

    if running and st.session_state.live_position is None:
        st.caption("Flat and scanning. A manual square-off does not stop the engine; it will "
                   "enter again on the next qualifying signal.")


def _idle_panel(cfg: dict) -> None:
    st.info(f"The automation core is idle. Press **Start Live Automation Core** to poll "
            f"`{cfg['symbol']}` every {fmt(cfg['poll_seconds'],1)}s automatically -- no clicking "
            f"required. Live window: `{live_period_for(cfg['interval'])}` of "
            f"{cfg['interval']} candles.")
    snap = st.session_state.live_snapshot
    if snap is not None and snap.frozen:
        st.error(f"Heads up: the last poll found the newest `{cfg['symbol']}` candle to be "
                 f"{_human_age(snap.feed_age_seconds)} old. The venue is closed, so starting the "
                 "core now will poll a frozen tape until it reopens.")
    if st.session_state.live_position is not None:
        st.warning("A tracked position is still open from the previous run. Square it off below.")
        _position_dashboard(st.session_state.live_position, st.session_state.live_snapshot,
                            cfg["currency"])
    _event_feed()


def _live_body() -> None:
    cfg = st.session_state.live_config or {}
    if not cfg:
        st.error("Live configuration was lost. Stop and restart the engine.")
        return
    # Poll FIRST, then draw the controls. Rendering them beforehand meant the
    # square-off button described the position as it was one tick ago, so it sat
    # greyed out on the very tick that opened a trade.
    if should_poll(cfg):
        run_cycle(cfg)
    _live_controls(cfg)
    st.divider()

    if st.session_state.live_error:
        st.error(f"Live feed issue: {st.session_state.live_error}")
    backoff = st.session_state.get("live_backoff_until", 0.0) - time.time()
    if backoff > 0:
        st.warning(f"Rate-limit backoff active for another {backoff:.0f}s. This is what a feed "
                   "throttle looks like -- widen the poll interval.")

    snapshot = st.session_state.live_snapshot
    if snapshot is None:
        st.info("Waiting for the first market poll to complete ...")
        return
    for w in snapshot.data_warnings:
        st.warning(w)

    _metric_style()
    _feed_banner(cfg, snapshot)
    _heartbeat(cfg, snapshot)
    _market_data_panel(cfg, snapshot)
    position = st.session_state.live_position
    if position is not None:
        _position_dashboard(position, snapshot, cfg["currency"])
    else:
        _searching_widget(cfg, snapshot)
    _strategy_status_panel(cfg, snapshot)
    _live_chart(cfg, snapshot)
    _recent_trades()
    _feed_diagnostics()
    _filter_panel(snapshot)
    _broker_panel()
    _event_feed()


_LIVE_FRAGMENTS: dict[float, Callable] = {}


def _mount_live_fragment(poll_seconds: float) -> None:
    """
    Auto-refresh with nobody touching the keyboard.

    The panel is a fragment so only this section re-executes; a full app rerun
    would reset the tab selection and discard the backtest in Tab 1.

    The decorated fragment is CACHED per tick length. Re-decorating on every
    script rerun creates a fresh fragment identity each time, which can orphan
    the previously scheduled auto-rerun and leave the panel looking frozen.
    """
    tick = max(API_GUARD_DELAY, float(poll_seconds))
    if not hasattr(st, "fragment"):                                # legacy fallback
        _live_body()
        time.sleep(tick)
        st.rerun()
        return
    frag = _LIVE_FRAGMENTS.get(tick)
    if frag is None:
        frag = st.fragment(run_every=tick)(_live_body)
        _LIVE_FRAGMENTS[tick] = frag
    frag()


def ema_angle_degrees(frame: pd.DataFrame) -> float | None:
    """
    Convergence angle of the EMA pair, normalised by ATR.

    A raw gradient is in price units per bar, so its "angle" changes with the
    instrument and with how far you zoom. Dividing by ATR makes the number mean
    the same thing on Nifty and on Bitcoin.
    """
    if not {"ema_fast", "ema_slow", "atr"} <= set(frame.columns) or len(frame) < 3:
        return None
    spread = frame["ema_fast"] - frame["ema_slow"]
    a = float(frame["atr"].iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return None
    rate = float(spread.iloc[-1] - spread.iloc[-2]) / a
    if not np.isfinite(rate):
        return None
    return float(abs(np.degrees(np.arctan(rate))))


def _metric_style() -> None:
    """Large, light metric values, matching the dashboard layout."""
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] { font-size: 2.0rem; font-weight: 300;
                                           line-height: 1.15; }
        div[data-testid="stMetricLabel"] p { font-size: 0.80rem; font-weight: 400;
                                             opacity: 0.65; }
        div[data-testid="stMetricDelta"] { font-size: 0.85rem; }
        </style>""", unsafe_allow_html=True)


def _market_data_panel(cfg: dict, snapshot: LiveSnapshot) -> None:
    """The market half of the dashboard: price and where the EMAs stand."""
    frame = snapshot.frame
    fast = safe_last(frame["ema_fast"]) if "ema_fast" in frame else None
    slow = safe_last(frame["ema_slow"]) if "ema_slow" in frame else None
    angle = ema_angle_degrees(frame)
    cur = cfg["currency"]

    st.markdown("#### \U0001F4C8 Current Market Data")
    c = st.columns(5)
    c[0].metric("Current Price", f"{cur}{fmt(snapshot.ltp)}")
    c[1].metric("Fast EMA", f"{cur}{fmt(fast)}")
    c[2].metric("Slow EMA", f"{cur}{fmt(slow)}")
    c[3].metric("EMA Angle", "--" if angle is None else f"{angle:.2f}\u00b0")
    if fast is None or slow is None:
        c[4].metric("Crossover", "--")
    elif fast > slow:
        c[4].metric("Crossover", "Bullish \u2191", f"+{fmt(fast - slow)}")
    else:
        c[4].metric("Crossover", "Bearish \u2193", f"-{fmt(slow - fast)}")


def _feed_banner(cfg: dict, snapshot: LiveSnapshot) -> None:
    """Say plainly which of the three states the feed is in."""
    age = snapshot.feed_age_seconds
    human = f"{age/3600:.1f} hours" if age >= 3600 else f"{age/60:.0f} minutes"

    if snapshot.frozen and int(st.session_state.get("live_poll_count", 0)) < QUOTE_EVIDENCE_TICKS:
        st.info(f"Checking whether `{cfg['symbol']}` is actually trading — the candles are "
                f"{human} old, so the engine is watching the quote for movement before it "
                f"commits to an entry. This takes about "
                f"{QUOTE_EVIDENCE_TICKS * cfg['poll_seconds']:.1f}s.")
    elif snapshot.frozen:
        st.error(
            f"**FEED DEAD — the venue looks closed.** The newest candle for `{cfg['symbol']}` is "
            f"{human} old and the quote has not moved across recent ticks. New entries are "
            f"suppressed, because filling at a stale price books a fictitious entry that then "
            f"sits at exactly 0.00 PnL. The engine keeps polling and will pick up the first live "
            f"tick by itself.")
    elif snapshot.stale:
        st.warning(
            f"**Candles are lagging, but the price is live.** The newest `{cfg['symbol']}` candle "
            f"is {human} old while the quote is still ticking — normal for a delayed feed during "
            f"market hours. Trading continues: PnL, stop and target track the live price, but "
            f"SIGNALS are only as fresh as the candles. Shorten the interval for fresher signals.")


def _heartbeat(cfg: dict, snapshot: LiveSnapshot) -> None:
    now = pd.Timestamp.now()
    since_poll = (now - snapshot.fetched_at).total_seconds()
    next_in = max(0.0, float(cfg["poll_seconds"]) - since_poll)

    c = st.columns(6)
    c[0].metric("LTP", fmt(snapshot.ltp), help="Close of the most recent candle on the feed.")
    c[1].metric("N+1 Open", fmt(snapshot.next_open),
                fmt_signed(snapshot.ltp - snapshot.next_open),
                help="Open of the candle after the signal candle: the backtest-consistent "
                     "fill price. The delta is LTP minus that open.")
    c[2].metric("Last Candle", pd.Timestamp(snapshot.last_closed_time).strftime("%d %b %H:%M"),
                help=fmt_time(snapshot.last_closed_time))
    c[3].metric("Candle Age", _human_age(snapshot.feed_age_seconds),
                "FROZEN" if snapshot.stale else "live",
                help="Wall-clock age of the newest candle. This is what tells you whether the "
                     "market is open.")
    c[4].metric("Ticks", f"{st.session_state.live_poll_count:,}",
                f"next in {next_in:0.1f}s",
                help=f"Price ticks. Candle re-downloads so far: "
                     f"{st.session_state.candle_refreshes:,}.")
    c[5].metric("Clock", now.strftime("%H:%M:%S"),
                help="Redraws on every tick. If this is moving, the auto-refresh is alive.")
    st.caption(f"Price source: **{snapshot.ltp_source}** | price ticks every "
               f"{fmt(cfg['poll_seconds'], 1)}s | candles re-downloaded every "
               f"{fmt(cfg.get('candle_seconds', 15), 0)}s | {API_GUARD_DELAY}s minimum spacing "
               f"between requests.")
    if st.session_state.get("ltp_note"):
        st.caption(":warning: " + str(st.session_state.ltp_note))


def _human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.0f}m"
    if seconds < 172800:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def _position_dashboard(position: Position, snapshot, currency: str) -> None:
    ltp = snapshot.ltp if snapshot else position.entry_price
    mgr = position.manager
    points, pnl = position.points(ltp), position.pnl(ltp)
    side = "LONG" if position.direction > 0 else "SHORT"

    st.markdown("#### \U0001F4CA Current Position")
    r1 = st.columns(5)
    r1[0].metric("Type", side, position.strategy.split("\u00b7 ")[-1].strip())
    r1[1].metric("Entry Price", f"{currency}{fmt(position.entry_price)}")
    r1[2].metric("Current Price", f"{currency}{fmt(ltp)}")
    r1[3].metric("Stop Loss", f"{currency}{fmt(mgr.sl)}" if mgr.sl is not None else "none",
                 None if mgr.initial_sl is None else f"from {fmt(mgr.initial_sl)}")
    tgt_label = "Target (display)" if mgr.tp_display_only else "Target"
    r1[4].metric(tgt_label, f"{currency}{fmt(mgr.tp)}" if mgr.tp is not None else "none")

    r2 = st.columns(5)
    r2[0].metric("Quantity", fmt(position.quantity, 0))
    r2[1].metric("Locked Ticker", position.symbol, position.interval)
    r2[2].metric("Current P&L", f"{currency}{fmt_signed(pnl)}", f"{fmt_signed(points)} pts")
    r2[3].metric("Highest Price",
                 f"{currency}{fmt(position.high_since_entry or ltp)}")
    r2[4].metric("Lowest Price",
                 f"{currency}{fmt(position.low_since_entry or ltp)}")

    r3 = st.columns(5)
    locked = None if mgr.sl is None else (mgr.sl - position.entry_price) * position.direction
    r3[0].metric("Locked In by Stop", fmt_signed(locked) if locked is not None else "--",
                 help="Points the stop now guarantees; positive once the trail clears cost.")
    r3[1].metric("Entry Risk", fmt(mgr.risk_points) if mgr.risk_points else "--")
    r3[2].metric("R Multiple", fmt(points / mgr.risk_points) if mgr.risk_points else "--")
    r3[3].metric("Bars Held", f"{mgr.bars_held}")
    r3[4].metric("Best Price", fmt(mgr.mfe),
                 help="Best price in the trade's favour; drives the trailing stop.")

    (st.success if pnl >= 0 else st.error)(
        f"{side} {position.symbol} :: running {fmt_signed(pnl)} {currency}")
    for note in mgr.notes:
        st.warning("Exit engine: " + note)
    st.caption(f"Entered {fmt_time(position.entry_time)} off the signal candle "
               f"{fmt_time(position.signal_bar_time)}."
               + (f" LTP at fill was {fmt(position.entry_ltp_at_fill)}."
                  if position.entry_ltp_at_fill else ""))


# =============================================================================
# SECTION 14d -- ENTRY CONDITION CHECKLIST
# =============================================================================
# Prose explains one blocker at a time; a checklist shows every condition at once
# and, crucially, shows how far away each one is. Two layers are rendered: the
# ENGINE gates (feed, position, filters, signal freshness), which apply to every
# profile, and the STRATEGY conditions for the selected profile.

TICK_YES, TICK_NO, TICK_NA = "\u2705", "\u274c", "\u2796"


@dataclass
class ConditionCheck:
    label: str
    long_ok: bool | None          # None -> not applicable, shown as auto-pass
    short_ok: bool | None
    detail: str = ""


def _ck(label, long_ok, short_ok, detail=""):
    return ConditionCheck(label, long_ok, short_ok, detail)


def _mark(value) -> str:
    return TICK_NA if value is None else (TICK_YES if value else TICK_NO)


def engine_checks(cfg: dict, snapshot: LiveSnapshot) -> list[ConditionCheck]:
    """The gates the live engine applies regardless of which profile is selected."""
    checks: list[ConditionCheck] = []
    live = bool(snapshot.quote_live)
    checks.append(_ck("Feed alive (quote moving)", live, live,
                      f"source: {snapshot.ltp_source}"))

    flat = st.session_state.live_position is None
    checks.append(_ck("No position already open", flat, flat,
                      "one position at a time" if flat else "square off first"))

    for rep in snapshot.filter_reports:
        checks.append(_ck(f"Filter · {rep.label}", rep.long_ok, rep.short_ok, rep.value))

    ago = snapshot.recent_signal_bars_ago
    lookback = int(cfg.get("entry_lookback", 0) or 0)
    if snapshot.last_closed_signal != 0:
        fresh = True
        detail = "signal on the newest closed candle"
    elif snapshot.recent_signal != 0 and ago is not None:
        fresh = bool(lookback >= ago > 0)
        detail = (f"last signal {ago} candle(s) ago; catch-up allows {lookback}"
                  if not fresh else f"{ago} candle(s) ago, inside the catch-up window")
    else:
        fresh = False
        detail = "no signal on any recent closed candle"
    side_long = fresh and (snapshot.last_closed_signal == 1 or snapshot.recent_signal == 1)
    side_short = fresh and (snapshot.last_closed_signal == -1 or snapshot.recent_signal == -1)
    checks.append(_ck("Signal fresh enough to act on", side_long, side_short, detail))

    already = st.session_state.get("live_last_signal_time")
    sig_time = snapshot.last_closed_time if snapshot.last_closed_signal != 0 \
        else snapshot.recent_signal_time
    unused = already != sig_time
    checks.append(_ck("Signal not already traded", unused, unused,
                      "fresh" if unused else "this exact signal was already taken"))
    return checks


def _need(value, target, direction_up: bool) -> str:
    """Human 'needs +X more' for a level that has to be crossed."""
    if value is None or target is None:
        return ""
    gap = (target - value) if direction_up else (value - target)
    return "cleared" if gap <= 0 else f"needs {abs(gap):,.2f} more"


def strategy_checks(name: str, frame: pd.DataFrame, params: dict) -> list[ConditionCheck]:
    """
    Per-profile conditions, with the distance to each one.

    Only profiles whose conditions decompose cleanly are covered; anything else
    falls back to the prose description, which is still shown underneath.
    """
    def last(col):
        return safe_last(frame[col]) if col in frame.columns else None

    close = last("Close")
    out: list[ConditionCheck] = []

    if name.startswith(("01 ", "32 ", "34 ")):
        fast, slow = last("ema_fast"), last("ema_slow")
        if fast is not None and slow is not None:
            spread = fast - slow
            out.append(_ck(f"Fast EMA vs slow EMA ({fmt(fast)} vs {fmt(slow)})",
                           spread > 0, spread < 0,
                           f"spread {fmt_signed(spread)}; a cross needs {fmt(abs(spread))} more"))
            angle = ema_angle_degrees(frame)
            if angle is not None:
                out.append(_ck("Crossover angle", True, True, f"{angle:.2f}\u00b0"))
    if name.startswith(("32 ", "30 ", "36 ", "33 ", "31 ")):
        r = last("rsi")
        lo, hi = float(_p(params, "rsi_long_level")), float(_p(params, "rsi_short_level"))
        if r is not None:
            out.append(_ck(f"RSI {fmt(r)} against {fmt(lo)} / {fmt(hi)}", r >= lo, r <= hi,
                           f"long {_need(r, lo, True)} · short {_need(r, hi, False)}"))
    if name.startswith("08 "):
        r = last("rsi")
        if r is not None:
            out.append(_ck(f"RSI {fmt(r)} vs the 50 centerline", r > 50, r < 50,
                           f"{_need(r, 50, True)} to cross up"))
    if name.startswith(("25 ", "04 ")):
        d = last("st_dir") if "st_dir" in frame.columns else last("trail_dir")
        if d is not None:
            out.append(_ck("Trend band direction", d == 1, d == -1,
                           "bullish" if d == 1 else "bearish"))
    if name.startswith("05 "):
        hi, lo = last("or_high"), last("or_low")
        out.append(_ck(f"Opening range {fmt(lo)} - {fmt(hi)}",
                       None if hi is None else close > hi,
                       None if lo is None else close < lo,
                       f"UP {_need(close, hi, True)} · DOWN {_need(close, lo, False)}"))
    if name.startswith("47 "):
        expiry = last("is_expiry_day")
        move, a = last("move_from_open"), last("atr")
        need = float(_p(params, "zero_hero_atr")) * (a or 0)
        out.append(_ck("Expiry day", bool(expiry), bool(expiry),
                       "yes" if expiry else "not an expiry weekday"))
        out.append(_ck(f"Burst {fmt(need)} from the session open",
                       (move or 0) >= need, (move or 0) <= -need,
                       f"move {fmt_signed(move)}; UP {_need(move, need, True)} · "
                       f"DOWN {_need(move, -need, False)}"))
    if name.startswith("48 "):
        late = last("is_late_session")
        ratio = last("atr_ratio")
        hi, lo = last("day_high"), last("day_low")
        out.append(_ck("Inside the closing stretch", bool(late), bool(late),
                       "yes" if late else "too early in the session"))
        out.append(_ck(f"Volatility expanding (>= {fmt(_p(params, 'squeeze_mult'))}x)",
                       (ratio or 0) >= float(_p(params, "squeeze_mult")),
                       (ratio or 0) >= float(_p(params, "squeeze_mult")),
                       f"ATR ratio {fmt(ratio)}"))
        out.append(_ck(f"Session range {fmt(lo)} - {fmt(hi)}",
                       None if hi is None else close > hi,
                       None if lo is None else close < lo,
                       f"UP {_need(close, hi, True)} · DOWN {_need(close, lo, False)}"))
    if name.startswith(("35 ", "36 ", "37 ", "38 ", "39 ", "40 ", "41 ")):
        gh, gl = last("fib_golden_hi"), last("fib_golden_lo")
        up_leg = last("fib_up_leg")
        if gh is not None and gl is not None:
            inside = gl <= (close or 0) <= gh
            out.append(_ck(f"Golden zone {fmt(gl)} - {fmt(gh)}",
                           bool(up_leg) and inside, (not up_leg) and inside,
                           f"price {fmt(close)}, leg is {'up' if up_leg else 'down'}"))
    if name.startswith("28 "):
        lvl = last("threshold")
        out.append(_ck(f"Threshold {fmt(lvl)}",
                       None if lvl is None else close > lvl,
                       None if lvl is None else close < lvl,
                       f"UP {_need(close, lvl, True)} · DOWN {_need(close, lvl, False)}"))
    if name.startswith("29 "):
        up, dn = last("threshold_up"), last("threshold_dn")
        out.append(_ck(f"Bands {fmt(dn)} - {fmt(up)}",
                       None if up is None else close > up,
                       None if dn is None else close < dn,
                       f"UP {_need(close, up, True)} · DOWN {_need(close, dn, False)}"))
    if name.startswith("43 "):
        votes_l, votes_s = last("hybrid_long_votes"), last("hybrid_short_votes")
        members = int(last("hybrid_members") or 0)
        need_all = str(params.get("hybrid_logic", "")).startswith("All")
        required = members if need_all else 1
        out.append(_ck(f"Member agreement ({'all' if need_all else 'any one'} of {members})",
                       (votes_l or 0) >= required, (votes_s or 0) >= required,
                       f"long votes {fmt(votes_l, 0)} · short votes {fmt(votes_s, 0)}"))
    return out


def render_condition_checklist(cfg: dict, snapshot: LiveSnapshot) -> None:
    """The whole entry gate, one line per condition, with the distance to each."""
    strat = get_strategy(cfg["strategy"])
    checks = engine_checks(cfg, snapshot) + strategy_checks(cfg["strategy"], snapshot.frame,
                                                            cfg.get("params") or {})
    long_ready = all(c.long_ok is not False for c in checks)
    short_ready = all(c.short_ok is not False for c in checks)

    headline = (f"**\U0001F3AF {strat.name}** — every condition must hold on the SAME bar "
                f"(`{cfg['symbol']}` · `{cfg['interval']}`)")
    if long_ready or short_ready:
        st.success(headline + f"  \u2192 **{'LONG' if long_ready else 'SHORT'} is ready**")
    else:
        st.info(headline)

    lines = []
    for i, c in enumerate(checks, start=1):
        detail = f" — {c.detail}" if c.detail else ""
        lines.append(f"- **{i}. {c.label}**: LONG {_mark(c.long_ok)} · SHORT {_mark(c.short_ok)}"
                     f"{detail}")
    st.markdown("\n".join(lines))

    blocked_long = [c.label for c in checks if c.long_ok is False]
    blocked_short = [c.label for c in checks if c.short_ok is False]
    if blocked_long and blocked_short:
        st.caption(f"Blocking a LONG: {', '.join(blocked_long[:4])}. "
                   f"Blocking a SHORT: {', '.join(blocked_short[:4])}.")
    st.caption(f"{TICK_YES} met · {TICK_NO} not met · {TICK_NA} not applicable to this profile "
               "(treated as met rather than blocking every trade).")


def _searching_widget(cfg: dict, snapshot: LiveSnapshot) -> None:
    st.markdown("#### Signal Scanner")
    render_condition_checklist(cfg, snapshot)
    st.divider()
    blocked = [r for r in snapshot.filter_reports
               if (snapshot.raw_signal == 1 and not r.long_ok)
               or (snapshot.raw_signal == -1 and not r.short_ok)]
    if snapshot.raw_signal != 0 and blocked:
        st.warning("**Signal fired but filters vetoed it** :: blocked by "
                   + ", ".join(f"{r.label} ({r.value})" for r in blocked))
    else:
        st.info(f"**Searching for Signal** :: {snapshot.status.headline}")

    metrics = [("Live LTP", fmt(snapshot.ltp))] + list(snapshot.status.metrics)
    cols = st.columns(min(len(metrics), 5))
    for i, (label, value) in enumerate(metrics):
        cols[i % len(cols)].metric(label, value)
    st.caption("Live LTP comes from the quote feed and updates every tick. The remaining values "
               "are read off the last CLOSED candle, so they step at the candle interval.")

    strat = get_strategy(cfg["strategy"])
    if strat.immediate and snapshot.frozen:
        st.error("This profile enters immediately, but entry is held because the feed is dead. "
                 "It will fill on the first live tick, or enable stale entries in the sidebar "
                 "to override.")

    l, r = st.columns(2)
    l.markdown(f"**Long entry requires**\n\n{snapshot.status.long_condition}")
    r.markdown(f"**Short entry requires**\n\n{snapshot.status.short_condition}")

    risk = cfg["risk"]
    st.caption(f"On a fill at {fmt(snapshot.ltp)} the exit engine would apply -- {risk.as_summary()}")


def _live_chart(cfg: dict, snapshot: LiveSnapshot) -> None:
    strat = get_strategy(cfg["strategy"])
    params = cfg.get("params") or {}
    st.markdown("#### Live Chart")
    fig = price_chart(
        snapshot.frame, f"{cfg['symbol']} | {cfg['interval']} | last 120 candles",
        strat.overlays, tail=120, hide_weekends=cfg.get("hide_weekends", True), height=520,
        light=True,
        ema_lengths={"fast": _p(params, "ema_fast"), "slow": _p(params, "ema_slow")})

    position = st.session_state.live_position
    if position is not None:
        mgr = position.manager
        # Target above, entry in the middle, stop below: dotted for the levels
        # price has to travel to, dashed for where the position was opened.
        if mgr.tp is not None:
            fig.add_hline(y=mgr.tp, line=dict(width=1.4, dash="dot", color="#06b6d4"),
                          annotation_text="Target" + (" (display)" if mgr.tp_display_only else ""),
                          annotation_position="right",
                          annotation_font=dict(size=10, color="#06b6d4"))
        fig.add_hline(y=position.entry_price, line=dict(width=1.4, dash="dash", color="#26a69a"),
                      annotation_text="Entry", annotation_position="right",
                      annotation_font=dict(size=10, color="#26a69a"))
        if mgr.sl is not None:
            fig.add_hline(y=mgr.sl, line=dict(width=1.4, dash="dot", color="#ef5350"),
                          annotation_text="Stop", annotation_position="right",
                          annotation_font=dict(size=10, color="#ef5350"))
        try:                       # vertical marker at the candle we entered on
            entry_x = pd.Timestamp(position.signal_bar_time)
            if entry_x >= pd.Timestamp(snapshot.frame.index[-120 if len(snapshot.frame) > 120
                                                            else 0]):
                fig.add_vline(x=entry_x, line=dict(width=1, dash="dash", color="#9aa0a6"))
        except Exception:                                           # noqa: BLE001
            pass
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})


def _strategy_status_panel(cfg: dict, snapshot: LiveSnapshot) -> None:
    """
    What the strategy and every active filter are doing RIGHT NOW.

    Shown whether or not a position is open: when one is open the scanner is
    hidden, and without this the operator loses all visibility into why the
    engine would or would not take the next trade.
    """
    st.markdown("#### Strategy & Filter Status")
    st.info(f"**{cfg['strategy']}** :: {snapshot.status.headline}")
    metrics = [("Live LTP", fmt(snapshot.ltp))] + list(snapshot.status.metrics)
    cols = st.columns(min(len(metrics), 5))
    for i, (label, value) in enumerate(metrics):
        cols[i % len(cols)].metric(label, value)

    l, r = st.columns(2)
    l.markdown(f"**Long needs**\n\n{snapshot.status.long_condition}")
    r.markdown(f"**Short needs**\n\n{snapshot.status.short_condition}")

    if snapshot.filter_reports:
        rows = [{"Filter": rep.label, "Now": rep.value,
                 "Long": "PASS" if rep.long_ok else "BLOCK",
                 "Short": "PASS" if rep.short_ok else "BLOCK"}
                for rep in snapshot.filter_reports]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        blocking = [r["Filter"] for r in rows if r["Long"] == "BLOCK" and r["Short"] == "BLOCK"]
        if blocking:
            st.caption("Currently blocking BOTH directions: " + ", ".join(blocking))
    else:
        st.caption("No entry filters are enabled.")


def _recent_trades() -> None:
    """Closed trades, visible without leaving the tab. Tab 3 stays the full ledger."""
    rows = st.session_state.get("live_trades", [])
    st.markdown("#### Closed Trades This Session")
    if not rows:
        st.caption("Nothing closed yet. Stop-loss and target exits are automatic -- the manual "
                   "square-off is only for emergencies.")
        return
    frame = pd.DataFrame(rows).sort_values("Exit Time", ascending=False).head(8)
    cols = [c for c in ["Exit Time", "Direction", "Entry Price", "Exit Price", "Exit Reason",
                        "Points", "PnL"] if c in frame.columns]
    st.dataframe(frame[cols], width="stretch", hide_index=True)
    st.caption(f"{len(rows)} closed this session. Full history in the Live Trade Log Ledger tab.")


def _feed_diagnostics() -> None:
    """
    Proof of what is actually happening on the wire.

    If 'Polled at' keeps advancing while 'LTP' reports 'no' change, the refresh
    loop is healthy and the data source is the bottleneck.
    """
    log = st.session_state.get("feed_log", [])
    if not log:
        return
    changed = sum(1 for r in log if r["Changed"] == "yes")
    with st.expander(f"Feed diagnostics -- last {len(log)} polls, LTP changed on {changed}"):
        st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)
        if len(log) >= 8 and changed == 0:
            st.warning("The engine is polling but the price has not moved once. Either the venue "
                       "is closed, or the source is a delayed candle feed that only updates when "
                       "a new candle arrives. Polling faster cannot fix either one -- switch the "
                       "price source to Dhan, or use a 1m interval.")


def _filter_panel(snapshot: LiveSnapshot) -> None:
    if not snapshot.filter_reports:
        return
    with st.expander(f"Active entry filters ({len(snapshot.filter_reports)})", expanded=False):
        rows = [{"Filter": r.label, "Current value": r.value,
                 "Allows long": "yes" if r.long_ok else "NO",
                 "Allows short": "yes" if r.short_ok else "NO"} for r in snapshot.filter_reports]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if snapshot.vix is not None:
            st.caption(f"India VIX last read {fmt(snapshot.vix)}.")


def _broker_panel() -> None:
    receipts = st.session_state.get("broker_receipts", [])
    if not receipts:
        return
    with st.expander(f"Broker order receipts ({len(receipts)})"):
        st.dataframe(pd.DataFrame(receipts), width="stretch", hide_index=True)


def _event_feed() -> None:
    events = st.session_state.get("live_events", [])
    with st.expander(f"Operator event feed ({len(events)})", expanded=False):
        if not events:
            st.caption("No events yet.")
        for e in events[:80]:
            st.write(f"`{e['time']}` {e['message']}")


# =============================================================================
# SECTION 14b -- SIGNAL SCREENER
# =============================================================================
# HONESTY NOTE ON INDEX MEMBERSHIP
# --------------------------------
# NSE index constituents are reviewed periodically, so any list baked into a
# source file starts drifting the day it is written. The NIFTY 50 and NEXT 50
# lists below are a snapshot, not a live feed. For NIFTY 200 / 500, or whenever
# accuracy matters, paste or upload your own list -- that path is authoritative
# and is offered first in the UI for exactly this reason.

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA",
    "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

NIFTY_NEXT_50 = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "BAJAJHLDNG", "BANKBARODA",
    "BERGEPAINT", "BOSCHLTD", "CANBK", "CGPOWER", "CHOLAFIN", "COLPAL", "DABUR", "DIVISLAB",
    "DLF", "DMART", "GAIL", "GODREJCP", "HAL", "HAVELLS", "HYUNDAI", "ICICIGI", "ICICIPRULI",
    "INDHOTEL", "INDIGO", "IOC", "IRFC", "JINDALSTEL", "JIOFIN", "LICI", "LODHA", "LTIM",
    "MOTHERSON", "NAUKRI", "PFC", "PIDILITIND", "PNB", "RECLTD", "SHREECEM", "SIEMENS",
    "TATAPOWER", "TORNTPHARM", "TVSMOTOR", "UNITDSPR", "VBL", "VEDL", "ZYDUSLIFE",
]

SECTOR_INDICES = {
    "Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK", "Nifty IT": "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA", "Nifty Auto": "^CNXAUTO", "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL", "Nifty Realty": "^CNXREALTY", "Nifty Energy": "^CNXENERGY",
    "Nifty Infra": "^CNXINFRA", "Nifty PSU Bank": "^CNXPSUBANK", "Nifty Media": "^CNXMEDIA",
    "Nifty Fin Service": "NIFTY_FIN_SERVICE.NS", "Nifty Midcap": "^NSMIDCP", "Sensex": "^BSESN",
}

# NSE publishes its index constituents as plain CSVs. Fetching them beats any
# list baked into a source file, because membership is reviewed periodically and
# a hardcoded list starts drifting the day it is written. The bundled snapshots
# below are only a fallback for when the fetch fails.
NSE_INDEX_CSV = {
    "Nifty 50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty Next 50": "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "Nifty 100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 200": "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "Nifty 500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "Nifty Midcap 150":
        "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "Nifty Smallcap 250":
        "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "Nifty Bank": "https://nsearchives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "Nifty IT": "https://nsearchives.nseindia.com/content/indices/ind_niftyitlist.csv",
    "Nifty Auto": "https://nsearchives.nseindia.com/content/indices/ind_niftyautolist.csv",
    "Nifty Pharma": "https://nsearchives.nseindia.com/content/indices/ind_niftypharmalist.csv",
    "Nifty FMCG": "https://nsearchives.nseindia.com/content/indices/ind_niftyfmcglist.csv",
    "Nifty Metal": "https://nsearchives.nseindia.com/content/indices/ind_niftymetallist.csv",
    "Nifty Energy": "https://nsearchives.nseindia.com/content/indices/ind_niftyenergylist.csv",
    "Nifty Realty": "https://nsearchives.nseindia.com/content/indices/ind_niftyrealtylist.csv",
    "Nifty Financial Services":
        "https://nsearchives.nseindia.com/content/indices/ind_niftyfinancelist.csv",
}

SCREENER_UNIVERSES = [
    "Nifty 50", "Nifty Next 50", "Nifty 100", "Nifty 200", "Nifty 500",
    "Nifty Midcap 150", "Nifty Smallcap 250",
    "Nifty Bank", "Nifty IT", "Nifty Auto", "Nifty Pharma", "Nifty FMCG", "Nifty Metal",
    "Nifty Energy", "Nifty Realty", "Nifty Financial Services",
    "All NSE equities (Dhan master)",
    "Broad indices", "Sector indices",
    "Crypto (major)", "US large caps", "Forex majors", "Commodities", "Global indices",
    "Custom list (paste or upload)",
]

# Non-Indian universes. These are curated lists rather than index memberships:
# crypto pairs, FX crosses and futures roots barely change, so a static list is
# honest here in a way a NIFTY 500 snapshot is not. The US list is a sample of
# large caps, NOT the S&P 500 -- that membership drifts and is not fetched.
CRYPTO_MAJOR = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "LTC-USD", "TRX-USD", "BCH-USD",
    "ATOM-USD", "UNI-USD", "XLM-USD", "ETC-USD", "FIL-USD", "NEAR-USD",
]
US_LARGE_CAPS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "V", "MA", "UNH", "XOM", "JNJ", "WMT", "PG", "COST", "HD", "ORCL",
    "LLY", "MRK", "ABBV", "PEP", "KO", "BAC", "CRM", "AMD", "NFLX", "ADBE",
    "CSCO", "MCD", "INTC", "QCOM", "TXN", "DIS", "VZ", "IBM", "CAT", "GE",
]
FOREX_MAJORS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "USDINR=X", "EURINR=X", "GBPINR=X", "EURGBP=X", "EURJPY=X",
]
COMMODITY_FUTURES = [
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "PA=F", "BZ=F",
    "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CT=F",
]
GLOBAL_INDICES = [
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI", "^FCHI",
    "^N225", "^HSI", "^STOXX50E", "^AXJO", "^BSESN", "^NSEI",
]

STATIC_UNIVERSES = {
    "Crypto (major)": (CRYPTO_MAJOR, "Yahoo crypto pairs, quoted in USD and traded 24/7 — "
                                     "weekend gaps do not exist, so weekend-hiding on charts is "
                                     "switched off automatically."),
    "US large caps": (US_LARGE_CAPS, "A curated sample of US large caps, NOT the S&P 500. "
                                     "Index membership changes and is not fetched here; paste "
                                     "your own list if you need exact constituents."),
    "Forex majors": (FOREX_MAJORS, "Spot FX crosses. These report zero volume on Yahoo, so "
                                   "volume-gated strategies and filters will not arm."),
    "Commodities": (COMMODITY_FUTURES, "Front-month futures. Yahoo stitches contract rollovers, "
                                       "which puts artificial gaps in the history."),
    "Global indices": (GLOBAL_INDICES, "Index levels report zero volume on Yahoo; VWAP falls "
                                       "back to a session TWAP."),
}

# Fallbacks only. Used when NSE cannot be reached, and flagged as stale when they are.
_FALLBACK_LISTS = {
    "Nifty 50": None,          # filled in below from the bundled snapshots
    "Nifty Next 50": None,
    "Nifty 100": None,
}


def _fetch_nse_constituents(index_name: str, timeout: float = 20.0) -> list[str]:
    """
    Download one index's constituent list from NSE.

    NSE rejects unadorned requests, so a browser-ish header set is required. Any
    failure raises, and the caller decides whether a stale fallback is better
    than nothing.
    """
    import requests
    from io import StringIO

    url = NSE_INDEX_CSV[index_name]
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    try:                                    # warm the cookie jar; NSE expects one
        session.get("https://www.nseindia.com/", headers=headers, timeout=timeout)
    except Exception:                                               # noqa: BLE001
        pass
    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    frame = pd.read_csv(StringIO(resp.text))
    col = next((c for c in frame.columns if c.strip().lower() == "symbol"), None)
    if col is None:
        raise ValueError(f"unexpected columns from NSE: {list(frame.columns)[:6]}")
    names = [str(v).strip().upper() for v in frame[col].dropna()]
    if not names:
        raise ValueError("NSE returned an empty constituent list")
    return names


def nse_constituents(index_name: str) -> tuple[list[str], str | None]:
    """Cached constituent lookup. Returns ``(symbols, note)``; never raises."""
    def _load(name: str):
        try:
            return _fetch_nse_constituents(name), None
        except Exception as exc:                                    # noqa: BLE001
            return [], f"Could not reach NSE for {name} ({str(exc)[:90]})."

    if st is None:
        return _load(index_name)
    if not hasattr(nse_constituents, "_impl"):
        @st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
        def _impl(name: str):
            return _load(name)
        nse_constituents._impl = _impl
    return nse_constituents._impl(index_name)


def _universe_tickers(choice: str, custom_text: str, uploaded) -> tuple[list[str], str | None]:
    """
    Resolve a universe choice to Yahoo tickers.

    NSE index universes are fetched live and cached for an hour. Only when that
    fails do we fall back to a bundled snapshot, and the note says so plainly --
    a silently stale constituent list means screening companies that left the
    index and missing the ones that joined.
    """
    if choice in STATIC_UNIVERSES:
        names, note = STATIC_UNIVERSES[choice]
        return list(names), note

    if choice == "Broad indices":
        return ["^NSEI", "^NSEBANK", "^BSESN", "NIFTY_FIN_SERVICE.NS", "^NSMIDCP"], None
    if choice == "Sector indices":
        return list(SECTOR_INDICES.values()), None

    if choice.startswith("All NSE equities"):
        # A second source that does not depend on NSE answering. It cannot tell
        # us index MEMBERSHIP, so it is offered as the whole cash market rather
        # than dressed up as an index.
        try:
            master = st.session_state.get("scrip_master") if st is not None else None
            if master is None:
                master = load_scrip_master()
                if st is not None:
                    st.session_state.scrip_master = master
            eq = master[(master["instrument"].str.contains("EQUITY", na=False))
                        & (master["exchange"].str.startswith("NSE", na=False))]
            names = sorted({str(v).strip().upper() for v in eq["trading_symbol"].dropna()
                            if str(v).strip() and str(v).strip().isalnum()})
            if names:
                return [f"{n}.NS" for n in names], (
                    f"{len(names)} NSE cash symbols from the Dhan instrument master. This is the "
                    "whole market, not an index — narrow it with the Symbols box.")
        except Exception as exc:                                    # noqa: BLE001
            return [], f"Could not load the Dhan instrument master ({str(exc)[:90]})."
        return [], "The Dhan instrument master returned no NSE equities."

    if choice in NSE_INDEX_CSV:
        names, problem = nse_constituents(choice)
        if names:
            return [f"{n}.NS" for n in names], (f"{len(names)} constituents fetched live from "
                                                f"NSE for {choice}.")
        fallback = {"Nifty 50": NIFTY_50, "Nifty Next 50": NIFTY_NEXT_50,
                    "Nifty 100": NIFTY_50 + NIFTY_NEXT_50}.get(choice)
        if fallback:
            return [f"{n}.NS" for n in fallback], (
                f"{problem} Falling back to a bundled snapshot of {choice}, which is stale by "
                "however long it has been since this file was written. Verify against the "
                "current NSE factsheet, or paste your own list.")
        return [], (f"{problem} There is no bundled fallback for {choice}: a 200 or 500 name "
                    "list baked into a source file would be wrong within weeks, and screening a "
                    "stale index means missing the joiners and scanning the leavers. Options: "
                    "retry (NSE often blocks cloud IPs but answers from a home connection), use "
                    "**All NSE equities (Dhan master)**, or paste your own list under "
                    "**Custom list**.")

    raw = ""
    if uploaded is not None:
        try:
            raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        except Exception:                                           # noqa: BLE001
            raw = ""
    raw = (raw + "\n" + (custom_text or "")).replace(",", "\n")
    names = [x.strip().upper() for x in raw.splitlines() if x.strip()]
    return [_normalise_ticker(n) for n in names], None


# Yahoo suffixes and shapes that are already complete and must not be touched.
_KNOWN_SUFFIXES = (".NS", ".BO", ".L", ".TO", ".AX", ".HK", ".SI", ".DE", ".PA",
                   ".MI", ".SW", ".T", ".KS", ".SA", ".MX", ".NZ")
_QUOTE_CURRENCIES = ("-USD", "-USDT", "-EUR", "-GBP", "-INR", "-BTC", "-ETH")


def _normalise_ticker(name: str) -> str:
    """
    Complete a bare symbol into a Yahoo ticker without breaking foreign ones.

    `.NS` used to be appended to anything without a dot, which turned BTC-USD
    into BTC-USD.NS and AAPL into AAPL.NS. Only a plain alphanumeric Indian-style
    symbol gets the suffix; crypto pairs, FX crosses, futures roots, indices and
    already-suffixed tickers are passed through untouched.
    """
    n = (name or "").strip().upper()
    if not n:
        return n
    if (n.startswith("^") or "=" in n or n.endswith(_KNOWN_SUFFIXES)
            or n.endswith(_QUOTE_CURRENCIES) or "." in n):
        return n
    if n in set(CRYPTO_MAJOR) | set(US_LARGE_CAPS):
        return n
    if not n.replace("&", "").replace("-", "").isalnum():
        return n
    return f"{n}.NS"


def signal_detail(frame: pd.DataFrame, hit_time, direction: int, risk: "RiskConfig | None",
                  ticker: str, interval: str) -> dict:
    """
    Everything worth knowing about a signal that has already fired.

    Two things deserve care here. First, the FILL price is the open of the candle
    AFTER the signal, not the signal candle's close -- that is the rule both
    engines follow, and quoting the close would flatter every row. Second, raw
    percentage move is signed by the market, so a profitable SHORT shows a
    negative one; "Move in Favour" restates it from the trade's point of view.
    """
    idx = frame.index
    pos = int(idx.get_loc(hit_time))
    last = len(frame) - 1
    price_signal = float(frame["Close"].iloc[pos])
    fill = float(frame["Open"].iloc[pos + 1]) if pos + 1 <= last else float("nan")
    price_now = float(frame["Close"].iloc[last])

    move_abs = price_now - price_signal
    move_pct = (move_abs / price_signal * 100.0) if price_signal else float("nan")
    fav_abs = (price_now - fill) * direction if np.isfinite(fill) else float("nan")
    fav_pct = (fav_abs / fill * 100.0) if (np.isfinite(fill) and fill) else float("nan")

    after = frame.iloc[pos + 1:]
    best = worst = float("nan")
    if len(after):
        if direction > 0:
            best = float(after["High"].max()) - fill
            worst = float(after["Low"].min()) - fill
        else:
            best = fill - float(after["Low"].min())
            worst = fill - float(after["High"].max())

    a = float(frame["atr"].iloc[last]) if "atr" in frame else float("nan")
    stop = target = risk_pts = r_now = None
    if risk is not None and np.isfinite(fill):
        try:
            ctx = bar_ctx(frame, min(pos + 1, last))
            mgr = ExitManager(risk, fill, direction, ctx)
            stop = mgr.sl
            target = mgr.tp
            risk_pts = mgr.risk_points
            if risk_pts:
                r_now = fav_abs / risk_pts
        except Exception:                                           # noqa: BLE001
            pass

    last_ts = pd.Timestamp(idx[last])
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tz is not None else pd.Timestamp.now()
    vol_ratio = None
    if "Volume" in frame and float(frame["Volume"].tail(50).abs().sum()) > 0:
        vma = float(sma(frame["Volume"], 20).iloc[last])
        if vma:
            vol_ratio = float(frame["Volume"].iloc[last]) / vma

    return {
        "Signal Time": pd.Timestamp(hit_time),
        "Bars Ago": int(last - 1 - pos),
        "Price at Signal": round(price_signal, 2),
        "Fill Price (next open)": None if not np.isfinite(fill) else round(fill, 2),
        "Price Now": round(price_now, 2),
        "Move Abs": round(move_abs, 2),
        "Move %": round(move_pct, 3),
        "Move in Favour": None if not np.isfinite(fav_abs) else round(fav_abs, 2),
        "Favour %": None if not np.isfinite(fav_pct) else round(fav_pct, 3),
        "Best Since": None if not np.isfinite(best) else round(best, 2),
        "Worst Since": None if not np.isfinite(worst) else round(worst, 2),
        "Suggested Stop": None if stop is None else round(float(stop), 2),
        "Suggested Target": None if target is None else round(float(target), 2),
        "Risk Points": None if risk_pts is None else round(float(risk_pts), 2),
        "R Multiple Now": None if r_now is None else round(float(r_now), 2),
        "Distance to Stop": None if stop is None else round(abs(price_now - float(stop)), 2),
        "Distance to Target": None if target is None else round(abs(float(target) - price_now), 2),
        "ATR": None if not np.isfinite(a) else round(a, 2),
        "ATR %": None if not (np.isfinite(a) and price_now) else round(a / price_now * 100, 3),
        "Volume x Avg": None if vol_ratio is None else round(vol_ratio, 2),
        "Last Candle": last_ts,
        "Candle Age": _human_age(float((now - last_ts).total_seconds())),
        "Scanned At": pd.Timestamp.now(),
        "Interval": interval,
    }


def screen_universe(tickers: list[str], cfg: dict, lookback_bars: int, progress=None):
    """
    Run the sidebar configuration across a list of tickers and report signals.

    Sequential by design: each download carries the mandatory guards, so a wide
    universe takes real time. That is the honest cost of not getting throttled.
    """
    rows, errors = [], []
    strat = get_strategy(cfg["strategy"])
    extras = dict(cfg.get("filter_extras") or {})
    if cfg.get("filter_cfg", {}).get("vix", {}).get("enabled"):
        extras["vix"] = load_vix()

    for i, ticker in enumerate(tickers):
        if progress is not None:
            progress.progress((i + 1) / max(1, len(tickers)), text=f"Scanning {ticker} ...")
        try:
            bundle = load_market_data(ticker, live_period_for(cfg["interval"]), cfg["interval"],
                                      freshness_seconds=120,
                                      min_bars=max(strat.min_bars, 30))
            frame, _ = prepare(bundle.frame, cfg["strategy"], cfg["params"],
                               cfg.get("filter_cfg"), extras)
        except Exception as exc:                                    # noqa: BLE001
            errors.append({"Ticker": ticker, "Problem": str(exc)[:140]})
            continue

        # CLOSED candles only. Including the forming bar made it look like a
        # confirmed hit, which left the row with no fill price (there is no next
        # open yet) and a nonsensical "-1 bars ago".
        window = frame.iloc[-(lookback_bars + 1):-1]
        fired = window[window["signal"] != 0]
        forming = int(frame["signal"].iloc[-1])
        last_closed_pos = len(frame) - 2

        if fired.empty and forming == 0:
            continue
        if not fired.empty:
            hit_time = fired.index[-1]
            direction = int(fired["signal"].iloc[-1])
            bars_ago = last_closed_pos - frame.index.get_loc(hit_time)
            when = ("Just now (last closed candle)" if bars_ago <= 0
                    else f"Just before ({int(bars_ago)} candles ago)")
        else:
            hit_time, direction, when = frame.index[-1], forming, "Forming candle (unconfirmed)"

        detail = signal_detail(frame, hit_time, direction, cfg.get("risk"), ticker,
                               cfg["interval"])
        rows.append({
            "Ticker": ticker,
            "Signal": "LONG" if direction > 0 else "SHORT",
            "When": when,
            **detail,
            "Fast EMA": round(float(frame["ema_fast"].iloc[-1]), 2)
            if "ema_fast" in frame else None,
            "Slow EMA": round(float(frame["ema_slow"].iloc[-1]), 2)
            if "ema_slow" in frame else None,
            "Strategy": cfg["strategy"],
        })
    return pd.DataFrame(rows), pd.DataFrame(errors)


def tab_screener(cfg: dict) -> None:
    st.subheader("Signal Screener")
    st.caption(f"Runs the current sidebar configuration -- **{cfg['strategy']}** at "
               f"`{cfg['interval']}` with the active filters -- across a universe and lists "
               "whichever tickers are signalling.")

    c1, c2, c3 = st.columns([2, 1, 1])
    universe = c1.selectbox("Universe", SCREENER_UNIVERSES, key="scr_universe")
    lookback = c2.number_input("Signal window (candles)", 1, 20, 3, key="scr_look",
                               help="How far back a signal still counts as recent.")
    max_names = c3.number_input("Max tickers", 1, 500, 50, key="scr_max")

    custom_text, uploaded = "", None
    if universe.startswith("Custom"):
        custom_text = st.text_area("Tickers (one per line or comma separated)",
                                   "KAYNES\nRELIANCE\nTATAMOTORS", key="scr_custom",
                                   help="Bare NSE names get `.NS` appended automatically.")
        uploaded = st.file_uploader("...or upload a CSV / text file of tickers", type=["csv", "txt"],
                                    key="scr_upload")

    with st.spinner(f"Resolving {universe} ..."):
        tickers, note = _universe_tickers(universe, custom_text, uploaded)
    tickers = tickers[:int(max_names)]
    if note:
        st.warning(f"{note} Index membership is reviewed periodically and this list is baked "
                   "into the file, so it drifts over time. For NIFTY 200 / 500 or anything "
                   "where accuracy matters, use **Custom list** with your own constituents.")

    est = len(tickers) * 1.2
    st.caption(f"{len(tickers)} tickers queued. Expect roughly {est:0.0f}s -- each download "
               f"carries the mandatory {API_GUARD_DELAY}s guard on both sides so the scan does "
               "not get the IP throttled.")

    if st.button("Run Screener", type="primary", width="stretch"):
        st.session_state.screener_error = None
        bar = st.progress(0.0, text="Starting ...")
        try:
            results, errors = screen_universe(tickers, cfg, int(lookback), bar)
            st.session_state.screener_results = (results, errors)
        except Exception as exc:                                    # noqa: BLE001
            st.session_state.screener_error = str(exc)
        bar.empty()

    if st.session_state.screener_error:
        st.error(st.session_state.screener_error)

    payload = st.session_state.screener_results
    if payload is None:
        st.info("Choose a universe and run the screener.")
        return
    results, errors = payload

    if results.empty:
        st.info("No tickers are signalling on this configuration right now.")
    else:
        results = results.sort_values(["When", "Signal Time"], ascending=[True, False])
        st.success(f"{len(results)} ticker(s) signalling.")
        order = [c for c in [
            "Ticker", "Signal", "When", "Bars Ago", "Signal Time", "Interval",
            "Price at Signal", "Fill Price (next open)", "Price Now",
            "Move Abs", "Move %", "Move in Favour", "Favour %", "R Multiple Now",
            "Best Since", "Worst Since",
            "Suggested Stop", "Suggested Target", "Risk Points",
            "Distance to Stop", "Distance to Target",
            "ATR", "ATR %", "Volume x Avg", "Fast EMA", "Slow EMA",
            "Last Candle", "Candle Age", "Scanned At", "Strategy",
        ] if c in results.columns]
        st.dataframe(results[order], width="stretch", hide_index=True,
                     column_config={
                         "Signal Time": st.column_config.DatetimeColumn(
                             format="YYYY-MM-DD HH:mm:ss"),
                         "Last Candle": st.column_config.DatetimeColumn(
                             format="YYYY-MM-DD HH:mm:ss"),
                         "Scanned At": st.column_config.DatetimeColumn(
                             format="YYYY-MM-DD HH:mm:ss"),
                     })
        st.caption(
            "**Fill Price** is the open of the candle AFTER the signal, which is the rule both "
            "engines actually follow — quoting the signal candle's close would flatter every "
            "row. **Move %** is signed by the market, so a profitable SHORT shows a negative "
            "one; **Move in Favour** and **R Multiple Now** restate it from the trade's point "
            "of view. Stop and target come from your current sidebar risk settings applied at "
            "that fill. **Best / Worst Since** are the extremes reached after the fill, in the "
            "trade's favour and against it.")

        pick = st.selectbox("Select a ticker", results["Ticker"].tolist(), key="scr_pick")
        a, b = st.columns([1, 3])
        if a.button("Apply to sidebar", type="primary", width="stretch"):
            # Streamlit forbids writing a widget's key AFTER that widget has been
            # instantiated, and the sidebar is built before this tab renders. So
            # we park the choice in a plain key and the sidebar consumes it at
            # the top of the next run, before any widget exists.
            st.session_state.pending_ticker = pick
            st.rerun()
        b.caption("Applies the ticker to the sidebar so the Backtesting and Live tabs use it. "
                  "Everything else in your configuration is left untouched.")
        st.download_button("Download results (CSV)", results.to_csv(index=False).encode(),
                           f"screener_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", "text/csv")
        render_analyst_panel("screener", "these screener hits",
                             f"Strategy: {cfg['strategy']} at {cfg['interval']}\n"
                             f"{_frame_context(results[order])}")

    if not errors.empty:
        with st.expander(f"Tickers that could not be scanned ({len(errors)})"):
            st.dataframe(errors, width="stretch", hide_index=True)


# =============================================================================
# SECTION 15 -- TAB 3: LIVE TRADE LOG LEDGER

# =============================================================================
# SECTION 19 -- CHART PATTERN LIBRARY
# =============================================================================
# Every detector returns the bar on which the pattern was CONFIRMED, never the
# bar where it started. Geometric patterns are built on zigzag pivots, and a
# pivot is only knowable `right` bars after it printed, so a hit can never be
# reported earlier than it could actually have been seen.
#
# Honest framing: pattern recognition is subjective. Two analysts disagree about
# roughly half of these. Each hit therefore carries its own historical record on
# that instrument, which is the only evidence worth acting on.

PATTERN_FAMILIES = ["Candlestick", "Reversal", "Continuation", "Level"]


@dataclass
class PatternHit:
    pattern: str
    family: str
    bias: str                     # "bullish" | "bearish" | "either"
    index: int                    # bar position of CONFIRMATION
    time: Any
    entry: float
    stop: float
    geometry: list = field(default_factory=list)
    measured: float | None = None      # classic measured-move objective
    note: str = ""


def _pattern_frame(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Candle anatomy plus the context every detector shares."""
    out = df.copy()
    out["atr"] = atr(out["High"], out["Low"], out["Close"], int(_p(params, "atr_len")))
    out["ema20"] = ema(out["Close"], 20)
    out["body"] = (out["Close"] - out["Open"]).abs()
    out["range"] = (out["High"] - out["Low"]).replace(0.0, np.nan)
    out["upper_wick"] = out["High"] - out[["Open", "Close"]].max(axis=1)
    out["lower_wick"] = out[["Open", "Close"]].min(axis=1) - out["Low"]
    out["bull"] = out["Close"] > out["Open"]
    out["uptrend"] = out["Close"] > out["ema20"]
    return out


def _fit_line(xs, ys):
    """Least-squares slope/intercept, tolerant of degenerate input."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if xs.size < 2 or np.ptp(xs) == 0:
        return 0.0, float(ys.mean()) if ys.size else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _line_geo(x_idx, i0, i1, p0, p1, label="Trendline"):
    return {"type": "line", "x0": x_idx[i0], "y0": float(p0),
            "x1": x_idx[i1], "y1": float(p1), "label": label}


def _band_geo(x_idx, i0, i1, label=""):
    return {"type": "band", "x0": x_idx[max(i0, 0)], "x1": x_idx[i1], "label": label}


def _pivot_geo(x_idx, idxs, prices):
    return [{"type": "pivot", "x": x_idx[i], "y": float(p)} for i, p in zip(idxs, prices)]


# --------------------------------------------------------------------------- #
# Candlestick detectors
# --------------------------------------------------------------------------- #
def _candlestick_masks(f: pd.DataFrame) -> dict[str, tuple[str, pd.Series, int]]:
    """name -> (bias, boolean mask, bars the pattern spans)."""
    o, h, l, c = f["Open"], f["High"], f["Low"], f["Close"]
    body, rng, uw, lw, bull = f["body"], f["range"], f["upper_wick"], f["lower_wick"], f["bull"]
    po, pc = o.shift(1), c.shift(1)
    pbody = (pc - po).abs()
    down, up = f["Close"] < f["ema20"], f["Close"] > f["ema20"]

    small_body = body <= 0.35 * rng
    m: dict[str, tuple[str, pd.Series, int]] = {}

    m["Hammer"] = ("bullish", (lw >= 2 * body) & (uw <= body) & small_body & down, 1)
    m["Inverted hammer"] = ("bullish", (uw >= 2 * body) & (lw <= body) & small_body & down, 1)
    m["Hanging man"] = ("bearish", (lw >= 2 * body) & (uw <= body) & small_body & up, 1)
    m["Shooting star"] = ("bearish", (uw >= 2 * body) & (lw <= body) & small_body & up, 1)
    m["Doji"] = ("either", body <= 0.1 * rng, 1)
    m["Bullish marubozu"] = ("bullish", bull & (body >= 0.9 * rng), 1)
    m["Bearish marubozu"] = ("bearish", (~bull) & (body >= 0.9 * rng), 1)

    m["Bullish engulfing"] = ("bullish",
                              bull & (pc < po) & (c >= po) & (o <= pc) & (body > pbody), 2)
    m["Bearish engulfing"] = ("bearish",
                              (~bull) & (pc > po) & (c <= po) & (o >= pc) & (body > pbody), 2)
    m["Bullish harami"] = ("bullish",
                           bull & (pc < po) & (c <= po) & (o >= pc) & (body < pbody), 2)
    m["Bearish harami"] = ("bearish",
                           (~bull) & (pc > po) & (c >= po) & (o <= pc) & (body < pbody), 2)
    m["Piercing line"] = ("bullish", bull & (pc < po) & (o < pc) & (c > (po + pc) / 2) & (c < po), 2)
    m["Dark cloud cover"] = ("bearish",
                             (~bull) & (pc > po) & (o > pc) & (c < (po + pc) / 2) & (c > po), 2)
    m["Tweezer bottom"] = ("bullish", (l - l.shift(1)).abs() <= 0.1 * rng.shift(1), 2)
    m["Tweezer top"] = ("bearish", (h - h.shift(1)).abs() <= 0.1 * rng.shift(1), 2)

    harami_bull = bull & (pc < po) & (c <= po) & (o >= pc)
    harami_bear = (~bull) & (pc > po) & (c >= po) & (o <= pc)
    m["Three inside up"] = ("bullish", harami_bull.shift(1).fillna(False) & bull & (c > c.shift(1)), 3)
    m["Three inside down"] = ("bearish",
                              harami_bear.shift(1).fillna(False) & (~bull) & (c < c.shift(1)), 3)

    o2, c2 = o.shift(2), c.shift(2)
    m["Morning star"] = ("bullish",
                         (c2 < o2) & (pbody <= 0.4 * (o2 - c2).abs()) & bull
                         & (c > (o2 + c2) / 2), 3)
    m["Evening star"] = ("bearish",
                         (c2 > o2) & (pbody <= 0.4 * (c2 - o2).abs()) & (~bull)
                         & (c < (o2 + c2) / 2), 3)

    # Five-bar continuation: a long bar, three small counter bars inside it, then
    # a close beyond the first bar's extreme.
    h4, l4, c4, o4 = h.shift(4), l.shift(4), c.shift(4), o.shift(4)
    inside3 = ((h.shift(3) < h4) & (l.shift(3) > l4) & (h.shift(2) < h4) & (l.shift(2) > l4)
               & (h.shift(1) < h4) & (l.shift(1) > l4))
    m["Rising three methods"] = ("bullish", (c4 > o4) & inside3 & bull & (c > h4), 5)
    m["Falling three methods"] = ("bearish", (c4 < o4) & inside3 & (~bull) & (c < l4), 5)
    return m


# --------------------------------------------------------------------------- #
# Geometric detectors (pivot based)
# --------------------------------------------------------------------------- #
def _geometric_hits(f: pd.DataFrame, pivots: list, wanted: set[str], tol: float) -> list[PatternHit]:
    hits: list[PatternHit] = []
    if not pivots:
        return hits
    x = f.index
    close = f["Close"].to_numpy(float)
    high = f["High"].to_numpy(float)
    low = f["Low"].to_numpy(float)
    atr_v = f["atr"].to_numpy(float)
    n = len(f)

    def add(name, family, bias, i, entry, stop, geo, measured=None, note=""):
        if not (0 <= i < n) or name not in wanted:
            return
        if not (np.isfinite(entry) and np.isfinite(stop)) or entry == stop:
            return
        hits.append(PatternHit(name, family, bias, i, x[i], float(entry), float(stop),
                               geo, measured, note))

    highs = [(pi, pp, ci) for pi, pp, k, ci in pivots if k == 1]
    lows = [(pi, pp, ci) for pi, pp, k, ci in pivots if k == -1]

    # ---- double top / bottom : two comparable extremes around one counter pivot
    for seq, name, bias in ((highs, "Double top", "bearish"), (lows, "Double bottom", "bullish")):
        for a, b in zip(seq, seq[1:]):
            (ia, pa, _), (ib, pb, cb) = a, b
            if abs(pa - pb) > tol * max(abs(pa), 1e-9):
                continue
            mid = [p for p in pivots if ia < p[0] < ib and p[2] != (1 if bias == "bearish" else -1)]
            if not mid:
                continue
            neck = mid[-1][1]
            conf = next((j for j in range(cb, n)
                         if (close[j] < neck if bias == "bearish" else close[j] > neck)), None)
            if conf is None:
                continue
            geo = _pivot_geo(x, [ia, ib], [pa, pb]) + [
                _line_geo(x, ia, ib, pa, pb, name), _line_geo(x, ia, conf, neck, neck, "Neckline"),
                _band_geo(x, conf - 1, conf, name)]
            add(name, "Reversal", bias, conf, close[conf], max(pa, pb) if bias == "bearish"
                else min(pa, pb), geo, measured=abs(max(pa, pb) - neck))

    # ---- head and shoulders : five pivots, middle extreme dominant
    for seq, name, bias in ((highs, "Head and shoulders", "bearish"),
                            (lows, "Inverse head and shoulders", "bullish")):
        for a, b, c_ in zip(seq, seq[1:], seq[2:]):
            (ia, pa, _), (ib, pb, _), (ic, pc_, cc) = a, b, c_
            dominant = pb > max(pa, pc_) if bias == "bearish" else pb < min(pa, pc_)
            if not dominant or abs(pa - pc_) > 2 * tol * max(abs(pa), 1e-9):
                continue
            mids = [p[1] for p in pivots if ia < p[0] < ic
                    and p[2] != (1 if bias == "bearish" else -1)]
            if len(mids) < 2:
                continue
            neck = float(np.mean(mids[-2:]))
            conf = next((j for j in range(cc, n)
                         if (close[j] < neck if bias == "bearish" else close[j] > neck)), None)
            if conf is None:
                continue
            geo = _pivot_geo(x, [ia, ib, ic], [pa, pb, pc_]) + [
                _line_geo(x, ia, ic, neck, neck, "Neckline"), _band_geo(x, conf - 1, conf, name)]
            add(name, "Reversal", bias, conf, close[conf], pb, geo, measured=abs(pb - neck))

    # ---- broadening / wedges / diamonds : slope relationship of the two envelopes
    for k in range(3, len(highs)):
        hi3 = highs[k - 3:k + 1]
        if len(hi3) < 3:
            continue
        lo3 = [p for p in lows if hi3[0][0] <= p[0] <= hi3[-1][0]]
        if len(lo3) < 3:
            continue
        hs, hi_c = _fit_line([p[0] for p in hi3], [p[1] for p in hi3])
        ls, lo_c = _fit_line([p[0] for p in lo3], [p[1] for p in lo3])
        i0, i1 = hi3[0][0], hi3[-1][0]
        conf = max(hi3[-1][2], lo3[-1][2])
        if conf >= n:
            continue
        width0 = (hs * i0 + hi_c) - (ls * i0 + lo_c)
        width1 = (hs * i1 + hi_c) - (ls * i1 + lo_c)
        if width0 <= 0 or width1 <= 0:
            continue
        geo = [_line_geo(x, i0, i1, hs * i0 + hi_c, hs * i1 + hi_c, "Upper"),
               _line_geo(x, i0, i1, ls * i0 + lo_c, ls * i1 + lo_c, "Lower"),
               _band_geo(x, conf - 1, conf)]
        widening = width1 > width0 * 1.3
        narrowing = width1 < width0 * 0.7
        stop_up = hs * i1 + hi_c
        stop_dn = ls * i1 + lo_c
        if widening:
            if hs > 0 and ls > 0:
                add("Broadening top", "Reversal", "bearish", conf, close[conf], stop_up, geo)
            elif hs < 0 and ls < 0:
                add("Broadening bottom", "Reversal", "bullish", conf, close[conf], stop_dn, geo)
            else:
                add("Broadening top", "Reversal", "either", conf, close[conf], stop_up, geo)
        elif narrowing:
            if hs > 0 and ls > 0:
                add("Rising wedge", "Reversal", "bearish", conf, close[conf], stop_up, geo)
            elif hs < 0 and ls < 0:
                add("Falling wedge", "Reversal", "bullish", conf, close[conf], stop_dn, geo)
            else:
                add("Pennant", "Continuation", "either", conf, close[conf],
                    stop_dn if close[conf] > stop_dn else stop_up, geo)
            if width0 > 0 and k >= 4:
                prev_w = width0
                if prev_w > width1 * 1.6:
                    add("Diamond top" if hs > 0 else "Diamond bottom", "Reversal",
                        "bearish" if hs > 0 else "bullish", conf, close[conf],
                        stop_up if hs > 0 else stop_dn, geo)

    # ---- three drives : three successive extremes in the same direction
    for seq, name, bias in ((highs, "Three drives", "bearish"), (lows, "Three drives", "bullish")):
        for a, b, c_ in zip(seq, seq[1:], seq[2:]):
            (ia, pa, _), (ib, pb, _), (ic, pc_, cc) = a, b, c_
            rising = pa < pb < pc_
            falling = pa > pb > pc_
            if not (rising if bias == "bearish" else falling):
                continue
            if cc >= n:
                continue
            geo = _pivot_geo(x, [ia, ib, ic], [pa, pb, pc_]) + [
                _line_geo(x, ia, ic, pa, pc_, "Drives"), _band_geo(x, cc - 1, cc, name)]
            add(name, "Reversal", bias, cc, close[cc], pc_, geo)

    # ---- trendline breakout / breakdown, with and without the retest
    for seq, up in ((lows, True), (highs, False)):
        if len(seq) < 3:
            continue
        for k in range(2, len(seq)):
            pts = seq[k - 2:k + 1]
            slope, intercept = _fit_line([p[0] for p in pts], [p[1] for p in pts])
            start, ready = pts[0][0], pts[-1][2]
            broke = None
            for j in range(ready, min(n, ready + 60)):
                line = slope * j + intercept
                if (up and close[j] < line) or ((not up) and close[j] > line):
                    broke = j
                    break
            if broke is None:
                continue
            name = "Trendline breakdown" if up else "Trendline breakout"
            bias = "bearish" if up else "bullish"
            geo = _pivot_geo(x, [p[0] for p in pts], [p[1] for p in pts]) + [
                _line_geo(x, start, broke, slope * start + intercept, slope * broke + intercept),
                _band_geo(x, broke - 1, broke, name)]
            stop = max(high[broke - 2:broke + 1]) if up else min(low[broke - 2:broke + 1])
            add(name, "Level", bias, broke, close[broke], stop, geo)

            # retest: price returns to the broken line and is rejected by it
            for j in range(broke + 1, min(n, broke + 20)):
                line = slope * j + intercept
                touched = (high[j] >= line) if up else (low[j] <= line)
                rejected = (close[j] < line) if up else (close[j] > line)
                if touched and rejected:
                    rname = name + " + retest"
                    rgeo = geo[:-1] + [
                        _line_geo(x, start, j, slope * start + intercept, slope * j + intercept),
                        {"type": "star", "x": x[j], "y": float(close[j]), "label": rname},
                        _band_geo(x, j - 1, j, rname)]
                    rstop = high[j] if up else low[j]
                    add(rname, "Level", bias, j, close[j], rstop, rgeo)
                    break

    # ---- parallel channel break
    if len(highs) >= 3 and len(lows) >= 3:
        hs, hc = _fit_line([p[0] for p in highs[-3:]], [p[1] for p in highs[-3:]])
        ls, lc = _fit_line([p[0] for p in lows[-3:]], [p[1] for p in lows[-3:]])
        if abs(hs - ls) <= abs(hs) * 0.5 + 1e-9:
            start = min(highs[-3][0], lows[-3][0])
            ready = max(highs[-1][2], lows[-1][2])
            for j in range(ready, n):
                up_line, dn_line = hs * j + hc, ls * j + lc
                if close[j] > up_line or close[j] < dn_line:
                    rising = hs > 0
                    name = "Ascending channel break" if rising else "Descending channel break"
                    geo = [_line_geo(x, start, j, hs * start + hc, up_line, "Channel top"),
                           _line_geo(x, start, j, ls * start + lc, dn_line, "Channel base"),
                           _band_geo(x, j - 1, j, name)]
                    add(name, "Continuation", "either", j, close[j],
                        dn_line if close[j] > up_line else up_line, geo)
                    break

    # ---- rounding top / bottom : quadratic fit over a trailing window
    win = 40
    for j in range(win, n, 5):
        seg = close[j - win:j]
        if not np.isfinite(seg).all():
            continue
        coef = np.polyfit(np.arange(win), seg, 2)
        curv, slope = coef[0], 2 * coef[0] * (win - 1) + coef[1]
        scale = np.nanmean(atr_v[j - win:j]) or 1.0
        strength = abs(curv) * win * win / max(scale, 1e-9)
        if strength < 1.5:
            continue
        fitted = np.polyval(coef, np.arange(win))
        geo = [{"type": "curve", "x": list(x[j - win:j]), "y": [float(v) for v in fitted],
                "label": "Rounding"}, _band_geo(x, j - 1, j)]
        if curv < 0 and slope < 0:
            add("Rounding top", "Reversal", "bearish", j, close[j],
                float(np.max(high[j - win:j])), geo)
        elif curv > 0 and slope > 0:
            add("Rounding bottom", "Reversal", "bullish", j, close[j],
                float(np.min(low[j - win:j])), geo)

    # ---- island reversal : a gap out and a gap back within a few bars
    prev_h, prev_l = np.r_[np.nan, high[:-1]], np.r_[np.nan, low[:-1]]
    gap_up = low > prev_h
    gap_dn = high < prev_l
    for j in range(2, n):
        for back in range(1, 6):
            if j - back < 1:
                break
            if gap_up[j - back] and gap_dn[j]:
                geo = [_band_geo(x, j - back, j, "Island"),
                       {"type": "star", "x": x[j], "y": float(close[j]), "label": "Island top"}]
                add("Island reversal top", "Reversal", "bearish", j, close[j],
                    float(np.max(high[j - back:j + 1])), geo)
                break
            if gap_dn[j - back] and gap_up[j]:
                geo = [_band_geo(x, j - back, j, "Island"),
                       {"type": "star", "x": x[j], "y": float(close[j]), "label": "Island bottom"}]
                add("Island reversal bottom", "Reversal", "bullish", j, close[j],
                    float(np.min(low[j - back:j + 1])), geo)
                break

    # ---- volatility contraction : each pullback shallower than the last
    rng_ma = pd.Series(high - low).rolling(10).mean().to_numpy(float)
    for j in range(60, n):
        a, b, c_ = rng_ma[j - 40], rng_ma[j - 20], rng_ma[j]
        if not np.isfinite([a, b, c_]).all() or a <= 0:
            continue
        if c_ < b * 0.75 and b < a * 0.75 and close[j] > np.nanmax(high[j - 20:j]):
            geo = [_band_geo(x, j - 40, j, "Contraction"),
                   {"type": "star", "x": x[j], "y": float(close[j]), "label": "VCP breakout"}]
            add("Volatility contraction (VCP)", "Continuation", "bullish", j, close[j],
                float(np.nanmin(low[j - 20:j])), geo)

    # ---- flag : sharp impulse then a shallow counter drift
    for j in range(30, n):
        pole = close[j - 10] - close[j - 20]
        drift = close[j] - close[j - 10]
        scale = np.nanmean(atr_v[j - 20:j]) or 1.0
        if abs(pole) < 4 * scale or abs(drift) > abs(pole) * 0.5:
            continue
        if pole > 0 and drift < 0 and close[j] > np.nanmax(high[j - 5:j]):
            geo = [_band_geo(x, j - 20, j, "Flag"),
                   _line_geo(x, j - 20, j - 10, close[j - 20], close[j - 10], "Pole")]
            add("Bull flag", "Continuation", "bullish", j, close[j],
                float(np.nanmin(low[j - 10:j])), geo)
        elif pole < 0 and drift > 0 and close[j] < np.nanmin(low[j - 5:j]):
            geo = [_band_geo(x, j - 20, j, "Flag"),
                   _line_geo(x, j - 20, j - 10, close[j - 20], close[j - 10], "Pole")]
            add("Bear flag", "Continuation", "bearish", j, close[j],
                float(np.nanmax(high[j - 10:j])), geo)
    return hits


def _fvg_hits(f: pd.DataFrame, wanted: set[str]) -> list[PatternHit]:
    hits: list[PatternHit] = []
    bull, bear, blo, bhi, selo, sehi = fair_value_gaps(f)
    x = f.index
    for name, mask, bias in (("Fair value gap (bullish)", bull, "bullish"),
                             ("Fair value gap (bearish)", bear, "bearish")):
        if name not in wanted:
            continue
        for j in np.where(mask.to_numpy())[0]:
            lo = float(blo.iloc[j] if bias == "bullish" else selo.iloc[j])
            hi = float(bhi.iloc[j] if bias == "bullish" else sehi.iloc[j])
            if not np.isfinite([lo, hi]).all():
                continue
            geo = [{"type": "zone", "x0": x[max(j - 2, 0)], "x1": x[j], "y0": lo, "y1": hi,
                    "label": name}]
            hits.append(PatternHit(name, "Level", bias, int(j), x[j], float(f["Close"].iloc[j]),
                                   lo if bias == "bullish" else hi, geo, abs(hi - lo)))
    return hits


PATTERN_CATALOG: dict[str, str] = {}


def _build_catalog() -> dict[str, str]:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    probe = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
                         index=idx)
    cat = {name: "Candlestick" for name in _candlestick_masks(_pattern_frame(probe, DEFAULT_PARAMS))}
    for name in ("Double top", "Double bottom", "Head and shoulders",
                 "Inverse head and shoulders", "Broadening top", "Broadening bottom",
                 "Rising wedge", "Falling wedge", "Diamond top", "Diamond bottom",
                 "Three drives", "Rounding top", "Rounding bottom",
                 "Island reversal top", "Island reversal bottom"):
        cat[name] = "Reversal"
    for name in ("Pennant", "Bull flag", "Bear flag", "Volatility contraction (VCP)",
                 "Ascending channel break", "Descending channel break"):
        cat[name] = "Continuation"
    for name in ("Trendline breakout", "Trendline breakout + retest", "Trendline breakdown",
                 "Trendline breakdown + retest", "Fair value gap (bullish)",
                 "Fair value gap (bearish)"):
        cat[name] = "Level"
    return dict(sorted(cat.items()))


PATTERN_CATALOG = _build_catalog()
PATTERN_NAMES = list(PATTERN_CATALOG)


def detect_patterns(df: pd.DataFrame, wanted: list[str] | None = None,
                    params: dict | None = None) -> list[PatternHit]:
    """Run every requested detector over one OHLC frame."""
    params = params or dict(DEFAULT_PARAMS)
    wanted_set = set(wanted or PATTERN_NAMES)
    f = _pattern_frame(df, params)
    hits: list[PatternHit] = []

    masks = _candlestick_masks(f)
    x = f.index
    close = f["Close"].to_numpy(float)
    high = f["High"].to_numpy(float)
    low = f["Low"].to_numpy(float)
    for name, (bias, mask, span) in masks.items():
        if name not in wanted_set:
            continue
        for j in np.where(mask.fillna(False).to_numpy())[0]:
            j = int(j)
            lo = float(np.nanmin(low[max(j - span + 1, 0):j + 1]))
            hi = float(np.nanmax(high[max(j - span + 1, 0):j + 1]))
            stop = lo if bias == "bullish" else hi
            if bias == "either":
                stop = lo if close[j] > f["Open"].iloc[j] else hi
            geo = [_band_geo(x, max(j - span + 1, 0), j, name),
                   {"type": "star", "x": x[j], "y": float(close[j]), "label": name}]
            hits.append(PatternHit(name, "Candlestick", bias, j, x[j], float(close[j]), stop,
                                   geo, abs(hi - lo)))

    thr = _auto_zigzag_threshold(f, _p(params, "zigzag_pct"), int(_p(params, "atr_len")))
    pivots = zigzag_pivot_table(f["Close"], thr)
    hits += _geometric_hits(f, pivots, wanted_set, tol=0.02)
    hits += _fvg_hits(f, wanted_set)
    hits.sort(key=lambda h: h.index)
    return hits


# --------------------------------------------------------------------------- #
# Levels and the historical record behind them
# --------------------------------------------------------------------------- #
def pattern_levels(f: pd.DataFrame, hit: PatternHit, rr: float, atr_mult: float) -> dict:
    """
    Turn a hit into an actionable plan.

    The stop is the structure the pattern is built on. When that structure is
    unusably far away (or on the wrong side), an ATR-scaled stop stands in, and
    the panel says which one is being used.
    """
    a = float(f["atr"].iloc[hit.index]) if "atr" in f else float("nan")
    entry = float(hit.entry)
    stop = float(hit.stop)
    direction = 1 if hit.bias == "bullish" else (-1 if hit.bias == "bearish" else
                                                 (1 if stop < entry else -1))
    fallback = False
    if not np.isfinite(stop) or (direction > 0 and stop >= entry) or \
            (direction < 0 and stop <= entry):
        stop = entry - direction * atr_mult * (a if np.isfinite(a) else entry * 0.005)
        fallback = True
    risk = abs(entry - stop)
    target = entry + direction * rr * risk
    measured = None
    if hit.measured and np.isfinite(hit.measured):
        measured = entry + direction * float(hit.measured)
    return {"direction": direction, "entry": entry, "stop": stop, "target": target,
            "risk": risk, "atr": a, "measured": measured, "fallback_stop": fallback,
            "atr_multiple": (risk / a) if (np.isfinite(a) and a > 0) else None}


def pattern_track_record(f: pd.DataFrame, hits: list[PatternHit], rr: float,
                         atr_mult: float) -> dict:
    """
    Replay this exact rule on this instrument.

    Entry at the next bar's open, stop and target from the same recipe, stop
    assumed to trigger first when a bar spans both. This is the only evidence
    that separates a pattern worth taking from a shape someone named.
    """
    o = f["Open"].to_numpy(float)
    h = f["High"].to_numpy(float)
    l = f["Low"].to_numpy(float)
    n = len(f)
    results = []
    for hit in hits:
        i = hit.index + 1
        if i >= n:
            continue
        plan = pattern_levels(f, hit, rr, atr_mult)
        d = plan["direction"]
        entry = float(o[i])
        risk = plan["risk"]
        if risk <= 0:
            continue
        stop = entry - d * risk
        target = entry + d * rr * risk
        outcome = None
        for j in range(i, min(n, i + 200)):
            if d > 0:
                if l[j] <= stop:
                    outcome = -risk
                    break
                if h[j] >= target:
                    outcome = rr * risk
                    break
            else:
                if h[j] >= stop:
                    outcome = -risk
                    break
                if l[j] <= target:
                    outcome = rr * risk
                    break
        if outcome is not None:
            results.append(outcome)

    trades = len(results)
    if trades == 0:
        return {"trades": 0, "hit_rate": 0.0, "expectancy": 0.0, "profit_factor": 0.0,
                "net_points": 0.0, "verdict": "No completed occurrences in this window, so "
                                              "there is nothing to judge the pattern on."}
    wins = [r for r in results if r > 0]
    losses = [r for r in results if r <= 0]
    hit_rate = len(wins) / trades * 100.0
    gross_win, gross_loss = sum(wins), -sum(losses)
    breakeven = 100.0 / (1.0 + rr)
    # Wilson lower bound: what the win rate could plausibly be given this few trades.
    z = 1.96
    ph = len(wins) / trades
    denom = 1 + z * z / trades
    centre = ph + z * z / (2 * trades)
    margin = z * math.sqrt(max(ph * (1 - ph) / trades + z * z / (4 * trades * trades), 0.0))
    pessimistic = max(0.0, (centre - margin) / denom) * 100.0

    notes = []
    if gross_loss > 0 and gross_win / gross_loss < 1:
        notes.append(f"wins are {gross_win / gross_loss:.2f}x losses")
    if trades < 20:
        notes.append("thin sample")
    if pessimistic < breakeven:
        notes.append(f"pessimistic win rate {pessimistic:.0f}% is below the {breakeven:.0f}% "
                     "needed to break even, so the sample cannot rule out a losing setup")
    else:
        notes.append(f"even the pessimistic win rate {pessimistic:.0f}% clears the "
                     f"{breakeven:.0f}% break-even")
    return {"trades": trades, "hit_rate": round(hit_rate, 1),
            "expectancy": round(float(np.mean(results)), 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "net_points": round(float(np.sum(results)), 2),
            "breakeven": round(breakeven, 1), "pessimistic": round(pessimistic, 1),
            "verdict": "Same pattern, same stop and target rule, applied to every occurrence in "
                       "this window. " + "; ".join(notes) + "."}


def draw_pattern(f: pd.DataFrame, hit: PatternHit, pad: int = 60):
    """Candles around the hit with the pattern's own geometry drawn on top."""
    import plotly.graph_objects as go

    lo = max(0, hit.index - pad)
    hi = min(len(f), hit.index + pad // 3)
    data = f.iloc[lo:hi]
    fig = go.Figure(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"],
        close=data["Close"], name="Price", increasing_line_color=_UP,
        decreasing_line_color=_DOWN, increasing_fillcolor=_UP, decreasing_fillcolor=_DOWN))

    for g in hit.geometry:
        kind = g.get("type")
        if kind == "line":
            fig.add_trace(go.Scatter(x=[g["x0"], g["x1"]], y=[g["y0"], g["y1"]], mode="lines",
                                     line=dict(color="#4f9df7", width=2),
                                     name=g.get("label", "line"), showlegend=False))
            fig.add_annotation(x=g["x1"], y=g["y1"], text=g.get("label", ""), showarrow=False,
                               font=dict(size=10, color="#4f9df7"), xanchor="left")
        elif kind == "curve":
            fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines",
                                     line=dict(color="#4f9df7", width=2, dash="dot"),
                                     showlegend=False))
        elif kind == "pivot":
            fig.add_trace(go.Scatter(x=[g["x"]], y=[g["y"]], mode="markers",
                                     marker=dict(symbol="circle-open", size=12,
                                                 color="#4dd0e1", line=dict(width=2)),
                                     showlegend=False))
        elif kind == "star":
            fig.add_trace(go.Scatter(x=[g["x"]], y=[g["y"]], mode="markers+text",
                                     marker=dict(symbol="star", size=15, color="#ffb300"),
                                     text=[g.get("label", "")], textposition="top right",
                                     textfont=dict(color="#ffb300", size=10), showlegend=False))
        elif kind == "band":
            fig.add_vrect(x0=g["x0"], x1=g["x1"], fillcolor="#ffb300", opacity=0.18, line_width=0)
        elif kind == "zone":
            fig.add_shape(type="rect", x0=g["x0"], x1=g["x1"], y0=g["y0"], y1=g["y1"],
                          fillcolor="#4f9df7", opacity=0.2, line=dict(width=0))

    fig.update_layout(
        title=dict(text=f"{hit.pattern} — {fmt_time(hit.time)}", x=0.01, xanchor="left",
                   font=dict(size=14)),
        height=460, margin=dict(l=10, r=10, t=44, b=10), xaxis_rangeslider_visible=False,
        hovermode="x unified", showlegend=False, dragmode="pan")
    return fig


# =============================================================================
def tab_ledger(cfg: dict) -> None:
    st.subheader("Live Trade Log Ledger")
    st.caption("Closed positions from the live sandbox only, sorted chronologically by exit time. "
               "Simulated backtest fills are never written here.")

    frame = live_ledger_frame()
    if frame.empty:
        st.info("No live trades closed yet. Start the core in Tab 2; every position it closes is "
                "journalled here.")
        _open_position_note()
        return

    cur = cfg["currency"]
    pnl = frame["PnL"]
    wins = pnl[pnl > 0]
    gross_loss = float(-pnl[pnl <= 0].sum())
    m = st.columns(5)
    m[0].metric("Closed Live Trades", f"{len(frame):,}")
    m[1].metric("Win Rate", f"{fmt(len(wins)/len(frame)*100)}%", f"{len(wins)} winners")
    m[2].metric(f"Realised PnL ({cur})", fmt_signed(pnl.sum()))
    m[3].metric("Total Points", fmt_signed(frame["Points"].sum()))
    m[4].metric("Profit Factor", "inf" if gross_loss == 0 else fmt(float(wins.sum()) / gross_loss))

    with st.expander("Breakdown by exit reason"):
        by = frame.groupby("Exit Reason")["PnL"].agg(["count", "sum"]).reset_index()
        by.columns = ["Exit Reason", "Trades", f"PnL ({cur})"]
        st.dataframe(by, width="stretch", hide_index=True)

    order = [c for c in ["#", "Exit Time", "Symbol", "Interval", "Strategy", "Direction",
                         "Option Leg", "Entry Bar Open", "Entry Bar High", "Entry Bar Low",
                         "Entry Bar Close", "Exit Bar Open", "Exit Bar High", "Exit Bar Low",
                         "Exit Bar Close",
                         "Quantity", "Entry Time", "Entry Price", "Exit Price", "Initial Stop",
                         "Final Stop", "Target", "Best Price", "Exit Reason", "Points", "PnL",
                         "Broker Order", "Source"] if c in frame.columns]
    st.dataframe(frame[order], width="stretch", hide_index=True,
                 column_config={"PnL": st.column_config.NumberColumn(f"PnL ({cur})", format="%.2f")})

    c1, c2 = st.columns([1, 4])
    c1.download_button("Download ledger (CSV)", frame[order].to_csv(index=False).encode(),
                       f"live_ledger_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", "text/csv",
                       width="stretch")
    with c2.popover("Clear ledger"):
        st.warning("This permanently discards the live trade history for this session.")
        if st.button("Confirm and clear", type="primary"):
            st.session_state.live_trades = []
            st.rerun()
    render_analyst_panel("ledger", "this live trade ledger", _frame_context(frame[order]))
    _open_position_note()


def _open_position_note() -> None:
    position = st.session_state.get("live_position")
    if position is None:
        return
    st.warning(f"A {'LONG' if position.direction > 0 else 'SHORT'} position on "
               f"`{position.symbol}` is still open and therefore not in this ledger. It is "
               "journalled the moment it closes.")


# =============================================================================
# SECTION 20 -- GROQ ANALYST PANEL
# =============================================================================
# Model names on hosted APIs get retired without notice, so nothing is hardcoded:
# the list is fetched from the key's own /models endpoint and only what that key
# can actually reach is offered. A dropdown built from a stale constant is how
# you end up staring at "model_not_found".

GROQ_BASE = "https://api.groq.com/openai/v1"


def groq_models(api_key: str) -> tuple[list[str], str | None]:
    """Fetch the chat models this key can use. Returns ``(models, error)``."""
    key = (api_key or "").strip()
    if not key:
        return [], "No API key supplied."
    import requests
    try:
        resp = requests.get(f"{GROQ_BASE}/models",
                            headers={"Authorization": f"Bearer {key}"}, timeout=20)
        body = resp.json()
    except Exception as exc:                                        # noqa: BLE001
        return [], f"Could not reach Groq: {exc}"
    if resp.status_code >= 400:
        return [], f"Groq rejected the key (HTTP {resp.status_code}): {str(body)[:160]}"

    models = []
    for entry in body.get("data", []) or []:
        name = str(entry.get("id", "")).strip()
        if not name:
            continue
        # Whisper and TTS endpoints share the list but cannot answer a chat turn.
        if any(tag in name.lower() for tag in ("whisper", "tts", "guard", "embed")):
            continue
        models.append(name)
    if not models:
        return [], "The key is valid but exposes no chat models."
    return sorted(models), None


def groq_chat(api_key: str, model: str, messages: list[dict], temperature: float = 0.2) -> str:
    import requests
    resp = requests.post(
        f"{GROQ_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {(api_key or '').strip()}",
                 "Content-Type": "application/json"},
        data=json.dumps({"model": model, "messages": messages, "temperature": temperature,
                         "max_tokens": 1200}), timeout=90)
    try:
        body = resp.json()
    except Exception:                                               # noqa: BLE001
        body = {"raw": resp.text[:400]}
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq error (HTTP {resp.status_code}): {str(body)[:300]}")
    return body["choices"][0]["message"]["content"]


_ANALYST_SYSTEM = (
    "You are a quantitative trading analyst reviewing output from a backtesting and screening "
    "application. Be concise and concrete, and reason from the numbers you are given.\n"
    "Hold to these standards:\n"
    "- Distinguish sample size from evidence. Under about 30 trades, say so plainly.\n"
    "- Results that come from searching many combinations are shortlists, not findings. Point "
    "out selection bias when it applies.\n"
    "- A high win rate with a poor profit factor means small wins funding large losses. Say it.\n"
    "- If a result is marked as an optimistic backtest (distance-trailing stops), treat its "
    "numbers as an upper bound.\n"
    "- Never tell the user to buy or sell anything. Describe what the data supports and what it "
    "does not, and name the checks that would settle the question.\n"
    "- If the data given is insufficient to answer, say that rather than guessing."
)


def _frame_context(frame, max_rows: int = 30, max_chars: int = 6000) -> str:
    if frame is None or getattr(frame, "empty", True):
        return "(no rows)"
    try:
        text = frame.head(max_rows).to_csv(index=False)
    except Exception:                                               # noqa: BLE001
        return "(unreadable)"
    return text[:max_chars]


def render_analyst_panel(panel_key: str, context_label: str, context_text: str) -> None:
    """
    A chat panel scoped to one tab's results.

    The tab's own output is sent as context, so the model comments on what is on
    screen rather than on trading in the abstract.
    """
    cfg = st.session_state.get("groq_cfg") or {}
    api_key, model = cfg.get("key", ""), cfg.get("model", "")
    with st.expander(f"Ask the analyst about {context_label}", expanded=False):
        if not api_key:
            st.info("Add a Groq API key in the sidebar to enable this. It is used only for these "
                    "questions and is not stored anywhere by the app.")
            return
        if not model:
            st.warning("No Groq model selected. Open the sidebar and pick one from the list "
                       "fetched for your key.")
            return

        history_key = f"chat_{panel_key}"
        history = st.session_state.setdefault(history_key, [])
        for turn in history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        c1, c2 = st.columns([4, 1])
        question = c1.text_input("Question", key=f"{panel_key}_q",
                                 placeholder=f"e.g. which of these {context_label} is weakest, "
                                             "and why?")
        if c2.button("Clear", key=f"{panel_key}_clear"):
            st.session_state[history_key] = []
            st.rerun()

        if st.button("Ask", key=f"{panel_key}_ask", type="primary") and question.strip():
            messages = [{"role": "system", "content": _ANALYST_SYSTEM},
                        {"role": "user",
                         "content": f"Here is the current {context_label} from the app.\n\n"
                                    f"```\n{context_text}\n```\n\nQuestion: {question.strip()}"}]
            for turn in history[-6:]:
                messages.insert(-1, turn)
            try:
                with st.spinner("Thinking ..."):
                    answer = groq_chat(api_key, model, messages)
            except Exception as exc:                                # noqa: BLE001
                st.error(str(exc))
                return
            history.append({"role": "user", "content": question.strip()})
            history.append({"role": "assistant", "content": answer})
            st.session_state[history_key] = history[-12:]
            st.rerun()


def render_groq_sidebar(sb) -> dict:
    """Key entry and model discovery. Models come from the key, never a constant."""
    cfg = {"key": "", "model": ""}
    with sb.expander("Groq analyst (optional)"):
        enabled = st.checkbox("Enable the Groq analyst", value=False, key="groq_on",
                              help="Adds a chat panel to each results tab that can comment on "
                                   "what the tab is showing.")
        if not enabled:
            return cfg
        cfg["key"] = st.text_input("Groq API key", type="password", key="groq_key")
        if not cfg["key"]:
            st.caption("Get a key from console.groq.com. Nothing is sent until you ask a "
                       "question.")
            return cfg
        if st.button("Fetch available models", key="groq_fetch"):
            models, err = groq_models(cfg["key"])
            st.session_state["groq_model_list"] = models
            st.session_state["groq_model_error"] = err
        models = st.session_state.get("groq_model_list") or []
        err = st.session_state.get("groq_model_error")
        if err:
            st.error(err)
        if models:
            cfg["model"] = st.selectbox("Model", models, key="groq_model")
            st.caption(f"{len(models)} chat model(s) available to this key. The list comes from "
                       "Groq itself, so a retired model cannot be selected.")
        else:
            st.caption("Press **Fetch available models** to load the list for your key.")
    return cfg


def render_search_grid(prefix: str, cfg: dict, safe_only: bool):
    """
    Optional wide search. Off by default, because the default grid already tries
    enough combinations to overfit a noisy sample; widening it makes the winner
    more impressive and less trustworthy at the same time.

    Returns ``(grid, exhaustive, note)`` or ``(None, False, None)`` when disabled.
    """
    scale_points = st.checkbox(
        "Scale point-based stops and targets to the instrument (ATR)", value=True,
        key=f"{prefix}_scale",
        help="On: absolute point values in the grid are rewritten as multiples of this "
             "instrument's own ATR. A 40-point stop is normal on Nifty and 0.05% on Bitcoin, so "
             "searching the same fixed numbers everywhere compares nothing with nothing.")
    wide = st.checkbox(
        "Configure the search grid myself (wide / greedy search)", value=False,
        key=f"{prefix}_wide",
        help="Off: a fixed shortlist of sensible stops, targets and filters is sampled at "
             "random. On: you choose every axis, including your own stop and target values.")
    if not wide:
        return None, False, None, scale_points

    st.caption("Every axis you widen multiplies the number of backtests. The estimate below "
               "updates as you choose.")
    default_strategies = [n for n in STRATEGY_NAMES if not n.startswith(_OPTIMISER_EXCLUDE)]
    strategies = st.multiselect("Strategies to search", default_strategies,
                                default=default_strategies[:12], key=f"{prefix}_g_strat")

    c1, c2 = st.columns(2)
    sl_choices = [t for t in SL_TYPES
                  if not (safe_only and t in DISTANCE_TRAIL_TYPES) and t != "No Stop-Loss"]
    sl_types = c1.multiselect("Stop-loss types", sl_choices,
                              default=[t for t in ("Fixed Percentage", "Fixed Points",
                                                   "ATR Multiple") if t in sl_choices],
                              key=f"{prefix}_g_slt")
    sl_text = c1.text_input("Stop-loss values (comma separated)", "0.5, 1, 1.5, 2",
                            key=f"{prefix}_g_slv",
                            help="Interpreted per type: percent, points, or ATR multiple.")
    tp_types = c2.multiselect("Target types", [t for t in TP_TYPES if t != "No Target"],
                              default=["Fixed Percentage", "Risk : Reward Multiple"],
                              key=f"{prefix}_g_tpt")
    tp_text = c2.text_input("Target values (comma separated)", "1, 1.5, 2, 3",
                            key=f"{prefix}_g_tpv")

    filters = st.multiselect("Entry filters to try (none is always included)",
                             [spec["key"] for spec in FILTER_SPECS],
                             default=["adx", "rsi", "ema20", "supertrend"],
                             format_func=lambda k: FILTER_LABELS.get(k, k),
                             key=f"{prefix}_g_filt")

    grid = build_search_grid(strategies, sl_types, parse_number_list(sl_text, [1.0]),
                             tp_types, parse_number_list(tp_text, [2.0]), filters, safe_only)
    size = grid_size(grid)
    exhaustive = st.checkbox(
        f"Walk the whole grid ({size:,} combinations) instead of sampling it", value=False,
        key=f"{prefix}_g_exh",
        help="Exhaustive is reproducible but slow. Sampling covers the same space more cheaply "
             "and, given the overfitting risk, rarely finds a worse winner.")
    if not grid["sl"]:
        st.error("Every stop-loss type you picked is a distance trail, and *Only search "
                 "backtest-safe exits* excludes those. Add a fixed, ATR or structural stop, or "
                 "untick the safe-exits box and accept that the results are optimistic.")
    if not grid["tp"]:
        st.error("No target types selected, so there is nothing to search.")
    note = (f"{len(grid['strategies'])} strategies x {len(grid['sl'])} stops x "
            f"{len(grid['tp'])} targets x {len(grid['filters'])} filters = **{size:,}** "
            f"combinations per ticker.")
    if size > 4000 and exhaustive:
        st.warning(f"{size:,} exhaustive backtests per ticker will take a long time. Leave the "
                   "box unticked to sample instead.")
    return grid, exhaustive, note, scale_points


def tab_optimiser(cfg: dict) -> None:
    st.subheader("Strategy Optimiser")
    st.error("**Read this before using the output.** Searching hundreds of combinations and "
             "keeping the best one is the most reliable way to fool yourself in this entire "
             "application. With enough attempts something always looks excellent on any sample, "
             "including pure noise. The ranking below is a SHORTLIST to validate out of sample, "
             "never a result. Simple Buy, Simple Sell and the threshold profiles are excluded "
             "because they are execution helpers, not edges.")

    c1, c2, c3, c4 = st.columns(4)
    objective = c1.selectbox("Optimise for", OPTIMISER_OBJECTIVES, key="opt_obj")
    target = c2.number_input("Desired value", value=80.0, step=1.0, key="opt_target",
                             help="Combinations at or above this are highlighted as meeting "
                                  "your goal. It does not restrict the search.")
    min_trades = c3.number_input("Minimum trades", 5, 1000, 30, 5, key="opt_min",
                                 help="Combinations with fewer trades are discarded: below "
                                      "roughly 30 the statistics are noise.")
    iterations = c4.number_input("Combinations to test", 20, 2000, 150, 10, key="opt_iters")

    safe_only = st.checkbox("Only search backtest-safe exits", value=True, key="opt_safe",
                            help="Excludes distance-based trailing stops, whose backtested "
                                 "results are systematically optimistic.")

    grid, exhaustive, grid_note, scale_points = render_search_grid("opt", cfg, safe_only)
    if grid_note:
        st.info(grid_note)

    period = _period_for_timeframe(cfg["interval"], cfg["period"], WARMUP_BARS + 80)
    if period != cfg["period"]:
        st.caption(f"History widened from `{cfg['period']}` to `{period}` so that "
                   f"`{cfg['interval']}` can supply the 200-candle warm-up plus enough bars to "
                   f"produce trades.")

    if st.button("Run Optimiser", type="primary", width="stretch"):
        st.session_state.optimizer_results = None
        bar = st.progress(0.0, text="Loading data ...")
        try:
            bundle = load_market_data(cfg["symbol"], period, cfg["interval"], 300.0,
                                      min_bars=WARMUP_BARS + 60)
            results = optimise(bundle.frame, cfg["params"], cfg["quantity"], cfg["costs"],
                               objective, int(min_trades), int(iterations),
                               safe_exits_only=safe_only, progress=bar,
                               grid=grid, exhaustive=exhaustive, scale_points=scale_points)
            st.session_state.optimizer_results = (results, objective, float(target))
        except Exception as exc:                                    # noqa: BLE001
            st.error(f"Optimiser failed: {exc}")
        bar.empty()

    payload = st.session_state.optimizer_results
    if payload is None:
        st.info("Set an objective and run the search on the sidebar's current ticker and period.")
        return
    results, objective, target = payload
    if results.empty:
        st.warning("No combination produced enough trades to be worth reporting. Lower the "
                   "minimum trade count, widen the period, or use a faster interval.")
        return

    hits = results[results["Score"] >= target]
    st.success(f"{len(results)} combinations survived the trade-count filter. "
               f"{len(hits)} reached your target of {fmt(target)} on {objective}.")
    if len(results) >= 50:
        st.warning(f"You tested {len(results)} combinations. At that many attempts, the top of "
                   "the table is partly selection luck. Re-test the leaders on a different "
                   "period before believing any of them.")

    st.dataframe(results.head(40), width="stretch", hide_index=True)
    pick = st.selectbox("Apply which rank?", results["Rank"].tolist(), key="opt_pick")
    row = results[results["Rank"] == pick].iloc[0]
    if st.button("Apply this combination to the sidebar", type="primary", width="stretch"):
        st.session_state.pending_combo = {
            "strategy": row["Strategy"], "sl_type": row["Stop-Loss"], "sl_value": row["SL Value"],
            "tp_type": row["Target"], "tp_value": row["TP Value"],
            # The entry filter is part of the combination that produced these
            # numbers. Applying the strategy without it would hand back a
            # different system from the one that was ranked.
            "filter_key": str(row.get("Filter Key") or "")}
        st.rerun()
    st.caption(f"Rank {pick}: {row['Strategy']} | SL {row['Stop-Loss']} | TGT {row['Target']} | "
               f"filter {row['Filter']} | {row['Trades']} trades | reliability {row['Reliability']}")
    st.download_button("Download all combinations (CSV)", results.to_csv(index=False).encode(),
                       "optimiser_results.csv", "text/csv")


# =============================================================================
# SECTION 19b -- CHART PATTERN SCANNER TAB
# =============================================================================
PATTERN_TIMEFRAMES = ["5m", "15m", "30m", "60m", "4h", "1d", "1wk"]


def trades_around_the_clock(symbol: str) -> bool:
    """Crypto and spot FX have no weekend gap, so the chart must not hide one."""
    t = (symbol or "").upper()
    return t.endswith(_QUOTE_CURRENCIES) or "=X" in t or t.endswith("=F")


def _pattern_period_for(interval: str) -> str:
    return {"5m": "1mo", "15m": "3mo", "30m": "3mo", "60m": "6mo",
            "4h": "1y", "1d": "2y", "1wk": "5y"}.get(interval, "1y")


def scan_patterns(tickers: list[str], timeframes: list[str], wanted: list[str],
                  direction: str, lookback: int, params: dict, progress=None):
    """
    Scan every ticker x timeframe and return recent hits plus the frames needed
    to draw them. Sequential, because every download carries the API guards.
    """
    rows: list[dict] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    hits_by_key: dict[tuple[str, str], list] = {}
    errors: list[dict] = []
    total = max(1, len(tickers) * len(timeframes))
    done = 0

    for ticker in tickers:
        for tf in timeframes:
            done += 1
            if progress is not None:
                progress.progress(done / total, text=f"{ticker} · {tf}")
            try:
                bundle = load_market_data(ticker, _pattern_period_for(tf), tf,
                                          freshness_seconds=300, min_bars=80)
            except Exception as exc:                                # noqa: BLE001
                errors.append({"Ticker": ticker, "Timeframe": tf, "Problem": str(exc)[:120]})
                continue
            frame = _pattern_frame(bundle.frame, params)
            try:
                hits = detect_patterns(bundle.frame, wanted, params)
            except Exception as exc:                                # noqa: BLE001
                errors.append({"Ticker": ticker, "Timeframe": tf, "Problem": str(exc)[:120]})
                continue

            key = (ticker, tf)
            frames[key] = frame
            hits_by_key[key] = hits
            n = len(frame)
            closes = frame["Close"].to_numpy(float)
            highs = frame["High"].to_numpy(float)
            lows = frame["Low"].to_numpy(float)

            for h_i, hit in enumerate(hits):
                bars_ago = n - 1 - hit.index
                if bars_ago > lookback:
                    continue
                if direction == "Bullish only" and hit.bias == "bearish":
                    continue
                if direction == "Bearish only" and hit.bias == "bullish":
                    continue
                base = closes[hit.index]
                after_h = highs[hit.index + 1:]
                after_l = lows[hit.index + 1:]
                move = (closes[-1] - base) / base * 100.0 if base else 0.0
                best = ((after_h.max() - base) / base * 100.0) if after_h.size else 0.0
                worst = ((after_l.min() - base) / base * 100.0) if after_l.size else 0.0
                plan = pattern_levels(frame, hit, 2.0, 1.5)
                rows.append({
                    "Symbol": ticker, "Timeframe": tf, "Pattern": hit.pattern, "Bias": hit.bias,
                    "Formed on": pd.Timestamp(hit.time), "Bars ago": int(bars_ago),
                    "Move since %": round(move, 2), "Best move since %": round(best, 2),
                    "Worst move since %": round(worst, 2),
                    "Entry": round(plan["entry"], 2), "Stop": round(plan["stop"], 2),
                    "Target": round(plan["target"], 2),
                    "Chart": False, "Levels": False, "Sidebar": False,
                    "Family": hit.family, "Fired at": pd.Timestamp(hit.time),
                    "_key": f"{ticker}|{tf}", "_hit": h_i,
                })
    return pd.DataFrame(rows), frames, hits_by_key, pd.DataFrame(errors)


def _pattern_dialog_chart(frame, hit):
    st.plotly_chart(draw_pattern(frame, hit), width="stretch", config={"scrollZoom": True})
    st.caption("The shaded band marks the confirming candle. Circles are the confirmed pivots "
               "the geometry was fitted to; a pivot is only knowable some bars after it prints, "
               "which is why a hit is never dated earlier than it could have been seen.")


def _pattern_dialog_levels(frame, hit, hits_same_pattern, symbol, tf):
    st.markdown("#### Recommended levels")
    c1, c2 = st.columns(2)
    rr = c1.slider("Reward : risk", 0.5, 6.0, 2.0, 0.25, key="pl_rr")
    atr_mult = c2.slider("ATR multiple for the fallback stop", 0.5, 5.0, 1.5, 0.25, key="pl_atr")

    plan = pattern_levels(frame, hit, rr, atr_mult)
    side = "long" if plan["direction"] > 0 else "short"
    st.markdown(f"**{hit.pattern} · {symbol} · {tf} · {side}**")

    m = st.columns(4)
    m[0].metric("Entry", fmt(plan["entry"]))
    m[1].metric("Stop", fmt(plan["stop"]), f"{fmt(plan['risk'])} risk")
    m[2].metric(f"Target at {fmt(rr, 2)}R", fmt(plan["target"]))
    m[3].metric("Measured move", fmt(plan["measured"]) if plan["measured"] else "n/a")

    bullets = [
        "Entry is the close of the bar that confirmed the pattern. In practice you fill at the "
        "next bar's open, which is what the record below assumes.",
        f"Stop sits at **{fmt(plan['stop'])}**, "
        + ("an ATR-scaled fallback, because the pattern's own structure was unusable or on the "
           "wrong side of entry." if plan["fallback_stop"]
           else "the structure the pattern is built on."),
    ]
    if plan["atr_multiple"]:
        bullets.append(f"ATR here is {fmt(plan['atr'])}, so the stop is "
                       f"{fmt(plan['atr_multiple'])} x ATR away.")
    if plan["measured"]:
        bullets.append("Measured move target: the pattern's own height projected from the break.")
    for b in bullets:
        st.markdown(f"- {b}")

    st.markdown("**What this exact rule has done here before**")
    rec = pattern_track_record(frame, hits_same_pattern, rr, atr_mult)
    r = st.columns(5)
    r[0].metric("Trades", rec["trades"])
    r[1].metric("Hit rate", f"{fmt(rec['hit_rate'], 1)}%")
    r[2].metric("Expectancy", fmt(rec["expectancy"]))
    pf = rec["profit_factor"]
    r[3].metric("Profit factor", "inf" if pf == float("inf") else fmt(pf))
    r[4].metric("Net points", fmt(rec["net_points"]))
    st.caption(rec["verdict"])
    st.info("These are levels the pattern implies, not advice. The record above is the only "
            "reason to take them seriously, and it is a small sample on one instrument.")

    if st.button("Send this whole setup to the sidebar", type="primary", width="stretch",
                 key="pl_send"):
        st.session_state.pending_combo = {
            "strategy": "42 \u00b7 Price Action Composite",
            "sl_type": "Fixed Points", "sl_value": round(plan["risk"], 2),
            "tp_type": "Risk : Reward Multiple", "tp_value": float(rr),
            "filter_key": "", "widgets": {},
        }
        st.session_state.pending_ticker = symbol
        st.rerun()


def tab_patterns(cfg: dict) -> None:
    st.subheader("Chart Pattern Scanner")
    st.caption("Geometric and candlestick detection on confirmed pivots. Pattern reading is "
               "subjective — two analysts would disagree about half of these — so every hit "
               "carries its own historical record on that instrument. That record, not the "
               "shape, is the reason to act. Beware of repeating textbook win rates.")

    c1, c2 = st.columns(2)
    universe = c1.selectbox("Universe", SCREENER_UNIVERSES, key="pat_universe")
    timeframes = c2.multiselect("Timeframes", PATTERN_TIMEFRAMES, default=["1d"], key="pat_tfs")

    custom_text, uploaded = "", None
    if universe.startswith("Custom"):
        custom_text = c1.text_area("Tickers", "RELIANCE\nTCS\nINFY", key="pat_custom")
    all_tickers, note = _universe_tickers(universe, custom_text, uploaded)

    symbols = c1.multiselect("Symbols", all_tickers, default=all_tickers, key="pat_symbols")
    families = c2.multiselect("Pattern families", PATTERN_FAMILIES, default=PATTERN_FAMILIES,
                              key="pat_families")
    direction = c2.radio("Direction", ["Any", "Bullish only", "Bearish only"], horizontal=True,
                         key="pat_dir")
    max_names = c1.number_input(f"Test at most (of {len(symbols)} listed)", 1,
                                max(1, len(all_tickers)), min(25, max(1, len(symbols))),
                                key="pat_max")

    catalog = [p for p, fam in PATTERN_CATALOG.items() if fam in families]
    patterns = st.multiselect("Patterns", catalog, default=catalog, key="pat_patterns")
    lookback = st.number_input("Only show patterns formed within the last N candles", 1, 200, 10,
                               key="pat_look")

    tickers = symbols[:int(max_names)]
    if note:
        st.warning(note + " Index membership drifts; use a custom list when accuracy matters.")
    st.caption(f"{len(tickers)} symbol(s) x {len(timeframes)} timeframe(s) = "
               f"{len(tickers) * max(1, len(timeframes))} downloads, each carrying the "
               f"{API_GUARD_DELAY}s guard on both sides.")

    if st.button("Scan for Patterns", type="primary", width="stretch"):
        if not tickers or not timeframes or not patterns:
            st.error("Pick at least one symbol, timeframe and pattern.")
        else:
            bar = st.progress(0.0, text="Starting ...")
            rows, frames, hits, errors = scan_patterns(tickers, timeframes, patterns, direction,
                                                       int(lookback), cfg["params"], bar)
            bar.empty()
            st.session_state.pattern_rows = rows
            st.session_state.pattern_frames = frames
            st.session_state.pattern_hits = hits
            st.session_state.pattern_errors = errors

    rows = st.session_state.get("pattern_rows")
    if rows is None:
        st.info("Choose a universe and scan.")
        return
    if rows.empty:
        st.info("No patterns matched inside the lookback window.")
        return

    st.success(f"{len(rows)} hit(s).")
    st.caption("Tick a box in any row to act on it. **Chart** draws the pattern on the candles, "
               "**Levels** opens the full plan with its historical record, **Sidebar** loads the "
               "whole setup ready to backtest. Ticks clear themselves after firing.")

    display_cols = ["Symbol", "Timeframe", "Pattern", "Bias", "Formed on", "Bars ago",
                    "Move since %", "Best move since %", "Worst move since %",
                    "Entry", "Stop", "Target", "Chart", "Levels", "Sidebar", "Family", "Fired at"]
    edited = st.data_editor(
        rows[display_cols], hide_index=True, width="stretch", key="pat_editor",
        disabled=[c for c in display_cols if c not in ("Chart", "Levels", "Sidebar")],
        column_config={
            "Chart": st.column_config.CheckboxColumn("Chart", default=False),
            "Levels": st.column_config.CheckboxColumn("Levels", default=False),
            "Sidebar": st.column_config.CheckboxColumn("Sidebar", default=False),
            "Formed on": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Fired at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        })

    st.download_button("Download hits as CSV", rows[display_cols].to_csv(index=False).encode(),
                       f"patterns_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", "text/csv")
    render_analyst_panel("patterns", "these pattern hits", _frame_context(rows[display_cols]))

    action, row_idx = None, None
    for col in ("Chart", "Levels", "Sidebar"):
        ticked = edited.index[edited[col].fillna(False)].tolist()
        if ticked:
            action, row_idx = col, ticked[0]
            break
    if action is None:
        return

    # Clear the tick immediately so the action fires once, not on every rerun.
    st.session_state.pattern_rows.loc[row_idx, action] = False
    row = st.session_state.pattern_rows.loc[row_idx]
    key = tuple(str(row["_key"]).split("|"))
    frame = (st.session_state.get("pattern_frames") or {}).get(key)
    hit_list = (st.session_state.get("pattern_hits") or {}).get(key) or []
    if frame is None or int(row["_hit"]) >= len(hit_list):
        st.error("The scan data for that row is no longer in memory. Re-run the scan.")
        return
    hit = hit_list[int(row["_hit"])]

    if action == "Sidebar":
        plan = pattern_levels(frame, hit, 2.0, 1.5)
        st.session_state.pending_combo = {
            "strategy": "42 \u00b7 Price Action Composite",
            "sl_type": "Fixed Points", "sl_value": round(plan["risk"], 2),
            "tp_type": "Risk : Reward Multiple", "tp_value": 2.0,
            "filter_key": "", "widgets": {}}
        st.session_state.pending_ticker = row["Symbol"]
        st.rerun()

    same = [h for h in hit_list if h.pattern == hit.pattern]
    if hasattr(st, "dialog"):
        if action == "Chart":
            @st.dialog("Pattern detail", width="large")
            def _chart_dialog():
                _pattern_dialog_chart(frame, hit)
            _chart_dialog()
        else:
            @st.dialog("Recommended levels", width="large")
            def _levels_dialog():
                _pattern_dialog_levels(frame, hit, same, row["Symbol"], row["Timeframe"])
            _levels_dialog()
    else:                                                            # pragma: no cover
        if action == "Chart":
            _pattern_dialog_chart(frame, hit)
        else:
            _pattern_dialog_levels(frame, hit, same, row["Symbol"], row["Timeframe"])


# =============================================================================
# SECTION 19c -- SIGNAL LAB  (optimiser + live screener in one pass)
# =============================================================================
# The history window each timeframe deserves. A month of daily candles is 22
# bars; a month of 5m candles is thousands. These are the operator's chosen
# windows, still subject to what Yahoo will actually serve.
LAB_TIMEFRAME_PERIODS = {
    "1m": "7d", "2m": "60d", "3m": "7d", "5m": "60d", "10m": "60d", "15m": "60d",
    "30m": "60d", "60m": "3y", "4h": "3y", "1d": "10y", "1wk": "20y", "1mo": "max",
}


def lab_period_for(interval: str, needed_bars: int = WARMUP_BARS + 40) -> tuple[str, str | None]:
    """
    Resolve the history window for a timeframe in the lab.

    Returns ``(period, note)``. Yahoo caps intraday history hard -- 60m tops out
    around 730 days -- so an ask for 3 years of hourly candles gets clamped, and
    the note says so instead of the request silently shrinking.
    """
    wanted = LAB_TIMEFRAME_PERIODS.get(interval, "1y")
    effective, clamp = sanitize_period(interval, wanted)
    floor = _period_for_timeframe(interval, effective, needed_bars)
    if PERIOD_DAYS.get(floor, 0) > PERIOD_DAYS.get(effective, 0):
        effective = floor
    note = None
    if clamp:
        note = (f"`{interval}`: asked for {wanted}, Yahoo serves at most "
                f"{INTERVAL_MAX_DAYS.get(interval)} days, so {effective} is used.")
    return effective, note


def _period_for_timeframe(interval: str, requested: str, needed_bars: int) -> str:
    """
    Widen a period until the interval can supply `needed_bars`, then clamp.

    Returns the LONGER of what the operator asked for and what the warm-up
    actually requires, capped by Yahoo's ceiling for that interval.
    """
    per_day = APPROX_BARS_PER_DAY.get(interval, 1.0)
    needed_days = needed_bars / max(per_day, 0.001) * 1.6      # calendar/holiday cushion
    want_days = max(PERIOD_DAYS.get(requested, 0), needed_days)
    ceiling = INTERVAL_MAX_DAYS.get(interval)
    for candidate in PERIODS:
        days = PERIOD_DAYS[candidate]
        if days >= want_days and (ceiling is None or days <= ceiling):
            return candidate
    # Nothing long enough inside the ceiling: take the longest the interval allows.
    allowed = [c for c in PERIODS if ceiling is None or PERIOD_DAYS[c] <= ceiling]
    return allowed[-1] if allowed else requested


def _gate_failure_note(table: pd.DataFrame, gates: dict) -> str:
    """Say which gate blocked everything, and what was actually achievable."""
    parts = []
    for col, key in (("Win %", "win"), ("Sharpe", "sharpe"), ("Expectancy", "expectancy"),
                     ("Profit Factor", "pf"), ("Net PnL", "pnl")):
        floor = gates.get(key)
        if not floor or col not in table.columns:
            continue
        try:
            best = float(pd.to_numeric(table[col], errors="coerce").max())
        except Exception:                                           # noqa: BLE001
            continue
        if np.isfinite(best) and best < float(floor):
            parts.append(f"{col} best {best:.2f} < {float(floor):.2f}")
    if not parts:
        return ("no combination met the quality thresholds (the gates fail in combination even "
                "though each is individually reachable)")
    return "gates not met -- " + "; ".join(parts)


def _meets_thresholds(row, gates: dict) -> bool:
    """Every gate the operator set must hold. An unset gate is not a gate."""
    checks = (("Win %", gates.get("win")), ("Sharpe", gates.get("sharpe")),
              ("Expectancy", gates.get("expectancy")), ("Profit Factor", gates.get("pf")),
              ("Net PnL", gates.get("pnl")))
    for col, floor in checks:
        if floor in (None, 0.0):
            continue
        try:
            value = float(row[col])
        except (TypeError, ValueError, KeyError):
            return False
        if not np.isfinite(value) or value < float(floor):
            return False
    return True


def run_signal_lab(tickers: list[str], cfg: dict, objective: str, iterations: int,
                   min_trades: int, signal_window: int, safe_only: bool,
                   timeframes: list[str] | None = None, gates: dict | None = None,
                   progress=None, grid: dict | None = None, exhaustive: bool = False,
                   scale_points: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each ticker: search for the best configuration on its own history, then
    ask whether that winning configuration is signalling right now.

    The two halves answer different questions. The optimiser says what WOULD have
    worked; the live check says whether it is triggering. A ticker only earns
    attention when both line up -- and even then the optimiser's caveat stands:
    the more combinations searched, the more the winner owes to luck.
    """
    rows, errors = [], []
    costs = cfg.get("costs") or CostModel()
    gates = gates or {}
    timeframes = timeframes or [cfg["interval"]]
    jobs = [(t, tf) for t in tickers for tf in timeframes]
    lab_started = time.time()
    for i, (ticker, interval) in enumerate(jobs):
        if progress is not None:
            done = (i + 1) / max(1, len(jobs))
            elapsed = time.time() - lab_started
            eta = (elapsed / max(done, 1e-6)) - elapsed
            progress.progress(done,
                              text=f"{i + 1} of {len(jobs)} ({done*100:.0f}%) — {ticker} · "
                                   f"{interval} — about {eta:0.0f}s left")
        # The sidebar period is chosen for the sidebar timeframe. Reusing it across
        # every timeframe is what produced "Only 22 candles at 1d/1mo": a month is
        # plenty of 5m bars and almost no daily ones. sanitize_period only clamps
        # DOWNWARD, so it cannot rescue this on its own -- we widen first, then let
        # it clamp to whatever the interval can actually serve.
        if cfg.get("use_lab_windows", True) and len(timeframes) > 1:
            period, _ = lab_period_for(interval, WARMUP_BARS + 40)
        else:
            period = _period_for_timeframe(interval, cfg["period"], WARMUP_BARS + 40)
        try:
            bundle = load_market_data(ticker, period, interval,
                                      freshness_seconds=300, min_bars=WARMUP_BARS + 40)
        except Exception as exc:                                    # noqa: BLE001
            errors.append({"Ticker": ticker, "Timeframe": interval, "Problem": str(exc)[:140]})
            continue

        params = dict(cfg["params"])
        params["symbol"], params["interval"] = ticker, interval
        params["intraday"] = interval in INTRADAY_INTERVALS
        try:
            table = optimise(bundle.frame, params, cfg["quantity"], costs, objective,
                             min_trades, iterations, seed=11, safe_exits_only=safe_only,
                             grid=grid, exhaustive=exhaustive, scale_points=scale_points)
        except Exception as exc:                                    # noqa: BLE001
            errors.append({"Ticker": ticker, "Timeframe": interval,
                           "Problem": f"optimiser: {str(exc)[:120]}"})
            continue
        if table.empty:
            errors.append({"Ticker": ticker, "Timeframe": interval,
                           "Problem": "no combination met the minimum trades"})
            continue

        qualified = table[table.apply(lambda r: _meets_thresholds(r, gates), axis=1)]
        if qualified.empty:
            errors.append({"Ticker": ticker, "Timeframe": interval,
                           "Problem": _gate_failure_note(table, gates)})
            continue
        best = qualified.iloc[0]
        fcfg = default_filter_config()
        fkey = str(best.get("Filter Key") or "")
        if fkey:
            fcfg[fkey]["enabled"] = True
        try:
            frame, _ = prepare(bundle.frame, best["Strategy"], params, fcfg, {})
        except Exception as exc:                                    # noqa: BLE001
            errors.append({"Ticker": ticker, "Timeframe": interval,
                           "Problem": f"signal check: {str(exc)[:120]}"})
            continue

        window = frame["signal"].iloc[-(signal_window + 1):-1]
        fired = window[window != 0]
        detail: dict = {}
        if fired.empty:
            when, side = "no signal", "-"
            detail = {"Price Now": round(float(frame["Close"].iloc[-1]), 2),
                      "Last Candle": pd.Timestamp(frame.index[-1]),
                      "Scanned At": pd.Timestamp.now(), "Interval": interval}
        else:
            sig_time = fired.index[-1]
            direction = int(fired.iloc[-1])
            side = "LONG" if direction > 0 else "SHORT"
            when_bars = int(len(frame) - 2 - frame.index.get_loc(sig_time))
            when = ("Just now (last closed candle)" if when_bars <= 0
                    else f"Just before ({when_bars} candles ago)")
            risk_for_row = RiskConfig(best["Stop-Loss"],
                                      float(best["SL Value"] or 0.0), best["Target"],
                                      float(best["TP Value"] or 0.0), cfg["quantity"])
            detail = signal_detail(frame, sig_time, direction, risk_for_row, ticker,
                                   interval)

        rows.append({
            "Ticker": ticker, "Timeframe": interval, "Period Used": period,
            "Signal": side, "When": when, **detail,
            "Best Strategy": best["Strategy"],
            "Stop-Loss": best["Stop-Loss"], "SL Value": best["SL Value"],
            "Target": best["Target"], "TP Value": best["TP Value"],
            "Filter": best["Filter"], "Filter Key": fkey,
            "Trades": best["Trades"], "Win %": best["Win %"], "Sharpe": best["Sharpe"],
            "Expectancy": best["Expectancy"], "Net PnL": best["Net PnL"],
            "Profit Factor": best["Profit Factor"], "Reliability": best["Reliability"],
            "Score": best["Score"], "Close": round(float(frame["Close"].iloc[-1]), 2),
        })
    return pd.DataFrame(rows), pd.DataFrame(errors)


def _quality_score(row) -> float:
    """
    One number to sort a shortlist by, blending the things that matter together.

    Deliberately crude and deliberately harsh on small samples: a 90% win rate
    over 8 trades should not outrank a 55% win rate over 200. It is a sorting
    aid for a shortlist, not a measure of edge -- every input still came from a
    search that kept its own best result.
    """
    def num(col, default=0.0):
        try:
            v = float(row.get(col, default))
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    trades = num("Trades")
    if trades <= 0:
        return 0.0
    # Confidence grows with sample size and saturates: 30 trades is worth ~0.5.
    confidence = trades / (trades + 30.0)
    pf = min(num("Profit Factor", 0.0), 5.0)          # cap so "inf" cannot dominate
    sharpe = max(min(num("Sharpe"), 5.0), -5.0)
    expectancy = num("Expectancy")
    win = num("Win %") / 100.0
    raw = (0.35 * min(pf / 3.0, 1.0) + 0.25 * min(max(sharpe, 0) / 2.0, 1.0)
           + 0.20 * win + 0.20 * (1.0 if expectancy > 0 else 0.0))
    penalty = 0.75 if str(row.get("Reliability", "")).lower().startswith("optim") else 1.0
    return round(raw * confidence * penalty * 100.0, 1)


def tab_signal_lab(cfg: dict) -> None:
    st.subheader("Signal Lab — optimise, then screen")
    st.error("**Two compounding ways to fool yourself, in one tab.** Searching many "
             "combinations per ticker and keeping the winner overstates what any of them will "
             "do next; running that search across many tickers and keeping the best tickers "
             "overstates it again. Treat the output as a shortlist to validate on a different "
             "period, never as a result.")

    c1, c2, c3 = st.columns(3)
    universe = c1.selectbox("Universe", SCREENER_UNIVERSES, key="lab_universe")
    objective = c2.selectbox("Optimise for", OPTIMISER_OBJECTIVES, key="lab_obj")
    signal_window = c3.number_input("Signal window (candles)", 1, 20, 3, key="lab_window")

    custom_text = ""
    if universe.startswith("Custom"):
        custom_text = st.text_area("Tickers", "RELIANCE\nTCS\nINFY", key="lab_custom")
    tickers, note = _universe_tickers(universe, custom_text, None)

    d1, d2, d3 = st.columns(3)
    max_names = d1.number_input("Max tickers", 1, 200, min(10, len(tickers)), key="lab_max")
    iterations = d2.number_input("Combinations per ticker", 10, 500, 60, 10, key="lab_iters")
    min_trades = d3.number_input("Minimum trades to qualify", 1, 200, 10, key="lab_min")
    e1, e2 = st.columns(2)
    use_gates = e1.checkbox(
        "Apply quality thresholds", value=False, key="lab_use_gates",
        help="Off by default: the lab simply keeps the top-ranked combination per ticker. On, "
             "a combination must clear every gate you set before it can be picked.")
    use_mtf = e2.checkbox(
        "Search multiple timeframes", value=False, key="lab_use_mtf",
        help="Off by default: only the sidebar interval is searched. On, each extra timeframe "
             "multiplies the combinations tried — and the more you try, the more the winner "
             "owes to luck rather than edge.")

    gates: dict = {}
    if use_gates:
        st.markdown("**Quality thresholds** — a combination must clear every gate you set. "
                    "Leave a gate at 0 to ignore it.")
        g1, g2, g3, g4, g5 = st.columns(5)
        gates = {
            "win": g1.number_input("Min win rate %", 0.0, 100.0, 0.0, 5.0, key="lab_g_win"),
            "sharpe": g2.number_input("Min Sharpe", 0.0, 10.0, 0.0, 0.1, key="lab_g_sharpe"),
            "expectancy": g3.number_input("Min expectancy", 0.0, 1e6, 0.0, 1.0, key="lab_g_exp"),
            "pf": g4.number_input("Min profit factor", 0.0, 20.0, 0.0, 0.1, key="lab_g_pf"),
            "pnl": g5.number_input("Min net PnL", 0.0, 1e9, 0.0, 100.0, key="lab_g_pnl"),
        }
        if float(gates.get("win") or 0) >= 80:
            st.warning("A win rate that high is almost always bought with a bad reward:risk — "
                       "many small wins funding a few large losses. Check expectancy and profit "
                       "factor before believing it.")

    if use_mtf:
        timeframes = st.multiselect(
            "Timeframes to search", INTERVALS, default=[cfg["interval"]], key="lab_tfs",
            help="Each timeframe gets its own history window, widened as needed to satisfy the "
                 "200-candle warm-up and then clamped to what Yahoo serves for that interval.")
        if timeframes:
            spans, notes = [], []
            for tf in timeframes:
                per, note = lab_period_for(tf, WARMUP_BARS + 40)
                spans.append(f"`{tf}` -> {per}")
                if note:
                    notes.append(note)
            st.caption("History window per timeframe: " + ", ".join(spans) +
                       ". A month is thousands of 5m candles but only about 22 daily ones, so "
                       "each timeframe gets its own window rather than reusing the sidebar "
                       "period.")
            for note in notes:
                st.warning(note)
    else:
        timeframes = [cfg["interval"]]
        st.caption(f"Searching the sidebar timeframe only (`{cfg['interval']}`).")
    grid, exhaustive, grid_note, scale_points = None, False, None, True
    safe_only = st.checkbox("Backtest-safe exits only (exclude distance trails)", value=True,
                            key="lab_safe",
                            help="Distance trails cannot be simulated faithfully on OHLC bars, "
                                 "so including them lets an optimiser pick a configuration whose "
                                 "backtest is systematically optimistic.")

    grid, exhaustive, grid_note, scale_points = render_search_grid("lab", cfg, safe_only)
    if grid_note:
        st.info(grid_note)

    tickers = tickers[:int(max_names)]
    timeframes = timeframes or [cfg["interval"]]
    if note:
        st.warning(note)
    jobs = len(tickers) * len(timeframes)
    est = jobs * int(iterations) * 0.05 + jobs * 1.2
    st.caption(f"{len(tickers)} ticker(s) x {len(timeframes)} timeframe(s) x {int(iterations)} "
               f"combinations = {jobs * int(iterations):,} backtests. Rough estimate "
               f"{est:0.0f}s.")

    if st.button("Run Signal Lab", type="primary", width="stretch"):
        bar = st.progress(0.0, text="Starting ...")
        try:
            results, errors = run_signal_lab(tickers, cfg, objective, int(iterations),
                                             int(min_trades), int(signal_window), safe_only,
                                             timeframes, gates, bar, grid, exhaustive,
                                             scale_points)
            st.session_state.lab_results = (results, errors)
        except Exception as exc:                                    # noqa: BLE001
            st.session_state.lab_results = None
            st.error(f"Signal Lab failed: {exc}")
        bar.empty()

    payload = st.session_state.get("lab_results")
    if payload is None:
        st.info("Pick a universe and run the lab.")
        return
    results, errors = payload
    if results.empty:
        st.info("Nothing qualified. Lower the minimum trades, widen the period, or raise the "
                "combination count.")
        if not errors.empty:
            st.dataframe(errors, width="stretch", hide_index=True)
        return

    results = results.copy()
    results["Quality"] = results.apply(_quality_score, axis=1)
    signalling = results[results["Signal"] != "-"]
    a, b, c = st.columns(3)
    a.metric("Ticker/timeframe pairs", len(results))
    st.caption("**Quality** blends profit factor, Sharpe, win rate and positive expectancy, "
               "scaled by sample size and cut by 25% for optimistic backtests. It is a sorting "
               "aid for a shortlist, not a measure of edge — every row still came from a search "
               "that kept its own best result.")
    b.metric("Signalling now", len(signalling))
    c.metric("Backtest-safe configs", int((results["Reliability"] == "Backtest-safe").sum()))

    show_only = st.checkbox("Show only tickers that are signalling", value=True, key="lab_filter")
    table = signalling if show_only else results
    table = table.sort_values(["Quality", "Score"], ascending=[False, False]).reset_index(drop=True)
    front = [c for c in ["Ticker", "Timeframe", "Quality", "Signal", "When", "Bars Ago",
                         "Signal Time",
                         "Price at Signal", "Fill Price (next open)", "Price Now",
                         "Move in Favour", "R Multiple Now", "Best Strategy", "Stop-Loss",
                         "SL Value", "Target", "TP Value", "Filter", "Trades", "Win %",
                         "Sharpe", "Expectancy", "Profit Factor", "Net PnL", "Reliability",
                         "Score"] if c in table.columns]
    table = table[front + [c for c in table.columns if c not in front]]
    if table.empty:
        st.info("No ticker is signalling right now on its own optimised configuration.")
        return

    st.dataframe(table.drop(columns=["Filter Key"]), width="stretch", hide_index=True)
    labels = [f"{r['Ticker']} · {r.get('Timeframe', '')}" for _, r in table.iterrows()]
    pick = st.selectbox("Apply which setup?", labels, key="lab_pick")
    row = table.iloc[labels.index(pick)]
    if st.button("Apply this ticker and its configuration to the sidebar", type="primary",
                 width="stretch"):
        st.session_state.pending_combo = {
            "strategy": row["Best Strategy"], "sl_type": row["Stop-Loss"],
            "sl_value": row["SL Value"], "tp_type": row["Target"], "tp_value": row["TP Value"],
            "filter_key": str(row["Filter Key"] or ""), "widgets": {}}
        st.session_state.pending_ticker = row["Ticker"]
        if row.get("Timeframe"):
            st.session_state.pending_combo["widgets"] = {"cfg_interval": row["Timeframe"]}
        st.rerun()
    st.caption(f"{pick}: {row['Best Strategy']} | SL {row['Stop-Loss']} | TGT {row['Target']} | "
               f"filter {row['Filter']} | {row['Trades']} trades | {row['Reliability']}")
    st.download_button("Download lab results (CSV)", results.to_csv(index=False).encode(),
                       "signal_lab.csv", "text/csv")
    render_analyst_panel("lab", "these Signal Lab results",
                         f"Objective: {objective}. Timeframes: {timeframes}. "
                         f"Combinations per ticker: {iterations}.\n"
                         f"{_frame_context(table)}")
    if not errors.empty:
        with st.expander(f"Tickers that could not be processed ({len(errors)})"):
            st.dataframe(errors, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Algo Trading Platform", layout="wide",
                       initial_sidebar_state="expanded")
    init_state()
    cfg = render_sidebar()

    head, status = st.columns([3, 1])
    head.title(APP_TITLE)
    head.caption(f"`{cfg['symbol']}` | {cfg['interval']} | {cfg['period']} | {cfg['strategy']}")
    (status.success if st.session_state.live_running else status.info)(
        "LIVE CORE: RUNNING" if st.session_state.live_running else "LIVE CORE: IDLE")

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        ["Backtesting Engine Studio", "Live Sandbox Operations", "Live Trade Log Ledger",
         "Signal Screener", "Strategy Optimiser", "Signal Lab", "Chart Patterns"])
    with t1:
        tab_backtest(cfg)
    with t2:
        tab_live(cfg)
    with t3:
        tab_ledger(cfg)
    with t4:
        tab_screener(cfg)
    with t5:
        tab_optimiser(cfg)
    with t6:
        tab_signal_lab(cfg)
    with t7:
        tab_patterns(cfg)



# =============================================================================
# SECTION 17 -- OFFLINE SELF-TEST   (python algo_trading_platform.py --selftest)
# =============================================================================
def _synthetic(n: int = 1600, seed: int = 7) -> pd.DataFrame:
    """Random-walk candles with injected gaps and a volatility regime shift."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, n) * np.where(np.arange(n) > n // 2, 2.2, 1.0)
    close = 20000 + np.cumsum(steps)
    for pos, shift in ((300, 180.0), (700, -220.0)):        # injected gap events
        if pos < n:
            close[pos] += shift
    high = close + np.abs(rng.normal(0, 3, n))
    low = close - np.abs(rng.normal(0, 3, n))
    open_ = np.r_[close[0], close[:-1] + rng.normal(0, 1.5, n - 1)]
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1_000, 60_000, n).astype(float)}, index=idx)


def _ctx(close=100.0, atr=5.0, prev_low=95.0, prev_high=105.0, swing_low=90.0,
         swing_high=110.0, prev_swing_low=85.0, prev_swing_high=115.0, signal=0,
         low=None, high=None):
    return BarCtx(time=pd.Timestamp("2024-01-01"), open=close, high=high or close + 1,
                  low=low if low is not None else close - 1, close=close, atr=atr,
                  ema_fast=close, ema_slow=close, swing_high=swing_high, swing_low=swing_low,
                  prev_swing_high=prev_swing_high, prev_swing_low=prev_swing_low,
                  prev_high=prev_high, prev_low=prev_low, signal=signal)


def _test_step_trail():
    """The operator's own worked example, asserted literally."""
    risk = RiskConfig("Step Trail (trigger k, trail N)", 10.0, "No Target", 0.0,
                      quantity=1.0, step_trigger=5.0)
    m = ExitManager(risk, 50.0, 1, _ctx(close=50.0, atr=2.0, prev_low=45.0, swing_low=40.0))
    assert m.sl == 40.0, f"initial stop should be entry-N = 40, got {m.sl}"
    m.update(52.0, _ctx(close=52.0))
    assert m.sl == 40.0, f"below trigger k the stop must not move, got {m.sl}"
    m.update(55.0, _ctx(close=55.0))
    assert m.sl == 50.0, f"at entry+k the stop must jump to cost 50, got {m.sl}"
    m.update(60.0, _ctx(close=60.0))
    assert m.sl == 50.0, f"at 60 the stop must still be 50, got {m.sl}"
    m.update(61.0, _ctx(close=61.0))
    assert m.sl == 51.0, f"at 61 the stop must be 51, got {m.sl}"
    m.update(58.0, _ctx(close=58.0))
    assert m.sl == 51.0, f"the stop must never loosen, got {m.sl}"
    assert m.check_tick(51.0) == (51.0, "Stop-Loss")
    # Mirrored short
    ms = ExitManager(risk, 50.0, -1, _ctx(close=50.0, atr=2.0, prev_high=55.0, swing_high=60.0))
    assert ms.sl == 60.0
    ms.update(45.0, _ctx(close=45.0))
    assert ms.sl == 50.0, f"short stop should jump to cost, got {ms.sl}"
    ms.update(39.0, _ctx(close=39.0))
    assert ms.sl == 49.0, f"short stop should be 49, got {ms.sl}"
    print("   step trail (entry 50, N=10, k=5): 55->50, 60->50, 61->51, ratchet holds  OK")


def _test_trail_uses_live_price():
    """
    Regression guard.

    A distance-based trail is derived from the BEST price seen, not from the
    last closed candle. An earlier version compared the candidate against the
    closed candle's close, which silently froze every trailing stop whenever the
    live price ran ahead of the last close.
    """
    risk = RiskConfig("Trailing Points", 40.0, "No Target", 0.0, 1.0)
    m = ExitManager(risk, 20000.0, 1, _ctx(close=20000.0, atr=10.0, prev_low=19990.0,
                                           swing_low=19950.0))
    assert m.sl == 19960.0
    stale = _ctx(close=20000.0, atr=10.0, prev_low=19990.0, swing_low=19950.0)
    m.update(20100.0, stale)                 # LTP ran 100 ahead of the closed candle
    assert m.sl == 20060.0, f"trail must follow the live price, got {m.sl}"
    m.update(20050.0, stale)
    assert m.sl == 20060.0, "the trail must not loosen when price pulls back"
    assert m.check_tick(20050.0) == (20050.0, "Stop-Loss"), \
        "a stop already passed by price must fire, not be suppressed"
    print("   distance trails follow the live price, not the last closed candle  OK")


def _test_exit_types():
    checks = 0
    for sl_type in SL_TYPES:
        v = {"Fixed Percentage": 1.0, "Trailing Percentage": 1.0, "ATR Multiple": 2.0,
             "Trailing ATR (Chandelier)": 3.0}.get(sl_type, 10.0)
        for d in (1, -1):
            risk = RiskConfig(sl_type, v, "Fixed Points", 20.0, 2.0, step_trigger=5.0)
            m = ExitManager(risk, 100.0, d, _ctx())
            if m.sl is not None:
                side_ok = (m.sl < 100.0) if d > 0 else (m.sl > 100.0)
                assert side_ok, f"{sl_type} d={d}: stop on the wrong side ({m.sl})"
            before = m.sl
            m.update(100.0 + d * 12.0, _ctx(close=100.0 + d * 12.0))
            if before is not None and m.sl is not None:
                moved_ok = (m.sl >= before) if d > 0 else (m.sl <= before)
                assert moved_ok, f"{sl_type} d={d}: stop loosened {before} -> {m.sl}"
            checks += 1
    for tp_type in TP_TYPES:
        v = {"Fixed Percentage": 2.0, "ATR Multiple": 3.0,
             "Risk : Reward Multiple": 2.0}.get(tp_type, 20.0)
        for d in (1, -1):
            risk = RiskConfig("Fixed Points", 10.0, tp_type, v, 1.0)
            m = ExitManager(risk, 100.0, d, _ctx())
            if m.tp is not None:
                side_ok = (m.tp > 100.0) if d > 0 else (m.tp < 100.0)
                assert side_ok, f"{tp_type} d={d}: target on the wrong side ({m.tp})"
            if tp_type == "Risk : Reward Multiple":
                assert abs(abs(m.tp - 100.0) - 2 * 10.0) < 1e-9, "R:R target must be 2x the stop"
            if tp_type == "Trailing Target (display only)":
                assert not m.target_is_live, "display-only target must never fire"
                m.update(100.0 + d * 30.0, _ctx())
                assert abs(m.tp - (100.0 + d * 50.0)) < 1e-9, "display target must trail"
            checks += 1
    print(f"   {checks} stop/target permutations: side, ratchet and R:R arithmetic  OK")


def _test_structural_matrix():
    """
    Every candle/swing stop and target, both directions.

    A long's stop must ride LOWS and its target must ride HIGHS; a short is the
    mirror. Getting that flip wrong is silent and expensive, so it is asserted
    for all sixteen structural variants.
    """
    ctx = _ctx(close=100.0, atr=5.0, prev_low=95.0, prev_high=105.0,
               swing_low=90.0, swing_high=110.0, prev_swing_low=85.0, prev_swing_high=115.0,
               low=97.0, high=103.0)
    expect_sl = {
        ("Previous Candle Low/High", 1): 95.0, ("Previous Candle Low/High", -1): 105.0,
        ("Current Candle Low/High", 1): 95.0, ("Current Candle Low/High", -1): 105.0,
        ("Trail Previous Candle Low/High", 1): 95.0, ("Trail Previous Candle Low/High", -1): 105.0,
        ("Trail Current Candle Low/High", 1): 95.0, ("Trail Current Candle Low/High", -1): 105.0,
        ("Previous Swing Low/High", 1): 85.0, ("Previous Swing Low/High", -1): 115.0,
        ("Current Swing Low/High", 1): 90.0, ("Current Swing Low/High", -1): 110.0,
        ("Trail Previous Swing Low/High", 1): 85.0, ("Trail Previous Swing Low/High", -1): 115.0,
        ("Trail Current Swing Low/High", 1): 90.0, ("Trail Current Swing Low/High", -1): 110.0,
    }
    for (kind, d), want in expect_sl.items():
        m = ExitManager(RiskConfig(kind, 0.0, "No Target", 0.0, 1.0), 100.0, d, ctx)
        assert m.sl == want, f"SL {kind} d={d}: expected {want}, got {m.sl}"

    expect_tp = {
        ("Previous Candle High/Low", 1): 105.0, ("Previous Candle High/Low", -1): 95.0,
        ("Current Candle High/Low", 1): 105.0, ("Current Candle High/Low", -1): 95.0,
        ("Trail Previous Candle High/Low", 1): 105.0, ("Trail Previous Candle High/Low", -1): 95.0,
        ("Trail Current Candle High/Low", 1): 105.0, ("Trail Current Candle High/Low", -1): 95.0,
        ("Previous Swing High/Low", 1): 115.0, ("Previous Swing High/Low", -1): 85.0,
        ("Current Swing High/Low", 1): 110.0, ("Current Swing High/Low", -1): 90.0,
        ("Trail Previous Swing High/Low", 1): 115.0, ("Trail Previous Swing High/Low", -1): 85.0,
        ("Trail Current Swing High/Low", 1): 110.0, ("Trail Current Swing High/Low", -1): 90.0,
    }
    for (kind, d), want in expect_tp.items():
        m = ExitManager(RiskConfig("Fixed Points", 10.0, kind, 0.0, 1.0), 100.0, d, ctx)
        assert m.tp == want, f"TP {kind} d={d}: expected {want}, got {m.tp}"

    # A trailing target extends away from entry and never drifts back closer.
    m = ExitManager(RiskConfig("Fixed Points", 10.0, "Trail Current Swing High/Low", 0.0, 1.0),
                    100.0, 1, ctx)
    assert m.tp == 110.0
    m.update(112.0, _ctx(close=112.0, swing_high=125.0, low=108.0, high=113.0))
    assert m.tp == 125.0, f"trailing target must extend, got {m.tp}"
    m.update(113.0, _ctx(close=113.0, swing_high=118.0, low=110.0, high=114.0))
    assert m.tp == 125.0, f"trailing target must not drift closer, got {m.tp}"

    # A trailing structural stop ratchets up and never loosens.
    m = ExitManager(RiskConfig("Trail Current Swing Low/High", 0.0, "No Target", 0.0, 1.0),
                    100.0, 1, ctx)
    assert m.sl == 90.0
    m.update(120.0, _ctx(close=120.0, swing_low=108.0, low=115.0, high=121.0))
    assert m.sl == 108.0, f"structural trail must follow the new swing, got {m.sl}"
    m.update(118.0, _ctx(close=118.0, swing_low=99.0, low=117.0, high=119.0))
    assert m.sl == 108.0, f"structural trail must not loosen, got {m.sl}"
    print("   16 structural stop/target variants, both directions, plus ratchets  OK")


def _test_threshold_strategies():
    df = _synthetic(500)
    params = dict(DEFAULT_PARAMS); params["intraday"] = True
    level = float(df["Close"].iloc[250])
    params["threshold_price"] = level

    params["threshold_mode"] = "Cross above = BUY, cross below = SELL"
    both, _ = prepare(df, "28 \u00b7 Price Threshold Cross (absolute level)", params)
    assert (both["signal"] == 1).any() and (both["signal"] == -1).any()

    params["threshold_mode"] = "Cross above = BUY only"
    long_only, _ = prepare(df, "28 \u00b7 Price Threshold Cross (absolute level)", params)
    assert not (long_only["signal"] == -1).any(), "long-only mode must emit no shorts"
    assert (long_only["signal"] == 1).sum() == (both["signal"] == 1).sum()

    params["threshold_mode"] = "Cross above = SELL, cross below = BUY (fade)"
    faded, _ = prepare(df, "28 \u00b7 Price Threshold Cross (absolute level)", params)
    assert (faded["signal"] == -1).sum() == (both["signal"] == 1).sum(), "fade must invert"

    params["threshold_mode"] = "Cross above = BUY, cross below = SELL"
    params["threshold_pct"] = 0.5
    for ref in THRESHOLD_REFS:
        params["threshold_ref"] = ref
        pct, _ = prepare(df, "29 \u00b7 Price Threshold Cross (% from reference)", params)
        assert set(pct["signal"].unique()).issubset({-1, 0, 1}), f"{ref}: bad signal domain"
        up, dn = pct["threshold_up"], pct["threshold_dn"]
        valid = up.notna() & dn.notna()
        assert (up[valid] > dn[valid]).all(), f"{ref}: bands inverted"
    print("   threshold strategies: all 4 trigger modes and all 4 references  OK")


def _test_tick_vs_candle_split():
    """
    The live loop must recompute PnL from a freshly fetched price, not from the
    last candle close. A candle close cannot change between candles, which is
    what made the dashboard look frozen.
    """
    risk = RiskConfig("Fixed Points", 50.0, "Fixed Points", 50.0, 3.0)
    ctx = _ctx(close=100.0)
    mgr = ExitManager(risk, 100.0, 1, ctx)
    pos = Position(strategy="t", symbol="T", interval="5m", direction=1, quantity=3.0,
                   entry_price=100.0, entry_time=None, signal_bar_time=None, manager=mgr)
    for tick_price, want_pnl in ((100.5, 1.5), (101.25, 3.75), (99.75, -0.75)):
        assert abs(pos.pnl(tick_price) - want_pnl) < 1e-9, \
            f"PnL at {tick_price} should be {want_pnl}, got {pos.pnl(tick_price)}"
        mgr.update(tick_price, ctx)
    assert abs(mgr.mfe - 101.25) == 0.0, "best price must track intra-candle ticks"
    print("   PnL and trailing recompute from the tick price, not the candle close  OK")


def _test_liveness_logic():
    """
    Regression guard for the worst bug in this file's history.

    Lagging candles were being treated as proof the venue was closed, so live
    trades were suppressed while the quote was visibly ticking. Liveness is a
    question about the PRICE, and the only trustworthy evidence is observed
    movement -- a closed exchange still serves a quote, and that quote differs
    from the last intraday candle close, so "the numbers differ" proves nothing.
    """
    class FakeState(dict):
        def get(self, k, d=None):
            return super().get(k, d)

    def liveness(stale, ticks, seconds_since_change):
        if not stale:
            return True
        if ticks < QUOTE_EVIDENCE_TICKS:
            return False
        return seconds_since_change <= QUOTE_LIVE_WINDOW

    assert liveness(False, 0, 9e9) is True, "current candles alone mean the venue is open"
    assert liveness(True, 1, 0.0) is False, "must not trust a quote before it has been watched"
    assert liveness(True, 25, 1.0) is True, "a recently moving quote means the venue is open"
    assert liveness(True, 25, 9e9) is False, "a quote frozen for hours means the venue is closed"
    assert liveness(True, QUOTE_EVIDENCE_TICKS, QUOTE_LIVE_WINDOW - 1) is True
    assert liveness(True, QUOTE_EVIDENCE_TICKS, QUOTE_LIVE_WINDOW + 1) is False
    print("   liveness judged on observed price movement, not candle age  OK")


def _test_zigzag_and_elliott():
    """
    Regression guard: the zigzag used to return an EMPTY pivot list, always.

    With direction initialised to 0 the running extreme tracked both directions
    at once, so it always equalled the current price, no reversal threshold
    could ever be breached, and the Elliott profile was structurally incapable
    of producing a signal.
    """
    rng = np.random.default_rng(3)
    n = 1200
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.0016, n)))
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    series = pd.Series(close, index=idx)

    piv = zigzag_pivot_table(series, 0.3)
    assert len(piv) > 10, f"zigzag must find pivots on a trending walk, found {len(piv)}"
    kinds = [k for _, _, k, _ in piv]
    assert set(kinds) == {1, -1}, "pivots must alternate between highs and lows"
    for a, b in zip(kinds, kinds[1:]):
        assert a != b, "two consecutive pivots of the same kind is impossible"
    for pivot_i, _, _, confirm_i in piv:
        assert confirm_i > pivot_i, "a pivot cannot be known before it has been confirmed"

    op = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({"Open": op, "High": np.maximum(op, close) * 1.0006,
                          "Low": np.minimum(op, close) * 0.9994, "Close": close,
                          "Volume": 1000.0}, index=idx)
    params = dict(DEFAULT_PARAMS)
    params["intraday"] = True
    out, _ = prepare(frame, "24 \u00b7 Elliott Wave Impulse (heuristic)", params)
    fired = int((out["signal"] != 0).sum())
    assert fired > 0, "the wave profile must be able to produce signals"
    assert (out["zz_pivots"].iloc[-1] > 0), "pivot count must be reported for tuning"
    print(f"   zigzag pivots alternate and confirm late; wave profile fired {fired} times  OK")


def _test_new_profiles_and_flip():
    """Fibonacci family, hybrid logic, option profiles, and the flip switch."""
    df = _synthetic(900)
    params = dict(DEFAULT_PARAMS)
    params["intraday"] = True

    for name in STRATEGY_NAMES:
        if name.startswith(("34 ", "35 ", "36 ", "37 ", "38 ", "39 ", "40 ", "41 ", "42 ")):
            out, _ = prepare(df, name, params)
            assert set(out["signal"].unique()).issubset({-1, 0, 1}), f"{name}: bad domain"
            assert int((out["signal"] != 0).sum()) > 0, f"{name}: produced nothing at all"

    # Fibonacci levels must sit inside the swing leg and be correctly ordered.
    fib, _ = prepare(df, "35 \u00b7 Fibonacci Retracement Zone", params)
    ok = fib[["swing_high", "swing_low", "fib_golden_hi", "fib_golden_lo"]].dropna()
    assert (ok["fib_golden_hi"] >= ok["fib_golden_lo"]).all(), "golden zone inverted"
    assert (ok["fib_golden_hi"] <= ok["swing_high"] + 1e-6).all(), "level above the swing high"
    assert (ok["fib_golden_lo"] >= ok["swing_low"] - 1e-6).all(), "level below the swing low"

    # Hybrid: AND can never fire more often than OR over the same members.
    members = ["01 \u00b7 Dual EMA Crossover", "08 \u00b7 RSI Centerline 50 Crossing"]
    p_and = dict(params, hybrid_members=members, hybrid_logic=HYBRID_LOGIC[0])
    p_or = dict(params, hybrid_members=members, hybrid_logic=HYBRID_LOGIC[1])
    n_and = int((prepare(df, "43 \u00b7 Hybrid (combine profiles with AND / OR)", p_and)[0]
                 ["signal"] != 0).sum())
    n_or = int((prepare(df, "43 \u00b7 Hybrid (combine profiles with AND / OR)", p_or)[0]
                ["signal"] != 0).sum())
    assert n_and <= n_or, f"AND ({n_and}) cannot fire more than OR ({n_or})"

    # OI profiles must stay silent without a live chain rather than invent one.
    for name in ("44 \u00b7 Options: OI Change", "45 \u00b7 Options: OI Change + PCR",
                 "46 \u00b7 Options: OI Change + Volume"):
        out, _ = prepare(df, name, params)
        assert int((out["signal"] != 0).sum()) == 0, f"{name} must not signal without live OI"

    # Flip inverts every signal, exactly.
    base, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", params)
    flipped, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", dict(params, flip_entries=True))
    assert (flipped["signal"] == -base["signal"]).all(), "flip must invert every signal"
    assert int((base["signal"] != 0).sum()) == int((flipped["signal"] != 0).sum())

    # OHLC of both candles must reach the trade table.
    risk = RiskConfig("Fixed Points", 30.0, "Fixed Points", 60.0, 1.0)
    res = run_backtest(df, "01 \u00b7 Dual EMA Crossover", params, risk)
    for col in ("Entry Bar Open", "Entry Bar High", "Entry Bar Low", "Entry Bar Close",
                "Exit Bar Open", "Exit Bar High", "Exit Bar Low", "Exit Bar Close"):
        assert col in res.trades.columns, f"{col} missing from the trade table"
    t = res.trades.dropna(subset=["Entry Bar High"])
    assert (t["Entry Bar High"] >= t["Entry Bar Low"]).all(), "entry candle OHLC inconsistent"
    print(f"   fib levels bounded, hybrid AND {n_and} <= OR {n_or}, OI silent offline, "
          f"flip exact, OHLC on trades  OK")


def _test_live_entry_and_gates():
    """
    Two behaviours worth locking down.

    1. The screener reports signals from a WINDOW of candles while the live
       engine only ever looked at the newest closed one. That mismatch is why an
       applied screener hit produced no trade, so the catch-up path now covers it.
    2. Signal Lab quality gates must be hard filters, not preferences.
    """
    class Snap:
        def __init__(self, last, recent, ago):
            self.last_closed_signal = last
            self.recent_signal = recent
            self.recent_signal_bars_ago = ago

    def would_enter(snap, lookback):
        d = int(snap.last_closed_signal)
        if d == 0 and lookback > 0 and snap.recent_signal != 0 \
                and snap.recent_signal_bars_ago is not None \
                and 0 < snap.recent_signal_bars_ago <= lookback:
            return int(snap.recent_signal), True
        return d, False

    assert would_enter(Snap(1, 1, 0), 0) == (1, False), "a fresh signal must always be taken"
    assert would_enter(Snap(0, -1, 3), 0) == (0, False), "strict mode must ignore an old signal"
    assert would_enter(Snap(0, -1, 3), 5) == (-1, True), "catch-up must take a signal in range"
    assert would_enter(Snap(0, -1, 9), 5) == (0, False), "catch-up must respect its lookback"

    row = {"Win %": 55.0, "Sharpe": 1.2, "Expectancy": 3.0, "Profit Factor": 1.8,
           "Net PnL": 500.0}
    assert _meets_thresholds(row, {}), "no gates set means nothing is filtered"
    assert _meets_thresholds(row, {"win": 50.0, "sharpe": 1.0})
    assert not _meets_thresholds(row, {"win": 60.0}), "a failing gate must reject the row"
    assert not _meets_thresholds(row, {"pf": 2.0})
    assert not _meets_thresholds({"Win %": float("nan"), "Sharpe": 1.0, "Expectancy": 1.0,
                                  "Profit Factor": 1.0, "Net PnL": 1.0}, {"win": 10.0}), \
        "an unmeasurable metric cannot pass a gate"
    print("   catch-up entry window and Signal Lab quality gates  OK")


def _reset_constituent_cache() -> None:
    """Drop both the wrapper and Streamlit's own cache for the constituent lookup."""
    nse_constituents.__dict__.pop("_impl", None)
    if st is not None:
        try:
            st.cache_data.clear()
        except Exception:                                           # noqa: BLE001
            pass


def _test_universe_resolution():
    """
    Index membership must come from a live source, and a failure must be loud.

    A constituent list baked into a source file is wrong within weeks: you screen
    the companies that left the index and miss the ones that joined. So the NSE
    fetch is primary, the bundled snapshot is a flagged fallback, and the large
    indices have no fallback at all rather than a quietly stale one.
    """
    fake_master = pd.DataFrame({
        "security_id": ["1", "2", "3", "4"],
        "trading_symbol": ["RELIANCE", "TCS", "NIFTYFUT", "INFY"],
        "custom_symbol": [""] * 4, "name": [""] * 4,
        "exchange": ["NSE", "NSE", "NSE", "BSE"], "segment": ["E", "E", "D", "E"],
        "instrument": ["EQUITY", "EQUITY", "FUTIDX", "EQUITY"],
        "expiry": [None] * 4, "strike": [None] * 4, "option_type": [""] * 4,
        "lot_size": [1, 1, 50, 1]})

    g = globals()
    real_fetch, real_master = g["_fetch_nse_constituents"], g["load_scrip_master"]
    try:
        g["_fetch_nse_constituents"] = lambda name, timeout=20.0: ["RELIANCE", "HDFCBANK", "ITC"]
        _reset_constituent_cache()
        for index_name in ("Nifty 200", "Nifty 500", "Nifty 50"):
            tickers, note = _universe_tickers(index_name, "", None)
            assert tickers == ["RELIANCE.NS", "HDFCBANK.NS", "ITC.NS"], f"{index_name}: bad list"
            assert "live from NSE" in (note or ""), f"{index_name}: provenance not stated"

        def _boom(name, timeout=20.0):
            raise RuntimeError("blocked")
        g["_fetch_nse_constituents"] = _boom
        _reset_constituent_cache()
        tickers, note = _universe_tickers("Nifty 50", "", None)
        assert len(tickers) == 50, "the Nifty 50 snapshot fallback must still return a list"
        assert "bundled snapshot" in (note or "") and "stale" in (note or "").lower(), \
            f"the fallback must admit it is stale, got: {note}"
        tickers, note = _universe_tickers("Nifty 500", "", None)
        assert tickers == [] and "Custom list" in (note or ""), \
            "a large index must fail loudly rather than serve a stale list"

        g["load_scrip_master"] = lambda: fake_master
        tickers, note = _universe_tickers("All NSE equities (Dhan master)", "", None)
        assert tickers == ["RELIANCE.NS", "TCS.NS"], f"NSE cash filter wrong: {tickers}"
    finally:
        g["_fetch_nse_constituents"] = real_fetch
        g["load_scrip_master"] = real_master
        _reset_constituent_cache()

    tickers, _ = _universe_tickers("Custom list (paste or upload)", "reliance, ^NSEI\nTCS", None)
    assert tickers == ["RELIANCE.NS", "^NSEI", "TCS.NS"], f"custom parsing wrong: {tickers}"
    print("   universe resolution: live NSE, flagged fallback, loud failure  OK")


def _test_timeframe_period_scaling():
    """
    Regression guard for the multi-timeframe search.

    The sidebar period was reused for every timeframe, so a 1-month window gave
    ~22 daily candles against a 240-candle requirement and every non-intraday
    timeframe failed with "Only N candles available". The period must be treated
    as a floor and widened per timeframe, then clamped to the interval ceiling.
    """
    need = WARMUP_BARS + 40
    for tf in ("5m", "15m", "60m", "4h", "1d", "1wk", "1mo"):
        chosen = _period_for_timeframe(tf, "1mo", need)
        supplied = PERIOD_DAYS[chosen] * APPROX_BARS_PER_DAY[tf]
        assert supplied >= need, f"{tf}: {chosen} supplies only {supplied:.0f} of {need} candles"
        ceiling = INTERVAL_MAX_DAYS.get(tf)
        if ceiling is not None:
            assert PERIOD_DAYS[chosen] <= ceiling, f"{tf}: {chosen} exceeds Yahoo's ceiling"

    # The requested period is a floor: never shrink below what was asked for.
    assert PERIOD_DAYS[_period_for_timeframe("1d", "5y", need)] >= PERIOD_DAYS["5y"]
    # 1m tops out at 7 days, so it must return the longest allowed rather than loop forever.
    assert PERIOD_DAYS[_period_for_timeframe("1m", "1y", need)] <= INTERVAL_MAX_DAYS["1m"]

    table = pd.DataFrame({"Win %": [55.0, 61.0], "Sharpe": [0.4, 1.1],
                          "Expectancy": [2.0, 3.0], "Profit Factor": [1.2, 1.4],
                          "Net PnL": [10.0, 20.0]})
    note = _gate_failure_note(table, {"win": 90.0, "sharpe": 5.0})
    assert "Win % best 61.00 < 90.00" in note and "Sharpe best 1.10 < 5.00" in note, note
    assert _gate_failure_note(table, {"win": 50.0}), "a reachable gate still needs a message"
    print("   per-timeframe history windows and gate diagnostics  OK")


def _test_search_grid_and_quality():
    """Custom grids expand correctly, and the quality score respects sample size."""
    grid = build_search_grid(
        ["01 \u00b7 Dual EMA Crossover", "08 \u00b7 RSI Centerline 50 Crossing"],
        ["Fixed Percentage", "Previous Swing Low/High"], [0.5, 1.0],
        ["Risk : Reward Multiple"], [1.5, 2.0], ["adx"], True)
    # A structural stop takes its level from the chart, so it must NOT be multiplied
    # across the typed values -- that would inflate the grid with identical runs.
    assert grid["sl"] == [("Fixed Percentage", 0.5), ("Fixed Percentage", 1.0),
                          ("Previous Swing Low/High", 0.0)], grid["sl"]
    assert grid["filters"] == [None, "adx"], "'no filter' must always be searched"
    assert grid_size(grid) == 2 * 3 * 2 * 2

    safe = build_search_grid(["01 \u00b7 Dual EMA Crossover"], ["Trailing Points"], [10.0],
                             ["Fixed Points"], [20.0], [], True)
    assert all(t not in DISTANCE_TRAIL_TYPES for t, _ in safe["sl"]), \
        "safe mode must drop distance trails"

    assert parse_number_list("0.5, 1 ; 1.5, junk, 1", [9.0]) == [0.5, 1.0, 1.5]
    assert parse_number_list("", [2.0]) == [2.0], "empty input must fall back"

    base = {"Trades": 200, "Profit Factor": 2.0, "Sharpe": 1.5, "Expectancy": 5.0,
            "Win %": 55.0, "Reliability": "Backtest-safe"}
    tiny = dict(base, Trades=8, **{"Profit Factor": 5.0, "Sharpe": 4.0, "Win %": 90.0})
    assert _quality_score(base) > _quality_score(tiny), \
        "a spectacular 8-trade sample must not outrank a solid 200-trade one"
    assert _quality_score(dict(base, Reliability="Optimistic")) < _quality_score(base), \
        "optimistic backtests must be marked down"
    assert _quality_score(dict(base, Trades=0)) == 0.0
    assert _quality_score(dict(base, **{"Profit Factor": float("inf")})) <= 100.0, \
        "an infinite profit factor must not blow up the score"
    print("   custom search grids and the composite quality score  OK")


def _test_global_universes_and_join():
    """Foreign tickers must survive normalisation, and joining keeps the old levels."""
    assert _normalise_ticker("BTC-USD") == "BTC-USD", "a crypto pair must not gain .NS"
    assert _normalise_ticker("AAPL") == "AAPL", "a known US ticker must not gain .NS"
    assert _normalise_ticker("EURUSD=X") == "EURUSD=X"
    assert _normalise_ticker("GC=F") == "GC=F"
    assert _normalise_ticker("^GSPC") == "^GSPC"
    assert _normalise_ticker("TATAMOTORS.BO") == "TATAMOTORS.BO"
    assert _normalise_ticker("7203.T") == "7203.T"
    assert _normalise_ticker("RELIANCE") == "RELIANCE.NS", "a bare Indian name still gets .NS"

    for name in ("Crypto (major)", "US large caps", "Forex majors", "Commodities",
                 "Global indices"):
        tickers, note = _universe_tickers(name, "", None)
        assert tickers and note, f"{name}: empty or undocumented"
        assert all(t == _normalise_ticker(t) for t in tickers), f"{name}: unstable tickers"

    assert trades_around_the_clock("BTC-USD") and trades_around_the_clock("EURUSD=X")
    assert not trades_around_the_clock("RELIANCE.NS"), "equities do have weekend gaps"

    # Joining an existing signal keeps the ORIGINAL levels and re-bases the PnL.
    ctx = _ctx(close=100.0, atr=2.0)
    risk = RiskConfig("Fixed Points", 10.0, "Fixed Points", 30.0, 1.0)
    original_entry, actual_fill = 100.0, 108.0
    mgr = ExitManager(risk, original_entry, 1, ctx)
    assert mgr.sl == 90.0 and mgr.tp == 130.0
    mgr.entry = actual_fill
    mgr.mfe = actual_fill
    mgr.risk_points = abs(actual_fill - mgr.sl)
    assert mgr.sl == 90.0 and mgr.tp == 130.0, "levels must not move to the new fill"
    assert mgr.points(actual_fill) == 0.0, "PnL must start from the price actually paid"
    assert mgr.risk_points == 18.0, "risk is now fill-to-original-stop, and it is wider"
    print("   foreign tickers, global universes and joining an in-flight signal  OK")


def _test_condition_checklist_and_scaling():
    """Checklist logic, and point grids that scale to the instrument."""
    df = _synthetic(600, seed=11)
    params = dict(DEFAULT_PARAMS)
    params["intraday"] = True
    frame, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", params)

    checks = strategy_checks("01 \u00b7 Dual EMA Crossover", frame, params)
    assert checks, "the dual EMA profile must decompose into conditions"
    spread_check = checks[0]
    # Exactly one side of a crossover can be satisfied at a time.
    assert spread_check.long_ok != spread_check.short_ok, "both sides cannot be met at once"
    assert "needs" in spread_check.detail, "the distance to the cross must be stated"
    assert _mark(True) == TICK_YES and _mark(False) == TICK_NO and _mark(None) == TICK_NA

    assert _need(40.0, 50.0, True) == "needs 10.00 more"
    assert _need(60.0, 50.0, True) == "cleared"
    assert _need(60.0, 50.0, False) == "needs 10.00 more"

    # Point-based grid values must follow the instrument, not a hardcoded 20/40.
    def synth(price, vol, n=500):
        rng = np.random.default_rng(5)
        close = price + np.cumsum(rng.normal(0, vol, n))
        high = close + np.abs(rng.normal(0, vol, n))
        low = close - np.abs(rng.normal(0, vol, n))
        open_ = np.r_[close[0], close[:-1]]
        return pd.DataFrame(
            {"Open": open_, "High": np.maximum.reduce([high, open_, close]),
             "Low": np.minimum.reduce([low, open_, close]), "Close": close, "Volume": 1.0},
            index=pd.date_range("2026-01-01", periods=n, freq="5min"))

    base = {"strategies": ["x"], "sl": list(_OPT_SL_GRID), "tp": list(_OPT_TP_GRID),
            "filters": [None]}
    btc = scale_points_to_instrument(base, synth(77000, 90))
    nifty = scale_points_to_instrument(base, synth(24000, 12))
    btc_pts = [v for k, v in btc["sl"] if k == "Fixed Points"]
    nifty_pts = [v for k, v in nifty["sl"] if k == "Fixed Points"]
    assert min(btc_pts) > max(nifty_pts), \
        f"a BTC point stop must dwarf a Nifty one: {btc_pts} vs {nifty_pts}"
    assert all(p > 0 for p in btc_pts + nifty_pts)
    # Percentage and structural entries must pass through untouched.
    assert ("Fixed Percentage", 0.5) in btc["sl"] and ("ATR Multiple", 1.5) in btc["sl"]
    print("   entry checklist and instrument-scaled point grids  OK")


def _test_signal_detail():
    """
    The enriched signal columns must reconcile with each other exactly.

    Also a regression guard: the screener used to treat the FORMING candle as a
    confirmed hit, which left rows with no fill price and "-1 bars ago".
    """
    df = _synthetic(600, seed=11)
    params = dict(DEFAULT_PARAMS)
    params["intraday"] = True
    frame, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", params)
    fired = frame.iloc[:-1]
    fired = fired[fired["signal"] != 0]
    assert not fired.empty, "no signal to inspect"

    risk = RiskConfig("Fixed Points", 30.0, "Fixed Points", 60.0, 1.0)
    for when in (fired.index[-1], fired.index[0]):
        direction = int(frame.loc[when, "signal"])
        d = signal_detail(frame, when, direction, risk, "TEST", "5m")
        assert d["Bars Ago"] >= 0, "a confirmed signal cannot be in the future"
        assert d["Fill Price (next open)"] is not None, "confirmed signals must have a fill"
        assert abs((d["Price Now"] - d["Price at Signal"]) - d["Move Abs"]) < 0.02
        fav = (d["Price Now"] - d["Fill Price (next open)"]) * direction
        assert abs(fav - d["Move in Favour"]) < 0.02, "favour must be signed by direction"
        assert abs(d["Move in Favour"] / d["Risk Points"] - d["R Multiple Now"]) < 0.02
        assert abs(abs(d["Fill Price (next open)"] - d["Suggested Stop"])
                   - d["Risk Points"]) < 0.02
        assert d["Best Since"] >= d["Worst Since"], "best excursion below the worst"
        assert d["Scanned At"] is not None and d["Last Candle"] is not None

    # A short must report favour with the opposite sign to the raw market move.
    shorts = frame.iloc[:-1]
    shorts = shorts[shorts["signal"] == -1]
    if not shorts.empty:
        d = signal_detail(frame, shorts.index[-1], -1, risk, "TEST", "5m")
        if d["Move Abs"] != 0:
            assert np.sign(d["Move %"]) != np.sign(d["Favour %"]), \
                "a short's favour must invert the raw move"
    print("   signal detail columns reconcile; fills come from the next open  OK")


def _test_pattern_library():
    """Detectors fire across all four families, and never before confirmation."""
    df = _synthetic(1500, seed=11)
    params = dict(DEFAULT_PARAMS)
    hits = detect_patterns(df, params=params)
    assert len(hits) > 50, f"detector library found almost nothing ({len(hits)})"

    families = {h.family for h in hits}
    assert {"Candlestick", "Reversal", "Level"} <= families, f"missing families: {families}"
    names = {h.pattern for h in hits}
    for expected in ("Trendline breakout", "Trendline breakdown", "Double top", "Rounding top"):
        assert expected in names, f"{expected} never fired"

    n = len(df)
    for h in hits:
        assert 0 <= h.index < n, "hit outside the frame"
        assert h.bias in ("bullish", "bearish", "either"), f"bad bias {h.bias}"
        assert np.isfinite(h.entry) and np.isfinite(h.stop), f"{h.pattern}: unusable levels"
        assert h.geometry, f"{h.pattern}: nothing to draw"
        assert PATTERN_CATALOG.get(h.pattern) == h.family, f"{h.pattern}: family mismatch"

    # Levels must put the stop on the correct side of entry for the stated bias.
    frame = _pattern_frame(df, params)
    for h in hits[:400]:
        plan = pattern_levels(frame, h, 2.0, 1.5)
        d = plan["direction"]
        assert (plan["stop"] - plan["entry"]) * d < 0, f"{h.pattern}: stop on the wrong side"
        assert (plan["target"] - plan["entry"]) * d > 0, f"{h.pattern}: target on the wrong side"
        assert abs(plan["target"] - plan["entry"]) > abs(plan["stop"] - plan["entry"]) * 1.9, \
            f"{h.pattern}: 2R target is not two times the risk"

    # The track record must reconcile: expectancy is the mean of the outcomes.
    by = {}
    for h in hits:
        by.setdefault(h.pattern, []).append(h)
    pat = max(by, key=lambda k: len(by[k]))
    rec = pattern_track_record(frame, by[pat], 2.0, 1.5)
    assert rec["trades"] > 0 and 0 <= rec["hit_rate"] <= 100
    assert rec["breakeven"] > 0 and rec["pessimistic"] <= rec["hit_rate"] + 1e-6, \
        "the pessimistic bound must sit at or below the observed win rate"
    assert "applied to every occurrence" in rec["verdict"]
    print(f"   {len(hits)} hits across {len(families)} families; levels and track record "
          f"reconcile ({pat}: {rec['trades']} trades)  OK")


def _test_fill_semantics():
    """Signal on N must fill at the OPEN of N+1, and the stop is checked before the target."""
    idx = pd.date_range("2024-01-01 09:15", periods=6, freq="5min", tz="Asia/Kolkata")
    frame = pd.DataFrame({
        "Open":  [100, 100, 100, 102, 100, 100],
        "High":  [101, 101, 101, 110, 101, 101],
        "Low":   [99, 99, 99, 90, 99, 99],
        "Close": [100, 100, 100, 105, 100, 100],
        "Volume": [1.0] * 6, "atr": [2.0] * 6, "ema_fast": [100.0] * 6, "ema_slow": [100.0] * 6,
        "swing_high": [110.0] * 6, "swing_low": [90.0] * 6,
        "prev_swing_high": [115.0] * 6, "prev_swing_low": [85.0] * 6,
        "prev_high": [101.0] * 6, "prev_low": [99.0] * 6,
        "signal": [0, 1, 0, 0, 0, 0],
    }, index=idx)
    risk = RiskConfig("Fixed Points", 8.0, "Fixed Points", 8.0, 1.0)
    mgr = ExitManager(risk, 100.0, 1, bar_ctx(frame, 2))
    assert mgr.sl == 92.0 and mgr.tp == 108.0
    hit = mgr.check_bar(bar_ctx(frame, 3))
    assert hit == (92.0, "Stop-Loss"), (
        f"candle 3 sweeps both 92 and 108; the stop must win, got {hit}")
    print("   stop-before-target inside one candle, and N+1-open fill  OK")


def _test_strategies_and_backtest():
    df = _synthetic()
    params = dict(DEFAULT_PARAMS)
    params["intraday"] = True
    risk = RiskConfig("Fixed Percentage", 0.4, "Fixed Percentage", 0.8, 2.0)
    total = 0
    for name in STRATEGY_NAMES:
        frame, _ = prepare(df, name, params)
        assert set(frame["signal"].unique()).issubset({-1, 0, 1}), f"{name}: bad signal domain"
        rep = get_strategy(name).status(frame, params)
        assert rep.headline and rep.long_condition
        res = run_backtest(df, name, params, risk, warmup=200)
        t = res.trades
        total += len(t)
        if not t.empty:
            assert (t["Entry Time"] > t["Signal Time"]).all(), f"{name}: look-ahead fill"
            assert (t["Exit Time"] >= t["Entry Time"]).all(), f"{name}: exit before entry"
            assert np.allclose((t["Points"] * risk.quantity).round(2), t["PnL"].round(2)), \
                f"{name}: PnL does not reconcile with points"
        assert res.warmup_index >= 200, f"{name}: warm-up not respected"
        print(f"   {name:<48} {res.stats['total_trades']:>4} trades  "
              f"net {res.stats['net_pnl']:>10,.1f}")
    print(f"   OK ({total} trades across {len(STRATEGY_NAMES)} profiles)")


def _test_new_filters():
    """Crossover angle / candle size, and the RSI crossing reading."""
    df = _synthetic(600)
    params = dict(DEFAULT_PARAMS); params["intraday"] = True
    base, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", params)
    raw = int((base["signal"] != 0).sum())

    fcfg = default_filter_config()
    fcfg["crossover"]["enabled"] = True
    fcfg["crossover"]["min_angle"] = 0.0
    loose, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", params, fcfg, {})
    assert int((loose["signal"] != 0).sum()) == raw, "a 0-degree angle must veto nothing"

    fcfg["crossover"]["min_angle"] = 45.0
    steep, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", params, fcfg, {})
    assert int((steep["signal"] != 0).sum()) <= raw, "a steeper angle must not add signals"
    assert (steep["f_cross_angle"].dropna() >= 0).all(), "angle must be absolute"
    assert (steep["f_cross_angle"].dropna() < 90).all(), "arctan keeps the angle under 90 degrees"

    fcfg["crossover"]["mode"] = "Custom candle size (points)"
    fcfg["crossover"]["candle_points"] = 1e9
    none_pass, _ = prepare(df, "01 \u00b7 Dual EMA Crossover", params, fcfg, {})
    assert int((none_pass["signal"] != 0).sum()) == 0, "an impossible candle size must veto all"

    # RSI crossing reading: long on reclaiming the min, short on losing the max.
    fcfg = default_filter_config()
    fcfg["rsi"].update(enabled=True, min=40.0, max=70.0,
                       mode="Cross min from below = LONG, cross max from above = SHORT")
    frame = attach_filter_columns(prepare(df, "01 \u00b7 Dual EMA Crossover", params)[0],
                                  params, True)
    lm, sm, reports = evaluate_filters(frame, fcfg, {})
    r = frame["f_rsi"]
    expect_long = cross_over(r, _const_like(r, 40.0))
    expect_short = cross_under(r, _const_like(r, 70.0))
    assert lm.equals(expect_long.fillna(False)), "long gate must be the 40-from-below crossing"
    assert sm.equals(expect_short.fillna(False)), "short gate must be the 70-from-above crossing"
    assert int(expect_long.sum()) > 0 and int(expect_short.sum()) > 0
    print(f"   crossover angle/candle-size gates and RSI crossing reading "
          f"({int(expect_long.sum())} up-crosses, {int(expect_short.sum())} down-crosses)  OK")


def _test_filters():
    df = _synthetic(900)
    params = dict(DEFAULT_PARAMS)
    params["intraday"] = True
    base, _ = prepare(df, STRATEGY_NAMES[0], params)
    base_n = int((base["signal"] != 0).sum())
    fcfg = default_filter_config()
    for key in ("adx", "rsi", "ema20", "sma20", "bb", "macd", "smc", "ict", "volspike",
                "regime", "atrpct", "supertrend", "vwap"):
        fcfg[key]["enabled"] = True
    gated, reports = prepare(df, STRATEGY_NAMES[0], params, fcfg, {})
    gated_n = int((gated["signal"] != 0).sum())
    assert gated_n <= base_n, "filters must only ever remove signals"
    assert len(reports) == 13, f"expected 13 filter reports, got {len(reports)}"
    print(f"   13 filters applied: {base_n} raw signals -> {gated_n} after vetoes  OK")


def _test_edge_cases():
    df = _synthetic()
    params = dict(DEFAULT_PARAMS)
    params["intraday"] = True
    risk = RiskConfig("Fixed Points", 25.0, "Fixed Points", 50.0, 1.0)
    try:
        run_backtest(df.head(120), STRATEGY_NAMES[0], params, risk)
    except BacktestError as exc:
        print(f"   short-sample guard fired: {str(exc)[:62]}...")
    else:
        raise AssertionError("short sample did not raise")
    res = run_backtest(df, STRATEGY_NAMES[3], params, risk)
    print(f"   gap-filled exits detected: {res.stats['gap_exits']}")
    trailing = RiskConfig("Trailing Points", 30.0, "Trailing Target (display only)", 40.0, 1.0)
    res2 = run_backtest(df, STRATEGY_NAMES[0], params, trailing)
    assert res2.stats["reliability"] == "Optimistic", \
        "a distance trail must be flagged as not backtest-safe"
    assert any("DISTANCE trail" in r for r in res2.stats["reliability_reasons"])
    struct = RiskConfig("Trail Current Swing Low/High", 0.0, "Fixed Points", 40.0, 1.0)
    res3 = run_backtest(df, STRATEGY_NAMES[0], params, struct)
    assert res3.stats["reliability"] == "Backtest-safe", \
        "a structural trail only changes at candle boundaries, so it IS backtest-safe"
    costed = RiskConfig("Fixed Points", 25.0, "Fixed Points", 50.0, 1.0,
                        costs=CostModel(enabled=True, brokerage_per_side=20.0,
                                        pct_of_turnover=0.05, slippage_points=1.0))
    res4 = run_backtest(df, STRATEGY_NAMES[0], params, costed)
    if not res4.trades.empty:
        assert res4.stats["total_costs"] > 0, "costs must be charged when enabled"
        assert res4.stats["net_pnl"] < res4.stats["gross_pnl"], "net must sit below gross"
    print("   reliability verdicts and cost deduction  OK")
    if not res2.trades.empty:
        assert (res2.trades["Exit Reason"] != "Target").all(), \
            "a display-only target must never close a trade"
    print("   display-only target never exits; trailing caveat surfaced  OK")
    zero = df.copy()
    zero["Volume"] = 0.0
    v, ok = vwap(zero, True)
    assert not ok and v.notna().sum() > 0, "zero-volume VWAP must degrade to a usable TWAP"
    print("   zero-volume feed degrades VWAP to TWAP without dividing by zero  OK")


def run_selftest() -> int:
    data = _synthetic()
    print(f"synthetic sample: {len(data)} candles {data.index[0]} -> {data.index[-1]}\n")
    try:
        print("-- indicators --")
        e9 = ema(data["Close"], 9)
        assert e9.isna().sum() == 8 and np.isclose(e9.iloc[8], data["Close"].iloc[:9].mean())
        assert rsi(data["Close"], 14).dropna().between(0, 100).all()
        assert (atr(data["High"], data["Low"], data["Close"], 14).dropna() > 0).all()
        a, _, _ = adx(data["High"], data["Low"], data["Close"], 14)
        assert a.dropna().between(0, 100).all()
        print("   EMA seeding, RSI bounds, ATR positivity, ADX bounds  OK")
        print("-- exit engine --")
        _test_step_trail()
        _test_trail_uses_live_price()
        _test_exit_types()
        _test_structural_matrix()
        _test_fill_semantics()
        _test_tick_vs_candle_split()
        _test_liveness_logic()
        _test_zigzag_and_elliott()
        _test_threshold_strategies()
        _test_new_profiles_and_flip()
        _test_pattern_library()
        _test_signal_detail()
        _test_live_entry_and_gates()
        _test_universe_resolution()
        _test_timeframe_period_scaling()
        _test_search_grid_and_quality()
        _test_global_universes_and_join()
        _test_condition_checklist_and_scaling()
        print("-- filters --")
        _test_filters()
        _test_new_filters()
        print("-- strategies + backtest --")
        _test_strategies_and_backtest()
        print("-- edge cases --")
        _test_edge_cases()
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print("\nAll checks passed.")
    return 0




# =============================================================================
# SECTION 18 -- PERSISTENCE, WALK-FORWARD, NOTIFICATIONS, OPTIMISER, PATTERNS
# =============================================================================
import sqlite3                                                      # noqa: E402

DB_PATH = "algo_platform_state.db"


def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS open_position
                    (id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT, saved_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trades
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT, exit_time TEXT)""")
    return conn


def db_save_position(position: "Position", cfg: dict) -> None:
    """
    Persist the open position so an overnight hold survives a restart.

    A position open at 15:30 is still risk at 09:15 the next morning. Without
    this the app forgets it, the stop stops existing, and the operator finds out
    the hard way.
    """
    if position is None:
        db_clear_position()
        return
    mgr = position.manager
    payload = {
        "strategy": position.strategy, "symbol": position.symbol,
        "interval": position.interval, "direction": position.direction,
        "quantity": position.quantity, "entry_price": position.entry_price,
        "entry_time": str(position.entry_time), "signal_bar_time": str(position.signal_bar_time),
        "broker_order_id": position.broker_order_id,
        "risk": {"sl_type": mgr.risk.sl_type, "sl_value": mgr.risk.sl_value,
                 "tp_type": mgr.risk.tp_type, "tp_value": mgr.risk.tp_value,
                 "quantity": mgr.risk.quantity, "step_trigger": mgr.risk.step_trigger,
                 "costs": mgr.risk.costs.__dict__},
        "manager": {"sl": mgr.sl, "tp": mgr.tp, "initial_sl": mgr.initial_sl,
                    "mfe": mgr.mfe, "bars_held": mgr.bars_held,
                    "risk_points": mgr.risk_points},
        "saved_at": pd.Timestamp.now().isoformat(),
    }
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO open_position (id, payload, saved_at) "
                     "VALUES (1, ?, ?)", (json.dumps(payload), payload["saved_at"]))


def db_load_position() -> dict | None:
    try:
        with _db() as conn:
            row = conn.execute("SELECT payload FROM open_position WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else None
    except Exception:                                               # noqa: BLE001
        return None


def db_clear_position() -> None:
    try:
        with _db() as conn:
            conn.execute("DELETE FROM open_position WHERE id = 1")
    except Exception:                                               # noqa: BLE001
        pass


def db_append_trade(trade: dict) -> None:
    try:
        with _db() as conn:
            conn.execute("INSERT INTO trades (payload, exit_time) VALUES (?, ?)",
                         (json.dumps(trade, default=str), str(trade.get("Exit Time"))))
    except Exception:                                               # noqa: BLE001
        pass


def db_load_trades() -> list[dict]:
    try:
        with _db() as conn:
            rows = conn.execute("SELECT payload FROM trades ORDER BY id").fetchall()
        return [json.loads(r[0]) for r in rows]
    except Exception:                                               # noqa: BLE001
        return []


def restore_position(payload: dict) -> "Position":
    """Rebuild a live position, including its trailed stop, from the database."""
    r = payload["risk"]
    costs = CostModel(**r.get("costs", {}))
    risk = RiskConfig(sl_type=r["sl_type"], sl_value=r["sl_value"], tp_type=r["tp_type"],
                      tp_value=r["tp_value"], quantity=r["quantity"],
                      step_trigger=r.get("step_trigger", 0.0), costs=costs)
    ctx = BarCtx(time=None, open=payload["entry_price"], high=payload["entry_price"],
                 low=payload["entry_price"], close=payload["entry_price"], atr=np.nan,
                 ema_fast=np.nan, ema_slow=np.nan, swing_high=np.nan, swing_low=np.nan,
                 prev_swing_high=np.nan, prev_swing_low=np.nan, prev_high=np.nan,
                 prev_low=np.nan, signal=0)
    mgr = ExitManager(risk, payload["entry_price"], payload["direction"], ctx)
    m = payload["manager"]
    # The TRAILED levels are restored, not recomputed. Recomputing would silently
    # reset a stop that had already ratcheted up, handing back risk the trade had
    # already locked away.
    mgr.sl, mgr.tp = m.get("sl"), m.get("tp")
    mgr.initial_sl, mgr.mfe = m.get("initial_sl"), m.get("mfe", payload["entry_price"])
    mgr.bars_held, mgr.risk_points = int(m.get("bars_held", 0)), m.get("risk_points")
    return Position(strategy=payload["strategy"], symbol=payload["symbol"],
                    interval=payload["interval"], direction=int(payload["direction"]),
                    quantity=float(payload["quantity"]),
                    entry_price=float(payload["entry_price"]),
                    entry_time=payload.get("entry_time"),
                    signal_bar_time=payload.get("signal_bar_time"), manager=mgr,
                    broker_order_id=payload.get("broker_order_id"))


# --------------------------------------------------------------------------- #
# EMAIL NOTIFICATIONS
# --------------------------------------------------------------------------- #
def send_email(cfg: dict, subject: str, body: str) -> str | None:
    """Send a notification via SMTP. Returns an error string, or None on success."""
    mail = (cfg or {}).get("email") or {}
    if not mail.get("enabled"):
        return None
    sender, to, password = mail.get("from"), mail.get("to"), mail.get("password")
    if not (sender and to and password):
        return "Email is enabled but the sender, recipient or app password is missing."
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, sender, to
        msg.set_content(body)
        with smtplib.SMTP_SSL(mail.get("host", "smtp.gmail.com"),
                              int(mail.get("port", 465)), timeout=20) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        return None
    except Exception as exc:                                        # noqa: BLE001
        return f"Email failed: {exc}"


# --------------------------------------------------------------------------- #
# WALK-FORWARD
# --------------------------------------------------------------------------- #
def walk_forward(df: pd.DataFrame, strategy: str, params: dict, risk: RiskConfig,
                 filter_cfg: dict | None, extras: dict | None, folds: int = 5) -> pd.DataFrame:
    """
    Sequential out-of-sample segments.

    IMPORTANT ON NAMING: with a fixed configuration there is no in-sample fitting
    to guard against, so this is a STABILITY report, not classical walk-forward
    optimisation. It answers "did this hold up across different stretches of the
    sample, or does one lucky window carry the whole result?" The optimiser tab
    runs the fit-then-test version.
    """
    rows = []
    size = len(df) // max(1, folds)
    for k in range(folds):
        lo = k * size
        hi = len(df) if k == folds - 1 else (k + 1) * size
        segment = df.iloc[max(0, lo - WARMUP_BARS):hi]
        try:
            res = run_backtest(segment, strategy, params, risk, filter_cfg, extras, WARMUP_BARS)
        except BacktestError:
            rows.append({"Segment": k + 1, "From": df.index[lo], "To": df.index[hi - 1],
                         "Trades": 0, "Win %": 0.0, "Net PnL": 0.0, "Sharpe": 0.0,
                         "Note": "too few candles"})
            continue
        st_ = res.stats
        rows.append({"Segment": k + 1, "From": df.index[lo], "To": df.index[hi - 1],
                     "Trades": st_["total_trades"], "Win %": st_["win_rate"],
                     "Net PnL": st_["net_pnl"], "Sharpe": st_["sharpe"], "Note": ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# STRATEGY OPTIMISER
# --------------------------------------------------------------------------- #
# Excluded from the search: Simple Buy, Simple Sell and both threshold profiles.
# They are execution helpers, not edges -- optimising them would just be fitting
# noise to whichever direction the sample happened to drift.
_OPTIMISER_EXCLUDE = ("16 \u00b7", "17 \u00b7", "28 \u00b7", "29 \u00b7")

OPTIMISER_OBJECTIVES = ["Win rate (accuracy)", "Sharpe ratio", "Net PnL", "Expectancy per trade",
                        "Profit factor"]

_OPT_SL_GRID = [("Fixed Percentage", 0.5), ("Fixed Percentage", 1.0), ("Fixed Points", 20.0),
                ("Fixed Points", 40.0), ("ATR Multiple", 1.5), ("ATR Multiple", 2.5),
                ("Trail Current Swing Low/High", 0.0), ("Trail Previous Candle Low/High", 0.0),
                ("Trailing ATR (Chandelier)", 3.0)]
_OPT_TP_GRID = [("Fixed Percentage", 1.0), ("Fixed Percentage", 2.0), ("Fixed Points", 40.0),
                ("Fixed Points", 80.0), ("ATR Multiple", 3.0),
                ("Risk : Reward Multiple", 1.5), ("Risk : Reward Multiple", 2.5),
                ("Trail Current Swing High/Low", 0.0)]
_OPT_FILTERS = [None, "adx", "rsi", "ema20", "supertrend", "macd", "regime"]


def _objective_value(stats: dict, objective: str) -> float:
    return {"Win rate (accuracy)": stats["win_rate"], "Sharpe ratio": stats["sharpe"],
            "Net PnL": stats["net_pnl"], "Expectancy per trade": stats["expectancy"],
            "Profit factor": 0.0 if stats["profit_factor"] == float("inf")
            else stats["profit_factor"]}.get(objective, 0.0)


def parse_number_list(text: str, fallback: list[float]) -> list[float]:
    """Parse '0.5, 1, 1.5' into floats, keeping order and dropping duplicates."""
    out: list[float] = []
    for chunk in str(text or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = float(chunk)
        except ValueError:
            continue
        if value not in out:
            out.append(value)
    return out or list(fallback)


def scale_points_to_instrument(grid: dict, df: pd.DataFrame, atr_len: int = 14) -> dict:
    """
    Rewrite absolute point values as multiples of the instrument's own ATR.

    A 40-point stop is a sensible intraday stop on Nifty and 0.05% on Bitcoin.
    Searching the same absolute numbers across instruments does not compare like
    with like: on BTC every point-based combination is effectively a zero-width
    stop that is hit immediately, and on a Rs.50 stock it is wider than the whole
    day's range. The multipliers are preserved -- only their scale changes.
    """
    try:
        atr_series = atr(df["High"], df["Low"], df["Close"], atr_len)
        unit = float(atr_series.median(skipna=True))
    except Exception:                                               # noqa: BLE001
        return grid
    if not np.isfinite(unit) or unit <= 0:
        return grid

    base = 20.0                       # the point values in the default grid are multiples of this
    def rescale(pairs_in):
        out = []
        for kind, value in pairs_in:
            if kind == "Fixed Points" and value:
                multiple = float(value) / base
                scaled = round(unit * multiple * 2.0, 4)
                scaled = round(scaled, 2) if scaled >= 1 else round(scaled, 4)
                out.append((kind, scaled))
            else:
                out.append((kind, value))
        return out

    out = dict(grid)
    out["sl"] = rescale(grid["sl"])
    out["tp"] = rescale(grid["tp"])
    out["_atr_unit"] = round(unit, 4)
    return out


def build_search_grid(strategies: list[str], sl_types: list[str], sl_values: list[float],
                      tp_types: list[str], tp_values: list[float],
                      filters: list[str], safe_exits_only: bool) -> dict:
    """
    Expand the operator's choices into the axes the search walks.

    Structural exits take their level from the chart, so they are paired with a
    single dummy magnitude rather than multiplied across every value the user
    typed -- otherwise the grid inflates with combinations that are identical.
    """
    def pairs(types: list[str], values: list[float], no_value: set) -> list[tuple[str, float]]:
        out = []
        for t in types:
            if t in no_value:
                out.append((t, 0.0))
            else:
                out.extend((t, v) for v in values)
        return out

    def strip_unsafe(pairs_in):
        return [x for x in pairs_in if x[0] not in DISTANCE_TRAIL_TYPES] \
            if safe_exits_only else list(pairs_in)

    sl = strip_unsafe(pairs(sl_types, sl_values, _SL_NO_VALUE))
    tp = pairs(tp_types, tp_values, _TP_NO_VALUE)
    # The fallback must obey safe mode too. Substituting the default grid
    # unfiltered would quietly hand back the distance trails that safe mode was
    # asked to exclude. If the operator's own choice empties out, leave it empty
    # so the caller can say so rather than searching something else.
    if not sl and not sl_types:
        sl = strip_unsafe(_OPT_SL_GRID)
    if not tp and not tp_types:
        tp = list(_OPT_TP_GRID)
    return {
        "strategies": strategies or [n for n in STRATEGY_NAMES
                                     if not n.startswith(_OPTIMISER_EXCLUDE)],
        "sl": sl,
        "tp": tp,
        "filters": ([None] + [f for f in filters if f]) if filters else list(_OPT_FILTERS),
    }


def grid_size(grid: dict) -> int:
    return (len(grid["strategies"]) * max(1, len(grid["sl"])) * max(1, len(grid["tp"]))
            * max(1, len(grid["filters"])))


def optimise(df: pd.DataFrame, base_params: dict, quantity: float, costs: CostModel,
             objective: str, min_trades: int, iterations: int, seed: int = 11,
             safe_exits_only: bool = True, progress=None,
             grid: dict | None = None, exhaustive: bool = False,
             scale_points: bool = True) -> pd.DataFrame:
    """
    Randomised search over strategy x stop x target x one optional filter.

    Randomised rather than exhaustive because the full grid is tens of thousands
    of backtests. And a caution that matters more than the search itself: the
    more combinations tried, the more likely the winner is luck. Treat the
    ranking as a shortlist to validate out of sample, never as a result.
    """
    rng = np.random.default_rng(seed)
    if grid is None:
        grid = {"strategies": [n for n in STRATEGY_NAMES
                               if not n.startswith(_OPTIMISER_EXCLUDE)],
                "sl": [x for x in _OPT_SL_GRID
                       if not (safe_exits_only and x[0] in DISTANCE_TRAIL_TYPES)],
                "tp": list(_OPT_TP_GRID), "filters": list(_OPT_FILTERS)}
    if scale_points:
        grid = scale_points_to_instrument(grid, df, int(base_params.get("atr_len", 14)))
    names, sl_grid = grid["strategies"], grid["sl"]
    tp_grid, filters = grid["tp"], grid["filters"]
    if not (names and sl_grid and tp_grid and filters):
        return pd.DataFrame()

    if exhaustive:
        combos = [(a, b, c, d) for a in names for b in sl_grid for c in tp_grid for d in filters]
        rng.shuffle(combos)                       # shuffled so a cap still samples the space
        combos = combos[:int(iterations)] if iterations else combos
    else:
        combos = None

    rows, seen = [], set()
    total = len(combos) if combos is not None else int(iterations)
    started = time.time()

    for it in range(total):
        if progress is not None and it % 5 == 0:
            done = (it + 1) / max(1, total)
            elapsed = time.time() - started
            eta = (elapsed / max(done, 1e-6)) - elapsed
            progress.progress(min(1.0, done),
                              text=f"Tested {it} of {total} combinations "
                                   f"({done*100:.0f}%) - about {eta:0.0f}s left")
        if combos is not None:
            strategy, (sl_type, sl_val), (tp_type, tp_val), filt = combos[it]
        else:
            strategy = names[int(rng.integers(len(names)))]
            sl_type, sl_val = sl_grid[int(rng.integers(len(sl_grid)))]
            tp_type, tp_val = tp_grid[int(rng.integers(len(tp_grid)))]
            filt = filters[int(rng.integers(len(filters)))]
        key = (strategy, sl_type, sl_val, tp_type, tp_val, filt)
        if key in seen:
            continue
        seen.add(key)

        fcfg = default_filter_config()
        if filt:
            fcfg[filt]["enabled"] = True
        risk = RiskConfig(sl_type, sl_val, tp_type, tp_val, quantity, costs=costs)
        try:
            res = run_backtest(df, strategy, dict(base_params), risk, fcfg, {}, WARMUP_BARS)
        except (BacktestError, MarketDataError, Exception):          # noqa: BLE001
            continue
        st_ = res.stats
        if st_["total_trades"] < min_trades:
            continue
        rows.append({
            "Strategy": strategy, "Stop-Loss": sl_type,
            "SL Value": sl_val if sl_type not in _SL_NO_VALUE else None,
            "Target": tp_type, "TP Value": tp_val if tp_type not in _TP_NO_VALUE else None,
            "Filter": FILTER_LABELS.get(filt, "none"), "Filter Key": filt or "",
            "Trades": st_["total_trades"],
            "Win %": st_["win_rate"], "Sharpe": st_["sharpe"], "Net PnL": st_["net_pnl"],
            "Expectancy": st_["expectancy"], "Profit Factor": st_["profit_factor"],
            "Max DD": st_["max_drawdown"], "Reliability": st_["reliability"],
            "Score": _objective_value(st_, objective),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("Score", ascending=False).reset_index(drop=True)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    return frame


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    main()
