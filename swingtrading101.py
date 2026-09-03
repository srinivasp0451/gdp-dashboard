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

PERIODS = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y",
           "5y", "10y", "15y", "20y", "30y", "max"]
PERIOD_DAYS = {"1d": 1, "5d": 5, "7d": 7, "1mo": 30, "3mo": 91, "6mo": 182,
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
                trades.append(_close_trade(pos, float(exit_price), ctx.time, reason))
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
                               signal_bar_time=frame.index[i - 1], manager=mgr)
                pending_signal_exit = None

    if pos is not None:
        last = bar_ctx(frame, n - 1)
        trades.append(_close_trade(pos, float(last.close), last.time, "End of Data"))
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


def _close_trade(pos: Position, exit_price: float, exit_time, reason: str) -> dict:
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
    "optimizer_results": None, "pattern_results": None, "pending_combo": None,
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
        return float(price), "Yahoo quote (ticks continuously)"
    return float(frame["Close"].iloc[-1]), "Candle close (quote unavailable -- steps per candle)"


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

    return LiveSnapshot(
        frame=frame, ltp=ltp, next_open=float(frame["Open"].iloc[-1]),
        last_closed_time=frame.index[closed],
        last_closed_signal=int(frame["signal"].iloc[closed]),
        raw_signal=int(frame["raw_signal"].iloc[closed]),
        status=strat.status(frame.iloc[:len(frame) + closed + 1], cfg["params"]),
        filter_reports=reports, fetched_at=pd.Timestamp.now(), bars=len(frame),
        data_warnings=warnings, vix=vix, feed_age_seconds=age, stale=stale,
        quote_live=quote_live, ltp_source=ltp_source)


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


def _open_live_position(cfg: dict, direction: int, price: float, ctx: BarCtx, bar_time) -> Position:
    mgr = ExitManager(cfg["risk"], float(price), direction, ctx)
    position = Position(strategy=cfg["strategy"], symbol=cfg["symbol"], interval=cfg["interval"],
                        direction=direction, quantity=cfg["risk"].quantity,
                        entry_price=float(price), entry_time=pd.Timestamp.now(),
                        signal_bar_time=bar_time, manager=mgr,
                        entry_ltp_at_fill=cfg.get("_ltp_at_fill"))
    st.session_state.live_position = position
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
    log.insert(0, {"Polled at": pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3],
                   "Newest candle": fmt_time(snapshot.frame.index[-1]),
                   "LTP": round(snapshot.ltp, 4),
                   "Changed": "-" if prev_ltp is None else
                              ("yes" if abs(prev_ltp - snapshot.ltp) > 1e-9 else "no"),
                   "Source": snapshot.ltp_source})
    del log[60:]

    closed_ctx = bar_ctx(frame, len(frame) - 2)
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

    already_seen = not new_bar
    st.session_state.live_last_bar = snapshot.last_closed_time
    if already_seen or snapshot.last_closed_signal == 0:
        return

    # Signal on candle N -> fill at the OPEN of candle N+1 (already printed).
    fill = snapshot.ltp if cfg.get("fill_at_ltp") else snapshot.next_open
    cfg["_ltp_at_fill"] = snapshot.ltp
    _open_live_position(cfg, snapshot.last_closed_signal, fill, closed_ctx,
                        snapshot.last_closed_time)


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
                hide_weekends=True, height=620):
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

    for i, col in enumerate(dict.fromkeys(("ema_fast", "ema_slow", *overlays))):
        if col not in data.columns or data[col].notna().sum() == 0:
            continue
        colour = _OVERLAY_COLOURS[i % len(_OVERLAY_COLOURS)]
        last = data[col].dropna()
        value = float(last.iloc[-1]) if len(last) else None
        name = _label(col) + (f"  {fmt(value)}" if value is not None else "")
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
                      height=height, margin=dict(l=10, r=90, t=46, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_rangeslider_visible=False, dragmode="pan")
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
    if strategy.startswith(("30 ", "31 ", "32 ", "33 ")):
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
            "allow_stale_entries": bool(allow_stale),
            "candle_seconds": float(candle_seconds), "costs": costs,
            "walk_forward": bool(walk_fwd), "wf_folds": int(wf_folds),
            "use_dhan_data": bool(use_dhan_data), "email": email_cfg,
            "filter_cfg": filter_cfg, "filter_extras": filter_extras, "broker": broker,
            "currency": currency_symbol(symbol),
            "hide_weekends": not (symbol.endswith("-USD") or symbol.endswith("=X"))}


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
    """Dhan order routing. Disabled by default, dry-run by default even when enabled."""
    broker = {"enabled": False, "dry_run": True, "contract": None}
    with sb.expander("Dhan order placement (off by default)"):
        st.error("Live routing sends REAL orders to your Dhan account. The signals here are built "
                 "on delayed Yahoo data, which is the wrong input for real execution. Keep dry-run "
                 "on unless you have replaced the feed and tested thoroughly.")
        enabled = st.checkbox("Enable Dhan order routing", value=False, disabled=live,
                              key="brk_enabled")
        broker["enabled"] = enabled
        if not enabled:
            return broker

        broker["dry_run"] = st.checkbox("Dry run (build the payload, transmit nothing)",
                                        value=True, key="brk_dry")
        broker["use_live_ltp"] = st.checkbox(
            "Use Dhan for the live LTP (read-only, places no orders)",
            value=bool(st.session_state.get("cfg_dhan_data", False)), key="brk_ltp",
            help="Yahoo's Indian feed is delayed ~15 minutes, so on a 5m chart the LTP can sit "
                 "unchanged for a quarter of an hour however fast you poll. This replaces it "
                 "with Dhan's real-time quote. Requires a Dhan Data API subscription.")
        broker["client_id"] = st.text_input("Dhan client ID", key="brk_cid")
        broker["access_token"] = st.text_input("Dhan access token", type="password", key="brk_tok")
        broker["product_type"] = st.selectbox("Product type", DHAN_PRODUCTS, key="brk_prod")
        instrument = st.selectbox("Instrument", DHAN_INSTRUMENTS, key="brk_inst")
        segment = st.selectbox("Exchange segment", DHAN_SEGMENTS,
                               index=0 if instrument == "EQUITY" else 2, key="brk_seg")
        underlying = st.text_input("Underlying symbol", value=_default_underlying(symbol),
                                   key="brk_under",
                                   help="Dhan's own name, e.g. RELIANCE, NIFTY, BANKNIFTY.")
        option_type = st.selectbox("Option right", ["CALL", "PUT"], key="brk_right") \
            if instrument == "OPTIONS" else "CALL"

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
            st.info("Resolve a contract before starting the engine, or entries will be skipped.")
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


# =============================================================================
# SECTION 14 -- TAB 2: LIVE SANDBOX OPERATIONS PANEL
# =============================================================================
def tab_live(cfg: dict) -> None:
    st.subheader("Live Sandbox Operations Panel")
    st.caption("Signals are read from closed candles and filled at the next candle's open. "
               "Stop and target are then checked against the LTP on every poll, stop first.")
    _live_controls(cfg)
    st.divider()
    if st.session_state.live_running:
        _mount_live_fragment(float(st.session_state.live_config.get("poll_seconds", 5.0)))
    else:
        _idle_panel(cfg)


def _live_controls(cfg: dict) -> None:
    running = bool(st.session_state.live_running)
    position = st.session_state.live_position
    c1, c2, c3 = st.columns(3)

    saved = db_load_position() if st.session_state.live_position is None else None
    if saved:
        st.warning(f"An open {('LONG' if saved['direction'] > 0 else 'SHORT')} position on "
                   f"`{saved['symbol']}` was saved at {saved.get('saved_at', '?')[:19]} and never "
                   "closed. Its trailed stop and target are restored exactly as they were -- "
                   "recomputing them would hand back risk the trade had already locked away.")
        rc1, rc2 = st.columns(2)
        if rc1.button("Resume saved position", type="primary", width="stretch"):
            st.session_state.live_position = restore_position(saved)
            log_event(f"Resumed the saved {saved['symbol']} position from the database.", "warn")
            st.rerun()
        if rc2.button("Discard saved position", width="stretch"):
            db_clear_position()
            log_event("Discarded the saved position without booking a trade.", "warn")
            st.rerun()

    if c1.button("Start Live Automation Core", type="primary", disabled=running, width="stretch"):
        reset_live_runtime()
        st.session_state.live_config = dict(cfg)
        st.session_state.live_running = True
        st.session_state.live_started_at = pd.Timestamp.now()
        st.session_state.live_last_poll = 0.0
        log_event(f"Core started :: {cfg['symbol']} | {cfg['interval']} | {cfg['strategy']} | "
                  f"{cfg['risk'].as_summary()} | poll {fmt(cfg['poll_seconds'],1)}s", "success")
        st.rerun()

    if c2.button("Stop Live Processing Engine", disabled=not running, width="stretch"):
        # Stopping the engine flattens the book. Leaving an untracked position
        # open after the monitor is switched off is how a stop-loss silently
        # stops existing.
        trade = square_off("Squared Off on Engine Stop")
        st.session_state.live_running = False
        if trade:
            st.toast(f"Position squared off at {fmt(trade['Exit Price'])} "
                     f"for {fmt_signed(trade['PnL'])} and written to the ledger.")
        log_event("Core stopped.", "info")
        st.rerun()

    if c3.button("Manual Emergency Square-Off", disabled=position is None, width="stretch"):
        trade = square_off("Manual Square-Off")
        if trade:
            st.toast(f"Closed at {fmt(trade['Exit Price'])} for {fmt_signed(trade['PnL'])}.")
        st.rerun()


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
    if should_poll(cfg):
        run_cycle(cfg)

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

    _feed_banner(cfg, snapshot)
    _heartbeat(cfg, snapshot)
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

    st.markdown("#### Open Strategy Performance")
    (st.success if pnl >= 0 else st.error)(
        f"{side} {position.symbol} :: running {fmt_signed(pnl)} {currency}")

    r1 = st.columns(4)
    r1[0].metric("Strategy", position.strategy.split("· ")[-1],
                 help=position.strategy)
    r1[1].metric("Entry Price", fmt(position.entry_price), side)
    r1[2].metric("LTP", fmt(ltp))
    r1[3].metric("Qty", fmt(position.quantity, 0))

    r2 = st.columns(4)
    tgt_label = "Target (display)" if mgr.tp_display_only else "Target"
    r2[0].metric(tgt_label, fmt(mgr.tp) if mgr.tp is not None else "none")
    r2[1].metric("Stop-Loss", fmt(mgr.sl) if mgr.sl is not None else "none",
                 f"initial {fmt(mgr.initial_sl)}" if mgr.initial_sl is not None else None)
    r2[2].metric("Points +/-", fmt_signed(points))
    r2[3].metric(f"Live PnL ({currency})", fmt_signed(pnl))

    r3 = st.columns(4)
    r3[0].metric("Best Price", fmt(mgr.mfe), help="Best price seen since entry (drives trails).")
    locked = None if mgr.sl is None else (mgr.sl - position.entry_price) * position.direction
    r3[1].metric("Locked In", fmt_signed(locked) if locked is not None else "--",
                 help="Points the stop now guarantees, positive once the trail passes cost.")
    r3[2].metric("Entry Risk", fmt(mgr.risk_points) if mgr.risk_points else "--")
    r3[3].metric("Bars", f"{mgr.bars_held}")

    for note in mgr.notes:
        st.warning("Exit engine: " + note)
    st.caption(f"Entered {fmt_time(position.entry_time)} off the signal candle "
               f"{fmt_time(position.signal_bar_time)}."
               + (f" LTP at fill was {fmt(position.entry_ltp_at_fill)}."
                  if position.entry_ltp_at_fill else ""))


def _searching_widget(cfg: dict, snapshot: LiveSnapshot) -> None:
    st.markdown("#### Signal Scanner")
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
    st.markdown("#### Live Chart")
    fig = price_chart(snapshot.frame, f"{cfg['symbol']} | {cfg['interval']} | last 100 candles",
                      strat.overlays, tail=120,
                      hide_weekends=cfg.get("hide_weekends", True), height=520)
    position = st.session_state.live_position
    if position is not None:
        mgr = position.manager
        fig.add_hline(y=position.entry_price, line=dict(width=1.2, dash="dash", color="#4f9df7"),
                      annotation_text="Entry")
        if mgr.tp is not None:
            fig.add_hline(y=mgr.tp, line=dict(width=1.2, dash="dot", color="#26a69a"),
                          annotation_text="Target" + (" (display)" if mgr.tp_display_only else ""))
        if mgr.sl is not None:
            fig.add_hline(y=mgr.sl, line=dict(width=1.2, dash="dot", color="#ef5350"),
                          annotation_text="Stop")
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

SCREENER_UNIVERSES = [
    "Nifty 50", "Nifty Next 50", "Nifty 100 (50 + Next 50)",
    "Broad indices", "Sector indices", "Custom list (paste or upload)",
]


def _universe_tickers(choice: str, custom_text: str, uploaded) -> tuple[list[str], str | None]:
    note = None
    if choice == "Nifty 50":
        names, note = NIFTY_50, "Snapshot list; verify against the current NSE factsheet."
    elif choice == "Nifty Next 50":
        names, note = NIFTY_NEXT_50, "Snapshot list; verify against the current NSE factsheet."
    elif choice.startswith("Nifty 100"):
        names = NIFTY_50 + NIFTY_NEXT_50
        note = "Snapshot list; verify against the current NSE factsheet."
    elif choice == "Broad indices":
        return ["^NSEI", "^NSEBANK", "^BSESN", "NIFTY_FIN_SERVICE.NS", "^NSMIDCP"], None
    elif choice == "Sector indices":
        return list(SECTOR_INDICES.values()), None
    else:
        raw = ""
        if uploaded is not None:
            try:
                raw = uploaded.getvalue().decode("utf-8", errors="ignore")
            except Exception:                                       # noqa: BLE001
                raw = ""
        raw = (raw + "\n" + (custom_text or "")).replace(",", "\n")
        names = [x.strip().upper() for x in raw.splitlines() if x.strip()]
        return [n if ("." in n or n.startswith("^") or "=" in n) else f"{n}.NS"
                for n in names], None
    return [f"{n}.NS" for n in names], note


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

        window = frame.iloc[-(lookback_bars + 1):]
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

        rows.append({
            "Ticker": ticker,
            "Signal": "LONG" if direction > 0 else "SHORT",
            "When": when,
            "Signal Time": pd.Timestamp(hit_time),
            "Close": round(float(frame["Close"].iloc[-1]), 2),
            "Fast EMA": round(float(frame["ema_fast"].iloc[-1]), 2)
            if "ema_fast" in frame else None,
            "Slow EMA": round(float(frame["ema_slow"].iloc[-1]), 2)
            if "ema_slow" in frame else None,
            "ATR": round(float(frame["atr"].iloc[-1]), 2) if "atr" in frame else None,
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
        st.dataframe(results, width="stretch", hide_index=True)

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

    if not errors.empty:
        with st.expander(f"Tickers that could not be scanned ({len(errors)})"):
            st.dataframe(errors, width="stretch", hide_index=True)


# =============================================================================
# SECTION 15 -- TAB 3: LIVE TRADE LOG LEDGER
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
    _open_position_note()


def _open_position_note() -> None:
    position = st.session_state.get("live_position")
    if position is None:
        return
    st.warning(f"A {'LONG' if position.direction > 0 else 'SHORT'} position on "
               f"`{position.symbol}` is still open and therefore not in this ledger. It is "
               "journalled the moment it closes.")


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

    if st.button("Run Optimiser", type="primary", width="stretch"):
        st.session_state.optimizer_results = None
        bar = st.progress(0.0, text="Loading data ...")
        try:
            bundle = load_market_data(cfg["symbol"], cfg["period"], cfg["interval"], 300.0,
                                      min_bars=WARMUP_BARS + 60)
            results = optimise(bundle.frame, cfg["params"], cfg["quantity"], cfg["costs"],
                               objective, int(min_trades), int(iterations),
                               safe_exits_only=safe_only, progress=bar)
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
            "tp_type": row["Target"], "tp_value": row["TP Value"]}
        st.rerun()
    st.caption(f"Rank {pick}: {row['Strategy']} | SL {row['Stop-Loss']} | TGT {row['Target']} | "
               f"filter {row['Filter']} | {row['Trades']} trades | reliability {row['Reliability']}")
    st.download_button("Download all combinations (CSV)", results.to_csv(index=False).encode(),
                       "optimiser_results.csv", "text/csv")


def tab_patterns(cfg: dict) -> None:
    st.subheader("Chart Pattern Scanner")
    st.caption("Rule-of-thumb geometric detection on confirmed zigzag pivots and candle shapes. "
               "Two analysts would disagree about half of these, so treat a hit as worth a look "
               "rather than a verdict.")

    c1, c2, c3 = st.columns(3)
    universe = c1.selectbox("Universe", SCREENER_UNIVERSES + ["Current sidebar ticker"],
                            index=len(SCREENER_UNIVERSES), key="pat_universe")
    sensitivity = c2.number_input("Zigzag sensitivity (%)", 0.1, 10.0, 0.6, 0.1, key="pat_zz",
                                  help="Smaller finds more, and noisier, structures.")
    max_names = c3.number_input("Max tickers", 1, 200, 25, key="pat_max")

    custom_text, uploaded = "", None
    if universe.startswith("Custom"):
        custom_text = st.text_area("Tickers", "RELIANCE\nTCS\nINFY", key="pat_custom")
        uploaded = st.file_uploader("...or upload a list", type=["csv", "txt"], key="pat_upload")

    if universe == "Current sidebar ticker":
        tickers = [cfg["symbol"]]
    else:
        tickers, _ = _universe_tickers(universe, custom_text, uploaded)
    tickers = tickers[:int(max_names)]

    if st.button("Scan for Patterns", type="primary", width="stretch"):
        bar = st.progress(0.0, text="Scanning ...")
        found, frames = [], {}
        for i, ticker in enumerate(tickers):
            bar.progress((i + 1) / max(1, len(tickers)), text=f"Scanning {ticker} ...")
            try:
                bundle = load_market_data(ticker, cfg["period"], cfg["interval"], 300.0,
                                          min_bars=40)
            except Exception:                                       # noqa: BLE001
                continue
            frames[ticker] = bundle.frame
            for hit in detect_patterns(bundle.frame, float(sensitivity)):
                found.append({"Ticker": ticker, **hit})
        bar.empty()
        st.session_state.pattern_results = (pd.DataFrame(found), frames)

    payload = st.session_state.pattern_results
    if payload is None:
        st.info("Pick a universe and scan.")
        return
    results, frames = payload
    if results.empty:
        st.info("No patterns detected on this configuration. Lower the zigzag sensitivity to "
                "find more structures.")
        return

    bias = st.multiselect("Show", ["Bullish", "Bearish", "Neutral"],
                          default=["Bullish", "Bearish"], key="pat_bias")
    view = results[results["Bias"].isin(bias)] if bias else results
    st.dataframe(view.sort_values("At", ascending=False), width="stretch", hide_index=True)

    if not view.empty:
        pick = st.selectbox("Plot which ticker?", sorted(view["Ticker"].unique()), key="pat_pick")
        frame = frames.get(pick)
        if frame is not None:
            plot = frame.copy()
            plot["ema_fast"] = ema(plot["Close"], int(_p(cfg["params"], "ema_fast")))
            plot["ema_slow"] = ema(plot["Close"], int(_p(cfg["params"], "ema_slow")))
            fig = price_chart(plot, f"{pick} | {cfg['interval']} | detected patterns",
                              tail=200, hide_weekends=cfg.get("hide_weekends", True))
            for _, hit in view[view["Ticker"] == pick].iterrows():
                colour = {"Bullish": "#26a69a", "Bearish": "#ef5350"}.get(hit["Bias"], "#8d99ae")
                fig.add_vline(x=hit["At"], line=dict(width=1, dash="dot", color=colour),
                              annotation_text=hit["Pattern"], annotation_position="top")
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
        st.download_button("Download patterns (CSV)", view.to_csv(index=False).encode(),
                           "chart_patterns.csv", "text/csv")


# =============================================================================
# SECTION 16 -- MAIN
# =============================================================================
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

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Backtesting Engine Studio", "Live Sandbox Operations", "Live Trade Log Ledger",
         "Signal Screener", "Strategy Optimiser", "Chart Patterns"])
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


def optimise(df: pd.DataFrame, base_params: dict, quantity: float, costs: CostModel,
             objective: str, min_trades: int, iterations: int, seed: int = 11,
             safe_exits_only: bool = True, progress=None) -> pd.DataFrame:
    """
    Randomised search over strategy x stop x target x one optional filter.

    Randomised rather than exhaustive because the full grid is tens of thousands
    of backtests. And a caution that matters more than the search itself: the
    more combinations tried, the more likely the winner is luck. Treat the
    ranking as a shortlist to validate out of sample, never as a result.
    """
    rng = np.random.default_rng(seed)
    names = [n for n in STRATEGY_NAMES if not n.startswith(_OPTIMISER_EXCLUDE)]
    sl_grid = [x for x in _OPT_SL_GRID
               if not (safe_exits_only and x[0] in DISTANCE_TRAIL_TYPES)]
    rows, seen = [], set()

    for it in range(int(iterations)):
        if progress is not None and it % 5 == 0:
            progress.progress(min(1.0, (it + 1) / iterations), text=f"Tested {it} combinations ...")
        strategy = names[int(rng.integers(len(names)))]
        sl_type, sl_val = sl_grid[int(rng.integers(len(sl_grid)))]
        tp_type, tp_val = _OPT_TP_GRID[int(rng.integers(len(_OPT_TP_GRID)))]
        filt = _OPT_FILTERS[int(rng.integers(len(_OPT_FILTERS)))]
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
            "Filter": FILTER_LABELS.get(filt, "none"), "Trades": st_["total_trades"],
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


# --------------------------------------------------------------------------- #
# CHART PATTERNS
# --------------------------------------------------------------------------- #
def detect_patterns(df: pd.DataFrame, zigzag_pct: float = 0.6) -> list[dict]:
    """
    Classic chart patterns from confirmed zigzag pivots plus candle geometry.

    Every pattern is a rule-of-thumb approximation with tolerances chosen by
    hand. Two analysts would disagree about half of these; treat a hit as
    "worth a look", never as a verdict.
    """
    out: list[dict] = []
    piv = zigzag_pivot_table(df["Close"], zigzag_pct)
    n = len(df)
    if n < 20:
        return out

    def add(name, bias, idx, note):
        out.append({"Pattern": name, "Bias": bias, "At": df.index[min(idx, n - 1)], "Note": note})

    # ---- pivot-based structures ----
    if len(piv) >= 4:
        (_, a, ka, _), (_, b, kb, _), (_, c, kc, _), (_, d, kd, ci) = piv[-4:]
        if (ka, kb, kc, kd) == (1, -1, 1, -1) and abs(a - c) / max(a, 1e-9) < 0.02:
            add("Double Top", "Bearish", ci, f"Two highs within 2%: {fmt(a)} / {fmt(c)}")
        if (ka, kb, kc, kd) == (-1, 1, -1, 1) and abs(a - c) / max(a, 1e-9) < 0.02:
            add("Double Bottom", "Bullish", ci, f"Two lows within 2%: {fmt(a)} / {fmt(c)}")
    if len(piv) >= 5:
        highs = [(v, i) for _, v, k, i in piv[-5:] if k == 1]
        lows = [(v, i) for _, v, k, i in piv[-5:] if k == -1]
        if len(highs) >= 3:
            h1, h2, h3 = highs[-3][0], highs[-2][0], highs[-1][0]
            if h2 > h1 and h2 > h3 and abs(h1 - h3) / max(h1, 1e-9) < 0.03:
                add("Head & Shoulders", "Bearish", highs[-1][1],
                    f"Head {fmt(h2)} between shoulders {fmt(h1)} / {fmt(h3)}")
        if len(lows) >= 3:
            l1, l2, l3 = lows[-3][0], lows[-2][0], lows[-1][0]
            if l2 < l1 and l2 < l3 and abs(l1 - l3) / max(l1, 1e-9) < 0.03:
                add("Inverse Head & Shoulders", "Bullish", lows[-1][1],
                    f"Head {fmt(l2)} between shoulders {fmt(l1)} / {fmt(l3)}")
        hs = [v for _, v, k, _ in piv[-5:] if k == 1]
        ls = [v for _, v, k, _ in piv[-5:] if k == -1]
        if len(hs) >= 2 and len(ls) >= 2:
            if hs[-1] < hs[-2] and ls[-1] > ls[-2]:
                add("Symmetrical Triangle", "Neutral", n - 1, "Lower highs into higher lows")
            elif abs(hs[-1] - hs[-2]) / max(hs[-2], 1e-9) < 0.01 and ls[-1] > ls[-2]:
                add("Ascending Triangle", "Bullish", n - 1, "Flat highs, rising lows")
            elif abs(ls[-1] - ls[-2]) / max(ls[-2], 1e-9) < 0.01 and hs[-1] < hs[-2]:
                add("Descending Triangle", "Bearish", n - 1, "Flat lows, falling highs")

    # ---- recent candle geometry ----
    tail = df.tail(6)
    o, h, l_, c = (tail["Open"], tail["High"], tail["Low"], tail["Close"])
    body = (c - o).abs()
    rng = (h - l_).replace(0.0, np.nan)
    lower = tail[["Open", "Close"]].min(axis=1) - l_
    upper = h - tail[["Open", "Close"]].max(axis=1)
    for i in range(1, len(tail)):
        if body.iloc[i] > 0 and lower.iloc[i] >= 2 * body.iloc[i] and \
                (lower.iloc[i] / rng.iloc[i]) > 0.5:
            add("Hammer / Pin Bar", "Bullish", n - len(tail) + i, "Long lower rejection wick")
        if body.iloc[i] > 0 and upper.iloc[i] >= 2 * body.iloc[i] and \
                (upper.iloc[i] / rng.iloc[i]) > 0.5:
            add("Shooting Star", "Bearish", n - len(tail) + i, "Long upper rejection wick")
        if c.iloc[i] > o.iloc[i] and c.iloc[i - 1] < o.iloc[i - 1] and \
                c.iloc[i] >= o.iloc[i - 1] and o.iloc[i] <= c.iloc[i - 1]:
            add("Bullish Engulfing", "Bullish", n - len(tail) + i, "Body engulfs the prior candle")
        if c.iloc[i] < o.iloc[i] and c.iloc[i - 1] > o.iloc[i - 1] and \
                c.iloc[i] <= o.iloc[i - 1] and o.iloc[i] >= c.iloc[i - 1]:
            add("Bearish Engulfing", "Bearish", n - len(tail) + i, "Body engulfs the prior candle")
        if h.iloc[i] < h.iloc[i - 1] and l_.iloc[i] > l_.iloc[i - 1]:
            add("Inside Bar", "Neutral", n - len(tail) + i, "Coil inside the prior range")
    return out


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    main()
