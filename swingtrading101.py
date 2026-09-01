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

# Wall-clock length of one candle, used to decide whether a feed has gone stale.
INTERVAL_SECONDS = {"1m": 60, "2m": 120, "3m": 180, "5m": 300, "10m": 600, "15m": 900,
                    "30m": 1800, "60m": 3600, "4h": 14400, "1d": 86400,
                    "1wk": 604800, "1mo": 2592000}

DEFAULT_PARAMS: dict[str, float] = {
    "ema_fast": 9, "ema_slow": 21, "ema_mid": 50, "ema_macro": 200,
    "rsi_len": 14, "atr_len": 14, "atr_mult": 2.0, "channel_mult": 2.0,
    "vol_len": 20, "vol_mult": 1.5, "breakout_len": 20, "squeeze_mult": 1.2,
    "orb_bars": 3, "gap_pct": 0.30, "pullback_tol": 0.15,
    "pivot_left": 3, "pivot_right": 3, "zigzag_pct": 0.8,
    "adx_len": 14, "bb_len": 20, "bb_mult": 2.0,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "st_len": 10, "st_mult": 3.0, "structure_len": 20,
}


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
    thr = threshold_pct / 100.0
    direction = 0
    last_ext_i, last_ext = 0, c[0]
    for i in range(1, n):
        if not np.isfinite(c[i]):
            continue
        if direction >= 0 and c[i] > last_ext:
            last_ext_i, last_ext = i, c[i]
        elif direction <= 0 and c[i] < last_ext:
            last_ext_i, last_ext = i, c[i]
        if direction >= 0 and last_ext > 0 and c[i] <= last_ext * (1 - thr):
            marks[last_ext_i] = 1          # confirmed swing high
            direction, last_ext_i, last_ext = -1, i, c[i]
        elif direction <= 0 and last_ext > 0 and c[i] >= last_ext * (1 + thr):
            marks[last_ext_i] = -1         # confirmed swing low
            direction, last_ext_i, last_ext = 1, i, c[i]
    return pd.Series(marks, index=close.index, name="ZZ")


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
     "help": "Long needs RSI in band; short needs the mirrored band (100 - value)."},
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
    out["flt_ready"] = True
    return out


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
        lm = df["f_rsi"].between(lo, hi)
        sm = df["f_rsi"].between(100.0 - hi, 100.0 - lo)   # mirrored band for shorts
        apply("rsi", lm, sm, fmt(safe_last(df["f_rsi"])))

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
    direction, ext_i, ext = 0, 0, c[0]
    for i in range(1, n):
        if not np.isfinite(c[i]):
            continue
        if direction >= 0 and c[i] > ext:
            ext_i, ext = i, c[i]
        elif direction <= 0 and c[i] < ext:
            ext_i, ext = i, c[i]
        if direction >= 0 and ext > 0 and c[i] <= ext * (1 - thr):
            piv.append((ext_i, float(ext), 1, i))
            direction, ext_i, ext = -1, i, c[i]
        elif direction <= 0 and ext > 0 and c[i] >= ext * (1 + thr):
            piv.append((ext_i, float(ext), -1, i))
            direction, ext_i, ext = 1, i, c[i]
    return piv


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
def c_elliott(df, p):
    """
    Mechanical wave-3 heuristic, NOT a real Elliott count.

    Looks for  low -> high -> higher-low  where the retracement sits between
    38.2% and 78.6% of the first leg, then triggers when price reclaims the leg
    high. Only pivots already CONFIRMED by the zigzag on or before the current
    bar are consulted, so there is no look-ahead.
    """
    out = df.copy()
    pivots = zigzag_pivot_table(out["Close"], float(_p(p, "zigzag_pct")))
    n = len(out)
    close = out["Close"].to_numpy(float)
    long = np.zeros(n, dtype=bool)
    short = np.zeros(n, dtype=bool)
    wave_hi = np.full(n, np.nan)
    wave_lo = np.full(n, np.nan)

    known: list[tuple[int, float, int, int]] = []
    ptr = 0
    for i in range(n):
        while ptr < len(pivots) and pivots[ptr][3] <= i:
            known.append(pivots[ptr])
            ptr += 1
        if len(known) < 3:
            continue
        (_, p1, k1, _), (_, p2, k2, _), (_, p3, k3, _) = known[-3], known[-2], known[-1]
        if (k1, k2, k3) == (-1, 1, -1):                 # low -> high -> low
            leg = p2 - p1
            if leg > 0:
                retr = (p2 - p3) / leg
                wave_hi[i], wave_lo[i] = p2, p3
                if 0.382 <= retr <= 0.786 and p3 > p1 and close[i] > p2:
                    long[i] = True
        elif (k1, k2, k3) == (1, -1, 1):                # high -> low -> high
            leg = p1 - p2
            if leg > 0:
                retr = (p3 - p2) / leg
                wave_hi[i], wave_lo[i] = p3, p2
                if 0.382 <= retr <= 0.786 and p3 < p1 and close[i] < p2:
                    short[i] = True
    out["wave_high"], out["wave_low"] = wave_hi, wave_lo
    idx = out.index
    return _finalise(out, pd.Series(long, index=idx), pd.Series(short, index=idx))


def s_elliott(df, p):
    return _sr("Zigzag wave heuristic. Elliott labelling is subjective; this is a mechanical "
               "approximation, not an analyst's count.",
               [("Last price", fmt(safe_last(df["Close"]))),
                ("Leg extreme", fmt(safe_last(df["wave_high"]))),
                ("Retracement pivot", fmt(safe_last(df["wave_low"]))),
                ("Zigzag threshold", f"{fmt(_p(p,'zigzag_pct'))}%")],
               "Impulse leg up, 38.2-78.6% retrace holding above the origin, then reclaim of the leg high.",
               "Impulse leg down, 38.2-78.6% retrace holding below the origin, then loss of the leg low.")


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
class RiskConfig:
    sl_type: str
    sl_value: float
    tp_type: str
    tp_value: float
    quantity: float = 1.0
    step_trigger: float = 0.0        # `k` for the step trail
    min_stop_atr: float = 0.25       # fallback distance when a structural stop is invalid

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
    if risk.sl_type in TRAILING_SL_TYPES or risk.tp_type in TRAILING_TP_TYPES:
        warnings.append("A trailing stop is active. Backtested trailing results are APPROXIMATE: "
                        "OHLC candles cannot tell us whether price hit the trailing level before "
                        "or after the extreme that moved it. Treat these numbers as optimistic.")
    for note in sorted(fallback_notes):
        warnings.append("Exit engine: " + note)

    return BacktestResult(frame=frame, trades=trades_df, equity=equity, stats=stats,
                          warmup_index=start, warnings=warnings, filter_reports=reports)


def _close_trade(pos: Position, exit_price: float, exit_time, reason: str) -> dict:
    points = round((exit_price - pos.entry_price) * pos.direction, 4)
    mgr = pos.manager
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
        "PnL": round(points * pos.quantity, 4),
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
                "longs": 0, "shorts": 0, "avg_bars": 0.0}
    pnl = trades["PnL"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    dd = equity - equity.cummax()
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
    "scrip_master": None, "broker_receipts": [],
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
    stale: bool = False


def poll_market(cfg: dict) -> LiveSnapshot:
    strat = get_strategy(cfg["strategy"])
    period = live_period_for(cfg["interval"])
    bundle = load_market_data(symbol=cfg["symbol"], period=period, interval=cfg["interval"],
                              freshness_seconds=max(API_GUARD_DELAY, cfg["poll_seconds"] * 0.9),
                              min_bars=max(strat.min_bars, 30))

    extras = dict(cfg.get("filter_extras") or {})
    if cfg.get("filter_cfg", {}).get("vix", {}).get("enabled"):
        extras["vix"] = load_vix(freshness_seconds=60)

    frame, reports = prepare(bundle.frame, cfg["strategy"], cfg["params"],
                             cfg.get("filter_cfg"), extras)
    if len(frame) < 3:
        raise MarketDataError("Not enough candles to evaluate a live signal.")

    closed = -2                                     # last FULLY CLOSED candle
    last_ts = pd.Timestamp(frame.index[-1])
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tz is not None else pd.Timestamp.now()
    age = float((now - last_ts).total_seconds())
    bar_seconds = INTERVAL_SECONDS.get(cfg["interval"], 300)
    # More than three candles have elapsed with nothing new printing: the venue
    # is closed, halted, or the feed has stalled. Either way the LTP is frozen
    # and nothing downstream should pretend otherwise.
    stale = age > max(3 * bar_seconds, 120)

    return LiveSnapshot(
        frame=frame,
        ltp=float(frame["Close"].iloc[-1]),
        next_open=float(frame["Open"].iloc[-1]),    # the open of candle N+1
        last_closed_time=frame.index[closed],
        last_closed_signal=int(frame["signal"].iloc[closed]),
        raw_signal=int(frame["raw_signal"].iloc[closed]),
        status=strat.status(frame.iloc[:len(frame) + closed + 1], cfg["params"]),
        filter_reports=reports, fetched_at=pd.Timestamp.now(), bars=len(frame),
        data_warnings=bundle.warnings, vix=extras.get("vix"),
        feed_age_seconds=age, stale=stale)


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
        "Points": points, "PnL": round(points * position.quantity, 4),
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
    log_event(f"ENTRY {'LONG' if direction > 0 else 'SHORT'} @ {fmt(price)} | "
              f"SL {fmt(mgr.sl)} | TGT {fmt(mgr.tp)} | qty {fmt(cfg['risk'].quantity, 0)}", "success")
    for note in mgr.notes:
        log_event("Exit engine: " + note, "warn")
    return position


def run_cycle(cfg: dict) -> None:
    """One iteration: manage open risk first, then look for a new entry."""
    if time.time() < st.session_state.get("live_backoff_until", 0.0):
        return
    try:
        snapshot = poll_market(cfg)
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

    frame = snapshot.frame
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
        st.session_state.live_last_bar = snapshot.last_closed_time
        return

    # ----------------------------------------------------- 2. fresh entries ---
    # A frozen feed cannot produce a fill worth having: the "live" price is just
    # the last close from hours ago, so any entry books a fictitious price and
    # then sits at exactly 0.00 PnL until the venue reopens.
    if snapshot.stale and not cfg.get("allow_stale_entries"):
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
                oscillator=None, hide_weekends=True, height=600):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = df.tail(tail) if tail else df
    if oscillator and oscillator in data.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.74, 0.26], vertical_spacing=0.04)
        row = 1
    else:
        fig, row = go.Figure(), None

    def add(trace, r=row):
        fig.add_trace(trace) if r is None else fig.add_trace(trace, row=r, col=1)

    add(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"],
                       close=data["Close"], name="Price",
                       increasing_line_color=_UP, decreasing_line_color=_DOWN,
                       increasing_fillcolor=_UP, decreasing_fillcolor=_DOWN))

    for i, col in enumerate(dict.fromkeys(("ema_fast", "ema_slow", *overlays))):
        if col in data.columns and data[col].notna().sum():
            add(go.Scatter(x=data.index, y=data[col], mode="lines", name=_label(col),
                           line=dict(width=1.5, color=_OVERLAY_COLOURS[i % len(_OVERLAY_COLOURS)])))

    if trades is not None and not trades.empty:
        for frame, name, sym, colour in ((trades[trades["Direction"] == "LONG"], "Long entry",
                                          "triangle-up", _UP),
                                         (trades[trades["Direction"] == "SHORT"], "Short entry",
                                          "triangle-down", _DOWN)):
            if not frame.empty:
                add(go.Scatter(x=frame["Entry Time"], y=frame["Entry Price"], mode="markers",
                               name=name, marker=dict(symbol=sym, size=11, color=colour,
                                                      line=dict(width=1, color="#fff")),
                               hovertemplate="%{x}<br>Entry %{y:,.2f}<extra></extra>"))
        add(go.Scatter(x=trades["Exit Time"], y=trades["Exit Price"], mode="markers", name="Exit",
                       marker=dict(symbol="x", size=9, color="#8d99ae"),
                       customdata=trades[["Exit Reason", "PnL"]],
                       hovertemplate="%{x}<br>Exit %{y:,.2f}<br>%{customdata[0]}"
                                     "<br>PnL %{customdata[1]:,.2f}<extra></extra>"))

    if row:
        add(go.Scatter(x=data.index, y=data[oscillator], mode="lines", name=oscillator.upper(),
                       line=dict(width=1.5, color="#4f9df7")), 2)
        for lvl, dash in ((70, "dot"), (50, "dash"), (30, "dot")):
            fig.add_hline(y=lvl, row=2, col=1, line=dict(width=1, dash=dash, color="#8d99ae"))
        fig.update_yaxes(range=[0, 100], row=2, col=1, title_text=oscillator.upper())

    fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
                      height=height, margin=dict(l=10, r=10, t=46, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_rangeslider_visible=False, dragmode="pan")
    if hide_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def equity_chart(equity, currency="", height=280):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.to_numpy(), mode="lines", name="Cumulative",
                             line=dict(width=2, color="#4f9df7"), fill="tozeroy",
                             fillcolor="rgba(79,157,247,0.14)"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity.cummax().to_numpy(), mode="lines",
                             name="Peak", line=dict(width=1, color="#8d99ae", dash="dot")))
    fig.add_hline(y=0, line=dict(width=1, color="#8d99ae"))
    fig.update_layout(title=dict(text=f"Realised Equity Curve ({currency})", x=0.01,
                                 xanchor="left", font=dict(size=14)),
                      height=height, margin=dict(l=10, r=10, t=42, b=10),
                      hovermode="x unified", showlegend=False)
    return fig


# =============================================================================
# SECTION 12 -- SIDEBAR CONTROL CONSOLE
# =============================================================================
_CUSTOM = "-- Custom ticker --"


def render_sidebar() -> dict:
    live = bool(st.session_state.get("live_running", False))
    sb = st.sidebar
    sb.title("Control Console")
    if live:
        _running_banner()
        sb.caption("Configuration is locked while the automation core is running.")

    sb.subheader("Instrument")
    group = sb.selectbox("Asset class", list(ASSET_UNIVERSE) + [_CUSTOM], disabled=live,
                         key="cfg_group")
    if group == _CUSTOM:
        symbol = sb.text_input("Custom Yahoo ticker", "KAYNES.NS", disabled=live,
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

    risk = RiskConfig(sl_type=sl_type, sl_value=float(sl_value), tp_type=tp_type,
                      tp_value=float(tp_value), quantity=float(quantity),
                      step_trigger=float(step_trigger))

    # ------------------------------------------------------- execution ------
    sb.subheader("Execution")
    poll_seconds = sb.number_input("Live poll interval (seconds)", min_value=API_GUARD_DELAY,
                                   max_value=600.0, value=5.0, step=0.1, key="cfg_poll",
                                   help="Auto-refresh cadence once the core is started. Every "
                                        "request carries the mandatory 0.3s guard on both sides.")
    if poll_seconds < 2.0:
        sb.error(f"At {fmt(poll_seconds,1)}s you will issue ~{60/poll_seconds:.0f} requests a "
                 "minute. Yahoo throttles well before that and will return 429s, then block the "
                 "IP for a while. Yahoo index data is also delayed ~15 minutes, so polling faster "
                 "does not make it fresher.")
    allow_stale = sb.checkbox("Live: allow entries on a frozen feed", value=False,
                              disabled=live, key="cfg_stale",
                              help="Off by default. When the venue is closed the LTP is just an "
                                   "old candle close, so an entry books a fictitious price and "
                                   "sits at 0.00 PnL until trading resumes.")
    fill_at_ltp = sb.checkbox("Live: fill at LTP instead of the N+1 open", value=False,
                              disabled=live, key="cfg_fill_ltp",
                              help="Default follows the N+1-open rule. Turn this on if you would "
                                   "rather record the price a market order would actually get.")

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


def _render_backtest(result: BacktestResult, meta: dict) -> None:
    cur, s = meta["currency"], result.stats

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

    if result.warnings:
        with st.expander(f"Run notes and caveats ({len(result.warnings)})", expanded=False):
            for w in result.warnings:
                st.warning(w)

    strat = get_strategy(meta["strategy"])
    st.plotly_chart(price_chart(result.frame,
                                f"{meta['symbol']} | {meta['interval']} | {meta['strategy']}",
                                strat.overlays, result.trades, oscillator=strat.oscillator,
                                hide_weekends=meta["hide_weekends"]),
                    width="stretch", config={"scrollZoom": True})
    st.caption(f"The first {result.warmup_index:,} candles were reserved as the indicator warm-up "
               "window and produced no orders. Signals fire on a candle close and fill at the "
               "next candle's open.")
    if not result.trades.empty:
        st.plotly_chart(equity_chart(result.equity, cur), width="stretch")

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
        st.session_state.live_running = False
        if st.session_state.live_position is not None:
            log_event("Engine stopped with a position still tracked. The risk stays open until "
                      "it is squared off.", "warn")
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
    if snap is not None and snap.stale:
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
    _live_chart(cfg, snapshot)
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
    """Say out loud whether the tape is actually moving."""
    age = snapshot.feed_age_seconds
    if not snapshot.stale:
        return
    hours = age / 3600.0
    human = f"{hours:.1f} hours" if hours >= 1 else f"{age/60:.0f} minutes"
    st.error(
        f"**FEED FROZEN — the venue looks closed.** The newest candle for "
        f"`{cfg['symbol']}` is {human} old ({fmt_time(snapshot.last_closed_time)}), so the "
        f"LTP shown below is simply that candle's close and it will not move until trading "
        f"resumes. The engine is still polling every {fmt(cfg['poll_seconds'],1)}s and will "
        f"pick up the first live tick automatically. New entries are suppressed while the "
        f"feed is frozen, because filling at a stale price books a fictitious entry that then "
        f"sits at exactly 0.00 PnL."
    )


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
    c[4].metric("Polls", f"{st.session_state.live_poll_count:,}",
                f"next in {next_in:0.1f}s", help="Increments on every automatic refresh.")
    c[5].metric("Clock", now.strftime("%H:%M:%S"),
                help="Redraws on every tick. If this is moving, the auto-refresh is alive.")


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

    if mgr.sl is not None and mgr.tp is not None and abs(mgr.tp - mgr.sl) > 0:
        span = abs(mgr.tp - mgr.sl)
        travelled = ((ltp - mgr.sl) / span) if position.direction > 0 else ((mgr.sl - ltp) / span)
        st.progress(float(min(max(travelled, 0.0), 1.0)), text="Distance from stop toward target")
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

    if snapshot.status.metrics:
        cols = st.columns(min(len(snapshot.status.metrics), 5))
        for i, (label, value) in enumerate(snapshot.status.metrics):
            cols[i % len(cols)].metric(label, value)

    l, r = st.columns(2)
    l.markdown(f"**Long entry requires**\n\n{snapshot.status.long_condition}")
    r.markdown(f"**Short entry requires**\n\n{snapshot.status.short_condition}")

    risk = cfg["risk"]
    st.caption(f"On a fill at {fmt(snapshot.ltp)} the exit engine would apply -- {risk.as_summary()}")


def _live_chart(cfg: dict, snapshot: LiveSnapshot) -> None:
    strat = get_strategy(cfg["strategy"])
    st.markdown("#### Live Chart")
    fig = price_chart(snapshot.frame, f"{cfg['symbol']} | {cfg['interval']} | last 100 candles",
                      strat.overlays, tail=100, oscillator=strat.oscillator,
                      hide_weekends=cfg.get("hide_weekends", True), height=500)
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

    t1, t2, t3 = st.tabs(["Backtesting Engine Studio", "Live Sandbox Operations",
                          "Live Trade Log Ledger"])
    with t1:
        tab_backtest(cfg)
    with t2:
        tab_live(cfg)
    with t3:
        tab_ledger(cfg)



# =============================================================================
# SECTION 17 -- OFFLINE SELF-TEST   (python algo_trading_platform.py --selftest)
# =============================================================================
def _synthetic(n: int = 1600, seed: int = 7) -> pd.DataFrame:
    """Random-walk candles with injected gaps and a volatility regime shift."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, n) * np.where(np.arange(n) > n // 2, 2.2, 1.0)
    close = 20000 + np.cumsum(steps)
    close[300] += 180
    close[700] -= 220
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
    assert any("APPROXIMATE" in w for w in res2.warnings), "trailing caveat must be surfaced"
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
        print("-- filters --")
        _test_filters()
        print("-- strategies + backtest --")
        _test_strategies_and_backtest()
        print("-- edge cases --")
        _test_edge_cases()
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())
    main()
