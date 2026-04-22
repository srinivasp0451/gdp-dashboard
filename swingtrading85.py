"""
Smart Investing - Professional Algorithmic Trading Platform
Author: Smart Investing
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
from datetime import datetime, timedelta
import pytz
import math
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════
IST = pytz.timezone("Asia/Kolkata")

TICKER_MAP = {
    "Nifty 50":    "^NSEI",
    "BankNifty":   "^NSEBANK",
    "Sensex":      "^BSESN",
    "BTC/USD":     "BTC-USD",
    "ETH/USD":     "ETH-USD",
    "Gold":        "GC=F",
    "Silver":      "SI=F",
    "Custom":      None,
}

TIMEFRAME_PERIOD_MAP = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

WARMUP_PERIOD = {
    "1m": "7d", "5m": "1mo", "15m": "1mo",
    "1h": "3mo", "1d": "2y", "1wk": "5y",
}

SL_TYPES = [
    "Custom Points", "ATR Based", "Risk Reward Based",
    "Trailing SL", "EMA Reverse Crossover",
    "Swing Low/High", "Candle Low/High",
    "Support/Resistance", "Volatility Based",
]

TARGET_TYPES = [
    "Custom Points", "ATR Based", "Risk Reward Based",
    "Trailing Target (Display)", "EMA Crossover Target",
    "Swing High/Low", "Auto Target",
    "Volatility Based",
]

STRATEGIES = [
    "EMA Crossover",
    "EMA Anticipation",
    "Elliott Wave",
    "Simple Buy",
    "Simple Sell",
]

# ═══════════════════════════════════════════════════════════════
# THREAD-SAFE GLOBAL STATE
# ═══════════════════════════════════════════════════════════════
_TS_LOCK = threading.Lock()
_TS: dict = {
    "running":      False,
    "position":     None,
    "trades":       [],
    "last_ltp":     None,
    "last_candle":  None,
    "ema_fast_val": None,
    "ema_slow_val": None,
    "atr_val":      None,
    "stop_event":   None,
    "log":          [],
    "wave_info":    {},
    "current_pnl":  0.0,
    "last_signal":  None,
    "fetch_count":  0,
}

def _ts_get(key):
    with _TS_LOCK:
        return _TS.get(key)

def _ts_set(key, val):
    with _TS_LOCK:
        _TS[key] = val

def _ts_append(key, val):
    with _TS_LOCK:
        if key not in _TS:
            _TS[key] = []
        _TS[key].append(val)

def _sync() -> dict:
    with _TS_LOCK:
        import copy
        return copy.deepcopy(_TS)

# ═══════════════════════════════════════════════════════════════
# DATA FETCHING  (rate-limited, warmup-aware)
# ═══════════════════════════════════════════════════════════════
_api_lock = threading.Lock()
_last_api_time = 0.0

def _rate_limited_fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch with ≥1.5s spacing between calls."""
    global _last_api_time
    with _api_lock:
        gap = time.time() - _last_api_time
        if gap < 1.5:
            time.sleep(1.5 - gap)
        _last_api_time = time.time()

    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
        return df
    except Exception as e:
        return pd.DataFrame()


def fetch_data(ticker: str, interval: str, period: str,
               warmup: bool = True) -> pd.DataFrame:
    """
    Fetch data with optional warmup for EMA accuracy.
    Always fetches warmup period to avoid NaN indicators on gapup/gapdown days.
    Then trims index to what's needed for display — but returns full for calcs.
    """
    fetch_period = WARMUP_PERIOD.get(interval, period) if warmup else period
    return _rate_limited_fetch(ticker, interval, fetch_period)

# ═══════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS  (TradingView-accurate)
# ═══════════════════════════════════════════════════════════════

def tv_ema(series: pd.Series, period: int) -> pd.Series:
    """
    TradingView-accurate EMA:
      • First `period` values seeded with SMA
      • Then standard Wilder/EWM(span=period, adjust=False)
    """
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    result = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return result


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    prev  = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def calc_pivot_highs(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    out = pd.Series(np.nan, index=series.index)
    for i in range(left, len(series) - right):
        if series.iloc[i] == series.iloc[i-left:i+right+1].max():
            out.iloc[i] = series.iloc[i]
    return out


def calc_pivot_lows(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    out = pd.Series(np.nan, index=series.index)
    for i in range(left, len(series) - right):
        if series.iloc[i] == series.iloc[i-left:i+right+1].min():
            out.iloc[i] = series.iloc[i]
    return out

# ═══════════════════════════════════════════════════════════════
# SL / TARGET CALCULATORS
# ═══════════════════════════════════════════════════════════════

def _atr_at(df: pd.DataFrame, idx: int, fallback: float) -> float:
    atr = calc_atr(df)
    if idx < len(atr) and not np.isnan(atr.iloc[idx]):
        return float(atr.iloc[idx])
    return fallback


def _swing_low_near(df: pd.DataFrame, idx: int, lookback: int = 30) -> float:
    sub = df["Low"].iloc[max(0, idx - lookback): idx + 1]
    lows = calc_pivot_lows(sub, 2, 2).dropna()
    return float(lows.iloc[-1]) if len(lows) > 0 else float(sub.min())


def _swing_high_near(df: pd.DataFrame, idx: int, lookback: int = 30) -> float:
    sub = df["High"].iloc[max(0, idx - lookback): idx + 1]
    highs = calc_pivot_highs(sub, 2, 2).dropna()
    return float(highs.iloc[-1]) if len(highs) > 0 else float(sub.max())


def get_sl(entry: float, side: str, sl_type: str, sl_pts: float,
           df: pd.DataFrame, idx: int,
           atr_m: float = 1.5, rr: float = 2.0, tgt: float = None) -> float:

    if side == "buy":
        base = {
            "Custom Points":        entry - sl_pts,
            "ATR Based":            entry - _atr_at(df, idx, sl_pts) * atr_m,
            "Risk Reward Based":    entry - (tgt - entry) / rr if tgt else entry - sl_pts,
            "Trailing SL":          entry - sl_pts,
            "EMA Reverse Crossover":entry - sl_pts,
            "Swing Low/High":       _swing_low_near(df, idx) - 0.5,
            "Candle Low/High":      float(df["Low"].iloc[idx]),
            "Support/Resistance":   entry - sl_pts,
            "Volatility Based":     entry - _atr_at(df, idx, sl_pts) * 2,
        }
    else:
        base = {
            "Custom Points":        entry + sl_pts,
            "ATR Based":            entry + _atr_at(df, idx, sl_pts) * atr_m,
            "Risk Reward Based":    entry + (entry - tgt) / rr if tgt else entry + sl_pts,
            "Trailing SL":          entry + sl_pts,
            "EMA Reverse Crossover":entry + sl_pts,
            "Swing Low/High":       _swing_high_near(df, idx) + 0.5,
            "Candle Low/High":      float(df["High"].iloc[idx]),
            "Support/Resistance":   entry + sl_pts,
            "Volatility Based":     entry + _atr_at(df, idx, sl_pts) * 2,
        }
    return round(base.get(sl_type, entry - sl_pts if side == "buy" else entry + sl_pts), 2)


def get_tgt(entry: float, side: str, tgt_type: str, tgt_pts: float,
            df: pd.DataFrame, idx: int,
            atr_m: float = 2.0, rr: float = 2.0, sl: float = None) -> float:

    if side == "buy":
        base = {
            "Custom Points":          entry + tgt_pts,
            "ATR Based":              entry + _atr_at(df, idx, tgt_pts) * atr_m,
            "Risk Reward Based":      entry + (entry - sl) * rr if sl else entry + tgt_pts,
            "Trailing Target (Display)": entry + tgt_pts,
            "EMA Crossover Target":   entry + tgt_pts,
            "Swing High/Low":         _swing_high_near(df, idx),
            "Auto Target":            entry + (entry - sl) * rr if sl else entry + tgt_pts,
            "Volatility Based":       entry + _atr_at(df, idx, tgt_pts) * atr_m,
        }
    else:
        base = {
            "Custom Points":          entry - tgt_pts,
            "ATR Based":              entry - _atr_at(df, idx, tgt_pts) * atr_m,
            "Risk Reward Based":      entry - (sl - entry) * rr if sl else entry - tgt_pts,
            "Trailing Target (Display)": entry - tgt_pts,
            "EMA Crossover Target":   entry - tgt_pts,
            "Swing High/Low":         _swing_low_near(df, idx),
            "Auto Target":            entry - (sl - entry) * rr if sl else entry - tgt_pts,
            "Volatility Based":       entry - _atr_at(df, idx, tgt_pts) * atr_m,
        }
    return round(base.get(tgt_type, entry + tgt_pts if side == "buy" else entry - tgt_pts), 2)

# ═══════════════════════════════════════════════════════════════
# ELLIOTT WAVE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_elliott_waves(df: pd.DataFrame) -> dict:
    """Full Elliott Wave analysis with Fibonacci projections."""
    if len(df) < 30:
        return {"status": "Insufficient data"}

    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    ph = calc_pivot_highs(high, 4, 4)
    pl = calc_pivot_lows (low,  4, 4)

    pivot_h = ph.dropna()
    pivot_l = pl.dropna()

    if len(pivot_h) < 2 or len(pivot_l) < 2:
        return {"status": "Not enough pivots"}

    # Build combined zigzag
    pivots = []
    for t, v in pivot_h.items():
        pivots.append({"time": t, "price": float(v), "ptype": "H"})
    for t, v in pivot_l.items():
        pivots.append({"time": t, "price": float(v), "ptype": "L"})
    pivots.sort(key=lambda x: x["time"])

    # Deduplicate consecutive same types
    dedup = [pivots[0]]
    for p in pivots[1:]:
        if p["ptype"] == dedup[-1]["ptype"]:
            # Keep the extreme
            if p["ptype"] == "H" and p["price"] > dedup[-1]["price"]:
                dedup[-1] = p
            elif p["ptype"] == "L" and p["price"] < dedup[-1]["price"]:
                dedup[-1] = p
        else:
            dedup.append(p)

    if len(dedup) < 5:
        return {"status": "Not enough alternating pivots"}

    last5 = dedup[-5:]
    cur   = float(close.iloc[-1])
    last_h= float(pivot_h.iloc[-1])
    last_l= float(pivot_l.iloc[-1])

    # Determine wave trend
    trend = "bullish" if last5[0]["ptype"] == "L" else "bearish"

    # Label W0..W5 (or best effort)
    wave_labels = ["W0", "W1", "W2", "W3", "W4"]
    waves = {}
    for lbl, p in zip(wave_labels, last5):
        waves[lbl] = {"price": p["price"], "time": str(p["time"]), "type": p["ptype"]}

    # Fibonacci projections
    fib_range = abs(last_h - last_l)
    if trend == "bullish":
        fib_levels = {
            "Ext 0.618": last_h + fib_range * 0.618,
            "Ext 1.000": last_h + fib_range * 1.000,
            "Ext 1.618": last_h + fib_range * 1.618,
            "Ret 0.236": last_h - fib_range * 0.236,
            "Ret 0.382": last_h - fib_range * 0.382,
            "Ret 0.500": last_h - fib_range * 0.500,
            "Ret 0.618": last_h - fib_range * 0.618,
            "Ret 0.786": last_h - fib_range * 0.786,
        }
    else:
        fib_levels = {
            "Ext 0.618": last_l - fib_range * 0.618,
            "Ext 1.000": last_l - fib_range * 1.000,
            "Ext 1.618": last_l - fib_range * 1.618,
            "Ret 0.236": last_l + fib_range * 0.236,
            "Ret 0.382": last_l + fib_range * 0.382,
            "Ret 0.500": last_l + fib_range * 0.500,
            "Ret 0.618": last_l + fib_range * 0.618,
            "Ret 0.786": last_l + fib_range * 0.786,
        }

    # Determine current wave position & signal
    last_pivot = dedup[-1]
    if trend == "bullish":
        if last_pivot["ptype"] == "L":
            current_wave = "Wave 5 (or new impulse) — Rising"
            next_target  = fib_levels["Ext 1.000"]
            signal       = "buy"
        else:
            current_wave = "Correction (ABC) — Falling"
            next_target  = fib_levels["Ret 0.382"]
            signal       = "sell"
    else:
        if last_pivot["ptype"] == "H":
            current_wave = "Wave 5 (or new impulse) — Falling"
            next_target  = fib_levels["Ext 1.000"]
            signal       = "sell"
        else:
            current_wave = "Correction (ABC) — Rising"
            next_target  = fib_levels["Ret 0.382"]
            signal       = "buy"

    # Completed waves summary
    completed = []
    for lbl in wave_labels:
        if lbl in waves:
            completed.append(f"{lbl} @ {waves[lbl]['price']:.2f}")

    return {
        "status":        "OK",
        "trend":         trend,
        "current_wave":  current_wave,
        "next_target":   round(next_target, 2),
        "signal":        signal,
        "fib_levels":    {k: round(v, 2) for k, v in fib_levels.items()},
        "waves":         waves,
        "completed":     completed,
        "last_high":     round(last_h, 2),
        "last_low":      round(last_l, 2),
        "current_price": round(cur, 2),
        "pivots":        dedup[-10:],
    }


def ew_signal_at(df: pd.DataFrame, idx: int) -> dict:
    """Walk-forward Elliott Wave signal for backtesting (no lookahead)."""
    sub = df.iloc[max(0, idx - 80): idx + 1]
    info = analyze_elliott_waves(sub)
    if info.get("status") != "OK":
        return {}

    sig  = info.get("signal")
    entry= float(df["Close"].iloc[idx])
    fib  = info.get("fib_levels", {})
    lh   = info.get("last_high", entry)
    ll   = info.get("last_low",  entry)

    if sig == "buy":
        sl     = fib.get("Ret 0.618", entry * 0.98)
        target = fib.get("Ext 1.000", entry * 1.03)
    else:
        sl     = fib.get("Ret 0.618", entry * 1.02)
        target = fib.get("Ext 1.000", entry * 0.97)

    return {"signal": sig, "entry": entry, "sl": sl, "target": target,
            "reason": f"EW {info['current_wave']}", "wave_info": info}

# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, strategy: str,
                     fast: int = 9, slow: int = 15,
                     cross_type: str = "Simple Crossover",
                     min_angle: float = 0.0,
                     candle_size: float = 10.0) -> pd.DataFrame:
    """
    Returns df copy with columns:
      signal (1=buy, -1=sell, 0=none), signal_reason, ema_fast, ema_slow
    """
    df = df.copy()
    df["signal"] = 0
    df["signal_reason"] = ""

    if strategy in ("EMA Crossover", "EMA Anticipation"):
        df["ema_fast"] = tv_ema(df["Close"], fast)
        df["ema_slow"] = tv_ema(df["Close"], slow)

    if strategy == "EMA Crossover":
        for i in range(1, len(df)):
            f0 = df["ema_fast"].iloc[i];   f1 = df["ema_fast"].iloc[i-1]
            s0 = df["ema_slow"].iloc[i];   s1 = df["ema_slow"].iloc[i-1]
            if any(pd.isna(v) for v in [f0, f1, s0, s1]):
                continue

            buy_x  = (f1 <= s1) and (f0 > s0)
            sell_x = (f1 >= s1) and (f0 < s0)

            # Crossover type filter
            c_size = abs(float(df["Close"].iloc[i]) - float(df["Open"].iloc[i]))
            if cross_type == "Custom Candle Size" and c_size < candle_size:
                buy_x = sell_x = False
            elif cross_type == "ATR Based Candle Size":
                atr = calc_atr(df)
                av = float(atr.iloc[i]) if i < len(atr) and not pd.isna(atr.iloc[i]) else candle_size
                if c_size < av * 0.5:
                    buy_x = sell_x = False

            # Angle filter (absolute gap change proxy)
            if min_angle > 0:
                if abs(f0 - f1) < min_angle:
                    buy_x = sell_x = False

            if buy_x:
                df.loc[df.index[i], "signal"] = 1
                df.loc[df.index[i], "signal_reason"] = (
                    f"EMA{fast} crossed ↑ EMA{slow} ({f0:.2f} > {s0:.2f})")
            elif sell_x:
                df.loc[df.index[i], "signal"] = -1
                df.loc[df.index[i], "signal_reason"] = (
                    f"EMA{fast} crossed ↓ EMA{slow} ({f0:.2f} < {s0:.2f})")

    elif strategy == "EMA Anticipation":
        # Anticipate crossover before it happens:
        # When gap is shrinking rapidly + price action confirms
        for i in range(3, len(df)):
            f0 = df["ema_fast"].iloc[i];   s0 = df["ema_slow"].iloc[i]
            f1 = df["ema_fast"].iloc[i-1]; s1 = df["ema_slow"].iloc[i-1]
            f2 = df["ema_fast"].iloc[i-2]; s2 = df["ema_slow"].iloc[i-2]
            if any(pd.isna(v) for v in [f0,f1,f2,s0,s1,s2]):
                continue

            gap0 = f0 - s0
            gap1 = f1 - s1
            gap2 = f2 - s2

            # Narrowing gap from below with 2 consecutive candles
            if gap0 < 0 and gap1 < 0 and gap2 < 0:
                if abs(gap0) < abs(gap1) < abs(gap2):
                    # 3-bar convergence below
                    c_bull = float(df["Close"].iloc[i]) > float(df["Open"].iloc[i])
                    if c_bull and abs(gap0) < abs(gap2) * 0.35:
                        df.loc[df.index[i], "signal"] = 1
                        df.loc[df.index[i], "signal_reason"] = (
                            f"Anticipated bullish EMA cross (gap {gap0:.2f}, was {gap2:.2f})")

            # Narrowing from above (bearish)
            elif gap0 > 0 and gap1 > 0 and gap2 > 0:
                if abs(gap0) < abs(gap1) < abs(gap2):
                    c_bear = float(df["Close"].iloc[i]) < float(df["Open"].iloc[i])
                    if c_bear and abs(gap0) < abs(gap2) * 0.35:
                        df.loc[df.index[i], "signal"] = -1
                        df.loc[df.index[i], "signal_reason"] = (
                            f"Anticipated bearish EMA cross (gap {gap0:.2f}, was {gap2:.2f})")

    elif strategy == "Elliott Wave":
        for i in range(50, len(df)):
            res = ew_signal_at(df, i)
            if res.get("signal") == "buy":
                df.loc[df.index[i], "signal"] = 1
                df.loc[df.index[i], "signal_reason"] = res.get("reason", "EW Buy")
            elif res.get("signal") == "sell":
                df.loc[df.index[i], "signal"] = -1
                df.loc[df.index[i], "signal_reason"] = res.get("reason", "EW Sell")

    elif strategy == "Simple Buy":
        # Flag every bar — backtest engine will take only first / per-position
        df.loc[df.index[0], "signal"] = 1
        df.loc[df.index[0], "signal_reason"] = "Simple Buy (immediate entry)"

    elif strategy == "Simple Sell":
        df.loc[df.index[0], "signal"] = -1
        df.loc[df.index[0], "signal_reason"] = "Simple Sell (immediate entry)"

    return df

# ═══════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, strategy: str, direction: str,
                 sl_type: str, tgt_type: str, sl_pts: float, tgt_pts: float,
                 fast: int, slow: int, rr: float, atr_m_sl: float, atr_m_tgt: float,
                 qty: int, cross_type: str, min_angle: float, candle_sz: float):
    """
    Returns (trades_df, violations_df, signals_df).

    Entry logic:
      • Simple Buy / Simple Sell : enter at that candle's CLOSE (immediate)
      • All other strategies     : enter at NEXT candle's OPEN (N+1)

    Exit logic (conservative, no lookahead):
      • BUY  : check candle Low  ≤ SL first → SL hit; else candle High ≥ Tgt → Tgt hit
      • SELL : check candle High ≥ SL first → SL hit; else candle Low  ≤ Tgt → Tgt hit
    """
    sig_df = generate_signals(df, strategy, fast, slow, cross_type, min_angle, candle_sz)

    trades     = []
    violations = []
    pos        = None          # open position
    immediate  = strategy in ("Simple Buy", "Simple Sell")
    i = 0

    while i < len(sig_df):
        bar       = sig_df.iloc[i]
        c_high    = float(df["High"].iloc[i])
        c_low     = float(df["Low"].iloc[i])
        c_close   = float(df["Close"].iloc[i])
        c_open    = float(df["Open"].iloc[i])

        # ── Manage open position ──────────────────────────────────────
        if pos is not None:
            ep    = pos["entry_price"]
            sl    = pos["sl"]
            tgt   = pos["tgt"]
            side  = pos["side"]

            # Trail SL update
            if sl_type == "Trailing SL":
                if side == "buy":
                    new_sl = c_close - sl_pts
                    if new_sl > sl:
                        pos["sl"] = new_sl; sl = new_sl
                else:
                    new_sl = c_close + sl_pts
                    if new_sl < sl:
                        pos["sl"] = new_sl; sl = new_sl

            elif sl_type == "Swing Low/High":
                if side == "buy":
                    new_sl = _swing_low_near(df, i) - 0.5
                    if new_sl > pos["sl"]:
                        pos["sl"] = new_sl; sl = new_sl
                else:
                    new_sl = _swing_high_near(df, i) + 0.5
                    if new_sl < pos["sl"]:
                        pos["sl"] = new_sl; sl = new_sl

            elif sl_type == "Candle Low/High":
                if side == "buy":
                    new_sl = c_low
                    if new_sl > pos["sl"]:
                        pos["sl"] = new_sl; sl = new_sl
                else:
                    new_sl = c_high
                    if new_sl < pos["sl"]:
                        pos["sl"] = new_sl; sl = new_sl

            # EMA reverse crossover SL
            elif sl_type == "EMA Reverse Crossover":
                if "ema_fast" in sig_df.columns and i > 0:
                    f0 = sig_df["ema_fast"].iloc[i]; s0 = sig_df["ema_slow"].iloc[i]
                    f1 = sig_df["ema_fast"].iloc[i-1]; s1 = sig_df["ema_slow"].iloc[i-1]
                    if not any(pd.isna(v) for v in [f0, f1, s0, s1]):
                        rev_buy  = (side == "buy")  and (f1 >= s1) and (f0 < s0)
                        rev_sell = (side == "sell") and (f1 <= s1) and (f0 > s0)
                        if rev_buy or rev_sell:
                            exit_p = c_open
                            pnl = (exit_p - ep)*qty if side=="buy" else (ep - exit_p)*qty
                            trades.append(_make_trade(pos, exit_p, c_high, c_low,
                                          df.index[i], "EMA Reverse Crossover SL",
                                          pnl, qty, False))
                            pos = None; i += 1; continue

            exit_p = exit_r = None
            violated = False

            if side == "buy":
                # Conservative: SL (Low) first, then Target (High)
                if c_low <= sl:
                    exit_p = sl; exit_r = "Stop Loss Hit"
                elif c_high >= tgt:
                    exit_p = tgt; exit_r = "Target Hit"
                # Violation: both possible in same candle
                if c_low <= sl and c_high >= tgt:
                    violated = True
            else:  # sell
                # Conservative: SL (High) first, then Target (Low)
                if c_high >= sl:
                    exit_p = sl; exit_r = "Stop Loss Hit"
                elif c_low <= tgt:
                    exit_p = tgt; exit_r = "Target Hit"
                if c_high >= sl and c_low <= tgt:
                    violated = True

            if exit_p is not None:
                pnl = (exit_p - ep)*qty if side=="buy" else (ep - exit_p)*qty
                rec = _make_trade(pos, exit_p, c_high, c_low, df.index[i], exit_r, pnl, qty, violated)
                trades.append(rec)
                if violated:
                    violations.append(rec)
                pos = None

        # ── New signal ────────────────────────────────────────────────
        if pos is None:
            sig    = int(bar["signal"])
            reason = str(bar.get("signal_reason", ""))

            # Direction filter
            if direction == "Buy Only"  and sig != 1:  i += 1; continue
            if direction == "Sell Only" and sig != -1: i += 1; continue

            if sig in (1, -1):
                side = "buy" if sig == 1 else "sell"

                if immediate:
                    entry_price = c_close
                    entry_time  = df.index[i]
                    entry_idx   = i
                else:
                    # Enter at next candle open
                    ni = i + 1
                    if ni >= len(df): i += 1; continue
                    entry_price = float(df["Open"].iloc[ni])
                    entry_time  = df.index[ni]
                    entry_idx   = ni
                    i = ni  # advance to entry candle so position is checked starting there

                sl_val  = get_sl(entry_price, side, sl_type, sl_pts, df, entry_idx, atr_m_sl, rr)
                tgt_val = get_tgt(entry_price, side, tgt_type, tgt_pts, df, entry_idx, atr_m_tgt, rr, sl_val)

                pos = {
                    "side":        side,
                    "entry_price": entry_price,
                    "entry_time":  entry_time,
                    "sl":          sl_val,
                    "tgt":         tgt_val,
                    "reason":      reason,
                }

        i += 1

    # Close any remaining open position at last close
    if pos is not None:
        exit_p = float(df["Close"].iloc[-1])
        pnl = (exit_p - pos["entry_price"])*qty if pos["side"]=="buy" else (pos["entry_price"] - exit_p)*qty
        trades.append(_make_trade(pos, exit_p,
                      float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]),
                      df.index[-1], "Market Close (Open Trade)", pnl, qty, False))

    t_df = pd.DataFrame(trades)    if trades    else pd.DataFrame()
    v_df = pd.DataFrame(violations) if violations else pd.DataFrame()
    return t_df, v_df, sig_df


def _make_trade(pos, exit_p, c_high, c_low, exit_time, exit_reason, pnl, qty, violated):
    return {
        "Entry Time":    pos["entry_time"],
        "Exit Time":     exit_time,
        "Trade Type":    pos["side"].upper(),
        "Entry Price":   round(pos["entry_price"], 2),
        "Exit Price":    round(exit_p, 2),
        "SL":            round(pos["sl"], 2),
        "Target":        round(pos["tgt"], 2),
        "Candle High":   round(c_high, 2),
        "Candle Low":    round(c_low, 2),
        "Entry Reason":  pos["reason"],
        "Exit Reason":   exit_reason,
        "PnL":           round(pnl, 2),
        "Violated":      violated,
    }

# ═══════════════════════════════════════════════════════════════
# LIVE TRADING LOOP  (background thread)
# ═══════════════════════════════════════════════════════════════

def live_loop(cfg: dict, stop_event: threading.Event):
    ticker      = cfg["ticker"]
    interval    = cfg["interval"]
    period      = cfg["period"]
    strategy    = cfg["strategy"]
    sl_type     = cfg["sl_type"]
    tgt_type    = cfg["tgt_type"]
    sl_pts      = cfg["sl_pts"]
    tgt_pts     = cfg["tgt_pts"]
    fast        = cfg["fast"]
    slow        = cfg["slow"]
    rr          = cfg["rr"]
    atr_m_sl    = cfg["atr_m_sl"]
    atr_m_tgt   = cfg["atr_m_tgt"]
    qty         = cfg["qty"]
    cooldown    = cfg["cooldown"]
    dhan_en     = cfg["dhan_enabled"]
    opts_en     = cfg["opts_enabled"]
    dcfg        = cfg["dhan_cfg"]
    immediate   = strategy in ("Simple Buy", "Simple Sell")

    _ts_set("running", True)
    _ts_set("log", [])

    def log(msg):
        ts = datetime.now(IST).strftime("%H:%M:%S")
        _ts_append("log", f"[{ts}] {msg}")

    log(f"Started ─ {ticker} | {interval} | {strategy}")

    last_signal_time = None
    prev_candle_time = None

    while not stop_event.is_set():
        try:
            _ts_set("fetch_count", (_ts_get("fetch_count") or 0) + 1)
            df = fetch_data(ticker, interval, period, warmup=True)
            if df is None or df.empty:
                log("⚠ Empty data returned"); stop_event.wait(1.5); continue

            cur_time  = datetime.now(IST)
            ltp       = float(df["Close"].iloc[-1])
            ef_series = tv_ema(df["Close"], fast)
            es_series = tv_ema(df["Close"], slow)
            atr_v     = calc_atr(df)
            last_c    = df.iloc[-1]

            _ts_set("last_ltp",     ltp)
            _ts_set("ema_fast_val", float(ef_series.iloc[-1]) if not pd.isna(ef_series.iloc[-1]) else None)
            _ts_set("ema_slow_val", float(es_series.iloc[-1]) if not pd.isna(es_series.iloc[-1]) else None)
            _ts_set("atr_val",      float(atr_v.iloc[-1]) if not pd.isna(atr_v.iloc[-1]) else None)
            _ts_set("last_candle",  {
                "Time":   str(df.index[-1].strftime("%Y-%m-%d %H:%M:%S")),
                "Open":   round(float(last_c["Open"]),  2),
                "High":   round(float(last_c["High"]),  2),
                "Low":    round(float(last_c["Low"]),   2),
                "Close":  round(float(last_c["Close"]), 2),
                "Volume": int(last_c.get("Volume", 0) or 0),
            })

            # Elliott Wave analysis
            wave_info = analyze_elliott_waves(df)
            _ts_set("wave_info", wave_info)

            # ── Manage open position (tick-by-tick SL/Target vs LTP) ──
            pos = _ts_get("position")
            if pos is not None:
                ep   = pos["entry_price"]
                sl   = pos["sl"]
                tgt  = pos["tgt"]
                side = pos["side"]

                # Update trailing
                if sl_type == "Trailing SL":
                    if side == "buy":
                        new_sl = ltp - sl_pts
                        if new_sl > sl:
                            pos["sl"] = new_sl; _ts_set("position", pos)
                    else:
                        new_sl = ltp + sl_pts
                        if new_sl < sl:
                            pos["sl"] = new_sl; _ts_set("position", pos)

                # Live PnL
                live_pnl = (ltp - ep)*qty if side=="buy" else (ep - ltp)*qty
                _ts_set("current_pnl", round(live_pnl, 2))

                # SL/Target check against LTP
                exit_p = exit_r = None
                if side == "buy":
                    if ltp <= sl:    exit_p = ltp; exit_r = "Stop Loss Hit"
                    elif ltp >= tgt: exit_p = ltp; exit_r = "Target Hit"
                else:
                    if ltp >= sl:    exit_p = ltp; exit_r = "Stop Loss Hit"
                    elif ltp <= tgt: exit_p = ltp; exit_r = "Target Hit"

                if exit_p is not None:
                    pnl = (exit_p - ep)*qty if side=="buy" else (ep - exit_p)*qty
                    _ts_append("trades", {
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    cur_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   side.upper(),
                        "Entry Price":  round(ep, 2),
                        "Exit Price":   round(exit_p, 2),
                        "SL":           round(sl, 2),
                        "Target":       round(tgt, 2),
                        "PnL":          round(pnl, 2),
                        "Exit Reason":  exit_r,
                        "Entry Reason": pos.get("reason", ""),
                    })
                    _ts_set("position", None)
                    _ts_set("current_pnl", 0.0)
                    log(f"✅ {exit_r} | {side.upper()} | PnL: ₹{pnl:.2f}")
                    if dhan_en:
                        _place_order(side, dcfg, opts_en, ltp, exit_mode=True)

            # ── New signal check ──────────────────────────────────────
            if _ts_get("position") is None:
                # Timeframe alignment (for EMA/EW strategies)
                if not immediate:
                    candle_time = df.index[-1]
                    if candle_time == prev_candle_time:
                        stop_event.wait(1.5); continue
                    prev_candle_time = candle_time

                sig_df = generate_signals(df, strategy, fast, slow)
                last_sig    = int(sig_df["signal"].iloc[-1])
                last_reason = str(sig_df["signal_reason"].iloc[-1])

                # Cooldown
                if cooldown > 0 and last_signal_time is not None:
                    if (cur_time - last_signal_time).total_seconds() < cooldown:
                        stop_event.wait(1.5); continue

                if last_sig != 0:
                    side       = "buy" if last_sig == 1 else "sell"
                    entry_p    = ltp  # Live: enter at LTP
                    idx_last   = len(df) - 1
                    sl_val     = get_sl(entry_p, side, sl_type, sl_pts, df, idx_last, atr_m_sl, rr)
                    tgt_val    = get_tgt(entry_p, side, tgt_type, tgt_pts, df, idx_last, atr_m_tgt, rr, sl_val)

                    new_pos = {
                        "side":        side,
                        "entry_price": round(entry_p, 2),
                        "entry_time":  cur_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "sl":          sl_val,
                        "tgt":         tgt_val,
                        "reason":      last_reason,
                    }
                    _ts_set("position", new_pos)
                    _ts_set("last_signal", {
                        "side": side, "entry": entry_p, "sl": sl_val, "tgt": tgt_val,
                        "reason": last_reason, "time": cur_time.strftime("%H:%M:%S")
                    })
                    last_signal_time = cur_time
                    log(f"🔔 {side.upper()} signal | Entry:{entry_p:.2f} SL:{sl_val:.2f} Tgt:{tgt_val:.2f}")
                    if dhan_en:
                        _place_order(side, dcfg, opts_en, entry_p, exit_mode=False)

        except Exception as e:
            _ts_append("log", f"[ERR] {str(e)}")

        stop_event.wait(1.5)

    _ts_set("running", False)
    log("⏹ Live trading stopped")


def _place_order(side: str, dcfg: dict, opts_en: bool, price: float, exit_mode: bool):
    try:
        from dhanhq import dhanhq
        client_id = dcfg.get("client_id", "")
        token     = dcfg.get("access_token", "")
        if not client_id or not token:
            return
        dhan = dhanhq(client_id, token)

        if opts_en:
            # Options: always BUY (CE on algo-buy, PE on algo-sell)
            if exit_mode:
                # For options exit, sell the same leg
                sec_id = dcfg.get("ce_security_id") if side=="buy" else dcfg.get("pe_security_id")
                tx_type = "SELL"
            else:
                sec_id  = dcfg.get("ce_security_id") if side == "buy" else dcfg.get("pe_security_id")
                tx_type = "BUY"
            exch  = dcfg.get("fno_exchange", "NSE_FNO")
            qty   = int(dcfg.get("options_qty", 65))
            otype = dcfg.get("options_entry_type" if not exit_mode else "options_exit_type", "MARKET")
            lim   = round(price, 2) if otype == "LIMIT" else 0
            dhan.place_order(transactionType=tx_type, exchangeSegment=exch,
                             productType="INTRADAY", orderType=otype,
                             validity="DAY", securityId=str(sec_id),
                             quantity=qty, price=lim, triggerPrice=0)
        else:
            if exit_mode:
                tx_type = "SELL" if side == "buy" else "BUY"
            else:
                tx_type = "BUY" if side == "buy" else "SELL"
            exch  = dcfg.get("exchange", "NSE")
            prod  = dcfg.get("product_type", "INTRADAY")
            sec   = dcfg.get("security_id", "1333")
            qty   = int(dcfg.get("equity_qty", 1))
            otype = dcfg.get("equity_entry_type" if not exit_mode else "equity_exit_type", "MARKET")
            lim   = round(price, 2) if otype == "LIMIT" else 0
            dhan.place_order(transactionType=tx_type, exchangeSegment=exch,
                             productType=prod, orderType=otype,
                             validity="DAY", securityId=str(sec),
                             quantity=qty, price=lim, triggerPrice=0)
    except Exception as e:
        _ts_append("log", f"[DHAN ERR] {str(e)}")


def get_my_ip() -> str:
    try:
        import requests
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except:
        return "Unknown"

# ═══════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

def run_optimization(df, target_acc, fast_range, slow_range, sl_list, tgt_list, qty, progress_cb=None):
    results = []
    total   = sum(1 for f in fast_range for s in slow_range if f < s) * len(sl_list) * len(tgt_list)
    done    = 0

    for f in fast_range:
        for s in slow_range:
            if f >= s:
                continue
            for sl_p in sl_list:
                for tgt_p in tgt_list:
                    t_df, _, _ = run_backtest(
                        df, "EMA Crossover", "Both",
                        "Custom Points", "Custom Points",
                        sl_p, tgt_p, f, s, 2.0, 1.5, 2.0, qty,
                        "Simple Crossover", 0.0, 10.0
                    )
                    done += 1
                    if progress_cb:
                        progress_cb(done / max(total, 1))

                    if t_df.empty:
                        continue
                    wins  = len(t_df[t_df["PnL"] > 0])
                    total_t = len(t_df)
                    acc   = wins / total_t * 100 if total_t else 0
                    results.append({
                        "Fast EMA": f, "Slow EMA": s,
                        "SL Pts": sl_p, "Tgt Pts": tgt_p,
                        "Trades": total_t, "Wins": wins,
                        "Accuracy %": round(acc, 1),
                        "Total PnL": round(t_df["PnL"].sum(), 2),
                        "Avg PnL":   round(t_df["PnL"].mean(), 2),
                    })

    if not results:
        return pd.DataFrame()
    res_df = pd.DataFrame(results).sort_values("Accuracy %", ascending=False)
    filtered = res_df[res_df["Accuracy %"] >= target_acc]
    return filtered if not filtered.empty else res_df  # Always return something

# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
CHART_THEME = dict(
    template        = "plotly_dark",
    paper_bgcolor   = "rgba(13,17,23,1)",
    plot_bgcolor    = "rgba(13,17,23,1)",
    margin          = dict(l=0, r=0, t=40, b=0),
    legend          = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_rangeslider_visible = False,
)

def plot_backtest_chart(df, sig_df, t_df, fast, slow):
    rows = [0.72, 0.28]
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=rows, vertical_spacing=0.02)

    # ── Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a", increasing_fillcolor="#0d2f27",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#2d0d0d",
    ), row=1, col=1)

    # ── EMAs
    for col, color, lbl in [
        ("ema_fast", "#FF6B35", f"EMA {fast}"),
        ("ema_slow", "#4ECDC4", f"EMA {slow}"),
    ]:
        if col in sig_df.columns:
            fig.add_trace(go.Scatter(
                x=sig_df.index, y=sig_df[col],
                name=lbl, line=dict(color=color, width=1.8)
            ), row=1, col=1)

    # ── Trade markers
    if not t_df.empty:
        for side, sym_e, sym_x, col_e, col_x in [
            ("BUY",  "triangle-up",   "x",           "#26a69a", "#1565C0"),
            ("SELL", "triangle-down", "triangle-up", "#ef5350", "#FF9800"),
        ]:
            sub = t_df[t_df["Trade Type"] == side]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["Entry Time"], y=sub["Entry Price"], mode="markers",
                name=f"{side} Entry",
                marker=dict(symbol=sym_e, size=13, color=col_e,
                            line=dict(width=1.5, color="white")),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sub["Exit Time"], y=sub["Exit Price"], mode="markers",
                name=f"{side} Exit",
                marker=dict(symbol=sym_x, size=11, color=col_x,
                            line=dict(width=1.5, color="white")),
            ), row=1, col=1)

    # ── Volume
    v_colors = [
        "rgba(38,166,154,0.55)" if c >= o else "rgba(239,83,80,0.55)"
        for c, o in zip(df["Close"], df["Open"])
    ]
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume", marker_color=v_colors
        ), row=2, col=1)

    fig.update_layout(title="Backtest Chart", height=580, **CHART_THEME)
    return fig


def plot_live_chart(df, pos, ef_series, es_series, fast, slow):
    plot_df = df.iloc[-120:]
    rows = [0.72, 0.28]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=rows, vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        name="Price",
        increasing_line_color="#26a69a", increasing_fillcolor="#0d2f27",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#2d0d0d",
    ), row=1, col=1)

    for series, color, lbl in [
        (ef_series, "#FF6B35", f"EMA {fast}"),
        (es_series, "#4ECDC4", f"EMA {slow}"),
    ]:
        if series is not None and len(series) > 0:
            fig.add_trace(go.Scatter(
                x=series.iloc[-120:].index, y=series.iloc[-120:],
                name=lbl, line=dict(color=color, width=1.8)
            ), row=1, col=1)

    if pos:
        ep = pos["entry_price"]
        sl = pos["sl"]
        tgt = pos["tgt"]
        fig.add_hline(y=ep,  line_color="#FFD700", line_width=2, line_dash="solid",
                      annotation_text=f"Entry {ep:.2f}", row=1, col=1)
        fig.add_hline(y=sl,  line_color="#ef5350", line_width=1.8, line_dash="dash",
                      annotation_text=f"SL {sl:.2f}", row=1, col=1)
        fig.add_hline(y=tgt, line_color="#26a69a", line_width=1.8, line_dash="dash",
                      annotation_text=f"Tgt {tgt:.2f}", row=1, col=1)

    v_colors = [
        "rgba(38,166,154,0.55)" if c >= o else "rgba(239,83,80,0.55)"
        for c, o in zip(plot_df["Close"], plot_df["Open"])
    ]
    if "Volume" in plot_df.columns:
        fig.add_trace(go.Bar(
            x=plot_df.index, y=plot_df["Volume"], name="Volume", marker_color=v_colors
        ), row=2, col=1)

    fig.update_layout(title="Live Trading Chart", height=520, **CHART_THEME)
    return fig


def plot_pnl_curve(t_df):
    cum = t_df["PnL"].cumsum()
    fig = go.Figure()
    total = float(cum.iloc[-1]) if len(cum) else 0
    color = "#26a69a" if total >= 0 else "#ef5350"
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum)+1)), y=cum,
        mode="lines+markers", fill="tozeroy",
        line=dict(color=color, width=2),
        fillcolor=f"rgba(38,166,154,0.15)" if total >= 0 else "rgba(239,83,80,0.15)",
        marker=dict(size=6, color=[
            "#26a69a" if p > 0 else "#ef5350" for p in t_df["PnL"]
        ]),
        name="Cumulative PnL",
    ))
    fig.update_layout(
        title="Cumulative PnL Curve", height=250,
        xaxis_title="Trade #", yaxis_title="PnL (₹)",
        **CHART_THEME
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════
def metric_card(label, value, sub=None, color="#FFD700"):
    sub_html = f"<div style='color:#888;font-size:0.75rem;margin-top:2px'>{sub}</div>" if sub else ""
    return (
        f"<div style='background:linear-gradient(135deg,#1a2332,#0f1923);"
        f"border:1px solid #2a3545;border-radius:10px;padding:14px 18px;"
        f"margin:4px 0;min-height:70px'>"
        f"<div style='color:#aaa;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px'>{label}</div>"
        f"<div style='color:{color};font-size:1.5rem;font-weight:700;line-height:1.2'>{value}</div>"
        f"{sub_html}</div>"
    )

def ltp_banner(name, ticker, ltp, change, pct):
    c = "#26a69a" if change >= 0 else "#ef5350"
    sign = "+" if change >= 0 else ""
    arrow = "▲" if change >= 0 else "▼"
    return (
        f"<div style='background:linear-gradient(90deg,#1a2332 0%,#0f1923 100%);"
        f"border:1px solid #2a3545;border-radius:12px;padding:14px 20px;"
        f"display:flex;align-items:center;gap:20px;margin-bottom:12px'>"
        f"<div>"
        f"  <div style='color:#8899aa;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px'>{name}</div>"
        f"  <div style='color:#aabbcc;font-size:0.9rem'>{ticker}</div>"
        f"</div>"
        f"<div style='flex:1'></div>"
        f"<div style='font-size:2rem;font-weight:800;color:#FFD700'>{ltp:,.2f}</div>"
        f"<div style='color:{c};font-size:1.1rem;font-weight:600'>{arrow} {sign}{change:.2f} ({sign}{pct:.2f}%)</div>"
        f"</div>"
    )

def apply_table_style(df):
    if df.empty:
        return df.style
    def row_color(row):
        pnl = row.get("PnL", row.get("Total PnL", 0))
        if pnl > 0:
            return ["background-color:rgba(38,166,154,0.12)"] * len(row)
        elif pnl < 0:
            return ["background-color:rgba(239,83,80,0.12)"]  * len(row)
        return [""] * len(row)
    return df.style.apply(row_color, axis=1)

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title = "Smart Investing",
        page_icon  = "📈",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Global CSS ─────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    code, pre, .stTextInput input { font-family: 'JetBrains Mono', monospace; }

    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background: #0a0f16 !important; border-right: 1px solid #1e2a38; }
    .stTabs [data-baseweb="tab-list"] { background: #0f1923; border-radius: 10px; gap: 4px; padding: 4px; }
    .stTabs [data-baseweb="tab"]      { background: transparent; color: #8899aa; border-radius: 8px; font-weight:500; }
    .stTabs [aria-selected="true"]    { background: linear-gradient(135deg,#1e3a5f,#162a45) !important; color:#4fc3f7 !important; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg,#1a2332,#0f1923);
        border: 1px solid #2a3545; border-radius: 10px; padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { color: #8899aa !important; font-size:0.75rem !important; text-transform:uppercase; letter-spacing:0.8px; }
    div[data-testid="metric-container"] [data-testid="metric-value"] { color: #e8edf2 !important; font-size:1.4rem !important; font-weight:700 !important; }

    .stButton > button {
        background: linear-gradient(135deg,#1e3a5f,#0f2040);
        border: 1px solid #2a4a6f; color: #7ec8e3;
        border-radius: 8px; font-weight: 600; transition: all 0.2s;
    }
    .stButton > button:hover { background: linear-gradient(135deg,#2a4f7f,#1a3060); border-color: #4fc3f7; color: #fff; }
    button[kind="primary"] { background: linear-gradient(135deg,#0d4f3c,#0a3828) !important; border-color:#26a69a !important; color:#26a69a !important; }
    button[kind="primary"]:hover { background: linear-gradient(135deg,#1a7a5c,#115040) !important; color:#fff !important; }

    .stDataFrame { border: 1px solid #2a3545; border-radius: 8px; overflow: hidden; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { background: #0f1923 !important; border-color: #2a3545 !important; color: #e8edf2 !important; }
    .stCheckbox label { color: #8899aa !important; }
    .stSidebar label { color: #7888aa !important; font-size: 0.8rem !important; }
    hr { border-color: #1e2a38 !important; }
    .st-expander { background: #0f1923; border: 1px solid #2a3545; border-radius: 8px; }

    .live-running { animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
    </style>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:12px 0 8px'>"
            "<span style='font-size:1.6rem;font-weight:800;color:#FFD700'>📈 Smart Investing</span>"
            "<div style='color:#8899aa;font-size:0.75rem;margin-top:4px'>Algorithmic Trading Platform</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        # Instrument
        st.markdown("#### 🎯 Instrument")
        ticker_name = st.selectbox("Select Ticker", list(TICKER_MAP.keys()), label_visibility="collapsed")
        if ticker_name == "Custom":
            ticker = st.text_input("Custom Ticker", value="RELIANCE.NS", placeholder="e.g. TCS.NS")
        else:
            ticker = TICKER_MAP[ticker_name]

        # Timeframe / Period
        st.markdown("#### ⏱ Timeframe")
        c1, c2 = st.columns(2)
        with c1:
            interval = st.selectbox("Interval", list(TIMEFRAME_PERIOD_MAP.keys()), index=2)
        with c2:
            periods = TIMEFRAME_PERIOD_MAP[interval]
            period  = st.selectbox("Period", periods, index=min(2, len(periods)-1))

        st.markdown("---")

        # Strategy
        st.markdown("#### 🧠 Strategy")
        strategy = st.selectbox("Strategy", STRATEGIES)

        fast = slow = 9
        cross_type = "Simple Crossover"
        min_angle  = 0.0
        candle_sz  = 10.0

        if strategy in ("EMA Crossover", "EMA Anticipation"):
            c1, c2 = st.columns(2)
            with c1:
                fast = st.number_input("Fast EMA", value=9,  min_value=2, max_value=200, step=1)
            with c2:
                slow = st.number_input("Slow EMA", value=15, min_value=2, max_value=500, step=1)

            cross_type = st.selectbox("Crossover Filter", [
                "Simple Crossover", "Custom Candle Size", "ATR Based Candle Size"])
            if cross_type == "Custom Candle Size":
                candle_sz = st.number_input("Min Candle Size (pts)", value=10.0, min_value=0.1)

            use_angle = st.checkbox("Min Crossover Angle", value=False)
            if use_angle:
                min_angle = st.number_input("Min Angle (pts diff)", value=0.0, min_value=0.0, step=0.5)

        direction = st.selectbox("Trade Direction", ["Both", "Buy Only", "Sell Only"])

        st.markdown("---")

        # SL
        st.markdown("#### 🛡 Stop Loss")
        sl_type = st.selectbox("SL Type", SL_TYPES)
        c1, c2  = st.columns(2)
        with c1:
            sl_pts  = st.number_input("SL Points", value=10.0, min_value=0.1, step=1.0)
        with c2:
            atr_m_sl = st.number_input("ATR Mult", value=1.5, min_value=0.1, step=0.1)

        # Target
        st.markdown("#### 🎯 Target")
        tgt_type = st.selectbox("Target Type", TARGET_TYPES)
        c1, c2   = st.columns(2)
        with c1:
            tgt_pts  = st.number_input("Tgt Points", value=20.0, min_value=0.1, step=1.0)
        with c2:
            atr_m_tgt= st.number_input("ATR Mult ", value=2.0, min_value=0.1, step=0.1)

        rr = st.number_input("R:R Ratio", value=2.0, min_value=0.1, step=0.5)

        st.markdown("---")

        # Position
        st.markdown("#### 📦 Position")
        qty = st.number_input("Quantity", value=1, min_value=1)

        # Live settings
        st.markdown("#### ⚙ Live Settings")
        use_cooldown  = st.checkbox("Cooldown Between Trades", value=True)
        cooldown_secs = st.number_input("Cooldown (seconds)", value=5, min_value=1) if use_cooldown else 0
        prev_overlap  = st.checkbox("Prevent Overlapping Trades", value=True)

        st.markdown("---")

        # Dhan
        st.markdown("#### 🏦 Dhan Broker")
        dhan_en   = st.checkbox("Enable Dhan Broker", value=False)
        opts_en   = False
        dhan_cfg  = {}

        if dhan_en:
            dhan_cfg["client_id"]     = st.text_input("Client ID", value="1104779876")
            dhan_cfg["access_token"]  = st.text_input("Access Token", type="password")

            opts_en = st.checkbox("Options Trading", value=False)

            if opts_en:
                dhan_cfg["fno_exchange"]    = st.selectbox("FNO Exchange", ["NSE_FNO", "BSE_FNO"])
                dhan_cfg["ce_security_id"]  = st.text_input("CE Security ID", value="57749")
                dhan_cfg["pe_security_id"]  = st.text_input("PE Security ID", value="57716")
                dhan_cfg["options_qty"]     = st.number_input("Lots Qty", value=65, min_value=1)
                dhan_cfg["options_entry_type"] = st.selectbox("Entry Type", ["MARKET", "LIMIT"], key="oe")
                dhan_cfg["options_exit_type"]  = st.selectbox("Exit Type",  ["MARKET", "LIMIT"], key="ox")
            else:
                dhan_cfg["exchange"]        = st.selectbox("Exchange",  ["NSE", "BSE"])
                dhan_cfg["product_type"]    = st.selectbox("Product",   ["INTRADAY", "DELIVERY"])
                dhan_cfg["security_id"]     = st.text_input("Security ID", value="12092")
                dhan_cfg["equity_qty"]      = st.number_input("Qty", value=1, min_value=1)
                dhan_cfg["equity_entry_type"] = st.selectbox("Entry Type", ["MARKET", "LIMIT"], key="ee")
                dhan_cfg["equity_exit_type"]  = st.selectbox("Exit Type",  ["MARKET", "LIMIT"], key="ex")

            if st.button("🌐 Register IP"):
                ip = get_my_ip()
                st.info(f"Your IP: {ip}\n\nDhan auto-registers IP on first API call.\nEnsure this IP is whitelisted in Dhan portal.")

    # ═══════════════════════════════════════════════════════════
    # LTP BANNER  (top of all tabs)
    # ═══════════════════════════════════════════════════════════
    try:
        ltp_df = _rate_limited_fetch(ticker, "1d", "2d")
        if not ltp_df.empty and len(ltp_df) >= 2:
            ltp   = float(ltp_df["Close"].iloc[-1])
            prev  = float(ltp_df["Close"].iloc[-2])
            chg   = ltp - prev
            pct   = chg / prev * 100 if prev else 0
            st.markdown(ltp_banner(ticker_name, ticker, ltp, chg, pct), unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#8899aa;font-size:0.85rem'>LTP unavailable for {ticker}</div>",
                        unsafe_allow_html=True)
    except:
        pass

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab_bt, tab_live, tab_hist, tab_opt = st.tabs([
        "📊  Backtesting", "⚡  Live Trading",
        "📋  Trade History", "🔧  Optimization",
    ])

    # ╔══════════════════════════════════════════════════════════╗
    # ║  TAB 1 — BACKTESTING                                     ║
    # ╚══════════════════════════════════════════════════════════╝
    with tab_bt:
        st.markdown("### 📊 Backtesting Engine")
        hdr1, hdr2, hdr3 = st.columns([2, 1, 1])
        with hdr2:
            run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)
        with hdr3:
            st.markdown(
                f"<div style='background:#1a2332;border:1px solid #2a3545;border-radius:8px;"
                f"padding:8px;text-align:center;color:#8899aa;font-size:0.82rem'>"
                f"Strategy: <b style='color:#FFD700'>{strategy}</b></div>",
                unsafe_allow_html=True
            )

        if strategy in ("Simple Buy", "Simple Sell"):
            st.info(
                f"ℹ **{strategy}**: Entry happens **immediately** at the signal candle's "
                f"close price. SL and Target are calculated right at entry. PnL is shown per trade.",
                icon=None
            )

        if run_bt:
            with st.spinner("Fetching data…"):
                df_bt = fetch_data(ticker, interval, period, warmup=True)

            if df_bt is None or df_bt.empty:
                st.error(f"❌ Could not fetch data for {ticker}. Check the ticker symbol.")
            else:
                with st.spinner("Running backtest…"):
                    t_df, v_df, s_df = run_backtest(
                        df_bt, strategy, direction,
                        sl_type, tgt_type, sl_pts, tgt_pts,
                        fast, slow, rr, atr_m_sl, atr_m_tgt,
                        qty, cross_type, min_angle, candle_sz,
                    )

                # ── Summary metrics
                if not t_df.empty:
                    wins   = int((t_df["PnL"] > 0).sum())
                    losses = int((t_df["PnL"] < 0).sum())
                    total  = len(t_df)
                    acc    = wins / total * 100 if total else 0
                    tpnl   = float(t_df["PnL"].sum())
                    avg_w  = float(t_df.loc[t_df["PnL"]>0,"PnL"].mean()) if wins  else 0
                    avg_l  = float(t_df.loc[t_df["PnL"]<0,"PnL"].mean()) if losses else 0
                    pf     = abs(t_df.loc[t_df["PnL"]>0,"PnL"].sum() / t_df.loc[t_df["PnL"]<0,"PnL"].sum()) if losses else float("inf")

                    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
                    c1.metric("Total Trades", total)
                    c2.metric("Wins",  wins)
                    c3.metric("Losses",losses)
                    c4.metric("Accuracy", f"{acc:.1f}%")
                    c5.metric("Total PnL", f"₹{tpnl:,.0f}", delta=f"{tpnl:+.0f}")
                    c6.metric("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞")
                    c7.metric("SL Violations", len(v_df))

                    c1b, c2b, c3b = st.columns(3)
                    c1b.metric("Avg Win",  f"₹{avg_w:,.1f}")
                    c2b.metric("Avg Loss", f"₹{avg_l:,.1f}")
                    c3b.metric("Max PnL",  f"₹{float(t_df['PnL'].max()):,.1f}")

                    # ── Chart
                    fig_bt = plot_backtest_chart(df_bt, s_df, t_df, fast, slow)
                    st.plotly_chart(fig_bt, use_container_width=True)

                    # ── PnL curve
                    st.plotly_chart(plot_pnl_curve(t_df), use_container_width=True)

                    # ── Trades table
                    st.markdown("#### 📋 All Trades")
                    disp_cols = ["Entry Time","Exit Time","Trade Type","Entry Price",
                                 "Exit Price","SL","Target","Candle High","Candle Low",
                                 "Entry Reason","Exit Reason","PnL"]
                    avail = [c for c in disp_cols if c in t_df.columns]
                    st.dataframe(apply_table_style(t_df[avail]), use_container_width=True, height=380)

                    # ── Violations
                    if not v_df.empty:
                        st.markdown(
                            f"<div style='background:rgba(255,160,0,0.1);border:1px solid rgba(255,160,0,0.4);"
                            f"border-radius:8px;padding:12px;margin:8px 0'>"
                            f"⚠ <b>{len(v_df)} Violation(s)</b> — Both SL and Target fell within the same candle's range. "
                            f"Conservative SL-first rule applied. These trades may deviate from live trading results.</div>",
                            unsafe_allow_html=True
                        )
                        st.dataframe(v_df[avail], use_container_width=True, height=250)
                else:
                    st.warning("No trades generated with this configuration.")

    # ╔══════════════════════════════════════════════════════════╗
    # ║  TAB 2 — LIVE TRADING                                    ║
    # ╚══════════════════════════════════════════════════════════╝
    with tab_live:
        is_running = bool(_ts_get("running"))

        # ── Controls
        st.markdown("### ⚡ Live Trading")
        bc1, bc2, bc3, bc4 = st.columns([1,1,1,2])

        with bc1:
            if not is_running:
                if st.button("▶ START", type="primary", use_container_width=True):
                    stop_ev = threading.Event()
                    cfg = dict(
                        ticker=ticker, interval=interval, period=period,
                        strategy=strategy, sl_type=sl_type, tgt_type=tgt_type,
                        sl_pts=sl_pts, tgt_pts=tgt_pts, fast=fast, slow=slow,
                        rr=rr, atr_m_sl=atr_m_sl, atr_m_tgt=atr_m_tgt,
                        qty=qty, cooldown=cooldown_secs, dhan_enabled=dhan_en,
                        opts_enabled=opts_en, dhan_cfg=dhan_cfg,
                    )
                    _ts_set("stop_event", stop_ev)
                    _ts_set("position", None)
                    _ts_set("trades",   [])
                    _ts_set("log",      [])
                    t = threading.Thread(target=live_loop, args=(cfg, stop_ev), daemon=True)
                    t.start()
                    st.success("✅ Live trading started!")
                    time.sleep(0.2); st.rerun()
            else:
                if st.button("⏹ STOP", use_container_width=True):
                    ev = _ts_get("stop_event")
                    if ev: ev.set()
                    st.rerun()

        with bc2:
            if st.button("⚡ SQUAREOFF", use_container_width=True):
                pos = _ts_get("position")
                if pos:
                    ltp_sq = _ts_get("last_ltp") or pos["entry_price"]
                    pnl_sq = (ltp_sq - pos["entry_price"])*qty if pos["side"]=="buy" \
                             else (pos["entry_price"] - ltp_sq)*qty
                    _ts_append("trades", {
                        "Entry Time":   pos["entry_time"],
                        "Exit Time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "Trade Type":   pos["side"].upper(),
                        "Entry Price":  pos["entry_price"],
                        "Exit Price":   round(ltp_sq, 2),
                        "SL":           pos["sl"], "Target": pos["tgt"],
                        "PnL":          round(pnl_sq, 2),
                        "Exit Reason":  "Manual Squareoff",
                        "Entry Reason": pos.get("reason",""),
                    })
                    _ts_set("position", None); _ts_set("current_pnl", 0.0)
                    st.success(f"Squared off | PnL: ₹{pnl_sq:.2f}")
                    st.rerun()
                else:
                    st.warning("No open position")

        with bc3:
            badge_c = "#1a4a2e" if is_running else "#2d0d0d"
            badge_b = "#26a69a" if is_running else "#ef5350"
            badge_t = "🟢 LIVE" if is_running else "🔴 OFFLINE"
            st.markdown(
                f"<div class='{'live-running' if is_running else ''}' "
                f"style='background:{badge_c};border:1px solid {badge_b};border-radius:8px;"
                f"padding:10px;text-align:center;font-weight:700;color:{badge_b};margin-top:4px'>"
                f"{badge_t}</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Active Config Panel
        with st.expander("⚙ Active Configuration", expanded=False):
            cfg_rows = {
                "Ticker": f"{ticker_name}  ({ticker})",
                "Interval / Period": f"{interval} / {period}",
                "Strategy": strategy,
                "Fast EMA / Slow EMA": f"{fast} / {slow}",
                "SL Type / Points": f"{sl_type} / {sl_pts} pts",
                "Target Type / Points": f"{tgt_type} / {tgt_pts} pts",
                "ATR Mult (SL/Tgt)": f"{atr_m_sl} / {atr_m_tgt}",
                "R:R Ratio": rr,
                "Quantity": qty,
                "Cooldown": f"{cooldown_secs}s",
                "Dhan Broker": "✅ Enabled" if dhan_en else "❌ Disabled",
                "Options Trading": "✅ Enabled" if opts_en else "❌ Disabled",
            }
            cfg_df = pd.DataFrame(list(cfg_rows.items()), columns=["Parameter","Value"])
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)

        # ── Live data
        state = _sync()
        ltp_v    = state.get("last_ltp")
        ef_v     = state.get("ema_fast_val")
        es_v     = state.get("ema_slow_val")
        atr_v    = state.get("atr_val")
        opnl     = state.get("current_pnl", 0.0)
        fetches  = state.get("fetch_count", 0)

        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        mc1.metric("LTP",           f"{ltp_v:,.2f}"  if ltp_v else "—")
        mc2.metric(f"EMA {fast}",   f"{ef_v:.2f}"    if ef_v  else "—")
        mc3.metric(f"EMA {slow}",   f"{es_v:.2f}"    if es_v  else "—")
        mc4.metric("ATR",           f"{atr_v:.2f}"   if atr_v else "—")
        mc5.metric("Open PnL",      f"₹{opnl:.2f}",
                   delta=f"{opnl:+.2f}", delta_color="normal")

        # ── Last candle
        last_c = state.get("last_candle")
        if last_c:
            st.markdown("#### 📌 Last Fetched Candle")
            lc_df = pd.DataFrame([last_c])
            st.dataframe(lc_df, use_container_width=True, hide_index=True)

        # ── Open Position
        pos = state.get("position")
        st.markdown("#### 📍 Current Position")
        if pos:
            side_c = "#26a69a" if pos["side"]=="buy" else "#ef5350"
            p1,p2,p3,p4,p5 = st.columns(5)
            p1.markdown(
                f"<div style='background:rgba(38,166,154,0.1) if buy else;border:1px solid {side_c};"
                f"border-radius:8px;padding:12px;text-align:center'>"
                f"<div style='color:{side_c};font-size:1.5rem;font-weight:800'>{pos['side'].upper()}</div>"
                f"<div style='color:#8899aa;font-size:0.7rem'>SIDE</div></div>",
                unsafe_allow_html=True
            )
            p2.metric("Entry",   f"{pos['entry_price']:,.2f}")
            p3.metric("SL",      f"{pos['sl']:,.2f}")
            p4.metric("Target",  f"{pos['tgt']:,.2f}")
            p5.metric("Qty",     qty)
            st.caption(f"⏱ {pos['entry_time']}  |  📌 {pos.get('reason','—')}")

            # Distance to SL and Target
            if ltp_v:
                d_sl  = abs(ltp_v - pos['sl'])
                d_tgt = abs(ltp_v - pos['tgt'])
                d1, d2 = st.columns(2)
                d1.metric("Distance to SL",  f"{d_sl:.2f} pts")
                d2.metric("Distance to Tgt", f"{d_tgt:.2f} pts")
        else:
            st.info("No open position currently.")

        # ── Elliott Wave
        wave = state.get("wave_info", {})
        if wave and wave.get("status") == "OK":
            st.markdown("#### 🌊 Elliott Wave Analysis")
            w1,w2,w3 = st.columns(3)
            w1.metric("Trend",        wave.get("trend","—").capitalize())
            w2.metric("Current Wave", wave.get("current_wave","—"))
            w3.metric("EW Signal",    wave.get("signal","—").upper() if wave.get("signal") else "—")

            if wave.get("completed"):
                st.caption("Completed pivots: " + "  →  ".join(wave["completed"]))

            fib = wave.get("fib_levels", {})
            if fib:
                fib_df = pd.DataFrame([{"Level": k, "Price": v, "Vs LTP": round(v - (ltp_v or 0), 2)}
                                        for k, v in fib.items()])
                st.dataframe(fib_df, use_container_width=True, hide_index=True, height=240)

            if wave.get("next_target"):
                st.metric("Next Elliott Target", f"{wave['next_target']:,.2f}")

        # ── Live Chart
        if is_running:
            try:
                df_live = fetch_data(ticker, interval, period, warmup=False)
                if not df_live.empty:
                    ef_s = tv_ema(df_live["Close"], fast)
                    es_s = tv_ema(df_live["Close"], slow)
                    st.plotly_chart(
                        plot_live_chart(df_live, pos, ef_s, es_s, fast, slow),
                        use_container_width=True
                    )
            except:
                pass

        # ── Log
        logs = state.get("log", [])
        if logs:
            st.markdown("#### 📝 Activity Log")
            log_text = "\n".join(reversed(logs[-30:]))
            st.text_area("", log_text, height=200, label_visibility="collapsed")

        # Auto-refresh while running
        if is_running:
            time.sleep(0.3)
            st.rerun()

    # ╔══════════════════════════════════════════════════════════╗
    # ║  TAB 3 — TRADE HISTORY                                   ║
    # ╚══════════════════════════════════════════════════════════╝
    with tab_hist:
        st.markdown("### 📋 Trade History")
        st.caption("Updated live during trading. No need to stop live trading to view history.")

        all_trades = _ts_get("trades") or []

        if st.button("🔄 Refresh History"):
            st.rerun()

        if all_trades:
            hist_df = pd.DataFrame(all_trades)

            wins   = int((hist_df["PnL"] > 0).sum())
            losses = int((hist_df["PnL"] < 0).sum())
            total  = len(hist_df)
            acc    = wins/total*100 if total else 0
            tpnl   = float(hist_df["PnL"].sum())

            h1,h2,h3,h4,h5 = st.columns(5)
            h1.metric("Trades",   total)
            h2.metric("Wins",     wins)
            h3.metric("Losses",   losses)
            h4.metric("Accuracy", f"{acc:.1f}%")
            h5.metric("Total PnL",f"₹{tpnl:,.0f}", delta=f"{tpnl:+.0f}")

            st.plotly_chart(plot_pnl_curve(hist_df), use_container_width=True)

            st.markdown("#### All Trades")
            hist_cols = ["Entry Time","Exit Time","Trade Type","Entry Price",
                         "Exit Price","SL","Target","PnL","Exit Reason","Entry Reason"]
            avail = [c for c in hist_cols if c in hist_df.columns]
            st.dataframe(apply_table_style(hist_df[avail]), use_container_width=True, height=500)

            # Export
            csv = hist_df.to_csv(index=False)
            st.download_button("⬇ Download CSV", csv, "trade_history.csv", "text/csv")
        else:
            st.info("No completed trades yet. Trade history will appear here as trades are completed during live trading.")

    # ╔══════════════════════════════════════════════════════════╗
    # ║  TAB 4 — OPTIMIZATION                                    ║
    # ╚══════════════════════════════════════════════════════════╝
    with tab_opt:
        st.markdown("### 🔧 Strategy Optimization")
        st.caption("Grid-searches EMA periods, SL & Target points. Always returns best available results even if target accuracy isn't reached.")

        oc1, oc2 = st.columns(2)
        with oc1:
            tgt_acc  = st.number_input("Target Accuracy (%)", value=50.0, min_value=0.0, max_value=100.0, step=5.0)
            fast_min = st.number_input("Fast EMA Min", value=5,  min_value=2, max_value=50)
            fast_max = st.number_input("Fast EMA Max", value=15, min_value=2, max_value=50)
            fast_stp = st.number_input("Fast EMA Step",value=2,  min_value=1, max_value=10)

        with oc2:
            slow_min = st.number_input("Slow EMA Min", value=10, min_value=3,  max_value=200)
            slow_max = st.number_input("Slow EMA Max", value=30, min_value=3,  max_value=200)
            slow_stp = st.number_input("Slow EMA Step",value=5,  min_value=1,  max_value=20)

        sl_inp  = st.text_input("SL Points (comma-sep)",  "5,10,15,20")
        tgt_inp = st.text_input("Tgt Points (comma-sep)", "10,20,30,40")

        if st.button("🚀 Run Optimization", type="primary"):
            with st.spinner("Running grid search…"):
                df_opt = fetch_data(ticker, interval, period, warmup=True)

            if df_opt is None or df_opt.empty:
                st.error("Failed to fetch data.")
            else:
                try:
                    sl_list  = [float(x.strip()) for x in sl_inp.split(",")]
                    tgt_list = [float(x.strip()) for x in tgt_inp.split(",")]
                except:
                    sl_list  = [5.0, 10.0, 15.0, 20.0]
                    tgt_list = [10.0, 20.0, 30.0, 40.0]

                fast_range = list(range(fast_min, fast_max+1, max(fast_stp,1)))
                slow_range = list(range(slow_min, slow_max+1, max(slow_stp,1)))

                prog = st.progress(0.0, text="Optimizing…")
                def progress_cb(v):
                    prog.progress(min(v, 1.0))

                with st.spinner("Grid searching…"):
                    res_df = run_optimization(df_opt, tgt_acc, fast_range, slow_range,
                                              sl_list, tgt_list, qty, progress_cb)
                prog.empty()

                if not res_df.empty:
                    best = res_df.iloc[0]
                    st.success(f"Found {len(res_df)} combinations | Best accuracy: {best['Accuracy %']:.1f}%")

                    b1,b2,b3,b4,b5 = st.columns(5)
                    b1.metric("Best Accuracy", f"{best['Accuracy %']:.1f}%")
                    b2.metric("Fast EMA",      int(best["Fast EMA"]))
                    b3.metric("Slow EMA",      int(best["Slow EMA"]))
                    b4.metric("SL pts",        best["SL Pts"])
                    b5.metric("Total PnL",     f"₹{best['Total PnL']:,.0f}")

                    # Highlight rows meeting target accuracy
                    def style_opt(row):
                        if row["Accuracy %"] >= tgt_acc:
                            return ["background-color:rgba(38,166,154,0.15)"] * len(row)
                        return [""] * len(row)

                    styled_opt = res_df.style.apply(style_opt, axis=1)
                    st.dataframe(styled_opt, use_container_width=True, height=500)

                    csv_opt = res_df.to_csv(index=False)
                    st.download_button("⬇ Download Results", csv_opt, "optimization.csv", "text/csv")
                else:
                    st.warning("No valid combinations found. Try a wider parameter range.")


if __name__ == "__main__":
    main()
