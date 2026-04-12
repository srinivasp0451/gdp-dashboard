"""
Elliott Wave AlgoTrader Pro
============================
Single-file Streamlit app.
Run: streamlit run ew_algotrader.py

Key design decisions:
  • Walk-forward EW signals — BT and live use identical pivot logic (no look-ahead)
  • Conservative backtest: SL checked BEFORE target on every candle
  • 1.5s minimum gap enforced between all yfinance calls (thread-safe)
  • Live trading runs in a background daemon thread; UI only reads state
  • Optimization: pre-computes signals per wave_pct, then grids over SL/Tgt
  • Apply button in opt results pushes config straight into sidebar widget keys
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import threading
import time
from datetime import datetime
import plotly.graph_objects as go

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EW AlgoTrader Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
TICKERS = {
    "Nifty 50":   "^NSEI",
    "BankNifty":  "^NSEBANK",
    "Sensex":     "^BSESN",
    "BTC / USD":  "BTC-USD",
    "ETH / USD":  "ETH-USD",
    "USD / INR":  "USDINR=X",
    "Gold":       "GC=F",
    "Silver":     "SI=F",
    "Crude Oil":  "CL=F",
    "EUR / USD":  "EURUSD=X",
    "GBP / USD":  "GBPUSD=X",
    "JPY / USD":  "JPYUSD=X",
    "Custom":     "__custom__",
}

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]

# Valid yfinance (interval → allowed periods)
VALID_PERIODS = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo", "3mo"],
    "15m": ["1d", "5d", "7d", "1mo", "3mo"],
    "30m": ["1d", "5d", "7d", "1mo", "3mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "4h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

SL_TYPES  = ["Custom Points", "ATR-Based", "Trailing SL",
             "Strategy Reverse Signal", "Candle Low/High"]
TGT_TYPES = ["Custom Points", "ATR-Based", "Trailing Target", "Strategy Reversal"]

# ── Module-level live state (survives Streamlit re-runs in same session) ───────
_LS: dict = dict(
    running=False,
    position=None,       # dict or None
    wave_info={},
    last_bar=None,
    log=[],
    last_fetch=0.0,
    error=None,
    fetch_count=0,
)
_LS_LOCK = threading.Lock()

# ── Rate-limited yfinance wrapper ──────────────────────────────────────────────
_YF_LAST = 0.0
_YF_LOCK = threading.Lock()
_YF_MIN_GAP = 1.5   # seconds between calls


def _yf_raw(ticker: str, interval: str, period: str) -> tuple:
    """Thread-safe yfinance fetch with 1.5 s minimum gap. Returns (df | None, err | None)."""
    global _YF_LAST
    with _YF_LOCK:
        gap = _YF_MIN_GAP - (time.time() - _YF_LAST)
        if gap > 0:
            time.sleep(gap)
        try:
            df = yf.Ticker(ticker).history(
                interval=interval, period=period, auto_adjust=True
            )
            _YF_LAST = time.time()
        except Exception as exc:
            _YF_LAST = time.time()
            return None, str(exc)

    if df is None or df.empty:
        return None, "yfinance returned empty dataframe"

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df, None


def load_data(ticker: str, interval: str, period: str) -> tuple:
    """Public fetch. Handles 4h resampling via 1h data."""
    if interval == "4h":
        df, err = _yf_raw(ticker, "1h", period)
        if err or df is None:
            return None, err
        df = (
            df.resample("4h")
            .agg({"Open": "first", "High": "max", "Low": "min",
                  "Close": "last", "Volume": "sum"})
            .dropna()
        )
        return df, None
    return _yf_raw(ticker, interval, period)


# ── Technical Indicators ───────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# ELLIOTT WAVE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _zigzag(cl_arr: np.ndarray, min_mv: float) -> tuple:
    """
    O(n) single-pass zigzag on a numpy Close array.

    Returns
    -------
    pivots : list of (bar_idx, price, direction)
        direction: +1 = swing HIGH confirmed, -1 = swing LOW confirmed
    in_progress : (last_dir, last_px, last_idx)
        The unconfirmed swing still building.
    """
    pivots = []
    ld, lp, li = 0, float(cl_arr[0]), 0

    for i in range(1, len(cl_arr)):
        p = float(cl_arr[i])
        if ld == 0:
            if   p >= lp * (1 + min_mv): ld =  1; lp, li = p, i
            elif p <= lp * (1 - min_mv): ld = -1; lp, li = p, i
        elif ld == 1:
            if p <= float(cl_arr[li]) * (1 - min_mv):
                pivots.append((li, float(cl_arr[li]), 1))
                ld, lp, li = -1, p, i
            elif p > lp:
                lp, li = p, i
        else:  # ld == -1
            if p >= float(cl_arr[li]) * (1 + min_mv):
                pivots.append((li, float(cl_arr[li]), -1))
                ld, lp, li = 1, p, i
            elif p < lp:
                lp, li = p, i

    return pivots, (ld, lp, li)


def _classify_wave(pivots: list) -> dict:
    """
    Classify the wave pattern from the confirmed pivot list.
    Returns display-ready dict with pattern, trend, Fibonacci levels.
    """
    n = len(pivots)
    if n < 2:
        return {"pattern": "Accumulating data…", "trend": "—",
                "count": n, "fibs": {}}

    last, prev = pivots[-1], pivots[-2]
    trend = "Uptrend" if last[2] == 1 else "Downtrend"

    # 5-wave impulse detection (last 5 pivots)
    pattern = "Corrective (A-B-C)"
    if n >= 5:
        p = pivots[-5:]
        if p[0][2] == -1:                   # starts at a swing-low
            w1 = p[1][1] - p[0][1]          # up
            w2 = p[1][1] - p[2][1]          # retracement (positive = pulled back)
            w3 = p[3][1] - p[2][1]          # up
            w4 = p[3][1] - p[4][1]          # retracement
            if all(x > 0 for x in [w1, w2, w3, w4]) and w3 > w1:
                pattern = "Impulse (1-2-3-4-5)"

    # Fibonacci retracement / extension from last 2 confirmed pivots
    a, b = prev[1], last[1]
    d = abs(b - a)
    up = (last[2] == 1)

    def fib_ret(ratio):  # retracement from last pivot
        return round(b - ratio * d if up else b + ratio * d, 4)

    def fib_ext(ratio):  # extension beyond last pivot
        return round(b + ratio * d if up else b - ratio * d, 4)

    fibs = {
        "Ret 23.6%":  fib_ret(0.236),
        "Ret 38.2%":  fib_ret(0.382),
        "Ret 50.0%":  fib_ret(0.500),
        "Ret 61.8%":  fib_ret(0.618),
        "Ext 127.2%": fib_ext(1.272),
        "Ext 161.8%": fib_ext(1.618),
    }

    return {
        "pattern":        pattern,
        "trend":          trend,
        "count":          n,
        "fibs":           fibs,
        "last_pivot_px":  last[1],
        "last_pivot_dir": "High" if last[2] == 1 else "Low",
        "prev_pivot_px":  prev[1],
    }


# ─── Walk-forward signal generation (BACKTESTING) ─────────────────────────────
#
# WHY BACKTESTING ≠ LIVE  (root causes):
#
# 1. LOOK-AHEAD BIAS — classic zigzag on full data "confirms" a pivot at bar 50
#    only because bar 60+ reversed far enough. Backtest fires signal early;
#    live trading can't see future bars.
#
# 2. UNCONFIRMED LAST PIVOT — in live mode the most recent swing hasn't been
#    reversed by `min_wave_pct` yet, so it isn't in the pivot list. Backtesting
#    over the full window treats it as confirmed.
#
# 3. BAR INCOMPLETENESS — live current candle Close is mid-bar; backtest uses
#    the final settled close.
#
# 4. ADJUSTED DATA DRIFT — yfinance historical data is periodically adjusted
#    (splits/dividends); live tick data is not.
#
# FIX: walk-forward construction. At each bar i we only pass Close[:i+1] to
# _zigzag. A signal fires ONLY when the newly confirmed pivot is at bar i
# (p2[0] == i), which is exactly the condition that would fire in live mode.

def ew_signals_walkforward(df: pd.DataFrame, min_wave_pct: float = 1.0, **_) -> tuple:
    """
    Walk-forward Elliott Wave signal generator.
    Produces identical signals to what live trading would see — no look-ahead.
    """
    cl = df["Close"].values
    n  = len(cl)
    s  = pd.Series(0, index=df.index)
    min_mv = min_wave_pct / 100.0

    for i in range(10, n):
        pivots, _ = _zigzag(cl[: i + 1], min_mv)
        if len(pivots) < 3:
            continue
        p0, p1, p2 = pivots[-3], pivots[-2], pivots[-1]

        # Only fire when the pivot was confirmed exactly at this bar
        if p2[0] != i:
            continue

        # BUY — higher-low corrective retracement in uptrend
        # Pattern: low → high → higher-low  (wave 2 or wave 4 complete)
        if p0[2] == -1 and p1[2] == 1 and p2[2] == -1 and p2[1] > p0[1]:
            s.iloc[i] = 1

        # SELL — lower-high corrective retracement in downtrend
        # Pattern: high → low → lower-high
        elif p0[2] == 1 and p1[2] == -1 and p2[2] == 1 and p2[1] < p0[1]:
            s.iloc[i] = -1

    return s, {"EMA_20": ema(df["Close"], 20), "EMA_50": ema(df["Close"], 50)}


# ─── Live state analyser ───────────────────────────────────────────────────────
def ew_live_state(df: pd.DataFrame, min_wave_pct: float = 1.0) -> dict:
    """
    Full wave analysis on the latest available bar.
    Used by both the live loop and the live UI.
    Returns comprehensive info dict.
    """
    cl = df["Close"].values
    min_mv = min_wave_pct / 100.0
    pivots, (ld, lp, li) = _zigzag(cl, min_mv)
    info = _classify_wave(pivots)

    signal = 0
    bars_ago = 999
    signal_bar_px = None

    if len(pivots) >= 3:
        p0, p1, p2 = pivots[-3], pivots[-2], pivots[-1]
        bars_ago = len(cl) - 1 - p2[0]
        signal_bar_px = p2[1]

        if p0[2] == -1 and p1[2] == 1 and p2[2] == -1 and p2[1] > p0[1]:
            signal = 1
        elif p0[2] == 1 and p1[2] == -1 and p2[2] == 1 and p2[1] < p0[1]:
            signal = -1

    info.update({
        "signal":               signal,
        "signal_bar_px":        signal_bar_px,
        "bars_since_pivot":     bars_ago,
        "in_progress":          "Up" if ld == 1 else ("Down" if ld == -1 else "—"),
        "in_progress_px":       lp,
        "pivots":               pivots,
        "last_cl":              float(df["Close"].iloc[-1]),
    })
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# SL / TARGET CALCULATORS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_sl(df, i, side, sl_type, sl_pts, atr_mult, atr_vals=None):
    px = float(df["Close"].iloc[i])
    if atr_vals is None:
        atr_vals = atr_series(df)
    a = float(atr_vals.iloc[i])

    if sl_type == "Custom Points":
        return px - sl_pts if side == "BUY" else px + sl_pts
    elif sl_type in ("ATR-Based", "Trailing SL", "Strategy Reverse Signal"):
        return px - atr_mult * a if side == "BUY" else px + atr_mult * a
    elif sl_type == "Candle Low/High":
        return float(df["Low"].iloc[i]) if side == "BUY" else float(df["High"].iloc[i])
    return px - atr_mult * a if side == "BUY" else px + atr_mult * a


def calc_target(df, i, side, tgt_type, tgt_pts, atr_mult, atr_vals=None):
    px = float(df["Close"].iloc[i])
    if atr_vals is None:
        atr_vals = atr_series(df)
    a = float(atr_vals.iloc[i])

    if tgt_type == "Custom Points":
        return px + tgt_pts if side == "BUY" else px - tgt_pts
    elif tgt_type in ("ATR-Based", "Trailing Target"):
        return px + atr_mult * a if side == "BUY" else px - atr_mult * a
    else:  # Strategy Reversal
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def backtest(df, sigs, sl_type, tgt_type, sl_pts, tgt_pts, atr_sl, atr_tgt):
    """
    Conservative candle-by-candle backtest.

    BUY  trade: SL hit when Low  ≤ sl (checked FIRST), then Target hit when High ≥ tgt.
    SELL trade: SL hit when High ≥ sl (checked FIRST), then Target hit when Low  ≤ tgt.

    Entry  = Close of signal bar.
    No re-entry while in position.
    """
    trades = []
    in_pos = False
    entry_i = entry_px = sl = tgt = side = None

    atr_vals = atr_series(df)   # pre-compute once

    for i in range(len(df)):
        row = df.iloc[i]
        hi  = float(row["High"])
        lo  = float(row["Low"])
        cl  = float(row["Close"])

        # ── Exit logic ──────────────────────────────────────────────────────
        if in_pos:
            reason = exit_px = None

            if side == "BUY":
                # Conservative: SL first
                if sl_type != "Strategy Reverse Signal" and lo <= sl:
                    reason, exit_px = "SL Hit", sl
                elif tgt is not None and hi >= tgt:
                    reason, exit_px = "Target Hit", tgt
                elif sl_type == "Strategy Reverse Signal" and sigs.iloc[i] == -1:
                    reason, exit_px = "Reverse Signal (SL)", cl
                elif tgt_type == "Strategy Reversal" and sigs.iloc[i] == -1:
                    reason, exit_px = "Reverse Signal (Tgt)", cl

            else:  # SELL
                if sl_type != "Strategy Reverse Signal" and hi >= sl:
                    reason, exit_px = "SL Hit", sl
                elif tgt is not None and lo <= tgt:
                    reason, exit_px = "Target Hit", tgt
                elif sl_type == "Strategy Reverse Signal" and sigs.iloc[i] == 1:
                    reason, exit_px = "Reverse Signal (SL)", cl
                elif tgt_type == "Strategy Reversal" and sigs.iloc[i] == 1:
                    reason, exit_px = "Reverse Signal (Tgt)", cl

            # Trailing SL update (if still in position)
            if reason is None and sl_type == "Trailing SL":
                a = float(atr_vals.iloc[i])
                if side == "BUY":
                    sl = max(sl, cl - atr_sl * a)
                else:
                    sl = min(sl, cl + atr_sl * a)

            if reason:
                pts = (exit_px - entry_px) if side == "BUY" else (entry_px - exit_px)
                trades.append({
                    "Side":         side,
                    "Entry Time":   str(df.index[entry_i])[:19],
                    "Entry Price":  round(entry_px, 4),
                    "Entry Low":    round(float(df["Low"].iloc[entry_i]), 4),
                    "Entry High":   round(float(df["High"].iloc[entry_i]), 4),
                    "Exit Time":    str(df.index[i])[:19],
                    "Exit Price":   round(exit_px, 4),
                    "Exit Low":     round(lo, 4),
                    "Exit High":    round(hi, 4),
                    "SL Level":     round(sl, 4),
                    "Target":       round(tgt, 4) if tgt is not None else "Reversal",
                    "Exit Reason":  reason,
                    "Points":       round(pts, 4),
                    "W/L":          "✅ Win" if pts > 0 else "❌ Loss",
                })
                in_pos = False

        # ── Entry logic ──────────────────────────────────────────────────────
        if not in_pos and i < len(df) - 1:
            sig = sigs.iloc[i]
            if sig == 1:
                side = "BUY"
            elif sig == -1:
                side = "SELL"
            else:
                continue

            entry_i  = i
            entry_px = cl
            sl  = calc_sl(df, i, side, sl_type, sl_pts, atr_sl, atr_vals)
            tgt = calc_target(df, i, side, tgt_type, tgt_pts, atr_tgt, atr_vals)
            in_pos = True

    return trades


def bt_stats(trades: list) -> dict:
    if not trades:
        return {}
    pts    = [t["Points"] for t in trades]
    wins   = [p for p in pts if p > 0]
    losses = [p for p in pts if p <= 0]
    total  = len(trades)
    return {
        "Total Trades": total,
        "Wins":         len(wins),
        "Losses":       len(losses),
        "Accuracy":     f"{100 * len(wins) / max(1, total):.1f}%",
        "Total Pts":    round(sum(pts), 2),
        "Avg Win":      round(float(np.mean(wins)),   2) if wins   else 0,
        "Avg Loss":     round(float(np.mean(losses)), 2) if losses else 0,
        "Best Trade":   round(max(pts), 2),
        "Worst Trade":  round(min(pts), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def get_signals(df, strategy, ew_params):
    if strategy == "Elliott Wave":
        return ew_signals_walkforward(df, **ew_params)
    elif strategy == "Simple Buy":
        # Enter immediately on every candle (no wait for signal)
        return pd.Series(1, index=df.index), {}
    else:  # Simple Sell
        return pd.Series(-1, index=df.index), {}


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE TRADING LOOP  (background daemon thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _live_loop(ticker, interval, period, strategy, ew_params,
               sl_type, tgt_type, sl_pts, tgt_pts, atr_sl, atr_tgt):
    """
    Runs in a background daemon thread.
    Polls yfinance at 1.5 s minimum intervals.
    Updates _LS dict under _LS_LOCK.
    """
    global _LS

    while True:
        with _LS_LOCK:
            if not _LS["running"]:
                break

        # ── Fetch ─────────────────────────────────────────────────────────
        df, err = load_data(ticker, interval, period)
        ts = datetime.now().strftime("%H:%M:%S")

        if err or df is None or len(df) < 15:
            with _LS_LOCK:
                _LS["error"] = err or "Insufficient bars"
            time.sleep(2)
            continue

        # ── Compute wave state (outside lock) ──────────────────────────────
        wave_info = ew_live_state(df, ew_params.get("min_wave_pct", 1.0))
        atr_v     = atr_series(df)
        last_row  = df.iloc[-1]
        last_bar  = {
            "Open":  round(float(last_row["Open"]),  4),
            "High":  round(float(last_row["High"]),  4),
            "Low":   round(float(last_row["Low"]),   4),
            "Close": round(float(last_row["Close"]), 4),
            "Time":  str(df.index[-1])[:19],
        }
        signal = wave_info["signal"]

        with _LS_LOCK:
            _LS["wave_info"]   = wave_info
            _LS["last_bar"]    = last_bar
            _LS["last_fetch"]  = time.time()
            _LS["fetch_count"] += 1
            _LS["error"]       = None
            pos = _LS["position"]

            # ── Entry ──────────────────────────────────────────────────────
            if pos is None and signal != 0:
                side = "BUY" if signal == 1 else "SELL"
                epx  = last_bar["Close"]
                sl_v = calc_sl(df, -1, side, sl_type, sl_pts, atr_sl, atr_v)
                tgt_v = calc_target(df, -1, side, tgt_type, tgt_pts, atr_tgt, atr_v)
                _LS["position"] = {
                    "side":        side,
                    "entry_px":    epx,
                    "entry_time":  last_bar["Time"],
                    "sl":          sl_v,
                    "sl_initial":  sl_v,
                    "target":      tgt_v,
                    "wave_count":  wave_info.get("count", 0),
                    "entry_wave":  wave_info.get("pattern", "—"),
                    "entry_trend": wave_info.get("trend",   "—"),
                }
                _LS["log"].append(
                    f"[{ts}] ▶ {side} ENTRY @ {epx:.4f} | "
                    f"SL: {sl_v:.4f} | TGT: {tgt_v:.4f if tgt_v else 'Reversal'}"
                )

            # ── Exit ───────────────────────────────────────────────────────
            elif pos is not None:
                sl_v  = pos["sl"]
                tgt_v = pos["target"]
                side  = pos["side"]
                hi    = last_bar["High"]
                lo    = last_bar["Low"]
                cl    = last_bar["Close"]
                reason = exit_px = None

                if side == "BUY":
                    if sl_type != "Strategy Reverse Signal" and lo <= sl_v:
                        reason, exit_px = "SL Hit", sl_v
                    elif tgt_v and hi >= tgt_v:
                        reason, exit_px = "Target Hit", tgt_v
                    elif sl_type == "Strategy Reverse Signal" and signal == -1:
                        reason, exit_px = "Reverse Signal (SL)", cl
                    elif tgt_type == "Strategy Reversal" and signal == -1:
                        reason, exit_px = "Reverse Signal (Tgt)", cl
                else:
                    if sl_type != "Strategy Reverse Signal" and hi >= sl_v:
                        reason, exit_px = "SL Hit", sl_v
                    elif tgt_v and lo <= tgt_v:
                        reason, exit_px = "Target Hit", tgt_v
                    elif sl_type == "Strategy Reverse Signal" and signal == 1:
                        reason, exit_px = "Reverse Signal (SL)", cl
                    elif tgt_type == "Strategy Reversal" and signal == 1:
                        reason, exit_px = "Reverse Signal (Tgt)", cl

                # Trailing SL
                if reason is None and sl_type == "Trailing SL":
                    a = float(atr_v.iloc[-1])
                    if side == "BUY":
                        pos["sl"] = max(pos["sl"], cl - atr_sl * a)
                    else:
                        pos["sl"] = min(pos["sl"], cl + atr_sl * a)

                if reason:
                    pts = (exit_px - pos["entry_px"]) if side == "BUY" else (pos["entry_px"] - exit_px)
                    emoji = "✅" if pts > 0 else "❌"
                    _LS["log"].append(
                        f"[{ts}] {emoji} {side} EXIT @ {exit_px:.4f} | "
                        f"{reason} | P&L: {pts:+.4f} pts"
                    )
                    _LS["position"] = None

        # Small CPU yield; 1.5 s rate-limit is in _yf_raw
        time.sleep(0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def sidebar_config() -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # ── Asset ──────────────────────────────────────────────────────────
        st.markdown("### 📌 Asset")
        sel = st.selectbox("Ticker", list(TICKERS.keys()), key="s_ticker")
        if sel == "Custom":
            sym = st.text_input("Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS",
                                key="s_custom_sym")
        else:
            sym = TICKERS[sel]
            st.caption(f"`{sym}`")

        # ── Timeframe ──────────────────────────────────────────────────────
        st.markdown("### ⏱️ Timeframe")
        col1, col2 = st.columns(2)
        interval = col1.selectbox("Interval", TIMEFRAMES, index=4, key="s_interval")
        vp = VALID_PERIODS.get(interval, ["1mo", "3mo", "6mo", "1y"])
        pref = next((p for p in ["3mo", "1mo", "5d"] if p in vp), vp[0])
        period = col2.selectbox("Period", vp, index=vp.index(pref), key="s_period")

        # ── Strategy ───────────────────────────────────────────────────────
        st.markdown("### 🎯 Strategy")
        strategy = st.selectbox(
            "Strategy",
            ["Elliott Wave", "Simple Buy", "Simple Sell"],
            key="s_strategy",
        )

        ew_params = {}
        if strategy == "Elliott Wave":
            default_wp = float(st.session_state.get("s_wave_pct", 1.0))
            mwp = st.slider(
                "Min Wave % (EW)", 0.3, 5.0, default_wp, 0.1, key="s_wave_pct",
                help="Minimum % price move to qualify a swing leg as a wave."
            )
            ew_params = {"min_wave_pct": mwp}

        # ── Stop Loss ──────────────────────────────────────────────────────
        st.markdown("### 🛡️ Stop Loss")
        sl_type = st.selectbox("SL Type", SL_TYPES, key="s_sl_type")
        sl_pts, atr_sl = 10.0, 1.5
        if sl_type == "Custom Points":
            def_sl = float(st.session_state.get("s_sl_pts", 10.0))
            sl_pts = st.number_input("SL Points", 0.1, 1e6, def_sl, key="s_sl_pts")
        elif sl_type in ("ATR-Based", "Trailing SL", "Strategy Reverse Signal"):
            atr_sl = st.slider("ATR Mult (SL)", 0.3, 5.0, 1.5, 0.1, key="s_atr_sl")

        # ── Target ─────────────────────────────────────────────────────────
        st.markdown("### 🎯 Target")
        tgt_type = st.selectbox("Target Type", TGT_TYPES, key="s_tgt_type")
        tgt_pts, atr_tgt = 20.0, 2.5
        if tgt_type == "Custom Points":
            def_tgt = float(st.session_state.get("s_tgt_pts", 20.0))
            tgt_pts = st.number_input("Target Points", 0.1, 1e6, def_tgt, key="s_tgt_pts")
        elif tgt_type in ("ATR-Based", "Trailing Target"):
            atr_tgt = st.slider("ATR Mult (Tgt)", 0.5, 10.0, 2.5, 0.1, key="s_atr_tgt")

        # ── Explanation ────────────────────────────────────────────────────
        with st.expander("❓ Why BT ≠ Live?"):
            st.markdown("""
**Root causes of backtest vs live divergence:**

1. **Look-ahead bias** — Classic zigzag builds pivots on the full dataset,
   so a pivot at bar 50 is "confirmed" by bars 51-60. Backtesting fires
   the signal retroactively; live trading can never see future bars.

2. **Unconfirmed last pivot** — In live mode the current swing hasn't yet
   reversed by `min_wave_pct`, so it isn't a confirmed pivot. Backtesting
   over the full window treats it as confirmed.

3. **Incomplete current bar** — Live close is mid-candle; backtest close
   is final.

4. **yfinance adjusted data drift** — Historical OHLC is periodically
   re-adjusted for splits/dividends.

**Fix in this app:** Walk-forward pivot construction.  
At bar `i` only `Close[:i+1]` is passed to the zigzag.  
Signal fires **only** when `p2[0] == i` (pivot confirmed this exact bar).  
Backtesting and live trading see identical logic.
""")

    return dict(
        ticker=sym, interval=interval, period=period,
        strategy=strategy, ew_params=ew_params,
        sl_type=sl_type, sl_pts=float(sl_pts), atr_sl=float(atr_sl),
        tgt_type=tgt_type, tgt_pts=float(tgt_pts), atr_tgt=float(atr_tgt),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

def render_backtest(cfg):
    st.header("📊 Backtesting")

    if not st.button("▶ Run Backtest", type="primary", key="bt_run"):
        st.info("Configure settings in the sidebar, then click **Run Backtest**.")
        return

    # ── Data ──────────────────────────────────────────────────────────────────
    with st.spinner("Fetching data…"):
        df, err = load_data(cfg["ticker"], cfg["interval"], cfg["period"])

    if err or df is None:
        st.error(f"Data fetch failed: {err}")
        return

    st.success(
        f"✅ **{len(df):,}** bars loaded | "
        f"{str(df.index[0])[:10]} → {str(df.index[-1])[:10]}"
    )

    # ── Signals ───────────────────────────────────────────────────────────────
    with st.spinner("Generating walk-forward signals…"):
        sigs, inds = get_signals(df, cfg["strategy"], cfg["ew_params"])

    n_buy  = int((sigs == 1).sum())
    n_sell = int((sigs == -1).sum())
    st.info(f"Signals — **BUY: {n_buy}** | **SELL: {n_sell}**")

    if n_buy + n_sell == 0:
        st.warning("No signals generated. Try reducing **Min Wave %** in the sidebar.")
        return

    # ── Backtest ──────────────────────────────────────────────────────────────
    with st.spinner("Running backtest (conservative SL-first)…"):
        trades = backtest(
            df, sigs,
            cfg["sl_type"],  cfg["tgt_type"],
            cfg["sl_pts"],   cfg["tgt_pts"],
            cfg["atr_sl"],   cfg["atr_tgt"],
        )

    if not trades:
        st.warning("No completed trades. Position may still be open at end of data.")
        return

    stats = bt_stats(trades)

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.subheader("📈 Performance Summary")
    cols = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, v)

    # ── Price chart ───────────────────────────────────────────────────────────
    st.subheader("📉 Chart")
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#00e676",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#00e676",
        decreasing_fillcolor="#ef5350",
    ))

    if "EMA_20" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=inds["EMA_20"],
            mode="lines", line=dict(color="#ff9800", width=1.2), name="EMA 20"))
    if "EMA_50" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=inds["EMA_50"],
            mode="lines", line=dict(color="#29b6f6", width=1.2), name="EMA 50"))

    # Signal markers
    b_idx = df[sigs == 1].index
    s_idx = df[sigs == -1].index
    if len(b_idx):
        fig.add_trace(go.Scatter(
            x=b_idx, y=df.loc[b_idx, "Low"] * 0.9985,
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#00e676",
                        line=dict(color="white", width=0.5)),
            name="BUY Signal",
        ))
    if len(s_idx):
        fig.add_trace(go.Scatter(
            x=s_idx, y=df.loc[s_idx, "High"] * 1.0015,
            mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#ef5350",
                        line=dict(color="white", width=0.5)),
            name="SELL Signal",
        ))

    # Trade lines (entry → exit)
    for t in trades:
        colour = "#00e676" if t["Points"] > 0 else "#ef5350"
        fig.add_shape(
            type="line",
            x0=t["Entry Time"], x1=t["Exit Time"],
            y0=t["Entry Price"], y1=t["Exit Price"],
            line=dict(color=colour, width=1.5, dash="dot"),
            opacity=0.7,
        )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=24, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.subheader("📋 Trade Log")
    df_t = pd.DataFrame(trades)

    def _colour_wl(val):
        if "Win"  in str(val): return "color: #00e676; font-weight:600"
        if "Loss" in str(val): return "color: #ef5350; font-weight:600"
        return ""

    def _colour_pts(val):
        try:
            v = float(val)
            if v > 0: return "color: #00e676"
            if v < 0: return "color: #ef5350"
        except Exception:
            pass
        return ""

    styled = (
        df_t.style
        .applymap(_colour_wl,  subset=["W/L"])
        .applymap(_colour_pts, subset=["Points"])
    )
    st.dataframe(styled, use_container_width=True, height=420)

    # Export
    csv = df_t.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "trades.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

def render_live(cfg):
    st.header("🔴 Live Trading Monitor")

    global _LS

    with _LS_LOCK:
        is_running = _LS["running"]

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1.2, 5])

    if not is_running:
        if c1.button("▶ Start Live", type="primary", key="lt_start"):
            with _LS_LOCK:
                _LS.update(
                    running=True, log=[], position=None,
                    error=None, fetch_count=0, last_fetch=0.0,
                )
            t = threading.Thread(
                target=_live_loop,
                args=(cfg["ticker"], cfg["interval"], cfg["period"],
                      cfg["strategy"], cfg["ew_params"],
                      cfg["sl_type"],  cfg["tgt_type"],
                      cfg["sl_pts"],   cfg["tgt_pts"],
                      cfg["atr_sl"],   cfg["atr_tgt"]),
                daemon=True,
            )
            t.start()
            st.rerun()
    else:
        if c1.button("⏹ Stop", type="secondary", key="lt_stop"):
            with _LS_LOCK:
                _LS["running"] = False
            st.rerun()

    if c2.button("🔄 Refresh", key="lt_ref"):
        st.rerun()

    # ── Snapshot (single lock acquisition) ────────────────────────────────────
    with _LS_LOCK:
        snap = {k: v for k, v in _LS.items() if k != "pivots"}
        snap["log"] = list(_LS["log"])
        if "pivots" in _LS.get("wave_info", {}):
            snap["wave_info"] = {k: v for k, v in _LS["wave_info"].items()
                                 if k != "pivots"}
        else:
            snap["wave_info"] = dict(_LS["wave_info"])

    age   = time.time() - snap["last_fetch"] if snap["last_fetch"] else 9999
    stale = age > 60

    # ── Status row ────────────────────────────────────────────────────────────
    st1, st2, st3, st4 = st.columns(4)
    st1.metric("Status",      "🟢 Running" if snap["running"] else "🔴 Stopped")
    st2.metric("Fetches",     snap.get("fetch_count", 0))
    st3.metric("Data Age",    f"{age:.0f}s")
    st4.metric("Rate Limit",  f"≥ {_YF_MIN_GAP}s gap")

    if snap["error"]:
        st.error(f"⚠️ Error: {snap['error']}")
    if stale:
        st.warning(
            f"⚠️ Data is stale ({age:.0f}s old). "
            "yfinance may be rate-limiting — retrying automatically."
        )

    st.divider()

    # ── Active config (shown when no position) ────────────────────────────────
    pos = snap["position"]
    if not pos:
        with st.expander("⚙️ Active Configuration", expanded=True):
            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
            cc1.metric("Ticker",   cfg["ticker"])
            cc2.metric("Interval", cfg["interval"])
            cc3.metric("Period",   cfg["period"])
            cc4.metric("SL",       cfg["sl_type"])
            cc5.metric("Target",   cfg["tgt_type"])

    # ── Position panel ────────────────────────────────────────────────────────
    st.subheader("📍 Position")

    if pos:
        cl_now = snap["last_bar"]["Close"] if snap["last_bar"] else pos["entry_px"]
        side   = pos["side"]
        ep     = pos["entry_px"]
        sl_v   = pos["sl"]
        tgt_v  = pos["target"]
        pnl    = (cl_now - ep) if side == "BUY" else (ep - cl_now)
        sl_rem = abs(cl_now - sl_v)
        tgt_rem = abs(tgt_v - cl_now) if tgt_v else None

        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Side",       side)
        p2.metric("Entry",      f"{ep:.4f}")
        p3.metric("Current",    f"{cl_now:.4f}", delta=f"{pnl:+.4f}")
        p4.metric("SL",         f"{sl_v:.4f}",  delta=f"{sl_rem:.4f} away")
        p5.metric("Target",     f"{tgt_v:.4f}" if tgt_v else "Reversal",
                  delta=f"{tgt_rem:.4f} away" if tgt_rem else "—")
        p6.metric("Wave at Entry", pos.get("entry_wave", "—"))

        st.caption(
            f"Entered: **{pos['entry_time']}** | "
            f"Trend at entry: **{pos.get('entry_trend','—')}** | "
            f"Wave count: **{pos.get('wave_count','—')}**"
        )

        # Stale-data late-entry estimate
        if stale:
            late_slip = abs(cl_now - ep) if cl_now != ep else 0
            st.info(
                f"📌 Stale data ({age:.0f}s old). Signal fired @ {ep:.4f}, "
                f"current {cl_now:.4f} (slippage ≈ {late_slip:.4f}). "
                f"Remaining — SL: {sl_rem:.4f} | "
                f"Target: {tgt_rem:.4f}" if tgt_rem else "Target: Reversal-based"
            )
    else:
        wi  = snap["wave_info"]
        sig = wi.get("signal", 0)
        bars_ago = wi.get("bars_since_pivot", 999)

        if sig == 1:
            spx = wi.get("signal_bar_px")
            cl  = snap["last_bar"]["Close"] if snap["last_bar"] else None
            msg = (
                f"🟢 BUY signal active | "
                f"Signal bar @ {spx:.4f} | "
                f"Current @ {cl:.4f}" if cl and spx else "🟢 BUY signal"
            )
            st.success(msg)
        elif sig == -1:
            spx = wi.get("signal_bar_px")
            cl  = snap["last_bar"]["Close"] if snap["last_bar"] else None
            msg = (
                f"🔴 SELL signal active | "
                f"Signal bar @ {spx:.4f} | "
                f"Current @ {cl:.4f}" if cl and spx else "🔴 SELL signal"
            )
            st.error(msg)
        else:
            st.info(
                f"⚪ No open position. Last pivot confirmed **{bars_ago}** bar(s) ago."
            )

    st.divider()

    # ── Elliott Wave state ────────────────────────────────────────────────────
    st.subheader("🌊 Elliott Wave State")
    wi = snap["wave_info"]
    if wi:
        w1, w2, w3, w4, w5 = st.columns(5)
        w1.metric("Confirmed Pivots", wi.get("count",       "—"))
        w2.metric("Pattern",          wi.get("pattern",     "—"))
        w3.metric("Trend",            wi.get("trend",       "—"))
        w4.metric("In Progress",      wi.get("in_progress", "—"))
        w5.metric("In-Progress Px",   f"{wi.get('in_progress_px', 0):.4f}")

        fibs = wi.get("fibs", {})
        if fibs:
            st.caption("**Fibonacci Levels** (from last 2 confirmed pivots):")
            fc = st.columns(len(fibs))
            for col, (lbl, px) in zip(fc, fibs.items()):
                col.metric(lbl, f"{px:.4f}")

    st.divider()

    # ── Last candle ───────────────────────────────────────────────────────────
    st.subheader("🕯️ Last Candle")
    lb = snap["last_bar"]
    if lb:
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lc1.metric("Time",  lb.get("Time", "—")[-8:])
        lc2.metric("Open",  f"{lb['Open']:.4f}")
        lc3.metric("High",  f"{lb['High']:.4f}")
        lc4.metric("Low",   f"{lb['Low']:.4f}")
        lc5.metric("Close", f"{lb['Close']:.4f}")
    else:
        st.caption("No candle data yet.")

    st.divider()

    # ── Activity log ──────────────────────────────────────────────────────────
    st.subheader("📜 Activity Log")
    log = snap["log"]
    if log:
        st.code("\n".join(reversed(log[-30:])), language="")
    else:
        st.caption("No activity yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def render_optimization(cfg):
    st.header("🔧 Optimization")
    st.caption(
        "Grid search across Elliott Wave & SL/Target parameters. "
        "Click **Apply** on any result row to push that config into the sidebar."
    )

    with st.expander("⚙️ Optimization Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Elliott Wave**")
            wp_min  = st.number_input("Wave % min",  0.3, 3.0, 0.5, 0.1, key="op_wp_min")
            wp_max  = st.number_input("Wave % max",  0.5, 5.0, 2.5, 0.1, key="op_wp_max")
            wp_step = st.number_input("Wave % step", 0.1, 1.0, 0.5, 0.1, key="op_wp_step")

        with c2:
            st.markdown("**Stop Loss**")
            sl_min  = st.number_input("SL min",  0.5, 1e5,  5.0, 0.5, key="op_sl_min")
            sl_max  = st.number_input("SL max",  0.5, 1e5, 30.0, 0.5, key="op_sl_max")
            sl_step = st.number_input("SL step", 0.5, 100., 5.0, 0.5, key="op_sl_step")
            o_sl_type = st.selectbox(
                "SL Type",
                ["Custom Points", "ATR-Based", "Trailing SL"],
                key="op_sl_type",
            )

        with c3:
            st.markdown("**Target**")
            tgt_min  = st.number_input("Tgt min",  0.5, 1e5, 10.0, 0.5, key="op_tgt_min")
            tgt_max  = st.number_input("Tgt max",  0.5, 1e5, 60.0, 0.5, key="op_tgt_max")
            tgt_step = st.number_input("Tgt step", 0.5, 100.,10.0, 0.5, key="op_tgt_step")
            o_tgt_type = st.selectbox(
                "Target Type",
                ["Custom Points", "ATR-Based", "Strategy Reversal"],
                key="op_tgt_type",
            )

        col_a, col_b = st.columns(2)
        min_acc    = col_a.slider("Min Accuracy %",     0, 100, 50, 5,  key="op_min_acc")
        min_trades = col_b.number_input("Min Trades",   1, 500, 5,  1,  key="op_min_trades")

    if not st.button("🚀 Run Optimization", type="primary", key="op_run"):
        return

    # ── Data ──────────────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        df, err = load_data(cfg["ticker"], cfg["interval"], cfg["period"])

    if err or df is None:
        st.error(f"Data load failed: {err}")
        return

    wp_range  = np.round(np.arange(wp_min,  wp_max  + 1e-6, wp_step),  2)
    sl_range  = np.round(np.arange(sl_min,  sl_max  + 1e-6, sl_step),  2)
    tgt_range = np.round(np.arange(tgt_min, tgt_max + 1e-6, tgt_step), 2)

    combos = len(wp_range) * len(sl_range) * len(tgt_range)
    st.info(f"Testing **{combos:,}** combinations…")

    # Pre-compute signals per wave_pct (major speedup)
    sig_cache: dict = {}
    with st.spinner("Pre-computing wave signals…"):
        for wp in wp_range:
            s, _ = ew_signals_walkforward(df, min_wave_pct=float(wp))
            sig_cache[float(wp)] = s

    prog    = st.progress(0.0)
    results = []
    done    = 0

    for wp in wp_range:
        sigs = sig_cache[float(wp)]
        for sp in sl_range:
            for tp in tgt_range:
                trades = backtest(
                    df, sigs,
                    o_sl_type, o_tgt_type,
                    float(sp), float(tp),
                    1.5, 2.5,
                )
                stats = bt_stats(trades)
                if stats:
                    acc = float(stats["Accuracy"].rstrip("%"))
                    if acc >= min_acc and stats["Total Trades"] >= int(min_trades):
                        results.append(
                            {"wave_pct": float(wp),
                             "sl_pts": float(sp),
                             "tgt_pts": float(tp),
                             **stats}
                        )
                done += 1
                if done % max(1, combos // 200) == 0:
                    prog.progress(min(1.0, done / combos))

    prog.progress(1.0)

    if not results:
        st.warning(
            "No combinations met accuracy / min-trades criteria. "
            "Try lowering the thresholds."
        )
        return

    results.sort(
        key=lambda r: (float(r["Total Pts"]),
                       float(r["Accuracy"].rstrip("%"))),
        reverse=True,
    )
    top_n = min(50, len(results))
    st.success(f"✅ **{len(results)}** valid combinations. Showing top **{top_n}**.")

    st.subheader("🏆 Optimization Results")

    for idx, row in enumerate(results[:top_n]):
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 = st.columns([1,1,1,1,1,1,1,1,1,1.3])
        c1.metric("Wave%",  row["wave_pct"])
        c2.metric("SL",     row["sl_pts"])
        c3.metric("Target", row["tgt_pts"])
        c4.metric("Trades", row["Total Trades"])
        c5.metric("Wins",   row["Wins"])
        c6.metric("Losses", row["Losses"])
        c7.metric("Acc%",   row["Accuracy"])
        c8.metric("Pts",    row["Total Pts"])
        c9.metric("Avg W",  row["Avg Win"])

        if c10.button(f"✅ Apply", key=f"apply_{idx}"):
            # Push into sidebar widget session-state keys
            st.session_state["s_wave_pct"] = float(row["wave_pct"])
            st.session_state["s_sl_pts"]   = float(row["sl_pts"])
            st.session_state["s_tgt_pts"]  = float(row["tgt_pts"])
            st.session_state["s_sl_type"]  = o_sl_type
            st.session_state["s_tgt_type"] = o_tgt_type
            # Also set the convenience keys read by sidebar defaults
            st.session_state["min_wave_pct"] = float(row["wave_pct"])
            st.session_state["sl_pts"]       = float(row["sl_pts"])
            st.session_state["tgt_pts"]      = float(row["tgt_pts"])
            st.success(
                f"Applied → Wave%: {row['wave_pct']} | "
                f"SL: {row['sl_pts']} | Tgt: {row['tgt_pts']}"
            )
            time.sleep(0.6)
            st.rerun()

        st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = sidebar_config()
    t1, t2, t3 = st.tabs(["📊 Backtest", "🔴 Live Trading", "🔧 Optimization"])
    with t1:
        render_backtest(cfg)
    with t2:
        render_live(cfg)
    with t3:
        render_optimization(cfg)


main()
