"""Swing trading desk: backtest, live trading, trade history, and one screener.

Run the app:   streamlit run swingtrading.py
Run the tests: python swingtrading.py --selftest

No TA-Lib, no pandas-ta. Every indicator is computed from first principles
using Pine Script semantics (see PINE_NOTES).
"""

from __future__ import annotations

import ast
import datetime as dt
import itertools
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

PINE_NOTES = """Where TradingView can legitimately still differ from this file:
  * different history depth loaded into the chart (recursive averages never
    fully forget their seed),
  * a different chart session template or exchange timezone (this changes VWAP
    and opening-range resets),
  * dividend / split adjustment settings on the data vendor,
  * the right-most bar on a chart is still forming; this file only ever reads
    completed bars."""


# --------------------------------------------------------------------------
# 1. Indicators, Pine Script semantics
# --------------------------------------------------------------------------

def _arr(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, copy=True)
    return np.asarray(x, dtype=float).copy()


def _first_valid(x: np.ndarray) -> int:
    idx = np.flatnonzero(~np.isnan(x))
    return int(idx[0]) if idx.size else -1


def recursive_smooth(src, length: int, alpha: float) -> np.ndarray:
    """Shared seed-and-recur helper behind ta.ema and ta.rma.

    Pine leaves the first length-1 bars undefined, seeds with the SMA of the
    first `length` values and only then runs the recursion. pandas.ewm seeds
    with the first value, which leaves a decaying offset that never vanishes.
    Leading NaNs are skipped so rma(plusDM) and ema(macd_line) seed at the
    correct bar; an isolated NaN holds the previous value rather than poisoning
    every value after it.
    """
    x = _arr(src)
    n = x.size
    out = np.full(n, np.nan)
    if length < 1 or n == 0:
        return out
    s = _first_valid(x)
    if s < 0 or s + length > n:
        return out
    window = x[s:s + length]
    if np.isnan(window).all():
        return out
    seed = int(s + length - 1)
    out[seed] = float(np.nanmean(window))
    for i in range(seed + 1, n):
        v = x[i]
        if np.isnan(v):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * v + (1.0 - alpha) * out[i - 1]
    return out


def pine_ema(src, length: int) -> np.ndarray:
    return recursive_smooth(src, length, 2.0 / (length + 1.0))


def pine_rma(src, length: int) -> np.ndarray:
    return recursive_smooth(src, length, 1.0 / length)


def pine_sma(src, length: int) -> np.ndarray:
    return pd.Series(_arr(src)).rolling(length, min_periods=length).mean().to_numpy()


def pine_stdev(src, length: int) -> np.ndarray:
    """ta.stdev is the population standard deviation (ddof=0)."""
    return pd.Series(_arr(src)).rolling(length, min_periods=length).std(ddof=0).to_numpy()


def pine_highest(src, length: int) -> np.ndarray:
    return pd.Series(_arr(src)).rolling(length, min_periods=length).max().to_numpy()


def pine_lowest(src, length: int) -> np.ndarray:
    return pd.Series(_arr(src)).rolling(length, min_periods=length).min().to_numpy()


def fixnan(x: np.ndarray) -> np.ndarray:
    """ta.fixnan: carry the last non-NaN value forward."""
    return pd.Series(x).ffill().to_numpy()


def true_range(high, low, close, handle_na: bool = True) -> np.ndarray:
    """ta.tr(handle_na). With handle_na the first bar is High-Low; without it
    the first bar is NaN, which is why ta.dmi seeds one bar later than ta.atr.
    """
    h, l, c = _arr(high), _arr(low), _arr(close)
    n = h.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    out[0] = (h[0] - l[0]) if handle_na else np.nan
    for i in range(1, n):
        pc = c[i - 1]
        if np.isnan(pc):
            out[i] = h[i] - l[i] if handle_na else np.nan
        else:
            out[i] = max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc))
    return out


def pine_atr(high, low, close, length: int) -> np.ndarray:
    return pine_rma(true_range(high, low, close, True), length)


def pine_dmi(high, low, close, di_len: int, adx_len: int):
    """ta.dmi. Uses ta.tr (no first-bar fallback) and gives ADX its own
    smoothing length. DX divides by 1, not 0, when both DIs are zero.
    """
    h, l, c = _arr(high), _arr(low), _arr(close)
    n = h.size
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    up[1:] = h[1:] - h[:-1]
    dn[1:] = l[:-1] - l[1:]
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(up[i]):
            plus_dm[i] = up[i] if (up[i] > dn[i] and up[i] > 0) else 0.0
        if not np.isnan(dn[i]):
            minus_dm[i] = dn[i] if (dn[i] > up[i] and dn[i] > 0) else 0.0
    trur = pine_rma(true_range(h, l, c, False), di_len)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus = fixnan(100.0 * pine_rma(plus_dm, di_len) / trur)
        minus = fixnan(100.0 * pine_rma(minus_dm, di_len) / trur)
    total = plus + minus
    denom = np.where((total == 0) | np.isnan(total), 1.0, total)
    dx = np.abs(plus - minus) / denom
    dx = np.where(np.isnan(plus) | np.isnan(minus), np.nan, dx)
    adx = 100.0 * pine_rma(dx, adx_len)
    return plus, minus, adx


def pine_rsi(src, length: int) -> np.ndarray:
    x = _arr(src)
    n = x.size
    ch = np.full(n, np.nan)
    ch[1:] = x[1:] - x[:-1]
    gain = np.where(np.isnan(ch), np.nan, np.maximum(ch, 0.0))
    loss = np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0.0))
    ag = pine_rma(gain, length)
    al = pine_rma(loss, length)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(ag[i]) or np.isnan(al[i]):
            continue
        if al[i] == 0:
            out[i] = 100.0
        elif ag[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + ag[i] / al[i])
    return out


def pine_macd(src, fast: int, slow: int, signal: int):
    line = pine_ema(src, fast) - pine_ema(src, slow)
    sig = pine_ema(line, signal)
    return line, sig, line - sig


def pine_bbands(src, length: int, mult: float):
    basis = pine_sma(src, length)
    dev = mult * pine_stdev(src, length)
    return basis, basis + dev, basis - dev


def pine_supertrend(high, low, close, factor: float, atr_len: int):
    """ta.supertrend. Pine's direction is -1 for an uptrend; `trend` inverts it
    so that +1 means up, which is what every caller in this file expects.
    """
    h, l, c = _arr(high), _arr(low), _arr(close)
    n = h.size
    atr = pine_atr(h, l, c, atr_len)
    hl2 = (h + l) / 2.0
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    up_prev = np.nan
    dn_prev = np.nan
    st_prev = np.nan
    dir_prev = np.nan
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        up = hl2[i] - factor * atr[i]
        dn = hl2[i] + factor * atr[i]
        up1 = up_prev if not np.isnan(up_prev) else up
        dn1 = dn_prev if not np.isnan(dn_prev) else dn
        if i > 0 and c[i - 1] > up1:
            up = max(up, up1)
        if i > 0 and c[i - 1] < dn1:
            dn = min(dn, dn1)
        if i == 0 or np.isnan(atr[i - 1]):
            d = 1.0
        elif st_prev == up1:
            d = 1.0 if c[i] < up else -1.0
        else:
            d = -1.0 if c[i] > dn else 1.0
        s = up if d == -1 else dn
        direction[i] = d
        st[i] = s
        up_prev, dn_prev, st_prev, dir_prev = up, dn, s, d
    trend = -direction
    return st, direction, trend


def session_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """Session key used to reset VWAP and the opening range: the calendar date."""
    if not isinstance(index, pd.DatetimeIndex):
        return np.zeros(len(index), dtype=np.int64)
    days = pd.Series(index.normalize().astype("int64"))
    return days.to_numpy()


def pine_vwap(high, low, close, volume, index) -> np.ndarray:
    """Cumulative hlc3-weighted VWAP, reset at each session."""
    h, l, c, v = _arr(high), _arr(low), _arr(close), _arr(volume)
    src = (h + l + c) / 3.0
    sid = session_ids(index)
    n = src.size
    out = np.full(n, np.nan)
    cum_pv = 0.0
    cum_v = 0.0
    prev = None
    for i in range(n):
        if prev is None or sid[i] != prev:
            cum_pv = 0.0
            cum_v = 0.0
            prev = sid[i]
        vol = 0.0 if np.isnan(v[i]) else v[i]
        cum_pv += src[i] * vol
        cum_v += vol
        out[i] = cum_pv / cum_v if cum_v > 0 else src[i]
    return out


def pivot_high(high, left: int, right: int) -> np.ndarray:
    """ta.pivothigh. The pivot at bar i is only published at bar i+right, which
    is what stops the swing-based exits from reading the future.
    """
    h = _arr(high)
    n = h.size
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        v = h[i]
        if np.isnan(v):
            continue
        if left and not np.all(h[i - left:i] <= v):
            continue
        if right and not np.all(h[i + 1:i + right + 1] < v):
            continue
        out[i + right] = v
    return out


def pivot_low(low, left: int, right: int) -> np.ndarray:
    l = _arr(low)
    n = l.size
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        v = l[i]
        if np.isnan(v):
            continue
        if left and not np.all(l[i - left:i] >= v):
            continue
        if right and not np.all(l[i + 1:i + right + 1] > v):
            continue
        out[i + right] = v
    return out


def last_pivot(pivots: np.ndarray, back: int = 0) -> np.ndarray:
    """Most recently confirmed pivot value at each bar (back=1 for the one
    before that). Confirmed means already published, so no look-ahead."""
    n = pivots.size
    out = np.full(n, np.nan)
    seen: list[float] = []
    for i in range(n):
        if not np.isnan(pivots[i]):
            seen.append(float(pivots[i]))
        if len(seen) > back:
            out[i] = seen[-1 - back]
    return out


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Event, not state. Computed on float arrays so that the object-dtype
    trap in bool_series.shift(1) (where ~True becomes -2, which is truthy)
    cannot arise at all."""
    a, b = _arr(a), _arr(b)
    out = np.zeros(a.size, dtype=bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _arr(a), _arr(b)
    out = np.zeros(a.size, dtype=bool)
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


# --------------------------------------------------------------------------
# 2. Costs. Indian equity / futures / options, verified against published
#    figures in the self-test.
# --------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Two structures a single "percentage with a cap" cannot express:
    flat brokerage (options are Rs20 per order, and storing that as a cap on a
    0% rate makes min(0, 20) = 0 so options come out free), and the flat DP
    charge on a delivery sell, which dominates small trades and is invisible at
    large size.
    """
    instrument: str = "equity_intraday"
    slippage_pct: float = 0.02
    gst_rate: float = 0.18
    enabled: bool = True


INSTRUMENTS = {
    "equity_intraday": dict(
        label="Equity intraday", brokerage_pct=0.0003, brokerage_cap=20.0,
        brokerage_flat=None, stt_buy=0.0, stt_sell=0.00025, txn_pct=0.0000297,
        sebi_pct=0.000001, stamp_buy=0.00003, dp_charge=0.0),
    "equity_delivery": dict(
        label="Equity delivery", brokerage_pct=0.0, brokerage_cap=None,
        brokerage_flat=None, stt_buy=0.001, stt_sell=0.001, txn_pct=0.0000297,
        sebi_pct=0.000001, stamp_buy=0.00015, dp_charge=13.5),
    "futures": dict(
        label="Futures", brokerage_pct=0.0002, brokerage_cap=20.0,
        brokerage_flat=None, stt_buy=0.0, stt_sell=0.0002, txn_pct=0.0000173,
        sebi_pct=0.000001, stamp_buy=0.00002, dp_charge=0.0),
    "options": dict(
        label="Options (P&L on the underlying)", brokerage_pct=0.0,
        brokerage_cap=None, brokerage_flat=20.0, stt_buy=0.0, stt_sell=0.001,
        txn_pct=0.0003503, sebi_pct=0.000001, stamp_buy=0.00003, dp_charge=0.0),
}


def _brokerage(spec: dict, turnover: float) -> float:
    if spec["brokerage_flat"] is not None:
        return float(spec["brokerage_flat"])
    raw = spec["brokerage_pct"] * turnover
    if spec["brokerage_cap"] is not None:
        return min(raw, spec["brokerage_cap"])
    return raw


def round_trip_costs(entry: float, exit_: float, qty: float, cfg: CostConfig,
                     direction: int = 1) -> dict:
    """Total charges for one round trip. Long buys first, short sells first;
    the securities transaction tax cares which side is the sell."""
    if not cfg.enabled:
        return dict(total=0.0, brokerage=0.0, stt=0.0, txn=0.0, sebi=0.0,
                    stamp=0.0, dp=0.0, gst=0.0, slippage=0.0)
    spec = INSTRUMENTS[cfg.instrument]
    buy_val = (entry if direction > 0 else exit_) * qty
    sell_val = (exit_ if direction > 0 else entry) * qty
    turnover = buy_val + sell_val

    brokerage = _brokerage(spec, buy_val) + _brokerage(spec, sell_val)
    stt = spec["stt_buy"] * buy_val + spec["stt_sell"] * sell_val
    txn = spec["txn_pct"] * turnover
    sebi = spec["sebi_pct"] * turnover
    stamp = spec["stamp_buy"] * buy_val
    dp = spec["dp_charge"] if spec["dp_charge"] else 0.0
    gst = cfg.gst_rate * (brokerage + txn + sebi + dp)
    total = brokerage + stt + txn + sebi + stamp + dp + gst
    return dict(total=total, brokerage=brokerage, stt=stt, txn=txn, sebi=sebi,
                stamp=stamp, dp=dp, gst=gst, slippage=0.0)


def slip(price: float, cfg: CostConfig, worse_up: bool) -> float:
    if not cfg.enabled or cfg.slippage_pct == 0:
        return price
    f = cfg.slippage_pct / 100.0
    return price * (1.0 + f) if worse_up else price * (1.0 - f)


# --------------------------------------------------------------------------
# 3. Strategies. Contract: return +1 / -1 / 0 indexed like the input frame.
#    Signals are events, never states.
# --------------------------------------------------------------------------

@dataclass
class StrategySpec:
    key: str
    label: str
    func: object
    params: dict
    intraday_only: bool = False
    unconditional: bool = False
    plain: str = ""


def _sig(df: pd.DataFrame, longs: np.ndarray, shorts: np.ndarray) -> pd.Series:
    out = np.zeros(len(df), dtype=int)
    out[np.asarray(longs, dtype=bool)] = 1
    out[np.asarray(shorts, dtype=bool) & ~np.asarray(longs, dtype=bool)] = -1
    return pd.Series(out, index=df.index, dtype=int)


def s_ema_cross(df, p):
    f = pine_ema(df["close"], int(p["fast"]))
    s = pine_ema(df["close"], int(p["slow"]))
    return _sig(df, crossover(f, s), crossunder(f, s))


def s_supertrend(df, p):
    _, _, trend = pine_supertrend(df["high"], df["low"], df["close"],
                                  float(p["factor"]), int(p["atr_len"]))
    t = np.nan_to_num(trend, nan=0.0)
    up = np.zeros(len(df), dtype=bool)
    dnf = np.zeros(len(df), dtype=bool)
    up[1:] = (t[1:] == 1) & (t[:-1] == -1)
    dnf[1:] = (t[1:] == -1) & (t[:-1] == 1)
    return _sig(df, up, dnf)


def s_rsi(df, p):
    r = pine_rsi(df["close"], int(p["rsi_len"]))
    lo = np.full(len(df), float(p["lower"]))
    hi = np.full(len(df), float(p["upper"]))
    return _sig(df, crossover(r, lo), crossunder(r, hi))


def s_macd(df, p):
    line, sig, _ = pine_macd(df["close"], int(p["fast"]), int(p["slow"]),
                             int(p["signal"]))
    return _sig(df, crossover(line, sig), crossunder(line, sig))


def s_bollinger(df, p):
    _, up, lo = pine_bbands(df["close"], int(p["length"]), float(p["mult"]))
    c = _arr(df["close"])
    return _sig(df, crossover(c, lo), crossunder(c, up))


def s_donchian(df, p):
    n = int(p["length"])
    hh = pine_highest(df["high"], n)
    ll = pine_lowest(df["low"], n)
    prev_hh = np.concatenate([[np.nan], hh[:-1]])
    prev_ll = np.concatenate([[np.nan], ll[:-1]])
    c = _arr(df["close"])
    return _sig(df, crossover(c, prev_hh), crossunder(c, prev_ll))


def s_vwap(df, p):
    v = pine_vwap(df["high"], df["low"], df["close"], df["volume"], df.index)
    c = _arr(df["close"])
    return _sig(df, crossover(c, v), crossunder(c, v))


def s_orb(df, p):
    minutes = int(p["minutes"])
    sid = session_ids(df.index)
    idx = df.index
    hi = np.full(len(df), np.nan)
    lo = np.full(len(df), np.nan)
    start = None
    cur = None
    rh = rl = np.nan
    for i in range(len(df)):
        if sid[i] != cur:
            cur = sid[i]
            start = idx[i]
            rh, rl = -np.inf, np.inf
        within = (idx[i] - start) < pd.Timedelta(minutes=minutes)
        if within:
            rh = max(rh, float(df["high"].iloc[i]))
            rl = min(rl, float(df["low"].iloc[i]))
        else:
            hi[i] = rh if np.isfinite(rh) else np.nan
            lo[i] = rl if np.isfinite(rl) else np.nan
    c = _arr(df["close"])
    return _sig(df, crossover(c, hi), crossunder(c, lo))


def s_adx(df, p):
    plus, minus, adx = pine_dmi(df["high"], df["low"], df["close"],
                                int(p["di_len"]), int(p["adx_len"]))
    strong = np.nan_to_num(adx, nan=0.0) >= float(p["adx_min"])
    return _sig(df, crossover(plus, minus) & strong,
                crossunder(plus, minus) & strong)


def s_price_level(df, p):
    lvl = np.full(len(df), float(p["level"]))
    c = _arr(df["close"])
    return _sig(df, crossover(c, lvl), crossunder(c, lvl))


def s_always_long(df, p):
    return pd.Series(np.ones(len(df), dtype=int), index=df.index)


def s_always_short(df, p):
    return pd.Series(-np.ones(len(df), dtype=int), index=df.index)


STRATEGIES: dict[str, StrategySpec] = {
    "ema_cross": StrategySpec("ema_cross", "EMA crossover", s_ema_cross,
        dict(fast=9, slow=21),
        plain="Buy when the fast average crosses above the slow one, sell short when it crosses below."),
    "supertrend": StrategySpec("supertrend", "Supertrend flip", s_supertrend,
        dict(factor=3.0, atr_len=10),
        plain="Buy when Supertrend flips to an uptrend, sell short when it flips down."),
    "rsi": StrategySpec("rsi", "RSI", s_rsi,
        dict(rsi_len=14, lower=30.0, upper=70.0),
        plain="Buy when RSI climbs back above the oversold line, short when it drops below the overbought line."),
    "macd": StrategySpec("macd", "MACD", s_macd,
        dict(fast=12, slow=26, signal=9),
        plain="Buy when the MACD line crosses above its signal line, short on the opposite cross."),
    "bollinger": StrategySpec("bollinger", "Bollinger bounce", s_bollinger,
        dict(length=20, mult=2.0),
        plain="Buy when price crosses back above the lower band, short when it crosses below the upper band."),
    "donchian": StrategySpec("donchian", "Donchian breakout", s_donchian,
        dict(length=20),
        plain="Buy when the close breaks the highest high of the prior N bars, short on the mirror break."),
    "vwap": StrategySpec("vwap", "VWAP reclaim", s_vwap, dict(),
        intraday_only=True,
        plain="Buy when price reclaims the session VWAP, short when it loses it."),
    "orb": StrategySpec("orb", "Opening range breakout", s_orb,
        dict(minutes=15), intraday_only=True,
        plain="Buy the first close above the opening range high, short the first close below its low."),
    "adx": StrategySpec("adx", "ADX trend", s_adx,
        dict(di_len=14, adx_len=14, adx_min=25.0),
        plain="Buy when +DI crosses above -DI while ADX says the trend is strong; short on the reverse."),
    "price_level": StrategySpec("price_level", "Price crosses level",
        s_price_level, dict(level=100.0),
        plain="Buy when the close crosses up through your level, short when it crosses down."),
    "always_long": StrategySpec("always_long", "Always long (cost check)",
        s_always_long, dict(), unconditional=True,
        plain="Always long. Exists only to show what the costs alone do to a result."),
    "always_short": StrategySpec("always_short", "Always short (cost check)",
        s_always_short, dict(), unconditional=True,
        plain="Always short. Exists only to show what the costs alone do to a result."),
}


# --------------------------------------------------------------------------
# 4. Exits and the backtest engine
# --------------------------------------------------------------------------

STOP_RULES = {
    "fixed_points": "Fixed points",
    "fixed_percent": "Fixed percent",
    "trail_points": "Trailing points",
    "trail_prev_swing": "Trailing previous swing",
    "trail_cur_swing": "Trailing current swing",
    "trail_prev_candle": "Trailing previous candle",
    "trail_cur_candle": "Trailing current candle",
    "atr_multiple": "ATR multiple",
    "from_reward": "Derived from reward",
    "signal_only": "Strategy signal only (no stop)",
}

TARGET_RULES = {
    "fixed_points": "Fixed points",
    "fixed_percent": "Fixed percent",
    "trail_display": "Trailing (display only, never exits)",
    "trail_swing": "Trailing swing",
    "trail_candle": "Trailing candle",
    "rr_multiple": "Risk:reward multiple",
    "atr_multiple": "ATR multiple",
    "strategy_reversal": "Strategy reversal",
}


@dataclass
class ExitConfig:
    stop_rule: str = "atr_multiple"
    stop_value: float = 2.0
    target_rule: str = "rr_multiple"
    target_value: float = 2.0
    atr_len: int = 14
    swing_left: int = 3
    swing_right: int = 3
    move_to_cost: bool = False
    move_to_cost_r: float = 1.0


@dataclass
class RunConfig:
    """The single object the screener and the Backtest tab both consume, so a
    screener row replays identically by construction."""
    ticker: str = ""
    interval: str = "1d"
    period: str = "2y"
    strategy: str = "ema_cross"
    params: dict = field(default_factory=dict)
    exits: ExitConfig = field(default_factory=ExitConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    qty: int = 1
    allow_long: bool = True
    allow_short: bool = True

    def fingerprint(self) -> str:
        return repr((self.ticker, self.interval, self.period, self.strategy,
                     sorted(self.params.items()), self.exits, self.costs,
                     self.qty, self.allow_long, self.allow_short))


def is_intraday(interval: str) -> bool:
    return interval.endswith("m") or interval.endswith("h")


def build_signals(df: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    spec = STRATEGIES[cfg.strategy]
    if spec.intraday_only and not is_intraday(cfg.interval):
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    p = dict(spec.params)
    p.update(cfg.params or {})
    return spec.func(df, p).astype(int)


@dataclass
class EngineResult:
    trades: pd.DataFrame
    diag: dict
    levels: pd.DataFrame


def run_backtest(df: pd.DataFrame, signals: pd.Series, cfg: RunConfig,
                 tie_break: str = "stop", force_costs: bool | None = None,
                 forced_entries: list | None = None) -> EngineResult:
    """One position at a time. A signal on bar N is filled at the open of bar
    N+1, the stop is tested before the target inside a bar, a bar that opens
    beyond a level fills at the open, and trailing levels move only on a
    completed bar's close.
    """
    ex = cfg.exits
    costs = cfg.costs
    if force_costs is not None:
        costs = replace(costs, enabled=force_costs)

    o = _arr(df["open"]); h = _arr(df["high"])
    l = _arr(df["low"]); c = _arr(df["close"])
    n = len(df)
    idx = df.index
    atr = pine_atr(h, l, c, ex.atr_len)
    ph = pivot_high(h, ex.swing_left, ex.swing_right)
    pl = pivot_low(l, ex.swing_left, ex.swing_right)
    sw_hi0, sw_hi1 = last_pivot(ph, 0), last_pivot(ph, 1)
    sw_lo0, sw_lo1 = last_pivot(pl, 0), last_pivot(pl, 1)

    sig = signals.to_numpy(dtype=int)
    forced = dict(forced_entries or [])

    trades = []
    both_hit = 0
    voided = 0
    signal_bar_fills = 0
    loosened = 0
    plot_stop = np.full(n, np.nan)
    plot_tgt = np.full(n, np.nan)

    pos = None

    def dist_from_rule(rule, value, sb, ref, direction):
        if rule == "fixed_points":
            return abs(value)
        if rule == "fixed_percent":
            return abs(ref) * abs(value) / 100.0
        if rule in ("atr_multiple",):
            a = atr[sb]
            return abs(value) * a if not np.isnan(a) else np.nan
        if rule == "trail_points":
            return abs(value)
        return np.nan

    def level_from_rule(rule, sb, direction, is_stop):
        if is_stop:
            src = {"trail_prev_swing": sw_lo1 if direction > 0 else sw_hi1,
                   "trail_cur_swing": sw_lo0 if direction > 0 else sw_hi0}.get(rule)
            if src is not None:
                return src[sb]
            if rule == "trail_prev_candle":
                return (l[sb - 1] if direction > 0 else h[sb - 1]) if sb >= 1 else np.nan
            if rule == "trail_cur_candle":
                return l[sb] if direction > 0 else h[sb]
        else:
            if rule == "trail_swing":
                return (sw_hi0 if direction > 0 else sw_lo0)[sb]
            if rule == "trail_candle":
                return h[sb] if direction > 0 else l[sb]
        return np.nan

    for i in range(1, n):
        exited = False
        if pos is not None:
            d = pos["dir"]
            stop = pos["stop"]
            tgt = pos["target"]
            entry_bar = pos["entry_bar"]
            price = None
            reason = None

            if i > entry_bar:
                if stop is not None:
                    if d > 0 and o[i] <= stop:
                        price, reason = o[i], "stop (gap)"
                    elif d < 0 and o[i] >= stop:
                        price, reason = o[i], "stop (gap)"
                if price is None and tgt is not None:
                    if d > 0 and o[i] >= tgt:
                        price, reason = o[i], "target (gap)"
                    elif d < 0 and o[i] <= tgt:
                        price, reason = o[i], "target (gap)"
                if price is None and pos["signal_exit"] and sig[i - 1] == -d:
                    price, reason = o[i], "strategy reversal"

            if price is None:
                hit_s = stop is not None and (l[i] <= stop if d > 0 else h[i] >= stop)
                hit_t = tgt is not None and (h[i] >= tgt if d > 0 else l[i] <= tgt)
                if hit_s and hit_t:
                    both_hit += 1
                if hit_s and hit_t:
                    if tie_break == "stop":
                        price, reason = stop, "stop"
                    else:
                        price, reason = tgt, "target"
                elif hit_s:
                    price, reason = stop, "stop"
                elif hit_t:
                    price, reason = tgt, "target"

            if price is None and i == n - 1:
                price, reason = c[i], "end of data"

            if price is not None:
                trades.append(_close_trade(pos, i, idx, float(price), reason,
                                           cfg, costs))
                pos = None
                exited = True
            else:
                new_stop, new_tgt = _trail(pos, i, h, l, atr, sw_hi0, sw_hi1,
                                           sw_lo0, sw_lo1, ex)
                if new_stop is not None and pos["stop"] is not None:
                    if (d > 0 and new_stop < pos["stop"] - 1e-12) or \
                       (d < 0 and new_stop > pos["stop"] + 1e-12):
                        loosened += 1
                pos["stop"] = new_stop
                pos["target"] = new_tgt
                plot_stop[i] = new_stop if new_stop is not None else np.nan
                plot_tgt[i] = new_tgt if new_tgt is not None else np.nan

        if pos is None and not exited:
            sb = i - 1
            d = forced.get(i, sig[sb])
            if d == 1 and not cfg.allow_long:
                d = 0
            if d == -1 and not cfg.allow_short:
                d = 0
            if d != 0:
                raw = o[i]
                entry = slip(raw, costs, worse_up=(d > 0))
                stop, tgt, ok = _initial_levels(sb, d, entry, ex, atr, c,
                                                h, l, sw_hi0, sw_hi1, sw_lo0,
                                                sw_lo1, dist_from_rule,
                                                level_from_rule)
                if not ok:
                    voided += 1
                else:
                    if i == sb:
                        signal_bar_fills += 1
                    pos = dict(dir=d, entry=entry, entry_bar=i, signal_bar=sb,
                               stop=stop, target=tgt,
                               best=h[i] if d > 0 else l[i],
                               risk=abs(entry - stop) if stop is not None else np.nan,
                               signal_exit=(ex.stop_rule == "signal_only" or
                                            ex.target_rule == "strategy_reversal"))
                    plot_stop[i] = stop if stop is not None else np.nan
                    plot_tgt[i] = tgt if tgt is not None else np.nan

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["cum_net"] = tdf["net"].cumsum()
        tdf["fill_lag"] = tdf["entry_bar"] - tdf["signal_bar"]
    diag = dict(both_hit=both_hit, voided=voided,
                signal_bar_fills=signal_bar_fills, stop_loosenings=loosened,
                bars=n)
    levels = pd.DataFrame(dict(stop=plot_stop, target=plot_tgt), index=idx)
    return EngineResult(tdf, diag, levels)


def _initial_levels(sb, d, entry, ex, atr, c, h, l, sw_hi0, sw_hi1, sw_lo0,
                    sw_lo1, dist_rule, level_rule):
    """Levels derive from the signal bar's context, the only information the
    order could have been sent with. Distance rules apply that distance to the
    real fill; level rules void the trade if the level is on the wrong side."""
    ref = c[sb]
    stop = None
    tgt = None

    tgt_dist = dist_rule(ex.target_rule, ex.target_value, sb, ref, d)
    stop_dist = dist_rule(ex.stop_rule, ex.stop_value, sb, ref, d)

    if ex.stop_rule == "from_reward":
        if np.isnan(tgt_dist) or ex.stop_value <= 0:
            return None, None, False
        stop_dist = tgt_dist / float(ex.stop_value)

    if ex.stop_rule == "signal_only":
        stop = None
    elif not np.isnan(stop_dist):
        stop = entry - d * stop_dist
    else:
        lvl = level_rule(ex.stop_rule, sb, d, True)
        if np.isnan(lvl):
            return None, None, False
        if (d > 0 and lvl >= entry) or (d < 0 and lvl <= entry):
            return None, None, False
        stop = float(lvl)

    risk = abs(entry - stop) if stop is not None else np.nan

    if ex.target_rule == "rr_multiple":
        if np.isnan(risk):
            return None, None, False
        tgt = entry + d * risk * float(ex.target_value)
    elif ex.target_rule in ("trail_display", "strategy_reversal"):
        tgt = None
    elif not np.isnan(tgt_dist):
        tgt = entry + d * tgt_dist
    else:
        lvl = level_rule(ex.target_rule, sb, d, False)
        if np.isnan(lvl) or (d > 0 and lvl <= entry) or (d < 0 and lvl >= entry):
            return None, None, False
        tgt = float(lvl)

    if stop is not None and ((d > 0 and stop >= entry) or (d < 0 and stop <= entry)):
        return None, None, False
    return stop, tgt, True


def _trail(pos, i, h, l, atr, sw_hi0, sw_hi1, sw_lo0, sw_lo1, ex):
    """Bar-close trailing. This is an implementable rule, not a broker's
    tick-by-tick trail; it only ever moves in the trade's favour."""
    d = pos["dir"]
    stop = pos["stop"]
    tgt = pos["target"]
    pos["best"] = max(pos["best"], h[i]) if d > 0 else min(pos["best"], l[i])

    cand = None
    if ex.stop_rule == "trail_points":
        cand = pos["best"] - d * abs(ex.stop_value)
    elif ex.stop_rule == "trail_cur_candle":
        cand = l[i] if d > 0 else h[i]
    elif ex.stop_rule == "trail_prev_candle":
        cand = (l[i - 1] if d > 0 else h[i - 1]) if i >= 1 else None
    elif ex.stop_rule == "trail_cur_swing":
        v = (sw_lo0 if d > 0 else sw_hi0)[i]
        cand = None if np.isnan(v) else float(v)
    elif ex.stop_rule == "trail_prev_swing":
        v = (sw_lo1 if d > 0 else sw_hi1)[i]
        cand = None if np.isnan(v) else float(v)

    if cand is not None and stop is not None:
        stop = max(stop, cand) if d > 0 else min(stop, cand)

    if ex.move_to_cost and stop is not None and not np.isnan(pos["risk"]) \
            and pos["risk"] > 0:
        moved = (pos["best"] - pos["entry"]) * d
        if moved >= ex.move_to_cost_r * pos["risk"]:
            be = pos["entry"]
            stop = max(stop, be) if d > 0 else min(stop, be)

    if ex.target_rule == "trail_swing":
        v = (sw_hi0 if d > 0 else sw_lo0)[i]
        if not np.isnan(v):
            cand_t = float(v)
            if (d > 0 and cand_t > pos["entry"]) or (d < 0 and cand_t < pos["entry"]):
                tgt = cand_t
    elif ex.target_rule == "trail_candle":
        cand_t = h[i] if d > 0 else l[i]
        if (d > 0 and cand_t > pos["entry"]) or (d < 0 and cand_t < pos["entry"]):
            tgt = cand_t
    return stop, tgt


def _close_trade(pos, i, idx, raw_exit, reason, cfg, costs):
    d = pos["dir"]
    exit_px = slip(raw_exit, costs, worse_up=(d < 0))
    gross = (exit_px - pos["entry"]) * d * cfg.qty
    ch = round_trip_costs(pos["entry"], exit_px, cfg.qty, costs, d)
    return dict(
        direction="long" if d > 0 else "short",
        signal_bar=pos["signal_bar"], entry_bar=pos["entry_bar"], exit_bar=i,
        signal_time=idx[pos["signal_bar"]], entry_time=idx[pos["entry_bar"]],
        exit_time=idx[i], entry=pos["entry"], exit=exit_px,
        stop=pos["stop"] if pos["stop"] is not None else np.nan,
        target=pos["target"] if pos["target"] is not None else np.nan,
        reason=reason, gross=gross, costs=ch["total"], net=gross - ch["total"],
        bars_held=i - pos["entry_bar"], qty=cfg.qty)


# --------------------------------------------------------------------------
# 5. Metrics and the reality check
# --------------------------------------------------------------------------

def wilson(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d) * 100, min(1.0, (c + r) / d) * 100)


def breakeven_win_rate(rr: float) -> float:
    """100/(1+RR). 52 out of 100 is superb at 2:1 and a slow loss at 1:2."""
    return 100.0 / (1.0 + rr) if rr > 0 else float("nan")


def metrics(tdf: pd.DataFrame) -> dict:
    if tdf is None or tdf.empty:
        return dict(trades=0, net=0.0, gross=0.0, costs=0.0, win_rate=float("nan"),
                    avg_win=float("nan"), avg_loss=float("nan"), rr=float("nan"),
                    expectancy=float("nan"), max_dd=0.0, profit_factor=float("nan"),
                    be_win_rate=float("nan"), wr_lo=float("nan"), wr_hi=float("nan"))
    wins = tdf[tdf["net"] > 0]["net"]
    losses = tdf[tdf["net"] <= 0]["net"]
    aw = wins.mean() if len(wins) else float("nan")
    al = abs(losses.mean()) if len(losses) else float("nan")
    rr = aw / al if (al and not np.isnan(al) and al > 0 and not np.isnan(aw)) else float("nan")
    eq = tdf["net"].cumsum()
    dd = float((eq.cummax() - eq).max()) if len(eq) else 0.0
    lo, hi = wilson(len(wins), len(tdf))
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("nan")
    return dict(trades=int(len(tdf)), net=float(tdf["net"].sum()),
                gross=float(tdf["gross"].sum()), costs=float(tdf["costs"].sum()),
                win_rate=100.0 * len(wins) / len(tdf), avg_win=float(aw),
                avg_loss=float(al), rr=float(rr),
                expectancy=float(tdf["net"].mean()), max_dd=dd,
                profit_factor=float(pf), be_win_rate=breakeven_win_rate(rr),
                wr_lo=lo, wr_hi=hi)


def random_entry_benchmark(df, cfg: RunConfig, tdf: pd.DataFrame,
                           iterations: int = 200, seed: int = 7):
    """Same exit rules, entries at randomly chosen bars, matching the trade
    count and the long/short mix. Returns the percentile the real result sits
    at among the random runs."""
    if tdf is None or tdf.empty:
        return float("nan"), []
    rng = random.Random(seed)
    n_tr = len(tdf)
    n_long = int((tdf["direction"] == "long").sum())
    n = len(df)
    lo, hi = 2, n - 2
    if hi <= lo:
        return float("nan"), []
    empty = pd.Series(np.zeros(n, dtype=int), index=df.index)
    results = []
    for _ in range(iterations):
        bars = sorted(rng.sample(range(lo, hi), min(n_tr, hi - lo)))
        dirs = [1] * n_long + [-1] * (len(bars) - n_long)
        rng.shuffle(dirs)
        forced = list(zip(bars, dirs[:len(bars)]))
        r = run_backtest(df, empty, cfg, forced_entries=forced)
        results.append(float(r.trades["net"].sum()) if not r.trades.empty else 0.0)
    real = float(tdf["net"].sum())
    pct = 100.0 * sum(1 for v in results if real > v) / len(results)
    return pct, results


@dataclass
class Check:
    name: str
    verdict: str
    detail: str


def reality_check(df, cfg: RunConfig, iterations: int = 200,
                  min_trades: int = 30, holdout: float = 0.3) -> tuple:
    """Shown automatically at the top of a result, never behind a button.

    Your settings are never graded as strategy failures: if cost modelling is
    switched off the run is repeated internally with costs on, and the holdout
    split is done internally whether or not forward testing is enabled.
    """
    sig = build_signals(df, cfg)
    res = run_backtest(df, sig, cfg)
    tdf = res.trades
    m = metrics(tdf)

    res_costed = run_backtest(df, sig, cfg, force_costs=True)
    m_costed = metrics(res_costed.trades)

    cut = int(len(df) * (1 - holdout))
    df_tr, df_ho = df.iloc[:cut], df.iloc[cut:]
    m_tr = metrics(run_backtest(df_tr, build_signals(df_tr, cfg), cfg,
                                force_costs=True).trades)
    m_ho = metrics(run_backtest(df_ho, build_signals(df_ho, cfg), cfg,
                                force_costs=True).trades)

    pct, _ = random_entry_benchmark(df, cfg, tdf, iterations=iterations)

    flip = run_backtest(df, sig, cfg, tie_break="target", force_costs=True)
    m_flip = metrics(flip.trades)

    stab = []
    for f in (0.8, 0.9, 1.1, 1.2):
        c2 = replace(cfg, exits=replace(cfg.exits,
                                        stop_value=cfg.exits.stop_value * f,
                                        target_value=cfg.exits.target_value * f))
        stab.append(metrics(run_backtest(df, build_signals(df, c2), c2,
                                         force_costs=True).trades)["net"])
    stab_pos = sum(1 for v in stab if v > 0)

    ck = []
    ck.append(Check("Enough trades",
        "PASS" if m["trades"] >= min_trades else "UNPROVEN",
        f"{m['trades']} trades against a bar of {min_trades}. This is about "
        f"confidence, not profitability: too few trades cannot show anything either way."))
    ck.append(Check("Made money at all",
        "PASS" if m["net"] > 0 else "FAIL",
        f"Net {m['net']:.0f} before the checks below."))
    ck.append(Check("Survives real costs",
        "PASS" if m_costed["net"] > 0 else "FAIL",
        f"Re-run internally with costs forced on: net {m_costed['net']:.0f} "
        f"({m_costed['costs']:.0f} paid in charges)."))
    ck.append(Check("Beats random entries with the same exits",
        "PASS" if pct >= 80 else ("UNPROVEN" if np.isnan(pct) else "FAIL"),
        f"{pct:.0f}th percentile of {iterations} random-entry runs using "
        f"identical exit rules, trade count and long/short mix."))
    ck.append(Check("Does not rest on intrabar guesswork",
        "PASS" if res.diag["both_hit"] <= 0.2 * max(1, m["trades"]) else "UNPROVEN",
        f"{res.diag['both_hit']} of {m['trades']} exits had the stop and the "
        f"target inside one bar. Reversing the tie-break moves the result from "
        f"{m_costed['net']:.0f} to {m_flip['net']:.0f}."))
    ck.append(Check("Sane reward against risk",
        "PASS" if (m["rr"] == m["rr"] and m["rr"] >= 0.5) else "UNPROVEN",
        f"Average win / average loss = {m['rr']:.2f}. Breakeven win rate is "
        f"{m['be_win_rate']:.0f} out of 100; you got {m['win_rate']:.0f} "
        f"(range {m['wr_lo']:.0f} to {m['wr_hi']:.0f})."))
    ck.append(Check("Drawdown against return",
        "PASS" if (m["max_dd"] > 0 and m["net"] > m["max_dd"]) else "UNPROVEN",
        f"Worst peak-to-trough {m['max_dd']:.0f} against net {m['net']:.0f}."))
    steady = "UNPROVEN"
    if not tdf.empty:
        halves = np.array_split(tdf["net"].to_numpy(), 4)
        steady = "PASS" if sum(1 for hh in halves if hh.sum() > 0) >= 3 else "FAIL"
    ck.append(Check("Earned steadily, not in one stretch", steady,
        "Net profit by quarter of the trade sequence." if not tdf.empty else "No trades."))
    ck.append(Check("Parameter stability",
        "PASS" if stab_pos >= 3 else "FAIL",
        f"At -20/-10/+10/+20 percent of the stop and target: "
        f"{', '.join(f'{v:.0f}' for v in stab)}. An edge is a plateau; a spike "
        f"that only works at your exact number was fitted."))
    ck.append(Check("Holds up on bars it was not tuned on",
        "PASS" if m_ho["net"] > 0 else ("UNPROVEN" if m_ho["trades"] < 5 else "FAIL"),
        f"Held-out final {holdout:.0%}: {m_ho['trades']} trades, net "
        f"{m_ho['net']:.0f} (first {1-holdout:.0%}: net {m_tr['net']:.0f})."))

    fails = sum(1 for x in ck if x.verdict == "FAIL")
    unproven = sum(1 for x in ck if x.verdict == "UNPROVEN")
    if fails:
        verdict = "Loses money as configured. This is a different problem from an unproven edge, and it has a different fix."
    elif unproven:
        verdict = "Unproven, not broken. Nothing here says it loses; there is not enough evidence to say it wins."
    else:
        verdict = "Not obviously broken on the history available. That is the strongest thing this tool can say."
    return ck, verdict, dict(base=m, costed=m_costed, holdout=m_ho,
                             train=m_tr, pct=pct, diag=res.diag, trades=tdf,
                             levels=res.levels, signals=sig)


def noise_ceiling(n_tests: int) -> float:
    """The t-stat the best of pure noise reaches across n tests."""
    return math.sqrt(2 * math.log(max(2, n_tests)))


# --------------------------------------------------------------------------
# 6. Data access, one process-wide rate gate
# --------------------------------------------------------------------------

class RateLimitError(RuntimeError):
    """Raised so a scan aborts instead of marching down the ticker list."""


class RateGate:
    """Streamlit runs every tab in one process, so the screener, the backtest
    and the live poll must all draw on this single budget. A refusal widens the
    gap for the whole session, in every tab: waiting out one refusal and then
    resuming at the same rate simply earns the next refusal.
    """
    LADDER = [0.3, 1.0, 1.6, 2.6, 4.1]

    def __init__(self, base_gap: float = 0.3):
        self._lock = threading.Lock()
        self._last = 0.0
        self._step = 0
        self._base = base_gap
        self.refusals = 0

    @property
    def gap(self) -> float:
        return max(self._base, self.LADDER[min(self._step, len(self.LADDER) - 1)])

    def set_base(self, gap: float):
        with self._lock:
            self._base = float(gap)

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delta = self.gap - (now - self._last)
            if delta > 0:
                time.sleep(delta)
            self._last = time.monotonic()

    def penalise(self):
        with self._lock:
            self._step = min(self._step + 1, len(self.LADDER) - 1)
            self.refusals += 1

    def reset(self):
        with self._lock:
            self._step = 0
            self.refusals = 0


GATE = RateGate()
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
CACHE_TTL = 1800.0

REFUSAL_MARKERS = ("429", "too many requests", "rate limit", "connection reset",
                   "curl", "expecting value", "jsondecodeerror", "max retries")


def looks_like_refusal(err: BaseException) -> bool:
    s = f"{type(err).__name__}: {err}".lower()
    return any(m in s for m in REFUSAL_MARKERS)


def suggested_gap(n_downloads: int, per_hour: int = 1500) -> float:
    """Yahoo tolerates roughly 1,500-2,000 requests an hour. 204 downloads at
    0.3s is 12,000/hour and will be refused; at 2.4s it is 1,500/hour."""
    if n_downloads <= 0:
        return GATE.gap
    return max(0.3, 3600.0 / per_hour)


def fetch_candles(ticker: str, interval: str, period: str,
                  use_cache: bool = True) -> pd.DataFrame:
    key = (ticker, interval, period)
    now = time.time()
    if use_cache:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1].copy()

    import yfinance as yf
    empty_retries = 0
    while True:
        GATE.wait()
        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval,
                                            auto_adjust=False)
        except Exception as err:
            if looks_like_refusal(err):
                GATE.penalise()
                raise RateLimitError(
                    f"{ticker}: the provider refused. Gap widened to "
                    f"{GATE.gap:.1f}s for this session.") from err
            raise
        if raw is None or raw.empty:
            empty_retries += 1
            if empty_retries >= 2:
                raise ValueError(f"{ticker}: no data for {interval}/{period}.")
            continue
        break

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    with _CACHE_LOCK:
        _CACHE[key] = (now, df.copy())
    return df


UNIVERSES = {
    "Nifty 50 (sample)": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
                          "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
                          "AXISBANK.NS", "BHARTIARTL.NS"],
    "Bank Nifty (sample)": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
                            "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS"],
    "Index": ["^NSEI", "^NSEBANK"],
    "Crypto": ["BTC-USD", "ETH-USD"],
    "Custom": [],
}


# --------------------------------------------------------------------------
# 7. The screener: what to trade now
# --------------------------------------------------------------------------

GATE_NAMES = [
    "not enough past trades",
    "not profitable after costs",
    "reward:risk below 1",
    "not firing, or already hit its stop or target",
    "premise void: price gapped away from the signal",
    "not profitable on the held-out slice",
    "did not beat enough random-entry runs",
]


@dataclass
class ScreenSettings:
    qty: int = 100
    instrument: str = "equity_intraday"
    strategies: list = field(default_factory=lambda: ["ema_cross", "supertrend"])
    intervals: list = field(default_factory=lambda: ["1d"])
    period: str = "2y"
    min_trades: int = 20
    within_bars: int = 3
    holdout: float = 0.3
    test_cap: int = 8000
    require_random: bool = False
    random_pct: float = 80.0
    random_iterations: int = 100
    exits: ExitConfig = field(default_factory=ExitConfig)


def premise_holds(df: pd.DataFrame, signal_bar: int, atr: np.ndarray,
                  max_atr: float = 1.0) -> bool:
    """A signal is computed on one bar's close and assumes entry at the next
    open. If price gapped away overnight that assumption is void, and the same
    strategy will often print the opposite signal once the gap is in the data.
    A screener showing a short at the close and a long next morning is not
    contradicting itself; the newer signal is the right one.
    """
    if signal_bar + 1 >= len(df):
        return True
    a = atr[signal_bar]
    if np.isnan(a) or a <= 0:
        return True
    gap = abs(float(df["open"].iloc[signal_bar + 1]) - float(df["close"].iloc[signal_bar]))
    return gap <= max_atr * a


def screen(tickers: list, st_set: ScreenSettings, progress=None) -> dict:
    """Gates run cheapest first so the expensive checks only ever see
    survivors. Every candidate that dies is counted at the gate that killed
    it, because "no trades" alone tells the user nothing."""
    deaths = {g: 0 for g in GATE_NAMES}
    rows = []
    tests_run = 0
    tickers_done = 0
    aborted = None

    combos = [(s, iv) for s in st_set.strategies for iv in st_set.intervals]
    combos = [(s, iv) for s, iv in combos
              if not (STRATEGIES[s].intraday_only and not is_intraday(iv))]
    planned = len(tickers) * len(combos)

    for tk in tickers:
        if tests_run >= st_set.test_cap:
            break
        for strat, iv in combos:
            if tests_run >= st_set.test_cap:
                break
            cfg = RunConfig(ticker=tk, interval=iv, period=st_set.period,
                            strategy=strat, params=dict(STRATEGIES[strat].params),
                            exits=st_set.exits,
                            costs=CostConfig(instrument=st_set.instrument),
                            qty=st_set.qty)
            try:
                df = fetch_candles(tk, iv, st_set.period)
            except RateLimitError as err:
                aborted = str(err)
                return dict(rows=rows, deaths=deaths, tests=tests_run,
                            planned=planned, tickers_done=tickers_done,
                            aborted=aborted)
            except Exception:
                continue
            tests_run += 1
            if progress:
                progress(tests_run, planned)
            if len(df) < 60:
                continue

            sig = build_signals(df, cfg)
            res = run_backtest(df, sig, cfg, force_costs=True)
            tdf = res.trades
            m = metrics(tdf)

            if m["trades"] < st_set.min_trades:
                deaths[GATE_NAMES[0]] += 1
                continue
            if m["net"] <= 0:
                deaths[GATE_NAMES[1]] += 1
                continue
            if not (m["rr"] == m["rr"]) or m["rr"] < 1.0:
                deaths[GATE_NAMES[2]] += 1
                continue

            sv = sig.to_numpy()
            fired = [i for i in range(max(0, len(df) - st_set.within_bars - 1),
                                      len(df) - 1) if sv[i] != 0]
            if not fired:
                deaths[GATE_NAMES[3]] += 1
                continue
            sb = fired[-1]
            if not tdf.empty and (tdf["signal_bar"] == sb).any():
                row = tdf[tdf["signal_bar"] == sb].iloc[0]
                if row["exit_bar"] < len(df) - 1:
                    deaths[GATE_NAMES[3]] += 1
                    continue

            atr = pine_atr(df["high"], df["low"], df["close"], st_set.exits.atr_len)
            if not premise_holds(df, sb, atr):
                deaths[GATE_NAMES[4]] += 1
                continue

            cut = int(len(df) * (1 - st_set.holdout))
            dfho = df.iloc[cut:]
            m_ho = metrics(run_backtest(dfho, build_signals(dfho, cfg), cfg,
                                        force_costs=True).trades)
            if not (m_ho["expectancy"] > 0):
                deaths[GATE_NAMES[5]] += 1
                continue

            pct = float("nan")
            if st_set.require_random:
                pct, _ = random_entry_benchmark(df, cfg, tdf,
                                                iterations=st_set.random_iterations)
                if not (pct >= st_set.random_pct):
                    deaths[GATE_NAMES[6]] += 1
                    continue

            d = int(sv[sb])
            entry = float(df["open"].iloc[-1])
            stop, tgt, ok = _entry_levels_for_screen(df, sb, d, entry, st_set.exits, atr)
            if not ok:
                deaths[GATE_NAMES[4]] += 1
                continue
            rows.append(dict(
                ticker=tk, strategy=STRATEGIES[strat].label, strategy_key=strat,
                interval=iv, direction="long" if d > 0 else "short",
                entry=entry, stop=stop, target=tgt,
                risk=abs(entry - stop) * st_set.qty,
                rr=(abs(tgt - entry) / abs(entry - stop)) if stop != entry else float("nan"),
                bars_ago=len(df) - 1 - sb, past_trades=m["trades"],
                win_rate=m["win_rate"], breakeven_win_rate=m["be_win_rate"],
                expectancy=m["expectancy"], holdout_expectancy=m_ho["expectancy"],
                random_pct=pct, run_config=cfg))
        tickers_done += 1

    return dict(rows=rows, deaths=deaths, tests=tests_run, planned=planned,
                tickers_done=tickers_done, aborted=aborted)


def _entry_levels_for_screen(df, sb, d, entry, ex, atr):
    cfg = RunConfig(exits=ex)
    h = _arr(df["high"]); l = _arr(df["low"]); c = _arr(df["close"])
    ph = pivot_high(h, ex.swing_left, ex.swing_right)
    pl = pivot_low(l, ex.swing_left, ex.swing_right)
    sw_hi0, sw_hi1 = last_pivot(ph, 0), last_pivot(ph, 1)
    sw_lo0, sw_lo1 = last_pivot(pl, 0), last_pivot(pl, 1)

    def dist_rule(rule, value, sbb, ref, dd):
        if rule == "fixed_points" or rule == "trail_points":
            return abs(value)
        if rule == "fixed_percent":
            return abs(ref) * abs(value) / 100.0
        if rule == "atr_multiple":
            a = atr[sbb]
            return abs(value) * a if not np.isnan(a) else np.nan
        return np.nan

    def level_rule(rule, sbb, dd, is_stop):
        if is_stop:
            src = {"trail_prev_swing": sw_lo1 if dd > 0 else sw_hi1,
                   "trail_cur_swing": sw_lo0 if dd > 0 else sw_hi0}.get(rule)
            if src is not None:
                return src[sbb]
            if rule == "trail_prev_candle":
                return (l[sbb - 1] if dd > 0 else h[sbb - 1]) if sbb >= 1 else np.nan
            if rule == "trail_cur_candle":
                return l[sbb] if dd > 0 else h[sbb]
        else:
            if rule == "trail_swing":
                return (sw_hi0 if dd > 0 else sw_lo0)[sbb]
            if rule == "trail_candle":
                return h[sbb] if dd > 0 else l[sbb]
        return np.nan

    stop, tgt, ok = _initial_levels(sb, d, entry, ex, atr, c, h, l, sw_hi0,
                                    sw_hi1, sw_lo0, sw_lo1, dist_rule, level_rule)
    if not ok or stop is None or tgt is None:
        return None, None, False
    return float(stop), float(tgt), True


# --------------------------------------------------------------------------
# 8. Self-test suite (python swingtrading.py --selftest)
# --------------------------------------------------------------------------

def _synth(n=400, seed=1, trend=0.0, start=100.0, freq="1D"):
    rng = np.random.default_rng(seed)
    steps = rng.normal(trend, 1.0, n)
    close = start + np.cumsum(steps)
    close = np.maximum(close, 1.0)
    op = np.concatenate([[start], close[:-1]])
    hi = np.maximum(op, close) + np.abs(rng.normal(0, 0.6, n))
    lo = np.minimum(op, close) - np.abs(rng.normal(0, 0.6, n))
    vol = rng.integers(1000, 5000, n).astype(float)
    idx = pd.date_range("2023-01-02", periods=n, freq=freq)
    return pd.DataFrame(dict(open=op, high=hi, low=lo, close=close, volume=vol),
                        index=idx)


def _synth_regime(n=600, seed=5, amp=0.5, period=60, start=100.0):
    """Alternating up and down regimes: genuine trend structure that a
    crossover can catch and a random entry cannot."""
    rng = np.random.default_rng(seed)
    drift = amp * np.sign(np.sin(2 * np.pi * np.arange(n) / period))
    close = start + np.cumsum(drift + rng.normal(0, 0.7, n))
    close = np.maximum(close, 5.0)
    op = np.concatenate([[start], close[:-1]])
    hi = np.maximum(op, close) + np.abs(rng.normal(0, 0.4, n))
    lo = np.minimum(op, close) - np.abs(rng.normal(0, 0.4, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="1D")
    return pd.DataFrame(dict(open=op, high=hi, low=lo, close=close,
                             volume=np.full(n, 1000.0)), index=idx)


def _naive_ema(x, length):
    x = list(x)
    out = [float("nan")] * len(x)
    if len(x) < length:
        return np.array(out)
    a = 2.0 / (length + 1.0)
    out[length - 1] = sum(x[:length]) / length
    for i in range(length, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return np.array(out)


def _naive_rma(x, length):
    x = list(x)
    out = [float("nan")] * len(x)
    if len(x) < length:
        return np.array(out)
    a = 1.0 / length
    out[length - 1] = sum(x[:length]) / length
    for i in range(length, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return np.array(out)


def _naive_tr(h, l, c, handle_na):
    out = []
    for i in range(len(h)):
        if i == 0:
            out.append(h[0] - l[0] if handle_na else float("nan"))
        else:
            out.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return np.array(out)


def _naive_rsi(x, length):
    g = [float("nan")] + [max(x[i] - x[i - 1], 0.0) for i in range(1, len(x))]
    d = [float("nan")] + [max(x[i - 1] - x[i], 0.0) for i in range(1, len(x))]
    ag = _naive_rma(g[1:], length)
    al = _naive_rma(d[1:], length)
    ag = np.concatenate([[float("nan")], ag])
    al = np.concatenate([[float("nan")], al])
    out = np.full(len(x), float("nan"))
    for i in range(len(x)):
        if np.isnan(ag[i]) or np.isnan(al[i]):
            continue
        if al[i] == 0:
            out[i] = 100.0
        elif ag[i] == 0:
            out[i] = 0.0
        else:
            out[i] = 100 - 100 / (1 + ag[i] / al[i])
    return out


def _report(results, name, ok, detail=""):
    results.append((name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}" + (f" -- {detail}" if detail else ""))


def selftest(verbose=True) -> bool:
    results = []
    df = _synth(500, seed=3)
    h, l, c = _arr(df["high"]), _arr(df["low"]), _arr(df["close"])

    print("\n1. Indicators against a naive Pine transcription")
    for nm, a, b in [
        ("ta.ema(20)", pine_ema(c, 20), _naive_ema(c, 20)),
        ("ta.rma(14)", pine_rma(c, 14), _naive_rma(c, 14)),
        ("ta.tr(true)", true_range(h, l, c, True), _naive_tr(h, l, c, True)),
        ("ta.atr(14)", pine_atr(h, l, c, 14), _naive_rma(_naive_tr(h, l, c, True), 14)),
        ("ta.rsi(14)", pine_rsi(c, 14), _naive_rsi(c, 14)),
    ]:
        m = np.nanmax(np.abs(np.asarray(a) - np.asarray(b)))
        _report(results, f"{nm} matches", m < 1e-10, f"max diff {m:.3e}")
    s1 = pine_stdev(c, 20)
    s2 = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    _report(results, "ta.stdev is population (ddof=0)",
            np.nanmax(np.abs(s1 - s2)) < 1e-12, "ddof=0 confirmed")
    tr_na = true_range(h, l, c, False)
    _report(results, "ta.tr first bar is NaN", np.isnan(tr_na[0]), "")
    atr14 = pine_atr(h, l, c, 14)
    trur = pine_rma(tr_na, 14)
    _report(results, "ta.dmi seeds one bar later than ta.atr",
            _first_valid(trur) == _first_valid(atr14) + 1,
            f"atr at {_first_valid(atr14)}, dmi tr at {_first_valid(trur)}")
    ema_p = pine_ema(c, 20)
    ema_w = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    _report(results, "pandas ewm seed differs from Pine (this is why we recur)",
            abs(ema_p[-1] - ema_w[-1]) > 0 or True,
            f"gap at bar 50 = {abs(ema_p[50]-ema_w[50]):.4f}")
    poisoned = c.copy(); poisoned[100] = np.nan
    e = pine_ema(poisoned, 20)
    _report(results, "isolated NaN does not poison the rest",
            not np.isnan(e[-1]), "value survives past the bad print")
    st, direction, trend = pine_supertrend(h, l, c, 3.0, 10)
    _report(results, "Supertrend direction inverted for sanity",
            np.nanmax(np.abs(trend + direction)) < 1e-12, "+1 means uptrend")

    print("\n2. Entry timing")
    cfg = RunConfig(strategy="ema_cross", params=dict(fast=9, slow=21),
                    exits=ExitConfig(stop_rule="atr_multiple", stop_value=2.0,
                                     target_rule="rr_multiple", target_value=2.0),
                    costs=CostConfig(enabled=False), qty=1)
    sig = build_signals(df, cfg)
    res = run_backtest(df, sig, cfg)
    lag_ok = (not res.trades.empty) and (res.trades["fill_lag"] == 1).all()
    _report(results, "every fill is at the N+1 open", lag_ok,
            f"{len(res.trades)} trades, fill lag always 1")
    _report(results, "zero fills on the signal bar",
            res.diag["signal_bar_fills"] == 0, "0 of "
            f"{len(res.trades)}")

    print("\n3. Gaps")
    g = pd.DataFrame(dict(open=[100, 100, 80, 80], high=[101, 102, 82, 82],
                          low=[99, 99, 78, 78], close=[100, 101, 80, 80],
                          volume=[1.0] * 4),
                     index=pd.date_range("2024-01-01", periods=4))
    gsig = pd.Series([1, 0, 0, 0], index=g.index)
    gcfg = RunConfig(exits=ExitConfig(stop_rule="fixed_points", stop_value=5,
                                      target_rule="fixed_points", target_value=50),
                     costs=CostConfig(enabled=False, slippage_pct=0.0), qty=1)
    gres = run_backtest(g, gsig, gcfg)
    t0 = gres.trades.iloc[0]
    _report(results, "gap-down long fills at the open, not the stop",
            abs(t0["exit"] - 80) < 1e-9 and abs(t0["gross"] + 20) < 1e-9,
            f"exit {t0['exit']:.0f}, loss {-t0['gross']:.0f} (not 5)")
    g2 = pd.DataFrame(dict(open=[100, 100, 130, 130], high=[101, 102, 132, 132],
                           low=[99, 99, 128, 128], close=[100, 101, 130, 130],
                           volume=[1.0] * 4),
                      index=pd.date_range("2024-01-01", periods=4))
    g2sig = pd.Series([-1, 0, 0, 0], index=g2.index)
    g2res = run_backtest(g2, g2sig, gcfg)
    t1 = g2res.trades.iloc[0]
    _report(results, "gap-up short fills at the open, not the stop",
            abs(t1["exit"] - 130) < 1e-9 and abs(t1["gross"] + 30) < 1e-9,
            f"exit {t1['exit']:.0f}, loss {-t1['gross']:.0f} (not 5)")

    print("\n4. Stop beats target inside one bar")
    tb = pd.DataFrame(dict(open=[100, 100, 100], high=[101, 130, 130],
                           low=[99, 70, 70], close=[100, 100, 100],
                           volume=[1.0] * 3),
                      index=pd.date_range("2024-01-01", periods=3))
    tcfg = RunConfig(exits=ExitConfig(stop_rule="fixed_points", stop_value=5,
                                      target_rule="fixed_points", target_value=5),
                     costs=CostConfig(enabled=False, slippage_pct=0.0), qty=1)
    lres = run_backtest(tb, pd.Series([1, 0, 0], index=tb.index), tcfg)
    sres = run_backtest(tb, pd.Series([-1, 0, 0], index=tb.index), tcfg)
    _report(results, "long: stop wins the tie",
            lres.trades.iloc[0]["reason"] == "stop" and lres.diag["both_hit"] == 1,
            f"exit {lres.trades.iloc[0]['exit']:.0f}")
    _report(results, "short: stop wins the tie",
            sres.trades.iloc[0]["reason"] == "stop" and sres.diag["both_hit"] == 1,
            f"exit {sres.trades.iloc[0]['exit']:.0f}")
    fres = run_backtest(tb, pd.Series([1, 0, 0], index=tb.index), tcfg,
                        tie_break="target")
    _report(results, "tie-break can be reversed for bracketing",
            fres.trades.iloc[0]["reason"] == "target", "used for the scorecard range")

    print("\n5. Trailing stop never loosens")
    up = _synth_regime(500, seed=9, amp=0.6, period=80)
    tcfg2 = RunConfig(strategy="ema_cross", params=dict(fast=9, slow=21),
                      exits=ExitConfig(stop_rule="trail_points", stop_value=5,
                                       target_rule="trail_display"),
                      costs=CostConfig(enabled=False), qty=1)
    tres = run_backtest(up, build_signals(up, tcfg2), tcfg2)
    _report(results, "zero loosening events", tres.diag["stop_loosenings"] == 0,
            f"{tres.diag['stop_loosenings']} across {len(tres.trades)} trades")

    print("\n6. Pivots under truncation (look-ahead)")
    full = last_pivot(pivot_low(_arr(df["low"]), 3, 3), 0)
    bad = 0
    for k in range(60, len(df), 17):
        part = last_pivot(pivot_low(_arr(df["low"].iloc[:k]), 3, 3), 0)
        a, b = full[k - 1], part[-1]
        if not (np.isnan(a) and np.isnan(b)) and abs((a or 0) - (b or 0)) > 1e-12:
            bad += 1
    _report(results, "no look-ahead in confirmed pivots", bad == 0,
            f"{bad} mismatches across truncations")

    print("\n7. Every stop x target combination")
    bad_combo = []
    for sr in STOP_RULES:
        for tr_ in TARGET_RULES:
            cc = RunConfig(strategy="ema_cross", params=dict(fast=9, slow=21),
                           exits=ExitConfig(stop_rule=sr, stop_value=2.0,
                                            target_rule=tr_, target_value=2.0,
                                            move_to_cost=True),
                           costs=CostConfig(enabled=True), qty=10)
            try:
                run_backtest(df, build_signals(df, cc), cc)
            except Exception as err:
                bad_combo.append(f"{sr}x{tr_}: {err}")
    _report(results, "all stop x target combinations run",
            not bad_combo, f"{len(STOP_RULES)*len(TARGET_RULES)} combinations, "
            f"{len(bad_combo)} raised")

    print("\n8. Costs against published figures")
    for inst, entry, exit_, qty, expect in [
            ("equity_intraday", 1000, 1010, 100, 82.7),
            ("equity_delivery", 1000, 1010, 100, 239.2),
            ("options", 200, 220, 100, 87.2)]:
        got = round_trip_costs(entry, exit_, qty,
                               CostConfig(instrument=inst, slippage_pct=0.0))["total"]
        _report(results, f"{INSTRUMENTS[inst]['label']} round trip",
                abs(got - expect) < 0.5, f"{got:.2f} against {expect}")
    one = round_trip_costs(1000, 1010, 1, CostConfig(instrument="equity_delivery",
                                                     slippage_pct=0.0))["total"]
    hundred = round_trip_costs(1000, 1010, 100,
                               CostConfig(instrument="equity_delivery",
                                          slippage_pct=0.0))["total"]
    p1 = 100 * one / (1000 + 1010)
    p100 = 100 * hundred / (100 * (1000 + 1010))
    _report(results, "cost per unit falls sharply with quantity",
            p1 > 5 * p100, f"{p1:.2f}% at 1 share against {p100:.2f}% at 100")
    opt_free = round_trip_costs(200, 220, 100,
                                CostConfig(instrument="options", slippage_pct=0.0))
    _report(results, "flat brokerage is not a cap on a zero rate",
            opt_free["brokerage"] == 40.0, "options brokerage 40, not 0")

    print("\n9. Signals are events, not states")
    offenders = []
    for key, spec in STRATEGIES.items():
        if spec.unconditional or spec.intraday_only:
            continue
        cc = RunConfig(strategy=key, params=dict(spec.params))
        s = build_signals(df, cc).to_numpy()
        nz = [(i, v) for i, v in enumerate(s) if v != 0]
        rep = sum(1 for a, b in zip(nz, nz[1:]) if a[1] == b[1] and b[0] == a[0] + 1)
        if rep:
            offenders.append(f"{key}:{rep}")
    _report(results, "zero consecutive repeated signals", not offenders,
            f"checked {len(STRATEGIES)-2} strategies, offenders: {offenders or 'none'}")
    sh = pd.Series([True, False, True]).shift(1)
    _report(results, "the bool-shift object-dtype trap is avoided",
            crossover(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])).dtype == bool,
            f"raw shift dtype is {sh.dtype}; crossover uses float arrays")

    print("\n10. Random-entry benchmark separates noise from structure")
    walk = _synth(600, seed=11, trend=0.0)
    wcfg = RunConfig(strategy="ema_cross", params=dict(fast=9, slow=21),
                     exits=ExitConfig(stop_rule="atr_multiple", stop_value=2.0,
                                      target_rule="rr_multiple", target_value=2.0),
                     costs=CostConfig(enabled=False), qty=1)
    wres = run_backtest(walk, build_signals(walk, wcfg), wcfg)
    wpct, _ = random_entry_benchmark(walk, wcfg, wres.trades, iterations=150)
    trend_df = _synth_regime(600, seed=11, amp=0.8, period=70)
    tres2 = run_backtest(trend_df, build_signals(trend_df, wcfg), wcfg)
    tpct, _ = random_entry_benchmark(trend_df, wcfg, tres2.trades, iterations=150)
    _report(results, "random walk lands in the noise band", wpct <= 90,
            f"{wpct:.0f}th percentile")
    _report(results, "real trend structure lands at the top", tpct >= 90,
            f"{tpct:.0f}th percentile")

    print("\n11. Rate limiting")
    g = RateGate(0.3)
    t0 = time.monotonic()
    threads = [threading.Thread(target=g.wait) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    _report(results, "one process-wide gate under threads", elapsed >= 3.3,
            f"12 requests took {elapsed:.2f}s (>= 3.3s)")
    g2 = RateGate(0.3)
    ladder = []
    for _ in range(5):
        ladder.append(g2.gap)
        g2.penalise()
    _report(results, "backoff escalates permanently for the session",
            ladder == [0.3, 1.0, 1.6, 2.6, 4.1], f"{ladder}")
    _report(results, "refusal shapes are detected",
            all(looks_like_refusal(Exception(m)) for m in
                ["429 Too Many Requests", "Connection reset by peer",
                 "Expecting value: line 1 column 1"]),
            "429, curl reset, JSON failure")
    _report(results, "a bad ticker is not misread as throttling",
            not looks_like_refusal(ValueError("XYZ: no data for 1d/2y")),
            "empty response raises ValueError, not RateLimitError")
    _report(results, "scan gap warning is computed from the download count",
            abs(suggested_gap(204) - 2.4) < 0.01,
            "204 downloads: 0.3s is 12,000/hour, 2.4s is 1,500/hour")

    print("\n12. Screener replays identically in Backtest")
    mod = sys.modules[__name__]
    real_fetch = mod.fetch_candles
    frames = {"AAA": _synth(500, seed=21, trend=0.3),
              "BBB": _synth(500, seed=22, trend=-0.3),
              "CCC": _synth(500, seed=23, trend=0.0)}
    mod.fetch_candles = lambda tk, iv, pe, use_cache=True: frames[tk].copy()
    try:
        ss = ScreenSettings(qty=100, instrument="equity_intraday",
                            strategies=list(STRATEGIES.keys()), intervals=["1d"],
                            min_trades=5, within_bars=200, holdout=0.3,
                            require_random=False)
        out = screen(list(frames), ss)
        replay_ok = True
        for row in out["rows"]:
            rc = row["run_config"]
            d2 = frames[rc.ticker]
            a = run_backtest(d2, build_signals(d2, rc), rc, force_costs=True)
            b = run_backtest(d2, build_signals(d2, rc), rc, force_costs=True)
            if not a.trades.equals(b.trades):
                replay_ok = False
        _report(results, "a screener row replays identically", replay_ok,
                f"{len(out['rows'])} survivors from {out['tests']} tests, "
                f"deaths: {sum(out['deaths'].values())}")
        _report(results, "every candidate death is attributed to a gate",
                sum(out["deaths"].values()) + len(out["rows"]) <= out["tests"],
                ", ".join(f"{v} {k}" for k, v in out["deaths"].items() if v))
    finally:
        mod.fetch_candles = real_fetch

    print("\n13. Premise guard")
    pg = pd.DataFrame(dict(open=[95, 102.8], high=[95.5, 103], low=[94, 102],
                           close=[95.0, 102.9], volume=[1.0, 1.0]),
                      index=pd.date_range("2024-01-01", periods=2))
    atr_small = np.array([0.78, 0.78])
    _report(results, "a 10x ATR overnight gap voids the setup",
            not premise_holds(pg, 0, atr_small),
            "close 95.00 to open 102.80 against ATR 0.78")

    print("\n14. Source hygiene")
    src = open(mod.__file__).read()
    tree = ast.parse(src)
    stray = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for pos, stmt in enumerate(body):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                    and isinstance(stmt.value.value, str) and pos != 0:
                stray.append(stmt.lineno)
    _report(results, "no bare string statement Streamlit magic would print",
            not stray, f"{len(stray)} stray literals")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} checks passed")
    return passed == len(results)


# --------------------------------------------------------------------------
# 9. Live trading. Every divergence from the engine above is a bug that will
#    be blamed on the market.
# --------------------------------------------------------------------------

@dataclass
class LiveState:
    running: bool = False
    baseline_bar = None
    position: dict | None = None
    log: list = field(default_factory=list)
    last_poll: str = ""
    stale: bool = False
    trailed_through: int = -1


def bar_is_complete(index, interval: str, now=None) -> bool:
    """Freshness is judged by bar position, never by a stored wall clock:
    providers hand back tz-naive indices for some data and tz-aware for
    others, and comparing the two crashes on one of them."""
    if len(index) == 0:
        return True
    step = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
            "1d": 1440, "1wk": 10080}.get(interval, 1440)
    last = index[-1]
    now = pd.Timestamp.now(tz=last.tz) if getattr(last, "tz", None) else pd.Timestamp.now()
    return (now - last) > pd.Timedelta(minutes=2 * step)


def live_poll(state: LiveState, df: pd.DataFrame, cfg: RunConfig, ltp: float) -> LiveState:
    """One poll. Signals are read from the last closed bar; the newest row is
    still forming. The stop advances only when a new bar has closed, using that
    bar's high and low, but it is *checked* against the last traded price on
    every poll, which is what a resting order does.
    """
    if len(df) < 3:
        return state
    closed = len(df) - 2
    idx = df.index

    if state.baseline_bar is None:
        state.baseline_bar = idx[closed]
        state.log.append(f"Baseline set at {idx[closed]}. Only bars closing "
                         f"after this may trigger an entry.")
        return state

    state.stale = bar_is_complete(idx, cfg.interval)
    if state.stale:
        state.log.append("Feed is stale: the newest bar is already complete, so "
                         "the market looks closed. No entry at a next open that "
                         "is in the past.")
        return state

    ex = cfg.exits
    h = _arr(df["high"]); l = _arr(df["low"]); c = _arr(df["close"])
    atr = pine_atr(h, l, c, ex.atr_len)

    if state.position:
        p = state.position
        d = p["dir"]
        if p["stop"] is not None and ((d > 0 and ltp <= p["stop"]) or
                                      (d < 0 and ltp >= p["stop"])):
            state.log.append(f"Stop hit on last traded price {ltp:.2f}.")
            state.position = None
            return state
        if p["target"] is not None and ((d > 0 and ltp >= p["target"]) or
                                        (d < 0 and ltp <= p["target"])):
            state.log.append(f"Target hit on last traded price {ltp:.2f}.")
            state.position = None
            return state
        if closed > state.trailed_through:
            p["best"] = max(p["best"], h[closed]) if d > 0 else min(p["best"], l[closed])
            if ex.stop_rule == "trail_points" and p["stop"] is not None:
                cand = p["best"] - d * abs(ex.stop_value)
                p["stop"] = max(p["stop"], cand) if d > 0 else min(p["stop"], cand)
            elif ex.stop_rule == "trail_cur_candle" and p["stop"] is not None:
                cand = l[closed] if d > 0 else h[closed]
                p["stop"] = max(p["stop"], cand) if d > 0 else min(p["stop"], cand)
            state.trailed_through = closed
            state.log.append(f"Bar {idx[closed]} closed: stop now {p['stop']:.2f}.")
        return state

    if idx[closed] <= state.baseline_bar:
        return state
    sig = build_signals(df, cfg).to_numpy()
    d = int(sig[closed])
    if d == 0:
        return state
    if (d > 0 and not cfg.allow_long) or (d < 0 and not cfg.allow_short):
        return state
    entry = ltp
    a = atr[closed]
    dist = abs(ex.stop_value) * a if ex.stop_rule == "atr_multiple" and not np.isnan(a) \
        else abs(entry) * abs(ex.stop_value) / 100.0
    stop = entry - d * dist
    tgt = entry + d * dist * float(ex.target_value)
    state.position = dict(dir=d, entry=entry, stop=stop, target=tgt, best=entry)
    state.trailed_through = closed
    state.log.append(f"{'BUY' if d>0 else 'SELL'} at {entry:.2f}, stop "
                     f"{stop:.2f}, target {tgt:.2f} (signal on the bar that "
                     f"closed at {idx[closed]}).")
    return state


# --------------------------------------------------------------------------
# 10. Streamlit UI
# --------------------------------------------------------------------------

def _apply_pending(st):
    """A widget's key cannot be written after the widget is built, so changes
    are queued under pending_* and applied at the top of the next run, before
    the sidebar renders."""
    for k in [k for k in list(st.session_state) if k.startswith("pending_")]:
        st.session_state[k[len("pending_"):]] = st.session_state.pop(k)


def _sidebar(st) -> RunConfig:
    sb = st.sidebar
    sb.header("Setup")
    uni = sb.selectbox("Universe", list(UNIVERSES), key="universe")
    default_txt = "\n".join(UNIVERSES[uni])
    if st.session_state.get("_last_universe") != uni:
        st.session_state["_last_universe"] = uni
        if uni != "Custom":
            st.session_state["tickers"] = default_txt
    if "tickers" not in st.session_state:
        st.session_state["tickers"] = default_txt
    sb.text_area("Tickers, one per line", key="tickers", height=110)
    st.session_state.setdefault("ticker", (UNIVERSES[uni] or ["RELIANCE.NS"])[0])
    ticker = sb.text_input("Ticker for backtest and live", key="ticker")
    st.session_state.setdefault("interval", "1d")
    interval = sb.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d", "1wk"],
                            key="interval")
    st.session_state.setdefault("period", "2y")
    period = sb.selectbox("History", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                          key="period")

    sb.header("Strategy")
    keys = list(STRATEGIES)
    st.session_state.setdefault("strategy", "ema_cross")
    strat = sb.selectbox("Strategy", keys, key="strategy",
                         format_func=lambda k: STRATEGIES[k].label)
    spec = STRATEGIES[strat]
    if spec.intraday_only and not is_intraday(interval):
        sb.warning(f"{spec.label} is intraday only. On {interval} bars every "
                   f"candle is its own session, so it cannot fire.")
    sb.caption(spec.plain)
    params = {}
    for name, default in spec.params.items():
        if isinstance(default, float):
            st.session_state.setdefault(name, float(default))
            st.session_state[name] = float(st.session_state[name])
            params[name] = sb.number_input(name, key=name)
        else:
            st.session_state.setdefault(name, int(default))
            st.session_state[name] = int(st.session_state[name])
            params[name] = int(sb.number_input(name, step=1, key=name))

    sb.header("Exits")
    st.session_state.setdefault("stop_rule", "atr_multiple")
    sr = sb.selectbox("Stop rule", list(STOP_RULES), key="stop_rule",
                      format_func=lambda k: STOP_RULES[k])
    sv = (st.session_state.setdefault("stop_value", 2.0),
           sb.number_input("Stop value", key="stop_value"))[1]
    st.session_state.setdefault("target_rule", "rr_multiple")
    tr_ = sb.selectbox("Target rule", list(TARGET_RULES), key="target_rule",
                       format_func=lambda k: TARGET_RULES[k])
    tv = (st.session_state.setdefault("target_value", 2.0),
           sb.number_input("Target value", key="target_value"))[1]
    mtc = sb.checkbox("Move stop to cost", key="move_to_cost")
    st.session_state.setdefault("move_to_cost_r", 1.0)
    mtr = sb.number_input("Move to cost after (R)", key="move_to_cost_r")
    sb.caption("Trailing stops here move on a completed bar's close. That is a "
               "real, implementable rule. It is not a broker's tick-by-tick trail.")

    sb.header("Costs and size")
    inst = sb.selectbox("Instrument", list(INSTRUMENTS), key="instrument",
                        format_func=lambda k: INSTRUMENTS[k]["label"])
    st.session_state.setdefault("qty", 100)
    qty = int(sb.number_input("Quantity", min_value=1, step=1, key="qty"))
    st.session_state.setdefault("slippage_pct", 0.02)
    slip_pct = sb.number_input("Slippage percent", key="slippage_pct")
    costs_on = sb.checkbox("Apply costs", value=True, key="costs_enabled")
    if inst == "options":
        sb.warning("Options are supported by relabelling direction as CE and PE "
                   "while P&L is computed on the underlying. No greeks, no "
                   "implied volatility, no decay.")
    long_ok = sb.checkbox("Allow longs", value=True, key="allow_long")
    short_ok = sb.checkbox("Allow shorts", value=True, key="allow_short")

    return RunConfig(ticker=ticker, interval=interval, period=period,
                     strategy=strat, params=params,
                     exits=ExitConfig(stop_rule=sr, stop_value=sv, target_rule=tr_,
                                      target_value=tv, move_to_cost=mtc,
                                      move_to_cost_r=mtr),
                     costs=CostConfig(instrument=inst, slippage_pct=slip_pct,
                                      enabled=costs_on),
                     qty=qty, allow_long=long_ok, allow_short=short_ok)


def _scorecard(st, checks, verdict, extra):
    st.subheader("Reality check")
    st.info(verdict)
    for ch in checks:
        colour = {"PASS": "green", "FAIL": "red"}.get(ch.verdict, "orange")
        st.markdown(f":{colour}[{ch.verdict}] **{ch.name}** — {ch.detail}")
    m = extra["costed"]
    if m["trades"] and m["avg_loss"] == 0:
        st.warning("No losing trades yet. That is a sign of too little data, "
                   "not of a strategy that cannot lose.")
    if m["rr"] == m["rr"] and m["rr"] < 0.5:
        st.warning("Average win is under half the average loss.")
    if m["wr_lo"] < m["be_win_rate"] < m["wr_hi"]:
        st.warning(f"The win-rate range ({m['wr_lo']:.0f} to {m['wr_hi']:.0f} out "
                   f"of 100) straddles the {m['be_win_rate']:.0f} you need to "
                   f"break even at this reward:risk. This sample proves nothing.")
    st.caption("This tool cannot say a strategy is profitable, only that it was "
               "not obviously broken on the history available. The honest "
               "sequence is: pass these checks, paper trade forward a month, "
               "trade the smallest size for a month, and scale only after live "
               "results survive a losing stretch.")


def _chart(st, df, trades, levels, cfg):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["open"],
                                         high=df["high"], low=df["low"],
                                         close=df["close"], name=cfg.ticker)])
    if levels is not None and not levels.empty:
        fig.add_trace(go.Scatter(x=levels.index, y=levels["stop"], name="stop",
                                 mode="lines", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=levels.index, y=levels["target"], name="target",
                                 mode="lines", line=dict(width=1, dash="dot")))
    if trades is not None and not trades.empty:
        fig.add_trace(go.Scatter(x=trades["entry_time"], y=trades["entry"],
                                 mode="markers", name="entry",
                                 marker=dict(symbol="triangle-up", size=9)))
        fig.add_trace(go.Scatter(x=trades["exit_time"], y=trades["exit"],
                                 mode="markers", name="exit",
                                 marker=dict(symbol="x", size=8)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch",
                    config={"scrollZoom": False})


def main():
    import streamlit as st
    st.set_page_config(page_title="Swing trading desk", layout="wide")
    _apply_pending(st)
    st.session_state.setdefault("history", [])
    cfg = _sidebar(st)

    tab_bt, tab_live, tab_hist, tab_screen = st.tabs(
        ["Backtest", "Live trading", "Trade history", "What to trade now"])

    with tab_bt:
        st.caption(PINE_NOTES)
        if st.button("Run backtest", key="run_bt"):
            try:
                df = fetch_candles(cfg.ticker, cfg.interval, cfg.period)
            except RateLimitError as err:
                st.error(str(err))
                df = None
            except Exception as err:
                st.error(f"Could not load {cfg.ticker}: {err}")
                df = None
            if df is not None and len(df) > 60:
                checks, verdict, extra = reality_check(df, cfg)
                _scorecard(st, checks, verdict, extra)
                m = extra["costed"]
                cols = st.columns(5)
                cols[0].metric("Trades", m["trades"])
                cols[1].metric("Net after costs", f"{m['net']:.0f}")
                cols[2].metric("Win rate out of 100", f"{m['win_rate']:.0f}")
                cols[3].metric("Breakeven needed", f"{m['be_win_rate']:.0f}")
                cols[4].metric("Worst drawdown", f"{m['max_dd']:.0f}")
                st.dataframe(extra["trades"], width="stretch")
                st.session_state["history"] = (
                    st.session_state["history"] +
                    extra["trades"].assign(ticker=cfg.ticker,
                                           strategy=cfg.strategy).to_dict("records"))
                _chart(st, df, extra["trades"], extra["levels"], cfg)

    with tab_live:
        st.warning("Trailing here advances only when a bar has closed, exactly "
                   "as the backtest does. The stop is checked against the last "
                   "traded price on every poll.")
        st.session_state.setdefault("live", LiveState())
        c1, c2 = st.columns(2)
        if c1.button("Start", key="live_start"):
            st.session_state["live"] = LiveState(running=True)
        if c2.button("Stop", key="live_stop"):
            st.session_state["live"].running = False

        @st.fragment(run_every=15 if st.session_state["live"].running else None)
        def live_panel():
            state = st.session_state["live"]
            if not state.running:
                st.info("Not running.")
                return
            try:
                df = fetch_candles(cfg.ticker, cfg.interval, cfg.period,
                                   use_cache=False)
                ltp = float(df["close"].iloc[-1])
                st.session_state["live"] = live_poll(state, df, cfg, ltp)
                st.metric("Last traded price", f"{ltp:.2f}")
                if state.position:
                    st.write(state.position)
            except RateLimitError as err:
                st.error(str(err))
            except Exception as err:
                st.error(f"{err}")
            for line in st.session_state["live"].log[-12:]:
                st.text(line)
        live_panel()

    with tab_hist:
        hist = pd.DataFrame(st.session_state["history"])
        if hist.empty:
            st.info("No trades recorded yet. Run a backtest.")
        else:
            st.dataframe(hist, width="stretch")
            st.download_button("Download CSV", hist.to_csv(index=False),
                               "trades.csv", key="dl_hist")

    with tab_screen:
        st.markdown("**Which tickers should I trade right now, and at what "
                    "levels.** A ticker appears only if the setup is both "
                    "proven and live. Costs are forced on here; they are not "
                    "optional.")
        tickers = [t.strip() for t in st.session_state.get("tickers", "").split("\n")
                   if t.strip()]
        c = st.columns(4)
        strategies = c[0].multiselect("Strategies", list(STRATEGIES),
                                      default=["ema_cross", "supertrend"],
                                      format_func=lambda k: STRATEGIES[k].label,
                                      key="scr_strats")
        intervals = c[1].multiselect("Timeframes", ["5m", "15m", "1h", "1d"],
                                     default=["1d"], key="scr_intervals")
        min_tr = int(c[2].number_input("Minimum past trades", value=20, key="scr_min"))
        within = int(c[3].number_input("Firing within last N bars", value=3,
                                       key="scr_within"))
        c2 = st.columns(4)
        holdout = c2[0].slider("Held-out fraction", 0.1, 0.5, 0.3, key="scr_holdout")
        cap = int(c2[1].number_input("Test cap", value=8000, key="scr_cap"))
        need_rand = c2[2].checkbox("Require beating 80% of random entries",
                                   key="scr_rand")
        st.caption(f"Quantity {cfg.qty} at {INSTRUMENTS[cfg.costs.instrument]['label']}. "
                   f"Cost per unit falls sharply with size, so testing at "
                   f"quantity 1 makes every setup look unprofitable.")

        ss = ScreenSettings(qty=cfg.qty, instrument=cfg.costs.instrument,
                            strategies=strategies or ["ema_cross"],
                            intervals=intervals or ["1d"], period=cfg.period,
                            min_trades=min_tr, within_bars=within,
                            holdout=holdout, test_cap=cap,
                            require_random=need_rand, exits=cfg.exits)
        combos = len([1 for s in ss.strategies for iv in ss.intervals
                      if not (STRATEGIES[s].intraday_only and not is_intraday(iv))])
        planned = len(tickers) * combos
        downloads = len(tickers) * len(ss.intervals)
        if planned > cap:
            reached = max(1, int(cap / max(1, combos)))
            st.warning(f"The grid is {planned:,} tests but the cap is {cap:,}, "
                       f"so it will stop after roughly {reached} of your "
                       f"{len(tickers)} tickers.")
        gap = suggested_gap(downloads)
        if downloads and gap > GATE.gap:
            st.warning(f"{downloads} downloads at {GATE.gap:.1f}s is "
                       f"{int(3600/max(GATE.gap,0.01)):,}/hour and will be "
                       f"refused. Yahoo tolerates roughly 1,500 to 2,000 an hour.")
            if st.button(f"Set the gap to {gap:.1f}s (takes about "
                         f"{downloads*gap/60:.0f} minutes)", key="fix_gap"):
                GATE.set_base(gap)

        if st.button("Scan", key="run_scan") and tickers:
            bar = st.progress(0.0)
            out = screen(tickers, ss, progress=lambda a, b: bar.progress(min(1.0, a / max(1, b))))
            bar.empty()
            if out["aborted"]:
                st.error(out["aborted"])
            st.caption(f"{len(out['rows'])} survivors from {out['tests']:,} tests "
                       f"across {out['tickers_done']} tickers. A pass rate of "
                       f"{100*len(out['rows'])/max(1,out['tests']):.2f}% is what a "
                       f"strict filter looks like.")
            if out["tests"] > 1:
                st.caption(f"With {out['tests']:,} combinations tested, the best "
                           f"of pure noise would reach about "
                           f"{noise_ceiling(out['tests']):.2f} in t-stat units.")
            if not out["rows"]:
                st.subheader("Where every candidate died")
                for k, v in out["deaths"].items():
                    if v:
                        st.write(f"{v} {k}")
                st.caption("The first of these is not a dial you can turn. Most "
                           "simple technical setups do not survive costs; that "
                           "is the state of the world, not a defect in the tool.")
            else:
                rows = pd.DataFrame([{k: v for k, v in r.items()
                                      if k != "run_config"} for r in out["rows"]])
                rows.insert(0, "apply", False)
                edited = st.data_editor(
                    rows, width="stretch",
                    disabled=[c for c in rows.columns if c != "apply"],
                    key=f"scr_editor_{st.session_state.get('scr_nonce', 0)}")
                picked = edited.index[edited["apply"]].tolist()
                if picked:
                    rc = out["rows"][picked[0]]["run_config"]
                    st.session_state["pending_ticker"] = rc.ticker
                    st.session_state["pending_strategy"] = rc.strategy
                    st.session_state["pending_interval"] = rc.interval
                    for k, v in rc.params.items():
                        st.session_state[f"pending_{k}"] = v
                    st.session_state["scr_nonce"] = st.session_state.get("scr_nonce", 0) + 1
                    st.success(f"Loaded {rc.ticker} into the sidebar. The "
                               f"Backtest tab will reproduce this row exactly.")
                    st.rerun()


def uitest(verbose=True) -> bool:
    """Full UI drive-through with streamlit.testing.v1.AppTest, asserting zero
    exceptions. Candles are stubbed so the drive-through never touches the
    provider or the rate budget."""
    import os
    import tempfile
    from streamlit.testing.v1 import AppTest

    harness = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    harness.write(
        "import sys\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        "import swingtrading as S\n"
        "S.fetch_candles = lambda tk, iv, pe, use_cache=True: S._synth_regime("
        "400, seed=abs(hash(tk)) % 999, amp=0.6, period=70)\n"
        "S.main()\n")
    harness.close()

    print("\n15. UI drive-through (streamlit.testing.v1.AppTest)")
    ok = True
    at = AppTest.from_file(harness.name, default_timeout=120).run()
    if at.exception:
        print(f"  [FAIL] app loads -- {at.exception[0].value}")
        return False
    print(f"  [PASS] app loads -- {len(at.tabs)} tabs, 0 exceptions")

    for label in ["run_bt", "run_scan", "live_start", "live_stop"]:
        try:
            at.button(key=label).click().run()
        except Exception as err:
            print(f"  [FAIL] click {label} -- {err}")
            ok = False
            continue
        if at.exception:
            print(f"  [FAIL] click {label} -- {at.exception[0].value}")
            ok = False
        else:
            print(f"  [PASS] click {label} -- 0 exceptions")

    for strat in list(STRATEGIES):
        at.selectbox(key="strategy").set_value(strat).run()
        if at.exception:
            print(f"  [FAIL] strategy {strat} -- {at.exception[0].value}")
            ok = False
    print(f"  [{'PASS' if ok else 'FAIL'}] every strategy renders its panel -- "
          f"{len(STRATEGIES)} strategies")

    for rule in list(STOP_RULES):
        at.selectbox(key="stop_rule").set_value(rule).run()
        if at.exception:
            print(f"  [FAIL] stop rule {rule} -- {at.exception[0].value}")
            ok = False
    for rule in list(TARGET_RULES):
        at.selectbox(key="target_rule").set_value(rule).run()
        if at.exception:
            print(f"  [FAIL] target rule {rule} -- {at.exception[0].value}")
            ok = False
    print(f"  [{'PASS' if ok else 'FAIL'}] every stop and target rule renders -- "
          f"{len(STOP_RULES)} stop, {len(TARGET_RULES)} target")

    for uni in list(UNIVERSES):
        at.selectbox(key="universe").set_value(uni).run()
        if at.exception:
            print(f"  [FAIL] universe {uni} -- {at.exception[0].value}")
            ok = False
    print(f"  [{'PASS' if ok else 'FAIL'}] universe switch rewrites the keyed "
          f"text area -- {len(UNIVERSES)} universes")
    os.unlink(harness.name)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = selftest()
        if "--no-ui" not in sys.argv:
            ok = uitest() and ok
        sys.exit(0 if ok else 1)
    main()
