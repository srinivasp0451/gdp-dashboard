"""
Swing Desk — backtesting, live trading and order routing in one file.

    pip install streamlit pandas numpy plotly yfinance
    streamlit run swing_desk.py

Optional, only if you switch on broker routing:  pip install dhanhq

Every indicator is computed from first principles in the INDICATORS section
below. No TA-Lib, no pandas-ta, no `ta` package. The implementations follow
Pine Script's semantics so the numbers line up with a TradingView chart — see
the section header for exactly what that entails and where it can still drift.

Nothing here is financial advice.
"""
from __future__ import annotations

import functools
import smtplib
import ssl
import threading
import time
from dataclasses import asdict, dataclass, field
from email.message import EmailMessage

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# 1 · CONFIGURATION
# =============================================================================

APP_TITLE = "Swing Desk"
YF_MIN_DELAY = 0.3          # seconds between outbound data calls

UNIVERSE: dict[str, dict[str, str]] = {
    "Indices": {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
        "NIFTY MIDCAP 50": "^NSEMDCP50",
        "INDIA VIX": "^INDIAVIX",
    },
    "Stocks": {
        "KAYNES": "KAYNES.NS",
        "RELIANCE": "RELIANCE.NS",
        "HDFC BANK": "HDFCBANK.NS",
        "TCS": "TCS.NS",
        "INFOSYS": "INFY.NS",
        "ICICI BANK": "ICICIBANK.NS",
        "TATA MOTORS": "TATAMOTORS.NS",
        "SBIN": "SBIN.NS",
    },
    "Crypto": {"BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD", "SOLANA": "SOL-USD"},
    "Forex": {"USD/INR": "USDINR=X", "EUR/USD": "EURUSD=X", "GBP/INR": "GBPINR=X", "JPY/INR": "JPYINR=X"},
    "Commodities": {
        "GOLD (COMEX)": "GC=F",
        "SILVER (COMEX)": "SI=F",
        "CRUDE OIL": "CL=F",
        "GOLDBEES (NSE)": "GOLDBEES.NS",
        "SILVERBEES (NSE)": "SILVERBEES.NS",
    },
}
DEFAULT_GROUP, DEFAULT_NAME = "Indices", "NIFTY 50"

# Timeframe -> (allowed periods, default period). Bounded by provider limits:
# 1m gives 7 days, other intraday buckets 60 days, hourly 730 days.
TF_PERIODS: dict[str, tuple[list[str], str]] = {
    "1m":  (["1d", "5d", "7d"], "7d"),
    "3m":  (["1d", "5d", "7d"], "7d"),                       # resampled from 1m
    "5m":  (["1d", "5d", "7d", "1mo"], "1mo"),
    "15m": (["1d", "5d", "7d", "1mo"], "1mo"),
    "30m": (["1d", "5d", "7d", "1mo", "3mo"], "1mo"),
    "1h":  (["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"], "1mo"),
    "1d":  (["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "20y", "25y", "30y", "max"], "1y"),
    "1wk": (["6mo", "1y", "2y", "3y", "5y", "10y", "20y", "25y", "30y", "max"], "1y"),
}
DEFAULT_TF = "1m"
RESAMPLE_FROM: dict[str, tuple[str, str]] = {"3m": ("1m", "3min")}

# Dhan serves finer buckets; surfaced when the source is switched.
DHAN_TF_PERIODS: dict[str, tuple[list[str], str]] = {
    "1m":  (["1d", "5d"], "5d"),
    "5m":  (["1d", "5d"], "5d"),
    "15m": (["1d", "5d"], "5d"),
    "25m": (["1d", "5d"], "5d"),
    "60m": (["1d", "5d"], "5d"),
    "1d":  (["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], "1y"),
}

SIDES = ["Both", "Long only", "Short only"]
INSTRUMENTS = ["Equity intraday", "Equity delivery", "Futures", "Options"]

DHAN_SEGMENTS = {
    "Equity intraday": ("NSE_EQ", "INTRADAY"),
    "Equity delivery": ("NSE_EQ", "CNC"),
    "Index / stock options": ("NSE_FNO", "INTRADAY"),
    "Index / stock futures": ("NSE_FNO", "INTRADAY"),
    "Currency": ("NSE_CURRENCY", "INTRADAY"),
    "Commodity (MCX)": ("MCX_COMM", "INTRADAY"),
}
DHAN_SCRIP_MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"

OHLC = ["Open", "High", "Low", "Close", "Volume"]


# =============================================================================
# 2 · INDICATORS — computed manually, matching Pine Script / TradingView
# =============================================================================
# The one thing that trips up most Python ports is the seed. Pine's ta.ema and
# ta.rma are undefined for the first length-1 bars, take the SMA of the first
# `length` values as their starting point, and only then run the recursion.
# pandas' ewm(adjust=False) instead seeds with the very first value, which
# leaves a visible offset that decays but never quite disappears. Every
# recursive average below goes through _pine_recursive so the seed is right.
#
# Where TradingView can still disagree with this file:
#   * Different history depth. TradingView computes from the first bar it has
#     loaded. Ask for a longer period here and early values shift slightly.
#   * Different session or timezone settings on the chart, which change how
#     bars are bucketed and where VWAP resets.
#   * Dividend or split adjustment. This file requests unadjusted candles.
#   * Real-time ticks. A forming bar changes until it closes.
# Compare on closed bars, same symbol, same timeframe, same history depth.

def _pine_recursive(values: np.ndarray, alpha: float, length: int) -> np.ndarray:
    """Pine's recursive-average skeleton, shared by ta.ema and ta.rma.

    NaN until enough bars exist, SMA of the first `length` values as the seed,
    then out[i] = alpha * x[i] + (1 - alpha) * out[i-1]. Leading NaNs in the
    input are skipped, which is what lets rma(plusDM) and ema(macd) seed at the
    right bar even though their inputs start undefined.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    valid = ~np.isnan(x)
    if not valid.any():
        return out
    start = int(np.argmax(valid))
    if x.size - start < length:
        return out
    seed = start + length - 1
    out[seed] = x[start:seed + 1].mean()
    for i in range(seed + 1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def pine_sma(s: pd.Series, length: int) -> pd.Series:
    """ta.sma"""
    return s.rolling(length, min_periods=length).mean()


def pine_ema(s: pd.Series, length: int) -> pd.Series:
    """ta.ema — alpha = 2 / (length + 1), seeded with the SMA."""
    return pd.Series(_pine_recursive(s.to_numpy(float), 2.0 / (length + 1), length),
                     index=s.index, name=f"EMA{length}")


def pine_rma(s: pd.Series, length: int) -> pd.Series:
    """ta.rma — Wilder's smoothing, alpha = 1 / length, seeded with the SMA."""
    return pd.Series(_pine_recursive(s.to_numpy(float), 1.0 / length, length),
                     index=s.index, name=f"RMA{length}")


def pine_stdev(s: pd.Series, length: int) -> pd.Series:
    """ta.stdev — population standard deviation, not the sample estimate."""
    return s.rolling(length, min_periods=length).std(ddof=0)


def pine_tr(df: pd.DataFrame, handle_na: bool = True) -> pd.Series:
    """ta.tr(handle_na).

    With handle_na the first bar falls back to high - low. Without it, the
    first bar is undefined — ta.atr uses the former, ta.dmi the latter, and
    that single bar shifts where their averages seed.
    """
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    if handle_na:
        tr.iloc[0] = df["High"].iloc[0] - df["Low"].iloc[0]
    else:
        tr.iloc[0] = np.nan
    return tr


def pine_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ta.atr = ta.rma(ta.tr(true), length)"""
    return pine_rma(pine_tr(df, handle_na=True), length)


def pine_rsi(s: pd.Series, length: int = 14) -> pd.Series:
    """ta.rsi = 100 - 100 / (1 + rma(gain) / rma(loss))"""
    d = s.diff()
    up = pine_rma(d.clip(lower=0), length)
    dn = pine_rma((-d).clip(lower=0), length)
    rsi = 100.0 - 100.0 / (1.0 + up / dn)
    rsi = rsi.where(dn != 0, 100.0)          # no losses over the window
    rsi = rsi.where(up != 0, 0.0)            # no gains over the window
    return rsi.where(up.notna() & dn.notna())


def pine_macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """ta.macd — note the signal line is an EMA of a series that itself starts
    undefined, so it seeds `signal` bars after the MACD line appears."""
    line = pine_ema(s, fast) - pine_ema(s, slow)
    sig = pine_ema(line, signal)
    return line, sig, line - sig


def pine_bb(s: pd.Series, length: int = 20, mult: float = 2.0):
    """ta.bb"""
    basis = pine_sma(s, length)
    dev = mult * pine_stdev(s, length)
    return basis - dev, basis, basis + dev


def pine_dmi(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14):
    """ta.dmi — returns (adx, +DI, -DI).

    Two details that are easy to get wrong: the true range here is ta.tr, not
    ta.tr(true), so it seeds one bar later than ta.atr; and DX divides by 1
    rather than 0 when +DI and -DI are both zero.
    """
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    trur = pine_rma(pine_tr(df, handle_na=False), di_len)
    plus = 100.0 * pine_rma(plus_dm, di_len) / trur
    minus = 100.0 * pine_rma(minus_dm, di_len) / trur
    total = plus + minus
    dx = (plus - minus).abs() / total.where(total != 0, 1.0)
    adx = 100.0 * pine_rma(dx, adx_len)
    return adx, plus, minus


def pine_supertrend(df: pd.DataFrame, length: int = 10, mult: float = 3.0):
    """ta.supertrend, transcribed from the Pine reference.

    Pine's direction is -1 for an uptrend. This returns the opposite sign so
    +1 means bullish, which reads more naturally everywhere else in this file.
    """
    atr = pine_atr(df, length).to_numpy(float)
    hl2 = ((df["High"] + df["Low"]) / 2.0).to_numpy(float)
    close = df["Close"].to_numpy(float)
    n = len(df)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    strend = np.full(n, np.nan)

    for i in range(n):
        raw_up = hl2[i] + mult * atr[i]
        raw_lo = hl2[i] - mult * atr[i]
        prev_up = upper[i - 1] if i > 0 and not np.isnan(upper[i - 1]) else 0.0   # nz()
        prev_lo = lower[i - 1] if i > 0 and not np.isnan(lower[i - 1]) else 0.0
        prev_close = close[i - 1] if i > 0 else np.nan

        lower[i] = raw_lo if (raw_lo > prev_lo or (prev_close < prev_lo)) else prev_lo
        upper[i] = raw_up if (raw_up < prev_up or (prev_close > prev_up)) else prev_up

        if i == 0 or np.isnan(atr[i - 1]):
            direction[i] = 1.0                                   # Pine: downtrend
        elif strend[i - 1] == prev_up:
            direction[i] = -1.0 if close[i] > upper[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] < lower[i] else -1.0
        strend[i] = lower[i] if direction[i] == -1.0 else upper[i]

    line = pd.Series(strend, index=df.index).where(~np.isnan(atr))
    trend = pd.Series(-direction, index=df.index).where(~np.isnan(atr))   # +1 bullish
    return line, trend


def pine_vwap(df: pd.DataFrame) -> pd.Series:
    """ta.vwap on hlc3, anchored to the session and reset each trading day."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    day = pd.Series(df.index.date, index=df.index)
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vol = df["Volume"].groupby(day).cumsum()
    return pv / vol.where(vol != 0)


def pine_highest(s: pd.Series, length: int) -> pd.Series:
    """ta.highest — inclusive of the current bar."""
    return s.rolling(length, min_periods=length).max()


def pine_lowest(s: pd.Series, length: int) -> pd.Series:
    """ta.lowest — inclusive of the current bar."""
    return s.rolling(length, min_periods=length).min()


def pine_pivots(df: pd.DataFrame, left: int, right: int | None = None):
    """ta.pivotlow / ta.pivothigh, forward filled into a usable level.

    A pivot at bar i needs `right` further bars before it can be confirmed, so
    the confirmed level is shifted forward by that many bars. Nothing here can
    see a pivot earlier than a live chart would print it. Ties do not count as
    pivots, matching Pine's strict comparison.
    """
    right = left if right is None else right
    lo, hi = df["Low"], df["High"]
    win = left + right + 1
    is_low = (lo == lo.rolling(win, center=True).min()) & \
             (lo.rolling(win, center=True).apply(lambda w: (w == w[left]).sum(), raw=True) == 1)
    is_high = (hi == hi.rolling(win, center=True).max()) & \
              (hi.rolling(win, center=True).apply(lambda w: (w == w[left]).sum(), raw=True) == 1)
    return lo.where(is_low).shift(right).ffill(), hi.where(is_high).shift(right).ffill()


def running_extreme(df: pd.DataFrame, length: int):
    """The unconfirmed 'current' swing: the rolling extreme of recent bars."""
    return (df["Low"].rolling(length, min_periods=1).min(),
            df["High"].rolling(length, min_periods=1).max())


# =============================================================================
# 3 · DATA ACCESS
# =============================================================================

_rate_lock = threading.Lock()
_last_call = {"t": 0.0}


class DataError(RuntimeError):
    """A provider returned nothing usable."""


def throttle(min_delay: float = YF_MIN_DELAY) -> None:
    """Serialise outbound requests with a minimum gap.

    Live polling on 1-minute bars will get a session banned without this. The
    gate is process-wide, so the backtest loader and the live poll share it.
    """
    with _rate_lock:
        wait = min_delay - (time.time() - _last_call["t"])
        if wait > 0:
            time.sleep(wait)
        _last_call["t"] = time.time()


def _flatten(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    for c in OHLC:
        if c not in df.columns:
            df[c] = 0.0 if c == "Volume" else np.nan
    return df[OHLC]


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="right", closed="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def fetch_yf(symbol: str, interval: str, period: str, delay: float = YF_MIN_DELAY) -> pd.DataFrame:
    import yfinance as yf

    native, rule = RESAMPLE_FROM.get(interval, (interval, None))
    throttle(delay)
    raw = yf.Ticker(symbol).history(period=period, interval=native, auto_adjust=False)
    if raw is None or raw.empty:
        raise DataError(f"yfinance returned no rows for {symbol} at {native}/{period}.")
    df = _flatten(raw, symbol)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if rule:
        df = resample_ohlc(df, rule)
    df.index.name = "Datetime"
    return df


def _period_days(period: str) -> int:
    if period == "max":
        return 365 * 25
    for suffix, mult in (("mo", 30), ("wk", 7), ("y", 365), ("d", 1)):
        if period.endswith(suffix):
            return int(period[: -len(suffix)]) * mult
    return 365


def fetch_dhan(client, security_id: str, exchange_segment: str, instrument: str,
               interval: str, period: str) -> pd.DataFrame:
    """Dhan historical OHLC. Minute buckets available: 1, 5, 15, 25, 60."""
    throttle(0.3)
    if interval.endswith("m"):
        res = client.intraday_minute_data(security_id=str(security_id),
                                          exchange_segment=exchange_segment,
                                          instrument_type=instrument,
                                          interval=int(interval.rstrip("m")))
    else:
        end = pd.Timestamp.now().normalize()
        start = end - pd.Timedelta(days=_period_days(period))
        res = client.historical_daily_data(security_id=str(security_id),
                                           exchange_segment=exchange_segment,
                                           instrument_type=instrument,
                                           from_date=start.strftime("%Y-%m-%d"),
                                           to_date=end.strftime("%Y-%m-%d"))
    payload = (res or {}).get("data") or {}
    if not payload:
        raise DataError(f"Dhan returned no candles. Response: {res}")
    df = pd.DataFrame(payload)
    key = "timestamp" if "timestamp" in df else "start_Time"
    df.index = pd.to_datetime(df[key], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.rename(columns=str.title)[OHLC]
    df.index.name = "Datetime"
    return df.sort_index()


def ltp_yf(symbol: str, delay: float = YF_MIN_DELAY) -> float | None:
    """Best-effort last traded price, falling back to the newest 1-minute close."""
    import yfinance as yf

    throttle(delay)
    try:
        info = yf.Ticker(symbol).fast_info
        for key in ("last_price", "lastPrice", "regular_market_price"):
            val = info.get(key) if isinstance(info, dict) else getattr(info, key, None)
            if val:
                return float(val)
    except Exception:
        pass
    try:
        return float(fetch_yf(symbol, "1m", "1d", delay)["Close"].iloc[-1])
    except Exception:
        return None


# =============================================================================
# 4 · COST MODEL
# =============================================================================
# Published Indian market rates at the time of writing, editable in the sidebar.
# Check them against your own contract notes before trusting a P&L figure.

COST_DEFAULTS = {
    "Equity intraday": dict(brokerage_pct=0.03, brokerage_cap=20.0, stt_sell=0.025, stt_buy=0.0,
                            txn=0.00297, stamp_buy=0.003, sebi=0.0001, gst=18.0),
    "Equity delivery": dict(brokerage_pct=0.0, brokerage_cap=0.0, stt_sell=0.1, stt_buy=0.1,
                            txn=0.00297, stamp_buy=0.015, sebi=0.0001, gst=18.0),
    "Futures":         dict(brokerage_pct=0.03, brokerage_cap=20.0, stt_sell=0.02, stt_buy=0.0,
                            txn=0.00173, stamp_buy=0.002, sebi=0.0001, gst=18.0),
    "Options":         dict(brokerage_pct=0.0, brokerage_cap=20.0, stt_sell=0.1, stt_buy=0.0,
                            txn=0.03503, stamp_buy=0.003, sebi=0.0001, gst=18.0),
}


@dataclass
class CostModel:
    instrument: str = "Equity intraday"
    slippage_unit: str = "Points"
    slippage: float = 0.0
    brokerage_pct: float = 0.03
    brokerage_cap: float = 20.0
    stt_sell: float = 0.025
    stt_buy: float = 0.0
    txn: float = 0.00297
    stamp_buy: float = 0.003
    sebi: float = 0.0001
    gst: float = 18.0
    enabled: bool = False

    def slip(self, price: float) -> float:
        return price * self.slippage / 100 if self.slippage_unit == "Percent" else self.slippage

    def fill(self, price: float, side: int, entering: bool) -> float:
        """Buys fill higher, sells fill lower. Slippage always works against you."""
        if not self.enabled:
            return price
        buying = (side > 0) == entering
        return price + self.slip(price) * (1 if buying else -1)

    def _brokerage(self, turnover: float) -> float:
        raw = turnover * self.brokerage_pct / 100
        return min(raw, self.brokerage_cap) if self.brokerage_cap else raw

    def charges(self, entry: float, exit_: float, qty: int, side: int) -> dict:
        if not self.enabled:
            return {"total": 0.0}
        buy_val = (entry if side > 0 else exit_) * qty
        sell_val = (exit_ if side > 0 else entry) * qty
        turnover = buy_val + sell_val
        brokerage = self._brokerage(buy_val) + self._brokerage(sell_val)
        stt = sell_val * self.stt_sell / 100 + buy_val * self.stt_buy / 100
        txn = turnover * self.txn / 100
        sebi = turnover * self.sebi / 100
        stamp = buy_val * self.stamp_buy / 100
        gst = (brokerage + txn + sebi) * self.gst / 100
        return {"brokerage": brokerage, "stt": stt, "exchange": txn, "sebi": sebi, "stamp": stamp,
                "gst": gst, "total": brokerage + stt + txn + sebi + stamp + gst}

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# 5 · STRATEGIES
# =============================================================================
# Contract: return +1 (go long), -1 (go short) or 0 (stand aside), indexed like
# the input frame. A signal printed on bar N is acted on at the open of bar
# N+1, never on bar N itself.


@dataclass
class Signals:
    signal: pd.Series
    overlays: dict[str, pd.Series] = field(default_factory=dict)   # drawn on price
    panes: dict[str, pd.Series] = field(default_factory=dict)      # drawn below price
    levels: dict[str, float] = field(default_factory=dict)         # static h-lines
    note: str = ""


def cross_up(a: pd.Series, b) -> pd.Series:
    """ta.crossover"""
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_dn(a: pd.Series, b) -> pd.Series:
    """ta.crossunder"""
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))


def _sig(idx, longs, shorts) -> pd.Series:
    s = pd.Series(0, index=idx, dtype=int)
    s[shorts.fillna(False)] = -1
    s[longs.fillna(False)] = 1
    return s


def p_simple(st_, k):
    return {"every_bar": st_.checkbox("Re-enter on every flat bar", value=True, key=f"{k}_eb")}


def c_simple_buy(df, p):
    s = pd.Series(1 if p["every_bar"] else 0, index=df.index, dtype=int)
    if not p["every_bar"]:
        s.iloc[0] = 1
    return Signals(s, note="Unconditional long. Use it to sanity-check costs and exits.")


def c_simple_sell(df, p):
    s = pd.Series(-1 if p["every_bar"] else 0, index=df.index, dtype=int)
    if not p["every_bar"]:
        s.iloc[0] = -1
    return Signals(s, note="Unconditional short. Same purpose as simple buy.")


def p_threshold(st_, k):
    c1, c2 = st_.columns(2)
    d = c1.selectbox("Cross direction", ["Crosses up through level", "Crosses down through level"],
                     key=f"{k}_dir")
    mode = c2.selectbox("Level is", ["Absolute price", "Percent from reference"], key=f"{k}_mode")
    if mode == "Absolute price":
        val = st_.number_input("Level", value=0.0, step=0.05, format="%.4f", key=f"{k}_abs")
    else:
        val = st_.number_input("Percent offset (%)  ·  negative sits below the reference",
                               value=1.0, step=0.1, format="%.2f", key=f"{k}_pct")
    ref = st_.selectbox("Reference price", ["First close of window", "Last close of window"], key=f"{k}_ref")
    return {"dir": d, "mode": mode, "val": val, "ref": ref}


def c_threshold(df, p):
    base = float(df["Close"].iloc[0] if p["ref"].startswith("First") else df["Close"].iloc[-1])
    level = p["val"] if p["mode"] == "Absolute price" else base * (1 + p["val"] / 100)
    up = p["dir"].startswith("Crosses up")
    hit = cross_up(df["Close"], level) if up else cross_dn(df["Close"], level)
    empty = pd.Series(False, index=df.index)
    return Signals(_sig(df.index, hit if up else empty, empty if up else hit),
                   levels={"Trigger": level},
                   note=f"Trigger sits at {level:,.2f}, "
                        f"{'above' if level > base else 'below'} the reference {base:,.2f}.")


def p_ema(st_, k):
    c1, c2, c3 = st_.columns(3)
    return {"fast": c1.number_input("Fast EMA", 2, 400, 9, key=f"{k}_f"),
            "slow": c2.number_input("Slow EMA", 3, 800, 21, key=f"{k}_s"),
            "trend": c3.number_input("Trend EMA (0 = off)", 0, 800, 50, key=f"{k}_t")}


def c_ema(df, p):
    f, s = pine_ema(df["Close"], int(p["fast"])), pine_ema(df["Close"], int(p["slow"]))
    ov = {f"EMA {p['fast']}": f, f"EMA {p['slow']}": s}
    lc, sc = cross_up(f, s), cross_dn(f, s)
    if p["trend"]:
        t = pine_ema(df["Close"], int(p["trend"]))
        ov[f"EMA {p['trend']}"] = t
        lc, sc = lc & (df["Close"] > t), sc & (df["Close"] < t)
    return Signals(_sig(df.index, lc, sc), overlays=ov)


def p_supertrend(st_, k):
    c1, c2 = st_.columns(2)
    return {"n": c1.number_input("ATR length", 2, 100, 10, key=f"{k}_n"),
            "m": c2.number_input("Multiplier", 0.5, 10.0, 3.0, 0.1, key=f"{k}_m")}


def c_supertrend(df, p):
    line, trend = pine_supertrend(df, int(p["n"]), float(p["m"]))
    return Signals(_sig(df.index, (trend == 1) & (trend.shift(1) == -1),
                        (trend == -1) & (trend.shift(1) == 1)),
                   overlays={"Supertrend": line})


def p_rsi(st_, k):
    c1, c2, c3 = st_.columns(3)
    n = c1.number_input("RSI length", 2, 100, 14, key=f"{k}_n")
    lo = c2.number_input("Oversold", 1, 49, 30, key=f"{k}_lo")
    hi = c3.number_input("Overbought", 51, 99, 70, key=f"{k}_hi")
    mode = st_.radio("Trade the", ["Reversal (buy oversold)", "Momentum (buy strength)"],
                     horizontal=True, key=f"{k}_mode")
    return {"n": n, "lo": lo, "hi": hi, "mode": mode}


def c_rsi(df, p):
    r = pine_rsi(df["Close"], int(p["n"]))
    if p["mode"].startswith("Reversal"):
        lc, sc = cross_up(r, p["lo"]), cross_dn(r, p["hi"])
    else:
        lc, sc = cross_up(r, p["hi"]), cross_dn(r, p["lo"])
    return Signals(_sig(df.index, lc, sc), panes={f"RSI {p['n']}": r})


def p_macd(st_, k):
    c1, c2, c3 = st_.columns(3)
    return {"f": c1.number_input("Fast", 2, 200, 12, key=f"{k}_f"),
            "s": c2.number_input("Slow", 3, 400, 26, key=f"{k}_s"),
            "g": c3.number_input("Signal", 2, 100, 9, key=f"{k}_g"),
            "zero": st_.checkbox("Require the MACD line on the right side of zero", key=f"{k}_z")}


def c_macd(df, p):
    line, sig, _ = pine_macd(df["Close"], int(p["f"]), int(p["s"]), int(p["g"]))
    lc, sc = cross_up(line, sig), cross_dn(line, sig)
    if p["zero"]:
        lc, sc = lc & (line > 0), sc & (line < 0)
    return Signals(_sig(df.index, lc, sc), panes={"MACD": line, "Signal": sig})


def p_bb(st_, k):
    c1, c2 = st_.columns(2)
    n = c1.number_input("Length", 5, 300, 20, key=f"{k}_n")
    k_ = c2.number_input("Std-dev multiple", 0.5, 5.0, 2.0, 0.1, key=f"{k}_k")
    mode = st_.radio("Trade the", ["Breakout", "Mean reversion"], horizontal=True, key=f"{k}_m")
    return {"n": n, "k": k_, "mode": mode}


def c_bb(df, p):
    lo, mid, hi = pine_bb(df["Close"], int(p["n"]), float(p["k"]))
    ov = {"BB lower": lo, "BB basis": mid, "BB upper": hi}
    if p["mode"] == "Breakout":
        lc, sc = cross_up(df["Close"], hi), cross_dn(df["Close"], lo)
    else:
        lc, sc = cross_up(df["Close"], lo), cross_dn(df["Close"], hi)
    return Signals(_sig(df.index, lc, sc), overlays=ov)


def p_donchian(st_, k):
    return {"n": st_.number_input("Channel length", 5, 400, 20, key=f"{k}_n")}


def c_donchian(df, p):
    plo = pine_lowest(df["Low"], int(p["n"])).shift(1)
    phi = pine_highest(df["High"], int(p["n"])).shift(1)
    return Signals(_sig(df.index, df["High"] > phi, df["Low"] < plo),
                   overlays={"Donchian high": phi, "Donchian low": plo})


def p_vwap(st_, k):
    return {"buf": st_.number_input("Reclaim buffer (%)", 0.0, 5.0, 0.05, 0.01, key=f"{k}_b")}


def c_vwap(df, p):
    v = pine_vwap(df)
    b = float(p["buf"]) / 100
    return Signals(_sig(df.index, cross_up(df["Close"], v * (1 + b)),
                        cross_dn(df["Close"], v * (1 - b))), overlays={"VWAP": v})


def p_orb(st_, k):
    c1, c2 = st_.columns(2)
    return {"mins": c1.number_input("Opening range (minutes)", 1, 240, 15, key=f"{k}_m"),
            "once": c2.checkbox("One trade per day", value=True, key=f"{k}_o")}


def c_orb(df, p):
    day = pd.Series(df.index.date, index=df.index)
    start = df.groupby(day).apply(lambda g: g.index[0]).reindex(day.values).values
    elapsed = (df.index.values - start) / np.timedelta64(1, "m")
    inrange = pd.Series(elapsed < int(p["mins"]), index=df.index)
    hi = df["High"].where(inrange).groupby(day).cummax().groupby(day).ffill()
    lo = df["Low"].where(inrange).groupby(day).cummin().groupby(day).ffill()
    lc = (~inrange) & (df["Close"] > hi)
    sc = (~inrange) & (df["Close"] < lo)
    if p["once"]:
        lc &= lc.groupby(day).cumsum().eq(1)
        sc &= sc.groupby(day).cumsum().eq(1)
    return Signals(_sig(df.index, lc, sc), overlays={"OR high": hi, "OR low": lo})


def p_adx_trend(st_, k):
    c1, c2, c3 = st_.columns(3)
    return {"n": c1.number_input("DI length", 2, 100, 14, key=f"{k}_n"),
            "a": c2.number_input("ADX smoothing", 2, 100, 14, key=f"{k}_a"),
            "min": c3.number_input("Minimum ADX", 0, 100, 25, key=f"{k}_min")}


def c_adx_trend(df, p):
    a, pdi, mdi = pine_dmi(df, int(p["n"]), int(p["a"]))
    strong = a > p["min"]
    return Signals(_sig(df.index, cross_up(pdi, mdi) & strong, cross_dn(pdi, mdi) & strong),
                   panes={"ADX": a, "+DI": pdi, "-DI": mdi})


STRATEGIES: dict[str, dict] = {
    "Simple buy":              {"params": p_simple,     "compute": c_simple_buy},
    "Simple sell":             {"params": p_simple,     "compute": c_simple_sell},
    "Price crosses threshold": {"params": p_threshold,  "compute": c_threshold},
    "EMA crossover":           {"params": p_ema,        "compute": c_ema},
    "Supertrend flip":         {"params": p_supertrend, "compute": c_supertrend},
    "RSI reversal / momentum": {"params": p_rsi,        "compute": c_rsi},
    "MACD crossover":          {"params": p_macd,       "compute": c_macd},
    "Bollinger band":          {"params": p_bb,         "compute": c_bb},
    "Donchian breakout":       {"params": p_donchian,   "compute": c_donchian},
    "VWAP reclaim":            {"params": p_vwap,       "compute": c_vwap},
    "Opening range breakout":  {"params": p_orb,        "compute": c_orb},
    "ADX trend ride":          {"params": p_adx_trend,  "compute": c_adx_trend},
}
DEFAULT_STRATEGY = "EMA crossover"


# =============================================================================
# 6 · ENTRY FILTERS
# =============================================================================
# Each returns (allow_long, allow_short) masks. Filters gate an existing signal
# and never create one. Combine with AND (all must agree) or OR (any one).


def f_bollinger(st_, k):
    c1, c2 = st_.columns(2)
    n = c1.number_input("Length", 5, 300, 20, key=f"{k}_n")
    m = c2.number_input("Std-dev multiple", 0.5, 5.0, 2.0, 0.1, key=f"{k}_m")
    where = st_.selectbox("Allow entries when price is",
                          ["Outside the bands", "Inside the bands",
                           "Above the basis / below the basis"], key=f"{k}_w")
    return {"n": n, "m": m, "where": where}


def a_bollinger(df, p):
    lo, mid, hi = pine_bb(df["Close"], int(p["n"]), float(p["m"]))
    c = df["Close"]
    if p["where"] == "Outside the bands":
        return c > hi, c < lo
    if p["where"] == "Inside the bands":
        inside = (c <= hi) & (c >= lo)
        return inside, inside
    return c > mid, c < mid


def f_ema(st_, k):
    c1, c2 = st_.columns(2)
    return {"fast": c1.number_input("Fast EMA", 2, 400, 20, key=f"{k}_f"),
            "slow": c2.number_input("Slow EMA", 3, 800, 50, key=f"{k}_s")}


def a_ema(df, p):
    f = pine_ema(df["Close"], int(p["fast"]))
    s = pine_ema(df["Close"], int(p["slow"]))
    return f > s, f < s


def f_adx(st_, k):
    c1, c2, c3 = st_.columns(3)
    n = c1.number_input("DI length", 2, 100, 14, key=f"{k}_n")
    lo = c2.number_input("Min ADX", 0, 100, 20, key=f"{k}_lo")
    hi = c3.number_input("Max ADX", 0, 100, 60, key=f"{k}_hi")
    a = st_.number_input("ADX smoothing", 2, 100, 14, key=f"{k}_a")
    di = st_.checkbox("Also require +DI / -DI to point the same way as the trade", key=f"{k}_di")
    return {"n": n, "a": a, "min": lo, "max": hi, "di": di}


def a_adx(df, p):
    a, pdi, mdi = pine_dmi(df, int(p["n"]), int(p["a"]))
    band = (a >= p["min"]) & (a <= p["max"])
    if p["di"]:
        return band & (pdi > mdi), band & (mdi > pdi)
    return band, band


def f_rsi(st_, k):
    c1, c2, c3 = st_.columns(3)
    return {"n": c1.number_input("Length", 2, 100, 14, key=f"{k}_n"),
            "lmin": c2.number_input("Long: RSI above", 0, 100, 50, key=f"{k}_lm"),
            "smax": c3.number_input("Short: RSI below", 0, 100, 50, key=f"{k}_sm")}


def a_rsi(df, p):
    r = pine_rsi(df["Close"], int(p["n"]))
    return r >= p["lmin"], r <= p["smax"]


def f_regime(st_, k):
    c1, c2 = st_.columns(2)
    n = c1.number_input("Regime EMA", 10, 1000, 200, key=f"{k}_n")
    slope = c2.number_input("Slope lookback (bars)", 1, 200, 20, key=f"{k}_s")
    mode = st_.selectbox("Regime rule", ["Price vs EMA", "EMA slope", "Both must agree"], key=f"{k}_m")
    return {"n": n, "s": slope, "mode": mode}


def a_regime(df, p):
    e = pine_ema(df["Close"], int(p["n"]))
    pv_l, pv_s = df["Close"] > e, df["Close"] < e
    sl = e.diff(int(p["s"]))
    sl_l, sl_s = sl > 0, sl < 0
    if p["mode"] == "Price vs EMA":
        return pv_l, pv_s
    if p["mode"] == "EMA slope":
        return sl_l, sl_s
    return pv_l & sl_l, pv_s & sl_s


def f_volatility(st_, k):
    c1, c2 = st_.columns(2)
    return {"n": c1.number_input("ATR length", 2, 100, 14, key=f"{k}_n"),
            "min": c2.number_input("Min ATR as % of price", 0.0, 20.0, 0.05, 0.01, key=f"{k}_m")}


def a_volatility(df, p):
    ok = (100 * pine_atr(df, int(p["n"])) / df["Close"]) >= p["min"]
    return ok, ok


def f_volume(st_, k):
    c1, c2 = st_.columns(2)
    return {"n": c1.number_input("Average over (bars)", 2, 500, 20, key=f"{k}_n"),
            "x": c2.number_input("Volume must exceed average by (x)", 0.1, 10.0, 1.2, 0.1, key=f"{k}_x")}


def a_volume(df, p):
    ok = df["Volume"] > pine_sma(df["Volume"], int(p["n"])) * float(p["x"])
    return ok, ok


def f_session(st_, k):
    c1, c2 = st_.columns(2)
    return {"start": c1.time_input("No entries before", value=pd.Timestamp("09:20").time(), key=f"{k}_a"),
            "end": c2.time_input("No entries after", value=pd.Timestamp("15:00").time(), key=f"{k}_b")}


def a_session(df, p):
    t = pd.Series(df.index.time, index=df.index)
    ok = (t >= p["start"]) & (t <= p["end"])
    return ok, ok


FILTERS: dict[str, dict] = {
    "Bollinger band":   {"params": f_bollinger,  "apply": a_bollinger},
    "EMA alignment":    {"params": f_ema,        "apply": a_ema},
    "ADX band":         {"params": f_adx,        "apply": a_adx},
    "RSI":              {"params": f_rsi,        "apply": a_rsi},
    "Market regime":    {"params": f_regime,     "apply": a_regime},
    "Volatility floor": {"params": f_volatility, "apply": a_volatility},
    "Volume surge":     {"params": f_volume,     "apply": a_volume},
    "Trading session":  {"params": f_session,    "apply": a_session},
}


def combine_filters(df: pd.DataFrame, selected: dict[str, dict], logic: str):
    """Fold the active filters into one (allow_long, allow_short) pair."""
    if not selected:
        t = pd.Series(True, index=df.index)
        return t, t.copy()
    longs, shorts = [], []
    for name, params in selected.items():
        l, s = FILTERS[name]["apply"](df, params)
        longs.append(l.fillna(False))
        shorts.append(s.fillna(False))
    L, S = pd.concat(longs, axis=1), pd.concat(shorts, axis=1)
    return (L.all(axis=1), S.all(axis=1)) if logic == "AND" else (L.any(axis=1), S.any(axis=1))


# =============================================================================
# 7 · STOP LOSS AND TARGET ENGINE
# =============================================================================
# Backtest and live share this code so both obey identical rules. Stops ratchet
# in the trade's favour only; they never loosen.

SL_MODES = [
    "Fixed points",
    "Fixed percent",
    "Trailing points",
    "Trailing previous swing low / high",
    "Trailing current swing low / high",
    "Trailing previous candle low / high",
    "Trailing current candle low / high",
    "ATR multiple",
    "Derived from reward (risk:reward)",
    "Strategy signal only (no price stop)",
]
TGT_MODES = [
    "Fixed points",
    "Fixed percent",
    "Trailing target (display only)",
    "Trailing previous swing high / low",
    "Trailing current swing high / low",
    "Trailing previous candle high / low",
    "Trailing current candle high / low",
    "Risk:reward multiple",
    "ATR multiple",
    "Strategy reversal exit (no fixed target)",
]


@dataclass
class Position:
    side: int                       # +1 long, -1 short
    qty: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_reason: str
    entry_bar: int
    sl: float = np.nan
    target: float = np.nan
    init_risk: float = np.nan
    peak: float = -np.inf
    trough: float = np.inf
    be_done: bool = False
    trail_armed: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "LONG" if self.side > 0 else "SHORT"


def build_context(df: pd.DataFrame, sl_cfg: dict, tgt_cfg: dict) -> pd.DataFrame:
    """Everything the exit rules can read on a given bar, computed once."""
    ctx = pd.DataFrame(index=df.index)
    an = int(sl_cfg.get("atr_len") or tgt_cfg.get("atr_len") or 14)
    ctx["atr"] = pine_atr(df, an)
    sn = int(sl_cfg.get("swing_n") or tgt_cfg.get("swing_n") or 3)
    ctx["piv_lo"], ctx["piv_hi"] = pine_pivots(df, sn)
    ctx["run_lo"], ctx["run_hi"] = running_extreme(df, sn)
    ctx["prev_lo"], ctx["prev_hi"] = df["Low"].shift(1), df["High"].shift(1)
    ctx["cur_lo"], ctx["cur_hi"] = df["Low"], df["High"]
    return ctx


def _raw_stop(pos: Position, ctx: pd.Series, cfg: dict, ref: float) -> float:
    """Candidate stop from the configured rule, before ratcheting."""
    m, s = cfg["mode"], pos.side
    if m == "Fixed points":
        return pos.entry_price - s * cfg["points"]
    if m == "Fixed percent":
        return pos.entry_price * (1 - s * cfg["pct"] / 100)
    if m == "Trailing points":
        return ref - s * cfg["points"]
    if m == "Trailing previous swing low / high":
        return float(ctx["piv_lo"] if s > 0 else ctx["piv_hi"]) - s * cfg.get("buffer", 0.0)
    if m == "Trailing current swing low / high":
        return float(ctx["run_lo"] if s > 0 else ctx["run_hi"]) - s * cfg.get("buffer", 0.0)
    if m == "Trailing previous candle low / high":
        return float(ctx["prev_lo"] if s > 0 else ctx["prev_hi"]) - s * cfg.get("buffer", 0.0)
    if m == "Trailing current candle low / high":
        return float(ctx["cur_lo"] if s > 0 else ctx["cur_hi"]) - s * cfg.get("buffer", 0.0)
    if m == "ATR multiple":
        anchor = ref if cfg.get("atr_trail", True) else pos.entry_price
        return anchor - s * cfg["atr_mult"] * float(ctx["atr"])
    if m == "Derived from reward (risk:reward)":
        return pos.entry_price - s * (cfg.get("reward_points", 20.0) / max(cfg.get("rr", 2.0), 0.01))
    return np.nan


def init_stop(pos: Position, ctx: pd.Series, cfg: dict) -> None:
    sl = _raw_stop(pos, ctx, cfg, pos.entry_price)
    if not np.isnan(sl):
        # Never open a trade with the stop already on the wrong side of entry.
        sl = min(sl, pos.entry_price - 1e-9) if pos.side > 0 else max(sl, pos.entry_price + 1e-9)
    pos.sl = sl
    pos.init_risk = abs(pos.entry_price - sl) if not np.isnan(sl) else np.nan


def update_stop(pos: Position, ctx: pd.Series, cfg: dict, ref_high: float, ref_low: float) -> None:
    """Called at each bar close in the backtest, each poll when live."""
    pos.peak = max(pos.peak, ref_high)
    pos.trough = min(pos.trough, ref_low)
    ref = pos.peak if pos.side > 0 else pos.trough

    be = cfg.get("be", {})
    if be.get("on") and not pos.be_done:
        gain = (ref - pos.entry_price) * pos.side
        if be["unit"] == "Percent":
            need = pos.entry_price * be["value"] / 100
        elif be["unit"] == "R multiple" and not np.isnan(pos.init_risk):
            need = pos.init_risk * be["value"]
        else:
            need = be["value"]
        if gain >= need:
            cost = pos.entry_price + pos.side * be.get("offset", 0.0)
            pos.sl = cost if np.isnan(pos.sl) else (
                max(pos.sl, cost) if pos.side > 0 else min(pos.sl, cost))
            pos.be_done = pos.trail_armed = True

    if cfg["mode"] == "Strategy signal only (no price stop)":
        return
    if be.get("on") and be.get("trail_only_after") and not pos.trail_armed:
        return
    if cfg["mode"] in ("Fixed points", "Fixed percent", "Derived from reward (risk:reward)"):
        return                                            # anchored to entry, nothing to trail
    if cfg["mode"] == "ATR multiple" and not cfg.get("atr_trail", True):
        return
    cand = _raw_stop(pos, ctx, cfg, ref)
    if np.isnan(cand):
        return
    pos.sl = cand if np.isnan(pos.sl) else (
        max(pos.sl, cand) if pos.side > 0 else min(pos.sl, cand))


def _raw_target(pos: Position, ctx: pd.Series, cfg: dict, ref: float) -> float:
    m, s = cfg["mode"], pos.side
    if m == "Fixed points":
        return pos.entry_price + s * cfg["points"]
    if m == "Fixed percent":
        return pos.entry_price * (1 + s * cfg["pct"] / 100)
    if m == "Trailing target (display only)":
        return ref + s * cfg.get("points", 20.0)
    if m == "Trailing previous swing high / low":
        return float(ctx["piv_hi"] if s > 0 else ctx["piv_lo"])
    if m == "Trailing current swing high / low":
        return float(ctx["run_hi"] if s > 0 else ctx["run_lo"])
    if m == "Trailing previous candle high / low":
        return float(ctx["prev_hi"] if s > 0 else ctx["prev_lo"])
    if m == "Trailing current candle high / low":
        return float(ctx["cur_hi"] if s > 0 else ctx["cur_lo"])
    if m == "Risk:reward multiple":
        return np.nan if np.isnan(pos.init_risk) else pos.entry_price + s * cfg["rr"] * pos.init_risk
    if m == "ATR multiple":
        return pos.entry_price + s * cfg["atr_mult"] * float(ctx["atr"])
    return np.nan


def init_target(pos: Position, ctx: pd.Series, cfg: dict) -> None:
    pos.target = _raw_target(pos, ctx, cfg, pos.entry_price)


def update_target(pos: Position, ctx: pd.Series, cfg: dict) -> None:
    if cfg["mode"] in ("Fixed points", "Fixed percent", "Risk:reward multiple", "ATR multiple",
                       "Strategy reversal exit (no fixed target)"):
        return
    ref = pos.peak if pos.side > 0 else pos.trough
    cand = _raw_target(pos, ctx, cfg, ref)
    if np.isnan(cand):
        return
    pos.target = cand if np.isnan(pos.target) else (
        max(pos.target, cand) if pos.side > 0 else min(pos.target, cand))


def target_is_live(cfg: dict) -> bool:
    """A display-only target is drawn but never triggers an exit."""
    return cfg["mode"] not in ("Trailing target (display only)",
                               "Strategy reversal exit (no fixed target)")


# =============================================================================
# 8 · BACKTEST ENGINE
# =============================================================================
# Timing rules, deliberately pessimistic:
#   * A signal printed on bar N is filled at the OPEN of bar N+1.
#   * Inside a bar the stop is tested before the target. For a long that means
#     Low against the stop first, then High against the target; for a short,
#     High then Low. When both sit inside one bar, the stop wins.
#   * Trailing levels update only on a completed bar's close.
#   * The initial stop and target come from the signal bar's context, the only
#     information available when the order is sent.


@dataclass
class Engine:
    df: pd.DataFrame
    signal: pd.Series
    allow_long: pd.Series
    allow_short: pd.Series
    sl_cfg: dict
    tgt_cfg: dict
    qty: int = 1
    side_mode: str = "Both"
    flip: bool = False
    is_options: bool = False
    allow_reverse: bool = False
    square_off: str | None = None
    costs: CostModel = field(default_factory=CostModel)

    def directional_signal(self) -> pd.Series:
        """Apply the flip first, then the side filter, then the entry filters,
        so 'Long only' always refers to the final direction traded."""
        s = self.signal.astype(int).copy()
        if self.flip:
            s = -s
        if self.side_mode == "Long only":
            s = s.clip(lower=0)
        elif self.side_mode == "Short only":
            s = s.clip(upper=0)
        s = s.where(np.where(s > 0, self.allow_long, True), 0)
        s = s.where(np.where(s < 0, self.allow_short, True), 0)
        return s.fillna(0).astype(int)

    def leg(self, side: int) -> str:
        if not self.is_options:
            return "LONG" if side > 0 else "SHORT"
        return "CE (buy call)" if side > 0 else "PE (buy put)"

    def run(self) -> tuple[pd.DataFrame, pd.Series]:
        df = self.df
        ctx = build_context(df, self.sl_cfg, self.tgt_cfg)
        sig = self.directional_signal()
        o, h, l, c = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
        idx = df.index
        sq = pd.to_datetime(self.square_off).time() if self.square_off else None
        times = pd.Series(idx.time, index=idx).to_numpy() if sq else None
        live_tgt = target_is_live(self.tgt_cfg)
        strat_stop = self.sl_cfg["mode"] == "Strategy signal only (no price stop)"
        strat_tgt = self.tgt_cfg["mode"] == "Strategy reversal exit (no fixed target)"

        pos: Position | None = None
        trades: list[dict] = []
        equity: list[float] = []
        cum = 0.0
        pending: tuple[int, int] | None = None      # (side, signal bar index)

        for i in range(len(df)):
            # 1 · fill anything queued at the previous bar's close
            if pending is not None:
                side, sbar = pending
                pending = None
                if pos is None:
                    fill = self.costs.fill(o[i], side, entering=True)
                    pos = Position(side=side, qty=self.qty, entry_price=fill, entry_time=idx[i],
                                   entry_reason=f"{self.leg(side)} signal on {idx[sbar]:%d-%b %H:%M}",
                                   entry_bar=i, peak=fill, trough=fill)
                    init_stop(pos, ctx.iloc[sbar], self.sl_cfg)
                    init_target(pos, ctx.iloc[sbar], self.tgt_cfg)
                    pos.meta.update(sl_at_entry=pos.sl, target_at_entry=pos.target, mae=0.0, mfe=0.0)

            # 2 · intrabar exits, stop before target
            if pos is not None:
                adverse = (l[i] - pos.entry_price) if pos.side > 0 else (pos.entry_price - h[i])
                favour = (h[i] - pos.entry_price) if pos.side > 0 else (pos.entry_price - l[i])
                pos.meta["mae"] = min(pos.meta["mae"], adverse)
                pos.meta["mfe"] = max(pos.meta["mfe"], favour)

                px = reason = None
                if not np.isnan(pos.sl):
                    if (l[i] <= pos.sl) if pos.side > 0 else (h[i] >= pos.sl):
                        px, reason = pos.sl, "Stop loss"
                if px is None and live_tgt and not np.isnan(pos.target):
                    if (h[i] >= pos.target) if pos.side > 0 else (l[i] <= pos.target):
                        px, reason = pos.target, "Target"
                if px is None and sq is not None and times[i] >= sq:
                    px, reason = c[i], "Session square-off"
                if px is not None:
                    trades.append(self._close(pos, px, idx[i], reason, i, o, h, l, c))
                    cum += trades[-1]["Net P&L"]
                    pos = None

            # 3 · strategy-driven exit or reversal, queued for the next open
            s = int(sig.iloc[i])
            if pos is not None and s != 0 and s != pos.side and (strat_stop or strat_tgt):
                if i + 1 < len(df):
                    px = self.costs.fill(o[i + 1], pos.side, entering=False)
                    trades.append(self._close(pos, px, idx[i + 1], "Strategy reversal", i + 1, o, h, l, c))
                    cum += trades[-1]["Net P&L"]
                    pos = None
                    if self.allow_reverse:
                        pending = (s, i)

            # 4 · trail on the completed bar
            if pos is not None:
                update_stop(pos, ctx.iloc[i], self.sl_cfg, h[i], l[i])
                update_target(pos, ctx.iloc[i], self.tgt_cfg)

            # 5 · queue a fresh entry
            if pos is None and pending is None and s != 0 and i + 1 < len(df):
                pending = (s, i)

            equity.append(cum)

        if pos is not None:
            trades.append(self._close(pos, c[-1], idx[-1], "Open at end of data", len(df) - 1, o, h, l, c))

        tdf = pd.DataFrame(trades, columns=TRADE_COLS) if trades else pd.DataFrame(columns=TRADE_COLS)
        return tdf, pd.Series(equity, index=idx, name="Equity")

    def _close(self, pos: Position, px: float, when, reason: str, i: int, o, h, l, c) -> dict:
        if reason in ("Session square-off", "Open at end of data"):
            px = self.costs.fill(px, pos.side, entering=False)
        gross = (px - pos.entry_price) * pos.side * pos.qty
        ch = self.costs.charges(pos.entry_price, px, pos.qty, pos.side)["total"]
        r = gross / (pos.init_risk * pos.qty) if pos.init_risk and not np.isnan(pos.init_risk) else np.nan
        return {
            "Entry time": pos.entry_time, "Exit time": when, "Direction": self.leg(pos.side),
            "Qty": pos.qty, "Entry price": round(pos.entry_price, 4), "Exit price": round(px, 4),
            "Stop at entry": _round(pos.meta.get("sl_at_entry")), "Stop at exit": _round(pos.sl),
            "Target at exit": _round(pos.target), "Bar open": round(o[i], 4),
            "Bar high": round(h[i], 4), "Bar low": round(l[i], 4), "Bar close": round(c[i], 4),
            "Gross P&L": round(gross, 2), "Charges": round(ch, 2), "Net P&L": round(gross - ch, 2),
            "R multiple": round(r, 2) if r == r else np.nan,
            "MAE": round(pos.meta.get("mae", 0.0), 2), "MFE": round(pos.meta.get("mfe", 0.0), 2),
            "Bars held": i - pos.entry_bar, "Entry reason": pos.entry_reason, "Exit reason": reason,
        }


TRADE_COLS = ["Entry time", "Exit time", "Direction", "Qty", "Entry price", "Exit price",
              "Stop at entry", "Stop at exit", "Target at exit", "Bar open", "Bar high",
              "Bar low", "Bar close", "Gross P&L", "Charges", "Net P&L", "R multiple",
              "MAE", "MFE", "Bars held", "Entry reason", "Exit reason"]


def _round(v):
    return np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


def summarise(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if trades.empty:
        return {"Trades": 0}
    net = trades["Net P&L"]
    wins, losses = net[net > 0], net[net <= 0]
    dd = equity - equity.cummax()
    gp, gl = wins.sum(), -losses.sum()
    bw, bl = _streaks(net)
    return {
        "Trades": len(net), "Wins": len(wins), "Losses": len(losses),
        "Hit rate %": round(100 * len(wins) / len(net), 2),
        "Net P&L": round(net.sum(), 2), "Gross P&L": round(trades["Gross P&L"].sum(), 2),
        "Total charges": round(trades["Charges"].sum(), 2),
        "Average win": round(wins.mean(), 2) if len(wins) else 0.0,
        "Average loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "Profit factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "Expectancy per trade": round(net.mean(), 2),
        "Best trade": round(net.max(), 2), "Worst trade": round(net.min(), 2),
        "Max drawdown": round(dd.min(), 2),
        "Longest win streak": bw, "Longest loss streak": bl,
        "Average R": round(trades["R multiple"].mean(), 2) if trades["R multiple"].notna().any() else np.nan,
        "Average bars held": round(trades["Bars held"].mean(), 1),
        "Long trades": int(trades["Direction"].str.contains("LONG|CE").sum()),
        "Short trades": int(trades["Direction"].str.contains("SHORT|PE").sum()),
    }


def _streaks(net: pd.Series) -> tuple[int, int]:
    bw = bl = cw = cl = 0
    for v in net:
        if v > 0:
            cw, cl = cw + 1, 0
        else:
            cl, cw = cl + 1, 0
        bw, bl = max(bw, cw), max(bl, cl)
    return bw, bl


def insights(trades: pd.DataFrame, summary: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """A plain-language read of the result, plus what to change next."""
    if trades.empty:
        return (["The strategy produced no filled trades over this window."],
                ["Loosen the entry filters, widen the date range, or drop to a lower timeframe "
                 "so the signal has room to fire.",
                 "If you are using the threshold strategy, check the trigger level actually sits "
                 "inside the price range of the loaded window."])

    n, hit, pf = summary["Trades"], summary["Hit rate %"], summary["Profit factor"]
    exp, dd = summary["Expectancy per trade"], summary["Max drawdown"]
    reasons = trades["Exit reason"].value_counts(normalize=True).mul(100).round(1)
    read, rec = [], []

    read.append(f"{n} trades filled, {summary['Wins']} winners and {summary['Losses']} losers — "
                f"a {hit}% hit rate with an expectancy of {exp:,.2f} per trade.")
    read.append(f"Profit factor is {pf}: {pf:.2f} of gross profit for every unit of gross loss."
                if pf != float("inf") else "No losing trades in this window.")
    if summary["Total charges"] > 0:
        share = 100 * summary["Total charges"] / max(abs(summary["Gross P&L"]), 1e-9)
        read.append(f"Costs consumed {summary['Total charges']:,.2f}, {share:.1f}% of the gross "
                    f"result — {'the edge does not survive friction' if share > 60 else 'friction is manageable'}.")
    read.append("Exits split as " + ", ".join(f"{k.lower()} {v}%" for k, v in reasons.items()) + ".")
    read.append(f"Peak-to-trough drawdown reached {dd:,.2f}, against a net result of "
                f"{summary['Net P&L']:,.2f}.")
    if trades["MFE"].notna().any() and summary["Wins"]:
        cap = trades.loc[trades["Net P&L"] > 0, "Net P&L"].mean() / max(trades["MFE"].mean(), 1e-9)
        read.append(f"The average winner captured roughly {cap:.0%} of the best excursion available "
                    f"while the trade was open.")

    if n < 30:
        rec.append(f"Only {n} trades. Anything you conclude here is noise — extend the period or "
                   f"drop to a faster timeframe until you have 100+ before trusting the numbers.")
    if pf != float("inf") and pf < 1.0:
        rec.append("Profit factor is below 1. Losses outweigh wins — invert the logic, widen the "
                   "target, or tighten the entry filter rather than tuning the stop.")
    if hit > 60 and exp < 0:
        rec.append("You win often but lose more per loss than you make per win. The stop is too wide "
                   "relative to the target — try a risk:reward target of 1.5R or better.")
    if hit < 35 and pf != float("inf") and pf > 1.2:
        rec.append("A low hit rate with a healthy profit factor is a trend-following signature. Let "
                   "winners run: switch the target to a trailing swing or ATR multiple.")
    if "Stop loss" in reasons and reasons["Stop loss"] > 65:
        rec.append("Two-thirds of exits are stops. Test a wider ATR-based stop, or add an ADX or "
                   "regime filter so you only trade when the market is actually moving.")
    if "Session square-off" in reasons and reasons["Session square-off"] > 30:
        rec.append("Many trades end at the square-off bell rather than on their own terms. The "
                   "holding period is longer than the session allows — use a faster timeframe or a "
                   "tighter target.")
    if abs(dd) > abs(summary["Net P&L"]) and summary["Net P&L"] > 0:
        rec.append("Drawdown exceeds the net gain. Position sizing, not the signal, is the binding "
                   "constraint here.")
    if summary["Long trades"] == 0 or summary["Short trades"] == 0:
        rec.append("Only one side traded. Confirm that matches your intent — the side selector or a "
                   "directional filter may be suppressing the other half.")
    rec.append("Re-run with forward testing switched on. A result that holds on unseen bars is worth "
               "far more than a result you tuned into existence.")
    return read, rec


# =============================================================================
# 9 · DHAN BROKER ADAPTER (DhanHQ v2)
# =============================================================================
# Nothing fires unless the sidebar switch is on and credentials are present.
# Every call returns a plain dict so the UI can show the raw broker reply.


def dhan_connect(client_id: str, access_token: str):
    from dhanhq import DhanContext, dhanhq

    return dhanhq(DhanContext(client_id.strip(), access_token.strip()))


@functools.lru_cache(maxsize=1)
def dhan_scrip_master() -> pd.DataFrame:
    df = pd.read_csv(DHAN_SCRIP_MASTER, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def dhan_find_security_id(symbol: str, segment_key: str, limit: int = 25) -> pd.DataFrame:
    """Search Dhan's public instrument list and return candidates to pick from."""
    seg = DHAN_SEGMENTS[segment_key][0]
    df = dhan_scrip_master()
    sym_col = next((c for c in ("SEM_TRADING_SYMBOL", "SEM_CUSTOM_SYMBOL", "SYMBOL_NAME")
                    if c in df.columns), df.columns[0])
    id_col = next((c for c in ("SEM_SMST_SECURITY_ID", "SECURITY_ID") if c in df.columns), None)
    seg_col = next((c for c in ("SEM_EXM_EXCH_ID", "EXCH_ID") if c in df.columns), None)
    q = symbol.replace(".NS", "").replace("^", "").strip().upper()
    m = df[df[sym_col].astype(str).str.upper().str.contains(q, na=False, regex=False)]
    if seg_col is not None and not m.empty:
        m = m[m[seg_col].astype(str).str.upper().eq(seg.split("_")[0])]
    keep = [c for c in (sym_col, id_col, seg_col, "SEM_INSTRUMENT_NAME", "SEM_EXPIRY_DATE",
                        "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE", "SEM_LOT_UNITS")
            if c and c in m.columns]
    return m[keep].head(limit).reset_index(drop=True)


def dhan_place(client, *, security_id: str, segment_key: str, side: str, qty: int,
               order_type: str = "MARKET", price: float = 0.0, trigger: float = 0.0,
               tag: str | None = None) -> dict:
    """side is 'BUY' or 'SELL'. Market is the default so an exit is never left
    hanging on an unfilled limit."""
    exch, product = DHAN_SEGMENTS[segment_key]
    kwargs = dict(security_id=str(security_id), exchange_segment=exch,
                  transaction_type=side.upper(), quantity=int(qty),
                  order_type=order_type.upper(), product_type=product,
                  price=float(price) if order_type.upper() in ("LIMIT", "STOP_LOSS") else 0.0,
                  validity="DAY")
    if trigger:
        kwargs["trigger_price"] = float(trigger)
    if tag:
        kwargs["tag"] = tag
    try:
        return {"ok": True, "response": client.place_order(**kwargs), "request": kwargs}
    except Exception as e:                       # surface the broker's own message
        return {"ok": False, "error": str(e), "request": kwargs}


def dhan_positions(client) -> dict:
    try:
        return {"ok": True, "response": client.get_positions()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =============================================================================
# 10 · GMAIL NOTIFICATIONS
# =============================================================================
# Needs a Google App Password, not the account password. Generate one at
# myaccount.google.com under Security, App passwords.


def send_mail(sender: str, app_password: str, to: str, subject: str, body: str) -> tuple[bool, str]:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
            s.login(sender, app_password)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)


def _n(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.2f}"


def entry_mail(cfg: dict, pos: Position, ltp: float) -> tuple[str, str]:
    d_sl = abs(pos.entry_price - pos.sl) if pos.sl == pos.sl else float("nan")
    d_tg = abs(pos.target - pos.entry_price) if pos.target == pos.target else float("nan")
    risk = pos.init_risk * pos.qty if pos.init_risk == pos.init_risk else float("nan")
    subject = f"[{cfg['name']}] {pos.label} entry at {pos.entry_price:,.2f}"
    body = f"""Entry filled.

Instrument      {cfg['name']} ({cfg['symbol']})
Direction       {pos.label}
Quantity        {pos.qty}
Entry price     {_n(pos.entry_price)}
Entry time      {pos.entry_time:%d-%b-%Y %H:%M:%S}
Last traded     {_n(ltp)}

Stop loss       {_n(pos.sl)}   ({_n(d_sl)} away)
Target          {_n(pos.target)}   ({_n(d_tg)} away)
Risk per unit   {_n(pos.init_risk)}
Total risk      {_n(risk)}

Why this fired
{pos.entry_reason}

Setup
Strategy        {cfg['strategy']}
Timeframe       {cfg['interval']} over {cfg['period']}
Stop rule       {cfg['sl_mode']}
Target rule     {cfg['tgt_mode']}
Filters         {', '.join(cfg['filters']) or 'none'} ({cfg['filter_logic']})
Side allowed    {cfg['side_mode']}{'  ·  flipped' if cfg['flip'] else ''}

Sent by Swing Desk. This is a notification, not advice.
"""
    return subject, body


def exit_mail(cfg: dict, t: dict) -> tuple[str, str]:
    subject = f"[{cfg['name']}] {t['Exit reason']} · net {t['Net P&L']:+,.2f}"
    body = f"""Position closed.

Instrument      {cfg['name']} ({cfg['symbol']})
Direction       {t['Direction']}
Quantity        {t['Qty']}

Entry           {_n(t['Entry price'])} at {t['Entry time']}
Exit            {_n(t['Exit price'])} at {t['Exit time']}

Exit reason     {t['Exit reason']}
Stop at exit    {_n(t['Stop at exit'])}
Target at exit  {_n(t['Target at exit'])}

Gross P&L       {t['Gross P&L']:+,.2f}
Charges         {t['Charges']:,.2f}
Net P&L         {t['Net P&L']:+,.2f}
R multiple      {_n(t.get('R multiple'))}
Worst excursion {_n(t.get('MAE'))}
Best excursion  {_n(t.get('MFE'))}

Setup
Strategy        {cfg['strategy']}  ·  {cfg['interval']} / {cfg['period']}
Stop rule       {cfg['sl_mode']}
Target rule     {cfg['tgt_mode']}

Sent by Swing Desk. This is a notification, not advice.
"""
    return subject, body


# =============================================================================
# 11 · CHARTING
# =============================================================================

INK, GRID, BG = "#e8eaf0", "#242938", "#0e1117"
UP, DOWN = "#26a172", "#e05252"
ACCENT = ["#5b8dee", "#f2a541", "#a06cd5", "#3fc1c9", "#e8837b", "#8fd14f"]


def _is_intraday(df: pd.DataFrame) -> bool:
    return len(df) >= 3 and (df.index[-1] - df.index[-2]) < pd.Timedelta("1D")


def build_chart(df: pd.DataFrame, sig, trades: pd.DataFrame | None = None, title: str = "",
                live_lines: dict | None = None, max_bars: int = 1500) -> go.Figure:
    d = df.tail(max_bars)
    has_pane = bool(getattr(sig, "panes", None))
    fig = make_subplots(rows=2 if has_pane else 1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.74, 0.26] if has_pane else [1.0])

    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Price", line_width=1,
                                 increasing_line_color=UP, decreasing_line_color=DOWN,
                                 increasing_fillcolor=UP, decreasing_fillcolor=DOWN), row=1, col=1)

    for i, (name, series) in enumerate((getattr(sig, "overlays", None) or {}).items()):
        fig.add_trace(go.Scatter(x=d.index, y=series.reindex(d.index), name=name, mode="lines",
                                 line=dict(width=1.4, color=ACCENT[i % len(ACCENT)])), row=1, col=1)

    for i, (name, value) in enumerate((getattr(sig, "levels", None) or {}).items()):
        fig.add_hline(y=value, line=dict(color=ACCENT[i % len(ACCENT)], width=1, dash="dot"),
                      annotation_text=f"{name} {value:,.2f}", annotation_position="top left",
                      row=1, col=1)

    for name, spec in (live_lines or {}).items():
        fig.add_hline(y=spec["y"], line=dict(color=spec.get("color", "#888"), width=1.2, dash="dash"),
                      annotation_text=f"{name} {spec['y']:,.2f}",
                      annotation_position="bottom right", row=1, col=1)

    if trades is not None and not trades.empty:
        t = trades[trades["Entry time"].between(d.index[0], d.index[-1])]
        if not t.empty:
            pairs = ((t[t["Direction"].str.contains("LONG|CE")], "triangle-up", UP, "Long entry"),
                     (t[t["Direction"].str.contains("SHORT|PE")], "triangle-down", DOWN, "Short entry"))
            for frame, sym, colour, label in pairs:
                if not frame.empty:
                    fig.add_trace(go.Scatter(x=frame["Entry time"], y=frame["Entry price"],
                                             mode="markers", name=label,
                                             marker=dict(symbol=sym, size=11, color=colour,
                                                         line=dict(width=1, color=INK))), row=1, col=1)
            for frame, colour, label in ((t[t["Net P&L"] > 0], UP, "Exit · profit"),
                                         (t[t["Net P&L"] <= 0], DOWN, "Exit · loss")):
                if not frame.empty:
                    fig.add_trace(go.Scatter(x=frame["Exit time"], y=frame["Exit price"],
                                             mode="markers", name=label,
                                             marker=dict(symbol="x", size=9, color=colour)),
                                  row=1, col=1)

    if has_pane:
        for i, (name, series) in enumerate(sig.panes.items()):
            fig.add_trace(go.Scatter(x=d.index, y=series.reindex(d.index), name=name, mode="lines",
                                     line=dict(width=1.3, color=ACCENT[i % len(ACCENT)])), row=2, col=1)

    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15, color=INK)),
                      template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      height=640 if has_pane else 540, margin=dict(l=8, r=8, t=44, b=8),
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
                      font=dict(color=INK, size=11))
    fig.update_xaxes(gridcolor=GRID, showspikes=True, spikethickness=1,
                     rangebreaks=[dict(bounds=["sat", "mon"])] if _is_intraday(d) else None)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def equity_chart(equity: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, mode="lines", fill="tozeroy",
                               name="Cumulative net P&L", line=dict(color="#5b8dee", width=1.8),
                               fillcolor="rgba(91,141,238,0.12)"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=260,
                      margin=dict(l=8, r=8, t=28, b=8), showlegend=False,
                      title=dict(text="Cumulative net P&L", x=0, font=dict(size=13)),
                      font=dict(color=INK, size=11))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#3a4152")
    return fig


# =============================================================================
# 12 · SIDEBAR
# =============================================================================


def render_sidebar() -> dict:
    s = st.sidebar
    s.markdown("### Setup")

    source = s.selectbox("Data source", ["yfinance", "Dhan"], index=0,
                         help="yfinance is free and delayed. Dhan needs credentials but gives "
                              "finer buckets and a real last-traded price.")

    groups = list(UNIVERSE)
    g = s.selectbox("Asset class", groups, index=groups.index(DEFAULT_GROUP))
    names = list(UNIVERSE[g])
    name = s.selectbox("Instrument", names,
                       index=names.index(DEFAULT_NAME) if DEFAULT_NAME in names else 0)
    symbol = UNIVERSE[g][name]
    custom = s.text_input("Or type a ticker", value="", placeholder="KAYNES.NS")
    if custom.strip():
        symbol = name = custom.strip()

    tfmap = DHAN_TF_PERIODS if source == "Dhan" else TF_PERIODS
    tfs = list(tfmap)
    tf = s.selectbox("Timeframe", tfs, index=tfs.index(DEFAULT_TF) if DEFAULT_TF in tfs else 0)
    opts, default_period = tfmap[tf]
    period = s.selectbox("Period", opts, index=opts.index(default_period),
                         key=f"period_{source}_{tf}",
                         help="The list narrows to what the provider will actually return for "
                              "this timeframe.")

    c1, c2 = s.columns(2)
    qty = c1.number_input("Quantity", 1, 1_000_000, 1)
    side_mode = c2.selectbox("Sides", SIDES, index=0)
    instrument = s.selectbox("Instrument type", INSTRUMENTS, index=0,
                             help="Drives the cost model and how a flipped trade is named.")
    is_options = instrument == "Options"
    flip = s.checkbox("Flip every signal", value=False,
                      help=("A long signal buys a PE and a short signal buys a CE." if is_options
                            else "A long signal goes short and a short signal goes long."))

    s.markdown("---")
    s.markdown("### Strategy")
    snames = list(STRATEGIES)
    strategy = s.selectbox("Signal", snames, index=snames.index(DEFAULT_STRATEGY),
                           label_visibility="collapsed")
    with s.container(border=True):
        st.caption(f"{strategy} settings")
        strat_params = STRATEGIES[strategy]["params"](st, f"sp_{strategy}")

    s.markdown("---")
    s.markdown("### Exits")
    sl_cfg = _stop_panel(s)
    tgt_cfg = _target_panel(s)
    allow_reverse = s.checkbox("Reverse into the opposite trade on a flip signal", value=False)
    sq_on = s.checkbox("Square off intraday positions at a fixed time", value=False)
    square_off = s.text_input("Square-off time (HH:MM)", "15:15") if sq_on else None

    s.markdown("---")
    s.markdown("### Entry filters")
    logic = s.radio("Combine filters with", ["AND", "OR"], horizontal=True, index=0,
                    help="AND: every checked filter must agree. OR: any one is enough.")
    selected: dict[str, dict] = {}
    for fname, spec in FILTERS.items():
        if s.checkbox(fname, key=f"flt_{fname}"):
            with s.container(border=True):
                selected[fname] = spec["params"](st, f"fp_{fname}")

    s.markdown("---")
    s.markdown("### Validation")
    fwd_on = s.checkbox("Forward test on held-out bars", value=False,
                        help="Splits the window so you can see whether the result survives on "
                             "bars you did not tune on.")
    fwd_days = s.number_input("Hold out the last N days", 1, 3650, 30) if fwd_on else 0
    cost_on = s.checkbox("Model real costs", value=False,
                         help="Brokerage, STT, exchange charges, GST, stamp duty and slippage.")
    costs = _cost_panel(s, instrument) if cost_on else CostModel(enabled=False)

    s.markdown("---")
    s.markdown("### Live execution")
    refresh = s.number_input("Refresh every (seconds)", 3, 300, 15,
                             help="Only the live panel redraws. The page does not reload.")
    dhan = _dhan_panel(s, symbol, qty)
    gmail = _gmail_panel(s)

    return dict(source=source, group=g, name=name, symbol=symbol, interval=tf, period=period,
                qty=int(qty), side_mode=side_mode, instrument=instrument, is_options=is_options,
                flip=flip, strategy=strategy, strat_params=strat_params, sl_cfg=sl_cfg,
                tgt_cfg=tgt_cfg, sl_mode=sl_cfg["mode"], tgt_mode=tgt_cfg["mode"],
                allow_reverse=allow_reverse, square_off=square_off, filters=selected,
                filter_logic=logic, forward=dict(on=fwd_on, days=int(fwd_days)), costs=costs,
                refresh=int(refresh), dhan=dhan, gmail=gmail)


def _stop_panel(s) -> dict:
    mode = s.selectbox("Stop loss", SL_MODES, index=0)
    cfg = {"mode": mode}
    with s.container(border=True):
        if mode == "Fixed points":
            cfg["points"] = st.number_input("Stop distance (points)", 0.01, 1e6, 10.0, 0.5, key="sl_pts")
        elif mode == "Fixed percent":
            cfg["pct"] = st.number_input("Stop distance (%)", 0.01, 100.0, 1.0, 0.05, key="sl_pct")
        elif mode == "Trailing points":
            cfg["points"] = st.number_input("Trail by (points)", 0.01, 1e6, 10.0, 0.5, key="sl_tpts")
        elif mode == "ATR multiple":
            c1, c2 = st.columns(2)
            cfg["atr_len"] = c1.number_input("ATR length", 2, 200, 14, key="sl_alen")
            cfg["atr_mult"] = c2.number_input("Multiple", 0.1, 20.0, 2.0, 0.1, key="sl_amul")
            cfg["atr_trail"] = st.checkbox("Trail it from the best price", value=True, key="sl_atr_trail")
        elif mode == "Derived from reward (risk:reward)":
            c1, c2 = st.columns(2)
            cfg["reward_points"] = c1.number_input("Reward (points)", 0.01, 1e6, 20.0, 0.5, key="sl_rwd")
            cfg["rr"] = c2.number_input("Reward : risk", 0.1, 20.0, 2.0, 0.1, key="sl_rr")
        elif mode == "Strategy signal only (no price stop)":
            st.caption("The position stays open until the strategy prints the opposite signal. "
                       "There is no price stop, so size accordingly.")
        else:
            c1, c2 = st.columns(2)
            cfg["swing_n"] = c1.number_input("Swing lookback (bars)", 1, 100, 3, key="sl_sw")
            cfg["buffer"] = c2.number_input("Buffer beyond level", 0.0, 1e5, 0.0, 0.05, key="sl_buf")

        st.markdown("")
        be_on = st.checkbox("Move the stop to cost once the trade runs", value=False, key="sl_be")
        be = {"on": be_on}
        if be_on:
            c1, c2 = st.columns(2)
            be["unit"] = c1.selectbox("Trigger measured in", ["Points", "Percent", "R multiple"], key="be_u")
            be["value"] = c2.number_input("Trigger at", 0.01, 1e6, 10.0, 0.5, key="be_v")
            c3, c4 = st.columns(2)
            be["offset"] = c3.number_input("Lock in beyond cost", 0.0, 1e5, 0.0, 0.05, key="be_o")
            be["trail_only_after"] = c4.checkbox("Start trailing only after this", value=False, key="be_t")
        cfg["be"] = be
    return cfg


def _target_panel(s) -> dict:
    mode = s.selectbox("Target", TGT_MODES, index=0)
    cfg = {"mode": mode}
    with s.container(border=True):
        if mode == "Fixed points":
            cfg["points"] = st.number_input("Target distance (points)", 0.01, 1e6, 20.0, 0.5, key="tg_pts")
        elif mode == "Fixed percent":
            cfg["pct"] = st.number_input("Target distance (%)", 0.01, 500.0, 2.0, 0.05, key="tg_pct")
        elif mode == "Trailing target (display only)":
            cfg["points"] = st.number_input("Project ahead by (points)", 0.01, 1e6, 20.0, 0.5, key="tg_disp")
            st.caption("Drawn on the chart as a projection. It never closes the trade — the stop does.")
        elif mode == "Risk:reward multiple":
            cfg["rr"] = st.number_input("Reward : risk", 0.1, 50.0, 2.0, 0.1, key="tg_rr")
        elif mode == "ATR multiple":
            c1, c2 = st.columns(2)
            cfg["atr_len"] = c1.number_input("ATR length", 2, 200, 14, key="tg_alen")
            cfg["atr_mult"] = c2.number_input("Multiple", 0.1, 50.0, 3.0, 0.1, key="tg_amul")
        elif mode == "Strategy reversal exit (no fixed target)":
            st.caption("The trade runs until the strategy reverses or the stop is hit.")
        else:
            cfg["swing_n"] = st.number_input("Swing lookback (bars)", 1, 100, 3, key="tg_sw")
    return cfg


def _cost_panel(s, instrument: str) -> CostModel:
    d = COST_DEFAULTS.get(instrument, COST_DEFAULTS["Equity intraday"])
    with s.container(border=True):
        c1, c2 = st.columns(2)
        unit = c1.selectbox("Slippage in", ["Points", "Percent"], key="cost_su")
        slip = c2.number_input("Slippage per leg", 0.0, 1e4,
                               0.05 if unit == "Percent" else 1.0, 0.01, key="cost_sv")
        with st.expander("Charge rates (%)"):
            c1, c2 = st.columns(2)
            bp = c1.number_input("Brokerage", 0.0, 5.0, d["brokerage_pct"], 0.001, format="%.4f", key="c_bp")
            bc = c2.number_input("Brokerage cap per leg", 0.0, 1e4, d["brokerage_cap"], 1.0, key="c_bc")
            c1, c2 = st.columns(2)
            ss = c1.number_input("STT on sell", 0.0, 5.0, d["stt_sell"], 0.001, format="%.4f", key="c_ss")
            sb = c2.number_input("STT on buy", 0.0, 5.0, d["stt_buy"], 0.001, format="%.4f", key="c_sb")
            c1, c2 = st.columns(2)
            tx = c1.number_input("Exchange turnover", 0.0, 5.0, d["txn"], 0.0001, format="%.5f", key="c_tx")
            sd = c2.number_input("Stamp duty on buy", 0.0, 5.0, d["stamp_buy"], 0.0001, format="%.4f", key="c_sd")
            c1, c2 = st.columns(2)
            se = c1.number_input("SEBI", 0.0, 1.0, d["sebi"], 0.0001, format="%.5f", key="c_se")
            gs = c2.number_input("GST on fees", 0.0, 50.0, d["gst"], 0.5, key="c_gs")
    return CostModel(instrument=instrument, slippage_unit=unit, slippage=slip, brokerage_pct=bp,
                     brokerage_cap=bc, stt_sell=ss, stt_buy=sb, txn=tx, stamp_buy=sd, sebi=se,
                     gst=gs, enabled=True)


def _dhan_panel(s, symbol: str, qty: int) -> dict:
    on = s.checkbox("Send orders to Dhan", value=False)
    cfg = {"on": on}
    if not on:
        return cfg
    with s.container(border=True):
        st.warning("Live orders spend real money. Test with quantity 1 first.", icon="⚠️")
        cfg["client_id"] = st.text_input("Client ID", type="password", key="dh_cid")
        cfg["token"] = st.text_input("Access token", type="password", key="dh_tok")
        cfg["segment"] = st.selectbox("Product", list(DHAN_SEGMENTS), key="dh_seg")
        cfg["qty"] = st.number_input("Order quantity", 1, 1_000_000, int(qty), key="dh_qty")
        cfg["entry_type"] = st.selectbox("Entry order", ["MARKET", "LIMIT"], index=0, key="dh_eo")
        cfg["exit_type"] = st.selectbox("Exit / square-off order", ["MARKET", "LIMIT"], index=0, key="dh_xo")
        if st.button("Look up from Dhan's instrument list", width="stretch", key="dh_look"):
            try:
                st.session_state["dh_matches"] = dhan_find_security_id(symbol, cfg["segment"])
            except Exception as e:
                st.error(f"Lookup failed: {e}")
        matches = st.session_state.get("dh_matches")
        if matches is not None and not matches.empty:
            st.dataframe(matches, width="stretch", height=170)
        cfg["security_id"] = st.text_input("Security ID to trade", value="", key="dh_secid",
                                           placeholder="e.g. 1333")
    return cfg


def _gmail_panel(s) -> dict:
    on = s.checkbox("Email me on entry and exit", value=False)
    cfg = {"on": on}
    if not on:
        return cfg
    with s.container(border=True):
        cfg["sender"] = st.text_input("Gmail address", key="gm_from")
        cfg["password"] = st.text_input("App password", type="password", key="gm_pw",
                                        help="A 16-character Google App Password, not your login password.")
        cfg["to"] = st.text_input("Send to", key="gm_to")
        c1, c2 = st.columns(2)
        cfg["on_entry"] = c1.checkbox("On entry", value=True, key="gm_e")
        cfg["on_exit"] = c2.checkbox("On exit", value=True, key="gm_x")
    return cfg


# =============================================================================
# 13 · TABS
# =============================================================================


def config_summary(cfg: dict) -> pd.DataFrame:
    be = cfg["sl_cfg"].get("be", {})
    rows = [
        ("Instrument", f"{cfg['name']} · {cfg['symbol']}"),
        ("Source", cfg["source"]),
        ("Timeframe / period", f"{cfg['interval']} over {cfg['period']}"),
        ("Instrument type", cfg["instrument"]),
        ("Quantity", cfg["qty"]),
        ("Sides", cfg["side_mode"] + (" · flipped" if cfg["flip"] else "")),
        ("Strategy", cfg["strategy"]),
        ("Strategy settings", _fmt(cfg["strat_params"])),
        ("Stop rule", cfg["sl_mode"]),
        ("Stop settings", _fmt({k: v for k, v in cfg["sl_cfg"].items() if k not in ("mode", "be")})),
        ("Move stop to cost", _fmt(be) if be.get("on") else "off"),
        ("Target rule", cfg["tgt_mode"]),
        ("Target settings", _fmt({k: v for k, v in cfg["tgt_cfg"].items() if k != "mode"})),
        ("Entry filters", f"{', '.join(cfg['filters']) or 'none'}  ({cfg['filter_logic']})"),
        ("Reverse on flip", "yes" if cfg["allow_reverse"] else "no"),
        ("Session square-off", cfg["square_off"] or "off"),
        ("Cost model", "on" if cfg["costs"].enabled else "off"),
        ("Forward test", f"last {cfg['forward']['days']} days held out" if cfg["forward"]["on"] else "off"),
        ("Dhan orders", "armed" if cfg["dhan"].get("on") else "off"),
        ("Email alerts", "on" if cfg["gmail"].get("on") else "off"),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"]).astype(str)


def _fmt(d: dict) -> str:
    return " · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in d.items()) if d else "—"


def prepare(cfg: dict, df: pd.DataFrame):
    sig = STRATEGIES[cfg["strategy"]]["compute"](df, cfg["strat_params"])
    al, as_ = combine_filters(df, cfg["filters"], cfg["filter_logic"])
    return sig, al, as_


def make_engine(cfg: dict, df: pd.DataFrame, sig, al, as_) -> Engine:
    return Engine(df=df, signal=sig.signal, allow_long=al, allow_short=as_, sl_cfg=cfg["sl_cfg"],
                  tgt_cfg=cfg["tgt_cfg"], qty=cfg["qty"], side_mode=cfg["side_mode"],
                  flip=cfg["flip"], is_options=cfg["is_options"], allow_reverse=cfg["allow_reverse"],
                  square_off=cfg["square_off"], costs=cfg["costs"])


# ------------------------------------------------------------------ backtest
def render_backtest(cfg: dict, df: pd.DataFrame):
    sig, al, as_ = prepare(cfg, df)
    trades, equity = make_engine(cfg, df, sig, al, as_).run()
    summary = summarise(trades, equity)

    st.markdown("#### Result")
    if trades.empty:
        st.info("No trades were filled with this configuration.")
    else:
        k = st.columns(6)
        k[0].metric("Net P&L", f"{summary['Net P&L']:,.2f}")
        k[1].metric("Trades", summary["Trades"])
        k[2].metric("Hit rate", f"{summary['Hit rate %']}%")
        k[3].metric("Profit factor", summary["Profit factor"])
        k[4].metric("Expectancy", f"{summary['Expectancy per trade']:,.2f}")
        k[5].metric("Max drawdown", f"{summary['Max drawdown']:,.2f}")

    st.plotly_chart(build_chart(df, sig, trades,
                                title=f"{cfg['name']} · {cfg['interval']} · {cfg['strategy']}"),
                    width="stretch")
    if not trades.empty:
        st.plotly_chart(equity_chart(equity), width="stretch")

    read, rec = insights(trades, summary, cfg)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What happened")
        for line in read:
            st.markdown(f"- {line}")
    with c2:
        st.markdown("#### What to change")
        for line in rec:
            st.markdown(f"- {line}")

    if cfg["forward"]["on"] and not df.empty:
        _forward_test(cfg, df)

    st.markdown("#### Trades")
    st.dataframe(trades, width="stretch", height=340)
    if not trades.empty:
        st.download_button("Download trades as CSV", trades.to_csv(index=False).encode(),
                           f"{cfg['symbol']}_{cfg['interval']}_trades.csv", "text/csv")

    with st.expander("Full statistics"):
        st.dataframe(pd.DataFrame(summary.items(), columns=["Metric", "Value"]).astype(str),
                     width="stretch", height=560)
    with st.expander("Configuration used for this run"):
        st.dataframe(config_summary(cfg), width="stretch", hide_index=True, height=560)


def _forward_test(cfg: dict, df: pd.DataFrame):
    cutoff = df.index[-1] - pd.Timedelta(days=cfg["forward"]["days"])
    insample, out = df[df.index <= cutoff], df[df.index > cutoff]
    if len(insample) < 30 or len(out) < 5:
        st.warning("Not enough bars either side of the cutoff to split the window meaningfully.")
        return
    st.markdown("#### Forward test")
    st.caption(f"Bars up to {cutoff:%d-%b-%Y %H:%M} are the tuning set. Everything after is held out.")
    cols = st.columns(2)
    for col, part, label in ((cols[0], insample, "Tuning set"), (cols[1], out, "Held out")):
        s2, a2, b2 = prepare(cfg, part)
        t2, e2 = make_engine(cfg, part, s2, a2, b2).run()
        m = summarise(t2, e2)
        with col:
            st.markdown(f"**{label}** · {len(part)} bars")
            if m["Trades"] == 0:
                st.caption("No trades.")
            else:
                st.dataframe(pd.DataFrame(
                    [(k, str(m[k])) for k in ("Trades", "Hit rate %", "Net P&L", "Profit factor",
                                              "Expectancy per trade", "Max drawdown")],
                    columns=["Metric", "Value"]), width="stretch", hide_index=True)
    st.caption("If the held-out column is materially worse, the settings are fitted to the past "
               "rather than describing the market.")


# ---------------------------------------------------------------------- live
def render_live(cfg: dict):
    ss = st.session_state
    ss.setdefault("live_on", False)
    ss.setdefault("live_pos", None)
    ss.setdefault("live_trades", [])
    ss.setdefault("live_log", [])
    ss.setdefault("live_signal_bar", None)
    ss.setdefault("live_manual_exit", False)

    c1, c2, c3, c4 = st.columns([1, 1, 1.3, 3])
    if c1.button("Start trading", type="primary", width="stretch", disabled=ss.live_on):
        ss.live_on = True
        log_live("Trading started.")
    if c2.button("Stop trading", width="stretch", disabled=not ss.live_on):
        ss.live_on = False
        log_live("Trading stopped. Any open position is left as-is.")
    if c3.button("Square off now", width="stretch", disabled=ss.live_pos is None):
        ss.live_manual_exit = True
    c4.markdown(f"**Status** · {'running' if ss.live_on else 'idle'} · "
                f"{'in a position' if ss.live_pos else 'flat'} · refresh {cfg['refresh']}s")

    if cfg["dhan"].get("on"):
        st.warning("Dhan order routing is armed. Fills below go to your real account.", icon="⚠️")

    ss["live_cfg"] = cfg

    @st.fragment(run_every=cfg["refresh"] if ss.live_on else None)
    def panel():
        live_cycle(st.session_state.get("live_cfg", cfg))

    panel()


def log_live(msg: str):
    ss = st.session_state
    ss.live_log = ([f"{pd.Timestamp.now():%H:%M:%S}  {msg}"] + ss.get("live_log", []))[:200]


def live_cycle(cfg: dict):
    """One poll. Signals come from the last CLOSED bar; stops and targets are
    checked against the last traded price, which is what live data gives you."""
    ss = st.session_state
    try:
        df = fetch_yf(cfg["symbol"], cfg["interval"], cfg["period"])
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return
    if len(df) < 5:
        st.warning("Not enough bars to evaluate the strategy yet.")
        return

    closed = df.iloc[:-1]          # the newest row is still forming
    forming = df.iloc[-1]
    sig, al, as_ = prepare(cfg, df)
    ctx = build_context(df, cfg["sl_cfg"], cfg["tgt_cfg"])
    engine = make_engine(cfg, df, sig, al, as_)
    dsig = engine.directional_signal()

    ltp = ltp_yf(cfg["symbol"]) or float(forming["Close"])
    last_ts = closed.index[-1]
    pos: Position | None = ss.live_pos

    if pos is not None:
        update_stop(pos, ctx.loc[last_ts], cfg["sl_cfg"], ltp, ltp)
        update_target(pos, ctx.loc[last_ts], cfg["tgt_cfg"])
        px = reason = None
        if ss.live_manual_exit:
            px, reason = ltp, "Manual square-off"
        elif not np.isnan(pos.sl) and ((ltp <= pos.sl) if pos.side > 0 else (ltp >= pos.sl)):
            px, reason = ltp, "Stop loss"
        elif target_is_live(cfg["tgt_cfg"]) and not np.isnan(pos.target) and \
                ((ltp >= pos.target) if pos.side > 0 else (ltp <= pos.target)):
            px, reason = ltp, "Target"
        elif int(dsig.loc[last_ts]) not in (0, pos.side) and \
                (cfg["sl_mode"].startswith("Strategy") or cfg["tgt_mode"].startswith("Strategy")):
            px, reason = ltp, "Strategy reversal"
        if px is not None:
            close_live(cfg, pos, px, reason, forming)
            pos = ss.live_pos = None
            ss.live_manual_exit = False

    if ss.live_on and pos is None:
        s = int(dsig.loc[last_ts])
        if s != 0 and ss.live_signal_bar != last_ts:
            entry = float(forming["Open"])          # the open of candle N+1
            pos = Position(side=s, qty=cfg["qty"], entry_price=entry, entry_time=forming.name,
                           entry_reason=f"{engine.leg(s)} signal closed on {last_ts:%d-%b %H:%M}, "
                                        f"filled at the next bar's open",
                           entry_bar=len(df) - 1, peak=entry, trough=entry)
            init_stop(pos, ctx.loc[last_ts], cfg["sl_cfg"])
            init_target(pos, ctx.loc[last_ts], cfg["tgt_cfg"])
            pos.meta.update(sl_at_entry=pos.sl, target_at_entry=pos.target, mae=0.0, mfe=0.0)
            ss.live_pos, ss.live_signal_bar = pos, last_ts
            log_live(f"{pos.label} entry at {entry:,.2f}, stop {pos.sl:,.2f}, target {pos.target:,.2f}")
            route_order(cfg, "BUY" if s > 0 else "SELL", entry, "entry")
            if cfg["gmail"].get("on") and cfg["gmail"].get("on_entry"):
                ok, err = send_mail(cfg["gmail"]["sender"], cfg["gmail"]["password"],
                                    cfg["gmail"]["to"], *entry_mail(cfg, pos, ltp))
                log_live("Entry email sent." if ok else f"Entry email failed: {err}")

    live_metrics(cfg, pos, ltp, df, dsig, al, as_, sig, last_ts)


def live_metrics(cfg, pos, ltp, df, dsig, al, as_, sig, last_ts):
    ss = st.session_state
    st.markdown("#### Live")
    m = st.columns(6)
    m[0].metric("Last traded", f"{ltp:,.2f}")
    if pos is not None:
        pnl = (ltp - pos.entry_price) * pos.side * pos.qty
        m[1].metric("Entry", f"{pos.entry_price:,.2f}", pos.label)
        m[2].metric("Stop", "—" if np.isnan(pos.sl) else f"{pos.sl:,.2f}",
                    None if np.isnan(pos.sl) else f"{abs(ltp - pos.sl):,.2f} away")
        m[3].metric("Target", "—" if np.isnan(pos.target) else f"{pos.target:,.2f}",
                    None if np.isnan(pos.target) else f"{abs(pos.target - ltp):,.2f} away")
        m[4].metric("Open P&L", f"{pnl:+,.2f}")
        m[5].metric("Risk on entry", "—" if np.isnan(pos.init_risk)
                    else f"{pos.init_risk * pos.qty:,.2f}")
    else:
        for i, lbl in enumerate(("Entry", "Stop", "Target", "Open P&L", "Risk"), start=1):
            m[i].metric(lbl, "—")

    lines = {}
    if pos is not None:
        lines["Entry"] = {"y": pos.entry_price, "color": "#c8cbd4"}
        if not np.isnan(pos.sl):
            lines["Stop"] = {"y": pos.sl, "color": DOWN}
        if not np.isnan(pos.target):
            lines["Target"] = {"y": pos.target, "color": UP}
    st.plotly_chart(build_chart(df, sig, title=f"{cfg['name']} · {cfg['interval']}",
                                live_lines=lines, max_bars=400),
                    width="stretch", key=f"livechart_{pd.Timestamp.now().value}")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("##### Strategy readout")
        st.dataframe(live_readout(cfg, df, dsig, al, as_, sig, last_ts, pos),
                     width="stretch", hide_index=True, height=280)
    with c2:
        st.markdown("##### Activity")
        st.code("\n".join(ss.live_log[:12]) or "Nothing yet.", language=None)

    st.markdown("##### Trades this session")
    st.dataframe(pd.DataFrame(ss.live_trades, columns=TRADE_COLS), width="stretch", height=220)
    with st.expander("Configuration in force"):
        st.dataframe(config_summary(cfg), width="stretch", hide_index=True, height=560)


def live_readout(cfg, df, dsig, al, as_, sig, ts, pos) -> pd.DataFrame:
    rows = [("Last closed bar", f"{ts:%d-%b %H:%M}"),
            ("Signal on that bar", {1: "LONG", -1: "SHORT", 0: "none"}[int(dsig.loc[ts])])]
    for group in (getattr(sig, "overlays", {}) or {}, getattr(sig, "panes", {}) or {}):
        for name, series in group.items():
            upto = series.loc[:ts]
            v = upto.iloc[-1] if not upto.empty else np.nan
            rows.append((name, "—" if pd.isna(v) else f"{v:,.2f}"))
    for name, level in (getattr(sig, "levels", {}) or {}).items():
        rows.append((name, f"{level:,.2f}  ({float(df['Close'].iloc[-1]) - level:+,.2f} from price)"))
    for fname, params in cfg["filters"].items():
        l, s = FILTERS[fname]["apply"](df, params)
        rows.append((f"Filter · {fname}",
                     f"long {'pass' if bool(l.loc[ts]) else 'block'} · "
                     f"short {'pass' if bool(s.loc[ts]) else 'block'}"))
    if pos is None:
        rows.append(("Waiting for", waiting_for(cfg, sig, dsig, al, as_, ts)))
    return pd.DataFrame(rows, columns=["Item", "Value"]).astype(str)


def waiting_for(cfg, sig, dsig, al, as_, ts) -> str:
    if int(dsig.loc[ts]) != 0:
        return "A fill on the next bar's open."
    raw = int(sig.signal.loc[ts]) if ts in sig.signal.index else 0
    if raw != 0:
        side = "long" if raw > 0 else "short"
        passes = bool(al.loc[ts]) if raw > 0 else bool(as_.loc[ts])
        if not passes:
            return f"The strategy wants to go {side}, but the entry filters are blocking it."
        return f"A {side} signal is present but the side selector excludes it."
    return ("The strategy has not printed a signal. It fires when its own condition flips — the "
            "rows above show how close each input currently is.")


def close_live(cfg, pos: Position, px: float, reason: str, bar):
    ss = st.session_state
    costs = cfg["costs"]
    fill = costs.fill(px, pos.side, entering=False)
    gross = (fill - pos.entry_price) * pos.side * pos.qty
    ch = costs.charges(pos.entry_price, fill, pos.qty, pos.side)["total"]
    r = gross / (pos.init_risk * pos.qty) if pos.init_risk and not np.isnan(pos.init_risk) else np.nan
    tz = getattr(bar.name, "tz", None)
    trade = {
        "Entry time": pos.entry_time, "Exit time": pd.Timestamp.now(tz=tz),
        "Direction": pos.label, "Qty": pos.qty, "Entry price": round(pos.entry_price, 4),
        "Exit price": round(fill, 4), "Stop at entry": _round(pos.meta.get("sl_at_entry")),
        "Stop at exit": _round(pos.sl), "Target at exit": _round(pos.target),
        "Bar open": round(float(bar["Open"]), 4), "Bar high": round(float(bar["High"]), 4),
        "Bar low": round(float(bar["Low"]), 4), "Bar close": round(float(bar["Close"]), 4),
        "Gross P&L": round(gross, 2), "Charges": round(ch, 2), "Net P&L": round(gross - ch, 2),
        "R multiple": round(r, 2) if r == r else np.nan,
        "MAE": round(pos.meta.get("mae", 0.0), 2), "MFE": round(pos.meta.get("mfe", 0.0), 2),
        "Bars held": np.nan, "Entry reason": pos.entry_reason, "Exit reason": reason,
    }
    ss.live_trades = [trade] + ss.live_trades
    log_live(f"{reason} at {fill:,.2f} · net {trade['Net P&L']:+,.2f}")
    route_order(cfg, "SELL" if pos.side > 0 else "BUY", fill, "exit")
    if cfg["gmail"].get("on") and cfg["gmail"].get("on_exit"):
        ok, err = send_mail(cfg["gmail"]["sender"], cfg["gmail"]["password"], cfg["gmail"]["to"],
                            *exit_mail(cfg, trade))
        log_live("Exit email sent." if ok else f"Exit email failed: {err}")


def route_order(cfg, side: str, price: float, leg: str):
    d = cfg.get("dhan", {})
    if not d.get("on") or not d.get("security_id") or not d.get("client_id"):
        return
    try:
        client = dhan_connect(d["client_id"], d["token"])
        res = dhan_place(client, security_id=d["security_id"], segment_key=d["segment"], side=side,
                         qty=int(d.get("qty", cfg["qty"])),
                         order_type=d["entry_type"] if leg == "entry" else d["exit_type"],
                         price=price, tag="swingdesk")
        log_live(f"Dhan {leg} {side}: {'accepted' if res['ok'] else 'rejected'} · "
                 f"{res.get('response') or res.get('error')}")
    except Exception as e:
        log_live(f"Dhan {leg} order failed: {e}")


# ------------------------------------------------------------------- history
def render_history():
    ss = st.session_state
    trades = pd.DataFrame(ss.get("live_trades", []), columns=TRADE_COLS)
    c1, c2 = st.columns([1, 4])
    if c1.button("Clear history", width="stretch", disabled=trades.empty):
        ss.live_trades = []
        st.rerun()
    if trades.empty:
        c2.info("Closed trades appear here the moment a position exits.")
        return
    net = trades["Net P&L"]
    k = st.columns(5)
    k[0].metric("Trades", len(net))
    k[1].metric("Net P&L", f"{net.sum():,.2f}")
    k[2].metric("Wins", int((net > 0).sum()))
    k[3].metric("Losses", int((net <= 0).sum()))
    k[4].metric("Hit rate", f"{100 * (net > 0).mean():.1f}%")
    st.dataframe(trades, width="stretch", height=460)
    st.download_button("Download history as CSV", trades.to_csv(index=False).encode(),
                       "live_trade_history.csv", "text/csv")


# =============================================================================
# 14 · MAIN
# =============================================================================

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
  [data-testid="stMetricValue"] {font-size: 1.35rem; font-variant-numeric: tabular-nums;}
  [data-testid="stMetricLabel"] {font-size: .72rem; letter-spacing: .06em;
                                 text-transform: uppercase; opacity: .62;}
  [data-testid="stMetricDelta"] {font-size: .74rem;}
  section[data-testid="stSidebar"] {width: 372px !important;}
  section[data-testid="stSidebar"] h3 {font-size: .78rem; letter-spacing: .12em;
                                       text-transform: uppercase; opacity: .55;
                                       margin-bottom: .2rem;}
  .stTabs [data-baseweb="tab"] {font-size: .9rem; letter-spacing: .02em;}
  code {font-size: .78rem;}
  .rule {height: 1px; background: linear-gradient(90deg, #3a4152, transparent);
         margin: .2rem 0 1.1rem;}
</style>
"""


@st.cache_data(ttl=45, show_spinner="Loading candles…")
def load_candles(symbol: str, interval: str, period: str) -> pd.DataFrame:
    return fetch_yf(symbol, interval, period)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="◧", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"## {APP_TITLE}")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    cfg = render_sidebar()
    tab_bt, tab_live, tab_hist = st.tabs(["Backtest", "Live trading", "Trade history"])

    with tab_bt:
        try:
            df = load_candles(cfg["symbol"], cfg["interval"], cfg["period"])
        except Exception as e:
            st.error(f"Could not load {cfg['symbol']} at {cfg['interval']}/{cfg['period']}. {e}")
            st.caption("Check the ticker spelling, or pick a shorter period — providers cap "
                       "intraday history at 7 days for 1-minute bars and 60 days for other "
                       "intraday buckets.")
            st.stop()
        st.caption(f"{len(df):,} bars · {df.index[0]:%d-%b-%Y %H:%M} to {df.index[-1]:%d-%b-%Y %H:%M}")
        render_backtest(cfg, df)

    with tab_live:
        render_live(cfg)

    with tab_hist:
        render_history()

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.caption("Backtested results describe the past under assumptions you chose. They are not a "
               "forecast, and this tool is not financial advice.")


# Streamlit executes this script as __main__, so this both runs the app and keeps
# the file importable for testing.
if __name__ == "__main__":
    main()
