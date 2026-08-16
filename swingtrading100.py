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

# Index constituents for the scanner. These are a snapshot: index membership is
# reviewed twice a year, so treat them as a starting point and paste your own
# list in the scanner when accuracy matters.
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BEL", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

NIFTY_NEXT_50 = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "BAJAJHLDNG", "BANKBARODA",
    "BERGEPAINT", "BOSCHLTD", "CANBK", "CGPOWER", "CHOLAFIN", "COLPAL", "DABUR", "DIVISLAB",
    "DLF", "DMART", "GAIL", "GODREJCP", "HAVELLS", "HAL", "ICICIGI", "ICICIPRULI", "INDHOTEL",
    "INDIGO", "IOC", "IRFC", "JINDALSTEL", "JSWENERGY", "LICI", "LODHA", "MARICO", "MOTHERSON",
    "NAUKRI", "PFC", "PIDILITIND", "PNB", "POLYCAB", "RECLTD", "SIEMENS", "SHREECEM", "SRF",
    "TATAPOWER", "TORNTPHARM", "TVSMOTOR", "UNITDSPR", "VBL", "VEDL", "ZOMATO", "ZYDUSLIFE",
]

# A liquid mid- and small-cap extension. Not the full Nifty 500 — that list runs
# to 500 names and changes constantly, so paste it in the scanner if you need it.
LIQUID_EXTRA = [
    "AARTIIND", "ABCAPITAL", "ABFRL", "ALKEM", "APLAPOLLO", "ASHOKLEY", "ASTRAL", "AUBANK",
    "AUROPHARMA", "BALKRISIND", "BANDHANBNK", "BHARATFORG", "BHEL", "BIOCON", "CAMS", "COFORGE",
    "CONCOR", "CROMPTON", "CUMMINSIND", "DALBHARAT", "DEEPAKNTR", "DIXON", "ESCORTS", "EXIDEIND",
    "FEDERALBNK", "FORTIS", "GLENMARK", "GMRAIRPORT", "GODREJPROP", "GUJGASLTD", "HINDPETRO",
    "IDFCFIRSTB", "INDIANB", "INDUSTOWER", "IPCALAB", "IRCTC", "JUBLFOOD", "KALYANKJIL",
    "KAYNES", "KPITTECH", "LAURUSLABS", "LICHSGFIN", "LTF", "LUPIN", "MANAPPURAM", "MAXHEALTH",
    "MFSL", "MGL", "MPHASIS", "MRF", "MUTHOOTFIN", "NHPC", "NMDC", "NYKAA", "OBEROIRLTY",
    "OFSS", "OIL", "PAGEIND", "PATANJALI", "PAYTM", "PERSISTENT", "PETRONET", "PIIND",
    "POLICYBZR", "POONAWALLA", "PRESTIGE", "RAMCOCEM", "RBLBANK", "SAIL", "SJVN", "SONACOMS",
    "STARHEALTH", "SUNTV", "SUPREMEIND", "SYNGENE", "TATACHEM", "TATACOMM", "TATAELXSI",
    "TIINDIA", "TORNTPOWER", "TRIDENT", "UBL", "UNIONBANK", "UPL", "VOLTAS", "YESBANK",
]

SCAN_UNIVERSES = {
    "Nifty 50": NIFTY_50,
    "Nifty 100": NIFTY_50 + NIFTY_NEXT_50,
    "Nifty 500 (liquid subset)": NIFTY_50 + NIFTY_NEXT_50 + LIQUID_EXTRA,
    "Indices and benchmarks": ["^NSEI", "^NSEBANK", "^BSESN", "NIFTY_FIN_SERVICE.NS"],
    "Custom list": [],
}

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
    out[seed] = np.nanmean(x[start:seed + 1])
    for i in range(seed + 1, x.size):
        if np.isnan(x[i]):
            # One missing print from the data vendor would otherwise multiply
            # into NaN and destroy every value after it. TradingView has no
            # missing bars, so holding the previous value is the faithful
            # equivalent rather than a silent divergence.
            out[i] = out[i - 1]
        else:
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
    left_max = hi.shift(1).rolling(left, min_periods=left).max()
    right_max = hi.shift(-right).rolling(right, min_periods=right).max()
    left_min = lo.shift(1).rolling(left, min_periods=left).min()
    right_min = lo.shift(-right).rolling(right, min_periods=right).min()
    is_low = ((lo < left_min) & (lo <= right_min)).fillna(False)
    is_high = ((hi > left_max) & (hi >= right_max)).fillna(False)
    return lo.where(is_low).shift(right).ffill(), hi.where(is_high).shift(right).ffill()


BAR_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "25m": 25, "30m": 30, "60m": 60,
               "1h": 60, "1d": 375, "1wk": 1875}


def bar_duration(interval: str) -> pd.Timedelta:
    """How long one bar lasts. Daily and weekly use the NSE session length so a
    finished daily candle is not mistaken for one still forming."""
    return pd.Timedelta(minutes=BAR_MINUTES.get(interval, 5))


def warmup_report(sig, ctx: pd.DataFrame | None = None) -> list[str]:
    """Names any indicator still undefined on the most recent bar.

    Every recursive average is undefined until it has `length` bars, exactly as
    on a TradingView chart. If one is still empty at the end of the window the
    cause is a lookback longer than the data, not a calculation fault, and the
    strategy cannot signal until it fills.
    """
    missing = []
    for group in (getattr(sig, "overlays", {}) or {}, getattr(sig, "panes", {}) or {}):
        for name, series in group.items():
            if len(series) == 0 or pd.isna(series.iloc[-1]):
                first = series.first_valid_index()
                missing.append(f"{name} (no value at the latest bar"
                               + (f", first defined {first:%d-%b %H:%M})" if first is not None
                                  else ", never defined in this window)"))
    if ctx is not None and "atr" in ctx and len(ctx) and pd.isna(ctx["atr"].iloc[-1]):
        missing.append("ATR used by the stop or target")
    return missing


def running_extreme(df: pd.DataFrame, length: int):
    """The unconfirmed 'current' swing: the rolling extreme of recent bars."""
    return (df["Low"].rolling(length, min_periods=1).min(),
            df["High"].rolling(length, min_periods=1).max())


# =============================================================================
# 3 · DATA ACCESS
# =============================================================================

_rate_lock = threading.Lock()
_last_call = {"t": 0.0}

# One shared budget for the whole process. Streamlit runs every tab in a single
# process, so the backtest, all three screeners and the live poll draw on this
# same gate — a screener downloading 200 symbols cannot starve the live loop or
# double the request rate by running alongside it.
NET = {"delay": YF_MIN_DELAY, "requests": 0, "retries": 0, "limited": 0,
       "blocked_until": 0.0, "last_error": ""}

# Yahoo does not document a limit and does not always return a clean 429; the
# same throttling surfaces as an empty frame, a JSON decode failure or a curl
# error depending on where it trips. These are the fingerprints worth backing
# off on.
RATE_HINTS = ("429", "too many requests", "rate limit", "ratelimit", "rate-limit",
              "temporarily blocked", "try again later", "yfratelimit", "unauthorized",
              "max retries exceeded", "connection reset")


class RateLimitError(RuntimeError):
    """The provider is refusing requests. Callers must stop, not retry harder."""


class DataError(RuntimeError):
    """A provider returned nothing usable for an ordinary reason."""


def set_api_delay(seconds: float) -> None:
    NET["delay"] = max(0.05, float(seconds))


def looks_rate_limited(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(h in text for h in RATE_HINTS)


def throttle(min_delay: float | None = None) -> None:
    """Space out requests, and honour any cooling-off period already imposed.

    Live polling on 1-minute bars will get a session refused without this. The
    gate is process-wide and also enforces the backoff window set after a
    rate-limit response, so every caller waits it out rather than only the one
    that hit the wall.
    """
    d = NET["delay"] if min_delay is None else max(0.05, float(min_delay))
    with _rate_lock:
        now = time.time()
        wait = max(d - (now - _last_call["t"]), NET["blocked_until"] - now)
        if wait > 0:
            time.sleep(min(wait, 120.0))
        _last_call["t"] = time.time()
        NET["requests"] += 1


def _back_off(attempt: int, delay: float) -> float:
    """Exponential cooling-off, applied process-wide."""
    wait = min(90.0, max(2.0, delay * 4) * (2 ** attempt))
    NET["blocked_until"] = max(NET["blocked_until"], time.time() + wait)
    NET["limited"] += 1
    return wait


def net_status() -> str:
    remaining = max(0.0, NET["blocked_until"] - time.time())
    bits = [f"{NET['requests']:,} requests this session",
            f"{NET['delay']:.1f}s apart"]
    if NET["retries"]:
        bits.append(f"{NET['retries']} retries")
    if NET["limited"]:
        bits.append(f"{NET['limited']} rate-limit backoffs")
    if remaining > 1:
        bits.append(f"cooling off for {remaining:.0f}s")
    return " · ".join(bits)


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


def fetch_yf(symbol: str, interval: str, period: str, delay: float | None = None,
             retries: int = 3) -> pd.DataFrame:
    """Download candles, backing off rather than hammering when refused.

    An empty frame is retried once, because Yahoo returns one both for a bad
    ticker and for a throttled request, and the two are indistinguishable from
    the response alone. Anything that smells like throttling raises
    RateLimitError so callers stop instead of continuing down a symbol list and
    making the situation worse.
    """
    import yfinance as yf

    native, rule = RESAMPLE_FROM.get(interval, (interval, None))
    empties = 0
    for attempt in range(retries + 1):
        throttle(delay)
        try:
            raw = yf.Ticker(symbol).history(period=period, interval=native, auto_adjust=False)
        except Exception as e:
            NET["last_error"] = str(e)[:200]
            if looks_rate_limited(e):
                if attempt < retries:
                    NET["retries"] += 1
                    _back_off(attempt, NET["delay"] if delay is None else delay)
                    continue
                raise RateLimitError(
                    f"The data provider is refusing requests ({str(e)[:120]}). Increase the gap "
                    f"between API calls in the sidebar and wait a few minutes.") from e
            raise DataError(f"{symbol}: {str(e)[:160]}") from e

        if raw is None or raw.empty:
            empties += 1
            if empties == 1 and attempt < retries:
                NET["retries"] += 1
                _back_off(attempt, NET["delay"] if delay is None else delay)
                continue
            raise DataError(f"yfinance returned no rows for {symbol} at {native}/{period}.")

        df = _flatten(raw, symbol)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if rule:
            df = resample_ohlc(df, rule)
        df.index.name = "Datetime"
        return df
    raise DataError(f"Could not load {symbol}.")


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
    return {"every_bar": st_.checkbox("Re-enter on every flat bar", value=True, key=f"{k}_every_bar")}


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
        val = st_.number_input("Level", value=0.0, step=0.05, format="%.4f", key=f"{k}_val")
    else:
        val = st_.number_input("Percent offset (%)  ·  negative sits below the reference",
                               value=1.0, step=0.1, format="%.2f", key=f"{k}_val")
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
    return {"fast": c1.number_input("Fast EMA", 2, 400, 9, key=f"{k}_fast"),
            "slow": c2.number_input("Slow EMA", 3, 800, 21, key=f"{k}_slow"),
            "trend": c3.number_input("Trend EMA (0 = off)", 0, 800, 50, key=f"{k}_trend")}


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
            "zero": st_.checkbox("Require the MACD line on the right side of zero", key=f"{k}_zero")}


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
    mode = st_.radio("Trade the", ["Breakout", "Mean reversion"], horizontal=True, key=f"{k}_mode")
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
    # The breakout is the bar that first pierces the channel, not every bar that
    # happens to sit outside it. A state would re-fire for as long as price stays
    # beyond the band.
    up = (df["High"] > phi) & ~(df["High"].shift(1) > phi.shift(1)).astype(bool)
    dn = (df["Low"] < plo) & ~(df["Low"].shift(1) < plo.shift(1)).astype(bool)
    return Signals(_sig(df.index, up, dn),
                   overlays={"Donchian high": phi, "Donchian low": plo})


def p_vwap(st_, k):
    return {"buf": st_.number_input("Reclaim buffer (%)", 0.0, 5.0, 0.05, 0.01, key=f"{k}_buf")}


def c_vwap(df, p):
    v = pine_vwap(df)
    b = float(p["buf"]) / 100
    return Signals(_sig(df.index, cross_up(df["Close"], v * (1 + b)),
                        cross_dn(df["Close"], v * (1 - b))), overlays={"VWAP": v})


def p_orb(st_, k):
    c1, c2 = st_.columns(2)
    return {"mins": c1.number_input("Opening range (minutes)", 1, 240, 15, key=f"{k}_mins"),
            "once": c2.checkbox("One trade per day", value=True, key=f"{k}_once")}


def c_orb(df, p):
    day = pd.Series(df.index.date, index=df.index)
    start = df.groupby(day).apply(lambda g: g.index[0]).reindex(day.values).values
    elapsed = (df.index.values - start) / np.timedelta64(1, "m")
    inrange = pd.Series(elapsed < int(p["mins"]), index=df.index)
    hi = df["High"].where(inrange).groupby(day).cummax().groupby(day).ffill()
    lo = df["Low"].where(inrange).groupby(day).cummin().groupby(day).ffill()
    # Fire on the bar that breaks the range, not on every bar spent outside it.
    # shift(fill_value=) keeps the mask boolean. A plain .shift() on a bool
    # Series returns object dtype, where ~True evaluates to -2 and every row
    # passes — the mask silently stops filtering anything.
    above, below = df["Close"] > hi, df["Close"] < lo
    lc = (~inrange) & above & ~above.shift(1, fill_value=False)
    sc = (~inrange) & below & ~below.shift(1, fill_value=False)
    if p["once"]:
        lc &= lc.groupby(day).cumsum().eq(1)
        sc &= sc.groupby(day).cumsum().eq(1)
    return Signals(_sig(df.index, lc, sc), overlays={"OR high": hi, "OR low": lo})


def p_adx_trend(st_, k):
    c1, c2, c3 = st_.columns(3)
    return {"n": c1.number_input("DI length", 2, 100, 14, key=f"{k}_n"),
            "a": c2.number_input("ADX smoothing", 2, 100, 14, key=f"{k}_start"),
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

# Used when the scanner sweeps strategies rather than taking the sidebar's.
STRATEGY_DEFAULTS: dict[str, dict] = {
    "Simple buy": {"every_bar": True},
    "Simple sell": {"every_bar": True},
    "Price crosses threshold": {"dir": "Crosses up through level", "mode": "Percent from reference",
                                "val": 1.0, "ref": "First close of window"},
    "EMA crossover": {"fast": 9, "slow": 21, "trend": 50},
    "Supertrend flip": {"n": 10, "m": 3.0},
    "RSI reversal / momentum": {"n": 14, "lo": 30, "hi": 70, "mode": "Reversal (buy oversold)"},
    "MACD crossover": {"f": 12, "s": 26, "g": 9, "zero": False},
    "Bollinger band": {"n": 20, "k": 2.0, "mode": "Breakout"},
    "Donchian breakout": {"n": 20},
    "VWAP reclaim": {"buf": 0.05},
    "Opening range breakout": {"mins": 15, "once": True},
    "ADX trend ride": {"n": 14, "a": 14, "min": 25},
}
# Unconditional entries exist to sanity-check costs, not to be ranked.
SWEEPABLE = [k for k in STRATEGY_DEFAULTS if not k.startswith("Simple")]


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
                           "Above the basis / below the basis"], key=f"{k}_where")
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
    return {"fast": c1.number_input("Fast EMA", 2, 400, 20, key=f"{k}_fast"),
            "slow": c2.number_input("Slow EMA", 3, 800, 50, key=f"{k}_slow")}


def a_ema(df, p):
    f = pine_ema(df["Close"], int(p["fast"]))
    s = pine_ema(df["Close"], int(p["slow"]))
    return f > s, f < s


def f_adx(st_, k):
    c1, c2, c3 = st_.columns(3)
    n = c1.number_input("DI length", 2, 100, 14, key=f"{k}_n")
    lo = c2.number_input("Min ADX", 0, 100, 20, key=f"{k}_min")
    hi = c3.number_input("Max ADX", 0, 100, 60, key=f"{k}_max")
    a = st_.number_input("ADX smoothing", 2, 100, 14, key=f"{k}_start")
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
            "lmin": c2.number_input("Long: RSI above", 0, 100, 50, key=f"{k}_lmin"),
            "smax": c3.number_input("Short: RSI below", 0, 100, 50, key=f"{k}_smax")}


def a_rsi(df, p):
    r = pine_rsi(df["Close"], int(p["n"]))
    return r >= p["lmin"], r <= p["smax"]


def f_regime(st_, k):
    c1, c2 = st_.columns(2)
    n = c1.number_input("Regime EMA", 10, 1000, 200, key=f"{k}_n")
    slope = c2.number_input("Slope lookback (bars)", 1, 200, 20, key=f"{k}_s")
    mode = st_.selectbox("Regime rule", ["Price vs EMA", "EMA slope", "Both must agree"], key=f"{k}_mode")
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
            "min": c2.number_input("Min ATR as % of price", 0.0, 20.0, 0.05, 0.01, key=f"{k}_min")}


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
    return {"start": c1.time_input("No entries before", value=pd.Timestamp("09:20").time(), key=f"{k}_start"),
            "end": c2.time_input("No entries after", value=pd.Timestamp("15:00").time(), key=f"{k}_buf")}


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
    fine: pd.DataFrame | None = None      # finer bars used to resolve intrabar order
    interval: str = "5m"
    tie_break: str = "stop"               # which side wins when one bar holds both

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
        ambiguous = 0                     # bars containing both levels at once
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
            if pos is not None and self.fine is not None:
                lo_b = idx[i]
                hi_b = idx[i + 1] if i + 1 < len(df) else idx[i] + bar_duration(self.interval)
                seg = self.fine.loc[(self.fine.index >= lo_b) & (self.fine.index < hi_b)]
                px = reason = None
                for k in range(len(seg)):
                    fo = float(seg["Open"].iloc[k])
                    fh = float(seg["High"].iloc[k])
                    fl = float(seg["Low"].iloc[k])
                    if not np.isnan(pos.sl) and ((fo <= pos.sl) if pos.side > 0 else (fo >= pos.sl)) \
                            and (i > pos.entry_bar or k > 0):
                        px, reason = fo, "Stop loss (gap)"
                        break
                    if not np.isnan(pos.sl) and ((fl <= pos.sl) if pos.side > 0 else (fh >= pos.sl)):
                        px, reason = pos.sl, "Stop loss"
                        break
                    if live_tgt and not np.isnan(pos.target) and \
                            ((fh >= pos.target) if pos.side > 0 else (fl <= pos.target)):
                        px, reason = pos.target, "Target"
                        break
                if px is None and sq is not None and times[i] >= sq:
                    px, reason = c[i], "Session square-off"
                if px is not None:
                    adverse = (l[i] - pos.entry_price) if pos.side > 0 else (pos.entry_price - h[i])
                    pos.meta["mae"] = min(pos.meta["mae"], adverse)
                    trades.append(self._close(pos, px, idx[i], reason, i, o, h, l, c))
                    cum += trades[-1]["Net P&L"]
                    pos = None

            if pos is not None and self.fine is None:
                adverse = (l[i] - pos.entry_price) if pos.side > 0 else (pos.entry_price - h[i])
                favour = (h[i] - pos.entry_price) if pos.side > 0 else (pos.entry_price - l[i])
                pos.meta["mae"] = min(pos.meta["mae"], adverse)
                pos.meta["mfe"] = max(pos.meta["mfe"], favour)

                px = reason = None
                # A gap opens the bar beyond a level, so the fill is the OPEN, not
                # the level. Filling at the stop price after an overnight gap-down
                # is the single most flattering error a backtest can make. Skipped
                # on the entry bar, where the fill is the open by construction.
                gapped = i > pos.entry_bar
                if gapped and not np.isnan(pos.sl) and \
                        ((o[i] <= pos.sl) if pos.side > 0 else (o[i] >= pos.sl)):
                    px, reason = o[i], "Stop loss (gap)"
                elif gapped and live_tgt and not np.isnan(pos.target) and \
                        ((o[i] >= pos.target) if pos.side > 0 else (o[i] <= pos.target)):
                    px, reason = o[i], "Target (gap)"
                sl_hit = (px is None) and (not np.isnan(pos.sl)) and (
                    (l[i] <= pos.sl) if pos.side > 0 else (h[i] >= pos.sl))
                tg_hit = (px is None) and live_tgt and (not np.isnan(pos.target)) and (
                    (h[i] >= pos.target) if pos.side > 0 else (l[i] <= pos.target))
                if sl_hit and tg_hit:
                    # This bar reached both levels. Bar data cannot say which came
                    # first, so one of them has to be chosen; counting these is the
                    # only way to know how much of a result rests on that choice.
                    ambiguous += 1
                if sl_hit and tg_hit and self.tie_break == "target":
                    px, reason = pos.target, "Target"
                elif sl_hit:
                    px, reason = pos.sl, "Stop loss"
                elif tg_hit:
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
        self.ambiguous_bars = ambiguous
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
            "Target at entry": _round(pos.meta.get("target_at_entry")),
            "Target at exit": _round(pos.target), "Bar open": round(o[i], 4),
            "Bar high": round(h[i], 4), "Bar low": round(l[i], 4), "Bar close": round(c[i], 4),
            "Gross P&L": round(gross, 2), "Charges": round(ch, 2), "Net P&L": round(gross - ch, 2),
            "R multiple": round(r, 2) if r == r else np.nan,
            "MAE": round(pos.meta.get("mae", 0.0), 2), "MFE": round(pos.meta.get("mfe", 0.0), 2),
            "Bars held": i - pos.entry_bar, "Entry reason": pos.entry_reason, "Exit reason": reason,
        }


TRADE_COLS = ["Entry time", "Exit time", "Direction", "Qty", "Entry price", "Exit price",
              "Stop at entry", "Stop at exit", "Target at entry", "Target at exit", "Bar open", "Bar high",
              "Bar low", "Bar close", "Gross P&L", "Charges", "Net P&L", "R multiple",
              "MAE", "MFE", "Bars held", "Entry reason", "Exit reason"]


def _round(v):
    return np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


def bars_per_year(idx: pd.DatetimeIndex) -> float:
    """Annualisation factor taken from the data rather than assumed.

    Counting actual bars against elapsed wall-clock time means the same code
    works for a 1-minute NSE session and a 24/7 crypto series without a table
    of hardcoded session lengths.
    """
    if len(idx) < 3:
        return 252.0
    span = (idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 3600)
    return len(idx) / span if span > 0 else 252.0


def sharpe_sortino(equity: pd.Series) -> tuple[float, float]:
    """Annualised Sharpe and Sortino from the equity curve.

    Computed on per-bar P&L increments. Because both the mean and the standard
    deviation scale linearly with position size, this is identical to the ratio
    you would get from percentage returns on a fixed capital base, without
    needing to invent a capital base. Flat bars count as zero-return periods,
    which is what makes this a time-weighted rather than a trade-weighted
    number. The risk-free rate is taken as zero.
    """
    d = equity.diff().dropna()
    if len(d) < 3 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    ann = np.sqrt(bars_per_year(equity.index))
    sharpe = d.mean() / d.std(ddof=1) * ann
    # Target downside deviation over every period, not the standard deviation of
    # the losing subset. With a fixed stop every loss is the same size, so the
    # subset's deviation is zero and the ratio would be undefined.
    tdd = np.sqrt((np.minimum(d, 0.0) ** 2).mean())
    sortino = d.mean() / tdd * ann if tdd > 0 else float("nan")
    return float(sharpe), float(sortino)


def summarise(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if trades.empty:
        return {"Trades": 0}
    net = trades["Net P&L"]
    wins, losses = net[net > 0], net[net <= 0]
    dd = equity - equity.cummax()
    gp, gl = wins.sum(), -losses.sum()
    bw, bl = _streaks(net)
    sharpe, sortino = sharpe_sortino(equity)
    # Planned reward:risk, measured from the levels actually set at entry. A
    # target far tighter than the stop buys a high hit rate with a payoff that
    # cannot survive its own losses, and neither hit rate nor Sharpe exposes it.
    risk_d = (trades["Entry price"] - trades["Stop at entry"]).abs()
    rew_d = (trades["Target at entry"] - trades["Entry price"]).abs()
    ok = risk_d.notna() & rew_d.notna() & (risk_d > 0)
    planned_rr = float((rew_d[ok] / risk_d[ok]).median()) if ok.any() else np.nan
    net_points = trades["Gross P&L"].sum() / max(int(trades["Qty"].iloc[0]), 1)
    per_trade_sharpe = net.mean() / net.std(ddof=1) if len(net) > 1 and net.std(ddof=1) > 0 else np.nan
    return {
        "Trades": len(net), "Wins": len(wins), "Losses": len(losses),
        "Sharpe (annualised)": round(sharpe, 3) if sharpe == sharpe else np.nan,
        "Sortino (annualised)": round(sortino, 3) if sortino == sortino else np.nan,
        "Sharpe per trade": round(per_trade_sharpe, 3) if per_trade_sharpe == per_trade_sharpe else np.nan,
        "Planned reward:risk": round(planned_rr, 2) if planned_rr == planned_rr else np.nan,
        "Net points per unit": round(net_points, 2),
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


def apply_config_to_sidebar(cfg: dict) -> None:
    """Write a saved configuration back into the sidebar widgets.

    Streamlit widgets read their value from session_state under their key, so
    setting those keys before the widgets are built makes the sidebar show the
    configuration — and leaves every control editable, which an overlay banner
    would not. Queued here and applied at the top of the next run, because a
    widget's key cannot be written after it has been instantiated.
    """
    ss = st.session_state
    sym = cfg["symbol"]
    group = next((g for g, m in UNIVERSE.items() if sym in m.values()), None)
    if group:
        ss["sb_group"] = group
        ss["sb_name"] = next(n for n, v in UNIVERSE[group].items() if v == sym)
        ss["sb_custom"] = ""
    else:
        ss["sb_custom"] = sym
    ss["sb_tf"] = cfg["interval"]
    ss[f"period_{cfg.get('source', 'yfinance')}_{cfg['interval']}"] = cfg["period"]
    for key, val in (("sb_qty", cfg.get("qty")), ("sb_side", cfg.get("side_mode")),
                     ("sb_strategy", cfg.get("strategy"))):
        if val is not None:
            ss[key] = val
    for name, val in (cfg.get("strat_params") or {}).items():
        ss[f"sp_{cfg['strategy']}_{name}"] = val
    sl, tg = cfg.get("sl_cfg") or {}, cfg.get("tgt_cfg") or {}
    ss["sb_sl_mode"] = sl.get("mode", SL_MODES[0])
    for name, val in sl.items():
        if name == "be":
            ss["sl_be"] = bool(val.get("on"))
            for bk, bv in val.items():
                if bk != "on":
                    ss[f"be_{bk}"] = bv
        elif name != "mode":
            ss[f"sl_{name}"] = val
    ss["sb_tgt_mode"] = tg.get("mode", TGT_MODES[0])
    for name, val in tg.items():
        if name != "mode":
            ss[f"tg_{name}"] = val
    for fname in FILTERS:
        ss[f"flt_{fname}"] = fname in (cfg.get("filters") or {})
    for fname, params in (cfg.get("filters") or {}).items():
        for name, val in params.items():
            ss[f"fp_{fname}_{name}"] = val


def consume_pending_sidebar() -> str | None:
    """Applies anything queued by a screener, before the widgets are built."""
    pending = st.session_state.pop("pending_sidebar", None)
    if not pending:
        return None
    apply_config_to_sidebar(pending)
    return pending.get("_label") or pending.get("symbol")


def render_sidebar() -> dict:
    s = st.sidebar
    s.markdown("### Setup")

    source = s.selectbox("Data source", ["yfinance", "Dhan"], index=0, key="sb_source",
                         help="yfinance is free and delayed. Dhan needs credentials but gives "
                              "finer buckets and a real last-traded price.")

    groups = list(UNIVERSE)
    g = s.selectbox("Asset class", groups, index=groups.index(DEFAULT_GROUP), key="sb_group")
    names = list(UNIVERSE[g])
    name = s.selectbox("Instrument", names, key="sb_name",
                       index=names.index(DEFAULT_NAME) if DEFAULT_NAME in names else 0)
    symbol = UNIVERSE[g][name]
    custom = s.text_input("Or type a ticker", value="", placeholder="KAYNES.NS", key="sb_custom")
    if custom.strip():
        symbol = name = custom.strip()
        s.markdown(f"Trading **{symbol}** — typed in the box above, which overrides the two "
                   f"dropdowns. Clear the box to go back to them.")
    else:
        s.markdown(f"Trading **{symbol}**")

    tfmap = DHAN_TF_PERIODS if source == "Dhan" else TF_PERIODS
    tfs = list(tfmap)
    tf = s.selectbox("Timeframe", tfs, key="sb_tf",
                     index=tfs.index(DEFAULT_TF) if DEFAULT_TF in tfs else 0)
    opts, default_period = tfmap[tf]
    pkey = f"period_{source}_{tf}"
    phelp = "The list narrows to what the provider will actually return for this timeframe."
    if pkey in st.session_state and st.session_state[pkey] in opts:
        period = s.selectbox("Period", opts, key=pkey, help=phelp)
    else:
        period = s.selectbox("Period", opts, index=opts.index(default_period), key=pkey,
                             help=phelp)

    c1, c2 = s.columns(2)
    qty = c1.number_input("Quantity", 1, 1_000_000, 1, key="sb_qty")
    side_mode = c2.selectbox("Sides", SIDES, index=0, key="sb_side")
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
                           label_visibility="collapsed", key="sb_strategy")
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
    set_api_delay(st.session_state.get("sb_api_delay", YF_MIN_DELAY))
    r1, r2 = s.columns(2)
    refresh = r1.number_input("Refresh every (s)", 0.3, 300.0, 15.0, step=0.1, format="%.1f",
                              help="How often the live panel redraws. Only the panel redraws; "
                                   "the page does not reload.")
    api_delay = r2.number_input("Gap between API calls (s)", 0.1, 30.0, YF_MIN_DELAY, step=0.1,
                                format="%.1f", key="sb_api_delay",
                                help="Minimum spacing between every outbound request, shared by "
                                     "all tabs. Separate from the refresh rate. Raise it if the "
                                     "screeners start reporting rate-limit backoffs.")
    set_api_delay(api_delay)
    if refresh < 5:
        s.warning(f"Each cycle makes two requests, so at {api_delay:.1f}s apart a cycle cannot "
                  f"finish faster than {2 * api_delay:.1f}s. Polling this hard also risks a "
                  f"temporary block from the data provider.", icon="⚠️")
    with s.expander("Data usage and rate limits"):
        st.caption(net_status())
        if NET["limited"]:
            st.warning(f"The provider has thrown {NET['limited']} rate-limit backoff(s) this "
                       f"session. Raise the gap between API calls, or scan fewer symbols at once.")
        if NET["last_error"]:
            st.caption(f"Last provider error: {NET['last_error']}")
        st.caption("Candles are cached for 10 minutes, so re-running a screener over the same "
                   "universe costs no requests.")
        if st.button("Clear cached candles", width="stretch"):
            load_candles.clear()
            st.success("Cache cleared. The next run will refetch.")

    trail_on_close = s.checkbox("Move trailing stops only when a bar closes", value=True,
                                help="On by default because it is what the backtest models, so "
                                     "live and backtest agree. Turning it off trails on every "
                                     "price update, which exits earlier and will not match any "
                                     "backtest you ran.")
    dhan = _dhan_panel(s, symbol, qty)
    gmail = _gmail_panel(s)

    return dict(source=source, group=g, name=name, symbol=symbol, interval=tf, period=period,
                qty=int(qty), side_mode=side_mode, instrument=instrument, is_options=is_options,
                flip=flip, strategy=strategy, strat_params=strat_params, sl_cfg=sl_cfg,
                tgt_cfg=tgt_cfg, sl_mode=sl_cfg["mode"], tgt_mode=tgt_cfg["mode"],
                allow_reverse=allow_reverse, square_off=square_off, filters=selected,
                filter_logic=logic, forward=dict(on=fwd_on, days=int(fwd_days)), costs=costs,
                refresh=float(refresh), api_delay=float(api_delay),
                trail_on_close=trail_on_close, dhan=dhan, gmail=gmail)


def _stop_panel(s) -> dict:
    mode = s.selectbox("Stop loss", SL_MODES, index=0, key="sb_sl_mode")
    cfg = {"mode": mode}
    with s.container(border=True):
        if mode == "Fixed points":
            cfg["points"] = st.number_input("Stop distance (points)", 0.01, 1e6, 10.0, 0.5, key="sl_points")
        elif mode == "Fixed percent":
            cfg["pct"] = st.number_input("Stop distance (%)", 0.01, 100.0, 1.0, 0.05, key="sl_pct")
        elif mode == "Trailing points":
            cfg["points"] = st.number_input("Trail by (points)", 0.01, 1e6, 10.0, 0.5, key="sl_points")
        elif mode == "ATR multiple":
            c1, c2 = st.columns(2)
            cfg["atr_len"] = c1.number_input("ATR length", 2, 200, 14, key="sl_atr_len")
            cfg["atr_mult"] = c2.number_input("Multiple", 0.1, 20.0, 2.0, 0.1, key="sl_atr_mult")
            cfg["atr_trail"] = st.checkbox("Trail it from the best price", value=True, key="sl_atr_trail")
        elif mode == "Derived from reward (risk:reward)":
            c1, c2 = st.columns(2)
            cfg["reward_points"] = c1.number_input("Reward (points)", 0.01, 1e6, 20.0, 0.5, key="sl_reward_points")
            cfg["rr"] = c2.number_input("Reward : risk", 0.1, 20.0, 2.0, 0.1, key="sl_rr")
        elif mode == "Strategy signal only (no price stop)":
            st.caption("The position stays open until the strategy prints the opposite signal. "
                       "There is no price stop, so size accordingly.")
        else:
            c1, c2 = st.columns(2)
            cfg["swing_n"] = c1.number_input("Swing lookback (bars)", 1, 100, 3, key="sl_swing_n")
            cfg["buffer"] = c2.number_input("Buffer beyond level", 0.0, 1e5, 0.0, 0.05, key="sl_buffer")

        st.markdown("")
        be_on = st.checkbox("Move the stop to cost once the trade runs", value=False, key="sl_be")
        be = {"on": be_on}
        if be_on:
            c1, c2 = st.columns(2)
            be["unit"] = c1.selectbox("Trigger measured in", ["Points", "Percent", "R multiple"], key="be_unit")
            be["value"] = c2.number_input("Trigger at", 0.01, 1e6, 10.0, 0.5, key="be_value")
            c3, c4 = st.columns(2)
            be["offset"] = c3.number_input("Lock in beyond cost", 0.0, 1e5, 0.0, 0.05, key="be_offset")
            be["trail_only_after"] = c4.checkbox("Start trailing only after this", value=False, key="be_trail_only_after")
        cfg["be"] = be
    return cfg


def _target_panel(s) -> dict:
    mode = s.selectbox("Target", TGT_MODES, index=0, key="sb_tgt_mode")
    cfg = {"mode": mode}
    with s.container(border=True):
        if mode == "Fixed points":
            cfg["points"] = st.number_input("Target distance (points)", 0.01, 1e6, 20.0, 0.5, key="tg_points")
        elif mode == "Fixed percent":
            cfg["pct"] = st.number_input("Target distance (%)", 0.01, 500.0, 2.0, 0.05, key="tg_pct")
        elif mode == "Trailing target (display only)":
            cfg["points"] = st.number_input("Project ahead by (points)", 0.01, 1e6, 20.0, 0.5, key="tg_points")
            st.caption("Drawn on the chart as a projection. It never closes the trade — the stop does.")
        elif mode == "Risk:reward multiple":
            cfg["rr"] = st.number_input("Reward : risk", 0.1, 50.0, 2.0, 0.1, key="tg_rr")
        elif mode == "ATR multiple":
            c1, c2 = st.columns(2)
            cfg["atr_len"] = c1.number_input("ATR length", 2, 200, 14, key="tg_atr_len")
            cfg["atr_mult"] = c2.number_input("Multiple", 0.1, 50.0, 3.0, 0.1, key="tg_atr_mult")
        elif mode == "Strategy reversal exit (no fixed target)":
            st.caption("The trade runs until the strategy reverses or the stop is hit.")
        else:
            cfg["swing_n"] = st.number_input("Swing lookback (bars)", 1, 100, 3, key="tg_swing_n")
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
# Plotly grabs the mouse wheel, so a chart sitting mid-page hijacks scrolling.
# Charts go at the bottom of the tab and wheel-zoom is off; use the toolbar's
# box-zoom or drag to pan instead.
PLOT_CFG = {"scrollZoom": False, "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def config_fingerprint(cfg: dict) -> str:
    """Identifies a run. Changing any setting invalidates the shown result."""
    parts = [cfg["symbol"], cfg["interval"], cfg["period"], cfg["strategy"], str(cfg["qty"]),
             cfg["side_mode"], str(cfg["flip"]), cfg["instrument"], str(cfg["strat_params"]),
             str(cfg["sl_cfg"]), str(cfg["tgt_cfg"]), str(cfg["filters"]), cfg["filter_logic"],
             str(cfg["allow_reverse"]), str(cfg["square_off"]), str(cfg["costs"].to_dict()),
             str(cfg["forward"])]
    return str(hash("|".join(parts)))


def run_config(cfg: dict, df: pd.DataFrame):
    """The single code path used by the backtest tab, the forward test and the
    scanner. Sharing it is what makes a shortlisted result reproducible."""
    sig, al, as_ = prepare(cfg, df)
    trades, equity = make_engine(cfg, df, sig, al, as_).run()
    return sig, trades, equity


def render_backtest(cfg: dict, df: pd.DataFrame):
    ss = st.session_state
    fp = config_fingerprint(cfg)
    replay = ss.get("replay_cfg")

    if replay is not None:
        st.info(f"Showing a configuration sent over from the scanner: "
                f"**{replay['symbol']} · {replay['interval']} / {replay['period']} · "
                f"{replay['strategy']}**. The sidebar is ignored until you clear it.")
        c1, c2 = st.columns([1, 4])
        if c1.button("Clear and use the sidebar", width="stretch"):
            ss.pop("replay_cfg", None)
            ss.pop("bt_result", None)
            st.rerun()
        cfg = replay
        try:
            df = load_candles(cfg["symbol"], cfg["interval"], cfg["period"])
        except Exception as e:
            st.error(f"Could not reload candles for the replayed configuration: {e}")
            return
        fp = config_fingerprint(cfg)

    c1, c2 = st.columns([1, 4])
    run_now = c1.button("Run backtest", type="primary", width="stretch")
    cached = ss.get("bt_result")
    if cached is not None and cached["fp"] != fp:
        c2.warning("Settings changed since this result was produced. Run it again.")
    elif cached is None:
        c2.caption("Set the strategy and exits in the sidebar, then run.")

    if run_now:
        with st.spinner("Running…"):
            sig, al, as_ = prepare(cfg, df)
            worst = make_engine(cfg, df, sig, al, as_)
            trades, equity = worst.run()
            best = make_engine(cfg, df, sig, al, as_)
            best.tie_break = "target"
            t_best, e_best = best.run()
            ss["bt_result"] = {"fp": fp, "sig": sig, "trades": trades, "equity": equity,
                               "cfg": cfg, "df": df, "summary": summarise(trades, equity),
                               "ambiguous": worst.ambiguous_bars,
                               "best_net": float(t_best["Net P&L"].sum()) if len(t_best) else 0.0}
        cached = ss["bt_result"]

    if cached is None:
        with st.expander("Configuration ready to run", expanded=True):
            st.dataframe(config_summary(cfg), width="stretch", hide_index=True, height=560)
        return

    cfg, df = cached["cfg"], cached["df"]
    sig, trades, equity, summary = cached["sig"], cached["trades"], cached["equity"], cached["summary"]

    warm = warmup_report(sig)
    if warm:
        st.warning("Undefined on the final bar: " + "; ".join(warm) + ". A recursive average needs "
                   "its full lookback before it produces anything, so widen the period or shorten "
                   "the lookback.", icon="⚠️")

    st.markdown("#### Result")
    if trades.empty:
        st.info("No trades were filled with this configuration.")
    else:
        k = st.columns(7)
        k[0].metric("Net P&L", f"{summary['Net P&L']:,.2f}")
        k[1].metric("Net points", f"{summary['Net points per unit']:,.2f}")
        k[2].metric("Trades", summary["Trades"])
        k[3].metric("Hit rate", f"{summary['Hit rate %']}%")
        k[4].metric("Sharpe", _fmt_num(summary["Sharpe (annualised)"]))
        k[5].metric("Profit factor", summary["Profit factor"])
        k[6].metric("Max drawdown", f"{summary['Max drawdown']:,.2f}")
        k2 = st.columns(7)
        k2[0].metric("Sortino", _fmt_num(summary["Sortino (annualised)"]))
        k2[1].metric("Expectancy", f"{summary['Expectancy per trade']:,.2f}")
        k2[2].metric("Average R", _fmt_num(summary["Average R"]))
        k2[3].metric("Average win", f"{summary['Average win']:,.2f}")
        k2[4].metric("Average loss", f"{summary['Average loss']:,.2f}")
        k2[5].metric("Longest loss streak", summary["Longest loss streak"])
        k2[6].metric("Charges", f"{summary['Total charges']:,.2f}")

    amb = cached.get("ambiguous", 0)
    if not trades.empty:
        worst_net = summary["Net P&L"]
        best_net = round(cached.get("best_net", worst_net), 2)
        st.markdown("##### How much of this rests on an assumption")
        b = st.columns(3)
        b[0].metric("Worst case (what is shown above)", f"{worst_net:,.2f}")
        b[1].metric("Best case (same trades, other tie-break)", f"{best_net:,.2f}")
        b[2].metric("Exits where both levels were hit", f"{amb} of {len(trades)}")
        if amb == 0:
            st.success("No single bar ever contained both your stop and your target, so there was "
                       "nothing to guess. These numbers do not depend on intrabar order at all.")
        else:
            share = 100 * amb / max(len(trades), 1)
            st.warning(
                f"On {amb} exits ({share:.0f}% of trades) one bar reached **both** your stop and "
                f"your target. Bar data cannot say which came first. This app always assumes the "
                f"stop — the worst case — which is why the headline is {worst_net:,.2f}. Had every "
                f"one of those gone the other way it would read {best_net:,.2f}. The truth is "
                f"somewhere between. To narrow the gap, widen your levels relative to the bar or "
                f"drop to a smaller timeframe.")

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

    render_verification(cfg, df)

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

    st.markdown("#### Charts")
    st.caption("Wheel-zoom is disabled so the page scrolls normally. Use the toolbar to zoom.")
    st.plotly_chart(build_chart(df, sig, trades,
                                title=f"{cfg['name']} · {cfg['interval']} · {cfg['strategy']}"),
                    width="stretch", config=PLOT_CFG)
    if not trades.empty:
        st.plotly_chart(equity_chart(equity), width="stretch", config=PLOT_CFG)


def _fmt_num(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{nd}f}"


def verify_intrabar(cfg: dict, coarse: pd.DataFrame) -> dict | None:
    """Re-run the same configuration with exits resolved on 1-minute bars.

    OHLC data cannot say whether a bar reached its high or its low first. Every
    backtest on bar data guesses, and this one guesses conservatively — the stop
    is tested against the bar's adverse extreme before the target. That guess is
    usually harmless and occasionally decisive, and which it is depends entirely
    on how wide your stop is relative to the bar range.

    Rather than argue about it, this refetches 1-minute candles for the same
    window and walks them in true chronological order, so stop and target fire
    in the sequence they actually would have. The gap between the two runs is
    the size of the modelling error on your setup.

    Providers only keep about 7 days of 1-minute history, so this covers the
    recent window and not the whole backtest.
    """
    try:
        fine = load_candles(cfg["symbol"], "1m", "7d")
    except Exception:
        return None
    if fine.empty or coarse.empty:
        return None
    if fine.index.tz is not None and coarse.index.tz is None:
        fine.index = fine.index.tz_localize(None)
    elif fine.index.tz is None and coarse.index.tz is not None:
        fine.index = fine.index.tz_localize(coarse.index.tz)
    start = max(fine.index[0], coarse.index[0])
    window = coarse[coarse.index >= start]
    if len(window) < 30:
        return None
    sig, al, as_ = prepare(cfg, window)
    plain = make_engine(cfg, window, sig, al, as_)
    t_bar, e_bar = plain.run()
    resolved = make_engine(cfg, window, sig, al, as_)
    resolved.fine = fine
    resolved.interval = cfg["interval"]
    t_fine, e_fine = resolved.run()
    return {"bars": len(window), "from": window.index[0], "to": window.index[-1],
            "coarse": summarise(t_bar, e_bar), "fine": summarise(t_fine, e_fine),
            "t_coarse": t_bar, "t_fine": t_fine}


def render_verification(cfg: dict, df: pd.DataFrame):
    st.markdown("#### Exit-path verification")
    st.caption("Bar data cannot tell whether the high or the low came first. This re-runs the same "
               "settings with exits resolved on 1-minute candles, so you can see how much that "
               "assumption is worth on your setup rather than trusting it.")
    if not st.button("Verify exits on 1-minute data", width="stretch"):
        return
    with st.spinner("Refetching 1-minute candles and replaying…"):
        v = verify_intrabar(cfg, df)
    if v is None:
        st.warning("Could not run the check. It needs 1-minute history, which providers keep for "
                   "about 7 days, and at least 30 bars of overlap with your window.")
        return
    keys = ["Trades", "Hit rate %", "Net P&L", "Net points per unit", "Profit factor",
            "Expectancy per trade", "Sharpe (annualised)", "Max drawdown"]
    rows = []
    for k in keys:
        a, b = v["coarse"].get(k), v["fine"].get(k)
        drift = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == a and b == b:
            drift = f"{b - a:+,.2f}" if abs(a) > 0 else "—"
        rows.append((k, a, b, drift))
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Bar data (what the app reports)",
                                             "1-minute resolved", "Difference"]).astype(str),
                 width="stretch", hide_index=True)
    na, nb = v["coarse"].get("Net P&L", 0), v["fine"].get("Net P&L", 0)
    st.caption(f"Window {v['from']:%d-%b %H:%M} to {v['to']:%d-%b %H:%M}, {v['bars']:,} bars.")
    if isinstance(na, (int, float)) and isinstance(nb, (int, float)) and abs(na) > 1e-9:
        gap = abs(nb - na) / abs(na) * 100
        if gap < 10:
            st.success(f"The two runs differ by {gap:.1f}% on net result. Your stop is wide enough "
                       f"relative to the bar range that intrabar order rarely decides the trade.")
        elif gap < 35:
            st.warning(f"The two runs differ by {gap:.1f}% on net result. Intrabar order matters "
                       f"here. Treat the headline figure as indicative, not precise.", icon="⚠️")
        else:
            st.error(f"The two runs differ by {gap:.1f}% on net result. Your stop and target sit "
                     f"inside the typical bar range, so the backtest is mostly measuring an "
                     f"assumption about path. Use a lower timeframe or wider levels.", icon="⚠️")


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
        _, t2, e2 = run_config(cfg, part)
        m = summarise(t2, e2)
        with col:
            st.markdown(f"**{label}** · {len(part)} bars")
            if m["Trades"] == 0:
                st.caption("No trades.")
            else:
                st.dataframe(pd.DataFrame(
                    [(k, str(m[k])) for k in ("Trades", "Hit rate %", "Net P&L", "Profit factor",
                                              "Sharpe (annualised)", "Expectancy per trade",
                                              "Max drawdown")],
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
    ss.setdefault("live_start_bar", None)
    ss.setdefault("live_trail_bar", None)
    ss.setdefault("live_manual_exit", False)

    c1, c2, c3, c4 = st.columns([1, 1, 1.3, 3])
    if c1.button("Start trading", type="primary", width="stretch", disabled=ss.live_on):
        ss.live_on = True
        ss.live_start_bar = None       # the first poll records the baseline bar
        ss.live_trail_bar = None
        log_live("Trading started. The bar already closed is treated as history; "
                 "entries begin from the next one.")
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
        df = fetch_yf(cfg["symbol"], cfg["interval"], cfg["period"], cfg.get("api_delay", YF_MIN_DELAY))
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

    ltp = ltp_yf(cfg["symbol"], cfg.get("api_delay", YF_MIN_DELAY)) or float(forming["Close"])
    last_ts = closed.index[-1]
    pos: Position | None = ss.live_pos

    if pos is not None:
        # The backtest moves a trailing stop once per bar, using that bar's high
        # and low. Trailing here on every poll instead would be a different rule
        # — tighter, exiting earlier — and live results would drift from the
        # backtest for reasons that have nothing to do with the market. So the
        # stop advances only when a new bar has closed, on that bar's range.
        # The stop is still *checked* against the last price continuously, which
        # is what a resting stop order does.
        if cfg.get("trail_on_close", True):
            if ss.get("live_trail_bar") != last_ts:
                bar_h = float(closed["High"].iloc[-1])
                bar_l = float(closed["Low"].iloc[-1])
                update_stop(pos, ctx.loc[last_ts], cfg["sl_cfg"], bar_h, bar_l)
                update_target(pos, ctx.loc[last_ts], cfg["tgt_cfg"])
                ss["live_trail_bar"] = last_ts
        else:
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

    # Is the last row genuinely still forming, or is the market closed and the
    # provider handing back a finished candle? Entering at the "next open" when
    # that open is already in the past is a backfilled fill, not a trade.
    dur = bar_duration(cfg["interval"])
    now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz is not None else pd.Timestamp.now()
    forming_is_live = now < (df.index[-1] + dur * 2)

    # Freshness is judged by bar position, not by clock time. Some providers
    # hand back tz-naive daily indices and tz-aware intraday ones, so comparing
    # a stored wall-clock against the bar stamp breaks on one of them. The first
    # poll after Start records the bar already on the screen as the baseline;
    # only bars that close after it can trigger an entry.
    if ss.live_on and ss.get("live_start_bar") is None:
        ss["live_start_bar"] = last_ts
    baseline = ss.get("live_start_bar")
    fresh_signal = baseline is not None and last_ts > baseline

    if ss.live_on and pos is None:
        s = int(dsig.loc[last_ts])
        blocked = None
        if s != 0 and ss.live_signal_bar == last_ts:
            blocked = "already acted on this bar"
        elif s != 0 and not fresh_signal:
            blocked = (f"the signal is on {last_ts:%d-%b %H:%M}, the bar that was already closed "
                       f"when you pressed Start. Waiting for a new one.")
        elif s != 0 and not forming_is_live:
            blocked = (f"the feed's latest bar ({df.index[-1]:%d-%b %H:%M}) is already complete, so "
                       f"the market looks closed. No entry on a stale bar.")
        if blocked:
            ss["live_block_reason"] = blocked
            ss.live_signal_bar = last_ts if "already" not in blocked else ss.live_signal_bar
        else:
            ss["live_block_reason"] = None
        if s != 0 and not blocked:
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

    warm = warmup_report(sig, ctx)
    if warm:
        st.warning("These indicators have no value on the latest bar, so the strategy cannot "
                   "signal yet: " + "; ".join(warm) + ". Their lookback is longer than the loaded "
                   "window — choose a longer period or a shorter lookback.", icon="⚠️")
    if not forming_is_live:
        st.info(f"The newest bar from the feed ({df.index[-1]:%d-%b %H:%M}) is already complete, so "
                f"the market appears closed or the feed is stale. Exits still track the last "
                f"traded price; new entries are held back.", icon="ℹ️")

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
                    width="stretch", config=PLOT_CFG,
                    key=f"livechart_{pd.Timestamp.now().value}")

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
    warm = warmup_report(sig)
    if warm:
        rows.append(("Not enough history for", "; ".join(warm)))
    block = st.session_state.get("live_block_reason")
    if pos is None:
        rows.append(("Waiting for", block or waiting_for(cfg, sig, dsig, al, as_, ts)))
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
        "Stop at exit": _round(pos.sl), "Target at entry": _round(pos.meta.get("target_at_entry")),
        "Target at exit": _round(pos.target), "Bar open": round(float(bar["Open"]), 4), "Bar high": round(float(bar["High"]), 4),
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


# -------------------------------------------------------------------- scanner
# Sweeps instruments, timeframes and exit parameters against the strategy set
# in the sidebar, then ranks what survives.
#
# Two things shape the design. First, every combination is run on the FULL
# loaded window using run_config, the same function the backtest tab calls, so
# a shortlisted row replays to the identical number rather than something
# close. Second, searching thousands of combinations and keeping the best is
# the most reliable way to fool yourself in this business: the winner is
# whatever fitted the noise best. So the scanner reports how many combinations
# were tried, states the bar a result has to clear to be distinguishable from
# luck, and re-runs the shortlist on held-out bars.

POINT_BANDS = {"5 – 10": (5, 10), "10 – 20": (10, 20), "20 – 30": (20, 30), "30 – 40": (30, 40),
               "40 – 50": (40, 50), "50 – 100": (50, 100), "100 – 150": (100, 150),
               "150 – 300": (150, 300)}


def to_yf_symbol(sym: str) -> str:
    """Bare NSE tickers get the .NS suffix; anything already qualified is left alone."""
    s = sym.strip()
    if s.startswith("^") or "." in s or "=" in s or s.upper().endswith("-USD"):
        return s
    return f"{s}.NS"


def band_grid(bands: list[str], per_band: int, extra: str = "") -> list[float]:
    vals: list[float] = []
    for b in bands:
        lo, hi = POINT_BANDS[b]
        vals += list(np.linspace(lo, hi, max(int(per_band), 1)))
    for tok in extra.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return sorted({round(float(v), 2) for v in vals})


def _swing_modes(modes: list[str]) -> list[str]:
    return [m for m in modes if "swing" in m or "candle" in m]


def stop_grid(modes, points, pcts, atr_lens, atr_mults, rrs, swings) -> list[dict]:
    out = []
    for m in modes:
        if m in ("Fixed points", "Trailing points"):
            out += [{"mode": m, "points": v, "be": {"on": False}} for v in points]
        elif m == "Fixed percent":
            out += [{"mode": m, "pct": v, "be": {"on": False}} for v in pcts]
        elif m == "ATR multiple":
            out += [{"mode": m, "atr_len": a, "atr_mult": k, "atr_trail": True, "be": {"on": False}}
                    for a in atr_lens for k in atr_mults]
        elif m == "Derived from reward (risk:reward)":
            out += [{"mode": m, "reward_points": v, "rr": r, "be": {"on": False}}
                    for v in points for r in rrs]
        elif m == "Strategy signal only (no price stop)":
            out.append({"mode": m, "be": {"on": False}})
        else:
            out += [{"mode": m, "swing_n": w, "buffer": 0.0, "be": {"on": False}} for w in swings]
    return out


def target_grid(modes, points, pcts, atr_lens, atr_mults, rrs, swings) -> list[dict]:
    out = []
    for m in modes:
        if m in ("Fixed points", "Trailing target (display only)"):
            out += [{"mode": m, "points": v} for v in points]
        elif m == "Fixed percent":
            out += [{"mode": m, "pct": v} for v in pcts]
        elif m == "ATR multiple":
            out += [{"mode": m, "atr_len": a, "atr_mult": k} for a in atr_lens for k in atr_mults]
        elif m == "Risk:reward multiple":
            out += [{"mode": m, "rr": r} for r in rrs]
        elif m == "Strategy reversal exit (no fixed target)":
            out.append({"mode": m})
        else:
            out += [{"mode": m, "swing_n": w} for w in swings]
    return out


def scale_cfg(cfg: dict, factor: float) -> dict:
    """Point distances tuned on an index do not transfer to a 200-rupee stock.
    Scaling by price level keeps the grid comparable across instruments."""
    if factor == 1.0:
        return cfg
    out = dict(cfg)
    for key in ("points", "reward_points"):
        if key in out:
            out[key] = round(out[key] * factor, 2)
    return out


def describe(cfg: dict) -> str:
    keys = [k for k in cfg if k not in ("mode", "be")]
    return ", ".join(f"{k}={cfg[k]}" for k in keys) or "—"


def selection_bar(n_combos: int, years: float) -> float:
    """The Sharpe a result must clear before it says anything.

    The best of N independent noise draws has an expected t-statistic near
    sqrt(2 ln N). Converting that back into annualised Sharpe over the sample
    length gives the level at which the top row is indistinguishable from the
    best of N coin flips.
    """
    if n_combos < 2 or years <= 0:
        return 0.0
    return float(np.sqrt(2 * np.log(n_combos)) / np.sqrt(years))


SCAN_COLS = ["Symbol", "Strategy", "Timeframe", "Period", "Filters", "Stop rule", "Stop settings",
             "Target rule", "Target settings", "Trades", "Hit rate %", "Net points",
             "Net P&L", "Profit factor", "Expectancy", "Sharpe", "Sortino", "Max drawdown",
             "Average R", "Reward:risk", "Average win", "Average loss", "Worst trade", "Flags",
             "Config"]


def degeneracy_flags(m: dict) -> str:
    """Catches results that look excellent for uninteresting reasons.

    A target far tighter than the stop manufactures a high hit rate while one
    normal loss erases many wins. Ranking on accuracy finds these every time,
    so they are labelled rather than left to look like the best row on screen.
    """
    flags = []
    rr = m.get("Planned reward:risk")
    if rr == rr and rr < 1.0:
        flags.append(f"reward:risk {rr:.2f}:1 — the target is tighter than the stop")
    if m["Losses"] == 0:
        flags.append("no losses yet — one loss reprices this")
    avg_win, avg_loss = m["Average win"], abs(m["Average loss"])
    if avg_loss > 0 and avg_win > 0 and avg_win / avg_loss < 0.5:
        flags.append(f"wins are {avg_win / avg_loss:.2f}x losses — accuracy is bought with bad "
                     f"reward:risk")
    if m["Trades"] < 30:
        flags.append("thin sample")
    return "; ".join(flags) or "—"


def render_search(base_cfg: dict):
    ss = st.session_state
    st.markdown("#### Scanner")
    smode = st.radio("Strategy", [f"Use the sidebar's ({base_cfg['strategy']})",
                                  "Sweep several strategies"], index=0, horizontal=True)
    if smode.startswith("Sweep"):
        sweep_strats = st.multiselect("Strategies to sweep", SWEEPABLE, default=SWEEPABLE,
                                      help="Each runs with its default parameters. Tune one "
                                           "precisely in the sidebar and backtest it directly.")
        strat_axis = [(n, STRATEGY_DEFAULTS[n]) for n in sweep_strats]
    else:
        strat_axis = [(base_cfg["strategy"], base_cfg["strat_params"])]

    c1, c2 = st.columns([1.2, 1])
    with c1:
        uni_name = st.selectbox("Universe", list(SCAN_UNIVERSES), index=0)
        default_syms = SCAN_UNIVERSES[uni_name]
        pasted = st.text_area(
            "Symbols" + (" — paste your own, one per line or comma separated" if default_syms
                         else " — paste your list, one per line or comma separated"),
            value=", ".join(default_syms[:200]),
            height=110,
            help="Bare NSE tickers get .NS added automatically. Indices keep their ^ prefix.")
        symbols = [t.strip() for t in pasted.replace("\n", ",").split(",") if t.strip()]
        limit = st.number_input("Test at most this many symbols", 1, 500,
                                min(25, max(len(symbols), 1)),
                                help="Each symbol needs its own download, rate limited to one "
                                     "every 0.3 seconds.")
        symbols = symbols[:int(limit)]
    with c2:
        tfs = st.multiselect("Timeframes", list(TF_PERIODS), default=["15m", "1h", "1d"])
        period_mode = st.radio("Periods", ["Default for each timeframe", "All allowed periods"],
                               index=0, horizontal=False)
        st.caption(f"{len(symbols)} symbols selected.")

    tf_pairs: list[tuple[str, str]] = []
    for tf in tfs:
        opts, default = TF_PERIODS[tf]
        tf_pairs += [(tf, p) for p in (opts if period_mode.startswith("All") else [default])]

    st.markdown("##### Exit grid")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Stop loss")
        sl_modes = st.multiselect("Stop rules", SL_MODES,
                                  default=["Fixed points", "ATR multiple"], key="sc_slm")
        sl_bands = st.multiselect("Stop point bands", list(POINT_BANDS),
                                  default=["5 – 10", "10 – 20", "30 – 40"], key="sc_slb")
        sl_per = st.slider("Values sampled per band", 1, 6, 2, key="sc_slp")
        sl_extra = st.text_input("Extra stop point values", "", key="sc_sle",
                                 placeholder="e.g. 7.5, 12, 200")
        sl_pts = band_grid(sl_bands, sl_per, sl_extra)
    with g2:
        st.caption("Target")
        tg_modes = st.multiselect("Target rules", TGT_MODES,
                                  default=["Risk:reward multiple"], key="sc_tgm",
                                  help="Risk:reward keeps the target tied to the stop, which is "
                                       "the only way the two stay comparable across instruments "
                                       "at different price levels.")
        tg_bands = st.multiselect("Target point bands", list(POINT_BANDS),
                                  default=["10 – 20", "30 – 40", "50 – 100"], key="sc_tgb")
        tg_per = st.slider("Values sampled per band", 1, 6, 2, key="sc_tgp")
        tg_extra = st.text_input("Extra target point values", "", key="sc_tge",
                                 placeholder="e.g. 25, 250")
        tg_pts = band_grid(tg_bands, tg_per, tg_extra)

    with st.expander("Percent, ATR, risk:reward and swing values"):
        e1, e2, e3, e4 = st.columns(4)
        pcts = [float(x) for x in e1.text_input("Percent values", "0.5, 1, 2, 3").replace(",", " ").split()]
        atr_lens = [int(float(x)) for x in e2.text_input("ATR lengths", "14").replace(",", " ").split()]
        atr_mults = [float(x) for x in e3.text_input("ATR multiples", "1.5, 2, 3").replace(",", " ").split()]
        rrs = [float(x) for x in e4.text_input("Risk:reward multiples", "1, 1.5, 2, 3").replace(",", " ").split()]
        swings = [int(float(x)) for x in st.text_input("Swing / candle lookbacks", "3, 5").replace(",", " ").split()]

    st.markdown("##### Entry filters")
    f1, f2 = st.columns([1.5, 1])
    scan_filters = f1.multiselect("Filters to try", list(FILTERS), default=[],
                                  help="Filter parameters use their defaults. Tune a specific "
                                       "filter in the sidebar and backtest it directly.")
    filter_mode = f2.radio("Try them", ["Off, and each one alone", "Off, and all together",
                                        "Every combination"], index=0)

    st.markdown("##### Method")
    m1, m2, m3, m4 = st.columns(4)
    rank_by = m1.selectbox("Rank by", ["Expectancy", "Sharpe", "Net points", "Profit factor",
                                       "Net P&L", "Hit rate %"], index=0,
                           help="Expectancy is the average outcome per trade and cannot be "
                                "inflated by a tiny target. Hit rate can.")
    min_trades = m2.number_input("Minimum trades", 1, 10_000, 20)
    min_rr = st.number_input("Minimum reward:risk at entry", 0.0, 20.0, 1.0, 0.1,
                             help="Drops setups whose target sits closer than their stop. Those "
                                  "produce spectacular hit rates and lose money after costs. Set "
                                  "to 0 to see them anyway.")
    scale_pts = m3.checkbox("Scale point distances by price", value=True,
                            help="A 10-point stop means something different on Nifty at 24,000 "
                                 "than on a 200-rupee stock, so point values are rescaled per "
                                 "symbol. ATR, percent and risk:reward rules are already "
                                 "price-relative and are left alone — mixing a scaled point "
                                 "target with an unscaled ATR stop skews reward:risk, so check "
                                 "the Flags column.")
    ref_price = m4.number_input("Reference price for scaling", 1.0, 1e6, 20000.0, step=100.0,
                                disabled=not scale_pts)
    oos_pct = st.slider("Hold out the last % of bars to re-test the shortlist", 0, 50, 30)
    max_combos = st.number_input("Stop after this many combinations", 100, 200_000, 20_000, step=100)

    sl_grid = stop_grid(sl_modes, sl_pts, pcts, atr_lens, atr_mults, rrs, swings)
    tg_grid = target_grid(tg_modes, tg_pts, pcts, atr_lens, atr_mults, rrs, swings)
    fsets = build_filter_sets(scan_filters, filter_mode)
    per_symbol = (max(len(tf_pairs), 1) * max(len(fsets), 1) * max(len(sl_grid), 1)
                  * max(len(tg_grid), 1) * max(len(strat_axis), 1))
    total = per_symbol * max(len(symbols), 1)

    st.markdown("---")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Combinations", f"{total:,}")
    e2.metric("Per symbol", f"{per_symbol:,}")
    e3.metric("Downloads", f"{len(symbols) * max(len(tf_pairs), 1):,}")
    e4.metric("Rough runtime", _runtime_estimate(total, len(symbols) * max(len(tf_pairs), 1)))
    if total > max_combos:
        st.warning(f"The grid is {total:,} combinations and the cap is {max_combos:,}. The scan "
                   f"will stop early. Narrow the grid or raise the cap.", icon="⚠️")
    if not symbols or not sl_grid or not tg_grid:
        st.info("Pick at least one symbol, one stop rule and one target rule.")
        return

    if st.button("Run scan", type="primary", width="stretch"):
        _execute_scan(base_cfg, symbols, tf_pairs, fsets, sl_grid, tg_grid,
                      dict(rank_by=rank_by, min_trades=int(min_trades), scale=scale_pts,
                           ref=ref_price, oos=oos_pct / 100.0, cap=int(max_combos),
                           min_rr=float(min_rr)), strat_axis)

    _render_scan_results(base_cfg)


def build_filter_sets(chosen: list[str], mode: str) -> list[tuple[str, dict]]:
    """Filters carry their default parameters here; the sidebar is the place to
    tune one precisely."""
    defaults = {name: _filter_defaults(name) for name in chosen}
    sets: list[tuple[str, dict]] = [("none", {})]
    if not chosen:
        return sets
    if mode.startswith("Off, and each"):
        sets += [(n, {n: defaults[n]}) for n in chosen]
    elif mode.startswith("Off, and all"):
        sets.append(("all: " + ", ".join(chosen), {n: defaults[n] for n in chosen}))
    else:
        import itertools as _it
        for r in range(1, len(chosen) + 1):
            for combo in _it.combinations(chosen, r):
                sets.append((", ".join(combo), {n: defaults[n] for n in combo}))
    return sets


def _filter_defaults(name: str) -> dict:
    return {
        "Bollinger band": {"n": 20, "m": 2.0, "where": "Above the basis / below the basis"},
        "EMA alignment": {"fast": 20, "slow": 50},
        "ADX band": {"n": 14, "a": 14, "min": 20, "max": 60, "di": True},
        "RSI": {"n": 14, "lmin": 50, "smax": 50},
        "Market regime": {"n": 200, "s": 20, "mode": "Price vs EMA"},
        "Volatility floor": {"n": 14, "min": 0.05},
        "Volume surge": {"n": 20, "x": 1.2},
        "Trading session": {"start": pd.Timestamp("09:20").time(),
                            "end": pd.Timestamp("15:00").time()},
    }[name]


def _runtime_estimate(total: int, downloads: int) -> str:
    secs = downloads * 0.8 + total * 0.012          # fetch time plus per-run cost
    if secs < 90:
        return f"~{secs:.0f}s"
    if secs < 5400:
        return f"~{secs / 60:.0f} min"
    return f"~{secs / 3600:.1f} hr"


def _execute_scan(base_cfg, symbols, tf_pairs, fsets, sl_grid, tg_grid, opt, strat_axis):
    ss = st.session_state
    rows, configs, frames, skipped = [], [], {}, []
    done = 0
    total = (len(symbols) * len(tf_pairs) * len(fsets) * len(sl_grid) * len(tg_grid)
             * max(len(strat_axis), 1))
    total = min(total, opt["cap"])
    bar = st.progress(0.0)
    status = st.empty()
    stop_early = False

    for sym in symbols:
        if stop_early:
            break
        ysym = to_yf_symbol(sym)
        for tf, period in tf_pairs:
            if stop_early:
                break
            status.write(f"Loading {sym} · {tf} / {period}  ·  {done:,} of {total:,} combinations")
            try:
                df = load_candles(ysym, tf, period)
            except RateLimitError as e:
                st.error(f"Stopped early: {e}")
                skipped.append((sym, tf, period, "rate limited — scan aborted"))
                stop_early = True
                break
            except Exception as e:
                skipped.append((sym, tf, period, str(e)[:120]))
                continue
            if len(df) < 60:
                skipped.append((sym, tf, period, f"only {len(df)} bars"))
                continue
            frames[(ysym, tf, period)] = df
            factor = (float(df["Close"].median()) / opt["ref"]) if opt["scale"] else 1.0

            for fname, fset in fsets:
              al, as_ = combine_filters(df, fset, "AND")
              for sname, sparams in strat_axis:
                cfg_base = dict(base_cfg, symbol=ysym, name=sym, interval=tf, period=period,
                                filters=fset, filter_logic="AND", strategy=sname,
                                strat_params=sparams)
                try:
                    sig = STRATEGIES[sname]["compute"](df, sparams)
                except Exception:
                    continue
                for slc in sl_grid:
                    slc_s = scale_cfg(slc, factor)
                    for tgc in tg_grid:
                        if done >= opt["cap"]:
                            stop_early = True
                            break
                        tgc_s = scale_cfg(tgc, factor)
                        cfg = dict(cfg_base, sl_cfg=slc_s, tgt_cfg=tgc_s,
                                   sl_mode=slc_s["mode"], tgt_mode=tgc_s["mode"])
                        try:
                            trades, equity = make_engine(cfg, df, sig, al, as_).run()
                        except Exception:
                            done += 1
                            continue
                        done += 1
                        if len(trades) < opt["min_trades"]:
                            continue
                        m = summarise(trades, equity)
                        rr = m.get("Planned reward:risk")
                        if opt.get("min_rr", 0) > 0 and rr == rr and rr < opt["min_rr"]:
                            continue
                        rows.append([sym, sname, tf, period, fname, slc_s["mode"], describe(slc_s),
                                     tgc_s["mode"], describe(tgc_s), m["Trades"], m["Hit rate %"],
                                     m["Net points per unit"], m["Net P&L"], m["Profit factor"],
                                     m["Expectancy per trade"], m["Sharpe (annualised)"],
                                     m["Sortino (annualised)"], m["Max drawdown"],
                                     m["Average R"], m.get("Planned reward:risk"),
                                     m["Average win"], m["Average loss"],
                                     m["Worst trade"], degeneracy_flags(m), len(configs)])
                        configs.append(cfg)
                        if done % 40 == 0:
                            bar.progress(min(done / max(total, 1), 1.0))
                            status.write(f"{sym} · {tf}/{period}  ·  {done:,} of {total:,} "
                                         f"combinations  ·  {len(rows):,} kept")
                    if stop_early:
                        break
                if stop_early:
                    break
    bar.progress(1.0)
    status.write(f"Finished. {done:,} combinations tested, {len(rows):,} met the minimum trade count.")
    ss["scan"] = {"rows": rows, "configs": configs, "skipped": skipped, "tested": done,
                  "opt": opt, "frames_keys": list(frames)}
    ss.setdefault("scan_frames", {}).update(frames)


def _render_scan_results(base_cfg):
    ss = st.session_state
    scan = ss.get("scan")
    if not scan:
        return
    st.markdown("---")
    rows, configs, opt = scan["rows"], scan["configs"], scan["opt"]
    if not rows:
        st.warning(f"{scan['tested']:,} combinations tested, none reached "
                   f"{opt['min_trades']} trades. Lower the minimum, widen the grid, or use a "
                   f"longer period.")
        _render_skipped(scan)
        return

    res = pd.DataFrame(rows, columns=SCAN_COLS)
    rank = opt["rank_by"]
    res = res.sort_values(rank, ascending=False, na_position="last").reset_index(drop=True)

    frames = ss.get("scan_frames", {})
    key = (to_yf_symbol(res.iloc[0]["Symbol"]), res.iloc[0]["Timeframe"], res.iloc[0]["Period"])
    years = 1.0
    if key in frames:
        idx = frames[key].index
        years = max((idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 3600), 1e-6)
    bar_level = selection_bar(scan["tested"], years)

    st.markdown("#### How much of this is luck")
    c1, c2, c3 = st.columns(3)
    c1.metric("Combinations tested", f"{scan['tested']:,}")
    c2.metric("Top Sharpe", _fmt_num(res.iloc[0]["Sharpe"], 3))
    c3.metric("Noise benchmark", _fmt_num(bar_level, 3))
    top_sharpe = res.iloc[0]["Sharpe"]
    if top_sharpe == top_sharpe and top_sharpe < bar_level:
        st.error(f"The best Sharpe here ({top_sharpe:.2f}) is below the level you would expect from "
                 f"the luckiest of {scan['tested']:,} coin flips over a {years:.2f} year sample "
                 f"({bar_level:.2f}). Treat every row as noise until it survives held-out bars.",
                 icon="⚠️")
    else:
        st.info(f"Searching {scan['tested']:,} combinations over {years:.2f} years means the best "
                f"of pure noise would score around {bar_level:.2f}. Rows above that are worth a "
                f"look; they are not yet evidence. The held-out column below is the real test.",
                icon="ℹ️")

    st.markdown("#### Best by symbol")
    best = res.loc[res.groupby("Symbol")[rank].idxmax()].sort_values(rank, ascending=False)
    st.dataframe(best.drop(columns=["Config"]).head(40), width="stretch", height=340,
                 hide_index=True)

    st.markdown("#### Every result that cleared the trade minimum")
    st.caption(f"{len(res):,} rows, ranked by {rank}. Read the Flags column before the metrics — "
               f"a 100% hit rate almost always means the target is far tighter than the stop, "
               f"which trades a good-looking record for a single ruinous loss.")
    flagged = int((res["Flags"] != "—").sum())
    if flagged:
        st.warning(f"{flagged:,} of {len(res):,} rows carry a flag. Sort by Average R or Sharpe "
                   f"rather than hit rate to avoid them.")
    st.dataframe(res.drop(columns=["Config"]).head(400), width="stretch", height=420,
                 hide_index=True)
    st.download_button("Download all results as CSV",
                       _export_frame(res, configs).to_csv(index=False).encode(),
                       "scan_results.csv", "text/csv")

    st.markdown("#### Shortlist re-tested on held-out bars")
    n_short = st.slider("How many top rows to re-test", 3, 40, 10, key="sc_short")
    if st.button("Re-test the shortlist", width="stretch"):
        _validate_shortlist(res.head(int(n_short)), configs, opt)
    val = ss.get("scan_validation")
    if val is not None:
        st.dataframe(val, width="stretch", height=380, hide_index=True)
        st.caption("Full is the number the backtest tab will reproduce exactly. Tuning and "
                   "Held out split the same window in two. A row whose held-out Sharpe collapses "
                   "was fitted to the sample, not to the market.")

    st.markdown("#### Send a configuration to the backtest tab")
    st.caption("The scanner and the backtest tab call the same engine on the same cached candles, "
               "so a replayed row reproduces its numbers exactly.")
    labels = [f"#{i}  ·  {r['Symbol']} {r['Strategy']} {r['Timeframe']}/{r['Period']}  ·  {r['Stop rule']} "
              f"({r['Stop settings']})  ·  {r['Target rule']} ({r['Target settings']})  ·  "
              f"{r['Filters']}  ·  Sharpe {r['Sharpe']}  ·  {r['Trades']} trades"
              for i, r in res.head(50).iterrows()]
    pick = st.selectbox("Configuration", labels, index=0, label_visibility="collapsed")
    b1, b2 = st.columns(2)
    if b1.button("Apply to the sidebar", type="primary", width="stretch"):
        row = res.iloc[labels.index(pick)]
        cfg = dict(configs[int(row["Config"])])
        cfg["_label"] = f"{row['Symbol']} · {row['Strategy']} · {row['Timeframe']}"
        ss["pending_sidebar"] = cfg
        ss.pop("replay_cfg", None)
        ss.pop("bt_result", None)
        st.rerun()
    if b2.button("Pin it to the backtest tab instead", width="stretch",
                 help="Runs exactly this configuration without touching your sidebar settings."):
        row = res.iloc[labels.index(pick)]
        ss["replay_cfg"] = configs[int(row["Config"])]
        ss.pop("bt_result", None)
        st.success("Pinned. Open the Backtest tab and press Run backtest.")

    _render_skipped(scan)


def _export_frame(res: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Results plus the full settings needed to reproduce each row by hand."""
    out = res.copy()
    out["Strategy"] = [configs[int(i)]["strategy"] for i in out["Config"]]
    out["Strategy settings"] = [str(configs[int(i)]["strat_params"]) for i in out["Config"]]
    out["Stop config"] = [str(configs[int(i)]["sl_cfg"]) for i in out["Config"]]
    out["Target config"] = [str(configs[int(i)]["tgt_cfg"]) for i in out["Config"]]
    out["Filter config"] = [str(configs[int(i)]["filters"]) for i in out["Config"]]
    out["Sides"] = [configs[int(i)]["side_mode"] for i in out["Config"]]
    out["Quantity"] = [configs[int(i)]["qty"] for i in out["Config"]]
    return out.drop(columns=["Config"])


def _validate_shortlist(short: pd.DataFrame, configs: list[dict], opt: dict):
    ss = st.session_state
    frames = ss.get("scan_frames", {})
    out = []
    for _, r in short.iterrows():
        cfg = configs[int(r["Config"])]
        key = (cfg["symbol"], cfg["interval"], cfg["period"])
        df = frames.get(key)
        if df is None:
            continue
        _, t_full, e_full = run_config(cfg, df)
        m_full = summarise(t_full, e_full)
        cut = int(len(df) * (1 - opt["oos"]))
        ins, oos = df.iloc[:cut], df.iloc[cut:]
        m_in = summarise(*run_config(cfg, ins)[1:]) if len(ins) > 60 else {"Trades": 0}
        m_out = summarise(*run_config(cfg, oos)[1:]) if len(oos) > 60 else {"Trades": 0}
        out.append({
            "Symbol": r["Symbol"], "Timeframe": r["Timeframe"], "Period": r["Period"],
            "Stop": f"{r['Stop rule']} ({r['Stop settings']})",
            "Target": f"{r['Target rule']} ({r['Target settings']})",
            "Filters": r["Filters"],
            "Full trades": m_full["Trades"], "Full Sharpe": m_full.get("Sharpe (annualised)"),
            "Full net points": m_full.get("Net points per unit"),
            "Full hit rate %": m_full.get("Hit rate %"),
            "Tuning trades": m_in.get("Trades", 0), "Tuning Sharpe": m_in.get("Sharpe (annualised)"),
            "Held-out trades": m_out.get("Trades", 0),
            "Held-out Sharpe": m_out.get("Sharpe (annualised)"),
            "Held-out net points": m_out.get("Net points per unit"),
            "Held-out hit rate %": m_out.get("Hit rate %"),
            "Survives": "yes" if (m_out.get("Trades", 0) >= 5 and
                                  (m_out.get("Sharpe (annualised)") or -9) > 0) else "no",
        })
    ss["scan_validation"] = pd.DataFrame(out)


def _render_skipped(scan: dict):
    if scan.get("skipped"):
        with st.expander(f"{len(scan['skipped'])} symbol/timeframe pairs were skipped"):
            st.dataframe(pd.DataFrame(scan["skipped"],
                                      columns=["Symbol", "Timeframe", "Period", "Reason"]),
                         width="stretch", hide_index=True)


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
# 15 · CHART PATTERN LIBRARY
# =============================================================================
# Every detector returns a boolean Series, True on the bar the pattern is
# CONFIRMED — the breakout bar, the engulfing bar, the retest bar. Nothing is
# marked before the information existed, so a hit on the last bar is genuinely
# actionable and a hit in history is genuinely measurable.
#
# Structure patterns are built from confirmed fractal pivots, which need `right`
# further bars before they can be known. That lag is deliberate: it is what
# stops the scanner from drawing a beautiful head and shoulders that nobody
# could have traded.
#
# Pattern names carry folklore about accuracy. This file does not repeat any of
# it. The screener measures forward returns against a same-symbol baseline so
# you can see what each pattern is actually worth on your instrument and
# timeframe, which is usually far less than the textbooks claim.

PatternHit = tuple[pd.Series, str]


def _body(df):
    o, c = df["Open"], df["Close"]
    body = (c - o).abs()
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    upper = df["High"] - np.maximum(o, c)
    lower = np.minimum(o, c) - df["Low"]
    return o, c, body, rng, upper, lower


def pivot_flags(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Fractal pivots: strict against the left, permissive against the right.

    Requiring the pivot bar to be the unique extreme of its window sounds
    correct and throws away most real pivots — two adjacent bars share a high
    constantly, especially on daily bars and coarse tick sizes, and a plateau
    then produces no pivot at all. Strict on the left and >= on the right keeps
    exactly one pivot per plateau, the first bar of it.
    """
    h, l = df["High"], df["Low"]
    left_max = h.shift(1).rolling(left, min_periods=left).max()
    right_max = h.shift(-right).rolling(right, min_periods=right).max()
    left_min = l.shift(1).rolling(left, min_periods=left).min()
    right_min = l.shift(-right).rolling(right, min_periods=right).min()
    is_h = (h > left_max) & (h >= right_max)
    is_l = (l < left_min) & (l <= right_min)
    return is_h.fillna(False), is_l.fillna(False)


def pivot_series(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Ordered pivot list as (bar_index, price, 'H'|'L', confirmed_index)."""
    is_h, is_l = pivot_flags(df, left, right)
    out = []
    hi, lo = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    for i in np.flatnonzero(is_h.to_numpy()):
        out.append((int(i), float(hi[i]), "H", int(i) + right))
    for i in np.flatnonzero(is_l.to_numpy()):
        out.append((int(i), float(lo[i]), "L", int(i) + right))
    out.sort(key=lambda t: t[0])
    return out


def _line(p1, p2):
    """Slope and intercept through two (x, y) points."""
    (x1, y1), (x2, y2) = p1, p2
    if x2 == x1:
        return 0.0, y1
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1


def _blank(df):
    return pd.Series(False, index=df.index)


# ------------------------------------------------------------ candlesticks
def pat_bullish_engulfing(df, cx):
    o, c, body, rng, up, lo = _body(df)
    prev_bear = c.shift(1) < o.shift(1)
    return prev_bear & (c > o) & (c >= o.shift(1)) & (o <= c.shift(1)) & (body > body.shift(1))


def pat_bearish_engulfing(df, cx):
    o, c, body, rng, up, lo = _body(df)
    prev_bull = c.shift(1) > o.shift(1)
    return prev_bull & (c < o) & (c <= o.shift(1)) & (o >= c.shift(1)) & (body > body.shift(1))


def pat_hammer(df, cx):
    o, c, body, rng, up, lo = _body(df)
    return (lo >= 2 * body) & (up <= 0.15 * rng) & (body / rng > 0.03) & cx["down"]


def pat_shooting_star(df, cx):
    o, c, body, rng, up, lo = _body(df)
    return (up >= 2 * body) & (lo <= 0.15 * rng) & (body / rng > 0.03) & cx["up"]


def pat_doji(df, cx):
    o, c, body, rng, up, lo = _body(df)
    return (body / rng < 0.1) & (rng / df["Close"] > 0.001)


def pat_morning_star(df, cx):
    o, c, body, rng, up, lo = _body(df)
    b1 = (c.shift(2) < o.shift(2)) & (body.shift(2) > body.shift(2).rolling(20).mean())
    b2 = body.shift(1) < body.shift(2) * 0.5
    b3 = (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)
    return (b1 & b2 & b3).fillna(False)


def pat_evening_star(df, cx):
    o, c, body, rng, up, lo = _body(df)
    b1 = (c.shift(2) > o.shift(2)) & (body.shift(2) > body.shift(2).rolling(20).mean())
    b2 = body.shift(1) < body.shift(2) * 0.5
    b3 = (c < o) & (c < (o.shift(2) + c.shift(2)) / 2)
    return (b1 & b2 & b3).fillna(False)


def pat_three_white_soldiers(df, cx):
    o, c, body, rng, up, lo = _body(df)
    bull = c > o
    return (bull & bull.shift(1) & bull.shift(2) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
            & (o < c.shift(1)) & (o.shift(1) < c.shift(2))).fillna(False)


def pat_three_black_crows(df, cx):
    o, c, body, rng, up, lo = _body(df)
    bear = c < o
    return (bear & bear.shift(1) & bear.shift(2) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
            & (o > c.shift(1)) & (o.shift(1) > c.shift(2))).fillna(False)


def pat_piercing_line(df, cx):
    o, c, body, rng, up, lo = _body(df)
    mid = (o.shift(1) + c.shift(1)) / 2
    return ((c.shift(1) < o.shift(1)) & (o < df["Low"].shift(1)) & (c > mid)
            & (c < o.shift(1))).fillna(False)


def pat_dark_cloud(df, cx):
    o, c, body, rng, up, lo = _body(df)
    mid = (o.shift(1) + c.shift(1)) / 2
    return ((c.shift(1) > o.shift(1)) & (o > df["High"].shift(1)) & (c < mid)
            & (c > o.shift(1))).fillna(False)


def pat_inside_bar(df, cx):
    return ((df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))).fillna(False)


def pat_nr7(df, cx):
    rng = df["High"] - df["Low"]
    return (rng == rng.rolling(7).min()).fillna(False)


def pat_bullish_marubozu(df, cx):
    o, c, body, rng, up, lo = _body(df)
    return ((c > o) & (up / rng < 0.05) & (lo / rng < 0.05) & (body / rng > 0.9)).fillna(False)


def pat_bearish_marubozu(df, cx):
    o, c, body, rng, up, lo = _body(df)
    return ((c < o) & (up / rng < 0.05) & (lo / rng < 0.05) & (body / rng > 0.9)).fillna(False)


def pat_tweezer_bottom(df, cx):
    tol = df["Close"] * 0.0015
    return (((df["Low"] - df["Low"].shift(1)).abs() <= tol) & cx["down"]
            & (df["Close"] > df["Open"])).fillna(False)


def pat_tweezer_top(df, cx):
    tol = df["Close"] * 0.0015
    return (((df["High"] - df["High"].shift(1)).abs() <= tol) & cx["up"]
            & (df["Close"] < df["Open"])).fillna(False)


def pat_gap_up(df, cx):
    return (df["Open"] > df["High"].shift(1)).fillna(False)


def pat_gap_down(df, cx):
    return (df["Open"] < df["Low"].shift(1)).fillna(False)


def pat_volume_breakout(df, cx):
    v = df["Volume"]
    rng = df["High"] - df["Low"]
    return ((v > 2 * pine_sma(v, 20)) & (rng > 1.5 * pine_sma(rng, 20))
            & (df["Close"] > df["Open"])).fillna(False)


def pat_window_high_break(df, cx):
    n = min(252, max(20, len(df) // 4))
    return (df["Close"] > pine_highest(df["High"], n).shift(1)).fillna(False)


def pat_window_low_break(df, cx):
    n = min(252, max(20, len(df) // 4))
    return (df["Close"] < pine_lowest(df["Low"], n).shift(1)).fillna(False)


# ------------------------------------------------- pivot structure patterns
def _pivot_pattern(df, cx, fn):
    """Walk confirmed pivots and let `fn` decide. Confirmation is recorded on
    the trigger bar, which is always at or after the last pivot's confirm bar."""
    out = _blank(df)
    piv = cx["pivots"]
    if len(piv) < 3:
        return out
    fn(df, piv, out)
    return out


def _tol(df, i, pct):
    return float(df["Close"].iloc[i]) * pct


def _trigger_after(df, start, level, direction, limit=40):
    """First bar after `start` whose close breaks `level`. Returns its index."""
    c = df["Close"].to_numpy(float)
    end = min(len(df), start + limit)
    for j in range(start, end):
        if (c[j] < level) if direction < 0 else (c[j] > level):
            return j
    return None


def _double(df, piv, out, kind):
    """kind 'top': two similar highs with a low between; break of that low."""
    want, other = ("H", "L") if kind == "top" else ("L", "H")
    sign = -1 if kind == "top" else 1
    for a in range(len(piv) - 2):
        p1 = piv[a]
        if p1[2] != want:
            continue
        mid = next((q for q in piv[a + 1:] if q[2] == other), None)
        if mid is None:
            continue
        p2 = next((q for q in piv[a + 1:] if q[2] == want and q[0] > mid[0]), None)
        if p2 is None:
            continue
        if abs(p1[1] - p2[1]) > _tol(df, p2[0], 0.015):
            continue
        depth = abs(p1[1] - mid[1])
        if depth < _tol(df, p2[0], 0.01):
            continue
        j = _trigger_after(df, p2[3], mid[1], sign)
        if j is not None:
            out.iloc[j] = True


def pat_double_top(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _double(d, p, o, "top"))


def pat_double_bottom(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _double(d, p, o, "bottom"))


def _triple(df, piv, out, kind):
    want = "H" if kind == "top" else "L"
    other = "L" if kind == "top" else "H"
    sign = -1 if kind == "top" else 1
    same = [q for q in piv if q[2] == want]
    for a in range(len(same) - 2):
        t1, t2, t3 = same[a], same[a + 1], same[a + 2]
        tol = _tol(df, t3[0], 0.02)
        if abs(t1[1] - t2[1]) > tol or abs(t2[1] - t3[1]) > tol:
            continue
        mids = [q[1] for q in piv if q[2] == other and t1[0] < q[0] < t3[0]]
        if not mids:
            continue
        neck = min(mids) if kind == "top" else max(mids)
        j = _trigger_after(df, t3[3], neck, sign)
        if j is not None:
            out.iloc[j] = True


def pat_triple_top(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _triple(d, p, o, "top"))


def pat_triple_bottom(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _triple(d, p, o, "bottom"))


def _hns(df, piv, out, inverse=False):
    want = "L" if inverse else "H"
    other = "H" if inverse else "L"
    sign = 1 if inverse else -1
    same = [q for q in piv if q[2] == want]
    for a in range(len(same) - 2):
        ls, head, rs = same[a], same[a + 1], same[a + 2]
        deeper = head[1] < ls[1] and head[1] < rs[1] if inverse else head[1] > ls[1] and head[1] > rs[1]
        if not deeper:
            continue
        if abs(ls[1] - rs[1]) > _tol(df, rs[0], 0.03):
            continue
        necks = [q for q in piv if q[2] == other and ls[0] < q[0] < rs[0]]
        if len(necks) < 2:
            continue
        m, b = _line((necks[0][0], necks[0][1]), (necks[-1][0], necks[-1][1]))
        c = df["Close"].to_numpy(float)
        for j in range(rs[3], min(len(df), rs[3] + 40)):
            lvl = m * j + b
            if (c[j] > lvl) if inverse else (c[j] < lvl):
                out.iloc[j] = True
                break


def pat_head_shoulders(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _hns(d, p, o, False))


def pat_inverse_head_shoulders(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _hns(d, p, o, True))


def _wedge_triangle(df, piv, out, want):
    """Fits the last three highs and three lows, then classifies the shape by
    the sign and convergence of the two slopes."""
    c = df["Close"].to_numpy(float)
    highs = [q for q in piv if q[2] == "H"]
    lows = [q for q in piv if q[2] == "L"]
    for a in range(max(len(highs), len(lows))):
        hs = highs[a:a + 3]
        ls = lows[a:a + 3]
        if len(hs) < 2 or len(ls) < 2:
            continue
        start = max(hs[-1][3], ls[-1][3])
        if start >= len(df):
            continue
        mh, bh = _line((hs[0][0], hs[0][1]), (hs[-1][0], hs[-1][1]))
        ml, bl = _line((ls[0][0], ls[0][1]), (ls[-1][0], ls[-1][1]))
        px = float(df["Close"].iloc[start])
        flat = abs(px) * 2e-4
        h_flat, l_flat = abs(mh) < flat, abs(ml) < flat
        shape = None
        if h_flat and ml > flat:
            shape = "ascending triangle"
        elif l_flat and mh < -flat:
            shape = "descending triangle"
        elif mh < -flat and ml > flat:
            shape = "symmetrical triangle"
        elif mh > flat and ml > flat and ml > mh:
            shape = "rising wedge"
        elif mh < -flat and ml < -flat and mh < ml:
            # converging: the upper line falls faster than the lower one
            shape = "falling wedge"
        elif h_flat and l_flat:
            shape = "rectangle"
        if shape != want:
            continue
        up = want in ("ascending triangle", "falling wedge")
        dn = want in ("descending triangle", "rising wedge")
        for j in range(start, min(len(df), start + 40)):
            res, sup = mh * j + bh, ml * j + bl
            if (up or want in ("symmetrical triangle", "rectangle")) and c[j] > res:
                out.iloc[j] = True
                break
            if (dn or want in ("symmetrical triangle", "rectangle")) and c[j] < sup:
                out.iloc[j] = True
                break


def _mk_shape(name):
    def f(df, cx):
        return _pivot_pattern(df, cx, lambda d, p, o: _wedge_triangle(d, p, o, name))
    return f


pat_ascending_triangle = _mk_shape("ascending triangle")
pat_descending_triangle = _mk_shape("descending triangle")
pat_symmetrical_triangle = _mk_shape("symmetrical triangle")
pat_rising_wedge = _mk_shape("rising wedge")
pat_falling_wedge = _mk_shape("falling wedge")
pat_rectangle = _mk_shape("rectangle")


# --------------------------------------------------------------- trendlines
def _trendline_events(df, piv, kind: str, retest: bool, out,
                      lookahead: int = 60, retest_within: int = 20):
    """Fit a line through two consecutive pivots, find the break, then the retest.

    'res' joins pivot highs and looks for a close above; 'sup' joins pivot lows
    and looks for a close below. The line must not already be violated between
    the two pivots that define it, otherwise it is not a line anyone was
    watching.

    A retest is not the pullback itself. Price has to return to the broken line,
    within a tolerance scaled by ATR so the same code works on a 24,000 index
    and a 200-rupee stock, and then close back on the breakout side. The bar
    that closes back through is the confirmation.
    """
    want = "H" if kind == "res" else "L"
    pts = [q for q in piv if q[2] == want]
    if len(pts) < 2:
        return
    c = df["Close"].to_numpy(float)
    hi = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float)
    atr = pine_atr(df, 14).to_numpy(float)
    n = len(df)
    fired = set()

    for a in range(len(pts) - 1):
        p1, p2 = pts[a], pts[a + 1]
        if p2[0] - p1[0] < 3:
            continue
        m, b = _line((p1[0], p1[1]), (p2[0], p2[1]))
        # the line must hold between its own two touches
        span = range(p1[0] + 1, p2[0])
        violated = any((c[j] > m * j + b + 1e-9) if kind == "res" else (c[j] < m * j + b - 1e-9)
                       for j in span)
        if violated:
            continue
        start = max(p2[3], p2[0] + 1)
        brk = None
        for j in range(start, min(n, start + lookahead)):
            lvl = m * j + b
            if (c[j] > lvl) if kind == "res" else (c[j] < lvl):
                brk = j
                break
        if brk is None or brk in fired:
            continue
        fired.add(brk)
        if not retest:
            out.iloc[brk] = True
            continue

        tol = max(atr[brk] if atr[brk] == atr[brk] else 0.0, c[brk] * 0.002)
        touched = False
        for j in range(brk + 1, min(n, brk + retest_within)):
            lvl = m * j + b
            if (c[j] < lvl - tol) if kind == "res" else (c[j] > lvl + tol):
                break                                   # the breakout failed outright
            near = (lo[j] <= lvl + tol) if kind == "res" else (hi[j] >= lvl - tol)
            if near and not touched:
                touched = True
                continue
            if touched and ((c[j] > lvl + tol * 0.25) if kind == "res"
                            else (c[j] < lvl - tol * 0.25)) \
                    and ((c[j] > c[j - 1]) if kind == "res" else (c[j] < c[j - 1])):
                out.iloc[j] = True
                break


def _mk_trend(kind, retest):
    def f(df, cx):
        out = _blank(df)
        piv = cx["pivots"]
        if len(piv) >= 3:
            _trendline_events(df, piv, kind, retest, out)
        return out
    return f


pat_trendline_breakout = _mk_trend("res", False)
pat_trendline_breakdown = _mk_trend("sup", False)
pat_trendline_breakout_retest = _mk_trend("res", True)
pat_trendline_breakdown_retest = _mk_trend("sup", True)


def _channel(df, piv, out, direction):
    """Parallel support and resistance with similar slopes; flags the break."""
    highs = [q for q in piv if q[2] == "H"]
    lows = [q for q in piv if q[2] == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return
    c = df["Close"].to_numpy(float)
    for a in range(min(len(highs), len(lows)) - 1):
        h1, h2 = highs[a], highs[a + 1]
        l1, l2 = lows[a], lows[a + 1]
        mh, bh = _line((h1[0], h1[1]), (h2[0], h2[1]))
        ml, bl = _line((l1[0], l1[1]), (l2[0], l2[1]))
        if abs(mh) < 1e-12 and abs(ml) < 1e-12:
            continue
        if abs(mh - ml) > abs(mh + ml) * 0.35:      # slopes must be comparable
            continue
        rising = mh > 0 and ml > 0
        falling = mh < 0 and ml < 0
        if (direction == "up" and not rising) or (direction == "down" and not falling):
            continue
        start = max(h2[3], l2[3])
        for j in range(start, min(len(df), start + 40)):
            if c[j] > mh * j + bh or c[j] < ml * j + bl:
                out.iloc[j] = True
                break


def pat_ascending_channel_break(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _channel(d, p, o, "up"))


def pat_descending_channel_break(df, cx):
    return _pivot_pattern(df, cx, lambda d, p, o: _channel(d, p, o, "down"))


# -------------------------------------------------------- flags and rounding
def _flag(df, cx, bullish: bool):
    """A sharp pole, then a shallow counter-sloping pause, then continuation."""
    out = _blank(df)
    c = df["Close"]
    n = len(df)
    pole, pause = 8, 10
    if n < pole + pause + 5:
        return out
    ret = c.pct_change(pole)
    atrp = pine_atr(df, 14) / c
    strong = (ret > 4 * atrp) if bullish else (ret < -4 * atrp)
    hh = df["High"].rolling(pause).max()
    ll = df["Low"].rolling(pause).min()
    for i in range(pole + pause, n):
        if not bool(strong.iloc[i - pause]):
            continue
        window = df.iloc[i - pause:i]
        drift = (window["Close"].iloc[-1] - window["Close"].iloc[0]) / window["Close"].iloc[0]
        tight = (window["High"].max() - window["Low"].min()) / window["Close"].mean()
        if tight > 0.12:
            continue
        if bullish and -0.06 < drift < 0.005 and c.iloc[i] > hh.iloc[i - 1]:
            out.iloc[i] = True
        if (not bullish) and -0.005 < drift < 0.06 and c.iloc[i] < ll.iloc[i - 1]:
            out.iloc[i] = True
    return out


def pat_bull_flag(df, cx):
    return _flag(df, cx, True)


def pat_bear_flag(df, cx):
    return _flag(df, cx, False)


def _rounding(df, cx, bottom: bool, win: int = 40):
    """Quadratic fit over the window; curvature sign gives the saucer, and the
    break of the rim confirms it."""
    out = _blank(df)
    n = len(df)
    if n < win + 5:
        return out
    src = df["Low"] if bottom else df["High"]
    y = src.to_numpy(float)
    c = df["Close"].to_numpy(float)
    x = np.arange(win, dtype=float)
    xm = x - x.mean()
    denom = (xm ** 2).sum()
    for i in range(win, n):
        seg = y[i - win:i]
        try:
            a2, a1, a0 = np.polyfit(x, seg, 2)
        except Exception:
            continue
        if bottom and a2 <= 0:
            continue
        if (not bottom) and a2 >= 0:
            continue
        curve = abs(a2) * win * win / max(abs(seg.mean()), 1e-9)
        if curve < 0.01:
            continue
        rim = max(seg[0], seg[-1]) if bottom else min(seg[0], seg[-1])
        if bottom and c[i] > rim and c[i - 1] <= rim:
            out.iloc[i] = True
        if (not bottom) and c[i] < rim and c[i - 1] >= rim:
            out.iloc[i] = True
    return out


def pat_rounding_bottom(df, cx):
    return _rounding(df, cx, True)


def pat_rounding_top(df, cx):
    return _rounding(df, cx, False)


def pat_cup_and_handle(df, cx):
    """A rounding bottom whose rim is retested by a shallow handle, then broken."""
    out = _blank(df)
    cup = _rounding(df, cx, True)
    c = df["Close"].to_numpy(float)
    n = len(df)
    for i in np.flatnonzero(cup.to_numpy()):
        rim = c[i]
        pulled = False
        for j in range(i + 1, min(n, i + 20)):
            if c[j] < rim * 0.99:
                pulled = True
            elif pulled and c[j] > rim:
                out.iloc[j] = True
                break
    return out


PATTERNS: dict[str, dict] = {
    # name: detector, direction (+1 bullish, -1 bearish, 0 neutral), family
    "Trendline breakout":            {"fn": pat_trendline_breakout, "dir": 1, "fam": "Trendline"},
    "Trendline breakout + retest":   {"fn": pat_trendline_breakout_retest, "dir": 1, "fam": "Trendline"},
    "Trendline breakdown":           {"fn": pat_trendline_breakdown, "dir": -1, "fam": "Trendline"},
    "Trendline breakdown + retest":  {"fn": pat_trendline_breakdown_retest, "dir": -1, "fam": "Trendline"},
    "Ascending channel break":       {"fn": pat_ascending_channel_break, "dir": 0, "fam": "Trendline"},
    "Descending channel break":      {"fn": pat_descending_channel_break, "dir": 0, "fam": "Trendline"},
    "Double bottom":                 {"fn": pat_double_bottom, "dir": 1, "fam": "Reversal"},
    "Double top":                    {"fn": pat_double_top, "dir": -1, "fam": "Reversal"},
    "Triple bottom":                 {"fn": pat_triple_bottom, "dir": 1, "fam": "Reversal"},
    "Triple top":                    {"fn": pat_triple_top, "dir": -1, "fam": "Reversal"},
    "Head and shoulders":            {"fn": pat_head_shoulders, "dir": -1, "fam": "Reversal"},
    "Inverse head and shoulders":    {"fn": pat_inverse_head_shoulders, "dir": 1, "fam": "Reversal"},
    "Rounding bottom":               {"fn": pat_rounding_bottom, "dir": 1, "fam": "Reversal"},
    "Rounding top":                  {"fn": pat_rounding_top, "dir": -1, "fam": "Reversal"},
    "Cup and handle":                {"fn": pat_cup_and_handle, "dir": 1, "fam": "Reversal"},
    "Ascending triangle":            {"fn": pat_ascending_triangle, "dir": 1, "fam": "Continuation"},
    "Descending triangle":           {"fn": pat_descending_triangle, "dir": -1, "fam": "Continuation"},
    "Symmetrical triangle":          {"fn": pat_symmetrical_triangle, "dir": 0, "fam": "Continuation"},
    "Rising wedge":                  {"fn": pat_rising_wedge, "dir": -1, "fam": "Continuation"},
    "Falling wedge":                 {"fn": pat_falling_wedge, "dir": 1, "fam": "Continuation"},
    "Rectangle range break":         {"fn": pat_rectangle, "dir": 0, "fam": "Continuation"},
    "Bull flag":                     {"fn": pat_bull_flag, "dir": 1, "fam": "Continuation"},
    "Bear flag":                     {"fn": pat_bear_flag, "dir": -1, "fam": "Continuation"},
    "Bullish engulfing":             {"fn": pat_bullish_engulfing, "dir": 1, "fam": "Candlestick"},
    "Bearish engulfing":             {"fn": pat_bearish_engulfing, "dir": -1, "fam": "Candlestick"},
    "Hammer":                        {"fn": pat_hammer, "dir": 1, "fam": "Candlestick"},
    "Shooting star":                 {"fn": pat_shooting_star, "dir": -1, "fam": "Candlestick"},
    "Morning star":                  {"fn": pat_morning_star, "dir": 1, "fam": "Candlestick"},
    "Evening star":                  {"fn": pat_evening_star, "dir": -1, "fam": "Candlestick"},
    "Piercing line":                 {"fn": pat_piercing_line, "dir": 1, "fam": "Candlestick"},
    "Dark cloud cover":              {"fn": pat_dark_cloud, "dir": -1, "fam": "Candlestick"},
    "Three white soldiers":          {"fn": pat_three_white_soldiers, "dir": 1, "fam": "Candlestick"},
    "Three black crows":             {"fn": pat_three_black_crows, "dir": -1, "fam": "Candlestick"},
    "Bullish marubozu":              {"fn": pat_bullish_marubozu, "dir": 1, "fam": "Candlestick"},
    "Bearish marubozu":              {"fn": pat_bearish_marubozu, "dir": -1, "fam": "Candlestick"},
    "Tweezer bottom":                {"fn": pat_tweezer_bottom, "dir": 1, "fam": "Candlestick"},
    "Tweezer top":                   {"fn": pat_tweezer_top, "dir": -1, "fam": "Candlestick"},
    "Doji":                          {"fn": pat_doji, "dir": 0, "fam": "Candlestick"},
    "Inside bar":                    {"fn": pat_inside_bar, "dir": 0, "fam": "Candlestick"},
    "NR7 range contraction":         {"fn": pat_nr7, "dir": 0, "fam": "Candlestick"},
    "Gap up":                        {"fn": pat_gap_up, "dir": 1, "fam": "Level"},
    "Gap down":                      {"fn": pat_gap_down, "dir": -1, "fam": "Level"},
    "Volume breakout":               {"fn": pat_volume_breakout, "dir": 1, "fam": "Level"},
    "Range high breakout":           {"fn": pat_window_high_break, "dir": 1, "fam": "Level"},
    "Range low breakdown":           {"fn": pat_window_low_break, "dir": -1, "fam": "Level"},
}
PATTERN_FAMILIES = sorted({v["fam"] for v in PATTERNS.values()})


def pattern_context(df: pd.DataFrame, left: int = 3, right: int = 3) -> dict:
    """Shared inputs so 45 detectors do not each recompute pivots and trend."""
    ema50 = pine_ema(df["Close"], min(50, max(5, len(df) // 4)))
    return {"pivots": pivot_series(df, left, right),
            "up": (df["Close"] > ema50).fillna(False),
            "down": (df["Close"] < ema50).fillna(False)}


def detect_patterns(df: pd.DataFrame, names: list[str], left: int = 3, right: int = 3):
    """Returns {name: boolean Series}. Failures are skipped, never faked."""
    cx = pattern_context(df, left, right)
    hits = {}
    for name in names:
        try:
            s = PATTERNS[name]["fn"](df, cx)
            hits[name] = s.fillna(False).astype(bool) if s is not None else _blank(df)
        except Exception:
            hits[name] = _blank(df)
    return hits


def pattern_edge(df: pd.DataFrame, hit: pd.Series, direction: int, horizon: int) -> dict:
    """Forward outcome after each occurrence, against a same-symbol baseline.

    The baseline is every bar in the same window measured the same way. Without
    it a 60% win rate means nothing, because in a rising market almost any long
    signal wins 60% of the time. What matters is the gap between the two.
    """
    c = df["Close"].to_numpy(float)
    n = len(c)
    idx = np.flatnonzero(hit.to_numpy())
    idx = idx[idx + horizon < n]
    d = direction if direction != 0 else 1
    if len(idx) == 0:
        return {"Occurrences": 0}
    fwd = (c[idx + horizon] - c[idx]) / c[idx] * 100.0 * d
    all_i = np.arange(0, n - horizon)
    base = (c[all_i + horizon] - c[all_i]) / c[all_i] * 100.0 * d
    highs, lows = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    mfe, mae = [], []
    for i in idx:
        seg_h, seg_l = highs[i + 1:i + 1 + horizon], lows[i + 1:i + 1 + horizon]
        if len(seg_h) == 0:
            continue
        if d > 0:
            mfe.append((seg_h.max() - c[i]) / c[i] * 100)
            mae.append((seg_l.min() - c[i]) / c[i] * 100)
        else:
            mfe.append((c[i] - seg_l.min()) / c[i] * 100)
            mae.append((c[i] - seg_h.max()) / c[i] * 100)
    return {
        "Occurrences": int(len(idx)),
        "Win rate %": round(float((fwd > 0).mean() * 100), 1),
        "Avg move %": round(float(fwd.mean()), 3),
        "Median move %": round(float(np.median(fwd)), 3),
        "Baseline win %": round(float((base > 0).mean() * 100), 1),
        "Baseline avg %": round(float(base.mean()), 3),
        "Edge vs baseline %": round(float(fwd.mean() - base.mean()), 3),
        "Avg best move %": round(float(np.mean(mfe)), 3) if mfe else np.nan,
        "Avg worst move %": round(float(np.mean(mae)), 3) if mae else np.nan,
    }


def pattern_trade_plan(df: pd.DataFrame, name: str, i: int, piv_n: int = 3,
                       rr: float = 2.0, atr_mult: float = 1.0) -> dict:
    """Concrete levels implied by a pattern occurrence.

    The stop comes from the structure that would invalidate the pattern — the
    low the double bottom is built on, the swing the breakout launched from,
    the extreme of the signal bar — because a level with a reason behind it is
    the only kind worth using. ATR is the fallback when no structure is nearby,
    and also the floor: a structural stop closer than a fraction of ATR is
    noise, and price will take it out on its way to anywhere.

    Two targets are returned. The measured move is the pattern's own projection
    where the geometry defines one. The R-multiple target is the generic
    alternative. Neither is a forecast.
    """
    spec = PATTERNS[name]
    d = spec["dir"] or 1
    entry = float(df["Close"].iloc[i])
    atr = float(pine_atr(df, 14).iloc[i]) if i < len(df) else np.nan
    if not (atr == atr) or atr <= 0:
        atr = max(entry * 0.005, 1e-6)
    piv = pivot_series(df, piv_n, piv_n)
    prior = [q for q in piv if q[0] <= i]
    lows = [q[1] for q in prior if q[2] == "L"][-3:]
    highs = [q[1] for q in prior if q[2] == "H"][-3:]

    fam = spec["fam"]
    if fam == "Candlestick":
        span = 3 if name in ("Morning star", "Evening star", "Three white soldiers",
                             "Three black crows") else 2
        lo = float(df["Low"].iloc[max(0, i - span + 1):i + 1].min())
        hi = float(df["High"].iloc[max(0, i - span + 1):i + 1].max())
        stop = lo if d > 0 else hi
        why = f"beyond the extreme of the {span} bar(s) forming the pattern"
    elif fam == "Level":
        stop = entry - d * atr_mult * atr
        why = f"{atr_mult:g} x ATR, since a level break leaves no nearby structure"
    else:
        ref = (min(lows) if lows else None) if d > 0 else (max(highs) if highs else None)
        if ref is None:
            stop = entry - d * atr_mult * atr
            why = f"{atr_mult:g} x ATR, no confirmed swing available yet"
        else:
            stop = ref
            why = "the last confirmed swing the pattern is built on"

    # a structural stop tighter than half an ATR is inside the noise
    if abs(entry - stop) < 0.5 * atr:
        stop = entry - d * 0.5 * atr
        why += ", widened to half an ATR because the structure sat inside the noise"
    risk = abs(entry - stop)

    measured, mnote = np.nan, ""
    if fam in ("Reversal", "Continuation", "Trendline") and lows and highs:
        height = abs(max(highs) - min(lows))
        if height > 0:
            measured = entry + d * height
            mnote = "pattern height projected from the breakout"
    return {"direction": "long" if d > 0 else "short", "entry": round(entry, 2),
            "stop": round(stop, 2), "risk": round(risk, 2),
            "target_rr": round(entry + d * rr * risk, 2),
            "target_measured": round(measured, 2) if measured == measured else np.nan,
            "rr": rr, "atr": round(atr, 2), "stop_reason": why, "measured_reason": mnote}


# ------------------------------------------------- patterns as a strategy
# Registering the library as a strategy makes every pattern backtestable,
# scannable and sendable to the sidebar through the machinery that already
# exists, instead of living in a screener that can only point at things.

def p_pattern(st_, k):
    fam = st_.selectbox("Family", ["All"] + PATTERN_FAMILIES, key=f"{k}_family")
    pool = [n for n, v in PATTERNS.items() if fam == "All" or v["fam"] == fam]
    pat = st_.selectbox("Pattern", pool, key=f"{k}_pattern")
    c1, c2 = st_.columns(2)
    piv = c1.number_input("Pivot sensitivity", 1, 20, 3, key=f"{k}_piv_n")
    bias = c2.selectbox("Trade it", ["Pattern's own bias", "Long only", "Short only"],
                        key=f"{k}_bias")
    return {"family": fam, "pattern": pat, "piv_n": piv, "bias": bias}


def c_pattern(df, p):
    name = p["pattern"]
    hits = detect_patterns(df, [name], int(p["piv_n"]), int(p["piv_n"]))[name]
    d = PATTERNS[name]["dir"]
    if p["bias"] == "Long only":
        d = 1
    elif p["bias"] == "Short only":
        d = -1
    elif d == 0:
        d = 1                       # neutral patterns need a side to be tradeable
    empty = pd.Series(False, index=df.index)
    sig = _sig(df.index, hits if d > 0 else empty, hits if d < 0 else empty)
    return Signals(sig, note=f"{name}: {int(hits.sum())} occurrences in this window.")


STRATEGIES["Chart pattern"] = {"params": p_pattern, "compute": c_pattern}
STRATEGY_DEFAULTS["Chart pattern"] = {"family": "Trendline",
                                      "pattern": "Trendline breakout + retest",
                                      "piv_n": 3, "bias": "Pattern's own bias"}


def pattern_geometry(df: pd.DataFrame, name: str, i: int, piv_n: int) -> dict:
    """Drawable shapes explaining why this bar was flagged.

    Re-derived from the same pivots the detector used, so what you see is what
    fired — not a decorative overlay drawn afterwards.
    """
    spec = PATTERNS[name]
    fam = spec["fam"]
    shapes = {"bars": [], "lines": [], "levels": [], "points": [], "note": ""}
    n = len(df)
    if fam == "Candlestick":
        span = {"Morning star": 3, "Evening star": 3, "Three white soldiers": 3,
                "Three black crows": 3}.get(name, 2 if "engulf" in name.lower()
                                            or name in ("Piercing line", "Dark cloud cover",
                                                        "Inside bar", "Tweezer top",
                                                        "Tweezer bottom", "Gap up", "Gap down")
                                            else 1)
        shapes["bars"] = list(range(max(0, i - span + 1), i + 1))
        shapes["note"] = f"The pattern completes on the highlighted bar; {span} bar(s) form it."
        return shapes
    if fam == "Level":
        shapes["bars"] = [i]
        win = min(252, max(20, n // 4))
        if "high" in name.lower():
            shapes["levels"] = [("Prior range high", float(df["High"].iloc[max(0, i - win):i].max()))]
        elif "low" in name.lower():
            shapes["levels"] = [("Prior range low", float(df["Low"].iloc[max(0, i - win):i].min()))]
        shapes["note"] = "Triggered by the close clearing the level shown."
        return shapes

    piv = pivot_series(df, piv_n, piv_n)
    near = [q for q in piv if q[0] <= i][-6:]
    shapes["points"] = [(q[0], q[1], q[2]) for q in near]
    shapes["bars"] = [i]
    if fam == "Trendline":
        want = "H" if ("breakout" in name or "Ascending" in name) else "L"
        pts = [q for q in piv if q[2] == want and q[0] < i][-2:]
        if len(pts) == 2:
            m, b = _line((pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]))
            x0, x1 = pts[0][0], min(n - 1, i + 5)
            shapes["lines"] = [("Trendline", x0, m * x0 + b, x1, m * x1 + b)]
        shapes["note"] = ("The line is fitted through the two marked pivots. The flagged bar is "
                          "where price closed back through it after the break."
                          if "retest" in name else
                          "The line is fitted through the two marked pivots; the flagged bar broke it.")
        return shapes
    if name in ("Double top", "Double bottom", "Triple top", "Triple bottom",
                "Head and shoulders", "Inverse head and shoulders"):
        want = "L" if ("bottom" in name or "Inverse" in name) else "H"
        other = "H" if want == "L" else "L"
        necks = [q[1] for q in piv if q[2] == other and q[0] < i][-2:]
        if necks:
            shapes["levels"] = [("Neckline", float(np.mean(necks)))]
        shapes["note"] = "Marked pivots form the shape; the flagged bar closed through the neckline."
        return shapes
    if fam == "Continuation":
        hs = [q for q in piv if q[2] == "H" and q[0] < i][-2:]
        ls = [q for q in piv if q[2] == "L" and q[0] < i][-2:]
        for pts, label in ((hs, "Upper boundary"), (ls, "Lower boundary")):
            if len(pts) == 2:
                m, b = _line((pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]))
                x0, x1 = pts[0][0], min(n - 1, i + 5)
                shapes["lines"].append((label, x0, m * x0 + b, x1, m * x1 + b))
        shapes["note"] = "Boundaries fitted through the marked pivots; the flagged bar broke out."
        return shapes
    shapes["note"] = "The flagged bar is where the pattern completed."
    return shapes


def pattern_chart(df: pd.DataFrame, name: str, i: int, piv_n: int, pad: int = 60) -> go.Figure:
    lo = max(0, i - pad)
    hi = min(len(df), i + max(15, pad // 3))
    d = df.iloc[lo:hi]
    g = pattern_geometry(df, name, i, piv_n)
    fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                   close=d["Close"], name="Price", line_width=1,
                                   increasing_line_color=UP, decreasing_line_color=DOWN,
                                   increasing_fillcolor=UP, decreasing_fillcolor=DOWN))
    for b in g["bars"]:
        if lo <= b < hi:
            fig.add_vrect(x0=df.index[max(b - 1, 0)], x1=df.index[min(b + 1, len(df) - 1)],
                          fillcolor="#f2a541", opacity=0.20, line_width=0)
    for label, x0, y0, x1, y1 in g["lines"]:
        if x1 < lo or x0 >= hi:
            continue
        fig.add_shape(type="line", x0=df.index[max(x0, 0)], y0=y0,
                      x1=df.index[min(x1, len(df) - 1)], y1=y1,
                      line=dict(color="#5b8dee", width=1.8, dash="solid"))
        fig.add_annotation(x=df.index[min(x1, len(df) - 1)], y=y1, text=label, showarrow=False,
                           font=dict(size=10, color="#5b8dee"), yshift=10)
    for label, y in g["levels"]:
        fig.add_hline(y=y, line=dict(color="#a06cd5", width=1.2, dash="dot"),
                      annotation_text=f"{label} {y:,.2f}", annotation_position="top left")
    px_ = [(df.index[x], y) for x, y, k in g["points"] if lo <= x < hi]
    if px_:
        fig.add_trace(go.Scatter(x=[a for a, _ in px_], y=[b for _, b in px_], mode="markers",
                                 name="Pivots",
                                 marker=dict(symbol="circle-open", size=10, color="#3fc1c9",
                                             line=dict(width=2))))
    if lo <= i < hi:
        fig.add_trace(go.Scatter(x=[df.index[i]], y=[float(df["Close"].iloc[i])], mode="markers+text",
                                 name=name, text=[name], textposition="top center",
                                 marker=dict(symbol="star", size=15, color="#f2a541"),
                                 textfont=dict(size=11, color="#f2a541")))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=460,
                      margin=dict(l=8, r=8, t=40, b=8), xaxis_rangeslider_visible=False,
                      font=dict(color=INK, size=11), showlegend=False,
                      title=dict(text=f"{name} — {df.index[i]:%d-%b-%Y %H:%M}", x=0,
                                 font=dict(size=14)))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig, g["note"]


def _pattern_sidebar_cfg(row: dict, piv_n: int, rr: float = 2.0, atr_mult: float = 1.5) -> dict:
    """A complete, runnable configuration built from a pattern hit."""
    name = row["Pattern"]
    d = PATTERNS[name]["dir"]
    bias = "Pattern's own bias" if d != 0 else "Long only"
    return {
        "symbol": to_yf_symbol(row["Symbol"]), "interval": row["Timeframe"],
        "period": TF_PERIODS[row["Timeframe"]][1], "source": "yfinance",
        "strategy": "Chart pattern",
        "strat_params": {"family": PATTERNS[name]["fam"], "pattern": name,
                         "piv_n": int(piv_n), "bias": bias},
        "sl_cfg": {"mode": "ATR multiple", "atr_len": 14, "atr_mult": atr_mult,
                   "atr_trail": False, "be": {"on": False}},
        "tgt_cfg": {"mode": "Risk:reward multiple", "rr": rr},
        "filters": {},
        "_label": f"{row['Symbol']} · {row['Timeframe']} · {name}",
    }


@st.dialog("Recommended levels", width="large")
def show_plan_dialog(row: dict, piv_n: int):
    sym = to_yf_symbol(row["Symbol"])
    tf = row["Timeframe"]
    try:
        df = load_candles(sym, tf, TF_PERIODS[tf][1])
    except Exception as e:
        st.error(f"Could not reload candles: {e}")
        return
    ts = pd.Timestamp(row["Fired at"])
    if df.index.tz is not None and ts.tz is None:
        ts = ts.tz_localize(df.index.tz)
    elif df.index.tz is None and ts.tz is not None:
        ts = ts.tz_localize(None)
    i = int(np.argmin(np.abs(df.index.values - np.datetime64(ts))))

    c1, c2 = st.columns(2)
    rr = c1.slider("Reward : risk", 0.5, 6.0, 2.0, 0.5, key="plan_rr")
    am = c2.slider("ATR multiple for the fallback stop", 0.5, 5.0, 1.5, 0.25, key="plan_atr")
    plan = pattern_trade_plan(df, row["Pattern"], i, piv_n, rr, am)

    st.markdown(f"##### {row['Pattern']} · {row['Symbol']} · {tf} · **{plan['direction']}**")
    k = st.columns(4)
    k[0].metric("Entry", f"{plan['entry']:,.2f}")
    k[1].metric("Stop", f"{plan['stop']:,.2f}", f"{plan['risk']:,.2f} risk")
    k[2].metric(f"Target at {rr:g}R", f"{plan['target_rr']:,.2f}")
    k[3].metric("Measured move", "—" if plan["target_measured"] != plan["target_measured"]
                else f"{plan['target_measured']:,.2f}")
    st.markdown(f"- Entry is the close of the bar that confirmed the pattern. In practice you fill "
                f"at the next bar's open, which is what the backtest assumes.\n"
                f"- Stop sits at **{plan['stop']:,.2f}**, {plan['stop_reason']}.\n"
                f"- ATR here is {plan['atr']:,.2f}, so the stop is "
                f"{plan['risk'] / max(plan['atr'], 1e-9):.2f} x ATR away."
                + (f"\n- Measured move target: {plan['measured_reason']}."
                   if plan["measured_reason"] else ""))

    st.markdown("##### What this exact rule has done here before")
    cfg_stub = _pattern_sidebar_cfg(row, piv_n, rr, am)
    full = dict(base_config_stub(), **cfg_stub)
    with st.spinner("Backtesting the plan over every past occurrence…"):
        try:
            _, trades, eq = run_config(full, df)
            m = summarise(trades, eq)
        except Exception as e:
            st.error(f"Could not evaluate: {e}")
            return
    if m.get("Trades", 0) == 0:
        st.info("This pattern has not produced a completed trade under these levels in the loaded "
                "window, so there is nothing to judge it on yet.")
    else:
        g = st.columns(5)
        g[0].metric("Trades", m["Trades"])
        g[1].metric("Hit rate", f"{m['Hit rate %']}%")
        g[2].metric("Expectancy", f"{m['Expectancy per trade']:,.2f}")
        g[3].metric("Profit factor", m["Profit factor"])
        g[4].metric("Net points", f"{m['Net points per unit']:,.2f}")
        st.caption(f"Same pattern, same stop and target rule, applied to every occurrence in this "
                   f"window. {degeneracy_flags(m)}")
    st.info("These are levels the pattern implies, not advice. The record above is the only reason "
            "to take them seriously, and it is a small sample on one instrument.", icon="ℹ️")
    if st.button("Send this whole setup to the sidebar", type="primary", width="stretch"):
        st.session_state["pending_sidebar"] = cfg_stub
        st.session_state.pop("replay_cfg", None)
        st.session_state.pop("bt_result", None)
        st.rerun()


def base_config_stub() -> dict:
    """Neutral defaults so a pattern plan can be run without the sidebar."""
    return {"source": "yfinance", "group": "Stocks", "name": "", "qty": 1, "side_mode": "Both",
            "instrument": "Equity intraday", "is_options": False, "flip": False,
            "filter_logic": "AND", "allow_reverse": False, "square_off": None,
            "forward": {"on": False, "days": 0}, "costs": CostModel(enabled=False),
            "refresh": 15, "api_delay": YF_MIN_DELAY, "trail_on_close": True,
            "dhan": {"on": False},
            "gmail": {"on": False}, "sl_mode": "", "tgt_mode": ""}


@st.dialog("Pattern detail", width="large")
def show_pattern_dialog(row: dict, piv_n: int, horizon: int):
    sym = to_yf_symbol(row["Symbol"])
    try:
        df = load_candles(sym, row["Timeframe"], TF_PERIODS[row["Timeframe"]][1])
    except Exception as e:
        st.error(f"Could not reload candles: {e}")
        return
    ts = pd.Timestamp(row["Fired at"])
    if df.index.tz is not None and ts.tz is None:
        ts = ts.tz_localize(df.index.tz)
    elif df.index.tz is None and ts.tz is not None:
        ts = ts.tz_localize(None)
    pos = int(np.argmin(np.abs(df.index.values - np.datetime64(ts))))
    fig, note = pattern_chart(df, row["Pattern"], pos, piv_n)
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    st.caption(note)
    st.markdown("##### Everything behind this result")
    detail = [("Ticker", f"{row['Symbol']} ({sym})"), ("Timeframe", row["Timeframe"]),
              ("Period loaded", TF_PERIODS[row["Timeframe"]][1]),
              ("Pattern", row["Pattern"]), ("Family", row["Family"]), ("Bias", row["Bias"]),
              ("Fired at", f"{ts:%d-%b-%Y %H:%M}"), ("Bars ago", row["Bars ago"]),
              ("Price when it fired", row["Price then"]), ("Price now", row["Price now"]),
              ("Pivot sensitivity", f"{piv_n} bars each side"),
              ("Forward horizon measured", f"{horizon} bars"),
              ("Past occurrences here", row["Past occurrences"]),
              ("Past win rate", f"{row['Past win %']}%"),
              ("Past average move", f"{row['Past avg move %']}%"),
              ("Edge over baseline", f"{row['Edge vs baseline %']}%"),
              ("Bars loaded", f"{len(df):,}")]
    st.dataframe(pd.DataFrame(detail, columns=["Setting", "Value"]).astype(str),
                 width="stretch", hide_index=True, height=430)
    st.caption("The pattern screener finds the setup. It does not set a stop or a target — send "
               "the ticker and timeframe to the sidebar, then choose exits and backtest them.")
    if st.button("Send this setup to the sidebar", type="primary", width="stretch"):
        st.session_state["pending_sidebar"] = _pattern_sidebar_cfg(row, piv_n)
        st.session_state.pop("replay_cfg", None)
        st.session_state.pop("bt_result", None)
        st.rerun()


# ---------------------------------------------------------- pattern screener
def render_patterns(base_cfg: dict):
    ss = st.session_state
    st.markdown("#### Pattern screener")
    st.caption("Scans instruments and timeframes for chart patterns, then measures what each one "
               "has actually been worth on that instrument rather than repeating textbook "
               "win rates.")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        uni = st.selectbox("Universe", list(SCAN_UNIVERSES), index=0, key="pt_uni")
        pasted = st.text_area("Symbols", value=", ".join(SCAN_UNIVERSES[uni][:200]), height=110,
                              key="pt_syms")
        symbols = [t.strip() for t in pasted.replace("\n", ",").split(",") if t.strip()]
        limit = st.number_input("Test at most this many symbols", 1, 500,
                                min(25, max(len(symbols), 1)), key="pt_lim")
        symbols = symbols[:int(limit)]
    with c2:
        tfs = st.multiselect("Timeframes", list(TF_PERIODS), default=["1h", "1d"], key="pt_tf")
        fams = st.multiselect("Pattern families", PATTERN_FAMILIES, default=PATTERN_FAMILIES,
                              key="pt_fam")
        bias = st.radio("Direction", ["Any", "Bullish only", "Bearish only"], index=0,
                        horizontal=True, key="pt_bias")

    pool = [n for n, v in PATTERNS.items() if v["fam"] in fams
            and (bias == "Any" or (bias.startswith("Bullish") and v["dir"] >= 0)
                 or (bias.startswith("Bearish") and v["dir"] <= 0))]
    chosen = st.multiselect("Patterns", pool, default=pool, key="pt_pats")

    m1, m2, m3, m4 = st.columns(4)
    recent = m1.number_input("Fired within the last N bars", 1, 200, 5, key="pt_recent",
                             help="A structural pattern needs its pivots confirmed, so the most "
                                  "recent bar is rarely the trigger bar.")
    horizon = m2.number_input("Measure the move over N bars", 1, 200, 10, key="pt_h")
    min_occ = m3.number_input("Minimum past occurrences", 1, 1000, 15, key="pt_occ")
    piv_n = m4.number_input("Pivot sensitivity (bars each side)", 1, 20, 3, key="pt_piv",
                            help="Smaller finds more, smaller patterns and more noise.")

    est = max(len(symbols), 1) * max(len(tfs), 1)
    e1, e2 = st.columns(2)
    e1.metric("Downloads", f"{est:,}")
    e2.metric("Detectors per download", f"{len(chosen):,}")
    if not symbols or not chosen:
        st.info("Pick at least one symbol and one pattern.")
        return

    if st.button("Run pattern scan", type="primary", width="stretch"):
        _execute_pattern_scan(symbols, tfs, chosen, int(recent), int(horizon), int(min_occ),
                              int(piv_n))
    _render_pattern_results()


def _execute_pattern_scan(symbols, tfs, chosen, recent, horizon, min_occ, piv_n):
    ss = st.session_state
    hits_rows, edge_rows, skipped = [], [], []
    bar = st.progress(0.0)
    status = st.empty()
    total = max(len(symbols) * len(tfs), 1)
    done = 0

    for sym in symbols:
        ysym = to_yf_symbol(sym)
        for tf in tfs:
            period = TF_PERIODS[tf][1]
            status.write(f"{sym} · {tf}  ({done}/{total})")
            except_rate = False
            try:
                df = load_candles(ysym, tf, period)
            except RateLimitError as e:
                st.error(f"Stopped early: {e}")
                skipped.append((sym, tf, "rate limited — scan aborted"))
                except_rate = True
            except Exception as e:
                skipped.append((sym, tf, str(e)[:110]))
                done += 1
                continue
            if except_rate:
                break
            if len(df) < 80:
                skipped.append((sym, tf, f"only {len(df)} bars"))
                done += 1
                continue
            hits = detect_patterns(df, chosen, piv_n, piv_n)
            last_ts = df.index[-1]
            for name in chosen:
                h = hits[name]
                d = PATTERNS[name]["dir"]
                edge = pattern_edge(df, h, d, horizon)
                if edge["Occurrences"] >= min_occ:
                    edge_rows.append({"Symbol": sym, "Timeframe": tf, "Pattern": name,
                                      "Family": PATTERNS[name]["fam"],
                                      "Bias": {1: "bullish", -1: "bearish", 0: "either"}[d],
                                      **{k: v for k, v in edge.items()}})
                fired = np.flatnonzero(h.to_numpy())
                fired = fired[fired >= len(df) - recent]
                for i in fired:
                    ts = df.index[int(i)]
                    hits_rows.append({
                        "Symbol": sym, "Timeframe": tf, "Pattern": name,
                        "Family": PATTERNS[name]["fam"],
                        "Bias": {1: "bullish", -1: "bearish", 0: "either"}[d],
                        "Fired at": ts, "Bars ago": int(len(df) - 1 - i),
                        "Price then": round(float(df["Close"].iloc[int(i)]), 2),
                        "Price now": round(float(df["Close"].iloc[-1]), 2),
                        "Past occurrences": edge.get("Occurrences", 0),
                        "Past win %": edge.get("Win rate %"),
                        "Past avg move %": edge.get("Avg move %"),
                        "Edge vs baseline %": edge.get("Edge vs baseline %"),
                    })
            done += 1
            bar.progress(min(done / total, 1.0))
    bar.progress(1.0)
    status.write(f"Finished. {len(hits_rows):,} recent hits, {len(edge_rows):,} measurable "
                 f"pattern histories.")
    ss["pat"] = {"hits": hits_rows, "edges": edge_rows, "skipped": skipped, "horizon": horizon}


def _render_pattern_results():
    ss = st.session_state
    res = ss.get("pat")
    if not res:
        return
    st.markdown("---")
    hits = pd.DataFrame(res["hits"])
    edges = pd.DataFrame(res["edges"])

    st.markdown("#### Fired recently")
    if hits.empty:
        st.info("Nothing fired in the window you asked for. Structural patterns need their pivots "
                "confirmed, so widen the bar count or lower the pivot sensitivity.")
    else:
        only_edge = st.checkbox("Show only patterns with a positive measured edge on that symbol",
                                value=True)
        view = hits.copy()
        if only_edge:
            view = view[view["Edge vs baseline %"].fillna(-9) > 0]
        view = view.sort_values(["Edge vs baseline %", "Bars ago"], ascending=[False, True])
        st.dataframe(view, width="stretch", height=420, hide_index=True)
        st.download_button("Download hits as CSV", hits.to_csv(index=False).encode(),
                           "pattern_hits.csv", "text/csv")

        if not view.empty:
            st.markdown("##### Act on a hit")
            st.caption("Each row carries its own buttons. Levels open the recommended entry, stop "
                       "and target with the historical record of that exact rule.")
            per_page = 10
            pages = int(np.ceil(len(view) / per_page))
            page = st.number_input(f"Page (1–{pages})", 1, max(pages, 1), 1, key="pt_page") if pages > 1 else 1
            chunk = view.iloc[(int(page) - 1) * per_page: int(page) * per_page]
            piv_n = int(ss.get("pt_piv", 3))
            horizon = int(res["horizon"])
            for pos, (_, r) in enumerate(chunk.iterrows()):
                row = r.to_dict()
                cols = st.columns([3.2, 1.1, 1.1, 1.1])
                edge = row.get("Edge vs baseline %")
                cols[0].markdown(
                    f"**{row['Symbol']}** · {row['Timeframe']} · {row['Pattern']}  \n"
                    f"<span style='opacity:.65;font-size:.82rem'>"
                    f"{pd.Timestamp(row['Fired at']):%d-%b %H:%M} · {row['Bars ago']} bars ago · "
                    f"{row['Bias']} · past {row['Past occurrences']} hits, win {row['Past win %']}%, "
                    f"edge {edge}%</span>", unsafe_allow_html=True)
                uid = f"{row['Symbol']}_{row['Timeframe']}_{row['Pattern']}_{pos}_{page}"
                if cols[1].button("Chart", key=f"pc_{uid}", width="stretch"):
                    show_pattern_dialog(row, piv_n, horizon)
                if cols[2].button("Levels", key=f"pl_{uid}", width="stretch"):
                    show_plan_dialog(row, piv_n)
                if cols[3].button("Sidebar", key=f"ps_{uid}", width="stretch"):
                    ss["pending_sidebar"] = _pattern_sidebar_cfg(row, piv_n)
                    ss.pop("replay_cfg", None)
                    ss.pop("bt_result", None)
                    st.rerun()

    st.markdown("#### What each pattern has been worth")
    if edges.empty:
        st.info("No pattern reached the minimum occurrence count.")
        return
    agg = (edges.groupby(["Pattern", "Family", "Bias"])
           .agg(**{"Symbols": ("Symbol", "nunique"),
                   "Total occurrences": ("Occurrences", "sum"),
                   "Mean win %": ("Win rate %", "mean"),
                   "Mean baseline %": ("Baseline win %", "mean"),
                   "Mean edge %": ("Edge vs baseline %", "mean"),
                   "Mean move %": ("Avg move %", "mean"),
                   "Mean best move %": ("Avg best move %", "mean"),
                   "Mean worst move %": ("Avg worst move %", "mean")})
           .round(3).reset_index().sort_values("Mean edge %", ascending=False))
    st.dataframe(agg, width="stretch", height=420, hide_index=True)
    pos = int((agg["Mean edge %"] > 0).sum())
    st.caption(f"{pos} of {len(agg)} patterns show a positive average edge over the baseline on "
               f"this universe. Expect roughly half to do so by chance alone, so treat the size of "
               f"the edge and the occurrence count as the signal, not the sign.")
    st.download_button("Download pattern statistics as CSV", edges.to_csv(index=False).encode(),
                       "pattern_edges.csv", "text/csv")
    if res.get("skipped"):
        with st.expander(f"{len(res['skipped'])} downloads were skipped"):
            st.dataframe(pd.DataFrame(res["skipped"], columns=["Symbol", "Timeframe", "Reason"]),
                         width="stretch", hide_index=True)


# ------------------------------------------------------------ edge screener
# Answers questions of the form "how often does this instrument travel more
# than X% from its anchor before the period ends" — the containment statistics
# behind range-selling, and their mirror image for buyers.

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion.

    A raw 95% from 20 expiries and a 95% from 200 are different claims. The
    normal approximation breaks down near 0 and 1, which is exactly where
    containment rates live, so Wilson is used instead.
    """
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, (centre - half) * 100), min(1.0, centre + half) * 100)


def expiry_anchors(idx: pd.DatetimeIndex, mode: str) -> pd.Series:
    """Group bars into periods, labelled by the period they belong to."""
    if mode == "Monthly expiry (last Thursday)":
        month = idx.to_period("M")
        last_thu = {}
        for m in month.unique():
            days = idx[month == m]
            thu = days[days.weekday == 3]
            last_thu[m] = thu[-1].normalize() if len(thu) else days[-1].normalize()
        return pd.Series([last_thu[m] for m in month], index=idx)
    if mode == "Weekly expiry (Thursday)":
        offset = (3 - idx.weekday) % 7
        return pd.Series(idx.normalize() + pd.to_timedelta(offset, unit="D"), index=idx)
    if mode == "Calendar month":
        return pd.Series(idx.to_period("M").to_timestamp("M"), index=idx)
    if mode == "Calendar week":
        return pd.Series(idx.to_period("W").to_timestamp("W"), index=idx)
    return pd.Series(idx.normalize(), index=idx)


def excursion_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per period: the anchor price and how far price travelled either way."""
    grp = expiry_anchors(df.index, mode)
    rows = []
    for key, seg in df.groupby(grp, sort=True):
        if len(seg) < 2:
            continue
        anchor = float(seg["Close"].iloc[0])
        if anchor <= 0:
            continue
        up = (float(seg["High"].max()) - anchor) / anchor * 100
        dn = (anchor - float(seg["Low"].min())) / anchor * 100
        rows.append({"Period ending": key, "Bars": len(seg), "Anchor": round(anchor, 2),
                     "Max up %": round(up, 3), "Max down %": round(dn, 3),
                     "Max either way %": round(max(up, dn), 3),
                     "Close move %": round((float(seg["Close"].iloc[-1]) - anchor) / anchor * 100, 3)})
    return pd.DataFrame(rows)


def containment_stats(exc: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """For each threshold, how often the move stayed inside it, with intervals."""
    n = len(exc)
    out = []
    for t in thresholds:
        for label, col in (("Either side", "Max either way %"), ("Upside only", "Max up %"),
                           ("Downside only", "Max down %")):
            k = int((exc[col] <= t).sum())
            lo, hi = wilson_interval(k, n)
            breached = exc.loc[exc[col] > t, col]
            out.append({"Threshold %": t, "Side": label, "Periods": n,
                        "Stayed inside": k, "Contained %": round(100 * k / n, 1) if n else 0.0,
                        "95% CI low": round(lo, 1), "95% CI high": round(hi, 1),
                        "Breached": n - k,
                        "Avg breach size %": round(float(breached.mean()), 2) if len(breached) else 0.0,
                        "Worst breach %": round(float(breached.max()), 2) if len(breached) else 0.0})
    return pd.DataFrame(out)


# ---------------------------------------------- generalised edge hypotheses
# A containment rule ("price rarely travels more than X% before expiry") is one
# member of a family: a repeatable condition, a binary outcome, and a rate you
# can compare against a baseline. This registry expresses that family so the
# same statistics apply to all of them.
#
# Each hypothesis returns (successes, trials, baseline_rate, description). The
# baseline is what you would get without the condition, which is what makes the
# result meaningful — a 62% win rate is worthless if the instrument closes up
# 62% of all days anyway.


def log_binom_pmf(k: int, n: int, p: float) -> float:
    from math import lgamma, log
    if p <= 0:
        return 0.0 if k == 0 else -np.inf
    if p >= 1:
        return 0.0 if k == n else -np.inf
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
            + k * log(p) + (n - k) * log(1 - p))


def binom_p_value(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial test, no SciPy dependency.

    Sums the probability of every outcome at least as unlikely as the observed
    one. This is the number that says whether a rate could plausibly have come
    from the baseline by chance.
    """
    if n == 0:
        return 1.0
    p = min(max(p, 1e-12), 1 - 1e-12)
    obs = log_binom_pmf(k, n, p)
    tot = 0.0
    for i in range(n + 1):
        lp = log_binom_pmf(i, n, p)
        if lp <= obs + 1e-9:
            tot += np.exp(lp)
    return float(min(1.0, tot))


def benjamini_hochberg(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Which findings survive once you account for how many you tested.

    Testing 500 hypotheses at p<0.05 hands you ~25 false winners by
    construction. Controlling the false discovery rate at q means that among
    everything flagged, roughly q of it is expected to be noise — a far more
    useful guarantee than pretending each test stood alone.
    """
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    keep = np.zeros(n, dtype=bool)
    thresh = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / n:
            thresh = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= thresh:
            keep[idx] = True
    return list(keep)


def _daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) and (df.index[-1] - df.index[-2]) < pd.Timedelta("1D"):
        return resample_ohlc(df, "1D")
    return df


def hyp_containment(df, cfg):
    out = []
    for mode in ("Monthly expiry (last Thursday)", "Weekly expiry (Thursday)"):
        exc = excursion_table(df, mode)
        if len(exc) < cfg["min_n"]:
            continue
        for t in cfg["thresholds"]:
            k = int((exc["Max either way %"] <= t).sum())
            # The null is the containment level you intend to trade, not 50%.
            # "Beats a coin flip" is a meaningless bar here — a wide enough band
            # contains everything, and the only question is whether the rate
            # reliably clears the level your position depends on.
            out.append((f"Stays within {t}% of the {mode.split(' (')[0].lower()} anchor",
                        k, len(exc), cfg["target"], "containment"))
    return out


def hyp_overnight(df, cfg):
    d = _daily_frame(df)
    if len(d) < cfg["min_n"] + 1:
        return []
    on = (d["Open"].shift(-1) - d["Close"]).dropna()
    intra = (d["Close"] - d["Open"])
    base = float((intra > 0).mean())
    k, n = int((on > 0).sum()), len(on)
    return [("Overnight gap is positive (close to next open)", k, n, base, "drift")]


def hyp_weekday(df, cfg):
    d = _daily_frame(df)
    r = d["Close"].pct_change().dropna()
    if len(r) < cfg["min_n"] * 5:
        return []
    base = float((r > 0).mean())
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    out = []
    for w in range(5):
        sub = r[r.index.weekday == w]
        if len(sub) >= cfg["min_n"]:
            out.append((f"{names[w]} closes up", int((sub > 0).sum()), len(sub), base, "seasonality"))
    return out


def hyp_month_turn(df, cfg):
    d = _daily_frame(df)
    r = d["Close"].pct_change().dropna()
    if len(r) < cfg["min_n"] * 3:
        return []
    base = float((r > 0).mean())
    dom = r.index.day
    eom = r.index.to_period("M").to_timestamp("M").day
    turn = (dom <= 3) | (dom >= eom - 1)
    sub = r[turn]
    return ([("Month-turn days close up (last 2 and first 3)", int((sub > 0).sum()), len(sub),
              base, "seasonality")] if len(sub) >= cfg["min_n"] else [])


def hyp_gap_fill(df, cfg):
    d = _daily_frame(df)
    if len(d) < cfg["min_n"] + 1:
        return []
    out = []
    up = d["Open"] > d["Close"].shift(1)
    dn = d["Open"] < d["Close"].shift(1)
    pc = d["Close"].shift(1)
    base = float(((d["Low"] <= pc) & (d["High"] >= pc)).mean())
    for mask, label, filled in ((up, "Gap up fills the same day", d["Low"] <= pc),
                                (dn, "Gap down fills the same day", d["High"] >= pc)):
        m = mask.fillna(False)
        if int(m.sum()) >= cfg["min_n"]:
            out.append((label, int((filled & m).sum()), int(m.sum()), base, "gap"))
    return out


def hyp_streak(df, cfg):
    d = _daily_frame(df)
    r = d["Close"].pct_change().dropna()
    if len(r) < cfg["min_n"] * 4:
        return []
    base = float((r > 0).mean())
    up = r > 0
    out = []
    for k_ in (2, 3):
        down_run = (~up)
        for j in range(1, k_):
            down_run = down_run & (~up).shift(j).fillna(False)
        nxt = r.shift(-1)[down_run].dropna()
        if len(nxt) >= cfg["min_n"]:
            out.append((f"Bounces after {k_} consecutive down days", int((nxt > 0).sum()),
                        len(nxt), base, "reversion"))
        up_run = up
        for j in range(1, k_):
            up_run = up_run & up.shift(j).fillna(False)
        nxt2 = r.shift(-1)[up_run].dropna()
        if len(nxt2) >= cfg["min_n"]:
            out.append((f"Continues after {k_} consecutive up days", int((nxt2 > 0).sum()),
                        len(nxt2), base, "momentum"))
    return out


def hyp_big_move_reversion(df, cfg):
    d = _daily_frame(df)
    r = d["Close"].pct_change().dropna() * 100
    if len(r) < cfg["min_n"] * 4:
        return []
    base = float((r > 0).mean())
    out = []
    for t in (1.5, 2.5):
        for sign, label in ((-1, "down"), (1, "up")):
            cond = (r <= -t) if sign < 0 else (r >= t)
            nxt = r.shift(-1)[cond].dropna()
            if len(nxt) >= cfg["min_n"]:
                out.append((f"Closes up the day after a {t}% {label} day",
                            int((nxt > 0).sum()), len(nxt), base, "reversion"))
    return out


def hyp_range_containment(df, cfg):
    d = _daily_frame(df)
    if len(d) < cfg["min_n"] + 20:
        return []
    a = pine_atr(d, 14)
    rng = d["High"] - d["Low"]
    ok = (rng <= a).dropna()
    return ([("Daily range stays inside 1 x ATR(14)", int(ok.sum()), len(ok), 0.5, "volatility")]
            if len(ok) >= cfg["min_n"] else [])


def hyp_open_extreme(df, cfg):
    d = _daily_frame(df)
    if len(d) < cfg["min_n"]:
        return []
    tol = (d["High"] - d["Low"]) * 0.05
    out = []
    low_open = (d["Open"] - d["Low"]) <= tol
    high_open = (d["High"] - d["Open"]) <= tol
    for m, label in ((low_open, "Opens at the day's low (trend-up day)"),
                     (high_open, "Opens at the day's high (trend-down day)")):
        out.append((label, int(m.sum()), len(d), 0.05, "structure"))
    return out


EDGE_HYPOTHESES = {
    "Range containment": hyp_containment,
    "Overnight drift": hyp_overnight,
    "Day of week": hyp_weekday,
    "Month turn": hyp_month_turn,
    "Gap fill": hyp_gap_fill,
    "Streaks": hyp_streak,
    "Reversion after a big day": hyp_big_move_reversion,
    "Open at the extreme": hyp_open_extreme,
}


def run_edge_hypotheses(df: pd.DataFrame, families: list[str], thresholds: list[float],
                        min_n: int, target: float = 0.95) -> list[dict]:
    cfg = {"thresholds": thresholds, "min_n": min_n, "target": target}
    rows = []
    for fam in families:
        try:
            for desc, k, n, base, kind in EDGE_HYPOTHESES[fam](df, cfg):
                if n < min_n:
                    continue
                rate = k / n
                lo, hi = wilson_interval(k, n)
                rows.append({"Family": fam, "Kind": kind, "Hypothesis": desc,
                             "Hits": k, "Sample": n, "Rate %": round(100 * rate, 1),
                             "95% CI low": round(lo, 1), "95% CI high": round(hi, 1),
                             "Baseline %": round(100 * base, 1),
                             "Edge points": round(100 * (rate - base), 1),
                             "p-value": binom_p_value(k, n, base)})
        except Exception:
            continue
    return rows


def render_edge_screener(base_cfg: dict):
    ss = st.session_state
    st.markdown("#### Level and expiry edge screener")
    st.caption("Measures how far an instrument actually travels from an anchor before the period "
               "ends. That is the statistic behind range-selling, and its mirror is what a buyer "
               "is paying for.")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        uni = st.selectbox("Universe", ["Indices and benchmarks", "Nifty 50", "Nifty 100",
                                        "Custom list"], index=0, key="ed_uni")
        default = SCAN_UNIVERSES.get(uni, []) or ["^NSEI"]
        pasted = st.text_area("Symbols", value=", ".join(default[:100]), height=90, key="ed_syms")
        symbols = [t.strip() for t in pasted.replace("\n", ",").split(",") if t.strip()][:60]
    with c2:
        mode = st.selectbox("Period anchored to", ["Monthly expiry (last Thursday)",
                                                   "Weekly expiry (Thursday)", "Calendar month",
                                                   "Calendar week", "Daily"], index=0, key="ed_mode")
        tf = st.selectbox("Bar size", ["1d", "1h", "15m"], index=0, key="ed_tf")
        period = st.selectbox("History", TF_PERIODS[tf][0],
                              index=len(TF_PERIODS[tf][0]) - 1, key="ed_period")
    raw = st.text_input("Thresholds to test (% from the anchor)", "1, 2, 3, 4, 5, 6, 7, 8, 10",
                        key="ed_thr")
    thresholds = sorted({round(float(x), 3) for x in raw.replace(",", " ").split() if x})
    target_conf = st.slider("Containment level you want to trade", 50, 99, 95, key="ed_conf")
    min_periods = st.number_input("Ignore instruments with fewer periods than", 3, 500, 24,
                                  key="ed_minp")

    if not symbols or not thresholds:
        st.info("Add at least one symbol and one threshold.")
        return
    if st.button("Run edge scan", type="primary", width="stretch"):
        _execute_edge_scan(symbols, tf, period, mode, thresholds, int(min_periods))
    _render_edge_results(target_conf, mode)
    render_edge_search(target_conf)


def _execute_edge_scan(symbols, tf, period, mode, thresholds, min_periods):
    ss = st.session_state
    all_rows, per_symbol, skipped = [], {}, []
    bar = st.progress(0.0)
    status = st.empty()
    for n_done, sym in enumerate(symbols, start=1):
        ysym = to_yf_symbol(sym)
        status.write(f"{sym}  ({n_done}/{len(symbols)})")
        try:
            df = load_candles(ysym, tf, period)
        except RateLimitError as e:
            st.error(f"Stopped early: {e}")
            skipped.append((sym, "rate limited — scan aborted"))
            break
        except Exception as e:
            skipped.append((sym, str(e)[:110]))
            continue
        exc = excursion_table(df, mode)
        if len(exc) < min_periods:
            skipped.append((sym, f"only {len(exc)} complete periods"))
            continue
        stats = containment_stats(exc, thresholds)
        stats.insert(0, "Symbol", sym)
        all_rows.append(stats)
        per_symbol[sym] = exc
        bar.progress(n_done / len(symbols))
    bar.progress(1.0)
    status.write(f"Finished. {len(all_rows)} instruments measured.")
    ss["edge"] = {"stats": pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(),
                  "exc": per_symbol, "skipped": skipped, "mode": mode, "tf": tf, "period": period}


def _render_edge_results(target_conf: int, mode: str):
    ss = st.session_state
    res = ss.get("edge")
    if not res:
        return
    stats = res["stats"]
    if stats.empty:
        st.warning("Nothing had enough complete periods. Load a longer history or use a shorter "
                   "anchor such as weekly.")
        return
    st.markdown("---")
    st.markdown(f"#### Distance needed for {target_conf}% containment")
    st.caption("For each instrument, the smallest tested threshold whose lower confidence bound "
               "still clears your level. The lower bound is the honest number: it is what the "
               "sample can support, not what it happened to produce.")
    picks = []
    for sym, g in stats[stats["Side"] == "Either side"].groupby("Symbol"):
        g = g.sort_values("Threshold %")
        ok = g[g["95% CI low"] >= target_conf]
        naive = g[g["Contained %"] >= target_conf]
        picks.append({
            "Symbol": sym, "Periods": int(g["Periods"].iloc[0]),
            f"Distance for {target_conf}% (raw)": naive["Threshold %"].min() if len(naive) else np.nan,
            f"Distance for {target_conf}% (confidence-adjusted)":
                ok["Threshold %"].min() if len(ok) else np.nan,
            "Median max move %": round(float(res["exc"][sym]["Max either way %"].median()), 2),
            "Worst period %": round(float(res["exc"][sym]["Max either way %"].max()), 2),
        })
    pick_df = pd.DataFrame(picks).sort_values("Symbol")
    st.dataframe(pick_df, width="stretch", hide_index=True, height=320)
    missing = int(pick_df[f"Distance for {target_conf}% (confidence-adjusted)"].isna().sum())
    if missing:
        st.warning(f"{missing} instrument(s) never reached {target_conf}% containment with "
                   f"confidence at any tested threshold. Widen the thresholds or accept a lower "
                   f"level.")

    st.markdown("#### Full containment table")
    syms = sorted(stats["Symbol"].unique())
    pick_sym = st.selectbox("Instrument", syms, index=0, key="ed_pick")
    sub = stats[stats["Symbol"] == pick_sym]
    st.dataframe(sub.drop(columns=["Symbol"]), width="stretch", hide_index=True, height=380)

    exc = res["exc"][pick_sym]
    st.markdown("#### Every period measured")
    st.dataframe(exc.sort_values("Period ending", ascending=False), width="stretch", height=300,
                 hide_index=True)
    st.download_button("Download all containment statistics as CSV",
                       stats.to_csv(index=False).encode(), "edge_containment.csv", "text/csv")

    e = sub[sub["Side"] == "Either side"].sort_values("Threshold %")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e["Threshold %"], y=e["Contained %"], mode="lines+markers",
                             name="Contained %", line=dict(color="#5b8dee", width=2)))
    fig.add_trace(go.Scatter(x=e["Threshold %"], y=e["95% CI low"], mode="lines",
                             name="95% lower bound", line=dict(color="#e05252", width=1, dash="dot")))
    fig.add_hline(y=target_conf, line=dict(color="#8fd14f", width=1, dash="dash"),
                  annotation_text=f"{target_conf}% target")
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=320,
                      margin=dict(l=8, r=8, t=40, b=8), font=dict(color=INK, size=11),
                      title=dict(text=f"{pick_sym} — containment by distance", x=0,
                                 font=dict(size=13)),
                      xaxis_title="Distance from anchor (%)", yaxis_title="Periods contained (%)")
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, range=[0, 101])
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    st.markdown("#### Before you trade this")
    worst = float(exc["Max either way %"].max())
    n_per = len(exc)
    st.markdown(
        f"- Containment is not profitability. Selling a {target_conf}% level wins often by "
        f"construction; whether it makes money depends on the premium you collect against the "
        f"size of the {100 - target_conf}% of periods that breach. This screener has no options "
        f"chain, so it cannot tell you that half.\n"
        f"- {pick_sym} has {n_per} complete periods here. The worst single period moved "
        f"{worst:.2f}%. A strategy that survives the average and not the worst is a strategy that "
        f"ends at the worst.\n"
        f"- Selling undefined-risk options exposes you to more than the premium, and Indian "
        f"exchanges raise margins exactly when volatility makes these levels break.\n"
        f"- Anchored to {mode.lower()} on {res['tf']} bars over {res['period']}. Volatility "
        f"regimes shift; a containment rate measured across a calm stretch will not hold through "
        f"a violent one.")
    if res.get("skipped"):
        with st.expander(f"{len(res['skipped'])} instruments skipped"):
            st.dataframe(pd.DataFrame(res["skipped"], columns=["Symbol", "Reason"]),
                         width="stretch", hide_index=True)


# Plain-language rendering. The tables are complete but unreadable unless you
# already know what a p-value is. Everything a table says is restated here in
# counts out of a hundred, because that is how a rate is actually understood.

HYP_PLAIN = {
    "Overnight gap is positive (close to next open)":
        ("hold overnight", "buy at today's close and sell at tomorrow's open"),
    "Bounces after 2 consecutive down days":
        ("wait for 2 red candles in a row", "buy at the close of the 2nd red candle, "
         "sell at the next candle's close"),
    "Bounces after 3 consecutive down days":
        ("wait for 3 red candles in a row", "buy at the close of the 3rd red candle, "
         "sell at the next candle's close"),
    "Continues after 2 consecutive up days":
        ("wait for 2 green candles in a row", "buy at the close of the 2nd green candle, "
         "sell at the next candle's close"),
    "Continues after 3 consecutive up days":
        ("wait for 3 green candles in a row", "buy at the close of the 3rd green candle, "
         "sell at the next candle's close"),
    "Gap up fills the same day":
        ("wait for the day to open above yesterday's close",
         "sell the open, expecting price to come back down to yesterday's close"),
    "Gap down fills the same day":
        ("wait for the day to open below yesterday's close",
         "buy the open, expecting price to come back up to yesterday's close"),
    "Month-turn days close up (last 2 and first 3)":
        ("trade only the last 2 and first 3 days of a month",
         "buy at the previous close, sell at that day's close"),
}


def _plain_condition(h: str) -> tuple[str, str]:
    if h in HYP_PLAIN:
        return HYP_PLAIN[h]
    if h.endswith("closes up"):
        day = h.split()[0]
        return (f"trade only on a {day}", f"buy at Thursday's close before it, "
                f"sell at {day}'s close")
    if h.startswith("Closes up the day after"):
        return (h[10:].replace("the day after", "wait for").strip(),
                "buy at that day's close, sell at the next day's close")
    if h.startswith("Opens at the day's"):
        return (h.lower(), "no trade rule attached; this describes how the day started")
    return (h.lower(), "buy at the signal candle's close, sell at the next candle's close")


def plain_edge(row: dict, q: float) -> str:
    """One paragraph anyone can read, saying exactly what the numbers mean."""
    rate = row["Rate %"]
    base = row["Baseline %"]
    n = int(row["Sample"])
    hits = int(row["Hits"])
    lo, hi = row["95% CI low"], row["95% CI high"]
    cond, how = _plain_condition(row["Hypothesis"])
    gain = rate - base

    if row["Kind"] == "containment":
        pct = row["Hypothesis"].split("within ")[1].split("%")[0]
        anchor = "month" if "monthly" in row["Hypothesis"] else "week"
        verdict = row["Verdict"]
        head = (f"**{row['Symbol']} — price stayed within {pct}% of where the {anchor} started.**")
        body = (f"It happened in {hits} of the last {n} {anchor}s, which is about "
                f"{rate:.0f} times out of every 100. Because you only have {n} {anchor}s to look "
                f"at, the real long-run figure is probably somewhere between {lo:.0f} and "
                f"{hi:.0f} out of 100 — it may be a good deal worse than it looks.")
        if verdict == "clears the target":
            tail = (f" You asked for a level that holds at least {row['Target']}% of the time, and "
                    f"even the pessimistic end of that range clears it. This distance is wide "
                    f"enough.")
        else:
            tail = (f" You asked for a level that holds at least {row['Target']}% of the time. The "
                    f"pessimistic end ({lo:.0f} out of 100) falls short, so this distance is not "
                    f"safe enough. Look at a wider one.")
        return head + " " + body + tail

    head = f"**{row['Symbol']} — if you {cond}:**"
    body = (f" the trade won {hits} times out of {n}, about {rate:.0f} out of every 100. "
            f"But this instrument goes your way {base:.0f} times out of 100 anyway, without "
            f"waiting for anything. ")
    if gain > 0:
        body += (f"So the condition is worth roughly **{gain:.0f} extra wins per 100 trades**. ")
    else:
        body += (f"So the condition actually costs you about {abs(gain):.0f} wins per 100 trades — "
                 f"it is worse than doing nothing. ")
    body += (f"With {n} examples, the true win rate could be anywhere from {lo:.0f} to {hi:.0f} "
             f"out of 100. ")
    if row["Verdict"] == "edge holds":
        body += (f"This one survived the luck check, meaning a run this good is unlikely to be a "
                 f"coincidence — though about {q * 100:.0f} in every 100 survivors are still "
                 f"expected to be flukes.")
    else:
        body += ("This did **not** survive the luck check: a streak this good turns up by chance "
                 "often enough that it proves nothing.")
    body += f" How you would trade it: {how}."
    return head + body


def plain_edge_caveat() -> str:
    return ("**Read this before acting on any of the above.** Each line measures one thing only: "
            "did price close higher, or did it stay inside a distance. There is no stop loss and "
            "no target in these numbers, and no brokerage or slippage. A rule that wins 60 times "
            "out of 100 on close-to-close can still lose money once a stop takes you out early "
            "and costs come off every trade. Treat a surviving line as a lead worth testing in "
            "the Backtest tab with real exits and costs — not as a result.")


def render_edge_search(target_conf: int):
    """The generalised version: many edge shapes, tested and corrected together."""
    ss = st.session_state
    st.markdown("---")
    st.markdown("#### Automated edge search")
    st.caption("Containment is one shape of edge. This tests a catalogue of them — overnight "
               "drift, weekday and month-turn bias, gap fill, streak reversal, reversion after a "
               "large day, opening structure — against the rate you would get without the "
               "condition, then corrects for the fact that testing many things guarantees "
               "winners.")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        uni = st.selectbox("Universe", list(SCAN_UNIVERSES), index=3, key="es_uni")
        pasted = st.text_area("Symbols", value=", ".join(SCAN_UNIVERSES[uni][:60] or ["^NSEI"]),
                              height=90, key="es_syms")
        symbols = [t.strip() for t in pasted.replace("\n", ",").split(",") if t.strip()][:60]
    with c2:
        fams = st.multiselect("Edge families", list(EDGE_HYPOTHESES),
                              default=list(EDGE_HYPOTHESES), key="es_fam")
        tf = st.selectbox("Bar size", ["1d", "1h"], index=0, key="es_tf")
        period = st.selectbox("History", TF_PERIODS[tf][0], index=len(TF_PERIODS[tf][0]) - 1,
                              key="es_period")
    m1, m2 = st.columns(2)
    min_n = m1.number_input("Minimum sample per hypothesis", 10, 2000, 40, key="es_minn")
    q = m2.slider("False discovery rate allowed (q)", 0.01, 0.25, 0.10, 0.01, key="es_q",
                  help="Of everything flagged, roughly this fraction is expected to be noise.")

    if not symbols or not fams:
        st.info("Pick at least one symbol and one family.")
        return
    if st.button("Search for edges", type="primary", width="stretch"):
        rows, skipped = [], []
        bar = st.progress(0.0)
        status = st.empty()
        thresholds = sorted({round(float(x), 3) for x in
                             (ss.get("ed_thr") or "1 2 3 4 5 6 7 8 10").replace(",", " ").split()})
        for n_done, sym in enumerate(symbols, start=1):
            status.write(f"{sym} ({n_done}/{len(symbols)})")
            try:
                df = load_candles(to_yf_symbol(sym), tf, period)
            except RateLimitError as e:
                st.error(f"Stopped early: {e}")
                skipped.append((sym, "rate limited — scan aborted"))
                break
            except Exception as e:
                skipped.append((sym, str(e)[:110]))
                continue
            for r in run_edge_hypotheses(df, fams, thresholds, int(min_n), target_conf / 100.0):
                r["Symbol"] = sym
                r["Target"] = target_conf
                rows.append(r)
            bar.progress(n_done / len(symbols))
        bar.progress(1.0)
        status.write(f"Finished. {len(rows):,} hypotheses tested across {len(symbols)} instruments.")
        ss["edge_search"] = {"rows": rows, "skipped": skipped, "q": q, "target": target_conf}

    res = ss.get("edge_search")
    if not res or not res["rows"]:
        return
    r = pd.DataFrame(res["rows"])
    r["Passes FDR"] = benjamini_hochberg(list(r["p-value"]), res["q"])

    def verdict(row):
        if row["Kind"] == "containment":
            return ("clears the target" if row["95% CI low"] >= res["target"]
                    else "below the target")
        if not row["Passes FDR"]:
            return "not distinguishable from the baseline"
        return "edge holds" if row["Edge points"] > 0 else "edge holds, but negative"
    r["Verdict"] = r.apply(verdict, axis=1)
    r["p-value"] = r["p-value"].map(lambda v: f"{v:.2e}")

    tested = len(r)
    held = int((r["Verdict"] == "edge holds").sum())
    cleared = int((r["Verdict"] == "clears the target").sum())
    k = st.columns(4)
    k[0].metric("Hypotheses tested", f"{tested:,}")
    k[1].metric("Directional edges that hold", f"{held:,}")
    k[2].metric(f"Levels clearing {res['target']}%", f"{cleared:,}")
    k[3].metric("Expected false among flagged", f"~{res['q'] * 100:.0f}%")

    st.markdown("##### In plain English")
    st.caption("The same findings written out. Everything is expressed as counts out of a hundred.")
    plain_rows = r[(r["Verdict"] == "edge holds") | (r["Verdict"] == "clears the target")]
    if plain_rows.empty:
        st.info("Nothing survived, so there is nothing to describe. In plain terms: none of the "
                "patterns tested did better than what this instrument does anyway, once you allow "
                "for how many things were tested. That is the normal outcome and it is a real "
                "answer — it means you have been saved from trading a coincidence.")
    else:
        show = plain_rows.sort_values("Edge points", ascending=False).head(12)
        for _, rr in show.iterrows():
            st.markdown("- " + plain_edge(rr.to_dict(), res["q"]))
        if len(plain_rows) > 12:
            st.caption(f"{len(plain_rows) - 12} more in the tables below.")
    st.info(plain_edge_caveat())

    st.markdown("##### Directional edges that survived correction")
    d = r[(r["Kind"] != "containment") & (r["Verdict"] == "edge holds")] \
        .sort_values("Edge points", ascending=False)
    if d.empty:
        st.info("Nothing directional survived. That is the usual and correct outcome — most "
                "apparent edges are the best of many tries. It is a real answer, not a failure.")
    else:
        st.dataframe(d[["Symbol", "Family", "Hypothesis", "Rate %", "Baseline %", "Edge points",
                        "95% CI low", "Sample", "p-value"]], width="stretch", height=340,
                     hide_index=True)
        st.caption("Rate is how often it happened under the condition; Baseline is how often it "
                   "happened anyway. Only the gap between them is the edge, and only the "
                   "confidence bound tells you whether the sample supports it.")

    st.markdown("##### Levels that clear your containment target")
    cdf = r[(r["Kind"] == "containment") & (r["Verdict"] == "clears the target")] \
        .sort_values(["Symbol", "Rate %"])
    if cdf.empty:
        st.info(f"No tested distance reached {res['target']}% containment with confidence. Widen "
                f"the thresholds in the section above or lower the target.")
    else:
        best = cdf.loc[cdf.groupby(["Symbol", "Hypothesis"])["Rate %"].idxmin()]
        st.dataframe(best[["Symbol", "Hypothesis", "Rate %", "95% CI low", "Sample"]],
                     width="stretch", height=300, hide_index=True)

    with st.expander("Everything tested, including what did not survive"):
        st.dataframe(r.sort_values("p-value")[["Symbol", "Family", "Hypothesis", "Rate %",
                                               "Baseline %", "Edge points", "Sample", "p-value",
                                               "Verdict"]], width="stretch", height=420,
                     hide_index=True)
    st.download_button("Download the full hypothesis table as CSV", r.to_csv(index=False).encode(),
                       "edge_search.csv", "text/csv")
    st.markdown(
        f"- Every row here is a statistical regularity, not a trading system. Turn one into a "
        f"strategy in the sidebar and backtest it with costs before believing it.\n"
        f"- Conditions with small samples move a lot between instruments; prefer an edge that "
        f"shows up across many symbols over a large edge on one.\n"
        f"- A surviving edge is a claim about the past. Regimes shift, and the widely known "
        f"seasonal effects have decayed as more people traded them.")


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


@st.cache_data(ttl=600, show_spinner="Loading candles…")
def load_candles(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Cached for ten minutes. A screener re-run over the same universe costs no
    requests at all, which is the cheapest rate-limit protection available."""
    return fetch_yf(symbol, interval, period)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="◧", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"## {APP_TITLE}")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    applied = consume_pending_sidebar()
    cfg = render_sidebar()
    if applied:
        st.success(f"Sidebar loaded with the configuration for {applied}. Every control is still "
                   f"editable.")
    tab_bt, tab_live, tab_hist, tab_scan, tab_pat, tab_edge = st.tabs(
        ["Backtest", "Live trading", "Trade history",
         "Screener · strategies", "Screener · patterns", "Screener · levels"])

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

    with tab_scan:
        render_search(cfg)

    with tab_live:
        render_live(cfg)

    with tab_hist:
        render_history()

    with tab_pat:
        render_patterns(cfg)

    with tab_edge:
        render_edge_screener(cfg)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.caption("Backtested results describe the past under assumptions you chose. They are not a "
               "forecast, and this tool is not financial advice.")


# Streamlit executes this script as __main__, so this both runs the app and keeps
# the file importable for testing.
if __name__ == "__main__":
    main()
