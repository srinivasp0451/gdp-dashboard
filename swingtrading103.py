# =============================================================================
#  QUANTEDGE PRO — Enterprise Algo Trading Platform (single-file Streamlit app)
#  Data: yfinance (default, rate-limited 0.3s/request) or Dhan HQ API v2
#  All indicators hand-written to match TradingView Pine v5 built-ins
#  (ta.ema / ta.rma / ta.rsi / ta.atr / ta.dmi / ta.supertrend / ta.vwap ...)
#  NO talib / pandas-ta / ta dependencies.
#
#  Run:  pip install streamlit plotly yfinance pandas numpy requests openpyxl
#        streamlit run app.py
# =============================================================================
from __future__ import annotations

import io
import json
import math
import os
import smtplib
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

APP_NAME = "QuantEdge Pro"
DATA_DIR = os.environ.get("QE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "algo_data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "live_trades.db")
YF_DELAY = 0.30          # mandatory gap between yfinance requests (seconds)
DHAN_DELAY = 0.30
DHAN_BASE = "https://api.dhan.co/v2"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
MIN_WARMUP_BARS = 300    # bars before the first tradable bar (EMA/RSI/ATR convergence)


def _wide_kwargs() -> dict:
    """Streamlit ≥1.46 uses width='stretch'; older versions use use_container_width=True."""
    try:
        import inspect
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


WIDE = _wide_kwargs()


def now_ist() -> datetime:
    return datetime.now(IST)


def fmt_ts(ts) -> str:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return "—"
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize(IST)
        return t.tz_convert(IST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def fnum(x, d=2) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"{float(x):,.{d}f}"
    except Exception:
        return str(x)


def fmt_duration(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# =============================================================================
#  INSTRUMENT UNIVERSE
# =============================================================================
TICKER_GROUPS: Dict[str, Dict[str, str]] = {
    "NIFTY 50": {"yf": "^NSEI", "dhan_id": "13", "dhan_seg": "IDX_I", "kind": "index", "und": "NIFTY", "fno_exch": "NSE"},
    "BANK NIFTY": {"yf": "^NSEBANK", "dhan_id": "25", "dhan_seg": "IDX_I", "kind": "index", "und": "BANKNIFTY", "fno_exch": "NSE"},
    "SENSEX": {"yf": "^BSESN", "dhan_id": "51", "dhan_seg": "IDX_I", "kind": "index", "und": "SENSEX", "fno_exch": "BSE"},
    "FIN NIFTY": {"yf": "NIFTY_FIN_SERVICE.NS", "dhan_id": "27", "dhan_seg": "IDX_I", "kind": "index", "und": "FINNIFTY", "fno_exch": "NSE"},
    "BTC": {"yf": "BTC-USD", "kind": "crypto"},
    "ETH": {"yf": "ETH-USD", "kind": "crypto"},
    "USDINR": {"yf": "INR=X", "kind": "forex"},
    "GOLD": {"yf": "GC=F", "kind": "commodity"},
    "SILVER": {"yf": "SI=F", "kind": "commodity"},
}
FOREX_PAIRS = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X",
               "USD/CHF": "CHF=X", "EUR/INR": "EURINR=X", "GBP/INR": "GBPINR=X"}
COMMODITIES = {"Crude Oil (WTI)": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F", "Copper": "HG=F",
               "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F"}
TICKER_MENU = ["NIFTY 50", "BANK NIFTY", "SENSEX", "FIN NIFTY", "BTC", "ETH", "USDINR", "GOLD", "SILVER",
               "Forex", "Commodities", "Custom Ticker", "Options Trading"]
DEFAULT_LOTS = {"NIFTY": 65, "BANKNIFTY": 30, "SENSEX": 20, "FINNIFTY": 60}

# timeframe -> allowed periods, default period
TF_PERIODS: Dict[str, Tuple[List[str], str]] = {
    "1m": (["1d", "5d", "7d"], "7d"),
    "3m": (["1d", "5d", "7d"], "7d"),
    "5m": (["1d", "5d", "7d", "1mo", "60d"], "60d"),
    "15m": (["1d", "5d", "7d", "1mo", "60d"], "60d"),
    "1h": (["1mo", "3mo", "6mo", "1y", "2y", "3y"], "6mo"),
    "4h": (["1mo", "3mo", "6mo", "1y", "2y", "3y"], "1y"),
    "1d": (["1y", "2y", "3y", "5y", "10y", "15y", "20y", "25y", "30y"], "2y"),
    "3d": (["1y", "2y", "3y", "5y", "10y", "15y", "20y", "25y", "30y"], "5y"),
    "1wk": (["1y", "2y", "3y", "5y", "10y", "15y", "20y", "25y", "30y"], "10y"),
    "1mon": (["1y", "2y", "3y", "5y", "10y", "15y", "20y", "25y", "30y"], "20y"),
}
TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440, "3d": 4320,
              "1wk": 10080, "1mon": 43200}
INTRADAY_TFS = {"1m", "3m", "5m", "15m", "1h", "4h"}


def is_indian(symbol: str, kind: str) -> bool:
    u = symbol.upper()
    return (kind == "index" or u.endswith((".NS", ".BO")) or
            u.startswith(("^NSE", "^BSE", "^CNX", "^NSMIDCP", "^CRSLDX", "^INDIAVIX", "NIFTY_")))


# =============================================================================
#  RATE LIMITER (shared across all users / sessions of this process)
# =============================================================================
class RateLimiter:
    def __init__(self, min_gap: float):
        self.min_gap = min_gap
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_gap:
                time.sleep(self.min_gap - gap)
            self._last = time.monotonic()


@st.cache_resource
def _limiters() -> Dict[str, RateLimiter]:
    return {"yf": RateLimiter(YF_DELAY), "dhan": RateLimiter(DHAN_DELAY), "dhan_oc": RateLimiter(3.05)}


# =============================================================================
#  DATA HELPERS
# =============================================================================
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=cols)
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)
    return df


def resample_ohlc(df: pd.DataFrame, tf: str, indian: bool) -> pd.DataFrame:
    """Resample to TradingView-aligned bars (NSE sessions anchored at 09:15 IST)."""
    if df.empty:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if tf in ("3m", "4h"):
        rule = "3min" if tf == "3m" else "240min"
        if indian:
            out = df.resample(rule, origin="start_day", offset="9h15min", label="left", closed="left").agg(agg)
        else:
            out = df.resample(rule, origin="start_day", label="left", closed="left").agg(agg)
    elif tf == "3d":
        g = np.arange(len(df)) // 3
        out = df.groupby(g).agg(agg)
        out.index = df.index[::3][: len(out)]
    elif tf == "1wk":
        key = df.index.tz_localize(None).to_period("W-SUN").start_time
        out = df.groupby(key).agg(agg)
        out.index = pd.DatetimeIndex(out.index).tz_localize(IST)
    elif tf == "1mon":
        key = df.index.tz_localize(None).to_period("M").start_time
        out = df.groupby(key).agg(agg)
        out.index = pd.DatetimeIndex(out.index).tz_localize(IST)
    else:
        return df
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def warmup_fetch_window(tf: str, period: str) -> Tuple[datetime, datetime]:
    """Fetch window = selected period + generous warm-up history (no N/A indicators after gaps)."""
    end = now_ist() + timedelta(days=1)
    if tf in ("1m", "3m"):
        start = now_ist() - timedelta(days=7)          # yfinance hard limit for 1m
    elif tf in ("5m", "15m"):
        start = now_ist() - timedelta(days=59)         # yfinance hard limit for <1h
    elif tf in ("1h", "4h"):
        start = now_ist() - timedelta(days=729)        # yfinance hard limit for 1h
    else:
        yrs = int(period.replace("y", "")) if period.endswith("y") else 2
        start = now_ist() - timedelta(days=int((yrs + 4) * 365.25))
    return start, end


def trim_to_period(df: pd.DataFrame, period: str) -> int:
    """Return index position where the user's selected period begins (but never before warm-up)."""
    if df.empty:
        return 0
    last = df.index[-1]
    if period.endswith("d") and period != "60d":
        n = int(period[:-1])
        sessions = pd.Index(df.index.date).unique()
        first_day = sessions[-n] if len(sessions) >= n else sessions[0]
        start_ts = pd.Timestamp(first_day).tz_localize(IST)
    elif period == "60d":
        start_ts = last - pd.Timedelta(days=60)
    elif period.endswith("mo"):
        start_ts = last - pd.DateOffset(months=int(period[:-2]))
    elif period.endswith("y"):
        start_ts = last - pd.DateOffset(years=int(period[:-1]))
    else:
        start_ts = df.index[0]
    pos = int(df.index.searchsorted(start_ts))
    return max(pos, min(MIN_WARMUP_BARS, max(len(df) - 50, 0)))


# ----------------------------------------------------------------------------- yfinance
def yf_history(symbol: str, interval: str, start: datetime = None, end: datetime = None,
               period: str = None, retries: int = 3) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed")
    lim = _limiters()["yf"]
    last_err = None
    for attempt in range(retries):
        lim.wait()  # 0.3 s mandatory gap between every request
        try:
            tk = yf.Ticker(symbol)
            if period:
                raw = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False, actions=False)
            else:
                raw = tk.history(start=start, end=end, interval=interval, auto_adjust=False, prepost=False, actions=False)
            if raw is not None and not raw.empty:
                return _normalize_ohlc(raw)
            last_err = RuntimeError("empty response")
        except Exception as e:  # rate limit / network – back off, never crash
            last_err = e
        time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed for {symbol} {interval}: {last_err}")


YF_INTERVAL = {"1m": "1m", "3m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h",
               "1d": "1d", "3d": "1d", "1wk": "1wk", "1mon": "1mo"}


def fetch_yf(symbol: str, tf: str, period: str, kind: str, live_tail: bool = False) -> pd.DataFrame:
    indian = is_indian(symbol, kind)
    base = YF_INTERVAL[tf]
    if live_tail:
        tail_period = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "3mo", "1wk": "1y", "1mo": "2y"}[base]
        df = yf_history(symbol, base, period=tail_period)
    else:
        s, e = warmup_fetch_window(tf, period)
        df = yf_history(symbol, base, start=s, end=e)
    if tf in ("3m", "4h", "3d"):
        df = resample_ohlc(df, tf, indian)
    return df


# ----------------------------------------------------------------------------- Dhan
class DhanClient:
    def __init__(self, client_id: str, token: str):
        self.client_id = (client_id or "").strip()
        self.token = (token or "").strip()

    @property
    def ok(self) -> bool:
        return bool(self.client_id and self.token)

    def _headers(self):
        return {"access-token": self.token, "client-id": self.client_id,
                "Content-Type": "application/json", "Accept": "application/json"}

    def _post(self, path: str, payload: dict, limiter: str = "dhan", timeout: int = 15) -> Any:
        _limiters()[limiter].wait()
        r = requests.post(DHAN_BASE + path, headers=self._headers(), data=json.dumps(payload), timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Dhan {path} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get(self, path: str, timeout: int = 15) -> Any:
        _limiters()["dhan"].wait()
        r = requests.get(DHAN_BASE + path, headers=self._headers(), timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Dhan {path} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    @staticmethod
    def _to_df(js: dict) -> pd.DataFrame:
        if not js or "timestamp" not in js or not js.get("timestamp"):
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame({"Open": js["open"], "High": js["high"], "Low": js["low"], "Close": js["close"],
                           "Volume": js.get("volume", [0] * len(js["open"]))},
                          index=pd.to_datetime(js["timestamp"], unit="s", utc=True))
        if "open_interest" in js and js["open_interest"]:
            df["OI"] = js["open_interest"]
        out = _normalize_ohlc(df[["Open", "High", "Low", "Close", "Volume"]])
        if "OI" in df.columns:
            out["OI"] = pd.Series(df["OI"].values, index=out.index[: len(df)]) if len(out) == len(df) else np.nan
        return out

    def intraday(self, sec_id: str, seg: str, instrument: str, interval: int, start: datetime, end: datetime,
                 oi: bool = False) -> pd.DataFrame:
        frames, cur = [], start
        while cur < end:
            nxt = min(cur + timedelta(days=88), end)
            js = self._post("/charts/intraday", {
                "securityId": str(sec_id), "exchangeSegment": seg, "instrument": instrument,
                "interval": str(interval), "oi": oi,
                "fromDate": cur.strftime("%Y-%m-%d %H:%M:%S"), "toDate": nxt.strftime("%Y-%m-%d %H:%M:%S")})
            frames.append(self._to_df(js))
            cur = nxt
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.concat(frames)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def daily(self, sec_id: str, seg: str, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
        js = self._post("/charts/historical", {
            "securityId": str(sec_id), "exchangeSegment": seg, "instrument": instrument, "expiryCode": 0,
            "oi": False, "fromDate": start.strftime("%Y-%m-%d"), "toDate": end.strftime("%Y-%m-%d")})
        df = self._to_df(js)
        if not df.empty:  # daily bars labelled at session date
            df.index = pd.DatetimeIndex([pd.Timestamp(d.date()).tz_localize(IST) for d in df.index])
        return df

    def ltp(self, seg: str, sec_id: str) -> Optional[float]:
        js = self._post("/marketfeed/ltp", {seg: [int(sec_id)]})
        try:
            return float(js["data"][seg][str(sec_id)]["last_price"])
        except Exception:
            return None

    def option_chain(self, und_id: str, und_seg: str, expiry: str) -> dict:
        return self._post("/optionchain", {"UnderlyingScrip": int(und_id), "UnderlyingSeg": und_seg,
                                           "Expiry": expiry}, limiter="dhan_oc")

    def expiry_list(self, und_id: str, und_seg: str) -> List[str]:
        js = self._post("/optionchain/expirylist", {"UnderlyingScrip": int(und_id), "UnderlyingSeg": und_seg},
                        limiter="dhan_oc")
        return js.get("data", [])

    def place_order(self, *, side: str, seg: str, product: str, order_type: str, sec_id: str, qty: int,
                    price: float = 0.0, tag: str = "") -> dict:
        payload = {"dhanClientId": self.client_id, "correlationId": (tag or uuid.uuid4().hex)[:25],
                   "transactionType": side, "exchangeSegment": seg, "productType": product,
                   "orderType": order_type, "validity": "DAY", "securityId": str(sec_id),
                   "quantity": int(qty), "disclosedQuantity": 0,
                   "price": round(float(price), 2) if order_type == "LIMIT" else 0,
                   "triggerPrice": 0, "afterMarketOrder": False}
        return self._post("/orders", payload)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_scrip_master() -> pd.DataFrame:
    r = requests.get(SCRIP_MASTER_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_equity_id(symbol: str) -> Tuple[Optional[str], str]:
    """KAYNES.NS -> (security_id, 'NSE_EQ')."""
    sym = symbol.upper().replace(".NS", "").replace(".BO", "")
    exch = "BSE" if symbol.upper().endswith(".BO") else "NSE"
    try:
        sm = load_scrip_master()
        m = sm[(sm["SEM_EXM_EXCH_ID"] == exch) & (sm["SEM_TRADING_SYMBOL"] == sym) &
               (sm["SEM_INSTRUMENT_NAME"] == "EQUITY")]
        if "SEM_SERIES" in sm.columns and (m["SEM_SERIES"] == "EQ").any():
            m = m[m["SEM_SERIES"] == "EQ"]
        if m.empty:
            return None, f"{exch}_EQ"
        return str(int(m.iloc[0]["SEM_SMST_SECURITY_ID"])), f"{exch}_EQ"
    except Exception:
        return None, f"{exch}_EQ"


DEFAULT_STRIKE_STEP = {"NIFTY": 50.0, "BANKNIFTY": 100.0, "SENSEX": 100.0, "FINNIFTY": 50.0}
STRIKE_CHOICES = ["ATM"] + [f"ITM {i}" for i in range(1, 11)] + [f"OTM {i}" for i in range(1, 11)] + ["Manual"]


def offset_steps(label: str) -> int:
    """'ATM'→0, 'OTM k'→+k, 'ITM k'→-k (direction applied per CE/PE in strike_for_label)."""
    if not label or label in ("ATM", "Manual"):
        return 0
    side, k = label.split()[:2]
    return int(k) * (1 if side == "OTM" else -1)


def _guess_stock_step(spot: float) -> float:
    for lim, stp in ((250, 2.5), (500, 5.0), (1000, 10.0), (2500, 20.0), (5000, 50.0), (10000, 100.0)):
        if spot < lim:
            return stp
    return 250.0


def option_strike_ladder(und: str, exch: str, spot: float, stock: bool = False) -> Dict[str, Any]:
    """Actual listed strikes of the nearest expiry (Dhan scrip master); fallback = standard strike step around spot."""
    out = {"strikes": None, "step": None, "expiry": None, "source": "standard step (scrip master unavailable)"}
    try:
        sm = load_scrip_master()
        today = pd.Timestamp(now_ist().date())
        inst = "OPTSTK" if stock else "OPTIDX"
        d = sm[(sm["SEM_EXM_EXCH_ID"] == exch) & (sm["SEM_INSTRUMENT_NAME"] == inst)]
        d = d[d["SEM_TRADING_SYMBOL"].astype(str).str.startswith(und + "-")].copy()
        d["exp"] = pd.to_datetime(d["SEM_EXPIRY_DATE"], errors="coerce").dt.normalize()
        d = d[d["exp"] >= today]
        if not d.empty:
            nearest = d["exp"].min()
            k = np.sort(d.loc[d["exp"] == nearest, "SEM_STRIKE_PRICE"].astype(float).unique())
            if len(k) > 1:
                diffs = np.diff(k)
                out.update(strikes=k, step=float(np.median(diffs[diffs > 0])), expiry=str(nearest.date()),
                           source="Dhan scrip master (nearest expiry)")
                return out
    except Exception:
        pass
    step = DEFAULT_STRIKE_STEP.get(und) or _guess_stock_step(spot)
    atm = round(spot / step) * step
    out.update(strikes=np.array([atm + i * step for i in range(-30, 31)]), step=step)
    return out


def strike_for_label(label: str, opt_type: str, strikes: np.ndarray, spot: float, manual: Optional[float] = None) -> float:
    """CE: OTM = higher strikes, ITM = lower. PE: OTM = lower strikes, ITM = higher."""
    strikes = np.asarray(strikes, float)
    if label == "Manual" and manual:
        return float(strikes[int(np.argmin(np.abs(strikes - manual)))])
    atm_i = int(np.argmin(np.abs(strikes - spot)))
    off = offset_steps(label)
    i = int(np.clip(atm_i + (off if opt_type == "CE" else -off), 0, len(strikes) - 1))
    return float(strikes[i])


def resolve_derivative(und: str, exch: str, kind: str, spot: float, opt_type: str = "CE",
                       strike_offset: int = 0, stock: bool = False, strike_value: Optional[float] = None) -> Dict[str, Any]:
    """Resolve nearest-expiry option (CE/PE @ ATM+offset steps, or an exact manual strike) or future security id."""
    out = {"security_id": None, "symbol": None, "strike": None, "expiry": None, "lot": None,
           "segment": f"{exch}_FNO", "error": None}
    try:
        sm = load_scrip_master()
        today = pd.Timestamp(now_ist().date())
        inst = ("OPTSTK" if stock else "OPTIDX") if kind == "option" else ("FUTSTK" if stock else "FUTIDX")
        d = sm[(sm["SEM_EXM_EXCH_ID"] == exch) & (sm["SEM_INSTRUMENT_NAME"] == inst)].copy()
        d = d[d["SEM_TRADING_SYMBOL"].astype(str).str.startswith(und + "-")]
        d["exp"] = pd.to_datetime(d["SEM_EXPIRY_DATE"], errors="coerce").dt.normalize()
        d = d[d["exp"] >= today]
        if d.empty:
            out["error"] = f"No {inst} contracts found for {und}"
            return out
        nearest = d["exp"].min()
        d = d[d["exp"] == nearest]
        if kind == "option":
            d = d[d["SEM_OPTION_TYPE"] == opt_type]
            strikes = np.sort(d["SEM_STRIKE_PRICE"].astype(float).unique())
            if len(strikes) == 0:
                out["error"] = "No strikes"
                return out
            if strike_value:
                i = int(np.argmin(np.abs(strikes - float(strike_value))))  # manual strike (nearest listed)
            else:
                atm_i = int(np.argmin(np.abs(strikes - spot)))
                # ITM for CE = lower strikes; for PE = higher strikes. offset>0 => OTM, <0 => ITM
                step = strike_offset if opt_type == "CE" else -strike_offset
                i = int(np.clip(atm_i + step, 0, len(strikes) - 1))
            row = d[d["SEM_STRIKE_PRICE"].astype(float) == strikes[i]].iloc[0]
            out["strike"] = float(strikes[i])
        else:
            row = d.iloc[0]
        out.update(security_id=str(int(row["SEM_SMST_SECURITY_ID"])), symbol=str(row.get("SEM_CUSTOM_SYMBOL", row["SEM_TRADING_SYMBOL"])),
                   expiry=str(nearest.date()), lot=int(float(row.get("SEM_LOT_UNITS", 1) or 1)))
    except Exception as e:
        out["error"] = str(e)
    return out


def fetch_dhan(dc: DhanClient, symbol: str, tf: str, period: str, meta: dict, live_tail: bool = False) -> pd.DataFrame:
    if meta.get("kind") == "index":
        sec_id, seg, instrument = meta["dhan_id"], "IDX_I", "INDEX"
    else:
        sec_id, seg = resolve_equity_id(symbol)
        instrument = "EQUITY"
        if not sec_id:
            raise RuntimeError(f"Could not resolve Dhan security id for {symbol}")
    if live_tail:
        start = datetime.combine(now_ist().date() - timedelta(days=4 if tf in INTRADAY_TFS else 120), datetime.min.time())
        end = now_ist().replace(tzinfo=None) + timedelta(minutes=1)
    else:
        s, e = warmup_fetch_window(tf, period)
        if tf in ("1h", "4h") and period.endswith("y"):
            s = now_ist() - timedelta(days=int(int(period[:-1]) * 365.25) + 200)
        start, end = s.replace(tzinfo=None), e.replace(tzinfo=None)
    if tf in INTRADAY_TFS:
        base = {"1m": 1, "3m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 60}[tf]
        df = dc.intraday(sec_id, seg, instrument, base, start, end)
        if tf in ("3m", "4h"):
            df = resample_ohlc(df, tf, True)
    else:
        df = dc.daily(sec_id, seg, instrument, start, end)
        if tf in ("3d", "1wk", "1mon"):
            df = resample_ohlc(df, tf, True)
    return df


# =============================================================================
#  INDICATORS — exact TradingView Pine v5 semantics, pure numpy/pandas
# =============================================================================
def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(src, n: int) -> np.ndarray:
    s = pd.Series(_arr(src))
    return s.rolling(n, min_periods=n).mean().to_numpy()


def _seeded_recursive(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Pine: sum := na(sum[1]) ? ta.sma(src, n) : alpha*src + (1-alpha)*nz(sum[1]).
    Seeded with SMA of the first n valid values; NaN inputs later keep the previous value (no N/A gaps)."""
    out = np.full(len(x), np.nan)
    valid = ~np.isnan(x)
    run, start = 0, -1
    for i in range(len(x)):
        run = run + 1 if valid[i] else 0
        if run >= n:
            start = i
            break
    if start < 0:
        return out
    prev = np.mean(x[start - n + 1:start + 1])
    out[start] = prev
    for i in range(start + 1, len(x)):
        v = x[i]
        if not np.isnan(v):
            prev = alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out


def ema(src, n: int) -> np.ndarray:
    return _seeded_recursive(_arr(src), n, 2.0 / (n + 1))


def rma(src, n: int) -> np.ndarray:
    return _seeded_recursive(_arr(src), n, 1.0 / n)


def wma(src, n: int) -> np.ndarray:
    x = _arr(src)
    w = np.arange(1, n + 1, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        win = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = win @ w / w.sum()
    return out


def hma(src, n: int) -> np.ndarray:
    half = wma(src, max(int(n / 2), 1))
    full = wma(src, n)
    return wma(2 * half - full, max(int(math.floor(math.sqrt(n))), 1))


def stdev(src, n: int) -> np.ndarray:  # Pine ta.stdev = population (biased) stdev
    return pd.Series(_arr(src)).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def highest(src, n: int) -> np.ndarray:
    return pd.Series(_arr(src)).rolling(n, min_periods=n).max().to_numpy()


def lowest(src, n: int) -> np.ndarray:
    return pd.Series(_arr(src)).rolling(n, min_periods=n).min().to_numpy()


def shift(x, k: int = 1) -> np.ndarray:
    x = _arr(x)
    out = np.full(len(x), np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def true_range(h, l, c) -> np.ndarray:
    h, l, c = _arr(h), _arr(l), _arr(c)
    pc = shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]  # ta.tr(true): na(close[1]) -> high-low
    return tr


def atr(h, l, c, n: int = 14) -> np.ndarray:
    return rma(true_range(h, l, c), n)


def rsi(src, n: int = 14) -> np.ndarray:
    x = _arr(src)
    ch = np.diff(x, prepend=np.nan)
    up = np.where(np.isnan(ch), np.nan, np.maximum(ch, 0))
    dn = np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0))
    ru, rd = rma(up, n), rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rd == 0, 100.0, np.where(ru == 0, 0.0, 100 - 100 / (1 + ru / rd)))
    out[np.isnan(ru) | np.isnan(rd)] = np.nan
    return out


def macd(src, fast=12, slow=26, sig=9):
    m = ema(src, fast) - ema(src, slow)
    s = ema(m, sig)
    return m, s, m - s


def bollinger(src, n=20, mult=2.0):
    b = sma(src, n)
    d = mult * stdev(src, n)
    return b, b + d, b - d


def dmi(h, l, c, di_len=14, adx_len=14):
    h, l, c = _arr(h), _arr(l), _arr(c)
    up = h - shift(h)
    dn = shift(l) - l
    plus_dm = np.where(np.isnan(up), np.nan, np.where((up > dn) & (up > 0), up, 0.0))
    minus_dm = np.where(np.isnan(dn), np.nan, np.where((dn > up) & (dn > 0), dn, 0.0))
    tr = true_range(h, l, c)
    tr[0] = np.nan  # ta.tr (handle_na=false) as used inside ta.dmi
    trur = rma(tr, di_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = pd.Series(100 * rma(plus_dm, di_len) / trur).ffill().to_numpy()
        minus = pd.Series(100 * rma(minus_dm, di_len) / trur).ffill().to_numpy()
        s = plus + minus
        adx_ = 100 * rma(np.abs(plus - minus) / np.where(s == 0, 1, s), adx_len)
    return plus, minus, adx_


def supertrend(h, l, c, factor=3.0, period=10):
    h, l, c = _arr(h), _arr(l), _arr(c)
    n = len(c)
    a = atr(h, l, c, period)
    src = (h + l) / 2
    ub_raw, lb_raw = src + factor * a, src - factor * a
    ub, lb = np.full(n, np.nan), np.full(n, np.nan)
    st_, d = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if np.isnan(a[i]):
            continue
        plb = lb[i - 1] if i > 0 and not np.isnan(lb[i - 1]) else 0.0
        pub = ub[i - 1] if i > 0 and not np.isnan(ub[i - 1]) else 0.0
        pc = c[i - 1] if i > 0 else np.nan
        lb[i] = lb_raw[i] if (lb_raw[i] > plb or (not np.isnan(pc) and pc < plb)) else plb
        ub[i] = ub_raw[i] if (ub_raw[i] < pub or (not np.isnan(pc) and pc > pub)) else pub
        if i == 0 or np.isnan(a[i - 1]):
            d[i] = 1
        elif st_[i - 1] == pub:
            d[i] = -1 if c[i] > ub[i] else 1
        else:
            d[i] = 1 if c[i] < lb[i] else -1
        st_[i] = lb[i] if d[i] == -1 else ub[i]
    return st_, d  # d == -1 -> uptrend (TradingView convention)


def vwap_session(df: pd.DataFrame) -> np.ndarray:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    v = df["Volume"].replace(0, np.nan)
    if v.isna().all():  # indices have no volume -> equal weight (TradingView shows na; we keep usable)
        v = pd.Series(1.0, index=df.index)
    v = v.fillna(0)
    day = pd.Index(df.index.date)
    pv = (tp * v).groupby(day).cumsum()
    vv = v.groupby(day).cumsum().replace(0, np.nan)
    return (pv / vv).ffill().to_numpy()


def stoch(c, h, l, k_len=14, k_smooth=3, d_smooth=3):
    hh, ll = highest(h, k_len), lowest(l, k_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100 * (_arr(c) - ll) / (hh - ll)
    k = sma(raw, k_smooth)
    return k, sma(k, d_smooth)


def cci(h, l, c, n=20):
    tp = (_arr(h) + _arr(l) + _arr(c)) / 3
    m = sma(tp, n)
    md = pd.Series(tp).rolling(n).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (tp - m) / (0.015 * md)


def williams_r(h, l, c, n=14):
    hh, ll = highest(h, n), lowest(l, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * (_arr(c) - hh) / (hh - ll)


def psar(h, l, start=0.02, inc=0.02, mx=0.2):
    """Port of Pine ta.sar."""
    h, l = _arr(h), _arr(l)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return out, np.zeros(n)
    trend = np.zeros(n)
    up = h[1] > h[0] or l[1] >= l[0]  # matches Pine's first-bar logic closely
    af = start
    ep = h[1] if up else l[1]
    sar = l[0] if up else h[0]
    out[1], trend[1] = sar, 1 if up else -1
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], start
            elif h[i] > ep:
                ep, af = h[i], min(af + inc, mx)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], start
            elif l[i] < ep:
                ep, af = l[i], min(af + inc, mx)
        out[i], trend[i] = sar, 1 if up else -1
    return out, trend


def keltner(h, l, c, n=20, mult=2.0):
    m = ema(c, n)
    r = ema(true_range(h, l, c), n)
    return m, m + mult * r, m - mult * r


def linreg(src, n: int) -> np.ndarray:  # ta.linreg(src, n, 0)
    x = _arr(src)
    out = np.full(len(x), np.nan)
    t = np.arange(n, dtype=float)
    tm = t.mean()
    den = ((t - tm) ** 2).sum()
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if np.isnan(w).any():
            continue
        b = ((t - tm) * (w - w.mean())).sum() / den
        out[i] = w.mean() + b * (n - 1 - tm)
    return out


def heikin_ashi(df: pd.DataFrame):
    o, h, l, c = (df[k].to_numpy(float) for k in ["Open", "High", "Low", "Close"])
    hc = (o + h + l + c) / 4
    ho = np.empty(len(c))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return ho, hc


def pivots(h, l, left=5, right=5):
    """ta.pivothigh / ta.pivotlow. Returns arrays (value at CONFIRMATION bar, NaN elsewhere) + pivot bar index."""
    h, l = _arr(h), _arr(l)
    n = len(h)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    phi, pli = np.full(n, -1), np.full(n, -1)
    for i in range(left + right, n):
        c = i - right
        wh = h[c - left:i + 1]
        if h[c] == np.max(wh) and np.sum(wh == h[c]) == 1:
            ph[i], phi[i] = h[c], c
        wl = l[c - left:i + 1]
        if l[c] == np.min(wl) and np.sum(wl == l[c]) == 1:
            pl[i], pli[i] = l[c], c
    return ph, pl, phi, pli


def last_two_swings(piv_vals: np.ndarray):
    """For every bar: most recent confirmed swing value and the one before (forward-filled)."""
    n = len(piv_vals)
    cur, prev = np.full(n, np.nan), np.full(n, np.nan)
    c, p = np.nan, np.nan
    for i in range(n):
        if not np.isnan(piv_vals[i]):
            p, c = c, piv_vals[i]
        cur[i], prev[i] = c, p
    return cur, prev


def ema_angle(ma: np.ndarray, atr_arr: np.ndarray) -> np.ndarray:
    """Popular TradingView angle formula: atan((ma - ma[1]) / atr) in degrees (absolute value)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.degrees(np.arctan((ma - shift(ma)) / atr_arr)))


def cross_over(a, b) -> np.ndarray:
    a, b = _arr(a), _arr(b)
    pa, pb = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a > b) & (pa <= pb)


def cross_under(a, b) -> np.ndarray:
    a, b = _arr(a), _arr(b)
    pa, pb = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a < b) & (pa >= pb)


def daily_sma_map(df: pd.DataFrame, daily: pd.DataFrame, n: int) -> np.ndarray:
    """Higher-TF daily SMA mapped to each bar using the PREVIOUS completed day (no look-ahead)."""
    if daily is None or daily.empty:
        return np.full(len(df), np.nan)
    d = daily.copy()
    d["sma"] = sma(d["Close"], n)
    d.index = pd.Index(pd.DatetimeIndex(d.index).date)
    s = d["sma"].shift(1)  # previous completed session
    days = pd.Index(df.index.date)
    return pd.Series(s.reindex(days).values, index=df.index).ffill().to_numpy()


def prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(pd.Index(df.index.date))
    d = pd.DataFrame({"H": g["High"].max(), "L": g["Low"].min(), "C": g["Close"].last(), "O": g["Open"].first()})
    p = d.shift(1)
    days = pd.Index(df.index.date)
    out = pd.DataFrame({k: p[k].reindex(days).values for k in ["H", "L", "C"]}, index=df.index)
    out["dayO"] = d["O"].reindex(days).values
    return out


# =============================================================================
#  STRATEGY FRAMEWORK
# =============================================================================
class Ctx:
    """Bar data + memoised indicator cache shared by strategies, filters, SL/target logic."""

    def __init__(self, df: pd.DataFrame, daily: Optional[pd.DataFrame] = None, aux: Optional[dict] = None):
        self.df = df
        self.daily = daily
        self.aux = aux or {}
        self.o = df["Open"].to_numpy(float)
        self.h = df["High"].to_numpy(float)
        self.l = df["Low"].to_numpy(float)
        self.c = df["Close"].to_numpy(float)
        self.v = df["Volume"].to_numpy(float)
        self.n = len(df)
        self._m: Dict[Any, Any] = {}

    def memo(self, key, fn):
        if key not in self._m:
            self._m[key] = fn()
        return self._m[key]

    def ema(self, n): return self.memo(("ema", n), lambda: ema(self.c, n))
    def sma(self, n): return self.memo(("sma", n), lambda: sma(self.c, n))
    def ma(self, n, kind="EMA"): return self.ema(n) if kind == "EMA" else self.sma(n)
    def atr(self, n=14): return self.memo(("atr", n), lambda: atr(self.h, self.l, self.c, n))
    def rsi(self, n=14): return self.memo(("rsi", n), lambda: rsi(self.c, n))
    def dmi(self, n=14, a=14): return self.memo(("dmi", n, a), lambda: dmi(self.h, self.l, self.c, n, a))
    def supertrend(self, f=3.0, p=10): return self.memo(("st", f, p), lambda: supertrend(self.h, self.l, self.c, f, p))
    def vwap(self): return self.memo("vwap", lambda: vwap_session(self.df))
    def macd(self, f=12, s=26, g=9): return self.memo(("macd", f, s, g), lambda: macd(self.c, f, s, g))
    def bb(self, n=20, m=2.0): return self.memo(("bb", n, m), lambda: bollinger(self.c, n, m))
    def pivots(self, left=5, right=5): return self.memo(("piv", left, right), lambda: pivots(self.h, self.l, left, right))

    def swings(self, left=5, right=5):
        def f():
            ph, pl, _, _ = self.pivots(left, right)
            sh_c, sh_p = last_two_swings(ph)
            sl_c, sl_p = last_two_swings(pl)
            return sh_c, sh_p, sl_c, sl_p
        return self.memo(("sw", left, right), f)


def _p(spec_type, default, lo=None, hi=None, step=None, help_=None, options=None):
    return {"type": spec_type, "default": default, "min": lo, "max": hi, "step": step, "help": help_, "options": options}


@dataclass
class StratDef:
    name: str
    func: Optional[Callable]
    params: Dict[str, dict]
    immediate: bool = False
    desc: str = ""


def _out(long, short, lines=None, status=None, exit_long=None, exit_short=None):
    long = np.nan_to_num(np.asarray(long, dtype=float)).astype(bool)
    short = np.nan_to_num(np.asarray(short, dtype=float)).astype(bool)
    return {"long": long, "short": short, "lines": lines or {}, "status": status or [],
            "exit_long": exit_long if exit_long is not None else short,
            "exit_short": exit_short if exit_short is not None else long}


def _gap_status(a, b, an, bn, i=-1):
    av, bv = a[i], b[i]
    if np.isnan(av) or np.isnan(bv):
        return ["Indicators warming up"]
    g = av - bv
    if g > 0:
        return [f"{an} above {bn} by {g:,.2f} pts → bullish side (LONG fires only on a fresh cross)",
                f"SHORT needs {an} to drop {g:,.2f} pts below {bn}"]
    return [f"{an} below {bn} by {abs(g):,.2f} pts → bearish side (SHORT fires only on a fresh cross)",
            f"LONG needs {an} to rise {abs(g):,.2f} pts above {bn}"]


# --------------------------------------------------------------------------- core strategies
def s_ma_cross(x: Ctx, p):
    f, s = x.ma(p["fast"], p["ma_type"]), x.ma(p["slow"], p["ma_type"])
    t = p["ma_type"]
    return _out(cross_over(f, s), cross_under(f, s),
                {f"{t} {p['fast']}": f, f"{t} {p['slow']}": s},
                _gap_status(f, s, f"{t}{p['fast']}", f"{t}{p['slow']}"))


def s_rsi(x: Ctx, p):
    r = x.rsi(p["length"])
    up, lo = float(p["upper"]), float(p["lower"])
    rv = r[-1]
    st_ = [f"RSI {fnum(rv)} | LONG on cross above {up} (needs +{fnum(max(up - rv, 0))}), "
           f"SHORT on cross below {lo} (needs -{fnum(max(rv - lo, 0))})"]
    return _out(cross_over(r, np.full(x.n, up)), cross_under(r, np.full(x.n, lo)), {f"RSI {p['length']}": r}, st_)


def s_triple_ema(x: Ctx, p):
    a, b, c = x.ema(p["fast"]), x.ema(p["mid"]), x.ema(p["slow"])
    with np.errstate(invalid="ignore"):
        bull = (a > b) & (b > c)
        bear = (a < b) & (b < c)
    long = bull & ~shift(bull.astype(float), 1).astype(bool)
    short = bear & ~shift(bear.astype(float), 1).astype(bool)
    st_ = [f"Aligned bullish: {bool(bull[-1])} | bearish: {bool(bear[-1])}",
           f"EMA{p['fast']}-EMA{p['mid']}: {fnum(a[-1] - b[-1])} | EMA{p['mid']}-EMA{p['slow']}: {fnum(b[-1] - c[-1])}"]
    return _out(long, short, {f"EMA {p['fast']}": a, f"EMA {p['mid']}": b, f"EMA {p['slow']}": c}, st_)


def s_ema_pullback(x: Ctx, p):
    f, t = x.ema(p["fast"]), x.ema(p["trend"])
    with np.errstate(invalid="ignore"):
        up = (f > t) & (x.c > t)
        dn = (f < t) & (x.c < t)
        long = up & (x.l <= f) & (x.c > f) & (x.c > x.o) & (shift(x.c) <= shift(f) * (1 + p["touch_pct"] / 100))
        short = dn & (x.h >= f) & (x.c < f) & (x.c < x.o) & (shift(x.c) >= shift(f) * (1 - p["touch_pct"] / 100))
    st_ = [f"Trend: {'UP' if up[-1] else ('DOWN' if dn[-1] else 'FLAT')} | distance to EMA{p['fast']}: {fnum(x.c[-1] - f[-1])} pts"]
    return _out(long, short, {f"EMA {p['fast']}": f, f"EMA {p['trend']}": t}, st_)


def _sr_levels(x: Ctx, p):
    if p["mode"] == "Swing pivots":
        sh, _, sl_, _ = x.swings(p["length"], p["length"])
        return sh, sl_
    return shift(highest(x.h, p["length"])), shift(lowest(x.l, p["length"]))


def s_support_breakdown(x: Ctx, p):
    _, sup = _sr_levels(x, p)
    br = cross_under(x.c, sup)
    if p.get("buffer_pts", 0):
        with np.errstate(invalid="ignore"):
            br = br & (x.c < sup - p["buffer_pts"])
    st_ = [f"Support {fnum(sup[-1])} | close {fnum(x.c[-1])} | needs -{fnum(max(x.c[-1] - sup[-1], 0))} pts to break"]
    return _out(np.zeros(x.n), br, {"Support": sup}, st_)


def s_resistance_breakout(x: Ctx, p):
    res, _ = _sr_levels(x, p)
    br = cross_over(x.c, res)
    if p.get("buffer_pts", 0):
        with np.errstate(invalid="ignore"):
            br = br & (x.c > res + p["buffer_pts"])
    st_ = [f"Resistance {fnum(res[-1])} | close {fnum(x.c[-1])} | needs +{fnum(max(res[-1] - x.c[-1], 0))} pts to break"]
    return _out(br, np.zeros(x.n), {"Resistance": res}, st_)


def s_smc(x: Ctx, p):
    """Smart-money concepts: BOS sets bias + order block; entry on mitigation of OB with rejection."""
    L = int(p["swing"])
    sh, _, sl_, _ = x.swings(L, L)
    n = x.n
    long, short = np.zeros(n, bool), np.zeros(n, bool)
    bias, ob_lo, ob_hi, used = 0, np.nan, np.nan, True
    status = "No structure yet"
    for i in range(2, n):
        if not np.isnan(sh[i - 1]) and x.c[i] > sh[i - 1] and x.c[i - 1] <= sh[i - 1]:
            bias = 1
            j = i - 1
            while j > max(i - 30, 0) and not (x.c[j] < x.o[j]):
                j -= 1
            ob_lo, ob_hi, used = x.l[j], x.h[j], False
        elif not np.isnan(sl_[i - 1]) and x.c[i] < sl_[i - 1] and x.c[i - 1] >= sl_[i - 1]:
            bias = -1
            j = i - 1
            while j > max(i - 30, 0) and not (x.c[j] > x.o[j]):
                j -= 1
            ob_lo, ob_hi, used = x.l[j], x.h[j], False
        if not used and not np.isnan(ob_lo):
            if bias == 1 and x.l[i] <= ob_hi and x.c[i] > ob_hi and x.c[i] > x.o[i]:
                long[i], used = True, True
            elif bias == -1 and x.h[i] >= ob_lo and x.c[i] < ob_lo and x.c[i] < x.o[i]:
                short[i], used = True, True
            if bias == 1 and x.c[i] < ob_lo:
                used = True
            if bias == -1 and x.c[i] > ob_hi:
                used = True
    if bias:
        status = (f"Bias {'BULLISH' if bias == 1 else 'BEARISH'} | Order block {fnum(ob_lo)}–{fnum(ob_hi)} | "
                  f"{'awaiting mitigation' if not used else 'mitigated – waiting next BOS'}")
    return _out(long, short, {"Swing High": sh, "Swing Low": sl_}, [status])


def s_elliott(x: Ctx, p):
    """Elliott wave-3 entry: impulse wave1, corrective wave2 retracing fib_min..fib_max, entry on break of wave-1 extreme."""
    L = int(p["pivot"])
    ph, pl, phi, pli = x.pivots(L, L)
    piv = []  # (bar_confirmed, type, price)
    for i in range(x.n):
        if not np.isnan(ph[i]):
            piv.append((i, "H", ph[i]))
        if not np.isnan(pl[i]):
            piv.append((i, "L", pl[i]))
    zz = []
    for b, t, v in piv:  # build alternating zigzag
        if zz and zz[-1][1] == t:
            if (t == "H" and v > zz[-1][2]) or (t == "L" and v < zz[-1][2]):
                zz[-1] = (b, t, v)
        else:
            zz.append((b, t, v))
    long, short = np.zeros(x.n, bool), np.zeros(x.n, bool)
    lo_f, hi_f = float(p["fib_min"]), float(p["fib_max"])
    status = "Scanning for wave 1-2 structure"
    for k in range(2, len(zz)):
        (b0, t0, v0), (b1, t1, v1), (b2, t2, v2) = zz[k - 2], zz[k - 1], zz[k]
        end = zz[k + 1][0] if k + 1 < len(zz) else x.n
        if t0 == "L" and t1 == "H" and t2 == "L" and v2 > v0:
            r = (v1 - v2) / (v1 - v0) if v1 != v0 else 0
            if lo_f <= r <= hi_f:
                for i in range(b2 + 1, min(end + L, x.n)):
                    if x.c[i] > v1 and x.c[i - 1] <= v1:
                        long[i] = True
                        break
                if k == len(zz) - 1:
                    status = f"Bullish W2 done ({r:.2f} retrace). W3 trigger above {fnum(v1)}"
        if t0 == "H" and t1 == "L" and t2 == "H" and v2 < v0:
            r = (v2 - v1) / (v0 - v1) if v0 != v1 else 0
            if lo_f <= r <= hi_f:
                for i in range(b2 + 1, min(end + L, x.n)):
                    if x.c[i] < v1 and x.c[i - 1] >= v1:
                        short[i] = True
                        break
                if k == len(zz) - 1:
                    status = f"Bearish W2 done ({r:.2f} retrace). W3 trigger below {fnum(v1)}"
    return _out(long, short, {}, [status])


# --------------------------------------------------------------------------- 20+ advanced strategies
def s_macd(x: Ctx, p):
    m, s, h = x.macd(p["fast"], p["slow"], p["signal"])
    long, short = cross_over(m, s), cross_under(m, s)
    if p["zero_filter"]:
        with np.errstate(invalid="ignore"):
            long, short = long & (m < 0), short & (m > 0)
    return _out(long, short, {"MACD": m, "Signal": s, "Hist": h}, _gap_status(m, s, "MACD", "Signal"))


def s_supertrend(x: Ctx, p):
    st_, d = x.supertrend(p["factor"], p["period"])
    long = (d == -1) & (shift(d) == 1)
    short = (d == 1) & (shift(d) == -1)
    return _out(long, short, {"Supertrend": st_},
                [f"Trend {'UP' if d[-1] == -1 else 'DOWN'} | Supertrend {fnum(st_[-1])} | flip distance {fnum(abs(x.c[-1] - st_[-1]))} pts"])


def s_bb_breakout(x: Ctx, p):
    b, u, lo = x.bb(p["length"], p["mult"])
    return _out(cross_over(x.c, u), cross_under(x.c, lo), {"BB Upper": u, "BB Basis": b, "BB Lower": lo},
                [f"Needs +{fnum(max(u[-1] - x.c[-1], 0))} to upper / -{fnum(max(x.c[-1] - lo[-1], 0))} to lower band"])


def s_bb_reversion(x: Ctx, p):
    b, u, lo = x.bb(p["length"], p["mult"])
    long = cross_over(x.c, lo)
    short = cross_under(x.c, u)
    with np.errstate(invalid="ignore"):
        el = x.c >= b
        es = x.c <= b
    return _out(long, short, {"BB Upper": u, "BB Basis": b, "BB Lower": lo},
                ["Re-entry inside bands after an excursion; exits at basis"], exit_long=el | short, exit_short=es | long)


def s_vwap(x: Ctx, p):
    vw = x.vwap()
    t = x.ema(p["trend"])
    with np.errstate(invalid="ignore"):
        long = cross_over(x.c, vw) & (x.c > t)
        short = cross_under(x.c, vw) & (x.c < t)
    return _out(long, short, {"VWAP": vw, f"EMA {p['trend']}": t}, [f"Close-VWAP: {fnum(x.c[-1] - vw[-1])} pts"])


def s_donchian(x: Ctx, p):
    up, dn = shift(highest(x.h, p["length"])), shift(lowest(x.l, p["length"]))
    with np.errstate(invalid="ignore"):
        long = (x.c > up) & (shift(x.c) <= shift(up))
        short = (x.c < dn) & (shift(x.c) >= shift(dn))
    return _out(long, short, {"DC Upper": up, "DC Lower": dn},
                [f"Upper {fnum(up[-1])} (+{fnum(max(up[-1] - x.c[-1], 0))}) / Lower {fnum(dn[-1])}"])


def s_orb(x: Ctx, p):
    mins = int(p["minutes"])
    idx = x.df.index
    day = pd.Index(idx.date)
    first = pd.Series(idx, index=idx).groupby(day).transform("min")
    in_rng = (pd.Series(idx, index=idx) - first) < pd.Timedelta(minutes=mins)
    hi = pd.Series(np.where(in_rng, x.h, np.nan), index=idx).groupby(day).transform("max").to_numpy()
    lo = pd.Series(np.where(in_rng, x.l, np.nan), index=idx).groupby(day).transform("min").to_numpy()
    after = ~in_rng.to_numpy()
    long = cross_over(x.c, hi) & after
    short = cross_under(x.c, lo) & after
    # one breakout per side per day
    for arr in (long, short):
        seen = set()
        for i in np.where(arr)[0]:
            if day[i] in seen:
                arr[i] = False
            seen.add(day[i])
    return _out(long, short, {"ORB High": hi, "ORB Low": lo}, [f"ORB {fnum(lo[-1])} – {fnum(hi[-1])}"])


def s_stoch(x: Ctx, p):
    k, d = stoch(x.c, x.h, x.l, p["k"], p["smooth"], p["d"])
    with np.errstate(invalid="ignore"):
        long = cross_over(k, d) & (shift(k) < p["os"])
        short = cross_under(k, d) & (shift(k) > p["ob"])
    return _out(long, short, {"%K": k, "%D": d}, [f"%K {fnum(k[-1])} %D {fnum(d[-1])} (OS {p['os']} / OB {p['ob']})"])


def s_ichimoku(x: Ctx, p):
    conv = (highest(x.h, p["conv"]) + lowest(x.l, p["conv"])) / 2
    base = (highest(x.h, p["base"]) + lowest(x.l, p["base"])) / 2
    la = shift((conv + base) / 2, p["base"] - 1)
    lb = shift((highest(x.h, p["span_b"]) + lowest(x.l, p["span_b"])) / 2, p["base"] - 1)
    top, bot = np.fmax(la, lb), np.fmin(la, lb)
    with np.errstate(invalid="ignore"):
        long = cross_over(conv, base) & (x.c > top)
        short = cross_under(conv, base) & (x.c < bot)
    return _out(long, short, {"Tenkan": conv, "Kijun": base, "Span A": la, "Span B": lb},
                _gap_status(conv, base, "Tenkan", "Kijun"))


def s_adx_dmi(x: Ctx, p):
    pl, mi, ad = x.dmi(p["length"], p["length"])
    with np.errstate(invalid="ignore"):
        long = cross_over(pl, mi) & (ad >= p["adx_min"])
        short = cross_under(pl, mi) & (ad >= p["adx_min"])
    return _out(long, short, {"+DI": pl, "-DI": mi, "ADX": ad}, _gap_status(pl, mi, "+DI", "-DI") + [f"ADX {fnum(ad[-1])}"])


def s_heikin(x: Ctx, p):
    ho, hc = heikin_ashi(x.df)
    bull = hc > ho
    k = int(p["confirm"])
    bs = pd.Series(bull.astype(int)).rolling(k).sum().to_numpy()
    long = (bs == k) & (shift(bs) < k)
    brs = pd.Series((~bull).astype(int)).rolling(k).sum().to_numpy()
    short = (brs == k) & (shift(brs) < k)
    return _out(long, short, {"HA Open": ho, "HA Close": hc}, [f"HA candle {'GREEN' if bull[-1] else 'RED'}"])


def s_keltner(x: Ctx, p):
    m, u, lo = keltner(x.h, x.l, x.c, p["length"], p["mult"])
    return _out(cross_over(x.c, u), cross_under(x.c, lo), {"KC Upper": u, "KC Mid": m, "KC Lower": lo},
                [f"Upper {fnum(u[-1])} / Lower {fnum(lo[-1])}"])


def s_psar(x: Ctx, p):
    sar, tr = psar(x.h, x.l, p["start"], p["inc"], p["max"])
    long = (tr == 1) & (shift(tr) == -1)
    short = (tr == -1) & (shift(tr) == 1)
    return _out(long, short, {"PSAR": sar}, [f"SAR {fnum(sar[-1])} | trend {'UP' if tr[-1] == 1 else 'DOWN'}"])


def s_cci(x: Ctx, p):
    cc = cci(x.h, x.l, x.c, p["length"])
    lv = float(p["level"])
    return _out(cross_over(cc, np.full(x.n, lv)), cross_under(cc, np.full(x.n, -lv)), {"CCI": cc},
                [f"CCI {fnum(cc[-1])} (LONG > {lv}, SHORT < {-lv})"])


def s_willr(x: Ctx, p):
    w = williams_r(x.h, x.l, x.c, p["length"])
    return _out(cross_over(w, np.full(x.n, -80.0)), cross_under(w, np.full(x.n, -20.0)), {"%R": w},
                [f"%R {fnum(w[-1])} (LONG: cross up -80, SHORT: cross down -20)"])


def s_inside_bar(x: Ctx, p):
    mh, ml = shift(x.h, 2), shift(x.l, 2)
    with np.errstate(invalid="ignore"):
        inside = (shift(x.h) < mh) & (shift(x.l) > ml)
        long = inside & (x.c > mh)
        short = inside & (x.c < ml)
    return _out(long, short, {}, [f"Previous bar inside: {bool(inside[-1])}"])


def s_engulf(x: Ctx, p):
    t = x.ema(p["trend"])
    po, pc = shift(x.o), shift(x.c)
    with np.errstate(invalid="ignore"):
        bull = (pc < po) & (x.c > x.o) & (x.c >= po) & (x.o <= pc) & (x.l < t)
        bear = (pc > po) & (x.c < x.o) & (x.c <= po) & (x.o >= pc) & (x.h > t)
    return _out(bull, bear, {f"EMA {p['trend']}": t}, ["Engulfing at/around trend EMA"])


def s_rsi_div(x: Ctx, p):
    r = x.rsi(p["rsi"])
    L = int(p["pivot"])
    _, _, phi, pli = x.pivots(L, L)
    long, short = np.zeros(x.n, bool), np.zeros(x.n, bool)
    lastL, lastH = None, None
    for i in range(x.n):
        if pli[i] >= 0:
            c = pli[i]
            if lastL is not None and x.l[c] < x.l[lastL] and r[c] > r[lastL]:
                long[i] = True
            lastL = c
        if phi[i] >= 0:
            c = phi[i]
            if lastH is not None and x.h[c] > x.h[lastH] and r[c] < r[lastH]:
                short[i] = True
            lastH = c
    return _out(long, short, {f"RSI {p['rsi']}": r}, [f"RSI {fnum(r[-1])} | divergence confirmed {L} bars after pivot"])


def s_vol_spike(x: Ctx, p):
    av = sma(x.v, p["lookback"])
    up, dn = shift(highest(x.h, p["lookback"])), shift(lowest(x.l, p["lookback"]))
    with np.errstate(invalid="ignore"):
        spike = x.v > p["mult"] * av
        long = spike & cross_over(x.c, up)
        short = spike & cross_under(x.c, dn)
    return _out(long, short, {"Range High": up, "Range Low": dn},
                [f"Vol {fnum(x.v[-1], 0)} vs {p['mult']}x avg {fnum(p['mult'] * av[-1], 0)}"])


def s_hull(x: Ctx, p):
    hm = x.memo(("hma", p["length"]), lambda: hma(x.c, p["length"]))
    d = hm - shift(hm)
    long = (d > 0) & (shift(d) <= 0)
    short = (d < 0) & (shift(d) >= 0)
    return _out(long, short, {f"HMA {p['length']}": hm}, [f"HMA slope {fnum(d[-1])}"])


def s_squeeze(x: Ctx, p):
    n = int(p["length"])
    b, u, lo = x.bb(n, p["bb_mult"])
    _, ku, kl = keltner(x.h, x.l, x.c, n, p["kc_mult"])
    with np.errstate(invalid="ignore"):
        sq_on = (lo > kl) & (u < ku)
    mid = ((highest(x.h, n) + lowest(x.l, n)) / 2 + sma(x.c, n)) / 2
    mom = linreg(x.c - mid, n)
    release = ~sq_on & shift(sq_on.astype(float)).astype(bool)
    with np.errstate(invalid="ignore"):
        long = release & (mom > 0)
        short = release & (mom < 0)
    return _out(long, short, {"Momentum": mom}, [f"Squeeze {'ON' if sq_on[-1] else 'OFF'} | momentum {fnum(mom[-1])}"])


def s_cpr(x: Ctx, p):
    pdl = x.memo("pdl", lambda: prev_day_levels(x.df))
    pv = (pdl["H"] + pdl["L"] + pdl["C"]).to_numpy() / 3
    bc = (pdl["H"] + pdl["L"]).to_numpy() / 2
    tc = 2 * pv - bc
    top, bot = np.fmax(tc, bc), np.fmin(tc, bc)
    r1, s1 = 2 * pv - pdl["L"].to_numpy(), 2 * pv - pdl["H"].to_numpy()
    lvl_up = top if p["level"] == "CPR" else r1
    lvl_dn = bot if p["level"] == "CPR" else s1
    return _out(cross_over(x.c, lvl_up), cross_under(x.c, lvl_dn), {"CPR Top": top, "CPR Bottom": bot, "R1": r1, "S1": s1},
                [f"CPR {fnum(bot[-1])}–{fnum(top[-1])} | R1 {fnum(r1[-1])} S1 {fnum(s1[-1])}"])


def s_zscore(x: Ctx, p):
    m, sd = sma(x.c, p["length"]), stdev(x.c, p["length"])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x.c - m) / sd
    t = float(p["z"])
    return _out(cross_over(z, np.full(x.n, -t)), cross_under(z, np.full(x.n, t)), {"Z-score": z},
                [f"Z {fnum(z[-1])} (LONG: cross up -{t}, SHORT: cross down +{t})"],
                exit_long=cross_over(z, np.zeros(x.n)), exit_short=cross_under(z, np.zeros(x.n)))


def s_roc(x: Ctx, p):
    n = int(p["length"])
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = 100 * (x.c - shift(x.c, n)) / shift(x.c, n)
    t = x.ema(p["trend"])
    with np.errstate(invalid="ignore"):
        long = cross_over(roc, np.zeros(x.n)) & (x.c > t)
        short = cross_under(roc, np.zeros(x.n)) & (x.c < t)
    return _out(long, short, {f"ROC {n}": roc, f"EMA {p['trend']}": t}, [f"ROC {fnum(roc[-1])}%"])


def s_golden(x: Ctx, p):
    f, s = x.sma(p["fast"]), x.sma(p["slow"])
    return _out(cross_over(f, s), cross_under(f, s), {f"SMA {p['fast']}": f, f"SMA {p['slow']}": s},
                _gap_status(f, s, f"SMA{p['fast']}", f"SMA{p['slow']}"))


def s_rsi_ema_combo(x: Ctx, p):
    f, s = x.ema(p["fast"]), x.ema(p["slow"])
    r = x.rsi(14)
    with np.errstate(invalid="ignore"):
        long = cross_over(f, s) & (r > 50)
        short = cross_under(f, s) & (r < 50)
    return _out(long, short, {f"EMA {p['fast']}": f, f"EMA {p['slow']}": s, "RSI 14": r},
                _gap_status(f, s, f"EMA{p['fast']}", f"EMA{p['slow']}") + [f"RSI {fnum(r[-1])} (needs >50 / <50)"])


def s_st_ema(x: Ctx, p):
    st_, d = x.supertrend(3.0, 10)
    t = x.ema(p["trend"])
    flip_up = (d == -1) & (shift(d) == 1)
    flip_dn = (d == 1) & (shift(d) == -1)
    with np.errstate(invalid="ignore"):
        long = flip_up & (x.c > t)
        short = flip_dn & (x.c < t)
    return _out(long, short, {"Supertrend": st_, f"EMA {p['trend']}": t}, [f"ST {'UP' if d[-1] == -1 else 'DOWN'}"])


def s_fvg_reversal(x: Ctx, p):
    h2, l2 = shift(x.h, 2), shift(x.l, 2)
    n = x.n
    long, short = np.zeros(n, bool), np.zeros(n, bool)
    bull_zone, bear_zone = None, None
    for i in range(2, n):
        if x.l[i] > h2[i]:
            bull_zone = (h2[i], x.l[i], i)
        if x.h[i] < l2[i]:
            bear_zone = (x.h[i], l2[i], i)
        if bull_zone and i > bull_zone[2] and x.l[i] <= bull_zone[1] and x.c[i] > bull_zone[1] and x.c[i] > x.o[i]:
            long[i], bull_zone = True, None
        if bear_zone and i > bear_zone[2] and x.h[i] >= bear_zone[0] and x.c[i] < bear_zone[0] and x.c[i] < x.o[i]:
            short[i], bear_zone = True, None
    return _out(long, short, {}, ["Entry on rejection from the latest unfilled FVG"])


def s_vwap_st(x: Ctx, p):
    """VWAP + Supertrend: fires when BOTH agree and the combined state has just turned on."""
    vw = x.vwap()
    st_, d = x.supertrend(p["factor"], p["period"])
    with np.errstate(invalid="ignore"):
        bull = (d == -1) & (x.c > vw)
        bear = (d == 1) & (x.c < vw)
    long = bull & ~shift(bull.astype(float)).astype(bool)
    short = bear & ~shift(bear.astype(float)).astype(bool)
    return _out(long, short, {"VWAP": vw, "Supertrend": st_},
                [f"Supertrend {'UP' if d[-1] == -1 else 'DOWN'} ({fnum(st_[-1])}) | close-VWAP {fnum(x.c[-1] - vw[-1])} pts",
                 f"LONG state: {bool(bull[-1])} | SHORT state: {bool(bear[-1])} (entry only when state just turns on)"])


# --------------------------------------------------------------------------- VIX / volatility expected-move bands
VIX_NAME = "VIX Expected-Move Bands (σ levels)"
VOL_SOURCES = ["Auto (India VIX for NSE indices, else HV) + manual override", "Manual value (all bars)",
               "Historical volatility (HV) of the ticker"]
BAR_DAYS = {"1d": 1, "3d": 3, "1wk": 5, "1mon": 21}


def _session_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(df.index.tz_localize(None).normalize()) if df.index.tz is not None else pd.DatetimeIndex(df.index.normalize())


def _prior_value_map(ser: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """For every bar date D return the series value of the last date strictly before D (no look-ahead)."""
    ser = ser.dropna()
    if ser.empty:
        return np.full(len(dates), np.nan)
    sd = _session_dates(ser.to_frame()) if isinstance(ser.index, pd.DatetimeIndex) else pd.DatetimeIndex(ser.index)
    order = np.argsort(sd.values)
    sdv, vals = sd.values[order], ser.to_numpy(float)[order]
    pos = np.searchsorted(sdv, dates.values, side="left") - 1
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
    return out


def _daily_closes(x: Ctx) -> pd.Series:
    if x.daily is not None and not x.daily.empty:
        return x.daily["Close"]
    if x.aux.get("tf") in INTRADAY_TFS:
        g = x.df["Close"].groupby(_session_dates(x.df)).last()
        g.index = pd.DatetimeIndex(g.index).tz_localize(IST)
        return g
    return x.df["Close"]


def _vol_per_bar(x: Ctx, p) -> Tuple[np.ndarray, str, float, dict]:
    """Annualised volatility % used for each bar's session + label + value for the NEXT session + display info."""
    dates = _session_dates(x.df)
    mode, manual = p["vol_source"], float(p["manual_vix"])
    info: Dict[str, Any] = {"source": "", "fetched": np.nan, "fetched_date": None, "used": np.nan, "used_date": None,
                            "next": np.nan, "override": None, "session": dates[-1].date() if len(dates) else None}
    if mode.startswith("Manual"):
        v = manual if manual > 0 else 15.0
        info.update(source="Manual (all bars)", used=v, next=v, override=v)
        return np.full(x.n, v), f"Manual {v}", v, info
    vix = x.aux.get("vix")
    if mode.startswith("Auto") and vix is not None and not vix.empty:
        ser, label = vix["Close"], "India VIX (prev close)"
        info["source"] = "India VIX (^INDIAVIX, yfinance)"
    else:
        dc = _daily_closes(x)
        lr = np.log(dc.astype(float)).diff()
        ser = lr.rolling(int(p["hv_len"])).std(ddof=1) * math.sqrt(float(p["days_per_year"])) * 100
        label = f"HV{int(p['hv_len'])} ({int(p['days_per_year'])}d/yr)"
        info["source"] = f"Historical volatility {label}" + (" — India VIX unavailable" if mode.startswith("Auto") and x.aux.get("kind") == "index" else "")
    vals = _prior_value_map(ser, dates)
    sd = ser.dropna()
    nxt = float(sd.iloc[-1]) if not sd.empty else np.nan
    if not sd.empty:
        sdd = _session_dates(sd.to_frame())
        info.update(fetched=float(sd.iloc[-1]), fetched_date=sdd[-1].date())
        prior = np.where(sdd < dates[-1])[0]
        if len(prior):
            info.update(used=float(sd.iloc[prior[-1]]), used_date=sdd[prior[-1]].date())
    info["next"] = nxt
    if mode.startswith("Auto") and manual > 0:   # override latest session + next-session projection
        vals = np.where(dates == dates[-1], manual, vals)
        nxt, label = manual, label + " | manual override on latest session"
        info.update(used=manual, next=manual, override=manual)
    return vals, label, nxt, info


def vix_info_text(vi: dict) -> str:
    """One-line human summary of the volatility value fetched vs used."""
    if not vi:
        return ""
    d = lambda v: v.strftime("%d-%b-%Y") if v is not None else "—"
    parts = [f"**Source:** {vi['source']}"]
    if vi.get("fetched") == vi.get("fetched"):
        parts.append(f"**Latest fetched:** {fnum(vi['fetched'])} ({d(vi['fetched_date'])} close)")
    parts.append(f"**Used for {d(vi.get('session'))} bands:** {fnum(vi['used'])}"
                 + (f" (from {d(vi['used_date'])} close)" if vi.get("used_date") and vi.get("override") is None else ""))
    if vi.get("override") is not None:
        parts.append(f"**Manual override:** {fnum(vi['override'])}")
    parts.append(f"**Next-session bands use:** {fnum(vi['next'])}")
    return " · ".join(parts)


def s_vix(x: Ctx, p):
    """Expected move = PrevClose × Vol% ÷ factor (× √bar_days for D/W/M bars). 1600 = 100 × √256 for NIFTY/India VIX.
    Generic: Vol% = annualised implied (VIX-type) or historical volatility; factor = 100 × √(trading days per year)."""
    tf = x.aux.get("tf", "1d")
    if tf in INTRADAY_TFS:
        pdl = x.memo("pdl", lambda: prev_day_levels(x.df))
        anchor, dayo = pdl["C"].to_numpy(float), pdl["dayO"].to_numpy(float)
    else:
        anchor, dayo = shift(x.c), x.o
    vol, src, vol_next, vinfo = _vol_per_bar(x, p)
    scale = math.sqrt(BAR_DAYS.get(tf, 1))
    em = anchor * vol / float(p["factor"]) * scale
    lines = {"Prev close": anchor}
    for m in (1, 2, 3):
        lines[f"+{m}σ"] = anchor + m * em
        lines[f"-{m}σ"] = anchor - m * em
    if p["move_entry"]:
        ref = anchor if p["anchor"] == "Previous close" else dayo
        trig = np.full(x.n, float(p["move_pts"])) if p["move_type"] == "Absolute points" else em * float(p["move_pct"]) / 100
        up, dn, ud, dd = ref + trig, ref - trig, 1, -1
        lines["Long trigger"], lines["Short trigger"] = up, dn
        mode_txt = (f"Move entry: {fnum(p['move_pts'])} pts" if p["move_type"] == "Absolute points"
                    else f"Move entry: {p['move_pct']}% of 1σ") + f" from {p['anchor'].lower()}"
    else:
        m = int(p["entry_sigma"])
        up, dn = anchor + m * em, anchor - m * em
        if p["base_mode"].startswith("Breakout"):
            ud, dd = 1, -1
        else:
            ud, dd = -1, 1
        mode_txt = f"{p['base_mode']} at ±{m}σ"
    if p["allow"] == "Long only":
        up = up if ud == 1 else np.full(x.n, np.nan)
        dn = dn if dd == 1 else np.full(x.n, np.nan)
    elif p["allow"] == "Short only":
        up = up if ud == -1 else np.full(x.n, np.nan)
        dn = dn if dd == -1 else np.full(x.n, np.nan)
    tl = ts = np.full(x.n, np.nan)
    if p["sigma_exit"]:
        lvl = p["exit_level"]
        mult = 0 if lvl.startswith("Prev") else int(lvl.strip("±σ"))
        buf = float(p["exit_buffer"])
        tl = anchor + mult * em - buf
        ts = anchor - mult * em + buf
    i = -1
    status = [f"{mode_txt} | Vol {fnum(vol[i])} [{src}] | factor {p['factor']} | prev close {fnum(anchor[i])} | EM ±{fnum(em[i])}",
              f"1σ {fnum(anchor[i] - em[i])} – {fnum(anchor[i] + em[i])} | 2σ {fnum(anchor[i] - 2 * em[i])} – {fnum(anchor[i] + 2 * em[i])} | "
              f"3σ {fnum(anchor[i] - 3 * em[i])} – {fnum(anchor[i] + 3 * em[i])}",
              f"Entry levels: up {fnum(up[i])} → {'BUY' if ud == 1 else 'SELL'}, down {fnum(dn[i])} → {'BUY' if dd == 1 else 'SELL'}"
              + (f" | σ-exit target L {fnum(tl[i])} / S {fnum(ts[i])}" if p["sigma_exit"] else " | exit: sidebar SL/Target")]
    if vol_next == vol_next:
        em_n = x.c[-1] * vol_next / float(p["factor"]) * scale
        status.append(f"Next session (close {fnum(x.c[-1])}, vol {fnum(vol_next)}): EM ±{fnum(em_n)} → 1σ {fnum(x.c[-1] - em_n)} – "
                      f"{fnum(x.c[-1] + em_n)}, 2σ {fnum(x.c[-1] - 2 * em_n)} – {fnum(x.c[-1] + 2 * em_n)}")
    out = _out(np.zeros(x.n), np.zeros(x.n), lines, status)
    out.update(imm_up=up, imm_ud=ud, imm_dn=dn, imm_dd=dd, tgt_long=tl, tgt_short=ts, em=em, vol_info=vinfo)
    return out


STRATEGIES: Dict[str, StratDef] = {}


def _reg(name, func, params=None, immediate=False, desc=""):
    STRATEGIES[name] = StratDef(name, func, params or {}, immediate, desc)


MA_T = ["EMA", "SMA"]
_reg("Simple Buy", None, {}, True, "Enters LONG immediately; exits on SL/Target; re-enters on next candle after exit.")
_reg("Simple Sell", None, {}, True, "Enters SHORT immediately; exits on SL/Target; re-enters on next candle after exit.")
_reg("EMA/SMA Crossover", s_ma_cross, {"fast": _p("int", 9, 1, 500), "slow": _p("int", 21, 2, 1000),
                                       "ma_type": _p("select", "EMA", options=MA_T)})
_reg("Price Threshold (Above/Below)", None, {
    "above_price": _p("float", 0.0, 0.0, None, 0.05, "0 = disabled"),
    "above_action": _p("select", "Buy", options=["Buy", "Sell"]),
    "below_price": _p("float", 0.0, 0.0, None, 0.05, "0 = disabled"),
    "below_action": _p("select", "Sell", options=["Buy", "Sell"])}, True,
    "Immediate entry when price crosses the level (no next-candle wait).")
_reg("Percentage Threshold (Above/Below)", None, {
    "ref": _p("select", "Previous close", options=["Previous close", "Day open"]),
    "above_pct": _p("float", 1.0, 0.0, 100.0, 0.05, "0 = disabled"),
    "above_action": _p("select", "Buy", options=["Buy", "Sell"]),
    "below_pct": _p("float", 1.0, 0.0, 100.0, 0.05, "0 = disabled"),
    "below_action": _p("select", "Sell", options=["Buy", "Sell"])}, True,
    "Immediate entry when % move from reference crosses the threshold.")
_reg("RSI 60/40", s_rsi, {"length": _p("int", 14, 2, 200), "upper": _p("float", 60.0, 1.0, 99.0, 1.0),
                          "lower": _p("float", 40.0, 1.0, 99.0, 1.0)})
_reg("SMC (BOS + Order Block)", s_smc, {"swing": _p("int", 5, 2, 30)})
_reg("Triple EMA Crossover", s_triple_ema, {"fast": _p("int", 9, 1, 500), "mid": _p("int", 21, 2, 500), "slow": _p("int", 50, 3, 1000)})
_reg("EMA Pullback", s_ema_pullback, {"fast": _p("int", 21, 2, 500), "trend": _p("int", 50, 3, 1000),
                                      "touch_pct": _p("float", 0.5, 0.0, 10.0, 0.1)})
_reg("Support Breakdown", s_support_breakdown, {"mode": _p("select", "Swing pivots", options=["Swing pivots", "Lookback range"]),
                                                "length": _p("int", 10, 2, 200), "buffer_pts": _p("float", 0.0, 0.0, None, 0.5)})
_reg("Resistance Breakout", s_resistance_breakout, {"mode": _p("select", "Swing pivots", options=["Swing pivots", "Lookback range"]),
                                                    "length": _p("int", 10, 2, 200), "buffer_pts": _p("float", 0.0, 0.0, None, 0.5)})
_reg("Elliott Wave (Wave-3 Entry)", s_elliott, {"pivot": _p("int", 5, 2, 30), "fib_min": _p("float", 0.382, 0.1, 1.0, 0.01),
                                                "fib_max": _p("float", 0.786, 0.1, 1.0, 0.01)})
_reg("MACD Crossover", s_macd, {"fast": _p("int", 12, 1, 200), "slow": _p("int", 26, 2, 400), "signal": _p("int", 9, 1, 100),
                                "zero_filter": _p("bool", False)})
_reg("Supertrend Flip", s_supertrend, {"period": _p("int", 10, 1, 200), "factor": _p("float", 3.0, 0.1, 20.0, 0.1)})
_reg("Bollinger Breakout", s_bb_breakout, {"length": _p("int", 20, 2, 400), "mult": _p("float", 2.0, 0.1, 10.0, 0.1)})
_reg("Bollinger Mean Reversion", s_bb_reversion, {"length": _p("int", 20, 2, 400), "mult": _p("float", 2.0, 0.1, 10.0, 0.1)})
_reg("VWAP Cross + Trend", s_vwap, {"trend": _p("int", 50, 2, 500)})
_reg("Donchian Breakout (Turtle)", s_donchian, {"length": _p("int", 20, 2, 400)})
_reg("Opening Range Breakout", s_orb, {"minutes": _p("int", 15, 1, 240)})
_reg("Stochastic Crossover", s_stoch, {"k": _p("int", 14, 2, 200), "smooth": _p("int", 3, 1, 50), "d": _p("int", 3, 1, 50),
                                       "ob": _p("float", 80.0, 50.0, 100.0, 1.0), "os": _p("float", 20.0, 0.0, 50.0, 1.0)})
_reg("Ichimoku TK Cross", s_ichimoku, {"conv": _p("int", 9, 1, 200), "base": _p("int", 26, 2, 300), "span_b": _p("int", 52, 2, 500)})
_reg("ADX / DMI Crossover", s_adx_dmi, {"length": _p("int", 14, 2, 200), "adx_min": _p("float", 20.0, 0.0, 100.0, 1.0)})
_reg("Heikin Ashi Trend Flip", s_heikin, {"confirm": _p("int", 2, 1, 10)})
_reg("Keltner Channel Breakout", s_keltner, {"length": _p("int", 20, 2, 400), "mult": _p("float", 2.0, 0.1, 10.0, 0.1)})
_reg("Parabolic SAR Flip", s_psar, {"start": _p("float", 0.02, 0.001, 1.0, 0.001), "inc": _p("float", 0.02, 0.001, 1.0, 0.001),
                                    "max": _p("float", 0.2, 0.01, 1.0, 0.01)})
_reg("CCI ±100 Cross", s_cci, {"length": _p("int", 20, 2, 400), "level": _p("float", 100.0, 1.0, 500.0, 1.0)})
_reg("Williams %R Reversal", s_willr, {"length": _p("int", 14, 2, 200)})
_reg("Inside Bar Breakout", s_inside_bar, {})
_reg("Engulfing Reversal", s_engulf, {"trend": _p("int", 20, 2, 500)})
_reg("RSI Divergence", s_rsi_div, {"rsi": _p("int", 14, 2, 100), "pivot": _p("int", 5, 2, 30)})
_reg("Volume Spike Breakout", s_vol_spike, {"lookback": _p("int", 20, 2, 400), "mult": _p("float", 2.0, 1.0, 20.0, 0.1)})
_reg("Hull MA Turn", s_hull, {"length": _p("int", 21, 2, 400)})
_reg("Squeeze Momentum Release", s_squeeze, {"length": _p("int", 20, 2, 200), "bb_mult": _p("float", 2.0, 0.1, 10.0, 0.1),
                                             "kc_mult": _p("float", 1.5, 0.1, 10.0, 0.1)})
_reg("CPR / Pivot Breakout", s_cpr, {"level": _p("select", "CPR", options=["CPR", "R1/S1"])})
_reg("Z-Score Mean Reversion", s_zscore, {"length": _p("int", 20, 2, 400), "z": _p("float", 2.0, 0.5, 5.0, 0.1)})
_reg("ROC Momentum", s_roc, {"length": _p("int", 12, 1, 200), "trend": _p("int", 50, 2, 500)})
_reg("Golden / Death Cross", s_golden, {"fast": _p("int", 50, 2, 500), "slow": _p("int", 200, 3, 1000)})
_reg("EMA Cross + RSI Confirm", s_rsi_ema_combo, {"fast": _p("int", 9, 1, 500), "slow": _p("int", 21, 2, 1000)})
_reg("Supertrend + EMA Trend", s_st_ema, {"trend": _p("int", 200, 2, 1000)})
_reg("FVG Rejection", s_fvg_reversal, {})
_reg("VWAP + Supertrend", s_vwap_st, {"period": _p("int", 10, 1, 200), "factor": _p("float", 3.0, 0.1, 20.0, 0.1)})
_reg(VIX_NAME, s_vix, {
    "vol_source": _p("select", VOL_SOURCES[0], options=VOL_SOURCES),
    "manual_vix": _p("float", 0.0, 0.0, 500.0, 0.05, "India VIX / annualised vol %. 0 = no override (Auto mode)"),
    "factor": _p("float", 1600.0, 1.0, 100000.0, 1.0, "100×√256=1600 (NIFTY/VIX). Stocks/forex 100×√252≈1587, crypto 100×√365≈1911"),
    "hv_len": _p("int", 20, 2, 500), "days_per_year": _p("int", 252, 1, 366),
    "base_mode": _p("select", "Breakout beyond σ", options=["Breakout beyond σ", "Reversal at σ (fade)"]),
    "entry_sigma": _p("select", "1", options=["1", "2", "3"]),
    "move_entry": _p("bool", False), "move_type": _p("select", "Absolute points", options=["Absolute points", "% of 1σ move"]),
    "move_pts": _p("float", 120.0, 0.01, 1e7, 1.0), "move_pct": _p("float", 70.0, 1.0, 300.0, 1.0),
    "anchor": _p("select", "Previous close", options=["Previous close", "Day open"]),
    "sigma_exit": _p("bool", False), "exit_level": _p("select", "±1σ", options=["±1σ", "±2σ", "±3σ", "Prev close"]),
    "exit_buffer": _p("float", 5.0, 0.0, 1e6, 0.5), "max_entries_day": _p("int", 1, 1, 50),
    "allow": _p("select", "Both", options=["Both", "Long only", "Short only"])}, True,
    "Expected move = PrevClose × VIX ÷ 1600 (configurable). Intrabar entries at σ / move-trigger levels.")
_reg("Hybrid (Multi-Strategy Consensus)", None, {}, False, "Signal only when the selected strategies agree.")

IMMEDIATE = {k for k, v in STRATEGIES.items() if v.immediate}
SIGNAL_STRATS = [k for k, v in STRATEGIES.items() if v.func is not None and not v.immediate]


def run_strategy(x: Ctx, name: str, params: dict, hybrid: Optional[dict] = None) -> dict:
    if name == "Hybrid (Multi-Strategy Consensus)":
        hy = hybrid or {}
        members = hy.get("members") or []
        if not members:
            return _out(np.zeros(x.n), np.zeros(x.n), {}, ["Select hybrid members"])
        win = max(int(hy.get("window", 1)), 1)
        mode = hy.get("mode", "All agree")
        outs = {m: STRATEGIES[m].func(x, hy.get("params", {}).get(m, default_params(m))) for m in members}
        L = np.vstack([pd.Series(o["long"].astype(int)).rolling(win, min_periods=1).max().to_numpy() for o in outs.values()])
        S = np.vstack([pd.Series(o["short"].astype(int)).rolling(win, min_periods=1).max().to_numpy() for o in outs.values()])
        fresh_L = np.vstack([o["long"] for o in outs.values()]).any(0)
        fresh_S = np.vstack([o["short"] for o in outs.values()]).any(0)
        need = len(members) if mode == "All agree" else (len(members) // 2 + 1)
        long = (L.sum(0) >= need) & fresh_L
        short = (S.sum(0) >= need) & fresh_S
        lines, status = {}, []
        for m, o in outs.items():
            lines.update(o["lines"])
            status.append(f"[{m}] " + ("; ".join(o["status"][:1]) if o["status"] else "—")
                          + f" | L-vote {int(L[[*outs].index(m)][-1])} S-vote {int(S[[*outs].index(m)][-1])}")
        status.insert(0, f"Votes LONG {int(L[:, -1].sum())}/{len(members)}, SHORT {int(S[:, -1].sum())}/{len(members)} (need {need})")
        return _out(long, short, lines, status)
    sd = STRATEGIES[name]
    if sd.func is None:
        return _out(np.zeros(x.n), np.zeros(x.n), {}, [])
    return sd.func(x, params)


def default_params(name: str) -> dict:
    return {k: v["default"] for k, v in STRATEGIES[name].params.items()}


# =============================================================================
#  ENTRY FILTERS
# =============================================================================
FILTER_LIST = ["ADX range", "Break of Structure (BOS)", "Change of Character (CHoCH)", "Volume N×",
               "Absolute OI", "Change in OI", "Regime (trend) aligned", "Supertrend", "VWAP", "MACD",
               "Bollinger Bands", "RSI", "Fair Value Gap", "Liquidity Sweep", "Demand Zone Sustain",
               "Supply Zone Sustain", "EMA 20", "SMA 20", "EMA 63", "50D SMA", "100D SMA", "200D SMA"]


def apply_filters(x: Ctx, long: np.ndarray, short: np.ndarray, fc: dict, oi_info: Optional[dict] = None):
    """Returns filtered long/short + per-filter state at the last bar (for live status board)."""
    L, S = long.copy(), short.copy()
    report: List[Tuple[str, str, str]] = []
    act = set(fc.get("active", []))
    look = int(fc.get("lookback", 10))

    def gate(name, lok, sok):
        nonlocal L, S
        lok = np.nan_to_num(np.asarray(lok, dtype=float)).astype(bool)
        sok = np.nan_to_num(np.asarray(sok, dtype=float)).astype(bool)
        L &= lok
        S &= sok
        report.append((name, "✅" if lok[-1] else "❌", "✅" if sok[-1] else "❌"))

    def recent(mask):
        return pd.Series(np.nan_to_num(mask.astype(float))).rolling(look, min_periods=1).max().to_numpy().astype(bool)

    if "ADX range" in act:
        _, _, ad = x.dmi(fc.get("adx_len", 14), fc.get("adx_len", 14))
        with np.errstate(invalid="ignore"):
            ok = (ad >= fc.get("adx_min", 20)) & (ad <= fc.get("adx_max", 60))
        gate(f"ADX {fc.get('adx_min', 20)}–{fc.get('adx_max', 60)} (now {fnum(ad[-1])})", ok, ok)
    sw = int(fc.get("swing", 5))
    sh, shp, slw, slp = x.swings(sw, sw)
    if "Break of Structure (BOS)" in act:
        bu = recent(cross_over(x.c, sh))
        bd = recent(cross_under(x.c, slw))
        gate("BOS", bu, bd)
    if "Change of Character (CHoCH)" in act:
        with np.errstate(invalid="ignore"):
            downtrend = (sh < shp) & (slw < slp)
            uptrend = (sh > shp) & (slw > slp)
        ch_up = recent(cross_over(x.c, sh) & shift(downtrend.astype(float)).astype(bool))
        ch_dn = recent(cross_under(x.c, slw) & shift(uptrend.astype(float)).astype(bool))
        gate("CHoCH", ch_up, ch_dn)
    if "Volume N×" in act:
        if np.nansum(x.v) == 0:
            report.append(("Volume N× (no volume data – skipped)", "➖", "➖"))
        else:
            av = sma(x.v, fc.get("vol_len", 20))
            with np.errstate(invalid="ignore"):
                ok = x.v >= fc.get("vol_mult", 1.5) * av
            gate(f"Volume ≥ {fc.get('vol_mult', 1.5)}× avg", ok, ok)
    if "Absolute OI" in act or "Change in OI" in act:
        if oi_info and oi_info.get("oi") is not None:
            ok_abs = oi_info["oi"] >= fc.get("oi_min", 0)
            chg = oi_info.get("oi_change", 0) or 0
            if fc.get("oi_chg_mode", "Absolute") == "Absolute":
                ok_chg = abs(chg) >= fc.get("oi_chg_min", 0)
            else:
                prev = oi_info.get("prev_oi") or 0
                ok_chg = prev > 0 and (oi_info["oi"] / prev) >= fc.get("oi_chg_mult", 1.1)
            for nm, ok in (("Absolute OI", ok_abs), ("Change in OI", ok_chg)):
                if nm in act:
                    arr = np.full(x.n, bool(ok))
                    gate(f"{nm} (live option chain)", arr, arr)
        else:
            report.append(("OI filters (needs Dhan option chain – skipped)", "➖", "➖"))
    if "Regime (trend) aligned" in act:
        t = x.ema(fc.get("regime_ema", 50))
        _, _, ad = x.dmi(14, 14)
        with np.errstate(invalid="ignore"):
            bull = (x.c > t) & (t > shift(t, 5)) & (ad >= fc.get("regime_adx", 20))
            bear = (x.c < t) & (t < shift(t, 5)) & (ad >= fc.get("regime_adx", 20))
        gate("Regime aligned", bull, bear)
    if "Supertrend" in act:
        _, d = x.supertrend(fc.get("st_factor", 3.0), fc.get("st_period", 10))
        gate("Supertrend", d == -1, d == 1)
    if "VWAP" in act:
        vw = x.vwap()
        with np.errstate(invalid="ignore"):
            gate("VWAP", x.c > vw, x.c < vw)
    if "MACD" in act:
        m, s, _ = x.macd()
        with np.errstate(invalid="ignore"):
            gate("MACD", m > s, m < s)
    if "Bollinger Bands" in act:
        b, u, lo = x.bb()
        with np.errstate(invalid="ignore"):
            gate("BB (above/below basis, inside bands)", (x.c > b) & (x.c < u), (x.c < b) & (x.c > lo))
    if "RSI" in act:
        r = x.rsi(fc.get("rsi_len", 14))
        with np.errstate(invalid="ignore"):
            gate(f"RSI >{fc.get('rsi_long', 50)} / <{fc.get('rsi_short', 50)}", r > fc.get("rsi_long", 50), r < fc.get("rsi_short", 50))
    if "Fair Value Gap" in act:
        with np.errstate(invalid="ignore"):
            bull = x.l > shift(x.h, 2)
            bear = x.h < shift(x.l, 2)
        gate("FVG (recent)", recent(bull), recent(bear))
    if "Liquidity Sweep" in act:
        with np.errstate(invalid="ignore"):
            sweep_dn = (x.l < shift(slw)) & (x.c > shift(slw))
            sweep_up = (x.h > shift(sh)) & (x.c < shift(sh))
        gate("Liquidity sweep (recent)", recent(sweep_dn), recent(sweep_up))
    if "Demand Zone Sustain" in act:
        _, _, _, pli = x.pivots(sw, sw)
        zl, zh = np.full(x.n, np.nan), np.full(x.n, np.nan)
        cl = ch = np.nan
        for i in range(x.n):
            if pli[i] >= 0:
                j = pli[i]
                cl, ch = x.l[j], max(x.o[j], x.c[j])
            zl[i], zh[i] = cl, ch
        with np.errstate(invalid="ignore"):
            ok = (x.c > zh) & (lowest(x.l, look) >= zl)
        gate("Demand zone sustained", ok, np.ones(x.n))
    if "Supply Zone Sustain" in act:
        _, _, phi, _ = x.pivots(sw, sw)
        zl, zh = np.full(x.n, np.nan), np.full(x.n, np.nan)
        cl = ch = np.nan
        for i in range(x.n):
            if phi[i] >= 0:
                j = phi[i]
                cl, ch = min(x.o[j], x.c[j]), x.h[j]
            zl[i], zh[i] = cl, ch
        with np.errstate(invalid="ignore"):
            ok = (x.c < zl) & (highest(x.h, look) <= zh)
        gate("Supply zone sustained", np.ones(x.n), ok)
    for nm, arr in (("EMA 20", lambda: x.ema(20)), ("SMA 20", lambda: x.sma(20)), ("EMA 63", lambda: x.ema(63))):
        if nm in act:
            a = arr()
            with np.errstate(invalid="ignore"):
                gate(f"{nm} (close above/below)", x.c > a, x.c < a)
    for nm, n_ in (("50D SMA", 50), ("100D SMA", 100), ("200D SMA", 200)):
        if nm in act:
            a = x.memo(("dsma", n_), lambda n_=n_: daily_sma_map(x.df, x.daily, n_) if x.daily is not None else sma(x.c, n_))
            with np.errstate(invalid="ignore"):
                gate(f"{nm} ({fnum(a[-1])})", x.c > a, x.c < a)

    # ---- crossover candle quality
    ctype = fc.get("cross_type", "Simple")
    body = np.abs(x.c - x.o)
    if ctype == "ATR based candle":
        a = x.atr(14)
        with np.errstate(invalid="ignore"):
            ok = body >= fc.get("atr_mult", 1.0) * a
        gate(f"Signal candle body ≥ {fc.get('atr_mult', 1.0)}×ATR", ok, ok)
    elif ctype == "Custom size candle":
        ok = body >= fc.get("candle_pts", 5.0)
        gate(f"Signal candle body ≥ {fc.get('candle_pts', 5.0)} pts", ok, ok)
    if fc.get("angle_on"):
        ma = x.ema(fc.get("angle_ema", 9))
        ang = ema_angle(ma, x.atr(14))
        with np.errstate(invalid="ignore"):
            ok = (ang >= fc.get("angle_min", 0.0)) & (ang <= fc.get("angle_max", 100.0))
        gate(f"Cross angle {fc.get('angle_min', 0)}°–{fc.get('angle_max', 100)}° (now {fnum(ang[-1])}°)", ok, ok)
    return L, S, report


# =============================================================================
#  RISK MANAGEMENT — Stop-loss & Target engines (shared by backtest and live)
# =============================================================================
SL_TYPES = ["Custom Points", "Fixed Percentage", "Trailing SL (points)", "Current Candle Low/High",
            "Previous Candle Low/High", "Trailing Current Candle Low/High", "Trailing Previous Candle Low/High",
            "Trailing Current Swing Low/High", "Trailing Previous Swing Low/High", "ATR Based",
            "Risk Reward (from Target)", "Strategy Based Exit", "Reverse EMA Crossover", "Step Trail (n / l / k)"]
TGT_TYPES = ["Custom Points", "Fixed Percentage", "Trailing Target (display only)", "Current Candle High/Low",
             "Previous Candle High/Low", "Trailing Current Candle High/Low", "Trailing Previous Candle High/Low",
             "Trailing Current Swing High/Low", "Trailing Previous Swing High/Low", "ATR Based",
             "Risk Reward (from SL)", "Strategy Based Exit", "Reverse EMA Crossover"]


@dataclass
class Position:
    direction: int                 # +1 long, -1 short (signal direction, after flip)
    entry: float
    entry_time: Any
    qty: int
    sl: float
    tgt: float
    init_sl: float
    init_tgt: float
    entry_idx: int = -1
    sig_idx: int = -1
    hi: float = float("nan")
    lo: float = float("nan")
    label: str = ""
    strategy: str = ""
    pending_exit: Optional[str] = None
    instrument: Dict[str, Any] = field(default_factory=dict)  # live: option/future contract details
    inst_entry: Optional[float] = None                         # live: option premium at entry
    inst_ltp: Optional[float] = None
    orders: List[dict] = field(default_factory=list)
    fixed_tgt: bool = False                                    # strategy-defined target (e.g. σ level) → never trailed
    entry_bar_ts: Any = None                                   # live: timestamp of the candle the entry happened in
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


def _swing_vals(x: Ctx, k: int, rc: dict):
    L = int(rc.get("swing_len", 5))
    sh, shp, sl_, slp = x.swings(L, L)
    return sh[k], shp[k], sl_[k], slp[k]


def _base_dist(kind: str, entry: float, x: Ctx, k: int, rc: dict, which: str) -> float:
    pts = rc["sl_pts"] if which == "sl" else rc["tgt_pts"]
    if kind in ("Custom Points", "Trailing SL (points)", "Trailing Target (display only)", "Step Trail (n / l / k)"):
        return float(pts if kind != "Step Trail (n / l / k)" else rc["step_n"])
    if kind == "Fixed Percentage":
        return entry * (rc["sl_pct"] if which == "sl" else rc["tgt_pct"]) / 100.0
    if kind == "ATR Based":
        a = x.atr(int(rc.get("atr_len", 14)))[k]
        m = rc["sl_atr"] if which == "sl" else rc["tgt_atr"]
        return float(a * m) if not np.isnan(a) else float(pts)
    return float(pts)


def init_levels(x: Ctx, k: int, direction: int, entry: float, rc: dict) -> Tuple[float, float]:
    """k = index of the 'current' (signal) candle. Returns (sl, target); NaN means no price level."""
    d = direction
    st, tt = rc["sl_type"], rc["tgt_type"]
    kp = max(k - 1, 0)
    sh, shp, swl, slp = _swing_vals(x, k, rc)

    def sl_from(kind):
        if kind in ("Strategy Based Exit", "Reverse EMA Crossover"):
            return np.nan
        if kind in ("Current Candle Low/High", "Trailing Current Candle Low/High"):
            v = x.l[k] if d == 1 else x.h[k]
        elif kind in ("Previous Candle Low/High", "Trailing Previous Candle Low/High"):
            v = x.l[kp] if d == 1 else x.h[kp]
        elif kind == "Trailing Current Swing Low/High":
            v = swl if d == 1 else sh
        elif kind == "Trailing Previous Swing Low/High":
            v = slp if d == 1 else shp
        elif kind == "Risk Reward (from Target)":
            if tt in ("Risk Reward (from SL)",) or tt in ("Strategy Based Exit", "Reverse EMA Crossover", "Trailing Target (display only)"):
                dist = rc["sl_pts"]
            else:
                dist = abs(tgt_from(tt, None) - entry) / max(rc["rr"], 1e-9)
            return entry - d * dist
        else:
            return entry - d * _base_dist(kind, entry, x, k, rc, "sl")
        # level must be on the protective side; fall back to custom points
        if np.isnan(v) or (d == 1 and v >= entry) or (d == -1 and v <= entry):
            return entry - d * rc["sl_pts"]
        return float(v)

    def tgt_from(kind, sl_val):
        if kind in ("Strategy Based Exit", "Reverse EMA Crossover"):
            return np.nan
        if kind in ("Current Candle High/Low", "Trailing Current Candle High/Low"):
            v = x.h[k] if d == 1 else x.l[k]
        elif kind in ("Previous Candle High/Low", "Trailing Previous Candle High/Low"):
            v = x.h[kp] if d == 1 else x.l[kp]
        elif kind == "Trailing Current Swing High/Low":
            v = sh if d == 1 else swl
        elif kind == "Trailing Previous Swing High/Low":
            v = shp if d == 1 else slp
        elif kind == "Risk Reward (from SL)":
            if sl_val is None or np.isnan(sl_val):
                dist = rc["tgt_pts"]
            else:
                dist = abs(entry - sl_val) * rc["rr"]
            return entry + d * dist
        else:
            return entry + d * _base_dist(kind, entry, x, k, rc, "tgt")
        if np.isnan(v) or (d == 1 and v <= entry) or (d == -1 and v >= entry):
            return entry + d * rc["tgt_pts"]
        return float(v)

    sl = sl_from(st)
    tgt = tgt_from(tt, sl)
    return float(sl), float(tgt)


def trail_on_bar(x: Ctx, k: int, pos: Position, rc: dict):
    """Called after candle k CLOSES while in position (moves only in the favourable direction)."""
    d = pos.direction
    st, tt = rc["sl_type"], rc["tgt_type"]
    better = (lambda new, old: new > old) if d == 1 else (lambda new, old: new < old)
    sh, shp, swl, slp = _swing_vals(x, k, rc)
    cand = None
    if st == "Trailing Current Candle Low/High":
        cand = x.l[k] if d == 1 else x.h[k]
    elif st == "Trailing Previous Candle Low/High":
        cand = x.l[k - 1] if d == 1 else x.h[k - 1]
    elif st == "Trailing Current Swing Low/High":
        cand = swl if d == 1 else sh
    elif st == "Trailing Previous Swing Low/High":
        cand = slp if d == 1 else shp
    if cand is not None and not np.isnan(cand) and not np.isnan(pos.sl) and better(cand, pos.sl):
        # never trail through the close of the bar that set it
        if (d == 1 and cand < x.c[k]) or (d == -1 and cand > x.c[k]):
            pos.sl = float(cand)
    tc = None
    if tt == "Trailing Current Candle High/Low":
        tc = x.h[k] if d == 1 else x.l[k]
    elif tt == "Trailing Previous Candle High/Low":
        tc = x.h[k - 1] if d == 1 else x.l[k - 1]
    elif tt == "Trailing Current Swing High/Low":
        tc = sh if d == 1 else swl
    elif tt == "Trailing Previous Swing High/Low":
        tc = shp if d == 1 else slp
    if tc is not None and not pos.fixed_tgt and not np.isnan(tc) and not np.isnan(pos.tgt) and better(tc, pos.tgt):
        pos.tgt = float(tc)
    # price-based trailing using the bar's extreme (applied from next bar → no look-ahead)
    trail_on_price(pos, x.h[k] if d == 1 else x.l[k], rc)


def trail_on_price(pos: Position, price: float, rc: dict):
    """Tick/extreme-price trailing: points trailing, step trail (n/l/k), display-only trailing target."""
    d = pos.direction
    if np.isnan(price):
        return
    pos.hi = price if np.isnan(pos.hi) else max(pos.hi, price)
    pos.lo = price if np.isnan(pos.lo) else min(pos.lo, price)
    fav = pos.hi if d == 1 else pos.lo
    st = rc["sl_type"]
    if st == "Trailing SL (points)":
        new = fav - d * rc["sl_pts"]
        if (d == 1 and new > pos.sl) or (d == -1 and new < pos.sl):
            pos.sl = float(new)
    elif st == "Step Trail (n / l / k)":
        n_, l_, k_ = rc["step_n"], rc["step_l"], rc["step_k"]
        move = (fav - pos.entry) * d
        new = None
        if move >= n_:
            new = pos.entry + d * (l_ + (move - n_))
        elif move >= k_:
            new = pos.entry + d * l_
        if new is not None and ((d == 1 and new > pos.sl) or (d == -1 and new < pos.sl)):
            pos.sl = float(new)
    if rc["tgt_type"] == "Trailing Target (display only)" and not pos.fixed_tgt:
        pos.tgt = float(fav + d * rc["tgt_pts"])


def strategy_exit_reason(x: Ctx, k: int, pos: Position, rc: dict, sig: dict) -> Optional[str]:
    d = pos.direction
    kinds = {rc["sl_type"], rc["tgt_type"]}
    if "Strategy Based Exit" in kinds and sig is not None:
        ex = sig["exit_long"] if d * rc.get("flip_sign", 1) == 1 else sig["exit_short"]
        if ex[k]:
            return "Strategy exit signal"
    if "Reverse EMA Crossover" in kinds:
        f, s = x.ema(int(rc.get("rev_fast", 9))), x.ema(int(rc.get("rev_slow", 21)))
        if d == 1 and cross_under(f, s)[k]:
            return "Reverse EMA crossover"
        if d == -1 and cross_over(f, s)[k]:
            return "Reverse EMA crossover"
    return None


# =============================================================================
#  CHARGES (SEBI / exchange / statutory) — all rates editable in the sidebar
# =============================================================================
DEFAULT_RATES = {
    "Equity Intraday": {"brok_pct": 0.03, "brok_cap": 20.0, "stt_buy": 0.0, "stt_sell": 0.025, "exch": 0.00297,
                        "stamp_buy": 0.003, "sebi_per_cr": 10.0, "gst": 18.0, "dp_sell": 0.0},
    "Equity Delivery": {"brok_pct": 0.0, "brok_cap": 0.0, "stt_buy": 0.1, "stt_sell": 0.1, "exch": 0.00297,
                        "stamp_buy": 0.015, "sebi_per_cr": 10.0, "gst": 18.0, "dp_sell": 12.5},
    "Futures": {"brok_pct": 0.03, "brok_cap": 20.0, "stt_buy": 0.0, "stt_sell": 0.05, "exch": 0.00173,
                "stamp_buy": 0.002, "sebi_per_cr": 10.0, "gst": 18.0, "dp_sell": 0.0},
    "Options": {"brok_pct": 100.0, "brok_cap": 20.0, "stt_buy": 0.0, "stt_sell": 0.15, "exch": 0.03503,
                "stamp_buy": 0.003, "sebi_per_cr": 10.0, "gst": 18.0, "dp_sell": 0.0},
    "Crypto/Forex (flat %)": {"brok_pct": 0.1, "brok_cap": 1e12, "stt_buy": 0.0, "stt_sell": 0.0, "exch": 0.0,
                              "stamp_buy": 0.0, "sebi_per_cr": 0.0, "gst": 0.0, "dp_sell": 0.0},
}


def calc_charges(segment: str, buy_value: float, sell_value: float, rates: dict) -> float:
    r = rates.get(segment) or DEFAULT_RATES[segment]
    brok = sum(min(v * r["brok_pct"] / 100.0, r["brok_cap"]) for v in (buy_value, sell_value) if v > 0)
    stt = buy_value * r["stt_buy"] / 100 + sell_value * r["stt_sell"] / 100
    exch = (buy_value + sell_value) * r["exch"] / 100
    sebi = (buy_value + sell_value) * r["sebi_per_cr"] / 1e7
    stamp = buy_value * r["stamp_buy"] / 100
    gst = (brok + exch + sebi) * r["gst"] / 100
    dp = r["dp_sell"] * (1 + r["gst"] / 100) if sell_value > 0 and r["dp_sell"] else 0.0
    return round(brok + stt + exch + sebi + stamp + gst + dp, 2)


def trade_charges(cfg: dict, direction_action: str, entry: float, exit_: float, qty: int,
                  inst_entry: Optional[float] = None, inst_exit: Optional[float] = None) -> float:
    if not cfg["costs_on"]:
        return 0.0
    seg = cfg["cost_segment"]
    if seg == "Options":
        pe = inst_entry if inst_entry else cfg["assumed_premium"]
        px = inst_exit if inst_exit else max(cfg["assumed_premium"] + (exit_ - entry) * (1 if "CE" in direction_action else -1), 0.05)
        return calc_charges(seg, pe * qty, px * qty, cfg["rates"])  # option buyer: buy premium, sell premium
    if direction_action.startswith("BUY"):
        return calc_charges(seg, entry * qty, exit_ * qty, cfg["rates"])
    return calc_charges(seg, exit_ * qty, entry * qty, cfg["rates"])


def action_label(direction: int, cfg: dict) -> str:
    """Index/stock options are BUY-only (long→CE, short→PE). Stocks/futures: long=BUY, short=SELL."""
    mode = cfg.get("trade_mode")
    if mode in ("Index Options", "Stock Options") or (mode is None and cfg.get("kind") == "index"):
        return "BUY CE" if direction == 1 else "BUY PE"
    return "BUY" if direction == 1 else "SELL"


# =============================================================================
#  BACKTEST ENGINE  (signal on candle N → entry on candle N+1 OPEN; conservative SL-first)
# =============================================================================
def _intraday_cutoff(cfg) -> Optional[Tuple[int, int]]:
    if cfg.get("auto_sqoff") and cfg["tf"] in INTRADAY_TFS and cfg.get("indian"):
        hh, mm = [int(v) for v in cfg["sqoff_time"].split(":")]
        return hh, mm
    return None


def _immediate_levels(x: Ctx, cfg: dict, sig: Optional[dict] = None):
    """Return per-bar threshold levels for immediate strategies: (up_level, up_dir, dn_level, dn_dir)."""
    if sig is not None and "imm_up" in sig:
        return sig["imm_up"], sig["imm_ud"], sig["imm_dn"], sig["imm_dd"]
    p = cfg["params"]
    n = x.n
    if cfg["strategy"] == "Price Threshold (Above/Below)":
        up = np.full(n, p["above_price"]) if p["above_price"] > 0 else np.full(n, np.nan)
        dn = np.full(n, p["below_price"]) if p["below_price"] > 0 else np.full(n, np.nan)
    else:
        pdl = x.memo("pdl", lambda: prev_day_levels(x.df))
        if cfg["tf"] in INTRADAY_TFS:
            ref = (pdl["C"] if p["ref"] == "Previous close" else pdl["dayO"]).to_numpy(float)
        else:
            ref = shift(x.c) if p["ref"] == "Previous close" else x.o
        up = ref * (1 + p["above_pct"] / 100) if p["above_pct"] > 0 else np.full(n, np.nan)
        dn = ref * (1 - p["below_pct"] / 100) if p["below_pct"] > 0 else np.full(n, np.nan)
    ud = 1 if p["above_action"] == "Buy" else -1
    dd = 1 if p["below_action"] == "Buy" else -1
    return up, ud, dn, dd


def _recommend(t: dict) -> str:
    d = t["_dir"]
    mfe = (t["Highest Price"] - t["Entry Price"]) if d == 1 else (t["Entry Price"] - t["Lowest Price"])
    mae = (t["Entry Price"] - t["Lowest Price"]) if d == 1 else (t["Highest Price"] - t["Entry Price"])
    tdist = abs(t["Target Price"] - t["Entry Price"]) if not np.isnan(t["Target Price"]) else np.nan
    pts = t["Points"]
    r = t["Exit Reason"]
    if "gap" in r.lower():
        return "Gap moved through SL — reduce overnight/gap exposure or size down"
    if r.startswith("SL") and not np.isnan(tdist) and mfe >= 0.6 * tdist:
        return f"Ran {mfe:,.1f} pts in favour before SL — use breakeven/step trailing SL"
    if r.startswith("SL") and mfe < 0.1 * max(abs(pts), 1e-9):
        return "Immediate adverse move — tighten entry filter (ADX / volume / regime)"
    if r.startswith("Target") and mfe > 1.5 * max(abs(pts), 1e-9):
        return f"Left {mfe - pts:,.1f} pts on table — consider trailing target / higher RR"
    if r.startswith("Target") and t["Bars Held"] <= 1:
        return "Target hit within 1 bar — target may be too tight"
    if pts < 0 and mae > 2 * abs(pts):
        return "Deep drawdown during trade — SL may be too wide"
    if pts > 0:
        return "Good trade — rules followed"
    return "Loss within plan — acceptable"


def is_target_active(pos: Position, rc: dict) -> bool:
    return not np.isnan(pos.tgt) and (pos.fixed_tgt or rc["tgt_type"] != "Trailing Target (display only)")


def strategy_target(sig: Optional[dict], k: int, d: int, price: float) -> Optional[float]:
    """Strategy-defined target (VIX σ exit). Valid only if it lies beyond the entry in the trade direction."""
    if sig is None or "tgt_long" not in sig or k < 0:
        return None
    tv = (sig["tgt_long"] if d == 1 else sig["tgt_short"])[k]
    if tv == tv and ((d == 1 and tv > price) or (d == -1 and tv < price)):
        return float(tv)
    return None


def sigma_target_blocked(sig: Optional[dict], k: int, d: int, price: float) -> bool:
    """σ-exit enabled but price already beyond the σ target (e.g. gap open) → skip the entry."""
    if sig is None or "tgt_long" not in sig or k < 0:
        return False
    tv = (sig["tgt_long"] if d == 1 else sig["tgt_short"])[k]
    return tv == tv and not ((d == 1 and tv > price) or (d == -1 and tv < price))


def make_aux(cfg: dict) -> dict:
    return {"tf": cfg["tf"], "kind": cfg.get("kind")}


def run_backtest(df: pd.DataFrame, cfg: dict, daily: Optional[pd.DataFrame] = None,
                 aux: Optional[dict] = None) -> Tuple[List[dict], dict, Ctx, dict]:
    x = Ctx(df, daily, dict(aux or {}, **make_aux(cfg)))
    n = x.n
    start = trim_to_period(df, cfg["period"])
    rc = cfg["risk"]
    name = cfg["strategy"]
    sd = STRATEGIES[name]
    immediate = name in IMMEDIATE
    simple = name in ("Simple Buy", "Simple Sell")
    sig = None
    longs, shorts = np.zeros(n, bool), np.zeros(n, bool)
    report = []
    if sd.func is not None or name.startswith("Hybrid"):
        sig = run_strategy(x, name, cfg["params"], cfg.get("hybrid"))
    if not immediate:
        longs, shorts, report = apply_filters(x, sig["long"], sig["short"], cfg["filters"])
    flip = -1 if cfg["filters"].get("flip") else 1
    rc = dict(rc, flip_sign=flip)
    cutoff = _intraday_cutoff(cfg)
    path_model = str(cfg.get("fill_model", "")).startswith("TradingView")
    max_day = int(cfg["params"].get("max_entries_day", 0) or 0) if immediate else 0
    day_count: Dict[Any, int] = {}
    qty = int(cfg["qty"])
    trades: List[dict] = []
    pos: Optional[Position] = None
    last_exit_bar = -10
    if immediate and not simple:
        up, ud, dn, dd = _immediate_levels(x, cfg, sig)

    def close_pos(k, price, reason):
        nonlocal pos, last_exit_bar
        d = pos.direction
        ei = pos.entry_idx
        pts = (price - pos.entry) * d
        act = action_label(d, cfg)
        gross = pts * qty
        ch = trade_charges(cfg, act, pos.entry, price, qty)
        trades.append({
            "Entry Time (IST)": fmt_ts(pos.entry_time), "Exit Time (IST)": fmt_ts(df.index[k]),
            "Signal": "LONG" if d == 1 else "SHORT", "Action": act,
            "Entry Price": round(pos.entry, 4), "Exit Price": round(price, 4),
            "SL Price": round(pos.init_sl, 4) if not np.isnan(pos.init_sl) else np.nan,
            "Final SL": round(pos.sl, 4) if not np.isnan(pos.sl) else np.nan,
            "Target Price": round(pos.init_tgt, 4) if not np.isnan(pos.init_tgt) else np.nan,
            "Entry Candle Open": x.o[ei], "Entry Candle High": x.h[ei], "Entry Candle Low": x.l[ei], "Entry Candle Close": x.c[ei],
            "Exit Candle Open": x.o[k], "Exit Candle High": x.h[k], "Exit Candle Low": x.l[k], "Exit Candle Close": x.c[k],
            "Highest Price": round(max(pos.hi, x.h[k]) if pos.hi == pos.hi else x.h[k], 4),
            "Lowest Price": round(min(pos.lo, x.l[k]) if pos.lo == pos.lo else x.l[k], 4),
            "Points": round(pts, 4), "Qty": qty, "PnL (Gross)": round(gross, 2), "Charges": ch,
            "PnL (Net)": round(gross - ch, 2), "Exit Reason": reason, "Bars Held": k - ei,
            "_dir": d, "_ei": ei, "_xi": k})
        trades[-1]["Recommendation"] = _recommend(trades[-1])
        pos = None
        last_exit_bar = k

    def open_pos(k, d, price, ref_k):
        nonlocal pos
        sl, tg = init_levels(x, ref_k, d, price, rc)
        st_t = strategy_target(sig, k, d, price)
        pos = Position(direction=d, entry=float(price), entry_time=df.index[k], qty=qty, sl=sl,
                       tgt=st_t if st_t is not None else tg, init_sl=sl, init_tgt=st_t if st_t is not None else tg,
                       entry_idx=k, sig_idx=ref_k, hi=float(price), lo=float(price), label=action_label(d, cfg),
                       strategy=name, fixed_tgt=st_t is not None)
        dk = df.index[k].date()
        day_count[dk] = day_count.get(dk, 0) + 1

    def check_exits(k) -> bool:
        """Conservative intrabar order: LONG → low vs SL first, then high vs target. SHORT mirrored."""
        d = pos.direction
        sl, tg = pos.sl, pos.tgt
        o, h, l = x.o[k], x.h[k], x.l[k]
        tgt_on = is_target_active(pos, rc)
        if d == 1:
            if not np.isnan(sl) and k > pos.entry_idx and o <= sl:
                close_pos(k, o, "SL hit (gap open)"); return True
            if not np.isnan(sl) and l <= sl:
                close_pos(k, sl, "SL hit"); return True
            if tgt_on:
                if k > pos.entry_idx and o >= tg:
                    close_pos(k, o, "Target hit (gap open)"); return True
                if h >= tg:
                    close_pos(k, tg, "Target hit"); return True
        else:
            if not np.isnan(sl) and k > pos.entry_idx and o >= sl:
                close_pos(k, o, "SL hit (gap open)"); return True
            if not np.isnan(sl) and h >= sl:
                close_pos(k, sl, "SL hit"); return True
            if tgt_on:
                if k > pos.entry_idx and o <= tg:
                    close_pos(k, o, "Target hit (gap open)"); return True
                if l <= tg:
                    close_pos(k, tg, "Target hit"); return True
        return False

    def check_exits_path(k) -> bool:
        """TradingView broker-emulator path: open → nearer extreme → other extreme → close.
        Tick-style trailing (points / step-trail) is updated along the path, so it behaves much closer to live."""
        d = pos.direction
        o, h, l, c = x.o[k], x.h[k], x.l[k], x.c[k]
        if k == pos.entry_idx and abs(pos.entry - o) > 1e-9:
            return check_exits(k)  # intrabar (immediate) entry: path before entry unknown → conservative
        if k > pos.entry_idx:
            if not np.isnan(pos.sl) and ((d == 1 and o <= pos.sl) or (d == -1 and o >= pos.sl)):
                close_pos(k, o, "SL hit (gap open)"); return True
            if is_target_active(pos, rc) and ((d == 1 and o >= pos.tgt) or (d == -1 and o <= pos.tgt)):
                close_pos(k, o, "Target hit (gap open)"); return True
        path = [o, h, l, c] if (h - o) <= (o - l) else [o, l, h, c]
        prev = path[0]
        for p1 in path[1:]:
            rising = p1 > prev
            tgt_on = is_target_active(pos, rc)
            if d == 1:
                if rising and tgt_on and p1 >= pos.tgt:
                    close_pos(k, pos.tgt, "Target hit"); return True
                if not rising and not np.isnan(pos.sl) and p1 <= pos.sl:
                    close_pos(k, pos.sl, "SL hit"); return True
            else:
                if rising and not np.isnan(pos.sl) and p1 >= pos.sl:
                    close_pos(k, pos.sl, "SL hit"); return True
                if not rising and tgt_on and p1 <= pos.tgt:
                    close_pos(k, pos.tgt, "Target hit"); return True
            trail_on_price(pos, p1, rc)
            prev = p1
        return False

    for k in range(max(start, 2), n):
        ts = df.index[k]
        if pos is not None and pos.pending_exit:
            close_pos(k, x.o[k], pos.pending_exit)
        if pos is None:
            blocked = bool(cutoff and (ts.hour, ts.minute) >= cutoff)
            if not immediate:
                j = k - 1  # signal on candle N=k-1 → entry at candle N+1=k open
                if j >= start - 1 and j > last_exit_bar - 1 and (longs[j] or shorts[j]):
                    stale = bool(cutoff) and ts.date() != df.index[j].date()
                    if not blocked and not stale:
                        open_pos(k, (1 if longs[j] else -1) * flip, x.o[k], j)
            elif simple:
                if k > last_exit_bar and not blocked:
                    open_pos(k, (1 if name == "Simple Buy" else -1) * flip, x.o[k], k - 1)
            elif not blocked and k > last_exit_bar and (not max_day or day_count.get(ts.date(), 0) < max_day):
                pc = x.c[k - 1]
                if not np.isnan(up[k]) and pc < up[k] <= x.h[k]:
                    px_ = max(x.o[k], up[k])
                    if not sigma_target_blocked(sig, k, ud * flip, px_):
                        open_pos(k, ud * flip, px_, k - 1)
                elif not np.isnan(dn[k]) and pc > dn[k] >= x.l[k]:
                    px_ = min(x.o[k], dn[k])
                    if not sigma_target_blocked(sig, k, dd * flip, px_):
                        open_pos(k, dd * flip, px_, k - 1)
        if pos is not None:
            if (check_exits_path(k) if path_model else check_exits(k)):
                continue
            if cutoff and (ts.hour, ts.minute) >= cutoff:
                close_pos(k, x.c[k], "Intraday auto square-off"); continue
            if cutoff and k + 1 < n and df.index[k + 1].date() != ts.date():
                close_pos(k, x.c[k], "Session end square-off"); continue
            pos.hi, pos.lo = max(pos.hi, x.h[k]), min(pos.lo, x.l[k])
            trail_on_bar(x, k, pos, rc)
            r = strategy_exit_reason(x, k, pos, rc, sig)
            if r:
                pos.pending_exit = r
    if pos is not None:
        close_pos(n - 1, x.c[n - 1], "End of data (open trade closed)")
    if immediate and not simple:
        sl_ = slice(start, n)
        nl = int(np.nansum((shift(x.c)[sl_] < up[sl_]) & (x.h[sl_] >= up[sl_])))
        ns = int(np.nansum((shift(x.c)[sl_] > dn[sl_]) & (x.l[sl_] <= dn[sl_])))
    else:
        nl, ns = int(longs[start:].sum()), int(shorts[start:].sum())
    meta = {"start": start, "signals_long": nl, "signals_short": ns, "filter_report": report, "sig": sig}
    return trades, summarize(trades), x, meta


def summarize(trades: List[dict]) -> dict:
    if not trades:
        return {"Total Trades": 0}
    t = pd.DataFrame(trades)
    pts = t["Points"]
    wins, losses = t[pts > 0], t[pts <= 0]
    gross_win = wins["PnL (Gross)"].sum()
    gross_loss = -losses["PnL (Gross)"].sum()
    eq = t["PnL (Net)"].cumsum()
    dd = (eq - eq.cummax()).min()
    try:
        daily = t.groupby(pd.to_datetime(t["Exit Time (IST)"]).dt.date)["PnL (Net)"].sum()
        sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 and daily.std(ddof=1) > 0 else float("nan")
    except Exception:
        sharpe = float("nan")
    avg_w = wins["Points"].mean() if len(wins) else 0.0
    avg_l = abs(losses["Points"].mean()) if len(losses) else 0.0
    return {
        "Total Trades": len(t), "Winning Trades": len(wins), "Losing Trades": len(losses),
        "Points Won": round(wins["Points"].sum(), 2), "Points Lost": round(losses["Points"].sum(), 2),
        "Net Points": round(pts.sum(), 2), "Accuracy %": round(100 * len(wins) / len(t), 2),
        "Sharpe Ratio": round(sharpe, 2) if sharpe == sharpe else "—",
        "Profit Factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else ("∞" if gross_win > 0 else "—"),
        "Risk:Reward (avg)": f"1:{avg_w / avg_l:.2f}" if avg_l > 0 else "—",
        "Max Drawdown (₹)": round(float(dd), 2), "Total Charges": round(t["Charges"].sum(), 2),
        "PnL Before Charges": round(t["PnL (Gross)"].sum(), 2), "PnL After Charges": round(t["PnL (Net)"].sum(), 2),
        "Avg Win (pts)": round(avg_w, 2), "Avg Loss (pts)": round(avg_l, 2),
    }


# =============================================================================
#  PERSISTENCE (multi-user: every trade row is keyed by Trader ID)
# =============================================================================
_db_lock = threading.Lock()


def _db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    con.execute("CREATE TABLE IF NOT EXISTS trades (id TEXT PRIMARY KEY, user TEXT, ts TEXT, data TEXT)")
    return con


def db_save_trade(user: str, rec: dict):
    with _db_lock:
        con = _db()
        con.execute("INSERT OR REPLACE INTO trades VALUES (?,?,?,?)",
                    (rec["Trade ID"], user, rec.get("Exit Time (IST)", ""), json.dumps(rec, default=str)))
        con.commit()
        con.close()


def db_load_trades(user: str) -> List[dict]:
    with _db_lock:
        con = _db()
        rows = con.execute("SELECT data FROM trades WHERE user=? ORDER BY ts", (user,)).fetchall()
        con.close()
    return [json.loads(r[0]) for r in rows]


def db_clear(user: str):
    with _db_lock:
        con = _db()
        con.execute("DELETE FROM trades WHERE user=?", (user,))
        con.commit()
        con.close()


# =============================================================================
#  NOTIFICATIONS (non-blocking)
# =============================================================================
def _render_tpl(tpl: str, ctx: dict) -> str:
    try:
        return tpl.format(**{k: (fnum(v) if isinstance(v, float) else v) for k, v in ctx.items()})
    except Exception:
        return tpl + "\n" + "\n".join(f"{k}: {v}" for k, v in ctx.items())


def notify(cfg: dict, event: str, ctx: dict):
    ctx = dict(ctx, event=event, time=now_ist().strftime("%Y-%m-%d %H:%M:%S"))
    nc = cfg["notify"]

    def _email():
        try:
            msg = MIMEText(_render_tpl(nc["email_body"], ctx))
            msg["Subject"] = _render_tpl(nc["email_subject"], ctx)
            msg["From"] = nc["email_from"]
            msg["To"] = nc["email_to"]
            with smtplib.SMTP_SSL(nc["smtp_host"], int(nc["smtp_port"]), timeout=15) as s:
                s.login(nc["email_from"], nc["email_pass"])
                s.sendmail(nc["email_from"], [t.strip() for t in nc["email_to"].split(",")], msg.as_string())
        except Exception as e:
            _log_global(f"Email failed: {e}")

    def _wa():
        try:
            text = _render_tpl(nc["wa_body"], ctx)
            if nc["wa_provider"] == "CallMeBot":
                requests.get("https://api.callmebot.com/whatsapp.php",
                             params={"phone": nc["wa_to"], "text": text, "apikey": nc["wa_apikey"]}, timeout=15)
            else:
                requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{nc['tw_sid']}/Messages.json",
                              auth=(nc["tw_sid"], nc["tw_token"]),
                              data={"From": f"whatsapp:{nc['wa_from']}", "To": f"whatsapp:{nc['wa_to']}", "Body": text},
                              timeout=15)
        except Exception as e:
            _log_global(f"WhatsApp failed: {e}")

    if nc.get("email_on") and nc.get("email_from") and nc.get("email_to"):
        threading.Thread(target=_email, daemon=True).start()
    if nc.get("wa_on") and nc.get("wa_to"):
        threading.Thread(target=_wa, daemon=True).start()


_GLOBAL_ERRORS: List[str] = []


def _log_global(msg: str):
    _GLOBAL_ERRORS.append(f"{now_ist():%H:%M:%S} {msg}")
    del _GLOBAL_ERRORS[:-50]


# =============================================================================
#  CHARTS
# =============================================================================
PALETTE = ["#f5b700", "#00a6fb", "#e36414", "#9b5de5", "#00f5d4", "#f15bb5", "#8ac926", "#ff595e"]


def candle_chart(df: pd.DataFrame, lines: Dict[str, np.ndarray], title: str, trades: Optional[List[dict]] = None,
                 pos: Optional[Position] = None, height: int = 620, max_bars: Optional[int] = None):
    if go is None or df.empty:
        return None
    off = 0
    if max_bars and len(df) > max_bars:
        off = len(df) - max_bars
        df = df.iloc[off:]
        lines = {k: np.asarray(v)[off:] for k, v in lines.items()}
    xs = [t.strftime("%d-%b %H:%M") if (t.hour or t.minute) else t.strftime("%d-%b-%y") for t in df.index]
    lo, hi = df["Low"].min(), df["High"].max()
    rng = hi - lo if hi > lo else abs(hi) or 1.0
    overlay, osc = {}, {}
    for k, v in lines.items():
        v = np.asarray(v, float)
        if np.all(np.isnan(v)):
            continue
        med = np.nanmedian(v)
        (overlay if lo - rng <= med <= hi + rng else osc)[k] = v
    rows = 2 if osc else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.75, 0.25] if rows == 2 else [1.0])
    fig.add_trace(go.Candlestick(x=xs, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                                 increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"), row=1, col=1)
    for i, (k, v) in enumerate(overlay.items()):
        fig.add_trace(go.Scatter(x=xs, y=v, name=k, mode="lines", line=dict(width=1.6, color=PALETTE[i % len(PALETTE)])), row=1, col=1)
    for i, (k, v) in enumerate(osc.items()):
        fig.add_trace(go.Scatter(x=xs, y=v, name=k, mode="lines", line=dict(width=1.3, color=PALETTE[(i + 3) % len(PALETTE)])), row=2, col=1)
    if trades:
        for side, col, sym in ((1, "#00e676", "triangle-up"), (-1, "#ff1744", "triangle-down")):
            ex = [t for t in trades if t["_dir"] == side and t["_ei"] - off >= 0]
            if ex:
                fig.add_trace(go.Scatter(x=[xs[t["_ei"] - off] for t in ex], y=[t["Entry Price"] for t in ex], mode="markers",
                                         name="Long entry" if side == 1 else "Short entry",
                                         marker=dict(symbol=sym, size=12, color=col, line=dict(width=1, color="#111"))), row=1, col=1)
        xt = [t for t in trades if t["_xi"] - off >= 0]
        if xt:
            fig.add_trace(go.Scatter(x=[xs[t["_xi"] - off] for t in xt], y=[t["Exit Price"] for t in xt], mode="markers", name="Exit",
                                     marker=dict(symbol="x", size=9, color=["#00e676" if t["Points"] > 0 else "#ff1744" for t in xt]),
                                     text=[t["Exit Reason"] for t in xt], hovertemplate="%{text}<br>%{y}"), row=1, col=1)
    if pos is not None:
        for val, nm, col in ((pos.entry, "Entry", "#00a6fb"), (pos.sl, "SL", "#ff1744"), (pos.tgt, "Target", "#00e676")):
            if val == val and val is not None:
                fig.add_hline(y=val, line=dict(color=col, width=1.4, dash="dash"), annotation_text=f"{nm} {val:,.2f}",
                              annotation_position="right", row=1, col=1)
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15)), height=height, template="plotly_dark",
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=70, t=45, b=10), legend=dict(orientation="h", y=1.02, x=0.35, font=dict(size=11)),
                      hovermode="x unified", uirevision="keep")
    fig.update_xaxes(type="category", nticks=12, showgrid=False)
    fig.update_yaxes(gridcolor="#1f2633", side="right")
    return fig


# =============================================================================
#  EXPORT HELPERS
# =============================================================================
def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nm, d in sheets.items():
            d.to_excel(w, sheet_name=nm[:31], index=False)
    return buf.getvalue()


def export_buttons(df: pd.DataFrame, summary: dict, prefix: str):
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Export CSV", df.to_csv(index=False).encode(), f"{prefix}_{now_ist():%Y%m%d_%H%M}.csv",
                       "text/csv", **WIDE, key=f"m_{prefix}_csv")
    try:
        xb = to_excel_bytes({"Trades": df, "Summary": pd.DataFrame([summary]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})})
        c2.download_button("⬇️ Export Excel", xb, f"{prefix}_{now_ist():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           **WIDE, key=f"m_{prefix}_xlsx")
    except Exception as e:
        c2.caption(f"Excel export unavailable ({e}); install openpyxl")


# =============================================================================
#  STYLES
# =============================================================================
CSS = """
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1500px}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#141a24,#0f141c);border:1px solid #232c3b;
  border-radius:12px;padding:10px 14px}
div[data-testid="stMetric"] label, div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] *{color:#9fb0c8 !important;font-size:.78rem}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] *{font-size:1.25rem;color:#f1f5fb !important}
.qe-hero{background:linear-gradient(90deg,#0b3d91 0%,#1565c0 45%,#00a6fb 100%);padding:14px 20px;border-radius:14px;
  color:white;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
.qe-hero h1{font-size:1.45rem;margin:0;color:white}
.qe-hero span{opacity:.9;font-size:.85rem}
.qe-badge{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:700;font-size:.9rem;letter-spacing:.3px}
.qe-on{background:#0f5132;color:#75f0a8;border:1px solid #1e8e57}
.qe-short{background:#5c1a1a;color:#ff9a9a;border:1px solid #b23b3b}
.qe-off{background:#1f2633;color:#aab4c3;border:1px solid #33405a}
.qe-run{background:#0b3d91;color:#cfe3ff;border:1px solid #1565c0}
.qe-card{background:#111722;border:1px solid #232c3b;border-radius:12px;padding:10px 14px;margin-bottom:8px;color:#e6edf7}
.qe-kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:6px 14px;font-size:.86rem}
.qe-kv b{color:#9fb0c8;font-weight:500}
.qe-kv div{color:#e6edf7}
</style>
"""


# =============================================================================
#  SIDEBAR → CONFIG
# =============================================================================
def _seed(key: str, default, options: Optional[list] = None):
    """Defaults live in session_state (never passed to the widget) → values can be pushed from the screener safely."""
    ss = st.session_state
    if key not in ss or (options is not None and ss[key] not in options):
        ss[key] = default


def w_num(c, label, key, default, **kw):
    _seed(key, default)
    return c.number_input(label, key=key, **kw)


def w_sel(c, label, options, key, default=None, **kw):
    options = list(options)
    _seed(key, options[0] if default is None else default, options)
    return c.selectbox(label, options, key=key, **kw)


def w_cb(c, label, key, default=False, **kw):
    _seed(key, bool(default))
    return c.checkbox(label, key=key, **kw)


def w_txt(c, label, key, default="", **kw):
    _seed(key, default)
    return c.text_input(label, key=key, **kw)


def w_area(c, label, key, default="", **kw):
    _seed(key, default)
    return c.text_area(label, key=key, **kw)


def _param_widgets(name: str, prefix: str, c=None) -> dict:
    c = c or st
    out = {}
    for k, spec in STRATEGIES[name].params.items():
        key = f"{prefix}_{name}_{k}"
        lab = k.replace("_", " ").title()
        if spec["type"] == "int":
            out[k] = int(w_num(c, lab, key, int(spec["default"]), min_value=spec["min"], max_value=spec["max"], step=1))
        elif spec["type"] == "float":
            out[k] = float(w_num(c, lab, key, float(spec["default"]), min_value=None if spec["min"] is None else float(spec["min"]),
                                 max_value=None if spec["max"] is None else float(spec["max"]), step=float(spec["step"] or 0.1),
                                 help=spec["help"]))
        elif spec["type"] == "bool":
            out[k] = w_cb(c, lab, key, spec["default"])
        else:
            out[k] = w_sel(c, lab, spec["options"], key, spec["default"], help=spec["help"])
    return out


def _vix_widgets(prefix: str) -> dict:
    sp = STRATEGIES[VIX_NAME].params
    k = lambda n: f"{prefix}_{VIX_NAME}_{n}"
    p: Dict[str, Any] = {}
    st.caption("Expected Move = Prev Close × Vol ÷ Factor.  ±1σ/±2σ/±3σ = Close ± EM × 1/2/3.  "
               "Vol = India VIX for NSE indices; for stocks/crypto/forex/commodities use their implied vol if you have it "
               "(manual) or HV. Factor = 100×√(trading days/yr): 1600 (NIFTY convention, √256), ≈1587 (√252), ≈1911 crypto (√365).")
    p["vol_source"] = w_sel(st, "Volatility source", VOL_SOURCES, k("vol_source"))
    c1, c2 = st.columns(2)
    p["manual_vix"] = float(w_num(c1, "Manual VIX / vol %", k("manual_vix"), 0.0, min_value=0.0, max_value=500.0, step=0.05,
                                  help=sp["manual_vix"]["help"]))
    p["factor"] = float(w_num(c2, "Factor", k("factor"), 1600.0, min_value=1.0, max_value=100000.0, step=1.0, help=sp["factor"]["help"]))
    c1, c2 = st.columns(2)
    p["hv_len"] = int(w_num(c1, "HV length (days)", k("hv_len"), 20, min_value=2, max_value=500, step=1))
    p["days_per_year"] = int(w_num(c2, "Trading days / year", k("days_per_year"), 252, min_value=1, max_value=366, step=1))
    p["move_entry"] = w_cb(st, "Enable move-based entry (direction decided by the move)", k("move_entry"), False)
    if p["move_entry"]:
        p["move_type"] = w_sel(st, "Trigger", ["Absolute points", "% of 1σ move"], k("move_type"))
        c1, c2 = st.columns(2)
        p["move_pts"] = float(w_num(c1, "Move points", k("move_pts"), 120.0, min_value=0.01, max_value=1e7, step=1.0))
        p["move_pct"] = float(w_num(c2, "Move % of 1σ", k("move_pct"), 70.0, min_value=1.0, max_value=300.0, step=1.0))
        p["anchor"] = w_sel(st, "Measure move from", ["Previous close", "Day open"], k("anchor"))
    else:
        c1, c2 = st.columns(2)
        p["base_mode"] = w_sel(c1, "Base entry", ["Breakout beyond σ", "Reversal at σ (fade)"], k("base_mode"))
        p["entry_sigma"] = w_sel(c2, "Entry at ±σ", ["1", "2", "3"], k("entry_sigma"))
    p["sigma_exit"] = w_cb(st, "Exit near σ level (else sidebar SL/Target)", k("sigma_exit"), False)
    if p["sigma_exit"]:
        c1, c2 = st.columns(2)
        p["exit_level"] = w_sel(c1, "Exit level", ["±1σ", "±2σ", "±3σ", "Prev close"], k("exit_level"))
        p["exit_buffer"] = float(w_num(c2, "Exit buffer pts (near)", k("exit_buffer"), 5.0, min_value=0.0, max_value=1e6, step=0.5))
    c1, c2 = st.columns(2)
    p["max_entries_day"] = int(w_num(c1, "Max entries / day", k("max_entries_day"), 1, min_value=1, max_value=50, step=1))
    p["allow"] = w_sel(c2, "Direction", ["Both", "Long only", "Short only"], k("allow"))
    full = {kk: v["default"] for kk, v in sp.items()}
    full.update(p)
    full["entry_sigma"] = str(full["entry_sigma"])
    return full


def _vix_sidebar_preview(p: dict, meta: dict, symbol: str):
    """Show the India VIX value fetched and the value that will be used, right under the strategy settings."""
    mode, manual = p["vol_source"], float(p["manual_vix"])
    lines = []
    if mode.startswith("Manual"):
        lines.append(f"Using **manual vol {fnum(manual if manual > 0 else 15.0)}** for all bars.")
    elif mode.startswith("Auto") and meta.get("kind") == "index" and is_indian(symbol, "index"):
        try:
            v = get_daily("^INDIAVIX", 30)["Close"].dropna()
            today = now_ist().date()
            last_d, last_v = v.index[-1].date(), float(v.iloc[-1])
            prior = v[[d.date() < today for d in v.index]]
            lines.append(f"📊 **India VIX fetched:** {fnum(last_v)} ({last_d:%d-%b-%Y})")
            if len(prior):
                lines.append(f"Today's bands use **{fnum(manual if manual > 0 else float(prior.iloc[-1]))}**"
                             + (" (manual override)" if manual > 0 else f" ({prior.index[-1].date():%d-%b} close)"))
            lines.append(f"Next-session bands use **{fnum(manual if manual > 0 else last_v)}**"
                         + (" (manual override)" if manual > 0 else ""))
        except Exception as e:
            lines.append(f"⚠️ India VIX fetch failed ({str(e)[:60]}) — enter a manual value above (HV used meanwhile).")
    else:
        lines.append("Vol = historical volatility of this ticker (computed from its daily closes; shown on Backtest/Live).")
        if manual > 0 and mode.startswith("Auto"):
            lines.append(f"Manual override **{fnum(manual)}** applies to the latest session.")
    st.info("  \n".join(lines))


def infer_kind(sym: str) -> str:
    s = sym.upper()
    if s.endswith("-USD") or s.endswith("-USDT") or s.endswith("-INR"):
        return "crypto"
    if s.endswith("=X"):
        return "forex"
    if s.endswith("=F"):
        return "commodity"
    if s.startswith("^") or s.startswith("NIFTY_"):
        return "index_other"
    return "stock"


def _sidebar_spot(symbol: str) -> Optional[float]:
    L = st.session_state.get("live") or {}
    if L.get("ltp") and symbol in str(L.get("data_key", "")):
        return float(L["ltp"])
    try:
        return get_spot(symbol)
    except Exception:
        return None


def _strike_widgets(sb, symbol: str, meta: dict, stock: bool) -> dict:
    """CE/PE strike pickers that show real strike prices; each leg also has an editable strike-price box.
    Untouched box → follows the dropdown (re-resolved from live spot at entry). Edited box → fixed manual strike."""
    out: Dict[str, Any] = {}
    spot = _sidebar_spot(symbol)
    und, exch = meta.get("und", ""), meta.get("fno_exch", "NSE")
    lad = option_strike_ladder(und, exch, spot, stock) if spot else None
    choices = STRIKE_CHOICES[:-1]
    ss = st.session_state
    c1, c2 = sb.columns(2)
    for leg, col in (("CE", c1), ("PE", c2)):
        lk = leg.lower()
        if lad is not None:
            fmt = (lambda o, leg=leg: f"{o} · {strike_for_label(o, leg, lad['strikes'], spot):,.0f}")
        else:
            fmt = str
        choice = w_sel(col, f"{leg} strike", choices, f"{lk}_strike", "ATM", format_func=fmt)
        if lad is None:
            out[f"{lk}_strike"], out[f"{lk}_manual"] = choice, None
            continue
        auto_val = strike_for_label(choice, leg, lad["strikes"], spot)
        key = f"{lk}_px"
        if key not in ss or ss.get(f"_{lk}_choice") != choice or abs(float(ss.get(key) or 0) - float(ss.get(f"_{lk}_auto") or -1)) < 1e-9:
            ss[key] = float(auto_val)          # not edited by user → follow dropdown / spot
        ss[f"_{lk}_auto"], ss[f"_{lk}_choice"] = float(auto_val), choice
        val = col.number_input(f"{leg} strike price ✏️", key=key, min_value=0.0, step=float(lad["step"]),
                               help="Edit to fix a manual strike. Leave as-is to auto-follow the dropdown (ATM moves with spot).")
        manual = float(val) if abs(float(val) - auto_val) > 1e-9 else None
        out[f"{lk}_strike"] = "Manual" if manual else choice
        out[f"{lk}_manual"] = manual
    if lad is not None:
        ce_txt = f"{out['ce_manual']:,.0f} (manual)" if out.get("ce_manual") else f"{ss['_ce_auto']:,.0f} ({out['ce_strike']}, auto)"
        pe_txt = f"{out['pe_manual']:,.0f} (manual)" if out.get("pe_manual") else f"{ss['_pe_auto']:,.0f} ({out['pe_strike']}, auto)"
        sb.caption(f"Spot {fnum(spot)} · step {fnum(lad['step'], 1)} · expiry {lad['expiry'] or 'nearest'} · "
                   f"**CE {ce_txt}** · **PE {pe_txt}** · {lad['source']}. Auto strikes are re-resolved from live spot at entry.")
    else:
        sb.warning("Spot price unavailable — strike values can't be shown; ATM-relative selection will still resolve at entry.")
    return out


def sidebar_config() -> dict:
    sb = st.sidebar
    sb.markdown(f"## ⚡ {APP_NAME}")
    user = w_txt(sb, "👤 Trader ID (multi-user workspace)", "trader_id", "default").strip() or "default"
    use_dhan_data = w_cb(sb, "Use Dhan as data source (default: yfinance)", "use_dhan_data", False)
    cfg: Dict[str, Any] = {"user": user}

    # ------------------------------------------------------------------ instrument
    sb.markdown("### 📈 Instrument")
    tsel = w_sel(sb, "Ticker", TICKER_MENU, "tsel", "NIFTY 50")
    meta: Dict[str, Any]
    if tsel in TICKER_GROUPS:
        meta = dict(TICKER_GROUPS[tsel])
        symbol = meta["yf"]
    elif tsel == "Forex":
        pair = w_sel(sb, "Forex pair", list(FOREX_PAIRS), "fx")
        symbol, meta = FOREX_PAIRS[pair], {"kind": "forex"}
    elif tsel == "Commodities":
        com = w_sel(sb, "Commodity", list(COMMODITIES), "com")
        symbol, meta = COMMODITIES[com], {"kind": "commodity"}
    elif tsel == "Custom Ticker":
        symbol = w_txt(sb, "Yahoo symbol", "custom_sym", "KAYNES.NS").strip().upper() or "KAYNES.NS"
        meta = {"kind": infer_kind(symbol), "und": symbol.split(".")[0], "fno_exch": "BSE" if symbol.endswith(".BO") else "NSE"}
    else:  # Options Trading
        und = w_sel(sb, "Option underlying", ["NIFTY 50", "BANK NIFTY", "SENSEX", "FIN NIFTY", "Stock (custom)"], "opt_und")
        if und == "Stock (custom)":
            symbol = w_txt(sb, "Underlying stock (Yahoo)", "opt_stock", "KAYNES.NS").strip().upper() or "KAYNES.NS"
            meta = {"kind": "stock", "und": symbol.split(".")[0], "fno_exch": "NSE", "options_only": True}
        else:
            meta = dict(TICKER_GROUPS[und], options_only=True)
            symbol = meta["yf"]
    cfg.update(ticker_name=tsel, symbol=symbol, meta=meta, kind=meta.get("kind"), indian=is_indian(symbol, meta.get("kind", "")))

    c1, c2 = sb.columns(2)
    tf = w_sel(c1, "Timeframe", list(TF_PERIODS), "tf", "1m")
    plist, pdef = TF_PERIODS[tf]
    period = w_sel(c2, "Period", plist, f"period_{tf}", pdef)
    if tf in ("1h", "4h") and period == "3y" and not use_dhan_data:
        sb.caption("ℹ️ yfinance serves ≤730 days of hourly data; 3y uses the maximum available.")
    cfg.update(tf=tf, period=period)

    # ------------------------------------------------------------------ strategy
    sb.markdown("### 🧠 Strategy")
    names = list(STRATEGIES)
    strat = w_sel(sb, "Strategy", names, "strategy", "EMA/SMA Crossover",
                  help="Signals fire only on the bar the condition JUST became true (fresh cross).")
    if STRATEGIES[strat].desc:
        sb.caption(STRATEGIES[strat].desc)
    hybrid = None
    with sb.expander("Strategy parameters", expanded=True):
        if strat == VIX_NAME:
            params = _vix_widgets("sp")
            _vix_sidebar_preview(params, meta, symbol)
        else:
            params = _param_widgets(strat, "sp") if STRATEGIES[strat].params else {}
        if strat.startswith("Hybrid"):
            st.caption("Tick the strategies that must agree:")
            members = [m for m in SIGNAL_STRATS if w_cb(st, m, f"hy_{m}", m in ("EMA/SMA Crossover", "RSI 60/40"))]
            mode = w_sel(st, "Consensus", ["All agree", "Majority"], "hy_mode")
            window = w_num(st, "Agreement window (bars)", "hy_win", 3, min_value=1, max_value=20, step=1,
                           help="Each member's signal stays valid for N bars; fires when consensus is reached on a fresh signal.")
            hp = {}
            for m in members:
                if STRATEGIES[m].params:
                    st.markdown(f"**{m}**")
                    hp[m] = _param_widgets(m, "hy")
            hybrid = {"members": members, "mode": mode, "window": int(window), "params": hp}
    cfg.update(strategy=strat, params=params, hybrid=hybrid)

    # ------------------------------------------------------------------ risk
    sb.markdown("### 🛡️ Risk Management")
    rc: Dict[str, Any] = {}
    sl_type = w_sel(sb, "Stop-loss type", SL_TYPES, "sl_type")
    tgt_type = w_sel(sb, "Target type", TGT_TYPES, "tgt_type")
    with sb.expander("SL / Target parameters", expanded=True):
        a, b = st.columns(2)
        rc["sl_pts"] = w_num(a, "SL points", "sl_pts", 10.0, min_value=0.01, max_value=1e7, step=0.5)
        rc["tgt_pts"] = w_num(b, "Target points", "tgt_pts", 20.0, min_value=0.01, max_value=1e7, step=0.5)
        if "Percentage" in sl_type or "Percentage" in tgt_type:
            rc["sl_pct"] = w_num(a, "SL %", "sl_pct", 1.0, min_value=0.01, max_value=100.0, step=0.05)
            rc["tgt_pct"] = w_num(b, "Target %", "tgt_pct", 2.0, min_value=0.01, max_value=1000.0, step=0.05)
        if "ATR" in sl_type or "ATR" in tgt_type:
            rc["atr_len"] = w_num(st, "ATR length", "atr_len", 14, min_value=1, max_value=500, step=1)
            rc["sl_atr"] = w_num(a, "SL × ATR", "sl_atr", 1.5, min_value=0.1, max_value=50.0, step=0.1)
            rc["tgt_atr"] = w_num(b, "Target × ATR", "tgt_atr", 3.0, min_value=0.1, max_value=100.0, step=0.1)
        if "Risk Reward" in sl_type or "Risk Reward" in tgt_type:
            rr_s = w_sel(st, "Risk : Reward", ["1:1", "1:1.5", "1:2", "1:2.5", "1:3", "1:4", "1:5", "1:10"], "rr", "1:2")
            rc["rr"] = float(rr_s.split(":")[1])
        if "Swing" in sl_type or "Swing" in tgt_type:
            rc["swing_len"] = w_num(st, "Swing pivot strength (bars each side)", "swing_len", 5, min_value=1, max_value=50, step=1)
        if sl_type == "Step Trail (n / l / k)":
            st.caption("Entry 50, n=10 → SL 40. At entry+k (55) SL → entry+l (50). Above entry+n (60) SL trails 1:1 (61→51).")
            s1, s2, s3 = st.columns(3)
            rc["step_n"] = w_num(s1, "n (SL pts)", "step_n", 10.0, min_value=0.01, max_value=1e7, step=0.5)
            rc["step_l"] = w_num(s2, "l (lock pts)", "step_l", 0.0, min_value=-1e7, max_value=1e7, step=0.5)
            rc["step_k"] = w_num(s3, "k (trigger pts)", "step_k", 5.0, min_value=0.0, max_value=1e7, step=0.5)
        if "Reverse EMA" in sl_type or "Reverse EMA" in tgt_type:
            rc["rev_fast"] = w_num(a, "Reverse EMA fast", "rev_fast", 9, min_value=1, max_value=500, step=1)
            rc["rev_slow"] = w_num(b, "Reverse EMA slow", "rev_slow", 21, min_value=2, max_value=1000, step=1)
    rc = {"sl_pct": 1.0, "tgt_pct": 2.0, "atr_len": 14, "sl_atr": 1.5, "tgt_atr": 3.0, "rr": 2.0, "swing_len": 5,
          "step_n": rc.get("sl_pts", 10.0), "step_l": 0.0, "step_k": 5.0, "rev_fast": 9, "rev_slow": 21, **rc}
    rc.update(sl_type=sl_type, tgt_type=tgt_type)
    if sl_type == "Step Trail (n / l / k)":
        rc["sl_pts"] = rc["step_n"]
    cfg["risk"] = rc

    # ------------------------------------------------------------------ filters
    sb.markdown("### 🔍 Entry Filters")
    fc: Dict[str, Any] = {"active": []}
    with sb.expander("Additional entry filters (all off by default)", expanded=False):
        for f in FILTER_LIST:
            if w_cb(st, f, f"flt_{f}", False):
                fc["active"].append(f)
        act = set(fc["active"])
        if "ADX range" in act:
            c1, c2, c3 = st.columns(3)
            fc["adx_min"] = w_num(c1, "ADX min", "adx_min", 20.0, min_value=0.0, max_value=100.0)
            fc["adx_max"] = w_num(c2, "ADX max", "adx_max", 60.0, min_value=0.0, max_value=100.0)
            fc["adx_len"] = w_num(c3, "ADX len", "adx_len", 14, min_value=2, max_value=200, step=1)
        if act & {"Break of Structure (BOS)", "Change of Character (CHoCH)", "Liquidity Sweep", "Demand Zone Sustain",
                  "Supply Zone Sustain", "Fair Value Gap"}:
            c1, c2 = st.columns(2)
            fc["swing"] = w_num(c1, "Structure swing len", "f_swing", 5, min_value=1, max_value=50, step=1)
            fc["lookback"] = w_num(c2, "Recent-event lookback (bars)", "f_look", 10, min_value=1, max_value=500, step=1)
        if "Volume N×" in act:
            c1, c2 = st.columns(2)
            fc["vol_mult"] = w_num(c1, "Volume ×", "vol_mult", 1.5, min_value=0.1, max_value=50.0, step=0.1)
            fc["vol_len"] = w_num(c2, "Avg len", "vol_len", 20, min_value=2, max_value=500, step=1)
        if "Absolute OI" in act:
            fc["oi_min"] = w_num(st, "Min absolute OI", "oi_min", 100000, min_value=0, max_value=10**10, step=1000)
        if "Change in OI" in act:
            fc["oi_chg_mode"] = w_sel(st, "OI change mode", ["Absolute", "N× of previous"], "oi_mode")
            fc["oi_chg_min"] = w_num(st, "Min |ΔOI|", "oi_chg", 50000, min_value=0, max_value=10**10, step=1000)
            fc["oi_chg_mult"] = w_num(st, "OI / prev OI ≥", "oi_mult", 1.1, min_value=1.0, max_value=100.0, step=0.05)
        if "Regime (trend) aligned" in act:
            c1, c2 = st.columns(2)
            fc["regime_ema"] = w_num(c1, "Regime EMA", "reg_ema", 50, min_value=2, max_value=1000, step=1)
            fc["regime_adx"] = w_num(c2, "Regime ADX ≥", "reg_adx", 20.0, min_value=0.0, max_value=100.0)
        if "Supertrend" in act:
            c1, c2 = st.columns(2)
            fc["st_period"] = w_num(c1, "ST period", "f_stp", 10, min_value=1, max_value=200, step=1)
            fc["st_factor"] = w_num(c2, "ST factor", "f_stf", 3.0, min_value=0.1, max_value=20.0, step=0.1)
        if "RSI" in act:
            c1, c2, c3 = st.columns(3)
            fc["rsi_len"] = w_num(c1, "RSI len", "f_rsil", 14, min_value=2, max_value=200, step=1)
            fc["rsi_long"] = w_num(c2, "Long if >", "f_rsil2", 50.0, min_value=0.0, max_value=100.0)
            fc["rsi_short"] = w_num(c3, "Short if <", "f_rsis", 50.0, min_value=0.0, max_value=100.0)
        st.markdown("---")
        fc["cross_type"] = w_sel(st, "Crossover candle type", ["Simple", "ATR based candle", "Custom size candle"], "ctype")
        if fc["cross_type"] == "ATR based candle":
            fc["atr_mult"] = w_num(st, "Body ≥ × ATR(14)", "c_atr", 1.0, min_value=0.05, max_value=20.0, step=0.05)
        if fc["cross_type"] == "Custom size candle":
            fc["candle_pts"] = w_num(st, "Body ≥ points", "c_pts", 5.0, min_value=0.0, max_value=1e7, step=0.5)
        fc["angle_on"] = w_cb(st, "Angle of crossover filter", "ang_on", False,
                              help="|atan((EMA−EMA[1]) / ATR14)| in degrees, same for up/down crosses")
        if fc["angle_on"]:
            c1, c2, c3 = st.columns(3)
            fc["angle_min"] = w_num(c1, "Min °", "ang_min", 0.0, min_value=0.0, max_value=100.0)
            fc["angle_max"] = w_num(c2, "Max °", "ang_max", 100.0, min_value=0.0, max_value=100.0)
            fc["angle_ema"] = w_num(c3, "Fast EMA", "ang_ema", 9, min_value=1, max_value=500, step=1)
        fc["flip"] = w_cb(st, "Flip trades (long→short, short→long; options: long→PE, short→CE)", "flip", False)
    cfg["filters"] = fc

    # ------------------------------------------------------------------ Dhan
    sb.markdown("### 🏦 Broker (Dhan)")
    dhan_on = w_cb(sb, "Enable Dhan account", "dhan_on", False)
    cid = tok = ""
    orders_on = False
    trade_mode = None
    dcfg: Dict[str, Any] = {}
    if dhan_on or use_dhan_data:
        cid = w_txt(sb, "Dhan Client ID", "dhan_cid", "")
        tok = w_txt(sb, "Dhan Access Token", "dhan_tok", "", type="password")
    if dhan_on:
        orders_on = w_cb(sb, "Place live orders via Dhan", "dhan_orders", False)
    if orders_on:
        modes = ["Stocks Intraday", "Stocks Delivery", "Index Options", "Stock Options", "Index Futures", "Stock Futures"]
        dflt = "Index Options" if meta.get("kind") == "index" else "Stocks Intraday"
        trade_mode = w_sel(sb, "Product", modes, "trade_mode", dflt)
        if trade_mode in ("Index Options", "Stock Options"):
            dcfg.update(_strike_widgets(sb, symbol, meta, trade_mode == "Stock Options"))
        c1, c2 = sb.columns(2)
        dcfg["entry_type"] = w_sel(c1, "Entry order", ["MARKET", "LIMIT"], "entry_ot")
        dcfg["exit_type"] = w_sel(c2, "Square-off order", ["MARKET", "LIMIT"], "exit_ot")
        if "LIMIT" in (dcfg["entry_type"], dcfg["exit_type"]):
            dcfg["limit_off"] = w_num(sb, "Limit price buffer (pts, towards fill)", "lim_off", 0.5, min_value=0.0, max_value=1e5, step=0.05)
        sb.caption("Security IDs are resolved automatically from Dhan's scrip master (shown on the Live tab).")
    dcfg.setdefault("limit_off", 0.5)
    cfg.update(dhan_on=dhan_on, dhan_cid=cid, dhan_tok=tok, orders_on=orders_on, trade_mode=trade_mode, dhan=dcfg)
    dhan_ready = bool(cid and tok)
    cfg["source"] = "Dhan" if (use_dhan_data and dhan_ready and meta.get("kind") in ("index", "stock")) else "yfinance"
    if use_dhan_data and cfg["source"] != "Dhan":
        sb.warning("Dhan data needs Client ID + Token and an Indian index/stock — falling back to yfinance.")

    # ------------------------------------------------------------------ quantity
    und = meta.get("und", "")
    if trade_mode in ("Index Options", "Index Futures") or (trade_mode is None and meta.get("options_only")):
        qdef = DEFAULT_LOTS.get(und, 1)
    else:
        qdef = 1
    cfg["qty"] = int(w_num(sb, "Quantity (units; editable as SEBI lot sizes change)", f"qty_{und}_{trade_mode}", qdef,
                           min_value=1, max_value=10**7, step=1))
    if meta.get("options_only") and trade_mode is None:
        cfg["trade_mode"] = "Index Options" if meta.get("kind") == "index" else "Stock Options"

    # ------------------------------------------------------------------ session & live
    with sb.expander("⏱️ Session, live & backtest settings", expanded=False):
        cfg["refresh"] = float(w_num(st, "Live refresh (seconds)", "refresh", 2.0, min_value=1.0, max_value=60.0, step=0.5,
                                     help="Each yfinance request is additionally spaced ≥0.3 s by a global rate limiter."))
        cfg["auto_sqoff"] = w_cb(st, "Auto square-off intraday (Indian markets)", "auto_sq", True)
        cfg["sqoff_time"] = w_txt(st, "Square-off time (HH:MM IST)", "sq_time", "15:15")
        cfg["chart_bars"] = int(w_num(st, "Live chart candles", "chart_bars", 200, min_value=50, max_value=2000, step=10))
        cfg["fill_model"] = w_sel(st, "Backtest intrabar fill model",
                                  ["Conservative (SL checked first)", "TradingView path (O→near extreme→far extreme→C)"], "fill_model",
                                  help="Conservative = your rule: LONG low vs SL before high vs target. TradingView path = the "
                                       "broker-emulator assumption; points/step trailing SL updates along the path → closer to live.")

    # ------------------------------------------------------------------ costs
    sb.markdown("### 💰 Costs")
    cfg["costs_on"] = w_cb(sb, "Include SEBI/exchange charges in P&L", "costs_on", False)
    segs = list(DEFAULT_RATES)
    if trade_mode in ("Index Options", "Stock Options") or meta.get("options_only") or (meta.get("kind") == "index" and not trade_mode):
        sdef = "Options"
    elif trade_mode in ("Index Futures", "Stock Futures"):
        sdef = "Futures"
    elif trade_mode == "Stocks Delivery":
        sdef = "Equity Delivery"
    elif meta.get("kind") in ("crypto", "forex", "commodity"):
        sdef = "Crypto/Forex (flat %)"
    else:
        sdef = "Equity Intraday"
    cfg["cost_segment"], cfg["assumed_premium"], cfg["rates"] = sdef, 100.0, DEFAULT_RATES
    if cfg["costs_on"]:
        cfg["cost_segment"] = w_sel(sb, "Charges segment", segs, "cost_seg", sdef)
        if cfg["cost_segment"] == "Options":
            cfg["assumed_premium"] = w_num(sb, "Assumed option premium (backtest)", "prem", 100.0, min_value=0.05, max_value=1e6)
        with sb.expander("Edit charge rates (% unless noted)"):
            rdf = pd.DataFrame(DEFAULT_RATES).T
            ed = st.data_editor(rdf, key="rates_ed", **WIDE)
            cfg["rates"] = {k: {c: float(ed.loc[k, c]) for c in ed.columns} for k in ed.index}
            st.caption("STT F&O per Budget 2026 (eff. 1-Apr-2026): futures 0.05%, options 0.15% on sell premium. "
                       "Brokerage: Dhan ₹20 or 0.03% (lower). brok_pct=100 for options ⇒ flat ₹20 cap.")

    # ------------------------------------------------------------------ notifications
    sb.markdown("### 🔔 Notifications")
    nc: Dict[str, Any] = {}
    nc["email_on"] = w_cb(sb, "Email notifications", "em_on", False)
    if nc["email_on"]:
        nc["email_from"] = w_txt(sb, "From (Gmail)", "em_from", "")
        nc["email_pass"] = w_txt(sb, "App passcode", "em_pass", "", type="password")
        nc["email_to"] = w_txt(sb, "To (comma separated)", "em_to", "")
        nc["email_subject"] = w_txt(sb, "Subject", "em_sub", "[{event}] {ticker} {action} @ {price}")
        nc["email_body"] = w_area(sb, "Body", "em_body", "{event}\nTicker: {ticker}\nAction: {action}\nPrice: {price}\nSL: {sl}\n"
                                  "Target: {target}\nPnL: {pnl}\nReason: {reason}\nTime: {time}")
        c1, c2 = sb.columns(2)
        nc["smtp_host"] = w_txt(c1, "SMTP host", "smtp_h", "smtp.gmail.com")
        nc["smtp_port"] = w_num(c2, "Port", "smtp_p", 465, min_value=1, max_value=65535, step=1)
    nc["wa_on"] = w_cb(sb, "WhatsApp notifications", "wa_on", False)
    if nc["wa_on"]:
        nc["wa_provider"] = w_sel(sb, "Provider", ["CallMeBot", "Twilio"], "wa_prov")
        nc["wa_to"] = w_txt(sb, "To number (+91…)", "wa_to", "")
        if nc["wa_provider"] == "CallMeBot":
            nc["wa_apikey"] = w_txt(sb, "CallMeBot API key", "wa_key", "", type="password")
        else:
            nc["tw_sid"] = w_txt(sb, "Twilio SID", "tw_sid", "")
            nc["tw_token"] = w_txt(sb, "Twilio token", "tw_tok", "", type="password")
            nc["wa_from"] = w_txt(sb, "From number (Twilio WA)", "wa_from", "")
        nc["wa_body"] = w_area(sb, "Message", "wa_body",
                               "{event}: {ticker} {action} @ {price} | SL {sl} | TGT {target} | PnL {pnl} | {reason}")
    cfg["notify"] = nc
    return cfg


# =============================================================================
#  DATA ACCESS (cached for backtests)
# =============================================================================
@st.cache_data(ttl=120, show_spinner=False, max_entries=64)
def get_history(source: str, symbol: str, tf: str, period: str, meta_json: str, cid: str, tok: str) -> pd.DataFrame:
    meta = json.loads(meta_json)
    if source == "Dhan":
        return fetch_dhan(DhanClient(cid, tok), symbol, tf, period, meta)
    return fetch_yf(symbol, tf, period, meta.get("kind", ""))


@st.cache_data(ttl=30, show_spinner=False, max_entries=64)
def get_spot(symbol: str) -> Optional[float]:
    try:
        d = yf_history(symbol, "1m", period="1d")
    except Exception:
        d = yf_history(symbol, "1d", period="5d")
    return float(d["Close"].iloc[-1])


@st.cache_data(ttl=900, show_spinner=False, max_entries=64)
def get_daily(symbol: str, days: int = 900) -> pd.DataFrame:
    return yf_history(symbol, "1d", start=now_ist() - timedelta(days=days), end=now_ist() + timedelta(days=1))


def needs_daily(cfg) -> bool:
    if cfg["tf"] not in INTRADAY_TFS:
        return False
    return cfg["strategy"] == VIX_NAME or bool({"50D SMA", "100D SMA", "200D SMA"} & set(cfg["filters"]["active"]))


def aux_for(cfg: dict) -> dict:
    """Extra series some strategies need (India VIX for the expected-move strategy)."""
    aux: Dict[str, Any] = {}
    if cfg["strategy"] == VIX_NAME and str(cfg["params"].get("vol_source", "")).startswith("Auto") \
            and cfg.get("kind") == "index" and cfg.get("indian"):
        days = 900
        if cfg["tf"] not in INTRADAY_TFS and cfg["period"].endswith("y"):
            days = int((int(cfg["period"][:-1]) + 2) * 366)
        try:
            aux["vix"] = get_daily("^INDIAVIX", days)
        except Exception as e:
            aux["vix_error"] = str(e)
    return aux


def ema_overlays(x: Ctx, sig: Optional[dict]) -> Dict[str, np.ndarray]:
    lines = {"EMA 9": x.ema(9), "EMA 21": x.ema(21)}
    if sig:
        for k, v in sig["lines"].items():
            if k not in lines:
                lines[k] = v
    return lines


# =============================================================================
#  BACKTEST TAB
# =============================================================================
def backtest_tab(cfg: dict):
    st.markdown("#### 🧪 Backtest")
    st.caption("Signal on candle N → entry at candle N+1 OPEN. Inside each candle the SL is checked first "
               "(LONG: low vs SL then high vs target; SHORT: high vs SL then low vs target). "
               "Immediate strategies (Simple Buy/Sell, thresholds) enter on the trigger itself.")
    if st.button("▶️ Run Backtest", type="primary", **WIDE, key="m_run_bt"):
        with st.spinner(f"Fetching {cfg['symbol']} {cfg['tf']} from {cfg['source']} (with warm-up history)…"):
            try:
                df = get_history(cfg["source"], cfg["symbol"], cfg["tf"], cfg["period"], json.dumps(cfg["meta"]),
                                 cfg["dhan_cid"], cfg["dhan_tok"])
                daily = get_daily(cfg["symbol"]) if needs_daily(cfg) else None
                aux = aux_for(cfg)
                if aux.get("vix_error"):
                    st.warning(f"India VIX history unavailable ({aux['vix_error']}) — using HV / manual value.")
            except Exception as e:
                st.error(f"Data fetch failed: {e}")
                return
        if df.empty or len(df) < 30:
            st.error("Not enough data returned for this ticker/timeframe.")
            return
        with st.spinner("Running strategy…"):
            t0 = time.time()
            trades, summ, x, meta = run_backtest(df, cfg, daily, aux)
        st.session_state["bt"] = {"trades": trades, "summary": summ, "df": df, "meta": meta, "cfg": cfg,
                                  "x": x, "elapsed": time.time() - t0}
    bt = st.session_state.get("bt")
    if not bt:
        st.info("Configure the sidebar and click **Run Backtest**.")
        return
    trades, summ, df, meta, bcfg = bt["trades"], bt["summary"], bt["df"], bt["meta"], bt["cfg"]
    start = meta["start"]
    st.caption(f"{bcfg['symbol']} · {bcfg['tf']} · {bcfg['period']} · {bcfg['strategy']} · data {bcfg['source']} · "
               f"tested bars {len(df) - start:,} (warm-up {start:,} bars) · {fmt_ts(df.index[start])} → {fmt_ts(df.index[-1])} · "
               f"signals L{meta['signals_long']}/S{meta['signals_short']} · {bt['elapsed']:.2f}s")
    vi = (meta.get("sig") or {}).get("vol_info")
    if vi:
        st.info("📊 " + vix_info_text(vi))
    if not trades:
        st.warning("No trades generated. Loosen filters or change parameters/period.")
    else:
        s = summ
        r1 = st.columns(6)
        r1[0].metric("Total Trades", s["Total Trades"])
        r1[1].metric("Winning", s["Winning Trades"])
        r1[2].metric("Losing", s["Losing Trades"])
        r1[3].metric("Accuracy", f"{s['Accuracy %']}%")
        r1[4].metric("Net Points", fnum(s["Net Points"]))
        r1[5].metric("Profit Factor", s["Profit Factor"])
        r2 = st.columns(6)
        r2[0].metric("Points Won", fnum(s["Points Won"]))
        r2[1].metric("Points Lost", fnum(s["Points Lost"]))
        r2[2].metric("Sharpe", s["Sharpe Ratio"])
        r2[3].metric("Risk:Reward", s["Risk:Reward (avg)"])
        r2[4].metric("Max Drawdown ₹", fnum(s["Max Drawdown (₹)"]))
        r2[5].metric("Total Charges ₹", fnum(s["Total Charges"]))
        r3 = st.columns(3)
        r3[0].metric("PnL before charges ₹", fnum(s["PnL Before Charges"]))
        r3[1].metric("PnL after charges ₹", fnum(s["PnL After Charges"]),
                     delta=fnum(s["PnL After Charges"] - s["PnL Before Charges"]) if bcfg["costs_on"] else None)
        r3[2].metric("Avg win / loss (pts)", f"{fnum(s['Avg Win (pts)'])} / {fnum(s['Avg Loss (pts)'])}")
        tdf = pd.DataFrame(trades)
        show = ["Entry Time (IST)", "Exit Time (IST)", "Signal", "Action", "Entry Price", "Exit Price", "SL Price",
                "Final SL", "Target Price", "Entry Candle Open", "Entry Candle High", "Entry Candle Low", "Entry Candle Close",
                "Exit Candle Open", "Exit Candle High", "Exit Candle Low", "Exit Candle Close", "Highest Price",
                "Lowest Price", "Points", "Qty", "PnL (Gross)", "Charges", "PnL (Net)", "Exit Reason", "Bars Held", "Recommendation"]
        tdf = tdf[show]

        def _col(v):
            return "color:#26a69a;font-weight:600" if v > 0 else ("color:#ef5350;font-weight:600" if v < 0 else "")
        try:
            sty = tdf.style.map(_col, subset=["Points", "PnL (Gross)", "PnL (Net)"]).format(precision=2)
        except AttributeError:
            sty = tdf.style.applymap(_col, subset=["Points", "PnL (Gross)", "PnL (Net)"]).format(precision=2)
        st.dataframe(sty, **WIDE, height=380, hide_index=True)
        export_buttons(tdf, summ, "backtest")
        eq = tdf["PnL (Net)"].cumsum()
        if go is not None:
            ef = go.Figure(go.Scatter(y=eq, mode="lines", fill="tozeroy", line=dict(color="#00a6fb", width=2), name="Equity"))
            ef.update_layout(height=220, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                             margin=dict(l=10, r=10, t=30, b=10), title="Equity curve (net ₹)")
            st.plotly_chart(ef, **WIDE)
    if meta.get("filter_report"):
        with st.expander("Active filters (state on last bar)"):
            st.dataframe(pd.DataFrame(meta["filter_report"], columns=["Filter", "Long OK", "Short OK"]), hide_index=True)
    x = bt["x"]
    lines = ema_overlays(x, meta.get("sig"))
    sub = df.iloc[start:]
    lines_sub = {k: np.asarray(v)[start:] for k, v in lines.items()}
    tr_adj = [dict(t, _ei=t["_ei"] - start, _xi=t["_xi"] - start) for t in trades]
    max_b = 3000
    fig = candle_chart(sub, lines_sub, f"{bcfg['symbol']} · {bcfg['tf']} · {bcfg['strategy']}", tr_adj, height=650, max_bars=max_b)
    if fig:
        if len(sub) > max_b:
            st.caption(f"Chart shows the last {max_b} candles (all {len(sub):,} were tested).")
        st.plotly_chart(fig, **WIDE, key="m_bt_chart")


# =============================================================================
#  LIVE TRADING ENGINE
# =============================================================================
def _live_state() -> dict:
    ss = st.session_state
    if "live" not in ss:
        ss["live"] = {"running": False, "pos": None, "base": None, "data_key": None, "daily": None,
                      "last_closed": None, "last_exit_closed": None, "first_tick": True, "ltp": None,
                      "prev_ltp": None, "armed_up": True, "armed_dn": True, "log": [], "ticks": 0,
                      "started": None, "last_update": None, "error": None, "x": None, "sig": None,
                      "report": [], "forming": None, "oi": None, "day_count": {}, "aux": {}, "aux_ts": 0.0,
                      "app_rerun": False, "levels": (np.nan, np.nan)}
    return ss["live"]


def _log(L: dict, msg: str, level: str = "info"):
    L["log"].insert(0, {"time": now_ist().strftime("%H:%M:%S"), "level": level, "msg": msg})
    del L["log"][200:]


def _data_key(cfg) -> str:
    return f"{cfg['source']}|{cfg['symbol']}|{cfg['tf']}|{cfg['period']}"


def _split_closed(df: pd.DataFrame, tf: str) -> Tuple[pd.DataFrame, bool]:
    if df.empty:
        return df, False
    last = df.index[-1]
    nw = now_ist()
    if tf in INTRADAY_TFS:
        forming = last + pd.Timedelta(minutes=TF_MINUTES[tf]) > nw
    elif tf in ("1d", "3d"):
        forming = last.date() >= nw.date() - timedelta(days=0 if tf == "1d" else 2)
    elif tf == "1wk":
        forming = last + pd.Timedelta(days=7) > nw
    else:
        forming = last + pd.DateOffset(months=1) > nw
    return (df.iloc[:-1] if forming and len(df) > 1 else df), forming


def _live_fetch(cfg: dict, L: dict):
    key = _data_key(cfg)
    dc = DhanClient(cfg["dhan_cid"], cfg["dhan_tok"])
    if L["base"] is None or L["data_key"] != key:
        base = (fetch_dhan(dc, cfg["symbol"], cfg["tf"], cfg["period"], cfg["meta"]) if cfg["source"] == "Dhan"
                else fetch_yf(cfg["symbol"], cfg["tf"], cfg["period"], cfg["kind"]))
        L.update(base=base, data_key=key, last_closed=None, first_tick=True)
        L["aux_ts"] = 0.0
        _log(L, f"Warm-up history loaded: {len(base):,} candles ({cfg['source']})")
        df = base
    else:
        tail = (fetch_dhan(dc, cfg["symbol"], cfg["tf"], cfg["period"], cfg["meta"], live_tail=True) if cfg["source"] == "Dhan"
                else fetch_yf(cfg["symbol"], cfg["tf"], cfg["period"], cfg["kind"], live_tail=True))
        df = pd.concat([L["base"], tail])
        df = df[~df.index.duplicated(keep="last")].sort_index().iloc[-6000:]
        L["base"] = df
    aux_key = (cfg["strategy"], str(cfg["params"].get("vol_source")), needs_daily(cfg), cfg["symbol"])
    if time.time() - L.get("aux_ts", 0.0) > 900 or L.get("aux_key") != aux_key:  # daily / India VIX: 15-min refresh
        L["aux_ts"], L["aux_key"] = time.time(), aux_key
        if needs_daily(cfg):
            try:
                L["daily"] = get_daily(cfg["symbol"])
            except Exception as e:
                _log(L, f"Daily data unavailable: {e}", "warn")
        L["aux"] = aux_for(cfg)
        if L["aux"].get("vix_error"):
            _log(L, f"India VIX unavailable ({L['aux']['vix_error']}) — HV / manual value used", "warn")
    ltp = None
    if cfg["source"] == "Dhan":
        try:
            m = cfg["meta"]
            if m.get("kind") == "index":
                ltp = dc.ltp("IDX_I", m["dhan_id"])
            else:
                sid, seg = resolve_equity_id(cfg["symbol"])
                ltp = dc.ltp(seg, sid) if sid else None
        except Exception as e:
            _log(L, f"Dhan LTP failed, using last close: {e}", "warn")
    if ltp is None:
        ltp = float(df["Close"].iloc[-1])
    return df, float(ltp)


def _offset_steps(label: str) -> int:
    if label == "ATM":
        return 0
    side, k = label.split()
    return int(k) * (1 if side == "OTM" else -1)


def _resolve_instrument(cfg: dict, direction: int, spot: float) -> dict:
    """Pick the tradable instrument for the signal (options → CE/PE strike, futures, equity)."""
    mode = cfg.get("trade_mode")
    m = cfg["meta"]
    und, exch = m.get("und", ""), m.get("fno_exch", "NSE")
    if mode in ("Index Options", "Stock Options"):
        ot = "CE" if direction == 1 else "PE"
        lab = cfg["dhan"].get("ce_strike" if ot == "CE" else "pe_strike", "ATM")
        manual = cfg["dhan"].get("ce_manual" if ot == "CE" else "pe_manual") if lab == "Manual" else None
        r = resolve_derivative(und, exch, "option", spot, ot, offset_steps(lab), stock=(mode == "Stock Options"),
                               strike_value=manual)
        r.update(product="INTRADAY", kind="option", opt_type=ot)
        return r
    if mode in ("Index Futures", "Stock Futures"):
        r = resolve_derivative(und, exch, "future", spot, stock=(mode == "Stock Futures"))
        r.update(product="INTRADAY", kind="future")
        return r
    sid, seg = resolve_equity_id(cfg["symbol"])
    return {"security_id": sid, "segment": seg, "symbol": cfg["symbol"], "kind": "equity",
            "product": "CNC" if mode == "Stocks Delivery" else "INTRADAY", "error": None if sid else "equity id not found"}


def _place(cfg: dict, L: dict, pos: Position, entry: bool, ltp_inst: Optional[float]):
    if not cfg.get("orders_on"):
        return
    inst = pos.instrument
    if not inst or not inst.get("security_id"):
        _log(L, f"Order skipped: instrument unresolved ({inst.get('error') if inst else 'n/a'})", "error")
        return
    dc = DhanClient(cfg["dhan_cid"], cfg["dhan_tok"])
    buy_side = True if inst["kind"] == "option" else (pos.direction == 1)
    side = ("BUY" if buy_side else "SELL") if entry else ("SELL" if buy_side else "BUY")
    otype = cfg["dhan"]["entry_type"] if entry else cfg["dhan"]["exit_type"]
    px = 0.0
    if otype == "LIMIT":
        ref = ltp_inst if ltp_inst else (L["ltp"] or 0)
        off = cfg["dhan"].get("limit_off", 0.5)
        px = ref + off if side == "BUY" else max(ref - off, 0.05)
    try:
        res = dc.place_order(side=side, seg=inst["segment"], product=inst["product"], order_type=otype,
                             sec_id=inst["security_id"], qty=pos.qty, price=px, tag=f"qe{pos.id}{'E' if entry else 'X'}")
        pos.orders.append({"side": side, "type": otype, "price": px, "resp": res})
        _log(L, f"Dhan {side} {otype} {inst.get('symbol')} x{pos.qty} → {res.get('orderStatus', res)} id {res.get('orderId', '?')}", "order")
    except Exception as e:
        pos.orders.append({"side": side, "error": str(e)})
        _log(L, f"Dhan order FAILED ({side} {inst.get('symbol')}): {e}", "error")


def _inst_ltp(cfg: dict, pos: Position) -> Optional[float]:
    if not (cfg.get("orders_on") and pos.instrument.get("security_id") and pos.instrument.get("kind") in ("option", "future")):
        return None
    try:
        return DhanClient(cfg["dhan_cid"], cfg["dhan_tok"]).ltp(pos.instrument["segment"], pos.instrument["security_id"])
    except Exception:
        return None


def _enter(cfg: dict, L: dict, x: Ctx, direction: int, price: float, ref_k: int, reason: str,
           tgt_override: Optional[float] = None):
    sl, tg = init_levels(x, ref_k, direction, price, cfg["risk"])
    if tgt_override is not None:
        tg = tgt_override
    df = L.get("df")
    pos = Position(direction=direction, entry=price, entry_time=now_ist(), qty=cfg["qty"], sl=sl, tgt=tg,
                   init_sl=sl, init_tgt=tg, sig_idx=ref_k, hi=price, lo=price, label=action_label(direction, cfg),
                   strategy=cfg["strategy"], fixed_tgt=tgt_override is not None,
                   entry_bar_ts=df.index[-1] if df is not None and len(df) else None)
    dk = now_ist().date()
    L["day_count"][dk] = L["day_count"].get(dk, 0) + 1
    if cfg.get("orders_on"):
        pos.instrument = _resolve_instrument(cfg, direction, price)
        pos.inst_entry = _inst_ltp(cfg, pos)
        _place(cfg, L, pos, True, pos.inst_entry)
    L["pos"] = pos
    _log(L, f"ENTRY {pos.label} @ {fnum(price)} | SL {fnum(sl)} | TGT {fnum(tg)} | {reason}", "entry")
    notify(cfg, "ENTRY", {"ticker": cfg["symbol"], "action": pos.label, "price": price, "sl": sl, "target": tg,
                          "pnl": 0.0, "reason": reason})


def _exit(cfg: dict, L: dict, price: float, reason: str):
    pos: Position = L["pos"]
    if pos is None:
        return
    inst_exit = _inst_ltp(cfg, pos) if pos.instrument else None
    _place(cfg, L, pos, False, inst_exit)
    pts = (price - pos.entry) * pos.direction
    if pos.inst_entry and inst_exit and pos.instrument.get("kind") == "option":
        gross = (inst_exit - pos.inst_entry) * pos.qty
    elif pos.inst_entry and inst_exit and pos.instrument.get("kind") == "future":
        gross = (inst_exit - pos.inst_entry) * pos.direction * pos.qty
    else:
        gross = pts * pos.qty
    ch = trade_charges(cfg, pos.label, pos.entry, price, pos.qty, pos.inst_entry, inst_exit)
    et = now_ist()
    df = L.get("df")
    ec = {"Open": None, "High": None, "Low": None, "Close": None}
    xc = dict(ec)
    ets, xts = pos.entry_bar_ts, None
    if df is not None and len(df):
        if ets is not None and ets in df.index:
            ec = df.loc[ets, ["Open", "High", "Low", "Close"]].to_dict()
        xts = df.index[-1]
        xc = df.iloc[-1][["Open", "High", "Low", "Close"]].to_dict()
    rec = {"Trade ID": pos.id, "Ticker": cfg["symbol"], "Timeframe": cfg["tf"], "Strategy": pos.strategy,
           "Signal": "LONG" if pos.direction == 1 else "SHORT", "Action": pos.label,
           "Entry Time (IST)": fmt_ts(pos.entry_time), "Exit Time (IST)": fmt_ts(et),
           "Duration": fmt_duration((et - pos.entry_time).total_seconds()),
           "Entry Price": round(pos.entry, 4), "Exit Price": round(price, 4),
           "Initial SL": round(pos.init_sl, 4) if pos.init_sl == pos.init_sl else None,
           "Final SL": round(pos.sl, 4) if pos.sl == pos.sl else None,
           "Target": round(pos.init_tgt, 4) if pos.init_tgt == pos.init_tgt else None,
           "Entry Candle Time": fmt_ts(ets) if ets is not None else "—",
           "Entry Candle Open": ec["Open"], "Entry Candle High": ec["High"], "Entry Candle Low": ec["Low"], "Entry Candle Close": ec["Close"],
           "Exit Candle Time": fmt_ts(xts) if xts is not None else "—",
           "Exit Candle Open": xc["Open"], "Exit Candle High": xc["High"], "Exit Candle Low": xc["Low"], "Exit Candle Close": xc["Close"],
           "Highest Price": round(pos.hi, 4), "Lowest Price": round(pos.lo, 4), "Points": round(pts, 4),
           "Qty": pos.qty, "PnL (Gross)": round(gross, 2), "Charges": ch, "PnL (Net)": round(gross - ch, 2),
           "Exit Reason": reason, "Instrument": pos.instrument.get("symbol") if pos.instrument else cfg["symbol"],
           "Inst Entry": pos.inst_entry, "Inst Exit": inst_exit,
           "Orders": json.dumps([{k: v for k, v in o.items() if k != "resp"} for o in pos.orders]) if pos.orders else ""}
    db_save_trade(cfg["user"], rec)
    st.session_state.setdefault("history", []).append(rec)
    L["pos"] = None
    L["last_exit_closed"] = L["last_closed"]
    L["app_rerun"] = True  # refresh Trade History tab immediately
    _log(L, f"EXIT {pos.label} @ {fnum(price)} | {reason} | pts {fnum(pts)} | net ₹{fnum(gross - ch)}", "exit")
    notify(cfg, "EXIT", {"ticker": cfg["symbol"], "action": pos.label, "price": price, "sl": pos.sl, "target": pos.tgt,
                         "pnl": round(gross - ch, 2), "reason": reason})


def market_open(cfg: dict, nw: Optional[datetime] = None) -> bool:
    """NSE/BSE cash & F&O session 09:15–15:30 IST, Mon–Fri. Crypto/forex/commodities: always open."""
    if not cfg.get("indian"):
        return True
    nw = nw or now_ist()
    if nw.weekday() >= 5:
        return False
    return (9, 15) <= (nw.hour, nw.minute) < (15, 30)


def _refresh_oi(cfg: dict, L: dict):
    """ATM option-chain OI from Dhan (index or F&O stock), refreshed at most every 30 s (chain API: 1 req / 3 s)."""
    if not ({"Absolute OI", "Change in OI"} & set(cfg["filters"]["active"])):
        return
    if not (cfg["dhan_cid"] and cfg["dhan_tok"]) or cfg["kind"] not in ("index", "stock"):
        return
    if L.get("oi_ts") and (time.time() - L["oi_ts"]) < 30:
        return
    L["oi_ts"] = time.time()
    try:
        dc = DhanClient(cfg["dhan_cid"], cfg["dhan_tok"])
        m = cfg["meta"]
        if m.get("kind") == "index":
            uid, useg = m["dhan_id"], "IDX_I"
        else:
            uid, _ = resolve_equity_id(cfg["symbol"])
            useg = "NSE_FNO"
        exps = dc.expiry_list(uid, useg)
        if not exps:
            return
        oc = dc.option_chain(uid, useg, exps[0]).get("data", {})
        chain, spot = oc.get("oc", {}), float(oc.get("last_price") or L.get("ltp") or 0)
        if not chain:
            return
        atm = min(chain, key=lambda k: abs(float(k) - spot))
        row = chain[atm]
        oi = sum(float(row.get(s_, {}).get("oi", 0) or 0) for s_ in ("ce", "pe"))
        prev = sum(float(row.get(s_, {}).get("previous_oi", 0) or 0) for s_ in ("ce", "pe"))
        L["oi"] = {"oi": oi, "prev_oi": prev, "oi_change": oi - prev, "strike": float(atm), "expiry": exps[0]}
    except Exception as e:
        _log(L, f"Option-chain OI fetch failed: {e}", "warn")


def live_tick(cfg: dict):
    L = _live_state()
    rc = dict(cfg["risk"], flip_sign=-1 if cfg["filters"].get("flip") else 1)
    flip = rc["flip_sign"]
    df, ltp = _live_fetch(cfg, L)
    L["prev_ltp"], L["ltp"] = L["ltp"], ltp
    L["ticks"] += 1
    L["last_update"] = now_ist()
    closed, forming = _split_closed(df, cfg["tf"])
    L["forming"] = forming
    aux = dict(L.get("aux") or {}, **make_aux(cfg))
    x = Ctx(closed, L.get("daily"), aux)
    name = cfg["strategy"]
    immediate = name in IMMEDIATE
    sig = None
    _refresh_oi(cfg, L)
    if not immediate:
        sig = run_strategy(x, name, cfg["params"], cfg.get("hybrid"))
        longs, shorts, report = apply_filters(x, sig["long"], sig["short"], cfg["filters"], L.get("oi"))
        L["report"] = report
    L["x"], L["sig"], L["df"] = x, sig, df
    xf = sig_f = None
    if immediate and STRATEGIES[name].func:
        xf = Ctx(df, L.get("daily"), aux)   # full frame incl. forming candle → today's levels from the first tick
        sig_f = run_strategy(xf, name, cfg["params"])
        L["sig"] = sig_f
    k = x.n - 1
    new_candle = L["last_closed"] is not None and closed.index[-1] != L["last_closed"]
    first = L["first_tick"]
    L["last_closed"] = closed.index[-1]
    L["first_tick"] = False
    cutoff = _intraday_cutoff(cfg)
    nw = now_ist()
    past_cut = bool(cutoff and (nw.hour, nw.minute) >= cutoff)
    mkt_open = market_open(cfg, nw)
    L["market_open"] = mkt_open

    pos: Optional[Position] = L["pos"]
    # ------------------------------------------------ manage open position on every tick (LTP based)
    if pos is not None:
        if pos.instrument:
            pos.inst_ltp = _inst_ltp(cfg, pos)
        trail_on_price(pos, ltp, rc)
        d = pos.direction
        if not np.isnan(pos.sl) and ((d == 1 and ltp <= pos.sl) or (d == -1 and ltp >= pos.sl)):
            _exit(cfg, L, ltp, "SL hit"); pos = None
        elif (is_target_active(pos, rc)
              and ((d == 1 and ltp >= pos.tgt) or (d == -1 and ltp <= pos.tgt))):
            _exit(cfg, L, ltp, "Target hit"); pos = None
        elif past_cut and cfg["tf"] in INTRADAY_TFS:
            _exit(cfg, L, ltp, "Intraday auto square-off"); pos = None
        elif new_candle:
            trail_on_bar(x, k, pos, rc)
            r = strategy_exit_reason(x, k, pos, rc, sig)
            if r:
                _exit(cfg, L, ltp, r); pos = None
    # ------------------------------------------------ entries
    if L["pos"] is None and not past_cut and mkt_open:
        if name in ("Simple Buy", "Simple Sell"):
            if L["last_exit_closed"] is None or L["last_closed"] != L["last_exit_closed"]:
                _enter(cfg, L, x, (1 if name == "Simple Buy" else -1) * flip, ltp, k, name)
        elif immediate:
            # levels from the FULL frame (incl. the forming candle) → today's levels are right from the first tick
            xf = xf if xf is not None else Ctx(df, L.get("daily"), aux)
            up, ud, dn, dd = _immediate_levels(xf, cfg, sig_f)
            ul, dl = up[-1], dn[-1]
            L["levels"] = (ul, dl)
            max_day = int(cfg["params"].get("max_entries_day", 0) or 0)
            day_ok = not max_day or L["day_count"].get(nw.date(), 0) < max_day
            last_i = xf.n - 1
            if not np.isnan(ul):
                if ltp < ul:
                    L["armed_up"] = True
                elif L["armed_up"] and day_ok:
                    L["armed_up"] = False
                    d_ = ud * flip
                    if sigma_target_blocked(sig_f, last_i, d_, ltp):
                        _log(L, f"Skipped entry: price {fnum(ltp)} already beyond σ exit target", "warn")
                    else:
                        _enter(cfg, L, x, d_, ltp, k, f"Price ≥ {fnum(ul)}", strategy_target(sig_f, last_i, d_, ltp))
            day_ok = not max_day or L["day_count"].get(nw.date(), 0) < max_day
            if L["pos"] is None and not np.isnan(dl):
                if ltp > dl:
                    L["armed_dn"] = True
                elif L["armed_dn"] and day_ok:
                    L["armed_dn"] = False
                    d_ = dd * flip
                    if sigma_target_blocked(sig_f, last_i, d_, ltp):
                        _log(L, f"Skipped entry: price {fnum(ltp)} already beyond σ exit target", "warn")
                    else:
                        _enter(cfg, L, x, d_, ltp, k, f"Price ≤ {fnum(dl)}", strategy_target(sig_f, last_i, d_, ltp))
        elif new_candle and not first:
            if longs[k] or shorts[k]:
                d = (1 if longs[k] else -1) * flip
                _enter(cfg, L, x, d, ltp, k, f"{name} signal on candle {fmt_ts(closed.index[k])} → entry on N+1")


def _status_board(cfg: dict, L: dict) -> List[str]:
    x, sig, pos, ltp = L.get("x"), L.get("sig"), L.get("pos"), L.get("ltp")
    out = []
    if not L.get("market_open", True):
        out.append("⏸️ Indian market closed (09:15–15:30 IST, Mon–Fri) — no new entries; open positions still monitored.")
    if cfg["strategy"] in IMMEDIATE:
        if cfg["strategy"] in ("Simple Buy", "Simple Sell"):
            out.append("Immediate entry strategy — enters at LTP when flat (re-entry from the next candle after an exit).")
        else:
            if sig and sig.get("status"):
                out.extend(sig["status"])
            ul, dl = L.get("levels", (np.nan, np.nan))
            if ul == ul:
                out.append(f"Up-level {fnum(ul)} ({cfg['params'].get('above_action', '')}) → needs +{fnum(max(ul - ltp, 0))} "
                           f"{'(armed)' if L['armed_up'] else '(fired – re-arms below level)'}")
            if dl == dl:
                out.append(f"Down-level {fnum(dl)} ({cfg['params'].get('below_action', '')}) → needs -{fnum(max(ltp - dl, 0))} "
                           f"{'(armed)' if L['armed_dn'] else '(fired – re-arms above level)'}")
    elif sig:
        out.extend(sig["status"])
        k = x.n - 1
        out.append(f"Last closed candle {fmt_ts(x.df.index[-1])}: raw L={bool(sig['long'][k])} S={bool(sig['short'][k])}")
    if pos is not None:
        d = pos.direction
        if pos.sl == pos.sl:
            out.append(f"EXIT on SL needs {fnum(abs(ltp - pos.sl))} pts {'down' if d == 1 else 'up'}")
        if pos.tgt == pos.tgt and is_target_active(pos, cfg["risk"]):
            out.append(f"EXIT on Target needs {fnum(abs(pos.tgt - ltp))} pts {'up' if d == 1 else 'down'}")
        if {"Strategy Based Exit", "Reverse EMA Crossover"} & {cfg["risk"]["sl_type"], cfg["risk"]["tgt_type"]}:
            out.append("Signal-based exit is evaluated on each candle close")
    return out


# =============================================================================
#  LIVE TAB
# =============================================================================
def _cfg_summary(cfg: dict):
    rc, fc = cfg["risk"], cfg["filters"]
    params = cfg["params"] or ({"members": ", ".join(cfg["hybrid"]["members"]), "mode": cfg["hybrid"]["mode"]} if cfg.get("hybrid") else {})
    items = {
        "Ticker": f"{cfg['ticker_name']} ({cfg['symbol']})", "Timeframe / Period": f"{cfg['tf']} / {cfg['period']}",
        "Data source": cfg["source"], "Strategy": cfg["strategy"],
        "Params": ", ".join(f"{k}={v}" for k, v in params.items()) or "—",
        "Stop-loss": f"{rc['sl_type']} ({rc['sl_pts']} pts)", "Target": f"{rc['tgt_type']} ({rc['tgt_pts']} pts)",
        "Quantity": cfg["qty"], "Product": cfg.get("trade_mode") or ("Index → BUY CE/PE" if cfg["kind"] == "index" else "Long=BUY / Short=SELL"),
        "Filters": ", ".join(fc["active"]) or "None", "Crossover type": fc.get("cross_type"),
        "Flip trades": "ON" if fc.get("flip") else "OFF", "Costs": cfg["cost_segment"] if cfg["costs_on"] else "OFF",
        "Dhan orders": "ON" if cfg["orders_on"] else "OFF", "Refresh": f"{cfg['refresh']} s (yf gap ≥ {YF_DELAY}s)",
    }
    html = "".join(f"<div><b>{k}:</b> {v}</div>" for k, v in items.items())
    st.markdown(f"<div class='qe-card'><div class='qe-kv'>{html}</div></div>", unsafe_allow_html=True)


def live_tab(cfg: dict):
    L = _live_state()
    st.markdown("#### 📡 Live Trading")
    _cfg_summary(cfg)
    if cfg.get("orders_on"):
        with st.expander("🔑 Auto-resolved Dhan security IDs", expanded=False):
            spot = L.get("ltp") or (float(L["base"]["Close"].iloc[-1]) if L.get("base") is not None else None)
            if st.button("Resolve now", key="m_resolve_ids"):
                try:
                    if spot is None:
                        spot = float(fetch_yf(cfg["symbol"], "1m", "1d", cfg["kind"], live_tail=True)["Close"].iloc[-1])
                    rows = []
                    if cfg["trade_mode"] in ("Index Options", "Stock Options"):
                        for d_ in (1, -1):
                            r = _resolve_instrument(cfg, d_, spot)
                            rows.append({"Leg": r.get("opt_type"), "Strike": r.get("strike"),
                                         "Mode": "Manual" if cfg["dhan"].get(("ce" if d_ == 1 else "pe") + "_manual") else
                                         cfg["dhan"].get(("ce" if d_ == 1 else "pe") + "_strike", "ATM"), "Expiry": r.get("expiry"),
                                         "Symbol": r.get("symbol"), "Security ID": r.get("security_id"), "Lot": r.get("lot"), "Error": r.get("error")})
                    else:
                        r = _resolve_instrument(cfg, 1, spot)
                        rows.append({"Symbol": r.get("symbol"), "Security ID": r.get("security_id"), "Segment": r.get("segment"),
                                     "Lot": r.get("lot"), "Error": r.get("error")})
                    st.session_state["resolved"] = (spot, rows)
                except Exception as e:
                    st.error(f"Resolution failed: {e}")
            if st.session_state.get("resolved"):
                sp, rows = st.session_state["resolved"]
                st.caption(f"Spot used: {fnum(sp)} — IDs are re-resolved at every entry so ATM follows the market.")
                st.dataframe(pd.DataFrame(rows), hide_index=True, **WIDE)
    c1, c2, c4 = st.columns([1, 1.4, 2.2])
    if c1.button("▶️ Start", type="primary", disabled=L["running"], key="m_live_start", **WIDE):
        L.update(running=True, started=now_ist(), error=None, first_tick=True, armed_up=True, armed_dn=True)
        _log(L, f"Live trading STARTED — {cfg['symbol']} {cfg['tf']} {cfg['strategy']}")
        st.rerun()
    if c2.button("⏹️ Stop (square-off & stop)", disabled=not L["running"] and L["pos"] is None, key="m_live_stop", **WIDE):
        if L["pos"] is not None:
            ltp = L.get("ltp")
            try:
                _, ltp = _live_fetch(cfg, L)
            except Exception as e:
                _log(L, f"Fresh LTP unavailable at stop, using last LTP: {e}", "warn")
            _exit(cfg, L, ltp, "Stopped by user (square-off)")
        L["running"] = False
        L["app_rerun"] = False
        _log(L, "Live trading STOPPED", "warn")
        st.rerun()
    badge = "<span class='qe-badge qe-run'>● RUNNING</span>" if L["running"] else "<span class='qe-badge qe-off'>■ STOPPED</span>"
    c4.markdown(f"<div style='text-align:right;padding-top:6px'>{badge}</div>", unsafe_allow_html=True)

    @st.fragment(run_every=cfg["refresh"] if L["running"] else None)
    def _live_fragment():
        # widget lives INSIDE the fragment → clicking it reruns only the fragment; live trading keeps running
        if st.button("🟧 Manual Square-off (keep trading)", disabled=L["pos"] is None, key="m_live_sq"):
            try:
                ltp = L["ltp"]
                if L["running"]:
                    try:
                        _, ltp = _live_fetch(cfg, L)
                    except Exception as e:
                        _log(L, f"Fresh LTP unavailable, using last LTP: {e}", "warn")
                _exit(cfg, L, ltp, "Manual square-off")
                L["last_exit_closed"] = L["last_closed"]
                _log(L, "Manual square-off done — engine keeps scanning for the next signal")
            except Exception as e:
                st.error(f"Square-off failed: {e}")
        if L["running"]:
            try:
                live_tick(cfg)
                L["error"] = None
            except Exception as e:
                L["error"] = str(e)
                _log(L, f"Tick error (will retry): {e}", "error")
        if L.get("app_rerun"):
            L["app_rerun"] = False
            st.rerun(scope="app")  # instantly refresh Trade History (running state is preserved)
        _render_live(cfg, L)

    _live_fragment()


def _render_live(cfg: dict, L: dict):
    pos: Optional[Position] = L["pos"]
    ltp = L.get("ltp")
    if L.get("error"):
        st.warning(f"Last tick error (auto-retrying): {L['error']}")
    if pos is not None:
        cls = "qe-on" if pos.direction == 1 else "qe-short"
        st.markdown(f"<span class='qe-badge {cls}'>● POSITION ACTIVE — {pos.label} "
                    f"{pos.instrument.get('symbol', '') if pos.instrument else ''} × {pos.qty}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='qe-badge qe-off'>○ NO POSITION — FLAT / CLOSED</span>", unsafe_allow_html=True)
    st.write("")
    if ltp is None:
        st.info("Press **Start** — the panel refreshes itself in place (no page reload).")
        return
    d = pos.direction if pos else 0
    pts = (ltp - pos.entry) * d if pos else 0.0
    if pos and pos.inst_entry and pos.inst_ltp and pos.instrument.get("kind") == "option":
        pnl = (pos.inst_ltp - pos.inst_entry) * pos.qty
    else:
        pnl = pts * (pos.qty if pos else 0)
    r = st.columns(6)
    delta_ltp = (ltp - L["prev_ltp"]) if L.get("prev_ltp") else None
    r[0].metric("LTP", fnum(ltp), delta=fnum(delta_ltp) if delta_ltp else None)
    r[1].metric("Entry", fnum(pos.entry) if pos else "—")
    r[2].metric("Stop-loss", fnum(pos.sl) if pos else "—")
    r[3].metric("Target", fnum(pos.tgt) if pos else "—")
    r[4].metric("Live P&L ₹", fnum(pnl) if pos else "—", delta=f"{fnum(pts)} pts" if pos else None)
    r[5].metric("Duration", fmt_duration((now_ist() - pos.entry_time).total_seconds()) if pos else "—")
    r = st.columns(6)
    r[0].metric("Highest in trade", fnum(pos.hi) if pos else "—")
    r[1].metric("Lowest in trade", fnum(pos.lo) if pos else "—")
    r[2].metric("Distance to SL", fnum(abs(ltp - pos.sl)) if pos and pos.sl == pos.sl else "—")
    r[3].metric("Distance to Target", fnum(abs(pos.tgt - ltp)) if pos and pos.tgt == pos.tgt else "—")
    r[4].metric("Entry time (IST)", pos.entry_time.strftime("%H:%M:%S") if pos else "—")
    r[5].metric("Option/Fut LTP", fnum(pos.inst_ltp) if pos and pos.inst_ltp else "—")
    st.caption(f"Ticks {L['ticks']} · last update {L['last_update']:%H:%M:%S} IST · "
               f"{'current candle forming' if L.get('forming') else 'last candle closed'} · started {L['started']:%H:%M:%S}"
               if L.get("last_update") and L.get("started") else "")
    x: Ctx = L.get("x")
    vi = (L.get("sig") or {}).get("vol_info")
    if vi:
        st.info("📊 " + vix_info_text(vi))
    cA, cB = st.columns([1, 1.4])
    with cA:
        st.markdown("**Strategy parameters (last closed candle)**")
        rows = []
        if x is not None and x.n:
            rows.append(("EMA 9", fnum(x.ema(9)[-1])))
            rows.append(("EMA 21", fnum(x.ema(21)[-1])))
            if L.get("sig"):
                for k_, v in L["sig"]["lines"].items():
                    if k_ not in ("EMA 9", "EMA 21"):
                        rows.append((k_, fnum(np.asarray(v)[-1])))
            rows.append(("ATR 14", fnum(x.atr(14)[-1])))
            rows.append(("RSI 14", fnum(x.rsi(14)[-1])))
        st.dataframe(pd.DataFrame(rows, columns=["Indicator", "Value"]), hide_index=True, **WIDE)
    with cB:
        st.markdown("**Strategy status board**")
        for s_ in _status_board(cfg, L):
            st.markdown(f"- {s_}")
        if L.get("report"):
            st.dataframe(pd.DataFrame(L["report"], columns=["Filter", "Long OK", "Short OK"]), hide_index=True,
                         **WIDE)
    df = L.get("df")
    if df is not None and x is not None:
        lines = {"EMA 9": ema(df["Close"], 9), "EMA 21": ema(df["Close"], 21)}
        fig = candle_chart(df, lines, f"{cfg['symbol']} · {cfg['tf']} · live", pos=pos, height=560, max_bars=cfg["chart_bars"])
        if fig:
            st.plotly_chart(fig, **WIDE, key="m_live_chart")
    with st.expander("📜 Event log", expanded=False):
        if L["log"]:
            st.dataframe(pd.DataFrame(L["log"]), hide_index=True, **WIDE, height=240)
        for e in _GLOBAL_ERRORS[-5:]:
            st.caption(e)


# =============================================================================
#  TRADE HISTORY TAB (live trades only)
# =============================================================================
def history_tab(cfg: dict):
    L = _live_state()
    st.markdown("#### 📒 Live Trade History")
    st.caption(f"Live trades only (backtests are not stored). Workspace: **{cfg['user']}** · persisted in SQLite.")

    @st.fragment(run_every=max(cfg["refresh"], 3.0) if L["running"] else None)
    def _hist():
        recs = db_load_trades(cfg["user"])
        if not recs:
            st.info("No live trades yet — closed trades appear here automatically.")
            return
        h = pd.DataFrame(recs)
        pts = h["Points"].astype(float)
        net = h["PnL (Net)"].astype(float)
        wins = int((pts > 0).sum())
        summ = {"Total Trades": len(h), "Won": wins, "Lost": len(h) - wins,
                "Accuracy %": round(100 * wins / len(h), 2), "Points Won": round(pts[pts > 0].sum(), 2),
                "Points Lost": round(pts[pts <= 0].sum(), 2), "Net Points": round(pts.sum(), 2),
                "PnL Gross": round(h["PnL (Gross)"].astype(float).sum(), 2), "Charges": round(h["Charges"].astype(float).sum(), 2),
                "Total PnL (Net)": round(net.sum(), 2)}
        r = st.columns(5)
        r[0].metric("Total Trades", summ["Total Trades"])
        r[1].metric("Won / Lost", f"{summ['Won']} / {summ['Lost']}")
        r[2].metric("Accuracy", f"{summ['Accuracy %']}%")
        r[3].metric("Net Points", fnum(summ["Net Points"]))
        r[4].metric("Total P&L ₹ (net)", fnum(summ["Total PnL (Net)"]))
        r = st.columns(4)
        r[0].metric("Points Won", fnum(summ["Points Won"]))
        r[1].metric("Points Lost", fnum(summ["Points Lost"]))
        r[2].metric("P&L Gross ₹", fnum(summ["PnL Gross"]))
        r[3].metric("Charges ₹", fnum(summ["Charges"]))
        cols = ["Entry Time (IST)", "Exit Time (IST)", "Duration", "Ticker", "Strategy", "Action", "Instrument", "Entry Price",
                "Exit Price", "Initial SL", "Final SL", "Target", "Entry Candle Time", "Entry Candle Open", "Entry Candle High",
                "Entry Candle Low", "Entry Candle Close", "Exit Candle Time", "Exit Candle Open", "Exit Candle High",
                "Exit Candle Low", "Exit Candle Close", "Highest Price", "Lowest Price", "Points", "Qty",
                "PnL (Gross)", "Charges", "PnL (Net)", "Exit Reason", "Inst Entry", "Inst Exit", "Orders", "Trade ID"]
        h = h[[c for c in cols if c in h.columns]].iloc[::-1]
        st.dataframe(h, hide_index=True, **WIDE, height=420)
        export_buttons(h, summ, "live_trades")

    _hist()
    with st.expander("Danger zone"):
        if st.button("🗑️ Clear my live trade history", key="m_clr_hist"):
            db_clear(cfg["user"])
            st.session_state["history"] = []
            st.rerun()


# =============================================================================
#  SCREENER
# =============================================================================
NIFTY50_FALLBACK = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
                    "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
                    "HDFCLIFE", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK",
                    "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
                    "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TMPV", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"]
NSE_LIST_URL = "https://archives.nseindia.com/content/indices/{}"
INDEX_CSV = {"NIFTY 50 stocks": "ind_nifty50list.csv", "NIFTY 100 stocks": "ind_nifty100list.csv",
             "NIFTY 200 stocks": "ind_nifty200list.csv", "NIFTY 500 stocks": "ind_nifty500list.csv"}
SECTOR_CSV = {"NIFTY BANK": "ind_niftybanklist.csv", "NIFTY IT": "ind_niftyitlist.csv", "NIFTY AUTO": "ind_niftyautolist.csv",
              "NIFTY PHARMA": "ind_niftypharmalist.csv", "NIFTY FMCG": "ind_niftyfmcglist.csv", "NIFTY METAL": "ind_niftymetallist.csv",
              "NIFTY REALTY": "ind_niftyrealtylist.csv", "NIFTY ENERGY": "ind_niftyenergylist.csv", "NIFTY MEDIA": "ind_niftymedialist.csv",
              "NIFTY PSU BANK": "ind_niftypsubanklist.csv", "NIFTY FIN SERVICE": "ind_niftyfinancelist.csv"}
KEY_INDICES = ["^NSEI", "^NSEBANK", "^BSESN", "NIFTY_FIN_SERVICE.NS"]
BROAD_INDICES = ["^NSEI", "^NSEBANK", "^BSESN", "NIFTY_FIN_SERVICE.NS", "^NSMIDCP", "^CNX100", "^CRSLDX", "NIFTY_MIDCAP_100.NS",
                 "^NSEMDCP50"]
SECTOR_INDICES = {"NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT", "NIFTY AUTO": "^CNXAUTO", "NIFTY PHARMA": "^CNXPHARMA",
                  "NIFTY FMCG": "^CNXFMCG", "NIFTY METAL": "^CNXMETAL", "NIFTY REALTY": "^CNXREALTY", "NIFTY ENERGY": "^CNXENERGY",
                  "NIFTY MEDIA": "^CNXMEDIA", "NIFTY PSU BANK": "^CNXPSUBANK", "NIFTY INFRA": "^CNXINFRA",
                  "NIFTY PVT BANK": "NIFTY_PVT_BANK.NS", "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS", "NIFTY PSE": "^CNXPSE",
                  "NIFTY MNC": "^CNXMNC", "NIFTY COMMODITIES": "^CNXCMDT", "NIFTY CONSUMPTION": "^CNXCONSUM"}
CRYPTO_FAMILY = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD",
                 "LINK-USD", "LTC-USD", "TRX-USD"]
CORE_MACRO = ["BTC-USD", "ETH-USD", "INR=X", "GC=F", "SI=F"]
UNIVERSES = ["Key indices (NIFTY/BANKNIFTY/SENSEX/FINNIFTY)", "Broad indices", "NIFTY sector indices", "NIFTY 50 stocks",
             "NIFTY 100 stocks", "NIFTY 200 stocks", "NIFTY 500 stocks", "Sector constituents", "Crypto family",
             "BTC / ETH / USDINR / Gold / Silver", "Forex majors", "Commodities", "Custom tickers"]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def nse_constituents(csv_name: str) -> List[str]:
    r = requests.get(NSE_LIST_URL.format(csv_name), timeout=20,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"})
    r.raise_for_status()
    d = pd.read_csv(io.BytesIO(r.content))
    col = [c for c in d.columns if c.strip().lower() == "symbol"][0]
    return [f"{s.strip()}.NS" for s in d[col].astype(str)]


def build_universe(sel: List[str], sector: str, custom: str) -> Tuple[List[str], List[str]]:
    out, notes = [], []
    for u in sel:
        if u.startswith("Key indices"):
            out += KEY_INDICES
        elif u == "Broad indices":
            out += BROAD_INDICES
        elif u == "NIFTY sector indices":
            out += list(SECTOR_INDICES.values())
        elif u in INDEX_CSV:
            try:
                out += nse_constituents(INDEX_CSV[u])
            except Exception as e:
                notes.append(f"{u}: NSE list download failed ({e}).")
                if u == "NIFTY 50 stocks":
                    out += [f"{s}.NS" for s in NIFTY50_FALLBACK]
                    notes.append("Using built-in NIFTY 50 list (may be slightly outdated).")
        elif u == "Sector constituents":
            try:
                out += nse_constituents(SECTOR_CSV[sector])
            except Exception as e:
                notes.append(f"{sector} constituents download failed ({e}).")
        elif u == "Crypto family":
            out += CRYPTO_FAMILY
        elif u.startswith("BTC / ETH"):
            out += CORE_MACRO
        elif u == "Forex majors":
            out += list(FOREX_PAIRS.values())
        elif u == "Commodities":
            out += list(COMMODITIES.values())
        elif u == "Custom tickers":
            out += [t.strip().upper() for t in custom.replace("\n", ",").split(",") if t.strip()]
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq, notes


def meta_for_symbol(sym: str) -> dict:
    for g, m in TICKER_GROUPS.items():
        if m["yf"] == sym:
            return dict(m)
    k = infer_kind(sym)
    return {"kind": k, "und": sym.split(".")[0], "fno_exch": "BSE" if sym.endswith(".BO") else "NSE"}


def sidebar_keys_for_symbol(sym: str) -> dict:
    for g, m in TICKER_GROUPS.items():
        if m["yf"] == sym:
            return {"tsel": g}
    for pair, s_ in FOREX_PAIRS.items():
        if s_ == sym:
            return {"tsel": "Forex", "fx": pair}
    for com, s_ in COMMODITIES.items():
        if s_ == sym:
            return {"tsel": "Commodities", "com": com}
    return {"tsel": "Custom Ticker", "custom_sym": sym}


_NON_SIDEBAR = {"live", "bt", "history", "resolved", "scr", "rates_ed", "_tick"}


def sidebar_snapshot() -> dict:
    """All sidebar widget values (main-area widgets use the 'm_' prefix and are excluded)."""
    snap = {}
    for k, v in st.session_state.items():
        if not isinstance(k, str) or k in _NON_SIDEBAR or k.startswith(("m_", "_", "FormSubmitter")):
            continue
        if isinstance(v, (str, int, float, bool)) and not isinstance(v, np.generic):
            snap[k] = v
    return snap


def signal_arrays(x: Ctx, cfg: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    name = cfg["strategy"]
    if name in ("Simple Buy", "Simple Sell"):
        return None
    sd = STRATEGIES[name]
    sig = run_strategy(x, name, cfg["params"], cfg.get("hybrid")) if (sd.func or name.startswith("Hybrid")) else None
    if name in IMMEDIATE:
        up, ud, dn, dd = _immediate_levels(x, cfg, sig)
        pc = shift(x.c)
        with np.errstate(invalid="ignore"):
            hu = (pc < up) & (x.h >= up)
            hd = (pc > dn) & (x.l <= dn)
        L = (hu & (ud == 1)) | (hd & (dd == 1))
        S = (hu & (ud == -1)) | (hd & (dd == -1))
    else:
        L, S, _ = apply_filters(x, sig["long"], sig["short"], cfg["filters"])
    if cfg["filters"].get("flip"):
        L, S = S, L
    return np.asarray(L, bool), np.asarray(S, bool)


def _project(df: pd.DataFrame, steps: int, direction: int, atr_v: float, mult: float, tf: str) -> pd.DataFrame:
    """Append `steps` hypothetical candles drifting `mult`×ATR per candle (used to find signals that are ABOUT to fire)."""
    step = pd.Timedelta(minutes=TF_MINUTES[tf]) if tf in INTRADAY_TFS else pd.Timedelta(days={"1d": 1, "3d": 3, "1wk": 7}.get(tf, 30))
    c = float(df["Close"].iloc[-1])
    vol = float(df["Volume"].tail(20).mean() or 0)
    rows, idx = [], []
    for i in range(1, steps + 1):
        o = c
        c = o + direction * mult * atr_v
        rows.append((o, max(o, c) + 0.1 * atr_v, min(o, c) - 0.1 * atr_v, c, vol))
        idx.append(df.index[-1] + step * i)
    ext = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["Open", "High", "Low", "Close", "Volume"])
    return pd.concat([df, ext])


def screen_one(sym: str, cfg: dict, n_c: int, proj_mult: float) -> Optional[dict]:
    meta = meta_for_symbol(sym)
    c2 = dict(cfg, symbol=sym, meta=meta, kind=meta["kind"], indian=is_indian(sym, meta["kind"]))
    if c2["strategy"] == VIX_NAME and not (meta["kind"] == "index" and c2["indian"]):
        c2["params"] = dict(c2["params"], vol_source=VOL_SOURCES[2] if c2["params"]["vol_source"].startswith("Auto") else c2["params"]["vol_source"])
    src = cfg["source"] if (cfg["source"] == "Dhan" and meta["kind"] in ("index", "stock") and c2["indian"]) else "yfinance"
    df = get_history(src, sym, c2["tf"], c2["period"], json.dumps(meta), cfg["dhan_cid"], cfg["dhan_tok"])
    if df is None or len(df) < 60:
        return {"Ticker": sym, "Bucket": "⚠️ no / short data", "_b": 9}
    closed, _ = _split_closed(df, c2["tf"])
    daily = get_daily(sym) if needs_daily(c2) else None
    aux = dict(aux_for(c2), **make_aux(c2))
    x = Ctx(closed, daily, aux)
    arr = signal_arrays(x, c2)
    if arr is None:
        return {"Ticker": sym, "Bucket": "n/a (immediate Simple Buy/Sell)", "_b": 9}
    L, S = arr
    n = x.n
    a = x.atr(14)[-1]
    _, _, adx_ = x.dmi()
    base = {"Ticker": sym, "LTP": round(float(df["Close"].iloc[-1]), 4), "ADX": round(float(adx_[-1]), 1),
            "RSI": round(float(x.rsi(14)[-1]), 1), "ATR %": round(float(a / x.c[-1] * 100), 3) if a == a else np.nan}
    for ago in range(0, n_c + 1):
        j = n - 1 - ago
        if j < 0:
            break
        if L[j] or S[j]:
            return dict(base, Bucket="🟢 Just now" if ago == 0 else f"🕒 Fired {ago} candle(s) ago",
                        Direction="LONG" if L[j] else "SHORT", Candles=ago, **{"Signal candle": fmt_ts(x.df.index[j])},
                        _b=0 if ago == 0 else 1)
    best = None
    if a == a and a > 0:
        for d in (1, -1):
            xe = Ctx(_project(closed, n_c, d, a, proj_mult, c2["tf"]), daily, aux)
            ar2 = signal_arrays(xe, c2)
            if ar2 is None:
                continue
            L2, S2 = ar2
            for i in range(n, xe.n):
                if L2[i] or S2[i]:
                    ahead = i - n + 1
                    if best is None or ahead < best[0]:
                        best = (ahead, "LONG" if L2[i] else "SHORT", "up" if d == 1 else "down")
                    break
    if best:
        return dict(base, Bucket=f"🔮 Projected in {best[0]} candle(s)", Direction=best[1], Candles=best[0],
                    **{"Signal candle": f"if price drifts {best[2]} ~{proj_mult}×ATR/candle"}, _b=2)
    return dict(base, Bucket="— no signal", Direction="—", Candles=np.nan, _b=8)


def screener_tab(cfg: dict):
    st.markdown("#### 🔎 Screener")
    st.caption("Uses the sidebar timeframe, period, strategy, parameters and entry filters. Buckets: 🟢 signal on the last closed "
               "candle · 🕒 fired within the last N candles · 🔮 would fire within the next N candles if price keeps drifting "
               "(projection = N hypothetical candles of ±k×ATR). Ranked by bucket → candles → ADX.")
    c1, c2 = st.columns([2, 1])
    sel = c1.multiselect("Universe", UNIVERSES, default=["Key indices (NIFTY/BANKNIFTY/SENSEX/FINNIFTY)", "Custom tickers"], key="m_scr_univ")
    sector = c2.selectbox("Sector (for 'Sector constituents')", list(SECTOR_CSV), key="m_scr_sector")
    custom = st.text_area("Custom tickers (Yahoo symbols, comma / new-line separated)", "KAYNES.NS", key="m_scr_custom", height=68)
    c1, c2, c3, c4 = st.columns(4)
    n_c = int(c1.number_input("N candles (before / after)", 1, 50, 3, key="m_scr_n"))
    proj = float(c2.number_input("Projection drift (×ATR per candle)", 0.05, 5.0, 0.5, 0.05, key="m_scr_proj"))
    max_t = int(c3.number_input("Max tickers", 1, 1000, 150, key="m_scr_max"))
    show_all = c4.checkbox("Show 'no signal' rows", False, key="m_scr_all")
    if st.button("🔎 Screen tickers", type="primary", key="m_scr_run", **WIDE):
        syms, notes = build_universe(sel, sector, custom)
        for n_ in notes:
            st.warning(n_)
        syms = syms[:max_t]
        rows, prog = [], st.progress(0.0, text="Screening…")
        t0 = time.time()
        for i, sym in enumerate(syms):
            prog.progress((i + 1) / max(len(syms), 1), text=f"{sym} ({i + 1}/{len(syms)})")
            try:
                r = screen_one(sym, cfg, n_c, proj)
            except Exception as e:
                r = {"Ticker": sym, "Bucket": f"⚠️ error: {str(e)[:80]}", "_b": 9}
            if r:
                rows.append(r)
        prog.empty()
        st.session_state["scr"] = {"rows": rows, "snapshot": sidebar_snapshot(), "elapsed": time.time() - t0,
                                   "desc": f"{cfg['strategy']} · {cfg['tf']} · {cfg['period']} · filters: {', '.join(cfg['filters']['active']) or 'none'}"}
    res = st.session_state.get("scr")
    if not res:
        st.info("Pick a universe and click **Screen tickers**.")
        return
    df = pd.DataFrame(res["rows"])
    for c in ("Direction", "Candles", "Signal candle", "LTP", "ADX", "RSI", "ATR %"):
        if c not in df.columns:
            df[c] = np.nan
    df["_adx"] = pd.to_numeric(df["ADX"], errors="coerce").fillna(-1)
    df["_c"] = pd.to_numeric(df["Candles"], errors="coerce").fillna(999)
    df = df.sort_values(["_b", "_c", "_adx"], ascending=[True, True, False])
    hits = df[df["_b"] <= 2].copy()
    hits.insert(0, "Rank", range(1, len(hits) + 1))
    st.caption(f"{res['desc']} · scanned {len(df)} tickers in {res['elapsed']:.1f}s · {len(hits)} with signals")
    cols = ["Rank", "Ticker", "Bucket", "Direction", "Candles", "Signal candle", "LTP", "ADX", "RSI", "ATR %"]
    m = st.columns(3)
    m[0].metric("🟢 Just now", int((hits["_b"] == 0).sum()))
    m[1].metric(f"🕒 Last {n_c} candles", int((hits["_b"] == 1).sum()))
    m[2].metric(f"🔮 Next {n_c} candles", int((hits["_b"] == 2).sum()))
    if hits.empty:
        st.info(f"No ticker signalled in the last {n_c} candles or is projected to within the next {n_c}. "
                "Try a larger N, a larger projection drift, fewer filters or another timeframe.")
    else:
        st.dataframe(hits[cols], hide_index=True, height=360, **WIDE)
    if show_all:
        st.dataframe(df[df["_b"] > 2][["Ticker", "Bucket", "LTP", "ADX", "RSI"]], hide_index=True, **WIDE)
    if len(hits):
        export_buttons(hits[cols], {"Strategy": res["desc"], "Hits": len(hits)}, "screener")
        labels = [f"#{r.Rank} · {r.Ticker} · {r.Direction} · {r.Bucket}" for r in hits.itertuples()]
        pick = st.selectbox("Ranked tickers", labels, key="m_scr_pick")
        if st.button("✅ Apply selected ticker + screen configuration to sidebar", key="m_scr_apply", **WIDE):
            sym = hits.iloc[labels.index(pick)]["Ticker"]
            pend = dict(res["snapshot"])
            for k_ in ("fx", "com", "custom_sym"):
                pend.pop(k_, None)
            pend.update(sidebar_keys_for_symbol(sym))
            st.session_state["_apply_pending"] = pend
            st.session_state.pop("bt", None)
            st.rerun()


# =============================================================================
#  MAIN
# =============================================================================
def main():
    st.set_page_config(page_title=f"{APP_NAME} · Algo Trading", page_icon="⚡", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    pend = st.session_state.pop("_apply_pending", None)
    if pend:  # values pushed from the screener — written BEFORE widgets are created
        for k, v in pend.items():
            st.session_state[k] = v
        st.toast(f"Applied {pend.get('custom_sym') or pend.get('tsel')} with the screened configuration")
    cfg = sidebar_config()
    L = _live_state()
    st.markdown(f"<div class='qe-hero'><h1>⚡ {APP_NAME}</h1><span>{cfg['symbol']} · {cfg['tf']} · {cfg['strategy']} · "
                f"{cfg['source']} · {now_ist():%d %b %Y %H:%M} IST</span></div>", unsafe_allow_html=True)
    if L["running"] and L.get("data_key") and L["data_key"] != _data_key(cfg):
        st.warning("Sidebar data settings changed while live — the engine will reload warm-up history on the next tick.")
    t1, t2, t3, t4 = st.tabs(["🧪 Backtesting", "📡 Live Trading", "📒 Trade History", "🔎 Screener"])
    with t1:
        backtest_tab(cfg)
    with t2:
        live_tab(cfg)
    with t3:
        history_tab(cfg)
    with t4:
        screener_tab(cfg)
    st.caption("⚠️ Educational/engineering tool. Backtests use OHLC bars (no tick data) with conservative SL-first fills; "
               "option P&L in backtests uses underlying points. Verify every order with your broker.")


if __name__ == "__main__":
    main()
