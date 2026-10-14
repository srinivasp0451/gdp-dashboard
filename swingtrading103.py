"""
Algo Trading PLUS — upgrade layer for your existing Algorithmic Trading System
==============================================================================

Your original file is NOT modified. This file loads it and upgrades it in place.

HOW TO RUN
    1. Save your original code as  algo_trading.py  (same folder as this file)
       (or set env var ALGO_BASE_FILE=/path/to/your_file.py)
    2. pip install streamlit yfinance pandas numpy scipy plotly pytz requests dhanhq
    3. streamlit run algo_trading_plus.py

WHAT IT ADDS (all existing settings keep working exactly as before)
    ⚡ Data & Indicator Engine
        • Warm-up history pre-load  → indicators are ready on the FIRST candle of the
          session (no waiting 20 candles for EMA-20), gap-up/gap-down safe
        • TradingView-exact math  → EMA (SMA-seeded), RSI/ATR/ADX (Wilder RMA),
          Bollinger (population stdev), MACD, session-anchored VWAP
        • Dhan data source (optional) with automatic Security-ID resolution for
          indices (spot), index/stock futures (nearest expiry) and NSE/BSE stocks
    🎯 Signal Quality
        • Confirm on candle close (no repaint)   • Fresh-signal-only for every strategy
        • One entry per signal candle in live     • Auto ATM CE/PE IDs for Dhan orders
    🧰 22 optional entry filters (all OFF by default, defaults editable below)
"""

import os
import importlib.util
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

IST = pytz.timezone("Asia/Kolkata")

# ════════════════════════════════════════════════════════════════════════════
# EDIT DEFAULTS HERE — these are the values the filter boxes start with
# ════════════════════════════════════════════════════════════════════════════
FILTER_DEFAULTS = {
    "adx_min": 20.0, "adx_max": 50.0,
    "rsi_period": 14,
    "rsi_long_min": 50.0, "rsi_long_max": 70.0,
    "rsi_short_min": 30.0, "rsi_short_max": 50.0,
    "bb_period": 20, "bb_std": 2.0, "bb_bw_min": 0.0, "bb_bw_max": 100.0,
    "sma_period": 50,
    "ema_period": 200,
    "vol_nx": 1.5,
    "avgvol_nx": 1.5, "avgvol_period": 20,
    "vp_lookback": 120, "vp_bins": 30, "vp_value_area": 70.0,
    "swing_len": 5,
    "bos_lookback": 20,
    "choch_lookback": 20,
    "fvg_lookback": 15, "fvg_min_size_atr": 0.10,
    "ict_killzones": "09:15-10:30,13:30-15:00",
    "ict_ote_low": 0.62, "ict_ote_high": 0.79,
    "angle_min": 10.0, "angle_max": 90.0,
    "pullback_window": 10, "pullback_tol_atr": 0.10,
    "sustain_candles": 2,
    "sweep_lookback": 10,
    "atr_period": 14, "atr_min": 0.0, "atr_max": 1000.0,
    "atr_pct_min": 0.0, "atr_pct_max": 5.0,
    "chop_period": 14, "chop_max": 50.0, "regime_adx_min": 20.0,
    "vwap_max_dist_pct": 1.0,
    "htf_minutes": 15, "htf_ema": 50,
}

# ════════════════════════════════════════════════════════════════════════════
# LOAD ORIGINAL FILE (untouched)
# ════════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.environ.get("ALGO_BASE_FILE", os.path.join(_HERE, "algo_trading.py"))
_BASE_MARKERS = ("class DhanBrokerIntegration", "def render_config_ui", "def live_trading_iteration")


def _find_base_file():
    """Use algo_trading.py if present; otherwise auto-detect the original trading file
    (any .py in this folder or its sub-folders that contains the original app's code)."""
    if os.path.exists(BASE_FILE):
        return BASE_FILE
    me = os.path.abspath(__file__)
    for root, dirs, files in os.walk(_HERE):
        dirs[:] = [d for d in dirs if not d.startswith((".", "__")) and d not in ("venv", ".venv", "site-packages")]
        for fn in sorted(files):
            path = os.path.abspath(os.path.join(root, fn))
            if not fn.endswith(".py") or path == me:
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:
                continue
            if all(m in text for m in _BASE_MARKERS) and "algo_trading_plus" not in fn:
                return path
    return None


def _load_base():
    path = _find_base_file()
    if path is None:
        st.set_page_config(page_title="Algo Trading PLUS", layout="wide")
        here = sorted(f for f in os.listdir(_HERE) if f.endswith(".py"))
        st.error(
            "**Original trading code not found.**\n\n"
            "Add your original code to this repo as **`algo_trading.py`** (same folder as "
            "`algo_trading_plus.py`) and set the Streamlit Cloud *Main file path* to "
            f"`algo_trading_plus.py`.\n\nPython files found here: {', '.join(here) or 'none'}")
        st.stop()
    spec = importlib.util.spec_from_file_location("algo_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_base()

# ── Originals (kept so every existing behaviour is still reachable) ─────────
_orig_fetch = base.fetch_data
_orig_calc = base.calculate_all_indicators
_orig_rsi = base.calculate_rsi
_orig_atr = base.calculate_atr
_orig_adx = base.calculate_adx
_orig_render_config = base.render_config_ui
_orig_render_bt = base.render_backtest_ui
_orig_run_bt = base.run_backtest
_orig_enter_broker = base.DhanBrokerIntegration.enter_broker_position
_RAW_STRATEGIES = dict(base.STRATEGY_FUNCTIONS)

_ACTIVE_CFG = {}
_IN_BACKTEST = False
_FILTER_STATS = {}
_LOCAL_SS = {}


def _ss():
    """st.session_state that also works outside a Streamlit run (tests)."""
    try:
        return st.session_state
    except Exception:
        return _LOCAL_SS


def _cfg():
    if _ACTIVE_CFG:
        return _ACTIVE_CFG
    try:
        return _ss().get("_active_config", {}) or {}
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════════════
# TRADINGVIEW-EXACT INDICATOR MATH
# ════════════════════════════════════════════════════════════════════════════
def _seeded_recursive(series, length, alpha):
    """Pine ta.ema / ta.rma: first value = SMA(length), then recursive smoothing."""
    s = pd.Series(series, dtype="float64")
    vals = s.to_numpy()
    n = max(1, int(length))
    out = np.full(len(vals), np.nan)
    cnt, start = 0, None
    for i, v in enumerate(vals):
        cnt = cnt + 1 if not np.isnan(v) else 0
        if cnt >= n:
            start = i
            break
    if start is not None:
        out[start] = vals[start - n + 1:start + 1].mean()
        for i in range(start + 1, len(vals)):
            v = vals[i]
            out[i] = out[i - 1] if np.isnan(v) else alpha * v + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=s.index)


def tv_ema(series, length):
    return _seeded_recursive(series, length, 2.0 / (int(length) + 1))


def tv_rma(series, length):
    return _seeded_recursive(series, length, 1.0 / int(length))


def tv_true_range(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def tv_rsi(series, period=14):
    ch = pd.Series(series, dtype="float64").diff()
    up = tv_rma(ch.clip(lower=0), period)
    dn = tv_rma((-ch).clip(lower=0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + up / dn)
    rsi[(dn == 0) & up.notna()] = 100.0
    return rsi


def tv_atr(df, period=14):
    return tv_rma(tv_true_range(df), period)


def tv_adx(df, period=14):
    h, l = df["High"], df["Low"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    plus[up.isna()] = np.nan
    minus[dn.isna()] = np.nan
    trur = tv_rma(tv_true_range(df), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = (100 * tv_rma(plus, period) / trur).ffill()
        mdi = (100 * tv_rma(minus, period) / trur).ffill()
        s = pdi + mdi
        adx = 100 * tv_rma((pdi - mdi).abs() / s.where(s != 0, 1), period)
    return adx


def ols_slope(series, lookback=3):
    """Vectorised version of the original polyfit slope (identical numbers, ~100x faster)."""
    vals = pd.Series(series, dtype="float64").to_numpy()
    n = lookback + 1
    out = np.full(len(vals), np.nan)
    if len(vals) >= n:
        x = np.arange(n) - (n - 1) / 2.0
        w = np.lib.stride_tricks.sliding_window_view(vals, n)
        out[n - 1:] = (w * x).sum(axis=1) / (x ** 2).sum()
    return pd.Series(out, index=pd.Series(series).index)


def fast_ema_angle(ema_series, lookback=3):
    return np.degrees(np.arctan(ols_slope(ema_series, lookback)))


def has_volume(df):
    return "Volume" in df.columns and float(df["Volume"].tail(100).fillna(0).sum()) > 0


def session_vwap(df):
    """TradingView default VWAP: hlc3, anchored to each session (resets daily)."""
    if not has_volume(df):
        return pd.Series(np.nan, index=df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].fillna(0)
    d = df["Datetime"].dt.date
    pv = (tp * v).groupby(d).cumsum()
    vv = v.groupby(d).cumsum()
    return pv / vv.replace(0, np.nan)


def choppiness(df, n=14):
    tr = tv_true_range(df)
    rng = df["High"].rolling(n).max() - df["Low"].rolling(n).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)


def _tv_on():
    return _cfg().get("tv_exact_indicators", True)


def _patched_rsi(series, period=14):
    return tv_rsi(series, period) if _tv_on() else _orig_rsi(series, period)


def _patched_atr(df, period=14):
    return tv_atr(df, period) if _tv_on() else _orig_atr(df, period)


def _patched_adx(df, period=14):
    return tv_adx(df, period) if _tv_on() else _orig_adx(df, period)


def calculate_all_indicators_plus(df, config):
    global _ACTIVE_CFG
    config = config or {}
    _ACTIVE_CFG = config
    df = _orig_calc(df, config)
    close = df["Close"]
    fast, slow = int(config.get("ema_fast", 9)), int(config.get("ema_slow", 21))

    if config.get("tv_exact_indicators", True):
        df["EMA_Fast"] = tv_ema(close, fast)
        df["EMA_Slow"] = tv_ema(close, slow)
        p, k = int(config.get("custom_bb_period", 20)), float(config.get("custom_bb_std", 2.0))
        mid, sd = close.rolling(p).mean(), close.rolling(p).std(ddof=0)
        df["BB_Middle"], df["BB_Upper"], df["BB_Lower"] = mid, mid + k * sd, mid - k * sd
        macd = tv_ema(close, 12) - tv_ema(close, 26)
        df["MACD"], df["MACD_Signal"] = macd, tv_ema(macd, 9)
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    elif config.get("warmup_enabled", True):
        # No artificial NaN masking — value available from bar 1 (warm history makes it exact)
        df["EMA_Fast"] = close.ewm(span=fast, adjust=False).mean()
        df["EMA_Slow"] = close.ewm(span=slow, adjust=False).mean()

    df["EMA_Fast_Angle"] = fast_ema_angle(df["EMA_Fast"])
    df["EMA_Slow_Angle"] = fast_ema_angle(df["EMA_Slow"])
    atr = df["ATR"].replace(0, np.nan)
    df["EMA_Fast_Angle_Norm"] = np.degrees(np.arctan(ols_slope(df["EMA_Fast"]) / atr))
    df["EMA_Slow_Angle_Norm"] = np.degrees(np.arctan(ols_slope(df["EMA_Slow"]) / atr))

    # Aliases so "Mean Reversion with Bollinger Bands" (looks for Bollinger_* columns) can fire
    bp, bs = int(config.get("bb_period", 20)), float(config.get("bb_std", 2.0))
    ddof = 0 if config.get("tv_exact_indicators", True) else 1
    m2, s2 = close.rolling(bp).mean(), close.rolling(bp).std(ddof=ddof)
    df["Bollinger_Middle"], df["Bollinger_Upper"], df["Bollinger_Lower"] = m2, m2 + bs * s2, m2 - bs * s2

    df["VWAP_Session"] = session_vwap(df)
    df["CHOP"] = choppiness(df, int(config.get("flt_chop_period", FILTER_DEFAULTS["chop_period"])))
    return df


# ════════════════════════════════════════════════════════════════════════════
# DATA: WARM-UP PERIODS, DHAN DATA SOURCE, SECURITY-ID RESOLUTION
# ════════════════════════════════════════════════════════════════════════════
PERIOD_DAYS = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
               "1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
_PERIOD_ORDER = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
WARM_EXTRA_DAYS = {"1m": 3, "5m": 10, "15m": 25, "30m": 25, "1h": 120, "1d": 500, "1wk": 1500}
YF_MAX_PERIOD = {"1m": "5d", "5m": "1mo", "15m": "1mo", "30m": "1mo", "1h": "1y", "1d": "10y", "1wk": "10y"}
INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
LIVE_BASE_PERIOD = {"1m": "1d", "5m": "1d", "15m": "1d", "30m": "1d", "1h": "5d"}


def warmup_period(interval, requested):
    need = PERIOD_DAYS.get(requested, 30) + WARM_EXTRA_DAYS.get(interval, 30)
    cand = next((p for p in _PERIOD_ORDER if PERIOD_DAYS[p] >= need), "10y")
    cap = YF_MAX_PERIOD.get(interval, "10y")
    return cap if PERIOD_DAYS[cand] > PERIOD_DAYS[cap] else cand


DHAN_INDEX = {  # asset -> (security_id, segment, instrument, F&O root, F&O exchange)
    "NIFTY 50": ("13", "IDX_I", "INDEX", "NIFTY", "NSE"),
    "BANKNIFTY": ("25", "IDX_I", "INDEX", "BANKNIFTY", "NSE"),
    "FINNIFTY": ("27", "IDX_I", "INDEX", "FINNIFTY", "NSE"),
    "MIDCPNIFTY": ("442", "IDX_I", "INDEX", "MIDCPNIFTY", "NSE"),
    "SENSEX": ("51", "IDX_I", "INDEX", "SENSEX", "BSE"),
}
DHAN_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_MASTER_COLS = {"SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_INSTRUMENT_NAME",
                "SEM_TRADING_SYMBOL", "SEM_EXPIRY_DATE", "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE",
                "SEM_SERIES", "SM_SYMBOL_NAME", "SEM_LOT_UNITS"}


@st.cache_data(ttl=6 * 3600, show_spinner="Loading Dhan instrument master…")
def load_dhan_master():
    m = pd.read_csv(DHAN_MASTER_URL, usecols=lambda c: c in _MASTER_COLS, dtype=str, low_memory=False)
    m["_root"] = m["SEM_TRADING_SYMBOL"].fillna("").str.split("-").str[0].str.upper()
    m["_expiry"] = pd.to_datetime(m["SEM_EXPIRY_DATE"], errors="coerce")
    m["_strike"] = pd.to_numeric(m["SEM_STRIKE_PRICE"], errors="coerce")
    return m


def _stock_symbol(asset, custom_ticker):
    if asset != "Custom Ticker" or not custom_ticker:
        return None, None
    t = custom_ticker.strip().upper()
    if t.endswith(".NS"):
        return t[:-3], "NSE"
    if t.endswith(".BO"):
        return t[:-3], "BSE"
    return t, None


def resolve_dhan_instrument(cfg, asset, custom_ticker):
    """Returns dict(security_id, segment, instrument, label) or raises."""
    manual = str(cfg.get("dhan_data_manual_id", "") or "").strip()
    if manual:
        return {"security_id": manual, "segment": cfg.get("dhan_data_manual_segment", "NSE_EQ"),
                "instrument": cfg.get("dhan_data_manual_instrument", "EQUITY"), "label": f"manual {manual}"}
    want_fut = cfg.get("dhan_data_instrument", "").startswith("Futures")
    sym, exch = _stock_symbol(asset, custom_ticker)
    exch = exch or cfg.get("dhan_data_exchange", "NSE")

    if asset in DHAN_INDEX and not want_fut:
        sid, seg, inst, root, _ = DHAN_INDEX[asset]
        return {"security_id": sid, "segment": seg, "instrument": inst, "label": f"{asset} index ({sid})"}

    m = load_dhan_master()
    if want_fut:
        if asset in DHAN_INDEX:
            root, fexch, inst = DHAN_INDEX[asset][3], DHAN_INDEX[asset][4], "FUTIDX"
        elif sym:
            root, fexch, inst = sym, "NSE", "FUTSTK"
        else:
            raise ValueError(f"No Dhan futures mapping for {asset}")
        today = pd.Timestamp(datetime.now(IST).date())
        f = m[(m["SEM_EXM_EXCH_ID"] == fexch) & (m["SEM_INSTRUMENT_NAME"] == inst)
              & (m["_root"] == root) & (m["_expiry"] >= today)].sort_values("_expiry")
        if f.empty:
            raise ValueError(f"No live {inst} contract found for {root}")
        r = f.iloc[0]
        return {"security_id": str(r["SEM_SMST_SECURITY_ID"]), "segment": f"{fexch}_FNO",
                "instrument": inst, "label": f"{r['SEM_TRADING_SYMBOL']} ({r['SEM_SMST_SECURITY_ID']})"}

    if sym:
        e = m[(m["SEM_EXM_EXCH_ID"] == exch) & (m["SEM_SEGMENT"] == "E")
              & (m["SEM_TRADING_SYMBOL"].str.upper() == sym)]
        if e.empty:
            raise ValueError(f"{sym} not found on {exch} in Dhan master")
        eq = e[e["SEM_SERIES"].isin(["EQ", "A", "B"])]
        r = (eq if not eq.empty else e).iloc[0]
        return {"security_id": str(r["SEM_SMST_SECURITY_ID"]), "segment": f"{exch}_EQ",
                "instrument": "EQUITY", "label": f"{sym} {exch} ({r['SEM_SMST_SECURITY_ID']})"}
    raise ValueError(f"'{asset}' is not an Indian instrument — using yfinance")


def resolve_atm_options(config, spot):
    """Nearest-expiry ATM CE/PE security IDs for the selected underlying."""
    asset = config.get("asset", "NIFTY 50")
    m = load_dhan_master()
    if asset in DHAN_INDEX:
        root, exch, inst = DHAN_INDEX[asset][3], DHAN_INDEX[asset][4], "OPTIDX"
    else:
        sym, _ = _stock_symbol(asset, config.get("custom_ticker"))
        if not sym:
            raise ValueError("ATM resolution needs an Indian index or NSE stock")
        root, exch, inst = sym, "NSE", "OPTSTK"
    today = pd.Timestamp(datetime.now(IST).date())
    o = m[(m["SEM_EXM_EXCH_ID"] == exch) & (m["SEM_INSTRUMENT_NAME"] == inst)
          & (m["_root"] == root) & (m["_expiry"] >= today)]
    if o.empty:
        raise ValueError(f"No option contracts for {root}")
    o = o[o["_expiry"] == o["_expiry"].min()]
    strikes = np.sort(o["_strike"].dropna().unique())
    atm = float(strikes[np.abs(strikes - float(spot)).argmin()])
    ce = o[(o["_strike"] == atm) & (o["SEM_OPTION_TYPE"] == "CE")]
    pe = o[(o["_strike"] == atm) & (o["SEM_OPTION_TYPE"] == "PE")]
    if ce.empty or pe.empty:
        raise ValueError(f"ATM {atm} CE/PE missing")
    return {"ce": str(ce.iloc[0]["SEM_SMST_SECURITY_ID"]), "pe": str(pe.iloc[0]["SEM_SMST_SECURITY_ID"]),
            "strike": atm, "expiry": o["_expiry"].min().date(),
            "ce_symbol": ce.iloc[0]["SEM_TRADING_SYMBOL"], "pe_symbol": pe.iloc[0]["SEM_TRADING_SYMBOL"]}


def _dhan_post(path, body, cid, token):
    r = requests.post(f"https://api.dhan.co/v2{path}", json=body, timeout=20,
                      headers={"access-token": token, "client-id": str(cid),
                               "Content-Type": "application/json", "Accept": "application/json"})
    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"Dhan HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code != 200 or "open" not in j:
        raise RuntimeError(f"Dhan {r.status_code}: {str(j)[:200]}")
    return j


def _dhan_json_to_df(j, daily=False):
    ts = pd.to_datetime(pd.Series(j.get("timestamp", []), dtype="float64"), unit="s", utc=True).dt.tz_convert(IST)
    if daily:
        ts = ts.dt.normalize()
    n = len(ts)
    vol = j.get("volume") or [0] * n
    return pd.DataFrame({"Datetime": ts, "Open": j["open"], "High": j["high"], "Low": j["low"],
                         "Close": j["close"], "Volume": vol[:n]}).astype(
        {"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})


def _resample(df, rule, offset=None, label="left"):
    x = df.copy()
    x["Datetime"] = x["Datetime"].dt.tz_localize(None)
    kw = {"label": label, "closed": "left"}
    if offset:
        kw["offset"] = offset
    out = x.set_index("Datetime").resample(rule, **kw).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    out = out.reset_index()
    out["Datetime"] = out["Datetime"].dt.tz_localize(IST)
    return out


@st.cache_data(ttl=2, show_spinner=False)
def _dhan_ohlc_cached(cid, token, sec_id, seg, inst, interval, cal_days):
    now = datetime.now(IST)
    if interval in ("1d", "1wk"):
        body = {"securityId": str(sec_id), "exchangeSegment": seg, "instrument": inst, "expiryCode": 0,
                "oi": False, "fromDate": (now - timedelta(days=cal_days)).strftime("%Y-%m-%d"),
                "toDate": (now + timedelta(days=1)).strftime("%Y-%m-%d")}
        df = _dhan_json_to_df(_dhan_post("/charts/historical", body, cid, token), daily=True)
        if interval == "1wk":
            df = _resample(df, "W-MON")
        return df
    dhan_iv = {"1m": "1", "5m": "5", "15m": "15", "30m": "15", "1h": "60"}[interval]
    frames, end = [], now + timedelta(minutes=1)
    start_all = now - timedelta(days=cal_days)
    while end > start_all:                      # Dhan allows ~90 days per intraday call
        start = max(start_all, end - timedelta(days=85))
        body = {"securityId": str(sec_id), "exchangeSegment": seg, "instrument": inst,
                "interval": dhan_iv, "oi": False,
                "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": end.strftime("%Y-%m-%d %H:%M:%S")}
        frames.append(_dhan_json_to_df(_dhan_post("/charts/intraday", body, cid, token)))
        end = start
    df = pd.concat(frames).drop_duplicates("Datetime").sort_values("Datetime").reset_index(drop=True)
    if interval == "30m":
        df = _resample(df, "30min", offset="15min")      # 09:15, 09:45 … like TradingView NSE
    return df


def fetch_dhan_ohlc(cfg, asset, custom_ticker, interval, period_code):
    cid = cfg.get("dhan_data_client_id") or cfg.get("dhan_client_id")
    token = cfg.get("dhan_data_token") or cfg.get("dhan_access_token")
    if not cid or not token:
        raise ValueError("Dhan Client ID / Access Token missing for data")
    info = resolve_dhan_instrument(cfg, asset, custom_ticker)
    days = PERIOD_DAYS.get(period_code, 30)
    cal_days = int(days * 1.6) + 4 if days <= 5 else int(days * 1.05) + 3
    df = _dhan_ohlc_cached(cid, token, info["security_id"], info["segment"], info["instrument"],
                           interval, cal_days)
    if df is None or df.empty:
        raise RuntimeError("Dhan returned no candles")
    _ss()["plus_data_status"] = f"✅ Dhan data: {info['label']} · {len(df)} candles"
    return df.copy()


def _yf_robust(asset, interval, period, custom_ticker):
    """Fallback when the original fetch returns None (e.g. tz-naive daily index)."""
    import yfinance as yf
    t = custom_ticker if asset == "Custom Ticker" and custom_ticker else base.ASSET_MAPPING.get(asset, asset)
    df = yf.download(t, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
    df = df.reset_index().rename(columns={df.index.name or "index": "Datetime", "Date": "Datetime"})
    dt = pd.to_datetime(df["Datetime"])
    df["Datetime"] = dt.dt.tz_localize(IST) if dt.dt.tz is None else dt.dt.tz_convert(IST)
    return df


def fetch_data_plus(ticker_symbol, interval, period, is_live_trading=False, custom_ticker=None, config=None):
    cfg = config or _cfg()
    req = LIVE_BASE_PERIOD.get(interval, "1mo") if is_live_trading else period
    eff = warmup_period(interval, req) if cfg.get("warmup_enabled", True) else req

    if cfg.get("use_dhan_data", False):
        try:
            return fetch_dhan_ohlc(cfg, ticker_symbol, custom_ticker, interval, eff)
        except Exception as e:
            _ss()["plus_data_status"] = f"⚠️ Dhan data failed → yfinance used: {e}"

    df = _orig_fetch(ticker_symbol, interval, eff, False, custom_ticker)
    if df is None:
        try:
            df = _yf_robust(ticker_symbol, interval, eff, custom_ticker)
        except Exception:
            df = None
    if df is not None and not cfg.get("use_dhan_data", False):
        _ss()["plus_data_status"] = f"yfinance · period {eff} · {len(df)} candles"
    return df


# ════════════════════════════════════════════════════════════════════════════
# MARKET STRUCTURE (BOS / CHoCH / swings) — causal, no look-ahead
# ════════════════════════════════════════════════════════════════════════════
def ensure_structure(df, L):
    key = f"SMC_T_{L}"
    if key in df.columns:
        return
    H, Lo, C = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    n = len(df)
    T, E = np.zeros(n), np.zeros(n)
    SH, SL = np.full(n, np.nan), np.full(n, np.nan)
    cur_sh = cur_sl = np.nan
    sh_broken = sl_broken = True
    t = 0
    for j in range(n):
        p = j - L                                 # pivot at p is confirmed L bars later
        if p - L >= 0:
            if H[p] >= H[p - L:p + L + 1].max() and H[p] > H[p - L:p].max():
                cur_sh, sh_broken = H[p], False
            if Lo[p] <= Lo[p - L:p + L + 1].min() and Lo[p] < Lo[p - L:p].min():
                cur_sl, sl_broken = Lo[p], False
        ev = 0
        if not sh_broken and not np.isnan(cur_sh) and C[j] > cur_sh:
            ev, t, sh_broken = (2 if t == -1 else 1), 1, True        # 1=BOS up, 2=CHoCH up
        elif not sl_broken and not np.isnan(cur_sl) and C[j] < cur_sl:
            ev, t, sl_broken = (-2 if t == 1 else -1), -1, True      # -1=BOS dn, -2=CHoCH dn
        T[j], E[j], SH[j], SL[j] = t, ev, cur_sh, cur_sl
    df[f"SMC_T_{L}"], df[f"SMC_E_{L}"], df[f"SMC_SH_{L}"], df[f"SMC_SL_{L}"] = T, E, SH, SL


def ensure_prev_day_levels(df):
    if "PDH" in df.columns:
        return
    d = df["Datetime"].dt.date
    daily = df.groupby(d).agg(h=("High", "max"), l=("Low", "min"))
    prev = daily.shift(1)
    df["PDH"] = d.map(prev["h"]).astype(float).values
    df["PDL"] = d.map(prev["l"]).astype(float).values


def ensure_htf(df, minutes, length):
    kc, ke = f"HTF_C_{minutes}_{length}", f"HTF_E_{minutes}_{length}"
    if kc in df.columns:
        return kc, ke
    wall = df["Datetime"].dt.tz_localize(None)
    s = pd.Series(df["Close"].to_numpy(float), index=wall)
    rule, off = f"{minutes}min", pd.Timedelta("15min")
    htf = s.resample(rule, offset="15min", label="left", closed="left").last().dropna()
    ema = tv_ema(htf, length)
    lab = (wall - off).dt.floor(rule) + off
    df[kc] = htf.shift(1).reindex(lab.values).to_numpy()   # last COMPLETED htf bar only
    df[ke] = ema.shift(1).reindex(lab.values).to_numpy()
    df[ke + "_prev"] = ema.shift(2).reindex(lab.values).to_numpy()
    return kc, ke


# ════════════════════════════════════════════════════════════════════════════
# ENTRY FILTERS — each returns (passed, reason)
# ════════════════════════════════════════════════════════════════════════════
def _p(cfg, k):
    return cfg.get(f"flt_{k}", FILTER_DEFAULTS[k])


def _v(df, col, i):
    v = df[col].iat[i] if col in df.columns else np.nan
    return float(v) if pd.notna(v) else np.nan


def f_adx(df, i, long, c):
    v, lo, hi = _v(df, "ADX", i), _p(c, "adx_min"), _p(c, "adx_max")
    return (lo <= v <= hi), f"ADX {v:.1f} outside [{lo}, {hi}]"


def f_rsi(df, i, long, c):
    per = int(_p(c, "rsi_period"))
    col = "RSI" if per == 14 else f"_PLUS_RSI_{per}"
    if col not in df.columns:
        df[col] = _patched_rsi(df["Close"], per)
    v = _v(df, col, i)
    lo, hi = (_p(c, "rsi_long_min"), _p(c, "rsi_long_max")) if long else (_p(c, "rsi_short_min"), _p(c, "rsi_short_max"))
    return (lo <= v <= hi), f"RSI {v:.1f} outside [{lo}, {hi}]"


def f_bb(df, i, long, c):
    p, k = int(_p(c, "bb_period")), float(_p(c, "bb_std"))
    key = f"_PLUS_BB_{p}_{k}"
    if key + "_M" not in df.columns:
        ddof = 0 if c.get("tv_exact_indicators", True) else 1
        m, s = df["Close"].rolling(p).mean(), df["Close"].rolling(p).std(ddof=ddof)
        df[key + "_M"], df[key + "_U"], df[key + "_L"] = m, m + k * s, m - k * s
    m, u, l, cl = _v(df, key + "_M", i), _v(df, key + "_U", i), _v(df, key + "_L", i), _v(df, "Close", i)
    if np.isnan(m):
        return False, "BB not ready"
    bw = (u - l) / m * 100
    if not (_p(c, "bb_bw_min") <= bw <= _p(c, "bb_bw_max")):
        return False, f"bandwidth {bw:.2f}% outside range"
    mode = c.get("flt_bb_mode", "Above middle (LONG) / below middle (SHORT)")
    if mode.startswith("Above middle"):
        return (cl > m if long else cl < m), "price on wrong side of BB middle"
    if mode.startswith("Inside"):
        return (l <= cl <= u), "price outside bands (chasing)"
    if mode.startswith("Breakout"):
        return (cl > u if long else cl < l), "no band breakout"
    return True, ""


def _ma_side(df, i, long, col, slope, name):
    v, pv, cl = _v(df, col, i), _v(df, col, i - 1), _v(df, "Close", i)
    if np.isnan(v):
        return False, f"{name} not ready"
    ok = cl > v if long else cl < v
    if ok and slope:
        ok = v > pv if long else v < pv
    return ok, f"price/slope wrong side of {name} {v:.2f}"


def f_sma(df, i, long, c):
    p = int(_p(c, "sma_period"))
    col = f"_PLUS_SMA_{p}"
    if col not in df.columns:
        df[col] = df["Close"].rolling(p).mean()
    return _ma_side(df, i, long, col, c.get("flt_sma_slope", False), f"SMA{p}")


def f_ema(df, i, long, c):
    p = int(_p(c, "ema_period"))
    col = f"_PLUS_EMA_{p}"
    if col not in df.columns:
        df[col] = tv_ema(df["Close"], p)
    return _ma_side(df, i, long, col, c.get("flt_ema_slope", False), f"EMA{p}")


def f_macd(df, i, long, c):
    m, s, h = _v(df, "MACD", i), _v(df, "MACD_Signal", i), _v(df, "MACD_Hist", i)
    mode = c.get("flt_macd_mode", "MACD vs Signal")
    if mode == "MACD vs Signal":
        return (m > s if long else m < s), f"MACD {m:.2f} vs signal {s:.2f}"
    if mode == "Histogram sign":
        return (h > 0 if long else h < 0), f"histogram {h:.2f}"
    if mode == "Histogram rising/falling":
        ph = _v(df, "MACD_Hist", i - 1)
        return (h > ph if long else h < ph), "histogram not expanding"
    return (m > 0 if long else m < 0), f"MACD {m:.2f} wrong side of zero"


def f_vol(df, i, long, c):
    if not has_volume(df):
        return True, ""                                    # indices: ignored
    v, pv, nx = _v(df, "Volume", i), _v(df, "Volume", i - 1), _p(c, "vol_nx")
    return (v >= nx * pv), f"volume {v:.0f} < {nx}× prev {pv:.0f}"


def f_avgvol(df, i, long, c):
    if not has_volume(df):
        return True, ""
    p, nx = int(_p(c, "avgvol_period")), _p(c, "avgvol_nx")
    avg = float(df["Volume"].iloc[max(0, i - p):i].mean())       # average of PRIOR candles
    v = _v(df, "Volume", i)
    return (v >= nx * avg), f"volume {v:.0f} < {nx}× avg {avg:.0f}"


def volume_profile(df, i, lookback, bins, va_pct):
    w = df.iloc[max(0, i - lookback + 1):i + 1]
    lo, hi = float(w["Low"].min()), float(w["High"].max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    tp = ((w["High"] + w["Low"] + w["Close"]) / 3).to_numpy()
    b = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    hist = np.bincount(b, weights=w["Volume"].fillna(0).to_numpy(), minlength=bins)
    poc_b = int(hist.argmax())
    target, acc, lo_b, hi_b = hist.sum() * va_pct / 100, hist[poc_b], poc_b, poc_b
    while acc < target and (lo_b > 0 or hi_b < bins - 1):
        up = hist[hi_b + 1] if hi_b < bins - 1 else -1
        dn = hist[lo_b - 1] if lo_b > 0 else -1
        if up >= dn:
            hi_b += 1
            acc += up
        else:
            lo_b -= 1
            acc += dn
    return {"poc": (edges[poc_b] + edges[poc_b + 1]) / 2, "vah": edges[hi_b + 1], "val": edges[lo_b]}


def f_vp(df, i, long, c):
    if not has_volume(df):
        return True, ""
    vp = volume_profile(df, i, int(_p(c, "vp_lookback")), int(_p(c, "vp_bins")), float(_p(c, "vp_value_area")))
    if vp is None:
        return False, "profile n/a"
    cl = _v(df, "Close", i)
    if c.get("flt_vp_mode", "Price vs POC").startswith("Price vs POC"):
        return (cl > vp["poc"] if long else cl < vp["poc"]), f"price vs POC {vp['poc']:.2f}"
    return (cl > vp["vah"] if long else cl < vp["val"]), f"inside value area {vp['val']:.2f}-{vp['vah']:.2f}"


def _events_in(df, i, L, K, codes):
    E = df[f"SMC_E_{L}"].to_numpy()[max(0, i - K + 1):i + 1]
    return bool(np.isin(E, codes).any())


def f_bos(df, i, long, c):
    L, K = int(_p(c, "swing_len")), int(_p(c, "bos_lookback"))
    ensure_structure(df, L)
    codes = ([1, 2] if c.get("flt_bos_incl_choch", True) else [1]) if long else \
            ([-1, -2] if c.get("flt_bos_incl_choch", True) else [-1])
    trend_ok = df[f"SMC_T_{L}"].iat[i] == (1 if long else -1)
    return (trend_ok and _events_in(df, i, L, K, codes)), f"no {'bullish' if long else 'bearish'} BOS in last {K}"


def f_choch(df, i, long, c):
    L, K = int(_p(c, "swing_len")), int(_p(c, "choch_lookback"))
    ensure_structure(df, L)
    return _events_in(df, i, L, K, [2] if long else [-2]), f"no {'bullish' if long else 'bearish'} CHoCH in last {K}"


def f_fvg(df, i, long, c):
    K = int(_p(c, "fvg_lookback"))
    atr = _v(df, "ATR", i)
    min_sz = (_p(c, "fvg_min_size_atr") * atr) if not np.isnan(atr) else 0.0
    H, Lo, C = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    retest = c.get("flt_fvg_mode", "Unfilled FVG in direction").startswith("Price retesting")
    for j in range(max(2, i - K + 1), i + 1):
        if long:
            bot, top = H[j - 2], Lo[j]
            if top - bot >= max(min_sz, 1e-12):
                unfilled = (Lo[j + 1:i + 1].min() if j < i else np.inf) > bot
                if unfilled and (not retest or (j < i and Lo[i] <= top and C[i] >= bot)):
                    return True, ""
        else:
            top, bot = Lo[j - 2], H[j]
            if top - bot >= max(min_sz, 1e-12):
                unfilled = (H[j + 1:i + 1].max() if j < i else -np.inf) < top
                if unfilled and (not retest or (j < i and H[i] >= bot and C[i] <= top)):
                    return True, ""
    return False, f"no qualifying {'bullish' if long else 'bearish'} FVG"


def _swing_range(df, i, c):
    L = int(_p(c, "swing_len"))
    ensure_structure(df, L)
    return _v(df, f"SMC_SH_{L}", i), _v(df, f"SMC_SL_{L}", i), df[f"SMC_T_{L}"].iat[i]


def f_smc(df, i, long, c):
    sh, sl, t = _swing_range(df, i, c)
    if c.get("flt_smc_trend", True) and t != (1 if long else -1):
        return False, "structure trend not aligned"
    if c.get("flt_smc_pd", False):
        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            return False, "no swing range"
        eq, cl = (sh + sl) / 2, _v(df, "Close", i)
        if (long and cl > eq) or (not long and cl < eq):
            return False, "price in premium (LONG) / discount (SHORT)"
    return True, ""


def _in_killzone(ts, zones):
    t = ts.astimezone(IST).time() if ts.tzinfo else ts.time()
    for z in str(zones).split(","):
        try:
            a, b = [datetime.strptime(x.strip(), "%H:%M").time() for x in z.split("-")]
            if a <= t <= b:
                return True
        except Exception:
            continue
    return False


def f_ict(df, i, long, c):
    if c.get("flt_ict_kz", True) and not _in_killzone(df["Datetime"].iat[i], _p(c, "ict_killzones")):
        return False, "outside killzone"
    if c.get("flt_ict_ote", False):
        sh, sl, _ = _swing_range(df, i, c)
        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            return False, "no swing range for OTE"
        cl, rng = _v(df, "Close", i), sh - sl
        r = (sh - cl) / rng if long else (cl - sl) / rng
        lo, hi = _p(c, "ict_ote_low"), _p(c, "ict_ote_high")
        if not (lo <= r <= hi):
            return False, f"retracement {r:.2f} outside OTE {lo}-{hi}"
    if c.get("flt_ict_pd_bias", False):
        ensure_prev_day_levels(df)
        mid = (_v(df, "PDH", i) + _v(df, "PDL", i)) / 2
        cl = _v(df, "Close", i)
        if not np.isnan(mid) and ((long and cl < mid) or (not long and cl > mid)):
            return False, "against previous-day bias"
    return True, ""


def f_angle(df, i, long, c):
    mode = c.get("flt_angle_mode", "ATR-normalized (recommended)")
    if mode.startswith("Raw"):
        a = abs(_v(df, "EMA_Fast_Angle", i))
    elif mode.startswith("Crossover"):
        a = abs(_v(df, "EMA_Fast_Angle_Norm", i) - _v(df, "EMA_Slow_Angle_Norm", i))
    else:
        a = abs(_v(df, "EMA_Fast_Angle_Norm", i))
    lo, hi = _p(c, "angle_min"), _p(c, "angle_max")
    return (lo <= a <= hi), f"angle {a:.1f}° outside [{lo}, {hi}]"


def f_sweep(df, i, long, c):
    K = int(_p(c, "sweep_lookback"))
    mode = c.get("flt_sweep_mode", "Swing high/low")
    L = int(_p(c, "swing_len"))
    ensure_structure(df, L)
    ensure_prev_day_levels(df)
    Hh, Lo, C = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    SH, SL = df[f"SMC_SH_{L}"].to_numpy(), df[f"SMC_SL_{L}"].to_numpy()
    PH, PL = df["PDH"].to_numpy(), df["PDL"].to_numpy()
    for j in range(max(1, i - K + 1), i + 1):
        lv = []
        if mode in ("Swing high/low", "Either"):
            lv.append(SL[j - 1] if long else SH[j - 1])
        if mode in ("Previous day high/low", "Either"):
            lv.append(PL[j] if long else PH[j])
        for x in lv:
            if np.isnan(x):
                continue
            if long and Lo[j] < x and C[j] > x:
                return True, ""
            if not long and Hh[j] > x and C[j] < x:
                return True, ""
    return False, f"no liquidity sweep in last {K}"


def f_atr(df, i, long, c):
    per = int(_p(c, "atr_period"))
    col = "ATR" if per == 14 else f"_PLUS_ATR_{per}"
    if col not in df.columns:
        df[col] = _patched_atr(df, per)
    a = _v(df, col, i)
    if c.get("flt_atr_mode", "Points").startswith("%"):
        a = a / _v(df, "Close", i) * 100
        lo, hi = _p(c, "atr_pct_min"), _p(c, "atr_pct_max")
    else:
        lo, hi = _p(c, "atr_min"), _p(c, "atr_max")
    return (lo <= a <= hi), f"ATR {a:.2f} outside [{lo}, {hi}]"


def f_regime(df, i, long, c):
    ch = _v(df, "CHOP", i)
    if np.isnan(ch) or ch > _p(c, "chop_max"):
        return False, f"choppy market (CHOP {ch:.1f})"
    if c.get("flt_regime_use_adx", True):
        a = _v(df, "ADX", i)
        if np.isnan(a) or a < _p(c, "regime_adx_min"):
            return False, f"weak trend (ADX {a:.1f})"
    return True, ""


def f_vwap(df, i, long, c):
    if not has_volume(df):
        return True, ""                                    # indices have no volume → ignored
    vw, cl = _v(df, "VWAP_Session", i), _v(df, "Close", i)
    if np.isnan(vw):
        return False, "VWAP n/a"
    if not (cl > vw if long else cl < vw):
        return False, f"price wrong side of VWAP {vw:.2f}"
    dist = abs(cl - vw) / vw * 100
    return dist <= _p(c, "vwap_max_dist_pct"), f"{dist:.2f}% from VWAP (too extended)"


def f_htf(df, i, long, c):
    ltf = INTERVAL_MIN.get(base.INTERVAL_MAPPING.get(c.get("interval", ""), ""), None)
    m, n = int(_p(c, "htf_minutes")), int(_p(c, "htf_ema"))
    if ltf is None or ltf >= m:
        return True, ""
    kc, ke = ensure_htf(df, m, n)
    hc, he, hp = _v(df, kc, i), _v(df, ke, i), _v(df, ke + "_prev", i)
    if np.isnan(hc) or np.isnan(he):
        return False, "HTF not ready"
    ok = hc > he if long else hc < he
    if ok and c.get("flt_htf_slope", True) and not np.isnan(hp):
        ok = he > hp if long else he < hp
    return ok, f"{m}m trend against trade"


FILTERS = [  # key, label, function
    ("adx", "ADX range", f_adx), ("rsi", "RSI range", f_rsi), ("bb", "Bollinger Bands", f_bb),
    ("sma", "SMA", f_sma), ("ema", "EMA trend", f_ema), ("macd", "MACD", f_macd),
    ("vol", "Volume Nx", f_vol), ("avgvol", "Avg Volume Nx", f_avgvol),
    ("vp", "Volume Profile", f_vp), ("bos", "Break of Structure", f_bos),
    ("choch", "Change of Character", f_choch), ("fvg", "Fair Value Gap", f_fvg),
    ("smc", "SMC", f_smc), ("ict", "ICT", f_ict), ("angle", "Crossover angle", f_angle),
    ("sweep", "Liquidity sweep", f_sweep), ("atr", "ATR", f_atr),
    ("regime", "Market regime", f_regime), ("vwap", "VWAP", f_vwap), ("htf", "Higher-TF trend", f_htf),
]


def apply_entry_filters(df, i, sig, config):
    long = sig in ("BUY", "LONG")
    fails = []
    for key, label, fn in FILTERS:
        if not config.get(f"flt_{key}", False):
            continue
        try:
            ok, why = fn(df, i, long, config)
        except Exception as e:
            ok, why = False, f"error: {e}"
        if not ok:
            fails.append(f"{label}: {why}")
            if _IN_BACKTEST:
                _FILTER_STATS[label] = _FILTER_STATS.get(label, 0) + 1
    return (len(fails) == 0), fails


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL PIPELINE (wraps every strategy without editing it)
# ════════════════════════════════════════════════════════════════════════════
_NO_FRESH = {"Simple Buy", "Simple Sell", "Price Crosses Threshold"}


def _last_bar_forming(df, config):
    iv = base.INTERVAL_MAPPING.get(config.get("interval", "1 minute"), "1m")
    last = df["Datetime"].iloc[-1]
    now = datetime.now(IST)
    mins = INTERVAL_MIN.get(iv)
    if mins:
        return last + timedelta(minutes=mins) > now
    return last.date() == now.date() and now.time() < datetime.strptime("15:30", "%H:%M").time()


def _is_live_call(df, idx):
    return (not _IN_BACKTEST) and idx >= len(df) - 1


def _live_note(msg):
    ss = _ss()
    if ss.get("trading_active") and ss.get("_plus_last_note") != msg:
        ss["_plus_last_note"] = msg
        try:
            base.add_log(msg)
        except Exception:
            pass


def _sustained(df, j, i, sig, name, c):
    long = sig in ("BUY", "LONG")
    mode = c.get("flt_sustain_mode", "Close holds beyond Fast EMA")
    sig_row = df.iloc[j]
    for k in range(j + 1, i + 1):
        r = df.iloc[k]
        if mode.startswith("Close holds beyond Fast EMA") and pd.notna(r.get("EMA_Fast")):
            ok = (r["Close"] > r["EMA_Fast"] and r["EMA_Fast"] > r["EMA_Slow"]) if long else \
                 (r["Close"] < r["EMA_Fast"] and r["EMA_Fast"] < r["EMA_Slow"])
        elif mode.startswith("Close beyond signal-candle High/Low"):
            ok = r["Close"] > sig_row["High"] if long else r["Close"] < sig_row["Low"]
        else:
            ok = r["Close"] > sig_row["Close"] if long else r["Close"] < sig_row["Close"]
        if not ok:
            return False
    return True


def _ema_pullback_signal(raw, df, i, c):
    N = int(_p(c, "pullback_window"))
    j, sig = None, None
    for k in range(i - 1, max(1, i - N) - 1, -1):          # crossover must be BEFORE this bar
        s, _ = raw(df, k, c, None)
        if s:
            j, sig = k, s
            break
    if sig is None:
        return None, None
    long = sig == "BUY"
    F, S = df["EMA_Fast"].to_numpy(float), df["EMA_Slow"].to_numpy(float)
    Lo, Hh = df["Low"].to_numpy(float), df["High"].to_numpy(float)
    C, O = df["Close"].to_numpy(float), df["Open"].to_numpy(float)
    if not all((F[k] > S[k]) if long else (F[k] < S[k]) for k in range(j, i + 1)):
        return None, None                                     # trend flipped back → void
    atr = _v(df, "ATR", i)
    tol = (_p(c, "pullback_tol_atr") * atr) if not np.isnan(atr) else 0.0
    touch = (lambda k: Lo[k] <= F[k] + tol) if long else (lambda k: Hh[k] >= F[k] - tol)
    if any(touch(k) for k in range(j + 1, i)):
        return None, None                                     # only the FIRST pullback
    ok = touch(i) and ((C[i] > F[i] and C[i] > O[i]) if long else (C[i] < F[i] and C[i] < O[i]))
    return (sig, C[i]) if ok else (None, None)


def _make_wrapped(name, raw):
    def wrapped(df, idx, config, current_position):
        config = config or {}
        i = idx
        if config.get("confirm_on_close", False) and _is_live_call(df, idx) and _last_bar_forming(df, config):
            i = idx - 1                                         # evaluate last CLOSED candle

        if current_position is not None:                        # exit-signal checks: unchanged logic
            return raw(df, i, config, current_position) if i >= 1 else (None, None)
        if idx < int(config.get("_bt_min_idx", 0) or 0) or i < 1:
            return None, None

        if name == "EMA Crossover" and config.get("flt_pullback", False):
            sig, price = _ema_pullback_signal(raw, df, i, config)
        else:
            n_sus = int(_p(config, "sustain_candles")) if config.get("flt_sustain", False) else 0
            j = i - n_sus
            if j < 1:
                return None, None
            sig, price = raw(df, j, config, None)
            if sig and config.get("fresh_signal_only", False) and name not in _NO_FRESH and j - 1 >= 1:
                psig, _ = raw(df, j - 1, config, None)
                if psig == sig:
                    sig = None                                  # condition was already true → stale
            if sig and n_sus > 0:
                if not _sustained(df, j, i, sig, name, config):
                    if _IN_BACKTEST:
                        _FILTER_STATS["Candle sustain"] = _FILTER_STATS.get("Candle sustain", 0) + 1
                    sig = None
                else:
                    price = float(df["Close"].iat[i])
        if not sig:
            return None, None

        ok, reasons = apply_entry_filters(df, i, sig, config)
        if not ok:
            if _is_live_call(df, idx):
                _live_note(f"🧰 {sig} signal blocked → " + " | ".join(reasons))
            return None, None

        if _is_live_call(df, idx) and name not in _NO_FRESH:   # one entry per signal candle
            ss = _ss()
            tag = (str(df["Datetime"].iat[i]), sig)
            if ss.get("_plus_used_signal") == tag:
                return None, None
            ss["_plus_used_signal"] = tag
        if i != idx:
            price = float(df["Close"].iat[idx])                 # fill at the live price
        return sig, price

    wrapped.__name__ = f"plus_{getattr(raw, '__name__', 'strategy')}"
    return wrapped


# ════════════════════════════════════════════════════════════════════════════
# BACKTEST WRAPPER (keeps original engine; only trims warm-up history)
# ════════════════════════════════════════════════════════════════════════════
def period_start_index(df, period_code):
    d = df["Datetime"]
    if period_code in ("1d", "5d"):
        n = 1 if period_code == "1d" else 5
        days = sorted(d.dt.date.unique())
        first = days[-n] if len(days) >= n else days[0]
        return int(np.argmax((d.dt.date >= first).to_numpy()))
    cutoff = d.iloc[-1] - timedelta(days=PERIOD_DAYS.get(period_code, 30))
    return int(np.argmax((d >= cutoff).to_numpy()))


def run_backtest_plus(df, config):
    global _IN_BACKTEST, _FILTER_STATS
    cfg = dict(config)
    if cfg.get("warmup_enabled", True):
        cfg["_bt_min_idx"] = period_start_index(df, base.PERIOD_MAPPING.get(cfg.get("period", "1 month"), "1mo"))
    _IN_BACKTEST, _FILTER_STATS = True, {}
    try:
        result = _orig_run_bt(df, cfg)
    finally:
        _IN_BACKTEST = False
    _ss()["plus_filter_stats"] = dict(_FILTER_STATS)
    _ss()["plus_bt_start"] = str(df["Datetime"].iat[cfg.get("_bt_min_idx", 0)]) if len(df) else ""
    return result


def render_backtest_ui_plus(config):
    _orig_render_bt(config)
    if "backtest_results" not in _ss():
        return
    with st.expander("🧰 PLUS: warm-up window & filter rejections (last backtest)"):
        if config.get("warmup_enabled", True):
            st.caption(f"Trades counted from {_ss().get('plus_bt_start', '?')} — earlier candles were used only to warm up indicators.")
        stats = _ss().get("plus_filter_stats") or {}
        if stats:
            st.dataframe(pd.DataFrame(sorted(stats.items(), key=lambda x: -x[1]),
                                      columns=["Filter", "Signals blocked"]), hide_index=True)
        else:
            st.caption("No signals were blocked by filters.")


# ════════════════════════════════════════════════════════════════════════════
# BROKER: optional automatic ATM CE/PE security IDs
# ════════════════════════════════════════════════════════════════════════════
def enter_broker_position_plus(self, signal, price, config, log_func):
    if config.get("auto_atm_options", False) and config.get("dhan_is_options", False):
        try:
            r = resolve_atm_options(config, price)
            for d in (config, self.config):
                d["dhan_ce_security_id"], d["dhan_pe_security_id"] = r["ce"], r["pe"]
            log_func(f"🏦 🔎 ATM {r['strike']:.0f} exp {r['expiry']}: CE {r['ce_symbol']} ({r['ce']}) | "
                     f"PE {r['pe_symbol']} ({r['pe']})")
        except Exception as e:
            log_func(f"🏦 ⚠️ ATM auto-resolve failed ({e}) — using manual CE/PE IDs")
    return _orig_enter_broker(self, signal, price, config, log_func)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR UI (appended below your existing settings)
# ════════════════════════════════════════════════════════════════════════════
def _n(cfg, k, label, **kw):
    d = FILTER_DEFAULTS[k]
    if isinstance(d, int):
        cfg[f"flt_{k}"] = int(st.number_input(label, value=d, step=1, key=f"plus_{k}", **kw))
    else:
        cfg[f"flt_{k}"] = float(st.number_input(label, value=float(d), key=f"plus_{k}", **kw))


def _cb(cfg, k, label, default=False, help=None):
    cfg[f"flt_{k}"] = st.checkbox(label, value=default, key=f"plus_flt_{k}", help=help)
    return cfg[f"flt_{k}"]


def _sel(cfg, k, label, opts):
    cfg[f"flt_{k}"] = st.selectbox(label, opts, key=f"plus_sel_{k}")


def render_plus_sidebar(cfg):
    sb = st.sidebar
    sb.markdown("---")
    sb.subheader("🚀 PLUS Upgrades")

    with sb.expander("⚡ Data & Indicator Engine", expanded=False):
        cfg["warmup_enabled"] = st.checkbox(
            "Pre-load warm-up history", value=True, key="plus_warmup",
            help="Loads extra past sessions so EMA/RSI/ADX are fully formed on the very first candle "
                 "(no 20-candle wait, gap-up/gap-down safe). Backtest P&L still counts only your selected period.")
        cfg["tv_exact_indicators"] = st.checkbox(
            "TradingView-exact formulas", value=True, key="plus_tv",
            help="EMA (SMA-seeded), RSI/ATR/ADX with Wilder RMA, Bollinger population stdev, session VWAP.")
        cfg["use_dhan_data"] = st.checkbox("Use Dhan data instead of yfinance", value=False, key="plus_dhan_data")
        if cfg["use_dhan_data"]:
            cfg["dhan_data_client_id"] = st.text_input("Dhan Client ID (data)", value=cfg.get("dhan_client_id", ""), key="plus_dd_cid")
            cfg["dhan_data_token"] = st.text_input("Dhan Access Token (data)", value=cfg.get("dhan_access_token", ""),
                                                   type="password", key="plus_dd_tok")
            cfg["dhan_data_instrument"] = st.selectbox("Instrument", ["Spot / Cash (Index or Stock)", "Futures (nearest expiry)"],
                                                       key="plus_dd_inst")
            cfg["dhan_data_exchange"] = st.selectbox("Stock exchange (if no .NS/.BO suffix)", ["NSE", "BSE"], key="plus_dd_ex")
            cfg["dhan_data_manual_id"] = st.text_input("Override Security ID (optional)", value="", key="plus_dd_mid")
            if cfg["dhan_data_manual_id"].strip():
                cfg["dhan_data_manual_segment"] = st.selectbox("Segment", ["IDX_I", "NSE_EQ", "BSE_EQ", "NSE_FNO", "BSE_FNO", "MCX_COMM"], key="plus_dd_seg")
                cfg["dhan_data_manual_instrument"] = st.selectbox("Instrument type", ["INDEX", "EQUITY", "FUTIDX", "FUTSTK", "OPTIDX", "OPTSTK", "FUTCOM"], key="plus_dd_it")
            if st.button("🔎 Resolve Security ID", key="plus_dd_resolve"):
                try:
                    st.success(resolve_dhan_instrument(cfg, cfg.get("asset"), cfg.get("custom_ticker"))["label"])
                except Exception as e:
                    st.error(str(e))
        status = _ss().get("plus_data_status")
        if status:
            st.caption(status)

    with sb.expander("🎯 Signal Quality", expanded=False):
        cfg["confirm_on_close"] = st.checkbox(
            "Confirm on candle close (no repaint) — recommended", value=False, key="plus_close",
            help="Live: signals are taken from the last COMPLETED candle, so an EMA cross that appears and "
                 "disappears inside a forming 1-min candle can't trigger a trade.")
        cfg["fresh_signal_only"] = st.checkbox(
            "Fresh signals only (all strategies)", value=False, key="plus_fresh",
            help="Enter only on the candle where the signal FIRST appears (not on every candle the condition stays true). "
                 "Simple Buy/Sell and Price Threshold are exempt by design.")
        if cfg.get("dhan_enabled") and cfg.get("dhan_is_options"):
            cfg["auto_atm_options"] = st.checkbox("Auto-pick ATM CE/PE IDs (nearest expiry)", value=False, key="plus_atm")

    with sb.expander("🧰 Additional Entry Filters (all OFF by default)", expanded=False):
        st.caption("Every enabled filter must pass on the signal candle. Volume-based filters are skipped automatically for indices.")
        if _cb(cfg, "adx", "ADX range"):
            _n(cfg, "adx_min", "ADX min"); _n(cfg, "adx_max", "ADX max")
        if _cb(cfg, "rsi", "RSI range"):
            _n(cfg, "rsi_period", "RSI period", min_value=2)
            _n(cfg, "rsi_long_min", "LONG: RSI min"); _n(cfg, "rsi_long_max", "LONG: RSI max")
            _n(cfg, "rsi_short_min", "SHORT: RSI min"); _n(cfg, "rsi_short_max", "SHORT: RSI max")
        if _cb(cfg, "bb", "Bollinger Bands"):
            _n(cfg, "bb_period", "BB period", min_value=2); _n(cfg, "bb_std", "BB std dev")
            _sel(cfg, "bb_mode", "BB rule", ["Above middle (LONG) / below middle (SHORT)", "Inside bands (no chasing)",
                                             "Breakout beyond band", "Bandwidth only"])
            _n(cfg, "bb_bw_min", "Bandwidth min %"); _n(cfg, "bb_bw_max", "Bandwidth max %")
        if _cb(cfg, "sma", "SMA filter (price above for LONG / below for SHORT)"):
            _n(cfg, "sma_period", "SMA period", min_value=1); _cb(cfg, "sma_slope", "SMA must slope with trade")
        if _cb(cfg, "ema", "EMA trend filter"):
            _n(cfg, "ema_period", "EMA period", min_value=1); _cb(cfg, "ema_slope", "EMA must slope with trade")
        if _cb(cfg, "macd", "MACD filter"):
            _sel(cfg, "macd_mode", "MACD rule", ["MACD vs Signal", "Histogram sign", "Histogram rising/falling", "MACD vs zero"])
        if _cb(cfg, "vol", "Volume ≥ N × previous candle"):
            _n(cfg, "vol_nx", "N (×)")
        if _cb(cfg, "avgvol", "Volume ≥ N × average volume"):
            _n(cfg, "avgvol_nx", "N (×)", ); _n(cfg, "avgvol_period", "Average period", min_value=1)
        if _cb(cfg, "vp", "Fixed-range Volume Profile"):
            _n(cfg, "vp_lookback", "Range (candles)", min_value=10); _n(cfg, "vp_bins", "Rows", min_value=5)
            _n(cfg, "vp_value_area", "Value area %")
            _sel(cfg, "vp_mode", "VP rule", ["Price vs POC", "Breakout outside value area"])
        if _cb(cfg, "bos", "Break of Structure (BOS)"):
            _n(cfg, "bos_lookback", "BOS within last N candles", min_value=1)
            _cb(cfg, "bos_incl_choch", "Count CHoCH as a break too", default=True)
        if _cb(cfg, "choch", "Change of Character (CHoCH)"):
            _n(cfg, "choch_lookback", "CHoCH within last N candles", min_value=1)
        if _cb(cfg, "fvg", "Fair Value Gap"):
            _n(cfg, "fvg_lookback", "FVG within last N candles", min_value=3)
            _n(cfg, "fvg_min_size_atr", "Min gap size (× ATR)")
            _sel(cfg, "fvg_mode", "FVG rule", ["Unfilled FVG in direction", "Price retesting FVG"])
        if _cb(cfg, "smc", "SMC filters"):
            _cb(cfg, "smc_trend", "Structure trend must align", default=True)
            _cb(cfg, "smc_pd", "Premium/Discount (LONG in discount, SHORT in premium)")
        if _cb(cfg, "ict", "ICT filters"):
            _cb(cfg, "ict_kz", "Killzone time window", default=True)
            cfg["flt_ict_killzones"] = st.text_input("Killzones (IST)", value=FILTER_DEFAULTS["ict_killzones"], key="plus_ict_kz_txt")
            _cb(cfg, "ict_ote", "Optimal Trade Entry (fib retracement)")
            _n(cfg, "ict_ote_low", "OTE low"); _n(cfg, "ict_ote_high", "OTE high")
            _cb(cfg, "ict_pd_bias", "Previous-day bias (above/below PD mid)")
        if any(cfg.get(f"flt_{k}") for k in ("bos", "choch", "smc", "ict")):
            _n(cfg, "swing_len", "Swing pivot length (structure)", min_value=1)
        if _cb(cfg, "angle", "Crossover angle filter"):
            _sel(cfg, "angle_mode", "Angle type", ["ATR-normalized (recommended)", "Crossover angle Fast vs Slow (ATR-norm)",
                                                   "Raw degrees (existing calc)"])
            _n(cfg, "angle_min", "Min angle °"); _n(cfg, "angle_max", "Max angle °")
        if _cb(cfg, "pullback", "EMA crossover + pullback entry (EMA Crossover strategy)",
               help="After a crossover, wait for the FIRST pullback that touches the fast EMA and closes back in trend direction."):
            _n(cfg, "pullback_window", "Max candles after crossover", min_value=2)
            _n(cfg, "pullback_tol_atr", "Touch tolerance (× ATR)")
        if _cb(cfg, "sustain", "Candle sustaining filter",
               help="Signal must hold for N more closed candles before entry."):
            _n(cfg, "sustain_candles", "Candles to sustain", min_value=1)
            _sel(cfg, "sustain_mode", "Sustain rule", ["Close holds beyond Fast EMA", "Close beyond signal-candle Close",
                                                       "Close beyond signal-candle High/Low"])
        if _cb(cfg, "sweep", "Liquidity sweep"):
            _n(cfg, "sweep_lookback", "Sweep within last N candles", min_value=1)
            _sel(cfg, "sweep_mode", "Liquidity level", ["Swing high/low", "Previous day high/low", "Either"])
            if not any(cfg.get(f"flt_{k}") for k in ("bos", "choch", "smc", "ict")):
                _n(cfg, "swing_len", "Swing pivot length", min_value=1)
        if _cb(cfg, "atr", "ATR filter"):
            _n(cfg, "atr_period", "ATR period", min_value=1)
            _sel(cfg, "atr_mode", "ATR unit", ["Points", "% of price"])
            if cfg["flt_atr_mode"] == "Points":
                _n(cfg, "atr_min", "ATR min (pts)"); _n(cfg, "atr_max", "ATR max (pts)")
            else:
                _n(cfg, "atr_pct_min", "ATR min %"); _n(cfg, "atr_pct_max", "ATR max %")
        if _cb(cfg, "regime", "Market regime (trending only)"):
            _n(cfg, "chop_period", "Choppiness period", min_value=2); _n(cfg, "chop_max", "Choppiness max (<38 strong trend, >61 range)")
            if _cb(cfg, "regime_use_adx", "Also require ADX ≥", default=True):
                _n(cfg, "regime_adx_min", "Regime ADX min")
        if _cb(cfg, "vwap", "VWAP filter (session VWAP)"):
            _n(cfg, "vwap_max_dist_pct", "Max distance from VWAP %")
        if _cb(cfg, "htf", "Higher-timeframe trend filter"):
            _n(cfg, "htf_minutes", "HTF minutes", min_value=2); _n(cfg, "htf_ema", "HTF EMA length", min_value=2)
            _cb(cfg, "htf_slope", "HTF EMA must slope with trade", default=True)


def render_config_ui_plus():
    global _ACTIVE_CFG
    cfg = _orig_render_config()
    render_plus_sidebar(cfg)
    _ss()["_active_config"] = cfg
    _ACTIVE_CFG = cfg
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# INSTALL PATCHES
# ════════════════════════════════════════════════════════════════════════════
def install():
    base.calculate_rsi = _patched_rsi
    base.calculate_atr = _patched_atr
    base.calculate_adx = _patched_adx
    base.calculate_ema_angle = fast_ema_angle
    base.calculate_all_indicators = calculate_all_indicators_plus
    base.fetch_data = fetch_data_plus
    base.run_backtest = run_backtest_plus
    base.render_backtest_ui = render_backtest_ui_plus
    base.render_config_ui = render_config_ui_plus
    base.DhanBrokerIntegration.enter_broker_position = enter_broker_position_plus
    base.STRATEGY_FUNCTIONS = {k: _make_wrapped(k, v) for k, v in _RAW_STRATEGIES.items()}


install()

if __name__ == "__main__":
    base.main()
