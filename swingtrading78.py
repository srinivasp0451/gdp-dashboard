"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ALGO TRADING PLATFORM  —  Streamlit + yfinance                ║
║  Run with:  streamlit run algo_trading.py                                   ║
║  Install:   pip install streamlit yfinance pandas numpy plotly              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from itertools import product as itertools_product
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Algo Trading Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TICKER_MAP = {
    "Nifty 50":      "^NSEI",
    "Bank Nifty":    "^NSEBANK",
    "Nifty IT":      "^CNXIT",
    "Sensex":        "^BSESN",
    "BTC/USD":       "BTC-USD",
    "ETH/USD":       "ETH-USD",
    "USD/INR":       "USDINR=X",
    "Gold":          "GC=F",
    "Silver":        "SI=F",
    "EUR/USD":       "EURUSD=X",
    "GBP/USD":       "GBPUSD=X",
    "USD/JPY":       "JPYUSD=X",
    "Crude Oil":     "CL=F",
    "Custom":        "CUSTOM",
}

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]

PERIODS = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

STRATEGIES = [
    # ── Basic ──
    "EMA Crossover",
    "RSI Overbought/Oversold",
    "Simple Buy",
    "Simple Sell",
    "Price Threshold Cross",
    "Bollinger Bands",
    "RSI Divergence",
    # ── Advanced ──
    "MACD Crossover",
    "Supertrend",
    "ADX + DI Crossover",
    "Stochastic Oscillator",
    "VWAP Deviation",
    "Ichimoku Cloud",
    "BB + RSI Mean Reversion",
    "Donchian Breakout",
    "Triple EMA Trend",
    "Heikin Ashi EMA",
    "Volume Price Trend (VPT)",
    "Keltner Channel Breakout",
    "Williams %R Reversal",
    # ── Custom ──
    "Custom Strategy",
]

SL_TYPES = [
    "Custom Points",
    "Trailing SL (Points)",
    "Trailing Prev Candle Low/High",
    "Trailing Curr Candle Low/High",
    "Trailing Prev Swing Low/High",
    "Trailing Curr Swing Low/High",
    "Cost to Cost (Breakeven)",
    "EMA Reverse Crossover",
    "ATR Based",
]

TARGET_TYPES = [
    "Custom Points",
    "Trailing Target (Display Only)",
    "Trailing Prev Candle High/Low",
    "Trailing Curr Candle High/Low",
    "Trailing Prev Swing High/Low",
    "Trailing Curr Swing High/Low",
    "ATR Based",
    "Risk/Reward Based",
]

# yfinance interval mapping (4h not natively supported → fetch 1h, resample)
YF_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "1h",   # 4h fetched as 1h, then resampled
    "1d": "1d", "1wk": "1wk",
}

# ──────────────────────────────────────────────────────────────────────────────
# DHAN API PLACEHOLDER  (Uncomment & configure to use)
# ──────────────────────────────────────────────────────────────────────────────
# from dhanhq import dhanhq
#
# DHAN_CLIENT_ID    = "YOUR_CLIENT_ID"
# DHAN_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
# dhan = dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
#
# IS_OPTIONS = False   # True when trading Nifty/BankNifty options
# LOT_SIZE   = 50      # Nifty=50, BankNifty=15 — adjust per instrument
# PRODUCT_TYPE = "INTRADAY"   # or "DELIVERY" for positional / swing
#
# def dhan_place_order(symbol, direction, qty):
#     """
#     Place order via Dhan.
#
#     OPTIONS logic (I am a BUYER, not a seller):
#       BUY  signal → CE Buy  (bullish call buying)
#       SELL signal → PE Buy  (bearish put buying)
#       (Never selling/writing options)
#
#     STOCKS / SWING / POSITIONAL / INTRADAY:
#       BUY  signal → Buy stock  (can be buyer)
#       SELL signal → Short/Sell stock  (can be seller too)
#     """
#     if IS_OPTIONS:
#         option_type  = "CE" if direction == "BUY" else "PE"
#         security_id  = get_atm_option_security_id(symbol, option_type)
#         txn_type     = dhan.BUY   # always buying options (CE or PE)
#     else:
#         security_id  = symbol
#         txn_type     = dhan.BUY if direction == "BUY" else dhan.SELL
#
#     response = dhan.place_order(
#         security_id      = security_id,
#         exchange_segment = dhan.NSE,
#         transaction_type = txn_type,
#         quantity         = qty,
#         order_type       = dhan.MARKET,
#         product_type     = PRODUCT_TYPE,
#         price            = 0,
#     )
#     return response
#
# def dhan_exit_order(symbol, direction, qty):
#     """Square-off / exit an existing position."""
#     txn_type = dhan.SELL if direction == "BUY" else dhan.BUY
#     response = dhan.place_order(
#         security_id      = symbol,
#         exchange_segment = dhan.NSE,
#         transaction_type = txn_type,
#         quantity         = qty,
#         order_type       = dhan.MARKET,
#         product_type     = PRODUCT_TYPE,
#         price            = 0,
#     )
#     return response
#
# def get_atm_option_security_id(underlying, option_type):
#     """
#     Lookup ATM strike security_id from Dhan's instrument master.
#     Implement: fetch current price → round to nearest strike
#               → match in instrument CSV → return security_id
#     """
#     raise NotImplementedError("Implement ATM option lookup")

# ──────────────────────────────────────────────────────────────────────────────
# DATA FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def _flatten_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns & keep OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(c).strip().title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[keep].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample("4h").agg(agg).dropna()


@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV with 1.5 s rate-limit delay.  Returns empty df on error."""
    yf_interval = YF_INTERVAL_MAP.get(interval, interval)
    try:
        time.sleep(1.5)
        raw = yf.download(ticker, period=period, interval=yf_interval,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        df = _flatten_and_clean(raw)
        if interval == "4h" and not df.empty:
            df = _resample_4h(df)
        return df
    except Exception as exc:
        st.error(f"Data fetch error: {exc}")
        return pd.DataFrame()


def fetch_live_data(ticker: str, interval: str) -> pd.DataFrame:
    """Live fetch (no caching) with 1.5 s delay."""
    yf_interval = YF_INTERVAL_MAP.get(interval, interval)
    # For live, always grab the freshest window
    live_period = "1d" if interval in ("1m", "5m", "15m", "30m") else "5d"
    try:
        time.sleep(1.5)
        raw = yf.download(ticker, period=live_period, interval=yf_interval,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        df = _flatten_and_clean(raw)
        if interval == "4h" and not df.empty:
            df = _resample_4h(df)
        return df
    except Exception as exc:
        st.warning(f"Live fetch warning: {exc}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid  = sma(series, period)
    dev  = series.rolling(period).std()
    return mid - std_mult * dev, mid, mid + std_mult * dev


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ml    = ema(series, fast) - ema(series, slow)
    sl    = ema(ml, signal)
    return ml, sl, ml - sl


def stochastic(df: pd.DataFrame, k=14, d=3):
    lo   = df["Low"].rolling(k).min()
    hi   = df["High"].rolling(k).max()
    pk   = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    pd_  = sma(pk, d)
    return pk, pd_


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df.get("Volume", pd.Series(1, index=df.index)).replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum()


def supertrend(df: pd.DataFrame, period=7, mult=3.0):
    _atr  = calc_atr(df, period)
    hl2   = (df["High"] + df["Low"]) / 2
    basic_upper = hl2 + mult * _atr
    basic_lower = hl2 - mult * _atr

    f_upper = basic_upper.copy()
    f_lower = basic_lower.copy()
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        bu, bl = basic_upper.iloc[i], basic_lower.iloc[i]
        pu, pl = f_upper.iloc[i - 1], f_lower.iloc[i - 1]
        close_prev = df["Close"].iloc[i - 1]

        f_upper.iloc[i] = bu if bu < pu or close_prev > pu else pu
        f_lower.iloc[i] = bl if bl > pl or close_prev < pl else pl

        if df["Close"].iloc[i] > f_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < f_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    line = pd.Series(
        np.where(direction == 1, f_lower.values, f_upper.values),
        index=df.index,
    )
    return line, direction


def adx_plus_di(df: pd.DataFrame, period=14):
    _atr    = calc_atr(df, period)
    up      = df["High"].diff()
    down    = -df["Low"].diff()
    pdm     = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    ndm     = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    pdi     = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / _atr.replace(0, np.nan)
    ndi     = 100 * ndm.ewm(alpha=1/period, adjust=False).mean() / _atr.replace(0, np.nan)
    dx      = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    _adx    = dx.ewm(alpha=1/period, adjust=False).mean()
    return _adx, pdi, ndi


def ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou_b_p=52):
    def mid(hi, lo, p): return (hi.rolling(p).max() + lo.rolling(p).min()) / 2
    t  = mid(df["High"], df["Low"], tenkan)
    k  = mid(df["High"], df["Low"], kijun)
    sa = ((t + k) / 2).shift(kijun)
    sb = mid(df["High"], df["Low"], senkou_b_p).shift(kijun)
    return t, k, sa, sb


def donchian_channel(df: pd.DataFrame, period=20):
    upper = df["High"].rolling(period).max()
    lower = df["Low"].rolling(period).min()
    return upper, (upper + lower) / 2, lower


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha       = pd.DataFrame(index=df.index)
    ha["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha["Open"]  = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
    ha["Open"].iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    ha["High"]  = pd.concat([df["High"], ha["Open"], ha["Close"]], axis=1).max(axis=1)
    ha["Low"]   = pd.concat([df["Low"],  ha["Open"], ha["Close"]], axis=1).min(axis=1)
    return ha


def vpt(df: pd.DataFrame) -> pd.Series:
    vol = df.get("Volume", pd.Series(1, index=df.index))
    return (df["Close"].pct_change() * vol).cumsum()


def keltner_channel(df: pd.DataFrame, ema_p=20, atr_p=10, mult=2.0):
    mid   = ema(df["Close"], ema_p)
    _atr  = calc_atr(df, atr_p)
    return mid - mult * _atr, mid, mid + mult * _atr


def williams_r(df: pd.DataFrame, period=14) -> pd.Series:
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    return -100 * (hi - df["Close"]) / (hi - lo).replace(0, np.nan)


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY SIGNAL GENERATORS
# Returns: (signals: pd.Series[int], indicators: dict[str, pd.Series])
# signals: 1=LONG, -1=SHORT, 0=flat  — fires on the SIGNAL bar (bar i)
# ──────────────────────────────────────────────────────────────────────────────
def _crossover(a, b):
    """Series that is 1 when a crosses above b on this bar."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def sig_ema_crossover(df, fast=9, slow=15, **kw):
    fe = ema(df["Close"], fast)
    se = ema(df["Close"], slow)
    s  = pd.Series(0, index=df.index)
    s[_crossover(fe, se)]  =  1
    s[_crossunder(fe, se)] = -1
    return s, {"EMA_fast": fe, "EMA_slow": se}


def sig_rsi_os(df, period=14, ob=70, os_=30, **kw):
    r = rsi(df["Close"], period)
    s = pd.Series(0, index=df.index)
    s[_crossover(r, pd.Series(os_, index=df.index))]  =  1   # RSI crosses above oversold
    s[_crossunder(r, pd.Series(ob, index=df.index))]  = -1   # RSI crosses below overbought
    return s, {"RSI": r}


def sig_simple_buy(df, **kw):
    s = pd.Series(0, index=df.index)
    s.iloc[:-1] = 1
    return s, {}


def sig_simple_sell(df, **kw):
    s = pd.Series(0, index=df.index)
    s.iloc[:-1] = -1
    return s, {}


def sig_price_threshold(df, threshold=0.0, **kw):
    th = pd.Series(float(threshold), index=df.index)
    s  = pd.Series(0, index=df.index)
    s[_crossover(df["Close"], th)]  =  1
    s[_crossunder(df["Close"], th)] = -1
    return s, {"Threshold": th}


def sig_bollinger(df, period=20, std=2.0, **kw):
    lo, mid, hi = bollinger_bands(df["Close"], period, std)
    s = pd.Series(0, index=df.index)
    s[_crossover(df["Close"], lo)]  =  1   # price crosses above lower band (mean-rev buy)
    s[_crossunder(df["Close"], hi)] = -1   # price crosses below upper band (mean-rev sell)
    return s, {"BB_upper": hi, "BB_mid": mid, "BB_lower": lo}


def sig_rsi_divergence(df, period=14, lookback=5, **kw):
    r  = rsi(df["Close"], period)
    cl = df["Close"]
    s  = pd.Series(0, index=df.index)
    for i in range(lookback, len(df)):
        past_cl  = cl.iloc[i - lookback: i]
        past_rsi = r.iloc[i - lookback: i]
        # Bullish divergence: price lower low, RSI higher low
        if cl.iloc[i] < past_cl.min() and r.iloc[i] > past_rsi.min():
            s.iloc[i] = 1
        # Bearish divergence: price higher high, RSI lower high
        elif cl.iloc[i] > past_cl.max() and r.iloc[i] < past_rsi.max():
            s.iloc[i] = -1
    return s, {"RSI": r}


def sig_macd(df, fast=12, slow=26, signal=9, **kw):
    ml, sl, _ = macd(df["Close"], fast, slow, signal)
    s = pd.Series(0, index=df.index)
    s[_crossover(ml, sl)]  =  1
    s[_crossunder(ml, sl)] = -1
    return s, {"MACD": ml, "MACD_Signal": sl}


def sig_supertrend(df, period=7, multiplier=3.0, **kw):
    line, direction = supertrend(df, period, multiplier)
    s = pd.Series(0, index=df.index)
    s[(direction == 1) & (direction.shift(1) == -1)]  =  1
    s[(direction == -1) & (direction.shift(1) == 1)]  = -1
    return s, {"Supertrend": line}


def sig_adx_di(df, period=14, adx_thresh=25, **kw):
    _adx, pdi, ndi = adx_plus_di(df, period)
    at = pd.Series(float(adx_thresh), index=df.index)
    s  = pd.Series(0, index=df.index)
    s[_crossover(pdi, ndi) & (_adx > at)]  =  1
    s[_crossunder(pdi, ndi) & (_adx > at)] = -1
    return s, {"ADX": _adx, "+DI": pdi, "-DI": ndi}


def sig_stochastic(df, k=14, d=3, ob=80, os_=20, **kw):
    pk, pd_ = stochastic(df, k, d)
    s = pd.Series(0, index=df.index)
    # %K crosses above %D while below overbought → buy
    s[_crossover(pk, pd_) & (pk < ob)]  =  1
    # %K crosses below %D while above oversold → sell
    s[_crossunder(pk, pd_) & (pk > os_)] = -1
    return s, {"Stoch_K": pk, "Stoch_D": pd_}


def sig_vwap(df, dev_pct=1.0, **kw):
    vw   = calc_vwap(df)
    d    = dev_pct / 100
    hi_b = vw * (1 + d)
    lo_b = vw * (1 - d)
    s    = pd.Series(0, index=df.index)
    s[_crossunder(df["Close"], lo_b)] =  1   # price dips below lower band → mean-rev buy
    s[_crossover(df["Close"], hi_b)]  = -1   # price pops above upper band → mean-rev sell
    return s, {"VWAP": vw, "VWAP_hi": hi_b, "VWAP_lo": lo_b}


def sig_ichimoku(df, tenkan=9, kijun=26, **kw):
    t, k, sa, sb = ichimoku(df, tenkan, kijun)
    cloud_top    = pd.concat([sa, sb], axis=1).max(axis=1)
    cloud_bottom = pd.concat([sa, sb], axis=1).min(axis=1)
    s = pd.Series(0, index=df.index)
    s[_crossover(df["Close"], cloud_top)]    =  1
    s[_crossunder(df["Close"], cloud_bottom)] = -1
    return s, {"Tenkan": t, "Kijun": k, "Senkou_A": sa, "Senkou_B": sb}


def sig_bb_rsi(df, bb_period=20, bb_std=2.0, rsi_period=14, rsi_os=35, rsi_ob=65, **kw):
    lo, mid, hi = bollinger_bands(df["Close"], bb_period, bb_std)
    r = rsi(df["Close"], rsi_period)
    s = pd.Series(0, index=df.index)
    s[(df["Close"] < lo) & (r < rsi_os)] =  1
    s[(df["Close"] > hi) & (r > rsi_ob)] = -1
    return s, {"BB_upper": hi, "BB_lower": lo, "RSI": r}


def sig_donchian(df, period=20, **kw):
    hi, mid, lo = donchian_channel(df, period)
    s = pd.Series(0, index=df.index)
    # price closes above prior period high → breakout long
    s[df["Close"] > hi.shift(1)]  =  1
    s[df["Close"] < lo.shift(1)] = -1
    return s, {"Don_upper": hi, "Don_lower": lo}


def sig_triple_ema(df, f=9, m=21, s_=50, **kw):
    e1 = ema(df["Close"], f)
    e2 = ema(df["Close"], m)
    e3 = ema(df["Close"], s_)
    bull  = (e1 > e2) & (e2 > e3)
    bear  = (e1 < e2) & (e2 < e3)
    s     = pd.Series(0, index=df.index)
    s[bull & ~bull.shift(1).fillna(False)]  =  1
    s[bear & ~bear.shift(1).fillna(False)] = -1
    return s, {"EMA_fast": e1, "EMA_mid": e2, "EMA_slow": e3}


def sig_heikin_ashi_ema(df, ema_period=20, **kw):
    ha   = heikin_ashi(df)
    e    = ema(df["Close"], ema_period)
    bull = ha["Close"] > ha["Open"]
    s    = pd.Series(0, index=df.index)
    s[bull & ~bull.shift(1).fillna(False) & (df["Close"] > e)]  =  1
    s[~bull & bull.shift(1).fillna(True)  & (df["Close"] < e)] = -1
    return s, {"EMA": e}


def sig_vpt(df, vpt_ema_period=14, **kw):
    v      = vpt(df)
    v_sig  = ema(v, vpt_ema_period)
    s      = pd.Series(0, index=df.index)
    s[_crossover(v, v_sig)]  =  1
    s[_crossunder(v, v_sig)] = -1
    return s, {}


def sig_keltner(df, ema_p=20, atr_p=10, mult=2.0, **kw):
    lo, mid, hi = keltner_channel(df, ema_p, atr_p, mult)
    s = pd.Series(0, index=df.index)
    s[_crossover(df["Close"], hi)]  =  1
    s[_crossunder(df["Close"], lo)] = -1
    return s, {"KC_upper": hi, "KC_mid": mid, "KC_lower": lo}


def sig_williams_r(df, period=14, ob=-20, os_=-80, **kw):
    wr = williams_r(df, period)
    s  = pd.Series(0, index=df.index)
    s[_crossover(wr, pd.Series(float(os_), index=df.index))]  =  1
    s[_crossunder(wr, pd.Series(float(ob), index=df.index))] = -1
    return s, {"Williams_%R": wr}


def sig_custom(df, **kw):
    """
    ──────────────────────────────────────────────────────────
    CUSTOM STRATEGY PLACEHOLDER
    Replace the body of this function with your own logic.
    Return (signals, indicators) where:
      signals  — pd.Series of 1 / -1 / 0 indexed on df.index
      indicators — dict of pd.Series for chart overlay
    ──────────────────────────────────────────────────────────
    Example (EMA + RSI combo):
        fe = ema(df['Close'], 9)
        r  = rsi(df['Close'], 14)
        s  = pd.Series(0, index=df.index)
        s[(fe > ema(df['Close'], 21)) & (r < 40)] = 1
        s[(fe < ema(df['Close'], 21)) & (r > 60)] = -1
        return s, {'EMA9': fe, 'RSI': r}
    """
    s = pd.Series(0, index=df.index)
    return s, {}


STRATEGY_FN = {
    "EMA Crossover":              sig_ema_crossover,
    "RSI Overbought/Oversold":    sig_rsi_os,
    "Simple Buy":                 sig_simple_buy,
    "Simple Sell":                sig_simple_sell,
    "Price Threshold Cross":      sig_price_threshold,
    "Bollinger Bands":            sig_bollinger,
    "RSI Divergence":             sig_rsi_divergence,
    "MACD Crossover":             sig_macd,
    "Supertrend":                 sig_supertrend,
    "ADX + DI Crossover":         sig_adx_di,
    "Stochastic Oscillator":      sig_stochastic,
    "VWAP Deviation":             sig_vwap,
    "Ichimoku Cloud":             sig_ichimoku,
    "BB + RSI Mean Reversion":    sig_bb_rsi,
    "Donchian Breakout":          sig_donchian,
    "Triple EMA Trend":           sig_triple_ema,
    "Heikin Ashi EMA":            sig_heikin_ashi_ema,
    "Volume Price Trend (VPT)":   sig_vpt,
    "Keltner Channel Breakout":   sig_keltner,
    "Williams %R Reversal":       sig_williams_r,
    "Custom Strategy":            sig_custom,
}


# ──────────────────────────────────────────────────────────────────────────────
# SL / TARGET CALCULATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def _swing_low(df, idx, lb=5):
    start = max(0, idx - lb)
    return float(df["Low"].iloc[start:idx].min()) if idx > 0 else float(df["Low"].iloc[0])


def _swing_high(df, idx, lb=5):
    start = max(0, idx - lb)
    return float(df["High"].iloc[start:idx].max()) if idx > 0 else float(df["High"].iloc[0])


def _atr_at(df, idx, period=14) -> float:
    vals = calc_atr(df, period)
    v    = vals.iloc[idx]
    return float(v) if not np.isnan(v) else 10.0


def init_sl(df, idx, entry, direction, sl_type, sl_pts, params) -> float:
    atr_m = params.get("atr_mult_sl", 1.5)
    lb    = params.get("swing_lookback", 5)
    atr_v = _atr_at(df, idx)

    if sl_type == "Custom Points":
        return entry - direction * sl_pts

    elif sl_type == "Trailing SL (Points)":
        return entry - direction * sl_pts

    elif sl_type == "Trailing Prev Candle Low/High":
        if idx < 1:
            return entry - direction * sl_pts
        return float(df["Low"].iloc[idx - 1]) if direction == 1 else float(df["High"].iloc[idx - 1])

    elif sl_type == "Trailing Curr Candle Low/High":
        return float(df["Low"].iloc[idx]) if direction == 1 else float(df["High"].iloc[idx])

    elif sl_type in ("Trailing Prev Swing Low/High", "Trailing Curr Swing Low/High"):
        return _swing_low(df, idx, lb) if direction == 1 else _swing_high(df, idx, lb)

    elif sl_type == "Cost to Cost (Breakeven)":
        return entry - direction * sl_pts   # starts as custom; shifts to entry once in profit

    elif sl_type == "EMA Reverse Crossover":
        return entry - direction * sl_pts   # placeholder; actual exit driven by signal

    elif sl_type == "ATR Based":
        return entry - direction * atr_m * atr_v

    return entry - direction * sl_pts


def init_target(df, idx, entry, direction, tgt_type, tgt_pts, sl_val, params) -> float:
    atr_m = params.get("atr_mult_tgt", 2.0)
    rr    = params.get("rr_ratio", 2.0)
    lb    = params.get("swing_lookback", 5)
    atr_v = _atr_at(df, idx)

    if tgt_type == "Custom Points":
        return entry + direction * tgt_pts

    elif tgt_type == "Trailing Target (Display Only)":
        return entry + direction * tgt_pts   # display value; flag prevents triggering

    elif tgt_type in ("Trailing Prev Candle High/Low", "Trailing Curr Candle High/Low"):
        return float(df["High"].iloc[idx]) if direction == 1 else float(df["Low"].iloc[idx])

    elif tgt_type in ("Trailing Prev Swing High/Low", "Trailing Curr Swing High/Low"):
        return _swing_high(df, idx, lb) if direction == 1 else _swing_low(df, idx, lb)

    elif tgt_type == "ATR Based":
        return entry + direction * atr_m * atr_v

    elif tgt_type == "Risk/Reward Based":
        sl_dist = abs(entry - sl_val)
        return entry + direction * rr * sl_dist

    return entry + direction * tgt_pts


def update_sl(df, bar_idx, entry, direction, sl_type, sl_pts, cur_sl, params) -> float:
    lb = params.get("swing_lookback", 5)

    if sl_type == "Custom Points":
        return cur_sl   # fixed

    elif sl_type == "Trailing SL (Points)":
        if direction == 1:
            cand = float(df["High"].iloc[bar_idx]) - sl_pts
            return max(cur_sl, cand)
        else:
            cand = float(df["Low"].iloc[bar_idx]) + sl_pts
            return min(cur_sl, cand)

    elif sl_type == "Trailing Prev Candle Low/High":
        if bar_idx < 1:
            return cur_sl
        if direction == 1:
            return max(cur_sl, float(df["Low"].iloc[bar_idx - 1]))
        else:
            return min(cur_sl, float(df["High"].iloc[bar_idx - 1]))

    elif sl_type == "Trailing Curr Candle Low/High":
        if direction == 1:
            return max(cur_sl, float(df["Low"].iloc[bar_idx]))
        else:
            return min(cur_sl, float(df["High"].iloc[bar_idx]))

    elif sl_type == "Trailing Prev Swing Low/High":
        if direction == 1:
            return max(cur_sl, _swing_low(df, bar_idx, lb))
        else:
            return min(cur_sl, _swing_high(df, bar_idx, lb))

    elif sl_type == "Trailing Curr Swing Low/High":
        if direction == 1:
            return max(cur_sl, _swing_low(df, bar_idx + 1, lb))
        else:
            return min(cur_sl, _swing_high(df, bar_idx + 1, lb))

    elif sl_type == "Cost to Cost (Breakeven)":
        sl_dist = abs(entry - cur_sl)
        if direction == 1 and float(df["High"].iloc[bar_idx]) >= entry + sl_dist:
            return max(cur_sl, entry)
        if direction == -1 and float(df["Low"].iloc[bar_idx]) <= entry - sl_dist:
            return min(cur_sl, entry)
        return cur_sl

    elif sl_type in ("EMA Reverse Crossover", "ATR Based"):
        return cur_sl   # handled externally

    return cur_sl


def update_target(df, bar_idx, direction, tgt_type, tgt_pts, cur_tgt, params):
    """Returns (new_target, should_trigger).  Display-only → should_trigger=False."""
    lb = params.get("swing_lookback", 5)

    if tgt_type == "Custom Points":
        return cur_tgt, True

    elif tgt_type == "Trailing Target (Display Only)":
        # Advance display value but never trigger
        if direction == 1:
            new_t = max(cur_tgt, float(df["High"].iloc[bar_idx]))
        else:
            new_t = min(cur_tgt, float(df["Low"].iloc[bar_idx]))
        return new_t, False

    elif tgt_type == "Trailing Prev Candle High/Low":
        if bar_idx < 1:
            return cur_tgt, True
        if direction == 1:
            return max(cur_tgt, float(df["High"].iloc[bar_idx - 1])), True
        else:
            return min(cur_tgt, float(df["Low"].iloc[bar_idx - 1])), True

    elif tgt_type == "Trailing Curr Candle High/Low":
        if direction == 1:
            return max(cur_tgt, float(df["High"].iloc[bar_idx])), True
        else:
            return min(cur_tgt, float(df["Low"].iloc[bar_idx])), True

    elif tgt_type == "Trailing Prev Swing High/Low":
        if direction == 1:
            return max(cur_tgt, _swing_high(df, bar_idx, lb)), True
        else:
            return min(cur_tgt, _swing_low(df, bar_idx, lb)), True

    elif tgt_type == "Trailing Curr Swing High/Low":
        if direction == 1:
            return max(cur_tgt, _swing_high(df, bar_idx + 1, lb)), True
        else:
            return min(cur_tgt, _swing_low(df, bar_idx + 1, lb)), True

    elif tgt_type in ("ATR Based", "Risk/Reward Based"):
        return cur_tgt, True   # fixed after entry

    return cur_tgt, True


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(df, strategy, params, sl_type, sl_pts, tgt_type, tgt_pts):
    """
    ENTRY / EXIT RULES
    ══════════════════
    Bar i   → signal fires on CLOSE  (strategy condition met, bar fully closed)
    Bar i+1 → ENTRY at OPEN of this bar; SL & Target calculated here
    Bar i+1 onward → each bar:
        1. Update trailing SL / Target
        2. Check Low  <= SL   → SL Hit   (conservative: SL checked FIRST)
        3. Check High >= Target → Target Hit
        (Reversed for SHORT trades)
    If both breach in same candle → SL is taken (conservative)
    """
    fn = STRATEGY_FN.get(strategy, sig_custom)
    try:
        signals, indicators = fn(df, **params)
    except Exception as exc:
        st.warning(f"Strategy error: {exc}")
        return [], {}

    n = len(df)
    trades = []
    i = 0

    while i < n - 1:
        sig = int(signals.iloc[i])
        if sig == 0:
            i += 1
            continue

        direction = sig
        entry_idx = i + 1
        if entry_idx >= n:
            break

        entry = float(df["Open"].iloc[entry_idx])
        sl    = init_sl(df, entry_idx, entry, direction, sl_type, sl_pts, params)
        tgt   = init_target(df, entry_idx, entry, direction, tgt_type, tgt_pts, sl, params)

        highest   = entry
        lowest    = entry
        exit_bar  = None
        exit_px   = None
        exit_why  = None
        disp_tgt  = tgt

        for j in range(entry_idx, n):
            bar_hi = float(df["High"].iloc[j])
            bar_lo = float(df["Low"].iloc[j])
            highest = max(highest, bar_hi)
            lowest  = min(lowest,  bar_lo)

            # Update trailing SL
            sl = update_sl(df, j, entry, direction, sl_type, sl_pts, sl, params)

            # Update trailing Target
            disp_tgt, tgt_triggers = update_target(df, j, direction, tgt_type,
                                                     tgt_pts, disp_tgt, params)
            if tgt_triggers:
                tgt = disp_tgt

            # EMA reverse crossover exit
            if sl_type == "EMA Reverse Crossover":
                rev = int(signals.iloc[j])
                if rev != 0 and rev != direction:
                    exit_bar = j
                    exit_px  = float(df["Open"].iloc[j])
                    exit_why = "EMA Reverse Crossover"
                    break

            # ── Exit check: SL first (conservative) ───────────────────────
            if direction == 1:
                if bar_lo <= sl:
                    exit_bar, exit_px, exit_why = j, sl, "SL Hit"
                    break
                if tgt_triggers and bar_hi >= tgt:
                    exit_bar, exit_px, exit_why = j, tgt, "Target Hit"
                    break
            else:
                if bar_hi >= sl:
                    exit_bar, exit_px, exit_why = j, sl, "SL Hit"
                    break
                if tgt_triggers and bar_lo <= tgt:
                    exit_bar, exit_px, exit_why = j, tgt, "Target Hit"
                    break

        if exit_bar is None:
            exit_bar = n - 1
            exit_px  = float(df["Close"].iloc[exit_bar])
            exit_why = "End of Data"

        pnl = round((exit_px - entry) * direction, 4)
        trades.append({
            "Signal Bar":     df.index[i],
            "Entry DateTime": df.index[entry_idx],
            "Entry Price":    round(entry, 4),
            "Direction":      "LONG" if direction == 1 else "SHORT",
            "SL Type":        sl_type,
            "Target Type":    tgt_type,
            "SL Level":       round(sl, 4),
            "Target Level":   round(disp_tgt, 4),
            "Exit DateTime":  df.index[exit_bar],
            "Exit Price":     round(exit_px, 4),
            "Exit Reason":    exit_why,
            "Highest Price":  round(highest, 4),
            "Lowest Price":   round(lowest, 4),
            "Points Gained":  round(max(pnl, 0), 4),
            "Points Lost":    round(abs(min(pnl, 0)), 4),
            "PnL":            pnl,
        })

        i = exit_bar + 1

    return trades, indicators


def calc_performance(trades):
    if not trades:
        return {}
    t    = len(trades)
    wins = [x for x in trades if x["PnL"] > 0]
    loss = [x for x in trades if x["PnL"] < 0]
    pnls = [x["PnL"] for x in trades]
    return {
        "Total Trades":  t,
        "Wins":          len(wins),
        "Losses":        len(loss),
        "Accuracy (%)":  round(len(wins) / t * 100, 2),
        "Total PnL":     round(sum(pnls), 2),
        "Avg Win":       round(np.mean([x["PnL"] for x in wins]) if wins else 0, 2),
        "Avg Loss":      round(np.mean([x["PnL"] for x in loss]) if loss else 0, 2),
        "Max Win":       round(max(pnls), 2),
        "Max Loss":      round(min(pnls), 2),
        "Profit Factor": round(
            sum(x["PnL"] for x in wins) / abs(sum(x["PnL"] for x in loss))
            if loss else float("inf"), 2
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
PARAM_GRIDS = {
    "EMA Crossover":           {"fast": [5, 9, 12, 20],  "slow": [15, 21, 26, 50]},
    "RSI Overbought/Oversold": {"period": [9, 14, 21],   "ob": [65, 70, 75, 80], "os_": [20, 25, 30]},
    "Bollinger Bands":         {"period": [15, 20, 25],  "std": [1.5, 2.0, 2.5]},
    "MACD Crossover":          {"fast": [8, 12, 16],     "slow": [21, 26, 30],   "signal": [7, 9, 11]},
    "Supertrend":              {"period": [5, 7, 10, 14],"multiplier": [2.0, 2.5, 3.0, 3.5]},
    "ADX + DI Crossover":      {"period": [10, 14, 20],  "adx_thresh": [20, 25, 30]},
    "Stochastic Oscillator":   {"k": [9, 14, 21],        "d": [3, 5],  "ob": [75, 80], "os_": [20, 25]},
    "Donchian Breakout":       {"period": [10, 15, 20, 30]},
    "Triple EMA Trend":        {"f": [5, 9, 12],         "m": [15, 21, 26],      "s_": [40, 50, 60]},
    "BB + RSI Mean Reversion": {
        "bb_period": [15, 20],  "bb_std": [1.5, 2.0, 2.5],
        "rsi_period": [10, 14], "rsi_os": [25, 30, 35],    "rsi_ob": [65, 70, 75],
    },
    "Keltner Channel Breakout": {"ema_p": [14, 20, 26],  "atr_p": [10, 14],      "mult": [1.5, 2.0, 2.5]},
    "Williams %R Reversal":    {"period": [9, 14, 21],   "ob": [-20, -25],        "os_": [-75, -80]},
}


def optimize_strategy(df, strategy, sl_type, sl_pts, tgt_type, tgt_pts,
                       desired_acc, progress_cb=None):
    grid = PARAM_GRIDS.get(strategy)
    if not grid:
        trades, _ = run_backtest(df, strategy, {}, sl_type, sl_pts, tgt_type, tgt_pts)
        perf = calc_performance(trades)
        if perf.get("Total Trades", 0) > 0:
            return [{"params": {}, **perf}]
        return []

    keys   = list(grid.keys())
    combos = list(itertools_product(*[grid[k] for k in keys]))
    total  = len(combos)
    results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params.update({"atr_mult_sl": 1.5, "atr_mult_tgt": 2.0,
                        "rr_ratio": 2.0, "swing_lookback": 5})
        try:
            trades, _ = run_backtest(df, strategy, params, sl_type, sl_pts, tgt_type, tgt_pts)
            perf = calc_performance(trades)
            if perf.get("Total Trades", 0) >= 3 and perf.get("Accuracy (%)", 0) >= desired_acc:
                results.append({"params": params, **perf})
        except Exception:
            pass
        if progress_cb:
            progress_cb(min((idx + 1) / total, 1.0))

    results.sort(key=lambda r: (-r["Accuracy (%)"], -r["Total PnL"]))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
OVERLAY_SKIP = {"RSI", "MACD", "MACD_Signal", "ADX", "+DI", "-DI",
                "Stoch_K", "Stoch_D", "Williams_%R"}
LINE_COLORS  = ["#2196F3", "#FF9800", "#9C27B0", "#00BCD4",
                "#4CAF50", "#F44336", "#FFEB3B", "#E91E63"]


def plot_ohlc(df, trades=None, indicators=None, title="OHLC"):
    has_vol = "Volume" in df.columns and df["Volume"].sum() > 0
    rows    = [0.72, 0.28] if has_vol else [1.0]
    n_rows  = 2 if has_vol else 1

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=rows, vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if has_vol:
        bar_colors = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume", marker_color=bar_colors, opacity=0.5,
        ), row=2, col=1)

    # Overlay indicators
    if indicators:
        c_idx = 0
        for name, series in indicators.items():
            if not isinstance(series, pd.Series) or name in OVERLAY_SKIP:
                continue
            col = LINE_COLORS[c_idx % len(LINE_COLORS)]
            dash = "dash" if "lower" in name.lower() or "lo" in name.lower() else "solid"
            fig.add_trace(go.Scatter(
                x=series.index, y=series, name=name,
                line=dict(color=col, width=1.5, dash=dash), opacity=0.85,
            ), row=1, col=1)
            c_idx += 1

    # Trade markers
    if trades:
        entry_x = [t["Entry DateTime"] for t in trades]
        entry_y = [t["Entry Price"]    for t in trades]
        exit_x  = [t["Exit DateTime"]  for t in trades]
        exit_y  = [t["Exit Price"]     for t in trades]
        e_col   = ["#00E676" if t["Direction"] == "LONG" else "#FF5252" for t in trades]
        x_col   = ["#26a69a" if t["Exit Reason"] == "Target Hit" else "#ef5350" for t in trades]
        e_sym   = ["triangle-up"   if t["Direction"] == "LONG" else "triangle-down"
                   for t in trades]

        fig.add_trace(go.Scatter(
            x=entry_x, y=entry_y, mode="markers",
            marker=dict(symbol=e_sym, size=13, color=e_col,
                        line=dict(color="white", width=1)),
            name="Entry", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=exit_x, y=exit_y, mode="markers",
            marker=dict(symbol="x", size=11, color=x_col,
                        line=dict(color="white", width=1)),
            name="Exit", showlegend=True,
        ), row=1, col=1)

    fig.update_layout(
        title=title, template="plotly_dark", height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(t=60, b=20),
    )
    return fig


def plot_equity(trades):
    if not trades:
        return None
    cum   = np.cumsum([t["PnL"] for t in trades])
    times = [t["Exit DateTime"] for t in trades]
    color = "#00E676" if cum[-1] >= 0 else "#FF5252"
    fig   = go.Figure(go.Scatter(
        x=times, y=cum, mode="lines+markers",
        fill="tozeroy", fillcolor=f"rgba(0,230,118,0.1)" if cum[-1] >= 0
                                 else "rgba(255,82,82,0.1)",
        line=dict(color=color, width=2), name="PnL",
    ))
    fig.update_layout(
        title="Equity Curve", template="plotly_dark", height=300,
        yaxis_title="Cumulative PnL (Points)", margin=dict(t=40, b=20),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY PARAM UI BUILDER
# ──────────────────────────────────────────────────────────────────────────────
def strategy_params_ui(strategy: str, prefix: str) -> dict:
    p = {}

    if strategy == "EMA Crossover":
        c1, c2 = st.columns(2)
        p["fast"] = c1.number_input("EMA Fast", 2, 200, 9,  key=f"{prefix}_fast")
        p["slow"] = c2.number_input("EMA Slow", 2, 500, 15, key=f"{prefix}_slow")

    elif strategy == "RSI Overbought/Oversold":
        c1, c2, c3 = st.columns(3)
        p["period"] = c1.number_input("RSI Period", 2, 100, 14, key=f"{prefix}_rp")
        p["ob"]     = c2.number_input("Overbought", 50, 95,  70, key=f"{prefix}_ob")
        p["os_"]    = c3.number_input("Oversold",   5,  50,  30, key=f"{prefix}_os")

    elif strategy == "Price Threshold Cross":
        p["threshold"] = st.number_input("Threshold Price", value=0.0, format="%.2f",
                                          key=f"{prefix}_thresh")

    elif strategy == "Bollinger Bands":
        c1, c2 = st.columns(2)
        p["period"] = c1.number_input("BB Period", 5, 200, 20, key=f"{prefix}_bbp")
        p["std"]    = c2.number_input("Std Dev",  0.5, 5.0, 2.0, step=0.1, key=f"{prefix}_bbs")

    elif strategy == "RSI Divergence":
        c1, c2 = st.columns(2)
        p["period"]   = c1.number_input("RSI Period",   2, 100, 14, key=f"{prefix}_rp2")
        p["lookback"] = c2.number_input("Lookback Bars", 2, 50,  5, key=f"{prefix}_lb")

    elif strategy == "MACD Crossover":
        c1, c2, c3 = st.columns(3)
        p["fast"]   = c1.number_input("MACD Fast",   2, 100, 12, key=f"{prefix}_mf")
        p["slow"]   = c2.number_input("MACD Slow",   5, 200, 26, key=f"{prefix}_ms")
        p["signal"] = c3.number_input("MACD Signal", 2, 100,  9, key=f"{prefix}_msig")

    elif strategy == "Supertrend":
        c1, c2 = st.columns(2)
        p["period"]     = c1.number_input("Period",     2, 50,   7, key=f"{prefix}_stp")
        p["multiplier"] = c2.number_input("Multiplier", 0.5, 10.0, 3.0, step=0.5,
                                           key=f"{prefix}_stm")

    elif strategy == "ADX + DI Crossover":
        c1, c2 = st.columns(2)
        p["period"]     = c1.number_input("ADX Period",    2, 50, 14, key=f"{prefix}_ap")
        p["adx_thresh"] = c2.number_input("ADX Threshold", 10, 50, 25, key=f"{prefix}_at")

    elif strategy == "Stochastic Oscillator":
        c1, c2, c3, c4 = st.columns(4)
        p["k"]   = c1.number_input("%K",        2, 50, 14, key=f"{prefix}_k")
        p["d"]   = c2.number_input("%D",        2, 20,  3, key=f"{prefix}_d")
        p["ob"]  = c3.number_input("OB",       50, 95, 80, key=f"{prefix}_so")
        p["os_"] = c4.number_input("OS",        5, 50, 20, key=f"{prefix}_su")

    elif strategy == "VWAP Deviation":
        p["dev_pct"] = st.number_input("Deviation %", 0.1, 10.0, 1.0, step=0.1,
                                        key=f"{prefix}_vd")

    elif strategy == "Ichimoku Cloud":
        c1, c2 = st.columns(2)
        p["tenkan"] = c1.number_input("Tenkan",  2, 50,  9, key=f"{prefix}_it")
        p["kijun"]  = c2.number_input("Kijun",   5, 100, 26, key=f"{prefix}_ik")

    elif strategy == "BB + RSI Mean Reversion":
        c1, c2 = st.columns(2)
        p["bb_period"]  = c1.number_input("BB Period", 5, 100, 20,  key=f"{prefix}_brbbp")
        p["bb_std"]     = c2.number_input("BB Std",   0.5, 5.0, 2.0, step=0.1, key=f"{prefix}_brbbs")
        c3, c4, c5 = st.columns(3)
        p["rsi_period"] = c3.number_input("RSI Period", 2, 50, 14, key=f"{prefix}_brrp")
        p["rsi_os"]     = c4.number_input("RSI OS",     5, 50, 35, key=f"{prefix}_bro")
        p["rsi_ob"]     = c5.number_input("RSI OB",    50, 95, 65, key=f"{prefix}_brob")

    elif strategy == "Donchian Breakout":
        p["period"] = st.number_input("Channel Period", 5, 200, 20, key=f"{prefix}_dp")

    elif strategy == "Triple EMA Trend":
        c1, c2, c3 = st.columns(3)
        p["f"]   = c1.number_input("EMA1 Fast", 2, 50,   9, key=f"{prefix}_tf")
        p["m"]   = c2.number_input("EMA2 Mid",  5, 100, 21, key=f"{prefix}_tm")
        p["s_"]  = c3.number_input("EMA3 Slow",10, 300, 50, key=f"{prefix}_ts")

    elif strategy == "Heikin Ashi EMA":
        p["ema_period"] = st.number_input("EMA Period", 5, 200, 20, key=f"{prefix}_hap")

    elif strategy == "Volume Price Trend (VPT)":
        p["vpt_ema_period"] = st.number_input("Signal EMA", 2, 100, 14, key=f"{prefix}_vp")

    elif strategy == "Keltner Channel Breakout":
        c1, c2, c3 = st.columns(3)
        p["ema_p"] = c1.number_input("EMA Period", 5, 100, 20, key=f"{prefix}_kep")
        p["atr_p"] = c2.number_input("ATR Period", 2, 50,  10, key=f"{prefix}_kap")
        p["mult"]  = c3.number_input("Multiplier", 0.5, 5.0, 2.0, step=0.25, key=f"{prefix}_km")

    elif strategy == "Williams %R Reversal":
        c1, c2, c3 = st.columns(3)
        p["period"] = c1.number_input("Period", 2, 50, 14, key=f"{prefix}_wrp")
        p["ob"]     = c2.number_input("OB (e.g. -20)", -5, -1, -20, key=f"{prefix}_wrob")
        p["os_"]    = c3.number_input("OS (e.g. -80)", -99, -50, -80, key=f"{prefix}_wros")

    # Common hidden params
    p.setdefault("atr_mult_sl",    1.5)
    p.setdefault("atr_mult_tgt",   2.0)
    p.setdefault("rr_ratio",       2.0)
    p.setdefault("swing_lookback", 5)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
_defaults = {
    "live_active":   False,
    "live_trades":   [],
    "live_position": None,
    "live_tick":     0,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
st.title("🚀 Algo Trading Platform")
st.caption("yfinance · Nifty50 · BankNifty · Sensex · BTC · ETH · Forex · Gold · Silver")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Global Config")

    t_choice = st.selectbox("Instrument", list(TICKER_MAP.keys()), key="g_ticker")
    if t_choice == "Custom":
        c_sym = st.text_input("Yahoo Ticker Symbol", "RELIANCE.NS", key="g_custom")
        sym   = c_sym.strip()
    else:
        sym = TICKER_MAP[t_choice]

    interval = st.selectbox("Timeframe", TIMEFRAMES, index=4, key="g_interval")
    period   = st.selectbox("Period",    PERIODS,    index=4, key="g_period")

    st.markdown("---")
    st.subheader("📈 Strategy")
    strategy = st.selectbox("Strategy", STRATEGIES, key="g_strategy")

    st.subheader("🛡️ Stop Loss")
    sl_type = st.selectbox("SL Type",  SL_TYPES, key="g_sl_type")
    sl_pts  = st.number_input("SL Value (points)", 0.01, 100000.0, 10.0,
                               step=0.5, key="g_sl_pts")

    st.subheader("🎯 Target")
    tgt_type = st.selectbox("Target Type", TARGET_TYPES, key="g_tgt_type")
    tgt_pts  = st.number_input("Target Value (points)", 0.01, 100000.0, 20.0,
                                step=0.5, key="g_tgt_pts")

    st.markdown("---")
    st.caption("ℹ️ 1.5 s rate-limit delay enforced between yfinance requests.")


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_bt, tab_live, tab_opt = st.tabs(
    ["📊  Backtesting", "⚡  Live Trading", "🔬  Optimization"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader(f"Backtesting  ·  {t_choice}  [{interval} / {period}]")

    with st.expander("⚙️ Strategy Parameters", expanded=True):
        bt_params = strategy_params_ui(strategy, prefix="bt")

    if st.button("▶  Run Backtest", type="primary", key="btn_bt"):
        with st.spinner("Fetching data…"):
            df_bt = fetch_data(sym, period, interval)

        if df_bt is None or df_bt.empty:
            st.error("No data returned. Try a different ticker / interval / period.")
        else:
            with st.spinner("Running backtest…"):
                trades_bt, indics_bt = run_backtest(
                    df_bt, strategy, bt_params,
                    sl_type, sl_pts, tgt_type, tgt_pts,
                )
            perf_bt = calc_performance(trades_bt)

            # ── Performance metrics ───────────────────────────────────────
            st.markdown("### 📋 Performance Summary")
            if perf_bt:
                m_cols = st.columns(len(perf_bt))
                for col, (k, v) in zip(m_cols, perf_bt.items()):
                    delta_str = None
                    if k == "Accuracy (%)":
                        delta_str = f"{v:.1f}%"
                    col.metric(k, v, delta=delta_str)
            else:
                st.info("No trades generated.")

            # ── OHLC + Signal chart ───────────────────────────────────────
            st.markdown("### 📈 Price Chart  (OHLC · Signals · Indicators)")
            fig_bt = plot_ohlc(
                df_bt, trades_bt, indics_bt,
                title=f"{t_choice} — {strategy}  [{interval}]",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # ── Equity curve ──────────────────────────────────────────────
            if trades_bt:
                eq_fig = plot_equity(trades_bt)
                if eq_fig:
                    st.markdown("### 💹 Equity Curve")
                    st.plotly_chart(eq_fig, use_container_width=True)

                # ── Trade log ─────────────────────────────────────────────
                st.markdown("### 📜 Detailed Trade Log")
                tdf = pd.DataFrame(trades_bt)
                tdf["Entry DateTime"] = pd.to_datetime(tdf["Entry DateTime"])
                tdf["Exit DateTime"]  = pd.to_datetime(tdf["Exit DateTime"])

                def _style_pnl(v):
                    if isinstance(v, (int, float)):
                        return "color:#00E676" if v > 0 else ("color:#FF5252" if v < 0 else "")
                    return ""

                st.dataframe(
                    tdf.style.map(_style_pnl, subset=["PnL", "Points Gained", "Points Lost"]),
                    use_container_width=True, height=400,
                )

                # ── Raw OHLC ──────────────────────────────────────────────
                with st.expander("📂 Raw OHLC Data"):
                    st.dataframe(df_bt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader(f"⚡ Live Trading  ·  {t_choice}  [{interval}]")
    st.info("Rate-limit: 1.5 s minimum between yfinance requests.  "
            "Data auto-refreshes while active.")

    with st.expander("⚙️ Strategy Parameters (Live)", expanded=False):
        live_params = strategy_params_ui(strategy, prefix="lv")

    col_s, col_x, col_rst = st.columns([1, 1, 2])
    start_btn = col_s.button("▶ Start", type="primary",
                              disabled=st.session_state.live_active, key="btn_lv_start")
    stop_btn  = col_x.button("⏹ Stop",
                              disabled=not st.session_state.live_active, key="btn_lv_stop")
    reset_btn = col_rst.button("🗑 Clear History", key="btn_lv_reset")

    if start_btn:
        st.session_state.live_active   = True
        st.session_state.live_trades   = []
        st.session_state.live_position = None
        st.session_state.live_tick     = 0
        st.rerun()

    if stop_btn:
        st.session_state.live_active = False
        st.rerun()

    if reset_btn:
        st.session_state.live_trades   = []
        st.session_state.live_position = None
        st.rerun()

    live_monitor, live_hist = st.tabs(["📡  Live Monitor", "📜  Trade History"])

    with live_monitor:
        if st.session_state.live_active:
            st.session_state.live_tick += 1
            tick = st.session_state.live_tick

            tick_ph  = st.empty()
            param_ph = st.empty()
            chart_ph = st.empty()
            pos_ph   = st.empty()

            tick_ph.info(f"🔄 Tick #{tick}  ·  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")

            # ── Fetch live data ───────────────────────────────────────────
            lv_df = fetch_live_data(sym, interval)

            if lv_df is None or lv_df.empty:
                st.warning("⚠️ No live data received. Retrying…")
            else:
                last_close = float(lv_df["Close"].iloc[-1])
                last_bar   = lv_df.index[-1]

                # ── Strategy signals ──────────────────────────────────────
                fn = STRATEGY_FN.get(strategy, sig_custom)
                try:
                    lv_sigs, lv_indics = fn(lv_df, **live_params)
                except Exception as e:
                    lv_sigs   = pd.Series(0, index=lv_df.index)
                    lv_indics = {}
                    st.warning(f"Strategy error: {e}")

                last_sig = int(lv_sigs.iloc[-2]) if len(lv_sigs) > 1 else 0

                # ── Param display ─────────────────────────────────────────
                with param_ph.container():
                    st.markdown("#### 🔧 Active Parameters")
                    show_params = {k: v for k, v in live_params.items()
                                   if k not in ("atr_mult_sl", "atr_mult_tgt",
                                                "rr_ratio", "swing_lookback")}
                    show_params["SL Type"]  = sl_type
                    show_params["SL Pts"]   = sl_pts
                    show_params["Tgt Type"] = tgt_type
                    show_params["Tgt Pts"]  = tgt_pts

                    n_p = min(len(show_params), 8)
                    p_cols = st.columns(n_p)
                    for ci, (k, v) in enumerate(list(show_params.items())[:n_p]):
                        p_cols[ci].metric(k, v)

                    st.markdown("---")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Instrument",   t_choice)
                    m2.metric("Last Price",   f"{last_close:.2f}")
                    m3.metric("Signal",       "🟢 BUY" if last_sig == 1
                                              else ("🔴 SELL" if last_sig == -1 else "⚪ FLAT"))
                    m4.metric("Last Bar",     str(last_bar)[:19])

                # ── OHLC + EMA chart ──────────────────────────────────────
                with chart_ph.container():
                    lv_fig = plot_ohlc(
                        lv_df, indicators=lv_indics,
                        title=f"LIVE: {t_choice} ({interval}) — Tick #{tick}",
                    )
                    st.plotly_chart(lv_fig, use_container_width=True)

                # ── Position management ───────────────────────────────────
                pos = st.session_state.live_position

                if pos is None and last_sig != 0:
                    direction   = last_sig
                    entry_price = last_close
                    lv_sl  = init_sl(lv_df, len(lv_df) - 1, entry_price, direction,
                                     sl_type, sl_pts, live_params)
                    lv_tgt = init_target(lv_df, len(lv_df) - 1, entry_price, direction,
                                         tgt_type, tgt_pts, lv_sl, live_params)
                    st.session_state.live_position = {
                        "entry":     entry_price,
                        "direction": direction,
                        "sl":        lv_sl,
                        "target":    lv_tgt,
                        "disp_tgt":  lv_tgt,
                        "entry_time":last_bar,
                        "highest":   entry_price,
                        "lowest":    entry_price,
                    }
                    # ── DHAN PLACE ORDER PLACEHOLDER ──────────────────────
                    # dhan_place_order(sym, "BUY" if direction==1 else "SELL", LOT_SIZE)
                    # ─────────────────────────────────────────────────────
                    st.success(
                        f"🚀 NEW {'LONG' if direction==1 else 'SHORT'}  "
                        f"Entry={entry_price:.2f}  SL={lv_sl:.2f}  Target={lv_tgt:.2f}"
                    )

                elif pos is not None:
                    direction   = pos["direction"]
                    entry_price = pos["entry"]
                    bar_hi = float(lv_df["High"].iloc[-1])
                    bar_lo = float(lv_df["Low"].iloc[-1])

                    pos["highest"] = max(pos["highest"], bar_hi)
                    pos["lowest"]  = min(pos["lowest"],  bar_lo)

                    pos["sl"] = update_sl(lv_df, len(lv_df) - 1, entry_price, direction,
                                           sl_type, sl_pts, pos["sl"], live_params)
                    new_tgt, tgt_trig = update_target(lv_df, len(lv_df) - 1, direction,
                                                       tgt_type, tgt_pts,
                                                       pos["disp_tgt"], live_params)
                    pos["disp_tgt"] = new_tgt
                    if tgt_trig:
                        pos["target"] = new_tgt

                    exited     = False
                    exit_px    = None
                    exit_why   = None

                    if direction == 1:
                        if bar_lo <= pos["sl"]:
                            exited, exit_px, exit_why = True, pos["sl"], "SL Hit"
                        elif tgt_trig and bar_hi >= pos["target"]:
                            exited, exit_px, exit_why = True, pos["target"], "Target Hit"
                    else:
                        if bar_hi >= pos["sl"]:
                            exited, exit_px, exit_why = True, pos["sl"], "SL Hit"
                        elif tgt_trig and bar_lo <= pos["target"]:
                            exited, exit_px, exit_why = True, pos["target"], "Target Hit"

                    if exited:
                        pnl = round((exit_px - entry_price) * direction, 4)
                        st.session_state.live_trades.append({
                            "Entry Time":   pos["entry_time"],
                            "Entry Price":  entry_price,
                            "Direction":    "LONG" if direction == 1 else "SHORT",
                            "Exit Time":    last_bar,
                            "Exit Price":   exit_px,
                            "Exit Reason":  exit_why,
                            "SL":           pos["sl"],
                            "Target":       pos["disp_tgt"],
                            "Highest":      pos["highest"],
                            "Lowest":       pos["lowest"],
                            "PnL":          pnl,
                        })
                        st.session_state.live_position = None
                        # ── DHAN EXIT ORDER PLACEHOLDER ───────────────────
                        # dhan_exit_order(sym, "BUY" if direction==1 else "SELL", LOT_SIZE)
                        # ─────────────────────────────────────────────────
                        if pnl > 0:
                            st.success(f"✅ CLOSED  {exit_why}  |  PnL: +{pnl:.2f}")
                        else:
                            st.error(f"❌ CLOSED  {exit_why}  |  PnL: {pnl:.2f}")
                    else:
                        unreal = round((last_close - entry_price) * direction, 4)
                        with pos_ph.container():
                            st.markdown("#### 📌 Open Position")
                            o1, o2, o3, o4, o5, o6, o7 = st.columns(7)
                            o1.metric("Dir",    "LONG 🟢" if direction == 1 else "SHORT 🔴")
                            o2.metric("Entry",  f"{entry_price:.2f}")
                            o3.metric("LTP",    f"{last_close:.2f}")
                            o4.metric("SL",     f"{pos['sl']:.2f}")
                            o5.metric("Target", f"{pos['disp_tgt']:.2f}")
                            o6.metric("Highest",f"{pos['highest']:.2f}")
                            o7.metric("Unrealised PnL", f"{unreal:.2f}",
                                       delta=f"{unreal:.2f}")

            # Auto-refresh
            time.sleep(1.5)
            st.rerun()

        else:
            st.info("Press  ▶ Start  to begin live trading.")

    with live_hist:
        st.subheader("📜 Completed Live Trades")
        if st.session_state.live_trades:
            hist_df = pd.DataFrame(st.session_state.live_trades)
            wins_l  = sum(1 for t in st.session_state.live_trades if t["PnL"] > 0)
            tot_l   = len(st.session_state.live_trades)
            tp_l    = sum(t["PnL"] for t in st.session_state.live_trades)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Trades", tot_l)
            c2.metric("Accuracy",     f"{wins_l/tot_l*100:.1f}%" if tot_l else "—")
            c3.metric("Total PnL",    f"{tp_l:.2f}")

            def _style_lv_pnl(v):
                if isinstance(v, (int, float)):
                    return "color:#00E676" if v > 0 else ("color:#FF5252" if v < 0 else "")
                return ""

            st.dataframe(
                hist_df.style.map(_style_lv_pnl, subset=["PnL"]),
                use_container_width=True,
            )

            eq_lv = plot_equity(st.session_state.live_trades)
            if eq_lv:
                st.plotly_chart(eq_lv, use_container_width=True)
        else:
            st.info("No completed trades yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("🔬 Strategy Parameter Optimization")
    st.markdown(
        "Grid-search over parameter combinations to find those that achieve "
        "**≥ desired accuracy**.  Results sorted by accuracy ↓ then PnL ↓."
    )

    with st.expander("Optimization Inputs", expanded=True):
        oc1, oc2, oc3 = st.columns(3)

        opt_t_choice = oc1.selectbox("Instrument", list(TICKER_MAP.keys()), key="opt_t")
        if opt_t_choice == "Custom":
            opt_sym = oc1.text_input("Custom Ticker", "RELIANCE.NS", key="opt_csym")
        else:
            opt_sym = TICKER_MAP[opt_t_choice]

        opt_interval = oc1.selectbox("Timeframe", TIMEFRAMES, index=4, key="opt_iv")
        opt_period   = oc2.selectbox("Period",    PERIODS,    index=5, key="opt_pd")
        opt_strategy = oc2.selectbox(
            "Strategy",
            [s for s in STRATEGIES if s in PARAM_GRIDS],
            key="opt_strat",
        )
        opt_desired  = oc3.slider("Desired Accuracy (%)", 50, 99, 65, key="opt_acc")

        oc4, oc5 = st.columns(2)
        opt_sl_type  = oc4.selectbox("SL Type",     SL_TYPES,     key="opt_sl")
        opt_sl_pts   = oc4.number_input("SL Points", 0.01, 100000.0, 10.0, key="opt_slp")
        opt_tgt_type = oc5.selectbox("Target Type", TARGET_TYPES, key="opt_tgt")
        opt_tgt_pts  = oc5.number_input("Target Points", 0.01, 100000.0, 20.0, key="opt_tgtp")

    if st.button("🔬  Run Optimization", type="primary", key="btn_opt"):
        with st.spinner("Fetching data…"):
            df_opt = fetch_data(opt_sym, opt_period, opt_interval)

        if df_opt is None or df_opt.empty:
            st.error("No data returned.")
        else:
            n_combos = 1
            g = PARAM_GRIDS.get(opt_strategy, {})
            for vals in g.values():
                n_combos *= len(vals)

            st.info(f"Data: {len(df_opt)} bars  |  Grid size: {n_combos} combinations")
            prog_bar = st.progress(0)
            status   = st.empty()

            with st.spinner(f"Optimising {opt_strategy}…"):
                opt_results = optimize_strategy(
                    df_opt, opt_strategy,
                    opt_sl_type, opt_sl_pts,
                    opt_tgt_type, opt_tgt_pts,
                    opt_desired,
                    progress_cb=prog_bar.progress,
                )

            prog_bar.empty()

            if not opt_results:
                st.warning(
                    f"No combination achieved ≥ {opt_desired}% accuracy with ≥ 3 trades.  "
                    "Try lower desired accuracy, longer period, or different SL/Target."
                )
            else:
                st.success(f"✅  {len(opt_results)} combination(s) found with accuracy ≥ {opt_desired}%")

                # Build result table
                rows = []
                for r in opt_results:
                    row = {}
                    row.update(r["params"])
                    row.update({k: v for k, v in r.items() if k != "params"})
                    rows.append(row)
                res_df = pd.DataFrame(rows)

                def _hl_acc(v):
                    if isinstance(v, (int, float)) and v >= opt_desired:
                        return "background-color:#1b5e20; color:white"
                    return ""

                if "Accuracy (%)" in res_df.columns:
                    st.dataframe(
                        res_df.style.map(_hl_acc, subset=["Accuracy (%)"]),
                        use_container_width=True, height=400,
                    )
                else:
                    st.dataframe(res_df, use_container_width=True)

                # Best result deep-dive
                st.markdown("### 🥇 Best Parameter Set — Backtest Deep-Dive")
                best = opt_results[0]
                best_params = {**best["params"],
                                "atr_mult_sl": 1.5, "atr_mult_tgt": 2.0,
                                "rr_ratio": 2.0,    "swing_lookback": 5}

                best_trades, best_indics = run_backtest(
                    df_opt, opt_strategy, best_params,
                    opt_sl_type, opt_sl_pts, opt_tgt_type, opt_tgt_pts,
                )
                best_perf = calc_performance(best_trades)

                bm_cols = st.columns(len(best_perf))
                for col, (k, v) in zip(bm_cols, best_perf.items()):
                    col.metric(k, v)

                fig_best = plot_ohlc(
                    df_opt, best_trades, best_indics,
                    title=f"Best: {best['params']}  |  Acc={best['Accuracy (%)']:.1f}%",
                )
                st.plotly_chart(fig_best, use_container_width=True)

                eq_best = plot_equity(best_trades)
                if eq_best:
                    st.plotly_chart(eq_best, use_container_width=True)

                # Detailed trade log for best
                if best_trades:
                    with st.expander("📜 Detailed Trade Log (Best Params)"):
                        btdf = pd.DataFrame(best_trades)

                        def _sp2(v):
                            if isinstance(v, (int, float)):
                                return "color:#00E676" if v > 0 else ("color:#FF5252" if v < 0 else "")
                            return ""

                        st.dataframe(
                            btdf.style.map(_sp2, subset=["PnL", "Points Gained", "Points Lost"]),
                            use_container_width=True,
                        )
