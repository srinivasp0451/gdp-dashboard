"""
╔══════════════════════════════════════════════════════════════════╗
║         NIFTY 50 ALGO TRADER — Smart Stock System               ║
║  Walk-Forward Backtest | Regime Detection | Live Trading         ║
║  3 Strategies | Portfolio Risk Mgmt | Dhan API Integration       ║
╚══════════════════════════════════════════════════════════════════╝
Run: streamlit run nifty50_algo_trader.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import threading
import time
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# IST TIMEZONE HELPER
# ─────────────────────────────────────────────────────────────────────────────
IST_OFFSET = timedelta(hours=5, minutes=30)

def now_ist() -> datetime:
    """Return current datetime in IST."""
    return datetime.utcnow() + IST_OFFSET

def fmt_ist(dt=None, include_date=True) -> str:
    """Format a datetime (or now) as IST string."""
    if dt is None:
        dt = now_ist()
    if include_date:
        return dt.strftime("%d %b %Y %H:%M:%S IST")
    return dt.strftime("%H:%M:%S IST")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON — survives Streamlit reruns via Python module cache
# ─────────────────────────────────────────────────────────────────────────────
_STORE = {
    "live_running":    False,
    "live_thread":     None,
    "live_log":        [],
    "positions":       {},
    "closed_trades":   [],
    "daily_pnl":       0.0,
    "portfolio_pnl":   0.0,
    "signals":         [],
    "circuit_breaker": False,
    "last_scan_time":  None,
    "last_error":      None,
    "ticker_status":   {},   # sym → {adx, rsi, regime, vol_ratio, signal_str}
    "lock":            threading.Lock(),
}

def _is_engine_alive() -> bool:
    """Check if background thread is actually running."""
    t = _STORE.get("live_thread")
    return t is not None and t.is_alive()

STATE_FILE = "algo_positions.json"

def _save_state():
    """Persist positions + closed trades to disk so app restarts don't lose state."""
    try:
        with _STORE["lock"]:
            data = {
                "positions":     _STORE["positions"],
                "closed_trades": _STORE["closed_trades"],
                "daily_pnl":     _STORE["daily_pnl"],
                "saved_at":      fmt_ist(),
            }
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        pass  # never crash the trading loop over a save error

def _load_state():
    """Load persisted state on startup if file exists."""
    try:
        if not __import__("os").path.exists(STATE_FILE):
            return
        with open(STATE_FILE) as f:
            data = json.load(f)
        with _STORE["lock"]:
            _STORE["positions"]     = data.get("positions", {})
            _STORE["closed_trades"] = data.get("closed_trades", [])
            _STORE["daily_pnl"]     = data.get("daily_pnl", 0.0)
    except Exception:
        pass

# Load persisted state once at module import time
_load_state()

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE — all instruments supported via yfinance
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK",
    "HINDUNILVR","ITC","SBIN","BHARTIARTL","KOTAKBANK",
    "LT","AXISBANK","ASIANPAINT","MARUTI","SUNPHARMA",
    "TITAN","BAJFINANCE","NESTLEIND","WIPRO","ULTRACEMCO",
    "POWERGRID","NTPC","TATAMOTORS","HCLTECH","BAJAJFINSV",
    "TECHM","INDUSINDBK","CIPLA","DRREDDY","DIVISLAB",
    "EICHERMOT","BPCL","GRASIM","HEROMOTOCO","BRITANNIA",
    "TATACONSUM","HINDALCO","TATASTEEL","JSWSTEEL","COALINDIA",
    "ADANIENT","ADANIPORTS","ONGC","M&M","APOLLOHOSP",
    "BAJAJ-AUTO","SBILIFE","HDFCLIFE","UPL","LTIM",
]

# Full instrument universe — label → yfinance ticker
INSTRUMENTS = {
    # ── Indian Indices ─────────────────────────────────────────────────
    "🇮🇳 Nifty 50":        "^NSEI",
    "🇮🇳 Bank Nifty":      "^NSEBANK",
    "🇮🇳 Nifty IT":        "^CNXIT",
    "🇮🇳 Nifty Midcap 50": "^NSEMDCP50",
    "🇮🇳 Sensex":          "^BSESN",
    # ── Crypto ────────────────────────────────────────────────────────
    "₿ Bitcoin (BTC-USD)":   "BTC-USD",
    "Ξ Ethereum (ETH-USD)":  "ETH-USD",
    "◎ Solana (SOL-USD)":    "SOL-USD",
    # ── Commodities ───────────────────────────────────────────────────
    "🥇 Gold (MCX proxy)":   "GC=F",
    "🥈 Silver (MCX proxy)": "SI=F",
    "🛢 Crude Oil (WTI)":    "CL=F",
    "🛢 Crude Oil (Brent)":  "BZ=F",
    "⚡ Natural Gas":         "NG=F",
    # ── Forex ─────────────────────────────────────────────────────────
    "💱 USD/INR":            "USDINR=X",
    "💱 EUR/USD":            "EURUSD=X",
    "💱 GBP/USD":            "GBPUSD=X",
    "💱 USD/JPY":            "JPY=X",
    # ── US Indices ────────────────────────────────────────────────────
    "🇺🇸 S&P 500":          "^GSPC",
    "🇺🇸 NASDAQ":           "^IXIC",
    "🇺🇸 Dow Jones":        "^DJI",
    # ── ETFs ──────────────────────────────────────────────────────────
    "📦 Nifty BeES":         "NIFTYBEES.NS",
    "📦 Gold BeES":          "GOLDBEES.NS",
}
# Reverse lookup: ticker → label
TICKER_LABEL = {v: k for k, v in INSTRUMENTS.items()}

# NSE Stocks get ".NS" appended automatically
NSE_STOCKS = NIFTY_50  # displayed separately in dropdowns

# ─────────────────────────────────────────────────────────────────────────────
# REALISTIC NSE TRANSACTION COSTS
# ─────────────────────────────────────────────────────────────────────────────
# Intraday: brokerage(0.03%) + STT-sell(0.025%) + exchange(0.00335%) + GST ~ 0.05%/side
# Delivery: brokerage(0.03%) + STT-sell(0.1%) + stamp + exchange ~ 0.15%/side
# + slippage (market impact)
COST_INTRADAY = 0.0005  # per side
COST_DELIVERY = 0.0015  # per side
SLIPPAGE      = 0.0003  # per side (market impact)
ROUND_TRIP_INTRADAY = 2 * (COST_INTRADAY + SLIPPAGE)   # ≈ 0.16%
ROUND_TRIP_DELIVERY = 2 * (COST_DELIVERY + SLIPPAGE)   # ≈ 0.36%

# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(symbol_ns: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(symbol_ns, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df = df[["Open","High","Low","Close","Volume"]].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    df.index.name = "Date"
    df = df.reset_index()          # Date becomes a column — preserved through WFO
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(symbol_ns: str, period: str = "60d",
                   interval: str = "1h") -> pd.DataFrame:
    df = yf.download(symbol_ns, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df = df[["Open","High","Low","Close","Volume"]].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

def get_ltp_yf(symbol: str) -> float:
    """Quick last price via yfinance fast_info as fallback"""
    try:
        t = yf.Ticker(symbol + ".NS")
        return float(t.fast_info.get("last_price", 0) or 0)
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS  (all forward-fill safe, no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=1).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift(1)).abs()
    lc  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=1).mean()

def adx(df: pd.DataFrame, n: int = 14):
    """Returns (adx, plus_di, minus_di)"""
    tr_ = atr(df, n)
    up   = df["High"].diff()
    down = -df["Low"].diff()
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0),   index=df.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    pdi  = 100 * pdm.ewm(span=n, adjust=False).mean() / (tr_ + 1e-10)
    mdi  = 100 * mdm.ewm(span=n, adjust=False).mean() / (tr_ + 1e-10)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    adx_ = dx.ewm(span=n, adjust=False).mean()
    return adx_, pdi, mdi

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"]    = ema(df["Close"], 9)
    df["ema21"]   = ema(df["Close"], 21)
    df["ema50"]   = ema(df["Close"], 50)
    df["ema200"]  = ema(df["Close"], 200)
    df["atr14"]   = atr(df, 14)
    df["atr_pct"] = df["atr14"] / df["Close"]
    df["adx14"], df["pdi"], df["mdi"] = adx(df, 14)
    df["rsi14"]   = rsi(df["Close"], 14)
    df["vol_ma20"]  = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_ma20"] + 1)
    df["bb_mid"]    = df["Close"].rolling(20).mean()
    bb_std          = df["Close"].rolling(20).std()
    df["bb_upper"]  = df["bb_mid"] + 2 * bb_std
    df["bb_lower"]  = df["bb_mid"] - 2 * bb_std
    df["hh20"]      = df["High"].shift(1).rolling(20).max()   # prior 20 bars high
    df["ll20"]      = df["Low"].shift(1).rolling(20).min()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────
REGIME_TRENDING_UP   = "trending_up"
REGIME_TRENDING_DOWN = "trending_down"
REGIME_RANGING       = "ranging"
REGIME_VOLATILE      = "volatile"
REGIME_NEUTRAL       = "neutral"

def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each bar into market regime:
      trending_up   : ADX>25, price>EMA50, +DI>-DI
      trending_down : ADX>25, price<EMA50, -DI>+DI
      ranging       : ADX<20
      volatile      : ATR% > 2.5x rolling median
      neutral       : transition / mixed
    """
    df = df.copy()
    med_atr = df["atr_pct"].rolling(60, min_periods=20).median()
    regime = []
    for i in range(len(df)):
        r  = df.iloc[i]
        ma = med_atr.iloc[i] if not pd.isna(med_atr.iloc[i]) else df["atr_pct"].median()
        if r["atr_pct"] > 2.5 * ma:
            regime.append(REGIME_VOLATILE)
        elif r["adx14"] > 25 and r["pdi"] > r["mdi"] and r["Close"] > r["ema50"]:
            regime.append(REGIME_TRENDING_UP)
        elif r["adx14"] > 25 and r["mdi"] > r["pdi"] and r["Close"] < r["ema50"]:
            regime.append(REGIME_TRENDING_DOWN)
        elif r["adx14"] < 20:
            regime.append(REGIME_RANGING)
        else:
            regime.append(REGIME_NEUTRAL)
    df["regime"] = regime
    return df

REGIME_EMOJI = {
    REGIME_TRENDING_UP:   "🟢",
    REGIME_TRENDING_DOWN: "🔴",
    REGIME_RANGING:       "🟡",
    REGIME_VOLATILE:      "🟠",
    REGIME_NEUTRAL:       "⚪",
}

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — EMA SWING  (trending regime, 3–10 day hold)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_EMA_SWING = {"fast": 9, "slow": 21, "atr_sl": 2.0, "rr": 3.0, "max_hold": 10}

def strategy_ema_swing(df: pd.DataFrame, params: dict = None,
                       oos_start_bar: int = 0) -> pd.DataFrame:
    """
    Edge: EMA-9 crosses above EMA-21 in confirmed uptrend.
    Entry: NEXT bar's Open after signal (realistic — signal fires on close, enter next open).
    Regime filter: ADX>22, +DI>-DI, price>EMA-50.
    SL = entry - 2×ATR. Target = entry + 3×risk. Max hold = 10 bars.
    oos_start_bar: only trades with entry_bar >= this are reported (WFO warmup fix).
    """
    p = {**DEFAULT_EMA_SWING, **(params or {})}
    df = df.copy().reset_index(drop=True)  # Date col preserved as column
    df["ef"] = ema(df["Close"], p["fast"])
    df["es"] = ema(df["Close"], p["slow"])
    df["a"]  = atr(df, 14)
    df["e50"]= ema(df["Close"], 50)
    adx_, pdi_, mdi_ = adx(df, 14)
    df["adx_"], df["pdi_"], df["mdi_"] = adx_, pdi_, mdi_

    trades, in_trade = [], False
    ep = sl = tgt = ei = signal_reason = 0

    for i in range(50, len(df) - 1):   # -1 so i+1 always exists for next-open entry
        r, prev = df.iloc[i], df.iloc[i-1]
        trending = (r["adx_"] > 22 and r["pdi_"] > r["mdi_"]
                    and r["Close"] > r["e50"])

        if not in_trade:
            cross_up = prev["ef"] <= prev["es"] and r["ef"] > r["es"]
            if cross_up and trending:
                ep  = df.iloc[i+1]["Open"]       # ← enter next bar open (realistic)
                sl  = ep - p["atr_sl"] * r["a"]
                tgt = ep + p["rr"] * (ep - sl)
                signal_reason = (f"EMA9 crossed above EMA21 | "
                                 f"ADX={r['adx_']:.1f} | Close ₹{r['Close']:.2f} > EMA50 ₹{r['e50']:.2f}")
                in_trade, ei = True, i+1          # entry bar is i+1

        else:
            bars = i - ei
            if r["Low"] <= sl:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, sl, "SL", "swing", df, sl, tgt, "EMA Swing", signal_reason))
                in_trade = False
            elif r["High"] >= tgt:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, tgt, "Target", "swing", df, sl, tgt, "EMA Swing", signal_reason))
                in_trade = False
            elif bars >= p["max_hold"]:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, r["Close"], "MaxHold", "swing", df, sl, tgt, "EMA Swing", signal_reason))
                in_trade = False

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — MOMENTUM BREAKOUT  (trending, 1–5 day hold)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BREAKOUT = {"lookback": 20, "vol_mult": 1.3, "atr_sl": 1.5,
                    "rr": 2.5, "max_hold": 5}

def strategy_momentum_breakout(df: pd.DataFrame, params: dict = None,
                               oos_start_bar: int = 0) -> pd.DataFrame:
    """
    Edge: Price closes above 20-bar high on volume surge (>1.3× avg).
    Entry: NEXT bar's Open after signal.
    Regime filter: ADX>20.
    SL = entry - 1.5×ATR. Target = entry + 2.5×risk. Max hold = 5 bars.
    """
    p = {**DEFAULT_BREAKOUT, **(params or {})}
    df = df.copy().reset_index(drop=True)  # Date col preserved as column
    df["a"]    = atr(df, 14)
    adx_, _, _ = adx(df, 14)
    df["adx_"] = adx_
    df["vma"]  = df["Volume"].rolling(20).mean()
    df["hh"]   = df["High"].shift(1).rolling(p["lookback"]).max()  # prior bars only

    trades, in_trade = [], False
    ep = sl = tgt = ei = signal_reason = 0

    for i in range(50, len(df) - 1):
        r = df.iloc[i]
        if not in_trade:
            breakout = (r["Close"] > r["hh"]
                        and r["Volume"] > p["vol_mult"] * r["vma"]
                        and r["adx_"] > 20)
            if breakout and not pd.isna(r["hh"]):
                ep  = df.iloc[i+1]["Open"]
                sl  = ep - p["atr_sl"] * r["a"]
                tgt = ep + p["rr"] * (ep - sl)
                signal_reason = (f"Close ₹{r['Close']:.2f} > 20d-High ₹{r['hh']:.2f} | "
                                 f"Vol {r['Volume']/r['vma']:.1f}× avg | ADX={r['adx_']:.1f}")
                in_trade, ei = True, i+1
        else:
            bars = i - ei
            if r["Low"] <= sl:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, sl, "SL", "intraday", df, sl, tgt, "Momentum Breakout", signal_reason))
                in_trade = False
            elif r["High"] >= tgt:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, tgt, "Target", "intraday", df, sl, tgt, "Momentum Breakout", signal_reason))
                in_trade = False
            elif bars >= p["max_hold"]:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, r["Close"], "MaxHold", "intraday", df, sl, tgt, "Momentum Breakout", signal_reason))
                in_trade = False

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — MEAN REVERSION  (ranging regime, 2–5 day hold)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REVERSION = {"rsi_thresh": 35, "atr_sl": 1.0, "max_hold": 5}

def strategy_mean_reversion(df: pd.DataFrame, params: dict = None,
                            oos_start_bar: int = 0) -> pd.DataFrame:
    """
    Edge: RSI<35 AND Close<BB-lower in a ranging market.
    Entry: NEXT bar's Open after signal.
    Regime filter: ADX<22.
    SL = entry - 1×ATR. Target = BB-mid. Max hold = 5 bars.
    """
    p = {**DEFAULT_REVERSION, **(params or {})}
    df = df.copy().reset_index(drop=True)  # Date col preserved as column
    df["a"]      = atr(df, 14)
    adx_, _, _   = adx(df, 14)
    df["adx_"]   = adx_
    df["rsi_"]   = rsi(df["Close"], 14)
    df["bbmid"]  = df["Close"].rolling(20).mean()
    bb_std        = df["Close"].rolling(20).std()
    df["bblower"] = df["bbmid"] - 2 * bb_std

    trades, in_trade = [], False
    ep = sl = tgt = ei = signal_reason = 0

    for i in range(50, len(df) - 1):
        r = df.iloc[i]
        if not in_trade:
            ranging  = r["adx_"] < 22
            oversold = r["rsi_"] < p["rsi_thresh"] and r["Close"] < r["bblower"]
            if oversold and ranging:
                ep  = df.iloc[i+1]["Open"]
                sl  = ep - p["atr_sl"] * r["a"]
                tgt = r["bbmid"]
                if tgt <= ep:
                    continue
                signal_reason = (f"RSI={r['rsi_']:.1f} < {p['rsi_thresh']} | "
                                 f"Close ₹{r['Close']:.2f} < BB-Lower ₹{r['bblower']:.2f} | "
                                 f"ADX={r['adx_']:.1f} (ranging)")
                in_trade, ei = True, i+1
        else:
            bars = i - ei
            if r["Low"] <= sl:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, sl, "SL", "swing", df, sl, tgt, "Mean Reversion", signal_reason))
                in_trade = False
            elif r["High"] >= tgt:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, tgt, "Target", "swing", df, sl, tgt, "Mean Reversion", signal_reason))
                in_trade = False
            elif bars >= p["max_hold"]:
                if ei >= oos_start_bar:
                    trades.append(_trade(ei, i, ep, r["Close"], "MaxHold", "swing", df, sl, tgt, "Mean Reversion", signal_reason))
                in_trade = False

    return pd.DataFrame(trades)


def _trade(ei, xi, ep, xp, reason, ttype, df=None, sl=None, tgt=None, strategy="", signal_reason=""):
    pct = (xp - ep) / ep
    entry_date = str(df.iloc[ei]["Date"])[:10] if df is not None and "Date" in df.columns else ""
    exit_date  = str(df.iloc[xi]["Date"])[:10] if df is not None and "Date" in df.columns else ""
    hold_days  = xi - ei
    return {
        "entry_bar":     ei,
        "exit_bar":      xi,
        "entry_date":    entry_date,
        "exit_date":     exit_date,
        "hold_days":     hold_days,
        "entry":         round(ep, 4),
        "exit":          round(xp, 4),
        "sl":            round(sl, 4) if sl is not None else "",
        "target":        round(tgt, 4) if tgt is not None else "",
        "pnl_gross":     pct,
        "trade_type":    ttype,
        "exit_reason":   reason,
        "strategy":      strategy,
        "signal_reason": signal_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION COST APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def apply_costs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["costs"] = df["trade_type"].map(
        {"swing": ROUND_TRIP_DELIVERY, "intraday": ROUND_TRIP_INTRADAY}
    ).fillna(ROUND_TRIP_DELIVERY)
    df["pnl_net"] = df["pnl_gross"] - df["costs"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_backtest(df: pd.DataFrame, strategy_func,
                          params: dict = None,
                          train_bars: int = 252,
                          test_bars:  int = 63):
    """
    Corrected Walk-Forward:
      - Passes FULL [start : start+train_bars+test_bars] to strategy so indicators
        like EMA200, ATR, ADX have proper warmup history (fixes warmup bias).
      - oos_start_bar = train_bars tells the strategy to only REPORT trades whose
        entry falls in the OOS window.
      - Rolls forward by test_bars each iteration.
    Returns (all_oos_trades_df, windows_summary_df)
    """
    df = df.reset_index(drop=True)  # integer index only; Date column stays intact
    oos_blocks, windows = [], []
    start, w = 0, 0

    while start + train_bars + test_bars <= len(df):
        te = start + train_bars
        xt = te + test_bars
        # ← KEY FIX: pass full window [start:xt] so indicators use train period as warmup
        full_window = df.iloc[start:xt].copy().reset_index(drop=True)
        oos_start   = train_bars   # relative index within full_window

        trades = strategy_func(full_window, params, oos_start_bar=oos_start)
        if not trades.empty:
            trades = apply_costs(trades)
            trades["window"]    = w
            trades["oos_start"] = te
            trades["oos_end"]   = xt
            oos_blocks.append(trades)

        windows.append({"window": w,
                         "train_start": start, "train_end": te,
                         "oos_start": te, "oos_end": xt,
                         "n_trades": len(trades)})
        start += test_bars
        w += 1

    all_trades = pd.concat(oos_blocks, ignore_index=True) if oos_blocks else pd.DataFrame()
    return all_trades, pd.DataFrame(windows)


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(trades: pd.DataFrame, capital: float = 200_000) -> dict:
    if trades.empty or "pnl_net" not in trades.columns:
        return {}
    p = trades["pnl_net"].values
    wins   = p[p > 0]
    losses = p[p < 0]

    win_rate   = len(wins) / len(p)
    avg_win    = wins.mean()    if len(wins)   > 0 else 0
    avg_loss   = abs(losses.mean()) if len(losses) > 0 else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    equity  = capital + np.cumsum(p * capital)
    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / peak
    max_dd  = dd.min()

    pf = wins.sum() / (abs(losses.sum()) + 1e-10) if len(losses) > 0 else 999

    # Annualised Sharpe: assume ~50 trades/year for swing
    annual_scale = np.sqrt(max(len(p), 1))
    sharpe = (p.mean() / (p.std() + 1e-10)) * annual_scale if len(p) > 1 else 0

    total_ret = equity[-1] / capital - 1

    # Consecutive losses
    max_consec_loss = 0
    consec = 0
    for x in p:
        if x < 0:
            consec += 1
            max_consec_loss = max(max_consec_loss, consec)
        else:
            consec = 0

    return {
        "total_trades":     len(p),
        "win_rate":         win_rate,
        "avg_win_pct":      avg_win * 100,
        "avg_loss_pct":     avg_loss * 100,
        "expectancy_pct":   expectancy * 100,
        "profit_factor":    pf,
        "max_drawdown_pct": max_dd * 100,
        "sharpe":           sharpe,
        "total_return_pct": total_ret * 100,
        "max_consec_loss":  max_consec_loss,
        "equity_curve":     equity.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STOCK SCREENER
# ─────────────────────────────────────────────────────────────────────────────
def _scan_with_health(tickers: list, capital: float, risk_pct: float):
    """
    Runs scan_signals and simultaneously builds per-ticker health dict showing
    raw indicator values and which condition blocked each strategy.
    Returns (signals_list, ticker_health_dict)
    """
    signals = scan_signals(tickers, capital, risk_pct)
    sig_syms = {s["symbol"] for s in signals}
    ticker_status = {}

    for raw in tickers:
        raw = raw.strip()
        if not raw:
            continue
        yf_ticker = raw if ("." in raw or "-" in raw or raw.startswith("^")) else raw + ".NS"
        display   = raw
        try:
            df = fetch_daily(yf_ticker, period="1y")
            if df.empty or len(df) < 60:
                ticker_status[display] = {"error": "No data / too short", "blocked_by": "data"}
                continue
            df = add_indicators(df)
            df = detect_regime(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]

            adx_v   = round(float(last["adx14"]), 1)
            rsi_v   = round(float(last["rsi14"]), 1)
            pdi_v   = round(float(last["pdi"]),   1)
            mdi_v   = round(float(last["mdi"]),   1)
            ema9_v  = round(float(last["ema9"]),  2)
            ema21_v = round(float(last["ema21"]), 2)
            ema50_v = round(float(last["ema50"]), 2)
            close_v = round(float(last["Close"]), 2)
            vol_r   = round(float(last["vol_ratio"]), 2)
            bbl_v   = round(float(last["bb_lower"]), 2)
            bbm_v   = round(float(last["bb_mid"]),   2)
            regime  = last["regime"]

            # Check each strategy condition
            strat_status = {}

            # EMA Swing
            cross_up = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
            trend_ok = adx_v > 22 and pdi_v > mdi_v and close_v > ema50_v
            if display in sig_syms:
                ema_status = "✅ SIGNAL"
            elif not cross_up and not trend_ok:
                ema_status = f"⛔ No cross (EMA9 {ema9_v:.0f} vs EMA21 {ema21_v:.0f}) & ADX {adx_v}<22"
            elif not cross_up:
                ema_status = f"⛔ No EMA9/21 cross (EMA9={ema9_v:.0f}, EMA21={ema21_v:.0f})"
            elif not trend_ok:
                ema_status = f"⛔ Trend fail: ADX={adx_v} +DI={pdi_v} -DI={mdi_v} vs EMA50={ema50_v:.0f}"
            else:
                ema_status = "✅ SIGNAL"
            strat_status["EMA Swing"] = ema_status

            # Momentum Breakout
            hh20    = float(df["High"].iloc[-21:-1].max())
            brk     = close_v > hh20
            vol_ok  = vol_r > 1.3
            adx_ok  = adx_v > 20
            if display in sig_syms:
                brk_status = "✅ SIGNAL"
            elif not brk:
                brk_status = f"⛔ No breakout: Close {close_v} ≤ 20d-High {hh20:.2f}"
            elif not vol_ok:
                brk_status = f"⛔ Low volume: {vol_r}× (need >1.3×)"
            elif not adx_ok:
                brk_status = f"⛔ ADX {adx_v} < 20 (weak trend)"
            else:
                brk_status = "✅ SIGNAL"
            strat_status["Breakout"] = brk_status

            # Mean Reversion
            rng  = adx_v < 22
            ovs  = rsi_v < 35 and close_v < bbl_v
            if display in sig_syms:
                rev_status = "✅ SIGNAL"
            elif not rng:
                rev_status = f"⛔ Not ranging: ADX {adx_v} ≥ 22 (need <22)"
            elif rsi_v >= 35:
                rev_status = f"⛔ RSI {rsi_v} not oversold (need <35)"
            elif close_v >= bbl_v:
                rev_status = f"⛔ Close {close_v} > BB-Lower {bbl_v:.2f}"
            else:
                rev_status = "✅ SIGNAL"
            strat_status["Mean Reversion"] = rev_status

            has_signal = display in sig_syms
            blocked_by = "signal!" if has_signal else (
                "no-cross" if "No EMA" in ema_status else
                "no-breakout" if "No breakout" in brk_status else
                "trend/ADX" if not trend_ok else "ranging/RSI"
            )
            ticker_status[display] = {
                "close": close_v, "adx": adx_v, "rsi": rsi_v,
                "pdi": pdi_v, "mdi": mdi_v, "vol_ratio": vol_r,
                "regime": regime, "ema9": ema9_v, "ema21": ema21_v,
                "ema50": ema50_v, "bb_lower": bbl_v, "bb_mid": bbm_v,
                "strat_status": strat_status,
                "has_signal": has_signal,
                "blocked_by": blocked_by,
            }
        except Exception as e:
            ticker_status[display] = {"error": str(e), "blocked_by": "error"}

    return signals, ticker_status


def scan_signals(tickers: list, capital: float, risk_pct: float) -> list:
    """
    Scans each ticker for live signals from all 3 strategies.
    Accepts full yfinance tickers (RELIANCE.NS, BTC-USD, ^NSEI etc.)
    or bare NSE symbols (RELIANCE → auto-appends .NS).
    Returns sorted list of signal dicts (best R:R first).
    """
    signals = []

    for raw in tickers:
        raw = raw.strip()
        if not raw:
            continue
        # Determine yfinance ticker and display symbol
        if "." in raw or "-" in raw or raw.startswith("^"):
            yf_ticker = raw          # already a full yfinance ticker
            display   = raw
        else:
            yf_ticker = raw + ".NS"  # bare NSE symbol
            display   = raw

        try:
            df = fetch_daily(yf_ticker, period="1y")
            if df.empty or len(df) < 60:
                continue
            df = add_indicators(df)
            df = detect_regime(df)

            last = df.iloc[-1]
            prev = df.iloc[-2]
            regime = last["regime"]

            # ── Strategy 1: EMA Swing ─────────────────────────────────────
            cross_up = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
            trend_ok = (last["adx14"] > 22 and last["pdi"] > last["mdi"]
                        and last["Close"] > last["ema50"])
            if cross_up and trend_ok:
                ep  = last["Close"]
                sl  = ep - 2.0 * last["atr14"]
                tgt = ep + 3.0 * (ep - sl)
                qty = max(1, int((capital * risk_pct) / (ep - sl)))
                signals.append(_signal(display, "EMA Swing", "Swing (3-10d)",
                                       regime, ep, sl, tgt, qty, last, 3.0))

            # ── Strategy 2: Momentum Breakout ─────────────────────────────
            hh20 = df["High"].iloc[-21:-1].max()
            vol_ok = last["vol_ratio"] > 1.3
            adx_ok = last["adx14"] > 20
            brk_up = last["Close"] > hh20 and vol_ok and adx_ok
            if brk_up and not pd.isna(hh20):
                ep  = last["Close"]
                sl  = ep - 1.5 * last["atr14"]
                tgt = ep + 2.5 * (ep - sl)
                qty = max(1, int((capital * risk_pct) / (ep - sl)))
                signals.append(_signal(display, "Momentum Breakout", "Intraday/Swing (1-5d)",
                                       regime, ep, sl, tgt, qty, last, 2.5))

            # ── Strategy 3: Mean Reversion ────────────────────────────────
            ranging  = last["adx14"] < 22
            oversold = last["rsi14"] < 35 and last["Close"] < last["bb_lower"]
            if ranging and oversold:
                ep  = last["Close"]
                sl  = ep - 1.0 * last["atr14"]
                tgt = last["bb_mid"]
                if tgt > ep:
                    rr  = (tgt - ep) / (ep - sl)
                    qty = max(1, int((capital * risk_pct) / (ep - sl)))
                    signals.append(_signal(display, "Mean Reversion", "Swing (2-5d)",
                                           regime, ep, sl, tgt, qty, last, round(rr, 2)))

        except Exception:
            continue

    signals.sort(key=lambda x: x["rr"], reverse=True)
    return signals


def _signal(sym, strat, style, regime, ep, sl, tgt, qty, row, rr):
    return {
        "symbol":   sym,
        "strategy": strat,
        "style":    style,
        "regime":   regime,
        "signal":   "BUY",
        "entry":    round(ep, 2),
        "sl":       round(sl, 2),
        "target":   round(tgt, 2),
        "qty":      qty,
        "rr":       rr,
        "atr":      round(row["atr14"], 2),
        "adx":      round(row["adx14"], 1),
        "rsi":      round(row["rsi14"], 1),
        "vol_ratio": round(row["vol_ratio"], 2),
        "max_risk": round((ep - sl) * qty, 0),
        "max_profit": round((tgt - ep) * qty, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────
def risk_check(positions: dict, daily_pnl: float, settings: dict) -> dict:
    cap       = settings["capital"]
    daily_pct = daily_pnl / cap
    deployed  = sum(p["entry"] * p["qty"] for p in positions.values())
    dep_pct   = deployed / cap

    return {
        "daily_loss_ok":    daily_pct > -settings["daily_loss_limit"],
        "positions_ok":     len(positions) < settings["max_positions"],
        "deployment_ok":    dep_pct < settings["max_deployment"],
        "daily_loss_pct":   daily_pct * 100,
        "deployment_pct":   dep_pct * 100,
        "position_count":   len(positions),
        "deployed_capital": deployed,
    }


def position_size(capital: float, risk_pct: float,
                  entry: float, sl: float) -> int:
    risk_rs = capital * risk_pct
    risk_share = abs(entry - sl)
    if risk_share <= 0:
        return 1
    return max(1, int(risk_rs / risk_share))


# ─────────────────────────────────────────────────────────────────────────────
# DHAN API WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class DhanAPI:
    BASE = "https://api.dhan.co"

    def __init__(self, client_id: str, token: str):
        self.client_id = client_id
        self.headers   = {
            "access-token": token,
            "client-id":    client_id,
            "Content-Type": "application/json",
        }

    def _get(self, path: str):
        try:
            r = requests.get(f"{self.BASE}{path}", headers=self.headers, timeout=10)
            return r.json() if r.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}

    def _post(self, path: str, body: dict):
        try:
            r = requests.post(f"{self.BASE}{path}", headers=self.headers,
                              data=json.dumps(body), timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def funds(self):
        return self._get("/fundlimit")

    def positions(self):
        return self._get("/positions")

    def holdings(self):
        return self._get("/holdings")

    def orders(self):
        return self._get("/orders")

    def place_order(self, symbol: str, security_id: str, qty: int,
                    txn: str, order_type: str = "MARKET",
                    price: float = 0, product: str = "CNC"):
        return self._post("/orders", {
            "dhanClientId":    self.client_id,
            "transactionType": txn,
            "exchangeSegment": "NSE_EQ",
            "productType":     product,
            "orderType":       order_type,
            "validity":        "DAY",
            "tradingSymbol":   symbol,
            "securityId":      str(security_id),
            "quantity":        qty,
            "price":           price,
            "triggerPrice":    0,
            "disclosedQuantity": 0,
            "afterMarketOrder":  False,
        })

    def ltp(self, security_ids: list):
        return self._post("/v2/marketfeed/ltp", {"NSE_EQ": [int(i) for i in security_ids]})


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TRADING LOOP (background thread)
# ─────────────────────────────────────────────────────────────────────────────
def _log(store, msg: str):
    ts = fmt_ist(include_date=False)
    with store["lock"]:
        store["live_log"].append(f"[{ts}] {msg}")
        if len(store["live_log"]) > 150:
            store["live_log"] = store["live_log"][-150:]


def live_loop(dhan: DhanAPI, settings: dict, store: dict):
    _log(store, f"🟢 Live trading engine started at {fmt_ist()}")
    paper_mode = settings.get("paper_mode", True)

    while store["live_running"]:
        try:
            now = now_ist()
            mo  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
            mc  = now.replace(hour=15, minute=20, second=0, microsecond=0)
            market_open = mo <= now <= mc

            # Paper mode runs 24/7; live mode only during market hours
            if not paper_mode and not market_open:
                _log(store, f"⏳ Market closed (IST {now.strftime('%H:%M')}) — next open 09:15 IST")
                time.sleep(60)
                continue

            if paper_mode and not market_open:
                _log(store, f"📄 Paper mode — scanning outside market hours ({now.strftime('%H:%M IST')})")

            # Circuit breaker check
            with store["lock"]:
                if store["circuit_breaker"]:
                    _log(store, "🔴 Circuit breaker ACTIVE — no new entries")
                    time.sleep(60)
                    continue

            # ── Scan for signals + build ticker health ────────────────────
            watchlist = settings.get("watchlist", [])
            if not watchlist:
                _log(store, "⚠️ No tickers configured. Add tickers in the Live tab.")
                time.sleep(30)
                continue

            _log(store, f"🔍 Scanning {len(watchlist)} ticker(s): {', '.join(str(w) for w in watchlist[:6])}{'…' if len(watchlist)>6 else ''}")

            signals, ticker_status = _scan_with_health(
                watchlist, settings["capital"], settings["risk_pct"]
            )

            with store["lock"]:
                store["signals"]         = signals
                store["ticker_status"]   = ticker_status
                store["last_scan_time"]  = fmt_ist()

            _log(store, f"📊 Scan complete — {len(signals)} signal(s) found across {len(ticker_status)} tickers")
            if not signals:
                blocked = [f"{s}({v['blocked_by']})" for s,v in ticker_status.items() if v.get("blocked_by")]
                if blocked:
                    _log(store, f"   No signal reasons: {', '.join(blocked[:6])}")

            if signals:
                with store["lock"]:
                    positions = dict(store["positions"])
                    daily_pnl = store["daily_pnl"]

                rc = risk_check(positions, daily_pnl, settings)

                if not rc["daily_loss_ok"]:
                    with store["lock"]:
                        store["circuit_breaker"] = True
                    _log(store, f"🔴 Daily loss limit hit ({rc['daily_loss_pct']:.2f}%). CB activated.")
                elif not rc["positions_ok"]:
                    _log(store, f"⚠️ Max positions ({rc['position_count']}) reached")
                elif not rc["deployment_ok"]:
                    _log(store, f"⚠️ Max deployment ({rc['deployment_pct']:.1f}%) reached")
                else:
                    with store["lock"]:
                        existing = set(store["positions"].keys())
                    for sig in signals[:3]:
                        if sig["symbol"] in existing:
                            continue
                        _log(store, (f"📈 {sig['symbol']} | {sig['strategy']} | "
                                     f"Entry ₹{sig['entry']} | SL ₹{sig['sl']} | "
                                     f"Tgt ₹{sig['target']} | Qty {sig['qty']} | "
                                     f"Risk ₹{sig['max_risk']:,.0f}"))
                        if settings.get("auto_trade", False) and not paper_mode:
                            _log(store, f"  ↳ Placing BUY order for {sig['symbol']}…")
                            resp = dhan.place_order(
                                symbol=sig["symbol"], security_id="0",
                                qty=sig["qty"], txn="BUY",
                                order_type="MARKET",
                                product="CNC" if "Swing" in sig["style"] else "INTRADAY",
                            )
                            _log(store, f"  ↳ Order response: {resp}")
                        with store["lock"]:
                            store["positions"][sig["symbol"]] = {
                                "strategy":   sig["strategy"],
                                "entry":      sig["entry"],
                                "sl":         sig["sl"],
                                "target":     sig["target"],
                                "qty":        sig["qty"],
                                "ltp":        sig["entry"],
                                "pnl_pct":    0.0,
                                "entry_time": fmt_ist(include_date=False),
                                "entry_date": now_ist().strftime("%Y-%m-%d"),
                            }
                        _save_state()
                        break

            # ── Monitor existing positions ────────────────────────────────
            with store["lock"]:
                pos_copy = dict(store["positions"])

            state_changed = False
            for sym, pos in pos_copy.items():
                ltp = get_ltp_yf(sym)
                if ltp <= 0:
                    ltp = pos["entry"]
                pnl_pct = (ltp - pos["entry"]) / pos["entry"]
                with store["lock"]:
                    if sym in store["positions"]:
                        store["positions"][sym]["ltp"]     = ltp
                        store["positions"][sym]["pnl_pct"] = pnl_pct

                if pos.get("sl") and ltp <= pos["sl"]:
                    _log(store, f"🔴 SL hit: {sym} @ ₹{ltp:.2f} | P&L: {pnl_pct*100:+.2f}%")
                    with store["lock"]:
                        store["daily_pnl"] += (ltp - pos["entry"]) * pos["qty"]
                        store["closed_trades"].append({
                            "symbol": sym, "exit_reason": "SL",
                            "entry": pos["entry"], "exit": round(ltp, 2),
                            "pnl_rs": round((ltp - pos["entry"]) * pos["qty"], 2),
                            "pnl_pct": round(pnl_pct * 100, 3),
                            "time": fmt_ist(),
                        })
                        store["positions"].pop(sym, None)
                    state_changed = True
                elif pos.get("target") and ltp >= pos["target"]:
                    _log(store, f"🟢 Target hit: {sym} @ ₹{ltp:.2f} | P&L: {pnl_pct*100:+.2f}%")
                    with store["lock"]:
                        store["daily_pnl"] += (ltp - pos["entry"]) * pos["qty"]
                        store["closed_trades"].append({
                            "symbol": sym, "exit_reason": "Target",
                            "entry": pos["entry"], "exit": round(ltp, 2),
                            "pnl_rs": round((ltp - pos["entry"]) * pos["qty"], 2),
                            "pnl_pct": round(pnl_pct * 100, 3),
                            "time": fmt_ist(),
                        })
                        store["positions"].pop(sym, None)
                    state_changed = True

            if state_changed:
                _save_state()

            with store["lock"]:
                unrealised = sum(
                    p.get("pnl_pct", 0) * p["entry"] * p["qty"]
                    for p in store["positions"].values()
                )
                store["portfolio_pnl"] = store["daily_pnl"] + unrealised

            interval = settings.get("scan_interval", 60)
            _log(store, f"💼 P&L ₹{store['portfolio_pnl']:+,.0f} | next scan in {interval}s")

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            with store["lock"]:
                store["last_error"] = f"{fmt_ist()} — {err_msg}"
            _log(store, f"❌ Thread error: {e}")

        time.sleep(settings.get("scan_interval", 60))

    _log(store, f"🔴 Live trading engine stopped at {fmt_ist()}")


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGES
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard(settings: dict):
    st.title("📊 Portfolio Dashboard")

    with _STORE["lock"]:
        positions   = dict(_STORE["positions"])
        daily_pnl   = _STORE["daily_pnl"]
        port_pnl    = _STORE["portfolio_pnl"]
        live_on     = _STORE["live_running"]
        cb          = _STORE["circuit_breaker"]
        closed      = list(_STORE["closed_trades"])

    cap = settings["capital"]

    # ── Status row ──────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Engine",         "🟢 LIVE" if live_on else "🔴 OFF")
    c2.metric("Daily P&L",      f"₹{daily_pnl:+,.0f}",
              delta=f"{daily_pnl/cap*100:+.2f}%")
    c3.metric("Open Positions",  len(positions))
    c4.metric("Closed Today",    len(closed))
    c5.metric("Circuit Breaker", "🔴 ACTIVE" if cb else "✅ OK")

    st.markdown("---")

    # ── Risk gauges ─────────────────────────────────────────────────────
    st.subheader("⚡ Risk Status")
    r1,r2,r3 = st.columns(3)
    daily_used = abs(daily_pnl/cap*100) if daily_pnl < 0 else 0
    r1.metric("Daily Loss Used",
              f"{daily_used:.1f}% / {settings['daily_loss_limit']*100:.1f}%",
              delta=f"-{daily_used:.1f}%" if daily_used > 0 else "0%",
              delta_color="inverse")
    deployed = sum(p["entry"]*p["qty"] for p in positions.values())
    r2.metric("Capital Deployed",
              f"₹{deployed:,.0f}",
              delta=f"{deployed/cap*100:.1f}%")
    r3.metric("Positions",
              f"{len(positions)} / {settings['max_positions']}")

    # ── Open positions ───────────────────────────────────────────────────
    if positions:
        st.markdown("---")
        st.subheader("📋 Open Positions")
        rows = []
        for sym, p in positions.items():
            ltp    = p.get("ltp", p["entry"])
            pnl_rs = (ltp - p["entry"]) * p["qty"]
            rows.append({
                "Symbol":   sym,
                "Strategy": p.get("strategy", "—"),
                "Entry ₹":  f"₹{p['entry']:,.2f}",
                "LTP ₹":    f"₹{ltp:,.2f}",
                "SL ₹":     f"₹{p['sl']:,.2f}",
                "Target ₹": f"₹{p['target']:,.2f}",
                "Qty":      p["qty"],
                "P&L ₹":    f"₹{pnl_rs:+,.0f}",
                "P&L %":    f"{p.get('pnl_pct',0)*100:+.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # P&L bar chart
        syms = list(positions.keys())
        pnls = [(positions[s].get("ltp", positions[s]["entry"]) - positions[s]["entry"])
                * positions[s]["qty"] for s in syms]
        colors = ["#00c853" if x >= 0 else "#ff1744" for x in pnls]
        fig = go.Figure(go.Bar(x=syms, y=pnls, marker_color=colors, name="P&L"))
        fig.update_layout(title="Open Position P&L (₹)", template="plotly_dark",
                          height=260, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Closed trades ────────────────────────────────────────────────────
    if closed:
        st.markdown("---")
        st.subheader("✅ Closed Trades Today")
        st.dataframe(pd.DataFrame(closed), use_container_width=True, hide_index=True)

    # ── Market pulse (top 10 Nifty50) ───────────────────────────────────
    st.markdown("---")
    st.subheader("🌐 Nifty 50 — Market Pulse")
    with st.spinner("Loading…"):
        mkt = []
        for sym in NIFTY_50[:12]:
            try:
                df = fetch_daily(sym + ".NS", period="5d")
                if not df.empty and len(df) >= 2:
                    chg = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
                    df2 = add_indicators(df)
                    df2 = detect_regime(df2)
                    reg = df2["regime"].iloc[-1]
                    mkt.append({
                        "Symbol":  sym,
                        "Price ₹": f"₹{df['Close'].iloc[-1]:,.2f}",
                        "Chg %":   f"{chg:+.2f}%",
                        "ADX":     f"{df2['adx14'].iloc[-1]:.0f}",
                        "RSI":     f"{df2['rsi14'].iloc[-1]:.0f}",
                        "Regime":  REGIME_EMOJI.get(reg,"⚪") + " " + reg.replace("_"," ").title(),
                        "Vol×":    f"{df2['vol_ratio'].iloc[-1]:.1f}×",
                    })
            except Exception:
                pass
        if mkt:
            st.dataframe(pd.DataFrame(mkt), use_container_width=True, hide_index=True)


def page_screener(settings: dict):
    st.title("🔍 Daily Stock Screener")
    st.caption("Scans Nifty 50 for regime-filtered, risk-sized opportunities")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_btn = st.button("🔄 Scan Now", type="primary", use_container_width=True)
    with col2:
        n_scan = st.selectbox("Stocks to scan", [10, 15, 20, 30, 50], index=1,
                              label_visibility="collapsed")
    with col3:
        extra_raw = st.text_input("➕ Add custom tickers (comma-separated)",
                                  placeholder="KAYNES, ZOMATO, IRFC, TATATECH")
        extra_syms = [s.strip().upper() for s in extra_raw.split(",") if s.strip()]

    if run_btn:
        base_list = settings["watchlist"][:n_scan]
        # Merge custom tickers, deduplicate, preserve order
        # base_list = bare NSE symbols; extra_syms = full yfinance tickers
        watchlist = list(dict.fromkeys(base_list + extra_syms))
        with st.spinner(f"Scanning {len(watchlist)} ticker(s)…"):
            sigs = scan_signals(watchlist, settings["capital"], settings["risk_pct"])
        with _STORE["lock"]:
            _STORE["signals"] = sigs
        if sigs:
            st.success(f"✅ {len(sigs)} signal(s) found")
        else:
            st.info("No signals right now. Markets may be in transition.")

    with _STORE["lock"]:
        sigs = list(_STORE["signals"])

    if not sigs:
        st.info("Click **Scan Now** to find today's opportunities.")
        return

    # Summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Signals",    len(sigs))
    m2.metric("EMA Swing",        sum(1 for s in sigs if s["strategy"] == "EMA Swing"))
    m3.metric("Breakout",         sum(1 for s in sigs if s["strategy"] == "Momentum Breakout"))
    m4.metric("Mean Reversion",   sum(1 for s in sigs if s["strategy"] == "Mean Reversion"))

    st.markdown("---")

    for sig in sigs:
        rem = REGIME_EMOJI.get(sig["regime"], "⚪")
        with st.expander(
            f"{rem} **{sig['symbol']}** — {sig['strategy']} | "
            f"₹{sig['entry']} → ₹{sig['target']}  |  R:R {sig['rr']}:1  |  {sig['style']}",
            expanded=False
        ):
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Entry",         f"₹{sig['entry']:,.2f}")
            c2.metric("Stop Loss",     f"₹{sig['sl']:,.2f}",
                      delta=f"-₹{sig['entry']-sig['sl']:.2f}", delta_color="inverse")
            c3.metric("Target",        f"₹{sig['target']:,.2f}",
                      delta=f"+₹{sig['target']-sig['entry']:.2f}")
            c4.metric("Qty (1.5% risk)", sig["qty"])
            c5.metric("Max Risk",      f"₹{sig['max_risk']:,.0f}")

            ca,cb_,cc = st.columns(3)
            ca.info(f"📊 ADX: {sig['adx']} | RSI: {sig['rsi']} | Vol×: {sig['vol_ratio']}")
            cb_.info(f"🎯 Regime: {sig['regime'].replace('_',' ').title()}")
            cc.info(f"💰 Max Profit: ₹{sig['max_profit']:,.0f}")

            btn = st.button(f"➕ Paper-Trade {sig['symbol']}", key=f"pt_{sig['symbol']}_{sig['strategy']}")
            if btn:
                with _STORE["lock"]:
                    _STORE["positions"][sig["symbol"]] = {
                        "strategy":   sig["strategy"],
                        "entry":      sig["entry"],
                        "sl":         sig["sl"],
                        "target":     sig["target"],
                        "qty":        sig["qty"],
                        "ltp":        sig["entry"],
                        "pnl_pct":    0.0,
                        "entry_time": fmt_ist(include_date=False),
                    }
                st.success(f"Paper-position added for {sig['symbol']}")


def page_backtest(settings: dict):
    st.title("📈 Walk-Forward Backtest")
    st.caption("Pure OOS results only — no in-sample data is ever reported")

    # ── Timeframe & Period info banner ────────────────────────────────────────
    with st.expander("ℹ️ Timeframe, Period & Strategy Reference — click to expand", expanded=False):
        st.markdown("""
| Strategy | Timeframe | Data Period | Typical Trades/Year | Hold Period | Regime Required |
|---|---|---|---|---|---|
| **EMA Swing** | **Daily (1D)** | 5 years | 6–15 per stock | 3–10 days | ADX>22 + Trend |
| **Momentum Breakout** | **Daily (1D)** | 5 years | 8–20 per stock | 1–5 days | ADX>20 |
| **Mean Reversion** | **Daily (1D)** | 5 years | 10–25 per stock | 2–5 days | ADX<22 (ranging) |

**Why daily timeframe?**  Intraday (1m/5m) requires exchange tick data. Daily gives clean signals, lower costs (delivery STT), and avoids noise.

**Why 5 years?** Need enough regime cycles (bull + bear + ranging) to test strategy robustness.  
Minimum recommended: 3 years (750 bars). Less = not enough OOS windows.

**EMA Swing gives 0 trades?** — Normal. This strategy only fires when ADX>22 AND price above EMA50 AND +DI>-DI simultaneously. A stock can spend months ranging (ADX<20) with zero signals — **that's the regime filter protecting you**, not a bug.

**Other strategies give 0 trades?** — Also normal for the same reason. Mean Reversion only fires when ADX<22. In a strong trend both EMA Swing and Mean Reversion can go silent at the same time. **This is expected behaviour.**
        """)

    # ── Instrument selector ───────────────────────────────────────────────────
    inst_type = st.radio("Instrument Type", [
        "📈 NSE Stock (Nifty 50)", "🔢 Index / Crypto / Commodity / Forex", "✏️ Custom Ticker"
    ], horizontal=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        if inst_type == "📈 NSE Stock (Nifty 50)":
            raw_sym = st.selectbox("Stock (NSE)", NIFTY_50, index=0)
            ticker  = raw_sym + ".NS"
            sym     = raw_sym
        elif inst_type == "🔢 Index / Crypto / Commodity / Forex":
            label   = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
            ticker  = INSTRUMENTS[label]
            sym     = label
        else:
            raw_sym = st.text_input("Custom yfinance Ticker",
                                    placeholder="e.g. KAYNES.NS  |  BTC-USD  |  GC=F  |  ^NSEI",
                                    value="KAYNES.NS").strip()
            ticker  = raw_sym
            sym     = raw_sym

        st.caption(f"yfinance ticker: `{ticker}`")

    with c2:
        strat_name = st.selectbox("Strategy", [
            "EMA Swing", "Momentum Breakout", "Mean Reversion"
        ])
    with c3:
        bt_cap = st.number_input("Capital (₹)", value=int(settings["capital"]), step=10000)

    c4,c5 = st.columns(2)
    with c4:
        train_bars = st.slider("In-sample window (trading days)", 126, 504, 252, 63,
                               help="252 ≈ 1 year. Strategy fit here — NOT reported.")
    with c5:
        test_bars = st.slider("Out-of-sample window (trading days)", 21, 126, 63, 21,
                              help="63 ≈ 3 months. Only OOS results shown.")

    strat_map = {
        "EMA Swing":          strategy_ema_swing,
        "Momentum Breakout":  strategy_momentum_breakout,
        "Mean Reversion":     strategy_mean_reversion,
    }

    if st.button("▶ Run Walk-Forward Backtest", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {sym} ({ticker})…"):
            df = fetch_daily(ticker, period="5y")

        if df.empty:
            st.error(f"No data for `{ticker}`. Check the ticker and try again.")
            return
        if len(df) < train_bars + test_bars + 60:
            st.warning(f"Only {len(df)} bars available. Need ≥ {train_bars+test_bars+60}. "
                       f"Try reducing in-sample window or use a longer-listed instrument.")
            return

        with st.spinner("Running walk-forward…"):
            trades, windows = walk_forward_backtest(
                df, strat_map[strat_name],
                train_bars=train_bars, test_bars=test_bars
            )

        if trades.empty:
            st.warning(
                f"⚠️ No trades in OOS periods for **{sym}** with **{strat_name}**.  \n"
                f"This is **normal** — regime filter blocked all entries. "
                f"Try: Mean Reversion on a ranging stock, or a different instrument."
            )
            return

        m = compute_metrics(trades, bt_cap)

        # ── Metrics ─────────────────────────────────────────────────────
        st.markdown("### 📊 Out-of-Sample Performance")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Trades",        m["total_trades"])
        m2.metric("Win Rate",      f"{m['win_rate']*100:.1f}%")
        m3.metric("Profit Factor", f"{m['profit_factor']:.2f}")
        m4.metric("Sharpe",        f"{m['sharpe']:.2f}")
        m5.metric("Max DD",        f"{m['max_drawdown_pct']:.1f}%")
        m6.metric("Total Return",  f"{m['total_return_pct']:.1f}%")

        ma,mb,mc,md = st.columns(4)
        ma.metric("Avg Win",        f"{m['avg_win_pct']:.2f}%")
        mb.metric("Avg Loss",       f"{m['avg_loss_pct']:.2f}%")
        mc.metric("Expectancy/trade", f"{m['expectancy_pct']:.3f}%")
        md.metric("Max Consec Loss", m["max_consec_loss"])

        # ── Cost breakdown ───────────────────────────────────────────────
        with st.expander("💸 Transaction Cost Impact"):
            gross_ret = trades["pnl_gross"].sum() * 100
            net_ret   = trades["pnl_net"].sum() * 100
            cost_drag = gross_ret - net_ret
            ci1,ci2,ci3 = st.columns(3)
            ci1.metric("Gross Return (sum)",  f"{gross_ret:.2f}%")
            ci2.metric("Net Return (sum)",    f"{net_ret:.2f}%")
            ci3.metric("Cost Drag",           f"-{cost_drag:.2f}%",
                       delta=f"₹{cost_drag/100*bt_cap:,.0f}", delta_color="inverse")
            st.caption(
                f"Round-trip costs assumed: "
                f"Intraday {ROUND_TRIP_INTRADAY*100:.2f}% | "
                f"Delivery {ROUND_TRIP_DELIVERY*100:.2f}%"
            )

        st.markdown("---")

        # ── Equity curve ─────────────────────────────────────────────────
        eq = m["equity_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=eq, mode="lines", name="Equity",
            line=dict(color="#00e5ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.07)"
        ))
        fig.add_hline(y=bt_cap, line_dash="dot", line_color="#666",
                      annotation_text="Starting capital", annotation_position="bottom right")

        # Drawdown shading
        peak = np.maximum.accumulate(eq)
        dd   = [(e - p) / p for e,p in zip(eq,peak)]
        fig.add_trace(go.Scatter(
            y=[bt_cap + d * bt_cap for d in dd],
            mode="lines", name="Drawdown ref",
            line=dict(color="rgba(255,70,70,0.4)", width=1),
            showlegend=False
        ))
        fig.update_layout(
            title=f"{sym} — {strat_name} — Walk-Forward Equity (OOS only)",
            yaxis_title="Portfolio Value ₹", xaxis_title="Trade #",
            template="plotly_dark", height=360,
            margin=dict(l=0,r=0,t=50,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Trade distribution ───────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=trades["pnl_net"]*100, nbinsx=30, name="Net P&L %",
                marker_color=["#00c853" if x>0 else "#ff1744"
                              for x in trades["pnl_net"]]
            ))
            fig2.add_vline(x=0, line_dash="dash", line_color="white")
            fig2.update_layout(title="Trade Returns (net of costs)",
                               xaxis_title="P&L %", template="plotly_dark", height=300,
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            exit_counts = trades["exit_reason"].value_counts()
            fig3 = go.Figure(go.Pie(
                labels=exit_counts.index,
                values=exit_counts.values,
                hole=0.4,
                marker_colors=["#00c853","#ff1744","#ffd600"],
            ))
            fig3.update_layout(title="Exit Reasons", template="plotly_dark", height=300,
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig3, use_container_width=True)

        # ── Per-window breakdown ──────────────────────────────────────────
        st.markdown("### 🔄 Walk-Forward Windows")
        if not windows.empty:
            st.dataframe(windows, use_container_width=True, hide_index=True)

        # ── Full Trade History ────────────────────────────────────────────
        st.markdown("### 📋 Full Trade History")
        hist_cols = ["entry_date","exit_date","hold_days","strategy",
                     "entry","sl","target","exit",
                     "pnl_gross","pnl_net","exit_reason","trade_type","window"]
        # Only include columns that exist
        hist_cols = [c for c in hist_cols if c in trades.columns]
        hist = trades[hist_cols].copy()
        hist["pnl_gross"] = (hist["pnl_gross"] * 100).round(3)
        hist["pnl_net"]   = (hist["pnl_net"]   * 100).round(3)
        # Add ₹ P&L column
        hist["pnl_rs"]    = (hist["pnl_net"] / 100 * bt_cap).round(0).astype(int)
        hist["win"]       = hist["pnl_net"] > 0

        rename = {
            "entry_date":  "Entry Date",
            "exit_date":   "Exit Date",
            "hold_days":   "Hold Days",
            "strategy":    "Strategy",
            "entry":       "Entry ₹",
            "sl":          "Stop Loss ₹",
            "target":      "Target ₹",
            "exit":        "Exit ₹",
            "pnl_gross":   "Gross %",
            "pnl_net":     "Net %",
            "pnl_rs":      "P&L ₹",
            "exit_reason": "Exit Reason",
            "trade_type":  "Type",
            "win":         "Win",
            "window":      "WF Window",
        }
        hist = hist.rename(columns={k:v for k,v in rename.items() if k in hist.columns})

        # Colour winning rows green, losing rows red
        def colour_row(row):
            clr = "background-color: rgba(0,200,83,0.12)" if row.get("Win", False) \
                  else "background-color: rgba(255,23,68,0.10)"
            return [clr] * len(row)

        st.dataframe(
            hist.style.apply(colour_row, axis=1),
            use_container_width=True, hide_index=True, height=380
        )

        csv_data = hist.to_csv(index=False).encode()
        st.download_button("⬇️ Download Trade History CSV", csv_data,
                           file_name=f"{sym}_{strat_name}_trades.csv",
                           mime="text/csv")

        # ── Regime breakdown ─────────────────────────────────────────────
        st.markdown("### 🌡 Historical Regime Distribution")
        df_ind = add_indicators(df)
        df_reg = detect_regime(df_ind)
        regime_dist = df_reg["regime"].value_counts(normalize=True) * 100
        fig4 = go.Figure(go.Bar(
            x=regime_dist.index,
            y=regime_dist.values,
            marker_color=["#00c853","#ff1744","#ffd600","#ff9100","#90a4ae"],
        ))
        fig4.update_layout(
            title=f"{sym} — Historical Regime Distribution",
            yaxis_title="% of time", template="plotly_dark", height=280,
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── PORTFOLIO BACKTEST ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂 Portfolio Walk-Forward (All Selected Stocks)")
    st.caption(
        "Runs the chosen strategy across multiple stocks simultaneously. "
        "This is where real trade frequency comes from — single stocks rarely give enough trades."
    )

    port_stocks = st.multiselect(
        "Stocks to include", NIFTY_50,
        default=["HINDUNILVR","NESTLEIND","BRITANNIA","ASIANPAINT","TITAN",
                 "INFY","TCS","HCLTECH","WIPRO","TECHM"],
        key="port_stocks"
    )
    port_strat = st.selectbox("Strategy", list(strat_map.keys()),
                               key="port_strat", index=0)
    port_cap   = st.number_input("Portfolio Capital (₹)", 100_000, 50_000_000,
                                  int(settings["capital"]), 10_000, key="port_cap")

    if st.button("▶ Run Portfolio Backtest", type="secondary", use_container_width=True):
        if not port_stocks:
            st.warning("Select at least one stock.")
        else:
            all_port_trades = []
            prog = st.progress(0, text="Fetching & running…")
            for idx, psym in enumerate(port_stocks):
                try:
                    pdf = fetch_daily(psym + ".NS", period="5y")
                    if pdf.empty or len(pdf) < train_bars + test_bars + 60:
                        continue
                    pt, _ = walk_forward_backtest(
                        pdf, strat_map[port_strat],
                        train_bars=train_bars, test_bars=test_bars
                    )
                    if not pt.empty:
                        pt["symbol"] = psym
                        all_port_trades.append(pt)
                except Exception:
                    pass
                prog.progress((idx+1)/len(port_stocks),
                              text=f"Processed {psym} ({idx+1}/{len(port_stocks)})")

            prog.empty()

            if not all_port_trades:
                st.warning("No trades found across portfolio. Try different stocks/strategy.")
            else:
                port_df = pd.concat(all_port_trades, ignore_index=True)
                pm = compute_metrics(port_df, port_cap)

                st.success(
                    f"✅ Portfolio: **{pm['total_trades']} trades** across "
                    f"{len(port_stocks)} stocks — statistically meaningful!"
                )

                pm1,pm2,pm3,pm4,pm5,pm6 = st.columns(6)
                pm1.metric("Total Trades",    pm["total_trades"])
                pm2.metric("Win Rate",        f"{pm['win_rate']*100:.1f}%")
                pm3.metric("Profit Factor",   f"{pm['profit_factor']:.2f}")
                pm4.metric("Sharpe",          f"{pm['sharpe']:.2f}")
                pm5.metric("Max Drawdown",    f"{pm['max_drawdown_pct']:.1f}%")
                pm6.metric("Total Return",    f"{pm['total_return_pct']:.1f}%")

                pa,pb,pc,pd_ = st.columns(4)
                pa.metric("Avg Win",          f"{pm['avg_win_pct']:.2f}%")
                pb.metric("Avg Loss",         f"{pm['avg_loss_pct']:.2f}%")
                pc.metric("Expectancy/trade", f"{pm['expectancy_pct']:.3f}%")
                pd_.metric("Max Consec Loss", pm["max_consec_loss"])

                # Per-symbol breakdown
                sym_summary = []
                for s in port_df["symbol"].unique():
                    sdf = port_df[port_df["symbol"] == s]
                    sm  = compute_metrics(sdf, port_cap / len(port_stocks))
                    if sm:
                        sym_summary.append({
                            "Symbol":       s,
                            "Trades":       sm["total_trades"],
                            "Win %":        f"{sm['win_rate']*100:.0f}%",
                            "PF":           f"{sm['profit_factor']:.2f}",
                            "Return %":     f"{sm['total_return_pct']:.1f}%",
                            "MaxDD %":      f"{sm['max_drawdown_pct']:.1f}%",
                            "Expectancy":   f"{sm['expectancy_pct']:.3f}%",
                        })

                if sym_summary:
                    st.markdown("#### Per-Symbol Breakdown")
                    st.dataframe(
                        pd.DataFrame(sym_summary).sort_values("Return %", ascending=False),
                        use_container_width=True, hide_index=True
                    )

                # Portfolio equity curve (equal-weight allocation)
                port_eq = np.array(pm["equity_curve"])
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    y=port_eq, mode="lines", name="Portfolio Equity",
                    line=dict(color="#69ff47", width=2),
                    fill="tozeroy", fillcolor="rgba(105,255,71,0.07)"
                ))
                fig_p.add_hline(y=port_cap, line_dash="dot", line_color="#666",
                                annotation_text="Starting capital")
                fig_p.update_layout(
                    title=f"Portfolio Equity — {port_strat} across {len(port_stocks)} stocks",
                    yaxis_title="₹", xaxis_title="Trade #",
                    template="plotly_dark", height=340,
                    margin=dict(l=0,r=0,t=50,b=0)
                )
                st.plotly_chart(fig_p, use_container_width=True)


def page_live(settings: dict):
    st.title("🤖 Live Trading Engine")

    with _STORE["lock"]:
        live_on      = _STORE["live_running"] and _is_engine_alive()
        cb           = _STORE["circuit_breaker"]
        log          = list(_STORE["live_log"])
        sigs         = list(_STORE["signals"])
        last_scan    = _STORE.get("last_scan_time")
        last_error   = _STORE.get("last_error")
        ticker_health= dict(_STORE.get("ticker_status", {}))

    # Sync: if thread died unexpectedly, reset flag
    if _STORE["live_running"] and not _is_engine_alive():
        with _STORE["lock"]:
            _STORE["live_running"] = False
        live_on = False

    paper_mode  = settings.get("paper_mode", True)
    dhan_token  = settings.get("dhan_token", "")
    dhan_client = settings.get("dhan_client", "")

    # ── Standalone Live Ticker & Strategy Config ──────────────────────────────
    st.markdown("### ⚙️ Live Engine Configuration")
    st.caption("These settings are independent of the Screener / Backtest tabs.")

    lv1, lv2 = st.columns([3, 2])
    with lv1:
        live_nifty = st.multiselect(
            "Nifty 50 Stocks to scan",
            options=NIFTY_50,
            default=["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"],
            key="live_nifty_select",
            help="Select stocks from Nifty 50"
        )
        live_custom_raw = st.text_input(
            "Additional custom tickers (comma-separated, full yfinance format)",
            placeholder="KAYNES.NS, ZOMATO.NS, BTC-USD, GC=F, USDINR=X",
            key="live_custom_tickers"
        )
        live_custom = [t.strip() for t in live_custom_raw.split(",") if t.strip()]

        # Build final watchlist — stocks as bare symbols, custom as-is
        live_watchlist = list(dict.fromkeys(live_nifty + live_custom))

        st.info(f"**{len(live_watchlist)}** ticker(s) queued: "
                f"{', '.join(live_watchlist[:8])}{'…' if len(live_watchlist) > 8 else ''}")

    with lv2:
        live_strats = st.multiselect(
            "Active Strategies",
            ["EMA Swing", "Momentum Breakout", "Mean Reversion"],
            default=["EMA Swing", "Momentum Breakout", "Mean Reversion"],
            key="live_strats"
        )
        live_interval = st.select_slider(
            "Scan Interval (seconds)",
            options=[30, 60, 120, 300],
            value=settings.get("scan_interval", 60),
            key="live_interval"
        )
        st.caption(f"Mode: **{'📄 Paper' if paper_mode else '💰 LIVE'}**")
        st.caption(f"Market hours: 09:15–15:20 IST")

    st.markdown("---")

    # ── Engine controls ──────────────────────────────────────────────────────
    mode_badge = "📄 PAPER" if paper_mode else "💰 LIVE"
    sc, c1, c2, c3 = st.columns([3,1,1,1])
    with sc:
        if live_on:
            st.success(f"🟢 Engine RUNNING — {mode_badge}")
        elif cb:
            st.error("🔴 Circuit Breaker Active — daily loss limit hit")
        else:
            st.info(f"⚪ Engine stopped — {mode_badge} mode")

    with c1:
        if st.button("▶ Start", disabled=live_on, use_container_width=True, type="primary"):
            if not live_watchlist:
                st.error("Select at least one ticker above before starting.")
            elif not paper_mode and not dhan_token:
                st.error("Enter Dhan token or enable Paper mode")
            else:
                # ── Clean up stale JSON state file on every fresh start ──
                import os
                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)

                dhan = DhanAPI(dhan_client, dhan_token)
                live_settings = {
                    **settings,
                    "watchlist":     live_watchlist,
                    "live_strats":   live_strats,
                    "scan_interval": live_interval,
                }
                with _STORE["lock"]:
                    _STORE["live_running"]    = True
                    _STORE["circuit_breaker"] = False
                    _STORE["daily_pnl"]       = 0.0
                    _STORE["positions"]       = {}
                    _STORE["closed_trades"]   = []
                    _STORE["live_log"]        = []
                    _STORE["signals"]         = []
                t = threading.Thread(
                    target=live_loop, args=(dhan, live_settings, _STORE), daemon=True
                )
                t.start()
                with _STORE["lock"]:
                    _STORE["live_thread"] = t
                st.rerun()
    with c2:
        if st.button("⏹ Stop", disabled=not live_on, use_container_width=True):
            with _STORE["lock"]:
                _STORE["live_running"] = False
            st.rerun()
    with c3:
        if st.button("🔄 Reset CB", use_container_width=True):
            with _STORE["lock"]:
                _STORE["circuit_breaker"] = False
                _STORE["daily_pnl"]       = 0.0
            st.rerun()

    st.markdown("---")

    # ── Show last error prominently if any ───────────────────────────────────
    if last_error:
        with st.expander("❌ Last Thread Error — click to expand", expanded=True):
            st.code(last_error, language="text")

    # ── Live metrics ─────────────────────────────────────────────────────────
    with _STORE["lock"]:
        daily_pnl = _STORE["daily_pnl"]
        positions = dict(_STORE["positions"])
        closed    = list(_STORE["closed_trades"])

    lm1,lm2,lm3,lm4,lm5 = st.columns(5)
    lm1.metric("Daily P&L",     f"₹{daily_pnl:+,.0f}",
               delta=f"{daily_pnl/settings['capital']*100:+.2f}%")
    lm2.metric("Open Positions", len(positions))
    lm3.metric("Closed Today",   len(closed))
    lm4.metric("Mode",           "📄 Paper" if paper_mode else "💰 Live")
    lm5.metric("Last Scan",      last_scan or "—")

    # ── Ticker Health Dashboard ───────────────────────────────────────────────
    st.markdown("### 🩺 Ticker Health — Why No Signals?")
    st.caption("Shows raw indicator values and which condition is blocking each strategy signal.")

    if st.button("🔬 Check Ticker Health Now", key="health_check", use_container_width=False):
        with st.spinner(f"Fetching indicators for {len(live_watchlist)} ticker(s)…"):
            _, fresh_health = _scan_with_health(
                live_watchlist, settings["capital"], settings["risk_pct"]
            )
        with _STORE["lock"]:
            _STORE["ticker_status"] = fresh_health
        ticker_health = fresh_health

    if ticker_health:
        health_rows = []
        for sym, h in ticker_health.items():
            if "error" in h:
                health_rows.append({
                    "Ticker": sym, "Close": "—", "ADX": "—", "RSI": "—",
                    "Regime": "—", "Vol×": "—",
                    "EMA Swing": f"❌ {h['error'][:35]}",
                    "Breakout": "—", "Mean Rev": "—", "Signal?": "❌",
                })
            else:
                ss = h.get("strat_status", {})
                health_rows.append({
                    "Ticker":   sym,
                    "Close":    h.get("close","—"),
                    "ADX":      h.get("adx","—"),
                    "RSI":      h.get("rsi","—"),
                    "Regime":   REGIME_EMOJI.get(h.get("regime",""),"") + " " + h.get("regime","").replace("_"," "),
                    "Vol×":     h.get("vol_ratio","—"),
                    "EMA Swing": ss.get("EMA Swing","—"),
                    "Breakout":  ss.get("Breakout","—"),
                    "Mean Rev":  ss.get("Mean Reversion","—"),
                    "Signal?":   "✅" if h.get("has_signal") else "⛔",
                })
        st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True, height=320)

        sig_count = sum(1 for h in ticker_health.values() if h.get("has_signal"))
        if sig_count == 0:
            adx_vals = [h["adx"] for h in ticker_health.values() if isinstance(h.get("adx"),(int,float))]
            rsi_vals = [h["rsi"] for h in ticker_health.values() if isinstance(h.get("rsi"),(int,float))]
            avg_adx = sum(adx_vals)/len(adx_vals) if adx_vals else 0
            avg_rsi = sum(rsi_vals)/len(rsi_vals) if rsi_vals else 50
            if avg_adx > 22:
                advice = "Market is trending. EMA Swing / Breakout signals expected — wait for EMA9 to cross EMA21 or a 20-day high breakout with volume."
            elif avg_adx < 18:
                advice = "Market is ranging (ADX<18). Mean Reversion is your strategy — wait for RSI to drop below 35 and close below BB-Lower."
            else:
                advice = "Market in transition (ADX 18–22). Best to wait 1–2 days for regime to clarify before trading."
            st.info(f"**No signals right now — this is normal.**  \n"
                    f"Avg ADX: **{avg_adx:.1f}** | Avg RSI: **{avg_rsi:.1f}**  \n{advice}")
    else:
        st.info("Click **Check Ticker Health Now** to see real-time indicator values and why each ticker isn't generating a signal.")

    st.markdown("---")

    # ── Strategy Signal Status ────────────────────────────────────────────────
    st.markdown("### 📡 Active Signals")
    if sigs:
        df_sig = pd.DataFrame([{
            "Symbol":   s["symbol"], "Strategy": s["strategy"],
            "Entry ₹":  s["entry"],  "SL ₹": s["sl"], "Target ₹": s["target"],
            "Qty": s["qty"], "R:R": f"{s['rr']}:1",
            "Regime": REGIME_EMOJI.get(s["regime"],"") + " " + s["regime"],
            "Style": s["style"],
        } for s in sigs])
        st.dataframe(df_sig, use_container_width=True, hide_index=True)
    else:
        st.info("No active signals. Run ticker health check above to understand current market conditions.")

    st.markdown("---")

    # ── Open Positions — detailed tracker ────────────────────────────────────
    st.markdown("### 📋 Open Positions (Multi-Day Tracker)")
    st.caption("Positions persist across app restarts via `algo_positions.json`")

    if positions:
        now_date = now_ist().date()
        pos_rows = []
        for sym, pos in positions.items():
            ltp       = pos.get("ltp", pos["entry"])
            pnl_rs    = (ltp - pos["entry"]) * pos["qty"]
            pnl_pct   = (ltp - pos["entry"]) / pos["entry"] * 100
            entry_dt  = pos.get("entry_time", "—")
            # Days held: compute from entry_date if stored
            entry_date_str = pos.get("entry_date", "")
            days_held = "—"
            if entry_date_str:
                try:
                    ed = datetime.strptime(entry_date_str[:10], "%Y-%m-%d").date()
                    days_held = (now_date - ed).days
                except Exception:
                    pass
            sl_dist    = ((ltp - pos["sl"])   / ltp * 100) if pos.get("sl")     else "—"
            tgt_dist   = ((pos["target"] - ltp) / ltp * 100) if pos.get("target") else "—"
            pos_rows.append({
                "Symbol":       sym,
                "Strategy":     pos.get("strategy", "—"),
                "Entry Date":   entry_date_str[:10] if entry_date_str else entry_dt,
                "Days Held":    days_held,
                "Entry ₹":      f"₹{pos['entry']:,.2f}",
                "LTP ₹":        f"₹{ltp:,.2f}",
                "SL ₹":         f"₹{pos.get('sl',0):,.2f}",
                "Target ₹":     f"₹{pos.get('target',0):,.2f}",
                "Qty":           pos["qty"],
                "P&L ₹":        f"₹{pnl_rs:+,.0f}",
                "P&L %":        f"{pnl_pct:+.2f}%",
                "% to SL":      f"-{sl_dist:.1f}%" if isinstance(sl_dist, float) else sl_dist,
                "% to Target":  f"+{tgt_dist:.1f}%" if isinstance(tgt_dist, float) else tgt_dist,
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        # Individual squareoff buttons
        st.markdown("**Manual Squareoff:**")
        sq_cols = st.columns(min(len(positions), 6))
        for col, sym in zip(sq_cols, list(positions.keys())):
            with col:
                if st.button(f"❌ {sym}", key=f"sq_{sym}"):
                    with _STORE["lock"]:
                        pos = _STORE["positions"].get(sym, {})
                        if pos:
                            ltp = pos.get("ltp", pos["entry"])
                            _STORE["daily_pnl"] += (ltp - pos["entry"]) * pos["qty"]
                            _STORE["closed_trades"].append({
                                "symbol": sym, "exit_reason": "Manual",
                                "entry": pos["entry"], "exit": ltp,
                                "pnl_rs": (ltp - pos["entry"]) * pos["qty"],
                                "pnl_pct": pnl_pct,
                                "time": fmt_ist(),
                            })
                            _STORE["positions"].pop(sym, None)
                    _save_state()
                    st.rerun()

        if st.button("⚠️ Squareoff ALL", type="secondary"):
            with _STORE["lock"]:
                for sym, pos in list(_STORE["positions"].items()):
                    ltp = pos.get("ltp", pos["entry"])
                    _STORE["daily_pnl"] += (ltp - pos["entry"]) * pos["qty"]
                    _STORE["closed_trades"].append({
                        "symbol": sym, "exit_reason": "Manual Squareoff",
                        "entry": pos["entry"], "exit": ltp,
                        "pnl_rs": (ltp - pos["entry"]) * pos["qty"],
                        "pnl_pct": (ltp - pos["entry"]) / pos["entry"] * 100,
                        "time": fmt_ist(),
                    })
                _STORE["positions"].clear()
            _save_state()
            st.success("All positions squared off")
            st.rerun()
    else:
        st.info("No open positions. Positions persist to disk — they'll survive app restarts.")

    # ── Closed trades today ───────────────────────────────────────────────────
    if closed:
        st.markdown("---")
        st.markdown("### ✅ Closed Trades")
        df_closed = pd.DataFrame(closed)
        st.dataframe(df_closed, use_container_width=True, hide_index=True)
        tot = sum(t.get("pnl_rs", 0) for t in closed)
        st.metric("Closed P&L", f"₹{tot:+,.0f}")

    # ── Activity log ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Activity Log")
    if log:
        st.text_area("", value="\n".join(reversed(log[-50:])),
                     height=300, disabled=True)
    else:
        st.info("Start the engine to see live logs here.")

    # ── Persistence info ─────────────────────────────────────────────────────
    import os
    if os.path.exists(STATE_FILE):
        mtime = datetime.utcfromtimestamp(os.path.getmtime(STATE_FILE)) + IST_OFFSET
        st.caption(f"💾 State file: `{STATE_FILE}` — last saved {mtime.strftime('%d %b %Y %H:%M:%S IST')}")
    else:
        st.caption(f"💾 State file `{STATE_FILE}` will be created when first position is opened.")

    if live_on:
        time.sleep(2)
        st.rerun()


def page_learn():
    st.title("📚 Strategy & Risk Reference")

    st.markdown("""
## 🎯 The 3 Strategies at a Glance

| Strategy | Best Regime | Typical Win Rate | R:R Target | Hold Period |
|---|---|---|---|---|
| **EMA Swing** | Trending Up (ADX>25) | 40–50% | 3:1 | 3–10 days |
| **Momentum Breakout** | Trending (ADX>20) | 35–45% | 2.5:1 | 1–5 days |
| **Mean Reversion** | Ranging (ADX<20) | 55–65% | 1.2–1.8:1 | 2–5 days |

> A strategy with 40% win rate and 3:1 R:R has **positive expectancy**.
> You make money even if you're *wrong 60% of the time*.

---

## 📐 Position Sizing Formula (Fixed Fractional)

```
Quantity = (Capital × Risk%) ÷ (Entry Price − Stop Loss)

Example:
  Capital        = ₹2,00,000
  Risk per trade = 1.5%  →  ₹3,000
  Entry          = ₹800
  Stop Loss      = ₹776   →  ₹24 per share
  Quantity       = ₹3,000 ÷ ₹24 = 125 shares
  Max loss       = ₹3,000 (1.5% of capital)  ✅
```

---

## 💸 Realistic NSE Transaction Costs

| Cost Component | Intraday | Delivery/Swing |
|---|---|---|
| Brokerage | ₹20 flat | ₹20 flat |
| STT | 0.025% sell | 0.1% sell |
| Exchange + SEBI | 0.00335% | 0.00335% |
| Stamp duty | 0.003% buy | 0.015% buy |
| Slippage (mkt impact) | 0.03% | 0.03% |
| **Total Round Trip** | **~0.16%** | **~0.36%** |

> A swing trade needs to move **>0.36% net** just to break even.
> Always backtest with costs included.

---

## 🔄 Walk-Forward Backtest vs Regular Backtest

```
Regular backtest (BAD):
  Train on 2020–2024 → Test on 2020–2024 → OVERFITTED

Walk-forward (CORRECT):
  Window 1: Train 2020–2021 → Test Q1 2022  ← OOS
  Window 2: Train 2020–Q1/2022 → Test Q2 2022  ← OOS
  Window 3: Train 2020–Q2/2022 → Test Q3 2022  ← OOS
  ...
  Final: Combine all OOS periods → True performance estimate
```

---

## 🌡 Regime Detection Logic

```
ADX > 25 + Close > EMA50 + +DI > -DI  →  🟢 Trending Up
  → Use: EMA Swing, Momentum Breakout

ADX > 25 + Close < EMA50 + -DI > +DI  →  🔴 Trending Down
  → Use: Avoid longs. Short strategies only (if any).

ADX < 20                               →  🟡 Ranging
  → Use: Mean Reversion only

ATR% > 2.5× median                     →  🟠 Volatile
  → Action: Reduce position size by 50%, skip new entries.

Transition zone (20 < ADX < 25)        →  ⚪ Neutral
  → Action: Wait for confirmation.
```

---

## ⚡ Portfolio Risk Rules

```
Max risk per trade:     1–2% of capital
Max concurrent trades:  4–6 (sector-diversified)
Daily loss limit:       2.5% → stop trading for the day
Max capital deployed:   60% (keep 40% as buffer)
```

---

## 📊 Key Performance Metrics

| Metric | Poor | Acceptable | Good |
|---|---|---|---|
| Profit Factor | <1.0 | 1.0–1.5 | >1.5 |
| Sharpe Ratio | <0.5 | 0.5–1.0 | >1.0 |
| Win Rate | N/A | N/A | Depends on R:R |
| Max Drawdown | >30% | 15–30% | <15% |
| Expectancy | Negative | >0 | >0.3%/trade |

> **Win rate alone is meaningless.** A 30% win rate with 5:1 R:R
> beats a 70% win rate with 0.5:1 R:R every time.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Nifty 50 Algo Trader",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    div[data-testid="stExpander"] > div { border-left: 3px solid #1e88e5; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 Nifty 50 Algo Trader")
        st.caption("Walk-Forward | Regime | Live Trading")
        st.markdown("---")

        st.subheader("💰 Capital & Risk")
        capital       = st.number_input("Total Capital (₹)", 50_000, 50_000_000, 200_000, 10_000)
        risk_pct      = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.5, 0.25) / 100
        max_pos       = st.slider("Max Positions", 1, 8, 4)
        daily_limit   = st.slider("Daily Loss Limit (%)", 1.0, 5.0, 2.5, 0.5) / 100
        max_deploy    = st.slider("Max Deployment (%)", 30, 90, 60, 10) / 100

        st.markdown("---")
        st.subheader("🤖 Live Settings")
        paper_mode = st.toggle("📄 Paper Trading Mode", value=True,
                               help="ON = no real orders, full simulation. OFF = live Dhan orders.")
        if paper_mode:
            st.success("Paper mode: no real money at risk")
            dhan_client = "PAPER"
            dhan_token  = "PAPER"
            auto_trade  = False
        else:
            st.warning("⚠️ Real order mode — enter Dhan credentials")
            st.subheader("🔑 Dhan API")
            dhan_client = st.text_input("Client ID", value="1104779876")
            dhan_token  = st.text_input("Access Token", type="password",
                                         placeholder="Paste your Dhan token")
            auto_trade  = st.toggle("Auto Place Orders", value=False,
                                    help="Places real orders via Dhan API")

        scan_interval = st.select_slider("Scan Interval",
                                          options=[30, 60, 120, 300], value=60)

        st.markdown("---")
        st.subheader("📋 Watchlist")
        watchlist = st.multiselect("Stocks", NIFTY_50, default=NIFTY_50[:20])
        if not watchlist:
            watchlist = NIFTY_50[:20]

    settings = {
        "capital":          capital,
        "risk_pct":         risk_pct,
        "max_positions":    max_pos,
        "daily_loss_limit": daily_limit,
        "max_deployment":   max_deploy,
        "auto_trade":       auto_trade,
        "paper_mode":       paper_mode,
        "scan_interval":    scan_interval,
        "watchlist":        watchlist,
        "dhan_client":      dhan_client,
        "dhan_token":       dhan_token,
    }

    # ── Navigation ────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Dashboard",
        "🔍 Screener",
        "📈 Backtest",
        "🤖 Live Trading",
        "📚 Learn",
    ])

    with tabs[0]: page_dashboard(settings)
    with tabs[1]: page_screener(settings)
    with tabs[2]: page_backtest(settings)
    with tabs[3]: page_live(settings)
    with tabs[4]: page_learn()


if __name__ == "__main__":
    main()
