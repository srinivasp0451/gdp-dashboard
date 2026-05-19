"""
Trading Strategy Platform
=========================
Features:
  • Ratio Strategy (Niftybees/Goldbees) • EMA Crossover • SMC • Institutional Candle
  • Multiple SL/Target methods          • Backtesting   • Live Trading  • Trade History
  • Conservative backtest: entry N+1 open, SL via Low, TP via High
  • Live trading: SL/TP checked via LTP
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Strategy Platform",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace; }

[data-testid="stSidebar"] {
    background: #0a0d14 !important;
    border-right: 1px solid #1c2030;
}
[data-testid="stSidebar"] .stMarkdown h3 { color: #e2b86e; font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 2px; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label { font-size: 0.82rem; color: #8899aa; }

div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1c2434;
    border-radius: 8px;
    padding: 10px 14px;
}
div[data-testid="metric-container"] > label { font-size: 0.72rem !important; color: #667788 !important; letter-spacing: 0.05em; text-transform: uppercase; }
div[data-testid="metric-container"] > div { font-size: 1.15rem !important; font-weight: 600; color: #e8eaf0; }

.verdict-box {
    padding: 16px 24px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 18px;
    font-family: 'Inter', sans-serif;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0a0d14; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #667788;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #1c2434 !important;
    color: #e2b86e !important;
}
div[data-testid="stDataFrame"] { border: 1px solid #1c2434; border-radius: 8px; }
.stDownloadButton > button {
    background: #1c2434;
    border: 1px solid #2a3448;
    color: #e2b86e;
    border-radius: 6px;
    font-size: 0.82rem;
}
.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e2b86e;
    padding: 12px 0 6px;
    border-bottom: 1px solid #1c2434;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
TICKERS: Dict[str, str] = {
    "Niftybees":  "NIFTYBEES.NS",
    "Goldbees":   "GOLDBEES.NS",
    "Nifty 50":   "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex":     "^BSESN",
    "BTC/USD":    "BTC-USD",
    "ETH/USD":    "ETH-USD",
    "USD/INR":    "USDINR=X",
    "Gold":       "GC=F",
    "Silver":     "SI=F",
    "Crude Oil":  "CL=F",
    "S&P 500":    "^GSPC",
    "Nasdaq":     "^IXIC",
    "Custom":     "__custom__",
}

TIMEFRAME_PERIODS: Dict[str, list] = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d"],
    "15m": ["1d", "5d", "7d"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo"],
    "1d":  ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

SL_METHODS = [
    "Strategy Based",
    "Custom Points",
    "Volatility Based (ATR)",
    "Trailing – Current Candle High/Low",
    "Trailing – Previous Candle High/Low",
    "Trailing – Swing High/Low",
    "Trailing SL (Fixed Points)",
    "Trailing Target (Fixed Points)",
    "Shift SL to C2C after X% Target",
    "Partial Exit at TP1 then Trail",
    "Logical SL/Target",
    "Trading Logical SL/Target",
    "Liquidity Based SL/Target",
    "Volatility Based SL/Target",
]

# ─────────────────────────────────────────────────────────────
# DHAN BROKER – INSTRUMENT MASTER
# Security IDs from Dhan's instrument master (NSE ETFs / Indices).
# Verify / update at: https://images.dhan.co/api-data/api-scrip-master.csv
# ─────────────────────────────────────────────────────────────
DHAN_INSTRUMENTS: Dict[str, dict] = {
    "Niftybees":  {"security_id": "13303", "exchange": "NSE_EQ",  "tradable": True},
    "Goldbees":   {"security_id": "1344",  "exchange": "NSE_EQ",  "tradable": True},
    "Nifty 50":   {"security_id": "13",    "exchange": "IDX_I",   "tradable": False,
                   "note": "Index – trade via futures/ETF"},
    "Bank Nifty": {"security_id": "25",    "exchange": "IDX_I",   "tradable": False,
                   "note": "Index – trade via futures/ETF"},
    "Sensex":     {"security_id": "1",     "exchange": "BSE_EQ",  "tradable": False,
                   "note": "Index – trade via futures/ETF"},
    "BTC/USD":    {"security_id": None,    "exchange": None,      "tradable": False,
                   "note": "Not available on Dhan"},
    "ETH/USD":    {"security_id": None,    "exchange": None,      "tradable": False,
                   "note": "Not available on Dhan"},
    "Gold":       {"security_id": "624",   "exchange": "MCX_COMM","tradable": True},
    "Silver":     {"security_id": "625",   "exchange": "MCX_COMM","tradable": True},
    "Crude Oil":  {"security_id": "626",   "exchange": "MCX_COMM","tradable": True},
}


# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=180, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def fetch_live(symbol: str, interval: str = "1m") -> Optional[pd.DataFrame]:
    """No-cache fetch for live mode."""
    try:
        df = yf.download(symbol, interval=interval, period="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# DHAN BROKER HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_dhan_client(client_id: str, access_token: str):
    """
    Initialise dhanhq client. Cached as a resource so the
    same connection object is reused across reruns.
    Returns (client, error_string | None).
    """
    try:
        from dhanhq import dhanhq          # pip install dhanhq
        client = dhanhq(client_id, access_token)
        return client, None
    except ImportError:
        return None, "dhanhq not installed — run:  pip install dhanhq"
    except Exception as e:
        return None, f"Dhan connection error: {e}"


def dhan_place_order(client,
                     ticker_label: str,
                     action: str,           # "BUY" | "SELL"
                     qty: int,
                     product_type: str,     # "INTRADAY" | "CNC"
                     order_type: str = "MARKET",
                     limit_price: float = 0.0) -> dict:
    """
    Place an order on Dhan.
    Returns the raw response dict from dhanhq.
    On any error returns {"status": "failure", "remarks": <msg>}.
    """
    inst = DHAN_INSTRUMENTS.get(ticker_label)
    if inst is None:
        return {"status": "failure",
                "remarks": f"'{ticker_label}' not in DHAN_INSTRUMENTS mapping."}
    if not inst.get("tradable"):
        note = inst.get("note", "Not tradable on Dhan.")
        return {"status": "failure", "remarks": note}

    try:
        resp = client.place_order(
            security_id   = inst["security_id"],
            exchange_segment = inst["exchange"],
            transaction_type = action,           # "BUY" or "SELL"
            quantity      = qty,
            order_type    = order_type,          # "MARKET" / "LIMIT"
            product_type  = product_type,        # "INTRADAY" / "CNC"
            price         = limit_price,
        )
        return resp if isinstance(resp, dict) else {"status": "success", "data": resp}
    except Exception as e:
        return {"status": "failure", "remarks": str(e)}


def dhan_square_off_all(client, product_type: str) -> list:
    """
    Fetch open positions from Dhan and place SELL orders for all longs.
    Returns list of order responses.
    """
    results = []
    try:
        pos_resp = client.get_positions()
        positions = pos_resp.get("data", []) if isinstance(pos_resp, dict) else []
        for p in positions:
            try:
                net_qty = int(p.get("netQty", 0))
                if net_qty <= 0:
                    continue
                sec_id   = str(p.get("securityId", ""))
                exchange = str(p.get("exchangeSegment", "NSE_EQ"))
                resp = client.place_order(
                    security_id      = sec_id,
                    exchange_segment = exchange,
                    transaction_type = "SELL",
                    quantity         = net_qty,
                    order_type       = "MARKET",
                    product_type     = product_type,
                    price            = 0.0,
                )
                results.append({"security": sec_id, "qty": net_qty, "response": resp})
            except Exception as e:
                results.append({"security": p.get("tradingSymbol", "?"),
                                 "error": str(e)})
    except Exception as e:
        results.append({"error": f"Could not fetch positions: {e}"})
    return results


def dhan_get_positions(client) -> pd.DataFrame:
    """Return open positions as a DataFrame."""
    try:
        resp = client.get_positions()
        data = resp.get("data", []) if isinstance(resp, dict) else []
        if data:
            df = pd.DataFrame(data)
            keep = [c for c in ["tradingSymbol", "netQty", "buyAvgPrice",
                                  "sellAvgPrice", "unrealizedProfit",
                                  "realizedProfit", "exchangeSegment"]
                    if c in df.columns]
            return df[keep] if keep else df
    except Exception:
        pass
    return pd.DataFrame()


def dhan_get_orders(client) -> pd.DataFrame:
    """Return today's order book as a DataFrame."""
    try:
        resp = client.get_order_list()
        data = resp.get("data", []) if isinstance(resp, dict) else []
        if data:
            df = pd.DataFrame(data)
            keep = [c for c in ["orderId", "tradingSymbol", "transactionType",
                                  "quantity", "price", "orderStatus",
                                  "createTime", "exchangeOrderId"]
                    if c in df.columns]
            return df[keep] if keep else df
    except Exception:
        pass
    return pd.DataFrame()


def dhan_get_funds(client) -> dict:
    """Return available cash / margin dict."""
    try:
        resp = client.get_fund_limits()
        return resp.get("data", {}) if isinstance(resp, dict) else {}
    except Exception:
        return {}
# ─────────────────────────────────────────────────────────────
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def detect_swing_lows(df: pd.DataFrame, n: int = 3) -> pd.Series:
    w = 2 * n + 1
    return df["Low"].rolling(w, center=True).min() == df["Low"]


def detect_swing_highs(df: pd.DataFrame, n: int = 3) -> pd.Series:
    w = 2 * n + 1
    return df["High"].rolling(w, center=True).max() == df["High"]


# ─────────────────────────────────────────────────────────────
# STRATEGY SIGNAL GENERATORS
# ─────────────────────────────────────────────────────────────
def sig_ratio_strategy(df1: pd.DataFrame, df2: pd.DataFrame,
                       lookback: int = 20) -> pd.DataFrame:
    """
    Ratio = Ticker1.Close / Ticker2.Close  (daily)
    20-day rolling highest-high / lowest-low (shift(1) → no lookahead)
    RATIO BREAKS UP   (ratio > prior 20d high) → Buy Ticker1 (signal= +1)
    RATIO BREAKS DOWN (ratio < prior 20d low)  → Buy Ticker2 (signal= -1)
    Always invested.
    """
    ratio = df1["Close"] / df2["Close"]
    prev_high = ratio.shift(1).rolling(lookback).max()
    prev_low  = ratio.shift(1).rolling(lookback).min()
    sig = pd.Series(0, index=df1.index)
    sig[ratio > prev_high] =  1
    sig[ratio < prev_low]  = -1
    return pd.DataFrame({"signal": sig, "ratio": ratio,
                          "ratio_high": prev_high, "ratio_low": prev_low})


def sig_ema_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 15) -> pd.DataFrame:
    fe = calc_ema(df["Close"], fast)
    se = calc_ema(df["Close"], slow)
    sig = pd.Series(0, index=df.index)
    cross_up   = (fe > se) & (fe.shift(1) <= se.shift(1))
    cross_down = (fe < se) & (fe.shift(1) >= se.shift(1))
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return pd.DataFrame({"signal": sig, "fast_ema": fe, "slow_ema": se})


def sig_simple_buy(df: pd.DataFrame) -> pd.DataFrame:
    sig = pd.Series(0, index=df.index)
    if len(sig): sig.iloc[0] = 1
    return pd.DataFrame({"signal": sig})


def sig_simple_sell(df: pd.DataFrame) -> pd.DataFrame:
    sig = pd.Series(0, index=df.index)
    if len(sig): sig.iloc[0] = -1
    return pd.DataFrame({"signal": sig})


def sig_institutional_candle(df: pd.DataFrame, body_thresh: float = 0.7) -> pd.DataFrame:
    """Big-body candle (≥ body_thresh of range) → entry on breakout of candle hi/lo."""
    body  = (df["Close"] - df["Open"]).abs()
    rng   = (df["High"] - df["Low"]).replace(0, np.nan)
    is_ic = body / rng >= body_thresh
    bull  = is_ic & (df["Close"] > df["Open"])
    bear  = is_ic & (df["Close"] < df["Open"])
    sig   = pd.Series(0, index=df.index)
    sig[bull.shift(1).fillna(False) & (df["Open"] > df["High"].shift(1))] =  1
    sig[bear.shift(1).fillna(False) & (df["Open"] < df["Low"].shift(1))]  = -1
    return pd.DataFrame({"signal": sig})


def sig_smc(df: pd.DataFrame, swing_n: int = 5) -> pd.DataFrame:
    """
    SMC – Break of Structure (BOS).
    Long : close > last confirmed swing high  → BOS up
    Short: close < last confirmed swing low   → BOS down
    """
    sh_mask = detect_swing_highs(df, swing_n)
    sl_mask = detect_swing_lows(df, swing_n)
    last_sh = df["High"].where(sh_mask).ffill().shift(1)
    last_sl = df["Low"].where(sl_mask).ffill().shift(1)
    sig = pd.Series(0, index=df.index)
    bos_up   = (df["Close"] > last_sh) & (df["Close"].shift(1) <= last_sh.shift(1))
    bos_down = (df["Close"] < last_sl) & (df["Close"].shift(1) >= last_sl.shift(1))
    sig[bos_up]   =  1
    sig[bos_down] = -1
    return pd.DataFrame({"signal": sig, "swing_high": last_sh, "swing_low": last_sl})


def generate_signals(df: pd.DataFrame, strategy: str,
                     df2: Optional[pd.DataFrame] = None, **kw) -> pd.DataFrame:
    if strategy == "Ratio Strategy" and df2 is not None:
        return sig_ratio_strategy(df, df2, kw.get("lookback", 20))
    if strategy == "EMA Crossover":
        return sig_ema_crossover(df, kw.get("fast", 9), kw.get("slow", 15))
    if strategy == "Simple Buy":
        return sig_simple_buy(df)
    if strategy == "Simple Sell":
        return sig_simple_sell(df)
    if strategy == "Institutional Candle Entry":
        return sig_institutional_candle(df)
    if strategy == "SMC (Smart Money Concepts)":
        return sig_smc(df)
    return pd.DataFrame({"signal": pd.Series(0, index=df.index)})


# ─────────────────────────────────────────────────────────────
# SL / TARGET CALCULATION
# ─────────────────────────────────────────────────────────────
def calc_sl_tp(df: pd.DataFrame, idx: int, entry: float,
               direction: int, method: str, **p) -> Tuple[float, float, float, float]:
    """
    Returns (sl, tp1, tp2, tp3).
    direction: +1 long, -1 short.
    """
    safe_idx = min(max(0, idx), len(df) - 1)
    atr_s = calc_atr(df, 14)
    atr_v = float(atr_s.iloc[safe_idx])
    if np.isnan(atr_v) or atr_v <= 0:
        atr_v = float((df["High"].iloc[safe_idx] - df["Low"].iloc[safe_idx])) or entry * 0.005

    pts   = float(p.get("custom_pts",  10.0))
    atr_m = float(p.get("atr_mult",    1.5))

    def rr(sl_dist: float, m1: float = 1.5, m2: float = 2.5, m3: float = 3.5):
        if direction == 1:
            return (entry - sl_dist,
                    entry + sl_dist * m1,
                    entry + sl_dist * m2,
                    entry + sl_dist * m3)
        return (entry + sl_dist,
                entry - sl_dist * m1,
                entry - sl_dist * m2,
                entry - sl_dist * m3)

    if method in ("Strategy Based", "Volatility Based (ATR)", "Volatility Based SL/Target"):
        return rr(atr_v * atr_m)

    if method == "Custom Points":
        return rr(pts)

    if method in ("Trailing – Current Candle High/Low",
                  "Trailing – Previous Candle High/Low",
                  "Trailing – Swing High/Low",
                  "Trailing SL (Fixed Points)",
                  "Trailing Target (Fixed Points)",
                  "Shift SL to C2C after X% Target",
                  "Partial Exit at TP1 then Trail"):
        return rr(atr_v * atr_m)   # initial; trailing managed in loop

    if method == "Logical SL/Target":
        win = df.iloc[max(0, safe_idx - 10): safe_idx + 1]
        if direction == 1:
            sl_dist = max(entry - win["Low"].min() * 0.999, atr_v * 0.5)
        else:
            sl_dist = max(win["High"].max() * 1.001 - entry, atr_v * 0.5)
        return rr(sl_dist)

    if method == "Trading Logical SL/Target":
        return rr(atr_v * 1.5, 1.0, 1.5, 2.5)

    if method == "Liquidity Based SL/Target":
        win = df.iloc[max(0, safe_idx - 20): safe_idx + 1]
        if direction == 1:
            sl   = win["Low"].quantile(0.10)
            tp1  = win["High"].quantile(0.50)
            tp2  = win["High"].quantile(0.80)
            tp3  = win["High"].max()
        else:
            sl   = win["High"].quantile(0.90)
            tp1  = win["Low"].quantile(0.50)
            tp2  = win["Low"].quantile(0.20)
            tp3  = win["Low"].min()
        return sl, tp1, tp2, tp3

    return rr(atr_v * atr_m)


# ─────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, signals: pd.DataFrame,
                 sl_method: str, initial_capital: float,
                 df2: Optional[pd.DataFrame] = None, **p) -> Tuple[pd.DataFrame, float]:
    """
    Conservative Backtest Rules
    ───────────────────────────
    • Signal on candle N  →  entry at candle N+1 OPEN
    • LONG  : check SL via candle Low first  → then check TP via candle High
    • SHORT : check SL via candle High first → then check TP via candle Low
    • Live  : SL/TP checked via LTP (handled in live panel)
    """
    trades:  list = []
    capital: float = float(initial_capital)
    pos: Optional[dict] = None          # active position

    sig_arr  = signals["signal"].values
    opens    = df["Open"].values
    highs    = df["High"].values
    lows     = df["Low"].values
    closes   = df["Close"].values
    dates    = df.index

    trail_pts    = float(p.get("trailing_pts", 10.0))
    shift_pct    = float(p.get("shift_pct",    30.0)) / 100.0
    exit_qty_pct = float(p.get("exit_pct",     70.0)) / 100.0
    n = len(df)

    def close_pos(px: float, dt, reason: str, qty: Optional[int] = None):
        nonlocal capital, pos
        q   = qty if qty is not None else pos["qty"]
        pnl = (px - pos["entry"]) * pos["dir"] * q
        capital += pnl
        trades.append({
            "Entry Date":  pos["entry_dt"],
            "Exit Date":   dt,
            "Direction":   "Long" if pos["dir"] == 1 else "Short",
            "Entry Price": round(float(pos["entry"]), 4),
            "Exit Price":  round(float(px), 4),
            "SL":          round(float(pos["sl"]), 4),
            "TP1":         round(float(pos["tp1"]), 4),
            "TP2":         round(float(pos["tp2"]), 4),
            "TP3":         round(float(pos["tp3"]), 4),
            "Qty":         q,
            "PnL":         round(float(pnl), 2),
            "Exit Reason": reason,
            "Capital":     round(float(capital), 2),
        })
        if qty is None:
            pos = None

    i = 0
    while i < n:
        sig_val = int(sig_arr[i]) if not np.isnan(sig_arr[i]) else 0

        # ── Signal → open/switch position at next candle's open ──
        if sig_val != 0 and i + 1 < n:
            new_dir = sig_val
            if pos is not None and pos["dir"] == new_dir:
                i += 1
                continue
            if pos is not None:
                close_pos(float(opens[i + 1]), dates[i + 1], "Signal Reversal")
            entry_px = float(opens[i + 1])
            sl, tp1, tp2, tp3 = calc_sl_tp(df, i + 1, entry_px, new_dir, sl_method, **p)
            qty = max(1, int(capital * 0.95 / entry_px))
            pos = {
                "entry":    entry_px, "entry_dt": dates[i + 1],
                "dir":      new_dir,
                "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "qty": qty, "partial_done": False, "sl_shifted": False,
            }
            i += 1
            continue   # next iter manages this position on candle i+1

        # ── Manage open position ─────────────────────────────────
        if pos is not None:
            d  = pos["dir"]
            sl = pos["sl"]

            # ── Update trailing SL ──
            if sl_method == "Trailing – Current Candle High/Low":
                cand = float(lows[i]) if d == 1 else float(highs[i])
                pos["sl"] = max(sl, cand) if d == 1 else min(sl, cand)
                sl = pos["sl"]

            elif sl_method == "Trailing – Previous Candle High/Low" and i > 0:
                cand = float(lows[i - 1]) if d == 1 else float(highs[i - 1])
                pos["sl"] = max(sl, cand) if d == 1 else min(sl, cand)
                sl = pos["sl"]

            elif sl_method == "Trailing SL (Fixed Points)":
                cand = float(closes[i]) - trail_pts if d == 1 else float(closes[i]) + trail_pts
                pos["sl"] = max(sl, cand) if d == 1 else min(sl, cand)
                sl = pos["sl"]

            elif sl_method == "Trailing Target (Fixed Points)":
                # Trail TP1 as price advances
                new_tp1 = float(closes[i]) + trail_pts if d == 1 else float(closes[i]) - trail_pts
                pos["tp1"] = max(pos["tp1"], new_tp1) if d == 1 else min(pos["tp1"], new_tp1)

            elif sl_method == "Trailing – Swing High/Low" and i >= 6:
                sub = df.iloc[:i + 1]
                if d == 1:
                    sw_lows  = sub["Low"].where(detect_swing_lows(sub)).dropna()
                    if len(sw_lows):
                        new_sl = float(sw_lows.iloc[-1])
                        pos["sl"] = max(sl, new_sl)
                        sl = pos["sl"]
                else:
                    sw_highs = sub["High"].where(detect_swing_highs(sub)).dropna()
                    if len(sw_highs):
                        new_sl = float(sw_highs.iloc[-1])
                        pos["sl"] = min(sl, new_sl)
                        sl = pos["sl"]

            elif sl_method == "Shift SL to C2C after X% Target":
                risk = abs(pos["entry"] - sl)
                if risk > 0:
                    thresh = pos["entry"] + d * risk * (shift_pct / (1 - shift_pct + 1e-9))
                    cond_long  = d ==  1 and float(highs[i]) >= thresh
                    cond_short = d == -1 and float(lows[i])  <= thresh
                    if cond_long or cond_short:
                        if not pos["sl_shifted"]:
                            pos["sl"]       = pos["entry"]   # move to breakeven (C2C)
                            pos["sl_shifted"] = True
                        else:
                            step = risk * float(p.get("trail_pct", 30.0)) / 100.0
                            cand = pos["sl"] + d * step
                            pos["sl"] = max(pos["sl"], cand) if d == 1 else min(pos["sl"], cand)
                        sl = pos["sl"]

            # ── Conservative SL / TP check ──────────────────────
            exit_px     = None
            exit_reason = None

            if d == 1:   # LONG: check SL via Low first, TP via High
                if float(lows[i]) <= sl:
                    exit_px     = sl
                    exit_reason = "SL Hit"
                elif float(highs[i]) >= pos["tp1"]:
                    if sl_method == "Partial Exit at TP1 then Trail" and not pos["partial_done"]:
                        partial_q = int(pos["qty"] * exit_qty_pct)
                        if partial_q > 0:
                            close_pos(pos["tp1"], dates[i], "TP1 (Partial)", qty=partial_q)
                            if pos:
                                pos["qty"]         -= partial_q
                                pos["partial_done"] = True
                                pos["sl"]           = pos["entry"]  # move to breakeven
                                sl                  = pos["sl"]
                    else:
                        exit_px     = pos["tp1"]
                        exit_reason = "TP1 Hit"
                    if pos and float(highs[i]) >= pos["tp2"]:
                        exit_px     = pos["tp2"]
                        exit_reason = "TP2 Hit"
                    if pos and float(highs[i]) >= pos["tp3"]:
                        exit_px     = pos["tp3"]
                        exit_reason = "TP3 Hit"

            else:        # SHORT: check SL via High first, TP via Low
                if float(highs[i]) >= sl:
                    exit_px     = sl
                    exit_reason = "SL Hit"
                elif float(lows[i]) <= pos["tp1"]:
                    if sl_method == "Partial Exit at TP1 then Trail" and not pos["partial_done"]:
                        partial_q = int(pos["qty"] * exit_qty_pct)
                        if partial_q > 0:
                            close_pos(pos["tp1"], dates[i], "TP1 (Partial)", qty=partial_q)
                            if pos:
                                pos["qty"]         -= partial_q
                                pos["partial_done"] = True
                                pos["sl"]           = pos["entry"]
                                sl                  = pos["sl"]
                    else:
                        exit_px     = pos["tp1"]
                        exit_reason = "TP1 Hit"
                    if pos and float(lows[i]) <= pos["tp2"]:
                        exit_px     = pos["tp2"]
                        exit_reason = "TP2 Hit"
                    if pos and float(lows[i]) <= pos["tp3"]:
                        exit_px     = pos["tp3"]
                        exit_reason = "TP3 Hit"

            if exit_px is not None and pos is not None:
                close_pos(exit_px, dates[i], exit_reason)

        i += 1

    # Close any remaining open position at last price
    if pos is not None:
        close_pos(float(closes[-1]), dates[-1], "End of Data")

    return pd.DataFrame(trades), capital


def run_ratio_backtest(df1: pd.DataFrame, df2: pd.DataFrame,
                       signals: pd.DataFrame,
                       initial_capital: float) -> Tuple[pd.DataFrame, float]:
    """
    Ratio Strategy – switching long-only between two assets.
    ─────────────────────────────────────────────────────────
    • signal = +1 → ratio broke above 20d high:
        ✔ Buy  Niftybees  at next candle N+1 open
        ✖ Square off Goldbees (if held) at the same open
    • signal = -1 → ratio broke below 20d low:
        ✔ Buy  Goldbees   at next candle N+1 open
        ✖ Square off Niftybees (if held) at the same open
    • Never short either asset.
    • Entry on N+1 open after signal fires on N (no lookahead).
    """
    trades:  list  = []
    capital: float = float(initial_capital)
    pos: Optional[dict] = None   # {'asset': 1|-1, 'ticker': str, 'entry': float,
                                  #  'entry_dt': ts, 'qty': int}

    sig_arr = signals["signal"].values
    dates   = df1.index
    n       = len(df1)

    opens1 = df1["Open"].values
    closes1= df1["Close"].values
    opens2 = df2["Open"].values
    closes2= df2["Close"].values

    def switch(new_asset: int, idx: int):
        """Close current position (if any) and open new_asset position at opens[idx]."""
        nonlocal capital, pos

        # Close existing
        if pos is not None and pos["asset"] != new_asset:
            if pos["asset"] == 1:
                exit_px = float(opens1[idx])
            else:
                exit_px = float(opens2[idx])
            pnl = (exit_px - pos["entry"]) * pos["qty"]
            capital += pnl
            trades.append({
                "Entry Date":  pos["entry_dt"],
                "Exit Date":   dates[idx],
                "Holding":     "Niftybees" if pos["asset"] == 1 else "Goldbees",
                "Entry Price": round(pos["entry"], 4),
                "Exit Price":  round(exit_px, 4),
                "Qty":         pos["qty"],
                "PnL":         round(pnl, 2),
                "Exit Reason": "Switch Signal",
                "Capital":     round(capital, 2),
            })
            pos = None

        # Open new
        if pos is None or pos["asset"] != new_asset:
            if new_asset == 1:
                entry_px = float(opens1[idx])
                ticker   = "Niftybees"
            else:
                entry_px = float(opens2[idx])
                ticker   = "Goldbees"
            qty = max(1, int(capital * 0.95 / entry_px))
            pos = {
                "asset":    new_asset,
                "ticker":   ticker,
                "entry":    entry_px,
                "entry_dt": dates[idx],
                "qty":      qty,
            }

    i = 0
    while i < n:
        sig_val = int(sig_arr[i]) if not np.isnan(sig_arr[i]) else 0

        # Signal on candle i → switch at candle i+1 open
        if sig_val != 0 and i + 1 < n:
            if pos is None or pos["asset"] != sig_val:
                switch(sig_val, i + 1)
            i += 1
            continue

        i += 1

    # Close open position at last available close
    if pos is not None:
        exit_px = float(closes1[-1]) if pos["asset"] == 1 else float(closes2[-1])
        pnl     = (exit_px - pos["entry"]) * pos["qty"]
        capital += pnl
        trades.append({
            "Entry Date":  pos["entry_dt"],
            "Exit Date":   dates[-1],
            "Holding":     pos["ticker"],
            "Entry Price": round(pos["entry"], 4),
            "Exit Price":  round(exit_px, 4),
            "Qty":         pos["qty"],
            "PnL":         round(pnl, 2),
            "Exit Reason": "End of Data",
            "Capital":     round(capital, 2),
        })

    return pd.DataFrame(trades), capital


def calc_stats(trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty:
        return {}
    pnl    = trades["PnL"]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    cap    = trades["Capital"]

    total_ret = (float(cap.iloc[-1]) - initial_capital) / initial_capital * 100
    win_rate  = len(wins) / len(pnl) * 100 if len(pnl) else 0
    pf        = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
    max_dd    = ((cap - cap.cummax()) / cap.cummax() * 100).min()
    sharpe    = pnl.mean() / pnl.std() * (252 ** 0.5) if pnl.std() > 0 else 0

    return {
        "Total Trades":   int(len(pnl)),
        "Win Rate":        f"{win_rate:.1f}%",
        "Total PnL":       round(float(pnl.sum()), 2),
        "Total Return":    f"{total_ret:.2f}%",
        "Avg Win":         round(float(wins.mean()), 2) if len(wins) else 0.0,
        "Avg Loss":        round(float(losses.mean()), 2) if len(losses) else 0.0,
        "Profit Factor":   round(float(pf), 2),
        "Max Drawdown":    f"{max_dd:.2f}%",
        "Winning Trades":  int(len(wins)),
        "Losing Trades":   int(len(losses)),
        "Expectancy":      round(float(pnl.mean()), 2),
        "Sharpe Proxy":    round(float(sharpe), 2),
    }


# ─────────────────────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, signals: pd.DataFrame,
                t1_label: str,
                df2: Optional[pd.DataFrame], t2_label: str,
                show_ratio: bool, strategy: str,
                ema_fast: int = 9, ema_slow: int = 15,
                trades: Optional[pd.DataFrame] = None,
                ratio_lb: int = 20) -> go.Figure:

    has_ratio = show_ratio and df2 is not None and "ratio" in signals.columns
    rows    = 3 if has_ratio else 2
    heights = [0.55, 0.25, 0.20] if has_ratio else [0.72, 0.28]
    subtitles = ([t1_label, f"Ratio ({t1_label}/{t2_label})", "Volume"] if has_ratio
                 else [t1_label, "Volume"])

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=heights, subplot_titles=subtitles,
                        vertical_spacing=0.035)

    # ── Candlestick ─────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=t1_label,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#1e7a6e", decreasing_fillcolor="#b33a3a",
    ), row=1, col=1)

    # ── Strategy overlays ───────────────────────────────────
    if strategy == "EMA Crossover":
        fig.add_trace(go.Scatter(x=df.index, y=calc_ema(df["Close"], ema_fast),
                                  name=f"EMA {ema_fast}",
                                  line=dict(color="#f5a623", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=calc_ema(df["Close"], ema_slow),
                                  name=f"EMA {ema_slow}",
                                  line=dict(color="#5b9bd5", width=1.5)), row=1, col=1)

    if strategy == "SMC (Smart Money Concepts)" and "swing_high" in signals.columns:
        fig.add_trace(go.Scatter(x=signals.index, y=signals["swing_high"],
                                  name="Struct. High", line=dict(color="#cc44ff", width=1, dash="dot"),
                                  opacity=0.8), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["swing_low"],
                                  name="Struct. Low",  line=dict(color="#00ccff", width=1, dash="dot"),
                                  opacity=0.8), row=1, col=1)

    # ── Normalised overlay of ticker2 ───────────────────────
    if df2 is not None and t2_label:
        try:
            scale = float(df["Close"].iloc[0]) / float(df2["Close"].iloc[0])
            fig.add_trace(go.Scatter(
                x=df2.index, y=df2["Close"] * scale,
                name=f"{t2_label} (norm.)",
                line=dict(color="#e2b86e", width=1.2, dash="dot"), opacity=0.65
            ), row=1, col=1)
        except Exception:
            pass

    # ── Buy / Sell signals ──────────────────────────────────
    buys  = signals[signals["signal"] ==  1]
    sells = signals[signals["signal"] == -1]
    for mask, sym, color, offset in [
        (buys,  "triangle-up",   "#26a69a",  0.997),
        (sells, "triangle-down", "#ef5350",  1.003),
    ]:
        idx_in = mask.index.intersection(df.index)
        if not idx_in.empty:
            y_vals = df.loc[idx_in, "Low"] * offset if "up" in sym else df.loc[idx_in, "High"] * offset
            fig.add_trace(go.Scatter(
                x=idx_in, y=y_vals, mode="markers",
                marker=dict(symbol=sym, size=11, color=color,
                            line=dict(width=1, color="#0a0d14")),
                name="Buy" if "up" in sym else "Sell"
            ), row=1, col=1)

    # ── SL / TP lines from backtest ─────────────────────────
    if trades is not None and not trades.empty:
        for _, tr in trades.iterrows():
            try:
                fig.add_shape(type="line",
                              x0=tr["Entry Date"], x1=tr["Exit Date"],
                              y0=tr["SL"], y1=tr["SL"],
                              line=dict(color="rgba(239,83,80,0.45)", width=1, dash="dot"),
                              row=1, col=1)
                fig.add_shape(type="line",
                              x0=tr["Entry Date"], x1=tr["Exit Date"],
                              y0=tr["TP1"], y1=tr["TP1"],
                              line=dict(color="rgba(38,166,154,0.45)", width=1, dash="dot"),
                              row=1, col=1)
            except Exception:
                pass

    # ── Ratio subplot ───────────────────────────────────────
    vol_row = 2
    if has_ratio:
        ratio = signals["ratio"]
        rh    = signals["ratio_high"]
        rl    = signals["ratio_low"]

        # Shaded band between high and low
        fig.add_trace(go.Scatter(x=rh.index, y=rh, fill=None, mode="lines",
                                  line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=rl.index, y=rl, fill="tonexty", mode="lines",
                                  fillcolor="rgba(226,184,110,0.07)",
                                  line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio",
                                  line=dict(color="#e0e8f0", width=1.8)), row=2, col=1)
        fig.add_trace(go.Scatter(x=rh.index, y=rh, name=f"{ratio_lb}d High",
                                  line=dict(color="#26a69a", width=1.2, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=rl.index, y=rl, name=f"{ratio_lb}d Low",
                                  line=dict(color="#ef5350", width=1.2, dash="dash")), row=2, col=1)

        # Mark buy/sell on ratio
        buy_ratio  = ratio[signals["signal"] ==  1]
        sell_ratio = ratio[signals["signal"] == -1]
        if not buy_ratio.empty:
            fig.add_trace(go.Scatter(x=buy_ratio.index, y=buy_ratio,
                                      mode="markers",
                                      marker=dict(symbol="circle", size=8, color="#26a69a"),
                                      name="Ratio Buy", showlegend=False), row=2, col=1)
        if not sell_ratio.empty:
            fig.add_trace(go.Scatter(x=sell_ratio.index, y=sell_ratio,
                                      mode="markers",
                                      marker=dict(symbol="circle", size=8, color="#ef5350"),
                                      name="Ratio Sell", showlegend=False), row=2, col=1)
        vol_row = 3

    # ── Volume bars ─────────────────────────────────────────
    bar_colors = ["#26a69a" if float(c) >= float(o) else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=bar_colors, opacity=0.65, showlegend=False),
                  row=vol_row, col=1)

    # ── Layout ──────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=680,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.03, x=0,
                    bgcolor="rgba(10,13,20,0.6)", bordercolor="#1c2434",
                    font=dict(size=11)),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0a0d14",
        plot_bgcolor="#0d1117",
        hoverlabel=dict(bgcolor="#1c2434", font_size=12),
    )
    for axis in ["xaxis", "xaxis2", "xaxis3",
                 "yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{axis: dict(
            showgrid=True, gridcolor="#151e2c", gridwidth=1,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#334", spikethickness=1,
        )})
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<div style="font-family:Inter,sans-serif;font-size:1.5rem;font-weight:700;'
        'color:#e2b86e;letter-spacing:0.04em;margin-bottom:2px">📈 Trading Strategy Platform</div>'
        '<div style="font-size:0.78rem;color:#445566;margin-bottom:16px">'
        'Ratio Strategy • EMA Crossover • SMC • Institutional Candle • Live Streaming</div>',
        unsafe_allow_html=True
    )

    # ══════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════
    with st.sidebar:

        # Ratio toggle
        use_ratio = st.checkbox("📊 Enable Ratio Mode", value=True,
                                 help="Two-asset ratio strategy (Niftybees/Goldbees style).\n"
                                      "Uncheck for single-ticker analysis.")

        st.markdown('<div class="section-header">Ticker(s)</div>', unsafe_allow_html=True)

        if use_ratio:
            t1_choice = st.selectbox("Ticker 1 (Numerator)", list(TICKERS.keys()), index=0)
            t1_sym = TICKERS[t1_choice]
            if t1_sym == "__custom__":
                t1_sym = st.text_input("Symbol 1", "NIFTYBEES.NS")

            t2_options = ["Goldbees", "Gold", "Silver", "BTC/USD", "ETH/USD", "Custom"]
            t2_choice = st.selectbox("Ticker 2 (Denominator)", t2_options, index=0)
            _t2m = {"Goldbees": "GOLDBEES.NS", "Gold": "GC=F",
                    "Silver": "SI=F", "BTC/USD": "BTC-USD",
                    "ETH/USD": "ETH-USD", "Custom": "__custom__"}
            t2_sym = _t2m.get(t2_choice, "GOLDBEES.NS")
            if t2_sym == "__custom__":
                t2_sym = st.text_input("Symbol 2", "GOLDBEES.NS")
            ratio_lb = st.number_input("Ratio Lookback (candles)", 3, 200, 20)
        else:
            t1_choice = st.selectbox("Ticker", list(TICKERS.keys()), index=2)
            t1_sym = TICKERS[t1_choice]
            if t1_sym == "__custom__":
                t1_sym = st.text_input("Custom Symbol", "^NSEI")
            t2_sym    = None
            t2_choice = ""
            ratio_lb  = 20

        st.markdown('<div class="section-header">Timeframe</div>', unsafe_allow_html=True)
        interval  = st.selectbox("Interval", list(TIMEFRAME_PERIODS.keys()), index=4)
        v_periods = TIMEFRAME_PERIODS[interval]
        period    = st.selectbox("Period", v_periods, index=min(4, len(v_periods) - 1))

        st.markdown('<div class="section-header">Strategy</div>', unsafe_allow_html=True)
        strat_opts = (["Ratio Strategy"] if use_ratio else []) + [
            "EMA Crossover", "Simple Buy", "Simple Sell",
            "Institutional Candle Entry", "SMC (Smart Money Concepts)"
        ]
        strategy = st.selectbox("Select Strategy", strat_opts)

        ema_fast, ema_slow = 9, 15
        if strategy == "EMA Crossover":
            ca, cb = st.columns(2)
            with ca: ema_fast = st.number_input("Fast EMA", 2, 500, 9)
            with cb: ema_slow = st.number_input("Slow EMA", 2, 500, 15)

        st.markdown('<div class="section-header">SL / Target Method</div>', unsafe_allow_html=True)
        sl_method = st.selectbox("Method", SL_METHODS)

        sl_params: dict = {}
        if sl_method == "Custom Points":
            sl_params["custom_pts"] = st.number_input("Points", 0.01, 1e8, 10.0, step=0.5)
        if "ATR" in sl_method or "Volatility" in sl_method:
            sl_params["atr_mult"] = st.number_input("ATR Multiplier", 0.1, 20.0, 1.5, step=0.1)
        if "Trailing SL" in sl_method or "Trailing Target" in sl_method:
            sl_params["trailing_pts"] = st.number_input("Trail Points", 0.01, 1e8, 10.0, step=0.5)
        if "Shift SL" in sl_method:
            sl_params["shift_pct"] = st.number_input("Shift After % Target", 1.0, 99.0, 30.0)
            sl_params["trail_pct"] = st.number_input("Trail Every %",         1.0, 99.0, 30.0)
        if "Partial Exit" in sl_method:
            sl_params["exit_pct"] = st.number_input("Exit % at TP1", 1.0, 99.0, 70.0)

        st.markdown('<div class="section-header">Capital</div>', unsafe_allow_html=True)
        initial_capital = st.number_input("Initial Capital (₹)", 1_000, 50_000_000, 100_000, step=5_000)

        # ── Dhan Broker Integration ──────────────────────────
        st.markdown('<div class="section-header">Broker Integration</div>',
                    unsafe_allow_html=True)
        use_dhan = st.checkbox("🏦 Enable Dhan Broker", value=False,
                                help="When enabled, live signals will place real orders on Dhan.\n"
                                     "Requires Client ID and Access Token.")

        dhan_client   = None
        dhan_product  = "INTRADAY"
        dhan_connected = False

        if use_dhan:
            dhan_client_id    = st.text_input("Client ID",     type="password",
                                               placeholder="Your Dhan client ID")
            dhan_access_token = st.text_input("Access Token",  type="password",
                                               placeholder="Your Dhan access token")
            dhan_product      = st.selectbox("Product Type", ["INTRADAY", "CNC"], index=0,
                                              help="INTRADAY: squared off end-of-day.\n"
                                                   "CNC: delivery / carry-forward.")
            dhan_order_type   = st.selectbox("Order Type",   ["MARKET", "LIMIT"],  index=0)

            if dhan_client_id and dhan_access_token:
                dhan_client, dhan_err = init_dhan_client(dhan_client_id, dhan_access_token)
                if dhan_err:
                    st.error(f"❌ {dhan_err}")
                else:
                    dhan_connected = True
                    funds = dhan_get_funds(dhan_client)
                    avail = funds.get("availabelBalance",
                            funds.get("availableBalance",
                            funds.get("sodLimit", "—")))
                    st.success(f"✅ Connected  |  Available: ₹{avail}")
            else:
                st.info("Enter credentials above to connect.")
                dhan_order_type = "MARKET"

            # Instrument ID override
            with st.expander("🔧 Security ID overrides", expanded=False):
                st.caption("Override auto-mapped Dhan security IDs if needed.")
                for name, info in DHAN_INSTRUMENTS.items():
                    if info["tradable"]:
                        new_id = st.text_input(
                            f"{name}",
                            value=info["security_id"],
                            key=f"sec_{name}",
                        )
                        DHAN_INSTRUMENTS[name]["security_id"] = new_id
        else:
            dhan_order_type = "MARKET"

    # ══════════════════════════════════════════════════════
    # FETCH DATA  (cached)
    # ══════════════════════════════════════════════════════
    with st.spinner("⏳ Fetching market data…"):
        df1 = fetch_data(t1_sym, interval, period)
        df2 = fetch_data(t2_sym, interval, period) if t2_sym else None

    if df1 is None or df1.empty:
        st.error(f"❌ No data for **{t1_sym}** ({interval} / {period}). "
                  "Try a different period, interval, or check the symbol.")
        st.stop()

    if df2 is not None and not df2.empty:
        common = df1.index.intersection(df2.index)
        if len(common) == 0:
            st.warning("⚠️ Tickers have no overlapping dates. Ratio mode disabled.")
            df2 = None
        else:
            df1 = df1.loc[common]
            df2 = df2.loc[common]

    # ── Signals & Backtest ──────────────────────────────────
    sig_kw = {"lookback": ratio_lb, "fast": ema_fast, "slow": ema_slow}
    sigs   = generate_signals(df1, strategy, df2=df2, **sig_kw)

    is_ratio_strat = (strategy == "Ratio Strategy" and df2 is not None)

    if is_ratio_strat:
        trades_df, final_cap = run_ratio_backtest(
            df1, df2, sigs, float(initial_capital)
        )
    else:
        trades_df, final_cap = run_backtest(
            df1, sigs, sl_method=sl_method,
            initial_capital=float(initial_capital),
            df2=df2, **sl_params, **sig_kw
        )
    stats = calc_stats(trades_df, float(initial_capital))

    # ── Session state for live trading ──────────────────────
    for key, default in [
        ("live_running",  False),
        ("live_position", None),
        ("live_trades",   []),
        ("live_capital",  float(initial_capital)),
        ("square_off",    False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ══════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════
    tab_bt, tab_live, tab_hist = st.tabs(
        ["  📊  Backtesting  ", "  🔴  Live Trading  ", "  📋  Trade History  "]
    )

    # ──────────────────────────────────────────────────────
    # BACKTESTING TAB
    # ──────────────────────────────────────────────────────
    with tab_bt:

        if stats:
            ret_val      = float(stats["Total Return"].replace("%", ""))
            is_profitable = ret_val > 0
            bg_col   = "#0d2318" if is_profitable else "#1e0d0d"
            bdr_col  = "#26a69a" if is_profitable else "#ef5350"
            icon     = "✅" if is_profitable else "❌"
            verdict  = "PROFITABLE" if is_profitable else "NOT PROFITABLE"
            desc = ("Positive returns over the selected period. "
                     "Consider live deployment with strict risk management."
                     if is_profitable else
                     "Negative returns. Try adjusting parameters, period, or strategy.")

            st.markdown(
                f'<div class="verdict-box" style="background:{bg_col};border:1.5px solid {bdr_col}">'
                f'<span style="font-size:1.2rem;font-weight:700;color:{bdr_col}">'
                f'{icon} Strategy Verdict: {verdict}</span><br>'
                f'<span style="font-size:0.82rem;color:#8899aa">{desc}</span></div>',
                unsafe_allow_html=True
            )

            # Metrics row 1
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            for col, (lab, val) in zip(
                [c1, c2, c3, c4, c5, c6],
                [("Total Trades",  stats["Total Trades"]),
                 ("Win Rate",       stats["Win Rate"]),
                 ("Total Return",   stats["Total Return"]),
                 ("Profit Factor",  stats["Profit Factor"]),
                 ("Max Drawdown",   stats["Max Drawdown"]),
                 ("Sharpe (proxy)", stats["Sharpe Proxy"])]
            ):
                col.metric(lab, val)

            # Metrics row 2
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, (lab, val) in zip(
                [c1, c2, c3, c4, c5],
                [("Total PnL (₹)",  stats["Total PnL"]),
                 ("Avg Win (₹)",    stats["Avg Win"]),
                 ("Avg Loss (₹)",   stats["Avg Loss"]),
                 ("Winners",        stats["Winning Trades"]),
                 ("Expectancy",     stats["Expectancy"])]
            ):
                col.metric(lab, val)
        else:
            st.info("ℹ️ No trades generated. Try adjusting the strategy parameters or period.")

        # Main chart
        chart = build_chart(
            df1, sigs, t1_choice, df2, t2_choice,
            use_ratio, strategy, ema_fast, ema_slow,
            trades_df if not trades_df.empty else None,
            ratio_lb=ratio_lb
        )
        st.plotly_chart(chart, use_container_width=True)

        # Equity curve
        if not trades_df.empty:
            st.markdown("**Equity Curve**")
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=trades_df["Exit Date"], y=trades_df["Capital"],
                mode="lines", fill="tozeroy",
                line=dict(color="#26a69a", width=2),
                fillcolor="rgba(38,166,154,0.12)", name="Equity"
            ))
            eq_fig.add_hline(y=initial_capital, line_dash="dash",
                              line_color="#556677",
                              annotation_text="Initial Capital",
                              annotation_font_color="#778899")
            eq_fig.update_layout(
                template="plotly_dark", height=220,
                paper_bgcolor="#0a0d14", plot_bgcolor="#0d1117",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=True, gridcolor="#151e2c"),
                yaxis=dict(showgrid=True, gridcolor="#151e2c"),
            )
            st.plotly_chart(eq_fig, use_container_width=True)

    # ──────────────────────────────────────────────────────
    # LIVE TRADING TAB
    # ──────────────────────────────────────────────────────
    with tab_live:
        st.markdown(
            '<div style="color:#ef5350;font-weight:700;font-size:1.05rem;margin-bottom:4px">'
            '● LIVE TRADING MODE</div>'
            '<div style="color:#445566;font-size:0.78rem;margin-bottom:12px">'
            'Refreshes every 0.3 s &nbsp;|&nbsp; '
            'SL/TP checked via LTP &nbsp;|&nbsp; '
            'Entry signals use same strategy logic</div>',
            unsafe_allow_html=True
        )

        live_iv = st.selectbox("Live Interval", ["1m", "5m", "15m"],
                                index=0, key="live_iv")

        # ── START / STOP / SQUARE OFF controls ──────────────
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 3])
        with btn_col1:
            if st.button(
                "▶  START",
                type="primary",
                disabled=st.session_state.live_running,
                use_container_width=True,
            ):
                st.session_state.live_running = True
                st.session_state.live_capital = float(initial_capital)
                st.session_state.live_trades  = []
                st.rerun()

        with btn_col2:
            if st.button(
                "⏹  STOP",
                disabled=not st.session_state.live_running,
                use_container_width=True,
            ):
                st.session_state.live_running = False
                st.rerun()

        with btn_col3:
            if st.button(
                "🔴  SQUARE OFF",
                type="secondary",
                disabled=st.session_state.live_position is None,
                use_container_width=True,
            ):
                st.session_state.square_off = True
                st.rerun()

        with btn_col4:
            status_color = "#26a69a" if st.session_state.live_running else "#556677"
            status_text  = "● RUNNING" if st.session_state.live_running else "● STOPPED"
            pos_info = ""
            if st.session_state.live_position:
                lp = st.session_state.live_position
                pos_info = (f" &nbsp;|&nbsp; Holding: <b>{lp['ticker']}</b>"
                             f" &nbsp;@&nbsp; {lp['entry']:.4f}"
                             f" &nbsp;×&nbsp; {lp['qty']} units")
            st.markdown(
                f'<div style="padding:8px 14px;border-radius:8px;'
                f'background:#0d1117;border:1px solid #1c2434;'
                f'font-size:0.82rem;margin-top:4px;color:{status_color}">'
                f'<b>{status_text}</b>{pos_info}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Live panel (fragment) ────────────────────────────
        @st.fragment(run_every=0.3)
        def live_panel():
            # Only fetch & update when running
            if not st.session_state.live_running and not st.session_state.square_off:
                st.info("Press **▶ START** to begin live monitoring.")
                # Still show last known chart if data exists
                ldf1_static = fetch_live(t1_sym, live_iv)
                ldf2_static = fetch_live(t2_sym, live_iv) if t2_sym else None
                if ldf1_static is not None and not ldf1_static.empty:
                    eff_lb_s = min(ratio_lb, max(1, len(ldf1_static) - 1))
                    lsigs_s  = generate_signals(ldf1_static, strategy, df2=ldf2_static,
                                                lookback=eff_lb_s, fast=ema_fast, slow=ema_slow)
                    lfig_s = build_chart(ldf1_static, lsigs_s, f"{t1_choice} (Live)",
                                         ldf2_static, t2_choice,
                                         use_ratio, strategy, ema_fast, ema_slow,
                                         ratio_lb=eff_lb_s)
                    st.plotly_chart(lfig_s, use_container_width=True, key="live_chart_static")
                return

            ldf1 = fetch_live(t1_sym, live_iv)
            ldf2 = fetch_live(t2_sym, live_iv) if t2_sym else None

            if ldf1 is None or ldf1.empty:
                st.warning("⚠️ Waiting for live data… (market may be closed)")
                return

            if ldf2 is not None and not ldf2.empty:
                common_l = ldf1.index.intersection(ldf2.index)
                if len(common_l):
                    ldf1 = ldf1.loc[common_l]
                    ldf2 = ldf2.loc[common_l]
                else:
                    ldf2 = None

            last = ldf1.iloc[-1]
            ltp  = float(last["Close"])
            prev = float(ldf1["Close"].iloc[-2]) if len(ldf1) > 1 else ltp
            chg  = ltp - prev
            chgp = chg / prev * 100 if prev else 0.0

            # ── Handle Square Off ────────────────────────────
            if st.session_state.square_off and st.session_state.live_position is not None:
                lp      = st.session_state.live_position
                exit_px = ltp if lp["asset"] == 1 else (
                    float(fetch_live(t2_sym, live_iv)["Close"].iloc[-1])
                    if t2_sym else ltp
                )
                pnl = (exit_px - lp["entry"]) * lp["qty"]
                st.session_state.live_capital += pnl

                sq_status = "simulated"
                sq_order_id = "—"
                if use_dhan and dhan_connected and dhan_client:
                    sq_resp = dhan_place_order(
                        dhan_client, lp["ticker"], "SELL",
                        lp["qty"], dhan_product, dhan_order_type
                    )
                    sq_status   = sq_resp.get("status", "unknown")
                    sq_order_id = (sq_resp.get("data", {}) or {}).get("orderId", "—")
                    if sq_status not in ("success",):
                        st.error(f"⚠️ Square-off order failed: {sq_resp.get('remarks','')}")

                st.session_state.live_trades.append({
                    "Time":     datetime.now().strftime("%H:%M:%S"),
                    "Action":   f"SQUARE OFF {lp['ticker']}",
                    "Price":    round(exit_px, 4),
                    "Qty":      lp["qty"],
                    "PnL":      round(pnl, 2),
                    "Capital":  round(st.session_state.live_capital, 2),
                    "Broker":   sq_status,
                    "Order ID": sq_order_id,
                })
                st.session_state.live_position = None
                st.session_state.square_off    = False

            # ── Live signals ─────────────────────────────────
            eff_lb  = min(ratio_lb, max(1, len(ldf1) - 1))
            lsigs   = generate_signals(ldf1, strategy, df2=ldf2,
                                        lookback=eff_lb, fast=ema_fast, slow=ema_slow)
            last_sig = int(lsigs["signal"].iloc[-1])

            # ── Signal labels ────────────────────────────────
            if is_ratio_strat:
                _sig_map = {
                     1: (f"🟢  BUY {t1_choice}  |  Close {t2_choice}",  "#26a69a"),
                    -1: (f"🟢  BUY {t2_choice}  |  Close {t1_choice}",  "#e2b86e"),
                     0: ("⚪  HOLD — No switch signal",                   "#8899aa"),
                }
            else:
                _sig_map = {
                     1: ("🟢  BUY",  "#26a69a"),
                    -1: ("🔴  SELL", "#ef5350"),
                     0: ("⚪  HOLD", "#8899aa"),
                }
            sig_txt, sig_col = _sig_map[last_sig]

            # ── Auto-execute: signal → order ─────────────────
            if st.session_state.live_running and last_sig != 0:
                cur_pos    = st.session_state.live_position
                new_asset  = last_sig
                new_ticker = t1_choice if last_sig == 1 else t2_choice
                entry_px   = ltp if last_sig == 1 else (
                    float(ldf2["Close"].iloc[-1]) if ldf2 is not None else ltp
                )

                if cur_pos is None or cur_pos["asset"] != new_asset:

                    # ── Close existing position ───────────────
                    if cur_pos is not None:
                        old_exit = ltp if cur_pos["asset"] == 1 else (
                            float(ldf2["Close"].iloc[-1]) if ldf2 is not None else ltp
                        )
                        pnl = (old_exit - cur_pos["entry"]) * cur_pos["qty"]
                        st.session_state.live_capital += pnl

                        close_status   = "simulated"
                        close_order_id = "—"
                        if use_dhan and dhan_connected and dhan_client:
                            c_resp = dhan_place_order(
                                dhan_client, cur_pos["ticker"], "SELL",
                                cur_pos["qty"], dhan_product, dhan_order_type
                            )
                            close_status   = c_resp.get("status", "unknown")
                            close_order_id = (c_resp.get("data", {}) or {}).get("orderId", "—")
                            if close_status not in ("success",):
                                st.warning(f"Close order issue: {c_resp.get('remarks','')}")

                        st.session_state.live_trades.append({
                            "Time":     datetime.now().strftime("%H:%M:%S"),
                            "Action":   f"SELL {cur_pos['ticker']}",
                            "Price":    round(old_exit, 4),
                            "Qty":      cur_pos["qty"],
                            "PnL":      round(pnl, 2),
                            "Capital":  round(st.session_state.live_capital, 2),
                            "Broker":   close_status,
                            "Order ID": close_order_id,
                        })

                    # ── Open new position ─────────────────────
                    qty = max(1, int(
                        st.session_state.live_capital * 0.95 / max(entry_px, 0.01)
                    ))

                    buy_status   = "simulated"
                    buy_order_id = "—"
                    if use_dhan and dhan_connected and dhan_client:
                        b_resp = dhan_place_order(
                            dhan_client, new_ticker, "BUY",
                            qty, dhan_product, dhan_order_type
                        )
                        buy_status   = b_resp.get("status", "unknown")
                        buy_order_id = (b_resp.get("data", {}) or {}).get("orderId", "—")
                        if buy_status not in ("success",):
                            st.error(f"⚠️ Buy order failed: {b_resp.get('remarks','')}")

                    st.session_state.live_position = {
                        "asset":    new_asset,
                        "ticker":   new_ticker,
                        "entry":    entry_px,
                        "entry_dt": datetime.now().strftime("%H:%M:%S"),
                        "qty":      qty,
                    }
                    st.session_state.live_trades.append({
                        "Time":     datetime.now().strftime("%H:%M:%S"),
                        "Action":   f"BUY {new_ticker}",
                        "Price":    round(entry_px, 4),
                        "Qty":      qty,
                        "PnL":      0.0,
                        "Capital":  round(st.session_state.live_capital, 2),
                        "Broker":   buy_status,
                        "Order ID": buy_order_id,
                    })

            # ── Top price metrics ────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(t1_choice, f"{ltp:.4f}", f"{chg:+.4f} ({chgp:+.2f}%)")
            c2.metric("Open",   f"{float(last['Open']):.4f}")
            c3.metric("High",   f"{float(last['High']):.4f}")
            c4.metric("Low",    f"{float(last['Low']):.4f}")
            c5.metric("Volume", f"{int(last['Volume']):,}")

            if ldf2 is not None and not ldf2.empty:
                ltp2  = float(ldf2["Close"].iloc[-1])
                prev2 = float(ldf2["Close"].iloc[-2]) if len(ldf2) > 1 else ltp2
                chg2  = ltp2 - prev2
                chgp2 = chg2 / prev2 * 100 if prev2 else 0.0
                d1, d2, d3 = st.columns(3)
                d1.metric(t2_choice, f"{ltp2:.4f}", f"{chg2:+.4f} ({chgp2:+.2f}%)")
                ratio_now = ltp / ltp2 if ltp2 else 0
                d2.metric("Ratio (live)", f"{ratio_now:.4f}")
                d3.metric("Live Capital", f"₹{st.session_state.live_capital:,.2f}")

            # Signal box
            st.markdown(
                f'<div style="font-size:1.4rem;font-weight:700;color:{sig_col};'
                f'background:{sig_col}14;border:1.5px solid {sig_col}40;'
                f'border-radius:10px;padding:10px 20px;display:inline-block;margin:6px 0 10px">'
                f'Signal: {sig_txt}</div>',
                unsafe_allow_html=True,
            )

            # SL/TP block (non-ratio strategies only)
            if not is_ratio_strat and last_sig != 0:
                idx_l = max(0, len(ldf1) - 1)
                sl_l, tp1_l, tp2_l, tp3_l = calc_sl_tp(
                    ldf1, idx_l, ltp, last_sig, sl_method, **sl_params
                )
                risk = abs(ltp - sl_l)
                rr   = abs(tp1_l - ltp) / risk if risk > 0 else 0.0
                sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
                sc1.metric("Entry (LTP)", f"{ltp:.4f}")
                sc2.metric("Stop Loss",   f"{sl_l:.4f}",
                            f"{'-' if last_sig == 1 else '+'}{abs(ltp - sl_l):.4f}")
                sc3.metric("TP1", f"{tp1_l:.4f}")
                sc4.metric("TP2", f"{tp2_l:.4f}")
                sc5.metric("TP3", f"{tp3_l:.4f}")
                sc6.metric("R:R", f"1 : {rr:.2f}")

            # Live trade log
            if st.session_state.live_trades:
                st.markdown("**📋 Live Trade Log** (this session)")
                log_df = pd.DataFrame(st.session_state.live_trades)
                def _log_color(v):
                    if isinstance(v, (int, float)):
                        return "color:#26a69a;font-weight:600" if v > 0 else (
                               "color:#ef5350;font-weight:600" if v < 0 else "")
                    return ""
                styled_log = log_df.style.applymap(_log_color, subset=["PnL"])
                st.dataframe(styled_log, use_container_width=True, height=160)

            # ── Dhan positions / order book ───────────────────
            if use_dhan and dhan_connected and dhan_client:
                dhan_tabs = st.tabs(["📂 Positions", "📑 Order Book", "💰 Funds"])

                with dhan_tabs[0]:
                    pos_df = dhan_get_positions(dhan_client)
                    if pos_df.empty:
                        st.info("No open positions on Dhan.")
                    else:
                        st.dataframe(pos_df, use_container_width=True)
                        # Total unrealised P&L
                        if "unrealizedProfit" in pos_df.columns:
                            total_unrl = pd.to_numeric(
                                pos_df["unrealizedProfit"], errors="coerce"
                            ).sum()
                            color = "#26a69a" if total_unrl >= 0 else "#ef5350"
                            st.markdown(
                                f'<div style="font-size:1rem;font-weight:700;color:{color};'
                                f'margin-top:6px">Unrealised P&L: ₹{total_unrl:,.2f}</div>',
                                unsafe_allow_html=True,
                            )

                with dhan_tabs[1]:
                    ord_df = dhan_get_orders(dhan_client)
                    if ord_df.empty:
                        st.info("No orders today.")
                    else:
                        st.dataframe(ord_df, use_container_width=True)

                with dhan_tabs[2]:
                    funds = dhan_get_funds(dhan_client)
                    if funds:
                        fc = st.columns(3)
                        fc[0].metric("Available Balance",
                                      f"₹{funds.get('availabelBalance', funds.get('availableBalance','—'))}")
                        fc[1].metric("Used Margin",
                                      f"₹{funds.get('utilizedAmount', '—')}")
                        fc[2].metric("Total Balance",
                                      f"₹{funds.get('sodLimit', '—')}")
                    else:
                        st.info("Could not fetch fund details.")

            # Live chart
            lfig = build_chart(
                ldf1, lsigs, f"{t1_choice} (Live)",
                ldf2, t2_choice,
                use_ratio, strategy, ema_fast, ema_slow,
                ratio_lb=eff_lb,
            )
            st.plotly_chart(lfig, use_container_width=True, key="live_chart")
            st.caption(f"🕐 Last update: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        live_panel()

    # ──────────────────────────────────────────────────────
    # TRADE HISTORY TAB
    # ──────────────────────────────────────────────────────
    with tab_hist:
        st.markdown("### 📋 Trade History")

        if trades_df.empty:
            st.info("No trades to display. Run a backtest first.")
        else:
            # Filters — adapt columns to strategy type
            f1, f2, f3 = st.columns(3)
            with f1:
                if is_ratio_strat and "Holding" in trades_df.columns:
                    holding_opts = ["All"] + sorted(trades_df["Holding"].unique().tolist())
                    f_holding = st.selectbox("Holding", holding_opts)
                    f_dir = "All"
                else:
                    f_dir = st.selectbox("Direction", ["All", "Long", "Short"])
                    f_holding = "All"
            with f2:
                reasons = ["All"] + sorted(trades_df["Exit Reason"].unique().tolist())
                f_reason = st.selectbox("Exit Reason", reasons)
            with f3:
                f_res = st.selectbox("Result", ["All", "Winning", "Losing"])

            ftd = trades_df.copy()
            if is_ratio_strat and f_holding != "All" and "Holding" in ftd.columns:
                ftd = ftd[ftd["Holding"] == f_holding]
            elif f_dir != "All" and "Direction" in ftd.columns:
                ftd = ftd[ftd["Direction"] == f_dir]
            if f_reason != "All":   ftd = ftd[ftd["Exit Reason"] == f_reason]
            if f_res == "Winning":  ftd = ftd[ftd["PnL"] >  0]
            if f_res == "Losing":   ftd = ftd[ftd["PnL"] <= 0]

            # Summary
            if not ftd.empty:
                ms1, ms2, ms3, ms4, ms5 = st.columns(5)
                ms1.metric("Filtered Trades",    len(ftd))
                ms2.metric("Total PnL",           f"{ftd['PnL'].sum():.2f}")
                ms3.metric("Win Rate",
                            f"{len(ftd[ftd['PnL']>0]) / len(ftd) * 100:.1f}%")
                ms4.metric("Best Trade",          f"{ftd['PnL'].max():.2f}")
                ms5.metric("Worst Trade",         f"{ftd['PnL'].min():.2f}")

            # Styled table
            def _pnl_color(v):
                return "color: #26a69a; font-weight:600" if v > 0 else "color: #ef5350; font-weight:600"

            fmt_dict = {c: "{:.4f}" for c in ["Entry Price","Exit Price","SL","TP1","TP2","TP3"]}
            fmt_dict.update({"PnL": "{:.2f}", "Capital": "{:.2f}"})

            styled = (ftd.style
                       .applymap(_pnl_color, subset=["PnL"])
                       .format(fmt_dict, na_rep="—"))
            st.dataframe(styled, use_container_width=True, height=420)

            # ── Exit reason breakdown ─────────────────────
            st.markdown("**Exit Reason Breakdown**")
            reason_grp = (trades_df.groupby("Exit Reason")["PnL"]
                           .agg(Count="count", Total_PnL="sum",
                                Avg_PnL="mean",
                                Win_Rate=lambda x: (x > 0).mean() * 100)
                           .round(2))
            st.dataframe(reason_grp, use_container_width=True)

            # ── Monthly PnL bar chart ─────────────────────
            try:
                td_m = trades_df.copy()
                td_m["Month"] = pd.to_datetime(td_m["Exit Date"]).dt.to_period("M").astype(str)
                monthly = td_m.groupby("Month")["PnL"].sum().reset_index()
                if len(monthly) > 1:
                    m_fig = go.Figure(go.Bar(
                        x=monthly["Month"], y=monthly["PnL"],
                        marker_color=["#26a69a" if v >= 0 else "#ef5350"
                                      for v in monthly["PnL"]],
                        text=[f"{v:+.0f}" for v in monthly["PnL"]],
                        textposition="outside",
                    ))
                    m_fig.update_layout(
                        title="Monthly PnL (₹)",
                        template="plotly_dark", height=240,
                        paper_bgcolor="#0a0d14", plot_bgcolor="#0d1117",
                        margin=dict(l=0, r=0, t=35, b=0),
                        font=dict(size=11),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="#151e2c"),
                    )
                    st.plotly_chart(m_fig, use_container_width=True)
            except Exception:
                pass

            # Download
            csv_data = ftd.to_csv(index=False)
            st.download_button("⬇️ Download Filtered Trades (CSV)",
                                csv_data, "trade_history.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
