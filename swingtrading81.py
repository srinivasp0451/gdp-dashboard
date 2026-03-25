"""
╔══════════════════════════════════════════════════════════════╗
║       🌊  ELLIOTT WAVE ALGO TRADER  v4.0                     ║
║  Backtest · Live · Optimization · Multi-TF Analysis · Dhan  ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run elliott_wave_algo_trader.py
pip install streamlit yfinance plotly pandas numpy requests
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional
import requests
import itertools
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌊 Elliott Wave Algo Trader",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.main-hdr {
    background: linear-gradient(135deg,#0a0e1a 0%,#0d1b2a 50%,#0a1628 100%);
    border:1px solid #1e3a5f; border-radius:14px; padding:20px 26px;
    margin-bottom:14px; box-shadow:0 4px 24px rgba(0,229,255,.08);
}
.main-hdr h1 {
    font-family:'Exo 2',sans-serif; font-weight:700;
    background:linear-gradient(90deg,#00e5ff,#4dd0e1,#00bcd4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0; font-size:1.85rem; letter-spacing:.4px;
}
.main-hdr p { color:#546e7a; margin:5px 0 0; font-size:.87rem; }

.sig-buy  { background:rgba(76,175,80,.12); border:1.5px solid #4caf50;
    border-radius:10px; padding:13px 15px; }
.sig-sell { background:rgba(244,67,54,.12); border:1.5px solid #f44336;
    border-radius:10px; padding:13px 15px; }
.sig-hold { background:rgba(100,100,120,.10); border:1.5px solid #455a64;
    border-radius:10px; padding:13px 15px; }

.pos-card { background:rgba(0,150,200,.10); border:1px solid #0288d1;
    border-radius:8px; padding:11px 13px; font-size:.87rem; line-height:1.85; }

.info-box { background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
    padding:12px 14px; font-size:.86rem; line-height:1.9; }

.best-cfg { background:rgba(0,229,255,.07); border:1px solid #00bcd4;
    border-radius:10px; padding:13px 17px; margin:6px 0; }

.stTabs [data-baseweb="tab-list"] { gap:5px; background:transparent; }
.stTabs [data-baseweb="tab"] {
    background:#0d1b2a; border-radius:8px; color:#546e7a;
    border:1px solid #1e3a5f; padding:5px 13px; font-size:.81rem; }
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#0d3349,#0a2540) !important;
    color:#00e5ff !important; border-color:#00bcd4 !important; }

div[data-testid="metric-container"] {
    background:#0a1628; border:1px solid #1e3a5f;
    border-radius:8px; padding:9px 13px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TICKER PRESETS
# ═══════════════════════════════════════════════════════════════════════════
TICKER_GROUPS = {
    "🇮🇳 Indian Indices": {
        "Nifty 50":      "^NSEI",
        "Bank Nifty":    "^NSEBANK",
        "Sensex":        "^BSESN",
        "Nifty IT":      "^CNXIT",
        "Nifty Midcap":  "^NSEMDCP50",
        "Nifty Auto":    "^CNXAUTO",
        "Nifty Pharma":  "^CNXPHARMA",
    },
    "🇮🇳 NSE Top Stocks": {
        "Reliance":      "RELIANCE.NS",
        "TCS":           "TCS.NS",
        "HDFC Bank":     "HDFCBANK.NS",
        "Infosys":       "INFY.NS",
        "ICICI Bank":    "ICICIBANK.NS",
        "Kotak Bank":    "KOTAKBANK.NS",
        "Wipro":         "WIPRO.NS",
        "L&T":           "LT.NS",
        "Axis Bank":     "AXISBANK.NS",
        "SBI":           "SBIN.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Maruti":        "MARUTI.NS",
    },
    "₿ Crypto": {
        "Bitcoin":  "BTC-USD",
        "Ethereum": "ETH-USD",
        "BNB":      "BNB-USD",
        "Solana":   "SOL-USD",
        "XRP":      "XRP-USD",
        "Cardano":  "ADA-USD",
    },
    "💱 Forex": {
        "USD/INR":  "USDINR=X",
        "EUR/USD":  "EURUSD=X",
        "GBP/USD":  "GBPUSD=X",
        "USD/JPY":  "JPY=X",
        "AUD/USD":  "AUDUSD=X",
        "EUR/INR":  "EURINR=X",
    },
    "🥇 Commodities": {
        "Gold":          "GC=F",
        "Silver":        "SI=F",
        "Crude Oil WTI": "CL=F",
        "Natural Gas":   "NG=F",
        "Copper":        "HG=F",
    },
    "🌐 US Stocks": {
        "Apple":    "AAPL",
        "Tesla":    "TSLA",
        "NVIDIA":   "NVDA",
        "Microsoft":"MSFT",
        "Alphabet": "GOOGL",
        "Meta":     "META",
    },
    "✏️ Custom Ticker": {"Custom": "__CUSTOM__"},
}

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]
PERIODS    = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]

VALID_PERIODS = {
    "1m":  ["1d", "5d", "7d"],
    "5m":  ["1d", "5d", "7d", "1mo", "3mo"],
    "15m": ["1d", "5d", "7d", "1mo", "3mo"],
    "30m": ["1d", "5d", "7d", "1mo", "3mo"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "4h":  ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
}

# Multi-TF analysis default combos
MTF_COMBOS = [
    ("1d",  "1y",  "Daily (1y)"),
    ("4h",  "3mo", "4-Hour (3mo)"),
    ("1h",  "1mo", "1-Hour (1mo)"),
    ("15m", "5d",  "15-Min (5d)"),
]

SL_MAP = {
    "Wave Auto (Pivot Low/High)": "wave_auto",
    "0.5%": 0.005, "1%": 0.01, "1.5%": 0.015,
    "2%": 0.02,   "2.5%": 0.025, "3%": 0.03, "5%": 0.05,
}
TGT_MAP = {
    "Wave Auto (Fib 1.618 × W1)": "wave_auto",
    "R:R 1:1": 1.0, "R:R 1:1.5": 1.5, "R:R 1:2": 2.0,
    "R:R 1:2.5": 2.5, "R:R 1:3": 3.0,
    "Fib 1.618 × Wave 1": "fib_1618",
    "Fib 2.618 × Wave 1": "fib_2618",
}
SL_KEYS  = list(SL_MAP.keys())
TGT_KEYS = list(TGT_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "live_running":   False,
    "live_signals":   [],
    "live_log":       [],
    "last_bar_ts":    None,
    "live_position":  None,
    "live_pnl":       0.0,
    "live_trades":    [],
    "bt_results":     None,
    "opt_results":    None,
    "_scan_sig":      None,
    "_scan_df":       None,
    "_analysis_results": None,
    "_analysis_overall": "HOLD",
    "_analysis_symbol":  "",
    "applied_depth":     5,
    "applied_sl_lbl":    "Wave Auto (Pivot Low/High)",
    "applied_tgt_lbl":   "Wave Auto (Fib 1.618 × W1)",
    "best_cfg_applied":  False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════
# RATE-LIMIT SAFE FETCH
# ═══════════════════════════════════════════════════════════════════════════
_fetch_lock    = threading.Lock()
_last_fetch_ts = [0.0]


def fetch_ohlcv(symbol: str, interval: str, period: str,
                min_delay: float = 1.5) -> Optional[pd.DataFrame]:
    with _fetch_lock:
        gap = time.time() - _last_fetch_ts[0]
        if gap < min_delay:
            time.sleep(min_delay - gap)
        try:
            df = yf.download(symbol, interval=interval, period=period,
                             progress=False, auto_adjust=True)
            _last_fetch_ts[0] = time.time()
        except Exception:
            _last_fetch_ts[0] = time.time()
            return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["EMA_20"]  = c.ewm(span=20,  adjust=False).mean()
    df["EMA_50"]  = c.ewm(span=50,  adjust=False).mean()
    df["EMA_200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    if "Volume" in df.columns:
        df["Vol_Avg"] = df["Volume"].rolling(20).mean()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PIVOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════
def find_pivots(df: pd.DataFrame, depth: int = 5) -> list:
    H, L, n = df["High"].values, df["Low"].values, len(df)
    raw = []
    for i in range(depth, n - depth):
        wh = H[max(0, i - depth): i + depth + 1]
        wl = L[max(0, i - depth): i + depth + 1]
        if H[i] == wh.max():
            raw.append((i, float(H[i]), "H"))
        elif L[i] == wl.min():
            raw.append((i, float(L[i]), "L"))
    clean = []
    for p in raw:
        if not clean or clean[-1][2] != p[2]:
            clean.append(list(p))
        else:
            if p[2] == "H" and p[1] > clean[-1][1]:
                clean[-1] = list(p)
            elif p[2] == "L" and p[1] < clean[-1][1]:
                clean[-1] = list(p)
    return [tuple(x) for x in clean]


# ═══════════════════════════════════════════════════════════════════════════
# CORE SIGNAL  (same function for backtest + live = consistent results)
# ═══════════════════════════════════════════════════════════════════════════
def _blank(reason: str = "") -> dict:
    return {
        "signal": "HOLD", "entry_price": None, "sl": None, "target": None,
        "confidence": 0.0, "reason": reason or "No Elliott Wave pattern detected",
        "pattern": "—", "wave_pivots": None, "wave1_len": 0.0, "retracement": 0.0,
    }


def ew_signal(df: pd.DataFrame, depth: int = 5,
              sl_type="wave_auto", tgt_type="wave_auto") -> dict:
    n = len(df)
    if n < max(30, depth * 4):
        return _blank("Insufficient bars")
    pivots = find_pivots(df, depth)
    if len(pivots) < 4:
        return _blank("Not enough pivots — try smaller Pivot Depth")

    cur = float(df["Close"].iloc[-1])
    best, best_conf = _blank(), 0.0

    for i in range(len(pivots) - 2):
        p0, p1, p2 = pivots[i], pivots[i + 1], pivots[i + 2]
        bars_since  = n - 1 - p2[0]

        # BUY: Low→High→Low
        if p0[2] == "L" and p1[2] == "H" and p2[2] == "L":
            w1 = p1[1] - p0[1]
            if w1 <= 0: continue
            retr = (p1[1] - p2[1]) / w1
            if not (0.236 <= retr <= 0.886 and p2[1] > p0[1] and bars_since <= depth * 4):
                continue
            conf = 0.50
            if 0.382 <= retr <= 0.618: conf = 0.65
            if 0.50  <= retr <= 0.786: conf = 0.72
            if abs(retr - 0.618) < 0.04: conf = 0.87
            if abs(retr - 0.382) < 0.03: conf = 0.75
            if i + 3 < len(pivots) and pivots[i + 3][2] == "H":
                w3 = pivots[i + 3][1] - p2[1]
                if w3 > w1:                      conf = min(conf + 0.08, 0.95)
                if abs(w3 / w1 - 1.618) < 0.20: conf = min(conf + 0.05, 0.98)
            if conf <= best_conf: continue
            entry = cur
            sl_   = (p2[1] * 0.998) if sl_type == "wave_auto" \
                    else (entry * (1 - float(sl_type)))
            risk  = entry - sl_
            if risk <= 0: continue
            tgt_  = _calc_target(tgt_type, entry, "BUY", w1, risk)
            if tgt_ <= entry: continue
            best_conf = conf
            best = {
                "signal": "BUY", "entry_price": entry, "sl": sl_, "target": tgt_,
                "confidence": conf, "retracement": retr,
                "reason": f"Wave-2 bottom: {retr:.1%} retracement → Wave-3 up",
                "pattern": f"W2 Bottom ({retr:.1%})",
                "wave_pivots": [p0, p1, p2], "wave1_len": w1,
            }

        # SELL: High→Low→High
        elif p0[2] == "H" and p1[2] == "L" and p2[2] == "H":
            w1 = p0[1] - p1[1]
            if w1 <= 0: continue
            retr = (p2[1] - p1[1]) / w1
            if not (0.236 <= retr <= 0.886 and p2[1] < p0[1] and bars_since <= depth * 4):
                continue
            conf = 0.50
            if 0.382 <= retr <= 0.618: conf = 0.65
            if 0.50  <= retr <= 0.786: conf = 0.72
            if abs(retr - 0.618) < 0.04: conf = 0.87
            if abs(retr - 0.382) < 0.03: conf = 0.75
            if i + 3 < len(pivots) and pivots[i + 3][2] == "L":
                w3 = p2[1] - pivots[i + 3][1]
                if w3 > w1:                      conf = min(conf + 0.08, 0.95)
                if abs(w3 / w1 - 1.618) < 0.20: conf = min(conf + 0.05, 0.98)
            if conf <= best_conf: continue
            entry = cur
            sl_   = (p2[1] * 1.002) if sl_type == "wave_auto" \
                    else (entry * (1 + float(sl_type)))
            risk  = sl_ - entry
            if risk <= 0: continue
            tgt_  = _calc_target(tgt_type, entry, "SELL", w1, risk)
            if tgt_ >= entry: continue
            best_conf = conf
            best = {
                "signal": "SELL", "entry_price": entry, "sl": sl_, "target": tgt_,
                "confidence": conf, "retracement": retr,
                "reason": f"Wave-2 top: {retr:.1%} retracement → Wave-3 down",
                "pattern": f"W2 Top ({retr:.1%})",
                "wave_pivots": [p0, p1, p2], "wave1_len": w1,
            }
    return best


def _calc_target(tgt_type, entry: float, direction: str, w1: float, risk: float) -> float:
    sign = 1 if direction == "BUY" else -1
    if tgt_type in ("wave_auto", "fib_1618"):
        return entry + sign * w1 * 1.618
    elif tgt_type == "fib_2618":
        return entry + sign * w1 * 2.618
    elif isinstance(tgt_type, (int, float)):
        return entry + sign * risk * float(tgt_type)
    return entry + sign * risk * 2.0


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TF TEXT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def generate_mtf_summary(symbol: str, results: list, overall_sig: str) -> str:
    lines = [
        f"## 🌊 Elliott Wave Analysis — {symbol}",
        f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
    ]
    buy_c  = sum(1 for r in results if r["signal"]["signal"] == "BUY")
    sell_c = sum(1 for r in results if r["signal"]["signal"] == "SELL")
    hold_c = sum(1 for r in results if r["signal"]["signal"] == "HOLD")

    v_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(overall_sig, "⚪")
    lines.append(f"### {v_icon} Overall Verdict: **{overall_sig}**")
    lines.append(f"*{buy_c} BUY · {sell_c} SELL · {hold_c} HOLD across {len(results)} timeframes*\n")

    if overall_sig == "BUY":
        lines.append(
            "**📈 Bullish consensus across timeframes.** "
            "Elliott Wave structure points to an active Wave-3 impulse upward. "
            "Enter BUY positions on any pullback. Use Wave-2 pivot as stop loss."
        )
    elif overall_sig == "SELL":
        lines.append(
            "**📉 Bearish consensus across timeframes.** "
            "Elliott Wave structure points to a Wave-3 downward impulse. "
            "Enter SELL/SHORT on any minor bounce. Use Wave-2 pivot as stop loss."
        )
    else:
        lines.append(
            "**⚠️ No clear consensus.** Timeframes are conflicting or patterns incomplete. "
            "**Best action: Stay on sidelines.** Wait for a high-confidence (≥70%) signal."
        )

    lines.append("\n---\n### 📊 Timeframe Breakdown\n")

    for r in results:
        sig  = r["signal"]
        s    = sig["signal"]
        em   = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(s, "⚪")
        lines.append(f"#### {em} {r['tf_name']}")

        if s in ("BUY", "SELL"):
            retr = sig.get("retracement", 0)
            conf = sig["confidence"]
            ep   = sig["entry_price"]
            sl_  = sig["sl"]
            tgt_ = sig["target"]
            w1   = sig.get("wave1_len", 0)
            rr   = abs(tgt_ - ep) / abs(ep - sl_) if abs(ep - sl_) > 0 else 0

            lines.append(f"- **Signal**: {s} | **Pattern**: {sig['pattern']}")
            lines.append(f"- **Entry**: {ep:.2f}  |  **SL**: {sl_:.2f}  |  **Target**: {tgt_:.2f}")
            lines.append(f"- **R:R**: 1:{rr:.1f}  |  **Confidence**: {conf:.0%}")

            if w1 > 0:
                lines.append(
                    f"- **Wave-1** length: {w1:.2f} pts → "
                    f"Wave-3 projected at {w1*1.618:.2f} pts (1.618× Fib)"
                )

            if abs(retr - 0.618) < 0.04:
                lines.append("- ✨ **Golden Ratio (61.8%)** — strongest Elliott signal")
            elif abs(retr - 0.382) < 0.03:
                lines.append("- 💛 Shallow retracement (38.2%) — strong trend continuation")
            elif retr >= 0.786:
                lines.append("- ⚠️ Deep retracement (78.6%) — valid but watch invalidation")

            if conf >= 0.85:
                lines.append("- 🔥 **High confidence** — Wave-3 confirmation already visible")
            elif conf >= 0.70:
                lines.append("- ✅ **Medium-high confidence** — classic structure present")
            else:
                lines.append("- ⚠️ **Moderate confidence** — seek additional confirmation")

            # Indicator context
            df = r.get("df")
            if df is not None and len(df) > 55:
                try:
                    di = add_indicators(df)
                    cur_p = float(di["Close"].iloc[-1])
                    rsi   = float(di["RSI"].iloc[-1]) if not pd.isna(di["RSI"].iloc[-1]) else 50
                    ema20 = float(di["EMA_20"].iloc[-1]) if not pd.isna(di["EMA_20"].iloc[-1]) else cur_p
                    ema50 = float(di["EMA_50"].iloc[-1]) if not pd.isna(di["EMA_50"].iloc[-1]) else cur_p
                    macd_h = float(di["MACD"].iloc[-1] - di["MACD_Signal"].iloc[-1]) \
                             if not pd.isna(di["MACD"].iloc[-1]) else 0
                    lines.append(
                        f"- **RSI {rsi:.1f}**: "
                        + ("Overbought — caution on longs" if rsi > 70
                           else "Oversold — supportive for longs" if rsi < 30
                           else "Neutral")
                    )
                    lines.append(
                        f"- **EMA trend**: "
                        + ("Uptrend (price > EMA20 > EMA50)" if cur_p > ema20 > ema50
                           else "Downtrend (price < EMA20 < EMA50)" if cur_p < ema20 < ema50
                           else "Sideways / mixed EMAs")
                    )
                    lines.append(
                        f"- **MACD**: "
                        + ("Positive histogram — bullish momentum" if macd_h > 0
                           else "Negative histogram — bearish momentum")
                    )
                except Exception:
                    pass

            action = (
                f"**BUY** near ₹{ep:.2f}. SL at ₹{sl_:.2f}. Target ₹{tgt_:.2f}."
                if s == "BUY" else
                f"**SELL** near ₹{ep:.2f}. SL at ₹{sl_:.2f}. Target ₹{tgt_:.2f}."
            )
            lines.append(f"\n📋 **Action**: {action}")
        else:
            lines.append(f"- {sig.get('reason','No clear pattern')}")
            lines.append("- **Action**: Wait for Wave-2 pivot to complete before entering.")
        lines.append("")

    lines += [
        "---",
        "### 📚 Elliott Wave Quick Reference",
        "| Wave | What Happens | App Action |",
        "|------|-------------|------------|",
        "| W1 ↑ | First impulse up | Detected as pivot |",
        "| **W2 ↓** | **Retracement 38–79%** | **Signal fires here** |",
        "| **W3 ↑** | **Strongest move** | **Ride this wave** |",
        "| W4 ↓ | Minor pullback | Partial exit |",
        "| W5 ↑ | Final extension | Full exit |",
        "\n> ⚠️ *Not financial advice. Always use Stop Loss.*",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, depth: int = 5,
                 sl_type="wave_auto", tgt_type="wave_auto",
                 capital: float = 100_000.0) -> dict:
    MIN_BARS = max(30, depth * 4)
    if len(df) < MIN_BARS + 10:
        return {"error": f"Need ≥{MIN_BARS+10} bars. Use longer period or smaller depth."}

    trades, equity_curve = [], [capital]
    equity, pos = capital, None

    for i in range(MIN_BARS, len(df) - 1):
        bar_df   = df.iloc[:i + 1]
        next_bar = df.iloc[i + 1]
        hi_i, lo_i = float(df.iloc[i]["High"]), float(df.iloc[i]["Low"])

        if pos:
            exit_p, exit_r = None, None
            if pos["type"] == "BUY":
                if lo_i <= pos["sl"]:       exit_p, exit_r = pos["sl"],     "SL"
                elif hi_i >= pos["target"]: exit_p, exit_r = pos["target"], "Target"
            else:
                if hi_i >= pos["sl"]:       exit_p, exit_r = pos["sl"],     "SL"
                elif lo_i <= pos["target"]: exit_p, exit_r = pos["target"], "Target"
            if exit_p is not None:
                qty = pos["qty"]
                pnl = (exit_p - pos["entry"]) * qty if pos["type"] == "BUY" \
                      else (pos["entry"] - exit_p) * qty
                equity += pnl
                equity_curve.append(equity)
                trades.append({
                    "Entry Time": pos["entry_time"], "Exit Time": df.index[i],
                    "Type": pos["type"],
                    "Entry": round(pos["entry"], 2), "Exit": round(exit_p, 2),
                    "SL": round(pos["sl"], 2), "Target": round(pos["target"], 2),
                    "Exit Reason": exit_r,
                    "PnL Rs": round(pnl, 2),
                    "PnL %": round(pnl / (pos["entry"] * qty) * 100, 2),
                    "Equity Rs": round(equity, 2),
                    "Bars Held": i - pos["entry_bar"],
                    "Confidence": round(pos["conf"], 2),
                })
                pos = None

        if pos is None:
            sig = ew_signal(bar_df, depth, sl_type, tgt_type)
            if sig["signal"] in ("BUY", "SELL"):
                ep  = float(next_bar["Open"])
                w1  = sig.get("wave1_len", ep * 0.02) or (ep * 0.02)
                sl_ = sig["sl"] if sl_type == "wave_auto" else (
                    ep * (1 - float(sl_type)) if sig["signal"] == "BUY"
                    else ep * (1 + float(sl_type))
                )
                risk = abs(ep - sl_)
                if risk <= 0: continue
                tgt_ = _calc_target(tgt_type, ep, sig["signal"], w1, risk)
                if sig["signal"] == "BUY"  and tgt_ <= ep: continue
                if sig["signal"] == "SELL" and tgt_ >= ep: continue
                qty = max(1, int(equity * 0.95 / ep))
                pos = {
                    "type": sig["signal"], "entry": ep, "sl": sl_, "target": tgt_,
                    "entry_bar": i + 1, "entry_time": df.index[i + 1],
                    "qty": qty, "conf": sig["confidence"],
                }

    if pos:
        ep2 = float(df["Close"].iloc[-1])
        qty = pos["qty"]
        pnl = (ep2 - pos["entry"]) * qty if pos["type"] == "BUY" \
              else (pos["entry"] - ep2) * qty
        equity += pnl
        trades.append({
            "Entry Time": pos["entry_time"], "Exit Time": df.index[-1],
            "Type": pos["type"],
            "Entry": round(pos["entry"], 2), "Exit": round(ep2, 2),
            "SL": round(pos["sl"], 2), "Target": round(pos["target"], 2),
            "Exit Reason": "Open@End",
            "PnL Rs": round(pnl, 2),
            "PnL %": round(pnl / (pos["entry"] * qty) * 100, 2),
            "Equity Rs": round(equity, 2),
            "Bars Held": len(df) - 1 - pos["entry_bar"],
            "Confidence": round(pos["conf"], 2),
        })

    if not trades:
        return {"error": "No trades. Try smaller Pivot Depth, longer Period, or different SL/Target.",
                "equity_curve": equity_curve}

    tdf  = pd.DataFrame(trades)
    wins = tdf[tdf["PnL Rs"] > 0]
    loss = tdf[tdf["PnL Rs"] <= 0]
    ntot = len(tdf)
    wr   = len(wins) / ntot * 100 if ntot else 0
    pf   = abs(wins["PnL Rs"].sum() / loss["PnL Rs"].sum()) \
           if len(loss) and loss["PnL Rs"].sum() != 0 else 9999.0
    eq_arr = np.array(equity_curve)
    peak   = np.maximum.accumulate(eq_arr)
    mdd    = float(((eq_arr - peak) / peak * 100).min())
    rets   = tdf["PnL %"].values
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) \
             if len(rets) > 1 and rets.std() != 0 else 0.0

    return {
        "trades": tdf, "equity_curve": equity_curve,
        "metrics": {
            "Total Trades": ntot, "Win Rate %": round(wr, 1),
            "Profit Factor": round(pf, 2),
            "Total Return %": round((equity - capital) / capital * 100, 2),
            "Final Equity Rs": round(equity, 2),
            "Max Drawdown %": round(mdd, 2), "Sharpe Ratio": round(sharpe, 2),
            "Avg Win Rs": round(float(wins["PnL Rs"].mean()), 2) if len(wins) else 0.0,
            "Avg Loss Rs": round(float(loss["PnL Rs"].mean()), 2) if len(loss) else 0.0,
            "Wins": len(wins), "Losses": len(loss),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization(df: pd.DataFrame, capital: float = 100_000.0) -> pd.DataFrame:
    DEPTHS   = [3, 5, 7, 10]
    SL_OPTS  = [0.01, 0.02, 0.03, "wave_auto"]
    TGT_OPTS = [1.5, 2.0, 3.0, "wave_auto", "fib_1618"]
    combos   = list(itertools.product(DEPTHS, SL_OPTS, TGT_OPTS))
    prog     = st.progress(0, text="Optimizing…")
    rows     = []
    for idx, (dep, sl, tgt) in enumerate(combos):
        r = run_backtest(df, depth=dep, sl_type=sl, tgt_type=tgt, capital=capital)
        if "metrics" in r:
            m = r["metrics"]
            rows.append({
                "Depth": dep, "SL": str(sl), "Target": str(tgt),
                "Trades": m["Total Trades"], "Win %": m["Win Rate %"],
                "Return %": m["Total Return %"], "PF": m["Profit Factor"],
                "Max DD %": m["Max Drawdown %"], "Sharpe": m["Sharpe Ratio"],
            })
        prog.progress((idx + 1) / len(combos), text=f"Combo {idx+1}/{len(combos)}…")
    prog.empty()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["Score"] = (
        out["Return %"].clip(lower=0)
        * (out["Win %"] / 100)
        * out["PF"].clip(upper=10)
        / (out["Max DD %"].abs() + 1)
    )
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def sl_label_from_val(val) -> str:
    for k, v in SL_MAP.items():
        if str(v) == str(val): return k
    return SL_KEYS[0]


def tgt_label_from_val(val) -> str:
    for k, v in TGT_MAP.items():
        if str(v) == str(val): return k
    try:
        fv = float(val)
        for k, v in TGT_MAP.items():
            if isinstance(v, float) and abs(v - fv) < 0.01: return k
    except Exception:
        pass
    return TGT_KEYS[0]


# ═══════════════════════════════════════════════════════════════════════════
# DHAN API
# ═══════════════════════════════════════════════════════════════════════════
class DhanAPI:
    BASE = "https://api.dhan.co"

    def __init__(self, cid, tok):
        self.cid  = cid
        self.hdrs = {"Content-Type": "application/json", "access-token": tok}

    def place_order(self, sec_id, segment, txn, qty,
                    order_type="MARKET", price=0.0, product="INTRADAY"):
        try:
            r = requests.post(f"{self.BASE}/orders", headers=self.hdrs, timeout=10, json={
                "dhanClientId": self.cid, "transactionType": txn,
                "exchangeSegment": segment, "productType": product,
                "orderType": order_type, "validity": "DAY",
                "securityId": sec_id, "quantity": qty, "price": price,
            })
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def fund_limit(self):
        try:
            return requests.get(f"{self.BASE}/fundlimit", headers=self.hdrs, timeout=10).json()
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# LIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════
def live_loop(symbol, interval, period, depth, sl_type, tgt_type,
              dhan_on, dhan_api, sec_id, live_qty):
    def log(msg, lvl="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        if "live_log" in st.session_state:
            st.session_state.live_log.append(f"[{ts}][{lvl}] {msg}")
            st.session_state.live_log = st.session_state.live_log[-120:]

    POLL = {"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":3600,"1d":3600,"1wk":3600}
    sleep_s = min(POLL.get(interval, 60), 60)
    log(f"🚀 Started | {symbol} | {interval} | {period}")

    while st.session_state.get("live_running", False):
        try:
            log("📡 Fetching (1.5 s rate-limit)…")
            df = fetch_ohlcv(symbol, interval, period, min_delay=1.5)
            if df is None or len(df) < 35:
                log("⚠️  Insufficient data", "WARN")
                time.sleep(sleep_s)
                continue

            df_closed = df.iloc[:-1]
            latest_ts = str(df_closed.index[-1])
            cur_price = float(df["Close"].iloc[-1])

            if st.session_state.get("last_bar_ts") == latest_ts:
                log(f"⏭  Bar …{latest_ts[-8:]} already done — waiting for new candle")
                time.sleep(sleep_s)
                continue
            st.session_state.last_bar_ts = latest_ts

            pos = st.session_state.live_position
            if pos:
                hit = None
                if pos["type"] == "BUY":
                    if cur_price <= pos["sl"]:       hit = (pos["sl"],     "SL Hit")
                    elif cur_price >= pos["target"]: hit = (pos["target"], "Target Hit")
                else:
                    if cur_price >= pos["sl"]:       hit = (pos["sl"],     "SL Hit")
                    elif cur_price <= pos["target"]: hit = (pos["target"], "Target Hit")
                if hit:
                    ep_, rsn = hit
                    qty_ = pos["qty"]
                    pnl  = (ep_ - pos["entry"]) * qty_ if pos["type"] == "BUY" \
                           else (pos["entry"] - ep_) * qty_
                    st.session_state.live_pnl += pnl
                    st.session_state.live_trades.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Symbol": symbol, "TF": interval, "Period": period,
                        "Type": pos["type"],
                        "Entry": round(pos["entry"], 2), "Exit": round(ep_, 2),
                        "SL": round(pos["sl"], 2), "Target": round(pos["target"], 2),
                        "Qty": qty_, "PnL Rs": round(pnl, 2), "Reason": rsn,
                    })
                    if dhan_on and dhan_api:
                        xt = "SELL" if pos["type"] == "BUY" else "BUY"
                        log(f"📤 Dhan exit: {dhan_api.place_order(sec_id,'NSE_EQ',xt,qty_)}")
                    em = "✅" if "Target" in rsn else "❌"
                    log(f"{em} {pos['type']} closed @ {ep_:.2f} | {rsn} | Rs{pnl:.2f}")
                    st.session_state.live_position = None
                    pos = None

            sig = ew_signal(df_closed, depth, sl_type, tgt_type)
            if pos is None and sig["signal"] in ("BUY", "SELL"):
                ep   = cur_price
                w1   = sig.get("wave1_len", ep * 0.02) or (ep * 0.02)
                sl_  = sig["sl"]
                risk = abs(ep - sl_)
                if risk > 0:
                    tgt_ = _calc_target(tgt_type, ep, sig["signal"], w1, risk)
                    if (sig["signal"] == "BUY" and tgt_ > ep) or \
                       (sig["signal"] == "SELL" and tgt_ < ep):
                        st.session_state.live_position = {
                            "type": sig["signal"], "entry": ep, "sl": sl_,
                            "target": tgt_, "qty": live_qty,
                            "entry_time": datetime.now().strftime("%H:%M:%S"),
                            "symbol": symbol,
                        }
                        st.session_state.live_signals.append({
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Bar TS": latest_ts, "Symbol": symbol,
                            "TF": interval, "Period": period,
                            "Signal": sig["signal"],
                            "Entry": round(ep, 2), "SL": round(sl_, 2),
                            "Target": round(tgt_, 2),
                            "Conf": f"{sig['confidence']:.0%}",
                            "Pattern": sig["pattern"],
                        })
                        if dhan_on and dhan_api:
                            log(f"📤 Dhan: {dhan_api.place_order(sec_id,'NSE_EQ',sig['signal'],live_qty)}")
                        em = "🟢" if sig["signal"] == "BUY" else "🔴"
                        log(f"{em} {sig['signal']} @ {ep:.2f} | SL {sl_:.2f} | T {tgt_:.2f} | {sig['confidence']:.0%}")
            else:
                log(f"⏸  HOLD @ {cur_price:.2f} | {sig['reason']}")

            time.sleep(sleep_s)
        except Exception as exc:
            log(f"💥 {exc}", "ERROR")
            time.sleep(sleep_s)

    log("🛑 Stopped")


# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def chart_waves(df: pd.DataFrame, pivots: list,
                sig: Optional[dict] = None,
                trades: Optional[pd.DataFrame] = None,
                symbol: str = "", tf_label: str = "") -> go.Figure:
    sig    = sig or _blank()
    df_ind = add_indicators(df)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20], vertical_spacing=0.02,
        subplot_titles=("", "Volume", "RSI"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
    ), row=1, col=1)

    for col, clr, nm in [("EMA_20","#ffb300","EMA20"),
                         ("EMA_50","#ab47bc","EMA50"),
                         ("EMA_200","#ef5350","EMA200")]:
        if col in df_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind[col], mode="lines",
                line=dict(color=clr, width=1.2), name=nm, opacity=0.7,
            ), row=1, col=1)

    if "Volume" in df.columns:
        vcol = ["#26a69a" if c >= o else "#ef5350"
                for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                             marker_color=vcol, opacity=0.4, name="Vol",
                             showlegend=False), row=2, col=1)

    if "RSI" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"],
                                 mode="lines", line=dict(color="#00e5ff", width=1.5),
                                 name="RSI", showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line=dict(dash="dot", color="#ef5350", width=1), row=3, col=1)
        fig.add_hline(y=30, line=dict(dash="dot", color="#4caf50", width=1), row=3, col=1)

    vp = [p for p in pivots if p[0] < len(df)]
    if vp:
        fig.add_trace(go.Scatter(
            x=[df.index[p[0]] for p in vp], y=[p[1] for p in vp],
            mode="lines+markers",
            line=dict(color="rgba(255,180,0,.5)", width=1.5, dash="dot"),
            marker=dict(
                size=7,
                color=["#4caf50" if p[2] == "L" else "#f44336" for p in vp],
                symbol=["triangle-up" if p[2] == "L" else "triangle-down" for p in vp],
            ),
            name="ZigZag",
        ), row=1, col=1)

    wp = sig.get("wave_pivots")
    if wp:
        valid_wp = [p for p in wp if p[0] < len(df)]
        if valid_wp:
            clr  = "#00e5ff" if sig["signal"] == "BUY" else "#ff4081"
            lbls = ["W0","W1","W2"][:len(valid_wp)]
            fig.add_trace(go.Scatter(
                x=[df.index[p[0]] for p in valid_wp],
                y=[p[1] for p in valid_wp],
                mode="lines+markers+text",
                line=dict(color=clr, width=2.5),
                marker=dict(size=12, color=clr),
                text=lbls, textposition="top center",
                textfont=dict(color=clr, size=12, family="Share Tech Mono"),
                name="EW Pattern",
            ), row=1, col=1)

    if sig["signal"] in ("BUY", "SELL"):
        sc = "#4caf50" if sig["signal"] == "BUY" else "#f44336"
        ss = "triangle-up" if sig["signal"] == "BUY" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[df["Close"].iloc[-1]],
            mode="markers",
            marker=dict(size=20, color=sc, symbol=ss,
                        line=dict(color="white", width=1.5)),
            name=f"▶ {sig['signal']}",
        ), row=1, col=1)
        if sig.get("sl"):
            fig.add_hline(y=sig["sl"], line=dict(dash="dash", color="#ff7043", width=1.5),
                          annotation_text="  SL", annotation_position="right", row=1, col=1)
        if sig.get("target"):
            fig.add_hline(y=sig["target"], line=dict(dash="dash", color="#66bb6a", width=1.5),
                          annotation_text="  Target", annotation_position="right", row=1, col=1)

    if trades is not None and not trades.empty:
        for ttype, sym_, clr in [("BUY","triangle-up","#4caf50"),("SELL","triangle-down","#f44336")]:
            sub = trades[trades["Type"] == ttype]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Entry Time"], y=sub["Entry"], mode="markers",
                    marker=dict(size=9, color=clr, symbol=sym_,
                                line=dict(color="white", width=0.8)),
                    name=f"{ttype} Entry",
                ), row=1, col=1)
        for rsn_, sym_, clr in [("Target","circle","#66bb6a"),("SL","x","#ef5350")]:
            sub = trades[trades["Exit Reason"].str.contains(rsn_, na=False)]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Exit Time"], y=sub["Exit"], mode="markers",
                    marker=dict(size=7, color=clr, symbol=sym_),
                    name=f"Exit({rsn_})", visible="legendonly",
                ), row=1, col=1)

    title_str = f"🌊 {symbol}" + (f" · {tf_label}" if tf_label else "")
    fig.update_layout(
        title=dict(text=title_str, font=dict(size=14, color="#00e5ff")),
        template="plotly_dark", height=580,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#06101a", paper_bgcolor="#06101a",
        font=dict(color="#b0bec5", family="Exo 2"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        margin=dict(l=10, r=70, t=50, b=10),
    )
    return fig


def chart_equity(equity_curve: list) -> go.Figure:
    eq = np.array(equity_curve, dtype=float)
    pk = np.maximum.accumulate(eq)
    dd = (eq - pk) / pk * 100
    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=eq, mode="lines", name="Equity",
                             line=dict(color="#00bcd4", width=2),
                             fill="tozeroy", fillcolor="rgba(0,188,212,.07)"), row=1, col=1)
    fig.add_trace(go.Scatter(y=dd, mode="lines", name="Drawdown %",
                             line=dict(color="#f44336", width=1.5),
                             fill="tozeroy", fillcolor="rgba(244,67,54,.12)"), row=2, col=1)
    fig.add_hline(y=0, line=dict(dash="dot", color="#546e7a", width=1), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=360,
                      plot_bgcolor="#06101a", paper_bgcolor="#06101a",
                      font=dict(color="#b0bec5", family="Exo 2"),
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_opt_scatter(opt_df: pd.DataFrame) -> go.Figure:
    # colorbar title uses dict form with font key (Plotly v5+ requirement)
    fig = go.Figure(go.Scatter(
        x=opt_df["Max DD %"].abs(), y=opt_df["Return %"],
        mode="markers",
        marker=dict(
            size=(opt_df["Win %"] / 5).clip(lower=4),
            color=opt_df["Score"],
            colorscale="Plasma", showscale=True,
            colorbar=dict(
                title=dict(text="Score", font=dict(color="#b0bec5", size=12)),
                tickfont=dict(color="#b0bec5"),
            ),
            line=dict(color="rgba(255,255,255,.2)", width=0.5),
        ),
        text=[f"Depth={r.Depth} SL={r.SL} T={r.Target}" for _, r in opt_df.iterrows()],
        hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%  MaxDD %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Return vs Max Drawdown  (bubble size = Win Rate)",
                   font=dict(size=13, color="#00e5ff")),
        xaxis_title="Max Drawdown %", yaxis_title="Total Return %",
        template="plotly_dark", height=400,
        plot_bgcolor="#06101a", paper_bgcolor="#06101a",
        font=dict(color="#b0bec5", family="Exo 2"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  STREAMLIT UI  ███
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-hdr">
  <h1>🌊 Elliott Wave Algo Trader v4.0</h1>
  <p>Wave Analysis · Backtest · Live Trading · Optimization · Dhan Integration  |  1.5s rate-limit safe</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Instrument")
    group_sel = st.selectbox("Category", list(TICKER_GROUPS.keys()), index=0)
    group_map = TICKER_GROUPS[group_sel]

    if group_sel == "✏️ Custom Ticker":
        symbol = st.text_input("Enter Yahoo Finance ticker", "^NSEI",
                               help="^NSEI  RELIANCE.NS  BTC-USD  GC=F  USDINR=X")
    else:
        ticker_name = st.selectbox("Instrument", list(group_map.keys()))
        symbol      = group_map[ticker_name]
        st.caption(f"Yahoo Finance: `{symbol}`")

    st.markdown("---")

    if st.session_state.best_cfg_applied:
        st.markdown("""
        <div style="background:rgba(0,229,255,.08);border:1px solid #00bcd4;
        border-radius:8px;padding:8px 11px;font-size:.82rem;margin-bottom:8px">
        ✨ <b style="color:#00e5ff">Optimized Config Active</b><br>
        <small style="color:#78909c">Parameters from optimization result</small>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    interval = c1.selectbox("⏱ Timeframe", TIMEFRAMES, index=6)
    vp_list  = VALID_PERIODS.get(interval, PERIODS)
    period   = c2.selectbox("📅 Period", vp_list, index=min(4, len(vp_list) - 1))

    st.markdown("---")
    st.markdown("### 🌊 Elliott Wave")
    depth = st.slider("Pivot Depth", 2, 15, st.session_state.applied_depth,
                      help="Lower = more (noisier) signals. Higher = fewer (cleaner).")

    st.markdown("---")
    st.markdown("### 🛡️ Risk Management")
    sl_idx  = SL_KEYS.index(st.session_state.applied_sl_lbl) \
              if st.session_state.applied_sl_lbl in SL_KEYS else 0
    tgt_idx = TGT_KEYS.index(st.session_state.applied_tgt_lbl) \
              if st.session_state.applied_tgt_lbl in TGT_KEYS else 0
    sl_lbl  = st.selectbox("Stop Loss", SL_KEYS,  index=sl_idx)
    tgt_lbl = st.selectbox("Target",    TGT_KEYS, index=tgt_idx)
    sl_val  = SL_MAP[sl_lbl]
    tgt_val = TGT_MAP[tgt_lbl]
    capital = st.number_input("💰 Capital (Rs)", 10_000, 50_000_000, 100_000, 10_000)

    st.markdown("---")
    st.markdown("### 🏦 Dhan Brokerage")
    dhan_on = st.checkbox("Enable Dhan Integration", value=False)
    dhan_api, sec_id, live_qty = None, "1333", 1
    if dhan_on:
        d_cid    = st.text_input("Client ID")
        d_tok    = st.text_input("Access Token", type="password")
        sec_id   = st.text_input("Security ID", "1333")
        live_qty = st.number_input("Order Qty", 1, 100_000, 1)
        if d_cid and d_tok:
            dhan_api = DhanAPI(d_cid, d_tok)
            if st.button("🔌 Test Dhan"):
                st.json(dhan_api.fund_limit())
        else:
            st.info("Enter credentials to activate")

    st.markdown("---")
    st.caption(f"⚡ Rate-limit: **1.5 s** per fetch")
    st.caption(f"📌 `{symbol}` · `{interval}` · `{period}`")


# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────
t_analysis, t_live, t_bt, t_opt, t_help = st.tabs([
    "🔭  Wave Analysis",
    "🔴  Live Trading",
    "📊  Backtest",
    "🔬  Optimization",
    "❓  Help",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — WAVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with t_analysis:
    st.markdown("### 🔭 Multi-Timeframe Elliott Wave Analysis")
    st.caption("Scans multiple timeframes, plots wave structures + indicators, and gives buy/sell/hold recommendations with detailed reasoning.")

    ac1, ac2, ac3 = st.columns([1.2, 1, 2.4])
    with ac1:
        run_analysis = st.button("🔭 Run Full Analysis", type="primary", use_container_width=True)
    with ac2:
        custom_tf_only = st.checkbox("Sidebar TF only", value=False)
    with ac3:
        st.caption(f"Scanning: {'sidebar TF' if custom_tf_only else 'Daily · 4H · 1H · 15M'} for `{symbol}`")

    if run_analysis:
        if custom_tf_only:
            scan_combos = [(interval, period, f"{interval.upper()} · {period}")]
        else:
            scan_combos = [(tf, per, nm) for tf, per, nm in MTF_COMBOS
                           if per in VALID_PERIODS.get(tf, [])]

        results = []
        prog = st.progress(0, text="Scanning timeframes…")
        for idx, (tf, per, nm) in enumerate(scan_combos):
            prog.progress((idx + 1) / len(scan_combos), text=f"Fetching {nm}…")
            df_a = fetch_ohlcv(symbol, tf, per, min_delay=1.5)
            if df_a is not None and len(df_a) >= 35:
                sig_a = ew_signal(df_a.iloc[:-1], depth, sl_val, tgt_val)
                results.append({
                    "tf_name": nm, "interval": tf, "period": per,
                    "signal": sig_a, "df": df_a,
                    "pivots": find_pivots(df_a.iloc[:-1], depth),
                })
            else:
                results.append({
                    "tf_name": nm, "interval": tf, "period": per,
                    "signal": _blank(f"No data for {tf}/{per}"),
                    "df": None, "pivots": [],
                })
        prog.empty()

        buy_score  = sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"] == "BUY")
        sell_score = sum(r["signal"]["confidence"] for r in results if r["signal"]["signal"] == "SELL")
        if   buy_score > sell_score and buy_score > 0.5:  overall = "BUY"
        elif sell_score > buy_score and sell_score > 0.5: overall = "SELL"
        else:                                              overall = "HOLD"

        st.session_state["_analysis_results"] = results
        st.session_state["_analysis_overall"]  = overall
        st.session_state["_analysis_symbol"]   = symbol

    ar = st.session_state.get("_analysis_results")
    if ar:
        overall = st.session_state.get("_analysis_overall", "HOLD")
        a_sym   = st.session_state.get("_analysis_symbol", symbol)
        v_colors = {"BUY":"#4caf50","SELL":"#f44336","HOLD":"#ffb300"}
        v_bg     = {"BUY":"rgba(76,175,80,.10)","SELL":"rgba(244,67,54,.10)","HOLD":"rgba(255,179,0,.10)"}
        v_icons  = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}

        st.markdown(f"""
        <div style="background:{v_bg[overall]};border:2px solid {v_colors[overall]};
        border-radius:12px;padding:15px 22px;margin-bottom:12px;text-align:center">
        <span style="font-size:1.6rem;color:{v_colors[overall]};font-weight:700">
          {v_icons[overall]} Overall: {overall}
        </span><br>
        <span style="color:#78909c;font-size:.87rem">{a_sym} — Multi-Timeframe Elliott Wave Consensus</span>
        </div>""", unsafe_allow_html=True)

        n_r = len(ar)
        tf_cols = st.columns(min(n_r, 4))
        for i, r in enumerate(ar):
            with tf_cols[i % 4]:
                s   = r["signal"]["signal"]
                c_  = r["signal"]["confidence"]
                pat = r["signal"]["pattern"]
                sc  = v_colors.get(s, "#546e7a")
                em  = v_icons.get(s, "⚪")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.03);border:1px solid #1e3a5f;
                border-radius:8px;padding:10px 12px;text-align:center;margin-bottom:4px">
                <div style="font-size:.78rem;color:#546e7a">{r['tf_name']}</div>
                <div style="font-size:1.2rem;color:{sc};font-weight:700">{em} {s}</div>
                <div style="font-size:.76rem;color:#78909c">{pat}</div>
                <div style="font-size:.8rem;color:#00bcd4">{c_:.0%}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        for r in ar:
            if r["df"] is not None:
                s_   = r["signal"]["signal"]
                exp_ = s_ != "HOLD"
                with st.expander(
                    f"📈 {r['tf_name']}  —  {s_}  ({r['signal']['confidence']:.0%} confidence)",
                    expanded=exp_,
                ):
                    st.plotly_chart(
                        chart_waves(r["df"], r["pivots"], r["signal"],
                                    symbol=a_sym, tf_label=r["tf_name"]),
                        use_container_width=True,
                    )
                    if s_ in ("BUY", "SELL"):
                        sig_ = r["signal"]
                        ep, sl_, tgt_ = sig_["entry_price"], sig_["sl"], sig_["target"]
                        rr = abs(tgt_ - ep) / abs(ep - sl_) if abs(ep - sl_) > 0 else 0
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Entry",  f"{ep:.2f}")
                        mc2.metric("SL",     f"{sl_:.2f}", delta=f"-{abs(ep-sl_)/ep*100:.1f}%", delta_color="inverse")
                        mc3.metric("Target", f"{tgt_:.2f}", delta=f"+{abs(tgt_-ep)/ep*100:.1f}%")
                        mc4.metric("R:R",    f"1:{rr:.1f}")

        st.markdown("---")
        st.markdown("### 📋 Detailed Analysis & Recommendations")
        summary = generate_mtf_summary(a_sym, ar, overall)
        st.markdown(summary)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#37474f">
        <h3>Click 🔭 Run Full Analysis to begin</h3>
        <p>Automatically scans Daily · 4H · 1H · 15M timeframes<br>
        Plots Elliott Wave structure · RSI · EMA indicators<br>
        Generates detailed text recommendations with reasons</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ═══════════════════════════════════════════════════════════════════════════
with t_live:
    lc, rc = st.columns([1, 2.3], gap="medium")

    with lc:
        st.markdown("### 🎮 Controls")
        b1, b2 = st.columns(2)
        with b1:
            if not st.session_state.live_running:
                if st.button("▶ Start", type="primary", use_container_width=True):
                    st.session_state.live_running = True
                    st.session_state.live_log     = []
                    st.session_state.last_bar_ts  = None
                    threading.Thread(
                        target=live_loop,
                        args=(symbol, interval, period, depth,
                              sl_val, tgt_val, dhan_on, dhan_api, sec_id, live_qty),
                        daemon=True,
                    ).start()
                    st.rerun()
            else:
                if st.button("⏹ Stop", type="secondary", use_container_width=True):
                    st.session_state.live_running = False
                    st.rerun()
        with b2:
            if st.button("🔄 Reset", use_container_width=True,
                         help="Clears all state + bar timestamp tracker"):
                for k, v in _DEFAULTS.items():
                    st.session_state[k] = v
                st.success("State cleared ✓")
                time.sleep(0.3)
                st.rerun()

        if st.session_state.live_running:
            st.success("🟢 **LIVE — RUNNING**")
        else:
            st.warning("⚫ **STOPPED**")

        pos = st.session_state.live_position
        if pos:
            clr = "#4caf50" if pos["type"] == "BUY" else "#f44336"
            st.markdown(f"""
            <div class="pos-card">
            📍 <b style="color:{clr}">{pos['type']} OPEN</b><br>
            Entry : <b>{pos['entry']:.2f}</b> · Qty: <b>{pos['qty']}</b><br>
            SL    : <b style="color:#ff7043">{pos['sl']:.2f}</b> ·
            Target: <b style="color:#66bb6a">{pos['target']:.2f}</b><br>
            Since : {pos['entry_time']}
            </div>""", unsafe_allow_html=True)
        else:
            st.caption("No open position")

        st.markdown(f"""
        <div class="info-box" style="margin-top:8px">
        💰 <b>P&L</b>    : Rs{st.session_state.live_pnl:,.2f}<br>
        📡 <b>Signals</b>: {len(st.session_state.live_signals)}<br>
        🏁 <b>Trades</b> : {len(st.session_state.live_trades)}<br>
        🕒 <b>Last Bar</b>: {str(st.session_state.last_bar_ts or "—")[-19:]}
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔍 Manual Scan")
        if st.button("Scan Signal Now", use_container_width=True):
            with st.spinner("Fetching (1.5 s delay)…"):
                df_ = fetch_ohlcv(symbol, interval, period, min_delay=1.5)
            if df_ is not None and len(df_) >= 35:
                sig_ = ew_signal(df_.iloc[:-1], depth, sl_val, tgt_val)
                st.session_state._scan_sig = sig_
                st.session_state._scan_df  = df_
            else:
                st.error("⚠️ No data or too few bars")

        sc_ = st.session_state._scan_sig
        if sc_:
            s = sc_["signal"]
            if s == "BUY":
                st.markdown(f"""
                <div class="sig-buy">
                <div style="font-size:1.35rem;color:#4caf50;font-weight:700">🟢 BUY</div>
                <div style="font-size:.86rem;margin-top:5px">
                Entry <b>{sc_['entry_price']:.2f}</b><br>
                SL <b style="color:#ff7043">{sc_['sl']:.2f}</b> · Target <b style="color:#66bb6a">{sc_['target']:.2f}</b><br>
                Confidence <b>{sc_['confidence']:.0%}</b> · {sc_['pattern']}<br>
                <small style="color:#78909c">{sc_['reason']}</small>
                </div></div>""", unsafe_allow_html=True)
            elif s == "SELL":
                st.markdown(f"""
                <div class="sig-sell">
                <div style="font-size:1.35rem;color:#f44336;font-weight:700">🔴 SELL</div>
                <div style="font-size:.86rem;margin-top:5px">
                Entry <b>{sc_['entry_price']:.2f}</b><br>
                SL <b style="color:#ff7043">{sc_['sl']:.2f}</b> · Target <b style="color:#66bb6a">{sc_['target']:.2f}</b><br>
                Confidence <b>{sc_['confidence']:.0%}</b> · {sc_['pattern']}<br>
                <small style="color:#78909c">{sc_['reason']}</small>
                </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sig-hold">
                <div style="font-size:1.1rem;color:#78909c;font-weight:600">⏸ HOLD</div>
                <div style="font-size:.84rem;color:#546e7a;margin-top:4px">{sc_['reason']}</div>
                </div>""", unsafe_allow_html=True)

    with rc:
        st.markdown("### 📊 Live Dashboard")
        df_s = st.session_state._scan_df
        if df_s is not None:
            piv_ = find_pivots(df_s.iloc[:-1], depth)
            st.plotly_chart(
                chart_waves(df_s, piv_, st.session_state._scan_sig, symbol=symbol),
                use_container_width=True,
            )
        else:
            st.info("Click **Scan Signal Now** to load chart, or **Start** live trading.")

        if st.session_state.live_signals:
            st.markdown("##### 📋 Signal History")
            st.dataframe(pd.DataFrame(st.session_state.live_signals).tail(10),
                         use_container_width=True, height=175)

        if st.session_state.live_trades:
            st.markdown("##### 🏁 Completed Trades")
            trd_df = pd.DataFrame(st.session_state.live_trades)
            st.dataframe(trd_df, use_container_width=True, height=145)
            wins_  = (trd_df["PnL Rs"] > 0).sum()
            tot_   = len(trd_df)
            pnl_   = trd_df["PnL Rs"].sum()
            st.caption(f"{wins_}/{tot_} wins ({wins_/tot_*100:.0f}%)  ·  P&L Rs{pnl_:,.2f}")

        if st.session_state.live_log:
            with st.expander("📜 Activity Log", expanded=False):
                st.code("\n".join(reversed(st.session_state.live_log[-50:])), language=None)

    if st.session_state.live_running:
        time.sleep(3)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with t_bt:
    bl, br = st.columns([1, 2.6], gap="medium")

    with bl:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""
        <div class="info-box">
        📈 <b>Symbol</b>   : <code>{symbol}</code><br>
        ⏱  <b>TF/Period</b>: <code>{interval}</code> · <code>{period}</code><br>
        🌊 <b>Depth</b>    : <code>{depth}</code><br>
        🛡  <b>SL</b>      : <code>{sl_lbl}</code><br>
        🎯 <b>Target</b>   : <code>{tgt_lbl}</code><br>
        💰 <b>Capital</b>  : Rs<code>{capital:,}</code><br><br>
        <small style="color:#546e7a">Signal bar N → entry open of bar N+1<br>
        Same ew_signal() as live trading</small>
        </div>""", unsafe_allow_html=True)

        if st.session_state.best_cfg_applied:
            st.success("✨ Using optimized config")

        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Fetching data (1.5 s rate-limit)…"):
                df_bt = fetch_ohlcv(symbol, interval, period, min_delay=1.5)
            if df_bt is None or len(df_bt) < 40:
                st.error("Not enough data. Try longer Period.")
            else:
                with st.spinner(f"Running on {len(df_bt)} bars…"):
                    res = run_backtest(df_bt, depth, sl_val, tgt_val, capital)
                    res.update({"df": df_bt, "pivots": find_pivots(df_bt, depth),
                                "symbol": symbol, "interval": interval, "period": period})
                st.session_state.bt_results = res
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success(f"✅ {res['metrics']['Total Trades']} trades!")

    with br:
        r = st.session_state.bt_results
        if r and "metrics" in r:
            m = r["metrics"]
            st.markdown(f"### Results — `{r.get('symbol','')}` · `{r.get('interval','')}` · `{r.get('period','')}`")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Return", f"{m['Total Return %']}%", delta=f"Rs{m['Final Equity Rs']:,}")
            c2.metric("Win Rate",     f"{m['Win Rate %']}%",    delta=f"{m['Wins']}W/{m['Losses']}L")
            c3.metric("Profit Factor",str(m["Profit Factor"]))
            c4.metric("Max Drawdown", f"{m['Max Drawdown %']}%")
            c5,c6,c7,c8 = st.columns(4)
            c5.metric("Sharpe",       str(m["Sharpe Ratio"]))
            c6.metric("Total Trades", str(m["Total Trades"]))
            c7.metric("Avg Win",      f"Rs{m['Avg Win Rs']:,}")
            c8.metric("Avg Loss",     f"Rs{m['Avg Loss Rs']:,}")

            tc1,tc2,tc3 = st.tabs(["🕯 Wave Chart","📈 Equity Curve","📋 Trade Log"])
            with tc1:
                st.plotly_chart(chart_waves(r["df"],r["pivots"],_blank(),r["trades"],r["symbol"]),
                                use_container_width=True)
            with tc2:
                st.plotly_chart(chart_equity(r["equity_curve"]), use_container_width=True)
            with tc3:
                st.dataframe(r["trades"], use_container_width=True, height=420)
                st.download_button("📥 Download CSV",
                                   data=r["trades"].to_csv(index=False),
                                   file_name=f"ew_bt_{symbol}_{interval}_{period}.csv",
                                   mime="text/csv")
                st.info("💡 Same ew_signal() as Live Trading — a completed live trade will appear here with matching Symbol/TF/Period.")
        elif r and "error" in r:
            st.error(r["error"])
        else:
            st.markdown("<div style='text-align:center;padding:80px 20px;color:#37474f'>"
                        "<h3>Run backtest to see results</h3></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
with t_opt:
    ol, or_ = st.columns([1, 3], gap="medium")

    with ol:
        st.markdown("### ⚙️ Config")
        st.markdown(f"""
        <div class="info-box">
        📈 <b>Symbol</b>: <code>{symbol}</code><br>
        ⏱  <code>{interval}</code> · <code>{period}</code><br>
        💰 Rs<code>{capital:,}</code><br><br>
        <b>Grid: 4×4×5 = 80 combos</b><br>
        <small style="color:#546e7a">Depths: 3,5,7,10<br>
        SL: 1%,2%,3%,Wave Auto<br>
        Targets: 1.5R,2R,3R,Wave,Fib</small>
        </div>""", unsafe_allow_html=True)
        st.warning("⏳ ~80 backtests (~1–3 min)")

        if st.button("🔬 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Fetching data…"):
                df_opt = fetch_ohlcv(symbol, interval, period, min_delay=1.5)
            if df_opt is None or len(df_opt) < 50:
                st.error("Not enough data.")
            else:
                with st.spinner("Running 80 combinations…"):
                    opt_df = run_optimization(df_opt, capital)
                st.session_state.opt_results = {
                    "df": opt_df, "symbol": symbol,
                    "interval": interval, "period": period,
                }
                if not opt_df.empty:
                    st.success(f"✅ Best Score: {opt_df['Score'].iloc[0]:.2f}")

    with or_:
        opt_r = st.session_state.opt_results
        if opt_r and opt_r.get("df") is not None and not opt_r["df"].empty:
            opt_df = opt_r["df"]
            st.markdown(f"### Results — `{opt_r['symbol']}` · `{opt_r['interval']}` · `{opt_r['period']}`")

            best_row = opt_df.iloc[0]
            b1,b2,b3,b4,b5 = st.columns(5)
            b1.metric("Best Depth",  str(int(best_row["Depth"])))
            b2.metric("Best SL",     str(best_row["SL"]))
            b3.metric("Best Target", str(best_row["Target"]))
            b4.metric("Best Return", f"{best_row['Return %']:.1f}%")
            b5.metric("Best Score",  f"{best_row['Score']:.2f}")

            st.markdown("---")
            ac1, ac2 = st.columns([1.6, 2.4])
            with ac1:
                n_rows = st.number_input("Apply top N config", min_value=1,
                                         max_value=min(5, len(opt_df)), value=1, step=1,
                                         help="1=best, 2=2nd best, etc.")
                apply_row = opt_df.iloc[int(n_rows) - 1]

            with ac2:
                st.markdown(f"""
                <div class="best-cfg">
                <b style="color:#00e5ff">Config #{int(n_rows)}</b><br>
                Depth <b>{int(apply_row['Depth'])}</b> ·
                SL <b>{apply_row['SL']}</b> ·
                Target <b>{apply_row['Target']}</b><br>
                Win <b>{apply_row['Win %']}%</b> ·
                Return <b>{apply_row['Return %']}%</b> ·
                Score <b>{apply_row['Score']:.2f}</b>
                </div>""", unsafe_allow_html=True)

            if st.button(
                f"✨ Apply Config #{int(n_rows)} → Sidebar + Backtest + Live",
                type="primary", use_container_width=True,
            ):
                new_depth = int(apply_row["Depth"])
                new_sl    = sl_label_from_val(apply_row["SL"])
                new_tgt   = tgt_label_from_val(apply_row["Target"])
                st.session_state.applied_depth   = new_depth
                st.session_state.applied_sl_lbl  = new_sl
                st.session_state.applied_tgt_lbl = new_tgt
                st.session_state.best_cfg_applied = True
                st.success(
                    f"✅ Applied! Depth={new_depth} · SL={new_sl} · Target={new_tgt}  "
                    f"— Sidebar dropdowns updated. Run Backtest/Live to use new config."
                )
                time.sleep(0.4)
                st.rerun()

            st.markdown("---")
            oc1, oc2 = st.tabs(["📊 Scatter", "📋 Full Table"])
            with oc1:
                st.plotly_chart(chart_opt_scatter(opt_df), use_container_width=True)
                st.caption("Top-left = best. Bubble size = Win Rate. Color = Score.")
            with oc2:
                def _hl(row):
                    if row.name == 0:   return ["background-color:rgba(0,229,255,.18)"]*len(row)
                    elif row.name < 3:  return ["background-color:rgba(0,229,255,.08)"]*len(row)
                    return [""]*len(row)
                st.dataframe(opt_df.style.apply(_hl, axis=1),
                             use_container_width=True, height=500)
                st.download_button("📥 Download CSV",
                                   data=opt_df.to_csv(index=False),
                                   file_name=f"ew_opt_{symbol}_{interval}_{period}.csv",
                                   mime="text/csv")
        else:
            st.markdown("<div style='text-align:center;padding:80px 20px;color:#37474f'>"
                        "<h3>Run optimization to see results</h3></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — HELP
# ═══════════════════════════════════════════════════════════════════════════
with t_help:
    st.markdown("## 📖 Help — Elliott Wave Algo Trader v4.0")
    h1, h2 = st.columns(2, gap="large")

    with h1:
        st.markdown("""
### 🌊 What is Elliott Wave? (Simple)
Markets move in **repeating wave patterns**. The app detects when Wave 2 (pullback) completes so you can enter for Wave 3 (the biggest move).

| Wave | What Happens | App Action |
|------|-------------|------------|
| W1 ↑ | First impulse | Detected as pivot |
| **W2 ↓** | **Retracement 38–79%** | **🟢 BUY signal here** |
| **W3 ↑** | **Strongest move** | **Ride this for profit** |
| W4 ↓ | Minor pullback | Take partial profits |
| W5 ↑ | Final push | Exit fully |

---

### 🤖 What's Automated?
- ✅ Pivot detection (ZigZag algorithm)
- ✅ Wave counting and Fibonacci validation
- ✅ Confidence scoring (peaks at 61.8% golden ratio)
- ✅ SL at Wave-2 pivot (safest level)
- ✅ Target at 1.618× Wave-1 (Fibonacci)
- ✅ Multi-timeframe consensus
- ✅ RSI + EMA + MACD context

**You only need to:**
1. Pick instrument from dropdown
2. Click a button
3. Follow the signal

---

### 📊 Recommended Settings

| For... | TF | Period | Depth |
|--------|----|--------|-------|
| Swing trade | 1d | 1y | 5–7 |
| Position | 1wk | 5y | 7–10 |
| Intraday | 1h | 3mo | 3–5 |
| Scalping | 15m | 5d | 2–3 |

**Tip**: Run Optimization → Apply Best Config for automatic optimal settings.
        """)

    with h2:
        st.markdown("""
### 🔭 Wave Analysis Tab (Recommended Start)
1. Select instrument in sidebar
2. Click **Run Full Analysis**
3. App scans 4 timeframes (Daily, 4H, 1H, 15M)
4. Each chart: W0/W1/W2 labels + RSI + EMA overlay
5. Read **Detailed Analysis** — it explains every signal in plain English
6. Follow the Overall verdict: BUY / SELL / HOLD

---

### 🔬 Optimize → Apply Config
1. Click **Run Optimization** (80 combos, ~2 min)
2. Best combo highlighted in blue in the table
3. Select top-N with number input
4. Click **✨ Apply Config** button
5. Sidebar, Backtest, and Live all update automatically

---

### ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Signal already fired" | Click 🔄 Reset All |
| No trades in backtest | Reduce Pivot Depth, use longer Period |
| Too few bars | Use longer Period for selected TF |
| No data returned | Check ticker (^NSEI, RELIANCE.NS, BTC-USD) |
| 4h interval issues | Use 1h or 1d instead |

---

### 📊 Ticker Format Guide

| Market | Format | Example |
|--------|--------|---------|
| NSE Index | `^TICKER` | `^NSEI`, `^NSEBANK` |
| NSE Stock | `TICKER.NS` | `RELIANCE.NS` |
| BSE Stock | `TICKER.BO` | `RELIANCE.BO` |
| Crypto | `COIN-USD` | `BTC-USD` |
| Gold/Silver | `GC=F` / `SI=F` | Futures |
| Forex | `PAIR=X` | `USDINR=X` |
| US Stocks | Plain | `AAPL`, `TSLA` |
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#37474f;font-size:.84rem;padding:8px">
    🌊 Elliott Wave Algo Trader v4.0 ·
    Streamlit + yfinance + Plotly ·
    <b style="color:#f44336">Not financial advice. Always use Stop Loss.</b>
    </div>""", unsafe_allow_html=True)
