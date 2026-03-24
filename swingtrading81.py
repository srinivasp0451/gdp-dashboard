"""
╔══════════════════════════════════════════════════════════════╗
║       🌊  ELLIOTT WAVE ALGO TRADER  v3.0                     ║
║  Backtest · Live Trading · Optimization · Dhan Integration   ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run elliott_wave_algo_trader.py
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
    border:1px solid #1e3a5f; border-radius:14px; padding:22px 28px;
    margin-bottom:18px; box-shadow:0 4px 24px rgba(0,229,255,.08);
}
.main-hdr h1 {
    font-family:'Exo 2',sans-serif; font-weight:700;
    background:linear-gradient(90deg,#00e5ff,#4dd0e1,#00bcd4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0; font-size:2rem; letter-spacing:.5px;
}
.main-hdr p { color:#546e7a; margin:6px 0 0; font-size:.9rem; }

.sig-buy  { background:rgba(76,175,80,.12); border:1.5px solid #4caf50;
    border-radius:10px; padding:14px 16px; text-align:center; }
.sig-sell { background:rgba(244,67,54,.12); border:1.5px solid #f44336;
    border-radius:10px; padding:14px 16px; text-align:center; }
.sig-hold { background:rgba(100,100,120,.10); border:1.5px solid #455a64;
    border-radius:10px; padding:14px 16px; text-align:center; }

.pos-card { background:rgba(0,150,200,.10); border:1px solid #0288d1;
    border-radius:8px; padding:12px 14px; font-size:.88rem; line-height:1.8; }

.info-box { background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
    padding:14px 16px; font-size:.87rem; line-height:1.9; }

.stTabs [data-baseweb="tab-list"] { gap:6px; background:transparent; }
.stTabs [data-baseweb="tab"] {
    background:#0d1b2a; border-radius:8px; color:#546e7a;
    border:1px solid #1e3a5f; padding:6px 18px; font-size:.85rem; }
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#0d3349,#0a2540) !important;
    color:#00e5ff !important; border-color:#00bcd4 !important; }

div[data-testid="metric-container"] {
    background:#0a1628; border:1px solid #1e3a5f;
    border-radius:8px; padding:10px 14px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "live_running":  False,
    "live_signals":  [],
    "live_log":      [],
    "last_bar_ts":   None,   # ← prevents duplicate signals on same bar
    "live_position": None,
    "live_pnl":      0.0,
    "live_trades":   [],
    "bt_results":    None,
    "opt_results":   None,
    "_scan_sig":     None,
    "_scan_df":      None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

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

SL_MAP = {
    "Wave Auto (Pivot Low/High)": "wave_auto",
    "0.5%": 0.005, "1%": 0.01, "1.5%": 0.015,
    "2%": 0.02,   "2.5%": 0.025, "3%": 0.03, "5%": 0.05,
}
TGT_MAP = {
    "Wave Auto (Fib 1.618 × W1)": "wave_auto",
    "R:R 1:1":   1.0, "R:R 1:1.5": 1.5, "R:R 1:2": 2.0,
    "R:R 1:2.5": 2.5, "R:R 1:3":   3.0,
    "Fib 1.618 × Wave 1": "fib_1618",
    "Fib 2.618 × Wave 1": "fib_2618",
}

# ═══════════════════════════════════════════════════════════════════════════
# RATE-LIMIT SAFE DATA FETCH  (≥1.5 s between requests)
# ═══════════════════════════════════════════════════════════════════════════
_fetch_lock    = threading.Lock()
_last_fetch_ts = [0.0]


def fetch_ohlcv(symbol: str, interval: str, period: str,
                min_delay: float = 1.5) -> Optional[pd.DataFrame]:
    """Thread-safe download enforcing min_delay seconds between calls."""
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
    # Keep alternating pivots, preserve the more extreme on runs
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
# CORE SIGNAL  (identical for BOTH backtest and live — ensures consistency)
# ═══════════════════════════════════════════════════════════════════════════
def _blank(reason: str = "") -> dict:
    return {
        "signal":      "HOLD",
        "entry_price": None,
        "sl":          None,
        "target":      None,
        "confidence":  0.0,
        "reason":      reason or "No Elliott Wave pattern detected",
        "pattern":     "—",
        "wave_pivots": None,
        "wave1_len":   0.0,
    }


def ew_signal(df: pd.DataFrame, depth: int = 5,
              sl_type="wave_auto", tgt_type="wave_auto") -> dict:
    """
    Elliott Wave signal on CLOSED bars only.
    BUY  → W0(Low)→W1(High)→W2(Low): Wave-2 bottom, enter for Wave 3 up.
    SELL → W0(High)→W1(Low)→W2(High): Wave-2 top, enter for Wave 3 down.
    Retracement validity: 23.6% – 88.6% of Wave 1.
    Confidence peaks at 61.8% golden ratio retracement.
    """
    n = len(df)
    if n < max(30, depth * 4):
        return _blank("Insufficient bars")

    pivots = find_pivots(df, depth)
    if len(pivots) < 4:
        return _blank("Not enough pivots — try smaller Pivot Depth")

    cur       = float(df["Close"].iloc[-1])
    best      = _blank()
    best_conf = 0.0

    for i in range(len(pivots) - 2):
        p0, p1, p2  = pivots[i], pivots[i + 1], pivots[i + 2]
        bars_since  = n - 1 - p2[0]

        # ── BUY: Low→High→Low ────────────────────────────────────────────────
        if p0[2] == "L" and p1[2] == "H" and p2[2] == "L":
            w1 = p1[1] - p0[1]
            if w1 <= 0:
                continue
            retr = (p1[1] - p2[1]) / w1
            if not (0.236 <= retr <= 0.886 and p2[1] > p0[1] and bars_since <= depth * 4):
                continue
            conf = 0.55
            if 0.50 <= retr <= 0.786:            conf = 0.70
            if abs(retr - 0.618) < 0.04:         conf = 0.85
            if i + 3 < len(pivots) and pivots[i + 3][2] == "H":
                w3 = pivots[i + 3][1] - p2[1]
                if w3 > w1:                       conf = min(conf + 0.08, 0.95)
                if abs(w3 / w1 - 1.618) < 0.20:  conf = min(conf + 0.05, 0.98)
            if conf <= best_conf:
                continue

            entry = cur
            sl_   = (p2[1] * 0.998) if sl_type == "wave_auto" \
                    else (entry * (1 - float(sl_type)))
            risk  = entry - sl_
            if risk <= 0:
                continue
            tgt_  = _calc_target(tgt_type, entry, "BUY", w1, risk)
            if tgt_ <= entry:
                continue
            best_conf = conf
            best = {
                "signal": "BUY", "entry_price": entry, "sl": sl_, "target": tgt_,
                "confidence": conf,
                "reason": f"Wave-2 bottom at {retr:.1%} retracement → Wave 3 up",
                "pattern": f"W2_BOTTOM ({retr:.1%})",
                "wave_pivots": [p0, p1, p2], "wave1_len": w1,
            }

        # ── SELL: High→Low→High ──────────────────────────────────────────────
        elif p0[2] == "H" and p1[2] == "L" and p2[2] == "H":
            w1 = p0[1] - p1[1]
            if w1 <= 0:
                continue
            retr = (p2[1] - p1[1]) / w1
            if not (0.236 <= retr <= 0.886 and p2[1] < p0[1] and bars_since <= depth * 4):
                continue
            conf = 0.55
            if 0.50 <= retr <= 0.786:            conf = 0.70
            if abs(retr - 0.618) < 0.04:         conf = 0.85
            if i + 3 < len(pivots) and pivots[i + 3][2] == "L":
                w3 = p2[1] - pivots[i + 3][1]
                if w3 > w1:                       conf = min(conf + 0.08, 0.95)
                if abs(w3 / w1 - 1.618) < 0.20:  conf = min(conf + 0.05, 0.98)
            if conf <= best_conf:
                continue

            entry = cur
            sl_   = (p2[1] * 1.002) if sl_type == "wave_auto" \
                    else (entry * (1 + float(sl_type)))
            risk  = sl_ - entry
            if risk <= 0:
                continue
            tgt_  = _calc_target(tgt_type, entry, "SELL", w1, risk)
            if tgt_ >= entry:
                continue
            best_conf = conf
            best = {
                "signal": "SELL", "entry_price": entry, "sl": sl_, "target": tgt_,
                "confidence": conf,
                "reason": f"Wave-2 top at {retr:.1%} retracement → Wave 3 down",
                "pattern": f"W2_TOP ({retr:.1%})",
                "wave_pivots": [p0, p1, p2], "wave1_len": w1,
            }

    return best


def _calc_target(tgt_type, entry: float, direction: str,
                 w1: float, risk: float) -> float:
    sign = 1 if direction == "BUY" else -1
    if tgt_type in ("wave_auto", "fib_1618"):
        return entry + sign * w1 * 1.618
    elif tgt_type == "fib_2618":
        return entry + sign * w1 * 2.618
    elif isinstance(tgt_type, (int, float)):
        return entry + sign * risk * float(tgt_type)
    return entry + sign * risk * 2.0


# ═══════════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, depth: int = 5,
                 sl_type="wave_auto", tgt_type="wave_auto",
                 capital: float = 100_000.0) -> dict:
    """
    Walk-forward backtest using IDENTICAL ew_signal() as live trading.
    Signal on bar i → entry at open of bar i+1  (realistic fill).
    """
    MIN_BARS = max(30, depth * 4)
    if len(df) < MIN_BARS + 10:
        return {"error": f"Need ≥{MIN_BARS + 10} bars. Use a longer period or smaller depth."}

    trades, equity_curve = [], [capital]
    equity, pos = capital, None

    for i in range(MIN_BARS, len(df) - 1):
        bar_df   = df.iloc[: i + 1]
        next_bar = df.iloc[i + 1]
        hi_i     = float(df.iloc[i]["High"])
        lo_i     = float(df.iloc[i]["Low"])

        # ── Manage open position ──────────────────────────────────────────
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
                    "Entry Time":  pos["entry_time"],
                    "Exit Time":   df.index[i],
                    "Type":        pos["type"],
                    "Entry":       round(pos["entry"], 2),
                    "Exit":        round(exit_p, 2),
                    "SL":          round(pos["sl"], 2),
                    "Target":      round(pos["target"], 2),
                    "Exit Reason": exit_r,
                    "PnL ₹":       round(pnl, 2),
                    "PnL %":       round(pnl / (pos["entry"] * qty) * 100, 2),
                    "Equity ₹":    round(equity, 2),
                    "Bars Held":   i - pos["entry_bar"],
                    "Confidence":  round(pos["conf"], 2),
                })
                pos = None

        # ── New signal ────────────────────────────────────────────────────
        if pos is None:
            sig = ew_signal(bar_df, depth, sl_type, tgt_type)
            if sig["signal"] in ("BUY", "SELL"):
                ep   = float(next_bar["Open"])
                w1   = sig.get("wave1_len", ep * 0.02) or (ep * 0.02)
                sl_  = sig["sl"] if sl_type == "wave_auto" else (
                    ep * (1 - float(sl_type)) if sig["signal"] == "BUY"
                    else ep * (1 + float(sl_type))
                )
                risk = abs(ep - sl_)
                if risk <= 0:
                    continue
                tgt_ = _calc_target(tgt_type, ep, sig["signal"], w1, risk)
                if sig["signal"] == "BUY"  and tgt_ <= ep: continue
                if sig["signal"] == "SELL" and tgt_ >= ep: continue
                qty  = max(1, int(equity * 0.95 / ep))
                pos  = {
                    "type":       sig["signal"],
                    "entry":      ep,
                    "sl":         sl_,
                    "target":     tgt_,
                    "entry_bar":  i + 1,
                    "entry_time": df.index[i + 1],
                    "qty":        qty,
                    "conf":       sig["confidence"],
                }

    # Force-close if still open at end of data
    if pos:
        ep2 = float(df["Close"].iloc[-1])
        qty = pos["qty"]
        pnl = (ep2 - pos["entry"]) * qty if pos["type"] == "BUY" \
              else (pos["entry"] - ep2) * qty
        equity += pnl
        trades.append({
            "Entry Time":  pos["entry_time"],
            "Exit Time":   df.index[-1],
            "Type":        pos["type"],
            "Entry":       round(pos["entry"], 2),
            "Exit":        round(ep2, 2),
            "SL":          round(pos["sl"], 2),
            "Target":      round(pos["target"], 2),
            "Exit Reason": "Open@End",
            "PnL ₹":       round(pnl, 2),
            "PnL %":       round(pnl / (pos["entry"] * qty) * 100, 2),
            "Equity ₹":    round(equity, 2),
            "Bars Held":   len(df) - 1 - pos["entry_bar"],
            "Confidence":  round(pos["conf"], 2),
        })

    if not trades:
        return {
            "error": (
                "No trades generated. Try: smaller Pivot Depth, "
                "a longer Period, or different SL/Target settings."
            ),
            "equity_curve": equity_curve,
        }

    tdf  = pd.DataFrame(trades)
    wins = tdf[tdf["PnL ₹"] > 0]
    loss = tdf[tdf["PnL ₹"] <= 0]
    ntot = len(tdf)
    wr   = len(wins) / ntot * 100 if ntot else 0
    pf   = abs(wins["PnL ₹"].sum() / loss["PnL ₹"].sum()) \
           if len(loss) and loss["PnL ₹"].sum() != 0 else 9999.0

    eq_arr = np.array(equity_curve)
    peak   = np.maximum.accumulate(eq_arr)
    mdd    = float(((eq_arr - peak) / peak * 100).min())
    rets   = tdf["PnL %"].values
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) \
             if len(rets) > 1 and rets.std() != 0 else 0.0

    return {
        "trades":       tdf,
        "equity_curve": equity_curve,
        "metrics": {
            "Total Trades":   ntot,
            "Win Rate %":     round(wr, 1),
            "Profit Factor":  round(pf, 2),
            "Total Return %": round((equity - capital) / capital * 100, 2),
            "Final Equity ₹": round(equity, 2),
            "Max Drawdown %": round(mdd, 2),
            "Sharpe Ratio":   round(sharpe, 2),
            "Avg Win ₹":      round(float(wins["PnL ₹"].mean()), 2) if len(wins) else 0.0,
            "Avg Loss ₹":     round(float(loss["PnL ₹"].mean()), 2) if len(loss) else 0.0,
            "Wins":   len(wins),
            "Losses": len(loss),
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

    prog = st.progress(0, text="Optimizing… 0%")
    rows = []

    for idx, (dep, sl, tgt) in enumerate(combos):
        r = run_backtest(df, depth=dep, sl_type=sl, tgt_type=tgt, capital=capital)
        if "metrics" in r:
            m = r["metrics"]
            rows.append({
                "Depth":     dep,
                "SL":        str(sl),
                "Target":    str(tgt),
                "Trades":    m["Total Trades"],
                "Win %":     m["Win Rate %"],
                "Return %":  m["Total Return %"],
                "PF":        m["Profit Factor"],
                "Max DD %":  m["Max Drawdown %"],
                "Sharpe":    m["Sharpe Ratio"],
            })
        pct = int((idx + 1) / len(combos) * 100)
        prog.progress((idx + 1) / len(combos), text=f"Optimizing… {pct}%  ({idx+1}/{len(combos)})")

    prog.empty()
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Composite score: reward return × win-rate × PF, penalise drawdown
    out["Score"] = (
        out["Return %"].clip(lower=0)
        * (out["Win %"] / 100)
        * out["PF"].clip(upper=10)
        / (out["Max DD %"].abs() + 1)
    )
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# DHAN BROKERAGE API
# ═══════════════════════════════════════════════════════════════════════════
class DhanAPI:
    BASE = "https://api.dhan.co"

    def __init__(self, client_id: str, token: str):
        self.cid  = client_id
        self.hdrs = {"Content-Type": "application/json", "access-token": token}

    def place_order(self, sec_id: str, segment: str, txn: str,
                    qty: int, order_type: str = "MARKET",
                    price: float = 0.0, product: str = "INTRADAY") -> dict:
        body = {
            "dhanClientId":    self.cid,
            "transactionType": txn,
            "exchangeSegment": segment,
            "productType":     product,
            "orderType":       order_type,
            "validity":        "DAY",
            "securityId":      sec_id,
            "quantity":        qty,
            "price":           price,
        }
        try:
            r = requests.post(f"{self.BASE}/orders", headers=self.hdrs,
                              json=body, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def fund_limit(self) -> dict:
        try:
            return requests.get(f"{self.BASE}/fundlimit",
                                headers=self.hdrs, timeout=10).json()
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# LIVE TRADING LOOP  (background daemon thread)
# ═══════════════════════════════════════════════════════════════════════════
def live_loop(symbol, interval, period, depth, sl_type, tgt_type,
              dhan_on, dhan_api, sec_id, live_qty):
    """
    Polls for new candles. Enforces ≥1.5 s rate-limit delay on every yfinance
    fetch. Duplicate-signal guard: only processes each bar timestamp once.
    """
    def log(msg: str, lvl: str = "INFO"):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}][{lvl}] {msg}"
        if "live_log" in st.session_state:
            st.session_state.live_log.append(line)
            st.session_state.live_log = st.session_state.live_log[-120:]

    POLL = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 3600, "1d": 3600, "1wk": 3600,
    }
    sleep_s = min(POLL.get(interval, 60), 60)
    log(f"🚀 Started | {symbol} | {interval} | {period}")

    while st.session_state.get("live_running", False):
        try:
            log("📡 Fetching (enforcing 1.5 s rate-limit delay)…")
            df = fetch_ohlcv(symbol, interval, period, min_delay=1.5)

            if df is None or len(df) < 35:
                log("⚠️  Insufficient data — retrying", "WARN")
                time.sleep(sleep_s)
                continue

            df_closed = df.iloc[:-1]           # exclude forming bar
            latest_ts = str(df_closed.index[-1])
            cur_price = float(df["Close"].iloc[-1])

            # ── KEY FIX: skip if bar already processed ────────────────────
            if st.session_state.get("last_bar_ts") == latest_ts:
                log(f"⏭  Bar …{latest_ts[-8:]} already processed — awaiting new candle")
                time.sleep(sleep_s)
                continue
            st.session_state.last_bar_ts = latest_ts  # mark BEFORE generating signal

            # ── Manage open position ──────────────────────────────────────
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
                    ep_, reason_ = hit
                    qty_  = pos["qty"]
                    pnl   = (ep_ - pos["entry"]) * qty_ if pos["type"] == "BUY" \
                            else (pos["entry"] - ep_) * qty_
                    st.session_state.live_pnl += pnl
                    st.session_state.live_trades.append({
                        "Time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Symbol":  symbol,
                        "TF":      interval,
                        "Period":  period,
                        "Type":    pos["type"],
                        "Entry":   round(pos["entry"], 2),
                        "Exit":    round(ep_, 2),
                        "SL":      round(pos["sl"], 2),
                        "Target":  round(pos["target"], 2),
                        "Qty":     qty_,
                        "PnL ₹":  round(pnl, 2),
                        "Reason":  reason_,
                    })
                    if dhan_on and dhan_api:
                        xt   = "SELL" if pos["type"] == "BUY" else "BUY"
                        resp = dhan_api.place_order(sec_id, "NSE_EQ", xt, qty_)
                        log(f"📤 Dhan exit: {resp}")
                    em = "✅" if "Target" in reason_ else "❌"
                    log(f"{em} {pos['type']} closed @ {ep_:.2f} | {reason_} | ₹{pnl:.2f}")
                    st.session_state.live_position = None
                    pos = None

            # ── Generate new signal ───────────────────────────────────────
            sig = ew_signal(df_closed, depth, sl_type, tgt_type)

            if pos is None and sig["signal"] in ("BUY", "SELL"):
                ep   = cur_price
                w1   = sig.get("wave1_len", ep * 0.02) or (ep * 0.02)
                sl_  = sig["sl"]
                risk = abs(ep - sl_)
                if risk > 0:
                    tgt_ = _calc_target(tgt_type, ep, sig["signal"], w1, risk)
                    valid = (tgt_ > ep if sig["signal"] == "BUY" else tgt_ < ep)
                    if valid:
                        st.session_state.live_position = {
                            "type":       sig["signal"],
                            "entry":      ep,
                            "sl":         sl_,
                            "target":     tgt_,
                            "qty":        live_qty,
                            "entry_time": datetime.now().strftime("%H:%M:%S"),
                            "symbol":     symbol,
                        }
                        st.session_state.live_signals.append({
                            "Time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Bar TS":  latest_ts,
                            "Symbol":  symbol,
                            "TF":      interval,
                            "Period":  period,
                            "Signal":  sig["signal"],
                            "Entry":   round(ep, 2),
                            "SL":      round(sl_, 2),
                            "Target":  round(tgt_, 2),
                            "Conf":    f"{sig['confidence']:.0%}",
                            "Pattern": sig["pattern"],
                        })
                        if dhan_on and dhan_api:
                            resp = dhan_api.place_order(sec_id, "NSE_EQ",
                                                        sig["signal"], live_qty)
                            log(f"📤 Dhan entry: {resp}")
                        em = "🟢" if sig["signal"] == "BUY" else "🔴"
                        log(
                            f"{em} {sig['signal']} @ {ep:.2f} | "
                            f"SL {sl_:.2f} | T {tgt_:.2f} | "
                            f"Conf {sig['confidence']:.0%} | {sig['pattern']}"
                        )
            else:
                log(f"⏸  HOLD @ {cur_price:.2f} | {sig['reason']}")

            time.sleep(sleep_s)

        except Exception as exc:
            log(f"💥 Error: {exc}", "ERROR")
            time.sleep(sleep_s)

    log("🛑 Live trading stopped")


# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def chart_waves(df: pd.DataFrame, pivots: list,
                sig: Optional[dict] = None,
                trades: Optional[pd.DataFrame] = None,
                symbol: str = "") -> go.Figure:
    sig = sig or _blank()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.76, 0.24], vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
    ), row=1, col=1)

    vol  = df.get("Volume", pd.Series(0, index=df.index))
    vcol = ["#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=vol, name="Volume",
                         marker_color=vcol, opacity=0.4), row=2, col=1)

    vp = [p for p in pivots if p[0] < len(df)]
    if vp:
        fig.add_trace(go.Scatter(
            x=[df.index[p[0]] for p in vp],
            y=[p[1] for p in vp],
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
            lbls = ["W0", "W1", "W2"][: len(valid_wp)]
            fig.add_trace(go.Scatter(
                x=[df.index[p[0]] for p in valid_wp],
                y=[p[1] for p in valid_wp],
                mode="lines+markers+text",
                line=dict(color=clr, width=2.5),
                marker=dict(size=11, color=clr),
                text=lbls, textposition="top center",
                textfont=dict(color=clr, size=11, family="Share Tech Mono"),
                name=f"Wave ({sig['signal']})",
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
                          annotation_text="  SL", annotation_position="right",
                          row=1, col=1)
        if sig.get("target"):
            fig.add_hline(y=sig["target"], line=dict(dash="dash", color="#66bb6a", width=1.5),
                          annotation_text="  Target", annotation_position="right",
                          row=1, col=1)

    if trades is not None and not trades.empty:
        for ttype, sym_, clr in [("BUY",  "triangle-up",   "#4caf50"),
                                  ("SELL", "triangle-down", "#f44336")]:
            sub = trades[trades["Type"] == ttype]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Entry Time"], y=sub["Entry"],
                    mode="markers",
                    marker=dict(size=9, color=clr, symbol=sym_,
                                line=dict(color="white", width=0.8)),
                    name=f"{ttype} Entry",
                ), row=1, col=1)
        for reason_, sym_, clr in [("Target", "circle", "#66bb6a"),
                                    ("SL",     "x",      "#ef5350")]:
            sub = trades[trades["Exit Reason"].str.contains(reason_, na=False)]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Exit Time"], y=sub["Exit"],
                    mode="markers",
                    marker=dict(size=7, color=clr, symbol=sym_),
                    name=f"Exit ({reason_})",
                    visible="legendonly",
                ), row=1, col=1)

    fig.update_layout(
        title=dict(text=f"🌊 Elliott Wave — {symbol}",
                   font=dict(size=15, color="#00e5ff")),
        template="plotly_dark", height=580,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#06101a", paper_bgcolor="#06101a",
        font=dict(color="#b0bec5", family="Exo 2"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
        margin=dict(l=10, r=70, t=50, b=10),
    )
    return fig


def chart_equity(equity_curve: list) -> go.Figure:
    eq  = np.array(equity_curve, dtype=float)
    pk  = np.maximum.accumulate(eq)
    dd  = (eq - pk) / pk * 100
    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        y=eq, mode="lines", name="Equity",
        line=dict(color="#00bcd4", width=2),
        fill="tozeroy", fillcolor="rgba(0,188,212,.07)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        y=dd, mode="lines", name="Drawdown %",
        line=dict(color="#f44336", width=1.5),
        fill="tozeroy", fillcolor="rgba(244,67,54,.12)",
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(dash="dot", color="#546e7a", width=1), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=370,
        plot_bgcolor="#06101a", paper_bgcolor="#06101a",
        font=dict(color="#b0bec5", family="Exo 2"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_opt_scatter(opt_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=opt_df["Max DD %"].abs(), y=opt_df["Return %"],
        mode="markers",
        marker=dict(
            size=(opt_df["Win %"] / 5).clip(lower=4),
            color=opt_df["Score"],
            colorscale="Plasma", showscale=True,
            colorbar=dict(title="Score", titlefont=dict(color="#b0bec5")),
            line=dict(color="rgba(255,255,255,.2)", width=0.5),
        ),
        text=[f"Depth={r.Depth}  SL={r.SL}  T={r.Target}" for _, r in opt_df.iterrows()],
        hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%  MaxDD %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Return vs Max Drawdown  (bubble = Win Rate)",
                   font=dict(size=13, color="#00e5ff")),
        xaxis_title="Max Drawdown %", yaxis_title="Total Return %",
        template="plotly_dark", height=380,
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
  <h1>🌊 Elliott Wave Algo Trader</h1>
  <p>Automated Wave Detection · Backtest · Live Trading · Dhan Integration · Optimization</p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    symbol = st.text_input(
        "📊 Symbol (Yahoo Finance format)", "^NSEI",
        help="Examples: ^NSEI  RELIANCE.NS  HDFCBANK.NS  AAPL  BTC-USD",
    )

    c1, c2 = st.columns(2)
    interval = c1.selectbox("⏱ Timeframe", TIMEFRAMES, index=6)
    vp_list  = VALID_PERIODS.get(interval, PERIODS)
    period   = c2.selectbox(
        "📅 Period", vp_list, index=min(4, len(vp_list) - 1),
        help="Constrained by yfinance limits per interval",
    )

    st.markdown("---")
    st.markdown("### 🌊 Elliott Wave")
    depth = st.slider(
        "Pivot Depth", min_value=2, max_value=15, value=5,
        help="Bars on each side to confirm a pivot. "
             "Lower = more signals (noisier). Higher = fewer (cleaner).",
    )

    st.markdown("---")
    st.markdown("### 🛡️ Risk Management")
    sl_lbl  = st.selectbox("Stop Loss",  list(SL_MAP.keys()),  index=0)
    tgt_lbl = st.selectbox("Target",     list(TGT_MAP.keys()), index=0)
    sl_val  = SL_MAP[sl_lbl]
    tgt_val = TGT_MAP[tgt_lbl]
    capital = st.number_input("💰 Capital (₹)", 10_000, 50_000_000, 100_000, 10_000)

    st.markdown("---")
    st.markdown("### 🏦 Dhan Brokerage")
    dhan_on = st.checkbox("Enable Dhan Integration", value=False,
                          help="Unchecked by default. Enable to place real orders.")
    dhan_api, sec_id, live_qty = None, "1333", 1
    if dhan_on:
        d_cid    = st.text_input("Client ID", key="d_cid")
        d_tok    = st.text_input("Access Token", type="password", key="d_tok")
        sec_id   = st.text_input("Security ID", "1333",
                                  help="Dhan instrument security ID (e.g. 1333 = Nifty 50 index)")
        live_qty = st.number_input("Order Qty", 1, 100_000, 1)
        if d_cid and d_tok:
            dhan_api = DhanAPI(d_cid, d_tok)
            if st.button("🔌 Test Connection"):
                with st.spinner("Connecting…"):
                    res = dhan_api.fund_limit()
                st.json(res)
        else:
            st.info("Enter Client ID + Token to activate")

    st.markdown("---")
    st.caption("⚡ Rate-limit delay: **1.5 s** between every fetch")
    st.caption(f"📌 `{symbol}` · `{interval}` · `{period}`")
    st.caption(f"🌊 Depth: `{depth}` · SL: `{sl_lbl}`")
    st.caption(f"🎯 Target: `{tgt_lbl}`")

# ───────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────
t_live, t_bt, t_opt, t_help = st.tabs([
    "🔴  Live Trading",
    "📊  Backtest",
    "🔬  Optimization",
    "❓  Help",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE TRADING
# ═══════════════════════════════════════════════════════════════════════════
with t_live:
    lc, rc = st.columns([1, 2.3], gap="medium")

    # ── Controls ────────────────────────────────────────────────────────────
    with lc:
        st.markdown("### 🎮 Controls")
        b1, b2 = st.columns(2)
        with b1:
            if not st.session_state.live_running:
                if st.button("▶ Start", type="primary", use_container_width=True):
                    st.session_state.live_running = True
                    st.session_state.live_log     = []
                    st.session_state.last_bar_ts  = None  # reset on fresh start
                    threading.Thread(
                        target=live_loop,
                        args=(symbol, interval, period, depth,
                              sl_val, tgt_val,
                              dhan_on, dhan_api, sec_id, live_qty),
                        daemon=True,
                    ).start()
                    st.rerun()
            else:
                if st.button("⏹ Stop", type="secondary", use_container_width=True):
                    st.session_state.live_running = False
                    st.rerun()
        with b2:
            if st.button(
                "🔄 Reset All", use_container_width=True,
                help="Clears position, signals, log and the bar-timestamp tracker. "
                     "Fixes 'signal already fired' issue.",
            ):
                for k, v in _DEFAULTS.items():
                    st.session_state[k] = v
                st.success("State cleared ✓")
                time.sleep(0.3)
                st.rerun()

        if st.session_state.live_running:
            st.success("🟢 **LIVE — RUNNING**")
        else:
            st.warning("⚫ **STOPPED**")

        # Open position card
        pos = st.session_state.live_position
        if pos:
            clr = "#4caf50" if pos["type"] == "BUY" else "#f44336"
            st.markdown(f"""
            <div class="pos-card">
            📍 <b style="color:{clr}">{pos['type']} OPEN</b><br>
            Entry : <b>{pos['entry']:.2f}</b>  &nbsp;·&nbsp;  Qty : <b>{pos['qty']}</b><br>
            SL    : <b style="color:#ff7043">{pos['sl']:.2f}</b>
            &nbsp;·&nbsp;
            Target: <b style="color:#66bb6a">{pos['target']:.2f}</b><br>
            Since : {pos['entry_time']}
            </div>""", unsafe_allow_html=True)
        else:
            st.caption("No open position")

        st.markdown(f"""
        <div class="info-box" style="margin-top:10px">
        📊 <b>P&L</b> : ₹{st.session_state.live_pnl:,.2f}<br>
        📡 <b>Signals</b> : {len(st.session_state.live_signals)}<br>
        🏁 <b>Trades</b>  : {len(st.session_state.live_trades)}<br>
        🕒 <b>Last Bar</b>: {str(st.session_state.last_bar_ts or "—")[-19:]}
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔍 Manual Scan")
        st.caption("Fetches latest data and evaluates current Elliott Wave signal.")
        if st.button("Scan Signal Now", use_container_width=True):
            with st.spinner("Fetching (rate-limit 1.5 s)…"):
                df_ = fetch_ohlcv(symbol, interval, period, min_delay=1.5)
            if df_ is not None and len(df_) >= 35:
                sig_ = ew_signal(df_.iloc[:-1], depth, sl_val, tgt_val)
                st.session_state._scan_sig = sig_
                st.session_state._scan_df  = df_
            else:
                st.error("⚠️ No data or too few bars received")
                st.info(f"Try a longer Period for `{interval}` timeframe.")

        # Signal card
        sc_ = st.session_state._scan_sig
        if sc_:
            s = sc_["signal"]
            if s == "BUY":
                st.markdown(f"""
                <div class="sig-buy">
                <div style="font-size:1.4rem;color:#4caf50;font-weight:700">🟢 BUY</div>
                <div style="font-size:.88rem;margin-top:6px">
                  Entry <b>{sc_['entry_price']:.2f}</b><br>
                  SL <b style="color:#ff7043">{sc_['sl']:.2f}</b> &nbsp;·&nbsp;
                  Target <b style="color:#66bb6a">{sc_['target']:.2f}</b><br>
                  Confidence <b>{sc_['confidence']:.0%}</b><br>
                  Pattern: <b>{sc_['pattern']}</b><br>
                  <small style="color:#78909c">{sc_['reason']}</small>
                </div></div>""", unsafe_allow_html=True)
            elif s == "SELL":
                st.markdown(f"""
                <div class="sig-sell">
                <div style="font-size:1.4rem;color:#f44336;font-weight:700">🔴 SELL</div>
                <div style="font-size:.88rem;margin-top:6px">
                  Entry <b>{sc_['entry_price']:.2f}</b><br>
                  SL <b style="color:#ff7043">{sc_['sl']:.2f}</b> &nbsp;·&nbsp;
                  Target <b style="color:#66bb6a">{sc_['target']:.2f}</b><br>
                  Confidence <b>{sc_['confidence']:.0%}</b><br>
                  Pattern: <b>{sc_['pattern']}</b><br>
                  <small style="color:#78909c">{sc_['reason']}</small>
                </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sig-hold">
                <div style="font-size:1.1rem;color:#78909c;font-weight:600">⏸ HOLD / NO SIGNAL</div>
                <div style="font-size:.85rem;color:#546e7a;margin-top:4px">
                  {sc_['reason']}
                </div></div>""", unsafe_allow_html=True)

    # ── Dashboard ────────────────────────────────────────────────────────────
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
            st.info("Click **Scan Signal Now** to load the chart, or **Start** live trading.")

        # Signal history
        if st.session_state.live_signals:
            st.markdown("##### 📋 Signal History")
            sig_df = pd.DataFrame(st.session_state.live_signals)
            st.dataframe(sig_df.tail(10), use_container_width=True, height=180)

        # Completed trades
        if st.session_state.live_trades:
            st.markdown("##### 🏁 Completed Trades")
            trd_df = pd.DataFrame(st.session_state.live_trades)
            st.dataframe(trd_df, use_container_width=True, height=150)
            wins_  = (trd_df["PnL ₹"] > 0).sum()
            tot_   = len(trd_df)
            pnl_   = trd_df["PnL ₹"].sum()
            st.caption(
                f"Live stats → {wins_}/{tot_} wins "
                f"({wins_/tot_*100:.0f}%)  ·  Total P&L ₹{pnl_:,.2f}"
            )

        # Activity log
        if st.session_state.live_log:
            with st.expander("📜 Activity Log", expanded=False):
                st.code(
                    "\n".join(reversed(st.session_state.live_log[-50:])),
                    language=None,
                )

    # Auto-refresh while running (every 3 s to not hammer CPU)
    if st.session_state.live_running:
        time.sleep(3)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with t_bt:
    bl, br = st.columns([1, 2.6], gap="medium")

    with bl:
        st.markdown("### ⚙️ Backtest Config")
        st.markdown(f"""
        <div class="info-box">
        📈 <b>Symbol</b>    : <code>{symbol}</code><br>
        ⏱  <b>Timeframe</b> : <code>{interval}</code>
            &nbsp; 📅 <b>Period</b> : <code>{period}</code><br>
        🌊 <b>Pivot Depth</b>: <code>{depth}</code><br>
        🛡  <b>Stop Loss</b>  : <code>{sl_lbl}</code><br>
        🎯 <b>Target</b>      : <code>{tgt_lbl}</code><br>
        💰 <b>Capital</b>     : ₹<code>{capital:,}</code><br><br>
        <small style="color:#546e7a">
        ✅ Signal on bar N → entry at open of bar N+1<br>
        ✅ Uses identical ew_signal() as live trading
        </small>
        </div>""", unsafe_allow_html=True)

        run_bt = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
        if run_bt:
            with st.spinner("Fetching data (1.5 s rate-limit)…"):
                df_bt = fetch_ohlcv(symbol, interval, period, min_delay=1.5)
            if df_bt is None or len(df_bt) < 40:
                st.error("Not enough data. Try a longer Period or different Symbol.")
            else:
                with st.spinner(f"Running backtest on {len(df_bt)} bars…"):
                    res = run_backtest(df_bt, depth, sl_val, tgt_val, capital)
                    res["df"]       = df_bt
                    res["pivots"]   = find_pivots(df_bt, depth)
                    res["symbol"]   = symbol
                    res["interval"] = interval
                    res["period"]   = period
                st.session_state.bt_results = res
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success(f"✅ {res['metrics']['Total Trades']} trades generated!")

    with br:
        r = st.session_state.bt_results
        if r and "metrics" in r:
            m = r["metrics"]
            hdr = (
                f"### Results — `{r.get('symbol','')}` "
                f"· `{r.get('interval','')}` · `{r.get('period','')}`"
            )
            st.markdown(hdr)

            # Metric row 1
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return",  f"{m['Total Return %']} %",
                      delta=f"₹{m['Final Equity ₹']:,}")
            c2.metric("Win Rate",      f"{m['Win Rate %']} %",
                      delta=f"{m['Wins']}W / {m['Losses']}L")
            c3.metric("Profit Factor", str(m["Profit Factor"]))
            c4.metric("Max Drawdown",  f"{m['Max Drawdown %']} %")

            # Metric row 2
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Sharpe Ratio",  str(m["Sharpe Ratio"]))
            c6.metric("Total Trades",  str(m["Total Trades"]))
            c7.metric("Avg Win ₹",     f"₹{m['Avg Win ₹']:,}")
            c8.metric("Avg Loss ₹",    f"₹{m['Avg Loss ₹']:,}")

            tc1, tc2, tc3 = st.tabs(["🕯 Wave Chart", "📈 Equity Curve", "📋 Trade Log"])

            with tc1:
                st.plotly_chart(
                    chart_waves(r["df"], r["pivots"], _blank(),
                                r["trades"], r["symbol"]),
                    use_container_width=True,
                )

            with tc2:
                st.plotly_chart(
                    chart_equity(r["equity_curve"]),
                    use_container_width=True,
                )

            with tc3:
                st.dataframe(r["trades"], use_container_width=True, height=420)
                st.download_button(
                    "📥 Download Trade Log (CSV)",
                    data=r["trades"].to_csv(index=False),
                    file_name=f"ew_backtest_{symbol}_{interval}_{period}.csv",
                    mime="text/csv",
                )
                st.info(
                    "💡 **Backtest ↔ Live consistency**: Both use the identical `ew_signal()` "
                    "function. Signal detected on bar N → entry at bar N+1 open. "
                    "A completed live trade will appear in this table when run with the "
                    "same **Symbol · Timeframe · Period** settings."
                )

        elif r and "error" in r:
            st.error(r["error"])
        else:
            st.markdown(
                "<div style='text-align:center;padding:80px 20px;color:#37474f'>"
                "<h3>Run a backtest to see results</h3>"
                "<p>Configure the sidebar then click <b>Run Backtest</b></p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
with t_opt:
    ol, or_ = st.columns([1, 3], gap="medium")

    with ol:
        st.markdown("### ⚙️ Optimization Config")
        st.markdown(f"""
        <div class="info-box">
        📈 <b>Symbol</b>    : <code>{symbol}</code><br>
        ⏱  <b>Timeframe</b> : <code>{interval}</code>
            &nbsp; 📅 <b>Period</b> : <code>{period}</code><br>
        💰 <b>Capital</b>   : ₹<code>{capital:,}</code><br><br>
        <b>Grid search:</b><br>
        &nbsp;• Depths  : 3, 5, 7, 10<br>
        &nbsp;• SL      : 1%, 2%, 3%, Wave Auto<br>
        &nbsp;• Targets : 1.5, 2.0, 3.0 R:R, Wave, Fib<br>
        &nbsp;= <b>80 combinations</b><br><br>
        <small style="color:#546e7a">
        Score = Return% × WinRate × PF / (MaxDD+1)
        </small>
        </div>""", unsafe_allow_html=True)

        st.warning("⏳ 80 backtests — may take 1–3 min depending on data size.")
        run_opt = st.button("🔬 Run Optimization", type="primary", use_container_width=True)

        if run_opt:
            with st.spinner("Fetching data (1.5 s rate-limit)…"):
                df_opt = fetch_ohlcv(symbol, interval, period, min_delay=1.5)
            if df_opt is None or len(df_opt) < 50:
                st.error("Not enough data for optimization.")
            else:
                with st.spinner("Running 80 parameter combinations…"):
                    opt_df = run_optimization(df_opt, capital)
                st.session_state.opt_results = {
                    "df": opt_df, "symbol": symbol,
                    "interval": interval, "period": period,
                }
                if opt_df.empty:
                    st.error("No combinations produced valid results.")
                else:
                    st.success(f"✅ Optimization complete! Best score: {opt_df['Score'].iloc[0]:.2f}")

    with or_:
        opt_r = st.session_state.opt_results
        if opt_r and not opt_r["df"].empty:
            opt_df = opt_r["df"]
            hdr = (
                f"### Optimization Results — `{opt_r['symbol']}` "
                f"· `{opt_r['interval']}` · `{opt_r['period']}`"
            )
            st.markdown(hdr)

            # Best params highlight
            best_row = opt_df.iloc[0]
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Best Depth",  str(int(best_row["Depth"])))
            b2.metric("Best SL",     str(best_row["SL"]))
            b3.metric("Best Target", str(best_row["Target"]))
            b4.metric("Best Return", f"{best_row['Return %']:.1f}%")
            b5.metric("Best Score",  f"{best_row['Score']:.2f}")

            oc1, oc2 = st.tabs(["📊 Scatter Chart", "📋 Full Results Table"])

            with oc1:
                st.plotly_chart(chart_opt_scatter(opt_df), use_container_width=True)
                st.caption(
                    "Each bubble = one parameter combination. "
                    "Bubble size = Win Rate. Color = Score (higher is better). "
                    "Ideal: top-left corner (high return, low drawdown)."
                )

            with oc2:
                # Highlight top-3 rows
                def highlight_top(s):
                    return ["background-color: rgba(0,229,255,.15)"
                            if i < 3 else "" for i in range(len(s))]

                st.dataframe(
                    opt_df.style.apply(highlight_top, axis=0),
                    use_container_width=True,
                    height=500,
                )
                st.download_button(
                    "📥 Download Results (CSV)",
                    data=opt_df.to_csv(index=False),
                    file_name=f"ew_optim_{symbol}_{interval}_{period}.csv",
                    mime="text/csv",
                )

        else:
            st.markdown(
                "<div style='text-align:center;padding:80px 20px;color:#37474f'>"
                "<h3>Run optimization to see results</h3>"
                "<p>Click <b>Run Optimization</b> to grid-search the best parameters</p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — HELP
# ═══════════════════════════════════════════════════════════════════════════
with t_help:
    st.markdown("## 📖 How to Use Elliott Wave Algo Trader")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
### 🌊 Elliott Wave Theory — Quick Recap
Elliott Wave theory identifies **5-wave impulse cycles** followed by **3-wave corrections**:

| Wave | Type | Description |
|------|------|-------------|
| W1   | Impulse | Initial move |
| W2   | Correction | Pullback (23.6–88.6% of W1) |
| **W3**| **Impulse** | **Strongest & longest move** |
| W4   | Correction | Pullback |
| W5   | Impulse | Final extension |

**This app enters at the end of Wave 2** for maximum Wave 3 upside.

---

### 🎯 Signal Logic
- **BUY** → ZigZag Low → High → Low pattern where the second low retraces 23.6–88.6% of Wave 1
- **SELL** → ZigZag High → Low → High pattern with same retracement rule
- **Confidence boosts** at golden ratio (61.8%) retracement
- **Wave 3 bonus** applied if W3 already exists and is ≥ 1.618× W1

---

### ⚙️ Parameter Guide

| Parameter | Recommendation |
|-----------|---------------|
| Pivot Depth | 5–7 for daily, 3–5 for intraday |
| Timeframe | 1d/1wk for swing, 1h/4h for intraday |
| Period | 1y+ for daily; 3mo for intraday |
| SL | Wave Auto uses the actual Wave 2 pivot |
| Target | Wave Auto = 1.618× Wave 1 (Fibonacci) |

---

### 🔁 Backtest ↔ Live Consistency
Both tabs use the **identical `ew_signal()` function**.

Signal fires on bar N → entry at **open of bar N+1**.

If a live signal completes (hits SL or Target), run a backtest with the **same Symbol · Timeframe · Period** — that trade will appear in the backtest trade log, confirming consistency.
        """)

    with col2:
        st.markdown("""
### 🔴 Live Trading Tab

1. Configure Symbol, Timeframe, Period in sidebar
2. Click **▶ Start** to launch background polling thread
3. The thread fetches data every candle interval (max 60 s)
4. **Rate-limit**: 1.5 s enforced between all yfinance requests
5. Each candle bar timestamp is tracked — **no duplicate signals** on the same bar
6. Use **Scan Signal Now** for an instant one-shot check
7. Use **🔄 Reset All** to clear all state and start fresh

---

### 🏦 Dhan Brokerage Integration

1. Enable checkbox in sidebar
2. Enter **Client ID** and **Access Token** from Dhan portal
3. Enter **Security ID** for your instrument (e.g. 1333 = Nifty 50)
4. Set Order Quantity
5. Click **Test Connection** to verify funds
6. When live: BUY/SELL orders placed at MARKET price, INTRADAY product

> ⚠️ **Always test with small qty first. Paper trade before going live.**

---

### 🔬 Optimization Tab

Runs 80 combinations of:
- **Pivot Depth**: 3, 5, 7, 10
- **Stop Loss**: 1%, 2%, 3%, Wave Auto
- **Target**: 1.5R, 2R, 3R, Wave Auto, Fib 1.618

**Score formula**:  
`Score = Return% × WinRate × PF ÷ (|MaxDD|+1)`

Top-3 rows highlighted in table. Scatter chart shows Return vs DrawDown with bubble size = Win Rate.

---

### ⚠️ Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Signal already fired" | Click **🔄 Reset All** |
| No data / too few bars | Use longer Period for selected Timeframe |
| No trades in backtest | Reduce Pivot Depth or use longer Period |
| yfinance rate limit | Built-in 1.5 s delay handles this automatically |
| 4h interval errors | yfinance may not support 4h — use 1h or 1d |

---

### 📊 Supported Symbols

| Type | Example |
|------|---------|
| NSE Index | `^NSEI`, `^NSEBANK` |
| NSE Stock | `RELIANCE.NS`, `TCS.NS` |
| BSE Stock | `RELIANCE.BO` |
| Crypto | `BTC-USD`, `ETH-USD` |
| US Stock | `AAPL`, `TSLA` |
| Forex | `USDINR=X`, `EURUSD=X` |
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#37474f;font-size:.85rem;padding:10px">
    🌊 Elliott Wave Algo Trader v3.0 &nbsp;·&nbsp;
    Built with Streamlit + yfinance + Plotly &nbsp;·&nbsp;
    <b style="color:#f44336">Not financial advice. Trade at your own risk.</b>
    </div>
    """, unsafe_allow_html=True)
