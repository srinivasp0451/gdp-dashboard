# ══════════════════════════════════════════════════════════════════════════════
#  SMART INVESTING — Algorithmic Trading Platform  v6.0
#
#  Install:
#    pip install streamlit yfinance pandas numpy plotly pytz
#    pip install dhanhq                  # optional – Dhan broker
#    pip install streamlit-autorefresh   # REQUIRED for live auto-refresh
#
#  Run:  streamlit run smart_investing.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings("ignore")

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AR = True
except ImportError:
    HAS_AR = False

try:
    from dhanhq import dhanhq
    HAS_DHAN = True
except ImportError:
    HAS_DHAN = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Investing", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
IST = pytz.timezone("Asia/Kolkata")

# ════════════════════════════════════════════════════════════════════════════
# THREAD-SAFE LIVE STATE
# One instance per user session stored in st.session_state.
# ALL writes from background thread go through .set() which holds a lock.
# UI always calls .snapshot() to get a consistent copy — no iteration over
# a dict that could be modified mid-loop.
# ════════════════════════════════════════════════════════════════════════════
class LiveState:
    def __init__(self):
        self._lock = threading.RLock()
        self._d = dict(
            running=False, cfg=None,
            position=None, pending_entry=None,
            trade_history=[], log=[],
            chart_df=None, ltp=None,
            ema_fast=None, ema_slow=None, atr=None,
            last_exit_ts=None,
        )

    # ── write helpers (thread calls these) ───────────────────────────────
    def set(self, key, value):
        with self._lock:
            self._d[key] = value

    def get(self, key, default=None):
        with self._lock:
            return self._d.get(key, default)

    def add_log(self, msg):
        with self._lock:
            ts = datetime.now(IST).strftime("%H:%M:%S")
            self._d["log"].append(f"[{ts}] {msg}")
            if len(self._d["log"]) > 300:
                self._d["log"] = self._d["log"][-300:]

    def add_trade(self, rec):
        with self._lock:
            self._d["trade_history"].append(rec)

    def update_position(self, pos):
        with self._lock:
            self._d["position"] = pos

    # ── read helper (UI calls this once per render) ───────────────────────
    def snapshot(self) -> dict:
        """Returns a shallow copy — safe for UI to iterate over freely."""
        with self._lock:
            d = self._d
            return dict(
                running=d["running"], cfg=d["cfg"],
                position=dict(d["position"]) if d["position"] else None,
                pending_entry=dict(d["pending_entry"]) if d["pending_entry"] else None,
                trade_history=list(d["trade_history"]),
                log=list(d["log"]),
                chart_df=d["chart_df"],          # DataFrame immutable from UI side
                ltp=d["ltp"],
                ema_fast=d["ema_fast"], ema_slow=d["ema_slow"], atr=d["atr"],
                last_exit_ts=d["last_exit_ts"],
            )

    def reset(self):
        with self._lock:
            history = list(self._d["trade_history"])  # preserve history across resets
            self._d = dict(
                running=False, cfg=None,
                position=None, pending_entry=None,
                trade_history=history, log=[],
                chart_df=None, ltp=None,
                ema_fast=None, ema_slow=None, atr=None,
                last_exit_ts=None,
            )

    def clear_trades(self):
        with self._lock:
            self._d["trade_history"] = []

    def square_off(self, ltp_now, qty):
        """Thread-safely close open position, returns trade record or None."""
        with self._lock:
            pos = self._d["position"]
            if pos is None:
                return None
            tt  = pos["trade_type"]
            ep  = float(pos["entry_price"])
            sl  = pos.get("sl"); tgt = pos.get("target")
            pnl = ((ltp_now - ep) if tt == "buy" else (ep - ltp_now)) * qty
            pts = (ltp_now - ep) if tt == "buy" else (ep - ltp_now)
            et  = pos["entry_time"]
            try:
                et_dt = datetime.strptime(et, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            except Exception:
                et_dt = datetime.now(IST)
            xt_dt = datetime.now(IST)
            rec = dict(
                **{"Entry Time": et, "Exit Time": xt_dt.strftime("%Y-%m-%d %H:%M:%S"),
                   "Duration": _dur(et_dt, xt_dt),
                   "Trade Type": tt.upper(),
                   "Entry Price": round(ep, 2), "Exit Price": round(ltp_now, 2),
                   "SL": round(float(sl), 2) if sl is not None else "—",
                   "Target": round(float(tgt), 2) if tgt is not None else "—",
                   "Entry Reason": pos.get("entry_reason", ""),
                   "Exit Reason": "Manual Square Off",
                   "Points": round(pts, 2), "PnL (Rs)": round(pnl, 2),
                   "Qty": qty, "Mode": "Live"}
            )
            self._d["trade_history"].append(rec)
            self._d["position"] = None
            self._d["pending_entry"] = None
            self._d["last_exit_ts"] = datetime.now(IST)
            return rec


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE  (per-user, fully isolated by Streamlit)
# ════════════════════════════════════════════════════════════════════════════
if "live_state" not in st.session_state:
    st.session_state["live_state"] = LiveState()
if "live_thread" not in st.session_state:
    st.session_state["live_thread"] = None
if "stop_event" not in st.session_state:
    st.session_state["stop_event"] = threading.Event()
if "bt_results" not in st.session_state:
    st.session_state["bt_results"] = None
if "bt_chart_df" not in st.session_state:
    st.session_state["bt_chart_df"] = None
if "bt_run_key" not in st.session_state:
    st.session_state["bt_run_key"] = None
if "light_theme" not in st.session_state:
    st.session_state["light_theme"] = True

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
def inject_css(light: bool) -> None:
    # Suppress Streamlit's fading overlay during autorefresh
    no_fade = """
    [data-testid="stStatusWidget"]{visibility:hidden!important;height:0!important;}
    div[data-testid="stDecoration"]{display:none!important;}
    """
    if light:
        st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html,body,[data-testid="stApp"]{{background:#f5f7fa!important;color:#1a2236;font-family:'Syne',sans-serif;}}
[data-testid="stSidebar"]{{background:#eef1f7!important;border-right:1px solid #d0d7e6;}}
.stTabs [data-baseweb="tab-list"]{{background:transparent;border-bottom:1px solid #d0d7e6;}}
.stTabs [data-baseweb="tab"]{{font-family:'Syne',sans-serif;font-weight:700;font-size:13px;color:#7a8899;padding:10px 22px;border-radius:6px 6px 0 0;}}
.stTabs [aria-selected="true"]{{color:#007a6a!important;background:rgba(0,122,106,.07)!important;}}
[data-testid="metric-container"]{{background:#fff;border:1px solid #d0d7e6;border-radius:10px;padding:10px 14px;}}
[data-testid="stMetricLabel"]{{font-size:10px;color:#7a8899!important;letter-spacing:.6px;text-transform:uppercase;}}
[data-testid="stMetricValue"]{{font-family:'JetBrains Mono',monospace;font-size:15px;color:#1a2236;}}
[data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace;font-size:11px;}}
.stButton button{{border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;border:1px solid #d0d7e6;transition:all .15s;}}
.stButton button[kind="primary"]{{background:linear-gradient(135deg,#00b896,#007a6a);color:#fff;border:none;}}
.stSelectbox>div,.stNumberInput>div input,.stTextInput>div input{{background:#fff!important;border:1px solid #d0d7e6!important;border-radius:8px!important;color:#1a2236!important;font-family:'JetBrains Mono',monospace!important;}}
.shdr{{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#7a8899;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #d0d7e6;}}
.card{{background:#fff;border:1px solid #d0d7e6;border-radius:12px;padding:14px 18px;margin-bottom:10px;}}
.card-red{{background:rgba(220,50,80,.05);border-color:rgba(220,50,80,.3);}}
.card-green{{background:rgba(0,160,120,.05);border-color:rgba(0,160,120,.3);}}
.logbox{{background:#f5f7fa;border:1px solid #d0d7e6;border-radius:8px;padding:10px;height:260px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.7;color:#1a2236;}}
hr{{border-color:#d0d7e6!important;margin:8px 0!important;}}
{no_fade}</style>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html,body,[data-testid="stApp"]{{background:#080b12!important;color:#c8d0e0;font-family:'Syne',sans-serif;}}
[data-testid="stSidebar"]{{background:#0c1018!important;border-right:1px solid #1a2236;}}
.stTabs [data-baseweb="tab-list"]{{background:transparent;border-bottom:1px solid #1a2236;gap:4px;}}
.stTabs [data-baseweb="tab"]{{font-family:'Syne',sans-serif;font-weight:700;font-size:13px;color:#4a5568;padding:10px 22px;border-radius:6px 6px 0 0;border:1px solid transparent;transition:all .15s;}}
.stTabs [aria-selected="true"]{{color:#00e5b4!important;background:rgba(0,229,180,.06)!important;border-color:#1a2236 #1a2236 transparent!important;}}
[data-testid="metric-container"]{{background:#0f1623;border:1px solid #1a2236;border-radius:10px;padding:10px 14px;}}
[data-testid="stMetricLabel"]{{font-size:10px;color:#4a5568!important;letter-spacing:.6px;text-transform:uppercase;}}
[data-testid="stMetricValue"]{{font-family:'JetBrains Mono',monospace;font-size:15px;color:#c8d0e0;}}
[data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace;font-size:11px;}}
.stButton button{{border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;font-size:13px;border:1px solid #1a2236;transition:all .15s;color:#c8d0e0;}}
.stButton button[kind="primary"]{{background:linear-gradient(135deg,#00e5b4,#00a88a);color:#000;border:none;}}
.stButton button[kind="primary"]:hover{{opacity:.9;}}
.stButton button:not([kind="primary"]):hover{{border-color:#00e5b4;color:#00e5b4;}}
.stSelectbox>div,.stNumberInput>div input,.stTextInput>div input{{background:#0f1623!important;border:1px solid #1a2236!important;border-radius:8px!important;color:#c8d0e0!important;font-family:'JetBrains Mono',monospace!important;font-size:13px!important;}}
.shdr{{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4a5568;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #1a2236;}}
.card{{background:#0f1623;border:1px solid #1a2236;border-radius:12px;padding:14px 18px;margin-bottom:10px;}}
.card-red{{background:rgba(255,77,109,.07);border-color:rgba(255,77,109,.3);}}
.card-green{{background:rgba(0,229,180,.06);border-color:rgba(0,229,180,.25);}}
.logbox{{background:#080b12;border:1px solid #1a2236;border-radius:8px;padding:10px;height:260px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.7;}}
hr{{border-color:#1a2236!important;margin:8px 0!important;}}
[data-testid="stExpander"]{{background:#0f1623;border:1px solid #1a2236;border-radius:10px;}}
{no_fade}</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
TICKERS = {
    "Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN",
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD",
    "Gold": "GC=F", "Silver": "SI=F",
}
TF_PERIODS = {
    "1m":  ["1d","5d","7d"],
    "5m":  ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo"],
    "1h":  ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":  ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk": ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
WARMUP = {
    "1d":"5d","5d":"1mo","7d":"1mo","1mo":"3mo","3mo":"6mo",
    "6mo":"1y","1y":"2y","2y":"5y","5y":"max","10y":"max","20y":"max",
}
P_DAYS = {
    "1d":1,"5d":5,"7d":7,"1mo":30,"3mo":90,"6mo":180,
    "1y":365,"2y":730,"5y":1825,"10y":3650,"20y":7300,
}
MAX_DAYS = {"1m":7,"5m":60,"15m":60,"1h":730,"1d":36500,"1wk":36500}

STRATEGIES = ["EMA Crossover","Simple Buy","Simple Sell"]
SL_TYPES   = ["Custom Points","Trailing SL","Reverse EMA Crossover","Risk Reward Based","ATR Based"]
TGT_TYPES  = ["Custom Points","Trailing Target","EMA Crossover","Risk Reward Based","ATR Based"]

UP  = {"light":"#00b896","dark":"#00e5b4"}
DN  = {"light":"#e0284a","dark":"#ff4d6d"}
ACC = {"light":"#007a6a","dark":"#00e5b4"}

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _dur(t0, t1) -> str:
    try:
        s = int(abs((t1 - t0).total_seconds()))
        return f"{s//3600:02d}h {(s%3600)//60:02d}m {s%60:02d}s"
    except Exception:
        return "—"

# ════════════════════════════════════════════════════════════════════════════
# DATA FETCH  (cached only for historical OHLCV — never for live LTP)
# ════════════════════════════════════════════════════════════════════════════

def _clean(raw) -> pd.DataFrame | None:
    """Normalise yfinance output to a clean IST-indexed OHLCV DataFrame."""
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return None
    df = raw.copy()
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    need = ["Open","High","Low","Close"]
    if not all(c in df.columns for c in need):
        return None
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df   = df[keep].copy()
    # Ensure 1-D numeric columns (yfinance sometimes returns 2-D DataFrames)
    for c in keep:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:,0]
        df[c] = pd.to_numeric(col.squeeze(), errors="coerce")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df.dropna(subset=need, inplace=True)
    df.sort_index(inplace=True)
    return df if not df.empty else None


def _warmup_for(period: str, interval: str) -> str:
    max_d = MAX_DAYS.get(interval, 36500)
    pref  = WARMUP.get(period, period)
    if P_DAYS.get(pref, 9999) <= max_d:
        return pref
    ok = {p:d for p,d in P_DAYS.items() if d <= max_d}
    return max(ok, key=lambda p: ok[p]) if ok else period


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_ohlcv(ticker: str, interval: str, period: str):
    """
    Cached historical fetch. Key includes all three args so switching
    timeframe always fetches fresh data for the new combination.
    Returns (df_full, df_display) or None.
    """
    try:
        wu = _warmup_for(period, interval)
        df_full = _clean(yf.download(ticker, period=wu, interval=interval,
                                     auto_adjust=True, progress=False, prepost=False))
        if df_full is None or df_full.empty:
            return None
        d = P_DAYS.get(period)
        if d:
            cut  = datetime.now(IST) - timedelta(days=d)
            disp = df_full[df_full.index >= cut].copy()
            if disp.empty: disp = df_full.copy()
        else:
            disp = df_full.copy()
        return df_full, disp
    except Exception:
        return None


def _fetch_ltp_fresh(ticker: str) -> dict | None:
    """
    Fresh LTP fetch — NO cache. Called only for the LTP widget.
    Prevents cross-ticker contamination from st.cache_data.
    """
    try:
        df = _clean(yf.download(ticker, period="5d", interval="1d",
                                auto_adjust=True, progress=False))
        if df is None or len(df) < 2:
            return None
        prev = float(df["Close"].iloc[-2])
        ltp  = float(df["Close"].iloc[-1])
        chg  = ltp - prev
        pct  = chg / prev * 100 if prev else 0
        return {"price": ltp, "change": chg, "pct": pct, "prev": prev,
                "ticker": ticker}
    except Exception:
        return None

# ════════════════════════════════════════════════════════════════════════════
# INDICATORS  (TradingView-identical EMA)
# ════════════════════════════════════════════════════════════════════════════

def tv_ema(s: pd.Series, n: int) -> pd.Series:
    """EMA matching TradingView exactly: alpha=2/(n+1), adjust=False, seed from bar 0."""
    v = s.squeeze()
    if not isinstance(v, pd.Series):
        v = pd.Series(v)
    return v.astype(float).ewm(span=n, adjust=False, min_periods=1).mean()


def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi = df["High"].astype(float); lo = df["Low"].astype(float)
    pc = df["Close"].astype(float).shift(1)
    tr = pd.concat([hi-lo, (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=1).mean()


def angle_deg(ema: pd.Series, i: int) -> float:
    if i < 1 or i >= len(ema): return 0.0
    p = float(ema.iloc[i-1]); c = float(ema.iloc[i])
    return 0.0 if p == 0 else abs(float(np.degrees(np.arctan((c-p)/p*100))))


def add_indicators(df: pd.DataFrame, fast: int, slow: int, atr_n: int=14) -> pd.DataFrame:
    df = df.copy()
    ef = tv_ema(df["Close"], fast); es = tv_ema(df["Close"], slow)
    df[f"EMA_{fast}"] = ef; df[f"EMA_{slow}"] = es
    df["ATR"] = calc_atr(df, atr_n)
    df["Signal"] = 0
    df.loc[(ef > es) & (ef.shift(1) <= es.shift(1)), "Signal"] =  1  # bullish cross
    df.loc[(ef < es) & (ef.shift(1) >= es.shift(1)), "Signal"] = -1  # bearish cross
    return df

# ════════════════════════════════════════════════════════════════════════════
# SL / TARGET  CALCULATORS
# ════════════════════════════════════════════════════════════════════════════

def get_sl(ep, tt, sl_type, sl_pts, ef, es, rr, atr=0., atr_m=1.5):
    s = 1 if tt=="buy" else -1
    if sl_type == "Custom Points":         return ep - s*sl_pts
    if sl_type == "Trailing SL":           return ep - s*sl_pts
    if sl_type == "Reverse EMA Crossover":
        b = es if tt=="buy" else ef; fb = ep-s*sl_pts
        return min(b,fb) if tt=="buy" else max(b,fb)
    if sl_type == "Risk Reward Based":     return ep - s*sl_pts
    if sl_type == "ATR Based":
        v = atr if atr > 0 else sl_pts;   return ep - s*v*atr_m
    return ep - s*sl_pts


def get_tgt(ep, tt, tgt_type, tgt_pts, sl, ef, es, rr, atr=0., atr_m=2.0):
    s = 1 if tt=="buy" else -1
    if tgt_type == "Custom Points":    return ep + s*tgt_pts
    if tgt_type == "Trailing Target":  return ep + s*tgt_pts  # display only
    if tgt_type == "EMA Crossover":    return None
    if tgt_type == "Risk Reward Based":
        risk = abs(ep-sl) if sl is not None else tgt_pts
        return ep + s*risk*rr
    if tgt_type == "ATR Based":
        v = atr if atr > 0 else tgt_pts;  return ep + s*v*atr_m
    return ep + s*tgt_pts

# ════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════════════
#
#  EMA Crossover  : signal on CLOSE of bar N  →  entry at OPEN of bar N+1
#  Simple Buy/Sell: entry at OPEN of every bar where no position is active
#                   (no cooldown — tests raw SL/Target behaviour fully)
#
#  EXIT (both strategies):
#    BUY  : SL vs bar LOW first,  then Target vs bar HIGH
#    SELL : SL vs bar HIGH first, then Target vs bar LOW
#    When both breach same bar → SL wins (worst-case conservative)
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(df_full, df_display, strategy, fast, slow,
                 sl_type, sl_pts, tgt_type, tgt_pts, qty, rr,
                 atr_n, atr_sl_m, atr_tgt_m,
                 adv, min_ang, co_can, cs_pts, ca_m) -> pd.DataFrame:

    df = add_indicators(df_full.copy(), fast, slow, atr_n)
    display_start = df_display.index[0] if not df_display.empty else df.index[0]
    start_i = max(slow * 3, int(df.index.searchsorted(display_start)))
    start_i = min(start_i, len(df) - 1)

    trades        = []
    active        = None   # current open position
    pending_entry = None   # EMA crossover: set at signal bar, executed at next bar open

    for i in range(start_i, len(df)):
        row   = df.iloc[i]
        ts    = df.index[i]
        op    = float(row["Open"])
        hi    = float(row["High"])
        lo    = float(row["Low"])
        cl    = float(row["Close"])
        sig   = int(row.get("Signal", 0))
        ef    = float(row[f"EMA_{fast}"])
        es    = float(row[f"EMA_{slow}"])
        atr_v = float(row["ATR"]) if pd.notna(row.get("ATR", np.nan)) else sl_pts

        # ── Execute pending EMA crossover entry at THIS bar's OPEN ─────────
        if strategy == "EMA Crossover" and pending_entry is not None and active is None:
            ep    = op
            tt    = pending_entry["tt"]
            pef   = pending_entry["ef"]; pes = pending_entry["es"]; patr = pending_entry["atr"]
            sl_p  = get_sl(ep,tt,sl_type,sl_pts,pef,pes,rr,patr,atr_sl_m)
            tgt_p = get_tgt(ep,tt,tgt_type,tgt_pts,sl_p,pef,pes,rr,patr,atr_tgt_m)
            # Skip if open gapped past SL
            if not ((tt=="buy" and sl_p is not None and ep<=sl_p) or
                    (tt=="sell" and sl_p is not None and ep>=sl_p)):
                active = dict(ts=ts, ep=ep, tt=tt, sl=sl_p, tgt=tgt_p,
                              reason=pending_entry["reason"])
            pending_entry = None

        # ── Simple Buy/Sell: enter at THIS bar's OPEN if no position ────────
        if strategy in ["Simple Buy","Simple Sell"] and active is None:
            tt    = "buy" if strategy=="Simple Buy" else "sell"
            ep    = op
            sl_p  = get_sl(ep,tt,sl_type,sl_pts,ef,es,rr,atr_v,atr_sl_m)
            tgt_p = get_tgt(ep,tt,tgt_type,tgt_pts,sl_p,ef,es,rr,atr_v,atr_tgt_m)
            active = dict(ts=ts, ep=ep, tt=tt, sl=sl_p, tgt=tgt_p, reason=strategy)

        # ── EXIT ────────────────────────────────────────────────────────────
        if active is not None:
            ep  = active["ep"]; tt = active["tt"]
            sl  = active["sl"]; tgt = active["tgt"]

            # BUY : SL vs LOW first, then Target vs HIGH
            # SELL: SL vs HIGH first, then Target vs LOW
            if tt == "buy":
                sl_hit  = sl  is not None and lo <= sl
                tgt_hit = tgt is not None and hi >= tgt
            else:
                sl_hit  = sl  is not None and hi >= sl
                tgt_hit = tgt is not None and lo <= tgt

            ema_exit = (tgt_type=="EMA Crossover" and
                        ((tt=="buy" and sig==-1) or (tt=="sell" and sig==1)))

            exit_px = exit_rsn = None
            if sl_hit:
                exit_px  = sl
                exit_rsn = "SL Hit" + (" & Target same candle" if tgt_hit else "")
            elif tgt_hit:
                exit_px  = tgt;   exit_rsn = "Target Hit"
            elif ema_exit:
                exit_px  = cl;    exit_rsn = "EMA Crossover Exit"

            # Trailing SL update when no exit
            if sl_type == "Trailing SL" and exit_px is None:
                nsl = (cl - sl_pts) if tt=="buy" else (cl + sl_pts)
                if (tt=="buy" and nsl > active["sl"]) or (tt=="sell" and nsl < active["sl"]):
                    active["sl"] = nsl

            if exit_px is not None:
                pnl = ((float(exit_px)-ep) if tt=="buy" else (ep-float(exit_px))) * qty
                pts = (float(exit_px)-ep) if tt=="buy" else (ep-float(exit_px))
                trades.append(dict(
                    **{"Entry Time": active["ts"], "Exit Time": ts,
                       "Duration": _dur(active["ts"], ts),
                       "Trade Type": tt.upper(),
                       "Entry Price": round(ep,2), "Exit Price": round(float(exit_px),2),
                       "SL":     round(float(active["sl"]),2) if active["sl"]  is not None else "—",
                       "Target": round(float(tgt),2)          if tgt           is not None else "—",
                       "Candle High": round(hi,2), "Candle Low": round(lo,2),
                       "Entry Reason": active["reason"], "Exit Reason": exit_rsn,
                       "Points": round(pts,2), "PnL (Rs)": round(pnl,2), "Qty": qty}
                ))
                active = None

        # ── EMA Crossover signal detection ──────────────────────────────────
        #   Only set pending_entry; executes at next bar's OPEN
        if strategy=="EMA Crossover" and sig!=0 and active is None and pending_entry is None:
            allow = True
            info  = ""
            if adv:
                ang = angle_deg(df[f"EMA_{fast}"], i); info = f" ({ang:.1f}°)"
                if ang < min_ang:
                    allow = False
                elif co_can != "Simple Crossover":
                    body = abs(cl - op)
                    if co_can=="Custom Candle Size" and body < cs_pts: allow = False
                    elif co_can=="ATR Based Candle Size" and body < ca_m*atr_v: allow = False
            if allow:
                tt_ = "buy" if sig==1 else "sell"
                pending_entry = dict(tt=tt_, ef=ef, es=es, atr=atr_v,
                                     reason=f"EMA{fast}×EMA{slow} {'↑' if sig==1 else '↓'}{info}")

    # Force-close open trade at end of data
    if active is not None:
        lr  = df.iloc[-1]; ts_e = df.index[-1]; ep_x = float(lr["Close"])
        pnl = ((ep_x-active["ep"]) if active["tt"]=="buy" else (active["ep"]-ep_x))*qty
        pts = (ep_x-active["ep"]) if active["tt"]=="buy" else (active["ep"]-ep_x)
        trades.append(dict(
            **{"Entry Time": active["ts"], "Exit Time": ts_e,
               "Duration": _dur(active["ts"], ts_e),
               "Trade Type": active["tt"].upper(),
               "Entry Price": round(active["ep"],2), "Exit Price": round(ep_x,2),
               "SL":     round(float(active["sl"]),2)  if active["sl"]  is not None else "—",
               "Target": round(float(active["tgt"]),2) if active["tgt"] is not None else "—",
               "Candle High": round(float(lr["High"]),2), "Candle Low": round(float(lr["Low"]),2),
               "Entry Reason": active["reason"], "Exit Reason": "End of Data",
               "Points": round(pts,2), "PnL (Rs)": round(pnl,2), "Qty": qty}
        ))
    return pd.DataFrame(trades)

# ════════════════════════════════════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════════════════════════════════════

def _base_layout(light: bool) -> dict:
    bg  = "#f5f7fa" if light else "#080b12"
    plt = "#ffffff" if light else "#080b12"
    grd = "#e8edf5" if light else "#0f1623"
    txt = "#334155" if light else "#6b7a99"
    spk = "#94a3b8" if light else "#2a3456"
    return dict(paper_bgcolor=bg, plot_bgcolor=plt,
                font=dict(family="JetBrains Mono",color=txt,size=11),
                xaxis_rangeslider_visible=False,
                margin=dict(t=38,b=14,l=44,r=20),
                legend=dict(orientation="h",yanchor="bottom",y=1.01,bgcolor="rgba(0,0,0,0)"),
                hoverlabel=dict(bgcolor="#fff" if light else "#0f1623",
                                bordercolor="#d0d7e6" if light else "#1a2236",
                                font=dict(family="JetBrains Mono",size=11)),
                xaxis=dict(gridcolor=grd,zeroline=False,showspikes=True,spikecolor=spk,spikethickness=1),
                yaxis=dict(gridcolor=grd,zeroline=False,showspikes=True,spikecolor=spk,spikethickness=1))


def build_chart(df, fast, slow, trades_df=None, position=None, title="", light=False) -> go.Figure:
    up_ = UP["light"] if light else UP["dark"]
    dn_ = DN["light"] if light else DN["dark"]
    bg0 = "#000" if light else "#080b12"

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=list(df.index),
        open=list(df["Open"].astype(float)), high=list(df["High"].astype(float)),
        low=list(df["Low"].astype(float)),   close=list(df["Close"].astype(float)),
        name="OHLC",
        increasing=dict(line=dict(color=up_,width=1.2),fillcolor=up_),
        decreasing=dict(line=dict(color=dn_,width=1.2),fillcolor=dn_),
        whiskerwidth=0.4), row=1, col=1)

    for col_,clr,nm in [(f"EMA_{fast}","#f59e0b",f"EMA {fast}"),
                         (f"EMA_{slow}","#3b82f6",f"EMA {slow}")]:
        if col_ in df.columns:
            fig.add_trace(go.Scatter(x=list(df.index),y=list(df[col_].astype(float)),
                                     name=nm,line=dict(color=clr,width=1.8)), row=1, col=1)

    if trades_df is not None and not trades_df.empty:
        for _,t in trades_df.iterrows():
            tt_  = str(t["Trade Type"]); c_ = up_ if tt_=="BUY" else dn_
            esy  = "triangle-up"   if tt_=="BUY" else "triangle-down"
            exy  = "triangle-down" if tt_=="BUY" else "triangle-up"
            ep_  = float(t["Entry Price"]); xp_ = float(t["Exit Price"]); p_ = float(t["PnL (Rs)"])
            fig.add_trace(go.Scatter(x=[t["Entry Time"]],y=[ep_],mode="markers",showlegend=False,
                marker=dict(symbol=esy,size=13,color=c_,line=dict(width=1.5,color="#fff")),
                hovertemplate=f"<b>{tt_} ENTRY</b><br>{ep_:.2f}<br>{t.get('Entry Reason','')}<extra></extra>"),
                row=1,col=1)
            fig.add_trace(go.Scatter(x=[t["Exit Time"]],y=[xp_],mode="markers",showlegend=False,
                marker=dict(symbol=exy,size=11,color=c_,opacity=0.6,line=dict(width=1,color=bg0)),
                hovertemplate=f"<b>{tt_} EXIT</b><br>{xp_:.2f}<br>{t.get('Exit Reason','')}<br>PnL Rs{p_:+.2f}<extra></extra>"),
                row=1,col=1)

    if position is not None:
        lbls = [("entry_price","Entry","#666" if light else "#fff"),
                ("sl","SL",dn_),("target","Target",up_)]
        for key,lbl,clr in lbls:
            v = position.get(key)
            if v is not None:
                fig.add_hline(y=float(v),row=1,col=1,
                              line=dict(color=clr,width=1.3,dash="solid" if lbl=="Entry" else "dash"),
                              annotation_text=f" {lbl} {float(v):.2f}",
                              annotation_font=dict(color=clr,size=11))

    if "Volume" in df.columns:
        vc = [up_ if float(c)>=float(o) else dn_ for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=list(df.index),y=list(df["Volume"].fillna(0).astype(float)),
                             marker_color=vc,name="Vol",showlegend=False), row=2, col=1)

    base = _base_layout(light); xax = base.pop("xaxis",{}); yax = base.pop("yaxis",{})
    fig.update_layout(height=560,
                      title=dict(text=title,x=0.01,font=dict(family="Syne",size=13,
                                 color="#334155" if light else "#6b7a99")),**base)
    fig.update_xaxes(**xax); fig.update_yaxes(**yax)
    return fig

# ════════════════════════════════════════════════════════════════════════════
# LTP WIDGET  — takes an explicit `info` dict so the caller decides what data to show
# ════════════════════════════════════════════════════════════════════════════
def ltp_widget(label: str, info: dict | None, light: bool) -> None:
    if info is None or info.get("ticker") is None:
        return
    p,c,pct = info["price"], info["change"], info["pct"]
    up_ = ACC["light"] if light else ACC["dark"]
    dn_ = DN["light"]  if light else DN["dark"]
    col = up_ if c>=0 else dn_; arr = "▲" if c>=0 else "▼"
    bg  = "#fff"    if light else "#0f1623"
    brd = "#d0d7e6" if light else "#1a2236"
    tx  = "#1a2236" if light else "#e2e8f0"
    mt  = "#7a8899" if light else "#4a5568"
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {brd};border-radius:12px;
                padding:12px 20px;margin-bottom:12px;display:flex;
                align-items:center;gap:24px;flex-wrap:wrap">
        <div>
            <div style="font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:{mt};font-family:Syne,sans-serif">{label}</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:26px;
                        font-weight:700;color:{tx};line-height:1.1">{p:,.2f}</div>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:600;color:{col}">{arr} {c:+.2f}</div>
        <div style="font-family:JetBrains Mono,monospace;font-size:14px;font-weight:600;color:{col}">{pct:+.2f}%</div>
        <div style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:11px;color:{mt}">
            Prev: {info['prev']:,.2f}
        </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DHAN BROKER
# ════════════════════════════════════════════════════════════════════════════
def _dhan(cfg): return dhanhq(str(cfg["dhan_client_id"]),str(cfg["dhan_access_token"]))

def dhan_entry(cfg, tt, ltp):
    if not HAS_DHAN: return "dhanhq not installed"
    try:
        d = _dhan(cfg)
        if cfg.get("options_trading"):
            sec=cfg["ce_security_id"] if tt=="buy" else cfg["pe_security_id"]
            ot ="MARKET" if cfg.get("opts_entry_otype")=="Market Order" else "LIMIT"
            r  =d.place_order(transactionType="BUY",
                               exchangeSegment=str(cfg.get("opts_exchange","NSE_FNO")),
                               productType="INTRADAY",orderType=ot,validity="DAY",
                               securityId=str(sec),quantity=int(cfg.get("opts_qty",65)),
                               price=float(ltp) if ot=="LIMIT" else 0.0,triggerPrice=0)
        else:
            ex ="NSE_EQ" if cfg.get("eq_exchange","NSE")=="NSE" else "BSE"
            pt ="INTRADAY" if cfg.get("eq_product")=="Intraday" else "CNC"
            ot ="MARKET" if cfg.get("eq_entry_otype")=="Market Order" else "LIMIT"
            r  =d.place_order(security_id=str(cfg.get("eq_sec_id","1594")),
                               exchange_segment=ex,
                               transaction_type="BUY" if tt=="buy" else "SELL",
                               quantity=int(cfg.get("eq_qty",1)),order_type=ot,
                               product_type=pt,price=float(ltp) if ot=="LIMIT" else 0.0)
        return f"Entry OK: {r}"
    except Exception as e: return f"Entry FAILED: {e}"

def dhan_exit(cfg, tt, ltp):
    if not HAS_DHAN: return "dhanhq not installed"
    try:
        d = _dhan(cfg)
        if cfg.get("options_trading"):
            sec=cfg["ce_security_id"] if tt=="buy" else cfg["pe_security_id"]
            ot ="MARKET" if cfg.get("opts_exit_otype","Market Order")=="Market Order" else "LIMIT"
            r  =d.place_order(transactionType="SELL",
                               exchangeSegment=str(cfg.get("opts_exchange","NSE_FNO")),
                               productType="INTRADAY",orderType=ot,validity="DAY",
                               securityId=str(sec),quantity=int(cfg.get("opts_qty",65)),
                               price=float(ltp) if ot=="LIMIT" else 0.0,triggerPrice=0)
        else:
            ex ="NSE_EQ" if cfg.get("eq_exchange","NSE")=="NSE" else "BSE"
            pt ="INTRADAY" if cfg.get("eq_product")=="Intraday" else "CNC"
            ot ="MARKET" if cfg.get("eq_exit_otype","Market Order")=="Market Order" else "LIMIT"
            r  =d.place_order(security_id=str(cfg.get("eq_sec_id","1594")),
                               exchange_segment=ex,
                               transaction_type="SELL" if tt=="buy" else "BUY",
                               quantity=int(cfg.get("eq_qty",1)),order_type=ot,
                               product_type=pt,price=float(ltp) if ot=="LIMIT" else 0.0)
        return f"Exit OK: {r}"
    except Exception as e: return f"Exit FAILED: {e}"

# ════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE  (background daemon thread)
# ════════════════════════════════════════════════════════════════════════════
#
#  Thread-safety: ALL reads and writes to LiveState go through .set()/.get()
#  which hold an RLock. The UI always calls .snapshot() to get a consistent
#  dict copy. "dictionary changed size during iteration" is impossible.
#
#  Stop reliability: stop_event.set() is immediately seen by the thread's
#  stop_event.wait(1.5) call. Thread exits within 1.5 seconds guaranteed.
#
#  "Can't start new thread": Before starting, the old thread is joined with
#  a 3-second timeout so OS resources are freed before new thread starts.
# ════════════════════════════════════════════════════════════════════════════

def live_engine(cfg: dict, state: LiveState, stop_event: threading.Event) -> None:
    ticker    = cfg["ticker"];    interval = cfg["interval"];  period = cfg["period"]
    strategy  = cfg["strategy"];  fast     = cfg["fast_ema"]; slow   = cfg["slow_ema"]
    sl_type   = cfg["sl_type"];   sl_pts   = cfg["sl_pts"]
    tgt_type  = cfg["tgt_type"];  tgt_pts  = cfg["tgt_pts"]
    qty       = cfg["qty"];       cd_en    = cfg["cd_en"];    cd_s   = cfg["cd_s"]
    rr        = cfg["rr"]
    atr_n     = cfg.get("atr_period",14)
    atr_sl_m  = cfg.get("atr_sl_mult",1.5)
    atr_tgt_m = cfg.get("atr_tgt_mult",2.0)
    adv       = cfg.get("adv_filter",False)
    min_ang   = cfg.get("min_angle",0.0)
    co_can    = cfg.get("crossover_candle","Simple Crossover")
    cs_pts    = cfg.get("candle_size_pts",10.0)
    ca_m      = cfg.get("candle_atr_mult",1.0)

    state.add_log(f"Engine started — {ticker} {interval}/{period} [{strategy}]")
    last_closed_ts = None

    while not stop_event.is_set():
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=True, progress=False, prepost=False)
            df  = _clean(raw)

            if df is None or len(df) < max(slow*2+5, 10):
                state.add_log("Insufficient data — waiting…")
                stop_event.wait(1.5); continue

            # Build indicators
            df[f"EMA_{fast}"] = tv_ema(df["Close"], fast)
            df[f"EMA_{slow}"] = tv_ema(df["Close"], slow)
            df["ATR"]         = calc_atr(df, atr_n)

            # Completed-candle values (iloc[-2] = last closed bar)
            ef_c = float(df[f"EMA_{fast}"].iloc[-2])
            es_c = float(df[f"EMA_{slow}"].iloc[-2])
            at_c = float(df["ATR"].iloc[-2])
            ltp  = float(df["Close"].iloc[-1])   # forming bar close = LTP

            # Update shared state atomically
            state.set("chart_df", df.copy())
            state.set("ema_fast", ef_c)
            state.set("ema_slow", es_c)
            state.set("atr",      at_c)
            state.set("ltp",      ltp)

            # ── A: EMA Crossover — execute pending at N+1 OPEN ───────────
            pen = state.get("pending_entry")
            if pen is not None and state.get("position") is None:
                ep    = float(df["Open"].iloc[-1])   # N+1 open price
                tt    = pen["tt"]
                sl_p  = get_sl(ep,tt,sl_type,sl_pts,pen["ef"],pen["es"],rr,pen["atr"],atr_sl_m)
                tgt_p = get_tgt(ep,tt,tgt_type,tgt_pts,sl_p,pen["ef"],pen["es"],rr,pen["atr"],atr_tgt_m)
                gap   = ((tt=="buy"  and sl_p is not None and ep<=sl_p) or
                         (tt=="sell" and sl_p is not None and ep>=sl_p))
                if not gap:
                    state.set("position", dict(trade_type=tt, entry_price=ep,
                                               entry_time=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                                               sl=sl_p, target=tgt_p, entry_reason=pen["reason"]))
                    state.add_log(f"ENTRY {tt.upper()} @ Open {ep:.2f} | SL:{sl_p:.2f if sl_p else '—'} | Tgt:{tgt_p:.2f if tgt_p else '—'}")
                    if cfg.get("dhan_en"): state.add_log(dhan_entry(cfg,tt,ep))
                else:
                    state.add_log(f"Entry skipped — open {ep:.2f} gapped past SL")
                state.set("pending_entry", None)

            # ── B: Simple Buy/Sell — enter at LTP immediately ────────────
            if strategy in ["Simple Buy","Simple Sell"] and state.get("position") is None:
                enter = True
                if cd_en:
                    lx = state.get("last_exit_ts")
                    if lx and (datetime.now(IST)-lx).total_seconds() < cd_s:
                        enter = False
                if enter:
                    tt    = "buy" if strategy=="Simple Buy" else "sell"
                    sl_p  = get_sl(ltp,tt,sl_type,sl_pts,ef_c,es_c,rr,at_c,atr_sl_m)
                    tgt_p = get_tgt(ltp,tt,tgt_type,tgt_pts,sl_p,ef_c,es_c,rr,at_c,atr_tgt_m)
                    state.set("position", dict(trade_type=tt, entry_price=ltp,
                                               entry_time=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                                               sl=sl_p, target=tgt_p, entry_reason=strategy))
                    state.add_log(f"ENTRY {tt.upper()} @ LTP {ltp:.2f} | SL:{sl_p:.2f if sl_p else '—'} | Tgt:{tgt_p:.2f if tgt_p else '—'}")
                    if cfg.get("dhan_en"): state.add_log(dhan_entry(cfg,tt,ltp))

            # ── C: SL/Target check vs LTP every tick (conservative) ──────
            pos = state.get("position")
            if pos is not None:
                tt  = pos["trade_type"]; ep = float(pos["entry_price"])
                sl  = pos.get("sl");    tgt = pos.get("target")
                exited = False; exit_px = None; exit_rsn = None

                if tt == "buy":
                    if sl  is not None and ltp <= sl:
                        exit_px=ltp; exit_rsn="SL Hit (LTP≤SL)"; exited=True
                    elif tgt is not None and ltp >= tgt:
                        exit_px=ltp; exit_rsn="Target Hit (LTP≥Tgt)"; exited=True
                else:
                    if sl  is not None and ltp >= sl:
                        exit_px=ltp; exit_rsn="SL Hit (LTP≥SL)"; exited=True
                    elif tgt is not None and ltp <= tgt:
                        exit_px=ltp; exit_rsn="Target Hit (LTP≤Tgt)"; exited=True

                # EMA crossover exit on completed bar
                if not exited and tgt_type=="EMA Crossover" and len(df)>=3:
                    pf=float(df[f"EMA_{fast}"].iloc[-3]); ps=float(df[f"EMA_{slow}"].iloc[-3])
                    if tt=="buy"  and ef_c<es_c and pf>=ps: exit_px=ltp; exit_rsn="EMA X Exit"; exited=True
                    if tt=="sell" and ef_c>es_c and pf<=ps: exit_px=ltp; exit_rsn="EMA X Exit"; exited=True

                # Trailing SL
                if sl_type=="Trailing SL" and not exited:
                    nsl = (ltp-sl_pts) if tt=="buy" else (ltp+sl_pts)
                    if ((tt=="buy" and nsl>pos["sl"]) or (tt=="sell" and nsl<pos["sl"])):
                        pos["sl"] = nsl
                        state.set("position", pos)
                        state.add_log(f"Trailing SL → {nsl:.2f}")

                if exited and exit_px is not None:
                    pnl = ((float(exit_px)-ep) if tt=="buy" else (ep-float(exit_px)))*qty
                    pts = (float(exit_px)-ep)   if tt=="buy" else (ep-float(exit_px))
                    et  = pos["entry_time"]
                    try: et_dt = datetime.strptime(et,"%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    except: et_dt = datetime.now(IST)
                    xt_dt = datetime.now(IST)
                    state.add_trade(dict(
                        **{"Entry Time":et, "Exit Time":xt_dt.strftime("%Y-%m-%d %H:%M:%S"),
                           "Duration":_dur(et_dt,xt_dt), "Trade Type":tt.upper(),
                           "Entry Price":round(ep,2), "Exit Price":round(float(exit_px),2),
                           "SL":   round(float(sl),2)  if sl  is not None else "—",
                           "Target":round(float(tgt),2) if tgt is not None else "—",
                           "Entry Reason":pos.get("entry_reason",""), "Exit Reason":exit_rsn,
                           "Points":round(pts,2), "PnL (Rs)":round(pnl,2),
                           "Qty":qty, "Mode":"Live"}
                    ))
                    state.set("position", None)
                    state.set("last_exit_ts", datetime.now(IST))
                    if cfg.get("dhan_en"): state.add_log(dhan_exit(cfg,tt,float(exit_px)))
                    state.add_log(f"EXIT {tt.upper()} @ {float(exit_px):.2f} | {exit_rsn} | Rs{pnl:+.2f} ({pts:+.2f}pts)")

            # ── D: EMA crossover signal detection on new completed candle ─
            cur_ts = df.index[-2] if len(df)>=2 else None
            if strategy=="EMA Crossover" and cur_ts is not None and cur_ts!=last_closed_ts:
                last_closed_ts = cur_ts
                if state.get("position") is None and state.get("pending_entry") is None:
                    if cd_en:
                        lx = state.get("last_exit_ts")
                        if lx and (datetime.now(IST)-lx).total_seconds() < cd_s:
                            state.add_log(f"Cooldown…")
                            stop_event.wait(1.5); continue
                    if len(df) >= 3:
                        pf=float(df[f"EMA_{fast}"].iloc[-3]); ps=float(df[f"EMA_{slow}"].iloc[-3])
                        bull = ef_c>es_c and pf<=ps; bear = ef_c<es_c and pf>=ps
                        if bull or bear:
                            ok = True; ai = ""
                            if adv:
                                ang=angle_deg(df[f"EMA_{fast}"],len(df)-2); ai=f" ({ang:.1f}°)"
                                if ang < min_ang: ok = False
                                elif co_can != "Simple Crossover":
                                    body=abs(float(df["Close"].iloc[-2])-float(df["Open"].iloc[-2]))
                                    if co_can=="Custom Candle Size" and body<cs_pts: ok=False
                                    elif co_can=="ATR Based Candle Size" and body<ca_m*at_c: ok=False
                            if ok:
                                sig_tt = "buy" if bull else "sell"
                                rsn    = f"EMA{fast}×EMA{slow} {'↑' if bull else '↓'}{ai}"
                                state.set("pending_entry", dict(tt=sig_tt,reason=rsn,
                                                                ef=ef_c,es=es_c,atr=at_c))
                                state.add_log(f"Signal: {sig_tt.upper()} on {str(cur_ts)[:16]} — entry next open")

        except Exception as exc:
            state.add_log(f"Error: {exc}")

        stop_event.wait(1.5)   # replaces time.sleep — responds to stop instantly

    state.set("running", False)
    state.add_log("Engine stopped")

# ════════════════════════════════════════════════════════════════════════════
# STOP ENGINE HELPER
# ════════════════════════════════════════════════════════════════════════════
def stop_engine():
    """Signal stop and join old thread (frees OS resources before new thread starts)."""
    evt = st.session_state.get("stop_event")
    if evt: evt.set()
    t = st.session_state.get("live_thread")
    if t is not None and t.is_alive():
        t.join(timeout=3.0)   # wait max 3 s for thread to exit
    st.session_state["live_thread"] = None
    st.session_state["live_state"].set("running", False)

# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    light = st.session_state.get("light_theme", True)
    inject_css(light)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        ac = ACC["light"] if light else ACC["dark"]
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0 10px">
            <div style="font-size:36px">📈</div>
            <div style="font-family:Syne,sans-serif;font-size:18px;font-weight:800;
                        letter-spacing:3px;color:{ac}">SMART INVESTING</div>
            <div style="font-size:9px;color:#8899aa;letter-spacing:2px;margin-top:2px">ALGORITHMIC TRADING</div>
        </div>""", unsafe_allow_html=True)

        lt = st.checkbox("☀ Light Theme", value=light, key="light_theme")
        if lt != light: st.rerun()
        st.divider()

        st.markdown('<div class="shdr">Asset</div>', unsafe_allow_html=True)
        choice = st.selectbox("Ticker", list(TICKERS.keys())+["Custom →"],
                              label_visibility="collapsed")
        if choice=="Custom →":
            ticker_sym=st.text_input("Symbol","RELIANCE.NS"); ticker_label=ticker_sym
        else:
            ticker_sym=TICKERS[choice]; ticker_label=choice

        st.markdown('<div class="shdr">Timeframe</div>', unsafe_allow_html=True)
        tf_keys = list(TF_PERIODS.keys())
        c1,c2 = st.columns(2)
        with c1: interval = st.selectbox("Interval",tf_keys,index=tf_keys.index("5m"),
                                          label_visibility="collapsed")
        with c2:
            prd_opts  = TF_PERIODS[interval]
            dflt      = prd_opts.index("1mo") if "1mo" in prd_opts else min(1,len(prd_opts)-1)
            period    = st.selectbox("Period",prd_opts,index=dflt,label_visibility="collapsed")
        st.divider()

        st.markdown('<div class="shdr">Strategy</div>', unsafe_allow_html=True)
        strategy = st.selectbox("Strategy",STRATEGIES,label_visibility="collapsed")
        fast_ema=9; slow_ema=15; adv_filter=False; min_angle=0.0
        crossover_candle="Simple Crossover"; candle_size_pts=10.0; candle_atr_mult=1.0

        if strategy=="EMA Crossover":
            e1,e2=st.columns(2)
            with e1: fast_ema=st.number_input("Fast EMA",2,200,9)
            with e2: slow_ema=st.number_input("Slow EMA",2,500,15)
            adv_filter=st.checkbox("Advanced Entry Filters",False,
                                   help="Off = plain EMA crossover. Enable for angle/size filters.")
            if adv_filter:
                min_angle=st.number_input("Min Angle (°)",0.0,90.0,0.0,0.5)
                crossover_candle=st.selectbox("Candle Filter",
                    ["Simple Crossover","Custom Candle Size","ATR Based Candle Size"])
                if crossover_candle=="Custom Candle Size":
                    candle_size_pts=st.number_input("Min Body (pts)",0.1,1e6,10.0,0.5)
                elif crossover_candle=="ATR Based Candle Size":
                    candle_atr_mult=st.number_input("Min Body (ATR×)",0.1,10.0,1.0,0.1)
        st.divider()

        st.markdown('<div class="shdr">Stop Loss</div>', unsafe_allow_html=True)
        sl_type=st.selectbox("SL Type",SL_TYPES,label_visibility="collapsed")
        s1,s2=st.columns(2)
        with s1: sl_pts=st.number_input("SL Points",0.1,1e6,10.0,0.5)
        with s2: rr=st.number_input("R:R",0.5,20.0,2.0,0.5)

        st.markdown('<div class="shdr">Target</div>', unsafe_allow_html=True)
        tgt_type=st.selectbox("Target Type",TGT_TYPES,label_visibility="collapsed")
        tgt_pts=st.number_input("Target Points",0.1,1e6,20.0,0.5)

        atr_period=14; atr_sl_mult=1.5; atr_tgt_mult=2.0
        if sl_type=="ATR Based" or tgt_type=="ATR Based":
            st.markdown("**ATR Config**")
            atr_period=st.number_input("ATR Period",2,200,14)
            if sl_type=="ATR Based": atr_sl_mult=st.number_input("ATR×SL",0.1,10.0,1.5,0.1)
            if tgt_type=="ATR Based": atr_tgt_mult=st.number_input("ATR×Tgt",0.1,10.0,2.0,0.1)
        st.divider()

        st.markdown('<div class="shdr">Trade Settings</div>', unsafe_allow_html=True)
        qty=st.number_input("Quantity",1,10_000_000,1)
        d1,d2=st.columns([1.4,1])
        with d1: cd_en=st.checkbox("Cooldown (Live)",True)
        with d2: cd_s=st.number_input("Secs",1,86400,5,disabled=not cd_en,label_visibility="visible")
        st.divider()

        st.markdown('<div class="shdr">Dhan Broker</div>', unsafe_allow_html=True)
        dhan_en=st.checkbox("Enable Dhan",False); dhan_cfg={"dhan_en":dhan_en}
        if dhan_en:
            if not HAS_DHAN: st.warning("pip install dhanhq")
            dhan_cfg["dhan_client_id"]=st.text_input("Client ID","",type="password")
            dhan_cfg["dhan_access_token"]=st.text_input("Access Token","",type="password")
            opts=st.checkbox("Options Trading",False); dhan_cfg["options_trading"]=opts
            if not opts:
                dhan_cfg["eq_product"]=st.selectbox("Product",["Intraday","Delivery"])
                dhan_cfg["eq_exchange"]=st.selectbox("Exchange",["NSE","BSE"])
                dhan_cfg["eq_sec_id"]=st.text_input("Sec ID","1594")
                dhan_cfg["eq_qty"]=st.number_input("Broker Qty",1,1_000_000,1)
                dhan_cfg["eq_entry_otype"]=st.selectbox("Entry Order",["Limit Order","Market Order"])
                dhan_cfg["eq_exit_otype"]=st.selectbox("Exit Order",["Market Order","Limit Order"])
            else:
                dhan_cfg["opts_exchange"]=st.selectbox("FnO Exchange",["NSE_FNO","BSE_FNO"])
                dhan_cfg["ce_security_id"]=st.text_input("CE Sec ID","")
                dhan_cfg["pe_security_id"]=st.text_input("PE Sec ID","")
                dhan_cfg["opts_qty"]=st.number_input("Lots",1,1_000_000,65)
                dhan_cfg["opts_entry_otype"]=st.selectbox("Entry Order",["Market Order","Limit Order"])
                dhan_cfg["opts_exit_otype"]=st.selectbox("Exit Order",["Market Order","Limit Order"])
        st.divider()
        st.caption("Smart Investing v6.0  •  Educational use only")

    cfg = dict(ticker=ticker_sym, ticker_label=ticker_label,
               interval=interval, period=period,
               strategy=strategy, fast_ema=fast_ema, slow_ema=slow_ema,
               sl_type=sl_type, sl_pts=sl_pts, tgt_type=tgt_type, tgt_pts=tgt_pts,
               qty=qty, cd_en=cd_en, cd_s=cd_s, rr=rr,
               atr_period=atr_period, atr_sl_mult=atr_sl_mult, atr_tgt_mult=atr_tgt_mult,
               adv_filter=adv_filter, min_angle=min_angle,
               crossover_candle=crossover_candle,
               candle_size_pts=candle_size_pts, candle_atr_mult=candle_atr_mult,
               **dhan_cfg)

    # ── Header ────────────────────────────────────────────────────────────────
    ac = ACC["light"] if light else ACC["dark"]
    st.markdown(f"""
    <div style="padding:4px 0 10px">
        <span style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:{ac}">
            📈 Smart Investing
        </span>
        <span style="font-family:Syne,sans-serif;font-size:12px;color:#7a8899;
                     margin-left:12px;vertical-align:middle">Algorithmic Trading Platform v6</span>
    </div>""", unsafe_allow_html=True)

    if not HAS_AR:
        st.info("For live auto-refresh: `pip install streamlit-autorefresh`")

    # ── Fetch LTP for the currently selected ticker (NO cache to avoid cross-ticker bugs) ──
    ltp_info = _fetch_ltp_fresh(ticker_sym)
    if ltp_info:
        ltp_info["ticker"] = ticker_sym   # tag so widget can verify

    tab_bt, tab_live, tab_hist = st.tabs(["🔬  Backtesting","⚡  Live Trading","📋  Trade History"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — BACKTESTING
    # ════════════════════════════════════════════════════════════════════════
    with tab_bt:
        ltp_widget(ticker_label, ltp_info, light)
        st.markdown("### 🔬 Backtest Engine")

        r1,r2 = st.columns([4,1])
        with r1:
            run_btn = st.button("▶  Run Backtest", type="primary",
                                use_container_width=True, key="run_bt")
        with r2:
            if st.button("↺ Clear", use_container_width=True, key="clr_bt"):
                st.session_state["bt_results"]  = None
                st.session_state["bt_chart_df"] = None
                st.session_state["bt_run_key"]  = None
                st.rerun()

        if run_btn:
            # Clear previous results before running
            st.session_state["bt_results"]  = None
            st.session_state["bt_chart_df"] = None
            with st.spinner(f"Fetching {ticker_label} {interval}/{period}…"):
                res = _fetch_ohlcv(ticker_sym, interval, period)
            if res is None:
                st.error(f"Data fetch failed for {ticker_sym} {interval}/{period}. "
                         f"Check ticker or try a different period.")
            else:
                df_full, df_disp = res
                with st.spinner("Running backtest…"):
                    tdf = run_backtest(df_full, df_disp, strategy, fast_ema, slow_ema,
                                       sl_type, sl_pts, tgt_type, tgt_pts, qty, rr,
                                       atr_period, atr_sl_mult, atr_tgt_mult,
                                       adv_filter, min_angle, crossover_candle,
                                       candle_size_pts, candle_atr_mult)
                df_chart = add_indicators(df_full.copy(), fast_ema, slow_ema, atr_period)
                if not df_disp.empty:
                    df_chart = df_chart.loc[df_disp.index[0]:]
                st.session_state["bt_results"]  = tdf
                st.session_state["bt_chart_df"] = df_chart
                st.session_state["bt_run_key"]  = f"{ticker_sym}|{interval}|{period}|{strategy}"

        tdf     = st.session_state.get("bt_results")
        dfc     = st.session_state.get("bt_chart_df")
        run_key = st.session_state.get("bt_run_key","")

        if tdf is not None:
            if run_key: st.caption(f"Results: `{run_key}`")
            if tdf.empty:
                st.warning("No trades generated. Adjust parameters or check data.")
            else:
                n    = len(tdf)
                wp   = tdf["PnL (Rs)"] > 0; lp = tdf["PnL (Rs)"] < 0
                nw   = int(wp.sum()); nl = int(lp.sum())
                tot  = float(tdf["PnL (Rs)"].sum())
                acc  = nw/n*100 if n else 0
                avgw = float(tdf.loc[wp,"PnL (Rs)"].mean()) if nw else 0.0
                avgl = float(tdf.loc[lp,"PnL (Rs)"].mean()) if nl else 0.0
                best = float(tdf["PnL (Rs)"].max()); wst = float(tdf["PnL (Rs)"].min())
                twp  = float(tdf.loc[wp,"Points"].sum()) if "Points" in tdf.columns else 0.0
                twl  = float(tdf.loc[lp,"Points"].sum()) if "Points" in tdf.columns else 0.0
                viols= int(tdf["Exit Reason"].str.contains("same candle",na=False).sum()) if "Exit Reason" in tdf.columns else 0

                m = st.columns(8)
                for c_,l_,v_ in zip(m,
                    ["Trades","Wins","Losses","Accuracy","Total PnL","Avg Win","Avg Loss","Violations"],
                    [n,nw,nl,f"{acc:.1f}%",f"Rs{tot:+,.0f}",f"Rs{avgw:+.0f}",f"Rs{avgl:+.0f}",viols]):
                    c_.metric(l_,v_)
                m2 = st.columns(5)
                for c_,l_,v_ in zip(m2,
                    ["Win Points","Loss Points","Best Trade","Worst Trade","Avg Duration"],
                    [f"{twp:+.1f}",f"{twl:+.1f}",f"Rs{best:+.0f}",f"Rs{wst:+.0f}",
                     tdf["Duration"].iloc[0] if "Duration" in tdf.columns and n>0 else "—"]):
                    c_.metric(l_,v_)

                st.divider()

                if dfc is not None:
                    st.plotly_chart(build_chart(dfc,fast_ema,slow_ema,trades_df=tdf,
                                               title=f"{ticker_label} · {strategy} · {interval}/{period}",
                                               light=light), use_container_width=True)

                st.markdown("#### 📊 Trade Log")
                disp = tdf.copy()
                disp["Entry Time"] = disp["Entry Time"].astype(str)
                disp["Exit Time"]  = disp["Exit Time"].astype(str)

                wbg="#e6f5f2" if light else "rgba(0,229,180,.09)"
                lbg="#fde8ec" if light else "rgba(255,77,109,.09)"
                wfg="#004d3a" if light else "#00e5b4"
                lfg="#7a0000" if light else "#ff8fa3"
                nfg="#1a2236" if light else "#c8d0e0"

                def _srow(row):
                    p = row["PnL (Rs)"]
                    if p>0:   return [f"background-color:{wbg};color:{wfg}" for _ in row]
                    elif p<0: return [f"background-color:{lbg};color:{lfg}" for _ in row]
                    return [f"color:{nfg}" for _ in row]

                styled = (disp.style.apply(_srow,axis=1)
                          .format({"PnL (Rs)":"Rs{:+.2f}","Points":"{:+.2f}",
                                   "Entry Price":"{:.2f}","Exit Price":"{:.2f}"}))
                st.dataframe(styled,use_container_width=True,height=420,
                             column_config={"Trade Type":st.column_config.TextColumn(width=70)})

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — LIVE TRADING
    # ════════════════════════════════════════════════════════════════════════
    with tab_live:
        live_state: LiveState = st.session_state["live_state"]

        # Take a single consistent snapshot for this entire render
        s = live_state.snapshot()
        running = s["running"]

        # Auto-refresh only when running
        if running and HAS_AR:
            st_autorefresh(interval=3000, key="live_ar")

        # LTP widget — uses the pre-fetched info from this render
        ltp_widget(ticker_label, ltp_info, light)

        # Warn if running ticker ≠ sidebar ticker
        run_cfg = s.get("cfg") or {}
        if running and run_cfg.get("ticker","") != ticker_sym:
            st.warning(f"⚠ Engine running on **{run_cfg.get('ticker_label','?')}** — "
                       f"stop and restart to trade **{ticker_label}**.")

        st.markdown("### ⚡ Live Trading")

        if running:
            st.markdown("""
            <div style="display:inline-flex;align-items:center;gap:8px;
                        background:rgba(0,229,180,.1);border:1px solid rgba(0,229,180,.3);
                        border-radius:20px;padding:6px 16px;margin-bottom:8px">
                <span style="width:8px;height:8px;border-radius:50%;background:#00e5b4;
                             display:inline-block;animation:pulse 1s infinite"></span>
                <span style="font-family:Syne;font-size:12px;font-weight:700;
                             color:#00e5b4;letter-spacing:1px">LIVE RUNNING</span>
            </div>
            <style>@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(1.4)}}</style>
            """, unsafe_allow_html=True)

        b1,b2,b3,_ = st.columns([1.2,1.2,1.5,3])

        with b1:
            if st.button("▶ Start", type="primary", use_container_width=True,
                         disabled=running, key="live_start"):
                # Stop any old thread first (prevents "can't start new thread")
                stop_engine()
                # Fresh state + event
                st.session_state["live_state"]  = LiveState()
                st.session_state["stop_event"]  = threading.Event()
                live_state = st.session_state["live_state"]
                live_state.set("running", True)
                live_state.set("cfg", cfg.copy())
                t = threading.Thread(
                    target=live_engine,
                    args=(cfg, live_state, st.session_state["stop_event"]),
                    daemon=True,
                )
                t.start()
                st.session_state["live_thread"] = t
                st.rerun()

        with b2:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not running, key="live_stop"):
                stop_engine()
                time.sleep(0.3)
                st.rerun()

        with b3:
            if st.button("⚡ Square Off", use_container_width=True, key="live_sq"):
                ltp_now = s.get("ltp") or 0.0
                if ltp_now == 0.0 and s.get("position"):
                    ltp_now = float(s["position"]["entry_price"])
                rec = live_state.square_off(ltp_now, qty)
                if rec:
                    if cfg.get("dhan_en"): dhan_exit(cfg, rec["Trade Type"].lower(), ltp_now)
                    st.success(f"Squared off @ {ltp_now:.2f} | PnL Rs{rec['PnL (Rs)']:+.2f}")
                else:
                    st.info("No open position.")
                st.rerun()

        st.divider()

        # Config display
        dc = s.get("cfg") or cfg
        with st.expander("⚙ Active Configuration", expanded=True):
            cc=st.columns(5)
            cc[0].metric("Ticker",   dc.get("ticker_label","—"))
            cc[1].metric("Interval", dc.get("interval","—"))
            cc[2].metric("Period",   dc.get("period","—"))
            cc[3].metric("Strategy", dc.get("strategy","—"))
            cc[4].metric("Qty",      dc.get("qty","—"))
            cc2=st.columns(4)
            cc2[0].metric(f"EMA {dc.get('fast_ema','?')} (fast)",
                          f"{s['ema_fast']:.2f}" if s["ema_fast"] is not None else "—")
            cc2[1].metric(f"EMA {dc.get('slow_ema','?')} (slow)",
                          f"{s['ema_slow']:.2f}" if s["ema_slow"] is not None else "—")
            cc2[2].metric("SL",  f"{dc.get('sl_type','—')} / {dc.get('sl_pts','—')} pts")
            cc2[3].metric("TGT", f"{dc.get('tgt_type','—')} / {dc.get('tgt_pts','—')} pts")

        st.divider()
        main_col, log_col = st.columns([1.7,1])

        with main_col:
            ef_v  = s["ema_fast"]; es_v = s["ema_slow"]
            atr_v = s["atr"];      ltp_c = s["ltp"]
            pos   = s["position"]; pen  = s["pending_entry"]
            fast_l= dc.get("fast_ema", fast_ema)
            slow_l= dc.get("slow_ema", slow_ema)

            if ef_v is not None and es_v is not None:
                diff  = ef_v - es_v; trend = "Bullish ↑" if diff>0 else "Bearish ↓"
                mv    = st.columns(5)
                mv[0].metric(f"EMA {fast_l}", f"{ef_v:.2f}")
                mv[1].metric(f"EMA {slow_l}", f"{es_v:.2f}")
                mv[2].metric("EMA Diff", f"{diff:+.2f}",
                             delta=trend, delta_color="normal" if diff>0 else "inverse")
                mv[3].metric("ATR", f"{atr_v:.2f}" if atr_v else "—")
                mv[4].metric("LTP", f"{ltp_c:.2f}" if ltp_c else "—")

                if pos is not None and ltp_c is not None:
                    tt_p = pos["trade_type"]; ep_p = float(pos["entry_price"])
                    pnl_v = ((ltp_c-ep_p) if tt_p=="buy" else (ep_p-ltp_c))*qty
                    pts_v = (ltp_c-ep_p)   if tt_p=="buy" else (ep_p-ltp_c)
                    pclr  = (ACC["light"] if light else ACC["dark"]) if pnl_v>=0 else (DN["light"] if light else DN["dark"])
                    st.markdown(f"""
                    <div style="display:flex;gap:32px;font-family:JetBrains Mono,monospace;margin:8px 0 4px">
                        <div><span style="font-size:10px;color:#7a8899">OPEN PnL</span><br>
                             <span style="font-size:22px;font-weight:700;color:{pclr}">Rs{pnl_v:+.2f}</span></div>
                        <div><span style="font-size:10px;color:#7a8899">POINTS</span><br>
                             <span style="font-size:22px;font-weight:700;color:{pclr}">{pts_v:+.2f}</span></div>
                    </div>""", unsafe_allow_html=True)
            else:
                if not running:
                    st.info("Start the engine to see live EMA values and LTP.")

            if pen is not None:
                pa=ACC["light"] if light else ACC["dark"]
                st.markdown(f"""
                <div style="background:{'rgba(0,120,100,.07)' if light else 'rgba(0,229,180,.06)'};
                            border:1px solid {pa};border-radius:8px;padding:8px 14px;margin:8px 0;
                            font-family:Syne,sans-serif;font-size:12px">
                    ⏳ <b style="color:{pa}">Pending {pen.get('tt','').upper()}</b>
                    — entry at next candle open<br>
                    <small style="color:#8899aa">{pen.get('reason','')}</small>
                </div>""", unsafe_allow_html=True)

            if pos is not None:
                tt   = pos["trade_type"]; ep = float(pos["entry_price"])
                sl_  = pos.get("sl");    tgt_ = pos.get("target")
                tc   = (UP["light"] if light else UP["dark"]) if tt=="buy" else (DN["light"] if light else DN["dark"])
                tx   = "#1a2236" if light else "#c8d0e0"; mt = "#7a8899" if light else "#4a5568"
                pnl_live = ((ltp_c-ep) if (ltp_c and tt=="buy") else ((ep-ltp_c) if ltp_c else 0))*qty
                pc_cls = "card-green" if pnl_live>=0 else "card-red"
                st.markdown(f"""
                <div class="card {pc_cls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <span style="font-family:Syne,sans-serif;font-size:13px;font-weight:800;
                                 letter-spacing:2px;color:{tc}">● OPEN {tt.upper()}</span>
                  </div>
                  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;
                              font-family:JetBrains Mono,monospace;font-size:13px;color:{tx}">
                    <div><div style="color:{mt};font-size:10px">ENTRY</div>{ep:.2f}</div>
                    <div><div style="color:{mt};font-size:10px">LTP</div>{ltp_c:.2f if ltp_c else "—"}</div>
                    <div><div style="color:#e0284a;font-size:10px">SL</div>{f"{float(sl_):.2f}" if sl_ is not None else "—"}</div>
                    <div><div style="color:{UP["light"] if light else UP["dark"]};font-size:10px">TARGET</div>{f"{float(tgt_):.2f}" if tgt_ is not None else "N/A"}</div>
                    <div><div style="color:{mt};font-size:10px">QTY</div>{qty}</div>
                  </div>
                  <div style="margin-top:8px;font-family:Syne,sans-serif;font-size:11px;color:{mt}">
                    {pos["entry_time"]}  ·  {pos.get("entry_reason","")}
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                bg2="#fff" if light else "#0f1623"; bd2="#d0d7e6" if light else "#1a2236"
                st.markdown(f"""
                <div style="background:{bg2};border:1px solid {bd2};border-radius:12px;
                            text-align:center;padding:18px;color:#7a8899;font-family:Syne,sans-serif">
                    No open position
                </div>""", unsafe_allow_html=True)

            df_live = s.get("chart_df")
            if df_live is not None and not df_live.empty:
                ef_col = f"EMA_{fast_l}"; es_col = f"EMA_{slow_l}"
                # Add EMA columns if missing (e.g. after engine started)
                if ef_col not in df_live.columns:
                    df_live[ef_col] = tv_ema(df_live["Close"], fast_l)
                if es_col not in df_live.columns:
                    df_live[es_col] = tv_ema(df_live["Close"], slow_l)
                st.plotly_chart(
                    build_chart(df_live, fast_l, slow_l, position=pos,
                                title=f"Live · {dc.get('ticker_label','?')} · {dc.get('interval','?')}",
                                light=light),
                    use_container_width=True)
                lr = df_live.iloc[-1]
                st.markdown("**Last Fetched Candle**")
                st.dataframe(pd.DataFrame([{
                    "Time":str(df_live.index[-1]),
                    "Open":round(float(lr["Open"]),2), "High":round(float(lr["High"]),2),
                    "Low":round(float(lr["Low"]),2),   "Close":round(float(lr["Close"]),2),
                    "Vol":int(float(lr.get("Volume",0))),
                    f"EMA{fast_l}":round(float(df_live[ef_col].iloc[-1]),2) if ef_col in df_live.columns else "—",
                    f"EMA{slow_l}":round(float(df_live[es_col].iloc[-1]),2) if es_col in df_live.columns else "—",
                    "ATR":round(float(df_live["ATR"].iloc[-1]),2) if "ATR" in df_live.columns else "—",
                }]), use_container_width=True, hide_index=True)

        with log_col:
            st.markdown("**Activity Log**")
            logs = s.get("log", [])
            sep  = "#eef1f7" if light else "#0f1623"
            tx2  = "#1a2236" if light else "#c8d0e0"; mt2 = "#7a8899" if light else "#4a5568"
            log_html = (
                "".join(f"<div style='padding:2px 0;border-bottom:1px solid {sep};color:{tx2}'>{l}</div>"
                        for l in reversed(logs))
                if logs else f"<div style='color:{mt2}'>No activity — start the engine.</div>"
            )
            st.markdown(f'<div class="logbox">{log_html}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — TRADE HISTORY
    # ════════════════════════════════════════════════════════════════════════
    with tab_hist:
        ltp_widget(ticker_label, ltp_info, light)
        st.markdown("### 📋 Trade History")
        st.caption("Live completed trades. Auto-updates while engine is running.")

        s3 = st.session_state["live_state"].snapshot()
        hist = s3.get("trade_history", [])

        _,hcol = st.columns([5,1])
        with hcol:
            if st.button("🗑 Clear",use_container_width=True,key="clr_hist"):
                st.session_state["live_state"].clear_trades(); st.rerun()

        bg3="#fff" if light else "#0f1623"; brd3="#d0d7e6" if light else "#1a2236"
        if not hist:
            st.markdown(f"""
            <div style="background:{bg3};border:1px solid {brd3};border-radius:12px;
                        text-align:center;padding:40px;color:#7a8899;font-family:Syne,sans-serif">
                No live trades yet.
            </div>""", unsafe_allow_html=True)
        else:
            hdf = pd.DataFrame(hist)
            tot = float(hdf["PnL (Rs)"].sum())
            wp  = hdf["PnL (Rs)"]>0; lp = hdf["PnL (Rs)"]<0
            nw  = int(wp.sum()); n_h = len(hdf)
            acc = nw/n_h*100 if n_h else 0
            avgw= float(hdf.loc[wp,"PnL (Rs)"].mean()) if nw else 0.0
            avgl= float(hdf.loc[lp,"PnL (Rs)"].mean()) if lp.sum() else 0.0
            twp = float(hdf.loc[wp,"Points"].sum()) if "Points" in hdf.columns else 0.0
            twl = float(hdf.loc[lp,"Points"].sum()) if "Points" in hdf.columns else 0.0

            hm=st.columns(7)
            for c_,l_,v_ in zip(hm,
                ["Trades","Win Rate","Total PnL","Avg Win","Avg Loss","Win Pts","Loss Pts"],
                [n_h,f"{acc:.1f}%",f"Rs{tot:+,.0f}",f"Rs{avgw:+.0f}",f"Rs{avgl:+.0f}",
                 f"{twp:+.1f}",f"{twl:+.1f}"]):
                c_.metric(l_,v_)
            st.divider()

            wbg="#e6f5f2" if light else "rgba(0,229,180,.09)"
            lbg="#fde8ec" if light else "rgba(255,77,109,.09)"
            wfg="#004d3a" if light else "#00e5b4"
            lfg="#7a0000" if light else "#ff8fa3"
            nfg="#1a2236" if light else "#c8d0e0"
            def _hs(row):
                p=row["PnL (Rs)"]
                if p>0: return [f"background-color:{wbg};color:{wfg}" for _ in row]
                elif p<0: return [f"background-color:{lbg};color:{lfg}" for _ in row]
                return [f"color:{nfg}" for _ in row]
            st.dataframe(hdf.style.apply(_hs,axis=1)
                         .format({"PnL (Rs)":"Rs{:+.2f}","Points":"{:+.2f}",
                                  "Entry Price":"{:.2f}","Exit Price":"{:.2f}"}),
                         use_container_width=True,height=420)

            st.markdown("#### Cumulative PnL")
            hdf["Cum"]=hdf["PnL (Rs)"].cumsum()
            up_c=UP["light"] if light else UP["dark"]; dn_c=DN["light"] if light else DN["dark"]
            base=_base_layout(light); xax=base.pop("xaxis",{}); yax=base.pop("yaxis",{})
            fp=go.Figure()
            fp.add_trace(go.Scatter(x=list(range(1,n_h+1)),y=list(hdf["Cum"].astype(float)),
                                    mode="lines+markers",line=dict(color=up_c,width=2),
                                    marker=dict(size=5,color=[up_c if p>=0 else dn_c for p in hdf["PnL (Rs)"]]),
                                    fill="tozeroy",fillcolor="rgba(0,180,140,0.08)" if light else "rgba(0,229,180,0.06)"))
            fp.add_hline(y=0,line=dict(color="#94a3b8",width=1,dash="dash"))
            fp.update_layout(height=240,showlegend=False,xaxis_title="Trade #",yaxis_title="PnL (Rs)",**base)
            fp.update_xaxes(**xax); fp.update_yaxes(**yax)
            st.plotly_chart(fp,use_container_width=True)


if __name__ == "__main__":
    main()
