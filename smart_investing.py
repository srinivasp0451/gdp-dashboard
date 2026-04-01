# ══════════════════════════════════════════════════════════════════════════════
#  SMART INVESTING — Algorithmic Trading Platform  v5.0
#
#  Install:
#    pip install streamlit yfinance pandas numpy plotly pytz
#    pip install dhanhq                  # optional – Dhan broker live orders
#    pip install streamlit-autorefresh   # REQUIRED for live auto-refresh
#
#  Run:
#    streamlit run smart_investing.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings("ignore")

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_OK = True
except ImportError:
    AUTO_REFRESH_OK = False

try:
    from dhanhq import dhanhq
    DHAN_OK = True
except ImportError:
    DHAN_OK = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Smart Investing", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
IST = pytz.timezone("Asia/Kolkata")

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE  (each Streamlit user gets their own — fully isolated)
# ════════════════════════════════════════════════════════════════════════════
def _new_live() -> dict:
    """Fresh live-engine shared dict.  Thread holds a reference; UI reads from it."""
    return dict(running=False, position=None, pending_entry=None,
                trade_history=[], log=[], chart_df=None,
                ema_fast_val=None, ema_slow_val=None, atr_val=None,
                ltp=None, last_exit_ts=None, cfg=None)

_SS = dict(light_theme=True, bt_results=None, bt_chart_df=None, bt_run_key=None,
           live_shared=None, live_stop_evt=None)
for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state["live_shared"]  is None: st.session_state["live_shared"]  = _new_live()
if st.session_state["live_stop_evt"] is None: st.session_state["live_stop_evt"] = threading.Event()

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
def inject_css(light: bool) -> None:
    # Suppress Streamlit's grey "fading" overlay during autorefresh reruns
    fade_fix = """
    [data-testid="stStatusWidget"]{visibility:hidden!important;}
    .stSpinner>div{opacity:0!important;}
    """
    if light:
        st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html,body,[data-testid="stApp"]{{background:#f5f7fa!important;color:#1a2236;font-family:'Syne',sans-serif;}}
[data-testid="stSidebar"]{{background:#eef1f7!important;border-right:1px solid #d0d7e6;}}
.stTabs [data-baseweb="tab-list"]{{background:transparent;border-bottom:1px solid #d0d7e6;}}
.stTabs [data-baseweb="tab"]{{font-family:'Syne',sans-serif;font-weight:700;font-size:13px;color:#7a8899;padding:10px 22px;border-radius:6px 6px 0 0;}}
.stTabs [aria-selected="true"]{{color:#007a6a!important;background:rgba(0,122,106,.07)!important;}}
[data-testid="metric-container"]{{background:#fff;border:1px solid #d0d7e6;border-radius:10px;padding:12px 16px;}}
[data-testid="stMetricLabel"]{{font-size:10px;color:#7a8899!important;letter-spacing:.6px;text-transform:uppercase;}}
[data-testid="stMetricValue"]{{font-family:'JetBrains Mono',monospace;font-size:16px;color:#1a2236;}}
[data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace;font-size:12px;}}
.stButton button{{border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;border:1px solid #d0d7e6;transition:all .15s;}}
.stButton button[kind="primary"]{{background:linear-gradient(135deg,#00b896,#007a6a);color:#fff;border:none;}}
.stSelectbox>div,.stNumberInput>div input,.stTextInput>div input{{background:#fff!important;border:1px solid #d0d7e6!important;border-radius:8px!important;color:#1a2236!important;font-family:'JetBrains Mono',monospace!important;}}
.shdr{{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#7a8899;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #d0d7e6;}}
.card{{background:#fff;border:1px solid #d0d7e6;border-radius:12px;padding:14px 18px;margin-bottom:10px;}}
.card-red{{background:rgba(220,50,80,.05);border-color:rgba(220,50,80,.35);}}
.card-green{{background:rgba(0,160,120,.05);border-color:rgba(0,160,120,.35);}}
.logbox{{background:#f5f7fa;border:1px solid #d0d7e6;border-radius:8px;padding:12px;height:280px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.7;color:#1a2236;}}
hr{{border-color:#d0d7e6!important;margin:10px 0!important;}}
::-webkit-scrollbar{{width:4px;height:4px}}::-webkit-scrollbar-track{{background:#f5f7fa}}::-webkit-scrollbar-thumb{{background:#d0d7e6;border-radius:2px}}
[data-testid="stExpander"]{{background:#fff;border:1px solid #d0d7e6;border-radius:10px;}}
{fade_fix}</style>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html,body,[data-testid="stApp"]{{background:#080b12!important;color:#c8d0e0;font-family:'Syne',sans-serif;}}
[data-testid="stSidebar"]{{background:#0c1018!important;border-right:1px solid #1a2236;}}
.stTabs [data-baseweb="tab-list"]{{background:transparent;border-bottom:1px solid #1a2236;gap:4px;}}
.stTabs [data-baseweb="tab"]{{font-family:'Syne',sans-serif;font-weight:700;font-size:13px;color:#4a5568;padding:10px 22px;border-radius:6px 6px 0 0;border:1px solid transparent;transition:all .15s;}}
.stTabs [aria-selected="true"]{{color:#00e5b4!important;background:rgba(0,229,180,.06)!important;border-color:#1a2236 #1a2236 transparent!important;}}
[data-testid="metric-container"]{{background:#0f1623;border:1px solid #1a2236;border-radius:10px;padding:12px 16px;}}
[data-testid="stMetricLabel"]{{font-size:10px;color:#4a5568!important;letter-spacing:.6px;text-transform:uppercase;}}
[data-testid="stMetricValue"]{{font-family:'JetBrains Mono',monospace;font-size:16px;color:#c8d0e0;}}
[data-testid="stMetricDelta"]{{font-family:'JetBrains Mono',monospace;font-size:12px;}}
.stButton button{{border-radius:8px;font-family:'Syne',sans-serif;font-weight:700;font-size:13px;border:1px solid #1a2236;transition:all .15s;color:#c8d0e0;}}
.stButton button[kind="primary"]{{background:linear-gradient(135deg,#00e5b4,#00a88a);color:#000;border:none;}}
.stButton button[kind="primary"]:hover{{opacity:.9;transform:translateY(-1px);}}
.stButton button:not([kind="primary"]):hover{{border-color:#00e5b4;color:#00e5b4;}}
.stSelectbox>div,.stNumberInput>div input,.stTextInput>div input{{background:#0f1623!important;border:1px solid #1a2236!important;border-radius:8px!important;color:#c8d0e0!important;font-family:'JetBrains Mono',monospace!important;font-size:13px!important;}}
.shdr{{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4a5568;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #1a2236;}}
.card{{background:#0f1623;border:1px solid #1a2236;border-radius:12px;padding:14px 18px;margin-bottom:10px;}}
.card-red{{background:rgba(255,77,109,.07);border-color:rgba(255,77,109,.3);}}
.card-green{{background:rgba(0,229,180,.06);border-color:rgba(0,229,180,.25);}}
.logbox{{background:#080b12;border:1px solid #1a2236;border-radius:8px;padding:12px;height:280px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.7;}}
hr{{border-color:#1a2236!important;margin:10px 0!important;}}
::-webkit-scrollbar{{width:4px;height:4px}}::-webkit-scrollbar-track{{background:#080b12}}::-webkit-scrollbar-thumb{{background:#1a2236;border-radius:2px}}
[data-testid="stExpander"]{{background:#0f1623;border:1px solid #1a2236;border-radius:10px;}}
{fade_fix}</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
TICKER_MAP = {
    "Nifty 50":       "^NSEI",
    "Bank Nifty":     "^NSEBANK",
    "Sensex":         "^BSESN",
    "Bitcoin (BTC)":  "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Gold":           "GC=F",
    "Silver":         "SI=F",
}
TIMEFRAME_PERIODS = {
    "1m":  ["1d","5d","7d"],
    "5m":  ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo"],
    "1h":  ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":  ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk": ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
WARMUP_MAP = {
    "1d":"5d","5d":"1mo","7d":"1mo","1mo":"3mo","3mo":"6mo",
    "6mo":"1y","1y":"2y","2y":"5y","5y":"max","10y":"max","20y":"max",
}
PERIOD_DAYS = {
    "1d":1,"5d":5,"7d":7,"1mo":30,"3mo":90,"6mo":180,
    "1y":365,"2y":730,"5y":1825,"10y":3650,"20y":7300,
}
MAX_DAYS   = {"1m":7,"5m":60,"15m":60,"1h":730,"1d":36500,"1wk":36500}
STRATEGIES = ["EMA Crossover","Simple Buy","Simple Sell"]
SL_TYPES   = ["Custom Points","Trailing SL","Reverse EMA Crossover","Risk Reward Based","ATR Based"]
TGT_TYPES  = ["Custom Points","Trailing Target","EMA Crossover","Risk Reward Based","ATR Based"]

# ─── commented strategies / Dhan candle fetch (plug-in references) ───────────
# price_crosses_threshold strategy: see previous version comments
# fetch_candles_dhan: see previous version comments
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def _clean(raw) -> pd.DataFrame | None:
    """Flatten MultiIndex, ensure 1-D float Series per column, localise to IST."""
    if raw is None or (hasattr(raw,"empty") and raw.empty):
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    need = ["Open","High","Low","Close"]
    if not all(c in df.columns for c in need):
        return None
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df   = df[keep].copy()
    # Force each column to 1-D float Series (yfinance can return 2-D DataFrames)
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


def _best_warmup(period: str, interval: str) -> str:
    """Return the largest valid warmup period for this interval (fixes 5m/15m+1mo)."""
    max_d = MAX_DAYS.get(interval, 36500)
    pref  = WARMUP_MAP.get(period, period)
    if PERIOD_DAYS.get(pref, 9999) <= max_d:
        return pref
    ok = {p:d for p,d in PERIOD_DAYS.items() if d <= max_d}
    return max(ok, key=lambda p: ok[p]) if ok else period


@st.cache_data(ttl=90, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str):
    """Returns (df_full, df_display) or None."""
    try:
        warmup  = _best_warmup(period, interval)
        df_full = _clean(yf.download(ticker, period=warmup, interval=interval,
                                     auto_adjust=True, progress=False, prepost=False))
        if df_full is None or df_full.empty:
            return None
        d = PERIOD_DAYS.get(period)
        if d:
            cut        = datetime.now(IST) - timedelta(days=d)
            df_display = df_full[df_full.index >= cut].copy()
            if df_display.empty: df_display = df_full.copy()
        else:
            df_display = df_full.copy()
        return df_full, df_display
    except Exception:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_ltp_cached(ticker: str) -> dict | None:
    try:
        df = _clean(yf.download(ticker, period="5d", interval="1d",
                                auto_adjust=True, progress=False))
        if df is None or len(df) < 2: return None
        prev = float(df["Close"].iloc[-2]); ltp = float(df["Close"].iloc[-1])
        chg  = ltp - prev; pct = chg / prev * 100 if prev else 0
        return {"price":ltp,"change":chg,"pct":pct,"prev":prev}
    except: return None

# ════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def tv_ema(s: pd.Series, n: int) -> pd.Series:
    """TradingView-identical EMA: alpha=2/(n+1), adjust=False, seed from bar 0."""
    v = s.squeeze()
    if not isinstance(v, pd.Series): v = pd.Series(v)
    return v.astype(float).ewm(span=n, adjust=False, min_periods=1).mean()


def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi = df["High"].astype(float); lo = df["Low"].astype(float)
    pc = df["Close"].astype(float).shift(1)
    tr = pd.concat([hi-lo,(hi-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=1).mean()


def ema_angle_deg(ema_s: pd.Series, idx: int) -> float:
    if idx < 1 or idx >= len(ema_s): return 0.0
    p = float(ema_s.iloc[idx-1]); c = float(ema_s.iloc[idx])
    if p == 0: return 0.0
    return abs(float(np.degrees(np.arctan((c-p)/p*100.0))))


def add_indicators(df: pd.DataFrame, fast: int, slow: int, atr_n: int=14) -> pd.DataFrame:
    df = df.copy()
    ef = tv_ema(df["Close"], fast); es = tv_ema(df["Close"], slow)
    df[f"EMA_{fast}"] = ef; df[f"EMA_{slow}"] = es
    df["ATR"] = compute_atr(df, atr_n)
    df["Signal"] = 0
    df.loc[(ef > es) & (ef.shift(1) <= es.shift(1)), "Signal"] =  1
    df.loc[(ef < es) & (ef.shift(1) >= es.shift(1)), "Signal"] = -1
    return df

# ════════════════════════════════════════════════════════════════════════════
# SL / TARGET
# ════════════════════════════════════════════════════════════════════════════

def calc_sl(ep,tt,sl_type,sl_pts,ef,es,rr,atr=0.,atr_m=1.5):
    s = 1 if tt=="buy" else -1
    if sl_type=="Custom Points":          return ep - s*sl_pts
    if sl_type=="Trailing SL":            return ep - s*sl_pts
    if sl_type=="Reverse EMA Crossover":
        b = es if tt=="buy" else ef; fb = ep-s*sl_pts
        return min(b,fb) if tt=="buy" else max(b,fb)
    if sl_type=="Risk Reward Based":      return ep - s*sl_pts
    if sl_type=="ATR Based":
        v = atr if atr>0 else sl_pts; return ep - s*v*atr_m
    return ep - s*sl_pts


def calc_tgt(ep,tt,tgt_type,tgt_pts,sl,ef,es,rr,atr=0.,atr_m=2.0):
    s = 1 if tt=="buy" else -1
    if tgt_type=="Custom Points":    return ep + s*tgt_pts
    if tgt_type=="Trailing Target":  return ep + s*tgt_pts
    if tgt_type=="EMA Crossover":    return None
    if tgt_type=="Risk Reward Based":
        risk = abs(ep-sl) if sl is not None else tgt_pts
        return ep + s*risk*rr
    if tgt_type=="ATR Based":
        v = atr if atr>0 else tgt_pts; return ep + s*v*atr_m
    return ep + s*tgt_pts


def _dur(t0, t1) -> str:
    """Format duration between two timestamps as HHh MMm SSs."""
    try:
        delta = t1 - t0
        tot   = int(abs(delta.total_seconds()))
        return f"{tot//3600:02d}h {(tot%3600)//60:02d}m {tot%60:02d}s"
    except: return "—"

# ════════════════════════════════════════════════════════════════════════════
# ──────────────────────────  BACKTEST ENGINE  ────────────────────────────────
#
#  EMA Crossover:
#    Signal confirmed at CLOSE of candle N → set pending_entry
#    Execute at OPEN of candle N+1 (first available price after signal)
#
#  Simple Buy / Simple Sell:
#    Enter at OPEN of every bar where no position is active.
#    SL/Target checked on same bar's HIGH/LOW.
#    Re-enter on the very next bar after exit.
#    NO cooldown in backtest — tests maximum raw SL/Target behaviour.
#
#  Exit rule (both strategies, both directions):
#    BUY  → SL vs LOW first; if not hit → Target vs HIGH
#    SELL → SL vs HIGH first; if not hit → Target vs LOW
#    When both breach same candle → SL wins (conservative / worst-case)
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(df_full, df_display, strategy, fast, slow,
                 sl_type, sl_pts, tgt_type, tgt_pts, qty, rr,
                 atr_n, atr_sl_m, atr_tgt_m,
                 adv_filter, min_angle, co_candle, cs_pts, ca_mult):

    df = add_indicators(df_full.copy(), fast, slow, atr_n)

    # ── Warmup boundary ───────────────────────────────────────────────────
    display_start = df_display.index[0] if not df_display.empty else df.index[0]
    start_i = max(slow * 3, int(df.index.searchsorted(display_start)))
    start_i = min(start_i, len(df) - 1)

    trades        = []
    active        = None
    pending_entry = None   # EMA crossover only: signal on bar N → entry at bar N+1 open

    for i in range(start_i, len(df)):
        row   = df.iloc[i]
        ts    = df.index[i]
        open_ = float(row["Open"])
        hi    = float(row["High"])
        lo    = float(row["Low"])
        cl    = float(row["Close"])
        sig   = int(row.get("Signal", 0))
        ef_v  = float(row[f"EMA_{fast}"])
        es_v  = float(row[f"EMA_{slow}"])
        atr_v = float(row["ATR"]) if pd.notna(row.get("ATR", np.nan)) else sl_pts

        # ── STEP 1: EMA Crossover — execute pending entry at THIS bar's OPEN ──
        if strategy == "EMA Crossover" and pending_entry is not None and active is None:
            ep  = open_; tt = pending_entry["trade_type"]
            sl_p  = calc_sl(ep,tt,sl_type,sl_pts,pending_entry["ef"],pending_entry["es"],rr,pending_entry["atr"],atr_sl_m)
            tgt_p = calc_tgt(ep,tt,tgt_type,tgt_pts,sl_p,pending_entry["ef"],pending_entry["es"],rr,pending_entry["atr"],atr_tgt_m)
            # Skip if open gapped past SL
            gap = (tt=="buy" and sl_p is not None and ep<=sl_p) or (tt=="sell" and sl_p is not None and ep>=sl_p)
            if not gap:
                active = dict(entry_time=ts, entry_price=ep, trade_type=tt,
                              sl=sl_p, target=tgt_p, entry_reason=pending_entry["reason"])
            pending_entry = None

        # ── STEP 2: Simple Buy/Sell — enter at THIS bar's OPEN ──────────────
        #   No cooldown in backtest. Re-enter every bar after exit.
        if strategy in ["Simple Buy","Simple Sell"] and active is None:
            tt    = "buy" if strategy == "Simple Buy" else "sell"
            ep    = open_
            sl_p  = calc_sl(ep,tt,sl_type,sl_pts,ef_v,es_v,rr,atr_v,atr_sl_m)
            tgt_p = calc_tgt(ep,tt,tgt_type,tgt_pts,sl_p,ef_v,es_v,rr,atr_v,atr_tgt_m)
            active = dict(entry_time=ts, entry_price=ep, trade_type=tt,
                          sl=sl_p, target=tgt_p, entry_reason=strategy)

        # ── STEP 3: EXIT — compare SL/Target vs this bar's H/L ─────────────
        if active is not None:
            ep  = active["entry_price"]; tt = active["trade_type"]
            sl  = active["sl"];          tgt = active["target"]

            if tt == "buy":
                sl_hit  = sl  is not None and lo <= sl
                tgt_hit = tgt is not None and hi >= tgt
            else:
                sl_hit  = sl  is not None and hi >= sl
                tgt_hit = tgt is not None and lo <= tgt

            ema_exit = (tgt_type == "EMA Crossover" and
                        ((tt=="buy" and sig==-1) or (tt=="sell" and sig==1)))

            exit_px = exit_rsn = None
            # SL first (conservative) for both BUY and SELL
            if sl_hit:
                exit_px = sl;   exit_rsn = "SL Hit" + (" ⚠ (both)" if tgt_hit else "")
            elif tgt_hit:
                exit_px = tgt;  exit_rsn = "Target Hit"
            elif ema_exit:
                exit_px = cl;   exit_rsn = "EMA Crossover Exit"

            # Trailing SL update when not exiting
            if sl_type == "Trailing SL" and exit_px is None:
                if tt == "buy":
                    nsl = cl - sl_pts
                    if nsl > active["sl"]: active["sl"] = nsl
                else:
                    nsl = cl + sl_pts
                    if nsl < active["sl"]: active["sl"] = nsl

            if exit_px is not None:
                pnl = ((float(exit_px)-ep) if tt=="buy" else (ep-float(exit_px))) * qty
                pts = (float(exit_px)-ep) if tt=="buy" else (ep-float(exit_px))
                trades.append(dict(
                    **{"Entry Time":active["entry_time"], "Exit Time":ts,
                       "Duration":_dur(active["entry_time"], ts),
                       "Trade Type":tt.upper(),
                       "Entry Price":round(ep,2), "Exit Price":round(float(exit_px),2),
                       "SL":     round(float(active["sl"]),2) if active["sl"]  is not None else "—",
                       "Target": round(float(tgt),2)          if tgt           is not None else "—",
                       "Candle High":round(hi,2), "Candle Low":round(lo,2),
                       "Entry Reason":active["entry_reason"], "Exit Reason":exit_rsn,
                       "Points":round(pts,2), "PnL (Rs)":round(pnl,2), "Qty":qty}
                ))
                active = None

        # ── STEP 4: EMA Crossover signal detection at CLOSE ──────────────────
        #   Only set pending_entry — executes at OPEN of bar i+1
        if strategy == "EMA Crossover" and sig != 0 and active is None and pending_entry is None:
            allow = True
            angle_info = ""
            if adv_filter:
                angle = ema_angle_deg(df[f"EMA_{fast}"], i)
                angle_info = f" ({angle:.1f}°)"
                if angle < min_angle:
                    allow = False
                elif co_candle != "Simple Crossover":
                    body = abs(cl - open_)
                    if co_candle == "Custom Candle Size" and body < cs_pts: allow = False
                    elif co_candle == "ATR Based Candle Size" and body < ca_mult*atr_v: allow = False
            if allow:
                tt  = "buy" if sig == 1 else "sell"
                rsn = f"EMA{fast}×EMA{slow} {'↑' if sig==1 else '↓'}{angle_info}"
                pending_entry = dict(trade_type=tt, reason=rsn, ef=ef_v, es=es_v, atr=atr_v)

    # Force-close open position at end of data
    if active is not None:
        lr = df.iloc[-1]; ts_end = df.index[-1]
        ep_x = float(lr["Close"])
        pnl  = ((ep_x-active["entry_price"]) if active["trade_type"]=="buy" else (active["entry_price"]-ep_x))*qty
        pts  = (ep_x-active["entry_price"]) if active["trade_type"]=="buy" else (active["entry_price"]-ep_x)
        trades.append(dict(
            **{"Entry Time":active["entry_time"], "Exit Time":ts_end,
               "Duration":_dur(active["entry_time"],ts_end),
               "Trade Type":active["trade_type"].upper(),
               "Entry Price":round(active["entry_price"],2), "Exit Price":round(ep_x,2),
               "SL":     round(float(active["sl"]),2)     if active["sl"]     is not None else "—",
               "Target": round(float(active["target"]),2) if active["target"] is not None else "—",
               "Candle High":round(float(lr["High"]),2), "Candle Low":round(float(lr["Low"]),2),
               "Entry Reason":active["entry_reason"], "Exit Reason":"End of Data",
               "Points":round(pts,2), "PnL (Rs)":round(pnl,2), "Qty":qty}
        ))
    return pd.DataFrame(trades)

# ════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ════════════════════════════════════════════════════════════════════════════

def _base(light: bool) -> dict:
    bg  = "#f5f7fa" if light else "#080b12"
    plt = "#ffffff"  if light else "#080b12"
    grd = "#e8edf5"  if light else "#0f1623"
    txt = "#334155"  if light else "#6b7a99"
    spk = "#94a3b8"  if light else "#2a3456"
    return dict(paper_bgcolor=bg, plot_bgcolor=plt,
                font=dict(family="JetBrains Mono", color=txt, size=11),
                xaxis_rangeslider_visible=False,
                margin=dict(t=40,b=16,l=48,r=24),
                legend=dict(orientation="h",yanchor="bottom",y=1.01,bgcolor="rgba(0,0,0,0)"),
                hoverlabel=dict(bgcolor="#fff" if light else "#0f1623",
                                bordercolor="#d0d7e6" if light else "#1a2236",
                                font=dict(family="JetBrains Mono",size=11)),
                xaxis=dict(gridcolor=grd,zeroline=False,showspikes=True,spikecolor=spk,spikethickness=1),
                yaxis=dict(gridcolor=grd,zeroline=False,showspikes=True,spikecolor=spk,spikethickness=1))


def build_chart(df, fast, slow, trades_df=None, position=None, title="", light=False) -> go.Figure:
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    up  = "#00b896" if light else "#00e5b4"
    dn  = "#e0284a" if light else "#ff4d6d"
    bg0 = "#000" if light else "#080b12"

    fig.add_trace(go.Candlestick(
        x=list(df.index),
        open=list(df["Open"].astype(float)), high=list(df["High"].astype(float)),
        low=list(df["Low"].astype(float)),   close=list(df["Close"].astype(float)),
        name="OHLC",
        increasing=dict(line=dict(color=up,width=1.2),fillcolor=up),
        decreasing=dict(line=dict(color=dn,width=1.2),fillcolor=dn),
        whiskerwidth=0.4,
    ), row=1, col=1)

    for col_, clr_, nm in [(f"EMA_{fast}","#f59e0b",f"EMA {fast}"),
                            (f"EMA_{slow}","#3b82f6",f"EMA {slow}")]:
        if col_ in df.columns:
            fig.add_trace(go.Scatter(x=list(df.index),y=list(df[col_].astype(float)),
                                     name=nm,line=dict(color=clr_,width=1.8)), row=1, col=1)

    if trades_df is not None and not trades_df.empty:
        for _,t in trades_df.iterrows():
            tt_ = str(t["Trade Type"]); c_ = up if tt_=="BUY" else dn
            esy = "triangle-up" if tt_=="BUY" else "triangle-down"
            exy = "triangle-down" if tt_=="BUY" else "triangle-up"
            ep_ = float(t["Entry Price"]); xp_ = float(t["Exit Price"]); p_ = float(t["PnL (Rs)"])
            fig.add_trace(go.Scatter(x=[t["Entry Time"]],y=[ep_],mode="markers",showlegend=False,
                marker=dict(symbol=esy,size=13,color=c_,line=dict(width=1.5,color="#fff")),
                hovertemplate=f"<b>{tt_} ENTRY</b><br>{ep_:.2f}<br>{t.get('Entry Reason','')}<extra></extra>"),
                row=1, col=1)
            fig.add_trace(go.Scatter(x=[t["Exit Time"]],y=[xp_],mode="markers",showlegend=False,
                marker=dict(symbol=exy,size=11,color=c_,opacity=0.6,line=dict(width=1,color=bg0)),
                hovertemplate=f"<b>{tt_} EXIT</b><br>{xp_:.2f}<br>{t.get('Exit Reason','')}<br>PnL Rs{p_:+.2f}<extra></extra>"),
                row=1, col=1)

    if position is not None:
        for lvl,clr_,lbl in [(position.get("entry_price"),"#334155" if light else "#fff","Entry"),
                              (position.get("sl"),dn,"SL"),(position.get("target"),up,"Target")]:
            if lvl is not None:
                fig.add_hline(y=float(lvl),row=1,col=1,
                              line=dict(color=clr_,width=1.4,dash="dash" if lbl!="Entry" else "solid"),
                              annotation_text=f" {lbl} {float(lvl):.2f}",
                              annotation_font=dict(color=clr_,size=11))

    if "Volume" in df.columns:
        vc = [up if float(c)>=float(o) else dn for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=list(df.index),y=list(df["Volume"].fillna(0).astype(float)),
                             marker_color=vc,name="Vol",showlegend=False), row=2, col=1)

    base = _base(light); xax = base.pop("xaxis",{}); yax = base.pop("yaxis",{})
    fig.update_layout(height=570,title=dict(text=title,x=0.01,
                      font=dict(family="Syne",size=13,color="#334155" if light else "#6b7a99")),**base)
    fig.update_xaxes(**xax); fig.update_yaxes(**yax)
    return fig

# ════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ════════════════════════════════════════════════════════════════════════════

def ltp_widget(ticker: str, label: str, light: bool) -> None:
    info = fetch_ltp_cached(ticker)
    if not info: st.warning(f"LTP unavailable for {label}."); return
    p,c,pct = info["price"],info["change"],info["pct"]
    up_ = "#007a6a" if light else "#00e5b4"; dn_ = "#c0152d" if light else "#ff4d6d"
    col = up_ if c>=0 else dn_; arr = "▲" if c>=0 else "▼"
    bg  = "#fff" if light else "#0f1623"; brd = "#d0d7e6" if light else "#1a2236"
    tx  = "#1a2236" if light else "#e2e8f0"; mt = "#7a8899" if light else "#4a5568"
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {brd};border-radius:12px;
                padding:14px 22px;margin-bottom:14px;display:flex;
                align-items:center;gap:28px;flex-wrap:wrap">
        <div>
            <div style="font-size:10px;font-weight:700;letter-spacing:2px;
                        text-transform:uppercase;color:{mt};font-family:Syne,sans-serif">{label}</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:28px;
                        font-weight:700;color:{tx};line-height:1.1">{p:,.2f}</div>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:17px;font-weight:600;color:{col}">{arr} {c:+.2f}</div>
        <div style="font-family:JetBrains Mono,monospace;font-size:15px;font-weight:600;color:{col}">{pct:+.2f}%</div>
        <div style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:11px;color:{mt}">Prev: {info['prev']:,.2f}</div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DHAN ORDERS
# ════════════════════════════════════════════════════════════════════════════

def _dhan(cfg): return dhanhq(str(cfg["dhan_client_id"]),str(cfg["dhan_access_token"]))

def place_entry(cfg,tt,ltp):
    if not DHAN_OK: return "dhanhq not installed"
    try:
        d = _dhan(cfg)
        if cfg.get("options_trading"):
            sec=cfg["ce_security_id"] if tt=="buy" else cfg["pe_security_id"]
            ot="MARKET" if cfg.get("opts_entry_otype")=="Market Order" else "LIMIT"
            r=d.place_order(transactionType="BUY",exchangeSegment=str(cfg.get("opts_exchange","NSE_FNO")),
                            productType="INTRADAY",orderType=ot,validity="DAY",securityId=str(sec),
                            quantity=int(cfg.get("opts_qty",65)),price=float(ltp) if ot=="LIMIT" else 0.0,triggerPrice=0)
        else:
            ex="NSE_EQ" if cfg.get("eq_exchange","NSE")=="NSE" else "BSE"
            pt="INTRADAY" if cfg.get("eq_product")=="Intraday" else "CNC"
            ot="MARKET" if cfg.get("eq_entry_otype")=="Market Order" else "LIMIT"
            r=d.place_order(security_id=str(cfg.get("eq_sec_id","1594")),exchange_segment=ex,
                            transaction_type="BUY" if tt=="buy" else "SELL",
                            quantity=int(cfg.get("eq_qty",1)),order_type=ot,product_type=pt,
                            price=float(ltp) if ot=="LIMIT" else 0.0)
        return f"Entry OK: {r}"
    except Exception as e: return f"Entry FAILED: {e}"

def place_exit(cfg,tt,ltp):
    if not DHAN_OK: return "dhanhq not installed"
    try:
        d = _dhan(cfg)
        if cfg.get("options_trading"):
            sec=cfg["ce_security_id"] if tt=="buy" else cfg["pe_security_id"]
            ot="MARKET" if cfg.get("opts_exit_otype","Market Order")=="Market Order" else "LIMIT"
            r=d.place_order(transactionType="SELL",exchangeSegment=str(cfg.get("opts_exchange","NSE_FNO")),
                            productType="INTRADAY",orderType=ot,validity="DAY",securityId=str(sec),
                            quantity=int(cfg.get("opts_qty",65)),price=float(ltp) if ot=="LIMIT" else 0.0,triggerPrice=0)
        else:
            ex="NSE_EQ" if cfg.get("eq_exchange","NSE")=="NSE" else "BSE"
            pt="INTRADAY" if cfg.get("eq_product")=="Intraday" else "CNC"
            ot="MARKET" if cfg.get("eq_exit_otype","Market Order")=="Market Order" else "LIMIT"
            r=d.place_order(security_id=str(cfg.get("eq_sec_id","1594")),exchange_segment=ex,
                            transaction_type="SELL" if tt=="buy" else "BUY",
                            quantity=int(cfg.get("eq_qty",1)),order_type=ot,product_type=pt,
                            price=float(ltp) if ot=="LIMIT" else 0.0)
        return f"Exit OK: {r}"
    except Exception as e: return f"Exit FAILED: {e}"

# ════════════════════════════════════════════════════════════════════════════
# LIVE ENGINE (background daemon thread)
# ════════════════════════════════════════════════════════════════════════════
#
#  Multi-user: each user's Start creates a NEW threading.Event and a NEW
#  shared dict. The thread ONLY reads/writes its own `shared` dict and checks
#  its own `stop_event`. Zero cross-contamination between users.
#
#  Stop: threading.Event.set() is atomically visible cross-thread immediately.
#  stop_event.wait(timeout=1.5) replaces time.sleep — responds within 1.5 s.
#
#  Entry matching backtest:
#    EMA Crossover : signal at close of candle N → pending_entry
#                    → executes at OPEN of candle N+1 (df["Open"].iloc[-1])
#    Simple Buy/Sell: enter at LTP immediately (no candle wait)
#                     cooldown applies between re-entries (live only)
# ════════════════════════════════════════════════════════════════════════════

def live_engine(cfg: dict, shared: dict, stop_event: threading.Event) -> None:
    ticker   = cfg["ticker"];    interval = cfg["interval"];  period = cfg["period"]
    strategy = cfg["strategy"];  fast     = cfg["fast_ema"]; slow   = cfg["slow_ema"]
    sl_type  = cfg["sl_type"];   sl_pts   = cfg["sl_pts"]
    tgt_type = cfg["tgt_type"];  tgt_pts  = cfg["tgt_pts"]
    qty      = cfg["qty"];       cd_en    = cfg["cd_en"];    cd_s   = cfg["cd_s"]
    rr       = cfg["rr"]
    atr_n    = cfg.get("atr_period",14)
    atr_sl_m = cfg.get("atr_sl_mult",1.5); atr_tgt_m = cfg.get("atr_tgt_mult",2.0)
    adv      = cfg.get("adv_filter",False)
    min_ang  = cfg.get("min_angle",0.0)
    co_can   = cfg.get("crossover_candle","Simple Crossover")
    cs_pts   = cfg.get("candle_size_pts",10.0); ca_m = cfg.get("candle_atr_mult",1.0)

    def _log(msg):
        ts = datetime.now(IST).strftime("%H:%M:%S")
        shared["log"].append(f"[{ts}] {msg}")
        shared["log"] = shared["log"][-300:]

    _log("Engine started")
    last_closed_ts = None

    while not stop_event.is_set():
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              auto_adjust=True, progress=False, prepost=False)
            df  = _clean(raw)

            if df is None or len(df) < max(slow*2+5, 10):
                _log("Insufficient data…"); stop_event.wait(timeout=1.5); continue

            df[f"EMA_{fast}"] = tv_ema(df["Close"], fast)
            df[f"EMA_{slow}"] = tv_ema(df["Close"], slow)
            df["ATR"]         = compute_atr(df, atr_n)

            ef_c  = float(df[f"EMA_{fast}"].iloc[-2])
            es_c  = float(df[f"EMA_{slow}"].iloc[-2])
            atr_c = float(df["ATR"].iloc[-2])

            # Only update shared state for the ticker this thread is running on
            shared["chart_df"]     = df.copy()
            shared["ema_fast_val"] = ef_c
            shared["ema_slow_val"] = es_c
            shared["atr_val"]      = atr_c
            ltp = float(df["Close"].iloc[-1])
            shared["ltp"] = ltp

            # ── A: EMA Crossover — execute pending at N+1 OPEN ──────────
            pen = shared.get("pending_entry")
            if pen is not None and shared.get("position") is None:
                ep = float(df["Open"].iloc[-1])   # N+1 candle open
                tt = pen["trade_type"]
                sl_p  = calc_sl(ep,tt,sl_type,sl_pts,pen["ef"],pen["es"],rr,pen["atr"],atr_sl_m)
                tgt_p = calc_tgt(ep,tt,tgt_type,tgt_pts,sl_p,pen["ef"],pen["es"],rr,pen["atr"],atr_tgt_m)
                gap   = (tt=="buy" and sl_p is not None and ep<=sl_p) or \
                        (tt=="sell" and sl_p is not None and ep>=sl_p)
                if not gap:
                    shared["position"] = dict(trade_type=tt, entry_price=ep,
                                              entry_time=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                                              sl=sl_p, target=tgt_p, entry_reason=pen["reason"])
                    sl_s  = f"{sl_p:.2f}" if sl_p is not None else "—"
                    tgt_s = f"{tgt_p:.2f}" if tgt_p is not None else "—"
                    _log(f"ENTRY {tt.upper()} @ N+1 Open {ep:.2f} | SL:{sl_s} | Tgt:{tgt_s} | {pen['reason']}")
                    if cfg.get("dhan_en"): _log(f"Broker: {place_entry(cfg,tt,ep)}")
                else:
                    _log(f"Entry skipped — gapped past SL")
                shared["pending_entry"] = None

            # ── B: Simple Buy/Sell — enter at LTP immediately (with cooldown) ──
            if strategy in ["Simple Buy","Simple Sell"] and shared.get("position") is None:
                go = True
                if cd_en:
                    lx = shared.get("last_exit_ts")
                    if lx and (datetime.now(IST)-lx).total_seconds() < cd_s:
                        go = False
                if go:
                    sig_tt = "buy" if strategy=="Simple Buy" else "sell"
                    sl_p   = calc_sl(ltp,sig_tt,sl_type,sl_pts,ef_c,es_c,rr,atr_c,atr_sl_m)
                    tgt_p  = calc_tgt(ltp,sig_tt,tgt_type,tgt_pts,sl_p,ef_c,es_c,rr,atr_c,atr_tgt_m)
                    shared["position"] = dict(trade_type=sig_tt, entry_price=ltp,
                                              entry_time=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                                              sl=sl_p, target=tgt_p, entry_reason=strategy)
                    sl_s  = f"{sl_p:.2f}" if sl_p is not None else "—"
                    tgt_s = f"{tgt_p:.2f}" if tgt_p is not None else "—"
                    _log(f"ENTRY {sig_tt.upper()} @ LTP {ltp:.2f} | SL:{sl_s} | Tgt:{tgt_s}")
                    if cfg.get("dhan_en"): _log(f"Broker: {place_entry(cfg,sig_tt,ltp)}")

            # ── C: SL/Target check vs LTP (every tick) ────────────────────
            pos = shared.get("position")
            if pos is not None:
                tt  = pos["trade_type"]; ep = float(pos["entry_price"])
                sl  = pos.get("sl");     tgt = pos.get("target")
                exited = False; exit_px = None; exit_reason = None

                # Conservative SL first
                if tt == "buy":
                    if sl  is not None and ltp <= sl:
                        exit_px = ltp; exit_reason = "SL Hit (LTP≤SL)"; exited = True
                    elif tgt is not None and ltp >= tgt:
                        exit_px = ltp; exit_reason = "Target Hit (LTP≥Target)"; exited = True
                else:
                    if sl  is not None and ltp >= sl:
                        exit_px = ltp; exit_reason = "SL Hit (LTP≥SL)"; exited = True
                    elif tgt is not None and ltp <= tgt:
                        exit_px = ltp; exit_reason = "Target Hit (LTP≤Target)"; exited = True

                # EMA crossover exit (completed candle)
                if not exited and tgt_type == "EMA Crossover" and len(df) >= 3:
                    pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])
                    if tt=="buy"  and ef_c < es_c and pf >= ps:
                        exit_px=ltp; exit_reason="EMA Crossover Exit"; exited=True
                    if tt=="sell" and ef_c > es_c and pf <= ps:
                        exit_px=ltp; exit_reason="EMA Crossover Exit"; exited=True

                # Trailing SL
                if sl_type == "Trailing SL" and not exited:
                    if tt == "buy":
                        nsl = ltp - sl_pts
                        if nsl > pos["sl"]: pos["sl"] = nsl; shared["position"] = pos; _log(f"Trailing SL→{nsl:.2f}")
                    else:
                        nsl = ltp + sl_pts
                        if nsl < pos["sl"]: pos["sl"] = nsl; shared["position"] = pos; _log(f"Trailing SL→{nsl:.2f}")

                if exited and exit_px is not None:
                    entry_ts_str = pos["entry_time"]
                    try: entry_ts_dt = datetime.strptime(entry_ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    except: entry_ts_dt = datetime.now(IST)
                    exit_ts_dt = datetime.now(IST)
                    dur = _dur(entry_ts_dt, exit_ts_dt)
                    pnl = ((float(exit_px)-ep) if tt=="buy" else (ep-float(exit_px)))*qty
                    pts = (float(exit_px)-ep) if tt=="buy" else (ep-float(exit_px))
                    rec = dict(
                        **{"Entry Time":entry_ts_str,
                           "Exit Time": exit_ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
                           "Duration":  dur,
                           "Trade Type":tt.upper(),
                           "Entry Price":round(ep,2), "Exit Price":round(float(exit_px),2),
                           "SL":   round(float(sl),2)  if sl  is not None else "—",
                           "Target":round(float(tgt),2) if tgt is not None else "—",
                           "Entry Reason":pos.get("entry_reason",""), "Exit Reason":exit_reason,
                           "Points":round(pts,2), "PnL (Rs)":round(pnl,2), "Qty":qty, "Mode":"Live"}
                    )
                    shared["trade_history"].append(rec)
                    shared["position"]     = None
                    shared["last_exit_ts"] = datetime.now(IST)
                    if cfg.get("dhan_en"): _log(f"Broker exit: {place_exit(cfg,tt,float(exit_px))}")
                    sign = "+" if pnl>=0 else ""
                    _log(f"EXIT {tt.upper()} @ {float(exit_px):.2f} | {exit_reason} | Pts:{pts:+.2f} | Rs{sign}{pnl:.2f}")

            # ── D: EMA Crossover signal detection (new completed candle) ──
            cur_ts = df.index[-2] if len(df) >= 2 else None
            if (strategy == "EMA Crossover" and cur_ts is not None and
                    cur_ts != last_closed_ts):
                last_closed_ts = cur_ts
                if shared.get("position") is None and shared.get("pending_entry") is None:
                    if cd_en:
                        lx = shared.get("last_exit_ts")
                        if lx and (datetime.now(IST)-lx).total_seconds() < cd_s:
                            stop_event.wait(timeout=1.5); continue
                    if len(df) >= 3:
                        pf = float(df[f"EMA_{fast}"].iloc[-3]); ps = float(df[f"EMA_{slow}"].iloc[-3])
                        bull = ef_c > es_c and pf <= ps; bear = ef_c < es_c and pf >= ps
                        if bull or bear:
                            allow = True; ai = ""
                            if adv:
                                ang = ema_angle_deg(df[f"EMA_{fast}"], len(df)-2); ai = f" ({ang:.1f}°)"
                                if ang < min_ang: allow = False
                                elif co_can != "Simple Crossover":
                                    body = abs(float(df["Close"].iloc[-2])-float(df["Open"].iloc[-2]))
                                    if co_can == "Custom Candle Size" and body < cs_pts: allow = False
                                    elif co_can == "ATR Based Candle Size" and body < ca_m*atr_c: allow = False
                            if allow:
                                sig_tt = "buy" if bull else "sell"
                                rsn    = f"EMA{fast}×EMA{slow} {'↑' if bull else '↓'}{ai}"
                                _log(f"Signal: {sig_tt.upper()} on {str(cur_ts)[:16]} — waiting for N+1 open")
                                shared["pending_entry"] = dict(trade_type=sig_tt, reason=rsn,
                                                               ef=ef_c, es=es_c, atr=atr_c)
            elif last_closed_ts != cur_ts: last_closed_ts = cur_ts

        except Exception as exc:
            _log(f"Error: {exc}")

        stop_event.wait(timeout=1.5)   # ← responds to stop instantly; don't use time.sleep

    _log("Engine stopped")
    shared["running"] = False

# ════════════════════════════════════════════════════════════════════════════
# ████████████████████████   MAIN APP   ██████████████████████████████████████
# ════════════════════════════════════════════════════════════════════════════

def main():
    light  = st.session_state.get("light_theme", True)
    inject_css(light)
    shared = st.session_state["live_shared"]

    # ════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════════════════
    with st.sidebar:
        lc = "#007a6a" if light else "#00e5b4"
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0 10px">
            <div style="font-size:38px">📈</div>
            <div style="font-family:Syne,sans-serif;font-size:18px;font-weight:800;
                        letter-spacing:3px;color:{lc}">SMART INVESTING</div>
            <div style="font-size:9px;color:#8899aa;letter-spacing:2px;margin-top:2px">
                ALGORITHMIC TRADING PLATFORM
            </div>
        </div>""", unsafe_allow_html=True)

        lt = st.checkbox("☀ Light Theme", value=light, key="light_theme")
        if lt != light: st.rerun()
        st.divider()

        # ASSET
        st.markdown('<div class="shdr">Asset</div>', unsafe_allow_html=True)
        choice = st.selectbox("Ticker", list(TICKER_MAP.keys())+["Custom →"],
                              label_visibility="collapsed")
        if choice == "Custom →":
            ticker_sym = st.text_input("Symbol (e.g. RELIANCE.NS)", "RELIANCE.NS")
            ticker_label = ticker_sym
        else:
            ticker_sym = TICKER_MAP[choice]; ticker_label = choice

        # TIMEFRAME
        st.markdown('<div class="shdr">Timeframe</div>', unsafe_allow_html=True)
        tf_keys = list(TIMEFRAME_PERIODS.keys())
        c1, c2  = st.columns(2)
        with c1: interval = st.selectbox("Interval", tf_keys, index=tf_keys.index("5m"),
                                          label_visibility="collapsed")
        with c2:
            prd_opts  = TIMEFRAME_PERIODS[interval]
            dflt_prd  = prd_opts.index("1mo") if "1mo" in prd_opts else min(1,len(prd_opts)-1)
            period    = st.selectbox("Period", prd_opts, index=dflt_prd,
                                     label_visibility="collapsed")
        st.divider()

        # STRATEGY
        st.markdown('<div class="shdr">Strategy</div>', unsafe_allow_html=True)
        strategy = st.selectbox("Strategy", STRATEGIES, label_visibility="collapsed")
        fast_ema = 9; slow_ema = 15
        adv_filter = False; min_angle = 0.0; crossover_candle = "Simple Crossover"
        candle_size_pts = 10.0; candle_atr_mult = 1.0

        if strategy == "EMA Crossover":
            e1,e2 = st.columns(2)
            with e1: fast_ema = st.number_input("Fast EMA", 2, 200, 9)
            with e2: slow_ema = st.number_input("Slow EMA", 2, 500, 15)
            adv_filter = st.checkbox("Advanced Entry Filters", value=False,
                                     help="Off = plain crossover. Enable to add angle / candle size filters.")
            if adv_filter:
                min_angle = st.number_input("Min Angle (°)", 0.0, 90.0, 0.0, 0.5)
                crossover_candle = st.selectbox("Candle Filter",
                    ["Simple Crossover","Custom Candle Size","ATR Based Candle Size"])
                if crossover_candle == "Custom Candle Size":
                    candle_size_pts = st.number_input("Min Body (pts)", 0.1, 1e6, 10.0, 0.5)
                elif crossover_candle == "ATR Based Candle Size":
                    candle_atr_mult = st.number_input("Min Body (ATR×)", 0.1, 10.0, 1.0, 0.1)
        st.divider()

        # STOP LOSS
        st.markdown('<div class="shdr">Stop Loss</div>', unsafe_allow_html=True)
        sl_type = st.selectbox("SL Type", SL_TYPES, label_visibility="collapsed")
        s1,s2 = st.columns(2)
        with s1: sl_pts = st.number_input("SL Points", 0.1, 1e6, 10.0, 0.5)
        with s2: rr = st.number_input("R:R", 0.5, 20.0, 2.0, 0.5)

        # TARGET
        st.markdown('<div class="shdr">Target</div>', unsafe_allow_html=True)
        tgt_type = st.selectbox("Target Type", TGT_TYPES, label_visibility="collapsed")
        tgt_pts  = st.number_input("Target Points", 0.1, 1e6, 20.0, 0.5)

        atr_period = 14; atr_sl_mult = 1.5; atr_tgt_mult = 2.0
        if sl_type == "ATR Based" or tgt_type == "ATR Based":
            st.markdown("**ATR Config**")
            atr_period = st.number_input("ATR Period", 2, 200, 14)
            if sl_type  == "ATR Based": atr_sl_mult  = st.number_input("ATR × SL",  0.1, 10.0, 1.5, 0.1)
            if tgt_type == "ATR Based": atr_tgt_mult = st.number_input("ATR × TGT", 0.1, 10.0, 2.0, 0.1)
        st.divider()

        # TRADE SETTINGS
        st.markdown('<div class="shdr">Trade Settings</div>', unsafe_allow_html=True)
        qty = st.number_input("Quantity", 1, 10_000_000, 1)
        d1,d2 = st.columns([1.4,1])
        with d1: cd_en = st.checkbox("Cooldown (Live only)", value=True)
        with d2: cd_s  = st.number_input("Secs", 1, 86400, 5, disabled=not cd_en,
                                          label_visibility="visible")
        no_overlap = st.checkbox("No Overlapping Trades", value=True)
        st.divider()

        # DHAN BROKER
        st.markdown('<div class="shdr">Dhan Broker</div>', unsafe_allow_html=True)
        dhan_en = st.checkbox("Enable Dhan Broker", value=False)
        dhan_cfg: dict = {"dhan_en": dhan_en}
        if dhan_en:
            if not DHAN_OK: st.warning("pip install dhanhq")
            dhan_cfg["dhan_client_id"]    = st.text_input("Client ID",    "", type="password")
            dhan_cfg["dhan_access_token"] = st.text_input("Access Token", "", type="password")
            opts = st.checkbox("Options Trading", value=False)
            dhan_cfg["options_trading"] = opts
            if not opts:
                dhan_cfg["eq_product"]     = st.selectbox("Product",     ["Intraday","Delivery"])
                dhan_cfg["eq_exchange"]    = st.selectbox("Exchange",    ["NSE","BSE"])
                dhan_cfg["eq_sec_id"]      = st.text_input("Sec ID",     "1594")
                dhan_cfg["eq_qty"]         = st.number_input("Broker Qty",1,1_000_000,1)
                dhan_cfg["eq_entry_otype"] = st.selectbox("Entry Order", ["Limit Order","Market Order"])
                dhan_cfg["eq_exit_otype"]  = st.selectbox("Exit Order",  ["Market Order","Limit Order"])
            else:
                dhan_cfg["opts_exchange"]    = st.selectbox("FnO Exchange",["NSE_FNO","BSE_FNO"])
                dhan_cfg["ce_security_id"]   = st.text_input("CE Sec ID","")
                dhan_cfg["pe_security_id"]   = st.text_input("PE Sec ID","")
                dhan_cfg["opts_qty"]         = st.number_input("Lots",1,1_000_000,65)
                dhan_cfg["opts_entry_otype"] = st.selectbox("Entry Order",["Market Order","Limit Order"])
                dhan_cfg["opts_exit_otype"]  = st.selectbox("Exit Order", ["Market Order","Limit Order"])
        st.divider()
        st.caption("Smart Investing v5.0  •  Educational use only")

    cfg = dict(ticker=ticker_sym, ticker_label=ticker_label,
               interval=interval, period=period,
               strategy=strategy, fast_ema=fast_ema, slow_ema=slow_ema,
               sl_type=sl_type, sl_pts=sl_pts, tgt_type=tgt_type, tgt_pts=tgt_pts,
               qty=qty, cd_en=cd_en, cd_s=cd_s, no_overlap=no_overlap, rr=rr,
               atr_period=atr_period, atr_sl_mult=atr_sl_mult, atr_tgt_mult=atr_tgt_mult,
               adv_filter=adv_filter, min_angle=min_angle,
               crossover_candle=crossover_candle,
               candle_size_pts=candle_size_pts, candle_atr_mult=candle_atr_mult,
               **dhan_cfg)

    # Page header
    hc = "#007a6a" if light else "#00e5b4"
    st.markdown(f"""
    <div style="padding:4px 0 10px">
        <span style="font-family:Syne,sans-serif;font-size:30px;font-weight:800;color:{hc}">
            📈 Smart Investing
        </span>
        <span style="font-family:Syne,sans-serif;font-size:12px;color:#7a8899;
                     margin-left:14px;vertical-align:middle">Algorithmic Trading Platform</span>
    </div>""", unsafe_allow_html=True)

    if not AUTO_REFRESH_OK:
        st.info("Install `streamlit-autorefresh` for live tab auto-update: `pip install streamlit-autorefresh`")

    tab_bt, tab_live, tab_hist = st.tabs(["🔬  Backtesting","⚡  Live Trading","📋  Trade History"])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — BACKTESTING
    # ════════════════════════════════════════════════════════════════════
    with tab_bt:
        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### 🔬 Backtest Engine")

        r1,r2 = st.columns([4,1])
        with r1:
            run_btn = st.button("▶  Run Backtest", type="primary",
                                use_container_width=True, key="run_bt")
        with r2:
            if st.button("↺ Clear", use_container_width=True, key="clr_bt"):
                st.session_state.update(bt_results=None,bt_chart_df=None,bt_run_key=None)
                st.rerun()

        if run_btn:
            with st.spinner(f"Fetching {ticker_label}  {interval}/{period}…"):
                res = fetch_ohlcv(ticker_sym, interval, period)
            if res is None:
                st.error(f"Data fetch failed for {ticker_sym} {interval}/{period}.")
            else:
                df_full, df_display = res
                with st.spinner("Running backtest…"):
                    tdf = run_backtest(
                        df_full, df_display, strategy, fast_ema, slow_ema,
                        sl_type, sl_pts, tgt_type, tgt_pts, qty, rr,
                        atr_period, atr_sl_mult, atr_tgt_mult,
                        adv_filter, min_angle, crossover_candle,
                        candle_size_pts, candle_atr_mult,
                    )
                df_chart = add_indicators(df_full.copy(), fast_ema, slow_ema, atr_period)
                if not df_display.empty:
                    df_chart = df_chart.loc[df_display.index[0]:]
                st.session_state["bt_results"]  = tdf
                st.session_state["bt_chart_df"] = df_chart
                st.session_state["bt_run_key"]  = f"{ticker_sym}|{interval}|{period}|{strategy}"

        tdf     = st.session_state.get("bt_results")
        dfc     = st.session_state.get("bt_chart_df")
        run_key = st.session_state.get("bt_run_key")

        if tdf is not None:
            if run_key: st.caption(f"Results: `{run_key}`")
            if tdf.empty:
                st.warning("No trades generated. Adjust parameters or check data availability.")
            else:
                n    = len(tdf)
                wins = tdf["PnL (Rs)"] > 0; losses = tdf["PnL (Rs)"] < 0
                nw   = int(wins.sum()); nl = int(losses.sum())
                tot  = float(tdf["PnL (Rs)"].sum())
                acc  = nw/n*100 if n else 0
                avg_p= float(tdf.loc[wins,"PnL (Rs)"].mean()) if nw else 0.0
                avg_l= float(tdf.loc[losses,"PnL (Rs)"].mean()) if nl else 0.0
                best = float(tdf["PnL (Rs)"].max()); wst = float(tdf["PnL (Rs)"].min())
                tp_w = float(tdf.loc[wins,"Points"].sum()) if "Points" in tdf.columns else 0.0
                tp_l = float(tdf.loc[losses,"Points"].sum()) if "Points" in tdf.columns else 0.0
                viols= int(tdf["Exit Reason"].str.contains("⚠", na=False).sum()) if "Exit Reason" in tdf.columns else 0

                # Duration stats
                dur_avg = "—"
                try:
                    def _to_sec(row):
                        try:
                            et = row["Entry Time"]; xt = row["Exit Time"]
                            if hasattr(et,"total_seconds"): return 0
                            return (xt - et).total_seconds()
                        except: return 0
                    secs = tdf.apply(_to_sec, axis=1)
                    avg_s = float(secs.mean()); tot_s = int(avg_s)
                    dur_avg = f"{tot_s//3600:02d}h {(tot_s%3600)//60:02d}m {tot_s%60:02d}s"
                except: pass

                # Row 1 — trade counts
                m = st.columns(8)
                for col_,l_,v_ in zip(m,
                    ["Trades","Wins","Losses","Accuracy","Win Pts","Loss Pts","Violations","Avg Duration"],
                    [n, nw, nl, f"{acc:.1f}%",
                     f"{tp_w:+.1f}", f"{tp_l:+.1f}", viols, dur_avg]):
                    col_.metric(l_,v_)

                # Row 2 — PnL
                m2 = st.columns(5)
                for col_,l_,v_ in zip(m2,
                    ["Total PnL","Avg Win","Avg Loss","Best Trade","Worst Trade"],
                    [f"Rs{tot:+,.0f}",f"Rs{avg_p:+.0f}",f"Rs{avg_l:+.0f}",
                     f"Rs{best:+.0f}",f"Rs{wst:+.0f}"]):
                    col_.metric(l_,v_)

                st.divider()

                if dfc is not None:
                    st.plotly_chart(
                        build_chart(dfc, fast_ema, slow_ema, trades_df=tdf,
                                    title=f"{ticker_label} · {strategy} · {interval}/{period}",
                                    light=light),
                        use_container_width=True)

                st.markdown("#### 📊 Trade Log")
                disp = tdf.copy()
                disp["Entry Time"] = disp["Entry Time"].astype(str)
                disp["Exit Time"]  = disp["Exit Time"].astype(str)

                wbg = "rgba(0,150,120,.10)" if light else "rgba(0,229,180,.09)"
                lbg = "rgba(220,50,80,.08)"  if light else "rgba(255,77,109,.09)"
                wfg = "#004d3a" if light else "#00e5b4"
                lfg = "#7a0000" if light else "#ff8fa3"
                nfg = "#1a2236" if light else "#c8d0e0"

                def _srow(row):
                    p = row["PnL (Rs)"]
                    if p>0: return [f"background-color:{wbg};color:{wfg}" for _ in row]
                    elif p<0: return [f"background-color:{lbg};color:{lfg}" for _ in row]
                    return [f"color:{nfg}" for _ in row]

                styled = (disp.style.apply(_srow,axis=1)
                          .format({"PnL (Rs)":"Rs{:+.2f}","Points":"{:+.2f}",
                                   "Entry Price":"{:.2f}","Exit Price":"{:.2f}"}))
                st.dataframe(styled, use_container_width=True, height=420,
                             column_config={"Trade Type":st.column_config.TextColumn(width=70)})

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — LIVE TRADING
    # ════════════════════════════════════════════════════════════════════
    with tab_live:
        # Auto-refresh every 3 s while engine is running (suppressed fade via CSS)
        if shared.get("running") and AUTO_REFRESH_OK:
            st_autorefresh(interval=3000, key="live_ar")

        # Show LTP for CURRENT sidebar ticker always
        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### ⚡ Live Trading")

        running = shared.get("running", False)

        # Warn if sidebar ticker ≠ running ticker
        run_cfg = shared.get("cfg")
        if run_cfg and run_cfg.get("ticker") != ticker_sym and running:
            st.warning(f"⚠ Engine is running on **{run_cfg.get('ticker_label','?')}** "
                       f"but sidebar shows **{ticker_label}**. Stop and restart to change ticker.")

        if running:
            st.markdown("""
            <div style="display:inline-flex;align-items:center;gap:8px;
                        background:rgba(0,229,180,.1);border:1px solid rgba(0,229,180,.3);
                        border-radius:20px;padding:6px 16px;margin-bottom:10px">
                <span style="width:8px;height:8px;border-radius:50%;background:#00e5b4;
                             display:inline-block;animation:pulse 1s infinite"></span>
                <span style="font-family:Syne,sans-serif;font-size:12px;font-weight:700;
                             color:#00e5b4;letter-spacing:1px">LIVE RUNNING</span>
            </div>
            <style>@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(1.4)}}</style>
            """, unsafe_allow_html=True)

        b1,b2,b3,_ = st.columns([1.2,1.2,1.5,3])

        with b1:
            if st.button("▶ Start", type="primary", use_container_width=True,
                         disabled=running, key="live_start"):
                # Always create fresh objects — prevents "can't start thread" on reload
                new_evt    = threading.Event()                    # fresh, NOT set
                new_shared = _new_live()
                new_shared["running"] = True
                new_shared["cfg"]     = cfg.copy()
                st.session_state["live_stop_evt"] = new_evt
                st.session_state["live_shared"]   = new_shared
                # Launch thread — all state passed explicitly (no global deps)
                threading.Thread(
                    target=live_engine,
                    args=(cfg, new_shared, new_evt),
                    daemon=True,
                ).start()
                st.rerun()

        with b2:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not running, key="live_stop"):
                evt = st.session_state.get("live_stop_evt")
                if evt is not None:
                    evt.set()          # signals the thread to exit within 1.5 s
                shared["running"] = False
                shared["pending_entry"] = None
                time.sleep(0.4)        # brief wait so thread sees the event
                st.rerun()

        with b3:
            if st.button("⚡ Square Off", use_container_width=True, key="live_sq"):
                pos = shared.get("position")
                if pos:
                    ltp_now = shared.get("ltp") or float(pos["entry_price"])
                    tt_sq   = pos["trade_type"]
                    pnl = ((ltp_now-float(pos["entry_price"])) if tt_sq=="buy"
                           else (float(pos["entry_price"])-ltp_now))*qty
                    pts = (ltp_now-float(pos["entry_price"])) if tt_sq=="buy" else (float(pos["entry_price"])-ltp_now)
                    try:
                        et_dt = datetime.strptime(pos["entry_time"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    except: et_dt = datetime.now(IST)
                    xt_dt = datetime.now(IST)
                    shared["trade_history"].append(dict(
                        **{"Entry Time":pos["entry_time"],
                           "Exit Time": xt_dt.strftime("%Y-%m-%d %H:%M:%S"),
                           "Duration":  _dur(et_dt, xt_dt),
                           "Trade Type":tt_sq.upper(),
                           "Entry Price":round(float(pos["entry_price"]),2),
                           "Exit Price": round(float(ltp_now),2),
                           "SL":     round(float(pos["sl"]),2)     if pos.get("sl")     is not None else "—",
                           "Target": round(float(pos["target"]),2) if pos.get("target") is not None else "—",
                           "Entry Reason":pos.get("entry_reason",""),
                           "Exit Reason": "Manual Square Off",
                           "Points":round(pts,2), "PnL (Rs)":round(pnl,2),
                           "Qty":qty, "Mode":"Live"}
                    ))
                    shared["position"]     = None
                    shared["pending_entry"]= None
                    shared["last_exit_ts"] = datetime.now(IST)
                    if cfg.get("dhan_en"): place_exit(cfg, tt_sq, float(ltp_now))
                    st.success(f"Squared off @ {float(ltp_now):.2f}  |  PnL Rs{pnl:+.2f}")
                else:
                    st.info("No open position.")
                st.rerun()

        st.divider()

        # Active config — shows running config (not sidebar, which user may have changed)
        live_cfg_shown = shared.get("cfg") or cfg
        with st.expander("⚙ Active Configuration", expanded=True):
            cc = st.columns(5)
            cc[0].metric("Ticker",   live_cfg_shown.get("ticker_label","—"))
            cc[1].metric("Interval", live_cfg_shown.get("interval","—"))
            cc[2].metric("Period",   live_cfg_shown.get("period","—"))
            cc[3].metric("Strategy", live_cfg_shown.get("strategy","—"))
            cc[4].metric("Qty",      live_cfg_shown.get("qty","—"))
            cc2 = st.columns(4)
            cc2[0].metric(f"EMA {live_cfg_shown.get('fast_ema','?')} (fast)",
                          f"{shared['ema_fast_val']:.2f}" if shared.get("ema_fast_val") is not None else "—")
            cc2[1].metric(f"EMA {live_cfg_shown.get('slow_ema','?')} (slow)",
                          f"{shared['ema_slow_val']:.2f}" if shared.get("ema_slow_val") is not None else "—")
            cc2[2].metric("SL",  f"{live_cfg_shown.get('sl_type','—')} / {live_cfg_shown.get('sl_pts','—')} pts")
            cc2[3].metric("TGT", f"{live_cfg_shown.get('tgt_type','—')} / {live_cfg_shown.get('tgt_pts','—')} pts")

        st.divider()

        main_col, log_col = st.columns([1.7, 1])

        with main_col:
            ef_v  = shared.get("ema_fast_val")
            es_v  = shared.get("ema_slow_val")
            atr_v = shared.get("atr_val")
            ltp_c = shared.get("ltp")
            pos   = shared.get("position")
            pen   = shared.get("pending_entry")

            # EMA metrics — only show if engine is running on this ticker
            eng_ticker = (shared.get("cfg") or {}).get("ticker","")
            if ef_v is not None and es_v is not None:
                diff  = ef_v - es_v; trend = "Bullish ↑" if diff>0 else "Bearish ↓"
                mv    = st.columns(5)
                fast_label = live_cfg_shown.get("fast_ema", fast_ema)
                slow_label = live_cfg_shown.get("slow_ema", slow_ema)
                mv[0].metric(f"EMA {fast_label}", f"{ef_v:.2f}")
                mv[1].metric(f"EMA {slow_label}", f"{es_v:.2f}")
                mv[2].metric("EMA Diff", f"{diff:+.2f}",
                             delta=trend, delta_color="normal" if diff>0 else "inverse")
                mv[3].metric("ATR", f"{atr_v:.2f}" if atr_v else "—")
                mv[4].metric("LTP", f"{ltp_c:.2f}" if ltp_c else "—")

                # PnL of open position (real-time)
                if pos is not None and ltp_c is not None:
                    tt_pos  = pos["trade_type"]
                    ep_pos  = float(pos["entry_price"])
                    pnl_lv  = ((ltp_c-ep_pos) if tt_pos=="buy" else (ep_pos-ltp_c))*qty
                    pts_lv  = (ltp_c-ep_pos) if tt_pos=="buy" else (ep_pos-ltp_c)
                    pc = "#007a6a" if light else "#00e5b4"
                    nc = "#c0152d" if light else "#ff4d6d"
                    vc = pc if pnl_lv>=0 else nc
                    st.markdown(f"""
                    <div style="margin:8px 0 4px;display:flex;gap:32px;font-family:JetBrains Mono,monospace">
                        <div><span style="font-size:10px;color:#7a8899">OPEN PnL</span><br>
                             <span style="font-size:22px;font-weight:700;color:{vc}">Rs{pnl_lv:+.2f}</span></div>
                        <div><span style="font-size:10px;color:#7a8899">POINTS</span><br>
                             <span style="font-size:22px;font-weight:700;color:{vc}">{pts_lv:+.2f}</span></div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Start the engine to see live EMA and indicator values.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Pending entry notice
            if pen is not None:
                pa = "#007a6a" if light else "#00e5b4"
                pb = "rgba(0,120,100,.07)" if light else "rgba(0,229,180,.07)"
                st.markdown(f"""
                <div style="background:{pb};border:1px solid {pa};border-radius:8px;
                            padding:10px 16px;margin-bottom:10px;font-family:Syne,sans-serif;font-size:12px">
                    ⏳ <b style="color:{pa}">Pending {pen.get('trade_type','').upper()}</b>
                    — waiting for next candle open.<br>
                    <small style="color:#8899aa">{pen.get('reason','')}</small>
                </div>""", unsafe_allow_html=True)

            # Open position card
            if pos is not None:
                tt    = pos["trade_type"]; ep = float(pos["entry_price"])
                sl_   = pos.get("sl");    tgt_ = pos.get("target")
                tt_clr = ("#00b896" if light else "#00e5b4") if tt=="buy" else ("#e0284a" if light else "#ff4d6d")
                tx = "#1a2236" if light else "#c8d0e0"; mt = "#7a8899" if light else "#4a5568"
                sl_s  = f"{float(sl_):.2f}"  if sl_  is not None else "—"
                tgt_s = f"{float(tgt_):.2f}" if tgt_ is not None else "N/A"
                ltp_s = f"{ltp_c:.2f}" if ltp_c is not None else "—"
                pnl_live = ((ltp_c-ep) if (ltp_c is not None and tt=="buy")
                            else ((ep-ltp_c) if ltp_c is not None else 0.0))*qty
                pc_ = "card-green" if pnl_live>=0 else "card-red"
                st.markdown(f"""
                <div class="card {pc_}">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                    <span style="font-family:Syne,sans-serif;font-size:13px;font-weight:800;
                                 letter-spacing:2px;color:{tt_clr}">● OPEN {tt.upper()}</span>
                  </div>
                  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;
                              font-family:JetBrains Mono,monospace;font-size:13px;color:{tx}">
                    <div><div style="color:{mt};font-size:10px">ENTRY</div>{ep:.2f}</div>
                    <div><div style="color:{mt};font-size:10px">LTP</div>{ltp_s}</div>
                    <div><div style="color:#e0284a;font-size:10px">SL</div>{sl_s}</div>
                    <div><div style="color:{("#007a6a" if light else "#00e5b4")};font-size:10px">TARGET</div>{tgt_s}</div>
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
                            text-align:center;padding:20px;color:#7a8899;
                            font-family:Syne,sans-serif;font-size:13px">No open position</div>""",
                            unsafe_allow_html=True)

            # Chart — always visible if data available
            df_live = shared.get("chart_df")
            if df_live is not None and not df_live.empty:
                fast_l = live_cfg_shown.get("fast_ema", fast_ema)
                slow_l = live_cfg_shown.get("slow_ema", slow_ema)
                st.plotly_chart(
                    build_chart(df_live, fast_l, slow_l, position=pos,
                                title=f"Live · {live_cfg_shown.get('ticker_label','?')} · {interval}",
                                light=light),
                    use_container_width=True)

                # Last candle info
                lr = df_live.iloc[-1]; ec_f = f"EMA_{fast_l}"; ec_s = f"EMA_{slow_l}"
                st.markdown("**Last Fetched Candle**")
                st.dataframe(pd.DataFrame([{
                    "Time":  str(df_live.index[-1]),
                    "Open":  round(float(lr["Open"]),2), "High": round(float(lr["High"]),2),
                    "Low":   round(float(lr["Low"]),2),  "Close":round(float(lr["Close"]),2),
                    "Vol":   int(float(lr.get("Volume",0))),
                    f"EMA{fast_l}":round(float(df_live[ec_f].iloc[-1]),2) if ec_f in df_live.columns else "—",
                    f"EMA{slow_l}":round(float(df_live[ec_s].iloc[-1]),2) if ec_s in df_live.columns else "—",
                    "ATR":  round(float(df_live["ATR"].iloc[-1]),2) if "ATR" in df_live.columns else "—",
                }]), use_container_width=True, hide_index=True)

        with log_col:
            st.markdown("**Activity Log**")
            logs = shared.get("log",[])
            sep  = "#eef1f7" if light else "#0f1623"
            tx2  = "#1a2236" if light else "#c8d0e0"
            mt2  = "#7a8899" if light else "#4a5568"
            log_html = (
                "".join(f"<div style='padding:2px 0;border-bottom:1px solid {sep};color:{tx2}'>{l}</div>"
                        for l in reversed(logs))
                if logs else f"<div style='color:{mt2}'>No activity — start the engine.</div>"
            )
            st.markdown(f'<div class="logbox">{log_html}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — TRADE HISTORY  (live trades)
    # ════════════════════════════════════════════════════════════════════
    with tab_hist:
        ltp_widget(ticker_sym, ticker_label, light)
        st.markdown("### 📋 Trade History")
        st.caption("Live completed trades — auto-updates while engine is running.")

        hist = shared.get("trade_history",[])
        _,hcol = st.columns([5,1])
        with hcol:
            if st.button("🗑 Clear", use_container_width=True, key="clr_hist"):
                shared["trade_history"] = []; st.rerun()

        bg3="#fff" if light else "#0f1623"; brd3="#d0d7e6" if light else "#1a2236"
        if not hist:
            st.markdown(f"""
            <div style="background:{bg3};border:1px solid {brd3};border-radius:12px;
                        text-align:center;padding:40px;color:#7a8899;font-family:Syne,sans-serif">
                No live trades yet.
            </div>""", unsafe_allow_html=True)
        else:
            hdf     = pd.DataFrame(hist)
            tot_pnl = float(hdf["PnL (Rs)"].sum())
            wins_h  = hdf["PnL (Rs)"] > 0; losses_h = hdf["PnL (Rs)"] < 0
            nw_h    = int(wins_h.sum()); n_h = len(hdf)
            acc_h   = nw_h/n_h*100 if n_h else 0
            avg_p_h = float(hdf.loc[wins_h,"PnL (Rs)"].mean()) if nw_h else 0.0
            avg_l_h = float(hdf.loc[losses_h,"PnL (Rs)"].mean()) if losses_h.sum() else 0.0
            best_h  = float(hdf["PnL (Rs)"].max()); worst_h = float(hdf["PnL (Rs)"].min())
            tp_w_h  = float(hdf.loc[wins_h,"Points"].sum()) if "Points" in hdf.columns else 0.0
            tp_l_h  = float(hdf.loc[losses_h,"Points"].sum()) if "Points" in hdf.columns else 0.0

            hm = st.columns(8)
            for c_,l_,v_ in zip(hm,
                ["Trades","Win Rate","Total PnL","Avg Win","Avg Loss","Win Pts","Loss Pts","Best"],
                [n_h, f"{acc_h:.1f}%", f"Rs{tot_pnl:+,.0f}",
                 f"Rs{avg_p_h:+.0f}", f"Rs{avg_l_h:+.0f}",
                 f"{tp_w_h:+.1f}", f"{tp_l_h:+.1f}", f"Rs{best_h:+.0f}"]):
                c_.metric(l_,v_)

            st.divider()
            wbg = "rgba(0,150,120,.10)" if light else "rgba(0,229,180,.09)"
            lbg = "rgba(220,50,80,.08)"  if light else "rgba(255,77,109,.09)"
            wfg = "#004d3a" if light else "#00e5b4"
            lfg = "#7a0000" if light else "#ff8fa3"
            nfg = "#1a2236" if light else "#c8d0e0"

            def _hs(row):
                p = row["PnL (Rs)"]
                if p>0: return [f"background-color:{wbg};color:{wfg}" for _ in row]
                elif p<0: return [f"background-color:{lbg};color:{lfg}" for _ in row]
                return [f"color:{nfg}" for _ in row]

            styled_h = (hdf.style.apply(_hs,axis=1)
                        .format({"PnL (Rs)":"Rs{:+.2f}","Points":"{:+.2f}",
                                 "Entry Price":"{:.2f}","Exit Price":"{:.2f}"}))
            st.dataframe(styled_h, use_container_width=True, height=420)

            st.markdown("#### Cumulative PnL")
            hdf["Cum PnL"] = hdf["PnL (Rs)"].cumsum()
            up_c = "#00b896" if light else "#00e5b4"; dn_c = "#e0284a" if light else "#ff4d6d"
            base = _base(light); xax = base.pop("xaxis",{}); yax = base.pop("yaxis",{})
            fp   = go.Figure()
            fp.add_trace(go.Scatter(
                x=list(range(1,n_h+1)), y=list(hdf["Cum PnL"].astype(float)),
                mode="lines+markers", line=dict(color=up_c,width=2),
                marker=dict(size=5,color=[up_c if p>=0 else dn_c for p in hdf["PnL (Rs)"]]),
                fill="tozeroy",
                fillcolor="rgba(0,180,140,0.08)" if light else "rgba(0,229,180,0.06)"))
            fp.add_hline(y=0,line=dict(color="#94a3b8",width=1,dash="dash"))
            fp.update_layout(height=260,showlegend=False,
                             xaxis_title="Trade #",yaxis_title="PnL (Rs)",**base)
            fp.update_xaxes(**xax); fp.update_yaxes(**yax)
            st.plotly_chart(fp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
