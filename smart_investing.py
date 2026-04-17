# ╔══════════════════════════════════════════════════════════════════╗
# ║       SMART INVESTING  –  Professional Algo Trading  v7         ║
# ║                                                                  ║
# ║  ARCHITECTURE:                                                   ║
# ║  • Module-level singleton _STORE — created once per process,    ║
# ║    survives all Streamlit reruns (Python module is cached)       ║
# ║  • Background thread writes ONLY to _STORE — zero Streamlit     ║
# ║    calls, zero session_state access from thread                  ║
# ║  • Auto-refresh via st.rerun() after 1.5s sleep (metrics only)   ║
# ║  • dhan_client captured in main thread, passed as plain arg      ║
# ╚══════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, threading, math, warnings, requests, itertools
from datetime import datetime
import pytz

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
IST = pytz.timezone("Asia/Kolkata")

# ════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON  (persists across all Streamlit reruns)
# Python caches imported modules. This block runs ONCE per server
# process, not once per rerun. The thread and the UI share this object.
# ════════════════════════════════════════════════════════════════════
_LOCK = threading.Lock()

# _STORE is never re-assigned — only its contents change via _set/_get
# This guarantees thread and UI always see the same object.
_STORE = {
    "running"    : False,
    "status"     : "STOPPED",
    "pos"        : None,       # current open position dict or None
    "trades"     : [],         # list of completed trade dicts
    "log"        : [],         # activity log strings
    "df"         : None,       # latest OHLCV DataFrame
    "ltp"        : None,       # latest price float
    "prev_close" : None,       # previous day close
    "last_ts"    : None,       # datetime of last fetch
    "ind"        : {},         # {"ef":…, "es":…, "atr":…, …}
    "ew"         : {},         # Elliott Wave analysis dict
    "_rl_ts"     : 0.0,        # rate-limit timestamp for yfinance
    "_tid"       : None,        # active thread id — stops stale threads
}

def _get(key, default=None):
    with _LOCK: return _STORE.get(key, default)

def _set(key, val):
    with _LOCK: _STORE[key] = val

def _push(key, val, maxlen=500):
    with _LOCK:
        _STORE[key].append(val)
        if len(_STORE[key]) > maxlen:
            _STORE[key] = _STORE[key][-maxlen:]

def _log(msg):
    ts = datetime.now(IST).strftime("%H:%M:%S")
    _push("log", f"[{ts}] {msg}")

# ════════════════════════════════════════════════════════════════════
# SESSION STATE  (UI-only, survives reruns via Streamlit session)
# ════════════════════════════════════════════════════════════════════
for _k, _v in {
    "bt_trades": [], "bt_viol": [], "bt_df": None, "bt_ran": False,
    "dhan_client": None, "opt_results": [], "opt_ran": False,
    "_aFE": 9, "_aSE": 15, "_aSL": 10.0, "_aTG": 20.0, "_aRR": 2.0,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.main{background:#0e1117}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif}
.hdr{background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);padding:12px 22px;
  border-radius:12px;margin-bottom:12px;border:1px solid #2d4a6e;
  display:flex;align-items:center;justify-content:space-between}
.hdr h1{margin:0;color:#e2e8f0;font-size:21px}
.hdr p{margin:0;color:#90cdf4;font-size:11px}
.ltpbar{background:linear-gradient(135deg,#1a1f2e,#252b3d);border:1px solid #2d4a6e;
  border-radius:10px;padding:9px 18px;display:flex;align-items:center;
  gap:16px;margin-bottom:10px}
.ltpval{color:#e2e8f0;font-size:24px;font-weight:700}
.ltpup{color:#48bb78;font-size:13px;font-weight:600}
.ltpdn{color:#fc8181;font-size:13px;font-weight:600}
.ltpmeta{color:#718096;font-size:10px;margin-left:auto}
.card{background:#1a1f2e;border:1px solid #2d3748;border-radius:9px;padding:12px;margin:4px 0}
.ecard{background:#141d2b;border:1px solid #2d4a6e;border-radius:8px;
  padding:10px;text-align:center}
.cfgb{background:#1a1f2e;border-radius:8px;padding:11px;border-left:3px solid #4299e1;
  margin:7px 0;font-size:12px;line-height:1.8;color:#cbd5e0}
.ewrec{background:#111d2e;border:1px solid #2d5a8e;border-radius:10px;
  padding:14px;margin:8px 0}
.wrow{background:#1a2535;border:1px solid #2d3748;border-radius:7px;
  padding:9px 12px;margin:4px 0}
.logb{background:#0d1117;border-radius:8px;padding:8px;height:185px;
  overflow-y:auto;font-family:'Courier New',monospace;font-size:11px;color:#a0aec0}
.bdg{display:inline-block;padding:3px 12px;border-radius:16px;
  font-size:11px;font-weight:700;color:white;margin-bottom:6px}
.pos-box{border-radius:10px;padding:14px;border:1px solid}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:transparent}
.stTabs [data-baseweb="tab"]{background:#1a1f2e;border-radius:8px;
  padding:7px 18px;border:1px solid #2d3748;color:#a0aec0}
.stTabs [aria-selected="true"]{background:#2b6cb0!important;color:white!important}
section[data-testid="stSidebar"]{background:#0d1117}
section[data-testid="stSidebar"] label{color:#a0aec0!important;font-size:12px}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════
TICKERS = {
    "Nifty 50": "^NSEI", "BankNifty": "^NSEBANK", "Sensex": "^BSESN",
    "BTC": "BTC-USD", "ETH": "ETH-USD", "Gold": "GC=F", "Silver": "SI=F",
    "Custom": None,
}
TF_PERIODS = {
    "1m":  ["1d","5d","7d"],
    "5m":  ["1d","5d","7d","1mo"],
    "15m": ["1d","5d","7d","1mo"],
    "1h":  ["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":  ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk": ["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
MAX_FETCH = {"1m":"7d","5m":"60d","15m":"60d","1h":"730d","1d":"max","1wk":"max"}
TF_MIN    = {"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080}

# ════════════════════════════════════════════════════════════════════
# DATA FETCH
# ════════════════════════════════════════════════════════════════════
def _rate_wait():
    """Min 1.5s between yfinance calls. Thread-safe via _LOCK."""
    with _LOCK:
        gap = time.time() - _STORE["_rl_ts"]
        if gap < 1.5:
            time.sleep(1.5 - gap)
        _STORE["_rl_ts"] = time.time()

def _clean(raw):
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df.index = (df.index.tz_convert(IST) if df.index.tz
                else df.index.tz_localize("UTC").tz_convert(IST))
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    return df[~df.index.duplicated(keep="last")].sort_index()

def fetch_live(symbol, interval, period):
    """Call from background thread only."""
    _rate_wait()
    try:
        raw = yf.Ticker(symbol).history(
            period=MAX_FETCH.get(interval, period),
            interval=interval, auto_adjust=True, prepost=False)
        df = _clean(raw)
        if df is None or df.empty:
            _rate_wait()
            raw = yf.Ticker(symbol).history(
                period=period, interval=interval,
                auto_adjust=True, prepost=False)
            df = _clean(raw)
        return df
    except Exception as e:
        _log(f"[fetch] {e}")
        return None

def fetch_ui(symbol, interval, period):
    """Call from main thread (backtest / optimization) — no rate-wait."""
    try:
        raw = yf.Ticker(symbol).history(
            period=MAX_FETCH.get(interval, period),
            interval=interval, auto_adjust=True, prepost=False)
        df = _clean(raw)
        if df is None or df.empty:
            raw = yf.Ticker(symbol).history(
                period=period, interval=interval,
                auto_adjust=True, prepost=False)
            df = _clean(raw)
        return df
    except Exception as e:
        st.error(f"Fetch error: {e}")
        return None

# ════════════════════════════════════════════════════════════════════
# INDICATORS  (TradingView-accurate EMA)
# ════════════════════════════════════════════════════════════════════
def ema_tv(series, n):
    if len(series) < n:
        return pd.Series(np.nan, index=series.index)
    a    = 2.0 / (n + 1)
    vals = series.ffill().values.astype(float)
    out  = np.full(len(vals), np.nan)
    out[n - 1] = np.nanmean(vals[:n])
    for i in range(n, len(vals)):
        v      = vals[i] if not np.isnan(vals[i]) else out[i - 1]
        out[i] = a * v + (1 - a) * out[i - 1]
    return pd.Series(out, index=series.index)

def atr_s(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return ema_tv(tr, n)

def add_ind(df, fast, slow):
    if df is None or df.empty:
        return df
    df       = df.copy()
    df["ef"] = ema_tv(df["close"], fast)
    df["es"] = ema_tv(df["close"], slow)
    df["atr"]= atr_s(df)
    df["cu"] = (df["ef"] > df["es"]) & (df["ef"].shift(1) <= df["es"].shift(1))
    df["cd"] = (df["ef"] < df["es"]) & (df["ef"].shift(1) >= df["es"].shift(1))
    return df

def ema_angle(s, lb=3):
    v = s.dropna()
    if len(v) < lb + 1: return 0.0
    b = v.iloc[-lb]
    return 0.0 if b == 0 else abs(math.degrees(math.atan((v.iloc[-1]-b)/b*100/lb)))

# ════════════════════════════════════════════════════════════════════
# SIGNALS
# ════════════════════════════════════════════════════════════════════
def ema_sig(df, idx, cfg):
    if idx < 1 or idx >= len(df): return None, None
    r, p = df.iloc[idx], df.iloc[idx - 1]
    ef, es, pf, ps = r.get("ef",np.nan), r.get("es",np.nan), p.get("ef",np.nan), p.get("es",np.nan)
    if any(np.isnan([ef, es, pf, ps])): return None, None
    if cfg.get("use_angle", False):
        if ema_angle(df["ef"].iloc[:idx+1]) < float(cfg.get("min_angle", 0)): return None, None
    ct   = cfg.get("crossover_type", "Simple")
    body = abs(r["close"] - r["open"])
    def ok():
        if ct == "Simple":        return True
        if ct == "Custom Candle": return body >= float(cfg.get("custom_candle", 10))
        if ct == "ATR Candle":
            a = float(r.get("atr", 0) or 0); return a == 0 or body >= a
        return True
    f, s = int(cfg.get("fast_ema", 9)), int(cfg.get("slow_ema", 15))
    if ef > es and pf <= ps and ok(): return "buy",  f"EMA({f}) crossed ABOVE EMA({s})"
    if ef < es and pf >= ps and ok(): return "sell", f"EMA({f}) crossed BELOW EMA({s})"
    return None, None

def ema_rev(df, idx, pt):
    if idx < 1: return False
    r, p = df.iloc[idx], df.iloc[idx - 1]
    ef, es, pf, ps = r.get("ef",np.nan), r.get("es",np.nan), p.get("ef",np.nan), p.get("es",np.nan)
    if any(np.isnan([ef, es, pf, ps])): return False
    if pt == "buy"  and ef < es and pf >= ps: return True
    if pt == "sell" and ef > es and pf <= ps: return True
    return False

def fdt(dt):
    if hasattr(dt, "strftime"): return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)

# ════════════════════════════════════════════════════════════════════
# ELLIOTT WAVE ANALYSIS  — fully automated, detailed output
# ════════════════════════════════════════════════════════════════════
def _zigzag(df, win=5):
    h, l = df["high"].values, df["low"].values
    pts  = []
    for i in range(win, len(df) - win):
        lo, hi = max(0, i-win), min(len(df), i+win+1)
        if h[i] == h[lo:hi].max(): pts.append({"dt": df.index[i], "px": h[i], "i": i, "t": "H"})
        if l[i] == l[lo:hi].min(): pts.append({"dt": df.index[i], "px": l[i], "i": i, "t": "L"})
    pts.sort(key=lambda x: x["i"])
    cl = []
    for p in pts:
        if not cl: cl.append(p); continue
        if cl[-1]["t"] == p["t"]:
            if ((p["t"]=="H" and p["px"]>cl[-1]["px"]) or
                    (p["t"]=="L" and p["px"]<cl[-1]["px"])):
                cl[-1] = p
        else:
            cl.append(p)
    return cl

def _fq(ratio, targets):
    """Fibonacci quality: 1.0 = perfect, 0.0 = far from all targets."""
    if ratio <= 0: return 0.0
    return max(0.0, 1.0 - min(abs(ratio - t) / t for t in targets) / 0.15)

def detect_ew(df):
    """
    Returns rich dict describing detected wave pattern.
    Handles: 5-wave impulse, ABC correction, or 'forming'.
    Always includes automated trade signal when pattern is clear.
    """
    EMPTY = {
        "ok": False, "label": "Analysing…", "direction": "UNCLEAR",
        "wave_type": "—", "waves_plot": [], "rules": [],
        "confidence": 0, "current_wave": "—", "wave_rows": [],
        "projections": {}, "signal": None, "signal_reason": "—",
        "sl": None, "target": None, "after": "Waiting for data…",
        "current_price": 0, "bias": "",
    }
    if df is None or len(df) < 25:
        return {**EMPTY, "label": "Need ≥25 candles"}
    cp  = float(df["close"].iloc[-1])
    pts = _zigzag(df, win=max(3, len(df)//40))
    if len(pts) < 4:
        return {**EMPTY, "ok": True, "label": "Forming pivots", "current_price": cp,
                "after": "Wait for more swing highs/lows"}

    # ── Try 5-wave impulse (needs 6 pivot points) ─────────────────
    if len(pts) >= 6:
        for st0, direction, sign in [("L","BULLISH",1), ("H","BEARISH",-1)]:
            s = pts[-6:]
            if s[0]["t"] != st0: continue
            P = [p["px"] for p in s]
            W = [sign*(P[i+1]-P[i]) for i in range(5)]   # W[0]=W1, W[4]=W5
            if W[0] <= 0 or W[2] <= 0: continue
            w2r = abs(W[1])/W[0]; w3x = W[2]/W[0]; w4r = abs(W[3])/W[2]

            rules = [
                ("W3 not shortest",        W[2] >= min(W[0], max(W[4], 0.001)),
                 f"W1={W[0]:.1f}  W3={W[2]:.1f}  W5={W[4]:.1f}"),
                ("W2 retraces <100% W1",   w2r < 1.0,
                 f"W2 retrace = {w2r*100:.1f}%  (ideal 38.2–61.8%)"),
                ("W4 no overlap W1 end",   (P[4]>P[1]) if sign==1 else (P[4]<P[1]),
                 f"W4 end={P[4]:.0f}  W1 end={P[1]:.0f}"),
                ("W3 ≥ 61.8% of W1",       W[2] >= W[0]*0.618,
                 f"W3/W1 = {w3x:.2f}x  (ideal ≥1.618x)"),
                ("W4 retrace <100% W3",    w4r < 1.0,
                 f"W4 retrace = {w4r*100:.1f}%  (ideal 23.6–38.2%)"),
            ]
            n_pass = sum(1 for r in rules if r[1])
            if n_pass < 3: continue

            fib_qual = (_fq(w2r,[0.382,0.5,0.618]) +
                        _fq(w3x,[1.0,1.382,1.618,2.0,2.618]) +
                        _fq(w4r,[0.236,0.382,0.5])) / 3
            conf = min(100, int(n_pass/5*60 + fib_qual*40))

            wc = ["#f6ad55","#fc8181","#48bb78","#f6ad55","#76e4f7"]
            wave_rows = [
                ("1","Impulse",  "UP" if sign==1 else "DOWN", P[0], P[1],
                 round(abs(P[1]-P[0]),2), "—  (impulse start)", wc[0]),
                ("2","Correction","DOWN" if sign==1 else "UP",  P[1], P[2],
                 round(abs(P[2]-P[1]),2),
                 f"{w2r*100:.1f}% retrace of W1  (ideal 38.2–61.8%)", wc[1]),
                ("3","Impulse",  "UP" if sign==1 else "DOWN", P[2], P[3],
                 round(abs(P[3]-P[2]),2),
                 f"{w3x:.2f}× W1  (ideal ≥1.618×, strongest wave)", wc[2]),
                ("4","Correction","DOWN" if sign==1 else "UP",  P[3], P[4],
                 round(abs(P[4]-P[3]),2),
                 f"{w4r*100:.1f}% retrace of W3  (ideal 23.6–38.2%)", wc[3]),
                ("5","Impulse (forming)","UP" if sign==1 else "DOWN", P[4], P[5],
                 round(abs(P[5]-P[4]),2), "Currently forming — see targets below", wc[4]),
            ]
            projs = {
                f"W5 = 61.8% of W1":  round(P[4] + sign*W[0]*0.618, 2),
                f"W5 = 100% of W1":   round(P[4] + sign*W[0],       2),
                f"W5 = 161.8% of W1": round(P[4] + sign*W[0]*1.618, 2),
            }
            sl_price  = round(P[4] - sign * W[0] * 0.05, 2)   # just beyond W4
            tgt_price = round(P[4] + sign * W[0],          2)  # W5 = W1

            return {
                "ok": True,
                "label": f"5-Wave Impulse ({'Bullish' if sign==1 else 'Bearish'})",
                "direction": direction,
                "wave_type": "5-Wave Impulse",
                "waves_plot": [{"label":str(n+1),"start":s[n],"end":s[n+1]} for n in range(5)],
                "rules":  [{"rule":r,"passed":p,"detail":d} for r,p,d in rules],
                "confidence": conf,
                "current_wave": "Wave 5  (entering now)",
                "wave_rows": wave_rows,
                "projections": projs,
                "signal": "BUY" if sign == 1 else "SELL",
                "signal_reason": f"EW Wave 5 entry  ({direction}  5-wave impulse)",
                "sl":     sl_price,
                "target": tgt_price,
                "after":  "Expect ABC correction once Wave 5 completes",
                "current_price": cp,
                "bias": "BUY" if sign == 1 else "SELL",
            }

    # ── Try ABC correction (needs 4 pivot points) ─────────────────
    if len(pts) >= 4:
        for st0, direction, sign in [("H","CORRECTIVE_DOWN",-1), ("L","CORRECTIVE_UP",1)]:
            s = pts[-4:]
            if s[0]["t"] != st0: continue
            P  = [p["px"] for p in s]
            A  = sign * (P[0] - P[1])
            B  = sign * (P[2] - P[1])
            C  = sign * (P[2] - P[3])
            if A <= 0: continue
            bret = abs(B) / A; cpct = C / A
            rules = [
                ("B retraces A (20–100%)", 0.2 < bret < 1.0,
                 f"B = {bret*100:.1f}% of A  (ideal 38.2–78.6%)"),
                ("C moves in A direction", C > 0,
                 f"C = {C:.1f} pts  (same direction as A)"),
            ]
            fib_q = (_fq(bret,[0.382,0.5,0.618,0.786]) +
                     _fq(cpct,[0.618,1.0,1.272,1.618])) / 2
            conf  = min(100, int(fib_q * 100))

            wave_rows = [
                ("A","Impulse", "DOWN" if sign==-1 else "UP", P[0], P[1],
                 round(abs(P[1]-P[0]),2), "First leg of ABC correction", "#fc8181"),
                ("B","Correction","UP" if sign==-1 else "DOWN", P[1], P[2],
                 round(abs(P[2]-P[1]),2),
                 f"{bret*100:.1f}% retrace of A  (ideal 38.2–78.6%)", "#48bb78"),
                ("C","Impulse (forming)","DOWN" if sign==-1 else "UP", P[2], P[3],
                 round(abs(P[3]-P[2]),2),
                 f"Currently {cpct*100:.1f}% of A  (ideal 61.8–161.8%)", "#fc8181"),
            ]
            c_eq_a = round(P[2] - sign * A, 2)
            projs  = {
                "C = 61.8% A": round(P[2] - sign*A*0.618, 2),
                "C = 100% A":  c_eq_a,
                "C = 127.2% A":round(P[2] - sign*A*1.272, 2),
            }
            sl_price  = round(P[2] + sign * A * 0.1, 2)
            tgt_price = c_eq_a

            return {
                "ok": True,
                "label": f"ABC Correction ({'Bearish' if sign==-1 else 'Bullish'})",
                "direction": direction,
                "wave_type": "ABC Correction",
                "waves_plot": [{"label":lb,"start":s[i],"end":s[i+1]}
                               for i,lb in enumerate(["A","B","C"])],
                "rules":  [{"rule":r,"passed":p,"detail":d} for r,p,d in rules],
                "confidence": conf,
                "current_wave": "Wave C  (forming now)",
                "wave_rows": wave_rows,
                "projections": projs,
                "signal": "SELL" if sign == -1 else "BUY",
                "signal_reason": f"EW Wave C entry  ({'Bearish' if sign==-1 else 'Bullish'} ABC)",
                "sl":     sl_price,
                "target": tgt_price,
                "after":  "Expect new impulse wave after ABC completes",
                "current_price": cp,
                "bias": "SELL" if sign == -1 else "BUY",
            }

    return {**EMPTY, "ok": True, "label": "Pivots forming",
            "current_price": cp, "after": "More candles needed for pattern"}

# ════════════════════════════════════════════════════════════════════
# SL / TARGET
# ════════════════════════════════════════════════════════════════════
def calc_sl_tgt(entry, tt, row, cfg):
    atr  = float(row.get("atr", entry*0.01) or entry*0.01)
    if np.isnan(atr) or atr == 0: atr = entry * 0.01
    sl_t = cfg.get("sl_type", "Custom Points")
    tg_t = cfg.get("target_type", "Custom Points")
    slp  = float(cfg.get("sl_points", 10))
    tgp  = float(cfg.get("target_points", 20))
    sign = 1 if tt == "buy" else -1
    sl   = (entry - sign*atr*float(cfg.get("atr_sl_mult", 1.5))
            if sl_t == "ATR Based" else entry - sign * slp)
    sld  = abs(entry - sl)
    if tg_t == "ATR Based Target": tgt = entry + sign*atr*float(cfg.get("atr_tgt_mult", 3.0))
    elif tg_t == "Risk-Reward":    tgt = entry + sign * sld * float(cfg.get("rr_ratio", 2.0))
    else:                          tgt = entry + sign * tgp
    return round(sl, 4), round(tgt, 4)

# ════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════
def run_backtest(df, cfg):
    if df is None or df.empty: return [], []
    strat = cfg.get("strategy", "EMA Crossover")
    fast  = int(cfg.get("fast_ema", 9)); slow = int(cfg.get("slow_ema", 15))
    qty   = int(cfg.get("quantity", 1))
    sl_t  = cfg.get("sl_type", "Custom Points")
    tg_t  = cfg.get("target_type", "Custom Points")
    df    = add_ind(df.copy(), fast, slow)
    trades=[]; viol=[]; pos=None; pending=None; tnum=0

    for i in range(1, len(df)):
        row = df.iloc[i]
        if pos:
            sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
            if sl_t == "Trailing SL":
                tr = float(cfg.get("sl_points", 10))
                ns = row["high"]-tr if pt=="buy" else row["low"]+tr
                if (pt=="buy" and ns>sl) or (pt=="sell" and ns<sl): sl=pos["sl"]=ns
            if tg_t == "Trailing Target":
                tp = float(cfg.get("target_points", 20))
                nt = row["high"]+tp if pt=="buy" else row["low"]-tp
                if (pt=="buy" and nt>pos["tgt"]) or (pt=="sell" and nt<pos["tgt"]):
                    pos["tgt"] = nt
            ema_ex = (sl_t in ("Reverse EMA Cross",) or tg_t=="EMA Cross") and ema_rev(df,i,pt)
            ep=None; er=None; vd=False
            if pt == "buy":
                if row["low"] <= sl:
                    ep=sl; er=f"SL Low≤{sl:.2f}"; vd = tg_t!="Trailing Target" and row["high"]>=tgt
                elif tg_t!="Trailing Target" and row["high"]>=tgt: ep=tgt; er=f"Tgt Hi≥{tgt:.2f}"
                elif ema_ex: ep=row["open"]; er="EMA Rev exit"
            else:
                if row["high"] >= sl:
                    ep=sl; er=f"SL Hi≥{sl:.2f}"; vd = tg_t!="Trailing Target" and row["low"]<=tgt
                elif tg_t!="Trailing Target" and row["low"]<=tgt: ep=tgt; er=f"Tgt Lo≤{tgt:.2f}"
                elif ema_ex: ep=row["open"]; er="EMA Rev exit"
            if ep is None and i == len(df)-1: ep=row["close"]; er="End of data"
            if ep is not None:
                pnl = round(((ep-pos["e"]) if pt=="buy" else (pos["e"]-ep))*qty, 2)
                t = {"#":tnum,"Type":pt.upper(),"Entry":fdt(pos["et"]),"Exit":fdt(row.name),
                     "Entry Price":round(pos["e"],4),"Exit Price":round(ep,4),
                     "SL":round(pos["si"],4),"Target":round(pos["ti"],4),
                     "Bar High":round(row["high"],4),"Bar Low":round(row["low"],4),
                     "Entry Reason":pos["r"],"Exit Reason":er,"PnL":pnl,"Qty":qty,"Viol":vd}
                trades.append(t); viol.append(t) if vd else None; pos=None
        if pos is None and pending:
            sg,rsn=pending; pending=None; ep=float(row["open"])
            sl,tgt=calc_sl_tgt(ep,sg,row,cfg); tnum+=1
            pos={"type":sg,"e":ep,"et":row.name,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"r":rsn}
        if pos is None:
            if strat == "EMA Crossover":
                sg,rsn=ema_sig(df,i,cfg)
                if sg: pending=(sg,rsn)
            elif strat == "Simple Buy":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"buy",row,cfg); tnum+=1
                pos={"type":"buy","e":ep,"et":row.name,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"r":"Simple Buy"}
            elif strat == "Simple Sell":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"sell",row,cfg); tnum+=1
                pos={"type":"sell","e":ep,"et":row.name,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"r":"Simple Sell"}
            elif strat == "Elliott Wave" and i >= 20:
                ew = detect_ew(df.iloc[:i+1])
                if ew.get("signal") and ew.get("confidence",0) >= 50:
                    sg = "buy" if ew["signal"]=="BUY" else "sell"
                    pending = (sg, ew.get("signal_reason","EW signal"))
    return trades, viol

# ════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ════════════════════════════════════════════════════════════════════
def run_opt(df, base_cfg, grid, metric="Accuracy%", min_tr=5, cb=None):
    results=[]; keys=list(grid.keys()); combos=list(itertools.product(*grid.values()))
    total=len(combos)
    for idx, combo in enumerate(combos):
        cfg={**base_cfg,"sl_type":"Custom Points","target_type":"Custom Points"}
        for k,v in zip(keys,combo): cfg[k]=v
        try:
            f=int(cfg.get("fast_ema",9)); s=int(cfg.get("slow_ema",15))
            if f>=s:
                if cb: cb(idx+1,total); continue
            df_i=add_ind(df.copy(),f,s)
            trades,_=run_backtest(df_i,cfg)
            if len(trades)<min_tr:
                if cb: cb(idx+1,total); continue
            tdf=pd.DataFrame(trades); wins=len(tdf[tdf["PnL"]>0]); tot=len(tdf)
            acc=wins/tot*100; pnl=tdf["PnL"].sum()
            aw=tdf[tdf["PnL"]>0]["PnL"].mean() if wins else 0
            al=tdf[tdf["PnL"]<=0]["PnL"].mean() if tot-wins else 0
            rr=abs(aw/al) if al and al!=0 else 0
            sc=acc*0.4+min(rr,5)*4*0.3+(20 if pnl>0 else 0)*0.3
            row_d={"Trades":tot,"Wins":wins,"Losses":tot-wins,
                   "Accuracy%":round(acc,1),"TotalPnL":round(pnl,2),
                   "AvgWin":round(aw,2),"AvgLoss":round(al,2),
                   "RR":round(rr,2),"Score":round(sc,2)}
            for k2,v2 in zip(keys,combo): row_d[k2]=v2
            results.append(row_d)
        except Exception: pass
        if cb: cb(idx+1,total)
    results.sort(key=lambda x:x.get(metric,0),reverse=True)
    return results[:25]

# ════════════════════════════════════════════════════════════════════
# DHAN BROKER
# ════════════════════════════════════════════════════════════════════
def init_dhan(cid, tok):
    try:
        from dhanhq import dhanhq; return dhanhq(cid, tok)
    except ImportError: st.warning("pip install dhanhq"); return None
    except Exception as e: st.error(f"Dhan: {e}"); return None

def get_ip():
    try: return requests.get("https://api.ipify.org?format=json",timeout=4).json().get("ip","?")
    except Exception:
        try: return requests.get("https://ifconfig.me/ip",timeout=4).text.strip()
        except: return "N/A"

def place_order(dhan, cfg, sig, ltp, is_exit=False):
    if not dhan: return {"error":"Not connected"}
    try:
        if cfg.get("options_trading", False):
            ot  = cfg.get("opt_exit_type" if is_exit else "opt_entry_type","MARKET")
            sid = cfg.get("ce_id") if sig=="buy" else cfg.get("pe_id")
            return dhan.place_order(
                transactionType="SELL" if is_exit else "BUY",
                exchangeSegment=cfg.get("fno_exc","NSE_FNO"),
                productType="INTRADAY", orderType=ot, validity="DAY",
                securityId=str(sid), quantity=int(cfg.get("opts_qty",65)),
                price=round(ltp,2) if ot=="LIMIT" else 0, triggerPrice=0)
        ot  = cfg.get("exit_order" if is_exit else "entry_order","MARKET")
        txn = ("SELL" if is_exit else "BUY") if sig=="buy" else ("BUY" if is_exit else "SELL")
        return dhan.place_order(
            security_id=str(cfg.get("sec_id","1594")),
            exchange_segment=cfg.get("exchange","NSE"),
            transaction_type=txn, quantity=int(cfg.get("dhan_qty",1)),
            order_type=ot, product_type=cfg.get("product","INTRADAY"),
            price=round(ltp,2) if ot=="LIMIT" else 0)
    except Exception as e: return {"error":str(e)}

# ════════════════════════════════════════════════════════════════════
# LIVE THREAD  — pure Python, zero Streamlit, zero session_state
# ════════════════════════════════════════════════════════════════════
def live_thread(cfg: dict, symbol: str, dhan_client):
    """
    Runs independently of Streamlit. Writes only to _STORE (singleton).
    Never calls st.* or session_state — those crash background threads.
    """
    _log(f"▶ START  {symbol}  TF={cfg['timeframe']}  Strategy={cfg['strategy']}")
    tf_min   = TF_MIN.get(cfg["timeframe"], 5)
    fast     = int(cfg.get("fast_ema", 9))
    slow     = int(cfg.get("slow_ema", 15))
    strat    = cfg.get("strategy", "EMA Crossover")
    sl_t     = cfg.get("sl_type", "Custom Points")
    tg_t     = cfg.get("target_type", "Custom Points")
    qty      = int(cfg.get("quantity", 1))
    use_dhan = bool(cfg.get("enable_dhan", False)) and (dhan_client is not None)

    # Register as sole active thread. If user Stop+Starts quickly, new thread
    # overwrites _tid and this (old) thread exits on its next while check.
    my_tid = threading.current_thread().ident
    _set("_tid", my_tid)

    pending      = None
    last_sig_bar = None
    last_bdy     = -1

    # fetch prev_close once
    try:
        _rate_wait()
        raw = yf.Ticker(symbol).history(period="5d",interval="1d",auto_adjust=True,prepost=False)
        df2 = _clean(raw)
        if df2 is not None and len(df2) >= 2:
            _set("prev_close", float(df2["close"].iloc[-2]))
    except Exception as e:
        _log(f"[prev_close] {e}")

    # ── main loop ─────────────────────────────────────────────────
    while _get("running", False) and _get("_tid") == my_tid:
        try:
            just_exited = False             # reset each tick
            df = fetch_live(symbol, cfg["timeframe"], cfg["period"])
            if df is None or df.empty:
                _log("[WARN] No data — retrying in 3s")
                time.sleep(3); continue

            df  = add_ind(df, fast, slow)
            ltp = float(df["close"].iloc[-1])
            now = datetime.now(IST)
            lr  = df.iloc[-1]

            # store indicators for UI
            def _fv(v): return float(v) if v is not None and not (isinstance(v,float) and np.isnan(v)) else float("nan")
            ef_v  = _fv(lr.get("ef"))
            es_v  = _fv(lr.get("es"))
            atr_v = _fv(lr.get("atr"))
            ef_p  = _fv(df.iloc[-2].get("ef") if len(df)>=2 else None)
            es_p  = _fv(df.iloc[-2].get("es") if len(df)>=2 else None)
            _set("ind", {
                "ef": ef_v, "es": es_v, "atr": atr_v,
                "ef_p": ef_p, "es_p": es_p,
                "fast": fast, "slow": slow, "ltp": ltp,
                "bt": df.index[-1],
                "bo": float(lr["open"]), "bh": float(lr["high"]), "bl": float(lr["low"]),
            })
            _set("df",      df)
            _set("ltp",     ltp)
            _set("last_ts", now)
            if len(df) >= 20:
                _set("ew", detect_ew(df))

            pos = _get("pos")

            # ── EXIT CHECK (vs LTP, every tick) ──────────────────
            if pos:
                sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
                # Trailing SL
                if sl_t == "Trailing SL":
                    tr = float(cfg.get("sl_points", 10))
                    ns = ltp - tr if pt=="buy" else ltp + tr
                    if (pt=="buy" and ns>sl) or (pt=="sell" and ns<sl):
                        pos = dict(pos, sl=ns); _set("pos", pos); sl = ns
                # Trailing Target (display only)
                if tg_t == "Trailing Target":
                    tp = float(cfg.get("target_points", 20))
                    nt = ltp + tp if pt=="buy" else ltp - tp
                    if (pt=="buy" and nt>tgt) or (pt=="sell" and nt<tgt):
                        pos = dict(pos, tgt=nt); _set("pos", pos)

                # increment ticks_held so we know how long position has been open
                pos = dict(pos, ticks_held=pos.get("ticks_held",0)+1)
                _set("pos", pos)

                ep=None; er=None
                # Only check SL/Target after at least 1 full tick has passed.
                # This prevents false exits when yfinance returns a stale/updated
                # close price on the very first tick after entry.
                if pos["ticks_held"] >= 1:
                    if pt == "buy":
                        if ltp <= sl:                                   ep=sl;  er=f"SL hit ({ltp:.2f}≤{sl:.2f})"
                        elif tg_t!="Trailing Target" and ltp >= tgt:   ep=tgt; er=f"Target hit ({ltp:.2f}≥{tgt:.2f})"
                    else:
                        if ltp >= sl:                                   ep=sl;  er=f"SL hit ({ltp:.2f}≥{sl:.2f})"
                        elif tg_t!="Trailing Target" and ltp <= tgt:   ep=tgt; er=f"Target hit ({ltp:.2f}≤{tgt:.2f})"
                    # EMA cross exit — only for EMA Crossover / Elliott Wave strategies,
                    # NOT for Simple Buy/Sell (those only exit via SL/Target)
                    if ep is None and strat not in ("Simple Buy","Simple Sell"):
                        if (sl_t=="Reverse EMA Cross" or tg_t=="EMA Cross"):
                            if ema_rev(df, len(df)-1, pt): ep=ltp; er="EMA Reverse Cross exit"

                if ep is not None:
                    pnl = round(((ep-pos["e"]) if pt=="buy" else (pos["e"]-ep))*qty, 2)
                    t   = {"#": len(_get("trades",[]))+1, "Type": pt.upper(),
                           "Entry": fdt(pos["et"]),
                           "Exit":  now.strftime("%Y-%m-%d %H:%M:%S"),
                           "Entry Price": round(pos["e"],4), "Exit Price": round(ep,4),
                           "SL": round(pos["si"],4), "Target": round(pos["ti"],4),
                           "Entry Reason": pos["r"], "Exit Reason": er,
                           "PnL": pnl, "Qty": qty, "Source": "Live"}
                    _push("trades", t)
                    _set("pos", None)
                    pending = None          # ← clear any pending signal so it
                                            #   doesn't fire in the same tick
                    just_exited = True      # ← skip entry this tick
                    if use_dhan:
                        resp = place_order(dhan_client, cfg, pt, ltp, is_exit=True)
                        _log(f"EXIT order: {resp}")
                    _log(f"✖ EXIT {pt.upper()} @{ep:.2f} | {er} | PnL: {pnl:+.2f}")

            # ── ENTRY LOGIC ───────────────────────────────────────
            # just_exited guard: never re-enter in the same tick as an exit.
            # This prevents Simple Buy/Sell from immediately re-opening after
            # SL/Target, and prevents pending EMA signals from firing the same
            # tick a position closes.
            if _get("pos") is None and not just_exited:
                cd_ok = True
                if cfg.get("cooldown_enabled", True):
                    trd = _get("trades", [])
                    if trd:
                        try:
                            le = datetime.strptime(
                                trd[-1].get("Exit","2000-01-01 00:00:00"),
                                "%Y-%m-%d %H:%M:%S")
                            le = IST.localize(le)
                            if (now - le).total_seconds() < int(cfg.get("cooldown_s",5)):
                                cd_ok = False
                        except Exception: pass

                # Simple Buy/Sell → IMMEDIATE on first tick
                if strat == "Simple Buy" and cd_ok:
                    sl, tgt = calc_sl_tgt(ltp, "buy", lr, cfg)
                    _set("pos", {"type":"buy","e":ltp,"et":now,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"r":"Simple Buy"})
                    if use_dhan: _log(f"ENTRY: {place_order(dhan_client,cfg,'buy',ltp)}")
                    _log(f"▲ BUY @{ltp:.2f}  SL:{sl:.2f}  Tgt:{tgt:.2f}")

                elif strat == "Simple Sell" and cd_ok:
                    sl, tgt = calc_sl_tgt(ltp, "sell", lr, cfg)
                    _set("pos", {"type":"sell","e":ltp,"et":now,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"r":"Simple Sell"})
                    if use_dhan: _log(f"ENTRY: {place_order(dhan_client,cfg,'sell',ltp)}")
                    _log(f"▼ SELL @{ltp:.2f}  SL:{sl:.2f}  Tgt:{tgt:.2f}")

                # Pending from previous candle
                elif pending and cd_ok:
                    sg, rsn = pending; pending = None
                    ep = float(df["open"].iloc[-1])
                    sl, tgt = calc_sl_tgt(ep, sg, lr, cfg)
                    _set("pos", {"type":sg,"e":ep,"et":now,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"r":rsn})
                    if use_dhan: _log(f"ENTRY: {place_order(dhan_client,cfg,sg,ep)}")
                    _log(f"▲ ENTRY {sg.upper()} @{ep:.2f}  SL:{sl:.2f}  Tgt:{tgt:.2f}")

                else:
                    tm     = now.hour*60 + now.minute
                    at_bdy = (tm % tf_min == 0) and (tm != last_bdy)
                    if at_bdy and cd_ok:
                        last_bdy = tm
                        if strat == "EMA Crossover":
                            sg, rsn = ema_sig(df, len(df)-1, cfg)
                            if sg and df.index[-1] != last_sig_bar:
                                last_sig_bar = df.index[-1]
                                pending = (sg, rsn)
                                _log(f"◆ EMA SIGNAL {sg.upper()} → entering next candle open")
                        elif strat == "Elliott Wave":
                            ew   = _get("ew", {})
                            sig2 = ew.get("signal", "")
                            conf = ew.get("confidence", 0)
                            if sig2 and conf >= 50 and cd_ok:
                                sg = "buy" if sig2=="BUY" else "sell"
                                pending = (sg, ew.get("signal_reason","EW signal"))
                                _log(f"◆ EW SIGNAL {sg.upper()} conf={conf}% → entering next candle")

            time.sleep(1.5)

        except Exception as e:
            _log(f"[ERR] {e}")
            time.sleep(2)   # short pause, keep thread alive

    _log("■ STOPPED.")

# ════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════
def make_chart(df, trades=None, title="", fast=9, slow=15, pos=None, h=500):
    if df is None or df.empty: return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.78,0.22])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#48bb78", decreasing_line_color="#fc8181",
        increasing_fillcolor="#48bb78", decreasing_fillcolor="#fc8181"), row=1, col=1)
    if "ef" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ef"], name=f"EMA({fast})",
            line=dict(color="#f6ad55", width=1.5)), row=1, col=1)
    if "es" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["es"], name=f"EMA({slow})",
            line=dict(color="#76e4f7", width=1.5)), row=1, col=1)
    if "volume" in df.columns:
        vc = ["#48bb78" if c>=o else "#fc8181" for c,o in zip(df["close"],df["open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Vol",
            marker_color=vc, opacity=0.45), row=2, col=1)
    for t in (trades or [])[:60]:
        try:
            et = pd.to_datetime(t.get("Entry","")); xt = pd.to_datetime(t.get("Exit",""))
            ep_v = t.get("Entry Price",0); xp_v = t.get("Exit Price",0)
            sl_v = t.get("SL",0);          tg_v = t.get("Target",0)
            c   = "#48bb78" if t["Type"]=="BUY" else "#fc8181"
            sym = "triangle-up" if t["Type"]=="BUY" else "triangle-down"
            pc  = "#48bb78" if t.get("PnL",0)>=0 else "#fc8181"
            fig.add_trace(go.Scatter(x=[et],y=[ep_v],mode="markers+text",
                marker=dict(symbol=sym,size=11,color=c),showlegend=False,
                text=[f"{ep_v:.0f}"],textposition="top center",
                textfont=dict(size=7,color=c)),row=1,col=1)
            fig.add_trace(go.Scatter(x=[xt],y=[xp_v],mode="markers",
                marker=dict(symbol="x",size=8,color=pc),showlegend=False),row=1,col=1)
            fig.add_shape(type="line",x0=et,x1=xt,y0=sl_v,y1=sl_v,
                line=dict(color="#fc8181",width=1,dash="dot"),row=1,col=1)
            fig.add_shape(type="line",x0=et,x1=xt,y0=tg_v,y1=tg_v,
                line=dict(color="#48bb78",width=1,dash="dot"),row=1,col=1)
        except Exception: pass
    if pos:
        try:
            c   = "#48bb78" if pos["type"]=="buy" else "#fc8181"
            sym = "triangle-up" if pos["type"]=="buy" else "triangle-down"
            fig.add_trace(go.Scatter(x=[pos["et"]],y=[pos["e"]],mode="markers+text",
                marker=dict(symbol=sym,size=16,color=c,line=dict(color="white",width=2)),
                text=[f"ENTRY {pos['e']:.2f}"],textposition="top center"),row=1,col=1)
            fig.add_hline(y=pos["sl"],line=dict(color="#fc8181",width=2,dash="dot"),
                annotation_text=f"SL {pos['sl']:.2f}",
                annotation_font=dict(color="#fc8181"),row=1,col=1)
            fig.add_hline(y=pos["tgt"],line=dict(color="#48bb78",width=2,dash="dot"),
                annotation_text=f"Tgt {pos['tgt']:.2f}",
                annotation_font=dict(color="#48bb78"),row=1,col=1)
        except Exception: pass
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0",size=11),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0,r=0,t=36,b=0),height=h,xaxis_rangeslider_visible=False,
        xaxis2=dict(showgrid=True,gridcolor="#1a1f2e"),
        yaxis=dict(showgrid=True,gridcolor="#1a1f2e"),
        yaxis2=dict(showgrid=True,gridcolor="#1a1f2e"))
    if title: fig.update_layout(title=dict(text=title,font=dict(size=13,color="#90cdf4")))
    return fig

def make_ew_chart(df, ew, fast, slow):
    fig = make_chart(df.tail(120), fast=fast, slow=slow, h=400, title="Elliott Wave Chart")
    wc  = {"1":"#f6ad55","2":"#fc8181","3":"#48bb78","4":"#f6ad55","5":"#76e4f7",
           "A":"#fc8181","B":"#48bb78","C":"#fc8181"}
    for w in ew.get("waves_plot", []):
        s=w["start"]; e=w["end"]; c=wc.get(w["label"],"#a0aec0")
        fig.add_trace(go.Scatter(x=[s["dt"],e["dt"]],y=[s["px"],e["px"]],
            mode="lines+markers+text",line=dict(color=c,width=2.5),
            marker=dict(size=8,color=c),text=["",f"W{w['label']}"],
            textposition="top center",textfont=dict(color=c,size=12),
            name=f"W{w['label']}"),row=1,col=1)
    return fig

def pnl_chart(trades):
    pnls = [t.get("PnL",0) for t in trades]; cum = [0]+list(np.cumsum(pnls))
    c    = "#48bb78" if cum[-1]>=0 else "#fc8181"
    fill = "rgba(72,187,120,0.1)" if c=="#48bb78" else "rgba(252,129,129,0.1)"
    fig  = go.Figure(go.Scatter(x=list(range(len(cum))),y=cum,mode="lines+markers",
        fill="tozeroy",fillcolor=fill,line=dict(color=c,width=2),name="P&L"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",height=220,
        margin=dict(l=0,r=0,t=20,b=0),font=dict(color="#e2e8f0"))
    return fig

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
def sidebar():
    cfg = {}
    with st.sidebar:
        st.markdown("## 📈 Smart Investing"); st.markdown("---")
        tn  = st.selectbox("🎯 Ticker", list(TICKERS.keys()), key="s_tn")
        sym = (st.text_input("Symbol","RELIANCE.NS",key="s_sym").strip()
               if tn=="Custom" else TICKERS[tn])
        cfg.update(ticker_name=tn, symbol=sym)
        tf = st.selectbox("⏱ Interval", list(TF_PERIODS.keys()), index=2, key="s_tf")
        pr = TF_PERIODS[tf]
        per= st.selectbox("Period", pr, index=min(1,len(pr)-1), key="s_per")
        cfg.update(timeframe=tf, period=per)
        strat = st.selectbox("🧠 Strategy",
            ["EMA Crossover","Simple Buy","Simple Sell","Elliott Wave"], key="s_str")
        cfg["strategy"] = strat
        if strat == "EMA Crossover":
            c1,c2 = st.columns(2)
            fe = c1.number_input("Fast EMA",1,500,int(st.session_state.get("_aFE",9)),  key="s_fe")
            se = c2.number_input("Slow EMA",1,500,int(st.session_state.get("_aSE",15)), key="s_se")
            cfg.update(fast_ema=fe,slow_ema=se)
            uang = st.checkbox("Angle Filter",False,key="s_ang")
            mang = st.number_input("Min Angle°",0.0,90.0,0.0,0.5,key="s_mang") if uang else 0.0
            cfg.update(use_angle=uang,min_angle=mang)
            ct = st.selectbox("Crossover",["Simple","Custom Candle","ATR Candle"],key="s_ct")
            cfg["crossover_type"]=ct
            if ct=="Custom Candle":
                cfg["custom_candle"]=st.number_input("Min Body",0.0,value=10.0,key="s_cb")
        else:
            cfg.update(fast_ema=9,slow_ema=15,use_angle=False,min_angle=0.0,
                       crossover_type="Simple",custom_candle=10.0)
        cfg["quantity"] = st.number_input("📦 Qty",1,1_000_000,1,key="s_qty")
        sl_t = st.selectbox("🛡 SL",
            ["Custom Points","ATR Based","Trailing SL","Reverse EMA Cross","Risk-Reward"],key="s_slt")
        cfg["sl_type"] = sl_t
        slp = float(st.session_state.get("_aSL",10.0))
        if sl_t == "ATR Based":
            cfg["atr_sl_mult"] = st.number_input("ATR Mult SL",0.1,10.0,1.5,0.1,key="s_asm")
            cfg["sl_points"]   = slp
        else:
            cfg["sl_points"]   = st.number_input("SL Points",0.1,value=slp,step=0.5,key="s_slp")
            cfg["atr_sl_mult"] = 1.5
        tg_t = st.selectbox("🎯 Target",
            ["Custom Points","ATR Based Target","Trailing Target","EMA Cross","Risk-Reward"],key="s_tgt")
        cfg["target_type"] = tg_t
        tgp = float(st.session_state.get("_aTG",20.0))
        if tg_t == "ATR Based Target":
            cfg["atr_tgt_mult"]  = st.number_input("ATR Mult Tgt",0.1,20.0,3.0,0.1,key="s_atm")
            cfg["target_points"] = tgp
        elif tg_t == "Risk-Reward":
            cfg["rr_ratio"]      = st.number_input("R:R",0.1,20.0,
                float(st.session_state.get("_aRR",2.0)),0.1,key="s_rr")
            cfg["target_points"] = tgp; cfg["atr_tgt_mult"] = 3.0
        elif tg_t == "Trailing Target":
            cfg["target_points"] = st.number_input("Trail Dist",0.1,value=tgp,step=0.5,key="s_tgp")
            cfg["atr_tgt_mult"]  = 3.0; st.caption("Display only")
        else:
            cfg["target_points"] = st.number_input("Target Pts",0.1,value=tgp,step=0.5,key="s_tgp2")
            cfg["atr_tgt_mult"]  = 3.0
        cfg.setdefault("rr_ratio",2.0)
        cd  = st.checkbox("🔄 Cooldown",True,key="s_cd")
        cds = st.number_input("Cooldown s",1,3600,5,key="s_cds") if cd else 5
        cfg.update(cooldown_enabled=cd,cooldown_s=cds)
        en = st.checkbox("🏦 Enable Dhan",False,key="s_dhan"); cfg["enable_dhan"]=en
        if en:
            cid=st.text_input("Client ID",key="s_cid")
            tok=st.text_input("Token",key="s_tok",type="password")
            cfg.update(dhan_cid=cid,dhan_tok=tok)
            if st.button("Connect & Show IP",use_container_width=True,key="s_conn"):
                cl=init_dhan(cid,tok)
                if cl:
                    st.session_state.dhan_client=cl
                    ip=get_ip()
                    st.info(f"Your IP: **{ip}** — whitelist at Dhan Console → Profile → API → IP Whitelist")
                    st.success("Connected!")
                else: st.error("Failed")
            opts=st.checkbox("Options",False,key="s_opts"); cfg["options_trading"]=opts
            if opts:
                cfg["fno_exc"]=st.selectbox("FNO",["NSE_FNO","BSE_FNO"],key="s_fno")
                cfg["ce_id"]=st.text_input("CE ID",key="s_ceid"); cfg["pe_id"]=st.text_input("PE ID",key="s_peid")
                cfg["opts_qty"]=st.number_input("Opts Qty",1,value=65,key="s_oq")
                cfg["opt_entry_type"]=st.selectbox("Entry",["MARKET","LIMIT"],key="s_oent")
                cfg["opt_exit_type"] =st.selectbox("Exit", ["MARKET","LIMIT"],key="s_oext")
            else:
                cfg["product"]     =st.selectbox("Product", ["INTRADAY","DELIVERY"],key="s_prod")
                cfg["exchange"]    =st.selectbox("Exchange",["NSE","BSE"],           key="s_exc")
                cfg["sec_id"]      =st.text_input("Security ID","1594",              key="s_sid")
                cfg["dhan_qty"]    =st.number_input("Order Qty",1,value=1,           key="s_dqty")
                cfg["entry_order"] =st.selectbox("Entry Order",["LIMIT","MARKET"],   key="s_eord")
                cfg["exit_order"]  =st.selectbox("Exit Order", ["MARKET","LIMIT"],   key="s_xord")
    return cfg

# ════════════════════════════════════════════════════════════════════
# SHARED WIDGETS
# ════════════════════════════════════════════════════════════════════
def ltp_banner(cfg):
    ltp  = _get("ltp")  or 0.0
    prev = _get("prev_close") or ltp
    chg  = ltp - prev; pct = chg/prev*100 if prev else 0
    arrow= "▲" if chg>=0 else "▼"; cls="ltpup" if chg>=0 else "ltpdn"
    ts   = datetime.now(IST).strftime("%H:%M:%S IST")
    st.markdown(
        f'<div class="ltpbar">'
        f'<div style="color:#90cdf4;font-size:11px">📈 {cfg.get("ticker_name","—")} · {cfg.get("symbol","")}</div>'
        f'<div class="ltpval">₹ {ltp:,.2f}</div>'
        f'<div class="{cls}">{arrow} {abs(chg):.2f} ({abs(pct):.2f}%)</div>'
        f'<div class="ltpmeta">{cfg.get("timeframe","")} · {cfg.get("period","")} · {cfg.get("strategy","")} | {ts}</div>'
        f'</div>', unsafe_allow_html=True)

def cfg_box(cfg):
    st.markdown(
        f'<div class="cfgb">'
        f'<b>Ticker:</b> {cfg.get("ticker_name","—")} ({cfg.get("symbol","—")}) | '
        f'<b>TF:</b> {cfg.get("timeframe","—")}/{cfg.get("period","—")} | '
        f'<b>Strategy:</b> {cfg.get("strategy","—")} | <b>Qty:</b> {cfg.get("quantity",1)}<br>'
        f'<b>Fast EMA:</b> {cfg.get("fast_ema",9)} | <b>Slow EMA:</b> {cfg.get("slow_ema",15)}<br>'
        f'<b>SL:</b> {cfg.get("sl_type","—")} ({cfg.get("sl_points",10)} pts) | '
        f'<b>Target:</b> {cfg.get("target_type","—")} ({cfg.get("target_points",20)} pts)'
        f'</div>', unsafe_allow_html=True)

def style_df(df):
    if df.empty: return df.style
    def rc(row):
        pnl=row.get("PnL",0)
        c="rgba(72,187,120,0.10)" if pnl>0 else ("rgba(252,129,129,0.10)" if pnl<0 else "")
        return [f"background-color:{c}"]*len(row)
    styled=df.style.apply(rc,axis=1)
    if "PnL" in df.columns:
        styled=styled.map(lambda v:(
            f"color:{'#48bb78' if v>0 else '#fc8181'};font-weight:700"
            if isinstance(v,(int,float)) else ""),subset=["PnL"])
    if "Viol" in df.columns:
        styled=styled.map(lambda v:(
            "background-color:#2d1515;color:#fc8181;font-weight:700"
            if v is True else ""),subset=["Viol"])
    return styled

def trade_stats(trades):
    if not trades: return
    df  = pd.DataFrame(trades)
    w   = df[df["PnL"]>0]; l = df[df["PnL"]<=0]
    tot = df["PnL"].sum(); acc = len(w)/len(df)*100 if len(df) else 0
    aw  = w["PnL"].mean() if len(w) else 0; al = l["PnL"].mean() if len(l) else 0
    cols= st.columns(6)
    for col,lbl,val,clr in [
        (cols[0],"Trades",   len(df),          "#e2e8f0"),
        (cols[1],"Winners",  len(w),            "#48bb78"),
        (cols[2],"Losers",   len(l),            "#fc8181"),
        (cols[3],"Accuracy", f"{acc:.1f}%",     "#f6ad55"),
        (cols[4],"Total P&L",f"₹{tot:+.2f}",   "#48bb78" if tot>=0 else "#fc8181"),
        (cols[5],"Avg W/L",  f"₹{aw:.1f}/₹{al:.1f}","#76e4f7"),
    ]:
        col.markdown(
            f'<div class="card"><div style="color:#a0aec0;font-size:11px">{lbl}</div>'
            f'<div style="color:{clr};font-size:16px;font-weight:700;margin-top:2px">{val}</div>'
            f'</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# EW DISPLAY
# ════════════════════════════════════════════════════════════════════
def show_ew_panel(ew, cfg, df=None, fast=9, slow=15):
    if not ew or not ew.get("ok"): return
    label  = ew.get("label","—"); d = ew.get("direction","UNCLEAR")
    wtype  = ew.get("wave_type","—"); cw = ew.get("current_wave","—")
    conf   = ew.get("confidence",0); cp = ew.get("current_price",0)
    sig    = ew.get("signal",""); reason = ew.get("signal_reason","")
    sl_ew  = ew.get("sl"); tgt_ew = ew.get("target")
    projs  = ew.get("projections",{}); after = ew.get("after","")
    rows   = ew.get("wave_rows",[]); rules = ew.get("rules",[])

    dc = {"BULLISH":"#48bb78","BEARISH":"#fc8181",
          "CORRECTIVE_DOWN":"#f6ad55","CORRECTIVE_UP":"#76e4f7",
          "UNCLEAR":"#a0aec0"}.get(d,"#a0aec0")
    cc = "#48bb78" if conf>=70 else "#f6ad55" if conf>=40 else "#fc8181"

    # Top status row
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,lbl,val,clr in [
        (c1,"Pattern",   label,  dc),
        (c2,"Wave Type", wtype,  dc),
        (c3,"Direction", d,      dc),
        (c4,"Now",       cw,     "#e2e8f0"),
        (c5,"Confidence",f"{conf}%",cc),
    ]:
        col.markdown(
            f'<div class="card"><div style="color:#a0aec0;font-size:11px">{lbl}</div>'
            f'<div style="color:{clr};font-size:13px;font-weight:700;margin-top:2px">{val}</div>'
            f'</div>', unsafe_allow_html=True)

    # Automated signal box
    if sig:
        sig_c = "#48bb78" if sig=="BUY" else "#fc8181"
        arr   = "🔼" if sig=="BUY" else "🔽"
        sl_str  = f"₹{sl_ew:,.2f}"  if sl_ew  else "—"
        tgt_str = f"₹{tgt_ew:,.2f}" if tgt_ew else "—"
        rr_str  = "—"
        if sl_ew and tgt_ew and sl_ew != 0:
            rr_val = round(abs(tgt_ew - cp) / abs(cp - sl_ew),2) if cp != sl_ew else 0
            rr_str = f"{rr_val}x"
        strat_sl  = cfg.get("sl_points",10); strat_sl_t = cfg.get("sl_type","—")
        strat_tg  = cfg.get("target_points",20); strat_tg_t= cfg.get("target_type","—")
        st.markdown(
            f'<div class="ewrec">'
            f'<div style="font-size:16px;font-weight:700;color:{sig_c};margin-bottom:10px">'
            f'{arr} AUTO SIGNAL: {sig} &nbsp;·&nbsp; '
            f'<span style="font-size:12px;color:#a0aec0;font-weight:normal">{reason}</span></div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px">'
            f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="color:#a0aec0;font-size:11px">Entry (LTP)</div>'
            f'<div style="color:#e2e8f0;font-size:16px;font-weight:700">₹{cp:,.2f}</div></div>'
            f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="color:#a0aec0;font-size:11px">EW Stop Loss</div>'
            f'<div style="color:#fc8181;font-size:16px;font-weight:700">{sl_str}</div></div>'
            f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="color:#a0aec0;font-size:11px">EW Target</div>'
            f'<div style="color:#48bb78;font-size:16px;font-weight:700">{tgt_str}</div></div>'
            f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="color:#a0aec0;font-size:11px">EW R:R</div>'
            f'<div style="color:#f6ad55;font-size:16px;font-weight:700">{rr_str}</div></div></div>'
            f'<div style="background:#0e1117;border-radius:6px;padding:7px 12px;font-size:12px;color:#a0aec0">'
            f'💡 Your SL: {strat_sl}pts ({strat_sl_t}) | Target: {strat_tg}pts ({strat_tg_t}) — '
            f'adjust sidebar to match EW levels if desired'
            f'</div></div>', unsafe_allow_html=True)
        st.success(f"📊 EW Confidence: {conf}%  {'(High confidence — good setup!)' if conf>=70 else '(Medium — use with EMA confirmation)' if conf>=40 else '(Low — wait for better setup)'}")

    # Wave breakdown table
    if rows:
        st.markdown("##### 📊 Wave-by-Wave Breakdown")
        for lbl,wtp,wdir,spx,epx,sz,fib,wclr in rows:
            darr = "🔼" if "UP" in wdir else "🔽"
            ibg  = "#1a3a1a" if "Impulse" in wtp else "#3a1a1a"
            ibdg = "⚡ Impulse" if "Impulse" in wtp else "↩ Correction"
            st.markdown(
                f'<div class="wrow">'
                f'<span style="color:{wclr};font-weight:700;font-size:14px">Wave {lbl}</span> &nbsp;'
                f'<span style="background:{ibg};color:{wclr};padding:2px 8px;border-radius:4px;font-size:11px">'
                f'{ibdg} {darr}</span> &nbsp;'
                f'<b style="color:#e2e8f0">{spx:.2f} → {epx:.2f}</b> &nbsp;'
                f'<span style="color:#a0aec0;font-size:12px">({sz:.1f} pts)</span><br>'
                f'<span style="color:#718096;font-size:11px">📐 {fib}</span>'
                f'</div>', unsafe_allow_html=True)

    # Projections
    if projs:
        st.markdown("##### 🎯 Wave Projections (Next Targets)")
        tcols = st.columns(len(projs))
        for i,(n,v) in enumerate(projs.items()):
            cl = "#48bb78" if v>cp else "#fc8181"
            tcols[i].markdown(
                f'<div class="card" style="text-align:center">'
                f'<div style="color:#a0aec0;font-size:11px">{n}</div>'
                f'<div style="color:{cl};font-size:14px;font-weight:700">₹{v:,.2f}</div>'
                f'</div>', unsafe_allow_html=True)

    # Rules check
    if rules:
        st.markdown("##### ✅ EW Rules Checklist")
        bar = "█"*int(conf/5) + "░"*(20-int(conf/5))
        st.markdown(
            f'<div class="card">'
            f'Confidence: <span style="color:{cc};font-size:18px;font-weight:700">{conf}%</span>'
            f' &nbsp; <span style="color:{cc};letter-spacing:1px">{bar}</span><br>'
            f'<span style="color:#718096;font-size:11px">'
            f'≥70% = textbook quality  |  40–70% = acceptable  |  &lt;40% = wait for better setup</span>'
            f'</div>', unsafe_allow_html=True)
        rcols = st.columns(2)
        for i, r in enumerate(rules):
            icon = "✅" if r["passed"] else "❌"; clr2="#48bb78" if r["passed"] else "#fc8181"
            rcols[i%2].markdown(
                f'<div style="background:#0e1117;border-radius:6px;padding:6px 10px;margin:3px;font-size:11px">'
                f'{icon} <b style="color:{clr2}">{r["rule"]}</b><br>'
                f'<span style="color:#718096">{r["detail"]}</span>'
                f'</div>', unsafe_allow_html=True)

    if after: st.info(f"💡 {after}")

    # EW chart
    if df is not None and not df.empty and len(ew.get("waves_plot",[])) > 0:
        st.plotly_chart(make_ew_chart(df, ew, fast, slow),
                        use_container_width=True, key="ew_chart")

# ════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ════════════════════════════════════════════════════════════════════
def tab_backtest(cfg):
    ltp_banner(cfg); st.markdown("## 🔬 Backtesting")
    st.info("EMA: signal on candle N → entry at N+1 open  |  Buy: Low vs SL first (conservative)")
    if st.button("▶ Run Backtest", type="primary", key="btn_bt"):
        with st.spinner(f"Fetching {cfg.get('symbol')} …"):
            df = fetch_ui(cfg["symbol"], cfg["timeframe"], cfg["period"])
        if df is None or df.empty: st.error("No data — check symbol/interval."); return
        df = add_ind(df, int(cfg.get("fast_ema",9)), int(cfg.get("slow_ema",15)))
        with st.spinner("Running…"): trades, viol = run_backtest(df, cfg)
        st.session_state.bt_trades = trades
        st.session_state.bt_viol   = viol
        st.session_state.bt_df     = df
        st.session_state.bt_ran    = True
    if not st.session_state.bt_ran:
        st.markdown('<div style="color:#718096;text-align:center;padding:60px">Click ▶ Run Backtest</div>',
                    unsafe_allow_html=True); return
    trades = st.session_state.bt_trades; viol = st.session_state.bt_viol
    df = st.session_state.bt_df
    fast = int(cfg.get("fast_ema",9)); slow = int(cfg.get("slow_ema",15))
    if not trades: st.warning("No trades. Try wider timeframe/period."); return
    trade_stats(trades); st.markdown("---")
    if viol:
        st.error(f"⚠️ {len(viol)} SL/Target violations (both hit same candle)")
        with st.expander("View"): st.dataframe(style_df(pd.DataFrame(viol)),use_container_width=True)
    else: st.success("✅ No violations!")
    st.plotly_chart(make_chart(df.tail(500),trades=trades,
        title=f"Backtest — {cfg.get('ticker_name')}",fast=fast,slow=slow,h=560),
        use_container_width=True, key="bt_chart")
    st.plotly_chart(pnl_chart(trades),use_container_width=True,key="bt_pnl")
    tdf = pd.DataFrame(trades)
    co  = ["#","Type","Entry","Exit","Entry Price","Exit Price","SL","Target",
           "Bar High","Bar Low","Entry Reason","Exit Reason","PnL","Qty","Viol"]
    st.dataframe(style_df(tdf[[c for c in co if c in tdf.columns]]),
                 use_container_width=True, height=400)
    st.download_button("⬇ CSV",tdf.to_csv(index=False).encode(),"bt.csv","text/csv",key="bt_dl")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ════════════════════════════════════════════════════════════════════
def tab_live(cfg):
    ltp_banner(cfg); st.markdown("## 🚀 Live Trading")
    c1,c2,c3,_ = st.columns([1,1,1,3])
    start = c1.button("▶ Start",    type="primary", use_container_width=True, key="btn_start")
    stop  = c2.button("⏹ Stop",                    use_container_width=True, key="btn_stop")
    sq    = c3.button("✖ Squareoff",                use_container_width=True, key="btn_sq")

    if start and not _get("running", False):
        _set("running", True); _set("status","RUNNING")
        _set("log",[]); _set("pos",None); _set("trades",[])
        # capture dhan_client in main thread before passing to thread
        dhan_c = st.session_state.get("dhan_client")
        t = threading.Thread(
            target=live_thread,
            args=(cfg, cfg["symbol"], dhan_c),
            daemon=True)
        t.start()
        st.success("✅ Live trading started!  (refreshes every 3 seconds)")

    if stop and _get("running", False):
        _set("running", False); _set("status","STOPPED")
        st.info("⏹ Stopping…")

    if sq and _get("pos"):
        pos=_get("pos"); ltp=_get("ltp") or pos["e"]
        pnl=round(((ltp-pos["e"]) if pos["type"]=="buy" else (pos["e"]-ltp))*int(cfg.get("quantity",1)),2)
        t2={"#":len(_get("trades",[]))+1,"Type":pos["type"].upper(),
            "Entry":fdt(pos["et"]),"Exit":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "Entry Price":round(pos["e"],4),"Exit Price":round(ltp,4),
            "SL":round(pos["si"],4),"Target":round(pos["ti"],4),
            "Entry Reason":pos["r"],"Exit Reason":"Manual Squareoff",
            "PnL":pnl,"Qty":cfg.get("quantity",1),"Source":"Live"}
        _push("trades",t2); _set("pos",None)
        if cfg.get("enable_dhan") and st.session_state.dhan_client:
            place_order(st.session_state.dhan_client,cfg,pos["type"],ltp,is_exit=True)
        st.success(f"✅ Squared off!  PnL: ₹{pnl:+.2f}")

    # Status
    status = _get("status","STOPPED"); bc = "#48bb78" if status=="RUNNING" else "#fc8181"
    dot    = "🟢" if status=="RUNNING" else "🔴"
    st.markdown(f'<div class="bdg" style="background:{bc}">{dot} {status}</div>',
                unsafe_allow_html=True)
    cfg_box(cfg)

    # EMA values
    ind = _get("ind",{})
    if ind:
        ef_v=ind.get("ef",float("nan")); es_v=ind.get("es",float("nan"))
        atr_v=ind.get("atr",float("nan")); ef_p=ind.get("ef_p",float("nan"))
        es_p=ind.get("es_p",float("nan")); fast=ind.get("fast",9); slow=ind.get("slow",15)
        ltp_v=ind.get("ltp",0)
        ef_s  = f"{ef_v:.2f}"  if not np.isnan(ef_v)  else "—"
        es_s  = f"{es_v:.2f}"  if not np.isnan(es_v)  else "—"
        atr_s2= f"{atr_v:.2f}" if not np.isnan(atr_v) else "—"
        spd   = ef_v-es_v if (not np.isnan(ef_v) and not np.isnan(es_v)) else 0.0
        spd_s = f"{spd:.2f}"; spd_c = "#48bb78" if spd>=0 else "#fc8181"
        if not any(np.isnan([ef_v,es_v,ef_p,es_p])):
            if   ef_v>es_v and ef_p<=es_p: xst="🔀 JUST CROSSED UP (Bullish)"
            elif ef_v<es_v and ef_p>=es_p: xst="🔀 JUST CROSSED DOWN (Bearish)"
            elif ef_v>es_v:                xst="▲ Fast > Slow (Bullish)"
            else:                          xst="▼ Fast < Slow (Bearish)"
        else: xst="Calculating…"
        st.markdown("#### ⚡ EMA Values")
        ec=st.columns(5)
        ec[0].markdown(f'<div class="ecard"><div style="color:#a0aec0;font-size:11px">EMA({fast})</div><div style="color:#f6ad55;font-size:20px;font-weight:700">{ef_s}</div></div>',unsafe_allow_html=True)
        ec[1].markdown(f'<div class="ecard"><div style="color:#a0aec0;font-size:11px">EMA({slow})</div><div style="color:#76e4f7;font-size:20px;font-weight:700">{es_s}</div></div>',unsafe_allow_html=True)
        ec[2].markdown(f'<div class="ecard"><div style="color:#a0aec0;font-size:11px">Spread</div><div style="color:{spd_c};font-size:20px;font-weight:700">{spd_s}</div></div>',unsafe_allow_html=True)
        ec[3].markdown(f'<div class="ecard"><div style="color:#a0aec0;font-size:11px">ATR(14)</div><div style="color:#e9d8a6;font-size:20px;font-weight:700">{atr_s2}</div></div>',unsafe_allow_html=True)
        ec[4].markdown(f'<div class="ecard"><div style="color:#a0aec0;font-size:11px">Status</div><div style="color:#e2e8f0;font-size:12px;font-weight:700;margin-top:6px">{xst}</div></div>',unsafe_allow_html=True)
        ts_s = _get("last_ts"); ts_str = ts_s.strftime("%H:%M:%S") if ts_s else "—"
        bt   = ind.get("bt","")
        st.markdown(
            f'<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;'
            f'padding:7px 14px;font-size:12px;color:#a0aec0;margin:5px 0">'
            f'📡 Last Bar [{fdt(bt)}]: O:{ind.get("bo",0):.2f} H:{ind.get("bh",0):.2f} '
            f'L:{ind.get("bl",0):.2f} C:{ltp_v:.2f} | Fetched: '
            f'<b style="color:#48bb78">{ts_str} IST</b>'
            f'</div>', unsafe_allow_html=True)
    elif _get("running",False):
        st.info("⏳ Waiting for first data tick (~2 seconds)…")

    # Position
    pos=_get("pos"); st.markdown("#### 📌 Position")
    if pos:
        ltp=_get("ltp") or pos["e"]
        upnl=round(((ltp-pos["e"]) if pos["type"]=="buy" else (pos["e"]-ltp))*int(cfg.get("quantity",1)),2)
        bc2="#48bb78" if upnl>=0 else "#fc8181"; ptc="#48bb78" if pos["type"]=="buy" else "#fc8181"
        bg="rgba(72,187,120,0.12)" if upnl>=0 else "rgba(252,129,129,0.12)"
        arr="🔼 BUY" if pos["type"]=="buy" else "🔽 SELL"
        st.markdown(
            f'<div style="background:{bg};border:1px solid {bc2};border-radius:10px;padding:14px">'
            f'<b style="color:{ptc};font-size:16px">{arr}</b> &nbsp; '
            f'Entry: <b>{pos["e"]:.2f}</b> | LTP: <b style="color:{bc2}">{ltp:.2f}</b> | '
            f'SL: <b style="color:#fc8181">{pos["sl"]:.2f}</b> | '
            f'Target: <b style="color:#48bb78">{pos["tgt"]:.2f}</b> | '
            f'P&L: <b style="color:{bc2}">₹{upnl:+.2f}</b> | '
            f'Time: {fdt(pos["et"])}<br>'
            f'<span style="color:#a0aec0;font-size:11px">{pos["r"]}</span>'
            f'</div>', unsafe_allow_html=True)
    else: st.info("📭 No open position")

    # Chart
    df=_get("df")
    if df is not None and not df.empty:
        fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
        st.plotly_chart(make_chart(df.tail(300),
            title=f"Live — {cfg.get('ticker_name')}",fast=fast,slow=slow,pos=pos,h=480),
            use_container_width=True, key="live_chart")

    # Elliott Wave
    st.markdown("---")
    ew=_get("ew",{})
    if ew and ew.get("ok"):
        st.markdown("#### 🌊 Elliott Wave (Automated Signal)")
        show_ew_panel(ew, cfg, df if df is not None else None,
                      int(cfg.get("fast_ema",9)), int(cfg.get("slow_ema",15)))
    elif _get("running",False):
        st.info("🌊 Elliott Wave will appear once ≥25 candles load…")
    else:
        st.info("🌊 Start live trading to see Elliott Wave analysis.")

    # Log
    st.markdown("#### 📟 Log")
    logs=_get("log",[]); log_html='<div class="logb">'
    for ln in reversed(logs[-80:]):
        c=("#fc8181"   if "ERR" in ln or "WARN" in ln
           else "#48bb78" if any(x in ln for x in ["BUY","SELL","START","▲","▼"])
           else "#f6ad55" if any(x in ln for x in ["EXIT","SIGNAL","◆"])
           else "#a0aec0")
        log_html += f'<div style="color:{c}">{ln}</div>'
    st.markdown(log_html+"</div>", unsafe_allow_html=True)

    # Completed trades
    lt=_get("trades",[])
    if lt:
        st.markdown(f"#### ✅ Trades ({len(lt)})")
        trade_stats(lt)
        ltd=pd.DataFrame(lt); cs=["#","Type","Entry","Exit","Entry Price","Exit Price",
                                    "SL","Target","Entry Reason","Exit Reason","PnL","Qty"]
        st.dataframe(style_df(ltd[[c for c in cs if c in ltd.columns]]),
                     use_container_width=True, height=260)
        st.plotly_chart(pnl_chart(lt),use_container_width=True,key="live_pnl")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE HISTORY
# ════════════════════════════════════════════════════════════════════
def tab_history(cfg):
    ltp_banner(cfg); st.markdown("## 📚 Trade History")
    all_t=[]
    for t in st.session_state.bt_trades: tc=t.copy(); tc["Source"]="🔬 BT"; all_t.append(tc)
    for t in _get("trades",[]): tc=t.copy(); tc.setdefault("Source","🚀 Live"); all_t.append(tc)
    if not all_t:
        st.markdown('<div style="color:#718096;text-align:center;padding:60px">No trades yet.</div>',
                    unsafe_allow_html=True); return
    df=pd.DataFrame(all_t)
    f1,f2,f3=st.columns(3)
    src=f1.selectbox("Source",["All"]+sorted(df["Source"].unique().tolist()),key="h_src")
    typ=f2.selectbox("Type",["All","BUY","SELL"],key="h_typ")
    sv =f3.checkbox("Only Violations",False,key="h_v")
    if src!="All": df=df[df["Source"]==src]
    if typ!="All": df=df[df["Type"]==typ]
    if sv and "Viol" in df.columns: df=df[df["Viol"]==True]
    trade_stats(df.to_dict("records"))
    if not df.empty:
        st.plotly_chart(pnl_chart(df.to_dict("records")),use_container_width=True,key="hist_pnl")
        co=["#","Source","Type","Entry","Exit","Entry Price","Exit Price","SL","Target",
            "Entry Reason","Exit Reason","PnL","Qty","Viol"]
        sc=[c for c in co if c in df.columns]
        st.dataframe(style_df(df[sc]),use_container_width=True,height=450)
        st.download_button("⬇ CSV",df[sc].to_csv(index=False).encode(),"hist.csv","text/csv",key="h_dl")

# ════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ════════════════════════════════════════════════════════════════════
def tab_optimize(cfg):
    ltp_banner(cfg); st.markdown("## 🔧 Optimization")
    st.info("Grid-search → click ✅ Apply to load best parameters into sidebar.")
    with st.expander("⚙️ Setup", expanded=True):
        oc1,oc2=st.columns(2)
        with oc1:
            opt_s  =st.selectbox("Strategy",["EMA Crossover","Simple Buy","Simple Sell","Elliott Wave"],key="opt_str")
            opt_m  =st.selectbox("Optimize For",["Accuracy%","TotalPnL","Score","RR"],key="opt_m")
            min_tr =int(st.number_input("Min Trades",1,1000,5,key="opt_mt"))
            opt_qty=int(st.number_input("Qty",1,1_000_000,int(cfg.get("quantity",1)),key="opt_qty"))
        with oc2:
            opt_tf =st.selectbox("Interval",list(TF_PERIODS.keys()),
                index=list(TF_PERIODS.keys()).index(cfg.get("timeframe","15m")),key="opt_tf")
            opt_per=st.selectbox("Period",TF_PERIODS[opt_tf],key="opt_per")
            desired=float(st.number_input("Desired Accuracy %",0.0,100.0,60.0,key="opt_acc"))
            st.markdown(f"**Symbol:** `{cfg.get('symbol','—')}`")
        st.markdown("**Ranges:**"); pc1,pc2,pc3=st.columns(3)
        if opt_s=="EMA Crossover":
            fem=int(pc1.number_input("Fast min",1,100,5,key="opt_fem"))
            feM=int(pc2.number_input("Fast max",1,200,20,key="opt_feM"))
            feS=max(1,int(pc3.number_input("Fast step",1,50,2,key="opt_feS")))
            sem=int(pc1.number_input("Slow min",5,500,10,key="opt_sem"))
            seM=int(pc2.number_input("Slow max",5,500,50,key="opt_seM"))
            seS=max(1,int(pc3.number_input("Slow step",1,50,5,key="opt_seS")))
        slm=float(pc1.number_input("SL min",1.0,5000.0,5.0,key="opt_slm"))
        slM=float(pc2.number_input("SL max",1.0,5000.0,50.0,key="opt_slM"))
        slS=float(pc3.number_input("SL step",0.5,500.0,5.0,key="opt_slS"))
        tgm=float(pc1.number_input("Tgt min",1.0,10000.0,10.0,key="opt_tgm"))
        tgM=float(pc2.number_input("Tgt max",1.0,10000.0,100.0,key="opt_tgM"))
        tgS=float(pc3.number_input("Tgt step",0.5,500.0,10.0,key="opt_tgS"))

    if st.button("🚀 Run Optimization",type="primary",use_container_width=True,key="btn_opt"):
        with st.spinner("Fetching…"):
            df_o=fetch_ui(cfg.get("symbol","^NSEI"),opt_tf,opt_per)
        if df_o is None or df_o.empty: st.error("No data."); return
        def frange(a,b,s):
            out=[]; v=a
            while v<=b+1e-9: out.append(round(v,4)); v=round(v+s,8)
            return out or [a]
        grid={"sl_points":frange(slm,slM,slS),"target_points":frange(tgm,tgM,tgS)}
        if opt_s=="EMA Crossover":
            grid["fast_ema"]=list(range(fem,feM+1,feS))
            grid["slow_ema"]=list(range(sem,seM+1,seS))
        total_c=1
        for v in grid.values(): total_c*=len(v)
        st.info(f"Testing {total_c:,} combinations…")
        pb=st.progress(0); pt=st.empty()
        def cb(d,t): pb.progress(d/t); pt.markdown(f"**{d:,}/{t:,}** ({d/t*100:.0f}%)")
        base={**cfg,"strategy":opt_s,"quantity":opt_qty}
        results=run_opt(df_o,base,grid,metric=opt_m,min_tr=min_tr,cb=cb)
        pb.empty(); pt.empty()
        st.session_state.opt_results=results; st.session_state.opt_ran=True
        st.success(f"✅ Done! {len(results)} valid results.")

    if not st.session_state.opt_ran: return
    results=st.session_state.opt_results
    if not results: st.warning("No valid results. Widen ranges."); return
    des=float(st.session_state.get("opt_acc",60.0))
    fil=[r for r in results if r.get("Accuracy%",0)>=des]
    disp=fil if fil else results
    if fil: st.success(f"{len(fil)} results ≥{des:.0f}% accuracy")
    else:   st.warning(f"No results at {des:.0f}%. Showing best available.")
    st.markdown(f"### 🏆 Top {min(len(disp),15)} Results")
    hc=st.columns([3,1,1,1,1,1,1])
    for col,lbl in zip(hc,["Parameters","Accuracy","P&L","Trades","R:R","Score",""]):
        col.markdown(f'<span style="color:#90cdf4;font-size:11px;font-weight:700">{lbl}</span>',
                     unsafe_allow_html=True)
    for i,res in enumerate(disp[:15]):
        parts=[]
        if res.get("fast_ema") is not None: parts.append(f"F={int(res['fast_ema'])} S={int(res['slow_ema'])}")
        if res.get("sl_points") is not None: parts.append(f"SL={res['sl_points']}")
        if res.get("target_points") is not None: parts.append(f"Tgt={res['target_points']}")
        ps=" | ".join(parts) or "—"
        ac=float(res.get("Accuracy%",0)); pnl=float(res.get("TotalPnL",0))
        rr=float(res.get("RR",0)); sc=float(res.get("Score",0)); tr=int(res.get("Trades",0))
        ac2="#48bb78" if ac>=des else "#f6ad55"; pc2="#48bb78" if pnl>=0 else "#fc8181"
        rc=st.columns([3,1,1,1,1,1,1])
        rc[0].markdown(f'<div style="font-size:12px;color:#e2e8f0;padding:3px">{ps}</div>',unsafe_allow_html=True)
        rc[1].markdown(f'<div style="text-align:center;font-size:13px;color:{ac2};font-weight:700;padding:3px">{ac:.1f}%</div>',unsafe_allow_html=True)
        rc[2].markdown(f'<div style="text-align:center;font-size:13px;color:{pc2};font-weight:700;padding:3px">₹{pnl:+.0f}</div>',unsafe_allow_html=True)
        rc[3].markdown(f'<div style="text-align:center;font-size:12px;color:#a0aec0;padding:3px">{tr}</div>',unsafe_allow_html=True)
        rc[4].markdown(f'<div style="text-align:center;font-size:12px;color:#76e4f7;padding:3px">{rr:.2f}</div>',unsafe_allow_html=True)
        rc[5].markdown(f'<div style="text-align:center;font-size:12px;color:#f6ad55;padding:3px">{sc:.1f}</div>',unsafe_allow_html=True)
        if rc[6].button("✅ Apply",key=f"oa_{i}",use_container_width=True):
            if res.get("fast_ema") is not None:
                st.session_state["_aFE"]=int(res["fast_ema"]); st.session_state["_aSE"]=int(res["slow_ema"])
                st.session_state["s_fe"]=int(res["fast_ema"]); st.session_state["s_se"]=int(res["slow_ema"])
            if res.get("sl_points") is not None:
                st.session_state["_aSL"]=float(res["sl_points"]); st.session_state["s_slp"]=float(res["sl_points"])
            if res.get("target_points") is not None:
                st.session_state["_aTG"]=float(res["target_points"]); st.session_state["s_tgp2"]=float(res["target_points"])
            st.success(f"✅ Applied: {ps}"); st.rerun()
    rdf=pd.DataFrame(disp)
    sc2=[c for c in ["fast_ema","slow_ema","sl_points","target_points","Trades",
                     "Accuracy%","TotalPnL","AvgWin","AvgLoss","RR","Score"]
         if c in rdf.columns and rdf[c].notna().any()]
    if sc2: st.dataframe(style_df(rdf[sc2]),use_container_width=True,height=340)
    if not rdf.empty:
        st.download_button("⬇ CSV",rdf.to_csv(index=False).encode(),"opt.csv","text/csv",key="opt_dl")

# ════════════════════════════════════════════════════════════════════
# MAIN
# Auto-refresh: updates live metrics. Does NOT reload the tab.
# ════════════════════════════════════════════════════════════════════
def main():
    st.markdown(
        '<div class="hdr">'
        '<div><h1>📈 Smart Investing</h1>'
        '<p>Full Automation · NSE · BSE · Crypto · Commodities · Dhan Broker</p></div>'
        '<div style="color:#90cdf4;font-size:12px;text-align:right">'
        'EMA · Elliott Wave · Auto-Trade · Optimized</div>'
        '</div>', unsafe_allow_html=True)

    cfg = sidebar()
    tab1,tab2,tab3,tab4 = st.tabs(
        ["🔬 Backtest","🚀 Live Trading","📚 History","🔧 Optimize"])
    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history(cfg)
    with tab4: tab_optimize(cfg)

    # ── Auto-refresh: only live metrics, no full tab reload ─────────
    if _get("running", False):
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
