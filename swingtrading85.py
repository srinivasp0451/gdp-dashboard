# ╔══════════════════════════════════════════════════════════════════╗
# ║       SMART INVESTING  –  Full Auto Algo Trading  v5            ║
# ║  Root-cause fixes:                                              ║
# ║  1. No blocking calls in main thread (ltp cached by thread)     ║
# ║  2. Auto-refresh: time.sleep(1.5)+rerun AFTER full render       ║
# ║  3. Simple Buy/Sell fires on FIRST data tick (no candle wait)   ║
# ║  4. EMA values use pre-computed variables (no nested f-strings) ║
# ║  5. EW: shows each wave price, type, fib ratios, confidence     ║
# ║  6. EW in Optimization tab                                      ║
# ║  7. Thread never auto-stops (all exceptions caught+logged)      ║
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
st.set_page_config(page_title="Smart Investing", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
IST = pytz.timezone("Asia/Kolkata")

# ════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""<style>
.main{background:#0e1117}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif}
.hdr{background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);padding:12px 22px;
  border-radius:12px;margin-bottom:12px;border:1px solid #2d4a6e;display:flex;
  align-items:center;justify-content:space-between}
.hdr h1{margin:0;color:#e2e8f0;font-size:21px}
.hdr p{margin:0;color:#90cdf4;font-size:11px}
.ltpbar{background:linear-gradient(135deg,#1a1f2e,#252b3d);border:1px solid #2d4a6e;
  border-radius:10px;padding:9px 18px;display:flex;align-items:center;gap:16px;margin-bottom:10px}
.ltptick{color:#90cdf4;font-size:11px}
.ltpval{color:#e2e8f0;font-size:24px;font-weight:700}
.ltpup{color:#48bb78;font-size:13px;font-weight:600}
.ltpdn{color:#fc8181;font-size:13px;font-weight:600}
.ltpmeta{color:#718096;font-size:10px;margin-left:auto}
.mc{background:#1a1f2e;border:1px solid #2d3748;border-radius:9px;padding:12px;margin:4px 0}
.ec{background:#141d2b;border:1px solid #2d4a6e;border-radius:8px;padding:10px;text-align:center;margin:2px}
.cfgbox{background:#1a1f2e;border-radius:8px;padding:11px;border-left:3px solid #4299e1;
  margin:7px 0;font-size:12px;line-height:1.8;color:#cbd5e0}
.ewer{background:#111d2e;border:1px solid #2d5a8e;border-radius:10px;padding:14px;margin:8px 0}
.wrow{background:#1a2535;border:1px solid #2d3748;border-radius:7px;padding:9px 12px;margin:4px 0}
.lb{background:#0d1117;border-radius:8px;padding:8px;height:185px;overflow-y:auto;
  font-family:'Courier New',monospace;font-size:11px;color:#a0aec0}
.badge{display:inline-block;padding:3px 12px;border-radius:16px;font-size:11px;font-weight:700;color:white;margin-bottom:6px}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:transparent}
.stTabs [data-baseweb="tab"]{background:#1a1f2e;border-radius:8px;padding:7px 18px;
  border:1px solid #2d3748;color:#a0aec0}
.stTabs [aria-selected="true"]{background:#2b6cb0!important;color:white!important}
section[data-testid="stSidebar"]{background:#0d1117}
section[data-testid="stSidebar"] label{color:#a0aec0!important;font-size:12px}
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════
TICKERS = {
    "Nifty 50":"^NSEI","BankNifty":"^NSEBANK","Sensex":"^BSESN",
    "BTC":"BTC-USD","ETH":"ETH-USD","Gold":"GC=F","Silver":"SI=F","Custom":None
}
TF_PERIODS = {
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo"],"15m":["1d","5d","7d","1mo"],
    "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
MAX_FETCH = {"1m":"7d","5m":"60d","15m":"60d","1h":"730d","1d":"max","1wk":"max"}
TF_MIN    = {"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080}

# ════════════════════════════════════════════════════════════════════
# THREAD-SAFE STORE  (background thread writes here; main thread reads)
# NEVER call _rate_wait / yfinance from main thread — that blocks UI
# ════════════════════════════════════════════════════════════════════
_L = threading.Lock()
_D = {
    "running":False, "status":"STOPPED",
    "pos":None, "trades":[], "log":[],
    "df":None, "ltp":None, "prev_close":None,
    "last_ts":None, "ew":{}, "indicators":{},
    "_rl":0.0,
}

def _g(k, d=None):
    with _L: return _D.get(k, d)
def _s(k, v):
    with _L: _D[k] = v
def _app(k, v, mx=400):
    with _L:
        _D[k].append(v)
        if len(_D[k]) > mx: _D[k] = _D[k][-mx:]

def _sync():
    """Copy thread store → session_state so widgets see fresh data."""
    for k in ["running","status","pos","trades","log","df","ltp",
              "prev_close","last_ts","ew","indicators"]:
        st.session_state["_"+k] = _g(k)

# ── Session state (UI-only keys) ─────────────────────────────────
_UI_DEFAULTS = {
    "_running":False,"_status":"STOPPED","_pos":None,"_trades":[],"_log":[],
    "_df":None,"_ltp":None,"_prev_close":None,"_last_ts":None,"_ew":{},"_indicators":{},
    "live_thread":None,"bt_trades":[],"bt_viol":[],"bt_df":None,"bt_ran":False,
    "dhan_client":None,"opt_results":[],"opt_ran":False,
}
for _k,_v in _UI_DEFAULTS.items():
    if _k not in st.session_state: st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════
# DATA FETCH  (only called from background thread)
# ════════════════════════════════════════════════════════════════════
def _wait():
    """Rate-limit yfinance: min 1.5s between requests. THREAD ONLY."""
    with _L:
        gap = time.time() - _D["_rl"]
        if gap < 1.5: time.sleep(1.5 - gap)
        _D["_rl"] = time.time()

def _clean(raw):
    if raw is None or raw.empty: return None
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize("UTC").tz_convert(IST)
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    return df[~df.index.duplicated(keep="last")].sort_index()

def _fetch(symbol, interval, period):
    """Fetch with max warmup. THREAD ONLY."""
    _wait()
    try:
        raw = yf.Ticker(symbol).history(period=MAX_FETCH.get(interval,period),
                                         interval=interval, auto_adjust=True, prepost=False)
        df = _clean(raw)
        if df is None or df.empty:
            _wait()
            raw = yf.Ticker(symbol).history(period=period, interval=interval,
                                             auto_adjust=True, prepost=False)
            df = _clean(raw)
        return df
    except Exception as e:
        _log(f"[fetch] {e}"); return None

def fetch_for_ui(symbol, interval, period):
    """Backtest / optimization data fetch — called from main thread with spinner."""
    try:
        raw = yf.Ticker(symbol).history(period=MAX_FETCH.get(interval,period),
                                         interval=interval, auto_adjust=True, prepost=False)
        df = _clean(raw)
        if df is None or df.empty:
            raw = yf.Ticker(symbol).history(period=period, interval=interval,
                                             auto_adjust=True, prepost=False)
            df = _clean(raw)
        return df
    except Exception as e:
        st.error(f"Fetch error: {e}"); return None

# ════════════════════════════════════════════════════════════════════
# INDICATORS  (TradingView-accurate EMA)
# ════════════════════════════════════════════════════════════════════
def ema_tv(series, n):
    if len(series) < n: return pd.Series(np.nan, index=series.index)
    alpha = 2.0/(n+1); vals = series.ffill().values.astype(float)
    out = np.full(len(vals), np.nan); out[n-1] = np.nanmean(vals[:n])
    for i in range(n, len(vals)):
        v = vals[i] if not np.isnan(vals[i]) else out[i-1]
        out[i] = alpha*v + (1-alpha)*out[i-1]
    return pd.Series(out, index=series.index)

def atr_series(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()], axis=1).max(axis=1)
    return ema_tv(tr, n)

def add_ind(df, fast, slow):
    if df is None or df.empty: return df
    df = df.copy()
    df["ef"] = ema_tv(df["close"], fast)
    df["es"] = ema_tv(df["close"], slow)
    df["atr"] = atr_series(df)
    df["cu"] = (df["ef"]>df["es"]) & (df["ef"].shift(1)<=df["es"].shift(1))
    df["cd"] = (df["ef"]<df["es"]) & (df["ef"].shift(1)>=df["es"].shift(1))
    return df

def ema_angle(ema, lb=3):
    v = ema.dropna()
    if len(v)<lb+1: return 0.0
    base = v.iloc[-lb]
    return 0.0 if base==0 else abs(math.degrees(math.atan((v.iloc[-1]-base)/base*100/lb)))

# ════════════════════════════════════════════════════════════════════
# ELLIOTT WAVE — full detail with prices, fib ratios, confidence
# ════════════════════════════════════════════════════════════════════
def _swings(df, win=5):
    h,l = df["high"].values, df["low"].values; pts=[]
    for i in range(win, len(df)-win):
        lo,hi = max(0,i-win), min(len(df),i+win+1)
        if h[i]==h[lo:hi].max(): pts.append({"dt":df.index[i],"px":h[i],"i":i,"t":"H"})
        if l[i]==l[lo:hi].min(): pts.append({"dt":df.index[i],"px":l[i],"i":i,"t":"L"})
    pts.sort(key=lambda x:x["i"]); cl=[]
    for p in pts:
        if not cl: cl.append(p); continue
        if cl[-1]["t"]==p["t"]:
            if (p["t"]=="H" and p["px"]>cl[-1]["px"]) or (p["t"]=="L" and p["px"]<cl[-1]["px"]): cl[-1]=p
        else: cl.append(p)
    return cl

def _fib_score(ratio, targets):
    """How close ratio is to any Fibonacci target (0-1)."""
    if ratio<=0: return 0
    best = min(abs(ratio-t)/t for t in targets)
    return max(0, 1-best/0.15)

def detect_ew(df):
    """
    Returns rich EW dict with:
    - waves: list of wave dicts with start/end prices, size, fib_ratio
    - rules: list of EW rule check dicts {rule, passed, detail}
    - confidence: 0-100 score
    - recommendation: trade rec dict
    """
    BLANK = {"ok":False,"status":"Insufficient data","direction":"UNCLEAR","wave_type":"—",
             "waves":[],"rules":[],"confidence":0,"current_wave":"—","wave_detail":[],
             "next_targets":{},"after_msg":"Need ≥25 candles","current_price":0,
             "trade_bias":"","recommendation":None}
    cp = float(df["close"].iloc[-1]) if len(df) else 0
    if len(df)<25: return {**BLANK,"current_price":cp}
    swings = _swings(df, win=max(3,len(df)//40))
    if len(swings)<4: return {**BLANK,"ok":True,"status":"Forming","current_price":cp,"after_msg":"Wait for more swing points"}

    # ── Try 5-wave impulse ──────────────────────────────────────
    if len(swings)>=6:
        for st0,label,sign in [("L","BULLISH",1),("H","BEARISH",-1)]:
            s = swings[-6:]
            if s[0]["t"]!=st0: continue
            P = [pt["px"] for pt in s]
            # Wave sizes (all positive)
            W1=sign*(P[1]-P[0]); W2=sign*(P[2]-P[1]); W3=sign*(P[3]-P[2])
            W4=sign*(P[4]-P[3]); W5=sign*(P[5]-P[4])
            if W1<=0 or W3<=0: continue
            w2ret = abs(W2)/W1 if W1 else 0
            w3pct = W3/W1      if W1 else 0
            w4ret = abs(W4)/W3 if W3 else 0

            # EW rules
            rules = [
                {"rule":"Wave 3 not shortest",     "passed":W3>=min(W1,W5 if W5>0 else W1),  "detail":f"W1={W1:.1f} W3={W3:.1f}"},
                {"rule":"Wave 2 retraces <100% W1","passed":w2ret<1.0,                         "detail":f"W2 retrace={w2ret*100:.1f}%"},
                {"rule":"Wave 4 no overlap W1",    "passed":P[4]>P[1] if sign==1 else P[4]<P[1],"detail":f"W4 end {P[4]:.0f} vs W1 end {P[1]:.0f}"},
                {"rule":"Wave 3 ≥ 61.8% of W1",   "passed":W3>=W1*0.618,                      "detail":f"W3/W1={w3pct:.2f}x"},
                {"rule":"Wave 4 retrace <W3 high", "passed":w4ret<1.0,                         "detail":f"W4 retrace={w4ret*100:.1f}%"},
            ]
            n_pass = sum(1 for r in rules if r["passed"])
            if n_pass < 3: continue

            # Fibonacci quality
            fib_w2 = _fib_score(w2ret, [0.382,0.5,0.618])
            fib_w3 = _fib_score(w3pct, [1.0,1.382,1.618,2.0,2.618])
            fib_w4 = _fib_score(w4ret, [0.236,0.382,0.5])
            confidence = min(100, int(n_pass/5*50 + (fib_w2+fib_w3+fib_w4)/3*50))

            # Wave detail list with full info
            wave_detail = [
                {"label":"1","type":"Impulse","dir":"UP" if sign==1 else "DOWN",
                 "start":P[0],"end":P[1],"size":round(abs(P[1]-P[0]),2),
                 "pct":round(abs(P[1]-P[0])/P[0]*100,2) if P[0] else 0,
                 "fib_note":"Impulse start","color":"#f6ad55"},
                {"label":"2","type":"Correction","dir":"DOWN" if sign==1 else "UP",
                 "start":P[1],"end":P[2],"size":round(abs(P[2]-P[1]),2),
                 "pct":round(w2ret*100,1),
                 "fib_note":f"{w2ret*100:.1f}% ret of W1  (ideal 38.2%-61.8%)","color":"#fc8181"},
                {"label":"3","type":"Impulse","dir":"UP" if sign==1 else "DOWN",
                 "start":P[2],"end":P[3],"size":round(abs(P[3]-P[2]),2),
                 "pct":round(w3pct*100,1),
                 "fib_note":f"{w3pct:.2f}x W1  (ideal ≥1.618x)","color":"#48bb78"},
                {"label":"4","type":"Correction","dir":"DOWN" if sign==1 else "UP",
                 "start":P[3],"end":P[4],"size":round(abs(P[4]-P[3]),2),
                 "pct":round(w4ret*100,1),
                 "fib_note":f"{w4ret*100:.1f}% ret of W3  (ideal 23.6%-38.2%)","color":"#f6ad55"},
                {"label":"5","type":"Impulse (forming)","dir":"UP" if sign==1 else "DOWN",
                 "start":P[4],"end":P[5],"size":round(abs(P[5]-P[4]),2),
                 "pct":round(abs(P[5]-P[4])/abs(P[3]-P[2])*100,1) if abs(P[3]-P[2]) else 0,
                 "fib_note":f"Currently {abs(P[5]-P[4]):.1f}pts | Targets below","color":"#76e4f7"},
            ]

            projs = {
                f"W5@61.8%W1 = {round(P[4]+sign*W1*0.618,2):.2f}": round(P[4]+sign*W1*0.618,2),
                f"W5=W1 = {round(P[4]+sign*W1,2):.2f}":             round(P[4]+sign*W1,2),
                f"W5@161.8%W1 = {round(P[4]+sign*W1*1.618,2):.2f}": round(P[4]+sign*W1*1.618,2),
            }
            sl_s  = round(P[4] - sign*abs(W1)*0.05, 2)
            tgt_s = round(P[4] + sign*W1, 2)
            rec = {"signal":"BUY" if sign==1 else "SELL","entry":cp,
                   "wave_context":f"Wave 5 of {label.title()} Impulse",
                   "sl_suggestion":sl_s,"tgt_suggestion":tgt_s,
                   "sl_basis":f"Just below W4 end ({P[4]:.2f})","tgt_basis":"W5 = 100% of W1",
                   "confidence":confidence}

            waves = [{"label":str(n+1),"start":s[n],"end":s[n+1]} for n in range(5)]
            return {
                "ok":True,"status":f"5-Wave Impulse  ({'Bullish' if sign==1 else 'Bearish'})",
                "direction":label,"wave_type":"Impulse","waves":waves,
                "current_wave":"Wave 5 (in progress)","wave_detail":wave_detail,
                "rules":rules,"confidence":confidence,
                "next_targets":projs,"after_msg":"Expect ABC correction after W5",
                "current_price":cp,"trade_bias":"BUY" if sign==1 else "SELL","recommendation":rec,
            }

    # ── Try ABC correction ───────────────────────────────────────
    if len(swings)>=4:
        for st0,label,sign in [("H","CORRECTIVE_DOWN",-1),("L","CORRECTIVE_UP",1)]:
            s = swings[-4:]
            if s[0]["t"]!=st0: continue
            P = [pt["px"] for pt in s]
            A = sign*(P[0]-P[1]); B = sign*(P[2]-P[1]); C = sign*(P[2]-P[3])
            if A<=0: continue
            bret = abs(B)/A if A else 0
            cpct = C/A if A else 0
            rules = [
                {"rule":"B retraces A partially","passed":0.2<bret<1.0,"detail":f"B={bret*100:.1f}% of A"},
                {"rule":"C moves in A direction","passed":C>0,"detail":f"C size={C:.1f}pts"},
            ]
            confidence = min(100, int(_fib_score(bret,[0.382,0.5,0.618,0.786])*50 +
                                      _fib_score(cpct,[0.618,1.0,1.272,1.618])*50))
            wave_detail = [
                {"label":"A","type":"Impulse","dir":"DOWN" if sign==-1 else "UP",
                 "start":P[0],"end":P[1],"size":round(abs(P[1]-P[0]),2),"pct":round(A/P[0]*100,2) if P[0] else 0,
                 "fib_note":"Wave A — first leg","color":"#fc8181"},
                {"label":"B","type":"Correction","dir":"UP" if sign==-1 else "DOWN",
                 "start":P[1],"end":P[2],"size":round(abs(P[2]-P[1]),2),"pct":round(bret*100,1),
                 "fib_note":f"{bret*100:.1f}% retrace of A  (ideal 38.2%-78.6%)","color":"#48bb78"},
                {"label":"C","type":"Impulse (forming)","dir":"DOWN" if sign==-1 else "UP",
                 "start":P[2],"end":P[3],"size":round(abs(P[3]-P[2]),2),"pct":round(cpct*100,1),
                 "fib_note":f"Currently {cpct*100:.1f}% of A | Targets below","color":"#fc8181"},
            ]
            c_tgt = round(P[2]-sign*A,2)
            projs = {f"C=A = {c_tgt:.2f}":c_tgt,
                     f"C=127.2%A = {round(P[2]-sign*A*1.272,2):.2f}":round(P[2]-sign*A*1.272,2)}
            rec = {"signal":"SELL" if sign==-1 else "BUY","entry":cp,
                   "wave_context":f"Wave C  {'Bearish' if sign==-1 else 'Bullish'} ABC",
                   "sl_suggestion":round(P[2]+sign*A*0.1,2),"tgt_suggestion":c_tgt,
                   "sl_basis":"Above B end","tgt_basis":"C = A equal move","confidence":confidence}
            waves = [{"label":lb,"start":s[i],"end":s[i+1]} for i,lb in enumerate(["A","B","C"])]
            return {
                "ok":True,"status":f"ABC Correction  ({'Bearish' if sign==-1 else 'Bullish'})",
                "direction":label,"wave_type":"Correction","waves":waves,
                "current_wave":"Wave C (in progress)","wave_detail":wave_detail,
                "rules":rules,"confidence":confidence,
                "next_targets":projs,"after_msg":"Expect new impulse after ABC",
                "current_price":cp,"trade_bias":"SELL" if sign==-1 else "BUY","recommendation":rec,
            }

    return {**BLANK,"ok":True,"status":"Structure forming","current_price":cp,
            "after_msg":"Insufficient confirmed pivot points"}

# ════════════════════════════════════════════════════════════════════
# SL / TARGET
# ════════════════════════════════════════════════════════════════════
def calc_sl_tgt(entry, tt, row, cfg):
    atr = float(row.get("atr", entry*0.01) or entry*0.01)
    if np.isnan(atr) or atr==0: atr = entry*0.01
    sl_t = cfg.get("sl_type","Custom Points")
    tg_t = cfg.get("target_type","Custom Points")
    slp  = float(cfg.get("sl_points",10))
    tgp  = float(cfg.get("target_points",20))
    sign = 1 if tt=="buy" else -1
    if sl_t=="ATR Based":         sl = entry - sign*atr*float(cfg.get("atr_sl_mult",1.5))
    else:                          sl = entry - sign*slp
    sld = abs(entry-sl)
    if tg_t=="ATR Based Target":   tgt = entry + sign*atr*float(cfg.get("atr_tgt_mult",3.0))
    elif tg_t=="Risk-Reward":       tgt = entry + sign*sld*float(cfg.get("rr_ratio",2.0))
    else:                           tgt = entry + sign*tgp
    return round(sl,4), round(tgt,4)

# ════════════════════════════════════════════════════════════════════
# SIGNALS
# ════════════════════════════════════════════════════════════════════
def ema_sig(df, idx, cfg):
    if idx<1 or idx>=len(df): return None,None
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es = row.get("ef",np.nan), row.get("es",np.nan)
    pf,ps = prev.get("ef",np.nan),prev.get("es",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return None,None
    if cfg.get("use_angle",False):
        if ema_angle(df["ef"].iloc[:idx+1]) < float(cfg.get("min_angle",0)): return None,None
    ct   = cfg.get("crossover_type","Simple"); body = abs(row["close"]-row["open"])
    def ok():
        if ct=="Simple": return True
        if ct=="Custom Candle":  return body >= float(cfg.get("custom_candle",10))
        if ct=="ATR Candle": a=float(row.get("atr",0) or 0); return a==0 or body>=a
        return True
    f,s = int(cfg.get("fast_ema",9)), int(cfg.get("slow_ema",15))
    if ef>es and pf<=ps and ok(): return "buy",  f"EMA({f}) crossed ABOVE EMA({s})"
    if ef<es and pf>=ps and ok(): return "sell", f"EMA({f}) crossed BELOW EMA({s})"
    return None,None

def ema_rev(df, idx, pt):
    if idx<1: return False
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es=row.get("ef",np.nan),row.get("es",np.nan)
    pf,ps=prev.get("ef",np.nan),prev.get("es",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return False
    if pt=="buy"  and ef<es and pf>=ps: return True
    if pt=="sell" and ef>es and pf<=ps: return True
    return False

def fdt(dt):
    if hasattr(dt,"strftime"): return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)

# ════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════
def run_backtest(df, cfg):
    if df is None or df.empty: return [],[]
    strat=cfg.get("strategy","EMA Crossover")
    fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
    qty=int(cfg.get("quantity",1))
    sl_t=cfg.get("sl_type","Custom Points"); tg_t=cfg.get("target_type","Custom Points")
    df=add_ind(df.copy(),fast,slow)
    trades=[]; violations=[]; pos=None; pending=None; tnum=0

    for i in range(1,len(df)):
        row=df.iloc[i]
        if pos:
            sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
            if sl_t=="Trailing SL":
                tr=float(cfg.get("sl_points",10))
                ns=row["high"]-tr if pt=="buy" else row["low"]+tr
                if (pt=="buy" and ns>sl) or (pt=="sell" and ns<sl): sl=pos["sl"]=ns
            if tg_t=="Trailing Target":
                tp=float(cfg.get("target_points",20))
                nt=row["high"]+tp if pt=="buy" else row["low"]-tp
                if (pt=="buy" and nt>pos["tgt"]) or (pt=="sell" and nt<pos["tgt"]): pos["tgt"]=nt
            ema_ex=(sl_t=="Reverse EMA Cross" or tg_t=="EMA Cross") and ema_rev(df,i,pt)
            ep=None; er=None; viol=False
            if pt=="buy":
                if row["low"]<=sl: ep=sl; er=f"SL Low≤{sl:.2f}"; viol=tg_t!="Trailing Target" and row["high"]>=tgt
                elif tg_t!="Trailing Target" and row["high"]>=tgt: ep=tgt; er=f"Tgt High≥{tgt:.2f}"
                elif ema_ex: ep=row["open"]; er="EMA Rev exit"
            else:
                if row["high"]>=sl: ep=sl; er=f"SL High≥{sl:.2f}"; viol=tg_t!="Trailing Target" and row["low"]<=tgt
                elif tg_t!="Trailing Target" and row["low"]<=tgt: ep=tgt; er=f"Tgt Low≤{tgt:.2f}"
                elif ema_ex: ep=row["open"]; er="EMA Rev exit"
            if ep is None and i==len(df)-1: ep=row["close"]; er="End of data"
            if ep is not None:
                pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                t={"Trade #":tnum,"Type":pt.upper(),"Entry Time":fdt(pos["et"]),"Exit Time":fdt(row.name),
                   "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                   "SL":round(pos["si"],4),"Target":round(pos["ti"],4),
                   "Candle High":round(row["high"],4),"Candle Low":round(row["low"],4),
                   "Entry Reason":pos["reason"],"Exit Reason":er,"PnL":pnl,"Qty":qty,"Violated":viol}
                trades.append(t)
                if viol: violations.append(t)
                pos=None
        if pos is None and pending:
            sig,rsn=pending; pending=None; ep=float(row["open"])
            sl,tgt=calc_sl_tgt(ep,sig,row,cfg); tnum+=1
            pos={"type":sig,"entry":ep,"et":row.name,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"reason":rsn}
        if pos is None:
            if strat=="EMA Crossover":
                sig,rsn=ema_sig(df,i,cfg)
                if sig: pending=(sig,rsn)
            elif strat=="Simple Buy":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"buy",row,cfg); tnum+=1
                pos={"type":"buy","entry":ep,"et":row.name,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"reason":"Simple Buy"}
            elif strat=="Simple Sell":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"sell",row,cfg); tnum+=1
                pos={"type":"sell","entry":ep,"et":row.name,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"reason":"Simple Sell"}
            elif strat=="Elliott Wave" and i>=20:
                ew=detect_ew(df.iloc[:i+1]); bias=ew.get("trade_bias","")
                if bias: sig="buy" if bias=="BUY" else "sell"; pending=(sig,f"EW:{ew.get('current_wave','')}")
    return trades, violations

# ════════════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE
# ════════════════════════════════════════════════════════════════════
def run_opt(df, base_cfg, grid, metric="Accuracy %", min_tr=5, cb=None):
    results=[]; keys=list(grid.keys()); combos=list(itertools.product(*grid.values()))
    total=len(combos)
    for idx,combo in enumerate(combos):
        cfg={**base_cfg,"sl_type":"Custom Points","target_type":"Custom Points"}
        for k,v in zip(keys,combo): cfg[k]=v
        try:
            f=int(cfg.get("fast_ema",9)); s=int(cfg.get("slow_ema",15))
            if f>=s:
                if cb: cb(idx+1,total)
                continue
            df_i=add_ind(df.copy(),f,s)
            trades,_=run_backtest(df_i,cfg)
            if len(trades)<min_tr:
                if cb: cb(idx+1,total)
                continue
            tdf=pd.DataFrame(trades); wins=len(tdf[tdf["PnL"]>0]); tot=len(tdf)
            acc=wins/tot*100; pnl=tdf["PnL"].sum()
            aw=tdf[tdf["PnL"]>0]["PnL"].mean() if wins else 0
            al=tdf[tdf["PnL"]<=0]["PnL"].mean() if tot-wins else 0
            rr=abs(aw/al) if al and al!=0 else 0
            score=acc*0.4+(min(rr,5)*20)*0.3+(20 if pnl>0 else 0)*0.3
            row_d={"Trades":tot,"Wins":wins,"Losses":tot-wins,"Accuracy%":round(acc,1),
                   "TotalPnL":round(pnl,2),"AvgWin":round(aw,2),"AvgLoss":round(al,2),
                   "RR":round(rr,2),"Score":round(score,2)}
            for k2,v2 in zip(keys,combo): row_d[k2]=v2
            results.append(row_d)
        except Exception: pass
        if cb: cb(idx+1,total)
    results.sort(key=lambda x:x.get(metric.replace(" %","%"),0),reverse=True)
    return results[:25]

# ════════════════════════════════════════════════════════════════════
# DHAN BROKER
# ════════════════════════════════════════════════════════════════════
def init_dhan(cid,tok):
    try:
        from dhanhq import dhanhq; return dhanhq(cid,tok)
    except ImportError: st.warning("pip install dhanhq"); return None
    except Exception as e: st.error(f"Dhan: {e}"); return None

def get_ip():
    try: return requests.get("https://api.ipify.org?format=json",timeout=4).json().get("ip","?")
    except Exception:
        try: return requests.get("https://ifconfig.me/ip",timeout=4).text.strip()
        except: return "N/A"

def place_order(dhan,cfg,sig,ltp,is_exit=False):
    if not dhan: return {"error":"Not connected"}
    try:
        if cfg.get("options_trading",False):
            ot=cfg.get("opt_exit_type" if is_exit else "opt_entry_type","MARKET")
            sid=cfg.get("ce_id") if sig=="buy" else cfg.get("pe_id")
            return dhan.place_order(transactionType="SELL" if is_exit else "BUY",
                exchangeSegment=cfg.get("fno_exc","NSE_FNO"),productType="INTRADAY",
                orderType=ot,validity="DAY",securityId=str(sid),
                quantity=int(cfg.get("opts_qty",65)),
                price=round(ltp,2) if ot=="LIMIT" else 0,triggerPrice=0)
        ot=cfg.get("exit_order" if is_exit else "entry_order","MARKET")
        txn=("SELL" if is_exit else "BUY") if sig=="buy" else ("BUY" if is_exit else "SELL")
        return dhan.place_order(security_id=str(cfg.get("sec_id","1594")),
            exchange_segment=cfg.get("exchange","NSE"),transaction_type=txn,
            quantity=int(cfg.get("dhan_qty",1)),order_type=ot,
            product_type=cfg.get("product","INTRADAY"),
            price=round(ltp,2) if ot=="LIMIT" else 0)
    except Exception as e: return {"error":str(e)}

# ════════════════════════════════════════════════════════════════════
# LOG
# ════════════════════════════════════════════════════════════════════
def _log(msg):
    _app("log", f"[{datetime.now(IST).strftime('%H:%M:%S')}] {msg}", mx=400)

# ════════════════════════════════════════════════════════════════════
# LIVE THREAD  — ALL exceptions caught; never stops unless user stops
# ════════════════════════════════════════════════════════════════════
def live_thread(cfg, symbol):
    _log(f"▶ START {symbol} {cfg['timeframe']} {cfg['strategy']}")
    tf_min = TF_MIN.get(cfg["timeframe"],5)
    fast   = int(cfg.get("fast_ema",9))
    slow   = int(cfg.get("slow_ema",15))
    strat  = cfg.get("strategy","EMA Crossover")
    sl_t   = cfg.get("sl_type","Custom Points")
    tg_t   = cfg.get("target_type","Custom Points")
    qty    = int(cfg.get("quantity",1))
    dhan   = st.session_state.get("dhan_client")
    use_dhan = cfg.get("enable_dhan",False)
    pending = None
    last_sig_bar = None
    last_bdy = -1

    # ── fetch prev_close once at startup ──────────────────────────
    try:
        _wait()
        raw2 = yf.Ticker(symbol).history(period="5d",interval="1d",auto_adjust=True,prepost=False)
        df2  = _clean(raw2)
        if df2 is not None and len(df2)>=2:
            _s("prev_close", float(df2["close"].iloc[-2]))
    except Exception: pass

    while _g("running", False):
        try:
            df = _fetch(symbol, cfg["timeframe"], cfg["period"])
            if df is None or df.empty:
                _log("[WARN] no data"); time.sleep(1.5); continue

            df  = add_ind(df, fast, slow)
            ltp = float(df["close"].iloc[-1])
            now = datetime.now(IST)

            # Store computed EMA/ATR values for UI display
            lr = df.iloc[-1]
            ef_v  = float(lr.get("ef",  float("nan")) or float("nan"))
            es_v  = float(lr.get("es",  float("nan")) or float("nan"))
            atr_v = float(lr.get("atr", float("nan")) or float("nan"))
            ef_p  = float(df.iloc[-2].get("ef", float("nan")) if len(df)>=2 else float("nan"))
            es_p  = float(df.iloc[-2].get("es", float("nan")) if len(df)>=2 else float("nan"))
            _s("indicators",{"ef":ef_v,"es":es_v,"atr":atr_v,"ef_prev":ef_p,"es_prev":es_p,
                              "fast":fast,"slow":slow,"ltp":ltp,
                              "bar_time":df.index[-1],"bar_o":float(lr["open"]),"bar_h":float(lr["high"]),"bar_l":float(lr["low"])})
            _s("df",  df)
            _s("ltp", ltp)
            _s("last_ts", now)

            if len(df)>=20:
                _s("ew", detect_ew(df))

            pos = _g("pos")

            # ── EXIT CHECK (vs LTP every tick) ────────────────────
            if pos:
                sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
                if sl_t=="Trailing SL":
                    tr=float(cfg.get("sl_points",10))
                    ns=ltp-tr if pt=="buy" else ltp+tr
                    if (pt=="buy" and ns>sl) or (pt=="sell" and ns<sl):
                        pos=dict(pos,sl=ns); _s("pos",pos); sl=ns
                if tg_t=="Trailing Target":
                    tp=float(cfg.get("target_points",20))
                    nt=ltp+tp if pt=="buy" else ltp-tp
                    if (pt=="buy" and nt>tgt) or (pt=="sell" and nt<tgt):
                        pos=dict(pos,tgt=nt); _s("pos",pos)
                ep=None; er=None
                if pt=="buy":
                    if ltp<=sl:                                   ep=sl;  er=f"SL hit {ltp:.2f}<={sl:.2f}"
                    elif tg_t!="Trailing Target" and ltp>=tgt:   ep=tgt; er=f"Tgt hit {ltp:.2f}>={tgt:.2f}"
                else:
                    if ltp>=sl:                                   ep=sl;  er=f"SL hit {ltp:.2f}>={sl:.2f}"
                    elif tg_t!="Trailing Target" and ltp<=tgt:   ep=tgt; er=f"Tgt hit {ltp:.2f}<={tgt:.2f}"
                if ep is None and (sl_t=="Reverse EMA Cross" or tg_t=="EMA Cross"):
                    if ema_rev(df,len(df)-1,pt): ep=ltp; er="EMA Rev Cross exit"
                if ep is not None:
                    pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                    t={"Trade #":len(_g("trades",[]))+1,"Type":pt.upper(),
                       "Entry Time":fdt(pos["et"]),"Exit Time":now.strftime("%Y-%m-%d %H:%M:%S"),
                       "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                       "SL":round(pos["si"],4),"Target":round(pos["ti"],4),
                       "Entry Reason":pos["reason"],"Exit Reason":er,"PnL":pnl,"Qty":qty,"Source":"🚀 Live"}
                    _app("trades",t); _s("pos",None)
                    if use_dhan and dhan: _log(f"EXIT order: {place_order(dhan,cfg,pt,ltp,is_exit=True)}")
                    _log(f"✖ EXIT {pt.upper()} @{ep:.2f} | {er} | PnL:{pnl:+.2f}")

            # ── ENTRY LOGIC ───────────────────────────────────────
            if _g("pos") is None:
                # Cooldown
                cd_ok=True
                if cfg.get("cooldown_enabled",True):
                    trd=_g("trades",[])
                    if trd:
                        try:
                            le=datetime.strptime(trd[-1].get("Exit Time","2000-01-01 00:00:00"),"%Y-%m-%d %H:%M:%S")
                            le=IST.localize(le)
                            if (now-le).total_seconds()<int(cfg.get("cooldown_s",5)): cd_ok=False
                        except: pass

                # ── SIMPLE BUY/SELL: IMMEDIATE, no candle boundary ──
                if strat=="Simple Buy" and cd_ok:
                    sl,tgt=calc_sl_tgt(ltp,"buy",lr,cfg)
                    _s("pos",{"type":"buy","entry":ltp,"et":now,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"reason":"Simple Buy"})
                    if use_dhan and dhan: _log(f"ENTRY: {place_order(dhan,cfg,'buy',ltp)}")
                    _log(f"▲ BUY @{ltp:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")

                elif strat=="Simple Sell" and cd_ok:
                    sl,tgt=calc_sl_tgt(ltp,"sell",lr,cfg)
                    _s("pos",{"type":"sell","entry":ltp,"et":now,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"reason":"Simple Sell"})
                    if use_dhan and dhan: _log(f"ENTRY: {place_order(dhan,cfg,'sell',ltp)}")
                    _log(f"▼ SELL @{ltp:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")

                # ── EMA/EW: execute pending from previous boundary ──
                elif pending and cd_ok:
                    sig,rsn=pending; pending=None; ep=float(df["open"].iloc[-1])
                    sl,tgt=calc_sl_tgt(ep,sig,lr,cfg)
                    _s("pos",{"type":sig,"entry":ep,"et":now,"sl":sl,"si":sl,"tgt":tgt,"ti":tgt,"reason":rsn})
                    if use_dhan and dhan: _log(f"ENTRY: {place_order(dhan,cfg,sig,ep)}")
                    _log(f"▲ ENTRY {sig.upper()} @{ep:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")

                else:
                    tm=now.hour*60+now.minute; at_bdy=(tm%tf_min==0) and (tm!=last_bdy)
                    if at_bdy and cd_ok:
                        last_bdy=tm
                        if strat=="EMA Crossover":
                            sig,rsn=ema_sig(df,len(df)-1,cfg)
                            if sig and df.index[-1]!=last_sig_bar:
                                last_sig_bar=df.index[-1]; pending=(sig,rsn)
                                _log(f"◆ SIGNAL {sig.upper()} → next candle open")
                        elif strat=="Elliott Wave":
                            ew=_g("ew",{}); bias=ew.get("trade_bias","")
                            if bias:
                                s2="buy" if bias=="BUY" else "sell"
                                pending=(s2,f"EW:{ew.get('current_wave','')}"); _log(f"◆ EW {s2.upper()} → next candle")

            time.sleep(1.5)

        except Exception as e:
            _log(f"[ERR] {e}")
            time.sleep(2)  # brief pause then keep running

    _log("■ STOPPED.")

# ════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════
def make_chart(df, trades=None, title="", fast=9, slow=15, live_pos=None, h=520):
    if df is None or df.empty: return go.Figure()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.04,row_heights=[0.78,0.22])
    fig.add_trace(go.Candlestick(x=df.index,open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="Price",increasing_line_color="#48bb78",decreasing_line_color="#fc8181",
        increasing_fillcolor="#48bb78",decreasing_fillcolor="#fc8181"),row=1,col=1)
    if "ef" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ef"],name=f"EMA({fast})",line=dict(color="#f6ad55",width=1.5)),row=1,col=1)
    if "es" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["es"],name=f"EMA({slow})",line=dict(color="#76e4f7",width=1.5)),row=1,col=1)
    if "volume" in df.columns:
        vc=["#48bb78" if c>=o else "#fc8181" for c,o in zip(df["close"],df["open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Vol",marker_color=vc,opacity=0.5),row=2,col=1)
    if trades:
        for t in trades[:60]:
            try:
                et=pd.to_datetime(t.get("Entry Time","")); xt=pd.to_datetime(t.get("Exit Time",""))
                c="#48bb78" if t["Type"]=="BUY" else "#fc8181"
                sym="triangle-up" if t["Type"]=="BUY" else "triangle-down"
                pc="#48bb78" if t.get("PnL",0)>=0 else "#fc8181"
                fig.add_trace(go.Scatter(x=[et],y=[t["Entry Price"]],mode="markers+text",
                    marker=dict(symbol=sym,size=11,color=c),text=[f"E:{t['Entry Price']:.0f}"],
                    textposition="top center",textfont=dict(size=8,color=c),showlegend=False),row=1,col=1)
                fig.add_trace(go.Scatter(x=[xt],y=[t["Exit Price"]],mode="markers+text",
                    marker=dict(symbol="x",size=9,color=pc),text=[f"X:{t['Exit Price']:.0f}"],
                    textposition="bottom center",textfont=dict(size=8,color=pc),showlegend=False),row=1,col=1)
                fig.add_shape(type="line",x0=et,x1=xt,y0=t["SL"],y1=t["SL"],line=dict(color="#fc8181",width=1,dash="dot"),row=1,col=1)
                fig.add_shape(type="line",x0=et,x1=xt,y0=t["Target"],y1=t["Target"],line=dict(color="#48bb78",width=1,dash="dot"),row=1,col=1)
            except: pass
    if live_pos:
        try:
            c="#48bb78" if live_pos["type"]=="buy" else "#fc8181"
            sym="triangle-up" if live_pos["type"]=="buy" else "triangle-down"
            fig.add_trace(go.Scatter(x=[live_pos["et"]],y=[live_pos["entry"]],mode="markers+text",
                marker=dict(symbol=sym,size=16,color=c,line=dict(color="white",width=2)),
                text=[f"ENTRY {live_pos['entry']:.2f}"],textposition="top center"),row=1,col=1)
            fig.add_hline(y=live_pos["sl"],line=dict(color="#fc8181",width=2,dash="dot"),annotation_text=f"SL {live_pos['sl']:.2f}",annotation_font=dict(color="#fc8181"),row=1,col=1)
            fig.add_hline(y=live_pos["tgt"],line=dict(color="#48bb78",width=2,dash="dot"),annotation_text=f"Tgt {live_pos['tgt']:.2f}",annotation_font=dict(color="#48bb78"),row=1,col=1)
        except: pass
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0",size=11),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0,r=0,t=36,b=0),height=h,xaxis_rangeslider_visible=False,
        xaxis2=dict(showgrid=True,gridcolor="#1a1f2e"),yaxis=dict(showgrid=True,gridcolor="#1a1f2e"),
        yaxis2=dict(showgrid=True,gridcolor="#1a1f2e"))
    if title: fig.update_layout(title=dict(text=title,font=dict(size=13,color="#90cdf4")))
    return fig

def make_ew_chart(df, ew, fast, slow):
    fig=make_chart(df.tail(120),title="Elliott Wave",fast=fast,slow=slow,h=420)
    wc={"1":"#f6ad55","2":"#fc8181","3":"#48bb78","4":"#e9d8a6","5":"#76e4f7","A":"#fc8181","B":"#48bb78","C":"#fc8181"}
    for w in ew.get("waves",[]):
        s=w["start"]; e=w["end"]; c=wc.get(w["label"],"#a0aec0")
        fig.add_trace(go.Scatter(x=[s["dt"],e["dt"]],y=[s["px"],e["px"]],
            mode="lines+markers+text",line=dict(color=c,width=2.5),marker=dict(size=8,color=c),
            text=["",f"W{w['label']}"],textposition="top center",textfont=dict(color=c,size=12),name=f"W{w['label']}"),row=1,col=1)
    return fig

def pnl_chart(trades):
    pnls=[t.get("PnL",0) for t in trades]; cum=[0]+list(np.cumsum(pnls))
    c="#48bb78" if cum[-1]>=0 else "#fc8181"; fill="rgba(72,187,120,0.1)" if c=="#48bb78" else "rgba(252,129,129,0.1)"
    fig=go.Figure(go.Scatter(x=list(range(len(cum))),y=cum,mode="lines+markers",fill="tozeroy",fillcolor=fill,line=dict(color=c,width=2),name="P&L"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",height=240,margin=dict(l=0,r=0,t=20,b=0),font=dict(color="#e2e8f0"))
    return fig

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
def sidebar():
    cfg={}
    with st.sidebar:
        st.markdown("## 📈 Smart Investing"); st.markdown("---")
        # Instrument
        tn=st.selectbox("🎯 Ticker",list(TICKERS.keys()),key="s_tn")
        sym=(st.text_input("Symbol","RELIANCE.NS",key="s_sym").strip() if tn=="Custom" else TICKERS[tn])
        cfg.update(ticker_name=tn,symbol=sym)
        # Timeframe
        tf=st.selectbox("⏱ Interval",list(TF_PERIODS.keys()),index=2,key="s_tf")
        pr=TF_PERIODS[tf]; period=st.selectbox("Period",pr,index=min(1,len(pr)-1),key="s_period")
        cfg.update(timeframe=tf,period=period)
        # Strategy
        strat=st.selectbox("🧠 Strategy",["EMA Crossover","Simple Buy","Simple Sell","Elliott Wave"],key="s_strat")
        cfg["strategy"]=strat
        if strat=="EMA Crossover":
            c1,c2=st.columns(2)
            fe=c1.number_input("Fast EMA",1,500,int(st.session_state.get("_aFE",9)),key="s_fe")
            se=c2.number_input("Slow EMA",1,500,int(st.session_state.get("_aSE",15)),key="s_se")
            cfg.update(fast_ema=fe,slow_ema=se)
            uang=st.checkbox("Angle Filter",False,key="s_ang")
            mang=st.number_input("Min Angle°",0.0,90.0,0.0,0.5,key="s_mang") if uang else 0.0
            cfg.update(use_angle=uang,min_angle=mang)
            ct=st.selectbox("Crossover Type",["Simple","Custom Candle","ATR Candle"],key="s_ct"); cfg["crossover_type"]=ct
            if ct=="Custom Candle": cfg["custom_candle"]=st.number_input("Min Body",0.0,value=10.0,key="s_cb")
        else:
            cfg.update(fast_ema=9,slow_ema=15,use_angle=False,min_angle=0.0,crossover_type="Simple",custom_candle=10.0)
        cfg["quantity"]=st.number_input("📦 Qty",1,1_000_000,1,key="s_qty")
        # SL
        sl_t=st.selectbox("🛡 SL Type",["Custom Points","ATR Based","Trailing SL","Reverse EMA Cross","Risk-Reward"],key="s_slt")
        cfg["sl_type"]=sl_t
        slp_def=float(st.session_state.get("_aSL",10.0))
        if sl_t=="ATR Based":
            cfg["atr_sl_mult"]=st.number_input("ATR Mult",0.1,10.0,1.5,0.1,key="s_asm"); cfg["sl_points"]=slp_def
        else:
            cfg["sl_points"]=st.number_input("SL Points",0.1,value=slp_def,step=0.5,key="s_slp"); cfg["atr_sl_mult"]=1.5
        # Target
        tg_t=st.selectbox("🎯 Target Type",["Custom Points","ATR Based Target","Trailing Target","EMA Cross","Risk-Reward"],key="s_tgt")
        cfg["target_type"]=tg_t
        tgp_def=float(st.session_state.get("_aTG",20.0))
        if tg_t=="ATR Based Target":
            cfg["atr_tgt_mult"]=st.number_input("ATR Mult",0.1,20.0,3.0,0.1,key="s_atm"); cfg["target_points"]=tgp_def
        elif tg_t=="Risk-Reward":
            cfg["rr_ratio"]=st.number_input("R:R",0.1,20.0,float(st.session_state.get("_aRR",2.0)),0.1,key="s_rr")
            cfg["target_points"]=tgp_def; cfg["atr_tgt_mult"]=3.0
        elif tg_t=="Trailing Target":
            cfg["target_points"]=st.number_input("Trail Dist",0.1,value=tgp_def,step=0.5,key="s_tgp"); cfg["atr_tgt_mult"]=3.0; st.caption("Display only — no auto exit")
        else:
            cfg["target_points"]=st.number_input("Target Pts",0.1,value=tgp_def,step=0.5,key="s_tgp2"); cfg["atr_tgt_mult"]=3.0
        cfg.setdefault("rr_ratio",2.0)
        # Controls
        cd=st.checkbox("🔄 Cooldown",True,key="s_cd")
        cds=st.number_input("Cooldown s",1,3600,5,key="s_cds") if cd else 5
        cfg.update(cooldown_enabled=cd,cooldown_s=cds)
        # Dhan
        en_dhan=st.checkbox("🏦 Enable Dhan",False,key="s_dhan"); cfg["enable_dhan"]=en_dhan
        if en_dhan:
            cid=st.text_input("Client ID",key="s_cid"); tok=st.text_input("Token",key="s_tok",type="password")
            cfg.update(dhan_cid=cid,dhan_tok=tok)
            if st.button("Connect & Show IP",use_container_width=True,key="s_conn"):
                cl=init_dhan(cid,tok)
                if cl:
                    st.session_state.dhan_client=cl; ip=get_ip()
                    st.info(f"Your IP: **{ip}** — whitelist at Dhan Console → Profile → API → IP Whitelist"); st.success("Connected!")
                else: st.error("Failed")
            opts=st.checkbox("Options Trading",False,key="s_opts"); cfg["options_trading"]=opts
            if opts:
                cfg["fno_exc"]=st.selectbox("FNO",["NSE_FNO","BSE_FNO"],key="s_fno")
                cfg["ce_id"]=st.text_input("CE ID",key="s_ceid"); cfg["pe_id"]=st.text_input("PE ID",key="s_peid")
                cfg["opts_qty"]=st.number_input("Opts Qty",1,value=65,key="s_oq")
                cfg["opt_entry_type"]=st.selectbox("Entry",["MARKET","LIMIT"],key="s_oent")
                cfg["opt_exit_type"]=st.selectbox("Exit",["MARKET","LIMIT"],key="s_oext")
            else:
                cfg["product"]=st.selectbox("Product",["INTRADAY","DELIVERY"],key="s_prod")
                cfg["exchange"]=st.selectbox("Exchange",["NSE","BSE"],key="s_exc")
                cfg["sec_id"]=st.text_input("Security ID","1594",key="s_sid")
                cfg["dhan_qty"]=st.number_input("Order Qty",1,value=1,key="s_dqty")
                cfg["entry_order"]=st.selectbox("Entry Order",["LIMIT","MARKET"],key="s_eord")
                cfg["exit_order"]=st.selectbox("Exit Order",["MARKET","LIMIT"],key="s_xord")
    return cfg

# ════════════════════════════════════════════════════════════════════
# SHARED WIDGETS
# ════════════════════════════════════════════════════════════════════
def ltp_banner(cfg):
    """Reads from cached _TS — no network call."""
    ltp  = _g("ltp") or st.session_state.get("_ltp")
    prev = _g("prev_close") or st.session_state.get("_prev_close")
    ltp  = ltp  or 0.0
    prev = prev or ltp
    chg  = ltp - prev; pct = chg/prev*100 if prev else 0
    arrow= "▲" if chg>=0 else "▼"; cls = "ltpup" if chg>=0 else "ltpdn"
    ts_s = datetime.now(IST).strftime("%H:%M:%S IST")
    st.markdown(
        f'<div class="ltpbar">'
        f'<div><div class="ltptick">📈 {cfg.get("ticker_name","—")} · {cfg.get("symbol","")}</div>'
        f'<div class="ltpval">₹ {ltp:,.2f}</div></div>'
        f'<div class="{cls}">{arrow} {abs(chg):.2f} ({abs(pct):.2f}%)</div>'
        f'<div class="ltpmeta">{cfg.get("timeframe","")} · {cfg.get("period","")} · {cfg.get("strategy","")} | {ts_s}</div>'
        f'</div>', unsafe_allow_html=True)

def cfg_box(cfg):
    st.markdown(
        f'<div class="cfgbox">'
        f'<b>📌</b> {cfg.get("ticker_name","—")} ({cfg.get("symbol","—")}) | '
        f'<b>⏱</b> {cfg.get("timeframe","—")}/{cfg.get("period","—")} | '
        f'<b>🧠</b> {cfg.get("strategy","—")} | <b>📦</b> {cfg.get("quantity",1)}<br>'
        f'<b>EMA Fast:</b> {cfg.get("fast_ema",9)} | <b>EMA Slow:</b> {cfg.get("slow_ema",15)} | '
        f'<b>Crossover:</b> {cfg.get("crossover_type","—")}<br>'
        f'<b>🛡 SL:</b> {cfg.get("sl_type","—")} ({cfg.get("sl_points",10)} pts) | '
        f'<b>🎯 Tgt:</b> {cfg.get("target_type","—")} ({cfg.get("target_points",20)} pts)<br>'
        f'<b>🔄 Cooldown:</b> {"✅ "+str(cfg.get("cooldown_s",5))+"s" if cfg.get("cooldown_enabled") else "❌"}'
        f'</div>', unsafe_allow_html=True)

def style_df(df):
    if df.empty: return df.style
    def rc(row):
        pnl=row.get("PnL",0)
        c="rgba(72,187,120,0.10)" if pnl>0 else ("rgba(252,129,129,0.10)" if pnl<0 else "")
        return [f"background-color:{c}"]*len(row)
    styled=df.style.apply(rc,axis=1)
    if "PnL" in df.columns:
        styled=styled.map(lambda v:(f"color:{'#48bb78' if v>0 else '#fc8181'};font-weight:700"
                                     if isinstance(v,(int,float)) else ""),subset=["PnL"])
    if "Violated" in df.columns:
        styled=styled.map(lambda v:("background-color:#2d1515;color:#fc8181;font-weight:700" if v is True else ""),subset=["Violated"])
    return styled

def trade_stats(trades):
    if not trades: return
    df=pd.DataFrame(trades); wins=df[df["PnL"]>0]; loss=df[df["PnL"]<=0]
    tot=df["PnL"].sum(); acc=len(wins)/len(df)*100 if len(df) else 0
    aw=wins["PnL"].mean() if len(wins) else 0; al=loss["PnL"].mean() if len(loss) else 0
    cols=st.columns(6)
    for col,lbl,val,color in [
        (cols[0],"Trades",len(df),"#e2e8f0"),(cols[1],"Winners",len(wins),"#48bb78"),
        (cols[2],"Losers",len(loss),"#fc8181"),(cols[3],"Accuracy",f"{acc:.1f}%","#f6ad55"),
        (cols[4],"Total P&L",f"₹{tot:+.2f}","#48bb78" if tot>=0 else "#fc8181"),
        (cols[5],"Avg W/L",f"₹{aw:.2f}/₹{al:.2f}","#76e4f7")]:
        col.markdown(f'<div class="mc"><div style="color:#a0aec0;font-size:11px">{lbl}</div><div style="color:{color};font-size:16px;font-weight:700;margin-top:2px">{val}</div></div>',unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# WAVE DETAIL TABLE
# ════════════════════════════════════════════════════════════════════
def wave_detail_panel(ew):
    detail = ew.get("wave_detail",[])
    if not detail: return
    st.markdown("##### 📊 Wave-by-Wave Breakdown")
    for w in detail:
        lbl   = w["label"]
        wtype = w["type"]
        wdir  = w["dir"]
        s_px  = w["start"]
        e_px  = w["end"]
        size  = w["size"]
        pct   = w.get("pct",0)
        fib   = w.get("fib_note","")
        color = w.get("color","#a0aec0")
        dir_arrow = "🔼" if "UP" in wdir else "🔽"
        type_badge = "⚡ Impulse" if "Impulse" in wtype else "↩ Correction"
        # Build HTML without any nested f-string quotes
        badge_bg = "#1a3a1a" if "Impulse" in wtype else "#3a1a1a"
        price_range = f"{s_px:.2f} → {e_px:.2f}"
        st.markdown(
            f'<div class="wrow">'
            f'<span style="color:{color};font-weight:700;font-size:14px">Wave {lbl}</span> &nbsp;'
            f'<span style="background:{badge_bg};color:{color};padding:2px 8px;border-radius:4px;font-size:11px">{type_badge} {dir_arrow}</span> &nbsp;'
            f'<span style="color:#e2e8f0;font-size:13px">{price_range}</span> &nbsp;'
            f'<span style="color:#a0aec0;font-size:12px">({size:.1f} pts, {pct:.1f}%)</span><br>'
            f'<span style="color:#718096;font-size:11px">📐 {fib}</span>'
            f'</div>', unsafe_allow_html=True)

def ew_rules_panel(ew):
    rules = ew.get("rules",[])
    conf  = ew.get("confidence",0)
    if not rules: return
    st.markdown("##### ✅ Elliott Wave Rules Check")
    conf_color = "#48bb78" if conf>=70 else "#f6ad55" if conf>=40 else "#fc8181"
    # Confidence bar
    bar_filled = int(conf/5)  # 0-20 blocks
    bar = "█"*bar_filled + "░"*(20-bar_filled)
    st.markdown(
        f'<div class="mc">'
        f'<div style="color:#a0aec0;font-size:11px">Confidence Score</div>'
        f'<div style="color:{conf_color};font-size:20px;font-weight:700">{conf}%</div>'
        f'<div style="color:{conf_color};font-size:12px;letter-spacing:1px">{bar}</div>'
        f'<div style="color:#718096;font-size:10px">Higher = more textbook EW pattern | Use with other confirmations</div>'
        f'</div>', unsafe_allow_html=True)
    cols=st.columns(2 if len(rules)>2 else len(rules))
    for i,r in enumerate(rules):
        passed = r["passed"]
        icon   = "✅" if passed else "❌"
        rule_c = "#48bb78" if passed else "#fc8181"
        cols[i%2 if len(rules)>2 else i].markdown(
            f'<div style="background:#0e1117;border-radius:6px;padding:6px 10px;margin:3px;font-size:11px">'
            f'{icon} <b style="color:{rule_c}">{r["rule"]}</b><br>'
            f'<span style="color:#718096">{r["detail"]}</span>'
            f'</div>', unsafe_allow_html=True)

def ew_rec_panel(ew, cfg):
    rec = ew.get("recommendation")
    if not rec: return
    sig    = rec["signal"]
    entry  = rec["entry"]
    sl_s   = rec["sl_suggestion"]
    tgt_s  = rec["tgt_suggestion"]
    sl_b   = rec.get("sl_basis","EW structure")
    tgt_b  = rec.get("tgt_basis","EW projection")
    wctx   = rec.get("wave_context","")
    conf   = rec.get("confidence",0)
    sig_c  = "#48bb78" if sig=="BUY" else "#fc8181"
    arrow  = "🔼" if sig=="BUY" else "🔽"
    sl_pts = round(abs(entry-sl_s),2)
    tg_pts = round(abs(entry-tgt_s),2)
    rr_ew  = round(tg_pts/sl_pts,2) if sl_pts else 0
    strat_sl  = cfg.get("sl_points",10)
    strat_sl_t= cfg.get("sl_type","—")
    strat_tg  = cfg.get("target_points",20)
    strat_tg_t= cfg.get("target_type","—")
    st.markdown(
        f'<div class="ewer">'
        f'<div style="font-size:15px;font-weight:700;color:{sig_c};margin-bottom:10px">'
        f'{arrow} EW {sig} Signal &nbsp;·&nbsp; '
        f'<span style="font-size:12px;color:#a0aec0;font-weight:normal">{wctx}</span></div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px">'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
        f'<div style="color:#a0aec0;font-size:11px">Entry</div>'
        f'<div style="color:#e2e8f0;font-size:16px;font-weight:700">₹{entry:,.2f}</div>'
        f'<div style="color:#718096;font-size:10px">Current LTP</div></div>'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
        f'<div style="color:#a0aec0;font-size:11px">EW Stop Loss</div>'
        f'<div style="color:#fc8181;font-size:16px;font-weight:700">₹{sl_s:,.2f}</div>'
        f'<div style="color:#718096;font-size:10px">{sl_b}</div></div>'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
        f'<div style="color:#a0aec0;font-size:11px">EW Target</div>'
        f'<div style="color:#48bb78;font-size:16px;font-weight:700">₹{tgt_s:,.2f}</div>'
        f'<div style="color:#718096;font-size:10px">{tgt_b}</div></div>'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center">'
        f'<div style="color:#a0aec0;font-size:11px">EW R:R</div>'
        f'<div style="color:#f6ad55;font-size:16px;font-weight:700">{rr_ew}x</div>'
        f'<div style="color:#718096;font-size:10px">{tg_pts:.0f}pts / {sl_pts:.0f}pts</div></div>'
        f'</div>'
        f'<div style="background:#0e1117;border-radius:6px;padding:8px 12px;font-size:12px;color:#a0aec0">'
        f'💡 <b style="color:#f6ad55">Your strategy SL:</b> {strat_sl}pts ({strat_sl_t}) &nbsp;|&nbsp; '
        f'<b style="color:#48bb78">Your Target:</b> {strat_tg}pts ({strat_tg_t})<br>'
        f'<span style="color:#718096">↳ Adjust sidebar SL/Target to match EW levels if desired</span>'
        f'</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ════════════════════════════════════════════════════════════════════
def tab_backtest(cfg):
    ltp_banner(cfg)
    st.markdown("## 🔬 Backtesting Engine")
    st.info("EMA: signal on candle N → entry at N+1 open  |  Buy: check Low vs SL first  |  Sell: check High vs SL first")
    if st.button("▶ Run Backtest", type="primary", key="btn_bt"):
        with st.spinner(f"Fetching {cfg.get('symbol')} …"):
            df=fetch_for_ui(cfg["symbol"],cfg["timeframe"],cfg["period"])
        if df is None or df.empty: st.error("No data. Check symbol/interval."); return
        df=add_ind(df,int(cfg.get("fast_ema",9)),int(cfg.get("slow_ema",15)))
        with st.spinner("Running backtest…"): trades,viol=run_backtest(df,cfg)
        st.session_state.bt_trades=trades; st.session_state.bt_viol=viol
        st.session_state.bt_df=df; st.session_state.bt_ran=True
    if not st.session_state.bt_ran:
        st.markdown('<div style="color:#718096;text-align:center;padding:60px">Click ▶ Run Backtest</div>',unsafe_allow_html=True); return
    trades=st.session_state.bt_trades; viol=st.session_state.bt_viol; df=st.session_state.bt_df
    fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
    if not trades: st.warning("No trades generated."); return
    trade_stats(trades); st.markdown("---")
    if viol:
        st.error(f"⚠️ {len(viol)} SL/Target Violations (both hit in same candle)")
        with st.expander("View Violations"): st.dataframe(style_df(pd.DataFrame(viol)),use_container_width=True)
    else: st.success("✅ No violations!")
    st.plotly_chart(make_chart(df.tail(500),trades=trades,title=f"Backtest — {cfg.get('ticker_name')}",fast=fast,slow=slow,h=560),use_container_width=True,key="bt_chart")
    st.plotly_chart(pnl_chart(trades),use_container_width=True,key="bt_pnl")
    st.markdown(f"### 📋 Trade Log ({len(trades)} trades)")
    tdf=pd.DataFrame(trades)
    co=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target","Candle High","Candle Low","Entry Reason","Exit Reason","PnL","Qty","Violated"]
    st.dataframe(style_df(tdf[[c for c in co if c in tdf.columns]]),use_container_width=True,height=400)
    st.download_button("⬇ CSV",tdf.to_csv(index=False).encode(),"bt.csv","text/csv",key="bt_dl")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ════════════════════════════════════════════════════════════════════
def tab_live(cfg):
    _sync()
    ltp_banner(cfg)
    st.markdown("## 🚀 Live Trading")
    c1,c2,c3,_=st.columns([1,1,1,3])
    start=c1.button("▶ Start",type="primary",use_container_width=True,key="btn_start")
    stop =c2.button("⏹ Stop",               use_container_width=True,key="btn_stop")
    sq   =c3.button("✖ Squareoff",           use_container_width=True,key="btn_sq")

    if start and not _g("running",False):
        _s("running",True); _s("status","RUNNING"); _s("log",[]); _s("pos",None)
        t=threading.Thread(target=live_thread,args=(cfg,cfg["symbol"]),daemon=True)
        st.session_state.live_thread=t; t.start()
        st.success("✅ Started! First data tick in ~2 seconds…")

    if stop and _g("running",False):
        _s("running",False); _s("status","STOPPED")
        st.info("⏹ Stopping…")

    if sq and _g("pos"):
        pos=_g("pos"); ltp=_g("ltp") or pos["entry"]
        pnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        t2={"Trade #":len(_g("trades",[]))+1,"Type":pos["type"].upper(),
            "Entry Time":fdt(pos["et"]),"Exit Time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "Entry Price":round(pos["entry"],4),"Exit Price":round(ltp,4),
            "SL":round(pos["si"],4),"Target":round(pos["ti"],4),
            "Entry Reason":pos["reason"],"Exit Reason":"Manual Squareoff","PnL":pnl,"Qty":cfg.get("quantity",1),"Source":"Live"}
        _app("trades",t2); _s("pos",None)
        if cfg.get("enable_dhan") and st.session_state.dhan_client:
            place_order(st.session_state.dhan_client,cfg,pos["type"],ltp,is_exit=True)
        st.success(f"✅ Squared off! PnL: ₹{pnl:+.2f}")

    # Status badge
    status=_g("status","STOPPED"); bc="#48bb78" if status=="RUNNING" else "#fc8181"
    st.markdown(f'<div class="badge" style="background:{bc}">{"🟢" if status=="RUNNING" else "🔴"} {status}</div>',unsafe_allow_html=True)
    st.markdown("#### 🔧 Config"); cfg_box(cfg)

    # ── EMA Values Panel ──────────────────────────────────────────
    ind = _g("indicators",{})
    if ind:
        ef_v   = ind.get("ef",  float("nan"))
        es_v   = ind.get("es",  float("nan"))
        atr_v  = ind.get("atr", float("nan"))
        ef_p   = ind.get("ef_prev", float("nan"))
        es_p   = ind.get("es_prev", float("nan"))
        fast   = ind.get("fast",9)
        slow   = ind.get("slow",15)
        bar_t  = ind.get("bar_time","")
        bar_o  = ind.get("bar_o",0); bar_h=ind.get("bar_h",0); bar_l=ind.get("bar_l",0)
        ltp_v  = ind.get("ltp",0)

        # All values as plain strings — NO nested f-string quotes
        ef_str  = f"{ef_v:.2f}"   if not np.isnan(ef_v)  else "—"
        es_str  = f"{es_v:.2f}"   if not np.isnan(es_v)  else "—"
        atr_str = f"{atr_v:.2f}"  if not np.isnan(atr_v) else "—"
        spread  = (ef_v-es_v) if (not np.isnan(ef_v) and not np.isnan(es_v)) else 0.0
        spr_str = f"{spread:.2f}"
        # Colors as plain Python variables
        spr_color = "#48bb78" if spread>=0 else "#fc8181"

        if not any(np.isnan([ef_v,es_v,ef_p,es_p])):
            if ef_v>es_v and ef_p<=es_p:  xst="🔀 JUST CROSSED UP (Bullish)"
            elif ef_v<es_v and ef_p>=es_p: xst="🔀 JUST CROSSED DOWN (Bearish)"
            elif ef_v>es_v:                xst="▲ Fast > Slow (Bullish)"
            else:                          xst="▼ Fast < Slow (Bearish)"
        else: xst="Calculating…"

        st.markdown("#### ⚡ Live EMA Values")
        ec=st.columns(5)
        ec[0].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">EMA({fast}) Fast</div><div style="color:#f6ad55;font-size:20px;font-weight:700">{ef_str}</div></div>',unsafe_allow_html=True)
        ec[1].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">EMA({slow}) Slow</div><div style="color:#76e4f7;font-size:20px;font-weight:700">{es_str}</div></div>',unsafe_allow_html=True)
        ec[2].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">Spread</div><div style="color:{spr_color};font-size:20px;font-weight:700">{spr_str}</div></div>',unsafe_allow_html=True)
        ec[3].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">ATR(14)</div><div style="color:#e9d8a6;font-size:20px;font-weight:700">{atr_str}</div></div>',unsafe_allow_html=True)
        ec[4].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">Status</div><div style="color:#e2e8f0;font-size:12px;font-weight:700;margin-top:6px">{xst}</div></div>',unsafe_allow_html=True)

        ts_s=_g("last_ts"); ts_str=ts_s.strftime("%H:%M:%S") if ts_s else "—"
        st.markdown(f'<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:7px 14px;font-size:12px;color:#a0aec0;margin:5px 0">📡 <b style="color:#76e4f7">Last Bar [{fdt(bar_t)}]</b>: O:{bar_o:.2f} H:{bar_h:.2f} L:{bar_l:.2f} C:{ltp_v:.2f} | Fetched: <b style="color:#48bb78">{ts_str} IST</b></div>',unsafe_allow_html=True)
    elif _g("running",False):
        st.info("⏳ Fetching first data tick… (~2 seconds)")

    # ── Current Position ──────────────────────────────────────────
    pos=_g("pos"); st.markdown("#### 📌 Current Position")
    if pos:
        ltp=_g("ltp") or pos["entry"]
        upnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        bc2="#48bb78" if upnl>=0 else "#fc8181"; ptc="#48bb78" if pos["type"]=="buy" else "#fc8181"
        bg="rgba(72,187,120,0.12)" if upnl>=0 else "rgba(252,129,129,0.12)"
        arrow="🔼 BUY" if pos["type"]=="buy" else "🔽 SELL"
        sl_v=pos["sl"]; tgt_v=pos["tgt"]; en_v=pos["entry"]; et_v=fdt(pos["et"])
        rsn_v=pos["reason"]
        st.markdown(
            f'<div style="background:{bg};border:1px solid {bc2};border-radius:10px;padding:14px">'
            f'<b style="color:{ptc};font-size:16px">{arrow}</b> &nbsp;'
            f'Entry: <b>{en_v:.2f}</b> | LTP: <b style="color:{bc2}">{ltp:.2f}</b> | '
            f'SL: <b style="color:#fc8181">{sl_v:.2f}</b> | '
            f'Target: <b style="color:#48bb78">{tgt_v:.2f}</b> | '
            f'Unrealised P&L: <b style="color:{bc2}">₹{upnl:+.2f}</b> | Time: {et_v}<br>'
            f'<span style="color:#a0aec0;font-size:11px">Reason: {rsn_v}</span>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.info("📭 No open position")

    # ── Chart ─────────────────────────────────────────────────────
    df=_g("df")
    if df is not None and not df.empty:
        fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
        st.plotly_chart(make_chart(df.tail(300),title=f"Live — {cfg.get('ticker_name')}",
                                   fast=fast,slow=slow,live_pos=pos,h=500),use_container_width=True,key="live_chart")

    # ── Elliott Wave ──────────────────────────────────────────────
    st.markdown("---")
    ew=_g("ew",{})
    if ew and ew.get("ok") and ew.get("status","") not in ("","Insufficient data","Structure forming"):
        st.markdown("#### 🌊 Elliott Wave Analysis")
        s=ew.get("status","—"); d=ew.get("direction","UNCLEAR"); cw=ew.get("current_wave","—"); cp=ew.get("current_price",0)
        wtype=ew.get("wave_type","—"); conf=ew.get("confidence",0)
        dc={"BULLISH":"#48bb78","BEARISH":"#fc8181","CORRECTIVE_DOWN":"#f6ad55","CORRECTIVE_UP":"#76e4f7","UNCLEAR":"#a0aec0"}.get(d,"#a0aec0")
        conf_c="#48bb78" if conf>=70 else "#f6ad55" if conf>=40 else "#fc8181"
        wc=st.columns(5)
        for col,lbl,val,color in [(wc[0],"Structure",s,dc),(wc[1],"Wave Type",wtype,dc),
                                   (wc[2],"Direction",d,dc),(wc[3],"Current Wave",cw,"#e2e8f0"),
                                   (wc[4],"Confidence",f"{conf}%",conf_c)]:
            col.markdown(f'<div class="mc"><div style="color:#a0aec0;font-size:11px">{lbl}</div><div style="color:{color};font-size:13px;font-weight:700;margin-top:2px">{val}</div></div>',unsafe_allow_html=True)

        # Wave-by-wave detail
        wave_detail_panel(ew)

        # Projections
        tgts=ew.get("next_targets",{})
        if tgts:
            st.markdown("##### 🎯 Wave Projections / Next Targets")
            tcols=st.columns(len(tgts))
            for i,(n,v) in enumerate(tgts.items()):
                cl="#48bb78" if v>cp else "#fc8181"
                tcols[i].markdown(f'<div class="mc" style="text-align:center"><div style="color:#a0aec0;font-size:11px">{n}</div><div style="color:{cl};font-size:14px;font-weight:700">{v:,.2f}</div></div>',unsafe_allow_html=True)

        # Rules check + confidence
        ew_rules_panel(ew)

        # Trade recommendation
        if ew.get("recommendation"):
            st.markdown("##### 📋 EW Trade Recommendation"); ew_rec_panel(ew,cfg)

        am=ew.get("after_msg",""); bias=ew.get("trade_bias","")
        if am: st.info(f"💡 {am}")
        if bias: st.success(f"📊 EW Trade Bias: **{bias}**  (use with Confidence score)")

        if df is not None and not df.empty and len(ew.get("waves",[]))>0:
            fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
            st.plotly_chart(make_ew_chart(df,ew,fast,slow),use_container_width=True,key="ew_chart")
    elif _g("running",False):
        st.info("🌊 Elliott Wave analysis will appear once enough candles are loaded…")
    else:
        st.info("🌊 Start live trading to see Elliott Wave analysis.")

    # ── Log ───────────────────────────────────────────────────────
    st.markdown("#### 📟 Activity Log")
    logs=_g("log",[]); log_html='<div class="lb">'
    for ln in reversed(logs[-80:]):
        c=("#fc8181" if "ERR" in ln or "WARN" in ln
           else "#48bb78" if ("BUY" in ln or "SELL" in ln or "START" in ln)
           else "#f6ad55" if "EXIT" in ln or "SIGNAL" in ln else "#a0aec0")
        log_html+=f'<div style="color:{c}">{ln}</div>'
    st.markdown(log_html+"</div>",unsafe_allow_html=True)

    # ── Completed trades ──────────────────────────────────────────
    lt=_g("trades",[])
    if lt:
        st.markdown(f"#### ✅ Completed Trades ({len(lt)})"); trade_stats(lt)
        ltd=pd.DataFrame(lt); cs=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target","Entry Reason","Exit Reason","PnL","Qty"]
        st.dataframe(style_df(ltd[[c for c in cs if c in ltd.columns]]),use_container_width=True,height=270)
        st.plotly_chart(pnl_chart(lt),use_container_width=True,key="live_pnl")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE HISTORY
# ════════════════════════════════════════════════════════════════════
def tab_history(cfg):
    _sync(); ltp_banner(cfg); st.markdown("## 📚 Trade History")
    all_t=[]
    for t in st.session_state.bt_trades: tc=t.copy(); tc["Source"]="🔬 Backtest"; all_t.append(tc)
    for t in _g("trades",[]): tc=t.copy(); tc.setdefault("Source","🚀 Live"); all_t.append(tc)
    if not all_t:
        st.markdown('<div style="color:#718096;text-align:center;padding:60px">No trades yet.</div>',unsafe_allow_html=True); return
    df=pd.DataFrame(all_t)
    f1,f2,f3=st.columns(3)
    src=f1.selectbox("Source",["All"]+sorted(df["Source"].unique().tolist()),key="h_src")
    typ=f2.selectbox("Type",["All","BUY","SELL"],key="h_typ")
    sv=f3.checkbox("Only Violations",False,key="h_v")
    if src!="All": df=df[df["Source"]==src]
    if typ!="All": df=df[df["Type"]==typ]
    if sv and "Violated" in df.columns: df=df[df["Violated"]==True]
    trade_stats(df.to_dict("records"))
    if not df.empty:
        st.plotly_chart(pnl_chart(df.to_dict("records")),use_container_width=True,key="hist_pnl")
        co=["Trade #","Source","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target","Entry Reason","Exit Reason","PnL","Qty","Violated"]
        sc=[c for c in co if c in df.columns]
        st.dataframe(style_df(df[sc]),use_container_width=True,height=450)
        st.download_button("⬇ CSV",df[sc].to_csv(index=False).encode(),"history.csv","text/csv",key="h_dl")

# ════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ════════════════════════════════════════════════════════════════════
def tab_optimize(cfg):
    ltp_banner(cfg); st.markdown("## 🔧 Strategy Optimization")
    st.info("Grid-search strategy parameters → click **✅ Apply** on any result to instantly load settings into sidebar.")
    with st.expander("⚙️ Setup", expanded=True):
        oc1,oc2=st.columns(2)
        with oc1:
            opt_strat=st.selectbox("Strategy to Optimize",["EMA Crossover","Simple Buy","Simple Sell","Elliott Wave"],key="opt_strat")
            opt_metric=st.selectbox("Optimize For",["Accuracy%","TotalPnL","Score","RR"],key="opt_metric")
            min_trades=int(st.number_input("Min Trades",1,1000,5,key="opt_mt"))
            opt_qty=int(st.number_input("Qty",1,1_000_000,int(cfg.get("quantity",1)),key="opt_qty"))
        with oc2:
            opt_tf=st.selectbox("Interval",list(TF_PERIODS.keys()),
                                 index=list(TF_PERIODS.keys()).index(cfg.get("timeframe","15m")),key="opt_tf")
            opt_period=st.selectbox("Period",TF_PERIODS[opt_tf],key="opt_period")
            desired=float(st.number_input("Desired Accuracy %",0.0,100.0,60.0,key="opt_acc"))
            st.markdown(f"**Symbol:** `{cfg.get('symbol','—')}`")
        st.markdown("**Search Ranges:**"); pc1,pc2,pc3=st.columns(3)
        if opt_strat=="EMA Crossover":
            fe_min=int(pc1.number_input("Fast min",1,100,5,key="opt_fem"))
            fe_max=int(pc2.number_input("Fast max",1,200,20,key="opt_feM"))
            fe_stp=max(1,int(pc3.number_input("Fast step",1,50,2,key="opt_feS")))
            se_min=int(pc1.number_input("Slow min",5,500,10,key="opt_sem"))
            se_max=int(pc2.number_input("Slow max",5,500,50,key="opt_seM"))
            se_stp=max(1,int(pc3.number_input("Slow step",1,50,5,key="opt_seS")))
        elif opt_strat=="Elliott Wave":
            sw_min=int(pc1.number_input("Swing window min",2,20,3,key="opt_swm"))
            sw_max=int(pc2.number_input("Swing window max",2,30,10,key="opt_swM"))
            sw_stp=max(1,int(pc3.number_input("Swing window step",1,10,1,key="opt_swS")))
        sl_min=float(pc1.number_input("SL pts min",1.0,5000.0,5.0,key="opt_slm"))
        sl_max=float(pc2.number_input("SL pts max",1.0,5000.0,50.0,key="opt_slM"))
        sl_stp=float(pc3.number_input("SL pts step",0.5,500.0,5.0,key="opt_slS"))
        tg_min=float(pc1.number_input("Tgt pts min",1.0,10000.0,10.0,key="opt_tgm"))
        tg_max=float(pc2.number_input("Tgt pts max",1.0,10000.0,100.0,key="opt_tgM"))
        tg_stp=float(pc3.number_input("Tgt pts step",0.5,500.0,10.0,key="opt_tgS"))

    if st.button("🚀 Run Optimization",type="primary",use_container_width=True,key="btn_opt"):
        with st.spinner("Fetching data…"):
            df_opt=fetch_for_ui(cfg.get("symbol","^NSEI"),opt_tf,opt_period)
        if df_opt is None or df_opt.empty: st.error("No data."); return

        def frange(mn,mx,st_):
            out=[]; v=mn
            while v<=mx+1e-9: out.append(round(v,4)); v=round(v+st_,8)
            return out if out else [mn]

        grid={"sl_points":frange(sl_min,sl_max,sl_stp),"target_points":frange(tg_min,tg_max,tg_stp)}
        if opt_strat=="EMA Crossover":
            grid["fast_ema"]=list(range(fe_min,fe_max+1,fe_stp))
            grid["slow_ema"]=list(range(se_min,se_max+1,se_stp))
        elif opt_strat=="Elliott Wave":
            grid["ew_swing_window"]=list(range(sw_min,sw_max+1,sw_stp))

        total_c=1
        for v in grid.values(): total_c*=len(v)
        st.info(f"Testing {total_c:,} combinations…")
        pb=st.progress(0); pt_txt=st.empty()
        def pcb(done,total): pb.progress(done/total); pt_txt.markdown(f"**{done:,}/{total:,}** ({done/total*100:.0f}%)")

        base_cfg={**cfg,"strategy":opt_strat,"quantity":opt_qty}
        results=run_opt(df_opt,base_cfg,grid,metric=opt_metric,min_tr=min_trades,cb=pcb)
        pb.empty(); pt_txt.empty()
        st.session_state.opt_results=results; st.session_state.opt_ran=True
        st.success(f"✅ Done! {len(results)} valid results.")

    if not st.session_state.opt_ran: return
    results=st.session_state.opt_results
    if not results: st.warning("No valid results. Try wider ranges or fewer min trades."); return

    desired_acc=float(st.session_state.get("opt_acc",60.0))
    filtered=[r for r in results if r.get("Accuracy%",0)>=desired_acc]
    display=filtered if filtered else results
    if filtered: st.success(f"{len(filtered)} results meet {desired_acc:.0f}%+ accuracy")
    else: st.warning(f"No results at {desired_acc:.0f}%. Showing best available.")

    st.markdown(f"### 🏆 Top {min(len(display),15)} Results — click ✅ Apply to load into sidebar")
    hc=st.columns([3,1,1,1,1,1,1])
    for col,lbl in zip(hc,["Parameters","Accuracy","P&L","Trades","R:R","Score",""]): 
        col.markdown(f'<span style="color:#90cdf4;font-size:11px;font-weight:700">{lbl}</span>',unsafe_allow_html=True)

    for i,res in enumerate(display[:15]):
        # Build param string only from keys that have values
        parts=[]
        if res.get("fast_ema") is not None: parts.append(f"F={int(res['fast_ema'])} S={int(res['slow_ema'])}")
        if res.get("ew_swing_window") is not None: parts.append(f"SwingWin={res['ew_swing_window']}")
        if res.get("sl_points") is not None: parts.append(f"SL={res['sl_points']}")
        if res.get("target_points") is not None: parts.append(f"Tgt={res['target_points']}")
        param_str=" | ".join(parts) if parts else "—"
        acc=float(res.get("Accuracy%",0)); pnl=float(res.get("TotalPnL",0))
        rr=float(res.get("RR",0)); sc=float(res.get("Score",0)); tr=int(res.get("Trades",0))
        ac="#48bb78" if acc>=desired_acc else "#f6ad55"; pc="#48bb78" if pnl>=0 else "#fc8181"
        rc=st.columns([3,1,1,1,1,1,1])
        rc[0].markdown(f'<div style="font-size:12px;color:#e2e8f0;padding:3px">{param_str}</div>',unsafe_allow_html=True)
        rc[1].markdown(f'<div style="text-align:center;font-size:13px;color:{ac};font-weight:700;padding:3px">{acc:.1f}%</div>',unsafe_allow_html=True)
        rc[2].markdown(f'<div style="text-align:center;font-size:13px;color:{pc};font-weight:700;padding:3px">₹{pnl:+.0f}</div>',unsafe_allow_html=True)
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
            st.success(f"✅ Applied: {param_str}")
            st.rerun()

    # Full table
    rdf=pd.DataFrame(display)
    show_cols=["fast_ema","slow_ema","ew_swing_window","sl_points","target_points",
               "Trades","Accuracy%","TotalPnL","AvgWin","AvgLoss","RR","Score"]
    sc2=[c for c in show_cols if c in rdf.columns and rdf[c].notna().any()]
    if sc2:
        for col in ["fast_ema","slow_ema","Trades","ew_swing_window"]:
            if col in rdf.columns: rdf[col]=rdf[col].apply(lambda x:int(x) if pd.notna(x) else x)
        st.dataframe(style_df(rdf[sc2]),use_container_width=True,height=360)
    if not rdf.empty:
        st.download_button("⬇ CSV",rdf.to_csv(index=False).encode(),"opt.csv","text/csv",key="opt_dl")

# ════════════════════════════════════════════════════════════════════
# MAIN  — auto-refresh at bottom after full render
# ════════════════════════════════════════════════════════════════════
def main():
    st.markdown(
        '<div class="hdr">'
        '<div><h1>📈 Smart Investing</h1>'
        '<p>Professional Algorithmic Trading Platform · NSE · BSE · Crypto · Commodities</p></div>'
        '<div style="color:#90cdf4;font-size:12px;text-align:right">EMA · Elliott Wave · Auto-Trading · Dhan Broker</div>'
        '</div>', unsafe_allow_html=True)

    cfg = sidebar()
    tab1,tab2,tab3,tab4 = st.tabs(["🔬 Backtesting","🚀 Live Trading","📚 Trade History","🔧 Optimization"])
    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history(cfg)
    with tab4: tab_optimize(cfg)

    # ── AUTO-REFRESH: happens AFTER full page render — no blank screen ──
    # time.sleep(1.5) here is fine: page is already fully rendered,
    # user sees it for 1.5s then it refreshes with latest live data.
    if _g("running", False):
        time.sleep(1.5)
        st.rerun()

if __name__ == "__main__":
    main()
