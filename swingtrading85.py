# ╔══════════════════════════════════════════════════════════════╗
# ║          SMART INVESTING  –  Algo Trading Platform          ║
# ║  NSE · BSE · Crypto · Gold · Silver  |  Dhan Broker Ready  ║
# ╚══════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, threading, math, warnings, requests
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart Investing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

IST = pytz.timezone("Asia/Kolkata")

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.main{background:#0e1117}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif}
.app-hdr{background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
  padding:14px 24px;border-radius:14px;margin-bottom:14px;
  border:1px solid #2d4a6e;display:flex;align-items:center;justify-content:space-between}
.app-hdr h1{margin:0;color:#e2e8f0;font-size:22px}
.app-hdr p{margin:0;color:#90cdf4;font-size:12px}
.ltp-bar{background:linear-gradient(135deg,#1a1f2e,#252b3d);
  border:1px solid #2d4a6e;border-radius:10px;padding:10px 20px;
  display:flex;align-items:center;gap:20px;margin-bottom:12px}
.ltp-ticker{color:#90cdf4;font-size:12px}
.ltp-price{color:#e2e8f0;font-size:26px;font-weight:700}
.ltp-up{color:#48bb78;font-size:14px;font-weight:600}
.ltp-down{color:#fc8181;font-size:14px;font-weight:600}
.ltp-meta{color:#718096;font-size:11px;margin-left:auto}
.metric-card{background:linear-gradient(135deg,#1a1f2e,#252b3d);
  border:1px solid #2d3748;border-radius:10px;padding:14px;margin:5px 0}
.cfg-box{background:#1a1f2e;border-radius:8px;padding:12px;
  border-left:3px solid #4299e1;margin:8px 0;font-size:13px;
  line-height:1.8;color:#cbd5e0}
.wave-lbl{background:#2d3748;border:1px solid #4a5568;border-radius:6px;
  padding:5px 10px;display:inline-block;margin:3px;font-size:12px}
.log-box{background:#0d1117;border-radius:8px;padding:10px;
  height:200px;overflow-y:auto;
  font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;color:#a0aec0}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent}
.stTabs [data-baseweb="tab"]{background:#1a1f2e;border-radius:8px;
  padding:8px 20px;border:1px solid #2d3748;color:#a0aec0}
.stTabs [aria-selected="true"]{background:#2b6cb0!important;color:white!important}
section[data-testid="stSidebar"]{background:#0d1117}
section[data-testid="stSidebar"] label{color:#a0aec0!important;font-size:12px}
section[data-testid="stSidebar"] .stMarkdown h3{color:#90cdf4;margin-top:14px;font-size:14px}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
TICKERS = {
    "Nifty 50":"^NSEI","BankNifty":"^NSEBANK","Sensex":"^BSESN",
    "BTC":"BTC-USD","ETH":"ETH-USD","Gold":"GC=F","Silver":"SI=F","Custom":None,
}
TF_PERIODS = {
    "1m" :["1d","5d","7d"],
    "5m" :["1d","5d","7d","1mo"],
    "15m":["1d","5d","7d","1mo"],
    "1h" :["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d" :["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
MAX_PERIOD = {"1m":"7d","5m":"60d","15m":"60d","1h":"730d","1d":"max","1wk":"max"}
TF_MINUTES = {"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080}

# ── Session State ─────────────────────────────────────────────────
for k,v in dict(
    live_running=False, live_thread=None, live_position=None,
    live_trades=[], live_data=None, live_ltp=None, live_prev_close=None,
    live_status="STOPPED", live_log=[], live_last_ts=None,
    live_ew={}, backtest_trades=[], backtest_violations=[],
    backtest_df=None, backtest_ran=False,
    dhan_client=None, _rl_ts=0.0,
).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Rate-limited fetch ────────────────────────────────────────────
_rl_lock = threading.Lock()

def _rate_wait():
    with _rl_lock:
        gap = time.time() - st.session_state._rl_ts
        if gap < 1.5:
            time.sleep(1.5 - gap)
        st.session_state._rl_ts = time.time()

def _clean(df):
    if df is None or df.empty: return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize("UTC").tz_convert(IST)
    df.columns = [c.lower() for c in df.columns]
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df[cols].dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def fetch_data(symbol, interval, period):
    """Fetch with max warmup so EMA never shows NaN after gap-up/down."""
    _rate_wait()
    try:
        warmup = MAX_PERIOD.get(interval, period)
        raw = yf.Ticker(symbol).history(period=warmup, interval=interval,
                                         auto_adjust=True, prepost=False)
        df = _clean(raw)
        if df is None or df.empty:
            raw = yf.Ticker(symbol).history(period=period, interval=interval,
                                             auto_adjust=True, prepost=False)
            df = _clean(raw)
        return df
    except Exception as e:
        _log(f"[ERROR] fetch_data: {e}"); return None

def fetch_ltp(symbol):
    try:
        _rate_wait()
        raw = yf.Ticker(symbol).history(period="5d", interval="1d",
                                         auto_adjust=True, prepost=False)
        df = _clean(raw)
        if df is not None and len(df) >= 2:
            return float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        if df is not None and len(df) == 1:
            v = float(df["close"].iloc[-1]); return v, v
    except Exception: pass
    return None, None

# ═══════════════════════════════════════════════════════════════════
# INDICATORS  (TradingView-accurate)
# ═══════════════════════════════════════════════════════════════════

def ema_tv(series: pd.Series, n: int) -> pd.Series:
    """EMA seeded with first-n SMA — exactly matches TradingView."""
    if len(series) < n:
        return pd.Series(np.nan, index=series.index)
    alpha  = 2.0 / (n + 1)
    vals   = series.ffill().values.astype(float)
    out    = np.full(len(vals), np.nan)
    out[n-1] = np.nanmean(vals[:n])
    for i in range(n, len(vals)):
        out[i] = alpha * (vals[i] if not np.isnan(vals[i]) else out[i-1]) + (1-alpha)*out[i-1]
    return pd.Series(out, index=series.index)

def atr_s(df, n=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return ema_tv(tr, n)

def add_indicators(df, fast, slow):
    if df is None or df.empty: return df
    df = df.copy()
    df["ema_fast"]   = ema_tv(df["close"], fast)
    df["ema_slow"]   = ema_tv(df["close"], slow)
    df["atr"]        = atr_s(df)
    df["cross_up"]   = (df["ema_fast"]>df["ema_slow"])&(df["ema_fast"].shift(1)<=df["ema_slow"].shift(1))
    df["cross_down"] = (df["ema_fast"]<df["ema_slow"])&(df["ema_fast"].shift(1)>=df["ema_slow"].shift(1))
    return df

def ema_angle(ema, lb=3):
    v = ema.dropna()
    if len(v) < lb+1: return 0.0
    base = v.iloc[-lb]
    if base == 0: return 0.0
    return abs(math.degrees(math.atan((v.iloc[-1]-base)/base*100/lb)))

# ═══════════════════════════════════════════════════════════════════
# ELLIOTT WAVE DETECTION
# ═══════════════════════════════════════════════════════════════════

def find_swings(df, win=5):
    h_arr, l_arr = df["high"].values, df["low"].values
    pts = []
    for i in range(win, len(df)-win):
        lo, hi = max(0,i-win), min(len(df),i+win+1)
        if h_arr[i] == h_arr[lo:hi].max():
            pts.append({"dt":df.index[i],"price":h_arr[i],"idx":i,"type":"H"})
        if l_arr[i] == l_arr[lo:hi].min():
            pts.append({"dt":df.index[i],"price":l_arr[i],"idx":i,"type":"L"})
    pts.sort(key=lambda x:x["idx"])
    clean = []
    for p in pts:
        if not clean: clean.append(p); continue
        if clean[-1]["type"] == p["type"]:
            if (p["type"]=="H" and p["price"]>clean[-1]["price"]) or \
               (p["type"]=="L" and p["price"]<clean[-1]["price"]):
                clean[-1] = p
        else:
            clean.append(p)
    return clean

def detect_ew(df):
    cp = float(df["close"].iloc[-1]) if len(df) else 0
    if len(df) < 25:
        return {"status":"Insufficient data","direction":"UNCLEAR","waves":[],
                "current_wave":"—","wave_detail":{},"next_targets":{},
                "after_msg":"Need more candles","swing_points":[],"current_price":cp}
    swings = find_swings(df, win=max(3, len(df)//40))
    if len(swings) < 4:
        return {"status":"Forming","direction":"UNCLEAR","waves":[],
                "current_wave":"Accumulating","wave_detail":{},"next_targets":{},
                "after_msg":"Wait for clearer pattern","swing_points":swings,"current_price":cp}

    # Try 5-wave impulse
    if len(swings) >= 6:
        for start,label,sign in [("L","BULLISH",1),("H","BEARISH",-1)]:
            s = swings[-6:]
            if s[0]["type"] != start: continue
            p = [pt["price"] for pt in s]
            w1 = sign*(p[1]-p[0]); w3 = sign*(p[3]-p[2])
            w2r = (p[2]-p[1])/(p[1]-p[0]) if p[1]-p[0] else 0
            if w1>0 and w3>0 and abs(w2r)<1.0 and w3>=w1*0.618:
                waves = [{"label":str(n+1),"start":s[n],"end":s[n+1]} for n in range(5)]
                base = p[4]
                projs = {f"W5 @ {k}% W1":round(base+sign*w1*f,2)
                         for k,f in [(61.8,.618),(100,1.0),(161.8,1.618)]}
                return {"status":f"5-Wave Impulse ({'Bullish' if sign==1 else 'Bearish'})",
                        "direction":label,"waves":waves,
                        "current_wave":"Wave 5 (forming)","wave_detail":{
                            "W1":f"{p[0]:.2f}→{p[1]:.2f}",
                            "W2":f"{p[1]:.2f}→{p[2]:.2f} ({abs(w2r)*100:.1f}% ret)",
                            "W3":f"{p[2]:.2f}→{p[3]:.2f} ({w3/w1*100:.0f}% of W1)",
                            "W4":f"{p[3]:.2f}→{p[4]:.2f}",
                            "W5":f"{p[4]:.2f}→{p[5]:.2f} (live)",
                        },"next_targets":projs,"after_msg":"Expect ABC correction after W5",
                        "swing_points":swings,"current_price":cp,
                        "trade_bias":"BUY" if sign==1 else "SELL"}
    # Try ABC correction
    if len(swings) >= 4:
        for start,label,sign in [("H","CORRECTIVE_DOWN",-1),("L","CORRECTIVE_UP",1)]:
            s = swings[-4:]
            if s[0]["type"] != start: continue
            p = [pt["price"] for pt in s]
            a_mv = sign*(p[0]-p[1])
            br   = (p[2]-p[1])/(p[1]-p[0]) if p[1]-p[0] else 0
            return {"status":f"ABC Correction ({'Bearish' if sign==-1 else 'Bullish'})",
                    "direction":label,"waves":[{"label":lb,"start":s[i],"end":s[i+1]}
                                               for i,lb in enumerate(["A","B","C"])],
                    "current_wave":"Wave C (forming)","wave_detail":{
                        "A":f"{p[0]:.2f}→{p[1]:.2f}",
                        "B":f"{p[1]:.2f}→{p[2]:.2f} ({abs(br)*100:.1f}% ret)",
                        "C":f"{p[2]:.2f}→{p[3]:.2f} (live)",
                    },"next_targets":{
                        "C @ 61.8% A":round(p[2]-sign*a_mv*0.618,2),
                        "C = A":       round(p[2]-sign*a_mv,2),
                    },"after_msg":"Expect new impulse after ABC",
                    "swing_points":swings,"current_price":cp,
                    "trade_bias":"SELL" if sign==-1 else "BUY"}

    return {"status":"Structure Forming","direction":"UNCLEAR","waves":[],
            "current_wave":"Accumulating","wave_detail":{},"next_targets":{},
            "after_msg":"Insufficient pivots","swing_points":swings,"current_price":cp}

# ═══════════════════════════════════════════════════════════════════
# SL / TARGET CALC
# ═══════════════════════════════════════════════════════════════════

def calc_sl_tgt(entry, trade_type, row, cfg):
    atr  = float(row.get("atr", entry*0.01) or entry*0.01)
    if np.isnan(atr) or atr==0: atr = entry*0.01
    sl_t  = cfg.get("sl_type","Custom Points")
    tg_t  = cfg.get("target_type","Custom Points")
    slp   = float(cfg.get("sl_points",10))
    tgp   = float(cfg.get("target_points",20))
    sign  = 1 if trade_type=="buy" else -1
    sl_dist = slp  # default

    if sl_t=="Custom Points":          sl=entry-sign*slp
    elif sl_t=="ATR Based":            sl=entry-sign*atr*float(cfg.get("atr_sl_mult",1.5))
    else:                              sl=entry-sign*slp
    sl_dist = abs(entry-sl)

    if tg_t=="Custom Points":          tgt=entry+sign*tgp
    elif tg_t=="ATR Based Target":     tgt=entry+sign*atr*float(cfg.get("atr_tgt_mult",3.0))
    elif tg_t=="Risk-Reward Based":    tgt=entry+sign*sl_dist*float(cfg.get("rr_ratio",2.0))
    else:                              tgt=entry+sign*tgp
    return round(sl,4), round(tgt,4)

# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════

def ema_signal(df, idx, cfg):
    if idx<1 or idx>=len(df): return None,None
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es=row.get("ema_fast",np.nan),row.get("ema_slow",np.nan)
    pf,ps=prev.get("ema_fast",np.nan),prev.get("ema_slow",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return None,None
    if cfg.get("use_angle_filter",False):
        ang=ema_angle(df["ema_fast"].iloc[:idx+1])
        if ang < float(cfg.get("min_ema_angle",0)): return None,None
    ctype=cfg.get("crossover_type","Simple Crossover")
    body=abs(row["close"]-row["open"])
    def ok():
        if ctype=="Simple Crossover": return True
        if ctype=="Custom Candle Size": return body>=float(cfg.get("custom_candle_size",10))
        if ctype=="ATR Based Candle Size":
            a=float(row.get("atr",0) or 0); return a==0 or body>=a
        return True
    f,s=int(cfg.get("fast_ema",9)),int(cfg.get("slow_ema",15))
    if ef>es and pf<=ps and ok(): return "buy", f"EMA({f}) crossed ABOVE EMA({s})"
    if ef<es and pf>=ps and ok(): return "sell",f"EMA({f}) crossed BELOW EMA({s})"
    return None,None

def ema_rev_cross(df, idx, ptype):
    if idx<1: return False
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es=row.get("ema_fast",np.nan),row.get("ema_slow",np.nan)
    pf,ps=prev.get("ema_fast",np.nan),prev.get("ema_slow",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return False
    if ptype=="buy"  and ef<es and pf>=ps: return True
    if ptype=="sell" and ef>es and pf<=ps: return True
    return False

def _fdt(dt):
    if hasattr(dt,"strftime"): return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)

# ═══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_backtest(df, cfg):
    if df is None or df.empty: return [],[]
    strat = cfg.get("strategy","EMA Crossover")
    fast  = int(cfg.get("fast_ema",9))
    slow  = int(cfg.get("slow_ema",15))
    qty   = int(cfg.get("quantity",1))
    sl_t  = cfg.get("sl_type","Custom Points")
    tg_t  = cfg.get("target_type","Custom Points")
    df    = add_indicators(df.copy(), fast, slow)

    trades=[]; violations=[]; pos=None; pending=None; tnum=0

    for i in range(1, len(df)):
        row=df.iloc[i]

        # ── exit check ──
        if pos:
            sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
            # trailing SL update
            if sl_t=="Trailing SL":
                trail=float(cfg.get("sl_points",10))
                if pt=="buy":
                    ns=row["high"]-trail
                    if ns>sl: sl=pos["sl"]=ns
                else:
                    ns=row["low"]+trail
                    if ns<sl: sl=pos["sl"]=ns
            # trailing target (display only)
            if tg_t=="Trailing Target":
                tp=float(cfg.get("target_points",20))
                if pt=="buy":
                    nt=row["high"]+tp
                    if nt>pos["tgt"]: pos["tgt"]=nt
                else:
                    nt=row["low"]-tp
                    if nt<pos["tgt"]: pos["tgt"]=nt

            ema_ex=(sl_t=="Reverse EMA Crossover" or tg_t=="EMA Crossover") and ema_rev_cross(df,i,pt)
            ep=None; er=None; viol=False

            if pt=="buy":
                if row["low"]<=sl:
                    ep=sl; er=f"SL hit (Low {row['low']:.2f}≤SL {sl:.2f})"
                    if tg_t!="Trailing Target" and row["high"]>=tgt: viol=True
                elif tg_t!="Trailing Target" and row["high"]>=tgt:
                    ep=tgt; er=f"Target hit (High {row['high']:.2f}≥Tgt {tgt:.2f})"
                elif ema_ex:
                    ep=row["open"]; er="EMA Reverse Crossover exit"
            else:
                if row["high"]>=sl:
                    ep=sl; er=f"SL hit (High {row['high']:.2f}≥SL {sl:.2f})"
                    if tg_t!="Trailing Target" and row["low"]<=tgt: viol=True
                elif tg_t!="Trailing Target" and row["low"]<=tgt:
                    ep=tgt; er=f"Target hit (Low {row['low']:.2f}≤Tgt {tgt:.2f})"
                elif ema_ex:
                    ep=row["open"]; er="EMA Reverse Crossover exit"

            if ep is None and i==len(df)-1:
                ep=row["close"]; er="End of data"

            if ep is not None:
                pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                t={"Trade #":tnum,"Type":pt.upper(),
                   "Entry Time":_fdt(pos["entry_time"]),"Exit Time":_fdt(row.name),
                   "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                   "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
                   "Candle High":round(row["high"],4),"Candle Low":round(row["low"],4),
                   "Entry Reason":pos["reason"],"Exit Reason":er,
                   "PnL":pnl,"Qty":qty,"SL/Tgt Violated":viol}
                trades.append(t)
                if viol: violations.append(t)
                pos=None

        # ── pending entry ──
        if pos is None and pending:
            sig,rsn=pending; pending=None
            ep=float(row["open"])
            sl,tgt=calc_sl_tgt(ep,sig,row,cfg)
            tnum+=1
            pos={"type":sig,"entry":ep,"entry_time":row.name,
                 "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":rsn}

        # ── new signal ──
        if pos is None:
            if strat=="EMA Crossover":
                sig,rsn=ema_signal(df,i,cfg)
                if sig: pending=(sig,rsn)
            elif strat=="Simple Buy":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"buy",row,cfg)
                tnum+=1
                pos={"type":"buy","entry":ep,"entry_time":row.name,
                     "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Buy"}
            elif strat=="Simple Sell":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"sell",row,cfg)
                tnum+=1
                pos={"type":"sell","entry":ep,"entry_time":row.name,
                     "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Sell"}
            elif strat=="Elliott Wave" and i>=20:
                ew=detect_ew(df.iloc[:i+1])
                bias=ew.get("trade_bias","")
                if bias:
                    sig="buy" if bias=="BUY" else "sell"
                    pending=(sig,f"Elliott Wave: {ew.get('current_wave','')}")

    return trades, violations

# ═══════════════════════════════════════════════════════════════════
# DHAN BROKER
# ═══════════════════════════════════════════════════════════════════

def init_dhan(cid, tok):
    try:
        from dhanhq import dhanhq
        return dhanhq(cid, tok)
    except ImportError:
        st.warning("Install dhanhq: pip install dhanhq"); return None
    except Exception as e:
        st.error(f"Dhan error: {e}"); return None

def register_ip(cid, tok):
    try:
        ip=requests.get("https://api.ipify.org?format=json",timeout=5).json().get("ip","?")
        st.info(f"📡 Your IP: **{ip}** — Whitelist in Dhan Console → Profile → IP Whitelist (SEBI requirement)")
        return ip
    except Exception:
        st.warning("Could not fetch IP. Whitelist manually in Dhan Console."); return None

def place_order(dhan, cfg, sig, ltp, is_exit=False):
    if dhan is None: return {"error":"Not connected"}
    try:
        if cfg.get("options_trading",False):
            fno = cfg.get("fno_exchange","NSE_FNO")
            qty = int(cfg.get("options_qty",65))
            ot  = cfg.get("options_exit_type" if is_exit else "options_entry_type","MARKET")
            px  = round(ltp,2) if ot=="LIMIT" else 0
            sid = cfg.get("ce_security_id") if sig=="buy" else cfg.get("pe_security_id")
            txn = "SELL" if is_exit else "BUY"
            return dhan.place_order(transactionType=txn,exchangeSegment=fno,
                productType="INTRADAY",orderType=ot,validity="DAY",
                securityId=str(sid),quantity=qty,price=px,triggerPrice=0)
        else:
            prod_map={"INTRADAY":getattr(dhan,"INTRADAY","INTRADAY"),
                      "DELIVERY":getattr(dhan,"DELIVERY","CNC")}
            exc_map ={"NSE":getattr(dhan,"NSE","NSE"),
                      "BSE":getattr(dhan,"BSE","BSE")}
            ot   = cfg.get("exit_order_type" if is_exit else "entry_order_type","MARKET")
            prod = prod_map.get(cfg.get("product_type","INTRADAY"))
            exc  = exc_map.get(cfg.get("exchange","NSE"))
            px   = round(ltp,2) if ot=="LIMIT" else 0
            txn  = (getattr(dhan,"SELL","SELL") if is_exit else
                    (getattr(dhan,"BUY","BUY") if sig=="buy" else getattr(dhan,"SELL","SELL")))
            return dhan.place_order(security_id=str(cfg.get("security_id","1594")),
                exchange_segment=exc,transaction_type=txn,
                quantity=int(cfg.get("dhan_qty",1)),
                order_type=getattr(dhan,"MARKET" if ot=="MARKET" else "LIMIT","MARKET"),
                product_type=prod,price=px)
    except Exception as e:
        return {"error":str(e)}

# ═══════════════════════════════════════════════════════════════════
# LIVE TRADING THREAD
# ═══════════════════════════════════════════════════════════════════

def _log(msg):
    ts=datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.live_log.append(f"[{ts}] {msg}")
    if len(st.session_state.live_log)>300:
        st.session_state.live_log=st.session_state.live_log[-300:]

def live_thread(cfg, symbol):
    _log(f"▶ STARTED | {symbol} | {cfg['timeframe']} | {cfg['strategy']}")
    tf_min  = TF_MINUTES.get(cfg["timeframe"],5)
    fast    = int(cfg.get("fast_ema",9))
    slow    = int(cfg.get("slow_ema",15))
    strat   = cfg.get("strategy","EMA Crossover")
    sl_t    = cfg.get("sl_type","Custom Points")
    tg_t    = cfg.get("target_type","Custom Points")
    qty     = int(cfg.get("quantity",1))
    dhan    = st.session_state.dhan_client
    en_dhan = cfg.get("enable_dhan",False)
    pending = None
    last_sig_candle = None
    last_bdy_min    = -1

    while st.session_state.live_running:
        try:
            df  = fetch_data(symbol, cfg["timeframe"], cfg["period"])
            now = datetime.now(IST)
            if df is None or df.empty:
                _log("[WARN] No data, retrying…"); time.sleep(1.5); continue

            df  = add_indicators(df, fast, slow)
            ltp = float(df["close"].iloc[-1])
            st.session_state.live_data    = df
            st.session_state.live_ltp     = ltp
            st.session_state.live_last_ts = now

            if len(df)>=20:
                st.session_state.live_ew = detect_ew(df)

            pos = st.session_state.live_position

            # ── SL/Target tick check ──
            if pos:
                sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
                # trailing SL
                if sl_t=="Trailing SL":
                    trail=float(cfg.get("sl_points",10))
                    if pt=="buy":
                        ns=ltp-trail
                        if ns>sl: st.session_state.live_position["sl"]=ns; sl=ns
                    else:
                        ns=ltp+trail
                        if ns<sl: st.session_state.live_position["sl"]=ns; sl=ns
                # trailing tgt (display only)
                if tg_t=="Trailing Target":
                    tp=float(cfg.get("target_points",20))
                    if pt=="buy":
                        nt=ltp+tp
                        if nt>tgt: st.session_state.live_position["tgt"]=nt
                    else:
                        nt=ltp-tp
                        if nt<tgt: st.session_state.live_position["tgt"]=nt

                ep=None; er=None
                # conservative: SL first then target
                if pt=="buy":
                    if ltp<=sl:   ep=sl;  er=f"SL hit (LTP {ltp:.2f}≤SL {sl:.2f})"
                    elif tg_t!="Trailing Target" and ltp>=tgt: ep=tgt; er=f"Target hit (LTP {ltp:.2f}≥Tgt {tgt:.2f})"
                else:
                    if ltp>=sl:   ep=sl;  er=f"SL hit (LTP {ltp:.2f}≥SL {sl:.2f})"
                    elif tg_t!="Trailing Target" and ltp<=tgt: ep=tgt; er=f"Target hit (LTP {ltp:.2f}≤Tgt {tgt:.2f})"
                if ep is None and (sl_t=="Reverse EMA Crossover" or tg_t=="EMA Crossover"):
                    if ema_rev_cross(df,len(df)-1,pt):
                        ep=ltp; er="EMA Reverse Crossover exit"

                if ep is not None:
                    pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                    t={"Trade #":len(st.session_state.live_trades)+1,"Type":pt.upper(),
                       "Entry Time":_fdt(pos["entry_time"]),"Exit Time":now.strftime("%Y-%m-%d %H:%M:%S"),
                       "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                       "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
                       "Entry Reason":pos["reason"],"Exit Reason":er,
                       "PnL":pnl,"Qty":qty,"Source":"🚀 Live"}
                    st.session_state.live_trades.append(t)
                    st.session_state.live_position=None
                    if en_dhan and dhan:
                        resp=place_order(dhan,cfg,pt,ltp,is_exit=True)
                        _log(f"EXIT order → {resp}")
                    _log(f"✖ EXIT {pt.upper()} @{ep:.2f} | {er} | PnL:{pnl:+.2f}")

            # ── Entry logic ──
            if st.session_state.live_position is None:
                cd_ok=True
                if cfg.get("cooldown_enabled",True) and st.session_state.live_trades:
                    try:
                        le=datetime.strptime(st.session_state.live_trades[-1]["Exit Time"],"%Y-%m-%d %H:%M:%S")
                        le=IST.localize(le)
                        if (now-le).total_seconds()<int(cfg.get("cooldown_seconds",5)):
                            cd_ok=False
                    except Exception: pass

                if pending and cd_ok:
                    sig,rsn=pending; pending=None
                    ep=float(df["open"].iloc[-1])
                    sl,tgt=calc_sl_tgt(ep,sig,df.iloc[-1],cfg)
                    st.session_state.live_position={
                        "type":sig,"entry":ep,"entry_time":now,
                        "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":rsn}
                    if en_dhan and dhan:
                        resp=place_order(dhan,cfg,sig,ep,is_exit=False)
                        _log(f"ENTRY order → {resp}")
                    _log(f"▲ ENTRY {sig.upper()} @{ep:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")
                else:
                    tm=now.hour*60+now.minute
                    at_bdy=(tm%tf_min==0) and (tm!=last_bdy_min)
                    if at_bdy and cd_ok:
                        last_bdy_min=tm
                        if strat=="EMA Crossover":
                            sig,rsn=ema_signal(df,len(df)-1,cfg)
                            if sig and df.index[-1]!=last_sig_candle:
                                last_sig_candle=df.index[-1]
                                pending=(sig,rsn)
                                _log(f"◆ SIGNAL {sig.upper()} on {_fdt(df.index[-1])} → enter next candle")
                        elif strat=="Simple Buy" and cd_ok:
                            sl,tgt=calc_sl_tgt(ltp,"buy",df.iloc[-1],cfg)
                            st.session_state.live_position={
                                "type":"buy","entry":ltp,"entry_time":now,
                                "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Buy"}
                            _log(f"▲ Simple BUY @{ltp:.2f}")
                        elif strat=="Simple Sell" and cd_ok:
                            sl,tgt=calc_sl_tgt(ltp,"sell",df.iloc[-1],cfg)
                            st.session_state.live_position={
                                "type":"sell","entry":ltp,"entry_time":now,
                                "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Sell"}
                            _log(f"▼ Simple SELL @{ltp:.2f}")
                        elif strat=="Elliott Wave":
                            ew=st.session_state.live_ew
                            bias=ew.get("trade_bias","")
                            if bias and cd_ok:
                                s2="buy" if bias=="BUY" else "sell"
                                pending=(s2,f"EW: {ew.get('current_wave','')} ({ew.get('direction','')})")
                                _log(f"◆ EW SIGNAL {s2.upper()} → enter next candle")
            time.sleep(1.5)
        except Exception as e:
            _log(f"[ERR] {e}"); time.sleep(1.5)
    _log("■ STOPPED.")

# ═══════════════════════════════════════════════════════════════════
# CHARTING
# ═══════════════════════════════════════════════════════════════════

def make_chart(df, trades=None, title="", fast=9, slow=15, live_pos=None, h=520):
    if df is None or df.empty: return go.Figure()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
        vertical_spacing=0.04,row_heights=[0.78,0.22])
    fig.add_trace(go.Candlestick(x=df.index,open=df["open"],high=df["high"],
        low=df["low"],close=df["close"],name="Price",
        increasing_line_color="#48bb78",decreasing_line_color="#fc8181",
        increasing_fillcolor="#48bb78",decreasing_fillcolor="#fc8181"),row=1,col=1)
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_fast"],name=f"EMA({fast})",
            line=dict(color="#f6ad55",width=1.5),opacity=0.9),row=1,col=1)
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_slow"],name=f"EMA({slow})",
            line=dict(color="#76e4f7",width=1.5),opacity=0.9),row=1,col=1)
    if "volume" in df.columns:
        vc=["#48bb78" if c>=o else "#fc8181" for c,o in zip(df["close"],df["open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Vol",
            marker_color=vc,opacity=0.5),row=2,col=1)
    if trades:
        for t in trades:
            try:
                et=pd.to_datetime(t["Entry Time"]); xt=pd.to_datetime(t["Exit Time"])
                c="#48bb78" if t["Type"]=="BUY" else "#fc8181"
                sym="triangle-up" if t["Type"]=="BUY" else "triangle-down"
                pc="#48bb78" if t.get("PnL",0)>=0 else "#fc8181"
                fig.add_trace(go.Scatter(x=[et],y=[t["Entry Price"]],mode="markers+text",
                    marker=dict(symbol=sym,size=11,color=c),
                    text=[f"E:{t['Entry Price']:.0f}"],textposition="top center",
                    textfont=dict(size=8,color=c),name=f"#{t.get('Trade #','')}",showlegend=False),row=1,col=1)
                fig.add_trace(go.Scatter(x=[xt],y=[t["Exit Price"]],mode="markers+text",
                    marker=dict(symbol="x",size=9,color=pc),
                    text=[f"X:{t['Exit Price']:.0f}"],textposition="bottom center",
                    textfont=dict(size=8,color=pc),showlegend=False),row=1,col=1)
                fig.add_shape(type="line",x0=et,x1=xt,y0=t["SL"],y1=t["SL"],
                    line=dict(color="#fc8181",width=1,dash="dot"),row=1,col=1)
                fig.add_shape(type="line",x0=et,x1=xt,y0=t["Target"],y1=t["Target"],
                    line=dict(color="#48bb78",width=1,dash="dot"),row=1,col=1)
            except Exception: pass
    if live_pos:
        try:
            c="#48bb78" if live_pos["type"]=="buy" else "#fc8181"
            sym="triangle-up" if live_pos["type"]=="buy" else "triangle-down"
            fig.add_trace(go.Scatter(x=[live_pos["entry_time"]],y=[live_pos["entry"]],
                mode="markers+text",name="Live Entry",
                marker=dict(symbol=sym,size=16,color=c,line=dict(color="white",width=2)),
                text=[f"ENTRY\n{live_pos['entry']:.2f}"],textposition="top center"),row=1,col=1)
            fig.add_hline(y=live_pos["sl"],line=dict(color="#fc8181",width=2,dash="dot"),
                annotation_text=f"SL {live_pos['sl']:.2f}",
                annotation_font=dict(color="#fc8181"),row=1,col=1)
            fig.add_hline(y=live_pos["tgt"],line=dict(color="#48bb78",width=2,dash="dot"),
                annotation_text=f"Tgt {live_pos['tgt']:.2f}",
                annotation_font=dict(color="#48bb78"),row=1,col=1)
        except Exception: pass
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0",size=11),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0,r=0,t=36,b=0),height=h,xaxis_rangeslider_visible=False,
        xaxis2=dict(showgrid=True,gridcolor="#1a1f2e"),
        yaxis=dict(showgrid=True,gridcolor="#1a1f2e"),
        yaxis2=dict(showgrid=True,gridcolor="#1a1f2e"))
    if title: fig.update_layout(title=dict(text=title,font=dict(size=14,color="#90cdf4")))
    return fig

def make_ew_chart(df, ew, fast, slow):
    fig=make_chart(df.tail(150),title="Elliott Wave Structure",fast=fast,slow=slow,h=440)
    wcolors={"1":"#f6ad55","2":"#fc8181","3":"#48bb78","4":"#e9d8a6","5":"#76e4f7",
             "A":"#fc8181","B":"#48bb78","C":"#fc8181"}
    for w in ew.get("waves",[]):
        s=w["start"]; e=w["end"]; c=wcolors.get(w["label"],"#a0aec0")
        fig.add_trace(go.Scatter(x=[s["dt"],e["dt"]],y=[s["price"],e["price"]],
            mode="lines+markers+text",line=dict(color=c,width=2.5),
            marker=dict(size=8,color=c),
            text=["",f"W{w['label']}"],textposition="top center",
            textfont=dict(color=c,size=12),name=f"W{w['label']}"),row=1,col=1)
    return fig

def pnl_chart(trades):
    pnls=[t.get("PnL",0) for t in trades]
    cum=[0]+list(np.cumsum(pnls))
    c="#48bb78" if cum[-1]>=0 else "#fc8181"
    fig=go.Figure(go.Scatter(x=list(range(len(cum))),y=cum,mode="lines+markers",
        fill="tozeroy",fillcolor="rgba(72,187,120,0.1)" if c=="#48bb78" else "rgba(252,129,129,0.1)",
        line=dict(color=c,width=2),name="Cumulative P&L"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
        height=260,margin=dict(l=0,r=0,t=20,b=0),
        xaxis_title="Trade #",yaxis_title="P&L",font=dict(color="#e2e8f0"))
    return fig

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def sidebar():
    cfg={}
    with st.sidebar:
        st.markdown("## 📈 Smart Investing")
        st.markdown("---")

        # Instrument
        st.markdown("### 🎯 Instrument")
        tn=st.selectbox("Ticker",list(TICKERS.keys()),key="s_tn")
        if tn=="Custom":
            sym=st.text_input("Symbol (e.g. RELIANCE.NS)","RELIANCE.NS",key="s_cust").strip()
        else:
            sym=TICKERS[tn]
        cfg.update(ticker_name=tn,symbol=sym)

        # Timeframe
        st.markdown("### ⏱ Timeframe")
        tf=st.selectbox("Interval",list(TF_PERIODS.keys()),index=2,key="s_tf")
        periods=TF_PERIODS[tf]
        period=st.selectbox("Period",periods,index=min(1,len(periods)-1),key="s_period")
        cfg.update(timeframe=tf,period=period)

        # Strategy
        st.markdown("### 🧠 Strategy")
        strat=st.selectbox("Strategy",["EMA Crossover","Simple Buy","Simple Sell","Elliott Wave"],key="s_strat")
        cfg["strategy"]=strat
        if strat=="EMA Crossover":
            c1,c2=st.columns(2)
            fe=c1.number_input("Fast EMA",1,500,9,key="s_fe")
            se=c2.number_input("Slow EMA",1,500,15,key="s_se")
            cfg.update(fast_ema=fe,slow_ema=se)
            uang=st.checkbox("Angle Filter",False,key="s_uang")
            mang=st.number_input("Min Angle°",0.0,90.0,0.0,0.5,key="s_mang") if uang else 0.0
            cfg.update(use_angle_filter=uang,min_ema_angle=mang)
            ct=st.selectbox("Crossover Type",["Simple Crossover","Custom Candle Size","ATR Based Candle Size"],key="s_ct")
            cfg["crossover_type"]=ct
            if ct=="Custom Candle Size":
                cfg["custom_candle_size"]=st.number_input("Min Body Size",0.0,value=10.0,key="s_cs")
        else:
            cfg.update(fast_ema=9,slow_ema=15,use_angle_filter=False,
                       min_ema_angle=0.0,crossover_type="Simple Crossover",custom_candle_size=10.0)

        # Quantity
        st.markdown("### 📦 Quantity")
        cfg["quantity"]=st.number_input("Qty",1,1000000,1,key="s_qty")

        # SL
        st.markdown("### 🛡 Stop Loss")
        sl_t=st.selectbox("SL Type",["Custom Points","ATR Based","Trailing SL",
                                      "Reverse EMA Crossover","Risk-Reward Based"],key="s_slt")
        cfg["sl_type"]=sl_t
        if sl_t=="ATR Based":
            cfg["atr_sl_mult"]=st.number_input("ATR Mult",0.1,10.0,1.5,0.1,key="s_asm")
            cfg["sl_points"]=10.0
        else:
            cfg["sl_points"]=st.number_input("SL Points",0.1,value=10.0,step=0.5,key="s_slp")
            cfg["atr_sl_mult"]=1.5

        # Target
        st.markdown("### 🎯 Target")
        tg_t=st.selectbox("Target Type",["Custom Points","ATR Based Target","Trailing Target",
                                          "EMA Crossover","Risk-Reward Based"],key="s_tgt")
        cfg["target_type"]=tg_t
        if tg_t=="ATR Based Target":
            cfg["atr_tgt_mult"]=st.number_input("ATR Mult Tgt",0.1,20.0,3.0,0.1,key="s_atm")
            cfg["target_points"]=20.0
        elif tg_t=="Risk-Reward Based":
            cfg["rr_ratio"]=st.number_input("R:R Ratio",0.1,20.0,2.0,0.1,key="s_rr")
            cfg["target_points"]=20.0; cfg["atr_tgt_mult"]=3.0
        elif tg_t=="Trailing Target":
            cfg["target_points"]=st.number_input("Trail Dist",0.1,value=20.0,step=0.5,key="s_tgp")
            cfg["atr_tgt_mult"]=3.0
            st.caption("ℹ️ Trailing Target shown on chart but won't auto-exit")
        else:
            cfg["target_points"]=st.number_input("Target Pts",0.1,value=20.0,step=0.5,key="s_tgp2")
            cfg["atr_tgt_mult"]=3.0
        cfg.setdefault("rr_ratio",2.0)

        # Trade controls
        st.markdown("### ⚙️ Controls")
        cd_en=st.checkbox("Cooldown Between Trades",True,key="s_cdn")
        cd_s=st.number_input("Cooldown (s)",1,3600,5,key="s_cds") if cd_en else 5
        no_ovlp=st.checkbox("Prevent Overlapping Trades",True,key="s_novlp")
        cfg.update(cooldown_enabled=cd_en,cooldown_seconds=cd_s,no_overlap=no_ovlp)

        # Dhan
        st.markdown("### 🏦 Dhan Broker")
        en_dhan=st.checkbox("Enable Dhan Broker",False,key="s_dhan")
        cfg["enable_dhan"]=en_dhan
        if en_dhan:
            cid=st.text_input("Client ID",key="s_cid")
            tok=st.text_input("Access Token",key="s_tok",type="password")
            cfg.update(dhan_client_id=cid,dhan_access_token=tok)
            if st.button("🔗 Connect & Register IP",use_container_width=True,key="s_conn"):
                with st.spinner("Connecting…"):
                    cl=init_dhan(cid,tok)
                    if cl:
                        st.session_state.dhan_client=cl
                        register_ip(cid,tok)
                        st.success("✅ Connected!")
                    else: st.error("❌ Failed")
            opts=st.checkbox("Options Trading",False,key="s_opts")
            cfg["options_trading"]=opts
            if opts:
                cfg["fno_exchange"]=st.selectbox("FNO Exc",["NSE_FNO","BSE_FNO"],key="s_fnoe")
                cfg["ce_security_id"]=st.text_input("CE Security ID",key="s_ceid")
                cfg["pe_security_id"]=st.text_input("PE Security ID",key="s_peid")
                cfg["options_qty"]=st.number_input("Options Qty",1,value=65,key="s_oqty")
                cfg["options_entry_type"]=st.selectbox("Entry Order",["MARKET","LIMIT"],key="s_oent")
                cfg["options_exit_type"] =st.selectbox("Exit Order", ["MARKET","LIMIT"],key="s_oext")
            else:
                cfg["product_type"]    =st.selectbox("Product",["INTRADAY","DELIVERY"],key="s_prod")
                cfg["exchange"]        =st.selectbox("Exchange",["NSE","BSE"],key="s_exc")
                cfg["security_id"]     =st.text_input("Security ID","1594",key="s_sid")
                cfg["dhan_qty"]        =st.number_input("Order Qty",1,value=1,key="s_dqty")
                cfg["entry_order_type"]=st.selectbox("Entry Order",["LIMIT","MARKET"],key="s_eord")
                cfg["exit_order_type"] =st.selectbox("Exit Order", ["MARKET","LIMIT"],key="s_xord")
    return cfg

# ═══════════════════════════════════════════════════════════════════
# HELPER WIDGETS
# ═══════════════════════════════════════════════════════════════════

def ltp_banner(cfg):
    ltp=st.session_state.live_ltp; prev=st.session_state.live_prev_close
    if ltp is None:
        ltp,prev=fetch_ltp(cfg.get("symbol","^NSEI"))
        st.session_state.live_ltp=ltp; st.session_state.live_prev_close=prev
    ltp=ltp or 0.0; prev=prev or ltp
    chg=ltp-prev; pct=chg/prev*100 if prev else 0
    arrow="▲" if chg>=0 else "▼"; cls="ltp-up" if chg>=0 else "ltp-down"
    ts_str=datetime.now(IST).strftime("%H:%M:%S IST")
    st.markdown(f"""
    <div class="ltp-bar">
      <div>
        <div class="ltp-ticker">📈 {cfg.get('ticker_name','—')} · {cfg.get('symbol','')}</div>
        <div class="ltp-price">₹ {ltp:,.2f}</div>
      </div>
      <div class="{cls}">{arrow} {abs(chg):.2f} ({abs(pct):.2f}%)</div>
      <div class="ltp-meta">{cfg.get('timeframe','')} · {cfg.get('period','')} · {cfg.get('strategy','')} | {ts_str}</div>
    </div>""",unsafe_allow_html=True)

def cfg_box(cfg):
    st.markdown(f"""
    <div class="cfg-box">
      <b>📌 Ticker:</b> {cfg.get('ticker_name','—')} ({cfg.get('symbol','—')})
      &nbsp;|&nbsp;<b>⏱ TF:</b> {cfg.get('timeframe','—')}
      &nbsp;|&nbsp;<b>📅 Period:</b> {cfg.get('period','—')}<br>
      <b>🧠 Strategy:</b> {cfg.get('strategy','—')}
      &nbsp;|&nbsp;<b>📦 Qty:</b> {cfg.get('quantity',1)}<br>
      <b>⚡ EMA Fast:</b> {cfg.get('fast_ema',9)}
      &nbsp;|&nbsp;<b>⚡ EMA Slow:</b> {cfg.get('slow_ema',15)}
      &nbsp;|&nbsp;<b>Crossover:</b> {cfg.get('crossover_type','—')}<br>
      <b>🛡 SL:</b> {cfg.get('sl_type','—')} ({cfg.get('sl_points',10)} pts)
      &nbsp;|&nbsp;<b>🎯 Tgt:</b> {cfg.get('target_type','—')} ({cfg.get('target_points',20)} pts)<br>
      <b>🔄 Cooldown:</b> {'✅ '+str(cfg.get('cooldown_seconds',5))+'s' if cfg.get('cooldown_enabled') else '❌ Off'}
      &nbsp;|&nbsp;<b>🚫 No Overlap:</b> {'✅' if cfg.get('no_overlap') else '❌'}
      &nbsp;|&nbsp;<b>🏦 Dhan:</b> {'✅ '+('Options' if cfg.get('options_trading') else 'Equity') if cfg.get('enable_dhan') else '❌ Off'}
    </div>""",unsafe_allow_html=True)

def style_df(df):
    if df.empty: return df.style
    def row_color(row):
        pnl=row.get("PnL",0)
        c="rgba(72,187,120,0.10)" if pnl>0 else ("rgba(252,129,129,0.10)" if pnl<0 else "")
        return [f"background-color:{c}"]*len(row)
    styled=df.style.apply(row_color,axis=1)
    if "PnL" in df.columns:
        styled=styled.applymap(lambda v:f"color:{'#48bb78' if v>0 else '#fc8181'};font-weight:700"
                               if isinstance(v,(int,float)) else "",subset=["PnL"])
    if "SL/Tgt Violated" in df.columns:
        styled=styled.applymap(lambda v:"background-color:#2d1515;color:#fc8181;font-weight:700"
                               if v is True else "",subset=["SL/Tgt Violated"])
    return styled

def trade_stats(trades):
    if not trades: return
    df=pd.DataFrame(trades)
    wins=df[df["PnL"]>0]; loss=df[df["PnL"]<=0]
    total_pnl=df["PnL"].sum()
    acc=len(wins)/len(df)*100 if len(df) else 0
    avg_win=wins["PnL"].mean() if len(wins) else 0
    avg_los=loss["PnL"].mean() if len(loss) else 0
    cols=st.columns(6)
    for col,lbl,val,color in [
        (cols[0],"Total Trades",len(df),"#e2e8f0"),
        (cols[1],"Win Trades",len(wins),"#48bb78"),
        (cols[2],"Loss Trades",len(loss),"#fc8181"),
        (cols[3],"Accuracy",f"{acc:.1f}%","#f6ad55"),
        (cols[4],"Total P&L",f"₹{total_pnl:+.2f}","#48bb78" if total_pnl>=0 else "#fc8181"),
        (cols[5],"Avg Win / Loss",f"₹{avg_win:.2f} / ₹{avg_los:.2f}","#76e4f7"),
    ]:
        col.markdown(f"""<div class="metric-card">
          <div style="color:#a0aec0;font-size:11px">{lbl}</div>
          <div style="color:{color};font-size:16px;font-weight:700;margin-top:3px">{val}</div>
        </div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 1 – BACKTESTING
# ═══════════════════════════════════════════════════════════════════

def tab_backtest(cfg):
    ltp_banner(cfg)
    st.markdown("## 🔬 Backtesting Engine")
    st.info("📌 EMA signal on candle N → Entry at candle N+1 open | SL checked vs Low(buy)/High(sell) FIRST (conservative)")

    col1,col2=st.columns([1,4])
    run_btn=col1.button("▶ Run Backtest",use_container_width=True,type="primary",key="btn_bt")
    col2.markdown("<br>",unsafe_allow_html=True)

    if run_btn:
        with st.spinner(f"Fetching {cfg.get('symbol')} data…"):
            df=fetch_data(cfg["symbol"],cfg["timeframe"],cfg["period"])
        if df is None or df.empty:
            st.error("❌ No data returned. Check ticker/interval/period."); return
        df=add_indicators(df,int(cfg.get("fast_ema",9)),int(cfg.get("slow_ema",15)))
        with st.spinner("Running backtest…"):
            trades,violations=run_backtest(df,cfg)
        st.session_state.backtest_trades=trades
        st.session_state.backtest_violations=violations
        st.session_state.backtest_df=df
        st.session_state.backtest_ran=True

    if not st.session_state.backtest_ran:
        st.markdown("<div style='color:#718096;text-align:center;padding:60px'>Click ▶ Run Backtest to start</div>",
                    unsafe_allow_html=True); return

    trades=st.session_state.backtest_trades
    violations=st.session_state.backtest_violations
    df=st.session_state.backtest_df
    fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))

    if not trades:
        st.warning("No trades generated. Try a different timeframe or strategy."); return

    # Stats
    trade_stats(trades)
    st.markdown("---")

    # Violations
    if violations:
        st.error(f"⚠️ **{len(violations)} SL/Target Violations** — Both SL and Target were hit in the same candle (order of execution uncertain in live trading)")
        with st.expander("View Violations"):
            vdf=pd.DataFrame(violations)
            st.dataframe(style_df(vdf),use_container_width=True)
    else:
        st.success("✅ No SL/Target violations found!")

    # Chart
    st.markdown("### 📊 Backtest Chart")
    chart_trades=trades[:50]  # limit markers
    fig=make_chart(df.tail(500),trades=chart_trades,title=f"Backtest: {cfg.get('ticker_name')}",
                   fast=fast,slow=slow,h=560)
    st.plotly_chart(fig,use_container_width=True,key="bt_chart")

    # P&L curve
    st.markdown("### 📈 Cumulative P&L")
    st.plotly_chart(pnl_chart(trades),use_container_width=True,key="bt_pnl")

    # Trades table
    st.markdown(f"### 📋 Trade Log ({len(trades)} trades)")
    tdf=pd.DataFrame(trades)
    col_order=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price",
               "SL","Target","Candle High","Candle Low","Entry Reason","Exit Reason","PnL","Qty","SL/Tgt Violated"]
    tdf=tdf[[c for c in col_order if c in tdf.columns]]
    st.dataframe(style_df(tdf),use_container_width=True,height=400)

    # Download
    csv=tdf.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV",csv,"backtest_trades.csv","text/csv",key="bt_dl")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 – LIVE TRADING
# ═══════════════════════════════════════════════════════════════════

def tab_live(cfg):
    ltp_banner(cfg)
    st.markdown("## 🚀 Live Trading")

    # Control buttons
    c1,c2,c3,_=st.columns([1,1,1,3])
    start_btn  = c1.button("▶ Start",  use_container_width=True, type="primary", key="btn_start")
    stop_btn   = c2.button("⏹ Stop",   use_container_width=True, key="btn_stop")
    sq_btn     = c3.button("✖ Squareoff", use_container_width=True, key="btn_sq")

    if start_btn and not st.session_state.live_running:
        st.session_state.live_running  = True
        st.session_state.live_status   = "RUNNING"
        st.session_state.live_log      = []
        st.session_state.live_position = None
        t=threading.Thread(target=live_thread,args=(cfg,cfg["symbol"]),daemon=True)
        st.session_state.live_thread   = t
        t.start()
        st.success("✅ Live trading started!")

    if stop_btn and st.session_state.live_running:
        st.session_state.live_running = False
        st.session_state.live_status  = "STOPPED"
        st.info("⏹ Stopping…")

    if sq_btn and st.session_state.live_position:
        pos=st.session_state.live_position
        ltp=st.session_state.live_ltp or pos["entry"]
        pnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        t={"Trade #":len(st.session_state.live_trades)+1,"Type":pos["type"].upper(),
           "Entry Time":_fdt(pos["entry_time"]),"Exit Time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
           "Entry Price":round(pos["entry"],4),"Exit Price":round(ltp,4),
           "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
           "Entry Reason":pos["reason"],"Exit Reason":"Manual Squareoff",
           "PnL":pnl,"Qty":cfg.get("quantity",1),"Source":"🚀 Live"}
        st.session_state.live_trades.append(t)
        st.session_state.live_position=None
        if cfg.get("enable_dhan") and st.session_state.dhan_client:
            place_order(st.session_state.dhan_client,cfg,pos["type"],ltp,is_exit=True)
        st.success(f"✅ Position squared off! PnL: {pnl:+.2f}")

    # Status badge
    status=st.session_state.live_status
    badge_color="#48bb78" if status=="RUNNING" else "#fc8181"
    st.markdown(f"""<div style="display:inline-block;background:{badge_color};
        color:white;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;margin-bottom:8px">
        {'🟢' if status=='RUNNING' else '🔴'} {status}</div>""",unsafe_allow_html=True)

    # Config summary
    st.markdown("#### 🔧 Active Configuration")
    cfg_box(cfg)

    # Last fetched candle info
    df=st.session_state.live_data
    if df is not None and not df.empty:
        lr=df.iloc[-1]
        ts=st.session_state.live_last_ts
        st.markdown(f"""<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;
            padding:8px 16px;font-size:12px;color:#a0aec0;margin:6px 0">
            📡 <b style="color:#76e4f7">Last Candle</b>:
            O:{lr['open']:.2f} &nbsp; H:{lr['high']:.2f} &nbsp; L:{lr['low']:.2f} &nbsp;
            C:{lr['close']:.2f} &nbsp;|&nbsp;
            EMA({cfg.get('fast_ema',9)}): <b style="color:#f6ad55">{lr.get('ema_fast',float('nan')):.2f}</b> &nbsp;
            EMA({cfg.get('slow_ema',15)}): <b style="color:#76e4f7">{lr.get('ema_slow',float('nan')):.2f}</b>
            &nbsp;|&nbsp; ATR: {lr.get('atr',0):.2f}
            &nbsp;|&nbsp; Fetched: <b style="color:#48bb78">{ts.strftime('%H:%M:%S') if ts else '—'} IST</b>
        </div>""",unsafe_allow_html=True)

    # Current position
    pos=st.session_state.live_position
    st.markdown("#### 📌 Current Position")
    if pos:
        ltp=st.session_state.live_ltp or pos["entry"]
        upnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        bc="rgba(72,187,120,0.15)" if upnl>=0 else "rgba(252,129,129,0.15)"
        bc2="#48bb78" if upnl>=0 else "#fc8181"
        pt_color="#48bb78" if pos["type"]=="buy" else "#fc8181"
        st.markdown(f"""<div style="background:{bc};border:1px solid {bc2};border-radius:10px;padding:14px">
          <b style="color:{pt_color};font-size:16px">{'🔼 BUY' if pos['type']=='buy' else '🔽 SELL'}</b>
          &nbsp;&nbsp; Entry: <b>{pos['entry']:.2f}</b>
          &nbsp;|&nbsp; LTP: <b style="color:{bc2}">{ltp:.2f}</b>
          &nbsp;|&nbsp; SL: <b style="color:#fc8181">{pos['sl']:.2f}</b>
          &nbsp;|&nbsp; Target: <b style="color:#48bb78">{pos['tgt']:.2f}</b>
          &nbsp;|&nbsp; Unrealised P&L: <b style="color:{bc2}">₹{upnl:+.2f}</b>
          &nbsp;|&nbsp; Reason: {pos['reason']}
          &nbsp;|&nbsp; Entry Time: {_fdt(pos['entry_time'])}
        </div>""",unsafe_allow_html=True)
    else:
        st.info("📭 No open position")

    # Live chart
    if df is not None and not df.empty:
        fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
        fig=make_chart(df.tail(300),title=f"Live: {cfg.get('ticker_name')}",
                       fast=fast,slow=slow,live_pos=pos,h=500)
        st.plotly_chart(fig,use_container_width=True,key="live_chart")

    # Elliott Wave panel
    st.markdown("---")
    st.markdown("#### 🌊 Elliott Wave Analysis")
    ew=st.session_state.live_ew
    if ew:
        s=ew.get("status","—"); d=ew.get("direction","UNCLEAR"); cw=ew.get("current_wave","—")
        cp2=ew.get("current_price",0)
        dc={"BULLISH":"#48bb78","BEARISH":"#fc8181","CORRECTIVE_DOWN":"#f6ad55",
            "CORRECTIVE_UP":"#76e4f7","UNCLEAR":"#a0aec0"}.get(d,"#a0aec0")
        cols=st.columns(4)
        for c,l,v,col_c in [(cols[0],"Structure",s,dc),(cols[1],"Direction",d,dc),
                             (cols[2],"Current Wave",cw,"#e2e8f0"),(cols[3],"Price",f"{cp2:,.2f}","#e2e8f0")]:
            c.markdown(f"""<div class="metric-card">
              <div style="color:#a0aec0;font-size:11px">{l}</div>
              <div style="color:{col_c};font-size:14px;font-weight:700;margin-top:3px">{v}</div>
            </div>""",unsafe_allow_html=True)

        det=ew.get("wave_detail",{}); tgts=ew.get("next_targets",{})
        if det:
            st.markdown("**Completed Waves:**")
            html='<div style="display:flex;flex-wrap:wrap;gap:6px">'
            for k,v in det.items():
                html+=f'<div class="wave-lbl"><b style="color:#f6ad55">{k}</b>: {v}</div>'
            st.markdown(html+"</div>",unsafe_allow_html=True)
        if tgts:
            st.markdown("**Next Targets:**")
            tcols=st.columns(len(tgts))
            for i,(n,v) in enumerate(tgts.items()):
                clr="#48bb78" if v>cp2 else "#fc8181"
                tcols[i].markdown(f"""<div class="metric-card" style="text-align:center">
                  <div style="color:#a0aec0;font-size:11px">{n}</div>
                  <div style="color:{clr};font-size:14px;font-weight:700">{v:,.2f}</div>
                </div>""",unsafe_allow_html=True)
        after=ew.get("after_msg","")
        bias=ew.get("trade_bias","")
        if after: st.info(f"💡 {after}")
        if bias: st.success(f"📊 Trade Bias: **{bias}**")

        if df is not None and not df.empty and len(ew.get("waves",[]))>0:
            fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
            st.plotly_chart(make_ew_chart(df,ew,fast,slow),use_container_width=True,key="ew_live")
    else:
        st.info("🌊 Elliott Wave analysis will appear once live trading starts and data is fetched.")

    # Log
    st.markdown("#### 📟 Activity Log")
    logs=st.session_state.live_log
    log_html='<div class="log-box">'+"".join(
        f'<div style="color:{"#fc8181" if "ERR" in l or "WARN" in l else "#48bb78" if "ENTRY" in l or "START" in l else "#f6ad55" if "EXIT" in l else "#a0aec0"}">{l}</div>'
        for l in reversed(logs[-80:]))+'</div>'
    st.markdown(log_html,unsafe_allow_html=True)

    # Live completed trades (real-time, doesn't require stopping)
    lt=st.session_state.live_trades
    if lt:
        st.markdown(f"#### ✅ Completed Trades This Session ({len(lt)})")
        trade_stats(lt)
        ltd=pd.DataFrame(lt)
        cols_show=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price",
                   "SL","Target","Entry Reason","Exit Reason","PnL","Qty"]
        ltd=ltd[[c for c in cols_show if c in ltd.columns]]
        st.dataframe(style_df(ltd),use_container_width=True,height=280)
        st.plotly_chart(pnl_chart(lt),use_container_width=True,key="live_pnl")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 – TRADE HISTORY
# ═══════════════════════════════════════════════════════════════════

def tab_history(cfg):
    ltp_banner(cfg)
    st.markdown("## 📚 Trade History")

    all_trades=[]
    bt=st.session_state.backtest_trades
    lt=st.session_state.live_trades
    for t in bt:
        tc=t.copy(); tc["Source"]="🔬 Backtest"; all_trades.append(tc)
    for t in lt:
        tc=t.copy(); tc.setdefault("Source","🚀 Live"); all_trades.append(tc)

    if not all_trades:
        st.markdown("<div style='color:#718096;text-align:center;padding:60px'>No trades yet. Run backtest or start live trading.</div>",
                    unsafe_allow_html=True); return

    df=pd.DataFrame(all_trades)

    # Filters
    f1,f2,f3=st.columns(3)
    sources=["All"]+sorted(df["Source"].unique().tolist())
    types  =["All","BUY","SELL"]
    src=f1.selectbox("Source",sources,key="h_src")
    typ=f2.selectbox("Type",types,key="h_typ")
    show_viol=f3.checkbox("Show Only Violations",False,key="h_viol")

    if src!="All": df=df[df["Source"]==src]
    if typ!="All": df=df[df["Type"]==typ]
    if show_viol and "SL/Tgt Violated" in df.columns:
        df=df[df["SL/Tgt Violated"]==True]

    st.markdown(f"#### 📊 Summary — {len(df)} trades")
    trade_stats(all_trades if src=="All" else df.to_dict("records"))

    if not df.empty:
        st.plotly_chart(pnl_chart(df.to_dict("records")),use_container_width=True,key="hist_pnl")
        col_order=["Trade #","Source","Type","Entry Time","Exit Time","Entry Price","Exit Price",
                   "SL","Target","Entry Reason","Exit Reason","PnL","Qty","SL/Tgt Violated"]
        show_cols=[c for c in col_order if c in df.columns]
        st.dataframe(style_df(df[show_cols]),use_container_width=True,height=460)
        csv=df[show_cols].to_csv(index=False).encode()
        st.download_button("⬇ Download All Trades CSV",csv,"trade_history.csv","text/csv",key="h_dl")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div class="app-hdr">
      <div><h1>📈 Smart Investing</h1>
      <p>Professional Algorithmic Trading Platform · NSE · BSE · Crypto · Commodities</p></div>
      <div style="color:#90cdf4;font-size:12px;text-align:right">
        EMA · Elliott Wave · Dhan Broker · Fully Automated
      </div>
    </div>""",unsafe_allow_html=True)

    cfg=sidebar()
    tab1,tab2,tab3=st.tabs(["🔬 Backtesting","🚀 Live Trading","📚 Trade History"])
    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history(cfg)

if __name__=="__main__":
    main()
