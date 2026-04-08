# ╔══════════════════════════════════════════════════════════════╗
# ║      SMART INVESTING  –  Algo Trading Platform  v3          ║
# ║  NSE · BSE · Crypto · Gold · Silver  |  Dhan Broker Ready  ║
# ╚══════════════════════════════════════════════════════════════╝
# v3 fixes:
#  1. Simple Buy/Sell enters IMMEDIATELY on first tick — UI
#     auto-refreshes every 2s via st.rerun() so position shows
#  2. EMA values (numeric) shown in Last Candle bar
#  3. Elliott Wave shows Entry / SL / Target recommendations
#  4. New Optimization tab — grid search, click-to-apply results
#  5. applymap → map  (pandas ≥2.1)
#  6. Background thread writes to _TS (thread-safe); _sync() on rerun

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

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""<style>
.main{background:#0e1117}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif}
.app-hdr{background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);padding:14px 24px;
  border-radius:14px;margin-bottom:14px;border:1px solid #2d4a6e;
  display:flex;align-items:center;justify-content:space-between}
.app-hdr h1{margin:0;color:#e2e8f0;font-size:22px}
.app-hdr p{margin:0;color:#90cdf4;font-size:12px}
.ltp-bar{background:linear-gradient(135deg,#1a1f2e,#252b3d);border:1px solid #2d4a6e;
  border-radius:10px;padding:10px 20px;display:flex;align-items:center;gap:20px;margin-bottom:12px}
.ltp-ticker{color:#90cdf4;font-size:12px}.ltp-price{color:#e2e8f0;font-size:26px;font-weight:700}
.ltp-up{color:#48bb78;font-size:14px;font-weight:600}.ltp-down{color:#fc8181;font-size:14px;font-weight:600}
.ltp-meta{color:#718096;font-size:11px;margin-left:auto}
.metric-card{background:linear-gradient(135deg,#1a1f2e,#252b3d);border:1px solid #2d3748;
  border-radius:10px;padding:14px;margin:5px 0}
.cfg-box{background:#1a1f2e;border-radius:8px;padding:12px;border-left:3px solid #4299e1;
  margin:8px 0;font-size:13px;line-height:1.8;color:#cbd5e0}
.ew-rec{background:#1a2035;border:1px solid #2d4a6e;border-radius:10px;padding:14px;margin:8px 0}
.wave-lbl{background:#2d3748;border:1px solid #4a5568;border-radius:6px;
  padding:5px 10px;display:inline-block;margin:3px;font-size:12px}
.log-box{background:#0d1117;border-radius:8px;padding:10px;height:200px;overflow-y:auto;
  font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;color:#a0aec0}
.opt-card{background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:14px;margin:6px 0}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent}
.stTabs [data-baseweb="tab"]{background:#1a1f2e;border-radius:8px;padding:8px 20px;
  border:1px solid #2d3748;color:#a0aec0}
.stTabs [aria-selected="true"]{background:#2b6cb0!important;color:white!important}
section[data-testid="stSidebar"]{background:#0d1117}
section[data-testid="stSidebar"] label{color:#a0aec0!important;font-size:12px}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
TICKERS = {"Nifty 50":"^NSEI","BankNifty":"^NSEBANK","Sensex":"^BSESN",
           "BTC":"BTC-USD","ETH":"ETH-USD","Gold":"GC=F","Silver":"SI=F","Custom":None}
TF_PERIODS = {
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo"],"15m":["1d","5d","7d","1mo"],
    "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
MAX_PERIOD={"1m":"7d","5m":"60d","15m":"60d","1h":"730d","1d":"max","1wk":"max"}
TF_MINUTES={"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080}

# ══════════════════════════════════════════════════════════════════
# THREAD-SAFE STORE  (_TS)
# Background threads cannot safely access st.session_state.
# All live state lives in _TS. _sync() copies to session_state on
# every Streamlit rerun so the UI always sees fresh values.
# ══════════════════════════════════════════════════════════════════
_TS_LOCK = threading.Lock()
_TS: dict = {
    "live_running":False,"live_status":"STOPPED","live_position":None,
    "live_trades":[],"live_data":None,"live_ltp":None,"live_prev_close":None,
    "live_log":[],"live_last_ts":None,"live_ew":{},"_rl_ts":0.0,
}

def _ts_get(k,d=None):
    with _TS_LOCK: return _TS.get(k,d)
def _ts_set(k,v):
    with _TS_LOCK: _TS[k]=v
def _ts_append(k,v,mx=300):
    with _TS_LOCK:
        _TS[k].append(v)
        if len(_TS[k])>mx: _TS[k]=_TS[k][-mx:]
def _sync():
    for k in ["live_running","live_status","live_position","live_trades",
              "live_data","live_ltp","live_prev_close","live_log","live_last_ts","live_ew"]:
        st.session_state[k]=_ts_get(k)

for _k,_v in {"live_thread":None,"backtest_trades":[],"backtest_violations":[],
              "backtest_df":None,"backtest_ran":False,"dhan_client":None,
              "opt_results":[],"opt_ran":False}.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ══════════════════════════════════════════════════════════════════
# RATE-LIMITED DATA FETCH
# ══════════════════════════════════════════════════════════════════
def _rate_wait():
    with _TS_LOCK:
        gap=time.time()-_TS["_rl_ts"]
        if gap<1.5: time.sleep(1.5-gap)
        _TS["_rl_ts"]=time.time()

def _clean(df):
    if df is None or df.empty: return None
    df=df.copy()
    df.index=pd.to_datetime(df.index)
    df.index=df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize("UTC").tz_convert(IST)
    df.columns=[c.lower() for c in df.columns]
    keep=[c for c in ["open","high","low","close","volume"] if c in df.columns]
    df=df[keep].dropna(subset=["close"])
    return df[~df.index.duplicated(keep="last")].sort_index()

def fetch_data(symbol,interval,period):
    _rate_wait()
    try:
        raw=yf.Ticker(symbol).history(period=MAX_PERIOD.get(interval,period),interval=interval,auto_adjust=True,prepost=False)
        df=_clean(raw)
        if df is None or df.empty:
            raw=yf.Ticker(symbol).history(period=period,interval=interval,auto_adjust=True,prepost=False)
            df=_clean(raw)
        return df
    except Exception as e: _log(f"[ERROR] fetch: {e}"); return None

def fetch_ltp(symbol):
    try:
        _rate_wait()
        raw=yf.Ticker(symbol).history(period="5d",interval="1d",auto_adjust=True,prepost=False)
        df=_clean(raw)
        if df is not None and len(df)>=2: return float(df["close"].iloc[-1]),float(df["close"].iloc[-2])
        if df is not None and len(df)==1: v=float(df["close"].iloc[-1]); return v,v
    except Exception: pass
    return None,None

# ══════════════════════════════════════════════════════════════════
# INDICATORS  (TradingView-accurate)
# ══════════════════════════════════════════════════════════════════
def ema_tv(series,n):
    if len(series)<n: return pd.Series(np.nan,index=series.index)
    alpha=2.0/(n+1); vals=series.ffill().values.astype(float)
    out=np.full(len(vals),np.nan); out[n-1]=np.nanmean(vals[:n])
    for i in range(n,len(vals)):
        v=vals[i] if not np.isnan(vals[i]) else out[i-1]
        out[i]=alpha*v+(1-alpha)*out[i-1]
    return pd.Series(out,index=series.index)

def atr_s(df,n=14):
    h,l,c=df["high"],df["low"],df["close"]
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return ema_tv(tr,n)

def add_indicators(df,fast,slow):
    if df is None or df.empty: return df
    df=df.copy()
    df["ema_fast"]=ema_tv(df["close"],fast); df["ema_slow"]=ema_tv(df["close"],slow)
    df["atr"]=atr_s(df)
    df["cross_up"]=(df["ema_fast"]>df["ema_slow"])&(df["ema_fast"].shift(1)<=df["ema_slow"].shift(1))
    df["cross_down"]=(df["ema_fast"]<df["ema_slow"])&(df["ema_fast"].shift(1)>=df["ema_slow"].shift(1))
    return df

def ema_angle(ema,lb=3):
    v=ema.dropna()
    if len(v)<lb+1: return 0.0
    base=v.iloc[-lb]
    return 0.0 if base==0 else abs(math.degrees(math.atan((v.iloc[-1]-base)/base*100/lb)))

# ══════════════════════════════════════════════════════════════════
# ELLIOTT WAVE
# ══════════════════════════════════════════════════════════════════
def find_swings(df,win=5):
    h,l=df["high"].values,df["low"].values; pts=[]
    for i in range(win,len(df)-win):
        lo,hi=max(0,i-win),min(len(df),i+win+1)
        if h[i]==h[lo:hi].max(): pts.append({"dt":df.index[i],"price":h[i],"idx":i,"type":"H"})
        if l[i]==l[lo:hi].min(): pts.append({"dt":df.index[i],"price":l[i],"idx":i,"type":"L"})
    pts.sort(key=lambda x:x["idx"]); clean=[]
    for p in pts:
        if not clean: clean.append(p); continue
        if clean[-1]["type"]==p["type"]:
            if (p["type"]=="H" and p["price"]>clean[-1]["price"]) or \
               (p["type"]=="L" and p["price"]<clean[-1]["price"]): clean[-1]=p
        else: clean.append(p)
    return clean

def detect_ew(df):
    cp=float(df["close"].iloc[-1]) if len(df) else 0
    base={"status":"Forming","direction":"UNCLEAR","waves":[],"current_wave":"Accumulating",
          "wave_detail":{},"next_targets":{},"after_msg":"Accumulating data",
          "swing_points":[],"current_price":cp,"entry_rec":None,"sl_rec":None,"tgt_rec":None}
    if len(df)<25: return {**base,"status":"Insufficient data","after_msg":"Need more candles"}
    swings=find_swings(df,win=max(3,len(df)//40))
    if len(swings)<4: return {**base,"swing_points":swings}

    # 5-wave impulse
    if len(swings)>=6:
        for start,label,sign in [("L","BULLISH",1),("H","BEARISH",-1)]:
            s=swings[-6:]
            if s[0]["type"]!=start: continue
            p=[pt["price"] for pt in s]
            w1=sign*(p[1]-p[0]); w3=sign*(p[3]-p[2])
            w2r=(p[2]-p[1])/(p[1]-p[0]) if p[1]-p[0] else 0
            if w1>0 and w3>0 and abs(w2r)<1.0 and w3>=w1*0.618:
                waves=[{"label":str(n+1),"start":s[n],"end":s[n+1]} for n in range(5)]
                projs={f"W5 @ {int(k*100)}% W1":round(p[4]+sign*w1*k,2) for k in [0.618,1.0,1.618]}
                # Trade recommendation for Wave 5
                entry_rec=round(p[4],2)                      # enter at current W4 end
                sl_rec=round(p[3],2)                         # SL below W4 low (bull) or above W4 high (bear)
                tgt_rec=round(p[4]+sign*w1,2)                # Target = W5 100% projection
                return {**base,
                    "status":f"5-Wave Impulse ({'Bullish' if sign==1 else 'Bearish'})",
                    "direction":label,"waves":waves,"current_wave":"Wave 5 (forming)",
                    "wave_detail":{"W1":f"{p[0]:.2f}→{p[1]:.2f}",
                        "W2":f"{p[1]:.2f}→{p[2]:.2f} ({abs(w2r)*100:.1f}% retracement)",
                        "W3":f"{p[2]:.2f}→{p[3]:.2f} ({w3/w1*100:.0f}% of W1)",
                        "W4":f"{p[3]:.2f}→{p[4]:.2f}","W5":f"{p[4]:.2f}→{p[5]:.2f} (live)"},
                    "next_targets":projs,"after_msg":"Expect ABC correction after W5 completes",
                    "swing_points":swings,"current_price":cp,
                    "trade_bias":"BUY" if sign==1 else "SELL",
                    "entry_rec":entry_rec,"sl_rec":sl_rec,"tgt_rec":tgt_rec}

    # ABC correction
    if len(swings)>=4:
        for start,label,sign in [("H","CORRECTIVE_DOWN",-1),("L","CORRECTIVE_UP",1)]:
            s=swings[-4:]
            if s[0]["type"]!=start: continue
            p=[pt["price"] for pt in s]
            amv=sign*(p[0]-p[1]); br=(p[2]-p[1])/(p[1]-p[0]) if p[1]-p[0] else 0
            c_618=round(p[2]-sign*amv*0.618,2); c_100=round(p[2]-sign*amv,2)
            entry_rec=round(p[2],2)          # enter at B-wave end
            sl_rec=round(p[1],2)             # SL beyond B
            tgt_rec=c_618                    # Target = C at 61.8% of A
            return {**base,
                "status":f"ABC Correction ({'Bearish' if sign==-1 else 'Bullish'})",
                "direction":label,
                "waves":[{"label":lb,"start":s[i],"end":s[i+1]} for i,lb in enumerate(["A","B","C"])],
                "current_wave":"Wave C (forming)","wave_detail":{
                    "A":f"{p[0]:.2f}→{p[1]:.2f}","B":f"{p[1]:.2f}→{p[2]:.2f} ({abs(br)*100:.1f}% ret)",
                    "C":f"{p[2]:.2f}→{p[3]:.2f} (live)"},
                "next_targets":{"C @ 61.8% A":c_618,"C = A":c_100},
                "after_msg":"Expect new impulse after ABC completes",
                "swing_points":swings,"current_price":cp,
                "trade_bias":"SELL" if sign==-1 else "BUY",
                "entry_rec":entry_rec,"sl_rec":sl_rec,"tgt_rec":tgt_rec}

    return {**base,"swing_points":swings}

# ══════════════════════════════════════════════════════════════════
# SL / TARGET
# ══════════════════════════════════════════════════════════════════
def calc_sl_tgt(entry,trade_type,row,cfg):
    atr=float(row.get("atr",entry*0.01) or entry*0.01)
    if np.isnan(atr) or atr==0: atr=entry*0.01
    sl_t=cfg.get("sl_type","Custom Points"); tg_t=cfg.get("target_type","Custom Points")
    slp=float(cfg.get("sl_points",10)); tgp=float(cfg.get("target_points",20))
    sign=1 if trade_type=="buy" else -1
    sl=entry-sign*atr*float(cfg.get("atr_sl_mult",1.5)) if sl_t=="ATR Based" else entry-sign*slp
    sl_d=abs(entry-sl)
    if tg_t=="ATR Based Target": tgt=entry+sign*atr*float(cfg.get("atr_tgt_mult",3.0))
    elif tg_t=="Risk-Reward Based": tgt=entry+sign*sl_d*float(cfg.get("rr_ratio",2.0))
    else: tgt=entry+sign*tgp
    return round(sl,4),round(tgt,4)

# ══════════════════════════════════════════════════════════════════
# SIGNALS
# ══════════════════════════════════════════════════════════════════
def ema_signal(df,idx,cfg):
    if idx<1 or idx>=len(df): return None,None
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es=row.get("ema_fast",np.nan),row.get("ema_slow",np.nan)
    pf,ps=prev.get("ema_fast",np.nan),prev.get("ema_slow",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return None,None
    if cfg.get("use_angle_filter",False):
        ang=ema_angle(df["ema_fast"].iloc[:idx+1])
        if ang<float(cfg.get("min_ema_angle",0)): return None,None
    ct=cfg.get("crossover_type","Simple Crossover"); body=abs(row["close"]-row["open"])
    def ok():
        if ct=="Simple Crossover": return True
        if ct=="Custom Candle Size": return body>=float(cfg.get("custom_candle_size",10))
        if ct=="ATR Based Candle Size":
            a=float(row.get("atr",0) or 0); return a==0 or body>=a
        return True
    f,s=int(cfg.get("fast_ema",9)),int(cfg.get("slow_ema",15))
    if ef>es and pf<=ps and ok(): return "buy",f"EMA({f}) crossed ABOVE EMA({s})"
    if ef<es and pf>=ps and ok(): return "sell",f"EMA({f}) crossed BELOW EMA({s})"
    return None,None

def ema_rev_cross(df,idx,ptype):
    if idx<1: return False
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es=row.get("ema_fast",np.nan),row.get("ema_slow",np.nan)
    pf,ps=prev.get("ema_fast",np.nan),prev.get("ema_slow",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return False
    if ptype=="buy" and ef<es and pf>=ps: return True
    if ptype=="sell" and ef>es and pf<=ps: return True
    return False

def _fdt(dt):
    if hasattr(dt,"strftime"): return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)

# ══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════
def run_backtest(df,cfg):
    if df is None or df.empty: return [],[]
    strat=cfg.get("strategy","EMA Crossover")
    fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
    qty=int(cfg.get("quantity",1))
    sl_t=cfg.get("sl_type","Custom Points"); tg_t=cfg.get("target_type","Custom Points")
    df=add_indicators(df.copy(),fast,slow)
    trades=[]; violations=[]; pos=None; pending=None; tnum=0

    for i in range(1,len(df)):
        row=df.iloc[i]
        if pos:
            sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
            if sl_t=="Trailing SL":
                trail=float(cfg.get("sl_points",10))
                if pt=="buy":
                    ns=row["high"]-trail
                    if ns>sl: sl=pos["sl"]=ns
                else:
                    ns=row["low"]+trail
                    if ns<sl: sl=pos["sl"]=ns
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
                elif ema_ex: ep=row["open"]; er="EMA Reverse Crossover exit"
            else:
                if row["high"]>=sl:
                    ep=sl; er=f"SL hit (High {row['high']:.2f}≥SL {sl:.2f})"
                    if tg_t!="Trailing Target" and row["low"]<=tgt: viol=True
                elif tg_t!="Trailing Target" and row["low"]<=tgt:
                    ep=tgt; er=f"Target hit (Low {row['low']:.2f}≤Tgt {tgt:.2f})"
                elif ema_ex: ep=row["open"]; er="EMA Reverse Crossover exit"
            if ep is None and i==len(df)-1: ep=row["close"]; er="End of data"
            if ep is not None:
                pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                t={"Trade #":tnum,"Type":pt.upper(),
                   "Entry Time":_fdt(pos["entry_time"]),"Exit Time":_fdt(row.name),
                   "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                   "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
                   "Candle High":round(row["high"],4),"Candle Low":round(row["low"],4),
                   "Entry Reason":pos["reason"],"Exit Reason":er,
                   "PnL":pnl,"Qty":qty,"SL/Tgt Violated":viol}
                trades.append(t); violations.append(t) if viol else None; pos=None
        if pos is None and pending:
            sig,rsn=pending; pending=None
            ep=float(row["open"]); sl,tgt=calc_sl_tgt(ep,sig,row,cfg); tnum+=1
            pos={"type":sig,"entry":ep,"entry_time":row.name,
                 "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":rsn}
        if pos is None:
            if strat=="EMA Crossover":
                sig,rsn=ema_signal(df,i,cfg)
                if sig: pending=(sig,rsn)
            elif strat=="Simple Buy":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"buy",row,cfg); tnum+=1
                pos={"type":"buy","entry":ep,"entry_time":row.name,
                     "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Buy"}
            elif strat=="Simple Sell":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"sell",row,cfg); tnum+=1
                pos={"type":"sell","entry":ep,"entry_time":row.name,
                     "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Sell"}
            elif strat=="Elliott Wave" and i>=20:
                ew=detect_ew(df.iloc[:i+1]); bias=ew.get("trade_bias","")
                if bias:
                    sig="buy" if bias=="BUY" else "sell"
                    pending=(sig,f"Elliott Wave: {ew.get('current_wave','')}")
    return trades,violations

# ══════════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE
# ══════════════════════════════════════════════════════════════════
def run_optimization(df_raw, cfg_base, param_grid, desired_acc, max_combos=200):
    """Grid search over EMA / SL / Target parameters. Returns top results."""
    results=[]
    combos=list(itertools.product(*param_grid.values()))
    keys=list(param_grid.keys())
    # Limit combos
    if len(combos)>max_combos: combos=combos[:max_combos]

    for combo in combos:
        trial=dict(cfg_base)
        for k,v in zip(keys,combo): trial[k]=v
        try:
            trades,_=run_backtest(df_raw,trial)
            if not trades: continue
            df_t=pd.DataFrame(trades)
            wins=len(df_t[df_t["PnL"]>0]); total=len(df_t)
            acc=wins/total*100 if total else 0
            pnl=df_t["PnL"].sum()
            if acc>=desired_acc:
                results.append({**{k:trial[k] for k in keys},
                    "Trades":total,"Wins":wins,"Accuracy":round(acc,1),
                    "Total PnL":round(pnl,2),"Avg PnL/Trade":round(pnl/total,2)})
        except Exception: continue

    results.sort(key=lambda x:(-x["Accuracy"],-x["Total PnL"]))
    return results[:20]

# ══════════════════════════════════════════════════════════════════
# DHAN BROKER
# ══════════════════════════════════════════════════════════════════
def init_dhan(cid,tok):
    try:
        from dhanhq import dhanhq; return dhanhq(cid,tok)
    except ImportError: st.warning("Install: pip install dhanhq"); return None
    except Exception as e: st.error(f"Dhan error: {e}"); return None

def get_my_ip():
    try: return requests.get("https://api.ipify.org?format=json",timeout=5).json().get("ip","?")
    except Exception:
        try: return requests.get("https://ifconfig.me/ip",timeout=5).text.strip()
        except Exception: return "Could not detect"

def register_ip_ui():
    ip=get_my_ip()
    st.info(f"📡 **Your public IP:** `{ip}`\n\n"
            "**SEBI requires IP whitelisting before placing orders.**\n\n"
            "1. Login → [Dhan Console](https://console.dhan.co)\n"
            "2. **Profile → API → IP Whitelist → Add IP above → Save**\n"
            "3. Dynamic IP? Re-register each session or use a VPN with fixed IP.")

def place_order(dhan,cfg,sig,ltp,is_exit=False):
    if dhan is None: return {"error":"Not connected"}
    try:
        if cfg.get("options_trading",False):
            fno=cfg.get("fno_exchange","NSE_FNO"); qty=int(cfg.get("options_qty",65))
            ot=cfg.get("options_exit_type" if is_exit else "options_entry_type","MARKET")
            px=round(ltp,2) if ot=="LIMIT" else 0
            sid=cfg.get("ce_security_id") if sig=="buy" else cfg.get("pe_security_id")
            return dhan.place_order(transactionType="SELL" if is_exit else "BUY",
                exchangeSegment=fno,productType="INTRADAY",orderType=ot,validity="DAY",
                securityId=str(sid),quantity=qty,price=px,triggerPrice=0)
        else:
            ot=cfg.get("exit_order_type" if is_exit else "entry_order_type","MARKET")
            px=round(ltp,2) if ot=="LIMIT" else 0
            txn=("SELL" if is_exit else "BUY") if sig=="buy" else ("BUY" if is_exit else "SELL")
            return dhan.place_order(security_id=str(cfg.get("security_id","1594")),
                exchange_segment=cfg.get("exchange","NSE"),transaction_type=txn,
                quantity=int(cfg.get("dhan_qty",1)),order_type=ot,
                product_type=cfg.get("product_type","INTRADAY"),price=px)
    except Exception as e: return {"error":str(e)}

# ══════════════════════════════════════════════════════════════════
# LIVE LOG
# ══════════════════════════════════════════════════════════════════
def _log(msg):
    _ts_append("live_log",f"[{datetime.now(IST).strftime('%H:%M:%S')}] {msg}",mx=300)

# ══════════════════════════════════════════════════════════════════
# LIVE TRADING THREAD
# ══════════════════════════════════════════════════════════════════
def live_thread(cfg,symbol):
    _log(f"▶ STARTED | {symbol} | {cfg['timeframe']} | {cfg['strategy']}")
    tf_min=TF_MINUTES.get(cfg["timeframe"],5)
    fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
    strat=cfg.get("strategy","EMA Crossover")
    sl_t=cfg.get("sl_type","Custom Points"); tg_t=cfg.get("target_type","Custom Points")
    qty=int(cfg.get("quantity",1))
    dhan=st.session_state.get("dhan_client"); en_dhan=cfg.get("enable_dhan",False)
    pending=None; last_sig_candle=None; last_bdy_min=-1
    # For Simple Buy/Sell: track if we already entered once this cycle
    # (re-enter only after exit + cooldown, not immediately on same tick after entry)
    entered_this_tick=False

    while _ts_get("live_running",False):
        try:
            entered_this_tick=False
            df=fetch_data(symbol,cfg["timeframe"],cfg["period"])
            now=datetime.now(IST)
            if df is None or df.empty: _log("[WARN] No data, retrying…"); time.sleep(1.5); continue
            df=add_indicators(df,fast,slow)
            ltp=float(df["close"].iloc[-1])
            _ts_set("live_data",df); _ts_set("live_ltp",ltp); _ts_set("live_last_ts",now)
            if len(df)>=20: _ts_set("live_ew",detect_ew(df))
            pos=_ts_get("live_position")

            # ── SL/Target tick check (vs LTP) ──────────────────────
            if pos:
                sl=pos["sl"]; tgt=pos["tgt"]; pt=pos["type"]
                if sl_t=="Trailing SL":
                    trail=float(cfg.get("sl_points",10))
                    if pt=="buy":
                        ns=ltp-trail
                        if ns>sl: pos=dict(pos,sl=ns); _ts_set("live_position",pos); sl=ns
                    else:
                        ns=ltp+trail
                        if ns<sl: pos=dict(pos,sl=ns); _ts_set("live_position",pos); sl=ns
                if tg_t=="Trailing Target":
                    tp=float(cfg.get("target_points",20))
                    if pt=="buy":
                        nt=ltp+tp
                        if nt>tgt: pos=dict(pos,tgt=nt); _ts_set("live_position",pos)
                    else:
                        nt=ltp-tp
                        if nt<tgt: pos=dict(pos,tgt=nt); _ts_set("live_position",pos)
                ep=None; er=None
                if pt=="buy":
                    if ltp<=sl:   ep=sl;  er=f"SL hit (LTP {ltp:.2f}≤SL {sl:.2f})"
                    elif tg_t!="Trailing Target" and ltp>=tgt: ep=tgt; er=f"Target hit (LTP {ltp:.2f}≥Tgt {tgt:.2f})"
                else:
                    if ltp>=sl:   ep=sl;  er=f"SL hit (LTP {ltp:.2f}≥SL {sl:.2f})"
                    elif tg_t!="Trailing Target" and ltp<=tgt: ep=tgt; er=f"Target hit (LTP {ltp:.2f}≤Tgt {tgt:.2f})"
                if ep is None and (sl_t=="Reverse EMA Crossover" or tg_t=="EMA Crossover"):
                    if ema_rev_cross(df,len(df)-1,pt): ep=ltp; er="EMA Reverse Crossover exit"
                if ep is not None:
                    pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                    t={"Trade #":len(_ts_get("live_trades",[]))+1,"Type":pt.upper(),
                       "Entry Time":_fdt(pos["entry_time"]),"Exit Time":now.strftime("%Y-%m-%d %H:%M:%S"),
                       "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                       "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
                       "Entry Reason":pos["reason"],"Exit Reason":er,"PnL":pnl,"Qty":qty,"Source":"🚀 Live"}
                    _ts_append("live_trades",t); _ts_set("live_position",None)
                    if en_dhan and dhan: _log(f"EXIT order → {place_order(dhan,cfg,pt,ltp,is_exit=True)}")
                    _log(f"✖ EXIT {pt.upper()} @{ep:.2f} | {er} | PnL:{pnl:+.2f}")

            # ── Entry logic ─────────────────────────────────────────
            if _ts_get("live_position") is None and not entered_this_tick:
                cd_ok=True
                if cfg.get("cooldown_enabled",True):
                    trades_so_far=_ts_get("live_trades",[])
                    if trades_so_far:
                        try:
                            le=datetime.strptime(trades_so_far[-1].get("Exit Time","2000-01-01 00:00:00"),"%Y-%m-%d %H:%M:%S")
                            le=IST.localize(le)
                            if (now-le).total_seconds()<int(cfg.get("cooldown_seconds",5)): cd_ok=False
                        except Exception: pass

                # ── Simple Buy/Sell: IMMEDIATE on every qualifying tick ──
                if strat=="Simple Buy" and cd_ok:
                    sl,tgt=calc_sl_tgt(ltp,"buy",df.iloc[-1],cfg)
                    _ts_set("live_position",{"type":"buy","entry":ltp,"entry_time":now,
                        "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Buy"})
                    entered_this_tick=True
                    if en_dhan and dhan: _log(f"ENTRY order → {place_order(dhan,cfg,'buy',ltp)}")
                    _log(f"▲ Simple BUY @{ltp:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")

                elif strat=="Simple Sell" and cd_ok:
                    sl,tgt=calc_sl_tgt(ltp,"sell",df.iloc[-1],cfg)
                    _ts_set("live_position",{"type":"sell","entry":ltp,"entry_time":now,
                        "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Sell"})
                    entered_this_tick=True
                    if en_dhan and dhan: _log(f"ENTRY order → {place_order(dhan,cfg,'sell',ltp)}")
                    _log(f"▼ Simple SELL @{ltp:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")

                # ── EMA/EW: pending → execute on next candle open ──────
                elif pending and cd_ok:
                    sig,rsn=pending; pending=None
                    ep=float(df["open"].iloc[-1]); sl,tgt=calc_sl_tgt(ep,sig,df.iloc[-1],cfg)
                    _ts_set("live_position",{"type":sig,"entry":ep,"entry_time":now,
                        "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":rsn})
                    entered_this_tick=True
                    if en_dhan and dhan: _log(f"ENTRY order → {place_order(dhan,cfg,sig,ep)}")
                    _log(f"▲ ENTRY {sig.upper()} @{ep:.2f} | SL:{sl:.2f} | Tgt:{tgt:.2f}")

                else:
                    tm=now.hour*60+now.minute; at_bdy=(tm%tf_min==0) and (tm!=last_bdy_min)
                    if at_bdy and cd_ok:
                        last_bdy_min=tm
                        if strat=="EMA Crossover":
                            sig,rsn=ema_signal(df,len(df)-1,cfg)
                            if sig and df.index[-1]!=last_sig_candle:
                                last_sig_candle=df.index[-1]; pending=(sig,rsn)
                                _log(f"◆ SIGNAL {sig.upper()} on {_fdt(df.index[-1])} → enter next candle")
                        elif strat=="Elliott Wave":
                            ew=_ts_get("live_ew",{}); bias=ew.get("trade_bias","")
                            if bias:
                                s2="buy" if bias=="BUY" else "sell"
                                pending=(s2,f"EW: {ew.get('current_wave','')} ({ew.get('direction','')})")
                                _log(f"◆ EW SIGNAL {s2.upper()} → enter next candle")
            time.sleep(1.5)
        except Exception as e: _log(f"[ERR] {e}"); time.sleep(1.5)
    _log("■ STOPPED.")

# ══════════════════════════════════════════════════════════════════
# CHARTING
# ══════════════════════════════════════════════════════════════════
def make_chart(df,trades=None,title="",fast=9,slow=15,live_pos=None,h=520):
    if df is None or df.empty: return go.Figure()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.04,row_heights=[0.78,0.22])
    fig.add_trace(go.Candlestick(x=df.index,open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="Price",increasing_line_color="#48bb78",decreasing_line_color="#fc8181",
        increasing_fillcolor="#48bb78",decreasing_fillcolor="#fc8181"),row=1,col=1)
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_fast"],name=f"EMA({fast})",
            line=dict(color="#f6ad55",width=1.5),opacity=0.9),row=1,col=1)
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_slow"],name=f"EMA({slow})",
            line=dict(color="#76e4f7",width=1.5),opacity=0.9),row=1,col=1)
    if "volume" in df.columns:
        vc=["#48bb78" if c>=o else "#fc8181" for c,o in zip(df["close"],df["open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Volume",marker_color=vc,opacity=0.5),row=2,col=1)
    if trades:
        for t in trades:
            try:
                et=pd.to_datetime(t["Entry Time"]); xt=pd.to_datetime(t["Exit Time"])
                c="#48bb78" if t["Type"]=="BUY" else "#fc8181"
                sym="triangle-up" if t["Type"]=="BUY" else "triangle-down"
                pc="#48bb78" if t.get("PnL",0)>=0 else "#fc8181"
                fig.add_trace(go.Scatter(x=[et],y=[t["Entry Price"]],mode="markers+text",
                    marker=dict(symbol=sym,size=11,color=c),text=[f"E:{t['Entry Price']:.0f}"],
                    textposition="top center",textfont=dict(size=8,color=c),showlegend=False),row=1,col=1)
                fig.add_trace(go.Scatter(x=[xt],y=[t["Exit Price"]],mode="markers+text",
                    marker=dict(symbol="x",size=9,color=pc),text=[f"X:{t['Exit Price']:.0f}"],
                    textposition="bottom center",textfont=dict(size=8,color=pc),showlegend=False),row=1,col=1)
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
                annotation_text=f"SL {live_pos['sl']:.2f}",annotation_font=dict(color="#fc8181"),row=1,col=1)
            fig.add_hline(y=live_pos["tgt"],line=dict(color="#48bb78",width=2,dash="dot"),
                annotation_text=f"Tgt {live_pos['tgt']:.2f}",annotation_font=dict(color="#48bb78"),row=1,col=1)
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

def make_ew_chart(df,ew,fast,slow):
    fig=make_chart(df.tail(150),title="Elliott Wave Structure",fast=fast,slow=slow,h=440)
    wc={"1":"#f6ad55","2":"#fc8181","3":"#48bb78","4":"#e9d8a6","5":"#76e4f7","A":"#fc8181","B":"#48bb78","C":"#fc8181"}
    for w in ew.get("waves",[]):
        s=w["start"]; e=w["end"]; c=wc.get(w["label"],"#a0aec0")
        fig.add_trace(go.Scatter(x=[s["dt"],e["dt"]],y=[s["price"],e["price"]],
            mode="lines+markers+text",line=dict(color=c,width=2.5),marker=dict(size=8,color=c),
            text=["",f"W{w['label']}"],textposition="top center",textfont=dict(color=c,size=12),
            name=f"W{w['label']}"),row=1,col=1)
    # Draw EW recommended entry/SL/target if present
    er=ew.get("entry_rec"); sr=ew.get("sl_rec"); tr=ew.get("tgt_rec")
    if er: fig.add_hline(y=er,line=dict(color="#f6ad55",width=1.5,dash="dash"),
        annotation_text=f"EW Entry {er:.2f}",annotation_font=dict(color="#f6ad55",size=10),row=1,col=1)
    if sr: fig.add_hline(y=sr,line=dict(color="#fc8181",width=1.5,dash="dash"),
        annotation_text=f"EW SL {sr:.2f}",annotation_font=dict(color="#fc8181",size=10),row=1,col=1)
    if tr: fig.add_hline(y=tr,line=dict(color="#48bb78",width=1.5,dash="dash"),
        annotation_text=f"EW Target {tr:.2f}",annotation_font=dict(color="#48bb78",size=10),row=1,col=1)
    return fig

def pnl_chart(trades):
    pnls=[t.get("PnL",0) for t in trades]; cum=[0]+list(np.cumsum(pnls))
    c="#48bb78" if cum[-1]>=0 else "#fc8181"
    fill="rgba(72,187,120,0.1)" if c=="#48bb78" else "rgba(252,129,129,0.1)"
    fig=go.Figure(go.Scatter(x=list(range(len(cum))),y=cum,mode="lines+markers",
        fill="tozeroy",fillcolor=fill,line=dict(color=c,width=2),name="Cumulative P&L"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
        height=260,margin=dict(l=0,r=0,t=20,b=0),xaxis_title="Trade #",yaxis_title="P&L",
        font=dict(color="#e2e8f0"))
    return fig

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def sidebar():
    cfg={}
    with st.sidebar:
        st.markdown("## 📈 Smart Investing"); st.markdown("---")
        st.markdown("### 🎯 Instrument")
        tn=st.selectbox("Ticker",list(TICKERS.keys()),key="s_tn")
        sym=(st.text_input("Symbol","RELIANCE.NS",key="s_cust").strip() if tn=="Custom" else TICKERS[tn])
        cfg.update(ticker_name=tn,symbol=sym)

        st.markdown("### ⏱ Timeframe")
        tf=st.selectbox("Interval",list(TF_PERIODS.keys()),index=2,key="s_tf")
        periods=TF_PERIODS[tf]
        period=st.selectbox("Period",periods,index=min(1,len(periods)-1),key="s_period")
        cfg.update(timeframe=tf,period=period)

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
            if ct=="Custom Candle Size": cfg["custom_candle_size"]=st.number_input("Min Body",0.0,value=10.0,key="s_cs")
        else:
            cfg.update(fast_ema=9,slow_ema=15,use_angle_filter=False,
                       min_ema_angle=0.0,crossover_type="Simple Crossover",custom_candle_size=10.0)

        st.markdown("### 📦 Quantity")
        cfg["quantity"]=st.number_input("Qty",1,1_000_000,1,key="s_qty")

        st.markdown("### 🛡 Stop Loss")
        sl_t=st.selectbox("SL Type",["Custom Points","ATR Based","Trailing SL","Reverse EMA Crossover","Risk-Reward Based"],key="s_slt")
        cfg["sl_type"]=sl_t
        if sl_t=="ATR Based":
            cfg["atr_sl_mult"]=st.number_input("ATR Mult SL",0.1,10.0,1.5,0.1,key="s_asm"); cfg["sl_points"]=10.0
        else:
            cfg["sl_points"]=st.number_input("SL Points",0.1,value=10.0,step=0.5,key="s_slp"); cfg["atr_sl_mult"]=1.5

        st.markdown("### 🎯 Target")
        tg_t=st.selectbox("Target Type",["Custom Points","ATR Based Target","Trailing Target","EMA Crossover","Risk-Reward Based"],key="s_tgt")
        cfg["target_type"]=tg_t
        if tg_t=="ATR Based Target":
            cfg["atr_tgt_mult"]=st.number_input("ATR Mult Tgt",0.1,20.0,3.0,0.1,key="s_atm"); cfg["target_points"]=20.0
        elif tg_t=="Risk-Reward Based":
            cfg["rr_ratio"]=st.number_input("R:R",0.1,20.0,2.0,0.1,key="s_rr")
            cfg["target_points"]=20.0; cfg["atr_tgt_mult"]=3.0
        elif tg_t=="Trailing Target":
            cfg["target_points"]=st.number_input("Trail Dist",0.1,value=20.0,step=0.5,key="s_tgp")
            cfg["atr_tgt_mult"]=3.0; st.caption("ℹ️ Display only — won't auto-exit")
        else:
            cfg["target_points"]=st.number_input("Target Pts",0.1,value=20.0,step=0.5,key="s_tgp2")
            cfg["atr_tgt_mult"]=3.0
        cfg.setdefault("rr_ratio",2.0)

        st.markdown("### ⚙️ Controls")
        cd_en=st.checkbox("Cooldown Between Trades",True,key="s_cdn")
        cd_s=st.number_input("Cooldown (s)",1,3600,5,key="s_cds") if cd_en else 5
        no_ovlp=st.checkbox("Prevent Overlapping Trades",True,key="s_novlp")
        cfg.update(cooldown_enabled=cd_en,cooldown_seconds=cd_s,no_overlap=no_ovlp)

        st.markdown("### 🏦 Dhan Broker")
        en_dhan=st.checkbox("Enable Dhan Broker",False,key="s_dhan")
        cfg["enable_dhan"]=en_dhan
        if en_dhan:
            cid=st.text_input("Client ID",key="s_cid")
            tok=st.text_input("Access Token",key="s_tok",type="password")
            cfg.update(dhan_client_id=cid,dhan_access_token=tok)
            if st.button("🔗 Connect & Show IP",use_container_width=True,key="s_conn"):
                with st.spinner("Connecting…"):
                    cl=init_dhan(cid,tok)
                    if cl: st.session_state.dhan_client=cl; register_ip_ui(); st.success("✅ Connected!")
                    else: st.error("❌ Failed")
            opts=st.checkbox("Options Trading",False,key="s_opts")
            cfg["options_trading"]=opts
            if opts:
                cfg["fno_exchange"]=st.selectbox("FNO Exchange",["NSE_FNO","BSE_FNO"],key="s_fnoe")
                cfg["ce_security_id"]=st.text_input("CE Security ID",key="s_ceid")
                cfg["pe_security_id"]=st.text_input("PE Security ID",key="s_peid")
                cfg["options_qty"]=st.number_input("Options Qty",1,value=65,key="s_oqty")
                cfg["options_entry_type"]=st.selectbox("Entry Order",["MARKET","LIMIT"],key="s_oent")
                cfg["options_exit_type"]=st.selectbox("Exit Order",["MARKET","LIMIT"],key="s_oext")
            else:
                cfg["product_type"]=st.selectbox("Product",["INTRADAY","DELIVERY"],key="s_prod")
                cfg["exchange"]=st.selectbox("Exchange",["NSE","BSE"],key="s_exc")
                cfg["security_id"]=st.text_input("Security ID","1594",key="s_sid")
                cfg["dhan_qty"]=st.number_input("Order Qty",1,value=1,key="s_dqty")
                cfg["entry_order_type"]=st.selectbox("Entry Order",["LIMIT","MARKET"],key="s_eord")
                cfg["exit_order_type"]=st.selectbox("Exit Order",["MARKET","LIMIT"],key="s_xord")
    return cfg

# ══════════════════════════════════════════════════════════════════
# SHARED WIDGETS
# ══════════════════════════════════════════════════════════════════
def ltp_banner(cfg):
    ltp=_ts_get("live_ltp"); prev=_ts_get("live_prev_close")
    if ltp is None:
        ltp,prev=fetch_ltp(cfg.get("symbol","^NSEI"))
        _ts_set("live_ltp",ltp); _ts_set("live_prev_close",prev)
    ltp=ltp or 0.0; prev=prev or ltp
    chg=ltp-prev; pct=chg/prev*100 if prev else 0
    arrow="▲" if chg>=0 else "▼"; cls="ltp-up" if chg>=0 else "ltp-down"
    ts_s=datetime.now(IST).strftime("%H:%M:%S IST")
    st.markdown(f"""<div class="ltp-bar">
      <div><div class="ltp-ticker">📈 {cfg.get('ticker_name','—')} · {cfg.get('symbol','')}</div>
      <div class="ltp-price">₹ {ltp:,.2f}</div></div>
      <div class="{cls}">{arrow} {abs(chg):.2f} ({abs(pct):.2f}%)</div>
      <div class="ltp-meta">{cfg.get('timeframe','')} · {cfg.get('period','')} · {cfg.get('strategy','')} | {ts_s}</div>
    </div>""",unsafe_allow_html=True)

def cfg_box(cfg):
    st.markdown(f"""<div class="cfg-box">
      <b>📌 Ticker:</b> {cfg.get('ticker_name','—')} ({cfg.get('symbol','—')})
      &nbsp;|&nbsp;<b>⏱ TF:</b> {cfg.get('timeframe','—')} &nbsp;|&nbsp;<b>📅 Period:</b> {cfg.get('period','—')}<br>
      <b>🧠 Strategy:</b> {cfg.get('strategy','—')} &nbsp;|&nbsp;<b>📦 Qty:</b> {cfg.get('quantity',1)}<br>
      <b>⚡ Fast EMA:</b> {cfg.get('fast_ema',9)} &nbsp;|&nbsp;<b>⚡ Slow EMA:</b> {cfg.get('slow_ema',15)}
      &nbsp;|&nbsp;<b>Crossover:</b> {cfg.get('crossover_type','—')}<br>
      <b>🛡 SL:</b> {cfg.get('sl_type','—')} ({cfg.get('sl_points',10)} pts)
      &nbsp;|&nbsp;<b>🎯 Tgt:</b> {cfg.get('target_type','—')} ({cfg.get('target_points',20)} pts)<br>
      <b>🔄 Cooldown:</b> {'✅ '+str(cfg.get('cooldown_seconds',5))+'s' if cfg.get('cooldown_enabled') else '❌ Off'}
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
        styled=styled.map(lambda v:(f"color:{'#48bb78' if v>0 else '#fc8181'};font-weight:700"
            if isinstance(v,(int,float)) else ""),subset=["PnL"])
    if "SL/Tgt Violated" in df.columns:
        styled=styled.map(lambda v:("background-color:#2d1515;color:#fc8181;font-weight:700" if v is True else ""),
            subset=["SL/Tgt Violated"])
    return styled

def trade_stats(trades):
    if not trades: return
    df=pd.DataFrame(trades)
    wins=df[df["PnL"]>0]; loss=df[df["PnL"]<=0]
    tot=df["PnL"].sum(); acc=len(wins)/len(df)*100 if len(df) else 0
    aw=wins["PnL"].mean() if len(wins) else 0; al=loss["PnL"].mean() if len(loss) else 0
    cols=st.columns(6)
    for col,lbl,val,color in [
        (cols[0],"Total Trades",len(df),"#e2e8f0"),(cols[1],"Winners",len(wins),"#48bb78"),
        (cols[2],"Losers",len(loss),"#fc8181"),(cols[3],"Accuracy",f"{acc:.1f}%","#f6ad55"),
        (cols[4],"Total P&L",f"₹{tot:+.2f}","#48bb78" if tot>=0 else "#fc8181"),
        (cols[5],"Avg W/L",f"₹{aw:.2f} / ₹{al:.2f}","#76e4f7"),
    ]:
        col.markdown(f"""<div class="metric-card"><div style="color:#a0aec0;font-size:11px">{lbl}</div>
          <div style="color:{color};font-size:16px;font-weight:700;margin-top:3px">{val}</div></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ══════════════════════════════════════════════════════════════════
def tab_backtest(cfg):
    ltp_banner(cfg)
    st.markdown("## 🔬 Backtesting Engine")
    st.info("📌 EMA: signal on candle N → entry N+1 open  |  Buy: Low vs SL first then High vs Tgt  |  Sell: High vs SL first then Low vs Tgt")
    if st.button("▶ Run Backtest",type="primary",key="btn_bt"):
        with st.spinner(f"Fetching {cfg.get('symbol')} …"):
            df=fetch_data(cfg["symbol"],cfg["timeframe"],cfg["period"])
        if df is None or df.empty: st.error("❌ No data. Check ticker/interval/period."); return
        df=add_indicators(df,int(cfg.get("fast_ema",9)),int(cfg.get("slow_ema",15)))
        with st.spinner("Running backtest…"):
            trades,violations=run_backtest(df,cfg)
        st.session_state.backtest_trades=trades; st.session_state.backtest_violations=violations
        st.session_state.backtest_df=df; st.session_state.backtest_ran=True

    if not st.session_state.backtest_ran:
        st.markdown("<div style='color:#718096;text-align:center;padding:60px'>Click ▶ Run Backtest to start</div>",unsafe_allow_html=True); return

    trades=st.session_state.backtest_trades; violations=st.session_state.backtest_violations
    df=st.session_state.backtest_df; fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
    if not trades: st.warning("No trades generated."); return

    trade_stats(trades); st.markdown("---")
    if violations:
        st.error(f"⚠️ **{len(violations)} SL/Target Violations** — both SL & Target hit same candle")
        with st.expander("🔍 View Violations"): st.dataframe(style_df(pd.DataFrame(violations)),use_container_width=True)
    else: st.success("✅ No SL/Target violations!")

    st.markdown("### 📊 Backtest Chart")
    st.plotly_chart(make_chart(df.tail(500),trades=trades[:50],title=f"Backtest — {cfg.get('ticker_name')}",fast=fast,slow=slow,h=560),use_container_width=True,key="bt_chart")
    st.markdown("### 📈 Cumulative P&L")
    st.plotly_chart(pnl_chart(trades),use_container_width=True,key="bt_pnl")
    st.markdown(f"### 📋 Trade Log ({len(trades)} trades)")
    tdf=pd.DataFrame(trades)
    co=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target",
        "Candle High","Candle Low","Entry Reason","Exit Reason","PnL","Qty","SL/Tgt Violated"]
    tdf=tdf[[c for c in co if c in tdf.columns]]
    st.dataframe(style_df(tdf),use_container_width=True,height=420)
    st.download_button("⬇ Download CSV",tdf.to_csv(index=False).encode(),"backtest_trades.csv","text/csv",key="bt_dl")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ══════════════════════════════════════════════════════════════════
def tab_live(cfg):
    _sync()
    ltp_banner(cfg)
    st.markdown("## 🚀 Live Trading")

    c1,c2,c3,_=st.columns([1,1,1,3])
    start_btn=c1.button("▶ Start",type="primary",use_container_width=True,key="btn_start")
    stop_btn =c2.button("⏹ Stop",               use_container_width=True,key="btn_stop")
    sq_btn   =c3.button("✖ Squareoff",           use_container_width=True,key="btn_sq")

    if start_btn and not _ts_get("live_running",False):
        _ts_set("live_running",True); _ts_set("live_status","RUNNING")
        _ts_set("live_log",[]); _ts_set("live_position",None)
        t=threading.Thread(target=live_thread,args=(cfg,cfg["symbol"]),daemon=True)
        st.session_state.live_thread=t; t.start(); _sync()
        st.success("✅ Live trading started!")

    if stop_btn and _ts_get("live_running",False):
        _ts_set("live_running",False); _ts_set("live_status","STOPPED"); _sync()
        st.info("⏹ Stopping after current tick…")

    if sq_btn and _ts_get("live_position"):
        pos=_ts_get("live_position"); ltp=_ts_get("live_ltp") or pos["entry"]
        pnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        t2={"Trade #":len(_ts_get("live_trades",[]))+1,"Type":pos["type"].upper(),
            "Entry Time":_fdt(pos["entry_time"]),"Exit Time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "Entry Price":round(pos["entry"],4),"Exit Price":round(ltp,4),
            "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
            "Entry Reason":pos["reason"],"Exit Reason":"Manual Squareoff",
            "PnL":pnl,"Qty":cfg.get("quantity",1),"Source":"🚀 Live"}
        _ts_append("live_trades",t2); _ts_set("live_position",None)
        if cfg.get("enable_dhan") and st.session_state.dhan_client:
            place_order(st.session_state.dhan_client,cfg,pos["type"],ltp,is_exit=True)
        _sync(); st.success(f"✅ Squared off! PnL: {pnl:+.2f}")

    status=_ts_get("live_status","STOPPED"); bc="#48bb78" if status=="RUNNING" else "#fc8181"
    st.markdown(f'<div style="display:inline-block;background:{bc};color:white;padding:4px 14px;'
                f'border-radius:20px;font-size:12px;font-weight:700;margin-bottom:8px">'
                f'{"🟢" if status=="RUNNING" else "🔴"} {status}</div>',unsafe_allow_html=True)

    st.markdown("#### 🔧 Active Configuration"); cfg_box(cfg)

    # ── Last fetched candle + EMA VALUES ────────────────────────
    df=_ts_get("live_data"); ts=_ts_get("live_last_ts")
    if df is not None and not df.empty:
        lr=df.iloc[-1]
        ef=lr.get("ema_fast",float("nan")); es_v=lr.get("ema_slow",float("nan"))
        atr_v=lr.get("atr",0.0) or 0.0
        # Determine crossover state
        cross_state="—"
        if not (np.isnan(ef) or np.isnan(es_v)):
            if ef>es_v: cross_state=f'<span style="color:#48bb78">▲ Fast ABOVE Slow (Bullish)</span>'
            else:       cross_state=f'<span style="color:#fc8181">▼ Fast BELOW Slow (Bearish)</span>'
        st.markdown(f"""<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;
          padding:10px 16px;font-size:12px;color:#a0aec0;margin:6px 0;line-height:2.0">
          📡 <b style="color:#76e4f7">Last Candle</b> &nbsp;
          O:<b>{lr['open']:.2f}</b> &nbsp; H:<b>{lr['high']:.2f}</b> &nbsp;
          L:<b>{lr['low']:.2f}</b> &nbsp; C:<b>{lr['close']:.2f}</b>
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <b>EMA({cfg.get('fast_ema',9)}):</b>
          <b style="color:#f6ad55;font-size:13px">{ef:.2f}</b>
          &nbsp;&nbsp;
          <b>EMA({cfg.get('slow_ema',15)}):</b>
          <b style="color:#76e4f7;font-size:13px">{es_v:.2f}</b>
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <b>ATR:</b> <b style="color:#e2e8f0">{atr_v:.2f}</b>
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <b>Cross:</b> {cross_state}
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <b style="color:#48bb78">Fetched: {ts.strftime('%H:%M:%S') if ts else '—'} IST</b>
        </div>""",unsafe_allow_html=True)

    # ── Current position ────────────────────────────────────────
    pos=_ts_get("live_position")
    st.markdown("#### 📌 Current Position")
    if pos:
        ltp=_ts_get("live_ltp") or pos["entry"]
        upnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        bc2="#48bb78" if upnl>=0 else "#fc8181"; ptc="#48bb78" if pos["type"]=="buy" else "#fc8181"
        st.markdown(f"""<div style="background:{'rgba(72,187,120,0.12)' if upnl>=0 else 'rgba(252,129,129,0.12)'};
          border:1px solid {bc2};border-radius:10px;padding:14px">
          <b style="color:{ptc};font-size:16px">{'🔼 BUY' if pos['type']=='buy' else '🔽 SELL'}</b>
          &nbsp;&nbsp; Entry:<b>{pos['entry']:.2f}</b>
          &nbsp;|&nbsp;LTP:<b style="color:{bc2}">{ltp:.2f}</b>
          &nbsp;|&nbsp;SL:<b style="color:#fc8181">{pos['sl']:.2f}</b>
          &nbsp;|&nbsp;Target:<b style="color:#48bb78">{pos['tgt']:.2f}</b>
          &nbsp;|&nbsp;Unrealised P&L:<b style="color:{bc2}">₹{upnl:+.2f}</b>
          &nbsp;|&nbsp;Time:{_fdt(pos['entry_time'])}
          <br><span style="color:#a0aec0;font-size:11px">Reason: {pos['reason']}</span>
        </div>""",unsafe_allow_html=True)
    else: st.info("📭 No open position")

    if df is not None and not df.empty:
        fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
        st.plotly_chart(make_chart(df.tail(300),title=f"Live — {cfg.get('ticker_name')}",
            fast=fast,slow=slow,live_pos=pos,h=500),use_container_width=True,key="live_chart")

    # ── Elliott Wave panel ──────────────────────────────────────
    st.markdown("---"); st.markdown("#### 🌊 Elliott Wave Analysis")
    ew=_ts_get("live_ew",{})
    if ew and ew.get("status","") not in ("","Forming","Insufficient data"):
        s=ew.get("status","—"); d=ew.get("direction","UNCLEAR")
        cw=ew.get("current_wave","—"); cp=ew.get("current_price",0)
        dc={"BULLISH":"#48bb78","BEARISH":"#fc8181","CORRECTIVE_DOWN":"#f6ad55","CORRECTIVE_UP":"#76e4f7","UNCLEAR":"#a0aec0"}.get(d,"#a0aec0")
        cols=st.columns(4)
        for col,lbl,val,color in [(cols[0],"Structure",s,dc),(cols[1],"Direction",d,dc),
                                  (cols[2],"Current Wave",cw,"#e2e8f0"),(cols[3],"Price",f"{cp:,.2f}","#e2e8f0")]:
            col.markdown(f"""<div class="metric-card"><div style="color:#a0aec0;font-size:11px">{lbl}</div>
              <div style="color:{color};font-size:14px;font-weight:700;margin-top:3px">{val}</div></div>""",unsafe_allow_html=True)

        # Completed wave details
        det=ew.get("wave_detail",{})
        if det:
            st.markdown("**📊 Wave Details:**")
            html='<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">'
            for k,v in det.items(): html+=f'<div class="wave-lbl"><b style="color:#f6ad55">{k}</b>: {v}</div>'
            st.markdown(html+"</div>",unsafe_allow_html=True)

        # Next targets
        tgts=ew.get("next_targets",{})
        if tgts:
            st.markdown("**🎯 Wave Projection Targets:**")
            tcols=st.columns(len(tgts))
            for i,(n,v) in enumerate(tgts.items()):
                cl="#48bb78" if v>cp else "#fc8181"
                tcols[i].markdown(f"""<div class="metric-card" style="text-align:center">
                  <div style="color:#a0aec0;font-size:11px">{n}</div>
                  <div style="color:{cl};font-size:14px;font-weight:700">{v:,.2f}</div></div>""",unsafe_allow_html=True)

        # ── EW Trade Recommendation (entry/SL/target) ──────────
        entry_rec=ew.get("entry_rec"); sl_rec=ew.get("sl_rec"); tgt_rec=ew.get("tgt_rec")
        bias=ew.get("trade_bias","")
        if entry_rec or bias:
            bias_color="#48bb78" if bias=="BUY" else "#fc8181" if bias=="SELL" else "#a0aec0"
            bias_arrow="🔼 BUY" if bias=="BUY" else "🔽 SELL" if bias=="SELL" else "—"
            rr="—"
            if entry_rec and sl_rec and tgt_rec:
                sl_d=abs(entry_rec-sl_rec); tg_d=abs(tgt_rec-entry_rec)
                rr=f"{tg_d/sl_d:.2f}" if sl_d else "—"
            st.markdown(f"""<div class="ew-rec">
              <div style="font-size:14px;font-weight:700;margin-bottom:10px;color:#90cdf4">
                🌊 Elliott Wave Trade Recommendation</div>
              <div style="display:flex;flex-wrap:wrap;gap:16px;font-size:13px">
                <div><span style="color:#a0aec0">Bias</span><br>
                  <b style="color:{bias_color};font-size:15px">{bias_arrow}</b></div>
                <div><span style="color:#a0aec0">Suggested Entry</span><br>
                  <b style="color:#f6ad55;font-size:15px">{f'{entry_rec:,.2f}' if entry_rec else '—'}</b></div>
                <div><span style="color:#a0aec0">EW Stop Loss</span><br>
                  <b style="color:#fc8181;font-size:15px">{f'{sl_rec:,.2f}' if sl_rec else '—'}</b></div>
                <div><span style="color:#a0aec0">EW Target (W5 100%)</span><br>
                  <b style="color:#48bb78;font-size:15px">{f'{tgt_rec:,.2f}' if tgt_rec else '—'}</b></div>
                <div><span style="color:#a0aec0">EW Risk:Reward</span><br>
                  <b style="color:#76e4f7;font-size:15px">1 : {rr}</b></div>
                <div><span style="color:#a0aec0">Strategy SL</span><br>
                  <b style="color:#fc8181">{cfg.get('sl_points',10)} pts ({cfg.get('sl_type','Custom Points')})</b></div>
                <div><span style="color:#a0aec0">Strategy Target</span><br>
                  <b style="color:#48bb78">{cfg.get('target_points',20)} pts ({cfg.get('target_type','Custom Points')})</b></div>
              </div>
              <div style="margin-top:8px;font-size:11px;color:#718096">
                💡 EW levels are wave-structure based. Your active SL/Target from sidebar will be used for actual trade execution.
                Adjust sidebar SL/Target to align with EW recommendation if desired.
              </div>
            </div>""",unsafe_allow_html=True)

        am=ew.get("after_msg","")
        if am: st.info(f"💡 {am}")

        if df is not None and not df.empty and len(ew.get("waves",[]))>0:
            fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
            st.plotly_chart(make_ew_chart(df,ew,fast,slow),use_container_width=True,key="ew_live")
    else:
        st.info(f"🌊 {ew.get('after_msg','Elliott Wave analysis will appear once live trading starts and data loads.')}")

    # ── Activity log ────────────────────────────────────────────
    st.markdown("#### 📟 Activity Log")
    logs=_ts_get("live_log",[])
    log_html='<div class="log-box">'
    for ln in reversed(logs[-80:]):
        c=("#fc8181" if "ERR" in ln or "WARN" in ln
           else "#48bb78" if ("ENTRY" in ln or "STARTED" in ln or "BUY" in ln)
           else "#f6ad55" if "EXIT" in ln or "SIGNAL" in ln else "#a0aec0")
        log_html+=f'<div style="color:{c}">{ln}</div>'
    st.markdown(log_html+"</div>",unsafe_allow_html=True)

    # ── Completed trades (visible while running) ─────────────
    lt=_ts_get("live_trades",[])
    if lt:
        st.markdown(f"#### ✅ Completed Trades ({len(lt)})")
        trade_stats(lt)
        ltd=pd.DataFrame(lt)
        cs=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target","Entry Reason","Exit Reason","PnL","Qty"]
        st.dataframe(style_df(ltd[[c for c in cs if c in ltd.columns]]),use_container_width=True,height=260)
        st.plotly_chart(pnl_chart(lt),use_container_width=True,key="live_pnl")

    # ── AUTO-REFRESH when live trading is running ───────────────
    # This ensures Simple Buy/Sell position shows immediately on UI
    if _ts_get("live_running",False):
        time.sleep(2)
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# TAB 3 — TRADE HISTORY
# ══════════════════════════════════════════════════════════════════
def tab_history(cfg):
    _sync(); ltp_banner(cfg); st.markdown("## 📚 Trade History")
    all_trades=[]
    for t in st.session_state.backtest_trades: tc=t.copy(); tc["Source"]="🔬 Backtest"; all_trades.append(tc)
    for t in _ts_get("live_trades",[]): tc=t.copy(); tc.setdefault("Source","🚀 Live"); all_trades.append(tc)
    if not all_trades:
        st.markdown("<div style='color:#718096;text-align:center;padding:60px'>No trades yet. Run backtest or start live trading.</div>",unsafe_allow_html=True); return
    df=pd.DataFrame(all_trades)
    f1,f2,f3=st.columns(3)
    src=f1.selectbox("Source",["All"]+sorted(df["Source"].unique().tolist()),key="h_src")
    typ=f2.selectbox("Type",["All","BUY","SELL"],key="h_typ")
    show_viol=f3.checkbox("Only Violations",False,key="h_viol")
    if src!="All": df=df[df["Source"]==src]
    if typ!="All": df=df[df["Type"]==typ]
    if show_viol and "SL/Tgt Violated" in df.columns: df=df[df["SL/Tgt Violated"]==True]
    st.markdown(f"#### 📊 Summary — {len(df)} trades"); trade_stats(df.to_dict("records"))
    if not df.empty:
        st.plotly_chart(pnl_chart(df.to_dict("records")),use_container_width=True,key="hist_pnl")
        co=["Trade #","Source","Type","Entry Time","Exit Time","Entry Price","Exit Price",
            "SL","Target","Entry Reason","Exit Reason","PnL","Qty","SL/Tgt Violated"]
        sc=[c for c in co if c in df.columns]
        st.dataframe(style_df(df[sc]),use_container_width=True,height=460)
        st.download_button("⬇ Download CSV",df[sc].to_csv(index=False).encode(),"trade_history.csv","text/csv",key="h_dl")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════
def tab_optimize(cfg):
    ltp_banner(cfg)
    st.markdown("## ⚙️ Strategy Optimizer")
    st.info("🔍 Grid-searches EMA / SL / Target combinations on your selected ticker and period. "
            "Click **Apply** on any result to instantly load those parameters into the sidebar.")

    st.markdown("### 🎛 Parameter Ranges to Explore")
    col1,col2,col3=st.columns(3)
    with col1:
        st.markdown("**EMA Fast (comma-separated)**")
        fast_vals=st.text_input("Fast EMA values","5,7,9,12",key="opt_fast")
        st.markdown("**EMA Slow (comma-separated)**")
        slow_vals=st.text_input("Slow EMA values","15,20,26",key="opt_slow")
    with col2:
        st.markdown("**SL Points (comma-separated)**")
        sl_vals=st.text_input("SL Points","5,10,15,20",key="opt_sl")
        st.markdown("**Target Points (comma-separated)**")
        tgt_vals=st.text_input("Target Points","10,20,30,40",key="opt_tgt")
    with col3:
        desired_acc=st.number_input("Minimum Accuracy %",0.0,100.0,50.0,5.0,key="opt_acc")
        max_combos=st.number_input("Max Combinations",10,500,100,10,key="opt_max")
        opt_strat=st.selectbox("Strategy to Optimize",["EMA Crossover","Simple Buy","Simple Sell"],key="opt_strat")

    st.caption(f"ℹ️ Using ticker: **{cfg.get('ticker_name','—')} ({cfg.get('symbol','—')})** | "
               f"TF: **{cfg.get('timeframe','—')}** | Period: **{cfg.get('period','—')}** | "
               f"SL Type: **{cfg.get('sl_type','Custom Points')}** | Tgt Type: **{cfg.get('target_type','Custom Points')}**")

    run_opt=st.button("🚀 Run Optimization",type="primary",key="btn_opt")
    if run_opt:
        # Parse values
        def parse_vals(s,cast=float):
            try: return [cast(x.strip()) for x in s.split(",") if x.strip()]
            except: return []
        fv=parse_vals(fast_vals,int); sv=parse_vals(slow_vals,int)
        slv=parse_vals(sl_vals,float); tv=parse_vals(tgt_vals,float)
        if not (fv and sv and slv and tv):
            st.error("❌ Please enter valid comma-separated values for all parameters."); return
        # Only keep fast < slow combos
        param_grid={"fast_ema":fv,"slow_ema":sv,"sl_points":slv,"target_points":tv}
        cfg_base={**cfg,"strategy":opt_strat,"sl_type":"Custom Points","target_type":"Custom Points"}
        with st.spinner(f"Fetching data for {cfg.get('symbol')} …"):
            df_raw=fetch_data(cfg["symbol"],cfg["timeframe"],cfg["period"])
        if df_raw is None or df_raw.empty:
            st.error("❌ No data fetched. Check ticker/interval/period."); return
        total=len(list(itertools.product(*param_grid.values())))
        actual=min(total,int(max_combos))
        prog=st.progress(0,text=f"Running {actual} combinations…")
        results=[]
        combos=list(itertools.product(*param_grid.values()))[:actual]
        keys=list(param_grid.keys())
        for idx,combo in enumerate(combos):
            trial=dict(cfg_base)
            for k,v in zip(keys,combo): trial[k]=v
            if opt_strat=="EMA Crossover" and trial["fast_ema"]>=trial["slow_ema"]: continue
            try:
                trades,_=run_backtest(df_raw,trial)
                if not trades: continue
                df_t=pd.DataFrame(trades)
                wins=len(df_t[df_t["PnL"]>0]); total_t=len(df_t)
                acc=wins/total_t*100 if total_t else 0; pnl=df_t["PnL"].sum()
                if acc>=desired_acc:
                    results.append({**{k:trial[k] for k in keys},
                        "Trades":total_t,"Wins":wins,"Accuracy":round(acc,1),
                        "Total PnL":round(pnl,2),"Avg PnL/Trade":round(pnl/total_t,2)})
            except Exception: pass
            prog.progress((idx+1)/actual,text=f"Running… {idx+1}/{actual}")
        prog.empty()
        results.sort(key=lambda x:(-x["Accuracy"],-x["Total PnL"]))
        st.session_state.opt_results=results[:20]; st.session_state.opt_ran=True
        st.success(f"✅ Done! Found **{len(results)}** combinations meeting ≥{desired_acc}% accuracy out of {actual} tested.")

    if not st.session_state.opt_ran: return

    results=st.session_state.opt_results
    if not results:
        st.warning("😕 No combinations met the desired accuracy threshold. Try lowering the minimum accuracy."); return

    st.markdown(f"### 🏆 Top {len(results)} Results (sorted by Accuracy then P&L)")
    st.markdown("Click **✅ Apply** on any row to load those parameters into the sidebar.")

    # Show sortable table
    res_df=pd.DataFrame(results)
    def style_opt(df):
        def rc(row):
            acc=row.get("Accuracy",0)
            if acc>=70: c="rgba(72,187,120,0.15)"
            elif acc>=55: c="rgba(246,173,85,0.10)"
            else: c="rgba(252,129,129,0.08)"
            return [f"background-color:{c}"]*len(row)
        styled=df.style.apply(rc,axis=1)
        styled=styled.map(lambda v:(f"color:{'#48bb78' if v>0 else '#fc8181'};font-weight:700"
            if isinstance(v,(int,float)) else ""),subset=["Total PnL","Avg PnL/Trade"])
        styled=styled.map(lambda v:(f"color:{'#48bb78' if v>=65 else '#f6ad55' if v>=50 else '#fc8181'};font-weight:700"
            if isinstance(v,(int,float)) else ""),subset=["Accuracy"])
        return styled
    st.dataframe(style_opt(res_df),use_container_width=True,height=340)

    st.markdown("### 🎯 Apply a Result to Sidebar")
    st.markdown("Select a result number and click Apply to update all sidebar parameters:")
    sel_cols=st.columns([2,1,4])
    sel_idx=sel_cols[0].number_input("Result #",1,len(results),1,key="opt_sel_idx")-1
    apply_btn=sel_cols[1].button("✅ Apply to Sidebar",type="primary",key="btn_apply_opt")

    # Show preview of selected result
    sel=results[sel_idx]
    sel_cols[2].markdown(f"""<div style="background:#1a1f2e;border-radius:8px;padding:8px 14px;font-size:12px;color:#cbd5e0;margin-top:4px">
      <b>Preview #{ sel_idx+1}:</b> &nbsp;
      Fast EMA: <b style="color:#f6ad55">{sel.get('fast_ema','—')}</b> &nbsp;|&nbsp;
      Slow EMA: <b style="color:#76e4f7">{sel.get('slow_ema','—')}</b> &nbsp;|&nbsp;
      SL: <b style="color:#fc8181">{sel.get('sl_points','—')} pts</b> &nbsp;|&nbsp;
      Tgt: <b style="color:#48bb78">{sel.get('target_points','—')} pts</b> &nbsp;|&nbsp;
      Accuracy: <b style="color:#f6ad55">{sel.get('Accuracy','—')}%</b> &nbsp;|&nbsp;
      P&L: <b style="color:#48bb78">₹{sel.get('Total PnL',0):+.2f}</b>
    </div>""",unsafe_allow_html=True)

    if apply_btn:
        # Write values directly into sidebar widget session_state keys
        if "fast_ema" in sel: st.session_state["s_fe"]=int(sel["fast_ema"])
        if "slow_ema" in sel: st.session_state["s_se"]=int(sel["slow_ema"])
        if "sl_points" in sel: st.session_state["s_slp"]=float(sel["sl_points"])
        if "target_points" in sel:
            st.session_state["s_tgp"]=float(sel["target_points"])
            st.session_state["s_tgp2"]=float(sel["target_points"])
        # Also set strategy
        st.session_state["s_strat"]=opt_strat
        st.success(f"✅ Applied result #{sel_idx+1} to sidebar! "
                   f"Fast EMA={sel.get('fast_ema')}, Slow EMA={sel.get('slow_ema')}, "
                   f"SL={sel.get('sl_points')} pts, Target={sel.get('target_points')} pts. "
                   f"Accuracy={sel.get('Accuracy')}%, P&L=₹{sel.get('Total PnL',0):+.2f}")
        st.info("↩️ The sidebar has been updated. You can now run Backtest or Live Trading with these optimized parameters.")

    # Download
    st.download_button("⬇ Download Optimization Results",
        res_df.to_csv(index=False).encode(),"optimization_results.csv","text/csv",key="opt_dl")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    st.markdown("""<div class="app-hdr">
      <div><h1>📈 Smart Investing</h1>
      <p>Professional Algorithmic Trading Platform · NSE · BSE · Crypto · Commodities</p></div>
      <div style="color:#90cdf4;font-size:12px;text-align:right">EMA · Elliott Wave · Optimizer · Dhan Broker</div>
    </div>""",unsafe_allow_html=True)
    cfg=sidebar()
    tab1,tab2,tab3,tab4=st.tabs(["🔬 Backtesting","🚀 Live Trading","📚 Trade History","⚙️ Optimizer"])
    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history(cfg)
    with tab4: tab_optimize(cfg)

if __name__=="__main__":
    main()
