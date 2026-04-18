"""
Smart Investing — Algorithmic Trading Platform
"""
import threading, time, math, warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Smart Investing", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main .block-container{padding-top:.6rem;padding-bottom:1rem;max-width:100%}
h1,h2,h3,h4{color:#e6edf3}
.mc{background:linear-gradient(135deg,#161b22,#1c2128);border:1px solid #30363d;
    border-radius:10px;padding:12px 16px;text-align:center;margin:3px 0}
.mc-lbl{color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:1px}
.mc-val{color:#e6edf3;font-size:20px;font-weight:700;margin-top:3px}
.bd{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700}
.bd-buy{background:#0d2818;color:#3fb950;border:1px solid #238636}
.bd-sell{background:#2d1a1a;color:#f85149;border:1px solid #da3633}
.al{padding:10px 14px;border-radius:8px;margin:4px 0;font-size:13px}
.al-info{background:#1c2d3e;border-left:3px solid #58a6ff;color:#a0cfff}
.al-warn{background:#2d2318;border-left:3px solid #e3b341;color:#f0c672}
.al-err{background:#2d1a1a;border-left:3px solid #f85149;color:#ff8080}
.al-ok{background:#1a2d1a;border-left:3px solid #3fb950;color:#70d080}
.pos{background:linear-gradient(135deg,#1a2d1a,#161b22);border:1px solid #238636;
     border-radius:10px;padding:14px;margin:6px 0}
.pos-sell{background:linear-gradient(135deg,#2d1a1a,#161b22)!important;border-color:#da3633!important}
.pos-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-top:8px}
.pf-lbl{color:#8b949e;font-size:10px;text-transform:uppercase}
.pf-val{color:#e6edf3;font-size:15px;font-weight:600;margin-top:2px}
.pf-val.g{color:#3fb950}.pf-val.r{color:#f85149}
.cr{display:flex;gap:18px;flex-wrap:wrap;background:#161b22;border:1px solid #30363d;
    border-radius:8px;padding:10px 14px;margin:4px 0}
.cf-lbl{color:#8b949e;font-size:10px;text-transform:uppercase}
.cf-val{color:#e6edf3;font-size:14px;font-weight:600}
.cb{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin:4px 0}
.cb-row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d}
.cb-row:last-child{border-bottom:none}
.cb-k{color:#8b949e;font-size:12px}.cb-v{color:#e6edf3;font-size:12px;font-weight:600}
.log{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:8px;
     height:200px;overflow-y:auto;font-family:monospace;font-size:11px}
.li{color:#58a6ff}.lb{color:#3fb950}.ls{color:#f85149}.lw{color:#e3b341}.le{color:#a371f7}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:8px 0}
.sc{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}
.sc-lbl{color:#8b949e;font-size:10px;text-transform:uppercase}
.sc-val{color:#e6edf3;font-size:18px;font-weight:700;margin-top:3px}
.tw{overflow-x:auto}
table.tt{width:100%;border-collapse:collapse;font-size:11px}
table.tt th{background:#161b22;color:#8b949e;padding:7px 8px;text-align:left;
            border-bottom:2px solid #30363d;white-space:nowrap}
table.tt td{padding:5px 8px;border-bottom:1px solid #21262d;white-space:nowrap}
table.tt tr:hover td{background:#1c2128}
.ppos{color:#3fb950;font-weight:700}.pneg{color:#f85149;font-weight:700}
.vrow td{background:#2d1a1a!important}
[data-testid="stSidebar"]{background:#0d1117!important}
.stTabs [data-baseweb="tab-list"]{background:#161b22;border-radius:8px;padding:3px;gap:2px}
.stTabs [data-baseweb="tab"]{background:transparent;color:#8b949e;border-radius:6px}
.stTabs [data-baseweb="tab"][aria-selected="true"]{background:#21262d;color:#e6edf3}
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone("Asia/Kolkata")

TICKERS = {"Nifty 50":"^NSEI","BankNifty":"^NSEBANK","Sensex":"^BSESN",
           "BTC-USD":"BTC-USD","ETH-USD":"ETH-USD","Gold":"GC=F","Silver":"SI=F","Custom":""}
TF_PERIODS = {
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo"],"15m":["1d","5d","7d","1mo"],
    "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y"],
    "1wk":["1mo","3mo","6mo","1y","2y","5y","10y"],
}
WARMUP = {"1m":"5d","5m":"1mo","15m":"1mo","1h":"3mo","1d":"2y","1wk":"5y"}
STRATEGIES = ["EMA Crossover","Simple Buy","Simple Sell",
              "Anticipatory EMA Crossover","Elliott Wave Auto"]
SL_TYPES = ["Custom Points","ATR Based","Trailing SL","Risk-Reward Based","Auto SL",
            "EMA Reverse Crossover","Trail with Swing Low/High","Trail with Candle Low/High",
            "Support/Resistance Based","Volatility Based"]
TARGET_TYPES = ["Custom Points","ATR Based","Trailing Target (Display Only)",
                "Risk-Reward Based","Auto Target","EMA Crossover Exit",
                "Trail with Swing High/Low","Trail with Candle High/Low",
                "Support/Resistance Based","Book Profit: T1 then T2","Volatility Based"]
CO_TYPES = ["Simple Crossover","Custom Candle Size","ATR Based Candle Size"]
ORDER_TYPES = ["Market Order","Limit Order"]
FNO_SEGS = ["NSE_FNO","BSE_FNO"]

# ── Thread-safe state ─────────────────────────────────────────────────────────
_LOCK = threading.RLock()
_TS: Dict[str,Any] = {
    "running":False,"position":None,"trades":[],"log":[],
    "candle":None,"ltp":None,"ema_f":None,"ema_s":None,
    "tick":0,"error":None,"waves":{},"cfg":{},"stop":None,
    "cooldown_until":None,
}

def _get(k,d=None):
    with _LOCK: return _TS.get(k,d)
def _set(k,v):
    with _LOCK: _TS[k]=v
def _append(k,v):
    with _LOCK:
        if k not in _TS: _TS[k]=[]
        _TS[k].append(v)
def _log(msg,level="i"):
    ts=datetime.now(IST).strftime("%H:%M:%S")
    with _LOCK:
        _TS["log"].append({"t":ts,"m":msg,"l":level})
        if len(_TS["log"])>300: _TS["log"]=_TS["log"][-300:]

# ── Data fetching ─────────────────────────────────────────────────────────────
def _sf(v) -> float:
    try:
        if isinstance(v,pd.Series): v=v.iloc[0]
        if isinstance(v,pd.DataFrame): v=v.iloc[0,0]
        return float(v)
    except: return float("nan")

def _flatten(df:pd.DataFrame)->pd.DataFrame:
    if isinstance(df.columns,pd.MultiIndex):
        df=df.copy(); df.columns=[str(c[0]) for c in df.columns]
        df=df.loc[:,~df.columns.duplicated(keep="first")]
    return df

def fetch_ohlcv(symbol:str,interval:str,period:str,warmup:bool=False)->Optional[pd.DataFrame]:
    try:
        p=WARMUP.get(interval,period) if warmup else period
        raw=yf.download(symbol,interval=interval,period=p,
                        auto_adjust=True,progress=False,threads=False)
        if raw is None or (isinstance(raw,pd.DataFrame) and raw.empty): return None
        df=_flatten(raw)
        want=[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        if not want: return None
        df=df[want].copy()
        for col in ["Open","High","Low","Close"]:
            if col in df.columns:
                if isinstance(df[col],pd.DataFrame): df[col]=df[col].iloc[:,0]
                df[col]=pd.to_numeric(df[col],errors="coerce")
        df.dropna(subset=["Close"],inplace=True)
        return df if len(df)>=2 else None
    except: return None

# ── Indicators ────────────────────────────────────────────────────────────────
def ema(s,n): return s.ewm(span=n,adjust=False,min_periods=1).mean()
def atr_ind(df,n=14):
    pc=df["Close"].shift(1)
    tr=pd.concat([df["High"]-df["Low"],(df["High"]-pc).abs(),(df["Low"]-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False,min_periods=1).mean()
def rsi_ind(s,n=14):
    d=s.diff(); g=d.clip(lower=0).ewm(span=n,adjust=False,min_periods=1).mean()
    l=(-d).clip(lower=0).ewm(span=n,adjust=False,min_periods=1).mean()
    return 100-100/(1+g/l.replace(0,1e-9))

def add_ind(df,cfg):
    df=df.copy()
    fast=int(cfg.get("ema_fast",9)); slow=int(cfg.get("ema_slow",15))
    df["ema_fast"]=ema(df["Close"],fast); df["ema_slow"]=ema(df["Close"],slow)
    df["atr"]=atr_ind(df,14); df["rsi"]=rsi_ind(df["Close"],14)
    df["swing_lo"]=df["Low"].rolling(10,min_periods=1).min()
    df["swing_hi"]=df["High"].rolling(10,min_periods=1).max()
    df["std"]=df["Close"].rolling(20,min_periods=1).std().fillna(0)
    return df

# ── Signals ───────────────────────────────────────────────────────────────────
def sig_ema(df,cfg):
    f,s=df["ema_fast"],df["ema_slow"]
    up=(f>s)&(f.shift(1)<=s.shift(1)); dn=(f<s)&(f.shift(1)>=s.shift(1))
    ct=cfg.get("crossover_type","Simple Crossover"); body=(df["Close"]-df["Open"]).abs()
    if ct=="Custom Candle Size":
        cs=float(cfg.get("custom_candle_size",10)); up&=body>=cs; dn&=body>=cs
    elif ct=="ATR Based Candle Size":
        thr=df["atr"]*float(cfg.get("atr_candle_mult",0.5)); up&=body>=thr; dn&=body>=thr
    ang=float(cfg.get("min_crossover_angle",0))
    if ang>0:
        gc=(f-s).diff().abs(); up&=gc>=ang; dn&=gc>=ang
    out=pd.Series(0,index=df.index); out[up]=1; out[dn]=-1; return out

def sig_antic(df,cfg):
    out=pd.Series(0,index=df.index); f,s=df["ema_fast"],df["ema_slow"]; gap=f-s
    for i in range(3,len(df)):
        g,gp=gap.iloc[i],gap.iloc[i-1]
        if gp==0: continue
        conv=abs(g-gp)/(abs(gp)+1e-8)
        if gp<0 and g<0 and abs(g)<abs(gp) and conv>0.12 and df["rsi"].iloc[i]>45: out.iloc[i]=1
        elif gp>0 and g>0 and abs(g)<abs(gp) and conv>0.12 and df["rsi"].iloc[i]<55: out.iloc[i]=-1
    return out

def sig_wave(df,cfg):
    out=pd.Series(0,index=df.index)
    if len(df)<30: return out
    r=elliott_analyze(df); sig=r.get("signal")
    if sig=="buy": out.iloc[-1]=1
    elif sig=="sell": out.iloc[-1]=-1
    return out

def get_sigs(df,cfg):
    st=cfg.get("strategy","EMA Crossover")
    if st=="Simple Buy": return pd.Series(1,index=df.index)
    if st=="Simple Sell": return pd.Series(-1,index=df.index)
    if st=="EMA Crossover": return sig_ema(df,cfg)
    if st=="Anticipatory EMA Crossover": return sig_antic(df,cfg)
    if st=="Elliott Wave Auto": return sig_wave(df,cfg)
    return pd.Series(0,index=df.index)

def last_sig(df,cfg):
    st=cfg.get("strategy","EMA Crossover")
    if st=="Simple Buy": return 1
    if st=="Simple Sell": return -1
    try: return int(get_sigs(df,cfg).iloc[-1])
    except: return 0

# ── Elliott Wave ──────────────────────────────────────────────────────────────
def zigzag(df,dev=0.03):
    hi,lo,cl=df["High"].values,df["Low"].values,df["Close"].values; n=len(cl)
    pivots=[]; direction=1 if cl[min(4,n-1)]>cl[0] else -1
    idx,px=0,hi[0] if direction==1 else lo[0]
    for i in range(1,n):
        if direction==1:
            if hi[i]>px: idx,px=i,hi[i]
            elif cl[i]<px*(1-dev): pivots.append((idx,px,"H")); direction,idx,px=-1,i,lo[i]
        else:
            if lo[i]<px: idx,px=i,lo[i]
            elif cl[i]>px*(1+dev): pivots.append((idx,px,"L")); direction,idx,px=1,i,hi[i]
    ptype="H" if direction==1 else "L"
    if not pivots or pivots[-1][0]!=idx: pivots.append((idx,px,ptype))
    if not pivots: return pd.DataFrame(columns=["bar_idx","price","ptype","datetime"])
    out=pd.DataFrame(pivots,columns=["bar_idx","price","ptype"])
    out["datetime"]=out["bar_idx"].apply(lambda i:df.index[i]); return out

def elliott_analyze(df):
    res=dict(status="Insufficient data",waves=[],current_wave=None,direction=None,
             signal=None,sl=None,tp1=None,tp2=None,next_target=None,
             pivots=pd.DataFrame())
    if len(df)<20: return res
    av=_sf(df["atr"].iloc[-1]); px=_sf(df["Close"].iloc[-1])
    dev=max(0.015,min(0.07,av/max(px,1e-9)*2.5))
    pivots=zigzag(df,dev); res["pivots"]=pivots
    if len(pivots)<5: res["status"]="Not enough pivots"; return res
    prices=pivots["price"].values; ptypes=pivots["ptype"].values
    times=pivots["datetime"].values; n=len(prices)
    best=None
    for start in range(max(0,n-14),max(0,n-4)):
        cand=[start]; exp="H" if ptypes[start]=="L" else "L"
        for j in range(start+1,min(start+12,n)):
            if ptypes[j]==exp: cand.append(j); exp="H" if exp=="L" else "L"
            if len(cand)==6: break
        if len(cand)<5: continue
        p=[prices[i] for i in cand]; bull=p[1]>p[0]
        ok1=p[2]>p[0] if bull else p[2]<p[0]
        l1=abs(p[1]-p[0]); l3=abs(p[3]-p[2]) if len(p)>3 else 0; l5=abs(p[5]-p[4]) if len(p)>5 else 0
        ok2=not(l3<l1 and l3<l5) if len(p)>5 else True
        if ok1 and ok2: best=cand; break
    if best and len(best)>=5:
        p=[prices[i] for i in best]; t=[times[i] for i in best]; bull=p[1]>p[0]; direc=1 if bull else -1
        waves_info=[]
        for k in range(min(5,len(p)-1)):
            waves_info.append({"wave":str(k+1),"start_price":p[k],"end_price":p[k+1],
                               "start_time":t[k],"end_time":t[k+1],"status":"Complete",
                               "direction":"Up" if p[k+1]>p[k] else "Down"})
        res["waves"]=waves_info; res["direction"]="Up" if bull else "Down"
        if len(best)==5:
            w0,w1,w2,w3,w4=p[0],p[1],p[2],p[3],p[4]; l1=abs(w1-w0)
            tp1=w4+direc*l1*0.618; tp2=w4+direc*l1
            res.update(current_wave="5",signal="buy" if bull else "sell",
                       sl=w4*(0.995 if bull else 1.005),tp1=tp1,tp2=tp2,
                       next_target=tp2,status="Wave 5 forming")
        elif len(best)>=6:
            res.update(current_wave="A",status="Impulse complete → ABC correction")
    else: res["status"]="Partial wave pattern"
    return res

# ── SL / Target ───────────────────────────────────────────────────────────────
def calc_sl(entry,side,df,cfg,idx):
    slt=cfg.get("sl_type","Custom Points"); pts=float(cfg.get("sl_points",10))
    sign=-1 if side=="buy" else 1
    try: av=_sf(df["atr"].iloc[idx]); rl=df.iloc[idx]
    except: return entry+sign*pts
    if slt=="Custom Points": return entry+sign*pts
    if slt=="ATR Based": return entry+sign*av*float(cfg.get("atr_mult_sl",1.5))
    if slt in("Auto SL","Volatility Based"):
        std=_sf(df["std"].iloc[idx]) if "std" in df.columns else av
        return entry+sign*max(av*1.5,std*2.0)
    if slt=="Risk-Reward Based": return entry+sign*float(cfg.get("target_points",20))/float(cfg.get("rr_ratio",2))
    if slt=="EMA Reverse Crossover": return _sf(df["ema_slow"].iloc[idx])
    if slt=="Trail with Swing Low/High":
        return(_sf(df["swing_lo"].iloc[idx])-av*0.1 if side=="buy" else _sf(df["swing_hi"].iloc[idx])+av*0.1)
    if slt=="Trail with Candle Low/High":
        return(_sf(rl["Low"])-av*0.1 if side=="buy" else _sf(rl["High"])+av*0.1)
    if slt=="Support/Resistance Based":
        n=min(20,len(df))
        return(float(df["Low"].tail(n).min())-av*0.1 if side=="buy" else float(df["High"].tail(n).max())+av*0.1)
    if slt=="Trailing SL": return entry+sign*pts
    return entry+sign*pts

def calc_targets(entry,side,df,cfg,idx,sl):
    tgt=cfg.get("target_type","Custom Points"); pts=float(cfg.get("target_points",20))
    sign=1 if side=="buy" else -1; sl_dist=abs(entry-sl)
    try: av=_sf(df["atr"].iloc[idx]); std=_sf(df["std"].iloc[idx]) if "std" in df.columns else av
    except: av=pts/2; std=av
    if tgt=="Custom Points": return(entry+sign*pts,entry+sign*pts*1.5,entry+sign*pts*2)
    if tgt=="ATR Based": m=float(cfg.get("atr_mult_tgt",2.5)); return(entry+sign*av*m,entry+sign*av*m*1.5,entry+sign*av*m*2)
    if tgt in("Auto Target","Volatility Based"):
        b=max(av*2.5,std*3); return(entry+sign*b,entry+sign*b*1.618,entry+sign*b*2.618)
    if tgt=="Risk-Reward Based":
        rr=float(cfg.get("rr_ratio",2)); return(entry+sign*sl_dist*rr,entry+sign*sl_dist*rr*1.618,entry+sign*sl_dist*rr*2.618)
    if tgt=="Book Profit: T1 then T2": return(entry+sign*pts,entry+sign*pts*2,entry+sign*pts*3)
    if tgt=="Support/Resistance Based":
        n=min(20,len(df)); t1=float(df["High"].tail(n).max() if side=="buy" else df["Low"].tail(n).min())
        return(t1,t1+sign*av,t1+sign*av*2)
    if tgt=="EMA Crossover Exit":
        es=_sf(df["ema_slow"].iloc[idx]); return(es+sign*av*0.5,entry+sign*av*2.5,entry+sign*av*4)
    return(entry+sign*pts,entry+sign*pts*1.5,entry+sign*pts*2)

def trail_sl(sl,ltp,side,df,cfg,idx):
    slt=cfg.get("sl_type","Custom Points"); pts=float(cfg.get("sl_points",10))
    try: av=_sf(df["atr"].iloc[idx])
    except: av=pts
    if slt=="Trailing SL":
        ns=ltp-pts if side=="buy" else ltp+pts
        return max(sl,ns) if side=="buy" else min(sl,ns)
    if slt=="Trail with Swing Low/High":
        try:
            ns=_sf(df["swing_lo"].iloc[idx])-av*0.1 if side=="buy" else _sf(df["swing_hi"].iloc[idx])+av*0.1
            return max(sl,ns) if side=="buy" else min(sl,ns)
        except: pass
    if slt=="Trail with Candle Low/High":
        try:
            ns=_sf(df["Low"].iloc[idx])-av*0.05 if side=="buy" else _sf(df["High"].iloc[idx])+av*0.05
            return max(sl,ns) if side=="buy" else min(sl,ns)
        except: pass
    return sl

def ema_rev(df,idx,side):
    if idx<1 or "ema_fast" not in df.columns: return False
    fn,fp=_sf(df["ema_fast"].iloc[idx]),_sf(df["ema_fast"].iloc[idx-1])
    sn,sp=_sf(df["ema_slow"].iloc[idx]),_sf(df["ema_slow"].iloc[idx-1])
    return(fn<sn and fp>=sp) if side=="buy" else(fn>sn and fp<=sp)

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df_raw,cfg):
    strat=cfg.get("strategy","EMA Crossover"); immediate=strat in("Simple Buy","Simple Sell")
    df=add_ind(df_raw,cfg); sigs=get_sigs(df,cfg)
    trades=[]; viols=[]; pos=None
    for i in range(len(df)):
        row=df.iloc[i]; o=_sf(row["Open"]); h=_sf(row["High"]); l=_sf(row["Low"]); c=_sf(row["Close"]); ts=df.index[i]
        if pos is not None:
            ep=pos["entry_price"]; side=pos["type"]
            sl=trail_sl(pos["sl"],c,side,df,cfg,i); pos["sl"]=sl; tp1=pos["tp1"]
            xp=None; xr=None; viol=False
            if side=="buy":
                sl_hit=l<=sl; tp_hit=h>=tp1
                if sl_hit and tp_hit: viol=True; xp=sl; xr="SL Hit (ambiguous)"
                elif sl_hit: xp=sl; xr="SL Hit"
                elif tp_hit: xp=tp1; xr="TP1 Hit"
            else:
                sl_hit=h>=sl; tp_hit=l<=tp1
                if sl_hit and tp_hit: viol=True; xp=sl; xr="SL Hit (ambiguous)"
                elif sl_hit: xp=sl; xr="SL Hit"
                elif tp_hit: xp=tp1; xr="TP1 Hit"
            if xr is None and cfg.get("target_type")=="EMA Crossover Exit":
                if ema_rev(df,i,side): xp=c; xr="EMA Reversal"
            if xp is not None:
                sign=1 if side=="buy" else -1
                pnl=round(sign*(xp-ep)*pos.get("qty",1),2)
                t={**pos,"exit_time":ts,"exit_price":round(xp,2),"exit_reason":xr,
                   "exit_high":h,"exit_low":l,"pnl":pnl,
                   "pnl_pct":round(sign*(xp-ep)/ep*100,2),"violated":viol}
                trades.append(t)
                if viol: viols.append(t)
                pos=None
        if pos is None:
            sig=int(sigs.iloc[i])
            if sig==0: continue
            if immediate: ep=c; ets=ts; ei=i
            else:
                if i+1>=len(df): continue
                ep=_sf(df["Open"].iloc[i+1]); ets=df.index[i+1]; ei=i+1
            side="buy" if sig==1 else "sell"
            sl=calc_sl(ep,side,df,cfg,ei); t1,t2,t3=calc_targets(ep,side,df,cfg,ei,sl)
            er=df.iloc[ei]
            pos={"entry_time":ets,"entry_price":round(ep,2),"type":side,
                 "sl":round(sl,2),"tp1":round(t1,2),"tp2":round(t2,2),"tp3":round(t3,2),
                 "entry_high":_sf(er["High"]),"entry_low":_sf(er["Low"]),
                 "signal_reason":strat,"qty":cfg.get("quantity",1)}
    if pos is not None:
        lr=df.iloc[-1]; c=_sf(lr["Close"]); side=pos["type"]; sign=1 if side=="buy" else -1
        pnl=round(sign*(c-pos["entry_price"])*pos.get("qty",1),2)
        trades.append({**pos,"exit_time":df.index[-1],"exit_price":round(c,2),
                       "exit_reason":"End of Data","exit_high":_sf(lr["High"]),"exit_low":_sf(lr["Low"]),
                       "pnl":pnl,"pnl_pct":round(sign*(c-pos["entry_price"])/pos["entry_price"]*100,2),"violated":False})
    wins=[t for t in trades if t["pnl"]>0]; losses=[t for t in trades if t["pnl"]<=0]; tot=len(trades)
    total_pnl=sum(t["pnl"] for t in trades)
    aw=sum(t["pnl"] for t in wins)/len(wins) if wins else 0
    al2=sum(t["pnl"] for t in losses)/len(losses) if losses else 0
    pf=abs(aw*len(wins)/(al2*len(losses))) if losses and al2!=0 else 0
    return{"trades":trades,"violations":viols,"df":df,
           "summary":{"total_trades":tot,"winners":len(wins),"losers":len(losses),
                      "total_pnl":round(total_pnl,2),"accuracy":round(len(wins)/tot*100,1) if tot else 0,
                      "avg_win":round(aw,2),"avg_loss":round(al2,2),"profit_factor":round(pf,2),
                      "max_win":round(max((t["pnl"] for t in trades),default=0),2),
                      "max_loss":round(min((t["pnl"] for t in trades),default=0),2),
                      "violation_count":len(viols)}}

def advisor(trades,summary,cfg):
    if not trades: return "⚠️ No trades generated."
    tot=summary["total_trades"]; acc=summary["accuracy"]; pf=summary["profit_factor"]
    pnl=summary["total_pnl"]; aw=summary["avg_win"]; al2=summary["avg_loss"]
    vc=summary["violation_count"]; sl_pts=float(cfg.get("sl_points",10))
    tgt_pts=float(cfg.get("target_points",20)); fast=cfg.get("ema_fast",9); slow=cfg.get("ema_slow",15)
    strat=cfg.get("strategy","")
    wins=[t for t in trades if t["pnl"]>0]; losses=[t for t in trades if t["pnl"]<=0]; lines=[]
    v="✅ **Strong**" if acc>=60 and pf>=1.5 else "🟡 **Moderate**" if acc>=50 else "🔴 **Weak**"
    lines.append(f"{v} — {tot} trades · {acc}% accuracy · PF {pf:.2f} · P&L **{pnl:+.2f}**")
    if wins and losses and al2!=0:
        rr=abs(aw/al2)
        if rr<1: lines.append(f"⚠️ Inverted R:R ({rr:.2f}×). Raise target to **{round(abs(al2)*1.8,1)} pts**.")
        elif rr<1.8: lines.append(f"🟡 R:R {rr:.2f}×. Aim ≥2×. Try target **{round(abs(al2)*2,1)} pts**.")
        else: lines.append(f"✅ Good R:R {rr:.2f}×.")
    sl_hits=[t for t in losses if "SL" in t.get("exit_reason","")]
    if sl_hits:
        pct=len(sl_hits)/tot*100
        buy_sl=[t for t in sl_hits if t.get("type")=="buy"]
        if buy_sl:
            gap=sum(abs(t.get("entry_price",0)-t.get("entry_low",t.get("entry_price",0))) for t in buy_sl)/len(buy_sl)
            if gap>sl_pts*1.1: lines.append(f"💡 SL too tight for BUY — avg candle gap {gap:.1f} pts. Raise to **{round(gap*1.15,1)} pts**.")
        lines.append(f"📉 {len(sl_hits)} SL hits ({pct:.0f}%).{'Consider ATR Based SL.' if pct>35 else ''}")
    if strat=="EMA Crossover" and acc<50:
        lines.append(f"📊 EMA({fast},{slow}) below 50%. Try EMA(9,21) or enable Min Crossover Angle filter.")
    elif strat=="Elliott Wave Auto": lines.append("🌊 Elliott Wave best on 1h/1d with ≥3 months data.")
    if losses and al2!=0:
        rec_sl=max(round(abs(al2)*0.8,1),1.0); rec_tgt=round(rec_sl*2,1)
        be=round(100/(1+abs(aw/al2)),0) if aw else 50
        lines.append(f"📌 **Recommendation:** SL={rec_sl} pts, Target={rec_tgt} pts. Need {be:.0f}% win rate to break even (current {acc}%).")
    if vc>0: lines.append(f"⚡ {vc} ambiguous bars (SL+target same candle, SL-first applied).")
    return "\n\n".join(lines)

def run_optimization(df,cfg,target_acc):
    results=[]
    for fast in range(5,21,2):
        for slow in range(10,41,5):
            if slow<=fast: continue
            for sl_pts in [5,10,15,20,30]:
                for tgt_pts in [10,20,30,40,60]:
                    c={**cfg,"ema_fast":fast,"ema_slow":slow,"sl_points":sl_pts,"target_points":tgt_pts}
                    try:
                        r=run_backtest(df,c); s=r["summary"]
                        if s["total_trades"]<2: continue
                        results.append({"ema_fast":fast,"ema_slow":slow,"sl_pts":sl_pts,"tgt_pts":tgt_pts,
                                        "trades":s["total_trades"],"accuracy":s["accuracy"],
                                        "total_pnl":s["total_pnl"],"pf":s["profit_factor"],
                                        "meets":s["accuracy"]>=target_acc})
                    except: pass
    results.sort(key=lambda x:(-x["accuracy"],-x["total_pnl"])); return results

# ── Dhan broker ───────────────────────────────────────────────────────────────
def dhan_client_init(cid,tok):
    try: from dhanhq import dhanhq; return dhanhq(cid,tok)
    except: return None

def dhan_place(client,sec_id,seg,txn,qty,order_type,price=0,product="INTRADAY"):
    if client is None: return{"success":False,"msg":"Dhan not initialised"}
    try:
        r=client.place_order(security_id=str(sec_id),exchange_segment=seg,transaction_type=txn,
                             quantity=qty,order_type=order_type,product_type=product,
                             price=price if order_type=="LIMIT" else 0,trigger_price=0,validity="DAY")
        return{"success":True,"response":r,"msg":"placed"}
    except Exception as e: return{"success":False,"msg":str(e)}

def dhan_equity(client,cfg,side,price):
    seg={"NSE":"NSE_EQ","BSE":"BSE_EQ"}.get(cfg.get("exchange","NSE"),"NSE_EQ")
    prod={"Intraday":"INTRADAY","Delivery":"CNC"}.get(cfg.get("product_type","Intraday"),"INTRADAY")
    ord_t={"Market Order":"MARKET","Limit Order":"LIMIT"}.get(cfg.get("entry_order_type","Market Order"),"MARKET")
    return dhan_place(client,cfg.get("security_id","1594"),seg,"BUY" if side=="buy" else "SELL",
                      int(cfg.get("quantity",1)),ord_t,price,prod)

def dhan_option(client,cfg,side,price):
    sec_id=cfg.get("ce_security_id") if side=="buy" else cfg.get("pe_security_id")
    ord_t={"Market Order":"MARKET","Limit Order":"LIMIT"}.get(cfg.get("options_entry_order_type","Market Order"),"MARKET")
    return dhan_place(client,str(sec_id or ""),cfg.get("fno_segment","NSE_FNO"),"BUY",
                      int(cfg.get("options_quantity",65)),ord_t,price)

def dhan_exit_order(client,cfg,pos,price):
    ord_t={"Market Order":"MARKET","Limit Order":"LIMIT"}.get(cfg.get("exit_order_type","Market Order"),"MARKET")
    if cfg.get("options_enabled"):
        return dhan_place(client,str(pos.get("option_sec_id","")),cfg.get("fno_segment","NSE_FNO"),
                          "SELL",int(cfg.get("options_quantity",65)),ord_t,price)
    seg={"NSE":"NSE_EQ","BSE":"BSE_EQ"}.get(cfg.get("exchange","NSE"),"NSE_EQ")
    prod={"Intraday":"INTRADAY","Delivery":"CNC"}.get(cfg.get("product_type","Intraday"),"INTRADAY")
    txn="SELL" if pos["type"]=="buy" else "BUY"
    return dhan_place(client,str(cfg.get("security_id","1594")),seg,txn,
                      int(cfg.get("quantity",1)),ord_t,price,prod)

# ── Live thread ───────────────────────────────────────────────────────────────
def _enter(ltp,side,df,cfg,dclient):
    idx=len(df)-1
    sl=calc_sl(ltp,side,df,cfg,idx); t1,t2,t3=calc_targets(ltp,side,df,cfg,idx,sl)
    pos={"entry_time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),"entry_price":round(ltp,2),
         "type":side,"sl":round(sl,2),"tp1":round(t1,2),"tp2":round(t2,2),"tp3":round(t3,2),
         "qty":cfg.get("quantity",1),"signal_reason":cfg.get("strategy","")}
    _set("position",pos)
    _log(f"ENTRY {side.upper()} @ {ltp:.2f} | SL:{sl:.2f} | TP1:{t1:.2f}","b" if side=="buy" else "s")
    if cfg.get("dhan_enabled") and dclient:
        try:
            r=dhan_option(dclient,cfg,side,ltp) if cfg.get("options_enabled") else dhan_equity(dclient,cfg,side,ltp)
            if r.get("success") and cfg.get("options_enabled"):
                pos["option_sec_id"]=cfg.get("ce_security_id") if side=="buy" else cfg.get("pe_security_id")
            _log(f"Dhan: {r.get('msg','placed')}","i")
        except Exception as ex: _log(f"Dhan error: {ex}","w")
    if cfg.get("cooldown_enabled",True):
        _set("cooldown_until",datetime.now(IST)+timedelta(seconds=max(int(cfg.get("cooldown_seconds",5)),1)))

def live_thread(stop_ev):
    cfg=_get("cfg",{}); symbol=cfg.get("symbol","^NSEI"); interval=cfg.get("interval","5m")
    period=cfg.get("period","1d"); strategy=cfg.get("strategy","EMA Crossover")
    is_simple=strategy in("Simple Buy","Simple Sell")
    dclient=None
    if cfg.get("dhan_enabled"):
        dclient=dhan_client_init(cfg.get("dhan_client_id",""),cfg.get("dhan_access_token",""))
        if dclient: _log("Dhan initialised","i")
    prev_ct=None; queued=None
    _log(f"▶ {symbol} | {interval} | {period} | {strategy}","i")
    tick=0
    while not stop_ev.is_set():
        tick+=1; _set("tick",tick)
        try:
            df=fetch_ohlcv(symbol,interval,period,warmup=True)
            if df is None or len(df)<3: _log("No data — retry","w"); time.sleep(1.5); continue
            df=add_ind(df,cfg)
            row=df.iloc[-1]; ltp=_sf(row["Close"]); ct=df.index[-1]
            if math.isnan(ltp): _log("NaN LTP — skip","w"); time.sleep(1.5); continue
            ef=_sf(row["ema_fast"]) if "ema_fast" in df.columns else 0.0
            es=_sf(row["ema_slow"]) if "ema_slow" in df.columns else 0.0
            av=_sf(row["atr"]) if "atr" in df.columns else 0.0
            _set("ltp",ltp); _set("ema_f",ef); _set("ema_s",es)
            vol=0
            try:
                vv=row["Volume"] if "Volume" in df.columns else 0
                vv=vv.iloc[0] if isinstance(vv,pd.Series) else vv
                vol=int(float(vv)) if not math.isnan(float(vv)) else 0
            except: vol=0
            _set("candle",{"time":str(ct)[:19],"open":round(_sf(row["Open"]),2),
                           "high":round(_sf(row["High"]),2),"low":round(_sf(row["Low"]),2),
                           "close":round(ltp,2),"ema_fast":round(ef,2),"ema_slow":round(es,2),
                           "atr":round(av,4),"volume":vol})
            if strategy=="Elliott Wave Auto":
                try: _set("waves",elliott_analyze(df))
                except: pass
            # SL/TP check
            pos=_get("position")
            if pos is not None:
                side=pos["type"]; sl=trail_sl(pos["sl"],ltp,side,df,cfg,len(df)-1)
                if abs(sl-pos["sl"])>0.001: pos["sl"]=round(sl,2); _set("position",pos)
                hit=None
                if side=="buy":
                    if ltp<=pos["sl"]: hit=("SL Hit",ltp)
                    elif ltp>=pos["tp1"]: hit=("TP1 Hit",ltp)
                else:
                    if ltp>=pos["sl"]: hit=("SL Hit",ltp)
                    elif ltp<=pos["tp1"]: hit=("TP1 Hit",ltp)
                if hit is None and cfg.get("target_type")=="EMA Crossover Exit":
                    if ema_rev(df,len(df)-1,side): hit=("EMA Reversal Exit",ltp)
                if hit:
                    reason,xp=hit; sign=1 if side=="buy" else -1
                    pnl=round(sign*(xp-pos["entry_price"])*pos.get("qty",1),2)
                    _append("trades",{**pos,"exit_time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                                      "exit_price":round(xp,2),"exit_reason":reason,
                                      "pnl":pnl,"pnl_pct":round(sign*(xp-pos["entry_price"])/pos["entry_price"]*100,2),"source":"live"})
                    _set("position",None); _log(f"EXIT {side.upper()} @ {xp:.2f} | {reason} | P&L:{pnl:+.2f}","e")
                    queued=None
                    if cfg.get("dhan_enabled") and dclient:
                        try: dhan_exit_order(dclient,cfg,pos,xp)
                        except: pass
            # Entry logic
            cur_pos=_get("position")
            if cur_pos is None and queued is None:
                cd=_get("cooldown_until"); in_cd=cd is not None and datetime.now(IST)<cd
                if not in_cd:
                    if is_simple:
                        side="buy" if strategy=="Simple Buy" else "sell"
                        _log(f"Simple {side.upper()} → enter @ {ltp:.2f}","i")
                        _enter(ltp,side,df,cfg,dclient)
                    else:
                        if ct!=prev_ct:
                            prev_ct=ct; sig=last_sig(df,cfg)
                            _log(f"Candle {str(ct)[:16]} EMA:{ef:.1f}/{es:.1f} sig={sig}","i")
                            if sig!=0:
                                queued={"side":"buy" if sig==1 else "sell","ct":ct}
                                _log(f"Signal {queued['side'].upper()} queued → next candle","i")
            # Execute queued entry at next candle
            if queued is not None and _get("position") is None:
                if ct!=queued["ct"]:
                    cd=_get("cooldown_until"); in_cd=cd is not None and datetime.now(IST)<cd
                    if not in_cd:
                        _log(f"Executing queued {queued['side'].upper()} @ {ltp:.2f}","i")
                        _enter(ltp,queued["side"],df,cfg,dclient)
                    else: _log("Queued entry skipped — cooldown","w")
                    queued=None
        except Exception as exc:
            err=f"{type(exc).__name__}: {exc}"; _log(f"Error: {err}","w"); _set("error",err)
        time.sleep(1.5)
    _set("running",False); _log("⏹ Stopped","w")

def start_live(cfg):
    with _LOCK:
        _TS["position"]=None; _TS["trades"]=[]; _TS["log"]=[]
        _TS["tick"]=0; _TS["ltp"]=None; _TS["candle"]=None
        _TS["error"]=None; _TS["waves"]={}; _TS["cooldown_until"]=None
        _TS["cfg"]=cfg; _TS["running"]=True
    ev=threading.Event(); _set("stop",ev)
    t=threading.Thread(target=live_thread,args=(ev,),daemon=True); _set("thread",t); t.start()
    _log(f"Thread started: {cfg.get('symbol')} / {cfg.get('strategy')}","i")

def stop_live():
    ev=_get("stop")
    if ev: ev.set()
    _set("running",False)

def squareoff():
    pos=_get("position")
    if pos is None: return
    ltp=_get("ltp") or pos["entry_price"]; sign=1 if pos["type"]=="buy" else -1
    pnl=round(sign*(ltp-pos["entry_price"])*pos.get("qty",1),2)
    _append("trades",{**pos,"exit_time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                      "exit_price":round(float(ltp),2),"exit_reason":"Manual Squareoff",
                      "pnl":pnl,"pnl_pct":round(sign*(ltp-pos["entry_price"])/pos["entry_price"]*100,2),"source":"live"})
    _set("position",None); _log(f"SQUAREOFF {pos['type'].upper()} @ {ltp:.2f} | P&L:{pnl:+.2f}","e")

# ── Charts ────────────────────────────────────────────────────────────────────
def make_chart(df,trades,cfg,title=""):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
                                 name="Price",increasing_line_color="#3fb950",decreasing_line_color="#f85149",
                                 increasing_fillcolor="#0d2818",decreasing_fillcolor="#2d1a1a",line=dict(width=1)),row=1,col=1)
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_fast"],name=f"EMA {cfg.get('ema_fast',9)}",line=dict(color="#f0a500",width=1.5)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_slow"],name=f"EMA {cfg.get('ema_slow',15)}",line=dict(color="#58a6ff",width=1.5)),row=1,col=1)
    if "Volume" in df.columns:
        clrs=["#0d2818" if c>=o else "#2d1a1a" for c,o in zip(df["Close"],df["Open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Vol",marker_color=clrs,opacity=0.5),row=2,col=1)
    for t in trades:
        try:
            et=pd.to_datetime(t["entry_time"]); xt=pd.to_datetime(t["exit_time"])
            ep=t["entry_price"]; xp=t["exit_price"]; tp=t["type"]; pnl=t["pnl"]
            ec="#3fb950" if tp=="buy" else "#f85149"; xc="#3fb950" if pnl>=0 else "#f85149"
            es="triangle-up" if tp=="buy" else "triangle-down"
            fig.add_trace(go.Scatter(x=[et],y=[ep],mode="markers+text",
                                     marker=dict(symbol=es,size=12,color=ec,line=dict(color="white",width=1)),
                                     text=[f"{'B' if tp=='buy' else 'S'} {ep:.0f}"],textposition="top center",
                                     textfont=dict(size=8,color=ec),showlegend=False),row=1,col=1)
            fig.add_trace(go.Scatter(x=[xt],y=[xp],mode="markers+text",
                                     marker=dict(symbol="square",size=9,color=xc,line=dict(color="white",width=1)),
                                     text=[f"X {xp:.0f}"],textposition="bottom center",
                                     textfont=dict(size=8,color=xc),showlegend=False),row=1,col=1)
            fig.add_shape(type="line",x0=et,x1=xt,y0=t["sl"],y1=t["sl"],line=dict(color="#f85149",width=1,dash="dot"),row=1,col=1)
            fig.add_shape(type="line",x0=et,x1=xt,y0=t["tp1"],y1=t["tp1"],line=dict(color="#3fb950",width=1,dash="dot"),row=1,col=1)
        except: pass
    fig.update_layout(title=dict(text=title,font=dict(color="#e6edf3",size=13)),
                      paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font=dict(color="#8b949e",size=11),
                      xaxis=dict(gridcolor="#21262d",rangeslider=dict(visible=False)),
                      xaxis2=dict(gridcolor="#21262d"),yaxis=dict(gridcolor="#21262d"),yaxis2=dict(gridcolor="#21262d"),
                      legend=dict(bgcolor="#161b22",bordercolor="#30363d",borderwidth=1,font=dict(size=10)),
                      margin=dict(l=0,r=0,t=36,b=0),height=500)
    return fig

# ── HTML helpers ──────────────────────────────────────────────────────────────
def mc(lbl,val,color="#e6edf3"):
    return f'<div class="mc"><div class="mc-lbl">{lbl}</div><div class="mc-val" style="color:{color}">{val}</div></div>'

def al_html(msg,kind="info"):
    k={"info":"al-info","warn":"al-warn","error":"al-err","ok":"al-ok"}.get(kind,"al-info")
    return f'<div class="al {k}">{msg}</div>'

def cb_html(items):
    rows="".join(f'<div class="cb-row"><span class="cb-k">{k}</span><span class="cb-v">{v}</span></div>' for k,v in items.items())
    return f'<div class="cb">{rows}</div>'

def pos_card(pos,ltp):
    side=pos.get("type","buy"); ep=pos.get("entry_price",0); sign=1 if side=="buy" else -1
    upnl=sign*(ltp-ep)*pos.get("qty",1); uc="#3fb950" if upnl>=0 else "#f85149"
    cls=" pos-sell" if side=="sell" else ""
    bdg=('<span class="bd bd-buy">BUY</span>' if side=="buy" else '<span class="bd bd-sell">SELL</span>')
    return f"""<div class="pos{cls}">
<div style="font-weight:700;color:#e6edf3;margin-bottom:6px">{bdg} Active — {pos.get('signal_reason','')}</div>
<div class="pos-grid">
<div><div class="pf-lbl">Entry</div><div class="pf-val">{ep:.2f}</div></div>
<div><div class="pf-lbl">LTP</div><div class="pf-val">{ltp:.2f}</div></div>
<div><div class="pf-lbl">Unrealized P&L</div><div class="pf-val" style="color:{uc}">{upnl:+.2f}</div></div>
<div><div class="pf-lbl">Stop Loss</div><div class="pf-val r">{pos.get('sl','—')}</div></div>
<div><div class="pf-lbl">TP1</div><div class="pf-val g">{pos.get('tp1','—')}</div></div>
<div><div class="pf-lbl">TP2</div><div class="pf-val g">{pos.get('tp2','—')}</div></div>
<div><div class="pf-lbl">Qty</div><div class="pf-val">{pos.get('qty',1)}</div></div>
<div><div class="pf-lbl">Entry Time</div><div class="pf-val" style="font-size:11px">{str(pos.get('entry_time',''))[:16]}</div></div>
</div></div>"""

def sg_html(items):
    cards=""
    for k,v in items.items():
        try: fv=float(str(v).replace("%","").replace("+","").replace(",","")); c=("#3fb950" if fv>=0 else "#f85149") if any(x in k for x in ["P&L","Profit","Loss"]) else "#e6edf3"
        except: c="#e6edf3"
        cards+=f'<div class="sc"><div class="sc-lbl">{k}</div><div class="sc-val" style="color:{c}">{v}</div></div>'
    return f'<div class="sg">{cards}</div>'

def trade_table(trades,show_viols=True):
    if not trades: return al_html("No trades yet.","info")
    vc=sum(1 for t in trades if t.get("violated"))
    note=al_html(f"⚠️ {vc} ambiguous candles (SL+target same bar, SL-first applied). Live trading may differ.","warn") if show_viols and vc else ""
    hdrs=["#","Type","Entry Time","Entry","SL","TP1","Exit Time","Exit","Reason","High","Low","P&L","P&L%","Qty"]
    th="".join(f"<th>{h}</th>" for h in hdrs); rows=""
    for i,t in enumerate(trades,1):
        pnl=t.get("pnl",0); pc="ppos" if pnl>=0 else "pneg"; vc2="vrow" if t.get("violated") else ""
        bdg='<span class="bd bd-buy">BUY</span>' if t.get("type")=="buy" else '<span class="bd bd-sell">SELL</span>'
        rows+=f'<tr class="{vc2}"><td>{i}</td><td>{bdg}</td><td style="font-size:10px">{str(t.get("entry_time",""))[:16]}</td><td>{t.get("entry_price","")}</td><td style="color:#f85149">{t.get("sl","")}</td><td style="color:#3fb950">{t.get("tp1","")}</td><td style="font-size:10px">{str(t.get("exit_time",""))[:16]}</td><td>{t.get("exit_price","")}</td><td style="font-size:10px">{t.get("exit_reason","")}</td><td>{t.get("exit_high","")}</td><td>{t.get("exit_low","")}</td><td class="{pc}">{pnl:+.2f}</td><td class="{pc}">{t.get("pnl_pct",0):+.2f}%</td><td>{t.get("qty",t.get("quantity",1))}</td></tr>'
    return note+f'<div class="tw"><table class="tt"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>'

def log_html(entries):
    cm={"i":"li","b":"lb","s":"ls","w":"lw","e":"le"}
    lines="".join(f'<div class="{cm.get(e["l"],"li")}">[{e["t"]}] {e["m"]}</div>' for e in reversed(entries[-80:]))
    return f'<div class="log">{lines}</div>'

def wave_cards(res):
    waves={w["wave"] for w in res.get("waves",[])}; curr=res.get("current_wave","")
    labels=["1","2","3","4","5","A","B","C"]
    colors={"1":"#58a6ff","2":"#e3b341","3":"#3fb950","4":"#e3b341","5":"#58a6ff","A":"#f85149","B":"#3fb950","C":"#f85149"}
    cards=""
    for w in labels:
        st2="Complete" if w in waves else("Active" if w==curr else "Pending")
        bdr="#238636" if st2=="Complete" else("#e3b341" if st2=="Active" else "#30363d")
        cards+=f'<div style="background:#161b22;border:1px solid {bdr};border-radius:8px;padding:8px;text-align:center;min-width:60px"><div style="font-size:18px;font-weight:700;color:{colors.get(w,"#8b949e")}">W{w}</div><div style="color:#8b949e;font-size:10px">{st2}</div></div>'
    return f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0">{cards}</div>'

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Smart Investing")
        st.markdown("---")
        st.markdown("### 🔭 Instrument")
        tn=st.selectbox("Ticker",list(TICKERS.keys()),key="sb_tn")
        symbol=TICKERS.get(tn,"")
        if tn=="Custom": symbol=st.text_input("Symbol (e.g. RELIANCE.NS)",key="sb_cu").strip().upper()
        iv=st.selectbox("Timeframe",list(TF_PERIODS.keys()),index=1,key="sb_iv")
        pd2=st.selectbox("Period",TF_PERIODS[iv],key="sb_pd")
        st.markdown("---"); st.markdown("### 🎯 Strategy")
        strat=st.selectbox("Strategy",STRATEGIES,key="sb_st")
        ef=9; es=15; co="Simple Crossover"; cs=10.0; acm=0.5; ma=0.0
        if strat in("EMA Crossover","Anticipatory EMA Crossover"):
            c1,c2=st.columns(2)
            with c1: ef=st.number_input("Fast EMA",2,50,9,key="sb_ef")
            with c2: es=st.number_input("Slow EMA",3,100,15,key="sb_es")
            co=st.selectbox("Crossover Type",CO_TYPES,key="sb_co")
            if co=="Custom Candle Size": cs=st.number_input("Min Candle Size",0.1,value=10.0,key="sb_cs")
            elif co=="ATR Based Candle Size": acm=st.number_input("ATR Candle Mult",0.1,value=0.5,key="sb_acm")
            if st.checkbox("Min Angle Filter",key="sb_af"): ma=st.number_input("Min Angle°",0.0,90.0,0.0,key="sb_ma")
        qty=st.number_input("Quantity",1,value=1,key="sb_q")
        st.markdown("---"); st.markdown("### 🛡️ Stop Loss")
        slt=st.selectbox("SL Type",SL_TYPES,key="sb_slt")
        slp=10.0; rr=2.0; aslm=1.5; vslm=2.0
        if slt=="Custom Points": slp=st.number_input("SL Points",0.1,value=10.0,key="sb_slp")
        elif slt=="ATR Based": aslm=st.number_input("ATR Mult",0.1,value=1.5,key="sb_asm")
        elif slt=="Risk-Reward Based": rr=st.number_input("R:R",0.1,value=2.0,key="sb_rr")
        elif slt=="Volatility Based": vslm=st.number_input("Vol Mult",0.1,value=2.0,key="sb_vsm")
        st.markdown("---"); st.markdown("### 🎯 Target")
        tgt=st.selectbox("Target Type",TARGET_TYPES,key="sb_tgt")
        tgtp=20.0; atm=2.5; t1p=50.0
        if tgt=="Custom Points": tgtp=st.number_input("Target Points",0.1,value=20.0,key="sb_tp")
        elif tgt=="ATR Based": atm=st.number_input("ATR Mult",0.1,value=2.5,key="sb_atm")
        elif tgt=="Risk-Reward Based": rr=st.number_input("R:R",0.1,value=2.0,key="sb_rr2")
        elif tgt=="Book Profit: T1 then T2": t1p=st.number_input("Book% at T1",1.0,100.0,50.0,key="sb_t1p")
        st.markdown("---"); st.markdown("### ⚡ Live Settings")
        cden=st.checkbox("Cooldown Between Trades",True,key="sb_cd")
        cdsec=st.number_input("Cooldown (s)",1,value=5,key="sb_cds") if cden else 5
        noov=st.checkbox("Prevent Overlapping Trades",True,key="sb_ov")
        st.markdown("---"); st.markdown("### 🏦 Dhan Broker")
        dhen=st.checkbox("Enable Dhan",False,key="sb_dh")
        opten=False; exch="NSE"; prod="Intraday"; secid="1594"; eord="Market Order"; xord="Market Order"
        fnoseg="NSE_FNO"; ceid=""; peid=""; oqty=65; oent="Market Order"; oex="Market Order"; cid=""; tok=""
        if dhen:
            cid=st.text_input("Client ID",key="sb_cid",type="password"); tok=st.text_input("Access Token",key="sb_tok",type="password")
            opten=st.checkbox("Options Trading",False,key="sb_opts")
            if not opten:
                exch=st.selectbox("Exchange",["NSE","BSE"],key="sb_ex"); prod=st.selectbox("Product",["Intraday","Delivery"],key="sb_pr")
                secid=st.text_input("Security ID","1594",key="sb_si"); eord=st.selectbox("Entry Order",ORDER_TYPES,key="sb_eo"); xord=st.selectbox("Exit Order",ORDER_TYPES,key="sb_xo")
            else:
                fnoseg=st.selectbox("FNO Segment",FNO_SEGS,key="sb_fs"); ceid=st.text_input("CE Security ID",key="sb_ce"); peid=st.text_input("PE Security ID",key="sb_pe")
                oqty=st.number_input("Options Qty",1,value=65,key="sb_oq"); oent=st.selectbox("Entry Order",ORDER_TYPES,key="sb_oe"); oex=st.selectbox("Exit Order",ORDER_TYPES,key="sb_ox")
        st.markdown("---"); st.caption("Smart Investing v3.0 · yfinance + Dhan")
    return dict(ticker_name=tn,symbol=symbol,interval=iv,period=pd2,strategy=strat,quantity=qty,
                ema_fast=ef,ema_slow=es,crossover_type=co,custom_candle_size=cs,atr_candle_mult=acm,
                min_crossover_angle=ma,sl_type=slt,sl_points=slp,rr_ratio=rr,atr_mult_sl=aslm,
                vol_mult_sl=vslm,target_type=tgt,target_points=tgtp,atr_mult_tgt=atm,t1_book_pct=t1p,
                cooldown_enabled=cden,cooldown_seconds=cdsec,no_overlap=noov,
                dhan_enabled=dhen,options_enabled=opten,exchange=exch,product_type=prod,
                security_id=secid,entry_order_type=eord,exit_order_type=xord,fno_segment=fnoseg,
                ce_security_id=ceid,pe_security_id=peid,options_quantity=oqty,
                options_entry_order_type=oent,options_exit_order_type=oex,
                dhan_client_id=cid,dhan_access_token=tok,atr_period=14)

# ── Tab: Backtest ─────────────────────────────────────────────────────────────
def tab_backtest(cfg):
    c1,c2=st.columns([4,1])
    with c1: st.markdown(f"**{cfg['ticker_name']}** · `{cfg['interval']}` · `{cfg['period']}` · **{cfg['strategy']}**")
    with c2: run=st.button("▶ Run Backtest",key="bt_run",use_container_width=True,type="primary")
    if run:
        with st.spinner("Downloading & backtesting…"):
            df=fetch_ohlcv(cfg["symbol"],cfg["interval"],cfg["period"],warmup=True)
            if df is None: st.error("❌ Data fetch failed."); return
            st.session_state["bt"]=run_backtest(df,cfg)
    bt=st.session_state.get("bt")
    if bt is None: st.markdown(al_html("Click ▶ Run Backtest to start.","info"),unsafe_allow_html=True); return
    trades=bt["trades"]; s=bt["summary"]; df=bt["df"]
    st.markdown(sg_html({"Total Trades":s["total_trades"],"Winners":s["winners"],"Losers":s["losers"],
                          "Accuracy":f"{s['accuracy']}%","Total P&L":s["total_pnl"],"Avg Win":s["avg_win"],
                          "Avg Loss":s["avg_loss"],"Profit Factor":s["profit_factor"],
                          "Max Win":s["max_win"],"Max Loss":s["max_loss"],"Ambiguous":s["violation_count"]}),unsafe_allow_html=True)
    st.markdown("---"); st.markdown("#### 🧠 Strategy Advisor")
    for para in advisor(trades,s,cfg).split("\n\n"):
        para=para.strip()
        if not para: continue
        kind="ok" if para.startswith("✅") else "warn" if para.startswith(("🟡","⚠️","💡","📉","⚡")) else "err" if para.startswith("🔴") else "info"
        st.markdown(al_html(para,kind),unsafe_allow_html=True)
    st.markdown("---")
    if df is not None and len(df)>0:
        st.plotly_chart(make_chart(df,trades,cfg,f"{cfg['ticker_name']} — {cfg['strategy']}"),use_container_width=True,config={"displayModeBar":False})
    st.markdown("---"); st.markdown("#### 📋 Trade Log")
    st.markdown(trade_table(trades),unsafe_allow_html=True)

# ── Live dashboard fragment (MODULE LEVEL — stable Streamlit identity) ────────
@st.fragment(run_every=1.5)
def live_dashboard():
    running=_get("running",False); ltp=_get("ltp"); ef=_get("ema_f"); es=_get("ema_s")
    candle=_get("candle"); pos=_get("position"); tick=_get("tick",0); err=_get("error")
    trades=_get("trades",[]); cfg_live=_get("cfg") or {}
    sn=cfg_live.get("ticker_name",cfg_live.get("symbol","—")); strat=cfg_live.get("strategy","—")
    fast_n=cfg_live.get("ema_fast",9); slow_n=cfg_live.get("ema_slow",15)
    if running: st.markdown(al_html(f"● RUNNING — {sn} · {strat} · Tick #{tick}","ok"),unsafe_allow_html=True)
    else: st.markdown(al_html("■ STOPPED — Click ▶ Start above","err"),unsafe_allow_html=True)
    ltp_s=f"{ltp:.2f}" if ltp is not None else "—"
    efs=f"{ef:.2f}" if ef is not None else "—"; ess=f"{es:.2f}" if es is not None else "—"
    upnl=None
    if pos and ltp is not None: sign=1 if pos["type"]=="buy" else -1; upnl=sign*(ltp-pos["entry_price"])*pos.get("qty",1)
    spnl=sum(t["pnl"] for t in trades) if trades else 0.0
    sw=sum(1 for t in trades if t["pnl"]>0); sa=round(sw/len(trades)*100,1) if trades else 0.0
    ups=f"{upnl:+.2f}" if upnl is not None else "—"; uc="#3fb950" if(upnl or 0)>=0 else "#f85149"
    sc="#3fb950" if spnl>=0 else "#f85149"
    cols=st.columns(6)
    for col,(lbl,val,clr) in zip(cols,[("LTP",ltp_s,"#e6edf3"),(f"EMA {fast_n}",efs,"#f0a500"),
                                        (f"EMA {slow_n}",ess,"#58a6ff"),("Unrealized P&L",ups,uc),
                                        ("Session P&L",f"{spnl:+.2f}",sc),(f"Trades·WR",f"{len(trades)}·{sa}%","#e6edf3")]):
        with col: st.markdown(mc(lbl,val,clr),unsafe_allow_html=True)
    st.markdown("---")
    if candle:
        cv=candle
        st.markdown(f"""<div class="cr">
<div class="cf"><div class="cf-lbl">Time</div><div class="cf-val" style="font-size:11px">{cv.get('time','—')}</div></div>
<div class="cf"><div class="cf-lbl">Open</div><div class="cf-val">{cv.get('open','—')}</div></div>
<div class="cf"><div class="cf-lbl">High</div><div class="cf-val" style="color:#3fb950">{cv.get('high','—')}</div></div>
<div class="cf"><div class="cf-lbl">Low</div><div class="cf-val" style="color:#f85149">{cv.get('low','—')}</div></div>
<div class="cf"><div class="cf-lbl">Close</div><div class="cf-val">{cv.get('close','—')}</div></div>
<div class="cf"><div class="cf-lbl">EMA {fast_n}</div><div class="cf-val" style="color:#f0a500">{cv.get('ema_fast','—')}</div></div>
<div class="cf"><div class="cf-lbl">EMA {slow_n}</div><div class="cf-val" style="color:#58a6ff">{cv.get('ema_slow','—')}</div></div>
<div class="cf"><div class="cf-lbl">ATR</div><div class="cf-val">{cv.get('atr','—')}</div></div>
<div class="cf"><div class="cf-lbl">Volume</div><div class="cf-val" style="font-size:11px">{cv.get('volume',0):,}</div></div>
</div>""",unsafe_allow_html=True)
    if pos and ltp is not None: st.markdown("##### 💼 Active Position"); st.markdown(pos_card(pos,ltp),unsafe_allow_html=True)
    elif running and pos is None: st.markdown(al_html("No active position — waiting for signal.","info"),unsafe_allow_html=True)
    if trades:
        st.markdown(f"""<div class="cr" style="margin-top:6px">
<div class="cf"><div class="cf-lbl">Trades</div><div class="cf-val">{len(trades)}</div></div>
<div class="cf"><div class="cf-lbl">Winners</div><div class="cf-val" style="color:#3fb950">{sw}</div></div>
<div class="cf"><div class="cf-lbl">Win Rate</div><div class="cf-val">{sa}%</div></div>
<div class="cf"><div class="cf-lbl">Session P&L</div><div class="cf-val" style="color:{sc};font-weight:700">{spnl:+.2f}</div></div>
</div>""",unsafe_allow_html=True)
    st.markdown("---")
    waves=_get("waves",{})
    if waves and waves.get("waves"):
        st.markdown("##### 🌊 Elliott Wave"); wc,wi=st.columns([2,1])
        with wc: st.markdown(wave_cards(waves),unsafe_allow_html=True)
        with wi:
            st.markdown(cb_html({"Status":waves.get("status","—"),"Curr Wave":waves.get("current_wave","—"),
                                  "Direction":waves.get("direction","—"),"Signal":waves.get("signal","—"),
                                  "Wave SL":f"{waves['sl']:.2f}" if waves.get("sl") else "—",
                                  "TP1":f"{waves['tp1']:.2f}" if waves.get("tp1") else "—"}),unsafe_allow_html=True)
    st.markdown("##### 📝 Live Log")
    st.markdown(log_html(_get("log",[])),unsafe_allow_html=True)
    if err: st.markdown(al_html(f"⚠️ {err}","error"),unsafe_allow_html=True)

# ── Tab: Live ─────────────────────────────────────────────────────────────────
def tab_live(cfg):
    running=_get("running",False)
    b1,b2,b3=st.columns(3)
    with b1:
        if st.button("▶ Start",key="lv_start",disabled=running,use_container_width=True,type="primary"):
            start_live(cfg); st.rerun()
    with b2:
        if st.button("⏹ Stop",key="lv_stop",disabled=not running,use_container_width=True):
            stop_live(); st.rerun()
    with b3:
        if st.button("⚡ Squareoff",key="lv_sq",use_container_width=True): squareoff()
    if _get("running",False):
        with st.expander("⚙️ Active Configuration",expanded=False):
            a,b_=st.columns(2)
            with a: st.markdown(cb_html({"Symbol":cfg.get("symbol","—"),"Interval":cfg.get("interval","—"),"Period":cfg.get("period","—"),"Strategy":cfg.get("strategy","—"),"Quantity":cfg.get("quantity",1)}),unsafe_allow_html=True)
            with b_: st.markdown(cb_html({f"EMA Fast":cfg.get("ema_fast",9),f"EMA Slow":cfg.get("ema_slow",15),"SL Type":cfg.get("sl_type","—"),"SL Points":cfg.get("sl_points",10),"Target":cfg.get("target_type","—"),"Tgt Points":cfg.get("target_points",20),"Cooldown":f"{cfg.get('cooldown_seconds',5)}s" if cfg.get("cooldown_enabled") else "Off","Dhan":"On" if cfg.get("dhan_enabled") else "Off"}),unsafe_allow_html=True)
    st.markdown("---")
    live_dashboard()

# ── Tab: History ──────────────────────────────────────────────────────────────
def tab_history(cfg):
    st.markdown("### 📚 Trade History")
    st.markdown(al_html("Updates in real-time even while live trading is running.","info"),unsafe_allow_html=True)
    live_t=_get("trades",[]); bt_t=st.session_state.get("bt",{}).get("trades",[]) if st.session_state.get("bt") else []
    t1,t2=st.tabs(["Live Trades","Backtest Trades"])
    with t1:
        if live_t:
            tot=len(live_t); w=sum(1 for t in live_t if t["pnl"]>0); sp=sum(t["pnl"] for t in live_t)
            st.markdown(sg_html({"Trades":tot,"Winners":w,"Losers":tot-w,"Accuracy":f"{round(w/tot*100,1)}%","Session P&L":round(sp,2)}),unsafe_allow_html=True)
            st.markdown(trade_table(live_t,show_viols=False),unsafe_allow_html=True)
        else: st.markdown(al_html("No live trades yet.","info"),unsafe_allow_html=True)
    with t2:
        if bt_t: st.markdown(trade_table(bt_t),unsafe_allow_html=True)
        else: st.markdown(al_html("Run a backtest first.","info"),unsafe_allow_html=True)

# ── Tab: Optimization ─────────────────────────────────────────────────────────
def tab_optimization(cfg):
    st.markdown("### ⚙️ Strategy Optimization")
    st.markdown(al_html("Grid search over EMA / SL / Target. Shows all results even if target accuracy not met.","info"),unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,1,2])
    with c1: ta=st.number_input("Target Accuracy (%)",0.0,100.0,50.0,key="opt_a")
    with c2: mr=st.number_input("Max rows",5,200,20,key="opt_mr")
    with c3:
        if st.button("🔍 Run Optimization",key="opt_run",type="primary",use_container_width=True):
            with st.spinner("Running grid search…"):
                df=fetch_ohlcv(cfg["symbol"],cfg["interval"],cfg["period"],warmup=True)
                if df is None: st.error("Data fetch failed."); return
                df=add_ind(df,cfg); st.session_state["opt"]=run_optimization(df,cfg,float(ta))
    res=st.session_state.get("opt")
    if res is None: return
    meets=sum(1 for r in res if r["meets"])
    st.markdown(f"**{len(res)} combos · {meets} meet {ta}% target**")
    rows=""
    for i,r in enumerate(res[:int(mr)],1):
        ac="#3fb950" if r["meets"] else "#e3b341"; pc="#3fb950" if r["total_pnl"]>=0 else "#f85149"
        rows+=f"<tr><td>{i}</td><td>{r['ema_fast']}</td><td>{r['ema_slow']}</td><td>{r['sl_pts']}</td><td>{r['tgt_pts']}</td><td>{r['trades']}</td><td style='color:{ac};font-weight:700'>{r['accuracy']}%</td><td style='color:{pc};font-weight:700'>{r['total_pnl']:.2f}</td><td>{r['pf']:.2f}</td></tr>"
    st.markdown(f'<div class="tw"><table class="tt"><thead><tr><th>#</th><th>Fast</th><th>Slow</th><th>SL</th><th>Tgt</th><th>Trades</th><th>Accuracy</th><th>P&L</th><th>PF</th></tr></thead><tbody>{rows}</tbody></table></div>',unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg=render_sidebar()
    st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px"><span style="font-size:26px">📈</span><div><div style="font-size:20px;font-weight:700;color:#e6edf3">Smart Investing</div><div style="color:#8b949e;font-size:11px">Algorithmic Trading · NSE · BSE · Crypto · Elliott Wave · Dhan</div></div></div>',unsafe_allow_html=True)
    tabs=st.tabs(["📊 Backtesting","⚡ Live Trading","📚 Trade History","⚙️ Optimization"])
    with tabs[0]: tab_backtest(cfg)
    with tabs[1]: tab_live(cfg)
    with tabs[2]: tab_history(cfg)
    with tabs[3]: tab_optimization(cfg)

if __name__=="__main__":
    main()
