# ╔══════════════════════════════════════════════════════════════╗
# ║        SMART INVESTING  –  Algo Trading Platform v4         ║
# ╚══════════════════════════════════════════════════════════════╝
# Root-cause fixes v4:
#  1. EMA spread: removed nested f-string quotes (caused silent crash)
#  2. Auto-refresh: time-based rerun WITHOUT blocking time.sleep()
#  3. Simple Buy/Sell: fires on very first tick; UI refreshes within 2s
#  4. Optimization: includes Elliott Wave; no None rows in results table
#  5. EW / EMA panels guarded – only render when data is actually available

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
.mc{background:linear-gradient(135deg,#1a1f2e,#252b3d);border:1px solid #2d3748;border-radius:10px;padding:14px;margin:5px 0}
.ec{background:#1a2535;border:1px solid #2d4a6e;border-radius:8px;padding:10px 14px;text-align:center;margin:2px}
.cfg{background:#1a1f2e;border-radius:8px;padding:12px;border-left:3px solid #4299e1;margin:8px 0;font-size:13px;line-height:1.8;color:#cbd5e0}
.ewr{background:#162032;border:1px solid #2d5a8e;border-radius:10px;padding:14px;margin:8px 0}
.wl{background:#2d3748;border:1px solid #4a5568;border-radius:6px;padding:5px 10px;display:inline-block;margin:3px;font-size:12px}
.lb{background:#0d1117;border-radius:8px;padding:10px;height:200px;overflow-y:auto;
  font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;color:#a0aec0}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent}
.stTabs [data-baseweb="tab"]{background:#1a1f2e;border-radius:8px;padding:8px 20px;border:1px solid #2d3748;color:#a0aec0}
.stTabs [aria-selected="true"]{background:#2b6cb0!important;color:white!important}
section[data-testid="stSidebar"]{background:#0d1117}
section[data-testid="stSidebar"] label{color:#a0aec0!important;font-size:12px}
</style>""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
TICKERS={"Nifty 50":"^NSEI","BankNifty":"^NSEBANK","Sensex":"^BSESN",
          "BTC":"BTC-USD","ETH":"ETH-USD","Gold":"GC=F","Silver":"SI=F","Custom":None}
TF_PERIODS={"1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo"],"15m":["1d","5d","7d","1mo"],
            "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
            "1d":["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
            "1wk":["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"]}
MAX_PERIOD={"1m":"7d","5m":"60d","15m":"60d","1h":"730d","1d":"max","1wk":"max"}
TF_MINUTES={"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440,"1wk":10080}

# ── Thread-safe store ─────────────────────────────────────────────
_TS_LOCK=threading.Lock()
_TS={"live_running":False,"live_status":"STOPPED","live_position":None,"live_trades":[],
     "live_data":None,"live_ltp":None,"live_prev_close":None,
     "live_log":[],"live_last_ts":None,"live_ew":{},"_rl_ts":0.0}

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

for _k,_v in {"live_thread":None,"bt_trades":[],"bt_viol":[],"bt_df":None,"bt_ran":False,
              "dhan_client":None,"opt_results":[],"opt_ran":False,"_live_tick":0.0}.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ── Data fetch ────────────────────────────────────────────────────
def _rate_wait():
    with _TS_LOCK:
        gap=time.time()-_TS["_rl_ts"]
        if gap<1.5: time.sleep(1.5-gap)
        _TS["_rl_ts"]=time.time()

def _clean(df):
    if df is None or df.empty: return None
    df=df.copy(); df.index=pd.to_datetime(df.index)
    df.index=df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize("UTC").tz_convert(IST)
    df.columns=[c.lower() for c in df.columns]
    keep=[c for c in ["open","high","low","close","volume"] if c in df.columns]
    df=df[keep].dropna(subset=["close"])
    return df[~df.index.duplicated(keep="last")].sort_index()

def fetch_data(symbol,interval,period,quiet=False):
    _rate_wait()
    try:
        raw=yf.Ticker(symbol).history(period=MAX_PERIOD.get(interval,period),
                                      interval=interval,auto_adjust=True,prepost=False)
        df=_clean(raw)
        if df is None or df.empty:
            raw=yf.Ticker(symbol).history(period=period,interval=interval,
                                          auto_adjust=True,prepost=False)
            df=_clean(raw)
        return df
    except Exception as e:
        if not quiet: _log(f"[ERR] fetch: {e}")
        return None

def fetch_ltp(symbol):
    try:
        _rate_wait()
        raw=yf.Ticker(symbol).history(period="5d",interval="1d",auto_adjust=True,prepost=False)
        df=_clean(raw)
        if df is not None and len(df)>=2: return float(df["close"].iloc[-1]),float(df["close"].iloc[-2])
        if df is not None and len(df)==1: v=float(df["close"].iloc[-1]); return v,v
    except Exception: pass
    return None,None

# ── Indicators (TradingView-accurate) ─────────────────────────────
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

def add_ind(df,fast,slow):
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

# ── Elliott Wave ──────────────────────────────────────────────────
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
    EMPTY={"ok":False,"status":"Insufficient data","direction":"UNCLEAR","waves":[],
           "current_wave":"—","wave_detail":{},"next_targets":{},"after_msg":"Need more candles",
           "swing_points":[],"current_price":cp,"recommendation":None,"trade_bias":""}
    if len(df)<25: return EMPTY
    swings=find_swings(df,win=max(3,len(df)//40))
    if len(swings)<4: return {**EMPTY,"ok":True,"status":"Forming","current_wave":"Accumulating","swing_points":swings}
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
                projs={f"W5@{int(k*100)}%":round(p[4]+sign*w1*k,2) for k in [0.618,1.0,1.618]}
                sl_s=round(p[4]-sign*abs(w1)*0.05,2); tgt_s=round(p[4]+sign*w1,2)
                rec={"signal":"BUY" if sign==1 else "SELL","entry":cp,
                     "wave_context":f"Wave 5 of {label.title()} Impulse",
                     "sl_suggestion":sl_s,"tgt_suggestion":tgt_s,
                     "sl_basis":f"Below W4 end ({p[4]:.2f})","tgt_basis":"W5 = 100% of W1"}
                return {"ok":True,"status":f"5-Wave Impulse ({'Bullish' if sign==1 else 'Bearish'})",
                        "direction":label,"waves":waves,"current_wave":"Wave 5 (forming)",
                        "wave_detail":{"W1":f"{p[0]:.2f}→{p[1]:.2f}",
                            "W2":f"{p[1]:.2f}→{p[2]:.2f} ({abs(w2r)*100:.1f}%ret)",
                            "W3":f"{p[2]:.2f}→{p[3]:.2f} ({w3/w1*100:.0f}%W1)",
                            "W4":f"{p[3]:.2f}→{p[4]:.2f}","W5":f"{p[4]:.2f}→{p[5]:.2f}(live)"},
                        "next_targets":projs,"after_msg":"Expect ABC after W5","swing_points":swings,
                        "current_price":cp,"trade_bias":"BUY" if sign==1 else "SELL","recommendation":rec}
    # ABC
    if len(swings)>=4:
        for start,label,sign in [("H","CORRECTIVE_DOWN",-1),("L","CORRECTIVE_UP",1)]:
            s=swings[-4:]
            if s[0]["type"]!=start: continue
            p=[pt["price"] for pt in s]; amv=sign*(p[0]-p[1])
            br=(p[2]-p[1])/(p[1]-p[0]) if p[1]-p[0] else 0; c_tgt=round(p[2]-sign*amv,2)
            sl_s=round(p[2]+sign*abs(amv)*0.1,2)
            rec={"signal":"SELL" if sign==-1 else "BUY","entry":cp,
                 "wave_context":f"Wave C – {'Bearish' if sign==-1 else 'Bullish'} ABC",
                 "sl_suggestion":sl_s,"tgt_suggestion":c_tgt,
                 "sl_basis":"Above Wave B end","tgt_basis":"C = A equal move"}
            return {"ok":True,"status":f"ABC Correction ({'Bearish' if sign==-1 else 'Bullish'})",
                    "direction":label,"waves":[{"label":lb,"start":s[i],"end":s[i+1]} for i,lb in enumerate(["A","B","C"])],
                    "current_wave":"Wave C (forming)","wave_detail":{"A":f"{p[0]:.2f}→{p[1]:.2f}",
                        "B":f"{p[1]:.2f}→{p[2]:.2f} ({abs(br)*100:.1f}%ret)","C":f"{p[2]:.2f}→{p[3]:.2f}(live)"},
                    "next_targets":{"C@61.8%A":round(p[2]-sign*amv*0.618,2),"C=A":c_tgt},
                    "after_msg":"Expect new impulse after ABC","swing_points":swings,
                    "current_price":cp,"trade_bias":"SELL" if sign==-1 else "BUY","recommendation":rec}
    return {**EMPTY,"ok":True,"status":"Structure Forming","swing_points":swings}

# ── SL/Target ─────────────────────────────────────────────────────
def calc_sl_tgt(entry,ttype,row,cfg):
    atr=float(row.get("atr",entry*0.01) or entry*0.01)
    if np.isnan(atr) or atr==0: atr=entry*0.01
    sl_t=cfg.get("sl_type","Custom Points"); tg_t=cfg.get("target_type","Custom Points")
    slp=float(cfg.get("sl_points",10)); tgp=float(cfg.get("target_points",20)); sign=1 if ttype=="buy" else -1
    sl=entry-sign*atr*float(cfg.get("atr_sl_mult",1.5)) if sl_t=="ATR Based" else entry-sign*slp
    sld=abs(entry-sl)
    if tg_t=="ATR Based Target": tgt=entry+sign*atr*float(cfg.get("atr_tgt_mult",3.0))
    elif tg_t=="Risk-Reward Based": tgt=entry+sign*sld*float(cfg.get("rr_ratio",2.0))
    else: tgt=entry+sign*tgp
    return round(sl,4),round(tgt,4)

# ── Signals ────────────────────────────────────────────────────────
def ema_sig(df,idx,cfg):
    if idx<1 or idx>=len(df): return None,None
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es=row.get("ema_fast",np.nan),row.get("ema_slow",np.nan)
    pf,ps=prev.get("ema_fast",np.nan),prev.get("ema_slow",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return None,None
    if cfg.get("use_angle_filter",False):
        if ema_angle(df["ema_fast"].iloc[:idx+1])<float(cfg.get("min_ema_angle",0)): return None,None
    ct=cfg.get("crossover_type","Simple Crossover"); body=abs(row["close"]-row["open"])
    def ok():
        if ct=="Simple Crossover": return True
        if ct=="Custom Candle Size": return body>=float(cfg.get("custom_candle_size",10))
        if ct=="ATR Based Candle Size": a=float(row.get("atr",0) or 0); return a==0 or body>=a
        return True
    f,s=int(cfg.get("fast_ema",9)),int(cfg.get("slow_ema",15))
    if ef>es and pf<=ps and ok(): return "buy",f"EMA({f}) crossed ABOVE EMA({s})"
    if ef<es and pf>=ps and ok(): return "sell",f"EMA({f}) crossed BELOW EMA({s})"
    return None,None

def ema_rev(df,idx,pt):
    if idx<1: return False
    row=df.iloc[idx]; prev=df.iloc[idx-1]
    ef,es=row.get("ema_fast",np.nan),row.get("ema_slow",np.nan)
    pf,ps=prev.get("ema_fast",np.nan),prev.get("ema_slow",np.nan)
    if any(np.isnan([ef,es,pf,ps])): return False
    if pt=="buy" and ef<es and pf>=ps: return True
    if pt=="sell" and ef>es and pf<=ps: return True
    return False

def _fdt(dt):
    if hasattr(dt,"strftime"): return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)

# ── Backtest Engine ───────────────────────────────────────────────
def run_backtest(df,cfg):
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
            ema_ex=(sl_t=="Reverse EMA Crossover" or tg_t=="EMA Crossover") and ema_rev(df,i,pt)
            ep=None; er=None; viol=False
            if pt=="buy":
                if row["low"]<=sl:
                    ep=sl; er=f"SL hit Low {row['low']:.2f}<=SL {sl:.2f}"
                    viol=(tg_t!="Trailing Target" and row["high"]>=tgt)
                elif tg_t!="Trailing Target" and row["high"]>=tgt: ep=tgt; er=f"Tgt hit High {row['high']:.2f}>=Tgt {tgt:.2f}"
                elif ema_ex: ep=row["open"]; er="EMA Rev Cross exit"
            else:
                if row["high"]>=sl:
                    ep=sl; er=f"SL hit High {row['high']:.2f}>=SL {sl:.2f}"
                    viol=(tg_t!="Trailing Target" and row["low"]<=tgt)
                elif tg_t!="Trailing Target" and row["low"]<=tgt: ep=tgt; er=f"Tgt hit Low {row['low']:.2f}<=Tgt {tgt:.2f}"
                elif ema_ex: ep=row["open"]; er="EMA Rev Cross exit"
            if ep is None and i==len(df)-1: ep=row["close"]; er="End of data"
            if ep is not None:
                pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                t={"Trade #":tnum,"Type":pt.upper(),"Entry Time":_fdt(pos["entry_time"]),"Exit Time":_fdt(row.name),
                   "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                   "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
                   "Candle High":round(row["high"],4),"Candle Low":round(row["low"],4),
                   "Entry Reason":pos["reason"],"Exit Reason":er,"PnL":pnl,"Qty":qty,"SL/Tgt Violated":viol}
                trades.append(t)
                if viol: violations.append(t)
                pos=None
        if pos is None and pending:
            sig,rsn=pending; pending=None; ep=float(row["open"]); sl,tgt=calc_sl_tgt(ep,sig,row,cfg); tnum+=1
            pos={"type":sig,"entry":ep,"entry_time":row.name,"sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":rsn}
        if pos is None:
            if strat=="EMA Crossover":
                sig,rsn=ema_sig(df,i,cfg)
                if sig: pending=(sig,rsn)
            elif strat=="Simple Buy":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"buy",row,cfg); tnum+=1
                pos={"type":"buy","entry":ep,"entry_time":row.name,"sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Buy"}
            elif strat=="Simple Sell":
                ep=float(row["close"]); sl,tgt=calc_sl_tgt(ep,"sell",row,cfg); tnum+=1
                pos={"type":"sell","entry":ep,"entry_time":row.name,"sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Sell"}
            elif strat=="Elliott Wave" and i>=20:
                ew=detect_ew(df.iloc[:i+1]); bias=ew.get("trade_bias","")
                if bias: sig="buy" if bias=="BUY" else "sell"; pending=(sig,f"EW:{ew.get('current_wave','')}")
    return trades,violations

# ── Optimization ──────────────────────────────────────────────────
def run_opt(df,base_cfg,param_grid,target="Accuracy %",min_tr=5,cb=None):
    """
    Grid-search. Returns list of result dicts (max 20), sorted by target.
    Each result has flattened param keys + stats columns.
    Uses only Custom Points for SL/Target to keep search clean.
    """
    results=[]
    keys=list(param_grid.keys()); combos=list(itertools.product(*param_grid.values()))
    total=len(combos)
    for idx,combo in enumerate(combos):
        cfg={**base_cfg,"sl_type":"Custom Points","target_type":"Custom Points"}
        for k,v in zip(keys,combo): cfg[k]=v
        try:
            fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
            if fast>=slow: 
                if cb: cb(idx+1,total)
                continue
            df_i=add_ind(df.copy(),fast,slow)
            trades,_=run_backtest(df_i,cfg)
            if len(trades)<min_tr:
                if cb: cb(idx+1,total)
                continue
            tdf=pd.DataFrame(trades); wins=len(tdf[tdf["PnL"]>0]); tot_t=len(tdf)
            acc=wins/tot_t*100 if tot_t else 0; total_pnl=tdf["PnL"].sum()
            aw=tdf[tdf["PnL"]>0]["PnL"].mean() if wins else 0
            al=tdf[tdf["PnL"]<=0]["PnL"].mean() if (tot_t-wins) else 0
            rr=abs(aw/al) if al and al!=0 else 0
            score=acc*0.4+(rr*10)*0.3+(20 if total_pnl>0 else 0)*0.3
            res={"Trades":tot_t,"Wins":wins,"Losses":tot_t-wins,
                 "Accuracy %":round(acc,1),"Total P&L":round(total_pnl,2),
                 "Avg Win":round(aw,2),"Avg Loss":round(al,2),"R:R":round(rr,2),"Score":round(score,2)}
            # Store param values (only those actually in the grid, not all cfg)
            for k2,v2 in zip(keys,combo):
                res[k2]=v2
            results.append(res)
        except Exception: pass
        if cb: cb(idx+1,total)
    results.sort(key=lambda x:x.get(target,0),reverse=True)
    return results[:25]

# ── Dhan ─────────────────────────────────────────────────────────
def init_dhan(cid,tok):
    try:
        from dhanhq import dhanhq; return dhanhq(cid,tok)
    except ImportError: st.warning("pip install dhanhq"); return None
    except Exception as e: st.error(f"Dhan: {e}"); return None

def get_ip():
    try: return requests.get("https://api.ipify.org?format=json",timeout=5).json().get("ip","?")
    except Exception:
        try: return requests.get("https://ifconfig.me/ip",timeout=5).text.strip()
        except Exception: return "Could not detect"

def reg_ip_ui():
    ip=get_ip()
    st.info(f"**Your IP:** `{ip}`\n\nSEBI: whitelist at [Dhan Console](https://console.dhan.co) → Profile → API → IP Whitelist. Re-register if IP changes.")

def place_order(dhan,cfg,sig,ltp,is_exit=False):
    if dhan is None: return {"error":"Not connected"}
    try:
        if cfg.get("options_trading",False):
            fno=cfg.get("fno_exchange","NSE_FNO"); qty=int(cfg.get("options_qty",65))
            ot=cfg.get("options_exit_type" if is_exit else "options_entry_type","MARKET"); px=round(ltp,2) if ot=="LIMIT" else 0
            sid=cfg.get("ce_security_id") if sig=="buy" else cfg.get("pe_security_id")
            return dhan.place_order(transactionType="SELL" if is_exit else "BUY",exchangeSegment=fno,productType="INTRADAY",orderType=ot,validity="DAY",securityId=str(sid),quantity=qty,price=px,triggerPrice=0)
        else:
            ot=cfg.get("exit_order_type" if is_exit else "entry_order_type","MARKET"); px=round(ltp,2) if ot=="LIMIT" else 0
            txn=("SELL" if is_exit else "BUY") if sig=="buy" else ("BUY" if is_exit else "SELL")
            return dhan.place_order(security_id=str(cfg.get("security_id","1594")),exchange_segment=cfg.get("exchange","NSE"),transaction_type=txn,quantity=int(cfg.get("dhan_qty",1)),order_type=ot,product_type=cfg.get("product_type","INTRADAY"),price=px)
    except Exception as e: return {"error":str(e)}

# ── Log ───────────────────────────────────────────────────────────
def _log(msg): _ts_append("live_log",f"[{datetime.now(IST).strftime('%H:%M:%S')}] {msg}",mx=300)

# ── Live thread ───────────────────────────────────────────────────
def live_thread(cfg,symbol):
    _log(f"STARTED {symbol} {cfg['timeframe']} {cfg['strategy']}")
    tf_min=TF_MINUTES.get(cfg["timeframe"],5); fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
    strat=cfg.get("strategy","EMA Crossover"); sl_t=cfg.get("sl_type","Custom Points"); tg_t=cfg.get("target_type","Custom Points")
    qty=int(cfg.get("quantity",1)); dhan=st.session_state.get("dhan_client"); en_dhan=cfg.get("enable_dhan",False)
    pending=None; last_sig_candle=None; last_bdy=-1

    while _ts_get("live_running",False):
        try:
            df=fetch_data(symbol,cfg["timeframe"],cfg["period"],quiet=True)
            now=datetime.now(IST)
            if df is None or df.empty: _log("[WARN] No data"); time.sleep(1.5); continue
            df=add_ind(df,fast,slow)
            ltp=float(df["close"].iloc[-1])
            _ts_set("live_data",df); _ts_set("live_ltp",ltp); _ts_set("live_last_ts",now)
            if len(df)>=20: _ts_set("live_ew",detect_ew(df))
            pos=_ts_get("live_position")

            # Exit check
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
                    if ltp<=sl: ep=sl; er=f"SL hit {ltp:.2f}<={sl:.2f}"
                    elif tg_t!="Trailing Target" and ltp>=tgt: ep=tgt; er=f"Tgt hit {ltp:.2f}>={tgt:.2f}"
                else:
                    if ltp>=sl: ep=sl; er=f"SL hit {ltp:.2f}>={sl:.2f}"
                    elif tg_t!="Trailing Target" and ltp<=tgt: ep=tgt; er=f"Tgt hit {ltp:.2f}<={tgt:.2f}"
                if ep is None and (sl_t=="Reverse EMA Crossover" or tg_t=="EMA Crossover"):
                    if ema_rev(df,len(df)-1,pt): ep=ltp; er="EMA Rev Cross exit"
                if ep is not None:
                    pnl=round(((ep-pos["entry"]) if pt=="buy" else (pos["entry"]-ep))*qty,2)
                    t={"Trade #":len(_ts_get("live_trades",[]))+1,"Type":pt.upper(),
                       "Entry Time":_fdt(pos["entry_time"]),"Exit Time":now.strftime("%Y-%m-%d %H:%M:%S"),
                       "Entry Price":round(pos["entry"],4),"Exit Price":round(ep,4),
                       "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
                       "Entry Reason":pos["reason"],"Exit Reason":er,"PnL":pnl,"Qty":qty,"Source":"Live"}
                    _ts_append("live_trades",t); _ts_set("live_position",None)
                    if en_dhan and dhan: _log(f"EXIT order:{place_order(dhan,cfg,pt,ltp,is_exit=True)}")
                    _log(f"EXIT {pt.upper()} @{ep:.2f} | {er} | PnL:{pnl:+.2f}")

            # Entry
            if _ts_get("live_position") is None:
                cd_ok=True
                if cfg.get("cooldown_enabled",True):
                    tf_so_far=_ts_get("live_trades",[])
                    if tf_so_far:
                        try:
                            le=datetime.strptime(tf_so_far[-1].get("Exit Time","2000-01-01 00:00:00"),"%Y-%m-%d %H:%M:%S")
                            le=IST.localize(le)
                            if (now-le).total_seconds()<int(cfg.get("cooldown_seconds",5)): cd_ok=False
                        except Exception: pass

                # ── SIMPLE BUY/SELL: fire on FIRST tick, NO candle boundary ──
                if strat=="Simple Buy" and cd_ok:
                    sl,tgt=calc_sl_tgt(ltp,"buy",df.iloc[-1],cfg)
                    _ts_set("live_position",{"type":"buy","entry":ltp,"entry_time":now,
                        "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Buy"})
                    if en_dhan and dhan: _log(f"ENTRY:{place_order(dhan,cfg,'buy',ltp)}")
                    _log(f"BUY @{ltp:.2f} SL:{sl:.2f} Tgt:{tgt:.2f}")

                elif strat=="Simple Sell" and cd_ok:
                    sl,tgt=calc_sl_tgt(ltp,"sell",df.iloc[-1],cfg)
                    _ts_set("live_position",{"type":"sell","entry":ltp,"entry_time":now,
                        "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":"Simple Sell"})
                    if en_dhan and dhan: _log(f"ENTRY:{place_order(dhan,cfg,'sell',ltp)}")
                    _log(f"SELL @{ltp:.2f} SL:{sl:.2f} Tgt:{tgt:.2f}")

                # ── EMA/EW pending → execute on next candle open ──
                elif pending and cd_ok:
                    sig,rsn=pending; pending=None
                    ep=float(df["open"].iloc[-1]); sl,tgt=calc_sl_tgt(ep,sig,df.iloc[-1],cfg)
                    _ts_set("live_position",{"type":sig,"entry":ep,"entry_time":now,
                        "sl":sl,"sl_init":sl,"tgt":tgt,"tgt_init":tgt,"reason":rsn})
                    if en_dhan and dhan: _log(f"ENTRY:{place_order(dhan,cfg,sig,ep)}")
                    _log(f"ENTRY {sig.upper()} @{ep:.2f} SL:{sl:.2f} Tgt:{tgt:.2f}")

                else:
                    # Candle boundary for EMA/EW signal
                    tm=now.hour*60+now.minute; at_bdy=(tm%tf_min==0) and (tm!=last_bdy)
                    if at_bdy and cd_ok:
                        last_bdy=tm
                        if strat=="EMA Crossover":
                            sig,rsn=ema_sig(df,len(df)-1,cfg)
                            if sig and df.index[-1]!=last_sig_candle:
                                last_sig_candle=df.index[-1]; pending=(sig,rsn)
                                _log(f"SIGNAL {sig.upper()} → next candle")
                        elif strat=="Elliott Wave":
                            ew=_ts_get("live_ew",{}); bias=ew.get("trade_bias","")
                            if bias:
                                s2="buy" if bias=="BUY" else "sell"
                                pending=(s2,f"EW:{ew.get('current_wave','')}"); _log(f"EW {s2.upper()} → next candle")
            time.sleep(1.5)
        except Exception as e: _log(f"[ERR] {e}"); time.sleep(1.5)
    _log("STOPPED.")

# ── Charts ────────────────────────────────────────────────────────
def make_chart(df,trades=None,title="",fast=9,slow=15,live_pos=None,h=520):
    if df is None or df.empty: return go.Figure()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.04,row_heights=[0.78,0.22])
    fig.add_trace(go.Candlestick(x=df.index,open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="Price",increasing_line_color="#48bb78",decreasing_line_color="#fc8181",
        increasing_fillcolor="#48bb78",decreasing_fillcolor="#fc8181"),row=1,col=1)
    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_fast"],name=f"EMA({fast})",line=dict(color="#f6ad55",width=1.5)),row=1,col=1)
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["ema_slow"],name=f"EMA({slow})",line=dict(color="#76e4f7",width=1.5)),row=1,col=1)
    if "volume" in df.columns:
        vc=["#48bb78" if c>=o else "#fc8181" for c,o in zip(df["close"],df["open"])]
        fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Vol",marker_color=vc,opacity=0.5),row=2,col=1)
    if trades:
        for t in trades:
            try:
                et=pd.to_datetime(t["Entry Time"]); xt=pd.to_datetime(t["Exit Time"])
                c="#48bb78" if t["Type"]=="BUY" else "#fc8181"; sym="triangle-up" if t["Type"]=="BUY" else "triangle-down"
                pc="#48bb78" if t.get("PnL",0)>=0 else "#fc8181"
                fig.add_trace(go.Scatter(x=[et],y=[t["Entry Price"]],mode="markers+text",marker=dict(symbol=sym,size=11,color=c),text=[f"E:{t['Entry Price']:.0f}"],textposition="top center",textfont=dict(size=8,color=c),showlegend=False),row=1,col=1)
                fig.add_trace(go.Scatter(x=[xt],y=[t["Exit Price"]],mode="markers+text",marker=dict(symbol="x",size=9,color=pc),text=[f"X:{t['Exit Price']:.0f}"],textposition="bottom center",textfont=dict(size=8,color=pc),showlegend=False),row=1,col=1)
                fig.add_shape(type="line",x0=et,x1=xt,y0=t["SL"],y1=t["SL"],line=dict(color="#fc8181",width=1,dash="dot"),row=1,col=1)
                fig.add_shape(type="line",x0=et,x1=xt,y0=t["Target"],y1=t["Target"],line=dict(color="#48bb78",width=1,dash="dot"),row=1,col=1)
            except Exception: pass
    if live_pos:
        try:
            c="#48bb78" if live_pos["type"]=="buy" else "#fc8181"; sym="triangle-up" if live_pos["type"]=="buy" else "triangle-down"
            fig.add_trace(go.Scatter(x=[live_pos["entry_time"]],y=[live_pos["entry"]],mode="markers+text",name="Entry",marker=dict(symbol=sym,size=16,color=c,line=dict(color="white",width=2)),text=[f"ENTRY {live_pos['entry']:.2f}"],textposition="top center"),row=1,col=1)
            fig.add_hline(y=live_pos["sl"],line=dict(color="#fc8181",width=2,dash="dot"),annotation_text=f"SL {live_pos['sl']:.2f}",annotation_font=dict(color="#fc8181"),row=1,col=1)
            fig.add_hline(y=live_pos["tgt"],line=dict(color="#48bb78",width=2,dash="dot"),annotation_text=f"Tgt {live_pos['tgt']:.2f}",annotation_font=dict(color="#48bb78"),row=1,col=1)
        except Exception: pass
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",font=dict(color="#e2e8f0",size=11),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0,r=0,t=36,b=0),height=h,xaxis_rangeslider_visible=False,
        xaxis2=dict(showgrid=True,gridcolor="#1a1f2e"),yaxis=dict(showgrid=True,gridcolor="#1a1f2e"),yaxis2=dict(showgrid=True,gridcolor="#1a1f2e"))
    if title: fig.update_layout(title=dict(text=title,font=dict(size=14,color="#90cdf4")))
    return fig

def make_ew_chart(df,ew,fast,slow):
    fig=make_chart(df.tail(150),title="Elliott Wave",fast=fast,slow=slow,h=420)
    wc={"1":"#f6ad55","2":"#fc8181","3":"#48bb78","4":"#e9d8a6","5":"#76e4f7","A":"#fc8181","B":"#48bb78","C":"#fc8181"}
    for w in ew.get("waves",[]):
        s=w["start"]; e=w["end"]; c=wc.get(w["label"],"#a0aec0")
        fig.add_trace(go.Scatter(x=[s["dt"],e["dt"]],y=[s["price"],e["price"]],mode="lines+markers+text",line=dict(color=c,width=2.5),marker=dict(size=8,color=c),text=["",f"W{w['label']}"],textposition="top center",textfont=dict(color=c,size=12),name=f"W{w['label']}"),row=1,col=1)
    return fig

def pnl_chart(trades):
    pnls=[t.get("PnL",0) for t in trades]; cum=[0]+list(np.cumsum(pnls))
    c="#48bb78" if cum[-1]>=0 else "#fc8181"; fill="rgba(72,187,120,0.1)" if c=="#48bb78" else "rgba(252,129,129,0.1)"
    fig=go.Figure(go.Scatter(x=list(range(len(cum))),y=cum,mode="lines+markers",fill="tozeroy",fillcolor=fill,line=dict(color=c,width=2),name="P&L"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",height=260,margin=dict(l=0,r=0,t=20,b=0),xaxis_title="Trade #",yaxis_title="P&L",font=dict(color="#e2e8f0"))
    return fig

# ── Sidebar ────────────────────────────────────────────────────────
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
        periods=TF_PERIODS[tf]; period=st.selectbox("Period",periods,index=min(1,len(periods)-1),key="s_period")
        cfg.update(timeframe=tf,period=period)

        st.markdown("### 🧠 Strategy")
        strat=st.selectbox("Strategy",["EMA Crossover","Simple Buy","Simple Sell","Elliott Wave"],key="s_strat")
        cfg["strategy"]=strat
        if strat=="EMA Crossover":
            c1,c2=st.columns(2)
            fe=c1.number_input("Fast EMA",1,500,int(st.session_state.get("_apply_fe",9)),key="s_fe")
            se=c2.number_input("Slow EMA",1,500,int(st.session_state.get("_apply_se",15)),key="s_se")
            cfg.update(fast_ema=fe,slow_ema=se)
            uang=st.checkbox("Angle Filter",False,key="s_uang")
            mang=st.number_input("Min Angle",0.0,90.0,0.0,0.5,key="s_mang") if uang else 0.0
            cfg.update(use_angle_filter=uang,min_ema_angle=mang)
            ct=st.selectbox("Crossover Type",["Simple Crossover","Custom Candle Size","ATR Based Candle Size"],key="s_ct")
            cfg["crossover_type"]=ct
            if ct=="Custom Candle Size": cfg["custom_candle_size"]=st.number_input("Min Body",0.0,value=10.0,key="s_cs")
        else: cfg.update(fast_ema=9,slow_ema=15,use_angle_filter=False,min_ema_angle=0.0,crossover_type="Simple Crossover",custom_candle_size=10.0)

        st.markdown("### 📦 Quantity")
        cfg["quantity"]=st.number_input("Qty",1,1_000_000,1,key="s_qty")

        st.markdown("### 🛡 Stop Loss")
        sl_t=st.selectbox("SL Type",["Custom Points","ATR Based","Trailing SL","Reverse EMA Crossover","Risk-Reward Based"],key="s_slt")
        cfg["sl_type"]=sl_t
        if sl_t=="ATR Based":
            cfg["atr_sl_mult"]=st.number_input("ATR Mult SL",0.1,10.0,1.5,0.1,key="s_asm"); cfg["sl_points"]=10.0
        else:
            cfg["sl_points"]=st.number_input("SL Pts",0.1,value=float(st.session_state.get("_apply_slp",10.0)),step=0.5,key="s_slp"); cfg["atr_sl_mult"]=1.5

        st.markdown("### 🎯 Target")
        tg_t=st.selectbox("Target Type",["Custom Points","ATR Based Target","Trailing Target","EMA Crossover","Risk-Reward Based"],key="s_tgt")
        cfg["target_type"]=tg_t
        if tg_t=="ATR Based Target":
            cfg["atr_tgt_mult"]=st.number_input("ATR Mult Tgt",0.1,20.0,3.0,0.1,key="s_atm"); cfg["target_points"]=20.0
        elif tg_t=="Risk-Reward Based":
            cfg["rr_ratio"]=st.number_input("R:R",0.1,20.0,float(st.session_state.get("_apply_rr",2.0)),0.1,key="s_rr"); cfg["target_points"]=20.0; cfg["atr_tgt_mult"]=3.0
        elif tg_t=="Trailing Target":
            cfg["target_points"]=st.number_input("Trail Dist",0.1,value=20.0,step=0.5,key="s_tgp"); cfg["atr_tgt_mult"]=3.0; st.caption("Display only")
        else:
            cfg["target_points"]=st.number_input("Tgt Pts",0.1,value=float(st.session_state.get("_apply_tgp",20.0)),step=0.5,key="s_tgp2"); cfg["atr_tgt_mult"]=3.0
        cfg.setdefault("rr_ratio",2.0)

        st.markdown("### ⚙️ Controls")
        cd=st.checkbox("Cooldown",True,key="s_cdn"); cds=st.number_input("Cooldown s",1,3600,5,key="s_cds") if cd else 5
        no_ovlp=st.checkbox("No Overlap",True,key="s_novlp"); cfg.update(cooldown_enabled=cd,cooldown_seconds=cds,no_overlap=no_ovlp)

        st.markdown("### 🏦 Dhan Broker")
        en_dhan=st.checkbox("Enable Dhan",False,key="s_dhan"); cfg["enable_dhan"]=en_dhan
        if en_dhan:
            cid=st.text_input("Client ID",key="s_cid"); tok=st.text_input("Token",key="s_tok",type="password")
            cfg.update(dhan_client_id=cid,dhan_access_token=tok)
            if st.button("Connect & Show IP",use_container_width=True,key="s_conn"):
                cl=init_dhan(cid,tok)
                if cl: st.session_state.dhan_client=cl; reg_ip_ui(); st.success("Connected!")
                else: st.error("Failed")
            opts=st.checkbox("Options Trading",False,key="s_opts"); cfg["options_trading"]=opts
            if opts:
                cfg["fno_exchange"]=st.selectbox("FNO",["NSE_FNO","BSE_FNO"],key="s_fnoe")
                cfg["ce_security_id"]=st.text_input("CE ID",key="s_ceid"); cfg["pe_security_id"]=st.text_input("PE ID",key="s_peid")
                cfg["options_qty"]=st.number_input("Opts Qty",1,value=65,key="s_oqty")
                cfg["options_entry_type"]=st.selectbox("Entry",["MARKET","LIMIT"],key="s_oent"); cfg["options_exit_type"]=st.selectbox("Exit",["MARKET","LIMIT"],key="s_oext")
            else:
                cfg["product_type"]=st.selectbox("Product",["INTRADAY","DELIVERY"],key="s_prod")
                cfg["exchange"]=st.selectbox("Exchange",["NSE","BSE"],key="s_exc")
                cfg["security_id"]=st.text_input("Security ID","1594",key="s_sid"); cfg["dhan_qty"]=st.number_input("Order Qty",1,value=1,key="s_dqty")
                cfg["entry_order_type"]=st.selectbox("Entry Order",["LIMIT","MARKET"],key="s_eord"); cfg["exit_order_type"]=st.selectbox("Exit Order",["MARKET","LIMIT"],key="s_xord")
    return cfg

# ── Shared widgets ─────────────────────────────────────────────────
def ltp_banner(cfg):
    ltp=_ts_get("live_ltp"); prev=_ts_get("live_prev_close")
    if ltp is None: ltp,prev=fetch_ltp(cfg.get("symbol","^NSEI")); _ts_set("live_ltp",ltp); _ts_set("live_prev_close",prev)
    ltp=ltp or 0.0; prev=prev or ltp; chg=ltp-prev; pct=chg/prev*100 if prev else 0
    arrow="▲" if chg>=0 else "▼"; cls="ltp-up" if chg>=0 else "ltp-down"
    st.markdown(f'<div class="ltp-bar"><div><div class="ltp-ticker">📈 {cfg.get("ticker_name","—")} · {cfg.get("symbol","")}</div><div class="ltp-price">₹ {ltp:,.2f}</div></div><div class="{cls}">{arrow} {abs(chg):.2f} ({abs(pct):.2f}%)</div><div class="ltp-meta">{cfg.get("timeframe","")} · {cfg.get("period","")} · {cfg.get("strategy","")} | {datetime.now(IST).strftime("%H:%M:%S IST")}</div></div>',unsafe_allow_html=True)

def cfg_box(cfg):
    st.markdown(f'<div class="cfg"><b>📌</b> {cfg.get("ticker_name","—")} ({cfg.get("symbol","—")}) | <b>⏱</b> {cfg.get("timeframe","—")}/{cfg.get("period","—")} | <b>🧠</b> {cfg.get("strategy","—")} | <b>📦</b> {cfg.get("quantity",1)}<br><b>EMA:</b> Fast={cfg.get("fast_ema",9)} Slow={cfg.get("slow_ema",15)} | <b>Crossover:</b> {cfg.get("crossover_type","—")}<br><b>🛡 SL:</b> {cfg.get("sl_type","—")} ({cfg.get("sl_points",10)} pts) | <b>🎯 Tgt:</b> {cfg.get("target_type","—")} ({cfg.get("target_points",20)} pts)<br><b>🔄 Cooldown:</b> {"✅ "+str(cfg.get("cooldown_seconds",5))+"s" if cfg.get("cooldown_enabled") else "❌"}</div>',unsafe_allow_html=True)

def style_df(df):
    if df.empty: return df.style
    def rc(row):
        pnl=row.get("PnL",0); c="rgba(72,187,120,0.10)" if pnl>0 else ("rgba(252,129,129,0.10)" if pnl<0 else "")
        return [f"background-color:{c}"]*len(row)
    styled=df.style.apply(rc,axis=1)
    if "PnL" in df.columns:
        styled=styled.map(lambda v:(f"color:{'#48bb78' if v>0 else '#fc8181'};font-weight:700" if isinstance(v,(int,float)) else ""),subset=["PnL"])
    if "SL/Tgt Violated" in df.columns:
        styled=styled.map(lambda v:("background-color:#2d1515;color:#fc8181;font-weight:700" if v is True else ""),subset=["SL/Tgt Violated"])
    return styled

def trade_stats(trades):
    if not trades: return
    df=pd.DataFrame(trades); wins=df[df["PnL"]>0]; loss=df[df["PnL"]<=0]
    tot=df["PnL"].sum(); acc=len(wins)/len(df)*100 if len(df) else 0
    aw=wins["PnL"].mean() if len(wins) else 0; al=loss["PnL"].mean() if len(loss) else 0
    cols=st.columns(6)
    for col,lbl,val,color in [(cols[0],"Trades",len(df),"#e2e8f0"),(cols[1],"Winners",len(wins),"#48bb78"),
        (cols[2],"Losers",len(loss),"#fc8181"),(cols[3],"Accuracy",f"{acc:.1f}%","#f6ad55"),
        (cols[4],"Total P&L",f"₹{tot:+.2f}","#48bb78" if tot>=0 else "#fc8181"),(cols[5],"Avg W/L",f"₹{aw:.2f}/₹{al:.2f}","#76e4f7")]:
        col.markdown(f'<div class="mc"><div style="color:#a0aec0;font-size:11px">{lbl}</div><div style="color:{color};font-size:16px;font-weight:700;margin-top:3px">{val}</div></div>',unsafe_allow_html=True)

def ew_rec_panel(ew,cfg):
    """Full EW trade recommendation with wave context, entry, SL, target."""
    rec=ew.get("recommendation")
    if not rec: return
    sig=rec["signal"]; entry=rec["entry"]; sl_s=rec["sl_suggestion"]; tgt_s=rec["tgt_suggestion"]
    sl_b=rec.get("sl_basis","EW structure"); tgt_b=rec.get("tgt_basis","EW projection"); wctx=rec.get("wave_context","")
    sig_c="#48bb78" if sig=="BUY" else "#fc8181"
    ltp_sl=round(abs(entry-sl_s),2); ltp_tgt=round(abs(entry-tgt_s),2)
    rr_ew=round(ltp_tgt/ltp_sl,2) if ltp_sl else 0
    # Build the recommendation HTML using separate variables (NO nested f-strings with quotes)
    arrow_icon="🔼" if sig=="BUY" else "🔽"
    strat_sl=cfg.get("sl_points",10); strat_sl_type=cfg.get("sl_type","—")
    strat_tgt=cfg.get("target_points",20); strat_tgt_type=cfg.get("target_type","—")
    html_inner=(
        f'<div class="ewr">'
        f'<div style="font-size:15px;font-weight:700;color:{sig_c};margin-bottom:8px">{arrow_icon} EW {sig} Signal &nbsp;|&nbsp; <span style="font-size:12px;color:#a0aec0">{wctx}</span></div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px">'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center"><div style="color:#a0aec0;font-size:11px">Entry</div><div style="color:#e2e8f0;font-size:16px;font-weight:700">₹{entry:,.2f}</div><div style="color:#718096;font-size:10px">Current LTP</div></div>'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center"><div style="color:#a0aec0;font-size:11px">EW Stop Loss</div><div style="color:#fc8181;font-size:16px;font-weight:700">₹{sl_s:,.2f}</div><div style="color:#718096;font-size:10px">{sl_b}</div></div>'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center"><div style="color:#a0aec0;font-size:11px">EW Target</div><div style="color:#48bb78;font-size:16px;font-weight:700">₹{tgt_s:,.2f}</div><div style="color:#718096;font-size:10px">{tgt_b}</div></div>'
        f'<div style="background:#0e1117;border-radius:8px;padding:10px;text-align:center"><div style="color:#a0aec0;font-size:11px">EW R:R</div><div style="color:#f6ad55;font-size:16px;font-weight:700">{rr_ew}x</div><div style="color:#718096;font-size:10px">{ltp_tgt:.0f}/{ltp_sl:.0f}pts</div></div>'
        f'</div>'
        f'<div style="padding:8px;background:#0e1117;border-radius:6px;font-size:12px;color:#a0aec0">'
        f'💡 <b style="color:#f6ad55">Your strategy:</b> SL={strat_sl}pts ({strat_sl_type}) | Target={strat_tgt}pts ({strat_tgt_type}) → <span style="color:#718096">Adjust in sidebar to match EW levels if desired</span>'
        f'</div></div>'
    )
    st.markdown(html_inner,unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 1 — BACKTESTING
# ══════════════════════════════════════════════════════════════════
def tab_backtest(cfg):
    ltp_banner(cfg); st.markdown("## 🔬 Backtesting Engine")
    st.info("EMA: signal on candle N → entry at N+1 open | Buy: Low vs SL first | Sell: High vs SL first")
    if st.button("▶ Run Backtest",type="primary",key="btn_bt"):
        with st.spinner("Fetching…"):
            df=fetch_data(cfg["symbol"],cfg["timeframe"],cfg["period"])
        if df is None or df.empty: st.error("No data."); return
        df=add_ind(df,int(cfg.get("fast_ema",9)),int(cfg.get("slow_ema",15)))
        with st.spinner("Running…"): trades,viol=run_backtest(df,cfg)
        st.session_state.bt_trades=trades; st.session_state.bt_viol=viol
        st.session_state.bt_df=df; st.session_state.bt_ran=True
    if not st.session_state.bt_ran:
        st.markdown('<div style="color:#718096;text-align:center;padding:60px">Click ▶ Run Backtest</div>',unsafe_allow_html=True); return
    trades=st.session_state.bt_trades; viol=st.session_state.bt_viol; df=st.session_state.bt_df
    fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
    if not trades: st.warning("No trades."); return
    trade_stats(trades); st.markdown("---")
    if viol:
        st.error(f"⚠️ {len(viol)} SL/Target Violations")
        with st.expander("View Violations"): st.dataframe(style_df(pd.DataFrame(viol)),use_container_width=True)
    else: st.success("✅ No violations!")
    st.markdown("### 📊 Chart")
    st.plotly_chart(make_chart(df.tail(500),trades=trades[:50],title=f"Backtest — {cfg.get('ticker_name')}",fast=fast,slow=slow,h=560),use_container_width=True,key="bt_chart")
    st.markdown("### 📈 P&L"); st.plotly_chart(pnl_chart(trades),use_container_width=True,key="bt_pnl")
    st.markdown(f"### 📋 Trade Log ({len(trades)})")
    tdf=pd.DataFrame(trades); co=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target","Candle High","Candle Low","Entry Reason","Exit Reason","PnL","Qty","SL/Tgt Violated"]
    st.dataframe(style_df(tdf[[c for c in co if c in tdf.columns]]),use_container_width=True,height=420)
    st.download_button("⬇ CSV",tdf.to_csv(index=False).encode(),"bt.csv","text/csv",key="bt_dl")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ══════════════════════════════════════════════════════════════════
def tab_live(cfg):
    _sync()
    # ── AUTO-REFRESH: no time.sleep() – just rerun without blocking ──
    # Uses a session_state timestamp to throttle to max 1 rerun per 2s
    if _ts_get("live_running",False):
        now_ts=time.time()
        if now_ts - st.session_state.get("_live_tick",0.0) >= 2.0:
            st.session_state["_live_tick"]=now_ts
            st.rerun()  # <-- Called BEFORE rendering so we get fresh data

    ltp_banner(cfg); st.markdown("## 🚀 Live Trading")
    c1,c2,c3,_=st.columns([1,1,1,3])
    start_btn=c1.button("▶ Start",type="primary",use_container_width=True,key="btn_start")
    stop_btn =c2.button("⏹ Stop",               use_container_width=True,key="btn_stop")
    sq_btn   =c3.button("✖ Squareoff",           use_container_width=True,key="btn_sq")

    if start_btn and not _ts_get("live_running",False):
        _ts_set("live_running",True); _ts_set("live_status","RUNNING")
        _ts_set("live_log",[]); _ts_set("live_position",None)
        t=threading.Thread(target=live_thread,args=(cfg,cfg["symbol"]),daemon=True)
        st.session_state.live_thread=t; t.start()
        st.session_state["_live_tick"]=0.0  # force immediate rerun on next cycle
        st.rerun()

    if stop_btn and _ts_get("live_running",False):
        _ts_set("live_running",False); _ts_set("live_status","STOPPED"); _sync(); st.info("Stopping…")

    if sq_btn and _ts_get("live_position"):
        pos=_ts_get("live_position"); ltp=_ts_get("live_ltp") or pos["entry"]
        pnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        t2={"Trade #":len(_ts_get("live_trades",[]))+1,"Type":pos["type"].upper(),
            "Entry Time":_fdt(pos["entry_time"]),"Exit Time":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "Entry Price":round(pos["entry"],4),"Exit Price":round(ltp,4),
            "SL":round(pos["sl_init"],4),"Target":round(pos["tgt_init"],4),
            "Entry Reason":pos["reason"],"Exit Reason":"Manual Squareoff","PnL":pnl,"Qty":cfg.get("quantity",1),"Source":"Live"}
        _ts_append("live_trades",t2); _ts_set("live_position",None)
        if cfg.get("enable_dhan") and st.session_state.dhan_client:
            place_order(st.session_state.dhan_client,cfg,pos["type"],ltp,is_exit=True)
        _sync(); st.success(f"Squared off! PnL:{pnl:+.2f}")

    status=_ts_get("live_status","STOPPED"); bc="#48bb78" if status=="RUNNING" else "#fc8181"
    st.markdown(f'<div style="display:inline-block;background:{bc};color:white;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;margin-bottom:8px">{"🟢" if status=="RUNNING" else "🔴"} {status}</div>',unsafe_allow_html=True)

    st.markdown("#### 🔧 Config"); cfg_box(cfg)

    # ── EMA Values Panel ─────────────────────────────────────────
    # Only render when we have actual data from the live thread
    df=_ts_get("live_data"); ts=_ts_get("live_last_ts")
    has_data = df is not None and not df.empty

    if has_data:
        lr=df.iloc[-1]
        ef_val=lr.get("ema_fast",float("nan")); es_val=lr.get("ema_slow",float("nan")); atr_val=float(lr.get("atr",0.0) or 0.0)
        fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
        ef_prev = df.iloc[-2].get("ema_fast",float("nan")) if len(df)>=2 else float("nan")
        es_prev = df.iloc[-2].get("ema_slow",float("nan")) if len(df)>=2 else float("nan")

        # Determine crossover status
        if not any(np.isnan([ef_val,es_val,ef_prev,es_prev])):
            if ef_val>es_val and ef_prev<=es_prev:   xst="🔀 JUST CROSSED UP (Bullish)"
            elif ef_val<es_val and ef_prev>=es_prev: xst="🔀 JUST CROSSED DOWN (Bearish)"
            elif ef_val>es_val:                      xst="▲ Fast > Slow (Bullish)"
            else:                                    xst="▼ Fast < Slow (Bearish)"
        else:
            xst="Calculating…"

        # ── FIX: compute spread_color as a variable to avoid nested f-string quotes ──
        ef_str  = f"{ef_val:.2f}"  if not np.isnan(ef_val) else "—"
        es_str  = f"{es_val:.2f}"  if not np.isnan(es_val) else "—"
        atr_str = f"{atr_val:.2f}" if not np.isnan(atr_val) else "—"
        spread  = ef_val - es_val if (not np.isnan(ef_val) and not np.isnan(es_val)) else 0.0
        spread_str   = f"{spread:.2f}"
        spread_color = "#48bb78" if spread >= 0 else "#fc8181"

        st.markdown("#### ⚡ Live EMA Values")
        ecols=st.columns(5)
        ecols[0].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">EMA({fast})</div><div style="color:#f6ad55;font-size:20px;font-weight:700">{ef_str}</div><div style="color:#718096;font-size:10px">Fast EMA</div></div>',unsafe_allow_html=True)
        ecols[1].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">EMA({slow})</div><div style="color:#76e4f7;font-size:20px;font-weight:700">{es_str}</div><div style="color:#718096;font-size:10px">Slow EMA</div></div>',unsafe_allow_html=True)
        ecols[2].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">Spread</div><div style="color:{spread_color};font-size:20px;font-weight:700">{spread_str}</div><div style="color:#718096;font-size:10px">Fast − Slow</div></div>',unsafe_allow_html=True)
        ecols[3].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">ATR(14)</div><div style="color:#e9d8a6;font-size:20px;font-weight:700">{atr_str}</div><div style="color:#718096;font-size:10px">Avg True Range</div></div>',unsafe_allow_html=True)
        ecols[4].markdown(f'<div class="ec"><div style="color:#a0aec0;font-size:11px">EMA Status</div><div style="color:#e2e8f0;font-size:12px;font-weight:700;margin-top:6px">{xst}</div></div>',unsafe_allow_html=True)

        ts_str = ts.strftime("%H:%M:%S") if ts else "—"
        st.markdown(f'<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:8px;padding:8px 16px;font-size:12px;color:#a0aec0;margin:6px 0">📡 <b style="color:#76e4f7">Last Candle [{_fdt(df.index[-1])}]</b>: O:{lr["open"]:.2f} H:{lr["high"]:.2f} L:{lr["low"]:.2f} C:{lr["close"]:.2f} | Fetched: <b style="color:#48bb78">{ts_str} IST</b></div>',unsafe_allow_html=True)
    else:
        if _ts_get("live_running",False):
            st.info("⏳ Fetching first data tick… (1–2 seconds)")

    # ── Current Position ──────────────────────────────────────────
    pos=_ts_get("live_position"); st.markdown("#### 📌 Current Position")
    if pos:
        ltp=_ts_get("live_ltp") or pos["entry"]
        upnl=round(((ltp-pos["entry"]) if pos["type"]=="buy" else (pos["entry"]-ltp))*int(cfg.get("quantity",1)),2)
        bc2="#48bb78" if upnl>=0 else "#fc8181"; ptc="#48bb78" if pos["type"]=="buy" else "#fc8181"
        bg_c="rgba(72,187,120,0.12)" if upnl>=0 else "rgba(252,129,129,0.12)"
        arrow="🔼 BUY" if pos["type"]=="buy" else "🔽 SELL"
        st.markdown(f'<div style="background:{bg_c};border:1px solid {bc2};border-radius:10px;padding:14px"><b style="color:{ptc};font-size:16px">{arrow}</b> &nbsp; Entry:<b>{pos["entry"]:.2f}</b> | LTP:<b style="color:{bc2}">{ltp:.2f}</b> | SL:<b style="color:#fc8181">{pos["sl"]:.2f}</b> | Target:<b style="color:#48bb78">{pos["tgt"]:.2f}</b> | P&L:<b style="color:{bc2}">₹{upnl:+.2f}</b> | Time:{_fdt(pos["entry_time"])}<br><span style="color:#a0aec0;font-size:11px">{pos["reason"]}</span></div>',unsafe_allow_html=True)
    else:
        st.info("📭 No open position")

    # ── Chart ─────────────────────────────────────────────────────
    if has_data:
        fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
        st.plotly_chart(make_chart(df.tail(300),title=f"Live — {cfg.get('ticker_name')}",fast=fast,slow=slow,live_pos=pos,h=500),use_container_width=True,key="live_chart")

    # ── Elliott Wave ──────────────────────────────────────────────
    st.markdown("---"); st.markdown("#### 🌊 Elliott Wave Analysis")
    ew=_ts_get("live_ew",{})
    # Only show EW panel when we have a real result (ew["ok"]==True)
    if ew and ew.get("ok") and ew.get("status","") not in ("","Insufficient data"):
        s=ew.get("status","—"); d=ew.get("direction","UNCLEAR"); cw=ew.get("current_wave","—"); cp=ew.get("current_price",0)
        dc={"BULLISH":"#48bb78","BEARISH":"#fc8181","CORRECTIVE_DOWN":"#f6ad55","CORRECTIVE_UP":"#76e4f7","UNCLEAR":"#a0aec0"}.get(d,"#a0aec0")
        wcols=st.columns(4)
        for col,lbl,val,color in [(wcols[0],"Structure",s,dc),(wcols[1],"Direction",d,dc),(wcols[2],"Current Wave",cw,"#e2e8f0"),(wcols[3],"Price",f"{cp:,.2f}","#e2e8f0")]:
            col.markdown(f'<div class="mc"><div style="color:#a0aec0;font-size:11px">{lbl}</div><div style="color:{color};font-size:14px;font-weight:700;margin-top:3px">{val}</div></div>',unsafe_allow_html=True)
        det=ew.get("wave_detail",{}); tgts=ew.get("next_targets",{})
        if det:
            st.markdown("**Completed Waves:**"); html='<div style="display:flex;flex-wrap:wrap;gap:6px">'
            for k,v in det.items(): html+=f'<div class="wl"><b style="color:#f6ad55">{k}</b>: {v}</div>'
            st.markdown(html+"</div>",unsafe_allow_html=True)
        if tgts:
            st.markdown("**Wave Projections:**"); tcols=st.columns(len(tgts))
            for i,(n,v) in enumerate(tgts.items()):
                cl="#48bb78" if v>cp else "#fc8181"
                tcols[i].markdown(f'<div class="mc" style="text-align:center"><div style="color:#a0aec0;font-size:11px">{n}</div><div style="color:{cl};font-size:14px;font-weight:700">{v:,.2f}</div></div>',unsafe_allow_html=True)
        if ew.get("recommendation"):
            st.markdown("#### 🎯 Trade Recommendation"); ew_rec_panel(ew,cfg)
        am=ew.get("after_msg",""); bias=ew.get("trade_bias","")
        if am: st.info(f"💡 {am}")
        if bias: st.success(f"📊 EW Trade Bias: **{bias}**")
        if has_data and len(ew.get("waves",[]))>0:
            fast=int(cfg.get("fast_ema",9)); slow=int(cfg.get("slow_ema",15))
            st.plotly_chart(make_ew_chart(df,ew,fast,slow),use_container_width=True,key="ew_live")
    elif _ts_get("live_running",False):
        st.info("🌊 Elliott Wave will appear once enough candles are loaded…")
    else:
        st.info("🌊 Start live trading to see Elliott Wave analysis.")

    # ── Log ───────────────────────────────────────────────────────
    st.markdown("#### 📟 Activity Log")
    logs=_ts_get("live_log",[])
    log_html='<div class="lb">'
    for ln in reversed(logs[-80:]):
        c="#fc8181" if "ERR" in ln or "WARN" in ln else "#48bb78" if ("BUY" in ln or "SELL" in ln or "START" in ln) else "#f6ad55" if "EXIT" in ln or "SIGNAL" in ln else "#a0aec0"
        log_html+=f'<div style="color:{c}">{ln}</div>'
    st.markdown(log_html+"</div>",unsafe_allow_html=True)

    # ── Completed trades ──────────────────────────────────────────
    lt=_ts_get("live_trades",[])
    if lt:
        st.markdown(f"#### ✅ Completed Trades ({len(lt)})"); trade_stats(lt)
        ltd=pd.DataFrame(lt); cs=["Trade #","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target","Entry Reason","Exit Reason","PnL","Qty"]
        st.dataframe(style_df(ltd[[c for c in cs if c in ltd.columns]]),use_container_width=True,height=280)
        st.plotly_chart(pnl_chart(lt),use_container_width=True,key="live_pnl")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — TRADE HISTORY
# ══════════════════════════════════════════════════════════════════
def tab_history(cfg):
    _sync(); ltp_banner(cfg); st.markdown("## 📚 Trade History")
    all_trades=[]
    for t in st.session_state.bt_trades: tc=t.copy(); tc["Source"]="🔬 Backtest"; all_trades.append(tc)
    for t in _ts_get("live_trades",[]): tc=t.copy(); tc.setdefault("Source","🚀 Live"); all_trades.append(tc)
    if not all_trades:
        st.markdown('<div style="color:#718096;text-align:center;padding:60px">No trades yet.</div>',unsafe_allow_html=True); return
    df=pd.DataFrame(all_trades)
    f1,f2,f3=st.columns(3)
    src=f1.selectbox("Source",["All"]+sorted(df["Source"].unique().tolist()),key="h_src")
    typ=f2.selectbox("Type",["All","BUY","SELL"],key="h_typ"); sv=f3.checkbox("Only Violations",False,key="h_viol")
    if src!="All": df=df[df["Source"]==src]
    if typ!="All": df=df[df["Type"]==typ]
    if sv and "SL/Tgt Violated" in df.columns: df=df[df["SL/Tgt Violated"]==True]
    trade_stats(df.to_dict("records"))
    if not df.empty:
        st.plotly_chart(pnl_chart(df.to_dict("records")),use_container_width=True,key="hist_pnl")
        co=["Trade #","Source","Type","Entry Time","Exit Time","Entry Price","Exit Price","SL","Target","Entry Reason","Exit Reason","PnL","Qty","SL/Tgt Violated"]
        sc=[c for c in co if c in df.columns]
        st.dataframe(style_df(df[sc]),use_container_width=True,height=460)
        st.download_button("⬇ CSV",df[sc].to_csv(index=False).encode(),"history.csv","text/csv",key="h_dl")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════
def tab_optimize(cfg):
    ltp_banner(cfg); st.markdown("## 🔧 Strategy Optimization")
    st.info("Grid-search → click **✅ Apply** on any result to load those parameters into the sidebar.")

    with st.expander("⚙️ Setup", expanded=True):
        oc1,oc2=st.columns(2)
        with oc1:
            opt_strat=st.selectbox("Strategy",["EMA Crossover","Simple Buy","Simple Sell","Elliott Wave"],key="opt_strat")
            opt_metric=st.selectbox("Optimize For",["Accuracy %","Total P&L","Score","R:R"],key="opt_metric")
            min_trades=int(st.number_input("Min Trades",1,1000,5,key="opt_min_t"))
            opt_qty=int(st.number_input("Qty",1,1_000_000,int(cfg.get("quantity",1)),key="opt_qty"))
        with oc2:
            opt_tf=st.selectbox("Interval",list(TF_PERIODS.keys()),
                index=list(TF_PERIODS.keys()).index(cfg.get("timeframe","15m")),key="opt_tf")
            opt_period=st.selectbox("Period",TF_PERIODS[opt_tf],key="opt_period")
            desired_acc=float(st.number_input("Desired Accuracy %",0.0,100.0,60.0,key="opt_acc"))
            st.markdown(f"**Symbol:** `{cfg.get('symbol','—')}`")

        st.markdown("**Search Ranges:**")
        pc1,pc2,pc3=st.columns(3)

        if opt_strat=="EMA Crossover":
            fe_min=int(pc1.number_input("Fast min",1,100,5,key="opt_fem"))
            fe_max=int(pc2.number_input("Fast max",1,200,20,key="opt_feM"))
            fe_step=max(1,int(pc3.number_input("Fast step",1,50,2,key="opt_feS")))
            se_min=int(pc1.number_input("Slow min",5,500,10,key="opt_sem"))
            se_max=int(pc2.number_input("Slow max",5,500,50,key="opt_seM"))
            se_step=max(1,int(pc3.number_input("Slow step",1,50,5,key="opt_seS")))

        sl_min=float(pc1.number_input("SL pts min",1.0,5000.0,5.0,key="opt_slm"))
        sl_max=float(pc2.number_input("SL pts max",1.0,5000.0,50.0,key="opt_slM"))
        sl_step=float(pc3.number_input("SL pts step",0.5,500.0,5.0,key="opt_slS"))
        tg_min=float(pc1.number_input("Tgt pts min",1.0,10000.0,10.0,key="opt_tgm"))
        tg_max=float(pc2.number_input("Tgt pts max",1.0,10000.0,100.0,key="opt_tgM"))
        tg_step=float(pc3.number_input("Tgt pts step",0.5,500.0,10.0,key="opt_tgS"))

    if st.button("🚀 Run Optimization",type="primary",use_container_width=True,key="btn_opt"):
        with st.spinner("Fetching data…"):
            df_opt=fetch_data(cfg.get("symbol","^NSEI"),opt_tf,opt_period)
        if df_opt is None or df_opt.empty: st.error("No data."); return

        def frange(mn,mx,st_):
            out=[]; v=mn
            while v<=mx+1e-9: out.append(round(v,4)); v=round(v+st_,8)
            return out if out else [mn]

        # Build param grid — only params that vary for the chosen strategy
        param_grid={"sl_points":frange(sl_min,sl_max,sl_step),
                    "target_points":frange(tg_min,tg_max,tg_step)}
        if opt_strat=="EMA Crossover":
            param_grid["fast_ema"]=list(range(fe_min,fe_max+1,fe_step))
            param_grid["slow_ema"]=list(range(se_min,se_max+1,se_step))

        total_c=1
        for v in param_grid.values(): total_c*=len(v)
        st.info(f"Testing {total_c:,} combinations (filtered for valid fast<slow)…")

        pb=st.progress(0); pt_txt=st.empty()
        def pcb(done,total):
            pct=done/total; pb.progress(pct)
            pt_txt.markdown(f"**{done:,}/{total:,}** ({pct*100:.0f}%)")

        base_cfg={**cfg,"strategy":opt_strat,"quantity":opt_qty}
        results=run_opt(df_opt,base_cfg,param_grid,target=opt_metric,min_tr=min_trades,cb=pcb)
        pb.empty(); pt_txt.empty()
        st.session_state.opt_results=results; st.session_state.opt_ran=True
        st.success(f"✅ Done! {len(results)} valid results found.")

    if not st.session_state.opt_ran: return
    results=st.session_state.opt_results
    if not results: st.warning("No valid results. Try wider ranges or fewer min trades."); return

    desired=float(st.session_state.get("opt_acc",60.0))
    filtered=[r for r in results if r.get("Accuracy %",0)>=desired]
    display=filtered if filtered else results
    if filtered: st.success(f"{len(filtered)} results meet {desired:.0f}%+ accuracy")
    else: st.warning(f"No results at {desired:.0f}%. Showing best available.")

    st.markdown(f"### 🏆 Top Results — {len(display)} shown")

    # ── Header row ────────────────────────────────────────────────
    hc=st.columns([3,1,1,1,1,1,1])
    for col,lbl in zip(hc,["Parameters","Accuracy","P&L","Trades","R:R","Score","Action"]):
        col.markdown(f'<div style="color:#90cdf4;font-size:11px;font-weight:700;padding:2px 4px">{lbl}</div>',unsafe_allow_html=True)
    st.markdown('<hr style="margin:4px 0;border-color:#2d3748">',unsafe_allow_html=True)

    for i,res in enumerate(display[:15]):
        # Build parameter string only from keys actually in the result
        parts=[]
        if "fast_ema" in res and res["fast_ema"] is not None:
            parts.append(f"Fast={int(res['fast_ema'])} Slow={int(res['slow_ema'])}")
        sl_v=res.get("sl_points"); tg_v=res.get("target_points")
        if sl_v is not None: parts.append(f"SL={sl_v}")
        if tg_v is not None: parts.append(f"Tgt={tg_v}")
        param_str=" | ".join(parts) if parts else "—"

        acc=float(res.get("Accuracy %",0)); pnl=float(res.get("Total P&L",0))
        rr=float(res.get("R:R",0)); sc=float(res.get("Score",0)); tr=int(res.get("Trades",0))
        ac="#48bb78" if acc>=desired else "#f6ad55"; pc="#48bb78" if pnl>=0 else "#fc8181"

        rc=st.columns([3,1,1,1,1,1,1])
        rc[0].markdown(f'<div style="padding:4px;font-size:12px;color:#e2e8f0">{param_str}</div>',unsafe_allow_html=True)
        rc[1].markdown(f'<div style="padding:4px;text-align:center;font-size:13px;color:{ac};font-weight:700">{acc:.1f}%</div>',unsafe_allow_html=True)
        rc[2].markdown(f'<div style="padding:4px;text-align:center;font-size:13px;color:{pc};font-weight:700">₹{pnl:+.0f}</div>',unsafe_allow_html=True)
        rc[3].markdown(f'<div style="padding:4px;text-align:center;font-size:12px;color:#a0aec0">{tr}</div>',unsafe_allow_html=True)
        rc[4].markdown(f'<div style="padding:4px;text-align:center;font-size:12px;color:#76e4f7">{rr:.2f}</div>',unsafe_allow_html=True)
        rc[5].markdown(f'<div style="padding:4px;text-align:center;font-size:12px;color:#f6ad55">{sc:.1f}</div>',unsafe_allow_html=True)

        if rc[6].button("✅ Apply",key=f"oa_{i}",use_container_width=True):
            # Apply fast/slow EMA if present
            if "fast_ema" in res and res["fast_ema"] is not None:
                fe=int(res["fast_ema"]); se=int(res["slow_ema"])
                st.session_state["_apply_fe"]=fe; st.session_state["_apply_se"]=se
                st.session_state["s_fe"]=fe; st.session_state["s_se"]=se
            # Apply SL / Target
            if sl_v is not None:
                st.session_state["_apply_slp"]=float(sl_v); st.session_state["s_slp"]=float(sl_v)
            if tg_v is not None:
                st.session_state["_apply_tgp"]=float(tg_v); st.session_state["s_tgp2"]=float(tg_v)
            applied=param_str; st.success(f"✅ Applied: {applied} — reload sidebar to confirm.")
            st.rerun()

    # ── Full results table ─────────────────────────────────────────
    st.markdown("### 📋 Full Results Table")
    rdf=pd.DataFrame(display)
    # Only include columns that exist in the dataframe and have at least one non-null value
    candidate_cols=["fast_ema","slow_ema","sl_points","target_points","Trades","Accuracy %","Total P&L","Avg Win","Avg Loss","R:R","Score"]
    show_cols=[c for c in candidate_cols if c in rdf.columns and rdf[c].notna().any()]
    if show_cols:
        # Convert numeric columns cleanly
        for c in ["fast_ema","slow_ema","Trades"]:
            if c in rdf.columns: rdf[c]=rdf[c].apply(lambda x: int(x) if pd.notna(x) else x)
        st.dataframe(style_df(rdf[show_cols]),use_container_width=True,height=380)
    if not rdf.empty:
        st.download_button("⬇ CSV",rdf.to_csv(index=False).encode(),"opt.csv","text/csv",key="opt_dl")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    st.markdown('<div class="app-hdr"><div><h1>📈 Smart Investing</h1><p>Professional Algorithmic Trading Platform · NSE · BSE · Crypto · Commodities</p></div><div style="color:#90cdf4;font-size:12px;text-align:right">EMA · Elliott Wave · Optimization · Dhan Broker</div></div>',unsafe_allow_html=True)
    cfg=sidebar()
    tab1,tab2,tab3,tab4=st.tabs(["🔬 Backtesting","🚀 Live Trading","📚 Trade History","🔧 Optimization"])
    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history(cfg)
    with tab4: tab_optimize(cfg)

if __name__=="__main__":
    main()
