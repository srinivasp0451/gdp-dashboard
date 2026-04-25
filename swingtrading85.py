"""
Smart Investing – Professional Algorithmic Trading Platform v3.0
Multi-strategy Backtesting & Live Trading | Elliott Wave | Dhan Broker
Run: streamlit run smart_investing.py
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading, time, datetime, requests, warnings, re, json
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo
warnings.filterwarnings("ignore")

# ── Timezone ────────────────────────────────────────────────────────────────
_IST = zoneinfo.ZoneInfo("Asia/Kolkata")

def now_ist():
    return datetime.datetime.now(_IST).strftime("%H:%M:%S IST")

def now_ist_full():
    return datetime.datetime.now(_IST).strftime("%d-%b-%Y %H:%M:%S IST")

def ts_to_ist(ts):
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert(_IST).strftime("%d-%b-%Y %H:%M IST")
    except Exception:
        return str(ts)

# ── Constants ────────────────────────────────────────────────────────────────
TICKERS = {
    "Nifty 50":"^NSEI","BankNifty":"^NSEBANK","Sensex":"^BSESN",
    "BTC/USD":"BTC-USD","ETH/USD":"ETH-USD","Gold":"GC=F","Silver":"SI=F","Custom":"__CUSTOM__",
}
TF_PERIODS = {
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo"],"15m":["1d","5d","7d","1mo"],
    "1h":["1d","5d","7d","1mo","3mo","6mo","1y","2y"],
    "1d":["5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
    "1wk":["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"],
}
WARMUP = {"1m":"7d","5m":"1mo","15m":"1mo","1h":"6mo","1d":"5y","1wk":"10y"}
STRATEGIES   = ["EMA Crossover","Anticipatory EMA","Elliott Wave","Simple Buy","Simple Sell"]
SL_TYPES     = ["Custom Points","ATR Based","Risk Reward Based","Trailing SL","Auto SL",
                "EMA Reverse Crossover","Swing Low/High","Candle Low/High",
                "Support / Resistance","Volatility Based"]
TGT_TYPES    = ["Custom Points","ATR Based","Risk Reward Based","Trailing Target (Display Only)",
                "Auto Target","EMA Reverse Crossover","Swing High/Low","Candle High/Low",
                "Support / Resistance","Volatility Based"]
CX_TYPES     = ["Simple Crossover","Custom Candle Size","ATR Based Candle Size"]
TRADE_DIRS   = ["Both","Long Only","Short Only"]

# ── Thread-safe state ────────────────────────────────────────────────────────
_TS_LOCK = threading.Lock()
_TS: dict = {}

def _ts_get(k, d=None):
    with _TS_LOCK: return _TS.get(k, d)

def _ts_set(k, v):
    with _TS_LOCK: _TS[k] = v

def _ts_append(k, v):
    with _TS_LOCK:
        if k not in _TS: _TS[k] = []
        _TS[k].append(v)

_YF_LOCK = threading.Lock()
_LAST_YF = 0.0

def init_ss():
    for k,v in {"live_running":False,"stop_event":None,"live_thread":None,
                "backtest_results":None,"opt_results":None}.items():
        if k not in st.session_state: st.session_state[k] = v

# ── Data helpers ──────────────────────────────────────────────────────────────
def _mkt_filter(df, interval):
    if interval in ("1d","1wk") or df is None or len(df)==0: return df
    try:
        idx = df.index
        if hasattr(idx,"tz") and idx.tz is None: idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(_IST)
        h,m = idx.hour, idx.minute
        mask = ((h>9)|((h==9)&(m>=15))) & ((h<15)|((h==15)&(m<=30)))
        out = df[mask]; return out if len(out)>0 else df
    except Exception: return df

def _yf_dl(ticker, period, interval):
    global _LAST_YF
    with _YF_LOCK:
        g = time.time()-_LAST_YF
        if g<1.5: time.sleep(1.5-g)
        _LAST_YF = time.time()
    try:
        df = yf.download(ticker,period=period,interval=interval,
                         auto_adjust=True,progress=False,threads=False)
        if df is None or len(df)==0: return None
        if isinstance(df.columns,pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        return _mkt_filter(df,interval)
    except Exception: return None

def fetch_data(ticker, period, interval, min_bars=200):
    df = _yf_dl(ticker,period,interval)
    if df is None: return None
    if len(df)>=min_bars: return df
    wup = WARMUP.get(interval,"1y")
    if wup==period: return df
    dfw = _yf_dl(ticker,wup,interval)
    if dfw is None or len(dfw)<=len(df): return df
    try:
        loc = dfw.index.get_indexer([df.index[0]],method="nearest")[0]
        prefix = dfw.iloc[max(0,loc-min_bars):loc]
        combined = pd.concat([prefix,df])
        return combined[~combined.index.duplicated(keep="last")].sort_index()
    except Exception: return df

# ── Indicators ────────────────────────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p,adjust=False,min_periods=1).mean()

def calc_atr(df, p=14):
    h,l,c = df["High"],df["Low"],df["Close"]; pc=c.shift(1)
    return pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)\
             .ewm(span=p,adjust=False,min_periods=1).mean()

def cx_angle(fe,se,i):
    if i<1: return 0.0
    return float(np.degrees(np.arctan(abs((fe.iloc[i]-fe.iloc[i-1])-(se.iloc[i]-se.iloc[i-1])))))

def swing_pivots(df,left=5,right=5):
    n,h,l = len(df),df["High"].values,df["Low"].values
    sh,sl=[],[]
    for i in range(left,n-right):
        if all(h[i]>=h[i-j] for j in range(1,left+1)) and all(h[i]>=h[i+j] for j in range(1,right+1)):
            sh.append((i,df.index[i],float(h[i])))
        if all(l[i]<=l[i-j] for j in range(1,left+1)) and all(l[i]<=l[i+j] for j in range(1,right+1)):
            sl.append((i,df.index[i],float(l[i])))
    return sh,sl

# ── Elliott Wave ──────────────────────────────────────────────────────────────
_EW_EMPTY = {"pattern":"Insufficient Data","wave_direction":None,"completed_waves":[],
             "current_wave":"Analyzing…","wave_labels":[],"wave_points":{},"fibonacci_levels":{},
             "next_target":None,"signal":"NONE","entry":None,"sl":None,"target":None,"pivots":[]}

def _clean_piv(raw,min_pct=0.5):
    if not raw: return []
    out=[raw[0]]
    for p in raw[1:]:
        last=out[-1]
        if p["type"]==last["type"]:
            if p["type"]=="H" and p["price"]>last["price"]: out[-1]=p
            elif p["type"]=="L" and p["price"]<last["price"]: out[-1]=p
        else:
            if abs(p["price"]-last["price"])/max(last["price"],1e-9)*100>=min_pct:
                out.append(p)
    return out

def _fib(base,peak):
    rng=peak-base
    return {"23.6%":round(peak-rng*0.236,2),"38.2%":round(peak-rng*0.382,2),
            "50.0%":round(peak-rng*0.500,2),"61.8%":round(peak-rng*0.618,2)}

def _ew_labels(sub,n=None):
    n = n or len(sub)
    return [{"label":str(i),"idx":sub[i]["idx"],"price":sub[i]["price"],"dt":sub[i]["dt"]} for i in range(n)]

def _imp_up(sub):
    if [p["type"] for p in sub]!=["L","H","L","H","L","H"]: return None
    p0,p1,p2,p3,p4,p5=[p["price"] for p in sub]
    if p2<=p0 or p3<=p1 or p4<=p1: return None
    w1,w3,w5=p1-p0,p3-p2,p5-p4
    if w3<w1 and w3<w5: return None
    tgt=round(p5-(p5-p0)*0.618,2)
    return {"wave_direction":"Bullish","pattern":"5-Wave Impulse (Bullish)",
            "completed_waves":["W1","W2","W3","W4","W5"],"current_wave":"ABC Correction Expected",
            "wave_points":{"W0":p0,"W1":p1,"W2":p2,"W3":p3,"W4":p4,"W5":p5},
            "wave_labels":_ew_labels(sub,6),
            "fibonacci_levels":{**_fib(p0,p5),"A-tgt(38.2%)":round(p5-(p5-p0)*0.382,2),"C-tgt(61.8%)":tgt},
            "next_target":tgt,"signal":"SELL","entry":p5,"sl":round(p5+w1*0.5,2),"target":tgt}

def _imp_dn(sub):
    if [p["type"] for p in sub]!=["H","L","H","L","H","L"]: return None
    p0,p1,p2,p3,p4,p5=[p["price"] for p in sub]
    if p2>=p0 or p3>=p1 or p4>=p1: return None
    w1,w3,w5=p0-p1,p2-p3,p4-p5
    if w3<w1 and w3<w5: return None
    tgt=round(p5+(p0-p5)*0.618,2)
    return {"wave_direction":"Bearish","pattern":"5-Wave Impulse (Bearish)",
            "completed_waves":["W1","W2","W3","W4","W5"],"current_wave":"ABC Correction Expected (Up)",
            "wave_points":{"W0":p0,"W1":p1,"W2":p2,"W3":p3,"W4":p4,"W5":p5},
            "wave_labels":_ew_labels(sub,6),
            "fibonacci_levels":{**_fib(p5,p0),"A-tgt(38.2%)":round(p5+(p0-p5)*0.382,2),"C-tgt(61.8%)":tgt},
            "next_target":tgt,"signal":"BUY","entry":p5,"sl":round(p5-w1*0.5,2),"target":tgt}

def _inprog(sub,df):
    if len(sub)<3: return None
    t=[p["type"] for p in sub[:3]]; lc=float(df["Close"].iloc[-1])
    if t==["L","H","L"]:
        p0,p1,p2=sub[0]["price"],sub[1]["price"],sub[2]["price"]
        if p2<=p0: return None
        w1=p1-p0; w3t=round(p2+w1*1.618,2)
        lbs=_ew_labels(sub,3)
        if len(sub)>=4 and sub[3]["type"]=="H":
            lbs.append({"label":"3?","idx":sub[3]["idx"],"price":sub[3]["price"],"dt":sub[3]["dt"]})
        return {"wave_direction":"Bullish","pattern":"In-Progress W1&W2 Done",
                "completed_waves":["W1","W2"],"current_wave":"W3 In Progress" if lc>p2 else "W2 Bottom – Buy Zone",
                "wave_points":{"W0":p0,"W1":p1,"W2(buy)":p2},"wave_labels":lbs,
                "fibonacci_levels":{"W3(1.618xW1)":w3t,"W3(2.618xW1)":round(p2+w1*2.618,2)},
                "next_target":w3t,"signal":"BUY","entry":p2,"sl":round(p0-w1*0.1,2),"target":w3t}
    if t==["H","L","H"]:
        p0,p1,p2=sub[0]["price"],sub[1]["price"],sub[2]["price"]
        if p2>=p0: return None
        w1=p0-p1; w3t=round(p2-w1*1.618,2)
        return {"wave_direction":"Bearish","pattern":"In-Progress W1&W2 Done (Bear)",
                "completed_waves":["W1","W2"],"current_wave":"W3 In Progress – Short Zone",
                "wave_points":{"W0":p0,"W1":p1,"W2(sell)":p2},"wave_labels":_ew_labels(sub,3),
                "fibonacci_levels":{"W3(1.618xW1)":w3t,"W3(2.618xW1)":round(p2-w1*2.618,2)},
                "next_target":w3t,"signal":"SELL","entry":p2,"sl":round(p0+w1*0.1,2),"target":w3t}
    return None

def _abc(sub):
    if len(sub)<3: return None
    if sub[0]["type"]=="L" and sub[1]["type"]=="H" and sub[2]["type"]=="L":
        a_s,a_e,b_e=sub[0]["price"],sub[1]["price"],sub[2]["price"]
        wa=a_e-a_s; wb=a_e-b_e
        if wa<=0 or not(0.382<=wb/wa<=0.886): return None
        tgt=round(b_e+wa,2)
        return {"wave_direction":"Bullish ABC","pattern":"ABC Corrective (C Up)",
                "completed_waves":["A","B"],"current_wave":"C In Progress (Up)",
                "wave_points":{"A_start":a_s,"A_end":a_e,"B_end(buy)":b_e},
                "wave_labels":[{"label":"A","idx":sub[1]["idx"],"price":a_e,"dt":sub[1]["dt"]},
                                {"label":"B","idx":sub[2]["idx"],"price":b_e,"dt":sub[2]["dt"]}],
                "fibonacci_levels":{"C=A":tgt,"C=1.618A":round(b_e+wa*1.618,2)},
                "next_target":tgt,"signal":"BUY","entry":b_e,"sl":round(a_s-wa*0.1,2),"target":tgt}
    if sub[0]["type"]=="H" and sub[1]["type"]=="L" and sub[2]["type"]=="H":
        a_s,a_e,b_e=sub[0]["price"],sub[1]["price"],sub[2]["price"]
        wa=a_s-a_e; wb=b_e-a_e
        if wa<=0 or not(0.382<=wb/wa<=0.886): return None
        tgt=round(b_e-wa,2)
        return {"wave_direction":"Bearish ABC","pattern":"ABC Corrective (C Down)",
                "completed_waves":["A","B"],"current_wave":"C In Progress (Down)",
                "wave_points":{"A_start":a_s,"A_end":a_e,"B_end(sell)":b_e},
                "wave_labels":[{"label":"A","idx":sub[1]["idx"],"price":a_e,"dt":sub[1]["dt"]},
                                {"label":"B","idx":sub[2]["idx"],"price":b_e,"dt":sub[2]["dt"]}],
                "fibonacci_levels":{"C=A":tgt,"C=1.618A":round(b_e-wa*1.618,2)},
                "next_target":tgt,"signal":"SELL","entry":b_e,"sl":round(a_s+wa*0.1,2),"target":tgt}
    return None

def detect_ew(df,min_pct=0.5,left=4,right=4):
    if len(df)<(left+right+1)*2: return _EW_EMPTY.copy()
    sh,sl_p=swing_pivots(df,left,right)
    raw=[{"idx":i,"dt":d,"price":p,"type":"H"} for i,d,p in sh]+\
        [{"idx":i,"dt":d,"price":p,"type":"L"} for i,d,p in sl_p]
    raw.sort(key=lambda x:x["idx"])
    pivots=_clean_piv(raw,min_pct)
    if len(pivots)<4: return {**_EW_EMPTY.copy(),"pivots":pivots}
    for s in range(max(0,len(pivots)-10),len(pivots)-5):
        sub=pivots[s:s+6]
        if len(sub)<6: continue
        for fn in [_imp_up,_imp_dn]:
            r=fn(sub)
            if r: return {**r,"pivots":pivots}
    for s in range(max(0,len(pivots)-8),len(pivots)-3):
        r=_inprog(pivots[s:],df)
        if r: return {**r,"pivots":pivots}
    for s in range(max(0,len(pivots)-6),len(pivots)-3):
        r=_abc(pivots[s:])
        if r: return {**r,"pivots":pivots}
    return {**_EW_EMPTY.copy(),"pattern":"No Clear Pattern","pivots":pivots}

# ── Signal Generation ─────────────────────────────────────────────────────────
def generate_signals(df, cfg):
    n=len(df); sigs=pd.Series(0,index=df.index,dtype=int)
    reasons=pd.Series("",index=df.index,dtype=str)
    strat=cfg.get("strategy","EMA Crossover"); inds={}
    tdir=cfg.get("trade_direction","Both")

    if strat=="Simple Buy":
        if tdir in ("Both","Long Only"): sigs[:]=1; reasons[:] = "Simple Buy – Immediate Entry"
        return sigs,reasons,inds
    if strat=="Simple Sell":
        if tdir in ("Both","Short Only"): sigs[:] = -1; reasons[:] = "Simple Sell – Immediate Entry"
        return sigs,reasons,inds

    fp=cfg.get("fast_ema",9); sp=cfg.get("slow_ema",15)
    fe=ema(df["Close"],fp); se=ema(df["Close"],sp)
    atr_s=calc_atr(df)
    inds["fast_ema"]=fe; inds["slow_ema"]=se; inds["atr"]=atr_s

    if strat=="EMA Crossover":
        min_a=cfg.get("min_angle",0.0); max_a=cfg.get("max_angle",90.0)
        min_d=cfg.get("min_delta",0.0); max_d=cfg.get("max_delta",1e9)
        cxt=cfg.get("crossover_type","Simple Crossover")
        cxsz=cfg.get("custom_candle_size",10)
        for i in range(1,n):
            f0,f1=float(fe.iloc[i]),float(fe.iloc[i-1])
            s0,s1=float(se.iloc[i]),float(se.iloc[i-1])
            ang=cx_angle(fe,se,i); delta=abs(f0-s0)
            csz=abs(float(df["Close"].iloc[i])-float(df["Open"].iloc[i]))
            if ang<min_a or ang>max_a: continue
            if delta<min_d or delta>max_d: continue
            if cxt=="Custom Candle Size" and csz<cxsz: continue
            if cxt=="ATR Based Candle Size" and csz<float(atr_s.iloc[i]): continue
            if tdir in ("Both","Long Only") and f1<=s1 and f0>s0:
                sigs.iloc[i]=1
                reasons.iloc[i]=f"EMA Bull Cross | F={f0:.2f} S={s0:.2f} Δ={delta:.2f} ∠={ang:.1f}°"
            elif tdir in ("Both","Short Only") and f1>=s1 and f0<s0:
                sigs.iloc[i]=-1
                reasons.iloc[i]=f"EMA Bear Cross | F={f0:.2f} S={s0:.2f} Δ={delta:.2f} ∠={ang:.1f}°"

    elif strat=="Anticipatory EMA":
        for i in range(3,n):
            f0,f1,f2=float(fe.iloc[i]),float(fe.iloc[i-1]),float(fe.iloc[i-2])
            s0,s1,s2=float(se.iloc[i]),float(se.iloc[i-1]),float(se.iloc[i-2])
            g_now=f0-s0; g_prev=f1-s1; g2=f2-s2; av=float(atr_s.iloc[i])
            if g2<0 and g_prev<0 and g_now<0:
                shrink=abs(g2)-abs(g_now)
                if shrink>0 and abs(g_now)<av*1.2 and (f0-f1)-(f1-f2)>0:
                    if tdir in ("Both","Long Only"):
                        sigs.iloc[i]=1; reasons.iloc[i]=f"Anticipatory BUY | Gap={g_now:.2f}"
                        continue
            if g2>0 and g_prev>0 and g_now>0:
                shrink=abs(g2)-abs(g_now)
                if shrink>0 and abs(g_now)<av*1.2 and (f0-f1)-(f1-f2)<0:
                    if tdir in ("Both","Short Only"):
                        sigs.iloc[i]=-1; reasons.iloc[i]=f"Anticipatory SELL | Gap={g_now:.2f}"

    elif strat=="Elliott Wave":
        mwp=cfg.get("min_wave_pct",0.5)
        ewl=cfg.get("ew_left",4); ewr=cfg.get("ew_right",4)
        min_bars=(ewl+ewr)*4; step=max(5,min_bars//4); last_bar=-9999
        for scan_end in range(min_bars,n,step):
            ew=detect_ew(df.iloc[:scan_end],mwp,ewl,ewr)
            sig_ew=ew.get("signal","NONE")
            if sig_ew not in ("BUY","SELL") or not ew.get("wave_labels"): continue
            anchor=min(ew["wave_labels"][-1].get("idx",scan_end-1),n-1)
            if anchor-last_bar<min_bars: continue
            if sig_ew=="BUY" and tdir not in ("Both","Long Only"): continue
            if sig_ew=="SELL" and tdir not in ("Both","Short Only"): continue
            sv=1 if sig_ew=="BUY" else -1
            sigs.iloc[anchor]=sv
            reasons.iloc[anchor]=f"EW|{ew.get('pattern','')}|{ew.get('current_wave','')}"
            last_bar=anchor
        inds["elliott"]=detect_ew(df,mwp,ewl,ewr)

    return sigs,reasons,inds

# ── SL / Target calculators ──────────────────────────────────────────────────
def calc_sl(entry, sig, cfg, df, idx):
    sl_t=cfg.get("sl_type","Custom Points")
    av=float(calc_atr(df).iloc[min(idx,len(df)-1)])
    pts=cfg.get("sl_points",10); sign=1 if sig==1 else -1
    if sl_t=="Custom Points":          return entry-sign*pts
    if sl_t=="ATR Based":              return entry-sign*av*cfg.get("atr_sl_mult",1.5)
    if sl_t=="Risk Reward Based":      return entry-sign*cfg.get("target_points",20)/max(cfg.get("risk_reward",2.0),0.1)
    if sl_t=="Volatility Based":
        vol=df["Close"].pct_change().std()*float(df["Close"].iloc[min(idx,len(df)-1)])
        return entry-sign*vol*cfg.get("vol_sl_mult",2.0)
    if sl_t in ("Swing Low/High","Support / Resistance"):
        w=cfg.get("swing_window",20); s=max(0,idx-w)
        return (float(df["Low"].iloc[s:idx+1].min())-av*0.3) if sig==1 \
               else (float(df["High"].iloc[s:idx+1].max())+av*0.3)
    if sl_t=="Candle Low/High":
        return (float(df["Low"].iloc[min(idx,len(df)-1)])-av*0.1) if sig==1 \
               else (float(df["High"].iloc[min(idx,len(df)-1)])+av*0.1)
    if sl_t in ("Auto SL","Trailing SL"):
        w=5; s=max(0,idx-w)
        return (min(float(df["Low"].iloc[s:idx+1].min()),entry-av*1.5)) if sig==1 \
               else (max(float(df["High"].iloc[s:idx+1].max()),entry+av*1.5))
    return entry-sign*pts  # EMA Reverse Crossover uses custom as initial

def calc_tgt(entry, sig, cfg, df, idx):
    tgt_t=cfg.get("target_type","Custom Points")
    av=float(calc_atr(df).iloc[min(idx,len(df)-1)])
    pts=cfg.get("target_points",20); sign=1 if sig==1 else -1
    if tgt_t in ("Custom Points","Trailing Target (Display Only)"): return entry+sign*pts
    if tgt_t=="ATR Based":              return entry+sign*av*cfg.get("atr_target_mult",2.0)
    if tgt_t=="Risk Reward Based":
        sl_dist=abs(entry-calc_sl(entry,sig,cfg,df,idx))
        return entry+sign*sl_dist*cfg.get("risk_reward",2.0)
    if tgt_t=="Volatility Based":
        vol=df["Close"].pct_change().std()*float(df["Close"].iloc[min(idx,len(df)-1)])
        return entry+sign*vol*cfg.get("vol_target_mult",3.0)
    if tgt_t in ("Swing High/Low","Support / Resistance"):
        w=cfg.get("swing_window",20); s=max(0,idx-w)
        return (max(float(df["High"].iloc[s:idx+1].max()),entry+av)) if sig==1 \
               else (min(float(df["Low"].iloc[s:idx+1].min()),entry-av))
    if tgt_t=="Candle High/Low":
        return (float(df["High"].iloc[min(idx,len(df)-1)])+av*0.1) if sig==1 \
               else (float(df["Low"].iloc[min(idx,len(df)-1)])-av*0.1)
    if tgt_t in ("Auto Target","EMA Reverse Crossover"):
        w=10; s=max(0,idx-w)
        return (max(float(df["High"].iloc[s:idx+1].max()),entry+av*2.5)) if sig==1 \
               else (min(float(df["Low"].iloc[s:idx+1].min()),entry-av*2.5))
    return entry+sign*pts

# ── Trailing SL ratchet (NEVER moves against the trade) ──────────────────────
def ratchet_sl(pos, ltp, cfg, df, bar_idx):
    sl_t=cfg.get("sl_type","Custom Points"); sig=pos["signal_type"]; cur=pos["current_sl"]
    av=float(calc_atr(df).iloc[min(bar_idx,len(df)-1)])
    if sl_t=="Trailing SL":
        trail=cfg.get("sl_points",10)
        new_sl=ltp-trail if sig==1 else ltp+trail
        if sig==1 and new_sl>cur:  pos["current_sl"]=round(new_sl,2)
        if sig==-1 and new_sl<cur: pos["current_sl"]=round(new_sl,2)
    elif sl_t=="Swing Low/High":
        w=cfg.get("swing_window",20); s=max(0,bar_idx-w)
        if sig==1:
            new_sl=float(df["Low"].iloc[s:bar_idx+1].min())-av*0.3
            if new_sl>cur: pos["current_sl"]=round(new_sl,2)
        else:
            new_sl=float(df["High"].iloc[s:bar_idx+1].max())+av*0.3
            if new_sl<cur: pos["current_sl"]=round(new_sl,2)
    elif sl_t=="Candle Low/High":
        if sig==1:
            new_sl=float(df["Low"].iloc[min(bar_idx,len(df)-1)])-av*0.1
            if new_sl>cur: pos["current_sl"]=round(new_sl,2)
        else:
            new_sl=float(df["High"].iloc[min(bar_idx,len(df)-1)])+av*0.1
            if new_sl<cur: pos["current_sl"]=round(new_sl,2)
    return pos

# Conservative exit: SL checked FIRST, then target
def check_exit_bt(pos, row, cfg):
    sig=pos["signal_type"]; sl=pos["current_sl"]; tgt=pos["target"]
    tgt_t=cfg.get("target_type","Custom Points")
    hi=float(row["High"]); lo=float(row["Low"])
    if sig==1:
        sl_hit=lo<=sl
        tgt_hit=hi>=tgt and tgt_t!="Trailing Target (Display Only)"
        if sl_hit and tgt_hit: return sl,"SL Hit (violation–both same candle–SL first)",True
        if sl_hit:  return sl,"SL Hit",False
        if tgt_hit: return tgt,"Target Hit",False
    else:
        sl_hit=hi>=sl
        tgt_hit=lo<=tgt and tgt_t!="Trailing Target (Display Only)"
        if sl_hit and tgt_hit: return sl,"SL Hit (violation–both same candle–SL first)",True
        if sl_hit:  return sl,"SL Hit",False
        if tgt_hit: return tgt,"Target Hit",False
    return None,None,False

# ── Backtest Engine ───────────────────────────────────────────────────────────
def _mk_trade(pos,row,exit_dt,xp,xr,qty,is_v):
    sig=pos["signal_type"]
    pnl=((xp-pos["entry_price"]) if sig==1 else (pos["entry_price"]-xp))*qty
    return {"Entry DateTime":ts_to_ist(pos["entry_dt"]),"Exit DateTime":ts_to_ist(exit_dt),
            "Signal":"BUY" if sig==1 else "SELL","Entry Price":round(pos["entry_price"],2),
            "Exit Price":round(xp,2),"SL":round(pos["initial_sl"],2),
            "Final SL":round(pos["current_sl"],2),"Target":round(pos["target"],2),
            "Candle High":round(float(row["High"]),2),"Candle Low":round(float(row["Low"]),2),
            "Entry Reason":pos.get("signal_reason",""),"Exit Reason":xr,
            "PnL":round(pnl,2),"Is Violation":is_v,"Mode":"Backtest"}

def run_backtest(df_full, cfg, progress_cb=None):
    strat=cfg.get("strategy","EMA Crossover"); qty=cfg.get("quantity",1)
    immediate=strat in ("Simple Buy","Simple Sell")
    sigs,reasons,inds=generate_signals(df_full,cfg)
    fe=inds.get("fast_ema",pd.Series(dtype=float))
    se=inds.get("slow_ema",pd.Series(dtype=float))
    trades=[]; violations=[]; pos=None; n=len(df_full); i=0
    while i<n:
        row=df_full.iloc[i]
        if pos is not None:
            pos=ratchet_sl(pos,float(row["Close"]),cfg,df_full,i)
            # EMA Reverse Crossover exit
            if cfg.get("sl_type")=="EMA Reverse Crossover" or cfg.get("target_type")=="EMA Reverse Crossover":
                if len(fe)>i and len(se)>i and i>0:
                    f0,f1=float(fe.iloc[i]),float(fe.iloc[i-1])
                    s0,s1=float(se.iloc[i]),float(se.iloc[i-1])
                    if pos["signal_type"]==1 and f1>=s1 and f0<s0:
                        trades.append(_mk_trade(pos,row,df_full.index[i],float(row["Open"]),"EMA Rev X Exit",qty,False))
                        pos=None; i+=1; continue
                    if pos["signal_type"]==-1 and f1<=s1 and f0>s0:
                        trades.append(_mk_trade(pos,row,df_full.index[i],float(row["Open"]),"EMA Rev X Exit",qty,False))
                        pos=None; i+=1; continue
            xp,xr,is_v=check_exit_bt(pos,row,cfg)
            if xp is not None:
                t=_mk_trade(pos,row,df_full.index[i],xp,xr,qty,is_v)
                trades.append(t)
                if is_v: violations.append(t)
                pos=None
        if pos is None and int(sigs.iloc[i])!=0:
            sv=int(sigs.iloc[i]); reason=str(reasons.iloc[i])
            if immediate:
                eidx=i; edt=df_full.index[i]; ep=float(df_full.iloc[i]["Open"])
            else:
                if i+1>=n: i+=1; continue
                eidx=i+1; edt=df_full.index[eidx]; ep=float(df_full.iloc[eidx]["Open"])
            sl_p=calc_sl(ep,sv,cfg,df_full,eidx)
            tgt_p=calc_tgt(ep,sv,cfg,df_full,eidx)
            if strat=="Elliott Wave":
                ew_local=detect_ew(df_full.iloc[:eidx+1],cfg.get("min_wave_pct",0.5),
                                   cfg.get("ew_left",4),cfg.get("ew_right",4))
                if ew_local.get("sl") is not None:     sl_p=float(ew_local["sl"])
                if ew_local.get("target") is not None: tgt_p=float(ew_local["target"])
            pos={"signal_idx":i,"entry_idx":eidx,"entry_dt":edt,"entry_price":ep,
                 "signal_type":sv,"initial_sl":sl_p,"current_sl":sl_p,
                 "target":tgt_p,"signal_reason":reason}
            if not immediate: i=eidx
        if progress_cb: progress_cb(i/n)
        i+=1
    if pos is not None:
        lr=df_full.iloc[-1]
        trades.append(_mk_trade(pos,lr,df_full.index[-1],float(lr["Close"]),"End of Data",qty,False))
    return trades,violations,inds

def calc_acc(trades):
    if not trades: return 0,0,0.0
    wins=sum(1 for t in trades if t["PnL"]>0)
    return wins,len(trades),round(wins/len(trades)*100,2)

# ── Dhan Broker ───────────────────────────────────────────────────────────────
def get_ip():
    try: return requests.get("https://api.ipify.org?format=json",timeout=5).json()["ip"]
    except: return "Unknown"

def _dhan(cfg):
    try:
        from dhanhq import dhanhq
        return dhanhq(cfg["client_id"],cfg["access_token"])
    except: return None

def _place(cfg,tx,seg,prod,ot,sec,qty,price=0):
    d=_dhan(cfg)
    if d is None: return {"error":"dhanhq not installed or bad creds"}
    try:
        return d.place_order(transactionType=tx,exchangeSegment=seg,productType=prod,
                             orderType=ot,validity="DAY",securityId=str(sec),
                             quantity=int(qty),price=float(price),triggerPrice=0)
    except Exception as e: return {"error":str(e)}

def place_equity_entry(cfg,sig,ltp=0):
    tx="BUY" if sig==1 else "SELL"
    seg={"NSE":"NSE_EQ","BSE":"BSE_EQ"}.get(cfg.get("exchange","NSE"),"NSE_EQ")
    ot=cfg.get("entry_order_type","MARKET")
    return _place(cfg,tx,seg,cfg.get("product_type","INTRADAY"),ot,
                  cfg.get("security_id","1594"),cfg.get("quantity",1),ltp if ot=="LIMIT" else 0)

def place_options_entry(cfg,sig,ltp=0):
    sec=cfg.get("ce_security_id","57749") if sig==1 else cfg.get("pe_security_id","57716")
    ot=cfg.get("options_entry_order_type","MARKET")
    return _place(cfg,"BUY",cfg.get("options_exchange","NSE_FNO"),"INTRADAY",ot,sec,
                  cfg.get("options_quantity",65),ltp if ot=="LIMIT" else 0)

def place_exit(cfg,pos,ltp=0,is_opt=False):
    sig=pos.get("signal_type",1)
    if is_opt:
        sec=cfg.get("ce_security_id","57749") if sig==1 else cfg.get("pe_security_id","57716")
        ot=cfg.get("options_exit_order_type","MARKET")
        return _place(cfg,"SELL",cfg.get("options_exchange","NSE_FNO"),"INTRADAY",ot,sec,
                      cfg.get("options_quantity",65),ltp if ot=="LIMIT" else 0)
    tx="SELL" if sig==1 else "BUY"
    seg={"NSE":"NSE_EQ","BSE":"BSE_EQ"}.get(cfg.get("exchange","NSE"),"NSE_EQ")
    ot=cfg.get("exit_order_type","MARKET")
    return _place(cfg,tx,seg,cfg.get("product_type","INTRADAY"),ot,
                  cfg.get("security_id","1594"),cfg.get("quantity",1),ltp if ot=="LIMIT" else 0)

# ── Live Trading Thread ───────────────────────────────────────────────────────
def _bar_boundary(iv):
    mins={"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"1d":1440,"1wk":10080}.get(iv,5)
    if mins>=1440: return True
    n=datetime.datetime.now(_IST); return ((n.hour*60+n.minute)%mins==0) and n.second<6

def live_thread(ticker,period,interval,cfg,stop_ev,dhan_cfg):
    global _LAST_YF
    _ts_set("live_running",True); _ts_set("current_position",None)
    _ts_set("live_log",[]); _ts_set("live_data",None)
    _ts_set("last_candle",None); _ts_set("current_pnl",0.0)
    _ts_set("simple_entered",False); _ts_set("day_pnl",0.0)
    def log(msg): _ts_append("live_log",f"[{now_ist()}] {msg}")

    strat=cfg.get("strategy","EMA Crossover"); qty=cfg.get("quantity",1)
    immediate=strat in ("Simple Buy","Simple Sell")
    tgt_t=cfg.get("target_type","Custom Points")
    en_pnl=cfg.get("enable_pnl_cap",False)
    max_loss=cfg.get("max_day_loss",-1000.0); max_prof=cfg.get("max_day_profit",1000.0)
    en_tf=cfg.get("enable_time_filter",False)
    tf_s=str(cfg.get("trade_start_time","09:15")); tf_e=str(cfg.get("trade_end_time","15:20"))
    use_cool=cfg.get("enable_cooldown",True); cooldown=cfg.get("cooldown",5)

    def in_window():
        if not en_tf: return True
        now_t=datetime.datetime.now(_IST).time()
        try:
            sh,sm=int(tf_s.split(":")[0]),int(tf_s.split(":")[1][:2])
            eh,em=int(tf_e.split(":")[0]),int(tf_e.split(":")[1][:2])
            return datetime.time(sh,sm)<=now_t<=datetime.time(eh,em)
        except: return True

    def day_ok():
        if not en_pnl: return True
        dp=_ts_get("day_pnl",0.0)
        return max_loss<dp<max_prof

    last_exit=0.0; last_bar_dt=None; cached_df=None; last_fetch=0.0
    log(f"▶ Started | {ticker} | {interval}/{period} | {strat}")

    # Immediate entry for Simple Buy/Sell
    if immediate:
        log(f"⚡ {strat} – fetching price for immediate entry…")
        try:
            with _YF_LOCK:
                g=time.time()-_LAST_YF
                if g<1.5: time.sleep(1.5-g)
                _LAST_YF=time.time()
            _df=yf.download(ticker,period="1d",interval="1m",auto_adjust=True,progress=False,threads=False)
            if _df is not None and len(_df)>0:
                if isinstance(_df.columns,pd.MultiIndex): _df.columns=_df.columns.get_level_values(0)
                _ltp=float(_df["Close"].iloc[-1])
                _sv=1 if strat=="Simple Buy" else -1
                _sl=_ltp-_sv*cfg.get("sl_points",10)
                _tg=_ltp+_sv*cfg.get("target_points",20)
                _pos={"entry_dt":now_ist_full(),"entry_price":_ltp,"signal_type":_sv,
                      "initial_sl":_sl,"current_sl":_sl,"target":_tg,
                      "signal_reason":f"{strat} @ {_ltp:.2f}","sl_refined":False}
                _ts_set("current_position",_pos); _ts_set("simple_entered",True)
                log(f"◆ {'BUY' if _sv==1 else 'SELL'} | EP={_ltp:.2f} SL={_sl:.2f} Tgt={_tg:.2f}")
                if dhan_cfg.get("enabled"):
                    r=place_options_entry(dhan_cfg,_sv,_ltp) if dhan_cfg.get("options_trading") else place_equity_entry(dhan_cfg,_sv,_ltp)
                    log(f"  Dhan: {r}")
        except Exception as e: log(f"[WARN quick fetch] {e}")

    # Main loop
    while True:
        stop_req=stop_ev.is_set(); has_pos=_ts_get("current_position") is not None
        if stop_req and not has_pos: break
        try:
            # Fetch every 2.5s
            if time.time()-last_fetch>=2.5:
                with _YF_LOCK:
                    g=time.time()-_LAST_YF
                    if g<1.5: time.sleep(1.5-g)
                    _LAST_YF=time.time()
                try:
                    df=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
                    if df is not None and len(df)>1:
                        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
                        df=df.dropna(how="all"); df=_mkt_filter(df,interval)
                        cached_df=df; last_fetch=time.time()
                        lc=df.iloc[-1]
                        _ts_set("last_candle",{"datetime":ts_to_ist(df.index[-1]),
                            "open":round(float(lc["Open"]),2),"high":round(float(lc["High"]),2),
                            "low":round(float(lc["Low"]),2),"close":round(float(lc["Close"]),2),
                            "volume":int(lc["Volume"]) if not pd.isna(lc["Volume"]) else 0})
                        _ts_set("live_data",{"open":df["Open"].tolist(),"high":df["High"].tolist(),
                            "low":df["Low"].tolist(),"close":df["Close"].tolist(),
                            "volume":df["Volume"].tolist(),"index":[ts_to_ist(x) for x in df.index]})
                except Exception as e: log(f"[FETCH ERR] {e}"); stop_ev.wait(2.0); continue

            if cached_df is None or len(cached_df)<3: stop_ev.wait(1.0); continue
            df=cached_df; ltp=float(df["Close"].iloc[-1])

            # Refine SL/target for Simple once full df loaded
            pos=_ts_get("current_position")
            if immediate and pos is not None and not pos.get("sl_refined",True) and len(df)>=5:
                sv=pos["signal_type"]; ep=pos["entry_price"]
                nsl=calc_sl(ep,sv,cfg,df,len(df)-1); ntg=calc_tgt(ep,sv,cfg,df,len(df)-1)
                pos["initial_sl"]=nsl; pos["current_sl"]=nsl; pos["target"]=ntg; pos["sl_refined"]=True
                _ts_set("current_position",pos)
                log(f"✅ SL/Tgt refined: SL={nsl:.2f} Tgt={ntg:.2f}")

            # Update live PnL
            pos=_ts_get("current_position")
            if pos is not None:
                pnl_pts=(ltp-pos["entry_price"]) if pos["signal_type"]==1 else (pos["entry_price"]-ltp)
                _ts_set("current_pnl",round(pnl_pts,2))

            # Trailing SL ratchet using LTP
            pos=_ts_get("current_position")
            if pos is not None:
                sl_t=cfg.get("sl_type","Custom Points"); sig=pos["signal_type"]; cur=pos["current_sl"]
                if sl_t=="Trailing SL":
                    trail=cfg.get("sl_points",10)
                    nsl=ltp-trail if sig==1 else ltp+trail
                    if (sig==1 and nsl>cur) or (sig==-1 and nsl<cur):
                        pos["current_sl"]=round(nsl,2); _ts_set("current_position",pos)

            # Conservative SL-first exit vs LTP
            pos=_ts_get("current_position")
            if pos is not None:
                sl=pos["current_sl"]; tgt=pos["target"]; sig=pos["signal_type"]; ep=pos["entry_price"]
                xp=None; xr=None
                # EMA Reverse Crossover
                if cfg.get("sl_type")=="EMA Reverse Crossover" or tgt_t=="EMA Reverse Crossover":
                    fe2=ema(df["Close"],cfg.get("fast_ema",9)); se2=ema(df["Close"],cfg.get("slow_ema",15))
                    if len(fe2)>=2:
                        f0,f1=float(fe2.iloc[-1]),float(fe2.iloc[-2])
                        s0,s1=float(se2.iloc[-1]),float(se2.iloc[-2])
                        if sig==1 and f1>=s1 and f0<s0: xp=ltp; xr="EMA Rev X Exit"
                        elif sig==-1 and f1<=s1 and f0>s0: xp=ltp; xr="EMA Rev X Exit"
                # SL first (conservative)
                if xp is None:
                    if sig==1:
                        if ltp<=sl: xp=ltp; xr="SL Hit"
                        elif tgt_t!="Trailing Target (Display Only)" and ltp>=tgt: xp=ltp; xr="Target Hit"
                    else:
                        if ltp>=sl: xp=ltp; xr="SL Hit"
                        elif tgt_t!="Trailing Target (Display Only)" and ltp<=tgt: xp=ltp; xr="Target Hit"
                if xp is not None:
                    pnl_v=((xp-ep) if sig==1 else (ep-xp))*qty
                    log(f"◼ EXIT {xr} | EP={ep:.2f} XP={xp:.2f} PnL={pnl_v:+.2f}")
                    hist=_ts_get("trade_history",[])
                    hist.append({"Entry DateTime":pos["entry_dt"],"Exit DateTime":now_ist_full(),
                        "Signal":"BUY" if sig==1 else "SELL","Entry Price":round(ep,2),
                        "Exit Price":round(xp,2),"SL":round(pos["initial_sl"],2),
                        "Final SL":round(pos["current_sl"],2),"Target":round(tgt,2),
                        "Entry Reason":pos.get("signal_reason",""),"Exit Reason":xr,
                        "PnL":round(pnl_v,2),"Mode":"Live"})
                    _ts_set("trade_history",hist)
                    _ts_set("day_pnl",round(_ts_get("day_pnl",0.0)+pnl_v,2))
                    if dhan_cfg.get("enabled"):
                        r=place_exit(dhan_cfg,pos,ltp,dhan_cfg.get("options_trading",False))
                        log(f"  Dhan exit: {r}")
                    _ts_set("current_position",None); _ts_set("current_pnl",0.0)
                    last_exit=time.time()
                    if immediate: _ts_set("simple_entered",False)

            # New signal check
            if stop_req: stop_ev.wait(1.0); continue
            pos=_ts_get("current_position")
            if pos is None:
                if not in_window(): stop_ev.wait(2.0); continue
                if not day_ok(): log("⛔ Daily PnL cap hit"); stop_ev.wait(30.0); continue
                if use_cool and (time.time()-last_exit)<cooldown: stop_ev.wait(0.5); continue
                if immediate:
                    if _ts_get("simple_entered",False): stop_ev.wait(1.0); continue
                    sv=1 if strat=="Simple Buy" else -1; ep=ltp
                    nsl=calc_sl(ep,sv,cfg,df,len(df)-1); ntg=calc_tgt(ep,sv,cfg,df,len(df)-1)
                    new_pos={"entry_dt":now_ist_full(),"entry_price":ep,"signal_type":sv,
                             "initial_sl":nsl,"current_sl":nsl,"target":ntg,
                             "signal_reason":f"{strat} @ {ep:.2f}","sl_refined":True}
                    _ts_set("current_position",new_pos); _ts_set("simple_entered",True)
                    log(f"◆ {'BUY' if sv==1 else 'SELL'} | EP={ep:.2f} SL={nsl:.2f} Tgt={ntg:.2f}")
                    if dhan_cfg.get("enabled"):
                        r=place_options_entry(dhan_cfg,sv,ep) if dhan_cfg.get("options_trading") else place_equity_entry(dhan_cfg,sv,ep)
                        log(f"  Dhan: {r}")
                else:
                    if not _bar_boundary(interval): stop_ev.wait(0.5); continue
                    cur_bar=ts_to_ist(df.index[-2]) if len(df)>=2 else ts_to_ist(df.index[-1])
                    if cur_bar==last_bar_dt: stop_ev.wait(0.5); continue
                    _sigs,_reas,_inds=generate_signals(df,cfg)
                    if "elliott" in _inds: _ts_set("elliott_wave_state",_inds["elliott"])
                    sidx=max(len(_sigs)-2,0); sv=int(_sigs.iloc[sidx])
                    if sv!=0:
                        last_bar_dt=cur_bar; ep=ltp
                        nsl=calc_sl(ep,sv,cfg,df,len(df)-1); ntg=calc_tgt(ep,sv,cfg,df,len(df)-1)
                        if cfg.get("strategy")=="Elliott Wave" and "elliott" in _inds:
                            ew=_inds["elliott"]
                            if ew.get("sl"): nsl=float(ew["sl"])
                            if ew.get("target"): ntg=float(ew["target"])
                        reason=str(_reas.iloc[sidx])
                        new_pos={"entry_dt":now_ist_full(),"entry_price":ep,"signal_type":sv,
                                 "initial_sl":nsl,"current_sl":nsl,"target":ntg,
                                 "signal_reason":reason,"sl_refined":True}
                        _ts_set("current_position",new_pos)
                        log(f"◆ {'BUY' if sv==1 else 'SELL'} | EP={ep:.2f} SL={nsl:.2f} Tgt={ntg:.2f}")
                        if dhan_cfg.get("enabled"):
                            r=place_options_entry(dhan_cfg,sv,ep) if dhan_cfg.get("options_trading") else place_equity_entry(dhan_cfg,sv,ep)
                            log(f"  Dhan: {r}")
        except Exception as e: log(f"[ERR] {e}")
        stop_ev.wait(1.5)
    _ts_set("live_running",False); log("⏹ Stopped.")

# ── Optimization ──────────────────────────────────────────────────────────────
def run_opt(df,base_cfg,target_acc=0.0,ph=None):
    rows=[]; total=len(range(5,22,2))*len(range(10,32,2))*5*4; done=0
    for fp in range(5,22,2):
        for sp in range(10,32,2):
            if fp>=sp: done+=20; continue
            for sl_p in [5,10,15,20,25]:
                for tgt_p in [10,20,30,50]:
                    try:
                        c={**base_cfg,"fast_ema":fp,"slow_ema":sp,"sl_points":sl_p,"target_points":tgt_p}
                        tr,_,_=run_backtest(df.copy(),c)
                        if len(tr)>=3:
                            wins,nt,acc=calc_acc(tr); pnl=sum(t["PnL"] for t in tr)
                            rows.append({"Fast EMA":fp,"Slow EMA":sp,"SL Pts":sl_p,"Tgt Pts":tgt_p,
                                         "Trades":nt,"Wins":wins,"Accuracy %":acc,
                                         "Total PnL":round(pnl,2),"Avg PnL":round(pnl/nt,2)})
                    except: pass
                    done+=1
                    if ph and done%20==0: ph.progress(min(done/total,1.0))
    if not rows: return pd.DataFrame()
    res=pd.DataFrame(rows).sort_values("Accuracy %",ascending=False).reset_index(drop=True)
    if target_acc>0:
        f=res[res["Accuracy %"]>=target_acc]; return f if not f.empty else res
    return res

# ── Chart helpers ─────────────────────────────────────────────────────────────
_G="#00e676"; _R="#ff1744"; _O="#ff9800"; _B="#40c4ff"; _Y="#ffea00"; _P="#ce93d8"

def _layout(title="",h=680):
    return dict(title=dict(text=title,font=dict(color="white",size=13)),height=h,
                template="plotly_dark",paper_bgcolor="rgba(13,17,23,1)",
                plot_bgcolor="rgba(13,17,23,1)",font=dict(color="white",family="Space Mono,monospace"),
                xaxis_rangeslider_visible=False,
                legend=dict(bgcolor="rgba(0,0,0,0.4)",bordercolor="rgba(255,255,255,0.1)",borderwidth=1))

def bt_chart(df,trades,inds,cfg):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="Price",increasing=dict(line=dict(color=_G),fillcolor="rgba(0,230,118,0.15)"),
        decreasing=dict(line=dict(color=_R),fillcolor="rgba(255,23,68,0.15)")),row=1,col=1)
    if "fast_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index,y=inds["fast_ema"],name=f"EMA{cfg.get('fast_ema',9)}",
                                  line=dict(color=_O,width=1.5)),row=1,col=1)
    if "slow_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index,y=inds["slow_ema"],name=f"EMA{cfg.get('slow_ema',15)}",
                                  line=dict(color=_B,width=1.5)),row=1,col=1)
    if "elliott" in inds and inds["elliott"]:
        for wl in (inds["elliott"].get("wave_labels") or []):
            try:
                fig.add_annotation(x=wl["dt"],y=wl["price"],text=wl["label"],showarrow=True,
                    arrowhead=2,arrowcolor=_Y,font=dict(color=_Y,size=11),
                    bgcolor="rgba(0,0,0,0.6)",row=1,col=1)
            except: pass
    bx,by,sx,sy,ex,ey,ec,et=[],[],[],[],[],[],[],[]
    for t in trades:
        try:
            edt=pd.to_datetime(t["Entry DateTime"],errors="coerce")
            xdt=pd.to_datetime(t["Exit DateTime"],errors="coerce")
        except: continue
        if t["Signal"]=="BUY": bx.append(edt); by.append(t["Entry Price"])
        else: sx.append(edt); sy.append(t["Entry Price"])
        ec.append("rgba(0,230,118,0.9)" if t["PnL"]>=0 else "rgba(255,23,68,0.9)")
        ex.append(xdt); ey.append(t["Exit Price"])
        et.append(f"{t['Signal']}|PnL:{t['PnL']:+.2f}<br>{t['Exit Reason']}")
    if bx: fig.add_trace(go.Scatter(x=bx,y=by,mode="markers",name="Buy Entry",
        marker=dict(symbol="triangle-up",size=12,color=_G,line=dict(color="white",width=0.5))),row=1,col=1)
    if sx: fig.add_trace(go.Scatter(x=sx,y=sy,mode="markers",name="Sell Entry",
        marker=dict(symbol="triangle-down",size=12,color=_R,line=dict(color="white",width=0.5))),row=1,col=1)
    if ex: fig.add_trace(go.Scatter(x=ex,y=ey,mode="markers",name="Exit",
        marker=dict(symbol="x-thin-open",size=11,color=ec,line=dict(width=2)),
        text=et,hovertemplate="%{text}<extra></extra>"),row=1,col=1)
    for t in trades[-8:]:
        try:
            e_dt=pd.to_datetime(t["Entry DateTime"],errors="coerce")
            x_dt=pd.to_datetime(t["Exit DateTime"],errors="coerce")
            fig.add_shape(type="line",x0=e_dt,x1=x_dt,y0=t["SL"],y1=t["SL"],
                line=dict(color=_R,width=1,dash="dot"),row=1,col=1)
            fig.add_shape(type="line",x0=e_dt,x1=x_dt,y0=t["Target"],y1=t["Target"],
                line=dict(color=_G,width=1,dash="dot"),row=1,col=1)
        except: pass
    vc=[_G if c>=o else _R for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Vol",marker_color=vc,opacity=0.5),row=2,col=1)
    fig.update_layout(**_layout("Backtest"))
    return fig

def live_chart(df,pos,inds,cfg):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        name="Price",increasing=dict(line=dict(color=_G),fillcolor="rgba(0,230,118,0.12)"),
        decreasing=dict(line=dict(color=_R),fillcolor="rgba(255,23,68,0.12)")),row=1,col=1)
    if "fast_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index,y=inds["fast_ema"],name=f"EMA{cfg.get('fast_ema',9)}",
                                  line=dict(color=_O,width=1.5)),row=1,col=1)
    if "slow_ema" in inds:
        fig.add_trace(go.Scatter(x=df.index,y=inds["slow_ema"],name=f"EMA{cfg.get('slow_ema',15)}",
                                  line=dict(color=_B,width=1.5)),row=1,col=1)
    if pos is not None:
        ep,sl,tgt=pos["entry_price"],pos["current_sl"],pos["target"]
        fig.add_hline(y=ep,line=dict(color="white",width=1.5,dash="dash"),
                      annotation_text=f"Entry {ep:.2f}",annotation_position="right",row=1,col=1)
        fig.add_hline(y=sl,line=dict(color=_R,width=1.5,dash="dash"),
                      annotation_text=f"SL {sl:.2f}",annotation_position="right",row=1,col=1)
        fig.add_hline(y=tgt,line=dict(color=_G,width=1.5,dash="dash"),
                      annotation_text=f"Tgt {tgt:.2f}",annotation_position="right",row=1,col=1)
    if "elliott" in inds and inds["elliott"]:
        for wl in (inds["elliott"].get("wave_labels") or []):
            try:
                fig.add_annotation(x=pd.to_datetime(wl["dt"],errors="coerce"),y=wl["price"],
                    text=wl["label"],showarrow=True,arrowhead=2,arrowcolor=_Y,
                    font=dict(color=_Y,size=10),bgcolor="rgba(0,0,0,0.6)",row=1,col=1)
            except: pass
    vc=[_G if c>=o else _R for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Vol",marker_color=vc,opacity=0.5),row=2,col=1)
    fig.update_layout(**_layout("Live",540))
    return fig

# ── Analysis Engine ───────────────────────────────────────────────────────────
def _enrich(trades):
    df=pd.DataFrame(trades).copy()
    for c in ["Entry Price","Exit Price","SL","Final SL","Target","Candle High","Candle Low","PnL"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["_edt"]=pd.to_datetime(df["Entry DateTime"].str.replace(" IST","",regex=False),errors="coerce")
    df["Day"]=df["_edt"].dt.day_name().fillna("Unknown")
    df["Hour"]=df["_edt"].dt.hour.fillna(-1).astype(int)
    df["Win"]=(df["PnL"]>0).astype(int)
    def _ex(pat,txt,d=np.nan):
        m=re.search(pat,str(txt)); return float(m.group(1)) if m else d
    df["Angle"]=df["Entry Reason"].apply(lambda x:_ex(r"∠=([\d.]+)",x))
    df["Delta"]=df["Entry Reason"].apply(lambda x:_ex(r"Δ=([\d.]+)",x))
    df["AngleBin"]=pd.cut(df["Angle"].fillna(0),bins=[0,5,10,20,30,45,60,90],
        labels=["0-5°","5-10°","10-20°","20-30°","30-45°","45-60°","60-90°"],include_lowest=True)
    df["DeltaBin"]=pd.cut(df["Delta"].fillna(0),bins=[0,1,2,5,10,20,50,1e6],
        labels=["0-1","1-2","2-5","5-10","10-20","20-50","50+"],include_lowest=True)
    return df

def _bar(x,y,colors,title,ytitle="PnL",h=260):
    fig=go.Figure(go.Bar(x=x,y=y,marker_color=colors,
        text=[f"{v:+.1f}" for v in y],textposition="outside",textfont=dict(size=10,color="white")))
    fig.update_layout(**_layout(title,h),
        xaxis=dict(showgrid=False,tickfont=dict(size=10)),
        yaxis=dict(title=ytitle,showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(t=40,b=40,l=40,r=20))
    return fig

def _sc(df,xcol,title,h=260):
    c=[_G if p>=0 else _R for p in df["PnL"]]
    fig=go.Figure(go.Scatter(x=df[xcol],y=df["PnL"],mode="markers",
        marker=dict(color=c,size=8,opacity=0.8,line=dict(color="rgba(255,255,255,0.2)",width=0.5)),
        text=df.apply(lambda r:f"{r.get('Signal','')} PnL:{r['PnL']:+.2f}",axis=1),
        hovertemplate="%{text}<extra></extra>"))
    fig.add_hline(y=0,line=dict(color="rgba(255,255,255,0.3)",dash="dot",width=1))
    fig.update_layout(**_layout(title,h),
        xaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="PnL",showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(t=40,b=40,l=40,r=20))
    return fig

def _grp(df,col):
    g=df.groupby(col,observed=True)["PnL"].agg(["sum","count","mean"])
    w=df.groupby(col,observed=True)["Win"].sum()
    g=g.join(w).rename(columns={"sum":"Net","count":"N","mean":"Avg","Win":"Wins"})
    g["WR"]=(g["Wins"]/g["N"]*100).round(1)
    return list(g.index),list(g["Net"].round(2)),[_G if v>=0 else _R for v in g["Net"]],list(g["WR"])

def render_analysis(trades):
    st.markdown("---")
    st.markdown("## 🔬 Trade Analysis & Insights")
    st.caption("All times in IST. Use these insights to find your edge and eliminate traps.")
    df=_enrich(trades)
    if df.empty: return
    total=df["PnL"].sum()
    bpnl=df[df["Signal"]=="BUY"]["PnL"].sum(); spnl=df[df["Signal"]=="SELL"]["PnL"].sum()
    bwr=(df[df["Signal"]=="BUY"]["Win"].mean()*100) if (df["Signal"]=="BUY").any() else 0
    swr=(df[df["Signal"]=="SELL"]["Win"].mean()*100) if (df["Signal"]=="SELL").any() else 0
    st.markdown(f"""<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px;">
      <div style="background:#1f2733;border:1px solid rgba(0,230,118,0.3);border-radius:8px;padding:10px 18px;">
        <div style="font-size:10px;color:#8b949e;">TOTAL PnL</div>
        <div style="font-size:22px;font-weight:700;color:{'#00e676' if total>=0 else '#ff1744'};">{total:+.2f}</div>
      </div>
      <div style="background:#1f2733;border:1px solid rgba(0,230,118,0.3);border-radius:8px;padding:10px 18px;">
        <div style="font-size:10px;color:#8b949e;">BUY SIDE</div>
        <div style="font-size:22px;font-weight:700;color:{'#00e676' if bpnl>=0 else '#ff1744'};">{bpnl:+.2f}</div>
        <div style="font-size:10px;color:#8b949e;">{bwr:.1f}% win rate</div>
      </div>
      <div style="background:#1f2733;border:1px solid rgba(0,230,118,0.3);border-radius:8px;padding:10px 18px;">
        <div style="font-size:10px;color:#8b949e;">SELL SIDE</div>
        <div style="font-size:22px;font-weight:700;color:{'#00e676' if spnl>=0 else '#ff1744'};">{spnl:+.2f}</div>
        <div style="font-size:10px;color:#8b949e;">{swr:.1f}% win rate</div>
      </div></div>""",unsafe_allow_html=True)

    st.markdown("### 📅 Time-Based Analysis")
    day_ord=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    df_d=df[df["Day"].isin(day_ord)]; c1,c2=st.columns(2)
    with c1:
        if not df_d.empty:
            x,y,c,_=_grp(df_d,"Day")
            st.plotly_chart(_bar(x,y,c,"PnL by Day of Week"),use_container_width=True,key="_an_d")
    with c2:
        df_h=df[df["Hour"]>=0]
        if not df_h.empty:
            x,y,c,_=_grp(df_h,"Hour"); xl=[f"{h:02d}:00 IST" for h in x]
            st.plotly_chart(_bar(xl,y,c,"PnL by Hour (IST)"),use_container_width=True,key="_an_h")

    if not df_d.empty and df_d["Hour"].ge(0).any():
        pv=df_d.pivot_table(values="Win",index="Day",columns="Hour",aggfunc="mean")
        pv=pv.reindex([d for d in day_ord if d in pv.index])
        if not pv.empty:
            z=(pv.fillna(np.nan)*100).values.tolist()
            fh=go.Figure(go.Heatmap(z=z,x=[f"{h:02d}:00" for h in pv.columns],y=list(pv.index),
                colorscale="RdYlGn",showscale=True,
                text=[[f"{v:.0f}%" if not(isinstance(v,float) and np.isnan(v)) else "" for v in row] for row in z],
                texttemplate="%{text}",textfont=dict(size=10)))
            fh.update_layout(**_layout("Win-Rate Heatmap: Day × Hour (IST)",300),
                margin=dict(t=40,b=60,l=70,r=20))
            st.plotly_chart(fh,use_container_width=True,key="_an_hm")

    st.markdown("### 📊 Long vs Short Analysis")
    c3,c4,c5=st.columns(3)
    with c3:
        g=df.groupby("Signal")["PnL"].sum().reset_index()
        cc=[_G if v>=0 else _R for v in g["PnL"]]
        st.plotly_chart(_bar(g["Signal"].tolist(),g["PnL"].round(2).tolist(),cc,"Net PnL: Long vs Short"),use_container_width=True,key="_an_s1")
    with c4:
        g2=(df.groupby("Signal")["Win"].mean()*100).reset_index()
        cc2=[_G if v>=50 else _R for v in g2["Win"]]
        st.plotly_chart(_bar(g2["Signal"].tolist(),g2["Win"].round(1).tolist(),cc2,"Win Rate %",ytitle="Win%"),use_container_width=True,key="_an_s2")
    with c5:
        trap=df[df["Exit Reason"].str.contains("SL",na=False)]
        if not trap.empty:
            tc=trap.groupby("Signal").size().reset_index(name="N")
            st.plotly_chart(_bar(tc["Signal"].tolist(),tc["N"].tolist(),[_R]*len(tc),"SL Traps by Side",ytitle="#"),use_container_width=True,key="_an_s3")
        else: st.info("No SL traps detected ✅")

    ha=df["Angle"].notna().any(); hd=df["Delta"].notna().any()
    if ha:
        st.markdown("### 📐 Crossover Angle Analysis")
        ca1,ca2=st.columns(2)
        with ca1:
            dfa=df.dropna(subset=["AngleBin"])
            if not dfa.empty:
                x,y,c,_=_grp(dfa,"AngleBin")
                st.plotly_chart(_bar([str(v) for v in x],y,c,"PnL by Angle Bin"),use_container_width=True,key="_an_ab")
        with ca2:
            dfa2=df.dropna(subset=["Angle"])
            if not dfa2.empty:
                st.plotly_chart(_sc(dfa2,"Angle","PnL vs Angle (°)"),use_container_width=True,key="_an_as")
    if hd:
        st.markdown("### ↔️ EMA Delta Analysis")
        cd1,cd2=st.columns(2)
        with cd1:
            dfd=df.dropna(subset=["DeltaBin"])
            if not dfd.empty:
                x,y,c,_=_grp(dfd,"DeltaBin")
                st.plotly_chart(_bar([str(v) for v in x],y,c,"PnL by Delta Bin"),use_container_width=True,key="_an_db")
        with cd2:
            dfd2=df.dropna(subset=["Delta"])
            if not dfd2.empty:
                st.plotly_chart(_sc(dfd2,"Delta","PnL vs Delta"),use_container_width=True,key="_an_ds")
    if ha and hd:
        df_ad=df.dropna(subset=["AngleBin","DeltaBin"])
        if len(df_ad)>=4:
            pv2=df_ad.pivot_table(values="PnL",index="AngleBin",columns="DeltaBin",aggfunc="sum",observed=True)
            z2=pv2.fillna(0).values.tolist()
            fad=go.Figure(go.Heatmap(z=z2,x=[str(c) for c in pv2.columns],y=[str(r) for r in pv2.index],
                colorscale="RdYlGn",showscale=True,
                text=[[f"{v:.1f}" for v in row] for row in z2],texttemplate="%{text}",textfont=dict(size=10)))
            fad.update_layout(**_layout("PnL Heatmap: Angle × Delta",300),margin=dict(t=40,b=60,l=80,r=20))
            st.plotly_chart(fad,use_container_width=True,key="_an_ad")

    st.markdown("### 🚪 Exit Analysis")
    ce1,ce2=st.columns(2)
    with ce1:
        er=df.groupby("Exit Reason")["PnL"].sum().sort_values(ascending=False).reset_index()
        cc3=[_G if v>=0 else _R for v in er["PnL"]]
        st.plotly_chart(_bar(er["Exit Reason"].tolist(),er["PnL"].round(2).tolist(),cc3,"PnL by Exit Reason",h=280),use_container_width=True,key="_an_ex")
    with ce2:
        ec2=df["Exit Reason"].value_counts().reset_index(); ec2.columns=["Reason","Count"]
        fp2=go.Figure(go.Pie(labels=ec2["Reason"],values=ec2["Count"],hole=0.45,
            marker=dict(colors=[_G,_R,_O,_B,_Y,_P][:len(ec2)]),textfont=dict(size=11)))
        fp2.update_layout(**_layout("Exit Distribution",280),margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fp2,use_container_width=True,key="_an_pie")

    st.markdown("### 📈 Equity Curve & Drawdown")
    ceq1,ceq2=st.columns(2)
    cum=df["PnL"].cumsum()
    with ceq1:
        feq=go.Figure(go.Scatter(y=cum.values,mode="lines",line=dict(color=_B,width=2),
            fill="tozeroy",fillcolor="rgba(64,196,255,0.07)",name="Equity"))
        feq.update_layout(**_layout("Equity Curve",240),
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
            margin=dict(t=40,b=30,l=40,r=20))
        st.plotly_chart(feq,use_container_width=True,key="_an_eq")
    with ceq2:
        dd=cum-cum.cummax()
        fdd=go.Figure(go.Scatter(y=dd.values,mode="lines",line=dict(color=_R,width=1.5),
            fill="tozeroy",fillcolor="rgba(255,23,68,0.1)",name="Drawdown"))
        fdd.update_layout(**_layout("Drawdown",240),
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
            margin=dict(t=40,b=30,l=40,r=20))
        st.plotly_chart(fdd,use_container_width=True,key="_an_dd")

    st.markdown("### 💡 Key Insights")
    ins=[]
    if not df_d.empty:
        dp=df_d.groupby("Day")["PnL"].sum()
        if not dp.empty:
            ins.append(f"🗓️ **Best day:** {dp.idxmax()} ({dp.max():+.2f}) · **Worst:** {dp.idxmin()} ({dp.min():+.2f}) → avoid {dp.idxmin()}")
    df_h2=df[df["Hour"]>=0]
    if not df_h2.empty:
        hp=df_h2.groupby("Hour")["PnL"].sum()
        if not hp.empty:
            bh=hp.idxmax(); wh=hp.idxmin()
            ins.append(f"⏰ **Best hour:** {bh:02d}:00–{bh+1:02d}:00 IST ({hp.max():+.2f}) · **Worst:** {wh:02d}:00 ({hp.min():+.2f})")
    ins.append(f"📊 **Side dominance:** BUY {bpnl:+.2f} ({bwr:.0f}% win) vs SELL {spnl:+.2f} ({swr:.0f}% win) → focus on {'BUY' if bpnl>spnl else 'SELL'}")
    if ha:
        dfa3=df.dropna(subset=["AngleBin"])
        if not dfa3.empty:
            ab=dfa3.groupby("AngleBin",observed=True)["PnL"].sum()
            if not ab.empty: ins.append(f"📐 **Best angle:** {ab.idxmax()} (PnL {ab.max():+.2f}) → set angle filter to this range")
    if hd:
        dfd3=df.dropna(subset=["DeltaBin"])
        if not dfd3.empty:
            db=dfd3.groupby("DeltaBin",observed=True)["PnL"].sum()
            if not db.empty: ins.append(f"↔️ **Best delta:** {db.idxmax()} (PnL {db.max():+.2f}) → set delta filter to this range")
    sl_h=(df["Exit Reason"].str.contains("SL",na=False)).sum()
    tgt_h=(df["Exit Reason"].str.contains("Target",na=False)).sum()
    if sl_h+tgt_h>0:
        tp=sl_h/(sl_h+tgt_h)*100
        ins.append(f"🪤 **SL trap rate:** {tp:.1f}% ({sl_h} SL vs {tgt_h} Target) — "
                   +("consider widening SL" if tp>50 else "good target hit rate ✅"))
    cs=0; ms=0
    for pnl in df["PnL"]:
        cs=cs+1 if pnl<0 else 0; ms=max(ms,cs)
    if ms>0: ins.append(f"⚠️ **Max consecutive losses:** {ms} — size positions accordingly")
    for i in ins:
        st.markdown(f'<div style="background:#1f2733;border-left:3px solid {_B};border-radius:4px;'
                    f'padding:10px 14px;margin:5px 0;font-size:12px;">{i}</div>',unsafe_allow_html=True)

# ── Elliott Wave info panel ───────────────────────────────────────────────────
def ew_panel(ew):
    if not ew or ew.get("pattern")=="Insufficient Data": return
    dc={"Bullish":_G,"Bearish":_R,"Bullish ABC":_Y,"Bearish ABC":_O}.get(ew.get("wave_direction",""),_B)
    sc=_G if ew.get("signal")=="BUY" else (_R if ew.get("signal")=="SELL" else "#8b949e")
    fib="".join(f'<div style="display:flex;justify-content:space-between;font-size:11px;padding:1px 0;'
                f'color:#a0aab4;"><span>{k}</span><span style="color:#e6edf3;">{v}</span></div>'
                for k,v in (ew.get("fibonacci_levels") or {}).items())
    wp="".join(f'<div style="display:flex;justify-content:space-between;font-size:11px;padding:1px 0;'
               f'color:#a0aab4;"><span>{k}</span><span style="color:#e6edf3;">{v}</span></div>'
               for k,v in (ew.get("wave_points") or {}).items())
    tgt=ew.get("next_target"); ts=f"{tgt:.2f}" if tgt else "–"
    st.markdown(f"""<div style="background:#1f2733;border:1px solid rgba(255,255,255,0.08);
      border-radius:10px;padding:14px;margin:4px 0;">
      <div style="font-size:13px;color:{_Y};font-weight:700;margin-bottom:6px;">🌊 {ew.get("pattern","–")}</div>
      <div style="font-size:11px;color:{dc};font-weight:700;">{ew.get("wave_direction","–")}</div>
      <div style="font-size:11px;color:#8b949e;margin:3px 0;">Done: {" → ".join(ew.get("completed_waves",[]) or ["–"])}</div>
      <div style="font-size:12px;color:#e6edf3;margin:3px 0;">▶ {ew.get("current_wave","–")}</div>
      <div style="font-size:12px;margin:4px 0;">Signal:<b style="color:{sc};"> {ew.get("signal","NONE")}</b>
        &nbsp;·&nbsp; Next Target:<b style="color:{_Y};"> {ts}</b></div>
      <div style="margin-top:6px;">{wp}</div>
      <div style="margin-top:6px;border-top:1px solid rgba(255,255,255,0.07);padding-top:6px;">{fib}</div>
    </div>""",unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root{--bg:#0d1117;--sf:#161b22;--sf2:#1f2733;--br:rgba(255,255,255,0.08);
  --gr:#00e676;--re:#ff1744;--or:#ff9800;--bl:#40c4ff;--ye:#ffea00;--tx:#e6edf3;--mu:#8b949e;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;
  color:var(--tx)!important;font-family:'Space Mono',monospace!important;}
[data-testid="stSidebar"]{background:var(--sf)!important;border-right:1px solid var(--br)!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--tx)!important;}
.log-box{background:#0a0e13;border:1px solid var(--br);border-radius:8px;padding:12px 16px;
  height:200px;overflow-y:auto;font-size:11px;color:#a0aab4;line-height:1.7;}
.cfg-bar{background:var(--sf2);border:1px solid var(--br);border-radius:8px;
  padding:8px 14px;font-size:12px;margin-bottom:8px;}
.stButton>button{font-family:'Space Mono',monospace!important;font-weight:700;
  border-radius:8px!important;border:1px solid var(--br)!important;}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif!important;font-size:14px!important;font-weight:600;}
[data-testid="metric-container"]{background:var(--sf2);border:1px solid var(--br);
  border-radius:10px;padding:12px!important;}
.ldot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--gr);
  animation:blink 1s step-end infinite;margin-right:4px;}
@keyframes blink{50%{opacity:0;}}
</style>"""

# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Smart Investing",page_icon="📈",layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS,unsafe_allow_html=True)
    init_ss()

    # ══ SIDEBAR ══════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:6px 0 12px;">
          <div style="font-family:Syne,sans-serif;font-size:23px;font-weight:800;color:#00e676;">
            📈 Smart Investing</div>
          <div style="font-size:10px;color:#8b949e;">Algo Trading Platform v3.0</div>
        </div>""",unsafe_allow_html=True)

        st.markdown("### 🎯 Instrument")
        tn=st.selectbox("Ticker",list(TICKERS.keys()),key="_tk")
        sym=st.text_input("Custom Symbol","RELIANCE.NS",key="_cs") if tn=="Custom" else TICKERS[tn]
        iv=st.selectbox("Timeframe",list(TF_PERIODS.keys()),index=1,key="_iv")
        pd_=st.selectbox("Period",TF_PERIODS[iv],index=min(1,len(TF_PERIODS[iv])-1),key="_pd")

        st.markdown("---\n### 📐 Strategy")
        strat=st.selectbox("Strategy",STRATEGIES,key="_str")
        tdir=st.selectbox("Trade Direction",TRADE_DIRS,key="_tdir")

        fast_ema=9; slow_ema=15; min_ang=0.0; max_ang=90.0
        min_delta=0.0; max_delta=1e9; cx_type="Simple Crossover"
        cx_csz=10; min_wave_pct=0.5

        if strat not in ("Simple Buy","Simple Sell"):
            c1,c2=st.columns(2)
            fast_ema=c1.number_input("Fast EMA",2,50,9,key="_fe")
            slow_ema=c2.number_input("Slow EMA",3,200,15,key="_se")

        if strat in ("EMA Crossover","Anticipatory EMA"):
            if st.checkbox("Angle Filter",False,key="_ua"):
                a1,a2=st.columns(2)
                min_ang=a1.number_input("Min°",0.0,89.0,0.0,1.0,key="_mna")
                max_ang=a2.number_input("Max°",1.0,90.0,90.0,1.0,key="_mxa")
            if st.checkbox("Delta Filter |Fast−Slow|",False,key="_ud"):
                d1,d2=st.columns(2)
                min_delta=d1.number_input("Min Δ",0.0,100000.0,0.0,0.5,key="_mnd")
                max_delta=d2.number_input("Max Δ",0.0,100000.0,9999.0,0.5,key="_mxd")
            cx_type=st.selectbox("Crossover Type",CX_TYPES,key="_cxt")
            if cx_type=="Custom Candle Size":
                cx_csz=st.number_input("Min Candle Size",1,10000,10,key="_ccs")

        if strat=="Elliott Wave":
            min_wave_pct=st.number_input("Min Wave %",0.1,10.0,0.5,0.1,key="_mwp")

        st.markdown("---\n### 🛡️ Stop Loss")
        sl_t=st.selectbox("SL Type",SL_TYPES,key="_slt")
        sl_pts=st.number_input("SL Points",1,1000000,10,key="_slp")
        atr_slm=1.5; rr=2.0; sw=20; vol_slm=2.0
        if "ATR" in sl_t:    atr_slm=st.number_input("ATR SL Mult",0.5,10.0,1.5,0.1,key="_asm")
        if "Risk" in sl_t:   rr=st.number_input("Risk:Reward",0.5,20.0,2.0,0.5,key="_rr")
        if "Swing" in sl_t or "Support" in sl_t: sw=st.number_input("Swing Bars",5,100,20,key="_sw")
        if "Volat" in sl_t:  vol_slm=st.number_input("Vol SL Mult",0.5,10.0,2.0,0.1,key="_vsm")

        st.markdown("---\n### 🎯 Target")
        tgt_t=st.selectbox("Target Type",TGT_TYPES,key="_tgt")
        tgt_pts=st.number_input("Target Points",1,1000000,20,key="_tgtp")
        atr_tgm=2.0; vol_tgm=3.0
        if "ATR" in tgt_t:   atr_tgm=st.number_input("ATR Tgt Mult",0.5,20.0,2.0,0.1,key="_atm")
        if "Risk" in tgt_t:  rr=st.number_input("R:R(Tgt)",0.5,20.0,2.0,0.5,key="_rrt")
        if "Volat" in tgt_t: vol_tgm=st.number_input("Vol Tgt Mult",0.5,20.0,3.0,0.1,key="_vtm")

        st.markdown("---\n### ⚙️ Trade Settings")
        qty=st.number_input("Quantity",1,10000000,1,key="_qty")
        en_cl=st.checkbox("Cooldown Between Trades",True,key="_ecl")
        cd_s=st.number_input("Cooldown (s)",1,3600,5,key="_cds") if en_cl else 5
        st.checkbox("Prevent Overlapping Trades",True,key="_po")

        en_pnl=st.checkbox("Daily PnL Cap",False,key="_epnl")
        mdl=-1000.0; mdp=1000.0
        if en_pnl:
            pp1,pp2=st.columns(2)
            mdl=pp1.number_input("Max Loss/Day",-10000000,0,-1000,100,key="_mdl")
            mdp=pp2.number_input("Max Profit/Day",0,10000000,1000,100,key="_mdp")

        en_tf=st.checkbox("Trade Time Filter (IST)",False,key="_etf")
        ts_=datetime.time(9,15); te_=datetime.time(15,20)
        if en_tf:
            tt1,tt2=st.columns(2)
            ts_=tt1.time_input("Start (IST)",datetime.time(9,15),key="_tsi")
            te_=tt2.time_input("End (IST)",datetime.time(15,20),key="_tei")

        st.markdown("---\n### 🏦 Dhan Broker")
        en_dhan=st.checkbox("Enable Dhan",False,key="_ed")
        dhan_cfg={"enabled":False}
        if en_dhan:
            dhan_cfg["enabled"]=True
            dhan_cfg["client_id"]=st.text_input("Client ID","1104779876",key="_did")
            dhan_cfg["access_token"]=st.text_input("Access Token","",type="password",key="_dat")
            is_opt=st.checkbox("Options Trading",False,key="_ot")
            dhan_cfg["options_trading"]=is_opt
            if not is_opt:
                dhan_cfg["exchange"]=st.selectbox("Exchange",["NSE","BSE"],key="_dex")
                dhan_cfg["product_type"]=st.selectbox("Product",["INTRADAY","DELIVERY"],key="_dpt")
                dhan_cfg["security_id"]=st.text_input("Security ID","1594",key="_dsid")
                dhan_cfg["quantity"]=st.number_input("Order Qty",1,10000000,1,key="_dqty")
                dhan_cfg["entry_order_type"]=st.selectbox("Entry Order",["MARKET","LIMIT"],index=1,key="_det")
                dhan_cfg["exit_order_type"]=st.selectbox("Exit Order",["MARKET","LIMIT"],key="_dxt")
            else:
                dhan_cfg["options_exchange"]=st.selectbox("Opt Exchange",["NSE_FNO","BSE_FNO"],key="_oex")
                dhan_cfg["ce_security_id"]=st.text_input("CE Sec ID","57749",key="_ce")
                dhan_cfg["pe_security_id"]=st.text_input("PE Sec ID","57716",key="_pe")
                dhan_cfg["options_quantity"]=st.number_input("Opt Qty",1,10000000,65,key="_oqty")
                dhan_cfg["options_entry_order_type"]=st.selectbox("Opt Entry",["MARKET","LIMIT"],key="_oet")
                dhan_cfg["options_exit_order_type"]=st.selectbox("Opt Exit",["MARKET","LIMIT"],key="_oxt")
            if st.button("🌐 Register IP (SEBI)",key="_rip"):
                ip=get_ip(); st.info(f"Your IP: **{ip}** — add to Dhan whitelist")

    cfg={"strategy":strat,"fast_ema":fast_ema,"slow_ema":slow_ema,"trade_direction":tdir,
         "sl_type":sl_t,"sl_points":sl_pts,"atr_sl_mult":atr_slm,
         "target_type":tgt_t,"target_points":tgt_pts,"atr_target_mult":atr_tgm,
         "risk_reward":rr,"swing_window":sw,"vol_sl_mult":vol_slm,"vol_target_mult":vol_tgm,
         "quantity":qty,"enable_cooldown":en_cl,"cooldown":cd_s,
         "min_angle":min_ang,"max_angle":max_ang,"min_delta":min_delta,"max_delta":max_delta,
         "crossover_type":cx_type,"custom_candle_size":cx_csz,"min_wave_pct":min_wave_pct,
         "enable_pnl_cap":en_pnl,"max_day_loss":mdl,"max_day_profit":mdp,
         "enable_time_filter":en_tf,"trade_start_time":str(ts_),"trade_end_time":str(te_)}

    # Header
    ldot='<span class="ldot"></span>' if st.session_state.live_running else ""
    st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;padding:10px 0;
      border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:14px;">
      <span style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#00e676;">📈 Smart Investing</span>
      <span style="font-size:10px;color:#8b949e;border:1px solid rgba(255,255,255,0.1);
        border-radius:4px;padding:2px 6px;">v3.0</span>
      {ldot}<span style="font-size:12px;color:#8b949e;">{sym} · {iv} · {pd_}</span>
    </div>""",unsafe_allow_html=True)

    tab_bt,tab_lt,tab_th,tab_opt=st.tabs(["📊 Backtest","🔴 Live Trading","📋 Trade History","⚙️ Optimization"])

    # ══ TAB 1 – BACKTEST ═════════════════════════════════════════════════════
    with tab_bt:
        st.markdown("### Backtesting Engine")
        if st.button("▶ Run Backtest", type="primary", key="_brun"):
            with st.spinner("Fetching data with warmup…"):
                df_bt = fetch_data(sym, pd_, iv)
            if df_bt is None or len(df_bt) < 5:
                st.error("❌ No data fetched. Check ticker / period / connection.")
            else:
                pb = st.progress(0.0, "Running backtest…")
                trades, viols, inds = run_backtest(df_bt.copy(), cfg, lambda v: pb.progress(min(v, 1.0)))
                pb.empty()
                st.session_state.backtest_results = {"trades": trades, "viols": viols, "inds": inds, "df": df_bt}
                st.success(f"✅ Backtest complete — {len(trades)} trades analysed.")

        if st.session_state.backtest_results:
            res = st.session_state.backtest_results
            trades = res["trades"]; viols = res["viols"]; inds = res["inds"]; df_bt = res["df"]
            wins, nt, acc = calc_acc(trades)
            tot_pnl = sum(t["PnL"] for t in trades)
            avg_pnl = tot_pnl / nt if nt else 0
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Trades", nt); m2.metric("Wins", wins); m3.metric("Losses", nt - wins)
            m4.metric("Accuracy", f"{acc}%"); m5.metric("Total PnL", f"{tot_pnl:+.2f}"); m6.metric("Avg PnL", f"{avg_pnl:+.2f}")
            if viols:
                st.markdown(f'<div style="background:rgba(255,23,68,0.1);border:1px solid rgba(255,23,68,0.4);'
                            f'border-radius:6px;padding:8px 14px;font-size:13px;color:#ff1744;">⚠️ '
                            f'{len(viols)} candle(s) where SL & Target both hit — conservative: SL taken first</div>',
                            unsafe_allow_html=True)
            st.plotly_chart(bt_chart(df_bt, trades, inds, cfg), use_container_width=True, key="_btfig")
            st.markdown("#### 📋 Trade Log")
            if trades:
                df_tr = pd.DataFrame(trades)
                for c in ["Entry Price","Exit Price","SL","Final SL","Target","Candle High","Candle Low","PnL"]:
                    if c in df_tr.columns: df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce")
                if "Is Violation" in df_tr.columns: df_tr["Is Violation"] = df_tr["Is Violation"].astype(bool)
                def _srow(row):
                    try: pnl = float(row["PnL"])
                    except: pnl = 0.0
                    bg = "background-color:rgba(0,230,118,0.07)" if pnl >= 0 else "background-color:rgba(255,23,68,0.07)"
                    try:
                        if bool(row.get("Is Violation", False)): bg += ";border-left:3px solid #ff9800"
                    except: pass
                    return [bg] * len(row)
                fmt = {c: "{:.2f}" for c in ["Entry Price","Exit Price","SL","Final SL","Target","Candle High","Candle Low"] if c in df_tr.columns}
                if "PnL" in df_tr.columns: fmt["PnL"] = "{:+.2f}"
                st.dataframe(df_tr.style.apply(_srow, axis=1).format(fmt, na_rep="N/A"), use_container_width=True, height=380)
            if viols:
                st.markdown(f"#### ⚠️ Violations ({len(viols)})")
                st.dataframe(pd.DataFrame(viols), use_container_width=True)
            if strat == "Elliott Wave" and "elliott" in inds:
                st.markdown("#### 🌊 Elliott Wave")
                ew_panel(inds["elliott"])
            if len(trades) >= 3:
                render_analysis(trades)

    # ══ TAB 2 – LIVE TRADING ═════════════════════════════════════════════════
    with tab_lt:
        st.markdown("### Live Trading")
        ctrl = st.columns([1.2, 1.2, 1.4, 5])
        start_btn = ctrl[0].button("▶ Start",     type="primary",   key="_start")
        stop_btn  = ctrl[1].button("⏹ Stop",      type="secondary", key="_stop")
        sq_btn    = ctrl[2].button("⚡ Squareoff", type="secondary", key="_sq")

        if start_btn and not st.session_state.live_running:
            se = threading.Event()
            st.session_state.stop_event   = se
            st.session_state.live_running = True
            _ts_set("trade_history", [])
            _ts_set("day_pnl", 0.0)
            _ts_set("elliott_wave_state", None)
            t = threading.Thread(target=live_thread,
                                  args=(sym, pd_, iv, cfg, se, dhan_cfg), daemon=True)
            st.session_state.live_thread = t
            t.start()
            st.toast("🟢 Live trading started!", icon="✅")

        if stop_btn and st.session_state.live_running:
            if st.session_state.stop_event: st.session_state.stop_event.set()
            st.session_state.live_running = False
            st.toast("⏹ Stop signal sent — will close after current trade exits.")

        if sq_btn:
            pos = _ts_get("current_position")
            if pos is not None:
                ltp_sq = _ts_get("last_candle", {}).get("close", pos["entry_price"])
                if dhan_cfg.get("enabled"):
                    r = place_exit(dhan_cfg, pos, ltp_sq, dhan_cfg.get("options_trading", False))
                    st.toast(f"⚡ Squareoff sent: {r}")
                _ts_set("current_position", None); _ts_set("current_pnl", 0.0)
                st.toast("✅ Position squared off.")
            else:
                st.toast("ℹ️ No open position.")

        # Fragment auto-refreshes every 1.5 s — no tab flicker
        @st.fragment(run_every=1.5)
        def _live_panel():
            _pos     = _ts_get("current_position")
            _pnl     = _ts_get("current_pnl", 0.0)
            _lc      = _ts_get("last_candle")
            _ld      = _ts_get("live_data")
            _logs    = _ts_get("live_log", [])
            _dpnl    = _ts_get("day_pnl", 0.0)
            _running = st.session_state.live_running

            # Config bar
            st.markdown(f'<div class="cfg-bar"><b>Config:</b> {sym} · {iv}/{pd_} · <b>{strat}</b>'
                        f' · EMA {fast_ema}/{slow_ema} · SL:{sl_t}({sl_pts}pts)'
                        f' · Tgt:{tgt_t}({tgt_pts}pts) · Qty:{qty}</div>', unsafe_allow_html=True)

            # Guard bar
            gp = []
            if en_tf:
                gp.append(f"⏰ Window: <b>{ts_}–{te_} IST</b>")
            if en_pnl:
                dc = "#00e676" if _dpnl >= 0 else "#ff1744"
                gp.append(f"💰 Day PnL: <b style='color:{dc};'>{_dpnl:+.2f}</b> (Cap {mdl:+.0f}/{mdp:+.0f})")
            if gp:
                st.markdown(f'<div style="background:#1f2733;border:1px solid rgba(255,255,255,0.08);'
                            f'border-radius:6px;padding:7px 14px;font-size:12px;margin-bottom:8px;">'
                            f'{" &nbsp;·&nbsp; ".join(gp)}</div>', unsafe_allow_html=True)

            # PnL Banner
            if _pos is not None:
                _sv = "🟢 LONG" if _pos["signal_type"] == 1 else "🔴 SHORT"
                _clr = "#00e676" if _pnl >= 0 else "#ff1744"
                _arr = "▲" if _pnl >= 0 else "▼"
                _ltp_v = _lc.get("close", _pos["entry_price"]) if _lc else _pos["entry_price"]
                st.markdown(f"""<div style="background:{'rgba(0,230,118,0.05)' if _pnl>=0 else 'rgba(255,23,68,0.05)'};
                  border:1px solid {_clr}44;border-radius:10px;padding:14px 20px;
                  display:flex;flex-wrap:wrap;align-items:center;gap:20px;margin-bottom:10px;">
                  <div><div style="font-size:10px;color:#8b949e;">POSITION</div>
                       <div style="font-size:20px;font-weight:700;">{_sv}</div></div>
                  <div><div style="font-size:10px;color:#8b949e;">ENTRY</div>
                       <div style="font-size:20px;font-weight:700;">{_pos['entry_price']:.2f}</div></div>
                  <div><div style="font-size:10px;color:#8b949e;">LTP</div>
                       <div style="font-size:20px;font-weight:700;">{_ltp_v:.2f}</div></div>
                  <div><div style="font-size:10px;color:#8b949e;">UNREALISED PnL (pts)</div>
                       <div style="font-size:28px;font-weight:800;color:{_clr};">{_arr} {abs(_pnl):.2f}</div></div>
                  <div><div style="font-size:10px;color:#8b949e;">PnL × QTY({qty})</div>
                       <div style="font-size:22px;font-weight:700;color:{_clr};">{_arr} {abs(_pnl*qty):.2f}</div></div>
                  <div><div style="font-size:10px;color:#8b949e;">SL / TARGET</div>
                       <div style="font-size:13px;">
                         <span style="color:#ff1744;">▼ {_pos['current_sl']:.2f}</span>&nbsp;/&nbsp;
                         <span style="color:#00e676;">▲ {_pos['target']:.2f}</span></div></div>
                  <div><div style="font-size:10px;color:#8b949e;">SINCE (IST)</div>
                       <div style="font-size:11px;color:#8b949e;">{_pos.get('entry_dt','')}</div></div>
                </div>""", unsafe_allow_html=True)
            else:
                msg = "⏳ No open position — scanning for next signal…" if _running else "▶ Click Start to begin live trading"
                st.markdown(f'<div style="background:#1f2733;border:1px solid rgba(255,255,255,0.08);'
                            f'border-radius:10px;padding:14px 20px;color:#8b949e;font-size:14px;">{msg}</div>',
                            unsafe_allow_html=True)

            # Last candle metrics
            if _lc:
                lc1,lc2,lc3,lc4,lc5,lc6 = st.columns(6)
                lc1.metric("Close",  f"{_lc.get('close',0):.2f}")
                lc2.metric("High",   f"{_lc.get('high',0):.2f}")
                lc3.metric("Low",    f"{_lc.get('low',0):.2f}")
                lc4.metric("Open",   f"{_lc.get('open',0):.2f}")
                lc5.metric("Volume", f"{_lc.get('volume',0):,}")
                lc6.metric("Bar (IST)", str(_lc.get("datetime",""))[-12:])

            st.markdown("---")
            ch_col, info_col = st.columns([2.5, 1])

            with ch_col:
                if _ld and len(_ld.get("close", [])) >= 3:
                    _dfl = pd.DataFrame({
                        "Open": _ld["open"], "High": _ld["high"],
                        "Low":  _ld["low"],  "Close": _ld["close"], "Volume": _ld["volume"],
                    }, index=pd.to_datetime(_ld["index"], errors="coerce", utc=False))
                    _li = {}
                    if strat in ("EMA Crossover","Anticipatory EMA","Elliott Wave"):
                        _li["fast_ema"] = ema(_dfl["Close"], fast_ema)
                        _li["slow_ema"] = ema(_dfl["Close"], slow_ema)
                    _ew = _ts_get("elliott_wave_state")
                    if _ew: _li["elliott"] = _ew
                    st.plotly_chart(live_chart(_dfl, _pos, _li, cfg), use_container_width=True, key="_lfig")
                else:
                    st.info("⏳ Fetching live data — first load ~3 s…" if _running else "▶ Click Start")

            with info_col:
                st.markdown("**📌 Position**")
                if _pos:
                    pc = "#00e676" if _pnl >= 0 else "#ff1744"
                    st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(0,230,118,0.06),rgba(64,196,255,0.06));
                      border:1px solid rgba(0,230,118,0.2);border-radius:10px;padding:14px;">
                      <div style="font-size:16px;font-weight:700;">{"🟢 BUY" if _pos["signal_type"]==1 else "🔴 SELL"}</div>
                      <div style="margin-top:8px;font-size:12px;">
                        <div>Entry: <b>{_pos['entry_price']:.2f}</b></div>
                        <div>SL: <b style="color:#ff1744;">{_pos['current_sl']:.2f}</b></div>
                        <div>Target: <b style="color:#00e676;">{_pos['target']:.2f}</b></div>
                        <div>Since: {_pos.get('entry_dt','')}</div>
                        <div style="margin-top:8px;font-size:16px;color:{pc};font-weight:800;">P&L: {_pnl:+.2f} pts</div>
                        <div style="font-size:10px;color:#8b949e;margin-top:4px;">{_pos.get('signal_reason','')[:70]}</div>
                      </div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#8b949e;font-size:12px;padding:10px;">No open position</div>', unsafe_allow_html=True)
                _ew2 = _ts_get("elliott_wave_state")
                if _ew2:
                    st.markdown("**🌊 Elliott Wave**")
                    ew_panel(_ew2)

            st.markdown("**📋 Live Log**")
            log_html = "<br>".join(reversed(_logs[-60:]))
            st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

        _live_panel()

    # ══ TAB 3 – TRADE HISTORY ════════════════════════════════════════════════
    with tab_th:
        st.markdown("### Trade History")
        live_hist = _ts_get("trade_history", [])
        bt_hist   = st.session_state.backtest_results["trades"] if st.session_state.backtest_results else []
        mf = st.radio("Show", ["All", "Live", "Backtest"], horizontal=True, key="_hf")
        all_hist  = (live_hist + bt_hist) if mf == "All" else (live_hist if mf == "Live" else bt_hist)
        if not all_hist:
            st.info("No trades recorded yet.")
        else:
            wins_h, nt_h, acc_h = calc_acc(all_hist)
            tot_h   = sum(t["PnL"] for t in all_hist)
            best_h  = max((t["PnL"] for t in all_hist), default=0)
            worst_h = min((t["PnL"] for t in all_hist), default=0)
            hc = st.columns(6)
            hc[0].metric("Trades",    nt_h)
            hc[1].metric("Wins",      wins_h)
            hc[2].metric("Losses",    nt_h - wins_h)
            hc[3].metric("Accuracy",  f"{acc_h}%")
            hc[4].metric("Total PnL", f"{tot_h:+.2f}")
            hc[5].metric("Best Trade",f"{best_h:+.2f}")

            pnl_v  = [t["PnL"] for t in all_hist]
            cum_v  = list(pd.Series(pnl_v).cumsum())
            fp2 = make_subplots(rows=1, cols=2, subplot_titles=("Cumulative PnL", "Per-Trade PnL"))
            fp2.add_trace(go.Scatter(y=cum_v, mode="lines+markers", name="Cum PnL",
                line=dict(color=_B, width=2),
                marker=dict(color=[_G if v >= 0 else _R for v in cum_v], size=5)), row=1, col=1)
            fp2.add_trace(go.Bar(y=pnl_v, name="PnL",
                marker_color=[_G if v >= 0 else _R for v in pnl_v]), row=1, col=2)
            fp2.update_layout(**_layout("PnL Analysis", 300))
            st.plotly_chart(fp2, use_container_width=True, key="_hpnl")

            df_h = pd.DataFrame(all_hist)
            for c in ["Entry Price","Exit Price","SL","Final SL","Target","PnL"]:
                if c in df_h.columns: df_h[c] = pd.to_numeric(df_h[c], errors="coerce")
            def _hs(row):
                try: v = float(row["PnL"])
                except: v = 0.0
                bg = "background-color:rgba(0,230,118,0.07)" if v >= 0 else "background-color:rgba(255,23,68,0.07)"
                return [bg] * len(row)
            fmt_h = {c: "{:.2f}" for c in ["Entry Price","Exit Price","SL","Final SL","Target"] if c in df_h.columns}
            if "PnL" in df_h.columns: fmt_h["PnL"] = "{:+.2f}"
            st.dataframe(df_h.style.apply(_hs, axis=1).format(fmt_h, na_rep="N/A"),
                         use_container_width=True, height=400)

    # ══ TAB 4 – OPTIMIZATION ═════════════════════════════════════════════════
    with tab_opt:
        st.markdown("### Strategy Optimization")
        st.caption("Grid-search over EMA periods and SL/Target points for EMA Crossover strategy.")
        oc1, oc2, oc3 = st.columns(3)
        target_acc = oc1.number_input("Min Accuracy % Filter", 0.0, 100.0, 0.0, 5.0, key="_tacc")
        min_trades = oc2.number_input("Min Trades Filter", 1, 500, 5, key="_mtr")
        run_opt_btn = oc3.button("🚀 Run Optimization", type="primary", key="_optrun")

        if run_opt_btn:
            with st.spinner("Fetching data…"):
                df_opt = fetch_data(sym, pd_, iv)
            if df_opt is None or len(df_opt) < 20:
                st.error("Not enough data for optimization.")
            else:
                pp = st.progress(0.0, "Optimizing…")
                opt_cfg = {**cfg, "strategy": "EMA Crossover"}
                res_opt = run_opt(df_opt.copy(), opt_cfg, target_acc, pp)
                pp.empty()
                if not res_opt.empty:
                    res_opt = res_opt[res_opt["Trades"] >= min_trades]
                st.session_state.opt_results = res_opt
                st.success(f"✅ Found {len(res_opt)} combinations.")

        if st.session_state.opt_results is not None and not st.session_state.opt_results.empty:
            opt_df = st.session_state.opt_results
            best   = opt_df.iloc[0]
            b1,b2,b3,b4 = st.columns(4)
            b1.metric("Best Accuracy", f"{best['Accuracy %']:.1f}%")
            b2.metric("Fast EMA",      int(best["Fast EMA"]))
            b3.metric("Slow EMA",      int(best["Slow EMA"]))
            b4.metric("Best PnL",      f"{best['Total PnL']:+.2f}")

            pv_h = opt_df.pivot_table(values="Accuracy %", index="Slow EMA",
                                       columns="Fast EMA", aggfunc="max")
            fig_h = go.Figure(go.Heatmap(
                z=pv_h.values, x=pv_h.columns, y=pv_h.index,
                colorscale="RdYlGn", showscale=True,
                text=pv_h.values.round(1), texttemplate="%{text}%",
                textfont=dict(size=10)))
            fig_h.update_layout(**_layout("EMA Accuracy Heatmap", 360))
            st.plotly_chart(fig_h, use_container_width=True, key="_optheat")

            st.dataframe(
                opt_df.head(50).style.format(
                    {"Accuracy %": "{:.1f}%", "Total PnL": "{:+.2f}", "Avg PnL": "{:+.2f}"}),
                use_container_width=True, height=360)

            if st.button("⚡ Apply Best Params", key="_applyopt"):
                st.info(f"Set → Fast EMA={int(best['Fast EMA'])}, Slow EMA={int(best['Slow EMA'])}, "
                        f"SL Points={int(best['SL Pts'])}, Target Points={int(best['Tgt Pts'])} in the sidebar.")

if __name__ == "__main__":
    main()
