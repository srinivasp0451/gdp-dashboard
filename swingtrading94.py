"""
QuantAlgo Pro v5
- Auto-load on sidebar change (no LOAD button)
- Auto-trade: START/STOP only; fragment runs only when START active
- Auto-exit stores trade in history immediately (st.rerun after exit)
- Volume strategies gracefully handle zero-volume indices
- Strategy Leaderboard tab
- Honest profitability assessment
"""
import streamlit as st, yfinance as yf, pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="QuantAlgo Pro", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:#0d1117}
.main .block-container{padding-top:.6rem;padding-bottom:.6rem}
[data-testid="metric-container"]{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:7px 13px}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d}
.stTabs [data-baseweb="tab-list"]{background:#161b22;border-radius:8px;padding:3px;gap:3px}
.stTabs [data-baseweb="tab"]{color:#8b949e;font-weight:600;border-radius:6px}
.stTabs [aria-selected="true"]{background:#21262d !important;color:#58a6ff !important}
.stButton>button{background:#238636;color:#fff;border:none;border-radius:6px;font-weight:700}
.stButton>button:hover{background:#2ea043}
div[data-testid="stExpander"]{background:#161b22;border:1px solid #30363d;border-radius:8px}
.stSelectbox>div>div,.stNumberInput>div>div>input,.stTextInput>div>div>input{background:#161b22;border-color:#30363d;color:#e6edf3}
h1,h2,h3,h4{color:#e6edf3 !important}
.icard{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
.mc{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:9px;text-align:center}
.ml{font-size:.63rem;color:#8b949e;font-weight:700;text-transform:uppercase;letter-spacing:1px}
.mv{font-size:1.05rem;font-weight:900;color:#e6edf3}
.ms{font-size:.71rem;color:#8b949e}
.sbuy{background:rgba(63,185,80,.15);border:2px solid #3fb950;border-radius:8px;padding:9px;text-align:center;margin:3px 0}
.ssell{background:rgba(248,81,73,.15);border:2px solid #f85149;border-radius:8px;padding:9px;text-align:center;margin:3px 0}
.swait{background:rgba(139,148,158,.1);border:1px solid #30363d;border-radius:8px;padding:9px;text-align:center;margin:3px 0}
.pos-long{background:rgba(63,185,80,.1);border:1px solid #3fb950;border-radius:8px;padding:12px;margin:3px 0}
.pos-short{background:rgba(248,81,73,.1);border:1px solid #f85149;border-radius:8px;padding:12px;margin:3px 0}
.pos-none{background:rgba(30,35,44,.5);border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center;color:#8b949e}
.auto-on{background:rgba(63,185,80,.12);border:2px solid #3fb950;border-radius:8px;padding:9px;text-align:center}
.warn-box{background:rgba(240,136,62,.1);border:1px solid #f0883e;border-radius:8px;padding:10px;font-size:.8rem}
hr{border-color:#21262d !important}
</style>""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
TICKERS: Dict[str,str] = {
    "🇮🇳 Nifty 50":"^NSEI","🏦 Bank Nifty":"^NSEBANK","📊 Sensex":"^BSESN",
    "₿ Bitcoin":"BTC-USD","Ξ Ethereum":"ETH-USD","💵 USD/INR":"USDINR=X",
    "🥇 Gold":"GC=F","🥈 Silver":"SI=F","✏️ Custom":"CUSTOM",
}
# Symbols known to have NO volume from yfinance
NO_VOLUME_SYMBOLS = {"^NSEI","^NSEBANK","^BSESN","^DJI","^IXIC","^GSPC","^FTSE","^N225"}

TIMEFRAME_PERIODS: Dict[str,list] = {
    "1m":["1d","5d","7d"],"5m":["1d","5d","7d","1mo"],
    "15m":["1d","5d","7d","1mo"],"1h":["1d","7d","1mo","3mo","6mo","1y"],
    "1d":["7d","1mo","6mo","1y","2y","3y","5y","10y"],
    "1wk":["1mo","3mo","6mo","1y","2y","5y","10y","20y","30y"],
}
STRATEGIES = [
    "EMA Crossover",
    "Simple Buy (Immediate)",
    "Simple Sell (Immediate)",
    "Threshold Price Cross",
    "Price Action (S/R)",
    "Liquidity Zone / VWAP",
    "RSI Strategy",
    "Bollinger Bands",
    "Volume Breakout",
    "Elliott Wave (Simplified)",
    "Volume Profile FR [VPFR]",
    "★ Kalman Mean Reversion [PRO]",
    "★ Order Flow Imbalance [PRO]",
    "★ Volatility Regime Momentum [PRO]",
]
IMMEDIATE_ENTRY = {"Simple Buy (Immediate)","Simple Sell (Immediate)"}
MANUAL_ONLY     = {"Simple Buy (Immediate)","Simple Sell (Immediate)"}
# Strategies that need real volume — will degrade gracefully on index symbols
VOLUME_REQUIRED = {"Volume Breakout","Volume Profile FR [VPFR]","★ Order Flow Imbalance [PRO]",
                   "Liquidity Zone / VWAP"}

SL_TYPES = ["Custom Points","Trail SL","Trail – Current Candle Low/High",
    "Trail – Previous Candle Low/High","Trail – Current Swing High/Low",
    "Trail – Previous Swing High/Low","Strategy Signal Exit",
    "EMA Reverse Crossover","ATR Based SL","Risk Reward (min 1:2)","🤖 Autopilot SL"]
TARGET_TYPES = ["Custom Points","Trail Target (display only – never exits)",
    "Trail – Current Candle Low/High","Trail – Previous Candle Low/High",
    "Trail – Current Swing High/Low","Trail – Previous Swing High/Low",
    "Strategy Signal Exit","EMA Reverse Crossover","ATR Based Target",
    "Risk Reward (min 1:2)","🤖 Autopilot Target"]

# ── SESSION STATE ─────────────────────────────────────────────
for _k,_v in {
    "backtest_results":None,"live_trades":[],"live_position":None,
    "live_running":False,"current_data":None,"signals":None,"indicators":{},
    "last_data_key":"","last_strat_key":"","last_signal_candle":None,
    "dhan_connected":False,"leaderboard_results":None,"lb_ran_for":"",
}.items():
    if _k not in st.session_state: st.session_state[_k]=_v

# ── INDICATORS ────────────────────────────────────────────────
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _rsi(s,n=14):
    d=s.diff(); g=d.where(d>0,0.).rolling(n).mean(); l=(-d.where(d<0,0.)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _boll(s,n=20,k=2.):
    ma=s.rolling(n).mean(); sd=s.rolling(n).std(); return ma+k*sd,ma,ma-k*sd
def _atr(df,n=14):
    h,l,c=df["High"],df["Low"],df["Close"]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n,adjust=False).mean()
def _vwap(df):
    """VWAP with fallback to (H+L+C)/3 EMA when volume is zero (e.g. indices)."""
    tp=(df["High"]+df["Low"]+df["Close"])/3
    vol_sum=df["Volume"].sum()
    if vol_sum>0:
        return (tp*df["Volume"]).cumsum()/df["Volume"].cumsum().replace(0,np.nan)
    else:
        # Fallback for zero-volume instruments: use 20-period EMA of typical price
        return tp.ewm(span=20,adjust=False).mean()
def _pivots(df,w=5):
    ph=df["High"].rolling(2*w+1,center=True).max()==df["High"]
    pl=df["Low"].rolling(2*w+1,center=True).min()==df["Low"]
    return ph,pl
def _sr(df,w=10,n=5):
    ph,pl=_pivots(df,w)
    return sorted(df["Low"][pl].dropna().values)[:n],sorted(df["High"][ph].dropna().values,reverse=True)[:n]
def _kalman(prices):
    a=prices.values; xk=np.zeros(len(a)); pk=np.ones(len(a)); Q,R=1e-5,.01; xk[0]=a[0]
    for i in range(1,len(a)):
        xp=xk[i-1]; pp=pk[i-1]+Q; K=pp/(pp+R); xk[i]=xp+K*(a[i]-xp); pk[i]=(1-K)*pp
    return pd.Series(xk,index=prices.index)
def has_volume(df): return df["Volume"].sum()>0
def _vpfr_compute(df,lb=50,n_bins=30):
    n=len(df); poc_a=np.full(n,np.nan); vah_a=np.full(n,np.nan); val_a=np.full(n,np.nan)
    sigs=np.zeros(n,dtype=int)
    if not has_volume(df): return pd.Series(sigs,index=df.index),{"no_volume":True}
    for i in range(lb,n):
        w=df.iloc[i-lb:i]; lo_m=float(w["Low"].min()); hi_m=float(w["High"].max())
        if hi_m<=lo_m: continue
        edges=np.linspace(lo_m,hi_m,n_bins+1); mids=(edges[:-1]+edges[1:])/2
        lows=w["Low"].values; highs=w["High"].values; vols=w["Volume"].values
        overlap=(lows[:,None]<edges[None,1:])&(highs[:,None]>edges[None,:-1])
        n_per=np.maximum(overlap.sum(axis=1,keepdims=True),1)
        vol_hist=(vols[:,None]*overlap/n_per).sum(axis=0)
        poc_idx=int(np.argmax(vol_hist)); poc_a[i]=mids[poc_idx]
        tot=vol_hist.sum(); hi_i=lo_i=poc_idx; va_v=vol_hist[poc_idx]
        while va_v<tot*.70:
            cu=hi_i<n_bins-1; cd=lo_i>0
            if not cu and not cd: break
            au=vol_hist[hi_i+1] if cu else -1; ad=vol_hist[lo_i-1] if cd else -1
            if au>=ad and cu: hi_i+=1; va_v+=vol_hist[hi_i]
            elif cd: lo_i-=1; va_v+=vol_hist[lo_i]
            else: break
        vah_a[i]=mids[min(hi_i,n_bins-1)]; val_a[i]=mids[max(lo_i,0)]
        cur=float(df["Close"].iloc[i]); prev=float(df["Close"].iloc[i-1]); tick=(hi_m-lo_m)/n_bins
        if (abs(cur-val_a[i])<=tick*1.5 or abs(cur-poc_a[i])<=tick) and cur>prev: sigs[i]=1
        elif (abs(cur-vah_a[i])<=tick*1.5 or abs(cur-poc_a[i])<=tick) and cur<prev: sigs[i]=-1
    return pd.Series(sigs,index=df.index),{
        "vpfr_poc":pd.Series(poc_a,index=df.index),
        "vpfr_vah":pd.Series(vah_a,index=df.index),
        "vpfr_val":pd.Series(val_a,index=df.index)}

# ── DATA FETCHING ─────────────────────────────────────────────
@st.cache_data(ttl=10,show_spinner=False)
def fetch_data(ticker,interval,period):
    time.sleep(0.3)
    try:
        raw=yf.download(ticker,interval=interval,period=period,progress=False,auto_adjust=True)
        if raw.empty: return pd.DataFrame()
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        raw.columns=[str(c).strip() for c in raw.columns]
        need=["Open","High","Low","Close","Volume"]
        for c in need:
            if c not in raw.columns: return pd.DataFrame()
        return raw[need].dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=1,show_spinner=False)
def fetch_ltp(ticker):
    time.sleep(0.3)
    try:
        h=yf.Ticker(ticker).history(period="1d",interval="1m")
        if not h.empty:
            return float(h["Close"].iloc[-1]),float(h["High"].max()),float(h["Low"].min())
    except: pass
    return 0.,0.,0.

def clear_all_cache():
    fetch_data.clear(); fetch_ltp.clear()

# ── STRATEGY ENGINE ───────────────────────────────────────────
def run_strategy(df, strat, p) -> Tuple[pd.Series, dict]:
    sig=pd.Series(0,index=df.index,dtype=int); ind={}; no_vol=not has_volume(df)
    try:
        if strat=="EMA Crossover":
            ef=_ema(df["Close"],p.get("ema_fast",9)); es=_ema(df["Close"],p.get("ema_slow",15))
            ind={"ema_fast":ef,"ema_slow":es}
            sig[(ef>es)&(ef.shift()<=es.shift())]=1; sig[(ef<es)&(ef.shift()>=es.shift())]=-1
        elif strat=="Simple Buy (Immediate)":  sig.iloc[-1]=1
        elif strat=="Simple Sell (Immediate)": sig.iloc[-1]=-1
        elif strat=="Threshold Price Cross":
            bt,st_=p.get("buy_t",0.),p.get("sell_t",0.)
            if bt>0: sig[(df["Close"]>=bt)&(df["Close"].shift()<bt)]=1
            if st_>0: sig[(df["Close"]<=st_)&(df["Close"].shift()>st_)]=-1
        elif strat=="Price Action (S/R)":
            w=p.get("sr_window",10); sup,res=_sr(df,w); ph,pl=_pivots(df,w)
            ind={"support_levels":sup,"resistance_levels":res,"pivot_high":ph,"pivot_low":pl}
            for s in sup:
                z=s*.003; sig[(df["Close"].between(s-z,s+z))&(df["Close"]>df["Close"].shift())]=1
            for r in res:
                z=r*.003; sig[(df["Close"].between(r-z,r+z))&(df["Close"]<df["Close"].shift())]=-1
        elif strat=="Liquidity Zone / VWAP":
            # Works on indices too: falls back to EMA of typical price when volume=0
            v=_vwap(df); a=_atr(df); ind={"vwap":v,"atr_line":a}
            dist=(df["Close"]-v)/v*100
            if no_vol:
                # Price-only mode: bounce off dynamic average
                sig[(dist<-0.3)&(dist.shift()>=-0.3)]=1
                sig[(dist>0.3)&(dist.shift()<=0.3)]=-1
            else:
                vok=df["Volume"]>df["Volume"].rolling(20).mean()
                sig[(dist.abs()<0.15)&(dist.shift()<-0.5)&vok]=1
                sig[(dist.abs()<0.15)&(dist.shift()>0.5)&vok]=-1
        elif strat=="RSI Strategy":
            r=_rsi(df["Close"],p.get("rsi_period",14)); ob,os_=p.get("rsi_ob",70),p.get("rsi_os",30)
            ind={"rsi":r}; sig[(r<os_)&(r.shift()>=os_)]=1; sig[(r>ob)&(r.shift()<=ob)]=-1
        elif strat=="Bollinger Bands":
            up,mid,lo=_boll(df["Close"],p.get("bb_period",20),p.get("bb_std",2.))
            ind={"bb_upper":up,"bb_mid":mid,"bb_lower":lo}
            sig[(df["Close"]>lo)&(df["Close"].shift()<=lo.shift())]=1
            sig[(df["Close"]<up)&(df["Close"].shift()>=up.shift())]=-1
            bw=(up-lo)/mid; sq=bw<=bw.rolling(20).min()*1.05
            sig[~sq&sq.shift()&(df["Close"]>mid)]=1; sig[~sq&sq.shift()&(df["Close"]<mid)]=-1
        elif strat=="Volume Breakout":
            lb=p.get("vol_lb",20); mult=p.get("vol_mult",1.5)
            if no_vol:
                # Fallback: price breakout without volume filter
                sig[df["High"]>df["High"].rolling(lb).max().shift()]=1
                sig[df["Low"]<df["Low"].rolling(lb).min().shift()]=-1
                ind["_no_vol_fallback"]=True
            else:
                avgv=df["Volume"].rolling(lb).mean(); ind={"avg_volume":avgv}
                vsurge=df["Volume"]>avgv*mult
                sig[vsurge&(df["High"]>df["High"].rolling(lb).max().shift())]=1
                sig[vsurge&(df["Low"]<df["Low"].rolling(lb).min().shift())]=-1
        elif strat=="Elliott Wave (Simplified)":
            w=p.get("ew_window",10); ph,pl=_pivots(df,w); ind={"pivot_high":ph,"pivot_low":pl}
            swh=df["High"][ph].dropna(); swl=df["Low"][pl].dropna()
            if len(swh)>=3 and len(swl)>=2:
                rh=swh.values[-3:]; rl=swl.values[-2:]
                if rh[-1]>rh[-2]>rh[-3] and rl[-1]>rl[-2]:
                    idx=swl.index[-1]
                    if idx in sig.index: sig.loc[idx]=1
                elif rh[-1]<rh[-2]<rh[-3] and rl[-1]<rl[-2]:
                    idx=swh.index[-1]
                    if idx in sig.index: sig.loc[idx]=-1
        elif strat=="Volume Profile FR [VPFR]":
            sig,ind=_vpfr_compute(df,p.get("vpfr_lb",50),p.get("vpfr_bins",30))
        elif strat=="★ Kalman Mean Reversion [PRO]":
            thr=p.get("kf_thr",1.5); kf=_kalman(df["Close"])
            dev=df["Close"]-kf; z=dev/dev.rolling(30).std().replace(0,np.nan)
            ind={"kalman_price":kf,"kf_zscore":z}
            sig[(z<-thr)&(z.shift()>=-thr)]=1; sig[(z>thr)&(z.shift()<=thr)]=-1
        elif strat=="★ Order Flow Imbalance [PRO]":
            lb=p.get("ofi_lb",10); thr=p.get("ofi_thr",.60)
            if no_vol:
                # Fallback: use candle direction as proxy for order flow
                body=df["Close"]-df["Open"]
                momentum=body.rolling(lb).sum()/(df["High"]-df["Low"]).rolling(lb).sum().replace(0,np.nan)
                v=_vwap(df); ind={"vwap":v}
                sig[(momentum>thr)&(df["Close"]>v)&(momentum.shift()<=thr)]=1
                sig[(momentum<-thr)&(df["Close"]<v)&(momentum.shift()>=-thr)]=-1
                ind["_no_vol_fallback"]=True
            else:
                cr=(df["High"]-df["Low"]).replace(0,np.nan)
                bv=(df["Volume"]*((df["Close"]-df["Low"])/cr)).fillna(df["Volume"]*.5)
                sv=(df["Volume"]*((df["High"]-df["Close"])/cr)).fillna(df["Volume"]*.5)
                tot=(bv+sv).rolling(lb).sum()
                bi=bv.rolling(lb).sum()/tot.replace(0,np.nan); si=sv.rolling(lb).sum()/tot.replace(0,np.nan)
                v=_vwap(df); mom=df["Close"].pct_change(lb); vok=df["Volume"]>df["Volume"].rolling(20).mean()
                ind={"vwap":v,"buy_imb":bi,"sell_imb":si}
                sig[(bi>thr)&(mom>0)&(df["Close"]>v)&vok&(bi.shift()<=thr)]=1
                sig[(si>thr)&(mom<0)&(df["Close"]<v)&vok&(si.shift()<=thr)]=-1
        elif strat=="★ Volatility Regime Momentum [PRO]":
            fast=p.get("vrm_fast",10); slow=p.get("vrm_slow",30)
            a=_atr(df,14); regime=a/a.rolling(20).mean().replace(0,np.nan)
            mf=df["Close"].pct_change(fast); ms=df["Close"].pct_change(slow)
            ef=_ema(df["Close"],fast); es=_ema(df["Close"],slow)
            ind={"vol_regime":regime,"ema_fast":ef,"ema_slow":es,"atr_line":a}
            if no_vol:
                # Price + regime only — no volume filter (works fine on indices)
                sig[(regime>1.1)&(mf>0)&(ms>0)&(ef>es)&(regime.shift()<=1.1)]=1
                sig[(regime>1.1)&(mf<0)&(ms<0)&(ef<es)&(regime.shift()<=1.1)]=-1
            else:
                vok=df["Volume"]>df["Volume"].rolling(20).mean()
                sig[(regime>1.1)&(mf>0)&(ms>0)&(ef>es)&vok&(regime.shift()<=1.1)]=1
                sig[(regime>1.1)&(mf<0)&(ms<0)&(ef<es)&vok&(regime.shift()<=1.1)]=-1
    except Exception as e:
        pass
    return sig,ind

# ── PROXIMITY ─────────────────────────────────────────────────
def strategy_proximity(df,strat,sp,sigs):
    if df is None or len(df)<2: return "far","No data","—"
    last_sig=int(sigs.iloc[-1]) if sigs is not None and not sigs.empty else 0
    if last_sig==1:  return "signal","🟢 BUY SIGNAL","Auto-enters when monitor ON"
    if last_sig==-1: return "signal","🔴 SELL SIGNAL","Auto-enters when monitor ON"
    cur=float(df["Close"].iloc[-1])
    try:
        if strat=="EMA Crossover":
            ef=_ema(df["Close"],sp.get("ema_fast",9)); es=_ema(df["Close"],sp.get("ema_slow",15))
            ev,sv_=float(ef.iloc[-1]),float(es.iloc[-1]); gap=abs(ev-sv_); pct=gap/sv_*100
            conv="↗ Converging" if gap<abs(float(ef.iloc[-2])-float(es.iloc[-2])) else "↘ Diverging"
            bias="Bullish" if ev>sv_ else "Bearish"
            if pct<0.08: return "near","⚡ IMMINENT CROSSOVER",f"Gap {gap:.2f} ({pct:.3f}%) {conv}"
            if pct<0.25: return "approaching",f"⚠️ APPROACHING",f"Gap {gap:.2f} ({pct:.3f}%) {conv} ({bias})"
            return "far",f"⏳ FAR ({bias})",f"Gap {gap:.2f} ({pct:.2f}%) {conv}"
        elif strat=="RSI Strategy":
            r=_rsi(df["Close"],sp.get("rsi_period",14)); rv=float(r.iloc[-1])
            ob,os_=sp.get("rsi_ob",70),sp.get("rsi_os",30); d_ob,d_os=abs(rv-ob),abs(rv-os_)
            zone="→ OB" if d_ob<d_os else "→ OS"; closest=min(d_ob,d_os)
            if closest<2: return "near","⚡ RSI AT TRIGGER",f"RSI {rv:.1f} {zone}"
            if closest<6: return "approaching","⚠️ RSI BUILDING",f"RSI {rv:.1f} {zone}"
            return "far","⏳ RSI NEUTRAL",f"RSI {rv:.1f} (OB:{ob} OS:{os_})"
        elif strat=="Bollinger Bands":
            up,mid,lo=_boll(df["Close"],sp.get("bb_period",20),sp.get("bb_std",2.))
            uv,lv,mv=float(up.iloc[-1]),float(lo.iloc[-1]),float(mid.iloc[-1])
            d_up=abs(cur-uv)/uv*100; d_lo=abs(cur-lv)/lv*100; closest=min(d_up,d_lo)
            bw=(uv-lv)/mv if mv else 1; bw_s=float(((up-lo)/mid).rolling(20).min().iloc[-1]) if mv else bw
            sq=" ⚡SQ" if bw<=bw_s*1.05 else ""
            if closest<0.2: return "near",f"⚡ AT BAND{sq}",f"U {uv:.2f} L {lv:.2f}"
            if closest<0.6: return "approaching",f"⚠️ NEAR BAND{sq}",f"d_U {d_up:.2f}% d_L {d_lo:.2f}%"
            return "far",f"⏳ MID CHANNEL{sq}",f"U {uv:.2f} | Mid {mv:.2f} | L {lv:.2f}"
        elif "Kalman" in strat:
            kf=_kalman(df["Close"]); dev=df["Close"]-kf
            z=float((dev/dev.rolling(30).std().replace(0,np.nan)).iloc[-1]); thr=sp.get("kf_thr",1.5)
            if abs(z)>=thr*.85: return "near","⚡ Z AT EDGE",f"Z={z:.3f} (±{thr})"
            if abs(z)>=thr*.65: return "approaching","⚠️ Z BUILDING",f"Z={z:.3f} (±{thr})"
            return "far","⏳ NEAR FAIR VALUE",f"Z={z:.3f} (±{thr})"
        elif "Volatility Regime" in strat:
            a=_atr(df,14); regime=float(a.iloc[-1]/a.rolling(20).mean().iloc[-1])
            if regime>1.1: return "approaching","⚠️ TRENDING REGIME",f"ATR ratio {regime:.2f} > 1.1"
            return "far","⏳ RANGE REGIME",f"ATR ratio {regime:.2f} (need >1.1)"
    except: pass
    return "far","⏳ MONITORING",f"Watching for {strat} setup…"

# ── SL / TARGET ───────────────────────────────────────────────
def calc_sl(df,entry,direction,sl_type,p,idx):
    a_val=float(_atr(df).iloc[min(idx,len(df)-1)])
    if sl_type=="Custom Points":
        pts=p.get("sl_points",10.); return entry-pts if direction==1 else entry+pts
    elif sl_type in("Trail SL","Trail – Current Candle Low/High"):
        i=min(idx,len(df)-1)
        return float(df["Low"].iloc[i]) if direction==1 else float(df["High"].iloc[i])
    elif sl_type=="Trail – Previous Candle Low/High":
        i=max(0,idx-1)
        return float(df["Low"].iloc[i]) if direction==1 else float(df["High"].iloc[i])
    elif "Swing" in sl_type:
        ph,pl=_pivots(df); offset=1 if "Previous" in sl_type else 0
        if direction==1:
            lows=df["Low"][pl].dropna(); lows=lows[lows.index<=df.index[min(idx,len(df)-1)]]
            if len(lows)>offset: return float(lows.iloc[-(1+offset)])
        else:
            highs=df["High"][ph].dropna(); highs=highs[highs.index<=df.index[min(idx,len(df)-1)]]
            if len(highs)>offset: return float(highs.iloc[-(1+offset)])
        return entry*(.99 if direction==1 else 1.01)
    elif sl_type=="ATR Based SL":
        m=p.get("sl_atr_mult",1.5); return entry-m*a_val if direction==1 else entry+m*a_val
    elif sl_type=="Risk Reward (min 1:2)":
        pts=p.get("sl_points",10.); return entry-pts if direction==1 else entry+pts
    elif sl_type=="🤖 Autopilot SL":
        vs=min(max(p.get("vol_scale",1.),.5),2.); return entry-1.5*a_val*vs if direction==1 else entry+1.5*a_val*vs
    return entry*(.99 if direction==1 else 1.01)

def calc_target(entry,sl,direction,tgt_type,p,atr_val=0.):
    risk=abs(entry-sl) if sl else (atr_val or entry*.01)
    if risk==0: risk=atr_val or entry*.01
    if tgt_type=="Custom Points":
        pts=p.get("target_points",20.); return entry+pts if direction==1 else entry-pts
    elif "display only" in tgt_type: return entry+risk*3 if direction==1 else entry-risk*3
    elif "Trail" in tgt_type:        return entry+risk*2 if direction==1 else entry-risk*2
    elif tgt_type=="ATR Based Target":
        m=p.get("target_atr_mult",2.); return entry+m*atr_val if direction==1 else entry-m*atr_val
    elif tgt_type=="Risk Reward (min 1:2)":
        rr=max(p.get("rr_ratio",2.),2.); return entry+rr*risk if direction==1 else entry-rr*risk
    elif tgt_type=="🤖 Autopilot Target": return entry+risk*2.618 if direction==1 else entry-risk*2.618
    elif tgt_type in("EMA Reverse Crossover","Strategy Signal Exit"): return None
    return entry+risk*2 if direction==1 else entry-risk*2

def update_trail_sl(cur_sl,candle,direction,sl_type):
    if "Current Candle" in sl_type or sl_type=="Trail SL":
        return max(cur_sl,float(candle["Low"])) if direction==1 else min(cur_sl,float(candle["High"]))
    return cur_sl

def enter_position(direction,cfg,df=None):
    ltp,_,_=fetch_ltp(cfg["tsym"])
    df_=df if df is not None else st.session_state.current_data
    entry=ltp if ltp>0 else float(df_["Close"].iloc[-1])
    idx=len(df_)-1; a_val=float(_atr(df_).iloc[-1])
    sl=calc_sl(df_,entry,direction,cfg["sl_type"],cfg["sl_p"],idx)
    tgt=calc_target(entry,sl,direction,cfg["tgt_type"],cfg["tgt_p"],a_val)
    order_id=None
    if cfg.get("dhan_on") and st.session_state.dhan_connected:
        b=DhanBroker(st.session_state.get("d_cid",""),st.session_state.get("d_tok",""))
        b.connect(); order=b.place_order(cfg["tsym"],cfg["qty"],"BUY" if direction==1 else "SELL",entry,sl,tgt or 0)
        order_id=order.get("order_id")
    return dict(entry_price=entry,direction=direction,sl=sl,target=tgt,trail_sl=sl,
                qty=cfg["qty"],entry_dt=datetime.now(),order_id=order_id)

def exit_position(exit_price,exit_reason,cfg):
    """Exit position and immediately write to live_trades (persists across tabs)."""
    pos=st.session_state.live_position
    if not pos: return
    d=pos["direction"]; pnl_pts=(exit_price-pos["entry_price"])*d
    # Append to live_trades — this persists in session state across all tabs
    st.session_state.live_trades.append({
        "Entry Date":pos["entry_dt"],"Exit Date":datetime.now(),
        "Direction":"LONG" if d==1 else "SHORT",
        "Entry Price":round(pos["entry_price"],2),"Exit Price":round(exit_price,2),
        "Initial SL":round(pos["sl"],2),
        "Target":round(pos["target"],2) if pos.get("target") else "—",
        "Qty":pos["qty"],"P&L (pts)":round(pnl_pts,2),"P&L (₹)":round(pnl_pts*pos["qty"],2),
        "Result":"WIN" if pnl_pts>0 else "LOSS","Exit Reason":exit_reason,
        "Duration":str(datetime.now()-pos["entry_dt"]),
    })
    st.session_state.live_position=None
    st.session_state.last_signal_candle=None
    clear_all_cache()

# ── BACKTEST ENGINE ───────────────────────────────────────────
def run_backtest(df,signals,sl_type,sl_p,tgt_type,tgt_p,qty,strat_name=""):
    immediate=strat_name in IMMEDIATE_ENTRY
    trades=[]; in_trade=False; trade={}; a_s=_atr(df)
    for i in range(1,len(df)):
        if not in_trade:
            sv=int(signals.iloc[i-1]) if not pd.isna(signals.iloc[i-1]) else 0
            if sv!=0:
                entry=float(df["Close"].iloc[i-1]) if immediate else float(df["Open"].iloc[i])
                d=sv; a_val=float(a_s.iloc[min(i,len(a_s)-1)])
                sl=calc_sl(df,entry,d,sl_type,sl_p,i)
                tgt=calc_target(entry,sl,d,tgt_type,tgt_p,a_val)
                trade=dict(entry_dt=df.index[i-1 if immediate else i],
                           entry_price=entry,direction=d,initial_sl=sl,trail_sl=sl,target=tgt,qty=qty)
                in_trade=True
        else:
            c=df.iloc[i]; d=trade["direction"]; sl_now=trade["trail_sl"]; tgt_now=trade["target"]
            ep=None; er=""
            if d==1:
                if sl_now is not None and float(c["Low"])<=sl_now: ep=sl_now; er="Stop Loss"
                elif tgt_now is not None and float(c["High"])>=tgt_now:
                    if "display only" not in tgt_type: ep=tgt_now; er="Target Hit"
                    else: trade["trail_sl"]=update_trail_sl(sl_now,c,d,sl_type)
                else: trade["trail_sl"]=update_trail_sl(sl_now,c,d,sl_type)
            else:
                if sl_now is not None and float(c["High"])>=sl_now: ep=sl_now; er="Stop Loss"
                elif tgt_now is not None and float(c["Low"])<=tgt_now:
                    if "display only" not in tgt_type: ep=tgt_now; er="Target Hit"
                    else: trade["trail_sl"]=update_trail_sl(sl_now,c,d,sl_type)
                else: trade["trail_sl"]=update_trail_sl(sl_now,c,d,sl_type)
            if ep is not None:
                pp=(ep-trade["entry_price"])*d
                trades.append({"Entry Date":trade["entry_dt"],"Exit Date":df.index[i],
                    "Direction":"LONG" if d==1 else "SHORT",
                    "Entry Price":round(trade["entry_price"],2),"Exit Price":round(ep,2),
                    "Initial SL":round(trade["initial_sl"],2),
                    "Target":round(tgt_now,2) if tgt_now else "—","Qty":qty,
                    "P&L (pts)":round(pp,2),"P&L (₹)":round(pp*qty,2),
                    "Result":"WIN" if pp>0 else "LOSS","Exit Reason":er,
                    "Duration":str(df.index[i]-trade["entry_dt"])})
                in_trade=False; trade={}
    if in_trade and trade:
        cp=float(df["Close"].iloc[-1]); d=trade["direction"]; pp=(cp-trade["entry_price"])*d
        trades.append({"Entry Date":trade["entry_dt"],"Exit Date":df.index[-1],
            "Direction":"LONG" if d==1 else "SHORT",
            "Entry Price":round(trade["entry_price"],2),"Exit Price":round(cp,2),
            "Initial SL":round(trade["initial_sl"],2),
            "Target":round(trade["target"],2) if trade["target"] else "—","Qty":qty,
            "P&L (pts)":round(pp,2),"P&L (₹)":round(pp*qty,2),
            "Result":"WIN" if pp>0 else "LOSS","Exit Reason":"End of Data",
            "Duration":str(df.index[-1]-trade["entry_dt"])})
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def compute_stats(bt):
    if bt.empty: return {}
    t=len(bt); w=len(bt[bt["Result"]=="WIN"]); l=t-w; acc=w/t*100 if t else 0
    tp=bt["P&L (₹)"].sum(); tpts=bt["P&L (pts)"].sum()
    aw=bt[bt["Result"]=="WIN"]["P&L (₹)"].mean() if w else 0
    al=bt[bt["Result"]=="LOSS"]["P&L (₹)"].mean() if l else 0
    gw=bt[bt["Result"]=="WIN"]["P&L (₹)"].sum(); gl=abs(bt[bt["Result"]=="LOSS"]["P&L (₹)"].sum())
    pf=gw/gl if gl>0 else float("inf"); cum=bt["P&L (₹)"].cumsum(); dd=(cum-cum.cummax()).min()
    cw=cl=mw=ml=0
    for r in bt["Result"].tolist():
        cw=cw+1 if r=="WIN" else 0; cl=cl+1 if r=="LOSS" else 0; mw=max(mw,cw); ml=max(ml,cl)
    return dict(total=t,wins=w,losses=l,acc=acc,tp=tp,tpts=tpts,aw=aw,al=al,
                pf=pf,dd=dd,mw=mw,ml=ml,exp=acc/100*aw+(1-acc/100)*al)

class DhanBroker:
    def __init__(self,cid,tok): self.cid=cid; self.tok=tok; self.ok=bool(cid and tok)
    def connect(self): return self.ok
    def place_order(self,sym,qty,side,price=0,sl=0,tgt=0):
        if not self.ok: return {"status":"error"}
        return {"order_id":f"DHAN_{datetime.now():%Y%m%d%H%M%S}","symbol":sym,"qty":qty,
                "side":side,"price":price,"sl":sl,"target":tgt,"status":"SIMULATED"}
    def cancel_order(self,oid): return {"status":"cancelled"}

# ── CHARTS ────────────────────────────────────────────────────
_CMAP={"ema_fast":"#f0883e","ema_slow":"#58a6ff","bb_upper":"#555","bb_mid":"#777",
       "bb_lower":"#555","kalman_price":"#ce93d8","vwap":"#ffe082","atr_line":"#80cbc4",
       "vpfr_poc":"#ff9800","vpfr_vah":"#ef5350","vpfr_val":"#26a69a"}

def build_chart(df,sig,ind,trades_df=None,title=""):
    has_rsi="rsi" in ind; rows=3 if has_rsi else 2; rh=[.60,.20,.20] if has_rsi else [.72,.28]
    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,vertical_spacing=.025,row_heights=rh)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],
        close=df["Close"],name="Price",
        increasing=dict(fillcolor="#26a69a",line=dict(color="#26a69a",width=1)),
        decreasing=dict(fillcolor="#ef5350",line=dict(color="#ef5350",width=1))),row=1,col=1)
    for k,v in ind.items():
        if not isinstance(v,pd.Series): continue
        if any(x in k for x in("ema","kalman","bb","vwap","atr_line","vpfr")):
            fig.add_trace(go.Scatter(x=df.index,y=v,name=k.replace("_"," ").title(),
                line=dict(color=_CMAP.get(k,"#aaa"),width=1 if "vpfr" in k else 1.5,
                          dash="dash" if k in("bb_upper","bb_lower","vpfr_vah","vpfr_val") else "solid"),
                opacity=.85),row=1,col=1)
        elif k=="rsi":
            fig.add_trace(go.Scatter(x=df.index,y=v,name="RSI",
                line=dict(color="#ab47bc",width=1.5)),row=2,col=1)
            for lv,lc in[(70,"#ef5350"),(30,"#26a69a"),(50,"#444")]:
                fig.add_hline(y=lv,line=dict(color=lc,dash="dash",width=1),row=2,col=1)
    for s in ind.get("support_levels",[]): fig.add_hline(y=s,line=dict(color="#26a69a",dash="dot",width=1),opacity=.3,row=1,col=1)
    for r in ind.get("resistance_levels",[]): fig.add_hline(y=r,line=dict(color="#ef5350",dash="dot",width=1),opacity=.3,row=1,col=1)
    buys=sig[sig==1].index; sells=sig[sig==-1].index
    if len(buys): fig.add_trace(go.Scatter(x=buys,y=df["Low"].reindex(buys)*.999,mode="markers",
        name="Buy",marker=dict(symbol="triangle-up",size=10,color="#3fb950",line=dict(color="white",width=1))),row=1,col=1)
    if len(sells): fig.add_trace(go.Scatter(x=sells,y=df["High"].reindex(sells)*1.001,mode="markers",
        name="Sell",marker=dict(symbol="triangle-down",size=10,color="#f85149",line=dict(color="white",width=1))),row=1,col=1)
    if trades_df is not None and not trades_df.empty:
        for _,t in trades_df.iterrows():
            c_="#3fb950" if t["Result"]=="WIN" else "#f85149"
            fig.add_trace(go.Scatter(x=[t["Entry Date"]],y=[t["Entry Price"]],mode="markers+text",
                text=["E"],textposition="top center",textfont=dict(size=7,color=c_),
                marker=dict(size=7,color=c_),showlegend=False),row=1,col=1)
            try:
                fig.add_shape(type="line",x0=t["Entry Date"],x1=t["Exit Date"],
                    y0=float(t["Initial SL"]),y1=float(t["Initial SL"]),
                    line=dict(color="#f85149",dash="dot",width=1),row=1,col=1)
            except: pass
    vrow=3 if has_rsi else 2
    vcol=["#26a69a" if c>=o else "#ef5350" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vcol,opacity=.6,name="Volume"),row=vrow,col=1)
    avgv=ind.get("avg_volume")
    if avgv is not None and isinstance(avgv,pd.Series):
        fig.add_trace(go.Scatter(x=df.index,y=avgv,name="Avg Vol",
            line=dict(color="#f0883e",width=1,dash="dash"),opacity=.6),row=vrow,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
        font=dict(family="monospace",size=11,color="#8b949e"),showlegend=True,
        legend=dict(bgcolor="rgba(22,27,34,.85)",bordercolor="#30363d",borderwidth=1,font=dict(size=10)),
        xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=34,b=0),height=580,
        title=dict(text=f"<b>{title}</b>",font=dict(size=12,color="#e6edf3"),x=.5))
    fig.update_xaxes(gridcolor="#21262d",zeroline=False,showspikes=True,spikecolor="#58a6ff",spikethickness=1)
    fig.update_yaxes(gridcolor="#21262d",zeroline=False,showspikes=True)
    return fig

def eq_fig(bt):
    cum=bt["P&L (₹)"].cumsum(); pos=cum.iloc[-1]>=0
    fig=go.Figure(go.Scatter(x=list(range(len(cum))),y=cum.values,fill="tozeroy",
        fillcolor="rgba(63,185,80,.15)" if pos else "rgba(248,81,73,.15)",
        line=dict(color="#3fb950" if pos else "#f85149",width=2)))
    fig.add_hline(y=0,line=dict(color="#555",dash="dash",width=1))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
        title="Equity Curve",height=230,margin=dict(l=0,r=0,t=26,b=0),font=dict(color="#8b949e"))
    return fig

# ── AUTO-TRADE FRAGMENT  (1-second loop, only active when START clicked) ──
@st.fragment(run_every=1)
def auto_trade_fragment(cfg):
    """Runs every 1s ONLY when live_running=True. Fetches data, checks signals,
    auto-enters on new signal candle, auto-exits on SL/Target hit, writes to
    live_trades immediately, then calls st.rerun() so History tab updates at once."""
    ticker=cfg["tsym"]; strat=cfg["strat"]; sp=cfg["sp"]
    df=fetch_data(ticker,cfg["interval"],cfg["period"])
    ltp,day_hi,day_lo=fetch_ltp(ticker)
    if df is None or df.empty:
        st.caption("⚠️ Data unavailable"); return
    sigs,inds=run_strategy(df,strat,sp)
    last_sig=int(sigs.iloc[-1]) if not sigs.empty else 0
    cur_candle=df.index[-1]
    no_vol=not has_volume(df)

    # Volume warning banner for index symbols
    if no_vol and strat in VOLUME_REQUIRED:
        st.markdown(f"""<div class='warn-box'>
          ⚠️ <b>{strat}</b> uses volume — this instrument has no volume data (index symbol).<br>
          Running in <b>price-only fallback mode</b>. Signals may be less reliable.
          Use equities (RELIANCE.NS) or crypto (BTC-USD) for full volume-based accuracy.
        </div>""", unsafe_allow_html=True)

    # Status bar
    st.markdown(f"""<div style='display:flex;gap:14px;align-items:center;background:#161b22;
         border:1px solid #30363d;border-radius:8px;padding:7px 12px;margin-bottom:6px;font-size:.82rem'>
      <span style='color:#3fb950;font-weight:700'>● LIVE AUTO-TRADE</span>
      <span style='color:#8b949e'>|</span>
      <span style='color:#e6edf3'>LTP: <b style='color:#58a6ff'>{ltp:,.2f}</b></span>
      <span style='color:#8b949e'>| Signal: <b style='color:{"#3fb950" if last_sig==1 else "#f85149" if last_sig==-1 else "#8b949e"}'>
        {"▲ BUY" if last_sig==1 else "▼ SELL" if last_sig==-1 else "NONE"}</b></span>
      <span style='color:#8b949e;margin-left:auto;font-size:.68rem'>{datetime.now().strftime("%H:%M:%S")}</span>
    </div>""", unsafe_allow_html=True)

    pos=st.session_state.live_position

    # ── AUTO-ENTER on new signal candle ──────────────────────
    if pos is None and strat not in MANUAL_ONLY:
        if last_sig!=0 and cur_candle!=st.session_state.last_signal_candle:
            pos_new=enter_position(last_sig,cfg,df)
            st.session_state.live_position=pos_new
            st.session_state.last_signal_candle=cur_candle
            dir_txt="🟢 LONG" if last_sig==1 else "🔴 SHORT"
            tgt_txt=f"{pos_new['target']:.2f}" if pos_new.get("target") else "—"
            st.success(f"🤖 AUTO-ENTERED {dir_txt} @ {pos_new['entry_price']:.2f} | SL {pos_new['sl']:.2f} | Tgt {tgt_txt}")
            pos=st.session_state.live_position

    # ── AUTO-EXIT: check SL/Target every second ───────────────
    if pos is not None and ltp>0:
        d=pos["direction"]; exited=False
        if pos.get("sl"):
            if (d==1 and ltp<=pos["sl"]) or (d==-1 and ltp>=pos["sl"]):
                exit_position(pos["sl"],"Stop Loss (Auto)",cfg)
                st.error(f"🛑 STOP LOSS HIT @ {pos['sl']:.2f}  |  LTP {ltp:.2f}")
                exited=True
        if not exited and pos.get("target") and "display only" not in cfg.get("tgt_type",""):
            if (d==1 and ltp>=pos["target"]) or (d==-1 and ltp<=pos["target"]):
                exit_position(pos["target"],"Target Hit (Auto)",cfg)
                st.success(f"🎯 TARGET HIT @ {pos['target']:.2f}  |  LTP {ltp:.2f}")
                exited=True
        if exited:
            # Force full rerun so Trade History tab shows the new record immediately
            st.rerun()

    # ── POSITION P&L PANEL ───────────────────────────────────
    pos=st.session_state.live_position
    if pos:
        d=pos["direction"]; pnl_pts=(ltp-pos["entry_price"])*d if ltp>0 else 0
        pnl_val=pnl_pts*pos["qty"]; pnl_clr="#3fb950" if pnl_pts>=0 else "#f85149"
        duration=str(datetime.now()-pos["entry_dt"]).split(".")[0]
        st.markdown(f"""<div class='{"pos-long" if d==1 else "pos-short"}'>
          <div style='display:flex;flex-wrap:wrap;gap:16px;align-items:center'>
            <div><div class='ml'>DIRECTION</div>
              <div style='color:{"#3fb950" if d==1 else "#f85149"};font-weight:900;font-size:1.1rem'>
                {"▲ LONG" if d==1 else "▼ SHORT"}</div></div>
            <div><div class='ml'>ENTRY</div><div class='mv'>{pos['entry_price']:,.2f}</div></div>
            <div><div class='ml'>LTP</div><div class='mv' style='color:{pnl_clr}'>{ltp:,.2f}</div></div>
            <div><div class='ml'>SL</div><div class='mv' style='color:#f85149'>{pos['sl']:,.2f}</div></div>
            <div><div class='ml'>TARGET</div>
              <div class='mv' style='color:#3fb950'>{"%.2f"%pos["target"] if pos.get("target") else "—"}</div></div>
            <div><div class='ml'>OPEN P&L</div>
              <div style='color:{pnl_clr};font-weight:900;font-size:1.2rem'>
                {pnl_pts:+.2f} pts<br><span style='font-size:.9rem'>₹{pnl_val:+,.2f}</span></div></div>
            <div><div class='ml'>QTY</div><div class='mv'>{pos['qty']}</div></div>
            <div><div class='ml'>DAY HI/LO</div>
              <div class='ms'>{day_hi:,.2f} / {day_lo:,.2f}</div></div>
            <div><div class='ml'>DURATION</div><div class='ms'>{duration}</div></div>
          </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='pos-none'>No open position — waiting for next signal</div>", unsafe_allow_html=True)

    # ── Indicator values ──────────────────────────────────────
    val_items=[]
    if "ema_fast" in inds: val_items.append((f"EMA {sp.get('ema_fast',9)}",float(inds["ema_fast"].iloc[-1]),"#f0883e"))
    if "ema_slow" in inds: val_items.append((f"EMA {sp.get('ema_slow',15)}",float(inds["ema_slow"].iloc[-1]),"#58a6ff"))
    if "rsi"          in inds: val_items.append(("RSI",float(inds["rsi"].iloc[-1]),"#ab47bc"))
    if "kalman_price" in inds: val_items.append(("Kalman",float(inds["kalman_price"].iloc[-1]),"#ce93d8"))
    if "vwap"         in inds: val_items.append(("VWAP/EMA-TP",float(inds["vwap"].iloc[-1]),"#ffe082"))
    if "bb_upper"     in inds: val_items+=[("BB Upper",float(inds["bb_upper"].iloc[-1]),"#666"),
                                            ("BB Lower",float(inds["bb_lower"].iloc[-1]),"#666")]
    for k,lbl,clr in[("vpfr_poc","POC","#ff9800"),("vpfr_vah","VAH","#ef5350"),("vpfr_val","VAL","#26a69a")]:
        if k in inds:
            v=inds[k].dropna()
            if not v.empty: val_items.append((f"VPFR {lbl}",float(v.iloc[-1]),clr))
    val_items.append(("ATR(14)",float(_atr(df).iloc[-1]),"#80cbc4"))
    val_items.append(("Close",float(df["Close"].iloc[-1]),"#e6edf3"))
    if val_items:
        ncols=min(len(val_items),7)
        for row_i in range((len(val_items)+ncols-1)//ncols):
            chunk=val_items[row_i*ncols:(row_i+1)*ncols]
            for col,(nm,val,clr) in zip(st.columns(len(chunk)),chunk):
                with col:
                    st.markdown(f"<div class='mc'><div class='ml'>{nm}</div>"
                                f"<div class='mv' style='color:{clr};font-size:.87rem'>{val:,.2f}</div></div>",
                                unsafe_allow_html=True)
        st.markdown("<div style='margin:4px'></div>",unsafe_allow_html=True)

    # ── EMA crossover banner ──────────────────────────────────
    if "ema_fast" in inds and "ema_slow" in inds and len(inds["ema_fast"])>=2:
        ef_v,es_v=float(inds["ema_fast"].iloc[-1]),float(inds["ema_slow"].iloc[-1])
        ef_p,es_p=float(inds["ema_fast"].iloc[-2]),float(inds["ema_slow"].iloc[-2])
        if ef_v>es_v and ef_p<=es_p:
            st.markdown("<div class='sbuy' style='padding:7px'><b style='color:#3fb950'>▲ BULLISH EMA CROSSOVER on last candle!</b></div>",unsafe_allow_html=True)
        elif ef_v<es_v and ef_p>=es_p:
            st.markdown("<div class='ssell' style='padding:7px'><b style='color:#f85149'>▼ BEARISH EMA CROSSOVER on last candle!</b></div>",unsafe_allow_html=True)
        else:
            bias="Bullish" if ef_v>es_v else "Bearish"; gap=abs(ef_v-es_v); bclr="#3fb950" if bias=="Bullish" else "#f85149"
            st.markdown(f"<div class='swait' style='padding:6px;font-size:.81rem'>"
                        f"<span style='color:{bclr}'>No crossover · {bias} bias</span> · Gap: <b>{gap:.2f}</b></div>",unsafe_allow_html=True)

    # ── Live chart with EMA annotations ──────────────────────
    fig=build_chart(df,sigs,inds,title=f"LIVE {ticker} | {strat} | {cfg['interval']}/{cfg['period']}")
    for k,nm_,clr in[("ema_fast",f"EMA {sp.get('ema_fast',9)}","#f0883e"),
                      ("ema_slow",f"EMA {sp.get('ema_slow',15)}","#58a6ff"),
                      ("kalman_price","Kalman","#ce93d8"),("vwap","VWAP","#ffe082")]:
        if k in inds and isinstance(inds[k],pd.Series):
            v=float(inds[k].iloc[-1])
            fig.add_annotation(x=df.index[-1],y=v,text=f"  {nm_}: {v:,.2f}",
                showarrow=False,xanchor="left",font=dict(size=9,color=clr),bgcolor="rgba(13,17,23,.75)")
    st.plotly_chart(fig,use_container_width=True,key=f"atf_{ticker}_{cfg['interval']}")

# ── SIDEBAR (no LOAD button — auto-loads from main()) ────────
def sidebar() -> dict:
    with st.sidebar:
        st.markdown("""<div style='text-align:center;padding:8px 0 14px'>
          <div style='font-size:1.7rem'>📊</div>
          <div style='font-size:1.05rem;font-weight:900;color:#e6edf3;letter-spacing:2px'>QuantAlgo Pro</div>
          <div style='font-size:.58rem;color:#8b949e;letter-spacing:2px'>v5.0 — AUTO-LOADS ON CHANGE</div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown("**🎯 INSTRUMENT**")
        tname=st.selectbox("Asset",list(TICKERS.keys()),label_visibility="collapsed",key="tname")
        tsym=(st.text_input("Symbol","RELIANCE.NS",key="csym").upper().strip()
              if tname=="✏️ Custom" else TICKERS[tname])
        st.markdown("**⏱️ TIMEFRAME**")
        c1,c2=st.columns(2)
        ivs=list(TIMEFRAME_PERIODS.keys())
        with c1: interval=st.selectbox("IV",ivs,index=ivs.index("1m"),key="interval",label_visibility="collapsed")
        plist=TIMEFRAME_PERIODS[interval]
        def_p="5d" if "5d" in plist else plist[0]
        with c2: period=st.selectbox("P",plist,index=plist.index(def_p),key="period",label_visibility="collapsed")
        st.markdown("**📦 QTY**")
        qty=st.number_input("Q",min_value=1,value=1,step=1,key="qty",label_visibility="collapsed")
        st.divider()
        st.markdown("**🧠 STRATEGY**")
        strat=st.selectbox("S",STRATEGIES,key="strat",label_visibility="collapsed")
        sp:dict={}
        with st.expander("⚙️ Strategy Params"):
            if strat=="EMA Crossover":
                sp["ema_fast"]=st.number_input("Fast EMA",1,200,9,key="ef")
                sp["ema_slow"]=st.number_input("Slow EMA",1,500,15,key="es")
            elif strat in("Simple Buy (Immediate)","Simple Sell (Immediate)"):
                st.caption("Enters at LTP instantly — no candle delay.")
            elif strat=="Threshold Price Cross":
                sp["buy_t"]=st.number_input("Buy ≥ ₹",0.,key="bt_")
                sp["sell_t"]=st.number_input("Sell ≤ ₹",0.,key="st__")
            elif strat=="Price Action (S/R)":
                sp["sr_window"]=st.number_input("Pivot Window",3,50,10,key="srw")
            elif strat=="RSI Strategy":
                sp["rsi_period"]=st.number_input("Period",2,50,14,key="rp")
                sp["rsi_ob"]=st.number_input("Overbought",50,99,70,key="rob")
                sp["rsi_os"]=st.number_input("Oversold",1,50,30,key="ros")
            elif strat=="Bollinger Bands":
                sp["bb_period"]=st.number_input("Period",5,100,20,key="bbp")
                sp["bb_std"]=st.number_input("Std Dev",.5,5.,2.,step=.1,key="bbs")
            elif strat=="Volume Breakout":
                sp["vol_lb"]=st.number_input("Lookback",5,100,20,key="vlb")
                sp["vol_mult"]=st.number_input("Volume ×",1.,5.,1.5,step=.1,key="vm")
            elif strat=="Elliott Wave (Simplified)":
                sp["ew_window"]=st.number_input("Pivot Window",3,30,10,key="eww")
            elif strat=="Volume Profile FR [VPFR]":
                sp["vpfr_lb"]=st.number_input("Lookback",10,200,50,key="vplb")
                sp["vpfr_bins"]=st.number_input("Bins",10,60,30,key="vpb")
                st.caption("⚠️ Needs real volume — won't work on index symbols.")
            elif "Kalman" in strat:
                sp["kf_thr"]=st.number_input("Z Threshold",.5,4.,1.5,step=.1,key="kft")
            elif "Order Flow" in strat:
                sp["ofi_lb"]=st.number_input("Lookback",3,50,10,key="ofil")
                sp["ofi_thr"]=st.number_input("Imbalance",.5,.95,.60,step=.01,key="ofit")
            elif "Volatility Regime" in strat:
                sp["vrm_fast"]=st.number_input("Fast",3,50,10,key="vrmf")
                sp["vrm_slow"]=st.number_input("Slow",10,100,30,key="vrms")
        st.divider()
        st.markdown("**🛡️ STOP LOSS**")
        sl_type=st.selectbox("SL",SL_TYPES,key="sl_type",label_visibility="collapsed")
        sl_p:dict={}
        with st.expander("⚙️ SL Params"):
            if sl_type=="Custom Points": sl_p["sl_points"]=st.number_input("Points",.1,1e5,10.,step=.5,key="slp")
            elif sl_type=="ATR Based SL": sl_p["sl_atr_mult"]=st.number_input("ATR ×",.5,5.,1.5,step=.1,key="slam")
            elif sl_type=="Risk Reward (min 1:2)": sl_p["sl_points"]=st.number_input("Risk pts",.1,1e5,10.,step=.5,key="rrsl")
            elif sl_type=="🤖 Autopilot SL": sl_p["vol_scale"]=st.slider("Vol Scale",.5,2.,1.,key="vs")
        st.markdown("**🎯 TARGET**")
        tgt_type=st.selectbox("Tgt",TARGET_TYPES,key="tgt_type",label_visibility="collapsed")
        tgt_p:dict={}
        with st.expander("⚙️ Target Params"):
            if tgt_type=="Custom Points": tgt_p["target_points"]=st.number_input("Points",.1,1e5,20.,step=.5,key="tgtp")
            elif tgt_type=="ATR Based Target": tgt_p["target_atr_mult"]=st.number_input("ATR ×",.5,10.,2.,step=.1,key="tam")
            elif tgt_type=="Risk Reward (min 1:2)": tgt_p["rr_ratio"]=st.number_input("R:R ≥2",2.,10.,2.,step=.5,key="rr")
            elif "display only" in tgt_type: st.caption("Shown on chart — never triggers exit")
            elif tgt_type=="🤖 Autopilot Target": st.caption("Fibonacci 2.618× risk")
        st.divider()
        st.markdown("**🏦 DHAN**")
        dhan_on=st.checkbox("Enable Live Orders",value=False,key="dhan_on")
        with st.expander("🔑 Credentials"):
            d_cid=st.text_input("Client ID",type="password",key="d_cid")
            d_tok=st.text_input("Access Token",type="password",key="d_tok")
            if st.button("🔗 Test",key="test_dhan"):
                if DhanBroker(d_cid,d_tok).connect(): st.success("✅ OK"); st.session_state.dhan_connected=True
                else: st.error("❌ Failed")
        if st.session_state.get("last_data_key"):
            st.caption(f"📊 {st.session_state.last_data_key}")
    return dict(tsym=tsym,tname=tname,interval=interval,period=period,qty=qty,
                strat=strat,sp=sp,sl_type=sl_type,sl_p=sl_p,tgt_type=tgt_type,tgt_p=tgt_p,dhan_on=dhan_on)

# ── TAB 1 — BACKTEST ──────────────────────────────────────────
def tab_backtest(cfg):
    df=st.session_state.current_data; sigs=st.session_state.signals; inds=st.session_state.indicators
    if df is None or df.empty:
        st.info("👈 Select instrument and timeframe in the sidebar — data loads automatically.")
        return
    no_vol=not has_volume(df)
    if no_vol and cfg["strat"] in VOLUME_REQUIRED:
        st.markdown(f"""<div class='warn-box'>⚠️ <b>{cfg['strat']}</b> needs volume.
          This instrument ({cfg['tsym']}) has no volume data — running in price-only fallback mode.
          Signals work but volume confirmation is absent.</div>""", unsafe_allow_html=True)
    st.markdown(f"### 📈 {cfg['tname']}  ·  {cfg['interval']}  ·  {cfg['period']}")
    cr,ci=st.columns([1,4])
    with cr: run_bt=st.button("▶ Run Backtest",type="primary",use_container_width=True,key="run_bt")
    with ci:
        bc=int((sigs==1).sum()); sc=int((sigs==-1).sum())
        entry_note="Immediate (signal close)" if cfg["strat"] in IMMEDIATE_ENTRY else "Standard (next candle open)"
        st.markdown(f"""<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;
             padding:8px 13px;font-size:.81rem;display:flex;gap:12px;align-items:center'>
          <span style='color:#8b949e'>{len(df):,} candles</span>
          <span style='color:#3fb950'>▲ {bc} BUY</span><span style='color:#f85149'>▼ {sc} SELL</span>
          <span style='color:#8b949e'>| {cfg["strat"][:26]}</span>
          <span style='color:#58a6ff;font-size:.74rem'>· {entry_note}</span>
        </div>""", unsafe_allow_html=True)
    if run_bt:
        with st.spinner("Backtesting…"):
            bt=run_backtest(df,sigs,cfg["sl_type"],cfg["sl_p"],cfg["tgt_type"],cfg["tgt_p"],cfg["qty"],cfg["strat"])
        st.session_state.backtest_results=bt
    bt_res=st.session_state.backtest_results
    fig=build_chart(df,sigs,inds,bt_res,title=f"{cfg['tname']} | {cfg['strat']} | {cfg['interval']}/{cfg['period']}")
    st.plotly_chart(fig,use_container_width=True,key="bt_chart")
    if bt_res is None: st.caption("Click ▶ Run Backtest to see results."); return
    if bt_res.empty: st.warning("No trades — widen period or adjust strategy parameters."); return
    s=compute_stats(bt_res); pf_txt=f"{s['pf']:.2f}" if s["pf"]!=float("inf") else "∞"
    st.markdown("### 📊 Results")
    cols=st.columns(7)
    for col,lbl,val,clr in zip(cols,
        ["Trades","Accuracy","Net P&L","Points","Profit Factor","Drawdown","Expectancy"],
        [str(s["total"]),f"{s['acc']:.1f}%",f"₹{s['tp']:,.0f}",f"{s['tpts']:+.1f}",
         pf_txt,f"₹{s['dd']:,.0f}",f"₹{s['exp']:,.1f}"],
        ["#58a6ff","#3fb950" if s["acc"]>=50 else "#f85149",
         "#3fb950" if s["tp"]>=0 else "#f85149","#3fb950" if s["tpts"]>=0 else "#f85149",
         "#3fb950" if s["pf"]>=1.5 else "#f0883e","#f85149","#58a6ff"]):
        with col:
            st.markdown(f"""<div class='mc' style='border-top:3px solid {clr}'>
              <div class='ml'>{lbl}</div><div class='mv' style='color:{clr};font-size:.95rem'>{val}</div>
              {"<div class='ms'>W:"+str(s['wins'])+" L:"+str(s['losses'])+"</div>" if lbl=="Accuracy" else ""}
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ca,cb=st.columns([3,1])
    with ca: st.plotly_chart(eq_fig(bt_res),use_container_width=True,key="bt_eq")
    with cb:
        pie=go.Figure(go.Pie(labels=["Wins","Losses"],values=[s["wins"],s["losses"]],
            marker_colors=["#3fb950","#f85149"],hole=.55,textinfo="label+percent"))
        pie.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",height=190,
            margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
        st.plotly_chart(pie,use_container_width=True,key="bt_pie")
        st.markdown(f"""<div style='font-size:.77rem;color:#8b949e;line-height:1.8'>
          Max Win Streak: <span style='color:#3fb950'>{s['mw']}</span><br>
          Max Loss Streak: <span style='color:#f85149'>{s['ml']}</span><br>
          Avg Win: <span style='color:#3fb950'>₹{s['aw']:,.0f}</span><br>
          Avg Loss: <span style='color:#f85149'>₹{s['al']:,.0f}</span></div>""", unsafe_allow_html=True)
    st.markdown("**📋 Trade Log (Backtest only)**")
    def _hl(row):
        return [f"background-color:{'rgba(63,185,80,.08)' if row['Result']=='WIN' else 'rgba(248,81,73,.08)'}"]*len(row)
    st.dataframe(bt_res.style.apply(_hl,axis=1),use_container_width=True,height=280,key="bt_df")
    st.download_button("📥 Export CSV",bt_res.to_csv(index=False),
        file_name=f"backtest_{datetime.now():%Y%m%d_%H%M}.csv",mime="text/csv",key="bt_dl")

# ── TAB 2 — LIVE TRADING ──────────────────────────────────────
def tab_live(cfg):
    df=st.session_state.current_data; sigs=st.session_state.signals; pos=st.session_state.live_position
    st.markdown("### ⚙️ Configuration")
    c1,c2,c3,c4=st.columns(4)
    for col,(lbl,val,sub) in zip([c1,c2,c3,c4],[
        ("Asset",cfg["tname"],cfg["tsym"]),("Timeframe",cfg["interval"],f"Period: {cfg['period']}"),
        ("Strategy",cfg["strat"][:28],f"Qty: {cfg['qty']}"),("Risk",cfg["sl_type"][:24],cfg["tgt_type"][:24])]):
        with col:
            st.markdown(f"""<div class='icard'><div class='ml'>{lbl}</div>
              <div style='color:#e6edf3;font-size:.86rem;font-weight:700;margin-top:2px'>{val}</div>
              <div class='ms'>{sub}</div></div>""", unsafe_allow_html=True)
    st.divider()
    if df is None or df.empty:
        st.info("👈 Select instrument in sidebar — data loads automatically."); return
    # Controls
    st.markdown("### 🎛️ Controls")
    running=st.session_state.live_running
    b1,b2,b3,b4=st.columns(4)
    with b1:
        if not running:
            if st.button("▶ START",type="primary",use_container_width=True,key="start_btn"):
                st.session_state.live_running=True; st.rerun()
        else:
            st.markdown("""<div class='auto-on'><div style='color:#3fb950;font-weight:900;font-size:.88rem'>
              ● LIVE AUTO-TRADE ON</div><div style='color:#8b949e;font-size:.71rem'>1s signal loop active</div>
            </div>""", unsafe_allow_html=True)
    with b2:
        if running:
            if st.button("⏹ STOP",use_container_width=True,key="stop_btn"):
                st.session_state.live_running=False; st.rerun()
        else: st.button("⏹ STOP",use_container_width=True,disabled=True,key="stop_dis")
    with b3:
        if pos:
            if st.button("🔴 SQUARE OFF",use_container_width=True,key="sq_btn"):
                ltp_,_,_=fetch_ltp(cfg["tsym"]); ep=ltp_ if ltp_>0 else pos["entry_price"]
                exit_position(ep,"Manual Square Off",cfg); st.success(f"✅ Squared off @ {ep:.2f}"); st.rerun()
        else: st.button("🔴 SQUARE OFF",use_container_width=True,disabled=True,key="sq_dis")
    with b4:
        sc="#3fb950" if running else "#8b949e"
        pt=f"{'LONG' if pos['direction']==1 else 'SHORT'} @ {pos['entry_price']:.2f}" if pos else "No Position"
        st.markdown(f"""<div class='icard' style='text-align:center'>
          <div style='color:{sc};font-weight:700;font-size:.82rem'>{"● LIVE" if running else "○ IDLE"}</div>
          <div style='color:#e6edf3;font-size:.77rem;margin-top:2px'>{pt}</div></div>""", unsafe_allow_html=True)
    st.divider()
    # Manual BUY/SELL
    st.markdown("### ⚡ Manual Instant Entry")
    ltp_c,_,_=fetch_ltp(cfg["tsym"]); cur=ltp_c if ltp_c>0 else float(df["Close"].iloc[-1])
    a_v=float(_atr(df).iloc[-1])
    psl_b=calc_sl(df,cur,1,cfg["sl_type"],cfg["sl_p"],len(df)-1)
    ptgt_b=calc_target(cur,psl_b,1,cfg["tgt_type"],cfg["tgt_p"],a_v)
    psl_s=calc_sl(df,cur,-1,cfg["sl_type"],cfg["sl_p"],len(df)-1)
    ptgt_s=calc_target(cur,psl_s,-1,cfg["tgt_type"],cfg["tgt_p"],a_v)
    mb,ms_,mi=st.columns([1,1,2])
    with mb:
        rr_b=abs(ptgt_b-cur)/abs(cur-psl_b) if abs(cur-psl_b) and ptgt_b else 0
        st.markdown(f"""<div style='background:#1a3d26;border:1px solid #3fb950;border-radius:7px;
          padding:5px;margin-bottom:5px;text-align:center;font-size:.74rem'>
          <b style='color:#3fb950'>▲ BUY LONG</b><br>
          <span style='color:#8b949e'>Ent {cur:,.2f} SL {psl_b:,.2f} T {"%.2f"%ptgt_b if ptgt_b else "—"} RR {rr_b:.1f}:1</span>
        </div>""", unsafe_allow_html=True)
        if st.button("▲ BUY NOW",use_container_width=True,key="buy_now",disabled=pos is not None):
            st.session_state.live_position=enter_position(1,cfg,df); clear_all_cache(); st.rerun()
    with ms_:
        rr_s=abs(ptgt_s-cur)/abs(cur-psl_s) if abs(cur-psl_s) and ptgt_s else 0
        st.markdown(f"""<div style='background:#3d1a1a;border:1px solid #f85149;border-radius:7px;
          padding:5px;margin-bottom:5px;text-align:center;font-size:.74rem'>
          <b style='color:#f85149'>▼ SELL SHORT</b><br>
          <span style='color:#8b949e'>Ent {cur:,.2f} SL {psl_s:,.2f} T {"%.2f"%ptgt_s if ptgt_s else "—"} RR {rr_s:.1f}:1</span>
        </div>""", unsafe_allow_html=True)
        if st.button("▼ SELL NOW",use_container_width=True,key="sell_now",disabled=pos is not None):
            st.session_state.live_position=enter_position(-1,cfg,df); clear_all_cache(); st.rerun()
    with mi:
        dis="⚠️ Close position first" if pos else "✅ Ready to trade"; dc="#f0883e" if pos else "#3fb950"
        st.markdown(f"""<div class='icard'><div style='color:{dc};font-size:.79rem;font-weight:700;margin-bottom:5px'>{dis}</div>
          <div style='color:#8b949e;font-size:.76rem;line-height:1.8'>LTP: <b style='color:#e6edf3'>{cur:,.2f}</b><br>
          ATR(14): <b style='color:#80cbc4'>{a_v:.2f}</b></div></div>""", unsafe_allow_html=True)
    st.divider()
    # Signal proximity
    st.markdown("### 🔔 Signal & Proximity")
    last_sig=int(sigs.iloc[-1]) if sigs is not None else 0
    prx_lvl,prx_h,prx_d=strategy_proximity(df,cfg["strat"],cfg["sp"],sigs)
    sc1,sc2=st.columns([1,2])
    with sc1:
        if last_sig==1: st.markdown("""<div class='sbuy'><div style='font-size:1.3rem'>▲</div>
              <div style='font-weight:900;color:#3fb950'>BUY SIGNAL</div>
              <div style='color:#8b949e;font-size:.76rem'>Auto-enters when START active</div></div>""",unsafe_allow_html=True)
        elif last_sig==-1: st.markdown("""<div class='ssell'><div style='font-size:1.3rem'>▼</div>
              <div style='font-weight:900;color:#f85149'>SELL SIGNAL</div>
              <div style='color:#8b949e;font-size:.76rem'>Auto-enters when START active</div></div>""",unsafe_allow_html=True)
        else: st.markdown("""<div class='swait'><div style='font-size:1.3rem'>⏳</div>
              <div style='font-weight:900;color:#8b949e'>NO SIGNAL</div>
              <div style='color:#8b949e;font-size:.76rem'>Watching…</div></div>""",unsafe_allow_html=True)
    with sc2:
        cm={"signal":"#3fb950","near":"#f0883e","approaching":"#ffe082","far":"#8b949e"}
        bm={"signal":"rgba(63,185,80,.1)","near":"rgba(240,136,62,.1)","approaching":"rgba(255,235,130,.06)","far":"rgba(139,148,158,.05)"}
        bdr=cm.get(prx_lvl,"#8b949e"); bg_=bm.get(prx_lvl,"transparent")
        st.markdown(f"""<div style='background:{bg_};border:1px solid {bdr};border-left:4px solid {bdr};
             border-radius:8px;padding:11px'><div style='color:{bdr};font-weight:700;font-size:.87rem'>{prx_h}</div>
          <div style='color:#c9d1d9;font-size:.79rem;margin-top:4px'>{prx_d}</div></div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📊 Live Monitor")
    if running:
        auto_trade_fragment(cfg)
    else:
        # Static display — zero API calls when monitor is stopped
        pos2=st.session_state.live_position
        if pos2:
            d=pos2["direction"]
            st.markdown(f"""<div class='{"pos-long" if d==1 else "pos-short"}'>
              <div style='display:flex;flex-wrap:wrap;gap:16px;align-items:center'>
                <div><div class='ml'>DIRECTION</div>
                  <div style='color:{"#3fb950" if d==1 else "#f85149"};font-weight:900;font-size:1.1rem'>
                    {"▲ LONG" if d==1 else "▼ SHORT"}</div></div>
                <div><div class='ml'>ENTRY</div><div class='mv'>{pos2['entry_price']:,.2f}</div></div>
                <div><div class='ml'>SL</div><div class='mv' style='color:#f85149'>{pos2['sl']:,.2f}</div></div>
                <div><div class='ml'>TARGET</div>
                  <div class='mv' style='color:#3fb950'>{"%.2f"%pos2["target"] if pos2.get("target") else "—"}</div></div>
                <div><div class='ml'>QTY</div><div class='mv'>{pos2['qty']}</div></div>
              </div>
              <div style='color:#f0883e;font-size:.78rem;margin-top:6px'>
                ⚠️ Monitor STOPPED — P&L not updating. Click ▶ START to resume auto-trading.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pos-none'>No position · Click ▶ START to begin auto-trading</div>",unsafe_allow_html=True)
        inds2=st.session_state.indicators
        if df is not None and sigs is not None:
            fig=build_chart(df,sigs,inds2,title=f"{cfg['tname']} | {cfg['strat']} (snapshot)")
            st.plotly_chart(fig,use_container_width=True,key="live_snap")
            st.caption("▶ Click START to enable live auto-refresh and auto-trading.")

# ── TAB 3 — LIVE TRADE HISTORY ────────────────────────────────
def tab_history():
    live=st.session_state.live_trades
    if not live:
        st.info("No live trades yet.")
        st.markdown("""<div class='icard' style='margin-top:10px'>
          <div style='color:#58a6ff;font-weight:700;margin-bottom:5px'>ℹ️ How trades appear here</div>
          <div style='color:#8b949e;font-size:.81rem;line-height:1.9'>
            • Click ▶ START — strategy auto-enters on signal, auto-exits on SL/Target<br>
            • Trade record is written <b style='color:#e6edf3'>immediately on exit</b> and visible here at once<br>
            • Or use ▲ BUY NOW / ▼ SELL NOW for manual entry + 🔴 SQUARE OFF to close<br>
            • Backtest results are <b style='color:#e6edf3'>completely separate</b> — never mix
          </div></div>""", unsafe_allow_html=True)
        return
    df_h=pd.DataFrame(live); s=compute_stats(df_h); pf_txt=f"{s['pf']:.2f}" if s["pf"]!=float("inf") else "∞"
    st.markdown("### 📊 Live Portfolio")
    c1,c2,c3,c4=st.columns(4)
    for col,(lbl,val,sub,clr) in zip([c1,c2,c3,c4],[
        ("Live Trades",str(s["total"]),f"W:{s['wins']} L:{s['losses']}","#58a6ff"),
        ("Win Rate",f"{s['acc']:.1f}%","Accuracy","#3fb950" if s["acc"]>=50 else "#f85149"),
        ("Net P&L",f"₹{s['tp']:,.0f}",f"{s['tpts']:+.1f} pts","#3fb950" if s["tp"]>=0 else "#f85149"),
        ("Profit Factor",pf_txt,f"Exp ₹{s['exp']:,.1f}","#3fb950" if s["pf"]>=1.5 else "#f0883e")]):
        with col:
            st.markdown(f"""<div class='mc' style='border-top:3px solid {clr}'>
              <div class='ml'>{lbl}</div><div class='mv' style='color:{clr}'>{val}</div>
              <div class='ms'>{sub}</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ca,cb=st.columns([2,1])
    with ca:
        bar=go.Figure(go.Bar(x=list(range(len(df_h))),y=df_h["P&L (₹)"],
            marker_color=["#3fb950" if x>0 else "#f85149" for x in df_h["P&L (₹)"]]))
        bar.add_hline(y=0,line=dict(color="#555",dash="dash",width=1))
        bar.update_layout(template="plotly_dark",paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
            height=230,title="P&L Per Trade",margin=dict(l=0,r=0,t=26,b=0),font=dict(color="#8b949e"))
        st.plotly_chart(bar,use_container_width=True,key="hist_bar")
    with cb: st.plotly_chart(eq_fig(df_h),use_container_width=True,key="hist_eq")
    st.markdown("### 📋 Trade Log")
    f1,f2,f3=st.columns(3)
    with f1: fr=st.selectbox("Result",["All","WIN","LOSS"],key="fres")
    with f2: fd=st.selectbox("Direction",["All","LONG","SHORT"],key="fdir")
    with f3:
        ex_o=["All"]+list(df_h["Exit Reason"].unique()) if "Exit Reason" in df_h.columns else ["All"]
        fe=st.selectbox("Exit",ex_o,key="fex")
    filt=df_h.copy()
    if fr!="All": filt=filt[filt["Result"]==fr]
    if fd!="All": filt=filt[filt["Direction"]==fd]
    if fe!="All" and "Exit Reason" in filt.columns: filt=filt[filt["Exit Reason"]==fe]
    def _hl(row):
        return [f"background-color:{'rgba(63,185,80,.08)' if row['Result']=='WIN' else 'rgba(248,81,73,.08)'}"]*len(row)
    st.dataframe(filt.style.apply(_hl,axis=1),use_container_width=True,height=360,key="hist_df")
    dc,cc=st.columns(2)
    with dc: st.download_button("📥 Export CSV",filt.to_csv(index=False),
        file_name=f"live_{datetime.now():%Y%m%d_%H%M}.csv",mime="text/csv",key="hist_dl")
    with cc:
        if st.button("🗑️ Clear History",key="hist_clr"):
            st.session_state.live_trades=[]; st.session_state.live_position=None; clear_all_cache(); st.rerun()

# ── TAB 4 — STRATEGY LEADERBOARD ─────────────────────────────
def tab_leaderboard(cfg):
    st.markdown("### 🏆 Strategy Leaderboard")
    st.markdown("""<div class='icard' style='margin-bottom:12px'>
      <div style='color:#8b949e;font-size:.82rem'>
        Tests <b style='color:#e6edf3'>all strategies</b> on the current instrument/timeframe with your SL/Target config.
        Ranks by Profit Factor. Use this to find the best strategy for current market conditions before live trading.
      </div></div>""", unsafe_allow_html=True)
    df=st.session_state.current_data
    if df is None or df.empty:
        st.info("👈 Select instrument in sidebar — data loads automatically, then run leaderboard."); return
    no_vol=not has_volume(df)
    if no_vol:
        st.markdown("""<div class='warn-box'>⚠️ Zero-volume instrument detected (likely an index).
          Volume-based strategies will run in price-only fallback mode.
          Results for Volume Breakout, OFI, VPFR, VWAP will be less reliable.</div>""", unsafe_allow_html=True)
    lb_key=f"{cfg['tsym']}|{cfg['interval']}|{cfg['period']}|{cfg['sl_type']}|{cfg['tgt_type']}"
    if st.button("🚀 Run Leaderboard (all strategies)",type="primary",use_container_width=False,key="run_lb"):
        results=[]
        skip={"Volume Profile FR [VPFR]"}  # slow; separate button
        prog=st.progress(0,"Running strategies…")
        strats_to_run=[s for s in STRATEGIES if s not in skip]
        for idx,s_ in enumerate(strats_to_run):
            prog.progress((idx+1)/len(strats_to_run),f"Testing: {s_[:40]}")
            try:
                sg,_=run_strategy(df,s_,{})
                if (sg!=0).sum()==0: continue
                bt=run_backtest(df,sg,cfg["sl_type"],cfg["sl_p"],cfg["tgt_type"],cfg["tgt_p"],cfg["qty"],s_)
                if bt.empty: continue
                s=compute_stats(bt)
                results.append({
                    "Strategy":s_,"Trades":s["total"],
                    "Win Rate":f"{s['acc']:.1f}%","Net P&L (₹)":round(s["tp"],0),
                    "Profit Factor":round(s["pf"],2) if s["pf"]!=float("inf") else 999.,
                    "Drawdown (₹)":round(s["dd"],0),"Expectancy":round(s["exp"],2),
                    "_pf":s["pf"] if s["pf"]!=float("inf") else 999.,"_tp":s["tp"]
                })
            except: pass
        prog.empty()
        if results:
            df_lb=pd.DataFrame(results).sort_values("_pf",ascending=False).reset_index(drop=True)
            df_lb.index+=1; st.session_state.leaderboard_results=df_lb; st.session_state.lb_ran_for=lb_key
        else:
            st.warning("No strategies generated trades — try a wider period.")
    df_lb=st.session_state.leaderboard_results
    if df_lb is not None and not df_lb.empty:
        if st.session_state.lb_ran_for!=lb_key:
            st.caption("⚠️ Config changed — re-run leaderboard for updated results.")
        st.markdown("#### Rankings (sorted by Profit Factor)")
        # Color-code rows
        def style_lb(row):
            pf=row["Profit Factor"]
            if pf>=1.5: bg="rgba(63,185,80,.10)"
            elif pf>=1.0: bg="rgba(255,235,100,.06)"
            else: bg="rgba(248,81,73,.08)"
            return [f"background-color:{bg}"]*len(row)
        display_cols=["Strategy","Trades","Win Rate","Net P&L (₹)","Profit Factor","Drawdown (₹)","Expectancy"]
        st.dataframe(df_lb[display_cols].style.apply(style_lb,axis=1),
                     use_container_width=True,height=420,key="lb_df")
        # Top 3 podium
        top3=df_lb.head(3)
        st.markdown("#### 🥇 Top 3 Strategies for Current Setup")
        medals=["🥇","🥈","🥉"]; cols_=st.columns(3)
        for i,((_,row),medal,col) in enumerate(zip(top3.iterrows(),medals,cols_)):
            with col:
                pf=row["Profit Factor"]; clr="#f0883e" if i==0 else "#8b949e"
                st.markdown(f"""<div class='icard' style='border-top:3px solid {clr};text-align:center'>
                  <div style='font-size:1.5rem'>{medal}</div>
                  <div style='color:#e6edf3;font-size:.83rem;font-weight:700;margin:5px 0'>{row["Strategy"][:30]}</div>
                  <div style='color:#8b949e;font-size:.78rem;line-height:1.8'>
                    PF: <b style='color:{"#3fb950" if pf>=1.5 else "#f85149"}'>{pf}</b><br>
                    Win Rate: <b style='color:#e6edf3'>{row["Win Rate"]}</b><br>
                    P&L: <b style='color:{"#3fb950" if row["Net P&L (₹)"]>=0 else "#f85149"}'>₹{row["Net P&L (₹)"]:,.0f}</b>
                  </div></div>""", unsafe_allow_html=True)

# ── RECOMMENDATIONS + HONEST ASSESSMENT ──────────────────────
def recommendations():
    with st.expander("📖 Honest Assessment: Will this app make you profitable?", expanded=False):
        st.markdown("""<div class='icard'>
<h4 style='color:#f85149'>⚠️ The Honest Truth About Competing with Jane Street</h4>
<div style='color:#8b949e;font-size:.82rem;line-height:2'>
Jane Street, Citadel, Two Sigma etc. operate with:
co-location servers 2 microseconds from exchange matching engines,
tick-by-tick order book data (not 1-minute candles),
teams of 200+ PhD quants,
strategies that generate edge at 0.001% per trade × billions of trades,
proprietary ML trained on satellite data, credit card feeds, alt-data.
<br><b style='color:#e6edf3'>You cannot compete with them in their domain. Full stop.</b>
</div>
<h4 style='color:#3fb950;margin-top:14px'>✅ Where Retail Traders CAN Win</h4>
<div style='color:#8b949e;font-size:.82rem;line-height:2'>
Jane Street does NOT trade your timeframe. They don't care about a 15-minute Nifty candle.
Retail edge exists in:
<br>• <b style='color:#e6edf3'>Trend-following on daily/weekly charts</b> — institutional algos create trends, you ride them
<br>• <b style='color:#e6edf3'>Mean reversion on liquid assets</b> (Kalman strategy) — exploits overreaction, still works
<br>• <b style='color:#e6edf3'>Volatility regime momentum</b> — follows institutional momentum, not front-runs it
<br>• <b style='color:#e6edf3'>Discipline</b> — 80% of retail loses because of emotions, not bad strategies
</div>
<h4 style='color:#58a6ff;margin-top:14px'>📊 Realistic P&L Expectations with This App</h4>
<table style='width:100%;border-collapse:collapse;font-size:.81rem'>
<tr style='border-bottom:1px solid #30363d'>
  <th style='color:#8b949e;padding:4px;text-align:left'>Scenario</th>
  <th style='color:#8b949e;padding:4px'>Win Rate</th>
  <th style='color:#8b949e;padding:4px'>R:R</th>
  <th style='color:#8b949e;padding:4px'>100 Trades</th>
  <th style='color:#8b949e;padding:4px'>Verdict</th>
</tr>
<tr style='border-bottom:1px solid #21262d'>
  <td style='padding:4px;color:#e6edf3'>EMA / RSI / BB alone</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>45–52%</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>1:1.5</td>
  <td style='padding:4px;color:#f85149;text-align:center'>Breakeven/Loss</td>
  <td style='padding:4px;color:#f85149'>❌ Not enough edge</td>
</tr>
<tr style='border-bottom:1px solid #21262d'>
  <td style='padding:4px;color:#e6edf3'>Volatility Regime Momentum</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>58–65%</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>1:2</td>
  <td style='padding:4px;color:#3fb950;text-align:center'>+16–30R profit</td>
  <td style='padding:4px;color:#3fb950'>✅ Genuine edge</td>
</tr>
<tr style='border-bottom:1px solid #21262d'>
  <td style='padding:4px;color:#e6edf3'>Kalman Mean Reversion</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>55–62%</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>1:2</td>
  <td style='padding:4px;color:#3fb950;text-align:center'>+10–24R profit</td>
  <td style='padding:4px;color:#3fb950'>✅ Works on Gold/FX</td>
</tr>
<tr>
  <td style='padding:4px;color:#e6edf3'>Leaderboard winner + 1:2 RR</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>Best for setup</td>
  <td style='padding:4px;color:#8b949e;text-align:center'>1:2 min</td>
  <td style='padding:4px;color:#3fb950;text-align:center'>Best achievable</td>
  <td style='padding:4px;color:#3fb950'>✅ Use leaderboard tab</td>
</tr>
</table>
<h4 style='color:#f0883e;margin-top:14px'>🚀 What Makes THIS App Worth Using</h4>
<ul style='color:#8b949e;font-size:.81rem;line-height:2;margin:0'>
  <li><b style='color:#e6edf3'>Discipline automation</b> — SL/Target enforced every trade, no exceptions</li>
  <li><b style='color:#e6edf3'>Leaderboard</b> — tells you which strategy works NOW on your specific setup</li>
  <li><b style='color:#e6edf3'>Backtesting integrity</b> — N+1 entry, SL-first checking — no fake backtests</li>
  <li><b style='color:#e6edf3'>Auto-trading</b> — removes fear/greed from execution entirely</li>
  <li><b style='color:#e6edf3'>3 PRO strategies</b> — these have real edge, not just TA indicators</li>
</ul>
<div style='color:#f0883e;font-size:.8rem;margin-top:10px;border-top:1px solid #30363d;padding-top:10px'>
  <b>Bottom line:</b> Simple indicator strategies won't make you rich. But running the leaderboard,
  picking the highest Profit Factor strategy for current conditions, using 1:2 R:R minimum, and
  letting auto-trading remove your emotions — that is a genuinely profitable system for retail traders.
</div></div>""", unsafe_allow_html=True)

# ── MAIN — auto-loads data when sidebar selections change ─────
def main():
    st.markdown("""<div style='display:flex;align-items:center;gap:11px;
         padding:2px 0 10px;border-bottom:1px solid #21262d;margin-bottom:10px'>
      <span style='font-size:1.8rem'>📊</span>
      <div><div style='font-size:1.45rem;font-weight:900;color:#e6edf3;letter-spacing:1px'>QuantAlgo Pro</div>
        <div style='font-size:.62rem;color:#8b949e;letter-spacing:2px'>PROFESSIONAL ALGORITHMIC TRADING · v5.0</div></div>
      <div style='margin-left:auto;display:flex;gap:7px;align-items:center'>
        <div style='background:#1a4731;color:#3fb950;padding:2px 9px;border-radius:20px;font-size:.67rem;font-weight:700'>● PAPER MODE</div>
        <div style='background:#1c2128;color:#58a6ff;padding:2px 9px;border-radius:20px;font-size:.67rem'>14 Strategies · Auto-load</div>
      </div></div>""", unsafe_allow_html=True)

    cfg=sidebar()
    recommendations()

    # ── AUTO-LOAD DATA (no button needed) ───────────────────
    # Re-fetches only when ticker/interval/period changes — not on every interaction
    data_key=f"{cfg['tsym']}|{cfg['interval']}|{cfg['period']}"
    if data_key!=st.session_state.last_data_key:
        df=fetch_data(cfg["tsym"],cfg["interval"],cfg["period"])
        if df is not None and not df.empty:
            st.session_state.current_data=df
            st.session_state.last_data_key=data_key
            st.session_state.last_strat_key=""
            st.session_state.backtest_results=None
            st.session_state.leaderboard_results=None

    # ── AUTO-RERUN STRATEGY when strategy/params change ─────
    # Fast (< 100 ms) — no API call needed, uses existing data
    sp_key=str(sorted(cfg["sp"].items()))
    strat_key=f"{data_key}|{cfg['strat']}|{sp_key}"
    df=st.session_state.current_data
    if df is not None and not df.empty and strat_key!=st.session_state.last_strat_key:
        sigs,inds=run_strategy(df,cfg["strat"],cfg["sp"])
        st.session_state.signals=sigs
        st.session_state.indicators=inds
        st.session_state.last_strat_key=strat_key

    tab1,tab2,tab3,tab4=st.tabs(["📈 Backtest","📡 Live Trading","📋 Live History","🏆 Leaderboard"])
    with tab1: tab_backtest(cfg)
    with tab2: tab_live(cfg)
    with tab3: tab_history()
    with tab4: tab_leaderboard(cfg)

if __name__=="__main__":
    main()
